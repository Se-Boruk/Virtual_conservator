###################################################################
# ( 0 ) Libs and dependencies 
###################################################################

import sys
import os
import json
import joblib
from tqdm import tqdm
import torch
import torch.nn.functional as F
import comet_ml

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from Config import DATABASE_FOLDER, RECONSTRUCTION_DATASET_PATH, RECONSTRUCTION_DATASET_NAME
from Config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_STATE
import Utilities_lib as Ut_lib

sys.path.insert(0, DATABASE_FOLDER)
from DataBase_Functions import Custom_DataSet_Manager, Reconstruction_data_tests, Async_DataLoader, Random_Damage_Generator, augment_image_and_mask
import Architectures
import Inpainter_functions as Inp_f

# Metric libs
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

###################################################################
# ( 1 ) Hardware setup
###################################################################

print("\nSearching for cuda device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###################################################################
# ( 2 ) Loading data
###################################################################

manager = Custom_DataSet_Manager(DataSet_path = RECONSTRUCTION_DATASET_PATH,
                                 train_split = TRAIN_SPLIT,
                                 val_split = VAL_SPLIT,
                                 test_split = TEST_SPLIT,
                                 random_state = RANDOM_STATE)

manager.download_database(RECONSTRUCTION_DATASET_NAME)
Train_set, Val_set, Test_set = manager.load_dataset_from_disk()

# Run dataset tests
Reconstruction_data_tests(train_subset = Train_set, val_subset = Val_set, test_subset = Test_set)

###################################################################
# ( 3 ) Setting parameters
###################################################################

bs = 16
n_workers = 4
max_queue = 10
input_channels = 3  
n_residual_blocks = 6
base_filters = 32
Visualization_rows = 9

baseline_model_path = "models/V5_BASELINE_Fresh_decoder_January" 

# Loss Weights from Inpainter_functions.py
L1_W, L2_W, SSIM_W, TV_W = 1.0, 0.1, 0.05, 0.001

###################################################################
# ( 4 ) Model creation, weight loading and Clusterizer setup
###################################################################

mapping_file = "class_map.json"
shared_mapping = Ut_lib.build_class_mapping(Train_set, Val_set, Test_set, mapping_file=mapping_file, style_field="style")

def load_model_pair(path, device):
    enc = Architectures.Inpainter_V5.Encoder(input_channels=input_channels, n_residual=n_residual_blocks, base_filters=base_filters)
    dec = Architectures.Inpainter_V5.Decoder(output_channels=input_channels, base_filters=base_filters)
    ckpt = torch.load(path, map_location='cpu')
    enc.load_state_dict(ckpt['encoder_state_dict'])
    dec.load_state_dict(ckpt['decoder_state_dict'])
    return enc.to(device).eval(), dec.to(device).eval()

print("\nLoading weights...")
models_to_test = {
    "baseline": load_model_pair("models/V5_BASELINE_Fresh_decoder_January/best_inpainter.pth", device),
    "real":     load_model_pair("models/V5_REAL_January/best_inpainter.pth", device),
    "maxclass": load_model_pair("models/V5_ARTIFICIAL_January/best_inpainter.pth", device) # Using REAL weights for MaxClass comparison
}

# --- Load Clusterizer for the REAL model ---
print("Loading PCA and K-Means for unsupervised labeling...")
pca_loaded = joblib.load(os.path.join(baseline_model_path , "pca_model.joblib"))
loaded_kmeans = joblib.load(os.path.join(baseline_model_path , 'minibatch_kmeans_model.joblib'))

pca_components = torch.from_numpy(pca_loaded.components_).to(device).T.float()
pca_mean = torch.from_numpy(pca_loaded.mean_).to(device).float()
kmeans_centroids = torch.from_numpy(loaded_kmeans.cluster_centers_).to(device).float()
global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

test_loader = Async_DataLoader(dataset=Test_set, batch_size=bs, num_workers=n_workers, device=device,
                               max_queue=max_queue, add_damaged=True, label_map=shared_mapping)

test_loader.start_epoch(shuffle=False)

###################################################################
# ( 5 ) Quantitative Evaluation
###################################################################

lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    return (20 * torch.log10(2.0 / torch.sqrt(mse))).item() if mse > 0 else 100.0

# Added "damaged_only" to results to see the raw error before repair
model_names = list(models_to_test.keys()) + ["damaged_only"]
export_data = {
    "results": {k: {"loss": [], "psnr": [], "lpips": []} for k in model_names},
    "averages": {}
}

num_batches = test_loader.get_num_batches()

print("\nRunning quantitative evaluation...")
with torch.no_grad():
    for _ in tqdm(range(num_batches)):
        batch_data = test_loader.get_batch()
        if batch_data is None: break
        
        damaged = (batch_data["original_damaged"] * 2) - 1 # Normalize to [-1, 1]
        original = (batch_data["original"] * 2) - 1 # Normalize to [-1, 1]
        gt_labels = batch_data["labels"]
        
        # --- 1. Evaluate "Prediction vs Original" for each model ---
        for name, (enc, dec) in models_to_test.items():
            latent, skips = enc(damaged)
            s0, s1, s2 = skips
            
            # Label Selection Logic
            if name == "baseline":
                class_vec = torch.zeros(damaged.shape[0], 1, device=device)
            elif name == "real":
                z = global_avg_pool(latent).view(latent.size(0), -1)
                z_pca = torch.mm(z - pca_mean, pca_components)
                dist_matrix = torch.cdist(z_pca, kmeans_centroids)
                class_vec = torch.argmin(dist_matrix, dim=1).unsqueeze(1).to(z.dtype)
            else: # MaxClass logic with REAL weights
                class_vec = gt_labels 
            
            pred = dec(latent, s0, s1, s2, class_vec)
            
            # Metrics calculation
            l = Inp_f.inpainting_loss(pred, original, L1_W, L2_W, SSIM_W, TV_W).item()
            p = calculate_psnr(pred, original)
            lp = lpips_fn((pred + 1) / 2, (original + 1) / 2).item()
            
            export_data["results"][name]["loss"].append(l)
            export_data["results"][name]["psnr"].append(p)
            export_data["results"][name]["lpips"].append(lp)

        # --- 2. Evaluate "Damaged vs Original" (Raw Damage Stats) ---
        l_dmg = Inp_f.inpainting_loss(damaged, original, L1_W, L2_W, SSIM_W, TV_W).item()
        p_dmg = calculate_psnr(damaged, original)
        lp_dmg = lpips_fn((damaged + 1) / 2, (original + 1) / 2).item()

        export_data["results"]["damaged_only"]["loss"].append(l_dmg)
        export_data["results"]["damaged_only"]["psnr"].append(p_dmg)
        export_data["results"]["damaged_only"]["lpips"].append(lp_dmg)

# Aggregation and Export
for name in model_names:
    count = len(export_data["results"][name]["loss"])
    export_data["averages"][name] = {
        m: sum(export_data["results"][name][m]) / count for m in ["loss", "psnr", "lpips"]
    }

with open(output_path := "model_evaluation_scores.json", 'w') as f:
    json.dump(export_data, f, indent=4)

print(f"\nScores saved to {output_path}")

print("\n" + "="*75)
print(f"{'Model':<20} | {'Avg Loss':<10} | {'PSNR (dB)':<10} | {'LPIPS':<10}")
print("-" * 75)
# Sorting so damaged_only is at the top or bottom for contrast
sorted_names = ["damaged_only"] + list(models_to_test.keys())
for name in sorted_names:
    avg = export_data["averages"][name]
    print(f"{name:<20} | {avg['loss']:<10.4f} | {avg['psnr']:<10.2f} | {avg['lpips']:<10.4f}")
print("="*75)

###################################################################
# ( 6 ) Visual Comparison
###################################################################

damage_generator = Random_Damage_Generator(device=device)
restored_paintings_path = os.path.join("Restored_paintings", "Tests")
os.makedirs(restored_paintings_path, exist_ok=True)

visual_batch = test_loader.get_random_batch(batch_size=Visualization_rows, shuffle=True, random_state=RANDOM_STATE)
visual_batch = (visual_batch * 2) - 1

for i, (name, (enc, dec)) in enumerate(models_to_test.items()):
    viz = Inp_f.InpaintingVisualizer(enc, dec, damage_generator, rows=Visualization_rows, device=device, save_dir=restored_paintings_path)
    if i == 0: shared_masks = viz.masks
    else: viz.masks = shared_masks
    
    viz.visualize(visual_batch, epoch=i, prefix=f"test_{name}")

print("Visualizations complete.")