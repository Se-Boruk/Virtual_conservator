import torchvision.transforms as T
import torch
import joblib
import os
import InPainter.Architectures as Architectures

import random
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

# Loss Weights from Inpainter_functions.py
L1_W, L2_W, SSIM_W, TV_W = 1.0, 0.1, 0.05, 0.001

# --- Load Clusterizer for the REAL model ---
print("Loading PCA and K-Means for unsupervised labeling...")
pca_loaded = joblib.load("InPainter/models/pca_model.joblib")
loaded_kmeans = joblib.load('InPainter/models/minibatch_kmeans_model.joblib')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

pca_components = torch.from_numpy(pca_loaded.components_).to(device).T.float()
pca_mean = torch.from_numpy(pca_loaded.mean_).to(device).float()
kmeans_centroids = torch.from_numpy(loaded_kmeans.cluster_centers_).to(device).float()
global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))


def load_model_pair(path, device):
    enc = Architectures.Inpainter_V5.Encoder(input_channels=input_channels, n_residual=n_residual_blocks, base_filters=base_filters)
    dec = Architectures.Inpainter_V5.Decoder(output_channels=input_channels, base_filters=base_filters)
    ckpt = torch.load(path, map_location='cpu')
    enc.load_state_dict(ckpt['encoder_state_dict'])
    dec.load_state_dict(ckpt['decoder_state_dict'])
    return enc.to(device).eval(), dec.to(device).eval()


def InPainteR_GUI(input_tensor, output_name = 'FixedImage.png'):
    global device, pca_loaded, loaded_kmeans

    input_tensor = (input_tensor * 2) - 1
    pca_components = torch.from_numpy(pca_loaded.components_).to(device).T.float()
    pca_mean = torch.from_numpy(pca_loaded.mean_).to(device).float()
    kmeans_centroids = torch.from_numpy(loaded_kmeans.cluster_centers_).to(device).float()
    global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    output_name = "GUI/" + output_name
    encoder, decoder = load_model_pair("InPainter/models/best_inpainter.pth", device)

    with torch.no_grad():
        # Enkoder: Obraz -> Latent + Skips
        latent, skips = encoder(input_tensor)
        s0, s1, s2 = skips
        # 1. Global Average Pooling
        z = global_avg_pool(latent).view(latent.size(0), -1)

        # 2. Projekcja PCA
        z_pca = torch.mm(z - pca_mean, pca_components)

        # 3. Szukanie najbliższego centroidu (K-Means)
        dist_matrix = torch.cdist(z_pca, kmeans_centroids)
        class_vec = torch.argmin(dist_matrix, dim=1)
        class_no = class_vec.item()
        class_vec = class_vec.unsqueeze(1).to(z.dtype)


        # Dekoder: Latent -> Obraz
        reconstructed = decoder(latent, s0, s1, s2, class_vec)
            
        # 4. Post-processing i zapis
        # Odwrócenie normalizacji: (-1, 1) -> (0, 1)
        reconstructed = (reconstructed + 1) / 2
        reconstructed = torch.clamp(reconstructed, 0, 1)
            
        save_transform = T.ToPILImage()
        result_img = save_transform(reconstructed.squeeze(0).cpu())
        result_img.save(output_name) 

    return output_name, random.randint(0,21)