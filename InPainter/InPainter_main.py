###################################################################
# ( 0 ) Libs and dependencies 
###################################################################

#Basic libs
import sys
import os
from tqdm import tqdm
import csv
import joblib
#Go up directly before we can take the Project ROOT from the Config
#Get the parent folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Add parent folder to Python path
sys.path.insert(0, parent_dir)

from Config import DATABASE_FOLDER, RECONSTRUCTION_DATASET_PATH, RECONSTRUCTION_DATASET_NAME
from Config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_STATE
import Utilities_lib as Ut_lib


##########################################################################################
#Move folder up to go to database folder to use manager from here
sys.path.insert(0, DATABASE_FOLDER)
from DataBase_Functions import Custom_DataSet_Manager
from DataBase_Functions import Reconstruction_data_tests
from DataBase_Functions import Async_DataLoader
from DataBase_Functions import Random_Damage_Generator

import Architectures
import Inpainter_functions as Inp_f

#Other libs
import comet_ml
import torch
import torch.nn.functional as F


###################################################################
# ( 1 ) Hardware setup
###################################################################

print("\nSearching for cuda device...")
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Print available GPUs
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


###################################################################
# ( 2 ) Loading data
###################################################################

#Data loading (Not stored in RAM but is accessed on demand)

#Load manager and execute
manager = Custom_DataSet_Manager(DataSet_path = RECONSTRUCTION_DATASET_PATH,
                                 train_split = TRAIN_SPLIT,
                                 val_split = VAL_SPLIT,
                                 test_split = TEST_SPLIT,
                                 random_state = RANDOM_STATE
                                 )

#Download data if not present
manager.download_database(RECONSTRUCTION_DATASET_NAME)
#Load dataset
Train_set, Val_set, Test_set = manager.load_dataset_from_disk()


#Run tests to see if operating on data which is the same for all:
Reconstruction_data_tests(train_subset = Train_set,
                          val_subset = Val_set,
                          test_subset = Test_set
                          )

###################################################################
# ( 3 ) Setting parameters
###################################################################

#Hyperparams for training
model_ID = "V5_BASELINE_Fresh_decoder_January"
baseline_model_path = "V5_Baseline_January"
training_mode = "BASELINE"      #Training with no info about classes  (How model behaves without class info)
#training_mode = "REAL"         #Training with info about classes from clusterizator 
#training_mode = "ARTIFICIAL"   #Training with infor about classes from dataset (theoretical performance limit)



epochs = 50
bs = 6
lr = 1e-4
patience = 5
train_set_fraction = 0.4

l1_weight = 1
l2_weight = 0.1
ssim_weight = 0.05
tv_weight=  0.001

#Cluster based weights
latent_weight = 0.1           #weight of this loss compared to the others

#Datalodaers params
n_workers = 4
max_queue = 10

#Inpainter model params
input_channels = 3  #Same as data channels if the input is rgb only (harder problem)
                    #If we can apply also the mask then it will be 4
                    
n_residual_blocks = 6
base_filters = 32


Visualization_rows = 9

###################################################################
# ( 4 ) Model creation, dataloader preparation
###################################################################
print("Mapping the pseudo-classes. JUST FOR the limit testing. Original classes would come in the final model from clusterizer!")

mapping_file = "class_map.json"

shared_mapping = Ut_lib.build_class_mapping(Train_set, Val_set, Test_set,
                                            mapping_file=mapping_file,
                                            style_field="style"
                                            )



# Training loader
train_loader = Async_DataLoader(dataset = Train_set,
                                batch_size=bs,
                                num_workers=n_workers,
                                device='cuda',
                                max_queue=max_queue,
                                add_damaged = True,
                                add_augmented = True,
                                label_map = shared_mapping,
                                fraction = train_set_fraction
                                )

# Validation loader
val_loader = Async_DataLoader(dataset = Val_set,
                              batch_size=bs,
                              num_workers=n_workers,
                              device='cuda',
                              max_queue=max_queue,
                              add_damaged = True,
                              add_augmented = True,
                              label_map = shared_mapping,
                              fraction = None
                              )

# Test loader
test_loader = Async_DataLoader(dataset = Test_set,
                              batch_size=bs,
                              num_workers=n_workers,
                              device='cuda',
                              max_queue=max_queue,
                              add_damaged = True,
                              add_augmented = True,
                              label_map = shared_mapping,
                              fraction = None
                              )





#Taking shape of the data
img_channels = train_loader.C
img_h = train_loader.H
img_w = train_loader.W


#Creating Inpainter model
print("\nPreparing Inpainter Encoder...")
Inpainter_encoder = Architectures.Inpainter_V5.Encoder(input_channels = input_channels,
                                                     n_residual=n_residual_blocks,
                                                     base_filters=base_filters
                                                     ).to('cpu')

print("\nPreparing Inpainter Decoder...")
Inpainter_decoder = Architectures.Inpainter_V5.Decoder(output_channels = input_channels,
                                                       base_filters=base_filters
                                                       ).to('cpu')


##########################
#if training_mode != "BASELINE":    #Original
if training_mode == "BASELINE":
    #Loading the trained encoder weights:
        
    pretrained_autoencoder_path = os.path.join("models", baseline_model_path, "best_inpainter.pth")
    checkpoint = torch.load(pretrained_autoencoder_path, map_location='cpu')
    Inpainter_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    #Freeze the weights
    for param in Inpainter_encoder.parameters():
        param.requires_grad = False
    print("Encoder frozen; only decoder will be trained")

#########################

#Optimizer
#
if training_mode == "BASELINE":
    """
    #Original
    Opt_inpainter = torch.optim.AdamW( list(Inpainter_encoder.parameters()) + list(Inpainter_decoder.parameters()),
                                      lr=lr,
                                      betas=(0.9, 0.999),
                                      weight_decay=1e-6 
                                      )
    """
    #New optimizer just for the decoder
    Opt_inpainter = torch.optim.AdamW(
        list(Inpainter_decoder.parameters()), 
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-6
    )
else:
    #New optimizer just for the decoder
    Opt_inpainter = torch.optim.AdamW(
        list(Inpainter_decoder.parameters()), 
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-6
    )


###################################################################
# ( 5 ) Showing / saving architecture scheme [ Optional ]
###################################################################

source_shape = (input_channels, img_h, img_w)

dummy_encoder_input = torch.zeros(source_shape)

Ut_lib.Show_encoder_summary(
    encoder=Inpainter_encoder,
    input_tensor=dummy_encoder_input,
    save_png=True,
    folder="Model_graphs",
    name="Inpainter_Encoder"
)

Ut_lib.Show_decoder_summary(
    decoder=Inpainter_decoder,
    encoder=Inpainter_encoder,
    input_tensor=dummy_encoder_input,
    class_vector_size=1,
    device="cpu",
    save_png=True,
    folder="Model_graphs",
    name="Inpainter_Decoder"
)






###################################################################
# ( 6 ) Final model moving just in case
###################################################################
print("\nMoving models to GPU...")
Inpainter_encoder = Inpainter_encoder.to(device)
Inpainter_decoder = Inpainter_decoder.to(device)
print("Done!")





###################################################################
# ( 7 ) Preparation for saving model results in form of plots and logs
###################################################################
# For plots
####
# Initialize once before training
damage_generator = Random_Damage_Generator()

restored_paintings_path = os.path.join("Restored_paintings", model_ID)
visualizer = Inp_f.InpaintingVisualizer(Inpainter_encoder, Inpainter_decoder, damage_generator, rows=Visualization_rows, H=img_h, W=img_w, device=device,
                                      save_dir=restored_paintings_path, save=True)



#Random sample from the test set so its not affecting training in any way
visual_batch = test_loader.get_random_batch(batch_size = Visualization_rows, shuffle = True, random_state = RANDOM_STATE)
visual_batch = (visual_batch *2)-1

#For logs and models
###
os.makedirs(os.path.join("models", model_ID), exist_ok=True)
os.makedirs(os.path.join("models", model_ID, "backup"), exist_ok=True)
backup_interval = 5  # save backup every N epochs
log_csv_path = os.path.join("models", model_ID, "train_loss_log.csv")

#Remove old file
if os.path.exists(log_csv_path):
    os.remove(log_csv_path)

# create fresh CSV with header
with open(log_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss"])
        
        
best_val_loss = float("inf")
best_model_path = os.path.join("models", model_ID, "best_inpainter.pth")



###################################################################
# ( 8 ) Preparing comet logging
###################################################################

#api key from env variables on local machine
api_key = os.getenv("COMET_API_KEY")
    
# Initialize experiment
comet_experiment = comet_ml.start(
    api_key=api_key,
    project_name="SIUM_2_Unsupervised_learning_project",
    workspace="sium-2-team",
    online=False
)

comet_experiment.add_tags(["Inpainter", model_ID])


# Hyperparameters
comet_experiment.log_parameters({
    "train_set_fraction" : train_set_fraction,
    "epochs": epochs,
    "batch_size": bs,
    "patience": patience,
    "learning_rate": lr,
    "optimizer": type(Opt_inpainter).__name__,  # get class name of optimizer
    "loss_weights": {
        "l1": l1_weight,
        "l2": l2_weight,
        "ssim": ssim_weight,
        "tv": tv_weight,
        "latent_sim": latent_weight
    }
})

# Dataloader parameters
comet_experiment.log_parameters({
    "num_workers": n_workers,
    "max_queue_size": max_queue
})

# Inpainter model architecture
comet_experiment.log_parameters({
    "encoder_class": type(Inpainter_encoder).__name__,
    "decoder_class": type(Inpainter_decoder).__name__,
    "num_residual_blocks": n_residual_blocks,
    "base_filters": base_filters,
    "encoder_params": sum(p.numel() for p in Inpainter_encoder.parameters() if p.requires_grad),
    "decoder_params": sum(p.numel() for p in Inpainter_decoder.parameters() if p.requires_grad)
})


# Log model graph
#comet_experiment.set_model_graph(Inpainter_model)

###################################################################
# ( 8.1 ) Preparing clusterizer
###################################################################
if training_mode == "REAL":
    pca_loaded = joblib.load(os.path.join("models", baseline_model_path , "pca_model.joblib"))
    loaded_kmeans = joblib.load(os.path.join("models", baseline_model_path , 'minibatch_kmeans_model.joblib'))

    # PCA Parameters
    # We transpose (.T) because sklearn stores them as (n_components, n_features)
    # Mathematically, we want: [batch, n_features] @ [n_features, n_components]
    pca_components = torch.from_numpy(pca_loaded.components_).to(device).T.float()
    pca_mean = torch.from_numpy(pca_loaded.mean_).to(device).float()
    
    # K-Means Parameters
    # These represent the 'prototypes' in the reduced PCA space
    kmeans_centroids = torch.from_numpy(loaded_kmeans.cluster_centers_).to(device).float()
###################################################################
# ( 9 ) Training loop
###################################################################

#Scaler for halfprecision training
scaler = torch.amp.GradScaler()
global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
#Fallback for comet experiment to log the results even if training crashes
try:
    # initialize early-stopping counters if not already
    if 'best_val_loss' not in globals():
        best_val_loss = float("inf")
    epochs_no_improve = 0

    for e in range(epochs):
        #################################################
        #Training part
        #################################################
        
        train_loader.start_epoch(shuffle=True)
        
        epoch_loss = 0.0
        num_batches = train_loader.get_num_batches()
    
        Inpainter_encoder.train()
        with tqdm(total=num_batches, desc=f"Epoch {e+1}", unit=" batch") as pbar:
            while True:
                
                #Load batch from loader
                batch = train_loader.get_batch()
                if batch is None:
                    break
                
                #Unload_batch
                original_batch = batch['original']
                damaged_batch = batch['original_damaged']
                
                aug_batch = batch['augmented']
                aug_damaged_batch = batch['augmented_damaged']
                
                artificial_labels = batch['labels']

                
                #Load batches and normalize them to -1 1 range
                original_batch = (original_batch * 2)-1
                damaged_batch = (damaged_batch * 2)-1
                
                aug_batch = (aug_batch * 2)-1
                aug_damaged_batch = (aug_damaged_batch * 2)-1
                
                ##############################
                #Simulate training 
                
                
                # ----Inpainter update----
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    
                    #Make restored batch
                    latent_tensor, skips = Inpainter_encoder(damaged_batch)
                    s0, s1, s2 = skips
                    
                    #Make restored batch aug
                    latent_tensor_aug, skips_aug = Inpainter_encoder(aug_damaged_batch)
                    s0_a, s1_a, s2_a = skips_aug
                    
                    #==============================================================
                    #Palce for the clusterization from the latent tensor.
                    #Now its just filled with 0s for the baseline (if baseline is trained)
                    if training_mode == "BASELINE":
                        bs = damaged_batch.shape[0]
                        artificial_labels = torch.zeros(bs,1, device=damaged_batch.device)
                        artificial_labels_aug = artificial_labels
                        
                    elif training_mode == "REAL":
                        # 1. Global average pooling to get feature vectors
                        z0 = global_avg_pool(latent_tensor).view(latent_tensor.size(0), -1)
                        z1 = global_avg_pool(latent_tensor_aug).view(latent_tensor_aug.size(0), -1)
                    
                        # 2. Project into PCA Space: (X - mu) @ V
                        z0_pca = torch.mm(z0 - pca_mean, pca_components)
                        z1_pca = torch.mm(z1 - pca_mean, pca_components)
                    
                        # 3. Assign labels based on proximity to Centroids
                        dist_matrix0 = torch.cdist(z0_pca, kmeans_centroids)
                        dist_matrix1 = torch.cdist(z1_pca, kmeans_centroids)
                    
                        # 4. Predict cluster indices (Hard Assignment) + Match Dtype
                        # .to(z0.dtype) ensures compatibility with autocast (float16/float32)
                        artificial_labels = torch.argmin(dist_matrix0, dim=1).unsqueeze(1).to(z0.dtype)
                        artificial_labels_aug = torch.argmin(dist_matrix1, dim=1).unsqueeze(1).to(z1.dtype)
                       
                        
                    elif training_mode == "ARTIFICIAL":
                        artificial_labels_aug = artificial_labels

                    else:
                        artificial_labels_aug = artificial_labels #Same as artificial
                    #==============================================================


                    #Now we use artificial labels from the dataset (assuming we got 100% accuracy check if we can make improvement)
                    restored_batch = Inpainter_decoder(latent_tensor, s0, s1, s2, artificial_labels)
                    restored_aug_batch = Inpainter_decoder(latent_tensor_aug, s0_a, s1_a, s2_a, artificial_labels_aug)
                    

                    #Loss function calculation (inpainting)
                    Loss_origin = Inp_f.inpainting_loss(pred = restored_batch,
                                                        target = original_batch,
                                                        l1_weight = l1_weight,
                                                        l2_weight = l2_weight,
                                                        ssim_weight = ssim_weight,
                                                        tv_weight = tv_weight
                                                        )
                    
                    Loss_aug = Inp_f.inpainting_loss(pred = restored_aug_batch,
                                                     target = aug_batch,
                                                     l1_weight = l1_weight,
                                                     l2_weight = l2_weight,
                                                     ssim_weight = ssim_weight,
                                                     tv_weight = tv_weight
                                                     )
                    
                    
                    #Loss function calculation (latent similarity)
                    ######################################### 

                    
                    #Take part of vector so only this part is forced to be similar (other part can encode repair and mask more easily)
                    z0 = global_avg_pool(latent_tensor).view(latent_tensor.size(0), -1)
                    z1 = global_avg_pool(latent_tensor_aug).view(latent_tensor.size(0), -1)
                    
                    # Normalize
                    z0 = F.normalize(z0, dim=1)
                    z1 = F.normalize(z1, dim=1)
                    
                    # Per-sample cosine similarity (so we do not mix the losses to not mix uinrelated samples)
                    loss_cos_per_sample = 1 - F.cosine_similarity(z0, z1, dim=1)
                    
                    #Mean
                    loss_cos = loss_cos_per_sample.mean()
                    #########################################
                    
                    #Calculate the final loss
                    Loss = (Loss_origin + Loss_aug) / 2 + loss_cos * latent_weight
                    
                    
                #Gradients update with exception for explosion (addon for adding I suppose, model would collapse anyway though)
                if torch.isfinite(Loss):
                    Opt_inpainter.zero_grad()
                    scaler.scale(Loss).backward()
                    scaler.step(Opt_inpainter)
                    scaler.update()
                else:
                    print(f"[Warning] Skipped Generator step due to non-finite loss: {Loss.item()}")
                ##############################
                
                
                
                epoch_loss += Loss.item()
                #Finished training step, log scores and calculate avg times
                pbar.update(1)
                pbar.set_postfix( { "train_loss": f"{Loss.item():.4f}" } )
                
                
                
        #################################################
        #Validation part
        #################################################
        val_loader.start_epoch(shuffle=False) # No need for shuffle generally in validation
        val_num_batches = val_loader.get_num_batches()
        batch_val_times = []
        
        Inpainter_encoder.eval()
        Inpainter_decoder.eval()
        
        val_epoch_loss = 0.0
        with tqdm(total=val_num_batches, desc=f"Epoch {e+1} - Val", unit="batch") as pbar:
            while True:

                #Load batch from loader
                batch = val_loader.get_batch()
                if batch is None:
                    break
                
                #Unload batch
                original_batch_v = batch['original']
                damaged_batch_v = batch['original_damaged']
                
                aug_batch_v = batch['augmented']
                aug_damaged_batch_v = batch['augmented_damaged']
                
                artificial_labels_v = batch['labels']

                #Normalize
                original_batch_v = (original_batch_v * 2) - 1
                damaged_batch_v = (damaged_batch_v * 2) - 1
                aug_batch_v = (aug_batch_v * 2) - 1
                aug_damaged_batch_v = (aug_damaged_batch_v * 2) - 1

                ##############################
                # simulate validation forward (same ops as train but no optimizer step)
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        # Encode original damaged batch
                        latent_tensor_v, skips_v = Inpainter_encoder(damaged_batch_v)
                        vs0, vs1, vs2 = skips_v

                        # Encode augmented damaged batch
                        latent_tensor_aug_v, skips_aug_v = Inpainter_encoder(aug_damaged_batch_v)
                        vs0_a, vs1_a, vs2_a = skips_aug_v
                    

                        #==============================================================
                        #Palce for the clusterization from the latent tensor.
                        #Now its just filled with 0s for the baseline (if baseline is trained)
                        if training_mode == "BASELINE":
                            bs = damaged_batch_v.shape[0]
                            artificial_labels_v = torch.zeros(bs,1, device=damaged_batch_v.device)
                            artificial_labels_v_aug= artificial_labels_v
                            
                        elif training_mode == "REAL":
                            # 1. Latent Feature Extraction
                            z0_v = global_avg_pool(latent_tensor_v).view(latent_tensor_v.size(0), -1)
                            z1_v = global_avg_pool(latent_tensor_aug_v).view(latent_tensor_aug_v.size(0), -1)
                            
                            # 2. GPU PCA Projection
                            # Using the same pre-loaded pca_mean and pca_components
                            z0_v_pca = torch.mm(z0_v - pca_mean, pca_components)
                            z1_v_pca = torch.mm(z1_v - pca_mean, pca_components)
                            
                            # 3. GPU K-Means Prediction
                            # Compute distances to pre-loaded kmeans_centroids
                            dist_matrix_v0 = torch.cdist(z0_v_pca, kmeans_centroids)
                            dist_matrix_v1 = torch.cdist(z1_v_pca, kmeans_centroids)
                            
                            # 4. Hard assignment of labels + Correct Dtype Casting
                            # We cast to .to(z0_v.dtype) to match the decoder's expected input (float16/32)
                            artificial_labels_v = torch.argmin(dist_matrix_v0, dim=1).unsqueeze(1).to(z0_v.dtype)
                            artificial_labels_v_aug = torch.argmin(dist_matrix_v1, dim=1).unsqueeze(1).to(z1_v.dtype)

                            
                        else:
                            artificial_labels_v_aug = artificial_labels_v #Same as artificial
                        #==============================================================
                        

                        # Decode
                        restored_batch_v = Inpainter_decoder(latent_tensor_v, vs0, vs1, vs2, artificial_labels_v)
                        restored_aug_batch_v = Inpainter_decoder(latent_tensor_aug_v, vs0_a, vs1_a, vs2_a, artificial_labels_v_aug)

                        # Inpainting loss
                        Loss_origin_v = Inp_f.inpainting_loss(
                            pred=restored_batch_v,
                            target=original_batch_v,
                            l1_weight=l1_weight,
                            l2_weight=l2_weight,
                            ssim_weight=ssim_weight,
                            tv_weight=tv_weight
                        )

                        Loss_aug_v = Inp_f.inpainting_loss(
                            pred=restored_aug_batch_v,
                            target=aug_batch_v,
                            l1_weight=l1_weight,
                            l2_weight=l2_weight,
                            ssim_weight=ssim_weight,
                            tv_weight=tv_weight
                        )

                        # Latent similarity
                        z0_v = global_avg_pool(latent_tensor_v).view(latent_tensor_v.size(0), -1)
                        z1_v = global_avg_pool(latent_tensor_aug_v).view(latent_tensor_aug_v.size(0), -1)

                        # Normalize
                        z0_v = F.normalize(z0_v, dim=1)
                        z1_v = F.normalize(z1_v, dim=1)

                        # Per-sample cosine similarity
                        loss_cos_per_sample_v = 1 - F.cosine_similarity(z0_v, z1_v, dim=1)

                        # Mean latent similarity loss
                        loss_cos_v = loss_cos_per_sample_v.mean()

                        # Final validation loss (same weighting as train)
                        Loss_v = (Loss_origin_v + Loss_aug_v) / 2 + loss_cos_v * latent_weight

                # track val loss (sum of per-batch loss items)
                val_epoch_loss += Loss_v.item()

                # update progressbar/time
                batch_val_times.append(0.0)
                avg_time = sum(batch_val_times) / max(1, len(batch_val_times))
                pbar.update(1)
                pbar.set_postfix({"avg_batch_time_ms": f"{avg_time*1000:.2f}", "val_loss_batch": f"{Loss_v.item():.4f}"})
                

        #After epoch is finished
        ##########################################
        
        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / num_batches
        # Average validation loss (guard division by zero)
        avg_val_loss = val_epoch_loss / val_num_batches if val_num_batches > 0 else None

        # Log metrics to comet
        comet_experiment.log_metric("train_loss", avg_epoch_loss, step=e+1)
        comet_experiment.log_metric("val_loss", avg_val_loss, step=e+1)
        
        # Log to CSV (train, val)
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([e+1, avg_epoch_loss, avg_val_loss])


        # Save by best validation loss and handle patience
        ###################################################################
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            # Save encoder + decoder + optimizer state (single "best" checkpoint)
            torch.save({
                'epoch': e + 1,
                'encoder_state_dict': Inpainter_encoder.state_dict(),
                'decoder_state_dict': Inpainter_decoder.state_dict(),
                'optimizer_state_dict': Opt_inpainter.state_dict(),
                'val_loss': best_val_loss
            }, best_model_path)
            print(f"\n[Info] Saved new best encoder+decoder+optimizer at epoch {e+1} with val_loss {best_val_loss:.4f}")
            
            # Log to comet
            comet_experiment.log_metric("best_val_loss", best_val_loss, step=e+1)

        else:
            epochs_no_improve += 1
            print(f"[Info] No improvement in val loss ({avg_val_loss:.6f}); epochs_no_improve={epochs_no_improve}/{patience}")


        # Periodic backup (store encoder+decoder)
        if (e + 1) % backup_interval == 0:
            backup_name = f"inpainter_epoch_{e+1}.pth"
            backup_path = os.path.join("models", model_ID, "backup", backup_name)
            torch.save({
                'epoch': e + 1,
                'encoder_state_dict': Inpainter_encoder.state_dict(),
                'decoder_state_dict': Inpainter_decoder.state_dict(),
                'optimizer_state_dict': Opt_inpainter.state_dict(),
                'loss': avg_epoch_loss
            }, backup_path)
            print(f"\n[Info] Saved backup checkpoint at epoch {e+1}")
        
        if e % 1  == 0:
            visualizer.visualize(visual_batch, epoch= e+1, comet_experiment = comet_experiment)

        # Early stopping trigger based on patience
        if epochs_no_improve >= patience:
            print(f"\n[Info] Early stopping: no improvement in validation loss for {patience} epochs.")
            break

except Exception as ex:
    # Ensure Comet gets info on crash
    comet_experiment.log_text(f"Training crashed at epoch {e+1} with exception: {ex}")
    raise

                

 
        
except:
    print("TRAINING FALLBACK! Error during training. ending comet experiment.")
    #Fallback for errors in training
    comet_experiment.end()

    
try:
    comet_experiment.end()
except:
    print("Comet experiment already ended - probably error during training caused fallback of saving experiment")
