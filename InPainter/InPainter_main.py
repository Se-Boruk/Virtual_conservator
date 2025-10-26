###################################################################
# ( 0 ) Libs and dependencies 
###################################################################

#Basic libs
import sys
import os
from tqdm import tqdm
import csv

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
import torch


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
epochs = 50
bs = 16

#Datalodaers params
n_workers = 4
max_queue = 10

#Inpainter model params
input_channels = 3  #Same as data channels if the input is rgb only (harder problem)
                    #If we can apply also the mask then it will be 4
n_residual_blocks = 3
base_filters = 32

#Training params
l1_weight = 1
l2_weight = 0.1
ssim_weight = 0.05
tv_weight=  0.001

Visualization_rows = 8

###################################################################
# ( 4 ) Model creation, dataloader preparation
###################################################################

# Training loader
train_loader = Async_DataLoader(dataset = Train_set,
                                batch_size=bs,
                                num_workers=n_workers,
                                device='cuda',
                                max_queue=max_queue,
                                add_damaged = True
                                )

# Validation loader
val_loader = Async_DataLoader(dataset = Val_set,
                              batch_size=bs,
                              num_workers=n_workers,
                              device='cuda',
                              max_queue=max_queue,
                              add_damaged = True
                              )

#Taking shape of the data
img_channels = train_loader.C
img_h = train_loader.H
img_w = train_loader.W

#Creating Inpainter model
print("\nPreparing Inpainter...")
Inpainter_model = Architectures.Inpainter_v0(input_channels = input_channels,
                                             output_channels = input_channels,
                                             n_residual=n_residual_blocks,
                                             base_filters=base_filters
                                             ).to(device)

#Optimizer
Opt_inpainter = torch.optim.AdamW( list(Inpainter_model.parameters() ),
                                  lr=2e-4,
                                  betas=(0.9, 0.999),
                                  weight_decay=1e-6 
                                  )


###################################################################
# ( 5 ) Showing / saving architecture scheme [ Optional ]
###################################################################

source_shape = (input_channels,img_h,img_w)
Inp_input = torch.randn(source_shape)

Ut_lib.Show_architecture(model = Inpainter_model, 
                         input_tensor = Inp_input,
                         save_png = True,
                         name = "Inpainter_v0"
                         )


###################################################################
# ( 6 ) Final model moving just in case
###################################################################
print("\nMoving models to GPU...")
Inpainter_model = Inpainter_model.to(device)
print("Done!")





###################################################################
# ( 7 ) Preparation for saving model results in form of plots and logs
###################################################################

# For plots
####
# Initialize once before training
damage_generator = Random_Damage_Generator()

visualizer = Inp_f.InpaintingVisualizer(Inpainter_model, damage_generator, rows=Visualization_rows, H=img_h, W=img_w, device=device,
                                      save_dir="Restored_paintings", show=False, save=True)


visual_batch = val_loader.get_random_batch(batch_size = Visualization_rows, shuffle = True)
visual_batch = (visual_batch *2)-1

#For logs and models
###
os.makedirs("models", exist_ok=True)
os.makedirs("models/backup", exist_ok=True)
backup_interval = 5  # save backup every N epochs
log_csv_path = "train_loss_log.csv"

#Remove old file
if os.path.exists(log_csv_path):
    os.remove(log_csv_path)

# create fresh CSV with header
with open(log_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss"])
        
        
best_loss = float("inf")
best_model_path = "models/best_inpainter.pth"

###################################################################
# ( 8 ) Training loop
###################################################################
#Scaler for halfprecision training
scaler = torch.amp.GradScaler()


for e in range(epochs):
    #################################################
    #Training part
    #################################################
    
    train_loader.start_epoch(shuffle=True)
    
    epoch_loss = 0.0
    num_batches = train_loader.get_num_batches()

    Inpainter_model.train()
    with tqdm(total=num_batches, desc=f"Epoch {e+1}", unit=" batch") as pbar:
        while True:
            
            #Load batch from loader
            batch = train_loader.get_batch()
            if batch is None:
                break
            
            #Load batches and normalize them to -1 1 range
            original_batch = (batch[0] * 2)-1
            damaged_batch = (batch[1] * 2)-1
            ##############################
            #Simulate training 
            
            
            # ----Inpainter update----
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                
                #Make restored batch
                restored_batch = Inpainter_model(damaged_batch)


                #Loss function calculation
                Loss = Inp_f.inpainting_loss(pred = restored_batch,
                                             target = original_batch,
                                             l1_weight = l1_weight,
                                             l2_weight = l2_weight,
                                             ssim_weight = ssim_weight,
                                             tv_weight = tv_weight
                                             )
            
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

         
        #After epoch is finished
        ##########################################
        
        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / num_batches
        
        # Log to CSV
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([e+1, avg_epoch_loss])
          
            
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': e + 1,
                'model_state_dict': Inpainter_model.state_dict(),
                'optimizer_state_dict': Opt_inpainter.state_dict(),
                'loss': best_loss
            }, best_model_path)
            print(f"\n[Info] Saved new best model+optimizer at epoch {e+1} with loss {best_loss:.4f}")


        # Periodic backup
        if (e + 1) % backup_interval == 0:
            backup_path = f"models/backup/inpainter_epoch_{e+1}.pth"
            torch.save({
                'epoch': e + 1,
                'model_state_dict': Inpainter_model.state_dict(),
                'optimizer_state_dict': Opt_inpainter.state_dict(),
                'loss': avg_epoch_loss
            }, backup_path)
            print(f"\n[Info] Saved backup checkpoint at epoch {e+1}")
        
        if e % 1  == 0:
            visualizer.visualize(visual_batch, epoch= e)

          

    


    """        
    #################################################
    #Validation part
    #################################################
    val_loader.start_epoch(shuffle=False) # No need for shuffle generally in validation
    num_batches = val_loader.get_num_batches()
    batch_val_times = []
    
    Inpainter_model.eval()
    with tqdm(total=num_batches, desc=f"Epoch {e+1} - Val", unit="batch") as pbar:
        while True:
            t0 = time.time()
            #Load batch from loader
            batch = val_loader.get_batch()
            if batch is None:
                break
            
            ##############################
            # simulate validation
            time.sleep(0.05)
            ##############################
            
            
            t1 = time.time()
            batch_val_times.append(t1 - t0)
            avg_time = sum(batch_val_times) / len(batch_val_times)
            pbar.update(1)
            pbar.set_postfix({"avg_batch_time_ms": f"{avg_time*1000:.2f}"})
            
    """ 

