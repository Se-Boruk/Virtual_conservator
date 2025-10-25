###################################################################
# ( 0 ) Libs and dependencies 
###################################################################

#Basic libs
import sys
import os
from tqdm import tqdm
import time

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
import Architectures

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
epochs = 1
bs = 32

#Datalodaers params
n_workers = 4
max_queue = 10

#Inpainter model params
input_channels = 3  #Same as data channels if the input is rgb only (harder problem)
                    #If we can apply also the mask then it will be 4
n_residual_blocks = 4
base_filters = 64

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


dfgdfg


###################################################################
# ( 8 ) Training loop
###################################################################

for e in range(epochs):
    #################################################
    #Training part
    #################################################
    train_loader.start_epoch(shuffle=True)
    num_batches = train_loader.get_num_batches()
    batch_train_times = []
    
    
    with tqdm(total=num_batches, desc=f"Epoch {e+1} - Train", unit="batch") as pbar:
        while True:
            t0 = time.time()
            #Load batch from loader
            batch = train_loader.get_batch()
            if batch is None:
                break
            
            ##############################
            # simulate training 
            time.sleep(0.1)
            ##############################
            
            
            t1 = time.time()
            batch_train_times.append(t1 - t0)
            avg_time = sum(batch_train_times) / len(batch_train_times)
            pbar.update(1)
            pbar.set_postfix({"avg_batch_time_ms": f"{avg_time*1000:.2f}"})

            
            
    #################################################
    #Validation part
    #################################################
    val_loader.start_epoch(shuffle=True) # No need for shuffle generally in validation
    num_batches = val_loader.get_num_batches()
    batch_val_times = []
    
    
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





