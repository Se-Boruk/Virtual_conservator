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
##########################################################################################
#Move folder up to go to database folder to use manager from here
sys.path.insert(0, DATABASE_FOLDER)
from DataBase_Functions import Custom_DataSet_Manager
from DataBase_Functions import Reconstruction_data_tests
from DataBase_Functions import Async_DataLoader


##########################################################################################
#Data loading (Not stored in RAM but is accessed on demand)
##########################################################################################

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

##########################################################################################
#Preparing dataloaders and hyperparams
##########################################################################################

#Hyperparams
epochs = 1
bs = 32
n_workers = 4
max_queue = 10


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


##########################################################################################
#Test of train loop with async data processing (damaging and converting into tensors)
##########################################################################################

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





