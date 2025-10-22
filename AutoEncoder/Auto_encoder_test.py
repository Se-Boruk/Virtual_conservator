#Go up directly before we can take the Project ROOT from the Config
import sys
import os

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
from DataBase_manager import Custom_DataSet_Manager
from DataBase_manager import Reconstruction_data_tests
##########################################################################################


#Load manager and execute
manager = Custom_DataSet_Manager(DataSet_path = RECONSTRUCTION_DATASET_PATH,
                                 train_split = TRAIN_SPLIT,
                                 val_split = VAL_SPLIT,
                                 test_split = TEST_SPLIT,
                                 random_state = RANDOM_STATE
                                 )

#Download it if it is not flagged in the folder
if not manager.is_downloaded():
    print("DataSet is not present in given folder: Downloading...")
    manager.download_database(RECONSTRUCTION_DATASET_NAME)
    Train_set, Val_set, Test_set = manager.load_dataset_from_disk()
    print("Dataset loaded!")
    
else:
    print("DataSet is present in folder: Loading...")
    Train_set, Val_set, Test_set = manager.load_dataset_from_disk()
    print("Dataset loaded!")
    

#Run tests to see if operating on data which is the same for all:
Reconstruction_data_tests(train_subset = Train_set,
                          val_subset = Val_set,
                          test_subset = Test_set
                          )
    

    
    
