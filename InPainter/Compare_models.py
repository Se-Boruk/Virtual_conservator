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
import comet_ml
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

bs = 16

#Datalodaers params
n_workers = 4
max_queue = 10

#Inpainter model params
input_channels = 3  #Same as data channels if the input is rgb only (harder problem)
                    #If we can apply also the mask then it will be 4
n_residual_blocks = 3
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


# Test loader
test_loader = Async_DataLoader(dataset = Test_set,
                              batch_size=bs,
                              num_workers=n_workers,
                              device='cuda',
                              max_queue=max_queue,
                              add_damaged = True,
                              label_map = shared_mapping
                              )



#Taking shape of the data
img_channels = test_loader.C
img_h = test_loader.H
img_w = test_loader.W


#Creating Inpainter model
print("\nPreparing Inpainter Encoder (baseline)...")
Inpainter_encoder_baseline = Architectures.Inpainter_V1.Encoder(input_channels = input_channels,
                                                     n_residual=n_residual_blocks,
                                                     base_filters=base_filters
                                                     ).to('cpu')

print("\nPreparing Inpainter Decoder (baseline)...")
Inpainter_decoder_baseline = Architectures.Inpainter_V1.Decoder(output_channels = input_channels,
                                                       base_filters=base_filters
                                                       ).to('cpu')


#Creating Inpainter model
print("\nPreparing Inpainter Encoder (maxclass)...")
Inpainter_encoder_maxclass = Architectures.Inpainter_V2.Encoder(input_channels = input_channels,
                                                     n_residual=n_residual_blocks,
                                                     base_filters=base_filters
                                                     ).to('cpu')

print("\nPreparing Inpainter Decoder (maxclass)...")
Inpainter_decoder_maxclass = Architectures.Inpainter_V2.Decoder(output_channels = input_channels,
                                                       base_filters=base_filters
                                                       ).to('cpu')


##########################
#Loading the trained encoder weights:
pretrained_autoencoder_path = "models/V2_No_latent_uniforming/Baseline/best_inpainter.pth"    
checkpoint = torch.load(pretrained_autoencoder_path, map_location='cpu')
Inpainter_encoder_baseline.load_state_dict(checkpoint['encoder_state_dict'])
Inpainter_decoder_baseline.load_state_dict(checkpoint['decoder_state_dict'])

#Loading the trained encoder weights:
pretrained_autoencoder_path = "models/V2_No_latent_uniforming/Class_input_100_correct/best_inpainter.pth"    
checkpoint = torch.load(pretrained_autoencoder_path, map_location='cpu')
Inpainter_encoder_maxclass.load_state_dict(checkpoint['encoder_state_dict'])
Inpainter_decoder_maxclass.load_state_dict(checkpoint['decoder_state_dict'])


###################################################################
# ( 6 ) Final model moving just in case
###################################################################
print("\nMoving models to GPU...")
Inpainter_encoder_baseline = Inpainter_encoder_baseline.to(device)
Inpainter_decoder_baseline = Inpainter_decoder_baseline.to(device)
print("Done!")

print("\nMoving models to GPU...")
Inpainter_encoder_maxclass = Inpainter_encoder_maxclass.to(device)
Inpainter_decoder_maxclass = Inpainter_decoder_maxclass.to(device)
print("Done!")



###################################################################
# ( 7 ) Preparation for saving model results in form of plots and logs
###################################################################

#Initialize once before training
damage_generator = Random_Damage_Generator()

restored_paintings_path = os.path.join("Restored_paintings", "Tests")
visualizer_baseline = Inp_f.InpaintingVisualizer(Inpainter_encoder_baseline, Inpainter_decoder_baseline, damage_generator, rows=Visualization_rows, H=img_h, W=img_w, device=device,
                                      save_dir=restored_paintings_path, save=True)


visualizer_maxclass = Inp_f.InpaintingVisualizer(Inpainter_encoder_maxclass, Inpainter_decoder_maxclass, damage_generator, rows=Visualization_rows, H=img_h, W=img_w, device=device,
                                      save_dir=restored_paintings_path, save=True)



#Random sample from the test set so its not affecting training in any way
visual_batch = test_loader.get_random_batch(batch_size = Visualization_rows, shuffle = True, random_state = RANDOM_STATE)

#Apply small augmentation
########################
shared_masks = visualizer_baseline.masks
from DataBase_Functions import augment_image_and_mask

visual_batch, shared_mask = augment_image_and_mask( visual_batch, shared_masks )
visual_batch = (visual_batch *2)-1

#Ensure the same mask
visualizer_baseline.masks = shared_masks 
visualizer_maxclass.masks = shared_masks

#####################



#Visualize baseline
visualizer_baseline.visualize(visual_batch, epoch = 0, comet_experiment = None)
visualizer_maxclass.visualize(visual_batch, epoch = 1, comet_experiment = None)
