###################################################################
# ( 1 ) Libs 
###################################################################

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from DataBase.DataBase_Functions import Custom_DataSet_Manager as DB
from DataBase.DataBase_Functions import Custom_DataSet_Manager as DB
import torch
import torch.nn.functional as F
from InPainter.Architectures import Inpainter_V5 as Inp5
from torch.utils.data import DataLoader
import os
from sklearn.decomposition import PCA
from collections import Counter
import torchvision.transforms as T
import joblib
import matplotlib.pyplot as plt
from Config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_STATE, RECONSTRUCTION_DATASET_PATH
from Clusterizer_training import collate_fn_images_only, Image_to_vector
import numpy as np

###################################################################
# ( 2 ) Loading data
###################################################################
if __name__ == '__main__':
    manager = DB(
        DataSet_path = RECONSTRUCTION_DATASET_PATH,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        random_state=RANDOM_STATE
        )

    # Download database if not exists and load dataset
    manager.download_database("Artificio/WikiArt_Full")
    train, val, test = manager.load_dataset_from_disk()
    test_size = len(test)

    ###################################################################
    # ( 3 ) Setting up the model and dataloader
    ###################################################################

    #Set device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Uruchomione na:", device)

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

    # Load the pre-trained model checkpoint
    checkpoint = torch.load("InPainter/models/V5_Baseline/best_inpainter.pth", map_location=device)
    full_state = checkpoint["encoder_state_dict"]

    # Initialize the encoder and load the filtered state dict
    encoder = Inp5.Encoder(input_channels=3, base_filters=32).to(device)
    encoder_state = encoder.state_dict()     # encoder keys
    filtered_state = {
        k: v for k, v in full_state.items()
        if k in encoder_state
    }
    encoder_state.update(filtered_state)
    encoder.load_state_dict(encoder_state)

    BATCH_SIZE = 32  # Dostosuj tę wartość do dostępnej pamięci VRAM/RAM
    NUM_WORKERS = 4  # Dostosuj do liczby rdzeni CPU

    test_dataloader = DataLoader(
        dataset=test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn_images_only
    )

    ###################################################################
    # ( 4 ) Feature extraction and clustering
    ###################################################################

    print(f"\nPrzetwarzanie {test_size} obrazów ({int(test_size/BATCH_SIZE) + 1} batchy) na wektory cech...")
    features_list = Image_to_vector(test_dataloader, device, encoder, BATCH_SIZE)

    #Reduction of dimensions with PCA
    pca_loaded = joblib.load('pca_model.joblib')
    
    X_reduced = pca_loaded.transform(features_list)

    #KMeans clustering
    loaded_kmeans = joblib.load('minibatch_kmeans_model.joblib')
    labels_loaded = loaded_kmeans.predict(X_reduced)
    

    
    counts = dict(Counter(labels_loaded))

    # Plotting KMeans clustering results
    X = [i for i in counts]
    Y = [i for i in counts.values()]

    plt.figure(figsize=(12, 6))
    plt.title(f"Distribution of Images Across Clusters")
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Images")
    plt.bar(X, Y, color='orange')
    plt.show()

    # Labels saving in .npy file
    labels_path = 'test_set_cluster_labels.npy'
    np.save(labels_path, labels_loaded)
    print(f"Etykiety zostały zapisane do pliku: {labels_path}")

