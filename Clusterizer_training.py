###################################################################
# ( 1 ) Libs and Functions
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
import hdbscan
from collections import Counter
import torchvision.transforms as T
import time
from sklearn.cluster import MiniBatchKMeans
import joblib
import matplotlib.pyplot as plt
from Config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_STATE, RECONSTRUCTION_DATASET_PATH
from tqdm import tqdm

# Function to convert images to feature vectors using the encoder
def Image_to_vector(test_dataloader, device, encoder, batch_size):
    i = 0
    features_list = []
    start = time.time()

    # Iteracja po DataLoaderze, który dostarcza już wsady (batches)
    for batch_images in tqdm(test_dataloader, desc="Przetwarzanie"):
        batch_images = batch_images.to(device)  # Przenieś wsad na urządzenie

        with torch.no_grad():
            features, _ = encoder(batch_images) # (B,C,H',W')

        # Global Average Pooling do (B, 128, 1, 1)
        features_pooled = F.adaptive_avg_pool2d(features, (1, 1))
        # spłaszcz do 1D (B, 128)
        features_flat = features_pooled.cpu().numpy().squeeze()


        if batch_size == 1:
            # Rozszerz features_list o cechy z całego wsadu
            features_list.append(features_flat)
        else:
            # Rozszerzenie listy o wszystkie wektory cech z wsadu
            features_list.extend(features_flat)

    time_elapsed = time.time() - start
    print(f"Całkowity czas przetwarzania {i} obrazów: {time_elapsed/60:.2f} minut")

    return features_list

#Tranformation for input images
def collate_fn_images_only(batch):
    transform = T.Compose([
        T.ToTensor(),                         # [0,1]
        T.Normalize([0.5]*3, [0.5]*3)         # [-1,1]
        ])

    images = [transform(item['image']) for item in batch]
    return torch.stack(images)


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
    set_len = len(train)

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
    print("\nŁadowanie modelu Inpainter...")
    checkpoint = torch.load("InPainter/models/V5_baseline_January/best_inpainter.pth", map_location=device)
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
    encoder.eval()

    BATCH_SIZE = 48  # Dostosuj tę wartość do dostępnej pamięci VRAM/RAM
    NUM_WORKERS = 4  # Dostosuj do liczby rdzeni CPU

    test_dataloader = DataLoader(
        dataset=train,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn_images_only
    )

    ###################################################################
    # ( 4 ) Feature extraction and clustering
    ###################################################################
    print(f"\nPrzetwarzanie {set_len} obrazów ({int(set_len/BATCH_SIZE) + 1} batchy) na wektory cech...")
    features_list = Image_to_vector(test_dataloader, device, encoder, BATCH_SIZE)
    #Prepartion for clustering
    pca = PCA(n_components=10)
    
    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=40,       # minimum size of clusters
        min_samples=2           # controls how conservative the clustering is
    )

    print("\nPrzeprowadzanie klasteryzacji HDBSCAN...")
    X_reduced = pca.fit_transform(features_list)
    labels = clusterer.fit_predict(X_reduced)   # shape (N,)
    counts = dict(Counter(labels))

    print(f'Liczba znalezionych klas: {len(counts)}')

    print('\nTworzenie wykresu dla HDBSCAN...')
    # Plotting HDBSCAN clustering results
    X = [i for i in counts]
    Y = [i for i in counts.values()]

    plt.figure(figsize=(12, 6))
    plt.title(f"Distribution of Images Across {len(counts)}. Clusters (HDBSCAN)")
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Images")
    plt.bar(X, Y)
    plt.show()

    # MiniBatchKMeans clustering
    K = len(counts)  # K is taken from HDBSCAN results
    kmeans = MiniBatchKMeans(
        n_clusters=K,
        random_state=42,
        batch_size=32,
        n_init='auto'
    )

    labels_kmeans = kmeans.fit_predict(X_reduced)
    counts = dict(Counter(labels_kmeans))

    print('\nTworzenie wykresu dla KMeans...')
    # Plotting KMeans clustering results
    X = [i for i in counts]
    Y = [i for i in counts.values()]

    plt.figure(figsize=(12, 6))
    plt.title(f"Distribution of Images Across Clusters (kmeans, clusters= {K})")
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Images")
    plt.bar(X, Y, color='orange')
    plt.show()

    # Save the MiniBatchKMeans model
    filename = 'minibatch_kmeans_model.joblib'
    joblib.dump(kmeans, filename)
    
    joblib.dump(pca, 'pca_model.joblib')

    print(f"\nModel został zapisany jako {filename}")