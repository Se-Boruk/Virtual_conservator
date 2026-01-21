import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from DataBase.DataBase_Functions import Custom_DataSet_Manager as DB
from DataBase.DataBase_Functions import Custom_DataSet_Manager as DB
import Config as con
from datasets import load_dataset
import torch
import torch.nn.functional as F
from InPainter.Architectures import Inpainter_V5 as Inp5
from torch.utils.data import DataLoader
import os
from sklearn.decomposition import PCA
import hdbscan
from collections import Counter
import itertools
import torchvision.transforms as T
import numpy as np
import time
from sklearn.cluster import MiniBatchKMeans
import joblib
import matplotlib.pyplot as plt

manager = DB(
    DataSet_path=r'C:\Users\jakub\Desktop\Repozytoria Github\Databases\Uczenie ze wzmocnieniem', # Path to the folder where the database will be stored
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    random_state=259189
    )

# Download database if not exists
manager.download_database("Artificio/WikiArt_Full")

#Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Uruchomione na:", device)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# Load the pre-trained model checkpoint
checkpoint = torch.load("InPainter/V5_baseline.pth", map_location=device)
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

transform = T.Compose([
    T.ToTensor(),                         # [0,1]
    T.Normalize([0.5]*3, [0.5]*3)         # [-1,1]
])

def collate_fn_images_only(batch):
    images = [transform(item['image']) for item in batch]
    return torch.stack(images)

BATCH_SIZE = 32  # Dostosuj tę wartość do dostępnej pamięci VRAM/RAM
NUM_WORKERS = 0  # Dostosuj do liczby rdzeni CPU

test_dataloader = DataLoader(
    dataset=train,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn_images_only
    # lambda x: [torch.stack([transform(item['image']) for item in x])]
)

i = 0
features_list = []
start = time.time()

# Iteracja po DataLoaderze, który dostarcza już wsady (batches)
for batch_images in test_dataloader:
    batch_images = batch_images.to(device)  # Przenieś wsad na urządzenie

    with torch.no_grad():
        features, _ = encoder(batch_images) # (B,C,H',W')

    # Global Average Pooling do (B, 128, 1, 1)
    features_pooled = F.adaptive_avg_pool2d(features, (1, 1))
    # spłaszcz do 1D (B, 128)
    features_flat = features_pooled.cpu().numpy().squeeze()


    if BATCH_SIZE == 1:
        # Rozszerz features_list o cechy z całego wsadu
        features_list.append(features_flat)
    else:
        # Rozszerzenie listy o wszystkie wektory cech z wsadu
        features_list.extend(features_flat)

    i += BATCH_SIZE
    if i % 3200 == 0:
        print(f"Przetworzono {i} obrazów w czasie {(time.time() - start)/60:.2f} minut")

time_elapsed = time.time() - start
print(f"Całkowity czas przetwarzania {i} obrazów: {time_elapsed/60:.2f} minut")

pca = PCA(n_components=10)
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=30,       # minimalny rozmiar klastra
    min_samples=1           # bardziej konserwatywny = mniej szumu
)

X_reduced = pca.fit_transform(features_list)

labels = clusterer.fit_predict(X_reduced)   # shape (N,)

print(f'Wektor labelek: {labels}')

counts = dict(Counter(labels))

print(counts)
print(len(counts))

X = [i for i in counts]
Y = [i for i in counts.values()]

plt.figure(figsize=(12, 6))
plt.title("Distribution of Images Across Clusters (HDBSCAN)")
plt.xlabel("Cluster Label")
plt.ylabel("Number of Images")
plt.bar(X, Y)
plt.show()

max(counts.values())





# Używamy liczby unikalnych stylów (np. 134 z Cell 8) jako K
K = len(counts)  # Proszę dostosować
kmeans = MiniBatchKMeans(
    n_clusters=K,
    random_state=42,
    batch_size=256,
    n_init='auto' # Automatyczne wybieranie liczby inicjalizacji
)
labels_kmeans = kmeans.fit_predict(X_reduced)

counts = dict(Counter(labels_kmeans))

X = [i for i in counts]
Y = [i for i in counts.values()]

plt.figure(figsize=(12, 6))
plt.title(f"Distribution of Images Across Clusters (kmeans, clusters= {K})")
plt.xlabel("Cluster Label")
plt.ylabel("Number of Images")
plt.bar(X, Y, color='orange')
plt.show()

filename = 'minibatch_kmeans_model.joblib'

# Zapis do pliku
joblib.dump(kmeans, filename)

print(f"Model został zapisany jako {filename}")