# Lab 5: experimenting with PyTorch
# %%

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import CometLogger
from waste_dataset_multiclass import CombinedWasteDatasetMulti, final_classes
from torchsummary import summary
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import comet_ml

# Set seed for reproducibility
pl.seed_everything(42)

#%%



# Validation and test transforms
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#%% 

class WasteDataModuleMulti(LightningDataModule):
    def __init__(self, root_dir='datasets/combined_waste_dataset',
                 batch_size=32):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_classes = 134  # organic, battery, glass, metal, paper, cardboard, plastic, textiles, trash
        self.class_weights = None

    def setup(self, stage=None):
        self.train_dataset = CombinedWasteDatasetMulti(
            root_dir=self.root_dir, split='train', transform=train_transform
        )
        self.val_dataset = CombinedWasteDatasetMulti(
            root_dir=self.root_dir, split='val', transform=val_test_transform
        )
        self.test_dataset = CombinedWasteDatasetMulti(
            root_dir=self.root_dir, split='test', transform=val_test_transform
        )

        self.num_workers = os.cpu_count() - 1


        if self.class_weights is not None:
            self.sample_weights = [self.class_weights[label].item() for _, label in self.train_dataset.data]

    def train_dataloader(self):
        if hasattr(self, 'sample_weights'):

            sampler = WeightedRandomSampler(self.sample_weights, len(self.train_dataset), replacement=True)
            
            return DataLoader(self.train_dataset, batch_size=self.batch_size,
                              sampler=sampler, num_workers=self.num_workers)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size,
                              shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)


# %%
'''
Now that we feel comfortable with pytorch it is time to create some more advanced architectures.
Not all of them will make sense (at first), but let us try and create something, that will not only use many layers, 
BUT also show us, that we do not have to think very "lineary" when creating architectures.

'''
# =====================================================================

# architecutre 1. 

'''
We will be using the Waste dataset multiclass again for this experiment.
However, there will be a simple change to our architecture. 

We don't want to create a simple classifier, that just uses a sequence of layers.
We want to create a more complex architecture, that uses multiple paths and then combines them.
We will create a model that has two parallel paths:
- Path 1: A series of convolutional layers
- Path 2: A series of fully connected layers

The outputs of these two paths will be concatenated and then passed through a final classification layer.

'''

# %%
class DualPathClassifier(nn.Module):
    """
    A classifier with two parallel processing paths:
    - Path 1 (CNN): Processes image through convolutional layers (learns spatial features)
    - Path 2 (MLP): Processes flattened image through fully connected layers (learns global patterns)
    - Fusion: Concatenates both paths and makes final classification
    """
    def __init__(self, num_classes=9):
        super().__init__()
        
        # PATH 1: Convolutional path (learns spatial features)
        self.conv_path = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Reduce to 4x4
        )
        
        # Conv path output: 256 * 4 * 4 = 4096 features
        self.conv_flatten = nn.Flatten()
        self.conv_fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # PATH 2: Fully connected path (learns global patterns from raw pixels)
        self.fc_path = nn.Sequential(
            nn.Flatten(),  # 3 * 224 * 224 = 150,528 features
            nn.Linear(3 * 224 * 224, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # FUSION: Combine both paths
        # Conv path gives 512 features, FC path gives 512 features -> 1024 total
        self.fusion = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Path 1: Process through CNN
        conv_features = self.conv_path(x)
        conv_features = self.conv_flatten(conv_features)
        conv_features = self.conv_fc(conv_features)
        
        # Path 2: Process through FC layers (directly from input)
        fc_features = self.fc_path(x)
        
        # Concatenate both paths
        # remember - these are all tensors, so we can use simple operations
        combined = torch.cat([conv_features, fc_features], dim=1)
        
        # Final classification
        output = self.fusion(combined)
        
        return output


# %%
class DualPathLightningModule(LightningModule):
    """
    PyTorch Lightning wrapper for the DualPathClassifier.
    Simple setup: no Comet, minimal callbacks, just training essentials.
    """
    def __init__(self, num_classes=9, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = DualPathClassifier(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        
        # For tracking metrics
        self.train_acc = []
        self.val_acc = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler: reduce on plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }


# %%
# Test the model architecture
# dual path model - as a test to what can we do and why (cause why not)

model = DualPathClassifier(num_classes=9)
dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images

print("\nTesting forward pass...")
output = model(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Expected: (2, 9) for 2 samples, 9 classes")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

'''
Path 1 (CNN): Conv blocks → Global pooling → FC
  - Learns spatial features and patterns
  - Parameter efficient (weight sharing)
  - Output: 512 features

Path 2 (MLP): Flatten → FC → FC
  - Learns global patterns from all pixels
  - More parameters (no weight sharing)
  - Output: 512 features

Fusion: Concatenate → FC → Output
  - Combines both representations
  - Input: 1024 features (512 + 512)
  - Output: 9 classes
'''

# %%
# Training setup

# Create data module
data_module = WasteDataModuleMulti(
    root_dir='datasets/combined_waste_dataset',
    batch_size=32
)

# Create model
lightning_model = DualPathLightningModule(
    num_classes=9,
    learning_rate=1e-3
)

# Create callbacks
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=True,
    mode='min'
)

# Create trainer
trainer = Trainer(
    max_epochs=1,
    callbacks=[early_stop_callback],
    accelerator='auto',  # Automatically use GPU if available
    devices=1,
    log_every_n_steps=10,
)

print(f"Max epochs: {trainer.max_epochs}")
print(f"Early stopping patience: {early_stop_callback.patience}")
print(f"Batch size: {data_module.batch_size}")
print(f"Learning rate: {lightning_model.learning_rate}")


# %%
# Train the model
print("training model...")

trainer.fit(lightning_model, data_module)


# %%
# Test the model
print("testing model...")

test_results = trainer.test(lightning_model, data_module)

print("\nTest Results:")
print(f"Test Loss: {test_results[0]['test_loss']:.4f}")
print(f"Test Accuracy: {test_results[0]['test_acc']:.4f}")

# %%
# Visualize some predictions

# Get a batch from test set
data_module.setup()
test_loader = data_module.test_dataloader()
images, labels = next(iter(test_loader))

# Make predictions
lightning_model.eval()
with torch.no_grad():
    outputs = lightning_model(images)
    _, predicted = torch.max(outputs, 1)

# Denormalize images for visualization
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
images_denorm = images * std + mean
images_denorm = torch.clamp(images_denorm, 0, 1)

# Plot first 8 images
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

class_names = list(final_classes.keys())

for idx in range(min(8, len(images))):
    img = images_denorm[idx].permute(1, 2, 0).cpu().numpy()
    axes[idx].imshow(img)
    axes[idx].axis('off')
    
    true_label = class_names[labels[idx]]
    pred_label = class_names[predicted[idx]]
    color = 'green' if labels[idx] == predicted[idx] else 'red'
    
    axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}', 
                        color=color, fontsize=10)

plt.tight_layout()
plt.savefig('dual_path_predictions.png', dpi=150, bbox_inches='tight')
print("Predictions saved to 'dual_path_predictions.png'")
plt.show()


#%%


'''
Ok, but is this even useful?
Oh very much so. Many actual architectures use multiple paths to learn different types of features.
Not too look too far - Inception networks use multiple convolutional paths with different kernel sizes to capture features at multiple scales.
Here we did something simple - just two paths, but the concept is exactly the same

We can read more about Inception architectures here:
https://arxiv.org/abs/1409.4842
https://viso.ai/deep-learning/googlenet-explained-the-inception-model-that-won-imagenet/

'''


# %%

'''
Ok, we have created one "weird" architecure, let's create another one.

For this test let's try something different. What about a model, that has multiple inputs?

let's imagine a situation - we want to compare images, and just say if they belong to the same class or not.
Why would we do that? Simple - maybe we have two cameras looking at the same object from different angles, and we want to know if they are looking at the same type of object.
Or maybe we have two different sensors (like RGB and Infrared), and we want to know if they are looking at the same type of object.

We can basically train our model to learn the concept of "sameness" or "difference" between two images.
This can be useful in various applications like verification systems, duplicate detection, etc.

So lets try and create such a model.

This is called a SIAMESE NETWORK - it uses the same network weights to process both images,
then compares their representations to determine similarity.
'''

# %%
# SIAMESE NETWORK FOR IMAGE COMPARISON
# Welll, an "approximation" of the full siamese network concept
# For now, let's imagine we want to create a model that can tell if two images are of the same class or not.

# first of all - some preprocessing
# we need to start with a new dataset (based on what we have naturally)
# we get two images and simple information - are they from the same class or not


from torch.utils.data import Dataset
import random


class SiamesePairDataset(Dataset):
    """
    Dataset that creates pairs of images from the waste dataset.
    
    For each sample, returns:
    - img1: First image
    - img2: Second image
    - label: 1 if same class, 0 if different class
    """
    def __init__(self, base_dataset, num_pairs_per_epoch=5000):
        """
        Args:
            base_dataset: The original waste dataset (train/val/test)
            num_pairs_per_epoch: How many pairs to generate per epoch
        """
        self.base_dataset = base_dataset
        self.num_pairs_per_epoch = num_pairs_per_epoch
        
        # Organize images by class for efficient pair generation
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(base_dataset.data):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
        
        self.classes = list(self.class_to_indices.keys())
        print(f"SiamesePairDataset created with {len(self.classes)} classes")
        for cls in self.classes:
            print(f"  Class {cls}: {len(self.class_to_indices[cls])} images")
    
    def __len__(self):
        return self.num_pairs_per_epoch
    
    def __getitem__(self, idx):
        # Randomly decide if we want a positive pair (same class) or negative pair (different class)
        is_same_class = random.random() > 0.5
        
        if is_same_class:
            # Pick a random class
            target_class = random.choice(self.classes)
            # Pick two different images from that class
            if len(self.class_to_indices[target_class]) < 2:
                # If only one image in class, fall back to different class
                is_same_class = False
            else:
                idx1, idx2 = random.sample(self.class_to_indices[target_class], 2)
                img1, _ = self.base_dataset[idx1]
                img2, _ = self.base_dataset[idx2]
                label = 1  # Same class
        
        if not is_same_class:
            # Pick two different classes
            class1, class2 = random.sample(self.classes, 2)
            # Pick one image from each class
            idx1 = random.choice(self.class_to_indices[class1])
            idx2 = random.choice(self.class_to_indices[class2])
            img1, _ = self.base_dataset[idx1]
            img2, _ = self.base_dataset[idx2]
            label = 0  # Different class
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)


# %%
class SiameseNetwork(nn.Module):
    """
    Siamese Network for comparing two images.
    
    Architecture:
    1. Shared CNN (same weights for both images) - extracts features
    2. Feature comparison layer - computes similarity
    3. Classification - same or different class
    """
    def __init__(self):
        super().__init__()
        
        # Shared feature extractor (used for BOTH images)
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            
            nn.Flatten(),
        )
        
        # Embedding layer - reduces features to a fixed-size embedding
        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),  # Final embedding size
        )
        
        # Comparison layer - takes concatenated embeddings and decides similarity
        self.comparison = nn.Sequential(
            nn.Linear(128 * 2, 64),  # 128 from each image
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),  # Output: probability of being same class
            nn.Sigmoid(),  # Squash to [0, 1]
        )
    
    def forward_one(self, x):
        features = self.feature_extractor(x)
        embedding = self.embedding(features)
        return embedding
    
    def forward(self, img1, img2):
        """
        Process both images and compare them.
        
        Args:
            img1: First image batch (B, 3, 224, 224)
            img2: Second image batch (B, 3, 224, 224)
        
        Returns:
            similarity: Probability that images are from same class (B, 1)
        """
        # Extract embeddings for both images using SHARED weights
        # so inside our "forward" pass we have multiple "forwards"
        embedding1 = self.forward_one(img1)
        embedding2 = self.forward_one(img2)
        
        # Concatenate embeddings
        combined = torch.cat([embedding1, embedding2], dim=1)
        
        # Compare and output similarity
        similarity = self.comparison(combined)
        
        return similarity


# %%
class SiameseLightningModule(LightningModule):
    """
    PyTorch Lightning wrapper for Siamese Network.
    Uses Binary Cross Entropy loss since we're predicting same/different (binary).
    """
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = SiameseNetwork()
        self.criterion = nn.BCELoss()  # Binary classification - TODO: why this one this time???
        self.learning_rate = learning_rate
    
    def forward(self, img1, img2):
        return self.model(img1, img2)
    
    def training_step(self, batch, batch_idx):
        img1, img2, labels = batch
        outputs = self(img1, img2).squeeze()
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy (threshold at 0.5)
        # remember - thresholds are usually VERY arbitrary
        # we decide, when is something "same" or "different"
        # in real applications this might be tuned further
        predictions = (outputs > 0.5).float()
        acc = (predictions == labels).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        img1, img2, labels = batch
        outputs = self(img1, img2).squeeze()
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        predictions = (outputs > 0.5).float()
        acc = (predictions == labels).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        img1, img2, labels = batch
        outputs = self(img1, img2).squeeze()
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        predictions = (outputs > 0.5).float()
        acc = (predictions == labels).float().mean()
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }


# %%
class SiameseDataModule(LightningDataModule):
    """
    Data module that creates paired datasets for Siamese Network training.
    """
    def __init__(self, root_dir='datasets/combined_waste_dataset', batch_size=32):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = os.cpu_count() - 1
    
    def setup(self, stage=None):
        # Create base datasets
        train_base = CombinedWasteDatasetMulti(
            root_dir=self.root_dir, split='train', transform=train_transform
        )
        val_base = CombinedWasteDatasetMulti(
            root_dir=self.root_dir, split='val', transform=val_test_transform
        )
        test_base = CombinedWasteDatasetMulti(
            root_dir=self.root_dir, split='test', transform=val_test_transform
        )
        
        # Create paired datasets
        # Generate more pairs for training, fewer for val/test
        # TODO: are we doing this right???
        self.train_dataset = SiamesePairDataset(train_base, num_pairs_per_epoch=1000)
        self.val_dataset = SiamesePairDataset(val_base, num_pairs_per_epoch=200)
        self.test_dataset = SiamesePairDataset(test_base, num_pairs_per_epoch=200)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


# %%
# Test the Siamese Network architecture
siamese_model = SiameseNetwork()

# Test with two dummy images
dummy_img1 = torch.randn(2, 3, 224, 224)
dummy_img2 = torch.randn(2, 3, 224, 224)

print("\nTesting forward pass...")
similarity = siamese_model(dummy_img1, dummy_img2)
print(f"Image 1 shape: {dummy_img1.shape}")
print(f"Image 2 shape: {dummy_img2.shape}")
print(f"Similarity output shape: {similarity.shape}")
print(f"Similarity values (0=different, 1=same): {similarity.squeeze()}")

# Count parameters
total_params = sum(p.numel() for p in siamese_model.parameters())
trainable_params = sum(p.numel() for p in siamese_model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

'''
So, what we have:

1. Shared Feature Extractor:
   - SAME CNN processes BOTH images
   - Learns to extract meaningful features
   - Weight sharing ensures consistent feature extraction

2. Embedding Layer:
   - Reduces features to 128-dimensional embedding
        TODO: do we understand the concept of embedding???
   - Creates compact representation of each image

3. Comparison Layer:
   - Takes both embeddings (concatenated)
   - Learns to measure similarity
   - Output: probability that images are from same class

4. Key Concept: WEIGHT SHARING
   - Both images go through the SAME network
   - Ensures fair comparison
   - This is what makes it 'Siamese'
'''

# %%
# Training setup for Siamese Network
print("training siamese network...")
# Create data module
siamese_data_module = SiameseDataModule(
    root_dir='datasets/combined_waste_dataset',
    batch_size=32
)

# Create model
siamese_lightning_model = SiameseLightningModule(learning_rate=1e-3)

# Create callbacks
siamese_early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=True,
    mode='min'
)

# Create trainer
siamese_trainer = Trainer(
    max_epochs=1,  # Set to 1 for quick test, increase for real training
    callbacks=[siamese_early_stop],
    accelerator='auto',
    devices=1,
    log_every_n_steps=10,
)

print(f"Max epochs: {siamese_trainer.max_epochs}")
print(f"Early stopping patience: {siamese_early_stop.patience}")
print(f"Batch size: {siamese_data_module.batch_size}")
print(f"Learning rate: {siamese_lightning_model.learning_rate}")
print("\nNote: Each 'epoch' generates new random pairs from the dataset")


# %%
# Train the Siamese Network
print("training siamese network...")

siamese_trainer.fit(siamese_lightning_model, siamese_data_module)

# %%
# Test the Siamese Network
print("testing siamese network...")

siamese_test_results = siamese_trainer.test(siamese_lightning_model, siamese_data_module)

print("\nTest Results:")
print(f"Test Loss: {siamese_test_results[0]['test_loss']:.4f}")
print(f"Test Accuracy: {siamese_test_results[0]['test_acc']:.4f}")



# %%
# Visualize Siamese Network predictions

# Get a batch from test set
siamese_data_module.setup()
test_loader = siamese_data_module.test_dataloader()
img1_batch, img2_batch, labels_batch = next(iter(test_loader))

# Make predictions
siamese_lightning_model.eval()
with torch.no_grad():
    similarities = siamese_lightning_model(img1_batch, img2_batch).squeeze()
    predictions = (similarities > 0.5).float()

# Denormalize images for visualization
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

img1_denorm = img1_batch * std + mean
img1_denorm = torch.clamp(img1_denorm, 0, 1)

img2_denorm = img2_batch * std + mean
img2_denorm = torch.clamp(img2_denorm, 0, 1)

# Plot first 4 pairs
fig, axes = plt.subplots(4, 2, figsize=(10, 16))

for idx in range(min(4, len(img1_batch))):
    # Image 1
    img1 = img1_denorm[idx].permute(1, 2, 0).cpu().numpy()
    axes[idx, 0].imshow(img1)
    axes[idx, 0].axis('off')
    axes[idx, 0].set_title(f'Image 1 (Pair {idx+1})')
    
    # Image 2
    img2 = img2_denorm[idx].permute(1, 2, 0).cpu().numpy()
    axes[idx, 1].imshow(img2)
    axes[idx, 1].axis('off')
    axes[idx, 1].set_title(f'Image 2 (Pair {idx+1})')
    
    # Add text with prediction
    true_label = "SAME" if labels_batch[idx] == 1 else "DIFFERENT"
    pred_label = "SAME" if predictions[idx] == 1 else "DIFFERENT"
    similarity_score = similarities[idx].item()
    
    correct = (predictions[idx] == labels_batch[idx])
    color = 'green' if correct else 'red'
    
    fig.text(0.5, 1 - (idx + 0.5) / 4, 
             f'True: {true_label} | Predicted: {pred_label} | Similarity: {similarity_score:.3f}',
             ha='center', fontsize=12, color=color, weight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.95, hspace=0.3)
plt.savefig('siamese_predictions.png', dpi=150, bbox_inches='tight')
print("Predictions saved to 'siamese_predictions.png'")
plt.show()

print("=" * 70)

# %%

'''
Ok, let's try one more architecture - just for the fun of it.

For now something simple - an "autoencoder".

But what is an autoencoder?
This is a very fun concept, when we try to "compress" the information and create an "embedding". 
So a representation of the image in a smaller space, that still contains the important information about the image.

We do this by creating a model that has two parts:
- Encoder: compresses the image into a smaller representation (latent space)
- Decoder: reconstructs the image from the compressed representation

And, spoiler, we can use these parts independently, and that is how we will create this architecture, 
to see how we can both create two separate parts, but also use them together.
'''

# %%
# Convolutional autoencoder with separate Encoder and Decoder modules


class Encoder(nn.Module):
    """
    Encoder: Compresses image into a latent representation (embedding).
    
    Input: (B, 3, 224, 224) - RGB image
    Output: (B, 128) - Compressed latent vector
    
    This module can be used INDEPENDENTLY for:
    - Feature extraction
    - Transfer learning
    - Building classifiers
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Global pooling: 14x14 -> 1x1
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        
        # Final compression to latent space
        self.fc = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )
        
        self.latent_dim = latent_dim
    
    def forward(self, x):
        """
        Compress image to latent representation.
        
        Args:
            x: Input images (B, 3, 224, 224)
        
        Returns:
            latent: Latent representation (B, latent_dim)
        """
        features = self.conv_layers(x)
        latent = self.fc(features)
        return latent


class Decoder(nn.Module):
    """
    Decoder: Reconstructs image from latent representation.
    
    Input: (B, 128) - Latent vector
    Output: (B, 3, 224, 224) - Reconstructed RGB image
    
    Uses ConvTranspose2d for upsampling.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # Expand latent vector
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256 * 14 * 14),
            nn.ReLU(),
        )
        
        # Reshape to feature maps
        self.unflatten = nn.Unflatten(1, (256, 14, 14))
        
        # Transposed convolutions for upsampling
        self.deconv_layers = nn.Sequential(
            # 14x14 -> 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 56x56 -> 112x112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 112x112 -> 224x224
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Final layer to RGB
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh(),  # Output in range [-1, 1]
        )
    
    def forward(self, latent):
        """
        Reconstruct image from latent representation.
        
        Args:
            latent: Latent representation (B, latent_dim)
        
        Returns:
            reconstruction: Reconstructed images (B, 3, 224, 224)
        """
        x = self.fc(latent)
        x = self.unflatten(x)
        reconstruction = self.deconv_layers(x)
        return reconstruction


class Autoencoder(nn.Module):
    """
    Complete Autoencoder: Encoder + Decoder.
    
    Can use encoder and decoder INDEPENDENTLY after training.
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim
    
    def forward(self, x):
        """
        Encode and decode image.
        
        Args:
            x: Input images (B, 3, 224, 224)
        
        Returns:
            reconstruction: Reconstructed images (B, 3, 224, 224)
            latent: Latent representation (B, latent_dim)
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent
    
    def encode(self, x):
        """Encode only."""
        return self.encoder(x)
    
    def decode(self, latent):
        """Decode only."""
        return self.decoder(latent)


# %%
class AutoencoderLightningModule(LightningModule):
    """
    PyTorch Lightning wrapper for Autoencoder.
    
    Uses MSE loss to train the autoencoder to reconstruct images.
    We want to learn "meaningful" compressed representations.
    Something, that will represent the image well, even in a smaller space.
    This is slightly different from some other tasks, where we want to represent 
    our data in a different, "better" space, not necessarily smaller.

    TODO: is 128 dimmensional latent space good? maybe it should be smaller/larger???
    TODO the second (very simple): why MSE loss ???? 

    """
    def __init__(self, latent_dim=128, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = Autoencoder(latent_dim=latent_dim)
        self.criterion = nn.MSELoss()  # Reconstruction loss
        self.learning_rate = learning_rate
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, _ = batch  # We don't need labels for autoencoder!
        reconstruction, latent = self(images)
        
        # Calculate reconstruction loss
        loss = self.criterion(reconstruction, images)
        
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, _ = batch
        reconstruction, latent = self(images)
        
        loss = self.criterion(reconstruction, images)
        
        self.log('val_loss', loss, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        images, _ = batch
        reconstruction, latent = self(images)
        
        loss = self.criterion(reconstruction, images)
        
        self.log('test_loss', loss)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }


# %%
# Test the Autoencoder architecture
print("autoencoder architecture")
autoencoder = Autoencoder(latent_dim=128)
dummy_input = torch.randn(2, 3, 224, 224)

print("\nTesting forward pass...")
reconstruction, latent = autoencoder(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Latent shape: {latent.shape}")
print(f"Reconstruction shape: {reconstruction.shape}")
print(f"\nCompression: 224x224x3 = 150528 → 128 (latent) → 150528")
print(f"Compression ratio: {(224*224*3) / 128:.1f}x")

# Count parameters
encoder_params = sum(p.numel() for p in autoencoder.encoder.parameters())
decoder_params = sum(p.numel() for p in autoencoder.decoder.parameters())
total_params = sum(p.numel() for p in autoencoder.parameters())

print(f"\nEncoder parameters: {encoder_params:,}")
print(f"Decoder parameters: {decoder_params:,}")
print(f"Total parameters: {total_params:,}")


'''
1. ENCODER: Compresses image to latent representation
   - Captures essential features
   - Discards unnecessary details
   - Can be used independently!

2. LATENT SPACE: Compressed representation
   - 128-dimensional vector
   - Contains 'essence' of the image
   - Useful for many tasks

3. DECODER: Reconstructs image from latent
   - Learns to 'undo' the compression
   - Can generate new images!

4. MODULARITY: Encoder and Decoder are SEPARATE
   - Can use encoder for classification
   - Can use decoder for generation
   - This is the power of modular design!
'''

# %%
# Training setup for Autoencoder


# Use the same data module (we only need images, not labels)
ae_data_module = WasteDataModuleMulti(
    root_dir='datasets/combined_waste_dataset',
    batch_size=32
)

# Create model
ae_lightning_model = AutoencoderLightningModule(
    latent_dim=128,
    learning_rate=1e-3
)

# Create callbacks
ae_early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=True,
    mode='min'
)

# Create trainer
ae_trainer = Trainer(
    max_epochs=1,  # Increase for real training
    callbacks=[ae_early_stop],
    accelerator='auto',
    devices=1,
    log_every_n_steps=10,
)

print(f"Max epochs: {ae_trainer.max_epochs}")
print(f"Batch size: {ae_data_module.batch_size}")
print(f"Learning rate: {ae_lightning_model.learning_rate}")
print(f"Latent dimension: {ae_lightning_model.model.latent_dim}")
print("\nNote: Training autoencoder to reconstruct images")

# %%
# Train the Autoencoder
print("training autoencoder...")
ae_trainer.fit(ae_lightning_model, ae_data_module)

# %%
# Visualize Autoencoder reconstructions
print("testing autoencoder (visualizing the reconstructions)...")

# Get a batch from test set
ae_data_module.setup()
test_loader = ae_data_module.test_dataloader()
images, labels = next(iter(test_loader))

# Generate reconstructions
ae_lightning_model.eval()
with torch.inference_mode():
    reconstructions, latents = ae_lightning_model(images)

# Denormalize images for visualization
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

images_denorm = images * std + mean
images_denorm = torch.clamp(images_denorm, 0, 1)

# Denormalize reconstructions (they're in [-1, 1] from Tanh)
reconstructions_denorm = (reconstructions + 1) / 2  # [-1, 1] -> [0, 1]
reconstructions_denorm = torch.clamp(reconstructions_denorm, 0, 1)

# Plot original vs reconstruction
fig, axes = plt.subplots(4, 4, figsize=(16, 16))

for idx in range(min(4, len(images))):
    # Original image
    img_orig = images_denorm[idx].permute(1, 2, 0).cpu().numpy()
    axes[idx, 0].imshow(img_orig)
    axes[idx, 0].axis('off')
    axes[idx, 0].set_title('Original', fontsize=14, weight='bold')
    
    # Reconstructed image
    img_recon = reconstructions_denorm[idx].permute(1, 2, 0).cpu().numpy()
    axes[idx, 1].imshow(img_recon)
    axes[idx, 1].axis('off')
    axes[idx, 1].set_title('Reconstructed', fontsize=14, weight='bold')
    
    # Difference (amplified for visibility)
    diff = np.abs(img_orig - img_recon) * 5  # Amplify differences
    axes[idx, 2].imshow(diff)
    axes[idx, 2].axis('off')
    axes[idx, 2].set_title('Difference (5x)', fontsize=14, weight='bold')
    
    # Latent visualization (first 64 dimensions)
    latent_vec = latents[idx, :64].cpu().numpy()
    axes[idx, 3].bar(range(64), latent_vec, color='steelblue')
    axes[idx, 3].set_title('Latent (first 64 dims)', fontsize=14, weight='bold')
    axes[idx, 3].set_ylim(-3, 3)
    axes[idx, 3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('autoencoder_reconstructions.png', dpi=150, bbox_inches='tight')
print("Reconstructions saved to 'autoencoder_reconstructions.png'")
plt.show()

print("=" * 70)

# %%
# reusing the autoencoder's ENCODER for classification

class EncoderClassifier(nn.Module):
    """
    Classifier built on top of pretrained encoder.
    
    Architecture:
    1. Pretrained Encoder (frozen or fine-tuned)
    2. New classification head
    
    This demonstrates a new concept - transfering the knowledge learnged
    it is called - "transfer learning".
    We take a model trained on one task (autoencoding) and reuse its learned features
    for a different task (classification).

    This is a very powerful technique, especially when we have limited labeled data
    This way we can reuse the "knowledge" learned from a larger dataset or different task.
    And, especially important in the current landscape, we do not waste compute power and time
    (as of 11.2025 the price of compute and components is skyrocketing again, this time ALL the storage/RAM companies are jumping on the
    AI bandwagon, which is understandable - much higher marings).


    """
    def __init__(self, pretrained_encoder, num_classes=9, freeze_encoder=False):
        super().__init__()
        
        self.encoder = pretrained_encoder
        
        # Optionally freeze encoder weights
        # we will see what this means later in the demo and get back to this part
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder weights FROZEN - only training classifier head")
        else:
            print("Encoder weights TRAINABLE - fine-tuning entire model")
        
        # New classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Extract features using pretrained encoder
        latent = self.encoder(x)
        # Classify using new head
        logits = self.classifier(latent)
        return logits


class EncoderClassifierLightning(LightningModule):
    """
    Lightning module for encoder-based classifier.
    """
    def __init__(self, pretrained_encoder, num_classes=9, learning_rate=1e-3, freeze_encoder=False):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_encoder'])
        
        self.model = EncoderClassifier(
            pretrained_encoder=pretrained_encoder,
            num_classes=num_classes,
            freeze_encoder=freeze_encoder
        )
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }


# %%
# Extract pretrained encoder from autoencoder
print("\nExtracting pretrained encoder...")
pretrained_encoder = ae_lightning_model.model.encoder

# Create classifier with pretrained encoder
encoder_classifier = EncoderClassifierLightning(
    pretrained_encoder=pretrained_encoder,
    num_classes=9,
    learning_rate=1e-3,
    freeze_encoder=True  # Freeze encoder - only train classifier head
)

# Count parameters
trainable_params = sum(p.numel() for p in encoder_classifier.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in encoder_classifier.parameters())

print(f"\nEncoder-based Classifier:")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# %%
# Train the encoder-based classifier
print("training the encoder-based classifier...")
# Create callbacks
classifier_early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=True,
    mode='min'
)

# Create trainer
classifier_trainer = Trainer(
    max_epochs=1,  # Increase for real training
    callbacks=[classifier_early_stop],
    accelerator='auto',
    devices=1,
    log_every_n_steps=10,
)

print("Training classifier with pretrained encoder features...")
print("This should converge faster than training from scratch!")


classifier_trainer.fit(encoder_classifier, ae_data_module)


# %%
# Test the encoder-based classifier
print("testing the encoder-based classifier...")

classifier_test_results = classifier_trainer.test(encoder_classifier, ae_data_module)

print("Test Results:")
print(f"Test Loss: {classifier_test_results[0]['test_loss']:.4f}")
print(f"Test Accuracy: {classifier_test_results[0]['test_acc']:.4f}")

# %%
# Visualize classifier predictions
print("visualizing encoder classifier predictions...")

# Get a batch from test set
test_loader = ae_data_module.test_dataloader()
images, labels = next(iter(test_loader))

# Make predictions
encoder_classifier.eval()
with torch.no_grad():
    outputs = encoder_classifier(images)
    _, predicted = torch.max(outputs, 1)

# Denormalize images for visualization
images_denorm = images * std + mean
images_denorm = torch.clamp(images_denorm, 0, 1)

# Plot predictions
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

class_names = list(final_classes.keys())

for idx in range(min(8, len(images))):
    img = images_denorm[idx].permute(1, 2, 0).cpu().numpy()
    axes[idx].imshow(img)
    axes[idx].axis('off')
    
    true_label = class_names[labels[idx]]
    pred_label = class_names[predicted[idx]]
    color = 'green' if labels[idx] == predicted[idx] else 'red'
    
    axes[idx].set_title(f'True: {true_label}\nPred: {pred_label}', 
                        color=color, fontsize=10)

plt.tight_layout()
plt.savefig('encoder_classifier_predictions.png', dpi=150, bbox_inches='tight')
print("Predictions saved to 'encoder_classifier_predictions.png'")
plt.show()

# %%


# TODO task:
'''
This one is for you to try now:

implement a ResNET - like architecture for image classification
We will start together, by analyzing the ResNET paper and understanding the architecture

Then you will implement it yourself, train it on the waste dataset and see how it performs compared to our previous architectures
Warning - this is a more complex architecture, so it will take more time to train and tune
Your task is to mostly concentrate on proper immplementation of the architecture and training process, 
so take care to properly understand the main concept - residual (skip) connections
You can implement a smaller version of ResNET (e.g., ResNET-18) to keep training time reasonable

The source of knowledge: 
The original ResNET paper:
https://arxiv.org/pdf/1512.03385


As a bonus please look at the ResNext architecture as well:
https://arxiv.org/pdf/1611.05431

How does it relate to the original ResNET architecture? And what things from this lab are useful in implementing it?


For a bonus, bonus task, please read two topics:
- Depth-wise Separable Convolution
- EfficientNet architecture


'''