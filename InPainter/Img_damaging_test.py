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


##########################################################3

################################################################



epochs = 10
bs = 32
n_workers = 4
max_queue = 10

# Training loader
train_loader = Async_DataLoader(dataset = Train_set,
                                batch_size=bs,
                                num_workers=n_workers,
                                device='cuda',
                                max_queue=max_queue
                                )

for e in range(epochs):
    #################################################
    #Training part
    #################################################
    train_loader.start_epoch(shuffle=True)
    num_batches = train_loader.get_num_batches()
    batch_train_times = []
    
    
    #Running single batch to just perform damage on it
    while True:
        t0 = time.time()
        #Load batch from loader
        batch = train_loader.get_batch()
        if batch is None:
            break
        
        #Batch breaking:
        
        
        
        
        break


##########
import numpy as np
from scipy.ndimage import gaussian_filter

import torch
import torch.nn.functional as F

def torch_splatter_mask(shape, coverage=0.2, blur_sigma=4.0, device="cpu"):
    """
    Fast splatter mask generator using PyTorch.
    shape: (B, H, W)
    """
    B, H, W = shape
    x = torch.rand((B, 1, H, W), device=device)
    
    # Gaussian blur kernel (depthwise)
    ksize = int(blur_sigma * 4 + 1)
    coords = torch.arange(ksize, device=device) - ksize // 2
    g = torch.exp(-coords**2 / (2 * blur_sigma**2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    
    x = F.conv2d(x, kernel, padding=ksize//2)
    
    # threshold to keep top N%
    flat = x.view(B, -1)
    thresh = torch.quantile(flat, 1 - coverage, dim=1, keepdim=True)
    mask = (flat > thresh).float().view(B, H, W)
    return mask



def fast_brush_stroke_mask(
    shape,
    coverage=0.2,
    stroke_length=31,
    stroke_width=3,
    strokes_per_image=None,
    irregularity=1.5,      # lateral jitter STD in pixels
    longitudinal_wobble=0.1, # fraction of length used to modulate per-sample offset
    device='cpu',
    seed=None,
):
    """
    Fast vectorized irregular brush strokes:
      - lateral jitter per sample along each stroke
      - variable stroke thickness envelope along length
    Args:
      shape: (B,H,W)
      coverage: desired final coverage (enforced by threshold)
      stroke_length: samples per stroke
      stroke_width: nominal thickness (controls blur sigma)
      strokes_per_image: auto-estimated if None
      irregularity: STD in pixels of lateral jitter
      longitudinal_wobble: multiplies sin/cos wobble along stroke
    Returns:
      binary mask (B,H,W) float 0/1
    """
    if seed is not None:
        torch.manual_seed(seed)
    B, H, W = shape
    device = torch.device(device)

    if strokes_per_image is None:
        est = max(1, int((coverage * H * W) / max(1, stroke_length * stroke_width)))
        strokes_per_image = min(max(1, est), 2000)
    N = strokes_per_image
    L = stroke_length

    # stroke centers & angles
    y0 = torch.randint(0, H, (B, N), device=device, dtype=torch.long).to(torch.float32)  # (B,N)
    x0 = torch.randint(0, W, (B, N), device=device, dtype=torch.long).to(torch.float32)
    angles = torch.rand((B, N), device=device) * 2 * torch.pi  # (B,N)

    # param t along stroke in [-0.5, 0.5] scaled by length
    t = torch.linspace(-0.5, 0.5, steps=L, device=device).view(1, 1, L)  # (1,1,L)
    # direction components
    sin_a = torch.sin(angles).unsqueeze(-1)  # (B,N,1)
    cos_a = torch.cos(angles).unsqueeze(-1)  # (B,N,1)

    # main positions along the straight stroke (without jitter)
    # scale by stroke_length to get pixel units
    length_pixels = float(max(1.0, stroke_length))
    y_main = y0.unsqueeze(-1) + sin_a * (t * length_pixels * (1.0 + longitudinal_wobble))
    x_main = x0.unsqueeze(-1) + cos_a * (t * length_pixels * (1.0 + longitudinal_wobble))

    # compute perpendicular unit vectors for lateral jitter
    perp_y = -cos_a  # rotate 90 degrees
    perp_x = sin_a

    # lateral jitter per sample, per stroke: normal(0, irregularity)
    jitter = torch.randn((B, N, L), device=device) * float(irregularity)

    # optional longitudinal modulation so jitter varies along the stroke (e.g., bigger near middle)
    envelope = (1.0 - (t.abs() * 2.0))  # triangular envelope in [0..1], center=1
    envelope = envelope.clamp(min=0.0)
    jitter = jitter * envelope  # (B,N,L)

    # add jitter along perpendicular direction
    y_pos = y_main + perp_y * jitter
    x_pos = x_main + perp_x * jitter

    # also add small random radial perturbation for more organic shape
    radial = (torch.randn_like(y_pos) * (irregularity * 0.3))
    ang_rand = torch.rand_like(y_pos) * 2 * torch.pi
    y_pos = y_pos + radial * torch.sin(ang_rand)
    x_pos = x_pos + radial * torch.cos(ang_rand)

    # round/clamp to pixel indices
    y_idx = torch.clamp(torch.round(y_pos).to(torch.long), 0, H - 1)  # (B,N,L)
    x_idx = torch.clamp(torch.round(x_pos).to(torch.long), 0, W - 1)  # (B,N,L)

    # linearize indices and scatter_add
    idxs = (y_idx * W + x_idx).view(B, -1)  # (B, N*L)
    src = torch.ones_like(idxs, dtype=torch.float32, device=device)
    mask_flat = torch.zeros((B, H * W), dtype=torch.float32, device=device)
    mask_flat = mask_flat.scatter_add_(1, idxs, src)
    mask = mask_flat.view(B, 1, H, W)

    # broaden strokes with separable Gaussian blur; make sigma proportional to stroke_width and per-sample envelope
    sigma = max(0.5, float(stroke_width) / 2.0)
    ksize = int(max(3, 2 * int(3 * sigma) + 1))
    coords = torch.arange(ksize, device=device, dtype=torch.float32) - (ksize - 1) / 2
    g1 = torch.exp(-(coords ** 2) / (2 * (sigma ** 2)))
    g1 = g1 / g1.sum()
    kx = g1.view(1, 1, 1, ksize)
    ky = g1.view(1, 1, ksize, 1)
    mask = F.pad(mask, (ksize // 2, ksize // 2, 0, 0), mode='reflect')
    mask = F.conv2d(mask, kx)
    mask = F.pad(mask, (0, 0, ky.size(-2) // 2, ky.size(-2) // 2), mode='reflect')
    mask = F.conv2d(mask, ky)

    # create binary mask by thresholding to reach target coverage
    flat = mask.view(B, -1)
    coverage = float(min(max(coverage, 1e-8), 1.0 - 1e-8))
    thresh = torch.quantile(flat, 1.0 - coverage, dim=1, keepdim=True)
    binary = (flat > thresh).float().view(B, H, W)
    return binary


import random

t0 = time.time()
p = random.randint(0,4)

if p==0:
    masks_1 = torch_splatter_mask((32, 256, 256), coverage=0.1, blur_sigma=8, device = 'cuda')
    masks_2 = fast_brush_stroke_mask((32, 256, 256), coverage=0.1, device = 'cuda')
    masks = masks_1 * masks_2
    
elif p == 1:
    masks_1 = torch_splatter_mask((32, 256, 256), coverage=0.1, blur_sigma=8, device = 'cuda')
    masks_2 = fast_brush_stroke_mask((32, 256, 256), coverage=0.1, device = 'cuda')
    masks = masks_1 + masks_2
    
elif p == 2:
    masks_1 = torch_splatter_mask((32, 256, 256), coverage=0.1, blur_sigma=8, device = 'cuda')
    masks_2 = fast_brush_stroke_mask((32, 256, 256), coverage=0.1, device = 'cuda')
    
    if random.choice([True, False]):
        masks = masks_1 - masks_2
    else:
        masks = masks_2 - masks_1
        
    
elif p == 3:
    masks = torch_splatter_mask((32, 256, 256), coverage=0.1, blur_sigma=8, device = 'cuda')
    
elif p == 4:
    masks = fast_brush_stroke_mask((32, 256, 256), coverage=0.1, device = 'cuda')

masks = masks.clamp(0.0, 1.0)

t1 = time.time()

print(f"Time: {t1 - t0:.3f} seconds")
# visualize
import matplotlib.pyplot as plt

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(masks[i].cpu(), cmap = "gray")










