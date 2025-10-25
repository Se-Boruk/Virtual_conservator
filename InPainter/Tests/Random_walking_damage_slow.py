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
import math
import random
from typing import Optional, Tuple, Dict

import torch

# simple cache for disk kernels (stored on CPU)
__disk_kernel_cache: Dict[int, torch.Tensor] = {}

def _get_disk_kernel(radius: int, device: torch.device):
    """
    Returns a boolean disk kernel of integer radius 'radius' (radius >= 0),
    shape = (2*radius+1, 2*radius+1). Caches kernels on CPU and moves to device on request.
    """
    if radius < 0:
        radius = 0
    key = radius
    if key in __disk_kernel_cache:
        return __disk_kernel_cache[key].to(device).bool()
    # build on CPU then cache (small).
    r = radius
    diam = 2 * r + 1
    yy = torch.arange(0, diam, dtype=torch.float32).unsqueeze(1) - r
    xx = torch.arange(0, diam, dtype=torch.float32).unsqueeze(0) - r
    dist2 = xx * xx + yy * yy
    disk = dist2 <= (radius + 0.0) ** 2
    __disk_kernel_cache[key] = disk.bool()  # store bool on CPU
    return disk.to(device).bool()

def random_walk_brush_masks_v2(
    imgs: torch.Tensor,
    mask_fraction: float = 0.20,
    brush_range: Tuple[float, float] = (3.0, 20.0),
    steps_range: Tuple[int, int] = (8, 120),
    strokes_range: Tuple[int, int] = (6, 200),
    step_length_range: Tuple[float, float] = (1.0, 12.0),
    angle_jitter_range: Tuple[float, float] = (math.pi / 8.0, math.pi / 2.0),
    width_jitter_range: Tuple[float, float] = (0.1, 0.4),
    dir_momentum_range: Tuple[float, float] = (0.1, 1.0),
    width_momentum_range: Tuple[float, float] = (0.1, 1.0),
    max_attempts: int = 2000,
    random_seed: Optional[int] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Fixed & robust version: returns masks (N,1,H,W) bool on same device as imgs or provided device.
    """
    if device is None:
        device = imgs.device
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    N, C, H, W = imgs.shape
    total_pixels = float(H * W)
    masks = torch.zeros((N, 1, H, W), dtype=torch.bool, device=device)

    def _rand_in_range(rng):
        if isinstance(rng, (tuple, list)) and len(rng) == 2:
            a, b = rng
            if isinstance(a, int) and isinstance(b, int):
                return random.randint(a, b)
            else:
                return random.uniform(a, b)
        else:
            return rng

    for i in range(N):
        mask_i = masks[i]  # 1,H,W bool
        masked_count = 0  # integer masked pixels

        # per-image randomized hyperparameters
        min_brush, max_brush = brush_range if isinstance(brush_range, (tuple, list)) else (brush_range, brush_range)
        min_brush = float(min_brush); max_brush = float(max_brush)
        min_steps, max_steps = steps_range
        min_strokes, max_strokes = strokes_range

        dir_momentum = _rand_in_range(dir_momentum_range)
        width_momentum = _rand_in_range(width_momentum_range)
        width_jitter = _rand_in_range(width_jitter_range)
        min_angle_jitter, max_angle_jitter = angle_jitter_range if isinstance(angle_jitter_range, (tuple, list)) else (angle_jitter_range, angle_jitter_range)

        target_strokes = random.randint(min_strokes, max_strokes)
        attempts = 0
        strokes_done = 0

        if mask_fraction <= 0:
            masks[i] = mask_i
            continue

        while (masked_count / total_pixels) < mask_fraction and attempts < max_attempts and strokes_done < target_strokes:
            attempts += 1
            strokes_done += 1

            # margin so initial center isn't immediately out of bounds
            margin = int(math.ceil(max_brush))
            if W - 1 - margin <= margin:
                cx = random.uniform(0, W - 1)
            else:
                cx = random.uniform(margin, W - 1 - margin)
            if H - 1 - margin <= margin:
                cy = random.uniform(0, H - 1)
            else:
                cy = random.uniform(margin, H - 1 - margin)

            radius = random.uniform(min_brush, max_brush)
            angle = random.uniform(0, 2 * math.pi)
            vx = math.cos(angle); vy = math.sin(angle)
            width_vel = 0.0

            steps = random.randint(min_steps, max_steps)
            angle_jitter = random.uniform(min_angle_jitter, max_angle_jitter)

            for s in range(steps):
                # integer radius & kernel
                r_int = int(math.ceil(radius))
                kernel = _get_disk_kernel(r_int, device=device)  # (2r+1,2r+1) bool

                kh, kw = kernel.shape
                # robust kernel-to-mask placement using center = round(cx), round(cy)
                center_x = int(round(cx))
                center_y = int(round(cy))
                x0 = center_x - r_int
                x1 = center_x + r_int + 1   # exclusive
                y0 = center_y - r_int
                y1 = center_y + r_int + 1   # exclusive

                # clamp to image bounds
                x0_clamped = max(0, x0)
                x1_clamped = min(W, x1)
                y0_clamped = max(0, y0)
                y1_clamped = min(H, y1)

                # compute kernel crop indices
                kx0 = x0_clamped - x0  # how many columns into kernel we start
                ky0 = y0_clamped - y0
                kx1 = kx0 + (x1_clamped - x0_clamped)
                ky1 = ky0 + (y1_clamped - y0_clamped)

                if (x1_clamped <= x0_clamped) or (y1_clamped <= y0_clamped):
                    # nothing to stamp (completely out of bounds)
                    pass
                else:
                    kernel_crop = kernel[ky0:ky1, kx0:kx1]
                    mask_slice = mask_i[0, y0_clamped:y1_clamped, x0_clamped:x1_clamped]
                    # safety check: shapes must match
                    if kernel_crop.shape != mask_slice.shape:
                        # if mismatch occurs (shouldn't), skip this stamp
                        # (defensive; avoids runtime error)
                        # You could also align by min dims, but mismatch indicates math bug.
                        pass
                    else:
                        # newly added pixels
                        new_pixels = kernel_crop & (~mask_slice)
                        if new_pixels.any():
                            # update mask slice in-place
                            mask_slice |= kernel_crop
                            added = int(new_pixels.sum().item())
                            masked_count += added

                if (masked_count / total_pixels) >= mask_fraction:
                    break

                # walk update
                da = random.uniform(-angle_jitter, angle_jitter)
                prev_angle = math.atan2(vy, vx)
                new_angle = prev_angle * dir_momentum + (prev_angle + da) * (1.0 - dir_momentum)
                vx = math.cos(new_angle); vy = math.sin(new_angle)

                step_len = random.uniform(step_length_range[0], step_length_range[1])
                cx = float(min(max(0.0, cx + vx * step_len), W - 1))
                cy = float(min(max(0.0, cy + vy * step_len), H - 1))

                dwidth = random.uniform(-width_jitter, width_jitter) * radius
                width_vel = width_vel * width_momentum + dwidth * (1.0 - width_momentum)
                radius = float(max(1.0, min(max_brush, radius + width_vel)))

            # end stroke

        # attempts exhausted fallback: scatter small disks
        if (masked_count / total_pixels) < mask_fraction and attempts >= max_attempts:
            while (masked_count / total_pixels) < mask_fraction:
                cx = random.uniform(0, W - 1)
                cy = random.uniform(0, H - 1)
                r = random.uniform(max(1.0, min_brush * 0.4), min(max_brush, min_brush * 1.2))
                r_int = int(math.ceil(r))
                kernel = _get_disk_kernel(r_int, device=device)
                kh, kw = kernel.shape

                center_x = int(round(cx))
                center_y = int(round(cy))
                x0 = center_x - r_int
                x1 = center_x + r_int + 1
                y0 = center_y - r_int
                y1 = center_y + r_int + 1

                x0_clamped = max(0, x0)
                x1_clamped = min(W, x1)
                y0_clamped = max(0, y0)
                y1_clamped = min(H, y1)

                kx0 = x0_clamped - x0
                ky0 = y0_clamped - y0
                kx1 = kx0 + (x1_clamped - x0_clamped)
                ky1 = ky0 + (y1_clamped - y0_clamped)

                if (x1_clamped <= x0_clamped) or (y1_clamped <= y0_clamped):
                    pass
                else:
                    kernel_crop = kernel[ky0:ky1, kx0:kx1]
                    mask_slice = mask_i[0, y0_clamped:y1_clamped, x0_clamped:x1_clamped]
                    if kernel_crop.shape == mask_slice.shape:
                        new_pixels = kernel_crop & (~mask_slice)
                        if new_pixels.any():
                            mask_slice |= kernel_crop
                            masked_count += int(new_pixels.sum().item())

                if (masked_count / total_pixels) >= mask_fraction:
                    break

        masks[i] = mask_i

    return masks







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

t0 = time.time()
masks = random_walk_brush_masks_v2(batch, mask_fraction=0.1, brush_range=(4,12),
                                   steps_range=(12,50), strokes_range=(8,120) )


t1 = time.time()

print(f"Time: {t1 - t0:.3f} seconds")

# visualize
import matplotlib.pyplot as plt

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(masks[i,0].cpu(), cmap = "gray")













