from datasets import load_dataset, load_from_disk
import os
import numpy as np
import torch
from queue import Queue
import threading
import random
import math
import torch.nn.functional as F


class Custom_DataSet_Manager():
    
    #Checks if there is dataset folder present, if not it creates it
    def __init__(self, DataSet_path, train_split, val_split, test_split, random_state):
        self.dataset_path = DataSet_path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        
        self.flag_file = os.path.join(self.dataset_path, "download_complete.flag")
        
        
        
    def download_database(self, dataset_name):
        
        if not self.is_downloaded():
            #Create folder if not present 
            os.makedirs(self.dataset_path, exist_ok=True)
    
            print("Downloading dataset...")
            dataset = load_dataset(dataset_name)
            dataset.save_to_disk(self.dataset_path)
            
            #Add flag but only if the dataset is completed. 
            #If downloading above is interrupted then its not present
            with open(self.flag_file, "w") as f:
                f.write("downloaded")
    
            print("Dataset downloaded and flagged!")
        
        else:
            print("Dataset is alredy downloaded!")
        
    def is_downloaded(self):
        # Check if the flag file exists
        return os.path.exists(self.flag_file)     
    
    def load_dataset_from_disk(self):
        #Check for flag
        if not self.is_downloaded():
            raise RuntimeError("Dataset not downloaded or incomplete. Download it first")
            
        #Load it to split it on run
        Dataset = load_from_disk(self.dataset_path)
        
        train, val, test = self.split_dataset(Dataset)
        return train, val, test
    
    def split_dataset(self,dataset):
        #Split dataset into train, val and test. Ready for work :). 
        #OFc with given random state or diseaster 
        
        #Train bc this dataset (at least the one for reconstruction - upscaling one may 
        #be different and require changes/addons) has only train. 
        #We need to split it on our own
        
        #Just load the data and shuffle it (so we mix the classes and hopefully mix them uniformly for training)
        #Cant use stratifying as we do not know the classes a priori (unsupervised learning)
        Data =  dataset["train"].shuffle(seed=self.random_state)
        
        #Split it into train and subset
        split_dataset = Data.train_test_split(test_size= (1 -self.train_split) , seed=self.random_state)
        
        train_subset = split_dataset['train']
        subset = split_dataset['test']
        
        #Split the subset into the val and test 
        test_fraction = self.val_split / ((self.val_split + self.test_split))
        
        split_dataset_1 = subset.train_test_split(test_size= test_fraction , seed=self.random_state)
        
        val_subset = split_dataset_1['train']
        test_subset = split_dataset_1['test']

        return train_subset, val_subset, test_subset
        
    
##########################################################################

        
def Reconstruction_data_tests(train_subset, val_subset, test_subset):
        #############################################
        #Add some tests for given random_state = 111 (with prints) - again, valid only for reconstruction dataset as for now:
    
        print("Running datasaet tests...")
        #Train
        #From split we want to always replicate 
        n1 = "37806.jpg"
        n2 = "88027.jpg" 
        
        #From split
        name_1 = train_subset[0]['filename']
        name_2 = train_subset[1]['filename']
        
        #print(name_1)
        #print(name_2)
        
        assert n1 == name_1, f"Names do not match! Dataset is different than it should be by Config constants '{n1}' != '{name_1}'"
        assert n2 == name_2, f"Names do not match! Dataset is different than it should be by Config constants '{n2}' != '{name_2}'"
        
        print("Train passed")
        ###########
        #Val
        #From split we want to always replicate 
        n1 = "63339.jpg"
        n2 = "86573.jpg" 
        
        #From split
        name_1 = val_subset[0]['filename']
        name_2 = val_subset[1]['filename']
        
        #print(name_1)
        #print(name_2)
        
        assert n1 == name_1, f"Names do not match! Dataset is different than it should be by Config constants '{n1}' != '{name_1}'"
        assert n2 == name_2, f"Names do not match! Dataset is different than it should be by Config constants '{n2}' != '{name_2}'"
        
        print("Val passed")
        ###########
        #Test
        #From split we want to always replicate 
        n1 = "19946.jpg"
        n2 = "33194.jpg" 
        
        #From split
        name_1 = test_subset[0]['filename']
        name_2 = test_subset[1]['filename']
        
        #print(name_1)
        #print(name_2)
        
        assert n1 == name_1, f"Names do not match! Dataset is different than it should be by Config constants '{n1}' != '{name_1}'"
        assert n2 == name_2, f"Names do not match! Dataset is different than it should be by Config constants '{n2}' != '{name_2}'"
        
        print("Test passed")
    
    
##########################################################################    
    
    
class Async_DataLoader():
    def __init__(self, dataset, batch_size=32, num_workers=2, device='cuda', max_queue=10, add_damaged = True, add_augmented = True, label_map = None, fraction = None):
        self.dataset = dataset
        #Taking sample of from dataset to initialize the shape of images
        sample_img = np.array(dataset[0]["image"], dtype=np.uint8)
        self.C, self.H, self.W = sample_img.shape[2], sample_img.shape[0], sample_img.shape[1]
        
        self.fraction = fraction # Fraction of images taken in the epoch (randomized each epoch)
        
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.queue = Queue(maxsize=max_queue)
        self.num_workers = num_workers

        #Epoch control
        self.next_idx = 0               #Next step (batch) idx
        self.idx_lock = threading.Lock()
        self.active_workers = 0 
        self.threads = []
        self.epoch_event = threading.Event()
        self.indices = list(range(len(self.dataset)))
        
        #Label maps for the pseudomaps in testing !!IMPORTANT!! - They are arbitraly class which after clusterizer is trained wont be useful. Then only the classes from clusterizer must be used!!!!
        self.label_map = label_map

        #Preallocate pinned buffers
        self.pinned_bufs = [torch.empty((self.batch_size, self.C, self.H, self.W), 
                                        dtype=torch.float32).pin_memory() 
                            for _ in range(num_workers)]
        
        self.labels_bufs = [torch.empty((self.batch_size,1), 
                                        dtype=torch.float32).pin_memory() 
                            for _ in range(num_workers)]
        
        self.add_damaged = add_damaged
        self.add_augmented = add_augmented
        
        
        #create per-worker damage generators in main thread
        if self.add_damaged:
            # don't move device here unless you want the generator to allocate on cuda on init
            # If Random_Damage_Generator is lightweight on construction, this is fine.
            self.dmg_generators = [Random_Damage_Generator(device=self.device) for _ in range(self.num_workers)]
        else:
            self.dmg_generators = [None] * self.num_workers
        
        # threads will be started lazily in start_epoch (safer on Windows/spawn)
        self.threads_started = False
        # do not start prefetch here
        # self._start_prefetch()

    def _start_prefetch(self):
        
        def get_chunk():
            
            """
            Functions for getting start idx and end idx of batch 
            (for indexes list as we shuffle)
            """
        
            #Lock function so we only can acces it from one thread (worker)
            #It assures that we cannot have the same batch operated twice
            with self.idx_lock:
                start = self.next_idx
                
                if start >= len(self.dataset):
                    return None, None
                
                end = min(start + self.batch_size, len(self.dataset))
                self.next_idx = end
                
                return start, end


        def worker(worker_id):
            dmg_generator = self.dmg_generators[worker_id] if self.add_damaged else None
            pinned_buf = self.pinned_bufs[worker_id]
            labels_bufs = self.labels_bufs[worker_id]
        
            while True:
                self.epoch_event.wait()  # wait for epoch start
        
                while True:
                    start, end = get_chunk()
                    if start is None:
                        break
                    actual_bs = end - start
        
                    # ---------------------------
                    # Load original batch into pinned memory
                    # ---------------------------
                    for i in range(actual_bs):
                        idx = self.indices[start + i]
                        img = np.array(self.dataset[idx]["image"], dtype=np.float32) / 255.0
                        style = self.dataset[idx]['style']
                        label = self.label_map[style]
        
                        pinned_buf[i].copy_(torch.from_numpy(img).permute(2, 0, 1))
                        labels_bufs[i].fill_(label)
        
                    # Clone to avoid modifying pinned memory
                    original_batch = pinned_buf[:actual_bs].to(self.device, non_blocking=True).clone()
                    original_labels = labels_bufs[:actual_bs].to(self.device, non_blocking=True).clone()
        
                    # ---------------------------
                    # Prepare batch variants
                    # ---------------------------
                    batch_dict = {
                        "original": original_batch,
                        "labels": original_labels
                    }
        
                    # Original damaged
                    if self.add_damaged:
                        damage_masks, _ = dmg_generator.generate(shape=(actual_bs, self.H, self.W))
                        original_damaged = original_batch * (1.0 - damage_masks.unsqueeze(1))
                        batch_dict["original_damaged"] = original_damaged
                    
                    # Augmented
                    if self.add_augmented:
                        aug_input = original_batch.clone()
                        aug_mask_input = damage_masks.clone() if self.add_damaged else torch.zeros((actual_bs, 1, self.H, self.W), device=self.device)
        
                        aug_images, aug_masks = augment_image_and_mask(aug_input, aug_mask_input)
                        batch_dict["augmented"] = aug_images
                        
        
                        # Augmented + damaged
                        if self.add_damaged:
                            aug_damaged_images = aug_images * (1.0 - aug_masks)
                            batch_dict["augmented_damaged"] = aug_damaged_images

        
                    # Push to queue
                    self.queue.put(batch_dict)
        
                # Epoch end handling
                with self.idx_lock:
                    self.active_workers -= 1
                    if self.active_workers == 0:
                        self.queue.put(None)
                        self.epoch_event.clear()

        # start worker threads
        for wid in range(self.num_workers):
            t = threading.Thread(target=worker, args=(wid,))
            t.daemon = True
            t.start()
            self.threads.append(t)

    def start_epoch(self, shuffle=True):
        
        """Start a new epoch. It resets queue and shuffle data."""
        
        self.queue = Queue(maxsize=self.queue.maxsize)
        self.next_idx = 0
        self.active_workers = self.num_workers
        
        #Create worker threads only once
        if not self.threads_started:
            self._start_prefetch()
            self.threads_started = True
            
        #Shuffle and optionally reduce dataset fraction
        indices = np.arange(len(self.dataset))
        if shuffle:
            np.random.shuffle(indices)

        if self.fraction is not None and 0 < self.fraction < 1:
            reduced_size = int(len(indices) * self.fraction)
            indices = indices[:reduced_size]  # take only fraction of dataset
        
        self.indices = list(indices)
            
        self.epoch_event.set() #It allows workers to start
        
        


    def get_batch(self):
        batch = self.queue.get()
        if batch is None:
            return None
    
        # Move all tensors in dict to device (non-blocking)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        return batch

    def get_num_batches(self):
        """
        Get number of batches (steps) for the current epoch, 
        taking into account fraction of dataset if specified.
        """
        # Determine effective dataset length for this epoch
        if self.fraction is not None and 0 < self.fraction < 1:
            effective_len = int(len(self.dataset) * self.fraction)
        else:
            effective_len = len(self.dataset)

        steps = (effective_len + self.batch_size - 1) // self.batch_size
        return steps

    def get_random_batch(self, batch_size=None, shuffle=True, random_state=None):
        """
        Returns a random batch of original images from the dataset, reproducible if rng is provided.
        
        Args:
            batch_size: int, optional
            shuffle: bool, whether to pick random indices
            rng: np.random.Generator, optional — if given, sampling becomes deterministic
    
        Returns:
            batch tensor of shape (B, C, H, W)
        """
        bs = batch_size or self.batch_size
    
        #Random state for reproducibility
        if random_state is None:
            random_state = np.random.default_rng()
        else:
            random_state = np.random.default_rng(random_state)

        
        # pick indices
        if shuffle:
            indices = random_state.choice(len(self.dataset), bs, replace=False)
        else:
            indices = np.arange(bs)
    
        #preallocate pinned buffer
        pinned_buf = torch.empty((bs, self.C, self.H, self.W),
                                 dtype=torch.float32).pin_memory()
    
        #load images
        for i, idx in enumerate(indices):
            img = np.array(self.dataset[idx]["image"], dtype=np.float32) / 255.0
            pinned_buf[i] = torch.from_numpy(img).permute(2, 0, 1)
    
        #move to device
        batch = pinned_buf.to(self.device, non_blocking=True)
        
        return batch
    
    
##########################################################################

class Random_Damage_Generator:
    """
    Randomized mask generator for image augmentation.
    Produces masks by combining splatter, brush strokes, 
    and selects part of the damage for final damage mask
    """

    def __init__(self, device = 'cuda'):
        self.device = torch.device(device)
        
    
    #Function just for transforming the range into random value
    @staticmethod
    def _sample_param(p, dtype=float):
        """
        Sample either scalar or uniform in [a,b].
        """
        if isinstance(p, (list, tuple)) and len(p) == 2:
            a, b = p
            if dtype is int:
                return int(torch.randint(int(a), int(b)+1, (1,)).item())
            else:
                return float(torch.rand(1).item()*(b-a) + a)
        else:
            return int(p) if dtype is int else float(p)


    def _splatter_mask(self, shape, coverage=0.1, blur_sigma=8.0):
        """
        shape: shape of input tensor
        coverage: fraction of pixels to be set as damage (mask)
        blur_sigma: standard deviation of Gaussian blur applied to random noise
        """
    
        B,H,W = shape
        x = torch.rand((B,1,H,W), device=self.device)  #Generate uniform random noise for each batch
        sigma = max(0.5, float(blur_sigma))  #Ensure blur sigma is at least 0.5
        ksize = int(max(3, 2*int(3*sigma)+1))  #Compute kernel size, must be odd and >= 3
    
        #Gaussian filtering here effectively
        
        # Generate a 1D Gaussian kernel
        coords = torch.arange(ksize, device=self.device, dtype=torch.float32) - (ksize-1)/2
        g = torch.exp(-(coords**2)/(2*sigma*sigma))  #Gaussian formula
        g = g / g.sum()  #Normalize kernel to sum of 1
        kx = g.view(1,1,1,ksize)  #Horizontal kernel for conv2d
        ky = g.view(1,1,ksize,1)  #Vertical kernel for conv2d
    
        #Apply horizontal Gaussian blur
        x = F.pad(x, (ksize//2, ksize//2, 0, 0), mode='reflect')  #Pad height dimension so we end in the same size
        x = F.conv2d(x, kx)  # Convolve horizontally
    
        #Apply vertical Gaussian blur
        x = F.pad(x, (0,0,ksize//2,ksize//2), mode='reflect')  #Pad height dimension so we end in the same size
        x = F.conv2d(x, ky)  # Convolve vertically
    
        # Flatten spatial dimensions for thresholding
        flat = x.view(B,-1)
        thresh = torch.quantile(flat, 1.0 - float(coverage), dim=1, keepdim=True)  #Compute cutoff to retain 'coverage' fraction
        mask = (flat > thresh).float().view(B,H,W)  # Create binary mask based on threshold
    
        return mask  


    def _brush_mask(self, shape, coverage=0.1, stroke_length=31, stroke_width=3, strokes_per_image=None):
        """
        shape: tuple (B, H, W) where B=batch size, H=height, W=width
        coverage: fraction of pixels to be masked (damaged)
        stroke_length: number of pixels along each brush stroke
        stroke_width: approximate thickness of brush strokes
        strokes_per_image: optional override for number of strokes per image
        """
        
        B,H,W = shape  #Unpack shape
        
        # Determine number of strokes per image if not provided
        if strokes_per_image is None:
            strokes_per_image = max(
                8,  # minimum 8 strokes per image
                int((coverage * H * W) // (stroke_length * max(1, stroke_width)))  # scale with desired coverage
            )
            
        base = strokes_per_image
        jitter_max = max(0, int(base*0.25))  #extra randomness in stroke number
        
        N = base + jitter_max  #Total strokes per image
    
        #Random starting points for strokes
        y0 = torch.randint(0, H, (B,N), device=self.device).float()
        x0 = torch.randint(0, W, (B,N), device=self.device).float()
        
        #Random angles for strokes in o to 2pi
        angles = torch.rand((B,N), device=self.device) * 2 * math.pi
    
        L = int(stroke_length)
        t = torch.linspace(-0.5, 0.5, L, device=self.device).view(1,1,L)  #parametrize stroke
    
        #Compute stroke positions along the angle
        sin_a = torch.sin(angles).unsqueeze(-1)
        cos_a = torch.cos(angles).unsqueeze(-1)
        y_pos = y0.unsqueeze(-1) + sin_a*(t*L)
        x_pos = x0.unsqueeze(-1) + cos_a*(t*L)
    
        #Add Gaussian jitter for randomness in stroke path
        jitter = torch.randn_like(y_pos) * 0.8
        y_pos = y_pos + jitter
        x_pos = x_pos + jitter
    
        #Convert to integer pixel indices, clamped to image bounds
        y_idx = torch.clamp(torch.round(y_pos).to(torch.long), 0, H-1)
        x_idx = torch.clamp(torch.round(x_pos).to(torch.long), 0, W-1)
    
        #Flatten 2D indices into 1D for scatter operation
        idxs = (y_idx * W + x_idx).view(B,-1)
    
        #Initialize flat mask and set pixels along strokes
        mask_flat = torch.zeros((B,H*W), device=self.device)
        src = torch.ones_like(idxs, dtype=torch.float32, device=self.device)
        mask_flat = mask_flat.scatter_add_(1, idxs, src)  # add ones at stroke positions
    
        #Reshape back to image
        mask = mask_flat.view(B,1,H,W)
    
        #Apply Gaussian blur to simulate stroke thickness
        sigma = max(0.5, float(stroke_width)/2.0)
        ksize = int(max(3, 2*int(3*sigma)+1))
        coords = torch.arange(ksize, device=self.device, dtype=torch.float32) - (ksize-1)/2
        g = torch.exp(-(coords**2)/(2*sigma*sigma))
        g = g / g.sum()
        kx = g.view(1,1,1,ksize)  # horizontal kernel
        ky = g.view(1,1,ksize,1)  # vertical kernel
    
        # Horizontal blur
        mask = F.pad(mask, (ksize//2, ksize//2,0,0), mode='reflect')
        mask = F.conv2d(mask, kx)
    
        # Vertical blur
        mask = F.pad(mask, (0,0,ky.size(-2)//2, ky.size(-2)//2), mode='reflect')
        mask = F.conv2d(mask, ky)
    
        # Threshold to enforce given coverage of mask
        flat = mask.view(B,-1)
        thresh = torch.quantile(flat, 1.0 - float(coverage), dim=1, keepdim=True)
        
        return (flat > thresh).float().view(B,H,W)  # final binary mask


    def generate(self,shape=(32,256,256),
                      cover_s=(0.1, 0.2),
                      cover_b=(0.1, 0.2),
                      stroke_length=(31,81),
                      stroke_width=(2,8),
                      stroke_spread_jitter=0.2,
                      apply_lasso_prob=0.5,
                      lasso_frac=(0.2,0.7),
                      blur=(5,13),
                      lasso_blur=8
                      ):
        
        """
        Generates masks combining splatter, brush, and optional lasso.

        """
        B,H,W = shape

        # Sample batch-level parameters
        cover_s_s = self._sample_param(cover_s)
        cover_b_s = self._sample_param(cover_b)
        stroke_length_s = self._sample_param(stroke_length, dtype=int)
        stroke_width_s = self._sample_param(stroke_width, dtype=int)
        blur_size = self._sample_param(blur, dtype=int)

        # Random combination of operations
        ops = ['mul','add','sub_l1','sub_l2']
        op = random.choice(ops)

        # Generate masks

        strokes_est = max(8, int((cover_b_s*H*W)//(stroke_length_s*max(1,stroke_width_s))))
        strokes_est = max(1, int(strokes_est*(1+random.uniform(-stroke_spread_jitter, stroke_spread_jitter))))
        mask_br = self._brush_mask(shape, coverage=cover_b_s, stroke_length=stroke_length_s,
                                   stroke_width=stroke_width_s, strokes_per_image=strokes_est)

        
        #Combine masks in some combination
        if op=='mul':
            mask_sp = self._splatter_mask(shape, coverage=cover_s_s, blur_sigma=max(1,int(0.5*blur_size)))
            
            strokes_est = max(8, int((cover_b_s*H*W)//(stroke_length_s*max(1,stroke_width_s))))
            strokes_est = max(1, int(strokes_est*(1+random.uniform(-stroke_spread_jitter, stroke_spread_jitter))))
            mask_br = self._brush_mask(shape, coverage=cover_b_s, stroke_length=stroke_length_s,
                                       stroke_width=stroke_width_s, strokes_per_image=strokes_est)
            
            masks = mask_sp * mask_br
            
        elif op=='add':
            mask_sp = self._splatter_mask(shape, coverage=cover_s_s, blur_sigma=max(1,int(0.5*blur_size)))
            
            strokes_est = max(8, int((cover_b_s*H*W)//(stroke_length_s*max(1,stroke_width_s))))
            strokes_est = max(1, int(strokes_est*(1+random.uniform(-stroke_spread_jitter, stroke_spread_jitter))))
            mask_br = self._brush_mask(shape, coverage=cover_b_s, stroke_length=stroke_length_s,
                                       stroke_width=stroke_width_s, strokes_per_image=strokes_est)
            
            masks = (mask_sp + mask_br)
            
        elif op=='sub_l1':
            mask_sp = self._splatter_mask(shape, coverage=cover_s_s, blur_sigma=max(1,int(0.5*blur_size)))
            
            strokes_est = max(8, int((cover_b_s*H*W)//(stroke_length_s*max(1,stroke_width_s))))
            strokes_est = max(1, int(strokes_est*(1+random.uniform(-stroke_spread_jitter, stroke_spread_jitter))))
            mask_br = self._brush_mask(shape, coverage=cover_b_s, stroke_length=stroke_length_s,
                                       stroke_width=stroke_width_s, strokes_per_image=strokes_est)
            
            masks = (mask_sp - mask_br)
            
        elif op=='sub_l2':
            mask_sp = self._splatter_mask(shape, coverage=cover_s_s, blur_sigma=max(1,int(0.5*blur_size)))
            
            strokes_est = max(8, int((cover_b_s*H*W)//(stroke_length_s*max(1,stroke_width_s))))
            strokes_est = max(1, int(strokes_est*(1+random.uniform(-stroke_spread_jitter, stroke_spread_jitter))))
            mask_br = self._brush_mask(shape, coverage=cover_b_s, stroke_length=stroke_length_s,
                                       stroke_width=stroke_width_s, strokes_per_image=strokes_est)
            
            masks = (mask_br - mask_sp)
            
        else:
            if torch.rand(1).item() < 0.5:
                masks = self._splatter_mask(shape, coverage=cover_s_s, blur_sigma=max(1,int(0.5*blur_size)))
            
            else:
                strokes_est = max(8, int((cover_b_s*H*W)//(stroke_length_s*max(1,stroke_width_s))))
                strokes_est = max(1, int(strokes_est*(1+random.uniform(-stroke_spread_jitter, stroke_spread_jitter))))
                masks = self._brush_mask(shape, coverage=cover_b_s, stroke_length=stroke_length_s,
                                           stroke_width=stroke_width_s, strokes_per_image=strokes_est)


        #Clip mask values to 0 1 range
        masks = masks.clamp(0.0,1.0)

        #Optionally apply "lasso" for removing some damage (so its not like whole img is always damaged)
        if torch.rand(1).item() < apply_lasso_prob:
            c = random.uniform(lasso_frac[0], lasso_frac[1])
            lasso = self._splatter_mask((B,H,W), coverage=c, blur_sigma=lasso_blur)
            masks = masks * (1.0 - lasso)

        #Create metadata for optional checking of the parameters
        metadata = {
            'op': op,
            'cover_s': cover_s_s,
            'cover_b': cover_b_s,
            'stroke_length': stroke_length_s,
            'stroke_width': stroke_width_s,
            'strokes_est': strokes_est,
            'lasso_blur': lasso_blur,
        }

        return masks, metadata    
    
    


def augment_image_and_mask(image_batch, mask_batch,
                           brightness=0.2, contrast=0.2, saturation=0.2,
                           flip_prob=0.5, max_rot=15, crop_ratio=0.9):
    """
    image_batch: (B, C, H, W) tensor, float in [0,1]
    mask_batch: (B, 1, H, W) tensor, float in [0,1]
    Returns: augmented_image_batch, augmented_mask_batch (both in [0,1])
    """
    B, C, H, W = image_batch.shape
    device = image_batch.device
    dtype = image_batch.dtype

    if mask_batch.dim() == 3:
        mask_batch = mask_batch.unsqueeze(1)

    # clone to avoid in-place modifications
    image_batch = image_batch.clone()
    mask_batch = mask_batch.clone()

    # -----------------------------
    # 1. Random horizontal flip
    # -----------------------------
    flip_mask = torch.rand(B, device=device) < flip_prob
    if flip_mask.any():
        image_batch[flip_mask] = image_batch[flip_mask].flip(dims=[-1])
        mask_batch[flip_mask] = mask_batch[flip_mask].flip(dims=[-1])

    # -----------------------------
    # 2. Random rotation
    # -----------------------------
    angles = (torch.rand(B, device=device) * 2 - 1) * max_rot  # -max_rot .. +max_rot
    radians = angles * (3.14159265 / 180)

    cos = torch.cos(radians)
    sin = torch.sin(radians)
    rot_matrices = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    rot_matrices[:, 0, 0] = cos
    rot_matrices[:, 0, 1] = -sin
    rot_matrices[:, 1, 0] = sin
    rot_matrices[:, 1, 1] = cos
    rot_matrices[:, :, 2] = 0  # rotate around center

    grid = F.affine_grid(rot_matrices, image_batch.size(), align_corners=False)
    image_batch = F.grid_sample(image_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    mask_batch = F.grid_sample(mask_batch, grid, mode='nearest', padding_mode='zeros', align_corners=False)

    # -----------------------------
    # 3. Random crop + resize
    # -----------------------------
    if crop_ratio < 1.0:
        crop_h = int(H * crop_ratio)
        crop_w = int(W * crop_ratio)
        top = torch.randint(0, H - crop_h + 1, (B,), device=device)
        left = torch.randint(0, W - crop_w + 1, (B,), device=device)

        cropped_images = torch.zeros_like(image_batch)
        cropped_masks = torch.zeros_like(mask_batch)
        for i in range(B):
            cropped_images[i] = F.interpolate(
                image_batch[i:i+1, :, top[i]:top[i]+crop_h, left[i]:left[i]+crop_w],
                size=(H, W), mode='bilinear', align_corners=False
            )
            cropped_masks[i] = F.interpolate(
                mask_batch[i:i+1, :, top[i]:top[i]+crop_h, left[i]:left[i]+crop_w],
                size=(H, W), mode='nearest'
            )
        image_batch = cropped_images
        mask_batch = cropped_masks

    # -----------------------------
    # 4. Color jitter
    # -----------------------------
    b_factors = 1.0 + (torch.rand(B,1,1,1, device=device, dtype=dtype) * 2 - 1) * brightness
    image_batch = image_batch * b_factors

    mean = image_batch.mean(dim=[2,3], keepdim=True)
    c_factors = 1.0 + (torch.rand(B,1,1,1, device=device, dtype=dtype) * 2 - 1) * contrast
    image_batch = (image_batch - mean) * c_factors + mean

    if C == 3:
        gray = image_batch.mean(dim=1, keepdim=True)
        s_factors = 1.0 + (torch.rand(B,1,1,1, device=device, dtype=dtype) * 2 - 1) * saturation
        image_batch = (image_batch - gray) * s_factors + gray

    # -----------------------------
    # 5. Clamp to [0,1] range
    # -----------------------------
    image_batch = image_batch.clamp(0, 1)
    mask_batch = mask_batch.clamp(0, 1)

    return image_batch, mask_batch

    