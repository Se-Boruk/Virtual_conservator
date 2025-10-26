import torch
import torch.nn as nn
import torchmetrics
import os
import matplotlib.pyplot as plt
import io
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssim_fn = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()

def total_variation_loss(x):
    """
    x: (B, C, H, W) in [-1,1] or 
    Computes normalized total variation loss
    
    Function assumes the range of input in -1 to 1
    """
    
    x_scaled = (x + 1) / 2  # normalize to [0,1]
    dh = torch.abs(x_scaled[:, :, 1:, :] - x_scaled[:, :, :-1, :])
    dw = torch.abs(x_scaled[:, :, :, 1:] - x_scaled[:, :, :, :-1])
    
    return (dh.mean() + dw.mean())


def ssim_loss(pred, target):
    """
    pred, target: (B, C, H, W) in [-1,1] 
    Normalizes and computes 1-SSIM
    
    Function assumes the range of input in -1 to 1
    """
    
    pred_scaled = (pred + 1)/2
    target_scaled = (target + 1)/2
    return 1 - ssim_fn(pred_scaled, target_scaled)



def inpainting_loss(pred, target,
                    l1_weight=1.0,
                    l2_weight=0.1,
                    ssim_weight=0.05,
                    tv_weight=0.001):
    """
    Combined loss for image inpainting:
    L1 + alpha*L2 + beta*(1-SSIM) + gamma*TV
    """
    l1 = l1_loss(pred, target)
    l2 = l2_loss(pred, target)
    ssim = ssim_loss(pred, target)
    tv = total_variation_loss(pred)
    
    return l1_weight*l1 + l2_weight*l2 + ssim_weight*ssim + tv_weight*tv



class InpaintingVisualizer:
    def __init__(self, inpainter_model, damage_generator, rows=8, H=256, W=256, device=None,
                 save_dir="visualizations", show=True, save=True):
        """
        Args:
            inpainter_model: PyTorch model
            damage_generator: object with .generate((1,H,W)) method returning mask
            rows: number of images to visualize
            H, W: height and width of generated masks
            device: torch device
            save_dir: where to save plots
            show: whether to show plots
            save: whether to save plots
        """
        self.model = inpainter_model
        self.model.eval()  # ensure eval mode
        self.dmg_gen = damage_generator
        self.rows = rows
        self.H = H
        self.W = W
        self.device = device or next(self.model.parameters()).device
        self.save_dir = save_dir
        self.show = show
        self.save = save
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Pre-generate masks (fixed for all epochs)
        self.masks = torch.empty((rows, 1, H, W), dtype=torch.float32, device=self.device)
        for i in range(rows):
            mask, _ = self.dmg_gen.generate((1, H, W))
            self.masks[i] = mask.unsqueeze(0).to(self.device)
            
    def visualize(self, img_batch, epoch=None, prefix="epoch", comet_experiment = None):
        """
        Args:
            img_batch: tensor (B, C, H, W), normalized [-1,1]
            epoch: optional epoch number (used for filename)
            prefix: prefix for saved files
        """
        # Take first self.rows images
        original = img_batch[:self.rows].to(self.device)
        
        # Apply mask
        damaged = torch.where(self.masks.bool(), -1.0, original)
        
        # Inference
        with torch.inference_mode():
            restored = self.model(damaged).detach()
        
        # Move to HWC and normalize [0,1] for plotting
        def prep(t):
            t = t.permute(0,2,3,1)  # BCHW -> BHWC
            return ((t + 1)/2).clamp(0,1)
        
        original_p = prep(original)
        damaged_p = prep(damaged)
        restored_p = prep(restored)
        
        # Plot
        fig, axes = plt.subplots(3, self.rows, figsize=(3*self.rows, 9))
        for i in range(self.rows):
            axes[0,i].imshow(original_p[i].cpu())
            axes[0,i].axis('off')
            axes[0,i].set_title("Original")
            
            axes[1,i].imshow(damaged_p[i].cpu())
            axes[1,i].axis('off')
            axes[1,i].set_title("Damaged")
            
            axes[2,i].imshow(restored_p[i].cpu())
            axes[2,i].axis('off')
            axes[2,i].set_title("Restored")
        
        plt.tight_layout()
        
        if self.save and epoch is not None:
            filename = os.path.join(self.save_dir, f"{prefix}_{epoch:04d}.png")
            plt.savefig(filename)
        
            
        #Log to Comet if said so
        if comet_experiment is not None:
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            comet_experiment.log_image(
                image_data=img,
                name=f"{prefix}_{epoch:04d}" if epoch is not None else prefix,
                overwrite=False
            )
        
        if self.show:
            plt.show()
        else:
            plt.close(fig)
            








