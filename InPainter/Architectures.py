import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.utils as nn_utils


#############################################################
#Inpainter
#############################################################


#GroupNorm helper
def groupnorm(channels, num_groups=8):
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels)

#Blur layer (StyleGAN2-style, reduces checkerboard artifacts)
class Blur(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor([1, 2, 1], dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()
        kernel = kernel[None, None, :, :].repeat(channels, 1, 1, 1)
        self.register_buffer('kernel', kernel)
        self.groups = channels

    def forward(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding=1, groups=self.groups)

#StyleGAN2-inspired ResBlock
class StyleGAN2ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1)
        self.norm1 = groupnorm(channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1)
        self.norm2 = groupnorm(channels)

        self.skip_gain = 1 / (2 ** 0.5)  # Scale residual

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        return (x + out) * self.skip_gain

# Generator with skip connections & StyleGAN2ResBlocks
class Inpainter_v0(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, n_residual=4, base_filters=32):
        super().__init__()
        f1 = base_filters
        f2 = base_filters * 2
        f3 = base_filters * 3
        f4 = base_filters * 4

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, f1, kernel_size = 7, stride = 1, padding = 3),
            nn.InstanceNorm2d(f1, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(f1, f1, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(f1, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(f1, f2, kernel_size = 3, stride = 2, padding = 1),
            nn.InstanceNorm2d(f2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(f2, f2, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(f2, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(f2, f3, kernel_size = 3, stride = 2, padding = 1),
            nn.InstanceNorm2d(f3, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(f3, f3, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(f3, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(f3, f4, kernel_size = 3, stride = 2, padding = 1),
            nn.InstanceNorm2d(f4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(f4, f4, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(f4, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
            
        )

        #Residual bottleneck
        self.resblocks = nn.Sequential(*[StyleGAN2ResBlock(f4) for _ in range(n_residual)])

        #Decoder with skip connections
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(f4, f4, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            Blur(f4),
            nn.InstanceNorm2d(f4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(f4, f3, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(f3, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(f3 * 2, f3, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            Blur(f3),
            nn.InstanceNorm2d(f3, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(f3, f2, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(f2, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(f2 * 2, f2, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            Blur(f2),
            nn.InstanceNorm2d(f2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(f2, f1, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(f1, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        #Output conv block
        self.output_conv = nn.Sequential(
            # First: normal conv to reduce channels
            nn.Conv2d(f1 * 2, f1, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(f1, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Dilated convs for larger receptive field
            nn.Conv2d(f1, f1, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.InstanceNorm2d(f1, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(f1, f1, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.InstanceNorm2d(f1, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Optional normal conv to consolidate features
            nn.Conv2d(f1, f1, kernel_size=3, stride=1, padding=2),
            nn.InstanceNorm2d(f1, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final output conv: no padding, directly outputs final image
            nn.Conv2d(f1, output_channels, kernel_size=7, stride=1, padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.resblocks(e4)

        # Decoder + skips
        d1 = self.dec1(b)
        d1 = torch.cat([d1, e3], dim=1)  # skip from e3

        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e2], dim=1)  # skip from e2

        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e1], dim=1)  # skip from e1

        out = self.output_conv(d3)
        #out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out