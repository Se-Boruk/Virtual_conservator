import torch
import torch.nn as nn
import torch.nn.functional as F

class Inpainter_V0:
    #================================================================
    #Helper modules
    #================================================================
    @staticmethod
    def groupnorm(channels, num_groups=8):
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)

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

    class StyleGAN2ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
            self.norm1 = Inpainter_V0.groupnorm(channels)
            self.act1 = nn.LeakyReLU(0.2, inplace=True)
            self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
            self.norm2 = Inpainter_V0.groupnorm(channels)
            self.skip_gain = 1 / (2 ** 0.5)

        def forward(self, x):
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.act1(out)
            out = self.conv2(out)
            out = self.norm2(out)
            return (x + out) * self.skip_gain

    class Processing_Block(nn.Module):
        def __init__(self, input_channels, channels):
            super().__init__()
            half_c = channels // 2

            self.conv_1 = nn.Conv2d(input_channels, channels, 3, 1, 1)
            self.norm_1 = nn.InstanceNorm2d(channels, affine=True)
            self.act_1 = nn.LeakyReLU(0.2, inplace=True)

            self.conv_21 = nn.Conv2d(channels, half_c, 3, 1, 1, dilation=1, groups=half_c)
            self.norm_21 = nn.InstanceNorm2d(half_c, affine=True)
            self.conv_22 = nn.Conv2d(half_c, half_c, 1, 1, 0)
            self.norm_22 = nn.InstanceNorm2d(half_c, affine=True)
            self.act_22 = nn.LeakyReLU(0.2, inplace=True)

            self.conv_31 = nn.Conv2d(channels, half_c, 3, 1, 2, dilation=2, groups=half_c)
            self.norm_31 = nn.InstanceNorm2d(half_c, affine=True)
            self.conv_32 = nn.Conv2d(half_c, half_c, 1, 1, 0)
            self.norm_32 = nn.InstanceNorm2d(half_c, affine=True)
            self.act_32 = nn.LeakyReLU(0.2, inplace=True)

            self.conv_41 = nn.Conv2d(channels, half_c, 3, 1, 3, dilation=3, groups=half_c)
            self.norm_41 = nn.InstanceNorm2d(half_c, affine=True)
            self.conv_42 = nn.Conv2d(half_c, half_c, 1, 1, 0)
            self.norm_42 = nn.InstanceNorm2d(half_c, affine=True)
            self.act_42 = nn.LeakyReLU(0.2, inplace=True)

            multi_head_c = half_c * 3
            self.conv_61 = nn.Conv2d(multi_head_c, channels, 3, 1, 1)
            self.norm_61 = nn.InstanceNorm2d(channels, affine=True)
            self.act_61 = nn.LeakyReLU(0.2, inplace=True)
            self.conv_71 = nn.Conv2d(channels, channels, 3, 1, 1)
            self.norm_71 = nn.InstanceNorm2d(channels, affine=True)
            self.act_71 = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x = self.conv_1(x)
            x = self.norm_1(x)
            x = self.act_1(x)

            x1 = self.conv_21(x)
            x1 = self.norm_21(x1)
            x1 = self.conv_22(x1)
            x1 = self.norm_22(x1)
            x1 = self.act_22(x1)

            x2 = self.conv_31(x)
            x2 = self.norm_31(x2)
            x2 = self.conv_32(x2)
            x2 = self.norm_32(x2)
            x2 = self.act_32(x2)

            x3 = self.conv_41(x)
            x3 = self.norm_41(x3)
            x3 = self.conv_42(x3)
            x3 = self.norm_42(x3)
            x3 = self.act_42(x3)

            e = torch.cat([x1, x2, x3], dim=1)
            e = self.conv_61(e)
            e = self.norm_61(e)
            e = self.act_61(e)
            e = self.conv_71(e)
            e = self.norm_71(e)
            e = self.act_71(e)
            return e + x

    class Downsample_block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv_1 = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
            self.norm_1 = nn.InstanceNorm2d(in_channels, affine=True)
            self.act_1 = nn.LeakyReLU(0.2, inplace=True)
            self.conv_2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            self.norm_2 = nn.InstanceNorm2d(out_channels, affine=True)
            self.act_2 = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x = self.conv_1(x)
            x = self.norm_1(x)
            x = self.act_1(x)
            x = self.conv_2(x)
            x = self.norm_2(x)
            x = self.act_2(x)
            return x

    class Upsample_block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv_1 = nn.ConvTranspose2d(in_channels, in_channels, 3, 2, 1, output_padding=1)
            self.norm_1 = nn.InstanceNorm2d(in_channels, affine=True)
            self.act_1 = nn.LeakyReLU(0.2, inplace=True)
            self.conv_2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            self.norm_2 = nn.InstanceNorm2d(out_channels, affine=True)
            self.act_2 = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x = self.conv_1(x)
            x = self.norm_1(x)
            x = self.act_1(x)
            x = self.conv_2(x)
            x = self.norm_2(x)
            x = self.act_2(x)
            return x

    #================================================================
    #Main model
    #================================================================
    class Inpainter(nn.Module):
        def __init__(self, input_channels=3, output_channels=3, n_residual=4, base_filters=32):
            super().__init__()
            f1, f2, f3, f4 = base_filters, base_filters*2, base_filters*3, base_filters*4

            #Encoder
            self.input_conv_block = nn.Sequential(
                nn.Conv2d(input_channels, f1, 7, 1, 3),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, f1, 3, 1, 1),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            )

            self.enc_0 = Inpainter_V0.Processing_Block(f1, f1)
            self.down_0 = Inpainter_V0.Downsample_block(f1, f2)
            self.enc_1 = Inpainter_V0.Processing_Block(f2, f2)
            self.down_1 = Inpainter_V0.Downsample_block(f2, f3)
            self.enc_2 = Inpainter_V0.Processing_Block(f3, f3)
            self.down_2 = Inpainter_V0.Downsample_block(f3, f4)

            self.resblocks = nn.Sequential(*[Inpainter_V0.StyleGAN2ResBlock(f4) for _ in range(n_residual)])

            #Decoder
            self.up_2 = Inpainter_V0.Upsample_block(f4, f3)
            self.dec_2 = Inpainter_V0.Processing_Block(f3*2, f3)
            self.up_1 = Inpainter_V0.Upsample_block(f3, f2)
            self.dec_1 = Inpainter_V0.Processing_Block(f2*2, f2)
            self.up_0 = Inpainter_V0.Upsample_block(f2, f1)
            self.dec_0 = Inpainter_V0.Processing_Block(f1*2, f1)

            #Output conv
            self.output_conv = nn.Sequential(
                nn.Conv2d(f1, f1, 3, 1, 3, dilation=3),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, f1, 3, 1, 2, dilation=2),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, f1, 3, 1, 1),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, f1, 3, 1, 1),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, f1, 3, 1, 2),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, output_channels, 7, 1, 2),
                nn.Tanh()
            )

        def forward(self, x):
            #Encoder
            x = self.input_conv_block(x)
            x_0 = self.enc_0(x)
            x = self.down_0(x_0)
            x_1 = self.enc_1(x)
            x = self.down_1(x_1)
            x_2 = self.enc_2(x)
            x = self.down_2(x_2)

            x = self.resblocks(x)

            #Decoder
            x = self.up_2(x)
            x_2 = torch.cat([x, x_2], dim=1)
            x = self.dec_2(x_2)
            x = self.up_1(x)
            x_1 = torch.cat([x, x_1], dim=1)
            x = self.dec_1(x_1)
            x = self.up_0(x)
            x_0 = torch.cat([x, x_0], dim=1)
            x = self.dec_0(x_0)

            x = self.output_conv(x)
            return x



class Inpainter_V1:
    #============================================================
    # Helper modules
    #============================================================
    @staticmethod
    def groupnorm(channels, num_groups=8):
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)

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

    class StyleGAN2ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
            self.norm1 = Inpainter_V1.groupnorm(channels)
            self.act1 = nn.LeakyReLU(0.2, inplace=True)
            self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
            self.norm2 = Inpainter_V1.groupnorm(channels)
            self.skip_gain = 1 / (2 ** 0.5)

        def forward(self, x):
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.act1(out)
            out = self.conv2(out)
            out = self.norm2(out)
            return (x + out) * self.skip_gain

    class Processing_Block(nn.Module):
        def __init__(self, input_channels, channels):
            super().__init__()
            half_c = channels // 2
            self.conv_1 = nn.Conv2d(input_channels, channels, 3, 1, 1)
            self.norm_1 = nn.InstanceNorm2d(channels, affine=True)
            self.act_1 = nn.LeakyReLU(0.2, inplace=True)

            # First Head
            self.conv_21 = nn.Conv2d(channels, half_c, 3, 1, 1, dilation=1, groups=half_c)
            self.norm_21 = nn.InstanceNorm2d(half_c, affine=True)
            self.conv_22 = nn.Conv2d(half_c, half_c, 1, 1, 0)
            self.norm_22 = nn.InstanceNorm2d(half_c, affine=True)
            self.act_22 = nn.LeakyReLU(0.2, inplace=True)

            # Second Head
            self.conv_31 = nn.Conv2d(channels, half_c, 3, 1, 2, dilation=2, groups=half_c)
            self.norm_31 = nn.InstanceNorm2d(half_c, affine=True)
            self.conv_32 = nn.Conv2d(half_c, half_c, 1, 1, 0)
            self.norm_32 = nn.InstanceNorm2d(half_c, affine=True)
            self.act_32 = nn.LeakyReLU(0.2, inplace=True)

            # Third Head
            self.conv_41 = nn.Conv2d(channels, half_c, 3, 1, 3, dilation=3, groups=half_c)
            self.norm_41 = nn.InstanceNorm2d(half_c, affine=True)
            self.conv_42 = nn.Conv2d(half_c, half_c, 1, 1, 0)
            self.norm_42 = nn.InstanceNorm2d(half_c, affine=True)
            self.act_42 = nn.LeakyReLU(0.2, inplace=True)

            multi_head_c = half_c * 3
            self.conv_61 = nn.Conv2d(multi_head_c, channels, 3, 1, 1)
            self.norm_61 = nn.InstanceNorm2d(channels, affine=True)
            self.act_61 = nn.LeakyReLU(0.2, inplace=True)
            self.conv_71 = nn.Conv2d(channels, channels, 3, 1, 1)
            self.norm_71 = nn.InstanceNorm2d(channels, affine=True)
            self.act_71 = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x = self.conv_1(x)
            x = self.norm_1(x)
            x = self.act_1(x)

            x1 = self.conv_21(x)
            x1 = self.norm_21(x1)
            x1 = self.conv_22(x1)
            x1 = self.norm_22(x1)
            x1 = self.act_22(x1)

            x2 = self.conv_31(x)
            x2 = self.norm_31(x2)
            x2 = self.conv_32(x2)
            x2 = self.norm_32(x2)
            x2 = self.act_32(x2)

            x3 = self.conv_41(x)
            x3 = self.norm_41(x3)
            x3 = self.conv_42(x3)
            x3 = self.norm_42(x3)
            x3 = self.act_42(x3)

            e = torch.cat([x1, x2, x3], dim=1)
            e = self.conv_61(e)
            e = self.norm_61(e)
            e = self.act_61(e)
            e = self.conv_71(e)
            e = self.norm_71(e)
            e = self.act_71(e)
            return e + x

    class Downsample_block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv_1 = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
            self.norm_1 = nn.InstanceNorm2d(in_channels, affine=True)
            self.act_1 = nn.LeakyReLU(0.2, inplace=True)
            self.conv_2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            self.norm_2 = nn.InstanceNorm2d(out_channels, affine=True)
            self.act_2 = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x = self.conv_1(x)
            x = self.norm_1(x)
            x = self.act_1(x)
            x = self.conv_2(x)
            x = self.norm_2(x)
            x = self.act_2(x)
            return x

    class Upsample_block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv_1 = nn.ConvTranspose2d(in_channels, in_channels, 3, 2, 1, output_padding=1)
            self.norm_1 = nn.InstanceNorm2d(in_channels, affine=True)
            self.act_1 = nn.LeakyReLU(0.2, inplace=True)
            self.conv_2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            self.norm_2 = nn.InstanceNorm2d(out_channels, affine=True)
            self.act_2 = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x = self.conv_1(x)
            x = self.norm_1(x)
            x = self.act_1(x)
            x = self.conv_2(x)
            x = self.norm_2(x)
            x = self.act_2(x)
            return x

    #============================================================
    # Encoder
    #============================================================
    class Encoder(nn.Module):
        def __init__(self, input_channels=3, base_filters=32, n_residual=4):
            super().__init__()
            f1, f2, f3, f4 = base_filters, base_filters*2, base_filters*3, base_filters*4

            self.input_conv_block = nn.Sequential(
                nn.Conv2d(input_channels, f1, 7, 1, 3),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, f1, 3, 1, 1),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            )

            self.enc_0 = Inpainter_V1.Processing_Block(f1, f1)
            self.down_0 = Inpainter_V1.Downsample_block(f1, f2)
            self.enc_1 = Inpainter_V1.Processing_Block(f2, f2)
            self.down_1 = Inpainter_V1.Downsample_block(f2, f3)
            self.enc_2 = Inpainter_V1.Processing_Block(f3, f3)
            self.down_2 = Inpainter_V1.Downsample_block(f3, f4)

            self.resblocks = nn.Sequential(*[Inpainter_V1.StyleGAN2ResBlock(f4) for _ in range(n_residual)])

        def forward(self, x):
            x = self.input_conv_block(x)
            x0 = self.enc_0(x)
            x = self.down_0(x0)
            x1 = self.enc_1(x)
            x = self.down_1(x1)
            x2 = self.enc_2(x)
            x = self.down_2(x2)

            x = self.resblocks(x)

            # Return bottleneck and skip connections
            return x, (x0, x1, x2)

    #============================================================
    # Decoder with class map input
    #============================================================
    class Decoder(nn.Module):
        def __init__(self, output_channels=3, base_filters=32):
            super().__init__()
            f1, f2, f3, f4 = base_filters, base_filters*2, base_filters*3, base_filters*4
    
            self.up_2 = Inpainter_V1.Upsample_block(f4 + 1, f3)  # +1 for class_map
            self.dec_2 = Inpainter_V1.Processing_Block(f3*2, f3)
            self.up_1 = Inpainter_V1.Upsample_block(f3, f2)
            self.dec_1 = Inpainter_V1.Processing_Block(f2*2, f2)
            self.up_0 = Inpainter_V1.Upsample_block(f2, f1)
            self.dec_0 = Inpainter_V1.Processing_Block(f1*2, f1)
    
            self.output_conv = nn.Sequential(
                nn.Conv2d(f1, f1, 3, 1, 3, dilation=3),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, f1, 3, 1, 2, dilation=2),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, f1, 3, 1, 1),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, f1, 3, 1, 1),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, f1, 3, 1, 2),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, output_channels, 7, 1, 2),
                nn.Tanh()
            )
    
        def forward(self, x, skip_0, skip_1, skip_2, class_vector):

            H, W = x.shape[-2], x.shape[-1]
            
            #Take the class vector and transfor it into the uniform channel (feature) of some value (0,1,2...)
            class_map = class_vector.unsqueeze(-1).unsqueeze(-1)
            class_map = class_map.expand(-1, -1, H, W)
            
            #Concat the output tensor with the class information
            x = torch.cat([x, class_map], dim=1)
            
                
            x = self.up_2(x)
            x = torch.cat([x, skip_2], dim=1)
            x = self.dec_2(x)
    
            x = self.up_1(x)
            x = torch.cat([x, skip_1], dim=1)
            x = self.dec_1(x)
    
            x = self.up_0(x)
            x = torch.cat([x, skip_0], dim=1)
            x = self.dec_0(x)
    
            x = self.output_conv(x)
            return x




class Inpainter_V2:
    #============================================================
    # Helper modules
    #============================================================
    @staticmethod
    def groupnorm(channels, num_groups=8):
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)

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

    class StyleGAN2ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
            self.norm1 = Inpainter_V2.groupnorm(channels)
            self.act1 = nn.LeakyReLU(0.2, inplace=True)
            self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
            self.norm2 = Inpainter_V2.groupnorm(channels)
            self.skip_gain = 1 / (2 ** 0.5)

        def forward(self, x):
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.act1(out)
            out = self.conv2(out)
            out = self.norm2(out)
            return (x + out) * self.skip_gain

    class Processing_Block(nn.Module):
        def __init__(self, input_channels, channels):
            super().__init__()
            half_c = channels // 2
            self.conv_1 = nn.Conv2d(input_channels, channels, 3, 1, 1)
            self.norm_1 = nn.InstanceNorm2d(channels, affine=True)
            self.act_1 = nn.LeakyReLU(0.2, inplace=True)

            # First Head
            self.conv_21 = nn.Conv2d(channels, half_c, 3, 1, 1, dilation=1, groups=half_c)
            self.norm_21 = nn.InstanceNorm2d(half_c, affine=True)
            self.conv_22 = nn.Conv2d(half_c, half_c, 1, 1, 0)
            self.norm_22 = nn.InstanceNorm2d(half_c, affine=True)
            self.act_22 = nn.LeakyReLU(0.2, inplace=True)

            # Second Head
            self.conv_31 = nn.Conv2d(channels, half_c, 3, 1, 2, dilation=2, groups=half_c)
            self.norm_31 = nn.InstanceNorm2d(half_c, affine=True)
            self.conv_32 = nn.Conv2d(half_c, half_c, 1, 1, 0)
            self.norm_32 = nn.InstanceNorm2d(half_c, affine=True)
            self.act_32 = nn.LeakyReLU(0.2, inplace=True)

            # Third Head
            self.conv_41 = nn.Conv2d(channels, half_c, 3, 1, 3, dilation=3, groups=half_c)
            self.norm_41 = nn.InstanceNorm2d(half_c, affine=True)
            self.conv_42 = nn.Conv2d(half_c, half_c, 1, 1, 0)
            self.norm_42 = nn.InstanceNorm2d(half_c, affine=True)
            self.act_42 = nn.LeakyReLU(0.2, inplace=True)

            multi_head_c = half_c * 3
            self.conv_61 = nn.Conv2d(multi_head_c, channels, 3, 1, 1)
            self.norm_61 = nn.InstanceNorm2d(channels, affine=True)
            self.act_61 = nn.LeakyReLU(0.2, inplace=True)
            self.conv_71 = nn.Conv2d(channels, channels, 3, 1, 1)
            self.norm_71 = nn.InstanceNorm2d(channels, affine=True)
            self.act_71 = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x = self.conv_1(x)
            x = self.norm_1(x)
            x = self.act_1(x)

            x1 = self.conv_21(x)
            x1 = self.norm_21(x1)
            x1 = self.conv_22(x1)
            x1 = self.norm_22(x1)
            x1 = self.act_22(x1)

            x2 = self.conv_31(x)
            x2 = self.norm_31(x2)
            x2 = self.conv_32(x2)
            x2 = self.norm_32(x2)
            x2 = self.act_32(x2)

            x3 = self.conv_41(x)
            x3 = self.norm_41(x3)
            x3 = self.conv_42(x3)
            x3 = self.norm_42(x3)
            x3 = self.act_42(x3)

            e = torch.cat([x1, x2, x3], dim=1)
            e = self.conv_61(e)
            e = self.norm_61(e)
            e = self.act_61(e)
            e = self.conv_71(e)
            e = self.norm_71(e)
            e = self.act_71(e)
            return e + x

    class Downsample_block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv_1 = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
            self.norm_1 = nn.InstanceNorm2d(in_channels, affine=True)
            self.act_1 = nn.LeakyReLU(0.2, inplace=True)
            self.conv_2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            self.norm_2 = nn.InstanceNorm2d(out_channels, affine=True)
            self.act_2 = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x = self.conv_1(x)
            x = self.norm_1(x)
            x = self.act_1(x)
            x = self.conv_2(x)
            x = self.norm_2(x)
            x = self.act_2(x)
            return x

    class Upsample_block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv_1 = nn.ConvTranspose2d(in_channels, in_channels, 3, 2, 1, output_padding=1)
            self.norm_1 = nn.InstanceNorm2d(in_channels, affine=True)
            self.act_1 = nn.LeakyReLU(0.2, inplace=True)
            self.conv_2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            self.norm_2 = nn.InstanceNorm2d(out_channels, affine=True)
            self.act_2 = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x = self.conv_1(x)
            x = self.norm_1(x)
            x = self.act_1(x)
            x = self.conv_2(x)
            x = self.norm_2(x)
            x = self.act_2(x)
            return x
        
        
    class Class_MLP(nn.Module):
        def __init__(self, in_dim=1, hidden_dim=256, out_channels=4, out_h=16, out_w=16):
            super().__init__()        
            self.out_channels = out_channels
            self.out_h = out_h
            self.out_w = out_w
    
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, out_channels * out_h * out_w),
                nn.LeakyReLU(0.2, inplace=True),
            )

        def forward(self, x):

            B = x.shape[0]
            x = self.mlp(x)    
            
            x = x.view(B, self.out_channels, self.out_h, self.out_w)
            
            return x


    #============================================================
    # Encoder
    #============================================================
    class Encoder(nn.Module):
        def __init__(self, input_channels=3, base_filters=32, n_residual=4):
            super().__init__()
            f1, f2, f3, f4 = base_filters, base_filters*2, base_filters*3, base_filters*4

            self.input_conv_block = nn.Sequential(
                nn.Conv2d(input_channels, f1, 7, 1, 3),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, f1, 3, 1, 1),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            )

            self.enc_0 = Inpainter_V2.Processing_Block(f1, f1)
            self.down_0 = Inpainter_V2.Downsample_block(f1, f2)
            self.enc_1 = Inpainter_V2.Processing_Block(f2, f2)
            self.down_1 = Inpainter_V2.Downsample_block(f2, f3)
            self.enc_2 = Inpainter_V2.Processing_Block(f3, f3)
            self.down_2 = Inpainter_V2.Downsample_block(f3, f4)

            self.resblocks = nn.Sequential(*[Inpainter_V2.StyleGAN2ResBlock(f4) for _ in range(n_residual)])

        def forward(self, x):
            x = self.input_conv_block(x)
            x0 = self.enc_0(x)
            x = self.down_0(x0)
            x1 = self.enc_1(x)
            x = self.down_1(x1)
            x2 = self.enc_2(x)
            x = self.down_2(x2)

            x = self.resblocks(x)

            # Return bottleneck and skip connections
            return x, (x0, x1, x2)

    #============================================================
    # Decoder with class map input
    #============================================================
    class Decoder(nn.Module):
        def __init__(self, output_channels=3, base_filters=32):
            super().__init__()
            f1, f2, f3, f4 = base_filters, base_filters*2, base_filters*3, base_filters*4
    
            self.up_2 = Inpainter_V2.Upsample_block(f4 + 4, f3)  # +4 for class_map
            self.dec_2 = Inpainter_V2.Processing_Block(f3*2, f3)
            self.up_1 = Inpainter_V2.Upsample_block(f3, f2)
            self.dec_1 = Inpainter_V2.Processing_Block(f2*2, f2)
            self.up_0 = Inpainter_V2.Upsample_block(f2, f1)
            self.dec_0 = Inpainter_V2.Processing_Block(f1*2, f1)
            
            self.class_mapping = Inpainter_V2.Class_MLP(in_dim=1, hidden_dim=int(base_filters*8), out_channels=4, out_h=16, out_w=16)
    
            self.merge = nn.Sequential(nn.Conv2d(int(base_filters*4 + 4), int(base_filters*4 + 4), 1),
                                       nn.InstanceNorm2d(int(base_filters*4 + 4), affine=True),
                                       nn.LeakyReLU(0.2, inplace=True)
                                       )
                
    
    
            self.output_conv = nn.Sequential(
                nn.Conv2d(f1, f1, 3, 1, 3, dilation=3),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, f1, 3, 1, 2, dilation=2),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, f1, 3, 1, 1),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, f1, 3, 1, 1),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, f1, 3, 1, 2),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, output_channels, 7, 1, 2),
                nn.Tanh()
            )
    
        def forward(self, x, skip_0, skip_1, skip_2, class_vector):

            H, W = x.shape[-2], x.shape[-1]
            
            #Take the class vector and transfor it into the uniform channel (feature) of some value (0,1,2...)
            class_map = self.class_mapping(class_vector)
            class_map = F.interpolate(class_map, size=(H, W), mode='bilinear', align_corners=False)
            
            
            
            #Concat the output tensor with the class information
            x = torch.cat([x, class_map], dim=1)
            
            x = self.merge(x)    
            
            x = self.up_2(x)
            x = torch.cat([x, skip_2], dim=1)
            x = self.dec_2(x)
    
            x = self.up_1(x)
            x = torch.cat([x, skip_1], dim=1)
            x = self.dec_1(x)
    
            x = self.up_0(x)
            x = torch.cat([x, skip_0], dim=1)
            x = self.dec_0(x)
    
            x = self.output_conv(x)
            return x






class Inpainter_V3:
    #============================================================
    # Helper modules
    #============================================================
    @staticmethod
    def groupnorm(channels, num_groups=8):
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)

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

    class StyleGAN2ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
            self.norm1 = Inpainter_V3.groupnorm(channels)
            self.act1 = nn.LeakyReLU(0.2, inplace=True)
            self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
            self.norm2 = Inpainter_V3.groupnorm(channels)
            self.skip_gain = 1 / (2 ** 0.5)

        def forward(self, x):
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.act1(out)
            out = self.conv2(out)
            out = self.norm2(out)
            return (x + out) * self.skip_gain

    #============================================================
    # Processing block with learnable weighted add
    #============================================================
    class Processing_Block(nn.Module):
        def __init__(self, input_channels, channels):
            super().__init__()
            half_c = channels // 2
            self.conv_1 = nn.Conv2d(input_channels, channels, 3, 1, 1)
            self.norm_1 = nn.InstanceNorm2d(channels, affine=True)
            self.act_1 = nn.LeakyReLU(0.2, inplace=True)

            # Heads
            self.head1 = nn.Sequential(
                nn.Conv2d(channels, half_c, 3, 1, 1, dilation=1, groups=half_c),
                nn.InstanceNorm2d(half_c, affine=True),
                nn.Conv2d(half_c, half_c, 1, 1, 0),
                nn.InstanceNorm2d(half_c, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.head2 = nn.Sequential(
                nn.Conv2d(channels, half_c, 3, 1, 2, dilation=2, groups=half_c),
                nn.InstanceNorm2d(half_c, affine=True),
                nn.Conv2d(half_c, half_c, 1, 1, 0),
                nn.InstanceNorm2d(half_c, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.head3 = nn.Sequential(
                nn.Conv2d(channels, half_c, 3, 1, 3, dilation=3, groups=half_c),
                nn.InstanceNorm2d(half_c, affine=True),
                nn.Conv2d(half_c, half_c, 1, 1, 0),
                nn.InstanceNorm2d(half_c, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            )

            # Learnable weights for add
            self.head_weights = nn.Parameter(torch.ones(3))  # initialized equally

            # Post-processing
            self.conv_61 = nn.Conv2d(half_c, channels, kernel_size=1, stride=1, padding=0)
            self.norm_61 = nn.InstanceNorm2d(channels, affine=True)
            self.act_61 = nn.LeakyReLU(0.2, inplace=True)
            self.conv_71 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
            self.norm_71 = nn.InstanceNorm2d(channels, affine=True)
            self.act_71 = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x = self.act_1(self.norm_1(self.conv_1(x)))

            x1 = self.head1(x)
            x2 = self.head2(x)
            x3 = self.head3(x)

            w = torch.sigmoid(self.head_weights)
            e = w[0]*x1 + w[1]*x2 + w[2]*x3

            e = self.act_61(self.norm_61(self.conv_61(e)))
            e = self.act_71(self.norm_71(self.conv_71(e)))
            return e + x

    #============================================================
    # Downsample and Upsample blocks unchanged
    #============================================================
    class Downsample_block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv_stride = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
            self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
            self.act1 = nn.LeakyReLU(0.2, inplace=True)
            
            self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1, stride=1)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
            self.act2 = nn.LeakyReLU(0.2, inplace=True)
    
        def forward(self, x):
            x = self.act1(self.norm1(self.conv_stride(x)))
            x = self.act2(self.norm2(self.conv_1x1(x)))
            return x

    class Upsample_block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv_transpose = nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)
            self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
            self.act1 = nn.LeakyReLU(0.2, inplace=True)
            
            self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1, stride=1)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
            self.act2 = nn.LeakyReLU(0.2, inplace=True)
    
        def forward(self, x):
            x = self.act1(self.norm1(self.conv_transpose(x)))
            x = self.act2(self.norm2(self.conv_1x1(x)))
            return x

    #============================================================
    # Bottleneck MLP with 8 hidden channels (for adding class information)
    #============================================================
    class Class_MLP(nn.Module):
        def __init__(self, in_dim=1, hidden_dim=256, out_channels=8, out_h=8, out_w=8):
            super().__init__()
            self.out_channels = out_channels
            self.out_h = out_h
            self.out_w = out_w

            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, out_channels * out_h * out_w),
                nn.LeakyReLU(0.2, inplace=True),
            )

        def forward(self, x):
            B = x.shape[0]
            x = self.mlp(x)
            x = x.view(B, self.out_channels, self.out_h, self.out_w)
            return x

    #============================================================
    # Encoder
    #============================================================
    class Encoder(nn.Module):
        def __init__(self, input_channels=3, base_filters=32, n_residual=4):
            super().__init__()
            f1 = base_filters
            f2 = base_filters*2
            f3 = base_filters*3
            f4 = base_filters*4
            f5 = base_filters*5  # new deeper level = 160 if base=32

            self.input_conv_block = nn.Sequential(
                nn.Conv2d(input_channels, f1, 7, 1, 3),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(f1, f1, 3, 1, 1),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            )

            self.enc_0 = Inpainter_V3.Processing_Block(f1, f1)
            self.down_0 = Inpainter_V3.Downsample_block(f1, f2)
            self.enc_1 = Inpainter_V3.Processing_Block(f2, f2)
            self.down_1 = Inpainter_V3.Downsample_block(f2, f3)
            self.enc_2 = Inpainter_V3.Processing_Block(f3, f3)
            self.down_2 = Inpainter_V3.Downsample_block(f3, f4)
            self.enc_3 = Inpainter_V3.Processing_Block(f4, f4)
            self.down_3 = Inpainter_V3.Downsample_block(f4, f5)

            self.resblocks = nn.Sequential(*[Inpainter_V3.StyleGAN2ResBlock(f5) for _ in range(n_residual)])

        def forward(self, x):
            x = self.input_conv_block(x)
            x0 = self.enc_0(x)
            x = self.down_0(x0)
            x1 = self.enc_1(x)
            x = self.down_1(x1)
            x2 = self.enc_2(x)
            x = self.down_2(x2)
            x3 = self.enc_3(x)
            x = self.down_3(x3)

            x = self.resblocks(x)
            return x, (x0, x1, x2, x3)

    #============================================================
    # Decoder
    #============================================================
    class Decoder(nn.Module):
        def __init__(self, output_channels=3, base_filters=32):
            super().__init__()
            f1, f2, f3, f4, f5 = base_filters, base_filters*2, base_filters*3, base_filters*4, base_filters*5
    
            # Upsample blocks
            self.up_3 = Inpainter_V3.Upsample_block(f5 + 8, f4)  # +8 for class_map
            self.dec_3 = Inpainter_V3.Processing_Block(f4, f4)  # only f4, skip added separately
            self.up_2 = Inpainter_V3.Upsample_block(f4, f3)
            self.dec_2 = Inpainter_V3.Processing_Block(f3, f3)
            self.up_1 = Inpainter_V3.Upsample_block(f3, f2)
            self.dec_1 = Inpainter_V3.Processing_Block(f2, f2)
            self.up_0 = Inpainter_V3.Upsample_block(f2, f1)
            self.dec_0 = Inpainter_V3.Processing_Block(f1, f1)
    
            # Class mapping
            self.class_mapping = Inpainter_V3.Class_MLP(
                in_dim=1, hidden_dim=int(base_filters*8),
                out_channels=8, out_h=8, out_w=8
            )
    
            # Merge after class-map concat
            self.merge = nn.Sequential(
                nn.Conv2d(f5 + 8, f5 + 8, 1),
                nn.InstanceNorm2d(f5 + 8, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            )
    
    
            # Output conv
            self.output_conv = nn.Sequential(
                nn.Conv2d(f1, f1, 3, 1, 3, dilation=3),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(f1, f1, 3, 1, 2, dilation=2),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(f1, f1, 3, 1, 1),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(f1, f1, 3, 1, 2),
                nn.InstanceNorm2d(f1, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(f1, output_channels, 5, 1, 1),
                nn.Tanh()
            )
    
        def forward(self, x, skip_0, skip_1, skip_2, skip_3, class_vector):
            H, W = x.shape[-2], x.shape[-1]
            class_map = self.class_mapping(class_vector)
            class_map = F.interpolate(class_map, size=(H, W), mode='bilinear', align_corners=False)
    
            # Merge class map
            x = torch.cat([x, class_map], dim=1)
            x = self.merge(x)
    
            # Decoder with weighted skip connection
            x = self.up_3(x)
            x = x + 0.5 * skip_3
            x = self.dec_3(x)
    
            x = self.up_2(x)
            x = x + 0.5 * skip_2
            x = self.dec_2(x)
    
            x = self.up_1(x)
            x = x + 0.5 * skip_1
            x = self.dec_1(x)
    
            x = self.up_0(x)
            x = x + 0.5 * skip_0
            x = self.dec_0(x)
    
            # Output
            x = self.output_conv(x)
            return x







