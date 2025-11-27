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


class Processing_Block(nn.Module):
    def __init__(self, input_channels, channels):
        super().__init__()
        half_c = channels //2

        
        #Input layer for the initial processing
        self.conv_1 = nn.Conv2d(input_channels, channels, kernel_size = 3, stride = 1, padding = 1)
        self.norm_1 = nn.InstanceNorm2d(channels, affine=True)
        self.act_1 = nn.LeakyReLU(0.2, inplace=True)
        
        ############################
        #conv small receptive field ( no dilation so no atrous)
        self.conv_21 = nn.Conv2d(channels, half_c, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = half_c)
        self.norm_21 = nn.InstanceNorm2d(half_c, affine=True)
        
        self.conv_22 = nn.Conv2d(half_c, half_c, kernel_size = 1, stride = 1, padding = 0)
        self.norm_22 = nn.InstanceNorm2d(half_c, affine=True)
        self.act_22 = nn.LeakyReLU(0.2, inplace=True)
        ############################
        ############################
        #Atrous conv small receptive field (dilation = 2)
        self.conv_31 = nn.Conv2d(channels, half_c, kernel_size = 3, stride = 1, padding = 2, dilation = 2, groups = half_c)
        self.norm_31 = nn.InstanceNorm2d(half_c, affine=True)
        
        self.conv_32 = nn.Conv2d(half_c, half_c, kernel_size = 1, stride = 1, padding = 0)
        self.norm_32 = nn.InstanceNorm2d(half_c, affine=True)
        self.act_32 = nn.LeakyReLU(0.2, inplace=True)
        ############################
        ############################
        #Atrous conv medium receptive field (dilation = 3)
        self.conv_41 = nn.Conv2d(channels, half_c, kernel_size = 3, stride = 1, padding = 3, dilation = 3, groups = half_c)
        self.norm_41 = nn.InstanceNorm2d(half_c, affine=True)
        
        self.conv_42 = nn.Conv2d(half_c, half_c, kernel_size = 1, stride = 1, padding = 0)
        self.norm_42 = nn.InstanceNorm2d(half_c, affine=True)
        self.act_42 = nn.LeakyReLU(0.2, inplace=True)
        ############################
        ############################
        #Atrous conv big receptive field (dilation = 4)
        #self.conv_51 = nn.Conv2d(channels, half_c, kernel_size = 3, stride = 1, padding = 4, dilation = 4, groups = half_c)
        #self.norm_51 = nn.InstanceNorm2d(half_c, affine=True)
        
        #self.conv_52 = nn.Conv2d(half_c, half_c, kernel_size = 1, stride = 1, padding = 0)
        #self.norm_52 = nn.InstanceNorm2d(half_c, affine=True)
        #self.act_52 = nn.LeakyReLU(0.2, inplace=True)
        ############################
        
        
        #Multi atrous head processing conv
        multi_head_c = int((half_c*3))
        
        self.conv_61 = nn.Conv2d(multi_head_c, channels, kernel_size = 3, stride = 1, padding = 1)
        self.norm_61 = nn.InstanceNorm2d(channels, affine=True)
        self.act_61 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv_71 = nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1)
        self.norm_71 = nn.InstanceNorm2d(channels, affine=True)
        self.act_71 = nn.LeakyReLU(0.2, inplace=True)
        
        

    def forward(self, x):
        #Input
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.act_1(x)
        
        #First Head
        x1 = self.conv_21(x)
        x1 = self.norm_21(x1)
        x1 = self.conv_22(x1)
        x1 = self.norm_22(x1)
        x1 = self.act_22(x1)
        
        #2nd Head
        x2 = self.conv_31(x)
        x2 = self.norm_31(x2)
        x2 = self.conv_32(x2)
        x2 = self.norm_32(x2)
        x2 = self.act_32(x2)
        
        #3rd Head
        x3 = self.conv_41(x)
        x3 = self.norm_41(x3)
        x3 = self.conv_42(x3)
        x3 = self.norm_42(x3)
        x3 = self.act_42(x3)
        
        #4th Head
        #x4 = self.conv_51(x)
        #x4 = self.norm_51(x4)
        #x4 = self.conv_52(x4)
        #x4 = self.norm_52(x4)
        #x4 = self.act_52(x4)
        
        #Concat all atrous receptive heads
        #e = torch.cat([x1, x2, x3, x4], dim=1)  
        e = torch.cat([x1, x2, x3], dim=1)  
        
        e = self.conv_61(e)
        e = self.norm_61(e)
        e = self.act_61(e)
        
        e = self.conv_71(e)
        e = self.norm_71(e)
        e = self.act_71(e)
        
        #Residual skip for preseving some details from first processing 
        return e + x



class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #Downsample
        self.conv_1 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 2, padding = 1)
        self.norm_1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.act_1 = nn.LeakyReLU(0.2, inplace=True)
        
        #Refine
        self.conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
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
        #Downsample
        self.conv_1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size = 3, stride = 2, padding = 1, output_padding=1)
        self.norm_1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.act_1 = nn.LeakyReLU(0.2, inplace=True)
        
        #Refine
        self.conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
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
    


# Generator with skip connections & StyleGAN2ResBlocks
class Inpainter_v0(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, n_residual=4, base_filters=32):
        super().__init__()
        f1 = base_filters
        f2 = base_filters * 2
        f3 = base_filters * 3
        f4 = base_filters * 4

        # Encoder
        self.input_conv_block = nn.Sequential(nn.Conv2d(input_channels, f1, kernel_size = 7, stride = 1, padding = 3),
                                              nn.InstanceNorm2d(f1, affine=True),
                                              nn.LeakyReLU(0.2, inplace=True),
                                            
                                              nn.Conv2d(f1, f1, kernel_size = 3, stride = 1, padding = 1),
                                              nn.InstanceNorm2d(f1, affine=True),
                                              nn.LeakyReLU(0.2, inplace=True)
                                              )
        
        #Encoder parts
        self.enc_0 = Processing_Block(f1,f1)
        self.down_0 = Downsample_block(f1,f2)
        
        self.enc_1 = Processing_Block(f2,f2)
        self.down_1 = Downsample_block(f2,f3)
        
        self.enc_2 = Processing_Block(f3,f3)
        self.down_2 = Downsample_block(f3,f4)
        
        
        
        #Residual bottleneck
        self.resblocks = nn.Sequential(*[StyleGAN2ResBlock(f4) for _ in range(n_residual)])

        #Decoder parts
        self.up_2 = Upsample_block(f4, f3)
        self.dec_2 = Processing_Block(int(f3*2), f3)
        
        self.up_1 = Upsample_block(f3, f2)
        self.dec_1 = Processing_Block(int(f2*2), f2)
        
        self.up_0 = Upsample_block(f2, f1)
        self.dec_0 = Processing_Block(int(f1*2), f1)



        #Output conv block
        self.output_conv = nn.Sequential(

                                        
                                        # Dilated convs for larger receptive field
                                        nn.Conv2d(f1, f1, kernel_size=3, stride=1, padding=3, dilation=3),
                                        nn.InstanceNorm2d(f1, affine=True),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        
                                        nn.Conv2d(f1, f1, kernel_size=3, stride=1, padding=2, dilation=2),
                                        nn.InstanceNorm2d(f1, affine=True),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        
                                        #Conv to consolidate features and img
                                        nn.Conv2d(f1, f1, kernel_size=3, stride=1, padding=1),
                                        nn.InstanceNorm2d(f1, affine=True),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        
                                        nn.Conv2d(f1, f1, kernel_size=3, stride=1, padding=1),
                                        nn.InstanceNorm2d(f1, affine=True),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        
                                        nn.Conv2d(f1, f1, kernel_size=3, stride=1, padding=2),
                                        nn.InstanceNorm2d(f1, affine=True),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        
                                        # Final output conv: no padding, directly outputs final image
                                        nn.Conv2d(f1, output_channels, kernel_size=7, stride=1, padding=2),
                                        nn.Tanh()
                                    )

    def forward(self, x):
        #Input conv block
        x = self.input_conv_block(x)
        
        #Encoder part
        x_0 = self.enc_0(x)
        x = self.down_0(x_0)
        
        x_1 = self.enc_1(x)
        x = self.down_1(x_1)
        
        x_2 = self.enc_2(x)
        x = self.down_2(x_2)
        
        #Res bottleneck
        x = self.resblocks(x)
        
        #Decoder part
        x = self.up_2(x)
        x_2 = torch.cat([x, x_2], dim=1)  #Concat skip connection from encoder 2
        x = self.dec_2(x_2)
        
        x = self.up_1(x)
        x_1 = torch.cat([x, x_1], dim=1)  #Concat skip connection from encoder 1
        x = self.dec_1(x_1)
        
        x = self.up_0(x)
        x_0 = torch.cat([x, x_0], dim=1)  #Concat skip connection from encoder 0
        x = self.dec_0(x_0)
        
        x = self.output_conv(x)
        
        

        return x
    
#############################################################
#Encoder for feature extraction
#############################################################

class Encoder_v0(nn.Module):
    def __init__(self, input_channels=3, base_filters=32):
        super().__init__()
        f1 = base_filters
        f2 = base_filters * 2
        f3 = base_filters * 3
        f4 = base_filters * 4

        # Encoder
        self.input_conv_block = nn.Sequential(nn.Conv2d(input_channels, f1, kernel_size = 7, stride = 1, padding = 3),
                                              nn.InstanceNorm2d(f1, affine=True),
                                              nn.LeakyReLU(0.2, inplace=True),
                                            
                                              nn.Conv2d(f1, f1, kernel_size = 3, stride = 1, padding = 1),
                                              nn.InstanceNorm2d(f1, affine=True),
                                              nn.LeakyReLU(0.2, inplace=True)
                                              )
        
        #Encoder parts
        self.enc_0 = Processing_Block(f1,f1)
        self.down_0 = Downsample_block(f1,f2)
        
        self.enc_1 = Processing_Block(f2,f2)
        self.down_1 = Downsample_block(f2,f3)
        
        self.enc_2 = Processing_Block(f3,f3)
        self.down_2 = Downsample_block(f3,f4)
        

    def forward(self, x):
        #Input conv block
        x = self.input_conv_block(x)
        
        #Encoder part
        x_0 = self.enc_0(x)
        x = self.down_0(x_0)
        
        x_1 = self.enc_1(x)
        x = self.down_1(x_1)
        
        x_2 = self.enc_2(x)
        x = self.down_2(x_2)
        
        return x