from STORM.图像去噪.model import UNet3D, Noise2NoiseUNet3D, FCAUNet3D, UNet3D4, UNet3D3, UNet3D2, UNet_2d, \
    Noise2NoiseUNet_2d, UNet2_2d, UNet2_attention_v1_1_2, UNet2_attention_crg, UNet2_attention_clg, DnCNN

import torch.nn as nn

class Network_3D_Unet(nn.Module):
    def __init__(self, UNet_type = 'UNet', in_channels=1, out_channels=1, f_maps=64, final_sigmoid = True):
        super(Network_3D_Unet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_sigmoid = final_sigmoid
        self.UNet_type = UNet_type

        if UNet_type == 'UNet':
            self.Generator = UNet_2d( in_channels = in_channels,
                                     out_channels = out_channels,
                                     f_maps = f_maps,
                                     final_sigmoid = final_sigmoid) 
        elif UNet_type == 'UNet4':
            self.Generator = UNet3D4( in_channels = in_channels,
                                     out_channels = out_channels,
                                     f_maps = f_maps,
                                     final_sigmoid = final_sigmoid)   
        elif UNet_type == 'UNetn2n':
            self.Generator = Noise2NoiseUNet_2d( in_channels = in_channels,
                                                out_channels = out_channels,
                                                f_maps = f_maps,
                                                final_sigmoid = final_sigmoid)    
        elif UNet_type == 'UNetnfca':
            self.Generator = FCAUNet3D( in_channels = in_channels,
                                        out_channels = out_channels,
                                        f_maps = f_maps,
                                        final_sigmoid = final_sigmoid)  
        elif UNet_type == 'UNet3':
            self.Generator = UNet3D3( in_channels = in_channels,
                                     out_channels = out_channels,
                                     f_maps = f_maps,
                                     final_sigmoid = final_sigmoid)
        elif UNet_type == 'UNet2_2d':
            self.Generator = UNet2_2d( in_channels = in_channels,
                                     out_channels = out_channels,
                                     f_maps = f_maps,
                                     final_sigmoid = final_sigmoid)
        elif UNet_type == 'UNet2_attention_v1_1_2':
            self.Generator = UNet2_attention_v1_1_2( in_channels = in_channels,
                                     out_channels = out_channels,
                                     f_maps = f_maps,
                                     final_sigmoid = final_sigmoid)   
        elif UNet_type == 'UNet2_attention_crg':
            self.Generator = UNet2_attention_crg( in_channels = in_channels,
                                     out_channels = out_channels,
                                     f_maps = f_maps,
                                     final_sigmoid = final_sigmoid)
        elif UNet_type == 'UNet2_attention_clg':
            self.Generator = UNet2_attention_clg( in_channels = in_channels,
                                     out_channels = out_channels,
                                     f_maps = f_maps,
                                     final_sigmoid = final_sigmoid)
        elif UNet_type == 'DnCNN':
            self.Generator = DnCNN( in_channels = in_channels,
                                     out_channels = out_channels,
                                     f_maps = f_maps,
                                     final_sigmoid = final_sigmoid)                                                 

    def forward(self, x):
        print(self.UNet_type)
        fake_x = self.Generator(x)
        # if self.UNet_type == 'DnCNN':
        #     return x - fake_x
        # else:
        return fake_x


