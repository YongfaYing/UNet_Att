import torch
import torch.nn as nn
import torch.nn.functional as F
#from init_weights import init_weights


class unetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=1, ks=3, stride=1, padding=1):
        super(unetConv3, self).__init__()
        self.n = n
        self.ks = ks   #卷积核尺寸
        self.stride = stride  #步长
        self.padding = padding  #填充
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm3d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        #for m in self.children():
        #    init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = unetConv3(out_size*2, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear3d(scale_factor=2)

        # initialise the blocks
        #for m in self.children():
        #    if m.__class__.__name__.find('unetConv3') != -1: continue
        #    init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)
    
class unetUp_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp_origin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = unetConv3(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = unetConv3(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear3d(scale_factor=2)

        # initialise the blocks
        #for m in self.children():
        #    if m.__class__.__name__.find('unetConv3') != -1: continue
        #    init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)
