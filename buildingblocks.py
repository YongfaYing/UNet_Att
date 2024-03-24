import torch
from torch import nn as nn
from torch.nn import functional as F
#import cv2
import numpy as np
from torch.autograd import Variable
from grid_attention_layer import GridAttentionBlock3D


def conv3d(in_channels, out_channels, kernel_size, bias, padding=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=1):
    #这是创建了一个模块，通过改变order，可以更换多种模式
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int): add zero-padding to the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"    #assert的作用是只有符合这条命令才能运行，不符合就会报错
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'  #激活函数不能在第一层

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of gatchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)   #当order中没有g和b时，bias为True。bias就是和weight一起出现的那个bias
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))   #三维卷积层
        elif char == 'g':
            is_before_conv = i < order.index('c')  #True或False
            assert not is_before_conv, 'GroupNorm MUST go after the Conv3d'  #意思是g必须在c的后面，不然会报错
            # number of groups must be less or equal the number of channels
            if out_channels < num_groups:
                num_groups = out_channels
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)))  #将channel切分成很多组进行归一化
        elif char == 'b':
            is_before_conv = i < order.index('c')  
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules
#假如order为'cr'，则modules=[('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)) , ('ReLU', nn.ReLU(inplace=True))]
#既包含了需要的层运算，又包括了它们的变量名称，方便代入实际数据x=modules[0][0](input)

class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cr', num_groups=8, padding=1):   #'cr' -> conv + ReLU
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)  #实际情况是先进行了一次卷积层，又进行了一次ReLU激活


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='cr', num_groups=8):
        super(DoubleConv, self).__init__()
        if encoder:   #实际情况是第一层卷积层通道数不变，第二层卷积层通道数翻倍
            # we're in the encoder path
            conv1_in_channels = in_channels    
            conv1_out_channels = out_channels // 2    #当i=1时，in_channels=64,out_channels=128,也就是第一层卷积层输入输出通道数不变
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:       #实际情况是第一层卷积层通道数减半，第二层卷积层通道数不变
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        #有两个连续卷积层
        # conv1
        self.add_module('SingleConv1',    #add_module可以增加子模块或者替换子模块
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups))


class ExtResNetBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, **kwargs):
        super(ExtResNetBlock, self).__init__()

        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order,
                                num_groups=num_groups)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out

        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (tuple): the size of the window to take a max over
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=(2, 2, 2), pool_type='max', basic_module=DoubleConv, conv_layer_order='cr',
                 num_groups=8):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)  #池化中s,h,w每个维度的步幅都是2
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, x):   #这里才有x输入卷积层
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x      #池化-->卷积-->relu-->卷积-->relu
'''
def FCA(input,size_wh=64):
    input=input.cuda().data.cpu().numpy()  #将Variable张量转化为numpy
    fft=np.fft.fftn(input,axes=(3,4))  #傅里叶变换
    fft=torch.tensor(fft)
    absfft=pow(abs(fft)+1e-8,0.1)
    bs, ch, z, h, w = absfft.get_shape().as_list()
    fs111 = absfft[:, -h // 2:h, -w // 2:w, -z // 2 + 1:z, :]
    fs121 = absfft[:, -h // 2:h, 0:w // 2, -z // 2 + 1:z, :]
    fs211 = absfft[:, 0:h // 2, -w // 2:w, -z // 2 + 1:z, :]
    fs221 = absfft[:, 0:h // 2, 0:w // 2, -z // 2 + 1:z, :]
    fs112 = absfft[:, -h // 2:h, -w // 2:w, 0:z // 2 + 1, :]
    fs122 = absfft[:, -h // 2:h, 0:w // 2, 0:z // 2 + 1, :]
    fs212 = absfft[:, 0:h // 2, -w // 2:w, 0:z // 2 + 1, :]
    fs222 = absfft[:, 0:h // 2, 0:w // 2, 0:z // 2 + 1, :]
    output1 = torch.cat([torch.cat([fs111, fs211], axis=3), torch.cat([fs121, fs221], axis=3)], axis=4)
    output2 = torch.cat([torch.cat([fs112, fs212], axis=3), torch.cat([fs122, fs222], axis=1)], axis=4)
    output0 = torch.cat([output1, output2], axis=2)
    output = []
    for iz in range(z):
        output.append(cv2.resize(output0[:, :, :, iz, :], (size_wh, size_wh), interpolation=cv2.INTER_NEAREST))
    return output
'''

def FCA(input,size_wh=64):
    #input=input.cuda().data.cpu().numpy()  #将Variable张量转化为numpy
    input=input.cuda().data.cpu()   #因为不能从cuda取数，只能将它先转到cpu
    input=torch.tensor(input,dtype=torch.float32)
    #试试input=torch.tensor(input,dtype=torch.cuda.float32)
    input=torch.complex(input,torch.zeros(input.size()))
    fft=np.fft.fftn(input,axes=(3,4))  #傅里叶变换
    absfft=pow(abs(fft)+1e-8,0.1)    #运算后，复数变成浮点数
    output=np.fft.fftshift(absfft, axes=(3,4))
    output=torch.tensor(output,dtype=torch.float32)  #把numpy变成tensor，为了把它再重新传到cuda
    #试试output=torch.tensor(output,dtype=torch.cuda.float32)
    output=output.cuda()
    return output

'''
def FCA(input,size_wh=64):
    #input=input.cuda().data.cpu().numpy()  #将Variable张量转化为numpy
    #input=input.cuda().data.cpu()   #因为不能从cuda取数，只能将它先转到cpu
    #input=torch.tensor(input,dtype=torch.float32)
    #input=torch.from_numpy(input)
    input=input.cpu().detach().numpy()
    #input=input.numpy()
    #试试input=torch.tensor(input,dtype=torch.cuda.float32)
    #input=torch.complex(input,torch.zeros(input.size()))
    fft=np.fft.fftn(input,axes=(2,3,4))  #傅里叶变换
    absfft=pow(abs(fft)+1e-8,0.1)    #运算后，复数变成浮点数
    output=np.fft.fftshift(absfft, axes=(2,3,4)).astype('float32')
    output=torch.from_numpy(output)  #把numpy变成tensor，为了把它再重新传到cuda
    #试试output=torch.tensor(output,dtype=torch.cuda.float32)
    output=output.cuda()
    return output
'''
class Encoder_FCA(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=(2, 2, 2), pool_type='max', basic_module=DoubleConv, conv_layer_order='cr',
                 num_groups=8):
        super(Encoder_FCA, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)  #池化中s,h,w每个维度的步幅都是2
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None
        
        self.fca = basic_module(1, 1,
                                encoder=True,
                                kernel_size=conv_kernel_size,
                                order=conv_layer_order,
                                num_groups=num_groups)
        self.conv1=SingleConv(1, in_channels,kernel_size=conv_kernel_size, order=conv_layer_order, num_groups=num_groups)
        self.pooling1=nn.MaxPool3d(kernel_size=pool_kernel_size)
        self.conv2=SingleConv(in_channels, in_channels,kernel_size=1, order=conv_layer_order, num_groups=num_groups,padding=0)
        self.conv3=SingleConv(in_channels, in_channels,kernel_size=1, order='c', num_groups=num_groups,padding=0)
        self.sig_=nn.Sigmoid()    #要先有实例，不能直接代入数据
        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)


    def forward(self, x,fca):   #这里才有x输入卷积层
        if self.pooling is not None:
            fca = self.pooling(fca) 
        fca=self.fca(fca)
        #x = self.pooling1(x)
        #x=self.conv2(x)
        #x=self.conv3(x)
        fca=self.sig_(fca)
        #x=cv2.multiply(input,x)
        
        if self.pooling is not None:
            x = self.pooling(x)
        x=torch.mul(x,fca)
        x = self.basic_module(x)
        return x,fca    


class Decoder_FCA(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose3d or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order='cr', num_groups=8):
        super(Decoder_FCA, self).__init__()
        if basic_module == DoubleConv:
            # if DoubleConv is the basic_module use nearest neighbor interpolation for upsampling
            self.upsample = None   #使用邻插值进行上采样
        else:
            # otherwise use ConvTranspose3d (bear in mind your GPU memory)
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            # also scale the number of channels from in_channels to out_channels so that summation joining
            # works correctly
            self.upsample = nn.ConvTranspose3d(in_channels,     #转置卷积，可做上采样
                                               out_channels,
                                               kernel_size=kernel_size,
                                               stride=scale_factor,
                                               padding=1,
                                               output_padding=1)
            # adapt the number of in_channels for the ExtResNetBlock
            in_channels = out_channels

        ##若要拼接时使通道数相同，可在这里增加一个卷积层，使通道数减半，并在forward中实施运行，并把UNet3D中decoder的in_feature_num改变一下
        ##比如self.channel_conv=nn.Conv3d(in_channels,out_channels=out_channels/2,kernel_size=kernel_size,padding=1)
        
        self.conv1=SingleConv(out_channels, out_channels,kernel_size=3, order=conv_layer_order, num_groups=num_groups)
        self.pooling1=nn.MaxPool3d(kernel_size=scale_factor)
        self.conv2=SingleConv(out_channels, out_channels,kernel_size=1, order=conv_layer_order, num_groups=num_groups,padding=0)
        self.conv3=SingleConv(out_channels, out_channels,kernel_size=1, order='c', num_groups=num_groups,padding=0)
        self.sig_=nn.Sigmoid()    #要先有实例，不能直接代入数据
        
        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)


    def forward(self, encoder_features, x):
        if self.upsample is None:
            # use nearest neighbor interpolation and concatenation joining
            #encoder_features是在类UNet3D中forward函数中定义并实际运行的
            output_size = encoder_features.size()[2:]   #上采样插值的shape由对应的编码层决定，所以论文中才能将(37,8,8)变成(75,16,16)
            x = F.interpolate(x, size=output_size, mode='nearest')   #mode='nearest'临近值插值，就像复制，和mu-net上采样插值方法一样
            # concatenate encoder_features (encoder path) with the upsampled input across channel dimension
            ##若要通道数相同加一个x=self.channel_conv(x)
            x = torch.cat((encoder_features, x), dim=1)   #在'1'维度拼接，也就是通道维度
        else:
            # use ConvTranspose3d and summation joining
            x = self.upsample(x)
            x += encoder_features

        x = self.basic_module(x)  #上采样并且拼接后，进行两次卷积层和激活

        output=FCA(x)
        output=self.conv1(output)
        #x = self.pooling1(x)
        output=self.conv2(output)
        #x=self.conv3(x)
        output=self.sig_(output)
        #x=cv2.multiply(input,x)
        x=torch.mul(x,output)
        return x
        

class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose3d or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order='cr', num_groups=8):
        super(Decoder, self).__init__()
        if basic_module == DoubleConv:
            # if DoubleConv is the basic_module use nearest neighbor interpolation for upsampling
            self.upsample = None   #使用邻插值进行上采样
        else:
            # otherwise use ConvTranspose3d (bear in mind your GPU memory)
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            # also scale the number of channels from in_channels to out_channels so that summation joining
            # works correctly
            self.upsample = nn.ConvTranspose3d(in_channels,     #转置卷积，可做上采样
                                               out_channels,
                                               kernel_size=kernel_size,
                                               stride=scale_factor,
                                               padding=1,
                                               output_padding=1)
            # adapt the number of in_channels for the ExtResNetBlock
            in_channels = out_channels

        ##若要拼接时使通道数相同，可在这里增加一个卷积层，使通道数减半，并在forward中实施运行，并把UNet3D中decoder的in_feature_num改变一下
        ##比如self.channel_conv=nn.Conv3d(in_channels,out_channels=out_channels/2,kernel_size=kernel_size,padding=1)

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)


    def forward(self, encoder_features, x):
        if self.upsample is None:
            # use nearest neighbor interpolation and concatenation joining
            #encoder_features是在类UNet3D中forward函数中定义并实际运行的
            output_size = encoder_features.size()[2:]   #上采样插值的shape由对应的编码层决定，所以论文中才能将(37,8,8)变成(75,16,16)
            x = F.interpolate(x, size=output_size, mode='nearest')   #mode='nearest'临近值插值，就像复制，和mu-net上采样插值方法一样
            # concatenate encoder_features (encoder path) with the upsampled input across channel dimension
            ##若要通道数相同加一个x=self.channel_conv(x)
            x = torch.cat((encoder_features, x), dim=1)   #在'1'维度拼接，也就是通道维度
        else:
            # use ConvTranspose3d and summation joining
            x = self.upsample(x)
            x += encoder_features

        x = self.basic_module(x)  #上采样并且拼接后，进行两次卷积层和激活
        return x

class Decoder2(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order='cr', num_groups=8):
        super(Decoder2, self).__init__()
        
        ##若要拼接时使通道数相同，可在这里增加一个卷积层，使通道数减半，并在forward中实施运行，并把UNet3D中decoder的in_feature_num改变一下
        ##比如self.channel_conv=nn.Conv3d(in_channels,out_channels=out_channels/2,kernel_size=kernel_size,padding=1)

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)


    def forward(self, encoders_features):
        if len(encoders_features)==4:
            x1=encoders_features[0]
            x2=F.interpolate(encoders_features[1], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==5:
            x1=encoders_features[1]
            x2=F.interpolate(encoders_features[2], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==6:
            x1=encoders_features[2]
            x2=F.interpolate(encoders_features[3], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==7:
            x1=encoders_features[0]
            x2=encoders_features[4]
            x3=F.interpolate(encoders_features[5], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2,x3), dim=1)
        elif len(encoders_features)==8:
            x1=encoders_features[1]
            x2=encoders_features[5]
            x3=F.interpolate(encoders_features[6], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2,x3), dim=1)
        elif len(encoders_features)==9:
            x1=encoders_features[0]
            x2=encoders_features[4]
            x3=encoders_features[7]
            x4=F.interpolate(encoders_features[8], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2,x3,x4), dim=1)
        #encoders_features是在类UNet3D中forward函数中定义并实际运行的
       
        x = self.basic_module(x)  #上采样并且拼接后，进行两次卷积层和激活
        return x
        
class Decoder2_attention(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order='cr', num_groups=8):
        super(Decoder2_attention, self).__init__()
        
        ##若要拼接时使通道数相同，可在这里增加一个卷积层，使通道数减半，并在forward中实施运行，并把UNet3D中decoder的in_feature_num改变一下
        ##比如self.channel_conv=nn.Conv3d(in_channels,out_channels=out_channels/2,kernel_size=kernel_size,padding=1)

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)


    def forward(self, encoders_features, attention_features):
        if len(encoders_features)==4:
            x1=attention_features[0]
            x2=F.interpolate(encoders_features[1], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==5:
            x1=attention_features[1]
            x2=F.interpolate(encoders_features[2], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==6:
            x1=attention_features[2]
            x2=F.interpolate(encoders_features[3], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==7:
            x1=attention_features[0]
            x2=attention_features[3]
            x3=F.interpolate(encoders_features[5], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2,x3), dim=1)
        elif len(encoders_features)==8:
            x1=attention_features[1]
            x2=attention_features[4]
            x3=F.interpolate(encoders_features[6], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2,x3), dim=1)
        elif len(encoders_features)==9:
            x1=attention_features[0]
            x2=attention_features[3]
            x3=attention_features[5]
            x4=F.interpolate(encoders_features[8], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2,x3,x4), dim=1)
        #encoders_features是在类UNet3D中forward函数中定义并实际运行的
       
        x = self.basic_module(x)  #上采样并且拼接后，进行两次卷积层和激活
        return x
        
class Decoder3(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose3d or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order='cr', num_groups=8):
        super(Decoder3, self).__init__()
        
        ##若要拼接时使通道数相同，可在这里增加一个卷积层，使通道数减半，并在forward中实施运行，并把UNet3D中decoder的in_feature_num改变一下
        ##比如self.channel_conv=nn.Conv3d(in_channels,out_channels=out_channels/2,kernel_size=kernel_size,padding=1)

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)


    def forward(self, encoders_features, x):
        if len(encoders_features)==4:
            x1=encoders_features[-2]
            x2=F.interpolate(encoders_features[1], scale_factor=0.5, mode='nearest')
            x3=F.interpolate(encoders_features[0], scale_factor=0.25, mode='nearest')
        elif len(encoders_features)==5:
            x1=F.interpolate(encoders_features[-2], scale_factor=4, mode='nearest')
            x2=encoders_features[1]
            x3=F.interpolate(encoders_features[0], scale_factor=0.5, mode='nearest')
        elif len(encoders_features)==6:
            x1=F.interpolate(encoders_features[-2], scale_factor=4, mode='nearest')
            x2=F.interpolate(encoders_features[-3], scale_factor=8, mode='nearest')
            x3=encoders_features[0]
        #encoders_features是在类UNet3D中forward函数中定义并实际运行的

        x = F.interpolate(x, scale_factor=2, mode='nearest')   #mode='nearest'临近值插值，就像复制，和mu-net上采样插值方法一样
        # concatenate encoder_features (encoder path) with the upsampled input across channel dimension
        ##若要通道数相同加一个x=self.channel_conv(x)
        x = torch.cat((x,x1,x2,x3), dim=1)   #在'1'维度拼接，也就是通道维度
       
        x = self.basic_module(x)  #上采样并且拼接后，进行两次卷积层和激活
        return x

class Decoder4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order='cr', num_groups=8):
        super(Decoder4, self).__init__()
        
        ##若要拼接时使通道数相同，可在这里增加一个卷积层，使通道数减半，并在forward中实施运行，并把UNet3D中decoder的in_feature_num改变一下
        ##比如self.channel_conv=nn.Conv3d(in_channels,out_channels=out_channels/2,kernel_size=kernel_size,padding=1)
        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self,encoders_features, x):
        if len(encoders_features)==4:
            x1=encoders_features[-2]
            x2=F.interpolate(encoders_features[-4], scale_factor=0.25, mode='nearest')
        elif len(encoders_features)==5:
            x1=F.interpolate(encoders_features[-2], scale_factor=4, mode='nearest')
            x2=encoders_features[-4]
        elif len(encoders_features)==6:
            x1=F.interpolate(encoders_features[-2], scale_factor=4, mode='nearest')
            x2=F.interpolate(encoders_features[-4], scale_factor=4, mode='nearest')
        #encoders_features是在类UNet3D中forward函数中定义并实际运行的

        x = F.interpolate(x, scale_factor=2, mode='nearest')   #mode='nearest'临近值插值，就像复制，和mu-net上采样插值方法一样
        # concatenate encoder_features (encoder path) with the upsampled input across channel dimension
        ##若要通道数相同加一个x=self.channel_conv(x)
        x = torch.cat((x,x1,x2), dim=1)   #在'1'维度拼接，也就是通道维度
       
        x = self.basic_module(x)  #上采样并且拼接后，进行两次卷积层和激活
        return x

class GridAttentionBlockND(nn.Module):  #注意力模块
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=2, mode='concatenation',
                 sub_sample_factor=(2,2)):
        super(GridAttentionBlockND, self).__init__()       

        assert dimension in [2, 3]  #三维数据或者二维数据
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        #self.sub_sample_kernel_size = self.sub_sample_factor
        self.sub_sample_kernel_size = 3

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'   #三线性插值
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        # Initialise weights
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output
    
    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_debug(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.softplus(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


    def _concatenation_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = F.softmax(f, dim=2).view(batch_size, 1, *theta_x.size()[2:])

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f
        

class FinalConv(nn.Sequential):
    """
    A module consisting of a convolution layer (e.g. Conv3d+ReLU+GroupNorm3d) and the final 1x1 convolution
    which reduces the number of channels to 'out_channels'.
    with the number of output channels 'out_channels // 2' and 'out_channels' respectively.
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be change however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ReLU use order='cbr'.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cr', num_groups=8):
        super(FinalConv, self).__init__()

        # conv1
        self.add_module('SingleConv', SingleConv(in_channels, in_channels, kernel_size, order, num_groups))

        # in the last layer a 1×1 convolution reduces the number of output channels to out_channels
        final_conv = nn.Conv3d(in_channels, out_channels, 1)
        self.add_module('final_conv', final_conv)


##################################################### 2D #####################################################
def conv2d(in_channels, out_channels, kernel_size, bias, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def create_conv_2d(in_channels, out_channels, kernel_size, order, num_groups, padding=1):
    #这是创建了一个模块，通过改变order，可以更换多种模式
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int): add zero-padding to the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"    #assert的作用是只有符合这条命令才能运行，不符合就会报错
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'  #激活函数不能在第一层

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of gatchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)   #当order中没有g和b时，bias为True。bias就是和weight一起出现的那个bias
            modules.append(('conv', conv2d(in_channels, out_channels, kernel_size, bias, padding=padding)))   #三维卷积层
        elif char == 'g':
            is_before_conv = i < order.index('c')  #True或False
            assert not is_before_conv, 'GroupNorm MUST go after the Conv2d'  #意思是g必须在c的后面，不然会报错
            # number of groups must be less or equal the number of channels
            if out_channels < num_groups:
                num_groups = out_channels
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)))  #将channel切分成很多组进行归一化
        elif char == 'b':
            is_before_conv = i < order.index('c')  
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm2d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm2d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules
#假如order为'cr'，则modules=[('conv', conv2d(in_channels, out_channels, kernel_size, bias, padding=padding)) , ('ReLU', nn.ReLU(inplace=True))]
#既包含了需要的层运算，又包括了它们的变量名称，方便代入实际数据x=modules[0][0](input)

class SingleConv_2d(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv2d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='crb', num_groups=8, padding=1):   #'cr' -> conv + ReLU
        super(SingleConv_2d, self).__init__()

        for name, module in create_conv_2d(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)  #实际情况是先进行了一次卷积层，又进行了一次ReLU激活


class DoubleConv_2d(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='crb', num_groups=8):
        super(DoubleConv_2d, self).__init__()
        if encoder:   #实际情况是第一层卷积层通道数不变，第二层卷积层通道数翻倍
            # we're in the encoder path
            conv1_in_channels = in_channels    
            conv1_out_channels = out_channels // 2    #当i=1时，in_channels=64,out_channels=128,也就是第一层卷积层输入输出通道数不变
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:       #实际情况是第一层卷积层通道数减半，第二层卷积层通道数不变
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        #有两个连续卷积层
        # conv1
        self.add_module('SingleConv1',    #add_module可以增加子模块或者替换子模块
                        SingleConv_2d(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv_2d(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups))

class Encoder_2d(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (tuple): the size of the window to take a max over
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=(2, 2), pool_type='max', basic_module=DoubleConv_2d, conv_layer_order='crb',
                 num_groups=8):
        super(Encoder_2d, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool2d(kernel_size=pool_kernel_size)  #池化中s,h,w每个维度的步幅都是2
            else:
                self.pooling = nn.AvgPool2d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, x):   #这里才有x输入卷积层
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x

class Decoder_2d(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose3d or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2), basic_module=DoubleConv_2d, conv_layer_order='crb', num_groups=8):
        super(Decoder_2d, self).__init__()
        if basic_module == DoubleConv_2d:
            # if DoubleConv is the basic_module use nearest neighbor interpolation for upsampling
            self.upsample = None   #使用邻插值进行上采样
        else:
            # otherwise use ConvTranspose3d (bear in mind your GPU memory)
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            # also scale the number of channels from in_channels to out_channels so that summation joining
            # works correctly
            self.upsample = nn.ConvTranspose2d(in_channels,     #转置卷积，可做上采样
                                               out_channels,
                                               kernel_size=kernel_size,
                                               stride=scale_factor,
                                               padding=1,
                                               output_padding=1)
            # adapt the number of in_channels for the ExtResNetBlock
            in_channels = out_channels

        ##若要拼接时使通道数相同，可在这里增加一个卷积层，使通道数减半，并在forward中实施运行，并把UNet3D中decoder的in_feature_num改变一下
        ##比如self.channel_conv=nn.Conv3d(in_channels,out_channels=out_channels/2,kernel_size=kernel_size,padding=1)

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)


    def forward(self, encoder_features, x):
        if self.upsample is None:
            # use nearest neighbor interpolation and concatenation joining
            #encoder_features是在类UNet3D中forward函数中定义并实际运行的
            output_size = encoder_features.size()[2:]   #上采样插值的shape由对应的编码层决定，所以论文中才能将(37,8,8)变成(75,16,16)
            x = F.interpolate(x, size=output_size, mode='nearest')   #mode='nearest'临近值插值，就像复制，和mu-net上采样插值方法一样
            # concatenate encoder_features (encoder path) with the upsampled input across channel dimension
            ##若要通道数相同加一个x=self.channel_conv(x)
            x = torch.cat((encoder_features, x), dim=1)   #在'1'维度拼接，也就是通道维度
        else:
            # use ConvTranspose3d and summation joining
            x = self.upsample(x)
            x += encoder_features

        x = self.basic_module(x)  #上采样并且拼接后，进行两次卷积层和激活
        return x

class Decoder2_2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2), basic_module=DoubleConv_2d, conv_layer_order='crb', num_groups=8):
        super(Decoder2_2d, self).__init__()
        
        ##若要拼接时使通道数相同，可在这里增加一个卷积层，使通道数减半，并在forward中实施运行，并把UNet3D中decoder的in_feature_num改变一下
        ##比如self.channel_conv=nn.Conv3d(in_channels,out_channels=out_channels/2,kernel_size=kernel_size,padding=1)

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)


    def forward(self, encoders_features):
        if len(encoders_features)==4:
            x1=encoders_features[0]
            x2=F.interpolate(encoders_features[1], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==5:
            x1=encoders_features[1]
            x2=F.interpolate(encoders_features[2], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==6:
            x1=encoders_features[2]
            x2=F.interpolate(encoders_features[3], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==7:
            x1=encoders_features[0]
            x2=encoders_features[4]
            x3=F.interpolate(encoders_features[5], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2,x3), dim=1)
        elif len(encoders_features)==8:
            x1=encoders_features[1]
            x2=encoders_features[5]
            x3=F.interpolate(encoders_features[6], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2,x3), dim=1)
        elif len(encoders_features)==9:
            x1=encoders_features[0]
            x2=encoders_features[4]
            x3=encoders_features[7]
            x4=F.interpolate(encoders_features[8], scale_factor=2, mode='nearest')
            x = torch.cat((x1,x2,x3,x4), dim=1)
        #encoders_features是在类UNet3D中forward函数中定义并实际运行的
       
        x = self.basic_module(x)  #上采样并且拼接后，进行两次卷积层和激活
        return x

class Decoder2_attention_bilinear_2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2), basic_module=DoubleConv_2d, conv_layer_order='crb', num_groups=8):
        super(Decoder2_attention_bilinear_2d, self).__init__()
        
        ##若要拼接时使通道数相同，可在这里增加一个卷积层，使通道数减半，并在forward中实施运行，并把UNet3D中decoder的in_feature_num改变一下
        ##比如self.channel_conv=nn.Conv3d(in_channels,out_channels=out_channels/2,kernel_size=kernel_size,padding=1)

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)


    def forward(self, encoders_features, attention_features):
        if len(encoders_features)==4:
            x1=attention_features[0]
            x2=F.interpolate(encoders_features[1], scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==5:
            x1=attention_features[1]
            x2=F.interpolate(encoders_features[2], scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==6:
            x1=attention_features[2]
            x2=F.interpolate(encoders_features[3], scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat((x1,x2), dim=1)
        elif len(encoders_features)==7:
            x1=attention_features[0]
            x2=attention_features[3]
            x3=F.interpolate(encoders_features[5], scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat((x1,x2,x3), dim=1)
        elif len(encoders_features)==8:
            x1=attention_features[1]
            x2=attention_features[4]
            x3=F.interpolate(encoders_features[6], scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat((x1,x2,x3), dim=1)
        elif len(encoders_features)==9:
            x1=attention_features[0]
            x2=attention_features[3]
            x3=attention_features[5]
            x4=F.interpolate(encoders_features[8], scale_factor=2, mode='bilinear', align_corners=True)
            x = torch.cat((x1,x2,x3,x4), dim=1)
        #encoders_features是在类UNet3D中forward函数中定义并实际运行的
       
        x = self.basic_module(x)  #上采样并且拼接后，进行两次卷积层和激活
        return x
