import importlib

import torch
import torch.nn as nn

from buildingblocks import Encoder, Decoder, Decoder2, Decoder2_attention, Decoder3, Decoder4, FinalConv, DoubleConv, ExtResNetBlock, SingleConv, Encoder_FCA, Decoder_FCA, Encoder_2d, Decoder_2d, Decoder2_2d, SingleConv_2d, DoubleConv_2d, Decoder2_attention_bilinear_2d, GridAttentionBlockND
from utils import create_feature_maps
from grid_attention_layer import GridAttentionBlock3D

class UNet3D(nn.Module):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet3D, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        # if not self.training:
            # x = self.final_activation(x)

        return x

class UNet3D_attention(nn.Module):
    
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet3D_attention, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # attention blocks 
        self.attentionblock2 = GridAttentionBlock3D(in_channels=f_maps[1], gating_channels=f_maps[3],
                                                    inter_channels=f_maps[1], sub_sample_factor=(2,2,2), mode='concatenation')
        self.attentionblock3 = GridAttentionBlock3D(in_channels=f_maps[2], gating_channels=f_maps[3],
                                                    inter_channels=f_maps[2], sub_sample_factor=(2,2,2), mode='concatenation')
        self.attentionblock4 = GridAttentionBlock3D(in_channels=f_maps[3], gating_channels=f_maps[3],
                                                    inter_channels=f_maps[3], sub_sample_factor=(2,2,2), mode='concatenation')
        attentionblocks = [self.attentionblock4, self.attentionblock3, self.attentionblock2]
        self.attentionblocks = nn.ModuleList(attentionblocks)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features_all = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features_all.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features_all[1:]
        gating = encoders_features_all[0]  #最底部的编码层

        # decoder part

        for decoder, encoder_features, attentionblock in zip(self.decoders, encoders_features, self.attentionblocks):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            attention_x, sig_out = attentionblock(encoder_features, gating)   #注意力块
            x = decoder(attention_x, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        # if not self.training:
            # x = self.final_activation(x)

        return x

class UNet3D2(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet3D2, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        d_11=Decoder2(f_maps[0]+f_maps[1], f_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_11)
        d_21=Decoder2(f_maps[1]+f_maps[2], f_maps[1], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_21)
        d_31=Decoder2(f_maps[2]+f_maps[3], f_maps[2], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_31)
        d_12=Decoder2(f_maps[1]+f_maps[0]+f_maps[0], f_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_12)
        d_22=Decoder2(f_maps[2]+f_maps[1]+f_maps[1], f_maps[1], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_22)
        d_13=Decoder2(f_maps[1]+f_maps[0]*3, f_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_13)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.append(x)

        # decoder part
        d_i=0
        final_=[]
        for decoder in self.decoders:
            x = decoder(encoders_features)
            encoders_features.append(x)
            if d_i==0 or d_i==3 or d_i==5:
                final_i=self.final_conv(x)
                final_.append(final_i)
            d_i+=1

        x = (final_[0]+final_[1]+final_[2])/3

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        # if not self.training:
            # x = self.final_activation(x)

        return x

class UNet3D2_attention(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet3D2, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        attentionblocks = []
        attentionblock_11 = GridAttentionBlock3D(in_channels=f_maps[0], gating_channels=f_maps[1],
                                                    inter_channels=f_maps[0], sub_sample_factor=(2,2,2), mode='concatenation')
        attentionblocks.append(attentionblock_11)
        d_11=Decoder2_attention(f_maps[0]+f_maps[1], f_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_11)

        attentionblock_21 = GridAttentionBlock3D(in_channels=f_maps[1], gating_channels=f_maps[2],
                                                    inter_channels=f_maps[1], sub_sample_factor=(2,2,2), mode='concatenation')
        attentionblocks.append(attentionblock_21)
        d_21=Decoder2_attention(f_maps[1]+f_maps[2], f_maps[1], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_21)

        attentionblock_31 = GridAttentionBlock3D(in_channels=f_maps[2], gating_channels=f_maps[3],
                                                    inter_channels=f_maps[2], sub_sample_factor=(2,2,2), mode='concatenation')
        attentionblocks.append(attentionblock_31)
        d_31=Decoder2_attention(f_maps[2]+f_maps[3], f_maps[2], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_31)

        attentionblock_12 = GridAttentionBlock3D(in_channels=f_maps[0], gating_channels=f_maps[2],
                                                    inter_channels=f_maps[0], sub_sample_factor=(2,2,2), mode='concatenation')
        attentionblocks.append(attentionblock_12)
        d_12=Decoder2_attention(f_maps[1]+f_maps[0]+f_maps[0], f_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_12)

        attentionblock_22 = GridAttentionBlock3D(in_channels=f_maps[1], gating_channels=f_maps[3],
                                                    inter_channels=f_maps[1], sub_sample_factor=(2,2,2), mode='concatenation')
        attentionblocks.append(attentionblock_22)
        d_22=Decoder2_attention(f_maps[2]+f_maps[1]+f_maps[1], f_maps[1], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_22)

        attentionblock_13 = GridAttentionBlock3D(in_channels=f_maps[0], gating_channels=f_maps[3],
                                                    inter_channels=f_maps[0], sub_sample_factor=(2,2,2), mode='concatenation')
        attentionblocks.append(attentionblock_13)
        d_13=Decoder2_attention(f_maps[1]+f_maps[0]*3, f_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_13)

        self.decoders = nn.ModuleList(decoders)
        self.attentionblocks = nn.ModuleList(attentionblocks)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.append(x)

        # decoder part
        d_i=0
        final_=[]
        attention_features = []
        for decoder, attentionblock in zip(self.decoders,self.attentionblocks):
            if d_i < 3:
                attention_x, sig_out = attentionblock(encoders_features[d_i], encoders_features[d_i+1])
                attention_features.append(attention_x)
            elif 3 <= d_i <= 4:
                attention_x, sig_out = attentionblock(encoders_features[d_i+1], encoders_features[d_i-1])
                attention_features.append(attention_x)
            else:
                attention_x, sig_out = attentionblock(encoders_features[d_i+2], encoders_features[d_i-2])
                attention_features.append(attention_x)
            x = decoder(encoders_features, attention_features)
            encoders_features.append(x)   #将解码层也放了进去
            if d_i==0 or d_i==3 or d_i==5:
                final_i=self.final_conv(x)
                final_.append(final_i)
            d_i+=1

        x = (final_[0]+final_[1]+final_[2])/3

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        # if not self.training:
            # x = self.final_activation(x)

        return x

class UNet3D2_attention_v2(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet3D2_attention_v2, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        #注意力块
        self.attentionblock_1 = GridAttentionBlock3D(in_channels=f_maps[0], gating_channels=f_maps[1],
                                                    inter_channels=f_maps[0], sub_sample_factor=(2,2,2), mode='concatenation')
        self.attentionblock_2 = GridAttentionBlock3D(in_channels=f_maps[1], gating_channels=f_maps[2],
                                                    inter_channels=f_maps[1], sub_sample_factor=(2,2,2), mode='concatenation')
        self.attentionblock_3 = GridAttentionBlock3D(in_channels=f_maps[2], gating_channels=f_maps[3],
                                                    inter_channels=f_maps[2], sub_sample_factor=(2,2,2), mode='concatenation')
        attentionblocks = [self.attentionblock_1, self.attentionblock_2, self.attentionblock_3]
        self.attentionbloks = nn.ModuleList(attentionblocks)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        d_11=Decoder2(f_maps[0]+f_maps[1], f_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_11)
        d_21=Decoder2(f_maps[1]+f_maps[2], f_maps[1], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_21)
        d_31=Decoder2(f_maps[2]+f_maps[3], f_maps[2], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_31)
        d_12=Decoder2(f_maps[1]+f_maps[0]+f_maps[0], f_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_12)
        d_22=Decoder2(f_maps[2]+f_maps[1]+f_maps[1], f_maps[1], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_22)
        d_13=Decoder2(f_maps[1]+f_maps[0]*3, f_maps[0], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_13)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.append(x)

        #注意力块
        attention_i = 0
        for attentionblock in self.attentionbloks:
            attention_x, sig_out = attentionblock(encoders_features[attention_i], encoders_features[attention_i+1])
            encoders_features[attention_i] = attention_x
            attention_i += 1
            
        # decoder part
        d_i=0
        final_=[]
        for decoder in self.decoders:
            x = decoder(encoders_features)
            encoders_features.append(x)
            if d_i==0 or d_i==3 or d_i==5:
                final_i=self.final_conv(x)
                final_.append(final_i)
            d_i+=1

        x = (final_[0]+final_[1]+final_[2])/3

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        # if not self.training:
            # x = self.final_activation(x)

        return x

class UNet3D3(nn.Module):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet3D3, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        for i in range(len(f_maps) - 1):
            in_feature_num = f_maps[0] + f_maps[1] + f_maps[2] + f_maps[3]
            out_feature_num = f_maps[-i-2]
            decoder = Decoder3(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.append(x)

        # decoder part
        for decoder in self.decoders:
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoders_features, x)
            encoders_features.append(x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        # if not self.training:
            # x = self.final_activation(x)

        return x

class UNet3D4(nn.Module):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='cr', num_groups=8,
                 **kwargs):
        super(UNet3D4, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        fea_i=f_maps
        for i in range(len(fea_i) - 1):
            in_feature_num = fea_i[-1] + fea_i[-2] + fea_i[-4]
            out_feature_num = fea_i[2-i]
            decoder = Decoder4(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)
            fea_i.append(fea_i[2-i])

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.append(x)

        # decoder part
        for decoder in self.decoders:
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoders_features, x)
            encoders_features.append(x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        # if not self.training:
            # x = self.final_activation(x)

        return x

class FCAUNet3D(nn.Module):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='cr', num_groups=8, #layer_order是用来决定使用哪些层的，比如'cr'就是conv + ReLU，在create_conv()中有定义
                 **kwargs):   #**kwargs 允许将不定长度的键值对作为参数传递给一个函数
        super(FCAUNet3D, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)  #f_maps=[64, 128, 256, 512]
            #f_maps的作用主要是确定每一层的通道数
        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []  #UNet编码层每一层的网络信息都在里面
        for i, out_feature_num in enumerate(f_maps):   #enumerate()可以遍历输出索引和数据
            if i == 0:
                encoder = Encoder_FCA(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)   #第一层不池化
            else:
                encoder = Encoder_FCA(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders) #加入到nn.ModuleList里面的module会自动注册到整个网络上，同时module的parameters也会自动添加到整个网络中

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))   #将f_maps列表翻转
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]   #信息拼接，所以通道相加，不过它在上采样时没有改变通道数
            ##若要通道数相同，应该改为in_feature_num = reversed_f_maps[i+1]*2
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)  #1*1卷积层

        if final_sigmoid:   #类Network_3D_Unet中默认为True
            self.final_activation = nn.Sigmoid()   #归一化
        else:
            self.final_activation = nn.Softmax(dim=1)   #在通道维度上进行softmax

    def forward(self, x,fca):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x,fca = encoder(x,fca)
            # reverse the encoder outputs to be aligned with the decoder
            #每次都在0的位置插入，这样就把encoder每层的x输出，在顺序上颠倒了
            encoders_features.insert(0, x)   #在索引0的位置插入x

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]  #去掉encoder最后一层，为了在连接的时候对应上。encoder四层，decoder三层

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)   #类Decoder中的forward的输入，这里的encoder_features是已经遍历后的对应层的encoder输出

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        # if not self.training:
            # x = self.final_activation(x)

        return x


class ResidualUNet3D(nn.Module):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock instead of DoubleConv as a basic building block as well as summation joining instead
    of concatenation joining. Since the model effectively becomes a residual net, in theory it allows for deeper UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. NLLLoss (multi-class)
            or BCELoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4,5
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        conv_layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
        skip_final_activation (bool): if True, skips the final normalization layer (sigmoid/softmax) and returns the
            logits directly
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=32, conv_layer_order='cge', num_groups=8,
                 skip_final_activation=False, **kwargs):
        super(ResidualUNet3D, self).__init__()

        if isinstance(f_maps, int):
            # use 5 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=5)   #层数变成了5

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses ExtResNetBlock as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=ExtResNetBlock,  #把DoubleConv换成了ExtResNetBlock
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=ExtResNetBlock,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses ExtResNetBlock as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):  #通道数有变化
            decoder = Decoder(reversed_f_maps[i], reversed_f_maps[i + 1], basic_module=ExtResNetBlock,
                              conv_layer_order=conv_layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if not skip_final_activation:
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        return x


class Noise2NoiseUNet3D(nn.Module):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock instead of DoubleConv as a basic building block as well as summation joining instead
    of concatenation joining. Since the model effectively becomes a residual net, in theory it allows for deeper UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. NLLLoss (multi-class)
            or BCELoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4,5
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, f_maps=16, num_groups=8, **kwargs):
        super(Noise2NoiseUNet3D, self).__init__()

        # Use LeakyReLU activation everywhere except the last layer
        conv_layer_order = 'clg'   #卷积层，LeakyReLU激活，GroupNorm标准化

        if isinstance(f_maps, int):
            # use 5 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=5)   #多了一层

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=conv_layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # 1x1x1 conv + simple ReLU in the final convolution  
        self.final_conv = SingleConv(f_maps[0], out_channels, kernel_size=1, order='cr', padding=0)
        #把激活函数从Sigmoid换成了ReLU

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        return x


def get_model(config):
    def _model_class(class_name):
        m = importlib.import_module('unet3d.model') #向a模块中导入c.py中的对象
        clazz = getattr(m, class_name) #getattr() 函数用于返回一个对象属性值。
        return clazz

    assert 'model' in config, 'Could not find model configuration'
    model_config = config['model']
    model_class = _model_class(model_config['name'])
    return model_class(**model_config)


###############################################Supervised Tags 3DUnet###################################################
#有监督
class TagsUNet3D(nn.Module):
    """
    Supervised tags 3DUnet
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels; since most often we're trying to learn
            3D unit vectors we use 3 as a default value
        output_heads (int): number of output heads from the network, each head corresponds to different
            semantic tag/direction to be learned
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `DoubleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
    """

    def __init__(self, in_channels, out_channels=3, output_heads=1, conv_layer_order='crg', init_channel_number=32,
                 **kwargs):
        super(TagsUNet3D, self).__init__()

        # number of groups for the GroupNorm
        num_groups = min(init_channel_number // 2, 32)

        # encoder path consist of 4 subsequent Encoder modules
        # the number of features maps is the same as in the paper
        self.encoders = nn.ModuleList([
            Encoder(in_channels, init_channel_number, apply_pooling=False, conv_layer_order=conv_layer_order,
                    num_groups=num_groups),
            Encoder(init_channel_number, 2 * init_channel_number, conv_layer_order=conv_layer_order,
                    num_groups=num_groups),
            Encoder(2 * init_channel_number, 4 * init_channel_number, conv_layer_order=conv_layer_order,
                    num_groups=num_groups),
            Encoder(4 * init_channel_number, 8 * init_channel_number, conv_layer_order=conv_layer_order,
                    num_groups=num_groups)
        ])

        self.decoders = nn.ModuleList([
            Decoder(4 * init_channel_number + 8 * init_channel_number, 4 * init_channel_number,
                    conv_layer_order=conv_layer_order, num_groups=num_groups),
            Decoder(2 * init_channel_number + 4 * init_channel_number, 2 * init_channel_number,
                    conv_layer_order=conv_layer_order, num_groups=num_groups),
            Decoder(init_channel_number + 2 * init_channel_number, init_channel_number,
                    conv_layer_order=conv_layer_order, num_groups=num_groups)
        ])

        self.final_heads = nn.ModuleList(
            [FinalConv(init_channel_number, out_channels, num_groups=num_groups) for _ in
             range(output_heads)])

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        # apply final layer per each output head
        tags = [final_head(x) for final_head in self.final_heads]

        # normalize directions with L2 norm
        return [tag / torch.norm(tag, p=2, dim=1).detach().clamp(min=1e-8) for tag in tags]


################################################Distance transform 3DUNet##############################################
#距离变换
class DistanceTransformUNet3D(nn.Module):
    """
    Predict Distance Transform to the boundary signal based on the output from the Tags3DUnet. Fore training use either:
        1. PixelWiseCrossEntropyLoss if the distance transform is quantized (classification)
        2. MSELoss if the distance transform is continuous (regression)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. NLLLoss (multi-class)
            or BCELoss (two-class) respectively)
        final_sigmoid (bool): 'sigmoid'/'softmax' whether element-wise nn.Sigmoid or nn.Softmax should be applied after
            the final 1x1 convolution
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, init_channel_number=32, **kwargs):
        super(DistanceTransformUNet3D, self).__init__()

        # number of groups for the GroupNorm
        num_groups = min(init_channel_number // 2, 32)

        # encoder path consist of 4 subsequent Encoder modules
        # the number of features maps is the same as in the paper
        self.encoders = nn.ModuleList([
            Encoder(in_channels, init_channel_number, apply_pooling=False, conv_layer_order='crg',
                    num_groups=num_groups),
            Encoder(init_channel_number, 2 * init_channel_number, pool_type='avg', conv_layer_order='crg',
                    num_groups=num_groups)
        ])

        self.decoders = nn.ModuleList([
            Decoder(3 * init_channel_number, init_channel_number, conv_layer_order='crg', num_groups=num_groups)
        ])

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(init_channel_number, out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, inputs):
        # allow multiple heads
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            x = torch.cat(inputs, dim=1)
        else:
            x = inputs

        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        # apply final 1x1 convolution
        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x)

        return x


class EndToEndDTUNet3D(nn.Module):
    def __init__(self, tags_in_channels, tags_out_channels, tags_output_heads, tags_init_channel_number,
                 dt_in_channels, dt_out_channels, dt_final_sigmoid, dt_init_channel_number,
                 tags_net_path=None, dt_net_path=None, **kwargs):
        super(EndToEndDTUNet3D, self).__init__()

        self.tags_net = TagsUNet3D(tags_in_channels, tags_out_channels, tags_output_heads,
                                   init_channel_number=tags_init_channel_number)
        if tags_net_path is not None:
            # load pre-trained TagsUNet3D
            self.tags_net = self._load_net(tags_net_path, self.tags_net)

        self.dt_net = DistanceTransformUNet3D(dt_in_channels, dt_out_channels, dt_final_sigmoid,
                                              init_channel_number=dt_init_channel_number)
        if dt_net_path is not None:
            # load pre-trained DistanceTransformUNet3D
            self.dt_net = self._load_net(dt_net_path, self.dt_net)

    @staticmethod
    def _load_net(checkpoint_path, model):
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['model_state_dict'])
        return model

    def forward(self, x):
        x = self.tags_net(x)
        return self.dt_net(x)


############################################# 2D ###############################################
class UNet2_2d(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crb', num_groups=8,
                 **kwargs):
        super(UNet2_2d, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder_2d(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv_2d,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder_2d(f_maps[i - 1], out_feature_num, basic_module=DoubleConv_2d,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        d_11=Decoder2_2d(f_maps[0]+f_maps[1], f_maps[0], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_11)
        d_21=Decoder2_2d(f_maps[1]+f_maps[2], f_maps[1], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_21)
        d_31=Decoder2_2d(f_maps[2]+f_maps[3], f_maps[2], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_31)
        d_12=Decoder2_2d(f_maps[1]+f_maps[0]+f_maps[0], f_maps[0], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_12)
        d_22=Decoder2_2d(f_maps[2]+f_maps[1]+f_maps[1], f_maps[1], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_22)
        d_13=Decoder2_2d(f_maps[1]+f_maps[0]*3, f_maps[0], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_13)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.append(x)

        # decoder part
        d_i=0
        final_=[]
        for decoder in self.decoders:
            x = decoder(encoders_features)
            encoders_features.append(x)
            if d_i==0 or d_i==3 or d_i==5:
                final_i=self.final_conv(x)
                final_.append(final_i)
            d_i+=1

        x = (final_[0]+final_[1]+final_[2])/3

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        # if not self.training:
            # x = self.final_activation(x)

        return x

class Noise2NoiseUNet_2d(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=16, num_groups=8, **kwargs):
        super(Noise2NoiseUNet_2d, self).__init__()

        # Use LeakyReLU activation everywhere except the last layer
        conv_layer_order = 'clg'   #卷积层，LeakyReLU激活，GroupNorm标准化

        if isinstance(f_maps, int):
            # use 5 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)   #多了一层

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder_2d(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv_2d,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            else:
                encoder = Encoder_2d(f_maps[i - 1], out_feature_num, basic_module=DoubleConv_2d,
                                  conv_layer_order=conv_layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder_2d(in_feature_num, out_feature_num, basic_module=DoubleConv_2d,
                              conv_layer_order=conv_layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # 1x1x1 conv + simple ReLU in the final convolution  
        self.final_conv = SingleConv_2d(f_maps[0], out_channels, kernel_size=1, order='cr', padding=0)
        #把激活函数从Sigmoid换成了ReLU

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        if not self.training:
            x = self.final_activation(x/51-2)

        return x

class UNet_2d(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crb', num_groups=8,
                 **kwargs):
        super(UNet_2d, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder_2d(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv_2d,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder_2d(f_maps[i - 1], out_feature_num, basic_module=DoubleConv_2d,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder_2d(in_feature_num, out_feature_num, basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x/51-2)

        return x

class UNet2_attention_v1_1_2(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crb', num_groups=8,
                 **kwargs):
        super(UNet2_attention_v1_1_2, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder_2d(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv_2d,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder_2d(f_maps[i - 1], out_feature_num, basic_module=DoubleConv_2d,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        attentionblocks = []
        attentionblock_11 = GridAttentionBlockND(in_channels=f_maps[0], gating_channels=f_maps[1],
                                                    inter_channels=f_maps[0], sub_sample_factor=(2,2), mode='concatenation')
        attentionblocks.append(attentionblock_11)
        d_11=Decoder2_attention_bilinear_2d(f_maps[0]+f_maps[1], f_maps[0], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_11)

        attentionblock_21 = GridAttentionBlockND(in_channels=f_maps[1], gating_channels=f_maps[2],
                                                    inter_channels=f_maps[1], sub_sample_factor=(2,2), mode='concatenation')
        attentionblocks.append(attentionblock_21)
        d_21=Decoder2_attention_bilinear_2d(f_maps[1]+f_maps[2], f_maps[1], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_21)

        attentionblock_31 = GridAttentionBlockND(in_channels=f_maps[2], gating_channels=f_maps[3],
                                                    inter_channels=f_maps[2], sub_sample_factor=(2,2), mode='concatenation')
        attentionblocks.append(attentionblock_31)
        d_31=Decoder2_attention_bilinear_2d(f_maps[2]+f_maps[3], f_maps[2], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_31)

        attentionblock_12 = GridAttentionBlockND(in_channels=f_maps[0], gating_channels=f_maps[1],
                                                    inter_channels=f_maps[0], sub_sample_factor=(2,2), mode='concatenation')
        attentionblocks.append(attentionblock_12)
        d_12=Decoder2_attention_bilinear_2d(f_maps[1]+f_maps[0]+f_maps[0], f_maps[0], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_12)

        attentionblock_22 = GridAttentionBlockND(in_channels=f_maps[1], gating_channels=f_maps[2],
                                                    inter_channels=f_maps[1], sub_sample_factor=(2,2), mode='concatenation')
        attentionblocks.append(attentionblock_22)
        d_22=Decoder2_attention_bilinear_2d(f_maps[2]+f_maps[1]+f_maps[1], f_maps[1], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_22)

        attentionblock_13 = GridAttentionBlockND(in_channels=f_maps[0], gating_channels=f_maps[1],
                                                    inter_channels=f_maps[0], sub_sample_factor=(2,2), mode='concatenation')
        attentionblocks.append(attentionblock_13)
        d_13=Decoder2_attention_bilinear_2d(f_maps[1]+f_maps[0]*3, f_maps[0], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_13)

        self.decoders = nn.ModuleList(decoders)
        self.attentionblocks = nn.ModuleList(attentionblocks)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_decodes_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_decodes_features.append(x)

        # decoder part
        d_i=0
        final_=[]
        attention_features = []
        for decoder, attentionblock in zip(self.decoders,self.attentionblocks):
            if d_i < 3:
                attention_x, sig_out = attentionblock(encoders_decodes_features[d_i], encoders_decodes_features[d_i+1])
                attention_features.append(attention_x)
            elif 3 <= d_i <= 4:
                attention_x, sig_out = attentionblock(encoders_decodes_features[d_i+1], encoders_decodes_features[d_i+2])
                attention_features.append(attention_x)
            else:
                attention_x, sig_out = attentionblock(encoders_decodes_features[d_i+2], encoders_decodes_features[d_i+3])
                attention_features.append(attention_x)
            x = decoder(encoders_decodes_features, attention_features)
            encoders_decodes_features.append(x)   #将解码层也放了进去
            if d_i==0 or d_i==3 or d_i==5:
                final_i=self.final_conv(x)
                final_.append(final_i)
            d_i+=1

        x = (final_[0]+final_[1]+final_[2])/3

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x/51-2)

        return x

class UNet2_attention_crg(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(UNet2_attention_crg, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder_2d(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv_2d,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder_2d(f_maps[i - 1], out_feature_num, basic_module=DoubleConv_2d,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        attentionblocks = []
        attentionblock_11 = GridAttentionBlockND(in_channels=f_maps[0], gating_channels=f_maps[1],
                                                    inter_channels=f_maps[0], sub_sample_factor=(2,2), mode='concatenation')
        attentionblocks.append(attentionblock_11)
        d_11=Decoder2_attention_bilinear_2d(f_maps[0]+f_maps[1], f_maps[0], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_11)

        attentionblock_21 = GridAttentionBlockND(in_channels=f_maps[1], gating_channels=f_maps[2],
                                                    inter_channels=f_maps[1], sub_sample_factor=(2,2), mode='concatenation')
        attentionblocks.append(attentionblock_21)
        d_21=Decoder2_attention_bilinear_2d(f_maps[1]+f_maps[2], f_maps[1], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_21)

        attentionblock_31 = GridAttentionBlockND(in_channels=f_maps[2], gating_channels=f_maps[3],
                                                    inter_channels=f_maps[2], sub_sample_factor=(2,2), mode='concatenation')
        attentionblocks.append(attentionblock_31)
        d_31=Decoder2_attention_bilinear_2d(f_maps[2]+f_maps[3], f_maps[2], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_31)

        attentionblock_12 = GridAttentionBlockND(in_channels=f_maps[0], gating_channels=f_maps[1],
                                                    inter_channels=f_maps[0], sub_sample_factor=(2,2), mode='concatenation')
        attentionblocks.append(attentionblock_12)
        d_12=Decoder2_attention_bilinear_2d(f_maps[1]+f_maps[0]+f_maps[0], f_maps[0], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_12)

        attentionblock_22 = GridAttentionBlockND(in_channels=f_maps[1], gating_channels=f_maps[2],
                                                    inter_channels=f_maps[1], sub_sample_factor=(2,2), mode='concatenation')
        attentionblocks.append(attentionblock_22)
        d_22=Decoder2_attention_bilinear_2d(f_maps[2]+f_maps[1]+f_maps[1], f_maps[1], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_22)

        attentionblock_13 = GridAttentionBlockND(in_channels=f_maps[0], gating_channels=f_maps[1],
                                                    inter_channels=f_maps[0], sub_sample_factor=(2,2), mode='concatenation')
        attentionblocks.append(attentionblock_13)
        d_13=Decoder2_attention_bilinear_2d(f_maps[1]+f_maps[0]*3, f_maps[0], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_13)

        self.decoders = nn.ModuleList(decoders)
        self.attentionblocks = nn.ModuleList(attentionblocks)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_decodes_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_decodes_features.append(x)

        # decoder part
        d_i=0
        final_=[]
        attention_features = []
        for decoder, attentionblock in zip(self.decoders,self.attentionblocks):
            if d_i < 3:
                attention_x, sig_out = attentionblock(encoders_decodes_features[d_i], encoders_decodes_features[d_i+1])
                attention_features.append(attention_x)
            elif 3 <= d_i <= 4:
                attention_x, sig_out = attentionblock(encoders_decodes_features[d_i+1], encoders_decodes_features[d_i+2])
                attention_features.append(attention_x)
            else:
                attention_x, sig_out = attentionblock(encoders_decodes_features[d_i+2], encoders_decodes_features[d_i+3])
                attention_features.append(attention_x)
            x = decoder(encoders_decodes_features, attention_features)
            encoders_decodes_features.append(x)   #将解码层也放了进去
            if d_i==0 or d_i==3 or d_i==5:
                final_i=self.final_conv(x)
                final_.append(final_i)
            d_i+=1

        x = (final_[0]+final_[1]+final_[2])/3

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x/51-2)

        return x

class UNet2_attention_clg(nn.Module):

    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='clg', num_groups=8,
                 **kwargs):
        super(UNet2_attention_clg, self).__init__()

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=4)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder_2d(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv_2d,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder_2d(f_maps[i - 1], out_feature_num, basic_module=DoubleConv_2d,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        attentionblocks = []
        attentionblock_11 = GridAttentionBlockND(in_channels=f_maps[0], gating_channels=f_maps[1],
                                                    inter_channels=f_maps[0], sub_sample_factor=(2,2), mode='concatenation')
        attentionblocks.append(attentionblock_11)
        d_11=Decoder2_attention_bilinear_2d(f_maps[0]+f_maps[1], f_maps[0], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_11)

        attentionblock_21 = GridAttentionBlockND(in_channels=f_maps[1], gating_channels=f_maps[2],
                                                    inter_channels=f_maps[1], sub_sample_factor=(2,2), mode='concatenation')
        attentionblocks.append(attentionblock_21)
        d_21=Decoder2_attention_bilinear_2d(f_maps[1]+f_maps[2], f_maps[1], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_21)

        attentionblock_31 = GridAttentionBlockND(in_channels=f_maps[2], gating_channels=f_maps[3],
                                                    inter_channels=f_maps[2], sub_sample_factor=(2,2), mode='concatenation')
        attentionblocks.append(attentionblock_31)
        d_31=Decoder2_attention_bilinear_2d(f_maps[2]+f_maps[3], f_maps[2], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_31)

        attentionblock_12 = GridAttentionBlockND(in_channels=f_maps[0], gating_channels=f_maps[1],
                                                    inter_channels=f_maps[0], sub_sample_factor=(2,2), mode='concatenation')
        attentionblocks.append(attentionblock_12)
        d_12=Decoder2_attention_bilinear_2d(f_maps[1]+f_maps[0]+f_maps[0], f_maps[0], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_12)

        attentionblock_22 = GridAttentionBlockND(in_channels=f_maps[1], gating_channels=f_maps[2],
                                                    inter_channels=f_maps[1], sub_sample_factor=(2,2), mode='concatenation')
        attentionblocks.append(attentionblock_22)
        d_22=Decoder2_attention_bilinear_2d(f_maps[2]+f_maps[1]+f_maps[1], f_maps[1], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_22)

        attentionblock_13 = GridAttentionBlockND(in_channels=f_maps[0], gating_channels=f_maps[1],
                                                    inter_channels=f_maps[0], sub_sample_factor=(2,2), mode='concatenation')
        attentionblocks.append(attentionblock_13)
        d_13=Decoder2_attention_bilinear_2d(f_maps[1]+f_maps[0]*3, f_maps[0], basic_module=DoubleConv_2d,
                              conv_layer_order=layer_order, num_groups=num_groups)
        decoders.append(d_13)

        self.decoders = nn.ModuleList(decoders)
        self.attentionblocks = nn.ModuleList(attentionblocks)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_decodes_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_decodes_features.append(x)

        # decoder part
        d_i=0
        final_=[]
        attention_features = []
        for decoder, attentionblock in zip(self.decoders,self.attentionblocks):
            if d_i < 3:
                attention_x, sig_out = attentionblock(encoders_decodes_features[d_i], encoders_decodes_features[d_i+1])
                attention_features.append(attention_x)
            elif 3 <= d_i <= 4:
                attention_x, sig_out = attentionblock(encoders_decodes_features[d_i+1], encoders_decodes_features[d_i+2])
                attention_features.append(attention_x)
            else:
                attention_x, sig_out = attentionblock(encoders_decodes_features[d_i+2], encoders_decodes_features[d_i+3])
                attention_features.append(attention_x)
            x = decoder(encoders_decodes_features, attention_features)
            encoders_decodes_features.append(x)   #将解码层也放了进去
            if d_i==0 or d_i==3 or d_i==5:
                final_i=self.final_conv(x)
                final_.append(final_i)
            d_i+=1

        x = (final_[0]+final_[1]+final_[2])/3

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if not self.training:
            x = self.final_activation(x/51-2)

        return x

class DnCNN(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, f_maps=64, layer_order='crb', num_groups=8, #layer_order是用来决定使用哪些层的，比如'cr'就是conv + ReLU，在create_conv()中有定义
                 **kwargs):   #**kwargs 允许将不定长度的键值对作为参数传递给一个函数
        super(DnCNN, self).__init__()
        convs = []
        for i in range(8):
            if i == 0:
                conv_i = SingleConv_2d(in_channels, 64, 3, 'cr', num_groups)
                convs.append(conv_i)
            else:
                conv_i = SingleConv_2d(64, 64, 3, layer_order, num_groups)
                convs.append(conv_i)
        
        self.conv_ = nn.ModuleList(convs)

        self.final_conv = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        for conv_ in self.conv_:
            x = conv_(x)
        x = self.final_conv(x)
        return x

