import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_blocks import EncodingBlocks, MidBlocks, DecodingBlocks


class Unet(nn.Module):
    def __init__(self, EncodingDepth, In_channels, Out_channels):
        super(Unet, self).__init__()
        self.EncodeConvs = []
        self.DecodeConvs = []
        self.EncodingDepth = EncodingDepth
        initial_num_layers = 16
        temp = list(range(1, EncodingDepth + 1))

        for encodingLayer in temp:
            if encodingLayer == 1:
                num_outputs = initial_num_layers * 2 ** (encodingLayer - 1)
                self.EncodeConvs.append(EncodingBlocks(In_channels, num_outputs))
            else:
                num_outputs = initial_num_layers * 2 ** (encodingLayer - 1)
                self.EncodeConvs.append(EncodingBlocks(num_outputs // 2, num_outputs))
        self.EncodeConvs = nn.ModuleList(self.EncodeConvs)

        self.MidConv1 = MidBlocks(num_outputs)
        initial_decode_num_ch = num_outputs

        for decodingLayer in temp:
            if decodingLayer == EncodingDepth:
                num_inputs = initial_decode_num_ch // 2 ** (decodingLayer - 1)
                self.DecodeConvs.append(DecodingBlocks(num_inputs, num_inputs))
            else:
                num_inputs = initial_decode_num_ch // 2 ** (decodingLayer - 1)
                self.DecodeConvs.append(DecodingBlocks(num_inputs, num_inputs // 2))
        self.DecodeConvs = nn.ModuleList(self.DecodeConvs)
        self.FinalConv = nn.Conv3d(num_inputs, Out_channels, 1, stride=1, padding=0)

    def forward(self, x_b, x_d):
        Input = x_b
        x = torch.cat([x_b, x_d], dim=1)
        names = self.__dict__
        temp = list(range(1, self.EncodingDepth + 1))

        for encodingLayer in temp:
            x = self.EncodeConvs[encodingLayer - 1](x)
            names["EncodeX" + str(encodingLayer)] = x
            x = F.max_pool3d(x, 2)

        x = self.MidConv1(x)

        for decodingLayer in temp:
            x2 = names["EncodeX" + str(self.EncodingDepth - decodingLayer + 1)]
            x = self.DecodeConvs[decodingLayer - 1](x, x2)

        x = self.FinalConv(x)
        return x + Input
