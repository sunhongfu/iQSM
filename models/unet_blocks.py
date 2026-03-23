import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LoTLayer(nn.Module):
    def __init__(self, conv_x):
        super(LoTLayer, self).__init__()
        self.conv_x = nn.Parameter(conv_x, requires_grad=False)
        self.conv_y = nn.Parameter(conv_x.repeat([15, 1, 1, 1, 1]).clone(), requires_grad=True)

    def forward(self, phi, mask, TE, B0):
        expPhi_r = torch.cos(phi)
        expPhi_i = torch.sin(phi)

        a_r = self.LG(expPhi_r, self.conv_x)
        a_i = self.LG(expPhi_i, self.conv_x)
        b_i = a_i * expPhi_r - a_r * expPhi_i
        b_i = b_i * mask
        b_i = b_i / (B0 * TE)
        b_i = b_i * (3 * 20e-3)

        a_r = self.LG(expPhi_r, self.conv_y)
        a_i = self.LG(expPhi_i, self.conv_y)
        d_i = a_i * expPhi_r - a_r * expPhi_i
        d_i = d_i * mask
        d_i = d_i / (B0 * TE)
        d_i = d_i * (3 * 20e-3)

        return b_i, d_i

    def LG(self, tensor_image, weight):
        out = F.conv3d(tensor_image, weight, bias=None, stride=1, padding=1)
        h, w, d = out.shape[2], out.shape[3], out.shape[4]
        out[:, :, [0, h - 1], :, :] = 0
        out[:, :, :, [0, w - 1], :] = 0
        out[:, :, :, :, [0, d - 1]] = 0
        return out


class EncodingBlocks(nn.Module):
    def __init__(self, num_in, num_out):
        super(EncodingBlocks, self).__init__()
        self.EncodeConv = nn.Sequential(
            nn.Conv3d(num_in, num_out, 3, padding=1),
            nn.BatchNorm3d(num_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_out, num_out, 3, padding=1),
            nn.BatchNorm3d(num_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.EncodeConv(x)


class MidBlocks(nn.Module):
    def __init__(self, num_ch):
        super(MidBlocks, self).__init__()
        self.MidConv = nn.Sequential(
            nn.Conv3d(num_ch, 2 * num_ch, 3, padding=1),
            nn.BatchNorm3d(2 * num_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(2 * num_ch, num_ch, 3, padding=1),
            nn.BatchNorm3d(num_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.MidConv(x)


class DecodingBlocks(nn.Module):
    def __init__(self, num_in, num_out, bilinear=False):
        super(DecodingBlocks, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.BatchNorm3d(num_in),
                nn.ReLU(inplace=True),
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(num_in, num_in, 2, stride=2),
                nn.BatchNorm3d(num_in),
                nn.ReLU(inplace=True),
            )
        self.DecodeConv = nn.Sequential(
            nn.Conv3d(2 * num_in, num_in, 3, padding=1),
            nn.BatchNorm3d(num_in),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_in, num_out, 3, padding=1),
            nn.BatchNorm3d(num_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.DecodeConv(x)
