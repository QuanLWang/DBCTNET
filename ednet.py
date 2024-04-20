#!/usr/bin/env python
# coding=utf-8

import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.basic_model import ConvBlock, ResnetBlock
from torchvision.transforms import *
import torch.nn.functional as F

class att_spatial(nn.Module):
    def __init__(self):
        super(att_spatial, self).__init__()
        kernel_size = 3
        block = [
            ConvBlock(2, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(6):
            block.append(ResnetBlock(32, 3, 1, 1, 0.1, activation='prelu', norm=None))
        self.block = nn.Sequential(*block)
        self.spatial = ConvBlock(2, 1, 3, 1, 1, activation='prelu', norm=None, bias = False)
        
    def forward(self, x):
        x = self.block(x)
        x_compress = torch.cat([torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)], dim=1)
        x_out = self.spatial(x_compress)

        scale = F.sigmoid(x_out)
        return scale

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, attention=True):
        super(InvertedResidualBlock, self).__init__()

        hidden_dim = in_channels * expansion_factor

        layers = []
        layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # attention

        if attention:
            layers.append(AttentionModule(hidden_dim))

        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.residual = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.residual(x)


class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        super(AttentionModule, self).__init__()

        self.expand_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.squeeze_conv = nn.Conv2d(out_channels, out_channels // reduction_ratio, kernel_size=1)

        self.query_conv = nn.Conv2d(out_channels // reduction_ratio, out_channels // reduction_ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(out_channels // reduction_ratio, out_channels // reduction_ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(out_channels // reduction_ratio, out_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Expand information
        x_expand = self.expand_conv(x)

        # Squeeze information
        x_squeeze = self.squeeze_conv(x_expand)

        # Generate Query (Q), Key (K), and Value (V)
        Q = self.query_conv(x_squeeze)
        K = self.key_conv(x_squeeze)
        V = self.value_conv(x_squeeze)

        # Reshape Q and V for matrix multiplication to generate a new transposed attention map
        Q = Q.view(Q.size(0), -1, Q.size(2) * Q.size(3))
        V = V.view(V.size(0), -1, V.size(2) * V.size(3))

        attention_map = torch.bmm(Q.permute(0, 2, 1), K)
        attention_map = self.softmax(attention_map)

        output = torch.bmm(V, attention_map.permute(0, 2, 1))
        output = output.view(x.size(0), -1, x.size(2), x.size(3))

        return output