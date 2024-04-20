#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FocalFrequencyLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalFrequencyLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        batch_size, channels, height, width = input.size()

        input_fft = torch.fft.fftn(input, dim=[2, 3])
        target_fft = torch.fft.fftn(target, dim=[2, 3])

        input_amp = torch.abs(input_fft)
        target_amp = torch.abs(target_fft)

        loss = torch.pow(1 - torch.exp(-self.alpha * torch.pow(target_amp, self.gamma)), 2)
        loss = torch.mean(loss)

        return loss