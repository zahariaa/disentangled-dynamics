#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
encoders from bVAE (Higgings, 2017) or inspired by the paper

"""

import torch
import torch.nn as nn
import torch.nn.init as init


def kaiming_init(m):
    """ from https://github.com/1Konny/Beta-VAE/blob/master/model.py """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class encoderBVAE(nn.Module):
    """ encoder from Higgins for VAE (Chairs, 3DFaces) - input 64x64ximg_channels
        from Table 1 in Higgins et al., 2017, ICLR
        
        number of latents can be adapted, spatial input dimensions are fixed
        
    """
    
    def __init__(self, n_latent = 32, img_channels = 1):
        super(encoderBVAE, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = img_channels, out_channels = 32, kernel_size = 4, stride = 2, padding = 0)     # B, 32, 31, 31
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2, padding = 0)               # B, 32, 14, 14
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 0)               # B, 64, 6, 6
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 4, stride = 2, padding = 0)               # B, 64, 2, 2
        
        self.fc = nn.Linear(256, n_latent, bias = True)
        # in VAE -> here mapping to n_latent * 2 (for mu and sigma)
        
        self.weight_init()
        
    def weight_init(self):
        for m in self._modules:
            kaiming_init(m)
        
    def forward(self, x):
        
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        
        x = self.fc(x.view(-1, 256)) 
        
        return x
    

class encoderBVAE_like(nn.Module):
    """ encoder inspired by Higgins, 2017 - for 32x32ximg_channels input and 4 latents
        
        number of latents can be adapted, spatial input dimensions are fixed
    
    """
    
    def __init__(self, n_latent = 4, img_channels = 1):
        super(encoderBVAE_like, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = img_channels, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)     # B, 32, 16, 16
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)               # B, 32, 8, 8
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)               # B, 64, 4, 4
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)               # B, 64, 2, 2
        
        self.fc = nn.Linear(256, n_latent, bias = True)
        
        self.weight_init()
        
    def weight_init(self):
        for m in self._modules:
            kaiming_init(m)
        
    def forward(self, x):
        
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        
        x = self.fc(x.view(-1, 256)) 
        
        return x
    
