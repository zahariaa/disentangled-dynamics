#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decoders from bVAE (Higgings, 2017) or inspired by the paper

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

class decoderBVAE(nn.Module):
    """ decoder from Higgins for VAE (Chairs, 3DFaces) - output 64x64x1 
        from Table 1 in Higgins et al., 2017, ICLR
        
        number of latents can be adapted, spatial input dimensions are fixed
        
    """
    
    def __init__(self, n_latent = 32, img_channels = 1):
        super(decoderBVAE, self).__init__()        
                                                                                # output shape (B = batch size)
        self.fc = nn.Linear(n_latent, 256, bias = True)                         # B, 256 (after .view(): B, 64, 2, 2)
        
        self.convT4 = nn.ConvTranspose2d(64, 64, 4, 2, 0)                       # B, 64, 6, 6
        self.convT3 = nn.ConvTranspose2d(64, 32, 4, 2, 0)                       # B, 32, 14, 14
        self.convT2 = nn.ConvTranspose2d(32, 32, 4, 2, 0, output_padding = 1)   # B, 32, 31, 31
        self.convT1 = nn.ConvTranspose2d(32, img_channels, 4, 2, 0)             # B, img_channels, 64, 64
        
        self.weight_init()
        
    def weight_init(self):
        for m in self._modules:
            kaiming_init(m)
        
    def forward(self, x):

        x = self.fc(x).view(-1,64,2,2)
        x = torch.relu(self.convT4(x))
        x = torch.relu(self.convT3(x))
        x = torch.relu(self.convT2(x))
        x = torch.relu(self.convT1(x))    
        
        return x
    
class decoderBVAE_like(nn.Module):
    """ decoder inspired by Higgins, 2017 - for 32x32ximg_channels output and 4 latents as input
        
        number of latents can be adapted, spatial input dimensions are fixed
    
    """
    
    def __init__(self, n_latent = 4, img_channels = 1):
        super(decoderBVAE_like, self).__init__()        
                                                                                # output shape (B = batch size)
        self.fc = nn.Linear(n_latent, 256, bias = True)                         # B, 256 (after .view(): B, 64, 2, 2)
        
        self.convT4 = nn.ConvTranspose2d(64, 64, 3, 2, 1, 1)                       # B, 64, 4, 4
        self.convT3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)                       # B, 32, 8, 8
        self.convT2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)                       # B, 32, 16, 16
        self.convT1 = nn.ConvTranspose2d(32, img_channels, 3, 2, 1, 1)             # B, img_channels, 32, 32
        
        self.weight_init()
        
    def weight_init(self):
        for m in self._modules:
            kaiming_init(m)
        
    def forward(self, x):

        x = self.fc(x).view(-1,64,2,2)
        x = torch.relu(self.convT4(x))
        x = torch.relu(self.convT3(x))
        x = torch.relu(self.convT2(x))
        x = torch.relu(self.convT1(x))    
        
        return x

class decoderBVAE_like_wElu(nn.Module):
    """ decoder inspired by Higgins, 2017 - for 32x32ximg_channels output and 4 latents as input
        
        number of latents can be adapted, spatial input dimensions are fixed
        
        training of decoderBVAE_like depends heavily on initialization: output of untrained network might be very sparse because negative input to relus (little gradients to learn on)
        here, therefore elu non-linearities
            
    
    """
    
    def __init__(self, n_latent = 4, img_channels = 1):
        super(decoderBVAE_like_wElu, self).__init__()        
                                                                                # output shape (B = batch size)
        self.fc = nn.Linear(n_latent, 256, bias = True)                         # B, 256 (after .view(): B, 64, 2, 2)
        
        self.convT4 = nn.ConvTranspose2d(64, 64, 3, 2, 1, 1)                       # B, 64, 4, 4
        self.convT3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)                       # B, 32, 8, 8
        self.convT2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)                       # B, 32, 16, 16
        self.convT1 = nn.ConvTranspose2d(32, img_channels, 3, 2, 1, 1)             # B, img_channels, 32, 32
        
        self.weight_init()
        
    def weight_init(self):
        for m in self._modules:
            kaiming_init(m)
        
    def forward(self, x):

        x = self.fc(x).view(-1,64,2,2)
        x = torch.nn.functional.elu(self.convT4(x))
        x = torch.nn.functional.elu(self.convT3(x))
        x = torch.nn.functional.elu(self.convT2(x))
        x = torch.nn.functional.elu(self.convT1(x))    
        
        return x

class decoderBVAE_like_wElu_SigmoidOutput(nn.Module):
    """ decoder inspired by Higgins, 2017 - for 32x32ximg_channels output and 4 latents as input
        
        number of latents can be adapted, spatial input dimensions are fixed
        
        training of decoderBVAE_like depends heavily on initialization: output of untrained network might be very sparse because negative input to relus (little gradients to learn on)
        here, therefore elu non-linearities
        
        output non-linearity: sigmoid (as in staticVAE - to compare the difference between the two)
            
    
    """
    
    def __init__(self, n_latent = 4, img_channels = 1):
        super(decoderBVAE_like_wElu_SigmoidOutput, self).__init__()        
                                                                                # output shape (B = batch size)
        self.fc = nn.Linear(n_latent, 256, bias = True)                         # B, 256 (after .view(): B, 64, 2, 2)
        
        self.convT4 = nn.ConvTranspose2d(64, 64, 3, 2, 1, 1)                       # B, 64, 4, 4
        self.convT3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)                       # B, 32, 8, 8
        self.convT2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)                       # B, 32, 16, 16
        self.convT1 = nn.ConvTranspose2d(32, img_channels, 3, 2, 1, 1)             # B, img_channels, 32, 32
        
        self.weight_init()
        
    def weight_init(self):
        for m in self._modules:
            kaiming_init(m)
        
    def forward(self, x):

        x = self.fc(x).view(-1,64,2,2)
        x = torch.nn.functional.elu(self.convT4(x))
        x = torch.nn.functional.elu(self.convT3(x))
        x = torch.nn.functional.elu(self.convT2(x))
        x = torch.nn.functional.sigmoid(self.convT1(x))    
        
        return x
    