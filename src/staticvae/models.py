
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bVAE (Higgings, 2017) or inspired by the paper

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




class VAE(nn.Module):
    """ encoder/decoder from Higgins for VAE (Chairs, 3DFaces) - image size 64x64x1
        from Table 1 in Higgins et al., 2017, ICLR

        number of latents can be adapted, spatial input dimensions are fixed

    """


    def __init__(self, n_latents 10, img_channels = 1):
        super(VAE, self).__init__()

        # encoder
        self.conv1 = nn.Conv2d(in_channels = img_channels, out_channels = 32, kernel_size = 4, stride = 2, padding = 0)     # B, 32, 31, 31
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2, padding = 0)               # B, 32, 14, 14
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 0)               # B, 64, 6, 6
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 4, stride = 2, padding = 0)               # B, 64, 2, 2

        self.fc_enc = nn.Linear(256, n_latent*2, bias = True)

        # decoder
        self.fc = nn.Linear(n_latent*2, 256, bias = True)                         # B, 256 (after .view(): B, 64, 2, 2)

        self.convT4 = nn.ConvTranspose2d(64, 64, 4, 2, 0)                       # B, 64, 6, 6
        self.convT3 = nn.ConvTranspose2d(64, 32, 4, 2, 0)                       # B, 32, 14, 14
        self.convT2 = nn.ConvTranspose2d(32, 32, 4, 2, 0, output_padding = 1)   # B, 32, 31, 31
        self.convT1 = nn.ConvTranspose2d(32, img_channels, 4, 2, 0)             # B, img_channels, 64, 64

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

        x = self.fc(x).view(-1,64,2,2)
        x = torch.relu(self.convT4(x))
        x = torch.relu(self.convT3(x))
        x = torch.relu(self.convT2(x))
        x = torch.relu(self.convT1(x))

        return x
