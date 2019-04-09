#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:22:30 2019

@author: benjamin
"""


import torch
from torch.nn import MSELoss
from torch.optim import Adam, RMSprop


import sys
sys.path.append("..") # Adds higher directory to python modules path.

from data.dspritesb import dSpriteBackgroundDataset, Rescale
from models.encoders import encoderBVAE_like
from models.decoders import decoderBVAE_like, decoderBVAE_like_wElu


class Solver(object):
    
    def __init(self, args):
        
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_epochs = args.max_epochs

        self.model = args.model
        self.lr = args.lr
        
        dataloaderparams = {'batch_size': args.batch_size,
                            'shuffle': args.shuffle,
                            'num_workers': args.num_workers}

        if args.dataset.lower() == 'dsprites_circle':
            train_loader = torch.utils.data.DataLoader(dSpriteBackgroundDataset(transform=Rescale(32),
                                            shapetype = 'circle'), **dataloaderparams)
        elif args.dataset.lower() == 'dsprites':
            train_loader = torch.utils.data.DataLoader(dSpriteBackgroundDataset(transform=Rescale(32),
                                            shapetype = 'dsprite'), **dataloaderparams)
        
        
        if args.model.lower() == "encoderbvae_like":
            net = encoderBVAE_like
        elif args.model.lower() == "decoderbvae_like":
            net = decoderBVAE_like
        elif args.model.lower() == "decoderbvae_like_welu":
            net = decoderBVAE_like_wElu