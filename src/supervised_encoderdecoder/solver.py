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
            self.train_loader = torch.utils.data.DataLoader(dSpriteBackgroundDataset(transform=Rescale(32),
                                            shapetype = 'circle'), **dataloaderparams)
        elif args.dataset.lower() == 'dsprites':
            self.train_loader = torch.utils.data.DataLoader(dSpriteBackgroundDataset(transform=Rescale(32),
                                            shapetype = 'dsprite'), **dataloaderparams)
        
        
        
        if args.model.lower() == "encoderbvae_like":
            net = encoderBVAE_like
            self.modeltype = 'encoder'
        elif args.model.lower() == "decoderbvae_like":
            net = decoderBVAE_like
            self.modeltype = 'decoder'
        elif args.model.lower() == "decoderbvae_like_welu":
            net = decoderBVAE_like_wElu
            self.modeltype = 'decoder'
            
        
        self.net = net(n_latent = args.n_latent, img_channels = args.img_channels)
        self.optimizer = RMSprop(self.net.parameters(), lr=self.lr)
        
        self.loss = MSELoss()
        
    def train(self):
        
        device = torch.device("cuda:0" if self.use_cuda else "cpu")

        for epoch in range(self.max_epochs):
            for samples in self.train_loader: # not sure how long the train_loader spits out data (possibly infinite?)
                
                self.net.zero_grad()
                
                # get current batch and push to device
                # ([:, None, :, :] currently because channels are not existent yet in the Dataset output)
                img_batch, code_batch = samples['image'][:,None,:,:].float().to(device), samples['latents'].float().to(device)
                
                # scale the coordinates such that both the circle and the gaussian center have the same scale
                code_batch[:, 2:] = code_batch[:,2:] /  32
                
                if self.modeltype == 'encoder':
                    input_batch = img_batch
                    output_batch = code_batch
                elif self.modeltype == 'decoder':
                    input_batch = code_batch
                    output_batch = img_batch
                                    
                predicted_batch = self.net(input_batch)
                actLoss = self.loss(predicted_batch, output_batch)
                
                actLoss.backward()
                self.optimizer.step()
