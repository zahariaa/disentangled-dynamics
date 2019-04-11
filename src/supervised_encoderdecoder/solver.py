#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:22:30 2019

@author: benjamin
"""


import torch
from torch.nn import MSELoss
from torch.optim import Adam, RMSprop

from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import warnings

import sys
sys.path.append("..") # Adds higher directory to python modules path.

from data.dspritesb import dSpriteBackgroundDataset, Rescale
from models.encoders import encoderBVAE_like
from models.decoders import decoderBVAE_like, decoderBVAE_like_wElu


class Solver(object):
    
    def __init__(self, args):
        
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")

        
        self.max_iter = args.max_iter
        self.global_iter = 0
        
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
        else:
            raise Exception('model "%s" unknown' % args.model)
            
        
        self.net = net(n_latent = args.n_latent, img_channels = args.img_channels).to(self.device)
        self.optim = RMSprop(self.net.parameters(), lr=self.lr)
        
        self.loss = MSELoss()
        
        # prepare checkpointing
        if not os.path.isdir(args.ckpt_dir):
            os.mkdir(args.ckpt_dir)
        
        self.ckpt_dir = args.ckpt_dir
        self.load_last_checkpoint = args.load_last_checkpoint
        self.ckpt_name = '{}_{}_last'.format(self.model.lower(), args.dataset.lower())
        
        
        self.save_step = args.save_step
        if self.load_last_checkpoint is not None:
            self.load_checkpoint(self.ckpt_name)        
        
        
        self.display_step = args.display_step
        
        
    def train(self):
        
        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        
        out = False
        
        while not out:
            for samples in self.train_loader: # not sure how long the train_loader spits out data (possibly infinite?)
                
                self.global_iter += 1
                
                self.net.zero_grad()
                
                # get current batch and push to device
                # ([:, None, :, :] currently because channels are not existent yet in the Dataset output)
                img_batch, code_batch = samples['image'][:,None,:,:].float().to(self.device), samples['latents'].float().to(self.device)
                
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
                self.optim.step()
                
                if self.global_iter % self.display_step == 0:
                    pbar.write('iter:{}, loss:{:.3e}'.format(self.global_iter, actLoss))

                
                if self.global_iter % self.save_step == 0:
                    self.save_checkpoint('last')
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))
                
                
                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()

    """
        checkpointing from:
        https://github.com/1Konny/Beta-VAE/blob/master/solver.py
    """                    
    
    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        """
        win_states = {'recon':self.win_recon,
                      'kld':self.win_kld,
                      'mu':self.win_mu,
                      'var':self.win_var,}
        """
        win_states = {'none':None}
        
        states = {'iter':self.global_iter,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))
    
    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            """
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            """
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def visualizeReconstruction(self, image, reconstruction):
        """ so far only copied from initial train.py
        
            needs to be modified (e.g., integrated with visdom)
            
        """
        
        plt.subplot(1,2,1)
        plt.imshow(image[0,:,:,:].detach().cpu().numpy().squeeze())
        plt.title('true')
            
        plt.subplot(1,2,2)
        plt.imshow(reconstruction[0,:,:,:].detach().cpu().numpy().squeeze())
        plt.title('reconstructed')
            
        plt.pause(.05)
        plt.show()
