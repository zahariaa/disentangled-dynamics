#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:22:30 2019

@author: benjamin
"""


import torch
from torch.optim import RMSprop
from torchvision import transforms

from tqdm import tqdm
import os
import pickle

import sys
sys.path.append("..") # Adds higher directory to python modules path.

from data.dspritesb import dSpriteBackgroundDataset
from models import staticVAE32
from models import loss_function, reconstruction_loss, kl_divergence
from models import normalized_beta_from_beta, beta_from_normalized_beta


class DataGather(object):
    """ 
        contains a dict that is updated with training related information
        
        filename: for checkpointing 
        
        might need different/new keys depending on trained model
    
        modified from
        from https://github.com/1Konny/Beta-VAE/blob/master/solver.py
    """
    def __init__(self, filename):
        
        self.filename = '{}.pkl'.format(filename)
        
        if os.path.isfile(self.filename):
            self.load_data_dict()
        else:
            self.data = self.get_empty_data_dict()            

    def get_empty_data_dict(self):
        return dict(iter=[],
                    total_loss=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    target=[],
                    reconstructed=[],)

    def save_data_dict(self):
        f = open(self.filename,"wb")
        pickle.dump(self.data,f)
        f.close()

    def load_data_dict(self):
        f = open(self.filename,"rb")
        self.data = pickle.load(f)
        f.close()

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


class Solver(object):
    
    def __init__(self, args):
        
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        
        """ 
            /begin of non-generic part that needs to be modified for each new model
        """
        
        # model name is used for checkpointing (and here for setting self.net)
        self.model = args.model
        
        self.image_size = args.image_size
	self.n_latent = args.n_latent
        self.img_channels = args.img_channels
        
        # beta for beta VAE
        if args.beta_is_normalized:
            self.beta_norm = args.beta
            self.beta = beta_from_normalized_beta(self.beta_norm,N= self.image_size * self.image_size * self.img_channels,M=args.n_latent)
        else:
            self.beta = args.beta
            self.beta_norm = normalized_beta_from_beta(self.beta,N= self.image_size * self.image_size * self.img_channels,M=args.n_latent)            
        
        dataloaderparams = {'batch_size': args.batch_size,
                            'shuffle': args.shuffle,
                            'num_workers': args.num_workers}

        if args.dataset.lower() == 'dsprites_circle':
            self.train_loader = torch.utils.data.DataLoader(dSpriteBackgroundDataset(transform=transforms.Resize((self.image_size,self.image_size)),
                                            shapetype = 'circle'), **dataloaderparams)
        elif args.dataset.lower() == 'dsprites':
            self.train_loader = torch.utils.data.DataLoader(dSpriteBackgroundDataset(transform=transforms.Resize((self.image_size,self.image_size)),
                                            shapetype = 'dsprite'), **dataloaderparams)
        
        
        if args.model.lower() == "staticvae32":
            net = staticVAE32
            self.modeltype = 'staticVAE'
        else:
            raise Exception('model "%s" unknown' % args.model)
            
        
        self.net = net(n_latent = self.n_latent, img_channels = self.img_channels).to(self.device)
        
        self.reconstruction_loss = reconstruction_loss
        self.kl_divergence = kl_divergence
        
        self.loss = loss_function

        self.lr = args.lr
        self.optim = RMSprop(self.net.parameters(), lr=self.lr)
        
        
        """ 
            /end of non-generic part that needs to be modified for each new model
        """       
                
        self.max_iter = args.max_iter
        self.global_iter = 0
        
        # prepare checkpointing
        if not os.path.isdir(args.ckpt_dir):
            os.mkdir(args.ckpt_dir)
        
        self.ckpt_dir = args.ckpt_dir
        self.load_last_checkpoint = args.load_last_checkpoint
        self.ckpt_name = '{}_nlatent={}_betanorm={}_{}_last'.format(self.model.lower(), self.n_latent, self.beta_norm, args.dataset.lower())
        
        
        self.save_step = args.save_step
        if self.load_last_checkpoint is not None:
            self.load_checkpoint(self.ckpt_name)        
                
        self.display_step = args.display_step
        
        # will store training-related information
        self.trainstats_gather_step = args.trainstats_gather_step
        self.trainstats_dir = args.trainstats_dir
        if not os.path.isdir(self.trainstats_dir):
            os.mkdir(self.trainstats_dir)        
        self.trainstats_fname = '{}_nlatent={}_betanorm={}_{}'.format(self.model.lower(), self.n_latent, self.beta_norm, args.dataset.lower())
        self.gather = DataGather(filename = os.path.join(self.trainstats_dir, self.trainstats_fname))
        
        
        
    def train(self):
        
        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        
        out = False        

        running_loss_terminal_display = 0.0 # running loss for the trainstats (gathered and pickeled)

        running_loss_trainstats = 0.0 # running loss for the trainstats (gathered and pickeled)
        """ /begin of non-generic part (might need to be adapted for different models / data)"""
        running_recon_loss_trainstats = 0.0
        running_total_kld = 0.0
        running_dim_wise_kld = 0.0
        running_mean_kld = 0.0
        
        while not out:
            for [samples,latents] in self.train_loader: # not sure how long the train_loader spits out data (possibly infinite?)
                
                self.global_iter += 1
                pbar.update(1)
                
                self.net.zero_grad()               
               
                # get current batch and push to device
                img_batch, _ = samples.to(self.device), latents.to(self.device)
                
                # in VAE, input = output/target
                if self.modeltype == 'staticVAE':
                    input_batch = img_batch
                    output_batch = img_batch

                                    
                predicted_batch, mu, logvar = self.net(input_batch)
                
                recon_loss = self.reconstruction_loss(x = output_batch, x_recon = predicted_batch)
                total_kld, dimension_wise_kld, mean_kld = self.kl_divergence(mu, logvar)
                
                actLoss = self.loss(recon_loss=recon_loss, total_kld=total_kld, beta = self.beta)
                
                actLoss.backward()
                self.optim.step()                

                running_loss_terminal_display += actLoss.item()
                
                running_loss_trainstats += actLoss.item()
                running_recon_loss_trainstats += recon_loss.item()
                running_total_kld += total_kld.item()
                running_dim_wise_kld += dimension_wise_kld.cpu().detach().numpy()
                running_mean_kld += mean_kld.item()
                
                # update gather object with training information
                if self.global_iter % self.trainstats_gather_step == 0:
                    running_loss_trainstats = running_loss_trainstats / self.trainstats_gather_step
                    running_recon_loss_trainstats = running_recon_loss_trainstats / self.trainstats_gather_step
                    running_total_kld = running_total_kld / self.trainstats_gather_step
                    running_dim_wise_kld = running_dim_wise_kld / self.trainstats_gather_step
                    running_mean_kld = running_mean_kld / self.trainstats_gather_step
                    self.gather.insert(iter=self.global_iter,
                                       total_loss=running_loss_trainstats,
                                       target = output_batch[0].detach().cpu().numpy(),
                                       reconstructed = predicted_batch[0].detach().cpu().numpy(),
                                       recon_loss=running_recon_loss_trainstats,
                                       total_kld = running_total_kld,
                                       dim_wise_kld = running_dim_wise_kld,
                                       mean_kld = running_mean_kld,
                                       )
                    running_loss_trainstats = 0.0
                    running_recon_loss_trainstats = 0.0
                    running_total_kld = 0.0
                    running_dim_wise_kld = 0.0
                    running_mean_kld = 0.0                    
                
                """ /end of non-generic part"""

                if self.global_iter % self.display_step == 0:
                    pbar.write('iter:{}, loss:{:.3e}'.format(self.global_iter, running_loss_terminal_display / self.display_step))
                    running_loss_terminal_display = 0.0

                if self.global_iter % self.save_step == 0:
                    self.save_checkpoint(self.ckpt_name)
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))
                    self.gather.save_data_dict()
                
                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()

    
    def save_checkpoint(self, filename, silent=True): 
        """ saves model and optimizer state as checkpoint 
            modified from https://github.com/1Konny/Beta-VAE/blob/master/solver.py                   
        """
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}

        
        states = {'iter':self.global_iter,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))
    
    def load_checkpoint(self, filename):
        """ loads model and optimizer state from checkpoint 
            modified from https://github.com/1Konny/Beta-VAE/blob/master/solver.py                   
        """        
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

