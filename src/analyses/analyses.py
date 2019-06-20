#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyses for bVAE entanglement, etc
"""

import torch

import sys
sys.path.append("..") # Adds higher directory to python modules path.

import matplotlib.pyplot as plt
import numpy as np

from data.dspritesb import dSpriteBackgroundDataset
from torchvision import transforms
ds = dSpriteBackgroundDataset(transform=transforms.Resize((32,32)),shapetype = 'circle')

# Build sweeps through model ...
def sweepCircleLatents(model,latents=np.linspace(0,1,16),def_latents=None):
    """sweepCircleLatents(model,latents,def_latents):
        generates input images that sweep through each latent variable,
        and evaluates them on given model
          model       = loaded model, e.g., vae = staticVAE32(n_latent = 4)
          latents     = latents to sweep through. defaults to
                          np.linspace(0,1,16)
          def_latents = 'default latents': defines the non-swept latents.
                         defaults to [0.5,0.5,0.5,0.5] if None
        ---e.g.,---
        yhat, x = sweepCircleLatents(vae)
    """
    # Initialization
    nsweep = len(latents)
    if type(model).__name__ == 'staticVAE32':
        n_latent = model.n_latent
        encoder  = model.encode
    elif type(model).__name__ == 'encoderBVAE_like':
        n_latent = model.fc.out_features
        encoder  = model
    if def_latents is None:
        def_latents = 0.5*np.ones(n_latent)
        
    # Generate stimulus sweeps
    x = torch.zeros((nsweep*n_latent,1,32,32))
    for i in np.arange(0,nsweep):
        x[0*nsweep+i,:,:,:] = ds.arbitraryCircle(latents[i],def_latents[1],def_latents[2],def_latents[3])
        x[1*nsweep+i,:,:,:] = ds.arbitraryCircle(def_latents[0],latents[i],def_latents[2],def_latents[3])
        x[2*nsweep+i,:,:,:] = ds.arbitraryCircle(def_latents[0],def_latents[1],latents[i],def_latents[3])
        x[3*nsweep+i,:,:,:] = ds.arbitraryCircle(def_latents[0],def_latents[1],def_latents[2],latents[i])

    # ... and evaulate them all at once
    if type(model).__name__ == 'staticVAE32':
        yhat,_ = encoder(x)
    elif type(model).__name__ == 'encoderBVAE_like':
        yhat   = encoder(x)
    return yhat,x

# Plot sweeps through model
def plotCircleSweep(x=None,nimgs=5):
    """plotCircleSweep(yhat,x):
        plots a subset of stimuli,
        generated from sweepCircleLatents()
        ---e.g.,---
        yhat, x = sweepCircleLatents(vae)
        plotCircleSweep(x)
        alternatively,
        plotCircleSweep(sweepCircleLatents(vae))
    """
    # Initialization
    if x is None and type(nimgs) is tuple:
        x    = yhat[1]
    
    # Start a-plottin'
    fig, ax = plt.subplots(nimgs,4,figsize=(9, 15), dpi= 80, facecolor='w', edgecolor='k')
    
    for latentdim in range(4):
        cnt = -1
        for img in np.linspace(0,15,nimgs).astype(int):
            cnt+=1
            plt.sca(ax[cnt,latentdim])
            plt.set_cmap('gray')
            ax[cnt,latentdim].imshow(
                x[latentdim*16+img,:,:,:].squeeze(), vmin=0, vmax=1)
            plt.axis('off')
            
    return fig, ax

def plotLatentsSweep(yhat,nmodels=1):
    """plotLatentsSweep(yhat):
        plots model latents and a subset of the corresponding stimuli,
        generated from sweepCircleLatents()
        ---e.g.,---
        yhat, x = sweepCircleLatents(vae)
        plotCircleSweep(yhat,x)
        alternatively,
        plotLatentsSweep(sweepCircleLatents(vae))
    """
    # Initialization
    if type(yhat) is tuple:
        yhat = yhat[0]
    
    # Start a-plottin'
    fig, ax = plt.subplots(nmodels,4,figsize=(9, 15), dpi= 80, facecolor='w', edgecolor='k', sharey='row',sharex='col')
    
    for latentdim in range(4):
        if nmodels > 1:
            for imodel in range(nmodels):
                plt.sca(ax[imodel,latentdim])
                plt.plot(yhat[imodel][latentdim*16+np.arange(0,16),:].detach().numpy())
#                 ax[imodel,latentdim].set_aspect(1./ax[imodel,latentdim].get_data_ratio())


                ax[imodel,latentdim].spines['top'].set_visible(False)
                ax[imodel,latentdim].spines['right'].set_visible(False)
                if latentdim>0:
                    ax[imodel,latentdim].spines['left'].set_visible(False)
#                     ax[imodel,latentdim].set_yticklabels([])
                    ax[imodel,latentdim].tick_params(axis='y', length=0)

    #             if imodel<nmodels-1 or latentdim>0:
                ax[imodel,latentdim].spines['bottom'].set_visible(False)
                ax[imodel,latentdim].set_xticklabels([])
                ax[imodel,latentdim].tick_params(axis='x', length=0)
        else:
            imodel=0
            plt.sca(ax[latentdim])
            plt.plot(yhat[latentdim*16+np.arange(0,16),:].detach().numpy())
            ax[latentdim].set_aspect(1./ax[latentdim].get_data_ratio())


            ax[latentdim].spines['top'].set_visible(False)
            ax[latentdim].spines['right'].set_visible(False)
            if latentdim>0:
                ax[latentdim].spines['left'].set_visible(False)
                ax[latentdim].tick_params(axis='y', length=0)

#             if imodel<nmodels-1 or latentdim>0:
            ax[latentdim].spines['bottom'].set_visible(False)
            ax[latentdim].set_xticklabels([])
            ax[latentdim].tick_params(axis='x', length=0)
            
            
    return fig, ax

def colorAxisNormalize(colorbar):
    """colorAxisNormalize(colorbar):
        normalizes a color axis so it is centered on zero.
        useful for diverging colormaps
        (e.g., cmap='bwr': blue=negative, red=positive, white=0)
        input is already initialized colorbar object from a plot
        ---e.g.,---
        corr_vae = np.corrcoef(yhat_vae.detach().numpy().T)
        plt.set_cmap('bwr')
        plt.imshow(corr_vae)
        cb = plt.colorbar()
        colorAxisNormalize(cb)
        ---or---
        colorAxisNormalize(plt.colorbar())
    """
    cm = np.max(np.abs(colorbar.get_clim()))
    colorbar.set_clim(-cm,cm)
    
def showReconstructionsAndErrors(model):
    """showReconstructionsAndErrors(model):
        generates random inputs, runs them through a specified model
        to generate their reconstructions. plots the inputs,
        reconstructions, and their difference
        ---e.g.---
        from staticvae.models import staticVAE32
        vae = staticVAE32(n_latent = 4)
        vae.eval()
        checkpoint = torch.load('../staticvae/trained/staticvae32_dsprites_circle_last_500K',map_location='cpu')
        vae.load_state_dict(checkpoint['model_states']['net'])
        showReconstructionsAndErrors(model)
    """
    fig=plt.figure(figsize=(18, 16), dpi= 80, facecolor='w',
                   edgecolor='k')
    cnt = 0
    for ii in range(12):
        x,label = ds[np.random.randint(1000)]
        x = x[np.newaxis, :, :]

        mu,logvar = model.encode(x.float())
        recon = model.decode(mu).detach()
        diff = x - recon

        cnt += 1
        ax = plt.subplot(6,6,cnt)
        plt.set_cmap('gray')
        ax.imshow(x.squeeze(), vmin=0, vmax=1)
        plt.title('true')
        plt.axis('off')

        cnt += 1
        ax = plt.subplot(6,6,cnt)    
        ax.imshow(recon.squeeze(), vmin=0, vmax=1)
        plt.title('recon')
        plt.axis('off')

        cnt += 1
        ax = plt.subplot(6,6,cnt)    
        plt.set_cmap('bwr')
        img = ax.imshow(diff.numpy().squeeze())
        colorAxisNormalize(fig.colorbar(img))
        plt.title('diff')
        plt.axis('off')
