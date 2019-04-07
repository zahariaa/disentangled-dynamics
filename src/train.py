#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trains Encoder (so far)

04/06/2019


TODO:
    - checkpointing
    - visualizing results (loss curve, predictedlabels/reconstructed images)
    - training decoder
    
    
"""

import torch
from torch.nn import MSELoss
from torch.optim import Adam

from dspritesb import dSpriteBackgroundDataset, Rescale
from supervised_encoderdecoder.encoders import encoderBVAE_like

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6} # how many (CPU) processes for data

max_epochs = 100

train_loader = torch.utils.data.DataLoader(dSpriteBackgroundDataset(transform=Rescale(32),
                                            shapetype = 'circle'), **params)

"""
# for testing:
train_iter = iter(train_loader)
samples = train_iter.next()
img_batch, label_batch = samples['image'][:,None,:,:].float().to(device), samples['latents'].float().to(device)
"""

model = encoderBVAE_like().to(device)
loss = MSELoss()
optimizer = Adam(model.parameters(), lr=1e-4)

for epoch in range(max_epochs):
    for samples in train_loader: # not sure how long the train_loader spits out data (possibly infinite?)
        
        model.zero_grad()
        
        # get current batch and push to device
        # ([:, None, :, :] currently because channels are not existent yet in the Dataset output)
        img_batch, label_batch = samples['image'][:,None,:,:].float().to(device), samples['latents'].float().to(device)
        
        # scale the coordinates such that both the circle and the gaussian center have the same scale
        label_batch[:, 2:] = label_batch[:,2:] /  32
        
        predicted_label = model(img_batch)
        actLoss = loss(predicted_label, label_batch)
        
        actLoss.backward()
        optimizer.step()
        
        print("current loss: %0.2e" % actLoss)
    


"""
## visualize filter

import matplotlib.pyplot as plt
plt.imshow(model.conv1.weight[3].detach().cpu().numpy().squeeze())

##
"""