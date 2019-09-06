
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bVAE (Higgins, 2017) or inspired by the paper

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


def reconstruction_loss(x, x_recon, distribution='gaussian'):
    """
     from https://github.com/1Konny/Beta-VAE/blob/master/solver.py
     
    """
    
    batch_size = x.size(0)
    assert batch_size != 0
    n_frames = x.size(1)
    
    if distribution == 'bernoulli':
        recon_loss = nn.functional.binary_cross_entropy(
            x_recon[:,1:,:,:,:].contiguous().view(batch_size,n_frames-2,x.size(2),x.size(3),x.size(4)),
            x[:,2:,:,:,:].contiguous(), reduction='sum').div(batch_size)
    elif distribution == 'gaussian':
        #x_recon = nn.functional.sigmoid(x_recon)
        recon_loss = nn.functional.mse_loss(
            x_recon[:,1:,:,:,:].contiguous().view(batch_size,n_frames-2,x.size(2),x.size(3),x.size(4)),
            x[:,2:,:,:,:].contiguous(), reduction='sum').div(batch_size)
    else:
        recon_loss = None
        
    return recon_loss

def prediction_loss(mu,mu_pred):
    return 0.5*torch.sum((mu[:2,:]-mu_pred[:2,:])**2)

def loss_function(recon_loss):
    
#     print('recon={}'.format(recon_loss))
    return recon_loss

class dynamicVAE64(nn.Module):
    """ encoder/decoder from Higgins for VAE (Chairs, 3DFaces) - image size 64x64x1
        from Table 1 in Higgins et al., 2017, ICLR

        number of latents can be adapted, spatial input dimensions are fixed

    """

    def __init__(self, n_latent = 10, img_channels = 1):
        super(staticVAE64, self).__init__()
        
        self.n_latent = n_latent
        self.img_channels = img_channels

        # encoder
        self.conv1 = nn.Conv2d(in_channels = img_channels, out_channels = 32, kernel_size = 4, stride = 2, padding = 0)     # B, 32, 31, 31
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2, padding = 0)               # B, 32, 14, 14
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 0)               # B, 64, 6, 6
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 4, stride = 2, padding = 0)               # B, 64, 2, 2

        self.fc_enc_mu = nn.Linear(256, n_latent, bias = True)
        self.fc_enc_logvar = nn.Linear(256, n_latent, bias = True)

        # decoder
        self.fc_dec = nn.Linear(n_latent, 256, bias = True)                         # B, 256 (after .view(): B, 64, 2, 2)

        self.convT4 = nn.ConvTranspose2d(64, 64, 4, 2, 0)                       # B, 64, 6, 6
        self.convT3 = nn.ConvTranspose2d(64, 32, 4, 2, 0)                       # B, 32, 14, 14
        self.convT2 = nn.ConvTranspose2d(32, 32, 4, 2, 0, output_padding = 1)   # B, 32, 31, 31
        self.convT1 = nn.ConvTranspose2d(32, img_channels, 4, 2, 0)             # B, img_channels, 64, 64

        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            kaiming_init(m)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))        
        mu = self.fc_enc_mu(x.view(-1, 256))
        logvar = self.fc_enc_logvar(x.view(-1, 256))
        return mu, logvar

    def decode(self, z):
        
        x = self.fc_dec(z).view(-1,64,2,2)
        x = torch.relu(self.convT4(x))
        x = torch.relu(self.convT3(x))
        x = torch.relu(self.convT2(x))
        x = torch.relu(self.convT1(x)) # maybe use sigmoid instead here?
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar



class inertiaAE32(nn.Module):
    """ encoder/decoder from Higgins for VAE (Chairs, 3DFaces), adapted to have
        representational inertia, and no variational component - image size 32x32x1
        from Table 1 in Higgins et al., 2017, ICLR

        number of latents can be adapted, spatial input dimensions are fixed

    """

    def __init__(self, n_latent = 10, img_channels = 1, n_frames = 10, gamma=0.75):
        super(inertiaAE32, self).__init__()
        
        self.n_latent = n_latent
        self.img_channels = img_channels
        self.n_frames = n_frames #=T
        self.gamma = gamma #nn.Parameter(torch.FloatTensor(1))

        # encoder
        self.conv1 = nn.Conv2d(in_channels = img_channels, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)     # B*T, 32, 16, 16
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)               # B*T, 32, 8, 8
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)               # B*T, 64, 4, 4
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)               # B*T, 64, 2, 2

        self.fc_enc_mu = nn.Linear(256, n_latent, bias = True)
        self.fc_enc_mu_pred = nn.Linear(256, n_latent, bias = True)

        # decoder
        self.fc_dec = nn.Linear(n_latent, 256, bias = True)                         # B*T, 256 (after .view(): B*T, 64, 2, 2)

        self.convT4 = nn.ConvTranspose2d(64, 64, 3, 2, 1, 1)                       # B*T, 64, 4, 4
        self.convT3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)                       # B*T, 32, 8, 8
        self.convT2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)                       # B*T, 32, 16, 16
        self.convT1 = nn.ConvTranspose2d(32, img_channels, 3, 2, 1, 1)             # B*T, img_channels, 32, 32

        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            kaiming_init(m)

    def encode(self, x):
        # all but last x make all but mu at first time point
        x = x[:,:-1,:,:,:].contiguous().view(-1,self.img_channels,x.shape[-2],x.shape[-1])
        # Note: if you really want to save gpu memory ops, do the above indexing in the solver
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))        
        mu_enc = self.fc_enc_mu(x.view(-1, 256)) # this has T-1 frames
        
        #### INERTIA
        mu_enc = mu_enc.view(-1,self.n_frames-1,self.n_latent)
        # first mu_pred is just mu_enc
        mu_pred = torch.zeros_like(mu_enc)
        mu = torch.zeros_like(mu_enc)
        mu_pred[:,0,:] = mu_enc[:,0,:]
        # as a consequence, first mu is also just mu_enc
        mu[:,0,:] = mu_pred[:,0,:]
        # second mu_pred is same as first mu_pred
        mu_pred[:,1,:] = mu_pred[:,0,:]
        for i in range(1,self.n_frames-1):
            mu[:,i,:] = (1-self.gamma)*mu_enc[:,i,:] + self.gamma*mu_pred[:,i,:]
            if i < self.n_frames-2:
                mu_pred[:,i+1,:] = 1*(mu[:,i,:] - mu[:,i-1,:]) + mu[:,i,:]

        return mu, mu_enc, mu_pred

    def decode(self, z):
        x = self.fc_dec(z).view(-1,64,2,2)
        x = torch.nn.functional.elu(self.convT4(x))
        x = torch.nn.functional.elu(self.convT3(x))
        x = torch.nn.functional.elu(self.convT2(x))
        x = torch.nn.functional.sigmoid(self.convT1(x)) # maybe use sigmoid instead here?
        x = x.view(-1,self.n_frames-1,self.img_channels,x.shape[-2],x.shape[-1])
        return x
    
    def forward(self, x):
        mu, mu_enc, mu_pred = self.encode(x)

        return self.decode(mu), mu, mu_enc, mu_pred
