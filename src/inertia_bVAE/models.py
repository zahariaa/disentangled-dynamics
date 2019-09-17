
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

def beta_from_normalized_beta(beta_normalized, N, M):
    """
       input:
           beta_normalized
           N: total number of values in each input image (pixels times channels)
           M: number of latent dimensions
    
        computes beta = beta_normalized * N / M
        
        given the relationship:
            
            \beta_\text{norm} = \frac{\beta M}{N}
            
            from the Higgins, 2017, bVAE paper (p. 15)
    """
    
    beta = beta_normalized * N / M
    return beta

def normalized_beta_from_beta(beta, N, M):
    """
       input:
           beta
           N: total number of values in each input image (pixels times channels)
           M: number of latent dimensions
    
        computes beta_normalized = beta * latent_code_size / image_size
        
        given the relationship:
            
            \beta_\text{norm} = \frac{\beta M}{N}
            
            from the Higgins, 2017, bVAE paper (p. 15)
    """
    
    beta_normalized = beta * M / N
    return beta_normalized

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

def kl_divergence(mu, logvar):
    """
     from https://github.com/1Konny/Beta-VAE/blob/master/solver.py
     
    """    
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, keepdim=True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, keepdim=True)
    
    return total_kld.sum(), dimension_wise_kld.mean(1), mean_kld.mean()

def prediction_loss(mu,mu_pred):
    return 0.5*torch.sum((mu[2:,:]-mu_pred[2:,:])**2)

def loss_function(recon_loss, total_kld, beta = 1):

#     print('recon={}, kld={}'.format(recon_loss, total_kld))
    beta_vae_loss = recon_loss + beta*total_kld

    return beta_vae_loss

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



class inertiaVAE32(nn.Module):
    """ encoder/decoder from Higgins for VAE (Chairs, 3DFaces), adapted to have
        representational inertia, and no variational component - image size 32x32x1
        from Table 1 in Higgins et al., 2017, ICLR

        number of latents can be adapted, spatial input dimensions are fixed

    """

    def __init__(self, n_latent = 10, img_channels = 1, n_frames = 10, gamma=0.75):
        super(inertiaVAE32, self).__init__()
        
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
        self.fc_enc_logvar = nn.Linear(256, n_latent, bias = True)

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

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        # compute n_frames locally
        n_frames = x.shape[1]
        # all but last x make all but mu at first time point
        x = x[:,:-1,:,:,:].contiguous().view(-1,self.img_channels,x.shape[-2],x.shape[-1])
        # Note: if you really want to save gpu memory ops, do the above indexing in the solver
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        # these have T-1 frames
        mu_enc = self.fc_enc_mu(x.view(-1, 256)).view(-1,n_frames-1,self.n_latent)
        logvar_enc = self.fc_enc_logvar(x.view(-1, 256)).view(-1,n_frames-1,self.n_latent)
        
        #### INERTIA (mu's)
        # first mu_pred is just mu_enc
        mu_pred = torch.zeros_like(mu_enc)
        mu = torch.zeros_like(mu_enc)
        mu_pred[:,0,:] = mu_enc[:,0,:]
        # as a consequence, first mu is also just mu_enc
        mu[:,0,:] = mu_pred[:,0,:]
        # second mu_pred is same as first mu_pred
        mu_pred[:,1,:] = mu_pred[:,0,:]
#         #### INERTIA (logvar's)
#         var_enc = torch.exp(0.5*logvar_enc)
#         # first logvar_pred is just logvar_enc
#         var_pred = torch.Tensor(size=var_enc.size())# exp(0.5*logvar_enc)
#         var = torch.Tensor(size=var_enc.size())#var = torch.exp(0.5*logvar_enc)
#         var_pred[:,0,:] = var_enc[:,0,:]
#         # as a consequence, first logvar is also just logvar_enc
#         var[:,0,:] = var_pred[:,0,:]
#         # second logvar_pred is same as first logvar_pred
#         var_pred[:,1,:] = var_pred[:,0,:]

        #### INERTIA LOOP (mu's and logvar's)
        for i in range(1,n_frames-1):
            mu[:,i,:] = (1-self.gamma)*mu_enc[:,i,:] + self.gamma*mu_pred[:,i,:]
            # variance for weighted mixture of gaussians has additional term accounting for the weighted dispersion of the means
# ######## THIS LINE IS MESSING UP AUTOGRAD
# #             logvar[:,i,:] = (1-self.gamma)*logvar_enc[:,i,:] + self.gamma*logvar_pred[:,i,:] \
# #                           + (1-self.gamma)*(mu_enc[:,i,:]**2) + self.gamma*(mu_pred[:,i,:]**2) - mu[:,i,:]**2
#             var[:,i,:] = (1-self.gamma)*var_enc[:,i,:] + self.gamma*var_pred[:,i,:] \
#                        + (1-self.gamma)*(mu_enc[:,i,:]**2) + self.gamma*(mu_pred[:,i,:]**2) - mu[:,i,:]**2
# #             var[:,i,:] = var_pred[:,i,:] # this is the problem; autograd doesn't like this in-place operation

            if i < n_frames-2:
                mu_pred[:,i+1,:] = 1*(mu[:,i,:] - mu[:,i-1,:]) + mu[:,i,:]
#                 # Equally weight the variances from the previous two time steps
#                 # Simplified based on expansion of (0.5*mu[:,i-1,:] + 0.5*mu[:,i,:])**2 :
#                 # 0.25*logvar[:,i-1,:]**2 + 0.25*logvar[:,i,:]**2 + 0.5*logvar[:,i-1,:]*logvar[:,i,:]
# #                 logvar_pred[:,i+1,:] = 0.5*logvar[:,i-1,:] + 0.5*logvar[:,i,:] + 0.25*(mu[:,i-1,:]**2) \
# #                                      + 0.25*(mu[:,i,:]**2) - 0.5*mu[:,i-1,:]*mu[:,i,:]
#                 var_pred[:,i+1,:] = var[:,i-1,:] + var[:,i,:] + 0.25*(mu[:,i-1,:]**2) \
#                                      + 0.25*(mu[:,i,:]**2) - 0.5*mu[:,i-1,:]*mu[:,i,:]

        return mu, mu_enc, mu_pred, logvar_enc#2*torch.log(var), logvar_enc, 2*torch.log(var_pred)

    def decode(self, z):
        x = self.fc_dec(z).view(-1,64,2,2)
        x = torch.nn.functional.elu(self.convT4(x))
        x = torch.nn.functional.elu(self.convT3(x))
        x = torch.nn.functional.elu(self.convT2(x))
        x = torch.nn.functional.sigmoid(self.convT1(x)) # maybe use sigmoid instead here?
        x = x.view(-1,self.n_frames-1,self.img_channels,x.shape[-2],x.shape[-1])
        return x
    
    def forward(self, x):
#         mu, mu_enc, mu_pred, logvar, logvar_enc, logvar_pred = self.encode(x)
        mu, mu_enc, mu_pred, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)

        return self.decode(z), mu, mu_enc, mu_pred, logvar#, logvar_enc, logvar_pred
