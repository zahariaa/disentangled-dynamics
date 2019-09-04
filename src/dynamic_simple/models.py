
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
        recon_loss = nn.functional.binary_cross_entropy(x_recon.view(batch_size,n_frames,x.size(2),x.size(3),x.size(4)), x, reduction='sum').div(batch_size)
    elif distribution == 'gaussian':
        #x_recon = nn.functional.sigmoid(x_recon)
        recon_loss = nn.functional.mse_loss(x_recon.view(batch_size,n_frames,x.size(2),x.size(3),x.size(4)), x, reduction='sum').div(batch_size)
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
    
    return total_kld, dimension_wise_kld, mean_kld

def prediction_loss(mu,mu_pred):
    return 0.5*torch.sum((mu[:2,:]-mu_pred[:2,:])**2)

def loss_function(recon_loss, total_kld, pred_loss, beta = 1, gamma=1):
    
#     print('recon={}, kld={}, pred={}'.format(recon_loss, total_kld, pred_loss))
    beta_vae_loss = recon_loss + beta*total_kld + gamma*pred_loss

    return beta_vae_loss

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



class dynamicVAE32(nn.Module):
    """ encoder/decoder from Higgins for VAE (Chairs, 3DFaces) - image size 32x32x1
        from Table 1 in Higgins et al., 2017, ICLR

        number of latents can be adapted, spatial input dimensions are fixed

    """

    def __init__(self, n_latent = 10, img_channels = 1, n_frames = 10, alpha=0.75):
        super(dynamicVAE32, self).__init__()
        
        self.n_latent = n_latent
        self.img_channels = img_channels
        self.n_frames = n_frames #=T
        self.alpha = 1.0 #nn.Parameter(torch.FloatTensor(1))

        # encoder
        self.conv1 = nn.Conv2d(in_channels = img_channels, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)     # B*T, 32, 16, 16
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)               # B*T, 32, 8, 8
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)               # B*T, 64, 4, 4
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)               # B*T, 64, 2, 2

        self.fc_enc_mu = nn.Linear(256, n_latent, bias = True)
        self.fc_enc_mu_pred = nn.Linear(256, n_latent, bias = True)
        self.fc_enc_logvar = nn.Linear(256, n_latent, bias = True)
        
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

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        x = x.view(-1,self.img_channels,x.shape[-2],x.shape[-1])
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))        
        mu = self.fc_enc_mu(x.view(-1, 256))
        logvar = self.fc_enc_logvar(x.view(-1, 256))
        return mu, logvar

    def decode(self, z):
        
        x = self.fc_dec(z).view(-1,64,2,2)
        x = torch.nn.functional.elu(self.convT4(x))
        x = torch.nn.functional.elu(self.convT3(x))
        x = torch.nn.functional.elu(self.convT2(x))
        x = torch.nn.functional.sigmoid(self.convT1(x)) # maybe use sigmoid instead here?
        x = x.view(-1,self.n_frames,self.img_channels,x.shape[-2],x.shape[-1])
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        
        z = self.reparametrize(mu, logvar)

        ## Set up mu prediction variables here
        mu1 = mu.view(-1,self.n_frames,self.n_latent)
        mu2 = torch.zeros_like(mu1)
        mu_pred = torch.zeros_like(mu1)
        # Compute prediction
        mu2[:,:-1,:] = mu1[:,1:,:]
        mu_pred[:,2:,:] = mu2[:,:-2,:] + self.alpha*(mu2[:,:-2,:]-mu1[:,:-2,:])
        
        return self.decode(z), mu, logvar, mu_pred.view_as(mu)

class dynamicVAE32a(nn.Module):
    """ encoder/decoder from Higgins for VAE (Chairs, 3DFaces) - image size 32x32x1
        from Table 1 in Higgins et al., 2017, ICLR

        number of latents can be adapted, spatial input dimensions are fixed

    """

    def __init__(self, n_latent = 10, img_channels = 1, n_frames = 10, alpha=0.75):
        super(dynamicVAE32a, self).__init__()
        
        self.n_latent = n_latent
        self.img_channels = img_channels
        self.n_frames = n_frames #=T
        self.alpha = nn.Parameter(torch.FloatTensor(1))

        # encoder
        self.conv1 = nn.Conv2d(in_channels = img_channels, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)     # B*T, 32, 16, 16
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)               # B*T, 32, 8, 8
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)               # B*T, 64, 4, 4
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)               # B*T, 64, 2, 2

        self.fc_enc_mu = nn.Linear(256, n_latent, bias = True)
        self.fc_enc_mu_pred = nn.Linear(256, n_latent, bias = True)
        self.fc_enc_logvar = nn.Linear(256, n_latent, bias = True)
        
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

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        x = x.view(-1,self.img_channels,x.shape[-2],x.shape[-1])
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))        
        mu = self.fc_enc_mu(x.view(-1, 256))
        logvar = self.fc_enc_logvar(x.view(-1, 256))
        return mu, logvar

    def decode(self, z):
        
        x = self.fc_dec(z).view(-1,64,2,2)
        x = torch.nn.functional.elu(self.convT4(x))
        x = torch.nn.functional.elu(self.convT3(x))
        x = torch.nn.functional.elu(self.convT2(x))
        x = torch.nn.functional.sigmoid(self.convT1(x)) # maybe use sigmoid instead here?
        x = x.view(-1,self.n_frames,self.img_channels,x.shape[-2],x.shape[-1])
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        
        z = self.reparametrize(mu, logvar)

        ## Set up mu prediction variables here
        mu1 = mu.view(-1,self.n_frames,self.n_latent)
        mu2 = torch.zeros_like(mu1)
        mu_pred = torch.zeros_like(mu1)
        # Compute prediction
        mu2[:,:-1,:] = mu1[:,1:,:]
        mu_pred[:,2:,:] = mu2[:,:-2,:] + self.alpha*(mu2[:,:-2,:]-mu1[:,:-2,:])
        
        return self.decode(z), mu, logvar, mu_pred.view_as(mu)

    