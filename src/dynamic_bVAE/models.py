
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


def reconstruction_loss(x, x_recon, distribution='gaussian'):
    """
     from https://github.com/1Konny/Beta-VAE/blob/master/solver.py
     
    """
    
    batch_size = x.size(0)
    n_frames = x.size(1)
    assert batch_size != 0
    
    if distribution == 'bernoulli':
        recon_loss = nn.functional.binary_cross_entropy(x_recon.view(batch_size,n_frames,x.size(2),x.size(3),x.size(4)), 
                                                        x, reduction='sum').div(batch_size)
    elif distribution == 'gaussian':
        #x_recon = nn.functional.sigmoid(x_recon)
        recon_loss = nn.functional.mse_loss(x_recon.view(batch_size,n_frames,x.size(2),x.size(3),x.size(4)), 
                                            x, reduction='sum').div(batch_size)
    else:
        recon_loss = None
        
    return recon_loss    

def kl_divergence(mu, covariance_mats, precision_mats):
    """
     from https://github.com/1Konny/Beta-VAE/blob/master/solver.py
     
    """ 
    batch_size = mu.size(0)
    k = mu.size(1)
    assert batch_size != 0
#     if mu.data.ndimension() == 4:
#         mu = mu.view(mu.size(0), mu.size(1))
    # if logvar.data.ndimension() == 4:
    #     logvar = logvar.view(logvar.size(0), logvar.size(1))

    # Compute KL divergence
    klds = torch.zeros(batch_size).to(precision_mats.device)
    for b in range(batch_size):
        klds[b] = 0.5*( torch.trace(covariance_mats[b,:,:]) + mu[b,:].view(-1).dot(mu[b,:].view(-1)) - k * torch.logdet(precision_mats[b,:,:]) )

    total_kld = klds.sum(0).mean(0, keepdim=True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(0).mean(0, keepdim=True)
    
    return total_kld, dimension_wise_kld, mean_kld

def loss_function(recon_loss, total_kld, beta = 1):
    
    beta_vae_loss = recon_loss + beta*total_kld

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

def build_tridiag(D, B, alpha=0.5):
    """ Sigma_inv = build_tridiag(D,B)
        Builds the tridiagonal precision matrix where:
            D : all the diagonal blocks concatenated (1 x nnT)
            B : all the off-diagonal blocks concatenated (1 x nn(T-1))
        The final result is (nT by nT), where
            Sigma_inv = [D_0   B_0^t   0's        ...
                         B_0   D_1     B_1^t  0's ...
                         0's   B_1     D_2    0's ...
                                     .
                                     .
                                     .
                         0's   ...        B_(T-1) D_T]
    """
    # Initialize
    batch_size = D.size(0)
    p2 = D.size(1)-B.size(1) # number of frames (time bins), number of pixels
    t = int(D.size(1)/p2)
    p = int(p2**0.5)
    tridiag = torch.zeros(batch_size,p*t,p*t)
    
    # Go row-block by row-block
    for b in range(batch_size):
        for i in range(t):
            # Build diagonal blocks (D_i)
            tridiag[b,(p*i):(p*(i+1)),(p*i):(p*(i+1))] = D[b,(p2*i):(p2*(i+1))].view(p,p)
            # Build off-diagonal blocks (B_i)
            if i > 0:
                tridiag[b,(p*i):(p*(i+1)),(p*(i-1)):(p*i)] = B[b,(p2*(i-1)):(p2*i)].view(p,p)
            if i < t-1:
                tridiag[b,(p*i):(p*(i+1)),(p*(i+1)):(p*(i+2))] = torch.t(B[b,(p2*i):(p2*(i+1))].view(p,p))
            
    # Force to be positive semi-definite
    tridiag = tridiag + alpha*torch.eye(p*t).repeat(batch_size,1,1)
    
    return tridiag

class dynamicVAE64(nn.Module):
    """ encoder/decoder from Higgins for VAE (Chairs, 3DFaces) - image size 64x64x1
        from Table 1 in Higgins et al., 2017, ICLR

        number of latents can be adapted, spatial input dimensions are fixed

    """

    def __init__(self, n_latent = 10, img_channels = 1):
        super(dynamicVAE64, self).__init__()
        
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
        self.fc_dec = nn.Linear(n_latent, 256, bias = True)                     # B, 256 (after .view(): B, 64, 2, 2)

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

    def __init__(self, n_latent = 10, img_channels = 1, n_frames = 10):
        super(dynamicVAE32, self).__init__()
        
        self.n_latent = n_latent
        self.img_channels = img_channels
        self.n_frames = n_frames #=T

        # encoder
        self.conv1 = nn.Conv2d(in_channels = img_channels, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)     # B*T, 32, 16, 16
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)               # B*T, 32, 8, 8
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)               # B*T, 64, 4, 4
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)               # B*T, 64, 2, 2
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)               # B*T, 64, 1, 1
        
        # Construct mu_t vector and D_t and B_t matrices which comprise mean and the inverse covariance
# they must be constructed separately, so there is inappropriate cross-talk across time
        self.fc_enc_mu = nn.Linear(64*n_frames, n_latent*n_frames, bias = True)     # mu_t = NN_{phi_mu}(x_t), stacked
        self.fc_enc_D  = nn.Linear(64*n_frames, n_latent*n_latent*n_frames, bias = True) # just D_t stacked
        self.fc_enc_B  = nn.Linear(64*n_frames, n_latent*n_latent*(n_frames-1), bias = True) # just B_t stacked

        # decoder
        self.fc_dec = nn.Linear(n_latent*n_frames, 64*n_frames, bias = True)                         # 1, B*64 (after .view(): B, 64, 1, 1)

        self.convT5 = nn.ConvTranspose2d(64, 64, 3, 2, 1, 1)                       # B, 64, 2, 2
        self.convT4 = nn.ConvTranspose2d(64, 64, 3, 2, 1, 1)                       # B, 64, 4, 4
        self.convT3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)                       # B, 32, 8, 8
        self.convT2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)                       # B, 32, 16, 16
        self.convT1 = nn.ConvTranspose2d(32, img_channels, 3, 2, 1, 1)             # B, img_channels, 32, 32

        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            kaiming_init(m)

    def reparametrize(self, mu, D, B, alpha=0.5):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        precision_mats = build_tridiag(D,B,alpha).to(self.device)
        covariance_mats = torch.zeros_like(precision_mats)
        z = torch.zeros_like(mu)
        eps = torch.randn_like(mu)
        for i in range(precision_mats.size(0)):
            U = torch.cholesky(precision_mats[i,:,:])
            covariance_mats[i,:,:] = torch.inverse(U)
            z[i,:] = mu[i,:] + covariance_mats[i,:,:].matmul(eps[i,:])
        return z, covariance_mats, precision_mats

    def encode(self, x):
        x = torch.relu(self.conv1(x.view(self.n_frames*x.size(0),1,32,32)))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        mu = self.fc_enc_mu(x.view(-1, 64*self.n_frames))
        D = self.fc_enc_D(x.view(-1, 64*self.n_frames))
        B = self.fc_enc_B(x.view(-1 ,64*self.n_frames))
        return mu, D, B

    def decode(self, z):
        x = self.fc_dec(z).view(-1,64,1,1)
        x = torch.nn.functional.elu(self.convT5(x))
        x = torch.nn.functional.elu(self.convT4(x))
        x = torch.nn.functional.elu(self.convT3(x))
        x = torch.nn.functional.elu(self.convT2(x))
        x = torch.sigmoid(self.convT1(x))
        return x
    
    def forward(self, x, alpha=0.5):
        mu, D, B = self.encode(x)
        z, covariance_mat, precision_mat = self.reparametrize(mu, D, B, alpha)
        return self.decode(z), mu, covariance_mat, precision_mat
