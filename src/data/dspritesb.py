"""dspritesb.py

Stuff to know:
- a basic class to load the dSprites and paint a randomly centered 2D gaussian blob behind each sprite.
- uses pytorch DataLoader.
- dspritesb.demo() runs basic unit tests

"""

from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms, utils

"""
Possibly to-do
    
### seperate training and validation sets
    - CURRENTLY: no separation (overfitting / fitting to traing samples is likely)
    - e.g.: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

class dSpriteBackgroundDataset(Dataset):
    """ dSprite with (gaussian) background dataset.
    __getitem__ returns a 3D Tensor [n_channels, image_size_x, image_size_y] 
    """
    
    def __init__(self, shapetype='dsprite', transform=None, data_dir='../data/dsprites-dataset/',pixels=64):
        """
        Args:
            shapetype (string): circle or dsprite
            transform (callable, optional): Optional transform to be applied
                on a sample.
            data_dir  (string): path to download(ed) dsprites dataset
        """
        # Load dataset
        root = os.path.join(data_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if not os.path.exists(root):
            import subprocess
            print('Now downloading dsprites-dataset')
            subprocess.call(['../data/./download_dsprites.sh'])
            print('Finished')

        data = np.load(root,encoding='latin1')
        
        self.shapetype = shapetype
        self.imgs = data['imgs']*255
        self.latents_values = data['latents_values'].astype('float32')
        self.latents_classes = data['latents_classes']
        self.metadata = data['metadata'][()]
        self.latents_sizes = self.metadata['latents_sizes']
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))
        self.pixels = pixels
        
        if transform is None:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transform,
                                                 transforms.ToTensor()])
    
    def __len__(self):
        if self.shapetype == 'circle':
            # THIS ASSUMES A SINGLE SHAPE AND A SINGLE SCALE
            return self.latents_bases[2]
        else:
            return self.latents_bases[0]
    
    
    def __getitem__(self, idx, mu=None):
        # Set up foreground object
        if self.shapetype == 'circle':
            center = (0.75*self.pixels)*self.latents_values[idx,-2:] + np.array([1/8,1/8])*self.pixels
            ###TODO: translate latent scale to radius
            foreground = 255*self.circle2D(center)
        elif self.shapetype == 'dsprite':
            foreground = self.pick_dSprite(idx)
        
        # Set up background
        if mu is None:
            mu = 2*np.random.randint(self.pixels/2,size=2)/self.pixels
        background = self.gaussian2D(mu)
        background = (255*background).reshape(background.shape+(1,))

        # Combine foreground and background
        sample = np.clip(foreground+0.8*background,0,255).astype('uint8')

        # Output
        latent = np.concatenate((self.latents_values[idx,-2:],mu.astype('float32')))

        if self.transform:
            sample = self.transform(sample)
            
        return sample,latent
    

    def arbitraryCircle(self,objx=None,objy=None,backx=None,backy=None,radius=None):
       # Pick an arbitrary sample by directly addressing it
        if (objx is not None) and (objy is not None):
            ###TODO: make sure pad scales with circle scale, so as to never hit the border
            center = (0.75*self.pixels)*np.array([objx,objy]) + np.array([0.125,0.125])*self.pixels
            ###TODO: translate latent scale to radius
            foreground = 255*self.circle2D(center,radius)
            foreground = foreground.reshape((1,)+foreground.shape)
        else:
            foreground = np.zeros((self.pixels,self.pixels,1))
        if (backx is not None) and (backy is not None):
            background = self.gaussian2D(np.array([backx,backy]))
            background = (255*background).reshape(background.shape+(1,))
        else:
            background = np.zeros((self.pixels,self.pixels,foreground.shape[2]))
            
        # Combine foreground and background
        ims = np.clip(foreground+0.8*background,0,255).astype('uint8')
        
        return self.transformArray(ims)
        
            
    def findDsprite(self,shape=None,scale=None,orientation=None,posX=None,posY=None,back=None):
        # Outputs an image or set of images based on the search parameters given by the inputs
        # Second output is just a list of bools where the matching imgs are true
        
        # initialize query to include all dsprites
        query = np.full((self.latents_bases[0],), True)
        # narrow search
        if shape is not None:
            query = query & (self.latents_classes[:,1]==shape)
        if scale is not None:
            query = query & (self.latents_classes[:,2]==scale)
        if orientation is not None:
            query = query & (self.latents_classes[:,3]==orientation)
        if posX is not None:
            query = query & (self.latents_classes[:,4]==posX)
        if posY is not None:
            query = query & (self.latents_classes[:,5]==posY)
        # Convert list of bools to list of indices
        idx = np.where(query)[0]
        # Pick the images
        ims = self.pick_dSprite(idx)
        
        if back is not None:
            # Add background
            if type(back) is not np.ndarray:
                background = 2*self.gaussian2D(mu=2*np.random.randint(self.pixels/2,size=2))/self.pixels
            else:
                background = self.gaussian2D()
            background = 255*background.reshape((1,)+ims.shape[1:])

            # Combine foreground and background
            ims = np.clip(ims+0.8*np.tile(background,(ims.shape[0],1,1,1)),0,255).astype('uint8')

        return self.transformArray(ims),idx
    
    
    def transformArray(self,ims):
        # Takes numopy array of image(s) and outputs the image(s), each transformed with self.transform
        return torch.unsqueeze(torch.cat([self.transform(ims[i,:,:,:]) for i in np.arange(ims.shape[0])]),1)
    
    
    # Generate 2D gaussian backgrounds
    def gaussian2D(self,mu=np.array([0.5,0.5]),Sigma=np.array([[15, 0], [0, 15]]),pos=None):
        if pos is None:
            gridx, gridy = np.meshgrid(np.arange(0,self.pixels),np.arange(0,self.pixels))
            pos = np.empty(gridx.shape + (2,))
            pos[:,:,0] = gridx
            pos[:,:,1] = gridy
        mu = mu*self.pixels
        Sigma = Sigma*self.pixels

        # from https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
        #n = mu.shape[0]
        #Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        #N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
        #return np.exp(-fac / 2) / N
        fac = np.exp(-fac/2)

        # Normalized to peak at 1
        return fac/np.max(fac)

    
    # Generate a circle in a random position
    def circle2D(self,center,radius=None,pos=None):
        if radius is None:
            radius = 0.1
        radius = radius*self.pixels
        if pos is None:
            gridx, gridy = np.meshgrid(np.arange(0,self.pixels),np.arange(0,self.pixels))
        z = np.square(gridx-center[0]) + np.square(gridy-center[1]) - radius
        # Threshold by radius
        z = (z<=np.square(radius)).astype('uint8')
        # Output 3D [h,w,channel] tensor
        return z.reshape(z.shape+(1,))

    # Generate dSprite with 2D gaussian background
    def pick_dSprite(self,idx=None):
        if idx is None:
            np.random.randint(self.latents_bases[0])            
        im = self.imgs[idx,:,:]
        im = im.reshape(im.shape+(1,)) # add channel to end (assumes numpy ndarray)

        return im
    
# Helper function to show images
def show_images_grid(samples):
    num_images=samples.size(0)
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
          ax.imshow(samples[ax_i,0,:,:], cmap='Greys_r',  interpolation='nearest')
          ax.set_xticks([])
          ax.set_yticks([])
        else:
          ax.axis('off')


### DEMO/TESTING

def demo(shapetype='dsprite',data_dir='../data/dsprites-dataset/'):
   dSpritesB = dSpriteBackgroundDataset(shapetype=shapetype,data_dir=data_dir)
   
   idx = 300001
   print('One sample (#{}), addressed'.format(idx))
   sample,latent = dSpritesB[idx]
   h = plt.imshow(transforms.ToPILImage()(sample),cmap=plt.cm.gray)
   plt.show()
   print('Latents: {}'.format(latent))
   
   transformed_dataset = dSpriteBackgroundDataset(transform=transforms.Resize((32,32)),
                                                  shapetype=shapetype,data_dir=data_dir)
   
   print('One rescaled sample (#{}), addressed'.format(idx))
   sample_trans,latent_trans = transformed_dataset[idx]
   h = plt.imshow(transforms.ToPILImage()(sample_trans),cmap=plt.cm.gray)
   plt.show()
   print('Latents: {}'.format(latent_trans)) 
   
   # Use pytorch dataloader and plot a random batch
   dataloader = DataLoader(transformed_dataset, batch_size=25,
                           shuffle=True, num_workers=1)
   print('One minibatch, shuffled and scaled down')
   for i,[samples,latents] in enumerate(dataloader):
       print('Minibatch {}, batch size: {}'.format(i,samples.size())) 
       show_images_grid(samples)
       break


if __name__ == "__main__":
    demo()

