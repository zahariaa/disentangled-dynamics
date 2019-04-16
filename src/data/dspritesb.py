"""dspritesb.py

Stuff to know:
- a basic class to load the dSprites and paint a randomly centered 2D gaussian blob behind each sprite.
- uses pytorch DataLoader.
- dspritesb.demo() runs basic unit tests

"""

from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms, utils

"""
Possibly to-do

### ouput image dimensions:    
 - CURRENTLY: Channel axis is added in the training loop
 
### output images and labels as float
 - CURRENTLY: done in training loop

### latents should have the same scaling
    - otherwise they contribute differently to the loss
    - CURRENTLY: scaling is done in the training.py
    
### seperate training and validation sets
    - CURRENTLY: no separation (overfitting / fitting to traing samples is likely)
    - e.g.: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    
### Minor (not necessary, but I think this would be more conventional)
    - So far I have seen the output of the train_loader / dataset to be of this form:
        
        image, label = train_iter.next()
        or, resp.:
        
        for image_batch, label_batch in train_loader:
            do training...
            
        
        Probably this would mean that, __getitem__() of dSpriteBackgroundDataset would "return image, label" instead of returning a dictionary (?)
        
    
"""

class dSpriteBackgroundDataset(Dataset):
    """ dSprite with (gaussian) background dataset.
    __getitem__ returns a 3D Tensor [n_channels, image_size_x, image_size_y] 
    """
    
    def __init__(self, shapetype='dsprite', transform=None):
        """
        Args:
            shapetype (string): circle or dsprite
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Load dataset
        dset_dir = 'data/dsprites-dataset/'
        root = os.path.join(dset_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if not os.path.exists(root):
            import subprocess
            print('Now downloading dsprites-dataset')
            subprocess.call(['../data/./download_dsprites.sh'])
            print('Finished')

        data = np.load(root,encoding='latin1')

#         data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
#         train_kwargs = {'data_tensor':data}
        
        self.shapetype = shapetype
        self.imgs = data['imgs']*255
        self.latents_values = data['latents_values']
        self.latents_classes = data['latents_classes']
        self.metadata = data['metadata'][()]
        self.latents_sizes = self.metadata['latents_sizes']
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))
        
        if transform is None:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transform,
                                                 transforms.ToTensor()])
    def __len__(self):
        return self.latents_bases[0]
    
    def __getitem__(self, idx, mu=None):
        # Set up foreground object
        if self.shapetype == 'circle':
            center = 48*self.latents_values[idx,-2:] + np.array([8,8])
            ###TODO: translate latent scale to radius
            foreground = 255*self.circle2D(center)
        elif self.shapetype == 'dsprite':
            foreground = self.pick_dSprite(idx)
        
        # Set up background
        if mu is None:
            mu = 2*np.random.randint(32,size=2)
        background = self.gaussian2D(mu)
        background = (255*background).reshape(background.shape+(1,))

        # Combine foreground and background
        sample = np.clip(foreground+0.8*background,0,255).astype('uint8')

        # Output
        latent = np.concatenate((self.latents_values[idx,-2:],mu.astype('float64')))

        if self.transform:
            sample = self.transform(sample)
            
        return sample,latent
    
    # Generate 2D gaussian backgrounds
    def gaussian2D(self,mu=np.array([31,31]),Sigma=np.array([[1000, 0], [0, 1000]]),pos=None):
        if pos is None:
            gridx, gridy = np.meshgrid(np.arange(0,64),np.arange(0,64))
            pos = np.empty(gridx.shape + (2,))
            pos[:,:,0] = gridx
            pos[:,:,1] = gridy

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
    def circle2D(self,center,radius=6,pos=None):
        if pos is None:
            gridx, gridy = np.meshgrid(np.arange(0,64),np.arange(0,64))
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

def demo(shapetype='dsprite'):
   dSpritesB = dSpriteBackgroundDataset(shapetype=shapetype)
   
   idx = 300001
   print('One sample (#{}), addressed'.format(idx))
   sample,latent = dSpritesB[idx]
   h = plt.imshow(transforms.ToPILImage()(sample),cmap=plt.cm.gray)
   plt.show()
   print('Latents: {}'.format(latent))
   
   transformed_dataset = dSpriteBackgroundDataset(transform=transforms.Resize((32,32)),shapetype=shapetype)
   
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

