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
from skimage import io, transform
from torchvision import transforms, utils

"""
Possibly to-do


### ouput image dimensions:
 - return image 3D [n_channels, image_size_x, image_size_y] 
     (all networks expect [n_batch, n_channels, image_size_x, image_size_y], the trainloader concatenates images along (a new) 0-th dimension)
 - OR: torchvision.transforms.ToTensor looks like the right choice and could be composed with Rescale (torchvision.transforms.ToTensor expects [H,W,C] and uint8 input)

   train_loader = torch.utils.data.DataLoader(dSpriteBackgroundDataset(transform=transforms.Compose(
                                                                    Rescale(32), transforms.ToTensor()),
                                            shapetype = 'circle'), **params)
    
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
    """ dSprite with (gaussian) background dataset."""
    
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
        self.imgs = data['imgs']
        self.latents_values = data['latents_values']
        self.latents_classes = data['latents_classes']
        self.metadata = data['metadata'][()]
        self.latents_sizes = self.metadata['latents_sizes']
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))
        
        self.transform = transform
    def __len__(self):
        return self.latents_bases[0]
    
    def __getitem__(self, idx, mu=None):
        # Set up foreground object
        if self.shapetype == 'circle':
            center = 48*self.latents_values[idx,-2:] + np.array([8,8])
            ###TODO: translate latent scale to radius
            foreground = self.circle2D(center)
        elif self.shapetype == 'dsprite':
            foreground = self.pick_dSprite(idx)
        
        # Set up background
        if mu is None:
            mu = 2*np.random.randint(32,size=2)
        background = self.gaussian2D(mu)

        # Combine foreground and background
        im = np.clip(255*(foreground+0.8*background),0,255).astype('uint8')

        # Output
        sample = {'image': im,
                  'latents': np.concatenate((self.latents_values[idx,-2:],mu.astype('float64')))} 

        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
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
    
        return (z<=np.square(radius)).astype('uint8')

    # Generate dSprite with 2D gaussian background
    def pick_dSprite(self,idx=None):
        if idx is None:
            np.random.randint(self.latents_bases[0])
        im = self.imgs[idx,:,:]

        return im


# Rescale images
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        pix (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, pix):
        assert isinstance(pix, (int, tuple))
        self.pix = pix

    def __call__(self, sample):
        image, latents = sample['image'], sample['latents']

        h,w = image.shape[:2]
        img = transform.resize(image, (self.pix, self.pix))
        
        # h and w are swapped because for images,
        # x and y axes are axis 1 and 0 respectively
        latents[-2:] = latents[-2:] * [self.pix / w, self.pix / h]

        return {'image': img, 'latents': latents}
    
# Helper function to show images
def show_images_grid(samplebatch):
    num_images=samplebatch['image'].size(0)
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
          ax.imshow(samplebatch['image'][ax_i], cmap='Greys_r',  interpolation='nearest')
          ax.set_xticks([])
          ax.set_yticks([])
        else:
          ax.axis('off')


### DEMO/TESTING

def demo(shapetype='dsprite'):
   dSpritesB = dSpriteBackgroundDataset(shapetype=shapetype)
   
   idx = 300001
   print('One sample (#{}), addressed'.format(idx))
   sample = dSpritesB[idx]
   h = plt.imshow(sample['image'],cmap=plt.cm.gray)
   plt.show()
   print('Latents: {}'.format(sample['latents'])) 
   
   transformed_dataset = dSpriteBackgroundDataset(transform=Rescale(32),shapetype=shapetype)
   
   print('One rescaled sample (#{}), addressed'.format(idx))
   sample_trans = transformed_dataset[idx]
   h = plt.imshow(sample_trans['image'],cmap=plt.cm.gray)
   plt.show()
   print('Latents: {}'.format(sample_trans['latents'])) 
   
   # Use pytorch dataloader and plot a random batch
   dataloader = DataLoader(transformed_dataset, batch_size=25,
                           shuffle=True, num_workers=1)
   print('One minibatch, shuffled and scaled down')
   for i,samplebatched in enumerate(dataloader):
       print('Minibatch {}, batch size: {}'.format(i,samplebatched['image'].size())) 
       show_images_grid(samplebatched)
       break

if __name__ == "__main__":
    demo()

