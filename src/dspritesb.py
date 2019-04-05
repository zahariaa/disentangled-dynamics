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

class dSpriteBackgroundDataset(Dataset):
    """ dSprite with (gaussian) background dataset."""
    
    def __init__(self, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Load dataset
        dset_dir = 'data/dsprites-dataset/'
        root = os.path.join(dset_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if not os.path.exists(root):
            import subprocess
            print('Now downloading dsprites-dataset')
            subprocess.call(['./download_dsprites.sh'])
            print('Finished')

        self.data = np.load(root,encoding='latin1')

#         data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
#         train_kwargs = {'data_tensor':data}
        
        self.imgs = self.data['imgs']
        self.latents_values = self.data['latents_values']
        self.latents_classes = self.data['latents_classes']
        self.metadata = self.data['metadata'][()]
        self.latents_sizes = self.metadata['latents_sizes']
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))
        
        self.transform = transform
    
    def __len__(self):
        return self.latents_bases[0]
    
    def __getitem__(self, idx, mu=None):
        if mu is None:
            mu = np.random.randint(64,size=2)
        image, mu = self.dSprite_gaussian_background(idx,mu)
        
        sample = {'image': image, 'latents': self.latents_classes[idx,:], 'background': mu}

        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    # Generate 2D gaussian backgrounds
    def gaussian2D(self,mu,Sigma,pos=None):
        if pos is None:
            gridx, gridy = np.meshgrid(np.arange(0,64),np.arange(0,64))
            pos = np.empty(gridx.shape + (2,))
            pos[:,:,0] = gridx
            pos[:,:,1] = gridy

        # from https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        #N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
        #return np.exp(-fac / 2) / N
        fac = np.exp(-fac/2)

        # Normalized to peak at 1
        return fac/np.max(fac)

    # Generate dSprite with 2D gaussian background
    def dSprite_gaussian_background(self,idx=None,mu=np.array([31,31]),Sigma=np.array([[1000, 0], [0, 1000]])):
        if idx is None:
            np.random.randint(self.latents_bases[0])
        background = self.gaussian2D(mu,Sigma)
        im = self.imgs[idx,:,:]
        im = np.clip(255*(im+0.8*background),0,255).astype('uint8')

        return im, mu


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
        image, latents, background = sample['image'], sample['latents'], sample['background']

        h,w = image.shape[:2]
        img = transform.resize(image, (self.pix, self.pix))
        
        # h and w are swapped because for images,
        # x and y axes are axis 1 and 0 respectively
        background = background * [self.pix / w, self.pix / h]

        return {'image': img, 'latents': latents, 'background': background}
    
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

def demo():
   dSpritesB = dSpriteBackgroundDataset()
   
   print('One sample (#300000), addressed')
   sample = dSpritesB[300000]
   h = plt.imshow(sample['image'],cmap=plt.cm.gray)
   plt.show()
   print('Latents: {}, Background gaussian center: {}'.format(sample['latents'],sample['background'])) 
   
   transformed_dataset = dSpriteBackgroundDataset(transform=Rescale(32))
   
   print('One rescaled sample (#300000), addressed')
   sample_trans = transformed_dataset[300000]
   h = plt.imshow(sample_trans['image'],cmap=plt.cm.gray)
   plt.show()
   print('Latents: {}, Background gaussian center: {}'.format(sample_trans['latents'],sample_trans['background'])) 
   
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

