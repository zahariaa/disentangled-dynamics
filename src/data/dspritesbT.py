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
# for parallel loop to cache data
from joblib import Parallel, delayed
import multiprocessing


def generateLatentLinearMotion(N, n_timesteps, min_travelled_distance, min_coordinate=0., max_coordinate=1.):
    """ generateLatentLinearMotion(N, n_timesteps, min_travelled_distance, min_coordinate=0., max_coordinate=1.):
        
        general purpose function that samples an equidistantly spaced linear trajectory 
        in the space of [min_coordinate, max_coordinate] x [min_coordinate, max_coordinate] (e.g., [0,1] x [0,1])
        
        Args:
            N: number of trajectories
            n_timesteps: number of samples on the line
            min_travelled_distance: how far should the object travel at least
            min_coordinate: lower bound on both x and y coordinates
            max_coordinate: upper bound on both x and y coordinates
        
    """
    d_coordinate = max_coordinate - min_coordinate
    
    def dist(from_to_points):
        dists = np.sqrt(np.sum((from_to_points[:,0,:] - from_to_points[:,1,:])**2, axis=1))
        return dists
    
    def tooCloseToEachOther(dists):
        idx = np.where(dists <= min_travelled_distance)[0]
        n = len(idx)
        return n, idx
    
    # step 1: sample start and end point with minimum distance between the two
    start_and_end_points = d_coordinate * (np.random.rand(N, 2, 2) - min_coordinate)
    
    n_to_close, idx = tooCloseToEachOther(dist(start_and_end_points))
    while n_to_close>0:
        start_and_end_points[idx,:,:] = d_coordinate * (np.random.rand(n_to_close, 2, 2) - min_coordinate)
        n_to_close, idx = tooCloseToEachOther(dist(start_and_end_points))        
        
    # step 2: interpolate intermediate samples (chop line into n_timesteps points)
    motion_latents = np.moveaxis(np.linspace(start_and_end_points[:,0,:],start_and_end_points[:,1,:], num=n_timesteps),0,-1).astype('float32')
        
    return motion_latents

    
class dSpriteBackgroundDatasetTime(Dataset):
    """ dSprite with (gaussian) background dataset and time dimension (moving foreground object)
    __getitem__ returns a 4D Tensor [m_timepoints, n_channels, image_size_x, image_size_y] 
    """
    
    def __init__(self, idx=None, shapetype='dsprite', transform=None, data_dir='../data/dsprites-dataset/',pixels=64, n_timesteps = 10, min_travelled_distance = 0.1):
        """
        Args:
            shapetype (string): circle or dsprite
            transform (callable, optional): Optional transform to be applied
                on a sample.
            data_dir  (string): path to download(ed) dsprites dataset
            pixels: x,y number of pixels of sample (before transform)
            n_timesteps: number of timesteps for the movie
            min_travelled_distance: how far should the object travel at least (image assumed to have size 1 x 1. Therefore, the maximum distance travelled can be sqrt(2))
        """
        
        self.shapetype = shapetype
        self.pixels = pixels
        
        self.n_timesteps = n_timesteps
        self.n_channels = 1
        
        self.min_travelled_distance = min_travelled_distance
        
        # Cache the data, or load the cache if already exists
        filename = '%s_%spix_%ststeps_%smindist.pt' % (shapetype,pixels,n_timesteps,min_travelled_distance)
        self.root = os.path.join('../data/', filename)
                    
        # Load dataset
        if shapetype == 'dsprite':
            raise Exception('moving dSprite is not implemented')
            #data = loadDspriteFile(data_dir)
            #
            #self.imgs = data['imgs']*255
            #self.latents_values = data['latents_values'].astype('float32')
            #self.latents_classes = data['latents_classes']
            #metadata = data['metadata'][()]
            #self.latents_sizes = metadata['latents_sizes']
            #self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
            #                        np.array([1,])))
        elif shapetype == 'circle':
            """
             latents_values are of shape (N x 4 x n_timesteps), where N = self.__len__()
             latents_value[0,:,0] -> [x Background, y Background, x Circle, y Circle]          
            """            
            self.latents_bases = [200000]
            # grid points from which background gaussian x,y coordinates are drawn
            self.background_1d_sampling_points = np.linspace(0,1,32,dtype='float32')
            self.latents_values = self.generateLatentSequence()


        if idx is not None:
            self.latents_values = self.latents_values[idx,:]
            self.latents_bases[0] = np.shape(self.latents_values)[0]

        if transform is None:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transform,
                                                 transforms.ToTensor()])
        # Generate all data
        if os.path.exists(self.root):
            self.img = torch.load(self.root)
            return
        else:
            #self.img = np.ndarray(shape=(self.latents_bases[0],n_timesteps,self.n_channels,32,32), dtype='uint8')

            num_cores = multiprocessing.cpu_count()
     
            self.img = Parallel(n_jobs=num_cores)(delayed(self.__getitem__)(idx) for idx in range(self.latents_bases[0]))
            
#             for idx in range(self.latents_bases[0]):
# #                 print(self.img.shape)
# #                 print(self.__getitem__(idx))
#                 self.img[idx,:,:,:,:],_ = self.__getitem__(idx)
#                 if idx % 1000 == 0:
#                     print(idx)
            
            # Cache dataset
            torch.save(self.img, self.root)
    
    def __len__(self):
        return self.latents_bases[0]
    
    
    def __getitem__(self, idx, mu=None):
        if os.path.exists(self.root):
            return self.img[idx],self.latents_values[idx,:,:]        
        else:
            # Set up foreground object
            if self.shapetype == 'circle':
                center = (0.75*self.pixels)*self.latents_values[idx,-2:,:] + np.array([1/8,1/8])[:,np.newaxis]*self.pixels
                # create foreground movie
                # loop over time to create succesive circle images
                foreground = 255*self.circle2D(center[:,0])[:,:,:,np.newaxis]
                for t in range(1,self.n_timesteps):
                    ###TODO: translate latent scale to radius
                    foreground = np.append(foreground, 255*self.circle2D(center[:,t])[:,:,:,np.newaxis], axis=3)        

            elif self.shapetype == 'dsprite':
                foreground = self.pick_dSprite(idx)

            # Set up background
            mu = self.latents_values[idx,:2,:]

            # create background movie
            bg = self.gaussian2D(mu[:,0])
            bg = (255*bg).reshape(bg.shape+(1,)+(1,))
            if self.background_static: # if background is static, we can save some time
                background = np.repeat(bg,self.n_timesteps,axis=3)
            else: # otherwise loop over background latents
                background = bg
                for t in range(1, self.n_timesteps):
                    bg = self.gaussian2D(mu[:,t])
                    bg = (255*bg).reshape(bg.shape+(1,)+(1,))                
                    background = np.append(background, bg, axis=3)

            # Combine foreground and background
            sample = np.clip(foreground+0.8*background,0,255).astype('uint8')

            # Output
            #latent = self.latents_values[idx,:,:]

            # transform indididual images sequentially
            transf_sample = self.transform(sample[:,:,:,0]).unsqueeze(0)
            if self.transform:
                for t in range(1, self.n_timesteps):
                    transf_sample = torch.cat((transf_sample, self.transform(sample[:,:,:,t]).unsqueeze(0)), dim=0)

            # transf_sample: [n_timesteps, n_channels, n_resizedpixel, n_resizedpixel]

            return transf_sample#,latent
    
    
    def generateLatentSequence(self):
        N = self.__len__()
        np.random.seed(0)
        
        # step 1: create forground latens as linear motion
        foreground_latents = generateLatentLinearMotion(N = N, 
                                                        n_timesteps = self.n_timesteps,
                                                        min_travelled_distance = self.min_travelled_distance, 
                                                        min_coordinate = 0.0, 
                                                        max_coordinate = 1.0)      

        # step 2: static background
        background_latents = np.zeros_like(foreground_latents, dtype='float32')
        # background x-coordinate (constant across frames)
        background_latents[:,0,:] = np.random.choice(self.background_1d_sampling_points, N, replace=True)[:,np.newaxis]
        # background y-coordinate (constant across frames)
        background_latents[:,1,:] = np.random.choice(self.background_1d_sampling_points, N, replace=True)[:,np.newaxis]
        self.background_static = True
        
        latents = np.concatenate((background_latents,foreground_latents),axis=1)
                
        return latents

    def arbitraryCircle(self,objx=None,objy=None,backx=None,backy=None,radius=None):
       # Pick an arbitrary sample by directly addressing it
#        if (objx is not None) and (objy is not None):
#            ###TODO: make sure pad scales with circle scale, so as to never hit the border
#            center = (0.75*self.pixels)*np.array([objx,objy]) + np.array([0.125,0.125])*self.pixels
#            ###TODO: translate latent scale to radius
#            foreground = 255*self.circle2D(center,radius)
#            foreground = foreground.reshape((1,)+foreground.shape)
#        else:
#            foreground = np.zeros((self.pixels,self.pixels,1))
#            foreground = foreground.reshape((1,)+foreground.shape)
#        if (backx is not None) and (backy is not None):
#            background = self.gaussian2D(np.array([backx,backy]))
#            background = (255*background).reshape((1,)+background.shape+(1,))
#        else:
#            background = np.zeros((self.pixels,self.pixels,1))
#            background = (255*background).reshape((1,)+background.shape)
#            
#        # Combine foreground and background
#        ims = np.clip(foreground+0.8*background,0,255).astype('uint8')
#        
#        return self.transformArray(ims)
       pass
        
            
    def findDsprite(self,shape=None,scale=None,orientation=None,posX=None,posY=None,back=None):
#        # Outputs an image or set of images based on the search parameters given by the inputs
#        # Second output is just a list of bools where the matching imgs are true
#        
#        # initialize query to include all dsprites
#        query = np.full((self.latents_bases[0],), True)
#        # narrow search
#        if shape is not None:
#            query = query & (self.latents_classes[:,1]==shape)
#        if scale is not None:
#            query = query & (self.latents_classes[:,2]==scale)
#        if orientation is not None:
#            query = query & (self.latents_classes[:,3]==orientation)
#        if posX is not None:
#            query = query & (self.latents_classes[:,4]==posX)
#        if posY is not None:
#            query = query & (self.latents_classes[:,5]==posY)
#        # Convert list of bools to list of indices
#        idx = np.where(query)[0]
#        # Pick the images
#        ims = self.pick_dSprite(idx)
#        
#        if back is not None:
#            # Add background
#            if type(back) is not np.ndarray:
#                background = 2*self.gaussian2D(mu=2*np.random.randint(self.pixels/2,size=2))/self.pixels
#            else:
#                background = self.gaussian2D()
#            background = 255*background.reshape((1,)+ims.shape[1:])
#
#            # Combine foreground and background
#            ims = np.clip(ims+0.8*np.tile(background,(ims.shape[0],1,1,1)),0,255).astype('uint8')
#
#        return self.transformArray(ims),idx
        pass
    
    
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
    
    def getCircleSegmentationMasks(self, objx, objy, dr_in = .05, dr_margin = .05, radius = .1, thresh = .4):
        """ 
            getCircleSegmentationMasks(self, objx, objy, dr_in = .05, dr_margin = .05, radius = .1, thresh = .4):
    
            returns a segmentation of the image:
                objectMask: mask corresponding to the circle object
                objectEdgeMask: mask corresponding to the edge of the circle object
                insideObjectMask: mask corresponding to the inner part of the circle (in the object but not part of the objectEdgeMask)
                backgroundMask: mask of the rest (not in the objectMask or the objectEdgeMask)
    
            inputs:
                - dr_in: controls by how much smaller (absolute) the inner circle is than the actual circle object
                - dr_margin: controls the radius of the area which is _not_ considered to be background (allows for a safe margin)
                - radius: is the radius of the circle object. Currently this is set to 0.1 (as it is handcoded in the self.circle2D function)
                - thresh: threshold above which a pixel is considered "on" (part of the object)
    
                Displaying example:
                    ds = dSpriteBackgroundDataset(transform=transforms.Resize((32,32)),shapetype = 'circle')
                    x = ds.arbitraryCircle(objx=.6, objy=.2, backx = .3, backy = .1)
                    backgroundMask, objectMask, insideObjectMask, objectEdgeMask = ds.getCircleSegmentationMasks(objx,objy)
    
                    _,ax = plt.subplots(1,5, figsize = (12, 3))
                    ax[0].imshow(x.squeeze())
                    ax[1].imshow(backgroundMask.squeeze())
                    ax[2].imshow(objectMask.squeeze())
                    ax[3].imshow(insideObjectMask.squeeze())
                    ax[4].imshow(objectEdgeMask.squeeze())
    
            (for circles only now -> possibly in the future this would work for arbirary shapes when replacing radius with scale parameter)
        """
        
        objectMask = self.arbitraryCircle(objx, objy, backx = None, backy = None, radius = radius) > thresh
        smallerMask = self.arbitraryCircle(objx, objy, backx = None, backy = None, radius = radius - dr_in) > thresh        
        biggerMask = self.arbitraryCircle(objx, objy, backx = None, backy = None, radius = radius + dr_margin) > thresh
    
        objectEdgeMask = np.logical_xor(objectMask, smallerMask)
    
        insideObjectMask = torch.zeros_like(objectMask)
        insideObjectMask[objectMask] = 1
        insideObjectMask[objectEdgeMask] = 0
        
        backgroundMask = ~biggerMask
    
        return backgroundMask, objectMask, insideObjectMask, objectEdgeMask  

def loadDspriteFile(data_dir='../data/dsprites-dataset'):
    root = os.path.join(data_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    if not os.path.exists(root):
        import subprocess
        print('Now downloading dsprites-dataset')
        subprocess.call(['../data/./download_dsprites.sh'])
        print('Finished')

    return np.load(root,encoding='latin1',mmap_mode='r')
    
## Helper function to initialize partition of dsprite dataset into training and validation
#def partition_init(training_proportion=0.8,dimensionwise_partition=True,shapetype='dsprite', data_dir='../data/dsprites-dataset/'):
#    
#    if not dimensionwise_partition:
#        if shapetype == 'dsprite':
#            data = loadDspriteFile(data_dir)
#            metadata = data['metadata'][()]
#            totaldata = np.prod(metadata['latents_sizes'])
#        else:
#            totaldata = 16*16*32*32 # ASSUMPTION!
#    
#        # Initialize random partitions of data
#        ridx = np.random.permutation(totaldata)
#        t = int(training_proportion*totaldata)
#        
#        partition = {'train': ridx[:t], 'validation': ridx[t:]}
#    
#    else: # apply proportion to unique values in each dimension (-> proportion of actual samples is therefore much lower)
#        if shapetype == 'dsprite':
#            raise Exception('dimensionwise_partition for shapetype dSprite is not implemented')
#        elif shapetype == 'circle':
#            ds = dSpriteBackgroundDataset(shapetype = 'circle')
#            
#            latents_values = ds.latents_values
#            
#            datadimvals = list()
#            dimwise_train_idx = list()
#            dimwise_val_idx = list()
#            dimwise_train_vals = list()
#            dimwise_val_vals = list()
#            
#            for ii in range(latents_values.shape[1]):
#                datadimvals.append(np.unique(latents_values[:,ii]))
#                n_train = np.round(training_proportion * len(datadimvals[ii]))
#                
#                dimwise_train_idx.append(np.sort(np.random.choice(len(datadimvals[ii]), int(n_train), replace = False)))
#                dimwise_train_vals.append(datadimvals[ii][dimwise_train_idx[ii]])
#                
#                dimwise_val_idx.append(np.where(np.isin(np.arange(len(datadimvals[ii])), dimwise_train_idx[ii]) == False)[0])
#                dimwise_val_vals.append(datadimvals[ii][dimwise_val_idx[ii]])
#            
#        else:
#            raise Exception('dimensionwise_partition for shapetype {} is not implemented'.format(shapetype))
#            
#        
#        n_dim = len(datadimvals)
#        traintrials = np.ones(latents_values.shape[0], dtype = bool)
#        for ii in range(n_dim):
#            traintrials = traintrials & np.isin(latents_values[:,ii], dimwise_train_vals[ii])
#        valtrials = traintrials == False
#        
#        n_train_samples = np.sum(traintrials)
#        
#        idx = np.arange(latents_values.shape[0], dtype = int)        
#        partition = {'train': idx[traintrials], 'validation': idx[valtrials], 'datadimvals': datadimvals, 'dimwise_train_idx': dimwise_train_idx, 'dimwise_val_idx': dimwise_val_idx, 'n_train_samples': n_train_samples}
#        
#    return partition
        
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

def show_movies_grid(samples):
    num_images=samples.size(0)
    ncols = samples.size(1)
    nrows = num_images
    _, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3,nrows * 3))
    
    if len(axes.shape)==1:
        axes = axes[None,:]

    for ii in range(num_images):
        for tt in range(ncols):
          axes[ii,tt].imshow(samples[ii,tt,0,:,:], cmap='Greys_r',  interpolation='nearest')
          axes[ii,tt].set_xticks([])
          axes[ii,tt].set_yticks([])
          axes[ii,tt].set_title('t={}'.format(tt))

### DEMO/TESTING

def demo(shapetype='circle',n_timesteps=10, data_dir='../data/dsprites-dataset/'):
      
   circles = dSpriteBackgroundDatasetTime(shapetype=shapetype,n_timesteps=10,data_dir=data_dir)
   
   idx = 3001
   print('One sample (#{}), addressed'.format(idx))
   sample,latent = circles[idx]
   show_movies_grid(sample.unsqueeze(0))
   #print('Latents: {}'.format(latent))
   
   transformed_dataset = dSpriteBackgroundDatasetTime(transform=transforms.Resize((32,32)),
                                                      n_timesteps=10,
                                                      shapetype=shapetype,data_dir=data_dir)
   
   print('One rescaled sample (#{}), addressed'.format(idx))
   sample_trans,latent_trans = transformed_dataset[idx]
   show_movies_grid(sample_trans.unsqueeze(0))
#   print('Latents: {}'.format(latent_trans)) 
#   
   # Use pytorch dataloader and plot a random batch
   dataloader = DataLoader(transformed_dataset, batch_size=25,
                           shuffle=True, num_workers=1)
   print('One minibatch, shuffled and scaled down')
   for i,[samples,latents] in enumerate(dataloader):
       print('Minibatch {}, batch size: {}'.format(i,samples.size())) 
       show_movies_grid(samples)
       break


if __name__ == "__main__":
    demo()

