# DGM project journal

## 2019/01/28: start

I cloned the repo.


## 2019/03/12: decided to merge projects

## 2019/03/17: meeting with John

## 2019/04/01: meeting with John and Niko

Aim: learning disentangled and predictable latent representations in a VAE framework by constraining dynamics of latent representations

Task: object(s) moving in front of a background

Success measures:
- dissociating background from object (e.g., in-painting)
- measures of disentanglement

#### Roadmap:
- Step 0:
  - Training Encoder and Decoder separately, supervised
  - Simple input: 2D, white ball with linear dynamics. Background: GRF or Fourier-basis generated.
  - decoder: Can occlusion be decoded (correctly painted)?
  - encoder: Latent code for position as one-hot or as Cartesian coordinates?

- Step 1:
  - VAE with constrained linear dynamics
  - moving Sprites


#### future directions:
- billiard dynamics
- more complex dynamics: Lorenz attractor
- gravity
- more dynamic dimensions (e.g., color or  size of sprites change)
- trade-off between reconstruction and (latent) future prediction
- multiple objects
- learning shading
- moving MNIST digit in front of MNIST digit background
- partial/full object occlusions

### related literature:
- interaction networks

## 2019/04/07: Step 0 - intermediate results

Domain: circle object (with x,y center coordinate) and Gaussian blob background object (with x,y center coordinate, fixed s.d.). x,y \in [0,1]x[0,1]. No dynamics/time yet, simply single images.

Encoder/Decoder are inspired by Higgins (b-VAE). Four conv / transposed conv layers plus a fully connected layer (attaching to the 4D latent code). 

Separate supervised training for encoder and decoder.

Intermediate results:

- Encoder learns the 4D latent vector.
- Decoder learns to reconstruct from 4D latent vector (no need for one-hot position encoding!).

Some observations:
- Setting the x,y of the Gaussian blob beyond training range (outside [0,1]) yields reasonable results. The same for the circle blob is not true - however, circle was always present during training. 
- Kaiming initialization (normal for Conv and Linear) plus Relu might be unfavorable for the decoder (yielding zero output). I therefore used Elu units that don't go dead.

## 2019/04/17: meeting with John

Next steps should involve understanding the static case (encoder/decoder/vae/bvae) and models' capabilities and limitations.

Questions:

 Analyses:
 
 Do reconstruction error differ between edges / inside / outside of the circle? 
  - Analysis for supervised decoder and VAE
  - Make output layer consistent for decoder and VAE for that case
  
 Do circle-reconstructions differ as a function of distance to the gaussian center?
  - quantify both for decoder/VAE
  
 Entangling/disentangling in the VAE
  - correlation between latents?
  - sweep through generative factors and plot/quantify effect in the latents of the VAE
  
 Experiments:
 
 Training data related:
  - can decoder interpolate between untrained positions?
    - train on random sample of (x,y) positions -> validate on unseen positions
  - what is the effect biased class (i.e., position) frequencies?
    - biased distribution of training samples per label/class
 
 Exploring the effect of beta in the b-VAE
  
 Robustness to other datasets:
  - non-linear mapping for positions
  - "elongated" ball (make it 5D latent)
  - different shapes
  - potentially different scales
 
