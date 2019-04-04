# DGM project journal

## 2019/01/28: start

I cloned the repo.


## 2019/03/12: decided to merge projects

## 2019/03/17: meeting with John

## 2019/04/01: meeting with John and Niko

Aim: learning disentangled and predictable latent represenations in a VAE framework by constraining dynamics of latent represenations

Task: object(s) moving in front of a background

Success measures:
- dissociating background from object (e.g., inpainting)
- measures of disentanglement

#### Roadmap:
- Step 0:
  - Training Encdoer and Decoder separatly, supervised
  - Simple input: 2D, white ball with linear dynamics. Background: GRF or fourier-basis generated.
  - decoder: Can occlusion be decoded (correctly painted)?
  - encoder: Latent code for position as one-hot or as cartesian coordinates?

- Step 1:
  - VAE with constrained linear dynamics
  - moving Sprites


#### future directions:
- billiard dynamics
- more complex dynamics: Lorenz attractor
- gravity
- more dynamic dimensions (e.g., coloror  size of sprites change)
- trade-off between reconstruction and (latent) future prediction
- multiple objects
- learning shading
- moving MNIST digit in front of MNIST digit background
- partial/full object occlusions

### related literature:
- interaction networks
