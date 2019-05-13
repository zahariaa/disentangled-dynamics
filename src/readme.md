this directory for code implementation, testing, etc

### Todo / next-steps (04/17/19)

###### 0)

- <del>equate architecture of decoder and VAE (sigmoid in decoder)</del>

###### 1) Building analyses pipelines/scripts (apply to decoder and VAE)

- Do reconstruction error differ between edges / inside / outside of the circle?
- Do circle-reconstructions differ as a function of distance to the gaussian center?
- Entangling/disentangling in the VAE (correlations in latent space)
- Sweep through generative factors and plot/quantify effect in the latents (of the VAE)
- consider posterior variance as well, i.e., \sigma(x)

###### 2) Experiments

- train on random sample of (x,y) positions -> validate on unseen positions
- different shapes
- dataset where circle has 0.5 luminance value
- "elongated" ball (make it 5D latent)
- potentially different scales
(lower priority):
- non-linear mapping for positions
- biased distribution of training samples per label/class (more circles at the edge, fewer at center)

###### 3) new architectures

- bVAE
- potentially: how many channels/layers are necessary? (for the decoder)
- PCA?

### Todo / next-steps (04/08/19)

###### 1) Stimulus generation


- implementational details (in dsprites.py)
- I trained the decoder using the solver.py in supervised_encoderdecoder -> seems like the coordinates of the gaussian background blob have changed (added by BP 04/23/19)
- more complex backgrounds
- more object shapes / size / orientation
- <del>skimage produces warning: "Anti-aliasing will be enabled by default in skimage 0.15 to avoid aliasing artifacts when down-sampling images.
  warn("Anti-aliasing will be enabled by default in skimage 0.15 to ". </del> Currently, I suppress warnings in main.py because of that.
- making stimuli dynamic (3d output)


###### 2) train/solver coding

- <del>train.py in each modeling folder</del>
- <del>convert train.py into solver.py</del>
- <del>implement checkpointing</del>
- <del>create main.py (for command line training)</del>
- add vizdom board to solver
- consider learning scheduler (for lr changes) / at least option to change lr from some checkpoint on

###### 3) visualization

- <del> visualize static vae </del>

###### 4) modeling steps:

- <del>auto-encoding circle/gaussian</del>
- dynamic auto-encoder
