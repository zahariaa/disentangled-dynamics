this directory for code implementation, testing, etc

### Todo / next-steps (04/08/19)

###### 1) Stimulus generation


- implementational details (in dsprites.py)
- skimage produces warning: "Anti-aliasing will be enabled by default in skimage 0.15 to avoid aliasing artifacts when down-sampling images.
  warn("Anti-aliasing will be enabled by default in skimage 0.15 to ". Currently, I suppress warnings in main.py because of that.
- making stimuli dynamic (3d output)
- more complex backgrounds
- more object shapes / size / orientation


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
