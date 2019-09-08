### related papers

not very well ordered yet

[Archer16: Black-box variational inference](https://arxiv.org/abs/1511.07367)

[Watter2015: Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images ](http://papers.nips.cc/paper/5964-embed-to-control-a-locally-linear-latent-dynamics-model-for-control-from-raw-images.pdf)

[Becker18: Recurrent Kalman Networks: Factorized Inference in High-Dimensional Deep Feature Spaces](https://openreview.net/forum?id=rkx1m2C5YQ)

[Chung15: A recurrent latent variable model for sequential data](http://papers.nips.cc/paper/5653-a-recurrent-latent-variable-model-for-sequential-data.pdf)

[Fraccaro16: Sequential neural models with stochastic layers](http://papers.nips.cc/paper/6039-sequential-neural-models-with-stochastic-layers.pdf)

[Fraccaro17: A disentangled recognition and nonlinear dynamics model for unsupervised learning. NIPS2017](http://papers.nips.cc/paper/6039-sequential-neural-models-with-stochastic-layers.pdf)


Their idea is to separate the modeling of dynamics from the model of the recognition/decoding model. The encoder/decoder of observables x_t is captured by (the same) VAE at each time point. The resulting latent states a_t are modeled as the output of a_t. This dynamics of a_t are then modeled via a linear gaussian state space model (LGSSM) with state variable z_t and output variable a_t. Hence,
- p_\gamma_t(a_t | z_t) = N(a_t; C_t z_t, R)
and
- p_\gamma_t(z_t|z_{t-1}, u_t) = N(z_t; A_t z_{t-1} + B_t u_t, Q)

To deal with possible non-linear dynamics (e.g., ball bounces off a wall), the parameters of the model \gamma are dependent on time. In particular, they learn a *dynamics parameter network* \alpha_t(a_{0:t-1}) with the output being weights that sum to one. These weights choose and interpolate between K different linear operating modes (e.g., $A_t = \sum^{K}_{k=1} \alpha^(k)_t(A_{0:t-1}) A^ (k)$, the same for B_t = ..., C_t = ...) of the LGSSM at different time-points (p. 4). They show empirically, that this is indeed what happens. Some mode A^(i) is "active" during linear motion across the image, whereas another A^(j) is active during the contact with the vertical boundaries left/right of the image.

[Karl17. Deep variational bayes filters: Unsupervised learning of state space models from raw data. ICLR2017](https://arxiv.org/pdf/1605.06432.pdf)
  
  Previous state-space models did not work well because they overemphasized reconstruction of the input over learning transition in latent space. Here, they force the latent space to fit the transition by requiring:
z_{t+1} = f(z_t, u_t, \beta_t),
where u_t are control inputs and the mapping of z_t and u_t to z_{t+1} is deterministic. \beta_t are (stochastic) parameters of the transition model. Two points they make here:
- "given the stochastic parameters \beta_t, the state transition is deterministic [...]. The immediate and crucial consequence is that errors in reconstruction of x_t from z_t are backpropagated directly through time." (p. 4)
- "the recognition model no longer infers latent states z_t, but transition parameters \beta_t." (p.4)

\beta_t = {w_t, v_t} is split into two sets of parameters: w_t, the sample-specific process noise which can be inferred from the input. And v_t: universal transition parameters, which are sample-independent.  Hence,
q_\phi(\beta_{1:T}|x_{1:T}) = q_\phi(\w_{1:T}|x_{1:T}) q_\phi(v_{1:T}) 

They derive the lower bound as:
\E_q_\phi [\ln p_\theta(x_{1:T}|z_{1:T}) - KL(q_\phi(\beta_{1:T}| x_{1:T},u_{1:T}) || p(\beta_{1:T}))

In their example, they look at locally linear state transitions:
z_{t+1} = A_t z_t + B_t u_t + C_t w_t

where A_t, B_t, C_t are linear superpositions of M matrices from
v_t = {A^(i)_t, B^(i)_t, C^(i)_t | i = 1, ..., M}, which are learned.

Their main comparison is with the Deep Kalman Filter (DKF, [Krishnan15](https://arxiv.org/pdf/1511.05121.pdf)). They take the inspiration for the locally linear functions from [Watter2015](http://papers.nips.cc/paper/5964-embed-to-control-a-locally-linear-latent-dynamics-model-for-control-from-raw-images).


[Krishnan17: structured inference networks for nonlinear state space models](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14215/14380)

[Henaff19: Perceptual straightening of natural videos](http://www.nature.com/articles/s41593-019-0377-4)

[Chung18: Classification and Geometry of General Perceptual Manifolds](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.031003)

[Cohen2019: Separability and Geometry of Object Manifoldsin Deep Neural Networks](https://www.biorxiv.org/content/biorxiv/early/2019/05/23/644658.full.pdf)

[Gregor19: Temporal difference variational auto-encoder](https://openreview.net/pdf?id=S1x4ghC9tQ)

[Caselles-Dupre2019: symmetry-based disentangled representation learning  requires interaction  with environments. ICLR2019](http://spirl.info/2019/camera-ready/spirl_camera-ready_17.pdf)
They present sequences of stimuli to the network which are generated by a random sequence of actions (up, down, left right shift of a white ball). They introduce the idea that transformations of latent representation across the sequence of stimuli should follow a linear transformation A_a. Where each action a, has its own transformation matric A_a.

[Watters2019: Spatial Broadcast Decoder:  A Simple Architecture for Learning Disentangled Representations in VAEs](https://arxiv.org/pdf/1901.07017.pdf)
By not using a deconvolution network but a "spatial broadcast decoder", they are able to better disentangle. Also, contains good thoughts on how to measure / visualize disentanglement.

[Tschannen18: Recent Advances in Autoencoder-Based Representation Learning](http://bayesiandeeplearning.org/2018/papers/151.pdf)
Good overview over recent VAE approaches.


Potentially interesting:

[Exarchakis2017: Discrete Sparse Coding](https://exarchakis.net/files/papers/NECO-09-16-2696R2-PDF.pdf)

[Shelhamer2016: Clockwork Convnets for Video Semantic Segmentation](https://link.springer.com/chapter/10.1007/978-3-319-49409-8_69)
From the abstract: "We propose a videorecognition framework that relies on two key observations: (1) while pixels may change rapidly from frame to frame, the semantic content of a  scene  evolves  more  slowly,  and  (2)  execution  can  be  viewed  as  an aspect of architecture, yielding purpose-fit computation schedules fornetworks. We define a novel family of “clockwork” convnets driven byfixed or adaptive clock signals that schedule the processing of differentlayers at different update rates according to their semantic stability."


[Lundquist17: Sparse Coding on Stereo Video for Object Detection](https://arxiv.org/pdf/1705.07144.pdf)
