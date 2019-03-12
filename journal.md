# DGM project journal

## 2019/01/28: start

I cloned the repo.


## 2019/02/06: project vision

### The Goal
The primary goal of my project is to create a generative model that can identify objects and predict their future locations, from sequences of images which have no labels. This “self-supervised” network would use its predictions of its own future latent states (which are a function of both its inputs and recurrent activity) as a training signal to learn relevant features.

This use of prediction for self-supervision is partly inspired by slow feature analysis, which assumes that object identities, and the collection of features that feed into those identity representations, remain relatively stable over time. The model should therefore only represent features and objects that have predictive value for the future.

The hope is that this learning procedure leads to more robust representations of objects (cf. [Lotter Kreiman Cox 2017](https://arxiv.org/abs/1605.08104)).


### Possible tasks
One simple task for such a network would be to predict future frames of image sequences in which MNIST digits move across the image frame (see figure 4 in [Srivastava et al 2015](https://arxiv.org/abs/1502.04681) for one possible version of this task). I’d also want to plot next to the predicted frames a visualization of the latent representation (e.g., the learned object identity label and its predicted location).

Another task (probably outside the scope of this project) would be to caption the objects present in videos, again without being given any labels during training. Existing labeled datasets could be used to quantify model performance. 

### Half-baked implementation ideas
I’m still fuzzy on the details of how to implement this, because I don’t have a good sense of what has already been done, and what is achievable.

My first thought is that I’d want something like a combination of the GMM SVAE and LDS SVAE. This seems like a logical solution, but potentially less biologically-motivated.

I also liked the idea of using the categorical distribution to attempt to make the latent code as explicit as possible of an object identity representation, with each neuron in the layer having a one-to-one mapping to each object’s identity, rather than having a distributed code. The problem, though, is the categorical distribution wouldn’t allow simultaneous objects being present. So perhaps an L<sub>0</sub>-norm regularizer would be a more appropriate solution.


## 2019/03/12: literature search update
Some of the class readings had either (1) come up in early my literature searches for this project (Importance Weighted Autoencoders), or (2) informed me of relevant work that I was previously unaware of (InfoGAN, &\beta;-VAE, Fixing a broken ELBO).
Specifically, these share the ideas of an alternative objective function; together, they prioritize a high entropy latent code and high mutual information between the latents and the inputs.
The apparent factorization of input generating factors is appealing, and directly related to my goal of learning object identity representations that are robust to different views, lighting, etc, as the Lotter et al (2017) paper aims to do.

More directly related to the project is the Contrastive Predictive Coding (CPC) paper by [van den Oord et al 2019](https://arxiv.org/abs/1807.03748), and related, older papers such as [Whitney et al 2016](https://arxiv.org/abs/1602.06822) from the Tenenbaum group and [Doersch et al 2016](https://arxiv.org/abs/1505.05192).
These papers use prediction (spatially in the first and last papers, and temporally, in the first and second) as a self-supervision signal to attempt to learn more "useful" object representations.
Furthermore, CPC uses an objective function that implicitly maximizes the mutual information between latent "context" variables and _future_ inputs, by explicitly minimizing normalized density ratios of inferred _future latent_ variables and samples from the posterior.

Though the CPC paper mentions training the model on a visual spatial prediction task (given half an image, predict the other half), they do not actually show these results.
They do, however, report state-of-the-art image classification performance for unsupervised learning from decoding these learned representations.
Their network, however, uses a ResNet for the encoder and a "PixelCNN-style autoregressive model".
I am still not familiar with PixelCNN and autoregressive models in general (including [Kingma et al 2016 Improved Variational Inference with Inverse Autoregressive Flow](https://papers.nips.cc/paper/6581-improved-variational-inference-with-inverse-autoregressive-flow.pdf)), so this is the natural next step in the literature search.

As for the direction of this project, I hope to implement a variant of CPC that can work at multiple layers---that is, the model should be making predictions of future latent variables at multiple levels of abstraction.

## 
