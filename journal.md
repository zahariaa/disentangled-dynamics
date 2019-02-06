# DGM project journal

## 2019/01/28: start

I cloned the repo.


## 2019/02/06: project vision

### The Goal
The primary goal of my project is to create a generative model that can identify objects and predict their future locations, from sequences of images which have no labels. This “self-supervised” network would use its predictions of its own future latent states (which are a function of both its inputs and recurrent activity) as a training signal to learn relevant features.

This use of prediction for self-supervision is partly inspired by slow feature analysis, which assumes that object identities, and the collection of features that feed into those identity representations, remain relatively stable over time. The model should therefore only represent features and objects that have predictive value for the future.

The hope is that this learning procedure leads to more robust representations of objects (cf. [Lotter Kreiman Cox 2017](https://arxiv.org/1605.08104)).


### Possible tasks
One simple task for such a network would be to predict future frames of image sequences in which MNIST digits move across the image frame (see figure 4 in [Srivastava et al 2015](https://arxiv.org/abs/1502.04681) for one possible version of this task). I’d also want to plot next to the predicted frames a visualization of the latent representation (e.g., the learned object identity label and its predicted location).

Another task (probably outside the scope of this project) would be to caption the objects present in videos, again without being given any labels during training. Existing labeled datasets could be used to quantify model performance. 

### Half-baked implementation ideas
I’m still fuzzy on the details of how to implement this, because I don’t have a good sense of what has already been done, and what is achievable.

My first thought is that I’d want something like a combination of the GMM SVAE and LDS SVAE. This seems like a logical solution, but potentially less biologically-motivated.

I also liked the idea of using the categorical distribution to attempt to make the latent code as explicit as possible of an object identity representation, with each neuron in the layer having a one-to-one mapping to each object’s identity, rather than having a distributed code. The problem, though, is the categorical distribution wouldn’t allow simultaneous objects being present. So perhaps an L0-norm regularizer would be a more appropriate solution.


## 
