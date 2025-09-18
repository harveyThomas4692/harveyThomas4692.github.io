---
layout: default
---

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true
  }
});
</script>
# Some Comments and Discussion on Natural Gradient Methods
**Date: 17th of September 2025**

I’ve been interested in optimisation methods for some time and, unsurprisingly considering my background, have a particular interest in the application of geometry to optimisation. For example, in my [recent pre-print](https://arxiv.org/abs/2509.03594), I propose using the metric commonly employed in visualisations of loss landscapes (known as the pull-back metric) as a preconditioner for gradients. This approach led to a slight performance improvement over Adam, with negligible increases in computational or memory costs.

In that preprint, I note that the metric I chose is, from a geometric perspective, completely arbitrary. I specifically included the line:  “...different choices of metrics are, from a geometric rather than numerical perspective, somewhat arbitrary (except, perhaps, those used in natural gradient techniques [22,23])” [^1]. This naturally raises the question: what are natural gradient methods, and why are they less arbitrary?

I decided to write this blog post to present natural gradient methods as I understand them, with the aim of helping me remember my thoughts and hopefully providing value to others. I suspect this may be particularly useful for people with theoretical physics backgrounds who have (like myself) found introductions to this subject, typically presented through information geometry, difficult to follow. I should emphasise that I am far from an expert, but I believe this represents a somewhat different perspective from how the actual experts in this area typically present the topic. I just found myself writing down the same objects (specifically the FIM - more on that later), through my own explorations and also derived some interesting properties about it. Since I don't think anything I'm presenting here is fundamentally new, but rather a different perspective, it didn't seem appropriate to write this as a formal paper—hence my decision to present it as a blog post.

From the start I should say that I will be writing the equations as gradient flow:

$$\frac{d\theta^i}{dt} = -\frac{\partial L(\theta)}{\partial \theta^i},$$

as opposed to gradient descent:

$$\delta\theta^i = -\eta\frac{\partial L(\theta)}{\partial \theta^i}.$$

Gradient descent can be considered as the linearised approximation to gradient flow, where the learning rate $\eta = \delta t$ is the finite time step. This will also allow us to exactly solve some of the equations.

[^1]: The citations were to the seminal paper S. Amari, “Natural gradient works efficiently in learning,”  and a recent review R. Shrestha, “Natural gradient methods: Perspectives, efficient-scalable approximations, and analysis,”.

## Natural Gradient Methods
Natural gradient methods propose the use of a very specific metric as a preconditioner for your gradients. It's claimed in the literature that this is the optimal choice of metric on parameters for gradient flow/descent. In particular, given our gradient flow equation as:

$$\frac{d\theta^i}{dt} = - \sum_j g^{ij}\frac{\partial L(\theta)}{\partial \theta^j}.$$

Specifically, if we are optimising a function $f_\theta(x)$, parametrised by $\\\{\theta^i, i=1\ldots N\\\}$, then we should use:

$$g_{ij} = E_x\left(\frac{\partial f_\theta(x)}{\partial \theta^i}\frac{\partial f_\theta(x)}{\partial \theta^j}\right),$$

where $g_{ij}$ is the inverse of $g^{ij}$, and $E$ indicates expectation over the data $\\\{x^\alpha,\alpha=1...D\\\}$. Similarly, when we have a probability distribution $\rho_\theta$, we should then use:

$$g_{ij} = E_x\left(\frac{\partial\log \rho_\theta(x)}{\partial \theta^i}\frac{\partial\log \rho_\theta(x)}{\partial \theta^j}\right).$$

In this second case, $g_{ij}$ is referred to as the Fisher Information Matrix (or FIM for short). From now on I'll focus on the case of functions, mostly because it's simpler, however I will occasionally comment on the probability distribution case.

## Where does this metric come from?
When beginning to work on the paper I mentioned at the start, I actually began with the question: is there a **natural metric** on the space of parameters for a parametrised function? In the end, I didn't address that question in that paper, but instead asked whether there were any **useful metrics** on this space and the proposed one based on the common visualisations of loss landscapes. However, I did make some headway into the first question, which is where I rediscovered some results from natural gradient methods.

I tried to answer this question by first defining a metric on the more abstract space of functions, and then pulling back that metric to the sub-manifold defined by restricting to those parametrised functions. **This submanifold is the loss-landscape.** Given two functions $f(x)$ and $g(x)$, we can define a square-distance between them by integrating their difference squared (i.e. the $L_2$ distance):

$$d^2_{L_2}[f,g] = \int d\mu(x) (f(x) - g(x))^2,$$

where $d\mu(x)$ is **the measure induced by the data**. Intuitively, one should think of this as an integral over the data manifold. Now, if we consider the distance between $f(x)$ and $f(x) + \delta f(x)$, where $\delta f(x)$ is some small function variation, we can define a line element on this space of functions, given by:

$$|d f|^2 = d_{L_2}[f,f + \delta f] = \int d\mu(x) (\delta f(x))^2.$$

We restrict ourselves to our function class, be that a neural network architecture or something else, and in the process induce a metric on the space of parameters:

$$ \hat g = \sum_{i,j} g_{ij} d\theta^i d\theta^j = \sum_{i,j}\left(\int d\mu(x) \frac{\partial f_\theta(x)}{\partial \theta^i}\frac{\partial f_\theta(x)}{\partial \theta^j}\right)d\theta^i d\theta^j.$$

We will refer to this metric as the natural gradient metric from now. Rewriting the term in brackets as an expectation over the data:

$$g_{ij} = E_x\left(\frac{\partial f_\theta(x)}{\partial \theta^i}\frac{\partial f_\theta(x)}{\partial \theta^j}\right),$$

we see that we have the same metric proposed in the natural gradient literature. We could have played the same game with probability distributions, but we can't beginning with a different definition of distance (the $L_2$ distance is not well defined for two probability distributions). We instead start with two probability distributions $\rho$ and $\sigma$ and the Jensen-Shannon Divergence (which is a symmetrised form of the Kullback–Leibler divergence): 

$$ D_{JS}^2[\rho,\sigma] = \frac{1}{2}\int d\mu(x)\left[\rho(x)\log\left(\frac{\rho(x)}{\sigma(x)}\right) +  \sigma(x)\log\left(\frac{\sigma(x)}{\rho(x)}\right)\right]. $$

Once you define an analogous line element, and pull that back, you once again find the same metric proposed in the natural gradient literature.

## Why is this metric a good idea?
So far I have presented a metric on the space of parameters, but haven't explained why this is a good metric to train with. Let's say that we can actually realise an arbitrary function—in other words, we go to the infinite parameter limit. In this case, we really can impose functional gradient descent. That is:

$$\frac{df_t(x)}{dt} = -\frac{\delta L[f_t]}{\delta f_t(x)}.$$

In the above section, what we essentially did was say: **We want functional gradient flow—what is the closest I can get to that in terms of parameters?** This sounds theoretically elegant, but is it a good idea? To answer this, we first consider when it's possible for the natural gradient metric to reduce to the identity:

$$g_{ij} = \left(\int d\mu(x) \frac{\partial f_\theta(x)}{\partial \theta^i}\frac{\partial f_\theta(x)}{\partial \theta^j}\right) = \delta_{ij}.$$

This would imply that the set of functions 

$$f_i(x) = \frac{\partial f_\theta(x)}{\partial \theta^i}$$

are orthonormal to each other. This means that **the natural gradient metric reduces to the identity when we parameterise the function by orthonormal basis functions on the data manifold!** Whilst it isn't true that a set of orthonormal basis functions exist on any manifold, one sufficient condition is that the manifold is compact. Clearly, for any finite amount of data, there exist many possible smooth data manifolds that it could come from, some of which must also be compact. Therefore, it appears reasonable to assume such a set of functions exist, even if they are hard to compute in practice. As there exists a choice of coordinates such that the metric becomes the Euclidean metric everywhere, **the natural gradient metric is actually a flat metric on the loss landscape**. We were previously writing that metric in awkward coordinates.

Let's then proceed with, at least formally, parametrising our function in this way and write

$$f_\alpha(x) = \sum_i \alpha^i(\theta) f_i(x),$$

where we have written $\alpha$ as a function of our original parametrisation $\theta$[^2]. This really is just a reparametrisation—i.e., a change of coordinates on the data manifold. Gradient flow with $\alpha$, using the Euclidean metric, is mathematically the same as gradient flow with $\theta$ using the natural gradient metric. Given this, let's say we are running gradient descent on a regression problem, with target function:

$$g(x) = \sum_i \beta^i f_i(x),$$

where we use the $L_2$-loss:

$$L(\alpha) = \int d \mu(x)  (f(x) - g(x))^2, \quad \frac{d\alpha^i}{dt} = -\frac{\partial L(\alpha)}{\partial\alpha^i}.$$

Using the orthogonality of the basis functions, we can actually solve this differential equation exactly. The solution is given by

$$\alpha^i(t) = (\alpha^i(0)-\beta^i)e^{-2t} + \beta^i,$$

which implies that the loss is going as

$$L(\alpha(t)) \propto e^{-4t}.$$

So **when we train with the natural gradient metric, we should reach the optimal solution exponentially quickly**. This is clearly in stark contrast to the observed power law scaling laws observed when training neural networks with the flat metric.[^3]

Following this exact calculation with probability distributions seems much more difficult. There isn't really an equivalent to an orthonormal basis of functions in which you can expand a probability distribution. However, given you regularise your probability distributions to not be exactly zero or one anywhere, you can make an analogous argument using the logits.

[^2]: There is a slight subtlety here regarding inevitability of $\alpha^i(\theta)$, which we acknowledge but ignore. Though we do assume that we've quotiented out degeneracies in the parameters of neural networks (such as a rotation of the keys and inverse rotation of the queries in the attention mechanism).

[^3]: I should say that, strictly speaking, modern neural networks are usually trained with Adam/AdamW, which is actually a metric that changes during training. However, at each time step, it is still a flat metric on the loss landscape.


## Comparison to the Neural Tangent Kernel (NTK)
The natural gradient metric looks like a trace over the NTK. However, there are important conceptual differences. In a sense, we derived this equation from the opposite perspective. We stated we wanted functional gradient descent, and went looking for the metric on parameters that is induced from this:

$$\frac{d f_t(x)}{dt} = - \frac{\delta \mathscr L [f]}{\delta f_t(x)} \Rightarrow  \frac{df_{\theta(t)}(x)}{dt} = -g^{ij}\frac{\partial\mathscr L(\theta(t))}{\partial \theta^j(t)},$$

where $g^{ij}$ is the inverse of the induced metric. The NTK, however, is the opposite approach, we gradient descent on parameters, and look for the metric on functions that this corresponds to. In equations this is

$$\frac{d\theta^i(t)}{dt} = -\frac{\partial\mathscr L(\theta(t))}{\partial \theta^i(t)}\Rightarrow \frac{d f_t(x)}{dt} = - G^{-1}\left(\cdot\,, \frac{\delta \mathscr L [f]}{\delta f_t(x)}\right),$$

where $G^{-1}$ is the NTK. As such it should not come as a surprise that $g^{ij}$ has similarities to an "inverse NTK".

## Is this practical?
In short: very much no... 

Calculating this metric requires integrating $N^2$ integrals over the data manifold, which scales as $O(DN^2)$—far from practical for deep learning. Furthermore, even if we were given the inverse natural gradient metric, we would still need to perform $N^2$ calculations to compute the update, since the metric is generically non-diagonal.

Despite this, I remain hopeful that approximate versions of the natural gradient metric can be calculated. K-FAC is, in fact, one such approximation where $g^{ij}$ is replaced with a low-rank approximation. I've been actively thinking in this direction and remain optimistic that methods may exist that offer benefits over Adam/AdamW.