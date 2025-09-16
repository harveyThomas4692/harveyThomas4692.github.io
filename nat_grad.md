# Some Comments and Discussion on Natural Gradient Methods
I’ve been interested in optimisation methods for some time and, unsurprisingly considering my background, have a particular interest in the application of geometry to optimisation. For example, in my [recent pre-print](https://arxiv.org/abs/2509.03594), I propose using the metric commonly employed in visualisations of loss landscapes (known as the pull-back metric) as a preconditioner for gradients. This approach led to a slight performance improvement over Adam, with negligible increases in computational or memory costs.

In that preprint, I note that the metric I chose is, from a geometric perspective, completely arbitrary. I specifically included the line:  “...different choices of metrics are, from a geometric rather than numerical perspective, somewhat arbitrary (except, perhaps, those used in natural gradient techniques [22,23])” [^1]. This naturally raises the question: what are natural gradient methods, and why are they less arbitrary?

I decided to write this blog post to present natural gradient methods as I understand them, with the aim of helping me remember my thoughts and hopefully providing value to others. I suspect this may be particularly useful for people with theoretical physics backgrounds who have found introductions to this subject, typically presented through information geometry, difficult to follow. I should emphasise that I am far from an expert, but I believe this represents a somewhat different perspective from how the actual experts in this area typically present the topic. I just found myself writing down the same objects (specifically the FIM - more on that later), through my own explorations and also derived some some interesting properties about it. Since I don't think anything I'm presenting here is fundamentally new, it didn't seem appropriate to write this as a formal paper—hence my decision to present it as a blog post.

**TLDR:** Natural gradient methods propose the use of a very specific metric as a preconditioner for your gradients. In particular, given our gradient flow equation as:

$$\frac{d\theta^i}{dt} = - \sum_j g^{ij}\frac{\partial L_\theta(f_\theta(x))}{\partial \theta^j}.$$

Specifically, if we are optimising a function $f_\theta(x)$, parameterised by $\\\{\theta^i, i=1\ldots N\\\}$, then we should use:

$$g_{ij} = E_x\left(\frac{\partial f_\theta(x)}{\partial \theta^i}\frac{\partial f_\theta(x)}{\partial \theta^j}\right),$$

where $g_{ij}$ is the inverse of $g^{ij}$, and $E$ indicates expectation over the data $\\\{x^\alpha,\alpha=1...D\\\}$. Similarly, when we have a probability distribution $\rho_\theta$, we should then use:

$$g_{ij} = E_x\left(\frac{\partial\log \rho_\theta(x)}{\partial \theta^i}\frac{\partial\log \rho_\theta(x)}{\partial \theta^j}\right).$$

In this second case, $g_{ij}$ is referred to as the Fisher Information Matrix (or FIM for short). 

I'll focus on the case of functions, mostly because it's simpler, however I will occasionally comment on the probability distribution case.

## Where does this metric come from?
When beginning to work on the paper I mentioned at the start, I actually began with the question: is there a _natural metric_ on the space of parameters for a parameterised function? In the end, I didn't address that question in that paper, but instead asked, and answered, whether there were any _useful metrics_ on this space. However, I did make some headway into the first question, which is where I rediscovered some results from natural gradient methods.

I tried to answer this question by first defining a metric on the more abstract space of functions, and then pulling back that metric to the sub-manifold defined by restricting to those parameterised functions. Given two functions $f(x)$ and $g(x)$, we can define a distance between them by integrating their difference squared (i.e. the $L_2$ distance):

$$d_{L_2}[f,g] = \int d\mu(x) (f(x) - g(x))^2,$$

where $d\mu(x)$ is _the measure induced by the data_. Intuitively, one should think of this as an integral over the data manifold. Now, if we consider the distance between $f(x)$ and $f(x) + \delta f(x)$, where $\delta f(x)$ is some small function variation, we can define a line element on this space of functions, given by:

$$|d f|^2 = d_{L_2}[f,f + \delta f] = \int d\mu(x) (\delta f(x))^2.$$

We restrict ourselves to our function class, be that a neural network architecture or something else, and in the process induce a metric on the space of parameters:

$$ \hat g = \sum_{i,j} g_{ij} d\theta^i d\theta^j = \sum_{i,j}\left(\int d\mu(x) \frac{\partial f_\theta(x)}{\partial \theta^i}\frac{\partial f_\theta(x)}{\partial \theta^j}\right)d\theta^i d\theta^j.$$

Rewriting the term in brackets as an expectation over the data:

$$g_{ij} = E_x\left(\frac{\partial f_\theta(x)}{\partial \theta^i}\frac{\partial f_\theta(x)}{\partial \theta^j}\right),$$

we see that we have the same metric proposed in the natural gradient literature. We could have played the same game with probability distributions, but we can't begining with a different definition of distance (the $L_2$ distance is not well defined for two probability distributions). We instead start with the Jensen-Shannon Divergence (which is a symmetrised form of the Kullback–Leibler divergence). On pulling back you once again find the same metric propeosed in the natural gradient literature.

## Why is this metric a good idea?
So far I have presented a metric on the space of paramters, but haven't explained why this is a good metric to train with. Let's say that we can actually realise an arbitary function, in other words we go to the infinite paramter limit. In this case, we really can impose functional gradient descent. That is:

$$\frac{df_t(x)}{dt} = -\frac{\delta L[f_t]}{\delta f_t(x)}.$$

## Comparison to the Neural Tangent Kernel (NTK)

## Is this practical?
In short: very much no...

[^1]: The citations were to the seminal paper S. Amari, “Natural gradient works efficiently in learning,”  and a recent review R. Shrestha, “Natural gradient methods: Perspectives, efficient-scalable
approximations, and analysis,”.
