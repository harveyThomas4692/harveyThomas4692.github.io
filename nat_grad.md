# Some Comments and Discussion on Natural Gradient Methods
I’ve been interested in optimisation methods for some time and, unsurprisingly considering my background, have a particular interest in the application of geometry to optimisation. For example, in my [recent pre-print](https://arxiv.org/abs/2509.03594), I propose using the metric commonly employed in visualisations of loss landscapes (known as the pull-back metric) as a preconditioner for gradients. This approach led to a slight performance improvement over Adam, with negligible increases in computational or memory costs.

In that preprint, I note that the metric I chose is, from a geometric perspective, completely arbitrary. I specifically included the line:  “...different choices of metrics are, from a geometric rather than numerical perspective, somewhat arbitrary (except, perhaps, those used in natural gradient techniques [22,23])” [^1]. This naturally raises the question: what are natural gradient methods, and why are they less arbitrary?

I decided to write this blog post to present natural gradient methods as I understand them, with the aim of helping me remember my thoughts and hopefully providing value to others. I suspect this may be particularly useful for people with theoretical physics backgrounds who have found introductions to this subject, typically presented through information geometry, difficult to follow. I should emphasise that I am far from an expert, but I believe this represents a somewhat different perspective from how the actual experts in this area typically present the topic. Since I don't think anything I'm presenting here is fundamentally new, it didn't seem appropriate to write this as a formal paper—hence my decision to present it as a blog post.

**TLDR:** Natural gradient methods propose the use of a very specific metric as a preconditioner for your gradients. In particular, given our gradient flow equation as:

$$\frac{d\theta^i}{dt} = - \sum_j g^{ij}\frac{\partial L_\theta(f_\theta(x))}{\partial \theta^j}$$

Specifically, if we are optimising a function $f_\theta(x)$, parameterised by $\\\{\theta^i, i=1\ldots N\\\}$, then we should use:

$$g_{ij} = E_x\left(\frac{\partial f_\theta(x)}{\partial \theta^i}\frac{\partial f_\theta(x)}{\partial \theta^j}\right)$$

where $g_{ij}$ is the inverse of $g^{ij}$, and $E$ indicates expectation over the data $\\\{x^\alpha,\alpha=1...D\\\}$. Similarly, when we have a probability distribution $\rho_\theta$, we should then use:

$$g_{ij} = E_x\left(\frac{\partial\log \rho_\theta(x)}{\partial \theta^i}\frac{\partial\log \rho_\theta(x)}{\partial \theta^j}\right)$$

In this second case, $g_{ij}$ is referred to as the Fisher Information Matrix (or FIM for short). 

I'll focus on the case of functions, mostly because it's simpler, however I will occasionally comment on the probability distribution case.

## Where does this metric come from?

## Why is this metric a good idea?

## Is this practical?
In short: very much no...

[^1]: The citations were to the seminal paper S. Amari, “Natural gradient works efficiently in learning,”  and a recent review R. Shrestha, “Natural gradient methods: Perspectives, efficient-scalable
approximations, and analysis,”.
