Welcome to my website! I'm an [IAIFI (Institute of Artificial Intelligence and Fundamental Interactions)](https://iaifi.org/) fellow at MIT. My research predominately lies at the intersection of machine learning, theoretical physics and mathematics. See the research section further down for more details.

## Research

### Numerical Geometry with Neural Networks
Compact Calabi-Yau manifolds are of great importance to string theorists and mathematicians. They serve as the building blocks for many supersymmetric string compactifications. Despite being studied for half a decade, and many examples being known to exist (see Enumerating Calabi-Yau Manifolds below), no are no closed analytic expressions for the Ricci-flat metrics on these spaces for D>4. However, recent progress has been made in finding numerical approximations to the Ricci-flat metrics, where the metric is represented by the neural network.

Access to the metric has allowed for the calculation of previously inaccessible data. For example, the masses and mixings of particles in the resulting 4D physics[^1] - a crucial component for the phenomenology of string theory. These have also allowed one to check a number of the assumptions made in string compactifications, in particular if the α' expansion is really under control in when the volume of the compact space is large. I've contributed to both of these problems in the papers and repositories below.

**Not so flat metrics**, Cristofero S. Fraser-Taliente, <ins>Thomas R. Harvey</ins>, Manki Kim. [2024 preprint](https://arxiv.org/abs/2411.00962). [GitHub Repository](https://github.com/kitft/cymetricap).

**Computation of quark masses from string theory**, Andrei Constantin, Cristofero S. Fraser-Taliente, <ins>Thomas R. Harvey</ins>, Andre Lukas, Burt Ovrut. Published in: Nucl.Phys.B 1010 (2025) 116778. [2024 preprint](https://arxiv.org/abs/2402.01615). [GitHub Repository](https://github.com/kitft/heteroticyukawas).

Although it wasn't used in any of the above publications, I have written [my own code](https://github.com/harveyThomas4692/jax_metric_CalabiYau) for finding these Ricci-flat metrics. The method is equivalent to the phi-model introduced in [cymetric](https://github.com/pythoncymetric/cymetric) but with projective neural networks that automatically satisfy the transition loss criteria. This was written mostly for my own amusement, as I wanted something, unlike the alternatives, in a functional programming paradigm. My hope being that this would maximise clarity, over speed, for someone who is more familiar with the mathematics than the computational methods. As such, I've used this when introducing mathematicians and physicists to numerical geometry. For research purposes, I suggest using either [cymetric](https://github.com/pythoncymetric/cymetric) or [cymyc](https://github.com/Justin-Tan/cymyc) which are both more generic.

[^1]: We should note that there is a special exception here for non-geometric models and the standard embedding for the heterotic string. Both of which have proven to be very restrictive for obtaining quasi-realistic physics.

### Discrete Optimisation for String Theory with Reinforcement Learning and Genetic Algorithms
All of the phenomenologically promising constructions from string theory result in vast landscapes. Focusing on perturbative IIB
sector, early estimates suggest O(10⁵⁰⁰) vacua by enumerating the possible choices
of flux. Later work extended the previous number to O(10²⁷²⁰⁰⁰) by considering
the strong-coupling regime with F-Theory. One may worry, that with such a gigantic “string landscape”, identifying those
few constructions of phenomenological interest may seem impossible. However, in
recent years, developments in machine learning have allowed for the navigation of
other large environments. For example, the AlphaGo project was able to train a
neural network to play the game of Go (a game with 10¹⁷² board configurations)
at the same level as the world’s best players. I have been at the forefront of using such heuristic algorithms, such as long time-horizon reinforcement learning, in this context. This line of research also formed the majority of [my DPhil thesis](https://ora.ox.ac.uk/objects/uuid:faa8c757-75f7-4660-a9e2-ba60bc3eacf1).

In order to tie in with an existing Mathematica codebase for string theory calculations, I realised REINFORCE, A2C and a genetic algorithm in Mathematica. For another project I also realise a genetic algorithm in C. The repositories of which can be found by following the links below
- [RL in Mathematica](https://github.com/harveyThomas4692/Mathematica-RL)
- [GA in Mathematica](https://github.com/harveyThomas4692/GAMathematica)
- [GA in C](https://github.com/harveyThomas4692/GA-C)

**Fermion Masses and Mixing in String-Inspired Models**
Andrei Constantin, Cristofero S. Fraser-Taliente, <ins>Thomas R. Harvey</ins>, Lucas T.Y. Leung, Andre Lukas. [2024 preprint](https://arxiv.org/abs/2410.17704)

**Decoding Nature with Nature's Tools: Heterotic Line Bundle Models of Particle Physics with Genetic Algorithms and Quantum Annealing**, Steve A. Abel, Andrei Constantin, <ins>Thomas R. Harvey</ins>, Andre Lukas, Luca A. Nutricati. Fortsch.Phys. 72 (2024) 2, 2300260. [2023 preprint](https://arxiv.org/abs/2306.03147). [GitHub Repository](https://github.com/harveyThomas4692/GA-LineBundles).

**String Model Building, Reinforcement Learning and Genetic Algorithms** Steve A. Abel, Andrei Constantin, <ins>Thomas R. Harvey</ins>, Andre Lukas. Contribution to: Nankai Symposium on Mathematical Dialogues. [2021 preprint](https://arxiv.org/abs/2111.07333).

**Evolving Heterotic Gauge Backgrounds: Genetic Algorithms versus Reinforcement Learning** Steve A. Abel, Andrei Constantin, <ins>Thomas R. Harvey</ins>, Andre Lukas. Fortsch.Phys. 70 (2022) 5, 2200034. [2021 preprint](https://arxiv.org/abs/2110.14029).

**Heterotic String Model Building with Monad Bundles and Reinforcement Learning** Andrei Constantin, <ins>Thomas R. Harvey</ins>, Andre Lukas. Fortsch.Phys. 70 (2022) 2-3, 2100186. [2021 preprint](https://arxiv.org/abs/2108.07316).

String theory is not the only area of high energy theory that has large landscapes. As such, I've also worked on applying these methods to inflationary cosmology and models of flavour physics

**Cosmic Inflation and Genetic Algorithms** Steve A. Abel, Andrei Constantin, <ins>Thomas R. Harvey</ins>, Andre Lukas. Fortsch.Phys. 71 (2023) 1, 2200161. [2022 preprint](https://arxiv.org/abs/2208.13804).

**Quark Mass Models and Reinforcement Learning** <ins>Thomas R. Harvey</ins>, Andre Lukas. JHEP 08 (2021) 161. [2021 preprint](https://arxiv.org/abs/2103.04759).

### Large language models for mathematics
**Generative Modelling for Mathematical Discovery** - Upcoming publication. [GitHub Repository](https://github.com/kitft/funsearch).

### Other Work
**Enumerating Calabi-Yau Manifolds: Placing Bounds on the Number of Diffeomorphism Classes in the Kreuzer-Skarke List** Aditi Chandra, Andrei Constantin, Cristofero S. Fraser-Taliente, <ins>Thomas R. Harvey</ins>, Andre Lukas. Fortsch.Phys. 72 (2024) 5, 2300264. [2023 preprint](https://arxiv.org/abs/2310.05909)

**Spatially homogeneous universes with late-time anisotropy**, Andrei Constantin, <ins>Thomas R. Harvey</ins>, Sebastian von Hausegger, Andre Lukas. Class.Quant.Grav. 40 (2023) 24, 245015. [2023 preprint](https://arxiv.org/abs/2212.03234)

## Education
**University of Oxford (DPhil) - Theoretical Physics**

Thesis Title: ["Navigating the string landscape with machine learning techniques"](https://ora.ox.ac.uk/objects/uuid:faa8c757-75f7-4660-a9e2-ba60bc3eacf1)

Advisor: Andre Lukas

**University of Cambridge (MASt) - MASt Part III of the Mathematical Tripos**

**Royal Holloway, University of London (MPhys) - Physics**

Martin-Holloway Prize for graduating top of science faculty

## Links and Contact
[Email](mailto:trharvey@mit.edu)
  [Inspire-HEP](https://inspirehep.net/authors/1851078?ui-citation-summary=true)
  [Google Scholar](https://scholar.google.com/citations?user=JwMJUhwAAAAJ&hl=en)
  [GitHub](https://github.com/harveyThomas4692)
  [Linkedin](https://www.linkedin.com/in/thomas-harvey-1ab91012a/)
  [Twitter/X](https://x.com/Airsofter4692) 
