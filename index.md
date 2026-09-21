Welcome to my website! I'm an [IAIFI (Institute of Artificial Intelligence and Fundamental Interactions)](https://iaifi.org/) fellow at MIT. My research predominantly lies at the intersection of machine learning, theoretical physics and mathematics. See the research section further down for more details.
- [Research](#research)
- [Notes](#notes)
- [Blog Posts](#blog-posts)
- [Education](#education)
- [Outreach](#outreach)
- [Links and Contact](#links-and-contact)

# Research
- [AI Concepts for Physics](#ai-concepts-for-physics)
- [Large Language Models for Agentic Mathematics](#large-language-models-for-agentic-mathematics)
- [Numerical Geometry with Neural Networks](#numerical-geometry-with-neural-networks)
- [Physics for/of AI](#physics-forof-ai)
- [Discrete Optimisation for String Theory with Reinforcement Learning and Genetic Algorithms](#discrete-optimisation-for-string-theory-with-reinforcement-learning-and-genetic-algorithms)
- [Other Work](#other-work)

**Unless indicated with a † all papers are listed with authors in alphabetical order. This is the standard convention in high energy theory and pure mathematics.**

## AI Concepts for Physics
I am exploring how conceptual and statistical frameworks from AI theory can provide new perspectives on foundational questions in theoretical physics. While there has been some work in this direction, this remains a relatively underexplored avenue. 

**What Neural Network Field Theory Can and Cannot Realise on a Computer**, <ins>Thomas R. Harvey</ins>. [2026 preprint](https://arxiv.org/abs/2608.21523)

**Naturalness and Fisher Information**, James Halverson <ins>Thomas R. Harvey</ins>, Michael Nee. [2026 preprint](https://arxiv.org/abs/2603.01411).

## Large Language Models for Agentic Mathematics
Large language models have proven to be surprisingly effective coders. Funsearch makes use of this by incorporating a genetic algorithm for the large language model prompts, creating a new form of genetic programming. My research on this has demonstrated that Funsearch is able to learn various combinatorial and number-theoretic problems. Further to this, we presented a comprehensive codebase allowing for others to easily run funsearch themselves. On a related front, I used very similar methodology to create a new approach to symbolic regression based on multimodel large language models and Kolmogorov–Arnold networks.

**Symbolic Regression with Multimodal Large Language Models and Kolmogorov–Arnold Networks †**, <ins>Thomas R. Harvey</ins>, Fabian Ruehle, Kit Fraser-Taliente, James Halverson. Published in: MLST (2026). [2025 preprint](https://arxiv.org/abs/2505.07956). [GitHub Repository](https://github.com/harveyThomas4692/llmlex).

**Generative Modelling for Mathematical Discovery**, Jordan S. Ellenberg, Kit Fraser-Taliente, <ins>Thomas R. Harvey</ins>, Karan Srivastava, Andrew V. Sutherland. Advances in Theoretical and Mathematical Physics: proceedings of the CMSA Mathematics and Machine Learning Program 2025. [2025 preprint](https://arxiv.org/abs/2503.11061). [GitHub Repository](https://github.com/kitft/funsearch).

## Numerical Geometry with Neural Networks
Compact Calabi-Yau manifolds are of great importance to string theorists and mathematicians. They serve as the building blocks for many supersymmetric string compactifications. Despite being studied for half a century, and many examples being known to exist (see Enumerating Calabi-Yau Manifolds below), there are no closed analytic expressions for the Ricci-flat metrics on these spaces for D>4. However, recent progress has been made in finding numerical approximations to the Ricci-flat metrics, where the metric is represented by the neural network. These are sometimes called Physics-informed neural networks or PINNs for short.

Access to the metric has allowed for the calculation of previously inaccessible data. For example, the masses and mixings of particles in the resulting 4D physics[^1] - a crucial component for the phenomenology of string theory. These have also allowed one to check a number of the assumptions made in string compactifications, in particular if the α' expansion is really under control in when the volume of the compact space is large. I've contributed to both of these problems in the papers and repositories below.

**Calculating Axion-Matter Couplings in String Theory** Joshua N. Benabou, Naomi Gendler, <ins>Thomas R. Harvey</ins>, Jakob Moritz. [2026 preprint](https://arxiv.org/abs/2609.20911). [GitHub Repository](https://github.com/harveyThomas4692/axion-matter-couplings).

**Not so flat metrics**, Kit Fraser-Taliente, <ins>Thomas R. Harvey</ins>, Manki Kim. [2024 preprint](https://arxiv.org/abs/2411.00962). [GitHub Repository](https://github.com/kitft/cymetricap).

**Computation of quark masses from string theory**, Andrei Constantin, Kit Fraser-Taliente, <ins>Thomas R. Harvey</ins>, Andre Lukas, Burt Ovrut. Nucl.Phys.B 1010 (2025) 116778. [2024 preprint](https://arxiv.org/abs/2402.01615). [GitHub Repository](https://github.com/kitft/heteroticyukawas).

Although it wasn't used in any of the above publications, I have written [my own code](https://github.com/harveyThomas4692/jax_metric_CalabiYau) for finding these Ricci-flat metrics. The method is equivalent to the phi-model introduced in [cymetric](https://github.com/pythoncymetric/cymetric) but with projective neural networks that automatically satisfy the transition loss criteria. This was written mostly for my own amusement, as I wanted something, unlike the alternatives, in a functional programming paradigm. My hope being that this would maximise clarity, over speed, for someone who is more familiar with the mathematics than the computational methods. As such, I've used this when introducing mathematicians and physicists to numerical geometry. For research purposes, I suggest using either [cymetric](https://github.com/pythoncymetric/cymetric) or [cymyc](https://github.com/Justin-Tan/cymyc) which are both more generic.

[^1]: We should note that there is a special exception here for non-geometric models and the standard embedding for the heterotic string. Both of which have proven to be very restrictive for obtaining quasi-realistic physics.

## Physics for/of AI
Training neural networks is central to deep learning, and the choice of optimiser is often critical for success. Many standard methods (such as Adam) can be viewed as imposing a particular metric on the parameter space. My recent work explores new optimisers that take this perspective literally, using principled metrics on the loss landscape as a guide for training.

**Sven: Singular Value Descent as a Computationally Efficient Natural Gradient Method †**, Samuel Bright-Thonney*, <ins>Thomas R. Harvey</ins>\*, Andre Lukas, Jesse Thaler. [2026 preprint](https://arxiv.org/abs/2604.01279). [GitHub Repository](https://github.com/sambt/sven-experiments/).

*Equal contribution.

**The Optimiser Hidden in Plain Sight: Training with the Loss Landscape's Induced Metric**, <ins>Thomas R. Harvey</ins>. [2025 preprint](https://arxiv.org/abs/2509.03594). [GitHub Repository](https://github.com/harveyThomas4692/Induced-Metric-Optimiser).

## Discrete Optimisation for String Theory with Reinforcement Learning and Genetic Algorithms
All of the phenomenologically promising constructions from string theory result in vast landscapes. Focusing on perturbative IIB
sector, early estimates suggest O(10⁵⁰⁰) vacua by enumerating the possible choices
of flux. Later work extended the previous number to O(10²⁷²⁰⁰⁰) by considering
the strong-coupling regime with F-Theory. One may worry, that with such a gigantic “string landscape”, identifying those
few constructions of phenomenological interest may seem impossible. However, in
recent years, developments in machine learning have allowed for the navigation of
other large environments. For example, the AlphaGo project was able to train a
neural network to play the game of Go (a game with 10¹⁷² board configurations)
at the same level as the world’s best players. I have been at the forefront of using such heuristic algorithms, such as long time-horizon reinforcement learning, in this context. This line of research also formed the majority of [my DPhil thesis](https://ora.ox.ac.uk/objects/uuid:faa8c757-75f7-4660-a9e2-ba60bc3eacf1).

In order to tie in with an existing Mathematica codebase for string theory calculations, I implemented REINFORCE, A2C and a genetic algorithm in Mathematica. For another project I also realised a genetic algorithm in C. The repositories of which can be found by following the links below
- [RL in Mathematica](https://github.com/harveyThomas4692/Mathematica-RL)
- [GA in Mathematica](https://github.com/harveyThomas4692/GAMathematica)
- [GA in C](https://github.com/harveyThomas4692/GA-C)

**Fermion Masses and Mixing in String-Inspired Models**
Andrei Constantin, Kit Fraser-Taliente, <ins>Thomas R. Harvey</ins>, Lucas T.Y. Leung, Andre Lukas. JHEP 06 (2025) 175. [2024 preprint](https://arxiv.org/abs/2410.17704). [GitHub Respository](https://github.com/LucasLeung149/stringFNmodels)

**Decoding Nature with Nature's Tools: Heterotic Line Bundle Models of Particle Physics with Genetic Algorithms and Quantum Annealing**, Steve A. Abel, Andrei Constantin, <ins>Thomas R. Harvey</ins>, Andre Lukas, Luca A. Nutricati. Fortsch.Phys. 72 (2024) 2, 2300260. [2023 preprint](https://arxiv.org/abs/2306.03147). [GitHub Repository](https://github.com/harveyThomas4692/GA-LineBundles).

**String Model Building, Reinforcement Learning and Genetic Algorithms** Steve A. Abel, Andrei Constantin, <ins>Thomas R. Harvey</ins>, Andre Lukas. Contribution to: Nankai Symposium on Mathematical Dialogues. [2021 preprint](https://arxiv.org/abs/2111.07333).

**Evolving Heterotic Gauge Backgrounds: Genetic Algorithms versus Reinforcement Learning** Steve A. Abel, Andrei Constantin, <ins>Thomas R. Harvey</ins>, Andre Lukas. Fortsch.Phys. 70 (2022) 5, 2200034. [2021 preprint](https://arxiv.org/abs/2110.14029).

**Heterotic String Model Building with Monad Bundles and Reinforcement Learning** Andrei Constantin, <ins>Thomas R. Harvey</ins>, Andre Lukas. Fortsch.Phys. 70 (2022) 2-3, 2100186. [2021 preprint](https://arxiv.org/abs/2108.07316).

String theory is not the only area of high energy theory that has large landscapes. As such, I've also worked on applying these methods to inflationary cosmology and models of flavour physics

**Cosmic Inflation and Genetic Algorithms** Steve A. Abel, Andrei Constantin, <ins>Thomas R. Harvey</ins>, Andre Lukas. Fortsch.Phys. 71 (2023) 1, 2200161. [2022 preprint](https://arxiv.org/abs/2208.13804).

**Quark Mass Models and Reinforcement Learning** <ins>Thomas R. Harvey</ins>, Andre Lukas. JHEP 08 (2021) 161. [2021 preprint](https://arxiv.org/abs/2103.04759).

## Other Work
**Enumerating Calabi-Yau Manifolds: Placing Bounds on the Number of Diffeomorphism Classes in the Kreuzer-Skarke List** Aditi Chandra, Andrei Constantin, Kit Fraser-Taliente, <ins>Thomas R. Harvey</ins>, Andre Lukas. Fortsch.Phys. 72 (2024) 5, 2300264. [2023 preprint](https://arxiv.org/abs/2310.05909)

**Spatially homogeneous universes with late-time anisotropy**, Andrei Constantin, <ins>Thomas R. Harvey</ins>, Sebastian von Hausegger, Andre Lukas. Class.Quant.Grav. 40 (2023) 24, 245015. [2023 preprint](https://arxiv.org/abs/2212.03234)

# Notes
**String Theory** - 2026

In Mid 2026, I found myself with a free Claude Max subscription (thanks Sid!). While I have found this particularly useful for research, I decided to try something a bit different with Claude Code. Over the years, I have built up a large collection of notes on various topics in string theory. None of these notes are complete, and many are imprecise. I was curious whether Claude could turn these into a complete set of lecture notes for string theory. It took several days, but the result is the set of notes below. There is, perhaps unsurprisingly, a heavy weighting toward geometry and compactifications, as these notes are based on my own personal research interests. Claude also claims in the abstract that this can fit into a single graduate course over one semester; I think that is a little ambitious, but maybe others will disagree.

A word on the process: most of my steering happened in the planning and syllabus stages, after which Claude worked largely on its own. In truth, I do not even know how much it actually drew on my notes versus its own training. I had it create virtual committees of experts to debate the correctness of each chapter as it was written, of course this is no substitute for genuine peer review.

A caveat before you dive in: I have not yet finished reading through the notes myself (especially the exercises and solutions), so nothing here should be treated as verified. Please do reach out if you find any typos or mistakes. I am particularly nervous about the citations, since language models are known to fabricate plausible-looking references.

I should state something important up front: I love physics textbooks and have a significant collection. I do not want this to be viewed as in any way a replacement for the human effort that goes into writing such works, and the unique perspectives they often provide. The material in these notes is relatively standard; the only difference is the selection and emphasis of topics, tailored to my own tastes.

[Notes](strings.pdf)

[TeX Files](tex_string.zip)

Solutions to appear...

# Blog Posts
I recently decided to add this section to my website, as I occasionally find myself working on or thinking about topics that, for various reasons, are not always appropriate for formal publication, but which I would still like to share with the broader community. My website seemed like the natural place for these less formal discussions.

[Some Comments and Discussion on Natural Gradient Methods](nat_grad.md) - 17th Sept 2025

# Education
**University of Oxford (DPhil)** - Theoretical Physics (2020-2024)

Thesis Title: ["Navigating the string landscape with machine learning techniques"](https://ora.ox.ac.uk/objects/uuid:faa8c757-75f7-4660-a9e2-ba60bc3eacf1)

Advisor: Andre Lukas

**University of Cambridge (MASt)** - Part III of the Mathematical Tripos (2019-2020)

**Royal Holloway, University of London (MSci)** - Physics (2015-2019)

Martin-Holloway Prize for graduating top of science faculty

# Outreach
When at Cambridge I met Alessandro Roussel, who runs the popular science channel [ScienceClic](https://www.youtube.com/@ScienceClicEN), along with the [original French Channel](https://www.youtube.com/c/ScienceClic). I have been involved (and am still involved) in giving scientific advise for the channel. 

Some of the most watched videos include I am credited for (views listed at time of writing):
- 4.7M [String Theory](https://youtu.be/n7cOlBxtKSo?si=tTSxxldB1bXPRMoQ)
- 2.4M [Quantum Field Theory Visualised](https://youtu.be/MmG2ah5Df4g?si=mFnxUnJmMgKlIco9)
- 516K [M-Theory](https://youtu.be/plU31-qtZ78?si=2zIdoiSaThUk7RVf)
- 480K [Supersymmetry, Explained Visually](https://youtu.be/0GUTJQCeKBE?si=mruhR-EgCX9h-k3l)
- 270k [The Physics of A.I.](https://youtu.be/dRkehLL19Wo?si=hdSWcdLfqiH_-3zz)

# Links and Contact
- [Email](mailto:trharvey@mit.edu)
- [Google Scholar](https://scholar.google.com/citations?user=JwMJUhwAAAAJ&hl=en)
- [Inspire-HEP](https://inspirehep.net/authors/1851078?ui-citation-summary=true)
- [GitHub](https://github.com/harveyThomas4692)
- [LinkedIn](https://www.linkedin.com/in/thomas-harvey-1ab91012a/)
- [Twitter/X](https://x.com/Airsofter4692)
- [IMDB (Apparently)](https://www.imdb.com/name/nm13520016/)
