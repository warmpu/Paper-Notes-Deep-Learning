# StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio _10 Jun 2014_

>We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. 

* Official paper: [ArXiv](https://arxiv.org/abs/1406.2661)
* Official code: [Github](https://github.com/goodfeli/adversarial)



# OVERVIEW

In the proposed adversarial nets framework:

  - the **generative model** is pitted against an adversary : **discriminative model** that learns to determine whether a sample is from the model distribution or the data distribution 

In this article:
  - they explore the special case when the generative model generates samples by passing random noise through a multilayer perceptron
  - the discriminative model is also a multilayer perceptron
  - they can train both models using only the highly successful backpropagation and dropout algorithms and sample from the generative model using only forward propagation

# Theory

## Adversarial nets

The adversarial modeling framework is most straightforward to apply when the models are both multilayer perceptrons. To learn the generator’s distribution $p_g$ over data x, we define:

  -  a prior on input noise variables : $p_\mathcal{z} (\mathcal{z})$
  -  a mapping to data space as $G(z; \theta_g)$. with G is differentiable function with hyperparameter theta_g
  -  second multilayer perceptron $D(x; \theta_d)$ that outputs a single scalar
     -    _D(x)_ represents the probability that x came from the data rather than $p_g$
  - Train D to maximize the probability of assigning the correct label to both training examples and samples from G

In other words, $D$ and $G$ play the following two-player minimax game with value function $V(G, D)$ :

$$\min _G \max _D V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]$$

See Figure 1 for a less formal, more pedagogical explanation of the approach:

![fig-theory](../asset/images/GAN/Gan-theory.png)

In practice, we must implement the game using an iterative, numerical approach:
  - Optimizing **D** to completion in the inner loop of training is computationally prohibitive, and on finite datasets would result in overfitting. 
  - Instead, we alternate between _k_ steps of optimizing **D** and one step of optimizing **G**. 
  - This results in **D** being maintained near its optimal solution, so long as **G** changes slowly enough. 

However, In practice, equation 1 may not provide sufficient gradient for **G** to learn well:
  - Early in learning, when **G** is poor, **D** can reject samples with high confidence because they are clearly different from the training data. 
  
  $$\log (1-D(G(\boldsymbol{z}))) \qquad \text{ is saturates}$$  
  
  - => rather minimize $\log (1-D(G(\boldsymbol{z}))) \text{ we can train G to maximize }  \log D(G(\boldsymbol{z}))$. 
  - This objective function results in the same fixed point of the dynamics of **G and D** but provides much stronger gradients early in learning.

# Alogrithm

![Alg](../asset/images/GAN/Gan-Alg.png)

## Global Optimality of $p_g=p_{\text {data }}$

* Proposition 1. For **G** fixed, the optimal discriminator **D** is
  
$$D_G^*(\boldsymbol{x})=\frac{p_{\text {data }}(\boldsymbol{x})}{p_{\text {data }}(\boldsymbol{x})+p_g(\boldsymbol{x})}$$

- the training objective for _D_ can be interpreted as maximizing the log-likelihood for estimating the conditional probability $P(Y=y \mid \boldsymbol{x})$
  -  **Y** indicates whether $\boldsymbol{x}$ comes from $p_{\text {data }}$ (with _y=1_ ) or from **p_g** (with _y=0_ )
  - The minimax game in Eq. 1 can now be reformulated as:

  $$\begin{aligned}C(G) &=\max _D V(G, D) \\
  &=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}}\left[\log D_G^*(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}}\left[\log \left(1-D_G^*(G(\boldsymbol{z}))\right)\right] \\
  &=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}}\left[\log D_G^*(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{x} \sim p_g}\left[\log \left(1-D_G^*(\boldsymbol{x})\right)\right] \\
  &=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}}\left[\log \frac{p_{\text {data }}(\boldsymbol{x})}{P_{\text {data }}(\boldsymbol{x})+p_g(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{x} \sim p_g}\left[\log \frac{p_g(\boldsymbol{x})}{p_{\text {data }}(\boldsymbol{x})+p_g(\boldsymbol{x})}\right]\end{aligned}$$


* Theorem 1. The global minimum of the virtual training criterion **C(G)** is achieved if and only if $p_g=p_{\text {data }}$. 
  - At that point, **C(G)** achieves the value $-\log 4$.
  - From the Proof in the paper, they have shown that:
    -  $C^*=-\log (4)$ is the global minimum of **C(G)**
    -  the only solution is $p_g=p_{\text {data }}$, i.e., the generative model perfectly replicating the data generating process.



* Proposition 2. If **G and D** have enough capacity, and at each step of Algorithm 1, the discriminator is allowed to reach its optimum given **G**, and $p_g$ is updated so as to improve the criterion

$$\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}}\left[\log D_G^*(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{x} \sim p_g}\left[\log \left(1-D_G^*(\boldsymbol{x})\right)\right]$$

  then $p_g$ converges to $p_{\text {data }}$

- In practice: 
  - adversarial nets represent a limited family of ***p_g*** distributions via the function $G\left(\boldsymbol{z} ; \theta_g\right)$
  - we optimize $\theta_g \text{ rather than } p_g$ 





