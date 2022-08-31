# Paper title
Re-Ranking Person Re-Identification With k-Reciprocal Encoding. _09 November 2017(IEEE)_

>Our hypothesis is that if a gallery image is similar to the probe in the k-reciprocal nearest neighbors, it is more likely to be a true match. Specifically, given an image, a k-reciprocal feature is calculated by encoding its k-reciprocal nearest neighbors into a single vector, which is used for re-ranking under the Jaccard distance. The final distance is computed as the combination of the original distance and the Jaccard distance.

* Official paper: [IEEE](https://ieeexplore.ieee.org/document/8099872)

# Overview

1. Problem Definition
2. K-reciprocal Nearest Neighbors
3. Jaccard Distance
4. Local Query Expansion
5. Final Distance
6. Complexity Analysis

# 1. Problem Definition
- Given a probe person $p$ and the gallery set \ 
$\mathcal{G}=\{g_{i} \mid i=1,2, \ldots N \}$, 
the original distance between two persons $p$ and $g_{i}$ can be measured by Mahalanobis distance,

$$d\left(p, g_{i}\right)=\left(x_{p}-x_{g_{i}}\right)^{\top} \mathbf{M}\left(x_{p}-x_{g_{i}}\right)$$

where $x_{p}, x_{g_{i}}$ are probe _p_ and gallery _ith-g_, and __M__ is a positive semidefinite matrix.

- The initial ranking list $\mathcal{L}(p, \mathcal{G})=\{g_{1}^{0}, g_{2}^{0}, \ldots g_{N}^{0}\}$ can be obtained according to the pairwise original distance between probe $p$ and gallery $g_{i}$, 
where $\mathcal{d} (p, g_{i}^{0}) < \mathcal{d}(p, g_{i+1}^{0})$. 

> Our goal is to re-rank $\mathcal{L}(p, \mathcal{G})$, so that more positive samples rank top in the list, and thus to improve the performance of person re-identification (re-ID).

# 2. K-reciprocal Nearest Neighbors
- Define $N(p, k)$ as the _k_-nearest neighbors of _p_
$$ N(p, k) = \{g_{1}^{0}, g_{2}^{0}, \ldots, g_{k}^{0} \},|N(p, k)|=k $$
where |.| is the number of candidates in the set

- The _k_-reciprocal nearest neighbors $\mathcal{R}(p,k)$ can be defined as the _k_-reciprocal nearest neighbors are more related to probe _p_ than _k_-nearest neighbors:
$$ \mathcal{R}(p, k)= \{g_{i} \mid (g_{i} \in N(p, k) ) \wedge (p \in N (g_{i}, k))\} $$

- However, due to variations in illuminations, poses, views and occlusions, the positive images may be excluded from the _k_-nearest neighbors. To address this problem, we incrementally add the $\frac{1}{2}$ _k_-reciprocal nearest neighbors of each candidate in $\mathcal{R}(p, k)$ into a more robust set  
$\mathcal{R}^{*}(p, k)$ according to the following condition:

$$\mathcal{R}^{*}(p, k) \leftarrow \mathcal{R}(p, k) \cup \mathcal{R}\left(q, \frac{1}{2} k\right)$$

$$ \text { s.t. }\left|\mathcal{R}(p, k) \cap \mathcal{R}\left(q, \frac{1}{2} k\right)\right| \geqslant \frac{2}{3}\left|\mathcal{R}\left(q, \frac{1}{2} k\right)\right| $$

$$\forall q \in \mathcal{R}(p, k)$$

# 3. Jaccard Distance
- The new distance between _p_ and $g_i$ can be calculated by the Jaccard metric of their _k_-reciprocal sets as E.q 5:

$$d_{J}(p, g_{i}) = 1-\frac{\left|\mathcal{R}^{*}(p, k) \cap \mathcal{R}^{*} g_{i}, k\right)|}{|\mathcal{R}^{*}(p, k) \cup \mathcal{R}^{*}(g_{i}, k)|}$$

  where |.| denote the number of candidates in the set.
- by encoding the _k_-reciprocal nearest neighbor set into a vector $V_p = [V_{p,g_1},...,V_{p, g_N}]$ as Eq.6:
  
$$\mathcal{V}_{p, g_{i}}= \begin{cases}1 & \text { if } g_{i} \in \mathcal{R}^{*}(p, k) \\ 0 & \text { otherwise }\end{cases}$$

  then, the k-reciprocal neighbor set can be represented as an N-dimensional vector

-  we redefine Eq. 6 by the Gaussian kernel of the pairwise distance as

$$\mathcal{V}_{p, g_{i}} = \begin{cases}\mathrm{e}^{-d(p, g_{i})} & \text { if } g_{i} \in \mathcal{R}^{*}(p, k) \\ 
 0 & \text{otherwise.}\end{cases}$$

- Based on the above definition, the number of candidates in the intersection and union set can be calculated as

$$ |\mathcal{R}^{*}(p, k) \cap \mathcal{R}^{*}(g_{i}, k)| = \|\min (\mathcal{V}_{p}, \mathcal{V}_{g_{i}})\|_{1} $$

$$ |\mathcal{R}^{*}(p, k) \cup \mathcal{R}^{*}(g_{i}, k)| = \|\max (\mathcal{V}_{p}, \mathcal{V}_{g_{i}})\|_{1} $$

- The Jaccard distance in Eq. 5 can rewrite as

$$ d_{J}(p, g_{i}) = 1 -\frac{\sum_{j=1}^{N} \min (\mathcal{V}_{p, g_{j}}, \mathcal{V}_{g_{i}, g_{j}})}{\sum_{j=1}^{N} \max (\mathcal{V}_{p, g_{j}}, \mathcal{V}_{g_{i}, g_{j}})} $$


# 4. Local Query Expansion
> Emulating the idea that the images from the same class may share similar features, we use the $k$-nearest neighbors of the probe $p$ to implement the local query expansion. 
- The local query expansion is defined as
$$ \mathcal{V}_{p}=\frac{1}{|N(p, k)|} \sum_{g_{i} \in N(p, k)} \mathcal{V}_{g_{i}} $$
- LQE expanded by the _k_-nearest neighbors of probe _p_. 
and implement both on the probe _p_ and galleries $g_{i}$. 
- In order to distinguish between the size of $\mathcal{R}^{*}\left(g_{i}, k\right)$ and $N(p, k)$ used in before and this, we denote the former as $k_{1}$ and the latter as $k_{2}$, respectively, where $k_{1}>k_{2}$.

# 5. Final Distance

$$d^{*}\left(p, g_{i}\right)=(1-\lambda) d_{J}\left(p, g_{i}\right)+\lambda d\left(p, g_{i}\right)$$

# 6. Complexity Analysis
most of the computation costs focus on pairwise distance computing for all gallery pairs. If the size of the gallery set is $\mathcal{N}$: 
   - the computation complexity for the distance measure: $\mathcal{O}(N^2)$
   - the computation complexity  raking process: $\mathcal{O}(N^2 log N)$

If ranking lists and pairwise distance are advance offline calcuate: 
   - the computation complexity for the distance measure: $\mathcal{O}(N)$
   - the computation complexity  raking process: $\mathcal{O}(N log N)$ 
