# Part-based Pseudo Label Refinement for Unsupervised Person Re-identification
Yoonki Cho, Woo Jae Kim, Seunghoon Hong, Sung-Eui Yoon. _28 Mar 2022_
>  In this paper, we propose a novel Part-based Pseudo Label Refinement (PPLR) framework that reduces the label noise by employing the complementary relationship between global and part features. Specifically, we design a cross agreement score as the similarity of k-nearest neighbors between feature spaces to exploit the reliable complementary relationship. 
> Based on the cross agreement, we refine pseudo-labels of global features by ensembling the predictions of part features, which collectively alleviate the noise in global feature clustering. We further refine pseudo-labels of part features by applying label smoothing according to the suitability of given labels for each part. Thanks to the reliable complementary information provided by the cross agreement score, our PPLR effectively reduces the influence of noisy labels and learns discriminative representations with rich local contexts. 

* Official paper: [arXiv](https://arxiv.org/abs/2203.14675)
* Official code: [Github](https://github.com/yoonkicho/pplr)

## Overview

1. Related work
2. Architecture
3. Ablation Study

## 1. Related work
- Learning with noisy labels
  - Loss adjustment approaches to design a robust loss function against the label noise as mean absolute error (MAE) loss, Generalized crossentropy (GCE), symmetric crossentropy (SCE) loss
  - the sample re-weighting scheme based on the reliability of a given label
  -  These loss functions, however, are designed for simple image classification tasks and are not suitable for open-set person re-ID tasks.
  
- Part-based approaches for person re-ID
  - Fine-grained information on human body parts( human parsing, pose estimation, attention mechanism) is an essential clue to distinguish people
  - Recent methods  utilize part features to exploit robust feature similarity for accurate pseudo-labels.   

- Unsupervised approaches for person re-ID
  -  unsupervised domain adaptation (UDA)
  -  unsupervised learning depending on whether an external labeled source domain data is used
  -  Clustering-based methods apply the contrastive learning scheme with cluster proxies

## 2. Architecture
There are 2 stages: 
   -  the clustering stage: extract global and part features and assign pseudo-labels through global feature clustering. Then compute a cross agreement score for each sample based on the similarity between k-nearest neighbors of global and part features
   -  the training stage  : mitigate the label noise using the proposed pseudo-label refinement methods based on the cross agreement( agreement-aware label smoothing for part feature, part-guided label refinement for global feature)
  
![fig](../../asset/images/P_ReId/PPLR/fig1.png#center)

### 2.1 Part-based Unsupervised re-ID Framework
> present a part-based unsupervised person reID framework that utilizes fine-grained information of the part features.  Use both the global and part features to represent an image.

- step 1: model first extracts the _shared representation_  $F_{\theta}(x_i) \in R^{C*H*W}$ where _C, H, W_ are channel, height, width of the **feature map**. 
  - Using GAP over feature map ==> **Global feature** $f_i^g$
  - Dividing horizontally(H' direction) to _N parts_ and applying AP ==> **local features**(part features) $\{f_i^{p_n}\}_{n=1}^{N_p}$ with shape $R^{C*\frac{H}{N_p}*W}$

- step 2: Generating the pseudo-labels based on clustering results. 
  - Perform DBSCAN clustering on the globel feature set $\{f_i^g\}_{i=1}^N$ with _N is number samples_.
  - use the cluster assignment as pseudo-labels => $(x_i, y_i)_{i=1}^N | y_i \in R^K$ where K is the number of clusters.
- step 3: Compute loss funtion:
  - global cross-entropy loss: $L_{gce} = - \sum_{i=1}^N y_i . log(q_i^g)$
  
  where $q_i^g = h_{\phi_g}(f_i^g) \in R^K$ is the prediction vector by global feature, and _h_ is the global feature classifier.
  - local(part) cross entropy: $L_{pce} = - \frac{1}{N_p} \sum_{i=1}^N \sum_{n=1}^{N_p} y_i . log(q_i^{p_n})$
    where $q_i^{p_n} = h_{\phi_g}(f_i^{p_n}) \in R^K$ is the prediction vector by _n_-th part feature space $p_n$, and _h_ is the classifier for the part feature space. 
  - softmax-triplet loss:
  $$L_{softTriplet} =  - \sum_{i=1}^N log (\frac{e^{|f_i^g - f_{i,n}^g|}}{e^{|f_i^g - f_{i,p}^g|} + e^{|f_i^g - f_{i,n}^g|}})$$
  where |.| denotes the L2-norm, the subcripts _(i,p)_ and _(i,n)_ respectively the hardest positive and negative samples of the image $x_i$ in mini-batch.
  - *Optional loss*: camera-aware proxy to improve the discriminability across camera views. This loss attemp _pull_ together the proxies are within the same cluster but in different cameras, _reduce_ the intra-class variance caused by disjoint camera views.
    - compute the camrera-aware proxy $c_{a,b}$ as the cenntroid of the features that have _same camera label_ **a** and _same cluter(plabel)_ **b**

    $$c_{(a,b)} = \frac{1}{|S_{a,b}|} \sum_{i \in S_{a,b}} f_i$$
    where $S_{a,b} = {i|c_i = a \cap y_i = b}$ is the index set for the proxy $c_{(a,b)}$

    - with $P_i, Q_i$ are the index sets of the positive and hard negative camera-aware proxies for $f_i^g$, the inter-camera contrastive loss as:
    $$L_{cam} = - \sum_{i=1}^N \frac{1}{|P_i|} \sum_{j \in P_i} log \frac{exp(\frac{c_j^{\tau} f_i^g }{\tau})}{\sum_{k \in P_i \cup Q_i} exp (\frac{c_k^{\tau} f_i^g}{\tau})}$$ 

      - $P_i$ defined as the proxy indices that have the same pseudo-label but differnet camera labels with $f_i$
      - $Q_i$ for the hard negative proxies of the feature $f_i$ is defined as the indices of nearest proxies that have different pseudo-labels to $y_i$

  **Final objective**
  $$L = L_{gce} + L_{pce} + L_{softTriplet} + \lambda L_{cam}$$


### 2.2 Cross Agreement
> a cross agreement score that captures how reciprocally similar the k-nearest neighbors of global and part features are for refining pseudo-labels of global features

  - the cross agreement score is defined as the Jaccard similarity between the knearest neighbors of the global and part features.

  ![fig2](../../asset/images/P_ReId/PPLR/fig2.png#center)


**STEPS**
  - to perform a KNN search on the gobal and each local feature spaces independently to produce $(1+N_p)$ ranked lists on each image.
  - Compute the cross agreement score between the global feature space _g_ and the _n_-th parth feature space $p_n$ for each _i_-th image by:
    $$C_i(g, p_n) = \frac{R_i(g,k) \cup R_i(p_n,k)}{|R_i(g,k) \cap R_i(p_n,k|} \in [0,1]$$
  where $R_i(g,k) ; R_i(p_n,k)$ are the sets of the indices for top-k samples in the ranked list.
  - higl cross agreement implies the speudo label have high reliable complementary information.

### 2.3 Pseudo Label Refinement
Based on the cross agreement scores, we alleviate the pseudo label noise by considering:
* whether the pseudo-labels by global feature clustering are suitable for each part feature
* whether the predictions of part features are appropriate for refining pseudo-labels of global features

- **Agreement-aware label smoothing - AALS**
> we utilize a label smoothing to refine the pseudo-label of each part depending on the corresponding cross agreement score.

  - The label smoothing for the part feature $f_i^{p_n}$ are formulated as:
  $$\overline{y_i^{p_n}} = \alpha_i^{p_n} y_i + (1-\alpha_i^{p_n}) u$$
  where _u_ is a uniform vector (zeros vector), and $\alpha_i^{p_n}$ is a weight determing the strength of label smoothing and dynamically adjuted for each part(local) features using the cross agreement score.

  - $L_{pce}$ in final objective is reformulated with Kullback leiler divergence by:
      $$L_{aals} = \frac{1}{N_p} \sum_{i=1}^N \sum_{n=1}^{N_p} (\alpha_i^{p_n} H(y_i, q_i^{p_n})) + (1-\alpha_i^{p_n}) D_{KL}(u || q_i^{p_n})$$
  with **H(.,.)** and **D(.||.)** are cross-entropy and KL divergence.

  - code example:
    ```python
    class AALS(nn.Module):
    """ Agreement-aware label smoothing """
    def __init__(self):
        super(AALS, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
    def forward(self, logits, targets, ca):
        log_preds = self.logsoftmax(logits)  # B * C
        targets = torch.zeros_like(log_preds).scatter_(1, targets.unsqueeze(1), 1)
        uni = (torch.ones_like(log_preds) / log_preds.size(-1))
        loss_ce = (- targets * log_preds).sum(1)
        loss_kld = F.kl_div(log_preds, uni,reduction='none').sum(1)
        return (ca * loss_ce + (1-ca) * loss_kld).mean()
    ```

- **Part-guided label refinement - PGLR**  
> To generates refined labels for global features using the predictions by part features.
> Since less discriminative parts can provide misleading information, we aggregate the predictions of part features with different weights depending on each cross agreement score, thus refining the labels with more reliable information.

  - The label smoothing for the global feature $f_i^{p_n}$ are formulated as:
  $$\overline{y_i^g} = \beta y_i + (1-\beta) \sum_{n=1}^{N_p} w_i^{p_n} q_i^{p_n}$$
  
  subject to
  $$w_i^{p_n} = \frac{exp(C_i(g, p_n))}{\sum_k exp(C_i(g,p_k))}$$
  where _w_ and _q_ are the ensemble weight and the prediction vector of the part feature $f_i^{p_n}$, respectively.
  - Then, the refined labels $\overline{y_i^g}$ are plugged the $L_{gce}$ by:
    $$L_{pglr} = - \sum_{i=1}^N \overline{y_i^g} . log(q_i^g)$$

    ```python
    class PGLR(nn.Module):
    """ Part-guided label refinement """
    def __init__(self, lam=0.5):
        super(PGLR, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.lam = lam

    def forward(self, logits_g, logits_p, targets, ca):
        targets = torch.zeros_like(logits_g).scatter_(1, targets.unsqueeze(1), 1)
        w = torch.softmax(ca, dim=1)  # B * P
        w = torch.unsqueeze(w, 1)  # B * 1 * P
        preds_p = self.softmax(logits_p)  # B * C * P
        ensembled_preds = (preds_p * w).sum(2).detach()  # B * class_num
        refined_targets = self.lam * targets + (1-self.lam) * ensembled_preds

        log_preds_g = self.logsoftmax(logits_g)
        loss = (-refined_targets * log_preds_g).sum(1).mean()
        return loss
    ```
    

- **Overall training objective.** 
  $$L_{PPLR} = L_{aals} + L_{pglr} + L_{softTriplet} + \lambda L_{cam}$$

  If $\lambda=0$, training process will exclude camera loss.



## 3. Ablation Study
  
**Parameter analysis**

  - Large k values result in more frequent false matches in top-k ranked lists of global and part features, producing lower cross agreement scores overall.
  - When we set β to 0, our method decomposes down to using only the ensembled part predictions showing the significant performance drop
  - Based on these experimental results, we set _k = 20_ and $\beta = 0.5$
