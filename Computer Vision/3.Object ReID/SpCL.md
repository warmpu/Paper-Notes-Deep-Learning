# Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID,
Yixiao Ge, Feng Zhu, Dapeng Chen, Rui Zhao, Hongsheng Li. _13 Oct 2020(v2)_

>Although state-of-the-art pseudo-label-based methods have achieved great success, they did not make full use of all valuable information because of the domain gap and unsatisfying clustering performance. To solve these problems, we propose a novel self-paced contrastive learning framework with hybrid memory.The hybrid memory dynamically generates source-domain class-level, target-domain cluster-level and un-clustered instance-level supervisory signals for learning feature representations. Different from the conventional contrastive learning strategy, the proposed framework jointly distinguishes source-domain classes, and target-domain clusters and un-clustered instances. Most importantly, the proposed self-paced method gradually creates more reliable clusters to refine the hybrid memory and learning targets, and is shown to be the key to our outstanding performance.

* Official paper: [ArXiv](https://arxiv.org/abs/2006.02713)
* Official code: [Github](https://github.com/yxgeee/SpCL)

# Overview

Although the pseudo-label-based methods have led to great performance advances, we argue that there exist two major limitations that hinder their further improvements.

![fig1](../../asset/images/P_ReId/SpCL/fig1.jpg)


- During the target-domain fine-tuning, the source-domain images were either not considered or were even found harmful to the final performance.
- The accurate source-domain ground-truth labels are valuable but were ignored during target-domain training
- the clustering process might result in individual outliers, thus pseudo labels might be unreliable.

> To overcome the problems, we propose a hybrid memory to encode all available information from both source and target domains for feature learning.

- Review Structure:
  1. Relate work
  2. Methodology
  3. Ablation studies

# 1.Relate work

1. ***Contrastive learning***
   - a contrastive loss was adopted to learn instance discriminative representations by treating each unlabeled sample as a distinct class. 
   - Although the instance-level contrastive loss could be used to train embeddings that can be generalized well to
downstream tasks with fine-tuning, it does not perform well on the domain adaptive object re-ID tasks which require to correctly measure the inter-class affinities on the unsupervised target domain.


2. ***Self-paced learning***
   - The “easy-to-hard” training scheme is core of self-paced learning, which was originally found effective in supervised learning methods, especially with noisy labels

# 2.Methodology

Our training scheme alternates between two steps: 
 - grouping the target-domain samples into clusters and un-clustered instances by clustering the target-domain instance features in the hybrid memory with the self-paced strategy [Section 2.2](#22-self-paced-learning-with-reliable-clusters)
 -  optimizing the encoder fθ with a unified contrastive loss and dynamically updating the hybrid memory with encoded features [Section 2.1](#21--constructing-and-updating-hybrid-memory-for-contrastive-learning)

![fig2](../../asset/images/P_ReId/SpCL/fig2.jpg)

## 2.1 Constructing and Updating Hybrid Memory for Contrastive Learning
Given:
   - the target-domain training samples $\mathbb{X}^{t}$ without any ground-truth label
   - we employ the selfpaced clustering strategy [Section 2.2](#22-self-paced-learning-with-reliable-clusters) to group the samples into clusters and the un-clustered outliers. The whole training set of both domains can therefore be divided into three parts:
     - the source-domain samples $\mathbb{X}^{s}$ with ground-truth identity labels
     - the target-domain pseudo-labeled data $\mathbb{X}_{c}^{t}$ within clusters 
     - the target-domain instances $\mathbb{X}_{o}^{t}$ not belonging to any cluster
     - $X^t = X_c^t \cup X_o^t$. 
  
   - We design a novel contrastive loss to fully exploit available data by treating all the source-domain classes, target-domain clusters and target-domain un-clustered instances as independent classes.

- **Unified Contrastive Learning**
  
Given a general feature vector $\boldsymbol{f}=f_{\theta}(x), x \in \mathbb{X}^{s} \cup \mathbb{X}_{c}^{t} \cup \mathbb{X}_{o}^{t}$ , our unified contrastive loss is

$$\mathcal{L}_{\boldsymbol{f}}=-\log \frac{\exp \left(\left\langle\boldsymbol{f}, \boldsymbol{z}^{+}\right\rangle / \tau\right)}{\sum_{k=1}^{n^{s}} \exp \left(\left\langle\boldsymbol{f}, \boldsymbol{w}_{k}\right\rangle / \tau\right)+\sum_{k=1}^{n_{c}^{t}} \exp \left(\left\langle\boldsymbol{f}, \boldsymbol{c}_{k}\right\rangle / \tau\right)+\sum_{k=1}^{n_{o}^{t}} \exp \left(\left\langle\boldsymbol{f}, \boldsymbol{v}_{k}\right\rangle / \tau\right)} \quad, \quad (1)$$

  -   $z^+$ indicats the positive class protype corresponding to ***f***
  -   `<.,.>` denotes the inner product between 2 vectors to measure their similarity.
  -   $n^s, n_c^t, n_o^t$ is numberr of source-domain classes, target-domain clusters, target-domain un-clustered instances respectively.
  -   if $\mathcal{f}$ is:
      -   **a source-domain feature**, $z^+ = w_k$ is the centroid of the centroid of the cource-domain class that **f** belongs to.
      -   **belongs to the k-th target-domain cluster**, $z^+ = c_k$ is the _k-th_ cluster centroid.
      -   **a target-domain un-clustered outlier**, $z^+ = v_k$ as the outlier instance feature corresponding to **f** 
  -  this loss encourages the encoded feature vector to approach its assigned classes, clusters or instances. 

- **Hybrid Memory**

  -  the cluster number and outlier instance $n_c^t  \text{and}  n_o^t$ may change during training.
  -  a novel hybrid memory to provide source class centroids $w_i$, target-domain clusters $c_{i_c^t}$, target-domain un-clustered instance features $v_{i_o^t}$
  -  For continuously storing and updating the above three types of entries, they **cache** :
     -  source-domain class centroids
     -  all the target-domain instance features $v_1, ..., v_n^t \qquad \text{where} \qquad n^t <> n_c^t + n_o^t$ is the number of all the target-domain instances
  
- **Memory initialization**

The hybrid memory is initialized with the extracted features by performing forward computation of $f_{\theta}$:
  - source-domain class centroids $\{\boldsymbol{w}\}$ can be obtained as the mean feature vectors of each class
  - target-domain instance features $\{\boldsymbol{v}\}$ are directly encoded 
    - After that, the target-domain cluster centroids $\{\boldsymbol{c}\}$ are initialized with the mean feature vectors of each cluster from $\{\boldsymbol{v}\}$

_Example Code_
```python
    # Create hybrid memory
    memory = HybridMemory(model.module.num_features, len(dataset.train),
                            temp=args.temp, momentum=args.momentum).cuda()

    # Initialize target-domain instance features
    print("==> Initialize instance features in the hybrid memory")
    cluster_loader = get_test_loader(dataset, args.height, args.width,
                                    args.batch_size, args.workers, testset=sorted(dataset.train))
    features, _ = extract_features(model, cluster_loader, print_freq=50)
    features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
    memory.features = F.normalize(features, dim=1).cuda()
    del cluster_loader, features
```

**Memory update**
At each iteration, the encoded feature vectors in each mini-batch would be involved in hybrid memory updating
  - For the source-domain class centroids :
  
    $$\boldsymbol{w}_{k} \leftarrow m^{s} \boldsymbol{w}_{k}+\left(1-m^{s}\right) \cdot \frac{1}{\left|\mathcal{B}_{k}\right|} \sum_{\boldsymbol{f}_{i}^{s} \in \mathcal{B}_{k}} \boldsymbol{f}_{i}^{s}$$

    where:
    - $\mathcal{B}_{k}$ denotes the _feature set_ belonging to source-domain class ***k*** in the current mini-batch 
    - $m^{s} \in[0,1]$ is a momentum coefficient for updating source-domain class centroids. 
    - $m^{s} = 0.2$.

  - For the target-domain cluster centroids, as the hybrid memory caches all the target-domain features $\{\boldsymbol{v}\}$
    -  each encoded feature vector $\boldsymbol{f}_{i}^{t}$ in the mini-batch is utilized to update its corresponding instance entry $\boldsymbol{v}_{i}$ by:
  
    $$\boldsymbol{v}_{i} \leftarrow m^{t} \boldsymbol{v}_{i}+\left(1-m^{t}\right) \boldsymbol{f}_{i}^{t}$$

    _Code Example_

    ```python
    class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


    def hm(inputs, indexes, features, momentum=0.5):
        return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))
    ```

    In hybridMeory class:

    ```python
    class HybridMemory(nn.Module):
        def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2)
            ...
        def forward(self, inputs, indexes):
            # inputs: B*2048, features: L*2048
            inputs = hm(inputs, indexes, self.features, self.momentum)
            ...
    ```


## 2.2 Self-paced Learning with Reliable Clusters

> A simple way to split the target-domain data into clusters and un-clustered outliers is to cluster the target-domain instance features {v1, · · · , vnt } from the hybrid memory by a certain algorithm (e.g., DBSCAN)
> If the clustering is perfect, merging all the instances into their true clusters would no doubt improve the final performance. 
> A reliability criterion is proposed to identify unreliable clusters by measuring the independence and compactness

**Independence of clusters.**
- A reliable cluster should be independent from other clusters and individual samples.
  - if a cluster is far away from other samples, it can be considered as highly independent
  - However, due to the uneven density in the latent space, we cannot naïvely use the distances between the cluster centroid and outside-cluster samples to measure the cluster independence.
> We propose the following metric to measure the cluster independence, which is formulated as an intersection-over-union (IoU) score as

$$\mathcal{R}_{\text {indep }}\left(\boldsymbol{f}_{i}^{t}\right)=\frac{\left|\mathcal{I}\left(\boldsymbol{f}_{i}^{t}\right) \cap \mathcal{I}_{\text {loose }}\left(\boldsymbol{f}_{i}^{t}\right)\right|}{\left|\mathcal{I}\left(\boldsymbol{f}_{i}^{t}\right) \cup \mathcal{I}_{\text {loose }}\left(\boldsymbol{f}_{i}^{t}\right)\right|} \in[0,1]$$

where $\mathcal{I}_{\text {loose }}\left(\boldsymbol{f}_{i}^{t}\right)$ is the cluster set 

- Larger $\mathcal{R}_{\text {indep }}\left(\boldsymbol{f}_{i}^{t}\right)$ indicates a more independent cluster for _f(i)_, 
- Even one looses the clustering criterion, there would be no more sample to be included into the new cluster $\mathcal{I}_{\text {loose }}\left(\boldsymbol{f}_{i}^{t}\right)$. 
- Samples within the same cluster set (e.g., $\mathcal{I}\left(\boldsymbol{f}_{i}^{t}\right)$ ) generally have the same independence score.

**Compactness of clusters.**
> A reliable cluster should also be compact,  the samples within the same cluster should have small inter-sample distances
> when a cluster is most compact, all the samples in the cluster have zero inter-sample distances. Its samples would not be split into different clusters even when the clustering criterion is tightened.

$$\mathcal{R}_{\text {comp }}\left(\boldsymbol{f}_{i}^{t}\right)=\frac{\left|\mathcal{I}\left(\boldsymbol{f}_{i}^{t}\right) \cap \mathcal{I}_{\text {tight }}\left(\boldsymbol{f}_{i}^{t}\right)\right|}{\left|\mathcal{I}\left(\boldsymbol{f}_{i}^{t}\right) \cup \mathcal{I}_{\text {tight }}\left(\boldsymbol{f}_{i}^{t}\right)\right|} \in[0,1]$$

where $\mathcal{I}_{\text {tight }}\left(\boldsymbol{f}_{i}^{t}\right)$ is the cluster set 

- Larger $\mathcal{R}_{\text {comp }}\left(\boldsymbol{f}_{i}^{t}\right)$ indicates smaller inter-sample distances around _f(i)_ 
within $\mathcal{I}\left(\boldsymbol{f}_{i}^{t}\right)$, since a cluster with larger intersample distances is more likely to include fewer points when a tightened criterion is adopted.
- The same cluster's data points may have different compactness scores due to the uneven density.
- we preserve independent clusters with compact data points whose $\mathcal{R}_{\text {indep }}>\alpha$ and $\mathcal{R}_{\text {comp }}>\beta$, while the remaining data are treated as un-clustered outlier instances.

**Code example**

```python
# compute R_indep and R_comp
        N = pseudo_labels.size(0)
        label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
        label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
        label_sim_loose = pseudo_labels_loose.expand(N, N).eq(pseudo_labels_loose.expand(N, N).t()).float()

        R_comp = 1-torch.min(label_sim, label_sim_tight).sum(-1)/torch.max(label_sim, label_sim_tight).sum(-1)
        R_indep = 1-torch.min(label_sim, label_sim_loose).sum(-1)/torch.max(label_sim, label_sim_loose).sum(-1)
        assert((R_comp.min()>=0) and (R_comp.max()<=1))
        assert((R_indep.min()>=0) and (R_indep.max()<=1))

        cluster_R_comp, cluster_R_indep = collections.defaultdict(list), collections.defaultdict(list)
        cluster_img_num = collections.defaultdict(int)
        for i, (comp, indep, label) in enumerate(zip(R_comp, R_indep, pseudo_labels)):
            cluster_R_comp[label.item()].append(comp.item())
            cluster_R_indep[label.item()].append(indep.item())
            cluster_img_num[label.item()]+=1

        cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
        cluster_R_indep = [min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())]
        cluster_R_indep_noins = [iou for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys())) if cluster_img_num[num]>1]
        if (epoch==0):
            indep_thres = np.sort(cluster_R_indep_noins)[min(len(cluster_R_indep_noins)-1,np.round(len(cluster_R_indep_noins)*0.9).astype('int'))]

        pseudo_labeled_dataset = []
        outliers = 0
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            indep_score = cluster_R_indep[label.item()]
            comp_score = R_comp[i]
            if ((indep_score<=indep_thres) and (comp_score.item()<=cluster_R_comp[label.item()])):
                pseudo_labeled_dataset.append((fname,label.item(),cid))
            else:
                pseudo_labeled_dataset.append((fname,len(cluster_R_indep)+outliers,cid))
                pseudo_labels[i] = len(cluster_R_indep)+outliers
                outliers+=1
```

![fig3](../../asset/images/P_ReId/SpCL/fig3.jpg)


# 3. Ablation studies

- 3.1 Network Optimization & Training Data Organization
  - Backbone: an ImageNet-pretrained ResNet-50 up to the global average pooling layer, followed by a 1D BatchNorm layer and an L2-normalization layer
  - Adam optimizer is adopted to optimize fθ with a weight decay of 0.0005
  - initial learning rate is set to 0.00035, decayed 1/10 every 20 epochs in the total 50 epochs
  -  The temperature = 0.5
  -  each mini-batch contains 64 source-domain images of 16 ground-truth classe
- 3.2 Target-domain Clustering
  -  using DBSCAN, the maximum distance between neighbors is set as d = 0.6
> In our proposed self-paced learning strategy described in Section 3.2, we tune the value of d to loosen or tighten the clustering criterion. Specifically, we adopt d = 0.62 to form the looser criterion and d = 0.58 for the tighter criterion
  -  Jaccard distance(k=30)
