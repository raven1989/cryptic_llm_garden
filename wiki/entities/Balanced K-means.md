---
tags:
  - machine-learning
  - clustering
  - quantization
  - recommendation
aliases:
  - Balanced K-means
  - Balanced K-means Clustering
date: 2026-06-05
sources: ["[[wiki/research/OneRec Summary.md]]"]
---

# Balanced K-means Clustering

**Balanced K-means Clustering** is an optimized, cardinality-constrained variant of the traditional $K$-means algorithm. It partitions a continuous vector space into $K$ disjoint clusters while forcing **all clusters to have the exact same number of items ($w$)**:
$$|\mathcal{V}_k| = w = \frac{|\mathcal{V}|}{K}$$
Where $\mathcal{V}$ is the total dataset and $K$ is the target number of clusters.

In recommendation systems, it is used as an alternative to Residual Quantization (RQ-VAE) to convert continuous video embeddings into highly balanced discrete **semantic IDs**, completely solving the **Hourglass Phenomenon (Codebook Collapse)** where a small set of tokens represents the vast majority of items.

---

## The Hourglass Phenomenon vs. Balanced K-means

Standard vector quantization methods (like VQ-VAE or RQ-VAE) partition high-dimensional vector spaces purely based on geographic proximity:
$$\mathcal{V}_k = \{ \mathbf{r} \in \mathcal{V} \mid k = \arg\min_j \| \mathbf{r} - \mathbf{c}_j \|_2^2 \}$$

Because high-dimensional multimodal spaces are highly non-uniform (data is dense in some areas and extremely sparse in others), standard clustering results in:
* **Over-congested centroids**: A few centroids represent a vast number of items, leading to high collision rates.
* **Dead centroids**: Many centroids represent zero or near-zero items, rendering those codebook tokens useless.

This is known as the **hourglass phenomenon**. 

By introducing the hard constraint $|\mathcal{V}_k| = w$, **Balanced K-means** guarantees that the vocabulary is perfectly balanced, making autoregressive next-token logits extremely stable and preventing model collapse during sequence training.

---

## Algorithm Formulation

To satisfy the strict cardinality constraint while keeping computation fast enough to process millions of items, a greedy, sequential-priority assignment is used:

```text
================================================================================
Algorithm: Balanced K-means Clustering
================================================================================
Input: Item set V, number of clusters K
Output: Optimized codebook C_l = {c_1, ..., c_K}

1. Compute capacity threshold: w = |V| / K
2. Initialize centroids C_l = {c_1, ..., c_K} using random selection
3. Repeat
4.     Initialize unassigned item pool: U = V
5.     For each cluster k in {1, ..., K} Do:
6.         Compute Euclidean distance between all remaining items in U and centroid c_k
7.         Sort U in ascending order of distance
8.         Assign V_k = U[0 : w - 1] (assign the closest w remaining items to cluster k)
9.         Update centroid: c_k = (1 / w) * Sum_{r in V_k} (r)
10.        Remove assigned items from pool: U = U \ V_k
11.    End
12. Until assignment convergence
================================================================================
```

---

## Mathematical Properties and Trade-offs

### 1. Distance Distortion vs. Uniform Distribution
Because assignments are made sequentially from $k=1$ to $K$, an item assigned to cluster $k$ is **not guaranteed to have centroid $\mathbf{c}_k$ as its absolute closest centroid** globally. 
* Cluster $1$ gets first pick of its closest items.
* Cluster $K$ is forced to accept whatever items are left in $\mathcal{U}$, regardless of their distance to $\mathbf{c}_K$.

This represents a classic mathematical compromise: **We sacrifice local spatial proximity (Voronoi cell accuracy) to achieve a globally uniform token distribution.**

### 2. Centroid Migration and Convergence
The outer `repeat...until convergence` loop acts as a self-correcting force. In iteration $t$, if Cluster $K$ is assigned far-away leftovers, the centroid update step:
$$\mathbf{c}_K \leftarrow \frac{1}{w} \sum_{\mathbf{r} \in \mathcal{V}_K} \mathbf{r}$$
pulls centroid $\mathbf{c}_K$ directly toward the spatial region of those leftovers. In the next iteration $t+1$, the centroids have migrated to better fit the global item density, and the sequential assignment pass achieves much higher spatial cohesion.

### 3. Complexity
Finding the absolute mathematically optimal equal-cardinality partitioning under a distance-minimization objective is a **Linear Sum Assignment Problem (LSAP)**, which has a complexity of $\mathcal{O}(N^3)$ (or $\mathcal{O}(N^2)$ for fast approximations). 

Balanced K-means runs in **$\mathcal{O}(I \cdot K \cdot N \log N)$** (where $I$ is iterations, $K$ is codebook size, and sorting $\mathcal{U}$ takes $N \log N$). This makes it highly scalable and production-viable for corpora containing **tens of millions of active items**.

---

## Hierarchical Residual Quantization in OneRec

In [[OneRec Summary|OneRec]], Balanced K-means is applied **hierarchically ($L=3$ layers)** on behavior-aligned video embeddings $\mathbf{e}_i$:

1. **Layer 1**: Run Balanced K-means on raw embeddings $\mathbf{r}^1_i = \mathbf{e}_i$ with codebook size $K = 8192$. Select token $s^1_i$. Compute residual:
   $$\mathbf{r}^2_i = \mathbf{r}^1_i - \mathbf{c}^1_{s^1_i}$$
2. **Layer 2**: Run Balanced K-means on residuals $\mathbf{r}^2_i$. Select token $s^2_i$. Compute next residual:
   $$\mathbf{r}^3_i = \mathbf{r}^2_i - \mathbf{c}^2_{s^2_i}$$
3. **Layer 3**: Run Balanced K-means on residuals $\mathbf{r}^3_i$. Select token $s^3_i$.

The final generated semantic ID is the token sequence $\langle s^1_i \rangle \langle s^2_i \rangle \langle s^3_i \rangle$. This representation provides a coarse-to-fine description of the video, which is highly suited for autoregressive decoding.

---

## Related Wiki Pages
* [[OneRec Summary]]: The main research summary page for the Kuaishou paper.
* [[OneRec]]: High-level concept page for the OneRec architecture.
* [[Transformers]]: Foundation model architectures.
