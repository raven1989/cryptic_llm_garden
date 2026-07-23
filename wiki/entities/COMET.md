---
tags: ["entity", "architecture", "time-series"]
aliases: ["COMET"]
date: 2026-07-23
sources: ["[[wiki/research/COMET Summary.md]]"]
---

# COMET

**COMET** (Codebook-based Online-adaptive Multi-scale Embedding) is an unsupervised time-series anomaly detection framework. It combines multi-resolution temporal features, codebook discrete representation learning, density-adaptive scoring, and contrastive test-time adaptation to solve non-stationarity and context aggregation issues in industrial multivariate time series.

## Core Architectural Pillars

### 1. Multi-Scale Patch Encoding
Instead of single-resolution or channel-independent embedding, COMET captures both local temporal patterns and cross-variable correlations:
* **Multi-Scale Patching**: Segments the inputs across $K$ scales with various patch sizes $\mathcal{P} = \{p_1, \dots, p_K\}$ and strides $\mathcal{S} = \{s_1, \dots, s_K\}$.
* **SeriesPatch-Specific Path**: Passes individual variable patches through a specialized linear layer:
  $$\mathbf{h}_{k}^{(s,i,j)} = \mathbf{W}_{k}^{(s,i)}\mathbf{P}_{k}^{(i,j)} + \mathbf{b}_{k}^{(s,i)}$$
* **Core Path (Shared)**: Concatenates all variable patches at interval $j$ to capture shared inter-variable relationships:
  $$\mathbf{h}_{k}^{(c,j)} = \mathbf{W}_{k}^{(c)}\tilde{\mathbf{P}}_{k}^{(j)} + \mathbf{b}_{k}^{(c)}$$
* **Fusion Layer**: Linearly merges the local and global perspectives into a cohesive embedding:
  $$\mathbf{z}_{e,k}^{(i,j)} = \mathbf{W}_{k}^{(g)}\left[\mathbf{h}_{k}^{(s,i,j)}; \mathbf{h}_{k}^{(c,j)}\right] + \mathbf{b}_{k}^{(g)}$$

### 2. Vector-Quantized Coreset
During training on normal-only datasets, COMET trains a codebook $\mathbf{C}_k$ for each scale to learn prototypical normal waveforms:
* **Quantization**: Maps continuous representations $\mathbf{z}_{e,k}^{(i,j)}$ to their nearest codebook index:
  $$\mathbf{z}_{q,k}^{(i,j)} = \mathbf{c}_{q_{k}^{(i,j)}}$$
* **Commitment & Codebook Losses**: Backpropagates gradients through VQ constraints using a combination of MSE reconstruction, codebook, and commitment objectives.
* **Coreset Construction**: Gathers all codebook vectors activated during normal-data training to act as a compact normal-behavior memory bank:
  $$\mathcal{M} = \bigcup_{k=1}^K \left\{ \mathbf{c}_m \mid m \in \mathcal{A}_k \right\}$$

### 3. Density-Adaptive scoring
To calculate accurate anomaly metrics in non-uniform distribution spaces, COMET utilizes local scale normalization:
* **Local Scaling Distance**: Measures distance normalized by local neighborhood densities (k-NN proxy metrics $\sigma_q$ and $\sigma_j$):
  $$d_{\text{local}}(\mathbf{z}_q, \mathbf{m}_j) = \frac{\|\mathbf{z}_q - \mathbf{m}_j\|_2^2}{(\sigma_q + \sigma_j)/2 + \epsilon}$$
* **Quantization Error**: Serves as a complementary score measuring distance to codebook space, capturing compressed normal-representation capabilities.

### 4. Deviation-Based Variable Selection
Filters noisy or redundant variables from aggregating final anomaly metrics:
* Computes standardized deviations $\delta_t^{(i)}$ for each variable.
* Retains only variables within stable thresholds, avoiding anomaly-signal dilution by high-variance irrelevant variables:
  $$\tilde{S}_t = \frac{1}{|\mathcal{V}_t|}\sum_{i \in \mathcal{V}_t}S_t^{(i)}$$

### 5. Contrastive Online Codebook Adaptation (TTA)
Dynamically adapts the model at inference time to handle distribution shifts without labels:
* **Pseudo-labeling**: Patches mapping to indices in $\mathcal{A}_{\text{train}}$ are marked as Normal ($\tilde{y} = 0$), while unrecognized codes are marked as Abnormal ($\tilde{y} = 1$).
* **Contrastive Learning**: Encourages the model to learn a clear boundary separating the pseudo-labeled classes.
* **Inference-then-Train**: Performs testing on each batch *prior* to model weight adaptation, ensuring fair, non-leaking evaluations.

---
*See [[wiki/research/COMET Summary.md]] for the full research summary and benchmark metrics.*
