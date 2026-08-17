---
tags: ["entity", "architecture", "recommendation", "attention", "CTR"]
aliases: ["DIN", "Deep Interest Network"]
date: 2026-08-17
sources: ["[[wiki/research/DIN Summary.md]]"]
---

# DIN

**DIN (Deep Interest Network)** is a CTR prediction architecture published at KDD 2018 by Alibaba Group. It introduces a **local activation unit** that adaptively computes user interest representations **with respect to each candidate ad**, replacing the fixed-length pooling of the standard Embedding&MLP paradigm.

## Core Architectural Pillars

### 1. Feature Representation (Multi-Group Categorical)
Industrial CTR data is encoded as concatenated sparse binary vectors across feature groups:
- **One-hot** groups (gender, ad goods_id): lookup returns a single embedding vector
- **Multi-hot** groups (visited_goods_ids): lookup returns a variable-length list of embedding vectors — one per nonzero element
- Four categories: User Profile, User Behavior, Ad, Context
- No handcrafted combination features — all interactions learned by the network

### 2. Local Activation Unit (Key Innovation)
Replaces sum/average pooling on user behavior features with candidate-aware weighted pooling:

$$\mathbf{v}_U(A) = \sum_{j=1}^{H} a(\mathbf{e}_j, \mathbf{v}_A) \cdot \mathbf{e}_j$$

The activation function $a(\cdot)$ is a feed-forward network taking the behavior embedding $\mathbf{e}_j$, the ad embedding $\mathbf{v}_A$, and their **outer product** $\mathbf{e}_j \otimes \mathbf{v}_A$ (explicit relevance signal).

**Key difference from standard attention:** Softmax normalization ($\sum w_i = 1$) is **deliberately dropped** to preserve **interest intensity**. The unnormalized $\sum w_i$ approximates how strongly the user's interests are activated — a user with 90% clothing history should produce a larger-magnitude $\mathbf{v}_U$ for a T-shirt ad than for a phone ad.

### 3. Mini-batch Aware Regularization
Standard L2 requires computing the norm over **all** embedding parameters (billions) per mini-batch — infeasible. MBA approximates L2 by only penalizing features that **appear in the current mini-batch**:

$$L_2(\mathrm{W}) \approx \sum_{j=1}^{K} \sum_{m=1}^{B} \frac{\alpha_{mj}}{n_j} \|\mathbf{w}_j\|_2^2$$

where $\alpha_{mj}$ indicates whether feature $j$ appears in mini-batch $\mathcal{B}_m$, and $n_j$ is the total occurrence count. The $\frac{1}{n_j}$ weighting gives rare features a proportionally larger penalty (they overfit most).

### 4. Dice Activation Function
A data-adaptive generalization of PReLU. Replaces the hard rectification at 0 with a smooth sigmoid centered at the input mean:

$$p(s) = \frac{1}{1 + e^{-\frac{s - E[s]}{\sqrt{Var[s] + \epsilon}}}}, \quad f(s) = p(s) \cdot s + (1 - p(s)) \cdot \alpha s$$

- Rectification point adapts to the input distribution (moves to $E[s]$)
- Smooth transition between identity and leaky channels
- Uses mini-batch statistics during training, moving averages during testing
- Degenerates to PReLU when $E[s] = 0$ and $Var[s] = 0$

---

## Key Properties

| Property | Detail |
| :--- | :--- |
| **User representation** | Varies per candidate ad (not fixed-length) |
| **Attention normalization** | None (softmax deliberately removed) |
| **Activation unit input** | Behavior embedding + ad embedding + outer product |
| **Regularization** | Mini-batch aware L2 (only penalizes features in current batch) |
| **Activation function** | Dice (data-adaptive generalization of PReLU) |
| **Sequence modeling** | None (LSTM tried, no improvement) |

---

## Experimental Results

**Alibaba Dataset (2.14B samples):**

| Model | AUC | RelaImpr |
| :--- | :--- | :--- |
| BaseModel | 0.5970 | 0.00% |
| DeepFM | 0.5993 | 2.37% |
| DIN | 0.6029 | 6.08% |
| **DIN + MBA + Dice** | **0.6083** | **11.65%** |

**Online A/B test:** +10.0% CTR, +3.8% RPM over previous production model.

**Key finding:** At industrial scale, 0.001 absolute AUC gain is significant. DIN's +0.0113 gain over BaseModel is a major improvement.

---

## Related Wiki Pages
* [[DIN Summary]]: Complete, section-by-section research summary of the paper.
* [[DCN]]: Another foundational CTR architecture using explicit feature crossing instead of attention.
* [[NCF]]: Neural Collaborative Filtering, learning user-item interaction functions with neural networks.
* [[DSSM]]: Deep Structured Semantic Model for semantic matching.
* [[STCA]]: ByteDance's target cross-attention mechanism, a successor direction to DIN's attention.
* [[HSTU]]: Meta's Generative Recommenders, addressing sequential user behavior modeling.
