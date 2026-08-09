---
title: "SASRec (Self-Attention Sequential Recommendation)"
tags: ["entity", "architecture", "sequential-recommendation", "transformer", "mathematics"]
aliases: ["Self-Attention Sequential Recommendation", "SASRec Algorithm"]
date: 2026-08-08
sources: [
  "[[wiki/research/LLaRA Summary.md]]",
  "[[wiki/research/A-LLMRec Summary.md]]",
  "[[wiki/research/Conversational Recommender Systems.md]]"
]
---

# SASRec (Self-Attention Sequential Recommendation)

Introduced by Wang-Cheng Kang and Julian McAuley at WWW 2018, **SASRec** (Self-Attention Sequential Recommendation) is a seminal sequential recommendation architecture. It was the first model to successfully apply the self-attention mechanism (Transformer) to sequential recommendation, establishing a powerful bridge between low-resource Markov Chains (MC) and high-resource Recurrent Neural Networks (RNNs).

---

## 1. Core Motivation

Sequential recommendation aims to predict a user's next action based on their historical interaction sequence. Historically, two paradigms dominated:
1.  **Markov Chains (MCs):** Model local, short-term transitions (e.g., predicting item $t$ based *only* on item $t-1$). Highly efficient and robust on sparse datasets, but fails to capture long-term context or multi-step dependencies.
2.  **Recurrent Neural Networks (RNNs):** Model global, long-term sequential dependencies. Capable of capturing complex histories, but prone to overfitting on sparse data, computationally expensive, and assumes a strict step-by-step sequential transition.

**SASRec** uses a **decoder-only self-attention mechanism** to dynamically weight past items. It behaves like an adaptive Markov Chain: for sparse histories, it can focus strictly on the most recent item, while for dense histories, it can attend to long-range, multi-step dependencies across the entire sequence.

---

## 2. Mathematical Architecture

SASRec takes an input sequence of item IDs: $\mathcal{S}^u = (i_1^u, i_2^u, \dots, i_n^u)$ and outputs a predicted probability distribution over the item catalog.

```text
Input Sequence: (i_1, i_2, ..., i_n)
       │
       ▼ (Embedding Lookup)
Item Embeddings (M) + Positional Embeddings (P)
       │
       ▼ (Summation)
Input Representation Matrix E_0
       │
       ▼ (Multi-Head Self-Attention Block)
  ┌──► Query, Key, Value Projections (Q, K, V)
  │    │
  │    ▼ (Scaled Dot-Product Attention)
  │    Attention = Softmax( (Q K^T) / sqrt(d) + Mask ) V
  │    │
  │    ▼ (Dropout & Residual Addition & LayerNorm)
  │    Intermediate Vector F
  │    │
  │    ▼ (Point-wise Feed-Forward Network - FFN)
  │    FFN(F) = ReLU(F W_1 + b_1) W_2 + b_2
  │    │
  │    ▼ (Dropout & Residual Addition & LayerNorm)
  └─── Feed to Next Layer (E_l)
       │ (After L Blocks)
       ▼
Final Latent State Representation (x^u)
       │
       ▼ (Compute Similarity with Item Catalog Matrix E)
Scores = x^u * E^T (Dot-product output)
```

### 2.1. Embedding Layer
Since self-attention contains no recurrent or convolutional structures, it has no native concept of sequence order. SASRec injects order information by combining item embeddings with trainable positional embeddings.

Given a maximum sequence length $n$ and embedding dimension $d$:
*   **Item Embedding Matrix:** $\mathbf{M} \in \mathbb{R}^{|\mathcal{I}| \times d}$
*   **Positional Embedding Matrix:** $\mathbf{P} \in \mathbb{R}^{n \times d}$

The input representation matrix $\mathbf{E}_0 \in \mathbb{R}^{n \times d}$ is constructed by summing the embeddings of the sequence items and their corresponding positions:
$$\mathbf{E}_0 = \left[ \mathbf{M}_{i_1} + \mathbf{P}_1; \, \mathbf{M}_{i_2} + \mathbf{P}_2; \, \dots; \, \mathbf{M}_{i_n} + \mathbf{P}_n \right]$$

### 2.2. Self-Attention Block (Layer $l$)
For each block, the input embedding matrix $\mathbf{E}_l \in \mathbb{R}^{n \times d}$ is projected into Query ($\mathbf{Q}$), Key ($\mathbf{K}$), and Value ($\mathbf{V}$) matrices using linear projection layers:
$$\mathbf{Q} = \mathbf{E}_l \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{E}_l \mathbf{W}_K, \quad \mathbf{V} = \mathbf{E}_l \mathbf{W}_V$$
*(where $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d}$ are learnable projection weights).*

#### Scaled Dot-Product Attention:
$$\mathbf{S} = \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}} + \mathbf{M}_{mask} \right) \mathbf{V}$$

Where:
*   $\sqrt{d}$ is the scaling factor to prevent variance growth and vanishing gradients during softmax.
*   **$\mathbf{M}_{mask}$ (Causal Attention Masking):** To prevent data leakage during training, the model must not look into the future. It enforces a strict upper-triangular causal mask:
    $$\mathbf{M}_{mask(i, j)} = \begin{cases} 
      0 & \text{if } i \ge j \\
      -\infty & \text{if } i < j 
    \end{cases}$$
    This sets the softmax weight of any future item $j > i$ to exactly $0$, ensuring the prediction at step $i$ is conditioned strictly on items interacted with *before or at* step $i$.

### 2.3. Point-wise Feed-Forward Network (FFN)
To incorporate non-linearity and cross-dimensional feature interactions, the attention output is passed through a two-layer point-wise FFN:
$$\mathbf{F} = \text{LayerNorm}\left( \mathbf{E}_l + \text{Dropout}(\mathbf{S}) \right)$$
$$\text{FFN}(\mathbf{F}_i) = \text{ReLU}(\mathbf{F}_i \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$
*(where $\mathbf{F}_i$ is the representation at sequence index $i$, $\mathbf{W}_1 \in \mathbb{R}^{d \times 2d}$, and $\mathbf{W}_2 \in \mathbb{R}^{2d \times d}$).*

---

## 3. Training Objective (Binary Cross-Entropy)

SASRec is trained using a sequence-to-sequence autoregressive objective. At each step $i \in \{1, \dots, n-1\}$, the model is trained to predict the next item $i_{i+1}$ (positive label) and ignore a randomly sampled negative item $j \in \mathcal{I} \setminus \mathcal{S}^u$:

$$\mathcal{L}_{SASRec} = -\sum_{u \in \mathcal{U}} \sum_{i=1}^{n-1} \left[ \log\left(\sigma\left( \mathbf{x}_i^u \cdot \mathbf{M}_{i_{i+1}}^T \right)\right) + \log\left(1 - \sigma\left( \mathbf{x}_i^u \cdot \mathbf{M}_{j}^T \right)\right) \right]$$

Where:
*   $\mathbf{x}_i^u$ is the final-layer representation of user $u$ at step $i$.
*   $\mathbf{M}_{i_{i+1}}$ is the embedding of the ground-truth next item.
*   $\mathbf{M}_j$ is the embedding of the negative item.
*   $\sigma$ is the Sigmoid activation function.

---

## 4. Role as a Behavioral Prior in LLMRec

In modern LLM-based recommender systems studied in this wiki, a pre-trained SASRec is frequently used as the **foundational collaborative encoder**:

*   **In LLaRA:** The pre-trained SASRec ID embedding $\mathbf{e}_s^i$ is extracted and aligned with the LLM embedding space using the **SR2LLM** projector. This infuses sequential co-occurrence behaviors directly into LLaRA's hybrid prompts.
*   **In A-LLMRec:** SASRec generates the raw user profile vector $\mathbf{x}^u$ and item embeddings $\mathbf{E}_i$ which are aligned with SBERT text semantics and projected onto the frozen LLM as soft prompt tokens ($\mathbf{O}_u, \mathbf{O}_i$).

---

## Related Concepts
*   [[wiki/research/LLaRA Summary.md|LLaRA Research Paper Summary]]
*   [[wiki/research/A-LLMRec Summary.md|A-LLMRec Research Paper Summary]]
*   [[wiki/research/Conversational Recommender Systems.md|Conversational Recommender Systems Synthesis Page]]
