---
tags:
  - recommendation
  - generative-retrieval
  - architectures
aliases:
  - OneRec Model
date: 2026-06-05
sources: ["[[wiki/research/OneRec Summary.md]]"]
---

# OneRec

**OneRec** is a unified, end-to-end generative recommendation model developed by **Kuaishou Inc.** (published in Feb 2025). It replaces the entire traditional cascade ranking pipeline (Recall $\to$ Pre-ranking $\to$ Ranking) with a **single-stage sequence-to-sequence Transformer**.

---

## Architectural Pillars

OneRec achieves high sorting accuracy and deployment-level efficiency through three primary innovations:

### 1. Unified Sequence Generation
Rather than using separate models for coarse retrieval and fine sorting, OneRec uses a T5-like **Encoder-Decoder** architecture:
* **Encoder**: Encodes the user's historical behavior sequence represented as discrete semantic IDs.
* **Decoder**: Autoregressively generates the semantic ID tokens of the entire recommended session (a cohesive list of 5-10 items).

### 2. High Computational Efficiency (Sparse MoE)
To scale up parameter capacity to represent complex user behaviors, OneRec uses **Sparse Mixture-of-Experts ([[Sparsely-Gated MoE Layer|MoE]])** in the decoder:
* Uses $N_{\text{MoE}} = 24$ experts, activating $K_{\text{MoE}} = 2$ experts per token.
* This allows OneRec to scale to a **1 Billion parameter model** while only activating **13% of its parameters** during any single forward pass, keeping online serving latency extremely low.

### 3. Iterative Preference Alignment (IPA)
Since traditional Direct Preference Optimization ([[Fine-tuning|DPO]]) cannot collect simultaneous positive/negative session lists for a single user, OneRec introduces:
* A pre-trained, multi-objective **Personalized Reward Model** to score generated lists.
* An **Iterative DPO loop** that extracts "self-hard negatives" from the generator's own beam search candidates, aligning the model's outputs directly with user engagement metrics (watch time, clicks, and likes).

---

## Technical Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model Size** | 1.0 Billion parameters | Total parameter capacity |
| **Activated Parameters** | ~130 Million (13%) | Parameters computed per forward pass |
| **Layers of Semantic ID ($L$)** | 3 layers | Length of semantic identifiers |
| **Codebook Size ($K$)** | 8192 | Centroids per semantic layer |
| **Expert Count ($N_{\text{MoE}}$)** | 24 | Total FFN experts in decoder |
| **Active Experts ($K_{\text{MoE}}$)** | 2 | Active experts per token |
| **Context Length ($n$)** | 256 items | User's historical behavior context |
| **Session Length ($m$)** | 5 target items | Chained target list length |

---

## Production Impact (Kuaishou Feed)
* **Total Watch Time**: **+1.68%**
* **Average View Duration**: **+6.56%**

This proves that end-to-end generative retrieval can successfully surpass carefully hand-tuned multi-stage cascaded ranking systems in large-scale real-world applications.

---

## Related Wiki Pages
* [[OneRec Summary]]: Complete, section-by-section research summary of the paper.
* [[Balanced K-means]]: Detailed breakdown of the quantization algorithm.
* [[Sparsely-Gated MoE Layer]]: Core MoE concepts.
* [[KV Cache]]: Memory and latency optimization for real-time autoregressive serving.
