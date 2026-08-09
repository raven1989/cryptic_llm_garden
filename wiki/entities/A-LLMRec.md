---
title: "A-LLMRec: All-round LLM-based Recommender"
tags: ["entity", "architecture", "LLMRec", "Knowledge-Alignment"]
aliases: ["All-round LLM-based Recommender Architecture"]
date: 2026-08-08
sources: ["[[wiki/research/A-LLMRec Summary.md]]"]
---

# A-LLMRec: All-round LLM-based Recommender

**A-LLMRec** is an aligned multimodal-style recommendation framework. It aligns a completely frozen traditional collaborative model and a completely frozen LLM using dual-stage projection MLPs, treating pre-trained collaborative features and text-semantically generated features as continuous virtual tokens.

---

## 1. Stage 1: Collaborative & Textual Knowledge Alignment

This stage trains a unified joint-embedding generator for items, projecting collaborative factors ($\mathbf{E}_i$) and textual factors ($\mathbf{Q}_i$) into a shared latent space.

### 1.1. Dual-Encoder latent matching
*   **Item Encoder ($f_I^{enc}$):** A 1-layer MLP projecting frozen pre-trained CF embeddings $\mathbf{E}_i \in \mathbb{R}^d$ into latent embedding $\mathbf{e}_i \in \mathbb{R}^{d^{\prime}}$.
*   **Text Encoder ($f_T^{enc}$):** A 1-layer MLP projecting fine-tuned SBERT embeddings $\mathbf{Q}_i \in \mathbb{R}^{768}$ into latent embedding $\mathbf{q}_i \in \mathbb{R}^{d^{\prime}}$.
*   **Latent Matching Loss:** Optimized using Mean Squared Error (MSE) averaged across all user interaction logs $\mathcal{S}$:
    $$\mathcal{L}_{\text{matching}} = \underset{\mathcal{S}^{u}\in\mathcal{S}}{\mathbb{E}}\left[\underset{i\in\mathcal{S}^{u}}{\mathbb{E}}\left[MSE(f_{I}^{enc}(\mathbf{E}_{i}), f_{T}^{enc}(\mathbf{Q}_{i}))\right]\right]$$
    *(where $\mathbb{E}$ represents the Expectation / average across sets).*

### 1.2. Reconstruction Decoders (Avoiding Over-smoothing)
To prevent the encoders from collapsing to trivial zero-mappings, A-LLMRec introduces decoders ($f_I^{dec}, f_T^{dec}$) to reconstruct the original high-dimensional features:
$$\mathcal{L}_{\text{item-recon}} = \underset{\mathcal{S}^{u}\in\mathcal{S}}{\mathbb{E}}\left[\underset{i\in\mathcal{S}^{u}}{\mathbb{E}}\left[MSE\left(\mathbf{E}_{i}, \, f_{I}^{dec}\left(f_{I}^{enc}(\mathbf{E}_{i})\right)\right)\right]\right]$$
$$\mathcal{L}_{\text{text-recon}} = \underset{\mathcal{S}^{u}\in\mathcal{S}}{\mathbb{E}}\left[\underset{i\in\mathcal{S}^{u}}{\mathbb{E}}\left[MSE\left(\mathbf{Q}_{i}, \, f_{T}^{dec}\left(f_{T}^{enc}(\mathbf{Q}_{i})\right)\right)\right]\right]$$

### 1.3. Sequential Recommendation Loss
To maintain downstream user-matching utility, the latent embeddings are trained to maximize sequential recommendation predictions:
$$\mathcal{L}_{\text{rec}} = -\sum_{\mathcal{S}^{u}\in\mathcal{S}}\left[ \log\left(\sigma\left(s\left(\mathbf{x}^{u}_{|\mathcal{S}^{u}|-1}, \, f^{dec}_{I}(f^{enc}_{I}(\mathbf{E}_{i^{u}_{|\mathcal{S}^{u}|}}))\right)\right)\right) + \log\left(1-\sigma\left(s\left(\mathbf{x}^{u}_{|\mathcal{S}^{u}|-1}, \, f^{dec}_{I}(f^{enc}_{I}(\mathbf{E}_{i^{u,-}_{|\mathcal{S}^{u}|}}))\right)\right)\right) \right]$$
*(where $\mathbf{x}^{u}_{|\mathcal{S}^{u}|-1}$ is the frozen CF user state, $i^{u}$ is the positive item, and $i^{u,-}$ is the negative item).*

### 1.4. Joint Embedding Output
The final Stage-1 loss is:
$$\mathcal{L}_{\text{stage-1}} = \mathcal{L}_{\text{matching}} + \alpha \mathcal{L}_{\text{item-recon}} + \beta \mathcal{L}_{\text{text-recon}} + \mathcal{L}_{\text{rec}}$$
Once trained, the latent item vector is designated as **Joint Collaborative-Text Embedding**:
$$\text{Joint Embedding} = \begin{cases} 
\mathbf{e}_i = f_I^{enc}(\mathbf{E}_i) & \text{for Warm Items} \\
\mathbf{q}_i = f_T^{enc}(\mathbf{Q}_i) & \text{for Cold Items (emulating collaborative patterns via text)}
\end{cases}$$

---

## 2. Stage 2: Token Projection & LLM Alignment

This stage projects the Stage-1 joint embeddings into a frozen LLM's vocabulary embedding space.

```text
Instance Pre-trained User State (x^u)        Joint Collaborative-Text Emb (e_i)
                 │                                           │
                 ▼ (2-layer MLP: F_U)                        ▼ (2-layer MLP: F_I)
           User Token: O_u                             Item Token: O_i
                 │                                           │
                 └───────────────────┬───────────────────────┘
                                     ▼
                      Inserted into prompt template:
                 "O_u This user watched Movie A O_a... 
                  Recommend a movie to watch next... y_u"
                                     │
                                     ▼
                Autoregressive Target-Masked Cross-Entropy
                  (Only next item title tokens predicted)
```

### 2.1. Coordinate Projection
Two 2-layer MLPs project the user representation $\mathbf{x}^u \in \mathbb{R}^d$ and the item's joint embedding $\mathbf{e}_i \in \mathbb{R}^{d^{\prime}}$ into the LLM token dimension $d^{\text{token}}$:
$$\mathbf{O}_u = F_U(\mathbf{x}^u) \in \mathbb{R}^{d^{\text{token}}} \quad \text{and} \quad \mathbf{O}_i = F_I(\mathbf{e}_i) \in \mathbb{R}^{d^{\text{token}}}$$

### 2.2. Prompt Soft-Prompt Wrapping
These projected continuous vectors are treated directly as standard LLM token embeddings, bypassing the token-lookup table, and are wrapped inside a template:
$$\text{Prompt Input } p^u = [\mathbf{O}_u] \texttt{ This user has watched Movie A } [\mathbf{O}_a] \texttt{... Recommend a movie to watch next...}$$

### 2.3. Causal Target Masking Objective
The LLM parameters $\Theta$ are kept **100% frozen**. Only the projector parameters $\theta = \{F_U, F_I\}$ are optimized. 

Importantly, standard task instructions, template words, and candidate lists in $p^u$ are **masked out during backpropagation** (assigned index `-100` in cross-entropy). The model optimizes gradients strictly over the tokens representing the textual title of the recommended movie $y^u$:

$$\max_{\theta} \sum_{\mathcal{S}^{u}\in\mathcal{S}}\sum_{k=1}^{|y^{u}|}\log(P_{\theta, \Theta}(y^{u}_{k} \mid p^{u},y^{u}_{<k}))$$

This optimizes $F_U$ and $F_I$ to map collaborative patterns into soft-prompt embeddings that naturally guide the frozen LLM attention heads to predict the correct next movie title string.

---

## Related Concepts
*   [[wiki/research/A-LLMRec Summary.md|A-LLMRec Research Paper Summary]]
*   [[wiki/entities/SASRec.md|SASRec (Self-Attention Sequential Recommendation)]]
*   [[wiki/research/Conversational Recommender Systems.md|Conversational Recommender Systems Synthesis Page]]
