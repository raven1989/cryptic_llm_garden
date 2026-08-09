---
title: "CTRL (Connect Collaborative and Language Model)"
tags: ["entity", "architecture", "CTR", "Contrastive-Learning", "Multimodal"]
aliases: ["Connect Collaborative and Language Model Architecture"]
date: 2026-08-08
sources: ["[[wiki/research/CTRL Summary.md]]"]
---

# CTRL (Connect Collaborative and Language Model)

**CTRL** is an aligned multi-modal CTR prediction framework. It decouples collaborative and semantic feature encoding into separate tabular and textual towers, utilizing a two-stage pre-training and fine-tuning scheme to distill semantic and world knowledge from a Pre-trained Language Model directly into a lightweight collaborative model.

---

## 1. Prompt Construction & Templates

Tabular features are serialized into structured natural language strings to preserve both semantic content and tabular field boundaries.

*   **Punctuation Syntax:**
    *   **Period `.`:** Separates the **user-side features** from the **item-side features**.
    *   **Comma `,`:** Separates individual feature fields.
    *   **Vertical Bar `|`:** Separates individual user historical interaction sequences.

*   **Prompt String Example:**
    $$\text{Prompt} = \texttt{"This is a user, gender is male, age is 25, occupation is doctor,}$$
    $$\texttt{who has recently watched Titanic|Avatar. This is a movie, title is Inception,}$$
    $$\texttt{genre is Sci-Fi, director is Christopher Nolan."}$$

---

## 2. Cross-Modal Knowledge Alignment (Stage 1)

Stage 1 aligns the tabular and textual representation spaces.

```text
Instance Tabular Row                        Instance Prompt String
        │                                             │
        ▼ (Embedding Lookups)                         ▼ (Tokenization)
One-Hot Sparse Features                       Linguistic Tokens
        │                                             │
        ▼ (Feature Interaction Layer)                 ▼ (Transformer Layer)
Col Encoder: M_col (AutoInt / DeepFM)         Sem Encoder: M_sem (RoBERTa-base)
        │                                             │
        ▼ (L2 Normalization)                          ▼ (CLS token + L2 Norm)
Global Vector: h_tab                          Global Vector: h_text
        │                                             │
        ├──────────────────────┬──────────────────────┤
        ▼ (Project to Subspaces)                      ▼ (Project to Subspaces)
h_tab_m = W_m h_tab + b_m                     h_text_m = W_m h_text + b_m
        │                                             │
        └──────────────────────┬──────────────────────┘
                               ▼
            Fine-Grained Similarity (sim_max)
                               │
                               ▼
               Symmetric InfoNCE Loss (L_ccl)
```

### 2.1. Dual-Tower Encoders
*   **Collaborative Encoder ($\mathcal{M}_{col}$):** Encodes sparse tabular inputs via embedding lookups and interaction layers to yield global vector $\mathbf{h}^{tab} \in \mathbb{R}^{d_{tab}}$ (where $\mathcal{M}_{col}$ is randomly initialized).
*   **Semantic Encoder ($\mathcal{M}_{sem}$):** Encodes the prompt string using a pre-trained **RoBERTa-base** model. The global vector $\mathbf{h}^{text} \in \mathbb{R}^{d_{text}}$ is extracted from the `[CLS]` token embedding at the final layer.

### 2.2. Global Symmetric Contrastive Loss
To prevent spatial/modality collapse in either encoder, CTRL minimizes a symmetric InfoNCE loss (inspired by the Jensen–Shannon divergence) over a batch of $N$ instances:

$$\mathcal{L}_{ccl} = \frac{1}{2} \left( \mathcal{L}^{textual2tabular} + \mathcal{L}^{tabular2textual} \right)$$

$$\mathcal{L}^{textual2tabular} = -\frac{1}{N} \sum_{k=1}^{N} \log \frac{\exp(\text{sim}(\mathbf{h}^{text}_k, \mathbf{h}^{tab}_k) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{h}^{text}_k, \mathbf{h}^{tab}_j) / \tau)}$$

$$\mathcal{L}^{tabular2textual} = -\frac{1}{N} \sum_{k=1}^{N} \log \frac{\exp(\text{sim}(\mathbf{h}^{tab}_k, \mathbf{h}^{text}_k) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{h}^{tab}_k, \mathbf{h}^{text}_j) / \tau)}$$

*   *Note on Denominators:* While cosine similarity is symmetric, the partition functions sum over different dimensions. $\mathcal{L}^{textual2tabular}$ prevents a single text query from matching multiple table entries, while $\mathcal{L}^{tabular2textual}$ prevents a single table query from matching multiple text descriptions.

---

### 2.3. Fine-Grained Sub-Space Alignment
Rather than computing simple global cosine similarity, CTRL projects the global vectors into $M$ sub-spaces:
$$\mathbf{h}_m^{tab} = \mathbf{W}_m^{tab} \mathbf{h}^{tab} + \mathbf{b}_m^{tab} \quad \text{for } m = 1, \dots, M$$
$$\mathbf{h}_m^{text} = \mathbf{W}_m^{text} \mathbf{h}^{text} + \mathbf{b}_m^{text} \quad \text{for } m = 1, \dots, M$$

Where $\mathbf{W}_m^{tab} \in \mathbb{R}^{d_{sub} \times d_{tab}}$ and $\mathbf{W}_m^{text} \in \mathbb{R}^{d_{sub} \times d_{text}}$ are trainable projection matrices. The similarity score is calculated as the sum of maximum correlations across these $M$ sub-spaces:
$$\text{sim}(\mathbf{h}_i, \mathbf{h}_j) = \sum_{m_i = 1}^{M} \max_{m_j \in 1, \dots, M} \left\{ (\mathbf{h}_{i, m_i})^T \mathbf{h}_{j, m_j} \right\}$$

This multi-aspect alignment enables the model to map local sub-features (e.g., aligning user demographic attributes in sub-space 1, and historical behavior logs in sub-space 2) rather than just overall global similarities.

---

## 3. Supervised Fine-Tuning (Stage 2)

After Stage 1 alignment is completed, the computationally heavy semantic encoder $\mathcal{M}_{sem}$ is **discarded**.

*   **Downstream Fine-Tuning:** The lightweight collaborative model ($\mathcal{M}_{col}$) is fine-tuned strictly on the downstream click labels ($y \in \{0, 1\}$) using standard **Binary Cross Entropy (BCE)** loss:
    $$\mathcal{L}_{ctr} = -\frac{1}{N} \sum_{k=1}^{N} \left( y_k \log(\hat{y}_k) + (1-y_k) \log(1-\hat{y}_k) \right)$$
*   **Online Serving:** At runtime, only $\mathcal{M}_{col}$ is loaded. This allows the system to serve predictions containing semantic and world knowledge signals instantly within standard 10ms SLAs.

---

## Related Concepts
*   [[wiki/research/CTRL Summary.md|CTRL Research Paper Summary]]
*   [[wiki/research/Conversational Recommender Systems.md|Conversational Recommender Systems Synthesis Page]]
