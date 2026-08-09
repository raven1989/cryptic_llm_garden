---
title: "MCCRS (Multi-Type Context-Aware Conversational Recommender System)"
tags: ["entity", "architecture", "CRS", "MoE"]
aliases: ["Multi-Type Context-Aware Conversational Recommender System Architecture"]
date: 2026-08-06
sources: ["[[wiki/research/MCCRS Summary.md]]"]
---

# MCCRS (Multi-Type Context-Aware Conversational Recommender System)

**MCCRS** is a conversational recommendation framework structured around a **Mixture-of-Experts (MoE)** pattern. It decomposes the modeling of heterogeneous structured and unstructured contexts into individual neural experts, coordinated by a central routing layer named **ChairBot**.

---

## 1. Dialogue Processing & Annotation

At runtime, the system handles unstructured dialogue utterances ($s_t$) by performing named entity extraction and linking.

*   **Anchor Extraction (TagMe):** An on-the-fly entity linker matches phrases in the conversation text to unique DBpedia resource URIs. This maps both explicit item mentions (movies) and implicit entity anchors (e.g., actors, genres, directors) into standard database indices $\mathcal{E}$.
*   **User Action Sequence:** These linked entities are ordered chronologically based on dialogue history to construct the active user context profile.

---

## 2. Specialized Experts

MCCRS deploys three domain-specific experts:

### 2.1. Conversation Expert ($C$)
*   **Architecture:** A sequential Transformer utilizing a self-attention Cloze-task training scheme.
*   **Formula:** Input embeddings for the sequence elements are composed of entity embeddings $\mathbf{s}_k$ combined with positional embeddings $\mathbf{p}_k$:
    $$\mathbf{h}_k = \mathbf{s}_k + \mathbf{p}_k$$
*   **Output:** The final hidden state $\mathbf{h}^N$ at layer $N$ is mapped to a probability distribution over the item set $\mathcal{I}$ via:
    $$P_C(i) = \text{Softmax}(\mathbf{W}_C \mathbf{h}^N + \mathbf{b}_C)$$

### 2.2. Graph Expert ($G$)
*   **Architecture:** A 1-layer Relational Graph Convolutional Network (R-GCN) trained over the DBpedia subgraph.
*   **Formula:** Active entity embeddings are pooled via self-attention to generate the graph-based user profile $\mathbf{n}_{e_u}$. It scores all candidates $i$ by computing similarity with their R-GCN representations $\mathbf{n}_i$:
    $$P_G(i) = \text{Softmax}(\mathbf{n}_{e_u}^T \mathbf{n}_i)$$

### 2.3. Review Expert ($R$)
*   **Architecture:** A hierarchical model that encodes unstructured user reviews scraped from IMDb.
*   **Formula:** For movie $i$ with $m$ review sentences, its representations are compiled into matrix $\mathbf{D}^m$ and encoded via a Transformer + self-attention to form review embedding $\mathbf{v}_{R_i}$:
    $$\mathbf{v}_{R_i} = \text{SelfAttention}(\text{Transformer}(\mathbf{D}^m))$$
*   Similar self-attention pooling over historical entity reviews yields user review profile $\mathbf{v}_{R_u}$. Scoring matches via dot-product:
    $$P_R(i) = \text{Softmax}(\mathbf{v}_{R_u}^T \mathbf{v}_{R_i})$$

---

## 3. Mixture-of-Experts Recommender (ChairBot)

The **ChairBot** coordinates the domain experts by evaluating their output states and computing dynamic authority weights for the final recommendation.

### 3.1. Representation Concatenation
For each expert $b \in \{C, G, R\}$ and item candidate $i$, the ChairBot concatenates the hidden embedding $\mathbf{h}_i^b$ and the generated output probability $\mathbf{p}_i^b$:
$$\mathbf{h}_b = \mathbf{h}_i^b \oplus \mathbf{p}_i^b$$

### 3.2. Dynamic Routing Weights
A Multi-Layer Perceptron (MLP) processes the consolidated expert representation $\mathbf{h}_b$ to score individual expert relevance, which is then normalized using a softmax-like ratio:
$$\beta_b = \text{MLP}(\mathbf{h}_b)$$
$$\lambda_b = \frac{\beta_b}{\beta_C + \beta_G + \beta_R}$$

Where $\lambda_C, \lambda_G, \lambda_R \ge 0$ and $\lambda_C + \lambda_G + \lambda_R = 1$.

### 3.3. Probability Fusion
The final recommendation probability $P_{rec}(i)$ is computed as the weighted sum of predictions across all experts:
$$P_{rec}(i) = \lambda_C P_C(i) + \lambda_G P_G(i) + \lambda_R P_R(i)$$

The entire recommender is fine-tuned end-to-end using cross-entropy loss over the target user-accepted items.

---

## 4. Response Generation & Bias Injection

The dialogue generator is a Transformer decoder that integrates cross-attention layers to align the pre-trained hidden representations ($\mathbf{F}_C$, $\mathbf{F}_G$, $\mathbf{F}_R$) of all three experts.

To ensure consistency between the recommended items and the generated text, the decoder utilizes the recommendation distribution $P_{rec}(i)$ to inject **vocabulary bias** on the decoder's token prediction logits, steering the conversation naturally toward the selected items.

---

## Related Concepts
*   [[wiki/research/MCCRS Summary.md|MCCRS Research Paper Summary]]
*   [[wiki/entities/R-GCN.md|R-GCN (Relational Graph Convolutional Network)]]
