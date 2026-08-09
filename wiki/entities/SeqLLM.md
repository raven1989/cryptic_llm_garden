---
title: "SeqLLM (Behavioral-Sequence Augmented LLM)"
tags: ["entity", "architecture", "SeqLLM", "WeChat-Pay", "Prefix-Guided-SFT"]
aliases: ["Behavioral-Sequence Augmented LLM Architecture"]
date: 2026-08-09
sources: ["[[wiki/research/SeqLLM Summary.md]]"]
---

# SeqLLM (Behavioral-Sequence Augmented LLM)

**SeqLLM** is an aligned multi-modal sequence-language framework. It treats chronological transaction events as a discrete behavioral vocabulary, projecting them into a fully trainable LLM's high-dimensional latent space using a lightweight projector, and injecting sequence capability through prefix-guided supervised fine-tuning.

---

## 1. Discrete Behavioral Vocabulary & Rescaling

Rather than serializing transaction logs as plain text or using whole-event IDs, SeqLLM factorizes each event $s_t$ into separate discrete fields.

### 1.1. Field-Level Factorization
An event $s_t \in \mathcal{E}$ is represented as a concatenated string of field value tokens:
$$s_t = \langle f_1:v_1 \rangle \langle f_2:v_2 \rangle \dots \langle f_k:v_k \rangle$$
Where $f_i$ is the field type (e.g., Time, Scene, Channel, Amount, Status) and $v_i \in V_i$ is the discretized value (e.g., log-scale amount buckets). The unified behavioral vocabulary is:
$$\mathcal{V}_b = \bigcup_{i} \{ \langle f_i:v \rangle \mid v \in V_i \}$$

### 1.2. Semantic-Aware Embedding Rescaling
To stabilize initialization, a new behavior token $v \in \mathcal{V}_b$ is initialized using the LLM's standard text embeddings $\mathbf{E}$ of its descriptive words $Z_v$:
$$\bar{\mathbf{e}}_{v} = \frac{1}{|Z_{v}|}\sum_{z\in Z_{v}}\mathbf{E}[z]$$

Because averaging vectors shrinks their variance, the pooled vector $\bar{\mathbf{e}}_v$ is rescaled to match the statistical properties of the native LLM embedding table:
$$\mathbf{e}_{v}^{(0)} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \frac{\bar{\mathbf{e}}_{v} - \hat{\boldsymbol{\mu}}}{\hat{\boldsymbol{\sigma}}}$$

Where:
*   $\boldsymbol{\mu}, \boldsymbol{\sigma}$ are the per-dimension mean and standard deviation of the entire **native LLM embedding table**.
*   $\hat{\boldsymbol{\mu}}, \hat{\boldsymbol{\sigma}}$ are the mean and standard deviation of the **newly pooled set of behavioral vectors** $\{\bar{\mathbf{e}}_{v}\}$.
*   $\odot$ represents the element-wise Hadamard product.

---

## 2. Shared Behavior Projector ($g_\psi$)

To enforce a shared cross-token coordinate constraint, every behavioral token vector $\mathbf{e}$ passes through a lightweight shared projector $g_\psi$:
$$g_{\psi}(\mathbf{e}) = \mathbf{e} + \text{MLP}_{\psi}(\mathbf{e})$$
Where $\text{MLP}_{\psi}$ is a 2-layer MLP (Linear–GELU–Linear) with a skip connection whose final linear layer is **zero-initialized** (so the projector starts as the identity and learns small, stable alignments).

---

## 3. Two-Stage Sequence-Language Alignment

The projector and LLM backbone are pre-aligned using a two-stage curriculum:

### 3.1. Stage 1: Translation (Grounding)
*   **Input ($x$):** A short sequence of ungrounded behavioral tokens: `[<Time:Sun_14h> <Channel:Msg link> <Status:Success>]`.
*   **Instruction ($\mathcal{I}$):** `"Translate these behavior tokens."`
*   **Target ($y$):** Text translation: `"Sunday at 14:00; successful message-link payment."`.
*   **Objective ($\mathcal{L}_{\text{bridge}}$):** Autoregressive next-token prediction loss over target translation $y$:
    $$\mathcal{L}_{\text{bridge}} = -\sum_{(\mathcal{I}, x, y) \in \mathcal{D}_{\text{trans}}} \log p_{\Theta,\psi}(y \mid \mathcal{I}, x)$$

### 3.2. Stage 2: Reasoning (Contextualization)
*   **Input ($x$):** Multi-event transaction sequences: `Tx1: [...] Tx2: [...]`.
*   **Instruction ($\mathcal{I}$):** `"What pattern is shared by these transactions?"`
*   **Target ($y$):** Summary text: `"Both are Tuesday-afternoon QR payments."`.

---

## 4. Prefix-Guided Capability Injection

To prevent catastrophic forgetting of language skills during sequence learning, SeqLLM avoids global Continual Pre-training (CPT) and turns sequence completion into an **instruction-conditioned SFT task**.

### 4.1. Global CPT vs. Prefix-Guided SFT Objectives
*   **Continual Pre-training Loss ($\mathcal{L}_{\mathrm{CPT}}$):** Computes loss on 100% of the behavioral sequence tokens $b_{1:N}$:
    $$\mathcal{L}_{\mathrm{CPT}} = -\sum_{t=1}^{N}\log p_{\Theta}(b_t \mid b_{<t})$$
*   **Prefix-Guided SFT Loss ($\mathcal{L}_{\mathrm{Prefix}}$):** Cuts the sequence at boundary $m$ ($70\%$). Loss is computed strictly on the remaining $30\%$ future response target:
    $$\mathcal{L}_{\mathrm{Prefix}} = -\sum_{t=m+1}^{N}\log p_{\Theta,\psi}(b_t \mid \mathcal{I}, b_{<t})$$

### 4.2. Unified SFT Optimization ($\mathcal{L}_{\text{inject}}$)
The model (projector $\psi$ and LLM backbone $\Theta$) is fine-tuned jointly using a unified next-token prediction loss:
$$\mathcal{L}_{\text{inject}} = -\sum_{(\mathcal{I},c,y) \in \mathcal{D}_{\text{inj}}} \log p_{\Theta,\psi}(y \mid \mathcal{I}, c)$$

Where $\mathcal{D}_{\text{inj}}$ is a mixed dataset containing:
1.  **Task-Prefixed Sequence Completion instances:** (Where prefix $b_{1:m}$ is context $c$, and suffix $b_{m+1:N}$ is target response $y$). This trains the model on temporal transitions and transactional transitions.
2.  **General Instruction SFT instances:** Standard language Q&A. This acts as a regularizer, forcing the LLM to retain its natural language and reasoning structures.

---

## Related Concepts
*   [[wiki/research/SeqLLM Summary.md|SeqLLM Research Paper Summary]]
*   [[wiki/research/Conversational Recommender Systems.md|Conversational Recommender Systems Synthesis Page]]
