---
title: "LLaRA (Large Language-Recommendation Assistant)"
tags: ["entity", "architecture", "CRS", "Sequential-Recommendation"]
aliases: ["Large Language-Recommendation Assistant Architecture"]
date: 2026-08-08
sources: ["[[wiki/research/LLaRA Summary.md]]"]
---

# LLaRA (Large Language-Recommendation Assistant)

**LLaRA** is an aligned multi-modal sequential recommendation framework. It treats user sequential behaviors as a non-textual modality, using a two-layer MLP projector named **SR2LLM** to align traditional sequential recommender embeddings with a frozen LLM's high-dimensional text token space.

---

## 1. Item Representation Modalities

LLaRA represents each item in the prompt by concatenating independent semantic and behavioral feature tokens.

### 1.1. Textual Tokens ($\langle\mathbf{emb}_{t}^i\rangle$)
For item $i$ with text metadata $txt_i$, the textual representation is obtained directly via the LLM tokenizer and embedding lookup table:
$$\langle\mathbf{emb}_{t}^i\rangle = \mathbf{LLM\text{-}TKZ}(txt_i) \in \mathbb{R}^{M \times d_2}$$
Where $M$ is the variable number of tokens mapped to the text string, and $d_2$ is the LLM dimension (e.g., `4096`).

### 1.2. Behavioral Token ($\langle\mathbf{emb}_{s}^i\rangle$)
The behavioral representation is extracted from a pre-trained sequential recommender (SR) (e.g., SASRec, Caser, or GRU4Rec):
$$\mathbf{e}_{s}^i = \mathbf{SR\text{-}EMB}(i; \Theta_e) \in \mathbb{R}^{d_1}$$
Where $d_1$ is the low-rank recommender dimension (typically `64` or `128`). To bridge the modality and dimensionality gaps, the **SR2LLM** projector (a 2-layer MLP) projects this embedding into the LLM token space:
$$\langle\mathbf{emb}_{s}^i\rangle = \mathbf{Proj}(\mathbf{e}_{s}^i; \Theta_p) \in \mathbb{R}^{1 \times d_2}$$
$$\mathbf{Proj}(\mathbf{x}) = \mathbf{W}_2 \cdot \sigma\left(\mathbf{W}_1 \cdot \mathbf{x} + \mathbf{b}_1\right) + \mathbf{b}_2$$
*(where the activation $\sigma$ is typically a non-linear GELU function).*

### 1.3. Hybrid Token Representation ($\langle\mathbf{emb}_{c}^i\rangle$)
The complete representation of item $i$ concatenates the variable-length textual sequence and the singular projected behavioral token:
$$\langle\mathbf{emb}_{c}^i\rangle = \mathbf{Concat}(\langle\mathbf{emb}_{t}^i\rangle, \langle\mathbf{emb}_{s}^i\rangle) \in \mathbb{R}^{(M+1) \times d_2}$$

---

## 2. Hybrid Prompt Design

Unlike point-wise classification, LLaRA formats sequential recommendation as a list-wise ranking task over candidate sets $\mathbb{C} = \{c_1, \dots, c_m\}$.

*   **Text-Only Prompt Layout:**
    *   *Interaction Sequence:* `Titanic <PH>, Interstellar <PH>`
    *   *Candidate Set:* `Inception <PH>, Toy Story <PH>`
    *   Where `<PH>` is mapped to a static, randomly initialized embedding.
*   **Hybrid Prompt Layout:**
    *   *Interaction Sequence:* `Titanic <emb_s^14>, Interstellar <emb_s^88>`
    *   *Candidate Set:* `Inception <emb_s^140>, Toy Story <emb_s^42>`
    *   Where `<PH>` is dynamically replaced with the continuous projected behavioral token $\langle\mathbf{emb}_{s}^i\rangle$.

---

## 3. Curriculum Prompt Tuning Paradigm

LLaRA structures training using a progressive curriculum learning scheduler to avoid optimization instability caused by sudden representation shifts.

### 3.1. Dual Optimization Objectives

#### Easy Task Objective (Text-Only Loss):
Tunes only the LLM's low-rank LoRA adapters $\Theta$ on text-only prompts ($x^t$, $y^t$):
$$\mathcal{L}_{easy} = -\sum_{j=1}^{|y^t|} \log \left( P_{\Phi_0 + \Delta\Phi(\Theta)}(y^t_j \mid x^t, y^t_{<j}) \right)$$

#### Hard Task Objective (Hybrid Loss):
Jointly tunes the LoRA adapters $\Theta$, the SR2LLM projector parameters $\Theta_p$, and the base recommender embedding table $\Theta_e$ on hybrid prompts ($x^h$, $y^h$):
$$\mathcal{L}_{hard} = -\sum_{j=1}^{|y^h|} \log \left( P_{\Phi_0 + \Delta\Phi(\Theta) + \Theta_p + \Theta_e}(y^h_j \mid x^h, y^h_{<j}) \right)$$

---

### 3.2. Curriculum Scheduler $p(\tau)$
The scheduler defines the probability of selecting the Hard (Hybrid) Task at training step $\tau$, out of total training time $T$:
$$p(\tau) = \frac{\tau}{T} \quad (0 \le \tau \le T)$$

At each step $\tau$, an indicator function determines the training path:
$$\mathbb{I}(\tau) = \begin{cases} 
1 & \text{Train on Hard Task } (\mathcal{L}_{hard}) \quad \text{with probability } p(\tau) \\
0 & \text{Train on Easy Task } (\mathcal{L}_{easy}) \quad \text{with probability } 1 - p(\tau)
\end{cases}$$

This linear progression allows the LLM to first warm up and master the text recommendation instructions, before gradually aligning the continuous behavioral coordinate space.

---

## Related Concepts
*   [[wiki/research/LLaRA Summary.md|LLaRA Research Paper Summary]]
*   [[wiki/entities/SASRec.md|SASRec (Self-Attention Sequential Recommendation)]]
