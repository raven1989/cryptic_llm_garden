---
tags: ["entity", "architecture", "recommendation", "attention", "CTR", "long-sequence"]
aliases: ["TWIN", "TWo-stage Interest Network"]
date: 2026-08-21
sources: ["[[wiki/research/TWIN Summary.md]]"]
---

# TWIN (TWo-stage Interest Network)

**TWIN (TWo-stage Interest Network)** is a lifelong user behavior modeling framework published at KDD 2023 by Kuaishou Technology. It scales Multi-Head Target Attention (MHTA) from sequence length $10^2$ to $10^4$–$10^5$ by making the two cascaded stages — **CP-GSU (Consistency-Preserved General Search Unit)** and **ESU (Exact Search Unit)** — use the **identical relevance metric in both structure and parameters**, solving the GSU–ESU inconsistency problem of prior two-stage models ([[SIM]], ETA, SDIM).

---

## Core Architectural Pillars

### 1. Consistency-Preserved Two-Stage Cascade ("Twins")
Both stages share **one single set of parameter values** for the entire scoring function — not two copies, no distillation, no periodic copying. Gradients from the joint end-to-end training update both stages simultaneously.

| Parameter | Role | Shared? |
| :--- | :--- | :--- |
| $W^q$ (per head) | query projection of target | ✅ identical values |
| $W^h$ (per head) | inherent-feature projection | ✅ identical values |
| $W^c$ (per head) | cross-feature projection (block-diagonal) | ✅ identical values |
| $\boldsymbol{\beta}$ | cross-feature importance weights | ✅ identical values |
| Embeddings $E_A$ | feature embeddings | ✅ shared (end-to-end) |

Differences exist only outside the scoring formula: softmax (ESU only), value/output projections $W^v, W^o$ (ESU only), and the unsplit value matrix $K$ (ESU only). Ablation: structure consistency contributes more than parameter consistency, but both help (SIM Soft < TWIN w/o Para-Cons < TWIN).

### 2. Behavior Feature Splitting (by Cacheability)
The behavior feature matrix is split column-wise: $K = [K_h \quad K_c]$.

- **$K_h \in \mathbb{R}^{L \times H}$ — inherent features** (video id, author, topic, duration): shared across users → projection $K_h W^h$ **pre-computed & cached**, gathered in $O(L)$ independent of $H$.
- **$K_c \in \mathbb{R}^{L \times C}$ — user-item cross features** (click timestamp, play time, position): not shared → each of $J$ features compressed to **one dimension** via per-feature vectors $\mathbf{w}_j^c \in \mathbb{R}^8$, equivalent to restricting $W^c$ to a **block diagonal matrix** (off-diagonal zeros forbid cross-feature interactions). Cost: $O(L \cdot C)$ with $C \ll H$.

### 3. Cross Features as Bias in the Attention Score
The single unified relevance metric used in both stages:

$$\boldsymbol{\alpha} = \frac{(K_h W^h)(\mathbf{q}^\top W^q)^\top}{\sqrt{d_k}} + (K_c W^c)\,\boldsymbol{\beta}$$

Standard scaled dot-product attention on inherent features, plus cross features entering as **additive bias terms** (learnable weights $\boldsymbol{\beta} \in \mathbb{R}^J$). Ablations: removing cross features (w/o Bias) significantly hurts; using the full unsplit projection (w/ Raw MHTA) gains nothing but is far slower.

### 4. Stage-Specific Usage of $\boldsymbol{\alpha}$
- **CP-GSU** ($L = 10^4$): raw $\boldsymbol{\alpha}$ → hard top-100 (4 heads recursively traversed to 100 unique behaviors). No softmax, no value projection.
- **ESU** ($L = 100$): $\text{Softmax}(\boldsymbol{\alpha})^\top KW^v$ per head, then $\text{TWIN} = \text{Concat}(\text{head}_1..\text{head}_4)W^o$. Value projection uses unsplit $K$.

### 5. Single-Loss Training
Only standard binary cross-entropy on the final CTR prediction — **no auxiliary retrieval loss, no consistency loss**. The gradient flows through ESU's differentiable $\boldsymbol{\alpha}$ computation to the shared $W^q, W^h, W^c, \beta$, implicitly training CP-GSU's retrieval. (Config: AdaGrad lr 0.05 for embeddings, Adam lr 5e-06 for DNN, batch 8192.)

### 6. System Co-Design
- **Nearline training:** 46B logs/day, model updated < 8 min, params synced every 5 min.
- **Offline inferring:** inherent feature projector refreshes $K_h W^h$ for an 8B-video pool every 15 min; embedding server covers 97% of requests.
- **Online serving:** cached lookup + realtime $\boldsymbol{\alpha}$ → top-100 → realtime ESU. Bottleneck reduced by **99.3%**. Serves 346M DAU, 30M videos/sec peak. The 15-min cache staleness is why consistency measures 94/100 instead of 100/100.

---

## Key Properties

| Property | Detail |
| :--- | :--- |
| **Max Sequence Length** | $10^4$–$10^5$ (GSU input); 100 (ESU input) |
| **Consistency** | Structure ✅ + Parameters ✅ (first to achieve both) |
| **Relevance Metric** | Identical MHTA in both stages (the "twin" property) |
| **Feature Split** | Inherent $K_h$ (cached) + Cross $K_c$ (1-dim bias) |
| **Cross-Feature Projection** | Block diagonal $W^c$, per-feature $\mathbf{w}_j^c \in \mathbb{R}^8$ |
| **Loss** | Single binary cross-entropy, no auxiliary objectives |
| **Heads** | 4 (CP-GSU merges to 100 unique; ESU concatenates) |
| **Consistency (measured)** | 94/100 Oracle hits (vs. SIM Hard's 40/100) |
| **Deployment** | 346M DAU, 30M videos/sec, 99.3% bottleneck reduction |

---

## Experimental & Business Highlights

- **Offline (46B-scale Kuaishou dataset):** AUC $0.7962$ (+0.29%), GAUC $0.7336$ (+0.51%) vs. best baseline SIM Soft. (0.05% is already business-significant at this scale.)
- **Consistency:** 94/100 Oracle hits; the 6% gap is the 15-min cache refresh delay.
- **Scaling:** advantage over baselines *grows* with behavior sequence length.
- **Online A/B (Watch Time):** vs. SIM Hard: +4.9% / +3.7% / +6.2% (Featured / Discovery / Slide tabs); vs. SIM Soft: +2.8% / +1.4% / +2.7%. (0.1% is significant at Kuaishou.)

---

## Related Wiki Pages
* [[TWIN Summary]]: Complete research summary with worked examples, derivations, and ablation details.
* [[SIM]]: Predecessor two-stage architecture (inconsistent GSU) that TWIN improves upon.
* [[SIM Summary]]: Research summary of SIM.
* [[DIN]]: Origin of Target Attention for short sequences.
* [[DIN Summary]]: Research summary of DIN.
* [[STCA]]: ByteDance's linear-time target cross-attention alternative.
* [[Why is Attention divided by Root d_k]]: The $\sqrt{d_k}$ scaling in TWIN's relevance score.
