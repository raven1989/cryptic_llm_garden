---
tags:
  - "research"
  - "recommendation-systems"
  - "sequential-modeling"
  - "scaling-laws"
aliases:
  - "STCA Summary"
  - "Douyin STCA"
  - "Make It Long, Keep It Fast"
date: 2026-07-10
sources: ["[[raw/Recommendation/字节抖音STCA-万级长序列End2End建模.md]]"]
---

# Douyin STCA Summary (Make It Long, Keep It Fast)

This page details **STCA (Stacked Target-to-History Cross Attention)** and **RLB (Request Level Batching)**, a highly innovative, production-proven system developed by ByteDance/Douyin to scale end-to-end user behavior sequence modeling up to **10,000+ interactions** at billion-user scale.

The paper demonstrates that when the recommendation architecture supports true end-to-end long-sequence training and inference, the predictive performance scales predictably with both context length and capacity—resembling the **scaling laws** observed in Large Language Models (LLMs).

---

## 1. System Overview & Core Challenge

Traditional sequential recommender systems rely on a two-stage extraction paradigm (e.g., SIM or TWIN) due to the $O(L^2)$ time/space complexity of Transformer self-attention. This heuristic sub-sequence extraction breaks end-to-end gradient optimization. 

The Douyin system solves this and enables end-to-end $O(L)$ modeling through three main pillars:
1. **Stacked Target-to-History Cross Attention (STCA):** Linear-in-$L$ cross-attention layers.
2. **Request Level Batching (RLB):** User-centric layout that amortizes user history sequence transfer and encoding.
3. **Extrapolation-Aware Training (EAT):** Training on sparse sequences (average ~2k) but serving on dense sequences (up to 10k).

![STCA Architecture Overview](overview_of_STCA.png)

```text
========================================================================================
                              OVERALL STCA ARCHITECTURE
========================================================================================

 [Historical User Behaviors]                            [Target Candidate Video]
   (e.g., v_1, ..., v_L)                                         (x_t)
            │                                                      │
            ▼                                                      ▼
  ┌──────────────────┐                                   ┌──────────────────┐
  │    Embedding     │                                   │    Embedding     │
  └─────────┬────────┘                                   └─────────┬────────┘
            │ (X in R^{L x d})                                     │ (x_t in R^d)
            ▼                                                      ▼
  ┌──────────────────┐                                   ┌──────────────────┐
  │  SwiGLUFFN Layer │                                   │  SwiGLUFFN Layer │
  ├──────────────────┤                                   ├──────────────────┤
  │    LayerNorm     │                                   │    LayerNorm     │
  └─────────┬────────┘                                   └─────────┬────────┘
            │                                                      │
            │ (ã_X in R^{L x d})                                   │ (q^(1) in R^d)
            │                                                      ▼
            │                                            ┌──────────────────┐
            ├───────────────────────────────────────────>│     STCA Layer 1 │
            │                                            │                  │
            │                                            │  Q = q^(1)       │
            │                                            │  K, V = ã_X      │
            │                                            └─────────┬────────┘
            │                                                      │
            │                                                      │ (o^(1) in R^d)
            │                                                      ▼
            │                                            ┌──────────────────┐
            │                                            │ Target-Cond.     │
            │                                            │ Query Update     │ ── (Fused with x_t)
            │                                            └─────────┬────────┘
            │                                                      │ (q^(2) in R^d)
            │                                                      ▼
            │                                            ┌──────────────────┐
            ├───────────────────────────────────────────>│     STCA Layer 2 │
            │                                            │                  │
            │                                            │  Q = q^(2)       │
            │                                            │  K, V = ã_X      │
            │                                            └─────────┬────────┘
            │                                                      │
            │                                                      │ (o^(2) in R^d)
            │                                                      ▼
            │                                                     ...
            │                                            (Stacked up to Layer M)
            │                                                     ...
            │                                                      │
            │                                                      │ (o^(M) in R^d)
            │                                                      ▼
            │                                            ┌──────────────────┐
            └───────────────────────────────────────────>│     STCA Layer M │
                                                         └─────────┬────────┘
                                                                   │ (o^(M) in R^d)
                                                                   ▼
                                                         ┌──────────────────┐
                                                         │ Layer Summaries  │
                                                         │   Aggregation    │ <── (Fuses o^(1)...o^(M)
                                                         └─────────┬────────┘      and target x_t)
                                                                   │ (z in R^d)
                                                                   ▼
 [Auxiliary User Tokens] ────────────────────────┐       ┌──────────────────┐
    (e.g., profiles)                             ├──────>│   Concat Block   │
                                                 │       └─────────┬────────┘
 [Candidate Side Tokens] ────────────────────────┘                 │ (X_mix)
    (e.g., modalities)                                             ▼
                                                         ┌──────────────────┐
                                                         │    RankMixer     │
                                                         ├──────────────────┤
                                                         │  Sigmoid Output  │
                                                         └─────────┬────────┘
                                                                   │
                                                                   ▼
                                                            [Predict CTR/CVR]
                                                                (ŷ in 0~1)
========================================================================================
```

### Highway Query Generator Detail & Input Dimensionality Growth
Since the input vector to the query generator is a concatenation of all previous layers' outputs, its input dimension expands linearly by $d$ at each step, passing through sequential projection and SwiGLUFFN blocks:

```text
========================================================================================
                      STCA QUERY INPUT DIMENSIONALITY GROWTH
========================================================================================

 LAYER  1   ┌─────────┐
 Query      │   x_t   │  ◄── Dimension: 1 * d
 Input      └─────────┘
                 │
                 ▼  [STCA Layer 1 Block]
                 │
                 ▼
              o^(1) (Dimension: d)

────────────────────────────────────────────────────────────────────────────────────────

 LAYER  2   ┌─────────┬─────────┐
 Query      │  o^(1)  │   x_t   │  ◄── Dimension: 2 * d
 Input      └─────────┴─────────┘
                 │
                 ▼  [Projected by W_C^(2) (2d x d) -> SwiGLU & LN]
                 │
                 ▼
              o^(2) (Dimension: d)

────────────────────────────────────────────────────────────────────────────────────────

 LAYER  3   ┌─────────┬─────────┬─────────┐
 Query      │  o^(1)  │  o^(2)  │   x_t   │  ◄── Dimension: 3 * d
 Input      └─────────┴─────────┴─────────┘
                 │
                 ▼  [Projected by W_C^(3) (3d x d) -> SwiGLU & LN]
                 │
                 ▼
              o^(3) (Dimension: d)

────────────────────────────────────────────────────────────────────────────────────────

   ...      ... (Repeats for layers 4 to M) ...

────────────────────────────────────────────────────────────────────────────────────────

 LAYER M+1  ┌─────────┬─────────┬───┬─────────┬─────────┐
 (Final     │  o^(1)  │  o^(2)  │...│  o^(M)  │   x_t   │  ◄── Dimension: (M + 1) * d
 Prediction └─────────┴─────────┴───┴─────────┴─────────┘
 Input)          │
                 ▼  [Projected by W_Z ((M+1)d x d) -> SwiGLU] -> Final z
========================================================================================
```

---

## 2. Deep Dive: Stacked Target-to-History Cross Attention (STCA)

### 2.1 Preprocessing and Feature Refinement (Section 3.1.1)
Both the historical sequence $\mathbf{X} \in \mathbb{R}^{L \times d}$ and the target video embedding $\mathbf{x}_t \in \mathbb{R}^d$ are projected using highly expressive **SwiGLUFFN** blocks followed by LayerNorm (LN):

$$\text{SwiGLUFFN}(\mathbf{x}) = \left( (\mathbf{x} \mathbf{W}_u) \odot \text{Swish}(\mathbf{x} \mathbf{W}_v) \right) \mathbf{W}_o$$

*   **History Sequence Preprocessing:** 
    $$\tilde{\mathbf{X}} = \text{LN}\left( \text{SwiGLUFFN}(\mathbf{X}) \right) \in \mathbb{R}^{L \times d}$$
*   **Target Query Preprocessing:** 
    $$\mathbf{q}^{(1)} = \text{LN}\left( \text{SwiGLUFFN}(\mathbf{x}_t) \right) \in \mathbb{R}^d$$

Switching standard Feed-Forward Networks (FFN) to SwiGLUFFN yielded a massive **$+0.11\%$ AUC improvement** in offline ablation.

### 2.2 Target-to-History Multi-head Cross Attention
Instead of history-to-history self-attention, STCA uses the target candidate vector as the **sole Query ($Q$)**, with the history matrix acting as the static **Keys ($K$) and Values ($V$)**. For head $j$ of dimension $d_h = d/h$ at layer $i$:

*   **Attention Weights (Similarity Match):**
    $$\alpha^{(i,j)} = \text{softmax} \left( \frac{\mathbf{q}^{(i)} \mathbf{W}_Q^{(i,j)} \left( \tilde{\mathbf{X}}^{(i)} \mathbf{W}_K^{(i,j)} \right)^\top}{\sqrt{d_h}} \right) \in \mathbb{R}^{1 \times L}$$
*   **Head Value Aggregation:**
    $$\mathbf{o}^{(i,j)} = \alpha^{(i,j)} \left( \tilde{\mathbf{X}}^{(i)} \mathbf{W}_V^{(i,j)} \right) \in \mathbb{R}^{1 \times d_h}$$
*   **Head Concatenation & Projection:**
    $$\mathbf{o}^{(i)} = \left[ \mathbf{o}^{(i,1)} \parallel \cdots \parallel \mathbf{o}^{(i,h)} \right] \mathbf{W}_O^{(i)} \in \mathbb{R}^d$$

This reduces complexity per layer from quadratic $O(L^2 d_h)$ to strictly linear **$O(L d_h)$**.

### 2.3 Layer Stacking & Highway Query Update (Section 3.1.2)
To capture multi-hop, higher-order dependencies without direct history-to-history communication, the model stacks $M$ layers. 

The query vector for layer $i+1$ is updated by concatenating **all preceding layers' outputs** along with the original target $\mathbf{x}_t$, projected via $\mathbf{W}_C^{(i+1)}$ and normalized:

$$\mathbf{q}^{(i+1)} = \text{LN}\left( \text{SwiGLUFFN}^{(i+1)}\left( \left[ \mathbf{o}^{(1)} \parallel \cdots \parallel \mathbf{o}^{(i)} \parallel \mathbf{x}_t \right] \mathbf{W}_C^{(i+1)} \right) \right)$$

This dynamic highway query design lets the search query progressively adapt based on earlier target-history findings.

#### Query Generator Input Dimensionality Growth
Since the input vector to the query generator is a concatenation of previous layers, its input dimension expands linearly by $d$ at each step:

```text
Layer 1 Query Input: [ x_t ]                         ──► Shape: [d]
Layer 2 Query Input: [ o^(1) || x_t ]                 ──► Shape: [2d]
Layer 3 Query Input: [ o^(1) || o^(2) || x_t ]         ──► Shape: [3d]
...
Layer M Query Input: [ o^(1) || ... || o^(M-1) || x_t ] ──► Shape: [Md]
```
The query projection matrix $\mathbf{W}_C^{(i+1)}$ correspondingly scales up to shape $(i+1)d \times d$ to compress it back to $d$ for LayerNorm and attention.

### 2.4 Final Output Fusion & Downstream Prediction (Section 3.1.3)
Once all $M$ layers of attention are complete, the final target-aware token $\mathbf{z}$ is constructed by concatenating all $M$ layer summaries plus the target video, passing through a SwiGLU block **without LayerNorm**:

$$\mathbf{z} = \text{SwiGLUFFN}\left( \left[ \mathbf{o}^{(1)} \parallel \cdots \parallel \mathbf{o}^{(M)} \parallel \mathbf{x}_t \right] \mathbf{W}_Z \right)$$

The omission of LayerNorm is crucial to preserve the raw intensity/magnitude of the matches. This $\mathbf{z}$ is then merged with auxiliary user tokens $\{\mathbf{u}_k\}$ and candidate-side tokens $\{\mathbf{c}_\ell\}$ before entering **[[RankMixer Summary|RankMixer]]** for CTR/CVR score prediction:

$$\mathbf{X}_{\text{mix}} = \text{concat}\left(\mathbf{z}, \{\mathbf{u}_k\}_{k=1}^K, \{\mathbf{c}_\ell\}_{\ell=1}^C\right)$$
$$\mathbf{h} = \text{RankMixer}\left(\mathbf{X}_{\text{mix}}\right), \quad \hat{y} = \text{sigmoid}(\mathbf{w}^\top \mathbf{h} + b)$$

---

## 3. High-Performance Algebra: Single-Query Optimization (Section 3.1.4)

In standard attention, projecting history to compute Key ($\mathbf{K} = \mathbf{X}\mathbf{W}_K \in \mathbb{R}^{L \times d_h}$) and Value ($\mathbf{V} = \mathbf{X}\mathbf{W}_V \in \mathbb{R}^{L \times d_h}$) requires materializing these intermediate matrices of size $L \times d_h$ in GPU High Bandwidth Memory (HBM). When $L=10k$, this incurs extreme memory bandwidth bottlenecks.

Because the Query $\mathbf{q}$ is 1-dimensional ($1 \times d$), STCA applies the associative law of matrix multiplication to reorder the projections:

$$\text{Attn}(\mathbf{q}, \mathbf{X}) = \left[ \text{softmax} \left( \frac{\left[ (\mathbf{q} \mathbf{W}_Q) \mathbf{W}_K^\top \right] \mathbf{X}^\top}{\sqrt{d_h}} \right) \mathbf{X} \right] \mathbf{W}_V = (\mathbf{\alpha} \mathbf{X}) \mathbf{W}_V$$

### How this saves HBM:
1.  **Original Order:** Computes $\mathbf{X} \mathbf{W}_K$ first, generating an $L \times d_h$ Key matrix in GPU memory.
2.  **STCA Optimized Order:** Computes $(\mathbf{q}\mathbf{W}_Q)\mathbf{W}_K^\top$ first. Since $(\mathbf{q}\mathbf{W}_Q)$ is just $1 \times d_h$, multiplying it by $\mathbf{W}_K^\top$ ($d_h \times d$) produces a tiny $1 \times d$ vector.
3.  **Result:** The giant $L \times d_h$ key and value matrices are **never materialized** in memory. Only lightweight intermediate vectors of size $1 \times d$ and the attention vector of size $1 \times L$ are stored, dramatically reducing HBM load.

---

## 4. Request Level Batching (RLB) (Section 3.2)

When scoring a request containing one user and $m$ candidate items, pointwise models duplicate the user history sequence $\mathcal{H}$ of length $L$ across $m$ samples, wasting bandwidth (CPU ──► GPU) and repeating identical computations.

**RLB** restructures the training/inference batch layout: it loads and encodes the user history sequence $\mathcal{H}$ **exactly once** per request and reuses it as the key/value background for all $m$ candidate targets.

### 4.1 Tensor Layout Comparison
*   **Point-wise Batch Layout:**
    *   User History Tensor: $[B \times m, L, d]$ (duplicated)
    *   Target Video Tensor: $[B \times m, 1, d]$
    *   Output Shape: $[B \times m, 1]$
*   **RLB Batch Layout:**
    *   Shared User History Tensor: $[B, L, d]$ (loaded once)
    *   Batched Target Videos Tensor: $[B, m, d]$
    *   Output Shape: $[B, m]$ (where each target preserves its independent, unmodified labels $y_k \in \{0, 1\}$)

### 4.2 Mathematical Unbiasedness
By rearranging the BCE objective as a user-level average of target losses, RLB remains an **unbiased estimator** of standard empirical risk. It changes only the execution structure on hardware, preserving convergence properties.

### 4.3 Production Benefits
*   **77% to 84% Bandwidth Reduction:** Cuts PCIe/NVLink inter-module data traffic.
*   **5.1× Training Throughput:** Combining RLB with custom GPU kernels yields immense speedups.
*   **50% PS Relief:** Parameter Server CPU and network bandwidth overhead dropped by half.

---

## 5. Length Extrapolation: Train Sparsely, Infer Densely (Section 3.3)

To avoid the extreme cost of training on massive 10,000-length sequences, the authors train on variable-length histories averaging **~2k tokens**, but serve on dense histories up to **10k tokens** online.

### 5.1 Randomized Length Sampling
For each user sequence during training, a random length $L_{\text{train}}$ is sampled via a **Beta Distribution**:

$$L_{\text{train}} = L_{\text{train}}^{\text{min}} + s \cdot \left( L_{\text{train}}^{\text{max}} - L_{\text{train}}^{\text{min}} \right), \quad s \sim \text{Beta}(\alpha, \beta)$$

To trigger hardware Tensor Core acceleration, $L_{\text{train}}$ is rounded to the nearest multiple of 8. A **U-shaped Beta distribution** yielded the highest offline AUC.

### 5.2 Temporal Suffix Sampling
Once $L_{\text{train}}$ is drawn, the system selects the $L_{\text{train}}$ **most recent** user interactions (Temporal Suffix). In recommendations, recency prior carries the strongest preference signal. 
*   *Performance:* Keep Suffix (most recent) > Random > Keep Prefix (oldest).

### 5.3 Batch-Level Load Balancing & Offsets Tracker
Different users in a batch have highly variable random lengths $L_{\text{train}}$, which traditionally forces wasteful zero-padding. To eliminate padding:
1.  **Flat Active Array:** Concatenates all active user history tokens into a single contiguous flat 1D tensor $\mathbf{X}_{\text{flat}}$.
2.  **Offsets Tracker Array:** An auxiliary offset array (e.g. `[0, 1920, 2432, 5932]`) tracks sequence boundaries.
3.  **Ragged Target Attention:** Customized GPU attention kernels use these offsets to enforce perfect sequence isolation within the flattened array, achieving near 100% compute efficiency.

---

## 6. Real-World Gains & Scaling Laws

### 6.1 Scaling Laws
STCA demonstrates a predictable, monotonic scaling relationship (shown in Figure 1). Scaling context length ($500 \rightarrow 10k$) and model capacity (Simple: $6M$, Medium: $23M$, Complex: $133M$) yields continuous gains:
*   At $10k$ length, scaling capacity to complex results in a **$+1.68\%$ Finish AUC lift** over short-sequence baselines.

### 6.2 Serving Extrapolation (Table 3)
Training on sequences averaging $2k$ while serving at longer lengths achieves predictable quality extrapolation:
*   Serve at $2k$: $+0.03\%$ AUC lift.
*   Serve at $4k$: $+0.09\%$ AUC lift.
*   Serve at $10k$: **$+0.21\%$ AUC lift** (zero extra training cost!).

---

## 7. Ablation Summary (Table 2)

| Component Ablated | Δ AUC Lift | Notes |
| :--- | :--- | :--- |
| **Enlarge Video-ID Embeddings** | $+0.08\%$ | 128 ──► 320 width. Requires small initialization range to stabilize. |
| **FFN Depth: 2 ──► 4** | $+0.18\%$ | Deepens the token-wise history path; doubles STCA depth. |
| **FFN ──► SwiGLU** | $+0.11\%$ | Upgrades standard FFN layers to SwiGLU gating blocks. |
| **Attention Heads: 8 ──► 16** | $+0.05\%$ | Improves target-conditioned selectivity at a modest cost. |
| **Query Fusion (Eq. 7)** | **$+0.06\%$** | Validates the highway concatenation: $[\mathbf{o}^{(1)} \dots \mathbf{o}^{(i)}, \mathbf{x}_t]$. |
| **Time-delta Side Info** | $+0.08\%$ | Adds recency prior (current request time minus item interaction timestamp). |
