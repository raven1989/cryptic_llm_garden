---
title: "HSTU: Hierarchical Sequential Transduction Unit"
tags:
  - "entity"
  - "architecture"
  - "attention"
aliases:
  - "HSTU"
  - "Meta GRs"
  - "Generative Recommenders"
date: 2026-07-06
sources: ["[[raw/Recommendation/Meta GRs : 万亿参数级别的生成式推荐.md]]"]
---

# HSTU (Hierarchical Sequential Transduction Unit) & Meta GRs

The **Hierarchical Sequential Transduction Unit (HSTU)** is a high-performance encoder architecture developed by Meta to power trillion-parameter **Generative Recommenders (GRs)**, introduced in the ICML 2024 paper *"Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"*. 

By treating recommendation as a Sequence-to-Sequence (seq2seq) transduction task, it redefines how heterogenous features are processed and scaled.

---

## 1. Input & Feature Space Organization (Section 2.1)

Unlike traditional DLRMs that send disjoint features through messy preprocessing pipelines, Generative Recommenders format all inputs into a **single, chronologically interleaved timeline**.

![Comparison of features and training procedures: DLRMs vs GRs](wiki/media/HSTU-fig2_Comparison_of_features_and_training_procedures_DLRMs_vs_GRs.png)

### A. Overview of the Interleaved Input Sequence
The unified model processes a homogeneous tensor representing a sequential history. The final input sequence has the following structure:

$$\text{Sequence} = [ G_0, \Phi_0, a_0, \Phi_1, a_1, \dots, H_0, \dots, \Phi_i ]$$

Within this single timeline:
*   $\Phi_i$ represents the high-cardinality **content IDs** (e.g., video/post tokens).
*   $a_i$ represents the user's multi-hot **actions** (e.g., `like`, `skip`, `watch complete`).
*   $G_j, H_k$ represent auxiliary **categorical metadata** (e.g., demographics, followed creators).
*   All features are mapped through dedicated lookup tables in the **Embedding Layer** and projected via **optional Local MLPs** into a unified hidden space of dimension $d$ (producing a clean `(Batch, Sequence, d)` tensor for HSTU).

### B. Categorical (Sparse) Features
To prevent categorical metadata from overwhelming the timeline and bloating sequence lengths:
*   **Compression:** Slowly changing attributes (e.g., user location or demographics) undergo **earliest-entry compression**. The system retains only the first occurrence of a consecutive state and drops subsequent identical entries.
*   **Chronological Merging:** These compressed tokens are merged into the main content-interaction timeline using their exact physical timestamps.
*   **Causal Masking:** Static demographics cannot be predicted or recommended. During training, their expected outputs are marked as undefined ($\emptyset$), and their loss masks $m_i$ are set to `0`. They act strictly as context for subsequent content tokens.

### C. Continuous (Dense) Features (Eliminated)
Traditional DLRM features such as hand-engineered counter statistics, historical click-through rates (CTR), and decay metrics are **completely abandoned**:
*   **Internal Reconstruction:** Rather than computing and logging complex sliding-window counts in real time, GRs rely on the sequence window ($N$). Since raw categorical interactions and timestamps are fully preserved in the timeline, the self-attention mechanism within HSTU organically learns to compute its own internal decay, counting, and ratio representations.

### D. Integration of Embedding Tables
All embedding tables and localized projection MLPs are **fully integrated within the unified model** instead of being separated:
*   **Sparse Parameters:** Over 99% of the 1.5T parameter capacity lies in these integrated embedding tables (primarily the massive, dynamic billion-scale content ID table).
*   **Dense Parameters:** The remaining parameters represent the dense HSTU weights and local MLP projections.
*   **End-to-End Optimization:** By integrating these layers within a single model architecture, gradients propagate backward end-to-end directly from final sequential loss functions to individual item embedding states.

---

## 2. Sequential Transduction Formulation (Section 2.2)

Rather than treating recommendation as separate discriminative classifier tasks, Section 2.2 reformulates **Retrieval** and **Ranking** as causal autoregressive Sequence-to-Sequence (seq2seq) transduction tasks mapped onto the interleaved temporal timeline.

### Table 1: Task Specifications (Inputs & Outputs)

| Stage | Input Sequence ($x_i$) | Expected Output Sequence ($y_i$) |
| :--- | :--- | :--- |
| **Ranking** | $\Phi_0, a_0, \Phi_1, a_1, \dots, \Phi_{n_c-1}, a_{n_c-1}$ | $a_0, \emptyset, a_1, \emptyset, \dots, a_{n_c-1}, \emptyset$ |
| **Retrieval** | $(\Phi_0, a_0), (\Phi_1, a_1), \dots, (\Phi_{n_c-1}, a_{n_c-1})$ | $\Phi'_1, \Phi'_2, \dots, \Phi'_{n_c-1}, \emptyset$ |

*(Where $\Phi_i$ is a content ID, $a_i$ is a multi-hot action vector, and $\emptyset$ represents an undefined target).*

### Task Mechanics & Target-Masking Operations

Both tasks utilize an identical mathematical masking scheme during sequence loss optimization. A binary mask $m_i \in \{0, 1\}$ filters out undefined targets ($\emptyset$) during backpropagation:

$$\mathcal{L} = -\sum_{i=0}^{n-1} m_i \log P(y_i \mid u_i)$$

#### **A. The Ranking Task (Predicting Actions)**
*   **The Goal:** Predict the user's action $a_k$ (e.g., skip vs. click) directly following a candidate content item presentation $\Phi_k$.
*   **Target-Aware Advantage:** By interleaving content and action tokens sequentially ($[\Phi_0, a_0, \Phi_1, a_1 \dots]$), the candidate item acts as causal context in the timeline. The encoder can perform **target-aware cross-attention** within its very first layers, bypassing late-stage attention limitations of traditional architectures.
*   **Masking Rules:** 
    *   When $x_i = \Phi_k$, the target $y_i = a_k$ and the mask $m_i = 1$ (loss is computed normally).
    *   When $x_i = a_k$, the target $y_i = \emptyset$ and the mask $m_i = 0$. Gradients are multiplied by $0$, meaning the model is never penalized for what it predicts after an action token.

#### **B. The Retrieval Task (Predicting Next Content)**
*   **The Goal:** Predict the next target content item $\Phi'_{i+1}$ that the user is likely to engage with.
*   **The Positive-Only Constraint:** To ensure the retrieval model only learns to source items the user actually liked, targets are filtered dynamically based on actions:
    $$\Phi'_{i+1} = \begin{cases} \Phi_{i+1} & \text{if } a_{i+1} \text{ is a positive action (e.g., watch completion, share, like)} \\ \emptyset & \text{if } a_{i+1} \text{ is a negative action (e.g., skip, hide)} \end{cases}$$
*   **Masking Rules:**
    *   If $a_{i+1}$ is positive, the target $y_i = \Phi_{i+1}$ and the mask $m_i = 1$.
    *   If $a_{i+1}$ is negative, the target $y_i = \emptyset$ and the mask $m_i = 0$. The model's gradients at this step are nullified, ensuring skipped/disliked items do not pollute the retrieval latent space.

### B.2 Discussion: Generative vs. Discriminative Modeling

Appendix B.2 outlines the theoretical advantage of moving from traditional discriminative sequential frameworks to joint generative modeling.

*   **Traditional Discriminative Models (e.g., SASRec, GRU4Rec):**
    These models calculate a conditional next-item probability:
    $$P(\Phi_i \mid \Phi_0, a_0, \dots, \Phi_{i-1}, a_{i-1})$$
    They ignore the dual-sided nature of modern recommendations—namely, the recommendation system suggesting content and the user reacting to it.
*   **Generative Recommenders (GRs):**
    GRs model the **joint probability distribution** of both the suggested contents and user actions over the entire timeline:
    $$P(\Phi_0, a_0, \Phi_1, a_1, \dots, \Phi_{n_c-1}, a_{n_c-1})$$
    
This paradigm shift yields two major clinical advantages:
1.  **List-wise Generation:** Modeling the joint distribution allows the system to generate and optimize a sequence of recommendations simultaneously using **Beam Search**, rendering handcrafted diversity heuristics (like DPP) obsolete.
2.  **Early Target-aware Fusion:** Natively interleaving content and action tokens allows the self-attention matrices within HSTU to fuse target information with precise past interaction states in the early layers, vastly outperforming late-interaction decoders.

---

## 3. Generative Training & Sequence-Level Sampling (Section 2.3)

In high-throughput streaming environments, training long-sequence self-attention models on billions of user logs poses a severe quadratic compute wall. Section 2.3 resolves this bottleneck by implementing a dual-mechanism optimization.

### A. How Generative Training Cuts Complexity by $O(N)$
Traditional DLRMs use **impression-level logging**, treating every single engagement as a standalone training example. For a user history of length $N$, this requires running the encoder $N$ separate times, leading to cubic time complexity:

$$\text{Impression-level compute} \approx \sum_{k=1}^N (k^2 d + k d^2) \approx O(N^3 d + N^2 d^2)$$

Generative Recommenders switch to **Generative Training**, packaging the user's entire sequence $[ \Phi_0, \Phi_1, \dots, \Phi_{N-1} ]$ and feeding it to the causally-masked HSTU encoder **exactly once**. 
*   By calculating prediction losses across all $N$ targets in a single forward/backward pass, the intermediate sequence representations are shared.
*   This shifts the time complexity per user directly to **$O(N^2 d + N d^2)$**, yielding an entire **$O(N)$ factor speedup**.

### B. How $1/n_i$ Sampling Solves User Workload Imbalance
User sequence lengths on large web platforms are heavily skewed (e.g., casual users with $n_i = 10$ vs. power users with $n_i = 1000$).
*   **The Bottleneck:** If processed equally, a power user sequence costs $1000^2$ (1M ops), which is **10,000 times more attention compute** than a casual user ($10^2 = 100$ ops). Power users would choke the GPU training cluster.
*   **The Math:** By downsampling users with a rate $s_u(n_i)$ inversely proportional to their sequence length:
    $$s_u(n_i) = \frac{1}{n_i}$$
    The expected compute cost of any user scales linearly:
    $$\text{Expected Cost} = s_u(n_i) \times n_i^2 = \frac{1}{n_i} \times n_i^2 = n_i$$
    this reduces the power-user to casual-user compute ratio from a quadratic **$10,000\times$** down to a highly manageable linear **$100\times$**.
*   **Engineering Implementation:** To achieve this effortlessly, the platform's logging system only emits training examples **at the very end of a user's request or session**. This naturally and implicitly produces a session sampling rate of $\hat{s}_u(n_i) \propto 1/n_i$, matching the mathematical ideal without manual sequence filtering.

---

## 4. Backbone Architecture & Micro-Design Layers (Section 3)

The macro-architecture of Generative Recommenders completely replaces the fragmented multi-stage structure of traditional DLRMs (consisting of disjoint Bottom MLPs, Feature Interaction layers, and Top MLPs) with a streamlined stack of identical modular blocks.

![Macro-Architecture Comparison: DLRMs vs HSTU-based GRs](wiki/media/HSTU-fig3_architecture.png)

### Overall Architecture Data Flow Paradigm

The complete end-to-end data flow traces raw interleaved user history tokens down to final multi-task predictions (Resized to max 78-character layout):

```
==============================================================================
                                INPUT TIMELINE
==============================================================================
 Timeline Sequence: [ Demographics (G0) -> Item (Φ0) -> Action (a0) -> Φ1 ]
 Timestamps:        [      t0        ->     t1    ->    t1     ->    t2  ]
==============================================================================
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                           EMBEDDING & PROJECTION                           │
├────────────────────────────────────────────────────────────────────────────┤
│  [Sparse Categorical ID]  [High-Cardinality Item ID]  [Multi-Hot Action]   │
│            │                          │                        │           │
│            ▼                          ▼                        ▼           │
│     ┌─────────────┐            ┌─────────────┐          ┌─────────────┐    │
│     │ G0 Embeds   │            │ Φi Embeds   │          │Action Embeds│    │
│     └──────┬──────┘            └──────┬──────┘          └──────┬──────┘    │
│            │                          │                        │           │
│            ▼                          ▼                        ▼           │
│     ┌─────────────┐            ┌─────────────┐          ┌─────────────┐    │
│     │Local Proj/MLP│           │Local Proj/MLP│          │Local Proj/MLP│    │
│     └──────┬──────┘            └──────┬──────┘          └──────┬──────┘    │
│            │                          │                        │           │
│            └──────────────────────────┼────────────────────────┘           │
│                                       ▼                                    │
│                 Homogeneous Tensor Stream: [Batch, Sequence, d]            │
└───────────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      HSTU BACKBONE (REPEATED L-TIMES)                      │
├────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 1. POINTWISE PROJECTION                                              │  │
│  │    Input (X) ──► [SiLU(W1·X+b1)] ──► Split ──► [ Gating U, V, Q, K ]  │  │
│  │                                                  │      │  │  │      │  │
│  └──────────────────────────────────────────────────┼──────┼──┼──┼──────┘  │
│                                                     │      │  │  │         │
│  ┌──────────────────────────────────────────────────┼──────┼──┼──┼──────┐  │
│  │ 2. SPATIAL AGGREGATION                           │      │  │  │      │  │
│  │    Q, K, Position/Temporal Biases (rab) ─────────┼──────┼──►[+]      │  │
│  │                                                  │      │   │        │  │
│  │                                                  │      │   ▼        │  │
│  │                                                  │      │ [SiLU]     │  │
│  │                                                  │      │   │        │  │
│  │                                                  │      └──►[x]      │  │
│  │                                                  ▼          ▼        │  │
│  │                                                 [U]      [Attn·V]    │  │
│  └──────────────────────────────────────────────────┬──────────┬────────┘  │
│                                                     │          │           │
│  ┌──────────────────────────────────────────────────┼──────────┼──────┐  │
│  │ 3. POINTWISE TRANSFORMATION                      │          │      │  │
│  │                                                  │     [LayerNorm] │  │
│  │                                                  │          │      │  │
│  │                                                  └─────────►[⊙](Gate)  │
│  │                                                             │      │  │
│  │                                                             ▼      │  │
│  │                                                        [f2(Linear)]│  │
│  └─────────────────────────────────────────────────────────────┬──────┘  │
│                                                                │           │
│     Residual Add Connection (Y + X) ◄──────────────────────────┘           │
└───────────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                           MULTI-TASK PREDICTION                            │
├────────────────────────────────────────────────────────────────────────────┤
│                  Final Unified Latent Representation (ui)                  │
│                                     │                                      │
│                  ┌──────────────────┴──────────────────┐                   │
│                  ▼                                     ▼                   │
│        ┌──────────────────┐                  ┌──────────────────┐          │
│        │  RETRIEVAL HEAD  │                  │   RANKING HEAD   │          │
│        ├──────────────────┤                  ├──────────────────┤          │
│        │ Next-Item        │                  │ Multi-Task MLP   │          │
│        │ Softmax Target:  │                  │ Prediction:      │          │
│        │ p(Φi+1 | ui)     │                  │ p(ai+1 | ui)     │          │
│        └──────────────────┘                  └──────────────────┘          │
==============================================================================
```

### HSTU Encoder Block Layers

The primary backbone comprises L residual layers stacking exactly three sub-layers calculated mathematically as:

#### Equation 1: Pointwise Projection (Fusing Input Maps)
$$U(X), V(X), Q(X), K(X) = \text{Split}\Big(\phi_1\big(f_1(X)\big)\Big)$$

*   **Projection:** The input representation $X \in \mathbb{R}^d$ is mapped through a single pointwise linear projection $f_1(X) = W_1 X + b_1$.
*   **Activation:** Applies a **SiLU** non-linear activation $\phi_1$.
*   **Splitting:** Slices the resulting activated tensor channel-wise into four contiguous arrays representing Gating $U(X) \in \mathbb{R}^d$, Content Values $V(X) \in \mathbb{R}^{d_v}$, and attention Queries $Q(X)$ and Keys $K(X) \in \mathbb{R}^{d_{qk}}$.
*   **Unified GEMM Speedup:** This single wide projection pools separate projections into a highly optimized, single-fused GPU matrix-multiplication kernel.

#### Equation 2: Spatial Aggregation (Pointwise Attention Pooling)
$$A(X)V(X) = \phi_2\Big(Q(X)K(X)^T + rab^{p, t}\Big) V(X)$$

*   **Spatial Collapse:** This is the **only** layer component that aggregates historical context along the sequence timeline axis.
*   **Non-Softmax Formulation:** Uses **SiLU** ($\phi_2$) pointwise activation rather than Softmax normalization. This preserves raw interaction frequency intensity (essential for modeling preference intensity) and stabilizes models during high-cardinality streaming training.
*   **Relative Biases:** Direct inclusion of positional ($p$) and temporal elapsed-time ($t$) relative attention biases ($rab^{p,t}$) within spatial calculation.

#### Equation 3: Pointwise Transformation (SwiGLU Gating & FFN Elimination)
$$Y(X) = f_2\Big(\text{Norm}\big(A(X)V(X)\big) \odot U(X)\Big)$$

*   **SwiGLU Gating Integration:** Multiplying the normalized attention output element-wise ($\odot$) with the SiLU-activated gating vector $U(X)$ creates a localized **SwiGLU variant** of Shazeer's gating function, boosting model expressiveness.
*   **MMoE-Style Implicit Gating:** This pointwise gating behaves as a soft routing mechanism similar to Multi-Gate Mixture of Experts, turning off irrelevant feature channels dynamically.
*   **Feed-Forward Network (FFN) Elimination:** By directly fusing pooled context with channel-wise features via the cheap Hadamard product ($\odot$), **HSTU completely discards the heavy Transformer FFN block**. This prunes parameter compute, saving **66% of the layer's activation memory** and enabling deep stacking.

---

## 5. Key Optimization Features

*   **Pointwise Aggregated Attention:** HSTU replaces standard softmax attention. Pointwise pooling avoids soft-max normalization to capture user interaction intensity, which is vital for concurrently predicting value and order.
*   **Stochastic Length Method:** Speeds up training by dynamically truncating long sequence lengths with probability $1 - N^\alpha / n_i^2$, trading minor accuracy for large throughput gains.
*   **Memory Efficiency:** Reduces the number of non-attention linear layers from 6 to 2, heavily fuses mathematical operators into single kernels, and utilizes row-wise AdamW optimizer with state pinning in memory.
*   **Inference Optimization:** Leverages **M-FALCON**, yielding up to $700\times$ speedup at inference time.

---
## References
*   Paper Code: [GitHub - facebookresearch/generative-recommenders](https://github.com/facebookresearch/generative-recommenders)
*   Wiki Index: [[index]]
