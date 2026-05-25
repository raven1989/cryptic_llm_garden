---
title: "Mixture of Experts (MoE) LLMs Summary"
tags:
  - research
  - moe
  - LLM
  - architecture
  - evolution
sources: ["[[raw/LLM/Mixture of Experts Explained.md]]", "[[raw/LLM/Mixture-of-Experts (MoE) LLMs.md]]"]
aliases:
  - MoE LLMs Summary
  - Mixture of Experts Summary
  - Evolution of MoE
date: 2026-05-21
---
# Mixture of Experts (MoE) LLMs Summary

## Introduction & The "Sparsity" Paradigm

In the pursuit of advancing Large Language Models (LLMs), scaling up parameters has consistently yielded superior performance. However, scaling dense architectures carries a linear increase in both computational cost (FLOPs) and training time, eventually colliding with physical hardware limits. 

The **Mixture of Experts (MoE)** architecture circumvents this by introducing **sparsity** through **conditional computation**. Instead of executing all parameters for every input token, the model dynamically routes each token to a specific subset of specialized sub-networks called "experts."

### Resolving the Chinchilla Scaling Law vs. MoE Debate
The Chinchilla scaling laws state that for a compute-optimal dense model, parameters and training tokens should scale proportionally. MoEs bypass this bottleneck:
*   They maintain a massive **total parameter count** (storing rich factual data and semantic capacity).
*   They limit the **active parameter count** per token to a fraction of the total (keeping FLOP compute-requirements low during both pre-training and inference).
*   *Result:* MoEs achieve the representation capacity of massive dense models but pre-train up to **7x faster** and host faster inference relative to their total parameter size.

---

## The Chronological Evolution of MoEs (Connecting the Dots)

The architecture of modern sparse MoEs has been forged by a series of foundational breakthroughs, resolving challenges of scaling, training stability, and hardware layout.


![[wiki/media/moe_evolution_timeline.svg]]


### 1. 1991: Adaptive Mixture of Local Experts (Jacobs et al.)
The root of the concept is introduced as an ensemble-like supervised procedure where separate sub-networks specialize in processing different regions of the input space, adjudicated by a learned gating network.

### 2. 2017: Sparsely-Gated MoE Layer (Shazeer et al.)
The seminal work that scaled conditional computation to deep learning (using LSTMs). It introduced:
*   **Sparsity:** Restricting the gating network to select only the Top-$K$ experts, leaving unselected experts uncomputed.
*   **Noisy Top-$K$ Gating:** Adding tunable Gaussian noise to the gating logits before applying Softmax to assist in early-stage exploration and load balancing.
*   **Auxiliary Importance Loss:** Penalizing unbalanced expert selections to avoid early-stage routing collapse.
*   *Detail Page:* See [[Sparsely-Gated MoE Layer]].

### 3. 2020: GShard (Lepikhin et al.)
The first model to integrate sparse expert layers into the Transformer architecture (replacing every other FFN layer with an MoE block). It contributed:
*   **Top-2 Gating with Random Routing:** The first expert is selected deterministically (highest score); the second is selected probabilistically proportional to its weight to encourage exploration.
*   **Expert Capacity:** The maximum number of tokens allowed to be sent to a single expert, defined statically as $Capacity = (\frac{\text{tokens}}{\text{experts}}) \times \text{Capacity Factor}$. Tokens exceeding this limit are "overflowed" and bypass the expert layer via residual connections.
*   *Detail Page:* See [[GShard]].

### 4. 2021: Switch Transformer (Fedus et al.)
A major simplification scaling models up to 1.6 Trillion parameters. It proved:
*   **Top-1 Gating:** Routing to a single expert preserves quality, reduces router compute overhead, halves expert batch sizes, and lowers communication costs.
*   **Scale-Invariant Balancing Loss:** Simplification of the balancing loss, scaling it by the number of experts $N$ to maintain optimization pressure regardless of architectural scale.
*   **Selective Precision:** Training routers in `float32` (retaining exponentiation precision) while keeping experts in `bfloat16` to stabilize training.
*   *Detail Page:* See [[Switch Transformer]].

### 5. 2022: ST-MoE (Zoph et al.)
Addressed severe pre-training instabilities occurring at the largest scales (such as Switch-XXL). 
*   **Instability Diagnosis:** Large logits passing through `bfloat16` exponential functions create catastrophic rounding errors (since `bfloat16` has up to 65,536x worse roundoff errors than `float32`).
*   **The [[Router Z-Loss]] Solution:** Rather than clipping (which adds discontinuities), they added an auxiliary loss penalizing the squared Log-Sum-Exp of routing logits:
    $$L_z(x) = \frac{1}{B} \sum_{i=1}^B \left(\log \sum_{j=1}^N e^{x_{ij}}\right)^2$$
    This penalizes the absolute magnitude of the maximum logit, keeping values small and roundoff-free.
*   *Detail Page:* See [[ST-MoE Summary]].

### 6. 2024: DeepSeekMoE (Dai et al., DeepSeek-V2 & V3)
Reimagined the internal structures of experts and load-balancing algorithms:
*   **[[DeepSeek Shared Experts|Shared Expert Isolation]]:** Traditional MoE routed experts store highly redundant, general syntactic and contextual knowledge. DeepSeek introduced **Shared Experts** (always active for all tokens) alongside **Routed Experts** (fine-grained and sparsely gated). This isolates general context into shared parameters and allows routed experts to become highly specialized.
*   **[[DeepSeek Load Balancing|Auxiliary-Loss-Free Load Balancing]]:** Standard auxiliary losses contaminate gradients, degrading downstream model performance. DeepSeek-V3 pioneered a dynamic routing bias ($s_{i,t} + b_i$) used *only* during the Top-K selection. The actual gating weight remains the pure, uncorrupted Sigmoid score ($s_{i,t}$), keeping training gradients completely untainted.
*   *Detail Page:* See [[DeepSeek Load Balancing]].

---

## Comparative Matrix of Core MoE Paradigms

| Architecture | Gating Style | Active Experts | Auxiliary Loss Formulation | Key Stability Technique | Core Architectural Contribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sparsely-Gated MoE (2017)** | Softmax (Noisy Top-K) | $K \ge 2$ | Expert Importance (CV of probabilities) | Gaussian Logit Noise | Sparked sparse conditional computation at scale. |
| **GShard (2020)** | Softmax (Random Top-2) | $K = 2$ | Differentiable load balance loss | Static expert capacity limits | First integration with Transformers; invented **Expert Capacity**. |
| **Switch Transformer (2021)** | Softmax (Top-1) | $K = 1$ | Scale-invariant load-balance loss (multiplied by $N$) | Selective Precision (`float32` router, `bf16` experts) | Proved Top-1 routing viability; scaled to 1.6T parameters. |
| **ST-MoE (2022)** | Softmax (Top-1/Top-2) | $K \le 2$ | Combined scale-invariant load balance loss | **[[Router Z-Loss]]** (squared LSE penalty) | Solved mixed-precision exponential roundoff training instabilities. |
| **DeepSeekMoE (V2/V3)** | Sigmoid (Top-K) | $N_s$ (Shared) + $K_r$ (Routed) | V2: Expert, Device, and Communication losses. V3: Sequence-wise balancing | **Auxiliary-Loss-Free Gating Biases** ($b_i \pm \gamma$) | **Shared Expert Isolation**; untainted dynamic routing gradients. |

---

## Critical Trade-offs, Implementation, & Deployment

### 1. Memory Wall vs. FLOP Efficiency (Serving Costs)
While a sparse MoE achieves much faster inference speeds (proportional to its active parameters), serving MoEs is incredibly hardware intensive.
*   **The VRAM Bottleneck:** Unlike activation memory, the model's parameters cannot be easily page-swapped. To execute any expert at any given step, **100% of the total parameters must reside in VRAM (HBM)**.
*   For example, Mixtral 8x7B (approx. 47B parameters) requires the physical GPU footprint of a 47B dense model, even though its compute step is comparable to a 12B dense model.

### 2. Expert Parallelism & Inter-Device Communication
In distributed clusters, non-MoE layers (Attention, Normalization) behave like standard data-parallel systems. However, during the MoE layer forward pass:
*   A token is mapped to an expert residing on another GPU.
*   This triggers massive **All-to-All communication overheads**, sending token representations across NVLink/InfiniBand.
*   If network bandwidth is low, inter-node communication latency quickly completely negates the FLOP computational speedups of sparsity.

### 3. Regularization & Fine-tuning Overfitting
MoE models behave differently than dense models during fine-tuning:
*   Because of their massive capacity, they are highly prone to **overfitting**, especially on small downstream datasets.
*   **Mitigation:** They require aggressive regularization, including implementing a higher dropout rate inside the expert layers compared to the dense layers (Expert Dropout).
*   **Optimization Setup:** Sparse models benefit from **smaller batch sizes** and **higher learning rates** during downstream fine-tuning compared to their dense counterparts.

### 4. The Instruction Tuning Synergy (Flan-MoE)
Fascinatingly, research (e.g., *MoEs Meets Instruction Tuning*) demonstrates that **sparse MoEs benefit far more from multi-task instruction tuning than dense models**. 
*   When training standard models, adding instruction-following datasets yields linear improvements.
*   For MoEs, instruction tuning on a high number of diverse tasks acts as a massive natural regularizer. 
*   Instruct-tuned MoEs (e.g., Flan-MoE) outperform their dense counterparts by significantly larger margins than non-instruct-tuned variants, proving that massive sparse parameters are exceptionally suited to parsing highly diverse instructional tasks.