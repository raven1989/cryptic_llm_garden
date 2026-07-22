---
tags:
  - "entities"
  - "attention-mechanisms"
  - "sequential-modeling"
aliases:
  - "Stacked Target-to-History Cross Attention"
  - "STCA"
date: 2026-07-10
sources: ["[[wiki/research/Douyin STCA Summary.md]]"]
---

# STCA (Stacked Target-to-History Cross Attention)

**STCA** is a highly efficient sequential modeling attention architecture proposed by ByteDance/Douyin in 2025. It is specifically designed to handle ultra-long user behavior sequences (up to **10,000+ items**) in industrial recommender systems under strict latency and computational budgets.

## Core Mechanism

STCA's central innovation lies in making a deliberate capacity-cost trade-off: **it completely eliminates explicit history-to-history self-attention** ($O(L^2)$ complexity) and instead uses target-to-history cross-attention.

1.  **Target as Query ($Q$):** The candidate video represents the sole Query vector $\mathbf{q} \in \mathbb{R}^d$.
2.  **History as Key and Value ($K, V$):** The user's historical interaction sequence $\mathbf{X} \in \mathbb{R}^{L \times d}$ acts as the background keys and values.
3.  **Linear Complexity:** This formulation bounds the attention computation to strictly **$O(L)$ linear complexity** with respect to sequence length $L$.

## Highway Query Update (Target-Conditioned Fusion)

Because history items do not communicate directly with each other, STCA stacks $M$ layers and uses a **Highway Query Update** mechanism. The Query $\mathbf{q}^{(i+1)}$ for the next layer is dynamically updated by concatenating all previous layers' outputs with the original target embedding:

$$\mathbf{q}^{(i+1)} = \text{LN}\left( \text{SwiGLUFFN}^{(i+1)}\left( \left[ \mathbf{o}^{(1)} \parallel \cdots \parallel \mathbf{o}^{(i)} \parallel \mathbf{x}_t \right] \mathbf{W}_C^{(i+1)} \right) \right)$$

This allows subsequent cross-attention layers to perform increasingly refined search steps on the user's history.

## Performance Optimization

To avoid materializing Key and Value matrices of shape $L \times d_h$ in GPU memory (which is a massive High Bandwidth Memory bottleneck), STCA reorders the computation algebraically:

$$\text{Attn}(\mathbf{q}, \mathbf{X}) = \left[ \text{softmax} \left( \frac{\left[ (\mathbf{q} \mathbf{W}_Q) \mathbf{W}_K^\top \right] \mathbf{X}^\top}{\sqrt{d_h}} \right) \mathbf{X} \right] \mathbf{W}_V$$

By executing the vector-matrix product on the Query side first, the sequence projection steps are bypassed, preserving GPU memory and throughput.

## See Also
*   [[Douyin STCA Summary|Douyin STCA Summary & Deep Dive]]
*   [[RankMixer]] (Downstream ranking backbone)
*   [[HSTU]] (Alternative sequential recommendation architecture)
