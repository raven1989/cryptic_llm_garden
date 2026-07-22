---
tags:
  - "entities"
  - "systems"
  - "recommendation-systems"
  - "distributed-training"
aliases:
  - "RLB"
  - "Request Level Batching"
date: 2026-07-10
sources: ["[[wiki/research/Douyin STCA Summary.md]]"]
---

# Request Level Batching (RLB)

**Request Level Batching (RLB)** is a highly optimized systems-level batching paradigm developed by ByteDance/Douyin to support ultra-long user behavior sequence modeling. It shifts the recommendation pipeline from a sample-centric (point-wise) layout to a **user-centric (request-wise)** layout.

## The Problem solved by RLB

In real-world ranking systems, when a user refreshes their feed, the model is asked to score $m$ candidate items (e.g., $m = 100 \sim 1000$). 
In traditional pointwise batching:
*   The identical, massive user sequence $\mathcal{H}$ of length $L$ (up to 10k) must be duplicated and copied from CPU host to GPU $m$ times.
*   The GPU runs identical sequence-encoding transformations $m$ times, causing extreme memory, I/O, and compute redundancy.

## How RLB Works

RLB groups all candidate targets for the same user request into a unified micro-batch:
$$\mathcal{B}_u = \{ (u, t_k, y_k) \}_{k=1}^m$$

This structures the tensor computations in a **"compute-once, reuse-$m$-times"** pattern:
1.  **Shared User History:** The user history is transferred to the GPU and encoded exactly **once** per request (Tensor shape: $[B, L, d]$).
2.  **Batched Targets:** The targets are grouped into a matrix of shape $[B, m, d]$.
3.  **Cross-attention Parallelization:** The GPU runs cross-attention between the $m$ target queries and the single, shared key-value sequence background in parallel.
4.  **Prediction Output:** The model outputs a prediction of shape $[B, m]$, which maps directly against the batched targets.

## Unbiasedness

Grouping and averaging targets at the request level preserves the mathematical expectation of the pointwise Binary Cross-Entropy gradients:
$$\mathcal{L}_u = \frac{1}{m} \sum_{k=1}^m \mathcal{L}_{BCE}\left(\hat{y}(u, t_k), y_k\right)$$

Because of the **linearity of expectation**, RLB is a mathematically **unbiased estimator** of standard pointwise empirical risk, altering only the memory layout and execution efficiency on hardware.

## Production Impact
*   **84% memory bandwidth reduction** on PCIe/NVLink channels.
*   **5.1× training throughput improvement** over pointwise baselines.
*   **50% CPU and network overhead savings** on the distributed Parameter Server.

## See Also
*   [[Douyin STCA Summary|Douyin STCA Summary & Deep Dive]]
*   [[STCA]] (Parallel linear cross-attention layer)
