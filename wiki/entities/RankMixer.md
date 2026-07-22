---
tags: ["entity", "architecture", "recommendation"]
aliases: ["RankMixer"]
date: 2026-07-08
sources: ["[[wiki/research/RankMixer Summary.md]]"]
---

# RankMixer

**RankMixer** is a GPU-friendly, compute-bound recommendation ranking architecture developed by ByteDance. It is specifically designed to maximize **Model FLOPs Utilization (MFU)**, pushing it from a traditional ~4.5% to **45%** and enabling recommendation models to scale up to **1B parameters** under strict industrial latency constraints.

## Core Architectural Pillars

### 1. Group-and-Split Feature Tokenization
Instead of allocating a token to every individual heterogeneous feature (which underutilizes GPU warps), RankMixer aggregates features by semantic categories (e.g., demographic groupings) and splits/projects them into uniform embedding dimensions, outputting a clean set of tokens: $\mathbf{X}_0 \in \mathbb{R}^{T \times D}$.

### 2. Multi-Head Token Mixing
A parameter-free explicit feature interaction layer. Unlike standard Self-Attention, it does not rely on inner products (which are prone to representation collapse when applied across different, highly heterogeneous semantic feature spaces).

### 3. Per-Token FFN (PFFN)
Rather than sharing a single Feed-Forward Network across all tokens, RankMixer assigns a non-shared, parameter-isolated FFN (a two-layer MLP) to each individual token channel. This prevents high-frequency features from drowning out long-tail signals and scales model parameters linearly without increasing computational latency.

### 4. Sparse MoE with ReLU Routing
To scale parameters further:
* **ReLU Routing:** Replaces standard Top-$k$ + Softmax gating with a ReLU gating layer. This dynamically adjusts expert allocations based on the token's information density.
* **Dense-to-Sparse:** Uses dense expert activation during training to prevent expert under-training, while running sparse expert routing during online inference to fit strict production latency constraints.

---
*See [[wiki/research/RankMixer Summary.md]] for the full research paper breakdown.*