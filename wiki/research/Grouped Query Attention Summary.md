---
tags: [llm, algorithms, attention, optimization, performance]
date: 2026-05-12
sources: ["[[raw/LLM/What is grouped query attention.md]]"]
---

# Summary: What is grouped query attention (GQA)?

**Source:** IBM Think - "What is grouped query attention (GQA)?"

## Overview
This article provides an accessible explanation of [[Grouped Query Attention]] (GQA) and how it optimizes the standard [[Multi-Head Attention]] (MHA) mechanism found in [[Transformers]]. As Large Language Models (LLMs) scale, MHA introduces significant memory and compute bottlenecks. The article details how early attempts like [[Multi-Query Attention]] (MQA) sought to solve this, and how GQA emerged as the ideal middle ground.

## Key Concepts Extracted

### Standard Multi-Head Attention (MHA)
*   **Mechanism:** MHA calculates self-attention in parallel by splitting attention layers into multiple heads. Each head has its own unique set of Query (Q), Key (K), and Value (V) weights.
*   **The Bottleneck:** The compute and memory (VRAM) required for MHA scale quadratically with sequence length. More importantly, storing all unique K and V vectors for every head creates massive memory overhead (the [[KV Cache]]), slowing down autoregressive inference due to memory bandwidth limits.

### Multi-Query Attention (MQA)
*   **Optimization:** Instead of training a unique K and V head for every single Q head, MQA uses a *single* K head and a *single* V head shared across all $h$ Query heads.
*   **Benefits:** Dramatically reduces the size of the [[KV Cache]] (10-100x smaller) and speeds up decoder inference (up to 12x faster).
*   **Drawbacks:** Noticeable performance and accuracy degradation. It also requires training models from scratch (cannot easily adapt MHA checkpoints) and creates redundancies in tensor parallelism since that single K/V pair must be replicated across all GPU nodes anyway.

### Grouped Query Attention (GQA)
*   **The Sweet Spot:** Conceived by Ainslie et al., GQA is a generalization of MQA. Instead of sharing one K/V pair across *all* Q heads, it partitions the Q heads into multiple "groups." Each group shares one set of K and V heads.
*   **Spectrum:** 
    *   If groups = number of heads $\rightarrow$ Standard MHA.
    *   If groups = 1 $\rightarrow$ MQA.
*   **Benefits:** 
    *   *Effective Compromise:* Delivers nearly the same accuracy as MHA while retaining almost all the speed and memory benefits of MQA.
    *   *Efficient GPU Usage:* Distributing K-V pairs across groups aligns perfectly with tensor parallelism, avoiding the "wasted compute" of replicating a single MQA pair.
    *   *Uptraining:* Unlike MQA, existing MHA models can be fine-tuned ("uptrained") to use GQA.
*   **Adoption:** Widely adopted in modern leading LLMs, including Meta's Llama 2 and Llama 3, Mistral 7B, and IBM's Granite models.