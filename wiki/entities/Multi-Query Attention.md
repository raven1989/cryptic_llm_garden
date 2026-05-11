---
tags: [llm, algorithms, attention, optimization]
date: 2026-05-12
aliases: [MQA]
related: ["[[Multi-Head Attention]]", "[[Grouped Query Attention]]", "[[KV Cache]]", "[[Transformers]]"]
sources: ["[[raw/LLM/What is grouped query attention.md]]", "[[wiki/research/Grouped Query Attention Summary.md]]"]
---

# Multi-Query Attention (MQA)

Multi-Query Attention (MQA) is an architectural optimization for [[Transformers]] designed to reduce the memory bandwidth requirements of the attention mechanism during inference.

## Concept
In standard [[Multi-Head Attention]] (MHA), each attention head has its own unique Query ($Q$), Key ($K$), and Value ($V$) weights. When generating text autoregressively, the model must store the computed $K$ and $V$ vectors for all previous tokens in the [[KV Cache]]. Storing unique $K$ and $V$ vectors for *every* head quickly exhausts GPU VRAM and bottlenecks inference speeds due to memory bandwidth limits.

MQA solves this by drastically simplifying the architecture: 
*   It retains multiple independent **Query ($Q$) heads**.
*   It uses only a **single shared Key ($K$) head and a single shared Value ($V$) head** across all Query heads.

## Benefits
*   **Massively Reduced KV Cache:** By only computing and storing one set of $K$ and $V$ vectors per token (instead of $h$ sets), the size of the KV cache is reduced by a factor of $h$ (typically 10x to 100x smaller).
*   **Faster Inference:** With less data to shuttle between GPU memory (HBM) and compute cores, decoder inference can be significantly faster (up to 12x). It also frees up memory to allow for much larger batch sizes.

## Drawbacks
While highly efficient, MQA has distinct disadvantages that led to the development of the more balanced [[Grouped Query Attention]] (GQA):
1.  **Performance Degradation:** Forcing all queries to attend to the exact same Key and Value representations reduces the model's capacity to represent complex nuance, leading to a drop in accuracy compared to MHA.
2.  **Training from Scratch:** A model trained with standard MHA cannot simply be adapted to MQA; it must be trained from scratch with the MQA architecture, incurring high initial compute costs.
3.  **Tensor Parallelism Inefficiency:** In distributed GPU clusters, tensor parallelism requires $K$ and $V$ values to be present on each node. Because MQA only has one $K/V$ pair, this pair must be redundantly replicated across all nodes, wasting computational routing efficiency.