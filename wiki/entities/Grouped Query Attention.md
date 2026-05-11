---
tags: [llm, algorithms, attention, optimization]
date: 2026-05-12
aliases: [GQA]
related: ["[[Multi-Head Attention]]", "[[Multi-Query Attention]]", "[[KV Cache]]", "[[Transformers]]"]
sources: ["[[raw/LLM/What is grouped query attention.md]]", "[[wiki/research/Grouped Query Attention Summary.md]]"]
---

# Grouped Query Attention (GQA)

Grouped Query Attention (GQA) is an optimization of the standard [[Multi-Head Attention]] (MHA) mechanism designed to dramatically reduce memory bandwidth bottlenecks and speed up inference in large [[Transformers]], particularly for autoregressive generation tasks. 

## The Problem with MHA
In standard [[Multi-Head Attention]], each of the $h$ attention heads possesses its own independent Query ($Q$), Key ($K$), and Value ($V$) projection matrices. During generation, storing the $K$ and $V$ tensors for every preceding token across every single head consumes massive amounts of VRAM. This is known as the [[KV Cache]]. As sequence length and model size grow, reading and writing this massive KV Cache severely bottlenecks inference speed due to GPU memory bandwidth limits.

## The GQA Solution
Introduced as a balanced evolution of [[Multi-Query Attention]] (MQA), GQA partitions the model's Query heads into several **groups**. 
*   Instead of each Query head having its own unique Key and Value head (like in MHA), all the Query heads within a single *group* share a single Key and Value head.

### The Attention Spectrum
GQA generalizes the attention mechanism, allowing architectural tuning between accuracy and efficiency:
*   **Number of Groups = Number of Heads:** This is functionally identical to standard **Multi-Head Attention**. High accuracy, high memory cost.
*   **Number of Groups = 1:** This is functionally identical to **[[Multi-Query Attention]] (MQA)**. All Query heads share a single K and V head. Fast, low memory, but suffers accuracy degradation.
*   **Intermediate Groups (GQA):** E.g., 32 Query heads divided into 8 groups (4 Query heads per group). Achieves nearly the accuracy of MHA while retaining the inference speed and memory footprint benefits of MQA.

## Visualizing the Attention Spectrum

Here is a structural comparison of how Query ($Q$), Key ($K$), and Value ($V$) heads are allocated across the three main attention architectures (assuming a model with 4 Query heads).

```text
Multi-Head Attention (MHA)
Every Q head gets its own K and V head.
(High Accuracy, High Memory)

  Q1     Q2     Q3     Q4
  |      |      |      |
  K1     K2     K3     K4
  V1     V2     V3     V4

-----------------------------------------
Grouped Query Attention (GQA) 
Q heads are grouped. Each group shares a K and V head.
(High Accuracy, Low Memory - The Sweet Spot)

  Group 1       Group 2
  Q1    Q2      Q3    Q4
   \    /        \    /
     K1            K2
     V1            V2

-----------------------------------------
Multi-Query Attention (MQA)
All Q heads share a single K and V head.
(Lower Accuracy, Lowest Memory)

  Q1    Q2      Q3    Q4
   \     \      /     /
    \     \    /     /
     \-----\  /-----/
            K1
            V1
```

## Key Benefits
1.  **Reduced KV Cache:** By drastically reducing the number of Key and Value vectors that must be computed and stored, GQA shrinks the KV Cache, directly accelerating decoding speed and enabling larger batch sizes or longer context windows.
2.  **Hardware Efficiency:** Unlike MQA (where a single KV pair must be redundantly copied to all GPU nodes to satisfy tensor parallelism operations), GQA's multiple groups naturally map to different GPU shards, making optimal use of distributed hardware.
3.  **Uptraining:** Unlike MQA which usually requires training from scratch, an existing model pre-trained with standard MHA can be fine-tuned ("uptrained") to convert its attention layers into GQA, saving massive compute costs.

## Industry Adoption
Due to its optimal tradeoff between speed, memory, and performance, GQA has become a standard architectural choice in state-of-the-art models, including:
*   Meta's LLaMA 2 and LLaMA 3
*   Mistral 7B
*   IBM Granite 3.0