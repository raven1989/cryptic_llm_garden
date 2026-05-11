---
tags: [llm, inference, memory, optimization]
aliases: [Key-Value Cache]
date: 2026-04-15
sources: ["[[raw/llm_study_plan/Outline_Layer3.md]]", "[[raw/LLM/KV Cache Explained.md]]", "[[raw/LLM/Understanding and coding the KV Cache in LLMs from Scratch.md]]"]
---
# KV Cache

**KV Cache (Key-Value Cache)** is a crucial optimization technique used during the inference (generation) phase of autoregressive language models (like [[Transformers]]). It trades GPU memory (VRAM) for compute speed.

## The Problem: Redundant Computation
During text generation, an LLM predicts one token at a time. To predict token $N$, the [[Self-Attention Mechanism]] needs to look back at all previous $N-1$ tokens. 

Without caching, the model would have to recompute the **Key (K)** and **Value (V)** matrices for *every single historical token* at *every single generation step*. As the sequence gets longer, this redundant recalculation grows quadratically, making generation impossibly slow.

## The Solution: Space-for-Time Tradeoff
Instead of recomputing historical representations, the model calculates the Key and Value vectors for a token once, and then **caches** them in the GPU's memory. 
1. When generating a new token, the model only computes the Query (Q), Key (K), and Value (V) for that *single* new token.
2. It appends the new K and V to the existing KV Cache.
3. It multiplies the new Query against the entire cached K and V matrices to compute attention.

## The New Bottleneck: Memory Bandwidth
While KV Cache solves the compute bottleneck, it creates a severe memory bottleneck. The cache is stored in the GPU's **VRAM (Video RAM)**, and it grows with every generated token.

Crucially, **the KV Cache is not a single global matrix**. Because the hidden state of a token evolves as it passes deeper into the model, every single attention layer (transformer block) must compute and maintain its own completely independent KV cache. If a model has 32 layers, the total memory footprint is $32 \times \text{Cache Size Per Layer}$.

Because of this massive memory footprint, modern LLM inference is often **Memory-bound** (limited by how fast the GPU can read the cache from VRAM to the compute cores) rather than Compute-bound.

## Key Optimizations

Because the KV Cache is so massive, several innovations have emerged to manage it:

### 1. Architectural: Shrinking the Cache Size
Model architectures have evolved to fundamentally reduce how many K and V vectors need to be stored:
*   **MHA (Multi-Head Attention):** The classic approach. Every attention head has its own independent K and V matrices. The KV Cache is enormous.
*   **MQA (Multi-Query Attention):** All attention heads share a *single* set of K and V matrices. The KV Cache shrinks drastically (by a factor of the number of heads), dramatically speeding up inference, but at the cost of some model expressiveness.
*   **GQA (Grouped-Query Attention):** The modern standard (used in LLaMA 2/3). Heads are divided into groups, and each group shares a K and V matrix. This offers a perfect compromise between the speed of MQA and the quality of MHA.

### 2. Infrastructure: Memory Management
How the cache is allocated in VRAM makes a massive difference in performance:

*   **Naive Implementation (`torch.cat`):** Appending new vectors to the cache tensor at every generation step causes severe memory fragmentation and reallocation overhead on the GPU.
*   **Pre-allocation:** The optimized approach is to pre-allocate a massive zero-tensor up to the model's `max_seq_len`. However, for models with massive context windows (like LLaMA 3's 131k tokens), pre-allocating an empty cache tensor consumes ~8GB of VRAM *per sequence* before generation even begins.
*   **PagedAttention:** Traditionally, frameworks pre-allocated large contiguous chunks of VRAM for the KV Cache. Because the final length of a generated sequence is unknown, this led to massive internal fragmentation (wasted memory space), much like old hard drives. **PagedAttention** (pioneered by the vLLM framework) solves this by borrowing **Virtual Memory Paging** concepts from operating systems. It divides the KV Cache into fixed-size, non-contiguous "blocks" (pages). This allows for dynamic memory allocation with near-zero waste, allowing a single GPU to handle 2-3x more concurrent requests.

### 3. Implementation: Sliding Window Truncation
To prevent the KV Cache from exhausting GPU memory during very long or infinite generations, implementations often use a "sliding window." The cache is dynamically truncated at each step to only keep the last $N$ tokens (e.g., `cache_k = cache_k[:, :, -window_size:, :]`).

## Calculating KV Cache VRAM Footprint

To accurately estimate the VRAM consumed by the KV Cache for a single token, you must account for the model's architecture (specifically the number of layers and whether it uses GQA/MQA). The formula is:

```python
Bytes_Per_Token = (
    (hidden_size / num_attention_heads) * # Dimensionality of one head (d_k)
    num_key_value_heads *                 # Number of KV heads
    num_hidden_layers *                   # Total layers in the network
    bytes_per_parameter *                 # 2 bytes for FP16/BF16, 1 byte for INT8/FP8
    2                                     # Multiplier to account for both K and V matrices
)
```

*   **Total Cache Size:** `Bytes_Per_Token * total_sequence_length * batch_size`.
*   **The GQA Advantage:** In standard Multi-Head Attention (MHA), `num_key_value_heads` equals `num_attention_heads`. In Grouped-Query Attention (GQA), `num_key_value_heads` is significantly smaller (e.g., 8 instead of 64), which drastically reduces the multiplier in this formula, saving massive amounts of VRAM.

## Generation Lifecycle & Cleanup
The cache is maintained for the entire duration of a single sequence generation. Because the model only processes one new token at a time during cached inference, the generation loop must explicitly track a `current_pos` integer to manually offset the positional embeddings for the new token (otherwise the model treats the new token as position `0` and hallucinates).

When the sequence is complete (an `<EOS>` token is generated), the framework flushes the VRAM blocks for that sequence so the inference server can reallocate them for a different user's request.
