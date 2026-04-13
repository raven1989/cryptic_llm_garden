---
tags: [llm, architecture, models]
date: 2026-04-08
---

# Transformers & Macro-Architectures

The fundamental architecture behind most modern Large Language Models (LLMs). The evolution of Transformers focuses on balancing computation, parameter size, and the ability to process long sequences.

## Key Paradigms

*   **Dense Transformer:**
    *   **Decoder-only:** The dominant architecture (e.g., GPT, Llama). Suited for auto-regressive Next-Token Prediction and Few-shot learning due to its single-directional attention.
    *   **Encoder-Decoder:** (e.g., T5, BART). Suited for seq2seq tasks like translation and summarization, though less common in the largest models today.
    *   **Encoder-only:** (e.g., BERT). Primarily used for embedding generation, as in retrieval models for [[Vector Database]]s.

*   **Mixture of Experts (MoE):**
    *   **Concept:** Increases parameter count without increasing single-pass computation (FLOPs). Replaces the Feed-Forward Network (FFN) with multiple "Expert" networks.
    *   **Mechanism:** A lightweight Router network assigns each incoming Token to the Top-K (often Top-2) experts, leaving the rest dormant.
    *   **Challenges:** Susceptible to "winner-takes-all" routing. Mitigated via a Load Balancing Loss.
    *   **Examples:** Mixtral 8x7B, DeepSeek-V2/V3.

*   **State Space Models (SSM):**
    *   **Concept:** Aims to solve the $O(N^2)$ complexity of traditional Attention, which hits a memory/compute wall on very long sequences.
    *   **Mechanism:** (e.g., Mamba) Discards Attention for discrete state space equations. Enables hardware-aware linear recurrence.
    *   **Advantage:** Can parallelize during training. During inference, it behaves like an RNN, requiring $O(1)$ complexity to update the hidden state, theoretically supporting infinite generation length.

## Micro-Designs & Optimizations

*   **Attention Variants:** Multi-Query Attention (MQA) shares KV matrices across heads to reduce KV Cache size. Grouped-Query Attention (GQA), the current standard, groups heads to share KV matrices, offering a balance between speed and quality.
*   **FlashAttention (v1/v2/v3):** An IO-aware optimization. Uses "Tiling" to compute Softmax and matrix multiplication entirely within the fast SRAM, avoiding slow HBM reads/writes.
*   **Position Embeddings:** Rotary Position Embedding (RoPE) maps tokens to complex space and rotates them, elegantly combining absolute and relative positions. Highly effective for length extrapolation.
*   **Normalization:** RMSNorm replaces LayerNorm, skipping mean calculation for speed. Pre-Norm architecture places normalization before Attention/FFN to prevent deep gradient explosion.
