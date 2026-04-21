---
tags: [study-plan, llm, algorithms, engineering, layer3]
date: 2026-04-13
aliases: [Layer 3 Syllabus, Architectures and Operators]
related: [[My LLM Learning Sequence]], [[LLM Study Plan]]
sources: [[raw/llm_study_plan/Outline_Layer3.md]]
---

# Layer 3: Architectures and Operators Syllabus

This syllabus represents **Step 1** of [[My LLM Learning Sequence]]. It focuses on the core algorithmic battlefield of Large Language Models (LLMs) from an engineering perspective, detailing not just *what* components exist, but the physical and mathematical bottlenecks they solve.

## Module A: Macro-Architectures (Compute vs. Parameters)
*Focuses on structural evolution and balancing parameter count, compute (FLOPs), and sequence length.*

### 1. The Dense Transformer Baseline
*   **Focus:** Why **Decoder-only** architectures (e.g., GPT, Llama) dominate over Encoder-Decoder (T5) and Encoder-only (BERT) for generative tasks.
*   **Key Concept:** Autoregressive Next-Token Prediction and its efficiency in Few-shot learning.

### 2. Mixture of Experts (MoE)
*   **The Problem:** The linear relationship between dense parameters and inference latency/FLOPs.
*   **The Solution:** Replacing dense Feed-Forward Networks (FFN) with parallel Experts and a lightweight **Router**.
*   **Engineering Challenge:** Sparse activation requires a **Load Balancing Loss** to prevent the "winner-takes-all" routing collapse.
*   *Examples: Mixtral 8x7B, DeepSeek-V3.*

### 3. State Space Models (SSM) & Linear RNNs
*   **The Problem:** The quadratic time/space complexity ($O(N^2)$) of traditional Attention when processing infinite context (e.g., 1M tokens).
*   **The Solution (Mamba):** Discretizing continuous state space equations into hardware-aware linear recurrences.
*   **The Magic:** SSMs achieve parallel training (like [[Transformers]] via parallel prefix scans) but maintain $O(1)$ constant-time generation during inference (like RNNs).

## Module B: Micro-Designs & Innovations (Optimizing the Math)
*Focuses on the highly optimized mathematical operators that make up the deep network.*

### 1. Squeezing the Attention Mechanism
*   **MHA (Multi-Head Attention):** The classic, but terrible for KV Cache memory bandwidth.
*   **MQA (Multi-Query Attention):** Shares one Key/Value pair across all queries to shrink KV Cache to $1/H$, at the cost of expressiveness.
*   **GQA (Grouped-Query Attention):** The modern industry standard (Llama 3). Groups heads to balance MQA's speed and MHA's accuracy.

### 2. Hardware-Aware Operators: FlashAttention
*   **The Truth:** It doesn't change the math; it changes how GPUs read/write (IO-Aware).
*   **The Mechanism:** Bypasses the slow GPU HBM (High Bandwidth Memory) via **Tiling**. Fuses Softmax and Matrix Multiplication directly inside the ultra-fast SRAM to avoid writing intermediate states back to memory.

### 3. Position Embeddings
*   **RoPE (Rotary Position Embedding):** The core of modern LLMs. Maps embeddings to complex space and rotates them by an angle to represent position. Offers superior extrapolation for long texts.
*   **ALiBi:** Adds linear decay negative penalties directly to Attention Scores based on token distance.

### 4. Activations & Normalization
*   **SwiGLU / GeGLU:** Gated activation functions that maintain effective gradient flow in ultra-deep networks better than ReLU/GELU.
*   **RMSNorm & Pre-Norm:** Replaces LayerNorm by dropping the mean-centering calculation (which contributes little) to speed up forward passes. Placing norm *before* Attention/FFN layers prevents gradient explosion early in training.

## Module C: Inference & Decoding (The Engineering of Generation)
*How probability distributions are turned into text at maximum throughput.*

### 1. Sampling Algorithms
*   **Temperature ($T$):** Manipulates the logits before Softmax ($T>1$ flattens, $T<1$ sharpens).
*   **Top-k vs. Top-p (Nucleus Sampling):** Why dynamic probability mass truncation (Top-p) produces more natural text than fixed vocabulary cutoff (Top-k).

### 2. KV Cache Management: PagedAttention
*   **The Bottleneck:** Autoregressive generation recalculates keys/values unless cached, but contiguous cache allocation causes massive memory fragmentation.
*   **The Solution (vLLM):** Borrows **Virtual Memory Paging** from OS design to split KV Cache into non-contiguous blocks. Yields near-zero waste and boosts single-GPU concurrent throughput 2-3x.

### 3. Speculative Decoding
*   **The Bottleneck:** LLM generation is Memory-bound, not Compute-bound.
*   **The Hack:** Uses a tiny, ultra-fast "Draft Model" to guess $N$ tokens ahead. The large model then evaluates all $N$ tokens in parallel in a single forward pass. Mathematically guarantees the exact same output distribution but dramatically speeds up generation.
