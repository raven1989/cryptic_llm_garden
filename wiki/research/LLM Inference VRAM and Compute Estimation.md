---
tags: [inference, vram, flops, optimization, memory-bound, compute-bound]
aliases: [Estimating LLM Inference VRAM and Compute]
date: 2026-04-15
sources: ["[[raw/LLM/大模型推理显存和计算量估计方法.md]]"]
---

# LLM Inference VRAM and Compute Estimation

Estimating the required VRAM (Video RAM) and Computational load (FLOPs) for Large Language Model (LLM) inference is critical for optimizing throughput, setting maximum batch sizes, and avoiding Out-Of-Memory (OOM) errors. 

Based on the [Aliyun Developer Community article](https://developer.aliyun.com/article/1668044), this page breaks down the exact mathematical formulas for estimating both memory consumption and raw computational time during the two phases of inference: Prefill and Decode.

## 1. VRAM Estimation (The [[KV Cache]] Bottleneck)

During inference, the model weights take up static memory, but the dynamic memory is almost entirely consumed by the **KV Cache**. The size of the KV Cache scales linearly with the sequence length and batch size.

### Calculating the VRAM for a Single Token
To find out how many bytes a single token consumes in the KV cache across the entire model:

```python
bytes_per_token = (
    (hidden_size / num_attention_heads) * # Dimension per head (d_k)
    num_key_value_heads *                 # Number of KV heads (GQA/MQA optimization)
    num_hidden_layers *                   # Total transformer layers
    kvcache_dtype_byte *                  # Bytes per parameter (e.g., FP16/BF16 = 2, INT8 = 1)
    2                                     # Multiplier for both K and V
)
```
*Note: For standard Multi-Head Attention (MHA), `num_key_value_heads` equals `num_attention_heads`. For Grouped-Query Attention (GQA) used in models like Llama 3 or Qwen 2.5, `num_key_value_heads` is much smaller, drastically reducing the cache size.*

### Calculating Max Batch Size
To determine the maximum `batch_size` a GPU can support:
1.  **Available VRAM for Cache:** `(Total_GPU_VRAM - Model_Weights_VRAM) * Safety_Coefficient (e.g., 0.8)`
2.  **Sequence VRAM:** `bytes_per_token * (input_length + output_length)`
3.  **Max Batch Size:** `Available_VRAM / Sequence_VRAM`

## 2. Compute Estimation (FLOPs)

Transformer computational load comes primarily from the Self-Attention mechanism, the Feed-Forward Network (FFN), and the final LM Head. 

**Key Variables:**
*   **$L$**: Number of Layers
*   **$H$**: Hidden Size
*   **$I$**: FFN Intermediate Size
*   **$V$**: Vocabulary Size
*   **$S$**: Prefill Sequence Length
*   **$T$**: Total Sequence Length (Prefill + Current Decode Step)

### Prefill Stage (Compute-Bound)
In the prefill stage, the entire prompt is processed at once. This utilizes large Matrix-Matrix multiplications (GEMM).
*   **Attention:** $L \times (8SH^2 + 4S^2H)$
*   **FFN:** $L \times (6SHI)$
*   **LM Head:** $2SHV$

### Decode Stage (Memory-Bound)
In the decode stage, tokens are generated one by one. The input sequence length is essentially 1, but the attention mechanism must read the entire historical sequence $T$ from the KV Cache. This utilizes smaller Matrix-Vector multiplications (GEMV).
*   **Attention:** $L \times (8H^2 + 4HT)$
*   **FFN:** $L \times (6HI)$
*   **LM Head:** $2HV$

*(To calculate average decode time across $K$ generated tokens, $T$ can be approximated as $S + (K/2)$).*

## 3. The Reality Gap (Theoretical vs. Actual Latency)

When calculating theoretical execution time: `Time = Total_FLOPs / GPU_Peak_FLOPs`.

**Observations:**
1.  **Prefill matches theory:** Theoretical prefill time closely matches actual runtime because GEMM operations highly utilize the GPU's Tensor Cores (often hitting ~60%+ of peak theoretical limits).
2.  **Decode defies theory:** Theoretical decode time is usually **orders of magnitude faster** than actual runtime. 

**Why Decode is so much slower than theoretical FLOPs suggest:**
*   **Memory Bandwidth Bottleneck:** Decode calculations are GEMV (Matrix-Vector) operations. The GPU spends almost all its time waiting to move model weights and the massive KV Cache from VRAM into the compute cores, rather than actually doing the math. 
*   **Hardware Utilization:** GPUs are terrible at utilizing peak compute for small batch sizes (batch_size = 1 is the worst-case scenario for an AI accelerator).
*   **Overheads:** For very fast operations, the overhead of memory movement, kernel dispatch, and network communication in distributed setups (like `AllToAll` operations in MoE models) easily dwarfs the actual compute time.