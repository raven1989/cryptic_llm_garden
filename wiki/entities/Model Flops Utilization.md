---
tags: ["entity", "metric", "efficiency", "infrastructure"]
aliases: ["MFU", "Model FLOPs Utilization"]
date: 2026-07-08
sources: ["[[wiki/research/RankMixer Summary.md]]"]
---

# Model FLOPs Utilization (MFU)

**Model FLOPs Utilization (MFU)** is a critical hardware-efficiency metric that measures the ratio of the actual floating-point operations (FLOPs) executed per second by a machine during model training/inference to the theoretical peak performance of the underlying hardware (such as GPUs or TPUs).

$$\text{MFU} = \frac{\text{Actual FLOPs executed per second}}{\text{Theoretical Peak FLOPs of the Hardware}}$$

## Context in Machine Learning

### 1. Large Language Models (LLMs)
Due to their highly structured compute-bound nature (massive matrix multiplications sharing dense parameters), LLMs routinely achieve an MFU of **40% to 50%+** on modern hardware accelerators (e.g., NVIDIA H100, A100).

### 2. Traditional Recommendation Models
Traditional recommendation ranking networks (e.g., Wide & Deep, DeepFM, DCN) are heavily **memory-bound**. They spend the majority of execution cycles fetching sparse user/item embeddings from massive high-bandwidth memory or host memory tables. Consequently, their MFU is extremely low—typically around **4% to 5%**—meaning the vast majority of GPU tensor core computational power is left idle.

### 3. Compute-Bound Recommendation Scaling
To bridge this efficiency gap and unlock LLM-style scaling laws in recommendation models, modern architectures focus on restructuring feature interactions to be compute-bound:
* [[OneRec-V2]] scales up user sequence representation learning via a unified generative framework.
* [[RankMixer]] projects grouped sparse embeddings into uniform tokens and runs heavy **Per-Token FFN (PFFN)** layers to boost the ranking model's MFU to **45%**, allowing parameters to reach the **1B** scale under strict industrial SLA latency limits.
