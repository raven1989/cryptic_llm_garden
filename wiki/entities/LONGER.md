---
tags: ["entity", "architecture", "recommendation", "CTR", "CVR", "attention", "long-sequence", "scaling-laws"]
aliases: ["LONGER", "Long-sequence Optimized traNsformer for GPU-Efficient Recommenders"]
date: 2026-08-24
sources: ["[[wiki/research/LONGER Summary.md]]"]
---

# LONGER (Long-sequence Optimized traNsformer for GPU-Efficient Recommenders)

**LONGER** is an end-to-end ultra-long user behavior sequence modeling framework published at **RecSys 2025** by ByteDance. It scales **end-to-end** transformer modeling to sequence length **10,000**, deliberately abandoning the two-stage retrieval paradigm of [[SIM]]/[[TWIN]] in favor of attending over the full sequence directly — made feasible by aggressive compression (token merge, query sampling) and system co-design (synchronous GPU training, KV cache serving). Deployed across dozens of ByteDance scenarios serving billions of users.

![LONGER Model Architecture](../media/LONGER_model_architecture.png)

---

## Core Architectural Pillars

### 1. Global Tokens
Auxiliary tokens with a **full attention receptive field** — target item representation, learnable CLS tokens, UID embeddings, and high-order compressed user–item interaction features. Two roles: centralized anchors for feature interaction (history × context × candidate), and attention stabilizers that mitigate the [[Attention Sink]] effect (per StreamLLM).

### 2. Token Merge + InnerTrans
Groups every $K$ adjacent tokens into one ($L \to L/K$), attacking the $O(L^2d)$ attention cost in the industrial regime $L \gg d$ ($L=2000$, $d=32$). At $K=4$: **−42.8% FLOPs**. Merge options: plain **concat**, or **InnerTrans** — a lightweight per-group Transformer preserving intra-group interactions. Merging *expands* parameters ($12K^2d^2+13Kd$), converting saved compute into capacity, and empirically *improves* AUC (denoising prior).

### 3. Hybrid Causal Attention
- **Input:** keys/values $\mathbf{R} = [\mathbf{G}; \mathbf{H}]$ (global + **full** merged sequence); queries $\mathbf{O} = [\mathbf{G}; \mathbf{H}_S]$ (global + $k$ **sampled** sequence tokens). Two PE signals: absolute time-difference (concatenated) + learnable absolute position (added).
- **Layer 1 — Cross-causal attention:** $\mathbf{O}$ attends over $\mathbf{R}$, compressing $(m+L) \to (m+k)$. The causal mask enforces sequence→candidate invisibility, which makes [[KV Cache]] serving valid.
- **Layers 2..N — Self-causal attention:** operate only on the compressed $(m+k)$ working set for high-order interactions. "Attend widely once, then refine deeply."

### 4. Query Sampling: Recency Wins
Only the query side is sampled; the full sequence stays visible as keys/values. Ablation: **Recent-$k$** (AUC 0.85290) > Uniform > Recent+Uniform > **Learnable** (worst, 0.84946) — a notable negative result for Perceiver/Q-Former-style learnable queries. $k=100$ is the sweet spot (54% of FLOPs vs. $k=250$, near-equal AUC).

### 5. System Co-Design
- **Fully synchronous GPU training:** unified dense+sparse parameter storage on GPU (no external parameter server); hierarchical embeddings by frequency — HBM (hot) / CPU MEM (warm) / SSD (cold).
- **Mixed precision (BF16/FP16) + activation recompute** (via TF `custom_gradient`): +18% throughput, −16% training time, −18% memory.
- **KV cache serving** (motivated by M-FALCON / [[HSTU]]): cache user-sequence K/V once, reuse across candidates; throughput degradation cut from −40% to −6.8%.

---

## Key Properties

| Property | Detail |
| :--- | :--- |
| **Paradigm** | End-to-end (no retrieval stage) |
| **Max Sequence Length** | 10,000 |
| **Compression** | Token merge ($L \to L/K$) + query sampling (recent-$k$) |
| **Attention** | Hybrid: 1 cross-causal (compress) + N self-causal (refine) |
| **Global Tokens** | Target item, CLS, UID, compressed interaction features |
| **Query Selection** | Recent-$k$ (beats learnable/uniform) |
| **Loss** | Single binary cross-entropy |
| **Training** | Fully synchronous GPU, hierarchical embedding storage (HBM/MEM/SSD) |
| **Serving** | KV cache (sequence K/V reused across candidates) |
| **Scaling** | Power law in seq length, params ($R^2=0.987$), FLOPs ($R^2=0.967$) |

---

## Experimental & Business Highlights

- **Offline (5.2B-sample Douyin Ads CVR):** AUC **0.85290**, LogLoss 0.47103 — +1.57% AUC over Base, +0.21% over the strongest baseline (Transformer). Beats [[TWIN]], [[DIN]], [[HSTU]]. (0.1% AUC is online-significant.)
- **Online A/B — Douyin Ads:** ADSS/ADVV up to +2.097%/+2.151% (Short Video).
- **Online A/B — Douyin E-Commerce:** Order/U up to +7.92%, GMV/U up to +6.54% (Live Streaming).

---

## Position in the Landscape

LONGER and [[STCA]] are ByteDance's two **end-to-end** answers to ultra-long sequence modeling — STCA compresses via stacked target cross-attention ($O(L)$), LONGER via token merge + recent-$k$ query sampling + global tokens. Both reject the two-stage retrieval of [[SIM]]/[[TWIN]] and validate recsys scaling laws at 10K length.

---

## Related Wiki Pages
* [[LONGER Summary]]: Complete research summary with derivations, ablations, and landscape comparison.
* [[SIM]] / [[TWIN]]: Two-stage retrieval predecessors LONGER positions against.
* [[STCA]]: Sibling ByteDance end-to-end long-sequence system.
* [[HSTU]]: Source of the M-FALCON KV-cache serving idea.
* [[KV Cache]]: The LLM serving optimization LONGER adapts.
* [[Attention Sink]]: The effect global tokens mitigate.
* [[Positional Encoding]]: Context for LONGER's dual PE signals.
