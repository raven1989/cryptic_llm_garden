---
tags: ["entity", "architecture", "recommendation", "attention", "CTR", "long-sequence"]
aliases: ["SIM", "Search-based Interest Model"]
date: 2026-08-18
sources: ["[[wiki/research/SIM Summary.md]]"]
---

# SIM (Search-based Interest Model)

**SIM (Search-based User Interest Model)** is a long-sequence CTR prediction framework published at CIKM 2020 by Alibaba Group. It scales sequential user behavior modeling up to **54,000 items** ($54\times$ beyond previous SOTA models like MIMN) by employing a **two-stage cascaded search architecture**: **General Search Unit (GSU)** and **Exact Search Unit (ESU)**.

---

## Core Architectural Pillars

### 1. Cascaded Two-Stage Search Paradigm
Instead of computing attention across tens of thousands of items (which violates $<30\text{ms}$ online serving SLAs) or compressing behaviors into static query-unaware memory matrices (which loses candidate-specific details), SIM breaks interest modeling into two stages:
- **Stage 1: General Search Unit (GSU):** Fast-filters raw lifelong sequence $\mathbf{B}$ ($T \le 54,000$) down to a candidate-relevant Sub-behavior Sequence (SBS) $\mathbf{B}^*$ of length $K \le 200$.
- **Stage 2: Exact Search Unit (ESU):** Applies multi-head target attention and temporal distance modeling on the filtered SBS.

### 2. GSU: Hard-Search vs. Soft-Search

$$r_i = \begin{cases} \text{Sign}(C_i = C_a) & \text{Hard-Search (Non-Parametric)} \\ (W_b \mathbf{e}_i) \odot (W_a \mathbf{e}_a)^T & \text{Soft-Search (Parametric, MIPS/ALSH)} \end{cases}$$

- **Hard-Search:** Non-parametric category matching. Requires zero inference compute when paired with the offline **User Behavior Tree (UBT)**.
- **Soft-Search:** Continuous inner-product vector retrieval using ALSH. Trained with a dedicated auxiliary long-term CTR prediction task to resolve short/long-term distribution shift.

### 3. ESU: Temporal Multi-Head Target Attention
- **Time Interval Embedding ($\mathbf{D}$):** Models elapsed days $\Delta_j$ between the candidate impression and historical behaviors:
  $$\mathbf{z}_j = \text{concat}(\mathbf{e}_j^*, \mathbf{e}_j^t)$$
- **Multi-Head Attention:**
  $$\mathbf{att}^i_{\text{score}} = \text{Softmax}(W_{bi}\mathbf{z}_b \odot W_{ai}\mathbf{e}_a), \quad \mathbf{head}_i = \mathbf{att}^i_{\text{score}} \mathbf{z}_b$$
  $$U_{lt} = \text{concat}(\mathbf{head}_1; \dots; \mathbf{head}_q)$$

### 4. System Co-Design: User Behavior Tree (UBT)
- Distributed 2-level `Key-Key-Value` index: $\text{User ID} \to \text{Category ID} \to [\text{Items} + \text{Timestamps}]$ (size: $\sim 22\text{ TB}$).
- Turns online candidate search into an $O(1)$ category lookup.
- Combined with GPU deep kernel fusion, handles 54,000 behavior sequence items with only **5ms latency increase** over truncated 1,000-item baselines.

---

## Key Properties

| Property | Detail |
| :--- | :--- |
| **Max Sequence Length** | Up to **54,000 items** ($54\times$ over MIMN) |
| **Two-Stage Search** | GSU (coarse denoising & filtering) $\to$ ESU (fine attention) |
| **GSU Hard Search** | Category-level filtering ($O(1)$ via User Behavior Tree) |
| **GSU Soft Search** | Embedding inner product + ALSH / MIPS |
| **ESU Modeling** | Multi-head target attention + Time interval embedding |
| **Online SLA** | $< 30\text{ ms}$ at $> 1\text{M QPS}$ serving main Taobao ads traffic |

---

## Experimental & Business Highlights

- **Industrial Dataset (Alibaba):** AUC $0.6624$ (SIM-hard + Time) vs. $0.6541$ (MIMN), providing a $+0.0083$ AUC boost on sequences up to 54,000 items.
- **Online A/B Testing:** $+7.1\%$ CTR and $+4.4\%$ RPM in Alibaba's Taobao display advertising system.
- **Dormant Interest Activation:** Increased average $d_{category}$ from $11.2 \to 13.3$ days, demonstrating effective retrieval of long-term purchase intentions.

---

## Related Wiki Pages
* [[SIM Summary]]: Complete research summary, architectural derivations, and ablation study.
* [[DIN]]: Predecessor candidate-aware local activation model for short sequences.
* [[DIN Summary]]: Research summary of the original Deep Interest Network.
* [[STCA]]: ByteDance's streaming target cross-attention mechanism for long sequences.
* [[HSTU]]: Meta's Generative Recommenders architecture.
