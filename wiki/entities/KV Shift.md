---
title: "KV Shift"
aliases: ["Key-Value Shift", "Temporal-Spatial KV Shift"]
date: 2026-07-23
tags: ["entity", "attention", "transformer", "time-series", "covariates"]
sources: ["[[wiki/research/CITRAS Summary.md]]"]
---

# KV Shift (Key-Value Shift)

**KV Shift** (Key-Value Shift) is a specialized attention alignment mechanism introduced in **[[CITRAS Summary|CITRAS]]** to integrate future known covariates into decoder-only (causal) transformers without causing target data leakage. It enables a target variable token at step $i$ to query the concurrent covariates at step $i$, while retrieving the shifted future covariate values at step $i+1$.

---

## Technical Motivation

In auto-regressive, patch-based time series models, the target variable at patch step $i$ ($\mathbf{H}_i^{tgt,:}$) is mapped to predict the next target variable at step $i+1$ ($\mathbf{H}_{i+1}^{tgt,:}$). 

To optimize this prediction, the model should ingest the known covariate at tomorrow's step $i+1$ ($\mathbf{H}_{i+1}^{knw,:}$). However, directly matching $\mathbf{H}_i^{tgt,:}$ (Query) against $\mathbf{H}_{i+1}^{knw,:}$ (Key) fails because they represent different times, distorting concurrent correlation matching.

---

## Technical Formulation

KV Shift resolves this by decoupling the temporal steps used for the **Keys** and **Values** in cross-variate attention:

$$\text{Key at step } i: ~~~ \mathbf{H}_i^{k,:} = \left[\mathbf{H}_i^{tgt,:}, ~ \mathbf{H}_i^{obs,:}, ~ \mathbf{H}_i^{knw,:}\right]$$

$$\text{Value at step } i: ~~~ \mathbf{H}_i^{v,:} = \left[\mathbf{H}_i^{tgt,:}, ~ \mathbf{H}_i^{obs,:}, ~ \mathbf{H}_{i+1}^{knw,:}\right]$$

where:
- $\mathbf{H}_i^{tgt,:}$ represents target embeddings at step $i$.
- $\mathbf{H}_i^{obs,:}$ represents observed (past-only) covariate embeddings at step $i$.
- $\mathbf{H}_i^{knw,:}$ represents known covariate embeddings. **In the Value matrix, the known covariate slice is shifted forward by exactly one step ($\mathbf{H}_{i+1}^{knw,:}$)**.

The attention computation is formulated as:

$$\widetilde{\mathbf{H}}_{i}^{tgt,:} = \operatorname{LN}\left(\mathbf{H}_{i}^{tgt,:} + \operatorname{MHA}\left(\mathbf{H}_{i}^{tgt,:}, ~ \mathbf{H}_i^{k,:}, ~ \mathbf{H}_i^{v,:}\right)\right)$$

### **Why this prevents data leakage:**
1. **Concurrent Key-Query Matching**: The Query ($\mathbf{Q}_i = \mathbf{W}_q \mathbf{H}_i^{tgt,:}$) dot-products with the concurrent Key ($\mathbf{H}_i^{k,:}$) which represents variables strictly at step $i$. This maintains exact temporal alignment for capturing semantic correlations.
2. **Shifted Value Information Flow**: Once the attention weights are computed, they multiply the Value matrix ($\mathbf{H}_i^{v,:}$). For the known covariates, this retrieves tomorrow's data ($\mathbf{H}_{i+1}^{knw,:}$), safely injecting tomorrow's external plans into today's target embedding to guide the prediction of tomorrow's target.
3. **Selective Shift**: Targets and observed covariates in the Value matrix are **not shifted**, mathematically preventing lookahead leakage.

---

## See Also
- [[CITRAS Summary]]
- [[Group Attention]]
- [[Attention Score Smoothing]]
- [[TimesFM XReg Modes]]
