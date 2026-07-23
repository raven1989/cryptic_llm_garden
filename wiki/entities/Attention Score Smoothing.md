---
title: "Attention Score Smoothing"
aliases: ["Attention Smoothing", "Temporal Attention Smoothing"]
date: 2026-07-23
tags: ["entity", "attention", "transformer", "time-series", "smoothing"]
sources: ["[[wiki/research/CITRAS Summary.md]]"]
---

# Attention Score Smoothing

**Attention Score Smoothing** is an attention regularization mechanism introduced in the **[[CITRAS Summary|CITRAS]]** forecasting model. It smooths step-wise, cross-variate attention maps across time using an Exponential Moving Average (EMA). This stabilizes the model's variable dependency estimations, establishing global relationships while filtering localized noise and sensor outliers.

---

## Technical Motivation

When computing cross-variate attention independently at each temporal patch step $i$, two critical issues occur:
1. **Local Noise**: High-frequency spikes or sensor glitches in a single step distort the token embeddings, causing the query-key dot-product matching to hallucinate or miss correlations.
2. **Sparse Constant Covariates**: Binary event indicators (e.g., holiday or discount flags) are often filled with constant zeros across several temporal patches. During these inactive patches, the cross-variate attention score collapses because the constant input offers no distinct feature similarity to target tokens, causing the model to "forget" the global covariate correlation.

---

## Technical Formulation

To bridge local step-wise precision with global consistency, Attention Score Smoothing applies an **Exponential Moving Average (EMA)** to raw attention scores across sequential patch steps.

For a raw dot-product attention score matrix $\widetilde{\mathbf{A}}_i$ at patch step $i$:

$$\mathbf{A}_i = \alpha \widetilde{\mathbf{A}}_i + (1 - \alpha) \mathbf{A}_{i-1}, \quad i = 2, \dots, N_{tgt}$$

where:
- $\mathbf{A}_1 = \widetilde{\mathbf{A}}_1$
- $\alpha \in [0, 1]$ is a shared smoothing factor (hyperparameter).
- $\mathbf{A}_i$ is the smoothed attention score matrix used to multiply the shifted Values ($\mathbf{H}_i^{v,:}$).

### **Key Advantages:**
1. **Correlation Memory**: The smoothed score carries historical variable-interaction states forward. A sparse holiday indicator is remembered even in flat segments where the indicator is all zeros.
2. **Noise Low-Pass Filter**: Erroneous spikes in the raw step score $\widetilde{\mathbf{A}}_i$ are attenuated by the term $(1-\alpha)\mathbf{A}_{i-1}$.
3. **Smooth Adaptability**: Unlike global averaging, the EMA decay rate allows CITRAS to adapt to slow, gradual shifts in correlations (e.g., gradual seasonality transitions) over longer timelines.

---

## See Also
- [[CITRAS Summary]]
- [[KV Shift]]
- [[TimesFM XReg Modes]]
