---
title: "Group Attention"
aliases: ["Group Attention Mechanism", "Group Masking"]
date: 2026-07-23
tags: ["entity", "attention", "transformer", "time-series"]
sources: ["[[wiki/research/Chronos-2 Summary.md]]"]
---

# Group Attention

**Group Attention** is an attention pooling mechanism introduced in **[[Chronos-2 Summary|Chronos-2]]** that facilitates in-context learning (ICL) by sharing information across multiple related time series or covariates. It operates as a specialized self-attention layer that dynamically isolates information routing to specific groups within a batch.

---

## Architectural Purpose

In multivariate or covariate-informed forecasting, models need to share representation details across variables (e.g., target series, promotional flags, or weather metrics) to infer mutual dynamics. However, in a large batched input, unrelated series or tasks must be prevented from cross-contaminating each other. 

Group Attention resolves this by using **Group IDs** instead of traditional static causal masks.

---

## Technical Formulation

Let $\mathbf{H} \in \mathbb{R}^{B \times D_{\text{model}}}$ be the token embeddings of a batched sequence at a specific temporal index, where $B$ is the batch size. Every item in the batch is assigned a group ID $\mathbf{g} = (g_1, g_2, \dots, g_B)$.

1. **2D Mask Construction**:
   The vector of group IDs $\mathbf{g}$ is mapped to a two-dimensional binary attention mask $\mathbf{M} \in \{0, 1\}^{B \times B}$:
   
   $$\mathbf{M}_{u, v} = \begin{cases} 
   1 & \text{if } g_u = g_v \\ 
   0 & \text{otherwise} 
   \end{cases}$$
   
2. **Attention Routing**:
   The standard softmax self-attention is masked using $\mathbf{M}$:
   
   $$\operatorname{GroupAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}} + \mathbf{M}'\right)\mathbf{V}$$
   
   where $\mathbf{M}'_{u, v} = 0$ if $\mathbf{M}_{u, v} = 1$, and $-\infty$ otherwise. This guarantees that token representation aggregation occurs *strictly* within defined groups and never leaks across unrelated tasks or batches.

3. **Permutation Invariance**:
   Because there is no natural spatial ordering between different channels or covariates within a group, Group Attention **omits position embeddings** (such as RoPE or absolute embeddings), making the cross-variate representation pooling permutation-invariant.

---

## Dynamic Application Modes

By modifying the Group IDs vector $\mathbf{g}$, Group Attention can be reconfigured dynamically at inference time:

- **Univariate Mode**: Every channel has a unique group ID $\mathbf{g} = (1, 2, 3)$. No variable interacts with any other.
- **Multivariate Mode**: Channels sharing the same physical system share a group ID $\mathbf{g} = (1, 1, 1)$.
- **Covariate Mode**: Targets, past-only covariates, and known future covariates share a group ID $\mathbf{g} = (1, 1, 1, 1)$. The target query attends to the covariates' keys and values to capture external influences.

---

## See Also
- [[Chronos-2 Summary]]
- [[KV Shift]]
- [[TimesFM XReg Modes]]
