---
tags: [llm, architecture, residual-connections, identity-mapping]
aliases: [Residual Connection, Skip Connection, ResNet]
date: 2026-05-18
sources: ["[[wiki/research/mHC Summary.md]]", "[[wiki/entities/Transformers.md]]"]
---

# Residual Connections

Residual connections (or skip connections) are a foundational architectural component in deep learning, introduced by ResNet, and heavily utilized in [[Transformers]].

## The Identity Mapping Property

In standard deep neural networks, a single-layer residual connection is formulated as:
$$
\mathbf{x}_{l+1}=\mathbf{x}_{l}+\mathcal{F}(\mathbf{x}_{l},\mathcal{W}_{l})
$$
where $\mathbf{x}_{l}$ is the input feature, and $\mathcal{F}$ is the residual function (such as [[Self-Attention Mechanism|Self-Attention]] or a Feed-Forward Network).

The defining characteristic of this equation is that the coefficient of $\mathbf{x}_{l}$ is exactly **1**. When expanded recursively across multiple layers from $l$ to $L$:
$$
\mathbf{x}_{L}=\mathbf{x}_{l}+\sum_{i=l}^{L-1}\mathcal{F}(\mathbf{x}_{i},\mathcal{W}_{i})
$$

This guarantees the **identity mapping property**. Signals from earlier layers map directly to deeper layers without scaling or modification. During backpropagation, this creates a direct path for gradients to flow to early layers, preventing the vanishing gradient problem and allowing networks to scale to hundreds of layers.

## Modifications in LLMs

### Pre-Norm vs. Post-Norm
In original Transformers, Layer Normalization was applied after the residual addition (Post-Norm). Modern LLMs use a Pre-Norm architecture (often using [[RMSNorm]]), where normalization is applied to the input *before* the residual function $\mathcal{F}$, but the residual stream $\mathbf{x}_{l}$ itself remains un-normalized until the final output. This further stabilizes training.

### Hyper-Connections (HC) and mHC
Recent research aims to increase the width of the residual stream (to $n \times C$) without adding computational FLOPs to the attention or FFN blocks. 
*   **Hyper-Connections (HC)** uses learnable matrices to route the wide residual stream through narrow functions, but its unconstrained nature destroys the identity mapping, leading to severe instability.
*   **[[mHC|Manifold-Constrained Hyper-Connections (mHC)]]** solves this by projecting the residual mixing matrix onto the Birkhoff polytope (making it a doubly stochastic matrix), which strictly bounds its spectral norm and restores the stabilizing identity mapping property.