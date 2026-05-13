---
tags: [llm, architecture, normalization]
aliases: [Root Mean Square Normalization]
date: 2026-05-12
sources: ["[[raw/LLM/大模型原理与架构/03_components/3.6_layer_norm.md]]"]
---

# RMSNorm (Root Mean Square Normalization)

**RMSNorm** is a streamlined and highly efficient variant of standard Layer Normalization. Introduced as a computational optimization, it has become the standard normalization technique for modern Large Language Models (including Llama, Mistral, and Gemma).

## The Core Insight

Standard Layer Normalization performs two distinct operations on the inputs to normalize the distribution of activations:
1. **Mean-centering:** Subtracting the mean to zero-center the data.
2. **Variance scaling:** Dividing by the standard deviation. 

Additionally, LayerNorm learns two parameters: $\gamma$ (scaling) and $\beta$ (offset).

Researchers discovered a key insight: **the mean-centering step contributes almost nothing to model performance** in [[Transformers]]. Because the network's residual connections already do a good job of keeping the data distributions stable, subtracting the mean is computationally wasteful. The true benefit of normalization comes almost entirely from the variance scaling.

## Mathematical Formulation

RMSNorm simplifies the process by dropping the mean calculation and the $\beta$ offset parameter entirely. It only scales the activations using the Root Mean Square (RMS) of the inputs.

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}} \cdot \gamma$$

Where:
* $x$ is the input vector.
* $d$ is the dimensionality of the vector.
* $\epsilon$ is a small constant for numerical stability.
* $\gamma$ is the learnable scaling parameter.

## Why Modern LLMs Use RMSNorm

By skipping the mean calculation, RMSNorm offers several distinct advantages over standard LayerNorm:
* **Faster Computation:** It removes a full reduction operation (calculating the mean) across the hidden dimension, which speeds up the forward pass. Overall computational overhead of the normalization step drops by about 10–15%.
* **Fewer Parameters:** It only requires learning the $\gamma$ parameter, cutting the normalization parameter count in half.
* **Equal Performance:** Despite being mathematically simpler, empirical benchmarks show that RMSNorm converges just as well and yields the same model quality as standard LayerNorm.

## Pre-Norm Architecture

In modern architectures, RMSNorm is almost always used in a **Pre-Norm** configuration. Instead of normalizing the output of the attention and feed-forward blocks (Post-Norm), it normalizes the *input* to these blocks:

$$\text{output} = x + \text{Sublayer}(\text{RMSNorm}(x))$$

This combination (Pre-Norm + RMSNorm) allows the gradient to flow directly through the identity (residual) path without being interrupted by the normalization layer, which prevents early gradient explosion in very deep networks and allows for much more stable training.