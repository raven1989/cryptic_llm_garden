---
title: "Router Z-Loss"
tags:
  - math
  - loss_function
  - moe
  - optimization
sources:
  - "[[raw/LLM/ST-MoE - Designing Stable and Transferable Sparse Expert Models.pdf]]"
aliases:
  - z-loss
date: 2026-05-21
---
# Router Z-Loss

**Source:** Introduced in [[ST-MoE Summary|ST-MoE: Designing Stable and Transferable Sparse Expert Models]]

## Concept
The Router Z-Loss is an auxiliary loss term designed specifically to stabilize the training of large-scale [[Sparsely-Gated MoE Layer|Mixture-of-Experts (MoE)]] models. It solves training divergence caused by massive floating-point roundoff errors during the routing phase.

## The Problem
In an MoE layer, the router calculates a probability distribution over the available experts by passing raw logit scores through a softmax function. This requires calculating the exponential ($e^x$) of the logits.

As MoE models scale, the absolute value of these logits can become very large. When exponentiating large numbers—especially in mixed-precision formats like `bfloat16`, which has up to 65,536x worse roundoff errors than `float32`—the network suffers from severe floating-point roundoff errors. These errors distort probability scaling and routing thresholds, ultimately causing the training loss to spike and diverge.

## Mathematical Formulation
Rather than arbitrarily clipping the logits (which introduces harmful mathematical discontinuities), the Router Z-Loss *penalizes the network for producing large routing logits in the first place*.

The loss is defined mathematically as:

$$L_z(x) = \frac{1}{B} \sum_{i=1}^B \left(\log \sum_{j=1}^N e^{x_{ij}}\right)^2$$

### Variable Breakdown:
*   **$L_z(x)$**: The router z-loss computed for the given input logits $x$. 
*   **$B$**: The total number of **tokens** in the current training batch.
*   **$N$**: The total number of **experts** in this specific MoE layer.
*   **$i$**: The index iterating over each token in the batch (from $1$ to $B$).
*   **$j$**: The index iterating over each expert (from $1$ to $N$).
*   **$x_{ij}$**: The **logit** (the raw, unnormalized score) predicted by the routing network for token $i$ being assigned to expert $j$. 
*   **$e^{x_{ij}}$**: The exponentiated logit. (The first step of the softmax function).
*   **$\sum_{j=1}^N e^{x_{ij}}$**: The sum of the exponentiated logits for token $i$ across all $N$ experts. (The denominator of the softmax function, representing total unnormalized probability mass).
*   **$\log \left( \sum_{j=1}^N e^{x_{ij}} \right)$**: The natural logarithm of the sum above, mathematically known as the **Log-Sum-Exp (LSE)** function. 
*   **$(\dots)^2$**: Squaring the Log-Sum-Exp function ensures that the penalty grows quadratically as the values increase.
*   **$\frac{1}{B} \sum_{i=1}^B \dots$**: Averages the penalty across all tokens in the batch so the loss scales stably regardless of batch size.

## Intuition
The Log-Sum-Exp function is a smooth mathematical approximation of the maximum function: $\log \sum e^{x} \approx \max(x)$.

By heavily penalizing the squared Log-Sum-Exp of the logits, the Router Z-Loss directly penalizes the largest logit for each token. The network learns to push its maximum logit values toward zero, keeping all raw router scores small. 

* Small logits $\rightarrow$ Small inputs to the exponential function $\rightarrow$ Accurately modeled numbers with no catastrophic roundoff errors.

During training, this auxiliary loss is multiplied by a small coefficient (e.g., $c_z = 0.001$) and added to the total loss alongside the cross-entropy loss and the expert load-balancing loss. Using this loss was shown to completely resolve instability in 269B parameter sparse models without degrading model quality.