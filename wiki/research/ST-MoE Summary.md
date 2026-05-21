---
title: "ST-MoE - Designing Stable and Transferable Sparse Expert Models Summary"
tags:
  - summary
  - moe
  - LLM
  - architecture
  - research
sources:
  - "[[raw/LLM/ST-MoE - Designing Stable and Transferable Sparse Expert Models.pdf]]"
aliases:
  - ST-MoE
date: 2026-05-21
---
# ST-MoE Summary

**Source:** [[raw/LLM/ST-MoE - Designing Stable and Transferable Sparse Expert Models.pdf]]

## Overview
ST-MoE (Stable and Transferable Mixture-of-Experts) is a 269B parameter sparse expert model designed to address the severe training instabilities that plagued earlier large-scale Mixture-of-Experts (MoE) architectures (like the [[Switch Transformer]] and [[GShard]]). As sparse models scaled up to match the computational footprint of massive dense models, pre-training was consistently hampered by loss spikes and divergences that were not present at smaller scales.

The ST-MoE paper systematically categorizes these instability issues, analyzing the trade-offs of various stabilization techniques, and proposes a highly effective, quality-preserving solution: the [[Router Z-Loss]].

## The Root Cause of MoE Instability
The authors identified that the primary source of instability in sparse models stems from the exponential functions inside the routing mechanism. 

MoE routers determine which experts process a token by passing raw scores (logits) through a softmax function to generate a probability distribution. Softmax requires calculating the exponential ($e^x$) of these logits. 

At large scales, the absolute values of the router logits can grow very large. When exponentiated, these large numbers encounter severe floating-point roundoff errors. This is exacerbated because modern distributed Transformers rely on mixed-precision training formats like `bfloat16`. 
* `bfloat16` represents numbers using fewer mantissa bits (7 bits compared to `float32`'s 23 bits).
* Because of this, `bfloat16` has up to 65,536x worse roundoff errors than `float32`.
* The larger the number, the larger the absolute roundoff error.

While a roundoff error might not change the top-1 sorting order of a softmax, it severely impacts relative thresholding (e.g., routing a token to its 2nd expert only if its probability is a certain fraction of the 1st) and alters the probability weights used to scale the final expert outputs. These cascading errors ultimately destabilize the network.

## The Solution: Router Z-Loss
The researchers found that standard stability fixes (like tight gradient update clipping, using smaller learning rates, or removing multiplicative interactions like RMSNorm) successfully stabilized training but caused an untenable degradation in final model quality. Furthermore, manually clipping the logits before the softmax acts as a harsh mathematical discontinuity that also harms training.

To solve this, they introduced the [[Router Z-Loss]], an auxiliary penalty added to the total training loss. Instead of hard-clipping logits, the z-loss penalizes the network for producing large routing logits in the first place, encouraging the model to keep its unnormalized scores small and entirely bypassing the exponential roundoff errors.

*(For the mathematical formulation and breakdown of this loss, see the [[Router Z-Loss]] entity page).*

## Fine-Tuning and Generalization
ST-MoE also highlights that sparse models face a generalization problem when fine-tuning. Sparse models and dense models have distinct hyperparameter sensitivities. Using fine-tuning hyperparameters optimized for a dense model on a sparse model often masks the gains of sparse pre-training, resulting in little to no improvement. Specifically, sparse models require different batch sizes and learning rates during fine-tuning to prevent rapid overfitting, particularly on smaller datasets.

## Architectural Improvements
The paper also explores the injection of more multiplicative interactions into the network. Adding multiplicative biases or using `GEGLU` instead of `ReLU` in the expert feed-forward networks improved pre-training speed by roughly 4% with no step-time overhead. While these multiplicative components naturally worsen model stability, applying the Router Z-Loss prevented any subsequent training divergence.