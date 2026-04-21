---
tags: [machine-learning, transformers, attention, statistics]
aliases: [Scaled Dot-Product Attention, Attention Scaling Factor]
date: 2026-04-14
sources: ["[[raw/transformer/Why is Attention divided by √dₖ.md]]"]
---

# Why is Attention divided by Root d_k

This page summarizes the mathematical and statistical reasoning behind the $\sqrt{d_k}$ scaling factor in the [[Self-Attention Mechanism]], based on the article *"Why is Attention divided by √dₖ: The Secret Behind Scaled Attention in Transformers"* by Srivatsa Narasimha.

## Core Problem: Variance Growth in High Dimensions
In the self-attention formula $Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$, the dot product of a Query vector ($q$) and a Key vector ($k$) is taken before applying the softmax function.

$d_k$ represents the dimensionality of the Key (and Query) vectors.

Statistically, if we assume the components of $q$ and $k$ are independent random variables with a mean of $0$ and a variance of $1$:
1. The variance of the product of two components ($q_i \times k_i$) is $1 \times 1 = 1$.
2. The dot product $q \cdot k$ is the sum of $d_k$ such independent terms.
3. According to the properties of variance, the variance of the sum of independent variables is the sum of their variances. Thus, the variance of the raw dot product $q \cdot k$ becomes $d_k$.

As the dimensionality $d_k$ grows (e.g., to 64 in the original Transformer), the variance of the dot products grows linearly with it.

## The Consequence: Softmax Saturation and Vanishing Gradients
If these high-variance dot products are fed directly into the Softmax function, they push the Softmax into extreme regions:
- The outputs become "peaky" (nearly one-hot), where one token dominates the attention distribution.
- In these extreme regions, the gradients of the Softmax function approach zero.
- This leads to the **vanishing gradient problem**, causing the model to stop learning because weight updates become infinitesimally small.

## The Solution: Standardization
To fix this, the dot product must be standardized back to a **unit variance** (variance of 1). 
In statistics, dividing a random variable by a constant $c$ divides its variance by $c^2$. Therefore, to reduce the variance from $d_k$ to $1$, the dot product must be divided by the standard deviation, which is $\sqrt{d_k}$.

This scaling step ensures that the attention scores remain bounded and stable, allowing for a smooth attention distribution and effective gradient flow regardless of the model's dimensional size.