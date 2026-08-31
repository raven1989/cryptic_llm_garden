---
tags: [positional-encoding, attention, concept]
aliases: [Attention with Linear Biases, ALiBi Positional Encoding]
date: 2026-04-21
sources: ["[[大模型位置编码-ALiBi位置编码 Summary]]"]
---
# ALiBi (Attention with Linear Biases)

**ALiBi** stands for **Attention with Linear Biases**. It is a [[Positional Encoding]] technique introduced to improve the input length extrapolation capabilities of Large Language Models (LLMs), allowing them to evaluate on sequences much longer than they were trained on.

## Core Idea
Unlike traditional methods, ALiBi does not add positional embeddings to the initial word embeddings at the bottom of the network. Instead, it directly biases the query-key attention scores by applying a penalty that is proportional to the relative distance between the tokens. 

This means that the further apart two tokens are, the lower their mutual contribution to the attention score, naturally focusing the attention mechanism on more recent/closer context.

## Equation
The attention score before the softmax operation is calculated as:
$$ \text{Softmax} \left( q_i K^T + m [-(i-1), \dots, -2, -1, 0] \right) $$

- $m$ is a head-specific scaling factor (slope). It is a fixed, non-learned scalar that determines the magnitude of the distance penalty for a given attention head. 
- For a model with $n$ heads, the $m$ values form a geometric sequence starting from $2^{\frac{-8}{n}}$. For example, with 8 attention heads, the slopes would be: $\frac{1}{2^1}, \frac{1}{2^2}, \dots, \frac{1}{2^8}$.

## Matrix Diagram representation

Below is a mathematical representation of how the relative distance bias ($m$) is added to the unmasked elements of the standard attention query-key dot product matrix before Softmax is applied.

$$
\begin{aligned}
\text{Original Scores } (q_i \cdot k_j) \quad \quad \quad \quad \text{Linear Bias } (\text{Dist} \times m) \quad \quad \quad \quad \quad \text{Biased Scores (Softmax Input)} \\
\begin{matrix}
\mathbf{q}_1 \\ \mathbf{q}_2 \\ \mathbf{q}_3 \\ \mathbf{q}_4
\end{matrix}
\begin{bmatrix}
q_1{\cdot}k_1 & \text{-inf} & \text{-inf} & \text{-inf} \\
q_2{\cdot}k_1 & q_2{\cdot}k_2 & \text{-inf} & \text{-inf} \\
q_3{\cdot}k_1 & q_3{\cdot}k_2 & q_3{\cdot}k_3 & \text{-inf} \\
q_4{\cdot}k_1 & q_4{\cdot}k_2 & q_4{\cdot}k_3 & q_4{\cdot}k_4
\end{bmatrix}
+
\begin{bmatrix}
0 & \text{-inf} & \text{-inf} & \text{-inf} \\
-1 & 0 & \text{-inf} & \text{-inf} \\
-2 & -1 & 0 & \text{-inf} \\
-3 & -2 & -1 & 0
\end{bmatrix} \times m
=
\begin{bmatrix}
q_1{\cdot}k_1 & \text{-inf} & \text{-inf} & \text{-inf} \\
q_2{\cdot}k_1 - m & q_2{\cdot}k_2 & \text{-inf} & \text{-inf} \\
q_3{\cdot}k_1 - 2m & q_3{\cdot}k_2 - m & q_3{\cdot}k_3 & \text{-inf} \\
q_4{\cdot}k_1 - 3m & q_4{\cdot}k_2 - 2m & q_4{\cdot}k_3 - m & q_4{\cdot}k_4
\end{bmatrix}
\end{aligned}
$$

*Notes:*
- `-inf` cells represent masked out values for autoregressive causal modeling.
- The scalar $m$ is mathematically pulled out of the constant linear bias matrix, multiplying against the relative distance.
- The attention scores ($q_i \cdot k_j$) are explicitly shown as the dot product between queries and keys.
- Positional embeddings are omitted from the initial embeddings entirely.

## Related Pages
- [[Positional Encoding]]: The high-level overview page comparing ALiBi with Sinusoidal and RoPE.
- [[RoPE]]: The mainstream rotary position encoding that does *not* extrapolate unmodified — contrast with ALiBi's built-in extrapolation.
- [[Long-Context Positional Encoding]]: Deep dive on RoPE extrapolation fixes, where ALiBi serves as the naturally-extrapolating contrast case.
- [[Pre-training Large Language Models]]: Chapter 15 cites ALiBi (with T5 bias and xPos) as an example of position encodings with extrapolation ability in long-context modeling.