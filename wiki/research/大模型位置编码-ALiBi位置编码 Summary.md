---
tags: [positional-encoding, alibi, summary, length-extrapolation]
aliases: []
date: 2026-04-21
sources: ["raw/LLM/大模型位置编码-ALiBi位置编码.md"]
---
# 大模型位置编码-ALiBi位置编码 Summary

This is a summary of the source document regarding [[ALiBi]] (Attention with Linear Biases).

## Problem Addressed: Length Extrapolation
Large language models trained with absolute positional encodings (like BERT) or even functional ones (like Sinusoidal or [[RoPE]]) struggle with **length extrapolation**—the ability to process sequences during inference that are longer than the sequences seen during training. As sequences get longer, the attention mechanism distributes focus across more tokens, increasing entropy (more uniform attention) and degrading performance (perplexity spikes). 

## The ALiBi Solution
The paper *"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"* introduces a simple but highly effective method called **[[ALiBi]]**.

Instead of adding positional embeddings to the token embeddings at the bottom of the network, ALiBi directly modifies the attention scores:
1. It computes the standard query-key dot products ($q_i \cdot k_j$).
2. It subtracts a predefined, constant bias matrix based on the relative distance between the query and key. 
3. The further apart the tokens are, the higher the penalty applied to their attention score.

This ensures that closer tokens naturally receive higher attention weights, allowing the model to extrapolate smoothly to sequences significantly longer than its training context window without a sudden spike in perplexity.