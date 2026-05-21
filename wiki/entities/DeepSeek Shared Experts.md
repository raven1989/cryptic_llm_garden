---
title: "DeepSeek Shared Experts"
tags:
  - architecture
  - moe
  - LLM
  - research
sources:
  - "[[raw/LLM/DeepSeek-V2 - A Strong, Economical, and Efficient Mixture-of-Experts Language Model.pdf]]"
  - "[[raw/LLM/DeepSeek-V3 Technical Report.pdf]]"
aliases:
  - Shared Experts
  - Shared Expert Isolation
  - DeepSeekMoE
date: 2026-05-21
---
# Shared Experts (DeepSeekMoE)

**Source:** [[raw/LLM/DeepSeek-V2 - A Strong, Economical, and Efficient Mixture-of-Experts Language Model.pdf]]

## Overview
Shared Experts are a core architectural innovation of **DeepSeekMoE**, introduced in the DeepSeek-V2 and DeepSeek-V3 models. In a traditional [[Sparsely-Gated MoE Layer|Mixture-of-Experts (MoE)]] layer, all experts are "routed"—meaning a token is only sent to an expert if the routing mechanism selects it. 

The DeepSeekMoE architecture modifies this by splitting the Feed-Forward Network (FFN) experts into two distinct categories:
1. **Routed Experts:** Fine-grained experts that are sparsely selected via a gating mechanism (like standard MoE).
2. **Shared Experts:** A small set of experts that are *always* activated for *every* token, bypassing the routing mechanism entirely.

![[wiki/media/deepseek_shared_experts.png]]
*(The DeepSeekMoE Architecture. Notice how the Shared Expert is always active, while the Routed Experts are gated.)*

## The Intuition: "Shared Expert Isolation"
In traditional MoE architectures (like [[GShard]] or [[Switch Transformer]]), the routed experts are forced to learn both:
* **General, broad context:** Syntax, common grammar, base semantics.
* **Specialized context:** Niche domain knowledge, specific facts, complex reasoning paths.

Because every token needs basic syntactic and semantic processing, traditional routed experts end up storing a massive amount of redundant, generalized knowledge across all experts. 

**Shared Expert Isolation** solves this redundancy. By forcing every token to pass through a few Shared Experts, the model naturally pushes generalized, common knowledge into the shared parameters. This frees up the Routed Experts to become highly specialized ("finer granularity"), as they no longer need to waste capacity storing common knowledge.

## Mathematical Formulation
Let $\mathbf{u}_t$ be the input to the MoE Feed-Forward Network for the $t$-th token. In the DeepSeekMoE architecture, the final output $\mathbf{h}'_t$ is computed as the sum of the residual connection, the output of the Shared Experts, and the probabilistically weighted output of the Routed Experts.

$$ \mathbf{h}'_t = \mathbf{u}_t + \sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t} \text{FFN}_i^{(r)}(\mathbf{u}_t) $$

### Variable Breakdown:
*   **$\mathbf{u}_t$**: The input hidden state for the $t$-th token.
*   **$\mathbf{h}'_t$**: The final output hidden state for the $t$-th token.
*   **$N_s$**: The total number of **Shared Experts** in the layer. (These are always active).
*   **$N_r$**: The total number of **Routed Experts** in the layer.
*   **$\text{FFN}_i^{(s)}(\cdot)$**: The computation of the $i$-th Shared Expert.
*   **$\text{FFN}_i^{(r)}(\cdot)$**: The computation of the $i$-th Routed Expert.
*   **$g_{i,t}$**: The gating value (probability score) assigned by the router for the $t$-th token to the $i$-th Routed Expert. If the expert is not in the Top-$K$, this value is $0$.

### The Routing Mechanism
The gating value $g_{i,t}$ determines whether a routed expert is activated. 

$$ 
g_{i,t} = 
\begin{cases} 
s_{i,t}, & \text{if } s_{i,t} \in \text{TopK}(\{ s_{j,t} \mid 1 \leqslant j \leqslant N_r \}, K_r) \\
0, & \text{otherwise}
\end{cases}
$$

Where $s_{i,t}$ is the token-to-expert affinity score (calculated via softmax), and $K_r$ is the number of routed experts activated per token.

## Implementation Details (DeepSeek-V2)
To achieve this, DeepSeek heavily segments their experts into much finer granularity. 
* Instead of having a few massive experts (e.g., 8 large experts like Mixtral), DeepSeek-V2 has **160 highly segmented Routed Experts** per MoE layer.
* The model activates **$K_r = 6$** of these 160 routed experts per token.
* Alongside the 160 routed experts, there are **$N_s = 2$ Shared Experts**.
* Every single token passes through the 2 Shared Experts + the 6 selected Routed Experts.

Because the experts are fine-grained (smaller hidden dimensions), activating 8 total experts per token keeps the active parameter count highly economical (21B active out of 236B total) while drastically improving specialization.