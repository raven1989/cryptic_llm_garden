---
title: "Speculative Decoding"
tags:
  - research
  - optimization
  - inference
  - speculative-decoding
  - mathematics
sources: ["[[raw/LLM/Looking back at speculative decoding.md]]", "[[raw/LLM/An Introduction to Speculative Decoding for Reducing Latency in AI Inference.md]]"]
aliases:
  - Speculative Decoding
  - Speculative Sampling
  - Draft-Target Decodes
  - Assisted Generation
date: 2026-05-25
---

# Speculative Decoding

**Speculative Decoding** is an inference optimization paradigm that significantly accelerates autoregressive Large Language Model (LLM) generation without compromising the output quality. By executing several token computations in parallel, speculative decoding collapses the execution times of sequential decoding steps while mathematically guaranteeing that the final output distribution remains identical to that of standard autoregressive generation.

---

## 1. The Core Bottleneck: The Memory Bandwidth Wall

Autoregressive text generation produces tokens one-by-one, requiring a complete forward pass of the model for each generated token. 

### Why Standard Decoding Underutilizes GPUs (Memory-Bound)
During the **Decode stage**, the sequence length of the new input is exactly $1$. The GPU must perform Matrix-Vector (GEMV) multiplications: it loads the massive, static model parameters (weights) from High-Bandwidth Memory (HBM) into its local SRAM cache, performs a tiny amount of computation against the single token vector, and writes the output back to HBM.

*   **GPU Peak FLOPs:** Modern GPUs can perform hundreds of trillions of floating-point operations per second (e.g., $10^{14}$ FLOPs/sec).
*   **HBM Bandwidth:** GPU memory bandwidth is typically around a few terabytes per second (e.g., $3 \times 10^{12}$ bytes/sec).
*   **The Arithmetic Intensity:** During decoding, a model performs only about 2 floating-point operations per byte of memory transferred. Thus, the compute cores are idle **$\approx 99\%$ of the time**, waiting for weights to finish loading from memory.

### The Speculative Breakthrough (Compute-Bound Parallelism)
Speculative decoding breaks this bottleneck by replacing multiple sequential, memory-bound steps with a **single, parallel compute-bound step**. 

If a fast, lightweight mechanism can guess $K$ future candidate tokens (a "draft"), we can feed all $K$ tokens into the massive "target" model simultaneously. Computing the target model's forward pass on $K$ tokens at once uses Matrix-Matrix (GEMM) multiplications. Because of the GPU's highly parallel architecture, this parallel pass takes **virtually the same amount of time** as generating a single token, essentially utilizing the GPU's idle compute cores for "free."

---

## 2. Classic Draft-Target Speculative Decoding

The classic approach (Leviathan et al., 2022) operates as a **two-model system**:
1.  **Target Model ($\mathcal{M}_{\text{target}}$):** The massive, highly intelligent model we want to accelerate (e.g., 70B parameters).
2.  **Draft Model ($\mathcal{M}_{\text{draft}}$):** A much smaller, fast model trained on the same data distribution (e.g., 1B parameters).

```text
               DRAFT MODEL (Fast)                         TARGET MODEL (Heavy)
               
         Generates K tokens sequentially            Verifies all K tokens in parallel
         
  [t0] ──► [ M_draft ] ──► t'_1, t'_2, ... t'_K ──► [ M_target ] ──► Accept / Reject
```

### Step-by-Step Execution Loop
1.  **Draft Generation:** The lightweight draft model $\mathcal{M}_{\text{draft}}$ runs autoregressively for $K$ steps, producing a candidate prefix:
    $$\hat{t}_1, \hat{t}_2, \dots, \hat{t}_K \sim q(t)$$
    *Where $q(t)$ is the probability distribution of the draft model.*
2.  **Parallel Verification:** The target model $\mathcal{M}_{\text{target}}$ processes the original prompt and all $K$ draft tokens simultaneously in a single parallel forward pass. This yields $K+1$ probability distributions:
    $$P_1, P_2, \dots, P_{K+1}$$
    *Where $P_i(t) = p(t_i \ | \ \text{prompt}, t_1, \dots, t_{i-1})$ represents the target model's true probability distribution.*
3.  **Rejection Sampling:** The system evaluates each candidate token in order, accepting or discarding them using speculative sampling mathematics.

---

## 3. Speculative Sampling Mathematics

To guarantee that the final generated output matches the exact probability distribution of the target model (preserving creativity and reasoning depth), we cannot rely on simple deterministic match-checks. Instead, we use **Speculative Sampling** (Leviathan et al., 2022).

This section provides the complete, mathematically rigorous proof of speculative sampling's equivalence to the target model, detailing the underlying probability derivations.

### 3.1. The Acceptance and Rejection Mechanics

Let:
*   $q(x)$ be the probability distribution of the draft model.
*   $p(x)$ be the probability distribution of the target model.
*   $x^*$ be the specific draft token sampled at this step ($x^* \sim q(x)$).

We accept the draft token $x^*$ with an **acceptance probability $\alpha$**:

$$\alpha = \min\left(1, \frac{p(x^*)}{q(x^*)}\right)$$

*   **If Accepted:** We keep $x^*$ and proceed to evaluate the next draft token.
*   **If Rejected:** We discard $x^*$ and all subsequent draft tokens, and sample a replacement token $t_i$ from the **normalized difference correction distribution $p'(x)$**:
    $$p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{y} \max(0, p(y) - q(y))}$$

The correction distribution $p'(x)$ acts as a "gap-filler." It only contains probabilities for tokens that the target model liked more than the draft model did, scaled to sum to $1.0$.

---

### 3.2. Conditional vs. Marginal Rejection (The Law of Total Probability)

A crucial mathematical distinction must be made between **conditional rejection** (the probability of rejecting a specific token *after* it has been proposed) and **marginal rejection** (the total probability of rejecting *any* token *before* the step executes).

#### 1. Conditional Rejection: $1 - \alpha$
Once the draft model has already proposed a specific token $x^*$, the probability of rejecting that specific token is:
$$P(\text{Reject} \mid \text{Drafted } x^*) = 1 - \alpha = 1 - \min\left(1, \frac{p(x^*)}{q(x^*)}\right)$$

#### 2. Marginal Rejection: $P_{\text{reject}}$
To find the total probability of encountering a rejection during this step, we must sum the probability of the draft model choosing each possible token $y$ in the vocabulary multiplied by the probability of rejecting that choice.

By applying the **Law of Total Probability (全概率公式)**:

$$P_{\text{reject}} = \sum_{y} P(\text{Reject} \mid \text{Drafted } y) \cdot P(\text{Drafted } y)$$

$$P_{\text{reject}} = \sum_{y} \left(1 - \min\left(1, \frac{p(y)}{q(y)}\right)\right) \cdot q(y)$$

Let's expand and distribute the sum over the terms:

$$P_{\text{reject}} = \sum_{y} q(y) - \sum_{y} q(y) \min\left(1, \frac{p(y)}{q(y)}\right)$$

Since $q(y)$ is a valid probability distribution, its sum over the entire vocabulary space is exactly $1.0$ ($\sum_{y} q(y) = 1$). Next, we distribute the scalar $q(y)$ inside the $\min$ function:
$$q(y) \min\left(1, \frac{p(y)}{q(y)}\right) = \min\left(q(y) \cdot 1, \ q(y) \cdot \frac{p(y)}{q(y)}\right) = \min(q(y), p(y))$$

Plugging these back in yields the fundamental equation for marginal rejection:

$$P_{\text{reject}} = 1 - \sum_{y} \min(q(y), p(y))$$

*This equation shows that the total probability of rejection is simply $1$ minus the total shared "overlap" area between the draft and target probability distributions.*

---

### 3.3. Set-Theoretic Symmetry of $P_{\text{reject}}$

The marginal rejection probability can be written in two mathematically identical forms:

$$P_{\text{reject}} = \sum_{y} \max(0, q(y) - p(y)) = \sum_{y} \max(0, p(y) - q(y))$$

This symmetry represents the **"Overlap vs. Remainder"** principle. Because both $p(y)$ and $q(y)$ are valid probability distributions, their total volumes are identical ($\sum_y q(y) = \sum_y p(y) = 1$). 

If we split each distribution into two parts—the area where they agree (the overlap) and the remaining areas where they disagree (the remainders):

1.  **Shared Overlap (Agreement):**
    $$\text{Overlap} = \sum_{y} \min(q(y), p(y))$$
2.  **Draft Remainder (Excess of $q$ over $p$):**
    $$\sum_{y} \max(0, q(y) - p(y)) = \sum_{y} q(y) - \text{Overlap} = 1 - \text{Overlap}$$
3.  **Target Remainder (Excess of $p$ over $q$):**
    $$\sum_{y} \max(0, p(y) - q(y)) = \sum_{y} p(y) - \text{Overlap} = 1 - \text{Overlap}$$

Because both remainders equal $1 - \text{Overlap}$, they are mathematically equal:

$$\sum_{y} \max(0, q(y) - p(y)) = \sum_{y} \max(0, p(y) - q(y))$$

This proves that the total volume of over-estimation of $q$ compared to $p$ (which triggers a rejection) is perfectly balanced by the total volume of under-estimation of $q$ compared to $p$ (which is filled by the correction distribution). This allows us to simplify the correction distribution's denominator:

$$\sum_{y} \max(0, p(y) - q(y)) = P_{\text{reject}}$$

Thus, the correction distribution simplifies to:
$$p'(x) = \frac{\max(0, p(x) - q(x))}{P_{\text{reject}}}$$

---

### 3.4. Proof of Equivalence: The Dual-Path Generation Flow

Let's calculate the total probability of generating any specific token $x$ (e.g., the word "fox") under speculative sampling. 

There are only **two mutually exclusive paths** that can lead to generating the token $x$:

```text
                                  ┌──► [Path A: Direct Match] ──► Drafts "fox" ──► Accepts "fox"
                                  │      P(Draft and Accept x) = min(q(x), p(x))
                                  │
  Generate token x (e.g., "fox") ─┤
                                  │
                                  └──► [Path B: Correction] ──► Drafts any "y" ──► Rejects "y" ──► Samples "fox" from p'
                                         P(Reject and Sample x) = P_reject * p'(x)
```

#### Path A: The Draft Model Directly Proposes $x$ and it is Accepted
The probability of the draft model selecting this specific token $x$ and the target model accepting it is:
$$P(\text{Draft and Accept } x) = q(x) \min\left(1, \frac{p(x)}{q(x)}\right) = \min(q(x), p(x))$$
*   **Why there is no summation:** We only care about the draft model proposing this *specific* token $x$. If the draft model proposed some other token $y \neq x$, accepting it would lead to generating $y$, not $x$.

#### Path B: The Draft Model Proposes an Incorrect Token, we Reject it, and the Correction Distribution Selects $x$
The probability of proposing *any* draft token $y$, rejecting it, and then sampling the corrected token $x$ from $p'(x)$ is:
$$P(\text{Reject and Sample } x \text{ from } p') = P_{\text{reject}} \cdot p'(x)$$
*   **Why there is a summation here:** The summation is hidden inside $P_{\text{reject}}$! Because we can arrive at this correction stage by rejecting *any* incorrect draft $y$ in the vocabulary, we must sum over all possible draft selections $y$ to find the total rejection probability (the Law of Total Probability established in Section 3.2).

#### The Total Probability of Outputting Token $x$
We sum the probabilities of these two mutually exclusive paths:

$$P_{\text{gen}}(x) = P(\text{Draft and Accept } x) + P(\text{Reject and Sample } x \text{ from } p')$$

$$P_{\text{gen}}(x) = \min(q(x), p(x)) + P_{\text{reject}} \cdot p'(x)$$

Now, we substitute the simplified correction distribution $p'(x) = \frac{\max(0, p(x) - q(x))}{P_{\text{reject}}}$:

$$P_{\text{gen}}(x) = \min(q(x), p(x)) + P_{\text{reject}} \cdot \left( \frac{\max(0, p(x) - q(x))}{P_{\text{reject}}} \right)$$

The $P_{\text{reject}}$ terms cancel out algebraically:

$$P_{\text{gen}}(x) = \min(q(x), p(x)) + \max(0, p(x) - q(x))$$

Using the algebraic identity $\min(a, b) + \max(0, a - b) = a$ where $a = p(x)$ and $b = q(x)$, the equation simplifies perfectly to:

$$P_{\text{gen}}(x) = p(x)$$

This completes the proof. **Speculative sampling guarantees that the probability of generating any token $x$ is identical to $p(x)$ (the target model's true distribution), making the speedups completely lossless in quality.**

---

## 4. Modern Paradigms of Speculative Decoding

As the field has evolved, several advanced architectural optimizations have replaced the classic separate-model draft-target approach:

### A. Integrated Self-Speculation (Multi-Token Prediction)
Instead of running a separate model, the model itself is pre-trained to speculate. 
*   **Mechanism:** Special auxiliary heads (such as single Transformer layers) are integrated directly into the model during training. The main model generates the first token, and these heads predict sequential future tokens ($t_{i+2}$, $t_{i+3}$) branching from the main model's hidden states.
*   **Example:** [[Multi-Token Prediction]] (MTP) in DeepSeek-V3, which delivers a **1.8x speedup** with an **85% - 90% acceptance rate**.

### B. Feature-Level Extrapolation (EAGLE & EAGLE-3)
Instead of operating in the word/token embedding space, speculation happens in the *internal latent feature space*.
*   **Mechanism:** A lightweight "EAGLE head" (a single Transformer layer) is attached to the internal layers of the target model. It drafts future candidate tokens by autoregressively extrapolating from the target model's top hidden state features rather than decoding to tokens first.
*   **EAGLE-3 Integration:** Merges low, middle, and high-level layer embeddings to form richer feature states, making predictions highly robust.

### C. Tree Attention Verification (EAGLE & Medusa)
Instead of generating a single linear chain of tokens (which fails if even a single early token is rejected), modern speculative decoders generate a **Dynamic Draft Tree**.
*   **Mechanism:** The draft head generates multiple branching hypotheses (e.g., if token 1 could be "is" or "has", it drafts continuations for both).
*   **Parallel Tree Verification:** The target model processes this entire tree in a single forward pass. By modifying the **Attention Mask** (Tree Attention), different branches are prevented from attending to each other, allowing the target model to evaluate dozens of paths simultaneously and select the longest valid branch.

---

## 5. Summary of Speculative Approaches

| Dimension / Metric | Classic Draft-Target (Leviathan et al.) | Medusa | EAGLE-3 | DeepSeek MTP |
| :--- | :--- | :--- | :--- | :--- |
| **Draft Approach** | Separate smaller model | Non-recurrent MLP heads | Feature-extrapolating head | Integrated sequential layers |
| **Draft Structure** | Linear sequence | Linear/Tree | Dynamic Draft Tree | Linear sequence |
| **Verification Method** | Parallel causal mask | Tree Attention mask | Parallel Tree Attention | Parallel causal mask |
| **Loss of Quality?** | **None** (Speculative Sampling) | **None** (when using rejection) | **None** | **None** |
| **Weight Overhead** | High ($10\% - 20\%$) | Near-Zero ($<1\%$) | Low ($1\% - 2\%$) | Near-Zero ($1\%$) |
| **Typical Speedup** | $1.2\text{x} - 1.5\text{x}$ | $1.3\text{x} - 1.6\text{x}$ | $1.5\text{x} - 1.8\text{x}$ | **$1.8\text{x} - 2.0\text{x}$** |
