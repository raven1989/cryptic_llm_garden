---
tags: [llm, generation, decoding, algorithms, optimization]
date: 2026-06-02
sources: ["[[raw/Recommendation/深入理解 Beam Search.md]]"]
---

# Summary: 深入理解 Beam Search

**Source:** [[raw/Recommendation/深入理解 Beam Search.md]] — "深入理解 Beam Search：原理, 示例与代码实现"

## Overview
This article provides a comprehensive and practical explanation of [[Beam Search]] (集束搜索), a widely used heuristic decoding strategy in auto-regressive text generation tasks. It details how Beam Search serves as an extension of greedy decoding, how parameters like `num_beams` (beam width) and `early_stopping` control the search process, and provides both mathematical scoring formulations and concrete PyTorch/Hugging Face implementations.

## Key Concepts Extracted

### 1. Working Principle of Beam Search
*   **Definition:** Beam Search is a breadth-first search (BFS) algorithm that explores the output sequence space by maintaining a fixed number of candidate sequences (the beam width $k$).
*   **Contrast with Greedy:** Greedy search ($k=1$) selects the single highest-probability token at each step, making it myopic and prone to suboptimal local optima. Beam Search ($k > 1$) retains $k$ candidates, expanding search space and allowing a higher chance of finding globally optimal sequences.
*   **Generation Steps:**
    1.  **Initialize:** Start with the candidate set $B_0 = \{ \text{start} \}$.
    2.  **Expand:** For each candidate in $B_{t-1}$, generate all possible next-token extensions from vocabulary $V$ and compute cumulative probabilities.
    3.  **Top-K Selection:** Select the $k$ highest-scoring sequences as the new active candidate set $B_t$.
    4.  **Terminate:** Stop when all candidates end with the `<eos>` token, or the maximum sequence length $T$ is reached.
    5.  **Final Selection:** Choose the candidate with the highest overall score as the output.

### 2. Numerical Underflow & Log-Likelihood
*   **The Issue:** Autoregressive generation involves multiplying probabilities $P(y_t | y_{<t})$. For long sequences, multiplying many small probabilities ($\le 1.0$) causes floating-point **numerical underflow**.
*   **The Solution:** Apply natural logarithms to convert products of probabilities into sums of log-probabilities:
    $$S(Y) = \log P(Y) = \sum_{t=1}^{T} \log P(y_t | y_1, y_2, \dots, y_{t-1})$$
    Since probabilities are in $[0, 1]$, log-likelihood scores are negative or zero. The objective is to maximize the score (i.e., make it closest to 0 / least negative).

### 3. Step-by-Step Generation Example ($k=2$)
The source illustrates Beam Search with a concrete vocabulary $\{A, B, C, \text{<eos>}\}$:
*   **Step 1:** Initial probabilities yield $A$ ($0.4$) and $B$ ($0.3$). Candidates: $\{A: 0.4, B: 0.3\}$.
*   **Step 2 (Expansion):**
    *   Expand $A$: $AA$ ($0.12$), $AB$ ($0.04$), $AC$ ($0.16$), $A\text{<eos>}$ ($0.08$).
    *   Expand $B$: $BA$ ($0.15$), $BB$ ($0.03$), $BC$ ($0.09$), $B\text{<eos>}$ ($0.03$).
    *   **Top-2 Selection:** $AC$ ($0.16$) and $BA$ ($0.15$) become the new candidates.
*   **Step 3:**
    *   Expand $AC$: $ACA$ ($0.016$), $ACB$ ($0.064$), $ACC$ ($0.048$), $AC\text{<eos>}$ ($0.032$).
    *   Expand $BA$: $BAA$ ($0.06$), $BAB$ ($0.03$), $BAC$ ($0.015$), $BA\text{<eos>}$ ($0.045$).
    *   **Top-2 Selection:** $ACB$ ($0.064$) and $BAA$ ($0.06$) are chosen.
*   **Termination:** Eventually, completed sequences ending in `<eos>` are collected, scored, and compared.

### 4. Handling `<eos>` & Early Stopping
*   **Completed Sequences:** Once a candidate sequence appends `<eos>`, it is completed. It is moved out of the active search set to a completed list.
*   **Early Stopping (`early_stopping`):**
    *   **`True`:** Stops generation immediately when $k$ completed sequences are found. Highly efficient.
    *   **`False` / `heuristic`:** Continues search until it is mathematically impossible for any active unfinished sequence to surpass the score of the best completed sequence.
    *   **`never`:** Search proceeds until all active sequences produce `<eos>` or max length is reached.
