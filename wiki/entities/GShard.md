---
tags: [llm, architecture, operators, moe, conditional-computation]
aliases: [GShard, Top-2 Gating]
date: 2026-05-20
sources: ["[[raw/LLM/GShard - Scaling Giant Models with Conditional Computation and Automatic Sharding.pdf]]"]
related: ["[[Sparsely-Gated MoE Layer]]", "[[Transformers]]"]
---

# GShard: Scaling Giant Models

Introduced by Google in 2020, **GShard** was the first architecture to successfully scale a Transformer model to 600 Billion parameters using a [[Sparsely-Gated MoE Layer|Mixture-of-Experts]] structure. It proved that model quality could scale sub-linearly with compute costs by aggressively decoupling the parameter count from active FLOPs.

![Figure 3: GShard Architecture](../media/gshard_architecture.png)
*Figure 3: Scaling the Transformer Encoder with MoE. Every other Feed-Forward layer is replaced with a Top-2 MoE layer. When scaled across multiple devices, the dense layers are replicated everywhere, but the experts are sharded across devices.*

## 1. Wall Time vs. Compute Time

The core argument of GShard is its efficiency in large-scale distributed training. 
*   **Compute Time:** The theoretical total processing time of all chips combined. GShard's 600B model required **22 TPU v3 core-years** of compute.
*   **Wall Time:** The physical, real-world time the training job takes from start to finish. By distributing the model across 2,048 TPU cores, the *wall time* was only **4 days**.

In contrast, their best Dense model (only 2.3B parameters) required over 235 core-years of compute and achieved worse translation quality. MoE architectures inherently optimize for Wall Time by reducing the math required per token.

## 2. The Mathematics of the MoE Layer

GShard defines its MoE layer mathematically using three core equations for a given input token $x_s$:

### Equation 1: The Gating Function
$$G_{s,E} = \text{GATE}(x_s)$$
*   This applies the routing function (a Softmax) to the token $x_s$.
*   The output $G_{s,E}$ is an $E$-dimensional vector containing the probability weights assigning the token to each of the $E$ experts. 
*   Because GShard uses **Top-2 routing**, this vector is almost entirely zeros. Only the 2 chosen experts receive non-zero probabilities.

### Equation 2: The Expert Computation
$$\text{FFN}_e(x_s) = w_{o e} \cdot \text{ReLU}(w_{i e} \cdot x_s)$$
*   This defines a single Expert. It is simply a standard 2-layer Feed-Forward Network.
*   $w_{i e}$ is the input projection (expanding the dimension), and $w_{o e}$ is the output projection. 
*   Instead of one giant FFN for the whole model, there are $E$ independent copies of these weights.

### Equation 3: The Final Output
$$y_s = \sum_{e=1}^{E} G_{s,e} \cdot \text{FFN}_e(x_s)$$
*   The final output $y_s$ is the weighted blend of the experts.
*   Because $G_{s,e}$ is exactly $0$ for all but 2 experts, the network completely skips computing $\text{FFN}_e$ for the unselected experts, which is the source of MoE's massive computational savings.

## 3. Engineering the Router: Algorithm 1

Simply running a mathematical Softmax across millions of tokens across 2,048 GPUs simultaneously would cause massive network bottlenecks and memory crashes. GShard solved this with strict hardware-aware constraints in their routing algorithm.

### Local Group Dispatching & The Capacity Factor
Tokens are partitioned into $G$ independent groups of size $S$. Routing mathematics are calculated strictly within these isolated local groups in parallel.
*   $S = N / G$: The total tokens in the batch ($N$) divided by the number of groups ($G$).
*   **Expert Capacity ($C$):** $C = \frac{2N}{G \cdot E}$. This is the hard limit on how many tokens a single expert can take from a group.
    *   $N/G/E$ is the perfectly uniform baseline (the exact number of tokens an expert *should* receive if the load were perfectly balanced).
    *   The multiplier **2** acts as the **Capacity Factor**. It acts as a double buffer, allowing an expert to take up to 2x its normal load if the data naturally skews. A capacity factor of 1.0 forces perfect load balancing but hurts accuracy; 2.0 improves accuracy but wastes VRAM because hardware buffers must be statically allocated at the 2x maximum.

### Algorithm 1 Reference

> **Algorithm 1:** Group-level top-2 gating with auxiliary loss
> **Data:** $x_S$, a group of tokens of size $S$
> **Data:** $C$, Expert capacity allocated to this group
> **Result:** $G_{S,E}$, group combine weights
> **Result:** $\ell_{aux}$, group auxiliary loss
> 
> 1. $c_E \leftarrow 0$ &nbsp;&nbsp; *(gating decisions per expert)*
> 2. $g_{S,E} \leftarrow \text{softmax}(w_g \cdot x_S)$ &nbsp;&nbsp; *(gates per token per expert)*
> 3. $m_E \leftarrow \frac{1}{S} \sum_{s=1}^S g_{s,E}$ &nbsp;&nbsp; *(mean gates per expert)*
> 4. **for** $s \leftarrow 1$ **to** $S$ **do**
> 5. &nbsp;&nbsp;&nbsp;&nbsp; $g1, e1, g2, e2 = \text{top\_2}(g_{s,E})$ &nbsp;&nbsp; *(top-2 gates and expert indices)*
> 6. &nbsp;&nbsp;&nbsp;&nbsp; $g1 \leftarrow g1 / (g1 + g2)$ &nbsp;&nbsp; *(normalized g1)*
> 7. &nbsp;&nbsp;&nbsp;&nbsp; $c \leftarrow c_{e1}$ &nbsp;&nbsp; *(position in e1 expert buffer)*
> 8. &nbsp;&nbsp;&nbsp;&nbsp; **if** $c < C$ **then**
> 9. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $G_{s,e1} \leftarrow g1$ &nbsp;&nbsp; *(e1 expert combine weight for $x_s$)*
> 10. &nbsp;&nbsp;&nbsp;&nbsp; **end**
> 11. &nbsp;&nbsp;&nbsp;&nbsp; $c_{e1} \leftarrow c + 1$ &nbsp;&nbsp; *(incrementing e1 expert decisions count)*
> 12. **end**
> 13. $\ell_{aux} = \frac{1}{E} \sum_{e=1}^E \frac{c_e}{S} \cdot m_e$
> 14. **for** $s \leftarrow 1$ **to** $S$ **do**
> 15. &nbsp;&nbsp;&nbsp;&nbsp; $g1, e1, g2, e2 = \text{top\_2}(g_{s,E})$
> 16. &nbsp;&nbsp;&nbsp;&nbsp; $g2 \leftarrow g2 / (g1 + g2)$ &nbsp;&nbsp; *(normalized g2)*
> 17. &nbsp;&nbsp;&nbsp;&nbsp; $rnd \leftarrow \text{uniform}(0, 1)$ &nbsp;&nbsp; *(dispatch to 2nd-best with probability $\propto 2 \cdot g2$)*
> 18. &nbsp;&nbsp;&nbsp;&nbsp; $c \leftarrow c_{e2}$ &nbsp;&nbsp; *(position in e2 expert buffer)*
> 19. &nbsp;&nbsp;&nbsp;&nbsp; **if** $c < C$ **and** $2 \cdot g2 > rnd$ **then**
> 20. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $G_{s,e2} \leftarrow g2$ &nbsp;&nbsp; *(e2 expert combine weight for $x_s$)*
> 21. &nbsp;&nbsp;&nbsp;&nbsp; **end**
> 22. &nbsp;&nbsp;&nbsp;&nbsp; $c_{e2} \leftarrow c + 1$
> 23. **end**

### Algorithm 1: Line-by-Line Breakdown

#### Part 1: Initialization & Raw Probabilities (Lines 1-3)
*   **(1) $c_E \leftarrow 0$:** Initializes a counter $c$ for every single expert to `0`. This tracks raw token **demand** (not just successful dispatches).
*   **(2) $g_{S,E} \leftarrow \text{softmax}(w_g \cdot x_S)$:** Does the standard math: multiplies the input tokens ($x_S$) by the routing weights ($w_g$) and applies Softmax. This gives the raw probability distribution.
*   **(3) $m_E \leftarrow \frac{1}{S} \sum_{s=1}^{S} g_{s,E}$:** Calculates the *mean* (average) gate probability for each expert across all $S$ tokens. This smooth, differentiable value ($m_E$) is crucial for the loss function later.

#### Part 2: Routing the 1st Choice Expert (Lines 4-12)
*The algorithm uses two separate loops. This first loop ONLY handles everyone's #1 choice expert.*
*   **(4-5) The Loop:** Start looping through each individual token $s$. Look at the raw probabilities for token $s$. Find the highest probability ($g1$) and its expert ID ($e1$), and the second highest ($g2$, $e2$).
*   **(6) $g1 \leftarrow g1 / (g1 + g2)$:** **Normalization.** Because we are throwing away all the other experts (choices 3 through $E$), the probabilities no longer add up to 1.0. This line scales $g1$ so that $g1$ and $g2$ perfectly sum to 1.0.
*   **(7-10) The Hard Cutoff:** Check the current capacity counter ($c$) for our top-choice expert ($e1$). Is $c$ less than the maximum allowed capacity ($C$)?
    *   If yes, save the weight $g1$ into the final output matrix. The token is officially going to expert $e1$.
    *   If no, the token has "overflowed" and is dropped. The weight remains 0.
*   **(11-12) $c_{e1} \leftarrow c + 1$:** Increment the counter for expert $e1$. **Crucially, it increments the counter EVEN IF the token was dropped in step 8.** This ensures $c_e$ tracks the true demand for the expert, which is needed for the loss calculation.

#### Part 3: The Differentiable Auxiliary Loss Trick (Line 13)
*   **(13) $\ell_{aux} = \frac{1}{E} \sum_{e=1}^{E} \frac{c_e}{S} \cdot m_e$:** This calculates the penalty for unbalanced routing.
    *   **The Problem:** We want to minimize the variance of physical demand ($c_e$). But $c_e$ is derived using `top_2()` and `if` statements, making it discrete and non-differentiable (gradient descent fails).
    *   **The Solution:** Multiply the discrete physical demand fraction ($\frac{c_e}{S}$) by the smooth Softmax probability ($m_e$). Backpropagation gradients can flow through $m_e$. If physical demand ($c_e$) is too high, the loss spikes, forcing the network to lower the Softmax probabilities ($m_e$) in future steps, smoothly balancing the load.

#### Part 4: Routing the 2nd Choice Expert With Randomness (Lines 14-23)
*A second loop goes through all tokens again to handle their #2 choice expert.*
*   **(14-18):** Repeat the steps to grab and normalize the 2nd choice expert's weight ($g2$), and check its capacity counter.
*   **(19) $\text{if } c < C \land 2 \cdot g2 > rnd \text{ then}$:** A token is only sent to its 2nd choice if there is capacity left AND it passes a **Proportional Random Drop**.
    *   If $g2$ (the second-best weight) is very small (e.g., `0.05`), $2 \times 0.05 = 0.1$. There is only a 10% chance this passes the random number check (`rnd`). Thus, 90% of the time, weak secondary tokens are dropped, saving precious network bandwidth.
*   **(20-23):** Assign the weight if it passed, and increment the $e2$ counter.