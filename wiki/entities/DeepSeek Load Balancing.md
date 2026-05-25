---
title: "DeepSeek Load Balancing"
tags:
  - architecture
  - moe
  - LLM
  - research
sources: ["[[raw/LLM/DeepSeek-V2 - A Strong, Economical, and Efficient Mixture-of-Experts Language Model.pdf]]", "[[raw/LLM/DeepSeek-V3 Technical Report.pdf]]"]
aliases:
  - DeepSeek Load Balancing
  - Auxiliary-Loss-Free Load Balancing
  - Device-Level Balance Loss
  - Communication Balance Loss
date: 2026-05-21
---
# DeepSeek Load Balancing

**Source:** [[raw/LLM/DeepSeek-V2 - A Strong, Economical, and Efficient Mixture-of-Experts Language Model.pdf]] & [[raw/LLM/DeepSeek-V3 Technical Report.pdf]]

## The Big Picture: Why MoE Requires Load Balancing

In a traditional dense Transformer, every token passes through the exact same parameters, meaning computational load is naturally identical across all inputs. In a [[Sparsely-Gated MoE Layer|Mixture-of-Experts (MoE)]] architecture, however, computational routing is dynamic. 

This dynamic sparsity creates two critical challenges that make load balancing mandatory:

### 1. The Hardware Bottleneck (Expert Parallelism)
Massive MoE models cannot fit onto a single GPU's VRAM. Therefore, their experts are partitioned and distributed across a physical cluster of GPUs using **Expert Parallelism (EP)**. 
*   During a forward pass, tokens must be dispatched to their assigned experts on other GPUs via high-speed network connections using **All-to-All communication collectives**.
*   If the routing is unbalanced, some GPUs will receive zero tokens and sit idle (wasting compute), while other GPUs will be overloaded with tokens. 
*   Because modern training runs are synchronous, **the entire cluster can only progress as fast as the slowest (most overloaded) GPU**. An unbalanced load drastically degrades training and inference throughput.

### 2. The Statistical Bottleneck (Routing Collapse)
Sparsely-gated routing networks are prone to a self-reinforcing feedback loop known as **routing collapse**:
*   Early in training, if a specific expert happens to receive slightly better weight initialization, the gating router will assign it slightly higher probability scores.
*   Because this favored expert is selected more frequently, its parameters are updated (trained) much faster than those of other experts, making it even more capable and even more likely to be selected by the router in subsequent steps.
*   This feedback loop rapidly cascades until the router collapses into repeatedly selecting the same few experts for every single token, rendering the remaining experts useless and effectively reducing a massive sparse model to a much smaller dense model.

---

## Part 1: DeepSeek-V2 — Multi-Faceted Auxiliary Losses

DeepSeek-V2 utilizes **160 fine-grained routed experts** distributed across $D$ physical devices (GPUs). Under expert parallelism, standard expert-level balancing is insufficient to prevent hardware bottlenecks. If all highly active experts happen to reside on a single GPU, or if network communication is asymmetric, training throughput plummets. To resolve this, DeepSeek-V2 introduced **three distinct auxiliary losses** to balance training on different physical and topological levels.

### 1. Expert-Level Balance Loss ($L_{ExpBal}$)
This loss mitigates the risk of routing collapse, encouraging tokens to be uniformly distributed across all available routed experts.

$$L_{ExpBal} = \alpha_1 \sum_{i=1}^{N_r} f_i P_i \quad \quad \text{(Equation 23)}$$

#### Mathematical Breakdown of Indicators:
*   **$N_r$**: The total number of routed experts in the MoE layer ($N_r = 160$ in DeepSeek-V2).
*   **$\alpha_1$**: The expert-level balance factor. This is a manually tuned hyperparameter, set to $0.003$ in pre-training.
*   **$f_i$ (Equation 24)**: The actual fraction of tokens routed to expert $i$ over a sequence of length $T$:
    $$f_i = \frac{N_r}{K_r T} \sum_{t=1}^T \mathbb{1}(\text{Token } t \text{ selects Expert } i)$$
    *   **$K_r$**: The number of active routed experts per token ($K_r = 6$ in DeepSeek-V2).
    *   **$T$**: The total number of tokens in the sequence.
    *   **$\mathbb{1}(\cdot)$**: The indicator function. It outputs $1$ if the statement is true (i.e., Token $t$ selects Expert $i$ as one of its Top-$K_r$ choices), and $0$ otherwise.
    *   *Normalization Intuition:* Since there are $T$ tokens, and each is dispatched to $K_r$ experts, the total number of expert dispatches is $K_r T$. If the load is perfectly uniform, each of the $N_r$ experts receives exactly a $\frac{1}{N_r}$ fraction of the total dispatches. Multiplying the sum by the coefficient $\frac{N_r}{K_r T}$ ensures that $f_i = \frac{1}{N_r}$ under a perfectly balanced uniform distribution.
*   **$P_i$ (Equation 25)**: The average routing probability allocated to expert $i$ across all tokens in the sequence:
    $$P_i = \frac{1}{T} \sum_{t=1}^T s_{i,t}$$
    Where $s_{i,t}$ is the softmax affinity score of token $t$ for expert $i$.

---

### 2. Device-Level Balance Loss ($L_{DevBal}$)
To prevent physical GPUs from becoming bottlenecks, DeepSeek-V2 partitions all $N_r$ routed experts into $D$ groups $\{E_1, E_2, \dots, E_D\}$, deploying each group on a single physical device.

$$L_{DevBal} = \alpha_2 \sum_{i=1}^D f'_i P'_i \quad \quad \text{(Equation 26)}$$

#### Mathematical Breakdown of Indicators:
*   **$D$**: The number of physical devices across which the experts are distributed ($D = 8$ in DeepSeek-V2's training setup).
*   **$\alpha_2$**: The device-level balance factor hyperparameter (set to $0.05$).
*   **$f'_i$ (Equation 27)**: The average fraction of tokens routed to physical device $i$:
    $$f'_i = \frac{1}{|E_i|} \sum_{j \in E_i} f_j$$
    Where $|E_i|$ is the number of experts residing on device $i$, and $f_j$ is the expert-level token fraction from Eq. 24.
*   **$P'_i$ (Equation 28)**: The sum of routing probabilities for all experts residing on device $i$:
    $$P'_i = \sum_{j \in E_i} P_j$$

---

### 3. Communication Balance Loss ($L_{CommBal}$)
To control network overhead, DeepSeek-V2 uses a **device-limited routing** mechanism: a token is restricted to choosing routed experts distributed across at most $M$ physical devices ($M=3$ in V2). While this caps outgoing communication bandwidth, a single physical device could still *receive* a disproportionate number of tokens. The communication balance loss ensures that the *incoming* communication across all devices is uniform.

$$L_{CommBal} = \alpha_3 \sum_{i=1}^D f''_i P''_i \quad \quad \text{(Equation 29)}$$

#### Mathematical Breakdown of Indicators:
*   **$\alpha_3$**: The communication balance factor hyperparameter (set to $0.02$).
*   **$f''_i$ (Equation 30)**: The fraction of token transmissions sent to device $i$:
    $$f''_i = \sum_{t=1}^T \frac{\mathbb{1}(\text{Token } t \text{ is sent to Device } i)}{\frac{MT}{D}} = \frac{D}{MT} \sum_{t=1}^T \mathbb{1}(\text{Token } t \text{ is sent to Device } i)$$
    *   **$M$**: The device-limit hyperparameter ($M=3$).
    *   *Mathematical Intuition:* Since each of the $T$ tokens is sent to at most $M$ devices, the total number of device-to-device hidden state transmissions across the sequence is $MT$. Assuming a perfectly balanced network, these transmissions should be distributed uniformly across all $D$ devices. Therefore, the target capacity budget for each individual device is $\frac{MT}{D}$ incoming tokens. 
    *   Representing the coefficient as the reciprocal of the target capacity budget—$\frac{1}{(MT/D)}$—shows that $f''_i$ directly calculates the ratio of the *actual* number of tokens sent to Device $i$ relative to its *target uniform budget*.
*   **$P''_i$ (Equation 31)**: The sum of routing probabilities for the experts hosted on device $i$ (identical to $P'_i$ from Eq. 28):
    $$P''_i = \sum_{j \in E_i} P_j$$

---

### Hardware Fallback: Token-Dropping Strategy
Because soft auxiliary losses cannot guarantee a perfect load balance at every step, DeepSeek-V2 implements a **device-level token-dropping strategy** during training. 

The system computes an average computational capacity budget for each physical device equivalent to a capacity factor of $1.0$. If a device receives more tokens than this budget, the extra tokens with the lowest routing affinity scores are dropped. Their representations bypass the expert FFN entirely and flow directly to the next layer via the transformer's residual connection.

---
---

## Part 2: DeepSeek-V3 — Auxiliary-Loss-Free Load Balancing

The DeepSeek-V3 authors identified a fundamental limitation of traditional MoE training: **forcing the model to balance via auxiliary loss penalties inherently damages down-stream performance**. 

If the weight of the auxiliary loss is too large, the model is penalized during training for routing decisions. This forces the model to route tokens to sub-optimal, underloaded experts purely to satisfy the loss function, contaminating the gradients and degrading final accuracy. 

To resolve this, DeepSeek-V3 dropped complex loss penalties and pioneered an **Auxiliary-Loss-Free Load Balancing** strategy using a dynamic routing bias.

---

### The Base Gating Logic (Equations 12–15)
To understand the auxiliary-loss-free mechanism, we must first look at the standard gating mathematics of the [[DeepSeek Shared Experts|DeepSeekMoE]] block.

**1. The Forward Pass (Equation 12)**
The final hidden state output $\mathbf{h}'_t$ for token $t$ is calculated as:
$$ \mathbf{h}'_t = \mathbf{u}_t + \sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t} \text{FFN}_i^{(r)}(\mathbf{u}_t) $$
*   **$\mathbf{u}_t \in \mathbb{R}^d$**: The input hidden state vector for the $t$-th token entering the current layer ($d$ is the total hidden dimension, e.g., $5120$).
*   **$N_s$**: The number of **Shared Experts** (always active for every token, bypassing the router).
*   **$N_r$**: The total number of **Routed Experts** in the layer ($N_r = 256$ in DeepSeek-V3).
*   **$g_{i,t}$**: The normalized gating weight assigned to Routed Expert $i$.

**2. Token-to-Expert Affinity (Equation 15)**
Unlike standard MoEs that use a Softmax across all experts, DeepSeek calculates an independent affinity score $s_{i,t}$ for each expert using a **Sigmoid** function on the dot product of the token's representation and the expert's centroid:
$$ s_{i,t} = \text{Sigmoid}(\mathbf{u}_t^T \mathbf{e}_i) $$
*   **$\mathbf{e}_i \in \mathbb{R}^d$**: The learned centroid vector (or routing embedding) for the $i$-th routed expert, representing its semantic specialization.

**3. Base Top-K Selection (Equation 14)**
The raw, unnormalized routing score $g'_{i,t}$ is kept only if the expert's affinity is in the Top-$K_r$ for that token:
$$ 
g'_{i,t} = 
\begin{cases} 
s_{i,t}, & \text{if } s_{i,t} \in \text{TopK}(\{ s_{j,t} \mid 1 \leqslant j \leqslant N_r \}, K_r) \\
0, & \text{otherwise}
\end{cases}
$$

**4. Gating Normalization (Equation 13)**
Because independent Sigmoid outputs do not naturally sum to 1.0, the active Top-K gating scores are normalized to generate a valid probability scaling distribution:
$$ g_{i,t} = \frac{g'_{i,t}}{\sum_{j=1}^{N_r} g'_{j,t}} $$

---

### The Gating Bias Injection (Equation 16)
To balance the expert load without using an auxiliary loss, DeepSeek-V3 introduces a dynamic bias term $b_i$ for each expert. **Equation 16 replaces Equation 14:**

$$ 
g'_{i,t} = 
\begin{cases} 
s_{i,t}, & \text{if } s_{i,t} + b_i \in \text{TopK}(\{ s_{j,t} + b_j \mid 1 \leqslant j \leqslant N_r \}, K_r) \\
0, & \text{otherwise}
\end{cases}
$$

#### The Core Breakthrough:
The bias $b_i$ is **only** added to the affinity score ($s_{i,t} + b_i$) to determine **who wins the Top-K competition**. 
However, once the winning experts are selected, the raw gating value $g'_{i,t}$ is populated using the **pure, unbiased Sigmoid output $s_{i,t}$**. 

This separates the **Routing Decision** (which is balanced via the biases) from the **Gating Weight** (which is calculated from pure semantic affinity). Because the forward and backward passes use the uncorrupted $s_{i,t}$ scores, the model's gradients are completely untainted by the balancing mechanism, preserving optimal performance.

---

### The Dynamic Bias Update Rule
During training, the system monitors the expert load across the entire batch at each training step. At the end of every step, the bias terms are updated:
*   If Expert $i$ is **overloaded** (received more tokens than its fair capacity budget):
    $$b_i \leftarrow b_i - \gamma$$
*   If Expert $i$ is **underloaded**:
    $$b_i \leftarrow b_i + \gamma$$

#### Characteristics of $\gamma$ (gamma):
*   **It is a Hyperparameter, not a Learned Weight:** $\gamma$ is the **bias update speed**. It never receives gradients and is not updated by backpropagation. It acts exactly like a manual "learning rate" for the routing biases.
*   **The Schedule:** In DeepSeek-V3, $\gamma$ is set to $0.001$ for the first 14.3 Trillion tokens to actively achieve uniform load balancing. For the final 500 Billion tokens of pre-training, $\gamma$ is decayed to $0.0$, freezing the biases and letting the routing boundaries stabilize before training finishes.

---

### Complementary Sequence-Wise Auxiliary Loss (Equations 17–20)
While the bias trick successfully handles global, batch-wide balancing, DeepSeek-V3 still requires a mechanism to prevent extreme, highly localized load imbalances within a *single sequence* (such as a single document that repeatedly targets the exact same expert, causing a hardware spike). 

To prevent this, V3 retains a **complementary sequence-wise balance loss** ($L_{Bal}$), but assigns it a microscopic weight ($\alpha = 0.0001$) so it does not interfere with overall gradient quality.

$$L_{Bal} = \alpha \sum_{i=1}^{N_r} f_i P_i \quad \quad \text{(Equation 17)}$$

#### Mathematical Breakdown of Indicators:
*   **$\alpha$**: The sequence-wise balance factor hyperparameter (set to an extremely small $0.0001$).
*   **$K_r$**: The number of **activated routed experts** selected for each token ($K_r = 8$ in DeepSeek-V3).
*   **$N_r$**: The total number of **routed experts** available in the layer ($N_r = 256$ in DeepSeek-V3).
*   **$T$**: The **sequence length** (the total number of tokens in the current sequence/document). This loss is computed per-sequence rather than batch-wise to prevent localized hotspot anomalies.
*   **$f_i$ (Equation 18)**: The sequence-wise fraction of tokens dispatched to expert $i$:
    $$f_i = \frac{N_r}{K_r T} \sum_{t=1}^T \mathbb{1}\left( s_{i,t} \in \text{TopK}(\{ s_{j,t} \mid 1 \leqslant j \leqslant N_r \}, K_r) \right)$$
*   **$s'_{i,t}$ (Equation 19)**: The normalized sequence-wise affinity score:
    $$s'_{i,t} = \frac{s_{i,t}}{\sum_{j=1}^{N_r} s_{j,t}}$$
    *   *Why this is normalized:* Unlike the gating normalization in Eq. 13 which only sums over the *Top-K* selected experts, $s'_{i,t}$ is normalized across **all** $N_r$ routed experts. This converts the independent Sigmoid affinity scores $s_{i,t}$ into a valid sequence-wide probability distribution that is fully differentiable across all experts.
*   **$P_i$ (Equation 20)**: The average normalized sequence-wise probability assigned to expert $i$:
    $$P_i = \frac{1}{T} \sum_{t=1}^T s'_{i,t}$$