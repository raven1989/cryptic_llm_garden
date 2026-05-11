---
tags: [llm, algorithms, engineering, attention]
date: 2026-05-11
aliases: [MHA, Multi-Head Attention Mechanism]
related: ["[[Self-Attention Mechanism]]", "[[Transformers]]", "[[KV Cache]]"]
sources: ["[[raw/LLM/大模型原理与架构/02_attention/2.3_multi_head.md]]"]
---

# Multi-Head Attention (MHA)

Multi-Head Attention (MHA) is a core structural component of modern [[Transformers]]. Instead of performing a single massive attention calculation over the embedding dimension ($d_{\text{model}}$), MHA projects the input into $h$ different, independent "representation subspaces" (heads), performs attention in parallel, and concatenates the results.

## Why Do We Need Multiple Heads?

Why split one large $d_{\text{model}}$ attention calculation into $h$ smaller heads? It solves critical mathematical and representational bottlenecks found in standard [[Self-Attention Mechanism|Self-Attention]].

### 1. Functional Specialization (Representation Subspaces)
Language is highly complex. A single word needs to connect to other words for many different reasons (syntax, semantics, proximity). If a model only has one attention head, it only has one set of $Q, K, V$ matrices. That single transformation must try to capture *every* possible relationship, which is a heavy representational burden.

By projecting the input into $h$ independent subspaces, MHA allows the model to "divide and conquer." Different heads naturally learn to specialize during training:
*   **Syntax:** One head might only look for the verb attached to a noun.
*   **Proximity:** Another head might always look at the previous 2 words to establish local context.
*   **Semantics:** A third head might look across the whole document to resolve coreferences (e.g., figuring out what the pronoun "it" refers to).

### 2. Preventing "Attention Thinning" (The Softmax Zero-Sum Problem)
The core operation of attention ends with a `Softmax` function, which ensures that all attention weights sum to exactly `1.0`.

Imagine the sentence: *"The **bank** of the river where I put my **money**."*
If the word "money" needs to look at the word "bank" (for semantic context) and the word "my" (for grammatical ownership) simultaneously, a single head is forced to split its attention (e.g., 0.5 to "bank", 0.5 to "my"). 

This is called **Attention Thinning (摊薄)**. The more things a single head needs to look at, the weaker its focus becomes on any individual target. With Multi-Head Attention, Head 1 can give a strong `0.99` weight to "bank", while Head 2 simultaneously gives a `0.99` weight to "my". Multiple heads allow the model to hold strong, confident focus on multiple targets at once.

### 3. Compute Efficiency (It's a "Free" Upgrade)
$h$ heads do not require $h$ times more computing power. Because we mathematically split the dimension of the heads ($d_k = d_{\text{model}} / h$), the compute balances out perfectly.
*   A single head doing math on a dimension of 512.
*   8 heads doing math on a dimension of 64.

The total number of parameters and the total number of floating-point operations (FLOPs) are exactly the same. MHA is simply a smarter way to allocate the model's capacity without increasing computational cost.

---

## Matrix Dimensions Step-by-Step

Understanding the exact matrix dimensions is crucial for estimating [[KV Cache]] memory and FLOPs. 

**Core Variables:**
*   $n$: Sequence length (number of tokens in context).
*   $d_{\text{model}}$: Total embedding dimension (e.g., 4096 in Llama 3 8B).
*   $h$: Number of attention heads (e.g., 32).
*   $d_k, d_v$: Dimension of each individual head for Q/K and V. Usually $d_k = d_v = d_{\text{model}} / h$ (e.g., $4096 / 32 = 128$).

### Step 1: The Input
The input is a sequence of tokens converted into embeddings.
*   **Matrix ($X$)**: $\mathbb{R}^{n \times d_{\text{model}}}$

### Step 2: Projections to Q, K, V (Per Head)
For a single head $i$:
*   **Weight Matrices**: $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
*   **Head Projections**:
    *   $Q_i = X \cdot W_i^Q \implies \mathbb{R}^{n \times d_k}$
    *   $K_i = X \cdot W_i^K \implies \mathbb{R}^{n \times d_k}$
    *   $V_i = X \cdot W_i^V \implies \mathbb{R}^{n \times d_v}$

### Step 3: Scaled Dot-Product Attention (Per Head)
Each head independently calculates attention scores.
1.  **Raw Scores**: $Q_i \cdot K_i^T \implies (n \times d_k) \cdot (d_k \times n) \implies \mathbb{R}^{n \times n}$
2.  **Scaling and Softmax**: $\text{Softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) \implies \mathbb{R}^{n \times n}$
3.  **Applying to Values**: $\text{Scores} \cdot V_i \implies (n \times n) \cdot (n \times d_v) \implies \mathbb{R}^{n \times d_v}$
    *   **Output of Head $i$ ($\text{head}_i$)**: $\mathbb{R}^{n \times d_v}$

### Step 4: Concatenation
The model concatenates the $h$ different outputs.
*   **Concatenation**: $[ \text{head}_1, \text{head}_2, \dots, \text{head}_h ]$
*   Since $d_v = d_{\text{model}} / h$, concatenating $h$ heads restores the original dimension: $\mathbb{R}^{n \times (h \cdot d_v)} \implies \mathbb{R}^{n \times d_{\text{model}}}$

### Step 5: Final Linear Projection
The concatenated output is passed through a final linear transformation ($W^O$) to allow the independent subspaces to mix their findings.
*   **Output Weight Matrix ($W^O$)**: $\mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$
*   **Final Result**: $\text{Concat} \cdot W^O \implies (n \times d_{\text{model}}) \cdot (d_{\text{model}} \times d_{\text{model}}) \implies \mathbb{R}^{n \times d_{\text{model}}}$

Notice that the input goes in as $n \times d_{\text{model}}$ and comes out as $n \times d_{\text{model}}$, making MHA a perfect modular block.

---

## Architectural Optimizations

While standard MHA is highly effective at capturing relationships, it creates massive computational and memory bottlenecks during autoregressive generation because of the [[KV Cache]]. Storing unique Key and Value vectors for *every* head quickly limits context length and inference speed.

To solve this, two major architectural optimizations are commonly used in modern models:
1.  **[[Multi-Query Attention]] (MQA):** Shares a single Key and Value head across *all* Query heads to drastically reduce memory usage, at the cost of some accuracy.
2.  **[[Grouped Query Attention]] (GQA):** The modern standard. It strikes a balance by grouping Query heads together, where each group shares a Key and Value head. It retains the speed benefits of MQA while preserving the accuracy of MHA.