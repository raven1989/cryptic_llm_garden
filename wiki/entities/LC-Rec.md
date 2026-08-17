---
title: "LC-Rec: Adapting Large Language Models by Integrating Collaborative Semantics"
tags: ["entity", "architecture", "LLMRec", "Optimal-Transport", "Sinkhorn-Knopp", "RQ-VAE"]
aliases: ["Language-Collaborative Recommender Architecture"]
date: 2026-08-09
sources: ["[[wiki/research/LC-Rec Summary.md]]"]
---

# LC-Rec: Adapting Large Language Models by Integrating Collaborative Semantics

**LC-Rec** is a generative, index-based sequential recommendation framework. It converts item metadata into discrete coordinate indices using an RQ-VAE optimized globally via an optimal transport formulation, and fine-tunes a decoder-only LLM (LLaMA-7B) using a multi-task instruction alignment curriculum.

---

## 1. Item Indexing via Optimal Transport

To represent items uniquely and semantically, LC-Rec maps textual features onto discrete multi-level coordinate indices:
$$\mathrm{SID}(i) = \langle a\_c_1 \rangle \langle b\_c_2 \rangle \langle c\_c_3 \rangle \langle d\_c_4 \rangle \quad \text{where} \quad c_l \in \{1, \dots, 256\}$$

### 1.1. Contextual Feature Extraction
Item text metadata (title, description) is serialized and processed through LLaMA's frozen 32 layers. The final-layer hidden states are aggregated via **mean pooling** to generate a continuous contextualized semantic vector $\mathbf{e} \in \mathbb{R}^{4096}$, capturing rich contextual associations and pre-trained world knowledge.

---

## 2. RQ-VAE with Uniform Semantic Mapping (Algorithm 1)

The continuous vector $\mathbf{e}$ is recursively quantized over $H=4$ levels using an RQ-VAE with standard MLP layers and ReLU activations. To prevent **index collisions** (where multiple similar items map to identical final leaf codes), LC-Rec avoids local greedy search at the last level and instead frames codebook assignment as a **global combinatorial optimization problem (Optimal Transport)**.

### 2.1. Algorithm 1: Line-by-Line Technical Breakdown

The global residual quantization and uniform mapping procedure is formally executed over a batch $B$ of items according to the following steps:

1.  **Initialize Residuals:**
    $$\mathbf{r}_1^n = \mathbf{z}_n, \quad \forall \mathbf{z}_n \in B$$
    *   *Explanation:* Set the initial residual vector at the first quantization level ($l=1$) as the raw, continuous latent representation $\mathbf{z}_n$ (projected from $\mathbf{e}$) for every item $n$ in the batch.
2.  **Level Quantization Loop:**
    $$\text{for } i = 1 \text{ to } H \text{ do}$$
    *   *Explanation:* Loop sequentially through each coarse-to-fine quantization level $i$, starting from $1$ up to the final level $H$ (typically $H=4$).
3.  **Boundary Evaluation:**
    $$\text{if } i < H \text{ then}$$
    *   *Explanation:* Check if we are currently at an intermediate quantization level (coarse layers) before reaching the last level $H$.
4.  **Intermediate Codebook Assignment (Symmetric argmin):**
    $$\text{Solve } \{c_i^n\}_{n=1}^{|B|} \text{ according to Eqn. (1):} \quad c_i^n = \text{argmin}_k \|\mathbf{r}_i^n - \mathbf{v}_k^i\|_2^2$$
    *   *Explanation:* For intermediate levels, assign the nearest codeword $c_i^n$ using standard greedy nearest-neighbor clustering in codebook $\mathcal{C}_i$.
5.  **Final Level Collision Boundary:**
    $$\text{else}$$
    *   *Explanation:* Enter this block only when we reach the final quantization level $i = H$, where index collisions and leaf-node conflicts typically occur.
6.  **Global Optimal Transport Alignment (Sinkhorn-Knopp):**
    $$\text{Solve } \{c_H^n\}_{n=1}^{|B|} \text{ according to Eqn. (6) via Sinkhorn-Knopp algorithm}$$
    *   *Explanation:* Instead of greedy matching, solve the globally balanced matching matrix $q(c_H = k \mid r_H)$ over the entire batch $B$ under the **Uniform Distribution Constraint** (Equation 6) using the **[[Sinkhorn-Knopp Algorithm]]**. This distributes similar items evenly and guarantees unique final coordinate indexes ($c_H^n$) for every item in the batch.
7.  **Terminate Level Decision:**
    $$\text{end if}$$
    *   *Explanation:* Terminate the level-wise decision block.
8.  **Compute Next-Level Residuals:**
    $$\text{Obtain } \{\mathbf{r}_{i+1}^n\}_{n=1}^{|B|} \text{ according to Eqn. (2):} \quad \mathbf{r}_{i+1}^n = \mathbf{r}_i^n - \mathbf{v}_{c_i}^i$$
    *   *Explanation:* Calculate the residual vector $\mathbf{r}_{i+1}^n$ for the next level by subtracting the selected codeword vector $\mathbf{v}_{c_i}^i$ from the current residual.
9.  **End Level Loop:**
    $$\text{end for}$$
    *   *Explanation:* Terminate the quantization level loop. Every item $n$ now possesses a unique multi-level code $[c_1^n, \dots, c_H^n]$.
10. **Reconstruction Loop:**
    $$\text{for all } \mathbf{z}_n \in B \text{ do}$$
    *   *Explanation:* Loop through each item in the batch to calculate its final reconstructed vector.
11. **Codeword Summation:**
    $$\text{Calculate quantified representations by } \hat{\mathbf{z}}_n = \sum_{i=1}^{H} \mathbf{v}_{c_i}^i$$
    *   *Explanation:* Construct the reconstructed dense vector $\hat{\mathbf{z}}_n$ by summing the codebook vectors corresponding to the selected discrete indices across all $H$ levels.
12. **End Reconstruction:**
    $$\text{end for}$$
    *   *Explanation:* Terminate the reconstruction loop.
13. **Return Results:**
    $$\text{return } \{[c_1^n, \dots, c_H^n]\}_{n=1}^{|B|} \text{ and } \{\hat{\mathbf{z}}_n\}_{n=1}^{|B|}$$
    *   *Explanation:* Return the final unique, discrete coordinate index sequences (the item indices) and the continuous reconstructed vectors.

---

### 2.2. The Optimal Transport Constraint (Equation 6)
To satisfy the uniform distribution of codes during the final level $H$, LC-Rec minimizes the distance cost subject to a global constraint:

$$\min_{X} \sum_{r_H \in B} \sum_{k=1}^{K} q(c_H = k \mid r_H) \|r_H - \mathbf{v}_k^H\|_2^2$$

$$\text{subject to:} \quad \sum_{k=1}^{K} q(c_H = k \mid r_H) = 1 \quad \text{(Row constraint: Every item is fully assigned)}$$
$$\sum_{r_H \in B} q(c_H = k \mid r_H) = \frac{|B|}{K} \quad \text{(Column constraint: Every codeword gets an equal, uniform share)}$$

This doubly constrained allocation matrix is solved globally and differentiably during training using the **[[Sinkhorn-Knopp Algorithm]]**, which iteratively normalizes the rows and columns of the exponentiated distance matrix. This guarantees 100% unique, collision-free codes while preserving physical proximity on the semantic manifold.

---

## 3. The Loss Functions of RQ-VAE (Equations 3, 4, and 5)

The RQ-VAE parameters are optimized end-to-end using a joint loss function composed of two primary objectives:

$$\mathcal{L}_{\text{RQ-VAE}} = \mathcal{L}_{\text{RECON}} + \mathcal{L}_{\text{RQ}}$$

### 3.1. Decoder Reconstruction Loss (Equation 3):
$$\mathcal{L}_{\text{RECON}} = \|\mathbf{e} - \hat{\mathbf{e}}\|_2^2$$
*   **What it does:** Measures the Mean Squared Error (MSE) between the original LLaMA text embedding $\mathbf{e}$ and the decoder's reconstructed embedding $\hat{\mathbf{e}}$ (generated from $\hat{\mathbf{z}}$ using an MLP decoder with ReLU).
*   **The Goal:** Force the encoder and decoder to retain maximum semantic information during vector compression.

### 3.2. Residual Quantization Loss (Equation 4):
$$\mathcal{L}_{\text{RQ}} = \sum_{i=1}^{H} \left( \|\text{sg}[\mathbf{r}_i] - \mathbf{v}_{c_i}^i\|_2^2 + \beta \|\mathbf{r}_i - \text{sg}[\mathbf{v}_{c_i}^i]\|_2^2 \right)$$
*   **Stop-Gradient ($\text{sg}[\cdot]$):** Prevents gradients from backpropagating through the non-differentiable $\text{argmin}$ selection during the backward pass (Straight-Through Estimator).
*   **First Term ($\|\text{sg}[\mathbf{r}_i] - \mathbf{v}_{c_i}^i\|_2^2$):** Updates the selected codebook vector $\mathbf{v}_{c_i}^i$ to move closer to the encoder's residual vector $\mathbf{r}_i$.
*   **Second Term ($\beta \|\mathbf{r}_i - \text{sg}[\mathbf{v}_{c_i}^i]\|_2^2$):** The **commitment loss** (with scale coefficient $\beta = 0.25$). It penalizes the distance between the encoder outputs and their selected codewords, forcing the encoder to "commit" stably to a codeword and preventing its representations from shifting wildly between different clusters.

### 3.3. Overall Joint Loss (Equation 5):
$$\mathcal{L}_{\text{RQ-VAE}} = \mathcal{L}_{\text{RECON}} + \mathcal{L}_{\text{RQ}}$$
By minimizing this joint loss, the RQ-VAE encoder MLPs, decoder MLPs, and $H$-level codebooks are optimized in tandem.

---

## 4. Deep Insight: How Collaborative Semantics are Formed

A critical theoretical boundary in LC-Rec's architecture is that **the RQ-VAE indexing is purely semantic and contains zero collaborative or behavioral knowledge.** 

Because the RQ-VAE is optimized strictly to encode and reconstruct individual item text metadata (titles, brands, descriptions) via Equation 5, its 4-digit coordinate index codes represent **nothing but hierarchical textual similarity** on LLaMA's frozen semantic manifold.

**The collaborative semantics are integrated entirely within LLaMA's attention weights during the SFT alignment phase.**
By training LLaMA on sequential interaction logs using these purely semantic index coordinate codes, the following occurs:
1.  **Implicit Co-occurrence Graph Learning:** When optimizing the Symmetric Sequential Item Prediction task ($\mathcal{L}$, Equation 7) over index sequences:
    $$\mathcal{S}^u = [\langle a\_5 \rangle \langle b\_4 \rangle \langle c\_2 \rangle \langle d\_1 \rangle \quad \mathbf{\to} \quad \langle a\_5 \rangle \langle b\_3 \rangle \langle c\_5 \rangle \langle d\_7 \rangle]$$
    LLaMA's 32 layers of multi-head self-attention learn the transition and co-occurrence probabilities between these coordinate nodes.
2.  **Mapping Behavior to Semantics:** Through backpropagation, the attention heads learn that although $\langle a\_5 \rangle$ represents role-playing games semantically, users who play game $A$ often purchase game $B$ next behaviorally.
3.  **Perfect Generalization:** This design cleanly separates the **indexing tree** (which remains purely semantic and generalizable to cold items via the text encoder $f_T^{enc}$) from the **sequential interest transitions** (which are deeply learned by LLaMA's intermediate attention matrices), achieving complete, robust semantic integration.

---

## 5. Multi-Task Alignment Tuning Curriculum

To align the OOV coordinate indices with language semantics, the LLaMA-7B backbone is optimized over three categories of instruction tasks:

### 5.1. Task A: Symmetric Sequential Item Prediction
*   **Purpose:** Models implicit user-item interaction transitions, teaching the LLM how to identify sequential co-occurrence behaviors directly in the coordinate-index token space.
*   **Example:**
    *   **Instruction:** `"Here are the user's historical interactions: <a_124><b_192><c_41><d_17>, ..., <a_82><b_59><c_191><d_66>, try to recommend another item to the user. Note that the historical interactions are arranged in chronological order."`
    *   **Response:** `"<a_112><b_32><c_5><d_175>"`

### 5.2. Task B: Explicit Index-Language Alignment (Cross-Modal Mapping)
*   **Purpose:** Performs bidirectional cross-modal translation, grounding coordinate indices in language so the OOV indices are semantically registered inside the LLM's language manifold.
*   **Task B1: Index-to-Text Mapping**
    *   *Instruction:* `"Please tell me what item <a_66><b_197><c_236><d_223> is called, along with a brief description."`
    *   *Response:* `"Item Title: Pokémon Moon - Nintendo 3DS. Item Description: Pokémon Moon will launch in the US on November 18, 2016... become a Pokémon Champion!"`
*   **Task B2: Text-to-Index Mapping**
    *   *Instruction:* `"Translate the item Pokémon Moon - Nintendo 3DS into its discrete indices."`
    *   *Response:* `"<a_66><b_197><c_236><d_223>"`

### 5.3. Task C: Implicit Recommendation-Oriented Alignment
Enhances contextual reasoning over user intents, attributes, and preferences by mixing modalities across inputs and targets:

*   **Asymmetric Item Prediction (C1):** 
    *   *Asymmetric 1 (Index History $\to$ Text Title Target):*
        *   *Instruction:* `"Based on the user's historical interactions: <a_38><b_94><c_198><d_59>, ..., <a_190><b_60><c_94><d_86>, try to predict the title of the item that the user may need next."`
        *   *Response:* `"NBA 2K16 - PlayStation 4"`
    *   *Asymmetric 2 (Index History $\to$ Text Description Target):*
        *   *Instruction:* `"Here is the item interaction history of the user: <a_38><b_94><c_198><d_59>, ..., <a_190><b_60><c_94><d_86>, please tell me what features he expects from his next item."`
        *   *Response:* `"The NBA 2K franchise is... cover of your choice into the front of box."`
    *   *Asymmetric 3 (Text Title History $\to$ Index Target):*
        *   *Instruction:* `"Given the title sequence of user historical interactive items: 'The Biggest Loser Ultimate Workout - Xbox 360', ..., 'Lego: Marvel Super Heroes, XBOX 360', can you recommend a suitable next item for the user?"`
        *   *Response:* `"<a_27><b_58><c_138><d_201>"` *(Lego Star Wars - Xbox 360)*

*   **User Intention Prediction (C2 - Search/Retrieval):** 
    *   *Instruction:* `"As a recommender system, you are assisting a user who has recently interacted with the following items: <a_64><b_159><c_1><d_89>, ..., <a_119><b_98><c_162><d_155>. The user expresses a desire to obtain another item with the following characteristics: 'The console offers 500GB of storage, ... 4K HDR gaming'. Please recommend an item that meets these criteria."`
    *   *Response:* `"<a_227><b_206><c_156><d_156>"` *(PlayStation 4 500GB Console)*

*   **Personalized Preference Inference (C3):** 
    *   *Instruction:* `"Utilizing the ordered list of the user's historical interaction items as a reference, please make an informed estimation of the user's preferences. The historical interactions are as follows: <a_227><b_186><c_52><d_166>, ..., <a_120><b_208><c_146><d_153>."`
    *   *Response:* `"The user has recently been playing a game that requires more data storage and/or higher data processing capabilities."`

---

## 6. Training & Inference

### 6.1. Loss Function (Equation 7)
All tasks are formatted as conditional language generation sequences and optimized jointly using standard autoregressive next-token cross-entropy loss over target response $Y$:

$$\mathcal{L} = -\sum_{\langle \mathcal{I}, Y \rangle \in \mathcal{B}} \sum_{j=1}^{|Y|} \log P(Y_j \mid \mathcal{I}, Y_{<j})$$

### 6.2. Full-Ranking Beam Search Inference
During sequential recommendation, the model decodes the 4-level target coordinate tokens sequentially. **Beam Search (Size 20)** explores candidate path branches over the entire catalog, enabling **generative full-ranking** without relying on limited, pre-selected candidate sets.

---

## Related Concepts
*   [[wiki/research/LC-Rec Summary.md|LC-Rec Research Paper Summary]]
*   [[wiki/research/Conversational Recommender Systems.md|Conversational Recommender Systems Synthesis Page]]
