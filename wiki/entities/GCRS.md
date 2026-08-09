---
title: "GCRS (Generative Conversational Recommender System)"
tags: ["entity", "architecture", "CRS", "Generative-Recommendation"]
aliases: ["Generative Conversational Recommender System Architecture"]
date: 2026-08-06
sources: ["[[wiki/research/GCRS Summary.md]]"]
---

# GCRS (Generative Conversational Recommender System)

**GCRS** is a fully generative, end-to-end conversational recommender system. It unifies recommendation and response generation into a single next-token prediction task by representing items as discrete semantic IDs and factorizing the generation sequence into structured, interdependent decisions.

---

## 1. Semantic ID Construction

The semantic ID construction pipeline translates continuous item metadata into discrete coordinate tokens that are compatible with autoregressive language generation.

```text
Item Metadata 
  │  (Title, Year, Genres, Plot, Keywords)
  ▼
Serialized Text String (x_i)
  │
  ▼  (Pretrained Text Encoder: Sentence-T5 / BGE)
Continuous Dense Embedding [dim: 768 / 1024]
  │
  ▼  (4-layer RQ-VAE Quantization)
Raw Code Sequence [Code 17, Code 63, Code 0, Code 25]
  │
  ▼  (Recursive Backtracking Collision Resolution)
Unique Code Sequence [Code 17, Code 63, Code 0, Code 25]
  │
  ▼  (Mapped to LLM Special Tokens)
Semantic ID Token Sequence: <a_17><b_63><c_0><d_25>
```

### 1.1. Metadata Encoding
For each item $i$, its attributes are serialized into a text string $x_i$:
$$x_{i}=\texttt{"title: }t_{i}\texttt{ | year: }y_{i}\texttt{ | genres: }g_{i}\texttt{ | keywords: }k_{i}\texttt{ | plot: }p_{i}\texttt{"}$$

A fixed pretrained text encoder (Sentence-T5 for ReDial, BGE for INSPIRED) maps this string to a dense continuous embedding vector.

### 1.2. Residual Quantization (RQ-VAE)
The continuous embedding is compressed into $L$ discrete codes using a Residual Quantized Variational Autoencoder. For a 4-layer RQ-VAE ($L=4$) and a codebook size of $K=64$ per layer, the raw item is mapped to a 4-digit sequence:
$$\mathrm{SID}(i) = \langle c_1, c_2, c_3, c_4 \rangle \quad \text{where} \quad c_l \in \{0, 1, \dots, 63\}$$

These are mapped to LLM special tokens:
$$\langle a\_c_1 \rangle \langle b\_c_2 \rangle \langle c\_c_3 \rangle \langle d\_c_4 \rangle$$

### 1.3. Collision Resolution Algorithm (Appendix A)
To resolve collisions where multiple items map to identical IDs, GCRS employs a backtracking greedy matching algorithm:
1.  Compute a distance tensor $\mathbf{D} \in \mathbb{R}^{N \times L \times K}$ containing Euclidean distances between the residual vectors of the $N$ colliding items and all codewords at each level:
    $$d_{i,k}^{(l)} = \left\|\mathbf{r}_{i}^{(l)} - \mathbf{c}_{k}^{(l)}\right\|_{2}^{2}$$
2.  Rank the colliding items in descending order of their confidence (smallest distance at the last quantization level $L$).
3.  Greedily allocate the closest available codeword at level $L$. If a collision occurs, assign the next-closest codeword.
4.  If codewords at level $L$ are exhausted, backtrack to level $L-1$, reallocate that level's code based on distance ranking, and reassign level $L$ codes. This is repeated recursively until all items have unique IDs.

---

## 2. Structured Generation Paradigm

Rather than entangling item mentions within standard sentence text, GCRS factorizes the conditional generation probability:
$$P(u_t \mid C) = P(m \mid C) \cdot P(i \mid m, C) \cdot P(r \mid i, m, C)$$

Where:
*   $C$: Dialog history context.
*   $m$: Response intent mode.
*   $i$: Target item semantic ID.
*   $r$: Natural language response.

### 2.1. Token Structure
*   **Item Boundaries:** Item IDs are wrapped in special boundary tokens: `<BOI> SID(i) <EOI>`.
*   **Mode Control Tokens:** Represent high-level intent:
    *   `<MODE=CHAT>`: Standard conversational responses (no recommendations).
    *   `<MODE=REC>`: Recommendation responses.
*   **Response Boundary:** The natural language text block is prepended with the `<RESP>` token.

### 2.2. Training Template Formats
*   **Non-recommendation Turn:**
    $$\texttt{Assistant: <MODE=CHAT><RESP> [Natural Language Response]}$$
*   **Recommendation Turn:**
    $$\texttt{Assistant: <MODE=REC><BOI> <a\_c1><b\_c2><c\_c3><d\_c4> <EOI><RESP> [Recommender Response Text]}$$

---

## 3. Fine-Tuning and Optimization

### 3.1. Loss Function
GCRS is fine-tuned using the standard next-token prediction objective:
$$\mathcal{L}_{\text{NTP}} = -\sum_{j=1}^{|Y|} \log P_{\theta}\left(y_j \mid C, y_{<j}\right)$$

The loss is computed over all generated tokens after the `Assistant:` prefix, representing response intent ($m$), target item ID ($i$), and textual tokens ($r$) simultaneously.

### 3.2. Parameter-Efficient Fine-Tuning (QLoRA)
*   **Quantization (NF4):** The base model (Qwen2.5-7B) is loaded in 4-bit NormalFloat (NF4) with double quantization. This reduces the base model parameters VRAM footprint to ~5GB, enabling training on a single 48GB GPU (NVIDIA RTX 6000 Ada).
*   **LoRA adapters:** Applied to all linear layers (Rank $R=16$, Alpha $\alpha=32$, Dropout $\text{dp}=0.05$).
*   **Token Embeddings:** The native LLM vocabulary embeddings are frozen. The input and output embeddings of the newly added special tokens are set to **trainable**:
    $$\text{Trainable Vocabulary} = \{\langle\text{BOI}\rangle, \langle\text{EOI}\rangle, \langle\text{RESP}\rangle, \langle\text{MODE=REC}\rangle, \langle\text{MODE=CHAT}\rangle\} \cup \bigcup_{l \in \{a, b, c, d\}} \{\langle l\_k \rangle\}_{k=0}^{63}$$

This aligns the semantic coordinate spaces directly with the frozen word representations of the LLM.

---

## 4. Controlled Inference

*   **Forced Evaluation:** During Recall@k evaluation, GCRS overrides the intent selection and prepends `<MODE=REC>` to all ground-truth recommendation turns. This forces the model to generate a recommendation candidate instead of choosing to only chat.
*   **Constrained Beam Search Decoding:** During semantic ID generation, beam search (width 50) restricts the candidate generation vocabulary strictly to the valid paths defined in the item catalog, preventing hallucinations of invalid movie coordinates.

---

## Related Concepts
*   [[wiki/research/GCRS Summary.md|GCRS Research Paper Summary]]
*   [[wiki/entities/RQ-VAE.md|RQ-VAE (Residual Quantized Variational Autoencoder)]]
