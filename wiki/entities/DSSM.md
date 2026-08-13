---
tags: ["entity", "architecture", "web-search", "semantic-matching", "representation-learning", "recommendation"]
aliases: ["DSSM", "Deep Structured Semantic Model", "Deep Structured Semantic Models"]
date: 2026-08-13
sources: ["[[wiki/research/DSSM Summary.md]]"]
---

# DSSM

**DSSM (Deep Structured Semantic Model)** is a web-search semantic-matching architecture published in 2013 (CIKM) by Po-Sen Huang (UIUC) and Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck (Microsoft Research). It uses a deep neural network to project queries and documents into a common low-dimensional semantic space, where relevance is the cosine similarity between their vectors — enabling semantic matching even with zero shared terms. It is discriminatively trained on clickthrough data to directly optimize for document ranking.

![DSSM Architecture](../media/DSSM_illustration.png)

## Core Architectural Pillars

### 1. Input Term Vector
The input is a high-dimensional **bag-of-words term vector** — raw counts of terms in a query or document (no normalization). Its size equals the vocabulary used to index the document collection, which is very large in real web search.

### 2. Word Hashing Layer (Key Innovation)
The first hidden layer (~30K units) is a **fixed, non-learned linear transformation** that reduces dimensionality via **letter n-grams**:

- Add word boundary markers: `good` → `#good#`
- Break into letter trigrams: `#good#` → `#go, goo, ood, od#`
- Represent the word as a sparse **bag-of-trigrams** vector.

**Vector composition.** Each dimension indexes one distinct letter trigram; the value is that trigram's count. With a trigram vocabulary `{0:#go, 1:goo, 2:ood, 3:od#, 4:#ca, 5:cat, 6:at#, ...}`:

```
       #go  goo  ood  od#  #ca  cat  at#
good    1    1    1    1    0    0    0
cat     0    0    0    0    1    1    1
```

A full query/document vector is the **sum of its words' trigram vectors** (a bag-of-trigrams over the text); a repeated word adds to the counts.

**Benefits:** 16-fold dimensionality reduction (500K vocab → 30,621 dims) with 0.0044% collision rate; robust to **out-of-vocabulary** words and **morphological variants** (e.g., `good`/`goods` share most trigrams); enables DNN training at web scale.

> [!note] Still Bag-of-Words
> Each vector dimension indexes one distinct letter trigram (e.g., `#goo`); the value is that trigram's count. This carries **no sequential/order signal** — it is a character-level BOW. The semantic power comes from the learned deep projection + click supervision, not sequence modeling.

### 3. Deep Non-Linear Projection
The word-hashed features pass through multiple fully-connected hidden layers (two layers of 300 units each) with **tanh** activation, producing a **128-dim concept vector**:

$$l_i = f(W_i l_{i-1} + b_i), \qquad f(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}$$

### 4. Cosine Relevance Scoring
Relevance between query $Q$ and document $D$ is the cosine similarity of their concept vectors:

$$R(Q, D) = \frac{y_Q^T y_D}{\|y_Q\| \|y_D\|}$$

Documents are ranked by descending semantic relevance score.

---

## Training (Discriminative, on Clickthrough Data)

- **Signal:** clicked documents are positives; for each (query, clicked-doc) pair, add **4 randomly sampled unclicked docs** as negatives.
- **Objective:** maximize the conditional likelihood of the clicked document via a softmax over the candidate set:

$$P(D|Q) = \frac{\exp(\gamma R(Q, D))}{\sum_{D' \in \mathbb{D}} \exp(\gamma R(Q, D'))}$$

- **Loss:** $L(\Lambda) = -\log \prod_{(Q, D^+)} P(D^+|Q)$, minimized with mini-batch SGD (batch 1024, ~20 epochs).
- **Setup:** ~100M query-title pairs from popular URLs (rich clicks) for training; evaluated on tail/new URLs with no clicks. Only document **title fields** are used.

---

## Key Properties

| Property | Detail |
| :--- | :--- |
| **Input** | Raw term-count vector (bag-of-words) |
| **Word-hashing layer** | ~30K units, fixed (non-learned) linear projection |
| **Hidden layers** | 2 × 300 units, tanh |
| **Output (semantic) dim** | 128 |
| **Vocabulary** | 500K words → 30,621 letter trigrams (16× reduction, 0.0044% collisions) |
| **Relevance metric** | Cosine similarity of concept vectors |
| **Training signal** | Clickthrough (clicked = positive, 4 unclicked = negatives) |
| **Optimizer** | Mini-batch SGD (batch 1024, ~20 epochs) |

---

## Experimental Results

Web document ranking (16,510 queries, NDCG):

| Model | NDCG@1 | NDCG@3 | NDCG@10 |
| :--- | :--- | :--- | :--- |
| TF-IDF | 0.319 | 0.382 | 0.462 |
| BM25 | 0.308 | 0.373 | 0.455 |
| BLTM-PR | 0.337 | 0.403 | 0.480 |
| DPM | 0.329 | 0.401 | 0.479 |
| DAE (unsupervised) | 0.310 | 0.377 | 0.459 |
| **L-WH DNN (DSSM)** | **0.362** | **0.425** | **0.498** |

**Key ablations:**
- Supervised DNN beats unsupervised DAE (same architecture/vocab) by **3.2 NDCG@1** — click supervision is the biggest driver.
- 500K-vocab word-hashed model beats the 40K-vocab DNN — word hashing unlocks large vocabularies.
- 3 nonlinear layers beat 1 layer by **0.4–0.5 NDCG** — deeper is better.

---

## Related Wiki Pages
* [[DSSM Summary]]: Complete, section-by-section research summary of the paper.
* [[DCN]]: Google/Stanford's Deep & Cross Network for CTR prediction — another foundational deep architecture for recommendation/search.
* [[RankMixer]]: ByteDance's GPU-friendly ranking architecture addressing the memory-bound bottleneck of traditional DLRMs.
