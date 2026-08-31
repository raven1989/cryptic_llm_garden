---
tags:
  - algorithm
  - hashing
  - pretraining
  - data-processing
date: 2026-08-31
sources: ["[[Pre-training Large Language Models]]"]
---

# Locality-Sensitive Hashing (LSH, 局部敏感哈希)

LSH is a family of hashing techniques whose property is the **opposite of ordinary hashes**: while cryptographic hashes (MD5, SHA) maximize the avalanche effect to *distinguish* inputs, LSH makes similar inputs collide with high probability — collisions *signal* similarity. It is the enabling technology for approximate deduplication of web-scale pre-training corpora (see [[Pre-training Large Language Models]] Part 2, Step 3).

## 1. Formal Property

A hash family is locality-sensitive if for any two objects $x, y$:

$$
P[h(x) = h(y)] = \text{sim}(x, y)
$$

i.e., the collision probability equals (or is monotonic in) their similarity.

## 2. Why It Is Needed

Pairwise similarity comparison over a corpus of $n$ documents costs $O(n^2)$ — for 100M documents that is $5 \times 10^{15}$ comparisons, infeasible. LSH hashes each document into buckets and only compares documents sharing a bucket, reducing the cost to roughly $O(n)$.

## 3. MinHash: LSH for Jaccard Similarity

The instance cited by Chapter 15. Targets the Jaccard similarity of set-represented documents (e.g., sets of n-grams):

$$
\text{sim}(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

**Procedure:** pick a hash function $h$, hash every element of the set, and keep the **minimum** hash value as the set's signature: $\text{MinHash}(A) = \min_{a \in A} h(a)$.

### Why the minimum works

Hashing is deterministic — the same element always maps to the same value in both sets. Consider the global minimum over $A \cup B$: it belongs to some specific element $x^*$, and both sets find the *same* $x^*$ iff $x^*$ lies in the intersection:

$$
\text{MinHash}(A) = \text{MinHash}(B) \iff x^* \in A \cap B
$$

Since hashing acts as a **uniform random shuffle** of all elements (random per-element ranks, but *shared* across sets — the randomness is the sampling mechanism, not noise), the probability that the top-of-shuffle element comes from the intersection is exactly the intersection's share of the union:

$$
P[\text{MinHash}(A) = \text{MinHash}(B)] = \frac{|A \cap B|}{|A \cup B|} = \text{Jaccard}(A, B)
$$

### Multi-hash estimation and banding

- One hash gives a single Bernoulli draw; use $k$ independent hash functions and estimate similarity as the fraction of matching minima ($k = 128$–$256$ typical).
- **Banding:** split the $k$-dim signature into $b$ bands of $r$ values ($k = b \times r$); hash each band to buckets. Two documents become candidate duplicates if *any* band matches fully. The candidate probability $1 - (1 - s^r)^b$ is an S-curve in the true similarity $s$ — tune $b, r$ to place the threshold (e.g., 0.8).

## 4. Other LSH Families

| Similarity / distance | LSH family |
|---|---|
| Jaccard | MinHash |
| Cosine | SimHash / random hyperplane projection (sign of which side a random hyperplane the vector falls on) |
| Euclidean | Random projection + bucketing (p-stable distributions, L2-LSH) |
| Hamming | Random bit sampling |

## 5. Role in Pre-training Data Deduplication

Chapter 15's dedup matching methods split into **exact matching** (suffix arrays finding character-identical substrings) and **approximate matching** (MinHash/LSH finding similar documents). LSH cannot tell you *where* two documents overlap, but it surfaces "likely duplicate" candidate pairs in linear time — the standard combination is approximate LSH at the document level plus exact matching at the sentence level. The construction of C4, Dolma, and the Pile all rely on this.

## Related Pages

- [[Pre-training Large Language Models]]: LSH/MinHash appears in Part 2, Step 3 (deduplication).
- [[Perplexity]]: the complementary heuristic signal used in the quality-filtering step of the same pipeline.
