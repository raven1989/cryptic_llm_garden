---
tags: [concept, algorithm, optimal-transport, doubly-stochastic, normalization]
aliases: [Sinkhorn-Knopp, Sinkhorn Algorithm, SK Algorithm, 辛克霍恩算法]
date: 2026-08-16
related: ["[[mHC]]", "[[LC-Rec]]", "[[RQ-VAE]]", "[[DeepSeek Load Balancing]]"]
---

# Sinkhorn-Knopp Algorithm

## Overview

The **Sinkhorn-Knopp algorithm** ( Sinkhorn 1964; Sinkhorn & Knopp 1967 ) is an iterative matrix-scaling procedure that converts any matrix with strictly positive entries into a **doubly stochastic matrix** — one where every row sums to 1 and every column sums to 1 — by alternately normalizing rows and columns. 

In modern deep learning it has become the standard differentiable tool for two distinct jobs:

1. **Projecting an unconstrained matrix onto the Birkhoff polytope** (the manifold of doubly stochastic matrices), e.g. [[mHC]] uses it to stabilize widened residual streams.
2. **Solving entropic-regularized Optimal Transport (OT)** problems, e.g. [[LC-Rec]] uses it to enforce a uniform-distribution constraint on codebook assignment and eliminate ID collisions.

## 1. Mathematical Foundations

### Doubly Stochastic Matrices & the Birkhoff Polytope

A matrix $P \in \mathbb{R}^{n \times n}$ is **doubly stochastic** if:

$$
P \mathbf{1}_n = \mathbf{1}_n, \qquad \mathbf{1}_n^{\top} P = \mathbf{1}_n^{\top}, \qquad P_{ij} \geq 0
$$

*   Every row is a probability distribution; every column is also a probability distribution.
*   Geometrically, the set of all such matrices forms the **Birkhoff polytope**. By the Birkhoff–von Neumann theorem, its vertices are exactly the permutation matrices, so every doubly stochastic matrix is a convex combination of permutations — a "soft permutation."
*   Key stability properties: its spectral norm is bounded by 1 ($\|P\|_2 \le 1$), it acts as a convex combination on inputs (preserving feature means), and the product of two doubly stochastic matrices is still doubly stochastic. This is why [[mHC]] constrains residual mixing matrices to this manifold.

### Sinkhorn's Theorem

> For any matrix $A$ with strictly positive entries, there exist unique (up to scalar) positive diagonal matrices $D_1 = \mathrm{diag}(u)$ and $D_2 = \mathrm{diag}(v)$ such that $D_1 A D_2$ is doubly stochastic.

The Sinkhorn-Knopp algorithm is simply the iterative procedure that finds $u$ and $v$.

## 2. The Core Algorithm

Given a raw, unconstrained matrix $\tilde{M}$ (e.g. the output of a neural network layer):

**Step 0 — Ensure positivity.**
$$
M^{(0)} = \exp(\tilde{M})
$$
Exponentiating makes every entry strictly positive (a requirement of Sinkhorn's theorem) and is fully differentiable.

**Step t — Alternate row and column normalization.**

Define the operators:
$$
\mathcal{T}_r(M)_{ij} = \frac{M_{ij}}{\sum_k M_{ik}} \qquad \mathcal{T}_c(M)_{ij} = \frac{M_{ij}}{\sum_k M_{kj}}
$$

Iterate:
$$
M^{(t)} = \mathcal{T}_r\left(\mathcal{T}_c(M^{(t-1)})\right)
$$

*   Row normalization makes all rows sum to 1 but breaks column sums.
*   Column normalization fixes columns but slightly perturbs rows.
*   Repeating this ping-pong process converges linearly to a unique doubly stochastic fixed point.

In practice only a small, fixed number of iterations is used (e.g. $t_{\max} = 3$–$20$); the result is "doubly stochastic enough" for gradient-based training.

### Numerically Stable Log-Domain Version

Repeated exponentiation and division can overflow/underflow. The standard implementation works in log-space using log-sum-exp (LSE):

$$
\begin{aligned}
\tilde{M} &\leftarrow \tilde{M} - \mathrm{LSE}_{\text{row}}(\tilde{M}) \\
\tilde{M} &\leftarrow \tilde{M} - \mathrm{LSE}_{\text{col}}(\tilde{M})
\end{aligned}
$$

and finally $\exp(\tilde{M})$ yields the doubly stochastic matrix. This is the version used in [[mHC]] and most production implementations.

## 3. Connection to Optimal Transport

The algorithm's most famous modern application is **entropy-regularized Optimal Transport** (Cuturi 2013, "Sinkhorn Distances").

**Problem:** Match a source distribution $a$ (e.g. $N$ items) to a target distribution $b$ (e.g. $K$ codes) given a cost matrix $C_{ij}$, while keeping the problem differentiable:

$$
\min_{P \in U(a,b)} \langle P, C \rangle - \varepsilon \, H(P)
$$

where $U(a,b) = \{P \ge 0 : P\mathbf{1} = a, P^{\top}\mathbf{1} = b\}$ and $H(P) = -\sum_{ij} P_{ij}(\log P_{ij} - 1)$ is the entropic regularizer.

**Solution:** The optimum has the form $P^* = \mathrm{diag}(u) \, e^{-C/\varepsilon} \, \mathrm{diag}(v)$, which is exactly what Sinkhorn-Knopp computes starting from the kernel $K = e^{-C/\varepsilon}$, with row/column targets $a$ and $b$ instead of uniform $\mathbf{1}$.

*   The temperature $\varepsilon$ controls the softness of the assignment: small $\varepsilon$ → near-permutation (hard matching); large $\varepsilon$ → blurred, near-uniform transport plan.
*   Because the iterations are just matrix multiplications and normalizations, gradients flow through them — making OT usable as a differentiable layer inside a neural network.

## 4. Applications in This Wiki

### [[mHC]] — Projection onto the Birkhoff Polytope
Manifold-Constrained Hyper-Connections need the residual mixing matrix $\mathcal{H}^{\mathrm{res}}_l$ to be doubly stochastic to preserve the identity-mapping property of [[Residual Connections]]. Sinkhorn-Knopp is applied directly: $\exp(\tilde{\mathcal{H}}^{\mathrm{res}}_l)$ followed by ~20 row/column normalizations. The bounded spectral norm ($\le 1$) and closure under multiplication guarantee signals neither explode nor vanish across arbitrary depth.

### [[LC-Rec]] — Collision-Free Codebook Assignment via OT
In generative recommendation, [[RQ-VAE]]-based semantic IDs suffer collisions when similar items greedily map to the same leaf code. LC-Rec formulates final-level code assignment as a global Optimal Transport problem over the whole batch with a **uniform distribution constraint** (each code gets exactly $|B|/K$ items). Sinkhorn-Knopp solves this differentiably during training, yielding 100% unique, collision-free codes while preserving semantic proximity.

#### Problem Construction (the OT formulation)

At the final quantization level $H$, for a batch of residual vectors $\mathbf{r}_H$, LC-Rec seeks the soft assignment matrix $Q = [q(c_H = k \mid \mathbf{r}_H)] \in [0,1]^{|B| \times K}$ that minimizes total semantic cost subject to marginal constraints:

$$
\min_{Q} \sum_{\mathbf{r}_H \in B} \sum_{k=1}^{K} q(c_H = k \mid \mathbf{r}_H)\, \|\mathbf{r}_H - \mathbf{v}_k^H\|_2^2
$$

$$
\text{s.t.} \quad \sum_{k=1}^{K} q(c_H = k \mid \mathbf{r}_H) = 1 \;\; (\text{row: every item fully assigned}), \qquad \sum_{\mathbf{r}_H \in B} q(c_H = k \mid \mathbf{r}_H) = \frac{|B|}{K} \;\; (\text{column: every codeword gets an equal share})
$$

Three elements of this construction:

*   **Decision variable $Q$:** a soft (fractional) assignment, kept in $[0,1]$ rather than $\{0,1\}$ so the problem stays differentiable. The final hard code is obtained by row-wise argmax.
*   **Cost matrix:** $C_{nk} = \|\mathbf{r}_H^n - \mathbf{v}_k^H\|_2^2$ is the OT *ground cost* — the semantic distance from item $n$'s residual to codeword $k$.
*   **Uniform column constraint:** the capacity $|B|/K$ is the mathematical form of "collision-free." When $|B| \le K$ (capacity $\le 1$), no column can absorb two full items, so the hard assignment is necessarily unique. When $|B| > K$ (capacity $> 1$), the constraint only enforces *even crowding*, not uniqueness — which is why collision-freeness is only achievable at the small scale of conflict groups.

#### Why row entries are never uniform: the cost term dominates

The row constraint $\sum_k q_k = 1$ alone permits a fully uniform row $(1/K, \dots, 1/K)$ — so what prevents it? The **cost term** $\langle Q, C \rangle$. Spreading weight onto distant codewords inflates the semantic loss, so the optimizer concentrates weight on the nearest codeword. The entropy term $-\varepsilon H(Q)$ pulls toward uniformity, but is suppressed by a small temperature $\varepsilon$. Mechanically, Sinkhorn starts from the kernel $K = e^{-C/\varepsilon}$, whose exponential **amplifies distance differences into weight differences** (a nearest-codeword advantage of a few distance units becomes orders-of-magnitude weight ratio) — and the subsequent row/column rescalings are multiplicative, adjusting marginals without flattening these relative ratios.

#### Two-stage usage: training vs. inference

| | Training | Inference (index construction) |
|---|---|---|
| Scope | Whole batch $B$, every step, last level | First greedy (Eqn. 1) for all items, then Sinkhorn **only on conflict groups** |
| Purpose | Shape a balanced codebook via the RQ loss | Produce a deterministic, globally consistent final index |
| Why | Differentiable OT integrates into training | Full-corpus OT is intractable and (capacity $\gg 1$) cannot guarantee uniqueness anyway |

The two-stage inference is justified by a precise equivalence:

> **When the greedy solution happens to satisfy the uniform column constraint, it is already the OT optimum** — each row independently attains its minimum cost, so no feasible solution can do better. Running Sinkhorn on such items would only blur the (already optimal) assignment via the entropy term without changing the argmax. Hence: greedy first, invoke Sinkhorn only where greedy fails (collisions).

Caveat: the equivalence holds at the **argmax (hard assignment)** level with small $\varepsilon$; at the soft-assignment level the entropy term always perturbs the solution away from one-hot. Also note "no collision" does not automatically mean "column constraint satisfied" — the constraint is an equality (each column *exactly* $|B|/K$), so a collision-free greedy solution with idle codewords still violates it at the soft level; the equivalence kicks in after argmax re-hardens the assignment.

The paper does **not** specify whether, during conflict-group redistribution, the codebook is restricted to codewords not already taken by non-conflicting items under the same prefix — this is an implementation detail left to the released code.

#### Worked example

Batch of $|B| = 4$ items sharing prefix `a0-b1-`, final-level codebook $K = 8$ (c0–c7), so column capacity $= |B|/K = 0.5$. Suppose the residuals of items 1–3 all sit close to codeword c0, while item 4 sits close to c1. Distance matrix (rows = items; only the relevant columns shown):

$$
C = \begin{pmatrix} 0.10 & 3.0 & 2.5 & 4.0 & \cdots \\ 0.20 & 2.8 & 0.90 & 3.5 & \cdots \\ 0.15 & 3.2 & 1.10 & 0.8 & \cdots \\ 3.00 & 0.3 & 4.5 & 5.0 & \cdots \end{pmatrix}
$$

*   **Greedy (Eqn. 1):** items 1, 2, 3 all pick c0 (distances 0.10 / 0.20 / 0.15), item 4 picks c1 → items 1–3 **collide** on leaf `a0-b1-c0`.
*   **OT with uniform constraint:** each column may absorb total weight $0.5$, so no column can take two full items. Item 1 has the strongest claim on c0 (0.10) and keeps it. Items 2 and 3 are displaced from c0; both look to their next-nearest codeword c2, but c2's capacity also fits only one — item 2's claim on c2 is stronger (0.90 < 1.10), so item 2 takes c2 and item 3 is pushed one step further to its own next-nearest, c3 (0.8). Sinkhorn yields a soft assignment like:

$$
Q \approx \begin{pmatrix} 0.48 & 0.01 & 0.01 & 0.00 & \cdots \\ 0.01 & 0.02 & 0.45 & 0.02 & \cdots \\ 0.01 & 0.02 & 0.04 & 0.43 & \cdots \\ 0.00 & 0.45 & 0.00 & 0.05 & \cdots \end{pmatrix}, \qquad \text{column sums} = (0.5,\ 0.5,\ 0.5,\ 0.5,\ \dots)
$$

*   **Argmax:** item 1→c0, item 2→c2, item 3→c3, item 4→c1 — four items fan out onto **four distinct codewords**, collision resolved. Crucially, each displaced item lands on its **nearest available** codeword (guided by the cost term), not an arbitrary free one — this is the advantage over post-hoc local-search patching.

Had the greedy assignment already been collision-free and balanced, Sinkhorn's argmax would reproduce it unchanged — the two-stage design skips exactly those redundant computations.

### Related: [[DeepSeek Load Balancing]]
The same "balanced assignment" philosophy appears in MoE routing: DeepSeek-V3's auxiliary-loss-free bias adjustment and Sinkhorn-based balanced routing both aim to distribute tokens evenly across experts without distorting the main loss.

### Other notable uses (broader literature)
*   **SwAV** (Caron et al., 2020): self-supervised learning by matching features to prototypes with Sinkhorn-balanced cluster assignments, avoiding collapse.
*   **Differentiable sorting/ranking** (e.g. NeuralSort): Sinkhorn produces soft permutation matrices.

## 5. Practical Notes

| Property | Value |
|---|---|
| Iterations typically used | 3 – 20 |
| Cost per iteration | $\mathcal{O}(n^2)$ (two normalizations) |
| Convergence | Linear; guaranteed for strictly positive matrices |
| Failure mode | Zero/negative entries — always start with $\exp(\cdot)$ or clip |
| Stability trick | Work in log-space with LSE subtraction |

## References
*   Sinkhorn, R. (1964). *A Relationship Between Arbitrary Positive Matrices and Doubly Stochastic Matrices.*
*   Sinkhorn, R. & Knopp, P. (1967). *Concerning Nonnegative Matrices and Doubly Stochastic Matrices.*
*   Cuturi, M. (2013). *Sinkhorn Distances: Lightspeed Computation of Optimal Transport.*
*   Peyré, G. & Cuturi, M. (2019). *Computational Optimal Transport.* (Chapter 4 gives the definitive treatment of Sinkhorn's algorithm and its convergence.)
