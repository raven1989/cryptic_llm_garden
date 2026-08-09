---
title: "EASE (Embarrassingly Shallow Autoencoders)"
tags: ["entity", "collaborative-filtering", "mathematics", "machine-learning"]
aliases: ["EASE Algorithm", "Embarrassingly Shallow Autoencoders for Sparse Data"]
date: 2026-08-05
sources: ["[[wiki/research/CRAG Summary.md]]"]
---

# EASE (Embarrassingly Shallow Autoencoders)

Introduced by Harald Steck at WWW 2019, **EASE** (Embarrassingly Shallow Autoencoders) is a highly elegant, linear collaborative filtering model. It scales exceptionally well to sparse, high-dimensional user-item interaction data because it has an exact, closed-form mathematical solution that avoids gradient descent entirely.

---

## 1. Core Concept and Motivation

A classical neural autoencoder trains an encoder to compress input vector $\mathbf{x}$ into a latent bottleneck, and a decoder to reconstruct $\mathbf{x}$ from the bottleneck. EASE simplifies this architecture by stripping out the hidden layers, non-linear activation functions, and deep weights. 

It is a single-layer, linear autoencoder that directly maps the sparse user-item interaction matrix $\mathbf{X}$ back to itself via a weight matrix $\mathbf{B}$:
$$\mathbf{S} = \mathbf{X}\mathbf{B}$$
Where $\mathbf{X} \in \{0, 1\}^{U \times I}$ is the binary interaction matrix, and $\mathbf{B} \in \mathbb{R}^{I \times I}$ represents the item-to-item similarity weights.

### The Self-Reconstruction Trap
Without constraints, minimizing the reconstruction error $\|\mathbf{X} - \mathbf{X}\mathbf{B}\|_F^2$ yields the trivial solution:
$$\mathbf{B} = \mathbf{I} \quad \text{(the Identity Matrix)}$$
If $\mathbf{B} = \mathbf{I}$, the model has perfect reconstruction but learns absolutely nothing about collaborative relationships between different items. To prevent this, EASE enforces a strict **zero-diagonal constraint**:
$$\text{diag}(\mathbf{B}) = 0 \quad (\mathbf{B}_{i,i} = 0, \, \forall i)$$
This forces the model to predict whether a user will interact with item $i$ based **entirely** on their interactions with *other* items ($j \neq i$).

---

## 2. Mathematical Optimization and Closed-Form Solution

### 2.1. Objective Function
The objective function of EASE combines the squared Frobenius norm of the reconstruction error with an $L_2$ regularization penalty on the weight matrix $\mathbf{B}$:

$$\min_{\mathbf{B}} \|\mathbf{X} - \mathbf{X}\mathbf{B}\|_F^2 + \lambda \|\mathbf{B}\|_F^2 \quad \text{subject to} \quad \text{diag}(\mathbf{B}) = 0$$

Where:
*   $\mathbf{X} \in \mathbb{R}^{U \times I}$ is the user-item interaction matrix.
*   $\mathbf{B} \in \mathbb{R}^{I \times I}$ is the item similarity matrix.
*   $\lambda > 0$ is the regularization strength.
*   $\| \cdot \|_F$ is the Frobenius norm.

### 2.2. Lagrangian Derivation
To incorporate the constraint $\text{diag}(\mathbf{B}) = 0$, we formulate the Lagrangian function using a vector of multipliers $\boldsymbol{\gamma} \in \mathbb{R}^I$:

$$\mathcal{L}(\mathbf{B}, \boldsymbol{\gamma}) = \frac{1}{2} \|\mathbf{X} - \mathbf{X}\mathbf{B}\|_F^2 + \frac{\lambda}{2} \|\mathbf{B}\|_F^2 + \boldsymbol{\gamma}^T \text{diag}(\mathbf{B})$$

We expand the squared Frobenius norm of the reconstruction error:
$$\|\mathbf{X} - \mathbf{X}\mathbf{B}\|_F^2 = \text{Tr}\left( (\mathbf{X} - \mathbf{X}\mathbf{B})^T (\mathbf{X} - \mathbf{X}\mathbf{B}) \right)$$
$$\mathcal{L}(\mathbf{B}, \boldsymbol{\gamma}) = \frac{1}{2} \text{Tr}\left( \mathbf{X}^T\mathbf{X} - 2\mathbf{X}^T\mathbf{X}\mathbf{B} + \mathbf{B}^T\mathbf{X}^T\mathbf{X}\mathbf{B} \right) + \frac{\lambda}{2} \text{Tr}(\mathbf{B}^T\mathbf{B}) + \boldsymbol{\gamma}^T \text{diag}(\mathbf{B})$$

Taking the partial derivative of the Lagrangian with respect to $\mathbf{B}$ and setting it to zero:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{B}} = -\mathbf{X}^T\mathbf{X} + \mathbf{X}^T\mathbf{X}\mathbf{B} + \lambda \mathbf{B} + \text{diag}(\boldsymbol{\gamma}) = 0$$

Let $\mathbf{G} = \mathbf{X}^T\mathbf{X}$ represent the symmetric co-occurrence Gram matrix, and $\hat{\mathbf{G}} = \mathbf{G} + \lambda \mathbf{I}$ be the regularized Gram matrix. Substituting these into the derivative:
$$\hat{\mathbf{G}}\mathbf{B} - \mathbf{G} + \text{diag}(\boldsymbol{\gamma}) = 0$$
Since $\hat{\mathbf{G}} - \lambda \mathbf{I} = \mathbf{G}$, we can rewrite the equation as:
$$\hat{\mathbf{G}}\mathbf{B} - \hat{\mathbf{G}} + \lambda\mathbf{I} + \text{diag}(\boldsymbol{\gamma}) = 0$$
$$\hat{\mathbf{G}}(\mathbf{B} - \mathbf{I}) = -(\lambda\mathbf{I} + \text{diag}(\boldsymbol{\gamma}))$$

Let $\mathbf{P} = \hat{\mathbf{G}}^{-1}$ be the inverse of the regularized Gram matrix. Multiplying both sides by $\mathbf{P}$:
$$\mathbf{B} - \mathbf{I} = -\mathbf{P} \left( \lambda\mathbf{I} + \text{diag}(\boldsymbol{\gamma}) \right)$$
Let $\tilde{\boldsymbol{\gamma}} = \lambda\mathbf{I} + \text{diag}(\boldsymbol{\gamma})$ represent the diagonal scaling term. Then:
$$\mathbf{B} = \mathbf{I} - \mathbf{P} \text{diag}(\tilde{\boldsymbol{\gamma}})$$

### 2.3. Solving for the Constraint
We enforce the constraint that the diagonal of $\mathbf{B}$ must be zero ($\text{diag}(\mathbf{B}) = 0$). For each diagonal entry $j$:
$$\mathbf{B}_{j,j} = 1 - \mathbf{P}_{j,j} \tilde{\gamma}_j = 0$$
This allows us to solve directly for the Lagrange terms $\tilde{\gamma}_j$:
$$\tilde{\gamma}_j = \frac{1}{\mathbf{P}_{j,j}}$$
Substituting $\tilde{\gamma}_j$ back into the formula for $\mathbf{B}$, we get the final closed-form solution:
$$\mathbf{B} = \mathbf{I} - \mathbf{P} \cdot \text{diag}(\mathbf{P})^{-1}$$

In scalar form, for any cell $i, j$:
$$\mathbf{B}_{i, j} = \begin{cases} 
      -\frac{\mathbf{P}_{i, j}}{\mathbf{P}_{j, j}} & \text{if } i \neq j \\
      0 & \text{if } i = j
   \end{cases}$$

---

## 3. Adaptation in CRAG (Asymmetric Catalog Mapping)

In the CRAG architecture, the set of items mentioned in dialogues ($\mathcal{I}$) is often a subset or a different catalog than the recommendable catalog ($\mathcal{Q}$). 

To support this asymmetry, the authors adapt the EASE formulation to learn an asymmetric weight matrix $\mathbf{W} \in \mathbb{R}^{|\mathcal{I}| \times |\mathcal{Q}|}$ that maps mentioned items to catalog items, by optimizing:

$$\min_{\mathbf{W}} \|\mathbf{R}_{\mathcal{Q}} - \mathbf{R}\mathbf{W}\|_F^2 + \lambda \|\mathbf{W}\|_F^2 \quad \text{subject to} \quad \mathbf{W}_{i, j} = 0, \, \forall i = \text{ReID}(j)$$

Where:
*   $\mathbf{R}$ is the binary interaction matrix from the historical dialogue training logs (treating each dialogue as a user session).
*   $\mathbf{R}_{\mathcal{Q}}$ extracts columns corresponding to the catalog set $\mathcal{Q}$.
*   The constraint prevents a mentioned item $i$ from recommending its own remapped catalog ID index $j = \text{ReID}^{-1}(i)$, blocking self-reconstruction.
*   Once solved, retrieval scores for a multi-hot active session profile $\mathbf{r}_k$ are computed as:
    $$\text{Scores} = \mathbf{r}_k^T \mathbf{W}$$

---

## Related Concepts
*   [[wiki/research/CRAG Summary.md|CRAG Research Paper Summary]]
*   [[wiki/entities/CRAG.md|CRAG (Collaborative Retrieval Augmented Generation)]]
