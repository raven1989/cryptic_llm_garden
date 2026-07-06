---
tags:
  - generative-models
  - research
  - tokenization
date: 2026-07-01
sources: ["[[raw/recommendation/一文详解codebook技术史.md]]"]
---

# Codebook Technology History Summary

This page serves as the master compilation and summary of the evolutionary history of codebook and quantization technologies used in generative modeling and sequence compression. It charts the paradigm shift from continuous representation learning to discrete tokenization, tracing a direct lineage from Variational Autoencoders (VAEs) to Vector Quantized VAEs (VQ-VAEs), VQ-VAE-2, Residual Quantized VAEs (RQ-VAEs), and finally the projection-free Finite Scalar Quantization (FSQ).

---

## 1. The Lineage at a Glance

The evolutionary timeline of discrete tokenization is driven by three main engineering objectives: **eliminating representation collapse**, **scaling codebook capacity**, and **minimizing training/inference complexity**.

```mermaid
graph TD
    VAE["**1. VAE (Continuous)**<br>• Regularized latent space z<br>• Scaled via Gaussian KL<br>• *Problem: Posterior Collapse*"]
    
    VQ_VAE["**2. VQ-VAE (Discrete)**<br>• Discrete codebook index K<br>• Constant KL log(K)<br>• Updates via Straight-Through Estimator<br>• *Problem: Space-vs-Capacity Wall*"]
    
    VQ_VAE2["**3. VQ-VAE-2 (Hierarchical)**<br>• Multi-scale spatial grids<br>• Top (Global) & Bottom (Local) layers<br>• Gated prior conditioning<br>• *Problem: Sequentially Long Codes*"]
    
    RQ_VAE["**4. RQ-VAE (Residual)**<br>• Recursive residual quantization<br>• Multi-depth tuples (k1...kD)<br>• Capacity scaled to K^D via shared codebook<br>• *Problem: Lookup-based Complexity*"]
    
    FSQ["**5. FSQ (Finite Scalar Quantization)**<br>• Bounded decimal rounding<br>• Symmetric odd levels L_i<br>• Implicit codebook grid<br>• *Lookup-free & collapse-free*"]

    VAE ── "Discrete mapping & STE" ──> VQ_VAE
    VQ_VAE ── "Spatially split layers" ──> VQ_VAE2
    VQ_VAE ── "Recursive depth cascades" ──> RQ_VAE
    VQ_VAE ── "Coordinate grid rounding" ──> FSQ
```

---

## 2. Technical Comparison of Paradigms

| Feature | [[Variational Autoencoder]] | [[VQ-VAE]] | [[VQ-VAE-2]] | [[RQ-VAE]] | [[FSQ]] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Latent Space** | Continuous | Discrete | Discrete (Hierarchical) | Discrete (Stacked) | Discrete (Coordinate Grid) |
| **Quantization Step** | None (Sampling $z \sim \mathcal{N}$) | Nearest Neighbor Lookup | Nearest Neighbor Lookup | Recursive Residual Lookup | Bounded Rounding (`round(f(z))`) |
| **KL Divergence** | Dynamic: Minimizes distance to Gaussian | Constant: Fixed at $\log K$ | Constant: Multiple grids | Constant: Multiple depths | Constant: Bounded coordinates |
| **Auxiliary Losses** | Gaussian KL Loss | Codebook & Commitment Loss | Multi-scale Codebook & Commitment Loss | Cumulative Multi-depth Commitment Loss | None (Zero auxiliary losses) |
| **Gradient Flow** | Reparameterization Trick | Straight-Through Estimator (STE) | Hierarchical STE | Multi-depth STE | Rounding STE |
| **Codebook Update** | Gradient Descent | Gradient Descent / EMA | Hierarchical EMA | Multi-scale Residual EMA | None (Grid is static/implicit) |
| **Collapse Risks** | Posterior Collapse | Codebook / Index Collapse | Codebook / Index Collapse | Prevented via Shared Residual EMA | 100% Collapse-free by design |

---

## 3. Deep Architectural Walkthroughs

### A. [[Variational Autoencoder]] (Continuous Base)
*   **Core Math:** Models data $x$ by maximizing the Expected Evidence Lower Bound (ELBO):
    $$\text{ELBO}(x) = \mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] - KL\big(q_\phi(z|x) \parallel p(z)\big)$$
*   **The Bottleneck:** Highly expressive decoders (e.g., PixelCNN/Transformers) ignore $z$ and use their own autoregressive capacity, driving the posterior $q_\phi(z|x) \to p(z) = \mathcal{N}(0, \mathbf{I})$, resulting in **Posterior Collapse** (KL vanishing).

### B. [[VQ-VAE]] (The Discrete Leap)
*   **Core Math:** Replaces continuous latent space with a discrete codebook $C = \{e_1, \dots, e_K\}$:
    $$z_q = e_{k^*} \quad \text{where} \quad k^* = \arg\min_j \|z_e - e_j\|_2^2$$
*   **Preventing Collapse:** By defining a uniform prior ($p(z) = 1/K$) and deterministic posterior ($q(z|x) = 1$), the KL term is mathematically forced to a constant $\log K$, making its gradient zero ($\nabla_\phi KL = 0$). This removes the "KL vanishing" shortcut.
*   **Gradients:** Bypasses non-differentiable argmin using the Straight-Through Estimator (STE):
    $$z_q = z_e + \text{sg}[z_q - z_e]$$
*   **Loss Formulation:** Adds auxiliary alignment terms:
    $$\mathcal{L} = \|X - \hat{X}\|_2^2 + \|\text{sg}[z_e] - z_q\|_2^2 + \beta \|z_e - \text{sg}[z_q]\|_2^2$$

### C. [[VQ-VAE-2]] (The Spatial Split)
*   **Core Math:** Splits representation into Top (global structural shapes, heavily compressed) and Bottom (local fine-grained textures, moderately compressed) latent grids:
    $$h_{\text{top}} = E_{\text{top}}(x), \quad h_{\text{bottom}} = E_{\text{bottom}}(x, e_{\text{top}})$$
*   **Decoder Fusion:** The top latents are upsampled and channel-wise concatenated to condition the bottom decoder:
    $$h_{\text{fused}} = \Big[ z_{q,\text{bottom}}, \; \text{Upsample}\big(\text{Decoder}_{\text{top}}(z_{q,\text{top}})\big) \Big]$$
*   **The Prior:** Multi-head self-attention is used in the Top prior to ensure global coherence, while upsampled top-conditioning is passed to Gated PixelCNN layers in the Bottom prior to ensure semantic harmony.

### D. [[RQ-VAE]] (The Cumulative Cascade)
*   **Core Math:** Solves the spatial-vs-codebook dilemma by recursively quantizing residual vectors over $D$ depths using a shared codebook of size $K$:
    $$\mathbf{r}_0 = z, \quad \mathbf{r}_d = \mathbf{r}_d - e(M_d), \quad \hat{Z}^{(d)} = \sum_{d'=1}^d e(M_{d'})$$
*   **Cumulative Loss:** The commitment loss enforces coarse-to-fine decomposition by summing errors across every intermediate depth:
    $$\mathcal{L}_{\text{commit}} = \sum_{d=1}^D \left\| Z - \text{sg}\left[ \hat{Z}^{(d)} \right] \right\|_2^2$$
*   **RQ-Transformer:** Employs a **Spatial Transformer** (modeling spatial context $h_t$ based on the fully reconstructed previous pixel $u_t = \text{PE}_T + \hat{Z}_{t-1}$) and a **Depth Transformer** (predicting the code tuple $(S_{t1}, \dots, S_{tD})$) to reduce attention complexity from $\mathcal{O}(NT^2D^2)$ to $\mathcal{O}(N_{\text{spatial}}T^2 + N_{\text{depth}}TD^2)$.
*   **Exposure Bias:** Mitigated by applying **Stochastic Sampling** on the input sequences during training and **Soft-Labeling** on the Transformer's targets.

### E. [[FSQ]] (The Coordinate Simplification)
*   **Core Math:** Replaces vector lookup and codebook learning entirely with bounded rounding on a coordinate grid of symmetric odd levels $L_i$:
    $$\hat{z}_i = \text{round}_{\text{ste}}\left( \lfloor L_i/2 \rfloor \tanh\left(\frac{z_i}{\lfloor L_i/2 \rfloor}\right) \right)$$
*   **Self-Regularizing Gradients:** The derivative of the bounding tanh function:
    $$\frac{\partial \mathcal{L}}{\partial z_i} = g_i \cdot \left[ 1 - \tanh^2\left(\frac{z_i}{\lfloor L_i/2 \rfloor}\right) \right]$$
    naturally suppresses gradients as $z_i$ approaches the grid boundaries. This keeps continuous coordinates stable and bounded, completely replacing the need for commitment losses or codebook EMA updates.
*   **Coordinate Projection:** Converts multi-dimensional coordinates to flat scalar indices analytically:
    $$\text{index} = \sum_{i=1}^d \left( \hat{z}_i + \left\lfloor \frac{L_i}{2} \right\rfloor \right) \prod_{j=1}^{i-1} L_j$$

---

## 4. Architectural Synthesis (Tokenization vs. Generation)

Across all modern codebook architectures, the two-stage image generation pipeline remains consistent:

```
[ STAGE 1: Lossy Compression ]
Raw Image (X) ───> [ Encoder ] ───> [ Quantizer: VQ / RQ / FSQ ] ───> Quantized Latents (z_q) ───> [ Decoder ] ───> Reconstructed (X_hat)

                                                  │
                                                  ▼ (Tokenize Dataset)
                                          Sequence of Tokens (S)
                                                  │
                                                  ▼
                                      [ STAGE 2: Prior Modeling ]
                                      Train Autoregressive Prior:
                                      P(S_t | S_<t) via Transformer
```

### **Which one to choose?**
1.  Use **[[FSQ]]** for standard large-vocabulary codebooks ($K \ge 1024$). It is mathematically simple, fast, has zero auxiliary losses, and is completely immune to codebook collapse.
2.  Use **[[RQ-VAE]]** if you need extreme spatial compression (e.g., down to $8 \times 8$ or $4 \times 4$ sequences) and can tolerate lookup-based depth stacks, as its cumulative residual cascades maintain high reconstruction fidelity.
3.  Use **[[VQ-VAE-2]]** if you require explicit, physical multi-scale spatial separation (e.g., editing background layouts independently of foreground details).
