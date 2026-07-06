---
tags:
  - generative-models
  - architecture
  - tokenization
  - math
date: 2026-06-29
sources: ["[[raw/Recommendation/Autoregressive Image Generation with Residual Quantization.pdf]]"]
---

# Residual Quantized Variational Autoencoder (RQ-VAE) Deep Dive

This page provides a mathematically rigorous, structurally detailed examination of the **Residual Quantization (RQ)** framework and the **RQ-VAE / RQ-Transformer** architecture, first introduced in the paper *Autoregressive Image Generation with Residual Quantization* (Lee et al., 2022).

It covers the spatial-resolution-vs-codebook-size bottleneck, layer-by-layer execution, the two-component commitment loss formulation, shared residual EMA updates, the dual Spatial/Depth Transformer structures, and the exposure-bias-mitigating training tricks (Soft-Labeling and Stochastic Sampling).

---

## 1. The Spatial-vs-Codebook Bottleneck

In standard VQ-VAE architectures, generating images quickly requires downsampling the spatial resolution of the latent grid heavily (e.g., to $8 \times 8 = 64$ tokens). However, under **rate-distortion theory**, shrinking the spatial grid forces each individual token to represent a larger physical area of the image. To preserve reconstruction fidelity, the capacity of each token must grow exponentially, requiring an extremely large codebook size $K$ (e.g., $K \ge 2^{16} = 65,536$).

Giant codebooks suffer from severe **Codebook Collapse** (or index collapse), where only a tiny fraction of the codebook vectors are ever selected, leaving the rest unselected, untrained, and wasted.

### The RQ-VAE Solution: Residual Quantization (RQ)
Instead of scaling the codebook size $K$, RQ-VAE recursively quantizes **residuals** over $D$ depths (iterations) using a single, compact, shared codebook of size $K$ (e.g., $K = 512$). This represents each spatial location with a **tuple of $D$ discrete codes** $(k_1, \dots, k_D)$ instead of a single index, scaling the representation capacity to $K^D$ possible clusters while only consuming the physical memory of a single codebook of size $K$.

The complete, two-stage framework consisting of RQ-VAE and the subsequent RQ-Transformer is illustrated below:

![[wiki/media/overview_two-stage_framework_of_RQ-VAE_and_RQ-Transforme.png]]

---

## 2. Layer-by-Layer Execution Flow

Below is the precise, shape-tracked execution flow of an image $x$ passing through an RQ-VAE model with depth $D_{\text{depth}} = 4$ and a shared codebook of size $K=512$, embedding dimension $D_{\text{embed}} = 256$:

### Layer 1: Encoder ($E$)
*   **Input:** Raw image batch $X$ (Shape: $[B, 3, 256, 256]$).
*   **Operations:** Downsampling convolutional blocks and residual blocks.
*   **Output:** Continuous latent representation $Z = E(X)$ (Shape: $[B, 256, 8, 8]$).

### Layer 2: Reshape & Flatten
*   **Input:** $Z$ (Shape: $[B, 256, 8, 8]$).
*   **Operations:** Transpose channel dimension to last and flatten spatial grid:
    $$Z \in \mathbb{R}^{N \times 256} \quad \text{where} \quad N = B \times 8 \times 8 = B \times 64$$
*   **Output:** Flat continuous vectors $\{z_1, \dots, z_N\}$ (Shape: $[N, 256]$).

### Layer 3: Recursive Residual Quantization ($\mathcal{RQ}$)
We perform iterative quantization over 4 depths using a shared codebook $C = \{e_1, \dots, e_K\} \subset \mathbb{R}^{256}$:

$$\begin{array}{l}
\textbf{Initialize residual:} \quad \mathbf{r}_0 = z \in \mathbb{R}^{256} \\
\hline
\textbf{Depth } d = 1: \\
1: k_1 = \arg\min_{j \in [1, K]} \|\mathbf{r}_0 - e_j\|_2^2 \quad \text{(Shape: } [N]\text{)} \\
2: \mathbf{r}_1 = \mathbf{r}_0 - e_{k_1} \quad \text{(Shape: } [N, 256]\text{)} \\
\hline
\textbf{Depth } d = 2: \\
3: k_2 = \arg\min_{j \in [1, K]} \|\mathbf{r}_1 - e_j\|_2^2 \quad \text{(Shape: } [N]\text{)} \\
4: \mathbf{r}_2 = \mathbf{r}_1 - e_{k_2} \quad \text{(Shape: } [N, 256]\text{)} \\
\hline
\textbf{Depth } d = 3: \\
5: k_3 = \arg\min_{j \in [1, K]} \|\mathbf{r}_2 - e_j\|_2^2 \quad \text{(Shape: } [N]\text{)} \\
6: \mathbf{r}_3 = \mathbf{r}_2 - e_{k_3} \quad \text{(Shape: } [N, 256]\text{)} \\
\hline
\textbf{Depth } d = 4: \\
7: k_4 = \arg\min_{j \in [1, K]} \|\mathbf{r}_3 - e_j\|_2^2 \quad \text{(Shape: } [N]\text{)} \\
8: \mathbf{r}_4 = \mathbf{r}_3 - e_{k_4} \quad \text{(The final quantization error)} \\
\hline
\end{array}$$

### Layer 4: Reconstruction Assembly & STE
*   **Input:** Selected embeddings $\{e_{k_1}, e_{k_2}, e_{k_3}, e_{k_4}\}$.
*   **Operations:**
    1. Sum up the codebook approximations to construct the final quantized tensor:
       $$\hat{Z} = \sum_{d=1}^4 e_{k_d} \quad \text{(Shape: } [N, 256]\text{)}$$
    2. Apply the Straight-Through Estimator (STE) to copy gradients back to the encoder:
       $$z_q = Z + \text{sg}[\hat{Z} - Z] \quad \text{(Shape: } [N, 256]\text{)}$$
    3. Reshape and permute back to channel-first representation: $[N, 256] \to [B, 256, 8, 8]$.
*   **Output:** Quantized latent grid $z_q$ (Shape: $[B, 256, 8, 8]$) and discrete code index tuples $M$ (Shape: $[B, 8, 8, 4]$).

### Layer 5: Decoder ($G$)
*   **Input:** Quantized latent grid $z_q$ (Shape: $[B, 256, 8, 8]$).
*   **Operations:** Upsampling transposed convolutions and residual blocks.
*   **Output:** Reconstructed image $\hat{X} = G(z_q)$ (Shape: $[B, 3, 256, 256]$).

---

## 3. Training Loss and Decoupled Updates

The autoencoder is trained using a highly specific two-component loss function:

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{commit}}$$

### 1. Reconstruction Loss ($\mathcal{L}_{\text{recon}}$)
$$\mathcal{L}_{\text{recon}} = \|X - \hat{X}\|_2^2$$
*(Measures pixel-level reconstruction error. It optimizes the encoder and decoder parameters).*

### 2. Commitment Loss ($\mathcal{L}_{\text{commit}}$)
$$\mathcal{L}_{\text{commit}} = \sum_{d=1}^D \left\| Z - \text{sg}\left[ \hat{Z}^{(d)} \right] \right\|_2^2 \quad \text{where} \quad \hat{Z}^{(d)} = \sum_{d'=1}^d e(M_{d'})$$

#### **The Role of the Cumulative Sum $\hat{Z}^{(d)}$:**
Instead of a single commitment term on the final sum ($\|Z - \text{sg}[\hat{Z}^{(D)}]\|_2^2$), the commitment loss sums the errors across **every intermediate depth $d$**. This forces the model to decompose the information sequentially in a coarse-to-fine manner (the first depth must capture the bulk of the signal, while subsequent depths resolve finer residuals).

#### **The Role of the Stop-Gradient (`sg`):**
Because the Codebook is updated analytically via Exponential Moving Averages (EMA), the codebook embeddings must not receive any gradients from the backpropagation optimizer. The `sg[...]` operator freezes the codebook vectors, ensuring that the commitment loss **exclusively updates the Encoder parameters**, acting as a stable spatial anchor.

---

## 4. Codebook Updates via Multi-Scale Residual EMA

Since the codebook $C$ is shared across all depths $D$, we accumulate K-means cluster counts ($n_i$) and coordinate sums ($m_i$) across **all depths simultaneously** inside each training batch using the residual vectors $\mathbf{r}_{d-1}$:

1.  **Count Assignments:** Count how many times codebook vector $e_i$ was selected across all coordinates and depths in the batch:
    $$n_i^{(t)} = \sum_{h, w} \sum_{d=1}^D \mathbb{1}[M_{hwd} = i]$$
2.  **Sum Residual Coordinates:** Sum up all the residual vectors $\mathbf{r}_{d-1}$ that mapped to codebook vector $e_i$ in the batch:
    $$\sum_j \mathbf{r}_{i, j}^{(t)} = \sum_{h, w} \sum_{d=1}^D \mathbf{r}_{d-1} \cdot \mathbb{1}[M_{hwd} = i]$$
3.  **Apply Exponential Moving Average (EMA) with decay $\gamma = 0.99$:**
    $$N_i^{(t)} := \gamma N_i^{(t-1)} + (1-\gamma) n_i^{(t)}$$
    $$m_i^{(t)} := \gamma m_i^{(t-1)} + (1-\gamma) \sum_j \mathbf{r}_{i, j}^{(t)}$$
4.  **Overwrite Embedding Coordinate:**
    $$e_i^{(t)} := \frac{m_i^{(t)}}{N_i^{(t)}}$$

By updating the shared codebook with the multi-scale residual vectors $\mathbf{r}_{d-1}$ rather than the raw continuous representations $Z$, the codebook embeddings naturally learn to represent coordinate offsets across different statistical scales, eliminating index collapse completely.

---

## 5. Stage 2 prior training: The RQ-Transformer

Unfolding a 2D grid of size $T = H \times W$ with depth $D$ into a flat 1D sequence of length $T \times D$ results in a quadratic attention complexity of $\mathcal{O}(N T^2 D^2)$, which is computationally prohibitive. 

To solve this, the **RQ-Transformer** splits the sequence learning into two separate, cooperating transformers: the **Spatial Transformer** and the **Depth Transformer**, reducing complexity to $\mathcal{O}(N_{\text{spatial}}T^2 + N_{\text{depth}}TD^2)$.

### A. The Spatial Transformer (Capturing Space)
The Spatial Transformer processes the sequence of spatial positions $t \in [1, T]$ to extract a context vector $h_t$ summarizing all historical spatial information $S_{<t}$:

$$h_t = \text{SpatialTransformer}(u_1, \dots, u_t) \quad \text{(Equation 11)}$$

#### **The Input Vector $u_t$ (Equation 10):**
The input $u_t$ to the Spatial Transformer at position $t$ is defined as the sum of a learnable positional embedding and the fully reconstructed continuous latent vector of the *previous* spatial position $t-1$:

$$u_t = \text{PE}_T(t) + \sum_{d=1}^D e(S_{t-1, d}) \quad \text{for } t > 1 \quad \text{(Equation 10)}$$

*   $\text{PE}_T(t)$ is a **learnable positional embedding** for spatial position $t$ in the raster-scan order.
*   The second term $\sum_{d=1}^D e(S_{t-1, d})$ is the **fully reconstructed continuous vector** of the previous pixel, condensing the coarse-to-fine information of the previous coordinate into a single $D_{\text{model}}$-dimensional vector.
*   For the first position $t=1$, the input is a dedicated **learnable start-of-sequence embedding** $u_1$.

---

### B. The Depth Transformer (Capturing Residual Depth)
Given the context vector $h_t$ at spatial position $t$, the Depth Transformer autoregressively predicts the $D$ codebook indices $(S_{t1}, \dots, S_{tD})$:

$$p_{td} = \text{DepthTransformer}(v_{t1}, \dots, v_{td}) \quad \text{(Equation 13)}$$

Where $p_{td}$ is the conditional probability distribution:
$$p_{td}(k) = P\left( S_{td} = k \;\middle|\; S_{<t}, \; S_{t, <d} \right)$$

#### **The Input Vector $v_{td}$ (Equation 12):**
The input $v_{td}$ to the Depth Transformer at depth $d$ is defined as:

$$v_{td} = \text{PE}_D(d) + \sum_{d'=1}^{d-1} e(S_{td'}) \quad \text{for } d > 1 \quad \text{(Equation 12)}$$
$$v_{t1} = \text{PE}_D(1) + h_t \quad \text{for } d = 1$$

*   $\text{PE}_D(d)$ is a **learnable positional embedding** for depth $d$, shared across all spatial positions $t$.
*   The second term represents the cumulative sum of the reconstructed embeddings up to depth $d-1$ (the continuous partial reconstruction $\hat{Z}^{(d-1)}$).

---

## 6. Mitigating Exposure Bias via Coordinated Training Tricks

During sequential inference, prediction errors at early depths accumulate and propagate exponentially, deteriorating generation quality. To solve this **Exposure Bias**, RQ-Transformer employs two cooperative, specialized training techniques:

```
                         [ Coordinated Training Tricks ]

     1. INPUT SEQUENCES (Stochastic Sampling)  ───>  Introduces Realistic "Noise"
                                                                │
                                                                ▼ (Prior Model)
                                                    [ RQ-Transformer predictions ]
                                                                │
                                                                ▼
     2. TARGET LABELS (Soft-Labeling)          ───>  Rewards "Geometrically Close" choices
```

### A. Stochastic Sampling (Adopted on the INPUT SEQUENCES)
During Stage 1 dataset tokenization, instead of selecting the codebook index deterministically using hard `argmin` (which is the mathematical equivalent of setting the temperature $\tau \to 0$), we **sample** the index $S_{td}$ directly from the soft probability distribution $Q_\tau(k \mid \mathbf{r}_{d-1})$. 
*   **The Effect:** This intentionally injectes realistic, minor "mistakes" or "jitter" into the training sequences. This teaches the Transformer how to recover and self-correct when conditioned on slightly corrupted historical contexts.

### B. Soft-Labeling (Adopted on the TARGET LABELS)
Instead of standard hard one-hot targets, the Transformer's supervision labels are calculated based on a temperature-controlled soft distribution $Q_\tau$ from the geometric distances between the continuous residual vector and the codebook embeddings:

$$\text{Target Label}_k = Q_\tau(k \mid \mathbf{r}_{d-1}) \propto \exp \left( -\frac{\|\mathbf{r}_{d-1} - e_k\|_2^2}{\tau} \right) \quad \text{(Equation 15)}$$

*   **The Effect:** This is used as the target in our Cross-Entropy Loss. Instead of penalizing all incorrect indices equally, it **supervises the Transformer with explicit geometric relationships**, rewarding the model for choosing codebook vectors that are physically close in coordinate space to our target residual vector.

*Ablation studies show that utilizing **both** Stochastic Sampling and Soft-Labeling yields the best performance, improving ImageNet FID from **14.06** (baseline) to **13.11**.*

---

## 7. Production PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, depth, commitment_beta=0.25):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.depth = depth
        self.beta = commitment_beta
        
        # Shared Codebook Embeddings
        self.codebook = nn.Embedding(self.K, self.D)
        self.codebook.weight.data.uniform_(-1.0 / self.K, 1.0 / self.K)

    def forward(self, z_e):
        # Permute channel to last: [B, C, H, W] -> [B, H, W, C]
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        flat_z_e = z_e.view(-1, self.D) # [N, D]
        
        residual = flat_z_e.clone()
        quantized_sum = torch.zeros_like(flat_z_e)
        
        indices_list = []
        loss_commit = 0.0
        
        # Recursive Residual Quantization
        for d in range(self.depth):
            # Calculate L2 distance to all K codebook vectors
            distances = (torch.sum(residual**2, dim=1, keepdim=True) 
                         - 2 * torch.matmul(residual, self.codebook.weight.t())
                         + torch.sum(self.codebook.weight**2, dim=1)) # [N, K]
            
            # Argmin lookup
            indices = torch.argmin(distances, dim=1) # [N]
            indices_list.append(indices.unsqueeze(1))
            
            # Look up embedding vector
            e_k = self.codebook(indices) # [N, D]
            
            # Cumulative quantized sum
            quantized_sum = quantized_sum + e_k
            
            # Calculate intermediate commitment loss (Equation 7)
            # Loss = Sum of || Z_e - sg[ Z_hat^(d) ] ||^2
            loss_commit += F.mse_loss(flat_z_e, quantized_sum.detach())
            
            # Compute next residual
            residual = residual - e_k
            
        # Final quantized state
        z_q_final = quantized_sum
        
        # Total commitment loss (scaled by beta)
        loss_vq = self.beta * loss_commit
        
        # Straight-Through Estimator (STE)
        z_q = flat_z_e + (z_q_final - flat_z_e).detach()
        
        # Reshape back to spatial grid
        z_q = z_q.view(z_e.shape).permute(0, 3, 1, 2).contiguous() # [B, C, H, W]
        indices_map = torch.cat(indices_list, dim=1).view(z_e.shape[0], z_e.shape[1], z_e.shape[2], self.depth) # [B, H, W, depth]
        
        return z_q, loss_vq, indices_map
```
