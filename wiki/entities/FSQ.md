---
tags:
  - generative-models
  - architecture
  - tokenization
  - math
date: 2026-07-01
sources: ["[[raw/Recommendation/Finite Scalar Quantization VQ-VAE Made Simple.pdf]]"]
---

# Finite Scalar Quantization (FSQ) Deep Dive

This page provides a mathematically rigorous, code-level deep dive into **Finite Scalar Quantization (FSQ)**, first introduced in the paper *Finite Scalar Quantization: VQ-VAE Made Simple* (Mentzer et al., Google Research, 2023). 

It details the core mechanics of lookup-free discrete representation learning, explains the symmetric level allocations ($L_i$), provides the mathematical derivation of the bounding function derivative chain, demonstrates flat scalar indexing, and provides a production PyTorch module.

---

## 1. Why FSQ? (Curing the Drawbacks of Vector Quantization)

In standard Vector Quantization (VQ) used in VQ-VAEs and RQ-VAEs, latent variables are discretized by mapping continuous features to an unstructured point cloud of codebook vectors using a nearest-neighbor lookup:

$$z_q = \arg\min_j \|z_e - e_j\|_2^2$$

This approach suffers from several severe training bottlenecks:
1.  **Codebook Collapse (Index Collapse):** The continuous vectors are free to drift, causing the optimizer to underutilize the codebook. Many embeddings are never selected or trained, requiring complex engineering heuristics (EMA, restarts, clustering resets) to resolve.
2.  **Complex Loss Formulation:** Stabilizing the continuous-discrete boundary requires auxiliary loss terms—**Codebook Loss** and **Commitment Loss**—which are scaled by hyperparameters ($\beta$) that require tuning.
3.  **High Computational Complexity:** Finding the closest vector in a $D$-dimensional space requires an $O(H \times W \times K \times D)$ distance matrix calculation, which is slow and memory-intensive for large codebooks.

### The FSQ Simplification
FSQ completely **eliminates the codebook, lookup operations, commitment losses, and index collapse** by replacing vector lookup with simple **element-wise integer rounding**. Instead of searching an unstructured point cloud, FSQ structures the discrete latent space as a regular, bounded coordinate grid.

---

## 2. Mathematical Formulation of FSQ

For a low-dimensional continuous representation $z \in \mathbb{R}^d$ (typically $d \in [3, 8]$):

### Step A: Symmetric Odd Levels ($L_i$)
For each of our $d$ dimensions, we assign an **odd number of discrete levels** $L_i$ (e.g., $L_1 = 5, L_2 = 3$). Because they are odd integers, the coordinate set is perfectly symmetrical around zero, ensuring that **absolute zero is an allowed discrete state**:

$$\mathcal{S}_i = \left\{ -\left\lfloor \frac{L_i}{2} \right\rfloor, \; \dots, \; 0, \; \dots, \; \left\lfloor \frac{L_i}{2} \right\rfloor \right\}$$

*   If $L_i = 3$, the allowed discrete coordinate set is $\{-1, 0, 1\}$.
*   If $L_i = 5$, the allowed discrete coordinate set is $\{-2, -1, 0, 1, 2\}$.

The total size of the implicit codebook is the product of the levels across all dimensions:
$$K = \prod_{i=1}^d L_i$$

### Step B: The Bounding Function ($f$)
To restrict the continuous encoder output $z_i$ to the open interval $(-L_i/2, L_i/2)$, we pass it through a scaled bounding function $f$:

$$f(z_i) = \left\lfloor \frac{L_i}{2} \right\rfloor \tanh\left( \frac{z_i}{\lfloor L_i/2 \rfloor} \right)$$

### Step C: Bounded Rounding with STE
We round the bounded coordinate to the nearest integer and apply the Straight-Through Estimator (STE) to copy gradients during backpropagation:

$$\hat{z}_i = \text{round}_{\text{ste}}\big( f(z_i) \big) = f(z_i) + \text{sg}\big[ \text{round}(f(z_i)) - f(z_i) \big]$$

This ensures that:
*   **Forward Pass:** Evaluates to exactly $\text{round}(f(z_i))$, mapping the coordinate directly to a discrete integer in $\mathcal{S}_i$.
*   **Backward Pass:** The gradient of the rounding step is treated as identity: $\frac{\partial \hat{z}_i}{\partial f(z_i)} = 1$.

---

## 3. The Derivative Chain: Self-Regularizing Boundaries

Because FSQ is lookup-free, it is trained using **only the reconstruction loss**:

$$\mathcal{L} = \|X - \text{Decoder}(\hat{z})\|_2^2$$

Let $g_i = \frac{\partial \mathcal{L}}{\partial \hat{z}_i}$ be the gradient of the loss backpropagated from the Decoder to our quantized coordinate. We compute the derivative of the loss with respect to our raw continuous encoder output $z_i$ using the chain rule:

$$\frac{\partial \mathcal{L}}{\partial z_i} = \frac{\partial \mathcal{L}}{\partial \hat{z}_i} \cdot \frac{\partial \hat{z}_i}{\partial f(z_i)} \cdot \frac{\partial f(z_i)}{\partial z_i}$$

Evaluating the derivatives:
1.  **Decoder Gradient:** $\frac{\partial \mathcal{L}}{\partial \hat{z}_i} = g_i$
2.  **STE Rounding:** $\frac{\partial \hat{z}_i}{\partial f(z_i)} = 1$
3.  **Bounding Derivative:** Let $C_i = \lfloor L_i/2 \rfloor$. The derivative of $f(z_i) = C_i \tanh(z_i/C_i)$ is:
    $$\frac{\partial f(z_i)}{\partial z_i} = 1 - \tanh^2\left( \frac{z_i}{C_i} \right) = 1 - \left( \frac{f(z_i)}{C_i} \right)^2$$

Substituting these back into the chain yields the final derivative formula:

$$\frac{\partial \mathcal{L}}{\partial z_i} = g_i \cdot \left[ 1 - \tanh^2\left( \frac{z_i}{\lfloor L_i/2 \rfloor} \right) \right] = g_i \cdot \left[ 1 - \left( \frac{f(z_i)}{\lfloor L_i/2 \rfloor} \right)^2 \right]$$

### **The Self-Regularizing Phenomenon:**
*   **When $z_i \approx 0$ (Center of the Grid):**
    The derivative becomes $\frac{\partial \mathcal{L}}{\partial z_i} \approx g_i$. Gradients flow back completely unhindered to update the encoder weights.
*   **When $z_i \to \pm \infty$ (Far Out/At the Boundaries):**
    The term $\tanh^2(z_i/C_i) \to 1$, causing the bracketed term $1 - \tanh^2(z_i/C_i) \to 0$. The gradient $\frac{\partial \mathcal{L}}{\partial z_i}$ is **completely suppressed to 0**.

This gradient suppression creates a natural, soft bounding constraint. It prevents $z_i$ from growing infinitely and forces the continuous values to cluster stably near the outermost plateaus of the $\tanh$ curve, completely replacing the need for an auxiliary commitment loss!

---

## 4. Flat Scalar Indexing (Coordinate Projection)

Autoregressive models (PixelCNN / Transformers) expect a single flat integer token ID in the range $[0, K-1]$. We project our multi-dimensional rounded integer coordinate tuple $(\hat{z}_1, \dots, \hat{z}_d)$ to a single scalar index using a base-conversion sum:

$$\text{index} = \sum_{i=1}^d \left( \hat{z}_i + \left\lfloor \frac{L_i}{2} \right\rfloor \right) \prod_{j=1}^{i-1} L_j$$

### **Example Projection:**
Suppose we have a $d=3$ dimensional space with levels $L = [3, 5, 4]$ (implicit codebook size $K = 3 \times 5 \times 4 = 60$). Let our rounded coordinates be $\hat{\mathbf{z}} = [0, 2, -1]$:
1.  Shift coordinates to be non-negative:
    *   $\hat{z}_1 + \lfloor 3/2 \rfloor = 0 + 1 = 1$
    *   $\hat{z}_2 + \lfloor 5/2 \rfloor = 2 + 2 = 4$
    *   $\hat{z}_3 + \lfloor 4/2 \rfloor = -1 + 2 = 1$
2.  Project:
    $$\text{index} = 1 \cdot (1) + 4 \cdot (3) + 1 \cdot (3 \times 5) = 1 + 12 + 15 = 28$$
    The coordinate $[0, 2, -1]$ maps uniquely to flat index **28**.

---

## 5. FSQ vs. VQ Efficiency Boundary

Empirical studies show a clear performance boundary based on the target codebook size $K$:
*   **For large codebooks ($K \ge 2^{10} = 1024$):** FSQ **outperforms** standard VQ. Because FSQ structures the discrete representation as a dense coordinate grid, it is completely immune to index collapse and codebook drift, allowing stable scaling.
*   **For small codebooks ($K < 2^{10}$):** Standard VQ can occasionally be slightly more optimal. This is because VQ has the freedom to position cluster centroids anywhere in continuous space (an unstructured point cloud), whereas FSQ is strictly constrained to a regular grid coordinate structure.

---

## 6. Production PyTorch Implementation

```python
import torch
import torch.nn as nn
import numpy as np

class FiniteScalarQuantizer(nn.Module):
    def __init__(self, levels):
        super().__init__()
        # levels is a list of odd integers, e.g. [8, 5, 5, 5]
        self.levels = np.array(levels)
        self.d = len(levels)
        
        # Calculate shifts: floor(L_i / 2)
        self.shifts = self.levels // 2
        
        # Calculate cumulative products for base conversion
        self.basis = np.concatenate([[1], np.cumprod(self.levels[:-1])])
        
        # Total codebook size K
        self.K = int(np.prod(self.levels))
        
        # Convert to PyTorch buffers
        self.register_buffer("shifts_tensor", torch.tensor(self.shifts, dtype=torch.float32))
        self.register_buffer("levels_tensor", torch.tensor(self.levels, dtype=torch.float32))
        self.register_buffer("basis_tensor", torch.tensor(self.basis, dtype=torch.int64))

    def forward(self, z_e):
        # z_e shape: [B, d, H, W] -> convert to [B, H, W, d]
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        
        # 1. Apply Bounding Function f(z) = shift * tanh(z / shift)
        # Note: we use self.shifts_tensor to scale the bounds
        bounded_z = self.shifts_tensor * torch.tanh(z_e / self.shifts_tensor)
        
        # 2. Decimal Rounding
        rounded_z = torch.round(bounded_z)
        
        # 3. Straight-Through Estimator (STE)
        z_q = bounded_z + (rounded_z - bounded_z).detach()
        
        # Convert back to [B, d, H, W] for the decoder
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q

    def coordinates_to_indices(self, z_q):
        # Input z_q: [B, d, H, W] -> convert to [B, H, W, d]
        z_q = z_q.permute(0, 2, 3, 1).contiguous()
        
        # Shift to be non-negative integers: round(z_q) + shift
        shifted_z = torch.round(z_q).to(torch.int64) + self.shifts_tensor.to(torch.int64)
        
        # Project multi-dimensional coordinates to flat scalar indices
        # index = sum( shifted_z_i * basis_i )
        indices = torch.sum(shifted_z * self.basis_tensor, dim=-1) # [B, H, W]
        
        return indices

    def indices_to_coordinates(self, indices):
        # Input indices: [B, H, W]
        B, H, W = indices.shape
        flat_indices = indices.view(-1, 1) # [B*H*W, 1]
        
        # De-project indices back to multi-dimensional coordinates
        # shifted_z_i = (flat_indices // basis_i) % level_i
        shifted_z = (flat_indices // self.basis_tensor) % self.levels_tensor.to(torch.int64)
        
        # Shift back to signed representation: shifted_z - shift
        coords = shifted_z.to(torch.float32) - self.shifts_tensor
        
        # Reshape to standard [B, d, H, W]
        coords = coords.view(B, H, W, self.d).permute(0, 3, 1, 2).contiguous()
        
        return coords
```
