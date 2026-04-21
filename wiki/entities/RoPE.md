---
tags:
  - llm
  - architecture
  - attention
  - math
date: 2026-04-17
sources:
  - "[[十分钟看懂RoPE Summary]]"
---

# Rotary Position Embedding (RoPE) Deep Dive

This page provides a deep mathematical and structural dive into Rotary Position Embedding (RoPE), a state-of-the-art relative [[Positional Encoding]] technique first proposed in the *RoFormer* paper and widely adopted in modern foundation models like LLaMA and ChatGLM.

For a high-level comparison of RoPE against other encoding schemes (like Sinusoidal and ALiBi), see the [[Positional Encoding]] page.

## 1. Mathematical Formulation

RoPE hypothesizes that there is a function $g$ that computes the inner product (the attention score) between a Query vector at position $m$ ($\mathbf{\mathit{q}}_m$) and a Key vector at position $n$ ($\mathbf{\mathit{k}}_n$) such that the output depends *only* on their relative distance $m-n$.

$$ \langle \mathbf{\mathit{f}}_{q} ( \mathbf{\mathit{x}}_{m} , m ) , f_{k} ( \mathbf{\mathit{x}}_{n} , n ) \rangle = g ( \mathbf{\mathit{x}}_{m} , \mathbf{\mathit{x}}_{n} , m - n ) $$

Assuming a 2-dimensional token embedding ($d=2$), RoPE proves (using Euler's formula $e^{i\theta} = \cos \theta + i\sin \theta$) that this relative encoding can be achieved by multiplying the vectors by a **Rotation Matrix**. 

$$ f_q(\mathbf{x}_m, m) = (\mathbf{W}_q \mathbf{x}_m)e^{im\theta} $$
Which translates geometrically to:
$$ \begin{aligned} f_q(\mathbf{x}_m, m) &= \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_m^{(1)} \\ q_m^{(2)} \end{pmatrix} \end{aligned} $$

### The Core Breakthrough: Relative Inner Product
The brilliance of RoPE is revealed when taking the inner product of the Query (rotated by $m$) and the Key (rotated by $n$). 

By taking the dot product of these two absolutely rotated vectors, the math simplifies. The inner product of two vectors rotated by $m\theta$ and $n\theta$ is identical to taking the original vectors and rotating one by the *difference* of their angles ($(m-n)\theta$). Therefore, the target function $g$ resolves perfectly to a rotation matrix based *exclusively* on the relative distance $m-n$:

$$ g( \mathbf{\mathit{x}}_{m} , \mathbf{\mathit{x}}_{n} , m - n ) = \begin{pmatrix}q_{m}^{( 1 )} & q_{m}^{( 2 )}\end{pmatrix} \begin{pmatrix} \cos((m-n)\theta) & -\sin((m-n)\theta) \\ \sin((m-n)\theta) & \cos((m-n)\theta) \end{pmatrix} \begin{pmatrix}k_{n}^{( 1 )} \\ k_{n}^{( 2 )}\end{pmatrix} $$

This proves that applying an absolute positional rotation to individual tokens naturally yields a purely relative attention score.

### Multi-Dimensional Extension ($d$-dimension)
To scale this 2D rotation to an arbitrary dimension $d$ (which must be even), RoPE pairs the dimensions up (e.g., $x$ with $y$) and constructs a large, block-diagonal orthogonal matrix $\mathbf{\mathit{R}}_{\Theta, m}^d$.

$$ \mathbf{\mathit{R}}_{\Theta , m}^{d} = \begin{pmatrix}\cos m \theta_{0} & -\sin m \theta_{0} & 0 & 0 & \cdots & 0 & 0 \\ \sin m \theta_{0} & \cos m \theta_{0} & 0 & 0 & \cdots & 0 & 0 \\ 0 & 0 & \cos m \theta_{1} & -\sin m \theta_{1} & \cdots & 0 & 0 \\ 0 & 0 & \sin m \theta_{1} & \cos m \theta_{1} & \cdots & 0 & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & 0 & \cdots & \cos m \theta_{d/2 - 1} & -\sin m \theta_{d/2 - 1} \\ 0 & 0 & 0 & 0 & \cdots & \sin m \theta_{d/2 - 1} & \cos m \theta_{d/2 - 1}\end{pmatrix} $$

Where $\Theta = \{ \theta_i = 10000^{-2(i-1)/d}, i \in [1, 2, ..., d/2] \}$.

## 2. Efficient Computation

Directly multiplying the $Q$ or $K$ vector by $\mathbf{\mathit{R}}_{\Theta, m}^d$ is computationally wasteful because the matrix is extremely sparse (mostly zeros outside the diagonal blocks). In practice, RoPE is implemented efficiently using Hadamard (element-wise) products ($\otimes$).

$$ \mathbf{\mathit{R}}_{\Theta , m}^{d} \mathbf{\mathit{x}} = \begin{pmatrix}x_{0} \\ x_{1} \\ x_{2} \\ x_{3} \\ \vdots \\ x_{d - 2} \\ x_{d - 1}\end{pmatrix} \bigotimes \begin{pmatrix}\cos m \theta_{0} \\ \cos m \theta_{0} \\ \cos m \theta_{1} \\ \cos m \theta_{1} \\ \vdots \\ \cos m \theta_{d/2 - 1} \\ \cos m \theta_{d/2 - 1}\end{pmatrix} + \begin{pmatrix}- x_{1} \\ x_{0} \\ - x_{3} \\ x_{2} \\ \vdots \\ - x_{d - 1} \\ x_{d - 2}\end{pmatrix} \bigotimes \begin{pmatrix}\sin m \theta_{0} \\ \sin m \theta_{0} \\ \sin m \theta_{1} \\ \sin m \theta_{1} \\ \vdots \\ \sin m \theta_{d/2 - 1} \\ \sin m \theta_{d/2 - 1}\end{pmatrix} $$

## 3. Key Properties

### Long-Range Decay (远程衰减)
RoPE mathematically guarantees that tokens further apart have lower attention scores. Using the **Abel Transformation** (Summation by parts), it can be proven that the inner product (attention score) has an upper bound that strictly decays as the relative distance $m-n$ increases. 

This decay is carefully tuned by the base frequency $\theta_i = 10000^{-2i/d}$. As distance increases, the high-frequency components undergo rapid "rotation cancellation" (oscillating wildly between positive and negative), which organically drives the overall magnitude of the dot product down. 

This implicit locality bias is highly beneficial: without it, LLMs would lose focus and suffer from attention dilution over long contexts. The model is mathematically biased to prioritize nearby tokens.

### Length Extrapolation (外推性)
Because RoPE relies exclusively on Rotation Matrices ($\mathbf{\mathit{R}}$), it is mathematically continuous for any arbitrary position, unlike explicitly sized Learnable Positional Encodings. A rotation matrix is an **orthogonal matrix**, meaning it guarantees that:
1. Vectors are only rotated (angles change), but their distance/magnitude (norms) remain perfectly preserved.
2. It prevents numerical overflow or underflow at extreme context lengths.
3. The geometric relationship is fully reversible via the inverse matrix $\mathbf{\mathit{R}}^{-1}$.

#### Advanced Extrapolation Techniques
While RoPE theoretically supports infinite length, performance degrades when extending the sequence significantly past the trained length (e.g., training on 4096 and inferencing on 8192). To bridge this gap, modern models (like LLaMA 3) employ length extrapolation algorithms:
1. **Position Interpolation (PI):** Linearly compresses the inference positions to fit within the training bounds. E.g., if inferencing at $2x$ the trained length, the position index is simply divided by 2.
2. **NTK-aware Scaling:** Acknowledges that high-frequency dimensions (which handle local precision) are more sensitive to stretching than low-frequency dimensions. It applies a non-uniform scaling factor across the dimensions.
3. **YaRN (Yet another RoPE extensioN):** An advanced combination of dimension-specific frequency scaling and attention temperature adjustments, allowing models to extrapolate to 128K+ context windows with near-zero performance degradation.

## 4. Production PyTorch Implementations

In production, models like **Meta's LLaMA** take a highly elegant approach to implementing the efficient computation equation shown above. Instead of manually slicing and interleaving the dimensions with negatives, they natively cast the vectors into the complex domain `torch.view_as_complex`, multiply them directly by complex polar coordinates `torch.polar`, and cast them back `torch.view_as_real`. 

*This complex multiplication perfectly emulates the 2D rotation matrix mathematically.*

```python
# LLaMA RoPE Implementation Excerpt
def apply_rotary_emb(xq, xk, freqs_cis):
    # Reshape vectors into 2D pairs (e.g., [x, y])
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # Cast to complex numbers [x + yi]
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    # Perform complex multiplication (which natively rotates the vector)
    # Then flatten the 2D pairs back into standard real vectors
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)
```