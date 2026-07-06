---
tags:
  - generative-models
  - architecture
  - tokenization
  - math
date: 2026-06-23
sources: ["[[raw/Recommendation/Generating Diverse High-Fidelity Images with VQ-VAE-2.pdf]]"]
---

# Vector Quantized Variational Autoencoder 2 (VQ-VAE-2) Deep Dive

This page provides a mathematically rigorous, structurally detailed examination of the **VQ-VAE-2** architecture, first proposed in *Generating Diverse High-Fidelity Images with VQ-VAE-2* (Razavi et al., 2019). 

It details the hierarchical multi-scale latent space, provides exact mathematical formulations for the conditional links in both Stage 1 (Reconstruction) and Stage 2 (Prior generation), details the line-by-line operations of Algorithms 1 and 2 exactly as written in the original paper using LaTeX formatting, defines the equations referenced in the algorithms, explains how unconditioned and class-conditional starting tokens are handled, and outlines the deep spectral/temporal connection between hierarchical autoencoding and Diffusion models.

---

## 1. Hierarchical Multi-Scale Latent Space

To generate high-resolution, complex images (e.g., $256 \times 256$ or $512 \times 512$ pixels), VQ-VAE-2 introduces a **hierarchical multi-scale representation** consisting of two (or more) tiers:

```
                          [ VQ-VAE-2 Hierarchy ]
                          
                           x (Input: 256x256x3)
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        [ Bottom Encoder ] ─────────┐        [ Top Encoder ]
                │                   │                 │
                ▼                   ▼                 ▼
          Local Features     z_e_top (32x32xD) ──> z_q_top (32x32xD) 
                │                   │                 │
                ▼                   │                 │ (Upsample)
        [ Concatenate ] <───────────┼─────────────────┘ 
                │                   │
                ▼                   │
       z_q_bottom (64x64xD)         │
                │                   │
                ▼                   ▼
         [ Bottom Decoder ] <───────┴──────── [ Top Decoder ]
                │
                ▼
        x_hat (256x256x3)
```

The physical architecture diagram of VQ-VAE-2's hierarchical encoding, decoding, and generation processes can be seen below:

![[wiki/media/VQ-VAE_2_architecture.png]]

### The Division of Labor
1.  **Top Level (Global/Coarse):** Compresses the input heavily (e.g., down to $32 \times 32 \times D$, approximately $3072\times$ smaller than pixel space). It is responsible for modeling **global geometry, shape, and overall semantic structure** (e.g., identifying that "there is a dog").
2.  **Bottom Level (Local/Fine):** Compresses the input moderately (e.g., down to $64 \times 64 \times D$, approximately $192\times$ smaller than pixel space). Conditioned on the Top level, it represents **fine-grained local details, textures, and sharp edges** (e.g., the texture of the dog's fur).

---

## 2. The Conditional Link

To ensure semantic harmony and eliminate capacity redundancy, the bottom level is explicitly conditioned on the top level. Without this conditional link:
*   The bottom level would waste significant capacity trying to re-learn spatial coordinates.
*   During generation, the independent top prior (global shape) and bottom prior (local textures) could clash semantically (e.g., generating bird shapes covered in fish scales), leading to chaotic visual artifacts.

### A. Mathematical Formulation: Decoder Fusion (Stage 1)
In the reconstruction phase, the top latent $z_{q,\text{top}}$ is decoded, upsampled, and channel-wise concatenated with the bottom latent $z_{q,\text{bottom}}$ before being passed to the final decoding layers:

1.  **Top-Level Feature Mapping:**
    $$h_{\text{top}} = \text{Decoder}_{\text{top}}(z_{q,\text{top}}) \quad \text{(Shape: } 32 \times 32 \times D\text{)}$$
2.  **Bilinear/Transpose Convolutional Upsampling:**
    $$\tilde{h}_{\text{top}} = \text{Upsample}(h_{\text{top}}) \quad \text{(Shape: } 64 \times 64 \times D\text{)}$$
3.  **Channel Concatenation & Bottom Decoding:**
    $$h_{\text{bottom}} = \text{Decoder}_{\text{bottom}}\Big( \big[ z_{q,\text{bottom}}, \; \tilde{h}_{\text{top}} \big] \Big) \quad \text{(Shape: } 64 \times 64 \times 2D\text{)}$$
    *(where $[\cdot, \cdot]$ represents channel-wise concatenation).*
4.  **Final Reconstruction:**
    $$\hat{x} = \text{OutputProjection}(h_{\text{bottom}}) \quad \text{(Shape: } 256 \times 256 \times 3\text{)}$$

### B. Mathematical Formulation: Prior Gated Conditioning (Stage 2)
During autoregressive prior training, the bottom PixelCNN is conditioned on the top prior's output $h = \text{Upsample}(s^{\text{top}})$. The conditional link is implemented at each layer using a **gated activation unit**:

$$y = \tanh\Big( W_{f} * x + V_{f} * h \Big) \odot \sigma\Big( W_{g} * x + V_{g} * h \Big)$$

Where:
*   $x$ represents the bottom-prior convolutional feature map.
*   $*$ is the 2D masked convolution operator.
*   $W_f, W_g$ are trainable convolution kernels for bottom features.
*   $V_f, V_g$ are $1 \times 1$ convolutions projecting the top-level conditioning vector $h$.
*   $\odot$ is the element-wise (Hadamard) product.
*   $\sigma$ is the sigmoid gating function.

---

## 3. Original Equations Referenced in Algorithms

To make the pseudocode completely self-contained, here are the original mathematical equations referenced as **eq. 1** and **eq. 2** in the VQ-VAE-2 paper:

### Equation 1: Nearest-Neighbor Vector Quantization
Let $z_e(x)$ be the continuous output of an encoder. The discrete representation $e$ is obtained by mapping each continuous coordinate to its closest embedding vector in the codebook $C = \{e_1, \dots, e_K\}$:

$$e = \text{Quantize}\big(z_e(x)\big) = e_{k^*} \quad \text{where} \quad k^* = \arg\min_{j \in [1, K]} \|z_e(x) - e_j\|_2^2 \quad \text{(Equation 1)}$$

### Equation 2: The Three-Part VQ-VAE Loss Function
Let $E$ represent the encoder, $D$ represent the decoder, $e$ represent the quantized representation (from Equation 1), and $\text{sg}$ represent the **stop-gradient** operator. The joint autoencoder and codebook learning objective is defined as:

$$\mathcal{L}\big(x, D(e)\big) = \underbrace{\|x - D(e)\|_2^2}_{\text{Reconstruction Loss}} + \underbrace{\|\text{sg}[E(x)] - e\|_2^2}_{\text{Codebook Loss}} + \underbrace{\beta \|\text{sg}[e] - E(x)\|_2^2}_{\text{Commitment Loss}} \quad \text{(Equation 2)}$$

Where:
*   $\mathcal{L}_{\text{recon}} = \|x - D(e)\|_2^2$ trains both the encoder and decoder.
*   $\mathcal{L}_{\text{codebook}} = \|\text{sg}[E(x)] - e\|_2^2$ updates the codebook vectors toward the encoder's outputs.
*   $\mathcal{L}_{\text{commitment}} = \beta \|\text{sg}[e] - E(x)\|_2^2$ anchors the encoder's output near the codebook embeddings to stabilize training.

---

## 4. Original Algorithms (LaTeX Formatting)

Below are the exact algorithms from the paper, rendered with standard LaTeX formatting for mathematical and structural clarity.

### **Algorithm 1: VQ-VAE training (stage 1)**

$$\begin{array}{l}
\textbf{Algorithm 1: VQ-VAE training (stage 1)} \\
\hline
\textbf{Require:} \text{ Functions } E_{\text{top}}, E_{\text{bottom}}, D, \text{ batch of training images } x \\
\hline
1: \mathbf{h}_{\text{top}} \leftarrow E_{\text{top}}(x) \\
2: \mathbf{e}_{\text{top}} \leftarrow \text{Quantize}(\mathbf{h}_{\text{top}}) \quad \vartriangle \text{quantize with top codebook (eq. 1)} \\
3: \mathbf{h}_{\text{bottom}} \leftarrow E_{\text{bottom}}(x, \mathbf{e}_{\text{top}}) \\
4: \mathbf{e}_{\text{bottom}} \leftarrow \text{Quantize}(\mathbf{h}_{\text{bottom}}) \quad \vartriangle \text{quantize with bottom codebook (eq. 1)} \\
5: \hat{x} \leftarrow D(\mathbf{e}_{\text{top}}, \mathbf{e}_{\text{bottom}}) \\
6: \theta \leftarrow \text{Update}\big( \mathcal{L}(x, \hat{x}) \big) \quad \vartriangle \text{Loss according to eq. 2} \\
\hline
\end{array}$$

#### **Line-by-Line Explanation:**

*   **Header (`Require:`):**
    Establishes the prerequisites for training. It requires the Top Encoder ($E_{\text{top}}$), the Bottom Encoder ($E_{\text{bottom}}$), the joint Hierarchical Decoder ($D$), and a training mini-batch of images ($x \in \mathbb{R}^{B \times C \times H \times W}$).
*   **Line 1 ($\mathbf{h}_{\text{top}} \leftarrow E_{\text{top}}(x)$):**
    The raw pixel image batch $x$ is passed through the top encoder $E_{\text{top}}$, which downsamples the spatial dimensions heavily (e.g., $8\times$) to produce continuous latent feature maps $\mathbf{h}_{\text{top}} \in \mathbb{R}^{B \times 32 \times 32 \times D}$. These feature maps capture high-level global outlines and structural shapes.
*   **Line 2 ($\mathbf{e}_{\text{top}} \leftarrow \text{Quantize}(\mathbf{h}_{\text{top}})$):**
    Performs nearest-neighbor vector quantization on $\mathbf{h}_{\text{top}}$ using the top-level discrete codebook $C_{\text{top}}$ according to **Equation 1**, resulting in the quantized top representation $\mathbf{e}_{\text{top}} \in \mathbb{R}^{B \times 32 \times 32 \times D}$.
*   **Line 3 ($\mathbf{h}_{\text{bottom}} \leftarrow E_{\text{bottom}}(x, \mathbf{e}_{\text{top}})$):**
    Passes both the raw input image batch $x$ and the newly quantized top-level representation $\mathbf{e}_{\text{top}}$ into the bottom encoder $E_{\text{bottom}}$. $\mathbf{e}_{\text{top}}$ serves as an explicit **spatial conditioning input**. This ensures $E_{\text{bottom}}$ does not waste capacity re-learning global layout coordinates, allowing it to focus entirely on encoding fine local textures relative to the top layout.
*   **Line 4 ($\mathbf{e}_{\text{bottom}} \leftarrow \text{Quantize}(\mathbf{h}_{\text{bottom}})$):**
    Performs vector quantization on $\mathbf{h}_{\text{bottom}}$ using the bottom-level discrete codebook $C_{\text{bottom}}$ according to **Equation 1**. This maps local continuous details to the nearest bottom codebook vectors, resulting in the quantized bottom representation $\mathbf{e}_{\text{bottom}} \in \mathbb{R}^{B \times 64 \times 64 \times D}$.
*   **Line 5 ($\hat{x} \leftarrow D(\mathbf{e}_{\text{top}}, \mathbf{e}_{\text{bottom}})$):**
    Feeds both $\mathbf{e}_{\text{top}}$ and $\mathbf{e}_{\text{bottom}}$ into the unified decoder $D$. The decoder upsamples $\mathbf{e}_{\text{top}}$, channel-wise concatenates it with $\mathbf{e}_{\text{bottom}}$, and passes the joint representation through transposed convolutional blocks to reconstruct the original pixel image batch $\hat{x} \in \mathbb{R}^{B \times C \times H \times W}$.
*   **Line 6 ($\theta \leftarrow \text{Update}\big( \mathcal{L}(x, \hat{x}) \big)$):**
    Computes the total hierarchical loss function $\mathcal{L}(x, \hat{x})$ according to **Equation 2** (the sum of reconstruction errors, codebook losses, and commitment losses for both layers). It updates the model parameters $\theta$ (all weights in the encoders, decoder, and codebook vectors) using backpropagation via the Straight-Through Estimator (STE).

---

### **Algorithm 2: Prior training (stage 2)**

$$\begin{array}{l}
\textbf{Algorithm 2: Prior training (stage 2)} \\
\hline
1: T_{\text{top}}, T_{\text{bottom}} \leftarrow \emptyset \quad \vartriangle \text{training set} \\
2: \textbf{for } x \in \text{training set} \textbf{ do} \\
3: \quad \mathbf{e}_{\text{top}} \leftarrow \text{Quantize}\big( E_{\text{top}}(x) \big) \\
4: \quad \mathbf{e}_{\text{bottom}} \leftarrow \text{Quantize}\big( E_{\text{bottom}}(x, \mathbf{e}_{\text{top}}) \big) \\
5: \quad T_{\text{top}} \leftarrow T_{\text{top}} \cup \mathbf{e}_{\text{top}} \\
6: \quad T_{\text{bottom}} \leftarrow T_{\text{bottom}} \cup \mathbf{e}_{\text{bottom}} \\
7: \textbf{end for} \\
8: p_{\text{top}} = \text{TrainPixelCNN}(T_{\text{top}}) \\
9: p_{\text{bottom}} = \text{TrainCondPixelCNN}(T_{\text{bottom}}, T_{\text{top}}) \\
\textbf{Sampling procedure} \\
10: \textbf{while } \text{true} \textbf{ do} \\
11: \quad \mathbf{e}_{\text{top}} \sim p_{\text{top}} \\
12: \quad \mathbf{e}_{\text{bottom}} \sim p_{\text{bottom}}(\mathbf{e}_{\text{top}}) \\
13: \quad x \leftarrow D(\mathbf{e}_{\text{top}}, \mathbf{e}_{\text{bottom}}) \\
14: \textbf{end while} \\
\hline
\end{array}$$

#### **Line-by-Line Explanation:**

*   **Line 1 ($T_{\text{top}}, T_{\text{bottom}} \leftarrow \emptyset$):**
    Initializes two empty datasets, $T_{\text{top}}$ and $T_{\text{bottom}}$. These will store the flattened grids of discrete codebook integer indices (tokens) for our entire dataset.
*   **Line 2 ($\textbf{for } x \in \text{training set} \textbf{ do}$):**
    Starts a loop to iterate through every individual raw image $x$ in our training dataset.
*   **Line 3 & 4 ($\mathbf{e}_{\text{top}} \leftarrow \text{Quantize}\big( E_{\text{top}}(x) \big)$ and $\mathbf{e}_{\text{bottom}} \leftarrow \text{Quantize}\big( E_{\text{bottom}}(x, \mathbf{e}_{\text{top}}) \big)$):**
    Passes image $x$ through the frozen encoders to extract their discrete representations. Specifically, it extracts $\mathbf{e}_{\text{top}}$ (the $32 \times 32$ global index grid) and $\mathbf{e}_{\text{bottom}}$ (the $64 \times 64$ local index grid, conditioned on $\mathbf{e}_{\text{top}}$) according to **Equation 1**.
*   **Line 5 & 6 ($T_{\text{top}} \leftarrow T_{\text{top}} \cup \mathbf{e}_{\text{top}}$ and $T_{\text{bottom}} \leftarrow T_{\text{bottom}} \cup \mathbf{e}_{\text{bottom}}$):**
    Appends (unions $\cup$) the newly extracted index grids of image $x$ into our prior training datasets. Once this loop completes, the image dataset has been fully tokenized into integer-based coordinate grids, similar to a text corpus.
*   **Line 8 ($p_{\text{top}} = \text{TrainPixelCNN}(T_{\text{top}})$):**
    Trains an autoregressive model $p_{\text{top}}$ (such as a PixelCNN or Transformer with self-attention) on the top-token dataset $T_{\text{top}}$. It learns the joint probability of global shapes:
    $$P(s^{\text{top}}) = \prod_{i=1}^{N_{\text{top}}} P\left(s^{\text{top}}_i \;\middle|\; s^{\text{top}}_{<i}\right)$$
*   **Line 9 ($p_{\text{bottom}} = \text{TrainCondPixelCNN}(T_{\text{bottom}}, T_{\text{top}})$):**
    Trains a conditional autoregressive model $p_{\text{bottom}}$ on the bottom-token dataset $T_{\text{bottom}}$, explicitly conditioned on the top-tokens $T_{\text{top}}$. It learns how local textures should be placed relative to the global layout:
    $$P(s^{\text{bottom}} \mid s^{\text{top}}) = \prod_{j=1}^{N_{\text{bottom}}} P\left(s^{\text{bottom}}_j \;\middle|\; s^{\text{bottom}}_{<j}, \; s^{\text{top}}\right)$$
*   **Line 10 ($\textbf{while } \text{true} \textbf{ do}$):**
    Starts an infinite generation loop to draw completely new, synthetic images from scratch during inference.
*   **Line 11 ($\mathbf{e}_{\text{top}} \sim p_{\text{top}}$):**
    Samples a new global layout $\mathbf{e}_{\text{top}}$ (a $32 \times 32$ integer token grid) step-by-step from our trained global prior model $p_{\text{top}}$.
*   **Line 12 ($\mathbf{e}_{\text{bottom}} \sim p_{\text{bottom}}(\mathbf{e}_{\text{top}})$):**
    Passes the newly generated global layout $\mathbf{e}_{\text{top}}$ as a conditioning vector to our bottom prior $p_{\text{bottom}}$, and samples a $64 \times 64$ integer token grid $\mathbf{e}_{\text{bottom}}$ representing fine local details.
*   **Line 13 ($x \leftarrow D(\mathbf{e}_{\text{top}}, \mathbf{e}_{\text{bottom}})$):**
    Looks up the continuous $D$-dimensional embeddings for the generated grids $\mathbf{e}_{\text{top}}$ and $\mathbf{e}_{\text{bottom}}$ from their respective frozen codebooks, and feeds them into the frozen decoder $D$. The decoder upsamples the top embeddings, merges them with the bottom embeddings, and outputs the final synthetic image $x \in \mathbb{R}^{C \times H \times W}$ in pixel space.

---

## 5. Receptive Fields & The Start Token

Because an image is spatial, autoregressive models (like PixelCNN) flatten the 2D grid into a 1D sequence using a **Raster Scan** order. 

At the very first generation step $t=1$ (coordinate $(1,1)$), there is no prior token history. The first token is sampled as follows:

### Unconditioned Sampling
At coordinate $(1,1)$, the masked convolutions only see zero-padded pixels. The probability of the first token $s_{1,1}$ is determined entirely by the model's constant learned biases $b$:
$$\text{logits}_{1,1} = b \implies P(s_{1,1}) = \text{Softmax}(b)$$
The model draws a generic, dataset-averaged starting visual token.

### Class-Conditional Sampling
If a user specifies a class label $y$ (e.g., "sea anemone"), $y$ is projected to a class embedding vector $h_y$ and added directly inside the gating activations of all prior layers ($V \cdot h_y$). At step $t=1$:
$$\text{logits}_{1,1} = V \cdot h_y + b \implies P(s_{1,1} \mid y) = \text{Softmax}(V \cdot h_y + b)$$
The first token is immediately biased toward the requested class (e.g., sampling a blue texture for "ocean").

---

## 6. Philosophical Connection to Diffusion Models

The hierarchical generation flow of VQ-VAE-2 shares a profound conceptual and mathematical philosophy with modern **Diffusion Models**:

### Coarse-to-Fine Generation (Spectral Decomposition)
*   **VQ-VAE-2** generates the global low-frequency layout first ($z_{\text{top}}$) and then refines it with local high-frequency textures ($z_{\text{bottom}} \mid z_{\text{top}}$) using spatial hierarchies.
*   **Diffusion Models** generate low-frequency shapes during the early denoising steps ($t = T \to T/2$) and paint in the fine-grained high-frequency textures in the final steps ($t = T/2 \to 0$) using temporal hierarchies.

### Progressive Conditioning Chain
Both models write their data generation probability as a chained conditional dependency:
$$\text{VQ-VAE-2:} \quad P(x) = \sum_{z} P(x \mid z_{\text{bottom}}, z_{\text{top}}) P(z_{\text{bottom}} \mid z_{\text{top}}) P(z_{\text{top}})$$
$$\text{Diffusion:} \quad P(x_0) = \int P(x_0 \mid x_1) P(x_1 \mid x_2) \dots P(x_{T-1} \mid x_T) P(x_T) dx_{1:T}$$

This core realization—that spatial hierarchies are mathematically equivalent to temporal denoising paths—is what eventually led to the development of **Latent Diffusion Models (Stable Diffusion)**, which run a temporal diffusion process directly inside a frozen autoencoder's spatial latent space.
