---
tags:
  - llm
  - architecture
  - attention
date: 2026-04-17
sources: ["[[Why is Attention divided by Root d_k]]", "[[You could have designed state of the art positional encoding]]"]
---

# Positional Encoding

Because [[Transformers]] lack recurrence (like RNNs) and convolutions (like CNNs), they process all tokens in a sequence simultaneously. The [[Self-Attention Mechanism]] is a **set operation**, meaning it is *permutation equivariant*. Without positional information, the attention score between the word "dog" in "The dog chased another dog" would be identical for both occurrences, leaving the model incapable of understanding word order.

Positional Encoding techniques are used to explicitly inject sequence order into the model. 

## The Evolution of Positional Encodings

Finding the optimal way to represent position involved optimizing for properties like unique encoding across sequence lengths, simple linear relationships between positions, continuous/smooth math (for gradients), and the ability to generalize to sequence lengths beyond the training distribution.

### Early Flawed Approaches
1. **Integer Encoding:** Directly adding the position index ($0 \rightarrow L$) to the token embedding. 
   ![[raw/LLM/IntegerEncoding.mp4]]
   - *Problem:* The position values vastly exceed the input embedding values (which cluster near 0), destroying the signal-to-noise ratio and causing exploding/vanishing gradients.
2. **Binary Encoding:** Stretches the binary representation of the position across the embedding dimension.
   ![[raw/LLM/BinaryEncoding.mp4]]
   ![[raw/LLM/BinaryVector3D.mp4]]
   - *Problem:* The values jump discretely between 0 and 1, making optimization difficult (gradient descent requires smooth, continuous functions).

### Sinusoidal Positional Encoding (Absolute Additive)
Introduced in the original "Attention Is All You Need" paper. It creates a positional vector where each component $i$ is alternatively drawn from $sin$ and $cos$ waves with gradually increasing base wavelengths ($\theta = 10,000$).
![[raw/LLM/SteppedPositionalEncodingPlot.mp4]]

**The Equations:**
$$ PE_{(pos, 2i)} = \sin \left( \frac{pos}{10000^{2i/d}} \right) $$
$$ PE_{(pos, 2i+1)} = \cos \left( \frac{pos}{10000^{2i/d}} \right) $$
Where $pos$ is the token's position index, $i$ is the component index, and $d$ is the model dimension.

- **Mechanism:** The generated positional vector is **added** directly to the token embedding before any $Q, K, V$ projections.
- **The "Graceful Math" of Linear Translation:** The authors cleverly used both $sin$ and $cos$ to allow the model to easily learn relative positions. To prove this, we want to find a linear transformation matrix $\mathbf{M}$ that can shift a sinusoidal function at position $p$ by a fixed offset $k$:
  
  $$ \begin{bmatrix} u_1 & v_1 \\ u_2 & v_2 \end{bmatrix} \begin{bmatrix} \sin(\omega_i p) \\ \cos(\omega_i p) \end{bmatrix} = \begin{bmatrix} \sin(\omega_i (p + k)) \\ \cos(\omega_i (p + k)) \end{bmatrix} $$
  
  By applying the trigonometric addition theorem ($\sin(a+b) = \sin a \cos b + \cos a \sin b$ and $\cos(a+b) = \cos a \cos b - \sin a \sin b$) to the right-hand side, we expand it into:
  
  $$ \begin{bmatrix} u_1 \sin(\omega_i p) + v_1 \cos(\omega_i p) \\ u_2 \sin(\omega_i p) + v_2 \cos(\omega_i p) \end{bmatrix} = \begin{bmatrix} \sin(\omega_i p)\cos(\omega_i k) + \cos(\omega_i p)\sin(\omega_i k) \\ \cos(\omega_i p)\cos(\omega_i k) - \sin(\omega_i p)\sin(\omega_i k) \end{bmatrix} $$
  
  By matching the coefficients for the $\sin(\omega_i p)$ and $\cos(\omega_i p)$ terms on both sides, we solve for the unknowns:
  
  $$ u_1 = \cos(\omega_i k), \quad v_1 = \sin(\omega_i k) $$
  $$ u_2 = -\sin(\omega_i k), \quad v_2 = \cos(\omega_i k) $$
  
  This yields our final transformation matrix $\mathbf{M_k}$:
  
  $$ \mathbf{M_k} = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix} $$
  
  *This brilliant derivation proves that Sinusoidal Encoding was already encoding relative position via the Rotation Matrix in 2017, laying the groundwork for RoPE years later.*
- **Limitations:** Adding the position vector directly to the token embedding pollutes the semantic meaning of the token (modifying the vector's norm). Furthermore, it relies heavily on absolute positions rather than directly targeting relative relations during attention.

### Learnable Positional Encoding (Absolute Additive)
Rather than forcing a rigid mathematical structure (like Sinusoidal), early pre-trained models like **GPT** and **BERT** opted for a purely data-driven approach: letting the model learn the optimal positional representations during training.

- **Mechanism:** A positional embedding matrix $\mathbf{P} \in \mathbb{R}^{L_{\text{max}} \times d_{\text{model}}}$ is instantiated, where $L_{\text{max}}$ is the predetermined maximum sequence length. The positional encoding for position $pos$ is simply the $pos$-th row extracted from this matrix and added to the token embedding.
- **Advantage:** Maximum flexibility. The model learns representations perfectly suited to its training data, resulting in smooth transitions between adjacent positions.
- **Critical Limitation (Inability to Extrapolate):** Because an explicit, fixed matrix size bounds the model, it is fundamentally impossible for a Learnable Positional Encoding to evaluate sequence lengths longer than $L_{\text{max}}$. If a sequence length of 513 is passed to a BERT model trained on 512, the positional vector for index 513 simply does not exist. This rigid ceiling catalyzed the industry shift toward relative encoding schemes.

## Rotary Position Embedding (RoPE) (Relative Multiplicative)

RoPE (often used in modern models like LLaMA) treats positional encoding as a relative mechanism and applies it where it matters most: right before the dot product of Query and Key vectors. Building directly off the mathematical properties discovered in Sinusoidal Encodings, RoPE realizes that the *relative shift* derived via rotation matrices can be applied directly to the $Q$ and $K$ vector components before self-attention.

*(For an in-depth mathematical derivation of RoPE, including its block-diagonal matrices, Abel transformation for long-range decay, length extrapolation proofs, and PyTorch implementations, see the dedicated **[[RoPE]]** page).*

![[raw/LLM/RopeEncoding.mp4]]

- **Mechanism:** Instead of adding a vector to the base embedding, RoPE decomposes $\mathbf{q}$ and $\mathbf{k}$ vectors into 2D component pairs. It then **multiplies** each pair by the corresponding Rotation Matrix ($\mathbf{M_i}$). 
$$ R(\mathbf{q}, p) = \begin{bmatrix} \mathbf{M_1} \\ \mathbf{M_2} \\ \vdots \\ \mathbf{M_{d/2}} \end{bmatrix} \begin{bmatrix} q_1 \\ q_2 \\ \vdots \\ q_d \end{bmatrix} $$
Where $\mathbf{M_i}$ is the rotation matrix for position $p$:
$$ \mathbf{M_i} = \begin{bmatrix} \cos(\omega_i p) & \sin(\omega_i p) \\ -\sin(\omega_i p) & \cos(\omega_i p) \end{bmatrix} $$
- **Geometric Intuition:** The dot product of two vectors is $\vec{a} \cdot \vec{b} = |\vec{a}| |\vec{b}| \cos(\theta)$. By rotating a vector, we change its angle but *never change its magnitude (norm)*. 
- **Result:** By applying independent rotation matrices to $Q$ and $K$ components based on their absolute positions, their resulting dot product (attention score) inherently encodes their **relative distance**. The model determines relationships purely via the angle difference (distance) without ever polluting the base semantic information represented by the vector's norm.
- **Advantage:** Highly robust for length extrapolation, allowing models to process context windows much longer than they were trained on. Because each pair is rotated independently *within* the same dimension, RoPE natively and elegantly scales to n-Dimensional data (like 2D images). 

## ALiBi (Attention with Linear Biases)

An alternative to embeddings entirely. 
- **Mechanism:** ALiBi does not add or multiply vectors into the input embeddings. Instead, it applies a **linear decay negative penalty** directly to the final Attention Scores prior to the Softmax operation. 
- **Result:** The penalty is strictly based on the distance between the two tokens (the further apart the tokens are, the larger the negative penalty). This simple heuristic also provides exceptionally strong length extrapolation. See the full mathematical definition and bias matrix visualization at [[ALiBi]].

### Other Relative Position Bias Schemes

**T5's Relative Position Bias:**
A predecessor to ALiBi, T5 utilizes a parameterized, learnable scalar bias added to the attention score corresponding to the relative distance between tokens. To bound the sequence length during extrapolation, T5 groups larger relative distances into uniform "buckets" and shares the same learnable scalar value for any distance past a threshold.

**Transformer-XL:**
Explicitly incorporates relative positional embeddings by decomposing the attention score into four distinct terms (content-content, content-position, position-content, and position-position interactions). Highly effective, but structurally complex.

## The Design Trend of Positional Encodings

Looking at the evolution from standard Sinusoidal arrays to ALiBi and RoPE, 4 structural philosophies dominate modern LLM architecture:
1. **Absolute $\rightarrow$ Relative:** Modern models care about *how far apart* "dog" and "bite" are in a sentence, not that they appear explicitly at indices 401 and 402. Relative distances generalize dramatically better.
2. **Embedding Layer $\rightarrow$ Attention Layer:** Encoding position dynamically directly into the $Q/K$ computation cleanly separates semantic meaning (Value vectors) from positional routing.
3. **Fixed Length $\rightarrow$ Extrapolatable:** Context windows have exploded from BERT's 512 limits to modern 128k+. Encoding schemes *must* support infinite length mathematically.
4. **Complex $\rightarrow$ Simple:** From 4-term decompositions to simple block diagonal matrices (RoPE) and linear scalar matrices (ALiBi).

### Comparison Matrix

| Scheme | Type | Injection Point | Extrapolation | Additional Params | Key Models |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sinusoidal** | Absolute | Embedding Addition | Theoretical | 0 | Vanilla Transformer |
| **Learnable** | Absolute | Embedding Addition | None | $L_{\text{max}} \times d$ | GPT-2, BERT |
| **RoPE** | Relative | Q/K Rotation | Strong (w/ Scaling) | 0 | LLaMA, DeepSeek |
| **ALiBi** | Relative | Attention Bias | Very Strong | 0 | BLOOM |
| **T5 Bias** | Relative | Attention Bias | Moderate (Bucketed)| Small amount | T5 |
