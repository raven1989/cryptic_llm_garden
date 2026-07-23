---
tags:
  - time-series
  - forecasting
  - architecture
  - vector-quantization
  - llm
date: 2026-07-22
sources: ["[[wiki/research/From Values to Tokens Summary.md]]"]
---

# TokenCast (LLM-Driven Time Series Forecasting Framework)

**TokenCast** is a large language model (LLM) driven framework for context-aware time series forecasting (TSF). It uses language-based symbolic representations as a unified intermediary to align continuous numerical time series sequences with unstructured textual contextual features (such as clinical notes, financial reports, or logs).

Below is the complete architectural overview of TokenCast, tracing the pipeline from discretization to vocabulary-level cross-modality alignment and generative forecasting:

![TokenCast Architecture Overview](../media/overview_context-aware_time_series_forecasting.png)

---

## 1. Core Architectural Pillars

1. **Symbolic Discretization:** Converting continuous historical/predicted numerical sequences into symbolic indices (tokens) via a decoupled vector-quantized autoencoder.
2. **Cross-Modality Vocabulary-Level Alignment:** Merging temporal and contextual tokens into a unified vocabulary and semantic space via unsupervised next-token prediction pre-training.
3. **Generative Fine-Tuning:** Adapting the aligned LLM through structured prompt tuning to autoregressively predict future temporal tokens and reasoning outputs, which are then decoded back to continuous predictions.

---

## 2. Preventing Lookahead Leakage: Decoupled RIN

### How Standard RevIN works (Dimensionality & Parameters)
In standard forecasting literature, **Reversible Instance Normalization (RevIN)** is used to handle non-stationary time series.
* **Temporal Calculation ($L$):** Given a multivariate time-series tensor of shape $[B, L, C]$, RevIN computes normalization statistics strictly along the **temporal sequence length dimension ($L$)**, independently for each batch item $b \in \{1, \dots, B\}$ and each multi-variate channel $c \in \{1, \dots, C\}$:
  $$\mu_{b,c} = \frac{1}{L} \sum_{t=1}^{L} X_{b,t,c}, \quad \sigma_{b,c} = \sqrt{\frac{1}{L} \sum_{t=1}^{L} (X_{b,t,c} - \mu_{b,c})^2 + \epsilon}$$
  *There is no mixing of channels/variables.* For example, temperature and wind speed are normalized entirely independently.
* **No Permanent Statistics:** Unlike Batch Normalization which must keep a running average of means and variances to perform inference on a batch size of 1, RevIN holds **no permanent mean or standard deviation weights** in the model checkpoint. Because its calculations are temporal, all the information needed to normalize a sequence is contained within its own lookup window. Even at inference on a single sequence ($B=1$), the full sequence length $L$ is present, allowing RevIN to calculate the statistics **on-the-fly and discard them** after the forward pass.
* **Learnable Affine Parameters:** The only parameters RevIN permanently stores are its learnable affine weights: Scale ($\gamma \in \mathbb{R}^C$) and Shift ($\beta \in \mathbb{R}^C$), with one value per channel.

### The Decoupled RIN Innovation
Because standard RevIN normalizes over the entire input-output horizon, using it in an autoregressive predictive framework would cause **future lookahead information leakage** (since future target properties would help scale historical inputs). 

To prevent this, TokenCast implements **Decoupled RIN**:
1. It computes the mean $\mu(H)$ and standard deviation $\sigma(H)$ **solely** on the historical sequence $H \in \mathbb{R}^{L_H \times C}$:
   $$\mu(H) = \frac{1}{L_H} \sum_{t=1}^{L_H} H_t, \quad \sigma(H) = \sqrt{\frac{1}{L_H} \sum_{t=1}^{L_H} (H_t - \mu(H))^2 + \epsilon}$$
2. The entire series $X = [H; P] \in \mathbb{R}^{(L_H + L_P) \times C}$ (where $P$ is the future sequence) is normalized using these historical statistics:
   $$X_{\text{norm}} = \frac{X - \mu(H)}{\sigma(H)}$$
This guarantees that no target-horizon statistical properties leak into the encoding of historical variables, preserving strict temporal causality.

---

## 3. Discretization: Causal Encoding and Lookahead Training (Section 3.3)

### Concatenated Sequence Processing
During discretization, TokenCast does **not** process historical ($H$) and predicted ($P$) sequences separately. They are processed together as a concatenated tensor $X = [H; P]$ through a **shared causal encoder** ($f_{\text{enc}}$) and a **shared causal decoder** ($f_{\text{dec}}$).

### Why include the Predicted Series ($P$) in Tokenizer Training?
At Stage 1 (Tokenizer training), the system acts as an offline, self-supervised **Vector Quantized Autoencoder (VQ-VAE)**. Since we already have the ground-truth sequences ($H$ and $P$) from the training set, we pass both into the causal autoencoder:
- **Codebook Coverage:** It ensures the codebook $C_i = \{e_{i,k}\}_{k=1}^K \subset \mathbb{R}^d$ learns representations for both historical patterns and future trend profiles.
- **De-tokenizer Optimization:** It trains the causal decoder (de-tokenizer) to accurately reconstruct future segments from discrete indices. This is critical because during forecasting (Stage 3), the frozen decoder must translate predicted tokens back into continuous numerical predictions.
- **Strict Causality:** Making the autoencoder layers strictly causal forces the network to learn how to transition naturally from historical states into future target states without looking ahead.

### Multi-Task Loss and Codebook Diversity
$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta\left(\mathcal{L}_{\text{commit}} + \mathcal{L}_{\text{codebook}}\right) + \gamma\mathcal{L}_{\text{diversity}}$$
To prevent **codebook collapse** (where the encoder maps continuous representations to only a few codebook indices, leaving the rest underutilized), TokenCast adds a **Diversity Loss**:
$$\mathcal{L}_{\text{diversity}} = \frac{1}{K} \sum_{k=1}^K \frac{1}{d_k + \epsilon}, \quad \text{where } d_k = \min_{j 
eq k} \|e_k - e_j\|_2$$
This pushes codebook vector embeddings apart, encouraging uniform usage of the representation space.

---

## 4. Vocabulary-Level Semantic Alignment (Section 3.5)

To bridge the multi-modal semantic gap, TokenCast expands the language model's native vocabulary matrix instead of using linear projections.

### Extended Vocabulary Matrix
The pre-trained LLM's original vocabulary $V_{\text{orig}}$ is expanded to $V$ by appending $K$ temporal tokens and $S$ task-specific special tokens:
$$V = V_{\text{orig}} \cup \{ \text{temp}_1, \dots, \text{temp}_K \} \cup \{ \text{spec}_1, \dots, \text{spec}_S \}$$

The **$S$ special tokens** act as formatting, structural, and behavioral markers:
1. **Sequence Partitions (Boundary Markers):** Triggers like `<start_ts>` and `<end_ts>` to partition natural language reasoning text from predicted temporal tokens.
2. **Domain Indicators:** Switching attention and context dynamically for multi-task evaluation (e.g., `<domain_health>`, `<domain_stock>`).
3. **Statistical Boundaries:** Demarcating numerical statistics (mean, std, trends) inside the structured prompt template.
4. **Padding / Mask Flags:** Handling missing inputs or sequence alignment without confusing standard English vocabulary words.

### Distributional Embedding Initialization
Initializing the newly added embeddings randomly disrupts the pre-trained geometric semantic structure of the word embeddings. To maintain representation alignment, TokenCast initializes the temporal and special token embeddings by sampling from a multivariate Gaussian distribution derived from the mean $\mu_E$ and covariance matrix $\Sigma_E$ of the original pre-trained embedding matrix $E_{\text{orig}}$:
$$E_{\text{new}} \sim \mathcal{N}(\mu_E, \Sigma_E)$$

### Unsupervised Autoregressive Alignment Pre-training
During the alignment phase, all LLM self-attention and feed-forward weights are **frozen**, and only the unified embedding matrix $E \in \mathbb{R}^{|V| \times d_{\text{llm}}}$ is trained using a next-token prediction language modeling objective:
$$\mathcal{L}_{\text{align}} = -\sum_{t=1}^{T} \log p(z_t \mid z_1, \dots, z_{t-1}; E)$$
This aligns the continuous-space representations of the new temporal tokens with the pre-trained token embeddings.

---

## 5. Structured Prompt-Based Generative SFT (Section 3.6)
During fine-tuning, the LLM is optimized to generate a structured response containing natural language reasoning followed by future temporal tokens. At inference:
1. The model receives a structured prompt containing instructions, history statistics, history tokens, and context.
2. It autoregressively generates a complete text response.
3. The predicted temporal tokens are extracted from inside the boundary markers.
4. The frozen causal de-tokenizer maps the indices back to continuous vectors: $\hat{X}_{\text{norm}} = f_{\text{dec}}(Z_{\text{pred}})$.
5. Decoupled RIN rescales and centers the predictions to form the final continuous physical forecast:
   $$\hat{P} = \hat{X}_{\text{norm}} \cdot \sigma(H) + \mu(H)$$
