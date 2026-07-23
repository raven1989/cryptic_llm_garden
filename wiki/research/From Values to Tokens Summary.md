---
tags: ["time-series", "forecasting", "tokenization", "llm", "discretization", "vector-quantization"]
aliases: ["TokenCast", "TokenCast Summary", "From Values to Tokens"]
date: 2026-07-22
sources: ["[[raw/time_series_forecast/From Values to Tokens: An LLM-Driven Framework for Context-Aware Time Series Forecasting via Symbolic Discretization.md]]"]
---

# From Values to Tokens: TokenCast Summary

## 1. Core Problem & Motivations (Section 1)
Traditional and deep learning-based Time Series Forecasting (TSF) methods (such as ARIMA, RNNs, CNNs, Transformers, and MLPs) primarily focus on modeling homogeneous numerical sequences. However, real-world forecasting tasks often require understanding how these numerical patterns interact with high-value, unstructured **contextual features** (e.g., clinical notes, financial reports, or user logs) that are expressed in textual form.

To integrate these multi-modal signals, recent TSF methodologies rely on two main paradigms:
- **Linear adapters** that project continuous temporal features into the language model's latent embedding space.
- **Soft prompts** to guide a frozen LLM's behavioral response.

### Limitations of Prior Work
Section 1 identifies that these promising adapter-based approaches are essentially "shallow fusion" strategies. They fail to resolve the structural discrepancies between continuous numerical sequences and discrete textual features, and they do not fully exploit the pre-trained generative and reasoning capabilities of LLMs. 

This motivates a fundamental question: **Can time series be effectively modeled in a discrete token space to unlock the potential of LLMs?**

To address this, [[TokenCast]] proposes a more expressive paradigm: formulating context-aware TSF as a **multimodal discrete context understanding and generation task** powered by pre-trained LLMs. This is achieved by mapping continuous numerical sequences into discrete symbolic representations that align structurally and semantically with language tokens.

---

## 2. [[TokenCast]] Framework Overview
[[TokenCast]] consists of three major pipeline stages:
1. **[[Symbolic Discretization|Time Series Discretization]]:** Transforming continuous historical and predicted sequences into discrete temporal tokens while preserving local statistical attributes and avoiding lookahead leakage.
2. **Cross-Modality Vocabulary-Level Alignment:** Mapping temporal and contextual tokens into a unified vocabulary space to bridge the semantic gap via unsupervised next-token prediction pre-training.
3. **Generative Fine-Tuning & Decoding:** Adapting the aligned LLM through structured prompt tuning to autoregressively predict future temporal tokens and reasoning outputs, which are then decoded back to continuous predictions.

Below is the complete framework architecture of TokenCast, illustrating the pipeline from discretization to cross-modality alignment and generative forecasting:

![TokenCast Architecture Overview](../media/overview_context-aware_time_series_forecasting.png)

---

## 3. Detailed Methodology (Section 3)

### 3.1. Problem Formulation (Section 3.1 & 3.2)
Given a dataset $\mathcal{D}=\{(H_{i},T_{i},P_{i})\}_{i=1}^{N}$ of $N$ multimodal instances:
- $H \in \mathbb{R}^{L_H \times C}$ denotes the historical multivariate time series of length $L_H$.
- $T$ denotes the unstructured textual contextual features, which are tokenized by the native LLM tokenizer into text tokens $Y$.
- $P \in \mathbb{R}^{L_P \times C}$ is the future target time series of horizon $L_P$.

The historical time series $H$ is discretized into discrete temporal tokens $Z_q$ through a learnable tokenizer $f_\theta$. The input sequence is formed by concatenating these temporal tokens with text tokens: $Z = [Z_q; Y]$. The aligned LLM autoregressively predicts future tokens $\hat{Z}$ (demarcated by boundary markers), which are then translated back into continuous values $\hat{P}$ using a frozen de-tokenizer $g_\phi : \hat{Z} \mapsto \hat{P}$.

---

### 3.2. [[Symbolic Discretization|Time Series Discretization]] (Section 3.3)
To establish a symbolic bridge, TokenCast employs a decoupled and dynamic tokenizer.

#### Unified Processing of $H$ and $P$ (Concatenated Sequence)
The author does **not** process the historical ($H$) and predicted ($P$) sequences separately during discretization. Instead, they are processed as a **unified, concatenated sequence** $X = [H; P] \in \mathbb{R}^{(L_H + L_P) \times C}$ through a **shared causal encoder** ($f_{\text{enc}}$) and a **shared causal decoder** ($f_{\text{dec}}$). 

Using a shared causal autoencoder guarantees consistent reconstruction quality across the boundary and ensures the predicted part can dynamically exploit richer representations from the historical features.

#### Why is the "Predicted" Series ($P$) Included in Reconstruction?
During Stage 1 (Tokenizer Training), the system operates as an offline, self-supervised **Vector Quantized Autoencoder (VQ-VAE)**:
- We already have the full ground-truth sequences ($H$ and $P$) from the training set.
- Reconstructing $P$ is necessary to **train the universal de-tokenizer (decoder)** to map predicted temporal tokens back into continuous numerical predictions during Stage 3.
- It ensures the codebook learns symbolic shapes representing both historical patterns and future trends.
- The encoder and decoder are made strictly **causal** (no step $t$ can look at step $t+1$), which forces the network to learn transition dynamics without future lookahead cheating.

#### Decoupled Reversible Instance Normalization (Decoupled RIN)
To handle non-stationarity, standard RevIN is typically used. However, computing statistics over the entire sequence $X$ would cause **future lookahead information leakage** in a predictive task. To resolve this, TokenCast implements **Decoupled RIN**:
1. Compute the mean $\mu(H)$ and standard deviation $\sigma(H)$ **solely** on the historical sequence $H$:
   $$\mu(H) = \frac{1}{L_H} \sum_{t=1}^{L_H} H_t, \quad \sigma(H) = \sqrt{\frac{1}{L_H} \sum_{t=1}^{L_H} (H_t - \mu(H))^2 + \epsilon}$$
2. Apply these history-only statistics to normalize the entire concatenated sequence $X = [H; P]$:
   $$X_{\text{norm}} = \frac{X - \mu(H)}{\sigma(H)}$$
This caches the historical statistics for inverse transformation and ensures no target-horizon statistical characteristics leak into the continuous representation $Z = f_{\text{enc}}(X_{\text{norm}})$.

#### Learnable Codebook and Vector Quantization
A domain-specific learnable codebook $C_i = \{e_{i,k}\}_{k=1}^K \subset \mathbb{R}^d$ is maintained. The continuous latent representations $z_t \in \mathbb{R}^d$ are quantized to the nearest codebook vector:
$$z_t^q = e_{i, k^*}, \quad \text{where } k^* = \arg\min_{k} \|z_t - e_{i,k}\|_2^2$$
The indices $\{k^*\}$ serve as the discrete temporal tokens $Z_q$.

#### Multi-Task Tokenizer Loss
$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta\left(\mathcal{L}_{\text{commit}} + \mathcal{L}_{\text{codebook}}\right) + \gamma\mathcal{L}_{\text{diversity}}$$
- **Reconstruction Loss:** $\mathcal{L}_{\text{recon}} = \|\hat{X} - X\|_2^2$ (computed using Straight-Through Estimators (STE) to bypass discrete gradients).
- **Codebook & Commitment Loss:** $\mathcal{L}_{\text{codebook}} = \|\text{sg}[Z] - Z_q\|_2^2$ and $\mathcal{L}_{\text{commit}} = \|Z - \text{sg}[Z_q]\|_2^2$.
- **Diversity Loss:** To prevent codebook collapse, a nearest-neighbor distance separation penalty is added:
  $$\mathcal{L}_{\text{diversity}} = \frac{1}{K} \sum_{k=1}^K \frac{1}{d_k + \epsilon}, \quad \text{where } d_k = \min_{j 
eq k} \|e_k - e_j\|_2$$

---

### 3.3. Cross-Modality Vocabulary-Level Alignment (Section 3.5)
Instead of using external projection layers (which bypass the LLM's native vocabulary mechanics), TokenCast implements a direct vocabulary-level alignment strategy.

#### Extended Vocabulary & Special Tokens ($S$)
The LLM's original vocabulary $V_{\text{orig}}$ is expanded to $V$ by appending the $K$ temporal tokens and $S$ task-specific special tokens:
$$V = V_{\text{orig}} \cup \{ \text{temp}_1, \dots, \text{temp}_K \} \cup \{ \text{spec}_1, \dots, \text{spec}_S \}$$

The **$S$ special tokens** are necessary formatting and behavioral anchors, representing:
1. **Sequence Partitions (Boundary Markers):** Triggers like `<start_ts>` and `<end_ts>` to partition reasoning text from predicted temporal tokens.
2. **Domain Indicators:** Switching context dynamically for multi-task evaluation (e.g., `<domain_health>`, `<domain_stock>`).
3. **Statistical Boundaries:** Separating raw textual descriptions from normalized numerical parameters (mean/std) in structured prompts.
4. **Padding/Mask Flags:** Representing missing data or aligning sequences cleanly without reusing natural language words.

#### Distributional Embedding Initialization
To prevent representation space disruption, the embeddings for the newly added tokens are initialized by sampling from a multivariate Gaussian distribution fitted over the original pre-trained embeddings:
$$E_{\text{new}} \sim \mathcal{N}(\mu_E, \Sigma_E)$$

#### Unsupervised Embedding Alignment
The core LLM parameters are frozen, and only the unified embedding matrix $E \in \mathbb{R}^{|V| \times d_{\text{llm}}}$ is trained under an autoregressive next-token prediction objective:
$$\mathcal{L}_{\text{align}} = -\sum_{t=1}^{T} \log p(z_t \mid z_1, \dots, z_{t-1}; E)$$

---

### 3.4. Generative Fine-Tuning and Decoding (Section 3.6)
The aligned model is adapted for forecasting using Supervised Fine-Tuning (SFT) over structured prompts containing domain instructions, historical statistics, and concatenated tokens:
$$\mathcal{L}_{\text{sft}} = -\sum_t \log p(w_t \mid w_1, \dots, w_{t-1})$$
During inference, the LLM autoregressively generates text reasoning and predicted discrete tokens. These tokens are extracted from inside the boundary markers and decoded:
$$\hat{P} = f_{\text{denorm}}(f_{\text{dec}}(Z_{\text{pred}})) = f_{\text{dec}}(Z_{\text{pred}}) \cdot \sigma(H) + \mu(H)$$

---

## 4. Key Takeaways & Performance

| Dataset / Domain | **TokenCast (MSE / MAE)** | Best Baseline |
| :--- | :---: | :---: |
| **Economic** | **68.911 / 1.701** | 81.542 / 1.672 (Time-LLM / SimMTM) |
| **Health** | **2.525 / 0.081** | 2.565 / 0.083 (GPT4TS) |
| **Web** | **497.410 / 1.246** | 540.492 / 1.327 (GPT4TS / SimMTM) |
| **Stock-NY** | **0.482 / 0.455** | 0.613 / 0.502 (SimMTM / GPT4TS) |
| **Stock-NA** | **1.134 / 0.780** | 1.200 / 0.834 (Time-LLM / SimMTM) |

[[TokenCast]] ranks first across 5 out of 6 context-rich benchmarks, validating that discrete symbolic tokenization combined with vocabulary-level alignment is superior to shallow projection adapters.
