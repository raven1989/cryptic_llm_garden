# Wiki Index

This is the content-oriented catalog of the wiki. The LLM updates this file whenever a new source is ingested or a new page is created.

## Personal Knowledge

- [[LLM Study Plan]]: A comprehensive, top-down algorithm engineering guide spanning Applications, Frameworks, and Foundation Models.
- [[My LLM Learning Sequence]]: A step-by-step, bottom-up execution plan derived from the main study outline.
- [[Layer 3 - Architectures and Operators Syllabus]]: A deep-dive syllabus for Step 1 of the learning sequence, focusing on macro-architectures, micro-designs, and inference bottlenecks.

## Research & Topics
- [[Chronos-2 Summary]]: Amazon AWS zero-shot time series foundation model shifting to an encoder-only architecture with continuous 21-quantile regression and Group Attention.
- [[CITRAS Summary]]: Hitachi decoder-only transformer with 2D alternating temporal-spatial attention, KV Shift, and Attention Score Smoothing.
- [[In-Context Learning and Future Covariates in Decoder-Only Forecasting]]: Technical synthesis comparing the covariate-integration strategies of Chronos-2, CITRAS, and TimesFM 2.5 (XReg) under autoregressive constraints.

- [[RankMixer Summary]]: ByteDance's RankMixer architecture boosting ranking model MFU from 4.5% to 45% and scaling parameters to 1B under strict SLA constraints.
- [[Douyin STCA Summary]]: ByteDance's E2E long-sequence recommendation system scaling behavior sequences to 10k+ under strict latency and training budgets using stacked target cross-attention and Request Level Batching.
- [[HSTU]]: Meta's complete Generative Recommenders summary, sequential transduction task formulation (Table 1), target-masking details, and the custom HSTU encoder architecture.
- [[Mixture of Experts Summary]]: The ultimate, high-fidelity research guide charting the chronological evolution and critical trade-offs of Mixture of Experts (MoE) LLMs, linking seminal architectures from 2017 to DeepSeek-V3.
- [[Speculative Decoding]]: An analysis of speculative decoding paradigms (classic draft-target, feature-extrapolating EAGLE-3, dynamic draft trees, and MTP) exploring memory-bandwidth bottlenecks and speculative sampling mathematical guarantees.
- [[ST-MoE Summary]]: A summary of the ST-MoE paper detailing how the Router Z-Loss prevents training instability in massive sparse models caused by bfloat16 exponential roundoff errors.
- [[Grouped Query Attention Summary]]: A summary of the IBM article detailing how Grouped Query Attention and Multi-Query Attention optimize the standard MHA mechanism.
- [[DeepSeek-V3.2 Training Pipeline]]: A comprehensive breakdown of DeepSeek-V3.2's training protocol, including DSA pre-training integration, GRPO scaling stability improvements (unbiased KL estimate, off-policy sequence masking), and agentic synthetic data generation.
- [[大模型位置编码-ALiBi位置编码 Summary]]: A summary of Attention with Linear Biases (ALiBi), a positional encoding technique for improving length extrapolation in Large Language Models.
- [[Encoder Only, Encoder Decoder, And Decoder Only Models]]: An overview summarizing the three primary macro-architectures in transformer-based NLP and their distinct use cases.
- [[LLM Inference VRAM and Compute Estimation]]: Mathematical formulas for calculating the precise memory footprint of the KV Cache and the computational FLOPs required during the Prefill and Decode stages.
- [[十分钟看懂RoPE Summary]]: A deep dive article detailing the mathematical proofs, long-range decay (Abel transformation), extrapolation properties, and PyTorch production code (LLaMA) for Rotary Position Embedding.
- [[Why is Attention divided by Root d_k]]: A mathematical breakdown of why the dot product of Query and Key vectors in Transformers is scaled by $\sqrt{d_k}$ to prevent variance growth and vanishing gradients.
- [[Web Agent Architecture]]: An analysis of 4 different architectural paradigms for building AI web agents.
- [[Taxy.ai Implementation]]: Code-level analysis of how Taxy.ai extracts DOM state and executes hardware-level actions.
- [[Browser-Use Implementation]]: Code-level analysis of how Browser-Use utilizes Pure Python CDP (Chrome DevTools Protocol) to execute stealthy web agents without JavaScript injection.
- [[Action Execution - JS vs CDP]]: A detailed breakdown of why modern web agents must use the Chrome DevTools Protocol to simulate hardware interrupts rather than executing simple JavaScript.
- [[Beam Search Summary]]: A comprehensive summary of Beam Search decoding mechanics, log-probability scoring, and early stopping behavior.
- [[OneRec Summary]]: A comprehensive, unified generative recommender from Kuaishou Inc. replacing the traditional multi-stage retrieval and ranking cascade.
- [[OneRec-V2 Summary]]: Detailed breakdown of OneRec-V2's Lazy Decoder-Only architecture, context processor optimization, and scaling laws.
- [[Codebook Technology History Summary]]: The master, high-fidelity research guide charting the chronological evolution and critical trade-offs of codebook discrete representation learning, from continuous VAE to FSQ.
- [[From Values to Tokens Summary]]: An LLM-driven framework using language-based symbolic representation as a unified intermediary for context-aware time series forecasting.
- [[COMET Summary]]: SOTA codebook-based online-adaptive multi-scale embedding framework for multivariate time-series anomaly detection.

## Media Companion (Books, Movies, etc.)

- (Empty)

## Entities & Concepts (Cross-cutting)
- [[Attention Score Smoothing]]: An attention regularization technique applying EMA across temporal steps to stabilize cross-variate dependencies and filter high-frequency noise.
- [[Group Attention]]: A masked self-attention pooling mechanism that dynamically isolates token aggregation within designated task groups.
- [[KV Shift]]: A step-wise cross-variate attention design that aligns Query and Key at step $i$ but shifts the known covariate Value to step $i+1$ to retrieve future context without data leakage.
- [[TimesFM XReg Modes]]: Dual decoupled linear-residual modeling integration strategies (Residual and Error Correction) used to incorporate external regressors in Google TimesFM 2.5.

- [[Agents]]: Workflows, State Machines, multi-agent systems, and tool calling sandboxes.
- [[ALiBi]]: Attention with Linear Biases, a positional encoding method that applies distance penalties directly to attention scores to enable length extrapolation.
- [[Balanced K-means]]: An optimized clustering algorithm with strict cluster-capacity constraints used to generate balanced semantic IDs.
- [[Beam Search]]: A heuristic search decoding algorithm that balances the speed of greedy search with the optimality of exhaustive search by maintaining a fixed beam width of candidates.
- [[COMET]]: Codebook-based online-adaptive multi-scale embedding combining dual-path patching, vector-quantized coreset, local scaling distance, and test-time adaptation.
- [[Decoder-Only Models]]: Transformer architectures designed exclusively for auto-regressive text generation (e.g., GPT, LLaMA).
- [[DeepSeek Load Balancing]]: The mathematical evolution of DeepSeek's MoE routing load-balancing strategies, covering V2's multi-faceted soft auxiliary losses ($L_{ExpBal}, L_{DevBal}, L_{CommBal}$) and V3's groundbreaking Auxiliary-Loss-Free dynamic bias routing.
- [[DeepSeek Shared Experts]]: An architectural innovation in DeepSeekMoE that isolates general knowledge into a set of permanently active experts, allowing the remaining routed experts to achieve fine-grained specialization.
- [[DeepSeek Sparse Attention]]: DeepSeek-V3.2's sparse attention mechanism that uses a lightning indexer to select top-k tokens, reducing computational complexity for long contexts.
- [[DOM State Compression]]: Techniques for stripping non-interactive elements from HTML to reduce token usage and improve privacy.
- [[Encoder-Decoder Models]]: Transformer architectures designed for sequence-to-sequence tasks like translation and summarization (e.g., T5, BART).
- [[Encoder-Only Models]]: Transformer architectures designed for understanding input text and generating contextual embeddings (e.g., BERT).
- [[Fine-tuning]]: The training pipeline, from Pre-training to SFT and RLHF/DPO alignment, plus PEFT methods like LoRA.
- [[FSQ]]: Finite Scalar Quantization, a highly simplified, codebook-free alternative to VQ that replaces vector quantization with bounded rounding, completely eliminating auxiliary losses and index collapse.
- [[Grouped Query Attention]]: An architectural optimization for MHA that partitions Query heads into groups to share Key/Value caches, maximizing efficiency and accuracy.
- [[GShard]]: Google's 600B parameter MoE architecture that introduced Top-2 gating, proportional random dropping, and a differentiable auxiliary loss to successfully scale transformers across distributed TPUs.
- [[KV Cache]]: The Key-Value cache optimization used during autoregressive generation to trade VRAM for compute speed.
- [[mHC]]: Manifold-Constrained Hyper-Connections, a method for widening residual streams while maintaining stability by projecting matrices onto the Birkhoff polytope using the Sinkhorn-Knopp algorithm.
- [[Model Flops Utilization]]: A hardware-efficiency metric measuring actual vs. theoretical peak FLOPs, crucial for scaling recommendations.
- [[Multi-Head Attention]]: The structural enhancement that splits attention into parallel subspaces to prevent attention thinning without adding compute.
- [[Multi-Head Latent Attention]]: DeepSeek's architectural innovation that drastically reduces KV Cache size by compressing Keys and Values into a single shared latent vector while decoupling RoPE to enable weight absorption.
- [[Multi-Query Attention]]: An extreme optimization of MHA that shares a single Key and Value head across all Query heads to drastically reduce KV cache size.
- [[Multi-Token Prediction]]: DeepSeek's sequential multi-token prediction paradigm that enables zero-overhead, highly aligned self-speculative decoding with up to 1.8x throughput speedups.
- [[OneRec]]: An end-to-end, single-stage generative recommendation architecture scaling up to 1B parameters with Sparse MoE.
- [[OneRec-V2]]: Second-generation generative recommendation model from Kuaishou Inc., scaling up to 8B with projection-free KV generation and user feedback RL.
- [[Positional Encoding]]: Techniques (like Sinusoidal, RoPE, and ALiBi) used to inject sequence order information into Transformers, compensating for their lack of recurrence.
- [[RAG]]: Advanced Retrieval-Augmented Generation architectures including dense/sparse retrieval, reranking, and Knowledge Graph extraction.
- [[RankMixer]]: A compute-bound, GPU-friendly ranking network architecture from ByteDance featuring Per-Token FFNs and ReLU-MoE routing.
- [[Request Level Batching]]: A user-centric, request-wise sample batching layout that amortizes user history sequence transfer and encoding across target candidates.
- [[Residual Connections]]: Explains the identity mapping property and its variations, including mHC for widening the residual stream.
- [[RMSNorm]]: Root Mean Square Normalization, a highly efficient variant of LayerNorm that skips mean-centering to speed up training and inference in modern LLMs.
- [[RoPE]]: A mathematical deep dive into Rotary Position Embedding, including its block-diagonal multi-dimensional formulation, efficient Hadamard product computation, Long-Range Decay proof (Abel Transformation), and LLaMA PyTorch code snippets.
- [[Router Z-Loss]]: An auxiliary loss function that stabilizes the training of massive MoE models by penalizing the squared Log-Sum-Exp of routing logits, preventing catastrophic exponential roundoff errors in bfloat16.
- [[RQ-VAE]]: Residual Quantized VAE recursively decomposing continuous vectors into multi-scale residual stacks to compress spatial dimensions without codebook collapse.
- [[Self-Attention Mechanism]]: The mathematical foundation of Transformers detailing why Query, Key, and Value matrices must use independent weights to prevent identity matrix degeneration.
- [[Sparsely-Gated MoE Layer]]: The seminal 2017 architecture that introduced conditional computation at scale via Noisy Top-K Gating and auxiliary load-balancing losses, forming the basis for modern MoE models.
- [[STCA]]: A linear-time target-to-history cross-attention mechanism with associative query-side algebraic optimization to bypass Key/Value HBM bottlenecks.
- [[Switch Transformer]]: Google's 2021 architecture that simplified MoEs using Top-1 Routing, introduced a scale-invariant load balancing loss, and utilized Selective Precision to train stable trillion-parameter models.
- [[Symbolic Discretization]]: The process of mapping continuous time-series numerical sequences into discrete symbolic tokens to bridge the gap with language-based models.
- [[TokenCast]]: An LLM-driven context-aware time series forecasting framework using decoupled reversible instance normalization and vocabulary-level symbolic discretization.
- [[Transformers]]: Foundation model architectures, including MoE and SSM, along with micro-designs like FlashAttention and RoPE.
- [[Variational Autoencoder]]: The continuous, probabilistic generative autoencoder mapping features to Gaussian distributions using KL divergence and ELBO maximization.
- [[Vector Database]]: Storage and search mechanisms for embeddings using algorithms like HNSW and IVF-PQ.
- [[VQ-VAE]]: Vector Quantized VAE, replacing continuous features with discrete codebook tokens using STE to prevent posterior collapse.
- [[VQ-VAE-2]]: Second-generation hierarchical multi-scale VQ-VAE separating global shape (using self-attention) from local details.
- [[Web Agent]]: Autonomous or semi-autonomous AI agents designed to navigate and manipulate web interfaces.
