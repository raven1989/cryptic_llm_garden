# Wiki Index

This is the content-oriented catalog of the wiki. The LLM updates this file whenever a new source is ingested or a new page is created.

## Personal Knowledge

- [[LLM Study Plan]]: A comprehensive, top-down algorithm engineering guide spanning Applications, Frameworks, and Foundation Models.
- [[My LLM Learning Sequence]]: A step-by-step, bottom-up execution plan derived from the main study outline.
- [[Layer 3 - Architectures and Operators Syllabus]]: A deep-dive syllabus for Step 1 of the learning sequence, focusing on macro-architectures, micro-designs, and inference bottlenecks.

## Research & Topics
- [[DSSM Summary]]: Microsoft Research's Deep Structured Semantic Model, projecting queries and documents into a shared semantic space via a deep network with letter-trigram word hashing, trained discriminatively on clickthrough data for web search.
- [[DCN Summary]]: Google/Stanford's Deep & Cross Network, introducing a cross network that explicitly learns bounded-degree feature interactions for CTR prediction.
- [[DIN Summary]]: Alibaba's Deep Interest Network, introducing a local activation unit that adaptively computes user interest representations per candidate ad, plus mini-batch aware regularization and Dice activation.
- [[SIM Summary]]: Alibaba's Search-based Interest Model for lifelong user behavior sequences up to 54k items, using a cascaded GSU (hard/soft search) and ESU (temporal multi-head attention) with a 2-level User Behavior Tree index.
- [[TWIN Summary]]: Kuaishou's TWo-stage Interest Network solving GSU–ESU inconsistency by making both stages share the identical MHTA relevance metric (structure + parameters), enabled by behavior feature splitting (cached inherent features + cross features compressed to 1-dim bias terms), scaling target attention to 10⁴–10⁵ behaviors.
- [[LONGER Summary]]: ByteDance's RecSys 2025 end-to-end ultra-long (10K) sequence transformer, abandoning two-stage retrieval via token merge (InnerTrans), recent-k query sampling, global-token anchors, and hybrid causal attention — with synchronous GPU training, KV-cache serving, and industrial scaling laws.
- [[NCF Summary]]: Neural Collaborative Filtering, replacing matrix factorization's fixed inner product with a learned neural interaction function (GMF, MLP, NeuMF) for implicit-feedback recommendation.
- [[A-LLMRec Summary]]: KAIST and NAVER's training-free framework aligning frozen sequential collaborative recommenders with frozen LLMs for warm and cold scenarios.
- [[Chronos-2 Summary]]: Amazon AWS zero-shot time series foundation model shifting to an encoder-only architecture with continuous 21-quantile regression and Group Attention.
- [[CITRAS Summary]]: Hitachi decoder-only transformer with 2D alternating temporal-spatial attention, KV Shift, and Attention Score Smoothing.
- [[In-Context Learning and Future Covariates in Decoder-Only Forecasting]]: Technical synthesis comparing the covariate-integration strategies of Chronos-2, CITRAS, and TimesFM 2.5 (XReg) under autoregressive constraints.

- [[RankMixer Summary]]: ByteDance's RankMixer architecture boosting ranking model MFU from 4.5% to 45% and scaling parameters to 1B under strict SLA constraints.
- [[Graphormer Summary]]: Microsoft's landmark Graphormer architecture injecting graph topology directly into attention via Centrality, Spatial, and Edge encodings.
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
- [[Pre-training Large Language Models]]: Complete Chapter 15 summary in six parts — data-source taxonomy + data-mix distribution of 15 representative models (Part 1); the preprocessing pipeline: quality filtering, sensitive-content filtering, deduplication (MinHash/LSH), tokenization (Part 2); encoder-only/causal-decoder/prefix-decoder architectures (Part 3); long-context modeling via position-encoding extension and context-window restructuring (Part 4); data scheduling: mixture ratios and curricula for code/math/long-context (Part 5); and pre-training objectives (LM/Prefix-LM/FIM/DAE/MoD), optimization settings, and 3D parallelism + ZeRO + mixed precision (Part 6).
- [[Why is Attention divided by Root d_k]]: A mathematical breakdown of why the dot product of Query and Key vectors in Transformers is scaled by $\sqrt{d_k}$ to prevent variance growth and vanishing gradients.
- [[Web Agent Architecture]]: An analysis of 4 different architectural paradigms for building AI web agents.
- [[Taxy.ai Implementation]]: Code-level analysis of how Taxy.ai extracts DOM state and executes hardware-level actions.
- [[Browser-Use Implementation]]: Code-level analysis of how Browser-Use utilizes Pure Python CDP (Chrome DevTools Protocol) to execute stealthy web agents without JavaScript injection.
- [[Action Execution - JS vs CDP]]: A detailed breakdown of why modern web agents must use the Chrome DevTools Protocol to simulate hardware interrupts rather than executing simple JavaScript.
- [[Beam Search Summary]]: A comprehensive summary of Beam Search decoding mechanics, log-probability scoring, and early stopping behavior.
- [[OneRec Summary]]: A comprehensive, unified generative recommender from Kuaishou Inc. replacing the traditional multi-stage retrieval and ranking cascade.
- [[OneRec-V2 Summary]]: Detailed breakdown of OneRec-V2's Lazy Decoder-Only architecture, context processor optimization, and scaling laws.
- [[Codebook Technology History Summary]]: The master, high-fidelity research guide charting the chronological evolution and critical trade-offs of codebook discrete representation learning, from continuous VAE to FSQ.
- [[Conversational Recommender Systems]]: Synthesis of the architectural evolution, core trade-offs, and taxonomic paradigms of LLM-based Conversational Recommender Systems (CoLLM, LLaRA, CRAG, MCCRS, GCRS).
- [[From Values to Tokens Summary]]: An LLM-driven framework using language-based symbolic representation as a unified intermediary for context-aware time series forecasting.
- [[COMET Summary]]: SOTA codebook-based online-adaptive multi-scale embedding framework for multivariate time-series anomaly detection.
- [[CoLLM Summary]]: University of Science and Technology of China and Meta AI's innovative approach explicitly integrating low-rank collaborative embeddings into Large Language Models.
- [[CRAG Summary]]: University of Virginia, Cornell, and Netflix's landmark framework combining state-of-the-art LLMs with Collaborative Filtering for conversational recommendations.
- [[CTRL Summary]]: Huawei Noah's Ark Lab's model-agnostic, two-stage framework distilling semantic and world knowledge from Pre-trained Language Models into lightweight collaborative models.
- [[GCRS Summary]]: Nanyang Technological University's fully generative Conversational Recommender System integrating semantic IDs and structured next-token generation.
- [[LC-Rec Summary]]: Renmin University's generative recommendation model using RQ-VAE index codes, Sinkhorn-Knopp collision resolution, and multi-task alignment-tuning.
- [[LLaRA Summary]]: USTC and PolyU's Large Language-Recommendation Assistant utilizing hybrid prompting and progressive curriculum prompt tuning for sequential recommendation.
- [[MCCRS Summary]]: UESTC, Tongji, and SWUFE's Multi-Type Context-Aware Conversational Recommender System using Mixture-of-Experts.
- [[SeqLLM Summary]]: WeChat Pay and SJTU's framework injecting long behavioral-sequence modeling into LLMs for high-stakes merchant risk control and SOTA recommendation.

## Media Companion (Books, Movies, etc.)

- (Empty)

## Entities & Concepts (Cross-cutting)
- [[DCN]]: Google/Stanford's Deep & Cross Network, introducing a cross network that explicitly learns bounded-degree feature interactions for CTR prediction.
- [[DIN]]: Alibaba's Deep Interest Network, introducing a local activation unit for candidate-aware user interest representation, plus mini-batch aware regularization and Dice activation.
- [[A-LLMRec]]: All-round LLM-based Recommender, a framework aligning pre-trained collaborative user/item embeddings with frozen LLMs using dual-stage projection MLPs.
- [[Attention Score Smoothing]]: An attention regularization technique applying EMA across temporal steps to stabilize cross-variate dependencies and filter high-frequency noise.
- [[Group Attention]]: A masked self-attention pooling mechanism that dynamically isolates token aggregation within designated task groups.
- [[KV Shift]]: A step-wise cross-variate attention design that aligns Query and Key at step $i$ but shifts the known covariate Value to step $i+1$ to retrieve future context without data leakage.
- [[TimesFM XReg Modes]]: Dual decoupled linear-residual modeling integration strategies (Residual and Error Correction) used to incorporate external regressors in Google TimesFM 2.5.

- [[Agents]]: Workflows, State Machines, multi-agent systems, and tool calling sandboxes.
- [[ALiBi]]: Attention with Linear Biases, a positional encoding method that applies distance penalties directly to attention scores to enable length extrapolation.
- [[Attention Sink]]: The StreamLLM phenomenon where deep layers dump attention on initial tokens; LONGER counteracts it by injecting global-token anchors.
- [[Balanced K-means]]: An optimized clustering algorithm with strict cluster-capacity constraints used to generate balanced semantic IDs.
- [[Beam Search]]: A heuristic search decoding algorithm that balances the speed of greedy search with the optimality of exhaustive search by maintaining a fixed beam width of candidates.
- [[COMET]]: Codebook-based online-adaptive multi-scale embedding combining dual-path patching, vector-quantized coreset, local scaling distance, and test-time adaptation.
- [[CoLLM]]: Collaborative Large Language Model, a framework aligning low-rank traditional collaborative embeddings with LLM token embedding space.
- [[CRAG]]: Collaborative Retrieval Augmented Generation, a pipeline bridging dialogue context and behavioral user-item interaction data using LLMs.
- [[CTRL]]: Connect Collaborative and Language Model, a multi-modal CTR prediction framework aligning tabular features with textual semantic features using dual-encoders.
- [[GCRS]]: Generative Conversational Recommender System, a unified next-token prediction framework representing items as discrete RQ-VAE coordinates.
- [[LC-Rec]]: Language-Collaborative Recommender, a generative recommendation framework utilizing Sinkhorn-Knopp optimal transport constraints and multi-task alignment.
- [[LONGER]]: ByteDance's end-to-end ultra-long (10K) sequence transformer for industrial recommenders, replacing two-stage retrieval with token merge, recent-k query sampling, global tokens, and KV-cache serving.
- [[LLaRA]]: Large Language-Recommendation Assistant, an aligned multi-modal sequential recommendation framework using an MLP projector (SR2LLM).
- [[MCCRS]]: Multi-Type Context-Aware Conversational Recommender System, a modular MoE-based architecture managed by a ChairBot.
- [[Decoder-Only Models]]: Transformer architectures designed exclusively for auto-regressive text generation (e.g., GPT, LLaMA).
- [[DeepSeek Load Balancing]]: The mathematical evolution of DeepSeek's MoE routing load-balancing strategies, covering V2's multi-faceted soft auxiliary losses ($L_{ExpBal}, L_{DevBal}, L_{CommBal}$) and V3's groundbreaking Auxiliary-Loss-Free dynamic bias routing.
- [[DeepSeek Shared Experts]]: An architectural innovation in DeepSeekMoE that isolates general knowledge into a set of permanently active experts, allowing the remaining routed experts to achieve fine-grained specialization.
- [[DeepSeek Sparse Attention]]: DeepSeek-V3.2's sparse attention mechanism that uses a lightning indexer to select top-k tokens, reducing computational complexity for long contexts.
- [[DOM State Compression]]: Techniques for stripping non-interactive elements from HTML to reduce token usage and improve privacy.
- [[DSSM]]: Deep Structured Semantic Model, a deep architecture projecting queries and documents into a shared semantic space via letter-trigram word hashing and discriminative clickthrough training, enabling semantic matching beyond keyword overlap.
- [[EASE]]: Embarrassingly Shallow Autoencoders, a linear collaborative filtering model solved via a regularized closed-form matrix inverse.
- [[Encoder-Decoder Models]]: Transformer architectures designed for sequence-to-sequence tasks like translation and summarization (e.g., T5, BART).
- [[Encoder-Only Models]]: Transformer architectures designed for understanding input text and generating contextual embeddings (e.g., BERT).
- [[Fine-tuning]]: The training pipeline, from Pre-training to SFT and RLHF/DPO alignment, plus PEFT methods like LoRA.
- [[FSQ]]: Finite Scalar Quantization, a highly simplified, codebook-free alternative to VQ that replaces vector quantization with bounded rounding, completely eliminating auxiliary losses and index collapse.
- [[Grouped Query Attention]]: An architectural optimization for MHA that partitions Query heads into groups to share Key/Value caches, maximizing efficiency and accuracy.
- [[Graphormer]]: A graph-configured Transformer architecture incorporating centrality, topological distance (SPD), and edge features directly into the self-attention logits.
- [[GShard]]: Google's 600B parameter MoE architecture that introduced Top-2 gating, proportional random dropping, and a differentiable auxiliary loss to successfully scale transformers across distributed TPUs.
- [[KV Cache]]: The Key-Value cache optimization used during autoregressive generation to trade VRAM for compute speed.
- [[Locality-Sensitive Hashing]]: Hash families where collision probability equals similarity — covering MinHash's min-signature trick for Jaccard estimation, banding, and its role in web-scale pre-training deduplication.
- [[Long-Context Positional Encoding]]: Why relative RoPE still fails to extrapolate (low-frequency subspaces), and the full remedy landscape — PI, sliding-window truncation, NTK-aware / Dynamic NTK / YaRN frequency scaling, base enlargement — plus the DeepSeek-V4 1M-context case study.
- [[Perplexity]]: The exponentiated average negative log-likelihood metric, doubling as an LM evaluation metric and a data-quality heuristic for pre-training corpus filtering.
- [[mHC]]: Manifold-Constrained Hyper-Connections, a method for widening residual streams while maintaining stability by projecting matrices onto the Birkhoff polytope using the Sinkhorn-Knopp algorithm.
- [[Model Flops Utilization]]: A hardware-efficiency metric measuring actual vs. theoretical peak FLOPs, crucial for scaling recommendations.
- [[Multi-Head Attention]]: The structural enhancement that splits attention into parallel subspaces to prevent attention thinning without adding compute.
- [[Multi-Head Latent Attention]]: DeepSeek's architectural innovation that drastically reduces KV Cache size by compressing Keys and Values into a single shared latent vector while decoupling RoPE to enable weight absorption.
- [[Multi-Query Attention]]: An extreme optimization of MHA that shares a single Key and Value head across all Query heads to drastically reduce KV cache size.
- [[Multi-Token Prediction]]: DeepSeek's sequential multi-token prediction paradigm that enables zero-overhead, highly aligned self-speculative decoding with up to 1.8x throughput speedups.
- [[NCF]]: Neural Collaborative Filtering, a general framework learning the user–item interaction function with a neural network (GMF, MLP, NeuMF) instead of a fixed inner product.
- [[OneRec]]: An end-to-end, single-stage generative recommendation architecture scaling up to 1B parameters with Sparse MoE.
- [[OneRec-V2]]: Second-generation generative recommendation model from Kuaishou Inc., scaling up to 8B with projection-free KV generation and user feedback RL.
- [[Positional Encoding]]: Techniques (like Sinusoidal, RoPE, and ALiBi) used to inject sequence order information into Transformers, compensating for their lack of recurrence.
- [[R-GCN]]: Relational Graph Convolutional Networks, a multi-relational GNN designed to propagate messages along distinct edge categories.
- [[RAG]]: Advanced Retrieval-Augmented Generation architectures including dense/sparse retrieval, reranking, and Knowledge Graph extraction.
- [[RankMixer]]: A compute-bound, GPU-friendly ranking network architecture from ByteDance featuring Per-Token FFNs and ReLU-MoE routing.
- [[Request Level Batching]]: A user-centric, request-wise sample batching layout that amortizes user history sequence transfer and encoding across target candidates.
- [[Residual Connections]]: Explains the identity mapping property and its variations, including mHC for widening the residual stream.
- [[RMSNorm]]: Root Mean Square Normalization, a highly efficient variant of LayerNorm that skips mean-centering to speed up training and inference in modern LLMs.
- [[RoPE]]: A mathematical deep dive into Rotary Position Embedding, including its block-diagonal multi-dimensional formulation, efficient Hadamard product computation, Long-Range Decay proof (Abel Transformation), and LLaMA PyTorch code snippets.
- [[Router Z-Loss]]: An auxiliary loss function that stabilizes the training of massive MoE models by penalizing the squared Log-Sum-Exp of routing logits, preventing catastrophic exponential roundoff errors in bfloat16.
- [[RQ-VAE]]: Residual Quantized Variational Autoencoder, a hierarchical vector quantization model decomposing continuous embeddings into multi-stage residual stacks to compress vector dimensionality without codebook collapse.
- [[Self-Attention Mechanism]]: The mathematical foundation of Transformers detailing why Query, Key, and Value matrices must use independent weights to prevent identity matrix degeneration.
- [[SeqLLM]]: Behavioral-Sequence Augmented LLM, a sequence-language framework representing events as field-level discrete tokens and injecting sequence capabilities without catastrophic forgetting.
- [[SIM]]: Search-based Interest Model, a two-stage cascaded CTR prediction framework (GSU + ESU) scaling sequential user behavior modeling up to 54,000 items with sub-30ms latency.
- [[TWIN]]: TWo-stage Interest Network, Kuaishou's lifelong behavior model where CP-GSU and ESU share one MHTA relevance metric (structure + parameters), made feasible by splitting behavior features into cacheable inherent features and 1-dim-bias cross features.
- [[Sinkhorn-Knopp Algorithm]]: An iterative row/column normalization procedure that projects positive matrices onto the Birkhoff polytope (doubly stochastic matrices) and differentiably solves entropy-regularized Optimal Transport, used by mHC and LC-Rec.
- [[SASRec]]: Self-Attention Sequential Recommendation, a Transformer-based decoder-only model that dynamically attends to historical item sequences to predict next actions.
- [[Sparsely-Gated MoE Layer]]: The seminal 2017 architecture that introduced conditional computation at scale via Noisy Top-K Gating and auxiliary load-balancing losses, forming the basis for modern MoE models.
- [[STCA]]: A linear-time target-to-history cross-attention mechanism with associative query-side algebraic optimization to bypass Key/Value HBM bottlenecks.
- [[SwiGLU FFN]]: A gated activation variant of the feed-forward network using three projection matrices and a SiLU/Swish gate, forming the standard FFN component of modern LLMs.
- [[Switch Transformer]]: Google's 2021 architecture that simplified MoEs using Top-1 Routing, introduced a scale-invariant load balancing loss, and utilized Selective Precision to train stable trillion-parameter models.
- [[Symbolic Discretization]]: The process of mapping continuous time-series numerical sequences into discrete symbolic tokens to bridge the gap with language-based models.
- [[TokenCast]]: An LLM-driven context-aware time series forecasting framework using decoupled reversible instance normalization and vocabulary-level symbolic discretization.
- [[Tokenizer]]: Subword tokenization for LLMs — BPE/WordPiece/Unigram algorithms compared (merge vs. prune, frequency vs. likelihood), the SentencePiece framework, and Chinese tokenization practice.
- [[Transformers]]: Foundation model architectures, including MoE and SSM, along with micro-designs like FlashAttention and RoPE.
- [[Variational Autoencoder]]: The continuous, probabilistic generative autoencoder mapping features to Gaussian distributions using KL divergence and ELBO maximization.
- [[Vector Database]]: Storage and search mechanisms for embeddings using algorithms like HNSW and IVF-PQ.
- [[VQ-VAE]]: Vector Quantized VAE, replacing continuous features with discrete codebook tokens using STE to prevent posterior collapse.
- [[VQ-VAE-2]]: Second-generation hierarchical multi-scale VQ-VAE separating global shape (using self-attention) from local details.
- [[Web Agent]]: Autonomous or semi-autonomous AI agents designed to navigate and manipulate web interfaces.
