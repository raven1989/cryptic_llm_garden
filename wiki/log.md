# Operation Log

An append-only chronological record of what happened and when. Useful for tracking the evolution of the wiki.

## [2026-04-08] Initialization
Wiki structure and schema created.
## [2026-04-08] Ingest | LLM Algorithm Engineer Study Plan
Ingested four raw outline files (`raw/Outline.md`, `raw/Outline_Layer1.md`, `raw/Outline_Layer2.md`, `raw/Outline_Layer3.md`). Created a high-level study plan (`wiki/personal/LLM Study Plan.md`) and extracted key concepts into new entity pages: `[[Transformers]]`, `[[RAG]]`, `[[Agents]]`, `[[Fine-tuning]]`, and `[[Vector Database]]`. Updated the index.
## [2026-04-08] Query Synthesis | My LLM Learning Sequence
Synthesized an execution plan from a user query about the `[[LLM Study Plan]]`. Filed the answer back into the wiki as a new page (`wiki/personal/My LLM Learning Sequence.md`) and updated the index.

## [2026-04-09] Ingest | Web Agent Architecture
Ingested the raw brainstorm document (`raw/web_agent/web_agent_abstract.md`). Created a summary page at `wiki/research/Web Agent Architecture.md`. Extracted and created new entity pages for `[[Web Agent]]` and `[[DOM State Compression]]` in `wiki/entities/`. Updated `wiki/index.md` with new links.

## [2026-04-09] Ingest | Taxy.ai Codebase
Cloned and moved the Taxy.ai open-source repository to `raw/web_agent/taxyai/`. Analyzed the codebase and synthesized a new research page at `wiki/research/Taxy.ai Implementation.md` detailing its two-pass DOM extraction (`getAnnotatedDOM.ts`, `simplifyDom.ts`) and hardware-level action execution via CDP (`domActions.ts`). Updated `wiki/index.md` and added cross-links to existing entity pages.

## [2026-04-09] Ingest | Browser-Use Codebase
Cloned and moved the `browser-use/browser-use` repository to `raw/web_agent/browser-use/`. Analyzed its modernized architecture, discovering a shift away from JS injection toward pure Python Chrome DevTools Protocol (`cdp_use`) for stealthy DOM snapshotting and hardware-simulated mouse actions. Created `wiki/research/Browser-Use Implementation.md`. Updated the index and entity definitions accordingly.

## [2026-04-09] Synthesis | Action Execution (JS vs CDP)
Synthesized our conversation regarding the limitations of JavaScript execution in modern web environments into a new research page at `wiki/research/Action Execution - JS vs CDP.md`. Outlined the importance of `isTrusted` flags, React state management, and z-index visual overlays when building Web Agents. Updated `wiki/entities/Web Agent.md` and `wiki/entities/DOM State Compression.md` with links to this new concept.## [2026-04-13] Create | [[Layer 3 - Architectures and Operators Syllabus]] generated from raw/llm_study_plan/Outline_Layer3.md
## [2026-04-13] Synthesis | Self-Attention QKV Weight Sharing
Synthesized a detailed mathematical explanation of why Q, K, and V matrices do not share weights in vanilla self-attention. Specifically documented the "Identity Matrix Degeneration" effect caused by $W_Q = W_K$ and the necessity of breaking symmetry. Created the new entity page `wiki/entities/Self-Attention Mechanism.md` and updated `wiki/index.md`.

## [2026-04-14] Ingest | Why is Attention divided by √dₖ
Ingested the raw source `raw/transformer/Why is Attention divided by √dₖ.md`. Created a new research summary page at `wiki/research/Why is Attention divided by Root d_k.md`. Updated the `wiki/entities/Self-Attention Mechanism.md` page with a new section on Scaled Dot-Product Attention (variance growth and vanishing gradients). Updated `wiki/index.md`.

## [2026-04-14] Ingest | KV Cache Synthesis
* Synthesized existing knowledge about KV Cache optimization from `raw/llm_study_plan/Outline_Layer3.md` and related notes.
* Created `wiki/entities/KV Cache.md`.
* Updated `wiki/index.md`.

## [2026-04-15] Ingest | KV Cache Deep Dive
* Extracted implementation details from `raw/LLM/Understanding and coding the KV Cache in LLMs from Scratch.md` and `raw/LLM/KV Cache Explained.md`.
* Updated `wiki/entities/KV Cache.md` to explain VRAM usage, per-layer cache independence, memory fragmentation (torch.cat vs pre-allocation), sliding window truncation, and cache cleanup lifecycle.

## [2026-04-15] Ingest | LLM Inference VRAM & Compute Estimation
* Ingested `raw/LLM/大模型推理显存和计算量估计方法.md`.
* Created `wiki/research/LLM Inference VRAM and Compute Estimation.md` detailing exact mathematical formulas for Prefill/Decode FLOPs and KV Cache memory footprint.
* Updated `wiki/entities/KV Cache.md` with the explicit byte-size calculation formula showing the impact of GQA (`num_key_value_heads`).
* Updated `wiki/index.md`.

## [2026-04-16] Ingest | Encoder-Only, Encoder-Decoder, and Decoder-Only Models
* Ingested `raw/LLM/ Saptarshi Datta A curious explorer of Technology and Life   Kolkata, India datta.3@iitj.ac.in saptarshidatta97 Encoder Only Encoder Decoder And Decoder Only Models.md`.
* Created a new summary page at `wiki/research/Encoder Only, Encoder Decoder, And Decoder Only Models.md`.
* Created dedicated entity pages for `wiki/entities/Encoder-Only Models.md`, `wiki/entities/Encoder-Decoder Models.md`, and `wiki/entities/Decoder-Only Models.md`.
* Updated `wiki/entities/Transformers.md` to link to these new entity pages.
* Updated `wiki/index.md` with links to the new research summary and entity pages.

## [2026-04-17] Ingest & Synthesis | Positional Encoding
* Ingested `raw/LLM/You could have designed state of the art positional encoding.md` explaining the evolutionary history of position embeddings from Integer to RoPE.
* Synthesized an answer answering why Transformers need positional encoding, detailing early flaws (Integer/Binary), Sinusoidal encoding, Rotary Position Embedding (RoPE), and ALiBi.
* Created new entity page `wiki/entities/Positional Encoding.md`.
* Updated `wiki/entities/Transformers.md` to link to the new entity.
* Updated `wiki/index.md` with the new entity page.

## [2026-04-17] Enhancement | Positional Encoding Videos
* Embedded natively supported `.mp4` video files from `raw/LLM/` into `wiki/entities/Positional Encoding.md` to visually demonstrate the mathematical differences between Integer, Binary, Sinusoidal, and Rotary Position Encodings.

## [2026-04-17] Maintenance | Fix KaTeX rendering in raw/LLM source
* Repaired multiple LaTeX syntax errors in `raw/LLM/You could have designed state of the art positional encoding.md` where matrices and aligned equations were missing proper `\begin{bmatrix}`, `\begin{pmatrix}`, and `\begin{aligned}` wrappers, causing KaTeX parsing errors in Obsidian.

## [2026-04-17] Maintenance | Fix Unicode Math Anomaly in raw/LLM source
* Replaced anomalous Unicode combining characters (like the `U+20D7` combining right arrow) and non-standard spacing (`\mid` instead of `|`, invisible `U+2061` function application) with standard, clean KaTeX (`\vec{a} \cdot \vec{b} = |\vec{a}| |\vec{b}| \cos \theta`) in `raw/LLM/You could have designed state of the art positional encoding.md` to prevent rendering glitches in Obsidian's Edit Mode.

## [2026-04-17] Maintenance | Fix KaTeX rendering in 十分钟看懂RoPE.md
* Cleaned up invalid left/right delimiter hacks (`\left[\right.` -> `[`, `\left{\right.` -> `\{`) throughout `raw/LLM/十分钟看懂RoPE.md` which were causing KaTeX parsing failures in Obsidian.
* Replaced `\begin{matrix}` wrappers with `\begin{aligned}` or `\begin{pmatrix}` where alignment and matrix columns were failing.

## [2026-04-17] Maintenance | Fix \hdots KaTeX error in raw/LLM/十分钟看懂RoPE.md
* Replaced instances of the undefined control sequence `\hdots` with the standard KaTeX command `\cdots` in the 1.4 section matrix to resolve a final parsing error.

## [2026-04-17] Ingest | 十分钟看懂RoPE
* Ingested `raw/LLM/十分钟看懂RoPE.md` which contains a highly technical deep dive into Rotary Position Embedding.
* Created `wiki/research/十分钟看懂RoPE Summary.md` to summarize the mathematical themes of the article.
* Created `wiki/entities/RoPE.md` as a dedicated entity page. Documented the $d$-dimensional block-diagonal matrix formulation, efficient Hadamard product computation, Long-Range Decay (via Abel Transformation), Length Extrapolation properties (orthogonal matrices), and the complex-domain PyTorch implementation used in LLaMA.
* Updated `wiki/entities/Positional Encoding.md` to point users seeking deeper math/code toward the new `[[RoPE]]` page.
* Updated `wiki/index.md` with links to the new research summary and entity pages.

## [2026-04-17] Enhancement | Add missing inner product proof to RoPE
* Appended the relative inner product function $g(\mathbf{x}_m, \mathbf{x}_n, m-n)$ to the `wiki/entities/RoPE.md` deep dive to explicitly mathematically prove that absolute rotation of query and key vectors naturally resolves to a single rotation based on their relative distance angle $(m-n)\theta$.
## [2026-04-21] Ingest | ALiBi Positional Encoding
* Ingested `raw/LLM/大模型位置编码-ALiBi位置编码.md` discussing Attention with Linear Biases (ALiBi) and length extrapolation.
* Created a summary page at `wiki/research/大模型位置编码-ALiBi位置编码 Summary.md`.
* Created a detailed entity page `wiki/entities/ALiBi.md` complete with the core equation, the scalar $m$ calculation, and an ASCII visualization matrix of the linear bias operation before softmax.
* Updated `wiki/index.md` and `wiki/entities/Positional Encoding.md` to link the new entity and concepts.

## [2026-04-21] Ingest | 大模型原理与架构 (Chapter 4)
* Ingested Chapter 4 (`raw/LLM/大模型原理与架构/04_position_encoding/`) covering the evolution of Position Encoding designs.
* Updated `wiki/entities/Positional Encoding.md` to include:
  * Learnable Positional Encodings (GPT/BERT) and its fundamental inability to extrapolate.
  * T5's bucked relative position bias and Transformer-XL interactions.
  * A concluding analysis on the 4 major architectural shifts in Positional Encodings (Absolute->Relative, Embedding->Attention, Fixed->Extrapolatable, Complex->Simple) and a markdown comparison matrix.
* Updated `wiki/entities/RoPE.md` to document explicit length extrapolation optimization techniques (`Position Interpolation`, `NTK-aware scaling`, `YaRN`) and elaborated on the "rotation cancellation" effect that mathematically enforces Long-Range Decay (implicit locality bias).

## [2026-05-11] Ingest & Synthesis | Multi-Head Attention
* Synthesized the explanations for matrix dimensions and functional advantages of Multi-Head Attention based on `raw/LLM/大模型原理与架构/02_attention/2.3_multi_head.md`.
* Created a new dedicated entity page `wiki/entities/Multi-Head Attention.md`.
* Updated `wiki/entities/Transformers.md` and `wiki/entities/Self-Attention Mechanism.md` to cross-link to the new page.
* Updated `wiki/index.md` to include the new entity.

## [2026-05-12] Ingest | What is grouped query attention (GQA)
* Used a Python HTML parser to manually extract the article from IBM since WebFetch was blocked. Saved to `raw/LLM/What is grouped query attention.md`.
* Created `wiki/research/Grouped Query Attention Summary.md`.
* Created entity pages for `wiki/entities/Grouped Query Attention.md` and `wiki/entities/Multi-Query Attention.md`.
* Updated `wiki/entities/Multi-Head Attention.md` to link to these KV cache optimizations.
* Updated `wiki/index.md`.
## [2026-05-12] Enhancement | GQA Visualization
* Added an ASCII visualization to `wiki/entities/Grouped Query Attention.md` comparing the structural routing of Q, K, and V heads between MHA, GQA, and MQA.
## [2026-05-12] Maintenance | Fix YAML Array Parsing
* Corrected an anomaly in the automated YAML fix script that resulted in doubly-wrapped brackets for some `related:` and `sources:` arrays.
* Converted older list-based YAML source blocks in `wiki/entities/KV Cache.md`, `DOM State Compression.md`, `Web Agent.md`, and `wiki/research/Browser-Use Implementation.md` into proper quoted wikilink arrays for Obsidian Properties support.
## [2026-05-12] Maintenance | Clean up straggler YAML properties
* Found and corrected the remaining unformatted `sources:` list arrays in `wiki/research/Web Agent Architecture.md`, `wiki/research/Taxy.ai Implementation.md`, and `wiki/research/大模型位置编码-ALiBi位置编码 Summary.md`, ensuring every single file in the wiki conforms perfectly to Obsidian's quoted wikilink array format for properties.
## [2026-05-12] Create | RMSNorm
* Created a dedicated entity page `wiki/entities/RMSNorm.md` detailing Root Mean Square Normalization, its mathematical formulation, and its relationship with Pre-Norm architectures.
* Updated `wiki/index.md` to link the new entity page.
## [2026-05-12] Ingest | DeepSeek's Multi-Head Latent Attention
* Ingested two source articles (`raw/LLM/Build DeepSeek-V3: Multi-Head Latent Attention (MLA) Architecture.md` and `raw/LLM/DeepSeek's Multi-Head Latent Attention.md`).
* Created a dedicated entity page `wiki/entities/Multi-Head Latent Attention.md` detailing KV Cache Compression, decoupled RoPE, weight absorption, and the asymmetric design of positional vectors.
* Updated `wiki/index.md` to link the new entity page.

## [2026-05-12] Ingest | DeepSeek-V3.2 Architecture
* Ingested the newly extracted markdown paper `raw/LLM/DeepSeek-V3.2- Pushing the Frontier of Open Large Language Models.md`.
* Created a new entity page `wiki/entities/DeepSeek Sparse Attention.md` (DSA) to document the lightning indexer and fine-grained token selection mechanisms. Included the extracted diagram `wiki/media/dsa_architecture.jpeg`.
* Updated `wiki/index.md` to link the new DSA entity page.

## [2026-05-13] Synthesis | DeepSeek-V3.2 Training Pipeline
* Created `wiki/research/DeepSeek-V3.2 Training Pipeline.md` summarizing the post-training innovations.
* Documented the two-stage DSA continued pre-training process and included the specific KL-divergence loss functions.
* Documented the GRPO scaling optimizations (Unbiased KL Estimate, Off-Policy Sequence Masking) including their exact mathematical formulations.
* Documented the agentic tool-use integration, specifically the context management retention rules and the synthetic environment generation pipeline.
* Updated `wiki/index.md` to link to the new research page.
## [2026-05-13] Ingest | mHC: Manifold-Constrained Hyper-Connections
- Read: `raw/LLM/mHC: Manifold-Constrained Hyper-Connections.md`
- Created: `wiki/research/mHC Summary.md`

## [2026-05-20] Ingest | Switch Transformer Architecture
- Read: `raw/LLM/Switch Transformers - Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.pdf`
- Created: `wiki/entities/Switch Transformer.md` detailing Top-1 Switch Routing, the scale-invariant Differentiable Load Balancing Loss (multiplying by N), and the three stability techniques (Selective Precision, Expert Dropout, Initialization Scaling).
- Added: Figure 2 architecture diagram to `wiki/media/switch_architecture.png`.
- Updated: `wiki/index.md`
## [2026-05-21] Ingest | ST-MoE: Designing Stable and Transferable Sparse Expert Models
Ingested ST-MoE paper. Created `wiki/research/ST-MoE Summary.md` and `wiki/entities/Router Z-Loss.md`. Updated index.

## [2026-05-21] Ingest | DeepSeek Shared Experts (DeepSeekMoE)
Created `wiki/entities/DeepSeek Shared Experts.md` from DeepSeek-V2/V3 papers, including architecture diagram and math. Updated index.

## [2026-05-21] Ingest | Multi-Head Latent Attention
Added DeepSeek-V2 equations 9-11 covering low-rank key-value joint compression into `wiki/entities/Multi-Head Latent Attention.md`.

## [2026-05-21] Ingest | DeepSeek Load Balancing
Created `wiki/entities/DeepSeek Load Balancing.md` to capture the detailed hardware and statistical reasons for MoE load balancing, and mathematically compared DeepSeek-V2 auxiliary losses and DeepSeek-V3 Auxiliary-Loss-Free dynamic bias routing. Updated index.

## [2026-05-21] Ingest | Mixture of Experts Summary
Created the master compilation page `wiki/research/Mixture of Experts Summary.md` to review and cleanly connect all the dots, concepts, and papers in our MoE sub-wiki (from Shazeer 2017 up to DeepSeek-V3). Updated index.

## [2026-05-21] Enhancement | MoE Chronological Evolution Timeline Diagram
- Generated a high-fidelity vector diagram `wiki/media/moe_evolution_timeline.svg` representing the timeline of MoE milestones from 1991 to 2026 in the premium Claude Official style.
- Updated the main `wiki/research/Mixture of Experts Summary.md` page to replace the outdated ASCII block diagram with the new professionally-designed vector diagram using direct SVG embedding (`![[wiki/media/moe_evolution_timeline.svg]]`) for native Obsidian rendering and high-scalability vector editing.


## [2026-05-25] Ingest & Create | Multi-Token Prediction (MTP)
- Read: `raw/LLM/DeepSeek-V3 Technical Report.pdf` Section 2.2 and relevant evaluation metrics.
- Extracted: Figure 3 ("Illustration of our Multi-Token Prediction implementation") from page 10 of the PDF, cropped and formatted as `wiki/media/mtp_architecture.png`.
- Created: `wiki/entities/Multi-Token Prediction.md` containing highly rigorous mathematical formulations, physical/hardware reasons for MTP performance, a detailed comparative architectural analysis (against Medusa, EAGLE, traditional speculative decoding), and the step-by-step speculative decoding causal mask execution traces.
- Updated: `wiki/index.md` to index the new MTP entity page.

## [2026-05-25] Synthesis & Create | Speculative Decoding
- Read: `raw/LLM/Looking back at speculative decoding.md` and `raw/LLM/An Introduction to Speculative Decoding for Reducing Latency in AI Inference.md`.
- Created: `wiki/research/Speculative Decoding.md` containing the fundamental explanation of the GPU memory bandwidth wall, the mathematical proof of Speculative Sampling (proving probability equivalence with zero quality degradation), and detailed comparative breakdowns of modern speculative paradigms (EAGLE-3, dynamic draft trees, and MTP).
- Updated: `wiki/index.md` to index the new Speculative Decoding research page.





## [2026-06-02] Ingest | 深入理解 Beam Search
- Ingested the raw guide `raw/Recommendation/深入理解 Beam Search.md`.
- Created a new summary page at `wiki/research/Beam Search Summary.md` detailing beam width, numerical underflow/log-likelihood, step-by-step decoding trace, and early stopping.
- Created a new entity page at `wiki/entities/Beam Search.md` providing detailed mathematical scoring formulations, length normalization penalties, key parameter definitions, and standard production code implementation.
- Updated `wiki/index.md` with links to both new pages.

## [2026-06-05] Ingest | OneRec: Unifying Retrieve and Rank
- Ingested: `raw/Recommendation/OneRec-Unifying Retrieve and Rank with Generative Recommender and Preference Alignment.pdf`.
- Created: `wiki/research/OneRec Summary.md` containing a comprehensive summary of the Kuaishou paper with equations, algorithms, implementation details, and deployment architecture, referencing the framework diagram at `wiki/media/onerec_overall_framework.png`.
- Created: `wiki/entities/Balanced K-means.md` detailing the mathematical formulations, cardinality constraints, distance distortion trade-offs, and hierarchical residual quantization flow.
- Created: `wiki/entities/OneRec.md` detailing the unified encoder-decoder architecture, parameters scaling with Sparse MoE, and deployment specifications.
- Updated: `wiki/index.md` to link and categorize all new pages under topics and concepts.

## [2026-06-08] Ingest | OneRec-V2 Lazy Decoder-Only Architecture
- Ingested: `raw/Recommendation/OneRec-V2 Technical Report.pdf` Section 1 and 2.
- Created: `wiki/research/OneRec-V2 Summary.md` containing a comprehensive summary of the Lazy Decoder-Only architecture, including New Impression Only (NIO) organization, Context Processor, and empirical scaling laws.
- Created: `wiki/entities/OneRec-V2.md` containing the technical specification comparisons (V1 vs. V2), projection-free KV generation details, and architectural formulas.
- Updated: `wiki/index.md` to link and categorize these new pages under topics and concepts.


## [2026-06-22] Maintenance | VAE Notation Heads Up
- Added a `[!WARNING]` callout to `[[wiki/entities/Variational Autoencoder.md]]` alerting readers that the notation letters for $p$ and $q$ are swapped relative to standard literature (using $p$ for encoder and $q$ for prior/decoder), while confirming mathematical internal consistency.

## [2026-06-30] Ingest | 一文详解 codebook 技术史 (From VAE to VQ/RQ-VAE)
- Ingested raw source `raw/recommendation/一文详解codebook技术史.md`, along with direct analysis of `raw/Recommendation/Generating Diverse High-Fidelity Images with VQ-VAE-2.pdf` and `raw/Recommendation/Autoregressive Image Generation with Residual Quantization.pdf`.
- Created detailed entity page `[[Variational Autoencoder]]` at `wiki/entities/Variational Autoencoder.md` providing geometric proofs of Jensen's inequality (concave version), KL divergence non-negativity ($KL \ge 0$), and a complete derivation of the closed-form Gaussian KL loss.
- Created detailed entity page `[[VQ-VAE]]` at `wiki/entities/VQ-VAE.md` highlighting how deterministic one-hot posteriors and uniform priors reduce KL divergence to a constant ($\log K$), completely curing posterior collapse. Detail the Straight-Through Estimator (STE) and its PyTorch implementation.
- Created detailed entity page `[[VQ-VAE-2]]` at `wiki/entities/VQ-VAE-2.md` explaining the multi-scale spatial hierarchy (Top/Bottom), detailing original Algorithms 1 and 2 in LaTeX, explaining gated PixelCNN prior conditioning, and discussing starting tokens and the connection to Diffusion models.
- Created detailed entity page `[[RQ-VAE]]` at `wiki/entities/RQ-VAE.md` outlining how recursive residual quantization decomposes continuous features into coordinate tuples $(k_1, \dots, k_D)$ to scale latent capacity to $K^D$ without index collapse. Detail the two-transformer RQ-Transformer (Spatial/Depth), and explain how Soft-Labeling and Stochastic Sampling mitigate exposure bias.
- Updated `wiki/index.md` with links and short descriptions for all four new entity pages.

- Created detailed entity page `[[FSQ]]` at `wiki/entities/FSQ.md` providing mathematical derivations of the bounding derivative chain, the self-regularizing saturation limits, flat scalar base-conversion projections, and a production PyTorch module.
- Created master compilation summary page `[[Codebook Technology History Summary]]` at `wiki/research/Codebook Technology History Summary.md` linking and comparing the entire evolutionary lineage (VAE, VQ-VAE, VQ-VAE-2, RQ-VAE, FSQ) with an elegant Mermaid flowchart, technical feature tables, and core math summaries.
## [2026-07-06] Ingest | Meta GRs : 万亿参数级别的生成式推荐
- Ingested paper summary on Meta's generative recommendation framework in `wiki/research/Meta GRs Summary.md`.
- Created entity page for the custom sequence transduction encoder `wiki/entities/HSTU.md`.
- Updated central catalog in `wiki/index.md`.

## [2026-07-06] Consolidate | Merged Meta GR Summary into HSTU
- Merged the entire content of `wiki/research/Meta GRs Summary.md` into `wiki/entities/HSTU.md` to prevent duplicate structures and centralize HSTU/Meta GRs research.
- Deleted obsolete summary file `wiki/research/Meta GRs Summary.md`.
- Updated central index in `wiki/index.md` to point researchers directly to the unified `[[HSTU]]` file.

## [2026-07-07] Refactor | Enhanced HSTU Unified Documentation
- Structured the entire backbone layers breakdown under the title "### HSTU Encoder Block Layers".
- Resized the physical ASCII paradigm diagram to respect a strict maximum display width of 78 characters.
- Extracted and cropped Page 4's Figure 3 structure visual into `wiki/media/HSTU-fig3_architecture.png` and embedded it natively.
- Cleaned up duplicate index pointers to ensure a pristine and non-fragmented entry state in `wiki/index.md`.

## [2026-07-08] Ingest | ByteDance RankMixer Architecture (Part 1)
- Started in-depth analysis of `raw/Recommendation/RankMixer.pdf`.
- Created structured summary page `wiki/research/RankMixer Summary.md` with absolute media embedding of `/Users/louie/AppleRepo/cryptic_llm_garden/wiki/media/architecture_RankMixer_block.png`.
- Analyzed and documented low MFU bottlenecks, overall pipeline formulation, and tokenization layer mechanics (Group-and-split, linear projections, and heterogeneous continuous/sequential feature preprocessing).
- Updated central index page `wiki/index.md`.

## [2026-07-08] Ingest | ByteDance RankMixer Complete Methodology
- Completed analysis and final documentation of `raw/Recommendation/RankMixer.pdf`.
- Added **Section 4: Multi-Head Token Mixing** detailing splitting, transposition, recombination, and why setting $H = T$ is mathematically required for residual dimension alignment.
- Added **Section 5: Per-Token FFN (PFFN)** explaining parameter isolation and why it prevents inter-feature-space domination.
- Added **Section 6: Sparse MoE** outlining the mechanics of **ReLU Routing** (gating, $L_1$ penalty) and **Dense-Training/Sparse-Inference (DTSI-MoE)** to prevent expert under-training.
- Added **Section 7: Scaling Laws** mathematical formulations.
- Generated and registered cross-cutting entities: `[[RankMixer]]` and `[[Model Flops Utilization]]` under the core catalog.



## [2026-07-10] Ingest | ByteDance Douyin STCA Architecture
- Ingested the raw document `raw/Recommendation/字节抖音STCA-万级长序列End2End建模.md` and downloaded the complete arXiv paper `STCA_paper.pdf`.
- Created structured summary page `wiki/research/Douyin STCA Summary.md` with complete technical breakdown of Stacked Target-to-History Cross Attention (STCA), Single-Query Optimization, Request Level Batching (RLB), and Length Extrapolation (Train Sparsely, Infer Densely).
- Created reusable concept/entity pages: `[[STCA]]` and `[[Request Level Batching]]` under the core catalog.
- Updated central index page `wiki/index.md` with the new entries.

## [2026-07-22] Ingest | From Values to Tokens: An LLM-Driven Framework for Context-Aware Time Series Forecasting via Symbolic Discretization
- Ingested paper summary on the TokenCast framework for context-aware time series forecasting in .
- Created structured entity page  detailing decoupled RIN, causal vector quantization, codebook diversity loss, and vocabulary-level semantic alignment.
- Created reusable concept/entity page  charting the evolution of time-series discretization from SAX to vector quantization, and detailing its mathematical foundations and advantages.
- Updated central index page  with links and short descriptions for the new pages.


## [2026-07-22] Ingest | From Values to Tokens: An LLM-Driven Framework for Context-Aware Time Series Forecasting via Symbolic Discretization
- Ingested paper summary on the TokenCast framework for context-aware time series forecasting in `wiki/research/From Values to Tokens Summary.md`.
- Created structured entity page `wiki/entities/TokenCast.md` detailing decoupled RIN, causal vector quantization, codebook diversity loss, and vocabulary-level semantic alignment.
- Created reusable concept/entity page `wiki/entities/Symbolic Discretization.md` charting the evolution of time-series discretization from SAX to vector quantization, and detailing its mathematical foundations and advantages.
- Updated central index page `wiki/index.md` with links and short descriptions for the new pages.


## [2026-07-22] Refactor | Enhanced TokenCast Summary & Entity Pages
- Revised `wiki/research/From Values to Tokens Summary.md` to incorporate deep-dive discussions on Section 1 (multimodal context-aware TSF motivations) and Section 3 (including concatenated causal processing, de-tokenizer offline targets, and the nature of $S$ special tokens).
- Embedded the native architecture visualization `wiki/media/overview_context-aware_time_series_forecasting.png` inside the summary and entity files.
- Revised the entity page `wiki/entities/TokenCast.md` and concept page `wiki/entities/Symbolic Discretization.md` to document the exact operational dimensionality, parameter details, and on-the-fly inference characteristics of temporal-length ($L$) based Reversible Instance Normalization (RevIN/RIN) compared to Batch Normalization.

## [2026-07-23] Ingest | COMET: Codebook-based Online-adaptive Multi-scale Embedding for Time-series Anomaly Detection
- Ingested the raw document `raw/time_series_forecast/COMET.md`.
- Created structured summary page `wiki/research/COMET Summary.md` detailing Multi-scale Patch Encoding, Vector-Quantized Coreset, Local Scaling Distance scoring, Deviation-based Variable Selection, and Online Codebook Adaptation.
- Integrated the overall architecture visualization `wiki/media/Overall_architecture_of_COMET.png` inside the summary.
- Created reusable concept/entity page `wiki/entities/COMET.md` with detailed explanations of its five core architectural pillars.
- Updated central index page `wiki/index.md` with links and short descriptions for both pages.

## [2026-07-23] Ingest | Chronos-2 & CITRAS: In-Context Learning and Future Covariates in Decoder-Only Forecasting
- Ingested raw documents `raw/time_series_forecast/Chronos-2.md` and `raw/time_series_forecast/CITRAS.md`.
- Created structured summary page `wiki/research/Chronos-2 Summary.md` detailing robust scaling ($\sinh^{-1}$ transform), alternating Time/Group attention mechanisms, and the direct 21-quantile regression head. Integrated `wiki/media/chronos-2_pipeline.png` inside the summary.
- Created structured summary page `wiki/research/CITRAS Summary.md` detailing the 2D alternating attention blocks, the Key-Value (KV) Shift mechanism for lag-less future covariate alignment, and Attention Score Smoothing (EMA) for global variate-level dependency stabilization. Integrated `wiki/media/Overall_structure_of_CITRAS.png` inside the summary.
- Created comprehensive synthesis/topic page `wiki/research/In-Context Learning and Future Covariates in Decoder-Only Forecasting.md` comparing and categorizing covariate-informed decoder architectures, including TimesFM 2.5's `XReg` decoupled linear-residual modes (`xreg + timesfm` vs. `timesfm + xreg`).
- Created reusable concept/entity pages:
  - `wiki/entities/Group Attention.md` documenting task-group isolation and permutation-invariant spatial aggregation.
  - `wiki/entities/KV Shift.md` documenting temporal-spatial decoupled query-key aligned attention.
  - `wiki/entities/Attention Score Smoothing.md` documenting temporal EMA smoothing over inter-variable attention scores.
  - `wiki/entities/TimesFM XReg Modes.md` documenting the mathematical and architectural pipeline of TimesFM 2.5's decoupled residual and error correction models.
- Updated central index page `wiki/index.md` with links and short descriptions for all new pages.

## [2026-07-29] Ingest | SwiGLU FFN: Feed-Forward Network and Activation Evolution
- Created a dedicated, high-fidelity concept page `wiki/entities/SwiGLU FFN.md` detailing the role of the FFN as a "Memory Layer", the mathematical evolution from ReLU/GELU to SwiGLU, parameter dimension alignment (scaling intermediate dimension to 8/3 d), and PyTorch implementation.
- Ingested raw insights on FFN mechanisms from `raw/LLM/大模型原理与架构/03_components/3.4_feedforward.md` and Llama innovations from `raw/LLM/大模型原理与架构/13_decoder_models/13.2_llama.md`.
- Updated central index page `wiki/index.md` to link and describe the new entity.

## [2026-08-03] Ingest | Do Transformers Really Perform Bad for Graph Representation? (Graphormer)
- Downloaded and analyzed the foundational Graphormer paper at `raw/temporal_spatial_transformer/2106.05234.pdf`.
- Created structured summary page `wiki/research/Graphormer Summary.md` detailing Pre-LN, Centrality Encoding, Spatial Encoding (via Shortest Path Distance), Path-based Edge Encoding, and the virtual `[VNode]` graph readout mechanism.
- Created dedicated entity concept page `wiki/entities/Graphormer.md` documenting the formal deconstruction of centrality-augmented attention query-key products into four distinct semantic-structural terms, and comparing discrete spatial lookup biases with continuous linear multipliers.
- Updated central index page `wiki/index.md` with links and short descriptions for both pages.

## [2026-08-05] Ingest | Collaborative Retrieval for Large Language Model-based Conversational Recommender Systems
- Ingested raw paper at `raw/ConversationalRecommendation/CRAG.md`.
- Created structured summary page `wiki/research/CRAG Summary.md` detailing CRAG's overall three-stage pipeline (Entity Link, Context-Aware Collaborative Retrieval, Reflect & Rerank) and empirical findings comparing it to zero-shot and naive text-RAG baselines.
- Created dedicated entity page `wiki/entities/CRAG.md` documenting the exact inputs, outputs, formal query rewriting, and LLM-based reflection filters of each stage.
- Created dedicated entity page `wiki/entities/EASE.md` documenting the complete mathematics of Steck's Embarrassingly Shallow Autoencoder, deriving its closed-form regularized optimization under a zero-diagonal constraint, and detailing how CRAG adapts it to asymmetric catalog mapping.
- Updated central index page `wiki/index.md` with links and short descriptions for all three new pages.

## [2026-08-06] Ingest | Multi-Type Context-Aware Conversational Recommender Systems via Mixture-of-Experts
- Ingested raw paper at `raw/ConversationalRecommendation/MCCRS.md`.
- Created structured summary page `wiki/research/MCCRS Summary.md` detailing MCCRS's multi-expert architecture (Conversation Expert, Graph Expert, Review Expert) and the ChairBot routing-fusion mechanism.
- Created dedicated entity page `wiki/entities/MCCRS.md` documenting inputs, outputs, mathematical representations, routing weight calculations, and response-bias decoder injections.
- Created dedicated entity page `wiki/entities/R-GCN.md` documenting the formal multi-relational message-passing mechanics, overparameterization mitigation methods (basis and block-diagonal decompositions), and downstream fine-tuning objective functions.
- Updated central index page `wiki/index.md` with links and short descriptions for all three new pages in strict alphabetical order.

## [2026-08-06] Ingest | Generative Conversational Recommender System
- Ingested raw paper at `raw/ConversationalRecommendation/GCRS_paper.md`.
- Created structured summary page `wiki/research/GCRS Summary.md` detailing the GCRS unified next-token framework, Semantic ID construction, structured factorization, QLoRA fine-tuning parameters, and evaluation protocols. Integrated the diagram `wiki/media/Overview_of_GCRS_framework.png` at the top of the summary.
- Created dedicated entity page `wiki/entities/GCRS.md` documenting exact metadata formatting, 4-digit code sequence wrapping, `MODE` control tokens, NF4-quantized token embeddings training configuration, and constrained beam search logic.
- Created dedicated entity page `wiki/entities/RQ-VAE.md` documenting the mathematics of multi-stage residual vector quantization, straight-through estimators (STE), codebook updates, commitment loss functions, and its application as a semantic tokenizer.
- Updated central index page `wiki/index.md` with links and short descriptions for all three new pages in strict alphabetical order.

## [2026-08-07] Ingest | CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation
- Ingested raw paper at `raw/ConversationalRecommendation/CoLLM_paper.md`.
- Created structured summary page `wiki/research/CoLLM Summary.md` detailing the integration of low-rank collaborative embeddings into continuous LLM token input space, prompt placeholder construction, dual-phase training (LoRA task learning then CIE alignment), and performance in warm vs. cold-start scenarios. Integrated the diagram `wiki/media/architecture_of_CoLLM.png` in the summary.
- Created dedicated entity page `wiki/entities/CoLLM.md` documenting prompt variables, continuous embedding interception equations, low-rank prior constraints compared to naive `w/ UI-token` direct vocabulary mappings, and mathematical formulations of the CIE alignment layers.
- Updated central index page `wiki/index.md` with links and short descriptions for all new pages in strict alphabetical order.

## [2026-08-08] Ingest | LLaRA: Large Language-Recommendation Assistant
- Ingested raw paper at `raw/ConversationalRecommendation/LLaRA_paper.md`.
- Created structured summary page `wiki/research/LLaRA Summary.md` detailing the LLaRA sequential recommendation framework, hybrid token construction (concatenating textual and behavioral tokens), list-wise ranking prompts, curriculum tuning phases, and mathematical loss formulations. Integrated both `wiki/media/LLaRA_framewrok.png` and `wiki/media/LLaRA_item_representation.png` diagrams at proper positions.
- Created dedicated entity page `wiki/entities/LLaRA.md` documenting hybrid token representations, 2-layer MLP coordinate mapping via the SR2LLM projector, easy (text-only) vs. hard (hybrid) training objectives, and the curriculum scheduler $p(\tau)$ probability progression.
- Updated central index page `wiki/index.md` with links and short descriptions for all new pages in strict alphabetical order.

## [2026-08-08] Ingest | Conversational Recommender Systems (CRS) Synthesis Page
- Created a comprehensive master synthesis and topic page at `wiki/research/Conversational Recommender Systems.md` tracking the evolution of the 5 studied models (CoLLM, LLaRA, CRAG, MCCRS, GCRS) on aspects like item representations, modality fusion strategies, output paradigms, training schemes, and core technical learnings (such as structured vs implicit generation, low-rank prior constraints, and curriculum prompt schedules).
- Integrated a Mermaid flowchart illustrating the taxonomic paradigms of LLMRec (Aligned Soft-Prompts, Collaborative Retrieval, Multi-Expert Modular Fusion, and Fully Generative Autoregressive).
- Updated central index page `wiki/index.md` with links and short descriptions for the new synthesis page.

## [2026-08-08] Ingest | CTRL: Connect Collaborative and Language Model for CTR Prediction
- Ingested raw paper at `raw/ConversationalRecommendation/CTRL_paper.md`.
- Created structured summary page `wiki/research/CTRL Summary.md` detailing the two-stage model-agnostic alignment scheme, structured prompt construction rules, symmetric global contrastive loss, fine-grained sub-space project-alignment layers, and downstream supervised fine-tuning. Integrated the diagram `wiki/media/illustration_of_CTRL.png` at a proper position.
- Created dedicated entity page `wiki/entities/CTRL.md` documenting punctuation syntax, dual-tower encoders (RoBERTa and lightweight tabular skeleton architectures), dual-directional InfoNCE partition functions, and the math of multi-aspect sub-space maximum correlation alignment.
- Updated central index page `wiki/index.md` with links and short descriptions for all new pages in strict alphabetical order.

## [2026-08-08] Ingest | A-LLMRec: Large Language Models meet Collaborative Filtering
- Ingested raw paper at `raw/ConversationalRecommendation/A-LLMRec_paper.md`.
- Created structured summary page `wiki/research/A-LLMRec Summary.md` detailing the training-free dual-stage alignment pipeline, SBERT/SASRec alignment, reconstruction losses to prevent representation collapse/over-smoothing, LLM soft-prompting, target-masked token cross-entropy training, and resource efficiency. Integrated the diagram `wiki/media/overview_of_A-LLMRec.png` at a proper position.
- Created dedicated entity page `wiki/entities/A-LLMRec.md` documenting Stage-1 latent space expectation MSE mapping, reconstruction autoencoders, recommendation losses, Stage-2 coordinate projection, prompt templates, and causal target masking.
- Updated central index page `wiki/index.md` with links and short descriptions for all new pages in strict alphabetical order.

## [2026-08-09] Ingest | SeqLLM: Augmenting LLMs with Behavioral-Sequence Modeling for High-Stakes Decisions at WeChat Pay
- Ingested raw paper at `raw/ConversationalRecommendation/SeqLLM_paper.md`.
- Created structured summary page `wiki/research/SeqLLM Summary.md` detailing the field-level discrete behavioral vocabulary, shared 2-layer MLP projector ($g_{\psi}$), Stage-1 (Translation) and Stage-2 (Reasoning) alignment curriculum, Prefix-Guided Capability Injection, and industrial risk-screening deployments. Integrated the diagram `wiki/media/overview_of_SeqLLM.png` at a proper position.
- Created dedicated entity page `wiki/entities/SeqLLM.md` documenting field-level factorization, semantic-aware embedding rescaling mathematical formulations, translation vs. reasoning SFT templates, global CPT vs. prefix-guided SFT losses, and unified capability-injection dataset optimization.
- Updated central index page `wiki/index.md` with links and short descriptions for all new pages in strict alphabetical order.

## [2026-08-09] Ingest | LC-Rec: Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation
- Ingested raw PDF paper at `raw/ConversationalRecommendation/LC-Rec_paper.pdf`.
- Created structured summary page `wiki/research/LC-Rec Summary.md` detailing the LLaMA-based contextual metadata encoding, RQ-VAE discrete item indexing, Sinkhorn-Knopp-based optimal transport collision resolution, and the multi-task alignment-tuning curriculum (symmetric prediction, cross-modal explicit mapping, asymmetric predictions, user intent search, and personalized preference inference).
- Created dedicated entity page `wiki/entities/LC-Rec.md` documenting contextual hidden states mean-pooling, multi-level vector quantization formulas, optimal transport uniform distribution constraints, Sinkhorn-Knopp normalizations, and conditional sequence-to-sequence loss functions.
- Updated central index page `wiki/index.md` with links and short descriptions for all new pages in strict alphabetical order.

## [2026-08-13] Ingest | Deep & Cross Network (DCN)
- Ingested raw paper at `raw/Recommendation/DCN_paper.md`.
- Created structured summary page `wiki/research/DCN Summary.md` detailing the embedding & stacking layer, cross network (residual connection, original-input read every layer, rank-one vector weight trick, linear-in-d complexity), deep network, combination layer, Theorem 3.1 polynomial approximation with a worked degree-growth example, FM generalization via parameter sharing, efficient block-diagonal projection, and experimental results on Criteo / forest covertype / Higgs. Integrated diagrams `wiki/media/DCN_network.png` and `wiki/media/DCN_cross_layer_vasualization.png` at proper positions.
- Created dedicated entity page `wiki/entities/DCN.md` documenting the four architectural pillars, key properties table, theoretical analysis, and experimental results.
- Updated central index page `wiki/index.md` with links and short descriptions for the new pages in strict alphabetical order.

## [2026-08-13] Ingest | Deep Structured Semantic Models (DSSM)
- Ingested raw PDF paper at `raw/Recommendation/DSSM_paper.pdf`.
- Created structured summary page `wiki/research/DSSM Summary.md` detailing the DNN semantic-feature architecture (tanh layers, cosine relevance scoring), letter-trigram word hashing (fixed non-learned projection, 16x dimensionality reduction, OOV/morphological robustness, BOW caveat with no sequential signal), discriminative clickthrough training (softmax over clicked doc + 4 sampled negatives), implementation details, and experimental results/ablations on the 16,510-query web ranking set. Integrated the diagram `wiki/media/DSSM_illustration.png` at a proper position.
- Created dedicated entity page `wiki/entities/DSSM.md` documenting the four architectural pillars, key properties table, training setup, and experimental results.
- Updated central index page `wiki/index.md` with links and short descriptions for the new pages in strict alphabetical order.

## [2026-08-16] Concept | Sinkhorn-Knopp Algorithm
- Created dedicated concept page `wiki/entities/Sinkhorn-Knopp Algorithm.md` covering doubly stochastic matrices and the Birkhoff polytope, Sinkhorn's theorem, the core alternating row/column normalization procedure, the numerically stable log-domain variant, the entropy-regularized Optimal Transport connection (Cuturi 2013), and applications in mHC (Birkhoff projection) and LC-Rec (collision-free codebook assignment).
- Added cross-links from `wiki/entities/mHC.md` (section heading) and `wiki/entities/LC-Rec.md` (two in-text mentions) to the new concept page.
- Updated central index page `wiki/index.md` with the new entity entry in alphabetical order.

## [2026-08-16] Refine | Sinkhorn-Knopp in LC-Rec (adoption details)
- Expanded the LC-Rec section of `wiki/entities/Sinkhorn-Knopp Algorithm.md` with the precise adoption analysis worked out in discussion: the full OT problem construction (decision variable Q, ground cost, uniform column constraint as the mathematical form of "collision-free"), why row entries are never uniform (cost term dominates entropy; the exp kernel amplifies distance gaps), the capacity semantics of |B|/K (uniqueness only when |B| ≤ K), the training-vs-inference two-stage usage, and the precise equivalence "greedy = OT optimum when the greedy solution already satisfies the column constraint" (holding at the argmax level for small temperature).
- Added a self-consistent worked example (|B|=4 items under prefix a0-b1-, K=8 codewords, capacity 0.5) showing greedy collision on c0 and the capacity-driven fan-out to distinct codewords after Sinkhorn + argmax.
- Noted that the paper does not specify whether conflict-group redistribution restricts the codebook to unoccupied codewords (implementation detail left to the released code).

## [2026-08-16] Refine | mHC (n-expansion example + why constrain H_res)
- Added a worked example to `wiki/entities/mHC.md` (HC section) showing how the expansion rate $n$ widens the residual stream: with $C=3$, $n=2$, a token becomes $n=2$ parallel 3-dim streams (6-dim), and demonstrating the roles of $\mathcal{H}^{\mathrm{pre}}$ (collapse $n$ streams to $C$ for $\mathcal{F}$), $\mathcal{H}^{\mathrm{post}}$ (spread output back), and $\mathcal{H}^{\mathrm{res}}$ (cross-stream mixing). Emphasized that $n$ widens the stream but not $\mathcal{F}$'s compute (~zero extra FLOPs), and that the example's $\mathcal{H}^{\mathrm{res}}$ is intentionally not doubly stochastic to foreshadow mHC's fix.
	- Added a "Why the Constraint Lands on $\mathcal{H}^{\mathrm{res}}$" subsection to the mHC section. After two rounds of feedback, tightened it into a terse, one-idea-per-line style that leads with the single essential point — **$\mathcal{H}^{\mathrm{res}}$ is the only matrix multiplied across layers, and that compounding is what breaks training** — and re-inserted the HC recursion with $\underbrace{}$ annotations ("only $\mathcal{H}^{\mathrm{res}}$" / "again a product") so the claim is grounded in the formula. Then briefly: sigmoid suffices for the once-per-layer $\mathcal{H}^{\mathrm{pre}}$/$\mathcal{H}^{\mathrm{post}}$; doubly-stochastic closure under multiplication bounds the product at any depth (mHC ~1.6 vs HC ~3000); pinning to identity would kill cross-stream mixing; Sinkhorn-Knopp is the differentiable, cheap projection.

## [2026-08-17] Ingest | Neural Collaborative Filtering (NCF)
- Ingested raw paper at `raw/Recommendation/NCF_paper.md`.
- Created structured summary page `wiki/research/NCF Summary.md` detailing the implicit-feedback problem setup and MF inner-product limitation (Jaccard ground-truth counterexample), the general NCF framework (ID one-hot → embedding lookup → neural CF layers → sigmoid), the probabilistic log-loss + negative-sampling treatment, the three instantiations (GMF, MLP, NeuMF with separate embeddings and pre-training), and experimental findings (NeuMF > MLP > GMF; ~3–6 negative sampling optimum; depth helps via non-linearity). Integrated the three diagrams `wiki/media/NCF_fig1_example_illustrates_MF_limitation.png`, `wiki/media/NCF_general_framework.png`, and `wiki/media/NCF_neural_matrix_factorization_model.png` at proper positions.
- Created dedicated entity page `wiki/entities/NCF.md` documenting the four architectural pillars, the three-model comparison table, key properties, training details, and experimental results.
- Updated central index page `wiki/index.md` with links and short descriptions for the new pages in strict alphabetical order.

## [2026-08-17] Refactor | Absorb MF & Implicit Feedback into NCF Summary
- The standalone concept pages `wiki/entities/Matrix Factorization.md` and `wiki/entities/Implicit Feedback.md` were redundant with the NCF summary's coverage. Folded their unique content — the MF-variant loss-strategy table (WMF/BPR/eALS vs. log loss) and the leave-one-out evaluation protocol — into Section 1.1 of `wiki/research/NCF Summary.md`, then deleted both pages.
- Fixed the resulting dangling wikilinks in `wiki/entities/NCF.md` and `wiki/research/NCF Summary.md` (plain-text mentions now, with the summary as the canonical reference; added an `[[EASE]]` pointer for the linear-CF alternative).
- Removed the two corresponding entries from `wiki/index.md`.

## [2026-08-17] Ingest | Deep Interest Network (DIN)
- Ingested raw paper at `raw/Recommendation/DIN_paper.md`.
- Created structured summary page `wiki/research/DIN Summary.md` detailing the fixed-length vector bottleneck in Embedding&MLP, multi-group categorical feature representation (one-hot vs. multi-hot), the local activation unit (candidate-aware weighted pooling with outer product input, softmax deliberately removed to preserve interest intensity), mini-batch aware regularization (3-step derivation, per-batch L2 approximation with inverse-frequency weighting), Dice activation function (data-adaptive generalization of PReLU with sigmoid control at input mean), negative LSTM result, and experimental results on Amazon/MovieLens/Alibaba datasets (+11.65% RelaImpr, +10% CTR online). Integrated the three diagrams `wiki/media/DIN_feature_representation.png`, `wiki/media/DIN_network_architecture.png`, and `wiki/media/DIN_PReLU_and_Dice.png` at proper positions.
- Created dedicated entity page `wiki/entities/DIN.md` documenting the four architectural pillars, key properties table, and experimental results.
- Updated central index page `wiki/index.md` with links and short descriptions for the new pages in strict alphabetical order.

## [2026-08-18] Ingest | Search-based User Interest Model (SIM)
- Ingested raw paper at `raw/Recommendation/SIM_paper.md`.
- Created structured research summary page `wiki/research/SIM Summary.md` detailing the lifelong sequential user behavior problem (up to 54,000 items), the cascaded two-stage search paradigm (General Search Unit [GSU] and Exact Search Unit [ESU]), the distinction and formulation of Hard-Search (non-parametric category matching) vs. Soft-Search (parametric MIPS/ALSH with auxiliary CTR objective for distribution shift), temporal distance modeling (time interval embedding $\mathbf{D}$), and industrial online co-design via the 2-level User Behavior Tree (UBT) index (~22 TB) and GPU kernel fusion. Embedded architectural workflow diagram `wiki/media/SIM_overall_workflow.png` at Section 2.
- Created dedicated entity page `wiki/entities/SIM.md` documenting the four core architectural pillars, mathematical formulations, key properties table, and experimental/business highlights (+7.1% CTR, +4.4% RPM online).
- Updated central catalog `wiki/index.md` with `[[SIM Summary]]` and `[[SIM]]` entries.

## [2026-08-21] Ingest | TWo-stage Interest Network (TWIN)
- Ingested raw paper at `raw/Recommendation/TWIN_paper.md` (KDD 2023, Kuaishou).
- Created structured research summary page `wiki/research/TWIN Summary.md` detailing the GSU–ESU inconsistency problem (SIM Hard's top-100 hits only ~40 of ESU's real top-100), the three highlighted innovations — (①) behavior feature splitting by cacheability (inherent $K_h$ cached vs. cross $K_c$ compressed to 1-dim via block-diagonal $W^c$, with the two equivalent views and a worked $H=200/C=24$ example), (②) cross features as additive bias terms in the attention score $\boldsymbol{\alpha}$, (③) the shared "twin" attention score with the exact shared/unshared parameter table — plus the single-loss training design (no auxiliary retrieval/consistency objectives, gradient path through ESU), embedding-layer worked examples (one-hot/multi-hot lookup, dim-64-vs-8 rationale), the upper mixer (top-right of Figure 2, with the note that short-term and other-task modules are unspecified), system deployment (8-min nearline training, 15-min projection cache refresh over 8B videos covering 97% of requests, 99.3% bottleneck reduction), and full experimental results (offline AUC/GAUC, 94/100 consistency, length scaling, ablations, online Watch Time A/B). Embedded architecture diagram `wiki/media/TWIN_architecture.png` at Section 2.
- Created dedicated entity page `wiki/entities/TWIN.md` documenting the six architectural pillars, key properties table, and experimental/business highlights.
- Updated central catalog `wiki/index.md` with `[[TWIN Summary]]` and `[[TWIN]]` entries.

## [2026-08-24] Ingest | LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders
- Ingested raw paper at `raw/Recommendation/LONGER.md` (RecSys 2025, ByteDance).
- Created structured research summary page `wiki/research/LONGER Summary.md` detailing the end-to-end ultra-long (10K) sequence paradigm (vs. two-stage retrieval of SIM/TWIN), Global Tokens (anchor + attention-sink stabilizer), Token Merge with InnerTrans (L→L/K, ~43% FLOPs cut, parameter expansion as capacity), the hybrid attention core (cross-causal compression of queries O=[G;H_S] over full R=[G;H], then N self-causal layers over the m+k working set), query sampling ablation (recent-k beats learnable/uniform), dual positional encodings (time-diff concat + learnable add), system co-design (synchronous GPU training with hierarchical HBM/MEM/SSD embedding storage, mixed precision + recompute, KV cache serving −40%→−6.8%), offline/online results (AUC 0.85290, up to +2.15% ADVV, +7.92% Order/U), and industrial scaling laws (power law in length/params/FLOPs). Embedded architecture diagram `wiki/media/LONGER_model_architecture.png` at Section 2.
- Created dedicated entity page `wiki/entities/LONGER.md` documenting the five architectural pillars, key properties table, and experimental/business highlights.
- Created new entity page `wiki/entities/Attention Sink.md` for the StreamLLM effect that LONGER's global tokens counteract.
- Updated central catalog `wiki/index.md` with `[[LONGER Summary]]`, `[[LONGER]]`, and `[[Attention Sink]]` entries. Cross-linked against `[[SIM]]`, `[[TWIN]]`, `[[STCA]]`, `[[HSTU]]`, `[[KV Cache]]`, `[[Positional Encoding]]`.

## [2026-08-31] Ingest | 第十五章预训练大语言模型 (Part 1: Data Sources & Distribution)
- Read raw source `raw/LLM/第十五章预训练大语言模型.md` (Transformers 快速入门, based on 赵鑫等《大语言模型》2024).
- Created `wiki/research/Pre-training Data Sources and Distribution.md` covering §15.1's data-source taxonomy (general vs. specialized text: multilingual / scientific / code) and Figure 15-1's pre-training data distribution across 15 representative LLMs (T5, Falcon, LLaMA, GPT-3, Yi, MT-NLG, Gopher, Chinchilla, GLaM, PaLM, LaMDA, Galactica, GPT-NeoX, CodeGen, AlphaCode), with the legend's datasets listed in a categorized table (C4, OpenWebText, Wikipedia, the Pile subsets, BookCorpus, CC-Stories-R, CC-NEWES, BigQuery).
- Downloaded Figure 15-1 and saved it locally as `wiki/media/pretrain_data_dist_of_llms.jpg`, embedded in the new page.
- Updated `wiki/index.md` with the new page entry.

## [2026-08-31] Rename | Pre-training Data Sources and Distribution → Pre-training Large Language Models
- Renamed `wiki/research/Pre-training Data Sources and Distribution.md` to `wiki/research/Pre-training Large Language Models.md` so the page serves as the growing Chapter 15 summary (parts appended one by one) rather than a Part-1-only page.
- Restructured the page: chapter-level H1 + intro, Part 1 content moved under a "Part 1: Data Preparation — Sources & Distribution" section; old title kept as an alias in frontmatter.
- Updated the entry in `wiki/index.md`.

## [2026-08-31] Ingest | 第十五章预训练大语言模型 (Part 2: Data Preprocessing)
- Appended "Part 2: Data Preprocessing (数据预处理, §15.1.1)" to `wiki/research/Pre-training Large Language Models.md`: the four-step pipeline — quality filtering (heuristic rules with concrete statistical/keyword examples; classifier-based options FastText/BERT/GPT-4 with efficiency trade-offs and the cascade strategy), sensitive-content filtering (Jigsaw toxicity classifier; Dolma PII rules with the <5 replace / ≥6 delete policy), deduplication (granularity levels; exact matching via suffix arrays, approximate matching via MinHash/LSH), and tokenization (BPE/WordPiece/Unigram, custom SentencePiece tokenizers) — plus the quantity/quality/repetition/privacy impact summary (incl. hallucination and double descent).
- Downloaded Figure 15-2 (preprocessing pipeline) as `wiki/media/pretrain_data_preprocess.jpg` and embedded it in the new section.
- Updated `wiki/index.md` entry for the page.

## [2026-08-31] Rework | Pre-training LLMs Part 2 restructured as table
- Rewrote Part 2 (Data Preprocessing) of `wiki/research/Pre-training Large Language Models.md` per user feedback (wall of text): the four pipeline steps (quality filtering, sensitive-content filtering, deduplication, tokenization) are now summarized in a single table with columns Step / Problem to be solved / Methodology / Handful tools; kept the "Why Data Quantity and Quality Matter" bullets.

## [2026-08-31] Create | Entities: Perplexity, Locality-Sensitive Hashing, Tokenizer
- Created three entity pages from the Q&A discussion during the Chapter 15 ingest:
  - `wiki/entities/Perplexity.md` — PPL definition, fluency-vs-factuality caveats, dual role as LM evaluation metric and data-filtering heuristic.
  - `wiki/entities/Locality-Sensitive Hashing.md` — LSH vs. ordinary hashing, MinHash's minimum-signature trick and why P(collision)=Jaccard (randomness as the sampling mechanism), banding, other LSH families, role in corpus deduplication.
  - `wiki/entities/Tokenizer.md` — BPE (frequency merging, byte-level BBPE), WordPiece (PMI-style likelihood-gain merging, `##` convention), Unigram (top-down pruning with the exact ΔNLL loss definition, EM, Viterbi/sampling segmentation, subword regularization), SentencePiece framework (no pre-tokenization, `▁` escaping, byte_fallback), and Chinese tokenizer practice (Qwen/DeepSeek vs. LLaMA).
- Applied the entities in `wiki/research/Pre-training Large Language Models.md`: wikilinks added inside the Part 2 pipeline table (Perplexity in quality filtering, LSH in dedup tools, Tokenizer in tokenization) and in the Related Pages section.
- Added all three entries to `wiki/index.md` under Entities & Concepts.

## [2026-08-31] Ingest | 第十五章预训练大语言模型 (Part 4: Long-Context Modeling)
- Appended "Part 4: Long-Context Modeling (长上下文模型, §15.2.2)" to `wiki/research/Pre-training Large Language Models.md`, ahead of the still-pending Part 3 (mainstream architectures, §15.2.1) for which a placeholder was inserted:
  - Direction 1 — extending position encodings: RoPE's poor unmodified extrapolation, position interpolation/truncation; extrapolation-capable encodings (T5 bias, ALiBi, xPos) with the generation-vs-understanding caveat.
  - Direction 2 — restructuring the context window: three methods (parallel context windows, Λ-shaped window, token selection with kNN/chunk-similarity flavors) summarized in a table with mechanisms and drawbacks.
- Downloaded Figure 15-5 as `wiki/media/pretrain_context_window_methods.jpg` and embedded it in the section.
- Cross-linked `[[RoPE]]` and `[[ALiBi]]` from the new section; added "Related Pages" sections to both entity pages linking back.
- Updated `wiki/index.md` entry for the page.

## [2026-08-31] Create | Entity: Long-Context Positional Encoding
- Created `wiki/entities/Long-Context Positional Encoding.md` consolidating the long-context discussion: the RoPE relative-yet-bounded paradox (low-frequency subspace OOD with a d=128/b=10000 numeric table), the complete RoPE formula with notation pinned (m = absolute position, θ_i = b^{-2i/d}), PI and sliding-window truncation, the NTK-aware → Dynamic NTK → YaRN evolution (frequency ramps, per-band interpolation, temperature correction), base enlargement (LLaMA-3 500K, Qwen2 1M), a six-method comparison table, and the DeepSeek-V4 1M case study (Partial RoPE on last 64 dims + output de-rotation; CSA+HCA compression with lightning indexer, sliding-window branch, attention sink; 4K→16K→64K→1M curriculum with dense→sparse staging; honest 128K/512K limits).
- Referenced the entity from `wiki/research/Pre-training Large Language Models.md` (Part 4 direction-1 paragraph + Related Pages), and added reciprocal cross-links on `wiki/entities/RoPE.md` and `wiki/entities/ALiBi.md`.
- Added the entry to `wiki/index.md` under Entities & Concepts.

## [2026-08-31] Ingest | 第十五章预训练大语言模型 (Parts 3, 5, 6 — chapter complete)
- Completed the remaining parts of `wiki/research/Pre-training Large Language Models.md`:
  - Part 3: Mainstream Architectures (§15.2.1) — encoder-decoder vs. causal decoder vs. prefix decoder attention patterns table (Flan-T5 / GPT / GLM-130B, U-PaLM), with Figure 15-4 embedded.
  - Part 5: Data Scheduling (§15.1.2) — data mixing (LLaMA reference mix, diversity / learned-mixture / capability-targeting strategies) and data curricula table (CodeLLaMA 2T→500B, CodeLLaMA-Python +100B, Llemma +50–200B math with 5% general regularization, CodeLLaMA 4K→16K long-context).
  - Part 6: Model Pre-training (§15.3) — objectives table (LM, Prefix LM, FIM, DAE, MoD/UL2 with S/R/X denoisers), optimization settings (dynamic batch ramp, warmup+decay LR with GPT-3/LLaMA peaks, Adam/AdamW/Adafactor hyperparameters, gradient clipping 1.0, checkpoints, weight decay 0.1), and scalable training (3D parallelism table, ZeRO/FSDP, FP16 vs. BF16 mixed precision), with Figures 15-7 and 15-8 embedded.
- Downloaded Figures 15-4, 15-7, 15-8 as `wiki/media/pretrain_architecture_of_llms.jpg`, `pretrain_lr_decay_strategy.jpg`, `pretrain_parallel_training.jpg`.
- Updated the page intro (all parts covered) and the `wiki/index.md` entry to reflect the complete six-part summary.
