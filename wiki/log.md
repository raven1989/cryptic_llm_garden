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