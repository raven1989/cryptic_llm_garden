# Wiki Index

This is the content-oriented catalog of the wiki. The LLM updates this file whenever a new source is ingested or a new page is created.

## Personal Knowledge
- [[LLM Study Plan]]: A comprehensive, top-down algorithm engineering guide spanning Applications, Frameworks, and Foundation Models.
- [[My LLM Learning Sequence]]: A step-by-step, bottom-up execution plan derived from the main study outline.
- [[Layer 3 - Architectures and Operators Syllabus]]: A deep-dive syllabus for Step 1 of the learning sequence, focusing on macro-architectures, micro-designs, and inference bottlenecks.

## Research & Topics
- [[大模型位置编码-ALiBi位置编码 Summary]]: A summary of Attention with Linear Biases (ALiBi), a positional encoding technique for improving length extrapolation in Large Language Models.
- [[Encoder Only, Encoder Decoder, And Decoder Only Models]]: An overview summarizing the three primary macro-architectures in transformer-based NLP and their distinct use cases.
- [[LLM Inference VRAM and Compute Estimation]]: Mathematical formulas for calculating the precise memory footprint of the KV Cache and the computational FLOPs required during the Prefill and Decode stages.
- [[十分钟看懂RoPE Summary]]: A deep dive article detailing the mathematical proofs, long-range decay (Abel transformation), extrapolation properties, and PyTorch production code (LLaMA) for Rotary Position Embedding.
- [[Why is Attention divided by Root d_k]]: A mathematical breakdown of why the dot product of Query and Key vectors in Transformers is scaled by $\sqrt{d_k}$ to prevent variance growth and vanishing gradients.
- [[Web Agent Architecture]]: An analysis of 4 different architectural paradigms for building AI web agents.
- [[Taxy.ai Implementation]]: Code-level analysis of how Taxy.ai extracts DOM state and executes hardware-level actions.
- [[Browser-Use Implementation]]: Code-level analysis of how Browser-Use utilizes Pure Python CDP (Chrome DevTools Protocol) to execute stealthy web agents without JavaScript injection.
- [[Action Execution - JS vs CDP]]: A detailed breakdown of why modern web agents must use the Chrome DevTools Protocol to simulate hardware interrupts rather than executing simple JavaScript.

## Media Companion (Books, Movies, etc.)
- (Empty)

## Entities & Concepts (Cross-cutting)
- [[Agents]]: Workflows, State Machines, multi-agent systems, and tool calling sandboxes.
- [[ALiBi]]: Attention with Linear Biases, a positional encoding method that applies distance penalties directly to attention scores to enable length extrapolation.
- [[Decoder-Only Models]]: Transformer architectures designed exclusively for auto-regressive text generation (e.g., GPT, LLaMA).
- [[DOM State Compression]]: Techniques for stripping non-interactive elements from HTML to reduce token usage and improve privacy.
- [[Encoder-Decoder Models]]: Transformer architectures designed for sequence-to-sequence tasks like translation and summarization (e.g., T5, BART).
- [[Encoder-Only Models]]: Transformer architectures designed for understanding input text and generating contextual embeddings (e.g., BERT).
- [[Fine-tuning]]: The training pipeline, from Pre-training to SFT and RLHF/DPO alignment, plus PEFT methods like LoRA.
- [[KV Cache]]: The Key-Value cache optimization used during autoregressive generation to trade VRAM for compute speed.
- [[Positional Encoding]]: Techniques (like Sinusoidal, RoPE, and ALiBi) used to inject sequence order information into Transformers, compensating for their lack of recurrence.
- [[RAG]]: Advanced Retrieval-Augmented Generation architectures including dense/sparse retrieval, reranking, and Knowledge Graph extraction.
- [[RoPE]]: A mathematical deep dive into Rotary Position Embedding, including its block-diagonal multi-dimensional formulation, efficient Hadamard product computation, Long-Range Decay proof (Abel Transformation), and LLaMA PyTorch code snippets.
- [[Self-Attention Mechanism]]: The mathematical foundation of Transformers detailing why Query, Key, and Value matrices must use independent weights to prevent identity matrix degeneration.
- [[Transformers]]: Foundation model architectures, including MoE and SSM, along with micro-designs like FlashAttention and RoPE.
- [[Vector Database]]: Storage and search mechanisms for embeddings using algorithms like HNSW and IVF-PQ.
- [[Web Agent]]: Autonomous or semi-autonomous AI agents designed to navigate and manipulate web interfaces.
