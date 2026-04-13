---
tags: [llm, study-plan, engineering, algorithms]
date: 2026-04-08
sources: [[raw/Outline.md]], [[raw/Outline_Layer1.md]], [[raw/Outline_Layer2.md]], [[raw/Outline_Layer3.md]]
---

# LLM Full-Stack Algorithm Engineering Study Plan

This study plan outlines a comprehensive, top-down approach to mastering Large Language Models (LLMs) from the perspective of an algorithm engineer. It is structured into three layers: Top (Harness), Middle (Frameworks), and Bottom (Algorithms).

## Layer 1: Top (Harness / Application & Interaction)
The application layer focuses on adding business constraints to the LLM, managing its non-deterministic nature, and delivering concrete value.

*   **NL2X (Natural Language to Everything):** Converting language into structured commands. Includes NL2SQL (e.g., Vanna, DB-GPT), NL2Code (GitHub Copilot, Cursor), NL2API (Gorilla), and Generative UI (Vercel v0).
*   **Knowledge Discovery & Synthesis:** Enterprise search and recommendation. Uses LLMs as "super researchers."
*   **Workflow Automation:** Building towards autonomous task execution and intelligent customer support.
*   **Core Harness Skills:** Advanced prompt engineering (XML, CRISPE), AI-native interaction design (SSE, streaming), and robust output parsing/guardrails (Pydantic, NeMo Guardrails).

## Layer 2: Middle (Frameworks / AI Engineering)
This layer treats the LLM as a CPU, upgrading it with memory, knowledge, and tools to solve complex problems.

*   **[[RAG]] (Advanced RAG Architecture):** Moving beyond simple chunking to heterogeneous data ingestion, semantic chunking, hybrid search (Dense + Sparse), and reranking.
*   **Double-Brain Memory:** Utilizing [[Vector Database]]s for intuitive memory (HNSW, IVF-PQ) and Knowledge Graphs for logical, multi-hop reasoning. Includes Microsoft's GraphRAG.
*   **[[Agents]] (Agentic Workflows):** Transitioning from zero-shot to multi-step reasoning. Covers memory persistence, state machines (LangGraph), and multi-agent coordination.
*   **Tool Calling & Execution:** Schema design for APIs and safe execution environments (E2B sandboxes).
*   **Evaluation & LLMOps:** Tracing execution with tools like LangSmith and using LLM-as-a-Judge for automated evaluation (e.g., RAG Triad).

## Layer 3: Bottom (Foundation Models & Algorithms)
The core algorithmic battlefield involving model architecture, optimization, and distributed training.

*   **Macro-Architectures:** From [[Transformers]] (Decoder-only dominance) to Mixture of Experts (MoE) and State Space Models (SSM) like Mamba.
*   **Micro-Designs:** Attention variants (MQA, GQA), FlashAttention (breaking the memory wall), RoPE position embeddings, and RMSNorm.
*   **Inference & Decoding:** Sampling strategies (Top-p), PagedAttention (for KV Cache optimization), and Speculative Decoding.
*   **Training Pipeline:** Data engineering (MinHash LSH), Pre-training, and [[Fine-tuning]] (SFT). Alignment techniques include RLHF and DPO (Direct Preference Optimization).
*   **Multimodal LMMs:** Architectures (Fusion vs. Native Any-to-Any), vision encoders (CLIP, SigLIP, DINOv2), and AnyRes (dynamic resolution).
*   **Infrastructure:** 3D Parallelism (DP/ZeRO, TP, PP), PEFT techniques like LoRA/QLoRA, and Quantization (PTQ, QAT).

## Execution Advice
1.  Master architectures and operators (Layer 3).
2.  Implement the fine-tuning and alignment pipeline (Layer 3).
3.  Build an Agent system using LangGraph and fine-tuned models (Layer 2).
4.  Explore the multimodal frontier (Layer 3).
