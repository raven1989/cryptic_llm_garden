---
aliases: [Learning Sequence, LLM Execution Plan, Study Path]
tags: [study, ai, llm, plan]
date: 2026-04-08
related: ["[[LLM Study Plan]]"]
---

# My LLM Learning Sequence

This is the step-by-step, bottom-up execution sequence derived from the broader [[LLM Study Plan]]. While the main outline maps out the field top-down (Application to Algorithms), the actual execution advice dictates building a foundational understanding first before moving up the stack.

## 1. Master Architectures and Operators (Layer 3)
*   Start at the foundation level by studying macro-architectures like [[Transformers]], Mixture of Experts (MoE), and State Space Models (SSM).
*   Dive into micro-designs (FlashAttention, RoPE) and inference optimizations like Speculative Decoding and PagedAttention.

## 2. Implement the Fine-tuning and Alignment Pipeline (Layer 3)
*   Move on to training pipelines, encompassing data engineering, Pre-training, and [[Fine-tuning]] (SFT).
*   Master alignment techniques such as RLHF and DPO, along with parameter-efficient fine-tuning (PEFT) methods like LoRA/QLoRA.

## 3. Build an Agent System (Layer 2)
*   Move up to the Frameworks layer to build [[Agents]] using the models you have fine-tuned.
*   Utilize state machines (e.g., LangGraph) and integrate memory systems like a [[Vector Database]] and advanced [[RAG]] architectures.
*   Implement tool calling and safe execution sandboxes.

## 4. Explore the Multimodal Frontier (Layer 3)
*   Return to the foundation layer to study Multimodal LMMs.
*   Focus on fusion vs. native architectures, vision encoders (CLIP, SigLIP), and dynamic resolution techniques.

*Note: The sequence emphasizes building a deep, fundamental understanding of algorithms and models first, then applying that knowledge to construct robust agentic frameworks, eventually empowering the high-level applications.*