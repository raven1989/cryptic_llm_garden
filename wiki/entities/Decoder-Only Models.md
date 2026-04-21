---
tags:
  - llm
  - architecture
  - transformers
date: 2026-04-16
---

# Decoder-Only Models

Decoder-only models are a class of [[Transformers]] built exclusively for **generating text** based on a given prompt. They have become the dominant architecture for modern Large Language Models (LLMs).

## Key Characteristics
*   **Unidirectional Attention (Causal Masking):** During generation and training, decoder models use a masked self-attention mechanism. This prevents the model from "peeking" at future words; it can only attend to previous tokens and the current token.
*   **Auto-regressive Generation:** They take in a sequence of tokens and predict the next token one step at a time, feeding the newly generated token back into the sequence to predict the next one. This mechanism heavily relies on the [[KV Cache]] to optimize performance.
*   **Training Methodology:** Primarily trained via **Causal Language Modeling (CLM)**, learning to predict the next token given all previous tokens. Models are then often subject to instruction tuning and alignment like [[Fine-tuning|RLHF/DPO]].

## Common Use Cases
*   Conversational AI (Chatbots)
*   Storytelling and creative writing
*   Code Generation
*   Few-shot prompting for general tasks

## Notable Examples
*   **GPT Family** (Generative Pretrained Transformer): GPT-2, GPT-3, GPT-4
*   **LLaMA / LLaMA 2 / LLaMA 3**
*   **Mistral**

## References
*   [[Encoder Only, Encoder Decoder, And Decoder Only Models]]
*   [[Transformers]]
