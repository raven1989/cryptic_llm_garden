---
tags:
  - llm
  - architecture
  - transformers
source: "[[Encoder Only, Encoder Decoder, And Decoder Only Models.md]]"
author: "[[Saptarshi Datta]]"
date: 2026-04-16
---

# Encoder-Only, Encoder-Decoder, and Decoder-Only Models

This page summarizes Saptarshi Datta's overview of the three primary macro-architectures in transformer-based Natural Language Processing.

## Overview
Transformer models generally fall into three categories based on their architecture and intended NLP tasks. The choice of architecture determines whether the model is best suited for understanding text, transforming text, or generating text.

### [[Encoder-Only Models]]
*   **Purpose:** Understanding and encoding input text into rich, contextual representations.
*   **Mechanism:** Looks at the entire input sequence simultaneously (bidirectional attention) to learn relationships between all tokens.
*   **Training:** Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).
*   **Examples:** BERT, RoBERTa, DistilBERT.
*   **Use Cases:** Classification, Named Entity Recognition (NER), semantic similarity.

### [[Encoder-Decoder Models]]
*   **Purpose:** Sequence-to-sequence transformation.
*   **Mechanism:** The encoder processes and compresses the input into a context vector. The decoder then uses cross-attention to generate an output sequence based on this vector and previously generated tokens.
*   **Training:** Denoising Autoencoding (reconstructing corrupted text) or Text-to-Text formulation.
*   **Examples:** T5, BART, mBART, MarianMT.
*   **Use Cases:** Translation, summarization.

### [[Decoder-Only Models]]
*   **Purpose:** Text generation based on a prompt.
*   **Mechanism:** Uses unidirectional (left-to-right) causal masking to predict the next token step-by-step, ensuring the model cannot "peek" at future words.
*   **Training:** Causal Language Modeling (CLM) predicting the next token, often followed by instruction tuning or RLHF.
*   **Examples:** GPT-family (GPT-3, GPT-4), LLaMA, Mistral.
*   **Use Cases:** Storytelling, code generation, conversational AI (chatbots).

## Related Concepts
*   [[Transformers]]
*   [[Self-Attention Mechanism]]
