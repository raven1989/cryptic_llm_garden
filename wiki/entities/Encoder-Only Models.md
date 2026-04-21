---
tags:
  - llm
  - architecture
  - transformers
date: 2026-04-16
---

# Encoder-Only Models

Encoder-only models are a class of [[Transformers]] designed primarily to understand and encode input text by converting it into rich, contextual representations. 

## Key Characteristics
*   **Bidirectional Attention:** Unlike decoder models, encoder-only models look at the entire input sequence simultaneously. This allows them to capture deep contextual meaning from both the left and right sides of any given token.
*   **Training Methodology:** Usually trained using **Masked Language Modeling (MLM)** (where random tokens are masked and the model must predict them) and sometimes **Next Sentence Prediction (NSP)**.
*   **Strengths:** Highly effective at tasks requiring deep comprehension of the input sequence.

## Common Use Cases
*   Text Classification (e.g., sentiment analysis)
*   Named Entity Recognition (NER)
*   Semantic Similarity
*   Generating embeddings for [[Vector Database]] retrieval.

## Notable Examples
*   **BERT** (Bidirectional Encoder Representations from Transformers)
*   **RoBERTa** (Robustly optimized BERT approach)
*   **DistilBERT** (A smaller, faster distillation of BERT)

## References
*   [[Encoder Only, Encoder Decoder, And Decoder Only Models]]
