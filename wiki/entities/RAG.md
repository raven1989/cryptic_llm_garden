---
tags: [llm, application, search, knowledge-graph]
date: 2026-04-08
aliases: [Retrieval-Augmented Generation, GraphRAG]
---

# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation enables LLMs to answer queries based on a given set of source documents, rather than relying solely on its pre-trained weights. Modern RAG systems are highly complex search and recommendation engines.

## Advanced RAG Architecture

The foundation of a modern enterprise RAG system is a multi-step pipeline.

*   **Ingestion & Parsing:** Handling complex documents (PDFs, dual-column, tables) using visual LLMs and OCR (e.g., LlamaParse, MinerU).
*   **Chunking Strategies:** Moving past fixed-size chunks to Semantic Chunking (breaking text based on embedding similarity to preserve meaning) and Structural Chunking (following Markdown/HTML hierarchies).
*   **Retrieval Engine:**
    *   **Query Transformation:** Rewriting user queries to handle ambiguity (e.g., HyDE, which asks the LLM to generate a hypothetical answer first).
    *   **Dense Retrieval:** Using Embedding models (e.g., BGE, OpenAI) to capture semantic similarity.
    *   **Sparse Retrieval:** Exact matching of terminology and long-tail words using algorithms like BM25 or SPLADE.
    *   **Hybrid Search:** Combining Dense and Sparse retrieval using Reciprocal Rank Fusion (RRF).
*   **Reranking:** Passing the top-K retrieved chunks and the original query through a Cross-encoder model (e.g., BGE-Reranker, Cohere) to compute true interaction scores and filter noise.

## Knowledge Graphs & GraphRAG

Traditional Vector RAG struggles with multi-hop reasoning (connecting information across documents) and global macroeconomic questions.

*   **Graph Construction:** LLMs extract entities and relationships (triplets) from text, perform entity disambiguation (e.g., mapping "Apple" and "Apple Inc."), and store them in a Graph Database like Neo4j.
*   **Graph Retrieval:** Multi-hop traversal.
*   **Community Summarization:** Microsoft's GraphRAG approach groups the graph using the Leiden algorithm and pre-computes global summaries using an LLM. When asked macro questions, the system queries these summaries directly instead of base documents.

For a robust system, an LLM Router typically decides whether a question requires a micro-fact lookup (Vector DB) or a macro-trend synthesis (GraphRAG).

See also: [[Vector Database]], [[LLM Study Plan]]
