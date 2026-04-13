---
tags: [llm, database, storage]
date: 2026-04-08
aliases: [Vector DB]
---

# Vector Database

Vector Databases are the infrastructure underpinning modern [[RAG]] systems, acting as the "intuitive memory" for Large Language Models. They store and search high-dimensional embeddings.

## Core Mechanisms

*   **Approximate Nearest Neighbor (ANN):** The algorithmic foundation for quickly finding similar vectors in massive datasets without exhaustive search.
*   **HNSW (Hierarchical Navigable Small World):** A graph-based index algorithm. Extremely fast and accurate but requires significant memory. It constructs multiple layers of interconnected nodes.
*   **IVF-PQ (Inverted File with Product Quantization):** Combines clustering (IVF) to narrow down the search space with Product Quantization (PQ) to compress the vectors themselves. Slower and slightly less accurate than HNSW but highly memory efficient.

## Engineering Challenges

*   **Single-Stage Filtering:** The critical requirement to apply scalar (metadata) filters (e.g., `year == 2023`) *while* traversing the HNSW graph, rather than as a post-processing step, to prevent the recall rate from collapsing.
*   **Metrics:** Choosing the appropriate distance metric (Cosine Similarity, L2 distance, or Inner Product).

## Notable Tools
*   **Milvus / Zilliz:** Enterprise-scale vector databases.
*   **Qdrant:** Known for advanced filtering capabilities.
*   **pgvector:** An extension turning PostgreSQL into a vector database, popular for integrated architectures.

See also: [[RAG]], [[Transformers]], [[LLM Study Plan]]
