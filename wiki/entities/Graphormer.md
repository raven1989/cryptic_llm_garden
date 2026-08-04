---
tags:
  - graph
  - architecture
  - transformers
  - attention
date: 2026-08-03
sources: ["[[raw/temporal_spatial_transformer/Do Transformers Really PerformBadforGraphRepresentation.pdf]]"]
---

# Graphormer

**Graphormer** is a graph-configured Transformer architecture designed to model graph-structured data by integrating structural and topological priors directly into the standard self-attention mechanism. By bypassing the local neighborhood restrictions of classical message-passing GNNs, Graphormer establishes a global receptive field where any node can attend to any other node, modulated by topological distance and edge qualities.

---

## The Three Core Structural Encodings

Graphormer models graph structure through three primary inductive biases:

1.  **Centrality Encoding**: Leverages node degree (in-degree and out-degree) to inject a continuous representation of node importance directly into the input node features.
2.  **Spatial Encoding**: Modulates attention logits using a learnable scalar bias indexed by the **Shortest Path Distance (SPD)** between node pairs, allowing the model to adaptively learn local or global connectivity patterns.
3.  **Edge Encoding**: Projects and averages the feature vectors of edges along the shortest path between two nodes, injecting physical connection attributes (like molecular bonds or highway capacities) into the attention score.

---

## Mathematical Derivation: Query-Key Interaction in Centrality Encoding

To understand why Centrality Encoding is so powerful, we can expand the mathematical product of the Query and Key vectors in the self-attention layer. 

For an undirected graph, let $x_i \in \mathbb{R}^d$ and $x_j \in \mathbb{R}^d$ be the raw semantic features of nodes $v_i$ and $v_j$. Let $z_{\operatorname{deg}(v_i)} \in \mathbb{R}^d$ and $z_{\operatorname{deg}(v_j)} \in \mathbb{R}^d$ be their respective learnable degree centrality embeddings. 

At the input layer, these representations are summed:
$$h_i^{(0)} = x_i + z_{\operatorname{deg}(v_i)}$$
$$h_j^{(0)} = x_j + z_{\operatorname{deg}(v_j)}$$

When projected into Query ($\mathbf{q}_i$) and Key ($\mathbf{k}_j$) spaces using projection matrices $W_Q, W_K \in \mathbb{R}^{d \times d_k}$, we obtain:
$$\mathbf{q}_i = h_i^{(0)} W_Q = (x_i + z_{\operatorname{deg}(v_i)}) W_Q = x_i W_Q + z_{\operatorname{deg}(v_i)} W_Q$$
$$\mathbf{k}_j = h_j^{(0)} W_K = (x_j + z_{\operatorname{deg}(v_j)}) W_K = x_j W_K + z_{\operatorname{deg}(v_j)} W_K$$

The raw attention logit $A_{ij}$ is proportional to the dot-product $\mathbf{q}_i \mathbf{k}_j^T$. Expanding this product yields **four distinct structural-semantic terms**:

$$A_{ij} \propto \mathbf{q}_i \mathbf{k}_j^T = \underbrace{(x_i W_Q)(x_j W_K)^T}_{\text{1. Semantic-to-Semantic}} + \underbrace{(x_i W_Q)(z_{\operatorname{deg}(v_j)} W_K)^T}_{\text{2. Semantic-to-Centrality}} + \underbrace{(z_{\operatorname{deg}(v_i)} W_Q)(x_j W_K)^T}_{\text{3. Centrality-to-Semantic}} + \underbrace{(z_{\operatorname{deg}(v_i)} W_Q)(z_{\operatorname{deg}(v_j)} W_K)^T}_{\text{4. Centrality-to-Centrality}}$$

### **Semantic-to-Structural Deconstruction:**
1.  **Semantic-to-Semantic**: This is vanilla self-attention. It measures the pure attribute similarity between node $v_i$ and node $v_j$, ignoring their structural positions.
2.  **Semantic-to-Centrality**: Measures how much a node with semantic state $x_i$ should pay attention to a highly central hub (node $v_j$ with degree centrality $z_{\operatorname{deg}(v_j)}$). For example, a peripheral node learning to attend to major distribution hubs.
3.  **Centrality-to-Semantic**: Measures how a hub node $v_i$ distributes its attention to other nodes based on their attributes $x_j$. This allows hubs to selectively query relevant semantic features from the rest of the graph.
4.  **Centrality-to-Centrality**: Computes structural affinity. It allows the model to learn structural hierarchies (e.g., whether hub nodes should strongly coordinate with other hub nodes, or whether leaf nodes should prioritize leaf-to-leaf connections).

---

## Comparison: Discrete Spatial Bias vs. Continuous Distance Weights

A natural alternative to Graphormer's discrete spatial lookup is to multiply topological distance by a learnable continuous weight (e.g., $w \cdot d$). However, the discrete lookup table ($b_{\phi(v_i, v_j)}$) is chosen due to several critical representational advantages:

| Property | Continuous Multiplier ($w \cdot d$) | Discrete Spatial Bias ($b_{\phi(v_i, v_j)}$) |
| :--- | :--- | :--- |
| **Mathematical Shape** | Strictly linear and monotonic. Attention must decay or grow smoothly. | Arbitrary and non-monotonic. Can model step-functions, periodic cycles, or localized ring structures. |
| **Structural Interpretation** | Assumes physical spatial continuity. | Fits irregular graph motifs (e.g., bipartite user-item hops where even-hop steps are highly prioritized). |
| **Disconnected Components** | Fails or requires complex heuristics to represent infinite distance ($\infty$). | Handled elegantly using a dedicated lookup index $b_{-1}$ for disconnected node pairs. |
| **Head Diversity** | Limited. Attention heads can only adjust the slope of distance decay. | High. Each attention head can learn a completely unique topological filter. |

---

## Connection to Spatial-Temporal Transformers

In Spatial-Temporal (ST) Transformers, Graphormer's structural encodings serve as the foundational spatial layer:
*   **Static ST-Transformers**: Use Graphormer's spatial bias to represent physical sensor grids (e.g., distances along road networks in traffic forecasting).
*   **Dynamic ST-Transformers**: Use dynamic centrality and edge encodings to track active correlation links and traffic bottleneck capacities as they fluctuate over time.

---

## See Also
- [[Graphormer Summary]]: Research summary detailing the paper specifications.
- [[Positional Encoding]]
- [[CITRAS Summary]]
