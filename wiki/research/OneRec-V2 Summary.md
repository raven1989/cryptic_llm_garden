---
tags:
  - recommendation
  - generative-retrieval
  - architectures
  - scaling-laws
aliases:
  - OneRec-V2
  - Lazy Decoder-Only Architecture
date: 2026-06-08
sources: ["[[raw/Recommendation/OneRec-V2 Technical Report.pdf]]"]
---

# OneRec-V2: Lazy Decoder-Only Generative Recommender

![OneRec-V2 Architecture](../../wiki/media/onerec_v2_architecture.png)

## 1. Introduction and Core Bottlenecks of OneRec-V1
While Kuaishou's **OneRec-V1** pioneered industrial-scale generative recommendation (GR) by replacing the multi-stage cascade with an Encoder-Decoder pipeline, it encountered two massive limits:
1. **Computational Imbalance:** With a typical context length of $N = 512$, **97.66% of total FLOPs were wasted on the Encoder** for context encoding (user histories), while only **2.34%** went into the Decoder for target sequence generation. As context grew, this imbalance choked model scalability.
2. **Preference Alignment Limits:** Relying purely on offline Reward Models (RMs) for Reinforcement Learning suffered from poor sampling efficiency and potential **reward hacking** (where the policy exploits proxy RM shortcuts).

**OneRec-V2** eliminates these bottlenecks using:
* A **Lazy Decoder-Only Architecture** which strips out the encoder and cross-attention weight matrices, dropping training resources by **90%** and FLOPs by **94%**, allowing scaling up to **8 Billion parameters**.
* A post-training RL framework driven directly by **real-world user feedback** with duration-aware reward shaping and adaptive ratio clipping.

---

## 2. Design Principles & Data Organization

![OneRec-V2 Sample Organization](../../wiki/media/Onerec_v2_sample_organization.png)

To adapt Transformers safely to generative recommendation without redundancy or look-ahead biases, OneRec-V2 organizes data based on three core concepts:

### 2.1 Chronological Data Splitting & NIO
* **Naive Impression Organization:** Leads to redundant training of identical transitions (e.g., $A \to B$) across multiple sequences.
* **User-Centric Organization:** Groups a user's entire history into one long sequence. While it reduces redundancy, it introduces severe **temporal data leakage** (look-ahead bias) during offline training/evaluation where future events update weights before past events are predicted, ruining offline-to-online metric translation.

### 2.2 Deep-Dive: How New Impression Only (NIO) Solves Temporal Leakage and Redundancy
OneRec-V2's **New Impression Only (NIO) Organization** perfectly solves both challenges:

#### **A. Solving Temporal Leakage via Physical Time Causality**
In NIO, **each individual impression event at physical time $T$ is sliced into its own standalone training sample**. 
* **Input-Level Causality (Intra-Sample):** For an impression at $T$, the context sequence is strictly restricted to interactions that occurred *before* $T$ (i.e., physical times $< T$). Any future interactions at times $> T$ are physically omitted from the input context.
* **Weight-Level Causality (Inter-Sample):** By slicing histories into standalone impressions, the entire dataset is trained strictly chronologically based on $T$. For instance, when training on User-2's impression at $t_3$, the model weights have only been updated by past events ($t_1, t_2$). The future interaction of User-1 at $t_4$ is completely unseen, preventing look-ahead "cheating."
* **Clean Dataset Splitting:** This allows splitting the dataset using a global cutoff timestamp $T_{\text{split}}$. All samples with $T < T_{\text{split}}$ form the training set, and those with $T \ge T_{\text{split}}$ form the test/validation set. Offline metrics are thus an honest, leakage-free indicator of online performance.

#### **B. Solving Redundancy via NIO Loss Masking**
Under a standard Next Token Prediction (NTP) objective over chronological sequences, older transitions (like $A \to B$) are trained repeatedly in subsequent sessions (like $A \to B \to C$), leading to massive computational waste and over-biasing.
* NIO solves this by **applying a loss mask to all former context items**.
* For any sample representing an impression at physical time $T$, the NTP cross-entropy loss is evaluated **exclusively on the semantic ID tokens of the newest target item**.
* Former items are processed purely as static conditioning information (gray blocks in Figure 3), completely bypassing gradient computation and redundant weight updates.

---

## 3. Overall Architecture of Lazy Decoder-Only

OneRec-V2 ditches the separate Encoder stack entirely. The model instead consists of a lightweight projection layer and stacked **Lazy Decoder Blocks**.

### 3.1 Deep-Dive: Context Processor (Projection-Free KV Generation)
The Context Processor converts heterogeneous user pathways (User Static, Short-term Behavior, Long-term Behavior) into a unified sequence of length $(N_s + T_{\text{short}} + T_{\text{long}})$ and projects them to a uniform dimension:
$$d_{\text{context}} = S_{\text{kv}} \cdot L_{\text{kv}} \cdot G_{\text{kv}} \cdot d_{\text{head}}$$

#### **A. Pathway Input Length and Attention Masking**
* **User Static Features ($N_s$):** A fixed number of categorical user-profile feature tokens (e.g., age, gender, city).
* **Short-term ($T_{\text{short}}$) and Long-term ($T_{\text{long}}$) Behavior Sequences:** These have highly variable lengths in their raw state. To handle them on GPUs, they are right-padded or truncated to a maximum context window (capped at $N \approx 512$ during pre-training, and expanded up to $N \approx 3000$ during online serving).
* **Active Attention Masking:** A strict attention mask is applied to the generated key-value tensors, forcing the decoder blocks to ignore any padded positions and focus solely on actual user interactions.

#### **B. The Major Breakthrough: Projection-Free KV Generation**
In standard Transformers, context hidden states must be multiplied by heavy projection matrices to compute keys and values ($K = \text{Context} \cdot W_k$, $V = \text{Context} \cdot W_v$). OneRec-V2 **completely eliminates the projection weights $W_k$ and $W_v$** from cross-attention:
1. **Partitioning:** The unified `Context` tensor is directly partitioned along its feature dimension into $L_{\text{kv}}$ chunks:
   $$\text{Context} = [C_0, C_1, \dots, C_{S_{\text{kv}} \cdot L_{\text{kv}} - 1}]$$
2. **RMSNorm KV Generation (Lazy Sharing):** Elements-wise **RMSNorm** is applied directly to these chunks to produce keys ($k_l$) and values ($v_l$) without any matrix multiplications:
   $$k_l = \text{RMSNorm}_{k, l}\left(C_{l \cdot S_{\text{kv}}}\right)$$
   $$v_l = \begin{cases} 
   \text{RMSNorm}_{v, l}\left(C_{l \cdot S_{\text{kv}} + 1}\right) & \text{if } S_{\text{kv}} = 2 \text{ (separated key-value)} \\
   k_l & \text{if } S_{\text{kv}} = 1 \text{ (shared representation)}
   \end{cases}$$
3. **Zero Runtime Overhead:** During autoregressive generation, keys and values are generated once by the Context Processor and stored in the KV cache. No matrix multiplications or projections are ever executed over the long context sequence, yielding a massive **90% training resource saving** and **94% FLOPs drop**.

#### **C. Retaining Target Query Projection ($W_q$)**
While context projections are eliminated, OneRec-V2 preserves a standard query projection matrix $W_q$ on the target decoder side:
* **Why?** Since the target sequence (semantic IDs `[BOS, s1, s2]`) is extremely short (length $\le 3$), performing the $W_q$ query projection is computationally trivial and runs with virtually zero FLOP overhead.
* **Benefit:** It preserves the multi-head search capabilities of Grouped Query Attention (GQA) and Multi-Query Attention (MQA), allowing the short target queries to search the projection-free context keys/values from multiple relational subspaces.

### 3.2 Lazy Decoder Block
Each of the $N_{\text{layer}}$ stacked decoder blocks combines Cross-Attention, Causal Self-Attention, and FFN/MoE layers. The forward pass is defined as:
$$h^{(l)}_{\text{cross}} = h^{(l-1)} + \text{CrossAttn}\left(\text{RMSNorm}\left(h^{(l-1)}\right), k_{l_{\text{kv}}}, v_{l_{\text{kv}}}\right)$$
$$h^{(l)}_{\text{self}} = h^{(l)}_{\text{cross}} + \text{SelfAttn}\left(\text{RMSNorm}\left(h^{(l)}_{\text{cross}}\right)\right)$$
$$h^{(l)} = h^{(l)}_{\text{self}} + \text{FFN}^{(l)}\left(\text{RMSNorm}\left(h^{(l)}_{\text{self}}\right)\right)$$

Where:
* **Lazy Index-Sharing ($l_{\text{kv}}$):** Blocks share the static KV cache representations. For decoder layer $l$, the active KV cache index $l_{\text{kv}}$ is:
  $$l_{\text{kv}} = \left\lfloor \frac{l \cdot L_{\text{kv}}}{N_{\text{layer}}} \right\rfloor$$

---

## 4. Key Empirical Findings (Section 2.3)

### 4.1 Comparative Efficiency (Table 2)
At a 1B parameter scale (with context size $N=512$):
* **OneRec-V1 (Encoder-Decoder):** Crunches **296.36 GFLOPs**, with **17.63B activations**. (Loss: 3.28)
* **Classic Naive Decoder-Only:** Crunches **634.83 GFLOPs**, with **31.53B activations** (quadratic history scaling). Clogs GPU memory, making scales $\ge 0.5\text{B}$ unrunnable.
* **OneRec-V2 (Lazy Decoder):** Requires only **18.89 GFLOPs** and **1.24B activations** (a **94% FLOPs reduction** and **93% activation memory reduction**) while improving convergence loss to **3.27**.

### 4.2 Key-Value Sharing and Grouped Query Attention (GQA)
* **KV-Sharing ($L_{\text{kv}}$ and $S_{\text{kv}}$):** Running the most aggressive sharing strategy ($L_{\text{kv}} = 1, S_{\text{kv}} = 1$, i.e., all 18 layers share a single set of tied key-value vectors) achieves identical convergence loss (3.27) to fully unshared layers, while halving activation footprint and cutting GFLOPs from 23.95 to 18.89.
* **GQA Head Compression ($G_{\text{kv}}$):** Capping KV group heads to $G_{\text{kv}} = 1$ (Multi-Query Attention) achieves identical convergence loss (3.27) compared to full attention, but **shrinks the active KV cache size by 92.5%** (from 94MB to 7MB), enabling extremely large context windows (up to 3000 tokens online).

### 4.3 Empirical Scaling Laws
Thanks to the Lazy Decoder's efficiency, OneRec-V2 scales predictably from 0.1B to **8 Billion parameters**. Keeping dataset size $D$ fixed, the empirical scaling law for generative recommendation is:
$$\hat{L}(N) = 3.13 + \frac{3660}{N^{0.489}}$$

* **Irreducible Loss $E = 3.13$:** Represents the baseline entropy/randomness of Kuaishou's recommendation dataset.
* **Scaling Exponent $\alpha = 0.489 \approx 0.5$:** Confirms that reducible loss scales down roughly as $1/\sqrt{N}$.


---


## 5. Post-Training Preference Alignment with Real-World User Interactions

To prevent model misalignment and bypass the limitations of proxy reward models, OneRec-V2 designs an end-to-end alignment framework utilizing direct real-world user feedback.

### 5.1 Duration-Aware Reward Shaping
Raw playing time on short videos is heavily biased by the total duration of the video. To extract an unbiased satisfaction signal, OneRec-V2 introduces a logarithmic bucketing and percentile rank normalization process.

#### **A. Logarithmic Duration Bucketing**
Because video durations follow a heavy long-tail distribution, they are partitioned into exponentially widening buckets using a logarithmic mapping:
$$F(d) = \lfloor \log_{\beta} (d + \epsilon) \rfloor$$
* where `` `d` `` is the video duration, `` `\beta` `` is a configurable logarithmic base controlling bucket granularity, and `` `\epsilon = 10^{-6}` `` is a numerical stabilizer.

#### **B. Empirical Percentile Ranking**
For a user `` `u` `` with a history of watched video durations and playtimes `` `H_u = \{(d_k, p_k)\}_{k=1}^N` ``, we collect all historical playtimes that fall into the target video's bucket `` `b = F(d_i)` ``:
$$P_{u, b} = \{ p_j \mid (d_j, p_j) \in H_u, \ F(d_j) = b \}$$

The duration-normalized engagement score `` `q_i` `` is computed as the empirical percentile rank of the target playtime `` `p_i` `` within that historical peer playtime bucket:
$$q_i = \frac{|\{ p_j \in P_{u, b} \mid p_j \le p_i \}|}{|P_{u, b}|}$$

#### **C. Strict Advantage Value Formulation**
Using the top-quartile (75th percentile) boundary `` `\tau_b` `` of scores within a training batch, and incorporating explicit negative feedback (such as user clicking "dislike", represented as `` `neg_i = 1` ``), the sparse advantage `` `A_i` `` is defined as:
$$A_i = \begin{cases} 
+1, & q_i > \tau_b \text{ and } neg_i = 0 \quad \text{(Positive feedback)} \\
-1, & neg_i = 1 \quad \text{(Negative feedback)} \\
0, & \text{otherwise (Filtered out)}
\end{cases}$$

---

### 5.2 Gradient-Bounded Policy Optimization (GBPO)
During reinforcement learning on hybrid exposure logs, many training samples originate from legacy multi-stage recommendation cascades. Because we cannot compute the old policy probability `` `\pi_{old}` `` for these external cascades, we must simplify it to the current model's stop-gradient probability:
$$\pi_{old}(o_i \mid u) \approx \text{sg}(\pi_\theta(o_i \mid u))$$

Under this approximation, the standard policy ratio is always exactly 1:
$$\text{Ratio} = \frac{\pi_\theta(o_i \mid u)}{\text{sg}(\pi_\theta(o_i \mid u))} \equiv 1$$

Standard clipping algorithms (like PPO, GRPO, or ECPO) view a ratio of 1 as completely stable, bypassing all truncation and clipping logic. However, this triggers **catastrophic gradient explosion** on negative samples.

---

### 5.3 Mathematical Proof of Gradient Explosion via the Quotient Rule
Let us analyze the gradient of the standard ECPO/GRPO loss objective for a specific video token `` `i` `` when the ratio is exactly 1:
$$J^{(i)}_{\text{ECPO}}(\theta) = - A_i \cdot \frac{\pi_\theta}{\text{sg}(\pi_\theta)} \tag{15}$$

Let us take the derivative of this objective with respect to the model weights `` `theta` `` using the **Quotient Rule** of calculus, where:
* Numerator $u = \pi_\theta$
* Denominator $v = \text{sg}(\pi_\theta)$

$$\frac{\partial}{\partial \theta} \left[ \frac{\pi_\theta}{\text{sg}(\pi_\theta)} \right] = \frac{\text{sg}(\pi_\theta) \frac{\partial \pi_\theta}{\partial \theta} - \pi_\theta \frac{\partial \text{sg}(\pi_\theta)}{\partial \theta}}{\text{sg}(\pi_\theta)^2}$$

By the fundamental mathematical definition of the **Stop-Gradient** operator, its derivative is exactly 0:
$$\frac{\partial \text{sg}(x)}{\partial \theta} = 0$$

Plugging this into the Quotient Rule:
$$\frac{\partial}{\partial \theta} \left[ \frac{\pi_\theta}{\text{sg}(\pi_\theta)} \right] = \frac{\text{sg}(\pi_\theta) \frac{\partial \pi_\theta}{\partial \theta} - \pi_\theta \cdot 0}{\text{sg}(\pi_\theta)^2} = \frac{\text{sg}(\pi_\theta) \frac{\partial \pi_\theta}{\partial \theta}}{\text{sg}(\pi_\theta)^2}$$

Canceling out one `` `sg(pi_theta)` `` term from the numerator and denominator:
$$\frac{\partial}{\partial \theta} \left[ \frac{\pi_\theta}{\text{sg}(\pi_\theta)} \right] = \frac{1}{\text{sg}(\pi_\theta)} \cdot \frac{\partial \pi_\theta}{\partial \theta}$$

Finally, because the stop-gradient operator is an identity function in the forward pass, its evaluated value is identical to `` `pi_theta` ``. Replacing `` `sg(pi_theta)` `` with `` `pi_theta` `` and multiplying by the constant factor `` `-A_i` `` yields **Equation 16**:
$$\frac{\partial J^{(i)}_{\text{ECPO}}(\theta)}{\partial \theta} = - A_i \cdot \frac{1}{\pi_\theta} \cdot \frac{\partial \pi_\theta}{\partial \theta} \tag{16}$$

#### **The Instability Mechanics:**
* For **negative samples ($A_i = -1$)**, the gradient is $+ \frac{1}{\pi_\theta} \cdot \frac{\partial \pi_\theta}{\partial \theta}$.
* As the model successfully learns to suppress the negative video, its generation probability approaches zero ($\pi_\theta \to 0$).
* However, because the probability is in the denominator, as $\pi_\theta \to 0$, the term $\frac{1}{\pi_\theta} \to \infty$ **explodes to infinity**. This triggers catastrophic training crashes and model overfitting on negative signals.

---

### 5.4 The GBPO Formulation (The Dynamic Shock Absorber)
GBPO solves this by introducing a **dynamic, bounded old-policy probability** `` `\pi'_{old}` `` in the denominator:

$$J_{\text{GBPO}}(\theta) = - \mathbb{E}_{u \sim P(U), \ \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{\pi_\theta(o_i \mid u)}{\pi'_{\text{old}}(o_i \mid u)} \cdot A_i \right] \tag{11}$$

Where the dynamic denominator `` `\pi'_{old}` `` is defined as:
$$\pi'_{\text{old}}(o_i \mid u) = \begin{cases}
\max\left(\pi_{\theta_{\text{old}}}, \ \text{sg}(\pi_\theta)\right), & A_i \ge 0 \quad \text{(Positive Samples)} \\
\max\left(\pi_{\theta_{\text{old}}}, \ \mathbf{1 - \text{sg}(\pi_\theta)}\right), & A_i < 0 \quad \text{(Negative Samples)}
\end{cases} \tag{12}$$

#### **How GBPO Bounds Negative Gradients:**
* For a negative sample ($A_i = -1$), as the model suppresses the bad video ($\pi_\theta \to 0$), the denominator term $(1 - \text{sg}(\pi_\theta)) \to 1$.
* The `max` operator selects $1 - \text{sg}(\pi_\theta) \approx 1$.
* The policy ratio becomes $\frac{\pi_\theta}{1}$. 
* Taking the derivative of this bounded ratio yields:
  $$\lim_{\pi_\theta \to 0} \frac{\partial J_{\text{GBPO}}^{(i)}}{\partial \theta} = \frac{\partial \pi_\theta}{\partial \theta}$$
* Since the probability derivative $\frac{\partial \pi_\theta}{\partial \theta}$ is naturally bounded, **the gradient safely vanishes to 0 as the negative sample is suppressed**, completely mimicking the stable optimization of standard Binary Cross-Entropy (BCE) loss.

---

## 6. Offline and Online Evaluation

### 6.1 On-Policy Self-Improvement (Table 6)
Streaming RL training results on Kuaishou's production logs show that incorporating self-generated samples (on-policy RL) triggers a powerful feedback loop:
* **Traditional Logs Only:** Off-policy training on legacy logs improves App Stay Time (+0.165%) but hurts other engagement metrics like video views (-0.901%) due to distribution mismatch.
* **On-Policy Self-Loop (w/ OneRec Samples):** Training on OneRec's own recommendations increases App Stay Time (+0.227%) and turns video views highly positive (+0.716%), with likes, follows, comments, and shares surging up to **+6.39%**.

### 6.2 Comparison: Reward Model vs. Direct User Feedback (Table 7)
* **Reward Model RL:** Tends to over-index on user interactions (+15.47% comments, +12.0% forwards) because the offline RM is optimized on multi-task action rates.
* **User Feedback RL:** Tends to drive duration-based metrics, spiking App Stay Time (+0.299%) and Video Views (+0.647%) because it directly optimizes playtime percentile.
* **The Hybrid Approach:** Combining both yields the optimal equilibrium, delivering excellent App Stay Time gains (+0.283%) alongside massive interaction boosts (+7.0% likes, +8.4% follows, +8.7% comments) with **zero seesaw trade-offs**.

### 6.3 Final Production Impact (Kuaishou & Kuaishou Lite Feed, Table 8)
When fully deployed on Kuaishou's main feed serving **400 Million Daily Active Users (DAU)** (utilizing a 1B model, context length 3000, and beam size 512):
* **Kuaishou Main App:** App Stay Time **+0.467%**, Watch Time **+1.367%**, Likes **+3.924%**, Follows **+4.730%**.
* **Kuaishou Lite App:** App Stay Time **+0.741%**, Watch Time **+0.762%**, Likes **+5.393%**, Follows **+5.627%**.

This final result confirms that the unified, Lazy Decoder-only generative architecture, aligned directly with real-world user interaction loops, completely outperforms highly optimized multi-stage ranking cascades in large-scale real-world production.


---

## Related Wiki Pages
* [[OneRec-V2]]: High-level concept and technological specification page.
* [[OneRec]]: Concept and architectural profile of the original OneRec model.
* [[OneRec Summary]]: Research summary of the original OneRec-V1 paper.
* [[RMSNorm]]: Root Mean Square Normalization powering projection-free KV generation.
* [[Grouped Query Attention]]: The head-sharing optimization used to shrink context caches.
* [[KV Cache]]: Memory and latency optimization for real-time autoregressive serving.
