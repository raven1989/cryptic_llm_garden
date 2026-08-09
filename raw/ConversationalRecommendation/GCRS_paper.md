---
title: "Generative Conversational Recommender System"
source: "https://arxiv.org/html/2605.21987v1"
author:
published:
created: 2026-08-06
description:
tags:
  - "clippings"
---
Sixiao Zhang  
College of Computing and Data Science  
Nanyang Technological University  
Singapore  
sixiao001@e.ntu.edu.sg  
&Mingrui Liu  
College of Computing and Data Science  
Nanyang Technological University  
Singapore  
mingrui001@e.ntu.edu.sg  
Cheng Long  
College of Computing and Data Science  
Nanyang Technological University  
Singapore  
c.long@ntu.edu.sg  
Corresponding author.

###### Abstract

Conversational recommender systems aim to provide personalized recommendations via natural language interactions. However, existing approaches either decouple recommendation from dialog generation or rely on retrieval-based pipelines, limiting the integration between recommendation and response generation and leading to suboptimal modeling of user intent. In this paper, we propose a fully generative conversational recommender system that unifies recommendation and dialog generation within a single autoregressive framework. Our approach represents items as discrete semantic IDs and integrates them directly into the generation process, enabling joint prediction of items and responses via next-token modeling. We further introduce a structured generation paradigm that factorizes conversational recommendation into a sequence of interdependent decisions, where the model first predicts the response intent and the recommendation target, and then generates the response conditioned on them. This design enables end-to-end optimization, enforces a more coherent dependency structure, and supports faithful item generation via constrained decoding. Extensive experiments demonstrate that our method consistently improves recommendation performance, achieving gains of up to 29% on Recall@1 over strong baselines, while maintaining competitive dialog quality.

## 1 Introduction

Conversational recommender systems (CRSs) aim to provide personalized recommendations through multi-turn natural language interactions, enabling systems to dynamically elicit user preferences and refine recommendations during dialog. With the rapid development of large language models (LLMs), recent studies have explored incorporating LLMs into CRSs to enhance language understanding and response generation. Despite their strong generative capabilities, effectively integrating LLMs with recommendation remains a challenging problem.

Most existing CRSs adopt a modular pipeline that separates conversation and recommendation [^16] [^40] [^17]. In these systems, the dialog context is first encoded into a query representation, and recommendation is formulated as a nearest-neighbor retrieval problem over an item embedding space. The rich textual signals in conversations mainly serve as auxiliary features to improve the query representation, rather than being directly used to generate recommendations. After retrieving top-k items, they are injected into the generated responses, for example by replacing placeholders such as “<movie>” with item titles. While effective, such a design suffers from two key limitations: (1) decoupled optimization, where the conversation module and recommendation module are trained separately, leading to suboptimal performance while making the system complex and resource-intensive; and (2) limited knowledge integration, where the language model lacks direct access to collaborative signals and item semantics during generation, often leading to generic and weakly personalized responses.

To better leverage LLMs, recent works explore using LLMs either as rerankers over candidate items [^32] [^5] [^14] or as end-to-end generators via fine-tuning [^29] [^2] [^31]. However, both approaches have inherent limitations. Reranking-based methods still rely on external candidate generators, making their performance heavily dependent on the quality of the candidate set. Moreover, LLM rerankers primarily operate based on textual matching and lack access to collaborative signals and domain-specific recommendation knowledge, limiting their ability to capture true user preferences. Fine-tuning-based methods attempt to represent items using natural language titles or discrete ID tokens, but face several critical challenges: (1) unfaithful generation, where item titles may be generated inaccurately, leading to hallucinations; (2) scalability issues, as assigning each item a unique token may lead to vocabulary explosion; and (3) poor generalization, since newly introduced items cannot be handled without modifying the model vocabulary.

In this work, we propose a fully generative conversational recommender system that addresses these challenges within a unified framework. Our approach consists of two key components. First, we represent items as structured semantic IDs and integrate them directly into the generation process. Specifically, item mentions in dialogs are replaced with their corresponding semantic IDs, and an LLM is fine-tuned to jointly generate item IDs and natural language responses via next-token prediction. Second, to better capture the interaction between recommendation and language generation, we introduce a structured generation paradigm that factorizes conversational recommendation into a sequence of interdependent decisions. Concretely, the model first determines the response intent and predicts the target item, and then generates the corresponding natural language response conditioned on these decisions.

This design offers several key advantages. By integrating recommendation and response generation within a single model, our approach eliminates the need for external retrieval or reranking modules and enables end-to-end optimization. Moreover, the structured factorization explicitly separates high-level decision making from surface realization, allowing the model to determine the response intent and target item before generating the final response. This leads to improved alignment between item prediction and language generation, resulting in more consistent and reliable recommendation behavior. In addition, semantic IDs enable faithful item generation: with constrained decoding, generated tokens are guaranteed to correspond to valid items, effectively preventing hallucinations. Their compositional structure further avoids vocabulary explosion and naturally supports generalization to unseen items. Overall, our framework moves towards a fully generative conversational recommender system, where recommendations emerge as explicit intermediate decisions within the generation process rather than being injected post hoc. Our contributions are summarized as follows <sup>1</sup>:

- We propose a unified generative framework for conversational recommendation that integrates recommendation and dialog generation within a single autoregressive model, enabling end-to-end optimization without external retrieval or reranking modules.
- We introduce a semantic ID representation together with a structured generation paradigm, which factorizes conversational recommendation into explicit intermediate decisions within the generation process.
- We conduct extensive experiments on benchmark datasets, demonstrating that our method significantly improves recommendation performance (up to +29% on Recall@1) while maintaining high-quality and diverse responses.

## 2 Related work

Conversational recommender systems. Early CRSs typically follow a *modular paradigm*, where recommendation and dialog generation are optimized separately. Methods such as ReDial [^16] and KBRD [^1] infer user preferences from dialog context and retrieve items based on embedding similarity, with responses generated conditioned on retrieved results. Later works incorporate additional signals such as knowledge graphs [^40] [^17], reviews [^19], and contrastive learning [^41] to improve representations. However, these methods remain largely *retrieval-based* and loosely couple recommendation with response generation.

LLM-based conversational recommendation. Recent work leverages LLMs to unify recommendation and dialog generation. Some approaches inject recommendation signals via special tokens or parameter-efficient tuning [^28] [^23], while others rely on prompting or external modules [^12] [^13]. Despite this, most methods still depend on *external retrieval pipelines*. For example, UniCRS and its extensions [^29] [^2] [^36] [^31] integrate LLMs with recommender models but retain separate objectives, while RecInDial [^28] and MESE [^35] rely on similarity-based retrieval. Another line of work retrieves candidate items using expert recommenders and applies LLMs for reranking and generation [^32] [^5] [^14] [^42]. These approaches suffer from limited integration between recommendation and generation, and often struggle with hallucination and weak modeling of collaborative signals.

Generative recommendation with semantic IDs. Generative recommendation directly predicts item identifiers instead of retrieving them. TIGER [^22] introduces semantic IDs via vector quantization, and subsequent works improve ID construction through better quantization [^3] [^17], collision handling [^39], and incorporation of collaborative signals [^30] [^33]. Extensions further explore multimodal information and richer training objectives [^37] [^38] [^15]. However, these methods focus on sequential recommendation and item generation, without explicitly modeling conversational context or jointly optimizing natural language responses.

## 3 Methodology

Our framework, Generative Conversational Recommender System (GCRS), consists of two key components that address the core challenges of existing approaches: (1) semantic ID construction, which provides a scalable and faithful representation of items for generation, and (2) structured generation, which models recommendation as an explicit intermediate step within response generation. An illustration of the method is shown in Fig. 1. Concretely, we first map each item into a discrete semantic ID space via residual quantization. Based on this representation, we reformulate conversational recommendation as a structured sequence generation problem, where the model first predicts the response intent, then predicts the target item, and finally generates the corresponding response. This design introduces an inductive bias that aligns the generation process with the underlying structure of conversational recommendation, tightly coupling item prediction with language generation.

### 3.1 Task definition

The goal of conversational recommendation is to generate a system response conditioned on the dialog history, while optionally recommending items. Formally, a dialog session is defined as a sequence of utterances:

$$
\mathcal{D}=(u_{1},u_{2},\dots,u_{T}),
$$

where each utterance $u_{t}$ is associated with a speaker role (user or system). The objective of CRS is to model the conditional probability:

$$
P(u_{t}\mid u_{1},\dots,u_{t-1}),
$$

where $u_{t}$ is a recommender utterance that may contain recommended items.

![Refer to caption](https://arxiv.org/html/2605.21987v1/x1.png)

Figure 1: Overview of the GCRS framework. During semantic ID construction, item metadata is encoded into dense embeddings by a pretrained text encoder, and an RQ-VAE with collision resolution is trained to map items into discrete semantic IDs. During conversational recommender training, item mentions in raw dialogs are replaced with their corresponding semantic IDs, and structured sequences are constructed based on the behavior of the ground-truth response. The LLM is fine-tuned on these structured inputs to jointly learn dialog generation and recommendation behaviors.

### 3.2 Semantic ID construction via RQ-VAE

A direct way to represent items in conversational recommendation is to use either their textual titles or assign each item a unique identifier. However, textual titles can lead to hallucination and ambiguity, while unique identifiers suffer from scalability and generalization issues [^8] [^9]. To address these limitations, we represent each item with a discrete semantic ID [^22] that captures its semantic content while remaining compatible with language model generation.

Metadata encoding. For each item $i$, we collect its metadata and encode it into a context embedding. For example, in the movie recommendation domain, the metadata includes title, year, genres, keywords, and plot. We serialize these attributes into a unified textual description:

$$
x_{i}=\texttt{"title: }t_{i}\texttt{ | year: }y_{i}\texttt{ | genres: }g_{i}\texttt{ | keywords: }k_{i}\texttt{ | plot: }p_{i}\texttt{"}.
$$

A fixed pretrained text encoder is then used to encode the textual description into a dense embedding.

Residual quantization with collision resolution. We further quantize the embedding into a sequence of discrete codes using RQ-VAE. An example of a 4-digit semantic ID is " $<a\_17><b\_63><c\_0><d\_25>$ ". Because vector quantization is many-to-one, different items may be assigned the same semantic ID, resulting in an unfair comparison on the recommendation performance. To ensure uniqueness, we resolve such collisions following the strategy of [^37]. Please refer to Appendix A for details. After collision resolution, each item is associated with a unique semantic ID. These semantic IDs are fixed in the remainder of the framework and are directly used as item identifiers for subsequent generative modeling.

### 3.3 Unified modeling via structured generation

From implicit generation to structured generation. After replacing item mentions in dialogs with semantic IDs, a straightforward approach is to fine-tune an LLM on the transformed dialogs using standard next-token prediction. Under this formulation, the model learns the joint probability of the response sequence as

$$
P(u_{t}\mid C)=\prod_{j=1}^{|u_{t}|}P(u_{t,j}\mid C,u_{t,<j}),
$$

where $C=(u_{1},\dots,u_{t-1})$ denotes the dialog context and $u_{t}$ is the target response, which may contain both semantic ID tokens and natural language tokens. While general, this formulation induces an implicit token-level factorization that does not align with the underlying structure of conversational recommendation. In particular, recommendation decisions (i.e., which item to recommend) and natural language responses are entangled within a single autoregressive sequence without explicit constraints. As a result, the model may generate natural language tokens before producing the corresponding semantic IDs, making item prediction conditioned on previously generated surface text. Such text can be noisy, ambiguous, or only partially reflect user preferences, thereby introducing spurious dependencies and increasing the difficulty of accurately modeling recommendation intent.

Design overview. To address this limitation, we introduce a structured generation framework that explicitly factorizes the generation process according to the inherent decision flow of conversational recommendation. Instead of treating the response as a flat token sequence, we decompose it into a sequence of interdependent variables with a prescribed generation order: the model first determines the response intent, then predicts the target item if recommendation is required, and finally generates the natural language response conditioned on the preceding decisions. Formally, this corresponds to the following factorization of the conditional generation probability:

$$
P(u_{t}\mid C)=P(m\mid C)\cdot P(i\mid m,C)\cdot P(r\mid i,m,C),
$$

where $m$ denotes the response intent (mode), $i$ is the target item, and $r$ is the natural language response. This structured factorization separates item prediction from surface text realization, reducing interference from previously generated tokens and enforcing a more faithful dependency structure. As a result, it enables more accurate recommendation decisions and improves the consistency between predicted items and generated responses.

Semantic ID replacement. Given the semantic IDs constructed in Sec. 3.2, we replace all item mentions in the dataset with their corresponding semantic IDs. Each semantic ID is wrapped with two new special tokens:

$$
\texttt{<BOI>}\ \mathrm{SID}(i)\ \texttt{<EOI>},
$$

where <BOI> and <EOI> denote the beginning and end of an item, respectively. This design explicitly marks item boundaries and facilitates structured generation.

MODE tokens. To represent the response intent, we introduce a set of *MODE tokens*. These tokens act as high-level indicators of the generation mode, guiding the generation behavior of the model. Specifically, we define:

- <MODE=CHAT>: the model generates free-form natural language responses without performing recommendation.

Training data construction. We construct training samples from each recommender response as follows.

(1) Non-recommendation responses. If a recommender response does not involve recommendation, we prepend:

<table><tbody><tr><td></td><td><MODE=CHAT><RESP></td><td></td><td rowspan="0">(7)</td></tr></tbody></table>

to the original response. The resulting sequence becomes:

<table><tbody><tr><td></td><td>Assistant:<MODE=CHAT><RESP>...</td><td></td><td rowspan="0">(8)</td></tr></tbody></table>

(2) Recommendation responses. If a recommender response contains a recommendation (item i), we prepend:

<table><tbody><tr><td></td><td><MODE=REC><BOI>SID(i)<EOI><RESP></td><td></td><td rowspan="0">(9)</td></tr></tbody></table>

before the ground-truth response, yielding:

<table><tbody><tr><td></td><td>Assistant:<MODE=REC><BOI>SID(i)<EOI><RESP>...</td><td></td><td rowspan="0">(10)</td></tr></tbody></table>

If multiple items appear in the same response, we create multiple training instances, each containing one target item in the <BOI>... <EOI> segment immediately following <MODE=REC>. This formulation enforces a structured generation process that directly corresponds to the factorization in Eq. 5. The model first predicts a response intent via the MODE token, which determines the subsequent generation pattern. When the response follows <MODE=CHAT>, the model generates <RESP> followed by a free-form natural language response, focusing on conversational behaviors such as preference elicitation or clarification. When the response follows <MODE=REC>, the model first generates the target item in a dedicated segment, i.e., <BOI>SID(i)<EOI>, and then produces <RESP>, followed by a natural language response conditioned on the predicted item. Such a design explicitly separates *response intent modeling*, *item prediction*, and *language generation*, thereby aligning recommendation with response generation within a unified and structured framework.

Model training. Under this factorization, the target sequence $Y$ is constructed to follow the order of $(m,i,r)$, allowing the next-token prediction objective to implicitly optimize each factor in the decomposition. Given the dialog context $C=(u_{1},\dots,u_{t-1})$ and the constructed target sequence $Y=(y_{1},\dots,y_{|Y|})$, we fine-tune a decoder-only model using the next-token prediction objective:

$$
\mathcal{L}_{\text{NTP}}=-\sum_{j=1}^{|Y|}\log P_{\theta}\left(y_{j}\mid C,y_{<j}\right).
$$

The loss is computed over all tokens after the Assistant prefix, including MODE tokens, <BOI>SID(i)<EOI>, <RESP>, and textual tokens. This unified formulation places recommendation generation at the core of the modeling process, rather than treating it as an auxiliary objective.

Inference. Our framework supports a unified and flexible inference paradigm, in which the model behavior can be either autonomously determined or influenced via MODE tokens. At inference time, the model can directly generate responses conditioned on the dialog context, starting from the Assistant: prefix. In this setting, the model predicts the appropriate MODE token based on the conversational context, thereby implicitly deciding whether to perform recommendation. Alternatively, the generation process can be guided by manually prepending a specific MODE token, which encourages the model to follow a desired behavior. Moreover, by specifying the semantic ID sequence following <MODE=REC>, the model can generate responses conditioned on a given item.

## 4 Experiments

We aim to answer the following research questions:

- RQ1: how does GCRS perform on top-k recommendation and response generation?
- RQ2: how does each component contributes to GCRS?
- RQ3: how do different LLMs impact the performance of GCRS?

Datasets. We evaluate our method on two widely used conversational recommendation benchmarks: ReDial [^16] and Inspired [^7], both consisting of multi-turn movie recommendation dialogs between users and assistants. Detailed dataset statistics are provided in Appendix B.

Baselines. We compare GCRS with two categories of methods: (1) *Conversational recommender systems*, including KGSF [^40], UniCRS [^29], MESE [^35], DCRS [^2], STEP [^36], and MSCRS [^31], which typically decouple recommendation and response generation; and (2) *generative recommendation models*, including TIGER [^22] and LC-Rec [^39], which model recommendation via semantic ID generation.

Evaluation metrics. We evaluate both recommendation and dialog quality. For recommendation, we report Recall@ $k$, NDCG@ $k$, and MRR@ $k$. For dialog quality, we use Perplexity (PPL), SacreBLEU [^21], and Distinct- $n$. Detailed metric definitions are provided in Appendix C.

Implementation. By default, we instantiate GCRS with Qwen2.5-7B-Instruct and encode item metadata using Sentence-T5 [^20] for ReDial and BGE [^34] for Inspired. We train the model using parameter-efficient fine-tuning while only updating newly introduced tokens. Full implementation details are introduced in Appendix D.

### 4.1 RQ1: performance comparison

#### 4.1.1 Recommendation performance

We report the recommendation performance on ReDial and Inspired in Table 1 and Table 2, respectively. Overall, GCRS consistently achieves state-of-the-art performance across all metrics on both datasets, with statistically significant improvements over the best baselines in most cases. We further analyze the results from three perspectives:

Comparison with generative recommenders. GCRS significantly outperforms TIGER and LC-Rec on all metrics. These methods model recommendation as pure sequence prediction over item IDs without leveraging dialog context. The inferior performance suggests that collaborative signals alone are insufficient for conversational recommendation, and modeling dynamic user preferences from dialog context is essential.

Comparison with CRS baselines. GCRS achieves consistent improvements over existing CRS methods, despite not relying on additional resources such as knowledge graphs, multimodal features, curriculum learning, or in-context learning. This indicates that a unified generative formulation can more effectively capture user preferences than modular or retrieval-based pipelines, even without external augmentation.

Performance across ranking depths. GCRS shows particularly strong gains on ranking-aware metrics (e.g., R@1, NDCG, MRR), suggesting that it can more accurately identify the most relevant items rather than merely improving recall at larger candidate sizes. This is crucial in conversational settings where only a few recommendations are presented to users.

Table 1: Recommendation performance on ReDial. The best results are bolded, the second best are underlined, and <sup>∗</sup> indicates statistical significance ($p<0.05$) compared with the best baseline.

| Model | R@1 | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 | M@5 | M@10 | M@20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KGSF | 3.52 | 11.87 | 17.91 | 26.18 | 7.63 | 9.59 | 11.67 | 6.25 | 7.06 | 7.62 |
| MESE | 4.10 | 13.50 | 21.01 | 29.45 | 8.82 | 11.24 | 13.38 | 7.29 | 8.28 | 8.87 |
| DCRS | 5.12 | 15.19 | 21.93 | 29.65 | 10.20 | 12.39 | 14.33 | 8.56 | 9.47 | 10.00 |
| STEP | 5.57 | 15.36 | 22.36 | 30.85 | 10.59 | 12.85 | 15.00 | 9.02 | 9.95 | 10.54 |
| UniCRS | 4.66 | 14.85 | 21.46 | 29.62 | 9.83 | 11.95 | 14.01 | 8.18 | 9.04 | 9.61 |
| MSCRS | 5.65 | 16.04 | 23.01 | 30.48 | 10.84 | 13.10 | 14.99 | 9.15 | 10.08 | 10.60 |
| TIGER | 2.69 | 11.16 | 17.83 | 25.41 | 6.92 | 9.06 | 10.97 | 5.53 | 6.41 | 6.93 |
| LC-Rec | 3.18 | 10.74 | 17.28 | 24.54 | 7.00 | 9.08 | 10.90 | 5.78 | 6.61 | 7.11 |
| GCRS | $\textbf{6.88}^{*}$ | $\textbf{17.73}^{*}$ | 24.32 | 31.94 | $\textbf{12.44}^{*}$ | $\textbf{14.55}^{*}$ | $\textbf{16.48}^{*}$ | $\textbf{10.70}^{*}$ | $\textbf{11.56}^{*}$ | $\textbf{12.09}^{*}$ |
| Improv. | +21.77% | +10.54% | +5.69% | +3.53% | +14.76% | +11.07% | +9.87% | +16.94% | +14.68% | +14.06% |

Table 2: Recommendation performance on Inspired. The best results are bolded, the second best are underlined, and <sup>∗</sup> indicates statistical significance ($p<0.05$) compared with the best baseline.

| Model | R@1 | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 | M@5 | M@10 | M@20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KGSF | 5.18 | 13.27 | 15.21 | 17.80 | 9.34 | 9.98 | 10.64 | 8.05 | 8.32 | 8.50 |
| MESE | 3.88 | 11.00 | 14.24 | 20.71 | 7.88 | 8.92 | 10.57 | 6.82 | 7.25 | 7.70 |
| DCRS | 4.30 | 10.55 | 16.80 | 21.88 | 7.11 | 9.09 | 10.36 | 6.02 | 6.82 | 7.15 |
| STEP | 7.03 | 17.58 | 22.27 | 27.34 | 12.67 | 14.20 | 15.51 | 11.04 | 11.68 | 12.05 |
| UniCRS | 9.77 | 19.53 | 23.83 | 30.86 | 14.69 | 16.03 | 17.77 | 13.11 | 13.63 | 14.09 |
| MSCRS | 10.16 | 20.70 | 27.73 | 33.20 | 15.46 | 17.77 | 19.14 | 13.74 | 14.72 | 15.09 |
| TIGER | 11.28 | 16.92 | 22.93 | 29.70 | 14.27 | 16.15 | 17.91 | 13.39 | 14.13 | 14.64 |
| LC-Rec | 9.02 | 17.29 | 22.18 | 25.94 | 13.21 | 14.76 | 15.71 | 11.87 | 12.50 | 12.76 |
| GCRS | $\textbf{14.56}^{*}$ | $\textbf{24.27}^{*}$ | $\textbf{29.45}^{*}$ | $\textbf{35.92}^{*}$ | $\textbf{19.55}^{*}$ | $\textbf{21.26}^{*}$ | $\textbf{22.88}^{*}$ | $\textbf{18.00}^{*}$ | $\textbf{18.72}^{*}$ | $\textbf{19.16}^{*}$ |
| Improv. | +29.08% | +17.25% | +6.20% | +8.19% | +26.46% | +19.64% | +19.54% | +31.00% | +27.17% | +26.97% |

Table 3: Dialog metrics. The best results are bolded, the second best are underlined, and <sup>∗</sup> indicates statistical significance ($p<0.05$) compared with the best baseline.

<table><tbody><tr><td></td><td colspan="6">ReDial</td><td colspan="6">Inspired</td></tr><tr><td></td><td>PPL</td><td>BLEU</td><td>D-1</td><td>D-2</td><td>D-3</td><td>D-4</td><td>PPL</td><td>BLEU</td><td>D-1</td><td>D-2</td><td>D-3</td><td>D-4</td></tr><tr><td>KGSF</td><td>237.68</td><td>26.65</td><td>0.0020</td><td>0.0504</td><td>0.1852</td><td>0.3513</td><td>464.84</td><td>11.82</td><td>0.0019</td><td>0.0153</td><td>0.0395</td><td>0.0624</td></tr><tr><td>MESE</td><td>8.48</td><td>4.53</td><td>0.0218</td><td>0.0927</td><td>0.1814</td><td>0.2633</td><td>15.44</td><td>5.23</td><td>0.0730</td><td>0.2200</td><td>0.3385</td><td>0.4068</td></tr><tr><td>DCRS</td><td>12.14</td><td>9.52</td><td>0.0383</td><td>0.1443</td><td>0.2638</td><td>0.3806</td><td>21.13</td><td>5.27</td><td>0.0826</td><td>0.2378</td><td>0.3899</td><td>0.5195</td></tr><tr><td>STEP</td><td>85.67</td><td>3.70</td><td>0.0023</td><td>0.0112</td><td>0.0781</td><td>0.1544</td><td>18.14</td><td>5.58</td><td>0.1080</td><td>0.3168</td><td>0.4988</td><td>0.6280</td></tr><tr><td>UniCRS</td><td>34.00</td><td>11.14</td><td>0.0169</td><td>0.0603</td><td>0.1186</td><td>0.1880</td><td>32.43</td><td>4.40</td><td>0.0659</td><td>0.2009</td><td>0.3386</td><td>0.4577</td></tr><tr><td>MSCRS</td><td>10.09</td><td>8.98</td><td>0.0353</td><td>0.1278</td><td>0.2374</td><td>0.3483</td><td>15.55</td><td>5.71</td><td>0.0968</td><td>0.2821</td><td>0.4513</td><td>0.5771</td></tr><tr><td>GCRS</td><td><math><semantics><msup><mtext>4.37</mtext> <mo>∗</mo></msup> <annotation>\textbf{4.37}^{*}</annotation></semantics></math></td><td>11.58</td><td>0.0303</td><td>0.1216</td><td>0.2333</td><td>0.3427</td><td><math><semantics><msup><mtext>6.36</mtext> <mo>∗</mo></msup> <annotation>\textbf{6.36}^{*}</annotation></semantics></math></td><td><math><semantics><munder><mn>5.75</mn> <mo>¯</mo></munder> <annotation>\underline{5.75}</annotation></semantics></math></td><td><math><semantics><msup><mtext>0.1406</mtext> <mo>∗</mo></msup> <annotation>\textbf{0.1406}^{*}</annotation></semantics></math></td><td><math><semantics><msup><mtext>0.3887</mtext> <mo>∗</mo></msup> <annotation>\textbf{0.3887}^{*}</annotation></semantics></math></td><td><math><semantics><msup><mtext>0.5661</mtext> <mo>∗</mo></msup> <annotation>\textbf{0.5661}^{*}</annotation></semantics></math></td><td><math><semantics><msup><mtext>0.6789</mtext> <mo>∗</mo></msup> <annotation>\textbf{0.6789}^{*}</annotation></semantics></math></td></tr></tbody></table>

#### 4.1.2 Response generation performance

Table 3 reports the dialog generation performance on both ReDial and Inspired across three aspects: fluency (PPL), lexical overlap (BLEU), and diversity (Distinct- $n$).

Fluency. Overall, GCRS achieves the best fluency on both datasets, significantly outperforming all baselines with a large margin in PPL. This indicates that modeling conversational recommendation as a unified next-token prediction problem leads to more coherent and well-formed responses.

Lexical overlap. In terms of BLEU, we observe that GCRS achieves competitive but not dominant performance. It obtains the second-best score, while KGSF attains the highest BLEU despite extremely poor PPL and diversity. This suggests that BLEU may favor conservative or template-like responses, which partially explains why early methods such as KGSF achieve high BLEU scores despite weak generative quality. In contrast, GCRS maintains a better balance between faithfulness and generation flexibility.

Diversity. For Distinct- $n$, GCRS shows different behaviors across datasets. On ReDial, methods like DCRS and MSCRS achieve higher Distinct scores, indicating more diverse but potentially less stable generation. In contrast, on Inspired, GCRS consistently achieves the best performance across all Distinct- $n$ metrics, demonstrating its strong ability to generate diverse and informative responses. This improvement can be attributed to the structured generation paradigm, which conditions response generation on predicted items and encourages richer contextualization.

Table 4: Ablation study on (i) semantic ID configurations, (ii) structured generation components, and (iii) embedding training strategies. The best results are bolded, the second best are underlined.

<table><tbody><tr><td></td><td colspan="5">ReDial</td><td colspan="5">Inspired</td></tr><tr><td></td><td>R@1</td><td>N@10</td><td>PPL</td><td>BLEU</td><td>D-2</td><td>R@1</td><td>N@10</td><td>PPL</td><td>BLEU</td><td>D-2</td></tr><tr><td>SID (<math><semantics><mrow><mn>3</mn> <mo>×</mo> <mn>64</mn></mrow> <annotation>3\times 64</annotation></semantics></math>)</td><td>6.66</td><td>13.69</td><td>4.90</td><td>11.14</td><td>0.1205</td><td>9.06</td><td>17.44</td><td>6.46</td><td>5.85</td><td>0.3824</td></tr><tr><td>SID (<math><semantics><mrow><mn>5</mn> <mo>×</mo> <mn>64</mn></mrow> <annotation>5\times 64</annotation></semantics></math>)</td><td>6.53</td><td>13.99</td><td>4.24</td><td>11.08</td><td>0.1234</td><td>12.95</td><td>20.01</td><td>6.27</td><td>4.67</td><td>0.4082</td></tr><tr><td>SID (<math><semantics><mrow><mn>4</mn> <mo>×</mo> <mn>32</mn></mrow> <annotation>4\times 32</annotation></semantics></math>)</td><td>6.41</td><td>13.54</td><td>4.50</td><td>11.21</td><td>0.1253</td><td>12.62</td><td>21.29</td><td>6.24</td><td>5.60</td><td>0.3913</td></tr><tr><td>SID (<math><semantics><mrow><mn>4</mn> <mo>×</mo> <mn>128</mn></mrow> <annotation>4\times 128</annotation></semantics></math>)</td><td>6.33</td><td>13.12</td><td>4.66</td><td>11.01</td><td>0.1317</td><td>13.92</td><td>19.57</td><td>6.51</td><td>6.00</td><td>0.3819</td></tr><tr><td>RESP</td><td>3.39</td><td>7.20</td><td>6.91</td><td>8.27</td><td>0.1171</td><td>4.85</td><td>8.41</td><td>8.32</td><td>4.73</td><td>0.3413</td></tr><tr><td>MODE+RESP</td><td>4.06</td><td>9.91</td><td>6.20</td><td>11.95</td><td>0.1202</td><td>8.41</td><td>16.88</td><td>7.65</td><td>5.74</td><td>0.3708</td></tr><tr><td>SID</td><td>5.16</td><td>11.82</td><td>—</td><td>—</td><td>—</td><td>11.00</td><td>19.52</td><td>—</td><td>—</td><td>—</td></tr><tr><td>Full embedding fine-tuning</td><td>5.84</td><td>12.47</td><td>4.60</td><td>11.19</td><td>0.1251</td><td>12.62</td><td>19.38</td><td>6.37</td><td>5.38</td><td>0.3863</td></tr><tr><td>GCRS</td><td>6.88</td><td>14.55</td><td>4.37</td><td>11.58</td><td>0.1216</td><td>14.56</td><td>21.26</td><td>6.36</td><td>5.75</td><td>0.3887</td></tr></tbody></table>

### 4.2 RQ2: study on model components

Table 4 presents an ablation study analyzing the impact of (i) semantic ID configurations, (ii) structured generation components, and (iii) embedding training strategies. For semantic ID configurations, rows labeled as SID ($\cdot$) vary the number of codebooks and codebook size (e.g., $4\times 64$ denotes 4 codebooks with 64 entries each, which is also our default setting). For structured generation, we compare different supervision targets: “RESP” uses only the natural language response, “MODE+RESP” further includes MODE tokens, and “SID” uses only semantic IDs without response generation. For training strategy, “Full embedding fine-tuning” updates the entire vocabulary embeddings, in contrast to our default setting which only trains newly introduced tokens. “GCRS” denotes the full model with the default configuration.

Semantic ID configurations. Different codebook sizes and depths lead to comparable performance, suggesting that the model is relatively robust to the exact SID design. However, extreme configurations (e.g., $4\times 128$ or $3\times 64$) may slightly degrade performance, indicating that overly large or insufficient code capacity may harm the balance between expressiveness and learnability. The default setting ($4\times 64$) achieves the most stable performance across datasets, providing a good trade-off.

Structured generation components. The comparison among different training targets highlights the importance of structured generation. Using only natural language supervision (“RESP”) results in a dramatic drop in recommendation performance, showing that standard language modeling alone is insufficient for accurate item prediction. Introducing a response intent indicator (“MODE+RESP”) significantly improves performance, showing the benefit of explicitly modeling high-level generation intent. However, it still falls short of the full model, suggesting that intent modeling alone is insufficient. The “SID” variant achieves reasonable recommendation performance but lacks the ability to generate responses, and remains inferior to the full model. This demonstrates that jointly modeling item prediction and response generation is beneficial for effective conversational recommendation.

Embedding training strategy. Full embedding fine-tuning leads to worse recommendation performance compared to the default setting. This suggests that updating the entire vocabulary may disrupt the pretrained semantic structure of the LLM, whereas restricting training to newly introduced tokens better preserves general language understanding while adapting to recommendation-specific representations.

Table 5: Impact of LLMs on ReDial. The best results for backbones are bolded, and the best results for encoders are underlined.

| Encoder | Backbone | R@1 | R@10 | N@10 | M@10 | PPL | BLEU | D@2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sentence-T5 | Llama | 6.00 | 21.65 | 12.82 | 10.13 | 4.73 | 10.74 | 0.0992 |
| Sentence-T5 | Qwen3 | 6.29 | 22.22 | 13.42 | 10.71 | 4.55 | 6.99 | 0.0748 |
| Sentence-T5 | Mistral | 5.45 | 21.69 | 12.51 | 9.71 | 4.69 | 10.70 | 0.1096 |
| Sentence-T5 | Ministral3 | 6.04 | 20.11 | 12.32 | 9.92 | 4.90 | 7.59 | 0.0690 |
| Sentence-T5 | Qwen2.5 | 6.88 | 24.32 | 14.55 | 11.56 | 4.37 | 11.58 | 0.1216 |
| E5 | Qwen2.5 | 6.31 | 21.32 | 12.81 | 10.21 | 4.62 | 11.24 | 0.1185 |
| Llama | Qwen2.5 | 6.74 | 22.71 | 13.69 | 10.93 | 4.54 | 11.12 | 0.1183 |
| Mxbai | Qwen2.5 | 6.45 | 21.75 | 12.99 | 10.33 | 4.55 | 11.65 | 0.1226 |
| BGE | Qwen2.5 | 6.47 | 22.79 | 13.60 | 10.80 | 4.57 | 11.06 | 0.1191 |

Table 6: Impact of LLMs on Inspired. The best results for backbones are bolded, and the best results for encoders are underlined.

| Encoder | Backbone | R@1 | R@10 | N@10 | M@10 | PPL | BLEU | D@2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BGE | Llama | 13.27 | 30.10 | 21.36 | 18.61 | 6.49 | 6.28 | 0.3127 |
| BGE | Qwen3 | 12.95 | 32.04 | 22.15 | 19.01 | 6.63 | 4.96 | 0.2756 |
| BGE | Mistral | 10.36 | 30.10 | 19.27 | 15.90 | 5.77 | 5.82 | 0.3544 |
| BGE | Ministral3 | 14.24 | 28.80 | 21.31 | 18.93 | 6.52 | 5.81 | 0.3319 |
| BGE | Qwen2.5 | 14.56 | 29.45 | 21.26 | 18.72 | 6.36 | 5.75 | 0.3887 |
| E5 | Qwen2.5 | 14.89 | 28.48 | 21.45 | 19.22 | 6.27 | 5.28 | 0.3797 |
| Llama | Qwen2.5 | 13.92 | 25.57 | 19.31 | 17.37 | 6.36 | 5.55 | 0.3894 |
| Mxbai | Qwen2.5 | 10.36 | 27.18 | 18.78 | 16.08 | 6.36 | 5.58 | 0.3803 |
| Sentence-T5 | Qwen2.5 | 12.62 | 28.80 | 20.30 | 17.64 | 6.36 | 5.08 | 0.3791 |

### 4.3 RQ3: impact of different LLMs

We analyze the impact of different backbone LLMs and semantic ID encoders on both recommendation and dialog performance, as shown in Table 5 and Table 6.

Backbone LLMs. We evaluate GCRS with five representative models, including Llama-3.1-8B [^6], Qwen3-8B [^25], Mistral-7B [^10], Ministral-3-8B [^18], and the default Qwen2.5-7B [^24]. We fix the item metadata encoder as default (Sentence-T5 for ReDial and BGE for Inspired) for a fair comparison. We observe that different backbones exhibit comparable results, with no single model consistently dominating across all metrics. Among them, Qwen2.5 achieves the most balanced performance, attaining strong recommendation accuracy while maintaining competitive dialog quality. Other backbones show distinct trade-offs. For instance, Qwen3 achieves competitive ranking performance, while Mistral-based models tend to yield lower perplexity but relatively weaker recommendation metrics. These results suggest that while the backbone choice influences specific aspects of performance, the overall effectiveness of GCRS is relatively robust across different LLMs.

Semantic ID encoders. We further examine the impact of different text encoders for constructing semantic IDs while fixing the backbone to Qwen2.5. Besides Sentence-T5 [^20] and BGE [^34], we tested E5 [^27], Llama [^26], and Mxbai [^11]. Different encoders show relatively small performance gaps, indicating that the proposed framework is robust to the choice of text encoder. On ReDial, Sentence-T5 achieves the best recommendation performance, while on Inspired, E5 and BGE show comparable results with slightly different strengths across metrics. In terms of dialog quality, no single encoder consistently dominates. These results suggest that different encoders capture complementary aspects of item semantics, but their overall impact remains limited.

## 5 Conclusion

We present GCRS, a fully generative conversational recommender system that integrates recommendation and dialog generation within a unified autoregressive framework. By representing items as semantic IDs and introducing a structured generation paradigm, our approach models conversational recommendation as a sequence of interdependent decisions, enabling end-to-end optimization with an explicit and well-aligned dependency structure. Extensive experiments on benchmark datasets demonstrate that GCRS significantly improves recommendation performance while maintaining competitive dialog quality.

## References

## Appendix A Collision resolution

Suppose $N$ items collide. For each colliding item, we compute the distances between its residual vectors and all codewords at every quantization level:

$$
d_{i,k}^{(l)}=\left\|\mathbf{r}_{i}^{(l)}-\mathbf{c}_{k}^{(l)}\right\|_{2}^{2},
$$

which yields a distance tensor

$$
\mathbf{D}\in\mathbb{R}^{N\times L\times K}.
$$

For each item and each level, all candidate codewords are sorted according to their distances.

We then rank the colliding items according to their minimum distance at the last quantization level, so that the item with the most confident last-level assignment is processed first. Starting from the last level, we assign to each item its nearest available codeword; if this choice still causes a collision, we use the next nearest codeword instead. When the available codewords at the last level are insufficient to resolve all collisions, we backtrack to the previous level, reallocate the code at that level according to distance ranking, and then reassign the subsequent levels accordingly. This procedure is repeated until all colliding items obtain unique semantic IDs.

## Appendix B Datasets

ReDial contains 10,006 training dialogs and 1,342 test dialogs, covering 6,924 unique items. Inspired contains 900 training dialogs and 99 test dialogs, covering 1,782 items. For each dialog, we construct training samples by treating each assistant utterance as the target response and the preceding dialog context as input. Movie metadata is collected from IMDb <sup>2</sup> and serialized into textual descriptions for semantic ID construction.

## Appendix C Dialog evaluation metrics

We evaluate the generated responses from both the language quality and diversity perspectives using Perplexity (PPL), SacreBLEU, and Distinct- $n$.

### C.1 Perplexity

Perplexity measures how well a probabilistic model predicts a sequence. Given a sequence of tokens $y=(y_{1},y_{2},\dots,y_{T})$, the sequence-level perplexity is defined as:

$$
\mathrm{PPL}(y)=\exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log p(y_{t}\mid y_{<t})\right).
$$

For a corpus $\mathcal{D}$ consisting of multiple sequences, we compute the corpus-level perplexity by aggregating over all tokens:

$$
\mathrm{PPL}(\mathcal{D})=\exp\left(-\frac{\sum_{i=1}^{N}\sum_{t=1}^{T_{i}}\log p(y_{i,t}\mid y_{i,<t})}{\sum_{i=1}^{N}T_{i}}\right),
$$

where $N$ is the number of sequences, and $T_{i}$ is the length of the $i$ -th sequence. This formulation corresponds to the exponentiated average negative log-likelihood per token across the entire corpus, ensuring that longer sequences contribute proportionally more to the final score.

### C.2 BLEU

BLEU evaluates the overlap between generated text and reference text based on $n$ -gram precision with a brevity penalty. It is defined as:

$$
\mathrm{BLEU}=\mathrm{BP}\cdot\exp\left(\sum_{n=1}^{N}w_{n}\log p_{n}\right),
$$

where $p_{n}$ is the modified $n$ -gram precision:

$$
p_{n}=\frac{\sum_{\text{ngram}\in y}\min(\mathrm{count}_{\text{gen}}(\text{ngram}),\mathrm{count}_{\text{ref}}(\text{ngram}))}{\sum_{\text{ngram}\in y}\mathrm{count}_{\text{gen}}(\text{ngram})},
$$

$w_{n}$ are weights (typically $w_{n}=\frac{1}{N}$), and $\mathrm{BP}$ is the brevity penalty:

$$
\mathrm{BP}=\begin{cases}1&\text{if }c>r,\\
\exp(1-\frac{r}{c})&\text{if }c\leq r,\end{cases}
$$

where $c$ and $r$ are the lengths of the candidate and reference sentences, respectively. We report SacreBLEU (scaled by 100).

### C.3 Distinct-nn

Distinct- $n$ measures the diversity of generated text by calculating the ratio of unique $n$ -grams over the total number of generated $n$ -grams:

$$
\mathrm{Distinct}\text{-}n=\frac{|\mathcal{G}_{n}^{\text{unique}}|}{|\mathcal{G}_{n}|},
$$

where $\mathcal{G}_{n}$ is the set of all $n$ -grams in the generated corpus, and $\mathcal{G}_{n}^{\text{unique}}$ is the set of distinct $n$ -grams. We report Distinct- $n$ for $n\in\{1,2,3,4\}$.

## Appendix D Implementation details

Baseline implementation. We follow official implementations for all baselines and ensure consistent evaluation settings. For TIGER and LC-Rec, we use the same semantic IDs as in GCRS for a fair comparison.

Semantic ID construction. We construct semantic IDs following Sec. 3.2. Specifically, item metadata is serialized into textual descriptions and encoded into dense embeddings using pretrained text encoders. We use Sentence-T5 [^20] for ReDial and BGE [^34] for Inspired. We apply a 4-layer RQ-VAE to quantize each embedding into a discrete semantic ID with 4 tokens, each selected from a codebook of size 64, resulting in a capacity of $64^{4}$ possible IDs. To ensure uniqueness, we apply the collision resolution strategy described in Appendix A.

Model training. We adopt Qwen2.5-7B-Instruct [^24] as the backbone model and perform parameter-efficient fine-tuning using QLoRA [^4] on all linear layers. For token embeddings, we freeze the original vocabulary embeddings and only train the embeddings of newly introduced tokens, including semantic ID tokens and control tokens (e.g., <BOI>, <EOI>, <RESP>, <MODE=REC> and <MODE=CHAT>). Detailed hyperparameters are provided in Table 7. Experiments are conducted on a single NVIDIA RTX 6000 Ada GPU (48GB VRAM).

Table 7: Hyper-parameter setting.

|  | ReDial | Inspired |
| --- | --- | --- |
| RQ-VAE |  |  |
| learning rate | 1e-3 | 1e-3 |
| weight decay | 1e-4 | 1e-4 |
| batch size | 1,024 | 1,024 |
| encoder layer | 7 | 7 |
| encoder output size | 32 | 32 |
| codebook layer | 4 | 4 |
| codebook size | 64 | 64 |
| Fine-tuning |  |  |
| learning rate | 2e-4 | 1e-4 |
| batch size | 72 | 72 |
| warmup steps | 150 | 50 |
| training steps | 1,800 | 240 |
| input max length | 768 | 768 |
| weight decay | 0.0 | 0.0 |
| lora rank | 16 | 16 |
| lora alpha | 32 | 32 |
| lora dropout | 0.05 | 0.05 |
| rec beam width | 50 | 50 |

Evaluation protocol. To ensure a fair comparison with existing conversational recommender systems, we adopt a controlled evaluation protocol for recommendation metrics. Specifically, we prepend <MODE=REC> to all evaluation samples whose ground-truth responses contain recommendations, and <MODE=CHAT> otherwise. This design is necessary because, under our unified autoregressive formulation, the model may choose not to generate any item tokens, leading to missing predictions and making ranking-based metrics undefined. In contrast, standard CRS evaluation assumes that a model produces a ranked list of candidate items for every input, enabling consistent computation of recommendation metrics. To bridge this mismatch, we enforce the recommendation mode during evaluation to ensure that a valid candidate list is always produced, aligning our evaluation protocol with prior work. For recommendation evaluation, we perform constrained beam search over the semantic ID space starting from the first <BOI> token to produce a top- $k$ candidate list. For response evaluation, we replace all semantic IDs in the generated responses with a <movie> placeholder, which is treated as a single token when computing dialog metrics, following common practice in prior work. We report average number over 5 runs.

## Appendix E Broader impact and limitations

### E.1 Limitations

While our proposed generative conversational recommender system shows promising results, several limitations should be noted. First, our framework relies on supervised fine-tuning on existing conversational recommendation datasets, which are relatively small and may not fully capture the diversity and complexity of real-world user preferences. Second, the effectiveness of semantic IDs depends on the quality of the underlying embedding and quantization process; suboptimal representations may negatively impact recommendation accuracy. Third, our experiments are conducted on benchmark datasets, and further evaluation in large-scale, real-world deployment scenarios is needed to assess scalability, robustness, and user satisfaction.

### E.2 Societal impact

This work has both potential positive and negative societal impacts. On the positive side, our approach enables more natural, context-aware, and personalized conversational recommendations, which may improve user experience in applications such as education, entertainment, and e-commerce. On the negative side, the system may inherit and amplify biases present in training data, potentially leading to unfair or unbalanced recommendations. Moreover, tightly integrating recommendation into natural language generation may increase the risk of persuasive or manipulative recommendations, which could influence user decisions without sufficient transparency. There are also potential privacy concerns if sensitive user preferences are inferred or exploited. These risks highlight the importance of incorporating fairness evaluation, transparency mechanisms, and user control in real-world deployments.

### E.3 Responsible release and safeguards

We acknowledge that systems combining large language models with recommendation capabilities may pose risks if misused. In this work, we do not release any sensitive user data, and all experiments are conducted on publicly available benchmark datasets. Our method does not introduce new forms of personal data collection. For responsible deployment, we recommend incorporating safeguards such as content filtering, user consent mechanisms, and transparency regarding when and how recommendations are generated. In addition, our structured generation paradigm provides an explicit interface (e.g., deciding whether to trigger recommendations), which may facilitate safer and more transparent system behavior. Future work should further investigate mechanisms for bias mitigation, privacy protection, and improved control over recommendation behavior to ensure responsible use in real-world applications.

[^1]: Q. Chen, J. Lin, Y. Zhang, M. Ding, Y. Cen, H. Yang, and J. Tang (2019) Towards knowledge-based recommender dialog system. arXiv preprint arXiv:1908.05391. Cited by: §2.

[^2]: H. Dao, Y. Deng, D. D. Le, and L. Liao (2024) Broadening the view: demonstration-augmented prompt learning for conversational recommendation. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 785–795. Cited by: §1, §2, §4.

[^3]: J. Deng, S. Wang, K. Cai, L. Ren, Q. Hu, W. Ding, Q. Luo, and G. Zhou (2025) OneRec: unifying retrieve and rank with generative recommender and iterative preference alignment. External Links: 2502.18965, [Link](https://arxiv.org/abs/2502.18965) Cited by: §2.

[^4]: T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer (2023) Qlora: efficient finetuning of quantized llms. Advances in neural information processing systems 36, pp. 10088–10115. Cited by: Appendix D.

[^5]: L. Friedman, S. Ahuja, D. Allen, Z. Tan, H. Sidahmed, C. Long, J. Xie, G. Schubiner, A. Patel, H. Lara, et al. (2023) Leveraging large language models in conversational recommender systems. arXiv preprint arXiv:2305.07961. Cited by: §1, §2.

[^6]: A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al. (2024) The llama 3 herd of models. arXiv preprint arXiv:2407.21783. External Links: [Link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) Cited by: §4.3.

[^7]: S. A. Hayati, D. Kang, Q. Zhu, W. Shi, and Z. Yu (2020) Inspired: toward sociable recommendation dialog systems. arXiv preprint arXiv:2009.14306. Cited by: §4.

[^8]: Z. He, Z. Xie, R. Jha, H. Steck, D. Liang, Y. Feng, B. P. Majumder, N. Kallus, and J. McAuley (2023) Large language models as zero-shot conversational recommenders. In Proceedings of the 32nd ACM international conference on information and knowledge management, pp. 720–730. Cited by: §3.2.

[^9]: Z. He, Z. Xie, H. Steck, D. Liang, R. Jha, N. Kallus, and J. McAuley (2025) Reindex-then-adapt: improving large language models for conversational recommendation. In Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining, pp. 866–875. Cited by: §3.2.

[^10]: A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de las Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed (2023) Mistral 7b. External Links: 2310.06825, [Link](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) Cited by: §4.3.

[^11]: S. Lee, A. Shakir, D. Koenig, and J. Lipp (2024)Open source strikes bread - new fluffy embeddings model(Website) External Links: [Link](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) Cited by: §4.3.

[^12]: C. Li, Y. Deng, H. Hu, M. Kan, and H. Li (2024) Incorporating external knowledge and goal guidance for llm-based conversational recommender systems. arXiv preprint arXiv:2405.01868. Cited by: §2.

[^13]: C. Li, Y. Deng, H. Hu, S. Ng, M. Kan, and H. Li (2025) CARE: contextual adaptation of recommenders for llm-based conversational recommendation. arXiv preprint arXiv:2508.13889. Cited by: §2.

[^14]: C. Li, W. Liang, H. Hu, S. Ng, M. Kan, H. Li, and Y. Deng (2026) Improving conversational recommendation with contextual adaptation of external recommenders and llm-based reranking. In European Conference on Information Retrieval, pp. 204–221. Cited by: §1, §2.

[^15]: K. Li, R. Xiang, Y. Bai, Y. Tang, Y. Cheng, X. Liu, P. Jiang, and K. Gai (2025) BBQRec: behavior-bind quantization for multi-modal sequential recommendation. External Links: 2504.06636, [Link](https://arxiv.org/abs/2504.06636) Cited by: §2.

[^16]: R. Li, S. Ebrahimi Kahou, H. Schulz, V. Michalski, L. Charlin, and C. Pal (2018) Towards deep conversational recommendations. Advances in neural information processing systems 31. Cited by: §1, §2, §4.

[^17]: D. Lin, J. Wang, and W. Li (2023) Cola: improving conversational recommender systems by collaborative augmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 37, pp. 4462–4470. Cited by: §1, §2, §2.

[^18]: A. H. Liu, K. Khandelwal, S. Subramanian, V. Jouault, A. Rastogi, A. Sadé, A. Jeffares, A. Jiang, A. Cahill, A. Gavaudan, et al. (2026) Ministral 3. arXiv preprint arXiv:2601.08584. External Links: [Link](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16) Cited by: §4.3.

[^19]: Y. Lu, J. Bao, Y. Song, Z. Ma, S. Cui, Y. Wu, and X. He (2021) RevCore: review-augmented conversational recommendation. arXiv preprint arXiv:2106.00957. Cited by: §2.

[^20]: J. Ni, G. H. Abrego, N. Constant, J. Ma, K. Hall, D. Cer, and Y. Yang (2022) Sentence-t5: scalable sentence encoders from pre-trained text-to-text models. In Findings of the association for computational linguistics: ACL 2022, pp. 1864–1874. External Links: [Link](https://huggingface.co/sentence-transformers/sentence-t5-large) Cited by: Appendix D, §4.3, §4.

[^21]: M. Post (2018) A call for clarity in reporting bleu scores. In Proceedings of the third conference on machine translation: Research papers, pp. 186–191. Cited by: §4.

[^22]: S. Rajput, N. Mehta, A. Singh, R. Hulikal Keshavan, T. Vu, L. Heldt, L. Hong, Y. Tay, V. Tran, J. Samost, M. Kula, E. Chi, and M. Sathiamoorthy (2023) Recommender systems with generative retrieval. In Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (Eds.), Vol. 36, pp. 10299–10315. External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf) Cited by: §2, §3.2, §4.

[^23]: M. Ravaut, H. Zhang, L. Xu, A. Sun, and Y. Liu (2024) Parameter-efficient conversational recommender system as a language processing task. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 152–165. Cited by: §2.

[^24]: Q. Team (2024-09) Qwen2.5: a party of foundation models. External Links: [Link](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) Cited by: Appendix D, §4.3.

[^25]: Q. Team (2025) Qwen3 technical report. External Links: 2505.09388, [Link](https://huggingface.co/Qwen/Qwen3-8B) Cited by: §4.3.

[^26]: H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. (2023) Llama: open and efficient foundation language models. arXiv preprint arXiv:2302.13971. External Links: [Link](https://huggingface.co/huggyllama/llama-7b) Cited by: §4.3.

[^27]: L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D. Jiang, R. Majumder, and F. Wei (2022) Text embeddings by weakly-supervised contrastive pre-training. arXiv preprint arXiv:2212.03533. External Links: [Link](https://huggingface.co/intfloat/e5-large-v2) Cited by: §4.3.

[^28]: L. Wang, H. Hu, L. Sha, C. Xu, D. Jiang, and K. Wong (2022) RecInDial: a unified framework for conversational recommendation with pretrained language models. In Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 489–500. Cited by: §2.

[^29]: X. Wang, K. Zhou, J. Wen, and W. X. Zhao (2022) Towards unified conversational recommender systems via knowledge-enhanced prompt learning. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 1929–1937. Cited by: §1, §2, §4.

[^30]: Y. Wang, Z. Ren, W. Sun, J. Yang, Z. Liang, X. Chen, R. Xie, S. Yan, X. Zhang, P. Ren, Z. Chen, and X. Xin (2024) Content-based collaborative generation for recommender systems. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, CIKM ’24, New York, NY, USA, pp. 2420–2430. External Links: ISBN 9798400704369, [Link](https://doi.org/10.1145/3627673.3679692), [Document](https://dx.doi.org/10.1145/3627673.3679692) Cited by: §2.

[^31]: Y. Wei, J. Zou, W. Guo, G. Wang, X. Xu, and Y. Yang (2025) MSCRS: multi-modal semantic graph prompt learning framework for conversational recommender systems. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 42–52. Cited by: §1, §2, §4.

[^32]: Y. Xi, W. Liu, J. Lin, B. Chen, R. Tang, W. Zhang, and Y. Yu (2024) Memocrs: memory-enhanced sequential conversational recommender systems with large language models. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, pp. 2585–2595. Cited by: §1, §2.

[^33]: L. Xiao, H. Wang, C. Wang, L. Ji, Y. Wang, J. Zhu, Z. Dong, R. Zhang, and R. Li (2025-10) UNGER: generative recommendation with a unified code via semantic and collaborative integration. ACM Trans. Inf. Syst.. Note: Just Accepted External Links: ISSN 1046-8188, [Link](https://doi.org/10.1145/3773771), [Document](https://dx.doi.org/10.1145/3773771) Cited by: §2.

[^34]: S. Xiao, Z. Liu, P. Zhang, and N. Muennighoff (2023) C-pack: packaged resources to advance general chinese embedding. External Links: 2309.07597, [Link](https://huggingface.co/BAAI/bge-large-en-v1.5) Cited by: Appendix D, §4.3, §4.

[^35]: B. Yang, C. Han, Y. Li, L. Zuo, and Z. Yu (2021) Improving conversational recommendation systems’ quality with context-aware item meta information. arXiv preprint arXiv:2112.08140. Cited by: §2, §4.

[^36]: Z. Yang, J. Chen, H. Li, X. Jin, X. Li, J. Zhang, H. Gao, K. Wei, and S. Wang (2025) STEP: stepwise curriculum learning for context-knowledge fusion in conversational recommendation. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management, pp. 3824–3833. Cited by: §2, §4.

[^37]: J. Zhai, Z. Mai, C. Wang, F. Yang, X. Zheng, H. Li, and Y. Tian (2025) Multimodal quantitative language for generative recommendation. External Links: 2504.05314, [Link](https://arxiv.org/abs/2504.05314) Cited by: §2, §3.2.

[^38]: F. Zhang, X. Liu, D. Xi, J. Yin, H. Chen, P. Yan, F. Zhuang, and Z. Zhang (2025) Multi-aspect cross-modal quantization for generative recommendation. External Links: 2511.15122, [Link](https://arxiv.org/abs/2511.15122) Cited by: §2.

[^39]: B. Zheng, Y. Hou, H. Lu, Y. Chen, W. X. Zhao, M. Chen, and J. Wen (2024) Adapting large language models by integrating collaborative semantics for recommendation. In 2024 IEEE 40th International Conference on Data Engineering (ICDE), Vol., pp. 1435–1448. External Links: [Document](https://dx.doi.org/10.1109/ICDE60146.2024.00118) Cited by: §2, §4.

[^40]: K. Zhou, W. X. Zhao, S. Bian, Y. Zhou, J. Wen, and J. Yu (2020) Improving conversational recommender systems via knowledge graph based semantic fusion. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 1006–1014. Cited by: §1, §2, §4.

[^41]: Y. Zhou, K. Zhou, W. X. Zhao, C. Wang, P. Jiang, and H. Hu (2022) C <sup>2</sup> -crs: coarse-to-fine contrastive learning for conversational recommender system. In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining, pp. 1488–1496. Cited by: §2.

[^42]: Y. Zhu, C. Wan, H. Steck, D. Liang, Y. Feng, N. Kallus, and J. Li (2025) Collaborative retrieval for large language model-based conversational recommender systems. In Proceedings of the ACM on Web Conference 2025, pp. 3323–3334. Cited by: §2.