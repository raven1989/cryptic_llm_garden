---
title: "SeqLLM: Augmenting LLMs with Behavioral-Sequence Modeling for High-Stakes Decisions at WeChat Pay"
source: "https://arxiv.org/html/2608.03063v1"
author:
published:
created: 2026-08-09
description:
tags:
  - "clippings"
---
Guilin Li WeChat Pay, TencentChina, Jiaxing Zhang Shanghai Jiao Tong UniversityChina, Matthias Hwai Yong Tan City University of Hong KongHong Kong SAR, China, Bo Wang WeChat Pay, TencentChina and Weiran Huang Shanghai Jiao Tong UniversityChina

###### Abstract.

Merchant risk control at large payment platforms screens tens of millions of merchants per day, where errors are costly in both directions: a false positive may harm a legitimate merchant, while a false negative leaves harmful activity undetected. The hardest cases cannot be settled from either modality alone; they require reading a merchant’s textual profile jointly with its long behavioral sequence. Large language models (LLMs) excel at the former but cannot natively model behavioral sequences, and endowing them with this ability typically erodes their language and reasoning skills through catastrophic forgetting. We present SeqLLM, a framework that injects behavioral-sequence modeling into a pretrained LLM while preserving its language ability, enabling joint use of content and behavioral sequences. SeqLLM has three components: a compact discrete behavior vocabulary that represents each event as native tokens; a lightweight projector, trained with a two-stage alignment curriculum, that grounds behavior tokens in the LLM’s semantic space; and prefix-guided capability injection, which learns sequence modeling through task-prefixed supervised fine-tuning rather than continual pre-training. SeqLLM is fully deployed at WeChat Pay, screening millions of merchants per day. Compared with the production DeepSeek-based LLM baseline, it improves screening precision from 92.0% to 97.5%; its pretrained behavior-token embeddings also lift Precision@Top-0.01% by 26.8 pp in a production fraud detector serving billion-scale transaction traffic. Beyond payment behavior, SeqLLM achieves state-of-the-art results against strong public baselines on open recommendation benchmarks. On MovieLens and Amazon, it outperforms the strong User-LLM baseline by up to 32% relative Recall@5 while retaining markedly stronger language ability. On the RecIF benchmark, SeqLLM improves Pass@32 by 14.2% over the full OneRec-8B pipeline, while using only one-fifth of OneRec-8B’s GPU-days. Our code is available at [https://github.com/125jx/SeqLLM.git](https://github.com/125jx/SeqLLM.git).

## 1\. Introduction

At WeChat Pay, merchants include both offline businesses and online payment-accepting entities, such as e-commerce platforms, mini-programs, and apps. Merchant risk control screens millions of merchants each day for fraud, money laundering, and other illicit fund flows [^17] [^15]. Criminals may create online stores that appear legitimate and use them to collect illicit payments. Errors are costly in both directions: a false positive may harm a legitimate merchant, while a false negative leaves harmful activity undetected, so each decision must be precise.

The hardest cases are still left to human experts. Consider a shop registered under “clothing” that in fact sells jade via livestreams, luring buyers to pay by a QR code sent in private chat: suspicious wording, but not conclusive on its own. Its payments look ordinary too, until its behavioral sequence reveals that they cluster at night, span many provinces, and target elderly buyers, with repeated deleted bills. Either clue alone is weak, and a pipeline that inspects text and behavioral statistics separately misses the case; only reading the wording jointly with the behavior reveals the fraud. Such expert review is accurate but slow, and does not scale to tens of millions of merchants.

This need reflects a broader pattern. Entities on modern platforms are described along two axes: their content, which captures what an entity is, and their behavior, long sequences of timestamped actions that reveal how it acts over time [^3] [^9] [^26]. These axes drive applications from recommendation [^2] [^19] [^12] [^11] and e-commerce search [^25] [^1] [^20] to risk control [^10], which increasingly demand reasoning over both at once. Yet the two capabilities have long lived in disjoint model families: large language models (LLMs) reason fluently over text but cannot natively consume behavioral sequences [^3] [^28], while behavior models capture temporal patterns but lack language, reasoning, and explanation [^9] [^10]. Recent work begins to bridge them, injecting discretized behavior tokens into an LLM [^4] [^14] [^23] or contextualizing it with a behavior-derived user embedding [^16]. This progress has so far centered on engagement-oriented recommendation, and its extension to high-stakes decisions, where reliable outcomes require jointly leveraging complementary evidence from text and behavioral sequences, remains largely unexplored.

Equipping an LLM with behavioral-sequence modeling, however, is harder than it looks. The obvious route—serializing a sequence as natural language—creates long inputs and overloads word tokens with behavioral meanings. It also represents temporal patterns in an embedding space designed for lexical semantics, weakening the sequence signal.

Learning behavior with dedicated tokens avoids these pitfalls but raises a second challenge: acquiring the new ability without eroding language competence. OpenOneRec [^22] acquires sequence capability through recommendation-oriented continual pre-training (CPT), then relies on mixed-domain training and general-ability distillation to counteract general-language degradation. Yet this adapt-then-repair pipeline does not eliminate forgetting: despite using roughly $4$ M auxiliary language examples for only $156$ K behavioral sequences and a separate distillation stage, OpenOneRec still loses general language ability. In our setting, sequences number in the tens of millions ($\sim\!10^{7}$), a scale at which direct CPT drives a strong LLM’s general language ability to near-collapse (C-Eval $0.78\!\rightarrow\!0.27$). General-language degradation therefore cannot be reliably offset simply by scaling auxiliary language data.

To meet both challenges, we propose SeqLLM, a framework that can be applied to pretrained LLMs, endowing them with native behavioral-sequence modeling ability while preserving their language ability and supporting downstream tasks that jointly use textual and behavioral information. SeqLLM rests on three components (Figure 1). First, a discrete behavior vocabulary encodes each event as field-level tokens, compressing it from dozens of tokens to about nine on average (roughly $6\times$) while avoiding collisions with the original word vocabulary. Second, inspired by visual–language alignment [^13], a lightweight behavior projector with text-grounded initialization aligns these tokens into the LLM’s semantic space through a two-stage, translation-then-reasoning curriculum, so the model first reads behavior tokens and then reasons over them. Third, and central to our findings, prefix-guided capability injection learns sequence modeling through instruction-conditioned SFT rather than CPT: by recasting next-event prediction as conditional generation under a task prefix, sequence learning is confined to the parameter pathways the prefix activates, avoiding the broad general-language degradation caused by behavior-oriented CPT.

![Refer to caption](https://arxiv.org/html/2608.03063v1/main.png)

Figure 1. Overview of SeqLLM. (a) At inference, projected behavior tokens are interleaved with the task prefix and textual context for joint text–behavior reasoning. (b) Phase 1 grounds text-initialized behavior tokens through token translation and multi-event reasoning. Phase 2 jointly mixes task-prefixed sequence completion with general-instruction SFT, applying loss only to response tokens. This injects sequence modeling without behavior-oriented CPT or a separate capability-recovery stage. The resulting behavior-capable SeqLLM backbone is adapted through risk-control SFT for deployment.

We evaluate SeqLLM in two large-scale business scenarios at WeChat Pay and on three public recommendation benchmarks. The results demonstrate three main findings. First, sequence capability can be injected at scale without eroding language ability: trained on approximately $20$ M unlabeled merchant behavior sequences, prefix-guided SFT matches CPT on next-event prediction (HR@10 $0.80$) while retaining a C-Eval score of $0.78$, versus $0.27$ under CPT. Second, the resulting model delivers measurable production impact at WeChat Pay. SeqLLM is deployed in two complementary roles: as an end-to-end merchant screening model that processes millions of merchants per day, and as a behavior-embedding provider for an existing fraud detector. In the first role, a three-month shadow evaluation confirms $97.5\%$ of flagged merchants as risky versus $92.0\%$ for the production baseline, while the post-launch appeal rate drops from $12\%$ to $\sim 2\%$ with no exonerations. In the second role, an online A/B test shows that initializing the fraud detector’s behavior-token embeddings with those learned by SeqLLM yields gains of 26.8 pp in Precision@Top-0.01% and 33.1 pp in Recall@Top-1%—the largest improvement across historical iterations of this online fraud-detection model. Third, the methodology generalizes beyond payment behavior to public recommendation benchmarks. On MovieLens and Amazon, SeqLLM improves Recall@5 over User-LLM [^16] by up to $32\%$. On RecIF, SeqLLM+RL improves Pass@32 by $14.2\%$ over OpenOneRec’s full OneRec-8B pipeline [^22] while using $4.8\times$ fewer GPU-days and no separate distillation stage. To test behavior-token understanding beyond ranking accuracy, we construct RecProbe, a four-task evaluation suite; SeqLLM outperforms OneRec-8B on all four tasks, raising Video–Topic Matching accuracy from $0.465$ to $0.745$.

Our contributions are fourfold:

1. We formulate *joint text–behavior modeling* for high-stakes decisions: a regime where risk emerges only from the interaction between textual evidence and long event streams, and is therefore systematically missed by text-only or behavior-only pipelines. Controlled modality experiments confirm the value of joint modeling (Table 2).
2. We introduce SeqLLM, a general framework that endows a pretrained LLM with native behavioral-sequence modeling through a compact field-level vocabulary, a text-grounded residual projector with a translation-to-reasoning curriculum, and prefix-guided capability injection.
3. We establish prefix-guided SFT as a scalable alternative to adapt-then-repair CPT. It matches CPT on sequence modeling while preserving language ability, without a separate general-ability distillation stage.
4. We demonstrate impact at both industrial and public scale. SeqLLM is fully deployed at WeChat Pay in two production systems for merchant screening and fraud detection. On public benchmarks, it substantially outperforms User-LLM and OpenOneRec’s full pipeline in both recommendation accuracy and behavior-token semantic understanding, while preserving general language ability and using up to $4.8\times$ fewer GPU-days.

## 2\. Related Work

#### Behavioral sequence modeling.

Behavioral sequence models learn temporal patterns from interaction histories [^26], from recurrent and Transformer architectures such as GRU4Rec [^7], SASRec [^9], and BERT4Rec [^21] to large generative systems like HSTU [^27] and TIGER [^18]. These models are strong at behavioral prediction but do not natively support free-form language reasoning over textual evidence.

#### LLMs for behavioral sequences.

Recent work brings behavior into LLMs [^11]. Text-based systems such as P5 [^5] serialize interactions as language, providing a natural interface at the cost of long contexts and overloaded word tokens. Generative recommenders instead represent items with dedicated codes: OneRec [^4] unifies retrieval and ranking, OpenOneRec [^22] grounds itemic tokens in language, and OneRec-Think [^14] and OneReason [^23] develop explicit reasoning over itemic histories. A complementary approach, User-LLM [^16], conditions an LLM on a compact user vector injected via cross-attention.

#### Positioning SeqLLM

These methods show LLMs can model behavioral histories, but our setting poses a different task: high-stakes risk control must jointly weigh heterogeneous textual evidence, such as merchant profiles and complaints, against field-rich transaction events to produce an auditable decision. OpenOneRec is our closest generative counterpart: it aligns newly added itemic embeddings directly in the expanded embedding table without an explicit modality projector, then acquires sequence modeling through full-parameter co-pretraining, with mixed-domain data and general-ability distillation repairing the resulting general-language degradation. Yet this adapt-then-repair pipeline does not eliminate forgetting: even with roughly $4$ M auxiliary language examples for only $156$ K behavioral sequences plus a separate distillation stage, OpenOneRec still loses general language ability, and the gap widens as the behavioral corpus grows. SeqLLM instead uses a text-grounded projector and prefix-guided SFT to inject sequence capability without a recovery stage, breaking this coupling. User-LLM avoids drift only in its frozen-backbone (Enc) variant; its default Full strategy finetunes the backbone, and either way the entire history is compressed into one vector. We report the matched quantitative comparison in Sections 4.2 and 4.5.

#### Risk detection.

Fraud and risk detection has progressed from rules and statistical scoring to learned models over transaction graphs and behavioral sequences [^17] [^15], e.g., PANTHER [^10] pretrains on transaction histories and transfers to downstream risk tasks. Existing systems nevertheless tend to score textual and behavioral signals in separate pipelines. SeqLLM instead reasons over both within one model, targeting risks that become conclusive only through their interaction.

## 3\. Methodology

SeqLLM injects behavioral-sequence modeling into a pretrained LLM through three components, each addressing one challenge of joint text–behavior modeling: how to represent behavior compactly, how to make the LLM understand it, and how to inject sequence modeling without eroding language ability. First, a discrete behavior vocabulary encodes each event as compact field-level tokens (Section 3.2). Second, a lightweight behavior projector aligns these tokens into the LLM’s semantic space via a translation-then-reasoning curriculum (Section 3.3). Third, prefix-guided capability injection acquires sequence modeling through task-prefixed SFT rather than CPT, preserving the backbone’s language ability (Section 3.4).

### 3.1. Problem Formulation

We consider an entity $e$ —a merchant, user, or item—with textual content $\mathcal{T}_{e}$ describing what it is and a timestamped behavioral sequence $\mathcal{S}_{e}=\langle s_{1},\ldots,s_{L_{e}}\rangle$ describing how it acts. Each event $s_{t}\in\mathcal{E}$ contains fields such as time, amount, channel, and event status (Section 3.2). Given a natural-language instruction $\mathcal{I}$, a single model $\Theta$ jointly uses both signals to model the task-specific output $\mathbf{y}_{e}$ as

$$
p_{\Theta}(\mathbf{y}_{e}\mid\mathcal{I},\mathcal{T}_{e},\mathcal{S}_{e}).
$$

Depending on $\mathcal{I}$, $\mathbf{y}_{e}$ may be a risk decision with a natural-language rationale or the next event $s_{L_{e}+1}$. In high-stakes screening, neither modality may suffice alone. Figure 1(a) illustrates the jade-shop case: a category mismatch and private-QR complaint become conclusive only when read with nighttime, cross-province, high-value payments and a deleted bill. SeqLLM must therefore consume long event streams without compromising the LLM’s language and reasoning ability.

### 3.2. Behavioral Vocabulary

The first design choice is how to represent each event. Instead of writing it out as text, we split the event into its fields—time, amount, event status, and so on—and turn each field value into its own dedicated token. For example, one payment is encoded as follows:

<Time:Thu\_00h> <Scene:App pay> <Channel:Scan QR>  
<Amount:200-500 CNY> … <Status:Success>

Each angle-bracketed unit is one token. Formally, a field $f_{i}$ takes discretized values $V_{i}$ (e.g., log-scale amount buckets or an event-status code); each value $v\in V_{i}$ maps to a token $\langle f_{i}\!:\!v\rangle$, and the behavioral vocabulary is their union $\mathcal{V}_{b}=\bigcup_{i}\{\langle f_{i}\!:\!v\rangle:v\in V_{i}\}$. An event $s_{t}$ is the concatenation of its field tokens and a sequence $\mathcal{S}_{e}$ the concatenation of its events; the event space $\mathcal{E}$ collects all such field-token strings. These tokens occupy a disjoint sub-range of the embedding table, insulating behavior from the word vocabulary while reusing the same backbone.

Field-level factorization offers three benefits. First, it is compact: an additive vocabulary of $\sum_{i}|V_{i}|$ entries represents the $\prod_{i}|V_{i}|$ combinatorial event space using about $9$ field tokens per event on average—each event type encodes its applicable field subset (Appendix B.1)—far fewer than serialized text and without reusing word tokens. Second, it supports compositional generalization: unlike whole-event IDs, as used by PANTHER [^10], events differing in one field share all remaining tokens, so novel combinations remain expressible and decisions can be attributed to individual fields. Third, the tokens are semantically readable: explicit field–value names enable text-based initialization and traceable decisions, whereas RQ-VAE codes carry no intrinsic textual meaning and require a separate quantizer (Section 3.3).

### 3.3. Behavior Projector and Sequence-Language Alignment

The new field tokens enter with no trained embedding. A model can learn to predict them as a bare sequence without any grounding, but understanding and reasoning over them requires aligning them to the LLM’s semantic space. OpenOneRec [^22], for instance, trains newly added itemic embeddings through item–text alignment before full-parameter co-pretraining, without an explicit projection interface.

SeqLLM instead casts grounding as an interface-and-curriculum problem: a lightweight behavior projector supplies each token a dedicated route into the LLM, and a two-stage translation-then-reasoning curriculum first anchors that route in language, then opens it to reasoning (Stages 1–2 below). Alignment thereby advances from understanding individual tokens to reasoning over them.

Behavior projector. Every newly added behavior token reaches the LLM through a shared projector

$$
g_{\psi}(\mathbf{e})=\mathbf{e}+\text{MLP}_{\psi}(\mathbf{e}),
$$

a two-layer MLP (Linear–GELU–Linear) with a residual (skip) connection whose final linear layer is zero-initialized, so $g_{\psi}$ starts as the identity and learns only a small correction to each input embedding $\mathbf{e}$. Applying the same transformation to every behavior token imposes a shared alignment constraint: related field tokens are mapped consistently into the LLM’s semantic space, allowing the backbone to compose them across fields and events. Directly tuning each token embedding provides no such cross-token constraint. This distinction is empirical, not merely architectural. In a controlled industrial ablation, the w/o-projector variant learns to translate individual tokens but fails to transfer their meanings to multi-event reasoning and downstream decisions; the shared projector closes this gap (Appendix E, Table 15). The same pattern appears on RecIF, where SeqLLM outperforms OpenOneRec’s direct-alignment pipeline on all four semantic-understanding and preference-reasoning probes (Table 6).

Because our field tokens are readable, we ground each in the text it denotes rather than initializing at random. Let $Z_{v}$ be the backbone tokenizer’s segmentation of token $v$ ’s readable text (e.g., <Amount:200-500 CNY> $\to$ Amount 200-500 CNY); we mean-pool the corresponding embeddings and rescale the result to the backbone’s embedding statistics,

$$
\bar{\mathbf{e}}_{v}=\frac{1}{|Z_{v}|}\sum_{z\in Z_{v}}\mathbf{E}[z],\qquad\mathbf{e}_{v}^{(0)}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\frac{\bar{\mathbf{e}}_{v}-\hat{\boldsymbol{\mu}}}{\hat{\boldsymbol{\sigma}}},
$$

where $\mathbf{E}$ is the backbone embedding table, $(\boldsymbol{\mu},\boldsymbol{\sigma})$ its per-dimension mean and standard deviation, and $(\hat{\boldsymbol{\mu}},\hat{\boldsymbol{\sigma}})$ those of the pooled vectors $\{\bar{\mathbf{e}}_{v}\}$. This rescaling step is essential: without it, the pooled vectors fall outside the backbone’s embedding distribution and disrupt the model; rescaling them to match its mean and variance places each token in the same region as its own descriptive words.

Recall from Section 3.2 that event $s_{t}$ concatenates field tokens; we write $\Phi(s_{t})$ for its token string ($\Phi(\mathcal{S}_{e})$ for a multi-event window) and train the projector in two stages.

Stage 1: Translation. We first optimize a translation objective that reconstructs the natural-language reading of behavior tokens, training the projector $\psi$ jointly with the LLM. Each instance pairs an instruction $\mathcal{I}$, an input $x=\Phi(s_{t})$ (or a short window), and a field-wise target $y$; an illustrative example from our translation corpus is:

Stage 1 example: translation Instruction: Translate these behavior tokens. Input: <Time:Sun\_14h> <Channel:Msg link> <Status:Success> Output: Sunday at 14:00; successful message-link payment.

The bridge loss is

$$
\mathcal{L}_{\text{bridge}}=-\!\!\!\sum_{(\mathcal{I},x,y)\in\mathcal{D}_{\text{trans}}}\!\!\!\log p_{\Theta,\psi}(y\mid\mathcal{I},x),
$$

where embeddings of tokens in $x$ pass through $g_{\psi}$ before entering the LLM. This objective anchors each field token to its meaning, grounding the vocabulary before the model is asked to reason over it.

Stage 2: Reasoning. We then continue with SFT on multi-event reasoning queries over real merchant sequences. Each instance provides $\mathcal{I}$, a window $x=\Phi(\mathcal{S}_{e})$ of tokenized transactions, and a free-text answer that aggregates or compares fields—customer profiling, anomaly spotting, or risk commentary. A representative example is:

Stage 2 example: reasoning Instruction: What pattern is shared by these transactions? Input: Tx1: <Time:Tue\_15h> <Channel:Scan QR>; Tx2: <Time:Tue\_16h> <Channel:Scan QR> Output: Both are Tuesday-afternoon QR payments.

Together, translation grounds each new token in the LLM’s semantic space, while reasoning alignment enables the model to interpret and use this new vocabulary across contexts, preparing it for joint text–behavior decisions in Section 3.4. We present the two as stages for clarity; since translation is lightweight, their data can also be mixed into a single pass with equivalent effect.

### 3.4. Prefix-Guided Capability Injection

Token grounding teaches the model what each behavior token denotes, but not what sequences of such tokens reveal about an entity. Platform-specific regularities—such as spending rhythms, amount transitions, channel preferences, and deviations from routine—are largely absent from language pretraining. Learning them requires modeling event co-occurrence and temporal evolution. Future-event prediction provides this supervision: generating a continuation requires the model to infer regularities from the observed history. Standard CPT applies next-token loss at every position in the behavior stream. Although this learns sequence structure, it also updates the backbone broadly toward behavior-token prediction, risking interference with pretrained language knowledge.

Our key idea is to retain future-event prediction while turning it from a default modeling objective into an instruction-conditioned capability. The instruction acts as a task condition: behavior continuation is optimized only when this condition is present, rather than being imposed on every input. For each sequence, we choose a cut $k$ at about $70\%$ of its length. The first part, $c=\Phi(s_{1},\ldots,s_{k})$, is provided as input under a natural-language instruction $\mathcal{I}$, while the remaining events, $y=\Phi(s_{k+1},\ldots,s_{L_{e}})$, form the answer. Loss is computed only on this answer. Let $b_{1:N}=\Phi(\mathcal{S}_{e})$ denote the resulting behavior-token sequence and $m$ the token boundary induced by the event cutoff $k$. Figure 1(b) illustrates the construction; at the token level, the two objectives differ in where the loss is applied:

$$
\displaystyle\mathcal{L}_{\mathrm{CPT}}
$$
 
$$
\displaystyle=-\sum_{t=1}^{N}\log p_{\Theta}(b_{t}\mid b_{<t}),
$$
$$
\displaystyle\mathcal{L}_{\mathrm{Prefix}}
$$
 
$$
\displaystyle=-\sum_{t=m+1}^{N}\log p_{\Theta,\psi}(b_{t}\mid\mathcal{I},b_{<t}).
$$

Why this injects sequence ability. The sequence supervision is retained rather than removed. The remaining $\sim 30\%$ of events form the response and receive the same autoregressive next-token loss used by CPT. Predicting each response token from the preceding history directly trains temporal dependencies, both from the observed prefix $s_{1{:}k}$ to the target suffix $s_{k+1{:}L_{e}}$ and within the suffix itself. Prefix masking only removes loss on the observed $70\%$; it leaves the future-event targets—and hence the core sequence-learning signal of CPT— intact.

Why this mitigates forgetting. CPT treats behavior prediction as an unconditional objective and applies loss at almost every position, causing the behavior corpus to update the backbone broadly. Prefix-guided SFT instead conditions behavior prediction on an explicit instruction and applies supervision only to the response suffix. This confines the behavior objective to a specific task context, reducing interference with pretrained language capabilities while preserving the future-prediction signal. Consistent with this interpretation, prefix-guided SFT produces smaller and more uniform per-layer weight changes than CPT (Figure 2). It matches sample-aligned CPT on sequence modeling (HR@10 $0.801$ vs. $0.802$) while preserving substantially more language ability (C-Eval $0.783$ vs. $0.271$; Table 1).

We optimize the backbone and projector with response-only loss:

$$
\mathcal{L}_{\text{inject}}=-\!\!\!\sum_{(\mathcal{I},c,y)\in\mathcal{D}_{\text{inj}}}\!\!\!\log p_{\Theta,\psi}(y\mid\mathcal{I},c),
$$

For sequence examples, all behavior tokens pass through $g_{\psi}$. $\mathcal{D}_{\text{inj}}$ also includes general instruction examples, whose response loss preserves language capabilities during sequence injection, avoiding a separate capability-recovery stage.

## 4\. Experiments & Results

We evaluate SeqLLM on large-scale payment data from WeChat Pay and on public recommendation benchmarks, complemented by evidence from production deployments. Our experiments address four questions:

- RQ1: Can SeqLLM enable a pretrained LLM to model behavioral sequences while retaining its general language capabilities?
- RQ2: Does joint text–behavior modeling improve merchant risk screening over text-only and behavior-only modeling?
- RQ3: Does deploying SeqLLM yield measurable production gains in merchant screening and fraud detection?
- RQ4: Does SeqLLM generalize beyond payment behavior to diverse recommendation domains, and how does it compare with strong public sequence–language baselines in recommendation performance, semantic understanding, and language retention?

We answer RQ1 through a controlled comparison with CPT on WeChat Pay data, RQ2 through a controlled modality comparison, and RQ3 through two production deployments. For RQ4, we compare with User-LLM on MovieLens and Amazon and with OpenOneRec’s OneRec-8B on RecIF, including semantic and preference probes.

### 4.1. Experiment Setup

#### Settings.

We consider three settings: (i) behavioral-sequence modeling on industrial payment data and two downstream deployments at WeChat Pay: merchant screening and fraud detection; (ii) next-item prediction, preference inference, and review generation on MovieLens-20M and Amazon Reviews; and (iii) sequential recommendation and item understanding on the RecIF benchmark, where we compare with the unified sequence–language model OneRec-8B released by OpenOneRec [^22].

#### Backbone & compute.

Unless stated otherwise, SeqLLM uses Qwen3-8B as its pretrained backbone. We report training cost in GPU-days; full hardware and training configurations are provided in Appendix C.1. Code for all public-benchmark experiments, including the RecProbe construction scripts, will be open-sourced.

#### Datasets.

We evaluate on industrial data from two WeChat Pay production scenarios and three public recommendation datasets; full statistics and preprocessing are provided in Appendix B.

- WeChat Pay. Each merchant includes a profile, complaint text, and transaction sequence. We inject sequence capability using $\sim$ 20M unlabeled sequences without risk-positive merchants, then perform risk SFT on 4.06M labeled merchants from a preceding multi-month window. Evaluation covers a subsequent 30-day window ($\sim$ 0.99M merchants/day); features are cut at scoring time. After scoring each merchant, we wait 30 days to collect subsequent evidence of risk before assigning its final label. Histories are capped at $1{,}000$ events and inputs at $10{,}000$ tokens.
- MovieLens-20M & Amazon Reviews. We evaluate next-item prediction, preference classification, and review generation following the User-LLM protocol [^16]; see Appendix B.2.
- RecIF. This OpenOneRec benchmark contains 96M interactions from 160K users and evaluates sequential recommendation and item understanding under its official protocol [^22]; see Appendix B.3.

#### Metrics.

For next-transaction prediction, HR@10 is the fraction of examples whose ground-truth next event appears in the top 10, following PANTHER [^10]. Recall@5 analogously tests whether the held-out next item appears in the top 5. For merchant risk, we report the risky fraction among the highest-scored $r\%$ of merchants ($r\in\{1,0.1,0.01\}$). Following OpenOneRec [^22], RecIF reports whether the target appears among $k$ generated candidates (Pass@ $k$) and the fraction of relevant items retrieved (Recall@ $k$). Other task metrics are Accuracy, NDCG@3, and ROUGE; general language ability is measured by MMLU [^6], C-Eval [^8], and AGIEval [^29]. Higher is better throughout; implementation details are in Appendix D.

### 4.2. Sequence Modeling without Catastrophic Forgetting (RQ1)

This section compares prefix-guided SFT with CPT to show that the former injects behavioral-sequence capability without catastrophic forgetting. Sample-aligned CPT matches our behavioral examples and updates, while gradient-aligned CPT matches our loss-bearing behavior tokens. We test both under no behavior-token alignment and the full alignment stack. All variants share the backbone, corpora, and optimizer; CPT additionally receives both raw-text and chat-format language replay, giving it strictly more language supervision (Appendix C.2). PANTHER [^10] and zero-shot Qwen3-8B provide sequence-only and language-only references.

Table 1. Sequence modeling and language retention for prefix-guided SFT and matched CPT controls under two alignment settings. “No alignment stack” disables semantic initialization, the projector $g_{\psi}$, and the translation $\to$ reasoning curriculum; “full alignment stack” enables all three. Component-level ablations: Table 4, Table 15.

| Method | HR@10 | C-Eval | MMLU | AGIEval |
| --- | --- | --- | --- | --- |
| PANTHER [^10] (sequence-only) | 0.680 | – | – | – |
| Qwen3-8B [^24] (zero-shot) | 0.312 | 0.779 | 0.769 | 0.636 |
| Qwen3-8B + CPT (sample-aligned) | 0.802 | 0.271 | 0.264 | 0.324 |
| Qwen3-8B + CPT (gradient-aligned) | 0.784 | 0.435 | 0.467 | 0.379 |
| Prefix-guided SFT (ours) | 0.801 | 0.783 | 0.745 | 0.640 |
| Qwen3-8B + CPT + align (sample-aligned) | 0.804 | 0.293 | 0.298 | 0.367 |
| Qwen3-8B + CPT + align (gradient-aligned) | 0.793 | 0.456 | 0.489 | 0.377 |
| SeqLLM (ours) | 0.806 | 0.789 | 0.765 | 0.643 |

Table 1 shows that SeqLLM matches CPT on sequence modeling (HR@10 $0.806$ vs. $0.804$) while preserving the original language ability (C-Eval $0.789$ vs. $0.293$). Two further controls rule out alternative explanations: the gradient-aligned CPT row matches the cumulative number of loss-bearing behavior tokens yet still collapses language ability, so the retention advantage is not explained by supervision volume; and the same pattern holds in the no-alignment setting, so it is not contingent on the alignment stack. The high-replay-ratio OpenOneRec comparison—where OneRec-8B also uses continual pre-training—reaches the same conclusion (Section 4.5).

#### Parameter-level evidence: prefix-guided SFT better preserves the backbone.

With the same behavioral and replay corpora, sample-aligned CPT produces much larger per-layer changes, especially in the middle Transformer blocks, whereas prefix-guided SFT produces smaller, more uniform updates (Figure 2). These smaller updates provide parameter-level evidence for why prefix-guided SFT better preserves general language ability.

![Refer to caption](https://arxiv.org/html/2608.03063v1/gradient1.png)

Figure 2. Per-layer relative weight change after sample-aligned CPT and prefix-guided SFT training from the same Qwen3-8B backbone.

### 4.3. Multimodal Merchant Risk Modeling (RQ2)

To answer RQ2, we compare controlled text-only, behavior-only, and joint variants that use the same SeqLLM architecture, training data, and supervision. This isolates whether text and behavioral sequences provide complementary evidence for merchant risk screening.

#### Task and downstream adaptation.

We first train the SeqLLM backbone through sequence–language alignment and prefix-guided sequence-capability injection (Sections 3.3 and 3.4), then fine-tune it on the labeled merchant risk-SFT corpus. Each example asks the model whether risk control is required, with the answer beginning with either the “control” or “no control” label. For ranking, rather than using only the generated answer, we score each merchant by the normalized probability of the positive label token at the first response position, $s=\exp(\ell^{+})/[\exp(\ell^{+})+\exp(\ell^{-})]$, and report precision among the highest-scored merchants. The controlled modality setup is detailed in Appendix C.3.

Table 2. Offline merchant-risk precision under controlled input modalities.

| Input modality | P@Top-1% | P@Top-0.1% | P@Top-0.01% |
| --- | --- | --- | --- |
| Behavior only | 9.0% | 20.1% | 70.0% |
| Text only | 23.0% | 70.1% | 88.0% |
| Text + Behavior | 32.6% | 79.2% | 97.0% |

Table 2 isolates the contribution of joint modeling by holding the model and supervision fixed while varying only the input. Behavior alone misses merchant context, while text alone misses temporal transaction patterns; combining both raises precision substantially at every cutoff, reaching $97.0\%$ at Top- $0.01\%$. This confirms that the two modalities provide complementary evidence for merchant risk screening.

### 4.4. Two Production Deployments at WeChat Pay (RQ3)

To answer RQ3, we evaluate SeqLLM in two complementary production roles. The first uses the joint model directly for merchant risk screening; the second uses SeqLLM-pretrained behavior-token embeddings to initialize the embedding layer for behavior-sequence events in a downstream fraud detector.

#### Deployment I: merchant risk screening.

SeqLLM is deployed as a 0.6B–8B cascade: a 0.6B scanner scores approximately 50 million active merchants daily and retrieves a fixed candidate set, which the 8B model ranks for risk-control action; the full service runs on 128 GPUs. The production baseline is a DeepSeek-based LLM adapted through SFT and RL, using merchant profiles and complaints but no behavioral sequence.

Before launch, we ran a three-month matched prospective shadow evaluation of three systems—the baseline, DeepSeek with text-serialized behavior (same backbone and recipe, differing only in input), and SeqLLM’s 8B ranking stage. Each day, all three ranked the same neutral candidate pool from an upstream pre-screen independent of all evaluated systems, and returned the same number of top candidates; shadow outputs did not affect review, enforcement, or labels. A candidate is labeled positive if confirmed within 30 days either by the existing production risk system—an ensemble of expert strategies operated independently of all three evaluated scorers, including the DeepSeek baseline—or through subsequently confirmed user-reported harm.

Table 3. Merchant screening: three-month shadow evaluation and subsequent production. All metrics are weekly-averaged; appeal and exoneration rates are operational indicators collected after the two systems are officially deployed.

| System (input) | Risk precision (paired shadow; $\uparrow$) | Appeal rate post-launch, among actioned; $\downarrow$ | Exoneration rate post-launch, among appeals; $\downarrow$ |
| --- | --- | --- | --- |
| DeepSeek baseline (text only) | $92.0\%$ | $12\%$ | $8\%$ |
| DeepSeek + serialization | $83.0\%$ | – | – |
| SeqLLM (text + behavior) | $97.5\%$ | $\sim 2\%$ | $0\%$ |
| Improvement | $+5.5$ pp | $10$ pp reduction | $8$ pp reduction |

As shown in Table 3, SeqLLM improves weekly-averaged 30-day risk precision from $92.0\%$ to $97.5\%$, outperforming the baseline in every weekly cohort ($p<10^{-3}$, paired sign test). Text serialization instead reduces precision to $83.0\%$: without sequence-capability injection, serializing up to $1{,}000$ events into tens of thousands of low-density tokens dilutes attention and interferes with the backbone’s text reasoning rather than aiding it. Thus merely appending behavior does not explain SeqLLM’s gain. The serialized variant was not deployed.

After launch, SeqLLM and the baseline were deployed concurrently in production. In the deployed 0.6B–8B cascade, the 0.6B scanner retains $91.24\%$ of confirmed risky merchants at a $0.4\%$ screening ratio over a six-day monitoring window (Table 14), so the 8B ranker’s precision gain is realized on top of near-complete funnel coverage rather than at its expense. Over the same concurrent window, SeqLLM’s appeal rate is $\sim 2\%$ versus the baseline’s $12\%$, and its exoneration rate is $0\%$ versus $8\%$ (zero observed exonerations). Full details are provided in Appendix C.3.

#### Deployment II: behavior-token embeddings for a downstream fraud detector.

The production baseline is a discriminative transaction-pair model whose behavior embeddings are learned end-to-end from fraud labels. We replace their initialization with SeqLLM’s pretrained behavior embeddings, leaving all other features, model components, and serving unchanged. In a concurrent three-month A/B test with users randomly assigned to two equal arms (billions of transactions per day; 14-day label maturation), Precision@Top- $0.01\%/0.1\%$ increases by $26.8/7.6$ pp and Recall@Top- $0.1\%/1\%$ by $12.9/33.1$ pp—the largest improvement achieved across all historical iterations of this production model. Intuitively, SeqLLM embeddings complement sparse fraud labels with text-grounded semantics and sequence patterns, providing stronger representations for long-tail behaviors.

<svg id="S4.F3.pic1" class="ltx_picture ltx_centering" height="166.81" overflow="visible" version="1.1" viewBox="0 0 322.19 166.81" width="322.19"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="translate(0,166.81) matrix(1 0 0 -1 0 0) translate(2.93,0) translate(0,26.33)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g style="--ltx-stroke-color:#D2D2D2;--ltx-fill-color:#D2D2D2;--ltx-fg-color:#D2D2D2;" stroke="#D2D2D2" fill="#D2D2D2" color="#D2D2D2"><path style="fill:none" d="M 0 0 L 250 0"></path></g><g style="--ltx-stroke-color:#B9B9B9;--ltx-fill-color:#B9B9B9;--ltx-fg-color:#B9B9B9;" stroke="#B9B9B9" fill="#B9B9B9" color="#B9B9B9"><path style="stroke:none" d="M 13.78 0 M 13.78 0 L 13.78 33.07 L 29.53 33.07 L 29.53 0 Z M 29.53 33.07"></path></g><g style="--ltx-stroke-color:#4169E1;--ltx-fill-color:#4169E1;--ltx-fg-color:#4169E1;" stroke="#4169E1" fill="#4169E1" color="#4169E1"><path style="stroke:none" d="M 33.46 0 M 33.46 0 L 33.46 106.93 L 49.21 106.93 L 49.21 0 Z M 49.21 106.93"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 20.35 113.43)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:4em;--ltx-fo-height:0.48em;--ltx-fo-depth:0.15em;" width="41.97" height="6.69" transform="matrix(1 0 0 -1 0 5.07)" overflow="visible">+26.8 pp</foreignObject></g> <g style="--ltx-stroke-color:#B9B9B9;--ltx-fill-color:#B9B9B9;--ltx-fg-color:#B9B9B9;" stroke="#B9B9B9" fill="#B9B9B9" color="#B9B9B9"><path style="stroke:none" d="M 74.8 0 M 74.8 0 L 74.8 33.07 L 90.55 33.07 L 90.55 0 Z M 90.55 33.07"></path></g><g style="--ltx-stroke-color:#4169E1;--ltx-fill-color:#4169E1;--ltx-fg-color:#4169E1;" stroke="#4169E1" fill="#4169E1" color="#4169E1"><path style="stroke:none" d="M 94.49 0 M 94.49 0 L 94.49 54.02 L 110.24 54.02 L 110.24 0 Z M 110.24 54.02"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 84 60.52)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:3.5em;--ltx-fo-height:0.48em;--ltx-fo-depth:0.15em;" width="36.72" height="6.69" transform="matrix(1 0 0 -1 0 5.07)" overflow="visible">+7.6 pp</foreignObject></g> <g style="--ltx-stroke-color:#B9B9B9;--ltx-fill-color:#B9B9B9;--ltx-fg-color:#B9B9B9;" stroke="#B9B9B9" fill="#B9B9B9" color="#B9B9B9"><path style="stroke:none" d="M 135.83 0 M 135.83 0 L 135.83 33.07 L 151.58 33.07 L 151.58 0 Z M 151.58 33.07"></path></g><g style="--ltx-stroke-color:#4169E1;--ltx-fill-color:#4169E1;--ltx-fg-color:#4169E1;" stroke="#4169E1" fill="#4169E1" color="#4169E1"><path style="stroke:none" d="M 155.51 0 M 155.51 0 L 155.51 68.62 L 171.26 68.62 L 171.26 0 Z M 171.26 68.62"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 142.4 75.12)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:4em;--ltx-fo-height:0.48em;--ltx-fo-depth:0.15em;" width="41.97" height="6.69" transform="matrix(1 0 0 -1 0 5.07)" overflow="visible">+12.9 pp</foreignObject></g> <g style="--ltx-stroke-color:#B9B9B9;--ltx-fill-color:#B9B9B9;--ltx-fg-color:#B9B9B9;" stroke="#B9B9B9" fill="#B9B9B9" color="#B9B9B9"><path style="stroke:none" d="M 196.85 0 M 196.85 0 L 196.85 33.07 L 212.6 33.07 L 212.6 0 Z M 212.6 33.07"></path></g><g style="--ltx-stroke-color:#4169E1;--ltx-fill-color:#4169E1;--ltx-fg-color:#4169E1;" stroke="#4169E1" fill="#4169E1" color="#4169E1"><path style="stroke:none" d="M 216.54 0 M 216.54 0 L 216.54 124.29 L 232.28 124.29 L 232.28 0 Z M 232.28 124.29"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 203.42 130.79)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:4em;--ltx-fo-height:0.48em;--ltx-fo-depth:0.15em;" width="41.97" height="6.69" transform="matrix(1 0 0 -1 0 5.07)" overflow="visible">+33.1 pp</foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 1.68 -19.78)" fill="#000000" stroke="#000000"><g class="ltx_tikzmatrix" transform="matrix(1 0 0 -1 0 13.88)"><g class="ltx_tikzmatrix_row" transform="matrix(1 0 0 1 0 6.62)"><g class="ltx_tikzmatrix_col ltx_nopad_l ltx_nopad_r" transform="matrix(1 0 0 -1 7.87 0)"><text transform="matrix(1 0 0 -1 0 0)">Precision</text></g></g> <g class="ltx_tikzmatrix_row" transform="matrix(1 0 0 1 0 13.88)"><g class="ltx_tikzmatrix_col ltx_nopad_l ltx_nopad_r" transform="matrix(1 0 0 -1 0 0)"><text transform="matrix(1 0 0 -1 0 0)">@Top-0.01%</text></g></g></g></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 65.47 -19.78)" fill="#000000" stroke="#000000"><g class="ltx_tikzmatrix" transform="matrix(1 0 0 -1 0 13.88)"><g class="ltx_tikzmatrix_row" transform="matrix(1 0 0 1 0 6.62)"><g class="ltx_tikzmatrix_col ltx_nopad_l ltx_nopad_r" transform="matrix(1 0 0 -1 5.12 0)"><text transform="matrix(1 0 0 -1 0 0)">Precision</text></g></g> <g class="ltx_tikzmatrix_row" transform="matrix(1 0 0 1 0 13.88)"><g class="ltx_tikzmatrix_col ltx_nopad_l ltx_nopad_r" transform="matrix(1 0 0 -1 0 0)"><text transform="matrix(1 0 0 -1 0 0)">@Top-0.1%</text></g></g></g></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 126.49 -19.83)" fill="#000000" stroke="#000000"><g class="ltx_tikzmatrix" transform="matrix(1 0 0 -1 0 13.99)"><g class="ltx_tikzmatrix_row" transform="matrix(1 0 0 1 0 6.73)"><g class="ltx_tikzmatrix_col ltx_nopad_l ltx_nopad_r" transform="matrix(1 0 0 -1 12.23 0)"><text transform="matrix(1 0 0 -1 0 0)">Recall</text></g></g> <g class="ltx_tikzmatrix_row" transform="matrix(1 0 0 1 0 13.99)"><g class="ltx_tikzmatrix_col ltx_nopad_l ltx_nopad_r" transform="matrix(1 0 0 -1 0 0)"><text transform="matrix(1 0 0 -1 0 0)">@Top-0.1%</text></g></g></g></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 191.84 -19.83)" fill="#000000" stroke="#000000"><g class="ltx_tikzmatrix" transform="matrix(1 0 0 -1 0 13.99)"><g class="ltx_tikzmatrix_row" transform="matrix(1 0 0 1 0 6.73)"><g class="ltx_tikzmatrix_col ltx_nopad_l ltx_nopad_r" transform="matrix(1 0 0 -1 7.9 0)"><text transform="matrix(1 0 0 -1 0 0)">Recall</text></g></g> <g class="ltx_tikzmatrix_row" transform="matrix(1 0 0 1 0 13.99)"><g class="ltx_tikzmatrix_col ltx_nopad_l ltx_nopad_r" transform="matrix(1 0 0 -1 0 0)"><text transform="matrix(1 0 0 -1 0 0)">@Top-1%</text></g></g></g></g> <g style="--ltx-stroke-color:#B9B9B9;--ltx-fill-color:#B9B9B9;--ltx-fg-color:#B9B9B9;" stroke="#B9B9B9" fill="#B9B9B9" color="#B9B9B9"><path style="stroke:none" d="M 242.13 85.43 M 242.13 85.43 L 242.13 99.21 L 251.97 99.21 L 251.97 85.43 Z M 251.97 99.21"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 260.01 88.19)" fill="#000000" stroke="#000000"><g class="ltx_tikzmatrix" transform="matrix(1 0 0 -1 0 9.99)"><g class="ltx_tikzmatrix_row" transform="matrix(1 0 0 1 0 4.8)"><g class="ltx_tikzmatrix_col ltx_nopad_l ltx_nopad_r" transform="matrix(1 0 0 -1 0 0)"><text transform="matrix(1 0 0 -1 0 0)">Baseline</text></g></g> <g class="ltx_tikzmatrix_row" transform="matrix(1 0 0 1 0 9.99)"><g class="ltx_tikzmatrix_col ltx_nopad_l ltx_nopad_r" transform="matrix(1 0 0 -1 0 0)"><text transform="matrix(1 0 0 -1 0 0)">(normalized)</text></g></g></g></g> <g style="--ltx-stroke-color:#4169E1;--ltx-fill-color:#4169E1;--ltx-fg-color:#4169E1;" stroke="#4169E1" fill="#4169E1" color="#4169E1"><path style="stroke:none" d="M 242.13 60.63 M 242.13 60.63 L 242.13 74.41 L 251.97 74.41 L 251.97 60.63 Z M 251.97 74.41"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" transform="matrix(1.0 0.0 0.0 1.0 260.01 65.84)" fill="#000000" stroke="#000000"><foreignObject style="--ltx-fo-width:5.8em;--ltx-fo-height:0.5em;--ltx-fo-depth:0.14em;" width="54.65" height="6.05" transform="matrix(1 0 0 -1 0 4.7)" overflow="visible">SeqLLM init.</foreignObject></g></g></svg>

Figure 3. Online A/B gains from SeqLLM-initialized behavior embeddings. Blue bars show absolute percentage-point gains over the baseline; absolute values are withheld for business confidentiality.

### 4.5. Generalization and Comparison with Strong Public Baselines (RQ4)

We assess generalization along three dimensions: MovieLens/Amazon test transfer from payment behavior to standard recommendation; RecIF provides a stage-matched comparison with OpenOneRec; and our constructed RecProbe probes semantic understanding and preference reasoning beyond ranking accuracy.

Table 4. Public recommendation and language results. Recommendation baselines from User-LLM [^16]; language scores reproduced by us.

<table><tbody><tr><td rowspan="2">Method</td><td colspan="2">MovieLens-20M</td><td colspan="3">Amazon Review</td><td colspan="3">Language Ability</td></tr><tr><td>Next-item (R@5)</td><td>Fav. genre (Acc.)</td><td>Next-item (R@5)</td><td>Fav. category (Acc.)</td><td>Review (ROUGE)</td><td>MMLU</td><td>C-Eval</td><td>AGIEval</td></tr><tr><td>Vanilla-Sequence</td><td>0.152</td><td>0.372</td><td>0.050</td><td>0.437</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>Textualized</td><td>0.140</td><td>0.787</td><td>0.042</td><td>0.885</td><td>22.82</td><td>–</td><td>–</td><td>–</td></tr><tr><td>User-LLM (baseline)</td><td>0.154</td><td>0.787</td><td>0.047</td><td>0.890</td><td>26.38</td><td>0.297</td><td>0.276</td><td>0.307</td></tr><tr><td>SeqLLM</td><td>0.174 (+13.0%)</td><td>0.973 (+23.6%)</td><td>0.062 (+31.9%)</td><td>0.987 (+10.9%)</td><td>28.62 (+8.5%)</td><td>0.738</td><td>0.786</td><td>0.657</td></tr><tr><td>SeqLLM w/o SemInit</td><td>0.169</td><td>0.955</td><td>0.052</td><td>0.956</td><td>27.99</td><td>0.737</td><td>0.786</td><td>0.656</td></tr><tr><td>SeqLLM w/o projector</td><td>0.172</td><td>0.934</td><td>0.061</td><td>0.925</td><td>27.23</td><td>0.735</td><td>0.784</td><td>0.655</td></tr></tbody></table>

#### Cross-domain transfer on MovieLens and Amazon.

User-LLM [^16] is a representative purpose-built baseline that conditions the LLM on a separate user encoder via cross-attention with the backbone finetuned jointly. Table 4 shows SeqLLM outperforms User-LLM and its Vanilla-Sequence/Textualized controls on all five recommendation and preference tasks—Recall@5 gains of $13.0\%$ on MovieLens and $31.9\%$ on Amazon—while retaining substantially higher MMLU/C-Eval/AGIEval. The two ablations show complementary benefits: semantic initialization helps next-item prediction, while the projector helps semantic preference and review generation.

#### Stage-matched comparison with OpenOneRec on RecIF

Open OneRec [^22] grounds itemic tokens and acquires sequence modeling through continual pretraining. We match its 156K behavioral sequences and 13M-caption alignment pool. Despite using only 100K general-instruction examples versus OneRec’s $\sim$ 28.6M general-domain and general-SFT examples, SeqLLM outperforms OneRec-8B-Pretrain on all six recommendation and language metrics at the sequence-acquisition stage using $4.4\times$ fewer GPU-days (Table 5).[^23] For final adaptation, OneRec adds SFT, general-ability distillation, and RL, whereas SeqLLM adds only RL; SeqLLM+RL still matches the full OneRec-8B pipeline on P@1 and language ability, improves P@32 by $14.2\%$, and uses $4.8\times$ fewer GPU-days. See Appendix B.3 and C.3 for data construction, training budgets, and GPU-day estimation.

Table 5. RecIF results: sequential recommendation, language ability, and efficiency. OneRec-PT/OneRec-full = Pretrain $\to$ (SFT $\to$ Distill $\to$)RL; SeqLLM/SeqLLM+RL = Align $\to$ SFT($\to$ RL). CE = C-Eval; AGI = AGIEval; GPU = GPU-days. Gray subscripts: relative change vs. the OneRec baseline in the same block (compute-reduction factor for GPU-days).

| Method | P@1 | P@32 | R@32 | MMLU | CE | AGI | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OneRec-PT | 0.0204 | 0.1521 | 0.0235 | 0.7112 | 0.7370 | 0.6080 | 460 |
| SeqLLM | 0.0411 (+101.5%) | 0.2398 (+57.7%) | 0.0354 (+50.6%) | 0.7221 (+1.5%) | 0.7667 (+4.0%) | 0.6646 (+9.3%) | 105 ($4.4\times$ fewer) |
| OneRec-full | 0.0542 | 0.2101 | 0.0356 | 0.7176 | 0.7489 | 0.6411 | $\sim$ 994 |
| SeqLLM+RL | 0.0540 ($-$ 0.4%) | 0.2399 (+14.2%) | 0.0360 (+1.1%) | 0.7201 (+0.3%) | 0.7485 ($-$ 0.1%) | 0.6465 (+0.8%) | 206 ($4.8\times$ fewer) |

#### Behavior-token understanding and preference reasoning.

We construct four RecProbe tasks covering video semantics and user preferences (Table 6). SeqLLM substantially outperforms OneRec-8B on all four—e.g., Video–Topic Matching accuracy rises from $0.465$ to $0.745$ and Video Interest Ranking NDCG@3 from $0.112$ to $0.896$. The three option-based probes are format-robust, so OneRec-8B’s lower accuracies reflect limited semantic grounding rather than formatting failures; the ranking probe additionally exposes an instruction-following failure of OneRec-8B (Appendix B.3). These results indicate that SeqLLM not only grounds new behavior tokens but also flexibly applies them across tasks—a capability we attribute to the behavior projector, which provides a shared semantic interface between the new vocabulary and the LLM. Our controlled projector ablation on industrial data supports this view: removing the projector preserves single-token translation but degrades multi-event reasoning and downstream transfer (Table 15), consistent with its role as a compositional interface rather than per-token storage.

Table 6. Semantic understanding and preference reasoning on RecIF. Gray text: relative improvement over OneRec-8B.

<table><tbody><tr><td>Task</td><td>Metric</td><td>OneRec-8B</td><td>Ours</td></tr><tr><td colspan="4"><em>Video semantics</em></td></tr><tr><td>Video–Topic Matching</td><td>Acc</td><td>0.4653</td><td>0.7450 (+60.1%)</td></tr><tr><td>Audience Targeting</td><td>Acc</td><td>0.4733</td><td>0.6160 (+30.2%)</td></tr><tr><td colspan="4"><em>User preference</em></td></tr><tr><td>Video Interest Ranking</td><td>NDCG@3</td><td>0.1121</td><td>0.8961 (<math><semantics><mrow><mn>8.0</mn> <mo>×</mo></mrow> <annotation>8.0\times</annotation></semantics></math>)</td></tr><tr><td>Interest–Category Consistency</td><td>Acc</td><td>0.4973</td><td>0.7420 (+49.2%)</td></tr></tbody></table>

Together, SeqLLM outperforms encoder-based User-LLM and delivers stronger overall recommendation and token-understanding performance than OpenOneRec at comparable language ability and substantially lower cost, demonstrating generalization across datasets, tasks, and sequence–language paradigms (RQ4).

### 4.6. Ablation Summary

Our ablations isolate each design choice: the training objective (prefix-guided SFT vs. CPT, Table 1), the alignment stack as a whole (on/off, Table 1) and its components individually—semantic initialization and the projector (Table 4, Table 15)—the input modality (text/behavior/joint, Table 2), and behavior serialization (none/text/native tokens, Table 3).

## 5\. Conclusion

We presented SeqLLM, a framework that gives pretrained LLMs native behavioral-sequence modeling while preserving language ability. Its three components—a compact field-level vocabulary, a text-grounded projector, and prefix-guided capability injection—match CPT on sequence modeling while avoiding catastrophic forgetting (C-Eval $0.78$ vs. $0.27$ under CPT) and the need for a separate recovery stage. Deployed at WeChat Pay, SeqLLM raises risk precision from $92.0\%$ to $97.5\%$ and yields the largest recall gain across historical iterations of the online fraud-detection model. On public benchmarks it outperforms User-LLM and OpenOneRec’s full pipeline at substantially lower training cost, establishing capability injection as a scalable route to unified language and behavioral-sequence models.

## References

## Appendix A Reproducibility Overview

This appendix provides everything needed to reproduce every result in the main text, organized by the reproduction workflow: data acquisition and preprocessing (Appendix B), model configuration and training with all baselines (Appendix C), and per-number computation from a trained checkpoint (Appendix D). To avoid duplication, we do not restate the method (Section 3) or the metric definitions (Section 4.1), reporting only the configurations, statistics, and protocols required for reproduction.

Table 7. Mapping from main results to the relevant parts of this appendix. Training configurations (Appendix C.1) and evaluation protocols (Appendix D) are shared across all rows.

| Result | Dataset | Data / baseline setup |
| --- | --- | --- |
| Tab. 1 | WeChat Pay | §B.1, §C.2, §C.3 |
| Tab. 2 | WeChat Pay | §B.1, §C.3 |
| Tab. 3 | WeChat Pay | §C.3 |
| Tab. 13 | WeChat Pay | §C.3 |
| Tab. 14 | WeChat Pay | §C.3 |
| Tab. 4 | MovieLens/Amazon | §B.2, §C.3 |
| Tab. 5 | RecIF | §B.3, §C.3 |
| Tab. 6 | RecIF (RecProbe) | §B.3 |
| Fig. 2 | WeChat Pay | §C.2 |

## Appendix B Data and Preprocessing

### B.1. WeChat Pay Dataset

#### Additional sample-construction details.

All records are anonymized and stripped of personally identifiable information. For offline evaluation, all merchants appearing in the risk-SFT corpus are excluded from the held-out test cohorts, making the split entity-disjoint for both positive and negative labels. Transactions are sorted chronologically before sequence truncation and tokenization. Table 8 consolidates the corpus and stage-level statistics reported in the main text.

#### Field-level vocabulary.

To respect confidentiality, individual field names are not disclosed. The full schema spans $28$ fields (Table 8), but not every field applies to every event: each event type is encoded with its own applicable field subset, so an event carries about $9$ field tokens on average. The behavioral vocabulary ($1{,}533$ entries) is the union of all field–value tokens across the $28$ fields.

Table 8. WeChat Pay corpus statistics. Mean sequence length is computed after the $1{,}000$ -event cap.

<table><tbody><tr><td></td><td>Train</td><td>Validation</td><td>Test</td></tr><tr><td>Time span</td><td>Preceding multi-month window</td><td>–</td><td>Subsequent 30-day window</td></tr><tr><td>Risk-SFT entities (merchants)</td><td><math><semantics><mn>4.06</mn> <annotation>4.06</annotation></semantics></math> M</td><td>–</td><td><math><semantics><mn>0.99</mn> <annotation>0.99</annotation></semantics></math> M/day <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 30</td></tr><tr><td>Unlabeled source merchants</td><td colspan="3"><math><semantics><mn>20</mn> <annotation>20</annotation></semantics></math> M active merchants</td></tr><tr><td>Mean / max behavior events</td><td>862 / <math><semantics><mrow><mn>1</mn><mo>,</mo><mn>000</mn></mrow> <annotation>1{,}000</annotation></semantics></math></td><td>–</td><td>845 / <math><semantics><mrow><mn>1</mn><mo>,</mo><mn>000</mn></mrow> <annotation>1{,}000</annotation></semantics></math></td></tr><tr><td>Maximum input tokens</td><td colspan="3"><math><semantics><mrow><mn>10</mn><mo>,</mo><mn>000</mn></mrow> <annotation>10{,}000</annotation></semantics></math></td></tr><tr><td>Behavior fields / vocab. size</td><td colspan="3"><math><semantics><mn>28</mn> <annotation>28</annotation></semantics></math> / <math><semantics><mrow><mn>1</mn><mo>,</mo><mn>533</mn></mrow> <annotation>1{,}533</annotation></semantics></math></td></tr><tr><td colspan="4"><em>Stage sample counts</em></td></tr><tr><td>Alignment</td><td colspan="3"><math><semantics><mn>117</mn> <annotation>117</annotation></semantics></math> k (<math><semantics><mn>50</mn> <annotation>50</annotation></semantics></math> k + <math><semantics><mn>19</mn> <annotation>19</annotation></semantics></math> k <math><semantics><mrow><mo>×</mo> <mn>3</mn></mrow> <annotation>\times 3</annotation></semantics></math> + <math><semantics><mn>10</mn> <annotation>10</annotation></semantics></math> k)</td></tr><tr><td>Capability injection</td><td colspan="3"><math><semantics><mrow><mo>∼</mo> <mn>20</mn></mrow> <annotation>\sim 20</annotation></semantics></math> M seq. + <math><semantics><mrow><mo>∼</mo> <mn>1.1</mn></mrow> <annotation>\sim 1.1</annotation></semantics></math> M text</td></tr><tr><td>Risk SFT</td><td><math><semantics><mn>4.06</mn> <annotation>4.06</annotation></semantics></math> M</td><td>–</td><td><math><semantics><mn>0.99</mn> <annotation>0.99</annotation></semantics></math> M/day <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 30</td></tr></tbody></table>

### B.2. MovieLens-20M and Amazon Reviews

#### Data sources and filtering.

MovieLens-20M is the GroupLens 20M-rating release; for Amazon Reviews we use the *Movies and TV* 5-core subset. Following User-LLM, we retain the users and items that meet its interaction-count thresholds, sort each user’s interactions by timestamp, and represent an interaction by its metadata: movie name, genre, and rating for MovieLens, and product title, category, rating, and review summary for Amazon. Table 9 reports the resulting user, item, and interaction counts.

Table 9. Statistics after the User-LLM preprocessing protocol. The number of test examples equals the number of users.

| Dataset | Users | Items | Train | Test |
| --- | --- | --- | --- | --- |
| MovieLens-20M | 82,977 | 27,280 | 13,821,405 | 82,977 |
| Amazon Reviews | 5,756 | 177,978 | 357,258 | 5,756 |

#### Preprocessing

A sliding window (length $50$) over each user’s chronological history yields supervised examples, where the events preceding a position form the history and the event at that position is the label. The most recent interaction of every user is held out for testing and never seen during training, so the test set has exactly one example per user ($82{,}977$ for MovieLens, $5{,}756$ for Amazon), and the *Train* column of Table 9 counts the windowed next-item pairs. Each distinct movie or product is assigned a dedicated item token, whose readable description (name and genre for MovieLens, title and category for Amazon) is used for sequence–language alignment (Section 3.3). At downstream time, a user history is the chronological sequence of these item tokens, and the three tasks differ only in the task prefix and target.

#### Next-item prediction.

In the prefix-guided capability injection stage, we split each sliding window of length 50 at a ratio of 35:15. Specifically, the first 35 items are taken as the conditional input to generate the subsequent 15 items for multi-item sequence continuation, instead of merely scoring individual candidate items. This operation is applied to both the training and evaluation sets. The model takes the item-token interaction history as input and is optimized to predict the next item token.

#### Favorite genre / category.

The label is the genre (MovieLens) or product category (Amazon) that occurs most often across the user’s history, with ties broken by the most recent occurrence. The input is the same item-token history and the target is the dominant genre/category, so the task probes long-range preference rather than the last interaction. Each user yields one instance per split, scored by accuracy.

Favorite-genre example (MovieLens) Instruction: Based on the user’s history, what is their favorite genre? Input: <movie\_924> <genre\_103> <ml\_rating\_6> <movie\_919> <genre\_210> <ml\_rating\_6> $\cdots$ <movie\_3030> <genre\_131> <ml\_rating\_5> Output: <genre\_171>   (Horror $|$ Mystery $|$ Thriller)

#### Review generation.

Defined for Amazon only, as MovieLens has no review text: conditioned on the item-token history and the target product (given by name and rating in the instruction), the model generates the user’s review, and the held-out review of the test item is the ROUGE reference; empty reviews are discarded.

Review-generation example (Amazon) Instruction: Please write a review for *The Palace of Versailles*. Your rating is 3. Input: <item\_461> <category\_1> <az\_rating\_4> <item\_2649> <category\_25> <az\_rating\_4> $\cdots$ <item\_51101> <category\_9> <az\_rating\_3> Output: Well done! Biltz was the original person to put …

### B.3. RecIF

#### Data preparation.

We obtain the data from two official OpenOneRec Hugging Face repositories: the RecIF dataset <sup>2</sup> and the general-purpose SFT dataset.<sup>3</sup> Following the official preprocessing scripts,<sup>4</sup> we build our splits while faithfully retaining the original semantic IDs (SIDs), their associated textual descriptions, and the standard train/test partition. *Alignment data* pair each SID with its textual caption to teach the projector to map video tokens into language, contain 13 $M$ + item-understanding data. *Capability-injection data* come from the Video-Rec behavioral sequences (96M interactions from 160K users): to increase sequence supervision, every training example with a multi-item output is expanded position-wise—each output position is used in turn as the target of a separate instance while retaining the associated user-history context (RL variant is optimized on the original, unexpanded examples). *Evaluation data* follow the official held-out test split. We additionally include 100000 general-language examples to preserve language ability and 5 $M$ + item-understanding examples derived from RecIF captions and SIDs.

#### RecProbe: item-understanding probing tasks.

RecProbe is constructed in this work from the public RecIF item captions and semantic IDs. It contains four probing tasks that evaluate video-token understanding beyond next-item prediction. Each task shares the system prompt below and is scored as reported in Table 6: Video–Topic Matching, Audience Targeting, and Interest–Category Consistency use accuracy, and Video Interest Ranking uses NDCG@3. The prompts are shown in English for presentation; semantic ID token strings (<|sid\_begin|> $\ldots$ <|sid\_end|>) are verbatim.

We observe that OneRec-8B follows instructions poorly on this task *only*: rather than producing the requested ranking, it merely re-emits the candidate SIDs and loops on them, which directly accounts for its low NDCG@3 here. Its score on this probe therefore reflects grounding and instruction following jointly, and we scope our claims accordingly (Section 4.5). On the three option-based probes (Video–Topic Matching, Audience Targeting, and Interest–Category Consistency), OneRec-8B consistently emits a valid option under the same answer-extraction protocol applied to both models, so its lower accuracies on those tasks measure semantic understanding rather than output formatting.

RecProbe tasks (shared system prompt and one example each) System: You are a video semantic-understanding expert who can infer video content from video tokens. Video–Topic Matching (acc.): Given videos A--D (<|sid\_begin|> $\ldots$ <|sid\_end|>), which one is most relevant to the topic ‘‘pets / cute animals’’? Answer with the option only. Audience Targeting (acc.): Given an ad video, which audience is it best suited for? *(A)* humorous chat screenshots / intimate-relationship interactions; *(B)* baking and dessert making; *(C)* cartoon characters and warm emotion; *(D)* game achievements and phone-screen demos. Answer with the option only. Video Interest Ranking (NDCG@3): Given the primary interest ‘‘gaming / esports’’ and recent views, rank candidate videos A--C by recommendation priority (high to low). Interest–Category Consistency (acc.): Given a behavior sequence and a candidate video, does the candidate match the sequence’s dominant interest? *(A)* Matches    *(B)* Does not match. Answer with the option only.

Video Interest Ranking: OneRec-8B vs. SeqLLM (same input as above) OneRec-8B: <|sid\_begin|><s\_a\_5719><s\_b\_513><s\_c\_7881><|sid\_end|> <|sid\_begin|><s\_a\_5719><s\_b\_2395><s\_c\_3488><|sid\_end|> <|sid\_begin|><s\_a\_2776><s\_b\_554><s\_c\_8066><|sid\_end|> $\cdots$ (re-emits candidate SIDs and keeps looping instead of ranking) SeqLLM: ‘‘Answer: Video A > Video B > Video C’’

## Appendix C Model, Training, and Baselines

The SeqLLM components are defined in Section 3; this section adds only reproduction-level configuration and the exact setup of every baseline and ablation. All settings share the stage order semantic initialization $\rightarrow$ translation alignment $\rightarrow$ reasoning alignment $\rightarrow$ prefix-guided injection $\rightarrow$ downstream tuning, with the projector and LLM jointly updated unless a baseline requires otherwise.

### C.1. Training Configurations

Table 12 reports the stage-wise configurations for all three settings. Behavior-token initialization follows Section 3.3; tokens whose readable text cannot be segmented fall back to random initialization.

### C.2. Controlled CPT vs. Prefix-Guided SFT Protocol

The RQ1 comparison (Table 1, Fig. 2) controls every factor except the training objective. Both runs start from the identical Qwen3-8B checkpoint, share the same corpora, token budget, and optimizer schedule, and deliberately omit the projector and alignment curriculum. The sole difference is the objective: CPT uses full-token next-token prediction on the packed behavior stream, while prefix-guided SFT computes loss only on the assistant span under a task prefix. Both CPT controls receive language replay in both raw-text and chat-format renderings (strictly more language supervision than prefix-guided SFT). Training uses 50k steps, 96 GPUs, DeepSpeed ZeRO-3, and BF16 (Table 10).

Table 10. Training configuration for CPT vs. prefix-guided SFT. CPT uses same configuration with stage=pt and full-token loss.

| Setting | Value |
| --- | --- |
| Backbone | Qwen3-8B (full FT) |
| General corpora | Alpaca-GPT4-ZH + Firefly + identity |
| Packing / cutoff | true / $10{,}000$ |
| Batch size | 2\*2\*8\*12 |
| Steps | $50k$ |
| Peak LR | $2.0\times 10^{-5}$ |
| LR schedule | cosine, warmup $=500$ steps |
| Precision / parallelism | BF16 / DeepSpeed ZeRO-3 |
| CPT objective | Next-token LM loss on packed stream |
| Prefix-guided SFT | Assistant-token loss only |

Table 11. Mechanistic diagnostics of the two objectives, each measured against the shared Qwen3-8B base (WeChat Pay). Lower weight change / KL and higher similarity to base indicate less disruption of the pretrained model.

<table><tbody><tr><td>Diagnostic (vs. base)</td><td>CPT</td><td>Prefix-guided SFT</td></tr><tr><td>Backbone mean <math><semantics><msub><mrow><mo>∥</mo> <mrow><mi>Δ</mi> <mo></mo><mi>W</mi></mrow> <mo>∥</mo></mrow> <mi>F</mi></msub> <annotation>\lVert\Delta W\rVert_{F}</annotation></semantics></math> (rel.)</td><td><math><semantics><mn>0.042</mn> <annotation>0.042</annotation></semantics></math></td><td><math><semantics><mn>0.022</mn> <annotation>\mathbf{0.022}</annotation></semantics></math></td></tr><tr><td>Stable rank of <math><semantics><mrow><mi>Δ</mi> <mo></mo><mi>W</mi></mrow> <annotation>\Delta W</annotation></semantics></math></td><td><math><semantics><mn>46.0</mn> <annotation>46.0</annotation></semantics></math></td><td><math><semantics><mn>26.6</mn> <annotation>\mathbf{26.6}</annotation></semantics></math></td></tr><tr><td>Next-token KL from base</td><td><math><semantics><mn>0.84</mn> <annotation>0.84</annotation></semantics></math></td><td><math><semantics><mn>0.21</mn> <annotation>\mathbf{0.21}</annotation></semantics></math></td></tr><tr><td>Last-layer CKA to base</td><td><math><semantics><mn>0.91</mn> <annotation>0.91</annotation></semantics></math></td><td><math><semantics><mn>0.98</mn> <annotation>\mathbf{0.98}</annotation></semantics></math></td></tr><tr><td>Task-vector cosine <math><semantics><mrow><mi>cos</mi> <mo>⁡</mo> <mrow><mo>(</mo><mtext>CPT</mtext><mo>,</mo><mtext>Prefix-guided SFT</mtext><mo>)</mo></mrow></mrow> <annotation>\cos(\text{CPT},\text{Prefix-guided SFT})</annotation></semantics></math></td><td colspan="2"><math><semantics><mn>0.14</mn> <annotation>0.14</annotation></semantics></math></td></tr><tr><td>Next-transaction HR@10</td><td><math><semantics><mn>0.802</mn> <annotation>0.802</annotation></semantics></math></td><td><math><semantics><mn>0.801</mn> <annotation>0.801</annotation></semantics></math></td></tr></tbody></table>

Table 12. Stage-wise training configurations across WeChat Pay, MovieLens-20M/Amazon Reviews, and RecIF. T-Align = Translation Alignment; R-Align = Reasoning Alignment; PGCI = P-G Capability Injection; RC-SFT = Risk-Control SFT; TS-SFT = Task-Specific SFT; TIA = Text–Item Alignment; RL = Reinforcement Learning. Global batch sizes are written as per-device batch $\times$ gradient-accumulation steps $\times$ GPUs per node $\times$ nodes; the RL row reports per-device batch $\times$ nodes.

<table><tbody><tr><td>Setting</td><td>Stage</td><td>Trainable modules</td><td>LR</td><td>Global batch</td><td>Max length</td><td>Steps/Epochs</td><td>Weight decay</td></tr><tr><td rowspan="4">WeChat Pay</td><td>T-Align</td><td rowspan="4">projector + LLM backbone</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>5</mn></mrow></msup></mrow> <annotation>1\times 10^{-5}</annotation></semantics></math></td><td>4*8*8*12</td><td>2048</td><td>2 epochs</td><td>0.01</td></tr><tr><td>R-Align</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>5</mn></mrow></msup></mrow> <annotation>1\times 10^{-5}</annotation></semantics></math></td><td>4*8*8*12</td><td>2048</td><td>2 epochs</td><td>0.01</td></tr><tr><td>PGCI</td><td><math><semantics><mrow><mn>2</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>5</mn></mrow></msup></mrow> <annotation>2\times 10^{-5}</annotation></semantics></math></td><td>2*2*8*12</td><td>10000</td><td>50k steps</td><td>0.01</td></tr><tr><td>RC-SFT</td><td><math><semantics><mrow><mn>5</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>6</mn></mrow></msup></mrow> <annotation>5\times 10^{-6}</annotation></semantics></math></td><td>2*2*8*12</td><td>10000</td><td>2 epochs</td><td>0.01</td></tr><tr><td rowspan="4">MovieLens-20M & Amazon Reviews</td><td>T-Align</td><td rowspan="4">projector + LLM backbone</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>5</mn></mrow></msup></mrow> <annotation>1\times 10^{-5}</annotation></semantics></math></td><td>8*16*8*12</td><td>1024</td><td>2 epochs</td><td>0.01</td></tr><tr><td>R-Align</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>5</mn></mrow></msup></mrow> <annotation>1\times 10^{-5}</annotation></semantics></math></td><td>8*16*8*12</td><td>1024</td><td>2 epochs</td><td>0.01</td></tr><tr><td>PGCI</td><td><math><semantics><mrow><mn>2</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>5</mn></mrow></msup></mrow> <annotation>2\times 10^{-5}</annotation></semantics></math></td><td>8*16*8*12</td><td>1024</td><td>5k steps</td><td>0.01</td></tr><tr><td>TS-SFT</td><td><math><semantics><mrow><mn>5</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>6</mn></mrow></msup></mrow> <annotation>5\times 10^{-6}</annotation></semantics></math></td><td>8*16*8*12</td><td>1024</td><td>2k steps</td><td>0.01</td></tr><tr><td rowspan="3">RecIF (OpenOneRec)</td><td>TIA</td><td rowspan="3">projector + LLM backbone</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>5</mn></mrow></msup></mrow> <annotation>1\times 10^{-5}</annotation></semantics></math></td><td>8*16*8*12</td><td>1024</td><td>1 epoch</td><td>0.01</td></tr><tr><td>PGCI</td><td><math><semantics><mrow><mn>2</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>5</mn></mrow></msup></mrow> <annotation>2\times 10^{-5}</annotation></semantics></math></td><td>2*8*8*12</td><td>3072</td><td>2 epochs</td><td>0.01</td></tr><tr><td>RL</td><td><math><semantics><mrow><mn>1</mn> <mo>×</mo> <msup><mn>10</mn> <mrow><mo>−</mo> <mn>6</mn></mrow></msup></mrow> <annotation>1\times 10^{-6}</annotation></semantics></math></td><td>64*12</td><td>8192</td><td>1 epoch</td><td>-</td></tr></tbody></table>

### C.3. Baselines

Unless noted otherwise, every baseline shares SeqLLM’s datasets, splits, tokenization, and metrics (Section 4.1); only the architecture, input modality, or training objective changes, so each comparison isolates a single factor. We group baselines by experiment.

#### Sequence modeling on WeChat Pay (RQ1, Table ).

PANTHER [^10] is the sequence-only reference (a behavioral pretraining transformer that outperforms SASRec/HSTU). We retrain it from scratch on the WeChat Pay corpus with whole-event IDs, matching Qwen3-8B in parameter count and Transformer configuration and using the capability-injection optimizer, schedule, and token budget (Table 12); it has no language head, so its language cells are dashes. Qwen3-8B (Text, zero-shot) [^24] serializes each transaction as text (Section 3.1) and predicts the next one; we map its free-form output back to the field vocabulary (Section 3.2) before scoring HR@10. Qwen3-8B + CPT (naive) is the primary matched control (same corpus, language replay, token budget, and updates as SeqLLM; only the objective differs).

#### Offline comparison of deployment candidates (RQ3, Table ).

Both DeepSeek-based candidates are fine-tuned on the same downstream merchant-risk data with the same SFT–RL recipe, optimization setup, and risk-scoring method. They differ only in sequence input: the text-only variant receives the merchant profile and complaint text, while the textualized-sequence variant additionally receives the transaction sequence serialized as plain text. P@ $r\%$ is the risky fraction among the $r\%$ highest-scored merchants, and latency is normalized to SeqLLM under the same inference stack. Relative changes in Table 13 use the textualized-sequence DeepSeek variant as the reference for each cutoff.

Table 13. Complete offline comparison of merchant-ranking deployment candidates. The textualized-sequence DeepSeek model was not deployed.

| Model | Max tokens | Latency | P@0.01% | P@0.1% | P@1% |
| --- | --- | --- | --- | --- | --- |
| SeqLLM (Ours) | $10{,}000$ | $1\times$ | 0.970 (+17.7%) | 0.792 (+28.6%) | 0.326 (+141.5%) |
| DeepSeek-based (text only, SFT+RL) | $40{,}960$ | $20\times$ | 0.930 (+12.9%) | 0.556 ($-9.7\%$) | 0.231 (+71.1%) |
| DeepSeek-based (text + serialized sequence, SFT+RL) | $40{,}960$ | $20\times$ | 0.824 (reference) | 0.616 (reference) | 0.135 (reference) |

SeqLLM achieves the best precision at all three ranking cutoffs while requiring one twentieth of the inference time. The textualized-sequence DeepSeek model degraded performance relative to the text-only DeepSeek model and was therefore not deployed.

#### Production merchant risk screening (RQ3, Table ).

The 0.6B and 8B cascade stages share the identical SeqLLM pipeline, differing only in backbone size (Qwen3-0.6B vs. Qwen3-8B; Table 12). The shadow precision evaluates the 8B ranking stage alone on a shared upstream candidate pool; the 0.6B scanner was introduced for deployment and is evaluated separately below. The three-month shadow evaluation, labeling protocol, and appeal/exoneration definitions are described in Section 4.4; here we note only that exact candidate, action, and appeal counts and traffic composition are withheld for business confidentiality. The serialized DeepSeek variant was not deployed.

#### Recall-side monitoring of the 0.6B scanner (Table ).

After deployment, we monitored the 0.6B scanner over a six-day window, scoring all $\sim$ 50M active merchants daily. Daily true positives are confirmed through the same 30-day maturation pipeline as the shadow labels (Section 4.4). Table 14 reports Recall@Top- $N$ at three screening depths. Retaining only the top $0.4\%$ ($\sim$ 200K candidates) covers $91.24\%$ of confirmed risky merchants on average. This recall result evaluates the deployed scanner separately from the 8B-only shadow precision reported in Table 3.

Table 14. Daily recall of the 0.6B scanner over a six-day window. The scanner scores $\sim$ 50M merchants per day; true positives are confirmed within a 30-day maturation window. R@Top- $N$ is the fraction of confirmed risky merchants captured among the $N$ highest-scored merchants.

| Day | True positives | R@Top-5w | R@Top-10w | R@Top-20w |
| --- | --- | --- | --- | --- |
|  |  | (0.1%) | (0.2%) | (0.4%) |
| 1 | 10,150 | 0.7778 | 0.8560 | 0.9126 |
| 2 | 10,046 | 0.7754 | 0.8562 | 0.9123 |
| 3 | 9,611 | 0.7756 | 0.8515 | 0.9119 |
| 4 | 9,475 | 0.7747 | 0.8502 | 0.9127 |
| 5 | 9,355 | 0.7746 | 0.8517 | 0.9123 |
| 6 | 9,279 | 0.7725 | 0.8510 | 0.9123 |
| Avg. | 9,653 | 0.7751 | 0.8528 | 0.9124 |

#### Public recommendation on MovieLens/Amazon (RQ4, Table ).

User-LLM [^16] is the main encoder-based competitor. We reproduce its default *Full* training strategy—the configuration used for all experiments reported in the original paper: a Transformer user encoder produces user embeddings that are projected to the LLM hidden size and injected via cross-attention at intermediate layers, with the Qwen3-8B backbone finetuned jointly with the encoder, projection, and cross-attention modules. Its lower language scores in Table 4 are consistent with this full finetuning of the backbone on recommendation data. We use the same backbone, preprocessing, and splits (Appendix B.2) and the TS-SFT budget (Table 12). Vanilla-Sequence (item-token history only) and Textualized (item history as text) reuse the sequence-only and text-serialization references above.

#### Item recommendation and understanding on RecIF (RQ4, Tables and ).

OneRec-8B [^22] is OpenOneRec’s unified item-token sequence–language model; we use the checkpoint from its official Hugging Face repository and the official RecIF split (Appendix B.3). At the sequence-acquisition stage, we compare OneRec-Pretrain with SeqLLM Align $\to$ SFT; for final adaptation, we compare OneRec Pretrain $\to$ SFT $\to$ Distill $\to$ RL with SeqLLM Align $\to$ SFT $\to$ RL. Both methods use 156K behavioral sequences and the same 13M-caption alignment pool. OneRec’s 33B-token co-pretraining allocates 62% to general-domain text ($\sim$ 26M examples) and is followed by $\sim$ 2.6M general SFT examples; SeqLLM uses 100K general-instruction examples.

The GPU-days in Table 5 are estimated as

$$
\text{GPU-days}\approx\frac{6ND}{\text{MFU}\cdot F_{\text{peak}}\cdot 86400},
$$

where $N\!\approx\!8\times 10^{9}$, $D$ is the processed-token count (for OneRec-8B, taken from its officially reported per-stage token budgets [^22]; for SeqLLM, measured from our training logs), $F_{\text{peak}}$ is the per-GPU BF16 throughput, and $\text{MFU}\in[0.4,0.5]$. Since both models share the backbone size and hardware, GPU-days scale with $D$, giving $460/\!\sim\!994$ (OneRec) vs. $105/206$ (SeqLLM), a $4.4\times/4.8\times$ reduction.

## Appendix D Details of Evaluation

#### Next-transaction HR@kk (WeChat Pay).

A transaction event is described by $28$ fields, but next-event prediction is scored on five core fields that identify a transaction. For each test position we generate the $k$ most probable next events by beam search over the core-field tokens, ranked by joint probability, and record a hit if the ground-truth event matches one of the top- $k$ candidates on all five core fields. HR@ $k$ is the fraction of positions with a hit; we report HR@10.

#### Next-item Recall@kk (MovieLens/Amazon).

Each user contributes a single held-out last item (Appendix B.2). We rank items by predicted probability and set Recall@ $k=1$ when the held-out item is among the top- $k$; we report Recall@5. With one relevant item this coincides with HR@ $k$.

#### Top-ranked precision (merchant risk).

The held-out test window covers 30 consecutive days; on each day we rank the daily cohort ($\approx\!0.99$ M merchants) by the risk score $s$ (Appendix C.3). Precision@Top- $r\%$ is the fraction of truly risky merchants among the highest-scored $r\%$ of each daily cohort, averaged over the 30 days. Even at the most selective cutoff ($r=0.01$), this aggregates $\approx$ 99 merchants per day, i.e., $\approx$ 3,000 candidates over the window. We report $r=1/0.1/0.01$ in Table 2.

#### Fraud-detection Precision/Recall@Top-r%r\\%.

We conduct a concurrent three-month A/B test with users randomly assigned to two equal-sized arms. The evaluation covers billions of transactions per day. Fraud labels return through the production feedback pipeline after a 14-day maturation window, and only fully matured transactions are included; exact traffic counts are withheld for business confidentiality. Within each online A/B arm, we rank transactions by the fraud detection model’s fraud score and select the highest-scored $r\%$. Precision is the fraudulent fraction within this set, while recall is the fraction of all labeled fraudulent transactions retrieved by it. Section 4.4 reports the SeqLLM arm’s absolute percentage-point gains over the production baseline: precision at $r=0.01/0.1$ and recall at $r=0.1/1$.

#### RecIF Pass@kk and Recall@kk.

Following OpenOneRec [^22],<sup>5</sup> Pass@ $k$ measures whether the ground-truth item appears among $k$ generated candidates and Recall@ $k$ the fraction of all relevant items retrieved; we report P@1, P@32, and R@32.

#### Preference and understanding metrics.

Favorite-genre/category prediction and the RecProbe classification tasks (Appendix B.3) are scored by accuracy over the extracted option; Video Interest Ranking uses NDCG@3 of the three candidates against the reference order; review generation uses ROUGE-L ($F_{1}$) between the generated and held-out reviews.

#### General language ability.

MMLU, C-Eval, and AGIEval are evaluated with the lm-evaluation-harness <sup>6</sup> on vLLM as log-likelihood completion without a chat template, matching the Qwen3-8B base protocol; C-Eval uses 5-shot, whereas MMLU and AGIEval are evaluated 0-shot. We report multiple-choice accuracy.

## Appendix E Controlled Ablation of the Behavior Projector

We isolate whether the projector helps the LLM merely memorize new token meanings or compose and apply them. The full model is compared with a *w/o-projector* variant (named consistently with Table 4) that removes $g_{\psi}$ and tunes the behavior-token embeddings directly. Both variants use the same backbone, text-based token initialization, translation-to-reasoning data, optimization schedule, and number of updates; the projector is the sole controlled difference.

We evaluate two capability levels. *Token translation* measures whether the model can recover the textual description of an individual behavior token. *Compositional application* measures multi-event reasoning and downstream risk decisions, where the model must combine token meanings across fields and time. The w/o-projector variant learns the translation task but transfers poorly to compositional application. Adding the shared residual projector leaves token translation intact while substantially improving multi-event reasoning and downstream transfer. This separation supports the projector’s intended role as a shared semantic interface rather than additional per-token storage.

Table 15. Controlled ablation of the behavior projector on industrial data. All factors except the projection interface are held fixed. Field Acc. is the fraction of behavior fields translated correctly; higher is better for all metrics.

| Capability | Metric | w/o projector | w/ projector |
| --- | --- | --- | --- |
| Token translation | Field Acc. | 0.87 | 0.93 |
| Multi-event Q&A | Accuracy | 0.32 | 0.86 |
| Risk-assessment Q&A | Accuracy | 0.21 | 0.78 |

[^1]: Learning a hierarchical embedding model for personalized product search. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, Shinjuku, Tokyo, Japan, August 7-11, 2017, N. Kando, T. Sakai, H. Joho, H. Li, A. P. de Vries, and R. W. White (Eds.), pp. 645–654. External Links: [Link](https://doi.org/10.1145/3077136.3080813), [Document](https://dx.doi.org/10.1145/3077136.3080813) Cited by: §1.

[^2]: Recommender systems survey. Knowledge-based systems 46, pp. 109–132. Cited by: §1.

[^3]: The revolution of multimodal large language models: a survey. Findings of the association for computational linguistics: ACL 2024, pp. 13590–13618. Cited by: §1.

[^4]: Onerec: unifying retrieve and rank with generative recommender and iterative preference alignment. arXiv preprint arXiv:2502.18965. Cited by: §1, §2.

[^5]: Recommendation as language processing (RLP): A unified pretrain, personalized prompt & predict paradigm (P5). In RecSys ’22: Sixteenth ACM Conference on Recommender Systems, Seattle, WA, USA, September 18 - 23, 2022, J. Golbeck, F. M. Harper, V. Murdock, M. D. Ekstrand, B. Shapira, J. Basilico, K. T. Lundgaard, and E. Oldridge (Eds.), pp. 299–315. External Links: [Link](https://doi.org/10.1145/3523227.3546767), [Document](https://dx.doi.org/10.1145/3523227.3546767) Cited by: §2.

[^6]: Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300. Cited by: §4.1.

[^7]: Session-based recommendations with recurrent neural networks. In 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings, Y. Bengio and Y. LeCun (Eds.), External Links: [Link](http://arxiv.org/abs/1511.06939) Cited by: §2.

[^8]: C-eval: a multi-level multi-discipline chinese evaluation suite for foundation models. Advances in neural information processing systems 36, pp. 62991–63010. Cited by: §4.1.

[^9]: Self-attentive sequential recommendation. In IEEE International Conference on Data Mining, ICDM 2018, Singapore, November 17-20, 2018, pp. 197–206. External Links: [Link](https://doi.org/10.1109/ICDM.2018.00035), [Document](https://dx.doi.org/10.1109/ICDM.2018.00035) Cited by: §1, §2.

[^10]: PANTHER: generative pretraining beyond language for sequential user behavior modeling. CoRR abs/2510.10102. External Links: [Link](https://doi.org/10.48550/arXiv.2510.10102), [Document](https://dx.doi.org/10.48550/ARXIV.2510.10102), 2510.10102 Cited by: §C.3, §1, §2, §3.2, §4.1, §4.2, Table 1.

[^11]: How can recommender systems benefit from large language models: a survey. ACM Transactions on Information Systems 43 (2), pp. 1–47. Cited by: §1, §2.

[^12]: Autofis: automatic feature interaction selection in factorization models for click-through rate prediction. In proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 2636–2645. Cited by: §1.

[^13]: Visual instruction tuning. Advances in neural information processing systems 36, pp. 34892–34916. Cited by: §1.

[^14]: OneRec-think: in-text reasoning for generative recommendation. CoRR abs/2510.11639. External Links: [Link](https://doi.org/10.48550/arXiv.2510.11639), [Document](https://dx.doi.org/10.48550/ARXIV.2510.11639), 2510.11639 Cited by: §1, §2.

[^15]: The application of data mining techniques in financial fraud detection: a classification framework and an academic review of literature. Decision support systems 50 (3), pp. 559–569. Cited by: §1, §2.

[^16]: User-llm: efficient LLM contextualization with user embeddings. In Companion Proceedings of the ACM on Web Conference 2025, WWW 2025, Sydney, NSW, Australia, 28 April 2025 - 2 May 2025, G. Long, M. Blumestein, Y. Chang, L. Lewin-Eytan, Z. H. Huang, and E. Yom-Tov (Eds.), pp. 1219–1223. External Links: [Link](https://doi.org/10.1145/3701716.3715463), [Document](https://dx.doi.org/10.1145/3701716.3715463) Cited by: §C.3, §1, §1, §2, 2nd item, §4.5, Table 4.

[^17]: A comprehensive survey of data mining-based fraud detection research. arXiv preprint arXiv:1009.6119. Cited by: §1, §2.

[^18]: Recommender systems with generative retrieval. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (Eds.), External Links: [Link](http://papers.nips.cc/paper%5C_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html) Cited by: §2.

[^19]: Factorization machines. In 2010 IEEE International conference on data mining, pp. 995–1000. Cited by: §1.

[^20]: A comparison of supervised learning to match methods for product search. arXiv preprint arXiv:2007.10296. Cited by: §1.

[^21]: BERT4Rec: sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, CIKM 2019, Beijing, China, November 3-7, 2019, W. Zhu, D. Tao, X. Cheng, P. Cui, E. A. Rundensteiner, D. Carmel, Q. He, and J. X. Yu (Eds.), pp. 1441–1450. External Links: [Link](https://doi.org/10.1145/3357384.3357895), [Document](https://dx.doi.org/10.1145/3357384.3357895) Cited by: §2.

[^22]: OpenOneRec technical report. CoRR abs/2512.24762. External Links: [Link](https://doi.org/10.48550/arXiv.2512.24762), [Document](https://dx.doi.org/10.48550/ARXIV.2512.24762), 2512.24762 Cited by: §C.3, §C.3, Appendix D, §1, §1, §2, §3.3, 3rd item, §4.1, §4.1, §4.5.

[^23]: OneReason technical report. CoRR abs/2606.06260. External Links: [Link](https://doi.org/10.48550/arXiv.2606.06260), [Document](https://dx.doi.org/10.48550/ARXIV.2606.06260), 2606.06260 Cited by: §1, §2, footnote 1.

[^24]: Qwen3 technical report. CoRR abs/2505.09388. External Links: [Link](https://doi.org/10.48550/arXiv.2505.09388), [Document](https://dx.doi.org/10.48550/ARXIV.2505.09388), 2505.09388 Cited by: §C.3, Table 1.

[^25]: Learning latent vector spaces for product search. In Proceedings of the 25th ACM international on conference on information and knowledge management, pp. 165–174. Cited by: §1.

[^26]: Sequential recommender systems: challenges, progress and prospects. arXiv preprint arXiv:2001.04830. Cited by: §1, §2.

[^27]: Actions speak louder than words: trillion-parameter sequential transducers for generative recommendations. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, R. Salakhutdinov, Z. Kolter, K. A. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp (Eds.), Proceedings of Machine Learning Research, pp. 58484–58509. External Links: [Link](https://proceedings.mlr.press/v235/zhai24a.html) Cited by: §2.

[^28]: Mm-llms: recent advances in multimodal large language models. Findings of the Association for Computational Linguistics: ACL 2024, pp. 12401–12430. Cited by: §1.

[^29]: Agieval: a human-centric benchmark for evaluating foundation models. In Findings of the association for computational linguistics: NAACL 2024, pp. 2299–2314. Cited by: §4.1.