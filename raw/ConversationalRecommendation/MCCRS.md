---
title: "Multi-Type Context-Aware Conversational Recommender Systems via Mixture-of-Experts"
source: "https://arxiv.org/html/2504.13655v2"
author:
published:
created: 2026-08-06
description:
tags:
  - "clippings"
---
Jie Zou Cheng Lin Weikang Guo Zheng Wang Jiwei Wei Yang Yang Heng Tao Shen

###### Abstract

Conversational recommender systems enable natural language conversations and thus lead to a more engaging and effective recommendation scenario. As the conversations for recommender systems usually contain limited contextual information, many existing conversational recommender systems incorporate external sources to enrich the contextual information. However, how to combine different types of contextual information is still a challenge. In this paper, we propose a multi-type context-aware conversational recommender system, called MCCRS, effectively fusing multi-type contextual information via mixture-of-experts to improve conversational recommender systems. MCCRS incorporates both structured information and unstructured information, including the structured knowledge graph, unstructured conversation history, and unstructured item reviews. It consists of several experts, with each expert specialized in a particular domain (i.e., one specific contextual information). Multiple experts are then coordinated by a ChairBot to generate the final results. Our proposed MCCRS model takes advantage of different contextual information and the specialization of different experts followed by a ChairBot breaks the model bottleneck on a single contextual information. Experimental results demonstrate that our proposed MCCRS method achieves significantly higher performance compared to existing baselines.

###### keywords:

Conversational recommendation, Recommender system, Mixture-of-experts

\[inst1\]organization= School of Computer Science and Engineering, addressline=University of Electronic Science and Technology of China, city=Chengdu, postcode=611731, country=China \[inst3\]organization=School of Management Science and Engineering, addressline=Southwestern University of Finance and Economics, city=Chengdu, postcode=611130, country=China

\[inst2\]organization=School of Computer Science and Technology, addressline=Tongji University, city=Shanghai, postcode=200092, country=China

## 1 Introduction

Conversational recommender systems have emerged as a prominent research topic [^1] [^2] in recent years. Unlike traditional recommender systems that mainly depend on historical data and user behavior, conversational recommender systems leverage the power of natural language conversations, so as to provide personalized and context-aware recommendations. With the interaction through conversations, users are supposed to experience a more engaging and effective recommendation.

As conversational recommender systems are gaining significant research interest, numerous approaches have been introduced to advance the field [^3] [^4] [^5] [^6] [^7] [^8] [^9] [^10] [^11] [^12] [^13]. Given that the conversations for recommender systems are usually short and contain limited contextual information, incorporating external sources to enrich the contextual information is soon become common in many existing studies [^14] [^4]. Among these external sources, the most frequently used ones are knowledge graphs [^15] [^3] and item reviews [^14]. However, how to combine different types of contextual information is still challenging.

One of the challenges is the heterogeneity of different external data. For instance, knowledge graphs are structured data while conversations and item reviews are unstructured data. There is a semantic gap between multi-type and heterogeneous data. Furthermore, different external data often belong to different semantic spaces and thus are difficult to align. One possible solution to fuse different external data is contrastive learning [^14] [^16]. However, contrastive learning requires the same entries contained in all different external data to calculate the contrastive loss. This is not always fulfilled.

In this work, we propose a Multi-type Context-aware Conversational Recommender System, called MCCRS, to effectively fuse multi-type contextual information via mixture-of-experts so as to improve conversational recommender systems. MCCRS incorporates both structured information and unstructured information, including the structured knowledge graph, unstructured conversation history, and unstructured item reviews. We carefully design our model as a mixture-of-experts structure, which consists of a ChairBot and several experts. Each expert specializes in a particular domain, i.e., one external source. The ChairBot coordinates multiple experts and generates the final results. Compared with previous approaches, the advantages of MCCRS are at least four-fold: First, our MCCRS model relieves the challenge of data heterogeneity and semantic fusion, as well as relaxes the limitation of contrastive learning. Second, our MCCRS model takes advantage of different contextual information and the specialization of different experts followed by a ChairBot breaks the model bottleneck on a single contextual information. Third, with each expert specialized in a particular domain, it is more easily traceable and analyzes who is responsible when the model generates a good or bad result. Fourth, our MCCRS model is more easily extendable as additional experts can be added easily to incorporate more additional external sources.

In summary, the main contributions of this work are threefold:

- We propose a new method for conversational recommender systems, named MCCRS.
- We propose a new mixture-of-experts framework consisting of a ChairBot and multi-experts to effectively leverage multi-type contextual information for conversational recommender systems.
- The extensive experiments show that our proposed model MCCRS significantly enhances performance compared to various state-of-the-art (SOTA) baselines.

To the best of our knowledge, MCCRS is the first model that devises a ChairBot and multi-experts for conversational recommender systems.

## 2 Related Work

Conversational recommender systems utilize human-like natural language to deliver personalized and engaging recommendations through conversational interfaces like chatbots and intelligence assistants [^1] [^17] [^18] [^19] [^20]. In recent years, it has attracted considerable attention, driven by the rapid evolution of dialog systems.

In general, conversational recommender systems can be classified into two primary classes [^12] [^7] [^21] [^22], including anchor-based conversational recommender systems [^23] and dialog-based conversational recommender systems [^2] [^24].

##### Anchor-based Conversational Recommender Systems

This category of conversational recommender systems [^23] is in the form of “system asks – user response” mode. In this context, the systems ask questions, prompting users to provide feedback, subsequently utilizing this user-provided information to refine their recommendations. For building effective anchor-based conversational recommender systems, recent research has introduced a variety of ways for generating anchors to characterize items, including intent slots (e.g., item aspects and facets) [^25] [^26], entities [^23], topics [^27], and attributes [^28] [^29] [^30] [^31]. These anchors are usually collected to construct a predefined question pool. Based on the constructed question pool, a line of work adopt multi-armed bandit [^32] [^27] [^33] [^34], reinforcement learning [^28], or greedy strategies [^32] [^35] to select appropriate questions to simulate multi-turn interactions to interact with users. By doing so, user feedback through conversational interactions is leveraged to optimize recommendation policies. In addition to selecting appropriate questions, deep reinforcement learning techniques are applied to train policy networks to determine whether to recommend items or ask a question in a turn [^29] [^30]. This category of studies typically makes use of predefined dialog templates to construct conversations, with a primary focus on how to offer recommendations within the fewest possible conversation turns [^23]. In other words, they do not focus on the modeling of natural language conversations [^3].

##### Dialog-based Conversational Recommender Systems

The other category, known as dialog-based conversational recommender system [^36] [^37] [^10] [^38] [^39] [^40], is based on human-generated dialogs. Unlike anchor-based conversational recommender systems simulating conversations based on extracted anchor text, dialog-based conversational recommender systems concentrate on generating human-like responses while making accurate recommendations. In the early stages, as there are no publicly available datasets for dialog-based conversational recommender systems, [^41] propose a conversational recommender system dataset in the movie domain, named ReDial. Subsequently, [^42] propose the INSPIRED dataset tailored for conversational recommender systems with sociable recommendation strategies. Dialogues in these datasets typically encompass items and entities that can be linked to knowledge bases like DBpedia. Consequently, many existing studies incorporate external knowledge graphs to augment conversation information and accurately capture user preference [^8] [^3] [^4] [^12] [^43] [^44] [^45] [^46]. These studies typically employ graph neural networks to encode the knowledge graph and user preferences, alongside a dialog management module to guide the conversation. For example, [^3] introduce the entity-oriented knowledge graph (DBpedia [^47]), to discern user intentions. They employ Relational Graph Convolutional Networks (R-GCNs) to learn entity representations from knowledge graphs [^48] for item recommendations and utilize the Transformer framework to generate natural language responses. Based on the work of [^3], which mainly focuses on the entity-oriented knowledge graph, [^4] expand upon this by introducing an additional knowledge graph, i.e., the word-oriented knowledge graph – ConceptNet [^49]. They propose a novel knowledge graph-based semantic fusion model for conversational recommender systems, harnessing the potential of both knowledge graphs. Subsequently, [^14] propose a contrastive learning approach [^16] from a coarse-to-fine perspective to enhance data semantic fusion. In order to mitigate redundant information in knowledge graphs, [^5] introduce different subgraphs to enhance recommendation performance for conversational recommender systems. To make use of multiple reasoning paths in knowledge graphs, [^6] and [^50] propose conversational recommendation models to perform multi-hop reasoning in knowledge graphs to track shifts in users’ interests. To alleviate the limitation of incomplete knowledge graphs, [^51] present a variational reasoning approach [^15] with dynamic knowledge reasoning over incomplete knowledge graphs. Beyond knowledge graphs, pre-trained language models are also deployed to enhance conversational recommender systems [^52] [^9] [^53] [^54] [^55] [^56] [^57] [^58] [^59] [^60]. [^9] finetune the large-scale pre-trained language models and propose a unified pre-trained language model-based framework to deal with the low-resource challenge in conversational recommender systems. Similarly, [^52] apply a pre-trained language model and propose a conversational recommendation model on top of the prompt learning paradigm. [^57] introduce a demonstration-enhanced prompt learning method for conversational recommendation by retrieving and learning from demonstrations based on dialogue contexts. [^60] propose a novel multi-modal semantic graph prompt learning framework for conversational recommender systems by considering collaborative information with multi-modal features. [^59] propose a novel model of contextual disentanglement for conversational recommender systems by disentangling focus and background information from complex conversation contexts. Another promising direction in conversational recommendation involves the integration of side information [^13] [^11]. For example, [^13] incorporate the item reviews to improve the recommendation performance in conversational recommender systems. [^61] leverage multi-aspect user information to learn multi-aspect user preferences, while [^11] introduce item meta information to improve item representations.

Similar to the existing studies using side information, we also leverage the side information (e.g., item reviews and external knowledge graph) into our model. Different from the aforementioned research, we model multi-type data, including conversation history, knowledge graphs, and item reviews, through a unified mixture-of-experts framework to improve conversational recommender systems.

![Refer to caption](https://arxiv.org/html/2504.13655v2/x1.png)

Figure 1: The framework of our model in a movie recommendation scenario. Our model uses a ChairBot and multiple experts, including a conversation expert, a graph expert, and a review expert, to improve conversational recommender systems.

## 3 Approach

In this section, we introduce the proposed MCCRS model. The proposed model contains a ChairBot and three experts, namely the conversation expert, the graph expert, and the review expert. Figure 1 provides an overview of our model. In the following, we first describe the three experts in detail. Then, we describe our mixture-of-experts recommender to effectively fuse the three experts and our response generator benefiting from the recommender.

### 3.1 Problem Formalization

Formally, in conversational recommender systems, assume there is a user set $\mathcal{U}$ = $\{u_{1},u_{2},\dots,u_{|\mathcal{U}|}\}$, item set $\mathcal{I}$ = $\{i_{1},i_{2},\dots,i_{|\mathcal{I}|}\}$, and conversations. For a conversation, we have the contextual information, including conversation history consisting of a list of utterances, related reviews of item mentions in the conversation, and an external knowledge graph $\mathcal{G}$ (i.e., DBpedia). Each utterance is a sentence at $t$ -th turn in the conversation, represented by a list of entity mentions (i.e., items or other item-related entities) in this paper. The entity set $\mathcal{E}$ = $\{e_{1},e_{2},\dots,e_{|\mathcal{E}|}\}$, consists of all the items and other item-related entities (i.e., $\mathcal{I}\subseteq\mathcal{E}$). Each review consists of a list of sentences $\mathcal{D}$ = $\{d_{1},d_{2},\dots,d_{|\mathcal{D}|}\}$. As the conversation goes on, at $t$ -th turn, we aim to accurately recommend the mentioned item $i^{*}$ from the entire item set $\mathcal{I}$ based on the inferred user preference, along with a response, i.e., the next utterance comprises a sequence of words, to reply to the user. Table 1 summarizes the main notations used in the paper.

Table 1: Notations.

| Notation | Explanation |
| --- | --- |
| $u$, $\mathcal{U}$ | a user, and the user set |
| $i$, $\mathcal{I}$ | an item, and the item set |
| $e$, $\mathcal{E}$ | an entity, and the entity set |
| $r$, $\mathcal{R}$ | a relation, and the relation set |
| $d$, $\mathcal{D}$ | a review sentence, and the set of review sentences |
| $\mathbf{h}$, $\mathbf{H}$ | the elemental embedding and embedding matrix of the input sequences |
| $\mathbf{s}_{k}$, $\mathbf{p}_{k}$ | the entity embedding and position embedding of $k$ -th element in the input sequence |
| $\mathcal{G}$ | a knowledge graph |
| $\mathbf{D}_{m}$ | a representation matrix for review sentences |
| $\mathbf{v}_{R}$ | a review embedding |
| $C$, $G$, $R$ | the conversation expert, graph expert, and review expert |
| $\mathbf{n}_{e}^{(l)}$ | the representation of node (entity) $e$ at the $l$ -th layer of R-GCN |
| $\lambda$ | the importance score of experts |

### 3.2 Conversation Expert

The conversation expert is made up of a sequential transformer [^2]. It extracts both the mentioned items and item-related entities in conversations to form a user sequence. For instance, in Figure 1, a sequence of \[Goodfellas (1990), Crime, Robert De Niro, Reservoir Dogs (1992)\] is extracted for the conversation history. Based on this user sequence, the conversation expert applies a Cloze task [^62] [^63] to randomly mask a portion of the sequences and puts them into a transformer to predict the masked items. The transformer is uniquely designed, with a matrix of hidden size $\times$ vocabulary size in place of the usual positional encoding. The presentation of $k$ -th element of the input sequence is the concentration of positional embedding $\mathbf{s}_{k}$ and sequence embedding $\mathbf{p}_{k}$.

$$
\mathbf{h}_{k}=\mathbf{s}_{k}+\mathbf{p}_{k}.
$$

All elemental embeddings of $\mathbf{h}_{k}$ from the previous steps form a matrix $\mathbf{H}$, which will then be passed through a multi-attention head. Suppose we are at the $n$ -th transformer layer, we will update $\mathbf{H}$ in the following manner:

$$
\mathbf{H}^{n+1}=\text{MultiHead}(\text{PFFN}(\mathbf{H}^{n})),
$$

where MultiHead is a multi-head self-attention sub-layer, and PFFN is a Position-wise Feed-Forward Network constructed by the Feed-Forward Network (FFN) with GELU activation [^64].

Afterward, we would combine their calculation with a residual link. The reason for the residual connection is that the network is more difficult to train as it becomes deeper, and thus employing such a technique, along with layer normalization and dropout, assists with the training.

$$
\mathbf{H}^{n}=\text{LayerNorm}(\mathbf{H}^{n}+\text{Dropout}(\text{sublayer}(\mathbf{H}^{n})),
$$

where the sublayer is either PFFN or MultiHead in Equation 2.

Assuming we are predicting $k$ -th movie, we utilize the final embedding at N-layer of Transformer $\mathbf{h}^{N}_{k}$ to generate our prediction. More specifically, we utilize a softmax function to output the item distribution:

$$
P_{C}(i)=\text{Softmax}(\mathbf{W}\times\mathbf{h}^{N}+\mathbf{b}),
$$

where $\mathbf{W}$ is a learnable transformation matrix, and $\mathbf{b}$ denotes the trainable bias matrix. On top of the produced probability of items, we use a cross-entropy loss to optimize the parameters of the conversation expert.

$$
L_{rec}=-\sum_{j=1}\sum_{i\in\mathcal{I}}y_{ij}\cdot\log P(i),
$$

where $j$ is the conversation index, and $y_{ij}$ indicates the ground-truth label of items. After the training, we collect both the predicted item probability $P_{C}(i)$ and hidden embedding $\mathbf{h}^{N}$ for the ChairBot.

![Refer to caption](https://arxiv.org/html/2504.13655v2/x2.png)

Figure 2: Illustration of an example graph.

### 3.3 Graph Expert

Given that it is difficult to comprehensively understand user preferences based solely on conversational context, the inclusion of external knowledge is necessary to encode user preferences. For dialogs in conversational recommender systems, item mentions and item-related entities can be extracted to construct external knowledge graphs. Inspired by the previous studies [^3] [^4], we introduce a knowledge graph sourced from DBpedia [^65] to encode structural and relational information in the knowledge graph. Specifically, we perform entity linking [^66] to map the item mentions and item-related entities in the dataset to DBpedia. With the help of the external knowledge graph, it enables us to model user preferences more accurately.

A knowledge graph $\mathcal{G}$ (i.e., DBpedia) comprises an entity set $\mathcal{E}$ and a relation set $\mathcal{R}$. The knowledge graph $\mathcal{G}$ stores semantic facts as triples $<e_{1},r,e_{2}>$, where $e_{1},e_{2}\in\mathcal{E}$ represents items or item-related entities and $r\in\mathcal{R}$ represents the relation between $e_{1}$ and $e_{2}$. For example, as shown in Figure 2, for a target movie $e_{i^{*}}$, it might be associated with entity $e_{1}$ with one type of relationship, while $e_{2}$ and $e_{3}$ are of another, such as $e_{1}$ being the genre while $e_{2}$ and $e_{3}$ are the actors in the movies.

In this paper, we employ R-GCN [^67] to encode entity representations in the knowledge graph $\mathcal{G}$. Specifically, we pre-train the representation $e\in\mathcal{E}$ by using R-GCN to initialize the offline embeddings of items or other item-related entities. Formally, in R-GCN, the node embedding at $(l+1)$ -th layer is calculated as:

$$
\mathbf{n}_{e}^{(l+1)}=\sigma(\sum_{r\in\mathcal{R}}\sum_{e^{\prime}\in\mathcal{E}_{e}^{r}}\frac{1}{Z_{e,r}}\mathbf{W}_{r}^{(l)}\mathbf{n}_{e^{\prime}}^{(l)}+\mathbf{W}^{(l)}\mathbf{n}_{e}^{(l)}),
$$

where $\mathbf{n}_{e}^{(l)}$ denotes the representation of node (i.e., entity) $e$ at the $l$ -th layer, and $\mathcal{E}_{e}^{r}$ is the set of neighboring nodes for $e$ under the relation $r$. $\mathbf{W}^{(l)}$ represents a learnable transformation matrix for transforming the representations of nodes at the $l$ -th layer $\mathbf{n}_{e}^{(l)}$, while $\mathbf{W}_{r}^{(l)}$ is another relation-specific learnable matrix for transforming the embedding of neighboring nodes under the relation $r$. $\sigma$ is the ReLU activation function and $Z_{e,r}$ denotes a normalization factor.

After aggregating the structural and relational information of the knowledge graph, we obtain all the node representations (i.e., representations of items and other item-related entities) on the top R-GCN layer. As historically interacted entities are critical for modeling user preference, we aggregate the overall representation for each conversation. Specifically, we obtain the entity mentions for each conversation, and then apply a self-attention mechanism, that can automatically consider the levels of entity importance, to obtain the graph-based user representation $\mathbf{n_{e_{u}}}$. We compute the item probability to rank all the items:

$$
P_{G}(i)=\text{Softmax}({\mathbf{n}^{\intercal}_{e_{u}}}\mathbf{n}_{i}),
$$

where $\mathbf{n}_{i}$ represents the learned embedding for item $i$. Again, a cross-entropy loss is used for the graph expert similar to the conversation expert (i.e., Equation 5). Finally, the graph expert produces the item probability $P_{G}(i)$ and node embeddings for the ChairBot.

### 3.4 Review Expert

Besides the conversation history and knowledge graph, the review data related to items is able to improve the performance of conversational recommender systems. The review data is a set of sentences about items written by online users. To encode review text, we apply a two-step approach. First, we encode each sentence by employing the standard Transformer model. Second, we utilize a sentence-level self-attention layer to produce the overall review-based representation. For item $i$, assuming it has $m$ sentences, we first obtain $\mathbf{D^{m}}$, which is a representation matrix for sentences where each column is a sentence representation [^67]. Then the overall review-based representation is:

$$
\mathbf{v}_{R}=\text{SelfAttention}(\text{Transformer}(\mathbf{D^{m}})).
$$

After the above encoding, we pass $\mathbf{v}_{R}$ through another self-attention layer to produce the final review embeddings.<sup>1</sup> We generate the item probability, by using the final review embeddings to replace the node embeddings in Equation 7:

$$
P_{R}(i)=\text{Softmax}({\mathbf{v}^{\intercal}_{R_{u}}}\mathbf{v}_{R_{i}}).
$$

Also, a cross-entropy loss is used for optimizing the review expert similar to the conversation expert (i.e., Equation 5). Again, the item probability $P_{R}(i)$ and review embeddings are passed to the ChairBot.

### 3.5 Mixture-of-Experts Recommender

After the above three experts, which are specialized in three particular types of data, we then fuse them by introducing a ChairBot. The ChairBot coordinates the three experts and integrates their results to generate more accurate and relevant recommendations. Inspired by LSTM-like structure [^68], we first collect the hidden embeddings from all experts $\mathbf{h}_{i}^{C},\mathbf{h}_{i}^{G},\mathbf{h}_{i}^{R}$, and the predictions from all experts $\mathbf{p}_{i}^{C},\mathbf{p}_{i}^{G},\mathbf{p}_{i}^{R}$ for the item $i$, and concatenate them. The $C$, $G$, and $R$ denote conversation expert, graph expert, and review expert respectively.

$$
\displaystyle\mathbf{h}_{C}
$$
 
$$
\displaystyle=\mathbf{h}_{i}^{C}\oplus\mathbf{p}_{i}^{C},
$$
$$
\displaystyle\mathbf{h}_{G}
$$
 
$$
\displaystyle=\mathbf{h}_{i}^{G}\oplus\mathbf{p}_{i}^{G},
$$
$$
\displaystyle\mathbf{h}_{R}
$$
 
$$
\displaystyle=\mathbf{h}_{i}^{R}\oplus\mathbf{p}_{i}^{R}.
$$

Then, we generate the normalized importance score:

$$
\displaystyle\beta_{b}=\textit{MLP}(\mathbf{h}_{b}),
$$
$$
\displaystyle\lambda_{b}=\frac{\beta_{b}}{\beta_{C}+\beta_{G}+\beta_{R}},
$$

where ${b={C,G,R}}$, and MLP is a linear layer.

Finally, we compute the recommendation probability for an item:

$$
P_{rec}(i)=\lambda_{C}*P_{C}(i)+\lambda_{G}*P_{G}(i)+\lambda_{R}*P_{R}(i),
$$

where $P_{C}(i)$, $P_{G}(i)$, and $P_{R}(i)$ are item probabilities generated by the conversation expert, graph expert, and review expert, respectively. To fine-tune the entire item recommender, we also apply a cross-entropy loss in the Mixture-of-Experts recommender module (i.e., Equation 5) to fine-tune the representations and improve the performance.<sup>2</sup>

### 3.6 Response Generator

The response generator can be improved by the recommender system as it provides recommendation-aware vocabulary bias. As such, the recommender system helps generate more consistent and diverse responses. Following [^4], we integrate multiple cross-attention layers within a standard Transformer decoder architecture to merge the pre-trained representations effectively. We modify the decoder by fusing the entity representations from our conversation expert, graph expert, and review expert:

$$
\displaystyle\mathbf{A}^{l}_{0}
$$
 
$$
\displaystyle=\text{MultiHead}[\mathbf{B}^{(l-1)},\mathbf{B}^{(l-1)},\mathbf{B}^{(l-1)}],
$$
$$
\displaystyle\mathbf{A}^{l}_{1}
$$
 
$$
\displaystyle=\text{MultiHead}[\mathbf{A}^{l}_{0},\mathbf{F}_{C},\mathbf{F}_{C}],
$$
$$
\displaystyle\mathbf{A}^{l}_{2}
$$
 
$$
\displaystyle=\text{MultiHead}[\mathbf{A}^{l}_{1},\mathbf{F}_{G},\mathbf{F}_{G}],
$$
$$
\displaystyle\mathbf{A}^{l}_{3}
$$
 
$$
\displaystyle=\text{MultiHead}[\mathbf{A}^{l}_{2},\mathbf{F}_{R},\mathbf{F}_{R}],
$$
$$
\displaystyle\mathbf{A}^{l}_{4}
$$
 
$$
\displaystyle=\text{MultiHead}[\mathbf{A}^{l}_{3},\mathbf{X},\mathbf{X}],
$$
$$
\displaystyle\mathbf{B}^{l}
$$
 
$$
\displaystyle=\text{FFN}(\mathbf{A}^{l}_{4}),
$$

where MultiHead\[·, ·, ·\] represents the multi-head attention function. FFN(·) denotes a feed-forward network. $\mathbf{X}$ is the representation matrix of dialogue history obtained from a standard Transformer encoder. $\mathbf{B}^{l}$ is the representation matrix at the $l$ -th layer from the decoder. $\mathbf{A}^{l}_{0}$, $\mathbf{A}^{l}_{1}$, $\mathbf{A}^{l}_{2}$, $\mathbf{A}^{l}_{3}$, and $\mathbf{A}^{l}_{4}$ are embeddings after self-attention, cross-attention with $\mathbf{F}_{C}$ from the conversation expert, cross-attention with $\mathbf{F}_{G}$ from the graph expert, cross-attention with $\mathbf{F}_{R}$ from the review expert, and cross-attention with the encoder output $\mathbf{X}$, respectively.

## 4 Experiments

In our experiments, we want to address the following research questions:

RQ1

How does our proposed model perform in comparison to prior baselines?

RQ2

What is the impact of different components of the model?

RQ3

How do various parameters of our proposed model influence its effectiveness?

### 4.1 Experimental Setup

Table 2: The dataset statistics in our experiments.

| Dataset | \# Conversations | \# Utterances | \# Users | \# Items |
| --- | --- | --- | --- | --- |
| INSPIRED | 1,001 | 35,811 | 1,482 | 1,783 |
| ReDial | 10,006 | 182,150 | 956 | 51,699 |

#### 4.1.1 Dataset

We use the two typical conversational recommendation datasets, ReDial [^41] and INSPIRED [^42], as done in [^52]. The dataset statistics are presented in Table 2. ReDial [^41] is an English dataset for conversational recommendation, including a collection of annotated dialogs where a recommender offers movie suggestions for the seeker. This dataset contains 10,006 conversations, 182,150 utterances, 956 users, and 51,699 movies. INSPIRED is another dataset of conversational movie recommendations, though it is smaller in scale compared to ReDial. Both datasets are commonly used to evaluate conversational recommender systems [^52]. The datasets are split into training, validation, and test sets by 8:1:1 ratio by prior studies [^41] [^52]. For review data that is not contained in the above datasets, we retrieve reviews for movies from IMDb <sup>3</sup> similar to [^14].

#### 4.1.2 Evaluation Metrics

For evaluating recommendations in conversational recommender systems, we utilize Recall@k (where k = 1, 10, and 50) as our metrics, following previous work [^3] [^4] [^14]. In each conversation, we begin evaluation from the first sentence of the system’s responses, i.e., we treat each item or utterance from the recommender as ground truth and assess them sequentially throughout the conversation, consistent with prior work [^3] [^4]. When testing, we use all items as candidates to rank for recommendations.

For evaluating the conversation quality, we use both automated and human evaluation methods. For automated evaluation, we apply Distinct-n (n = 2, 3, 4) to measure sentence-level diversity, as done in previous studies [^15] [^14]. For human evaluation, three annotators are invited to review the generated responses of all baselines and our model on two dimensions: fluency and informativeness. Fluency assesses the natural flow of the responses, while informativeness measures the relevance and interest of the information provided. The annotators rate the responses on a scale from 0 to 2, and the average score from all three annotators is calculated to report the results.

#### 4.1.3 Implementation Details

The model is trained via the Adam optimizer [^69] with a batch size of 256. The learning rate is 1e-4. We use 50 as the maximum length of sequence K. To ensure training stability, a gradient global norm clipping of 5 and an L2 regularization of 0.01 are adopted. For the Transformer, we set the number of Transformer layers N as 2 and the head number as 2. The number of R-GCN layers and the normalization factor $Z_{e,r}$ of R-GCN are set to 1. The parameters of mask proportion and hidden dimensionality are discussed in Section 4.4.

Table 3: Recommendation results on ReDial. ‘\*’ indicates significant improvements over the best baseline (Fisher random test, $p$ -value $<0.05$). The best performances are highlighted in bold, while the second-best are underlined (Unless otherwise reported, we use ‘\*’ to indicate significant improvements, bold values to indicate the best performances, and underlined values to indicate the second-best performances throughout the paper). Our proposed MCCRS method significantly outperforms SOTA baselines on the recommendation task on ReDial.

| Model | Recall@1 | Recall@10 | Recall@50 |
| --- | --- | --- | --- |
| Popularity | 0.012 | 0.061 | 0.179 |
| TextCNN | 0.013 | 0.068 | 0.191 |
| BERT | 0.014 | 0.117 | 0.191 |
| ReDial | 0.023 | 0.129 | 0.287 |
| KBRD | 0.031 | 0.150 | 0.336 |
| KGSF | 0.039 | 0.183 | 0.378 |
| RevCore | 0.046 | 0.220 | 0.396 |
| VRICR | 0.054 | 0.244 | 0.406 |
| $\text{C}^{2}$ -CRS | 0.053 | 0.233 | 0.407 |
| MCCRS | 0.057\* | 0.250\* | 0.473\* |

#### 4.1.4 Baselines

In this work, we compare our approach with four typical baselines, along with several representative baselines commonly used on ReDial and INSPIRED.

- Popularity is a typical approach ranking the items based on historical recommendation frequency.
- TextCNN [^70] is a typical model that is based on CNN representations.
- BERT [^62] is a typical method, which learns from the sentences to generate recommendations.
- Transformer [^71] is a typical encoder-decoder model based on the Transformer architecture, widely used for generating responses.
- ReDial [^41] serves as a benchmark model for ReDial by employing an autoencoder-based recommender system.
- KBRD [^3] leverages the DBpedia knowledge graph and incorporates a knowledge-enhanced recommender to improve conversational recommendation.
- KGSF [^4] incorporating word-oriented and entity-oriented knowledge graphs for the conversational recommendation.
- RevCore [^13] is a review-augmented conversational recommender by leveraging reviews to enrich item information.
- VRICR [^51] is one of the SOTA conversational recommendation methods, which is based on a variational reasoning method to complete the missing information in incomplete knowledge graphs.
- $\text{C}^{2}$ -CRS [^14] is one of the SOTA conversational recommendation methods that utilizes contrastive learning and semantic fusion of multi-type information including context sentences, reviews, and knowledge graphs.

Among these baselines, Popularity, TextCNN, BERT, and Transformer are classical methods, while ReDial, KBRD, KGSF, RevCore, $\text{C}^{2}$ -CRS, and VRICR are conversational recommendation methods. We do not compare with conversational recommendation methods which use large pre-trained language models due to different task settings. For parameters used in baselines, we utilize the optimal values as reported in the respective papers.

### 4.2 Overall Performance (RQ1)

Table 4: Recommendation results on INSPIRED. Our proposed MCCRS method significantly outperforms SOTA baselines on the recommendation task on INSPIRED.

| Model | Recall@1 | Recall@10 | Recall@50 |
| --- | --- | --- | --- |
| Popularity | 0.032 | 0.155 | 0.323 |
| TextCNN | 0.025 | 0.119 | 0.245 |
| BERT | 0.044 | 0.179 | 0.328 |
| ReDial | 0.031 | 0.117 | 0.285 |
| KBRD | 0.058 | 0.146 | 0.207 |
| KGSF | 0.058 | 0.165 | 0.256 |
| RevCore | 0.068 | 0.198 | 0.379 |
| VRICR | 0.043 | 0.141 | 0.336 |
| $\text{C}^{2}$ -CRS | 0.090 | 0.242 | 0.399 |
| MCCRS | 0.104\* | 0.275\* | 0.497\* |

#### 4.2.1 Performance on Recommendation Task

In this section, we explore the effectiveness of our proposed method in comparison to SOTA baselines. We evaluate our recommendation performance against baselines on ReDial and INSPIRED, as shown in Table 3 and Table 4, respectively. For the evaluation metrics, the average performance across the maximum number of conversational turns is reported.

For the recommendation performances on ReDial in Table 3, we observe that our proposed MCCRS significantly outperforms all the baselines on all metrics on the ReDial dataset. MCCRS outperforms the best classical recommendation baseline BERT by 307%, 114%, and 148% in terms of Recall@1, Recall@10, and Recall@50, respectively. Compared with the SOTA conversational recommendation baselines $\text{C}^{2}$ -CRS and VRICR, MCCRS outperforms $\text{C}^{2}$ -CRS by 7.5%, 7.3%, and 16.2% while outperforms VRICR by 5.6%, 2.5%, and 16.5%, in terms of Recall@1, Recall@10, and Recall@50, respectively. Similarly, observe from Table 4 for the recommendation performances on INSPIRED, our proposed model, MCCRS, still significantly outperforms all the baselines on all metrics. MCCRS achieves a significant improvement of 136.4% on Recall@1, 53.6% on Recall@10, and 51.5% on Recall@50 over the best classical baseline BERT. Compared with the SOTA conversational recommendation baseline $\text{C}^{2}$ -CRS, MCCRS achieves a significant improvement of 15.6% on Recall@1, 13.6% on Recall@10, and 24.6% on Recall@50. The above observations indicate the effectiveness of our MCCRS model by incorporating multiple experts. MCCRS utilizes multi-type external information, including conversational history, knowledge graph, and reviews, which is helpful in understanding the conversation context. The effective Mixture-of-Experts framework to fuse the multi-type external information enhances the data representations and improves the recommendation performance. For conversational baselines, we find that KBRD, and KGSF perform better than ReDial, due to their use of external knowledge graphs, which aid in interpreting user intentions more effectively. RevCore achieves better recommendation performance than KBRD and KGSF. This might be because RevCore incorporates additional review information on items. $\text{C}^{2}$ -CRS outperforms RevCore, as it performs semantic fusing techniques with more external information.

Table 5: Response generation results on ReDial (automatic evaluation). Our proposed MCCRS method achieves better performance of response generation than baselines on the ReDial dataset.

| Model | Distinct-2 | Distinct-3 | Distinct-4 |
| --- | --- | --- | --- |
| Transformer | 0.148 | 0.151 | 0.137 |
| ReDial | 0.225 | 0.236 | 0.228 |
| KBRD | 0.263 | 0.368 | 0.423 |
| KGSF | 0.330 | 0.417 | 0.521 |
| RevCore | 0.424 | 0.558 | 0.612 |
| VRICR | 0.382 | 0.453 | 0.496 |
| $\text{C}^{2}$ -CRS | 0.631 | 0.932 | 0.909 |
| MCCRS | 0.680\* | 0.976\* | 0.981\* |

Table 6: Response generation results on INSPIRED (automatic evaluation). Our proposed MCCRS method achieves better performance of response generation than baselines on the INSPIRED dataset.

| Model | Distinct-2 | Distinct-3 | Distinct-4 |
| --- | --- | --- | --- |
| Transformer | 1.020 | 2.248 | 3.582 |
| ReDial | 1.347 | 1.521 | 3.445 |
| KBRD | 1.369 | 2.259 | 3.592 |
| KGSF | 1.608 | 2.719 | 4.929 |
| RevCore | 2.419 | 3.820 | 4.648 |
| VRICR | 1.937 | 3.248 | 4.965 |
| $\text{C}^{2}$ -CRS | 2.456 | 4.432 | 5.092 |
| MCCRS | 2.584\* | 4.579\* | 5.251\* |

#### 4.2.2 Performance on Conversation Task

In this subsection, we verify the performance of our proposed model on the conversation task. Specifically, we report the evaluation metrics for both automatic and human evaluation.

##### Automatic Evaluation

Table 5 and Table 6 show the performance comparison of automatic evaluation on response generation, on ReDial and INSPIRED, respectively. Among baselines, we observe that ReDial, KBRD, and KGSF achieve better performance than Transformer on both ReDial and INSPIRED, as they apply pre-training or enhance word probability by leveraging items and item-related entities from external knowledge graphs. $\text{C}^{2}$ -CRS performs the best among baselines. One possible reason is that $\text{C}^{2}$ -CRS applies a semantic fusion approach and an instance weighting mechanism to improve the diversity of responses.

Compared with these baselines, we see that our MCCRS model outperforms all baselines in terms of automatic metrics (i.e., Distinct-2, Distinct-3, and Distinct-4) on both ReDial and INSPIRED datasets. This might be because MCCRS effectively leverages multi-type information to understand the context and against noisy information. The multi-type context-aware framework enhances the diversity of generated responses.

Table 7: Response generation results on ReDial (human evaluation). Our proposed MCCRS method generates more fluent and informative responses than baselines on the conversation task.

| Model | Fluency | Informativeness |
| --- | --- | --- |
| Transformer | 0.82 | 0.91 |
| ReDial | 1.25 | 1.09 |
| KBRD | 1.31 | 1.22 |
| KGSF | 1.53 | 1.32 |
| RevCore | 1.55 | 1.38 |
| VRICR | 1.52 | 1.34 |
| $\text{C}^{2}$ -CRS | 1.58 | 1.51 |
| MCCRS | 1.66\* | 1.59\* |

Table 8: Results of ablation study for recommendation. Removing any of the three experts lowers the recommendation performance.

| Model | Recall@1 | Recall@10 | Recall@50 |
| --- | --- | --- | --- |
| Conversation Only | 0.022 | 0.113 | 0.207 |
| Graph Only | 0.051 | 0.213 | 0.373 |
| Review Only | 0.021 | 0.093 | 0.369 |
| w/o conversation | 0.053 | 0.216 | 0.471 |
| w/o graph | 0.025 | 0.121 | 0.382 |
| w/o review | 0.052 | 0.228 | 0.472 |
| MCCRS | 0.057\* | 0.250\* | 0.473\* |

##### Human Evaluation

Table 7 shows the performance comparison of human evaluation on response generation. From the results, we find that ReDial outperforms the typical Transformer by incorporating a pre-trained RNN encoder. $\text{C}^{2}$ -CRS performs the best among baselines on the reported human evaluation metrics. This might be because $\text{C}^{2}$ -CRS facilitates semantic fusion of data by using contrastive learning, which improves the responses.

Compared with these baselines, we observe that our MCCRS method performs better than all baselines in terms of the reported human evaluation metrics, i.e., fluency and informativeness. This demonstrates that our MCCRS model can generate more informative words or entities while maintaining the fluency of the generated responses. This might be because our MCCRS model incorporates multi-type external information to generate fluent responses. The high accuracy of the recommender module leads to generating high-quality items, which facilitates the informativeness of responses.

### 4.3 Ablation Study (RQ2)

We explore the contributions of different components within our model in this section. We perform an ablation study on the ReDial dataset, by considering six variants of the model, including (1) “w/o conversation” removes the conversation expert from the framework; (2) “w/o graph” removes the graph expert; (3) “w/o review” removes the review expert; (4) “conversation only” only involves the single conversation expert; (5) “graph only” only involves the single graph expert; (6) “review only” only involves the single review expert. Observe from Table 8, among the three experts, “graph only” achieves the highest performance than the other two single experts. Also, we find that removing any of the experts, either the conversation expert, graph expert, or review expert, results in lower performance. This indicates that all three experts contributed to the final performance. After removing the graph expert, the performance drops more than the other two experts. This indicates the graph expert contributes the most among the three experts, highlighting the importance of the graph expert. This might be because the graph expert benefits from the structure of the knowledge graph. Moreover, although we observe that “review only” and “conversation only” achieve relatively low performance, removing the conversation expert or review expert results in a decrease in performance. This suggests that adding conversations or reviews will marginally improve the recommendation accuracy. The reviews, although did not play a major role, do help with the recommendation task even if reviews are missing for some items.

### 4.4 Parameter Sensitivity (RQ3)

In this section, we explore how our proposed model is affected by its main parameters, including the mask proportion and hidden dimensionality, on the ReDial dataset.

#### 4.4.1 Mask Probability

When modeling the conversation, we apply a Cloze task to randomly mask a portion of the sequences. We examine the impact of mask probability [^2] on model performance. Observe from Table 9, performance initially improves as the mask proportion increases, but then declines at higher mask proportions. We conclude that mask proportion indeed affects performance. The optimal mask proportion is 0.4.

#### 4.4.2 Hidden Dimensionality

Table 10 illustrates the effect of hidden dimensionality (embedding size) on model performance. Overall, the recommendation performance generally declines as dimensionality increases from 32 to 256. This is probably due to overfitting. The proposed model achieves its best performance with a hidden dimension of 32.

Table 9: Effect of Mask Probability.

| Mask Probability | Recall@1 | Recall@10 | Recall@50 |
| --- | --- | --- | --- |
| 0.2 | 0.044 | 0.211 | 0.419 |
| 0.4 | 0.057 | 0.250 | 0.473 |
| 0.6 | 0.041 | 0.209 | 0.413 |
| 0.8 | 0.047 | 0.220 | 0.433 |

Table 10: Effect of hidden dimensionality.

| Hidden Dimensionality | Recall@1 | Recall@10 | Recall@50 |
| --- | --- | --- | --- |
| 32 | 0.057 | 0.250 | 0.473 |
| 64 | 0.043 | 0.215 | 0.379 |
| 128 | 0.046 | 0.220 | 0.444 |
| 256 | 0.038 | 0.174 | 0.335 |

## 5 Conclusion and Future Work

In this paper, we proposed the MCCRS model for conversational recommender systems. The MCCRS model deploys a mixture-of-experts structure to leverage multi-type data, including the structured knowledge graph, unstructured conversation history, and unstructured item reviews. It consists of three experts, with each expert specialized in a particular type of contextual data. Multiple experts are then coordinated by a ChairBot by considering the importance of each expert to generate recommendations. The embeddings from the three experts are introduced to the response generator as cross-attention embeddings to generate responses. The experimental results on two widely used datasets demonstrate that our MCCRS model outperforms the SOTA conversational recommendation baselines and is highly effective.

One limitation of this work is that we use entities from the DBpedia knowledge graph following previous work of conversational recommender system [^3] [^5] [^4] [^6], the extracted entities may not be 100% accurate. Therefore, it is valuable to explore the conversational recommender system by integrating and modeling the uncertainty and noise in conversation contexts. Also, we do not incorporate sentiment analysis of items, entities, or reviews in the multi-type contextual data. In future work, we plan to explore conversational recommender systems by integrating and modeling the sentiment information (e.g., positive or negative) from the conversation and reviews in future work.

Also, for each conversation, we utilize all the data from multi-type resources. Although the experimental results validate its effectiveness, it is worth detecting the consistency of different types of data sources in conversational recommender systems, enabling the conversational recommendation model to predict the necessity for integrating each type of contextual data and model the inconsistency of different types of contextual data. Furthermore, we model the conversation by utilizing the full sequence. However, the inherent complexity of conversations, exemplified by scenarios such as multi-topic discussions where users seamlessly transition between different subjects, poses a challenge as these conversations may lack strict sequential dependencies within the full sequence. A potential research direction is to partition the entire conversation sequence into discrete subsequences (e.g., detecting topic threads in multi-topic conversations [^7] and modeling each topic as a subsequence) so that more strict sequential dependencies remain in subsequences. One can also model the noise in the sequence to relax the strict sequential dependencies.

Last, while MCCRS achieves strong performance in benchmark datasets and effectively fuses multi-type contextual information, its mixture-of-experts structure may lead to increased computational complexity, which may pose challenges in real-world deployment scenarios when involving large-scale or real-time recommendation tasks. In future work, we plan to explore potential techniques, such as knowledge distillation for model compression [^72] and expert pruning, for improving model efficiency and scalability. Also, for the mixture-of-experts structure, we combine the results of multiple experts with a followed ChairBot. One can also use other combination strategies of multiple experts, e.g., attention-based gating mechanisms, or a hierarchical structure that sequentially refines expert outputs.

## 6 Acknowledgement

We would like to thank Qingtian Cao for her contribution to the early exploration and discussion of this work. This research was supported by the National Natural Science Foundation of China (62402093) and, the Sichuan Science and Technology Program (2025ZNSFSC0479). This work was also supported in part by the National Natural Science Foundation of China under grants U20B2063 and 62220106008, and the Sichuan Science and Technology Program under Grant 2024NSFTD0034.

## References

[^1]: C. Gao, W. Lei, X. He, M. de Rijke, T.-S. Chua, Advances and challenges in conversational recommender systems: A survey, AI Open 2 (2021) 100–126.

[^2]: J. Zou, E. Kanoulas, P. Ren, Z. Ren, A. Sun, C. Long, Improving conversational recommender systems via transformer-based sequential modelling, in: Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2022, pp. 2319–2324.

[^3]: Q. Chen, J. Lin, Y. Zhang, M. Ding, Y. Cen, H. Yang, J. Tang, Towards knowledge-based recommender dialog system, in: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 2019, pp. 1803–1813.

[^4]: K. Zhou, W. X. Zhao, S. Bian, Y. Zhou, J.-R. Wen, J. Yu, Improving conversational recommender systems via knowledge graph based semantic fusion, in: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2020, pp. 1006–1014.

[^5]: R. Sarkar, K. Goswami, M. Arcan, J. P. McCrae, Suggest me a movie for tonight: Leveraging knowledge graphs for conversational recommendation, in: Proceedings of the 28th International Conference on Computational Linguistics, 2020, pp. 4179–4189.

[^6]: W. Ma, R. Takanobu, M. Huang, CR-walker: Tree-structured graph reasoning and dialog acts for conversational recommendation, in: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2021, pp. 1839–1851.

[^7]: K. Zhou, Y. Zhou, W. X. Zhao, X. Wang, J.-R. Wen, Towards topic-guided conversational recommender system, in: Proceedings of the 28th International Conference on Computational Linguistics, 2020, pp. 4128–4139.

[^8]: X. Wang, Y. Li, J. Liu, Conversation recommender system based on knowledge graph and time-series feature, in: The 5th International Conference on Computer Science and Application Engineering, 2021, pp. 1–5.

[^9]: L. Wang, H. Hu, L. Sha, C. Xu, D. Jiang, K.-F. Wong, Recindial: A unified framework for conversational recommendation with pretrained language models, in: Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing, 2022, pp. 489–500.

[^10]: K. Chen, S. Sun, Knowledge-based conversational recommender systems enhanced by dialogue policy learning, in: The 10th International Joint Conference on Knowledge Graphs, 2021, pp. 10–18.

[^11]: B. Yang, C. Han, Y. Li, L. Zuo, Z. Yu, Improving conversational recommendation systems’ quality with context-aware item meta information, arXiv preprint arXiv:2112.08140 (2021).

[^12]: T. Zhang, Y. Liu, P. Zhong, C. Zhang, H. Wang, C. Miao, Kecrs: Towards knowledge-enriched conversational recommendation system, arXiv preprint arXiv:2105.08261 (2021).

[^13]: Y. Lu, J. Bao, Y. Song, Z. Ma, S. Cui, Y. Wu, X. He, Revcore: Review-augmented conversational recommendation, arXiv preprint arXiv:2106.00957 (2021).

[^14]: Y. Zhou, K. Zhou, W. X. Zhao, C. Wang, P. Jiang, H. Hu, C <sup>2</sup> -crs: Coarse-to-fine contrastive learning for conversational recommender system, in: Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining, 2022, pp. 1488–1496.

[^15]: Z. Ren, Z. Tian, D. Li, P. Ren, L. Yang, X. Xin, H. Liang, M. de Rijke, Z. Chen, Variational reasoning about user preferences for conversational recommendation, in: Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2022, pp. 165–175.

[^16]: H. Won, B. Oh, H. Yang, K.-H. Lee, Cross-modal contrastive learning for aspect-based recommendation, Information Fusion 99 (2023) 101858.

[^17]: D. Jannach, A. Manzoor, W. Cai, L. Chen, A survey on conversational recommender systems, ACM Computing Surveys (CSUR) 54 (2021) 1–36.

[^18]: T. Shen, J. Li, M. R. Bouadjenek, Z. Mai, S. Sanner, Towards understanding and mitigating unintended biases in language model-driven conversational recommendation, Information Processing & Management 60 (2023) 103139.

[^19]: S. Li, R. Xie, Y. Zhu, F. Zhuang, Z. Tang, W. X. Zhao, Q. He, Self-supervised learning for conversational recommendation, Information Processing & Management 59 (2022) 103067.

[^20]: X. Zhang, X. Jia, H. Liu, X. Liu, X. Zhang, A goal interaction graph planning framework for conversational recommendation, in: Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 2024, pp. 19578–19587.

[^21]: D. He, J. Zhang, X. Wang, M. Ge, Z. Feng, L. Wang, X. Ma, Tut4crs: Time-aware user-preference tracking for conversational recommendation system, in: Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 5856–5864.

[^22]: J. Zou, A. Sun, C. Long, E. Kanoulas, Knowledge-enhanced conversational recommendation via transformer-based sequential modeling, ACM Transactions on Information Systems 42 (2024) 1–27.

[^23]: J. Zou, Y. Chen, E. Kanoulas, Towards question-based recommender systems, in: Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’20, 2020, pp. 881–890.

[^24]: X. Zhang, R. Xie, Y. Lyu, X. Xin, P. Ren, M. Liang, B. Zhang, Z. Kang, M. de Rijke, Z. Ren, Towards empathetic conversational recommender systems, in: Proceedings of the 18th ACM Conference on Recommender Systems, 2024, pp. 84–93.

[^25]: Y. Zhang, X. Chen, Q. Ai, L. Yang, W. B. Croft, Towards conversational search and recommendation: System ask, user respond, in: Proceedings of the 27th acm international conference on information and knowledge management, 2018, pp. 177–186.

[^26]: Y. Sun, Y. Zhang, Conversational recommender system, in: The 41st international acm sigir conference on research & development in information retrieval, 2018, pp. 235–244.

[^27]: K. Christakopoulou, A. Beutel, R. Li, S. Jain, E. H. Chi, Q&r: A two-stage approach toward interactive recommendation, in: Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2018, pp. 139–148.

[^28]: W. Lei, X. He, Y. Miao, Q. Wu, R. Hong, M.-Y. Kan, T.-S. Chua, Estimation-action-reflection: Towards deep interaction between conversational and recommender systems, in: Proceedings of the 13th International Conference on Web Search and Data Mining, 2020a, pp. 304–312.

[^29]: W. Lei, G. Zhang, X. He, Y. Miao, X. Wang, L. Chen, T.-S. Chua, Interactive path reasoning on graph for conversational recommendation, in: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’20, 2020b, pp. 2073–2083.

[^30]: C. Hu, S. Huang, Y. Zhang, Y. Liu, Learning to infer user implicit preference in conversational recommendation, in: Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2022, pp. 256–266.

[^31]: Y. Zhang, L. Wu, Q. Shen, Y. Pang, Z. Wei, F. Xu, B. Long, J. Pei, Multiple choice questions based multi-interest policy learning for conversational recommendation, in: Proceedings of the ACM Web Conference 2022, 2022, pp. 2153–2162.

[^32]: K. Christakopoulou, F. Radlinski, K. Hofmann, Towards conversational recommender systems, in: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, ACM, 2016, pp. 815–824.

[^33]: S. Li, W. Lei, Q. Wu, X. He, P. Jiang, T.-S. Chua, Seamlessly unifying attributes and items: Conversational recommendation for cold-start users, ACM Transactions on Information Systems (TOIS) 39 (2021) 1–29.

[^34]: X. Dai, Z. Wang, J. Xie, X. Liu, J. C. Lui, Conversational recommendation with online learning and clustering on misspecified users, IEEE Transactions on Knowledge and Data Engineering (2024).

[^35]: J. Zou, J. Huang, Z. Ren, E. Kanoulas, Learning to ask: Conversational product search via representation learning, ACM Transactions on Information Systems 41 (2022) 1–27.

[^36]: Z. Liu, H. Wang, Z.-Y. Niu, H. Wu, W. Che, T. Liu, Towards conversational recommendation over multi-type dialogs, in: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020, pp. 1036–1049.

[^37]: L. Liao, R. Takanobu, Y. Ma, X. Yang, M. Huang, T.-S. Chua, Deep conversational recommender in travel, arXiv preprint arXiv:1907.00710 (2019).

[^38]: Z. Liang, H. Hu, C. Xu, J. Miao, Y. He, Y. Chen, X. Geng, F. Liang, D. Jiang, Learning neural templates for recommender dialogue system, in: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2021, pp. 7821–7833.

[^39]: Y. Deng, W. Zhang, W. Xu, W. Lei, T.-S. Chua, W. Lam, A unified multi-task learning framework for multi-goal conversational recommender systems, ACM Transactions on Information Systems (2022).

[^40]: Y. Pan, Y. Yin, F. Huang, Keyword-guided topic-oriented conversational recommender system, in: 2022 International Joint Conference on Neural Networks (IJCNN), 2022, pp. 1–8.

[^41]: R. Li, S. E. Kahou, H. Schulz, V. Michalski, L. Charlin, C. Pal, Towards deep conversational recommendations, in: Advances in Neural Information Processing Systems, 2018, pp. 9748–9758.

[^42]: S. A. Hayati, D. Kang, Q. Zhu, W. Shi, Z. Yu, Inspired: Toward sociable recommendation dialog systems, in: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020, pp. 8142–8152.

[^43]: J. Zhang, Y. Yang, C. Chen, L. He, Z. Yu, Kers: A knowledge-enhanced framework for recommendation dialog systems with multiple subgoals, in: Findings of the Association for Computational Linguistics: EMNLP 2021, 2021, pp. 1092–1101.

[^44]: X. Ren, T. Chen, Q. V. H. Nguyen, L. Cui, Z. Huang, H. Yin, Explicit knowledge graph reasoning for conversational recommendation, ACM Transactions on Intelligent Systems and Technology 15 (2024) 1–21.

[^45]: W. Li, W. Wei, X. Qu, X.-L. Mao, Y. Yuan, W. Xie, D. Chen, Trea: Tree-structure reasoning schema for conversational recommendation, in: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2023, pp. 2970–2982.

[^46]: Z. Qiu, Y. Tao, S. Pan, A. W.-C. Liew, Knowledge graphs and pretrained language models enhanced representation learning for conversational recommender systems, IEEE Transactions on Neural Networks and Learning Systems (2024).

[^47]: C. Bizer, J. Lehmann, G. Kobilarov, S. Auer, C. Becker, R. Cyganiak, S. Hellmann, Dbpedia-a crystallization point for the web of data, Journal of web semantics 7 (2009) 154–165.

[^48]: J. Zhu, C. Huang, P. De Meo, Dfmke: A dual fusion multi-modal knowledge graph embedding framework for entity alignment, Information Fusion 90 (2023) 111–119.

[^49]: R. Speer, J. Chin, C. Havasi, Conceptnet 5.5: an open multilingual graph of general knowledge, in: Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence, 2017, pp. 4444–4451.

[^50]: J. Zhou, B. Wang, R. He, Y. Hou, Crfr: Improving conversational recommender systems via flexible fragments reasoning on knowledge graphs, in: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2021, pp. 4324–4334.

[^51]: X. Zhang, X. Xin, D. Li, W. Liu, P. Ren, Z. Chen, J. Ma, Z. Ren, Variational reasoning over incomplete knowledge graphs for conversational recommendation, in: Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining, 2023, pp. 231–239.

[^52]: X. Wang, K. Zhou, J.-R. Wen, W. X. Zhao, Towards unified conversational recommender systems via knowledge-enhanced prompt learning, in: Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2022, pp. 1929–1937.

[^53]: X. Wang, X. Tang, W. X. Zhao, J. Wang, J.-R. Wen, Rethinking the evaluation for conversational recommendation in the era of large language models, in: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 10052–10065.

[^54]: Y. Xi, W. Liu, J. Lin, B. Chen, R. Tang, W. Zhang, Y. Yu, Memocrs: Memory-enhanced sequential conversational recommender systems with large language models, arXiv preprint arXiv:2407.04960 (2024).

[^55]: K. D. Spurlock, C. Acun, E. Saka, O. Nasraoui, Chatgpt for conversational recommendation: Refining recommendations by reprompting with feedback, arXiv preprint arXiv:2401.03605 (2024).

[^56]: C. Zhang, X. Huang, J. An, S. Zou, Improving conversational recommender systems via multi-preference modelling and knowledge-enhanced, Knowledge-Based Systems 286 (2024) 111361.

[^57]: H. Dao, Y. Deng, D. D. Le, L. Liao, Broadening the view: Demonstration-augmented prompt learning for conversational recommendation, in: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2024, pp. 785–795.

[^58]: S. Kemper, J. Cui, K. Dicarlantonio, K. Lin, D. Tang, A. Korikov, S. Sanner, Retrieval-augmented conversational recommendation with prompt-based semi-structured natural language state tracking, in: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2024, pp. 2786–2790.

[^59]: G. An, J. Zou, J. Wei, C. Zhang, F. Sun, Y. Yang, Beyond whole dialogue modeling: Contextual disentanglement for conversational recommendation, arXiv preprint arXiv:2504.17427 (2025).

[^60]: Y. Wei, J. Zou, W. Guo, G. Wang, X. Xu, Y. Yang, Mscrs: Multi-modal semantic graph prompt learning framework for conversational recommender systems, arXiv preprint arXiv:2504.10921 (2025).

[^61]: S. Li, R. Xie, Y. Zhu, X. Ao, F. Zhuang, Q. He, User-centric conversational recommendation with multi-aspect user modeling, in: Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2022, pp. 223–233.

[^62]: J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, Bert: Pre-training of deep bidirectional transformers for language understanding, arXiv preprint arXiv:1810.04805 (2018).

[^63]: W. L. Taylor, “cloze procedure”: A new tool for measuring readability, Journalism quarterly 30 (1953) 415–433.

[^64]: D. Hendrycks, K. Gimpel, Bridging nonlinearities and stochastic regularizers with gaussian error linear units, CoRR, abs/1606.08415 3 (2016).

[^65]: J. Lehmann, R. Isele, M. Jakob, A. Jentzsch, D. Kontokostas, P. N. Mendes, S. Hellmann, M. Morsey, P. Van Kleef, S. Auer, et al., Dbpedia–a large-scale, multilingual knowledge base extracted from wikipedia, Semantic web 6 (2015) 167–195.

[^66]: P. Ferragina, U. Scaiella, Tagme: on-the-fly annotation of short text fragments (by wikipedia entities), in: Proceedings of the 19th ACM international conference on Information and knowledge management, 2010, pp. 1625–1628.

[^67]: M. Schlichtkrull, T. N. Kipf, P. Bloem, R. Van Den Berg, I. Titov, M. Welling, Modeling relational data with graph convolutional networks, in: The Semantic Web: 15th International Conference, ESWC 2018, Heraklion, Crete, Greece, June 3–7, 2018, Proceedings 15, 2018, pp. 593–607.

[^68]: J. Pei, P. Ren, M. de Rijke, A modular task-oriented dialogue system using a neural mixture-of-experts (2019). [arXiv:1907.05346](http://arxiv.org/abs/1907.05346).

[^69]: D. P. Kingma, J. Ba, Adam: A method for stochastic optimization, arXiv preprint arXiv:1412.6980 (2014).

[^70]: Y. Kim, Convolutional neural networks for sentence classification, arXiv preprint arXiv:1408.5882 (2014).

[^71]: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin, Attention is all you need, in: Advances in neural information processing systems, 2017, pp. 5998–6008.

[^72]: C. Lin, H. Hu, J. Zou, L. Li, J. Liu, Y. Gao, Y. Yang, H. T. Shen, Distilling grounding dino for an edge-cloud collaborative advanced driver assistance system, IEEE Transactions on Circuits and Systems for Video Technology (2025) 1–1. doi:[10.1109/TCSVT.2025.3586704](http://dx.doi.org/10.1109/TCSVT.2025.3586704).