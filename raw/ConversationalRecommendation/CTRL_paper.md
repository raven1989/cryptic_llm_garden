---
title: "CTRL: Connect Collaborative and Language Model for CTR Prediction"
source: "https://arxiv.org/html/2306.02841v4"
author:
published:
created: 2026-08-08
description:
tags:
  - "clippings"
---
Xiangyang Li [lixiangyang34@huawei.com](mailto:lixiangyang34@huawei.com) ChinaHuawei Noah’s Ark Lab, Bo Chen [chenbo116@huawei.com](mailto:chenbo116@huawei.com) ChinaHuawei Noah’s Ark Lab, Lu Hou [houlu3@huawei.com](mailto:houlu3@huawei.com) ChinaHuawei Noah’s Ark Lab and Ruiming Tang [tangruiming@huawei.com](mailto:tangruiming@huawei.com) ChinaHuawei Noah’s Ark Lab

###### Abstract.

Traditional click-through rate (CTR) prediction models convert the tabular data into one-hot vectors and leverage the collaborative relations among features for inferring the user’s preference over items. This modeling paradigm discards essential semantic information. Though some works like P5 and CTR-BERT have explored the potential of using Pre-trained Language Models (PLMs) to extract semantic signals for CTR prediction, they are computationally expensive and suffer from low efficiency. Besides, the beneficial collaborative relations are not considered, hindering the recommendation performance. To solve these problems, in this paper, we propose a novel framework CTRL, which is industrial-friendly and model-agnostic with superior inference efficiency. Specifically, the original tabular data is first converted into textual data. Both tabular data and converted textual data are regarded as two different modalities and are separately fed into the collaborative CTR model and pre-trained language model. A cross-modal knowledge alignment procedure is performed to fine-grained align and integrate the collaborative and semantic signals, and the lightweight collaborative model can be deployed online for efficient serving after fine-tuned with supervised signals. Experimental results on three public datasets show that CTRL outperforms the state-of-the-art (SOTA) CTR models significantly. Moreover, we further verify its effectiveness on a large-scale industrial recommender system.

## 1\. Introduction

Click-through rate (CTR) prediction is an important task for recommender systems and online advertising [^16] [^41], where users’ willingness to click on items is predicted based on historical behavior data. The estimated CTR is leveraged to determine whether an item can be displayed to the user. Consequently, accurate CTR prediction service is critical to improving user experience, product sales, and advertising platform revenue [^64].

For the CTR prediction task, historical data is organized in the form of tabular data. During the evolution of recommendation models, from the early Matrix Factorization (MF) [^30], to shallow machine learning era models like Logistic Regression (LR) [^8] and Factorization Machine (FM) [^49], and continuing to the deep neural models such as DeepFM [^18] and DIN [^66], collaborative signals have always been the core of recommendation modeling, which leverages the feature co-occurrences and label signals for inferring user preferences. After encoding the tabular features into one-hot features [^21], the co-occurrence relations (i.e., interactions) of the features are captured by various human-designed operations (e.g., inner product [^46] [^18], outer product [^56] [^35], non-linear layer [^63] [^7], etc.). By modeling these collaborative signals explicitly or implicitly, the relevance between users and items can be inferred.

![Refer to caption](https://arxiv.org/html/2306.02841v4/x1.png)

Figure 1. The external world knowledge and reasoning capabilities of pre-trained language models facilitate recommendations.

However, the collaborative-based modeling paradigm discards the semantic information among the original features due to the one-hot feature encoding process. Therefore, for cold-start scenarios or low-frequency long-tailed features, the recommendation performance is unsatisfactory, limited by the inadequate collaborative relations [^40]. For example, in Figure 7, when inferring the click probability of user John over a cold start movie World War III, the inadequate collaborative signals in historical data may impede accuracy recommendation. Recently, some works are proposed to address this drawback by involving Pre-trained Language Models (PLMs) to model semantic signals, such as P5 [^15], M6-Rec [^9], CTR-BERT [^43], TALLRec [^2], PALR [^6]. These works feed the original textual features directly into the language models for recommendation, rather than using one-hot encoded features. On the one hand, the linguistic and semantic knowledge in PLMs helps to extract the semantic information within the original textual features [^36]. On the other hand, the external world knowledge such as the director, actors, even story plot and reviews for the movie World War III, as well as knowledge reasoning capability in Large Language Models (LLMs) provide general knowledge beyond training data and scenarios [^65], thus enlightening a new technological path for recommender systems.

Although remarkable progress has been achieved, the existing semantic-based solutions suffer from several shortcomings: 1) Making predictions based on semantics merely without traditional collaborative modeling can be suboptimal [^15] because the feature co-occurrence patterns and user-item interactions are indispensable indicators for personalized recommendation [^18], which are not yet well equipped for PLMs [^65] [^37]. 2) Online inferences of language models are computationally expensive due to their complex structures. To adhere to low-latency constraints, massive computational resources, and engineering optimizations are involved, hindering large-scale industrial applications [^9] [^15].

Therefore, incorporating PLMs into recommendation systems to capture semantic signals confronts two major challenges:

- How to combine the collaborative signals with semantic signals to boost the performance of recommendation?
- How to ensure efficient online inference without involving extensive engineering optimizations?

To solve these two challenges above, inspired by the recent works in contrastive learning, we propose a novel framework to Connect Collaborative and Language Model (CTRL) for CTR prediction, which consists of two stages: Cross-modal Knowledge Alignment stage, and Supervised Fine-tuning stage. Specifically, the raw tabular data is first converted into textual data by human-designed prompts, which can be understood by language models. Then, the original tabular data and generative textual data are regarded as different modalities and fed into the collaborative CTR model and pre-trained language model, respectively. We execute a cross-modal knowledge alignment procedure, meticulously aligning and integrating collaborative signals with semantic signals. Finally, the collaborative CTR model is fine-tuned on the downstream task with supervised signals. During the online inference, only the lightweight fine-tuned CTR model is pushed for serving without the language model, thus ensuring efficient inference.

Our main contributions are summarized as follows:

- We first propose a novel training framework CTRL, which is capable of aligning signals from collaborative and language models, introducing semantic knowledge into the collaborative models.
- Through extensive experiments, we demonstrate that the incorporation of semantic knowledge significantly enhances the performance of collaborative models on CTR task.
- CTRL is industrial-friendly, model-agnostic, and can adapt with any collaborative models and PLMs, including LLMs. Moreover, the high inference efficiency is also retained, facilitating its application in industrial scenarios.
- In experiments conducted on three publicly available datasets from real-world industrial scenarios, CTRL achieved SOTA performance. Moreover, we further verify its effectiveness on large-scale industry recommender systems.

## 2\. Related Work

### 2.1. Collaborative Models for Recommendation

During the evolution of recommendation models, from the early matrix factorization (MF) [^30], to shallow machine learning era models like Logistic Regression (LR) [^8] and Factorization Machine (FM) [^49], to the deep neural models [^18] [^66], collaborative signals have always been the core of recommendation modeling. These collaborative-based models convert the tabular features into one-hot features and leverage various interaction functions to extract feature co-occurrence relations (a.k.a. feature interactions).

Different human-designed interaction functions are proposed to improve the modeling ability of collaborative signals. Wide&Deep [^7] uses the non-linear layers to extract implicit high-order interactions. DeepFM [^18] leverages the inner product to capture pairwise interactions with stacked and parallel structures. DCN [^55] and EDCN [^4] deploy cross layers to model bit-wise feature interactions.

Though collaborative-based models have achieved significant progress, they cannot capture the semantic information of the original features, thereby hindering the prediction effect in some scenarios such as cold-start or low-frequency long-tailed features.

### 2.2. Semantic Models for Recommendation

Transformer-based language models, such as BERT [^10], GPT-3 [^3], and T5 [^48], have emerged as foundational architectures in the realm of Natural Language Processing (NLP). Their dominance across various NLP subdomains, such as text classification [^42] [^33], sentiment analysis [^57] [^22], intelligent dialogue [^15] [^45], and style transfer [^24] [^32], is primarily attributed to their robust capabilities for knowledge reasoning and transfer. Nevertheless, since recommender systems mainly employ tabular data, which is heterogeneous with text data, it is difficult to apply the language model straightforwardly to the recommendation task.

In recent times, innovative research trends have surfaced, exploring the viability of language models in recommendation tasks. P5 [^15], serves as a generative model tailored for recommendations, underpinning all downstream recommendation tasks into a text generation task and utilizing the T5 [^48] model for training and prediction. P-Tab [^36] introduces a recommendation methodology based on discriminative language models, translating tabular data into prompts, pre-training these prompts with a Masked Language Model objective, and finally fine-tuning on downstream tasks. Concurrently, Amazon’s CTR-BERT [^43], a two-tower structure comprising two BERT models, encodes user and item text information respectively. More recently, a considerable upsurge in scholarly works has been observed, leveraging Large Language Models (LLMs) for recommendation systems [^52] [^23] [^2] [^61] [^62]. For instance, a study by Baidu [^52] investigates the possibility of using LLM for re-ranking within a search context. Similarly, RecLLM [^61] addresses the issue of fairness in the application of LLMs within recommendation systems. However, although the above semantic-based recommendation models have exposed the possibility of application in recommender systems, they have two fatal drawbacks: 1) Discarding the superior experience accumulation in collaborative modeling presented in Section 2.1 and making prediction with semantics only may be suboptimal [^15] and hinder the performance for cold-start scenarios or low-frequency long-tailed features. 2) Due to the huge number of parameters of the language models, it is quite arduous for language models to meet the low latency requirements of recommender systems, making online deployment much more challenging. Instead, our proposed CTRL overcomes these two shortcomings by combining both collaborative and semantic signals via two-stage training paradigm.

## 3\. preliminary

In this section, we present the collaborative-based deep CTR model and reveal the deficiencies in modeling semantic information. The CTR prediction is a supervised binary classification task, whose dataset consists of several instances $(\mathbf{x},y)$. Label $y\in\{0,1\}$ indicates user’s actual click action. Feature $\mathbf{x}$ is multi-fields that contains important information about the relations between users and items, including user profiles (e.g., gender, occupation), item features (e.g., category, price) as well as contextual information (e.g., time, location) [^17]. Based on the instances, the traditional deep CTR models leverage the collaborative signals to estimate the probability $P(y=1|\mathbf{x})$ for each instance.

The existing collaborative-based CTR models first encode the tabular features into one-hot features and then model the feature co-occurrence relations by various human-designed operations. Specifically, the multi-field tabular features are transformed into the high-dimensional sparse features via field-wise one-hot encoding [^21]. For example, the feature (Gender=Female, Occupation=Doctor, Genre=Sci-Fi, …, City=Hong Kong) of an instance can be represented as a one-hot vector:

$$
\small\mathbf{x}=\underbrace{[0,1]}_{\text{Gender}}\underbrace{[0,0,1,\dots,0]%
}_{\text{Occupation}}\underbrace{[0,1,0,\dots,0]}_{\text{Genre}}\dots%
\underbrace{[0,0,1,\dots,0]}_{\text{City}}.
$$

Generally, deep CTR models follow an “Embedding & Feature interaction” paradigm [^4] [^17]. The high-dimensional sparse one-hot vector is mapped into a low-dimensional dense space via an embedding layer with an embedding look-up operation. Specifically, for the $i$ -th feature, the corresponding feature embedding $\mathbf{e}_{i}$ can be obtained via $\mathbf{e}_{i}=\mathbf{E}_{i}\mathbf{x}_{i}$, where $\mathbf{E}_{i}$ is the embedding matrix. Following, feature interaction layers are proposed to capture the explicit or implicit feature co-occurrence relations. Massive effort has been made in designing specific interaction functions, such as product [^46] [^18], cross layer [^55] [^4] [^34], non-linear layer [^63] [^7], and attention layer [^66]. Finally, the predictive CTR score $\hat{y}$ is obtained via an output layer and optimized with the ground-truth label $y$ through the widely-used Binary Cross Entropy (BCE).

As we can observe, collaborative-based CTR models leverage the one-hot encoding to convert the original tabular data into one-hot vectors as E.q.(1), discarding the semantic information among the feature fields and values [^17]. By doing this, the feature semantics is lost and the only signals that can be used for prediction are the feature co-occurrence relations, which is suboptimal when the relations are weak in some scenarios such as cold-start or low-frequency long-tailed features. Therefore, introducing the language model to capture the essential semantic information is conducive to compensating for the information gaps and improving performance.

## 4\. METHOD

As depicted in Figure 3, the proposed CTRL is a two-stage training paradigm. The first stage is Cross-modal Knowledge Alignment, which feeds paired tabular data and textual data from two modalities into the collaborative model and the language model respectively, and then aligns them with the contrastive learning objective. The second stage is the Supervised Fine-tuning stage, where the collaborative model is fine-tuned on the downstream task with supervised signals.

![Refer to caption](https://arxiv.org/html/2306.02841v4/x2.png)

Figure 2. The overall process of prompt construction.

### 4.1. Prompt Construction

![Refer to caption](https://arxiv.org/html/2306.02841v4/x3.png)

Figure 3. An intuitive illustration of the CTRL, which is a two-stage framework, in the first stage, cross-modal contrastive learning is used to fine-grained align knowledge of the two modalities. In the second stage, the lightweight collaborative model is fine-tuned on downstream tasks. The red square represents a positive pair in the batch, while the green square represents a negative pair.

Before introducing the two-stage training paradigm, we first present the prompt construction process. As illustrated in Figure 2, to obtain textual prompt data, we design prompt templates to transform the tabular data into textual data for each training instance. As mentioned in previous work [^15] [^9], a proper prompt should contain sufficient semantic information about the user and the item. For example, user’s profiles such as age, identity, interests, and behaviors can be summarized in a single sentence. Besides, item’s description sentence can be organized with the features such as color, quality, and shape. For this purpose, we design the following template to construct the prompts:

> This is a user, gender is female, age is 18, occupation is doctor, who has recently watched Titanic|Avatar. This is a movie, title is The Terminator, genre is Sci-FI, director is Camelon.

In this prompt, the first sentence “This is a user, gender is female, age is 18, occupation is doctor, who has recently watched Titanic|Avatar.” describes the user-side features, including his/her profiles such as age, gender, occupation, and historical behaviors, etc. The following sentence “This is a movie, title is The Terminator, genre is Sci-FI, director is Camelon.” describes the item-side features such as title, category, director, etc. In the practical implementation, we use the period “.” to separate the user-side and item-side descriptions, the comma “,” to separate each feature, and vertical bar “|” to separate each user’s historical behavior <sup>2</sup>. We also explore the effect of different prompts, of which results are presented in Section 5.6.2.

### 4.2. Cross-modal Knowledge Alignment

As mentioned before, existing collaborative-based recommendation models [^55] [^50] leverage the feature co-occurrence relations to infer users’ preferences over items, facilitating the evolution of recommendations. Besides, the pre-trained language models [^10] specializes in capturing the semantic signals of recommendation scenarios with the linguistic and external world knowledge [^15]. In order to combine the modeling capabilities of both collaborative-based models and pre-trained language models, as well as ensure efficient online inference, CTRL proposes an implicit information integration method via contrastive learning [^5] [^14], where cross-modal knowledge (i.e., tabular and textual information) between collaborative and semantic space is aligned.

#### 4.2.1. Cross-modal Contrastive Learning

The cross-modal contrastive procedure is presented in Figure 3. First, the collaborative model and semantic model (a.k.a., pre-trained language model) are utilized to encode the tabular and textual data for obtaining the corresponding representations, respectively. Specifically, let $\mathcal{M}_{col}$ denotes collaborative model, and $\mathcal{M}_{sem}$ denotes semantic model, for an instance $\mathbf{x}$, $\mathbf{x}^{tab}$ denotes the tabular form, and $\mathbf{x}^{text}$ denotes the textual form of the same instance that is obtained after the prompt construction process. The instance representations under collaborative and semantic space can be presented as $\mathcal{M}_{col}({\bf x}^{tab})$ and $\mathcal{M}_{sem}({\bf x}^{text})$, respectively. To convert the unequal-length representations into the same dimension, a linear projection layer is designed, and the transformed instance representations can be obtained as follows:

$$
\displaystyle\mathbf{h}^{tab}=\mathcal{M}_{col}({\bf x}^{tab}){\bf W}^{tab}+{%
\bf b}^{tab},
$$
 
$$
\displaystyle\mathbf{h}^{text}=\mathcal{M}_{sem}({\bf x}^{text}){\bf W}^{text}%
+{\bf b}^{text},
$$

where $\mathbf{h}^{tab}$ and $\mathbf{h}^{text}$ are the transformed collaborative and semantic representations for the same instance ${\bf x}$, ${\bf W}^{tab},{\bf W}^{text}$ and ${\bf b}^{tab},{\bf b}^{text}$ are the transform matrices and biases of the linear projection layers.

Then, the contrastive learning is used to align the instance representations under different latent spaces, which is proved effective in both unimodal [^5] [^14] and cross-modal [^47] representation learning. The assumption behind this is that, under a distance metric, the correlated representations should be constrained to be close, and vice versa should be far away. We employ InfoNCE [^19] to align two representations under collaborative and semantic space for each instance. As shown in Figure 3, two different modalities (textual, tabular) of the same sample form a positive pair. Conversely, data from two different modalities (textual and tabular) belonging to diverse samples form a negative pair. Negative pairs are obtained through in-batch sampling. Denote $\mathbf{h}^{text}_{k},\mathbf{h}^{tab}_{k}$ are the representations of two modals for the $k$ -th instance, the textual-to-tabular contrastive loss can be formulated as:

$$
\displaystyle\mathcal{L}^{textual2tabular}=-\frac{1}{N}\sum_{k=1}^{N}log\frac{%
exp(sim(\mathbf{h}^{text}_{k},\mathbf{h}^{tab}_{k})/\tau)}{\sum_{j=1}^{N}exp(%
sim(\mathbf{h}^{text}_{k},\mathbf{h}^{tab}_{j})/\tau)},
$$

where $\tau$ is a temperature coefficient and $N$ is the number of instances in a batch. Besides, function $sim(\cdot,\cdot)$ measures the similarity between two vectors. Typically, cosine similarity is employed for this purpose. In order to avoid spatial bias towards collaborative modal, motivated by the Jensen–Shannon (J-S) divergence [^12], we also design a tabular-to-textual contrastive loss for uniformly aligning into a multimodal space, which is shown as:

$$
\displaystyle\mathcal{L}^{tabular2textual}=-\frac{1}{N}\sum_{k=1}^{N}log\frac{%
exp(sim(\mathbf{h}^{tab}_{k},\mathbf{h}^{text}_{k})/\tau)}{\sum_{j=1}^{N}exp(%
sim(\mathbf{h}^{tab}_{k},\mathbf{h}^{text}_{j})/\tau)}.
$$

Finally, the cross-modal contrastive learning loss $\mathcal{L}_{ccl}$ is defined as the average of $\mathcal{L}^{textual2tabular}$ and $\mathcal{L}^{tabular2textual}$, and all the parameters including collaborative model $\mathcal{M}_{col}$ and semantic model $\mathcal{M}_{sem}$ are trained.

$$
\mathcal{L}_{ccl}=\frac{1}{2}(\mathcal{L}^{textual2tabular}+\mathcal{L}^{%
tabular2textual}).
$$

#### 4.2.2. Fine-grained Alignment

As mentioned above, CTRL leverages the cross-modal contrastive learning to perform knowledge alignment, where the quality of alignment is measured by the cosine similarity function. However, this approach models the global similarities merely and ignores fine-grained information alignment between the two modalities $\mathbf{h}^{tab}$ and $\mathbf{h}^{text}$. To address this issue, CTRL adopts a fine-grained cross-modal alignment method.

Specifically, both collaborative and semantic representations $\mathbf{h}^{tab}$ and $\mathbf{h}^{text}$ are first transformed into $M$ sub-spaces to extract informative knowledge from different aspects. Taking the collaborative representation $\mathbf{h}^{tab}$ as example, the $m$ -th sub-representation $\mathbf{h}_{m}^{tab}$ is denoted as:

$$
\displaystyle\mathbf{h}_{m}^{tab}=\mathbf{W}^{tab}_{m}\mathbf{h}^{tab}+\mathbf%
{b}^{tab}_{m},\ \ \ \ \ \ \ m=1,2,\dots,M,
$$

where $\mathbf{W}^{tab}_{m}$ and $\mathbf{b}^{tab}_{m}$ are the transform matrix and bias vector for the $m$ -th sub-space, respectively. Similarly, the $m$ -th sub-representation for semantic representation is denoted as $\mathbf{h}_{m}^{text}$.

Then, the fine-grained alignment is performed by calculating the similarity score, which is conducted as a sum of maximum similarity over all sub-representations, shown as:

$$
\displaystyle sim(\mathbf{h}_{i},\mathbf{h}_{j})=\sum_{m_{i}=1}^{M}\max\limits%
_{m_{j}\in 1,2,\dots,M}\{(\mathbf{h}_{i,m_{i}})^{T}\mathbf{h}_{j,m_{j}}\},
$$

where $\mathbf{h}_{i,m}$ is the $m$ -th sub-representation for representation $\mathbf{h}_{i}$. By modeling fine-grained similarity over the cross-modal spaces, CTRL allows for more detailed alignment within instance representations to better integrate knowledge. In this stage, both the language model and collaborative model parameters are updated to better align the representations.

### 4.3. Supervised Fine-tuning

After the cross-modal knowledge alignment stage, the collaborative knowledge and semantic knowledge are aligned and aggregated in a hybrid representation space, where the relations between features are mutually strengthened. In this stage, CTRL further fine-tunes the collaborative models on different downstream tasks (CTR prediction task in this paper) with supervised signals.

At the top of the collaborative model, we add an extra linear layer with random initialization, acting as the output layer for final prediction $\hat{y}$. The widely-used Binary Cross Entropy (BCE) loss is deployed to measure the classification accuracy between the prediction score $\hat{y}$ and the ground-truth label $y$, which is defined as follows:

$$
\displaystyle\mathcal{L}_{ctr}=-\frac{1}{N}\sum_{k=1}^{N}(y_{k}log(\hat{y}_{k}%
)+(1-y_{k})log(1-\hat{y}_{k})),
$$

where $y_{k}$ and $\hat{y}_{k}$ are the ground-truth label and the model prediction score of the $k$ -th instance. After the supervised fine-tuning stage, only the lightweight collaborative model will be deployed online for serving, thus ensuring efficient online inference.

## 5\. Experiments

### 5.1. Experimental Setting

#### 5.1.1. Datasets and Evaluation Metrics

In the experiment, we deploy three large-scale public datasets, which are MovieLens, Amazon (Fashion), and Taobao, whose statistics are summarized in Table 1. Following previous work [^66] [^50] [^26], we use two popular metrics to evaluate the performance, i.e., AUC and Logloss. As acknowledged by many studies [^66] [^50] [^25], an improvement of 0.001 in AUC ($\uparrow$) or Logloss ($\downarrow$) can be regarded as significant because it will bring a large increase in the online revenue. RelaImpr metric [^66] measures the relative improvement with respect to base model, which is defined as follows:

$$
\displaystyle RelaImpr=(\frac{AUC(measure\ model)-0.5}{AUC(base\ model)-0.5}-1%
)\times 100\%.
$$

Besides, the two-tailed unpaired $t$ -test is performed to detect a significant difference between CTRL and the best baseline. The detailed description of datasets and metrics can be referred to Appendix A.

Table 1. Basic statistics of datasets.

| Dataset | Users | Items | User Field | Item Field | Samples |
| --- | --- | --- | --- | --- | --- |
| MovieLens-1M | 6,040 | 3,952 | 5 | 3 | 1,000,000 |
| Amazon(Fashion) | 749,232 | 196,637 | 2 | 4 | 883,636 |
| Alibaba | 1,061,768 | 785,597 | 9 | 6 | 26,557,961 |

#### 5.1.2. Competing Models

We compare CTRL with the following models, which are classified into two classes, i.e., 1) Collaborative Models:Wide&Deep [^7], DeepFM [^18], DCN [^55], PNN [^46], AutoInt [^50], FiBiNet [^25], and xDeepFM [^34]; and 2) Semantic Models: P5 [^15], CTR-BERT [^43], and P-Tab [^36]. The detailed description of these models can be referred to Appendix A.2.

Table 2. Performance comparison of different models. The boldface denotes the highest score and the underline indicates the best result of all baselines. $\star$ represents significance level $p$ -value $<0.05$ of comparing CTRL with the best baselines. RelaImpr denotes the relative AUC improvement rate of CTRL against each baseline.

\[!t\] Category Model MovieLens Amazon Alibaba AUC Logloss RelaImpr AUC Logloss RelaImpr AUC Logloss RelaImpr Collaborative Models Wide&Deep 0.8261 0.4248 3.52% 0.6968 0.4645 5.30% 0.6272 0.1943 5.19% DeepFM 0.8268 0.4219 3.30% 0.6969 0.4645 5.33% 0.6280 0.1951 4.53% DCN 0.8313 0.4165 1.90% 0.6999 0.4642 3.75% 0.6281 0.1949 4.45% PNN 0.8269 0.4220 3.27% 0.6979 0.4657 4.80% 0.6271 0.1956 5.27% AutoInt 0.8290 0.4178 2.61% 0.7012 0.4632 3.08% 0.6279 0.1948 4.61% FiBiNet 0.8196 0.4188 5.63% 0.7003 0.4704 3.54% 0.6270 0.1951 5.35% xDeepFM 0.8296 0.4178 2.43% 0.7009 0.4642 3.23% 0.6272 0.1959 5.19% Semantic Models P5 0.7583 0.4912 30.70% 0.6923 0.4608 7.85% 0.6034 0.3592 29.40% CTR-BERT 0.7650 0.4944 27.40% 0.6934 0.4629 7.24% 0.6005 0.3620 33.13% P-Tab 0.8031 0.4612 11.38% 0.6942 0.4625 6.80% 0.6112 0.3584 20.32% CTRL 0.8376 ${}^{\star}$ 0.4025 ${}^{\star}$ - 0.7074 ${}^{\star}$ 0.4577 ${}^{\star}$ - 0.6338 ${}^{\star}$ 0.1890 ${}^{\star}$ -

#### 5.1.3. Implementation Details

For the prompt construction process, only one type of prompt is used and the comparisons are presented in Section 5.6.2. In the first stage, we utilize AutoInt [^50] as the collaborative model and RoBERTa [^38] as the semantic model by default, as discriminative language models are more efficient at text representation extraction than generative models like GPT under the same parameter scale [^54]. Additionally, we also evaluated the performance of the LLM model like ChatGLM, with the results summarized in Table 4. The mean pooling results of the last hidden states are used as the semantic information representation. For the projection layer, we compress the collaborative representation and the semantic representation to 128 dimensions. Besides, the batch size of the cross-modal knowledge alignment stage is set to 6400 and the temperature coefficient is set to 0.7. The AdamW [^39] optimizer is used and the initial learning rate is set to $1\times 10^{-5}$, which is accompanied by a warm-up mechanism [^20] to $5\times 10^{-4}$. In the second stage, the learning rate of the downstream fine-tuning task is set to 0.001 with Adam [^29] optimizer, and batch size is set to 2048. Batch Normalization [^27] and Dropout [^51] are also applied to avoid overfitting. The feature embedding dimension $d$ for all models is set to 32 empirically. Besides, for all collaborative models, we set the number of hidden layers $L$ as 3 and the number of hidden units as $[256,128,64]$. To ensure a fair comparison, other hyperparameters such as training epochs are adjusted individually for all models to obtain the best results.

### 5.2. Performance Comparison

We compare the overall performance with some SOTA collaborative and semantic models, whose results are summarized in Table 2. From this, we obtain the following observations: 1) CTRL outperforms all the SOTA baselines including semantic and collaborative models over three datasets by a significant margin, showing superior prediction capabilities and proving the effectiveness of the paradigm of combining collaborative and semantic signals. 2) In comparison to the best collaborative model, our proposed CTRL achieves an improvement in AUC of 1.90%, 3.08%, and 4.45% on the three datasets respectively, which effectively demonstrates that integrating semantic knowledge into collaborative models contributes to boost performance. We attribute the significant improvements to the external world knowledge and knowledge reasoning capability in PLMs [^65]. 3) The performance of existing semantic models is lower than that of collaborative models, indicating that collaborative signals and co-occurrence relations are crucial for recommender systems, and relying solely on semantic modeling is difficult to surpass the existing collaborative-based modeling scheme [^15] [^43] [^36]. Instead, our proposed CTRL integrates the advantages of both by combining collaborative signals with semantic signals for recommendation. This approach is likely to be a key path for the future development of recommender systems.

Table 3. Inference efficiency comparison of different models in terms of Model Inference Parameters and Inference Time over testing set with single V100 GPU. As for CTRL, only the collaborative model is needed for online serving, so the number of model parameters is the same as the backbone AutoInt.

<table><tbody><tr><td></td><th colspan="2">Alibaba</th><th colspan="2">Amazon</th></tr><tr><th>Model</th><th>Params</th><th>Inf Time</th><th>Params</th><th>Inf Time</th></tr><tr><td>DeepFM</td><td>8.82 <math><semantics><mrow><mo>×</mo> <msup><mn>10</mn> <mn>7</mn></msup></mrow> <annotation-xml><apply><csymbol>absent</csymbol> <apply><csymbol>superscript</csymbol> <cn>10</cn> <cn>7</cn></apply></apply></annotation-xml> <annotation>\times 10^{7}</annotation> <annotation>× 10 start_POSTSUPERSCRIPT 7 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td>18s</td><td>3.45 <math><semantics><mrow><mo>×</mo> <msup><mn>10</mn> <mn>7</mn></msup></mrow> <annotation-xml><apply><csymbol>absent</csymbol> <apply><csymbol>superscript</csymbol> <cn>10</cn> <cn>7</cn></apply></apply></annotation-xml> <annotation>\times 10^{7}</annotation> <annotation>× 10 start_POSTSUPERSCRIPT 7 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td>0.58s</td></tr><tr><td>DCN</td><td>8.84 <math><semantics><mrow><mo>×</mo> <msup><mn>10</mn> <mn>7</mn></msup></mrow> <annotation-xml><apply><csymbol>absent</csymbol> <apply><csymbol>superscript</csymbol> <cn>10</cn> <cn>7</cn></apply></apply></annotation-xml> <annotation>\times 10^{7}</annotation> <annotation>× 10 start_POSTSUPERSCRIPT 7 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td>19s</td><td>3.46 <math><semantics><mrow><mo>×</mo> <msup><mn>10</mn> <mn>7</mn></msup></mrow> <annotation-xml><apply><csymbol>absent</csymbol> <apply><csymbol>superscript</csymbol> <cn>10</cn> <cn>7</cn></apply></apply></annotation-xml> <annotation>\times 10^{7}</annotation> <annotation>× 10 start_POSTSUPERSCRIPT 7 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td>0.59s</td></tr><tr><td>AutoInt</td><td>8.82 <math><semantics><mrow><mo>×</mo> <msup><mn>10</mn> <mn>7</mn></msup></mrow> <annotation-xml><apply><csymbol>absent</csymbol> <apply><csymbol>superscript</csymbol> <cn>10</cn> <cn>7</cn></apply></apply></annotation-xml> <annotation>\times 10^{7}</annotation> <annotation>× 10 start_POSTSUPERSCRIPT 7 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td>19s</td><td>3.45 <math><semantics><mrow><mo>×</mo> <msup><mn>10</mn> <mn>7</mn></msup></mrow> <annotation-xml><apply><csymbol>absent</csymbol> <apply><csymbol>superscript</csymbol> <cn>10</cn> <cn>7</cn></apply></apply></annotation-xml> <annotation>\times 10^{7}</annotation> <annotation>× 10 start_POSTSUPERSCRIPT 7 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td>0.59s</td></tr><tr><td>P5</td><td>2.23 <math><semantics><mrow><mo>×</mo> <msup><mn>10</mn> <mn>8</mn></msup></mrow> <annotation-xml><apply><csymbol>absent</csymbol> <apply><csymbol>superscript</csymbol> <cn>10</cn> <cn>8</cn></apply></apply></annotation-xml> <annotation>\times 10^{8}</annotation> <annotation>× 10 start_POSTSUPERSCRIPT 8 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td>10832s</td><td>1.10 <math><semantics><mrow><mo>×</mo> <msup><mn>10</mn> <mn>8</mn></msup></mrow> <annotation-xml><apply><csymbol>absent</csymbol> <apply><csymbol>superscript</csymbol> <cn>10</cn> <cn>8</cn></apply></apply></annotation-xml> <annotation>\times 10^{8}</annotation> <annotation>× 10 start_POSTSUPERSCRIPT 8 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td>440s</td></tr><tr><td>CTR-BERT</td><td>1.10 <math><semantics><mrow><mo>×</mo> <msup><mn>10</mn> <mn>8</mn></msup></mrow> <annotation-xml><apply><csymbol>absent</csymbol> <apply><csymbol>superscript</csymbol> <cn>10</cn> <cn>8</cn></apply></apply></annotation-xml> <annotation>\times 10^{8}</annotation> <annotation>× 10 start_POSTSUPERSCRIPT 8 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td>4083s</td><td>1.10 <math><semantics><mrow><mo>×</mo> <msup><mn>10</mn> <mn>8</mn></msup></mrow> <annotation-xml><apply><csymbol>absent</csymbol> <apply><csymbol>superscript</csymbol> <cn>10</cn> <cn>8</cn></apply></apply></annotation-xml> <annotation>\times 10^{8}</annotation> <annotation>× 10 start_POSTSUPERSCRIPT 8 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td>144s</td></tr><tr><td>CTRL(ours)</td><td>8.82 <math><semantics><mrow><mo>×</mo> <msup><mn>10</mn> <mn>7</mn></msup></mrow> <annotation-xml><apply><csymbol>absent</csymbol> <apply><csymbol>superscript</csymbol> <cn>10</cn> <cn>7</cn></apply></apply></annotation-xml> <annotation>\times 10^{7}</annotation> <annotation>× 10 start_POSTSUPERSCRIPT 7 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td>19s</td><td>3.45 <math><semantics><mrow><mo>×</mo> <msup><mn>10</mn> <mn>7</mn></msup></mrow> <annotation-xml><apply><csymbol>absent</csymbol> <apply><csymbol>superscript</csymbol> <cn>10</cn> <cn>7</cn></apply></apply></annotation-xml> <annotation>\times 10^{7}</annotation> <annotation>× 10 start_POSTSUPERSCRIPT 7 end_POSTSUPERSCRIPT</annotation></semantics></math></td><td>0.59s</td></tr></tbody></table>

### 5.3. Serving Efficiency

In industrial recommender systems, online model serving has a strict limit, e.g., 10 $\sim$ 20 milliseconds. Therefore, high service efficiency is essential for CTR models. In this section, we compare the model parameters and inference time of different CTR models over the Alibaba and Amazon datasets, shown in Table 3.

We can observe that existing collaborative-based CTR models have fewer model parameters and higher inference efficiency in comparison with semantic-based models. Moreover, the majority of parameters for the collaborative-based models are concentrated in the embedding layer while the hidden network has very few parameters, thus benefiting the online serving. On the contrary, the semantic-based models (e.g., P5 and CTR-BERT), have a larger number of parameters and lower inference efficiency due to the complex Transformer-based structures, hindering the industrial applications. Instead, for the CTRL with AutoInt as skeleton models, both model parameters and inference time are the same as the original AutoInt model, which is thanks to the decoupled training framework (semantic model is not required for online inference) and ensures the high online serving efficiency.

### 5.4. Visualization of Modal Alignment

To study in depth the distribution of tabular representations and textual representations in the latent space before and after the cross-modal knowledge alignment, we visualize the representations in the MovieLens dataset by projecting them into a two-dimensional space using t-SNE [^53], shown in Figure 4. The two colored points represent the tabular and textual representations, respectively. We can observe that, before the cross-modal knowledge alignment, the representations of the two modalities are distributed in two separate spaces and are essentially unrelated, while mapped into a unified multimodal space after the alignment. This phenomenon substantiates that CTRL aligns the space of two modalities (i.e., tabular and textual), thus injecting the semantic information and external general knowledge into the collaborative model.

![Refer to caption](https://arxiv.org/html/2306.02841v4/extracted/5300699/figure/tsne-2.png)

(a) Before Alignment

### 5.5. Compatibility Study

#### 5.5.1. Compatibility for semantic models

Specifically, for semantic models, we compare four pre-trained language models with different sizes: TinyBERT [^28] with 14.5M parameters ($\text{CTRL}_{\text{TinyBERT}}$), BERT-Base [^10] with 110M parameters ($\text{CTRL}_{\text{BERT}}$), RoBERTa [^38] with 110M parameters ($\text{CTRL}_{\text{RoBERTa}}$), and BERT-Large with 336M parameters ($\text{CTRL}_{\text{Large}}$). Moreover, we have introduced a novel LLM model, ChatGLM [^11], with 6B parameters ($\text{CTRL}_{\text{ChatGLM}}$). For $\text{CTRL}_{\text{ChatGLM}}$, during the training process, we freeze the majority of the parameters and only retain the parameters of the last layer. The experimental results are summarized in Table 4, from which we obtain some observations: 1) In comparison with the backbone model AutoInt, CTRL with different pre-trained language models achieves consistent and significant improvement, where AUC increases by 3.22% and 3.63% for $\text{CTRL}_{\text{ChatGLM}}$, demonstrating the effectiveness of semantics modeling and model compatibility. 2) Among the four CTRL variants ($\text{CTRL}_{\text{TinyBERT}}$, $\text{CTRL}_{\text{BERT}}$, and $\text{CTRL}_{\text{BERTLarge}}$, $\text{CTRL}_{\text{ChatGLM}}$), despite a substantial number of parameters being frozen in ChatGLM, $\text{CTRL}_{\text{ChatGLM}}$ achieves optimal performance. This phenomenon indicates that enlarging the size of the language model can imbue the collaborative model with a wealth of worldly knowledge. Furthermore, even when the parameter scale of the language model is elevated to the billion level, it continues to make a positive contribution to the collaborative model. 3) It can be observed that while the parameter size of ChatGLM is several times that of BERTLarge, the gains are mild. Therefore, when conducting modality alignment, it is only necessary to select language models of moderate scale, such as RoBERTa. 4) Using only TinyBert can lead to a 0.005 increase in AUC, indicating that we can use lightweight pre-trained language models to accelerate model training. 4) $\text{CTRL}_{\text{RoBERTa}}$ has a better performance in the case of an equal number of parameters compared to $\text{CTRL}_{\text{BERT}}$. We hypothesize that this improvement is due to RoBERTa possessing a broader range of world knowledge and a more robust capability for semantic modeling compared to BERT. This indirectly underscores the advantages of increased knowledge in facilitating the knowledge alignment process in collaborative models.

Table 4. Model compatibility study with different semantic models.

<table><thead><tr><th></th><th colspan="2">MovieLens</th><th colspan="2">Amazon</th></tr><tr><th>Model</th><th>AUC</th><th>Logloss</th><th>AUC</th><th>Logloss</th></tr></thead><tbody><tr><th>AutoInt (backbone)</th><td>0.8290</td><td>0.4178</td><td>0.7012</td><td>0.4632</td></tr><tr><th><math><semantics><msub><mtext>CTRL</mtext> <mtext>TinyBERT</mtext></msub> <annotation-xml><apply><csymbol>subscript</csymbol> <ci><mtext>CTRL</mtext></ci> <ci><mtext>TinyBERT</mtext></ci></apply></annotation-xml> <annotation>\text{CTRL}_{\text{TinyBERT}}</annotation> <annotation>CTRL start_POSTSUBSCRIPT TinyBERT end_POSTSUBSCRIPT</annotation></semantics></math> (14.5M)</th><td>0.8347</td><td>0.4137</td><td>0.7053</td><td>0.4612</td></tr><tr><th><math><semantics><msub><mtext>CTRL</mtext> <mtext>BERT</mtext></msub> <annotation-xml><apply><csymbol>subscript</csymbol> <ci><mtext>CTRL</mtext></ci> <ci><mtext>BERT</mtext></ci></apply></annotation-xml> <annotation>\text{CTRL}_{\text{BERT}}</annotation> <annotation>CTRL start_POSTSUBSCRIPT BERT end_POSTSUBSCRIPT</annotation></semantics></math> (110M)</th><td>0.8363</td><td>0.4114</td><td>0.7062</td><td>0.4609</td></tr><tr><th><math><semantics><msub><mtext>CTRL</mtext> <mtext>RoBERTa</mtext></msub> <annotation-xml><apply><csymbol>subscript</csymbol> <ci><mtext>CTRL</mtext></ci> <ci><mtext>RoBERTa</mtext></ci></apply></annotation-xml> <annotation>\text{CTRL}_{\text{RoBERTa}}</annotation> <annotation>CTRL start_POSTSUBSCRIPT RoBERTa end_POSTSUBSCRIPT</annotation></semantics></math> (110M)</th><td>0.8376</td><td>0.4025</td><td>0.7074</td><td>0.4577</td></tr><tr><th><math><semantics><msub><mtext>CTRL</mtext> <mtext>BERTLarge</mtext></msub> <annotation-xml><apply><csymbol>subscript</csymbol> <ci><mtext>CTRL</mtext></ci> <ci><mtext>BERTLarge</mtext></ci></apply></annotation-xml> <annotation>\text{CTRL}_{\text{BERTLarge}}</annotation> <annotation>CTRL start_POSTSUBSCRIPT BERTLarge end_POSTSUBSCRIPT</annotation></semantics></math> (336M)</th><td>0.8380</td><td>0.4040</td><td>0.7076</td><td>0.4574</td></tr><tr><th><math><semantics><msub><mtext>CTRL</mtext> <mtext>ChatGLM</mtext></msub> <annotation-xml><apply><csymbol>subscript</csymbol> <ci><mtext>CTRL</mtext></ci> <ci><mtext>ChatGLM</mtext></ci></apply></annotation-xml> <annotation>\text{CTRL}_{\text{ChatGLM}}</annotation> <annotation>CTRL start_POSTSUBSCRIPT ChatGLM end_POSTSUBSCRIPT</annotation></semantics></math> (6B)</th><td>0.8396</td><td>0.4010</td><td>0.7085</td><td>0.4537</td></tr></tbody></table>

Table 5. Model compatibility study with different collaborative models. The semantic model is set to RoBERTa.

<table><thead><tr><th></th><th colspan="2">MovieLens</th><th colspan="2">Amazon</th></tr><tr><th>Model</th><th>AUC</th><th>Logloss</th><th>AUC</th><th>Logloss</th></tr></thead><tbody><tr><th>Wide&Deep</th><td>0.8261</td><td>0.4348</td><td>0.6966</td><td>0.4645</td></tr><tr><th><math><semantics><msub><mtext>CTRL</mtext> <mtext>Wide&Deep</mtext></msub> <annotation-xml><apply><csymbol>subscript</csymbol> <ci><mtext>CTRL</mtext></ci> <ci><mtext>Wide&Deep</mtext></ci></apply></annotation-xml> <annotation>\text{CTRL}_{\text{Wide\&Deep}}</annotation> <annotation>CTRL start_POSTSUBSCRIPT Wide&Deep end_POSTSUBSCRIPT</annotation></semantics></math></th><td>0.8304</td><td>0.4135</td><td>0.7001</td><td>0.4624</td></tr><tr><th>DeepFM</th><td>0.8268</td><td>0.4219</td><td>0.6965</td><td>0.4646</td></tr><tr><th><math><semantics><msub><mtext>CTRL</mtext> <mtext>DeepFM</mtext></msub> <annotation-xml><apply><csymbol>subscript</csymbol> <ci><mtext>CTRL</mtext></ci> <ci><mtext>DeepFM</mtext></ci></apply></annotation-xml> <annotation>\text{CTRL}_{\text{DeepFM}}</annotation> <annotation>CTRL start_POSTSUBSCRIPT DeepFM end_POSTSUBSCRIPT</annotation></semantics></math></th><td>0.8305</td><td>0.4136</td><td>0.7004</td><td>0.4625</td></tr><tr><th>DCN</th><td>0.8313</td><td>0.4165</td><td>0.6999</td><td>0.4642</td></tr><tr><th><math><semantics><msub><mtext>CTRL</mtext> <mtext>DCN</mtext></msub> <annotation-xml><apply><csymbol>subscript</csymbol> <ci><mtext>CTRL</mtext></ci> <ci><mtext>DCN</mtext></ci></apply></annotation-xml> <annotation>\text{CTRL}_{\text{DCN}}</annotation> <annotation>CTRL start_POSTSUBSCRIPT DCN end_POSTSUBSCRIPT</annotation></semantics></math></th><td>0.8365</td><td>0.4029</td><td>0.7055</td><td>0.4615</td></tr><tr><th>AutoInt</th><td>0.8290</td><td>0.4178</td><td>0.7012</td><td>0.4632</td></tr><tr><th><math><semantics><msub><mtext>CTRL</mtext> <mtext>AutoInt</mtext></msub> <annotation-xml><apply><csymbol>subscript</csymbol> <ci><mtext>CTRL</mtext></ci> <ci><mtext>AutoInt</mtext></ci></apply></annotation-xml> <annotation>\text{CTRL}_{\text{AutoInt}}</annotation> <annotation>CTRL start_POSTSUBSCRIPT AutoInt end_POSTSUBSCRIPT</annotation></semantics></math></th><td>0.8376</td><td>0.4025</td><td>0.7063</td><td>0.4582</td></tr></tbody></table>

#### 5.5.2. Compatibility for collaborative models

Besides, we apply CTRL to different collaborative models, including Wide&Deep, DeepFM, DCN, and AutoInt. From Table 5, we can observe that CTRL achieves remarkable improvements with different collaborative models consistently. The average improvements over RelaImpr metric are 1.31% for Wide&Deep, 1.13% for DeepFM, 1.57% for DCN, and 2.61% for AutoInt respectively, which demonstrates the effectiveness and model compatibility.

### 5.6. Ablation Study

#### 5.6.1. Ablation Study Analysis.

In this section, we conduct ablation experiments to better understand the importance of different components. 1)We replace the maxsim similarity with cosine similarity; 2) we remove the pre-trained language model weights. 3) we investigate the impact of end-to-end training, which combines the two-stage process into a single stage(i.e., cross-modal knowledge alignment and CTR prediction tasks are trained together). From Figure 5, we observe the following results: 1) When we remove the weights of the pre-trained language model, the loss in model performance is quite significant. This demonstrates that the primary source of improvement in the collaborative model’s performance is attributed to the world knowledge and semantic modeling capabilities of the language model, rather than solely due to contrastive learning. 2) After replacing cosine similarity with maxsim similarity, there is a degradation in the model performance. This indicates that fine-grained alignment facilitates the collaborative model in learning semantic representations. 3) We observe that the performance of end-to-end training is inferior to the pre-training and fine-tuning paradigm of CTRL. We conjecture that this may be due to the multi-objective setting in end to end training paradigm, which hampers the performance of the collaborative model on the CTR prediction task.

![Refer to caption](https://arxiv.org/html/2306.02841v4/x5.png)

(a) Movielens

#### 5.6.2. Prompt Analysis

In this subsection, we explore the impact of different prompt construction methods on training CTRL. We believe that this exploration will inspire future work on how to better construct prompts. Below are several rules for constructing prompts: 1) Transform user and item features into natural language text that can be easily understood; 2) Remove auxiliary text descriptions and connect feature fields and values with “-" directly; 3) Remove the feature fields and transform all the feature values into a single phrase; 4) Mask the feature fields with a meaningless unified word “Field”; 5) Replace the separator “-" with separator “:".

We pre-train CTRL on these prompts and then fine-tune the CTR prediction task with the collaborative model, whose results are shown in Figure 6. From Figure 6, we can obtain the following observations: 1) Prompt-1 performs significantly better than all prompts, which indicates that constructing prompts in the form of natural language is beneficial for modeling. 2) The performance of Prompt-3 is weaker than Prompt-2, which confirms the importance of semantic information of feature fields, the lack of which will degrade the performance of the model remarkably. Meanwhile, the performance of Prompt-3 is weaker than Prompt-4, indicating that prompt with rules is stronger than prompt without rules. 3) The performance of Prompt-2 and Prompt-5 are similar, suggesting that the difference of connectives between feature field and feature value has little effect on the performance. Based on these findings, we can identify the following characteristics of designing a good prompt: 1) including feature fields such as age, gender, etc.; 2) having fluent and grammatically correct sentences and containing as much semantic information as possible.

![Refer to caption](https://arxiv.org/html/2306.02841v4/x7.png)

(a) AUC of different prompt

## 6\. Application in Industry System

![Refer to caption](https://arxiv.org/html/2306.02841v4/x9.png)

Figure 7. Online workflow of CTRL.

### 6.1. Deploying Details of CTRL Online

In this section, we deploy CTRL in a Huawei large-scale industrial system to verify its effectiveness. During the training, we collected and sampled seven days of user behavior data from Huawei large-scale recommendation platform, where millions of user logs are generated daily. More than 30 distinct features are used, including user profile features (e.g., department), user behavior features (e.g., list of items clicked by the user), item original features (e.g., item title), and statistical features (e.g., the number of clicks on the item), as well as contextual features (e.g., time). In the first stage of the training, we only train for one epoch. In the second stage, we train for five epochs. Together, this totals to approximately five hours. This relatively short training time ensures that we are able to update the model on a daily basis. In the end, we deploy the collaborative model in CTRL at the ranking stage.

### 6.2. Offline and Online Performance

We compare the CTRL model (backbone AutoInt and RoBERTa) with the SOTA models. The offline performance results are presented in Table 6. It is evident that CTRL outperforms the baseline models significantly in terms of AUC and Logloss, thereby demonstrating its superior performance. By incorporating the modeling capabilities of both the semantic and collaborative models, CTRL achieves a significant performance improvement over both collaborative models and semantic models. Moreover, according to the results in Table 3, CTRL would not increase any serving latency compared to the backbone collaborative model, which is an industrial-friendly framework with high accuracy and low inference latency. During the online A/B testing for seven days, we obtained a 5% gain of CTR compared with the base model. CTRL has now been deployed in online services, catering to tens of millions of HuaWei users.

Table 6. Huawei recommender system performance comparison.

<table><tbody><tr><td>Category</td><td>Model</td><td>AUC</td><td>Logloss</td><td>RelaImpr</td></tr><tr><td rowspan="3">Collaborative</td><td>DeepFM</td><td>0.6547</td><td>0.1801</td><td>8.79%</td></tr><tr><td>AutoInt</td><td>0.6586</td><td>0.1713</td><td>6.12%</td></tr><tr><td>DCN</td><td>0.6558</td><td>0.1757</td><td>8.02%</td></tr><tr><td rowspan="2">Semantic</td><td>CTR-BERT</td><td>0.6484</td><td>0.1923</td><td>13.41%</td></tr><tr><td>P5</td><td>0.6472</td><td>0.1974</td><td>14.33%</td></tr><tr><td colspan="2">CTRL</td><td>0.6683 <math><semantics><msup><mo>⋆</mo></msup> <annotation-xml><apply><ci>normal-⋆</ci></apply></annotation-xml> <annotation>{}^{\star}</annotation> <annotation>start_FLOATSUPERSCRIPT ⋆ end_FLOATSUPERSCRIPT</annotation></semantics></math></td><td>0.1606 <math><semantics><msup><mo>⋆</mo></msup> <annotation-xml><apply><ci>normal-⋆</ci></apply></annotation-xml> <annotation>{}^{\star}</annotation> <annotation>start_FLOATSUPERSCRIPT ⋆ end_FLOATSUPERSCRIPT</annotation></semantics></math></td><td>-</td></tr></tbody></table>

## 7\. Conclusion

In this paper, we reveal the importance of both collaborative and semantic signals for CTR prediction and present CTRL, an industrial-friendly and model-agnostic framework with high inference efficiency. CTRL treats the tabular data and converted textual data as two modalities and leverages contrastive learning for fine-grained knowledge alignment and integration. Finally, the lightweight collaborative model can be deployed online for efficient serving after fine-tuned with supervised signals. Our experiments demonstrate that CTRL outperforms state-of-the-art collaborative and semantic models while maintaining good inference efficiency. Future work includes exploring the application on other downstream tasks, such as sequence recommendation and explainable recommendation.

## References

## Appendix A Experimental Setting

### A.1. Datasets and Evaluation Metrics

MovieLens Dataset <sup>3</sup> is a movie recommendation dataset and following previous work [^50], we consider samples with ratings less than 3 as negative, samples with scores greater than 3 as positive, and remove neutral samples, i.e., rating equal to 3. Amazon Dataset <sup>4</sup> [^44] is a widely-used benchmark dataset [^66] [^59] [^46] [^60] and our experiment uses a subset Fashion following [^66]. We take the items with a rating of greater than 3 as positive and the rest as negative. Alibaba Dataset <sup>5</sup> [^13] is a Taobao ad click dataset. For the MovieLens and Amazon datasets, following previous work [^31], we divide the train, validation, and test sets by user interaction time in the ratio of 8:1:1. For the Alibaba dataset, we divide the datasets according to the official implementation [^66], and the data from the previous seven days are used as the training and validation samples with 9:1 ratio, and the data from the eighth day are used for test.

The area under the ROC curve (AUC) measures the probability that the model will assign a higher score to a randomly selected positive item than to a randomly selected negative item. Logloss is a widely used metric in binary classification to measure the distance between two distributions.

### A.2. Competing Models

Collaborative Models: Wide&Deep combines linear feature interactions (wide) with nonlinear feature learning (deep). DeepFM integrates a Factorization Machine with Wide&Deep, minimizing feature engineering. DCN enhances Wide&Deep with a cross-network to capture higher-order interactions. AutoInt uses Multi-head Self-Attention for feature interaction. PNN, xDeepFM, and FiBiNET all serve as strong baselines.

Semantic Models: P5 transforms recommendation into text generation with a T5 base, while CTR-BERT, an Amazon model, leverages BERT towers for semantic prediction. P-Tab employs pre-training with Masked Language Modeling (MLM) on recommendation datasets, then fine-tunes for prediction.

## Appendix B Hyperparameter Analysis

### B.1. The Impact of Contrastive Learning Temperature Coefficient

To explore the effect of different temperature parameters in the cross-modal knowledge alignment contrastive learning, we implement experiments on MovieLens and Amazon datasets, and the results are in Figure 8(a). From the results we can get the following observations: 1) The temperature coefficient in contrastive learning has an obvious impact on the performance. As the temperature coefficient increases, the performance will have a tendency to improve first and then decrease, indicating that increasing the coefficient within a certain range is beneficial to improve the performance. 2) For both MovieLens and Amazon datasets, the optimal temperature coefficient is below 1 in our experiments, which has also been verified in previous work [^47] [^58].

### B.2. The Impact of First Stage Batch Size

We also explore the impact of different batch sizes, and the results are shown in Figure 8(b). We can observe that as the batch size increases, the performance is also improved on both datasets, which indicates that increasing the batch size during the contrastive learning pre-training is conducive to achieving better cross-modal knowledge alignment effect and improving the prediction accuracy.

![Refer to caption](https://arxiv.org/html/2306.02841v4/x10.png)

(a) Temperature Coefficient

[^2]: Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023. TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation. *arXiv preprint arXiv:2305.00447* (2023).

[^3]: Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language Models are Few-Shot Learners. [https://doi.org/10.48550/ARXIV.2005.14165](https://doi.org/10.48550/ARXIV.2005.14165)

[^4]: Bo Chen, Yichao Wang, Zhirong Liu, Ruiming Tang, Wei Guo, Hongkun Zheng, Weiwei Yao, Muyu Zhang, and Xiuqiang He. 2021. Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models. In *Proceedings of the 30th ACM International Conference on Information & Knowledge Management*. 3757–3766.

[^5]: Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020. A simple framework for contrastive learning of visual representations. In *Proceedings of ICML*. PMLR, 1597–1607.

[^6]: Zheng Chen. 2023. PALR: Personalization Aware LLMs for Recommendation. *arXiv preprint arXiv:2305.07622* (2023).

[^7]: Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & deep learning for recommender systems. In *Proceedings of the 1st workshop on deep learning for recommender systems*. 7–10.

[^8]: David R Cox. 1958. The regression analysis of binary sequences. *Journal of the Royal Statistical Society: Series B (Methodological)* 20, 2 (1958), 215–232.

[^9]: Zeyu Cui, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. 2022. M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems. [https://doi.org/10.48550/ARXIV.2205.08084](https://doi.org/10.48550/ARXIV.2205.08084)

[^10]: Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805* (2018).

[^11]: Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. 2022. GLM: General Language Model Pretraining with Autoregressive Blank Infilling. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. 320–335.

[^12]: Bent Fuglede and Flemming Topsoe. 2004. Jensen-Shannon divergence and Hilbert space embedding. In *International symposium on Information theory, 2004. ISIT 2004. Proceedings.* IEEE, 31.

[^13]: Kun Gai, Xiaoqiang Zhu, Han Li, Kai Liu, and Zhe Wang. 2017. Learning piece-wise linear models from large scale data for ad click prediction. *arXiv preprint arXiv:1704.05194* (2017).

[^14]: Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple Contrastive Learning of Sentence Embeddings. In *Proceedings of EMNLP*. 6894–6910.

[^15]: Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022. Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5). In *Proceedings of the 16th ACM Conference on Recommender Systems*. 299–315.

[^16]: Thore Graepel, Joaquin Quinonero Candela, Thomas Borchert, and Ralf Herbrich. 2010. Web-scale bayesian click-through rate prediction for sponsored search advertising in microsoft’s bing search engine. Omnipress.

[^17]: Huifeng Guo, Bo Chen, Ruiming Tang, Weinan Zhang, Zhenguo Li, and Xiuqiang He. 2021. An embedding learning framework for numerical features in ctr prediction. In *SIGKDD*. 2910–2918.

[^18]: Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: a factorization-machine based neural network for CTR prediction. *arXiv preprint arXiv:1703.04247* (2017).

[^19]: Michael Gutmann and Aapo Hyvärinen. 2010. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In *Proceedings of AISTATS*. JMLR Workshop and Conference Proceedings, 297–304.

[^20]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In *CVPR*. 770–778.

[^21]: Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers, et al. 2014. Practical lessons from predicting clicks on ads at facebook. In *Proceedings of the eighth international workshop on data mining for online advertising*. 1–9.

[^22]: Mickel Hoang, Oskar Alija Bihorac, and Jacobo Rouces. 2019. Aspect-based sentiment analysis using bert. In *Proceedings of the 22nd nordic conference on computational linguistics*. 187–196.

[^23]: Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley, and Wayne Xin Zhao. 2023. Large Language Models are Zero-Shot Rankers for Recommender Systems. *arXiv preprint arXiv:2305.08845* (2023).

[^24]: Zhiqiang Hu, Roy Ka-Wei Lee, Charu C Aggarwal, and Aston Zhang. 2022. Text style transfer: A review and experimental evaluation. *ACM SIGKDD Explorations Newsletter* 24, 1 (2022), 14–45.

[^25]: Tongwen Huang, Zhiqi Zhang, and Junlin Zhang. 2019a. FiBiNET. In *Proceedings of the 13th ACM Conference on Recommender Systems*. ACM. [https://doi.org/10.1145/3298689.3347043](https://doi.org/10.1145/3298689.3347043)

[^26]: Tongwen Huang, Zhiqi Zhang, and Junlin Zhang. 2019b. FiBiNET: combining feature importance and bilinear feature interaction for click-through rate prediction. In *Proceedings of the 13th ACM Conference on Recommender Systems*. 169–177.

[^27]: Sergey Ioffe and Christian Szegedy. 2015. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *ICML*. PMLR, 448–456.

[^28]: Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. 2019. Tinybert: Distilling bert for natural language understanding. *arXiv preprint arXiv:1909.10351* (2019).

[^29]: Diederik P. Kingma and Jimmy Ba. 2014. Adam: A Method for Stochastic Optimization. [https://doi.org/10.48550/ARXIV.1412.6980](https://doi.org/10.48550/ARXIV.1412.6980)

[^30]: Daniel D Lee and H Sebastian Seung. 1999. Learning the parts of objects by non-negative matrix factorization. *Nature* 401, 6755 (1999), 788–791.

[^31]: Xiangyang Li, Bo Chen, HuiFeng Guo, Jingjie Li, Chenxu Zhu, Xiang Long, Sujian Li, Yichao Wang, Wei Guo, Longxia Mao, et al. 2022a. IntTower: the Next Generation of Two-Tower Model for Pre-Ranking System. In *CIKM*. 3292–3301.

[^32]: Xiangyang Li, Xiang Long, Yu Xia, and Sujian Li. 2022b. Low Resource Style Transfer via Domain Adaptive Meta Learning. *arXiv preprint arXiv:2205.12475* (2022).

[^33]: Xiangyang Li, Yu Xia, Xiang Long, Zheng Li, and Sujian Li. 2021. Exploring text-transformers in aaai 2021 shared task: Covid-19 fake news detection in english. In *Combating Online Hostile Posts in Regional Languages during Emergency Situation: First International Workshop, CONSTRAINT 2021, Collocated with AAAI 2021, Virtual Event, February 8, 2021, Revised Selected Papers 1*. Springer, 106–115.

[^34]: Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. 2018. xdeepfm: Combining explicit and implicit feature interactions for recommender systems. In *SIGKDD*. 1754–1763.

[^35]: Bin Liu, Ruiming Tang, Yingzhi Chen, Jinkai Yu, Huifeng Guo, and Yuzhou Zhang. 2019b. Feature generation by convolutional neural network for click-through rate prediction. In *The World Wide Web Conference*. 1119–1129.

[^36]: Guang Liu, Jie Yang, and Ledell Wu. 2022. PTab: Using the Pre-trained Language Model for Modeling Tabular Data. *arXiv preprint arXiv:2209.08060* (2022).

[^37]: Junling Liu, Chao Liu, Renjie Lv, Kang Zhou, and Yan Zhang. 2023. Is ChatGPT a Good Recommender? A Preliminary Study. *arXiv preprint arXiv:2304.10149* (2023).

[^38]: Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019a. Roberta: A robustly optimized bert pretraining approach. *arXiv preprint arXiv:1907.11692* (2019).

[^39]: Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101* (2017).

[^40]: Yuanfu Lu, Yuan Fang, and Chuan Shi. 2020. Meta-learning on heterogeneous information networks for cold-start recommendation. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. 1563–1573.

[^41]: H Brendan McMahan, Gary Holt, David Sculley, Michael Young, Dietmar Ebner, Julian Grady, Lan Nie, Todd Phillips, Eugene Davydov, Daniel Golovin, et al. 2013. Ad click prediction: a view from the trenches. In *SIGKDD*. 1222–1230.

[^42]: Marcin Michał Mirończuk and Jarosław Protasiewicz. 2018. A recent overview of the state-of-the-art elements of text classification. *Expert Systems with Applications* 106 (2018), 36–54.

[^43]: Aashiq Muhamed, Iman Keivanloo, Sujan Perera, James Mracek, Yi Xu, Qingjun Cui, Santosh Rajagopalan, Belinda Zeng, and Trishul Chilimbi. 2021. CTR-BERT: Cost-effective knowledge distillation for billion-parameter teacher models. In *NeurIPS Efficient Natural Language and Speech Processing Workshop*.

[^44]: Jianmo Ni, Jiacheng Li, and Julian McAuley. 2019. Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In *EMNLP-IJCNLP*. 188–197.

[^45]: Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. *arXiv preprint arXiv:2203.02155* (2022).

[^46]: Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, and Jun Wang. 2016. Product-based neural networks for user response prediction. In *ICDM*. IEEE, 1149–1154.

[^47]: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In *International conference on machine learning*. PMLR, 8748–8763.

[^48]: Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2019. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. [https://doi.org/10.48550/ARXIV.1910.10683](https://doi.org/10.48550/ARXIV.1910.10683)

[^49]: Steffen Rendle. 2010. Factorization machines. In *2010 IEEE International conference on data mining*. IEEE, 995–1000.

[^50]: Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, and Jian Tang. 2019. Autoint: Automatic feature interaction learning via self-attentive neural networks. In *CIKM*. 1161–1170.

[^51]: Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014. Dropout: a simple way to prevent neural networks from overfitting. *The journal of machine learning research* 15, 1 (2014), 1929–1958.

[^52]: Weiwei Sun, Lingyong Yan, Xinyu Ma, Pengjie Ren, Dawei Yin, and Zhaochun Ren. 2023. Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent. *arXiv preprint arXiv:2304.09542* (2023).

[^53]: Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-SNE. *Journal of machine learning research* 9, 11 (2008).

[^54]: Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2020. SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems. arXiv:1905.00537 \[cs.CL\]

[^55]: Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & cross network for ad click predictions. In *ADKDD*. 1–7.

[^56]: Xin Xin, Bo Chen, Xiangnan He, Dong Wang, Yue Ding, and Joemon M Jose. 2019. CFM: Convolutional Factorization Machines for Context-Aware Recommendation.. In *IJCAI*, Vol. 19. 3926–3932.

[^57]: Hu Xu, Bing Liu, Lei Shu, and Philip S Yu. 2019. BERT post-training for review reading comprehension and aspect-based sentiment analysis. *arXiv preprint arXiv:1904.02232* (2019).

[^58]: Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. 2021. FILIP: fine-grained interactive language-image pre-training. *arXiv preprint arXiv:2111.07783* (2021).

[^59]: Yantao Yu, Weipeng Wang, Zhoutian Feng, and Daiyue Xue. 2021. A Dual Augmented Two-tower Model for Online Large-scale Recommendation. (2021).

[^60]: Zeping Yu, Jianxun Lian, Ahmad Mahmoody, Gongshen Liu, and Xing Xie. 2019. Adaptive User Modeling with Long and Short-Term Preferences for Personalized Recommendation.. In *IJCAI*. 4213–4219.

[^61]: Jizhi Zhang, Keqin Bao, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He. 2023a. Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation. *arXiv preprint arXiv:2305.07609* (2023).

[^62]: Junjie Zhang, Ruobing Xie, Yupeng Hou, Wayne Xin Zhao, Leyu Lin, and Ji-Rong Wen. 2023b. Recommendation as instruction following: A large language model empowered recommendation approach. *arXiv preprint arXiv:2305.07001* (2023).

[^63]: Weinan Zhang, Tianming Du, and Jun Wang. 2016. Deep learning over multi-field categorical data. In *ECIR*. Springer, 45–57.

[^64]: Weinan Zhang, Shuai Yuan, and Jun Wang. 2014. Optimal real-time bidding for display advertising. In *Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining*. 1077–1086.

[^65]: Yuhui Zhang, Hao Ding, Zeren Shui, Yifei Ma, James Zou, Anoop Deoras, and Hao Wang. 2021. Language models as recommender systems: Evaluations and limitations. (2021).

[^66]: Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In *SIGKDD*. 1059–1068.