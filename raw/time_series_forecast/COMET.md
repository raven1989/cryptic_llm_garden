---
title: "COMET: Codebook-based Online-adaptive Multi-scale Embedding for Time-series Anomaly Detection"
source: "https://arxiv.org/html/2602.01635"
author:
published:
created: 2026-07-23
description:
tags:
  - "clippings"
---
Jinwoo Park    Hyeongwon Kang    Seung Hun Han    Pilsung Kang

###### Abstract

Time series anomaly detection is a critical task across various industrial domains. However, capturing temporal dependencies and multivariate correlations within patch-level representation learning remains underexplored, and reliance on single-scale patterns limits the detection of anomalies across different temporal ranges. Furthermore, focusing on normal data representations makes models vulnerable to distribution shifts at inference time. To address these limitations, we propose Codebook-based Online-adaptive Multi-scale Embedding for Time-series anomaly detection (COMET), which consists of three key components: (1) Multi-scale Patch Encoding captures temporal dependencies and inter-variable correlations across multiple patch scales. (2) Vector-Quantized Coreset learns representative normal patterns via codebook and detects anomalies with a dual-score combining quantization error and memory distance. (3) Online Codebook Adaptation generates pseudo-labels based on codebook entries and dynamically adapts the model at inference through contrastive learning. Experiments on five benchmark datasets demonstrate that COMET achieves the best performance in 36 out of 45 evaluation metrics, validating its effectiveness across diverse environments.

Time Series Anomaly Detection, Multivariate Time Series, Multi-Granularity Learning, Vector Quantization, Codebook, Test-Time Adaptation, Contrastive Learning

## 1 Introduction

Time series anomaly detection has become a core analytical tool for ensuring reliable system operation, driven by the increasing deployment of large-scale sensor systems across diverse industrial domains [^35] [^41] [^7]. Since anomalies occur infrequently within large volumes of normal data and labeling incurs substantial costs, unsupervised learning approaches have been widely adopted. In such approaches, training data are assumed to be normal, and models learn normal patterns to detect anomalies based on deviations from these learned representations [^26].

Unsupervised time series anomaly detection methods still face several challenges due to the inherent characteristics of time series data and task-specific requirements of anomaly detection. First, effective embeddings must capture temporal dependencies and multivariate correlations [^20]. While recent patch-based methods encode local temporal patterns by segmenting time series into fixed-length patches [^27], most rely on a single fixed patch scale and channel-independent designs. Since time series encode different information depending on the temporal scale [^30], such single-scale representations are insufficient to comprehensively capture normal and anomalous patterns that manifest across diverse temporal scales [^23]. Second, many existing time series anomaly detection methods compute anomaly scores at the individual timestep level, which limits their ability to capture temporal context and dependencies inherent in time series data [^17]. Time series anomalies are commonly categorized into point, contextual, and collective anomalies [^5]. Point anomalies, which correspond to isolated observations that deviate significantly from the normal range, can often be detected relatively easily based on timestep-level scores. In contrast, contextual anomalies refer to observations that appear normal in isolation but are abnormal given their surrounding temporal context, while collective anomalies involve sequences of observations that may individually seem normal but collectively form anomalous temporal patterns. As a result, timestep-level scoring approaches are inherently limited in capturing anomalous patterns that emerge only within broader temporal contexts across different time scales [^17]. Third, real-world time series data exhibit both time-invariant characteristics, such as system-specific operational principles, and time-variant properties that evolve dynamically over time, in addition to being influenced by unpredictable disturbances [^21] [^8]. As a result, time series data are inherently non-stationary, leading to distribution shifts between training and inference as their statistical properties change over time [^16]. This non-stationarity violates the i.i.d. assumption commonly adopted in machine learning, where training and test data are assumed to be drawn from the same distribution [^4], causing the normal patterns learned during training to gradually diverge from the true data distribution and degrading generalization performance. To address such distribution shifts, Test-Time Adaptation (TTA) techniques have been explored in time series anomaly detection [^13]. However, existing approaches often rely on post-hoc filtering strategies, such as anomaly score thresholding, without explicit mechanisms to reliably identify normal samples during adaptation, which can lead to mistaken assimilation of anomalous patterns as normal behavior due to incorrect model judgments.

In this paper, we propose Codebook-based Online-adaptive Multi-scale Embedding for time series anomaly detection (COMET). COMET models temporal dependencies and multivariate correlations via Multi-scale Patch Encoding, detects anomalies at the patch level by learning representative normal patterns through a Vector-Quantized Coreset, and adapts to distribution shifts at inference time using Online Codebook Adaptation. The main contributions of this work are summarized as follows:

- We propose Multi-scale Patch Encoding, which captures temporal dependencies and multivariate correlations by combining variable-wise encoders with a shared core encoder across multiple patch scales.
- We introduce Vector-Quantized Coreset, which learns representative normal patterns through codebook quantization and detects anomalies using a dual scoring scheme combining quantization error and memory distance.
- We propose Online Codebook Adaptation, which dynamically adapts to distribution shifts at inference time while preventing model contamination through codebook activation–based pseudo-labeling and contrastive learning.
- We conduct extensive experiments on five benchmark datasets, demonstrating the effectiveness of the proposed method by achieving state-of-the-art performance on 39 out of 45 evaluation metrics.

Taken together, these components form a unified framework that jointly supports expressive multi-scale representations and explicit distance-based modeling to normal patterns, enabling robust anomaly detection under distribution shifts.

## 2 Related work

### 2.1 Unsupervised Multivariate Time Series Anomaly Detection

Reconstruction-based methods. Reconstruction-based methods train models solely on normal data to reconstruct the input, assuming that anomalous data are difficult to reconstruct [^7]. Early studies modeled temporal dependencies using LSTM-based encoder–decoder architectures [^24] and LSTM-VAE models incorporating variational inference [^29], while USAD [^3] amplified reconstruction errors through adversarial learning. With the advent of Transformers [^33], subsequent methods have incorporated attention mechanisms, including Anomaly Transformer [^37], which detects anomalies via association discrepancy, and VTT [^12], which jointly models temporal and inter-variable dependencies. More recent approaches such as D3R [^34] and CATCH [^36] further address non-stationarity and frequency-aware multivariate patterns. Despite these advances, most reconstruction-based methods still compute anomaly scores at the timestep level, limiting their ability to capture anomalies formed by consecutive observations [^17].

Representation-Based Models. Representation-based models detect anomalies using learned latent representations rather than raw time series inputs, enabling robust modeling of complex temporal characteristics [^7]. THOC [^31] learns hierarchical temporal representations and models normal regions via one-class classification. DCdetector [^38] captures discriminative features through dual-attention mechanisms and contrastive learning across patch-wise and channel-wise views, while CARLA [^6] employs self-supervised learning with synthetic anomaly injection to learn decision boundaries in the representation space. In this work, we adopt a representation distance–based approach, leveraging codebook entries learned via vector quantization as a compact coreset of normal patterns.

### 2.2 Vector Quantization for Time Series

Vector Quantization (VQ) is a technique that maps continuous representations to a finite set of discrete codebook entries. VQ-VAE [^32] introduced this approach for learning discrete latent representations, and in the time series domain, TimeVQVAE [^18] successfully extended it to time series generation by modeling temporal dynamics in a quantized latent space. Its further extension to anomaly detection, TimeVQVAE-AD [^19], leverages masked generative modeling to identify abnormal segments based on reconstruction difficulty. In contrast, our approach employs VQ to learn representative normal patterns and detects anomalies based on distances in the embedding space.

### 2.3 Test-Time Adaptation for Time Series

Test-Time Adaptation (TTA) adapts a trained model to incoming data at test time and is particularly important for handling distribution shifts caused by time-series non-stationarity. In time series forecasting, various TTA methods have been proposed. TAFAS [^14] performs online adaptation by utilizing recent observations as partially observed ground truth. PETSA [^25] achieves parameter-efficient adaptation via low-rank adaptation, while PROCEED [^40] proposes a proactive approach that anticipates concept drift to respond to distribution shifts in advance. In contrast, research on TTA for time series anomaly detection remains in its early stages. M2N2 [^13] updates model parameters using samples classified as normal to mitigate contamination, but relies on threshold-based filtering, making it sensitive to threshold selection and misclassification. In this work, we overcome this limitation by identifying normal samples through codebook entries activated during training, without requiring explicit thresholds.

## 3 Proposed Method

Problem Definition. Given a multivariate time series $\mathbf{X}\in\mathbb{R}^{L\times D}$, where $L$ denotes the sequence length and $D$ the number of variables, unsupervised time series anomaly detection aims to train a model on unlabeled normal data and produce an anomaly score $S_{t}\in\mathbb{R}$ for each time step $t$ at inference.

The overall framework and its algorithmic procedure are provided in Appendix A, with an overview illustrated in Figure 1.

![Refer to caption](https://arxiv.org/html/2602.01635v2/x1.png)

Refer to caption

### 3.1 Multi-scale Patch Encoding

In time series anomaly detection, capturing patterns across multiple temporal scales while jointly modeling temporal dependencies within individual variables and correlations among variables is essential. To this end, we propose Multi-scale Patch Encoding, which leverages multi-scale patches and combines variable-wise encoders with a shared core encoder, inspired by SOFTS [^22]. While SOFTS was originally proposed for forecasting, its encoder is used here purely for representation learning, as scale-aware and correlation-preserving embeddings are equally important for anomaly detection.

Multi-scale Patching. Given an input time series $\mathbf{X}\in\mathbb{R}^{L\times D}$, we extract multi-scale patches using $K$ different patch sizes $\mathcal{P}=\{p_{1},...,p_{K}\}$ with corresponding strides $\mathcal{S}=\{s_{1},...,s_{K}\}$. At scale $k$, the $j$ -th patch of variable $i$ is defined as

$$
\mathbf{P}_{k}^{(i,j)}=\mathbf{X}_{j\cdot s_{k}:j\cdot s_{k}+p_{k},\,i}\in\mathbb{R}^{p_{k}}
$$

where $i\in\{1,...,D\}$ and $j\in\{0,...,N_{k}-1\}$, and the number of patches is given by $N_{k}=\lfloor(L-p_{k})/s_{k}\rfloor+1$.

SeriesPatch-specific Encoder. Since each variable in a multivariate time series exhibits distinct statistical characteristics and patterns, we apply an independent encoder $f_{k}^{(s,i)}:\mathbb{R}^{p_{k}}\rightarrow\mathbb{R}^{d/2}$ to each variable $i$:

$$
\mathbf{h}_{k}^{(s,i,j)}=f_{k}^{(s,i)}(\mathbf{P}_{k}^{(i,j)})=\mathbf{W}_{k}^{(s,i)}\mathbf{P}_{k}^{(i,j)}+\mathbf{b}_{k}^{(s,i)}
$$

where $\mathbf{W}_{k}^{(s,i)}\in\mathbb{R}^{(d/2)\times p_{k}}$ denotes learnable weights and $\mathbf{b}_{k}^{(s,i)}\in\mathbb{R}^{d/2}$ denotes a bias term.

Core Encoder. To address the limitation that variable-specific features alone cannot capture inter-variable correlations, we concatenate patches from all variables at each patch index $j$ and employ a shared encoder $f_{k}^{(c)}:\mathbb{R}^{D\cdot p_{k}}\rightarrow\mathbb{R}^{d_{c}}$ to model correlations across variables:

$$
\displaystyle\tilde{\mathbf{P}}_{k}^{(j)}
$$
 
$$
\displaystyle=\left[\mathbf{P}_{k}^{(1,j)};\cdots;\mathbf{P}_{k}^{(D,j)}\right]\in\mathbb{R}^{D\cdot p_{k}},
$$
$$
\displaystyle\mathbf{h}_{k}^{(c,j)}
$$
 
$$
\displaystyle=f_{k}^{(c)}(\tilde{\mathbf{P}}_{k}^{(j)})=\mathbf{W}_{k}^{(c)}\tilde{\mathbf{P}}_{k}^{(j)}+\mathbf{b}_{k}^{(c)}.
$$

The core encoder shares parameters across all variables within the same scale, enabling it to learn inter-variable interaction information.

Feature Fusion. To effectively integrate variable-specific features $\mathbf{h}_{k}^{(s,i,j)}$ with shared features $\mathbf{h}_{k}^{(c,j)}$, we concatenate them and apply a fusion layer:

$$
\mathbf{z}_{e,k}^{(i,j)}=\mathbf{W}_{k}^{(g)}\left[\mathbf{h}_{k}^{(s,i,j)};\mathbf{h}_{k}^{(c,j)}\right]+\mathbf{b}_{k}^{(g)}
$$

where $\mathbf{W}_{k}^{(g)}\in\mathbb{R}^{d\times((d/2)+d_{c})}$. Through the feature fusion process, we obtain a continuous embedding $\mathbf{Z}_{e,k}\in\mathbb{R}^{D\times N_{k}\times d}$ at scale $k$.

### 3.2 Vector-Quantized Coreset

In this work, we leverage Vector Quantization (VQ) to quantize continuous embeddings into a finite set of codebook entries, allowing the model to naturally learn representative normal patterns during training. In unsupervised anomaly detection, since the training data are assumed to be normal, the learned codebook entries can be regarded as prototypes of normal patterns. This design enables the construction of a memory bank using at most $M$ representative patterns, ensuring memory efficiency. Additionally, storing the quantization indices of patch embeddings allows efficient index-based updates during Online Codebook Adaptation.

Vector Quantization. For each scale $k$, we define a learnable codebook $\mathbf{C}_{k}=\{\mathbf{c}_{1},...,\mathbf{c}_{M}\}\in\mathbb{R}^{M\times d}$. The continuous embedding $\mathbf{z}_{e,k}^{(i,j)}$ is then quantized to the nearest entry in the codebook as follows:

$$
\displaystyle q_{k}^{(i,j)}
$$
 
$$
\displaystyle=\underset{m\in\{1,...,M\}}{\arg\min}\left\|\mathbf{z}_{e,k}^{(i,j)}-\mathbf{c}_{m}\right\|_{2}^{2}
$$
 
$$
\displaystyle\mathbf{z}_{q,k}^{(i,j)}
$$
 
$$
\displaystyle=\mathbf{c}_{q_{k}^{(i,j)}}
$$

Here, $q_{k}^{(i,j)}\in\{1,...,M\}$ denotes the quantization index, and $\mathbf{z}_{q,k}^{(i,j)}\in\mathbb{R}^{d}$ represents the quantized embedding.

Training Objective. To train the vector quantization module, we employ a codebook loss and a commitment loss:

$$
\displaystyle\mathcal{L}_{cb}
$$
 
$$
\displaystyle=\left\|\mathbf{z}_{q,k}^{(i,j)}-\text{sg}[\mathbf{z}_{e,k}^{(i,j)}]\right\|_{2}^{2}
$$
 
$$
\displaystyle\mathcal{L}_{cm}
$$
 
$$
\displaystyle=\left\|\text{sg}[\mathbf{z}_{q,k}^{(i,j)}]-\mathbf{z}_{e,k}^{(i,j)}\right\|_{2}^{2}
$$

where $\text{sg}[\cdot]$ denotes the stop-gradient operator. The codebook loss updates the codebook entries toward the encoder outputs, while the commitment loss encourages the encoder outputs to remain close to the selected codebook entries. The quantized embeddings are subsequently passed through a decoder $h_{k}(\cdot)$ to reconstruct the original patches, yielding the following reconstruction loss:

$$
\mathcal{L}_{rec}=\left\|\mathbf{P}_{k}^{(i,j)}-h_{k}(\mathbf{z}_{q,k}^{(i,j)})\right\|_{2}^{2}
$$

The overall training objective combines the reconstruction loss with the VQ losses as follows:

$$
\mathcal{L}_{total}=\mathcal{L}_{rec}+\alpha\mathcal{L}_{cb}+\beta\mathcal{L}_{cm}
$$

where $\alpha$ and $\beta$ are weighting coefficients for the respective VQ loss terms.

Coreset Memory Bank. After training, we construct a coreset memory bank $\mathcal{M}$ by collecting the codebook entries that are activated by the training data:

$$
\mathcal{M}=\bigcup_{k=1}^{K}\left\{\mathbf{c}_{m}\mid m\in\mathcal{A}_{k}\right\}
$$

where $\mathcal{A}_{k}\subseteq\{1,...,M\}$ denotes the set of codebook indices activated at scale $k$ during training.

### 3.3 Anomaly Scoring

At inference time, anomalies are detected using two complementary scores: (1) a memory score measuring distances to the memory bank and (2) a quantization score capturing codebook quantization error.

![Refer to caption](https://arxiv.org/html/2602.01635v2/x2.png)

Refer to caption

Memory Score with Local Scaling Distance. The memory score measures the deviation of an inference embedding from learned normal patterns. Since the memory bank is constructed from codebook entries activated during training, the sample density can be highly non-uniform. As a result, relying solely on global distance may cause false positives in sparse regions and false negatives in dense regions (Figure 2).

To address this issue, we introduce a local scaling distance that jointly accounts for distance and local density. Leveraging the fact that k-NN distances serve as a proxy for local neighborhood density [^39], we define the local scale $\sigma_{i}$ of each memory sample $\mathbf{m}_{i}\in\mathcal{M}$ as the median squared distance to its k-nearest neighbors. Here, $\mathcal{N}_{n}(\cdot)$ denotes the set of $n$ nearest neighbors:

$$
\sigma_{i}=\mathrm{median}\left(\left\{\left\|\mathbf{m}_{i}-\mathbf{m}_{j}\right\|_{2}^{2}\;\middle|\;\mathbf{m}_{j}\in\mathcal{N}_{n}(\mathbf{m}_{i})\right\}\right)
$$

Similarly, the local scale $\sigma_{q}$ of a quantized query embedding $\mathbf{z}_{q}$ is computed as the median squared distance to its k-nearest neighbors in the memory bank:

$$
\sigma_{q}=\mathrm{median}\left(\left\{\left\|\mathbf{z}_{q}-\mathbf{m}_{j}\right\|_{2}^{2}\;\middle|\;\mathbf{m}_{j}\in\mathcal{N}_{n}(\mathbf{z}_{q})\right\}\right)
$$

The local scaling distance normalizes the squared Euclidean distance between the quantized query embedding $\mathbf{z}_{q}$ and a neighbor $\mathbf{m}_{j}$ by the average of their local scales:

$$
d_{\text{local}}(\mathbf{z}_{q},\mathbf{m}_{j})=\frac{\|\mathbf{z}_{q}-\mathbf{m}_{j}\|_{2}^{2}}{(\sigma_{q}+\sigma_{j})/2+\epsilon}
$$

The proposed local scaling distance enables sensitivity to small deviations in dense regions while tolerating larger deviations in sparse regions, achieving density-adaptive anomaly detection. Finally, for each patch scale $k$, we compute the average local scaling distance over the k-nearest neighbors and then average across all scales to obtain the memory score $S_{\text{mem}}\in\mathbb{R}^{L}$:

$$
S_{\text{mem}}^{(t)}=\frac{1}{K}\sum_{k=1}^{K}\frac{1}{n}\sum_{r=1}^{n}d_{\text{local}}\!\left(\mathbf{z}_{q,k}^{(t)},\mathbf{m}_{r}^{(k,t)}\right)
$$

Here, $k$ indexes the patch scale, $t$ denotes the temporal position of the query embedding, and $\{\mathbf{m}_{r}^{(k,t)}\}_{r=1}^{n}$ are the $n$ nearest memory samples to $\mathbf{z}_{q,k}^{(t)}$ at scale $k$.

Quantization Score. The quantization score measures how well the encoder output $\mathbf{z}_{e}$ is represented by the codebook. Since the codebook is trained solely on normal data, normal patterns yield low quantization errors, whereas anomalous patterns are expected to exhibit higher errors due to the lack of similar codebook entries:

$$
S_{\text{quant}}^{(t)}=\frac{1}{K}\sum_{k=1}^{K}\left\|\mathbf{z}_{e,k}^{(t)}-\mathbf{z}_{q,k}^{(t)}\right\|_{2}
$$

Here, $t$ denotes the temporal position corresponding to the patch embedding. As with the memory score, the quantization score is averaged across all scales to obtain $S_{\text{quant}}\in\mathbb{R}^{L}$.

Deviation-based Variable Selection. In multivariate time series, aggregating anomaly scores across all variables can be unreliable due to redundant or noisy channels. We therefore perform temporal position-wise variable selection based on standardized temporal deviations, retaining variables with stable behavior while excluding unstable ones. Further implementation details are provided in Appendix B.

Score Normalization. Since the memory score and quantization score have different physical meanings and scales, normalization is required for effective combination. However, simple min–max normalization reflects only local statistics and may distort the relative degree of anomaly across the entire time series. To address this, we adopt Exponential Moving Average based min–max normalization:

$$
\displaystyle\mu_{\min}^{(t)}
$$
 
$$
\displaystyle=\gamma\cdot\mu_{\min}^{(t-1)}+(1-\gamma)\cdot\min(S^{(t)})
$$
 
$$
\displaystyle\mu_{\max}^{(t)}
$$
 
$$
\displaystyle=\gamma\cdot\mu_{\max}^{(t-1)}+(1-\gamma)\cdot\max(S^{(t)})
$$

where $\gamma$ denotes the momentum. The normalized score $\tilde{S}^{(t)}=(S^{(t)}-\mu_{\min}^{(t)})/(\mu_{\max}^{(t)}-\mu_{\min}^{(t)}+\epsilon)$ accumulates past statistics while gradually incorporating new information, enabling consistent comparison of anomaly scores over time.

Anomaly Score Aggregation. The final anomaly score is computed as a weighted combination of the two normalized scores:

$$
S^{(t)}=(1-\lambda)\cdot\tilde{S}_{\text{mem}}^{(t)}+\lambda\cdot\tilde{S}_{\text{quant}}^{(t)}
$$

where $\lambda$ is a weighting hyperparameter. The memory score captures the explicit distance to learned normal patterns, while the quantization score measures compressibility into the normal pattern space, allowing the two scores to complement each other in detecting anomalies.

### 3.4 Online Codebook Adaptation

To address distribution shifts between training and inference, we propose Online Codebook Adaptation, which dynamically updates the codebook at inference time.

A key challenge of Test-Time Adaptation (TTA) in anomaly detection lies in distinguishing normal from anomalous samples in unlabeled streaming data. Existing methods often rely on threshold-based filtering, which is sensitive to hyperparameter choices and susceptible to contamination from misclassified samples. In contrast, our approach leverages the fact that the codebook is trained solely on normal data to identify reliable normal samples without thresholds, and selectively updates the codebook to prevent contamination from anomalous data.

Codebook Activation Set. In unsupervised anomaly detection, training data are assumed to be normal. Hence, codebook entries activated during training can be regarded as prototypes of normal patterns. We define the set of activated codebook entries as:

$$
\mathcal{C}_{\text{seen}}=\left\{\mathbf{c}_{m}\mid\exists(i,j,k)\in\mathcal{D}_{\text{train}}:q_{k}^{(i,j)}=m\right\}
$$

Activation-based Pseudo-labeling. At inference time, we generate pseudo-labels based on whether the codebook entry corresponding to the quantization index $\hat{q}_{k}^{(i,j)}$ of a test sample was activated during training:

$$
\tilde{y}_{k}^{(i,j)}=\begin{cases}0&\text{if }\mathbf{c}_{\hat{q}_{k}^{(i,j)}}\in\mathcal{C}_{\text{seen}}\;(\text{Normal})\\
1&\text{otherwise}\;(\text{Abnormal})\end{cases}
$$

This enables normal sample identification without threshold tuning. Moreover, both unseen anomalous patterns and noise tend to be quantized to entries outside $\mathcal{C}_{\text{seen}}$.

Contrastive Codebook Adaptation. To encourage separation between normal and anomalous representations using pseudo-labels, we employ supervised contrastive learning. Given encoder outputs $\{\mathbf{z}_{e}^{(n)}\}$ and pseudo-labels $\{\tilde{y}^{(n)}\}$ from a test batch, the contrastive loss is defined as:

$$
\mathcal{L}_{\text{con}}=-\sum_{n}\frac{1}{|P(n)|}\sum_{p\in P(n)}\log\frac{e^{s_{np}/\tau}}{\sum_{a\neq n}e^{s_{na}/\tau}}
$$

where $s_{np}=\text{sim}(\mathbf{z}_{e}^{(n)},\mathbf{z}_{e}^{(p)})$ denotes cosine similarity, $P(n)=\{p\mid\tilde{y}^{(p)}=\tilde{y}^{(n)},p\neq n\}$ is the positive set with the same pseudo-label, and $\tau$ is a temperature parameter. The overall TTA objective is:

$$
\mathcal{L}_{\text{TTA}}=\frac{1}{|\mathcal{I}_{\text{norm}}|}\sum_{n\in\mathcal{I}_{\text{norm}}}\mathcal{L}_{\text{total}}^{(n)}+\gamma\mathcal{L}_{\text{con}}
$$

where $\mathcal{I}_{\text{norm}}=\{n\mid\tilde{y}^{(n)}=0\}$ denotes the set of normal samples. Notably, $\mathcal{L}_{\text{total}}$ is applied only to normal samples to prevent codebook contamination.

Index-wise Coreset Update. Once the codebook is updated, the coreset memory bank is directly updated using the quantization indices described in Section 3.2:

$$
\mathcal{M}^{\prime}=\left\{\mathbf{c}_{m}^{\text{updated}}\mid m\in\mathcal{A}_{\text{train}}\right\}
$$

where $\mathcal{A}_{\text{train}}=\bigcup_{k=1}^{K}\mathcal{A}_{k}$ denotes the set of all codebook indices activated during training.

To prevent test data leakage, we adopt an inference-then-train strategy during TTA. For each test batch, anomaly scores are first computed with the current model state, after which the model is adapted using the same batch. This ensures that the adapted model affects only subsequent batches, enabling fair evaluation.

## 4 Experiments

### 4.1 Experiment Setups

Datasets. We conduct experiments on five benchmark datasets widely used in time series anomaly detection, covering various domains: PSM [^1], SWaT [^9], SMAP [^11], MSL [^11], and WADI [^2]. Dataset statistics are summarized in Table 3, with detailed descriptions provided in Appendix C.

Baselines. To validate the effectiveness of the proposed method, we compare against seven baseline models: LSTM-AE [^24], LSTM-VAE [^29], USAD [^3], AnomalyTransformer [^37], VTT [^12], CATCH [^36], and D3R [^34]. Detailed descriptions of these baselines are provided in Appendix D.

Metrics. To evaluate anomaly detection performance from multiple perspectives, we employ both point-wise and range-based metrics. For point-wise evaluation, we report F1-score, AUC-ROC, and AUC-PR. The F1-score is divided into F1(K=0) and F1(K=100) depending on whether point adjustment is applied [^15], which considers an anomaly segment as correctly detected if any point within the segment is detected. For range-based evaluation, we use Affiliation F1 [^10], R-AUC-ROC, R-AUC-PR, VUS-ROC, and VUS-PR [^28]. For metrics requiring a threshold, best-F1 threshold search is applied.

Implementation Details. The input sequence length is set to 100 with a stride of 50. For multi-scale patch encoding, we use patch sizes $\{2,4,6\}$ with corresponding strides $\{1,2,3\}$. Further details are provided in Appendix E.

### 4.2 Main Results

Table 1: Main experimental results on five anomaly detection benchmarks. We report point-wise metrics and range-based metrics to evaluate detection robustness. Bold indicates the best score, and underline indicates the second best.

<table><tbody><tr><th rowspan="2">Dataset</th><th rowspan="2">Method</th><td colspan="9">Metric</td></tr><tr><td>F1(K=0)</td><td>F1(K=100)</td><td>Aff-F1</td><td>AUC-ROC</td><td>AUC-PR</td><td>R-AUC-ROC</td><td>R-AUC-PR</td><td>VUS-ROC</td><td>VUS-PR</td></tr><tr><th rowspan="9">PSM</th><th>CATCH</th><td>97.70</td><td>42.38</td><td>60.77</td><td>62.07</td><td>41.13</td><td>70.46</td><td>53.01</td><td>69.50</td><td>51.74</td></tr><tr><th>D3R</th><td>93.67</td><td>44.99</td><td>57.39</td><td>69.17</td><td>48.97</td><td>69.49</td><td>50.59</td><td>69.18</td><td>50.36</td></tr><tr><th>VTT</th><td>92.81</td><td>48.35</td><td>56.13</td><td>76.94</td><td>49.93</td><td>76.23</td><td>52.95</td><td>76.01</td><td>52.69</td></tr><tr><th>AT</th><td>97.43</td><td>2.20</td><td>65.19</td><td>50.48</td><td>27.88</td><td>52.33</td><td>34.81</td><td>52.13</td><td>34.61</td></tr><tr><th>LSTM-AE</th><td>93.41</td><td>45.54</td><td>53.37</td><td>69.70</td><td>50.76</td><td>70.26</td><td>51.35</td><td>69.94</td><td>51.15</td></tr><tr><th>LSTM-VAE</th><td>93.12</td><td>44.72</td><td>59.38</td><td>66.14</td><td>47.28</td><td>66.16</td><td>48.46</td><td>65.95</td><td>48.27</td></tr><tr><th>USAD</th><td>91.78</td><td>43.44</td><td>56.04</td><td>64.84</td><td>44.18</td><td>65.71</td><td>46.43</td><td>65.40</td><td>46.21</td></tr><tr><th>COMET (w/ TTA)</th><td>95.30</td><td>60.26</td><td>72.00</td><td>79.54</td><td>60.01</td><td>79.56</td><td>62.21</td><td>77.92</td><td>61.20</td></tr><tr><th>COMET (w/o TTA)</th><td>95.37</td><td>60.13</td><td>71.80</td><td>79.15</td><td>59.10</td><td>79.18</td><td>61.45</td><td>77.50</td><td>60.42</td></tr><tr><th rowspan="9">SWaT</th><th>CATCH</th><td>91.35</td><td>10.11</td><td>69.12</td><td>23.78</td><td>8.66</td><td>27.34</td><td>9.95</td><td>27.17</td><td>9.86</td></tr><tr><th>D3R</th><td>87.10</td><td>76.65</td><td>71.74</td><td>83.16</td><td>72.95</td><td>65.27</td><td>47.41</td><td>65.35</td><td>47.57</td></tr><tr><th>VTT</th><td>86.13</td><td>77.05</td><td>60.11</td><td>81.42</td><td>71.08</td><td>58.43</td><td>43.62</td><td>58.48</td><td>43.74</td></tr><tr><th>AT</th><td>95.98</td><td>3.91</td><td>61.05</td><td>49.29</td><td>13.06</td><td>49.63</td><td>12.35</td><td>49.63</td><td>12.36</td></tr><tr><th>LSTM-AE</th><td>85.39</td><td>76.73</td><td>72.46</td><td>81.95</td><td>72.56</td><td>63.95</td><td>46.72</td><td>64.01</td><td>46.75</td></tr><tr><th>LSTM-VAE</th><td>85.35</td><td>76.57</td><td>71.84</td><td>81.97</td><td>72.73</td><td>62.02</td><td>45.54</td><td>62.30</td><td>45.97</td></tr><tr><th>USAD</th><td>85.36</td><td>76.57</td><td>71.78</td><td>81.98</td><td>72.74</td><td>62.16</td><td>45.65</td><td>62.42</td><td>46.08</td></tr><tr><th>COMET (w/ TTA)</th><td>91.62</td><td>75.24</td><td>74.14</td><td>85.50</td><td>74.48</td><td>85.09</td><td>66.42</td><td>85.06</td><td>66.48</td></tr><tr><th>COMET (w/o TTA)</th><td>91.38</td><td>75.06</td><td>72.93</td><td>85.48</td><td>74.33</td><td>84.85</td><td>65.46</td><td>84.82</td><td>65.50</td></tr><tr><th rowspan="9">SMAP</th><th>CATCH</th><td>70.44</td><td>12.05</td><td>50.85</td><td>43.09</td><td>11.90</td><td>44.65</td><td>12.88</td><td>44.42</td><td>12.83</td></tr><tr><th>D3R</th><td>69.65</td><td>10.29</td><td>53.55</td><td>47.21</td><td>11.85</td><td>50.04</td><td>13.15</td><td>49.97</td><td>13.15</td></tr><tr><th>VTT</th><td>70.99</td><td>10.96</td><td>50.78</td><td>46.59</td><td>11.73</td><td>49.39</td><td>12.81</td><td>49.27</td><td>12.82</td></tr><tr><th>AT</th><td>96.42</td><td>2.21</td><td>67.54</td><td>50.03</td><td>12.89</td><td>50.44</td><td>14.16</td><td>50.43</td><td>14.20</td></tr><tr><th>LSTM-AE</th><td>69.87</td><td>8.56</td><td>53.55</td><td>39.10</td><td>10.31</td><td>42.22</td><td>11.38</td><td>42.15</td><td>11.40</td></tr><tr><th>LSTM-VAE</th><td>69.71</td><td>6.91</td><td>53.55</td><td>41.46</td><td>10.63</td><td>44.34</td><td>11.76</td><td>44.19</td><td>11.76</td></tr><tr><th>USAD</th><td>69.18</td><td>7.01</td><td>53.55</td><td>38.82</td><td>10.25</td><td>41.79</td><td>11.36</td><td>41.69</td><td>11.36</td></tr><tr><th>COMET (w/ TTA)</th><td>82.30</td><td>25.28</td><td>68.24</td><td>59.17</td><td>16.20</td><td>59.43</td><td>17.59</td><td>59.20</td><td>17.54</td></tr><tr><th>COMET (w/o TTA)</th><td>82.46</td><td>25.40</td><td>68.01</td><td>59.06</td><td>16.17</td><td>58.69</td><td>17.51</td><td>58.52</td><td>17.46</td></tr><tr><th rowspan="9">MSL</th><th>CATCH</th><td>81.79</td><td>18.08</td><td>55.95</td><td>60.41</td><td>14.38</td><td>67.78</td><td>20.78</td><td>67.05</td><td>20.48</td></tr><tr><th>D3R</th><td>87.81</td><td>18.37</td><td>62.56</td><td>53.88</td><td>14.75</td><td>61.91</td><td>19.97</td><td>61.42</td><td>19.72</td></tr><tr><th>VTT</th><td>89.01</td><td>13.47</td><td>59.55</td><td>59.01</td><td>15.01</td><td>66.03</td><td>20.51</td><td>65.38</td><td>20.25</td></tr><tr><th>AT</th><td>93.84</td><td>2.88</td><td>66.49</td><td>50.65</td><td>10.62</td><td>52.24</td><td>14.87</td><td>52.23</td><td>14.99</td></tr><tr><th>LSTM-AE</th><td>88.30</td><td>17.62</td><td>62.60</td><td>53.61</td><td>14.05</td><td>60.18</td><td>18.33</td><td>59.77</td><td>18.20</td></tr><tr><th>LSTM-VAE</th><td>88.27</td><td>17.62</td><td>62.60</td><td>53.22</td><td>13.97</td><td>59.83</td><td>18.18</td><td>59.44</td><td>18.06</td></tr><tr><th>USAD</th><td>88.27</td><td>17.62</td><td>62.60</td><td>55.93</td><td>14.41</td><td>62.67</td><td>18.80</td><td>62.13</td><td>18.64</td></tr><tr><th>COMET (w/ TTA)</th><td>87.44</td><td>26.28</td><td>71.04</td><td>65.40</td><td>16.56</td><td>69.90</td><td>22.67</td><td>68.87</td><td>22.25</td></tr><tr><th>COMET (w/o TTA)</th><td>87.27</td><td>26.44</td><td>69.40</td><td>65.49</td><td>16.63</td><td>69.27</td><td>22.67</td><td>68.39</td><td>22.27</td></tr><tr><th rowspan="9">WADI</th><th>CATCH</th><td>45.28</td><td>9.46</td><td>3.74</td><td>47.03</td><td>5.36</td><td>50.44</td><td>6.24</td><td>50.29</td><td>6.24</td></tr><tr><th>D3R</th><td>37.87</td><td>6.80</td><td>52.79</td><td>47.93</td><td>5.16</td><td>45.98</td><td>5.91</td><td>45.72</td><td>5.91</td></tr><tr><th>VTT</th><td>49.52</td><td>7.03</td><td>52.72</td><td>50.10</td><td>5.30</td><td>47.46</td><td>6.14</td><td>47.22</td><td>6.14</td></tr><tr><th>AT</th><td>91.10</td><td>2.53</td><td>53.81</td><td>50.93</td><td>5.77</td><td>51.02</td><td>6.69</td><td>51.07</td><td>6.72</td></tr><tr><th>LSTM-AE</th><td>33.34</td><td>7.06</td><td>52.72</td><td>47.19</td><td>4.96</td><td>45.47</td><td>5.76</td><td>45.24</td><td>5.77</td></tr><tr><th>LSTM-VAE</th><td>31.81</td><td>7.07</td><td>52.72</td><td>49.51</td><td>5.12</td><td>46.62</td><td>5.93</td><td>46.42</td><td>5.93</td></tr><tr><th>USAD</th><td>31.81</td><td>7.07</td><td>52.72</td><td>48.92</td><td>5.07</td><td>46.27</td><td>5.86</td><td>46.04</td><td>5.86</td></tr><tr><th>COMET (w/ TTA)</th><td>74.07</td><td>18.92</td><td>72.43</td><td>61.79</td><td>9.23</td><td>64.17</td><td>10.18</td><td>64.13</td><td>10.20</td></tr><tr><th>COMET (w/o TTA)</th><td>73.81</td><td>18.98</td><td>72.63</td><td>61.19</td><td>9.14</td><td>63.64</td><td>10.10</td><td>63.57</td><td>10.11</td></tr></tbody></table>

Table 1 summarizes the quantitative comparison with state-of-the-art baselines across five benchmark datasets. Compared against baseline methods without test-time adaptation, COMET (w/o TTA) achieves the best performance in 39 out of 45 evaluation metrics, demonstrating its effectiveness across diverse settings. Notably, COMET shows consistent gains on range-based metrics, which are considered more robust for time series anomaly detection. While several baselines achieve high F1(K=0) scores due to the Point Adjustment protocol, their performance degrades on unadjusted and range-based metrics. In contrast, COMET maintains balanced performance across all metrics, indicating its ability to accurately detect continuous anomalous segments rather than isolated points.

Applying test-time adaptation (TTA) further improves performance in 35 out of 45 evaluation metrics, validating the effectiveness of the proposed Online Codebook Adaptation strategy. Unlike training, where multiple overlapping instances can be generated by adjusting the stride, the streaming nature of inference data constrains the information available for adaptation. Nevertheless, COMET exhibits improvements on the majority of metrics, demonstrating effective adaptation to distribution shifts even under limited test-time information.

### 4.3 Ablation Study

Table 2: Ablation study results on COMET, averaged across all five benchmark datasets. Detailed ablation results for each dataset are provided in Appendix H. Bold indicates the best score, and underlined indicates the second best.

<table><tbody><tr><th rowspan="2">Category</th><th rowspan="2">Method</th><td colspan="9">Metric</td></tr><tr><td>F1(K=0)</td><td>F1(K=100)</td><td>Aff-F1</td><td>AUC-ROC</td><td>AUC-PR</td><td>R-AUC-ROC</td><td>R-AUC-PR</td><td>VUS-ROC</td><td>VUS-PR</td></tr><tr><th>Architecture</th><th>w/o Multi-Scale</th><td>76.45</td><td>36.31</td><td>69.48</td><td>63.17</td><td>30.78</td><td>63.62</td><td>28.66</td><td>63.25</td><td>28.59</td></tr><tr><th rowspan="5">Scoring</th><th>w/o Quant Score</th><td>83.70</td><td>35.74</td><td>71.00</td><td>67.84</td><td>24.77</td><td>69.27</td><td>26.77</td><td>68.69</td><td>26.55</td></tr><tr><th>w/o Memory Score</th><td>81.55</td><td>37.96</td><td>69.19</td><td>65.62</td><td>34.65</td><td>64.85</td><td>32.88</td><td>64.48</td><td>32.78</td></tr><tr><th>w/o Local Scaling</th><td>83.92</td><td>39.31</td><td>71.00</td><td>68.85</td><td>34.08</td><td>70.64</td><td>34.52</td><td>70.17</td><td>34.27</td></tr><tr><th>w/o Variable Selection</th><td>83.65</td><td>38.54</td><td>70.97</td><td>68.16</td><td>34.31</td><td>68.50</td><td>34.47</td><td>67.94</td><td>34.31</td></tr><tr><th>w/o Normalization</th><td>76.70</td><td>31.97</td><td>58.62</td><td>58.48</td><td>29.95</td><td>58.12</td><td>28.10</td><td>57.83</td><td>27.92</td></tr><tr><th rowspan="2">TTA</th><th>w/o TTA</th><td>86.06</td><td>41.20</td><td>70.95</td><td>70.07</td><td>35.07</td><td>71.13</td><td>35.44</td><td>70.56</td><td>35.15</td></tr><tr><th>w/o Contrastive</th><td>86.04</td><td>41.20</td><td>71.63</td><td>70.20</td><td>35.31</td><td>71.49</td><td>35.71</td><td>70.92</td><td>35.42</td></tr><tr><th colspan="2">COMET (Full)</th><td>86.15</td><td>41.20</td><td>71.57</td><td>70.28</td><td>35.30</td><td>71.63</td><td>35.81</td><td>71.04</td><td>35.53</td></tr></tbody></table>

To analyze the contribution of each component, we conduct an ablation study reported in Table 2. Removing multi-scale patch encoding and using a single scale with patch size 6 and stride 3 leads to performance degradation, highlighting the importance of capturing temporal patterns at multiple resolutions. Within the scoring module, removing either the quantization score or the memory score degrades performance, with the memory score being more critical for PR-based metrics and the quantization score contributing more to ROC-based metrics, confirming their complementary roles. Removing variable selection also causes a consistent performance drop, indicating that filtering unstable variables improves the robustness of anomaly scoring. Replacing local scaling distance with standard euclidean distance and removing score normalization both result in notable performance drops, demonstrating the importance of density-adaptive distance measurement and score aggregation. Removing test-time adaptation leads to overall performance degradation, confirming its effectiveness in handling distribution shifts at inference. In addition, disabling contrastive learning during TTA causes further performance drops, indicating that contrastive objectives improve adaptation by better separating normal and anomalous representations.

### 4.4 Analysis

![Refer to caption](https://arxiv.org/html/2602.01635v2/x3.png)

Refer to caption

Efficiency Analysis. Figure 3 compares the Affiliation F1-score, training time, and number of parameters for each model on the PSM dataset. With only 567K parameters, COMET achieves an Affiliation F1-score of 71.9%, outperforming larger models while using approximately 1.2% of the parameters of CATCH (48.6M). This efficiency stems from employing lightweight linear layers and a codebook-based design rather than complex Transformer architectures. The training time of COMET is 117.8 seconds, faster than VTT (335.7 seconds) and D3R (196.6 seconds), though slower than some models due to per-scale forward passes required by multi-scale patching. Overall, COMET achieves a favorable trade-off between performance and parameter efficiency, as further illustrated by additional visualizations for all benchmark datasets in Appendix F.

![Refer to caption](https://arxiv.org/html/2602.01635v2/x4.png)

Refer to caption

Embedding Space Analysis. Figure 4 presents UMAP visualizations of patch embeddings and coresets entries on the SWaT dataset. The codebook entries are distributed within regions corresponding to Train Normal and Test Normal samples, indicating that representative coresets of normal patterns are effectively learned via vector quantization. Test Anomaly samples are scattered across the embedding space but reside in regions separated from the codebook entries, supporting the feasibility of anomaly detection based on memory distance and quantization error. Notably, different patch scales exhibit complementary embedding distributions, with smaller patches capturing localized patterns and larger patches spanning broader regions, confirming the benefit of multi-scale representations.

## 5 Conclusion

In this paper, we propose COMET for time series anomaly detection. COMET captures temporal dependencies and multivariate correlations across diverse temporal scales through Multi-scale Patch Encoding and learns representative normal patterns via Vector-Quantized Coreset, enabling anomaly detection by combining memory distance and quantization error. Furthermore, COMET dynamically adapts to distribution shifts at inference time through Online Codebook Adaptation, which leverages pseudo-labeling based on codebook activation and contrastive learning. Experimental results on five benchmark datasets demonstrate that COMET achieves the best performance in 39 out of 45 evaluation metrics while exhibiting strong efficiency with a small number of parameters.

## Impact Statement

Reliable identification of anomalous patterns is essential in large-scale information systems, where undetected or mischaracterized abnormal behaviors can lead to operational failures or significant economic losses. Advances in time-series anomaly detection can contribute to improving the stability, safety, and reliability of real-world systems, including industrial monitoring environments, cyber-physical infrastructures, and other data-driven operational settings. This work is intended to support the development of more reliable monitoring and analysis capabilities in such domains. The proposed research is not designed to disadvantage or target any specific institution, organization, or group, and it does not raise ethical concerns beyond those commonly associated with data-driven monitoring and analytical systems.

## References

## Appendix A Algorithm Details

In this appendix, we provide the detailed algorithmic procedures of COMET. Algorithm 1 describes the training phase, which consists of model optimization and coreset construction. Algorithm 2 describes the inference phase with online codebook adaptation, where anomaly scores are computed before adaptation to prevent test data leakage.

Algorithm 1 COMET Training

 Input: Training data $\mathcal{D}_{\text{train}}$, patch sizes $\mathcal{P}=\{p_{1},\dots,p_{K}\}$, strides $\mathcal{S}=\{s_{1},\dots,s_{K}\}$, codebook size $M$

 Output: Encoder $f_{\theta}$, decoder $h_{\phi}$, codebooks $\{\mathbf{C}_{k}\}_{k=1}^{K}$, memory bank $\mathcal{M}$

 // Phase 1: Model Training

 for each epoch do

  for each batch $\mathbf{X}\in\mathcal{D}_{\text{train}}$ do

   for each scale $k=1$ to $K$ do

    Extract patches $\mathbf{P}_{k}$ with size $p_{k}$ and stride $s_{k}$

    Encode patches: $\mathbf{z}_{e,k}^{(i,j)}\leftarrow f_{\theta}^{(k)}(\mathbf{P}_{k}^{(i,j)})$

    Quantize embeddings:

      $q_{k}^{(i,j)}\leftarrow\arg\min_{m}\|\mathbf{z}_{e,k}^{(i,j)}-\mathbf{c}_{k,m}\|_{2}^{2}$       $\mathbf{z}_{q,k}^{(i,j)}\leftarrow\mathbf{c}_{k,q_{k}^{(i,j)}}$

    Decode patches: $\hat{\mathbf{P}}_{k}^{(i,j)}\leftarrow h_{\phi}^{(k)}(\mathbf{z}_{q,k}^{(i,j)})$

   end for

   Compute $\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{rec}}+\alpha\mathcal{L}_{\text{cb}}+\beta\mathcal{L}_{\text{cm}}$

   Update $\theta,\phi,\{\mathbf{C}_{k}\}$ via gradient descent

  end for

 end for

 // Phase 2: Coreset Construction

 Initialize activated index set $\mathcal{A}_{\text{train}}\leftarrow\emptyset$

 for each batch $\mathbf{X}\in\mathcal{D}_{\text{train}}$ do

  for each scale $k=1$ to $K$ do

   Extract patches $\mathbf{P}_{k}$ with size $p_{k}$ and stride $s_{k}$

   Encode patches: $\mathbf{z}_{e,k}^{(i,j)}\leftarrow f_{\theta}^{(k)}(\mathbf{P}_{k}^{(i,j)})$

   Quantize embeddings:

     $q_{k}^{(i,j)}\leftarrow\arg\min_{m}\|\mathbf{z}_{e,k}^{(i,j)}-\mathbf{c}_{k,m}\|_{2}^{2}$     $\mathcal{A}_{\text{train}}\leftarrow\mathcal{A}_{\text{train}}\cup\{(k,q_{k}^{(i,j)})\}$

  end for

 end for

 Construct memory bank $\mathcal{M}\leftarrow\{\mathbf{c}_{k,m}\mid(k,m)\in\mathcal{A}_{\text{train}}\}$

 Compute local scales $\{\sigma_{i}\}$ for all $\mathbf{m}_{i}\in\mathcal{M}$

The training procedure (Algorithm 1) consists of two phases. In the first phase, the model learns to encode patches into quantized representations by minimizing the combination of reconstruction loss, codebook loss, and commitment loss. In the second phase, the coreset memory bank is constructed by collecting all codebook entries activated during training, along with their local scales for density-adaptive distance computation.

Algorithm 2 COMET Inference with Online Codebook Adaptation

 Input: Test stream $\{\mathbf{X}^{(t)}\}$, trained model $(f_{\theta},h_{\phi},\{\mathbf{C}_{k}\})$, memory bank $\mathcal{M}$, activated indices $\mathcal{A}_{\text{train}}$

 Output: Anomaly scores $\{S^{(t)}\}$

 for each test batch $\mathbf{X}^{(t)}$ do

  // Step 1: Anomaly Scoring (before adaptation)

  for each scale $k=1$ to $K$ do

   Extract patches $\mathbf{P}_{k}$ with size $p_{k}$ and stride $s_{k}$

   Encode patches: $\mathbf{z}_{e,k}^{(t)}\leftarrow f_{\theta}^{(k)}(\mathbf{P}_{k}^{(t)})$

   Quantize embeddings:

     $\hat{q}_{k}^{(t)}\leftarrow\arg\min_{m}\|\mathbf{z}_{e,k}^{(t)}-\mathbf{c}_{k,m}\|_{2}^{2}$      $\mathbf{z}_{q,k}^{(t)}\leftarrow\mathbf{c}_{k,\hat{q}_{k}^{(t)}}$

   Compute query local scale $\sigma_{q}$ using $\mathcal{N}_{n}(\mathbf{z}_{q,k}^{(t)})$

   Compute memory score:

     $S_{\text{mem},k}^{(t)}\leftarrow\frac{1}{n}\sum_{r=1}^{n}d_{\text{local}}(\mathbf{z}_{q,k}^{(t)},\mathbf{m}_{r}^{(k,t)})$

   Compute quantization score:

     $S_{\text{quant},k}^{(t)}\leftarrow\|\mathbf{z}_{e,k}^{(t)}-\mathbf{z}_{q,k}^{(t)}\|_{2}$

  end for

  Aggregate across scales:

    $S_{\text{mem}}^{(t)}\leftarrow\frac{1}{K}\sum_{k}S_{\text{mem},k}^{(t)}$     $S_{\text{quant}}^{(t)}\leftarrow\frac{1}{K}\sum_{k}S_{\text{quant},k}^{(t)}$

  Apply EMA normalization and variable selection

  Compute final score $S^{(t)}\leftarrow(1-\lambda)\tilde{S}_{\text{mem}}^{(t)}+\lambda\tilde{S}_{\text{quant}}^{(t)}$

  // Step 2: Online Codebook Adaptation

  for each embedding $\mathbf{z}_{e,k}^{(t)}$ with index $\hat{q}_{k}^{(t)}$ do

   if $(k,\hat{q}_{k}^{(t)})\in\mathcal{A}_{\text{train}}$ then

     $\tilde{y}^{(t)}\leftarrow 0$ {Normal}

   else

     $\tilde{y}^{(t)}\leftarrow 1$ {Abnormal}

   end if

  end for

   $\mathcal{I}_{\text{norm}}\leftarrow\{n\mid\tilde{y}^{(n)}=0\}$

  Compute contrastive loss $\mathcal{L}_{\text{con}}$ using $\{\mathbf{z}_{e}^{(n)},\tilde{y}^{(n)}\}$

   $\mathcal{L}_{\text{TTA}}\leftarrow\frac{1}{|\mathcal{I}_{\text{norm}}|}\sum_{n\in\mathcal{I}_{\text{norm}}}\mathcal{L}_{\text{total}}^{(n)}+\gamma\mathcal{L}_{\text{con}}$

  Update $\theta,\phi,\{\mathbf{C}_{k}\}$ via gradient descent

  Update memory bank:

    $\mathcal{M}\leftarrow\{\mathbf{c}_{k,m}^{\text{updated}}\mid(k,m)\in\mathcal{A}_{\text{train}}\}$

  // Adaptation affects subsequent batches

 end for

The inference procedure (Algorithm 2) adopts an inference-then-train strategy to prevent test data leakage. For each test batch, anomaly scores are first computed using the current model state by combining memory distance and quantization error. Subsequently, the model is adapted using pseudo-labels generated from codebook activation patterns, where samples mapped to previously activated codebook entries are labeled as normal. The contrastive loss encourages separation between normal and abnormal representations, and the coreset is updated by directly replacing codebook entries using stored quantization indices.

## Appendix B Deviation-based Variable Selection

Here, the temporal index j corresponds to the time position t used in the main text. In this appendix, we provide detailed formulations for the deviation-based variable selection mechanism described in Section 3.3.

### B.1 Motivation

Given anomaly scores $\mathbf{S}\in\mathbb{R}^{D\times T}$ aligned to temporal positions, where $D$ denotes the number of variables and $T$ the number of temporal positions corresponding to patch embeddings at a given scale, a naive approach aggregates scores by uniformly averaging across all variables. However, this can dilute anomaly signals due to redundant or noisy variables. To address this issue, we propose a deviation-based variable selection strategy that selects reliable variables at each temporal position based on their standardized deviations. Additionally, to ensure stable coverage in practice, we always include the first variable ($i=1$) in the selected set.

### B.2 Standardized Deviation Computation

For each variable $i\in\{1,\dots,D\}$, we compute the temporal mean and standard deviation of its anomaly scores across all temporal positions:

$$
\mu^{(i)}=\frac{1}{T}\sum_{t=1}^{T}S_{t}^{(i)},\qquad\sigma^{(i)}=\sqrt{\frac{1}{T}\sum_{t=1}^{T}\left(S_{t}^{(i)}-\mu^{(i)}\right)^{2}}.
$$

The standardized deviation of variable $i$ at temporal position $t$ is then computed as:

$$
\delta_{t}^{(i)}=\frac{S_{t}^{(i)}-\mu^{(i)}}{\sigma^{(i)}+\epsilon},
$$

where $\epsilon$ is a small constant for numerical stability.

### B.3 Position-wise Variable Selection

At each temporal position $t$, we select a subset of variables $\mathcal{V}_{t}\subseteq\{1,\dots,D\}$ based on their absolute standardized deviation values. Variables with smaller absolute deviations exhibit more consistent temporal behavior and are therefore considered more reliable for anomaly scoring.

Percentile-based Selection. Given a percentile threshold $\rho\in[0,100]$, we compute the threshold value:

$$
\tau_{t}=Q_{\rho}\left(\left\{|\delta_{t}^{(i)}|\right\}_{i=1}^{D}\right),
$$

where $Q_{\rho}(\cdot)$ denotes the $\rho$ -th percentile. The selected variable set is defined as:

$$
\mathcal{V}_{t}=\left\{i\in\{1,\dots,D\}\mid|\delta_{t}^{(i)}|\leq\tau_{t}\right\}\cup\{1\}.
$$

Budget-constrained Selection. Alternatively, a fixed budget $B$ can be specified to select the $B$ most stable variables:

$$
\mathcal{V}_{t}=\underset{\mathcal{V}\subseteq\{1,\dots,D\},\,|\mathcal{V}|=B}{\arg\min}\sum_{i\in\mathcal{V}}|\delta_{t}^{(i)}|,
$$

where the first variable ($i=1$) is always included in $\mathcal{V}_{t}$.

### B.4 Score Aggregation

The final anomaly score at temporal position $t$ is computed by averaging the scores over the selected variables:

$$
\tilde{S}_{t}=\frac{1}{|\mathcal{V}_{t}|}\sum_{i\in\mathcal{V}_{t}}S_{t}^{(i)}.
$$

This selective aggregation ensures that anomaly scores are computed from variables exhibiting consistent temporal behavior, thereby improving robustness against noisy channels while preserving informative anomaly signals.

## Appendix C Dataset Descriptions

We conduct experiments on five benchmark datasets that are widely used in the time series anomaly detection. The statistics of each dataset are summarized in Table 3. For all datasets, the test split contains a mixture of normal and anomalous samples with ground-truth labels provided. For SWaT and WADI, the training data are collected during normal operation periods prior to attack scenarios and thus contain only normal samples, whereas the training data for SMAP, MSL, and PSM are provided without labels. For SMAP and MSL, which originally consist of multiple sub-datasets, we concatenate the training and test portions from all sub-datasets into a single time series to facilitate training. In our study, we reserve 10% of the training data as a validation set.

Table 3: Statistics of benchmark datasets. #TRAIN and #VALID denote the number of samples in training and validation sets, respectively, where the validation set is split from the original training data with a ratio of (0.1). ANOMALY (%) indicates the proportion of anomalous samples in the test set.

|  | PSM | SWaT | SMAP | MSL | WADI |
| --- | --- | --- | --- | --- | --- |
| Variables | 25 | 51 | 25 | 55 | 123 |
| #Train(0.9) | 116,805 | 445,500 | 124,184 | 52,473 | 1,088,595 |
| #Valid(0.1) | 12,979 | 49,500 | 13,820 | 5,844 | 120,956 |
| #Test | 87,841 | 449,919 | 435,826 | 73,729 | 172,801 |
| Anomaly (%) | 27.76 | 12.14 | 12.84 | 10.53 | 5.71 |

PSM (Pooled Server Metrics) [^1] is a dataset collected from internal server nodes at eBay, consisting of 25 server monitoring metrics such as CPU utilization, memory usage, and network traffic. It includes diverse anomalous patterns related to server failures and performance degradation.

SWaT (Secure Water Treatment) [^9] is an industrial control system dataset collected from a water treatment testbed at the Singapore University of Technology and Design (SUTD). It contains 51 sensor and actuator variables and includes anomalies arising from cyber-attack scenarios.

SMAP (Soil Moisture Active Passive) [^11] is a telemetry dataset collected from NASA’s soil moisture observation satellite. It consists of 25 variables and includes anomalous behaviors of spacecraft subsystems.

MSL (Mars Science Laboratory) [^11] is a sensor and actuator dataset collected from NASA’s Mars rover Curiosity. It consists of 55 variables and, together with SMAP, serves as a benchmark for anomaly detection in space systems released by NASA.

WADI (Water Distribution) [^2] is a dataset collected from a water distribution testbed at SUTD and can be regarded as an extension of SWaT. It includes 123 sensor variables and consists of 14 days of normal operation followed by 2 days of attack scenarios (with 15 attacks).

## Appendix D Baseline Models

This appendix provides brief descriptions of the baseline models used for comparison, covering reconstruction-based, probabilistic latent-variable, attention-based, frequency-domain, and diffusion-based approaches to multivariate time-series anomaly detection.

#### LSTM-AE.

LSTM-AE [^24], also known as EncDec-AD, is one of the earliest reconstruction-based approaches for time-series anomaly detection. It employs an LSTM encoder–decoder trained exclusively on normal data to learn temporal dependencies by compressing input sequences into latent representations and reconstructing them back to the input space. Anomalies are detected based on elevated reconstruction errors, under the assumption that abnormal patterns deviate from the learned normal dynamics.

#### LSTM-VAE.

LSTM-VAE [^29] extends LSTM-AE by incorporating variational inference to model uncertainty in the latent space. Instead of learning deterministic latent embeddings, the model learns a probabilistic latent distribution, allowing it to capture variability in normal temporal patterns. Anomaly scores are derived from reconstruction errors or reconstruction likelihoods, reflecting deviations from the learned latent distribution.

#### USAD.

USAD [^3] proposes a stable adversarial learning framework based on two autoencoders with asymmetric objectives. One autoencoder is trained to faithfully reconstruct normal data, while the other is optimized to amplify reconstruction discrepancies for anomalous patterns. This dual-objective design improves the separation between normal and abnormal samples compared to standard autoencoder-based reconstruction methods.

#### Anomaly Transformer.

Anomaly Transformer [^37] is an attention-based model that explicitly models temporal dependencies using Transformer architectures. It introduces the concept of *association discrepancy*, which measures the divergence between learned attention distributions and a predefined prior association. Anomalies are identified based on attention inconsistency rather than reconstruction error alone, enabling the detection of subtle temporal irregularities.

#### VTT.

Variable Temporal Transformer (VTT) [^12] focuses on modeling heterogeneous temporal dynamics and inter-variable dependencies in multivariate time series. It employs Transformer-based self-attention mechanisms to capture both temporal dependencies and correlations across variables. By learning variable-aware temporal representations, VTT enables anomaly detection based on deviations from the learned multivariate temporal structure.

#### CATCH.

CATCH [^36] is a frequency-aware anomaly detection framework designed to capture anomalies manifested in the spectral domain. It transforms time-series data into the frequency domain and applies frequency patching along with channel-aware modeling to learn spectral and temporal characteristics jointly. Anomaly detection is performed by leveraging discrepancies in the learned representations across time and frequency domains.

#### D3R.

D3R [^34] is a diffusion-based reconstruction framework designed for multivariate time-series anomaly detection under distributional instability and drift. It combines time-series decomposition with diffusion-based reconstruction to model stable normal dynamics while mitigating the effect of drift. Anomalies are detected based on reconstruction inconsistencies or denoising errors across diffusion steps, indicating deviations from learned normal trajectories.

## Appendix E Implementation Details

### E.1 Training Configuration

All experiments are conducted with a fixed random seed of 42 for reproducibility. The input sequence length $L$ is set to 100 with a stride of 50 for sliding window segmentation. We reserve 10% of the training data as a validation set. Models are trained for 20 epochs using the AdamW optimizer with a learning rate of $10^{-4}$ and weight decay of $5\times 10^{-4}$. The batch size is set to 128 for all datasets. We use Mean Squared Error (MSE) as the reconstruction loss.

### E.2 Model Architecture

For multi-scale patch encoding, we use patch sizes $\mathcal{P}=\{2,4,6\}$ with corresponding strides $\mathcal{S}=\{1,2,3\}$. The core encoder dimension $d_{c}$ is set to 64. The loss coefficients for codebook loss and commitment loss are set to $\alpha=1.0$ and $\beta=1.0$, respectively.

### E.3 Anomaly Scoring

For memory score computation, we use local scaling distance with $k=10$ nearest neighbors for density estimation and $n=10$ nearest neighbors for score aggregation. The score ratio $\lambda$ for combining memory score and quantization score is set to 0.5. For EMA-based score normalization, the momentum $\gamma$ is set to 0.75.

### E.4 Dataset-specific Hyperparameters

The codebook size $M$ and model dimension $d$ are tuned per dataset to accommodate varying data characteristics. Table 4 summarizes the dataset-specific hyperparameter settings.

Table 4: Dataset-specific hyperparameter settings.

| Dataset | Codebook Size ($M$) | Model Dimension ($d$) |
| --- | --- | --- |
| PSM | 128 | 256 |
| SWaT | 256 | 256 |
| SMAP | 128 | 128 |
| MSL | 256 | 128 |
| WADI | 32 | 64 |

## Appendix F Additional Visualizations

To further support the analysis of performance–efficiency trade-offs, we provide additional visualizations for all benchmark datasets in this appendix. These figures illustrate the relationship between anomaly detection performance and training time, with marker size indicating the number of model parameters. Consistent trends across datasets demonstrate that COMET maintains strong detection performance while using a relatively small number of parameters, complementing the quantitative results reported in the main text. Figure 5 presents these results for all datasets.

![Refer to caption](https://arxiv.org/html/2602.01635v2/x5.png)

(a) SWaT

## Appendix G Additional Ablation Results

This appendix presents detailed ablation study results of COMET on each individual benchmark dataset, including PSM (Table 5), SWaT (Table 6), SMAP (Table 7), MSL (Table 8), and WADI (Table 9). While the main paper reports results averaged across datasets for concise comparison, the per-dataset results provided here offer a more granular view of how each component contributes to performance under different data characteristics. When examined individually, using all modules does not always show the best performance in every case; however, we can confirm that it consistently demonstrates good average performance across all datasets.

Table 5: Ablation study results on COMET on the PSM dataset.

<table><tbody><tr><th rowspan="2">Category</th><th rowspan="2">Method</th><td colspan="9">Metric</td></tr><tr><td>F1(K=0)</td><td>F1(K=100)</td><td>Aff-F1</td><td>AUROC</td><td>AUPRC</td><td>R-AUC-ROC</td><td>R-AUC-PR</td><td>VUS-ROC</td><td>VUS-PR</td></tr><tr><th>Architecture</th><th>w/o Multi-Scale</th><td>87.60</td><td>46.49</td><td>71.33</td><td>67.04</td><td>48.75</td><td>63.50</td><td>46.25</td><td>62.71</td><td>45.83</td></tr><tr><th rowspan="5">Scoring</th><th>w/o Quant Score</th><td>96.78</td><td>49.30</td><td>70.78</td><td>68.31</td><td>44.68</td><td>71.03</td><td>49.38</td><td>69.33</td><td>48.61</td></tr><tr><th>w/o Memory Score</th><td>94.47</td><td>61.68</td><td>73.15</td><td>81.40</td><td>64.10</td><td>76.66</td><td>61.48</td><td>75.58</td><td>60.68</td></tr><tr><th>w/o Local Scaling NN</th><td>95.02</td><td>58.11</td><td>73.94</td><td>78.46</td><td>57.22</td><td>79.76</td><td>60.13</td><td>78.24</td><td>59.22</td></tr><tr><th>w/o Variable Selection</th><td>94.28</td><td>59.92</td><td>71.85</td><td>79.45</td><td>59.44</td><td>73.66</td><td>57.74</td><td>72.39</td><td>57.04</td></tr><tr><th>w/o Normalization</th><td>92.92</td><td>45.15</td><td>73.52</td><td>67.35</td><td>45.84</td><td>68.87</td><td>48.64</td><td>67.93</td><td>48.00</td></tr><tr><th rowspan="2">TTA</th><th>w/o TTA</th><td>95.37</td><td>60.13</td><td>71.80</td><td>79.15</td><td>59.10</td><td>79.18</td><td>61.45</td><td>77.50</td><td>60.42</td></tr><tr><th>w/o Contrastive</th><td>94.99</td><td>60.10</td><td>71.48</td><td>79.46</td><td>60.11</td><td>79.39</td><td>62.13</td><td>77.78</td><td>61.14</td></tr><tr><th colspan="2">COMET (Full)</th><td>95.30</td><td>60.26</td><td>72.00</td><td>79.54</td><td>60.01</td><td>79.56</td><td>62.21</td><td>77.92</td><td>61.20</td></tr></tbody></table>

Table 6: Ablation study results on COMET on the SWaT dataset.

<table><tbody><tr><th rowspan="2">Category</th><th rowspan="2">Method</th><td colspan="9">Metric</td></tr><tr><td>F1(K=0)</td><td>F1(K=100)</td><td>Aff-F1</td><td>AUROC</td><td>AUPRC</td><td>R-AUC-ROC</td><td>R-AUC-PR</td><td>VUS-ROC</td><td>VUS-PR</td></tr><tr><th>Architecture</th><th>w/o Multi-Scale</th><td>87.17</td><td>74.31</td><td>70.50</td><td>82.82</td><td>71.58</td><td>80.43</td><td>56.75</td><td>80.44</td><td>56.95</td></tr><tr><th rowspan="5">Scoring</th><th>w/o Quant Score</th><td>70.02</td><td>49.52</td><td>71.72</td><td>77.32</td><td>31.59</td><td>77.17</td><td>29.25</td><td>77.07</td><td>29.34</td></tr><tr><th>w/o Memory Score</th><td>90.83</td><td>75.19</td><td>75.65</td><td>87.25</td><td>75.61</td><td>83.16</td><td>62.73</td><td>83.34</td><td>63.28</td></tr><tr><th>w/o Local Scaling NN</th><td>91.81</td><td>75.14</td><td>73.06</td><td>84.24</td><td>73.58</td><td>83.79</td><td>65.39</td><td>83.75</td><td>65.46</td></tr><tr><th>w/o Variable Selection</th><td>94.01</td><td>74.91</td><td>74.94</td><td>85.21</td><td>74.51</td><td>85.77</td><td>70.52</td><td>85.75</td><td>70.67</td></tr><tr><th>w/o Normalization</th><td>89.14</td><td>75.99</td><td>73.73</td><td>82.63</td><td>73.87</td><td>71.41</td><td>55.08</td><td>71.95</td><td>55.14</td></tr><tr><th rowspan="2">TTA</th><th>w/o TTA</th><td>91.38</td><td>75.06</td><td>72.93</td><td>85.48</td><td>74.33</td><td>84.85</td><td>65.46</td><td>84.82</td><td>65.50</td></tr><tr><th>w/o Contrastive</th><td>91.53</td><td>75.37</td><td>73.54</td><td>85.56</td><td>74.59</td><td>85.02</td><td>66.15</td><td>84.99</td><td>66.16</td></tr><tr><th colspan="2">COMET (Full)</th><td>91.62</td><td>75.24</td><td>74.14</td><td>85.50</td><td>74.48</td><td>85.09</td><td>66.42</td><td>85.06</td><td>66.48</td></tr></tbody></table>

Table 7: Ablation study results on COMET on the SMAP dataset.

<table><tbody><tr><th rowspan="2">Category</th><th rowspan="2">Method</th><td colspan="9">Metric</td></tr><tr><td>F1(K=0)</td><td>F1(K=100)</td><td>Aff-F1</td><td>AUROC</td><td>AUPRC</td><td>R-AUC-ROC</td><td>R-AUC-PR</td><td>VUS-ROC</td><td>VUS-PR</td></tr><tr><th>Architecture</th><th>w/o Multi-Scale</th><td>74.64</td><td>15.58</td><td>62.62</td><td>44.41</td><td>11.65</td><td>45.97</td><td>12.93</td><td>45.74</td><td>12.89</td></tr><tr><th rowspan="5">Scoring</th><th>w/o Quant Score</th><td>85.63</td><td>26.39</td><td>68.17</td><td>60.12</td><td>17.01</td><td>59.69</td><td>18.14</td><td>59.49</td><td>18.09</td></tr><tr><th>w/o Memory Score</th><td>69.55</td><td>18.87</td><td>63.97</td><td>45.62</td><td>12.70</td><td>46.64</td><td>13.88</td><td>46.52</td><td>13.85</td></tr><tr><th>w/o Local Scaling NN</th><td>75.16</td><td>24.84</td><td>68.09</td><td>58.02</td><td>15.59</td><td>59.21</td><td>17.10</td><td>59.10</td><td>17.04</td></tr><tr><th>w/o Variable Selection</th><td>75.20</td><td>21.05</td><td>68.09</td><td>57.54</td><td>15.34</td><td>56.56</td><td>16.63</td><td>56.13</td><td>16.58</td></tr><tr><th>w/o Normalization</th><td>69.32</td><td>12.82</td><td>57.07</td><td>38.87</td><td>10.82</td><td>41.21</td><td>11.79</td><td>41.09</td><td>11.77</td></tr><tr><th rowspan="2">TTA</th><th>w/o TTA</th><td>82.46</td><td>25.40</td><td>68.01</td><td>59.06</td><td>16.17</td><td>58.69</td><td>17.51</td><td>58.52</td><td>17.46</td></tr><tr><th>w/o Contrastive</th><td>82.54</td><td>25.21</td><td>68.18</td><td>58.83</td><td>15.98</td><td>59.33</td><td>17.38</td><td>59.15</td><td>17.32</td></tr><tr><th colspan="2">COMET (Full)</th><td>82.30</td><td>25.28</td><td>68.24</td><td>59.17</td><td>16.20</td><td>59.43</td><td>17.59</td><td>59.20</td><td>17.54</td></tr></tbody></table>

Table 8: Ablation study results on COMET on the MSL dataset.

<table><tbody><tr><th rowspan="2">Category</th><th rowspan="2">Method</th><td colspan="9">Metric</td></tr><tr><td>F1(K=0)</td><td>F1(K=100)</td><td>Aff-F1</td><td>AUROC</td><td>AUPRC</td><td>R-AUC-ROC</td><td>R-AUC-PR</td><td>VUS-ROC</td><td>VUS-PR</td></tr><tr><th>Architecture</th><th>w/o Multi-Scale</th><td>55.48</td><td>25.00</td><td>69.77</td><td>59.05</td><td>12.44</td><td>63.18</td><td>16.89</td><td>62.37</td><td>16.81</td></tr><tr><th rowspan="5">Scoring</th><th>w/o Quant Score</th><td>84.42</td><td>33.08</td><td>71.76</td><td>69.19</td><td>20.20</td><td>71.54</td><td>26.35</td><td>70.76</td><td>25.97</td></tr><tr><th>w/o Memory Score</th><td>83.35</td><td>24.39</td><td>61.24</td><td>63.70</td><td>15.01</td><td>66.23</td><td>19.93</td><td>65.46</td><td>19.71</td></tr><tr><th>w/o Local Scaling NN</th><td>83.21</td><td>20.17</td><td>67.96</td><td>63.22</td><td>15.28</td><td>67.80</td><td>20.34</td><td>67.17</td><td>20.01</td></tr><tr><th>w/o Variable Selection</th><td>76.95</td><td>19.15</td><td>68.05</td><td>57.60</td><td>13.35</td><td>63.09</td><td>17.62</td><td>62.09</td><td>17.46</td></tr><tr><th>w/o Normalization</th><td>82.60</td><td>18.95</td><td>35.94</td><td>55.93</td><td>14.12</td><td>64.13</td><td>19.19</td><td>63.41</td><td>18.92</td></tr><tr><th rowspan="2">TTA</th><th>w/o TTA</th><td>87.27</td><td>26.44</td><td>69.40</td><td>65.49</td><td>16.63</td><td>69.27</td><td>22.67</td><td>68.39</td><td>22.27</td></tr><tr><th>w/o Contrastive</th><td>87.48</td><td>26.27</td><td>71.10</td><td>65.42</td><td>16.57</td><td>69.61</td><td>22.64</td><td>68.65</td><td>22.23</td></tr><tr><th colspan="2">COMET (Full)</th><td>87.44</td><td>26.28</td><td>71.04</td><td>65.40</td><td>16.56</td><td>69.90</td><td>22.67</td><td>68.87</td><td>22.25</td></tr></tbody></table>

Table 9: Ablation study results on COMET on the WADI dataset.

<table><tbody><tr><th rowspan="2">Category</th><th rowspan="2">Method</th><td colspan="9">Metric</td></tr><tr><td>F1(K=0)</td><td>F1(K=100)</td><td>Aff-F1</td><td>AUROC</td><td>AUPRC</td><td>R-AUC-ROC</td><td>R-AUC-PR</td><td>VUS-ROC</td><td>VUS-PR</td></tr><tr><th>Architecture</th><th>w/o Multi-Scale</th><td>77.34</td><td>20.19</td><td>73.17</td><td>62.54</td><td>9.47</td><td>65.04</td><td>10.47</td><td>65.00</td><td>10.48</td></tr><tr><th rowspan="5">Scoring</th><th>w/o Quant Score</th><td>81.64</td><td>20.39</td><td>72.57</td><td>64.25</td><td>10.36</td><td>66.91</td><td>10.74</td><td>66.81</td><td>10.75</td></tr><tr><th>w/o Memory Score</th><td>69.57</td><td>9.66</td><td>71.95</td><td>50.15</td><td>5.81</td><td>51.57</td><td>6.37</td><td>51.51</td><td>6.36</td></tr><tr><th>w/o Local Scaling NN</th><td>74.38</td><td>18.31</td><td>71.96</td><td>60.31</td><td>8.74</td><td>62.62</td><td>9.62</td><td>62.58</td><td>9.63</td></tr><tr><th>w/o Variable Selection</th><td>77.79</td><td>17.67</td><td>71.91</td><td>60.98</td><td>8.92</td><td>63.41</td><td>9.82</td><td>63.36</td><td>9.82</td></tr><tr><th>w/o Normalization</th><td>49.54</td><td>6.96</td><td>52.85</td><td>47.62</td><td>5.11</td><td>45.00</td><td>5.78</td><td>44.76</td><td>5.78</td></tr><tr><th rowspan="2">TTA</th><th>w/o TTA</th><td>73.81</td><td>18.98</td><td>72.63</td><td>61.19</td><td>9.14</td><td>63.64</td><td>10.10</td><td>63.57</td><td>10.11</td></tr><tr><th>w/o Contrastive</th><td>73.66</td><td>19.04</td><td>73.85</td><td>61.73</td><td>9.31</td><td>64.09</td><td>10.25</td><td>64.04</td><td>10.26</td></tr><tr><th colspan="2">COMET (Full)</th><td>74.07</td><td>18.92</td><td>72.43</td><td>61.79</td><td>9.23</td><td>64.17</td><td>10.18</td><td>64.13</td><td>10.20</td></tr></tbody></table>

## Appendix H Hyperparameter Sensitivity

We conduct a sensitivity analysis on key hyperparameters that control the temporal granularity, representation capacity, and locality of memory-based scoring in COMET. Unless otherwise specified, all hyperparameters not under study are fixed to the default configuration used in the main experiments.

### H.1 Multi-Scale Patch Configuration

To analyze the effect of temporal granularity, we vary the set of patch sizes and strides used in multi-scale patch encoding. Specifically, we consider both single-scale and multi-scale configurations by combining different patch sizes and strides. The evaluated patch size sets include $\{2\},\{4\},\{6\},\{2,4\},\{4,6\},\{2,4,6\},\{2,4,6,8\}$, with corresponding stride configurations $\{1\},\{2\},\{3\},\{1,2\},\{2,3\},\{1,2,3\},\{1,2,3,4\}$. This allows us to examine how increasing temporal coverage and scale diversity affects anomaly detection performance.

As shown in Table 10, single-scale configurations generally exhibit inferior performance compared to multi-scale settings, particularly on range-based metrics. Across all datasets, the configuration using patch sizes $\{2,4,6\}$ with strides $\{1,2,3\}$ consistently achieves the best or near-best performance, indicating that combining short-, mid-, and long-range temporal patterns is critical for accurately detecting both abrupt and persistent anomalies. In contrast, further extending the scale set to include very large patches (e.g., $\{2,4,6,8\}$) often leads to performance saturation or degradation, suggesting that excessive temporal smoothing can dilute fine-grained anomaly signals.

Table 10: Hyperparameter sensitivity analysis on multi-scale patch configurations. Patch sizes and strides are denoted as $\{\text{patch sizes}\}\mid\{\text{strides}\}$. All results are reported in percentage (%).

<table><thead><tr><th>Dataset</th><th>Patch & Stride Size</th><th>F1(K=0)</th><th>F1(K=100)</th><th>Aff-F1</th><th>AUROC</th><th>AUPRC</th><th>R-AUC-ROC</th><th>R-AUC-PR</th><th>VUS-ROC</th><th>VUS-PR</th></tr></thead><tbody><tr><th rowspan="7">PSM</th><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn> <mo>}</mo></mrow></mrow> <annotation>\{2\}\mid\{1\}</annotation></semantics></math></th><td>93.08</td><td>58.11</td><td>71.18</td><td>77.92</td><td>59.17</td><td>78.07</td><td>59.70</td><td>77.47</td><td>59.20</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>4</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>2</mn> <mo>}</mo></mrow></mrow> <annotation>\{4\}\mid\{2\}</annotation></semantics></math></th><td>92.06</td><td>57.90</td><td>70.98</td><td>76.78</td><td>53.04</td><td>76.30</td><td>56.54</td><td>74.22</td><td>55.47</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>6</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>3</mn> <mo>}</mo></mrow></mrow> <annotation>\{6\}\mid\{3\}</annotation></semantics></math></th><td>87.60</td><td>46.49</td><td>71.33</td><td>67.04</td><td>48.75</td><td>63.50</td><td>46.25</td><td>62.71</td><td>45.83</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>4</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn><mo>,</mo><mn>2</mn> <mo>}</mo></mrow></mrow> <annotation>\{2,4\}\mid\{1,2\}</annotation></semantics></math></th><td>82.58</td><td>45.25</td><td>71.48</td><td>67.02</td><td>46.41</td><td>67.56</td><td>46.91</td><td>66.67</td><td>46.32</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>4</mn><mo>,</mo><mn>6</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>3</mn> <mo>}</mo></mrow></mrow> <annotation>\{4,6\}\mid\{2,3\}</annotation></semantics></math></th><td>91.80</td><td>44.62</td><td>71.28</td><td>68.43</td><td>48.24</td><td>68.58</td><td>50.66</td><td>67.48</td><td>50.05</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>4</mn><mo>,</mo><mn>6</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn> <mo>}</mo></mrow></mrow> <annotation>\{2,4,6\}\mid\{1,2,3\}</annotation></semantics></math></th><td>95.37</td><td>60.13</td><td>71.80</td><td>79.15</td><td>59.10</td><td>79.18</td><td>61.45</td><td>77.50</td><td>60.42</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>4</mn><mo>,</mo><mn>6</mn><mo>,</mo><mn>8</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo>,</mo><mn>4</mn> <mo>}</mo></mrow></mrow> <annotation>\{2,4,6,8\}\mid\{1,2,3,4\}</annotation></semantics></math></th><td>93.99</td><td>36.95</td><td>70.39</td><td>59.76</td><td>36.55</td><td>59.88</td><td>39.65</td><td>58.75</td><td>39.29</td></tr><tr><th rowspan="7">SWaT</th><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn> <mo>}</mo></mrow></mrow> <annotation>\{2\}\mid\{1\}</annotation></semantics></math></th><td>89.79</td><td>74.15</td><td>71.23</td><td>83.39</td><td>72.83</td><td>78.64</td><td>61.17</td><td>78.79</td><td>61.33</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>4</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>2</mn> <mo>}</mo></mrow></mrow> <annotation>\{4\}\mid\{2\}</annotation></semantics></math></th><td>85.26</td><td>75.85</td><td>72.91</td><td>84.69</td><td>74.04</td><td>83.13</td><td>60.10</td><td>83.04</td><td>60.03</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>6</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>3</mn> <mo>}</mo></mrow></mrow> <annotation>\{6\}\mid\{3\}</annotation></semantics></math></th><td>87.17</td><td>74.31</td><td>70.50</td><td>82.82</td><td>71.58</td><td>80.43</td><td>56.75</td><td>80.44</td><td>56.95</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>4</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn><mo>,</mo><mn>2</mn> <mo>}</mo></mrow></mrow> <annotation>\{2,4\}\mid\{1,2\}</annotation></semantics></math></th><td>89.20</td><td>75.21</td><td>70.08</td><td>82.27</td><td>73.15</td><td>72.34</td><td>58.39</td><td>72.49</td><td>58.19</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>4</mn><mo>,</mo><mn>6</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>3</mn> <mo>}</mo></mrow></mrow> <annotation>\{4,6\}\mid\{2,3\}</annotation></semantics></math></th><td>87.60</td><td>75.20</td><td>70.07</td><td>80.84</td><td>71.89</td><td>76.49</td><td>56.10</td><td>76.58</td><td>56.17</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>4</mn><mo>,</mo><mn>6</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn> <mo>}</mo></mrow></mrow> <annotation>\{2,4,6\}\mid\{1,2,3\}</annotation></semantics></math></th><td>91.38</td><td>75.06</td><td>72.93</td><td>85.48</td><td>74.33</td><td>84.85</td><td>65.46</td><td>84.82</td><td>65.50</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>4</mn><mo>,</mo><mn>6</mn><mo>,</mo><mn>8</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo>,</mo><mn>4</mn> <mo>}</mo></mrow></mrow> <annotation>\{2,4,6,8\}\mid\{1,2,3,4\}</annotation></semantics></math></th><td>87.48</td><td>74.65</td><td>71.60</td><td>83.82</td><td>73.17</td><td>77.02</td><td>55.64</td><td>76.97</td><td>55.72</td></tr><tr><th rowspan="7">SMAP</th><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn> <mo>}</mo></mrow></mrow> <annotation>\{2\}\mid\{1\}</annotation></semantics></math></th><td>75.99</td><td>19.09</td><td>59.59</td><td>48.23</td><td>12.62</td><td>49.14</td><td>14.00</td><td>49.02</td><td>13.98</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>4</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>2</mn> <mo>}</mo></mrow></mrow> <annotation>\{4\}\mid\{2\}</annotation></semantics></math></th><td>73.54</td><td>14.59</td><td>64.39</td><td>43.38</td><td>11.28</td><td>45.59</td><td>12.49</td><td>45.45</td><td>12.47</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>6</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>3</mn> <mo>}</mo></mrow></mrow> <annotation>\{6\}\mid\{3\}</annotation></semantics></math></th><td>74.64</td><td>15.58</td><td>62.62</td><td>44.41</td><td>11.65</td><td>45.97</td><td>12.93</td><td>45.74</td><td>12.89</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>4</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn><mo>,</mo><mn>2</mn> <mo>}</mo></mrow></mrow> <annotation>\{2,4\}\mid\{1,2\}</annotation></semantics></math></th><td>75.06</td><td>15.67</td><td>65.02</td><td>43.03</td><td>11.53</td><td>45.87</td><td>12.75</td><td>45.73</td><td>12.73</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>4</mn><mo>,</mo><mn>6</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>3</mn> <mo>}</mo></mrow></mrow> <annotation>\{4,6\}\mid\{2,3\}</annotation></semantics></math></th><td>78.93</td><td>16.21</td><td>68.00</td><td>47.99</td><td>12.53</td><td>50.58</td><td>13.78</td><td>50.49</td><td>13.74</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>4</mn><mo>,</mo><mn>6</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn> <mo>}</mo></mrow></mrow> <annotation>\{2,4,6\}\mid\{1,2,3\}</annotation></semantics></math></th><td>82.46</td><td>25.40</td><td>68.01</td><td>59.06</td><td>16.17</td><td>58.69</td><td>17.51</td><td>58.52</td><td>17.46</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>4</mn><mo>,</mo><mn>6</mn><mo>,</mo><mn>8</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo>,</mo><mn>4</mn> <mo>}</mo></mrow></mrow> <annotation>\{2,4,6,8\}\mid\{1,2,3,4\}</annotation></semantics></math></th><td>69.80</td><td>16.22</td><td>64.61</td><td>43.05</td><td>12.14</td><td>45.98</td><td>13.20</td><td>45.89</td><td>13.16</td></tr><tr><th rowspan="7">MSL</th><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn> <mo>}</mo></mrow></mrow> <annotation>\{2\}\mid\{1\}</annotation></semantics></math></th><td>84.81</td><td>17.57</td><td>69.05</td><td>57.92</td><td>12.62</td><td>64.17</td><td>17.16</td><td>63.69</td><td>17.06</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>4</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>2</mn> <mo>}</mo></mrow></mrow> <annotation>\{4\}\mid\{2\}</annotation></semantics></math></th><td>81.42</td><td>17.67</td><td>71.80</td><td>60.81</td><td>15.07</td><td>65.46</td><td>19.94</td><td>64.90</td><td>19.68</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>6</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>3</mn> <mo>}</mo></mrow></mrow> <annotation>\{6\}\mid\{3\}</annotation></semantics></math></th><td>55.48</td><td>25.00</td><td>69.77</td><td>59.05</td><td>12.44</td><td>63.18</td><td>16.89</td><td>62.37</td><td>16.81</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>4</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn><mo>,</mo><mn>2</mn> <mo>}</mo></mrow></mrow> <annotation>\{2,4\}\mid\{1,2\}</annotation></semantics></math></th><td>86.68</td><td>21.95</td><td>71.06</td><td>55.65</td><td>13.80</td><td>63.54</td><td>18.52</td><td>62.68</td><td>18.28</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>4</mn><mo>,</mo><mn>6</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>3</mn> <mo>}</mo></mrow></mrow> <annotation>\{4,6\}\mid\{2,3\}</annotation></semantics></math></th><td>81.86</td><td>19.25</td><td>69.51</td><td>58.87</td><td>15.02</td><td>64.90</td><td>19.72</td><td>64.45</td><td>19.46</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>4</mn><mo>,</mo><mn>6</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn> <mo>}</mo></mrow></mrow> <annotation>\{2,4,6\}\mid\{1,2,3\}</annotation></semantics></math></th><td>87.27</td><td>26.44</td><td>69.40</td><td>65.49</td><td>16.63</td><td>69.27</td><td>22.67</td><td>68.39</td><td>22.27</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>4</mn><mo>,</mo><mn>6</mn><mo>,</mo><mn>8</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo>,</mo><mn>4</mn> <mo>}</mo></mrow></mrow> <annotation>\{2,4,6,8\}\mid\{1,2,3,4\}</annotation></semantics></math></th><td>78.46</td><td>19.65</td><td>69.33</td><td>57.23</td><td>14.25</td><td>63.83</td><td>18.34</td><td>62.98</td><td>18.16</td></tr><tr><th rowspan="7">WADI</th><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn> <mo>}</mo></mrow></mrow> <annotation>\{2\}\mid\{1\}</annotation></semantics></math></th><td>77.34</td><td>18.60</td><td>72.79</td><td>58.89</td><td>8.82</td><td>61.29</td><td>9.76</td><td>61.25</td><td>9.76</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>4</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>2</mn> <mo>}</mo></mrow></mrow> <annotation>\{4\}\mid\{2\}</annotation></semantics></math></th><td>35.19</td><td>13.69</td><td>73.08</td><td>55.87</td><td>6.23</td><td>57.69</td><td>6.90</td><td>57.57</td><td>6.90</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>6</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>3</mn> <mo>}</mo></mrow></mrow> <annotation>\{6\}\mid\{3\}</annotation></semantics></math></th><td>77.34</td><td>20.19</td><td>73.17</td><td>62.54</td><td>9.47</td><td>65.04</td><td>10.47</td><td>65.00</td><td>10.48</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>4</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn><mo>,</mo><mn>2</mn> <mo>}</mo></mrow></mrow> <annotation>\{2,4\}\mid\{1,2\}</annotation></semantics></math></th><td>54.91</td><td>11.25</td><td>70.82</td><td>51.73</td><td>5.93</td><td>54.02</td><td>6.60</td><td>53.96</td><td>6.60</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>4</mn><mo>,</mo><mn>6</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>3</mn> <mo>}</mo></mrow></mrow> <annotation>\{4,6\}\mid\{2,3\}</annotation></semantics></math></th><td>50.00</td><td>14.71</td><td>71.85</td><td>57.45</td><td>6.73</td><td>59.47</td><td>7.58</td><td>59.40</td><td>7.58</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>4</mn><mo>,</mo><mn>6</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn> <mo>}</mo></mrow></mrow> <annotation>\{2,4,6\}\mid\{1,2,3\}</annotation></semantics></math></th><td>73.81</td><td>18.98</td><td>72.63</td><td>61.19</td><td>9.14</td><td>63.64</td><td>10.10</td><td>63.57</td><td>10.11</td></tr><tr><th><math><semantics><mrow><mrow><mo>{</mo> <mn>2</mn><mo>,</mo><mn>4</mn><mo>,</mo><mn>6</mn><mo>,</mo><mn>8</mn> <mo>}</mo></mrow> <mo>∣</mo> <mrow><mo>{</mo> <mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo>,</mo><mn>4</mn> <mo>}</mo></mrow></mrow> <annotation>\{2,4,6,8\}\mid\{1,2,3,4\}</annotation></semantics></math></th><td>74.71</td><td>16.84</td><td>71.76</td><td>61.96</td><td>8.75</td><td>64.50</td><td>9.70</td><td>64.49</td><td>9.72</td></tr></tbody></table>

### H.2 Codebook Size

We study the sensitivity to the size of the vector-quantized codebook, which controls the capacity of discrete normal pattern representations. The codebook size is varied over $\{16,32,64,128,256\}$, while keeping the multi-scale patch configuration and scoring strategy fixed. This analysis evaluates the trade-off between representational expressiveness and over-fragmentation of normal patterns.

Table 11 shows that moderate codebook sizes yield the most stable and robust performance across datasets. In particular, a codebook size of 128 consistently achieves the best or second-best results on most metrics. Smaller codebooks (e.g., 16 or 32) tend to underfit, limiting the diversity of normal pattern prototypes, while excessively large codebooks (e.g., 256) often degrade performance due to over-fragmentation and increased sensitivity to noise. These results suggest that a moderately sized codebook provides an effective balance between representational capacity and robustness.

Although the optimal codebook size varies slightly across datasets, this sensitivity is expected since the codebook explicitly represents prototypical normal patterns and thus reflects dataset-specific normal dynamics. Nevertheless, once an appropriate codebook size is selected for a given dataset, the model consistently achieves strong performance across all evaluation metrics. This observation indicates that the codebook mechanism itself is robust, while its capacity should be adapted to the intrinsic complexity of the underlying data rather than fixed universally.

Table 11: Hyperparameter sensitivity analysis on codebook size. All results are reported in percentage (%).

<table><thead><tr><th>Dataset</th><th>Codebook</th><th>F1(K=0)</th><th>F1(K=100)</th><th>Aff-F1</th><th>AUROC</th><th>AUPRC</th><th>R-AUC-ROC</th><th>R-AUC-PR</th><th>VUS-ROC</th><th>VUS-PR</th></tr></thead><tbody><tr><th rowspan="5">PSM</th><th>16</th><td>90.67</td><td>53.31</td><td>71.05</td><td>74.38</td><td>50.98</td><td>75.70</td><td>53.55</td><td>74.87</td><td>53.01</td></tr><tr><th>32</th><td>90.22</td><td>47.72</td><td>71.73</td><td>71.86</td><td>50.40</td><td>71.20</td><td>51.10</td><td>70.23</td><td>50.46</td></tr><tr><th>64</th><td>95.12</td><td>34.03</td><td>71.01</td><td>58.46</td><td>36.82</td><td>57.73</td><td>39.14</td><td>56.51</td><td>38.80</td></tr><tr><th>128</th><td>95.37</td><td>60.13</td><td>71.80</td><td>79.15</td><td>59.10</td><td>79.18</td><td>61.45</td><td>77.50</td><td>60.42</td></tr><tr><th>256</th><td>94.85</td><td>55.69</td><td>73.54</td><td>75.43</td><td>53.53</td><td>77.20</td><td>56.85</td><td>75.83</td><td>56.03</td></tr><tr><th rowspan="5">SWaT</th><th>16</th><td>91.11</td><td>74.64</td><td>70.81</td><td>82.01</td><td>72.66</td><td>76.05</td><td>59.96</td><td>75.83</td><td>59.67</td></tr><tr><th>32</th><td>87.62</td><td>75.09</td><td>70.75</td><td>83.93</td><td>73.07</td><td>81.02</td><td>59.53</td><td>81.10</td><td>59.69</td></tr><tr><th>64</th><td>89.86</td><td>74.93</td><td>71.10</td><td>82.29</td><td>73.01</td><td>73.40</td><td>57.12</td><td>73.47</td><td>57.10</td></tr><tr><th>128</th><td>92.37</td><td>74.68</td><td>76.67</td><td>84.06</td><td>73.31</td><td>84.21</td><td>67.95</td><td>84.18</td><td>68.11</td></tr><tr><th>256</th><td>91.38</td><td>75.06</td><td>72.93</td><td>85.48</td><td>74.33</td><td>84.85</td><td>65.46</td><td>84.82</td><td>65.50</td></tr><tr><th rowspan="5">SMAP</th><th>16</th><td>78.21</td><td>18.42</td><td>68.45</td><td>52.75</td><td>13.39</td><td>53.03</td><td>14.92</td><td>52.78</td><td>14.87</td></tr><tr><th>32</th><td>89.54</td><td>22.08</td><td>68.14</td><td>53.14</td><td>14.08</td><td>55.94</td><td>15.52</td><td>55.71</td><td>15.46</td></tr><tr><th>64</th><td>78.39</td><td>19.47</td><td>67.99</td><td>52.24</td><td>14.00</td><td>54.74</td><td>15.33</td><td>54.62</td><td>15.29</td></tr><tr><th>128</th><td>82.46</td><td>25.40</td><td>68.01</td><td>59.06</td><td>16.17</td><td>58.69</td><td>17.51</td><td>58.52</td><td>17.46</td></tr><tr><th>256</th><td>69.88</td><td>20.07</td><td>68.03</td><td>50.88</td><td>13.57</td><td>52.85</td><td>14.85</td><td>52.71</td><td>14.81</td></tr><tr><th rowspan="5">MSL</th><th>16</th><td>86.56</td><td>18.68</td><td>71.71</td><td>60.10</td><td>14.15</td><td>66.04</td><td>18.79</td><td>65.31</td><td>18.59</td></tr><tr><th>32</th><td>54.36</td><td>16.94</td><td>70.74</td><td>51.92</td><td>11.83</td><td>58.98</td><td>15.32</td><td>58.28</td><td>15.18</td></tr><tr><th>64</th><td>83.64</td><td>17.81</td><td>71.25</td><td>60.33</td><td>14.25</td><td>66.62</td><td>19.31</td><td>65.38</td><td>19.03</td></tr><tr><th>128</th><td>78.81</td><td>20.85</td><td>69.59</td><td>55.78</td><td>13.19</td><td>62.72</td><td>17.74</td><td>62.01</td><td>17.55</td></tr><tr><th>256</th><td>87.27</td><td>26.44</td><td>69.40</td><td>65.49</td><td>16.63</td><td>69.27</td><td>22.67</td><td>68.39</td><td>22.27</td></tr><tr><th rowspan="5">WADI</th><th>16</th><td>63.98</td><td>9.74</td><td>73.39</td><td>47.39</td><td>5.28</td><td>49.90</td><td>5.90</td><td>49.82</td><td>5.89</td></tr><tr><th>32</th><td>73.81</td><td>18.98</td><td>72.63</td><td>61.19</td><td>9.14</td><td>63.64</td><td>10.10</td><td>63.57</td><td>10.11</td></tr><tr><th>64</th><td>65.04</td><td>7.15</td><td>74.51</td><td>39.76</td><td>4.61</td><td>41.84</td><td>5.09</td><td>41.79</td><td>5.09</td></tr><tr><th>128</th><td>67.69</td><td>15.57</td><td>73.21</td><td>55.24</td><td>6.88</td><td>57.75</td><td>7.67</td><td>57.69</td><td>7.67</td></tr><tr><th>256</th><td>59.85</td><td>7.41</td><td>73.09</td><td>41.09</td><td>4.50</td><td>42.79</td><td>4.90</td><td>42.75</td><td>4.90</td></tr></tbody></table>

### H.3 Number of Nearest Neighbors

We analyze the impact of the number of nearest neighbors used in memory-based retrieval for anomaly scoring and test-time adaptation. The number of neighbors is varied over $\{1,3,5,10,20\}$. This parameter controls the locality of neighborhood-based normalization and affects the stability of anomaly scores under distribution shifts.

As reported in Table 12, using a very small number of neighbors (e.g., $k=1$) leads to unstable performance due to high sensitivity to local noise, whereas overly large neighborhoods (e.g., $k=20$) tend to oversmooth anomaly scores and reduce discriminability. Across datasets, intermediate values—particularly $k=5$ and $k=10$ —consistently provide the best performance, achieving strong results on both point-wise and range-based metrics. This indicates that aggregating information from a moderate local neighborhood effectively balances robustness and sensitivity in memory-based anomaly scoring and test-time adaptation.

Similar to the codebook size analysis, the optimal number of nearest neighbors can differ across datasets due to variations in noise level and local density structure. However, selecting a reasonable intermediate value (e.g., $k=5$ or $k=10$) consistently yields strong performance without requiring dataset-specific tuning. This suggests that the neighborhood-based aggregation in COMET is relatively insensitive to the exact choice of $k$, provided that extreme values are avoided.

Table 12: Hyperparameter sensitivity analysis on the number of nearest neighbors. All results are reported in percentage (%).

<table><thead><tr><th>Dataset</th><th>#NN</th><th>F1(K=0)</th><th>F1(K=100)</th><th>Aff-F1</th><th>AUROC</th><th>AUPRC</th><th>R-AUC-ROC</th><th>R-AUC-PR</th><th>VUS-ROC</th><th>VUS-PR</th></tr></thead><tbody><tr><th rowspan="5">PSM</th><th>1</th><td>96.05</td><td>30.82</td><td>70.09</td><td>56.10</td><td>36.05</td><td>58.65</td><td>39.68</td><td>57.54</td><td>39.20</td></tr><tr><th>3</th><td>94.38</td><td>35.30</td><td>70.10</td><td>57.42</td><td>38.90</td><td>60.00</td><td>42.04</td><td>58.76</td><td>41.44</td></tr><tr><th>5</th><td>94.55</td><td>56.57</td><td>73.79</td><td>75.33</td><td>58.70</td><td>75.41</td><td>59.97</td><td>74.20</td><td>59.07</td></tr><tr><th>10</th><td>95.37</td><td>60.13</td><td>71.80</td><td>79.15</td><td>59.10</td><td>79.18</td><td>61.45</td><td>77.50</td><td>60.42</td></tr><tr><th>20</th><td>93.03</td><td>52.07</td><td>71.96</td><td>71.82</td><td>52.36</td><td>73.87</td><td>55.08</td><td>72.71</td><td>54.31</td></tr><tr><th rowspan="5">SWaT</th><th>1</th><td>89.36</td><td>74.71</td><td>72.30</td><td>82.84</td><td>72.20</td><td>79.74</td><td>60.43</td><td>79.71</td><td>60.36</td></tr><tr><th>3</th><td>89.44</td><td>75.35</td><td>72.26</td><td>85.40</td><td>73.89</td><td>84.52</td><td>65.33</td><td>84.37</td><td>64.85</td></tr><tr><th>5</th><td>93.96</td><td>74.95</td><td>74.29</td><td>85.09</td><td>74.27</td><td>85.62</td><td>70.48</td><td>85.58</td><td>70.44</td></tr><tr><th>10</th><td>91.38</td><td>75.06</td><td>72.93</td><td>85.48</td><td>74.33</td><td>84.85</td><td>65.46</td><td>84.82</td><td>65.50</td></tr><tr><th>20</th><td>89.64</td><td>75.05</td><td>72.35</td><td>85.44</td><td>74.02</td><td>84.35</td><td>63.81</td><td>84.26</td><td>63.67</td></tr><tr><th rowspan="5">SMAP</th><th>1</th><td>72.75</td><td>19.07</td><td>59.62</td><td>46.19</td><td>13.76</td><td>46.83</td><td>14.54</td><td>46.77</td><td>14.53</td></tr><tr><th>3</th><td>92.00</td><td>23.51</td><td>68.73</td><td>55.73</td><td>14.97</td><td>58.13</td><td>16.36</td><td>57.99</td><td>16.31</td></tr><tr><th>5</th><td>84.26</td><td>20.12</td><td>68.10</td><td>54.45</td><td>14.83</td><td>56.77</td><td>16.16</td><td>56.67</td><td>16.11</td></tr><tr><th>10</th><td>82.46</td><td>25.40</td><td>68.01</td><td>59.06</td><td>16.17</td><td>58.69</td><td>17.51</td><td>58.52</td><td>17.46</td></tr><tr><th>20</th><td>81.67</td><td>25.03</td><td>68.70</td><td>59.44</td><td>16.27</td><td>60.86</td><td>17.68</td><td>60.65</td><td>17.62</td></tr><tr><th rowspan="5">MSL</th><th>1</th><td>83.35</td><td>24.39</td><td>62.07</td><td>63.70</td><td>15.00</td><td>66.23</td><td>19.93</td><td>65.46</td><td>19.71</td></tr><tr><th>3</th><td>64.96</td><td>20.36</td><td>70.23</td><td>49.21</td><td>11.54</td><td>54.98</td><td>15.43</td><td>54.67</td><td>15.31</td></tr><tr><th>5</th><td>86.98</td><td>31.60</td><td>71.58</td><td>63.96</td><td>19.80</td><td>67.30</td><td>24.27</td><td>66.83</td><td>24.06</td></tr><tr><th>10</th><td>87.27</td><td>26.44</td><td>69.40</td><td>65.49</td><td>16.63</td><td>69.27</td><td>22.67</td><td>68.39</td><td>22.27</td></tr><tr><th>20</th><td>84.99</td><td>20.25</td><td>71.50</td><td>63.11</td><td>15.28</td><td>69.07</td><td>20.75</td><td>68.18</td><td>20.42</td></tr><tr><th rowspan="5">WADI</th><th>1</th><td>72.17</td><td>5.90</td><td>73.38</td><td>41.63</td><td>4.59</td><td>43.46</td><td>5.04</td><td>43.33</td><td>5.03</td></tr><tr><th>3</th><td>70.50</td><td>18.41</td><td>71.65</td><td>62.70</td><td>9.34</td><td>65.42</td><td>10.33</td><td>65.31</td><td>10.33</td></tr><tr><th>5</th><td>73.50</td><td>18.12</td><td>72.44</td><td>62.60</td><td>9.23</td><td>65.40</td><td>10.24</td><td>65.32</td><td>10.26</td></tr><tr><th>10</th><td>73.81</td><td>18.98</td><td>72.63</td><td>61.19</td><td>9.14</td><td>63.64</td><td>10.10</td><td>63.57</td><td>10.11</td></tr><tr><th>20</th><td>76.86</td><td>18.23</td><td>73.31</td><td>63.10</td><td>9.17</td><td>66.09</td><td>10.22</td><td>66.00</td><td>10.23</td></tr></tbody></table>

[^1]: Practical approach to asynchronous multivariate time series anomaly detection and localization. In KDD ’21: The 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Virtual Event, Singapore, August 14-18, 2021, F. Zhu, B. C. Ooi, and C. Miao (Eds.), pp. 2485–2494. External Links: [Link](https://doi.org/10.1145/3447548.3467174), [Document](https://dx.doi.org/10.1145/3447548.3467174) Cited by: Appendix C, §4.1.

[^2]: WADI: a water distribution testbed for research in the design of secure cyber physical systems. In Proceedings of the 3rd International Workshop on Cyber-Physical Systems for Smart Water Networks, CySWATER@CPSWeek 2017, Pittsburgh, Pennsylvania, USA, April 21, 2017, P. Tsakalides and B. Beferull-Lozano (Eds.), pp. 25–28. External Links: [Link](https://doi.org/10.1145/3055366.3055375), [Document](https://dx.doi.org/10.1145/3055366.3055375) Cited by: Appendix C, §4.1.

[^3]: USAD: unsupervised anomaly detection on multivariate time series. In KDD ’20: The 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Virtual Event, CA, USA, August 23-27, 2020, R. Gupta, Y. Liu, J. Tang, and B. A. Prakash (Eds.), pp. 3395–3404. External Links: [Link](https://doi.org/10.1145/3394486.3403392), [Document](https://dx.doi.org/10.1145/3394486.3403392) Cited by: Appendix D, §2.1, §4.1.

[^4]: Pattern recognition and machine learning. Vol. 4, Springer. Cited by: §1.

[^5]: Deep learning for anomaly detection in time-series data: review, analysis, and guidelines. IEEE Access 9, pp. 120043–120065. External Links: [Link](https://doi.org/10.1109/ACCESS.2021.3107975), [Document](https://dx.doi.org/10.1109/ACCESS.2021.3107975) Cited by: §1.

[^6]: CARLA: self-supervised contrastive representation learning for time series anomaly detection. Pattern Recognit. 157, pp. 110874. External Links: [Link](https://doi.org/10.1016/j.patcog.2024.110874), [Document](https://dx.doi.org/10.1016/J.PATCOG.2024.110874) Cited by: §2.1.

[^7]: Deep learning for time series anomaly detection: A survey. ACM Comput. Surv. 57 (1), pp. 15:1–15:42. External Links: [Link](https://doi.org/10.1145/3691338), [Document](https://dx.doi.org/10.1145/3691338) Cited by: §1, §2.1, §2.1.

[^8]: The vector innovations structural time series framework: a simple approach to multivariate forecasting. Statistical Modelling 10 (4), pp. 353–374. Cited by: §1.

[^9]: A dataset to support research in the design of secure water treatment systems. In Critical Information Infrastructures Security - 11th International Conference, CRITIS 2016, Paris, France, October 10-12, 2016, Revised Selected Papers, G. M. Havârneanu, R. Setola, H. Nassopoulos, and S. D. Wolthusen (Eds.), Lecture Notes in Computer Science, Vol. 10242, pp. 88–99. External Links: [Link](https://doi.org/10.1007/978-3-319-71368-7%5C_8), [Document](https://dx.doi.org/10.1007/978-3-319-71368-7%5F8) Cited by: Appendix C, §4.1.

[^10]: Local evaluation of time series anomaly detection algorithms. In KDD ’22: The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Washington, DC, USA, August 14 - 18, 2022, A. Zhang and H. Rangwala (Eds.), pp. 635–645. External Links: [Link](https://doi.org/10.1145/3534678.3539339), [Document](https://dx.doi.org/10.1145/3534678.3539339) Cited by: §4.1.

[^11]: Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD 2018, London, UK, August 19-23, 2018, Y. Guo and F. Farooq (Eds.), pp. 387–395. External Links: [Link](https://doi.org/10.1145/3219819.3219845), [Document](https://dx.doi.org/10.1145/3219819.3219845) Cited by: Appendix C, Appendix C, §4.1.

[^12]: Transformer-based multivariate time series anomaly detection using inter-variable attention mechanism. Knowl. Based Syst. 290, pp. 111507. External Links: [Link](https://doi.org/10.1016/j.knosys.2024.111507), [Document](https://dx.doi.org/10.1016/J.KNOSYS.2024.111507) Cited by: Appendix D, §2.1, §4.1.

[^13]: When model meets new normals: test-time adaptation for unsupervised time-series anomaly detection. In Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, M. J. Wooldridge, J. G. Dy, and S. Natarajan (Eds.), pp. 13113–13121. External Links: [Link](https://doi.org/10.1609/aaai.v38i12.29210), [Document](https://dx.doi.org/10.1609/AAAI.V38I12.29210) Cited by: §1, §2.3.

[^14]: Battling the non-stationarity in time series forecasting via test-time adaptation. In AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25 - March 4, 2025, Philadelphia, PA, USA, T. Walsh, J. Shah, and Z. Kolter (Eds.), pp. 17868–17876. External Links: [Link](https://doi.org/10.1609/aaai.v39i17.33965), [Document](https://dx.doi.org/10.1609/AAAI.V39I17.33965) Cited by: §2.3.

[^15]: Towards a rigorous evaluation of time-series anomaly detection. In Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022, pp. 7194–7201. External Links: [Link](https://doi.org/10.1609/aaai.v36i7.20680), [Document](https://dx.doi.org/10.1609/AAAI.V36I7.20680) Cited by: §4.1.

[^16]: Reversible instance normalization for accurate time-series forecasting against distribution shift. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, External Links: [Link](https://openreview.net/forum?id=cGDAkQo1C0p) Cited by: §1.

[^17]: Nominality score conditioned time series anomaly detection by point/sequential reconstruction. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (Eds.), External Links: [Link](http://papers.nips.cc/paper%5C_files/paper/2023/hash/f1cf02ce09757f57c3b93c0db83181e0-Abstract-Conference.html) Cited by: §1, §2.1.

[^18]: Vector quantized time series generation with a bidirectional prior model. In International Conference on Artificial Intelligence and Statistics, 25-27 April 2023, Palau de Congressos, Valencia, Spain, F. J. R. Ruiz, J. G. Dy, and J. van de Meent (Eds.), Proceedings of Machine Learning Research, Vol. 206, pp. 7665–7693. External Links: [Link](https://proceedings.mlr.press/v206/lee23d.html) Cited by: §2.2.

[^19]: Explainable time series anomaly detection using masked latent generative modeling. Pattern Recognit. 156, pp. 110826. External Links: [Link](https://doi.org/10.1016/j.patcog.2024.110826), [Document](https://dx.doi.org/10.1016/J.PATCOG.2024.110826) Cited by: §2.2.

[^20]: Multivariate time series anomaly detection and interpretation using hierarchical inter-metric and temporal embedding. In KDD ’21: The 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Virtual Event, Singapore, August 14-18, 2021, F. Zhu, B. C. Ooi, and C. Miao (Eds.), pp. 3220–3230. External Links: [Link](https://doi.org/10.1145/3447548.3467075), [Document](https://dx.doi.org/10.1145/3447548.3467075) Cited by: §1.

[^21]: Koopa: learning non-stationary time series dynamics with koopman predictors. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (Eds.), External Links: [Link](http://papers.nips.cc/paper%5C_files/paper/2023/hash/28b3dc0970fa4624a63278a4268de997-Abstract-Conference.html) Cited by: §1.

[^22]: SOFTS: efficient multivariate time series forecasting with series-core fusion. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, A. Globersons, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. M. Tomczak, and C. Zhang (Eds.), External Links: [Link](http://papers.nips.cc/paper%5C_files/paper/2024/hash/754612bde73a8b65ad8743f1f6d8ddf6-Abstract-Conference.html) Cited by: §3.1.

[^23]: Multi-scale anomaly detection for time series with attention-based recurrent autoencoders. In Asian Conference on Machine Learning, ACML 2022, 12-14 December 2022, Hyderabad, India, V. N. Balasubramanian and I. W. Tsang (Eds.), Proceedings of Machine Learning Research, Vol. 189, pp. 674–689. External Links: [Link](https://proceedings.mlr.press/v189/qingning23a.html) Cited by: §1.

[^24]: Long short term memory networks for anomaly detection in time series. In 23rd European Symposium on Artificial Neural Networks, ESANN 2015, Bruges, Belgium, April 22-24, 2015, External Links: [Link](https://www.esann.org/sites/default/files/proceedings/legacy/es2015-56.pdf) Cited by: Appendix D, §2.1, §4.1.

[^25]: Accurate parameter-efficient test-time adaptation for time series forecasting. CoRR abs/2506.23424. External Links: [Link](https://doi.org/10.48550/arXiv.2506.23424), [Document](https://dx.doi.org/10.48550/ARXIV.2506.23424), 2506.23424 Cited by: §2.3.

[^26]: Unsupervised anomaly detection in time-series: an extensive evaluation and analysis of state-of-the-art methods. Expert Syst. Appl. 256, pp. 124922. External Links: [Link](https://doi.org/10.1016/j.eswa.2024.124922), [Document](https://dx.doi.org/10.1016/J.ESWA.2024.124922) Cited by: §1.

[^27]: A time series is worth 64 words: long-term forecasting with transformers. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023, External Links: [Link](https://openreview.net/forum?id=Jbdc0vTOcol) Cited by: §1.

[^28]: Volume under the surface: A new accuracy evaluation measure for time-series anomaly detection. Proc. VLDB Endow. 15 (11), pp. 2774–2787. External Links: [Link](https://www.vldb.org/pvldb/vol15/p2774-paparrizos.pdf), [Document](https://dx.doi.org/10.14778/3551793.3551830) Cited by: §4.1.

[^29]: A multimodal anomaly detector for robot-assisted feeding using an lstm-based variational autoencoder. IEEE Robotics Autom. Lett. 3 (2), pp. 1544–1551. External Links: [Link](https://doi.org/10.1109/LRA.2018.2801475), [Document](https://dx.doi.org/10.1109/LRA.2018.2801475) Cited by: Appendix D, §2.1, §4.1.

[^30]: Granularity fusion transformer: learning multi-granularity patterns for time-series forecasting. Knowl. Based Syst. 320, pp. 113644. External Links: [Link](https://doi.org/10.1016/j.knosys.2025.113644), [Document](https://dx.doi.org/10.1016/J.KNOSYS.2025.113644) Cited by: §1.

[^31]: Timeseries anomaly detection using temporal hierarchical one-class network. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin (Eds.), External Links: [Link](https://proceedings.neurips.cc/paper/2020/hash/97e401a02082021fd24957f852e0e475-Abstract.html) Cited by: §2.1.

[^32]: Neural discrete representation learning. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, and R. Garnett (Eds.), pp. 6306–6315. External Links: [Link](https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html) Cited by: §2.2.

[^33]: Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, and R. Garnett (Eds.), pp. 5998–6008. External Links: [Link](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) Cited by: §2.1.

[^34]: Drift doesn’t matter: dynamic decomposition with diffusion reconstruction for unstable multivariate time series anomaly detection. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (Eds.), External Links: [Link](http://papers.nips.cc/paper%5C_files/paper/2023/hash/22f5d8e689d2a011cd8ead552ed59052-Abstract-Conference.html) Cited by: Appendix D, §2.1, §4.1.

[^35]: Detecting anomalies in time series data from a manufacturing system using recurrent neural networks. Journal of Manufacturing Systems 62, pp. 823–834. Cited by: §1.

[^36]: CATCH: channel-aware multivariate time series anomaly detection via frequency patching. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025, External Links: [Link](https://openreview.net/forum?id=m08aK3xxdJ) Cited by: Appendix D, §2.1, §4.1.

[^37]: Anomaly transformer: time series anomaly detection with association discrepancy. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, External Links: [Link](https://openreview.net/forum?id=LzQQ89U1qm%5C_) Cited by: Appendix D, §2.1, §4.1.

[^38]: DCdetector: dual attention contrastive representation learning for time series anomaly detection. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD 2023, Long Beach, CA, USA, August 6-10, 2023, A. K. Singh, Y. Sun, L. Akoglu, D. Gunopulos, X. Yan, R. Kumar, F. Ozcan, and J. Ye (Eds.), pp. 3033–3045. External Links: [Link](https://doi.org/10.1145/3580305.3599295), [Document](https://dx.doi.org/10.1145/3580305.3599295) Cited by: §2.1.

[^39]: Self-tuning spectral clustering. In Advances in Neural Information Processing Systems, Vol. 17. Cited by: §3.3.

[^40]: Proactive model adaptation against concept drift for online time series forecasting. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, V.1, KDD 2025, Toronto, ON, Canada, August 3-7, 2025, Y. Sun, F. Chierichetti, H. W. Lauw, C. Perlich, W. H. Tok, and A. Tomkins (Eds.), pp. 2020–2031. External Links: [Link](https://doi.org/10.1145/3690624.3709210), [Document](https://dx.doi.org/10.1145/3690624.3709210) Cited by: §2.3.

[^41]: Data-driven forecasting and its role in enhanced decision-making. Eng. Appl. Artif. Intell. 154, pp. 110934. External Links: [Link](https://doi.org/10.1016/j.engappai.2025.110934), [Document](https://dx.doi.org/10.1016/J.ENGAPPAI.2025.110934) Cited by: §1.