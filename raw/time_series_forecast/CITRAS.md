---
title: "CITRAS: Covariate-Informed Transformer for Time Series Forecasting"
source: "https://arxiv.org/html/2503.24007v4"
author:
published:
created: 2026-07-23
description:
tags:
  - "clippings"
---
14 2026 Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.

The final version is published in IEEE Access and is available at: [https://doi.org/10.1109/ACCESS.2026.3695717](https://doi.org/10.1109/ACCESS.2026.3695717)

Corresponding author: Yosuke Yamaguchi (e-mail: yosuke.yamaguchi.fy@hitachi.com).

YOSUKE YAMAGUCHI1    ISSEI SUEMITSU1    and WENPENG WEI1 Research & Development Group, Hitachi, Ltd., Tokyo 185-8601, Japan

###### Abstract

In time series forecasting, covariates represent external factors that influence target variables. Some covariates are observable only in the past (observed covariates, such as recorded weather data), while others are known in advance (known covariates, such as calendar events or discount schedules). Although covariates have the potential to enhance forecasting performance, most deep learning–based forecasting models struggle to address the length discrepancy between variables caused by the future portion of known covariates and fail to leverage them flexibly. Moreover, capturing dependencies between target variables and covariates is non-trivial, as models must accurately reflect the local impact of covariates while simultaneously modeling global cross-variate dependencies. To address these challenges, we propose CITRAS, a decoder-only Transformer that flexibly integrates multiple target variables, observed covariates, and known covariates. While preserving strong autoregressive modeling capabilities, CITRAS introduces two novel mechanisms within patch-wise cross-variate attention: Key–Value (KV) Shift and Attention Score Smoothing. KV Shift seamlessly incorporates the future portion of known covariates into the forecasting process by aligning them with target variables based on their concurrent dependencies. Attention Score Smoothing refines locally accurate patch-wise cross-variate dependencies into global variate-level dependencies by smoothing the historical attention scores. Experimentally, CITRAS demonstrates strong performance across a wide range of real-world datasets in both covariate-informed and multivariate settings, showcasing its versatile ability to leverage cross-variate and cross-time dependencies for improved forecasting accuracy.

\=-21pt

## I Introduction

Time series forecasting is a cornerstone in diverse fields such as energy [^1], retail [^2], and finance [^3], where accurate predictions can drive strategic decisions and enhance operational efficiency. In practical scenarios, forecasters have access not only to the target variables they aim to predict but also to covariates that represent external factors influencing the target variables. For instance, the “Electricity Demand” data shown in Figure 1 (Left) demonstrates a strong negative correlation with the “Holiday” indicator, which is accessible throughout the future forecasting horizon. This highlights the critical importance of appropriately incorporating covariates into the forecasting process.

Driven by these practical demands, this paper addresses time series forecasting that involves two types of covariates. The first is the observed covariate, whose values are available only in the historical period up to the prediction time, such as recorded weather information. The second is the known covariate, whose values are available from the past through the entire forecasting horizon. Known covariates may represent predetermined quantities (e.g., calendar events), estimated information (e.g., weather forecasts), or controllable variables (e.g., planned discounts in retail).

![Refer to caption](https://arxiv.org/html/2503.24007v4/x1.png)

Figure 1: Left: Hourly electricity demand data for three weeks in Panama 4. “Holiday” and “Weekend” indicators demonstrate a negative correlation with demand, providing important contextual information in the future forecasting horizon. “Humidity” displays a weak correlation but introduces temporal disturbances (red circle), which may distract the forecasting process. Right: Performance improvement of CITRAS through the utilization of observed covariates and known covariates.

This practical problem setting presents two significant challenges. First, the forecasting model must accommodate heterogeneous variables flexibly. The availability and number of covariates can vary depending on the scenario, and known covariates exhibit a length discrepancy with other variables. In addition, many real-world applications require forecasting multiple target variables simultaneously, i.e., multivariate forecasting. Second, the model must capture the dependencies between variables from both fine-grained and coarse-grained perspectives. The fine-grained perspective enables the model to precisely capture the local impact of covariates, such as sudden drops in electricity demand on holidays. However, a model that relies solely on a fine-grained perspective may fail to capture global dependencies between variables, as it can overlook covariates representing infrequent events or can be distracted by temporal disturbances. Therefore, it is crucial for the model to capture dependencies between variables at both fine-grained and coarse-grained levels to ensure robust and accurate forecasting.

In recent years, deep forecasting models—particularly Transformers [^5] —have achieved remarkable progress owing to their strong ability to capture temporal dependencies. However, most existing approaches still fall short in handling heterogeneous covariates. Many are tailored exclusively to multivariate forecasting [^6] [^7], others depend solely on observed covariates [^8], and some are limited to known covariates [^9]. These limitations suggest that the extension of Transformers to flexibly incorporate both observed and known covariates, while still preserving their strong temporal modeling capabilities, remains an open direction for further exploration.

Existing approaches to capture cross-variate dependencies in Transformers can be broadly categorized into two types: variate-level and patch-level. iTransformer [^10] is a representative variate-level approach that captures inter-variate correlations by embedding each sequence into a single variate token and applying attention across multiple variate tokens. On the other hand, patch-level approaches, represented by Crossformer [^6], apply cross-variate attention between locally semantic tokens obtained through patching. While variate-level approaches excel at capturing global dependencies robustly to noisy interactions, patch-level approaches are better at capturing fine-grained information. Unfortunately, an architecture that enjoys the advantages of both of these approaches in cross-variate modeling is yet to be developed.

To overcome these challenges, we propose CITRAS: Covariate-Informed Transformer for time series forecasting that flexibly leverages multiple target variables, observed covariates, and known covariates. This model is a decoder-only patch-based Transformer that has a cross-time attention module and a cross-variate attention module separately. To effectively model the cross-variate dependencies at both fine-grained and coarse-grained perspectives, we introduce Attention Score Smoothing in the cross-variate attention module, which refines locally accurate patch-wise dependencies into global variate-level dependencies. Furthermore, we introduce the Key-Value (KV) Shift mechanism, which associates the key of the known covariate with the value from one patch step ahead, thereby seamlessly integrating future information from known covariates into the prediction process along with current information from observed covariates. These innovations expand the covariate understanding capabilities of the canonical decoder-only Transformer, fully preserving its strong autoregressive properties.

In summary, our contributions are as follows:

- To effectively exploit the external impacts represented by covariates, we extend the decoder-only Transformer to flexibly accommodate observed covariates and known covariates while fully maintaining its original strong autoregressive capability.
- Our model, CITRAS, introduces two novel mechanisms into the cross-variate attention module. Attention Score Smoothing captures locally accurate patch-wise dependencies and refines them into global variate-level dependencies, while KV Shift seamlessly integrates future information from known covariates based on the obtained dependencies.
- Experimentally, CITRAS demonstrates strong forecasting performance in both covariate-informed and multivariate settings, ranking first more frequently than other state-of-the-art models by effectively leveraging cross-variate and cross-time dependencies.

## II Related Work

### II-A Transformer-based Time Series Forecasting

Given the significant success of Transformers in the fields of natural language processing [^11] and computer vision [^12], numerous studies have attempted to leverage their capabilities for time series forecasting. These works can be roughly categorized into two approaches: channel-dependent (CD) and channel-independent (CI). The CD approach assumes that the future values of a variable are determined by its past values as well as the values of other variables. In contrast, the CI approach posits that a specific variable depends only on its past values, omitting explicit interactions between variables.

Early applications of Transformers in time series forecasting can be viewed as CD approaches, where multiple variables at each time step are embedded into temporal tokens [^13] [^14] [^15]. To address the computational costs and intricate dependencies arising from long time series data, Informer [^16] designs a ProbSparse self-attention mechanism with a distilling operation to efficiently focus on important features. Autoformer [^17] incorporates decomposition and an auto-correlation mechanism to uncover reliable dependencies from complex temporal patterns. However, point-wise temporal tokens have limited local semantics, making it difficult to capture intricate cross-time and cross-variate dependencies.

In response to these challenges, PatchTST [^18] adopts a CI approach and introduces patching, which handles time series data by segmenting it into fixed-length segments. Patching reduces computational costs and enhances the local semantics in token representation, making it a prevalent technique in subsequent methods [^19] [^20] [^21]. Additionally, the CI approach is also adopted by many recent models as it avoids the risk of unnecessary noisy interactions between channels. For example, DLinear [^22] decomposes each time series into trend and seasonal components and applies simple linear projections independently to each channel. FITS [^23] transforms each univariate time series into the frequency domain via the Fourier transform, learning spectral representations separately for each variable before inverse transformation for forecasting. Moreover, recent large models [^24] [^25] [^26] and LLM-based forecasters [^27] [^28] [^29] also adopt the CI approach.

However, the CI approach is sometimes regarded as an oversimplification [^30], especially considering that fluctuations in time series data are often influenced by external factors. A representative CD approach is iTransformer [^10], which embeds each channel into a single variate token and applies cross-variate attention over them. TimeXer [^31] and Leddam [^32] also capture cross-variate dependencies at the variate level, with an additional cross-time attention to capture intra-variate temporal dependencies. TQNet [^33] captures cross-variate dependencies both from variate- and dataset-level perspectives by introducing learnable query vectors for each channel used throughout the entire dataset. Meanwhile, Crossformer [^6] applies cross-variate attention to patch-level representations. Any-variate attention techniques [^34] [^8] [^35] also apply attention to patch-level representations obtained by flattening all variables into one sequence. While the patch-level approach captures more fine-grained dependencies between variables, its local receptive field lacks global context awareness. Our method, CITRAS, first captures fine-grained dependencies at the patch level, and Attention Score Smoothing gradually refines them into global, variate-level dependencies.

### II-B Covariate-Informed Time Series Forecasting

To leverage the rich contextual information provided by covariates, numerous approaches have been proposed. Statistical models such as ARIMAX [^36] or VARX [^37] extend their original frameworks by assuming linear relationships between target variables and covariates. Recently, many deep learning models focus on covariate-informed settings: N-BEATSx [^38] enhances the original N-BEATS [^39] by introducing an additional MLP-based residual stack for covariates. TiDE [^40] introduces an MLP-based architecture that utilizes future information of known covariates in the final projection to leverage their direct effects on future target variables. ExoTST [^9] treats past and future information of known covariates as separate modalities and fuses them via cross-time attention to enrich autoregressive forecasting with projected covariate information. Timer-XL [^8] introduces a masking scheme, TimeAttention, in a decoder-only Transformer to model covariate influence within any-variate attention. DeformTime [^41] captures both inter-variable and intra-variable relationships by employing deformable attention to dynamically adjust receptive fields across variables and time. More recently, Sonnet [^42] proposes a Spectral Operator Neural Network that models dependencies between variables in the frequency domain by applying learnable wavelet transforms and an attention mechanism that measures inter-variable relationships based on spectral coherence rather than time-domain similarity.

TABLE I: Comparison of Base Architecture and Supported Variable Types for Covariate-Informed Forecasting

<table><thead><tr><th rowspan="2">Models</th><th>Base</th><th>Uni-</th><th>Multi-</th><th>Obs.</th><th>Kno.</th></tr><tr><th>arch.</th><th>variate</th><th>variate</th><th>cov.</th><th>cov.</th></tr><tr><th>CITRAS (ours)</th><th>TR(Dec)</th><th>✓</th><th>✓</th><th>✓</th><th>✓</th></tr></thead><tbody><tr><th>TFT <sup><a href="#fn:43">43</a></sup></th><td>TR+RNN</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><th>TSMixer-Ext <sup><a href="#fn:44">44</a></sup></th><td>MLP</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><th>TimeXer <sup><a href="#fn:31">31</a></sup></th><td>TR</td><td>✓</td><td>✓</td><td>✓</td><td>✧</td></tr><tr><th>N-BEATSx <sup><a href="#fn:38">38</a></sup></th><td>MLP</td><td>✓</td><td>✗</td><td>✗</td><td>✓</td></tr><tr><th>TiDE <sup><a href="#fn:40">40</a></sup></th><td>MLP</td><td>✓</td><td>✗</td><td>✗</td><td>✓</td></tr><tr><th>ExoTST <sup><a href="#fn:9">9</a></sup></th><td>TR</td><td>✓</td><td>✗</td><td>✗</td><td>✓</td></tr><tr><th>Timer-XL <sup><a href="#fn:8">8</a></sup></th><td>TR(Dec)</td><td>✓</td><td>✓</td><td>✓</td><td>✗</td></tr><tr><th>DeformTime <sup><a href="#fn:41">41</a></sup></th><td>TR+RNN</td><td>✓</td><td>✓</td><td>✓</td><td>✗</td></tr><tr><th>Sonnet <sup><a href="#fn:42">42</a></sup></th><td>TR</td><td>✓</td><td>✗</td><td>✓</td><td>✗</td></tr></tbody></table>

- The symbol ✧ in TimeXer indicates its numerical evaluation with known covariates is not originally reported.
- Arch.: Architecture; TR: Transformer; TR(Dec): Transformer (Decoder-only); Obs. cov.: Observed covariate; Kno. cov.: Known covariate.

However, these models fall short in flexibly handling multiple targets, observed covariates, and known covariates, as shown in Table I. Indeed, Timer-XL [^8], DeformTime [^41], and Sonnet [^42] rely solely on observed covariates, whereas N-BEATSx [^38], TiDE [^40], and ExoTST [^9] can handle only known covariates as they always require future covariate information. Although TimeXer [^31] is architecturally capable of handling both types of covariates, its numerical evaluations focus exclusively on settings with observed covariates. Our experiments indicate that its variate-level dependency modeling design fails to accurately reflect the localized impact of known covariates. More importantly, this design aggregates information from the entire temporal span into each variable token, making it incompatible with the autoregressive causal structure inherent to decoder-only Transformers such as CITRAS.

A few exceptions, such as the Temporal Fusion Transformer (TFT) [^43] and TSMixer-Ext [^44], can handle all variable types. TFT combines recurrent encoders with an interpretable attention mechanism to integrate both types of covariates within a unified framework. It employs gating layers and variable-selection networks to dynamically filter relevant features and utilizes a multi-head attention decoder to capture long-range temporal dependencies. TSMixer-Ext [^44] consists of MLP-based mixer layers along both temporal and variate dimensions, embedding observed and known covariates into a shared representation early in the model. However, these models adopt complex architectures to integrate heterogeneous variables, which are incompatible with decoder-only Transformers. Consequently, as highlighted in Table I, there is still no established method for extending decoder-only Transformers—known for their strong autoregressive capability—to flexibly incorporate both observed and known covariates. CITRAS addresses this gap by seamlessly integrating both types of covariates into a decoder-only Transformer via its KV Shift mechanism, fully preserving its strong autoregressive performance.

## III CITRAS

### III-A Problem Setting

Let $\mathbf{X}_{1:T}^{tgt,:}=\{\mathbf{X}_{1:T}^{tgt,1},\mathbf{X}_{1:T}^{tgt,2},...,\mathbf{X}_{1:T}^{tgt,C_{tgt}}\}\in\mathbb{R}^{T\times C_{tgt}}$ be a multivariate target time series of length $T$ with $C_{tgt}$ variables. With optional usage of observed covariates $\mathbf{X}_{1:T}^{obs,:}\in\mathbb{R}^{T\times C_{obs}}$ and known covariates $\mathbf{X}_{1:T+S}^{knw,:}\in\mathbb{R}^{(T+S)\times C_{knw}}$, the goal of the forecasting model $\mathcal{F}_{\theta}$ is to predict the future $S$ time steps of the target time series, which can be formulated as follows:

$$
\widehat{\mathbf{X}}_{T+1:T+S}^{tgt,:}=\mathcal{F}_{\theta}\left(\mathbf{X}_{1:T}^{tgt,:},\mathbf{X}_{1:T}^{obs,:},\mathbf{X}_{1:T+S}^{knw,:}\right)
$$

### III-B Architecture

As shown in Figure 2, CITRAS is a patch-based, decoder-only Transformer with $L$ layers of cross-time attention modules and cross-variate attention modules. Our core mechanisms, KV Shift and Attention Score Smoothing, reside in the cross-variate attention module.

![Refer to caption](https://arxiv.org/html/2503.24007v4/x2.png)

Figure 2: Overall structure of CITRAS. (a) The embedding module applies patching to each variable, yielding temporal token embeddings. (b) The cross-time attention module captures intra-variate cross-time dependencies. (c) The cross-variate attention module captures cross-variate dependencies. (d) KV Shift associates the key of the known covariate with the value from one patch step ahead. Attention Score Smoothing calculates the Exponential Moving Average (EMA) of patch-wise attention scores. (e) The projection layer projects each target token embedding to the values of the next patch.

#### III-B1 Embedding

First, all sequences are segmented into non-overlapping patches of length $P$, and each patch is embedded into a $D$ -dimensional token. For simplicity, we assume that both $T$ and $S$ are divisible by $P$. Taking a target variable $c$ as an example, this can be formalized as:

$$
\displaystyle\{\mathbf{s}_{1}^{tgt,c},\mathbf{s}_{2}^{tgt,c},.,\mathbf{s}_{N_{tgt}}^{tgt,c}\}=\operatorname{Patchify}\left(\mathbf{X}_{1:T}^{tgt,c}\right)
$$
 
$$
\displaystyle\mathbf{H}_{i}^{tgt,c}=\operatorname{Embed}\left(\mathbf{s}_{i}^{tgt,c}\right),~~i=1,.,N_{tgt}
$$

where $N_{tgt}=\frac{T}{P}$ is the number of patches and $\operatorname{Embed}:\mathbb{R}^{P}\to\mathbb{R}^{D}$ is a shared linear projector across all variables. We denote the token embeddings of a target variable at all patch steps as $\mathbf{H}_{:}^{tgt,c}=~\{\mathbf{H}_{i}^{tgt,c}\}_{i=1}^{N_{tgt}}~\in~\mathbb{R}^{N_{tgt}\times D}$. Similarly, the token embeddings of an observed covariate are denoted as $\mathbf{H}_{:}^{obs,c}\in\mathbb{R}^{N_{obs}\times D}$, and those of a known covariate are denoted as $\mathbf{H}_{:}^{knw,c}\in\mathbb{R}^{N_{knw}\times D}$, all obtained using the shared parameters with the target variables. Here, $N_{obs}=N_{tgt}=\frac{T}{P}$ and $N_{knw}=\frac{(T+S)}{P}$.

#### III-B2 Cross-Time Attention

In the cross-time attention module, we apply multi-head attention with causal masking to all variables to capture their intra-variate cross-time dependencies. Following Timer-XL [^8], we adopt Rotary Position Embedding (RoPE) [^45] to capture temporal order. Taking a target variable $c$ as an example, and dropping the layer index for brevity, this can be formalized as:

$$
\displaystyle\widetilde{\mathbf{H}}_{:}^{tgt,c}
$$
 
$$
\displaystyle=\operatorname{LN}\left(\mathbf{H}_{:}^{tgt,c}+\operatorname{MHA}\left(\mathbf{H}_{:}^{tgt,c},\mathbf{H}_{:}^{tgt,c},\mathbf{H}_{:}^{tgt,c}\right)\right)
$$
 
$$
\displaystyle\mathbf{H}_{:}^{tgt,c}
$$
 
$$
\displaystyle=\operatorname{LN}\left(\widetilde{\mathbf{H}}_{:}^{tgt,c}+\operatorname{FFN}\left(\widetilde{\mathbf{H}}_{:}^{tgt,c}\right)\right)
$$

where $\operatorname{LN}$ denotes layer normalization [^46], $\operatorname{MHA}\left(\mathbf{Q},\mathbf{K},\mathbf{V}\right)$ denotes the multi-head attention layer where $\mathbf{Q}$, $\mathbf{K}$, and $\mathbf{V}$ serve as queries, keys, and values, and $\operatorname{FFN}$ denotes a feed-forward network.

Again, token embeddings of observed covariates $\mathbf{H}_{:}^{obs,c}$ and those of known covariates $\mathbf{H}_{:}^{knw,c}$ are similarly processed with the shared parameters.

#### III-B3 Cross-Variate Attention — KV Shift

Attention is recognized as an effective approach for capturing cross-variate dependencies [^10]. To represent the dependencies of target variables while avoiding unnecessary interactions among covariates, we adopt multi-head attention, where queries originate from the target variables, whereas keys and values are derived from all available variables. However, applying attention across the variables at each patch step, as adopted by Crossformer [^6], fails to leverage pivotal future patches of known covariates, as they do not have corresponding target patches.

To address this issue, we introduce a KV Shift mechanism in the multi-head attention layer of the cross-variate attention module, which associates the key of the known covariate with the value from one patch step ahead. Specifically, the key and value at patch step $i$ can be formalized as:

$$
\displaystyle\text{Key:}~~~\mathbf{H}_{i}^{k,:}
$$
 
$$
\displaystyle=\left[\mathbf{H}_{i}^{tgt,:},\mathbf{H}_{i}^{obs,:},\mathbf{H}_{i}^{knw,:}\right]
$$
 
$$
\displaystyle\text{Value:}~~~\mathbf{H}_{i}^{v,:}
$$
 
$$
\displaystyle=\left[\mathbf{H}_{i}^{tgt,:},\mathbf{H}_{i}^{obs,:},\mathbf{H}_{i+1}^{knw,:}\right]
$$

where $\left[\cdot,\cdot\right]$ denotes the concatenation along the variate dimension and $\mathbf{H}_{i}^{k,:},\mathbf{H}_{i}^{v,:}\in\mathbb{R}^{\left(C_{tgt}+C_{obs}+C_{knw}\right)\times D}$. After this, cross-variate dependencies can be captured by the standard multi-head attention layer:

$$
\displaystyle\widetilde{\mathbf{H}}_{i}^{tgt,:}
$$
 
$$
\displaystyle=\operatorname{LN}\left(\mathbf{H}_{i}^{tgt,:}+\operatorname{MHA}\left(\mathbf{H}_{i}^{tgt,:},\mathbf{H}_{i}^{k,:},\mathbf{H}_{i}^{v,:}\right)\right)
$$
 
$$
\displaystyle\mathbf{H}_{i}^{tgt,:}
$$
 
$$
\displaystyle=\operatorname{LN}\left(\widetilde{\mathbf{H}}_{i}^{tgt,:}+\operatorname{FFN}\left(\widetilde{\mathbf{H}}_{i}^{tgt,:}\right)\right)
$$

for $i=1,...,N_{tgt}$. KV Shift maintains step alignment in the dot product calculation between queries and keys, enabling precise capture of concurrent dependencies. As a result, the model first identifies how strongly each known covariate influences the target variable at the current step. It then allows the target token to incorporate the next-step future known covariate (value) based on the strength of these obtained dependencies. As the target token gradually transforms into the prediction for the next step, this facilitates a natural flow of information for exploiting future information from known covariates. Note that KV Shift is applied exclusively to attention from target queries to known covariates, and never to target–target or target–observed-covariate interactions, thereby preventing unrealistic access to future information or temporal leakage.

#### III-B4 Cross-Variate Attention — Attention Score Smoothing

The attention score of $\operatorname{MHA}\left(\mathbf{Q},\mathbf{K},\mathbf{V}\right)$ is calculated as:

$$
\widetilde{\mathbf{A}}=\left(\mathbf{Q}\mathbf{W}_{q}\right)\left(\mathbf{K}\mathbf{W}_{k}\right)^{\top}
$$

where $\mathbf{W}_{q},\mathbf{W}_{k}\in\mathbb{R}^{D\times d_{k}}$ and $d_{k}$ is the dimension of the query, key, and value. In the cross-variate attention module, this attention score is calculated for each patch step separately, capturing fine-grained dependencies between variables at that step. However, time series data often exhibit local disturbances, and sparse variables representing infrequent events may maintain constant values within a patch. In such scenarios, local patch-wise attention fails to represent global variate-level dependencies.

To address this issue, we introduce Attention Score Smoothing in the multi-head attention layer of the cross-variate attention module, which transforms locally accurate patch-wise dependencies into global variate-level dependencies. Specifically, the attention score at each step is smoothed based on the attention scores up to that step in each head. We employ an Exponential Moving Average (EMA) as a smoothing method, as it enables adaptation to shifting correlations between variables [^47] by assigning exponentially decreasing weights over time. Denoting original attention score at patch step $i$ as $\widetilde{\mathbf{A}}_{i}$, the smoothed attention score $\mathbf{A}_{i}$ can be calculated as:

$$
\displaystyle\mathbf{A}_{i}
$$
 
$$
\displaystyle=\alpha\widetilde{\mathbf{A}}_{i}+\left(1-\alpha\right)\mathbf{A}_{i-1},~~i=2,.,N_{tgt}
$$

where $\mathbf{A}_{1}=\widetilde{\mathbf{A}}_{1}$ and $\alpha$ is a smoothing factor, which we consider as a shared hyperparameter across all cross-variate attention modules and heads. Its design is inspired by temporal smoothing and regularization principles in time series analysis, where aggregating information over time reduces variance in noisy estimates. By applying an exponential moving average to patch-wise attention scores, the model stabilizes cross-variate dependency estimation while remaining adaptive to gradual changes.

It is important to note that the objective and mechanism of Attention Score Smoothing are completely different from Exponential Smoothing Attention in ETSformer [^48]. Exponential Smoothing Attention aims for cross-time dependency modeling. It replaces the original token-similarity-based attention score with exponentially decreasing weights over time. In contrast, Attention Score Smoothing is for cross-variate dependency modeling. We first utilize the original attention to obtain cross-variate dependencies based on token similarity at each patch step, and then smooth these patch-wise dependencies over time to obtain global variate-level dependencies.

#### III-B5 Projection

Following the next-token prediction approach common in decoder-only Transformers, each token embedding of the target variables is used to predict the values of the next patch. For $i=1,...,N_{tgt}$ and $c=1,...,C_{tgt}$, this can be formalized as:

$$
\widehat{\mathbf{X}}_{iP+1:(i+1)P}^{tgt,c}=\operatorname{Project}\left(\mathbf{H}_{i}^{tgt,c}\right)
$$

where $\operatorname{Project}:\mathbb{R}^{D}\to\mathbb{R}^{P}$ is a shared linear projector across all steps and target variables. In the training phase, we calculate the squared loss using all of these outputs. In the testing phase, the outputs from the last patches are used for forecasting. When the forecasting horizon $S$ exceeds the patch length $P$, the output target values are integrated into subsequent inputs for recursive forecasting.

## IV Experiments

To verify the effectiveness and versatility of CITRAS, we extensively evaluated it in two settings, covariate-informed forecasting and well-established multivariate forecasting.

### IV-A Covariate-Informed Forecasting

#### IV-A1 Dataset

We use seven real-world datasets for covariate-informed forecasting, with detailed information provided in Table II. They include target variable(s), observed covariates, and known covariates, thereby providing a realistic representation of practical forecasting scenarios.

EPF datasets [^49] consist of five subsets from different day-ahead electricity markets, each spanning six years: EPF-NP documents the Nord Pool electricity market from 2013-01-01 to 2018-12-24, containing hourly electricity prices as a target variable, with corresponding grid load and wind power forecasts as known covariates. EPF-PJM records the Pennsylvania-New Jersey-Maryland market from 2013-01-01 to 2018-12-24, containing zonal electricity prices in the Commonwealth Edison (COMED) as a target variable, with corresponding system load and COMED load forecasts as known covariates. EPF-BE captures Belgium’s electricity market from 2011-01-09 to 2016-12-31, containing hourly electricity prices as a target variable, with corresponding load forecasts in Belgium and generation forecasts in France as known covariates. EPF-FR documents the electricity market in France from 2012-01-09 to 2017-12-31, containing hourly prices as a target variable, with corresponding load and generation forecasts as known covariates. EPF-DE records the German electricity market from 2012-01-09 to 2017-12-31, containing hourly prices as a target variable, with zonal load forecasts in the TSO Amprion zone and wind and solar generation forecasts as known covariates.

EDF [^4] is an hourly time series dataset spanning over five years from 2015-01-03 to 2020-06-27, where the target variable is the national electricity demand in Panama. Observed covariates include four meteorological indicators (air temperature, specific humidity, total precipitable liquid water, and wind speed) measured in three cities (Tocumen, Santiago, David) in Panama, resulting in a total of 12 variables. In addition, we adopt three binary calendar features (holiday, school day, weekend) as known covariates.

BS [^50] is an hourly time series dataset spanning two years from 2011-01-01 to 2012-12-31, where the target variables are three types of rental counts (casual, registered, and total count) in the bike sharing system in Washington, D.C. Observed covariates include five meteorological indicators (weather situation, temperature, feeling temperature, humidity, and windspeed). In addition, we adopt three calendar features (holiday, weekday, working day) as known covariates.

Following previous work [^38], we set the input length as 168 and forecasting length as 24.

TABLE II: Dataset descriptions.

<table><thead><tr><th>Task</th><th>Dataset</th><th>#Target</th><th>#Observed</th><th>#Known</th><th>Sampling Frequency</th><th>Dataset Size</th></tr></thead><tbody><tr><td rowspan="3">Covariate-informed Forecasting</td><td>EPFs</td><td>1</td><td>0</td><td>2</td><td>1 Hour</td><td>(36500, 5219, 10460)</td></tr><tr><td>EDF</td><td>1</td><td>12</td><td>3</td><td>1 Hour</td><td>(33442, 4783, 9586)</td></tr><tr><td>BS</td><td>3</td><td>5</td><td>3</td><td>1 Hour</td><td>(11974, 1716, 3452)</td></tr><tr><td rowspan="7">Multivariate Forecasting</td><td>ETTh</td><td>7</td><td>0</td><td>0</td><td>1 Hour</td><td>(8545, 2881, 2881)</td></tr><tr><td>ETTm</td><td>7</td><td>0</td><td>0</td><td>15 Minutes</td><td>(34465, 11521, 11521)</td></tr><tr><td>ECL</td><td>321</td><td>0</td><td>0</td><td>1 Hour</td><td>(18317, 2633, 5261)</td></tr><tr><td>Weather</td><td>21</td><td>0</td><td>0</td><td>10 Minutes</td><td>(36792, 5271, 10540)</td></tr><tr><td>Traffic</td><td>862</td><td>0</td><td>0</td><td>1 Hour</td><td>(12185, 1757, 3509)</td></tr><tr><td>PEMS04</td><td>307</td><td>0</td><td>0</td><td>5 Minutes</td><td>(10100, 3400, 3399)</td></tr><tr><td>PEMS08</td><td>170</td><td>0</td><td>0</td><td>5 Minutes</td><td>(10618, 3573, 3572)</td></tr></tbody></table>

- #Target, #Observed, and #Known refer to the number of target variables, observed covariates, and known covariates, respectively. The dataset size is presented as (Train, Validation, Test).

#### IV-A2 Baselines

TABLE III: Comparison of Supported Variable Types for Baseline Models, Including Our Extensions.

<table><tbody><tr><td rowspan="2">Models</td><td>Uni-</td><td>Multi-</td><td>Observed</td><td>Known</td></tr><tr><td>variate</td><td>variate</td><td>covariate</td><td>covariate</td></tr><tr><td>TFT <sup><a href="#fn:43">43</a></sup></td><td rowspan="3">✓</td><td rowspan="3">✓</td><td rowspan="3">✓</td><td rowspan="3">✓</td></tr><tr><td>TSMixer-Ext <sup><a href="#fn:44">44</a></sup></td></tr><tr><td>TimeXer <sup><a href="#fn:31">31</a></sup></td></tr><tr><td>TiDE <sup><a href="#fn:40">40</a></sup></td><td>✓</td><td>✗</td><td>✗</td><td>✓</td></tr><tr><td>Timer-XL <sup><a href="#fn:8">8</a></sup></td><td>✓</td><td>✓</td><td>✓</td><td>✧</td></tr><tr><td>iTransformer <sup><a href="#fn:10">10</a></sup></td><td rowspan="2">✓</td><td rowspan="2">✓</td><td rowspan="2">✦</td><td rowspan="2">✧</td></tr><tr><td>Leddam <sup><a href="#fn:32">32</a></sup></td></tr><tr><td>CARD <sup><a href="#fn:7">7</a></sup></td><td rowspan="4">✓</td><td rowspan="4">✓</td><td rowspan="4">✦</td><td rowspan="4">✗</td></tr><tr><td>ModernTCN <sup><a href="#fn:51">51</a></sup></td></tr><tr><td>TimesNet <sup><a href="#fn:52">52</a></sup></td></tr><tr><td>Crossformer <sup><a href="#fn:6">6</a></sup></td></tr><tr><td>FITS <sup><a href="#fn:23">23</a></sup></td><td rowspan="3">✓</td><td rowspan="3">✗</td><td rowspan="3">✗</td><td rowspan="3">✗</td></tr><tr><td>DLinear <sup><a href="#fn:22">22</a></sup></td></tr><tr><td>PatchTST <sup><a href="#fn:18">18</a></sup></td></tr></tbody></table>

- The symbol ✦ indicates that the model treats the target and observed covariates jointly without explicit distinction. The symbol ✧ indicates that the model was extended by us to support known covariates in our experiments.

We include 14 baseline models, and Table III summarizes the variable types supported by each model. TFT [^43], TSMixer-Ext [^44], and TimeXer [^31] originally support multivariate, observed covariates, and known covariates, while TiDE [^40] supports univariate with known covariates. These models are used in their original forms. Timer-XL [^8] originally supports only observed covariates, so we extend it to use known covariates. Specifically, we modify the masking scheme in TimeAttention so that target token embeddings can attend to one-step ahead future token embeddings of known covariates. Since iTransformer [^10], Leddam [^32], CARD [^7], ModernTCN [^51], TimesNet [^52], and Crossformer [^6] are originally multivariate models, they inherently treat the target and observed covariates jointly without explicit distinction. Among them, we additionally extend iTransformer and Leddam to incorporate known covariates by introducing a variate embedding layer to accommodate their longer sequence lengths. The details of the extension of Timer-XL, iTransformer, and Leddam are provided in the Appendix (Section VI-E). In contrast, FITS [^23], DLinear [^22], and PatchTST [^18] are univariate models and they are used in their original forms.

Future information from known covariates is expected to directly affect the forecasting accuracy. For a fair comparison with baselines that cannot utilize known covariates, we evaluate CITRAS both with and without known covariates (i.e., without using KV Shift). In the setting without known covariates, the future portion of known covariates is omitted and treated similarly to observed covariates. We reproduce all baseline implementations based on the TimesNet [^52] repository or their official repositories.

#### IV-A3 Implementation Details

All experiments were implemented in PyTorch [^53]. We utilize a single NVIDIA V100 32GB GPU for covariate-informed forecasting tasks. We apply Series Stationarization [^54] to the embedding and projection layers of CITRAS to mitigate nonstationarity. We adopt Adam [^55] with an initial learning rate of $10^{-4}$ and L2 loss for model optimization. The training process is conducted for up to 10 epochs with an early stopping mechanism, which terminates training if the validation performance does not improve for 3 consecutive epochs. We set the number of layers in our proposed model $L\in\{1,2,3,4\}$, the number of heads in the multi-head attention layers as 8, embedding dimension $D\in\{128,256,512,1024\}$, and the smoothing factor for Attention Score Smoothing $\alpha\in\{0.1,0.2,0.4,0.8\}$. All hyperparameters are selected based on the validation performance using a grid search approach. Hyperparameters of the baseline models are tuned within the ranges reported in their respective original papers following the same procedure. For all models, the patch length is uniformly fixed to 24.

We ensure that the drop last option is set to False in the test data loader to avoid any unfair impact, as pointed out by [^56]. All evaluations in covariate-informed forecasting tasks are conducted with three random seeds and the average performance is reported. The standard deviation is provided in the Appendix (Section VI-A).

#### IV-A4 Results

Table IV presents the results of covariate-informed forecasting with and without the use of known covariates. In the former setting, CITRAS outperforms TFT, TSMixer-Ext, and TiDE, all of which employ complex architectures to handle future information from known covariates. This superior performance can be attributed to the simple yet effective design of CITRAS, which seamlessly integrates the future information from covariates into the decoder-only Transformer through KV Shift, thereby preserving its inherent expressive power. Among recent models that adhere to the canonical Transformer architecture, TimeXer captures cross-variate dependencies at the variate level, while Timer-XL does so at the patch level. The relative superiority of these contrasting methods is dataset-dependent. In contrast, CITRAS leverages the advantages of both approaches through Attention Score Smoothing, consistently demonstrating strong performance across datasets. Moreover, the superiority of CITRAS remains evident even in conventional settings without known covariates. This is because multivariate baselines treat target variables and covariates equally and suffer from unnecessary interactions among covariates. Also, the univariate baselines cannot account for target fluctuations caused by observed covariates, leading to poor modeling of cross-time dependencies. In contrast, CITRAS applies attention only from target variables to covariates, thereby effectively utilizing observed covariates without causing unnecessary interactions.

Figure 3 presents the visualization results for the EPF-NP, EDF, and BS datasets. Forecasts without any covariates are shown in green, while those with all covariates are shown in blue. In the EPF-NP result, CITRAS accurately predicts the sudden rise in target electricity prices by taking into account the high grid load and low power generation forecasts during the forecasting horizon. In the EDF result, it adjusts the electricity demand downward by recognizing that the forecasting horizon falls on a holiday. In the BS result, it effectively captures the negative impact of the binary working day covariate and accurately forecasts the increase in rental counts on the day following a working day. These results indicate that CITRAS does not simply incorporate covariates that resemble the target shape, but rather captures complex relationships from a global perspective and accurately reflects the local impact of future known covariates. In contrast, TimeXer leverages cross-variate dependencies only at the variate level, and thus struggles to adjust the forecasts locally accurately. These comparisons highlight the strength of CITRAS that effectively utilizes covariates at multiple granularities.

TABLE IV: Results of the covariate-informed forecasting with and without known covariates. Average MSE and MAE across three random seeds are reported.

<table><thead><tr><th colspan="2">Model</th><th colspan="2">CITRAS</th><th colspan="2">TFT</th><th colspan="2">TSMixer-Ext</th><th colspan="2">TimeXer</th><th colspan="2">TiDE</th><th colspan="2">Timer-XL</th><th colspan="2">iTrans.</th><th colspan="2">Leddam</th></tr><tr><th colspan="2">Metric</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th></tr></thead><tbody><tr><th rowspan="7">w/ known</th><th>EPF–NP</th><td>0.172</td><td>0.215</td><td>0.193</td><td>0.236</td><td>0.187</td><td>0.251</td><td>0.180</td><td>0.228</td><td>0.298</td><td>0.311</td><td>0.175</td><td>0.218</td><td>0.211</td><td>0.254</td><td>0.254</td><td>0.285</td></tr><tr><th>EPF-PJM</th><td>0.063</td><td>0.153</td><td>0.084</td><td>0.177</td><td>0.078</td><td>0.172</td><td>0.081</td><td>0.176</td><td>0.106</td><td>0.214</td><td>0.065</td><td>0.157</td><td>0.077</td><td>0.170</td><td>0.090</td><td>0.196</td></tr><tr><th>EPF–BE</th><td>0.363</td><td>0.248</td><td>0.386</td><td>0.245</td><td>0.343</td><td>0.249</td><td>0.368</td><td>0.232</td><td>0.449</td><td>0.302</td><td>0.422</td><td>0.274</td><td>0.345</td><td>0.247</td><td>0.372</td><td>0.255</td></tr><tr><th>EPF–FR</th><td>0.360</td><td>0.176</td><td>0.373</td><td>0.188</td><td>0.426</td><td>0.216</td><td>0.366</td><td>0.191</td><td>0.411</td><td>0.253</td><td>0.429</td><td>0.199</td><td>0.384</td><td>0.207</td><td>0.418</td><td>0.222</td></tr><tr><th>EPF–DE</th><td>0.219</td><td>0.293</td><td>0.270</td><td>0.331</td><td>0.252</td><td>0.324</td><td>0.293</td><td>0.324</td><td>0.522</td><td>0.467</td><td>0.229</td><td>0.301</td><td>0.263</td><td>0.329</td><td>0.283</td><td>0.337</td></tr><tr><th>EDF</th><td>0.071</td><td>0.194</td><td>0.090</td><td>0.224</td><td>0.080</td><td>0.212</td><td>0.071</td><td>0.198</td><td>0.135</td><td>0.261</td><td>0.081</td><td>0.208</td><td>0.072</td><td>0.197</td><td>0.080</td><td>0.207</td></tr><tr><th>BS</th><td>0.282</td><td>0.314</td><td>0.375</td><td>0.377</td><td>0.344</td><td>0.358</td><td>0.340</td><td>0.356</td><td>0.490</td><td>0.447</td><td>0.286</td><td>0.317</td><td>0.350</td><td>0.355</td><td>0.383</td><td>0.388</td></tr><tr><th colspan="2">Model</th><th colspan="2">CITRAS</th><th colspan="2">CARD</th><th colspan="2">ModernTCN</th><th colspan="2">TimesNet</th><th colspan="2">Cross.</th><th colspan="2">FITS</th><th colspan="2">DLinear</th><th colspan="2">PatchTST</th></tr><tr><th colspan="2">Metric</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th></tr><tr><th rowspan="7">w/o known</th><th>EPF–NP</th><td>0.227</td><td>0.260</td><td>0.266</td><td>0.293</td><td>0.237</td><td>0.274</td><td>0.247</td><td>0.283</td><td>0.248</td><td>0.288</td><td>0.304</td><td>0.316</td><td>0.309</td><td>0.321</td><td>0.264</td><td>0.287</td></tr><tr><th>EPF-PJM</th><td>0.090</td><td>0.182</td><td>0.114</td><td>0.218</td><td>0.100</td><td>0.195</td><td>0.101</td><td>0.201</td><td>0.104</td><td>0.198</td><td>0.109</td><td>0.216</td><td>0.108</td><td>0.214</td><td>0.108</td><td>0.213</td></tr><tr><th>EPF–BE</th><td>0.405</td><td>0.276</td><td>0.425</td><td>0.287</td><td>0.393</td><td>0.267</td><td>0.416</td><td>0.280</td><td>0.414</td><td>0.283</td><td>0.460</td><td>0.308</td><td>0.463</td><td>0.314</td><td>0.408</td><td>0.260</td></tr><tr><th>EPF–FR</th><td>0.407</td><td>0.219</td><td>0.425</td><td>0.249</td><td>0.410</td><td>0.231</td><td>0.406</td><td>0.221</td><td>0.433</td><td>0.213</td><td>0.403</td><td>0.255</td><td>0.429</td><td>0.260</td><td>0.412</td><td>0.215</td></tr><tr><th>EPF–DE</th><td>0.420</td><td>0.404</td><td>0.470</td><td>0.434</td><td>0.445</td><td>0.416</td><td>0.484</td><td>0.435</td><td>0.525</td><td>0.423</td><td>0.533</td><td>0.473</td><td>0.523</td><td>0.465</td><td>0.463</td><td>0.431</td></tr><tr><th>EDF</th><td>0.087</td><td>0.206</td><td>0.089</td><td>0.212</td><td>0.088</td><td>0.216</td><td>0.108</td><td>0.238</td><td>0.112</td><td>0.236</td><td>0.134</td><td>0.258</td><td>0.133</td><td>0.259</td><td>0.103</td><td>0.232</td></tr><tr><th>BS</th><td>0.289</td><td>0.321</td><td>0.414</td><td>0.416</td><td>0.377</td><td>0.401</td><td>0.394</td><td>0.382</td><td>0.339</td><td>0.361</td><td>0.484</td><td>0.439</td><td>0.469</td><td>0.429</td><td>0.349</td><td>0.377</td></tr></tbody></table>

- The input length and forecasting length are set to 168 and 24, respectively. The best result is represented in bold, followed by underline.

![Refer to caption](https://arxiv.org/html/2503.24007v4/x3.png)

Figure 3: Forecasting examples of CITRAS and TimeXer without (shown in green) and with (shown in blue) the use of covariates. The target variable is displayed in the top box, while the known covariates are shown in the boxes below.

### IV-B Multivariate Forecasting

For a comprehensive comparison, we further evaluate our model on well-established multivariate forecasting benchmarks.

#### IV-B1 Dataset

We use nine real-world datasets for multivariate forecasting without using any covariates, with detailed information provided in Table II.

- ETT [^16]: Contains 7 monitoring factors in electricity transformers from July 2016 to July 2018. ETTh1 and ETTh2 are hourly subsets, and ETTm1 and ETTm2 are subsets recorded every 15 minutes.
- Weather [^17]: Includes 21 meteorological factors recorded every 10 minutes from the Weather Station of the Max Planck Biogeochemistry Institute in 2020.
- ECL [^17]: Comprises hourly electricity consumption data from 321 clients.
- Traffic [^17]: Records hourly road occupancy rates measured by 862 sensors on San Francisco Bay area freeways from January 2015 to December 2016.
- PEMS04 [^57]: Contains public traffic network data recorded at 307 locations in California every 5 minutes from 2018-01-01 to 2018-02-28.
- PEMS08 [^57]: Contains data at 170 locations from 2016-07-01 to 2016-08-31.

We use all the variables in these datasets as target variables. To reflect practical scenarios, we adopt the rolling forecasting approach [^58], where a model trained with predetermined input and forecasting lengths is evaluated with a longer forecasting length by iteratively integrating its output into the subsequent input. Following previous works [^58] [^8], we set the input length to 672, the training forecasting length to 96, and the testing forecasting length to {96, 192, 336, 720}.

#### IV-B2 Implementation Details

We utilize a single NVIDIA A100 80GB GPU for the ECL, Traffic, and PEMS datasets and a single NVIDIA V100 32GB GPU for the other datasets. The patch length is uniformly set to 96. All evaluations in multivariate forecasting tasks are conducted with a fixed random seed and a single run, following the standard practice in previous works [^31] [^8]. Other settings are the same as those in covariate-informed forecasting.

#### IV-B3 Results

Table V presents the results of multivariate forecasting. Compared to recent models specifically designed for multivariate settings, CITRAS achieves superior or competitive performance across most datasets. In particular, it performs strongly on the ECL, Traffic, PEMS04, and PEMS08 datasets, which involve a large number of target variables and require both accurate and scalable modeling of complex inter-variable relationships. Overall, CITRAS ranks first more frequently than any other model, highlighting the effectiveness of its design and its robustness across diverse forecasting scenarios.

TABLE V: Full results of the multivariate forecasting

<table><tbody><tr><th colspan="2">Models</th><td colspan="2">CITRAS</td><td colspan="2">TimeXer</td><td colspan="2">Timer-XL</td><td colspan="2">iTrans.</td><td colspan="2">Leddam</td><td colspan="2">CARD</td><td colspan="2"><p></p><p>ModernTCN</p><p></p></td><td colspan="2">FITS</td><td colspan="2">DLinear</td></tr><tr><th colspan="2">Metric</th><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><th rowspan="5">ETTh1</th><th>96</th><td>0.355</td><td>0.395</td><td>0.391</td><td>0.425</td><td>0.364</td><td>0.397</td><td>0.387</td><td>0.418</td><td>0.371</td><td>0.406</td><td>0.373</td><td>0.403</td><td>0.372</td><td>0.400</td><td>0.378</td><td>0.401</td><td>0.369</td><td>0.400</td></tr><tr><th>192</th><td>0.386</td><td>0.416</td><td>0.420</td><td>0.443</td><td>0.405</td><td>0.424</td><td>0.416</td><td>0.437</td><td>0.397</td><td>0.422</td><td>0.402</td><td>0.422</td><td>0.405</td><td>0.421</td><td>0.410</td><td>0.420</td><td>0.405</td><td>0.422</td></tr><tr><th>336</th><td>0.403</td><td>0.429</td><td>0.441</td><td>0.459</td><td>0.427</td><td>0.439</td><td>0.434</td><td>0.450</td><td>0.417</td><td>0.435</td><td>0.424</td><td>0.436</td><td>0.428</td><td>0.437</td><td>0.431</td><td>0.433</td><td>0.435</td><td>0.445</td></tr><tr><th>720</th><td>0.412</td><td>0.445</td><td>0.460</td><td>0.483</td><td>0.439</td><td>0.459</td><td>0.447</td><td>0.473</td><td>0.441</td><td>0.463</td><td>0.459</td><td>0.472</td><td>0.446</td><td>0.468</td><td>0.441</td><td>0.451</td><td>0.493</td><td>0.508</td></tr><tr><th>Ave</th><td>0.389</td><td>0.421</td><td>0.428</td><td>0.453</td><td>0.409</td><td>0.430</td><td>0.421</td><td>0.445</td><td>0.406</td><td>0.431</td><td>0.414</td><td>0.433</td><td>0.413</td><td>0.432</td><td>0.415</td><td>0.426</td><td>0.426</td><td>0.444</td></tr><tr><th rowspan="5">ETTh2</th><th>96</th><td>0.293</td><td>0.358</td><td>0.281</td><td>0.350</td><td>0.301</td><td>0.354</td><td>0.307</td><td>0.365</td><td>0.275</td><td>0.345</td><td>0.282</td><td>0.345</td><td>0.271</td><td>0.340</td><td>0.277</td><td>0.342</td><td>0.286</td><td>0.354</td></tr><tr><th>192</th><td>0.351</td><td>0.395</td><td>0.339</td><td>0.391</td><td>0.360</td><td>0.396</td><td>0.376</td><td>0.406</td><td>0.330</td><td>0.381</td><td>0.338</td><td>0.381</td><td>0.327</td><td>0.378</td><td>0.334</td><td>0.377</td><td>0.344</td><td>0.393</td></tr><tr><th>336</th><td>0.372</td><td>0.416</td><td>0.371</td><td>0.419</td><td>0.382</td><td>0.419</td><td>0.416</td><td>0.435</td><td>0.355</td><td>0.402</td><td>0.361</td><td>0.401</td><td>0.351</td><td>0.399</td><td>0.358</td><td>0.398</td><td>0.369</td><td>0.416</td></tr><tr><th>720</th><td>0.419</td><td>0.453</td><td>0.422</td><td>0.459</td><td>0.443</td><td>0.468</td><td>0.432</td><td>0.456</td><td>0.394</td><td>0.434</td><td>0.380</td><td>0.423</td><td>0.396</td><td>0.433</td><td>0.387</td><td>0.426</td><td>0.408</td><td>0.454</td></tr><tr><th>Ave</th><td>0.359</td><td>0.405</td><td>0.353</td><td>0.405</td><td>0.372</td><td>0.409</td><td>0.383</td><td>0.415</td><td>0.339</td><td>0.391</td><td>0.340</td><td>0.388</td><td>0.336</td><td>0.388</td><td>0.339</td><td>0.386</td><td>0.352</td><td>0.404</td></tr><tr><th rowspan="5">ETTm1</th><th>96</th><td>0.282</td><td>0.340</td><td>0.300</td><td>0.358</td><td>0.297</td><td>0.347</td><td>0.312</td><td>0.367</td><td>0.303</td><td>0.351</td><td>0.305</td><td>0.351</td><td>0.311</td><td>0.360</td><td>0.308</td><td>0.351</td><td>0.307</td><td>0.350</td></tr><tr><th>192</th><td>0.329</td><td>0.369</td><td>0.342</td><td>0.381</td><td>0.344</td><td>0.376</td><td>0.351</td><td>0.390</td><td>0.339</td><td>0.372</td><td>0.336</td><td>0.370</td><td>0.347</td><td>0.380</td><td>0.338</td><td>0.367</td><td>0.337</td><td>0.368</td></tr><tr><th>336</th><td>0.365</td><td>0.391</td><td>0.383</td><td>0.402</td><td>0.382</td><td>0.399</td><td>0.389</td><td>0.412</td><td>0.370</td><td>0.392</td><td>0.365</td><td>0.387</td><td>0.381</td><td>0.400</td><td>0.367</td><td>0.384</td><td>0.366</td><td>0.386</td></tr><tr><th>720</th><td>0.423</td><td>0.425</td><td>0.450</td><td>0.436</td><td>0.450</td><td>0.437</td><td>0.458</td><td>0.449</td><td>0.423</td><td>0.422</td><td>0.415</td><td>0.416</td><td>0.438</td><td>0.433</td><td>0.418</td><td>0.413</td><td>0.418</td><td>0.418</td></tr><tr><th>Ave</th><td>0.350</td><td>0.382</td><td>0.368</td><td>0.394</td><td>0.368</td><td>0.390</td><td>0.377</td><td>0.404</td><td>0.358</td><td>0.384</td><td>0.355</td><td>0.381</td><td>0.369</td><td>0.393</td><td>0.358</td><td>0.379</td><td>0.357</td><td>0.380</td></tr><tr><th rowspan="5">ETTm2</th><th>96</th><td>0.174</td><td>0.260</td><td>0.176</td><td>0.263</td><td>0.180</td><td>0.263</td><td>0.183</td><td>0.272</td><td>0.163</td><td>0.253</td><td>0.162</td><td>0.252</td><td>0.178</td><td>0.269</td><td>0.164</td><td>0.255</td><td>0.166</td><td>0.259</td></tr><tr><th>192</th><td>0.231</td><td>0.299</td><td>0.235</td><td>0.301</td><td>0.244</td><td>0.306</td><td>0.241</td><td>0.310</td><td>0.218</td><td>0.292</td><td>0.214</td><td>0.287</td><td>0.235</td><td>0.307</td><td>0.219</td><td>0.293</td><td>0.222</td><td>0.300</td></tr><tr><th>336</th><td>0.280</td><td>0.333</td><td>0.287</td><td>0.335</td><td>0.299</td><td>0.344</td><td>0.296</td><td>0.346</td><td>0.271</td><td>0.328</td><td>0.263</td><td>0.319</td><td>0.288</td><td>0.340</td><td>0.272</td><td>0.328</td><td>0.278</td><td>0.339</td></tr><tr><th>720</th><td>0.365</td><td>0.389</td><td>0.368</td><td>0.386</td><td>0.377</td><td>0.397</td><td>0.393</td><td>0.409</td><td>0.363</td><td>0.385</td><td>0.351</td><td>0.375</td><td>0.366</td><td>0.388</td><td>0.360</td><td>0.382</td><td>0.379</td><td>0.405</td></tr><tr><th>Ave</th><td>0.262</td><td>0.320</td><td>0.266</td><td>0.321</td><td>0.275</td><td>0.327</td><td>0.278</td><td>0.334</td><td>0.254</td><td>0.315</td><td>0.247</td><td>0.308</td><td>0.267</td><td>0.326</td><td>0.254</td><td>0.314</td><td>0.262</td><td>0.326</td></tr><tr><th rowspan="5">ECL</th><th>96</th><td>0.128</td><td>0.222</td><td>0.131</td><td>0.231</td><td>0.127</td><td>0.219</td><td>0.133</td><td>0.229</td><td>0.130</td><td>0.224</td><td>0.129</td><td>0.225</td><td>0.161</td><td>0.270</td><td>0.142</td><td>0.244</td><td>0.138</td><td>0.238</td></tr><tr><th>192</th><td>0.149</td><td>0.242</td><td>0.149</td><td>0.248</td><td>0.145</td><td>0.236</td><td>0.158</td><td>0.258</td><td>0.148</td><td>0.241</td><td>0.147</td><td>0.242</td><td>0.172</td><td>0.281</td><td>0.156</td><td>0.256</td><td>0.152</td><td>0.251</td></tr><tr><th>336</th><td>0.164</td><td>0.259</td><td>0.168</td><td>0.268</td><td>0.159</td><td>0.252</td><td>0.168</td><td>0.262</td><td>0.164</td><td>0.257</td><td>0.164</td><td>0.259</td><td>0.183</td><td>0.292</td><td>0.173</td><td>0.272</td><td>0.167</td><td>0.268</td></tr><tr><th>720</th><td>0.199</td><td>0.288</td><td>0.214</td><td>0.309</td><td>0.187</td><td>0.277</td><td>0.205</td><td>0.294</td><td>0.202</td><td>0.291</td><td>0.202</td><td>0.292</td><td>0.216</td><td>0.321</td><td>0.213</td><td>0.304</td><td>0.203</td><td>0.302</td></tr><tr><th>Ave</th><td>0.160</td><td>0.253</td><td>0.166</td><td>0.264</td><td>0.155</td><td>0.246</td><td>0.164</td><td>0.258</td><td>0.161</td><td>0.253</td><td>0.160</td><td>0.255</td><td>0.183</td><td>0.291</td><td>0.171</td><td>0.269</td><td>0.165</td><td>0.265</td></tr><tr><th rowspan="5">Weather</th><th>96</th><td>0.152</td><td>0.205</td><td>0.150</td><td>0.203</td><td>0.157</td><td>0.205</td><td>0.174</td><td>0.225</td><td>0.149</td><td>0.201</td><td>0.147</td><td>0.204</td><td>0.153</td><td>0.210</td><td>0.145</td><td>0.197</td><td>0.169</td><td>0.229</td></tr><tr><th>192</th><td>0.200</td><td>0.249</td><td>0.192</td><td>0.242</td><td>0.206</td><td>0.250</td><td>0.227</td><td>0.268</td><td>0.194</td><td>0.243</td><td>0.191</td><td>0.244</td><td>0.199</td><td>0.250</td><td>0.186</td><td>0.237</td><td>0.211</td><td>0.268</td></tr><tr><th>336</th><td>0.253</td><td>0.291</td><td>0.240</td><td>0.280</td><td>0.259</td><td>0.291</td><td>0.290</td><td>0.309</td><td>0.242</td><td>0.283</td><td>0.238</td><td>0.282</td><td>0.251</td><td>0.290</td><td>0.241</td><td>0.277</td><td>0.258</td><td>0.306</td></tr><tr><th>720</th><td>0.325</td><td>0.342</td><td>0.309</td><td>0.329</td><td>0.337</td><td>0.344</td><td>0.374</td><td>0.360</td><td>0.314</td><td>0.335</td><td>0.309</td><td>0.334</td><td>0.323</td><td>0.341</td><td>0.342</td><td>0.347</td><td>0.320</td><td>0.362</td></tr><tr><th>Ave</th><td>0.233</td><td>0.272</td><td>0.223</td><td>0.263</td><td>0.240</td><td>0.273</td><td>0.266</td><td>0.291</td><td>0.225</td><td>0.265</td><td>0.221</td><td>0.266</td><td>0.232</td><td>0.273</td><td>0.229</td><td>0.265</td><td>0.239</td><td>0.291</td></tr><tr><th rowspan="5">Traffic</th><th>96</th><td>0.338</td><td>0.235</td><td>0.362</td><td>0.264</td><td>0.340</td><td>0.238</td><td>0.353</td><td>0.259</td><td>0.356</td><td>0.256</td><td>0.373</td><td>0.266</td><td>0.432</td><td>0.320</td><td>0.390</td><td>0.275</td><td>0.399</td><td>0.285</td></tr><tr><th>192</th><td>0.365</td><td>0.247</td><td>0.381</td><td>0.275</td><td>0.360</td><td>0.247</td><td>0.373</td><td>0.267</td><td>0.376</td><td>0.264</td><td>0.386</td><td>0.272</td><td>0.434</td><td>0.321</td><td>0.402</td><td>0.279</td><td>0.409</td><td>0.290</td></tr><tr><th>336</th><td>0.392</td><td>0.258</td><td>0.402</td><td>0.288</td><td>0.377</td><td>0.256</td><td>0.386</td><td>0.275</td><td>0.390</td><td>0.271</td><td>0.401</td><td>0.279</td><td>0.441</td><td>0.323</td><td>0.415</td><td>0.285</td><td>0.422</td><td>0.297</td></tr><tr><th>720</th><td>0.419</td><td>0.276</td><td>0.457</td><td>0.324</td><td>0.418</td><td>0.279</td><td>0.425</td><td>0.296</td><td>0.423</td><td>0.290</td><td>0.441</td><td>0.303</td><td>0.476</td><td>0.344</td><td>0.453</td><td>0.306</td><td>0.461</td><td>0.319</td></tr><tr><th>Ave</th><td>0.379</td><td>0.254</td><td>0.401</td><td>0.288</td><td>0.374</td><td>0.255</td><td>0.384</td><td>0.274</td><td>0.386</td><td>0.270</td><td>0.400</td><td>0.280</td><td>0.446</td><td>0.327</td><td>0.415</td><td>0.286</td><td>0.423</td><td>0.298</td></tr><tr><th rowspan="5">PEMS04</th><th>96</th><td>0.105</td><td>0.201</td><td>0.123</td><td>0.238</td><td>0.104</td><td>0.207</td><td>0.111</td><td>0.216</td><td>0.105</td><td>0.209</td><td>0.141</td><td>0.248</td><td>0.125</td><td>0.245</td><td>0.212</td><td>0.306</td><td>0.196</td><td>0.296</td></tr><tr><th>192</th><td>0.118</td><td>0.212</td><td>0.136</td><td>0.249</td><td>0.117</td><td>0.218</td><td>0.123</td><td>0.225</td><td>0.115</td><td>0.218</td><td>0.162</td><td>0.264</td><td>0.133</td><td>0.251</td><td>0.229</td><td>0.317</td><td>0.214</td><td>0.310</td></tr><tr><th>336</th><td>0.130</td><td>0.221</td><td>0.150</td><td>0.262</td><td>0.129</td><td>0.228</td><td>0.135</td><td>0.236</td><td>0.127</td><td>0.228</td><td>0.187</td><td>0.285</td><td>0.142</td><td>0.260</td><td>0.252</td><td>0.334</td><td>0.236</td><td>0.328</td></tr><tr><th>720</th><td>0.151</td><td>0.244</td><td>0.180</td><td>0.295</td><td>0.155</td><td>0.255</td><td>0.168</td><td>0.269</td><td>0.158</td><td>0.258</td><td>0.254</td><td>0.342</td><td>0.163</td><td>0.289</td><td>0.354</td><td>0.404</td><td>0.330</td><td>0.398</td></tr><tr><th>Ave</th><td>0.126</td><td>0.220</td><td>0.147</td><td>0.261</td><td>0.126</td><td>0.227</td><td>0.135</td><td>0.237</td><td>0.126</td><td>0.228</td><td>0.186</td><td>0.285</td><td>0.141</td><td>0.261</td><td>0.262</td><td>0.340</td><td>0.244</td><td>0.333</td></tr><tr><th rowspan="5">PEMS08</th><th>96</th><td>0.140</td><td>0.192</td><td>0.185</td><td>0.242</td><td>0.163</td><td>0.207</td><td>0.190</td><td>0.225</td><td>0.174</td><td>0.218</td><td>0.239</td><td>0.277</td><td>0.261</td><td>0.286</td><td>0.362</td><td>0.338</td><td>0.325</td><td>0.335</td></tr><tr><th>192</th><td>0.223</td><td>0.207</td><td>0.252</td><td>0.261</td><td>0.245</td><td>0.224</td><td>0.266</td><td>0.240</td><td>0.250</td><td>0.233</td><td>0.315</td><td>0.302</td><td>0.321</td><td>0.305</td><td>0.422</td><td>0.357</td><td>0.381</td><td>0.358</td></tr><tr><th>336</th><td>0.282</td><td>0.220</td><td>0.296</td><td>0.282</td><td>0.283</td><td>0.237</td><td>0.315</td><td>0.255</td><td>0.303</td><td>0.246</td><td>0.363</td><td>0.326</td><td>0.359</td><td>0.323</td><td>0.460</td><td>0.376</td><td>0.418</td><td>0.378</td></tr><tr><th>720</th><td>0.300</td><td>0.246</td><td>0.346</td><td>0.332</td><td>0.299</td><td>0.267</td><td>0.349</td><td>0.291</td><td>0.333</td><td>0.280</td><td>0.403</td><td>0.377</td><td>0.406</td><td>0.381</td><td>0.526</td><td>0.441</td><td>0.506</td><td>0.449</td></tr><tr><th>Ave</th><td>0.236</td><td>0.216</td><td>0.270</td><td>0.279</td><td>0.247</td><td>0.234</td><td>0.280</td><td>0.253</td><td>0.265</td><td>0.244</td><td>0.330</td><td>0.321</td><td>0.337</td><td>0.324</td><td>0.442</td><td>0.378</td><td>0.408</td><td>0.380</td></tr><tr><th colspan="2"><math><semantics><msup><mn>1</mn> <mtext>st</mtext></msup> <annotation>1^{\text{st}}</annotation></semantics></math> count</th><td>16</td><td>20</td><td>1</td><td>2</td><td>12</td><td>7</td><td>0</td><td>0</td><td>3</td><td>0</td><td>11</td><td>6</td><td>4</td><td>1</td><td>2</td><td>10</td><td>0</td><td>0</td></tr></tbody></table>

- Full results of the multivariate forecasting, where the input length is set to 672, and the forecasting length is set to {96, 192, 336, 720}. We adopt the rolling forecasting approach [^58], where one model is used for four forecasting lengths. Results are cited from Timer-XL [^8] if available; otherwise reproduced.

### IV-C Model Analysis

#### IV-C1 Ablation Study

CITRAS introduces KV Shift and Attention Score Smoothing (ASS) on top of a decoder-only Transformer. To validate the effectiveness of these design choices, we conduct detailed ablation studies, as summarized in Table VI. In the KV Shift ablation (“w/o KV Shift”), the target token in cross-variate attention is restricted to attend only to known covariates at the same patch step, preventing access to future known covariate information. As a result, the model fails to leverage future covariates in covariate-informed forecasting, leading to a significant performance drop across datasets. On top of this “w/o KV Shift” variant, we further evaluate a simpler alternative (“w/ Late fusion”), which fuses future token embeddings of known covariates with target token embeddings using an additional learnable fusion layer immediately before the final projection layer (see Appendix, Section VI-E for details). Although this late-fusion strategy allows future known covariates to be included, it leads to inferior performance on datasets such as EPF-NP and BS compared to the full model. This result indicates that naive fusion is insufficient, as it bypasses the dependency modeling performed within cross-variate attention. In contrast, KV Shift integrates future known covariates directly into the attention mechanism without introducing additional parameters, enabling more principled and consistent exploitation of future information. In the ASS ablation (“w/o ASS”), the model is restricted to capturing cross-variate dependencies only from a local perspective, resulting in inferior performance in both covariate-informed and multivariate settings. A more in-depth analysis of the sensitivity to the smoothing factor is provided in Section IV-C2. For the decoder-only ablation, we alternatively adopt an encoder-only architecture (i.e., removing causal masking in cross-time attention and applying a projection layer to the flattened features of all tokens for prediction). However, this design cannot effectively model causality in forecasting and lacks token-wise supervision, thus failing to fully leverage the inherent autoregressive capability of the Transformer in most cases. Collectively, these results highlight the strength of CITRAS, which seamlessly integrates heterogeneous covariates into a decoder-only Transformer and effectively models both cross-variate and cross-time dependencies.

TABLE VI: MSE results of the ablation study. All ablated variants are evaluated using a fixed random seed. Values in parentheses indicate that the corresponding mechanism is not applicable to datasets without known covariates, and thus the base CITRAS results are duplicated.

| Design | EPF-NP | EDF | BS | ETTh1 | ECL |
| --- | --- | --- | --- | --- | --- |
| CITRAS | 0.172 | 0.071 | 0.282 | 0.389 | 0.160 |
| (w/o) KV Shift | 0.226 | 0.086 | 0.289 | (0.389) | (0.160) |
| (w/ ) Late Fusion | 0.195 | 0.071 | 0.327 | (0.389) | (0.160) |
| (w/o) ASS | 0.175 | 0.074 | 0.299 | 0.401 | 0.165 |
| (w/o) Decoder-only | 0.188 | 0.068 | 0.366 | 0.428 | 0.162 |

#### IV-C2 Attention Score Smoothing

Attention Score Smoothing captures locally accurate patch-level cross-variate dependencies and refines them into global variate-level dependencies. Note that a large smoothing factor $\alpha$ emphasizes patch-level dependencies (with $\alpha=1.0$ corresponding to no smoothing), while a smaller $\alpha$ emphasizes variate-level dependencies through heavier smoothing. Figure 4 illustrates the performance at different levels of granularity in cross-variate attention. The performance of the variate-level approach is represented by TimeXer. We observe that the suitability of variate-level versus patch-level approaches is dependent on the dataset. However, strong Attention Score Smoothing (around $\alpha=0.2$) strikes a balance between the two, highlighting the versatile effectiveness of this mechanism in both covariate-informed and multivariate forecasting.

![Refer to caption](https://arxiv.org/html/2503.24007v4/x4.png)

Figure 4: The performance at different levels of granularity in cross-variate attention. The performance at “Variate” level is represented by TimeXer. Strong Attention Score Smoothing (around α = 0.2 \\alpha=0.2 ) outperforms both variate-level and patch-level approaches. The performance is evaluated with a fixed random seed.

#### IV-C3 Representation Analysis

To uncover the covariate usage mechanism of CITRAS further, we present the visualization of the cross-variate attention scores between the target variable and other variables in the EDF dataset in Figure 5. It is observed that patch-wise attention scores effectively capture the local negative impact of “Holiday” on the target variable (“Electricity Demand”). However, these scores are also susceptible to local disturbances, as demonstrated by the relatively fluctuating scores of “Humidity” compared to the “Temperature” that shows stable correlation with the target variable. By applying Attention Score Smoothing, the obtained scores effectively convey the strong dependency on “Holiday” from past steps and mitigate the noisy interaction caused by the fluctuating scores of “Humidity”. In this way, CITRAS enjoys the advantages of both patch-level and variate-level approaches while offering improved interpretability.

![Refer to caption](https://arxiv.org/html/2503.24007v4/x5.png)

Figure 5: Visualization of attention scores between the target “Electricity Demand” and other variables observed in the EDF dataset. By applying Attention Score Smoothing, the smoothed scores capture a strong dependency on “Holiday” while mitigating noisy interactions caused by fluctuation in “Humidity”.

#### IV-C4 Computational Efficiency

We analyze the computational complexity of CITRAS in comparison to other models. Consider the covariate-informed setting where the number of target variables is $1$, the number of covariates (including observed and known covariates) is $C$, and the number of patches of the target variable is $N$. CITRAS calculates cross-variate attention between the target and covariates at each patch step, resulting in a computational complexity of $O(CN)$. Additionally, it applies cross-time attention to each variate separately, incurring a complexity of $O(CN^{2})$. This is favorable compared to the Any-variate Attention adopted by Timer-XL, which incurs $O(C^{2}N^{2})$ to flexibly accommodate covariates. Furthermore, the $O(CN)$ complexity of cross-variate attention is advantageous when the number of covariates $C$ increases, as the multivariate models like iTransformer that do not discriminate between target and covariate incur the cost of $O(C^{2})$.

Besides the theoretical analysis, we demonstrate the empirical model efficiency on the BS dataset and PEMS08 dataset in Figure 6. We use the best model configuration for each model and evaluate using the same batch size for a fair comparison. The results in the BS dataset underscore the efficient model design of CITRAS. In the PEMS08 dataset, we observe the trade-off between training time and model performance in terms of MSE. Among them, CITRAS achieves the best performance without significant compromise in efficiency.

![Refer to caption](https://arxiv.org/html/2503.24007v4/x6.png)

Figure 6: Comparison of model efficiency on the BS and PEMS08 datasets. Note that the x-axis is in logarithmic scale for the PEMS08 result.

## V Conclusion

Considering the critical roles of covariates in practical time series forecasting, we proposed CITRAS, a decoder-only Transformer that flexibly leverages multiple target variables, observed covariates, and known covariates within a unified framework. To this end, we introduced two novel mechanisms in patch-wise cross-variate attention: Key-Value Shift, which incorporates future known covariate information into autoregressive forecasting, and Attention Score Smoothing, which refines locally captured patch-level dependencies into more stable variate-level relationships. Through extensive experiments, CITRAS demonstrated strong and consistent performance in both covariate-informed and multivariate forecasting settings, ranking first more frequently than other state-of-the-art models and highlighting its ability to effectively model cross-variate and cross-time dependencies.

Despite these promising results, several limitations remain. First, CITRAS operates in a supervised setting and requires dataset-specific training to achieve optimal performance, which may limit its applicability in low-data regimes or unseen domains. Second, in its current form, CITRAS does not explicitly distinguish between continuous and categorical covariates. While this design choice simplifies the architecture, it may restrict the expressive capacity when modeling discrete or event-driven covariates, such as calendar indicators.

Future work will address these limitations by exploring large-scale pre-training strategies to improve generalization across datasets and tasks, as well as by incorporating categorical-aware encoding schemes, such as target encoding, to better capture the semantics of categorical covariates within the decoder-only Transformer framework.

## VI Appendix

### VI-A Standard Deviation of Covariate-Informed Forecasting

Table VII reports the standard deviation of MSE and MAE across three random seeds for all models in the covariate-informed forecasting experiments, corresponding to the results presented in Table IV in Section IV-A. The results indicate that CITRAS exhibits stable performance across different random initializations.

TABLE VII: Standard deviation of the covariate-informed forecasting results across three random seeds.

<table><thead><tr><th colspan="2">Model</th><th colspan="2">CITRAS</th><th colspan="2">TFT</th><th colspan="2">TSMixer-Ext</th><th colspan="2">TimeXer</th><th colspan="2">TiDE</th><th colspan="2">Timer-XL</th><th colspan="2">iTrans.</th><th colspan="2">Leddam</th></tr><tr><th colspan="2">Metric</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th></tr></thead><tbody><tr><th rowspan="7">w/ known</th><th>EPF–NP</th><td>0.001</td><td>0.002</td><td>0.004</td><td>0.005</td><td>0.005</td><td>0.003</td><td>0.001</td><td>0.001</td><td>0.001</td><td>0.001</td><td>0.005</td><td>0.003</td><td>0.005</td><td>0.003</td><td>0.025</td><td>0.017</td></tr><tr><th>EPF-PJM</th><td>0.001</td><td>0.000</td><td>0.004</td><td>0.004</td><td>0.000</td><td>0.002</td><td>0.001</td><td>0.002</td><td>0.000</td><td>0.001</td><td>0.001</td><td>0.001</td><td>0.001</td><td>0.000</td><td>0.004</td><td>0.002</td></tr><tr><th>EPF–BE</th><td>0.009</td><td>0.005</td><td>0.031</td><td>0.004</td><td>0.048</td><td>0.010</td><td>0.003</td><td>0.001</td><td>0.003</td><td>0.001</td><td>0.035</td><td>0.014</td><td>0.009</td><td>0.001</td><td>0.003</td><td>0.002</td></tr><tr><th>EPF–FR</th><td>0.004</td><td>0.001</td><td>0.008</td><td>0.005</td><td>0.026</td><td>0.004</td><td>0.002</td><td>0.002</td><td>0.004</td><td>0.002</td><td>0.022</td><td>0.003</td><td>0.006</td><td>0.001</td><td>0.002</td><td>0.001</td></tr><tr><th>EPF–DE</th><td>0.002</td><td>0.001</td><td>0.007</td><td>0.004</td><td>0.002</td><td>0.001</td><td>0.014</td><td>0.002</td><td>0.008</td><td>0.005</td><td>0.001</td><td>0.002</td><td>0.008</td><td>0.005</td><td>0.013</td><td>0.004</td></tr><tr><th>EDF</th><td>0.001</td><td>0.001</td><td>0.002</td><td>0.003</td><td>0.004</td><td>0.007</td><td>0.001</td><td>0.001</td><td>0.000</td><td>0.001</td><td>0.002</td><td>0.002</td><td>0.001</td><td>0.001</td><td>0.001</td><td>0.002</td></tr><tr><th>BS</th><td>0.005</td><td>0.003</td><td>0.068</td><td>0.037</td><td>0.014</td><td>0.011</td><td>0.002</td><td>0.001</td><td>0.002</td><td>0.002</td><td>0.001</td><td>0.001</td><td>0.009</td><td>0.005</td><td>0.007</td><td>0.004</td></tr><tr><th colspan="2">Model</th><th colspan="2">CITRAS</th><th colspan="2">CARD</th><th colspan="2">ModernTCN</th><th colspan="2">TimesNet</th><th colspan="2">Cross.</th><th colspan="2">FITS</th><th colspan="2">DLinear</th><th colspan="2">PatchTST</th></tr><tr><th colspan="2">Metric</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th></tr><tr><th rowspan="7">w/o known</th><th>EPF–NP</th><td>0.001</td><td>0.001</td><td>0.003</td><td>0.003</td><td>0.002</td><td>0.001</td><td>0.013</td><td>0.003</td><td>0.002</td><td>0.004</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.004</td><td>0.003</td></tr><tr><th>EPF-PJM</th><td>0.002</td><td>0.001</td><td>0.004</td><td>0.004</td><td>0.001</td><td>0.000</td><td>0.004</td><td>0.003</td><td>0.004</td><td>0.002</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.004</td><td>0.004</td></tr><tr><th>EPF–BE</th><td>0.001</td><td>0.001</td><td>0.011</td><td>0.005</td><td>0.002</td><td>0.001</td><td>0.011</td><td>0.005</td><td>0.013</td><td>0.013</td><td>0.000</td><td>0.000</td><td>0.001</td><td>0.001</td><td>0.003</td><td>0.003</td></tr><tr><th>EPF–FR</th><td>0.001</td><td>0.001</td><td>0.001</td><td>0.002</td><td>0.002</td><td>0.001</td><td>0.012</td><td>0.006</td><td>0.026</td><td>0.003</td><td>0.001</td><td>0.001</td><td>0.000</td><td>0.000</td><td>0.006</td><td>0.003</td></tr><tr><th>EPF–DE</th><td>0.011</td><td>0.004</td><td>0.010</td><td>0.004</td><td>0.001</td><td>0.001</td><td>0.049</td><td>0.014</td><td>0.034</td><td>0.001</td><td>0.000</td><td>0.000</td><td>0.004</td><td>0.002</td><td>0.007</td><td>0.005</td></tr><tr><th>EDF</th><td>0.001</td><td>0.001</td><td>0.002</td><td>0.003</td><td>0.001</td><td>0.001</td><td>0.008</td><td>0.009</td><td>0.001</td><td>0.002</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.001</td><td>0.000</td></tr><tr><th>BS</th><td>0.000</td><td>0.001</td><td>0.004</td><td>0.002</td><td>0.002</td><td>0.001</td><td>0.014</td><td>0.004</td><td>0.010</td><td>0.006</td><td>0.000</td><td>0.001</td><td>0.000</td><td>0.000</td><td>0.003</td><td>0.003</td></tr></tbody></table>

- Standard deviation is computed across three random seeds. The experimental settings are identical to those in Table IV.

### VI-B Validation Curves of Covariate-Informed Forecasting

In our experiments, all models are trained with the same maximum training budget of 10 epochs, using early stopping based on validation performance. To justify this training budget, we report the validation MSE loss curves for representative covariate-informed datasets (EPF-NP, EDF, and BS) for CITRAS (Transformer-based), TFT (Transformer- and RNN-based), and TiDE (MLP-based) over three random seeds. The results in Figure 7 show that the validation loss typically stabilizes within the chosen budget (often earlier due to early stopping) regardless of the base architecture, supporting the choice of a maximum training budget of 10 epochs.

![Refer to caption](https://arxiv.org/html/2503.24007v4/figures/val_curves.png)

Figure 7: Validation loss curves on EPF-NP (top), EDF (middle), and BS (bottom) for CITRAS (left), TFT (middle), and TiDE (right) across three random seeds.

### VI-C Extra Results of Covariate-Informed Forecasting

To further evaluate CITRAS against recent models, we compare it with Sonnet [^42], which was presented at AAAI 2026, on covariate-informed forecasting tasks. Sonnet is designed for settings with a univariate target and observed covariates only, and therefore cannot be applied to datasets with multiple target variables (e.g., the BS dataset) or to settings that involve known covariates. For this reason, the comparison is limited to the EPF datasets and the EDF dataset under the “w/o known” setting. The results in Table VIII show that CITRAS achieves competitive performance compared to this recent state-of-the-art model. Notably, unlike Sonnet, CITRAS can additionally leverage future known covariates when they are available to further improve forecasting accuracy.

TABLE VIII: Comparison of CITRAS and Sonnet on covariate-informed forecasting without known covariates. Average MSE and MAE across three random seeds are reported.

<table><thead><tr><th colspan="2">Model</th><th colspan="2">CITRAS</th><th colspan="2">Sonnet</th></tr><tr><th colspan="2">Metric</th><th>MSE</th><th>MAE</th><th>MSE</th><th>MAE</th></tr></thead><tbody><tr><th rowspan="6">w/o known</th><th>EPF–NP</th><td>0.227</td><td>0.260</td><td>0.250</td><td>0.281</td></tr><tr><th>EPF–PJM</th><td>0.090</td><td>0.182</td><td>0.106</td><td>0.205</td></tr><tr><th>EPF–BE</th><td>0.405</td><td>0.276</td><td>0.394</td><td>0.263</td></tr><tr><th>EPF–FR</th><td>0.407</td><td>0.219</td><td>0.400</td><td>0.218</td></tr><tr><th>EPF–DE</th><td>0.420</td><td>0.404</td><td>0.449</td><td>0.424</td></tr><tr><th>EDF</th><td>0.087</td><td>0.206</td><td>0.086</td><td>0.206</td></tr></tbody></table>

- The input length and forecasting length are set to 168 and 24, respectively.

### VI-D Hyperparameter Sensitivity Analysis

In addition to the sensitivity analysis on the smoothing factor $\alpha$ of Attention Score Smoothing (Section IV-C2), we further investigate the sensitivity of CITRAS to the patch size, which is a key hyperparameter in patch-based Transformer architectures.

Specifically, we evaluate CITRAS on the EPF-NP, EDF, and BS datasets using patch sizes of 12, 24, and 42, while fixing the input length and forecasting horizon to 168 and 24, respectively, following the same configuration as in Table IV. In CITRAS, the output length is aligned with the patch size through the KV Shift mechanism. As a result, when the patch size is shorter than the forecasting horizon (e.g., patch size = 12), recursive forecasting is applied to reach the full horizon, whereas for larger patch sizes (e.g., patch size = 42), predictions up to 24 steps are used for evaluation.

As shown in Figure 8, CITRAS is generally not overly sensitive to the choice of patch size within a reasonable range. One exception is the EDF dataset under a small patch size of 12, which exhibits relatively worse performance compared to larger patch sizes. This behavior is expected for electricity demand forecasting, which exhibits strong daily seasonality and therefore benefits from a patch size that spans a full daily cycle. Overall, the adopted patch size of 24 yields stable and competitive performance across all datasets, supporting the reasonableness of this choice under a fixed hyperparameter setting.

![Refer to caption](https://arxiv.org/html/2503.24007v4/figures/patch_sensitivity_epf_np.png)

Figure 8: Patch size sensitivity analysis on the EPF-NP, EDF, and BS datasets. The dashed line indicates the default patch size of 24 adopted in all experiments.

### VI-E Model Extension Details

In this section, we describe the implementation details of the extensions for the baseline models (Timer-XL, iTransformer, and Leddam) introduced in Section IV-A2, as well as the late fusion variant of CITRAS analyzed in the ablation study in Section IV-C1.

##### Extension of Timer-XL.

Timer-XL [^8] originally supports only observed covariates alongside multivariate targets. To enable the use of known covariates, we modify the causal masking scheme in its TimeAttention module, while keeping all other components identical to the original implementation.

In TimeAttention, time series variables—including the target and covariates—are tokenized into patch-level tokens and flattened into a single sequence, allowing self-attention to jointly model dependencies across time steps and variables. In the original masking scheme, a target token at patch step $i$ is permitted to attend to tokens at the same or earlier patch steps ($\leq i$) from all variables, while all future tokens are strictly masked.

To incorporate known covariates, we relax this masking scheme by allowing target token embeddings at patch step $i$ to attend to the one-step-ahead future tokens ($i+1$) corresponding to known covariates. Future tokens of the target variable and observed covariates remain masked. This design allows Timer-XL to exploit future covariate information that is available at prediction time, without violating the temporal causality of the target series.

##### Extensions of iTransformer and Leddam.

iTransformer [^10] and Leddam [^32] are originally designed as multivariate forecasting models, where each variable is represented by embedding its entire historical sequence into a single token and modeling inter-variable dependencies through attention mechanisms. In their original formulations, the target variable with length $T$ is embedded into a $D$ -dimensional representation using a variate embedding layer with weights of size $T\times D$.

To incorporate known covariates whose available sequence length extends beyond the target horizon, we introduce an additional embedding layer specifically for known covariates. This embedding layer maps the extended covariate sequences of length $T+S$ into the same $D$ -dimensional space using weights of size $(T+S)\times D$, allowing the model to ingest future covariate information. Afterwards, these embeddings are treated in the same manner as the original variable embeddings and processed by the model following the original implementation.

##### Late Fusion Variant of CITRAS.

As discussed in the ablation study in Section IV-C1, we evaluate a simpler alternative to KV Shift, referred to as the “w/ Late fusion” variant, which incorporates future known covariates without modifying the cross-variate attention mechanism. In this variant, the embedding layers are identical to those used in the full CITRAS model, producing target, observed-covariate, and known-covariate token embeddings, respectively. Cross-time attention with causal masking and cross-variate attention are then applied in the same manner as in the full model; however, KV Shift is not employed, and the target token is therefore unable to attend to future known-covariate tokens within the attention layers.

Instead, future known covariates are incorporated through an additional learnable fusion layer placed immediately before the final projection layer. Specifically, the last target token $\mathbf{H}_{N_{tgt}}^{tgt,c}\in\mathbb{R}^{1\times D}$ and the one-step-ahead known-covariate tokens $\mathbf{H}_{N_{tgt}+1}^{knw,:}\in\mathbb{R}^{C_{knw}\times D}$ are used as inputs. The known-covariate tokens are first pooled along the covariate dimension to obtain a single $\mathbb{R}^{1\times D}$ representation, concatenated with $\mathbf{H}_{N_{tgt}}^{tgt,c}$, and projected to $\mathbb{R}^{1\times D}$ through a linear fusion layer of size $2D\!\rightarrow\!D$. The fused target representation is then passed to the original projection layer to generate the final prediction. All components of this variant are trained from scratch.

## References

[^1]: R. Weron, “Electricity price forecasting: A review of the state-of-the-art with a look into the future,” *International journal of forecasting*, 2014.

[^2]: J.-H. Böse, V. Flunkert, J. Gasthaus, T. Januschowski, D. Lange, D. Salinas, S. Schelter, M. Seeger, and Y. Wang, “Probabilistic demand forecasting at scale,” *Proc. VLDB Endow.*, vol. 10, no. 12, p. 1694–1705, Aug. 2017. \[Online\]. Available: [https://doi.org/10.14778/3137765.3137775](https://doi.org/10.14778/3137765.3137775)

[^3]: C. Capistrán, C. Constandse, and M. Ramos-Francia, “Multi-horizon inflation forecasts using disaggregated data,” *Economic Modelling*, vol. 27, no. 3, pp. 666–677, 2010. \[Online\]. Available: [https://www.sciencedirect.com/science/article/pii/S0264999310000076](https://www.sciencedirect.com/science/article/pii/S0264999310000076)

[^4]: E. Aguilar Madrid and N. Antonio, “Short-term electricity load forecasting with machine learning,” *Information*, vol. 12, no. 2, 2021. \[Online\]. Available: [https://www.mdpi.com/2078-2489/12/2/50](https://www.mdpi.com/2078-2489/12/2/50)

[^5]: A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in *NeurIPS*, 2017, pp. 6000–6010.

[^6]: Y. Zhang and J. Yan, “Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting,” in *ICLR*, 2022.

[^7]: X. Wang, T. Zhou, Q. Wen, J. Gao, B. Ding, and R. Jin, “CARD: Channel aligned robust blend transformer for time series forecasting,” in *ICLR*, 2024. \[Online\]. Available: [https://openreview.net/forum?id=MJksrOhurE](https://openreview.net/forum?id=MJksrOhurE)

[^8]: Y. Liu, G. Qin, X. Huang, J. Wang, and M. Long, “Timer-XL: Long-context transformers for unified time series forecasting,” in *ICLR*, 2025. \[Online\]. Available: [https://openreview.net/forum?id=KMCJXjlDDr](https://openreview.net/forum?id=KMCJXjlDDr)

[^9]: K. Tayal, A. Renganathan, X. Jia, V. Kumar, and D. Lu, “ ExoTST: Exogenous-Aware Temporal Sequence Transformer for Time Series Prediction,” in *2024 IEEE International Conference on Data Mining (ICDM)*. Los Alamitos, CA, USA: IEEE Computer Society, Dec. 2024, pp. 857–862. \[Online\]. Available: [https://doi.ieeecomputersociety.org/10.1109/ICDM59182.2024.00105](https://doi.ieeecomputersociety.org/10.1109/ICDM59182.2024.00105)

[^10]: Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, “iTransformer: Inverted transformers are effective for time series forecasting,” in *ICLR*, 2024.

[^11]: J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of deep bidirectional transformers for language understanding,” *arXiv preprint arXiv:1810.04805*, 2018.

[^12]: A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in *ICLR*, 2021. \[Online\]. Available: [https://openreview.net/forum?id=YicbFdNTTy](https://openreview.net/forum?id=YicbFdNTTy)

[^13]: T. Zhou, Z. Ma, Q. Wen, X. Wang, L. Sun, and R. Jin, “FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting,” in *ICML*, 2022.

[^14]: S. Liu, H. Yu, C. Liao, J. Li, W. Lin, A. X. Liu, and S. Dustdar, “Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting,” in *ICLR*, 2022.

[^15]: J. Dong, H. Wu, H. Zhang, L. Zhang, J. Wang, and M. Long, “SimMTM: A simple pre-training framework for masked time-series modeling,” in *NeurIPS*, 2023.

[^16]: H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, “Informer: Beyond efficient transformer for long sequence time-series forecasting,” in *AAAI*, 2021.

[^17]: H. Wu, J. Xu, J. Wang, and M. Long, “Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting,” in *NeurIPS*, 2021, pp. 22 419–22 430.

[^18]: Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in *ICLR*, 2023.

[^19]: Y. Zhang, M. Liu, S. Zhou, and J. Yan, “UP2ME: Univariate pre-training to multivariate fine-tuning as a general-purpose framework for multivariate time series analysis,” in *ICML*, 2024. \[Online\]. Available: [https://openreview.net/forum?id=aR3uxWlZhX](https://openreview.net/forum?id=aR3uxWlZhX)

[^20]: J. Zhang, S. Zheng, X. Wen, X. Zhou, J. Bian, and J. Li, “ElasTST: Towards robust varied-horizon forecasting with elastic time-series transformer,” in *NeurIPS*, 2024. \[Online\]. Available: [https://openreview.net/forum?id=ucXUtMPWhv](https://openreview.net/forum?id=ucXUtMPWhv)

[^21]: X. Zhou, W. Wang, W. Buntine, S. Qu, A. Sriramulu, W. Tan, and C. Bergmeir, “Scalable transformer for high dimensional multivariate time series forecasting,” in *ACM International Conference on Information and Knowledge Management*, 2024. \[Online\]. Available: [https://doi.org/10.1145/3627673.3679757](https://doi.org/10.1145/3627673.3679757)

[^22]: A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are transformers effective for time series forecasting?” in *AAAI*, 2023, pp. 11 121–11 128.

[^23]: Z. Xu, A. Zeng, and Q. Xu, “FITS: Modeling time series with $10k$ parameters,” in *ICLR*, 2024. \[Online\]. Available: [https://openreview.net/forum?id=bWcnvZ3qMb](https://openreview.net/forum?id=bWcnvZ3qMb)

[^24]: A. F. Ansari, L. Stella, C. Turkmen, X. Zhang, P. Mercado, H. Shen, O. Shchur, S. S. Rangapuram, S. P. Arango, S. Kapoor, J. Zschiegner, D. C. Maddix, H. Wang, M. W. Mahoney, K. Torkkola, A. G. Wilson, M. Bohlke-Schneider, and Y. Wang, “Chronos: Learning the language of time series,” *Transactions on Machine Learning Research*, 2024. \[Online\]. Available: [https://openreview.net/forum?id=gerNCVqqtR](https://openreview.net/forum?id=gerNCVqqtR)

[^25]: A. Das, W. Kong, R. Sen, and Y. Zhou, “A decoder-only foundation model for time-series forecasting,” in *ICML*, 2024, pp. 10 148–10 167.

[^26]: M. Goswami, K. Szafer, A. Choudhry, Y. Cai, S. Li, and A. Dubrawski, “Moment: a family of open time-series foundation models,” in *ICML*, 2024, pp. 16 115–16 152.

[^27]: N. Gruver, M. Finzi, S. Qiu, and A. G. Wilson, “Large language models are zero-shot time series forecasters,” in *NeurIPS*, 2023.

[^28]: T. Zhou, P. Niu, L. Sun, R. Jin *et al.*, “One fits all: Power general time series analysis by pretrained lm,” in *NeurIPS*, 2023.

[^29]: M. Jin, S. Wang, L. Ma, Z. Chu, J. Y. Zhang, X. Shi, P.-Y. Chen, Y. Liang, Y.-F. Li, S. Pan, and Q. Wen, “Time-LLM: Time series forecasting by reprogramming large language models,” in *ICLR*, 2024. \[Online\]. Available: [https://openreview.net/forum?id=Unb5CVPtae](https://openreview.net/forum?id=Unb5CVPtae)

[^30]: I. Abdelmalak, K. Madhusudhanan, J. Choi, M. Stubbemann, and L. Schmidt-Thieme, “Channel dependence, limited lookback windows, and the simplicity of datasets: How biased is time series forecasting?” *arXiv preprint arXiv:2502.09683*, 2025.

[^31]: Y. Wang, H. Wu, J. Dong, G. Qin, H. Zhang, Y. Liu, Y. Qiu, J. Wang, and M. Long, “TimeXer: Empowering transformers for time series forecasting with exogenous variables,” in *NeurIPS*, 2024. \[Online\]. Available: [https://openreview.net/forum?id=INAeUQ04lT](https://openreview.net/forum?id=INAeUQ04lT)

[^32]: G. Yu, J. Zou, X. Hu, A. I. Aviles-Rivero, J. Qin, and S. Wang, “Revitalizing multivariate time series forecasting: Learnable decomposition with inter-series dependencies and intra-series variations modeling,” in *ICML*, 2024, pp. 57 818–57 841. \[Online\]. Available: [https://openreview.net/forum?id=87CYNyCGOo](https://openreview.net/forum?id=87CYNyCGOo)

[^33]: S. Lin, H. Chen, H. Wu, C. Qiu, and W. Lin, “Temporal query network for efficient multivariate time series forecasting,” in *ICML*, ser. Proceedings of Machine Learning Research, A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaff, and J. Zhu, Eds., vol. 267. PMLR, 13–19 Jul 2025, pp. 37 797–37 814. \[Online\]. Available: [https://proceedings.mlr.press/v267/lin25e.html](https://proceedings.mlr.press/v267/lin25e.html)

[^34]: G. Woo, C. Liu, A. Kumar, C. Xiong, S. Savarese, and D. Sahoo, “Unified training of universal time series forecasting transformers,” in *ICML*, 2024, pp. 53 140–53 164.

[^35]: X. Liu, J. Liu, G. Woo, T. Aksu, Y. Liang, R. Zimmermann, C. Liu, S. Savarese, C. Xiong, and D. Sahoo, “Moirai-MoE: Empowering time series foundation models with sparse mixture of experts,” *arXiv preprint arXiv:2410.10469*, 2024.

[^36]: B. M. Williams, “Multivariate vehicular traffic flow prediction: evaluation of arimax modeling,” *Transportation Research Record*, 2001.

[^37]: W. B. Nicholson, D. S. Matteson, and J. Bien, “VARX-L: Structured regularization for large vector autoregressions with exogenous variables,” *International Journal of Forecasting*, 2017.

[^38]: K. G. Olivares, C. Challu, G. Marcjasz, R. Weron, and A. Dubrawski, “Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx,” *International Journal of Forecasting*, 2023.

[^39]: B. N. Oreshkin, D. Carpov, N. Chapados, and Y. Bengio, “N-beats: Neural basis expansion analysis for interpretable time series forecasting,” *CoRR*, vol. abs/1905.10437, 2019. \[Online\]. Available: [http://arxiv.org/abs/1905.10437](http://arxiv.org/abs/1905.10437)

[^40]: A. Das, W. Kong, A. Leach, S. K. Mathur, R. Sen, and R. Yu, “Long-term forecasting with TiDE: Time-series dense encoder,” *Transactions on Machine Learning Research*, 2023. \[Online\]. Available: [https://openreview.net/forum?id=pCbC3aQB5W](https://openreview.net/forum?id=pCbC3aQB5W)

[^41]: Y. Shu and V. Lampos, “DeformTime: Capturing Variable Dependencies with Deformable Attention for Time Series Forecasting,” *Transactions on Machine Learning Research*, 2025. \[Online\]. Available: [https://openreview.net/forum?id=M62P7iOT7d](https://openreview.net/forum?id=M62P7iOT7d)

[^42]: ——, “Sonnet: Spectral operator neural network for multivariable time series forecasting,” in *AAAI*, 2026, pp. 25 419–25 427.

[^43]: B. Lim, S. Ö. Arık, N. Loeff, and T. Pfister, “Temporal fusion transformers for interpretable multi-horizon time series forecasting,” *International Journal of Forecasting*, 2021.

[^44]: S.-A. Chen, C.-L. Li, S. O. Arik, N. C. Yoder, and T. Pfister, “TSMixer: An all-MLP architecture for time series forecasting,” *Transactions on Machine Learning Research*, 2023. \[Online\]. Available: [https://openreview.net/forum?id=wbpxTuXgm0](https://openreview.net/forum?id=wbpxTuXgm0)

[^45]: J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu, “RoFormer: Enhanced transformer with rotary position embedding,” *Neurocomput.*, vol. 568, no. C, Feb. 2024. \[Online\]. Available: [https://doi.org/10.1016/j.neucom.2023.127063](https://doi.org/10.1016/j.neucom.2023.127063)

[^46]: J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer normalization,” *arXiv preprint arXiv:1607.06450*, 2016.

[^47]: J. Kang, Y. Shin, and J.-G. Lee, “VarDrop: Enhancing training efficiency by reducing variate redundancy in periodic time series forecasting,” in *AAAI*, 2025.

[^48]: G. Woo, C. Liu, D. Sahoo, A. Kumar, and S. Hoi, “ETSformer: Exponential smoothing transformers for time-series forecasting,” *arXiv preprint arXiv:2202.01381*, 2022.

[^49]: J. Lago, G. Marcjasz, B. De Schutter, and R. Weron, “Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark,” *Applied Energy*, vol. 293, p. 116983, 2021.

[^50]: H. Fanaee-T, “Bike Sharing,” UCI Machine Learning Repository, 2013, DOI: https://doi.org/10.24432/C5W894.

[^51]: L. Donghao and W. Xue, “ModernTCN: A modern pure convolution structure for general time series analysis,” in *ICLR*, 2024. \[Online\]. Available: [https://openreview.net/forum?id=vpJMJerXHU](https://openreview.net/forum?id=vpJMJerXHU)

[^52]: H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “TimesNet: Temporal 2d-variation modeling for general time series analysis,” in *ICLR*, 2023.

[^53]: A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga *et al.*, “Pytorch: An imperative style, high-performance deep learning library,” in *NeurIPS*, 2019.

[^54]: Y. Liu, H. Wu, J. Wang, and M. Long, “Non-stationary transformers: Exploring the stationarity in time series forecasting,” in *NeurIPS*, 2022.

[^55]: D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” *arXiv preprint arXiv:1412.6980*, 2014.

[^56]: X. Qiu, J. Hu, L. Zhou, X. Wu, J. Du, B. Zhang, C. Guo, A. Zhou, C. S. Jensen, Z. Sheng, and B. Yang, “TFB: Towards comprehensive and fair benchmarking of time series forecasting methods,” *Proc. VLDB Endow.*, vol. 17, no. 9, p. 2363–2377, May 2024. \[Online\]. Available: [https://doi.org/10.14778/3665844.3665863](https://doi.org/10.14778/3665844.3665863)

[^57]: M. Liu, A. Zeng, M. Chen, Z. Xu, Q. Lai, L. Ma, and Q. Xu, “SCINet: Time series modeling and forecasting with sample convolution and interaction,” in *NeurIPS*, 2022.

[^58]: Y. Liu, G. Qin, X. Huang, J. Wang, and M. Long, “AutoTimes: Autoregressive time series forecasters via large language models,” in *NeurIPS*, 2024. \[Online\]. Available: [https://openreview.net/forum?id=FOvZztnp1H](https://openreview.net/forum?id=FOvZztnp1H)