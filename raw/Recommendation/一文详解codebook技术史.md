---
title: "一文详解 codebook 技术史（从 VAE 到 VQ/RQ-VAE 到 FSQ）"
source: "https://zhuanlan.zhihu.com/p/2433292582"
author:
  - "[[翟泽鹏不知名 LLMer]]"
published:
created: 2026-06-16
description: "VAE：VAE (variational autoencoder，变分自编码器) 是一种强大的生成模型， Encoder 把数据编码到隐空间 z = Ecd(x) ，其学习条件概率 p_{\phi}(z|x) ， Decoder 把数据从隐空间中重建回来 x = Dcd(z) ，其学习另…"
tags:
  - "clippings"
---
目录

收起

VAE：

K-L散度：

VAE 理论框架（联合概率建模角度）：

ELBO：

VQ-VAE：

背景：

方法：

训练：

应用：

Issue：

VQ-VAE-2：

RQ-VAE：

背景：

方法：

训练：

Trick：

FSQ：

方法：

实验：

引用：

## VAE：

VAE (variational autoencoder，变分自编码器) 是一种强大的生成模型， Encoder 把数据编码到隐空间 $z = E c d \left(\right. x \left.\right)$ ，其学习条件概率 $p_{\phi} \left(\right. z \left|\right. x \left.\right)$ ， Decoder 把数据从隐空间中重建回来 $x = D c d \left(\right. z \left.\right)$ ，其学习另一个条件概率 $q_{\theta} \left(\right. x \left|\right. z \left.\right)$ 。VAE 额外有一个限制条件是让 $z$ 满足 Gaussian 分布。这样做的好处就是训练结束后可以扔掉 Encoder，直接从这个先验分布上随便采样 $z$ ，然后通过 Decoder 就能生成一个 $x$ 。

![](https://pic2.zhimg.com/v2-2786fc365ec122101906cbf93e6dee65_1440w.jpg)

VAE 最主要的是这个 ELBO ： $E L B O \left(\right. \theta , \phi \left.\right) = E_{z sim p_{\phi} \left(\right. z \left|\right. x \left.\right)} \left[\right. log ⁡ q_{\theta} \left(\right. x \left|\right. z \left.\right) \left]\right. - K L \left(\right. p_{\phi} \left(\right. z \left|\right. x \left.\right) \left|\right. \left|\right. q \left(\right. z \left.\right) \left.\right)$

ELBO，即evidence low bound，evidence指的就是 $x$ ，而 ELBO 表示 evidence 的最小期望。我们要让这个 lower bound 尽可能变大，得到的模型就会更可能产生我们期望看到的 $x$ 。

为解释 ELBO 是怎么来的，我们一步一步来看。

### K-L散度：

我们首先讲解 KL 散度，为衡量模型生成的分布与原始分布的相似度，常用的便是 K–L（ Kullback–Leibler ）散度。定义如下，对于两个具有概率密度函数 $p_{1} \left(\right. x \left.\right)$ 和 $p_{2} \left(\right. x \left.\right)$ 的分布：

$K L \left(\right. p_{1} \left(\right. x \left.\right) , p_{2} \left(\right. x \left.\right) \left.\right) = \int p_{1} \left(\right. x \left.\right) log ⁡ \frac{p_{1} \left(\right. x \left.\right)}{p_{2} \left(\right. x \left.\right)} d x$

K–L 散度具有两个重要性质：

1\. **不对称性** ：显然，K–L 散度对于 $p_{1} \left(\right. x \left.\right)$ 和 $p_{2} \left(\right. x \left.\right)$ 来说是不对称的。

2\. **[Gibbs 不等式](https://zhida.zhihu.com/search?content_id=249509248&content_type=Article&match_order=1&q=Gibbs+%E4%B8%8D%E7%AD%89%E5%BC%8F&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODA3MzQyODUsInEiOiJHaWJicyDkuI3nrYnlvI8iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNDk1MDkyNDgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.EEBo8M65HYbTJCndAAqGOGJOTK2QNDwkIMsZ4xkNKpA&zhida_source=entity)** ：它总是【非负】的，并且当且仅当 $p_{1} \left(\right. x \left.\right)$ 和 $p_{2} \left(\right. x \left.\right)$ 在每一处都相同时才为 0。

为了理解这一点，我们可以将 KL 散度分解为两部分：

$K L \left(\right. p_{1} \left(\right. x \left.\right) , p_{2} \left(\right. x \left.\right) \left.\right) & = \int p_{1} \left(\right. x \left.\right) log ⁡ p_{1} \left(\right. x \left.\right) d x - \int p_{1} \left(\right. x \left.\right) log ⁡ p_{2} \left(\right. x \left.\right) d x \\ = - \int p_{1} \left(\right. x \left.\right) log ⁡ p_{2} \left(\right. x \left.\right) d x - \left(\right. - \int p_{1} \left(\right. x \left.\right) log ⁡ p_{1} \left(\right. x \left.\right) d x \left.\right)$

第二项带有负号，其对应的是 $p_{1}$ 的信息熵；第一项也带有负号，代表 $p_{1}$ 和 $p_{2}$ 之间的交叉熵。第一项始终不大于每个给定符号下的第二项，这便是 **Gibbs 不等式** ；而 Gibbs 不等式的证明可以使用 **[Jensen 不等式](https://zhida.zhihu.com/search?content_id=249509248&content_type=Article&match_order=1&q=Jensen+%E4%B8%8D%E7%AD%89%E5%BC%8F&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODA3MzQyODUsInEiOiJKZW5zZW4g5LiN562J5byPIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjQ5NTA5MjQ4LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.xV4rCCw7mz1WdZY0JMJD07A5pM2niWc_y_AHp6sqnZk&zhida_source=entity)** ：

若 $\varphi$ 是凸函数，则有： $\varphi \left(\right. E \left(\right. X \left.\right) \left.\right) \leq E \left(\right. \varphi \left(\right. X \left.\right) \left.\right)$

设 $\varphi \left(\right. x \left.\right) = log ⁡ x$ 由于 $\varphi^{''} \left(\right. x \left.\right) = - 1 / x^{2} \leq 0$ 所以其为凸函数，以及 $E \left(\right. X \left.\right) = \mathbb{E}_{x sim p_{1} \left(\right. x \left.\right)} \left[\right. \frac{p_{2} \left(\right. x \left.\right)}{p_{1} \left(\right. x \left.\right)} \left]\right.$

那么：

$E \left(\right. \varphi \left(\right. X \left.\right) \left.\right) = \mathbb{E}_{x sim p_{1} \left(\right. x \left.\right)} \left[\right. l o g \frac{p_{2} \left(\right. x \left.\right)}{p_{1} \left(\right. x \left.\right)} \left]\right. = \int p_{1} \left(\right. x \left.\right) log ⁡ \frac{p_{2} \left(\right. x \left.\right)}{p_{1} \left(\right. x \left.\right)} d x = - K L \left(\right. p_{1} \left(\right. x \left.\right) , p_{2} \left(\right. x \left.\right) \left.\right)$

$\varphi \left(\right. E \left(\right. X \left.\right) \left.\right) = l o g \left(\right. \mathbb{E}_{x sim p_{1} \left(\right. x \left.\right)} \left[\right. \frac{p_{1} \left(\right. x \left.\right)}{p_{2} \left(\right. x \left.\right)} \left]\right. \left.\right) = l o g \int p_{1} \left(\right. x \left.\right) \frac{p_{2} \left(\right. x \left.\right)}{p_{1} \left(\right. x \left.\right)} d x = l o g \int p_{2} \left(\right. x \left.\right) d x = l o g 1 = 0$

由 $\varphi \left(\right. E \left(\right. X \left.\right) \left.\right) \leq E \left(\right. \varphi \left(\right. X \left.\right) \left.\right)$ 可得 $K L \left(\right. p_{1} \left(\right. x \left.\right) , p_{2} \left(\right. x \left.\right) \left.\right) \geq 0$

### VAE 理论框架（联合概率建模角度）：

VAE 框架可以从多个角度建立，例如 [概率分布视角](https://link.zhihu.com/?target=https%3A//kexue.fm/archives/5343) 、 [贝叶斯视角](https://link.zhihu.com/?target=https%3A//amaires.github.io/VAE/) 以及 [联合概率视角](https://link.zhihu.com/?target=https%3A//kexue.fm/archives/5343) ，这里我选用联合概率这一简单的方法来阐述：

假设原始数据样本为 $x$ ，分布为 $p \left(\right. x \left.\right)$ ，我们希望借助隐变量 $z$ （标准正态分布）来建模 $p \left(\right. x \left.\right)$ ，因此我们设立 $q \left(\right. x \left.\right)$ 来逼近 $p \left(\right. x \left.\right)$:$q \left(\right. x \left.\right) = \int q \left(\right. x \mid z \left.\right) q \left(\right. z \left.\right) d z$

$q \left(\right. z \left.\right)$ 是标准正态分布， $q \left(\right. x \mid z \left.\right)$ 是我们的生成式模型；此外还需明确的是 $p \left(\right. x \left.\right)$ 是 $x$ 的原始分布， $q \left(\right. z \mid x \left.\right)$ 是encoder生成的 $z$ ，训练时要让其逼近正态分布。

我们直接采用联合建模的角度，原来我们的目的是让 $q \left(\right. x \left.\right)$ 来逼近 $p \left(\right. x \left.\right)$ ，我们转变下思路变为让 $q \left(\right. x , z \left.\right)$ 与 $p \left(\right. x , z \left.\right)$ 越相近越好，注意除了， $p \left(\right. x , z \left.\right)$ 中也有参数：

$K L \left(\right. p \left(\right. x , z \left.\right) \parallel q \left(\right. x , z \left.\right) \left.\right) = \iint p \left(\right. x , z \left.\right) log ⁡ \frac{p \left(\right. x , z \left.\right)}{q \left(\right. x , z \left.\right)} d z d x$ KL 散度便是我们的终极目标，我们将从这个 KL 散度推导出最终的 ELBO：

$K L \left(\right. p \left(\right. x , z \left.\right) \parallel q \left(\right. x , z \left.\right) \left.\right) & = \int p \left(\right. x \left.\right) \left[\right. \int p \left(\right. z \mid x \left.\right) log ⁡ \frac{p \left(\right. x \left.\right) p \left(\right. z \mid x \left.\right)}{q \left(\right. x , z \left.\right)} d z \left]\right. d x \\ = \mathbb{E}_{x sim p \left(\right. x \left.\right)} \left[\right. \int p \left(\right. z \mid x \left.\right) log ⁡ \frac{p \left(\right. x \left.\right) p \left(\right. z \mid x \left.\right)}{q \left(\right. x , z \left.\right)} d z \left]\right. \\ = \mathbb{E}_{x sim p \left(\right. x \left.\right)} \left[\right. \int p \left(\right. z \mid x \left.\right) log ⁡ p \left(\right. x \left.\right) d z \left]\right. + \mathbb{E}_{x sim p \left(\right. x \left.\right)} \left[\right. \int p \left(\right. z \mid x \left.\right) log ⁡ \frac{p \left(\right. z \mid x \left.\right)}{q \left(\right. x \mid z \left.\right) q \left(\right. z \left.\right)} d z \left]\right.$ 这里被我们拆开为两项，第一项： $\mathbb{E}_{x sim p \left(\right. x \left.\right)} \left[\right. \int p \left(\right. z \mid x \left.\right) log ⁡ p \left(\right. x \left.\right) d z \left]\right. & = \mathbb{E}_{x sim p \left(\right. x \left.\right)} \left[\right. log ⁡ p \left(\right. x \left.\right) \int p \left(\right. z \mid x \left.\right) d z \left]\right. \\ = \mathbb{E}_{x sim p \left(\right. x \left.\right)} \left[\right. log ⁡ p \left(\right. x \left.\right) \left]\right. \\ = 常 数$ 无论 $p \left(\right. x \left.\right)$ 是什么，它一定是确定的，故第一项是常数

第二项： $\mathcal{L} & = \mathbb{E}_{x sim p \left(\right. x \left.\right)} \left[\right. \int p \left(\right. z \mid x \left.\right) log ⁡ \frac{p \left(\right. z \mid x \left.\right)}{q \left(\right. x \mid z \left.\right) q \left(\right. z \left.\right)} d z \left]\right. \\ = \mathbb{E}_{x sim p \left(\right. x \left.\right)} \left[\right. - \int p \left(\right. z \mid x \left.\right) log ⁡ q \left(\right. x \mid z \left.\right) d z + \int p \left(\right. z \mid x \left.\right) log ⁡ \frac{p \left(\right. z \mid x \left.\right)}{q \left(\right. z \left.\right)} d z \left]\right. \\ = \mathbb{E}_{x sim p \left(\right. x \left.\right)} \left[\right. \mathbb{E}_{z sim p \left(\right. z \mid x \left.\right)} \left[\right. - log ⁡ q \left(\right. x \mid z \left.\right) \left]\right. + K L \left(\right. p \left(\right. z \mid x \left.\right) \parallel q \left(\right. z \left.\right) \left.\right) \left]\right. \\ = \mathbb{E}_{z sim p \left(\right. z \mid x \left.\right)} \left[\right. - E L B O \left]\right.$ 因此我们很快便得到了最终的 ELBO，注意多了个负号。

### ELBO：

ELBO 有两项，分别为： $E_{z sim p_{\phi} \left(\right. z \left|\right. x \left.\right)} \left[\right. log ⁡ q_{\theta} \left(\right. x \left|\right. z \left.\right) \left]\right.$ 以及 $- K L \left(\right. p_{\phi} \left(\right. z \left|\right. x \left.\right) \left|\right. \left|\right. q \left(\right. z \left.\right) \left.\right)$ ，这两部分可以理解为【 **重构误差项】** 以及【 **KL散度项】：**

重构误差项： $E_{z sim p_{\phi} \left(\right. z \left|\right. x \left.\right)} \left[\right. log ⁡ q_{\theta} \left(\right. x \left|\right. z \left.\right) \left]\right.$ 这部分度量了模型生成数据的质量，即解码器 $D_{\theta}$ 使用从编码器 $E_{\phi}$ 采样的 $z$ 来重构输入 $x$ 的准确性，这是负对数似然，表明给定潜在变量 $z$ 后，重构原来的 $x$ 的概率有多大。目标是最大化这部分期望值，即希望模型能生成与输入 x 尽可能接近的数据。

KL散度项： $- K L \left(\right. p_{\phi} \left(\right. z \left|\right. x \left.\right) \left|\right. \left|\right. q \left(\right. z \left.\right) \left.\right)$ 是后验分布 $p_{\phi} \left(\right. z \mid x \left.\right)$ 和先验分布 $p \left(\right. z \left.\right)$ 之间的负K–L 散度，以此衡量编码器的输出分布与标准正态分布的差异。目标是最小化KL散度，确保潜在变量 $z$ 尽可能接近正态分布。

至此我们推导出了VAE的损失函数，了解了ELBO的原理。

## VQ-VAE：

paper： [Neural Discrete Representation Learning](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1711.00937)

![](https://picx.zhimg.com/v2-47060ecb6dffbffb199be1b1356efec1_1440w.jpg)

paper：Neural Discrete Representation Learning

### 背景：

VAE中的隐变量 $z$ 的每一维都是一个连续值， 而VQ-VAE 中 $z$ 的每一维都是离散的整数，这些整数便可 index 到已训练好的 codebook（码本，本质上就是一批 embedding）。这样做符合自然界模态的特点，例如语言本质上就是由很多字符组成，每个字符都可以是用数字索引到字符库里的某个字符，NLP中可以理解为token\_id索引到vocab里的某个token，所以VQ-VAE可以理解为 【 **图像tokenization** 】 的过程，事实上这种思想可以借鉴引用到很多领域，例如广告推荐里将广告用一串索引表示。

文章还指出，VAE 存在 **后验坍塌（Posterior Collapse）** 的问题，这一般是由散度消失（KL-Vanishinig）导致的，因此该问题也称为KL-vanishing。简单来说就是解码器太强，模型的 **潜在空间（latent space）无效化** ，即编码器 $q_{\phi} \left(\right. z \mid x \left.\right)$ 退化为与先验 $p \left(\right. z \left.\right) = N \left(\right. 0 , I \left.\right)$ 相同的分布，ELBO里的KL散度项为0，而忽略了输入数据的信息。

### 方法：

将隐变量 $z$ 离散化的关键操作是VQ, 即 vector quatization。

![](https://pic2.zhimg.com/v2-6f582127c21767504633a85e95614b9f_1440w.jpg)

图1. VQ-VAE 流程图

1. 图像 $x$ 输入至 encoder 中得到 $z_{e}$: $z_{e} = e n c o d e r \left(\right. x \left.\right)$
2. codebook 是一个K\*D 的 table（紫色方块）： $E = \left[\right. e_{1} , e_{2} , \ldots , e_{K} \left]\right.$
3. 将中每一维都映射为 codebook 中K个embedding之一 $z_{q} \left(\right. x \left.\right) = e_{k} , w h e r e k = argmin_{j} ⁡ \left(\parallel z_{e} \left(\right. x \left.\right) - e_{j} \parallel\right)_{2}$
4. 全部替换后图中绿色的变为紫色的，然后进行重构

从到的变化可以理解为聚类，如图中右子图所示，由于变化后的embedding位于codebook内，当然就可以只用整数来表示。

### 训练：

**ELBO 损失项：**

$E L B O \left(\right. \theta , \phi \left.\right) = E_{z sim q_{\phi} \left(\right. z \left|\right. x \left.\right)} \left[\right. log ⁡ p_{\theta} \left(\right. x \left|\right. z \left.\right) \left]\right. - K L \left(\right. q_{\phi} \left(\right. z \left|\right. x \left.\right) \left|\right. \left|\right. p \left(\right. z \left.\right) \left.\right)$ 我们先看原有的 ELBO ，这里p和q互换以与图示对应，q代表encoder，p代表decoder；

这里后验分布 $q \left(\right. z \left|\right. x \left.\right)$ 里都是one-hot向量，如下所示：

$q \left(\right. z = k \mid x \left.\right) = \left{\right. 1 & \text{for } k = argmin_{j} ⁡ \left(\parallel z_{e} \left(\right. x \left.\right) - e_{j} \parallel\right)_{2} , \\ 0 & \text{otherwise }$ 而非之前VAE里的正态分布，由此 $q_{\phi} \left(\right. z \left|\right. x \left.\right)$ 预估的每一维都是codebook里每个embedding的概率；我们假设采样的先验分布 $z$ 是均匀分布，则每一维对于某个embedding选取概率有 $\frac{1}{K}$ ，则有：

$K L \left(\right. q_{\phi} \left(\right. z \left|\right. x \left.\right) \left|\right. \left|\right. p \left(\right. z \left.\right) \left.\right) = \int q_{\phi} \left(\right. z \left|\right. x \left.\right) l o g \frac{q_{\phi} \left(\right. z \left|\right. x \left.\right)}{p \left(\right. z \left.\right)} d z = 1 \cdot log ⁡ \left(\right. \frac{1}{\frac{1}{K}} \left.\right) + \left(\right. K - 1 \left.\right) \times 0 \cdot log ⁡ \left(\right. \frac{0}{\frac{1}{K}} \left.\right) = log ⁡ K$

第一项表示one-hot中为1对应的那一维对KL散度的贡献，第二项代表其他维的贡献。

因此 ELBO 中第二项可以忽略，只有重构损失项。

那我们再看第一项损失，可以简单写为： $\left(\parallel x - decoder ⁡ \left(\right. z_{q} \left.\right) \parallel\right)_{2}^{2}$

然而 $z_{q}$ 包含了argmin，这个操作是没有梯度的，无法更新 encoder；VQ-VAE 使用了一个很精巧也很直接的方法，称为 **Straight-Through Estimator** ，称为“ [直通估计](https://link.zhihu.com/?target=https%3A//papers.cool/arxiv/1308.3432) ”。其思想是在前向传播的时候可以任意变量（可以不可导），而反向传播的时候，直接 **跳过** 这个不可导的操作。对应图1中红色箭头，表明跳过 $\nabla_{z} L$ 的操作。

根据这个思想，我们设计的目标函数是 $\left(\parallel x - decoder ⁡ \left(\right. z_{e} + sg ⁡ \left[\right. z_{q} - z_{e} \left]\right. \left.\right) \parallel\right)_{2}^{2}$

sg 代表阻止梯度回传

**codebook 损失项：**

为使得 $z_{q}$ 与 $z_{e}$ 尽量接近，设置损失： $\left(\parallel z_{e} - z_{q} \parallel\right)_{2}^{2}$ ；

这里我们理解下： $z_{e}$ 是编码器得到的， $z_{q}$ 是离得最近的embedding，两者都有可训练的参数；因此在实际训练时，codebook相对自由宽松，没什么限制条件，而编码器生成的要保证重建效果，我们更希望主要靠近，并且因为 $\left(\parallel z_{e} - z_{q} \parallel\right)_{2}^{2}$ 的梯度等于以及梯度之和，故可拆解为：

$\left(\parallel s g \left[\right. z_{e} \left]\right. - z_{q} \parallel\right)_{2}^{2} + \left(\parallel z_{e} - s g \left[\right. z_{q} \left]\right. \parallel\right)_{2}^{2}$

第一项可以理解为不变， $z_{q}$ 主要靠近，第二项相反，由此我们可以给第二项设置一个相对较小的权重，来达到更希望主要靠近的效果。

**整体损失项：**

$\mathcal{L} = \left(\parallel x - decoder ⁡ \left(\right. z_{e} + sg ⁡ \left[\right. z_{q} - z_{e} \left]\right. \left.\right) \parallel\right)_{2}^{2} + \left(\parallel s g \left[\right. z_{e} \left]\right. - z_{q} \parallel\right)_{2}^{2} + \beta \left(\parallel z_{e} - s g \left[\right. z_{q} \left]\right. \parallel\right)_{2}^{2}$ 文中指出，实验发现 $\beta$ 设置\[0,1\]均具有鲁棒性，故使用 $\beta = \frac{1}{4}$ ，还可以使用滑动平均的方式更新，下面阐述。

**滑动平均方法：**

具体来说使用 指数移动平均（EMA）来更新 codebook ： $\parallel s g \left[\right. z_{e} \left(\right. x \left.\right) \left]\right. - e \parallel_{2}^{2} .$

设 $z_{i , 1} , z_{i , 2} , \ldots , z_{i , n_{i}}$ 为编码器输出中最接近词典项 $e_{i}$ 的一组 $n_{i}$ 个元素，那么可以将损失写为：

$\sum_{j = 1}^{n_{i}} \parallel z_{i , j} - e_{i} \parallel_{2}^{2} .$

理论上可以求得 $e_{i}$ 的最优值，可以通过封闭形式的解求得，即该集合中所有元素的平均值： $e_{i} = \frac{1}{n_{i}} \sum_{j = 1}^{n_{i}} z_{i , j} .$

这种更新方法通常用于 K-Means 等算法。然而，当处理小批量（minibatches）时，无法直接使用上述更新方式。因此，我们可以采用指数移动平均，作为该更新的在线版本： $N_{i}^{\left(\right. t \left.\right)} & := N_{i}^{\left(\right. t - 1 \left.\right)} * \gamma + n_{i}^{\left(\right. t \left.\right)} \left(\right. 1 - \gamma \left.\right) \\ m_{i}^{\left(\right. t \left.\right)} & := m_{i}^{\left(\right. t - 1 \left.\right)} * \gamma + \underset{j}{\sum} z_{i , j}^{\left(\right. t \left.\right)} \left(\right. 1 - \gamma \left.\right) \\ e_{i}^{\left(\right. t \left.\right)} & := \frac{m_{i}^{\left(\right. t \left.\right)}}{N_{i}^{\left(\right. t \left.\right)}}$

其中， $\gamma$ 的取值范围在 0 到 1 之间，论文发现 0.99 是一个不错的选择。

### 应用：

按照之前 VAE 的逻辑，使用时去掉encoder，在正态分布里采样即可生成图片；那么VQ-VAE呢？其假设先验分布为均匀分布，然而并没有直接在均匀分布里采样，而是使用 **[PixelCNN](https://zhida.zhihu.com/search?content_id=249509248&content_type=Article&match_order=1&q=PixelCNN&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODA3MzQyODUsInEiOiJQaXhlbENOTiIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI0OTUwOTI0OCwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.ek2_13_Db4vgTfUxjP6frFDPA4oBOOsgyXeeS0dqo0w&zhida_source=entity)** 来学习编码的分布（这里非常奇怪，在issue一节讨论），即学习 $z_{q} \left(\right. x \left.\right)$ 。

简单介绍下，PixelCNN 是一种采用自回归方式逐像素从左上角生成的图像生成模型，其中使用了mask conv操作，可以类比 GPT，使用 mask self-attention 操作。

![](https://pica.zhimg.com/v2-c2d1e86942837ccf29d5278319fbdab4_1440w.jpg)

所以最后我们通过 PixelCNN 来随机生成 $z_{q} \left(\right. x \left.\right)$ ，然后再用VQ-VAE的 Decoder 来生成最后的图片。

### Issue：

**VQ-VAE 到底是不是 VAE ？**

VAE 的核心是encoder学习一个先验分布，最后只需要从这个先验分布里采样就可以用来生成，然而VQ-VAE事实上并不行，其假设先验分布为均匀分布，但并不能从均匀分布里采样解码得到真实图像，这就说明这就 **不过只是一个AE 类模型** 。

那么问题出在哪了？回顾 VQ-VAE 的设计，发现并没有类似 VAE 里的 KL散度loss 来迫使先验分布逼近均匀分布。你可能会问假设分布是均匀分布，KL散度是一个常数呀，上面不是还推导了？那么我们再回顾一下：

$K L \left(\right. q_{\phi} \left(\right. z \left|\right. x \left.\right) \left|\right. \left|\right. p \left(\right. z \left.\right) \left.\right) = log ⁡ K$

KL散度是常数，那么这一项就不会优化，也就不存在要让 $q_{\phi} \left(\right. z \left|\right. x \left.\right)$ 更逼近 $p \left(\right. z \left.\right)$ 的说法，也就是 $q_{\phi} \left(\right. z \left|\right. x \left.\right)$ 不会被更新，其生成的分布根本不可控。

那么继续深究，这一项为何会是常数？原因就在于 $q_{\phi} \left(\right. z \left|\right. x \left.\right)$ 始终是一个one-hot分布，无论怎么优化都是如此，而one-hot分布和均匀分布的 KL散度 始终是 logK，因此 ELBO里 的这一项毫无意义。

其实本质上VQ-VAE 做的是【 **图像 tokenization** 】的工作，生成模型部分交给自回归模型 PixelCNN 去负责了。

> 此外：苏神在 [博客](https://link.zhihu.com/?target=https%3A//spaces.ac.cn/archives/6760/comment-page-2%23comments) 评论里还指出 VQ-VAE里边从均匀分布采样离散的code直接传入decoder，生成结果也不至于差得完全不可看，还是勉强能看的，比纯AE要好点，但要保证质量，还是得 pixelcnn。

**VQ-VAE 的核心贡献？**

核心贡献不在于其提出了一种新的 VAE 架构，而在于提供了一个序列压缩技术。正如上所说，其本质是一个利用codebook 做图像 tokenization 的工作，然而这种 codebook 的思想不仅可以应用于图像，音频、视频甚至短视频、广告都是可以的，所以我们才看到VQ-VAE的思想应用于各个领域，这才是VQ-VAE的魅力所在。

### VQ-VAE-2：

论文： [arxiv.org/pdf/1906.0044](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1906.00446)

![](https://picx.zhimg.com/v2-9dbfac884baa83c2f223afef877a15f1_1440w.jpg)

主要变化就是把 VQ-VAE 的 encoder 和 decoder 都进行了分层, bottom层对local feature进行建模，top层采取全局自注意力机制。

![](https://pic3.zhimg.com/v2-0f1497620138f5957f565973846d091c_1440w.jpg)

## RQ-VAE：

paper： [arxiv.org/pdf/2203.0194](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2203.01941)

![](https://pic1.zhimg.com/v2-3ac8bbd6e225e771f3ab5a26a00d0746_1440w.jpg)

paper：https://arxiv.org/pdf/2203.01941

### 背景：

VQ-VAE 的序列长度较长，需要大量的codebook，这势必会导致 *codebook collapse（码本摊缩）* 问题，使得VQ-VAE的训练很不稳定；而 RQ-VAE 则采取一种 *residual quantization（残差量化）* 的新方法，通过D轮迭代，将feature map表示为D个堆叠的离散编码，可以进一步减小feature map（可以理解为经过encoder后的表示）的spatial resolution，例如从原始图像的256\*256变为8\*8。这样 **进一步增加下采样因子** 减少分辨率，使得 AR 模型能够减少计算成本、提高图像生成速度，并更好地学习codebook中各向量之间的长依赖关系。

### 方法：

![](https://pic4.zhimg.com/v2-68a25d23f527b7ce60cd9df0109a52eb_1440w.jpg)

**RQ v.s. VQ：**

VQ：

假设codebook表示为 $\mathcal{C}$ ，对于向量 $\mathbf{z}$ ，其映射为近邻向量的操作表示为： $\mathcal{Q} \left(\right. \mathbf{z} ; \mathcal{C} \left.\right) = \underset{k \in \left[\right. K \left]\right.}{arg ⁡ min} \parallel \mathbf{z} - \mathbf{e} \left(\right. k \left.\right) \parallel_{2}^{2}$ 给定图片输入为 $\mathbf{X} \in \mathbb{R}^{H_{o} \times W_{o} \times 3}$ ，提取的 feature map 为： $\mathbf{Z} = E \left(\right. \mathbf{X} \left.\right) \in \mathbb{R}^{H \times W \times n_{z}}$ （ $\left(\right. H , W \left.\right) = \left(\right. H_{o} / f , W_{o} / f \left.\right)$ ），通过映射后得到的code map为： $\mathbf{M}_{h w} = \mathcal{Q} \left(\right. \mathbf{Z}_{h w} ; \mathcal{C} \left.\right)$ ，其中 $\mathbf{Z}_{h w} \in \mathbb{R}^{n_{z}}$ 是feature map中(h,w)位置上的向量。

假设 codebook 大小为 K，那么整个feature map为 $H W log_{2} ⁡ K$ 个 bit，根据 *rate-distortion theory（率失真理论）* ，H和W每缩小一半，K都要增加到 $K^{4}$ ，因此说VQ-VAE需要大量的codebook。

RQ：

在RQ里，定义新的映射为近邻向量的操作：

$\mathcal{R} \mathcal{Q} \left(\right. \mathbf{z} ; \mathcal{C} , D \left.\right) = \left(\right. k_{1} , \hdots , k_{D} \left.\right) \in \left[\right. K \left]\right.^{D}$ 可以看到并非之前单一的数字，而是一个元组，那么每一位的k如何选择？首先初始化残差 $\mathbf{r}_{0} = \mathbf{z}$ ，然后按照如下方法计算： $k_{d} & = \mathcal{Q} \left(\right. \mathbf{r}_{d - 1} ; \mathcal{C} \left.\right) \\ \mathbf{r}_{d} & = \mathbf{r}_{d - 1} - \mathbf{e} \left(\right. k_{d} \left.\right)$ 可以这么理解，我要模拟 $\mathbf{z}$ ，但是我模拟的 $\mathbf{e} \left(\right. k_{1} \left.\right)$ 肯定和 $\mathbf{z}$ 有差距，我用 $\mathbf{r}_{1} = \mathbf{r}_{0} - \mathbf{e} \left(\right. k_{1} \left.\right) = \mathbf{z} - \mathbf{e} \left(\right. k_{1} \left.\right)$ 表示出来这两者的差，然后我继续模拟 $\mathbf{r}_{1}$ ，但是我模拟的 $\mathbf{e} \left(\right. k_{2} \left.\right)$ 肯定又和 $\mathbf{r}_{1}$ 有差距，我用 $\mathbf{r}_{2}$ 表示出来...... 因此每个 $\mathbf{r}$ 逐步相加，理论上和要模拟的 $\mathbf{z}$ 越来越逼近。

可以看出VQ将空间分为K个簇，而RQ将空间分为 $K^{D}$ 个簇，来实现更精确的量化。

**共享codebook机制：**

虽然我们可以为每一层深度 d 分别构建一个码本，但在每个量化深度上使用的是单个共享码本。共享码本在构建 RQ 近似向量 z 时有两个优势：

1. 使用单独的码本需要广泛的超参数搜索，以确定每一层的码本大小，而共享码本只需确定总码本大小 K。

2\. 共享码本使得所有的 embedding 在每一层量化时都可用。因此，每一层都可以使用相同的 embedding，以最大化其效用。

**RQ-Transformer：**

![](https://pic1.zhimg.com/v2-9c41422b57af4b8485ca82a6f83cd9bc_1440w.jpg)

可以看出编码得到的 feature map 输入给 Transformer 来作为自回归任务的输入，整个 RQ-Transformer 分为Spatial Transformer和 Depth Transformer 两部分。

**输入处理：**

RQ-VAE 提取的代码映射 $\mathbf{M} \in \left[\right. K \left]\right.^{H \times W \times D}$ 会按照 栅格扫描顺序（raster-scan order）重新排列为二维数组 $\mathbf{S} \in \left[\right. K \left]\right.^{T \times D}$ ，其中 $T = H \times W$ 。 每一行 $\mathbf{S}_{t}$ 包含 D 个代码：

$\mathbf{S}_{t} = \left(\right. \mathbf{S}_{t 1} , \hdots , \mathbf{S}_{t D} \left.\right) \in \left[\right. K \left]\right.^{D} , t \in \left[\right. T \left]\right.$

自回归建模总公式为：

$p \left(\right. \mathbf{S} \left.\right) = \prod_{t = 1}^{T} \prod_{d = 1}^{D} p \left(\right. \mathbf{S}_{t d} \mid \mathbf{S}_{< t , d} , \mathbf{S}_{t , < d} \left.\right)$

**建模动机：**

直接将 $\mathbf{S}$ 展开为长度 TD 的序列并输入传统 Transformer 的方法存在不足，无法利用导 RQ-VAE 降低后的长度 T的优势。此外，这种直接展开会增加计算成本。由此设计为 Spatial Transformer和 Depth Transformer 两部分。

**空间 Transformer（Spatial Transformer）** ：

首先空间 Transformer的输入为每个位置上的 feature（各个残差项之和），并加上位置编码（PE），如下：

$\mathbf{u}_{t} = \text{PE}_{T} \left(\right. t \left.\right) + \sum_{d = 1}^{D} \mathbf{e} \left(\right. \mathbf{S}_{t - 1 , d} \left.\right) , \text{for } t > 1.$

整个 Spatial Transformer 表示为： $\mathbf{h}_{t} = \text{SpatialTransformer} \left(\right. \mathbf{u}_{1} , \hdots , \mathbf{u}_{t} \left.\right) .$

**深度 Transformer (Depth Transformer)：**

深度 Transformer 的任务是在给定位置 t 自回归地预测 D 个残差项code，即 $\mathbf{S}_{t 1} , \hdots , \mathbf{S}_{t D}$

在深度 d 和位置 t 时，Transformer 的输入 $\mathbf{v}_{t d}$ 被定义为 **之前深度的嵌入之和** ： $\mathbf{v}_{t d} = \text{PE}_{D} \left(\right. d \left.\right) + \sum_{d^{'} = 1}^{d - 1} \mathbf{e} \left(\right. \mathbf{S}_{t d^{'}} \left.\right) , d > 1.$

每个深度的预测基于之前所有深度的估计，使得每一层的估计更加精细。

$\text{PE}_{D} \left(\right. d \left.\right)$ 是深度 d 的位置嵌入，且在所有位置 t 上共享。

整个 Depth Transformer 表示为：

$\mathbf{p}_{t d} = \text{DepthTransformer} \left(\right. \mathbf{v}_{t 1} , \hdots , \mathbf{v}_{t d} \left.\right) .$

### 训练：

RQ-VAE 的训练损失函数 $\mathcal{L}$ 包含两部分： $\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{commit}}$

**重构损失（Reconstruction Loss）** ： $\mathcal{L}_{\text{recon}} = \parallel \mathbf{X} - \hat{\mathbf{X}} \parallel_{2}^{2}$

这个损失度量的是输入 $\mathbf{X}$ 和重构结果 $\hat{\mathbf{X}}$ 之间的欧氏距离，用于确保重构后的样本尽可能接近原始输入。这里同样会采用 Straight-Through Estimator。

**承诺损失（Commitment Loss）** ： $\mathcal{L}_{\text{commit}} = \sum_{d = 1}^{D} \parallel \mathbf{Z} - \text{sg} \left[\right. \hat{\mathbf{Z}}^{\left(\right. d \left.\right)} \left]\right. \parallel_{2}^{2}$

（sg\[·\] 是 stop-gradient 操作符，用于在反向传播时阻止梯度的传递），该损失的作用是最小化每个维度 d 上的量化误差，从而鼓励编码器的输出 $\mathbf{Z}$ 更接近量化后的值 $\hat{\mathbf{Z}}^{\left(\right. d \left.\right)}$ 。

论文内提及codebook会采用聚类特征的 **指数滑动平均** 来更新，从而提升模型的训练效果和稳定性。

RQ-VAE 同时还采用了 **对抗训练** （Adversarial Training ）以提高重构图像的感知质量。采用了基于 patch 的对抗损失和感知损失。

**负对数似然损失 (Negative Log-Likelihood, NLL)** ：

用于训练 RQ-Transformer：

$\mathcal{L}_{A R} = \mathbb{E}_{S , t , d} \left[\right. - log ⁡ p \left(\right. \mathbf{S}_{t d} \mid \mathbf{S}_{< t , d} , \mathbf{S}_{t , < d} \left.\right) \left]\right. .$

### Trick：

**曝光偏差 (Exposure Bias)：**

曝光偏差是自回归（AR）模型中的常见问题。在训练和推断阶段，由于预测错误的累积，模型性能会下降。尤其是在 RQ-Transformer 中，随着深度 D 的增加，量化特征向量的估计变得更加困难，误差也会累积。

论文采用了软标签 (Soft Labeling) 和 随机采样 (Stochastic Sampling)策略：

**软标签（Soft Labeling）：**

基于 RQ-VAE 中代码嵌入之间的几何关系，定义了一个温度参数 $\tau > 0$ 控制的类别分布： $Q_{\tau} \left(\right. k \mid \mathbf{z} \left.\right) \propto e^{- \frac{\parallel \mathbf{z} - \mathbf{e} \left(\right. k \left.\right) \parallel_{2}^{2}}{\tau}} , k \in \left[\right. K \left]\right. .$

当 $\tau \rightarrow 0$ 时，分布 $Q_{\tau}$ 会收缩为一个 one-hot 分布： $Q_{0} \left(\right. k \mid \mathbf{z} \left.\right) = 1 \left[\right. k = Q \left(\right. \mathbf{z} ; C \left.\right) \left]\right. .$

软标签的作用：

利用嵌入之间的几何距离，为目标代码的监督引入了软标签分布; 在位置 $t$ 和深度 $d$ 上，假设特征向量为 $\mathbf{Z}_{t}$ ，并令残差向量为 $r_{t , d - 1}$ 。负对数似然（NLL）损失使用了该软分布作为监督。

区别于 one-hot 标签，该监督机制使用了软化后的分布 $Q_{\tau} \left(\right. \cdot \mid r_{t , d - 1} \left.\right)$ 。

**随机采样（Stochastic Sampling）：**

在原始的 RQ-VAE 中，代码选择是确定性的。然而，这里通过从软分布 $Q_{\tau} \left(\right. \cdot \mid r_{t , d - 1} \left.\right)$ 中进行采样来选择代码 $\mathbf{S}_{t d}$ 。 当 $\tau \rightarrow 0$ 时，随机采样等价于原始确定性代码选择。

优势：随机采样为特征映射提供了不同的代码组合，从而缓解了训练和推断中的不一致性。

## FSQ：

paper： [Finite Scalar Quantization: VQ-VAE Made Simple](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2309.15505)

![](https://pic2.zhimg.com/v2-f146b33878365d9ab4a8d27ffe6af96f_1440w.jpg)

### 方法：

![](https://picx.zhimg.com/v2-02f6ebdffdd1545cfacedd3bcc0bfa47_1440w.jpg)

论文提出使用 FSQ（Finite Scalar Quantization） 来替代 VQ-VAE中的“VQ”，其离散化思路非常简单，就是“四舍五入”。如上图所示，假设最后要把x映射为d维（图中d=3），我们把z的每一维用L个value表示（图中L=3），然后将z的每一维的L个value四舍五入（图中则变化为正方体的边线所在顶点处），由此便离散化了。

还有个区别图式中便是VQ里量化后的 $\hat{z}$ 会用一个单独的数字代替，表示codebook里的索引；而FSQ里会用L个数字组成的元组（例如(-1,0,1)）来替代，也表示索引，整体codebook数量为L^d，图里为9。

方案对比如下：

![](https://pic4.zhimg.com/v2-c63c592d630959636f500b506644ed17_1440w.jpg)

具体来说给定一个 $d$ -维表示 $z \in \mathbb{R}^{d}$ ，我们的目标是将 $z$ 量化为有限的码字集。为此，我们首先应用一个边界函数 $f$ ，然后将结果 **四舍五入为整数** 。我们选择 $f$ 使得 $\hat{z} = \text{round} \left(\right. f \left(\right. z \left.\right) \left.\right)$ 取得 $L$ 个唯一值之一（例如， $f : z \rightarrowtail \lfloor L / 2 \rfloor tanh ⁡ \left(\right. z \left.\right)$ ），上图的右子图可视化了这个转化，由于tanh取值范围为(-1，1)，由此z的范围是 $\left(\right. - \lfloor L / 2 \rfloor , \lfloor L / 2 \rfloor \left.\right)$ ，故四舍五入后便是L个取值，图中L=5，则有-2,-1,0,1,2这5个取值。

由此，我们得到 $\hat{z} \in C$ ，其中 $C$ 便是码本，且 $\left|\right. C \left|\right. = L^{d}$ 。

为了在整个四舍五入操作中传播梯度，使用了前述 STE（直通估计） 技巧 ，通过以下方式轻松实现“停止梯度（sg）”操作：

$\text{round}_{\text{ste}} : x \rightarrowtail x + \text{sg} \left(\right. \text{round} \left(\right. x \left.\right) - x \left.\right)$

### 实验：

![](https://pic1.zhimg.com/v2-d4660b9c3c6da74e2d901293ad05d8b2_1440w.jpg)

从图中可以看到，编码表大小2^10是一个分界点，在2^10左右时，FSQ与VQ的效果接近；超过2^10时，FSQ占优，反之小于2^10时，VQ占优。文中建议 $L \geq 5$ ，并且d是个位数，相比之下VQ-VAE中d是三位数。

## 引用：

[Elijha：VQ-VAE解读](https://zhuanlan.zhihu.com/p/91434658)

[Variational Autoencoders](https://link.zhihu.com/?target=https%3A//amaires.github.io/VAE/)

[变分自编码器（二）：从贝叶斯观点出发 - 科学空间|Scientific Spaces](https://link.zhihu.com/?target=https%3A//spaces.ac.cn/archives/5343)

[VQ-VAE的简明介绍：量子化自编码器 - 科学空间|Scientific Spaces](https://link.zhihu.com/?target=https%3A//spaces.ac.cn/archives/6760)

[简单得令人尴尬的FSQ：“四舍五入”超越了VQ-VAE - 科学空间|Scientific Spaces](https://link.zhihu.com/?target=https%3A//www.spaces.ac.cn/archives/9826)

编辑于 2024-10-25 15:11・广东[图像生成模型](https://www.zhihu.com/topic/29196693)[LLM（大型语言模型）](https://www.zhihu.com/topic/26797383)[推荐算法](https://www.zhihu.com/topic/19580544)