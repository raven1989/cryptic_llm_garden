---
title: "字节RankMixer:特征交叉新范式大幅提升推荐精排模型MFU"
source: "https://zhuanlan.zhihu.com/p/1937436392402683166"
author:
  - "[[州懂公众号:州懂学习笔记 & 某大厂算法工程师]]"
published:
created: 2026-07-07
description: "大家好, 我是州懂, 今天分享一篇字节基于特征交叉提升推荐模型MFU的文章。 标题: RankMixer: Scaling Up Ranking Models in Industrial Recommenders 地址: https://www.arxiv.org/pdf/2507.15551 公司: 字节1. 前…"
tags:
  - "clippings"
---
目录

收起

1\. 前言

2\. 方法

2.1 特征Token化处理

2.2 RankMixer Block

2.2.1 Multi-head Token Mixing

2.2.2 Per-token FFN

2.3 Sparse MoE

1）ReLU Routing

2) Dense训练/Sparse推理

3\. 实验部分

3.1 整体效果

3.2 Scaling Laws

3.3 消融实验

3.4 Sparse-MoE相关实验

3.5 线上实验

3.5.1 线上部署及开销

3.5.2 线上AB实验效果

大家好, 我是州懂, 今天分享一篇字节基于特征交叉提升推荐模型MFU的文章。

> 标题: [RankMixer](https://zhida.zhihu.com/search?content_id=261462287&content_type=Article&match_order=1&q=RankMixer&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODM1ODMzNDEsInEiOiJSYW5rTWl4ZXIiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNjE0NjIyODcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.vhfsuLwwQaiOFYAU8Cg26FgtUVYk6B_sRWtScOslh1Q&zhida_source=entity): Scaling Up Ranking Models in Industrial Recommenders  
> 地址: [arxiv.org/pdf/2507.1555](https://link.zhihu.com/?target=https%3A//www.arxiv.org/pdf/2507.15551)  
> 公司: 字节

## 1\. 前言

前些天快手OneRec技术报告提到当前推荐系统的MFU(Model Flops Utilization, 算力利用率)极低, 即使是最复杂的精排模型也通常只有百分之几, 远低于现在LLM的MFU(普遍能达40%+)。这显然阻碍了推荐模型的Scaling Up。

字节这篇论文认为, 现代推荐系统常用的特征交叉范式(如 [DeepFM](https://zhida.zhihu.com/search?content_id=261462287&content_type=Article&match_order=1&q=DeepFM&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODM1ODMzNDEsInEiOiJEZWVwRk0iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNjE0NjIyODcsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.px8495LSyyW5x_ViCBII0Ro3rcM_Sehpu9Tzpc3NCHg&zhida_source=entity) ， DCN等)是传统CPU时代的产物, 它们其实还是memory-bound的, 已经无法充分利用好现代GPU算力。因此, 字节这里尝试提出GPU时代的特征交叉新范式(compute-bound), 使推荐精排模型的MFU从4.5%一下提升到45%, 并在不提高推理时延的约束下, 成功将精排模型参数提升到1B量级。

## 2\. 方法

RankMixer的整体框架如下图所示:

![](https://pic4.zhimg.com/v2-59b8500eac2ccf2c5fb86e574daa12ef_1440w.jpg)

其由层RankMixer Blocks组成, 每层Block包含两个子模块: Token Mixing和 [Per-token FFN](https://zhida.zhihu.com/search?content_id=261462287&content_type=Article&match_order=1&q=Per-token+FFN&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODM1ODMzNDEsInEiOiJQZXItdG9rZW4gRkZOIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjYxNDYyMjg3LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.YYCeU85BQlC1zN2JxNNTslNRYB4oPhobJWppw_oms3I&zhida_source=entity) 模块, 形式化描述如下:

$\mathbf{S}_{n - 1} = L N \left(\right. T o k e n M i x i n g \left(\right. \mathbf{X}_{n - 1} \left.\right) + \mathbf{X}_{n - 1} \left.\right)$ $\mathbf{X}_{n} = L N \left(\right. P F F N \left(\right. \mathbf{S}_{n - 1} \left.\right) + \mathbf{S}_{n - 1} \left.\right)$

其中, $\mathbf{X}_{0} \in \mathbb{R}^{T \times D}$ 是基于原始推荐系统异构特征来处理得到 $\mathbf{x}_{1}$, $\mathbf{x}_{2}$,..., $\mathbf{x}_{T}$, 这个过程称为Feature Tokenization。

下面详细介绍各个子模块。

### 2.1 特征Token化处理

RankMixer的特征输入与常规精排模型没什么不同, 包含了User features、candidate features、Sequence Features、以及Cross Features。这些异构特征的embedding维度大小不一, 为了在后续阶段更高效的并行计算, 需要将这些特征处理成维度大小一的Token. 但是, 如果为每个特征都单独分配token会产生数百个小token，导致GPU核心利用率不足。

RankMixer在特征Token化处理方式上也比较直接: 先将特征按语义分成若干组, 比如将用户画像类的特征年龄、性别、城市、职业等划分为一组特征, 然后对每个分组按预设的固定维度大小 $\mathbb{R}^{D}$ 分割成若干个Token, 最后再将所有分组的Token汇总起来作为RankMixer模型的输入 $\mathbf{X}_{0} \in \mathbb{R}^{T \times D}$ 。

### 2.2 RankMixer Block

### 2.2.1 Multi-head Token Mixing

下图很直观地表述了多头Token Mixing的处理流程。这里论文设置以维持相同的Token数量, 并加了残差连接和LayerNorm:

$\mathbf{s}_{1} , \mathbf{s}_{2} , . . . , \mathbf{s}_{T} = L N \left(\right. T o k e n M i x i n g \left(\right. \mathbf{x}_{1} , \mathbf{x}_{2} , . . . , \mathbf{x}_{T} \left.\right) + \left(\right. \mathbf{x}_{1} , \mathbf{x}_{2} , . . . , \mathbf{x}_{T} \left.\right) \left.\right)$

![](https://pic2.zhimg.com/v2-6f676c83f7b3ca9df500ef1c34ed4fc5_1440w.jpg)

作者这里并没有使用Self-Attention, 只是做了显式的特征交叉, 它是parameter-free的。作者提到, self-attention通过计算Token之间的内积来衡量它们的相似性，这在自然语言处理（NLP）中是有效的，因为所有Token共享一个统一的embed空间。然而，在推荐系统中，User侧和Item侧可能都有数百个异质特征, 这些特征的语义差异可能还很大, 在这种情况下, 使用内积去衡量不同特征空间的embed向量之间的相似度效果就没那么好了。

### 2.2.2 Per-token FFN

Per-token FFN是让每个Token(对应 $\mathbf{s}_{t}$)都各自独立经过一个参数隔离不共享的FFN（即两层的MLP):

$\mathbf{v}_{t} = f_{\text{pffn}}^{t , 2} \left(\right. G e l u \left(\right. f_{\text{pffn}}^{t , 1} \left(\right. \mathbf{s}_{t} \left.\right) \left.\right) \left.\right)$

其中

$f_{\text{pffn}}^{t , i} \left(\right. \mathbf{x} \left.\right) = \mathbf{x} \mathbf{W}_{\text{pffn}}^{t , i} + \mathbf{b}_{\text{pffn}}^{t , i}$

这种方式的好处有:

- 可以防止高频特征主导模型并淹没低频或长尾信号
- 可以极大程度地提高模型参数规模, 同时计算复杂度保持不变。

此外, 在Multi-head Token Mixing后，如果使用参数共享的FFN，Token往往会坍缩成相同的表示。

### 2.3 Sparse MoE

为了在计算复杂度可控的情况下, 进一步提升参数规模, 可以借鉴LLM使用稀疏MoE替换掉前面的FFN. 但是, 在RankMixer这里, 直接使用稀疏MoE会存在两个问题:

- **均匀专家路由问题**

常规稀疏MoE采用Top- $k$ 专家选择策略，对所有特征Token同等对待。这种均匀路由会导致: 浪费计算预算在低信息量的Token上, 使高信息量Token得不到足够的专家资源, 并阻碍模型捕捉不同Token间的差异。这种"一刀切"的路由方式无法适应推荐系统中特征Token信息量差异大的特点，降低了模型的参数效率。

- **专家训练不足问题**

Per-token FFN已经倍增了参数，添加非共享专家会进一步爆炸式增加专家数量，导致路由高度不平衡以及专家训练不足问题。

为了缓解上述两个问题, 作者提出了两个措施:

### 1）ReLU Routing

作者将Top𝑘+softmax的机制, 改成ReLU的方式, 以使高信息量Token可激活更多专家

$G_{i , j} = R e L U \left(\right. h \left(\right. \mathbf{s}_{i} \left.\right) \left.\right) , \mathbf{v}_{i} = \sum_{j = 1}^{N_{e}} G_{i , j} e_{i , j} \left(\right. \mathbf{s}_{i} \left.\right)$

其中, $N_{e}$ 表示每个Token中Expert的数量, $N_{t}$ 表示总的Token数。同时, 在Loss上增加对应的惩罚项:

$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{r e g} , \mathcal{L}_{r e g} = \sum_{i = 1}^{N_{t}} \sum_{j = 1}^{N_{e}} G_{i , j}$

### 2) Dense训练/Sparse推理

在训练阶段使用Dense activation, 在推理阶段使用Sparse activation。通过这种方式来缓解专家训练不足的问题, 不过, 这也会带来训练成本的增加。

## 3\. 实验部分

### 3.1 整体效果

![](https://pic3.zhimg.com/v2-2ba4d9c73a820ddde23142d94dbd29ce_1440w.jpg)

### 3.2 Scaling Laws

![](https://pic1.zhimg.com/v2-8249c01833281622ec38dbf55a375d38_1440w.jpg)

### 3.3 消融实验

RankMixer各子模块的消融实验

![](https://pic2.zhimg.com/v2-39ce992e9dce846465a38bfdaccfd80d_1440w.jpg)

基于其它token mixing策略来验证论文所提方法的效果:

- **All-Concat-MLP:** 将所有token拼接后通过一个大MLP处理，再分割回相同数量的token
- **All-Share:** 整个输入向量共享，每个per-token FFN接收相同的完整输入（类似MoE）
- **Self-Attention:** 在token间应用自注意力机制进行路由
![](https://pica.zhimg.com/v2-f57ea4ed708d617e5bfd062855eb715a_1440w.jpg)

### 3.4 Sparse-MoE相关实验

- Scalability
![](https://pic1.zhimg.com/v2-539e78c1cc97410750c5ba3fa98c97ec_1440w.jpg)

- Expert balance and diversity

RankMixer中不同Token的激活专家比率

![](https://pic4.zhimg.com/v2-a925807030e2d884d70806f6966fd74f_1440w.jpg)

### 3.5 线上实验

### 3.5.1 线上部署及开销

![](https://pic4.zhimg.com/v2-0074001bf79281f8a1fe6c3cba7675a5_1440w.jpg)

### 3.5.2 线上AB实验效果

推荐场景

![](https://pic3.zhimg.com/v2-7be6374fafe78d73c140e6472d999362_1440w.jpg)

广告场景

![](https://pic3.zhimg.com/v2-b2cada0212e0635c8f1aed42ab979802_1440w.jpg)

编辑于 2025-12-14 09:14・广东[字节跳动（ByteDance）](https://www.zhihu.com/topic/20168793)[推荐系统](https://www.zhihu.com/topic/19563024)[LLM](https://www.zhihu.com/topic/20660508)