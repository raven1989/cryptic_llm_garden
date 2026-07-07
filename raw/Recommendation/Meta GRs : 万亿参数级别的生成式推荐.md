---
title: "Meta GRs : 万亿参数级别的生成式推荐"
source: "https://zhuanlan.zhihu.com/p/691397927"
author:
  - "[[州懂公众号:州懂学习笔记 & 某大厂算法工程师]]"
published:
created: 2026-07-06
description: "大家好，我是州懂, 今天和大家分享一篇Meta最新的文章，介绍Meta重塑推荐系统范式的尝试, 并在推荐模态观测到了语言模态的scaling law, 工作十分亮眼, 美中不足的是不少细节都一笔带过。 标题: Actions Speak Loud…"
tags:
  - "clippings"
---
目录

收起

1\. 前言

2\. 生成式推荐

2.1 统一描述异质特征

2.1.1 分类型(sparse)特征

2.1.2 连续型(dense)特征

2.2 序列直推式任务:重塑召回与排序

2.2.1 召回

2.2.2 排序

2.3 生成式训练

3\. HSTU: 新的Self-Attention Encoder

3.1 HSTU整体框架

3.2 HSTU优化细节

3.2.1 新的注意力机制

3.2.2 提高稀疏性加速训练

3.2.3 最小化Activation内存使用

3.2.4 推理侧优化

4\. 实验

5\. 相关材料

大家好，我是州懂, 今天和大家分享一篇 [Meta](https://zhida.zhihu.com/search?content_id=241774464&content_type=Article&match_order=1&q=Meta&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODM0OTYyOTEsInEiOiJNZXRhIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjQxNzc0NDY0LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.PFqZzGZuGpeW-ajftKGzmKv8Tjbo__jv44wdwKbagcs&zhida_source=entity) 最新的文章，介绍Meta重塑推荐系统范式的尝试, 并在推荐模态观测到了语言模态的scaling law, 工作十分亮眼, 美中不足的是不少细节都一笔带过。

> 标题: Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations  
> 地址: [arxiv.org/pdf/2402.1715](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2402.17152.pdf)  
> 公司: Meta  
> 会议: ICML'24  
> 代码: [GitHub - facebookresearch/generative-recommenders: Repository hosting code for "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations" (https://arxiv.org/abs/2402.17152).](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/generative-recommenders)

## 1\. 前言

相比于大语言模型主要的应用场景, 当前深度学习推荐系统(论文简称DLRMs)训练算力规模还比较小, 与GPT-3/ [LLaMa-2](https://zhida.zhihu.com/search?content_id=241774464&content_type=Article&match_order=1&q=LLaMa-2&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODM0OTYyOTEsInEiOiJMTGFNYS0yIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjQxNzc0NDY0LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.Cs2XKnAEiMGqnddZfFvS5uxwDhx1_ruhFiHefwg7F6I&zhida_source=entity) 等都还有大概3个数量级的差异(如下图的DLRM20~DLRM22), 也还没有呈现出像GPT那样的算力规模效应。

![](https://picx.zhimg.com/v2-3fb07760f61a2b88a74a5010ddf161e1_1440w.jpg)

受近期大语言模型成功的启发, 作者认为, 为了缓解上述问题, 现代工业级大规模推荐系统需要克服3个挑战:

1. **特征缺乏统一的结构描述**: 推荐系统中的特征是异质的, 缺乏明确一致的结构描述, 比如交叉特征, 高基数id特征, 计数特征, 比率特征等, 这些特征有些是sparse的, 有些是dense的。
2. **物料池规模大&动态变化**: 推荐系统中物料池经常是数以亿计的, 且物料池动态变化, 不像NLP那样只有相对静态的几十万量级词汇量, 这使得推荐系统训练和推理的开销都很高。
3. **计算成本高是大规模序列模型落地的主要瓶颈**: 推荐系统每天需要处理的token比GPT-3在1-2个月内处理的token还要多上几个数量级。

为了解决上述问题, 实现推荐系统的scaling effect, 作者提出了生成式推荐(Generative Recommenders, 下文简称GRs), 将当前推荐系统异质的特征结构进行统一描述, 同时为了加速训练和减少推理开销, 提出了"分层序列直推单元"框架(Hierarchical Sequential Transduction Units, 下文简称 [HSTU](https://zhida.zhihu.com/search?content_id=241774464&content_type=Article&match_order=1&q=HSTU&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODM0OTYyOTEsInEiOiJIU1RVIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjQxNzc0NDY0LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.46HGvcL2EcbqC_c_PIiyxaIEzidxLqnwRdbD8WYT0nI&zhida_source=entity))。

## 2\. 生成式推荐

### 2.1 统一描述异质特征

在生成式推荐系统GRs里, 作者将推荐系统的sparse特征和dense特征转换成了统一的时间序列来处理, 如下图所示, 具体处理如下:

![](https://pic4.zhimg.com/v2-87a270503cb6b41bf849976793a72c69_1440w.jpg)

### 2.1.1 分类型(sparse)特征

推荐系统常见的sparse特征有: 用户收藏的item序列, 用户关注的作者ID, 人口统计信息等, 具体转换过程如下:

1. 选择时间跨度最长的序列作为主时间序列, 按时间顺序记录用户交互过的Item特征, 将它们合并进主时间序列中, 序列中的信息主要包含交互itemID、交互时间戳、交互的行为类型等。
2. 压缩其它变化比较慢的特征,如人口统计信息,用户关注的作者等, 在一段时间内只保留最早的特征值, 合并进时间序列中。如上图中的两个辅助时间序列中的、 $H_{0}$ 和, 这些是偏静态的特征, 一段时间内只插入一次到主序列中。

### 2.1.2 连续型(dense)特征

常见的dense特征有计数, 比率(如ctr)等, 作者认为这些特征在生成式推荐系统里可以舍弃:

1. 这些特征通常更新频繁, 可能(user, item)每发生一次交互, 都要计算一次, 因此,以时间序列计算&存储这些特征代价太高, 并不太划算
2. 很多dense特征的信息已经在前面sparse的时间序列特征中多多少少携带了, 特别是, 当我们增加序列建模的长度, 通过直推式序列架构(sequential transduction architecture)以及目标感知(target-aware)的方式建模, 这些dense特征的信息,可能就能被捕获得很好了。

### 2.2 序列直推式任务:重塑召回与排序

首先给出文章使用的符号定义:

- $\mathbb{X}$: 动态更新的物料池(全集)
- $\mathbb{X}_{c}$: 用户历史交互的物料
- $x_{0} , x_{1} , . . . , x_{n - 1}$: 输入的tokens
- $y_{0} , y_{1} , . . . , y_{n - 1}$: 输出的tokens, $y_{i} \in \mathbb{X} \cup \left{\right. \emptyset \left.\right}$, 当时表示的值是undefined的。
- $a_{i}$ 与 $\Phi_{i} \in \mathbb{X}_{c}$ ($\mathbb{X}_{c} \subseteq \mathbb{X}$): 用户行为行为(如收藏,完播等), 以及行为所对应的内容(如视频,商品)
- $t_{0} , t_{1} , . . . , t_{n - 1}$: 行为所对应的时间点
- $m_{0} , m_{1} , . . . , m_{n - 1}$: 掩码序列, $m \in \left{\right. 0 , 1 \left.\right}$, 当时表示的值是undefined的。

标准的召回和排序, 在本文的框架下, 其输入输出可以表示如下:

![](https://pica.zhimg.com/v2-4c7ceb2b4798fb2556fa06b6b7eb712a_1440w.jpg)

论文里提到sequential transduction, 是seq2seq的概念, 也就是在给定输入 $x$ 和掩码 $m$ 情况下, 预测输出对应的 $y$ 。这里名字"transduction"涉及到 **归纳式学习** 与 **直推式学习** 的基本概念, 具体可参考下图:

![](https://pic4.zhimg.com/v2-3608da200118bbeae14e3868d4e5a699_1440w.jpg)

### 2.2.1 召回

召回任务学习一个概率分布, 其中为第个时刻所对应的用户表征, 学习目标可以为: arg $m a x_{x \in \mathbb{X}_{c}} p \left(\right. x \mid u_{i} \left.\right)$

这种方式与标准自回归方法存在两个区别:

1. 下一个token $x_{i + 1}$ 不一定是前面, $y_{i}$ 的监督信号, 因为用户可能对负反馈
2. 当下一个token $x_{i + 1}$ 是像前面提到人口统计信息这样相对静态的特征时, 这些特征会被压缩, 并不是交互的Item, 即, 这时是undefined

对于前面的两种情况, 会将对应的掩码设置为0, 其它情况则是按自回归方式处理。

### 2.2.2 排序

推荐系统中的排序通常是"目标感知"的(比如会做target attention), 当前target $x_{i + 1}$ 和历史行为特征需要做交叉, 且在模型底层越早越好, 但这个在标准自回归方法里是不太可行的, 因为它太晚才做交叉了(比如在encoder输出后才用softmax), 无法在Encoder内部底层提前做交叉，效果会大打折扣。

为了解决这个问题, 作者将item及用户行为action(如收藏,查看评论等)插入了主时间序列,这样新的时间序列就会是。对于这些action的位置, 会做mask, 让对应的设置为0, 然后, 在内容的位置会进行多任务预测, 即学习.

通过这样的处理, 可以一遍就对所有用户交互过的item做target-aware cross-attention了。

### 2.3 生成式训练

工业级的推荐系统通常是流式训练的, 在这种方式下, 使用self-attention的序列直推框架(如 [Transformer](https://zhida.zhihu.com/search?content_id=241774464&content_type=Article&match_order=1&q=Transformer&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODM0OTYyOTEsInEiOiJUcmFuc2Zvcm1lciIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI0MTc3NDQ2NCwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.5To9PS55G1D_2k78TzFtmUcaNlxWxg9cy2E0QtZqxLc&zhida_source=entity) s)训练, 总的计算量会是 $\underset{i}{\sum} n_{i} \left(\right. n_{i}^{2} d + n_{i} d_{f f} d \left.\right)$, 其中是用户的token数量, $d$ 是embedding的维度. 括号里的第一部分是self-attention的开销, 第二部分是pointwise MLP层的开销。取, 则总的计算时间复杂度为, 这个开销实在太高了。

为了缓解计算开销问题, 作者舍弃传统的曝光粒度的训练方式, 转向生成式训练, 将时间复杂度减少了。这里可以这么理解, 在训练时, 这些曝光粒度的Item, 都会被合并到主序列中, 只需训练一遍。另外, 作者也做了用户维度上的采样, 进一步减少了时间复杂度。

## 3\. HSTU: 新的Self-Attention Encoder

为了提高训练速度, 作者提出了新的Self-Attention Encoder结构:Hierarchical Sequential Transduction Unit (HSTU)。

### 3.1 HSTU整体框架

HSTU由残差连接的层堆叠而成。每一层包含3个子层, 分别是:

- Pointwise Projection:

$U \left(\right. X \left.\right) , V \left(\right. X \left.\right) , Q \left(\right. X \left.\right) , K \left(\right. X \left.\right) = Split ⁡ \left(\right. \phi_{1} \left(\right. f_{1} \left(\right. X \left.\right) \left.\right) \left.\right) \left(\right. 1 \left.\right)$

- Spatial Aggregation(这是论文的一个创新点, 后面详细阐述):

$A \left(\right. X \left.\right) V \left(\right. X \left.\right) = \phi_{2} \left(\right. Q \left(\right. X \left.\right) K \left(\right. X \left.\right)^{T} + rab^{p , t} \left.\right) V \left(\right. X \left.\right) \left(\right. 2 \left.\right)$

- Pointwise Transformation:

$Y \left(\right. X \left.\right) = f_{2} \left(\right. Norm ⁡ \left(\right. A \left(\right. X \left.\right) V \left(\right. X \left.\right) \left.\right) \bigodot U \left(\right. X \left.\right) \left.\right) \left(\right. 3 \left.\right)$

其中, $f_{i} \left(\right. X \left.\right)$ 表示MLP, 这里作者只使用了linear layer简化计算复杂度, $\phi_{1}$ 和 $\phi_{2}$ 为非线性激活函数, 本文作者使用了SiLU: $s i l u \left(\right. x \left.\right) = x * \sigma \left(\right. x \left.\right) \left.\right)$, $r a b^{p , t}$ 表示相对位置偏置。

与传统的深度学习推荐系统(DLRMs)相比, HSTU仅有一个单一的模块, 替换了DLRMs的分层结构(Feature Extraction, Feature Interactions, and Transformations of Representations), 如下图所示, 其中, 左边为DLRMs的分层结构, 右边为HSTU的结构。

![](https://picx.zhimg.com/v2-ce2477a318024fd85b48d57680bcd4f9_1440w.jpg)

### 3.2 HSTU优化细节

### 3.2.1 新的注意力机制

不同于Transformer直接使用softmax attention, HSTU使用了新的注意力机制(Pointwise aggregated attention, 见上面公式2), 出于两个方面考虑:

1. 在推荐中，与目标相关的先验数据点的数量是一个强有力的特征，表明用户偏好的强度，这在 softmax 归一化后很难捕捉到, 而这是至关重要的，因为我们需要同时预测值和序。
2. 虽然softmax 激活对噪声具有鲁棒性，但它不太适合物料动态更新的流式训练。

值得注意的是, 为了训练稳定, 在pointwise pooling之后, 需要加上Lyaer Norm。下表给出了不同注意力机制在仿真数据上的效果差异.

![](https://pic2.zhimg.com/v2-a1545ee03a7a3cdc57052b233ac0af15_1440w.jpg)

### 3.2.2 提高稀疏性加速训练

为了加速训练过程, 作者使用了两种方法:

1. 提出了类似 [FlashAttention](https://zhida.zhihu.com/search?content_id=241774464&content_type=Article&match_order=1&q=FlashAttention&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODM0OTYyOTEsInEiOiJGbGFzaEF0dGVudGlvbiIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI0MTc3NDQ2NCwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.JWtKtKAQ8wlFwehSmd9Vv0iytJs5ku_KHjqNyMwry9s&zhida_source=entity) 的高效的注意力实现
2. 使用随机长度Stochastic Length方法, 提高用户历史序列的稀疏性。随机长度方法的实现如下:

$x_{0} , \ldots , x_{n_{i} - 1} & \text{if } n_{i} \leq N^{\alpha / 2} \\ \Gamma \left(\right. n_{i} , N^{\alpha / 2} \left.\right) & \text{if } n_{i} > N^{\alpha / 2} , \text{ w}/\text{ probability } 1 - N^{\alpha} / n_{i}^{2} \\ x_{0} , \ldots , x_{n_{i} - 1} & \text{if } n_{i} > N^{\alpha / 2} , \text{ w}/\text{ probability } N^{\alpha} / n_{i}^{2}$

这里 $\alpha \in \left(\right. 1 , 2 \left]\right.$, 通过引入这种随机长度方法, 在牺牲一定模型精度的情况下, 极大提高了训练速度.

### 3.2.3 最小化Activation内存使用

一个模型的Memory Usage主要来自三部分, Model Memory、Optimizer Memory、Activation Memory, 其中Activation Memory是大头, 为了最小化Activation内存使用使用, 作者做了大量的简化和计算上的优化, 主要包括:

1. HSTU将注意力之外的线性层的数量从6减少到2, 并使用了elementwise的门控机制
2. HSTU做了完全融合的设计(原文:HSTU aggressively fuses computations into single operators)
3. 使用rowwise AdamW优化器, 并将optimizer states常驻内存

### 3.2.4 推理侧优化

作者通过新的推理算法 [M-FALCON](https://zhida.zhihu.com/search?content_id=241774464&content_type=Article&match_order=1&q=M-FALCON&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODM0OTYyOTEsInEiOiJNLUZBTENPTiIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI0MTc3NDQ2NCwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.SgrzxfbErPRX856YfHI7LtLv3zRDl4H9aZXPWIr3J4Y&zhida_source=entity) 达成了推理侧700倍加速（285倍复杂模型，2.48x推理QPS)。这里工程细节过多, 没有过多提及.

## 4\. 实验

作者在Meta的真实业务上落地应用了上述生成式推荐方案, 效果有些好的难以置信.

![](https://pic3.zhimg.com/v2-c486b5c22339a815dd815cf0425855b6_1440w.jpg)

据作者自述: 通过新架构HSTU+训练算法GR，我们模型总计算量达到了1000x级的提升，第一次达到GPT-3 175b/LLaMa-2 70b等LLM训练算力，且第一次我们在推荐模态观测到了语言模态的scaling law；

传统测试集MovieLens Amazon Reviews等相对经典SASRec提升20.3%-65.8% NDCG@10；实际中多产品界面上线单特定ranking界面提升12.4%；

## 5\. 相关材料

编辑于 2025-01-06 12:19・广东[推荐系统](https://www.zhihu.com/topic/19563024)[大模型](https://www.zhihu.com/topic/25402720)[推荐算法](https://www.zhihu.com/topic/19580544)