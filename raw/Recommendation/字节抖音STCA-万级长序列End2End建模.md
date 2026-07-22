---
title: "字节抖音STCA:万级长序列End2End建模"
source: "https://zhuanlan.zhihu.com/p/1983454627761508924"
author:
  - "[[州懂公众号:州懂学习笔记 & 某大厂算法工程师]]"
published:
created: 2026-07-09
description: "大家好, 我是州懂, 今天分享一篇抖音万级长序列的End2End建模方案 标题: Make It Long, Keep It Fast: End-to-End 10k-Sequence Modeling at Billion Scale on Douyin 地址: https://arxiv.org/pdf/2511.06077 公…"
tags:
  - "clippings"
---
目录

收起

1、前言

2\. 方法

2.1 Stacked Target Cross Attention(STCA)

2.1.1 输入信息Token化

2.1.2 堆叠的多头Target Attention

2.1.3 训练和预测

2.1.4 注意力机制性能优化

2.2 Request Level Batching (RLB)

2.3 序列长度外推能力

1) 随机长度采样

2) 采样策略

3) Batch-Level负载均衡

4)Ragged Target Attention

3\. 实验部分

3.1 整体效果

3.2 消融实验

3.3 Scaling Law

3.4 参数分析

3.5 线上AB实验

大家好, 我是州懂, 今天分享一篇抖音万级长序列的 [End2End建模](https://zhida.zhihu.com/search?content_id=267630278&content_type=Article&match_order=1&q=End2End%E5%BB%BA%E6%A8%A1&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODM3NTYyMzYsInEiOiJFbmQyRW5k5bu65qihIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjY3NjMwMjc4LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.War2mmzZgbOnvApwbNtFKzH7nuJSN-LiiOJlsNvU6xM&zhida_source=entity) 方案

> 标题: Make It Long, Keep It Fast: End-to-End 10k-Sequence Modeling at Billion Scale on Douyin  
> 地址: [arxiv.org/pdf/2511.0607](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2511.06077)  
> 公司: [字节抖音](https://zhida.zhihu.com/search?content_id=267630278&content_type=Article&match_order=1&q=%E5%AD%97%E8%8A%82%E6%8A%96%E9%9F%B3&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODM3NTYyMzYsInEiOiLlrZfoioLmipbpn7MiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNjc2MzAyNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.wsSQRn94nEgToFi-9tEQsiUj3smV6jvZUBc9083Rq40&zhida_source=entity)

## 1、前言

更快&更省&更长是长序列建模的长期命题, 字节这篇论文主要探讨如何在训练和推理成本可控的约束下，将用户历史行为序列建模长度从几百扩展到10K条:

- **更快(STCA):** 探索更高效&时间复杂度更好的注意力计算方式
- **更省(RLB):** 基于 [Request-wise样本组织方式](https://zhida.zhihu.com/search?content_id=267630278&content_type=Article&match_order=1&q=Request-wise%E6%A0%B7%E6%9C%AC%E7%BB%84%E7%BB%87%E6%96%B9%E5%BC%8F&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODM3NTYyMzYsInEiOiJSZXF1ZXN0LXdpc2XmoLfmnKznu4Tnu4fmlrnlvI8iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNjc2MzAyNzgsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.aFxUlG0tkRt-4fWuvgP7FbcAej86ENlslBkdsiBDQDc&zhida_source=entity), 最大程度复用User侧的计算
- **更长(EAT):** 训练时使用紧凑的"短"序列, 但推理时可外推到更长的序列

针对上述问题, 字节所提方法的整体框架如下图所示:

![](https://picx.zhimg.com/v2-26cc56ef1aa1471849c8010c4e9acee9_1440w.jpg)

## 2\. 方法

下面分别介绍前面框架图所提的各个模块:

### 2.1 Stacked Target Cross Attention(STCA)

记用户行为序列(长度为)为 $\mathcal{H} = \left{\right. \left(\right. v_{i} , a_{i} \left.\right) \left.\right}_{i = 1}^{L}$, 其中表示视频, $a_{i}$ 表示对应的行为类型。

### 2.1.1 输入信息Token化

论文没有阐述行为序列和Target Item具体是如何做Token化处理得到和 $\mathbf{x}_{t} \in \mathbb{R}^{d}$ 的。关于输入信息Token化, 业界常见的处理方式(参考美团MTGR, 字节OneTrans等):

- 对于User Profile, 美团MTGR将每个特征都表示为一个Token, 字节RankMixer则是按特征分组后按组分别过MLP再拆分成若干个Token(Group-wise方式), 字节OneTrans则是先Concat起来过MLP后再做Split(Auto-Split方式)
- 对于用户行为序列，这方面大家都差不多。美团MTGR是将每一个具体行为对应的item ID以及多个side info的Embedding表征拼接在一起再通过非线形映射组装成一个Token。做的再细一点的比如字节的OneTrans, 它会按行为类型来过不同的MLP。
- 对于Target Item, 这方面大家方案也差不多, 比如美团MTGR将item ID、对应的side info、交叉特征、时空等Context信息拼接在一起组成一Token, 或者分开来使用若干个Token表示(比如这篇论文)。

字节这篇文章比较特别的点是, 对Target Item和历史行为序列, 除了常规的LayerNorm处理外, 还额外过SwiGLUFFN(即SwiGLU + linear projection)做预处理, 论文在后面的实验中消融验证了加SwiGLU的好处(FFN $\rightarrow$ SwiGLU使 $\Delta$ AUC涨了+0.11%)

$S w i G L U F F N \left(\right. \mathbf{x} \left.\right) = \left(\right. \left(\right. \mathbf{x} W_{u} \left.\right) \bigodot \left(\right. \mathbf{x} W_{v} \bigodot s i g m o i d \left(\right. \mathbf{x} W_{v} \left.\right) \left.\right) \left.\right) W_{o}$

- 行为序列

$\overset{\sim}{X}^{\left(\right. i \left.\right)} = L N \left(\right. \left(S w i G L U F F N\right)^{\left(\right. i \left.\right)} \left(\right. X \left.\right) \left.\right) \in \mathbb{R}^{L \times d}$

- Target Item

$\mathbf{q}^{\left(\right. 1 \left.\right)} = L N \left(\right. \left(S w i G L U F F N\right)^{\left(\right. 1 \left.\right)} \left(\right. \mathbf{x}_{t} \left.\right) \left.\right) \in \mathbb{R}^{d}$

其中, $W_{u} , W_{v} \in \mathbb{R}^{d \times r d}$, $W_{o} \in \mathbb{R}^{r d \times d}$, 这里的为缩放系数。

### 2.1.2 堆叠的多头Target Attention

对行为序列和Target Item做了Token化后, 注意力的建模方式是比较常规的多头Target Attention了,再堆叠多层, 具体地:

- 拆分成个head, $d_{h} = d / h$:

$W_{Q}^{\left(\right. i , r \left.\right)} , W_{K}^{\left(\right. i , r \left.\right)} , W_{V}^{\left(\right. i , r \left.\right)} \in \mathbb{R}^{d \times d_{h}} , W_{O}^{\left(\right. i \left.\right)} \in \mathbb{R}^{d \times d}$

- 计算注意力得分:

$\alpha^{\left(\right. i , r \left.\right)} = s o f t m a x \left(\right. \frac{\mathbf{q}^{\left(\right. i \left.\right)} W_{Q}^{\left(\right. i , r \left.\right)} \left(\right. \overset{\sim}{X}^{\left(\right. i \left.\right)} W_{K}^{\left(\right. i , r \left.\right)} \left.\right)^{\top}}{\sqrt{d_{h}}} \left.\right) \in \mathbb{R}^{1 \times L}$

- 计算Target Attention结果

$\mathbf{o}^{\left(\right. i , r \left.\right)} = \alpha^{\left(\right. i , r \left.\right)} \left(\right. \overset{\sim}{X}^{\left(\right. i \left.\right)} W_{V}^{\left(\right. i , r \left.\right)} \left.\right) \in \mathbb{R}^{1 \times d_{h}}$ $\mathbf{o}^{\left(\right. i \left.\right)} = \left[\right. \mathbf{o}^{\left(\right. i , 1 \left.\right)} \parallel \hdots \parallel \mathbf{o}^{\left(\right. i , h \left.\right)} \left]\right. W_{O}^{\left(\right. i \left.\right)} \in \mathbb{R}^{d}$

对比Self-Attention的时间复杂度, Target Attention的时间复杂度为线性的。

为了进一步提升序列建模能力, 这里的多头Target Attention还堆叠了多层(记为层)。

具体地, 将前层的target attention和当前的Target Item一起Concat起来后, 再过一个MLP和SwiGLUFFN模块, 得到第层所使用的Query

$\mathbf{q}^{\left(\right. i + 1 \left.\right)} = \left(S w i G L U F F N\right)^{\left(\right. i + 1 \left.\right)} \left(\right. \left[\right. \mathbf{o}^{\left(\right. 1 \left.\right)} \parallel \hdots \parallel \mathbf{o}^{\left(\right. i \left.\right)} \parallel \mathbf{x}_{t} \left]\right. W_{C}^{\left(\right. i + 1 \left.\right)} \left.\right)$

其中, $W_{C}^{\left(\right. i + 1 \left.\right)} \in \mathbb{R}^{\left(\right. i + 1 \left.\right) d \times d}$ 。

### 2.1.3 训练和预测

经过前面层的Target Attention后, 能拿到个结果, 再将它们与target item Concat起来, 继续过SwiGLUFFN

$\mathbf{z} = S w i G L U F F N \left(\right. \left[\right. \mathbf{o}^{\left(\right. 1 \left.\right)} \parallel \hdots \parallel \mathbf{o}^{\left(\right. M \left.\right)} \parallel \mathbf{x}_{t} \left]\right. W_{Z} \left.\right) , W_{Z} \in \mathbb{R}^{\left(\right. M + 1 \left.\right) d \times d}$

然后, 将它们与user-side tokens $\left{\right. \mathbf{u}_{k} \left.\right}_{k = 1}^{K}$ (比如用户画像)和其它candidate-side tokens $\left{\right. \mathbf{c}_{ℓ} \left.\right}_{ℓ = 1}^{C}$ (如多模态特征等)也一起Concat起来

$\mathcal{X}_{m i x} = c o n c a t \left(\right. \mathbf{z} , \left{\right. \mathbf{u}_{k} \left.\right}_{k = 1}^{K} , \left{\right. \mathbf{c}_{ℓ} \left.\right}_{ℓ = 1}^{C} \left.\right)$

再给到RankMixer去做预测:

$\mathbf{h} = R a n k M i x e r \left(\right. \mathcal{X}_{m i x} ; \Theta \left.\right) , \hat{y} = s i g m o i d \left(\right. \mathbf{w}^{\top} \mathbf{h} + b \left.\right) .$

损失函数使用常规的交叉熵损失:

$\mathcal{L}_{B C E} = - y log ⁡ \hat{y} - \left(\right. 1 - y \left.\right) log ⁡ \left(\right. 1 - \hat{y} \left.\right)$

### 2.1.4 注意力机制性能优化

标准形式的Target Attention注意力机制($X \in \mathbb{R}^{L \times d}$, $q \in \mathbb{R}^{1 \times d}$, $d_{h} = \frac{d}{h}$) 计算如下:

$A t t n \left(\right. q , X \left.\right) = s o f t m a x \left(\right. \frac{\left(\right. q W_{Q} \left.\right) \left(\right. X W_{K} \left.\right)^{\top}}{\sqrt{d_{h}}} \left.\right) \cdot \left(\right. X W_{V} \left.\right)$

其中, $W_{Q} , W_{K} , W_{V} \in \mathbb{R}^{d \times d_{h}}$ 。

先拆解下标准形式的Target Attention的复杂度

- $K = X W_{K} \in \mathbb{R}^{L \times d_{h}}$ 和, 标准形式的Target Attention需要做两次Project, 需要两次的计算量, 从显存上需要KV这大小的中间张量。
- $q W_{Q} \in \mathbb{R}^{1 \times d_{h}}$, 时间复杂度为
- $s o f t m a x \left(\right. \frac{\left(\right. q W_{Q} \left.\right) \left(\right. X W_{K} \left.\right)^{\top}}{\sqrt{d_{h}}} \left.\right) \in \mathbb{R}^{1 \times L}$, 时间复杂度为
- $A t t n \left(\right. q , X \left.\right)$, 时间复杂度为

整体计算的时间复杂度为, 空间复杂度为, 计算开销上, 主要就是Project过程。由于矩阵乘法在计算Flops时同时需要一次乘法和一次加法, 2次Project总共需要 FLOPs

上述Target Attention的公式可以改写成:

$A t t n \left(\right. q , X \left.\right) = \left(\right. \underset{\alpha \in \mathbb{R}^{1 \times L}}{\underbrace{s o f t m a x \left(\right. \frac{\left(\right. \left(\right. q W_{Q} \left.\right) W_{K}^{\top} \left.\right) X^{\top}}{\sqrt{d_{h}}} \left.\right)}} X \left.\right) W_{V}$

就单次的cross attention计算上, 分成:

- $u = \left(\right. q W_{Q} \left.\right) W_{K}^{\top} \in \mathbb{R}^{1 \times d}$, 计算量是,时间复杂度是
- $\alpha = s o f t m a x \left(\right. \frac{u X^{\top}}{\sqrt{d_{h}}} \left.\right) \in \mathbb{R}^{1 \times L}$, 时间复杂度是
- $A t t n \left(\right. q , X \left.\right) = \left(\right. \alpha X \left.\right) W_{V}$, 时间复杂度是+

通过上面计算顺序的改写, 时间复杂度从+ $O \left(\right. d d_{h} \left.\right)$, 以 ($d_{h} = 32$)为例, FLOPs降低为原来的 $\frac{1}{64}$

### 2.2 Request Level Batching (RLB)

生成式推荐浪潮下, 按用户维度/请求维度来组织样本基本上是标配了。字节这里也是按请求维度来组样本, 将同一请求的个Target Item样本merge成一条样本, 以共享同一上下文的用户信息编码。在损失函数上:

- Point-wise:

$L_{B C E} \left(\right. \hat{y} \left(\right. u , v_{k} \left.\right) , y_{k} \left.\right)$

- Request-wise

$\mathcal{L}_{u} = \frac{1}{m} \sum_{k = 1}^{m} \mathcal{L}_{B C E} \left(\right. \hat{y} \left(\right. u , v_{k} \left.\right) , y_{k} \left.\right) , \mathcal{L} = \frac{1}{\left|\right. \mathcal{U} \left|\right.} \underset{u \in \mathcal{U}}{\sum} \mathcal{L}_{u}$

字节这里也详细提了下Request-wise构造样本的各种好处:

- **省存储:** 同一条请求的用户侧特征都复用了, 减少了重复性的样本存储成本
- **省带宽/硬件IO:** 一次请求不需要传份user/history, 现在只传1份就可以了
- **省显存:** 像对用户行为序列 $\overset{\sim}{X}^{\left(\right. i \left.\right)}$ 做处理的一些中间KV变量是个target item共享的, 减少了显存的占用
- **提Kernel效率:** 将多个Target Item拼成大矩阵, 其效率远比单独算个小矩阵的效率更高
- **省PS/allreduce通信:** 分布式训练下, 用户侧的梯度通信量级大大减少

### 2.3 序列长度外推能力

虽然前面STCA使得序列建模的时间复杂度是线性的, 但如果直接统一使用10k长度(不足长度用padding补齐)来训练, 训练开销会大很多。

这里的长度外推能力是指, 希望使用的平均长度来训练, 但推理时可以覆盖的长度, 提高训练效率。

用户的序列长度分布是很不均匀的, 论文这里训练策略的思路是, 将不同的用户序列长度通过采样和Batch内调整, 让它们的序列长度看起来变得更紧凑一些, 去掉"padding"补齐操作, 如下图所示:

![](https://pic3.zhimg.com/v2-e89dba9e59b269e30d57c62b8d8a76c6_1440w.jpg)

### 1) 随机长度采样

- 每条用户序列在训练时基于给定的分布采样, 来随机截断为长度
- 具体地,使用Beta分布来控制采样比例, $L_{\text{train}}^{\text{raw}} = L_{\text{train}}^{min} + s \cdot \left(\right. L_{\text{train}}^{max} - L_{\text{train}}^{min} \left.\right)$
- $L_{\text{train}}^{\text{raw}}$ 会四舍五入成8的倍数, 以对齐硬件加速
- 后面实验表示,通过调整参数 $\alpha$ 使beta分布越接近U型越好

### 2) 采样策略

- 训练时, 在确定截断长度后，保留最近的行为(temporal suffix)
- 推理时, 则截断到最近的长度,如10K
- 从效果来看：保留最近行为 > 随机采样 > 保留最早行为

### 3) Batch-Level负载均衡

为避免变长序列导致GPU负载不均，采用：

- 全局长度分配：控制每个batch的总token数接近
- 序列压缩：将序列长度过长的多余token分配给较短序列，消除padding；

### 4)Ragged Target Attention

- 使用Ragged Target Attention处理可变长度的序列
- 不同用户序列计算注意力时会做隔离, 互不干扰

## 3\. 实验部分

### 3.1 整体效果

在抖音的离线数据集上验证效果, 其中, 作为对比的baseline还额外带有TWIN的两阶段检索子序列。

![](https://pic1.zhimg.com/v2-eb3f6720b15b2cd75a1605d7c6a147c6_1440w.jpg)

### 3.2 消融实验

序列FFN的深度加深的收益最大(+0.18%), 其次是FFN切换成SwiGLU(+0.11%)

![](https://pic1.zhimg.com/v2-3d9af3c6fee110fa38b98f42d66b784c_1440w.jpg)

### 3.3 Scaling Law

论文所提方法的Scaling Law更好

![](https://pic4.zhimg.com/v2-c2669bb8f4d4831fe2e15c873dd775b3_1440w.jpg)

### 3.4 参数分析

- 外推到10K长度收益更大
![](https://pica.zhimg.com/v2-f9f2e09edc10ff5b77afd8d49cb042b8_1440w.jpg)

- 平均训练长度到2K性价比最高
![](https://pica.zhimg.com/v2-6e646cfaf1f614c86d45b546bcdeea2c_1440w.jpg)

- Beta分布呈U型最好
![](https://pic3.zhimg.com/v2-bfd8546dca3da608349b84413a3c107c_1440w.jpg)

### 3.5 线上AB实验

线上ab实验的指标提升挺大的

![](https://pic3.zhimg.com/v2-4314a93211aae28bd0dc95dcb1b9065c_1440w.jpg)

编辑于 2025-12-14 08:36・广东[抖音](https://www.zhihu.com/topic/20088356)[推荐系统](https://www.zhihu.com/topic/19563024)[字节跳动（ByteDance）](https://www.zhihu.com/topic/20168793)[注册领2000积分，体验腾讯AI专家，搞定繁琐Excel与PPT](https://www.codebuddy.cn/work?fromSource=gwzcw.15370171.15370171.15370171&utm_medium=cpc&utm_id=gwzcw.15370171.15370171.15370171&spu=biz%3D0%26ci%3D3775992%26si%3D44aaf2d7-35f0-4bb9-9850-ab56a7961c54%26ts%3D1783583438%26zid%3D1629)

[

WorkBuddy利用多智能体技术，模拟专家团队分工协作。无论是复杂的财务报表清洗，还是深度的竞品调研，都能自主...

](https://www.codebuddy.cn/work?fromSource=gwzcw.15370171.15370171.15370171&utm_medium=cpc&utm_id=gwzcw.15370171.15370171.15370171&spu=biz%3D0%26ci%3D3775992%26si%3D44aaf2d7-35f0-4bb9-9850-ab56a7961c54%26ts%3D1783583438%26zid%3D1629)