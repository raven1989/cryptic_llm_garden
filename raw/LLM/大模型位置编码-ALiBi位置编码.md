---
title: "大模型位置编码-ALiBi位置编码"
source: "https://zhuanlan.zhihu.com/p/656684326"
author:
  - "[[老苏的AI茶馆大模型解决方案架构师，有行业经验的技术专家]]"
published:
created: 2026-04-21
description: "1、背景上一节我们介绍了一下旋转位置编码，这一次我们再来介绍另外一种大模型常用的位置编码技术， 《Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation》简称：ALiBi位…"
tags:
  - "clippings"
---
## 1、背景

上一节我们介绍了一下 [旋转位置编码](https://zhida.zhihu.com/search?content_id=234058536&content_type=Article&match_order=1&q=%E6%97%8B%E8%BD%AC%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzY5NTUxNTIsInEiOiLml4vovazkvY3nva7nvJbnoIEiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyMzQwNTg1MzYsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.G-d37x1QJZJ8p3Q4Wx4Nm6YbI8hu93pdzK_KEMx_QKM&zhida_source=entity) ，这一次我们再来介绍另外一种大模型常用的位置编码技术， 《Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation》简称： [ALiBi位置编码](https://zhida.zhihu.com/search?content_id=234058536&content_type=Article&match_order=1&q=ALiBi%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzY5NTUxNTIsInEiOiJBTGlCaeS9jee9rue8lueggSIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjIzNDA1ODUzNiwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.sy3h4NcXwFejYNkLkZoj0__0Rv6ZyrD6lXFKWpCQRqs&zhida_source=entity)

Paper 地址： **[arxiv.org/abs/2108.1240](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2108.12409)**

自从Transform被提出以来，一个基本问题还没有被解决，一个模型如何在推断时对训练期间没有见过的更长序列进行外推。众所周知，Bert支持的最长句子长度是512，那为什么Bert只能支持512的句子长度呢？

我们看一下BertEmbeddings的初始化，我们可以看到position\_ids，被初始化成0-511，这个也就是BERT处理文本最大长度是512的原因，这里Bert使用的是绝对位置编码。

```
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
```

为了解决长度外推的问题，作者提出了一种更简单、更有效的位置方法，即具有线性偏置的注意力(ALiBi)。ALiBi不向词嵌入添加位置嵌入，相反，它通过与距离成比例的惩罚来偏置query-key注意力分数。

接下来我们再来聊一下长度外推，下面的内容取自苏神的博客，能更好的帮助我们理解长度外推的这个概念。 **[spaces.ac.cn/archives/9](https://link.zhihu.com/?target=https%3A//spaces.ac.cn/archives/9431)**

> 长度外推性是一个训练和预测的长度不一致的问题。 具体来说，不一致的地方有两点： 1、预测的时候用到了没训练过的位置编码（不管绝对还是相对）； 2、预测的时候注意力机制所处理的token数量远超训练时的数量。 第1点可能大家都容易理解，没训练过的就没法保证能处理好，这是DL中很现实的现象，哪怕是 **[Sinusoidal](https://link.zhihu.com/?target=https%3A//spaces.ac.cn/archives/8231)** 或 **[RoPE](https://link.zhihu.com/?target=https%3A//spaces.ac.cn/archives/8265)** 这种函数式位置编码也是如此。关于第2点，可能读者会有些迷惑，Attention理论上不就是可以处理任意长度的序列吗？训练和预测长度不一致影响什么呢？答案是熵，我们在 **[《从熵不变性看Attention的Scale操作》](https://link.zhihu.com/?target=https%3A//spaces.ac.cn/archives/8823)** 也已经分析过这个问题，越多的token去平均注意力，意味着最后的分布相对来说越“均匀”（熵更大），即注意力越分散；而训练长度短，则意味着注意力的熵更低，注意力越集中，这也是一种训练和预测的差异性，也会影响效果  

## 2、算法介绍

![](https://pic4.zhimg.com/v2-00ccc4e8663b1630e0d2194520ace49f_1440w.jpg)

image.png

我们看上图，左边的是模型在512数据集训练的模型，右边是在1024数据集上训练的模型，横坐标是推理是输入的句子长度，纵坐标的困惑度，我们的目标是，困惑度越小越好，通过观察上图我们可以看到， [Sinusoidal编码](https://zhida.zhihu.com/search?content_id=234058536&content_type=Article&match_order=1&q=Sinusoidal%E7%BC%96%E7%A0%81&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzY5NTUxNTIsInEiOiJTaW51c29pZGFs57yW56CBIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjM0MDU4NTM2LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.8LOTEcHCt17O42EYNDL_q1-UdSHaX5kVQEGzR5dilvs&zhida_source=entity) ， [Rotary编码](https://zhida.zhihu.com/search?content_id=234058536&content_type=Article&match_order=1&q=Rotary%E7%BC%96%E7%A0%81&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzY5NTUxNTIsInEiOiJSb3RhcnnnvJbnoIEiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyMzQwNTg1MzYsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.NWvFJa9aRFEtqk1MT1Mk9zQK5wCXca48500wWcA_QTI&zhida_source=entity) ， [T5 Bias](https://zhida.zhihu.com/search?content_id=234058536&content_type=Article&match_order=1&q=T5+Bias&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzY5NTUxNTIsInEiOiJUNSBCaWFzIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjM0MDU4NTM2LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.fHHx-jhEZwNx_Bzr1MVPadXsNgnQVsvWr3AfmsTNZ-Y&zhida_source=entity) 这三种位置编码，当输入特别长的时候，困惑度就会飙升，但是ALiBi编码会保持一个平稳的水平，说明ALiBi编码有很好的长度外推能力。

## 2-1、ALiBi算法介绍

ALiBi 的方法也是较为粗暴，是直接作用在attention score中，给 attention score 加上一个预设好的偏置矩阵，相当于和相对位置差 1 就加上一个 -1 的偏置。其实相当于假设两个 token 距离越远那么相互贡献也就越低。

$S o f t a m x \left(\right. q_{i} K^{T} + m \left[\right. - \left(\right. i - 1 \left.\right) , . . . , - 2 , - 1 , 0 \left]\right. \left.\right)$

我们重点看一下上面的这个公式，左边第一项是注意力的分数，跟self-Attention中一致 第二项是一个相对距离的矩阵，例如，他们直接的距离为0，所以对应的位置是0，下面有一张图，会更清楚的展示，相对距离的计算。

这里在举一个例子， 比如， $q_{2} , k_{1}$ 是相对位置为的索引1减去的索引2，得到1-2=-1， 其中ALiBi 位置编码是不需要通过训练的，给定的预设矩阵中还会乘上m的调节因子，m的设置与attention的头数有关，是2的指数差值。论文中也做了尝试把m作为学习参数，但是并没有获得更好的效果 如果m是8个头的话，我们使用的斜率是几何序列，

$\frac{1}{2^{1}} , \frac{1}{2^{2}} , . . . , \frac{1}{2^{8}}$

如果是16个头的话，使用下列几何序列

$\frac{1}{2^{0.5}} , \frac{1}{2^{1}} , . . . , \frac{1}{2^{8}}$

对于n和head的话，m的取值是 $2^{\frac{- 8}{n}}$

![](https://pic4.zhimg.com/v2-a2b8b3014ca73ced294ca49a35ad17ed_1440w.jpg)

为什么是一个下三角矩阵，因为我们研究的是，autoregressive language modeling，所以这就是上三角被MASK的原因，我们不关注未来，只关注过去，这个编码只应用在Query和Key中，不会应用到Value中

## 3、代码实现

这里我们以Baichuan-13-Base为例来进行代码讲解说明

**[huggingface.co/baichuan](https://link.zhihu.com/?target=https%3A//huggingface.co/baichuan-inc/Baichuan-13B-Base/blob/main/modeling_baichuan.py)**

下面函数的作用是生成缩放因子m的序列，需要注意的是论文中，只处理个head，这个函数也能处理这个head的个数不是2的幂次方的情况

```
def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return _get_interleave_power_of_2(closest_power_of_2) + \
               _get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

_get_interleave(8)

[0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]

_get_interleave(4)

[0.25, 0.0625, 0.015625, 0.00390625]
```

我们从 [BaichuanModel](https://zhida.zhihu.com/search?content_id=234058536&content_type=Article&match_order=1&q=BaichuanModel&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzY5NTUxNTIsInEiOiJCYWljaHVhbk1vZGVsIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjM0MDU4NTM2LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.Z-rBD7VdUi_qDjsVgfkhxfTT-MMk3aE1NVCqLDW_RWU&zhida_source=entity) 这个类看起，主要的代码有

```
def _fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

def _gen_alibi_mask(n_head, max_pos):
    """used in inference only"""
    slopes = torch.Tensor(_get_interleave(n_head))
    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_pos).unsqueeze(0).unsqueeze(0).expand(
        n_head, -1, -1)
    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = torch.triu(
        _fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1
    )
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask

def _buffered_future_mask(tensor, maxpos, alibi, attn_heads):
    """used in training only"""
    dim = tensor.size(1)
    _future_mask = torch.triu(
        _fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1
    )   
    _future_mask = _future_mask.unsqueeze(0) + alibi
    _future_mask = _future_mask.to(tensor)
    return _future_mask[:tensor.shape[0] * attn_heads, :maxpos, :maxpos]
```

我们看一下\_gen\_alibi\_mask函数，假设我们有4个头，句子的最大长度是8，那么生成的ALiBI矩阵长什么样子呢，我们来看一下，这个下三角矩阵，有同学可能会问，这里的矩阵为什么是一个下三角矩阵，因为这个任务是，Causal Language Modeling。

```
tensor([[[0.0000,   -inf,   -inf,   -inf,   -inf,   -inf,   -inf,   -inf],
         [0.0000, 0.2500,   -inf,   -inf,   -inf,   -inf,   -inf,   -inf],
         [0.0000, 0.2500, 0.5000,   -inf,   -inf,   -inf,   -inf,   -inf],
         [0.0000, 0.2500, 0.5000, 0.7500,   -inf,   -inf,   -inf,   -inf],
         [0.0000, 0.2500, 0.5000, 0.7500, 1.0000,   -inf,   -inf,   -inf],
         [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.2500,   -inf,   -inf],
         [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.2500, 1.5000,   -inf],
         [0.0000, 0.2500, 0.5000, 0.7500, 1.0000, 1.2500, 1.5000, 1.7500]],

        [[0.0000,   -inf,   -inf,   -inf,   -inf,   -inf,   -inf,   -inf],
         [0.0000, 0.0625,   -inf,   -inf,   -inf,   -inf,   -inf,   -inf],
         [0.0000, 0.0625, 0.1250,   -inf,   -inf,   -inf,   -inf,   -inf],
         [0.0000, 0.0625, 0.1250, 0.1875,   -inf,   -inf,   -inf,   -inf],
         [0.0000, 0.0625, 0.1250, 0.1875, 0.2500,   -inf,   -inf,   -inf],
         [0.0000, 0.0625, 0.1250, 0.1875, 0.2500, 0.3125,   -inf,   -inf],
         [0.0000, 0.0625, 0.1250, 0.1875, 0.2500, 0.3125, 0.3750,   -inf],
         [0.0000, 0.0625, 0.1250, 0.1875, 0.2500, 0.3125, 0.3750, 0.4375]],

        [[0.0000,   -inf,   -inf,   -inf,   -inf,   -inf,   -inf,   -inf],
         [0.0000, 0.0156,   -inf,   -inf,   -inf,   -inf,   -inf,   -inf],
         [0.0000, 0.0156, 0.0312,   -inf,   -inf,   -inf,   -inf,   -inf],
         [0.0000, 0.0156, 0.0312, 0.0469,   -inf,   -inf,   -inf,   -inf],
         [0.0000, 0.0156, 0.0312, 0.0469, 0.0625,   -inf,   -inf,   -inf],
         [0.0000, 0.0156, 0.0312, 0.0469, 0.0625, 0.0781,   -inf,   -inf],
         [0.0000, 0.0156, 0.0312, 0.0469, 0.0625, 0.0781, 0.0938,   -inf],
         [0.0000, 0.0156, 0.0312, 0.0469, 0.0625, 0.0781, 0.0938, 0.1094]],

        [[0.0000,   -inf,   -inf,   -inf,   -inf,   -inf,   -inf,   -inf],
         [0.0000, 0.0039,   -inf,   -inf,   -inf,   -inf,   -inf,   -inf],
         [0.0000, 0.0039, 0.0078,   -inf,   -inf,   -inf,   -inf,   -inf],
         [0.0000, 0.0039, 0.0078, 0.0117,   -inf,   -inf,   -inf,   -inf],
         [0.0000, 0.0039, 0.0078, 0.0117, 0.0156,   -inf,   -inf,   -inf],
         [0.0000, 0.0039, 0.0078, 0.0117, 0.0156, 0.0195,   -inf,   -inf],
         [0.0000, 0.0039, 0.0078, 0.0117, 0.0156, 0.0195, 0.0234,   -inf],
         [0.0000, 0.0039, 0.0078, 0.0117, 0.0156, 0.0195, 0.0234, 0.0273]]])

def get_alibi_mask(self, tensor, seq_length_with_past):
        if self.training:
            slopes = torch.Tensor(_get_interleave(self.n_head))
            alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(seq_length_with_past).unsqueeze(0).unsqueeze(0).expand(
                self.n_head,
                -1, -1) 
            alibi = alibi.view(self.n_head, 1, seq_length_with_past)
            mask = _buffered_future_mask(tensor, seq_length_with_past, alibi, self.n_head)
        else:
            if self.first_run:
                self.first_run = False
                self.register_buffer("future_mask", _gen_alibi_mask(self.n_head, self.max_cache_pos).to(tensor), persistent=False)
            if seq_length_with_past > self.max_cache_pos:
                self.max_cache_pos = seq_length_with_past
                self.register_buffer("future_mask", _gen_alibi_mask(self.n_head, self.max_cache_pos).to(tensor), persistent=False)
            mask = self.future_mask[:self.n_head, :seq_length_with_past, :seq_length_with_past] 
        return mask

def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot provide both input_ids and inputs_embeds simultaneously")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You need to provide input_ids or inputs_embeds")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        seq_length_with_past = seq_length

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.training:
            if self.alibi_mask is None or self.alibi_mask.shape[-1] != seq_length_with_past:
                self.alibi_mask = self.get_alibi_mask(inputs_embeds, seq_length_with_past)
            alibi_mask = self.alibi_mask
        else:
            alibi_mask = self.get_alibi_mask(inputs_embeds, seq_length_with_past)

        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                expanded_mask = attention_mask.to(alibi_mask.dtype)
                expanded_mask = torch.tril(torch.gt(expanded_mask[:, :, None] * expanded_mask[:, None, :], 0)
                                ) * torch.eq(expanded_mask[:, :, None] - expanded_mask[:, None, :], 0)
            else:
                expanded_mask = attention_mask 
            bsz = inputs_embeds.size(0)
            src_len, tgt_len = alibi_mask.size()[-2:]
            expanded_mask = expanded_mask.unsqueeze(1).expand(bsz, 1, src_len, tgt_len).to(alibi_mask.dtype)
            inverted_mask = 1.0 - expanded_mask
            inverted_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(alibi_mask.dtype).min)
            attention_mask = inverted_mask + alibi_mask.unsqueeze(0)
        else:
            attention_mask = alibi_mask
```

最终我们看到，如果attention\_mask 是None的话，alibi\_mask的矩阵就是attention\_mask矩阵

在Attention计算的时候，attention\_mask与attn\_weights进行相加

```
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

if attention_mask is not None:
    if q_len == 1: # inference with cache
        if len(attention_mask.size()) == 4:
            attention_mask = attention_mask[:, :, -1:, :]   
        else:
            attention_mask = attention_mask[:, -1:, :]    
    attn_weights = attn_weights + attention_mask
    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
```

我们通过阅读相关的代码，发现论文中的实现与code中实现还是有一点区别，针对ALiBi的编码我们就到这里，接下来我们会继续探索关于位置编码的相关应用。

## 4、参考资料

**[zhuanlan.zhihu.com/p/63](https://zhuanlan.zhihu.com/p/632780188)**

**[spaces.ac.cn/archives/9](https://link.zhihu.com/?target=https%3A//spaces.ac.cn/archives/9431)**

**[spaces.ac.cn/archives/9](https://link.zhihu.com/?target=https%3A//spaces.ac.cn/archives/9675/comment-page-2%3FreplyTo%3D22360)**

**[kexue.fm/archives/9431/](https://link.zhihu.com/?target=https%3A//kexue.fm/archives/9431/comment-page-1)**

**[medium.com/@pajakamy/al](https://link.zhihu.com/?target=https%3A//medium.com/%40pajakamy/alibi-attention-with-linear-biases-942abe042e9f)**

发布于 2023-09-16 21:01・北京[ChatGРТ](https://www.zhihu.com/topic/27042831)[大模型](https://www.zhihu.com/topic/25402720)[LLM（大型语言模型）](https://www.zhihu.com/topic/26797383)