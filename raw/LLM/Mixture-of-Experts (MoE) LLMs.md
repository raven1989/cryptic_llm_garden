---
title: "Mixture-of-Experts (MoE) LLMs"
source: "https://cameronrwolfe.substack.com/p/moe-llms"
author:
  - "[[Cameron R. Wolfe]]"
  - "[[Ph.D.]]"
published: 2025-01-27
created: 2026-05-19
description: "Understanding models like DeepSeek, Grok, and Mixtral from the ground up..."
tags:
  - "clippings"
---
### Understanding models like DeepSeek, Grok, and Mixtral from the ground up...

![](https://substackcdn.com/image/fetch/$s_!kZDt!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1cbbb885-f965-4f56-80fe-2b7e28842237_2254x1258.png)

(from \[2, 5, 14\])

In an area of study that is rapidly changing, the decoder-only transformer architecture has remained one of the few enduring staples in large language model (LLM) research. This architecture has been used since the proposal of the [original GPT model](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) and has remained largely unchanged, aside from minor tweaks to improve efficiency. One of the most meaningful modifications to be explored for this architecture, however, is the Mixture-of-Experts (MoE) layer.

> *“Using an MoE architecture makes it possible to attain better tradeoffs between model quality and inference efficiency than dense models typically achieve.”* - from \[11\]

MoE-based LLMs introduce sparsity to the model’s architecture, allowing us to significantly increase its size— *in terms of the number of total parameters* —without a corresponding increase in compute costs. This modification, which has been successfully adopted by recent models like Grok \[9\] and DeepSeek-v3 \[15\], makes the exploration of extremely large models more tractable and compute efficient. In this overview, we will learn about the fundamentals of MoEs and explore how this idea has been recently applied to create more powerful LLMs.

## Fundamentals of MoEs for LLMs

The MoE-based LLMs that we will study in this overview are based upon the decoder-only transformer architecture. We will not cover the details of this architecture here, but please see [this article](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse) if you are unfamiliar. The decoder-only transformer is comprised of repeated blocks containing normalization (e.g., layer normalization or [RMS layer normalization](https://arxiv.org/abs/1910.07467)), masked multi-headed self-attention or a feed-forward transformation, and a residual connection; see below.

![](https://substackcdn.com/image/fetch/$s_!5BkT!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F379c5d72-9aca-4b50-bd9c-d5ad9454f477_1622x798.png)

The decoder-only transformer architecture

In this section, we will cover the fundamentals of MoEs. This explanation is based upon seminal papers that *i)* proposed the standard MoE layer and *ii)* extended this idea to be used in transformer architectures. The papers are:

1. [The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) \[1\]
2. [Switch Transformers](http://Switch Transformers) \[2\]
3. [Stable and Transferable Mixture-of-Experts (ST-MoE)](https://arxiv.org/abs/2202.08906) \[3\]

For a more detailed break down of these papers and the origins of the MoE architecture, please see the more detailed overview of these ideas below.

![Mixture-of-Experts (MoE): The Birth and Rise of Conditional Computation](https://substackcdn.com/image/fetch/$s_!mjr9!,w_140,h_140,c_fill,f_webp,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa419dd3a-57da-4a9a-a446-31ce4b001a7d_2398x1346.png)

#### [Mixture-of-Experts (MoE): The Birth and Rise of Conditional Computation](https://cameronrwolfe.substack.com/p/conditional-computation-the-birth)

[Cameron R. Wolfe, Ph.D.](https://substack.com/profile/29736521-cameron-r-wolfe-phd)

·

2024年3月18日[Read full story](https://cameronrwolfe.substack.com/p/conditional-computation-the-birth)

**Quick preliminaries.** To understand MoEs and routing algorithms, we must first understand the structure of input for the decoder-transformer (and each of its layers). Of course, LLMs take text as input, but this text undergoes extensive processing before the LLM actually sees it. First, the text is tokenized (shown below)— *or converted into a list of discrete tokens*. These tokens are just words and sub-words. The LLM has a fixed set of tokens that it understands and is trained on, referred to as the model’s “vocabulary”. Vocabulary sizes change between models, but sizes of 64K to 256K total tokens are relatively common.

![](https://substackcdn.com/image/fetch/$s_!_81W!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb8aadf17-3bf6-4b79-9688-b6bfbc5840b1_1830x888.png)

Tokenizing and vectoring text for an LLM

After the text has been converted to tokens, we can vectorize each token in the input. In addition to having a vocabulary, an LLM has a token embedding layer, which stores a (learned [^1]) vector embedding for every token in its vocabulary. We can lookup the vector for each token in this layer, forming an input matrix. If each token embedding is `d` -dimensional and there are `C` total tokens in our input, then the total size of this input matrix is `C` by `d`; see below.

![](https://substackcdn.com/image/fetch/$s_!2lb3!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe2f723f2-056a-4fc0-a3f7-7aa151fe297e_1194x1026.png)

Input matrix of token vectors

Each layer of the transformer— *and each sub-layer within every transformer block* —maintains the size of this input. As a result, the input (and output) for any feed-forward or attention module in the transformer is a matrix of this same size!

#### What are “experts”?

In the decoder-only transformer architecture, the main modification made by an MoE is within the feed-forward component of the transformer block. In the standard architecture, we have a single feed-forward neural network— *usually made up of two feed-forward layers with a non-linear activation in between* —through which every token is passed individually; see below.

![](https://substackcdn.com/image/fetch/$s_!FMd9!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F95d3f6b5-316f-474b-a2cc-243cc22ac7ac_1870x548.png)

An MoE slightly modifies this block structure. Instead of having a single feed-forward network within the feed-forward component of the block, we create several feed-forward networks, *each with their own independent weights*. We refer to each of these networks as an “expert”. For example, an MoE-based LLM may have eight independent experts in each of its feed-forward sub-layers.

![](https://substackcdn.com/image/fetch/$s_!JOdT!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2a99797b-4392-421b-82b0-62932d968217_684x84.png)

The experts within a transformer layer can be defined as shown above. We have `N` experts in a layer, and we can refer to the `i` -th expert using the notation `E_i`.

**Creating an MoE-based transformer.** To create an MoE-based decoder-only transformer architecture, we simply convert the transformer’s feed-forward layers to MoE— *or expert* —layers. Each expert within the MoE layer has an architecture that is identical to the original, feed-forward network from that layer— *we just have several independent copies of the original feed-forward network*; see below.

![](https://substackcdn.com/image/fetch/$s_!tPDR!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8fbb9a24-440d-4d26-8092-b6d72dafb55e_1482x858.png)

Adding experts to a decoder-only transformer block (from \[2\])

However, we need not use experts for every feed-forward layer in the transformer. Most MoE-based LLMs use a stride of `P`, meaning that every `P` -th layer is converted into an expert layer and other layer are left untouched— *these are “interleaved” MoE layers*. This approach can be used to achieve a better balance between the resulting model’s performance and efficiency.

#### Routing Algorithms

One of the primary benefits of MoE-based architectures is their efficiency, but using experts alone does not improve efficiency! In fact, adding more experts to each layer of the model significantly increases the total number parameters— *and the amount of necessary compute* —for the model. To make the architecture more efficient, we must sparsely select the experts that should be used in each layer!

**Selecting experts.** Let’s consider a single token— *represented by a* `d` *\-dimensional token vector*. Our goal is to select a subset of experts (of size `k`) to process this token. In the MoE literature, *we usually say that the token will be “routed” to these experts*. We need an algorithm to compute and optimize this routing operation.

The simplest possible routing algorithm would apply a linear transformation to the token vector, forming a vector of size `N` (i.e., the number of experts). Then, we can apply a [softmax](https://en.wikipedia.org/wiki/Softmax_function) function to form a probability distribution over the set of experts for our token. We can use this distribution to choose experts to which our token should be routed by simply selecting the top- `K` experts in the distribution.

![](https://substackcdn.com/image/fetch/$s_!FZCc!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1189a50c-ad49-4e09-8fca-b800532e101a_1156x856.png)

Computing output of routing mechanism

This routing strategy was used in \[1\], the paper that proposed the sparse MoE layer structure that we use today; see above. However, *such a routing mechanism does not explicitly encourage a balanced selection of experts*. For this reason, the model is likely to converge to a state of repeatedly selecting the same few experts for every token instead of fully and uniformly utilizing its expert layers, as explained below. This phenomenon is commonly referred to as “routing collapse”.

> *“The gating network tends to converge to a state where it always produces large weights for the same few experts. This imbalance is self-reinforcing, as the favored experts are trained more rapidly and thus are selected even more by the gating network.”* - from \[1\]

**Active parameters.** Because we only select a subset of experts to process each token within an MoE layer, there is a concept of “active” parameters in the MoE literature. Put simply, only a small portion of the MoE model’s total parameters— *given by the experts selected at each MoE layer* —are active when processing a given token. As a result, the total computation performed by the MoE is proportional to the number of active parameters, rather than the total number of parameters.

#### Auxiliary Losses and Expert Load Balancing

To encourage a balanced selection of experts during training, we can simply add an additional constraint to the training loss that rewards the model for uniformly leveraging each of its experts. In \[1\], this is done by defining an “importance” score for each expert. The importance score is based upon the probability predicted for each expert by the routing mechanism.

![](https://substackcdn.com/image/fetch/$s_!8EhG!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F74fd12a9-7327-47e3-b7b7-400e801bf5c8_2640x1354.png)

Details of computing the importance loss (from \[1\])

Given a batch of data, we compute importance by taking a sum of the probabilities assigned to each expert across all tokens in the batch; see above. Then, to determine if these probabilities are balanced, we can take the squared [coefficient of variation (CV)](https://en.wikipedia.org/wiki/Coefficient_of_variation) of the expert importance scores. Put simply, *the CV will be a small value if all experts have similar importance scores and vice versa*.

From here, we can simply add the importance loss shown above to our standard language modeling loss to from our new (regularized) training objective. This additional importance loss term helps to ensure that the MoE assigns equal probability to experts throughout the training process.

**Load balancing.** Although the importance loss described above is useful, just because experts are assigned equal importance does not mean that tokens are routed uniformly. For example, experts would have equal importance with:

- A few tokens that assign them very high probability.
- A much larger number of tokens that assign lower probability.

As a result, the number of tokens dispatched to each expert can still be highly non-uniform even when using an importance loss, which can lead to excessive memory usage and generally degraded efficiency for the MoE [^2].

![](https://substackcdn.com/image/fetch/$s_!HmXE!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F644cec45-8dff-491e-9d41-e53ee4b0c7df_1574x764.png)

The expert load balancing loss (from \[2\])

To solve this problem, we can create a single auxiliary loss term (shown above) that captures both expert importance and load balancing, defined as the equal routing of tokens between each of the experts. Such an approach is proposed in \[2\], where authors create a loss that considers two quantities:

1. The fraction of router probability allocated to each expert [^3].
2. The fraction of tokens dispatched to each expert.

If we store both of these quantities in their own `N` -dimensional vectors, we can create a single loss term by taking the [dot product](https://en.wikipedia.org/wiki/Dot_product) [^4] of these two vectors. The resulting loss is minimized when experts receive uniform probability and load balancing, thus capturing both of our goals within a single auxiliary loss term!

![](https://substackcdn.com/image/fetch/$s_!gPGQ!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1790688e-5328-45f2-98c0-717ba6041470_2090x636.png)

The router-z loss (from \[3\])

**Router z-loss.** The auxiliary load balancing loss described above is widely used throughout the MoE literature, but authors in \[3\] propose an extra auxiliary loss term, called the router z-loss, that can further improve training stability. The router z-loss constrains the size of the [logits](https://wandb.ai/amanarora/Written-Reports/reports/Understanding-Logits-Sigmoid-Softmax-and-Cross-Entropy-Loss-in-Deep-Learning--Vmlldzo0NDMzNTU3#logits) — *not the probabilities, so this is before softmax is applied* —predicted by the routing mechanism. Ideally, we do not want these logits to be too big. However, these logits can become very large— *leading to [round-off](https://en.wikipedia.org/wiki/Round-off_error) errors that can destabilize the training process even when using full (*`float32`*) precision* —when passed through the router’s (exponential) softmax function.

> *“The router computes the probability distribution over the experts in float32 precision. However, at the largest scales, we find this is insufficient to yield reliable training.”* - from \[3\]

To encourage the router to predict smaller logits, we can use the loss term shown above. Given that this loss focuses solely upon regularizing the router’s logits and performs no load balancing, we typically use the router z-loss in tandem with the auxiliary load balancing loss proposed in \[2\]. Both of these losses are added on top of the LLM’s standard language modeling loss; see below.

![](https://substackcdn.com/image/fetch/$s_!oxpH!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F726a1e49-0aaa-45dd-a9d0-5386edc2ecc1_2522x288.png)

#### Expert Capacity

The computation performed in an MoE layer is dynamic due to the routing decisions made during both training and inference. However, when we look at most practical implementations of sparse models, we will see that they usually have static batch sizes— *this is a useful trick for improving hardware utilization*.

![](https://substackcdn.com/image/fetch/$s_!vE2b!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F417c5fc8-2524-48e1-a9ef-460b4476d323_1784x1184.png)

(from \[2\])

**Expert capacity.** To formalize the fixed batch size that we set for each expert, we can define the expert capacity. The expert capacity is defined as shown below.

![](https://substackcdn.com/image/fetch/$s_!Ef07!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb4039c04-bf0b-482c-abd8-c76f500f228f_2232x144.png)

The expert capacity defines the maximum number of tokens in a batch that can be sent to each expert. If the number of tokens routed to an expert exceeds the expert capacity, then we just “drop” these extra tokens. More specifically, we perform no computation for these tokens and let their representation flow directly to the next layer via the transformer’s residual connection.

> *“To improve hardware utilization, most implementations of sparse models have static batch sizes for each expert. The expert capacity refers to the number of tokens that can be routed to each expert. If this capacity is exceeded then the overflowed tokens… are passed to the next layer through a residual connection.”* - from \[3\]

Expert capacity is controlled via the capacity factor setting. A capacity factor of one means that tokens are routed in a perfectly balanced manner across experts. Alternatively, setting the capacity factor above one provides extra buffer to accommodate for an imbalance in tokens between experts. However, this comes at a cost (e.g., higher memory usage and lower efficiency).

![](https://substackcdn.com/image/fetch/$s_!iGiy!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe3b8826b-25dd-454b-a686-11c45a20bd50_1104x496.png)

(from \[2\])

**How do we set the capacity factor?** Interestingly, MoE models tend to perform well with relatively low capacity factors \[2, 3\]; see above. However, we need to ensure that the number of dropped tokens is not too large (i.e., this can be done empirically) to avoid any impact to the training run. We can also use different capacity factors for training and inference; e.g., ST-MoE \[3\] uses a capacity factor of 1.25 during training and a capacity factor of 2.0 during evaluation.

#### Computing the Output of an MoE Layer

![](https://substackcdn.com/image/fetch/$s_!jOhk!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Faf751012-739f-4bac-92be-4e22cce61e9d_1368x648.png)

Computing the output of an MoE layer

Once we have the router’s output, we compute the final output as follows:

1. Send the tokens to their selected experts.
2. Compute the output of the experts for these tokens.
3. Take a weighted average of expert outputs, where the weights are simply the probabilities assigned to each expert by the router.

In the equation above, we have formalized the process for computing the output of the MoE layer for a single token. The output for this token is just a weighted average of the outputs from each of its `K` active experts.

**Shared experts** are an idea that has been more recently introduced in the MoE literature \[14, 15\]. The idea is simple:

- We have two groups of experts— *shared experts and routed experts*.
- All tokens are always passed through the shared experts.
- Tokens are passed through the routed experts according to a normal MoE routing mechanism.

This idea of shared experts is depicted below, where we see that routing is only applied to a subset of the experts within an MoE layer. Usually, the number of shared experts must be lower than the number of routed experts— *increasing the number of shared experts degrades the sparsity benefits of the MoE*.

![](https://substackcdn.com/image/fetch/$s_!UlyU!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8ec49e67-8f67-4eea-8759-c27231ffacf5_1212x628.png)

Shared vs. routed experts (from \[14\])

The motivation behind using shared experts is minimizing the amount of redundant information between experts. By having a set of shared experts, *we can allow the network to store shared information within these experts*, rather than having to replicate the same information across several different experts. To compute the output of an MoE layer with shared experts, we simply add the shared experts’ output to the normal, routed output; see below.

![](https://substackcdn.com/image/fetch/$s_!w3mi!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9aceff4e-ff4c-4729-a628-a6ac28bfa600_1198x406.png)

Computing output of an MoE layer with shared experts

#### Putting it all Together: Decoder-only LLMs with MoE Layers

![](https://substackcdn.com/image/fetch/$s_!JUxj!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1e41ca54-f2be-437c-9b89-c45916b245cf_1634x818.png)

A full MoE block in a transformer (from \[2\])

A full depiction of an MoE layer is provided above. In an MoE, we modify the block structure of the standard decoder-only transformer by replacing the feed-forward network with an expert layer. Put simply, this expert layer contains several independent copies of the original feed-forward network. Notably, all of these components within the MoE layer— *the normal layer(s), the experts, and the routing mechanism* —are trained jointly via gradient descent.

For each token, we can choose which experts to be used via a routing mechanism, which is usually implemented via a simple linear transformation of the token vector. Putting this together, the modified block structure for an MoE contains:

- A self-attention layer.
- A residual connection and a normalization operation.
- A routing mechanism that determines the routing of tokens to experts.
- An expert layer with multiple independent feed-forward networks.
- A final add and normalize operation that is applied to the final output of the expert layer for each token.

Aside from the modified block structure, the transformer architecture remains the same. We also only convert every `P` -th block of the transformer to use an MoE layer— *other blocks remain unchanged*. Some MoEs use experts at every layer, but it is common to set `P` to two, four, or even six in practice. This trick can be useful for controlling the total number of parameters consumed by the MoE LLM.

## The Pros and Cons of Using MoEs

Now that we understand the basics of MoEs, we might wonder: *Why would we want to use an MoE instead of a dense model?* The biggest selling point of MoEs is their efficiency, but these models also have notable drawbacks. Let’s quickly go over some of the most important pros and cons of MoEs to be aware of.

**Benefits of MoEs.** [LLMs benefit from scale](https://cameronrwolfe.substack.com/p/llm-scaling-laws) — *larger models and larger datasets lead to better performance*. However, scaling up LLMs comes at a cost! One of the key benefits of MoEs is their ability to circumvent issues with scaling up— *they allow us to increase the size of our model at a fixed computational cost per token*. This way, we can train larger models than would be possible if we restricted ourselves to only dense models. In the language modeling domain, the extra parameters and representational capacity of these sparse models make a big difference.

> *“As LLMs become increasingly prevalent, enhancing their performance without proportionally increasing computational resources is a critical challenge.”* - from \[12\]

The computational benefit of MoEs is (arguably) most impactful during inference. MoE models are large in terms of their number of total parameters, so we need a sufficient number of GPUs available to store these parameters. However, we only use a fixed portion of these parameters when processing each token, which drastically improves compute efficiency. Inference is faster at low batch sizes, and throughput is higher at large batch sizes \[5\]. Interestingly, MoEs are also more efficient to train. For example, the Switch Transformer reported a 7× pretraining speedup from the use of an MoE architecture \[2\]; see below.

![](https://substackcdn.com/image/fetch/$s_!s-Jc!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F44b313dc-540d-4ba7-8711-fe9fbd081260_1184x786.png)

(from \[2\])

**Drawbacks of using MoEs.** Despite these benefits, MoEs are also:

- Prone to instabilities during training.
- Difficult to finetune (i.e., due to issues with overfitting).
- Sensitive to low / mixed precision training techniques.
- Sensitive to hyperparameter settings (e.g., weight initialization).

Put simply, there are more bells and whistles required to get the most out of an MoE. For this reason, *MoEs may not be the best choice in every scenario*; e.g., a dense model may be an easier choice if we are looking to finetune an LLM on some task. However, if we can use them properly, MoEs have a variety of benefits.

## Mixture-of-Experts Language Models

Now that we understand the most important and fundamental concepts of MoEs, lets take a deeper look at how these concepts have been applied in the language modeling domain. Due to the fact that LLMs benefit from increased scale, MoEs have been widely adopted and seen great success within LLM research.

#### Mixtral of Experts \[5\]

![](https://substackcdn.com/image/fetch/$s_!U7jt!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdff049a9-0a22-4894-8f62-997c4d3fad78_1350x532.png)

(from \[5\])

Mixtral 8×7B (a.k.a. Mixtral of Experts) is an MoE-based extension of the open-source [Mistral-7B model](https://arxiv.org/abs/2310.06825) \[6\] that is fluent in English, French, Italian, German, and Spanish. Both of these models have open weights under an [Apache 2.0](https://fossa.com/blog/open-source-licenses-101-apache-license-2-0/) license, as well as corresponding [technical reports](https://arxiv.org/abs/2310.06825) that provide details about the models.

Mixtral converts every layer of Mistral to an expert layer with eight experts. Two of these experts are active for each token, yielding a model with 47 billion total parameters and 13 billion active parameters. The model also has a [context length](https://cameronrwolfe.substack.com/i/143156742/what-is-prompt-engineering) of 32K, which is 4× larger than its non-MoE counterpart. As shown in the figure above, Mixtral outperforms Mistral across the board and especially excels in code generation, math, and multilingual benchmarks, even exceeding the performance of the larger [LLaMA-2-70B](https://arxiv.org/abs/2307.09288) model in some cases.

![](https://substackcdn.com/image/fetch/$s_!QELC!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8a7dc1e2-e66c-4a30-a0a7-518ae7e3a566_1536x596.png)

(from \[7\])

**Mistral-7B architecture.** The base LLM architecture for Mixtral 8×7B is a decoder-only transformer that exactly matches the architecture settings of Mistral-7B \[6\]. Compared to the standard decoder-only LLM architecture, there are a few changes made by Mistral-7B:

1. *[Grouped-Query Attention (GQA)](https://cameronrwolfe.substack.com/i/142044446/efficient-masked-self-attention) \[7\]*: shares key and value projections between groups of self-attention heads to improve efficiency; see above.
2. *[Sliding Window Attention (SWA)](https://arxiv.org/abs/2004.05150) \[8\]*: computes (masked) self-attention over a fixed window of size `W` for each token to allow the LLM to handle sequences of arbitrary length with a reduced inference cost [^5]; see below.

Because we use SWA, the model can use tricks like [rolling buffer / circular caches](https://github.com/NVIDIA/TensorRT-LLM/blob/b171e879563ff0ba4eb35b94cf0e59a471e13d80/docs/source/advanced/gpt-attention.md#sliding-window-attention-cyclic-rolling-buffer-kv-cache) to make the [KV cache](https://training.continuumlabs.ai/inference/why-is-inference-important/key-value-cache) more memory efficient or [chunked prefill](https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/#balancing_prefill_and_decode_phases_with_chunked_prefill) to increase inference speed. Mixtral 8×7B adopts the same architecture conventions.

![](https://substackcdn.com/image/fetch/$s_!wR_h!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F394e1379-d10e-4db6-af55-0b6c9b66c247_1182x648.png)

(from \[6\])

**More details.** As mentioned previously, Mixtral converts every layer of the LLM to an expert layer. Within each expert layer, a simple routing mechanism is adopted that takes a softmax over the Top- `K` logits of a linear layer for every token— *this matches the routing mechanism discussed at the start of this overview*; see below.

![](https://substackcdn.com/image/fetch/$s_!WEa8!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fae644a78-4c91-4ba4-9773-a8b0e146e41b_1608x576.png)

(from \[5\])

Authors mention in \[5\] that Mixtral is also pretrained over a multilingual corpus, allowing the model to understand multiple languages. As shown below, Mixtral universally outperforms LLaMA models on multi-lingual benchmarks.

![](https://substackcdn.com/image/fetch/$s_!97Yj!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F837543f9-aa0c-4fa3-b2c3-c1f7bed38896_1358x270.png)

(from \[5\])

**Routing analysis.** To conclude the paper, authors in \[5\] perform some detailed analysis of how experts are selected for tokens across several domains to see if any interpretable patterns can be deduced. When we plot the distribution of tokens assigned to different experts for various topic areas within [The Pile](https://pile.eleuther.ai/), there are no obvious patterns in token assignment that arise; see below.

![](https://substackcdn.com/image/fetch/$s_!9uDV!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5420191d-b73d-4b4f-bcd7-4f762006605f_1342x1068.png)

(from \[5\])

However, the MoE does exhibit some structured behavior. For example, the words “self” in Python code and “Question” in English— *even though they are comprised of multiple tokens* —often get routed through the same expert. Similarly, indentation tokens in code are usually sent to the same expert, and consecutive sequences— *just sequences of tokens that are close to each other* —generally are sent to the same expert; see below. These results indicate that *i)* experts are not specialized by topic but *ii)* the MoE’s routing mechanism does obey some structured behavior with respect to the syntax or contents of the model’s input.

![](https://substackcdn.com/image/fetch/$s_!6qx0!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5f121afd-9afa-46cd-be0d-c1833a67a024_1340x864.png)

(from \[5\])

**Scaling up.** After Mixtral, a larger version of the model was released, called [Mixtral-8×22B](https://mistral.ai/news/mixtral-8x22b/). This model has 141 billion total parameters and 39 billion active parameters, making it ~3× larger than the original Mixtral model. Mixtral-8×22B is especially capable on coding and mathematics tasks, has an expanded context length of 64K, and is natively capable of [function calling](https://cameronrwolfe.substack.com/p/teaching-language-models-to-use-tools). The key benefits of Mixtral-8×22B compared to other open models are summarized below.

![](https://substackcdn.com/image/fetch/$s_!YK3v!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbe4dbfbd-b740-4458-84da-baa705537b5d_2084x1206.png)

([source](https://mistral.ai/news/mixtral-8x22b/))

#### Grok (from xAI) \[9\]

Although there is no detailed technical report on the model, one of the most notable recent examples of MoE-based LLMs is xAI’s Grok. The initial Grok-1 model was released in early 2024. Researchers revealed that the model is a 314 billion parameter MoE with 25% of weights active for each token (i.e., ~70-80 billion active parameters). The architecture and [base model weights](https://huggingface.co/xai-org/grok-1) of Grok-1 were open-sourced under and Apache 2.0 license. However, this is a pretrained base model, and no details were provided on the model’s post training process.

![](https://substackcdn.com/image/fetch/$s_!mVmw!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9725fa0c-1631-4bc2-bf89-1b4417d60a10_1588x700.png)

(from \[10\])

**Grok-1.5 \[10\].** Shortly after the initial release of Grok-1 [^6], a follow-up version of the model was published with better reasoning and long context understanding capabilities. For example, Grok-1.5 performs much better on math and coding-related tasks; see above.

> *“The model can handle longer and more complex prompts, while still maintaining its instruction-following capability as its context window expands.”* - from \[10\]

Grok-1.5 can process sequences up to 128K tokens with perfect retrieval on the [needle in a haystack test](https://github.com/gkamradt/LLMTest_NeedleInAHaystack); see below. Authors also note that the model maintains solid instruction-following capabilities when given a lot of context, which is a [much better sign](https://arxiv.org/abs/2406.10149) of long context capabilities compared to pure retrieval [^7].

![](https://substackcdn.com/image/fetch/$s_!UTxu!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1d250ac8-e79a-4b30-8bc9-31dac5624b38_1604x862.png)

(from \[10\])

Given that Grok-1.5 and Grok-1 were released in such close succession, we can infer that the advancements made by Grok-1.5 are driven by post training— *it is extremely unlikely that a different pretrained base model was created during this time*.

**Grok-2.** More recently, [Grok-2](https://x.ai/blog/grok-2) was released, which has improved reasoning, coding, and chat— *as measured by [Chatbot Arena](https://arxiv.org/abs/2403.04132) —* capabilities. Grok-2 also has a variety of other small improvements (e.g., tool usage, retrieval, factuality, etc.) and a distilled version of Grok-2, called Grok-2-mini, was released with the main model. However, no public details were shared about the architecture of Grok-2— *the model was likely trained from scratch and may or may not be MoE-based*.

#### DBRX (from Mosaic) \[11\]

![](https://substackcdn.com/image/fetch/$s_!OkzI!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb350e0ed-a8b6-44f7-806a-d581b785e74d_1934x1230.png)

(from \[11\])

DBRX is the latest model in the [series of Open LLMs](https://cameronrwolfe.substack.com/p/democratizing-ai-mosaicmls-impact) released by [Mosaic](https://www.databricks.com/research/mosaic). Two versions of the model were released— *a base model ([DBRX base](https://huggingface.co/databricks/dbrx-base)) and a finetuned model ([DBRX Instruct](https://huggingface.co/databricks/dbrx-instruct))* —under an open license (i.e., the [Databricks open model license](https://www.databricks.com/legal/open-model-license)). DBRX is an MoE-based LLM with the following specifications:

- 132 billion total parameters with 36 billion active parameters.
- 16 experts in each MoE layer with 4 experts active for each token.
- Pretrained on 12 trillion tokens of optimized text.
- 4× improvement in pretraining efficiency.

Most notably, DBRX is a “fine-grained” MoE model. In other words, the model uses a larger number of experts in each MoE layer, but each individual expert is smaller. For reference, both Mixtral and Grok-1 contain eight experts— *two of which are active for any given token* —within each of their MoE layers. By using fine-grained experts, each MoE layer has more expert combinations (65× more in particular) to choose from, which was found to improve quality in \[11\].

**Training data.** The pretraining dataset for DBRX is very large [^8], but authors in \[11\] also invest significantly into improving the quality of the data. As a result, the statistical training efficiency of DBRX is higher than normal— *training is faster because we achieve higher accuracy with fewer tokens*. More specifically, authors in \[11\] estimate that the new data is 2× more efficient token-for-token, meaning that we can train over half as many tokens and achieve the same level of performance. This claim was verified by testing the impact of the new model’s pretraining data in isolation (i.e, using a fixed model with different pretraining data).

> *“In isolation, better pretraining data made a substantial impact on model quality. We trained a 7B model on 1T tokens using the DBRX pretraining data. It reached 39.0% on the Databricks Gauntlet compared to 30.9% for MPT-7B.”* - from \[11\]

Additionally, curriculum learning is used to train DBRX— *the mixture of pretraining data is dynamically changed throughout the pretraining process*. The details of this curriculum learning strategy were later outlined in [this paper](https://arxiv.org/abs/2406.03476). The curriculum learning strategy used by DBRX just upsamples smaller, domain-specific datasets towards the end of training because this data is higher quality relative to data obtained via web-crawling. This simple curriculum learning strategy is found to provide a significant boost in performance on difficult benchmarks; see below.

![](https://substackcdn.com/image/fetch/$s_!K8EV!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F56cc31c0-8721-416d-abcd-5cbf8e53c1d6_1596x704.png)

([source](https://arxiv.org/abs/2406.03476))

**Tokenizer and context window.** DBRX has a context length of 32K and uses the GPT-4 tokenizer(available via [tiktoken](https://github.com/openai/tiktoken)). According to the authors, the GPT-4 tokenizer was selected mostly due to performance. This tokenizer has a large vocabulary and is very token efficient, which naturally improves decoding and training speed by representing the same amount of text with fewer tokens.

![](https://substackcdn.com/image/fetch/$s_!FrL3!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F80f7d600-d692-4c95-93db-4078c2c4ca7e_1970x638.png)

(from \[11\])

**Efficiency wins.** The proposal of DBRX comes with large improvements in terms of pretraining efficiency. Beyond what we have learned about so far, there are several additional sources of efficiency gains mentioned in \[11\]:

- The MoE architecture, which is found in smaller-scale experiments to require 1.7× fewer FLOPS during training.
- Other modifications to the [decoder-only architecture](https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse) (i.e., [RoPE](https://cameronrwolfe.substack.com/i/142044446/better-positional-embeddings), [GLU activation](https://pytorch.org/docs/stable/generated/torch.nn.GLU.html), and [GQA](https://cameronrwolfe.substack.com/i/142044446/efficient-masked-self-attention)).
- *“Better optimization strategies”*.

When considering all data, architecture, and optimization changes, the end-to-end training process for DBRX requires 4× less compute when compared to the pretraining pipeline used for prior models. To determine this number, authors in \[11\] compare a smaller variants of DBRX to their prior [MPT-7B model](https://www.databricks.com/blog/mpt-7b), finding that the smaller DBRX models achieve similar performance on the [Databricks Gauntlet](https://github.com/mosaicml/llm-foundry) while using 3.7× fewer FLOPS during training.

![](https://substackcdn.com/image/fetch/$s_!r8zJ!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdb50b0e9-7f1a-457c-acf9-b9c9f495620b_1960x1228.png)

(from \[21\])

DBRX also comes with improvements to inference efficiency— *up to 2× faster than LLaMA-2-70B at 150 tokens per second per user in load tests*. These measurements are made using an [optimized serving infrastructure](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) with [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and 16 bit precision, which is [very fast](https://x.com/natolambert/status/1772999462538887493?s=20). The MoE architecture of DBRX also aids inference efficiency due to the relatively low number of active parameters. For example, DBRX is 40% of the size of Grok-1 in both total and active parameters.

> “Training mixture-of-experts models is hard. We had to overcome a variety of scientific and performance challenges to build a pipeline robust enough to repeatably train DBRX-class models in an efficient manner.” - from \[11\]

Training MoEs is generally difficult due to instabilities that arise during training, communication bottlenecks, and more. However, DBRX achieves impressive results in terms of stability, efficiency, and performance due to the optimized pretraining strategy outlined in \[11\]. Notably, there is not single change or advancement that enables these results. *The impressive pretraining pipeline used by DBRX is enabled by a large number of small, practical changes*.

**Empirical evaluation.** In comparison to other open LLMs, we see that DBRX-Instruct achieves better performance on composite benchmarks by a large margin when compared to Mixtral; see below. Despite being a general-purpose LLM, DBRX has impressive programming skills, outperforming Grok-1 (more than twice its size!) and even specialized coding models like [CodeLLaMA-70B](https://ai.meta.com/blog/code-llama-large-language-model-coding/). DBRX also performs well on reasoning and math-based tasks.

![](https://substackcdn.com/image/fetch/$s_!0Bw0!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F77fe464f-f391-4fc1-aa96-8649f07ca17e_1968x1182.png)

(from \[11\])

Compared to closed models, DBRX surpasses the performance of GPT-3.5 and is competitive with [Gemini-1.0 Pro](https://cameronrwolfe.substack.com/p/google-gemini-fact-or-fiction). Gemini-1.0 Pro only outperforms DBRX on [GSM8K](https://huggingface.co/datasets/gsm8k), while [Mixtral-Medium](https://docs.mistral.ai/guides/model-selection/#mistral-medium-intermediate-tasks-that-require-language-transformation) performs better on a select few tasks that are considered; see below. At a high level, DBRX seems to be good at programming, math, general knowledge, commonsense reasoning, and retrieval / [RAG](https://cameronrwolfe.substack.com/p/a-practitioners-guide-to-retrieval).

![](https://substackcdn.com/image/fetch/$s_!x5-U!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fea6abdb5-8d58-475e-84f4-4e567bceae00_1886x968.png)

(from \[21\])

#### OpenMoE: An Early Effort on Open Mixture-of-Experts Language Models \[12\]

Despite the success of MoEs in the language modeling domain, the number of truly open-source MoEs— *meaning that code, information, data, weights and more are all shared publicly* —is relatively low. To solve this issue, OpenMoE \[12\] conducts a large-scale effort to train a suite of decoder-only MoE LLMs ranging from 650 million to 34 billion parameters. These models adopt fine-grained experts with varying granularity (i.e., 16 or 32 experts). The findings from this effort are documented in \[12\] and all models are shared openly. The authors also provide a well-documented code repository that can be used to reproduce their results.

> *“Using an MoE every layer introduces more computational overhead during routing and induces a worse cost-effectiveness trade-off than interleaved MoE usage.”* - from \[12\]

**Design choices.** OpenMoE models adopt the settings of ST-MoE \[3\], including the same routing mechanism and number of active experts (i.e., `k = 2`). Authors choose to covert only every fourth or sixth transformer block to an MoE layer, finding that larger strides yield a better tradeoff in terms of cost and efficiency.

The pretraining dataset used for OpenMoE contains a heavy distribution of code. In fact, code comprised over 50% of the dataset during the early phases of pretraining, but this ratio was adjusted later in training due to being sub-optimal; see below. For alignment, OpenMoE undergoes SFT after pretraining— *using the data from [WildChat](https://arxiv.org/abs/2405.01470)* —to induce better instruction-following capabilities.

![](https://substackcdn.com/image/fetch/$s_!LQq2!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7cf1d6fd-5607-4b2a-8588-ae4005a35142_1356x744.png)

(from \[12\])

**Routing dynamics.** One of the key contributions of OpenMoE is a detailed analysis of the routing decisions made within the models. First, we see that— *similarly to results shown in prior work \[5\]* —experts do not tend to specialize in any particular domain; see below.

![](https://substackcdn.com/image/fetch/$s_!lGyr!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fad376545-5625-4629-858c-f1c93eb9e618_1176x538.png)

(from \[12\])

However, we do see some level of expert specialization across natural languages and specific tasks, as shown in the figure below.

![](https://substackcdn.com/image/fetch/$s_!P2rn!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd3a863d3-b6fb-460b-81a8-61225c286e11_952x526.png)

(from \[12\])

When we dig into this trend, however, we see that the dynamics of token routing are primarily dictated by the token ID. In other words, the same token will almost always be routed to the same expert, *no matter the context in which that token exists*. This pattern is referred to as “Context-Independent Specialization” in \[12\].

> *“This is a very interesting finding because the tokens with the same Token ID have very diverse contexts in different sentences. For instance, the token ‘an’ can also be part of ‘an apple’ or ‘another’. However, all these tokens have very strong specialization on only a few fixed experts.”* - from \[12\]

Interestingly, experts have observable patterns in the tokens that they prefer; see below. For example, “have”, “has” and “had” are all routed to the same expert, while one expert receives the “=”, “and” and “\\n” tokens— *very common tokens within coding languages*. We see in \[12\] that such routing patterns are solidified during the early stages of pretraining and rarely change later in training.

![](https://substackcdn.com/image/fetch/$s_!lMPH!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F93c65074-8ede-4efa-a501-c897334cd9e2_974x372.png)

(from \[12\])

**Routing issues.** Beyond the routing patterns observed in \[12\], we also see that OpenMoE models exhibit some routing behaviors that could damage their performance. For example, the models tend to drop tokens later in the sequence, which can damage performance on long sequence tasks (e.g., multi-turn chat).

![](https://substackcdn.com/image/fetch/$s_!hbmV!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F97d05563-0260-40a7-9af0-9acd82d68fba_2106x908.png)

OpenMoE models perform worse on multi-turn chat problems (from \[12\])

Because routing dynamics are fixed during the early phases of the pretraining process, these behaviors are hard to fix during post training. In fact, OpenMoE models are observed to generally struggle with the domain gap between data during pretraining and SFT [^9] — *the token routing dynamics become sporadic due to the difference in data composition*. To solve these issues, authors in \[12\] recommend mixing instruction-following data into the pretraining dataset.

**Model evaluation.** Overall, OpenMoE models do not set new state-of-the-art performance among MoE LLMs— *authors in \[12\] openly state this fact and admit that the performance of OpenMoE models could be greatly improved through better design*. The larger contribution of OpenMoE models is their transparency. The details and artifacts publicly shared in \[12\] can accelerate open research efforts on MoEs by providing necessary resources to conduct further research on this topic.

#### DeepSeek-v2 \[14\] and DeepSeek-V3 \[15\]

The recently-proposed DeepSeek MoE models, including DeepSeek-v2 \[14\] and DeepSeek-v3 \[15\], have made waves within the LLM research community for a variety of reasons:

- Their weights are shared publicly.
- They come with technical reports that share many details.
- Their performance is impressive— *on par with many closed models*.
- Their training costs are pretty reasonable.

As we will see, the DeepSeek models make a variety of unique design choices that maximize both their training efficiency and downstream performance.

**DeepSeek-v2 \[14\]—** *a 236 billion parameter MoE with 21 billion active parameters* —proposes the MoE architecture used by the later DeepSeek-V3 model. The DeepSeek MoE models are a bit different than prior models, as they slightly modify the underlying transformer block to boost performance and efficiency. As shown below, the DeepSeek-v2 model— *in addition to performing well* —is quite impressive from a training and inference efficiency perspective, making it a strong starting point for the much larger DeepSeek-v3 model.

![](https://substackcdn.com/image/fetch/$s_!7ZkH!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe76588c2-e5fe-48d7-9574-954a13db6c03_1542x838.png)

(from \[14\])

**Multi-head latent attention (MLA).** Instead of standard, multi-headed attention, DeepSeek-v2 adopts MLA, which is an efficient attention variant. Similarly to [multi-query attention](https://arxiv.org/abs/1911.02150) or [grouped-query attention](https://arxiv.org/abs/2305.13245), MLA aims to minimize memory consumed by the model’s [KV cache](https://dipkumar.dev/becoming-the-unbeatable/posts/gpt-kvcache/). Unlike other efficient attention variants, however, MLA does not have a significant performance tradeoff.

![](https://substackcdn.com/image/fetch/$s_!flov!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F39c9522f-3e36-42e9-98c8-3b580f913718_1170x488.png)

(from \[14\])

In particular, this gain in memory efficiency is achieved via a low-rank, joint projection that allows us to represent all key and value vectors with a much smaller (latent) vector; see above. We can upsample this vector— *just linearly project it to form several, larger vectors* —to restore the full key and value vectors, but we only have to store the latent vector in our KV cache, thus drastically reducing memory consumption. *Adopting MLA decreases the size of DeepSeek-v2’s KV cache by over 93% compared to a 67 billion parameter dense model*.

**DeepSeek MoE architecture.** Beyond using MLA, DeepSeek models adopt a unique MoE layer structure. Similar to DBRX, these models use fine-grained experts. However, a subset of these experts are shared. The motivation for adopting such a structure is to encourage specialization among a larger number of experts while minimizing redundant information between experts. A full schematic of the block structure used by DeepSeek models is provided below.

![](https://substackcdn.com/image/fetch/$s_!InD1!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe2bf05fc-8703-4bf5-9610-edce7c6a6a91_1534x1384.png)

(from \[14\])

Authors in \[14\] also adopt an interesting load balancing strategy for handling the fine-grained experts used by DeepSeek-v2. In addition to using the auxiliary load balancing loss proposed in \[2\], DeepSeek-v2 has two auxiliary loss terms that aim to balance communication between devices during distributed training.

![](https://substackcdn.com/image/fetch/$s_!RYwy!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6bf7a65d-5541-4b40-ad45-1ab02529544d_1340x954.png)

Device-level load balancing auxiliary loss (from \[14\])

Using fine-grained experts means that we must dispatch each token to a larger number of experts. In a distributed training setting, experts may be on different devices and multiple experts reside on each device. To ensure communication and computation are balanced between devices, we need additional auxiliary losses that *i)* group experts by the device on which they reside and *ii)* encourage the MoE to perform balanced routing on a per-device basis. For example, the auxiliary loss shown above encourages balanced *computation* among devices. There is an extra loss proposed in \[14\] to encourage balanced *communication* among devices as well.

![](https://substackcdn.com/image/fetch/$s_!a08q!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc26d7720-a597-49c3-82b7-5ee830132411_1846x1186.png)

(from \[15\])

**DeepSeek-v3 \[15\]** is a much larger version of DeepSeek-v2 [^10], having 671 billion total parameters and 37B active parameters. This larger model is pretrained on a massive corpus comprised of 14.8 trillion tokens. After pretraining, a multi-phase post training pipeline is applied:

- The model first undergoes a two-stage context extension procedure in which it is finetuned (via SFT) to have a maximum context length of 32K, then further finetuned to have a context length of 128K.
- After context extension, the model undergoes further SFT and RLHF to align it to human preferences.
- Capabilities from the recently-proposed [R1 reasoning model](https://arxiv.org/abs/2501.12948) are also distilled into DeepSeek-v2 during post training.

The final DeepSeek-v3 model outperforms closed-source models and achieves similar performance to even the best closed LLMs; see above. DeepSeek-v3 also makes several modifications to the training and load balancing strategy for the MoE, leading the model’s training process to be both efficient and stable.

> *“Despite its excellent performance, DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training. In addition, its training process is remarkably stable… we did not experience any irrecoverable loss spikes or perform any rollbacks.”* - from \[15\]

The architecture of DeepSeek-v3 is inspired by its predecessor; e.g., MLA, fine-grained experts, and shared experts are all used by DeepSeek-v3. Unlike DeepSeek-v2, however, DeepSeek-v3 uses a **Multi-Token Prediction (MTP)** training objective. This objective is an extension of the supervised, cross entropy-based [next token prediction objective](https://cameronrwolfe.substack.com/i/136638774/understanding-next-token-prediction) that is used almost universally for training LLMs. Instead of predicting the next token for each token within a sequence, MTP predicts `D` future tokens. These predictions are made sequentially by a set of additional modules that are added to the model’s architecture; see below.

![](https://substackcdn.com/image/fetch/$s_!dRxs!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7ac297b7-3d37-46ec-9b30-f45d2560bb04_1846x964.png)

(from \[15\])

Once several future tokens have been predicted, we can apply the cross-entropy loss normally. Applying this loss over several future tokens predicted via MTP provides a richer training signal to the model, which improves training efficiency and overall performance. Going further, these additional modules used for MTP can also be used to improve inference efficiency via [speculative decoding](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/). However, authors in \[15\] state that the MTP strategy is used purely to benefit model performance— *additional modules are discarded after training*.

![](https://substackcdn.com/image/fetch/$s_!yYzN!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Faa69f7cc-41ac-4b4f-9a13-c7b791a31430_1836x480.png)

Auxiliary-loss-free load balancing strategy (from \[15\])

An **auxiliary-loss-free load balancing** strategy is also used by DeepSeek-v3 that simply adds a per-expert bias term to the selection of Top- `K` experts; see above. At each iteration, the bias term for each expert is either increased or decreased by a fixed factor `γ` based upon whether that expert was underloaded or overloaded, respectively. Importantly, these biases are only used when selecting the top- `K` experts— *they do not impact the computation of expert probability within the router*. This approach is found to effectively balance expert utilization within the MoE and eliminate performance deterioration due to the use of load balancing losses. However, authors in \[15\] do note that they still use an auxiliary load balancing loss (with a very low scaling factor) when training DeepSeek-v3.

![](https://substackcdn.com/image/fetch/$s_!9qXc!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9ff45514-375d-4e7d-ab22-f95285b865a4_1834x336.png)

(from \[15\])

**Training efficiency.** Due to the efficiency and performance benefits of the strategies outlined above, DeepSeek-v3 is incredibly economical. Plus, the model is trained using a novel FP8 mixed precision training framework, *marking the first validation of 8-bit training for large-scale LLMs*. In total, training the final model was estimated [^11] to cost ~$5.6M; see above. In short, the DeepSeek-v3 is:

- Trained in a very economical fashion (and with several novel advancements like FP8 training and MTP!)
- Incredibly impressive for an open model— *highly competitive with even the best closed LLMs*.
- Based on an interesting MoE architecture with several novel modifications.

**Reasoning.** DeepSeek-v3 also serves as a base model for [DeepSeek-R1](https://arxiv.org/abs/2501.12948) \[13\], a recently-released open reasoning model. Put simply, R1 is an open replication of the [o1-style of models](https://openai.com/index/learning-to-reason-with-llms/) that have been recently explored by OpenAI. As explained in its detailed technical report, this model uses pure reinforcement learning to learn how to solve complex (verifiable) reasoning tasks by crafting extremely long chains of thought. As shown in the figure below, the performance of R1 is quite impressive, especially for an open model. However, the capabilities of R1 would not be possible without first having access to an incredibly-capable base model.

![](https://substackcdn.com/image/fetch/$s_!fIHe!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F97f49231-0329-4f2e-ab77-a274906e27ae_1948x1258.png)

(from \[13\])

## Final Thoughts

MoEs have many benefits that are especially suited to language modeling. They enable exploration of larger scales without drastic increases in compute, reduce training costs, and can be efficiently hosted. Although the idea of sparsity has existed for a long time within the machine learning literature, MoEs are an especially impactful instantiation of sparsity. They leverage sparsity in a manner that is compatible with modern hardware and can be practically implemented on a GPU. Interestingly, early MoE variants struggled with adoption due to their complexity, instability, and difficulty of use. However, the advancements we have seen in this overview have turned the MoE into something practical and impactful— *a simple and promising extension of the decoder-only transformer architecture*.

#### New to the newsletter?

Hi! I’m [Cameron R. Wolfe](https://cameronrwolfe.me/), Deep Learning Ph.D. and Machine Learning Scientist at [Netflix](https://research.netflix.com/research-area/nlp-and-conversations). This is the Deep (Learning) Focus newsletter, where I help readers better understand important topics in AI research. If you like the newsletter, please subscribe, share it, or follow me on [X](https://twitter.com/cwolferesearch) and [LinkedIn](https://www.linkedin.com/in/cameron-r-wolfe-ph-d-04744a238/)!

#### Bibliography

\[1\] Shazeer, Noam, et al. "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer." arXiv preprint arXiv:1701.06538 (2017).

\[2\] Fedus, William, Barret Zoph, and Noam Shazeer. "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity." Journal of Machine Learning Research 23.120 (2022): 1-39.

\[3\] Zoph, Barret, et al. "St-moe: Designing stable and transferable sparse expert models." arXiv preprint arXiv:2202.08906 (2022).

\[5\] Jiang, Albert Q., et al. "Mixtral of experts." arXiv preprint arXiv:2401.04088 (2024).

\[6\] Jiang, Albert Q., et al. "Mistral 7B." arXiv preprint arXiv:2310.06825 (2023).

\[7\] Ainslie, Joshua, et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." arXiv preprint arXiv:2305.13245 (2023).

\[8\] Beltagy, Iz, Matthew E. Peters, and Arman Cohan. "Longformer: The long-document transformer." *arXiv preprint arXiv:2004.05150* (2020).

\[9\] xAI. “Open Release of Grok-1” *[https://x.ai/blog/grok-os](https://x.ai/blog/grok-os)* (2024).

\[10\] xAI. “Announcing Grok-1.5” *[https://x.ai/blog/grok-1.5](https://x.ai/blog/grok-1.5)* (2024).

\[11\] Mosaic Research (Databricks). “Introducing DBRX: A New State-of-the-Art Open LLM” *[https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)* (2024).

\[12\] Xue, Fuzhao, et al. "Openmoe: An early effort on open mixture-of-experts language models." *arXiv preprint arXiv:2402.01739* (2024).

\[13\] Guo, Daya, et al. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." *arXiv preprint arXiv:2501.12948* (2025).

\[14\] Liu, Aixin, et al. "Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model." *arXiv preprint arXiv:2405.04434* (2024).

\[15\] Liu, Aixin, et al. "Deepseek-v3 technical report." *arXiv preprint arXiv:2412.19437* (2024).

---

[^1]: In other words, the contents of this embedding layer are updated (via gradient descent) throughout the training process, similarly to any other model parameter.

[^2]: To improve both hardware (i.e., GPU utilization, throughput, etc.) and statistical (i.e., how quickly the model learns from data) efficiency, we want to have a relatively uniform number of tokens dispatched to all experts in each batch of data.

[^3]: This quantity is predicted by our routing algorithm and is, therefore, differentiable. So, the loss function as a whole is differentiable even though the fraction of tokens sent to each expert is not itself a differentiable quantity.

[^4]: We also have to multiple this loss term by `N` to ensure that the loss remains constant as the number of experts is increased.

[^5]: It may seem like SWA limits each token to only “look at” a few tokens within a sequence. However, if we stack `k` consecutive layers of SWA on top of each other, the effective context window for each token increases. The representation of the current token is actually influenced by the `k⋅W` tokens that come before it.

[^6]: Both of these models were released in March of 2024 within ~10 days of each other!

[^7]: The needle in a haystack test tests an LLM’s ability to retrieve information in its context. However, the model may still have bad long context abilities even if it scores perfectly on the needle in a haystack test; e.g., instruction-following or reasoning capabilities could get way worse when given a long context.

[^8]: For reference, the prior models released by the same team— *[MPT-7B](https://www.databricks.com/blog/mpt-7b) and [MPT-30B](https://www.databricks.com/blog/mpt-30b)* —were pretrained on only 1 trillion tokens of text.

[^9]: We encounter much different styles of data during SFT compared to pretraining; e.g., multi-turn chat data, instruction templates, and more.

[^10]: Between these models, DeepSeek also released an intermediate model, called DeepSeek-v2.5; see [here](https://api-docs.deepseek.com/news/news1210) for details.

[^11]: These estimates are made by assuming a $2 rental price per H800 GPU hour. Additionally, they reflect the pure compute cost of training the final model only, excluding any supplemental costs or experiments. The actual total cost of training DeepSeek-v3 is [undoubtedly much larger](https://www.interconnects.ai/p/deepseek-v3-and-the-actual-cost-of) than this reported number.