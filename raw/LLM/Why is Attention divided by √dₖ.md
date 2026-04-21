---
title: "Why is Attention divided by √dₖ: The Secret Behind Scaled Attention in Transformers"
source: "https://medium.com/@srivatsa.n63/why-is-attention-divided-by-d%E2%82%96-the-secret-behind-scaled-attention-in-transformers-44f36465266f"
author:
  - "[[Srivatsa Narasimha]]"
published: 2025-06-05
created: 2026-04-14
description: "A deep dive into the logic behind scaling attention by √dₖ in Transformers. Learn how this crucial step avoids exploding logits, keeps softmax stable, and enables models like GPT and BERT to train effectively."
tags:
  - "clippings"
---
[Sitemap](https://medium.com/sitemap/sitemap.xml)

![[1*-_2n-xUquZ_Ks1xZ38GqNg.png]]

Source: https://www.linkedin.com/pulse/what-self-attention-impact-large-language-models-llm-nikhil-goel-srpbc/

Since its introduction in the seminal 2017 paper “ [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) *”* by Vaswani et al., the Transformer architecture has revolutionized deep learning. The self-attention mechanism allows models to dynamically weigh input tokens based on their relationships and relevance to one another. This functionality plays a pivotal role in enhancing the model’s ability to understand context and meaning within the input data, making it an essential core component of the architecture. This design choice has given rise to architectures like BERT, GPT, and T5, which now dominate the state-of-the-art models in natural language processing and beyond.

At the heart of this mechanism is a concept known as **scaled dot-product attention**. The computation:

![[1*YJwNhroxe3EE55YOf6t3Cg.png]]

Source: Attention is all you need

includes a seemingly innocuous yet crucial detail in this mechanism - the standardization term

![[1*96zbH1j3mDzPMzpj8BHQAA.png]]

where *dₖ* is the dimensionality of the “Key” vectors. This scaling factor might seem like a small implementation detail, but it plays a crucial role in the model’s ability to learn effectively.

In this blog, we will explore **why this scaling is necessary and why the scaling factor is “the square root of the dimensionality”**, using statistical reasoning.

But first, to deeply understand attention, it’s essential to unpack what the matrices *Q*, *K* and *V* actually mean. These aren’t just arbitrary terms - they form the foundation of how attention mechanisms compare and combine tokens.

### Intuition and Mathematics Behind Query, Key, and Value

Before this computation even begins, however, each token in the input sequence undergoes two important transformations. First, raw text is passed through a **tokenizer**, which maps each word or subword into a unique embedding vector. These embeddings represent the semantic content of the input text. Second, because the Transformer lacks any recurrence or convolution, **positional encodings** are added to the token embeddings to provide a notion of word order.

Thus, the true input to the attention mechanism is the sum of these two components:

> X = TokenEmbeddings + Positional Embeddings

This combined input is what gets linearly transformed into the query, key, and value matrices used in the computation of attention matrices.

*Q* (query), *K* (key), and *V* (value) are all projections of the input embeddings. Each token in the input sequence is transformed into these three representations. The dot product of a query and key determines how much focus (attention weight) a token should receive relative to others, and the value vector is used to compute the output representation accordingly.

### Intuitive Interpretation

- **Query (Q):** Think of this as a question you’re asking about the input. Each token produces a query vector that essentially asks: “What am I looking for?”
- **Key (K):** These are like labels or tags that describe the content of each token. They answer: “What do I contain?”
- **Value (V):** These vectors hold the actual information you want to pass forward. Think of them as the payload.

During attention computation, for every token, we ask: *“How well does my query match each key?”* The result of this comparison is a set of attention weights, which are then used to **blend the value vectors** accordingly.

### Mathematical Perspective

Given an input sequence *X = \[x₁, x₂,……, xₙ\]*,

each token *xᵢ*

![[1*Hl_gXir-87hqMCiXjpy4pQ.png]]

is projected using three learned matrices:

![[1*aIDzBdwEBsl1PubvMxBWBw.png]]

Source: Attention is all you need

where

![[1*HQI7QFd_NRH3VO7bO9iC-Q.png]]

- *n* is the sequence length of the input sentence, which, in the case of BERT, the maximum value is 512.
- ***d*** ₘₒ𝒹ₑₗis the dimensionality of each token embedding before it’s projected into queries, keys, and values. It reflects the size of the input representation - for example, in GPT-3, this might be 12288 or 768 in BERT.
- ***d*** *ₖ* is the dimensionality of the queries and keys after projection. In this case, ***d*** *ₖ =* ***dq*** *\=* ***d*** ᵥ. It is typically set to a smaller value than *dₘₒ𝒹ₑₗ*, especially when using multi-head attention.

Next, we compute the attention scores:

![[1*mbKDLm4IIhcbGtRHgyhw7w.png]]

Essentially, these scores measure the **similarity** between each query and all keys (using dot product), standardized by *√dₖ*. Why? If we examine the formula closely, it resembles the cosine similarity measure:

![[1*fGOog-JupfhCbBFQ1Qx0Jg.png]]

where the similarity between vectors **A** and **B** is determined by their dot product divided by the product of their magnitudes (norms). This can be written as:

![[1*lqgsRpFNh89LNdTuRlPU3w.png]]

The attention score formula mirrors this concept, with the scaling factor being *√dₖ*. These scores represent how strongly each token in the query attends to tokens in the key. Functionally, the resulting score matrix acts as a set of weights, capturing the extent to which each word contributes to the contextual encoding of every other word in the sequence.

### But why is the scaling factor √dₖ?

Let us dive into a bit of statistics now!

## Get Srivatsa Narasimha’s stories in your inbox

Join Medium for free to get updates from this writer.

In statistics, if *X* and *Y* are independent random variables:

If *E(X) r* epresents the expected value of *X* and *Var(X)* represents the variance:

![[1*TKpa1mn45_RO8Gg71VUKuw.png]]

Now, let *Q* and *K* be independent *dₖ×dₖ* matrices, where each entry is an independent random variable with zero mean (*E(X))* and unit variance:

![[1*e3d5Ct37JrcbBIEzTQd3HQ.png]]

Same for *K*. Because of this symmetry and independence, we can analyze the top-left element of *QK (denoted (QK)* ₁₁)without loss of generality:

![[1*fbZU6-8-6mOS8JH4cqjpwA.png]]

![[1*V9Ml7QPSeB8Y7jX_azWEkw.png]]

![[1*ngracCAJDmROG-gzU6BslA.png]]

So, without scaling, the variance of the dot product grows linearly with *dₖ*:

![[1*gLPOpempMoNfaNJWbDoo0A.png]]

Standardization involves bringing this down to **unit variance,** which is achieved by dividing the dot product by the standard deviation (***the square root of variance***). In this case, the **value of the standard deviation or the scaling factor is *√dₖ*!** Hence,

![[1*qHSMhbyrNcW5bYZajK2DHw.png]]

### Why is scaling required?

The next step after the scores are calculated is to pass them through a softmax function and multiply by *V*

![[1*n0SPWaFU9HCa8BRcyMBSbg.png]]

> “Softmax is an activation function employed in attention mechanisms -particularly within Transformer models -to transform raw attention scores into a normalized probability distribution between 0 and 1 across the input tokens before they are used to weight the encoding of the value vectors.”

![[1*Y3uqU53vpKrLgMnrD8ZZrg.png]]

Source: https://www.parasdahal.com/softmax-crossentropy

As mentioned earlier, without scaling, the variance of the dot product grows by *dₖ.* The softmax function is highly sensitive to the scale of its input. This implies that the values passed into the softmax could be large,which causes the softmax output to become very peaky (close to one-hot) or overly flat, depending on the sign and magnitude.Passing a high-variance value into the softmax function can lead to extremely peaked outputs (i.e., near one-hot), which may result in:

- A few scores dominating due to exponentiation.
- Vanishing gradients for non-maximum tokens, leading to poor learning
- The attention distribution becomes too sharp (i.e., nearly one-hot), harming the model’s ability to attend to multiple relevant tokens.

This scaling ensures that, regardless of the value of *dₖ*, the variance of the attention scores remains **bounded and stable (unit variance)**. By doing this, we ensure that the attention distribution is neither too sharp nor too flat, just right for effective gradient flow.

Below is a visualization of softmax sensitivity to scaling:

![[1*OuSJmN6HymHKjyu-30vwoA.png]]

Figure 1: Effect of Scaling on Softmax Output

As seen, lower scaling (i.e., higher variance) makes the softmax sharper, which is undesirable. When the input variance is high, even small differences between elements result in large differences after softmax, causing one token to dominate the attention distribution.

The scaling factor ***√dₖ*** brings the distribution into a more stable range, ensuring that multiple tokens can contribute meaningfully.

Let us visualize how attention scores look **before and after scaling:**

![[1*d9KNeyP-YeBbH2hQQwTo0g.png]]

Figure 2: Heatmaps of Attention Scores - Left: Raw dot products, Right: Scaled scores

Note how scaling compresses the range and results in values better suited for the softmax operation. This regularization effect leads to smoother attention distributions.

### Conclusion

The use of in-scaled dot-product attention is a practical fix for a statistical problem: the variance of dot products increases with the number of dimensions. Without this correction, the softmax would operate in an unstable regime, making training ineffective. By standardizing the scores, we ensure that the attention mechanism remains stable and learns effectively, no matter the size of the model.

> ***References***

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A, Kaiser, L., and Polosukhin, I. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *arXiv preprint arXiv:1706.03762.*
2. Bahdanau, D., Cho, K., & Bengio, Y. (2014). [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). *arXiv preprint arXiv:1409.0473.*
3. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). *arXiv preprint arXiv:1810.04805.*
4. Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). *OpenAI.*
5. Hochreiter, S., & Schmidhuber, J. (1997). [Long short-term memory. *Neural computation*](https://ieeexplore.ieee.org/abstract/document/6795963)*, 9(8), 1735–1780.*
6. [CS231n: Weight Initialization](https://cs231n.github.io/neural-networks-2/)
7. [Understanding Softmax and its Properties](https://www.parasdahal.com/softmax-crossentropy)

[![[Image.jpg|Srivatsa Narasimha]]](https://medium.com/@srivatsa.n63?source=post_page---post_author_info--44f36465266f---------------------------------------)

[![[Image 1.jpg|Srivatsa Narasimha]]](https://medium.com/@srivatsa.n63?source=post_page---post_author_info--44f36465266f---------------------------------------)

[19 following](https://medium.com/@srivatsa.n63/following?source=post_page---post_author_info--44f36465266f---------------------------------------)

Srivatsa is a Staff Data Scientist at Freshworks, working on solving business use cases using Data Science for Freddy CX and Freddy Freshsales

## Responses (1)

Write a response[What are your thoughts?](https://medium.com/m/signin?operation=register&redirect=https%3A%2F%2Fmedium.com%2F%40srivatsa.n63%2Fwhy-is-attention-divided-by-d%25E2%2582%2596-the-secret-behind-scaled-attention-in-transformers-44f36465266f&source=---post_responses--44f36465266f---------------------respond_sidebar------------------)

```c
Great article!
```