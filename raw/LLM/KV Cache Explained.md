---
title: "KV Caching Explained: Optimizing Transformer Inference Efficiency"
source: "https://huggingface.co/blog/not-lain/kv-caching"
author:
  - "[[Not Lain]]"
published: 2025-01-30
created: 2026-04-14
description: "A Blog post by Not Lain on Hugging Face"
tags:
  - "clippings"
---
## Introduction

When AI models generate text, they often repeat many of the same calculations, which can slow things down. **Key-Value caching** is a technique that helps speed up this process by remembering important information from previous steps. Instead of recomputing everything from scratch, the model reuses what it has already calculated, making text generation much faster and more efficient.

In this blogpost, we’ll break down KV caching in an easy-to-understand way, explain why it’s useful, and show how it helps AI models work faster.

![](https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/ZiRajz9XfXiPT3NIM05FS.png)

## Prerequisites

To fully grasp the content, readers should be familiar with:

1. **Transformer Architecture**: Familiarity with components such as the attention mechanism.
2. **Autoregressive Modeling**: Understanding of how models like GPT generate sequences.
3. **Linear Algebra Basics**: Concepts like matrix multiplication and transposition, which are essential for understanding attention computations.

This 👉 [**BLOG**](https://huggingface.co/blog/not-lain/tensor-dims) should cover up most of the prerequisites needed for this article.

click here for some of the most essential takeaways.
- attention weight has a shape of $\left[\right. \text{batch} , h , \left(S e q\right)_{l e n} , \left(S e q\right)_{l e n} \left]\right.$
- masked multi-head attention allows each token to be represented by itself and all the previous tokens.
- to generate a new token the model needs to look at all the previous tokens and their representations by their preceding tokens
 [![](https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/9n4ttDGvMkcZKF8puUBz0.png) ![](https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/zPsMCUsd_ohKun4r2axV0.png)](https://huggingface.co/blog/not-lain/tensor-dims)

<video src="https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/RsRm-SLIpIXdRwALshIB-.mp4" controls=""></video> 

## Standard Inference and the Rise of KV Caching

When a model generates text, it **looks at all the previous tokens** to predict the next one. Normally, it would *repeat the same calculations* for every new token, which can slow things down.

<video src="https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/PWI-EwqizVLInztmiI7Eo.mp4" controls=""></video>

> KV caching solves compute overlap by **remembering these calculations** from previous steps, this can be achieved by storing the intermediate states of attention layers during inference.

<video src="https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/HnzDhoJdAbJhSassYjzEy.mp4" controls=""></video>

## How Does KV Caching Work?

### Step-by-Step Process

1. **First Generation**: When the model sees the first input, it calculates and stores its keys and values in the cache. 
	$$
	\Downarrow
	$$
2. **Next Words**: For each new word, the model retrieves the stored keys and values and adds the new ones instead of starting over.
3. **Efficient Attention Computation**: calculate attention using the cached $K$ and $V$ along with the new $Q$ (query) to compute the output.
4. **Update Input**: add the newly generated token to the input and $\mathtt{go} \mathtt{back} \mathtt{to} \mathtt{step} \mathtt{2}$ until we finish generating.
![](https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/DbL2RbXFRoMWA5CrOaGB8.png)

The process is illustrated below:

```
Token 1: [K1, V1] ➔ Cache: [K1, V1]
Token 2: [K2, V2] ➔ Cache: [K1, K2], [V1, V2]
...
Token n: [Kn, Vn] ➔ Cache: [K1, K2, ..., Kn], [V1, V2, ..., Vn]
```

| KV Caching | Standard Inference |
| --- | --- |
| <video src="https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/DP2zDJTAU-yHrxVRh5GUt.mp4" controls=""></video> | <video src="https://cdn-uploads.huggingface.co/production/uploads/6527e89a8808d80ccff88b7a/x0L80MqTJ4VPovbRY4yb2.mp4" controls=""></video> |

In the table above we used a $d_{k} = 5$ for better visuals, note that this number can be significantly bigger than what we have presented.

## Comparison: KV Caching vs. Standard Inference

Here’s how KV caching compares to the regular generations:

| **Feature** | **Standard Inference** | **KV Caching** |
| --- | --- | --- |
| **Computation per Word** | The model repeats the same calculations for every word. | The model reuses past calculations for faster results. |
| **Memory Usage** | Uses less memory at each step, but memory grows with longer texts. | Uses extra memory to store past information, but keeps things efficient. |
| **Speed** | Gets slower as the text gets longer because it repeats work. | Stays fast even with longer texts by avoiding repeated work. |
| **Efficiency** | High computational cost and slower response times. | Faster and more efficient since the model remembers past work. |
| **Handling Long Texts** | Struggles with long texts due to repeated calculations. | Perfect for long texts as it remembers past steps. |

KV caching makes a big difference in **speed** and **efficiency**, especially for long texts. By saving and reusing past calculations, it avoids the need to start over each time, making it much faster than the regular way of generating text.

## Practical Implementation

This is a simplified example of implementing KV caching in PyTorch:

```python
# Pseudocode for KV Caching in PyTorch
class KVCache:
    def __init__(self):
        self.cache = {"key": None, "value": None}

    def update(self, key, value):
        if self.cache["key"] is None:
            self.cache["key"] = key
            self.cache["value"] = value
        else:
            self.cache["key"] = torch.cat([self.cache["key"], key], dim=1)
            self.cache["value"] = torch.cat([self.cache["value"], value], dim=1)

    def get_cache(self):
        return self.cache
```

When using the transformers library this behavior is enabled by default through the `use_cache` parameter, you can also access multiple caching methods through the [`cache_implementation`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.cache_implementation) parameter, here's a minimalistic code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-1.7B')
model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-1.7B').cuda()

tokens = tokenizer.encode("The red cat was", return_tensors="pt").cuda()
output = model.generate(
    tokens, max_new_tokens=300, use_cache = True # by default is set to True
)
output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
```

We benchmarked the code above with/without kv caching on a T4 GPU we got the following results:

| with KV Caching | Standard Inference | Speedup |
| --- | --- | --- |
| 11.7 s | 1min 1s | ~5.21x times faster |

## Conclusion

KV caching is a simple but powerful technique that helps AI models generate text faster and more efficiently. By remembering past calculations instead of repeating them, it reduces the time and effort needed to predict new words. While it does require extra memory, this method is especially useful for long conversations ensuring fast and efficient generation.

Understanding KV caching can help developers and AI enthusiasts build faster, smarter, and more scalable language models for real-world applications.

I would like to extend my sincerest gratitude to [Aritra Roy Gosthipaty](https://hf.co/ariG23498) 🤗 for his invaluable support, feedback, and dedication in developing this blog post.

## References & Further Reading

1. [Transformers KV Caching Explained](https://medium.com/@joaolages/kv-caching-explained-276520203249)
2. [Transformers Key-Value Caching Explained](https://neptune.ai/blog/transformers-key-value-caching)
3. [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
4. [Hugging Face Documentation - KV Caching in Transformers](https://huggingface.co/docs/transformers/main/en/generation_strategies#kv-caching)

### Community

[ryg81](https://huggingface.co/ryg81)

[Jan 31, 2025](#679cd0243cc265a444a09923)

Can this be similar for image generation models? (I am not a programmer:- or expert in AI))

·

[olegGerbylev](https://huggingface.co/olegGerbylev)

[Apr 2, 2025](#67ec368b0d37308e10786d89)

This comment has been hidden (marked as Spam)

[emilibennett](https://huggingface.co/emilibennett)

[Jun 11, 2025](#6848fd432b1c7fe843b0c203)

This comment has been hidden (marked as Spam)

[dutta18](https://huggingface.co/dutta18)

[Sep 13, 2025](#68c4f66d50b2167a8bfe34ad)

I really appreciate the effort that HF team puts in to create these easy-to-digest blogs. Thanks a ton!

·

[not-lain](https://huggingface.co/not-lain)

Article author [Sep 14, 2025](#68c59dad9e06a8db73e025b5)

Very grateful for the kind words [@dutta18](https://huggingface.co/dutta18) 🤗

[jonathon1964](https://huggingface.co/jonathon1964)

[Sep 26, 2025](#68d60ff7a65200ac91b4101d)

very clear ex!

[Student-Xiaoji](https://huggingface.co/Student-Xiaoji)

[Oct 26, 2025](#68fd85d5c35dcb59c382a379)

love this simple, easy understood and straight forward explanation❤  
thanks for you effort☺

[not-lain](https://huggingface.co/not-lain)

Article author [Oct 28, 2025](#69000217d65f71e8227fe941)

thanks for the kind feedback 🤗

[KANGKKANG](https://huggingface.co/KANGKKANG)

[Nov 14, 2025](#6916c70222626558d6dd306d)

maybe i can use this job on ACT model?

[kyars](https://huggingface.co/kyars)

[Dec 8, 2025](#6935ff250712d88b9e3ce84e)

I didn't understand the explanation

·

[not-lain](https://huggingface.co/not-lain)

Article author [Dec 10, 2025](#6938f9be4ad36d1320fbdc81)

Hi [@kyars](https://huggingface.co/kyars) is there any part that you think i can improve upon or is it everything?  
would appreciate any feedback!

[talrejaa8](https://huggingface.co/talrejaa8)

[Dec 14, 2025](#693d9b325e5078b5f90a6d81)

I really appreciate your effort to explaining this so well. Just one doubt I have, what exactly is being cached?

1. The QK^t dot product results and the Value vectors of the already generated tokens  
	or
2. The just the key vectors and the value vectors of already generated tokens?

Also, is this done for each transformer block in an LLM?

·

[kyars](https://huggingface.co/kyars)

[Dec 14, 2025](#693deeb8c7eb6b6bbe489408)

Yes, it's done for each transformer block in an LM because each transformer block has different attention heads. If you do it for only one transformer block across all blocks, then you don't get the same representation.