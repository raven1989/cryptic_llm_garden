---
title: "An Introduction to Speculative Decoding for Reducing Latency in AI Inference"
source: "https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/"
author:
  - "[[Jamie Li]]"
published: 2025-09-18
created: 2026-05-25
description: "Generating text with large language models (LLMs) often involves running into a fundamental bottleneck. GPUs offer massive compute, yet much of that power sits…"
tags:
  - "clippings"
---
Generating text with [large language models (LLMs)](https://www.nvidia.com/en-us/glossary/large-language-models/) often involves running into a fundamental bottleneck. GPUs offer massive compute, yet much of that power sits idle because autoregressive generation is inherently sequential: each [token](https://blogs.nvidia.com/blog/ai-tokens-explained/) requires a full forward pass, reloading weights, and synchronizing memory at every step. This combination of memory access and step-by-step dependency raises latency, underutilizes hardware, and limits system efficiency.

Speculative decoding helps break through this wall. By predicting and verifying multiple tokens simultaneously, this technique shortens the path to results and makes [AI inference](https://www.nvidia.com/en-us/solutions/ai/inference/) faster and more responsive, significantly [reducing latency](https://www.nvidia.com/en-us/solutions/ai/inference/balancing-cost-latency-and-performance-ebook/) while preserving output quality. This post explores how speculative decoding works, when to use it, and how to deploy the advanced [EAGLE-3](https://arxiv.org/abs/2503.01840) technique on NVIDIA GPUs.

## What is speculative decoding?

*Speculative decoding* is an inference optimization technique that pairs a target model with a lightweight draft mechanism that quickly proposes several next tokens. The target model verifies those proposals in a single forward pass, accepts the longest prefix that matches its own predictions, and continues from there. Compared with standard autoregressive decoding, which produces one token per pass, this technique lets the system generate multiple tokens at once, cutting latency and boosting throughput without any impact on accuracy.

Though highly capable, LLMs often push the limits of AI hardware, making it challenging to further optimize user experience at scale. Speculative decoding offers an alternative by offloading part of the work to a less resource-intensive model.

Speculative decoding works much like a chief scientist in a laboratory, relying on a less experienced but efficient assistant to handle routine experiments. The assistant rapidly works through the checklist, while the scientist focuses on validation and progress, stepping in to correct or take charge whenever necessary.

With speculative decoding, the lightweight assistant model proposes multiple possible continuations and the larger model verifies them in batches. The ultimate benefit is reducing the number of sequential steps, alleviating memory bandwidth bottlenecks. Critically, this acceleration occurs while preserving output quality, as verification mechanisms will discard any results divergent from what the baseline model itself might generate.

## Speculative decoding basics using draft-target and EAGLE-3

This section lays out the core concepts behind speculative decoding, breaking down the mechanics that make it effective. To begin, the transformer forward pass shows how sequences are processed in parallel. Subsequent steps include draft generation, verification, and sampling using a draft-target approach as an example. Together, these fundamentals provide the context needed to understand both the classic draft–target method and advanced techniques like [EAGLE-3](https://arxiv.org/abs/2503.01840).

### What is the draft-target approach to speculative decoding?

The draft-target approach is the classic implementation of speculative decoding, operating as a two-model system. The primary model is the large, high-quality target model whose output you want to accelerate. Working alongside it is a much smaller, faster draft model, which is often a distilled or simplified version of the target.

Returning to the lab scientist analogy, think of the target as the meticulous scientist ensuring correctness, while the draft is the quick assistant proposing possibilities that the scientist then verifies. Figure 1 shows this partnership in action, with the draft model quickly producing four draft tokens for the target model, which verifies and keeps two while also generating one additional token itself.

![A gif showing an example where the input is “The  Quick”. From this input, the draft model proposes “Brown”, “Fox”, “Hopped”, “Over”. The input and draft are ingested by the target model, which verifies “Brown” and “Fox” before rejecting “Hopped” and subsequently everything after. “Jumped” is the target model’s own generation resulting from the forward pass.
](https://developer-blogs.nvidia.com/wp-content/uploads/2025/09/speculative-decoding-draft-target-approach.gif)

Figure 1. The draft-target approach to speculative decoding operates as a two-model system

Speculative decoding using the draft-target approach involves the following steps:

#### Draft generation

A smaller, more efficient mechanism generates a sequence of candidate tokens (typically 3 to 12 tokens). Typically, this takes the form of a separate smaller model trained on the same data distribution. The target model’s output usually serves as the ground truth for the draft model’s training.

#### Parallel verification

The target model processes the input sequence and all draft tokens simultaneously in a single forward pass, computing probability distributions for each position. This parallel processing is the key efficiency gain, as it leverages the target model’s full computational capacity rather than leaving it underutilized during sequential generation. Thanks to the KV Cache, where the values for the original prefix have already been calculated and stored, only the new, speculated tokens incur a computational cost during this verification pass. The verified tokens are then selected to form the new prefix for the next generation step.

#### Rejection sampling

Rejection sampling is the decision-making stage that occurs after the probability distribution from the target model has been generated.

The key aspect of rejection sampling is the acceptance logic. As Figure 2 illustrates, this logic compares the proposed probability of the draft model, P(Draft), against the actual probability of the target model, P(Target).

For the first two tokens, “Brown” and “Fox,” P(Target) is higher than P(Draft), so they are accepted. However, for “Hopped,” P(Target) is significantly lower than P(Draft), indicating an unreliable prediction.

When a token such as “Hopped” is rejected by the acceptance logic, it and all subsequent tokens in the draft are discarded. The process then reverts to standard autoregressive generation from the last accepted token, “Fox,” to produce a corrected token.

![A gif showing the verification phase within the target model. P(Target) and P(Draft) are compared for each token. “Brown” passes because P(Target) ≥ P(Draft). “Hopped” Fails because P(Target) ≤ P(Draft). As each following token is affected by previous generations, all draft tokens past “Hopped” are discarded. The final generation is thus the prefix plus “Brown Fox Jumped”, where “Brown and Fox” are accepted draft generations and “Jumped” a generation solely from the target model.
](https://developer-blogs.nvidia.com/wp-content/uploads/2025/09/speculative-decoding-verification-phase-target-model.gif)

Figure 2. The acceptance logic is the key aspect of rejection sampling during parallel verification

Only when a draft token matches what the target model would have generated, is it accepted. This rigorous, token-by-token validation ensures that the final output is identical to what the target model would have produced, guaranteeing that the speedups come with no loss in accuracy.

The number of accepted tokens compared to the total generations is the acceptance rate. Higher acceptance rates equate to more significant speedups and at worst, if all draft tokens are rejected, then only the single target model token is generated.

### What is the EAGLE approach to speculative decoding?

[EAGLE](https://arxiv.org/abs/2401.15077), or Extrapolation Algorithm for Greater Language-Model Efficiency, is a speculative decoding method that operates at the feature level, extrapolating from the hidden state just before the target model’s output head. Unlike the draft–target approach, which relies on a separate draft model to propose tokens, EAGLE uses a lightweight autoregressive prediction head ingesting features from the target model’s hidden states. This eliminates the overhead of training and running a second model while still allowing the target model to verify multiple token candidates per forward pass.

[EAGLE-3](https://arxiv.org/abs/2503.01840), the third version, builds on this foundation by introducing a multi-layer fused feature representations from the target model, taking low, middle, and high-level embeddings directly into its drafting head. It also uses a context-aware, dynamic draft tree (inherited from [EAGLE-2](https://arxiv.org/abs/2406.16858)) to propose multiple chained hypotheses. These candidate tokens are then verified by the target model using parallel tree attention, effectively pruning invalid branches and improving both acceptance rate and throughput. Figure 3 shows this flow in action.

![A gif showing that the lightweight EAGLE head is not a standalone model. It drafts tokens from feature outputs taken from the target model’s layers, generates prediction trees, then feeds this back into the model for verification. 
](https://developer-blogs.nvidia.com/wp-content/uploads/2025/09/speculative-decoding-eagle-drafting-mechanism.gif)

Figure 3. The EAGLE-3 drafting mechanism generates a tree of candidate tokens from the target model

#### What is the EAGLE head?

Instead of using a separate, smaller model as in the draft-target approach, EAGLE-3 instead attaches a lightweight drafting component to the internal layers of the target model as an “EAGLE head.” The EAGLE head is typically made of a lightweight Transformer decoder layer followed by a final linear layer. It is essentially a miniature, stripped-down version of the building blocks that make up the main model.

This EAGLE head can generate not just a single sequence, but an entire tree of candidate tokens. This process is also instance-adaptive, where the head evaluates its own confidence as it builds the tree and stops drafting if the confidence drops below a threshold. This allows the EAGLE head to explore multiple generation paths efficiently, generating longer branches of predictable text and shorter ones for complex parts, all for the runtime cost of one forward pass of the target model.

### What is Multi-Token-Prediction in DeepSeek-R1?

Similar to EAGLE, Multi-Token Prediction (MTP) is a speculation technique used by many iterations of DeepSeek where the model learns to predict several future tokens at once rather than only the immediate next token. MTP uses a multi-head method where each head acts as a token drafter. The first head attached to the model guesses the first draft token, another guesses the one after that, another the third, and so on. The main model then checks those guesses in order and keeps the longest prefix that matches. This method naturally removes the need for a separate drafting model.

In essence, this technique is similar to EAGLE-style speculative decoding where both propose multiple tokens for verification. However, it differs in how proposals are formed: MTP uses specialized multi-token prediction heads, whereas EAGLE uses a single head that extrapolates internal feature states to construct candidates.

## How to implement speculative decoding

You can use the [NVIDIA TensorRT-Model Optimizer API](https://github.com/NVIDIA/TensorRT-Model-Optimizer) to apply speculative decoding to your own models. Follow the steps described below to convert a model to use EAGLE-3 speculative decoding using the [Model Optimizer Speculative Decoding module](https://nvidia.github.io/TensorRT-Model-Optimizer/reference/generated/modelopt.torch.speculative.html#module-modelopt.torch.speculative).

**Step 1**: Load the original Hugging Face model.

```
import transformers
 
import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import EAGLE3_DEFAULT_CFG
 
mto.enable_huggingface_checkpointing()
 
# Load original HF model
base_model = "meta-llama/Llama-3.2-1B"
model = transformers.AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype="auto", device_map="cuda"
)
```

**Step 2:** Import the default config for EAGLE-3 and convert it using `mtsp`.

```
# Read Default Config for EAGLE3
config = EAGLE3_DEFAULT_CFG["config"]
 
# Hidden size and vocab size must match base model
config["eagle_architecture_config"].update(
    {
        "hidden_size": model.config.hidden_size,
        "vocab_size": model.config.vocab_size,
        "draft_vocab_size": model.config.vocab_size,
        "max_position_embeddings": model.config.max_position_embeddings,
    }
)
 
# Convert Model for eagle speculative decoding
mtsp.convert(model, [("eagle", config)])
```

Check out the hands-on tutorial that expands this demo into a deployable end-to-end speculative decoding fine‑tuning pipeline in the [TensorRT-Model-Optimizer/examples/speculative\_decoding](https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/speculative_decoding/example.ipynb) GitHub repo.

## How does speculative decoding impact inference latency?

The core latency bottleneck in standard autoregressive generation is the fixed, sequential cost of each step. If a single forward pass (loading weights and computing a token) takes 200 milliseconds, generating three tokens will always take 600 ms (three sequential steps multiplied by 200 ms). The user experiences this delay as distinct cumulative waiting periods.

Speculative decoding can collapse these multiple waiting periods into one. By using a fast draft mechanism to speculate two candidate tokens then verifying them all in a single 250 ms forward pass, the model can generate three tokens (two speculations plus one base model generation) in 250 ms versus 600 ms. This concept is illustrated in Figure 4.

![A gif showing a base model (top) using standard autoregressive generation generating a single token in each 200 ms pass, taking 600 ms to generate three. A model with speculative decode (bottom) took slightly longer on one pass (250 ms), but generated three tokens in a single pass.
](https://developer-blogs.nvidia.com/wp-content/uploads/2025/09/speculative-decoding-generation-with-without.gif)

Figure 4. Generation with and without speculative decoding

Instead of watching the response appear word by word, the user sees it materialize in much faster, multi-token chunks. This is particularly noticeable in interactive applications like chatbots, where a lower response latency creates a much more fluid and natural conversation. Figure 5 simulates a hypothetical chatbot with speculative decode on and off.

![A gif with side-by-side chatbot outputs labeled ‘Speculative Decoding Off’ (left) and ‘Speculative Decoding On’ (right). The chatbot on the right shows how speculative decoding reduces the time it takes to generate each token or batch of tokens, shortening the user's waiting period. This makes the chatbot feel more responsive, fluid, and natural to interact with.
](https://developer-blogs.nvidia.com/wp-content/uploads/2025/09/speculative-decoding-on-off.gif)

Figure 5. A chatbot with speculative decoding on (right) generates text much faster than with speculative decoding off (left)

## Get started with speculative decoding

Speculative decoding is becoming a fundamental strategy for accelerating LLM inference. From the basics of draft–target generation and parallel verification to advanced methods like EAGLE-3, these approaches address the core challenge of idle compute during sequential token generation.

As workloads scale and demand grows for both faster response times and better system efficiency, techniques like speculative decoding will play an increasingly central role. Pairing these methods with frameworks such as [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), SGLANG, and vLLM ensures that developers can deploy models that are more performant, more practical, and more cost-effective in real-world environments.

Ready to get started? Check out the Jupyter notebook tutorial in the [TensorRT-Model-Optimizer/examples/speculative\_decoding](https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/speculative_decoding/example.ipynb) GitHub repo to try applying speculative decoding to your own model.