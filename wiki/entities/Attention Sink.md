---
tags: ["entity", "attention", "llm", "long-sequence", "inference"]
aliases: ["Attention Sink", "attention sink effect"]
date: 2026-08-24
sources: ["[[wiki/research/LONGER Summary.md]]"]
---

# Attention Sink

**Attention Sink** is the empirical phenomenon, named and analyzed by **StreamLLM** (Xiao et al., 2024, *Efficient Streaming Language Models with Attention Sinks*), where deeper attention layers disproportionately dump attention mass onto the **first few tokens** of a sequence — regardless of those tokens' semantic importance.

## Why It Happens

Softmax forces attention weights over all positions to sum to 1. When no token is genuinely informative for a given query, the model still must allocate that probability mass somewhere. It learns to "park" attention on the initial tokens as a default no-op sink. A side effect: those early tokens accumulate large hidden-state norms and dominate the attention distribution.

## Why It Matters

- **Streaming / sliding-window LLMs:** StreamLLM showed that naive windowing (evicting old tokens, including the sink tokens) collapses fluency; keeping just a few **attention-sink tokens** alongside the recent window restores stable generation over unbounded streams.
- **Long-sequence attention stability:** in very long contexts, the sink effect warps attention diversity and degrades long-range dependency modeling.

## In Recommendation: LONGER's Global Tokens

[[LONGER]] (RecSys 2025) invokes the same effect from the other direction. Rather than *preserving* natural sink tokens, it **deliberately injects global tokens** (target item, UID, CLS, compressed interaction features) with a full attention receptive field. These act as **anchor points** that stabilize attention distributions over ultra-long (10K) behavior sequences — maintaining attention diversity and preserving long-range dependency modeling, counteracting the sink tendency under sparse attention.

---

## Related Wiki Pages
* [[LONGER Summary]] / [[LONGER]]: Uses global tokens as anchors to mitigate the sink effect in 10K-length recommendation sequences.
* [[Self-Attention Mechanism]]: The underlying attention math.
* [[KV Cache]]: Streaming LLM serving, where sink tokens must be retained.
