---
tags:
  - llm
  - metric
  - pretraining
  - math
date: 2026-08-31
sources: ["[[Pre-training Large Language Models]]"]
---

# Perplexity (困惑度)

Perplexity (PPL) measures how "surprised" a language model is by a piece of text — formally, the exponentiated average negative log-likelihood per token. It doubles as (1) the standard **intrinsic evaluation metric for language models** and (2) a **data-quality signal** in pre-training corpus cleaning.

## 1. Definition

Given a token sequence $u_1, u_2, \ldots, u_T$ and a language model:

$$
\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(u_t \mid u_{<t})\right)
$$

Intuition: at each step, PPL is roughly the effective number of candidates the model is "hesitating between" — the geometric mean of the inverse probability assigned to the actual next token.

## 2. Interpretation

- **Low PPL** → the text is fluent, natural, predictable → (usually) high-quality language.
- **High PPL** → the text confuses the model → garbled text, machine-generated spam, broken grammar.
- **Suspiciously low PPL** can also be a red flag: highly repetitive text ("lol lol lol…") is trivially predictable and equally low-quality.

Examples: "The cat sat on the mat." → very low PPL; "asdkfj qweoiu zxcvbnm" → extremely high PPL.

## 3. Use in Pre-training Data Filtering

In the quality-filtering step of corpus preprocessing (see [[Pre-training Large Language Models]] Part 2), perplexity serves as a heuristic statistical feature:

1. Score every document with a (typically small) reference LM.
2. Drop documents with abnormally **high** PPL (unnatural/garbage text).
3. Optionally flag abnormally **low** PPL documents (repetitive content).

## 4. Caveats

- **Relative to the scoring model:** an English LM gives Chinese text systematically high PPL — multilingual corpora must be scored per-language with appropriate models.
- **Fluency ≠ factuality:** a grammatically perfect but factually false passage still has low PPL. Perplexity filtering is only one layer of quality control, complementing keyword rules and classifiers.

## 5. Dual Role

The same metric runs in both directions:

| Role | Fixed | Varies | Meaning of lower PPL |
|---|---|---|---|
| Model evaluation | Test set | The model | Better language modeling ability |
| Data filtering | Reference model | The documents | More natural/fluent text |

## Related Pages

- [[Pre-training Large Language Models]]: Chapter 15 summary where PPL appears as a quality-filtering heuristic.
- [[Tokenizer]]: PPL is computed over token sequences, so its absolute value depends on the tokenization granularity.
