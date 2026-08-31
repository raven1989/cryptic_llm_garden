---
tags:
  - llm
  - tokenization
  - pretraining
date: 2026-08-31
sources: ["[[Pre-training Large Language Models]]"]
---

# Tokenizer (分词器)

Tokenization splits raw text into the token sequences a language model consumes. Modern Transformer LMs almost universally use **subword tokenizers**, which keep the vocabulary compact while guaranteeing any text can be represented (no OOV). This page consolidates the three mainstream subword algorithms (BPE, WordPiece, Unigram), the SentencePiece framework, and Chinese-tokenizer practice — discussed during the Chapter 15 ingest ([[Pre-training Large Language Models]], Part 2, Step 4).

## 1. BPE (Byte Pair Encoding, 字节对编码)

Originally a 1994 **data compression** algorithm; introduced to NLP by Sennrich et al. (2016) for NMT. The name describes the mechanism: repeatedly **encode (merge) the most frequent adjacent pair** into a new unit.

- **Training:** start from characters; each round count all adjacent pairs (weighted by word frequency) and merge the **most frequent** one; stop at the target vocabulary size (e.g., GPT-3: 50,257). Output: a priority-ordered merge-rule table.
- **Encoding:** split the new word into characters, then apply merge rules **in learned priority order**. Unseen words decompose into familiar subwords ("lowest" → `low` + `est_`).
- **Why LLMs love it:** open vocabulary (no OOV), compact vocab (small embedding matrix), frequency-adaptive (common content costs fewer tokens → faster inference, better context-window utilization).
- **Byte-level BPE (BBPE):** GPT-2's key improvement — run BPE directly on UTF-8 **bytes**. Base vocabulary is fixed at 256 bytes; any language/symbol/emoji is encodable — a truly universal tokenizer. Standard for GPT series, LLaMA, Qwen, DeepSeek.

## 2. WordPiece

Google's algorithm, popularized by **BERT** (2018). Same bottom-up merging procedure as BPE, differing only in the **merge criterion** — instead of raw frequency, merge the pair maximizing likelihood gain:

$$
\text{score}(x, y) = \frac{\text{freq}(x, y)}{\text{freq}(x) \times \text{freq}(y)}
$$

This is the pointwise mutual information (PMI) form: BPE asks "*which pair occurs most?*", WordPiece asks "*which pair co-occurs more than their independent frequencies predict?*" — it prefers "sticky" pairs whose parts rarely appear apart (e.g., `zq` beats a frequent but promiscuous `ab`).

- **Encoding:** greedy **longest-match** left-to-right; continuation subwords carry BERT's signature `##` prefix ("unaffable" → `un` + `##aff` + `##able`), making segmentation losslessly reversible.
- Character-based (no byte fallback), so OOV is theoretically possible (→ `[UNK]`); mitigated by a large character set in practice.

## 3. Unigram (Unigram Language Model)

Proposed by Taku Kudo (2018, also SentencePiece's author); used by **T5, ALBERT, XLNet**. The direction is **reversed: top-down pruning instead of bottom-up merging**.

1. Start from a huge candidate vocabulary (all frequent substrings, e.g., via a BPE run).
2. Each round: for every token $x$, compute the **loss** of removing it — the increase in corpus negative log-likelihood:

$$
\text{loss}(x) = \mathcal{L}(V \setminus \{x\}) - \mathcal{L}(V), \qquad \mathcal{L}(V) = -\sum_{S} \log \sum_{\mathbf{x} \in \text{segmentations}(S)} \prod_{x_i \in \mathbf{x}} p(x_i)
$$

   where token probabilities follow a unigram LM ($p(x) = \text{count}(x)/\sum \text{count}$), a sentence's probability **sums over all possible segmentations** (Viterbi forward), and EM re-estimates probabilities each round.
3. Permanently delete the batch with the **smallest** loss (SentencePiece default: 10–20% per round, `shrink_factor`); repeat until the vocab shrinks to target size.

**Loss intuition:** deleting a token with good "backups" (e.g., `ab` while `a`, `b` remain) barely changes sentence probability → small loss → delete; deleting an irreplaceable token forces fallback segmentations with probability drops of orders of magnitude → huge loss → keep. Single characters/base units are pinned (never deleted) so every sentence stays segmentable ($P'(S) = 0 \Rightarrow \text{loss} = +\infty$ otherwise).

**Unique capability — subword sampling:** because Unigram keeps a full probabilistic model, it can either take the Viterbi-best segmentation (deterministic) or **sample** segmentations by probability — a natural data augmentation used by XLNet (subword regularization). BPE/WordPiece segmentations are unique and deterministic.

### Symmetry between WordPiece and Unigram

| | WordPiece | Unigram |
|---|---|---|
| Operation | **add** a merge | **delete** a token |
| Score | likelihood **gain** of merging $\frac{P(x,y)}{P(x)P(y)}$ | likelihood **loss** of deletion $\Delta\text{NLL}$ |
| Selection | max gain per round | min loss per round |

## 4. Algorithm Comparison

| | BPE | WordPiece | Unigram |
|---|---|---|---|
| Direction | bottom-up merge | bottom-up merge | top-down prune |
| Criterion | highest frequency | max likelihood gain (PMI) | min likelihood loss on removal |
| Probabilistic | no (rule table) | no | yes (per-token probabilities) |
| Encoding | apply rules in order | greedy longest match | Viterbi best / sampling |
| Multiple segmentations | no | no | yes (subword regularization) |
| Representative models | GPT, LLaMA, Qwen, DeepSeek | BERT family | T5, ALBERT, XLNet, mT5 |

## 5. SentencePiece

Google's open-source tokenizer **framework** (Kudo, 2018) — BPE/WordPiece/Unigram are algorithms; SentencePiece is the production tooling that trains and serves them. Recommended by Chapter 15 for training custom tokenizers. Core designs:

1. **No pre-tokenization:** trains directly on raw sentences. Whitespace is escaped to a normal character `▁` (U+2581), so word-initial vs. word-internal subwords are distinguished natively, decoding is losslessly reversible, and spaceless languages (Chinese/Japanese/Thai) go through the exact same pipeline as English — no external segmenter (e.g., Jieba) needed.
2. **Pluggable algorithms:** `--model_type=bpe` (LLaMA, GLM) or `--model_type=unigram` (default; T5, XLNet).
3. **Coverage & byte fallback:** `--character_coverage` (0.9995) keeps rare characters out of the vocab; `--byte_fallback=true` adds all 256 bytes so any input is encodable (LLaMA uses this — unseen Chinese characters degrade to 3 byte tokens, hurting Chinese efficiency).
4. **Self-contained artifact:** one `*.model` file (vocab + rules/probabilities + NFKC normalization), fast C++/Python inference; extras like BPE-dropout for regularization.

Ecosystem: HuggingFace `tokenizers` (Rust, HF integration), OpenAI `tiktoken` (inference-only BBPE).

## 6. Chinese Tokenization Practice

Chinese has no whitespace word boundaries, but modern LLMs **skip traditional word segmentation entirely** and apply subword tokenizers directly on raw text:

| Model | Tokenizer | Note |
|---|---|---|
| Qwen | byte-level BPE, ~152K vocab | Chinese-optimized; high Chinese compression |
| DeepSeek | byte-level BPE, ~129K vocab | trained on multilingual mix |
| GLM / ChatGLM | SentencePiece | bilingual optimization |
| LLaMA | SentencePiece BPE, 32K | English-centric; Chinese chars fall back to bytes (up to 3 tokens per character → 2–3× token cost) |

The efficiency gap comes from whether the vocab was **trained on sufficient Chinese data** (high-frequency characters/words like 我们、人工智能 become single tokens) — the canonical example of Chapter 15's point that a corpus-tailored tokenizer beats a reused one. Traditional Chinese word-segmentation tools (Jieba, pkuseg, THULAC, LTP, HanLP) target linguistic word boundaries for classical NLP pipelines, not LLMs.

## Related Pages

- [[Pre-training Large Language Models]]: tokenization appears as Part 2, Step 4 of the preprocessing pipeline.
- [[Perplexity]]: PPL values are tokenization-dependent — absolute PPL is only comparable under the same tokenizer.
- [[Fine-tuning]]: tokenizer choice carries over from pre-training into all downstream stages.
