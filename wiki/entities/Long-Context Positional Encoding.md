---
tags:
  - llm
  - positional-encoding
  - long-context
  - rope
  - math
aliases:
  - RoPE Length Extrapolation
  - 长上下文位置编码
date: 2026-08-31
sources: ["[[Pre-training Large Language Models]]", "[[RoPE]]"]
---

# Long-Context Positional Encoding (长上下文位置编码)

How to make [[RoPE]]-based LLMs work beyond their trained context window. This page consolidates the full discussion: why RoPE fails to extrapolate despite being a *relative* encoding, the complete RoPE formula, and the method families — Position Interpolation, position truncation, NTK-aware scaling, Dynamic NTK, YaRN, and base scaling — capped by the DeepSeek-V4 1M-context case study. For RoPE's base mathematics see [[RoPE]]; for the high-level comparison with other schemes see [[Positional Encoding]].

## 1. The Paradox: RoPE Is Relative, Yet Has a Length Limit

RoPE's relative property is a **mathematical identity**: rotating the query at absolute position $m$ and the key at $n$ yields an attention score depending only on $n - m$:

$$
\langle \mathbf{R}_{\Theta, m}\,\mathbf{q}_m,\ \mathbf{R}_{\Theta, n}\,\mathbf{k}_n \rangle
= \mathbf{q}_m^\top \mathbf{R}_{\Theta,\, n-m}\, \mathbf{k}_n
$$

This holds for *any* $m, n$ — so RoPE has no hard length ceiling in the formula. The failure is not mathematical but **statistical**: the identity only rewrites the score in terms of $\mathbf{R}_{\Theta,\, n-m}$, whose per-subspace rotation angles $(n-m)\theta_i$ the model has only ever seen within the training window $[0, L_{\text{train}}]$. Beyond it, attention logits drift out of distribution and performance collapses.

**Why the low-frequency subspaces are the killer.** Each 2D subspace $i$ rotates by $\phi_i(m) = m\theta_i$ with $\theta_i = b^{-2i/d}$. With $d=128$, $b=10000$, $L_{\text{train}}=4096$:

| Subspace | $\theta_i$ | Distance for one full turn $2\pi/\theta_i$ | Angle seen in training |
|---|---|---|---|
| $i=1$ (highest freq) | $\approx 0.866$ | $\approx 7$ tokens | hundreds of full turns — all angles familiar |
| $i=32$ | $\approx 0.0237$ | $\approx 265$ tokens | many turns |
| $i=64$ (lowest freq) | $10^{-4}$ | $\approx 62{,}832$ tokens | $\le 4095 \times 10^{-4} \approx 0.41$ rad — **not even 1/15 of a turn** |

High-frequency subspaces have already swept all of $[0, 2\pi]$ during training, so extrapolation just "keeps turning" over familiar angles. Low-frequency subspaces — the ones carrying long-range dependency — enter completely untrained angle intervals (e.g., $[0.41, 0.82]$ rad) the moment the context doubles. Hence the **cliff-like** degradation, not graceful decay.

**Contrast with [[ALiBi]]:** ALiBi adds a linear distance penalty $-m|i-j|$ directly to attention logits — the *form* never changes with distance, only the magnitude grows linearly, so no untrained state ever appears. RoPE's problem is that position is baked into the Q/K vectors' **rotation state**, so extrapolation shifts the distribution of everything downstream.

## 2. The Complete RoPE Formula (notation pinned)

- $m$: the token's **absolute position index** ($0, 1, \ldots, L-1$); $i$: the **dimension-pair index** ($1 \ldots d/2$); $\theta_i = b^{-2i/d}$: angular frequency of subspace $i$; $\phi_i(m) = m\theta_i$: the rotation angle applied to position $m$ in subspace $i$.

Chain of computation:

1. **Standard projection** (no position yet): $\mathbf{q}_m = \mathbf{W}_q \mathbf{x}_m$, $\mathbf{k}_n = \mathbf{W}_k \mathbf{x}_n$.
2. **Pair up dimensions** into $d/2$ 2D subspaces: $(q^{(1)}, q^{(2)}), (q^{(3)}, q^{(4)}), \ldots$
3. **Rotate each pair** by $m\theta_i$:
   $\begin{pmatrix} \tilde{q}^{(2i-1)} \\ \tilde{q}^{(2i)} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} q^{(2i-1)} \\ q^{(2i)} \end{pmatrix}$, i.e. $\tilde{\mathbf{q}}_m = \mathbf{R}_{\Theta, m}\,\mathbf{q}_m$ with $\mathbf{R}_{\Theta,m}$ block-diagonal.
4. **Inner product** — absolute positions cancel via orthogonality $\mathbf{R}_m^\top \mathbf{R}_n = \mathbf{R}_{n-m}$, leaving only $n-m$.

Efficient implementation (LLaMA-style) avoids the sparse matrix with element-wise products: $\tilde{\mathbf{q}}_m = \mathbf{q}_m \otimes \mathbf{c}_m + \mathbf{q}_m^{\text{rot}} \otimes \mathbf{s}_m$ where $\mathbf{c}_m, \mathbf{s}_m$ repeat each $\cos m\theta_i / \sin m\theta_i$ twice and $\mathbf{q}^{\text{rot}}$ swaps/negates each pair.

## 3. Method Family 1: Position Interpolation & Truncation

All methods answer one question: when $L_{\text{infer}} > L_{\text{train}}$, how do we keep rotation angles inside the trained range while preserving position discriminability?

**Position Interpolation (PI, Meta 2023):** linearly compress position indices, $m' = m \cdot L_{\text{train}} / L_{\text{infer}}$. All angles fall back into the trained range. Cost: adjacent-token angle differences shrink by the same factor — **position resolution drops**, hurting the high-frequency subspaces that encode local precision. Zero-shot PI is mediocre, but it converts an OOD problem into an in-distribution one: **~1000 steps of fine-tuning** suffice, vastly cheaper than long-context training from scratch.

**Position truncation / sliding-window attention (Mistral):** cap the visible relative distance with an attention mask ($n - m \le W \le L_{\text{train}}$), so rotation angles never leave the trained range. KV cache is bounded and extrapolation is a non-issue, but remote information can only propagate indirectly layer by layer (receptive field $\approx N \times W$) — poor at needle-in-haystack recall. The **Λ-shaped window** (StreamingLLM / [[Attention Sink]]) is a sibling: keep initial sink tokens + recent neighbors, discard the middle.

## 4. Method Family 2: Frequency Scaling — NTK-aware, Dynamic NTK, YaRN

Rewriting PI as $\frac{m}{s}\theta_i = m \cdot \frac{\theta_i}{s}$ (extension factor $s = L_{\text{infer}}/L_{\text{train}}$) reveals that PI is just *uniform* frequency scaling. The NTK family scales frequencies **non-uniformly per dimension**: high frequencies barely need scaling (all angles seen), low frequencies need full scaling (never completed a turn).

**NTK-aware scaling:** change the base $b \to b' = b \cdot s^{d/(d-2)}$, giving $\theta'_i = \theta_i \cdot s^{-2i/(d-2)}$ — an exponential ramp that leaves high-$i$ (high-frequency) dims nearly untouched while compressing low-frequency dims by $\approx 1/s$. One-line change, zero-shot usable. Drawbacks: the highest frequencies still get slightly compressed (local resolution suffers at large $s$), and the fixed base penalizes short sequences too.

**Dynamic NTK:** make the factor length-dependent, $s_{\text{now}} = \max(1, L_{\text{now}}/L_{\text{train}})$, recomputing $b'$ as the sequence grows. Short sequences are completely lossless; compression ramps in only as needed. Drawbacks: position encodings now depend on current length, **breaking KV cache** (must cache un-rotated K and re-rotate); still the same exponential ramp family.

**YaRN (Yet another RoPE extensioN, 2023 — current mainstream: Qwen, GLM):** two refinements.
1. **Piecewise frequency bands** keyed on wavelength $\lambda_i = 2\pi/\theta_i$ vs. $L_{\text{train}}$: high-frequency band ($\lambda_i \ll L_{\text{train}}$) → **no interpolation at all**; low-frequency band ($\lambda_i \gg L_{\text{train}}$) → **full PI** ($\theta_i/s$); transition band → linear ramp blending the two. High-frequency resolution is fully preserved, low-frequency protection is complete, and the ramp avoids discontinuities.
2. **Attention temperature correction:** interpolation shrinks relative angle differences, flattening softmax (entropy rises, attention wanders). YaRN scales logits by $\sqrt{1/t} \approx 0.1 \ln s + 1$ to re-sharpen — a zero-cost but significant boost.

Works zero-shot; a few hundred steps of long-text fine-tuning push it to 128K–1M.

## 5. Method Family 3: Enlarge the Base at Training Time

Since the low-frequency problem is "$L_{\text{train}} \cdot \theta_i$ too small", train with a **larger base** (10000 → 500K in LLaMA-3, 1M in Qwen2): every $\theta_i$ shrinks, so the same training length covers a larger fraction of a turn and extrapolation headroom grows natively. Costs low-end resolution, so it must be paired with long-text continued training.

## 6. Comparison Table

| Method | What changes | Training | Strength | Cost |
|---|---|---|---|---|
| Naive extrapolation | nothing | no | zero cost | low-frequency OOD → collapse |
| Position Interpolation | $m \to m/s$ | ~1K steps fine-tune | simple; angles back in range | high-freq resolution loss |
| Sliding window / truncation | attention mask caps $n-m$ | no | never OOD; bounded cache | indirect long-range access |
| NTK-aware | $b \to b'$ (exponential ramp) | zero-shot ok | preserves high freq, compresses low | high-freq degradation at large $s$; short-seq penalty |
| Dynamic NTK | $b'$ recomputed per length | zero-shot ok | short sequences lossless | breaks KV cache |
| **YaRN** | per-band interpolation + temperature | zero-shot ok, light fine-tune best | full high-freq preservation; 128K–1M | slightly complex |
| Larger base | train with $b$ = 500K–1M | needs continued training | native headroom | low-end resolution coarser |

## 7. Case Study: DeepSeek-V4's 1M Context (arXiv:2606.19348, 2026)

At the 1M scale, position-encoding tricks alone are insufficient — the bottleneck shifts to attention FLOPs and KV cache growing linearly/quadratically with length. DeepSeek-V4 combines three layers:

**Layer 1 — Positional encoding: Partial RoPE.** RoPE is applied only to the **last 64 dimensions** of each query and KV entry (the rest are NoPE, learned implicitly). Because V4's KV entries serve as both keys and values, attention outputs would inherit absolute position from the weighted sum of KV entries — countered by applying RoPE with position $-i$ on the last 64 dims of each output $\mathbf{o}_{t,i}$, so outputs carry only *relative* position.

**Layer 2 — Architecture: hybrid CSA + HCA.** Compressed Sparse Attention compresses every $m$ tokens' KV into one entry (sequence → $1/m$) and applies DeepSeek Sparse Attention's **lightning indexer** for sparse top-k selection over compressed entries (an industrial realization of **token selection**); HCA compresses even harder ($m' \gg m$). A supplementary **sliding-window branch** of uncompressed recent tokens restores local fine-grained access that block compression destroys (the **Λ-window/truncation** idea), and learnable **attention sink** logits let heads' total attention mass go below 1. Result: KV cache ≈ **2%** of a BF16 GQA8 baseline at 1M.

**Layer 3 — Training schedule: curriculum + staged sparsity.** Sequence length follows a **data curriculum** $4\text{K} \to 16\text{K} \to 64\text{K} \to 1\text{M}$ (so 1M is *natively trained*, not extrapolated); the first **1T tokens use dense attention**, sparse attention is introduced at 64K (with a short lightning-indexer warmup) and kept for the rest. Flash: 32T tokens, peak LR $2.7\times10^{-4}$; Pro: 33T tokens, peak LR $2.0\times10^{-4}$, longer dense stage; both decay to 1/10 with a cosine schedule.

**Honest limits:** retrieval stays stable within 128K; visible degradation beyond it (512K is the inflection). 1M is an *addressable* window, but understanding quality still decays with length — exactly the chapter's caveat that fluent long-text generation ≠ equal long-text understanding.

**Mapping to Chapter 15 concepts:** data curriculum (4K→1M), position-encoding extension (Partial RoPE + output de-rotation), Λ-window/truncation (sliding-window branch), token selection (lightning indexer over compressed KV), stability (Anticipatory Routing for MoE loss spikes), optimization (Muon+AdamW, batch ramp, warmup+cosine decay).

## 8. The Takeaway Recipe

Modern long-context systems are **combinations**: a large base (500K–1M) + YaRN/NTK-style per-band interpolation + long-text continued pre-training (short-window bulk tokens → long-window few tokens) + inference-side sliding window or token selection. Position encoding answers "*can it be represented*", compression architectures answer "*can it be afforded*", and the data curriculum answers "*can it be learned*".

## Related Pages

- [[RoPE]]: The base rotary encoding whose extrapolation this page fixes.
- [[ALiBi]]: Linear-bias encoding with native extrapolation — the contrast case.
- [[Positional Encoding]]: High-level overview of all encoding schemes.
- [[Pre-training Large Language Models]]: Chapter 15's long-context modeling section (Part 4), which this page deepens.
- [[Attention Sink]]: The sink-token effect used by Λ-window methods and DeepSeek-V4.
