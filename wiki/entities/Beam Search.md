---
tags: [llm, generation, decoding, algorithms, optimization]
aliases: [集束搜索, 束搜索]
date: 2026-06-02
sources: ["[[raw/Recommendation/深入理解 Beam Search.md]]"]
---

# Beam Search

**Beam Search (集束搜索 / 束搜索)** is a widely used heuristic search algorithm in auto-regressive sequence generation tasks for Large Language Models (LLMs) and Machine Translation. It acts as an optimal middle ground between greedy decoding and exhaustive search.

---

## The Core Problem

During text generation, an LLM predicts one token at a time. The model needs to find the most probable sequence of tokens $Y^* = (y_1, y_2, \dots, y_T)$ given the input prompt.

### 1. Greedy Search (`num_beams=1`)
*   **Mechanism:** Selects the single token with the highest probability at each step:
    $$y_t = \arg\max_{w \in V} P(w | y_{<t})$$
*   **Pros:** Fast ($O(T)$ complexity) and memory-efficient.
*   **Cons:** Myopic (short-sighted). It cannot backtrack. If it chooses a suboptimal token early on, it is locked into a bad path, leading to poor global sequence quality.

### 2. Exhaustive Search
*   **Mechanism:** Evaluates all possible paths of length $T$ across the vocabulary $V$ and selects the path with the highest joint probability.
*   **Pros:** Guarantees finding the globally optimal sequence.
*   **Cons:** The Exponential Wall. For a vocabulary size $|V|$ (typically 32,000 to 100,000+), the search space has a complexity of $O(|V|^T)$. Generating just 10 tokens would require evaluating $32000^{10} \approx 1.1 \times 10^{45}$ combinations, which is computationally impossible.

---

## The Solution: Beam Search

Beam Search addresses both issues by maintaining a fixed number of candidate sequences (called the **beam width** $k$, or `num_beams`) at each generation step.

### 1. Step-by-Step Algorithm
1.  **Initialize:** Start with an initial candidate set containing only the start token (or input prompt):
    $$B_0 = \{ \text{prompt} \}$$
2.  **Expand:** At step $t$, expand each of the $k$ active candidate sequences in $B_{t-1}$ by predicting all possible next tokens from vocabulary $V$. This produces $k \times |V|$ candidate sequences.
3.  **Compute Scores:** Calculate the cumulative probability score for each of the $k \times |V|$ paths.
4.  **Prune (Top-K Selection):** Select only the $k$ highest-scoring sequences to form the new active candidate set $B_t$. Discard the rest.
5.  **Terminate:** Stop when all active sequences generate the end-of-sequence token (`<eos>`) or the sequence reaches the maximum length $T$.
6.  **Select Final Path:** Choose the sequence with the highest final score from the completed list.

---

## Step-by-Step Execution Trace (Concrete Example)

To illustrate how Beam Search works in practice, let's walk through an execution trace using:
*   **Beam Width ($k$):** 2
*   **Vocabulary ($V$):** $\{A, B, C, \langle\text{eos}\rangle\}$

---

### Step 1: Initialization & First Token Expansion
At the start, the candidate pool contains only the empty prefix/prompt sequence. We retrieve the vocabulary probability distribution for the first token:

| Candidate Sequence | Next Token Probability | Cumulative Joint Probability | Status |
| :--- | :--- | :--- | :--- |
| **$A$** | $0.4$ | $P(A) = 0.4$ | **Selected (Top 1)** |
| **$B$** | $0.3$ | $P(B) = 0.3$ | **Selected (Top 2)** |
| $C$ | $0.2$ | $P(C) = 0.2$ | Discarded |
| $\langle\text{eos}\rangle$ | $0.1$ | $P(\langle\text{eos}\rangle) = 0.1$ | Discarded |

*   **Active Beam Set ($B_1$):** $\{A \ (0.4), B\ (0.3)\}$

---

### Step 2: Parallel Expansion of Active Beams
We now expand both active candidates ($A$ and $B$) separately by querying the model for the probability distribution of the next token *conditioned on the prefix*:

#### Branch 1: Expanding prefix $A$ (with prior $P(A) = 0.4$)
Conditional next-token probabilities: $\{A: 0.3, B: 0.1, C: 0.4, \langle\text{eos}\rangle: 0.2\}$
*   $P(AA) = 0.4 \times 0.3 = 0.12$
*   $P(AB) = 0.4 \times 0.1 = 0.04$
*   $P(AC) = 0.4 \times 0.4 = \mathbf{0.16}$
*   $P(A\langle\text{eos}\rangle) = 0.4 \times 0.2 = 0.08$

#### Branch 2: Expanding prefix $B$ (with prior $P(B) = 0.3$)
Conditional next-token probabilities: $\{A: 0.5, B: 0.1, C: 0.3, \langle\text{eos}\rangle: 0.1\}$
*   $P(BA) = 0.3 \times 0.5 = \mathbf{0.15}$
*   $P(BB) = 0.3 \times 0.1 = 0.03$
*   $P(BC) = 0.3 \times 0.3 = 0.09$
*   $P(B\langle\text{eos}\rangle) = 0.3 \times 0.1 = 0.03$

#### Selection Pool (Total $2 \times 4 = 8$ candidates)
We sort all 8 combinations and select the top $k=2$ highest cumulative probabilities:
1.  **$AC$** ($0.16$) $\rightarrow$ **Selected**
2.  **$BA$** ($0.15$) $\rightarrow$ **Selected**
3.  $AA$ ($0.12$) $\rightarrow$ Discarded
4.  $BC$ ($0.09$) $\rightarrow$ Discarded
5.  $A\langle\text{eos}\rangle$ ($0.08$) $\rightarrow$ Discarded
6.  $AB$ ($0.04$) $\rightarrow$ Discarded
7.  $BB$ ($0.03$) $\rightarrow$ Discarded
8.  $B\langle\text{eos}\rangle$ ($0.03$) $\rightarrow$ Discarded

*   **Active Beam Set ($B_2$):** $\{AC \ (0.16), BA\ (0.15)\}$

---

### Step 3: Deepening the Search Space
We repeat the expansion step for the new active candidate set:

#### Branch 1: Expanding prefix $AC$ (with prior $P(AC) = 0.16$)
Conditional next-token probabilities: $\{A: 0.1, B: 0.4, C: 0.3, \langle\text{eos}\rangle: 0.2\}$
*   $P(ACA) = 0.16 \times 0.1 = 0.016$
*   $P(ACB) = 0.16 \times 0.4 = \mathbf{0.064}$
*   $P(ACC) = 0.16 \times 0.3 = 0.048$
*   $P(AC\langle\text{eos}\rangle) = 0.16 \times 0.2 = 0.032$ (Completed Sequence)

#### Branch 2: Expanding prefix $BA$ (with prior $P(BA) = 0.15$)
Conditional next-token probabilities: $\{A: 0.4, B: 0.2, C: 0.1, \langle\text{eos}\rangle: 0.3\}$
*   $P(BAA) = 0.15 \times 0.4 = \mathbf{0.060}$
*   $P(BAB) = 0.15 \times 0.2 = 0.030$
*   $P(BAC) = 0.15 \times 0.1 = 0.015$
*   $P(BA\langle\text{eos}\rangle) = 0.15 \times 0.3 = 0.045$ (Completed Sequence)

#### Selection Pool
Sorting all active sequences and filtering:
1.  **$ACB$** ($0.064$) $\rightarrow$ **Selected** (Active Beam)
2.  **$BAA$** ($0.060$) $\rightarrow$ **Selected** (Active Beam)
3.  $BA\langle\text{eos}\rangle$ ($0.045$) $\rightarrow$ Completed / Tracked in background
4.  $ACC$ ($0.048$) $\rightarrow$ Discarded
5.  $AC\langle\text{eos}\rangle$ ($0.032$) $\rightarrow$ Completed / Tracked in background

*   **Active Beam Set ($B_3$):** $\{ACB \ (0.064), BAA\ (0.060)\}$

This process iterates until the stop conditions (e.g., maximum length or all beams hit `early_stopping`) are met. At that point, all tracked completed candidates (such as $BA\langle\text{eos}\rangle$ and $AC\langle\text{eos}\rangle$) along with the final active beams are compared (often after applying length normalization) to select the single best sequence.

---

## Mathematical Formulations

### 1. Sequence Joint Probability
The joint probability of a sequence $Y = (y_1, y_2, \dots, y_T)$ is the product of conditional probabilities at each step:
$$P(Y) = \prod_{t=1}^{T} P(y_t | y_1, y_2, \dots, y_{t-1})$$

### 2. Numerical Underflow & Log-Likelihood
Multiplying multiple floating-point probabilities (each $\le 1.0$) causes severe **numerical underflow** for long sequences. To resolve this, we take the natural logarithm to convert products into sums:
$$S(Y) = \log P(Y) = \sum_{t=1}^{T} \log P(y_t | y_1, y_2, \dots, y_{t-1})$$
Since probabilities lie in $[0, 1]$, their log values are $\le 0$. Maximizing the sequence probability translates to maximizing the sum of log-likelihoods (i.e., making the sum as close to $0$ or as "least negative" as possible).

### 3. Length Normalization (Length Penalty)
Simply summing negative log-probabilities naturally penalizes longer sequences (more additions of negative numbers). To prevent the search from heavily biasing toward very short sentences, a **length penalty** coefficient $\alpha$ is introduced to normalize the score:
$$S_{\text{norm}}(Y) = \frac{1}{L^\alpha} \sum_{t=1}^{T} \log P(y_t | y_1, y_2, \dots, y_{t-1})$$
Where:
*   $L$ is the length of the generated sequence $Y$.
*   $\alpha$ is the length penalty hyperparameter (typically set between $0.6$ and $1.0$). A higher $\alpha$ encourages longer generations.

---

## Key Parameters in Production

In Hugging Face’s `transformers` library, Beam Search is configured via the following parameters in `generate()`:

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `num_beams` | `int` | The beam width ($k$). Larger values explore more space but increase compute. Commonly set to `3` to `5`. |
| `early_stopping` | `bool` / `str` | Controls when the search stops once completed candidates are found.<br>• `True`: Halts as soon as $k$ completed sequences are found.<br>• `False` / `'heuristic'`: Continues until no active sequence can mathematically beat the best completed candidate.<br>• `'never'`: Continues until all beams produce `<eos>` or max length is reached. |
| `no_repeat_ngram_size` | `int` | Prevents repetitive generation loops (e.g., repeating a phrase). If set to $N$, any token that would form a repeating $N$-gram is assigned a probability of $0$. |

---

## Production Code Example

Below is a standard Python implementation of Beam Search using the Hugging Face `transformers` library with the lightweight `distilgpt2` model:

```python
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set model download mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Load tokenizer and model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad_token to eos_token to handle batching if necessary
tokenizer.pad_token = tokenizer.eos_token

# Move to GPU if available and set to eval mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Input prompt
input_text = "Hello GPT"
inputs = tokenizer.encode_plus(input_text, return_tensors="pt", padding=True).to(device)

# Generate with Beam Search
beam_width = 5
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=50,
        num_beams=beam_width,         # Beam width (k)
        no_repeat_ngram_size=2,       # Prevent repeating 2-grams
        early_stopping=True,          # Stop when k completed sequences are found
        pad_token_id=tokenizer.eos_token_id
    )

# Decode and print output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:")
print(generated_text)
```
