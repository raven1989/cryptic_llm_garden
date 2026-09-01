---
tags:
  - llm
  - evaluation
  - benchmark
  - research
aliases:
  - LLM 性能评估方法
  - LLM Evaluation Benchmarks
date: 2026-08-31
sources: ["[[LLM技术报告]]"]
---

# LLM Performance Evaluation Methods

This page records every LLM (and multimodal LLM) evaluation method described in **§5.2 性能评估方法和指标** of the *LLM技术报告*. It catalogs the automated metrics, the task-specific benchmark suites (评测集) with the exact capability each one probes (评测方面), human evaluation, LLM-as-a-Judge techniques, and the cross-cutting challenges.

Evaluation of LLMs is a complex, multi-dimensional task. No single metric captures a model's full capability, so practice combines automated metrics (fast, scalable, coarse) with human evaluation (slow, expensive, precise) and, increasingly, LLM-as-a-Judge (efficient middle ground). The aspects under scrutiny span language understanding, generation quality, factual accuracy, safety, and robustness — with a growing emphasis on real-world performance.

## 1. Automated Evaluation Metrics (自动化评估指标, §5.2.1)

Automated metrics measure performance on specific tasks, but they cannot fully capture complex capabilities or human preferences. They fall into two groups: classic NLP metrics and task-specific benchmarks.

### 1.1 Classic NLP Metrics

| Metric | Task | What it measures (评测方面) |
|---|---|---|
| **困惑度 Perplexity** | Language modeling | Fluency & grammar — how well the model predicts the next word; lower is better. |
| **BLEU** | Machine translation | n-gram overlap between model output and reference translation. |
| **ROUGE** | Text summarization | Overlap between generated and reference summaries (recall-oriented). |
| **F1 / 准确率 / 精确率 / 召回率** | Classification, NER | Classic supervised NLP task performance. |

### 1.2 Task-Specific Benchmarks (评测集 → 评测方面)

These benchmark suites are the heart of the answer to "which evaluation set tests which capability":

| Benchmark (评测集) | Capability tested (评测方面) |
|---|---|
| **MMLU** (Massive Multitask Language Understanding) | Knowledge & reasoning across **57 disciplines** (history, law, math, medicine, …). |
| **HellaSwag** | **Commonsense reasoning** — pick the most plausible continuation among seemingly-reasonable options. |
| **TruthfulQA** | **Factual accuracy** — exposes tendency to generate misinformation (hallucination). |
| **GSM8K** | **Math problem solving** — grade-school math word problems. |
| **HumanEval / MBPP** | **Code generation & completion**. |
| **BIG-bench** | **Broad capability** — 200+ tasks probing a wide range of abilities. |
| **SuperGLUE** | General **language understanding** (reading comprehension, inference, …). |
| **ARC** (AI2 Reasoning Challenge) | **Scientific reasoning**. |
| **DROP** | **Reading comprehension & logical reasoning**. |
| **Math 基准** | **Mathematical reasoning**. |

### 1.3 LLM Leaderboards (排行榜)

Institutions and communities maintain public leaderboards that rank models across many benchmarks for easy comparison:

- **Hugging Face Open LLM Leaderboard**
- **LiveBench**
- **Vellum AI LLM Leaderboard**

These update regularly to reflect the latest model releases.

## 2. Human Evaluation (人工评估, §5.2.2)

The most reliable but most expensive method — especially for generation quality, logical coherence, creativity, and safety. It captures nuances automated metrics miss.

| Method | Aspect evaluated (评测方面) |
|---|---|
| **Human Preference Ranking** | Relative quality — humans rank responses from different models; the foundation of reward-model training in RLHF. |
| **Dialogue Quality Evaluation** | Fluency, coherence, relevance, informativeness, user satisfaction of conversational systems. |
| **Factuality Evaluation** | Factual accuracy; detecting & correcting **hallucination** (requires manual verification). |
| **Safety Evaluation** | Harmful, biased, discriminatory, or unsafe content (hate speech, misinformation, privacy leaks). |

## 3. LLM-as-a-Judge (LLM 作为评估者, §5.2.3)

Using a strong LLM to evaluate another LLM's output via carefully crafted prompts — improves efficiency at scale.

| Technique | Mechanism |
|---|---|
| **G-Eval** | LLM generates evaluation criteria & scores, then grades model output. |
| **Reason-then-Score (RTS)** | LLM first reasons/analyzes the output, then gives a score. |

**Limitations:** potential bias, lack of rigorous fact-checking, and the judge's own hallucination.

## 4. Benchmark Comparison by Modality (§5.2.4)

Evaluation also breaks down by modality and by interaction type:

- **Language-understanding benchmarks:** MMLU, SuperGLUE, BigBench (commonsense, encyclopedic knowledge, language inference); GSM8K & Math (mathematical reasoning); ARC (scientific reasoning); DROP (reading comprehension & logical reasoning). Closed models (GPT-4, Claude 3, Gemini) typically lead; open models (LLaMA 3.1, Mistral Mixtral 8×22B) are competitive within their class — and Qwen 2.5-Max reportedly beat GPT-4o on several, signaling open-source catching up.
- **Multimodal benchmarks:** MMBench, MME (Multimodal Evaluation) for image+text or video+text understanding & generation. GPT-4V, Gemini 1.5 Pro, Qwen2-VL show strong cross-modal reasoning/description. No unified global ranking yet, but large public leaderboards (e.g., LMSYS MMBench) track VQA and cross-modal retrieval.
- **Dialogue evaluation & leaderboards:** Chatbot Arena ranks chatbots via user comparison votes using the **Bradley-Terry model**. OpenLLM Leaderboard, BentoML, etc. rank models on answer quality, speed, and cost. GPT-4 and Claude typically lead human-judged dialogue quality, though open models like Qwen 2.5-Max are shifting the rankings.

## 5. Challenges in Evaluation (性能评估的挑战, §5.2.5)

- **Metric limitations:** automated metrics cannot fully capture complex capabilities, especially generation quality and creativity.
- **Hallucination:** models generate plausible-but-false content, requiring human verification.
- **Safety & bias:** detecting bias/harmful content needs careful analysis and continuous monitoring.
- **Dynamism:** evaluation must keep pace with rapidly evolving models and techniques.
- **Cost:** human evaluation is expensive, automated evaluation is limited — balancing efficiency and accuracy is an open challenge.

## 6. Takeaway

Robust LLM evaluation combines all three layers: automated metrics/benchmarks for fast, large-scale screening; human evaluation for detailed quality control; and LLM-as-a-Judge as an efficient middle path. MMLU, TruthfulQA, BIG-bench, and other benchmarks provide quantitative comparison, while human and judge-based methods supply finer-grained insight into strengths and limitations.

## Related Pages

- [[Perplexity]] — the fluency metric doubling as an LM evaluation metric and data-quality heuristic.
- [[Pre-training Large Language Models]] — the training pipeline whose outputs these benchmarks evaluate.
- [[LLM技术报告]] — the source document (§5.2).
