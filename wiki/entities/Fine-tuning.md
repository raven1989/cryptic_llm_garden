---
tags: [llm, training, alignment]
date: 2026-04-08
aliases: [Supervised Fine-Tuning, SFT, RLHF, DPO, LoRA]
---

# Fine-tuning and Alignment

The process of evolving a pre-trained base model into an instruction-following assistant or specialized agent.

## The Training Pipeline

1.  **Pre-training:** The most expensive phase. Predicts the next token using Cross-Entropy Loss to build a foundational understanding of language and facts.
2.  **Supervised Fine-Tuning (SFT):** Injects thousands of high-quality QA pairs. Crucially, the loss is only calculated on the generated response portion, masking the prompt. This forces the model to adopt the format and tone of an answer.
3.  **Alignment:** SFT alone cannot handle complex boundary cases or human preferences. Alignment prevents toxic outputs and ensures helpfulness.

## Alignment Strategies

*   **Reinforcement Learning from Human Feedback (RLHF):**
    *   **Phase 1:** Train a Reward Model (RM) using human rankings of multiple responses to a single prompt.
    *   **Phase 2:** Use Proximal Policy Optimization (PPO) to treat the LLM as an agent, maximizing the RM's score. KL Divergence prevents the LLM from collapsing into unnatural language just to "hack" the reward.
*   **Direct Preference Optimization (DPO):**
    *   The modern industrial standard. It bypasses the fragile RM and complex PPO completely.
    *   Given a dataset of paired responses $(x, y_{chosen}, y_{rejected})$, it uses a binary classification loss directly on the SFT model to increase the probability of the chosen answer and decrease the rejected one.

## Parameter-Efficient Fine-Tuning (PEFT)

Methods to train models without the enormous compute requirements of full-parameter updates.

*   **LoRA (Low-Rank Adaptation):** Freezes the original weights and injects two small, low-rank matrices ($A \times B$) alongside linear layers. Drastically reduces the number of trainable parameters (by >90%).
*   **QLoRA:** Combines LoRA with 4-bit quantization of the base model, enabling fine-tuning of billion-parameter models on a single consumer GPU.

See also: [[Transformers]], [[LLM Study Plan]]
