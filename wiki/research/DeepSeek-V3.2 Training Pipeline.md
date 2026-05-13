---
tags: [llm, architecture, reinforcement-learning, alignment, agents, deepseek]
aliases: [DeepSeek-V3.2 Training, GRPO]
date: 2026-05-12
sources: ["[[raw/LLM/DeepSeek-V3.2- Pushing the Frontier of Open Large Language Models.md]]"]
---

# DeepSeek-V3.2 Training Pipeline

The training pipeline for DeepSeek-V3.2 builds upon the base foundation of DeepSeek-V3.1, focusing on integrating the [[DeepSeek Sparse Attention]] (DSA) mechanism and significantly scaling up Post-Training Reinforcement Learning (RL) to enhance both reasoning and agentic tool-use capabilities.

## 1. Continued Pre-Training (DSA Integration)

Because DeepSeek-V3.2 introduces the sparse DSA mechanism into the existing [[Multi-Head Latent Attention]] (MLA) architecture, the model must be adapted without losing its underlying capabilities. This is done in two stages on 128K context-length data:

1. **Dense Warm-up Stage:** 
   * **State:** Main model parameters are frozen. The model still uses dense attention.
   * **Action:** The new "lightning indexer" is trained to mimic the target distribution $p_{t,:}$ (the L1-normalized sum of the main dense attention scores across all heads). 
   * **Loss:** A KL-divergence loss over the full sequence:
     $$\mathcal{L}^{I} = \sum_{t} \mathbb{D}_{KL}(p_{t,:} \| \operatorname{Softmax}(I_{t,:}))$$
   * **Scale:** ~2.1B tokens.
   
2. **Sparse Training Stage:**
   * **State:** The fine-grained token selection is activated. The model now structurally relies on sparsity.
   * **Action:** All model parameters (main model + indexer) are unfrozen and optimized. The indexer's input is detached from the computational graph to separate its training signal from the main model's language modeling loss.
   * **Loss:** The KL-divergence is restricted only to the selected top-$k$ sparse token set $\mathcal{S}_{t}$:
     $$\mathcal{L}^{I} = \sum_{t} \mathbb{D}_{KL}(p_{t,\mathcal{S}_{t}} \| \operatorname{Softmax}(I_{t,\mathcal{S}_{t}}))$$
   * **Scale:** ~943.7B tokens.

## 2. Specialist Distillation

Before unified RL training, DeepSeek trains distinct domain-specific specialist models (mathematics, programming, logical reasoning, agentic tasks, agentic coding, and agentic search). 
* Both "thinking" (long chain-of-thought) and "non-thinking" mode specialists are created.
* These specialists generate high-quality domain-specific data, which is then distilled back into the unified DeepSeek-V3.2 model.

## 3. Mixed RL Training (GRPO Scaling)

DeepSeek-V3.2 employs **Group Relative Policy Optimization (GRPO)**, merging reasoning, agent, and human alignment into a single RL stage to prevent catastrophic forgetting.

Crucially, DeepSeek scaled the RL compute budget to exceed 10% of the entire pre-training cost. To stabilize GRPO at this massive scale, they implemented four critical engineering optimizations built on top of the base GRPO objective:

### Unbiased KL Estimate
Traditional K3 estimators create noisy, unbounded gradients when sampled tokens have much lower probabilities in the current policy versus the reference policy ($\pi_{\theta} \ll \pi_{\text{ref}}$). DeepSeek corrected this using the importance-sampling ratio, eliminating systematic estimation errors:
$$\mathbb{D}_{\mathrm{KL}}(\pi_{\theta}(o_{i,t}) \| \pi_{\mathrm{ref}}(o_{i,t})) = \frac{\pi_{\theta}(o_{i,t}|q, o_{i,< t})}{\pi_{\mathrm{old}}(o_{i,t}|q, o_{i,< t})} \left( \frac{\pi_{\mathrm{ref}}(o_{i,t}|q, o_{i,< t})}{\pi_{\theta}(o_{i,t}|q, o_{i,< t})} - \log \frac{\pi_{\mathrm{ref}}(o_{i,t}|q, o_{i,< t})}{\pi_{\theta}(o_{i,t}|q, o_{i,< t})} - 1 \right)$$

### Off-Policy Sequence Masking
RL generates large rollouts that are split into mini-batches, naturally introducing off-policy data as the model updates. DeepSeek introduced a mask $M_{i,t}$ into the GRPO loss to completely zero out sequences with *negative advantages* ($\hat{A}_{i,t} < 0$) if their policy divergence (KL between the sampling policy and the current policy) exceeds a threshold $\delta$.
$$M_{i,t} = \begin{cases} 0 & \hat{A}_{i,t} < 0, \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \log \frac{\pi_{\text{old}}(o_{i,t}|q,o_{i,< t})}{\pi_{\theta}(o_{i,t}|q,o_{i,< t})} > \delta \\ 1 & \text{otherwise,} \end{cases}$$
The final modified GRPO loss becomes:
$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min \left( r_{i,t}(\theta) \hat{A}_{i,t}, \operatorname{clip} \dots \right) M_{i,t} - \beta \mathbb{D}_{KL} \right) \right]$$

### Keep Routing (for MoE)
In Mixture-of-Experts architectures, policy updates can cause identical inputs to be routed to different experts between the inference (rollout) and training phases. This abrupt subspace shift destabilizes optimization. DeepSeek enforces that the exact routing paths taken during rollout generation are maintained during the RL backward pass.

### Keep Sampling Mask
To prevent the RL objective from trying to optimize extremely low-probability tokens that would normally be truncated by top-$p$ or top-$k$ sampling during generation, DeepSeek applies the exact same truncation masks to the current policy during training.

## 4. Agentic Tool-Use Integration

To translate the extended reasoning capabilities of DeepSeek-R1 into interactive, multi-turn environments, the training pipeline incorporated a novel agentic synthesis methodology.

### Thinking Context Management
In multi-turn tool calling, discarding reasoning tokens (`<think>`) after every turn forces the model to waste tokens re-reasoning from scratch. The pipeline enforces a rule where:
* Historical reasoning content is **retained** when processing subsequent tool-execution outputs.
* Reasoning content is only discarded when a completely new *user message* is received.

### Large-Scale Task Synthesis
To generate robust data for the agent RL phase, DeepSeek built an autonomous pipeline that generated 1,827 task-oriented environments. 
1. The synthesis agent constructs a virtual sandbox (e.g., a database) and a set of mock API tools.
2. It proposes a task, a solution path, and a programmatic python verifier.
3. The agent repeatedly attempts to solve the task using *only* the provided tools, refining the verifier until the task is structurally sound.
This synthetic data (spanning planning, math, and code execution) is then fed into the RL pipeline, massively improving the final model's out-of-domain generalization on benchmarks like SWE-bench and BrowseComp.