---
tags: [concept, rl, deepseek, alignment, policy-gradient, critic-free]
date: 2026-09-14
aliases: [GRPO, 组相对策略优化]
sources: ["[[raw/LLM/RL/GRPO-Group Relative Policy Optimization.md]]"]
---

# Group Relative Policy Optimization (GRPO)

**Group Relative Policy Optimization (GRPO)** 是一种专为大型语言模型（LLM）对齐与复杂逻辑推理设计的**无价值模型（Critic-Free）**强化学习算法，由 DeepSeek 团队在 DeepSeekMath 中提出，并在 DeepSeek-R1 系列中被证实为激发模型长思维链推理与自我反思（Aha-moment）的核心引擎。

---

## 核心机制与原理

1. **彻底废除 Critic 网络**：
   传统 [[Proximal Policy Optimization|PPO]] 必须训练一个与策略模型参数规模相当的 [[Critic Model|价值网络 $V(s)$]] 来计算基线与 [[Generalized Advantage Estimation|GAE]] 优势。GRPO 彻底移除了 Critic，直接省去其参数、梯度与优化器显存（降低 30%~50% 显存消耗），消除了价值网络拟合失真引发的策略崩溃风险。
2. **组内相对基线归一化（Group Relative Advantage）**：
   对于单个 Prompt $q$，利用当前策略并发采样生成一组（$G$ 个）候选输出 $\{o_1, \dots, o_G\}$。对每个完整序列打出标量奖励 $r_i$ 后，以该组候选奖励的统计均值 $\mu_G$ 为基线、以标准差 $\sigma_G$ 为尺度因子进行 Z-score 归一化：
   $$\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}$$
   所得标量优势被沿序列广播至各个 Token，直接指引策略梯度更新。
3. **独立外置的非负无偏 KL 正则**：
   将针对参考模型 $\pi_{ref}$ 的 KL 散度约束移出优势计算，采用凸函数形式：
   $$\mathbb{D}_{KL}(\pi_\theta \parallel \pi_{ref}) = \frac{\pi_{ref}}{\pi_\theta} - \log \frac{\pi_{ref}}{\pi_\theta} - 1 \ge 0$$
   在逐 Token 粒度上严格保证非负性，避免了采样不均导致的负散度反向奖励扰动。
4. **长度倒数归一化**：
   在目标函数中乘以 $\frac{1}{|o_i|}$，使梯度贡献与序列长度解耦，抑制利用长文本累积梯度的 Reward Hacking。

---

## 适用场景与权衡（Trade-offs）

* **最适场景（RLVR）**：特别契合数学解题、代码竞赛、形式化逻辑等拥有客观规则验证器（Rule-based Verifiers, 如 SymPy、单元测试沙箱）的任务。
* **潜在风险**：
  * **方差塌缩（Variance Collapse）**：当组大小 $G$ 较小且题目出现全对（全 1 分）或全错（全 0 分）时，组内方差为 0，有效梯度塌缩为零，需扩大组大小（如 $G=64$）并结合难度自适应调度；
  * **粗粒度信用分配**：标量广播无法对中间推理关键步骤（Pivotal Steps）进行细粒度归因。

---

## 关联页面
- 详细算法全景与推导：[[GRPO Summary]]
- 对比算法：[[Proximal Policy Optimization Summary]] | [[Proximal Policy Optimization]]
- 依赖概念：[[Critic Model]] | [[Generalized Advantage Estimation]] | [[RLHF Summary]] | [[Direct Preference Optimization]]
- 架构演进：[[DeepSeek-V3.2 Training Pipeline]]
