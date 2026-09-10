---
tags: [rlhf, reward-model, alignment, preferences]
date: 2026-09-08
aliases: [RM, 奖励模型, 偏好模型, Reward Model]
sources: ["[[raw/LLM/RL/RL — Proximal Policy Optimization (PPO) Explained.md]]"]
---

# Reward Model (RM)

**奖励模型（Reward Model, RM）** 是大语言模型通过人类反馈强化学习（RLHF）实现价值对齐的核心打分器，负责将人类对模型生成文本的质量偏好量化为一个实数标量得分。

## 模型架构与输入输出

* **网络骨干**：通常采用与策略模型（Actor）相同架构的预训练语言模型（Transformer Decoder）。
* **打分头（Value Head）**：将模型最后的分类头替换为一个单输出线性投影层 `Linear(hidden_dim, 1)`。
* **输入与输出**：输入完整的文本序列 `Prompt + Response`，输出一个代表整体回答质量的标量数值 $R_{RM}$（仅在序列最后一个有效 token 取出打分）。

## 训练机制：Bradley-Terry 偏好排序

由于人类标注员难以给出精准一致的绝对分，工业界采用**成对比较（Pairwise Comparison）**。给定 Prompt $x$ 与优选回答 $y_w$、淘汰回答 $y_l$，基于 **Bradley-Terry 模型** 建模偏好概率：
$$P(y_w \succ y_l \mid x) = \sigma\left( r_\psi(x, y_w) - r_\psi(x, y_l) \right)$$

通过最小化负对数似然损失进行端到端优化：
$$L_{RM}(\psi) = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( r_\psi(x, y_w) - r_\psi(x, y_l) \right) \right]$$

## 在 PPO 体系中的协同与传承

1. **考核评委**：在 [[Proximal Policy Optimization]] 的采样（Rollout）阶段，RM 参数被完全冻结，只在回答生成的最后一个 token 位置结算全局奖励 $R_{RM}$。
2. **冷启动初始化 Critic**：因 RM 与 Critic 的网络结构完全相同，且因果自注意力机制赋予了 RM 中间各 token 评估前缀潜力的先验能力，工业界普遍直接用训练好的 RM 权重初始化 Critic，极大加快了强化学习的初始收敛。

参见完整研究报告：[[Proximal Policy Optimization Summary]]，相关技术：[[Fine-tuning]]。
