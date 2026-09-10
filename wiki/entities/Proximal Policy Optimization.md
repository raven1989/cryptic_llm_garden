---
tags: [rl, algorithm, ppo, alignment, rlhf]
date: 2026-09-08
aliases: [PPO, 近端策略优化, Proximal Policy Optimization]
sources: ["[[raw/LLM/RL/RL — Proximal Policy Optimization (PPO) Explained.md]]"]
---

# Proximal Policy Optimization (PPO)

**近端策略优化（Proximal Policy Optimization, PPO）** 是由 OpenAI 于 2017 年提出的一种策略梯度强化学习算法，也是现代大语言模型（LLM）人类偏好对齐（RLHF）的事实标准基线之一。

## 核心设计与突破

1. **摆脱二阶计算瓶颈**：
   相比于其前身 TRPO（Trust Region Policy Optimization），PPO 彻底抛弃了需要求解高维 Fisher 信息矩阵（Hessian）及其逆矩阵的二次约束规划，转而使用标准的**一阶优化器（如 Adam）**进行高效训练。
2. **多轮更新与重要性采样**：
   引入重要性采样比率 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，使得在同一批环境交互采样（Rollout）数据上能够执行多轮（multi-epoch）小批量更新，成倍提升样本利用效率。
3. **截断代理目标（PPO-Clip）**：
   通过截断机制构造悲观下界，防止策略更新跨度过大导致“坠崖崩溃”：
   $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
   当动作概率比率偏离可信区间 $[1-\epsilon, 1+\epsilon]$（通常 $\epsilon=0.2$）时，梯度被清零，强制锁死更新幅度。

## 大模型 RLHF 架构中的角色

在 LLM RLHF 系统中，PPO 作为策略执行与对齐引擎，与以下模块协同构建完整闭环：
* **Actor 网络（$\pi_\theta$）**：待对齐的大语言模型；
* **Critic 网络（$V_\phi$）**：基于 [[Generalized Advantage Estimation]] 预测状态价值并构建基线；
* **[[Reward Model]]（$r_\psi$）**：提供序列级质量评价；
* **Reference Model（$\pi_{ref}$）**：提供 Token 级 KL 散度约束，防止模型语义漂移与 Reward Hacking。

参见完整研究报告：[[Proximal Policy Optimization Summary]]，以及对齐全流程：[[Fine-tuning]]。
