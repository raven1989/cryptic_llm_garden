---
tags: [rl, gae, advantage, value-estimation]
date: 2026-09-08
aliases: [GAE, 广义优势估计, Generalized Advantage Estimation]
sources: ["[[raw/LLM/RL/RL — Proximal Policy Optimization (PPO) Explained.md]]"]
---

# Generalized Advantage Estimation (GAE)

**广义优势估计（Generalized Advantage Estimation, GAE）** 是由 John Schulman 等人提出的一种强化学习优势函数 $\hat{A}_t$ 估计方法，旨在解决蒙特卡洛估计的高方差与单步时序差分（TD）的高偏差之间的矛盾。

## 数学定义与递推机制

定义单步时序差分残差（1-step TD Error）为：
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

GAE 引入衰减因子 $\lambda \in [0, 1]$，从序列末尾向前**反向递推**累加优势值：
$$\hat{A}_T = \delta_T$$
$$\hat{A}_t = \delta_t + (\gamma \lambda) \hat{A}_{t+1}$$

展开通项公式为未来残差的指数加权滑动平均：
$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

## 偏差与方差的动态平衡

* **当 $\lambda = 0$ 时**：$\hat{A}_t = \delta_t$。完全依赖单步即时奖励与 Critic 的状态估值，对应**单步 TD 估计**（方差最低，但若 Critic 不准则偏差极高）。
* **当 $\lambda = 1$ 时**：展开式中的中间价值预测项发生**裂项相消（Telescoping Sum）**，$\hat{A}_t = G_t - V(s_t)$。完全对应**蒙特卡洛（Monte Carlo）减基线**（无偏差，但方差极大）。
* **工业实践**：大模型 RLHF 普遍采用 $\gamma = 1.0, \lambda = 0.95$，取得极佳的收敛平稳度。

## 对 Critic 训练的赋能

GAE 不仅为 Actor 提供了低方差高精度的方向指导，还通过反向重构生成了 Critic 的最优训练标签：
$$V_t^{target} = \hat{A}_t^{GAE} + V_{old}(s_t)$$
该目标在梯度回传时保持冻结，使 Critic 能稳定拟合具有合理偏差控制的折现期望回报。

参见完整研究报告：[[Proximal Policy Optimization Summary]]。
