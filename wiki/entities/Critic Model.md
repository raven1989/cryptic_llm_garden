---
tags: [rl, critic-model, value-estimation, rlhf, baseline]
date: 2026-09-08
aliases: [Critic Model, Value Model, Critic, 价值模型, 价值网络, 评论家网络]
sources: ["[[raw/LLM/RL/RL — Proximal Policy Optimization (PPO) Explained.md]]"]
---

# Critic Model (价值模型 / 评论家网络)

**Critic 模型（Value Model / Critic Network）** 是强化学习 Actor-Critic 架构中的核心组成部分，负责估计状态价值函数 $V(s)$，为策略模型（Actor）的更新提供基线（Baseline），以大幅降低方差并计算优势函数（Advantage）。

## 模型架构与输入输出

* **网络骨干**：在 LLM RLHF 中，Critic 通常使用与策略网络（Actor）同款的预训练或指令微调（SFT）模型骨干（Transformer Decoder）。
* **打分头（Value Head）**：将最终的词表分类头替换为 `Linear(hidden_dim, 1)`，对输入的每个 token 隐状态映射输出一个实数标量。
* **输入**：文本序列（Prompt + 已生成的回答前缀）。
* **输出**：序列中每个 token 位置的状态价值预测序列 $[V(s_1), V(s_2), \dots, V(s_T)]$。

## 输出的物理本质与核心概念区分

$V(s_t)$ 代表：**站在时刻 $t$ 面对当前状态前缀 $s_t$（尚未做出动作 $a_t$），按照当前策略继续自回归写完整篇回答，未来预期能够获得的折现累积奖励总和的期望值**。

| 概念符号 | 本质物理含义 | 产出时机 | 核心作用 |
| :--- | :--- | :--- | :--- |
| **$V(s_t)$** | **事前预期基准**：当前前缀未来写完后的平均预期总分。 | 生成过程中由 Critic 逐 token 预测。 | 充当 Baseline 降低方差。 |
| **$r_t$** | **当步客观反馈**：即时步的 KL 惩罚或末尾的大奖 $R_{RM}$。 | 走完第 $t$ 步后由环境/RM 给出。 | 真实世界反馈信号。 |
| **$Q(s_t, a_t)$** | **事后实际总收益**：在 $s_t$ 执行具体 $a_t$ 后的即时奖与后续折现（$r_t + \gamma V(s_{t+1})$）。 | 做出动作 $a_t$ 之后推算得到。 | 该动作的综合总分。 |
| **$\hat{A}_t$** | **惊喜度（超出平均）**：$Q(s_t, a_t) - V(s_t)$。 | 综合计算后得出。 | 指引策略梯度更新方向。 |

## 训练机制与关键数学逻辑

1. **学习目标（Target）**：
   通过 [[Generalized Advantage Estimation]] 构造折现累积回报目标：
   $$V_t^{target} = \hat{A}_t^{GAE} + V_{\phi_{old}}(s_t)$$
2. **为什么拟合 $V(s_t)$ 而非 $V(s_{t+1})$（时间因果对齐）**：
   在时刻 $t$，Critic 仅输入了前缀 $s_t$，尚未知晓未来动作与状态。损失函数考核的是 Critic 在 $t$ 时刻做出的“事前预测”与后续事实总回报 $V_t^{target}$ 之间的差距，迫使其具备前瞻评估能力。
3. **为什么价值截断损失取 $\max$（悲观上界）**：
   $$L^{Critic}(\phi) = \frac{1}{2} \hat{\mathbb{E}}_t \left[ \max\left( (V_\phi(s_t) - V_t^{target})^2, \; (V_\phi^{clipped}(s_t) - V_t^{target})^2 \right) \right]$$
   当 Critic 参数更新幅度超过截断阈值 $\epsilon$ 时，截断误差常数大于未截断误差，取 $\max$ 使得损失函数选中截断常数，**对参数 $\phi$ 的反向传播梯度瞬间清零**，构筑了防止参数剧烈震荡的“安全刹车防护墙”。
4. **冷启动与自举（Bootstrapping）**：
   工业界普遍使用训练好的 [[Reward Model]] 权重来初始化 Critic。即使初始预测存在偏差，由于序列终止符 EOS 处的 $V(s_{T+1}) \equiv 0$ 和真实奖励 $r_T = R_{RM}$ 构成了绝对物理锚点，真实信号会通过反向递推如多米诺骨牌般迅速校准中间各步价值。

参见完整算法总结：[[Proximal Policy Optimization Summary]]，关联概念：[[Proximal Policy Optimization]]、[[Generalized Advantage Estimation]]。
