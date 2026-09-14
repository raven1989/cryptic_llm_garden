---
tags: [rl, gae, advantage, value-estimation]
date: 2026-09-08
aliases: [GAE, 广义优势估计, Generalized Advantage Estimation]
sources: ["[[raw/LLM/RL/RL — Proximal Policy Optimization (PPO) Explained.md]]"]
---

# Generalized Advantage Estimation (GAE)

**广义优势估计（Generalized Advantage Estimation, GAE）** 是由 John Schulman 等人在 2015 年提出的一种强化学习优势函数 $\hat{A}_t$ 估计方法，旨在彻底化解纯蒙特卡洛（MC）估计的**高方差**与单步时序差分（1-Step TD）的**高偏差**之间的根本矛盾。

---

## 1. 核心理论演进与直觉概括

在自回归大模型对齐（RLHF）的序列决策中，GAE 的本质可以精准提炼为：
> **优势函数 $\hat{A}_t^{\text{GAE}}$ 被定义为从当前时间步 $t$ 到序列结束各个位置的“单步超预期度/惊喜值（TD 误差 $\delta$）”的指数加权之和。其本质仍是以终局序列末尾的稀疏奖励（Reward）来反向嘉奖前面的各个中间 token，但引入衰减因子 $\lambda \in [0, 1]$ 进行指数收缩平滑，使其完美处于纯蒙特卡洛（MC）的全步无偏 Rollout 与时序差分（TD）的单步 Critic 脑补之间。**

---

## 2. 前置基石：蒙特卡洛（MC）方法中的递推机制与即时奖励 $r_t$

在引入 GAE 与时序差分之前，最经典的强化学习基线是**纯蒙特卡洛策略梯度（REINFORCE 范式）**。

### 2.1 MC 递推公式与不需要 Critic 的本质
在完整的采样轨迹（Rollout）生成结束后，每个时间步 $t$ 的累计回报（Return）$G_t$ 遵循逆向递推公式：
$$G_t = r_t + \gamma G_{t+1}$$

* **严格的纯 MC（REINFORCE）完全不需要 Critic 模型**：
  直接沿采样出的轨迹逆向迭代求和计算 $G_t$，策略梯度直接通过 $\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t$ 更新，**纯属无参数、闭式定义的统计累加**。
* **何时才需要 Critic？**：
  由于纯 $G_t$ 方差极大（哪怕差回答也会因正向奖励而盲目提升所有 token 概率），需要引入与动作无关的基线（Baseline）以消除方差。若采用状态期望基线 $\hat{A}_t^{\text{MC}} = G_t - V_\phi(s_t)$，此时才必须引入可学习的 Critic 网络 $V_\phi$。

### 2.2 LLM 对齐语境下的即时奖励 $r_t$
在文本生成中，模型每生成一个 token 即迈出一步（step $t$），直到生成特殊结束符 `<EOS>`（时间步 $T$）。即时奖励 $r_t$ 是一个**复合奖励（Composite Reward）**：
$$r_t = \begin{cases} -\beta \cdot \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})_t, & t < T \\ R_{\text{RM}} - \beta \cdot \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})_T, & t = T \text{ (序列末尾 <EOS>)} \end{cases}$$

1. **中间步骤（$t < T$）**：外部环境奖励为 0（单个中间词无法独立评判全局好坏），$r_t$ 仅包含即时防漂移的 **Token-level KL 惩罚**；
2. **终点步骤（$t = T$）**：生成完毕，奖励模型（RM）对整句 Prompt + Response 打出的标量总分 $R_{\text{RM}}$ 进场结算，与最后一位的 KL 罚分合并构成终点奖励。

### 2.3 Token-level KL 散度在因果上下文中的精确计算
在计算中间步的 $\mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})_t$ 时，常有“它是 $t \sim T$ 的差还是 $0 \sim t$ 的整体差”的困惑。**答案是：它既不是未来的差，也不是历史的累加，而是以历史上下文 $s_t = (x, y_{<t})$ 为完全相同的条件，衡量两模型在第 $t$ 步对下一个 token 选择分歧的条件概率之差。**

* **点估计形式（Pointwise KL）**：
  工业界并不对庞大的词表做积分，而是对真实采样出的那个 token $y_t$ 取对数概率之差：
  $$\text{KL}_t \approx \log \pi_\theta(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t) = \log \frac{\pi_\theta(y_t \mid s_t)}{\pi_{\text{ref}}(y_t \mid s_t)}$$
* **Schulman 严格非负估计器**：
  为防止点采样带来的偶发负值波动，PPO 常用 Schulman 提出的低方差无偏估计：令比值 $u_t = \frac{\pi_{\text{ref}}(y_t \mid s_t)}{\pi_\theta(y_t \mid s_t)}$，则：
  $$\text{KL}_t = u_t - 1 - \log u_t \quad (\forall u_t > 0, \text{KL}_t \ge 0)$$
* **并行张量计算（Parallel Forward）**：
  工程上无需写 for 循环逐步推理。得益于 Transformer 的因果自注意力掩码（Causal Mask），将整条序列一次性送入 Actor 和 Ref 分别跑一次前向传播，位置 $t$ 处的 Logits 天然只能看到前文 $y_{<t}$。将两个模型输出的 `log_prob` 张量按位相减即可并行得到所有时间步的 $\text{KL}_t$。

### 2.4 MC 展开式的物理图像：终局 Reward 锚定与罚分扣除
当把即时奖励展开并代入 MC 递推式时（假设 $\gamma = 1.0$）：
$$G_t = R_{\text{RM}} - \beta \sum_{k=t}^{T} \text{KL}_k$$
* **物理图景**：以序列终局的 $R_{\text{RM}}$ 为唯一收益源头，逆向沿途扣减从当前步 $t$ 一直到末尾 $T$ 产生的所有背离参考模型的 KL 违规罚金。越靠近开头的 token（$t$ 越小），未来面临的随机性与潜在罚分累加越多；越靠近末尾的 token，不确定性越少。

---

## 3. 时序差分（TD）方法中的单步误差与优势

当不愿承担 MC 遍历到序列末尾的巨大方差时，时序差分（TD）方法选择“只看一步”。

### 3.1 TD 误差 $\delta_t$ 的数学定义与物理含义
单步时序差分误差（TD Error）$\delta_t$ 衡量的是“这一步真实得到的现实奖惩 + 下一步的新期望”与“当前位置的旧心理预期”之间的落差：
$$\delta_t = \underbrace{r_t + \gamma V(s_{t+1})}_{\text{单步现实目标 (TD Target)}} - \underbrace{V(s_t)}_{\text{当前时刻旧预期}}$$
* 若 $\delta_t > 0$：代表生成当前词 $y_t$ 带来的局势**超出预期**（产生正向惊喜）；
* 若 $\delta_t < 0$：代表生成当前词 $y_t$ **低于预期**（把局势带崩）。

### 3.2 1-Step TD 下优势函数的等价性：$\hat{A}_t^{\text{1-Step TD}} = \delta_t$
优势函数的严格定义是动作价值减状态价值：$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$。
在 1-Step TD 中，未来动作价值通过 Critic 预测进行单步自举：$Q(s_t, a_t) \approx r_t + \gamma V(s_{t+1})$。
直接代入定义即得：
$$\hat{A}_t^{\text{1-Step TD}} = \big(r_t + \gamma V(s_{t+1})\big) - V(s_t) = \delta_t$$
**单步时序差分下的优势值，在数学上严格等同于当时的单步 TD 误差 $\delta_t$。**

### 3.3 终点终止边界（$t = T$）的严密定义
在生成至最后一个有效 Token（如 `<EOS>`，时间步 $T$）时，生成过程彻底终止，未来无后继状态，物理上：
$$V(s_{T+1}) \equiv 0$$
因此，序列终点的 TD 误差必须严格扣除当前 Critic 的预期：
$$\delta_T = \big(R_{\text{RM}} - \beta \cdot \text{KL}_T\big) - V(s_T)$$
> **关键细节**：终点 $\delta_T$ **绝非单纯的 `Reward - KL` 本身**，必须减去 $V(s_T)$ 作为基线。若 Critic 已提前在 $T-1$ 步预判出句尾能拿高分，则最后一步属于“正常发挥”，惊喜度 $\delta_T \approx 0$；减去基线方能有效消除长程方差，避免盲目拉高梯度。

---

## 4. GAE 逆向递推链条与通项公式

GAE 没有停留在 1-Step TD 的“单步近视”，而是以 $\delta_t$ 为砖块，通过衰减因子 $\lambda \in [0, 1]$ 串联起未来的所有余波。

### 4.1 逆向递推公式
从序列终点 $T$ 向前进行反向时序传播（从后往前递推）：
$$\begin{aligned}
\hat{A}_T^{\text{GAE}} &= \delta_T = \big(R_{\text{RM}} - \beta \cdot \text{KL}_T\big) - V(s_T) \\
\hat{A}_t^{\text{GAE}} &= \delta_t + (\gamma \lambda) \hat{A}_{t+1}^{\text{GAE}}, \quad (\forall t < T)
\end{aligned}$$

### 4.2 展开通项（指数加权滑动和）
展开为未来各步 TD 误差的几何衰减累加：
$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l} = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \dots + (\gamma \lambda)^{T-t} \delta_T$$

终局奖励 $R_{\text{RM}}$ 包含在末项 $\delta_T$ 中，顺着衰减链条 $(\gamma \lambda)^{T-t}$ 像波浪一样倒灌给前面每一个中间 token，完成了对长文本生成的**全局信用分配（Credit Assignment）**。

---

## 5. 偏差-方差谱系与三种范式对比

GAE 并非孤立算法，它是统合 1-Step TD 与纯蒙特卡洛（MC）的**连续过渡带**，通过调节超参数 $\lambda \in [0, 1]$ 自由操控方差与偏差的天平：

| 范式 / 算法 | $\lambda$ 取值 | 优势估计公式 $\hat{A}_t$ | 依赖的信息跨度 | 偏差（Bias）与方差（Variance）特性 |
| :--- | :--- | :--- | :--- | :--- |
| **1-Step TD** | $\lambda = 0$ | $\hat{A}_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ | 仅依赖当下迈出的 1 步，后序全靠 Critic 预测（Bootstrap） | **方差极小，偏差极高**（极度受限于 Critic 初始是否准确） |
| **GAE** | $0 < \lambda < 1$ | $\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$ | 当前步强依赖，未来步按指数衰减权衡吸收 | **最优帕累托前沿**（近处惊喜权重大，远处长程噪声衰减） |
| **纯蒙特卡洛 (MC)** | $\lambda = 1$ | $\hat{A}_t = G_t - V(s_t) = \sum_{l=0}^{T-t} \gamma^l \delta_{t+l}$<br>*(中间项裂项相消 Telescoping Sum)* | 整条真实生成轨迹累加到底，直到终局 $R_{\text{RM}}$ | **理论严格无偏，方差极大**（长序列采样扰动剧烈叠加） |

> **工业界标配超参**：
> 在大模型 RLHF 训练中，由于对话文本长度适中且必须严格兑现终局奖励，通常设定 **$\gamma = 1.0$**（不进行未来收益的时间折现）与 **$\lambda = 0.95$**（给未来各步保留极高权重，同时平滑长程采样抖动）。

---

## 6. 对 Critic 训练的赋能与循环演进

GAE 计算出的优势值不仅用于指导 Actor 策略网络的裁剪更新（Clip Loss），还通过逆向重构构造出 Critic 模型的监督回归目标（TD Target）：
$$V_t^{\text{target}} = \hat{A}_t^{\text{GAE}} + V_{\text{old}}(s_t)$$

该目标在反向传播计算 Critic 均方误差损失（MSE Loss）时保持冻结：
$$\mathcal{L}_{\text{Critic}}(\phi) = \frac{1}{2} \mathbb{E}\left[ \big( V_\phi(s_t) - V_t^{\text{target}} \big)^2 \right]$$
通过将带有平滑滤波优势与旧预测值相加，Critic 得以在一个方差可控、具备远见引导的稳定目标下持续迭代收敛。

---

参见完整研究报告：[[Proximal Policy Optimization Summary]]，核心实体：[[Proximal Policy Optimization]]、[[Reward Model]]、[[Critic Model]]、[[RLHF]]。
