---
tags: [rl, ppo, rlhf, policy-gradient, deep-learning, gae, reward-model, critic-model]
date: 2026-09-08
aliases: [PPO Summary, Proximal Policy Optimization Explained, 近端策略优化详解, PPO算法与大模型RLHF全景]
sources: ["[[raw/LLM/RL/RL — Proximal Policy Optimization (PPO) Explained.md]]"]
---

# Proximal Policy Optimization (PPO) 与大模型 RLHF 全景指南

> **“Simplicity rules in deep learning.”** —— PPO 算法通过引入一阶近似的概率比率截断（Clipped Objective），在复杂的二阶数学保证（TRPO）与脆弱的朴素策略梯度（Vanilla PG）之间找到了极佳的工程平衡点，成为强化学习领域及大语言模型人类偏好对齐（RLHF）的事实标准基石。

---

## 知识认知学习导航

本文档按照“**问题驱动、层层递进**”的人类认知学习路径组织，结合大语言模型（LLM RLHF）前沿工程实践，系统解答以下核心疑问链：
1. **起源与动机（Why PPO?）**：标准策略梯度为何会“掉下悬崖”？TRPO 为何难以工业级规模化？
2. **解构代理目标 $L(\theta)$**：公式中的概率比率、Token 级分布与重要性采样本质是什么？新旧模型数据是如何存储与对齐的？
3. **四大金刚角色与即时奖励 $r_t$ 闭环**：Actor、Critic、Reward Model、Reference Model 的职责分工、Token 级 KL 惩罚与 Reward Model 打分校准；
4. **优势 $\hat{A}_t$ 估算算法演进**：从蒙特卡洛（MC）到单步 TD，再到广义优势估计（GAE）的偏差-方差权衡与裂项相消数学证明；
5. **Critic 价值网络的训练机理**：冷启动多米诺骨牌效应、为什么拟合 $V(s_t)$ 而非 $V(s_{t+1})$？为什么 Loss 截断处要取 $\max$（悲观上界）？
6. **Actor 截断目标（PPO-Clip）的深层机理**：悲观下界如何像安全刹车片一样在正负优势下阻止梯度爆炸？
7. **完整工业闭环与代码实现**：Rollout 与 Training 阶段的严格解耦、PyTorch 核心实现代码。

---

## 一、算法背景与核心动机（Why PPO?）

### 1. 朴素策略梯度（Vanilla Policy Gradient）的致命悬崖
传统的策略梯度方法（如 REINFORCE）本质上是**同策略（On-policy）**方法：
* 智能体根据当前策略 $\pi_\theta$ 与环境交互采样（Rollout）收集一批轨迹。
* 优化器计算梯度并对参数 $\theta$ 进行一次更新：
  $$L^{PG}(\theta) = \hat{\mathbb{E}}_t \left[ \log \pi_\theta(a_t \mid s_t) \hat{A}_t \right]$$
* **致命缺陷 1（样本利用率极低）**：更新一次之后，参数变为 $\theta_{new}$，原数据分布不再匹配，整批昂贵的交互样本必须全部丢弃，必须重新采样。在大模型生成任务中，自回归推理极度耗时且昂贵，这在工业上不可承受。
* **致命缺陷 2（悬崖效应 Cliff Effect）**：在复杂的非凸优化表面上，Line Search（梯度下降）若步长迈得过大，策略可能瞬间跌入“坏策略深渊”（如文章中比喻的 Angels Landing 徒步失足）。由于强化学习的数据完全是由策略自身探索采集的，一旦策略崩溃，后续采集到的全是有毒/低质量数据，智能体几乎无法自愈。

### 2. TRPO 与二阶优化的理论困境
为了防止策略跨度过大，**TRPO（Trust Region Policy Optimization，信任域策略优化）** 引入了硬性约束：
$$\max_\theta L(\theta) \quad \text{s.t.} \quad \hat{\mathbb{E}}_t \left[ \mathbb{D}_{KL}\left( \pi_{\theta_{old}}(\cdot \mid s_t) \parallel \pi_\theta(\cdot \mid s_t) \right) \right] \le \delta$$

* **优势**：通过 MM 算法（Minorize-Maximization）在数学上证明了策略改进的理论单调性下界。
* **工业死穴**：求解该约束优化问题需要对目标函数做泰勒二阶展开，必须计算 **Fisher 信息矩阵（Hessian）及其逆矩阵**。对于具有数十亿至数千亿参数的大模型而言，$O(N^2)$ 的显存与 $O(N^3)$ 的求逆计算复杂度是绝对不可能落地的；此外，共轭梯度求解过程与 Dropout、参数共享等现代深度学习组件兼容性极差。

### 3. PPO 的破局思想
PPO 放弃了对硬约束的执念，提出：**用一阶优化器（如 Adam/SGD）配合软惩罚（Adaptive KL）或直接在目标函数中进行截断（Clipped Objective）**。即使偶尔在极少数样本上越出信任域，依靠截断机制也能瞬间抹平超出边界的有害梯度，兼顾了极致的数学稳定性与极简的代码实现。

---

## 二、解构代理目标函数 $L(\theta)$

PPO 首先利用**重要性采样（Importance Sampling）**将目标函数改写为无约束的代理形式：

$$L(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)} \hat{A}_t \right] = \hat{\mathbb{E}}_t \left[ r_t(\theta) \hat{A}_t \right]$$

### 1. $\pi_\theta$ 与 $\pi_{\theta_{old}}$ 是什么？是 Logits 吗？
* **结论**：**绝不是原始未归一化的 Logits，而是经过 Softmax 归一化后的条件概率（Probability）**。
* **数值稳定实现**：直接计算除法 $\frac{\pi_\theta}{\pi_{\theta_{old}}}$ 会因长尾概率极小而出现下溢或浮点溢出。工程上统一计算**对数概率差，再取指数 $\exp$**：
  $$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)} = \exp\left( \log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{old}}(a_t \mid s_t) \right)$$
* **分母是常数，分子带梯度**：
  * $\pi_{\theta_{old}}(a_t \mid s_t)$：在交互采样阶段由旧模型计算出的标量数值，在训练时被 `.detach()`，充当基准常数。
  * $\pi_\theta(a_t \mid s_t)$：当前正在连续迭代更新的模型输出，挂载在计算图上，提供反向传播梯度。

### 2. 算子具象化：Token 级逐点相除实例
假设采样阶段针对某 Prompt 生成了包含 3 个 Token 的回答：
$$\text{回答序列} = [a_1, a_2, a_3] = [\text{"我"}, \text{"喜欢"}, \text{"苹果"}]$$

生成时记录旧模型概率（标量值）：
* $\pi_{\theta_{old}}(a_1 = \text{"我"} \mid s_1) = 0.60$
* $\pi_{\theta_{old}}(a_2 = \text{"喜欢"} \mid s_2) = 0.80$
* $\pi_{\theta_{old}}(a_3 = \text{"苹果"} \mid s_3) = 0.50$

进入更新阶段，当前最新模型 $\pi_\theta$ 重新评估**完全相同的这串词**：
* $\pi_\theta(a_1 = \text{"我"} \mid s_1) = 0.66$
* $\pi_\theta(a_2 = \text{"喜欢"} \mid s_2) = 0.72$
* $\pi_\theta(a_3 = \text{"苹果"} \mid s_3) = 0.60$

比率 $r_t(\theta)$ 逐点计算为：
$$\mathbf{r}(\theta) = \left[ \frac{0.66}{0.60}, \; \frac{0.72}{0.80}, \; \frac{0.60}{0.50} \right] = [1.10, \; 0.90, \; 1.20]$$

```python
# PyTorch 工业级实现逻辑:
# logits 形状: [batch_size, seq_len, vocab_size]
log_probs = F.log_softmax(actor_model(input_ids), dim=-1)
# 关键: 只提取实际生成的那个 token 的对数概率
action_log_probs = log_probs.gather(dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)
# 与缓存的旧对数概率做差
ratios = torch.exp(action_log_probs - old_action_log_probs) # 形状: [batch_size, seq_len]
```

### 3. 核心误区澄清：两者的 Token 序列是否不同？需要缓存全词表概率吗？
* **序列绝对相同（固定剧本）**：
  在 PPO 的更新阶段（Epochs 内），**当前模型 $\pi_\theta$ 绝不会重新去自回归生成新文本**！因为一旦生成新文本，之前计算的奖励和状态价值全部脱节失效。
  $\pi_\theta$ 是以类似 SFT 的 **Teacher Forcing（并行前向）** 方式，完整接收旧策略采集好的固定文本 `Prompt + Response`。
* **存储开销极轻**：
  新旧模型都在评估“在历史现场执行完全相同的那个动作（选中的 Token）”。因此，每个位置**只需存储 1 个标量 float32 对数概率值**，显存大小仅为 `[batch_size, seq_len]`，**完全不需要缓存几万维词表的全量 Softmax 概率**。

### 4. 优势函数 $A(s, a)$ 的本质物理意义
$$A(s, a) = Q(s, a) - V(s)$$
* **$V(s)$（状态价值）**：在状态 $s$ 下，按照当前策略平均能拿多少分（大盘基准 Baseline）。
* **$Q(s, a)$（动作价值）**：在状态 $s$ 下特意执行动作 $a$ 之后，预期能够获得的总收益。
* **$A(s, a)$（优势）**：反映该特定动作比当前平均表现“**好多少（惊喜度 $>0$）**”或“**差多少（低于预期 $<0$）**”。

---

## 三、大模型 PPO 中的“四大金刚”模型与即时奖励 $r_t$

为了求出优势 $A$，在纯理论中给定的环境奖励必须具象化。在大语言模型 RLHF 中，系统内同时协同运行着 **4 个核心网络**：

### 1. 四大模型角色分工

| 模型角色 | 参数状态 | 模型结构 | 职责与作用 |
| :--- | :--- | :--- | :--- |
| **Actor（策略模型 $\pi_\theta$）** | **持续更新** | 完整 LLM 生成头 | 自回归生成回答；通过 PPO 损失更新参数。 |
| **Reference Model（参考模型 $\pi_{ref}$）** | **永久冻结** | SFT 初始模型 | 充当安全缰绳，提供 Token 级 KL 散度约束，防止模型退化。 |
| **Reward Model（奖励模型 $r_\psi$）** | **永久冻结** | LLM + 标量 Head | 评委考官。在句子完整生成完后对整篇质量打出全局分 $R_{RM}$。 |
| **Critic（价值模型 $V_\phi$）** | **持续更新** | LLM + 标量 Head | 预测当前状态未来的累积期望总回报，提供基线计算 Advantage。 |

### 2. 即时奖励 $r_t$ 的构造与 Token 级 KL 惩罚
在文本生成中，单个词（如“因为”）无法直接判定全局好坏，因此大模型即时奖励定义为：

$$r_t = \begin{cases} 
-\beta \cdot \text{KL}_t, & \text{当 } t < T \text{ (中间普通 Token)} \\[6pt]
R_{RM} - \beta \cdot \text{KL}_T, & \text{当 } t = T \text{ (序列结束 EOS)} 
\end{cases}$$

#### (1) KL 散度计算公式
* **标准对数差估计器**（最通用）：
  $$\text{KL}_t = \log \pi_\theta(a_t \mid s_t) - \log \pi_{ref}(a_t \mid s_t)$$
* **Schulman 严格非负估计器**（防止单样本估计出现负值）：
  令比率 $u_t = \frac{\pi_{ref}(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)}$，则：
  $$\text{KL}_t \approx u_t - 1 - \log u_t \ge 0$$
* **物理防线（防止 Reward Hacking）**：若没有 $-\beta \cdot \text{KL}_t$，Actor 会迅速抓住 Reward Model 的漏洞（如疯狂复读某种排比句或高频套话），导致语言逻辑崩塌。

#### (2) Reward Model 的训练与打分校准（Score Calibration）
* **训练机制**：人类无法给出精准绝对分，但擅长二选一排序。基于 **Bradley-Terry 偏好模型**，成对数据 $(x, y_w, y_l)$ 的损失函数为：
  $$L_{RM}(\psi) = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( r_\psi(x, y_w) - r_\psi(x, y_l) \right) \right]$$
* **工程校准（Normalization & Clipping）**：
  * 因为 Bradley-Terry 损失只约束打分差值，对绝对值整体平移常数不敏感（$P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$），导致不同批次训练出的 RM 打分尺度可能漂移。
  * 工业界标准做法：在填入 $R_{RM}$ 之前，进行**滑动均值方差白化**（Running Mean/Std Whitening）以及**数值截断**（如限制在 $[-5, +5]$ 之间），防止极端 Outlier 击穿策略梯度。
* **为什么 RM 结构与 Critic 一致，能用来初始化 Critic？**
  * 两者在底层都是 Transformer 后面接 `Linear(hidden_dim, 1)`。
  * RM 训练时虽然只提取最后一个 token 的值计算 Loss，但因果自注意力使得每个中间 token $h_t$ 均编码了“该前缀在未来能拿高分的质量潜力”。
  * 工业界直接用训练好的 RM 权重初始化 Critic，使 Critic 在第 0 步就拥有成熟的语义评估基准，避免随机初始化造成的剧烈策略震荡。

---

## 四、优势函数 $\hat{A}_t$ 的估算算法：从 MC、TD 到 GAE

在实际交互轨迹中，我们无法获知理论期望，必须进行经验估计。

### 1. 核心概念对比辨析表

| 概念符号 | 本质物理含义 | 产出时机 | 核心作用 |
| :--- | :--- | :--- | :--- |
| **$V(s_t)$** | **事前预期基准**：当前前缀在未来写完后的**平均预期总分**（尚未执行 $a_t$）。 | 生成过程中每个 token 步由 Critic 预测。 | 充当 Baseline 减小方差。 |
| **$r_t$** | **当步客观反馈**：即时步的 KL 惩罚或末尾的大奖 $R_{RM}$。 | 走完第 $t$ 步后由环境/RM 给出。 | 真实物理世界的收益输入。 |
| **$Q(s_t, a_t)$** | **事后实际总收益**：在 $s_t$ 执行具体 $a_t$ 后的即时奖与后续折现（$r_t + \gamma V(s_{t+1})$）。 | 做出动作 $a_t$ 之后推算得到。 | 该动作的综合总分。 |
| **$\hat{A}_t$** | **惊喜度（超出平均）**：$Q(s_t, a_t) - V(s_t)$。 | 综合计算后得出。 | 指引策略梯度更新的方向与大小。 |

### 2. 方法演进全景对比

#### 方法 1：蒙特卡洛（Monte Carlo Baseline）
$$G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}, \qquad \hat{A}_t = G_t - V(s_t)$$
* **$G_t$ 是 Critic 的真值标签（Label）**。
* **特性**：无偏差（No Bias），但一条几百 token 的长序列受随机采样干扰极大，方差极高（High Variance）。

#### 方法 2：单步时序差分（1-step TD Error）
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad \hat{A}_t = \delta_t$$
* **特性**：方差小，但未来的收益全靠 Critic 预测，在训练初期 Critic 自身不准时会带来极大的系统偏差（High Bias）。

#### 方法 3：广义优势估计（GAE, Generalized Advantage Estimation）
PPO 的核心标配方案，通过超参数 $\lambda \in [0, 1]$ 进行指数加权滑动平衡：
* **反向递推式**（从序列末尾 $T$ 往前推算）：
  $$\hat{A}_T = \delta_T$$
  $$\hat{A}_t = \delta_t + (\gamma \lambda) \hat{A}_{t+1}$$
* **展开通项式**：
  $$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}$$

### 3. 极端情况的数学证明
* **当 $\lambda = 0$ 时**：$\hat{A}_t = \delta_t$，完全退化为单步 TD（低方差、高偏差）。
* **当 $\lambda = 1$ 时**：
  $$\hat{A}_t^{GAE(\gamma, 1)} = \sum_{l=0}^{T-t} \gamma^l \delta_{t+l}$$
  展开每一项的 $\delta$：
  $$\begin{aligned}
  \hat{A}_t &= \left[ r_t + \gamma V(s_{t+1}) - V(s_t) \right] + \gamma \left[ r_{t+1} + \gamma V(s_{t+2}) - V(s_{t+1}) \right] + \dots \\
  &= \sum_{l=0}^{T-t} \gamma^l r_{t+l} + \gamma^{T-t+1} V(s_{T+1}) - V(s_t)
  \end{aligned}$$
  所有中间价值项发生**裂项相消（Telescoping Sum）**！由于终止态 $V(s_{T+1}) = 0$，最终精确等于：
  $$\hat{A}_t = G_t - V(s_t)$$
  完全退化为蒙特卡洛回报减基准（低偏差、高方差）。
* **工程取值**：大模型 RLHF 普遍取 $\gamma = 1.0, \lambda = 0.95$，取得最稳定的折中。

### 4. 优势标准化（Advantage Normalization）
在拿到一个 batch 的优势值后，必须执行归一化：
$$\hat{A}_t^{norm} = \frac{\hat{A}_t - \text{mean}(\hat{A})}{\text{std}(\hat{A}) + 10^{-8}}$$
确保约 50% 的动作受到正向激励，50% 受到负向惩罚，梯度更新尺度极度稳定。

---

## 五、Critic 网络的训练与更新机理

### 1. Critic 的学习目标（Target）
基于动作价值与优势的定义 $Q(s, a) = A(s, a) + V(s)$，Critic 自身的回归目标直接由 GAE 给出：
$$V_t^{target} = \hat{A}_t^{GAE} + V_{\phi_{old}}(s_t)$$
此处的 $V_t^{target}$ 在梯度更新时作为固定的常数标签（`detach()`）。

### 2. Critic 冷启动与自举（Bootstrapping）
如果 Critic 刚开始预测不准，为什么系统不会崩溃？
* **序列末端的绝对物理锚点**：
  在序列最后一步 $T$（终止符 EOS 处），终止态 $V(s_{T+1}) \equiv 0$ 严格成立，而 $r_T = R_{RM}$ 是外部给出的真实得分！
  $$\delta_T = R_{RM} - V(s_T)$$
  这个绝对真实的事后信号，会像**多米诺骨牌**一样，通过 GAE 逆向递推式一步一步往前传导渗透。
* **工程防护（Critic Warm-up）**：
  在 PPO 刚开始的若干 Steps，可设置策略模型（Actor）的学习率极小，甚至前几十步仅更新 Critic，待 Critic 预测误差平稳后再开启 Actor 的更新。

### 3. 为什么 Loss 中拟合的是 $V(s_t)$ 而非 $V(s_{t+1})$？
* **时间因果对齐**：
  * 在时刻 $t$，Critic 的输入只有前缀 $s_t$，它给出的预测是 $V_\phi(s_t)$；
  * 执行动作 $a_t$ 之后，现实观测到了从第 $t$ 步开始的真实总回报 $V_t^{target}$；
  * 损失函数考察的正是 Critic 在 $t$ 时刻的“事前预测”与“事后客观全貌”之间的差距：
    $$\text{Loss}_t = \frac{1}{2}\left( V_\phi(s_t) - V_t^{target} \right)^2$$
  * 对 $s_{t+1}$ 的预测考核，会在下一个时间步 $t+1$ 的局部损失中单独计算。

### 4. 为什么 Critic 损失截断处要取 $\max$（悲观上界）？
Critic 的损失函数形式为：
$$V_\phi^{clipped}(s_t) = V_{\phi_{old}}(s_t) + \text{clip}\left( V_\phi(s_t) - V_{\phi_{old}}(s_t), -\epsilon, \epsilon \right)$$
$$L^{Critic}(\phi) = \frac{1}{2} \hat{\mathbb{E}}_t \left[ \max\left( (V_\phi(s_t) - V_t^{target})^2, \; (V_\phi^{clipped}(s_t) - V_t^{target})^2 \right) \right]$$

* **数学原理（梯度防护墙）**：
  * Critic 的目标是**最小化 Loss**；
  * 当 Critic 更新过猛跑出截断范围时，截断项被卡在边界常数，而未截断误差随拟合继续变小，导致 $(V_\phi^{clipped} - V_t^{target})^2 > (V_\phi - V_t^{target})^2$；
  * 取 $\max$ 强行迫使 Loss 选用截断项；
  * 截断项对当前参数 $\phi$ 的导数为 **0**，**梯度瞬间被清零**！
  * 这相当于在 Critic 的更新边界上立起了一堵“刹车防护墙”，阻止其剧烈漂移。

---

## 六、Actor 截断目标（PPO-Clip）的深层机理

Actor 的优化目标是整个算法最核心的精髓：

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

取 $\min$ 构造了一个**悲观下界（Pessimistic Lower Bound）**。我们根据优势正负拆解其行为：

```
       Surrogate Objective
              ▲
              │                 / (未截断: r_t * A_t)
              │                /
   (1+ε)A_t ──┼───────────────/ (梯度截断为 0)
              │              /
              │             /
              │            /
         A_t ─┼───────────/
              │          /
              │         /
              └────────┴───────┴──────────────► r_t(θ)
                      1.0     1+ε
           【正优势 A_t > 0：好动作】
```

1. **当 $\hat{A}_t > 0$（好动作，应当鼓励）**：
   * 若比率 $r_t(\theta) \le 1+\epsilon$：正常计算梯度，增大该 token 概率；
   * 若比率 $r_t(\theta) > 1+\epsilon$：截断项卡在 $(1+\epsilon)\hat{A}_t$。因取 $\min$，目标函数锁定为常数，**对 $\theta$ 的梯度归零**！
   * **物理含义**：“虽然这是个好词，但本轮小样本上最多允许你涨 20% 的概率，杜绝贪婪爆炸。”
2. **当 $\hat{A}_t < 0$（坏动作，应当抑制）**：
   * 负数使得不等号反转，$\min$ 选择更严厉的惩罚项；
   * 即使比率下降严重，依然允许施加必要的负反馈，防止模型出现灾难性偏差。

### 完整 Actor 损失函数
在 PyTorch 中使用梯度下降优化，取负号并加入**策略熵（Entropy）**以维持生成多样性：
$$\text{Loss}^{Actor}(\theta) = - L^{CLIP}(\theta) \;-\; c_{ent} \cdot \mathcal{H}\left( \pi_\theta(\cdot \mid s_t) \right)$$

---

## 七、大模型 PPO-RLHF 端到端工程闭环与 PyTorch 实现

### 1. 两阶段解耦流程
整个算法的运行被严格划分为**两个解耦阶段**：

```
========================= 阶段 1: 交互采样 (Rollout Phase) =========================
1. 输入 Prompt Batch x
2. Actor 模型 (π_old) 自回归采样生成回答 y = [a_1, a_2, ..., a_T]
   - 记录 token 级对数概率: old_log_probs (标量张量, detach)
3. 冻结的 Reference 模型前向传播计算基准: ref_log_probs
4. 冻结的 Reward Model 评价整句完整语义: 产出全局分数 R_RM (经均值方差白化与截断)
5. 计算各步骤奖励: r_t = [-β*KL_1, ..., R_RM - β*KL_T]
6. 旧 Critic 模型预测前缀价值: old_values = [V_1, ..., V_T]
7. 反向递推计算 GAE: advantages, 并组装 Critic 目标 returns = advantages + old_values
8. 对 advantages 做批归一化 (Batch Normalization)

========================= 阶段 2: 参数训练 (Optimization Phase) =========================
将固定剧本 (x + y) 在同一批数据上循环训练多轮 (PPO Epochs):
   For epoch in 1 .. K:
       For mini_batch in dataset:
           a) Actor 模型前向传播: 针对完全相同的 token 计算 current_log_probs
           b) 概率比率: ratio = exp(current_log_probs - old_log_probs)
           c) 计算 PPO-Clip 目标并反向传播更新 Actor
           d) Critic 模型前向传播计算 current_values
           e) 计算 Value-Clip 均方误差并反向传播更新 Critic
```

### 2. 核心算法 PyTorch 代码实现

```python
import torch
import torch.nn.functional as F

def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """
    输入:
      rewards: [batch_size, seq_len] 每步即时奖励 (含 KL 与末尾 R_RM)
      values:  [batch_size, seq_len] Critic 预测的 V(s_t)
    输出:
      advantages: [batch_size, seq_len] 归一化后的优势值
      returns:    [batch_size, seq_len] Critic 的回归目标
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0.0

    # 从最后一步逆向递推
    for t in reversed(range(seq_len)):
        next_value = 0.0 if t == seq_len - 1 else values[:, t + 1]
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae_lam = delta + gamma * lam * last_gae_lam
        advantages[:, t] = last_gae_lam

    returns = advantages + values
    # 优势批标准化
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


def compute_actor_loss(current_log_probs, old_log_probs, advantages, clip_eps=0.2):
    """
    PPO-Clip Actor 损失函数计算
    """
    # 1. 重要性采样比率
    ratio = torch.exp(current_log_probs - old_log_probs)
    
    # 2. 未截断项与截断项
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    
    # 3. 悲观下界 (取 min) 并转为最小化损失
    actor_loss = -torch.min(surr1, surr2).mean()
    return actor_loss


def compute_critic_loss(current_values, old_values, returns, clip_eps=0.2):
    """
    带 Value-Clip 的 Critic 均方误差损失计算
    """
    # 未截断损失
    v_loss_unclipped = (current_values - returns) ** 2
    
    # 截断预测值
    v_clipped = old_values + torch.clamp(current_values - old_values, -clip_eps, clip_eps)
    v_loss_clipped = (v_clipped - returns) ** 2
    
    # 悲观上界 (取 max 确保越界时梯度归零)
    critic_loss = 0.5 * torch.mean(torch.max(v_loss_unclipped, v_loss_clipped))
    return critic_loss
```

---

## 八、相关实体与核心交叉链接

- 核心架构与算子：[[Transformers]], [[Decoder-Only Models]], [[Self-Attention Mechanism]]
- 训练与对齐全流程：[[RLHF Summary]], [[RLHF]], [[Fine-tuning]], [[Pre-training Large Language Models]]
- 专属实体页面：
  - [[RLHF]]：基于人类反馈强化学习的核心概念、MDP 形式化与演进图谱
  - [[Proximal Policy Optimization]]：算法理论演进与工业影响
  - [[Generalized Advantage Estimation]]：方差与偏差权衡的优势递推算法
  - [[Reward Model]]：人类偏好建模与打分器
  - [[Critic Model]]：状态价值网络与 Baseline 估算器
