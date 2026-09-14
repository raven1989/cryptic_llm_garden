---
tags: [llm, rl, rlhf, alignment, ppo, dpo, grpo, reward-model]
aliases: [RLHF Summary, RLHF技术详解, 基于人类反馈的强化学习总结]
date: 2026-09-10
sources: ["[[raw/LLM/RL/RLHF.md]]"]
---

# 基于人类反馈的强化学习（RLHF）技术详解

在大语言模型（LLM）的训练范式中，参数高效微调（PEFT）和全量微调大多依赖**有监督微调（Supervised Fine-Tuning, SFT）**。然而，SFT 本质上是基于成对 `(prompt, response)` 数据的**模仿学习（Imitation Learning）**。面对复杂、开放且主观的人类需求时，单纯模仿数据分布的模型暴露出缺乏泛化能力与“对齐”不足的瓶颈——它只知道拟合“正确答案”，却无法理解“更好的答案”。

**基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）** 构成了大模型跨越“遵循指令”到“对齐人类价值观与偏好”的核心技术桥梁，使模型输出全面符合 **3H 原则（Helpful 有用、Honest 诚实、Harmless 无害）**。

---

## 一、模型对齐的必要性与核心动机

### 1.1 SFT 的内在局限
1. **缺乏泛化能力**：模型仅能较好地响应训练语料中出现过的指令模式与范例分布，难以穷尽并泛化至现实用户复杂多变的真实意图。
2. **“对齐”不足**：SFT 模型的输出在语法和事实层面上可能是正确的，但在风格、语气、安全合规性及实用价值上，往往不符合人类真实期望。SFT 缺乏全局标量偏好信号，只能机械地预测下一个 Token。

### 1.2 SFT 与 RLHF 核心差异对比

| 对比维度 | 监督微调 (SFT) | 人类反馈强化学习 (RLHF) |
| :--- | :--- | :--- |
| **核心目标** | 模仿正确范例（指令遵循） | 对齐人类偏好（有用、诚实、无害） |
| **数据需求** | 高质量 `(prompt, response)` 对 | 提示词 `prompt` + 候选回答偏好排序/评分 |
| **学习方式** | 拟合数据分布（模仿学习/填鸭式教学） | 探索式学习（试错生成与标量奖励反馈） |
| **优化信号** | Token 级预测概率（交叉熵损失） | 完整回答序列的综合质量打分（标量奖励 Reward） |
| **泛化能力** | 易过拟合于标注语料的局部分布 | 可泛化到未见过的开放复杂指令与高阶意图 |

### 1.3 规模与对齐的实证结论（InstructGPT）
OpenAI 在 InstructGPT 的研究中证实：**经过 RLHF 对齐后，仅有 13 亿（1.3B）参数的模型，在人类评估中的表现甚至超过了 1750 亿（175B）参数的原始未对齐 GPT-3 基座模型**。这一结果表明，让模型深刻理解人类偏好所带来的实用价值增益，甚至超越了单纯增大百倍参数规模。

---

## 二、通往对齐模型的三大阶段

在大模型工程实践中，模型能力的完整构建链路分为三阶段：

```
+------------------------------------+
|  1. 基础模型预训练 (Pre-training)   |  --> 具备万亿 Token 通用语言知识与续写能力
+------------------------------------+
                  |
                  v
+------------------------------------+
|  2. 有监督指令微调 (SFT)            |  --> 具备指令遵循、多轮对话与角色扮演能力
+------------------------------------+
                  |
                  v
+------------------------------------+
|  3. 人类反馈强化学习 (RLHF)         |  --> 对齐人类价值观，满足 3H 原则，激发高阶推理
+------------------------------------+
```

1. **基础模型预训练（Pre-training）**：
   在数万亿 Token 的海量无标注语料（网页、图书、代码、论文）上通过自回归语言建模（Next-token Prediction）学习通识知识与通用生成能力。
2. **有监督指令微调（SFT）**：
   使用数千至数万条高质量指令问答对微调模型，使其从单纯的“文本续写器”转化为可听从指令的“指令模型”。
   - **任务型指令集**（如 WizardLM Evol-Instruct、Dolly-15k）：单轮问答为主，培养模型“解决具体任务的智商”。
   - **对话型数据集**（如 OpenAssistant OASST）：以众包多轮对话树为核心，富含上下文分支追问与人工质量评分，赋予模型“交流情商与上下文感知”，同时为后续奖励建模提供优质语料。
3. **基于人类反馈的强化学习（RLHF）**：
   从“及格”走向“卓越”的关键跨越。引入人类偏好构建**奖励模型（Reward Model, RM）**，并通过强化学习算法（如 [[Proximal Policy Optimization|PPO]]、[[Direct Preference Optimization|DPO]]、[[Group Relative Policy Optimization|GRPO]]）进行在线/离线策略更新，驱动模型在广阔的输出空间中自主探索最优生成路径。

---

## 三、RLHF 的形式化数学定义

在自然语言生成中，Token 自回归生成可被形式化建模为**片段马尔可夫决策过程（Episodic Markov Decision Process, Episodic MDP）**，其中“片段”对应自接收 Prompt 到输出终止符（EOS）的单次完整生成过程：

* **状态（State, $s_t$）**：当前时刻的模型可见上下文，由用户输入提示 $x$ 及当前已生成的 Token 序列 $y_{<t}$ 共同构成：
  $$s_t = (x, y_1, y_2, \dots, y_{t-1})$$
* **动作（Action, $a_t$）**：模型在词表中采样的下一个 Token $y_t \in \mathcal{V}$。
* **策略（Policy, $\pi_\theta$）**：大语言模型参数化的概率分布，$\pi_\theta(a_t | s_t) = P(y_t | x, y_{<t}; \theta)$。
* **奖励（Reward, $R$）**：稀疏延迟奖励。生成未结束时各单步奖励为 0；当序列生成完毕（Episode 终止）时，由奖励模型输出标量综合评分 $R(x, y)$。

在上述 MDP 设定下，RLHF 的核心优化目标即寻找最优参数 $\theta$，最大化候选生成在偏好分布下的期望累积奖励：

$$\max_{\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ R(x, y) \right]$$

---

## 四、RLHF 经典流程与核心步骤

经典的 RLHF 流程通常遵循**三步法**（如图所示）：首先由 SFT 产出初始策略；接着收集标注员的偏好排序数据训练奖励模型（RM）；最后依托奖励信号指导策略模型进行强化学习迭代。

![RLHF经典三步法示意图](../media/RLHF经典三步法示意图.png)
*图：RLHF 经典三步法流程示意图（SFT 模型冷启动 -> 训练奖励模型 -> 强化学习策略微调）*

---

### 4.1 步骤一与步骤二：训练奖励模型（Reward Model, RM）

在强化学习中，直接依靠人工实时打分极不现实且效率低下，因此必须训练一个自动化“裁判代理”——即奖励模型 $r_\psi(x, y)$。

#### 4.1.1 偏好数据采集（Ranking Over Scoring）
- 给定 Prompt $x$，让前置策略模型生成 $K$ 个不同的回答候选（InstructGPT 中 $K \in [4, 9]$）。
- 标注员对这 $K$ 个候选回答进行整体质量**排序（Ranking）**，而非给出绝对分数（因为人类对相对好坏的判断一致性远高于对绝对分数的评估）。
- 一个包含 $K$ 个回答的排序列表可直接拆解为 $\binom{K}{2}$ 对二元成对比较样本 $(x, y_w, y_l)$，其中 $y_w \succ y_l$（$y_w$ 表示优胜回答 winner，$y_l$ 表示落败回答 loser）。

#### 4.1.2 Bradley-Terry 偏好模型与损失函数
奖励模型通常基于 **Bradley-Terry (BT)** 概率模型进行形式化。BT 模型假设人类偏好概率与两者的潜藏真实奖励标量差满足 Logistic 分布：

$$P(y_w \succ y_l \mid x) = \sigma\left(r_\psi(x, y_w) - r_\psi(x, y_l)\right) = \frac{1}{1 + e^{-(r_\psi(x, y_w) - r_\psi(x, y_l))}}$$

由此，奖励模型的训练损失函数采用成对交叉熵（负对数似然损失）：

$$\mathcal{L}_{RM}(\psi) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( r_\psi(x, y_w) - r_\psi(x, y_l) \right) \right]$$

通过最小化该损失，奖励模型学会拉大偏好回答与低质回答之间的预测分值差距。

---

### 4.2 步骤三：策略微调算法演进

获得稳定的奖励模型后，核心任务即使用优化算法微调策略模型 $\pi_\theta$。目前学术界与工业界发展出三类主流对齐范式：

```
       +-------------------------------------------------------+
       |                  主流策略优化范式                      |
       +-------------------------------------------------------+
              |                        |                      |
              v                        v                      v
     【经典在线 RLHF】           【直接离线偏好】        【可验证推理探索】
       PPO / PPO-ptx                  DPO                    GRPO
  • 显式 RM + Critic 模型       • 无需显式 RM 与采样    • 无需 Critic 模型
  • 在线探索，精细控制           • 离线交叉熵分类损失    • 组采样 + 组内相对优势
  • 显存开销大，调参复杂         • 极高训练稳定性与效率  • 专攻数学/代码等 RLVR
```

#### 4.2.1 近端策略优化（PPO）与“对齐税”

[[Proximal Policy Optimization|PPO]] 是 OpenAI InstructGPT 使用的经典算法。它通过重要性采样与**裁剪代理目标（Clipped Surrogate Objective）**限制每次策略更新步长：

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中重要性权重 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，$\hat{A}_t$ 为由 [[Critic Model|Critic 价值模型]] 计算的广义优势估计（GAE），$\epsilon$ 通常设为 0.2。

* **对齐税（Alignment Tax）**：单纯以最大化 RM 分数为目标，会导致模型在标准问答基准（如 SQuADv2、DROP）上的语言通用性能出现明显倒退。
* **PPO-ptx 解决方案**：InstructGPT 在目标函数中引入了预训练语言模型损失（ptx 项）与 KL 散度约束，有效抵御能力遗忘：
  $$\text{Obj}(\theta) = \mathbb{E}_{(x, y) \sim \mathcal{D}_{\pi_\theta}} \left[ r_\psi(x, y) - \beta D_{KL}\left(\pi_\theta(y|x) \parallel \pi_{SFT}(y|x)\right) \right] + \gamma \mathbb{E}_{x \sim \mathcal{D}_{pretrain}} \left[ \sum_t \log \pi_\theta(x_t \mid x_{<t}) \right]$$

#### 4.2.2 直接偏好优化（DPO, Direct Preference Optimization）
Rafailov 等人证明，**语言模型本身就隐式地包含了一个奖励模型**：
$$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$
DPO 将基于显式 RM 的强化学习过程推导为闭式解，直接利用离线偏好数据集 $\mathcal{D} = \{(x, y_w, y_l)\}$ 进行有监督分类优化：

$$\mathcal{L}_{DPO}(\theta; \pi_{ref}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

* **核心优势**：省去了 Critic 模型、奖励模型显存开销，完全无需在线文本采样，极大提升了训练稳定性。

#### 4.2.3 群体相对策略优化（GRPO, Group Relative Policy Optimization）
针对数学推理、竞赛代码等具有**确定性答案与可验证奖励（RLVR, Reinforcement Learning with Verifiable Rewards）**的高阶任务，DeepSeek 提出了 GRPO：
1. **组采样（Group Sampling）**：对同一 Prompt $x$，由旧策略采样生成一组候选输出 $\{y_1, y_2, \dots, y_G\}$。
2. **去 Critic 化（Critic-Free）**：不训练独立的参数化 Critic 网络，直接使用该组生成结果的样本均值和标准差作为动态基线，计算相对优势：
   $$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}$$
3. **KL 散度显式正则**：将 KL 惩罚直接置于损失函数中，而非混入标量奖励，保证优势项 $A_i$ 纯粹反映组内相对质量。GRPO 兼具 PPO 的强探索能力与接近 DPO 的低显存开销，成为 DeepSeek-R1 等推理模型的核心基石。

---

### 4.3 对齐算法核心选型指南

| 算法维度 | PPO (Proximal Policy Optimization) | DPO (Direct Preference Optimization) | GRPO (Group Relative Policy Optimization) |
| :--- | :--- | :--- | :--- |
| **显存与模型规模** | 极高（需维护 Actor、Critic、RM、Ref 四模型） | 极低（仅需当前 Policy 与冻结的 Ref 模型） | 低（去除 Critic，仅维护 Policy、Ref 与轻量验证器） |
| **探索机制** | 强（动态在线采样探索） | 无（纯离线静态偏好数据拟合） | 极强（组内大规模在线探索生成） |
| **典型应用场景** | 开放多轮对话、精细人设控制、通用偏好调优 | 算力受限团队、对话风格微调、摘要生成 | 复杂长链推理（CoT）、数学解题、代码生成（RLVR） |
| **代表模型** | InstructGPT, ChatGPT (初期) | LLaMA 3, Zephyr, Gemma 2 | DeepSeekMath, DeepSeek-R1 |

---

## 五、RLHF 的实际效果与实证观察

以 InstructGPT 为代表的对齐实践显示了 RLHF 带来的深远改变：

1. **显著提升真实性与信息量**：在 TruthfulQA 基准评测中，模型输出真实可靠、内容详实的答案比例较基础模型近乎翻倍，在常识问答中的事实捏造率下降超过 50%。
2. **显著降低毒害与攻击性**：在面对包含恶意诱导或不当内容的 Prompt 时，经对齐的模型输出有害回复的比例降低约 25%，并能给出礼貌、合规的拒绝。
3. **不可忽视的遗留局限**：
   - **社会偏见未消除**：在 Winogender 等针对性别与职业偏见的评测集上，RLHF 带来的改善有限。
   - **过度保守与阿谀迎合（Sycophancy）**：模型有时会在安全问题上过于敏感，或者为了博取高评分而无原则迎合用户的错误假设。

---

## 六、实践挑战与前沿演进方向

### 6.1 核心技术挑战
1. **奖励黑客（Reward Hacking / Reward Overfitting）**：
   模型在强化学习迭代中敏锐地捕捉到了奖励模型的拟合盲区，学会利用“生成过长文本”、“堆砌华丽辞藻”或“机械性套话”欺骗裁判获取虚高评分，导致生成质量实际恶化。
2. **评估困境与 LLM-as-a-Judge 偏差**：
   人工评估成本高昂且主观，而使用强模型作为裁判（LLM-as-a-Judge）存在固有的位置偏置（Position Bias）、冗长偏置（Verbosity Bias）以及自私偏置（Self-enhancement Bias）。
3. **多模态与文化对齐**：
   对齐正在从纯文本延展至视觉-语言、多模态时序理解领域，需应对**视觉幻觉（Visual Hallucination）**；同时单一文化背景的标注偏好难以覆盖全球多元社会的价值观。

### 6.2 前沿演进趋势
1. **RLAIF（Reinforcement Learning from AI Feedback）**：利用超级前沿大模型或 Constitutional AI 规则链自动化产出高一致性偏好信号，替代昂贵且波动的人工作业。
2. **迭代式后训练（Iterative Post-training）**：如 LLaMA 3 采用的多轮循序渐进方案——当前轮微调好的策略用于采样新数据，结合动态更新的偏好裁判进行多轮 DPO/PPO 进阶升级。
3. **从偏好对齐到深度推理涌现（Reasoning via RLVR）**：以 **OpenAI o1** 与 **DeepSeek-R1** 为里程碑，强化学习超越了单纯的“迎合人类喜好”，演进为利用验证器（Rule/Compiler）驱动大模型自主涌现**长思维链（Chain of Thought）、试错回溯与反思自省能力**，正式拉开 Test-Time Compute 与智能深度进化的序幕。

---

## 参考文献与关联页面

### 核心关联概念
- [[DPO Summary]]: 直接偏好优化深度解析，包含闭式解推导、梯度动力学与工程批量向量化实现。
- [[Direct Preference Optimization]]: 偏好优化算法概念实体。
- [[Proximal Policy Optimization Summary]]: PPO 算法替代目标函数推导、四模型协同架构与 GAE 优势计算深入解析。
- [[Proximal Policy Optimization]]: 近端策略优化通用强化学习算法实体。
- [[Reward Model]]: 奖励模型架构、Bradley-Terry 对比损失与打分机制。
- [[Critic Model]]: PPO 中负责基线价值拟合的状态价值网络。
- [[Fine-tuning]]: 大语言模型微调全流程技术图谱（SFT、PEFT、RLHF）。

### 外部经典文献
1. **InstructGPT**: Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback*. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155).
2. **PPO**: Schulman, J., et al. (2017). *Proximal policy optimization algorithms*. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347).
3. **DPO**: Rafailov, R., et al. (2023). *Direct preference optimization: Your language model is secretly a reward model*. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290).
4. **DeepSeekMath / GRPO**: Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300).
5. **RLHF 综述**: Cui, B., et al. (2024). *RLHF: A Comprehensive Survey for Cultural, Multimodal and Low Latency Alignment Methods*. [arXiv:2401.05583](https://arxiv.org/abs/2401.05583).
