---
tags: [rlhf, reward-model, alignment, preferences]
date: 2026-09-08
aliases: [RM, 奖励模型, 偏好模型, Reward Model]
sources: ["[[raw/LLM/RL/RL — Proximal Policy Optimization (PPO) Explained.md]]"]
---

# Reward Model (RM)

**奖励模型（Reward Model, RM）** 是大语言模型通过人类反馈强化学习（[[RLHF]]）实现价值对齐的核心打分器，负责将人类对模型生成文本的质量偏好量化为一个实数标量得分。

## 1. 训练时序与初始化基石

### 1.1 时序前置与离线冻结
* **严格前置**：在经典的 [[RLHF]] 三步法中，奖励模型必须在 **Step 2（RM 训练阶段）** 离线训练完毕。
* **PPO 阶段冻结（Freeze）**：在随后的 [[Proximal Policy Optimization]] 强化学习阶段，RM 的参数必须完全冻结，只作为前向推理算子提供打分。如果在策略更新的同时更新 RM，将引入强化学习中的“非平稳环境（Non-Stationary Environment）”，导致策略和打分标准双重震荡崩溃。

### 1.2 为什么必须使用 SFT 权重初始化？
RM 并非从零或未经微调的预训练基座（Base Model）直接初始化，而是必须继承 **SFT（监督微调）模型的权重**：
1. **语义理解与格式对齐**：SFT 模型已经完成了指令遵循（Instruction Following）和多轮对话格式对齐，具备对 Prompt 与 Response 之间相关性与语义逻辑的深层表征能力。
2. **避免冷启动与表面特征过拟合**：若从 Base 模型直接训练打分头，模型需要同时兼顾“理解复杂任务指令”与“辨别质量高低”，极易走捷径过拟合到表面统计特征（如文本长度、空洞礼貌用语）；继承 SFT 权重后，模型只需专注于学习偏好映射。

## 2. 模型架构与打分机制

### 2.1 架构改造（Scalar Head）
* **骨干网络**：采用与 SFT 模型相同结构的 Transformer Decoder。
* **投影层改造**：移除原有语言模型头部 `Linear(hidden_dim, vocab_size)`，替换为打分头（Value Head）`Linear(hidden_dim, 1)`，将每个 Token 处的隐层向量 $h_t \in \mathbb{R}^d$ 映射为一个实数标量 $r_t \in \mathbb{R}$。

### 2.2 推理工作流：是否仅在 `<EOS>` 产生得分？
在绝大多数主流对齐框架（如 InstructGPT、LLaMA-RLHF、DeepSpeed-Chat）中：
* **输入形式**：将完整上下文拼接为序列 `[Prompt, Response, <EOS>]` 送入 RM。
* **标量提取**：由于自回归因果掩码（Causal Mask），每个 Token 位置都会输出一个即时标量值 $[r_1, r_2, \dots, r_T]$。**但在最终结算整句回答偏好时，仅提取序列末尾最后一个有效 Token（通常是 `<EOS>`）处的标量 $r_{\text{EOS}}$ 作为全局环境奖励得分**。
* **PPO 中的信用分配**：
  * 对策略模型（Actor）而言，生成中间 token 时的环境奖励为 $0$（仅承受与参考模型的 KL 散度漂移惩罚）；
  * 仅在生成至 `<EOS>` 步时结算 $r_{\text{EOS}}$：
    $$R_t = \begin{cases} -\beta \cdot \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})_t, & t < T \\ r_{\text{EOS}} - \beta \cdot \mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})_T, & t = T \end{cases}$$
  * 该终端稀疏奖励（Sparse Reward）随后由 Critic（价值模型）通过 GAE（广义优势估计）向序列前序各 Token 反向传递信用。

## 3. 训练机制与工程优化

### 3.1 基于成对偏好的 Bradley-Terry 目标
人类标注员对文本难以给出绝对分值，故工业界采用成对排序偏好数据 $(x, y_w, y_l)$。基于 **Bradley-Terry (BT) 模型**，优胜文本 $y_w$ 相对于淘汰文本 $y_l$ 的被偏好概率建模为：
$$P(y_w \succ y_l \mid x) = \sigma\left( r_\psi(x, y_w) - r_\psi(x, y_l) \right) = \frac{1}{1 + e^{-(r_\psi(x, y_w) - r_\psi(x, y_l))}}$$

通过最小化负对数似然损失完成监督训练：
$$\mathcal{L}_{\text{RM}}(\psi) = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( r_\psi(x, y_w) - r_\psi(x, y_l) \right) \right]$$

### 3.2 训练损失的两种计算范式
1. **EOS-only Loss（经典主流范式）**：
   仅取 $y_w$ 和 $y_l$ 在序列尾部 `<EOS>` 处的标量差值计算交叉熵，直观且与最终强化学习使用方式对齐。
2. **Token-level 分歧监督（InstructGPT 方案）**：
   若两段回答较长且在很早的 Token 位置就产生逻辑分歧，仅看尾部可能导致中间表示监督效率不足。InstructGPT 方案定位两者出现分歧的首个位置 $t_{\text{diff}}$，自 $t_{\text{diff}}$ 之后的每个 token 位置均计算分歧排序损失，增强隐层表征；但**在部署推理给 PPO 时依然仅取 `<EOS>` 处的得分**。

### 3.3 Batch 构造与 $C_K^2$ 组合优化
在数据标注中，通常由标注员对同一 Prompt $x$ 生成的 $K$ 个候选回复（如 $K \in [4, 9]$）进行完全排序：
* 若直接将拆出的 $C_K^2$ 个成对样本随机 Shuffle 放入不同 Batch，不同批次间的梯度方差极大，极易造成绝对打分尺度的系统性漂移（Score Drift）。
* **标准做法**：将同一个 Prompt 下派生的所有 $C_K^2$ 个配对绑定在同一个训练 Batch 中联合计算并平均 Loss。这不仅能复用 Prompt 部分的前向计算，更能维持批次内打分参照系的稳定。

## 4. 架构演进：ORM 与 PRM

* **结果奖励模型（ORM, Outcome-supervised Reward Model）**：
  即上述传统 RM，在结尾处输出单一标量。对于常规开放式对话非常高效，但在复杂多步推理任务（数学定理证明、代码实现）中，容易因最后一步逻辑谬误而对前序完全正确的推导过程施加负向惩罚（信用分配失效）。
* **过程奖励模型（PRM, Process-supervised Reward Model）**：
  为每个推理中间步骤（Step-level）显式预测正确性得分，为强化学习与树搜索（MCTS）提供密集过程反馈（Dense Rewards），是现代推理模型（如 o1 系列范式）的核心对齐组件。

## 5. 与 PPO 及 DPO 的技术谱系关联

* **与 [[Proximal Policy Optimization]] 的关系**：RM 为 PPO 提供外生奖励标尺；同时由于 Critic 架构与 RM 完全一致，工业界常直接使用训练收敛的 RM 权重来冷启动初始化 Critic，大幅提升 RL 收敛效率。
* **与 [[Direct Preference Optimization]] 的关系**：DPO 绕过了独立的 RM 显式训练与采样环节，直接将 Bradley-Terry 偏好概率代入强化学习最优策略对偶解中，用策略网络的对数几率比隐式替代了 RM 打分。

参见完整研究报告：[[RLHF Summary]]、[[DPO Summary]]、[[Proximal Policy Optimization Summary]]，核心实体：[[RLHF]]、[[Direct Preference Optimization]]、[[Proximal Policy Optimization]]，相关技术：[[Fine-tuning]]。
