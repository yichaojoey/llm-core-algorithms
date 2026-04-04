# PPO (Proximal Policy Optimization) 算法从零实现指南

> 📖 本教程按照**面试手写代码的顺序**组织，每一步都是 **理论 → 代码**。
> 跟着走一遍，你能在面试中迅速且清晰地写出 PPO 的核心结构。

---

## 目录

- [背景：为什么需要 PPO](#背景为什么需要-ppo)
- [Step 1：写 Actor-Critic 网络结构（1分钟）](#step-1写-actor-critic-网络结构)
- [Step 2：实现 GAE 计算（3分钟）](#step-2实现-gae-计算逻辑)
- [Step 3：实现 PPO-Clip 目标函数与更新逻辑（4分钟）](#step-3实现-ppo-clip-目标与更新)
- [Step 4：封装交互流程与 Rollout（1分钟）](#step-4封装交互流程)
- [面试常见追问](#面试常见追问)
- [运行方式](#运行方式)

---

## 背景：为什么需要 PPO

在 RL 面试中，**TRPO** 过于复杂（要求解共轭梯度和海森矩阵），而 **REINFORCE (Vanilla PG)** 又非常不稳定。
**PPO** (Proximal Policy Optimization) 由 OpenAI 在 2017 年提出，兼具了 Trust Region 的稳定性和一阶优化（Adam）的简单性，并且成为了大模型 RLHF (Reinforcement Learning from Human Feedback) 的基石算法。

> **核心思想**：
> 1. 通过计算 **Ratio** ($r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$) 来衡量新旧策略的区别。
> 2. 使用 **Clip 操作** 将 Ratio 限制在 $[1-\epsilon, 1+\epsilon]$，从而保证每次策略更新都在一个安全（Proximal）的范围内，不发生突变。
> 3. 结合 **GAE (Generalized Advantage Estimation)** 极大降低采样的方差从而实现更稳定的训练。

---

## Step 1：写 Actor-Critic 网络结构

### 对应理论
Actor 负责输出每个动作的概率，Critic 负责输出当前状态的 Value $V(s)$。在离散动作场景下，Actor 最后必须要接一层 `Softmax` 吐出多分类变量（或者利用 Category 分布处理 logits）。

### 对应代码骨架
```python
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        # Actor 输出对应动作的分数/概率
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=-1)
        )
        # Critic 输出当前 State 的评估值预测 V(s)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
```

> 🎯 **面试技巧**：面试时如果没有规定环境，直接写全连接前馈网络 (MLP) 最快最稳。

---

## Step 2：实现 GAE 计算逻辑

### 对应理论：权衡偏差和方差
常规的 Return 方案是直接累加折扣奖励，这种形式的方差极大。GAE 通过引入 $\lambda$ 参数平滑单步 TD-Error 与 多步 TD-Error，显著降低优势估计的方差。

**单步 TD Error**: $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
**GAE Advantage**: $A_t = \delta_t + \gamma \lambda A_{t+1}$

> 在实现时采用**逆序计算**，这是写 GAE 的标准手作范式，必须掌握。

### 对应代码片段
```python
def compute_gae(rewards, state_values, is_terminals, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    # Next value 我们假定为 0，方便统一处理最后一步的溢出越界
    values = state_values + [0]
    
    # 注意：这里需要 reversed 逆序计算
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - is_terminals[step]) - values[step]
        gae = delta + gamma * lam * (1 - is_terminals[step]) * gae
        advantages.insert(0, gae)
        
    returns = [adv + val for adv, val in zip(advantages, state_values)]
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)
```

---

## Step 3：实现 PPO-Clip 目标与更新

### 对应理论
PPO 在一次 Rollout 数据收集完毕后，会利用 `buffer` 中的数据进行 $K$ 个 epoch 的更新。
- **Actor Loss**: $\min( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t )$
- **Critic Loss**: $\text{MSE}(V_\theta(s), \text{returns})$

这是 PPO 被发明的**最核心代码段**，面试时**必定要手写出来并能够清楚解释**其中每个变量的含义。

### 对应代码片段
```python
# 1. 计算 Ratio: 新策略概率 / 旧策略概率
# (用相减的指数量 exp(log_new - log_old) 避免直接除法的数值下溢)
ratios = torch.exp(logprobs - old_logprobs.detach())

# 2. 计算 Surrogate Loss
surr1 = ratios * advantages
surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

# 3. Actor 损失要取负号，因为我们要用 Adam 来进行最大化
loss_actor = -torch.min(surr1, surr2).mean()

# 4. Critic 损失 (利用 MSELoss 让其逼近 GAE 算出好的 Return)
loss_critic = nn.MSELoss()(state_values, returns)

# 5. Backward & Step (也可以加入 entropy 鼓励探索)
loss = loss_actor + 0.5 * loss_critic - 0.01 * dist_entropy.mean()
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()
```

---

## 面试常见追问

**Q: 为什么要有 PPO Clip？**
> REINFORCE 每一步更新后，新旧策略差别可能极大，导致训练崩溃。PPO 的 Clip 限制了本次 update 前后输出概率的变化不超过 $1\pm\epsilon$，使得优化步伐“求稳”，即 Proximal（近端）的体现。

**Q: `ratios` 是 $\frac{\pi_{new}}{\pi_{old}}$，为什么实际代码里是用 `exp(log_new - log_old)`？**
> 为了数值稳定性。计算机算概率相除极易遇到浮点数越界问题，所以在神经网络 Actor 模型中我们常常存储并维护 `log_prob`，利用对数差的指数还原回原始商值。

**Q: Advantage 是什么？它和 Reward 的区别是？**
> Reward 是环境直接给出的标量反馈奖励；Advantage（优势）是“当前动作相对于在这个状态下平均动作 好多少”。Advantage 的均值为 0，用它可以极大地降低训练迭代过程的方差。

**Q: 为什么 GAE 计算要逆序 (reversed)？**
> 因为计算 $A_t$ 需要依赖于未来的 $A_{t+1}$，自后向前计算能利用递推公式让时间复杂度从 $O(N^2)$ 降为 $O(N)$。

---

## 运行方式
执行以下命令即可在控制台看到虚拟环境中 PPO 的网络参数更新日志（无外部 Gymnasium 依赖要求等）：

```bash
python demo.py
```

---

## 文件结构

```
PPO/
├── README.md     ← 你正在读的教程（面试步骤导向）
├── ppo.py        ← PPO 核心实现（带详细中文注释，重点看 PPO_Clip 和 GAE 阶段部分代码）
└── demo.py       ← 验证演示代理运行网络计算闭环的一个无环境依赖要求的设计脚本
```
