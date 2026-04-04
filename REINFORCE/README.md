# REINFORCE (Vanilla Policy Gradient) 面试指南

> 📖 **一句话解释**：这是所有纯 Policy Gradient（无 Critic 网络）体系比如 GPT/LLM 在进行对齐算法时的“祖师爷”。它的核心思想极为朴素：“如果你采取某个行动后，在整个对局结束算总账时拿到了一笔巨额奖励，那你就直接把当年产生那个分支行动的对数概率**暴力调高**！”

---

## 目录

- [背景：从哪里来，怎么演化成 PPO/GRPO 的？](#背景)
- [Step 1：理论与公式推导（如何使用折现）](#step-1理论知识)
- [Step 2：实现 REINFORCE 核心流并在代码中避免坑（2分钟）](#step-2代码实现)
- [面试必考追问：为什么要减去 Baseline？](#面试必考追问)

---

## 背景

REINFORCE 是大模型强化学习由 Value-based（如 DQN）向全概率网络 Policy-based 转型的基石。不要因为其古老就不背了，绝大多数大厂 ML 强化面试里，**让你十分钟在白板上手撕的代码经常是它而非巨庞大繁琐的 PPO。**

最原始版本的 REINFORCE 由于只按照当下的收益走，非常脆弱，容易因为一条路没走对就将整个模型梯度的方差扯爆。因此它后来演化出了两种极其成功的分支体系：
1. **老实交代加入 Critic**：花大量算力帮助模型学会评估当下的绝对价值 $\to$ 变成了 Advantage Actor-Critic (A2C)，最后加入防越界限制演变为了大名鼎鼎的 **PPO**。
2. **死也不用 Critic**：而是把一条 Prompt 提示词让模型同时跑好几次回答出句子，算这彼此几次的分布标准值当 Baseline 并取巧均值化 $\to$ 去除了高昂庞大的网络依赖，变成了在 O1/DeepSeekMath 中的 **GRPO / GSPO**。

---

## Step 1：理论知识

REINFORCE 本质上不会逐帧眼皮子底下评估每一步能拿多少分，它只认到达当前步（甚至是全剧终点）后的**总折现期望回报（Discounted Return）**：
$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

拿到这个折现包后，直接拿它作为一个乘法权重值，乘在使用 Actor 网络生成的对应行动的对数概率矩阵上，就可以求导求梯度了！
$$\nabla J(\theta) = \mathbb{E} \left[ \sum_t G_t \nabla_\theta \log \pi_\theta(a_t | s_t) \right]$$

如果在你的代码（比如使用 PyTorch 或者 TensorFlow）上要直接跑反向传播 `loss.backward()` 自动求得梯度的导数链条，那么根据上面恒等推导，你只需要构建出下面这样一个简单标量公式，因为优化器只认降低目标，因此只需要用负号求它最小化即可：
$$ \mathcal{L} = -(G_t \times \log \pi_\theta) $$

---

## Step 2：代码实现

```python
def compute_reinforce_loss(logprobs, rewards, gamma=0.99, use_baseline=True):
    returns = []
    G_t = 0
    # 🚨核心考点1：一定要逆序计算 (reversed)，因为“未来的决定不能影响过去的奖励”！
    for r in reversed(rewards):
        G_t = r + gamma * G_t
        returns.insert(0, G_t)
        
    returns = torch.tensor(returns)
    
    # 🚨核心考点2：要学会引入 Baseline。
    if use_baseline:
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        
    # 要求最大化因此在 Loss 那加了负号
    return -(logprobs * returns).sum()
```

---

## 面试必考追问

**Q: 在计算 $G_t$ 的实现时，为什么要写 `reversed()` 倒腾逆序来用 For 循环加起？**
> 这是基于马尔可夫决策体系（MDP）中严格的因果性时间箭头（Causality Arrow）。因为此时产生的作用不能改变上一秒的世界。在计算维度上：写正向双层嵌套去循环追溯计算会导致极其不雅观的 $O(N^2)$ 嵌套累加复杂度。而如果从序列末尾往前倒计算，我们仅凭一个简单的单参数变量通过一次循环 `G_t = r_t + gamma * G_{t+1}` 就能在这个 $O(N)$ 的线型复杂度内搞定这个巨大的时序推断！

**Q: 公式里的 Baseline 是什么？为何面试官看重让你在这里减去 Baseline？**
> 假设你处于某些不合理的环境下：保底分永远是 100 分。无论你写的乱七八糟还是完美全对，环境都会返回 100~150 全是巨大的正分评价。此时就算你大模型生成了极其恶劣毫无逻辑的话，因为 `G_t` 的硬绝对值依然是+100的正数，会导致这一步的错手操作因为被乘以了正数而依然顺理成章地将这个错误概率越推越大（在错路上越走越远很难改掉）。
> 
> 要彻底根治这个问题就要强制拔出对立极向——减去一个 Baseline （最基础就是这个 Episode 在全场生命周期里的平均分）。比如这回合你跑了均分 120 却只拿到 100，这一减它立马就原形毕露成了 -20 分的负数，变成一种惩罚去剥夺并缩小它产生这种回答的对数概率概率，实现真正的“奖惩分明”！这，也就是引发这后面浩如烟海各种花里胡哨 Advantage (优势度函数设计) 的开端。

---

执行 `python demo.py` 或依靠测试节点 `pytest test_reinforce.py` 回溯一下。
