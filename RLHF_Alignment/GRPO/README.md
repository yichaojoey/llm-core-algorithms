# GRPO (Group Relative Policy Optimization) 面试指南

> 📖 **一句话解释**：对一道 Query 原型生成 G 个结果打分。取消了传统 PPO 中负责评价 V(S_t) 的那一套庞大复杂的 `Critic`，改用这 G 个回复在这个场景里的“评分标准分”（也就是把这几个分数归一化）拿去作为各自 Token 的归一 Advantage！简单而稳定。

---

## 目录

- [背景：PPO/DPO中存在的局限，DeepSeekMath的破局](#背景)
- [Step 1：理论与公式推导（1分钟明白去 Critic 的逻辑）](#step-1理论推导)
- [Step 2：现成可实现的 GRPO 核心代码（2分钟）](#step-2代码实现)
- [面试必考追问](#面试必考追问)

---

## 背景

在进行大模型的纯逻辑代码强化学习时（如 DeepSeek Math/Coder 等）：
- DPO 的痛点：DPO 完全被参考所固定，它是通过已有的 pair 完成的，对于数学这种只需要看正误但是过程可以无限广的数据集探索是不利的。
- PPO 的痛点：必须要同时起 Actor 和 Critic，大模型训练环境显存紧张，多起一个等大的 Critic 将带来高昂的基础设施代价与同步通讯问题。

基于 **REINFORCE Leave-One-Out** 等方法的启发，GRPO 让模型多次解答同一道题生成 $G$ 个样本，给这 $G$ 个样本全打个分，再从这组分数自己内部算出一个带均值和方差控制的相对 Advantage，作为 Token Level 的 Surrogate 乘子！由于每道题自己跟自己比，天然不需要知道“这道题全局有多难”的绝对价值 V(s) 了。

---

## Step 1：理论推导

在同一个提示词/题目 $x$ 中，Actor 模型通过自己的采样给出了 $G$ 个结果：$y_1, y_2 ... y_G$。
评价器或奖励函数把这些回复转换为 $G$ 个奖励：$r_1, r_2 ... r_G$。

接着进行相对归一化操作获得 $\hat{A}_i$ ( Advantage)：
$$\hat{A}_i = \frac{r_i - \text{mean}(r_1..r_G)}{\text{std}(r_1..r_G)}$$

因为有正有负，直接将此替换进 PPO 内 Token 级的 Objective 函数：

$$J_{GRPO}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \left( \sum_{t=1}^{|y_i|} \min\left(ratio_{i, t} \hat{A}_i, \text{clip}( ratio_{i, t} ) \hat{A}_i\right) - \beta KL_{ref} \right)  \right]$$

---

## Step 2：代码实现

```python
def compute_grpo_loss(logprobs_new, logprobs_old, rewards):
    # 1. 消除 Critic! 计算自带均值的内部 Z-Score 当作 Advantage
    mean_reward = rewards.mean()
    std_reward = rewards.std(unbiased=False) 
    
    # shape从 [G] 延展为 [G, 1], 让序列能够广播
    advantages = ((rewards - mean_reward) / (std_reward + 1e-8)).unsqueeze(1)
    
    # 2. 跟 PPO 一模一样的 Clip 和 Ratio
    ratios = torch.exp(logprobs_new - logprobs_old)
    
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantages
    
    # 最大化因此加了负号
    return -torch.min(surr1, surr2).mean()
```

---

## 面试必考追问

**Q: 为什么 GRPO 天然去掉了 Critic?**
> 传统 PPO 用 Critic 预估当前状态的绝对分 $V(s)$ 作为 Advantage = $R - V(s)$ 的 Baseline 以降低方差。但在 GRPO 中，由于同一时刻我们回答了同一道题 $G$ 遍，这 $G$ 遍得分的平均分其实已经是这个状态下所有可行回答的最完美蒙特卡洛积分的 Baseline 均值表达了。

**Q: GRPO 中 $G$ （抽取总次数）如果太小或者太大怎么办？**
> 对于 7B 或 72B 级的模型，通常我们会在内存与效果间权衡，$G$ 普遍设定在 4 到 8 或 16。如果 $G$ 过小（比如 2），标准差计算极不准确反而放大了波动，此时不如退化成只计算正反向对决的 DPO；而太大会完全耗尽显存和增加 Rollout 时间。

---
运行此演示 `python demo.py` 或测试 `pytest test_grpo.py` 查阅其数值逻辑。
