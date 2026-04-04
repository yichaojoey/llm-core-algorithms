# DPO (Direct Preference Optimization) 实现指南

> 📖 **一句话解释**：DPO 直接绕过了在 RL 中显式的 Reward Model（奖励模型），通过直接利用极大极小的正负面偏好数据集，用二分类交叉熵替代掉了 PPO 中的复杂的策略梯度算法。“彻底干掉 Critic”。

---

## 目录

- [背景：RLHF的痛点与DPO的诞生](#背景)
- [Step 1：理论与公式推导（1分钟必背公式）](#step-1理论公式推导)
- [Step 2：实现 DPO 核心损失代码（2分钟）](#step-2实现-dpo-核心损失代码)
- [Step 3：封装与测试验证（1分钟）](#step-3封装与测试)
- [面试必考追问](#面试必考追问)

---

## 背景

在标准的 PPO RLHF 流程中，我们需要加载：
1. SFT 模型
2. Reward Model (奖励模型，训练成本高，占内存)
3. Actor 模型
4. Critic 模型 (Value 模型，通常跟 Actor 一样巨大)

共 4 个模型同台，对显存和超参调优都是灾难。**DPO** 的出发点是通过数学推导（依据 KL 散度与最优分布理论），得出结论：**强化学习中的最优策略本身就可以逆向去显式地表示出一个 Reward 函数**。于是，我们就能省去 Reward 和 Critic，全流程只需 Reference Model 和 Policy Model 参与前向计算即可。

---

## Step 1：理论与公式推导

> ⚠️ 面试官一定会让你在白板上手撕 DPO 损失函数的长相，并让你解释它的隐式 Reward 体现。

DPO 的隐式奖励建模为：
$$ r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} + C $$

给定一对人类偏好（或者编译器判断出的正误答案）作为响应对：$(y_{chosen}, y_{rejected})$。
通过经典的 Bradley-Terry 模型刻画：
$$P_{pref}(y_c \succ y_r | x) = \sigma(r(x,y_c) - r(x,y_r))$$

我们将隐式表达代入进去后，$C$ 这个扰动常数项会被完美相减抵消。于是，这就变成了一个我们只要使得 Chosen 概率大过 Rejected 的简单的**交叉熵分类损失**：
$$ \mathcal{L}_{DPO}(\theta) = - \mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_c|x)}{\pi_{ref}(y_c|x)} - \beta \log \frac{\pi_\theta(y_r|x)}{\pi_{ref}(y_r|x)} \right) \right] $$

---

## Step 2：实现 DPO 核心损失代码

### 代码骨架（纯享版）
```python
import torch.nn.functional as F

def compute_dpo_loss(policy_chosen_logps, policy_rejected_logps,
                     reference_chosen_logps, reference_rejected_logps, beta=0.1):
                     
    # Step 1: 获取 Log Ratio
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios
    
    # Step 2: 核心目标函数：-log(sigmoid(beta * diff))
    loss = -F.logsigmoid(beta * logits)
    
    return loss.mean()
```

> **注意**：在真实的大模型中，`Logps` 的获得是由在语言建模 `lm_logits` 取 `log_softmax`，然后在 token 进行 `gather`，最终对整个生成的非 pad 部分 `sum` 得到的概率。面试常考其维度对齐。

---

## 面试必考追问

**Q: DPO 中的隐式 Reward $R$ 是如何呈现/计算的？**
> Reward 正比于模型与参考模型的新旧概率比：$R \approx \beta (\log \pi_{\theta} - \log \pi_{ref})$。当新的 Policy 如果对一句非常好的话赋以更高的概率（拉开了跟原版 SFT 的差距），代表它隐式学到了更高的内部奖励。

**Q: 为什么代码实现中要有 `F.logsigmoid`？为什么不用 `- torch.log(torch.sigmoid())`**
> 防止数值溢出问题。如果参数过大 `sigmoid` 被推到抛出浮点计算的 `0.0` 时，直接 `log(0)` 会抛出 `NaN` 导致梯度爆掉。`F.logsigmoid` 的内部是用 `LogSumExp` 的数值稳定黑魔法去处理的。

**Q: DPO 有什么明显的短板吗？相对于 PPO 来说。**
> 第一，DPO 是 Offline（离线强化学习）算法，它只能死板地“看”以前固定的静态对比数据，不能像 PPO 一样在探索（Explore）过程中实时尝试发现新的错误解答或是更牛的最佳解。
> 第二，如果 Policy 更新过多，大大偏离了 Reference Model，整个散度近似假设基础就会破裂导致过拟合崩溃（模型开始说不知所云的胡话）。

---

## 验证方式
运行目录下的 `python demo.py` 或 `pytest test_dpo.py` 验证前向梯度的拉扯关系。
