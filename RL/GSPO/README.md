# GSPO (Group Sequence Policy Optimization) 面试指南

> 📖 **一句话解释**：这是 Qwen 团队针对传统 RLHF 设计提出的重磅算法。将 PPO/GRPO 中在单个 Token 上进行评估惩罚的做法（Token-level），拔高升级到针对整句生成结果做几何平均（Sequence-level）。通过天然抛弃 Critic、在全局求比例截断，大幅度解决了长文本和 MoE 大模型中奖励由于 Token 波幅积累引发的可怕方差爆炸。

---

## 目录

- [背景：Token-level 的方差噩梦](#背景)
- [Step 1：理论与公式推导（如何转为 Sequence-level）](#step-1理论知识)
- [Step 2：实现 GSPO Sequence Surrogate（2分钟）](#step-2代码实现)
- [面试必考追问](#面试必考追问)

---

## 背景

无论是基于 PPO 还是去 Critic 化的 GRPO，我们最终的 Reward 大多是**当模型吐出最后的结束 Token（EOS） 之后才统一根据判断规则给出的一口价分数**。
如果把一句话末尾的一口价强行分摊给之前的每一个 Token，而且 Token 这个层面的 Ratio $\frac{P_{new}(t)}{P_{old}(t)}$ 又是独立的且相互剧烈连乘波动的。对于上千字的推理过程而言（如长代码生成、O1 范式思维图），方差噪声会巨大。尤其对 Expert 路由极其敏感的 MoE 模型而言更易崩溃断层。

因此提出思考：有没有可能让优化裁决的**粒度**和奖励的**粒度**完全一致呢？既然你是给整个 Sequence 打分的，那我就针对这个 Sequence 作为整体的表现比例来算 PPO Clip 裁决！这就是 **Sequence-level RLHF: GSPO**。

---

## Step 1：理论知识

GSPO 中抛弃每个步骤 Token $t$ 的 Advantage 推断，直接将序列重要性 $s_i(\theta)$ 作为一个独立的唯一标量抽取表达：
$$s_i(\theta) = \left( \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)} \right)^{\frac{1}{|y_i|}}$$

取 $\frac{1}{|y_i|}$ (长度的倒数) 次幂是为求**几何平均数**，以此来做基准缩放并防范长句子对数连乘大爆炸的现象。而在实现中，我们都会取对数域（LogProb），此时它极其优雅——直接蜕变成了“整体序列差值的简单平均”：
$$s_i(\theta)= \exp\left( \frac{1}{|y_i|} (\log \pi_{\theta_{sum}} - \log \pi_{\theta_{old_{sum}}}) \right)$$

获得代表整个序列的标量 $s_i$ 后，放入 Surrogate 函数去 Clip！
$$J_{GSPO}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left(s_i(\theta) \hat{A}_i, \text{clip}(s_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i\right) \right]$$
其中 $\hat{A}_i$ 与 GRPO 一样利用（去 Critic，自身互相参考比较的）批内标准差归一化求得。

---

## Step 2：代码实现

```python
import torch

def compute_gspo_loss(logprobs_new_sum, logprobs_old_sum, seq_lengths, rewards, eps_clip=0.2):
    # 1. Advantage 归一化 (无 Critic 化，天然对比 Baseline 分)
    mean_reward = rewards.mean()
    std_reward = rewards.std(unbiased=False) 
    advantages = (rewards - mean_reward) / (std_reward + 1e-8)  # -> shape: [G]
    
    # 2. Sequence-level 重要比率 (几何平均处理)
    # ratio_seq = exp( (log_new_sum - log_old_sum) / seq_len )
    # 面试加解法绝杀：从此告别序列长度的 Padding 循环和 Token Mask 对齐等恶心操作！代码极其清爽
    ratios = torch.exp((logprobs_new_sum - logprobs_old_sum.detach()) / seq_lengths)
    
    # 3. PPO Surrogate Target
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
    
    # 4. 平均并取负供梯度求最小优化 (返回标量)
    loss = -torch.min(surr1, surr2).mean()
        
    return loss
```

---

## 面试必考追问

**Q: Token-level 相比 Sequence-level 问题到底在哪？**
> 以传统 PPO 为例，一旦某句话中途一个前置 Token 的概率受惩罚/奖励发生断崖改变，不仅后续 Token 都受影响，而且该词比率 $r_t$ 的极大极化会使得整个模型连乘回放传过来的梯度长链完全失控。而 GSPO 在等式外面把它全部当成单一乘积做几何平均打散，然后一次性进行唯一一次 $\text{clip}$，极具备长文数学鲁棒性。

**Q: GSPO 与 GRPO 在代码实现上的具体显著区别？**
> 第一：GRPO 是算了一个巨大的 Token-level 矩阵 $[G, SeqLen]$ 出来再调用 `clamp_min` 限制。而 GSPO 内部自始至终是一个纯标量的一维数组 $[G]$。
> 第二：由于不用算时间步（TimeStep）的优势递推，抛弃了复杂的 `cumsum`、反转 `reversed` 或 `Generalized Advantage Computation (GAE)` 倒序累加。这让它的代码复杂度和运算前向图深度开销近乎只相当于原始代码的零头级别，计算得快且稳！

---

## 验证
运行此演示 `python demo.py` 或跑通 `pytest test_gspo.py` 来查阅极其纯净但效果强悍的大巧不工代码。
