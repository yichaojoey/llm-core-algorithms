# RLHF 强化学习核心算法：代码级细节对比 (REINFORCE vs PPO vs DPO vs GRPO vs GSPO)

这份文档抛弃“更稳定、更快”这类宽泛的比较，而是直击**手撕代码时的细节痛点**。通过对比它们在输入张量（Tensor）形状、网络依赖、梯度流、以及特征标量计算上的根本区别，帮助你在写代码或者面试时思路清晰。

---

## 宏观依赖与架构对比

| 算法 | Actor (生层网络) | SFT Reference | Critic (Value) | Reward Model | 优化范式 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **REINFORCE** | 需要 (单网络) | 不需要 | 不需要 | 需要 | 在线采样，时序折现回传 (Monte Carlo) |
| **PPO** | 需要 (且保持一个 Old) | （可选项, KL 用）| **绝对需要** | 需要 | 在线采样，时序计算 (GAE) |
| **DPO** | 需要 (不保留 Old) | **绝对需要** | 不需要 | 不需要 (自带机制) | 离线固定对决对 `(y_w, y_l)` |
| **GRPO**| 需要 (且保持一个 Old) | （可选项, KL 用）| 不需要 | 需要 | 在线采样，同一 Query 多答 |
| **GSPO**| 需要 (且保持一个 Old) | （可选项, KL 用）| 不需要 | 需要 | 在线采样，序列级整体比对 |

---

## 具体代码细节实录对比

### 0. REINFORCE vs PPO：Baseline 与 Critic 的代沟
面试时 REINFORCE 作为一切梯度的基石，其代码极为纯粹，但这使得它的方差极大。PPO 通过引入 Critic 完美解决了这个问题：
*   **REINFORCE 的折现与惩罚**：代码中通过 `reversed()` 对 `rewards` 求和得出 `G_t`。由于没有 Critic，只能依靠这个 `G_t` 减去一条轨迹或多条轨迹的均值（纯标量计算的 Baseline）来压制那些“全局全加正分导致动作全部盲目变大”的灾难。
    *   **代码核心段**：`returns = (returns - returns.mean()) / (returns.std() + 1e-8)`
*   **PPO 的绝对价值替换**：代码中引入了 Critic（同样是一个巨大的网络模型），在每一个时刻吐出 $V(s)$ 作为高纬度的复杂参考分，将“简单的历史均值”的极简 Baseline 变为了能精准锚定各个时间步相对优势的 Advantage $\hat{A}$ 函数。

**⚠️ 代码细节点**：写 REINFORCE 时一定要熟练手搓且向面试官说明**具有因果倒推**的 `reversed(rewards)` 折扣求和步骤！

### 1. PPO vs DPO：“找梯度”的途径不同
面试手写代码时，最容易混淆两者究竟在优化什么：
*   **PPO 是“乘法+截断”**：代码里一定有 `ratio * advantage` 以及 `clip(ratio)*advantage`。PPO 是基于当前产生的某个行动所处的回报有多大（Advantage），去放大或缩小产生这个行动的概率。
    *   **代码核心段**：`surr = ratio * advantages`
*   **DPO 是“商的指数+Sigmoid”**：DPO 的底层是 Bradley-Terry 分布，它的目标只是一道单纯的二分类交叉熵选择题。DPO 是拿新模型对这两个句子的打分差距减去参考模型对这两个句子的打分差距，来做 `logsigmoid`。
    *   **代码核心段**：`logits = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)`
    *   **代码核心段**：`loss = -F.logsigmoid(beta * logits)`
    
**⚠️ 代码细节点**： PPO 所需的是所有抽样出的结果作为独立条目处理。DPO 强制要求数据流中带着 **Chosen** 和 **Rejected** 为一对的张量！

### 2. PPO vs GRPO：Advantage 的维度发生了质变
在 PPO 和 GRPO 中，核心都在于计算出那个神秘的 $\hat{A}$ (Advantage) 乘以 Ratio。但两者对 `Advantage` 的维度处理代码天差地别！

*   **PPO 的 Advantage**：
    *   **维度**：`[Batch, SeqLen]` 甚至更带有 Time-step 的色彩。
    *   **代码做法**：必须通过 Critic（Value 网络）预估每一个格子的值。甚至还要做带有 $\lambda$ 折扣因子的时间逆序 GAE：`adv_t = delta_t + gamma * lam * adv_{t+1}` 然后逆向 `insert(0)` 返回。
*   **GRPO 的 Advantage**：
    *   **维度**：只和组有关系，即 `[Group_size]` （对应每一个不同的句子作为一个整体标量）。
    *   **代码做法**：不需要任何 Critic！当同一个问题拿到了 $G$ 个不同的回答并分别打了 $G$ 个分后。直接取均值和方差：
        ```python
        advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
        advantages = advantages.unsqueeze(1) # 由 [G] 广播扩充成 [G, 1] 使得能和 Token Tensor 对齐乘法
        ```

**⚠️ 代码细节点**：写 GRPO 时，由于 Advantage 一开始只是一个纯标量数组（维度 `[G]`），在和 `[G, Seq_len]` 的 Token Ratios 相乘之前，千万不要忘记调用 `.unsqueeze(1)` ！

### 3. GRPO vs GSPO：Token 级限制 $\to$ Sequence 级限制
这可能是目前最值得推敲的面试考点。Qwen 爆火的 GSPO 在代码实现上到底改了 GRPO 哪里？
两者都是 `[Group_size]` 级别的同一道题重复做打分。但 Surrogate（裁剪比例目标）发生了降维：

*   **GRPO（Token-level Surrogate）**：
    *   GRPO 在组内打分得出 Advantage 后，由于其底层是 PPO 思路，它依然看重**每一个字**。它保留了巨大的概率矩阵来算比率。
    *   **代码**：`ratios = torch.exp(logprobs_new_matrix - logprobs_old_matrix)` 此时 ratio 尺寸高达 `[G, SeqLen]`。随后进行巨大的 `torch.clamp(ratios, 0.8, 1.2) * advantages_unsqueezed`，对每一个词应用裁剪！
*   **GSPO（Sequence-level Surrogate）**：
    *   在写代码时，GSPO 不再接受庞大的二维概率矩阵，而是接收**对数概率之和 (Sum)**！
    *   **代码的蜕变**：GSPO 首先消除了那个 `.unsqueeze(1)`，因为它干脆不在意单独的字了。
        ```python
        # 由于是求指数里的平均，等于外面的开 Seq_length 根号：
        ratios = torch.exp((logprobs_new_sum - logprobs_old_sum) / seq_lengths)
        ```
    *   此处 `logprobs_new_sum` 已经是尺寸刚好为 `[G]` 的一维数组了。它和 `advantages [G]` 完美结合，直接做成只有 $G$ 次计算量的标量求最小值，且直接在**句子的抽象总强度**层面上做出了 `torch.clamp(ratios, 0.8, 1.2)`！

**⚠️ 代码细节点**：
- GRPO 中 `logprobs` 是矩阵：包含每个 Token，最后求 loss 要对行和列一起 `mean`。
- GSPO 中 `logprobs_sum` 已经是行（序列）和了！直接利用 `seq_lengths` 来缩放，大幅避免了模型回答太长导致长尾效应的对数累加黑洞带来的方差失控。

---

## 面试总结 Checklist
如果考官让你写某一个，记住以下**防挂科挂钩点**：
0. **写 REINFORCE**：折扣回报 `G_t` 算的时候是不是从后往前通过 `reversed()` 因果回追计算了？最后是否减去了全局/Episode `mean()` 均值做 Baseline 控制偏移？
1. **写 DPO**：有没有使用 `F.logsigmoid` 防溢出？有没有漏掉 `beta` 乘法？有没有做 `pi - ref` 的差值运算？（如果用除法就是没搞懂对数域）
2. **写 PPO**：GAE 的倒序循环写对了吗？`next_value` 需要在序列结束或者补零，搞清楚了吗？
3. **写 GRPO**：组内归一化的方差是否加了 `1e-8` 防除零？方差是否设定成了无偏态 `unbiased=False`？计算 Ratio 之前是否用 `unsqueeze(1)` 展开了 Advantage 去顺配后续长长的序列。
4. **写 GSPO**：输入是不是换成整句总概率了（Sum）？指数里面是否巧妙地除以了 `seq_lengths` 完成求几何平均从而避免长度惩罚和方差爆炸了？
