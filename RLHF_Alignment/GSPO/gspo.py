"""
GSPO (Group Sequence Policy Optimization) 面试核心实现
===================================================
Qwen 团队提出的前沿 RLHF 算法。核心创新是将 Importance Ratio 的粒度从 Token-level 上升到 Sequence-level，
以此解决在长文本或 MoE 模型中将 Sequence 端点获取的 Reward 强涂在 Token 级别带来的高方差崩溃。
"""

import torch

def compute_gspo_loss(
    logprobs_new_sum: torch.Tensor,   # [G] (当前 Actor 对这 G 条生成的句子的累加/sum对数概率)
    logprobs_old_sum: torch.Tensor,   # [G] (旧 Actor 生成的 sum 对数概率)
    seq_lengths: torch.Tensor,        # [G] (每条生成序列的 token 长度)
    rewards: torch.Tensor,            # [G] (环境为各序列返回的 Reward 奖励)
    eps_clip: float = 0.2
):
    """
    计算基于单个 Query 对应 G 个采样的 GSPO Sequence-level Surrogate 损失。
    
    Args:
        logprobs_new_sum: 含有梯度的当前模型生成各个完整句子的对数概率标量和 [G]
        logprobs_old_sum: 没有梯度的旧模型对对应句子的对数概率和 [G]
        seq_lengths: 生成的序列文本有效长度 [G]
        rewards: 这 G 个句子的总体奖励评分 [G]
        eps_clip: 裁剪超参数
    
    Returns:
        loss: 标量
    """
    
    # 1. 组相对序列奖励优势计算 (与 GRPO 相同，去除 Critic 用相对标准化表示)
    mean_reward = rewards.mean()
    std_reward = rewards.std(unbiased=False) 
    advantages = (rewards - mean_reward) / (std_reward + 1e-8)  # [G]
    
    # 2. Sequence-level 重要性比率 ratios (区别于 GRPO 的最大要塞)
    # 【理论揭秘】：GRPO 对每一个局部的 Token 加注奖励，但如果我们环境评分只有在句子结束（比如验证代码能不能跑通）才能拿到结果呢？
    # 如果强行将整句奖励派分给各个字，当句子太长（Length>1024）：那些没犯罪的正常的无关词也受尽牵连或者全盘狂涨（积分累计黑洞带来爆炸方差）。
    # Qwen 的破局：不在字词层面惩罚，改在整个大句子（Sequence）级别去切蛋糕。
    
    # 数学上为了避免连乘多个字的概率导致长度不对等产生的惩罚，必须加上一个“平级开根号”（即几何平均操作）。
    # 几何比率 Ratio = [ Prod_{t=1}^L (P^{new}_t / P^{old}_t) ] ^ (1/L)
    # 落实在对数运算空间的降维写法上就是极其优美的一条命令： 1/L * (Sum(log P^{new}) - Sum(log P^{old}))
    ratios = torch.exp((logprobs_new_sum - logprobs_old_sum.detach()) / seq_lengths) # 稳稳压缩在 [G] 的级别不会膨胀
    
    # 3. Sequence Level Surrogate Target (序列级截断替身)
    surr1 = ratios * advantages
    # 这里由于 ratio 切分为句级，不再逐词截断。而是把整个大答案当做一块完整的方糕执行限制 eps_clip
    surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
    
    # 4. 平均并取负供梯度求最小
    loss = -torch.min(surr1, surr2).mean()
        
    return loss
