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
    
    # 2. Sequence-level 重要性比率 ratios 
    # 和 Token 相比，为了使得长度不会让不同长度间的对数差异指数爆炸，采取几何平均的方式！
    # Ratio = [ Prod(P_new/P_old) ] ^ (1/L) = exp( (Sum(logP_new) - Sum(logP_old)) / L )
    ratios = torch.exp((logprobs_new_sum - logprobs_old_sum.detach()) / seq_lengths) # [G]
    
    # 3. PPO Surrogate Target at Sequence Level
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
    
    # 4. 平均并取负供梯度求最小
    loss = -torch.min(surr1, surr2).mean()
        
    return loss
