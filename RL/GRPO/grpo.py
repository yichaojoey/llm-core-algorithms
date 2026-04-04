"""
GRPO (Group Relative Policy Optimization) 面试核心实现
===================================================
去除了 Critic 的强化学习 PPO 变体算法。
核心公式：Advantages = (Rewards - Mean_group(Rewards)) / Std_group(Rewards)
由于同一 Query 抽样出的回答在相同场景下，直接取批内标准分，可以避免估计 V(s) 的高昂代价。
"""

import torch

def compute_grpo_loss(
    logprobs_new: torch.Tensor,       # [G, SeqLen]
    logprobs_old: torch.Tensor,       # [G, SeqLen]
    rewards: torch.Tensor,            # [G]
    ref_logprobs: torch.Tensor = None, # [G, SeqLen] (如果用于 KL 惩罚可传入)
    eps_clip: float = 0.2,
    beta: float = 0.01
):
    """
    计算基于单个 Query 对应 G 个采样的 GRPO Token-level 损失。
    
    Args:
        logprobs_new: Actor 模型跑出当前抽样结果各 token 对数概率矩阵
        logprobs_old: 旧 Actor 网络对相同结果产生的对数概率矩阵
        rewards: 这 G 个句子的总体得分/奖励 [G]
        eps_clip: 裁剪常数 (与 PPO 相同)
        beta: (可选) 对于 KL 强约束惩罚参数
    
    Returns:
        total_loss: GRPO Agent 的最终 Loss 返回标量
    """
    
    # 1. 组相对优势计算: 无需 Critic，同组内自带天然 Baseline
    mean_reward = rewards.mean()
    # Pytorch 默认自由度为 1 (unbiased=True)，为了对标严格公式通常取偏方差 unbiased=False
    std_reward = rewards.std(unbiased=False) 
    
    # +1e-8 防止全部抽样答案一样导致除 0 错误
    advantages = (rewards - mean_reward) / (std_reward + 1e-8)  
    
    # 按照 SeqLen 维度进行升维填充：[G] -> [G, 1]，方便后续与 [G, SeqLen] 的 Ratios 大小靠拢使用广播机制
    advantages = advantages.unsqueeze(1) 
    
    # 2. Token-level 重要性比率 ratios
    ratios = torch.exp(logprobs_new - logprobs_old.detach())
    
    # 3. PPO Surrogate Target
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
    
    # actor_loss: 最大化目标，加负号用以梯度下降
    actor_loss = -torch.min(surr1, surr2).mean()
    
    # 4. 可选: 计算跟 Reference 模型之间的 KL 惩罚，防止偏离参考分布
    total_loss = actor_loss
    if ref_logprobs is not None:
        # 近似计算 KL Div 控制在 Reference 周围
        ref_ratios = torch.exp(ref_logprobs.detach() - logprobs_new)
        # KL(P||Q) ≈ -log(Q/P)
        kl_penalty = -torch.log(ref_ratios).mean()
        total_loss = actor_loss + beta * kl_penalty
        
    return total_loss
