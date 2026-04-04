"""
DPO (Direct Preference Optimization) 核心实现代码
===============================================
以极简风格构建，这是近两年面试问得极其频繁的代码题。
其核心在于不依赖外部 Critic 和繁琐采样的 -log(sigmoid(beta * (logits)))
"""

import torch
import torch.nn.functional as F

def compute_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1
):
    """
    计算基于 Batch 的 DPO 损失。
    
    Args:
        policy_chosen_logps: 新策略(Actor)产生 Chosen 序列句子的累加对数概率 (Batch,)
        policy_rejected_logps: 新策略(Actor)产生 Rejected 序列句子的累加对数概率 (Batch,)
        reference_chosen_logps: 参考模型(Ref)对 Chosen 句子的累加对数概率 (Batch,)
        reference_rejected_logps: 参考模型(Ref)对 Rejected 句子的累加对数概率 (Batch,)
        beta: 控制 KL 惩罚力度的超参数，通常为 0.1~0.5
        
    Returns:
        loss, chosen_rewards, rejected_rewards
    """
    
    # 1. 计算 Preference Model 的隐式 Reward (Reward ∝ log(pi / ref))
    # 数学上利用对数减法等同于商：log(A/B) = log(A) - log(B)
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # 2. R_chosen - R_rejected (奖励优势差值)
    logits = pi_logratios - ref_logratios
    
    # 3. DPO Loss = -log(sigmoid(beta * logits))
    # 使用 PyTorch 原生的 F.logsigmoid 防止 exp(-x) 由于数值过高带来的溢出
    loss = -F.logsigmoid(beta * logits)
    
    # 4. (额外) 监控指标计算，脱离梯度流返回
    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    
    return loss.mean(), chosen_rewards.mean(), rejected_rewards.mean()
