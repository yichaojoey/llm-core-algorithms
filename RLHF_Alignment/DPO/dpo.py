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
    
    # 1. 计算 Preference Model 的隐式 Reward (隐式奖励)
    # 【理论揭秘】：DPO 绝杀了 Reward Model 的精髓：它推导出你不需要另开显存放一个 Reward 预估网络！
    # 如果两个模型分别给出了回答对数概率 $\pi_{target}$ 和 $\pi_{ref}$，
    # 那么奖励分数 Reward 在数学上严格正比于它们对数概率的差值： R(x,y) = beta * log( pi_target(y|x) / pi_ref(y|x) )
    # 代码上利用对数特性完美替代除法：log(A/B) = log(A) - log(B)
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # 2. 计算 logits = R_chosen - R_rejected (好的回答的隐式奖励 - 坏的回答的隐式奖励)
    # 实际上就是通过交叉参照把好差区分开来。
    logits = pi_logratios - ref_logratios
    
    # 3. DPO Loss 目标锁定 = -log(sigmoid(beta * logits))
    # 【理论揭秘】：基于 Bradley-Terry 模型（用于体育队伍对决概率的古典算法）。
    # 如果 chosen 比 rejected 好，差值越大越好！为了在梯度流中不炸大数边界引发不稳，
    # 我们将差值丢入 sigmoid 让它介于 0和1，再求对数期望。
    # 代码坑点：必须要使用 PyTorch 内置的 F.logsigmoid，如果你手写 -torch.log(torch.sigmoid(logits)) 在极端差值下极易 NaN 溢出。
    loss = -F.logsigmoid(beta * logits)
    
    # 4. (额外) 监控指标计算，脱离梯度流返回
    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    
    return loss.mean(), chosen_rewards.mean(), rejected_rewards.mean()
