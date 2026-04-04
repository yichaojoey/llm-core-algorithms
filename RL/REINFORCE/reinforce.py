"""
REINFORCE (Vanilla Policy Gradient) 核心实现
==========================================
所有去 Critic 化且基于策略梯度强化的算法的“祖师爷”。
核心逻辑：对一条轨迹从后往前算折扣回报 (Discounted Returns G_t)
然后定下目标： Loss = - Sum(logprob * G_t)
"""

import torch

def compute_reinforce_loss(
    logprobs: torch.Tensor, # [T] 模型在一个 Episode 里产生的每一步动作的对数概率
    rewards: torch.Tensor,  # [T] 每一步对应的即时反馈奖励
    gamma: float = 0.99,
    use_baseline: bool = True
):
    """
    计算基于单个 Episode/Trajectory 的 REINFORCE 损失。
    """
    returns = []
    G_t = 0.0
    
    # 1. 逆序计算折扣回报 (Discounted Returns)
    # 因为 t 时刻的动作只能影响 t 时刻及之后的奖励，不能影响过去的奖励，所以从最后一刻倒排
    for r in reversed(rewards):
        G_t = r + gamma * G_t
        returns.insert(0, G_t)
        
    returns = torch.tensor(returns, dtype=torch.float32)
    
    # 2. 引入 Baseline 降低方差 (面试加分点：这是及其重要的一步，否则梯度很容易崩溃)
    # 在最纯粹的 REINFORCE 中，Baseline 常选用这条轨迹自身的均值来作为参考基线
    # 思考：如果不减 Baseline，而是由于某些环境所有机制永远给正数，那所有策略的概率将陷入盲目单向变大！
    if use_baseline and len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        
    # 3. 计算 Policy Gradient Loss
    # J = E [ log(pi) * G_t ] -> 我们要最大化目标，所以在前面加负号交给 optimizer 求最小的下坡方向
    loss = -(logprobs * returns).sum()
    
    return loss
