import torch
from grpo import compute_grpo_loss

def test_grpo_loss_formula():
    """验证 GRPO 的组内归一化及裁剪是否计算正确"""
    logprobs_new = torch.tensor([[-1.0, -1.0], [-1.5, -1.5]])  # G=2, Seq=2
    logprobs_old = torch.tensor([[-1.0, -1.0], [-1.5, -1.5]])
    rewards = torch.tensor([1.0, 0.0]) # G=2
    
    # === 手动演算预期优势 ===
    # rewards: [1.0, 0.0] -> mean: 0.5
    # std(unbiased=False): 方差 = ((1-0.5)^2 + (0-0.5)^2)/2 = 0.25 -> std = 0.5
    # advs: [ (1.0-0.5)/0.5, (0.0-0.5)/0.5 ] = [1.0, -1.0]
    
    # Ratio 全部等于 1.0
    # [1.0, 1.0] * adv([1], [-1])
    # [ [1.0, 1.0], [-1.0, -1.0] ]
    # 因为都未超 eps_clip, surrogate 全取未 clip 前的.
    # 负的最大化： mean([1, 1, -1, -1]) = 0 但是因为取负所以是 0
    
    loss = compute_grpo_loss(logprobs_new, logprobs_old, rewards)
    
    assert abs(loss.item()) < 1e-5

def test_grpo_clipping():
    """验证 GRPO Loss 是不是由于大幅更新被限制住了"""
    logprobs_new = torch.tensor([[-0.5], [-1.5]]) # ratio 非常大 => P_new >> P_old (假设 old 全是 -1.5)
    logprobs_old = torch.tensor([[-1.5], [-1.5]])
    # Ratios ≈ exp(1.0) ≈ 2.71，对于第二项是 exp(0) = 1
    rewards = torch.tensor([1.0, 0.0]) 
    
    # Advs = [1.0, -1.0]
    # Surrogate:
    # 组 1: Ratio 2.71, Adv 1.0 -> Clip: 1+0.2=1.2.  min(2.71, 1.2) = 1.2
    # 组 2: Ratio 1.0, Adv -1.0 -> Clip 不被限: 1.0 * -1.0 = -1.0
    # Total Surr = (1.2 + (-1.0)) / 2  = 0.1
    # Loss = -Total Surr = -0.1
    
    loss = compute_grpo_loss(logprobs_new, logprobs_old, rewards, eps_clip=0.2)
    # 取一个差不多的浮点精度
    assert abs(loss.item() - (-0.1)) < 1e-5
