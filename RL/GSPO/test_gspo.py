import torch
from gspo import compute_gspo_loss

def test_gspo_sequence_ratio_formula():
    """验证 GSPO 的 Sequence-level Ratio 的裁剪限制在单元测试场景能手动触发"""
    
    # 我们用手工设定的数据使得它很好计算
    logprobs_old_sum = torch.tensor([-5.0, -10.0])
    logprobs_new_sum = torch.tensor([-3.0, -10.0]) 
    seq_lengths = torch.tensor([2.0, 4.0])
    rewards = torch.tensor([1.0, -1.0])
    
    # === 手算预期值 ===
    # 优势归一化: mean=0, std=1 (如果只有1, -1, 其 unbiased=False 的方差为 ((1-0)^2 + (-1-0)^2)/2 = 1.0 => adv = (R-Mean) / 1 = R)
    # advs = [1.0, -1.0]
    
    # 序列1: old=-5, new=-3, L=2
    # ratio_1 = exp( (-3 - -5)/2 ) = exp(2/2) = exp(1) ≈ 2.718
    # 本次裁剪 clip 设置为 0.2，surrogate_1 = min(2.718 * 1.0, clip(2.718, 0.8, 1.2)*1.0) = 1.2
    
    # 序列2: old=-10, new=-10, L=4
    # ratio_2 = exp(0/4) = exp(0) = 1.0
    # surrogate_2 = min(1.0 * -1.0, clip(1, 0.8, 1.2) * -1.0) = -1.0
    
    # mean surrogate = (1.2 + -1.0) / 2 = 0.1
    # loss = - mean surrogate = -0.1
    
    loss = compute_gspo_loss(logprobs_new_sum, logprobs_old_sum, seq_lengths, rewards, eps_clip=0.2)
    
    assert abs(loss.item() - (-0.1)) < 1e-4

def test_gspo_backward_flow():
    """验证 GSPO 针对前向张量的梯度反馈通道畅通"""
    logprobs_new_sum = torch.tensor([-5.0, -10.0], requires_grad=True)
    loss = compute_gspo_loss(
        logprobs_new_sum, 
        logprobs_new_sum.clone().detach(), 
        torch.tensor([1., 1.]), 
        torch.tensor([1.0, -1.0])
    )
    loss.backward()
    
    assert logprobs_new_sum.grad is not None
    # 高分样本（正梯度回传，因此是负导数），低分是正导数压制
    assert logprobs_new_sum.grad[0].item() < 0
    assert logprobs_new_sum.grad[1].item() > 0
