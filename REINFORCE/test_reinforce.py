import torch
from reinforce import compute_reinforce_loss

def test_reinforce_discounted_return():
    """验证 REINFORCE 折扣回报的时序逻辑计算是否完全符合预期"""
    logprobs = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    rewards = torch.tensor([1.0, 0.0, 1.0])
    gamma = 0.5
    
    # === 手算折扣 Return 验证: ===
    # 步骤 2 往后无内容: G_2 = 1.0
    # 步骤 1: G_1 = R_1 + gamma * G_2 = 0 + 0.5 * 1.0 = 0.5
    # 步骤 0: G_0 = R_0 + gamma * G_1 = 1.0 + 0.5 * 0.5 = 1.25
    # 最终 returns = [1.25, 0.5, 1.0]
    
    # mean = (1.25 + 0.5 + 1.0)/3 = 2.75 / 3 ≈ 0.9167
    
    # 测试时不开启 Baseline (即保留硬乘形式) 验证基本损失大小
    loss = compute_reinforce_loss(logprobs, rewards, gamma=gamma, use_baseline=False)
    
    # 理论 loss = -( -1.0*1.25 + -2.0*0.5 + -3.0*1.0 ) = -( -1.25 - 1.0 - 3.0 ) = -(-5.25) = 5.25
    assert abs(loss.item() - 5.25) < 1e-4

def test_reinforce_baseline():
    """验证有 Baseline 加持下梯度的稳定性流向 (防止全环境给予正数奖励导致一直乱加概率)"""
    logprobs = torch.tensor([-1.0, -1.0], requires_grad=True)
    
    # 极端的偏分环境，故意全给正分。对于单纯没有 Baseline 的网络它甚至会让烂步（比如1.0分那一步）的概率因为得分为正而跟着变大
    rewards = torch.tensor([5.0, 1.0]) 
    
    loss = compute_reinforce_loss(logprobs, rewards, gamma=1.0, use_baseline=True)
    loss.backward()
    
    # G_1 = 1, G_0 = 5 + 1 = 6
    # 返回 Baseline 去除均值并除以方差后，G_0 相较于群体均值必定大于 0 (正向加强反馈)
    # 而由于 G_1 这步的发挥拖了大家后腿，其归一化后分数必定小于 0 (遭到负面压制)
    
    # 注意！在 loss.backward 中梯度由于外层加了负号：
    # `grad < 0` 意味梯度下降操作执行 "x_new = x - lr * grad" 时该对数概率被加大了。
    assert logprobs.grad[0].item() < 0 # 加大优异步概率
    assert logprobs.grad[1].item() > 0 # 减小拖后腿步概率
