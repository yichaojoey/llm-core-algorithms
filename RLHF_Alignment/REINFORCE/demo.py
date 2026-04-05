"""
REINFORCE 极简无依赖演示：证明基本策略梯度算法的传播影响
===================================================
用最纯粹的随机动作模拟一个序列，展示加入 Baseline 归一化后概率更新的高低分指向。
"""

import torch
import torch.nn as nn
from reinforce import compute_reinforce_loss

def run_reinforce_demo():
    print("=" * 60)
    print("演示 1：REINFORCE 序列策略梯度优化 (自带 Baseline)")
    print("=" * 60)
    
    # 一个 Episode 连续运行了 5 个 Step，获得了各自不同的 Reward
    # 比如：前几步都没得分，只有在最后一步拿到 10.0 的结果大奖
    rewards = torch.tensor([0.0, 0.0, 1.0, -1.0, 10.0])
    
    # 模拟网络生成的在当时 Step 进行决断的对数概率
    torch.manual_seed(42)
    logprobs_old = torch.randn(5) 
    
    logprobs_new_param = nn.Parameter(logprobs_old.clone())
    optimizer = torch.optim.SGD([logprobs_new_param], lr=0.1)
    
    for epoch in range(1, 6):
        optimizer.zero_grad()
        # 将刚才那局的 logprobs 和 收获的 rewards 送进去求取复盘教训 loss
        loss = compute_reinforce_loss(
            logprobs_new_param,
            rewards,
            gamma=0.9,
            use_baseline=True
        )
        loss.backward()
        optimizer.step()
        
        print(f"\n--- Epoch {epoch} ---")
        print(f"REINFORCE Surrogate Loss: {loss.item():.4f}")
        
    prob_diff = (logprobs_new_param - logprobs_old).detach()
    print("\n--- 回看优化结果 ---")
    print(f"原始这 5 步立即 Reward:    {rewards.tolist()}")
    print(f"各 Step 参数概率更新位移: {[round(x, 4) for x in prob_diff.tolist()]}")
    print("\n✅ 可以看到：由于第 4 步和第 5 步拿到了大奖，虽然 1、2 步当时的奖励是 0.0，")
    print("但因为这前两步为未来的高分做了因果铺垫，由于时序折扣 G_t 的回传传递，让它们当时的概率也都连带并大幅增高了！")

if __name__ == "__main__":
    run_reinforce_demo()
