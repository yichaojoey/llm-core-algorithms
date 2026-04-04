"""
GRPO 面试无依赖演示：证明无 Critic 网络收敛
========================================
通过构造一个 G 个抽样组的虚拟环境，展示 GRPO 中内部奖励归一化特性。
"""

import torch
import torch.nn as nn
from grpo import compute_grpo_loss

def run_grpo_demo():
    print("=" * 60)
    print("演示 1：GRPO 相对目标无 Critic 优化逻辑")
    print("=" * 60)
    
    G = 3 # 一个 Query 抽样出 3 个回答组
    seq_len = 4
    
    # ===============
    # 一道数学题的三个不同回答得到了以下打分（有的全对，有的部分对）
    # ===============
    rewards = torch.tensor([1.0, 0.5, 0.0])
    print(f"得分数组: {rewards.tolist()}")
    
    # 计算均值和方差，向求得内部 Advantage 的形态靠拢
    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False)
    advs = (rewards - mean_r) / (std_r + 1e-8)
    print(f"经 GRPO 去 Critic 化出来的相对 Advantage 数组: {advs.tolist()}")
    # advs => [1.22, 0.0, -1.22]
    
    # ===============
    # 模拟网络层产生的 Token-level log_probs
    # ===============
    torch.manual_seed(42)
    # 随机化旧权重
    logprobs_old = torch.randn(G, seq_len)
    
    # 开始可训练网络
    logprobs_new_param = nn.Parameter(logprobs_old.clone())
    optimizer = torch.optim.SGD([logprobs_new_param], lr=0.1)
    
    for epoch in range(1, 4):
        optimizer.zero_grad()
        loss = compute_grpo_loss(
            logprobs_new_param,
            logprobs_old,
            rewards,
            eps_clip=0.2
        )
        loss.backward()
        optimizer.step()
        
        print(f"\n--- Epoch {epoch} ---")
        print(f"GRPO Surrogate Loss: {loss.item():.4f}")
        
    # 我们看一下各个组别里权重的增减趋势
    # 有高 Reward (1.0) 的那组，对应的 logprob 的前向趋势是变大的：由于 adv > 0
    # Reward (0.0) 的那组，对应的 logprob 的趋势是被压制变小的
    prob_diff = (logprobs_new_param - logprobs_old).detach()
    
    print("\n--- 优化现象 ---")
    print("第一组句子(得分最高) Token对数概率位移:", prob_diff[0].tolist())
    print("第三组句子(得分最低) Token对数概率位移:", prob_diff[2].tolist())
    print("\n这便是 PPO-Clip 带着组内自参考 Baseline 直接进行的 Token 强化！")

if __name__ == "__main__":
    run_grpo_demo()
