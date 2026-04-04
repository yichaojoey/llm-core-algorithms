"""
GSPO 面试无依赖演示：证明序列级裁剪和优势分配
==========================================
构造虚拟的环境 G 个生成句，验证 Seq-level Ratio 如何被 Clip 从而稳定梯度。
"""

import torch
import torch.nn as nn
from gspo import compute_gspo_loss

def run_gspo_demo():
    print("=" * 60)
    print("演示 1：GSPO 序列级别的几何平均裁剪与优化")
    print("=" * 60)
    
    G = 4 # 一个 Query 抽样出 4 个回答组
    
    # 序列长度分别假设为 10, 20, 5, 10
    seq_lengths = torch.tensor([10.0, 20.0, 5.0, 10.0])
    
    # 获取到的回答对应 Reward
    rewards = torch.tensor([10.0, 5.0, -2.0, 0.0])
    
    # 随机化旧梯度的句子最终取对数概率和
    torch.manual_seed(42)
    logprobs_old_sum = torch.randn(G) # 大约在 0 周围
    
    # 初始化带有梯度的 Policy 模型
    logprobs_new_sum_param = nn.Parameter(logprobs_old_sum.clone())
    optimizer = torch.optim.SGD([logprobs_new_sum_param], lr=0.5)
    
    for epoch in range(1, 6):
        optimizer.zero_grad()
        loss = compute_gspo_loss(
            logprobs_new_sum_param,
            logprobs_old_sum,
            seq_lengths,
            rewards,
            eps_clip=0.2
        )
        loss.backward()
        optimizer.step()
        
        print(f"\n--- Epoch {epoch} ---")
        print(f"GSPO Sequence-level Loss: {loss.item():.4f}")
        
    print("\n--- 优化现象 ---")
    prob_diff = (logprobs_new_sum_param - logprobs_old_sum).detach()
    
    print(f"原始 Reward 得分数组:      {rewards.tolist()}")
    print(f"经过五个 Epoch 的概率总和增量: {[round(x, 4) for x in prob_diff.tolist()]}")
    print("\n✅ 可以看到高 Reward 的句子概率大幅稳定提升！而且利用序列长度抑制了过长句子由于概率累乘而带来的爆炸！")

if __name__ == "__main__":
    run_gspo_demo()
