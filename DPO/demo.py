"""
DPO (Direct Preference Optimization) 无依赖面试验证演示
===================================================
演示梯度回传及 DPO 对偏好收敛的影响过程。
"""

import torch
import torch.nn as nn
from dpo import compute_dpo_loss

def run_dpo_demo():
    print("=" * 60)
    print("演示 1：DPO (Direct Preference Optimization) 前向优化模拟")
    print("=" * 60)
    print("说明: 这里验证梯度是如何让模型增大 Chosen 生成概率，压制 Rejected 概率的。")
    print("-" * 60)
    
    beta = 0.5
    
    # ==========================
    # 数据模拟 (分别代表两个样本产生的值)
    # 实际 LLM 实现中是通过 sum(log_softmax(logits) * masks) 来获取的
    # ==========================
    
    # 假设 Reference 模型对 (chosen, rejected) 序列没有强烈偏向
    ref_chosen_logps = torch.tensor([-5.1, -6.2])
    ref_rejected_logps = torch.tensor([-5.0, -6.0])
    print(f"🔹 Reference (静态) 对 Chosen   的 LogProb: \t{ref_chosen_logps.tolist()}")
    print(f"🔹 Reference (静态) 对 Rejected 的 LogProb: \t{ref_rejected_logps.tolist()}")
    
    # 将要接受训练的 Policy 设置为 Parameter，捕获它们的梯度演化
    pol_chosen_logps = nn.Parameter(torch.tensor([-4.8, -6.1]))
    pol_rejected_logps = nn.Parameter(torch.tensor([-4.9, -5.8]))
    
    optimizer = torch.optim.SGD([pol_chosen_logps, pol_rejected_logps], lr=0.8)
    
    # ==========================
    # 模拟多轮梯度裁剪迭代
    # ==========================
    for epoch in range(1, 6):
        loss, r_chosen, r_rej = compute_dpo_loss(
            pol_chosen_logps, pol_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            beta=beta
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"\n--- Epoch {epoch} ---")
        print(f"Loss: {loss.item():.4f}")
        print(f"隐式 Reward 优势 (Chosen - Rej): {(r_chosen - r_rej).item():.4f}")
        print(f"📈 更新后 Policy 对 Chosen   的 LogProb: \t{pol_chosen_logps.detach().numpy()}")
        print(f"📉 更新后 Policy 对 Rejected 的 LogProb: \t{pol_rejected_logps.detach().numpy()}")

if __name__ == "__main__":
    run_dpo_demo()
