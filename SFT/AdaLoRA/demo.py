"""
AdaLoRA 无依赖纯净演示：展示可剪枝架构对冗余维度的灭杀功能
=====================================================
"""

import torch
from adalora import AdaLoRALinear

def run_adalora_demo():
    print("=" * 60)
    print("演示 1：AdaLoRA 通过评分矩阵 E 找出混日子的 Rank 并执行动态死刑")
    print("=" * 60)
    
    in_dim = 16
    out_dim = 16
    r = 8
    
    model = AdaLoRALinear(in_features=in_dim, out_features=out_dim, r=r)
    model.mark_only_lora_as_trainable()
    
    # 模拟经过了几轮艰苦微调训练后：
    # 系统赋予了不同奇异值矩阵权重 (人工给定以作示例)：
    # 前 4 个非常有用且数值高昂！后 4 个毫无用处混天度日
    with torch.no_grad():
        model.lora_E.copy_(torch.tensor([5.0, 4.3, -3.1, 8.8, 0.005, 0.001, -0.002, 0.000]))
    
    print("\n🔹 经历训练后的 E 矩阵原始权重分布（绝对值）:")
    print([round(e.item(), 4) for e in model.lora_E.abs()])
    
    # 考核动作：定期巡回检查并执行淘汰！
    print("\n--- ⚡ 执行剪枝门槛 (Threshold > 0.01) ---")
    model.mask_prune_e(threshold=0.01)
    
    print("\n🔹 剪裁裁撤后 E 矩阵所剩存活参数分布:")
    print([round(e.item(), 4) for e in model.lora_E])
    
    # 获取成活通道数
    alive_channels = (model.lora_E != 0).sum().item()
    print(f"\n✅ 判定结果：原预设配给秩容量 R=8，实际发力通道仅仅有 {alive_channels}。")
    print(f"剩余 {r - alive_channels} 个坑位由于长线价值低于界限已被彻底抹杀节约了下来分配给下位模型！这完成了动态预算重新洗牌的任务。")

if __name__ == "__main__":
    run_adalora_demo()
