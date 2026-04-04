"""
LoRA 无依赖纯净演示：展示可训练参数的急剧降低与初始化行为
===================================================
"""

import torch
from lora import LoRALinear

def run_lora_demo():
    print("=" * 60)
    print("演示 1：LoRA 张量形变与零初始化层对原分布的绝对保护机制")
    print("=" * 60)
    
    in_dim = 4096
    out_dim = 4096
    r = 8
    
    model = LoRALinear(in_features=in_dim, out_features=out_dim, r=r)
    # 模拟真实微调环节：将底层主参数强制锁结
    model.mark_only_lora_as_trainable()
    
    # 1. 证实其傲人的宏大规模缩减（PEFT：用最少显存调最大的模型）
    base_params = in_dim * out_dim
    lora_params = in_dim * r + r * out_dim
    print(f"\n🔹 原发模型该层预训练本体基准参数量: \t{base_params:,} (100% 当作冻结对照)")
    print(f"🔹 LoRA 外侧旁置参数量(用来放坡收梯度): \t{lora_params:,}  ({lora_params/base_params*100:.3f}% 极大减负缩减比!!)")
    
    # 2. 验证 Zero Initialization (零初始化) 的伟大之处
    x = torch.randn(1, 10, in_dim)
    
    # 强制将模型进入评估模式，关掉 dropout 概率屏蔽带来的随机跳跃算术干扰从而验证硬等价
    model.eval()
    
    out_lora = model(x)
    
    # 我们如果直接纯手工拔掉旁支，只拿那块大铁片子原始 Weight 去相乘，看结果是不是和刚建出来的 LoRA 一模一样
    out_base = torch.matmul(x, model.weight.T) + model.bias
    
    diff = (out_lora - out_base).abs().max().item()
    print(f"\n🔹 LoRA 总输出端与 不挂载任何 LoRA 前 的纯净原始线性差值: \t{diff:.6f}")
    if diff == 0.0:
        print("   ✅ 验证完美通过：由于 B 型矩阵严格强制全零诞生，初始微调模型介入拥有极其无损平滑的前驱一致性，彻底没伤到千亿根基大脑的脑回纹知识！")
        
    # 3. 验证训练下梯度的导向分封收容
    model.train()
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    print("\n--- 反向传播参数变通检查 ---")
    print(f"原模型权重大山 (W)  遭遇无差别打击并获取了梯度？   {model.weight.grad is not None} (预示完全不受累)")
    print(f"LoRA A 降维矩阵接管流控获取了被惩罚的新梯导向？   {model.lora_A.grad is not None}")
    print(f"LoRA B 抬升矩阵接管流控获取了被惩罚的新梯导向？   {model.lora_B.grad is not None}")
    print("🚀 至此：所有对微调语料中犯错或奖赏反馈的经验流逝梯度，完全被小巧轻灵的两块子矩阵 A 和 B 给吸纳住了！这就是它为显存优化背书的根源！\n")

if __name__ == "__main__":
    run_lora_demo()
