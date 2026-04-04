"""
QLoRA 无依赖纯净演示：展示主结构压缩下的伪量化与双轨精度架构
=====================================================
"""

import torch
from qlora import QLoRALinear

def run_qlora_demo():
    print("=" * 60)
    print("演示 1：QLoRA 的 4-Bit 极致压缩空间占用与全精度计算")
    print("=" * 60)
    
    in_dim = 4096
    out_dim = 4096
    r = 8
    block_size = 64
    
    model = QLoRALinear(in_features=in_dim, out_features=out_dim, r=r, block_size=block_size)
    model.mark_only_lora_as_trainable()
    
    # 用元素个数乘以该数据类型占用的字节数
    # 大部分传统模型的 32 位浮点 (fp32) 参数一个就占 4 Bytes
    full_precision_size_bytes = in_dim * out_dim * 4 
    
    # 我们 QLoRA 的底座采用的是极其压缩微小的数据模型格式（当前用 int8 即 1 byte 演示 4bit(0.5b) ）
    # 即便如此，它的绝对存储体积已如断崖般暴降：
    quantized_size_bytes = model.quantized_weight.element_size() * model.quantized_weight.nelement()
    absmax_blocks_bytes = model.absmax_blocks.element_size() * model.absmax_blocks.nelement()
    lora_bytes = (model.lora_A.nelement() + model.lora_B.nelement()) * 4
    
    total_qlora_bytes = quantized_size_bytes + absmax_blocks_bytes + lora_bytes
    
    print("\n🔹 如果它是一层普通的全参数冻结的全卡 Linear 层体积：")
    print(f"  --> {full_precision_size_bytes:,.0f} Bytes  (100.0%)")
    
    print("\n🔹 QLoRA 改装层实际使用的物理硬盘/显存盘查：")
    print(f"  [被量化压缩锁死的万年冰川阵列区]: \t{quantized_size_bytes:,.0f} Bytes （这里占尽大头，但精度降维）")
    print(f"  [量化补偿使用的各个零碎辅助分块刻度]: \t{absmax_blocks_bytes:,.0f} Bytes")
    print(f"  [鲜活全精度的纯旁置 A/B 流水线参数]: \t{lora_bytes:,.0f} Bytes")
    print(f"  --> 总共加起来只有：{total_qlora_bytes:,.0f} Bytes  ({total_qlora_bytes/full_precision_size_bytes*100:.2f}%!!!)\n")
    
    # 模拟真实微调环节：梯度的绝不互通
    x = torch.randn(2, 5, in_dim) 
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    # 量化区本就是粗糙且缺乏记录信息的状态体系，不允许任何有意义梯度流过去破坏：
    print("--- 反向梯导向最终审核 ---")
    if model.quantized_weight.grad is None:
        print("✅ 量化坚冰主网络阵列：没有被任何梯度波及与污染。纯死锁绝缘成功！")
    
    if model.lora_A.grad is not None and model.lora_B.grad is not None:
        print("✅ LoRA 吸盘矩阵阵列：获得了宝贵的全精度（Float32级别）纯净指路梯度！微调路线通畅！")
        
if __name__ == "__main__":
    run_qlora_demo()
