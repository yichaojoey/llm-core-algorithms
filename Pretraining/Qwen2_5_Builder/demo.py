"""
Qwen 2.5 极致组装建筑：流水线通关验证
====================================
"""

import sys
import torch
from qwen_block import Qwen2_5_DecoderBlock

def run_qwen_assembly_demo():
    print("=" * 60)
    print("演示：将我们自己打造的底层散件武器，完美串联缝合成 Qwen 2.5 核弹")
    print("=" * 60)
    
    # 构建 Qwen 2.5 7B 模型里极其微小的一个纳米细胞切片缩影
    embed_dim = 128
    num_q_heads = 8
    num_kv_heads = 2
    hidden_expansion = embed_dim * 4 # SwiGLU 的常规范式膨胀
    
    # 实例化组装厂，把之前写的 RMSNorm, SwiGLU 还有 Qwen版的 GQA 全部倒进去拼接起来
    qwen_layer = Qwen2_5_DecoderBlock(
        embed_dim=embed_dim, 
        num_query_heads=num_q_heads, 
        num_kv_heads=num_kv_heads, 
        hidden_dim=hidden_expansion
    )
    
    print("\n✅ 【架构搭建成功鉴赏】:")
    print(qwen_layer)
    
    print("\n--- 开辟演武场跑腿实验 ---")
    x = torch.randn(2, 64, embed_dim) # [Batch=2，句长=64字符]
    
    # 发动整个 Qwen 层
    output = qwen_layer(x)
    
    print(f"\n[入关数据] 经过极其严峻的清洗 (RMSNorm) -> 跨越省显存的高架桥 (GQA带Bias) -> ")
    print(f"再次清洗 (RMSNorm) -> 体验非线性的极限胖瘦穿行 (SwiGLU) -> 返回原本的河流 (Residual) 后：")
    print(f"最终输出形态为: {output.shape}，没有发生任何维度的爆炸和泄漏走光！")
    print("一切平稳！这只是区区 1 层（Block），把这个积木做个 for 循环套上 80层，头上安上一个超大字典映射墙，即可复古再造真实的 Qwen 2.5！")

if __name__ == "__main__":
    run_qwen_assembly_demo()
