"""
Attention 机制家族直观比试演示
==========================================
"""

import torch
from mha import MultiHeadAttention
from gqa import GroupedQueryAttention

def run_attention_demo():
    print("=" * 60)
    print("演示：MHA 全通连接 vs GQA 共享省显存组别 的物理状态对比")
    print("=" * 60)
    
    # 构建超小型场景
    embed_dim = 128
    num_query_heads = 8
    
    # 场景 1：经典守旧派 (MHA) - 必须按 1:1 给 KV 配发全尺寸 Head
    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_query_heads)
    
    # 场景 2：现代革新派 (GQA) - 依然想要 8 个 Q 头探听八方，但只给发放抠抠搜搜的 2 个 KV 头用于回忆
    gqa = GroupedQueryAttention(embed_dim=embed_dim, num_query_heads=num_query_heads, num_kv_heads=2)
    
    print("\n🔹 【架构底座参数量对比追踪】:")
    
    # 计算 K 矩阵在 MHA 下的总容量
    mha_k_size = mha.k_proj.weight.nelement() + mha.k_proj.bias.nelement()
    # 计算 K 矩阵在 GQA 下的总容量
    gqa_k_size = gqa.k_proj.weight.nelement() + gqa.k_proj.bias.nelement()
    
    print(f"MHA 维护的庞大笨重 Key 张量池: \t {mha_k_size} 个参数单元 （每 1 个 Q 配备 1 专属 K）")
    print(f"GQA 维护的轻量共享 Key 张量池: \t {gqa_k_size} 个参数单元 （暴降 {mha_k_size/gqa_k_size} 倍！4个Q蹭1个群K）")
    
    print("\n--- 极限推导试运行前向通路一致性验证 ---")
    x = torch.randn(2, 10, embed_dim) # [Batch=2, SeqLen=10]
    
    mha_out = mha(x)
    gqa_out = gqa(x)
    
    print(f"MHA 最终挤出的特征维度: {mha_out.shape} -> 可以看出，大家都能完好输出正常的原长形体。")
    print(f"GQA 最终挤出的特征维度: {gqa_out.shape} -> 但 GQA 在整个路途上的门槛费极其低廉。这也是为何当下的 Llama3 乃至所有的 70B 模型疯狂拥抱 GQA 的根源。")
    

if __name__ == "__main__":
    run_attention_demo()
