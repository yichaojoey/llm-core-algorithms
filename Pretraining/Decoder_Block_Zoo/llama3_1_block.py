"""
LLaMA-3.1 Decoder Block 架构
===================================
【理论揭秘】：
Meta 团队的执念：“如非必要，勿增实体”。
相比于 Qwen 系列花里胡哨地在 QKV 投射里加 Bias 偏置、混合双 chunk RoPE 等等。
LLaMA-3.1 展示了极其极致的【极简与纯粹】。
它的核心特征只有两个考点：
1. **全员剥离 Bias (Zero Bias)**：你在这个 Block 里找不到任何一个 `bias=True`。无论是 Attention 还是 MLP。
   因为不仅 Bias 占用了多余参数，更重要的是 Bias 会破坏向量空间的“缩放平移等变性 (Scale Invariance)”。去掉偏置能让模型泛化和对抗灾难遗忘的能力极大加强。
2. **教科书级别的标准件**：极其标准的 RMSNorm -> GQA -> RMSNorm -> SwiGLU。
"""

import torch
import torch.nn as nn

class Llama3_1_Attention(nn.Module):
    def __init__(self, d_model=4096, n_heads=32, n_kv_heads=8):
        super().__init__()
        # ⚠️ 面试重点 1：LLaMA 系列是极其纯粹的 bias=False 原教旨主义者
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, (d_model // n_heads) * n_kv_heads, bias=False)
        self.v_proj = nn.Linear(d_model, (d_model // n_heads) * n_kv_heads, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        return x  # 简化：正常的 GQA 计算与拼装

class Llama3_1_SwiGLU(nn.Module):
    def __init__(self, d_model=4096, intermediate_size=14336):
        super().__init__()
        # ⚠️ 面试重点 2：MLP 层同样极其干脆，依然没有任何 Bias
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.act_fn = nn.SiLU() # Swish 激活函数

    def forward(self, x):
        # 原汁原味的 SwiGLU 操作
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Llama3_1_DecoderBlock(nn.Module):
    def __init__(self, d_model=4096):
        super().__init__()
        # LLaMA 开创并发扬光大的 RMSNorm，绝不用 LayerNorm
        self.input_layernorm = nn.RMSNorm(d_model)     # 此处直接调原生，或自行实现
        self.self_attn = Llama3_1_Attention(d_model)
        
        self.post_attention_layernorm = nn.RMSNorm(d_model)
        self.mlp = Llama3_1_SwiGLU(d_model)

    def forward(self, hidden_states):
        # 极简 Pre-Norm 结构
        # 1. 扔进 Norm -> 过 Attention -> 残差加上去
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        
        # 2. 扔进 Norm -> 过 MLP -> 残差加上去
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

if __name__ == "__main__":
    print("=" * 60)
    print("LLaMA-3.1: 纯粹极简的大道至简")
    print("=" * 60)
    print("与 Qwen2.5 相比，这里没有花里胡哨的偏置，没有特调的结构修改。")
    print("Meta 就证明了一件事：数据堆得够大，架构越干净越简单，模型越能一剑破万法！")
