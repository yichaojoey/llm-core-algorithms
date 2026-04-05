"""
Qwen 2.5 核心 Transformer Block (解剖级复刻)
=====================================
【面试绝对防坑指南】：
千万不要一被问 “请手写一个目前主流大模型的 Decoder Block” 就下意识去纯抄 LLaMA。
由于 Qwen 2.5 惊艳的评测霸榜表现，现在考 Qwen 架构细微怪癖的面试官越来越多。

【与 LLaMA 架构的核心考点差异】：
1. **QKV 偏置 (Bias)**: LLaMA 所有线性层全线 `bias=False`。但 Qwen 极其执着且聪明地在 Q, K, V 的 Linear 查表投影层中保留了 `bias=True`！阿里研究表明这能在极端长文本漫游中极好地保留位置偏移特性。
2. **Untied Embeddings**: Qwen 顶层推流时，输入词表和最终算概率的分类词表是独立无血缘关系的两套极其巨大的参数（词表高达 15.2 万维），这意味着极其耗费显存但表达极其精准。
3. **经典 Pre-Norm 串联体系**: 彻底贯彻先用 RMSNorm 清洗镇压，再进入大计算阵列，最后加上源流特征的完美大循环残差逻辑。
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 强行连通并借用我们在这个库里亲手打磨出来的主力军火散件
sys.path.append('../Modern_Layers')
from rmsnorm import RMSNorm
from swiglu import SwiGLU

class QwenGroupedQueryAttention(nn.Module):
    """Qwen 专属架构的 GQA。跟我们普通的 GQA 最大的细微差别在于它那该死的 Bias 设定。"""
    def __init__(self, embed_dim: int, num_query_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_q_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.num_q_per_kv = num_query_heads // num_kv_heads 
        
        # ========================================================
        # ⚠️ 【面试高亮考点/陷阱】：Qwen 架构在这里必须要加 bias=True ！
        # 如果你这里习惯性写了 LLaMA 式的 False，面试官直接扣大分。
        # ========================================================
        self.q_proj = nn.Linear(embed_dim, self.num_q_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=True)
        
        # 往往输出映射层 O_proj 也可以回归为 False，不影响大局
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """GQA 经典的极省显存克隆大放送"""
        if n_rep == 1:
            return x
        B, num_kv_heads, T, HeadDim = x.shape
        x = x.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
        return x.reshape(B, num_kv_heads * n_rep, T, HeadDim)

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        
        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 【此处通常本该注入 RoPE (Rotary Position Embedding)】
        # Qwen 2.5 为了顶住 1M 长文使用的是极其复杂的 Dual Chunk Attention 改良魔改版 RoPE
        # 为了不喧宾夺主导致代码长到面试官不想看，我们用伪注释替代 RoPE 相乘
        # q, k = apply_rotary_emb(q, k)
        
        k = self._repeat_kv(k, self.num_q_per_kv)  
        v = self._repeat_kv(v, self.num_q_per_kv)  
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o_proj(out)

class Qwen2_5_DecoderBlock(nn.Module):
    """
    终极合体：大模型架构积木的最终安放大龙骨！
    这也就是在大厂内部，被称为 【一个 Block】的最小可复用原子结构。
    """
    def __init__(self, embed_dim: int, num_query_heads: int, num_kv_heads: int, hidden_dim: int):
        super().__init__()
        # 双重防御机制 (Pre-Norm 的关键)：
        # 1. 进 Attention 前要全身清洗
        self.input_layernorm = RMSNorm(embed_dim)
        self.self_attn = QwenGroupedQueryAttention(embed_dim, num_query_heads, num_kv_heads)
        
        # 2. 进 FFN 变胖之前也要全身清洗
        self.post_attention_layernorm = RMSNorm(embed_dim)
        self.mlp = SwiGLU(in_features=embed_dim, hidden_features=hidden_dim, out_features=embed_dim)

    def forward(self, x: torch.Tensor):
        """
        ========================================================
        【面试高亮考点】：无损串联流水线 (Pre-Norm Residual Pipeline)
        绝对不可写成: x = norm(attn(x) + x)   <- 这是早就被淘汰的 Post-Norm (Transformer 原版) ！极难收敛！
        必须要写成: x = x + attn(norm(x))   <- 这是当代天花板 Pre-Norm ！！
        ========================================================
        """
        # 第一座大门：清洗 -> 注意力打分 -> 加回原始自身体内积蓄 (残差连接)
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x
        
        # 第二座大门：清洗 -> 经过极其扭曲极速发福变瘦的非线性多岔路 MLP -> 加回体内
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x
