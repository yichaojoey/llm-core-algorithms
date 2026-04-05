"""
MHA (Multi-Head Attention) 核心面试级模拟实现
==========================================
【理论揭秘】：来自经典原论文 "Attention is All You Need"。
核心逻辑是将输入的 Embedding 获取的参数矩阵分解为多个互相独立的头（Heads），
让每个头从不同维度去独立捕捉特征上下文的注意力交互，最后再把所有的长短脑回路重组并凑起来。
它有一个最核心的硬性条件要求：Query, Key, Value 的 Head 数量永远是 [ 1 : 1 : 1 ] 严格对齐的！
这也是导致它在目前大模型长文本推理阶段遭遇惨痛的“KV Cache”显存溢出噩梦的最大元凶。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # 面试必考防坑：embed_dim 必须要被 num_heads 完美整除，否则无法切脑袋
        assert embed_dim % num_heads == 0, "Embedding 维度必须被 Heads 数量整除！"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 定义核心 Wq, Wk, Wv 线性投射矩阵
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 最后的输出重组矩阵
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x: [Batch, SeqLen, EmbedDim]
        """
        B, T, C = x.size()
        
        # 1. 经典三线投射 (形变前还是长在一起的 EmbedDim)
        q = self.q_proj(x)  # [B, T, C]
        k = self.k_proj(x)  # [B, T, C]
        v = self.v_proj(x)  # [B, T, C]
        
        # 2. 劈开脑袋 (Reshape)
        # 维度变形极其讲究：把 C 打碎为 (num_heads, head_dim)
        # 并用 transpose(1, 2) 把 seq_len 和 num_heads 对调，使得相同的 head 数据在内存里是连跳靠拢的
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, HeadDim]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, HeadDim]
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, HeadDim]
        
        # 3. 计算注意力分数 Attention Scores
        # 核心公式: Softmax( Q * K^T / sqrt(d_k) ) * V
        # Q [B, H, T, HeadDim] @ K.transpose [B, H, HeadDim, T]  ->  Scores [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # （可选）如果有 Causal Cask 用防偷看未来掩码挡住右上角
        if mask is not None:
            # 填入负无穷使得 softmax 归 0
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        
        # 4. 把权重视角汇聚到 Value 实体上 [B, H, T, T] @ [B, H, T, HeadDim] -> [B, H, T, HeadDim]
        out = torch.matmul(attn_weights, v)
        
        # 5. 拼凑重整！把碎脑袋倒退着变回最初的融合样态
        # contiguous() 是极高频考点，transpose 之后底层内存不连续了，后续强行 view 打平会报错！
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # 6. 利用 Output 投影层打乱融合特征
        return self.o_proj(out)
