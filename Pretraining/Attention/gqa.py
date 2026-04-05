"""
GQA (Grouped-Query Attention) 核心面试级模拟实现
============================================
【理论揭秘】：这是当代几乎所有极其能打的开源模型（LLaMA-2/3, Mistral, Qwen）的绝对底层标配！
主要为了解决当年 MHA （1 对 1）在长文本阶段做文字接龙时，“KV Cache” 撑爆单张显卡内存的千古绝唱。
GQA 采取了分组极简机制：把多个 Query Head 编成一个组，这一个组里的兄弟们共用（共享）同一个唯一的 Key 和 Value Head！
它比当年老旧的 MHA (1:1配比) 暴省出天际数量的显存，同时也比极其粗暴的 MQA (所有人抠搜共用1套KV头) 逻辑效果要更饱满更好！

【面试绝杀代码点】：在 Python Tensor 计算流里，你是如何优美高效地把那数量稀疏可怜的 KV 头，给复制（Broadcast）扩写对齐去跟人多势众的 Query 们强行做内积的？
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim: int, num_query_heads: int, num_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        # 【面试防坑检查点】：Q 的数量必须能够毫无残留地整除 KV 的数量形成正规组别
        assert embed_dim % num_query_heads == 0, "Embedding 维度必须被 Heads 数量整除！"
        assert num_query_heads % num_kv_heads == 0, "Query_heads 必须完整整除 KV_Heads 形成完美组团！"
        
        self.embed_dim = embed_dim
        self.num_q_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        
        # 一组里面分摊多少个兄弟共用一套房
        self.num_q_per_kv = num_query_heads // num_kv_heads 
        
        # 经典的投影层（关键不同点在这里：KV 的投射厚度大幅度降低了，这就是省参数省显存的关键！）
        self.q_proj = nn.Linear(embed_dim, self.num_q_heads * self.head_dim)
        # 注意看！下面两兄弟的规模只有可怜的 KV_heads，不再是大部队了
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim)
        
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        
    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        【极其高频考点/最核心精髓机制】：手写 KV-Repeat (复制对齐广播)
        既然咱们是多个 Q 盯上同一个 K 等着计算得分，K 就必须分身！
        它进来的尺寸是 [Batch, num_kv_heads, SeqLen, HeadDim]
        我们要巧妙借助 unsqueeze + expand 原地虚拟翻倍拉扯为 -> [Batch, q_heads, SeqLen, HeadDim]
        """
        # 如果是 1 对 1 则意味着等同于 MHA，直接放行
        if n_rep == 1:
            return x
        
        B, num_kv_heads, T, HeadDim = x.shape
        # 这就是一个极度天才的变体做法：在头维度后插入一道隐秘的缝隙然后拓展，紧接着直接拍扁缝隙达到复制效果！
        x = x.unsqueeze(2)                    # -> [B, num_kv_heads, 1,     T, HeadDim]
        x = x.expand(-1, -1, n_rep, -1, -1)   # -> [B, num_kv_heads, n_rep, T, HeadDim]
        
        # 强行重新用连续内存布局压缩维度，这样原本单薄的 KV 头名义上正式扩军迎战 Q_heads 了
        return x.reshape(B, num_kv_heads * n_rep, T, HeadDim)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x: [B, T, C]
        """
        B, T, C = x.size()
        
        # 1. 前向投影，注意此时 k 和 v 产出的特征大小远小于 q
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # [B, T, H_q, D]   ->  [B, H_q, T, D]
        q = q.view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)
        # [B, T, H_kv, D]  ->  [B, H_kv, T, D]
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 2. 召唤核心分身术！！
        # 只有分身后大家都在 H 这一维达到统一的 H_q 数量后，才能合法使用底层的 matmul 内积互算！
        k = self._repeat_kv(k, self.num_q_per_kv)  # 打肿脸充胖子变成 -> [B, H_q, T, D]
        v = self._repeat_kv(v, self.num_q_per_kv)  # 打肿脸充胖子变成 -> [B, H_q, T, D]
        
        # 3. 标准打分系统恢复如常 (后续的逻辑完全等价顺接 MHA 的机制！)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        
        out = torch.matmul(attn_weights, v)  # [B, H_q, T, HeadDim]
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o_proj(out)
