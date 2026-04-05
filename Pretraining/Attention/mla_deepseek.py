"""
核心揭秘 (2025 面试神题)：DeepSeek V3/R1 架构底座 - Multi-head Latent Attention (MLA)
=============================================================================
【理论揭秘】：
在老旧的 GQA 架构下，为了应对 128K 的恐怖长文，$KV \ Cache$ (上下文高速缓存) 即便是被分组(Groups)，依然大得足以撑爆任何 80G A100 甚至 H100 显卡。
【DeepSeek 的突围之战：MLA 压缩】：
DeepSeek 的研究员极其暴力地抛弃了直接缓存巨大的 $K$ 和 $V$ 矩阵：
1. **潜变量下拽 (Latent Compression)**: 他们设计了一个极度狭窄的“漏斗”向量 $c_{KV}$ (比如只有 512 维)。把原来浩如烟海的高维上下文信息，通过线性层直接拍成这 512 维的碎片！
2. **极速上采样 (Up-Projection)**: 在每一次生成新词时，拿出这个极小尺寸的 $c_{KV}$，实时乘以一个展开矩阵 ($Up\_Proj$) 把它瞬间再撑大还原回正常的 $K$ 和 $V$ 。
   **结论**：你的常驻内存里（显存上），只需要存那点小漏斗（$c_{KV}$）！显存占用直接从 10GB 暴降到 0.5 GB 以下！
3. **Decoupled RoPE (位置编码剥离解耦)**: 为了防止位置编码因为压缩被扭曲，他们额外给 $Q$ 和 $K$ 挂载了不参与压缩的游离 RoPE 向量 ($q_{RoPE}$, $k_{RoPE}$)。

这份代码是极度纯净版的中国巅峰大模型核心注意力机理！
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model=4096, num_heads=128, latent_dim=512, rope_dim=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.latent_dim = latent_dim  # 这个极度微小的 512 就是魔法核心！
        self.rope_dim = rope_dim
        
        # 1. 对 Query 落地的处理: 
        # DeepSeek 把 Q 也做了一次压缩，叫 c_Q (但这只是为了少算点乘法，不是为了省内存)
        self.q_down_proj = nn.Linear(d_model, latent_dim, bias=False)
        self.q_up_proj = nn.Linear(latent_dim, self.num_heads * self.head_dim, bias=False)
        
        # RoPE 这个坐标信息极其敏感，绝不能参杂在压缩包里，必须挂载在外面
        self.q_rope_proj = nn.Linear(d_model, self.num_heads * rope_dim, bias=False)
        
        # 2. 核心重头戏：真正的缓存杀手 c_KV
        self.kv_down_proj = nn.Linear(d_model, latent_dim, bias=False) 
        
        # 实时还原器（把 512 还原回浩瀚维度的 K 和 V）
        self.k_up_proj = nn.Linear(latent_dim, self.num_heads * self.head_dim, bias=False)
        self.v_up_proj = nn.Linear(latent_dim, self.num_heads * self.head_dim, bias=False)
        
        # 同理，KV 的位置坐标必须外挂，以防被毁坏
        self.k_rope_proj = nn.Linear(d_model, rope_dim, bias=False)
        
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, d_model, bias=False)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        B, seq_len, _ = x.shape
        
        print("\n=== [DeepSeek MLA 流程启动] ===")
        
        # ============== [第一步：压缩与重构 Query] ==============
        c_q = self.q_down_proj(x)                  # [B, seq_len, 512] 拍扁
        q_content = self.q_up_proj(c_q)            # [B, seq_len, num_heads * head_dim] 还原内容
        q_rope = self.q_rope_proj(x)               # [B, seq_len, num_heads * 64] 抽取位置信息
        
        # ============== [第二步：封神之战 - 产生潜变量 c_KV] ==============
        # 这个变量 c_kv 是在长下文推理时唯一需要塞入显存的张量！！它只有 512 维！！！
        c_kv = self.kv_down_proj(x)                # [B, seq_len, 512]
        print(f"[魔法]: 当前上下文虽然长达几十万，但我生成的潜向量极其精瘦: {c_kv.shape} (这是正常 KV 的几十万分之一！)")
        
        # 提取极少量的 RoPE 位置给上下文做标记
        k_rope = self.k_rope_proj(x)               # [B, seq_len, 64] 
        # 注意：DeepSeek 中所有的 Head 是共享这一个 k_rope 维度的，这也是极度的节约！
        
        # ============== [第三步：使用前，实时瞬间还原成庞然大物 (Up-Projection)] ==============
        # 这里只是在运算时存在 GPU L1/L2 缓存里，一闪而过，根本不占据长期的 Global Memory！
        k_content = self.k_up_proj(c_kv)           # [B, seq_len, num_heads * head_dim]
        v_content = self.v_up_proj(c_kv)           # [B, seq_len, num_heads * head_dim]
        
        print(f"[魔法]: 只有在准备打分相乘的瞬间，庞然大物才被还原出来: K->{k_content.shape}")
        
        # ============== [第四步：维度重塑 (Reshape for Heads)] ==============
        q_c = q_content.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q_r = q_rope.view(B, seq_len, self.num_heads, self.rope_dim).transpose(1, 2)
        
        k_c = k_content.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # DeepSeek 的 k_r 因为是共享的，所以我们在头上直接做广播扩展 (unsqueeze)
        k_r = k_rope.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        v_c = v_content.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # ============== [第五步：内容与位置合并拼接 (Concat)] ==============
        # 这是 DeepSeek 的巧思：把含有内容信息的 C 和含有坐标位置信息的 R 重新缝合拼回一起！
        q = torch.cat([q_c, q_r], dim=-1)         # q 的每一个头拥有了 [head_dim + rope_dim] 的长度 
        k = torch.cat([k_c, k_r], dim=-1)
        
        # ============== [第六步：传统的 Scaled Dot-Product 注意力] ==============
        scale_factor = (self.head_dim + self.rope_dim) ** 0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) / scale_factor
        
        # Causal mask (省略掉复杂的填 0 逻辑，简化展示)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 乘以恢复好的 Value
        out = torch.matmul(attn_weights, v_c)      # [B, num_heads, seq_len, head_dim]
        
        out = out.transpose(1, 2).contiguous().view(B, seq_len, self.num_heads * self.head_dim)
        final_output = self.o_proj(out)
        
        print("[大满贯]: 在全程不缓存全尺寸 K/V 的情况下，完美完成了所有注意力打分。这个结构支撑了今天统治榜单的神明 R1。")
        return final_output

if __name__ == "__main__":
    print("="*60)
    print("DeepSeek Latent Compression 震撼开局演练 (MLA)")
    print("="*60)
    
    # 模拟 Batch=2, 恐怖的三万字上下文 (这里用 10 做 demo)
    dummy_input = torch.randn(2, 10, 4096)
    
    mla = MultiHeadLatentAttention()
    output = mla(dummy_input)
    print(f"\nFinal Attention Output Shape: {output.shape}\n(它跟传统的庞然大物 MHA 吐出的尺寸一模一样，但内在消耗已经发生了翻天覆地的代差！)")
