"""
RoPE (Rotary Position Embedding) — 最小可运行实现
=================================================

本文件实现了 RoPE 的核心算法。RoPE 是现代大语言模型（如 LLaMA、GPT-NeoX）
中最常用的位置编码方式之一。

核心思想：
    不像传统的绝对位置编码（把位置向量"加"到 token embedding 上），
    RoPE 通过"旋转"query 和 key 向量来注入位置信息。
    旋转的角度取决于 token 的位置和所在维度。

数学原理：
    对于位置 m 和维度对 (2i, 2i+1)，RoPE 做的事情相当于：
    
    [q'_{2i}  ]   [cos(m·θ_i)  -sin(m·θ_i)] [q_{2i}  ]
    [q'_{2i+1}] = [sin(m·θ_i)   cos(m·θ_i)] [q_{2i+1}]
    
    其中 θ_i = 1 / (base^{2i/d})，base 通常取 10000，d 是 embedding 维度。
    
    这个旋转操作可以用复数乘法高效实现：
    (q_{2i} + j·q_{2i+1}) × (cos(m·θ_i) + j·sin(m·θ_i))
    
    也就是把相邻两个维度看成一个复数，然后乘以一个单位复数（旋转因子）。
"""

import torch
import torch.nn as nn
from typing import Optional


# ============================================================================
# 第一步：预计算旋转频率
# ============================================================================

def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """
    预计算所有位置的旋转频率（以复数 e^{j·m·θ} 的形式存储）。
    
    这个函数只需要在模型初始化时调用一次，之后反复使用缓存的结果即可。

    参数:
        dim (int): 每个 attention head 的维度大小（注意：必须是偶数，
                    因为 RoPE 将维度两两配对进行旋转）
        max_seq_len (int): 支持的最大序列长度
        theta (float): 频率基数，默认 10000.0（论文原始设定）

    返回:
        freqs_cis: shape = (max_seq_len, dim // 2)，类型为 complex64
                   每个元素是 cos(m·θ_i) + j·sin(m·θ_i)
    
    数学推导:
        第 i 个维度对的频率: θ_i = 1 / (theta^{2i/d})
        位置 m 的旋转角度: m * θ_i
        旋转因子（复数）: e^{j·m·θ_i} = cos(m·θ_i) + j·sin(m·θ_i)
    """

    # ---------- 步骤 1: 计算每个维度对的基础频率 θ_i ----------
    # 维度索引: [0, 2, 4, ..., dim-2]，共 dim//2 个
    # 公式: θ_i = 1 / (theta^{2i/dim})
    #
    # 实现技巧：用 torch.arange 生成 [0, 2, 4, ...] 然后除以 dim，
    #           再取 theta 的负幂次，等价于 1 / theta^{...}
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # freqs 的 shape: (dim // 2,)
    # 例如 dim=8 时: freqs = [θ_0, θ_1, θ_2, θ_3]
    #   θ_0 = 1/10000^{0/8} = 1.0        (变化最快的频率)
    #   θ_1 = 1/10000^{2/8} ≈ 0.0316
    #   θ_2 = 1/10000^{4/8} = 0.01
    #   θ_3 = 1/10000^{6/8} ≈ 0.00316    (变化最慢的频率)

    # ---------- 步骤 2: 生成位置索引 ----------
    # positions = [0, 1, 2, ..., max_seq_len - 1]
    positions = torch.arange(max_seq_len).float()
    # positions 的 shape: (max_seq_len,)

    # ---------- 步骤 3: 计算每个 (位置, 维度对) 的旋转角度 ----------
    # 用外积: angles[m][i] = m * θ_i
    # 这就是位置 m 在第 i 个维度对上的旋转角度
    angles = torch.outer(positions, freqs)
    # angles 的 shape: (max_seq_len, dim // 2)
    # 例如: angles[5][2] = 5 * θ_2 = 0.05  (位置5在第2个维度对的旋转角度)

    # ---------- 步骤 4: 转换为复数形式的旋转因子 ----------
    # e^{j·angle} = cos(angle) + j·sin(angle)
    # torch.polar(abs, angle) 生成复数: abs * e^{j·angle}
    # 这里 abs = 1（单位复数，只旋转不缩放）
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    # freqs_cis 的 shape: (max_seq_len, dim // 2), dtype = complex64
    # 每个元素都是模为1的复数，代表一个旋转操作

    return freqs_cis


# ============================================================================
# 第二步：将 RoPE 旋转应用到 query/key 向量上
# ============================================================================

def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """
    将旋转位置编码应用到输入张量 x 上。
    
    核心操作：把 x 的相邻维度两两配对，视为复数，然后乘以旋转因子。

    参数:
        x (torch.Tensor): 输入张量
            shape = (batch_size, seq_len, n_heads, head_dim)
            这是经过线性投影后的 query 或 key
        freqs_cis (torch.Tensor): 预计算的旋转频率
            shape = (max_seq_len, head_dim // 2), dtype = complex64

    返回:
        输出张量，shape 与输入 x 相同，但已经被"旋转"过了
    
    维度变化过程:
        x:           (batch, seq_len, n_heads, head_dim)
        → reshape:   (batch, seq_len, n_heads, head_dim//2, 2)
        → 转复数:     (batch, seq_len, n_heads, head_dim//2)  [complex]
        → 乘旋转因子: (batch, seq_len, n_heads, head_dim//2)  [complex]
        → 转回实数:   (batch, seq_len, n_heads, head_dim//2, 2)
        → reshape:   (batch, seq_len, n_heads, head_dim)
    """

    # ---------- 步骤 1: 将实数张量转为复数张量 ----------
    # 把最后一个维度 head_dim 拆成 (head_dim//2, 2)
    # 然后把每对 (a, b) 看成复数 a + jb
    #
    # 例如 head_dim=8 时:
    #   原始: [x0, x1, x2, x3, x4, x5, x6, x7]
    #   配对: [(x0,x1), (x2,x3), (x4,x5), (x6,x7)]
    #   复数: [x0+jx1, x2+jx3, x4+jx5, x6+jx7]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # x_complex 的 shape: (batch, seq_len, n_heads, head_dim // 2)

    # ---------- 步骤 2: 调整 freqs_cis 的 shape 以便广播 ----------
    # freqs_cis 原始 shape: (max_seq_len, head_dim // 2)
    # 我们需要它变成:        (1, seq_len, 1, head_dim // 2)
    # 这样才能和 x_complex 的 (batch, seq_len, n_heads, head_dim//2) 广播相乘
    #
    # 具体做法：只取需要的序列长度，然后 unsqueeze 加上 batch 和 heads 维度
    seq_len = x.shape[1]
    freqs_cis = freqs_cis[:seq_len]                     # (seq_len, head_dim//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)     # (1, seq_len, 1, head_dim//2)

    # ---------- 步骤 3: 复数乘法 = 旋转！！ ----------
    # 这就是 RoPE 的核心操作！
    #
    # 复数乘法的几何意义：
    #   (a + jb) × (cosθ + jsinθ) = (a·cosθ - b·sinθ) + j(a·sinθ + b·cosθ)
    #   这等价于对向量 [a, b] 做了角度为 θ 的旋转！
    #
    # 每个位置、每个维度对都有自己的旋转角度，是通过 freqs_cis 决定的
    x_rotated = x_complex * freqs_cis
    # x_rotated 的 shape: (batch, seq_len, n_heads, head_dim // 2), complex

    # ---------- 步骤 4: 将复数张量转回实数张量 ----------
    # 先把复数拆成实部和虚部: (batch, seq_len, n_heads, head_dim//2, 2)
    # 再 flatten 回原始 shape: (batch, seq_len, n_heads, head_dim)
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    # x_out 的 shape: (batch, seq_len, n_heads, head_dim) ← 和输入一样！

    return x_out.type_as(x)  # 保持和输入相同的 dtype


# ============================================================================
# 第三步：封装为 PyTorch Module
# ============================================================================

class RoPE(nn.Module):
    """
    RoPE (Rotary Position Embedding) 的 PyTorch Module 封装。
    
    使用方法:
        rope = RoPE(head_dim=64, max_seq_len=2048)
        q_rotated = rope(q)  # q shape: (batch, seq_len, n_heads, head_dim)
        k_rotated = rope(k)  # k shape: (batch, seq_len, n_heads, head_dim)
    
    为什么要封装成 Module:
        1. 旋转频率只需计算一次，存为 buffer，跟随模型移动到 GPU
        2. 接口更简洁，调用时只需传入 x
        3. 和其他 PyTorch 模块无缝集成
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        """
        初始化 RoPE 模块。

        参数:
            head_dim (int): 每个 attention head 的维度（必须是偶数）
            max_seq_len (int): 支持的最大序列长度
            theta (float): 频率基数
        """
        super().__init__()

        # 参数校验：head_dim 必须是偶数，因为 RoPE 将维度两两配对
        assert head_dim % 2 == 0, f"head_dim 必须是偶数，但收到 {head_dim}"

        # 预计算旋转频率，并注册为 buffer
        # register_buffer 的作用:
        #   1. 不会被当作可训练参数（不参与梯度计算）
        #   2. 会跟随 model.to(device) 自动移动到正确的设备
        #   3. 会被 state_dict() 保存和加载
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        # persistent=False 表示不保存到 state_dict（因为可以随时重新计算）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：对输入应用旋转位置编码。

        参数:
            x (torch.Tensor): shape = (batch_size, seq_len, n_heads, head_dim)
        
        返回:
            旋转后的张量，shape 不变
        """
        return apply_rotary_emb(x, self.freqs_cis)


# ============================================================================
# 附加工具函数：用于理解和调试
# ============================================================================

def rope_rotation_matrix(pos: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """
    为给定位置生成完整的旋转矩阵（仅用于教学目的，实际不会使用）。
    
    这个函数展示了 RoPE 背后的完整旋转矩阵，帮助理解数学原理。
    实际实现中我们用复数乘法来高效完成同样的操作。

    参数:
        pos (int): token 的位置索引
        dim (int): embedding 维度（必须是偶数）
        theta (float): 频率基数

    返回:
        rotation_matrix: shape = (dim, dim) 的旋转矩阵
    
    矩阵结构（以 dim=6 为例）:
        [cos(m·θ₀)  -sin(m·θ₀)  0           0           0           0         ]
        [sin(m·θ₀)   cos(m·θ₀)  0           0           0           0         ]
        [0            0          cos(m·θ₁)  -sin(m·θ₁)  0           0         ]
        [0            0          sin(m·θ₁)   cos(m·θ₁)  0           0         ]
        [0            0          0           0           cos(m·θ₂)  -sin(m·θ₂)]
        [0            0          0           0           sin(m·θ₂)   cos(m·θ₂)]
    
    可以看到，它是一个分块对角矩阵，每个 2×2 块是一个独立的旋转。
    """
    # 初始化为零矩阵
    R = torch.zeros(dim, dim)

    # 填充每个 2×2 旋转块
    for i in range(dim // 2):
        # 第 i 个维度对的频率
        freq = 1.0 / (theta ** (2.0 * i / dim))
        # 位置 pos 的旋转角度
        angle = pos * freq

        cos_val = torch.cos(torch.tensor(angle))
        sin_val = torch.sin(torch.tensor(angle))

        # 填充 2×2 旋转矩阵块
        # [cos  -sin]
        # [sin   cos]
        R[2 * i,     2 * i]     = cos_val
        R[2 * i,     2 * i + 1] = -sin_val
        R[2 * i + 1, 2 * i]     = sin_val
        R[2 * i + 1, 2 * i + 1] = cos_val

    return R
