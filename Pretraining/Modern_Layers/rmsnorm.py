"""
RMSNorm 面试级核心模拟实现 
========================
【理论揭秘】：老旧时代的 LayerNorm 要算一个大矩阵切面的均值并将其减掉（Mean Centering 均值中心化）。
这是对原本计算结果最底色的破坏。研究表明，其实起到规训网络并消除巨大异常梯度抖动作用的，根本不是把所有人的分数从中间归零平移（减去Mean均值）。
绝对核心功臣是后面的**方差缩放（Variance Scaling）**，保证大暴雪的超级巨大长尾不会炸穿数值域即可！
RMS (Root Mean Square 均方根) 直接放弃一切平移操作，凭硬气直接去除一个平方根大分母。省下了极大 GPU 带宽并且提速，效果纹丝不退。
"""

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        dim: 其实就是 embedding 的常驻厚度维度
        """
        super().__init__()
        self.eps = eps
        # 【考点/重点】：虽然放弃了所有的均值移动。
        # 但是最后的拉伸重塑打粉放行权限必须拥有：
        # 这里存在一个全是 1 的权重矩阵来承载属于那一层的最终独家缩放放行权重！
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        面试硬核公式：x_{normed} = x / sqrt( Mean(x^2) + eps )  * weight
        """
        # 不要搞传统的 x - x.mean(dim=-1) 了！
        # 直接拿去全局强行平方: x^2
        x_squared = x.pow(2)
        
        # 沿着最后一维特征方向求取无情的冷血大均值 (Mean(x^2))
        variance = x_squared.mean(dim=-1, keepdim=True)
        
        # 为了不发生除以 0 雪崩，加极小点缀微值后硬开根号
        # x 强行用作反向相乘广播缩水
        x_normed = x * torch.rsqrt(variance + self.eps)
        
        # 加上独家学习掌控权通过输出大门
        return self.weight * x_normed

    # 【理论附加揭秘】：
    # rsqrt (reciprocal square root = 1 / sqrt(x)) 在底层 CUDA/C++ 中有特定的极其残暴极限指令进行连贯处理。
    # 不要分开写 x / torch.sqrt(var) 而是要直接使用 torch.rsqrt() 结合乘法，这样速度不仅超越 LayerNorm，还能把 IO 并发推向巅峰。
