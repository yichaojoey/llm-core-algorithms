"""
LoRA (Low-Rank Adaptation) 面试级核心实现
=========================================
冻结原有庞大的 W 矩阵，转而在大模型外挂两个小矩阵 A (d_in -> r) 和 B (r -> d_out)。
数学近似等价于: W_new = W + B @ A * (alpha / r)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """
    用以替代标准 nn.Linear 的 LoRA 适配层。
    在大模型 PEFT 微调时，原本的模型线性层会被替换为此类的实例。
    """
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        
        # 核心面试点：通过除以 r 以维持初始学习率的超参稳定性一致性
        # 【理论揭秘】：在数学上，初始参数由于方差会在乘法中被放大。如果我们将矩阵秩 R 扩大 4 倍，
        # B * A 积出来的输出方差会跟着变大。如果我们强制全局乘以 lora_alpha / r，
        # 则能够保证在切换任意参数大小时，输出的前向/反向梯度的方差稳定平衡。
        # 让使用者无需更改 Learning Rate 这个极其敏感的超参数配置即可随意改变容量！
        self.scaling = self.lora_alpha / self.r
        
        # 1. 原本的大参数权重（训练时需要被彻底冻结，不传挂梯度要求）
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # 2. 初始化 LoRA 的核心小矩阵 A 和 B (这是我们要训练索求梯度的部位!)
        # A: (r, in_features)，负责降维拆解
        self.lora_A = nn.Parameter(torch.empty((r, in_features)))
        # B: (out_features, r)，负责回升恢复
        self.lora_B = nn.Parameter(torch.empty((out_features, r)))
        
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """极其关键且面试必考的非对称初始化"""
        # 原始 Linear 的正常初始化 (假装大模型已经在千亿文本上受过精妙预训练了)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # 核心考点 1：A 采取高斯正态或 Kaiming 随机打散初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # 🌟 核心考点 2：B 必须强制初始化为全部是 0 的零矩阵！
        # 解释：这样能保证未经训练接入大模型时的初始输出 ΔW = 0。不破坏原大模型已经学成的完美世界常识结构！
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor):
        # 原始主骨架支路计算（这部分的参数在微调过程中完全死锁死循环，不吃资源更新）
        original_output = F.linear(x, self.weight, self.bias)
        
        # LoRA 微调旁系支路计算（这部分负责吸纳接受反馈传过来的梯度误差）
        # 计算公式：B @ A @ x * scaling
        # 为了高效，不能让 B 直接乘 A 成大矩阵，必须将乘法交换结合律次序为：((x @ A^T) @ B^T)
        dropout_x = self.lora_dropout(x)
        lora_A_out = F.linear(dropout_x, self.lora_A)
        lora_B_out = F.linear(lora_A_out, self.lora_B)
        
        # 加入 α/r 平滑融合两路输出送向下一层架构
        return original_output + lora_B_out * self.scaling
        
    def mark_only_lora_as_trainable(self):
        """面试官常规加试题：如何关闭或开启计算树的反向叶子流？"""
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True
