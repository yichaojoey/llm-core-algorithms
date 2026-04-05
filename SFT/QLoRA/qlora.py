"""
QLoRA (Quantized LoRA) 面试级核心模拟实现
=========================================
通过将主模型参数锁库并进行极致压缩为 4-Bit 量化（节省海量显存）。
只有在 Forward 被调用计算时才会被临时“反量化”为 16-bit 或 32-bit 浮点用于矩阵乘法主推理流。
同时旁边并行外挂轻装上阵的全精度 LoRA A/B 矩阵吃满下发的各种模型纠正梯度。

【核心理念】：算力换时间。由于反量化是极其费计算的行为，QLoRA 往往比普通 LoRA 慢，但是在内存受限单卡上无敌！
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class QLoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: int = 16, block_size: int = 64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = lora_alpha / r
        
        # 高频考点：分块量化机制的 Block 长度控制
        self.block_size = block_size
        
        # ========================================================
        # 1. 模拟 4-Bit NF4 量化权重区 (Storage Precision 参数存储精度)
        # ========================================================
        # 在真实的 BitsAndBytes 库中这部分为打包的 uint8 压缩串区。此处使用自带的内存最低形态 int8 原生演示其压垮维度的理念。
        self.quantized_weight = nn.Parameter(
            torch.randint(-8, 7, (out_features, in_features), dtype=torch.int8), 
            requires_grad=False
        )
        
        # Block-wise 分块量化必然伴随的各个块的绝对最大值 (Absmax) 或缩放尺度对照表
        # 如果全矩阵只用 1 个最大值来放缩，长尾的个别巨大异常数字将碾压全区导致微小权重全盘缩成 0。
        num_blocks = in_features // block_size
        self.absmax_blocks = nn.Parameter(
            torch.abs(torch.randn(out_features, num_blocks)), 
            requires_grad=False
        )
        
        # ========================================================
        # 2. 常规 LoRA 更新接管区 (Compute Precision 计算计算精度 16/32-bit)
        # ========================================================
        self.lora_A = nn.Parameter(torch.empty((r, in_features), dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.empty((out_features, r), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        
        self.reset_parameters()

    def reset_parameters(self):
        # A 依然随机拉开，B 依然严格 0 等待激活
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def _dequantize_weight(self) -> torch.Tensor:
        """
        面试官绝对核心追问点：大模型反量化机制到底是在干嘛？
        【理论揭秘】：如果我们一棒子把一整个矩阵的万亿浮点数压扁为 4位(NF4阶梯)，
        如果有几个罕见的极大奇异值游离突刺（Outliers, 比如 100.0），
        它会把整个刻度压缩的尺子拉长，导致其余处于 -1.0 到 1.0 的正常主成分信息
        全被降维归并舍到了 0 号小槽内！全部被无情抹去清零，模型当场变傻子！
        
        解法：必须在局部执行 Block-wise 量化（比如每 64 个数字一组），给每组建一个私人缩放极值账本。
        """
        # 将原始量化的粗糙低精度整数基础强行提升转换回单/双精度浮点以参与正常算子流通
        w_float = self.quantized_weight.float() 
        
        # 我们按照之前打包切分的 block_size，用 repeat_interleave 将局部私有账本给复制还原铺开发射覆盖。
        repeated_absmax = self.absmax_blocks.repeat_interleave(self.block_size, dim=1)
        
        # (基础反量化示意) 利用还原铺开的长尺度阵列，精细覆盖乘平原本低辨识度数字阵列，补上细节：
        dequantized_w = w_float * repeated_absmax
        
        return dequantized_w

    def forward(self, x: torch.Tensor):
        # 1. 主骨干计算：前向推理必须花费相当的算力当场把 4bit 数据释放为高维展开的假精度，去喂原底层矩阵加法。
        # 此原架构路线输出的 tensor 里面没有任何节点 `requires_grad=True`（权重量化区是死物）。所以主梯度根本不会也无法回流到 4bit 量化底座！
        dequantized_w = self._dequantize_weight()
        original_output = F.linear(x, dequantized_w, self.bias)
        
        # 2. 第二条线：并行的 LoRA 梯队由于是天然纯粹的 16/32 bit 且大开着收梯度的大门，它将独享从损失那边回吐进来的反向报错信息！
        lora_A_out = F.linear(x, self.lora_A)
        lora_B_out = F.linear(lora_A_out, self.lora_B)
        
        return original_output + lora_B_out * self.scaling
        
    def mark_only_lora_as_trainable(self):
        """完全锁死量化原发区，只允许开启 A/B 路。"""
        self.quantized_weight.requires_grad = False
        self.absmax_blocks.requires_grad = False
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True
