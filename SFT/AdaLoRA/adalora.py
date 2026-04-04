"""
AdaLoRA 面试级核心实现
========================
相比干干巴巴的 LoRA，它利用 SVD (奇异值分解) 的完美数学形式：W_new = W + P @ diag(E) @ Q
在微调中通过 E（奇异值打分）的大小，自动甄别哪个权重层的哪一列更重要，从而把没用的 Rank 容量让给急需救火的关键网络层！
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaLoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # ==========================================
        # SVD 三重体（原装 LoRA 只有 AB 两个，这里掰成了3个）
        # ==========================================
        
        # 1. lora_Q (相当于 LoRA 的 A)
        self.lora_Q = nn.Parameter(torch.empty((r, in_features))) # 右奇异矩阵 V_T 类
        
        # 2. lora_E (绝对核心：动态对角阵，储存这每个维度通道的权重打分)
        # 巧妙做法：它只是一个一维向量，但在计算中充当 diag 乘法的作用以省去零开销
        self.lora_E = nn.Parameter(torch.empty(r)) 
        
        # 3. lora_P (相当于 LoRA 的 B)
        self.lora_P = nn.Parameter(torch.empty((out_features, r))) # 左奇异矩阵 U 类
        
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """🌟 面试硬核防坑点：死锁断分初始化大考验！🌟"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # 传统 LoRA 是 A 随机，B 给全 0。
        # 但在 AdaLoRA 中，乘积链是 P * E * Q。
        # 如果你这里 P 给 0，且 E 也给 0，那么恭喜你梯度链彻底爆掉死锁了！
        # 推导：d_L / d_E 包含了 P_T*...*Q_T。如果 P=0，那 E 的梯度永远是零！模型到死也学不会！
        
        # ✅ 正确做法：
        nn.init.normal_(self.lora_P) # P 必须有微弱的方差保证出点声音
        nn.init.normal_(self.lora_Q) # Q 同样打乱
        nn.init.zeros_(self.lora_E)  # 真正让初始 ΔW=0 不破坏系统的重负交给了把门的核心分数板 E！
        
    def forward(self, x: torch.Tensor):
        original_output = F.linear(x, self.weight, self.bias)
        dropout_x = self.lora_dropout(x)
        
        # Q 提取特征 
        lora_Q_out = F.linear(dropout_x, self.lora_Q)
        
        # E 进行控制阀缩放 
        # 对角矩阵 E 的乘法其实就是一个带广播机制的通道级逐元素相乘，比老实建立 nxn 甚至大得多的方阵去作纯正的低阻对角矩阵乘法极大地缩短了速度且暴省内存！
        lora_E_out = lora_Q_out * self.lora_E.view(1, 1, -1) if x.dim() == 3 else lora_Q_out * self.lora_E.view(1, -1)
        
        # P 还原特征
        lora_P_out = F.linear(lora_E_out, self.lora_P)
        
        return original_output + lora_P_out * self.scaling
    
    def mask_prune_e(self, threshold: float = 0.01):
        """
        验证考点：剪裁掉废柴结构的能力。
        由于分离出了 E 控制流，当我们查表发现有些通道 E 学了半天都在 0 附近游走时，
        果断将它斩首变为无（0）。这样等重参数化保存时，这个秩的计算全变为 0，从而腾出了参数空间余额给别人！
        """
        with torch.no_grad():
            # 找到数值不争气太过于弱小的弱通道位
            mask = (self.lora_E.abs() > threshold).float()
            # 物理层面剥夺其存活
            self.lora_E.mul_(mask)
            
    def mark_only_lora_as_trainable(self):
        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.lora_P.requires_grad = True
        self.lora_E.requires_grad = True
        self.lora_Q.requires_grad = True
