"""
SwiGLU 面试级核心模拟实现 
========================
【理论揭秘】：当前大模型（LLaMA / Qwen / Mistral）横扫千军的最强 FFN (前馈网络层) 独裁标配激活策略！
之前最传统的 Transformer：Linear层 -> ReLU -> Linear层。
这种用单线直通车配一个截断去逼迫产生非线性的方式实在太死板低级（连载力有限）。
GLU (Gated Linear Unit 门控逻辑算子) 用了两条双线齐发的通道在末尾通过门卫体系相乘合并！
SwiGLU = (x @ W_1 * Swish(x @ W_2)) @ W_3
其中一条路用魔法函数 Swish = x * sigmoid(\beta x) 去作为极其平滑顺滑且在负区间还具有细致微小下凹捕捉力的高级门卫来掌控大权！
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int = None):
        super().__init__()
        out_features = out_features or in_features
        
        # 传统模式只有一个 W1 进行膨胀，但在此为了形成 GLU 门卫并发机制需要兵分两路建立：
        # 并发道一 (准备作为掌控一切的 Gating 裁判门阀)
        self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
        # 并发道二 (只管带着各种原始基础信息野蛮冲撞上前的炮灰主体)
        self.w_up = nn.Linear(in_features, hidden_features, bias=False)
        
        # 最终合并融合大军汇聚阵列的降维出站层
        self.w_down = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x: torch.Tensor):
        # 1. 炮灰道二兵临城下
        x_up = self.w_up(x) 
        
        # 2. 裁判道一形成终极门阀
        gate_logits = self.w_gate(x)
        
        # 极其核心高超的平滑激活操作：Swish (也叫 SiLU) 函数。 
        # 因为平滑具有可导优胜性并带给参数非常舒服极度温柔的寻峰坡度保护。(即：x * sigmoid(x))
        x_gated = F.silu(gate_logits)  
        
        # 3. GLU 机制显现：以门卫裁判 x_gated 乘给主体 x_up 执行残酷放行过滤 (非线性在此极其复杂扭曲地被赋予进了数据)
        merged = x_gated * x_up
        
        # 4. 融合大军收束降维出击归位
        out = self.w_down(merged)
        return out
