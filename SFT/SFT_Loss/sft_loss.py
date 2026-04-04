"""
SFT (Supervised Fine-Tuning) Causal LM Loss 核心实现
===================================================
所有问答大模型的绝对基石。它的本质就是“看到前 t 个词，让你去猜第 t+1 个词到底是词典里的哪个”。
"""

import torch
import torch.nn.functional as F

def compute_sft_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100):
    """
    计算基于 Causal Language Modeling (CLM) 的 SFT 交叉熵损失。
    
    Args:
        logits: 模型输出的整个概率对数映射分布 [Batch, SeqLen, VocabSize]
        labels: 对应的目标金标准 Token ID 库，其中待忽略的部分已替换为 ignore_index [Batch, SeqLen]
        ignore_index: 损失函数中忽略梯度的 label 值编号，通常沿用 PyTorch 模型惯例设为 -100
        
    Returns:
        loss: 合成的标量
    """
    # 核心面试点 1：Shift-by-one (错位对齐)
    # 因为位置 t 在经过自注意力处理后拿到的隐变量 logits 实际上是去跟位置 t+1 的真实的下一个文本词去比照计算的！
    # logits 抛弃最后一个时间预测结果（因为它猜测的是原本序列终点之后的不存在的虚空词）
    shift_logits = logits[..., :-1, :].contiguous()
    
    # labels 则要抛弃第一个被直接塞入脑海的符号词（因为它没有被赋予任何前置信息去推算）
    shift_labels = labels[..., 1:].contiguous()
    
    # 核心面试点 2：拆墙铺平维度 (Flatten)
    # PyTorch 框架自带的 CrossEntropyLoss 函数强制要求 input 待查验对象压平成二维 [Batch * SeqLen, VocabSize]
    # 强制要求 target 压平成一维的大数组串 [Batch * SeqLen]
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1),
        ignore_index=ignore_index
    )
    
    return loss
