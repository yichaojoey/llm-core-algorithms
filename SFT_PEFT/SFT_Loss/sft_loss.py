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
    # 【理论揭秘】：大语言模型的本质是自回归 (Auto-Regressive) 生成，即最大化似然 P(x_{t+1} | x_1, x_2, ..., x_t)。
    # 这意味着隐状态 h_t 对应的是前 t 个词的历史信息，它必须被用来预测确切的第 t+1 个词！
    # 如果错位没对齐，让 h_t 去预测第 t 个词，那就成了一个毫无难度的“偷看现在答案的复制游戏”，模型瞬间垮塌。
    
    # logits 抛弃最后一个时间预测结果（因为它在前向推演时试图猜测原本序列终点之越界后的不存在的虚空词）
    shift_logits = logits[..., :-1, :].contiguous()
    
    # labels 则要严实抛弃第一个被直接一开始塞入脑海的源头词（因为它没有被赋予任何前置信息网络隐层去推算它）
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
