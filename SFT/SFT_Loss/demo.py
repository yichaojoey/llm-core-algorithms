"""
SFT 训练演示：证明训练框架下被无视的用户问答区域和梯度错位导向
======================================================
验证对 User 区域进行 -100 惩罚抑制的原理。
"""

import torch
import torch.nn as nn
from sft_loss import compute_sft_loss

def run_sft_demo():
    print("=" * 60)
    print("演示：SFT 训练时独有的 Mask (忽略 User Prompt) 以及 Shift 逻辑")
    print("=" * 60)
    
    batch_size = 1
    seq_len = 6
    vocab_size = 1000
    ignore_index = -100
    
    # 构造一条真实的微调对话序列输入值：
    # "USER: 你好！ ASSISTANT: 嗨！"
    # 我们假设系统切分 Token 索引化为: [101, 102, 201, 202, 301, 302]
    # 前面的 [101, 102] 代表了 User 的上文逻辑 (User Context)
    # [201] 姑且算作 Assistant 头标注 (Prompt Control Prefix)
    # 实际要求大语言模型生成的干货也就是只希望它能精准打分的只有: [202, 301, 302] 这三个词
    input_ids = torch.tensor([[101, 102, 201, 202, 301, 302]])
    
    # ===============================
    # 面试核心点：打 Mask (Labels遮挡设定)
    # ===============================
    # 绝不能拿大模型去强迫记住 User 等下会提问什么词而计算 Loss！
    labels = input_ids.clone()
    
    # 对前三个 token (USER: 你好！ ASSISTANT:) 打上负向屏蔽，不将它们送往交叉熵验证
    labels[0, :3] = ignore_index 
    
    print(f"🔹 原始灌入给模型的 Input IDs: \t\t{input_ids.tolist()}")
    print(f"🔹 处理好的金标准 Labels (带 Mask): \t{labels.tolist()}")
    print(f"   => 注意：由于错位切片影响，模型真切受到梯度的只有 {labels[0, 3:].tolist()} 目标字。\n")

    # 把它变成要求导监测的抽象变量集合
    torch.manual_seed(42)
    logits = nn.Parameter(torch.randn(batch_size, seq_len, vocab_size))
    
    optimizer = torch.optim.SGD([logits], lr=0.1)
    
    optimizer.zero_grad()
    loss = compute_sft_loss(logits, labels, ignore_index=ignore_index)
    loss.backward()
    
    print("--- 梯度反向流溯源跟踪现象 ---")
    print(f"交叉熵混合截断的最终 Loss 标量值: {loss.item():.4f}")
    
    # 梯度检查：由于我们 Shift 了一位 (logits[:-1] 拼接到 labels[1:])
    # labels[1:] => [-100, -100, 202, 301, 302]
    # 对齐后: 
    #   Step 0 (101): 无梯度 (由于标签被 -100)
    #   Step 1 (102): 无梯度 (由于标签被 -100)
    #   Step 2 (201): 有梯度！因为他要负责靠着之前的铺垫信息蒙出 202
    #   ...
    #   Step 5 (302): 无梯度！因为位于序列尾端由于 :-1 被 Shift 遗弃
    grad_norm = [torch.norm(logits.grad[0, i]).item() for i in range(seq_len)]
    
    print("\n🧐 各个时间片上的隐藏预测状态最终吃到了多少梯度 L2 模长：")
    for i, norm in enumerate(grad_norm):
        if norm == 0.0:
            status = "👻 无梯度 (被 Ignore Index 无视 或 被末端 Shift 强行遗弃)"
        else:
            status = "🔥 被点燃！接受模型梯度反馈更新！"
        print(f"  Step {i} ({input_ids[0, i].item():3}):  {norm:8.4f} -> {status}")
        
    print("\n✅ 从跟踪结果可以证实：只有生成器真正负责去‘说自己该说的话 (Assistant区域)’的隐含输出层状态受到了修正！")

if __name__ == "__main__":
    run_sft_demo()
