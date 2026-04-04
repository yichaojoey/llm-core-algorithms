import torch
from sft_loss import compute_sft_loss

def test_sft_shift_logic_formula():
    """纯手工演算对比校验 SFT Loss 是否精准计算了最后一个生效词的对数交叉熵"""
    
    # logits 有三项序列产出，字典数量为 2
    logits = torch.tensor([[[2.0, -1.0], [0.0, 3.0], [-1.0, 1.0]]])  # [1, 3, 2]
    
    # Labels 里面只有最后一位没带 -100，意味着整串数组最终全都被无视了，除了拿 [0.0, 3.0] 分布去推算种类索引为 1 的正确性
    labels = torch.tensor([[-100, -100, 1]])  # [1, 3]
    
    # == 手工公式演算过程 ===
    # shift_logits: logits[:, :-1, :] ->  抛弃掉最后一个预测, 切剩下：[2.0, -1.0] (步T=0) 与 [0.0, 3.0] (步T=1)
    # shift_labels: labels[:, 1:] -> 抛弃掉第一个无头源标签, 切剩下：[-100] (步T=1的标签被忽视) 与 [1] (步T=2拥有标签 1!)
    # 两相对比，发生效用的只在 T=1 那次去拿预测值 [0.0, 3.0] 靠向基准解 1 类。
    # 手算交叉熵公式 = -log( exp(类1得分) / (exp(类0得分) + exp(类1得分)) ) 
    #               = -log( exp(3) / (exp(0) + exp(3)) )  ≈ 0.0485 ...
    
    loss = compute_sft_loss(logits, labels, ignore_index=-100)
    
    expected_loss = -torch.log(torch.exp(torch.tensor(3.0)) / (torch.exp(torch.tensor(0.0)) + torch.exp(torch.tensor(3.0))))
    
    assert abs(loss.item() - expected_loss.item()) < 1e-4

def test_sft_gradient_flow_time_steps():
    """测试带有梯度反馈流的时序惩罚特性与切片抛弃特性"""
    logits = torch.randn(1, 4, 10, requires_grad=True)
    
    # 给一段 4 词长度的标签矩阵：无视、无视、强制5类、无视
    labels = torch.tensor([[-100, -100, 5, -100]])
    
    # [模型对齐 Shift 分析]：
    # logits[0] 用来推论 labels[1] (-100) -> 免死金牌无梯度
    # logits[1] 用来推论 labels[2] (5)    -> 产生倒逼惩罚梯度！
    # logits[2] 用来推论 labels[3] (-100) -> 免死金牌无梯度
    # logits[3] 由于序列长度没有第5个标签被末端遗弃切走 -> 完全安全无梯度
    
    loss = compute_sft_loss(logits, labels)
    loss.backward()
    
    # 强制验证该有的有，不该有的完全没被连累：
    assert torch.norm(logits.grad[0, 0]).item() < 1e-6
    assert torch.norm(logits.grad[0, 1]).item() > 0
    assert torch.norm(logits.grad[0, 2]).item() < 1e-6
    assert torch.norm(logits.grad[0, 3]).item() < 1e-6
