import torch
import torch.nn as nn
from adalora import AdaLoRALinear

def test_adalora_zero_init_deadlock_breaking():
    """验证至关重要的非双零死锁机制，保证模型不仅不破坏输出，还要能拿到复苏梯度"""
    model = AdaLoRALinear(in_features=16, out_features=16, r=4)
    model.mark_only_lora_as_trainable()
    model.eval() 
    
    x = torch.randn(2, 5, 16)
    
    out_lora = model(x)
    out_base = torch.matmul(x, model.weight.T) + model.bias
    
    # 1. 验证输出完美贴合一模一样保护了模型
    assert torch.allclose(out_lora, out_base, atol=1e-6)

    model.train()
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    # 2. 验证最核心的防梯度锁死
    # 如果代码写错了让 P 也是 0 并且 E 也是 0 的话，这里 E 永远是不可能通过链式法则拿到梯度的！
    assert model.lora_E.grad is not None
    # P 本身不为0由于 E 挡着门让输出零了，但当倒追逆流反向追溯 E 的时候能把信息顺下来，P最后能拿梯度吗？
    # L = P * E * Q -> dL_dP = dL * E * Q。 因为 E 这个时候是 0 ！ 所以刚开闸的时候 P 拿不到有效梯度的！
    assert torch.norm(model.lora_P.grad).item() < 1e-6

def test_adalora_pruning():
    """验证剪枝操作清缴垃圾分支通道维度"""
    model = AdaLoRALinear(in_features=16, out_features=16, r=4)
    with torch.no_grad():
        # 给 2 个大于 0.1 ，2个小于 0.1
        model.lora_E.copy_(torch.tensor([1.0, 0.05, -2.0, 0.00]))
    
    # 发动清除！
    model.mask_prune_e(threshold=0.1)
    
    assert model.lora_E[0].item() == 1.0
    assert model.lora_E[1].item() == 0.0  # 遇害被清除了屏蔽
    assert model.lora_E[2].item() == -2.0
    assert model.lora_E[3].item() == 0.0
