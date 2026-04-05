import torch
import torch.nn as nn
from lora import LoRALinear

def test_lora_zero_init_safeguards():
    """测试 LoRA 的无损对位：初始旁路增量应当保持原模型的硬输出毫无波动"""
    model = LoRALinear(in_features=16, out_features=16, r=4)
    # 取缔旁路 dropout 防止掉包导致前后分布有偶然差异
    model.eval() 
    
    x = torch.randn(2, 5, 16)
    
    out_lora = model(x)
    # 脱离 Forward 函数，直接从参数硬扣手算主模型骨干推理基线
    out_base = torch.matmul(x, model.weight.T) + model.bias
    
    assert torch.allclose(out_lora, out_base, atol=1e-6)

def test_lora_trainable_parameters_freeze():
    """测试被声明冻结后参数图流转阻断情况是否吻合预期设计要求"""
    model = LoRALinear(in_features=16, out_features=16, r=4)
    model.mark_only_lora_as_trainable()
    
    assert model.weight.requires_grad is False
    assert model.bias.requires_grad is False
    
    # 唯一允许开门要梯度吸经验的对象：
    assert model.lora_A.requires_grad is True
    assert model.lora_B.requires_grad is True

def test_lora_backward_flow_segregation():
    """测试反向传播在遇到阻塞梯度的叶子节点与开合分支节点上的定向落点正确性"""
    model = LoRALinear(in_features=16, out_features=16, r=4)
    model.mark_only_lora_as_trainable()
    
    x = torch.randn(2, 5, 16)
    model.train()
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    assert model.weight.grad is None
    assert model.bias.grad is None
    # 截流完成。A / B 将带着更新好的差异量进入下个世代前向结合态
    assert model.lora_A.grad is not None
    assert model.lora_B.grad is not None
