import torch
import torch.nn as nn
from qlora import QLoRALinear

def test_qlora_dequantization_dimension():
    """测试非常容易写出 Bug 的缩放广播形状回溯，必须确保还原后的矩阵符合最初被掩盖的全矩阵尺寸"""
    in_f = 128
    out_f = 64
    block_size = 16
    
    model = QLoRALinear(in_features=in_f, out_features=out_f, r=4, block_size=block_size)
    
    # 调用底层反量化进行恢复全尺寸
    dequantized_w = model._dequantize_weight()
    
    # 要求铺开后的 W 跟外部输入向量 X 最后做 nn.Linear 参数尺寸完全相抵符合
    assert dequantized_w.shape == (out_f, in_f)
    assert dequantized_w.requires_grad is False # 这堆释放出来的虽然是全精浮点，但也决不可以要求梯度记忆

def test_qlora_trainable_parameters_freeze():
    """测试量化参数块与附带的缩放索引决不可流出梯度信息"""
    model = QLoRALinear(in_features=32, out_features=32, r=4, block_size=16)
    model.mark_only_lora_as_trainable()
    
    # 量化阵列锁死
    assert model.quantized_weight.requires_grad is False
    # 所属这堆量化阵列对应辅助的 Block Scale 也死锁
    assert model.absmax_blocks.requires_grad is False
    
    # 唯一吸纳活水的源头
    assert model.lora_A.requires_grad is True
    assert model.lora_B.requires_grad is True
