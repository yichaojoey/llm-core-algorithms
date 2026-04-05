import torch
from rmsnorm import RMSNorm
from swiglu import SwiGLU

def test_rmsnorm_variance_stability():
    """测试无论均值发生何等惨烈剧烈的移平移波折飘移，平方根求和算法必定能降伏缩放率不爆栈"""
    dim = 16
    rmsnorm = RMSNorm(dim)
    
    # 彻底平移摧毁均值，拉高到 500 万
    x = torch.randn(4, 10, dim) + 5e6 
    
    # 但一旦进行根号方差压制...
    out = rmsnorm(x)
    
    # 它在绝对量级上必定被极度完美地瞬间重新缩水控制按回到低维常数域（绝对不会炸裂或维持五百万量级）
    assert out.abs().max().item() < 100.0  
    

def test_swiglu_dual_pathway_output():
    """测试经过两个并行兵分两路的巨大投射变宽处理后，能不能用第三道归总器重塑回原始的身体"""
    in_dim = 64
    # 在 Llama 等大模型中，隐藏层膨胀率其实高达 4~8 倍（这里比如给了 256）
    hidden_dim = 256  
    
    swiglu = SwiGLU(in_features=in_dim, hidden_features=hidden_dim, out_features=in_dim)
    
    x = torch.randn(8, 20, in_dim) 
    
    out = swiglu(x)
    
    # 哪怕中间胖出天际，出来的必须完美恢复为原来身段
    assert out.shape == (8, 20, in_dim)
