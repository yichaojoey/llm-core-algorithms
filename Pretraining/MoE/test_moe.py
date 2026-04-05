import torch
from moe import SparseMoE

def test_moe_forward_dimensions():
    """测试当词切片派发向全宇宙不同的专家兜里打碎执行工作后，合并还原后的维度有没有一丝损伤或泄露？"""
    batch_size = 3
    seq_len = 16
    d_model = 64
    
    # 构建包含 8 个专家的大型流水线
    moe = SparseMoE(d_model=d_model, num_experts=8, top_k=2)
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    out, aux_loss = moe(x)
    
    # 分分合合兜兜转转派出去运算，它拼回来作为 FFN 大网络替代，它的体形决不可以改变一丝毫。
    assert out.shape == (batch_size, seq_len, d_model)
    # 计算出的惩罚不为空且不为0 (除非我们强行做了完美无瑕切分)
    assert aux_loss.item() > 0.0

def test_moe_load_balancing_ideal():
    """测试完全均匀散布数据流量的时候那条数学公式是否恰如其分为 1.0 的平坦稳定基调"""
    d_model = 10
    num_experts = 4
    moe = SparseMoE(d_model=d_model, num_experts=num_experts, top_k=1)
    
    # 自己捏造完美全 0 且带极其微弱的抖动扰动
    # 让所有门控分数概率无限逼近 1/4 (0.25) 且客流量分布完美分属 4 家
    uniform_probs = torch.full((12, num_experts), 1.0 / num_experts)
    
    # 为了触发完全硬切分客流 f_i 完美相等，必须手动赋值，不测 forward
    # 这个是在理想世界中：任何人的 mean_prob=0.25，客单接待比率 fraction=0.25
    # L = N * Sum(P_i * F_i) = 4 * (4 * (0.25 * 0.25)) = 4 * (4 * 0.0625) = 4 * 0.25 = 1.0
    
    # 用它的数学引擎跑一遍
    loss = moe._compute_load_balancing_loss(uniform_probs)
    
    # 极为苛刻的理论数学打分锁定：
    # 如果平摊均沾不堕落，那惩罚线就是精准定海神针 1.0 ！
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-4)
