import torch
from gqa import GroupedQueryAttention

def test_gqa_repeat_logic_alignment():
    """测试 GQA 核心独家本领：用 Unsqueeze + Expand 虚拟拉皮复制 KV 缓存块的技术是否真能和 MHA 的 Q 对齐形状"""
    B = 2
    T = 6
    embed_dim = 16
    
    q_heads = 8
    # 给定极其苛刻的单一 kv head （这就等效于传说中的降维 MQA：Multi-Query Attention）
    kv_heads = 1 
    
    gqa_model = GroupedQueryAttention(embed_dim, q_heads, kv_heads)
    
    # 模拟出来的一个独苗 KV 特征块
    head_dim = embed_dim // q_heads
    mock_kv = torch.randn(B, kv_heads, T, head_dim)
    
    # 呼叫分裂复制法术！
    expanded_kv = gqa_model._repeat_kv(mock_kv, n_rep=8)
    
    # 检查法术成果：必须强行把那 1 个头打散膨胀为符合 Q_heads 的 8 个数量级以保证对碰
    assert expanded_kv.shape == (B, 8, T, head_dim)
    
    # 内部数据检查，确保它不是把不同的 T 时间线搞乱，而是单单只复制了纯净的一模一样的头维度
    # 判断它膨胀出来的 8 个头的数据全都是跟当年那唯一的 1 个头长得一模一样的死板克隆体
    assert torch.allclose(expanded_kv[:, 0, :, :], mock_kv[:, 0, :, :])
    assert torch.allclose(expanded_kv[:, 7, :, :], mock_kv[:, 0, :, :])
