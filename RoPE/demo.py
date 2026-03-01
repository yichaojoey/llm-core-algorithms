"""
RoPE 验证演示脚本
================

这个脚本通过几个具体的实验来验证 RoPE 实现的正确性，
并帮助你直观理解 RoPE 的工作原理。

运行方式:
    python demo.py
"""

import torch
from rope import precompute_freqs_cis, apply_rotary_emb, RoPE, rope_rotation_matrix


def demo_1_basic_usage():
    """
    演示 1：RoPE 的基本使用流程
    
    展示如何在一个简单的 self-attention 中使用 RoPE：
    1. 生成 query 和 key
    2. 对它们分别应用 RoPE
    3. 计算 attention scores
    """
    print("=" * 60)
    print("演示 1：RoPE 基本使用流程")
    print("=" * 60)

    # 超参数设定
    batch_size = 1       # 批次大小
    seq_len = 8          # 序列长度（比如8个token）
    n_heads = 2          # attention head 数量
    head_dim = 16        # 每个 head 的维度（必须是偶数）

    # 步骤 1: 创建 RoPE 模块
    rope = RoPE(head_dim=head_dim, max_seq_len=128)
    print(f"\n✅ 创建 RoPE 模块: head_dim={head_dim}, max_seq_len=128")

    # 步骤 2: 模拟线性投影后的 query 和 key
    # 在真实 Transformer 中，这些是 W_q @ x 和 W_k @ x 的结果
    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, n_heads, head_dim)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim)
    print(f"   Query shape: {q.shape}  (batch, seq_len, n_heads, head_dim)")
    print(f"   Key   shape: {k.shape}")

    # 步骤 3: 对 query 和 key 应用 RoPE
    q_rotated = rope(q)
    k_rotated = rope(k)
    print(f"\n✅ 应用 RoPE 后:")
    print(f"   Q_rotated shape: {q_rotated.shape}  (形状不变！)")
    print(f"   K_rotated shape: {k_rotated.shape}")

    # 步骤 4: 计算 attention scores
    # 在 self-attention 中: score = Q @ K^T / sqrt(d)
    # 注意要先转置 head 和 seq_len 维度: (batch, n_heads, seq_len, head_dim)
    q_t = q_rotated.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
    k_t = k_rotated.transpose(1, 2)
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / (head_dim ** 0.5)
    print(f"\n✅ Attention Scores shape: {scores.shape}  (batch, n_heads, seq_len, seq_len)")
    print(f"   Scores (head 0, 前4个位置):\n{scores[0, 0, :4, :4].detach()}")


def demo_2_norm_preservation():
    """
    演示 2: 验证 RoPE 不改变向量的模长（L2 范数）
    
    RoPE 的关键特性：旋转操作是正交变换，只改变方向，不改变大小。
    这意味着应用 RoPE 前后，向量的 L2 范数应该保持不变。
    
    为什么这很重要？
    如果位置编码改变了向量的大小，会影响 attention 的缩放，
    导致训练不稳定。RoPE 避免了这个问题。
    """
    print("\n" + "=" * 60)
    print("演示 2：验证模长不变性（正交变换性质）")
    print("=" * 60)

    head_dim = 64
    rope = RoPE(head_dim=head_dim)

    # 创建测试向量
    torch.manual_seed(123)
    x = torch.randn(2, 10, 4, head_dim)

    # 计算 RoPE 前后的 L2 范数
    norm_before = torch.norm(x, dim=-1)
    x_rotated = rope(x)
    norm_after = torch.norm(x_rotated, dim=-1)

    # 比较差异（应该接近 0）
    max_diff = (norm_before - norm_after).abs().max().item()
    mean_diff = (norm_before - norm_after).abs().mean().item()

    print(f"\n   RoPE 前后 L2 范数最大差异: {max_diff:.2e}")
    print(f"   RoPE 前后 L2 范数平均差异: {mean_diff:.2e}")

    if max_diff < 1e-5:
        print("   ✅ 通过! 模长保持不变（差异在浮点误差范围内）")
    else:
        print("   ❌ 失败! 模长发生了不可接受的变化")


def demo_3_relative_position():
    """
    演示 3: 验证 RoPE 的核心特性 —— 相对位置不变性
    
    RoPE 最重要的理论保证：
        q_m^T · k_n 只依赖于 m - n（相对位置），而不是 m 和 n 的绝对值。
    
    验证方法：
        如果位置 (2, 5) 和 (7, 10) 的 q·k 值相同（因为 5-2 = 10-7 = 3），
        那就验证了相对位置不变性。
    
    为什么这很重要？
    - 序列中 "I love you" 不管出现在开头还是中间，相对关系应该一样
    - 这让模型能够更好地泛化到不同长度的序列
    """
    print("\n" + "=" * 60)
    print("演示 3：验证相对位置不变性")
    print("=" * 60)

    head_dim = 64

    # 关键：为保证 q_m^T * k_n 只取决于 m-n，我们需要用相同的 q 和 k 向量
    # 但放在不同的"绝对位置"上
    torch.manual_seed(999)
    q_vec = torch.randn(1, 1, 1, head_dim)  # 一个 query 向量
    k_vec = torch.randn(1, 1, 1, head_dim)  # 一个 key 向量

    freqs_cis = precompute_freqs_cis(head_dim, 128)

    # 测试多组 (pos_q, pos_k)，但保持 pos_q - pos_k = 3
    relative_dist = 3
    test_pairs = [(2, 5), (7, 10), (20, 23), (50, 53)]

    print(f"\n   相对距离固定为 {relative_dist}，测试不同绝对位置对:")
    print(f"   {'位置对':<15} {'q·k 内积':<15}")
    print(f"   {'-' * 30}")

    scores = []
    for pos_q, pos_k in test_pairs:
        # 单独对 q_vec 和 k_vec 应用对应位置的旋转
        # 构造只包含该位置的 freqs_cis
        q_at_pos = apply_rotary_emb(q_vec, freqs_cis[pos_q:pos_q + 1])
        k_at_pos = apply_rotary_emb(k_vec, freqs_cis[pos_k:pos_k + 1])

        # 内积: q^T · k
        dot = (q_at_pos * k_at_pos).sum().item()
        scores.append(dot)
        print(f"   ({pos_q:>3}, {pos_k:>3})       {dot:.6f}")

    # 验证所有 score 是否相同
    max_score_diff = max(scores) - min(scores)
    print(f"\n   所有内积的最大差异: {max_score_diff:.2e}")

    if max_score_diff < 1e-4:
        print("   ✅ 通过! 相对位置不变性成立")
    else:
        print("   ❌ 失败! 相对位置不变性不成立")


def demo_4_rotation_matrix():
    """
    演示 4：可视化旋转矩阵
    
    展示 RoPE 旋转矩阵的结构，帮助理解其分块对角的特点。
    同时验证：矩阵乘法和复数乘法给出相同结果。
    """
    print("\n" + "=" * 60)
    print("演示 4：旋转矩阵可视化与等价性验证")
    print("=" * 60)

    dim = 6  # 小维度，方便观察
    pos = 3  # 位置 3

    # 生成旋转矩阵
    R = rope_rotation_matrix(pos, dim)
    print(f"\n   位置 {pos} 的旋转矩阵 (dim={dim}):")
    # 格式化打印
    for i in range(dim):
        row = "   ["
        for j in range(dim):
            val = R[i, j].item()
            if abs(val) < 1e-10:
                row += "    0.00"
            else:
                row += f"  {val:6.3f}"
        row += "  ]"
        print(row)

    # 验证：矩阵乘法 vs 复数乘法（两种方式应该给出相同结果）
    torch.manual_seed(42)
    x = torch.randn(dim)

    # 方法 1: 矩阵乘法
    x_rotated_matrix = R @ x

    # 方法 2: 复数乘法（通过 apply_rotary_emb）
    x_for_rope = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,dim)
    freqs_cis = precompute_freqs_cis(dim, 128)
    x_rotated_complex = apply_rotary_emb(x_for_rope, freqs_cis[pos:pos + 1])
    x_rotated_complex = x_rotated_complex.squeeze()

    diff = (x_rotated_matrix - x_rotated_complex).abs().max().item()
    print(f"\n   矩阵乘法 vs 复数乘法的最大差异: {diff:.2e}")

    if diff < 1e-5:
        print("   ✅ 通过! 两种实现方式等价")
    else:
        print("   ❌ 失败! 两种实现方式不一致")


def demo_5_frequency_pattern():
    """
    演示 5：展示不同维度的频率模式
    
    直观展示 RoPE 中 "低维高频、高维低频" 的设计：
    - 低维度（靠前的维度对）：频率高，旋转快 → 捕捉近距离关系
    - 高维度（靠后的维度对）：频率低，旋转慢 → 捕捉远距离关系
    
    这种多尺度设计类似于傅里叶变换，让不同维度"关注"不同范围的距离。
    """
    print("\n" + "=" * 60)
    print("演示 5：频率模式 —— 不同维度的旋转速度")
    print("=" * 60)

    dim = 16
    max_seq_len = 100
    freqs_cis = precompute_freqs_cis(dim, max_seq_len)

    # 提取每个维度对在不同位置的旋转角度
    # freqs_cis 的相位角就是旋转角度
    angles = torch.angle(freqs_cis)  # (max_seq_len, dim//2)

    print(f"\n   维度: {dim}, 共 {dim // 2} 个维度对")
    print(f"\n   每个维度对在不同位置的旋转角度 (弧度):")
    print(f"   {'位置':>6}", end="")
    for i in range(dim // 2):
        print(f"   {'维度对'+str(i):>8}", end="")
    print()
    print(f"   {'-' * (6 + 11 * (dim // 2))}")

    for pos in [0, 1, 5, 10, 50]:
        print(f"   {pos:>6}", end="")
        for i in range(dim // 2):
            print(f"   {angles[pos, i].item():>8.4f}", end="")
        print()

    print(f"\n   📌 观察: 维度对 0 (最左列) 的角度增长最快 → 高频")
    print(f"   📌 观察: 维度对 {dim//2 - 1} (最右列) 的角度增长最慢 → 低频")
    print(f"   📌 这就是 RoPE 的多尺度位置编码!")


if __name__ == "__main__":
    print()
    print("🔄 RoPE (Rotary Position Embedding) 验证演示")
    print("=" * 60)

    demo_1_basic_usage()
    demo_2_norm_preservation()
    demo_3_relative_position()
    demo_4_rotation_matrix()
    demo_5_frequency_pattern()

    print("\n" + "=" * 60)
    print("🎉 所有演示完成！")
    print("=" * 60)
