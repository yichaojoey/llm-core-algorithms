"""
RoPE 单元测试
=============

本文件包含 RoPE 实现的完整单元测试。
测试覆盖：输出 shape、数值正确性、数学性质、边界情况、Module 行为。

运行方式:
    pytest test_rope.py -v
    
    或者不依赖 pytest:
    python test_rope.py
"""

import math
import torch
import unittest
from rope import precompute_freqs_cis, apply_rotary_emb, RoPE, rope_rotation_matrix


# ============================================================================
# 测试组 1: precompute_freqs_cis 的测试
# ============================================================================

class TestPrecomputeFreqsCis(unittest.TestCase):
    """测试旋转频率预计算函数。"""

    # ----- 1.1 输出 shape 测试 -----

    def test_output_shape(self):
        """输出的 shape 应该是 (max_seq_len, dim // 2)。
        这是最基本的检查——确保函数对 dim 做了正确的二分。
        """
        dim, max_seq_len = 64, 128
        freqs = precompute_freqs_cis(dim, max_seq_len)
        self.assertEqual(freqs.shape, (max_seq_len, dim // 2))

    def test_output_shape_various_dims(self):
        """测试多种 dim 和 seq_len 组合，确保 shape 公式一致。"""
        for dim in [8, 16, 32, 128]:
            for seq_len in [1, 10, 256]:
                freqs = precompute_freqs_cis(dim, seq_len)
                self.assertEqual(
                    freqs.shape, (seq_len, dim // 2),
                    f"dim={dim}, seq_len={seq_len} 时 shape 不正确"
                )

    # ----- 1.2 数据类型测试 -----

    def test_output_dtype_is_complex(self):
        """输出必须是复数类型，因为旋转因子 e^{j*theta} 是复数。"""
        freqs = precompute_freqs_cis(16, 32)
        self.assertTrue(freqs.is_complex(), "输出应该是复数类型")

    # ----- 1.3 数值正确性测试 -----

    def test_position_zero_is_identity(self):
        """位置 0 的旋转因子应该全是 1+0j（不旋转）。
        
        原因: 角度 = 0 * theta_i = 0，所以 e^{j*0} = 1。
        这意味着位置 0 的 token 不会被旋转，符合直觉。
        """
        freqs = precompute_freqs_cis(32, 10)
        # 位置 0 的所有复数应该都是 1+0j
        expected = torch.ones(32 // 2, dtype=torch.complex64)
        torch.testing.assert_close(freqs[0], expected, atol=1e-6, rtol=1e-6)

    def test_unit_magnitude(self):
        """所有旋转因子的模长应该为 1（单位复数）。
        
        原因: 旋转因子 = e^{j*angle}，其模长 |e^{j*angle}| = 1。
        如果模长不为 1，就不是纯旋转，而是旋转 + 缩放了。
        """
        freqs = precompute_freqs_cis(64, 256)
        magnitudes = freqs.abs()  # 计算每个复数的模
        torch.testing.assert_close(
            magnitudes,
            torch.ones_like(magnitudes),
            atol=1e-6, rtol=1e-6,
        )

    def test_first_freq_value(self):
        """手动验证第一个维度对 (i=0) 的频率值。
        
        i=0 时: theta_0 = 1 / (10000^{0/d}) = 1.0
        位置 m 的角度 = m * 1.0 = m
        所以 freqs_cis[m][0] = cos(m) + j*sin(m)
        """
        dim = 16
        freqs = precompute_freqs_cis(dim, 10)
        for m in [1, 3, 7]:
            expected = torch.complex(
                torch.tensor(math.cos(m)),
                torch.tensor(math.sin(m)),
            )
            torch.testing.assert_close(
                freqs[m, 0], expected, atol=1e-5, rtol=1e-5,
            )

    def test_last_freq_is_slowest(self):
        """最后一个维度对的频率应该最小（旋转最慢）。
        
        theta_{d/2-1} = 1 / (10000^{(d-2)/d}) ≈ 很小的数
        所以相邻位置之间的角度变化应该很小。
        """
        dim = 64
        freqs = precompute_freqs_cis(dim, 100)
        # 最后一个维度对在位置1处的角度变化
        last_angle = torch.angle(freqs[1, -1]).item()
        # 第一个维度对在位置1处的角度变化
        first_angle = torch.angle(freqs[1, 0]).item()
        # 最后一个频率应该远小于第一个
        self.assertLess(abs(last_angle), abs(first_angle))

    def test_custom_theta(self):
        """自定义 theta 应该改变频率分布。
        
        theta 越大 → 高频和低频之间的差距越大。
        """
        dim = 16
        freqs_default = precompute_freqs_cis(dim, 10, theta=10000.0)
        freqs_small = precompute_freqs_cis(dim, 10, theta=100.0)
        # 两者应该不同
        self.assertFalse(torch.allclose(freqs_default, freqs_small))


# ============================================================================
# 测试组 2: apply_rotary_emb 的测试
# ============================================================================

class TestApplyRotaryEmb(unittest.TestCase):
    """测试旋转嵌入应用函数。"""

    def setUp(self):
        """每个测试前的准备工作。"""
        self.head_dim = 32
        self.max_seq_len = 128
        self.freqs_cis = precompute_freqs_cis(self.head_dim, self.max_seq_len)
        torch.manual_seed(42)

    # ----- 2.1 输出 shape 测试 -----

    def test_output_shape_unchanged(self):
        """输出 shape 应该和输入完全一致（旋转不改变维度）。"""
        x = torch.randn(2, 16, 4, self.head_dim)
        out = apply_rotary_emb(x, self.freqs_cis)
        self.assertEqual(out.shape, x.shape)

    def test_output_shape_various_configs(self):
        """测试不同 batch、seq_len、n_heads 组合。"""
        for batch in [1, 4]:
            for seq_len in [1, 8, 32]:
                for n_heads in [1, 8]:
                    x = torch.randn(batch, seq_len, n_heads, self.head_dim)
                    out = apply_rotary_emb(x, self.freqs_cis)
                    self.assertEqual(
                        out.shape, x.shape,
                        f"batch={batch}, seq_len={seq_len}, n_heads={n_heads}"
                    )

    # ----- 2.2 数据类型测试 -----

    def test_output_dtype_matches_input(self):
        """输出的 dtype 应该和输入一致。
        
        即使内部计算用了 float32 和 complex64，最终输出应该
        转回与输入相同的 dtype。
        """
        for dtype in [torch.float32, torch.float16]:
            x = torch.randn(1, 4, 2, self.head_dim, dtype=dtype)
            out = apply_rotary_emb(x, self.freqs_cis)
            self.assertEqual(out.dtype, dtype, f"dtype={dtype} 时输出类型不匹配")

    # ----- 2.3 核心数学性质: 模长不变性 -----

    def test_norm_preservation(self):
        """RoPE 前后向量的 L2 范数应该不变（旋转是正交变换）。
        
        这是 RoPE 最重要的数学性质之一。
        如果范数改变了，说明不是纯旋转，会影响 attention 的缩放。
        """
        x = torch.randn(4, 32, 8, self.head_dim)
        x_rotated = apply_rotary_emb(x, self.freqs_cis)

        norm_before = torch.norm(x, dim=-1)
        norm_after = torch.norm(x_rotated, dim=-1)

        torch.testing.assert_close(norm_before, norm_after, atol=1e-5, rtol=1e-5)

    def test_norm_preservation_single_vector(self):
        """对单个 token 的单个 head 也应该保持模长。"""
        x = torch.randn(1, 1, 1, self.head_dim)
        x_rotated = apply_rotary_emb(x, self.freqs_cis)
        self.assertAlmostEqual(
            x.norm().item(), x_rotated.norm().item(), places=5,
        )

    # ----- 2.4 核心数学性质: 相对位置不变性 -----

    def test_relative_position_invariance(self):
        """相同相对位置的 q·k 内积应该相同。
        
        这是 RoPE 最核心的理论保证:
            <R_m * q, R_n * k> 只取决于 m - n
        
        验证方法:
            固定同一组 q 和 k 向量，分别放在不同绝对位置，
            但保持相对距离不变，检查内积是否一致。
        """
        torch.manual_seed(123)
        head_dim = 64
        q_vec = torch.randn(1, 1, 1, head_dim)
        k_vec = torch.randn(1, 1, 1, head_dim)
        freqs = precompute_freqs_cis(head_dim, 256)

        relative_dist = 5
        # 多组绝对位置，相对距离都是 5
        test_pairs = [(0, 5), (10, 15), (50, 55), (100, 105)]

        dots = []
        for pos_q, pos_k in test_pairs:
            q_rot = apply_rotary_emb(q_vec, freqs[pos_q:pos_q + 1])
            k_rot = apply_rotary_emb(k_vec, freqs[pos_k:pos_k + 1])
            dot = (q_rot * k_rot).sum().item()
            dots.append(dot)

        # 所有内积应该相等（误差在浮点精度内）
        for i in range(1, len(dots)):
            self.assertAlmostEqual(
                dots[0], dots[i], places=4,
                msg=f"位置对 {test_pairs[0]} 和 {test_pairs[i]} 的内积不同"
            )

    def test_different_relative_positions_give_different_dots(self):
        """不同相对距离的 q·k 内积一般应该不同。
        
        这验证了 RoPE 确实在区分不同的相对位置，
        而不是退化成一个不起作用的恒等变换。
        """
        torch.manual_seed(456)
        head_dim = 64
        q_vec = torch.randn(1, 1, 1, head_dim)
        k_vec = torch.randn(1, 1, 1, head_dim)
        freqs = precompute_freqs_cis(head_dim, 256)

        # 相对距离 1 vs 相对距离 50
        q_rot_1 = apply_rotary_emb(q_vec, freqs[0:1])
        k_rot_1 = apply_rotary_emb(k_vec, freqs[1:2])
        dot_dist1 = (q_rot_1 * k_rot_1).sum().item()

        q_rot_50 = apply_rotary_emb(q_vec, freqs[0:1])
        k_rot_50 = apply_rotary_emb(k_vec, freqs[50:51])
        dot_dist50 = (q_rot_50 * k_rot_50).sum().item()

        self.assertNotAlmostEqual(
            dot_dist1, dot_dist50, places=1,
            msg="不同相对距离的内积不应该相同"
        )

    # ----- 2.5 位置 0 的特殊行为 -----

    def test_position_zero_is_identity(self):
        """位置 0 的旋转应该是恒等变换（输入不变）。
        
        原因: freqs_cis[0] = e^{j*0} = 1+0j，乘以 1 就是不变。
        """
        x = torch.randn(1, 1, 2, self.head_dim)
        x_rotated = apply_rotary_emb(x, self.freqs_cis)
        torch.testing.assert_close(x, x_rotated, atol=1e-6, rtol=1e-6)

    # ----- 2.6 不同 head 独立旋转 -----

    def test_heads_rotated_identically(self):
        """不同 head 的相同位置应该使用相同的旋转角度。
        
        RoPE 的旋转只取决于位置和维度，不取决于 head 编号。
        所以如果两个 head 有完全相同的输入，旋转后也应该相同。
        """
        x_single = torch.randn(1, 8, 1, self.head_dim)
        # 复制到 4 个 head
        x_multi = x_single.expand(1, 8, 4, self.head_dim).contiguous()

        out = apply_rotary_emb(x_multi, self.freqs_cis)
        # 每个 head 的输出应该完全一样
        for h in range(1, 4):
            torch.testing.assert_close(out[:, :, 0, :], out[:, :, h, :])


# ============================================================================
# 测试组 3: RoPE Module 的测试
# ============================================================================

class TestRoPEModule(unittest.TestCase):
    """测试 RoPE 的 nn.Module 封装。"""

    # ----- 3.1 初始化测试 -----

    def test_init_even_dim(self):
        """偶数 head_dim 应该正常初始化。"""
        rope = RoPE(head_dim=64)
        self.assertIsNotNone(rope.freqs_cis)

    def test_init_odd_dim_raises(self):
        """奇数 head_dim 应该抛出 AssertionError。
        
        因为 RoPE 将维度两两配对，奇数维度无法配对。
        """
        with self.assertRaises(AssertionError):
            RoPE(head_dim=63)

    # ----- 3.2 Module 功能测试 -----

    def test_forward_output_shape(self):
        """Module 的前向传播应该保持 shape 不变。"""
        rope = RoPE(head_dim=32, max_seq_len=128)
        x = torch.randn(2, 16, 4, 32)
        out = rope(x)
        self.assertEqual(out.shape, x.shape)

    def test_module_matches_functional(self):
        """Module 的输出应该和直接调用函数的输出完全一致。
        
        验证封装没有引入任何差异。
        """
        head_dim = 32
        rope = RoPE(head_dim=head_dim, max_seq_len=64)
        freqs = precompute_freqs_cis(head_dim, 64)

        x = torch.randn(1, 8, 2, head_dim)
        out_module = rope(x)
        out_func = apply_rotary_emb(x, freqs)
        torch.testing.assert_close(out_module, out_func)

    # ----- 3.3 Buffer 行为测试 -----

    def test_freqs_not_in_parameters(self):
        """freqs_cis 不应该出现在可训练参数中。
        
        RoPE 的旋转频率是固定的，不需要梯度更新。
        如果它被当成 Parameter，会浪费显存并引入不必要的优化。
        """
        rope = RoPE(head_dim=64)
        param_names = [name for name, _ in rope.named_parameters()]
        self.assertNotIn("freqs_cis", param_names)

    def test_freqs_is_buffer(self):
        """freqs_cis 应该注册为 buffer（非持久化）。"""
        rope = RoPE(head_dim=64)
        buffer_names = [name for name, _ in rope.named_buffers()]
        self.assertIn("freqs_cis", buffer_names)

    def test_no_trainable_parameters(self):
        """RoPE 模块不应该有任何可训练参数。
        
        RoPE 只做确定性的旋转操作，没有需要学习的权重。
        """
        rope = RoPE(head_dim=64)
        n_params = sum(p.numel() for p in rope.parameters())
        self.assertEqual(n_params, 0, "RoPE 不应该有可训练参数")

    def test_eval_mode_same_as_train(self):
        """eval 模式和 train 模式的输出应该完全一致。
        
        RoPE 没有 dropout 或 batch normalization，
        所以训练和推理行为应该完全一样。
        """
        rope = RoPE(head_dim=32)
        x = torch.randn(1, 8, 2, 32)

        rope.train()
        out_train = rope(x)

        rope.eval()
        out_eval = rope(x)

        torch.testing.assert_close(out_train, out_eval)


# ============================================================================
# 测试组 4: rope_rotation_matrix 的测试
# ============================================================================

class TestRotationMatrix(unittest.TestCase):
    """测试旋转矩阵生成函数，并验证它和复数实现的等价性。"""

    # ----- 4.1 矩阵性质测试 -----

    def test_output_shape(self):
        """旋转矩阵应该是 (dim, dim) 的方阵。"""
        R = rope_rotation_matrix(pos=3, dim=8)
        self.assertEqual(R.shape, (8, 8))

    def test_orthogonality(self):
        """旋转矩阵应该是正交矩阵: R^T R = I。
        
        正交矩阵的特征：
        - 行向量两两正交且模为 1
        - 列向量两两正交且模为 1
        - 逆矩阵等于转置
        """
        R = rope_rotation_matrix(pos=5, dim=16)
        I = torch.eye(16)
        product = R.T @ R
        torch.testing.assert_close(product, I, atol=1e-6, rtol=1e-6)

    def test_determinant_is_one(self):
        """旋转矩阵的行列式应该为 +1。
        
        行列式为 +1 表示是"纯旋转"（保持手性）。
        行列式为 -1 则是反射，不是我们想要的。
        """
        R = rope_rotation_matrix(pos=7, dim=8)
        det = torch.det(R).item()
        self.assertAlmostEqual(det, 1.0, places=5)

    def test_position_zero_is_identity(self):
        """位置 0 的旋转矩阵应该是单位矩阵（不旋转）。"""
        R = rope_rotation_matrix(pos=0, dim=8)
        I = torch.eye(8)
        torch.testing.assert_close(R, I, atol=1e-6, rtol=1e-6)

    def test_block_diagonal_structure(self):
        """旋转矩阵应该是分块对角的：只有 2x2 块内有非零值。
        
        块 (2i, 2i+1) × (2i, 2i+1) 是一个 2D 旋转矩阵，
        其他位置全是 0。
        """
        dim = 8
        R = rope_rotation_matrix(pos=3, dim=dim)
        for i in range(dim):
            for j in range(dim):
                block_i, block_j = i // 2, j // 2
                if block_i != block_j:
                    self.assertAlmostEqual(
                        R[i, j].item(), 0.0, places=6,
                        msg=f"R[{i},{j}] 应该为 0（不同块之间）"
                    )

    # ----- 4.2 和复数实现的等价性 -----

    def test_matrix_vs_complex_equivalence(self):
        """矩阵乘法 R @ x 应该等于复数乘法 apply_rotary_emb(x)。
        
        这验证了两种实现在数学上完全等价。
        矩阵方法直观但 O(d^2)，复数方法高效 O(d)。
        """
        dim = 16
        freqs = precompute_freqs_cis(dim, 128)

        for pos in [0, 1, 5, 20]:
            R = rope_rotation_matrix(pos, dim)
            x = torch.randn(dim)

            # 方法 1: 矩阵乘法
            result_matrix = R @ x

            # 方法 2: 复数乘法
            x_input = x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            result_complex = apply_rotary_emb(
                x_input, freqs[pos:pos + 1]
            ).squeeze()

            torch.testing.assert_close(
                result_matrix, result_complex, atol=1e-5, rtol=1e-5,
            )


# ============================================================================
# 测试组 5: 边界情况和鲁棒性测试
# ============================================================================

class TestEdgeCases(unittest.TestCase):
    """测试边界情况，确保实现的鲁棒性。"""

    def test_seq_len_one(self):
        """序列长度为 1（单个 token）应该正常工作。
        
        场景: 自回归生成时，每次只输入一个新 token。
        """
        rope = RoPE(head_dim=32, max_seq_len=64)
        x = torch.randn(1, 1, 4, 32)
        out = rope(x)
        self.assertEqual(out.shape, (1, 1, 4, 32))

    def test_batch_size_one(self):
        """batch_size 为 1 应该正常工作。"""
        rope = RoPE(head_dim=32)
        x = torch.randn(1, 16, 4, 32)
        out = rope(x)
        self.assertEqual(out.shape, x.shape)

    def test_single_head(self):
        """只有 1 个 attention head 也应该正常工作。"""
        rope = RoPE(head_dim=32)
        x = torch.randn(2, 8, 1, 32)
        out = rope(x)
        self.assertEqual(out.shape, x.shape)

    def test_min_head_dim(self):
        """最小 head_dim（2）应该正常工作（只有一个维度对）。"""
        rope = RoPE(head_dim=2, max_seq_len=64)
        x = torch.randn(1, 8, 1, 2)
        out = rope(x)
        self.assertEqual(out.shape, x.shape)

    def test_large_head_dim(self):
        """大 head_dim（128，LLaMA 使用的值）应该正常工作。"""
        rope = RoPE(head_dim=128, max_seq_len=64)
        x = torch.randn(1, 8, 1, 128)
        out = rope(x)
        self.assertEqual(out.shape, x.shape)

    def test_zero_input(self):
        """全零输入旋转后应该还是全零。
        
        原因: 旋转零向量还是零向量（R @ 0 = 0）。
        """
        rope = RoPE(head_dim=32)
        x = torch.zeros(1, 8, 2, 32)
        out = rope(x)
        torch.testing.assert_close(out, x)

    def test_deterministic(self):
        """对相同输入，两次调用应该给出完全相同的输出。
        
        RoPE 是确定性操作（没有随机性），重复调用结果必须一致。
        """
        rope = RoPE(head_dim=32)
        x = torch.randn(2, 8, 4, 32)
        out1 = rope(x)
        out2 = rope(x)
        torch.testing.assert_close(out1, out2)

    def test_different_positions_give_different_outputs(self):
        """相同的向量放在不同位置，旋转后应该不同。
        
        这确保 RoPE 确实在注入位置信息。
        如果不同位置给出相同输出，说明位置编码失效了。
        """
        head_dim = 32
        freqs = precompute_freqs_cis(head_dim, 128)
        x = torch.randn(1, 1, 1, head_dim)

        out_pos0 = apply_rotary_emb(x, freqs[0:1])
        out_pos10 = apply_rotary_emb(x, freqs[10:11])

        self.assertFalse(
            torch.allclose(out_pos0, out_pos10, atol=1e-6),
            "不同位置的旋转结果不应该相同"
        )


# ============================================================================
# 测试组 6: 梯度测试
# ============================================================================

class TestGradients(unittest.TestCase):
    """测试 RoPE 对梯度传播的影响。"""

    def test_gradient_flows_through(self):
        """梯度应该能够通过 RoPE 正常反向传播。
        
        在训练中，RoPE 需要让梯度流回 q/k 的线性投影层。
        如果梯度被阻断了，Transformer 就无法训练。
        """
        rope = RoPE(head_dim=32)
        x = torch.randn(1, 8, 2, 32, requires_grad=True)
        out = rope(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad, "梯度应该能传回输入")
        self.assertFalse(
            torch.all(x.grad == 0),
            "梯度不应该全为 0"
        )

    def test_gradient_shape_matches_input(self):
        """梯度的 shape 应该和输入完全一致。"""
        rope = RoPE(head_dim=64)
        x = torch.randn(2, 16, 4, 64, requires_grad=True)
        out = rope(x)
        out.sum().backward()
        self.assertEqual(x.grad.shape, x.shape)


# ============================================================================
# 运行入口
# ============================================================================

if __name__ == "__main__":
    # -v 显示每个测试的名字和结果
    unittest.main(verbosity=2)
