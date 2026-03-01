# RoPE (Rotary Position Embedding) 从零实现指南

> 📖 本教程手把手教你如何从零实现 RoPE 算法。
> 每一步都有详细解释，适合面试准备和教学使用。

---

## 目录

1. [背景知识](#1-背景知识)
2. [理解数学原理](#2-理解数学原理)
3. [搭建代码框架](#3-搭建代码框架)
4. [实现频率预计算](#4-实现频率预计算)
5. [实现旋转嵌入](#5-实现旋转嵌入)
6. [封装为 Module](#6-封装为-module)
7. [编写验证 Demo](#7-编写验证-demo)
8. [面试要点总结](#8-面试要点总结)

---

## 1. 背景知识

### 什么是位置编码？

Transformer 的 self-attention 本身是**排列不变的（permutation invariant）**——打乱输入顺序，输出也只是做了同样的打乱。但语言是有顺序的（"狗咬人" ≠ "人咬狗"），所以需要额外注入位置信息。

### 常见位置编码方案对比

| 方案 | 方式 | 优缺点 |
|------|------|--------|
| **Sinusoidal PE** (Vaswani 2017) | 加法，加到 embedding 上 | 固定、不可学习、缺乏相对位置性 |
| **Learned PE** (GPT-2) | 加法，可学习的位置向量 | 可学、但有最大长度限制 |
| **ALiBi** | 在 attention score 上加偏置 | 简单高效、外推性好 |
| **RoPE** (Su et al. 2021) | **乘法**，旋转 q/k 向量 | ✅ 相对位置、✅ 外推性好、✅ 已成主流 |

### 为什么 RoPE 成为主流？

- **LLaMA 系列**（Meta）使用 RoPE
- **Qwen**（阿里）使用 RoPE
- **Mistral**、**GLM** 等都使用 RoPE
- **GPT-NeoX** 也采用 RoPE

原因：它同时具备**相对位置编码**的优雅性和**绝对位置编码**的计算效率。

---

## 2. 理解数学原理

> 💡 **核心思想一句话总结**：RoPE 把 query 和 key 的每对维度视为一个 2D 向量，根据 token 的位置旋转不同的角度。

### 2.1 从 2D 旋转说起

假设有一个 2D 向量 $[a, b]$，逆时针旋转角度 $\theta$：

$$
\begin{bmatrix} a' \\ b' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix}
$$

用复数来表示更简洁：$(a + jb) \times e^{j\theta} = (a + jb) \times (\cos\theta + j\sin\theta)$

### 2.2 RoPE 的做法

对于维度为 $d$ 的 query 向量，RoPE 把它拆成 $d/2$ 对：

$$
q = [q_0, q_1 \mid q_2, q_3 \mid \ldots \mid q_{d-2}, q_{d-1}]
$$

每一对 $(q_{2i}, q_{2i+1})$ 被旋转一个角度 $m \cdot \theta_i$，其中：
- $m$ = token 的位置索引（0, 1, 2, ...）
- $\theta_i = \frac{1}{10000^{2i/d}}$ = 第 $i$ 个维度对的基础频率

### 2.3 为什么这样能编码"相对位置"？

关键定理（不需要证明，但理解直觉）：

$$
\langle R_m q, R_n k \rangle = \langle R_{m-n} q, k \rangle
$$

旋转后的 q 和 k 的内积，只取决于它们的**位置差 $m-n$**，而不是绝对位置。

**直觉**：旋转是正交变换。两个向量之间的"相对旋转角度"决定了它们的内积，而相对旋转角度恰好是 $(m-n) \cdot \theta_i$。

### 2.4 多尺度频率的设计

$$
\theta_i = \frac{1}{10000^{2i/d}}, \quad i = 0, 1, \ldots, d/2 - 1
$$

- **$i = 0$（低维）**：$\theta_0 = 1$，旋转最快 → 对**近距离**关系敏感
- **$i = d/2-1$（高维）**：$\theta$ 非常小，旋转很慢 → 对**远距离**关系敏感

这类似于傅里叶变换的多频率分析！

---

## 3. 搭建代码框架

> 🎯 **面试策略**：先写出函数签名和注释，让面试官看到你的思路清晰。

```python
import torch
import torch.nn as nn

def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """预计算旋转频率（复数形式）"""
    pass  # 第4步实现

def apply_rotary_emb(x, freqs_cis):
    """将旋转位置编码应用到输入张量"""
    pass  # 第5步实现

class RoPE(nn.Module):
    """RoPE 的 Module 封装"""
    pass  # 第6步实现
```

**要点**：
- 只需要 `torch` 和 `torch.nn`，不需要其他依赖
- 三个组件各有明确职责：计算频率 → 应用旋转 → 模块封装
- 函数命名遵循 Meta LLaMA 的风格

---

## 4. 实现频率预计算

> 🎯 **这是整个实现的第一步**，也是面试中应该最先写的部分。

### 4.1 思路

需要为每个 (位置, 维度对) 计算旋转因子 $e^{j \cdot m \cdot \theta_i}$

### 4.2 实现步骤

```python
def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    # 步骤 1: 计算每个维度对的基础频率
    #   θ_i = 1 / (theta^{2i/dim})
    #   技巧: 用 arange(0, dim, 2) 生成 [0, 2, 4, ...]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # shape: (dim // 2,)
    
    # 步骤 2: 生成位置索引 [0, 1, ..., max_seq_len-1]
    positions = torch.arange(max_seq_len).float()
    # shape: (max_seq_len,)
    
    # 步骤 3: 外积得到所有 (位置, 维度对) 的旋转角度
    angles = torch.outer(positions, freqs)
    # shape: (max_seq_len, dim // 2)
    
    # 步骤 4: 转为复数旋转因子 e^{j·angle}
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    # shape: (max_seq_len, dim // 2), dtype=complex64
    
    return freqs_cis
```

### 4.3 面试口述要点

> "首先计算每个维度对的基础频率 theta_i，用公式 1/(10000^{2i/d})。
> 然后和位置索引做外积，得到所有位置×维度对的旋转角度。
> 最后用 `torch.polar` 转成单位复数。"

---

## 5. 实现旋转嵌入

> 🎯 **这是核心计算步骤**，面试时要能清晰解释维度变化。

### 5.1 思路

把实数的 q/k 向量中的相邻维度配对为复数，乘以旋转因子，再转回实数。

### 5.2 实现步骤

```python
def apply_rotary_emb(x, freqs_cis):
    # x shape: (batch, seq_len, n_heads, head_dim)
    
    # 步骤 1: 实数 → 复数
    #   (batch, seq_len, n_heads, head_dim)
    #   → reshape → (batch, seq_len, n_heads, head_dim//2, 2)
    #   → view_as_complex → (batch, seq_len, n_heads, head_dim//2) [complex]
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)
    )
    
    # 步骤 2: 对齐 freqs_cis 的 shape 以便广播
    #   freqs_cis: (seq_len, head_dim//2)
    #   → unsqueeze → (1, seq_len, 1, head_dim//2)
    seq_len = x.shape[1]
    freqs_cis = freqs_cis[:seq_len].unsqueeze(0).unsqueeze(2)
    
    # 步骤 3: 复数乘法 = 旋转！
    x_rotated = x_complex * freqs_cis
    
    # 步骤 4: 复数 → 实数
    #   view_as_real → (batch, seq_len, n_heads, head_dim//2, 2)
    #   flatten(-2) → (batch, seq_len, n_heads, head_dim)
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    
    return x_out.type_as(x)
```

### 5.3 维度变化一览表

```
输入:      (batch, seq_len, n_heads, head_dim)      实数
  ↓ reshape + view_as_complex
复数:      (batch, seq_len, n_heads, head_dim//2)    复数
  ↓ × freqs_cis (广播)
旋转后:    (batch, seq_len, n_heads, head_dim//2)    复数
  ↓ view_as_real + flatten
输出:      (batch, seq_len, n_heads, head_dim)      实数
```

### 5.4 面试口述要点

> "apply_rotary_emb 的核心是四步：
> 1. 把实数张量的最后一维两两配对转成复数
> 2. 对齐 freqs_cis 的 shape 并广播
> 3. 复数乘法实现旋转
> 4. 转回实数并恢复原始 shape"

---

## 6. 封装为 Module

> 🎯 面试中如果时间充裕可以写，展示工程能力。

```python
class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len=4096, theta=10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
    
    def forward(self, x):
        return apply_rotary_emb(x, self.freqs_cis)
```

**要点**：
- 用 `register_buffer` 而不是 `nn.Parameter`（不需要梯度）
- `persistent=False` 因为频率可以随时重新计算
- Module 封装让使用更简洁：`rope(q)` 而不是 `apply_rotary_emb(q, freqs)`

---

## 7. 编写验证 Demo

写完代码后，应该验证以下性质：

### 验证 1：模长不变性

```python
norm_before = torch.norm(x, dim=-1)
x_rotated = rope(x)
norm_after = torch.norm(x_rotated, dim=-1)
assert torch.allclose(norm_before, norm_after, atol=1e-5)
```

**原理**：旋转是正交变换，不改变向量长度。

### 验证 2：相对位置不变性

```python
# 固定 q 和 k，放在不同绝对位置但保持相同相对距离
# 验证 q^T·k 的值相同
```

**原理**：$\langle R_m q, R_n k \rangle$ 只取决于 $m - n$。

### 验证 3：旋转矩阵等价性

```python
# 矩阵乘法 R @ x 应该等于复数乘法 apply_rotary_emb(x)
```

**原理**：两种实现方式在数学上完全等价。

---

## 8. 面试要点总结

### 🔑 必须记住的核心公式

```
θ_i = 1 / (10000^{2i/d})

旋转因子 = e^{j·m·θ_i} = cos(m·θ_i) + j·sin(m·θ_i)

应用方式：(q_{2i} + j·q_{2i+1}) × 旋转因子
```

### 🔑 必须理解的关键 API

| API | 用途 |
|-----|------|
| `torch.arange(0, dim, 2)` | 生成维度对索引 |
| `torch.outer(a, b)` | 外积，计算角度矩阵 |
| `torch.polar(abs, angle)` | 生成复数旋转因子 |
| `torch.view_as_complex(x)` | 实数 → 复数 |
| `torch.view_as_real(x)` | 复数 → 实数 |
| `register_buffer` | 注册不可训练的缓存 |

### 🔑 面试中的常见追问

**Q: RoPE 和 Sinusoidal PE 有什么区别？**
> A: Sinusoidal PE 是加法（加到 embedding 上），RoPE 是乘法（旋转 q/k）。
> RoPE 天然编码相对位置，Sinusoidal PE 只编码绝对位置。

**Q: 为什么 RoPE 只作用于 q 和 k，不作用于 v？**
> A: 因为位置信息只需要影响 attention weight（由 q·k 决定），
> 不需要影响 value 的加权求和。

**Q: RoPE 的计算开销大吗？**
> A: 不大。旋转因子可以预计算，应用时只是逐元素乘法（复数乘法）。

**Q: `theta = 10000` 的含义？**
> A: theta 控制频率衰减的速度。theta 越大，高频和低频之间的跨度越大，
> 模型能区分的最大距离越远。

**Q: 如何扩展到更长的序列？（长上下文）**
> A: 常见方法如 NTK-aware scaling、YaRN 等，核心思想是调整 theta 的值
> 或对频率做缩放插值。

### 🔑 面试手写代码的推荐顺序

```
1. 写函数签名和注释          (30秒)
2. 实现 precompute_freqs_cis  (2分钟)
3. 实现 apply_rotary_emb      (3分钟)
4. 封装 RoPE Module           (1分钟)
5. 口述验证方案               (1分钟)
```

总计约 **7-8 分钟**，是一个合理的面试时间分配。

---

## 运行方式

```bash
# 确保安装了 PyTorch
pip install torch

# 运行验证演示
python demo.py
```

---

## 文件结构

```
rope/
├── README.md     ← 你正在读的教程（分步骤教学）
├── rope.py       ← 核心实现（带详细中文注释）
└── demo.py       ← 验证演示（5个验证实验）
```

---

## 参考资料

- [RoPE 原始论文: RoFormer (Su et al., 2021)](https://arxiv.org/abs/2104.09864)
- [LLaMA 源码中的 RoPE 实现](https://github.com/facebookresearch/llama)
- [Eleuther AI 的 RoPE 讲解](https://blog.eleutherai.com/rotary-embeddings/)
