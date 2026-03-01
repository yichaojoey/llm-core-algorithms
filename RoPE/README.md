# RoPE (Rotary Position Embedding) 从零实现指南

> 📖 本教程按照**面试手写代码的顺序**组织，每一步都是 **理论 → 代码**。
> 跟着走一遍，你就能在面试中 7-8 分钟内完整写出 RoPE。

---

## 目录

- [背景：为什么需要 RoPE](#背景为什么需要-rope)
- [Step 1：写 import 和函数签名（30秒）](#step-1写-import-和函数签名30秒)
- [Step 2：实现 precompute_freqs_cis（2分钟）](#step-2实现-precompute_freqs_cis2分钟)
- [Step 3：实现 apply_rotary_emb（3分钟）](#step-3实现-apply_rotary_emb3分钟)
- [Step 4：封装 RoPE Module（1分钟）](#step-4封装-rope-module1分钟)
- [Step 5：口述验证方案（1分钟）](#step-5口述验证方案1分钟)
- [面试常见追问](#面试常见追问)
- [运行方式](#运行方式)

---

## 背景：为什么需要 RoPE

Transformer 的 self-attention 本身是**排列不变的**——打乱输入顺序，输出也只是做了同样的打乱。但语言是有顺序的（"狗咬人" ≠ "人咬狗"），所以需要额外注入位置信息。

| 方案 | 方式 | 优缺点 |
|------|------|--------|
| **Sinusoidal PE** (Vaswani 2017) | 加法，加到 embedding 上 | 固定、不可学习、缺乏相对位置性 |
| **Learned PE** (GPT-2) | 加法，可学习的位置向量 | 可学、但有最大长度限制 |
| **RoPE** (Su et al. 2021) | **乘法**，旋转 q/k 向量 | ✅ 相对位置、✅ 外推性好、✅ 已成主流 |

**LLaMA、Qwen、Mistral、GPT-NeoX** 都用 RoPE。一句话总结它的核心思想：

> **把 query 和 key 的每对维度视为一个 2D 向量，根据 token 的位置旋转不同的角度。**

---

## Step 1：写 import 和函数签名（30秒）

### 对应理论：确定 RoPE 的三个组件

RoPE 的实现只需要三个部分：
1. **预计算旋转频率** → 给每个 (位置, 维度对) 算好旋转角度
2. **应用旋转** → 把 q/k 向量旋转对应的角度
3. **Module 封装** → 工程化，方便集成到 Transformer 中

### 对应代码：写出骨架

```python
import torch
import torch.nn as nn

def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """预计算旋转频率（复数形式）"""
    ...

def apply_rotary_emb(x, freqs_cis):
    """将旋转位置编码应用到输入张量"""
    ...

class RoPE(nn.Module):
    """RoPE 的 Module 封装"""
    ...
```

> 🎯 **面试技巧**：先写出这个骨架，让面试官一眼看到你的思路清晰。
> 只需要 `torch` 和 `torch.nn`，不需要其他依赖。

---

## Step 2：实现 precompute_freqs_cis（2分钟）

### 对应理论：多尺度频率 + 单位复数

RoPE 为每个维度对分配一个"基础频率"：

```
theta_i = 1 / (10000^(2i/d)),   i = 0, 1, ..., d/2 - 1
```

- `d` = head_dim（每个 attention head 的维度）
- `i` = 维度对的索引

**频率的直觉**：
- `i = 0`（低维）→ `theta_0 = 1`，旋转最快 → 对**近距离**敏感
- `i = d/2-1`（高维）→ `theta` 非常小，旋转很慢 → 对**远距离**敏感
- 类似傅里叶变换的多频率分析

位置 `m` 在第 `i` 个维度对上的旋转因子用**单位复数**表示：

```
e^(j * m * theta_i) = cos(m * theta_i) + j * sin(m * theta_i)
```

这个复数的模为 1（只旋转，不缩放），辐角为 `m * theta_i`。

### 对应代码：4 行核心逻辑

```python
def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    # (1) 每个维度对的基础频率: theta_i = 1 / (10000^(2i/d))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # shape: (dim // 2,)
    
    # (2) 位置索引: [0, 1, 2, ..., max_seq_len-1]
    positions = torch.arange(max_seq_len).float()
    # shape: (max_seq_len,)
    
    # (3) 外积 → 每个(位置, 维度对)的旋转角度
    angles = torch.outer(positions, freqs)
    # shape: (max_seq_len, dim // 2)
    
    # (4) 转为单位复数旋转因子: e^(j*angle)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    # shape: (max_seq_len, dim // 2), dtype=complex64
    
    return freqs_cis
```

### 面试口述

> "先算每个维度对的基础频率 `1/(10000^(2i/d))`，
> 然后和位置索引做外积得到角度矩阵，
> 最后用 `torch.polar` 转成单位复数。"

---

## Step 3：实现 apply_rotary_emb（3分钟）

### 对应理论：2D 旋转 = 复数乘法

RoPE 把 q/k 向量的相邻维度两两配对，每对视为一个 2D 向量：

```
q = [q_0, q_1 | q_2, q_3 | ... | q_{d-2}, q_{d-1}]
     ------       ------          ----------------
     第0对        第1对            第 d/2-1 对
```

对每一对做 2D 旋转，角度为 `m * theta_i`：

```
[q'_{2i}  ]   [ cos(m*theta_i)  -sin(m*theta_i) ] [q_{2i}  ]
[q'_{2i+1}] = [ sin(m*theta_i)   cos(m*theta_i) ] [q_{2i+1}]
```

**关键洞察**：这个矩阵乘法等价于复数乘法！

```
(q_{2i} + j*q_{2i+1}) × (cos(m*theta_i) + j*sin(m*theta_i))
```

所以实现策略是：实数 → 复数 → 乘旋转因子 → 转回实数。

**为什么这样能编码"相对位置"？**

```
<R_m * q, R_n * k> = <R_{m-n} * q, k>
```

旋转后的 q·k 内积只取决于**位置差 `m-n`**，不取决于绝对位置。
直觉：旋转是正交变换，两个向量的"相对旋转角度"决定了内积。

### 对应代码：4 步转换

```python
def apply_rotary_emb(x, freqs_cis):
    # x shape: (batch, seq_len, n_heads, head_dim)
    
    # 步骤 1: 实数 → 复数（相邻维度配对）
    #   (batch, seq, heads, head_dim) → (batch, seq, heads, head_dim//2, 2) → complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # shape: (batch, seq_len, n_heads, head_dim // 2), dtype=complex
    
    # 步骤 2: 调整 freqs_cis 的 shape 以便广播
    #   freqs_cis: (max_seq_len, head_dim//2) → (1, seq_len, 1, head_dim//2)
    seq_len = x.shape[1]
    freqs_cis = freqs_cis[:seq_len].unsqueeze(0).unsqueeze(2)
    
    # 步骤 3: 复数乘法 = 旋转！！（这就是 RoPE 的全部核心）
    x_rotated = x_complex * freqs_cis
    
    # 步骤 4: 复数 → 实数（恢复原始 shape）
    #   complex → (batch, seq, heads, head_dim//2, 2) → flatten → (batch, seq, heads, head_dim)
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    
    return x_out.type_as(x)
```

### 维度变化一览

```
输入:     (batch, seq_len, n_heads, head_dim)       [实数]
  ↓ reshape + view_as_complex
复数:     (batch, seq_len, n_heads, head_dim//2)     [复数]
  ↓ × freqs_cis（广播乘法）
旋转后:   (batch, seq_len, n_heads, head_dim//2)     [复数]
  ↓ view_as_real + flatten
输出:     (batch, seq_len, n_heads, head_dim)       [实数]
```

### 面试口述

> "四步：1) 把最后一维两两配对转成复数；2) 对齐 freqs_cis 的 shape；
> 3) 复数乘法做旋转——这一步就是 RoPE 的全部核心；4) 转回实数。"

---

## Step 4：封装 RoPE Module（1分钟）

### 对应理论：预计算 + 缓存

旋转频率只取决于 `head_dim` 和 `max_seq_len`，和输入数据无关，
所以只需要在初始化时计算一次，之后反复使用。

### 对应代码

```python
class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len=4096, theta=10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len, theta)
        # register_buffer: 不是可训练参数，但会跟着 model.to(device) 走
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
    
    def forward(self, x):
        return apply_rotary_emb(x, self.freqs_cis)
```

**为什么用 `register_buffer` 而不是 `nn.Parameter`？**
- 旋转频率是固定值，不需要梯度
- 但需要跟随模型移到 GPU，所以不能是普通属性

---

## Step 5：口述验证方案（1分钟）

面试时不一定需要写验证代码，但要能说出来：

**验证 1 — 模长不变性**（旋转是正交变换，不改变向量长度）
```python
assert torch.allclose(torch.norm(x), torch.norm(rope(x)))
```

**验证 2 — 相对位置不变性**（相同相对距离 → 相同内积）
```python
# 位置 (2,5) 和 (7,10) 的 q·k 应该相同，因为 5-2 = 10-7 = 3
```

**验证 3 — 矩阵等价性**
```python
# 旋转矩阵 R @ x 应该等于复数乘法 apply_rotary_emb(x) 的结果
```

---

## 面试时间分配总结

```
Step 1: 写 import + 函数签名          →  30秒
Step 2: 实现 precompute_freqs_cis     →   2分钟
Step 3: 实现 apply_rotary_emb         →   3分钟
Step 4: 封装 RoPE Module              →   1分钟
Step 5: 口述验证方案                   →   1分钟
                                      ──────────
                                  总计 ≈ 7-8 分钟
```

---

## 必须记住的关键 API

| API | 用途 | 出现在 |
|-----|------|--------|
| `torch.arange(0, dim, 2)` | 生成维度对索引 `[0,2,4,...]` | Step 2 |
| `torch.outer(a, b)` | 外积，位置 × 频率 → 角度矩阵 | Step 2 |
| `torch.polar(abs, angle)` | 生成单位复数 `e^(j*angle)` | Step 2 |
| `torch.view_as_complex(x)` | 实数张量 → 复数张量 | Step 3 |
| `torch.view_as_real(x)` | 复数张量 → 实数张量 | Step 3 |
| `register_buffer` | 注册不可训练的缓存 | Step 4 |

---

## 面试常见追问

**Q: RoPE 和 Sinusoidal PE 有什么区别？**
> Sinusoidal PE 是加法（加到 embedding 上），RoPE 是乘法（旋转 q/k）。
> RoPE 天然编码相对位置，Sinusoidal PE 只编码绝对位置。

**Q: 为什么 RoPE 只作用于 q 和 k，不作用于 v？**
> 位置信息只需要影响 attention weight（由 q·k 决定），不需要影响 value 的加权求和。

**Q: RoPE 的计算开销大吗？**
> 不大。旋转因子可以预计算，应用时只是逐元素复数乘法。

**Q: `theta = 10000` 的含义？**
> 控制频率衰减速度。theta 越大，高频和低频跨度越大，模型能区分的最大距离越远。

**Q: 如何扩展到更长的序列？（长上下文）**
> NTK-aware scaling、YaRN 等方法，核心思想是调整 theta 或对频率做缩放插值。

---

## 运行方式

```bash
pip install torch
python demo.py
```

---

## 文件结构

```
RoPE/
├── README.md     ← 你正在读的教程（面试步骤导向）
├── rope.py       ← 核心实现（带详细中文注释）
└── demo.py       ← 验证演示（5个实验）
```

---

## 参考资料

- [RoPE 原始论文: RoFormer (Su et al., 2021)](https://arxiv.org/abs/2104.09864)
- [LLaMA 源码中的 RoPE 实现](https://github.com/facebookresearch/llama)
- [Eleuther AI 的 RoPE 讲解](https://blog.eleutherai.com/rotary-embeddings/)
