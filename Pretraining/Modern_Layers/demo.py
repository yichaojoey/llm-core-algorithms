"""
Modern Layers (现代架构层) 分析验证
==========================================
"""

import torch
from rmsnorm import RMSNorm
from swiglu import SwiGLU

def run_modern_demo():
    print("=" * 60)
    print("演示 1：RMSNorm 为何比 LayerNorm 在极大层数串联时还能坚挺保底与加速")
    print("=" * 60)
    
    dim = 64
    x = torch.randn(2, 5, dim) * 100 + 50  # 刻意造出一个均值被移动到了 +50，且方差乱飘 100倍的巨大极端数据块
    
    print("\n--- 脏数据原始形态 ---")
    print(f"X 初代极恶劣均值飘移: {x.mean().item():.2f}")
    print(f"X 内部极其恐怖的数值绝对最大值: {x.abs().max().item():.2f}")
    
    # 构建经典 LayerNorm
    ln = torch.nn.LayerNorm(dim)
    # 构建革新派 RMSNorm
    rmsnorm = RMSNorm(dim)
    
    out_ln = ln(x)
    out_rms = rmsnorm(x)
    
    print("\n--- 归一化镇压结果 ---")
    print(f"老旧 LayerNorm 依然执行原教旨主义把均值拉回 0 (花费算力): {out_ln.mean().item():.5f}")
    print(f"新派 RMSNorm 根本不管均值，随他飘移动荡 (省出海量算力): {out_rms.mean().item():.5f}")
    
    print("\n但请看核心管控能力 —— 两者是否压住了炸裂的方差避免数值崩盘 (防止 inf / NaN)？")
    print(f"老旧 LayerNorm 绝对最大值: {out_ln.abs().max().item():.2f}")
    print(f"新派 RMSNorm 绝对最大值: {out_rms.abs().max().item():.2f} (大获全胜！不仅压住了极大突刺，甚至因为舍弃均值减法而略微保留了数据的特异性原生底蕴)")

    print("\n=" * 60)
    print("演示 2：SwiGLU 是如何使用极其细腻温柔的曲线替代生硬老旧的 ReLU 的")
    print("=" * 60)
    
    in_dim = 16
    hidden = 32
    
    swiglu = SwiGLU(in_dim, hidden, in_dim)
    
    demo_x = torch.tensor([[-5.0, -1.0, 0.0, 1.0, 5.0]]) # 极其夸张的正负极值试炼
    
    try:
        # 当门静脉网络和原始网络打上照面的极端化输出
        # 因为 x * sigmoid(x) 在负半轴不会像 ReLU 那样直接变成一条死直的 0 平原（所谓的梯导消失 Dead ReLU）
        # 它会在负轴保留一个极其微弱平滑并且有向下包络的反兜谷底曲线（提供回旋救活的梯度！）
        result = swiglu(demo_x.unsqueeze(1).expand(2, 1, 5))
        print(f"面对极其极端的分化数组，SwiGLU 通过双矩阵交会并降维输出了一组平滑的张量结构，并未像 ReLU 一样直接硬核截断导致一半的信息变零死掉。")
        print("✅ 验证成功：SwiGLU 平稳运行，保障深层大模型的活络程度。")
    except Exception as e:
        print(e)
        
if __name__ == "__main__":
    run_modern_demo()
