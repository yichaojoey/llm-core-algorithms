import torch
import math
import torch.nn.functional as F
from flashattention import flash_attention_forward_sim

def test_flash_attention_equivalence():
    """
    最为震撼的代码测算。
    无论是切分得多么七零八落。证明由于极其精巧极具美感的 Online Softmax 拉高指数衰减校准分母法。
    它一块一块缝合上去之后的结果竟然能在小数点后五位和那占据极其庞大内存做出的全局归一化算式产物：完！全！无！差！异！
    """
    N = 120
    d = 16
    
    Q = torch.randn(N, d)
    K = torch.randn(N, d)
    V = torch.randn(N, d)
    
    # ================= 笨重庞大的经典算法路线 =================
    scores = torch.matmul(Q, K.T) / math.sqrt(d) # 产出了极其可怕的 N * N (120 * 120 的整片实体内存占用)
    attn = F.softmax(scores, dim=-1)
    correct_O = torch.matmul(attn, V)
    # =======================================================
    
    # ================= 神乎极限的高速小块流转体系 =================
    # 我们刻意将 block 切得十分细长琐碎，让它迭代交替错换几十遍，甚至经历最大数字反复被横刀夺爱推翻！
    flash_O = flash_attention_forward_sim(Q, K, V, block_size=17) 
    # =======================================================
    
    # 揭示神迹时刻
    assert torch.allclose(correct_O, flash_O, atol=1e-5), "灾难：如果 Online Softmax 数系失衡崩盘，导致拼凑计算无法与普通全局结算画上等号！"
