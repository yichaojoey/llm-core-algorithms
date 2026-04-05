"""
标准 KV Cache 自回归递推网络模拟 
=============================================
【理论揭秘】：面试中最底层必被问烂的系统瓶颈——什么是 KV Cache？为什么要它？
大模型生成文字是“接龙”的形式 (Auto-Regressive)。
如果没有它：每次生成第 100个词，必须把前面 99 个词再进一次网络全算一次注意力（导致重复爆炸的矩阵运算）。
有了它：每生成新词，旧词已经乘出来的 Key 和 Value 会被缓存下来放在显存里，新词只要抽出自己的 Query 跟缓存好的所有老 Key 相乘即可！极大省去计算时间！
副作用：用海量的显存去换计算速度，导致上下文一旦变长，极其容易吃光整个 GPU 显存。
"""

import torch

def generate_without_kv_cache(prompt_tokens: torch.Tensor, steps: int = 5, embed_dim: int = 32):
    """
    愚蠢的反复横跳做法（完全不用缓存），这是由于对于自注意力机制缺乏理解写出的灾难级推理代码
    """
    current_tokens = prompt_tokens.clone()
    print("\n--- [Bad] 没有 KV Cache 的傻瓜生成 ---")
    
    for step in range(steps):
        seq_len = current_tokens.size(1)
        
        # 灾难级行为：强行针对目前积攒的所有长达 seq_len 的词全部生成一波重新的 QKV
        # 浪费极大极大的计算量！
        Q = torch.randn(1, seq_len, embed_dim)
        K = torch.randn(1, seq_len, embed_dim)
        V = torch.randn(1, seq_len, embed_dim)
        
        # 算完一整个极其庞大的方阵
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        print(f"Step {step+1}: 刚刚在 GPU 里面发生了一次恐怖的 {seq_len} x {seq_len} 打分矩阵运算！(非常消耗算力因为包含大批重复计算)")
        
        # 模拟生成了一个新词，接到尾巴上
        new_token = torch.tensor([[step]])
        current_tokens = torch.cat([current_tokens, new_token], dim=1)
        
    return current_tokens


def generate_with_kv_cache(prompt_tokens: torch.Tensor, steps: int = 5, embed_dim: int = 32):
    """
    这是所有主流引擎最基本的加速缓存循环：缓存之前的隐层大计算资产！
    """
    print("\n--- [Good] 启用经典 KV Cache (常驻缓存) ---")
    # 初始化开天辟地第一批上下文，将其 K 和 V 永久封存进显存池中！
    # 这是唯一一次用到极其庞大维度的初始化动作
    init_seq_len = prompt_tokens.size(1)
    # [模拟] 这是庞大的历史遗产 (KV Cache 本尊)
    cached_K = torch.randn(1, init_seq_len, embed_dim)
    cached_V = torch.randn(1, init_seq_len, embed_dim)
    
    current_token = prompt_tokens[:, -1:] # 开始循环，但以后每次只需要携带 1 本身那个最新的词条！！
    
    for step in range(steps):
        # ✅ 精华 1：由于我是唯一的新生词，我只需要算我自己一份的那可怜弱小无助的单层 Q、K、V
        new_q = torch.randn(1, 1, embed_dim)
        new_k = torch.randn(1, 1, embed_dim)
        new_v = torch.randn(1, 1, embed_dim)
        
        # ✅ 精华 2：把最新的我的 K 拼接到庞大显存资产的尾巴上去，这就叫缓存追加 (Concat/Append)！
        cached_K = torch.cat([cached_K, new_k], dim=1)  # 长达 (1, Seq_len + 1, Dim)
        cached_V = torch.cat([cached_V, new_v], dim=1)
        
        total_historical_len = cached_K.size(1)
        
        # ✅ 精华 3：永远只用我那可怜的单独 1 维的新 Query 去跟长到天际的历史 K 队列全员问好！
        # 发生了极其微小的 1 行向量与庞大矩阵的内滴。不再是 NxN 矩阵算力。极度节省算力！
        scores = torch.matmul(new_q, cached_K.transpose(-2, -1))
        
        print(f"Step {step+1}: 算力发生剧烈锐减！只用 1 个新 Query去翻阅了长达 {total_historical_len} 的旧账本得分。省掉无数无用功！")
        
        current_token = torch.tensor([[step]]) # 假装产生了一个新的接龙字符，进入下一次极其轻巧的循环！

if __name__ == "__main__":
    initial_prompt = torch.tensor([[101, 45, 66, 89]])  # 假设是 4个初始 Prefix Prompt 单词
    generate_without_kv_cache(initial_prompt)
    generate_with_kv_cache(initial_prompt)
