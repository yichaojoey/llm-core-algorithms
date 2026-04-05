"""
大模型核心底层黑魔法：全贴合拼接 (Sequence Packing) 与块对角掩码 (Block-Diagonal Causal Mask)
=================================================================================
【理论揭秘】：
这是导致大厂普通预训练比你家训练进度能快 **10倍以上**，极其隐蔽节省千万钞票的代码细节！
当我们要在一张含有 4096 容量上限的显卡做预训练。
如果你的库里有三句话：
- A句："Hello！" (极短，长 2)
- B句："我是一个被制造的小怪兽。" (中等，长 11)

如果这在新手代码里：它会生生把 A 句后面塞满 4094 个彻头彻尾浪费算力和显存的 `<PAD>` 空白标记符去硬凑成 4096 的形状算矩阵乘法！！这会浪费 90% 以上显卡算力去乘 0！！

**大神流派：Packing (打大串烧)**
直接把它们首尾相连硬拼凑在一个串里 `[Hello][我是一个被制造的小怪兽][...]` 刚好塞满那 4096 甚至溢出断开喂入下个批次！彻底消灭 `<PAD>` 占位。
但是，致命问题来了：**A 里的 `Hello` 在算自回归注意力打分时，凭什么可以跨过界限去“看见”或者“干扰” B 里的话语？！难道把它们硬凑在一起就不会让模型神经错乱串线吗？**

**解法：Block-Diagonal Causal Mask (块对角遮罩) & Posiiton-IDs 控制术**
给这个拼接怪专门传一个小抄！明确哪段属于哪段的绝对护城河！
"""

import torch

def standard_wasteful_padding():
    print("=" * 60)
    print("【反面教材】 垃圾填充导致的灾难算力浪费 (Padding Causal Mask)")
    
    # A句话很短，补了俩 pad。 B句话稍微没那么短。
    A_padded = [101, 102, 0, 0] # 0 for pad
    B_padded = [88, 99, 100, 0]
    
    print(f"你把这两个矩阵喂入后，大模型为了对齐而做了巨大且毫无意义的空白计算。掩蔽层全被 0 填充物干扰。")


def sequence_packing_block_diagonal_mask():
    print("\n=" * 60)
    print("【天梯流派】 真正实现 0 内存发呆的 Sequence Packing 拼凑技巧")
    print("=" * 60)
    
    # 三段极其短小的话
    seq_A = ["天", "气", "真", "好"]
    seq_B = ["我", "饿"]
    seq_C = ["吃", "饭", "去"]
    
    # 💥 直接把它们死皮赖脸全贴凑在一起！这就叫打包压缩。没有任何 Padding 可以吃闲饭！
    packed_seq = seq_A + seq_B + seq_C
    print(f"被紧紧压缩成了一条肠粉的最终序列: {packed_seq}")
    
    total_len = len(packed_seq)
    
    # 【核心 1：造假的位置编码！Position ID 拦截】
    # 本来拼接后位置是 0,1,2,3...8。这会让大模型以为“我”排在“好”的后面，它们是连贯的一句话。
    # 我们一定要在传 Position的时候强行截断打脸！每一截断头必须强行重置为 0 ！！
    position_ids = [0, 1, 2, 3] + [0, 1] + [0, 1, 2]
    print(f"必须强行剥夺它的时间线，塞入极其诡异突变的 Position IDs:\n{position_ids}")
    
    # 【核心 2：极其华丽的黑板涂鸦。Block-Diagonal 因果遮罩 (Mask)】
    # 如果不修改 Mask，A 的词虽然时间被改了但仍然在矩阵里能相互串味到。
    mask = torch.zeros(total_len, total_len) # 0 表示允许互相注视
    
    # 我们开始在上面画对角区块。非自己团队的人（也就是不同句话），即使你前面，我也不想看你。全部标1屏蔽！
    
    ptr = 0
    lengths = [len(seq_A), len(seq_B), len(seq_C)]
    
    # 全局先把世界变得一片迷茫瞎掉 (所有的视野屏蔽掉)
    mask.fill_(float('-inf')) 
    
    for l in lengths:
        # 在对角的格栅里，只允许同族的人存在正常的阶梯状三角注视！
        chunk_causal_mask = torch.tril(torch.ones(l, l)) # 正常的下三角因果遮蔽
        
        # 凡是 0 的地方打死不管它
        mask[ptr:ptr+l, ptr:ptr+l] = chunk_causal_mask.masked_fill(chunk_causal_mask == 0, float('-inf'))
        ptr += l

    print("\n最终展现给 GPU 算子的极其神奇的 【块状对角因果遮蔽图 (Block-Diagonal Mask)】：")
    print("注：0 代表视野通透正常观看，-inf 代表你敢看直接爆瞎（不串台）")
    print(mask)
    
    print("\n✅ 面试核心亮点：你可以看到一个巨大的 9x9 正方形，被完美且彻底劈割成了 3个左上至右下的对角子正方形结界！")
    print("大模型虽然一次性吞吃了极其密集的肠粉信息，但在做乘法的时候，跨越结界的打分全被 -inf 强行摁死归零。彻底做到了【高密度满载吞吐计算】又做到了【绝不引起平行世界走火入魔】双丰收！")

if __name__ == "__main__":
    standard_wasteful_padding()
    sequence_packing_block_diagonal_mask()
