"""
FlashAttention (V1/V2 核心数学理念) 面试级绝杀机制实现 
=====================================================
【理论揭秘】：
如果你在深水区面试时被问到：“怎么才能算极其无穷长距离序列的注意力？显存吃不消怎么办？”
如果答：FlashAttention。必跟进连招拷问：“它是减少了计算量吗？”
如果你答 是，直接挂掉出局。
真正的秘密：FlashAttention **FLOPs (乘加计算点数) 其实不仅没降甚至略微升了点点，它的神作是在降低能把人逼疯的 内存读写IO墙 (Memory Wall)！**
它坚决杜绝把 O(N^2) 产生的那个巨大如同黑洞般的 Attention Score 表格给塞向 HBM (主显存)，
而是极度讨巧地划分为极小的豆腐块区域 (Tiling 分块法)，塞在晶体管旁边全速运转的极其残暴但微小的 SRAM 缓存里计算完，合并吐出低维最终结果！

面试终极死亡拷问：
你刚才说切碎成一小块一小块算，可是大哥，以前 Softmax 最后那个大除法分母可是要把一横排所有的数值全加了你才能算出来，你现在只切了一小块视野，根本不知道全局最大数字在哪儿防溢出，你怎么敢算分块小 Softmax ？？？
解法这就是这段数学的美学：【Online Softmax】！
"""

import torch
import math

def flash_attention_forward_sim(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, block_size: int = 2):
    """
    为了纯粹展现精深分块和合并降维哲学，不加入其它杂余维度如Batch和Heads
    Q, K, V 维度形状统定: [SeqLen, d_head] 
    block_size: 为了演示将庞大显卡切割挤成小块，特意设的一个极小的模拟 SRAM 容量池
    """
    N, d = Q.size()
    
    # 🌟在极慢龟速广阔的外部主显存 HBM（High Bandwidth Memory）里划定结果预留区
    # 从头到尾，我们绝不创建极其可怕巨大的 O(N^2) 实体打分死区！最高内存占用永远只有 O(N*d)
    O = torch.zeros((N, d))            # 要被慢慢切块回填组装的最终长相输出矩阵
    l = torch.zeros((N, 1))            # 全局分母追踪器：这一排到底被多大的群星璀璨分数膨胀压垮过？
    m = torch.full((N, 1), -torch.inf) # 全局最大势能追踪器：老大哥究竟是多大？目前为止见过的最恐怖数据是多少？
    
    # 开始挤进小黑屋！
    # 外圈大循环遍历 (模拟从外部 HBM 以极其不情愿的吝啬态度吸取每次只有那么一小搓的 Query 块)
    for i in range(0, N, block_size):
        q_block = Q[i:i+block_size, :]
        
        # 将我们手里的三个至关紧要的状态指针给拽进小黑屋握在手上
        o_i = O[i:i+block_size, :]
        l_i = l[i:i+block_size, :]
        m_i = m[i:i+block_size, :]
        
        # 内圈探索：拿着这一小批 Query 打起火把，去沿着无边无际的所有的 Key 和 Value 部落探查搜刮
        for j in range(0, N, block_size):
            k_block = K[j:j+block_size, :]
            v_block = V[j:j+block_size, :]
            
            # 【第一步】：老套的撞击火花打分。这一瞬间，因为切片极小，内存完全毫无负担。
            S_block = torch.matmul(q_block, k_block.T) / math.sqrt(d)
            
            # 【第二步】：重修族谱。找出在这个当前这微小时代的最新区域小老大哥 m_block
            m_block, _ = torch.max(S_block, dim=-1, keepdim=True)
            
            # ======================== ONLINE SOFTMAX 心跳操作 ========================
            # 伟大的改朝换代对决时刻：历史大军的历史老大哥和眼前探查到出头鸟的新老大哥决出新时代的无敌帝王！
            m_new = torch.max(m_i, m_block)
            
            # 核心修正：由于皇帝变了水涨船高，以前的旧世界旧时代打出来的分母必须被暴摔降级。
            # 这是因为我们 Softmax 为了防溢出一直在强行 P = exp(原数据 - max)，如果万物坐标轴被新大哥拉升了，
            # 那么所有曾经那些辉煌的老分数的底座都要按照此比例集体缩水掉这几截！这种回撤在数学上保证了极其神乎其神的完全闭环一致性。
            exp_scale_old = torch.exp(m_i - m_new)
            exp_scale_new = torch.exp(m_block - m_new)
            
            # 当前眼前这些生力军也是在扣下了此时的新帝王的分数后开始膨胀概率的
            P_block = torch.exp(S_block - m_new)
            
            # 这就是神迹！把旧有的族长分母无伤缩水融合掉眼前这群朝气蓬勃的新分母！！
            l_new = exp_scale_old * l_i + torch.sum(P_block, dim=-1, keepdim=True)
            
            # 最后：把降权过一轮的老输出加上此刻算出来的微小的被压掉新势能概率输出相加
            # 我们并没有把 P_block 给当真，这都是还没除以定死天下分母的半成品罢了
            o_i_unscaled = o_i * l_i * exp_scale_old + torch.matmul(P_block, v_block)
            
            # =========================================================================
            
            # 更新当朝这一大轮的追记指针，把目前为止在当前除以总分母的情况登记在小册子上
            o_i = o_i_unscaled / l_new
            l_i = l_new
            m_i = m_new
            
        # 辛苦拿着这一组 Query 彻底横扫走完检阅完全世界！
        # 把最终定音完全正确合乎逻辑的结果原封不动推流回外部缓慢大仓库保存
        O[i:i+block_size, :] = o_i
        l[i:i+block_size, :] = l_i
        m[i:i+block_size, :] = m_i
        
    return O
