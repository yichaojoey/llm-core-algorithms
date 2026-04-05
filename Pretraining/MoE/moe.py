"""
MoE (Mixture of Experts) 面试级核心模拟实现 
============================================
【理论揭秘】：Mixtral 8x7B 和 DeepSeek-V3 碾压纯 Dense 模型的根源！
传统的前馈层 FFN 是一块巨无霸的肥肉连通所有数据。
MoE 的核心精髓：把一块巨型 FFN 劈开变成 N 个精美的小型 FFN（称为 专家 Experts）。
然后加装一个极为聪明的路由器（Gating Router），当一个个字词流过来时，当场给它们发门票：
“你是关于政治学的 Token？你去号位2！” / “你是写代码的 Token？你去号位8！”

面试致命核心：你如果不通过手写一段名叫【Load Balancing Loss (负载均衡损失)】的代码挂靠作为辅助惩罚！
模型由于人类的偷懒天性，会退化到：不管接到什么活，我无脑全部把票投给当前能力稍微强一丁点的“最强万能卷王专家”，导致其余专家这辈子都没吃过一次数据没练过一次梯度被活活饿死！称为：【专家坍塌 Expert Collapse问题】。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """标准的微型大本营 (小型 FFN，如果是当代模型一般会塞个 SwiGLU 进去)这里为演示本质用简单版"""
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )
    def forward(self, x):
        return self.net(x)

class SparseMoE(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        # 面试必背数字：每次只激活几位专家？（通常 Top_K = 2，意味着其余的全休眠节省时间！）
        self.top_k = top_k
        
        # 定义专家群体
        # 面试装逼点：工业界实操绝对不用 ModuleList 来 for 循环调，那样的速度比吃屎还慢！
        # 这里为了展示清晰使用 ModuleList。工业界是手写自定义 CUDA Kernel（比如 Megablocks），让专家作为同尺寸巨型张量的 Batched Matmul 并行横扫。
        self.experts = nn.ModuleList([Expert(d_model, d_model * 4) for _ in range(num_experts)])
        
        # 路由器门控阀 (Router)
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
    def forward(self, x: torch.Tensor):
        """
        x 形状: [Batch, SeqLen, D_model]
        """
        batch_size, seq_len, d_model = x.shape
        # 把二维矩阵推平 [Num_Tokens, D_model] 方便让单个门卫一个个审阅并颁发门票
        x_flat = x.view(-1, d_model)
        
        # 1. 过闸机计分: [Num_Tokens, Num_Experts]
        gate_logits = self.gate(x_flat)
        # 求出专家推荐分数 (权重) 必须大于 > 0.00 才行
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # 2. 选出 Top-K 最该服务的候选专家
        # 提取出了最佳专家的得分以及他们的工号 ID
        topk_weights, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # 权重必须做一波局部再归一化，让进入干活的这K个专家的加权分数加和=1
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # 建立收集池
        out_flat = torch.zeros_like(x_flat)
        
        # 3. 计算极其致命的 面试加分绝杀点：【Load Balancing Loss (负载均衡损失)】
        # 惩罚手段：让所有参与这批词的专家的总体接待流量频率，必须跟他们收到的总平均分数匹配抗衡
        # 也就是说，如果号位 2 整天接客，它的 P 会爆炸导致巨大惩罚。
        aux_loss = self._compute_load_balancing_loss(gate_probs)
        
        # 4. 委派执行（伪工业界 For 循环派送逻辑）
        # 外循环遍历每个专家。看看这批次都有谁拿着进我的这道门的票！
        for i, expert in enumerate(self.experts):
            # 找到哪些可怜的 Token 发到了 i 号专家的牌照。可能是作为首发(专家1位置) 也可能是候补(专家2位置)
            # mask 是个 bool，找哪一行恰好有我的专家序列号
            expert_mask = (topk_indices == i) 
            
            # 使用 torch.any 压缩列维度查出：有没有选中我的？有的话进入干活！
            if expert_mask.any():
                # 【花式代码点】 用 torch.where 在 2 级 K 层抽人
                # 这种操作就是传说中的 dispatch (派样)
                # 由于这是简单的逐循环，实际上效率低
                # 把选中当前 专家 i 的 Token 以及对应它是以什么权重身份(位置) 选中的抽出来
                token_indices, k_position = expert_mask.nonzero(as_tuple=True)
                
                # 开始剥削该专家，做大计算干活！
                expert_out = expert(x_flat[token_indices])
                
                # 根据它的权重稀释加工后的活儿叠加入总输出盘
                out_flat[token_indices] += expert_out * topk_weights[token_indices, k_position].unsqueeze(-1)
                
        out = out_flat.view(batch_size, seq_len, d_model)
        return out, aux_loss

    def _compute_load_balancing_loss(self, gate_probs: torch.Tensor):
        """
        面试终极密码代码段：Load Balancing (辅助均衡)
        【理论揭秘】：专家坍塌解决的数学之道！
        公式 L_{aux} = N * \sum ( f_i * P_i )
        f_i : 这批数据中被真实硬派给专家 i 的路由绝对比例 (Fraction)
        P_i : 这批数据所有 token 对专家 i 的打分平均值总和 (Mean Probability)
        只有当所有的专家门庭若市均等地承担 1/N 流量时，这个计算出来的乘积 Loss 取点最低能够获得解脱！
        """
        num_tokens, num_experts = gate_probs.shape
        
        # 这里的 topk_indices 我们为了保证算入全部真实硬性分配，简单假设只取最强的第一志愿专家流量
        _, best_expert = gate_probs.max(dim=-1)
        
        # 1. 计算各个分包的粗暴接客绝对硬比例频次 (f_i)
        # 用 one_hot 强行求均值便可知各个工位客流量。
        tokens_per_expert = F.one_hot(best_expert, num_classes=num_experts).float()
        fraction_per_expert = tokens_per_expert.mean(dim=0) # [Num_Experts]
        
        # 2. 计算大黑盒内每个专家的被青睐权重平均得分概率 (P_i)
        mean_prob_per_expert = gate_probs.mean(dim=0) # [Num_Experts]
        
        # 3. 乘法交汇并乘满专家数量大系数 N
        aux_loss = num_experts * torch.sum(fraction_per_expert * mean_prob_per_expert)
        return aux_loss
