"""
MoE （混合专家网络） 惩罚机制功能验证
==========================================
"""

import torch
import torch.nn as nn
from moe import SparseMoE

def run_moe_demo():
    print("=" * 60)
    print("演示：MoE 如何进行路由分拣以及为何必须触发【负载均衡惩罚 Load Balancing Loss】来整改腐败？")
    print("=" * 60)
    
    d_model = 32
    num_experts = 4 # 有 4 个专职工作部门
    top_k = 2
    
    moe_layer = SparseMoE(d_model=d_model, num_experts=num_experts, top_k=top_k)
    
    print("\n--- 场景 1：完全摆烂！如果有个“万能卷王（比如专家号 2）”把活全霸占了 ---")
    # 我们强行黑进路由拦截系统，篡改发牌门控网络。使得他对所有的特征统统极其钟爱“专家二号”
    # 模拟极其倾斜极其腐败不健康的打分：对号位 2 打了 10 分！其他都是 -1。
    nn.init.constant_(moe_layer.gate.weight, -1.0)
    moe_layer.gate.weight.data[2, :] = 10.0
    
    x = torch.randn(1, 10, d_model)
    out, fake_aux_loss = moe_layer(x)
    
    print("由于所有流量不可避免地倾泻砸向了 号位 2 专家。此专家的接待量逼近 100% ！")
    print("此时刻不容缓触发警报计算出的【负载倾斜罚款 Loss (Aux_Loss)】高达:")
    print(f" -> {fake_aux_loss.item():.4f} !!! (极其恶劣，必须用反向传播砸下去)")
    
    print("\n--- 场景 2：健康发展体系。大家的饭碗都很平等端平了 ---")
    # 我们打平分数，大家雨露均沾
    nn.init.zeros_(moe_layer.gate.weight)
    
    out, good_aux_loss = moe_layer(x)
    
    print("因为所有人拿工签和分配进单量都一样。大锅饭导致了体系维系在了最初的平等安全线：")
    print("触发的【负载倾斜罚款 Loss (Aux_Loss)】暴降稳固在:")
    print(f" -> {good_aux_loss.item():.4f}。 (这证明了我们手撕的公式完美阻断了坍塌风险，保证了所有专家活络能学)")

if __name__ == "__main__":
    run_moe_demo()
