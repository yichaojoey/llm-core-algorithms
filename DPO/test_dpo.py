import torch
import torch.nn.functional as F
from dpo import compute_dpo_loss

def test_dpo_loss_formula():
    """验证 DPO 的损失计算是否在纯前向推导上契合手工算式"""
    pol_chosen = torch.tensor([-1.0])
    pol_rej = torch.tensor([-2.0])
    ref_chosen = torch.tensor([-1.5])
    ref_rej = torch.tensor([-1.8])
    
    # === 手动演算理论预期值 ===
    # pi_ratio = (-1.0) - (-2.0) = 1.0
    # ref_ratio = (-1.5) - (-1.8) = 0.3
    # Logits M = 1.0 - 0.3 = 0.7
    # -log(sigmoid(beta * M)) = -log(sigmoid(0.1 * 0.7))
    
    beta = 0.1
    expected_logits = 0.7
    expected_loss = -F.logsigmoid(torch.tensor(beta * expected_logits)).item()
    
    loss, rc, rr = compute_dpo_loss(pol_chosen, pol_rej, ref_chosen, ref_rej, beta)
    
    assert abs(loss.item() - expected_loss) < 1e-5
    # 验证 Implicit Reward 大小
    assert abs(rc.item() - beta * (-1.0 - (-1.5))) < 1e-5
    assert abs(rr.item() - beta * (-2.0 - (-1.8))) < 1e-5
