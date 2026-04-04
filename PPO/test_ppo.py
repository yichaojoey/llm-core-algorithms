import torch
import torch.nn as nn
from ppo import compute_gae, ActorCritic, PPO

def test_compute_gae():
    """测试单步和多步下的广义优势估计正确性"""
    rewards = [1.0, 0.0, 1.0]
    state_values = [0.5, 0.5, 0.5]
    is_terminals = [0, 0, 1]
    
    adv, ret = compute_gae(rewards, state_values, is_terminals, gamma=0.9, lam=1.0)
    assert adv.shape == (3,)
    assert ret.shape == (3,)
    
    # 验算最后一步的 TD error: delta = r + gamma*V(s')[因为is_done=1所以这项消去] - V(s) => 1.0 + 0 - 0.5 = 0.5
    # gae_last = 0.5 + 0 = 0.5
    assert (adv[-1].item() - 0.5) < 1e-5

def test_actor_critic():
    """测试网络的张量形状闭环"""
    ac = ActorCritic(state_dim=4, action_dim=2)
    state = torch.randn(4)
    action, logprob, val = ac.act(state)
    
    assert action.item() in [0, 1]
    assert logprob.shape == ()
    assert val.shape == (1,)

def test_ppo_agent_buffer_management():
    """测试 PPO 的前向收集机制和反向传播清空机制"""
    agent = PPO(state_dim=4, action_dim=2)
    state = torch.randn(4)
    _ = agent.select_action(state)
    
    # 模拟环境反馈
    agent.buffer.rewards.append(1.0)
    agent.buffer.is_terminals.append(True)
    
    assert len(agent.buffer.states) == 1
    
    agent.update()
    
    # 验证 buffer 在 PPO Clip 更新通过后会被规范清空
    assert len(agent.buffer.states) == 0
