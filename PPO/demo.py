"""
PPO 算法 面试展示代码 - 验证演示
=================================

由于面试现场无法等模型训练几万步，这个 demo 核心在于：
1. 展示如何与环境交互收集数据
2. 展示前向传播和 PPO Clip 网络参数更新的完整闭环

由于环境依赖碎片化的问题，我们在这里手写一个极简的“虚拟环境”（Random Dummy Environment）来走通代码逻辑，
以避免运行此脚本需要额外安装 Gym/Gymnasium 等依赖。
"""

import torch
from ppo import PPO

class DummyEnv:
    """一个随机奖励的 Dummy 环境，用来验证 PPO 的网络更新流程不出错"""
    def __init__(self):
        self.state_dim = 4
        self.action_dim = 2
        self.max_steps = 100
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0
        return torch.randn(self.state_dim).numpy()
        
    def step(self, action):
        self.current_step += 1
        next_state = torch.randn(self.state_dim).numpy()
        # 随机给点奖励的机制，证明它能收敛梯度
        reward = 1.0 if action == 1 else -1.0
        done = self.current_step >= self.max_steps
        return next_state, reward, done

def run_demo():
    print("=" * 60)
    print("演示 1：PPO 与环境交互并更新逻辑的完整闭环")
    print("=" * 60)
    
    # 1. 初始化
    env = DummyEnv()
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    ppo_agent = PPO(state_dim, action_dim, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=4, eps_clip=0.2)
    print(f"✅ PPO Agent 初始化成功")
    print(f"   State Dim: {state_dim}, Action Dim: {action_dim}")
    print(f"\n   --- Actor 网络结构 --- \n{ppo_agent.policy.actor}")
    print(f"\n   --- Critic 网络结构 --- \n{ppo_agent.policy.critic}")
    
    # 2. 模拟训练几个 Episode
    epochs = 3
    max_ep_len = 50
    update_timestep = 100 # 每进行 100 步统一更新一次 PPO Agent
    
    time_step = 0
    
    for ep in range(1, epochs + 1):
        state = env.reset()
        ep_reward = 0
        
        for t in range(1, max_ep_len + 1):
            time_step += 1
            
            # Agent 根据旧策略 (policy_old) 选择动作并记录轨迹到 Buffer
            action = ppo_agent.select_action(state)
            
            # 与环境交互
            state, reward, done = env.step(action)
            ep_reward += reward
            
            # 记录奖励和 done 信号 (前向传递参数已经被 select_action 添加好)
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            
            # PPO 定期更新 (在面试时可以说，一般收集固定 length 后，我们统一算 GAE 和 Update)
            if time_step % update_timestep == 0:
                print(f"\n🔄 达到 Update Timestep ({time_step})，开始执行 PPO Update 梯度下降:")
                
                # 开始记录更新前的参数
                old_sample_param = list(ppo_agent.policy.actor.parameters())[0].clone()
                
                # 开始在 PPO 内按 Epoch 更新
                ppo_agent.update()
                
                new_sample_param = list(ppo_agent.policy.actor.parameters())[0].clone()
                diff = (new_sample_param - old_sample_param).abs().mean().item()
                print(f"✅ 梯度反向传播更新完成! Actor 参数平均变化量: {diff:.6e}")
                print(f"✅ Buffer 已清空，当前缓冲轨迹数量重新核对为: {len(ppo_agent.buffer.states)}")
                
            if done:
                break
                
        print(f"\nEpisode {ep} 结束 | 本轮累计轨迹步数: {min(t, max_ep_len)} | 本轮奖励总和: {ep_reward}")

if __name__ == '__main__':
    print()
    print("🔄 PPO (Proximal Policy Optimization) 验证演示")
    run_demo()
    print("\n" + "=" * 60)
    print("🎉 逻辑验证全部跑通，无报错！")
    print("=" * 60)
