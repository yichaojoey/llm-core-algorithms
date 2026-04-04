"""
PPO (Proximal Policy Optimization) 核心实现代码
===============================================
包含面试时最常考察的几个部分：
1. 计算 GAE (Generalized Advantage Estimation)
2. PPO-Clip 目标函数 (Surrogate Loss)
3. 用于收集轨迹的 RolloutBuffer
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

class RolloutBuffer:
    """
    用于在交互期间收集所有转移(transitions)的缓冲区。
    每次 PPO 更新完成后将被清空。
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    """
    一个简单的基于离散动作的 Actor-Critic 网络结构
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Actor 网络：输出每个动作的概率分布
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic 网络：输出状态的价值估计 V(s)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def act(self, state):
        """用于环境交互，返回动作及其对应的对数概率和状态价值"""
        action_probs = self.actor(state)
        # 按照预测的概率分布采样
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return action.detach(), dist.log_prob(action).detach(), self.critic(state).detach()
    
    def evaluate(self, state, action):
        """用于 PPO 更新阶段，计算给定 state-action 的概率分布和状态价值"""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

def compute_gae(rewards, state_values, is_terminals, gamma=0.99, lam=0.95):
    """
    核心组件 1: 计算广义优势估计 (GAE - Generalized Advantage Estimation)
    
    Args:
        rewards: 奖励序列 [seq_len]
        state_values: 状态价值序列 [seq_len]
        is_terminals: 是否是终止状态 [seq_len]
        gamma: 折扣因子
        lam: GAE 的 lambda 参数, 用于权衡方差和偏差
    Returns:
        advantages: 优势函数估计 [seq_len]
        returns: 目标价值 (advantages + state_values) [seq_len]
    """
    advantages = []
    gae = 0
    # 在序列末尾追加一个 0 作为 next_value，方便处理最后一个 step
    values = state_values + [0]
    
    # 从后往前逆序计算 (这能将时间复杂度从 O(N^2) 降为 O(N))
    for step in reversed(range(len(rewards))):
        # 1. 计算单步 TD Error
        delta = rewards[step] + gamma * values[step + 1] * (1 - is_terminals[step]) - values[step]
        # 2. 衰减累加得到 GAE
        gae = delta + gamma * lam * (1 - is_terminals[step]) * gae
        advantages.insert(0, gae)
        
    # 计算 returns (Targets = Advantages + Values)
    returns = [adv + val for adv, val in zip(advantages, state_values)]
    
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3, 
                 gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        # 保存旧的策略，用于计算 PPO 的 ratio: pi_theta / pi_theta_old
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MSELoss = nn.MSELoss()

    def select_action(self, state):
        """选择动作并将结果记录到 buffer 中"""
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_logprob, state_val = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        
        return action.item()

    def update(self):
        """核心组件 2: PPO 的更新过程与 Loss 计算"""
        # 将 buffer 中的 list 整理为 tensor
        old_states = torch.stack(self.buffer.states).detach()
        old_actions = torch.stack(self.buffer.actions).detach()
        old_logprobs = torch.stack(self.buffer.logprobs).detach()
        
        # 计算 GAE 优势和 Returns
        advantages, returns = compute_gae(
            self.buffer.rewards,
            [val.item() for val in self.buffer.state_values],
            self.buffer.is_terminals,
            self.gamma
        )
        
        # 将 Advantage 进行 Normalization (正则化)，实践中能极大提升训练稳定性
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        # 优化策略 (Update K epochs)
        for _ in range(self.K_epochs):
            # 1. 评估当前状态和动作
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            # 2. 计算 Ratio (pi_theta / pi_theta_old)
            # 由于存储的是 logprob, 用 exp(log_b - log_a) = b/a 来计算
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # 3. 计算 Surrogate Loss (PPO Clip 目标)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Actor Loss 让 advantage 最大的动作概率变大 (取负是因为我们要最大化目标函数)
            loss_actor = -torch.min(surr1, surr2).mean() 
            # Critic Loss 让 value 逼近 return
            loss_critic = self.MSELoss(state_values, returns) 
            # 加入 entropy bonus 鼓励探索
            entropy_bonus = 0.01 * dist_entropy.mean()
            
            # 最终 Loss = Actor Loss + 0.5 * Critic Loss - bonus
            loss = loss_actor + 0.5 * loss_critic - entropy_bonus
            
            # 梯度下降
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        # 训练结束后，再用新 policy 的参数去覆盖旧 policy 的参数
        self.policy_old.load_state_dict(self.policy.state_dict())
        # 清空 buffer 准备收集新的 trajectories
        self.buffer.clear()
