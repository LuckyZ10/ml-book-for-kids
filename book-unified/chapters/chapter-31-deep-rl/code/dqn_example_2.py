"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) 完整实现
DDPG的改进版，解决过估计问题

改进：
1. Clipped Double Q-Learning: 两个Critic，取较小值
2. Delayed Policy Updates: 延迟策略更新
3. Target Policy Smoothing: 目标策略平滑

作者: 机器学习与深度学习：从小学生到大师
参考: Fujimoto et al. (2018)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Actor网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class GaussianNoise:
    """高斯噪声（TD3使用高斯噪声而非OU噪声）"""
    def __init__(self, action_dimension, sigma=0.1):
        self.action_dimension = action_dimension
        self.sigma = sigma
    
    def noise(self):
        return self.sigma * np.random.randn(self.action_dimension)


class TD3Agent:
    """
    TD3智能体
    
    相比DDPG的改进：
    1. 双Critic: 解决过估计问题
    2. 延迟策略更新: 每d步更新一次Actor
    3. 目标策略平滑: 给目标动作加噪声
    """
    def __init__(self, state_dim, action_dim, 
                 actor_lr=3e-4, critic_lr=3e-4,
                 gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5,
                 policy_delay=2,
                 buffer_capacity=100000,
                 hidden_dim=256, device='cpu'):
        """
        初始化TD3智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            actor_lr: Actor学习率
            critic_lr: Critic学习率
            gamma: 折扣因子
            tau: 软更新系数
            policy_noise: 目标策略噪声标准差
            noise_clip: 噪声裁剪范围
            policy_delay: 策略更新延迟（每几步更新一次Actor）
            buffer_capacity: 回放缓冲区容量
            hidden_dim: 隐藏层维度
            device: 计算设备
        """
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        
        # Actor网络
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # 双Critic网络（TD3的核心）
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr
        )
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 探索噪声
        self.exploration_noise = GaussianNoise(action_dim, sigma=0.1)
        
        # 训练计数
        self.train_step = 0
        self.actor_loss = 0
    
    def select_action(self, state, add_noise=True, noise_scale=1.0):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if add_noise:
            noise = self.exploration_noise.noise() * noise_scale
            action = action + noise
            action = np.clip(action, -1, 1)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转移"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def soft_update(self, source, target, tau):
        """软更新"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )
    
    def learn(self, batch_size=64):
        """
        学习一步
        
        Args:
            batch_size: 批量大小
        
        Returns:
            critic_loss: Critic损失
            actor_loss: Actor损失（可能为0如果没有更新Actor）
        """
        if len(self.replay_buffer) < batch_size:
            return None, None
        
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # =============== Critic更新 ===============
        with torch.no_grad():
            # 目标策略平滑：给目标动作加噪声
            next_actions = self.actor_target(next_states)
            noise = torch.randn_like(next_actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1, 1)
            
            # 双Critic目标值：取较小值（防止过估计）
            next_q1 = self.critic1_target(next_states, next_actions)
            next_q2 = self.critic2_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)  # ★ 关键：取最小值
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 当前Q值
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Critic损失：两个Critic都优化
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # =============== Actor更新（延迟）===============
        actor_loss = None
        
        if self.train_step % self.policy_delay == 0:
            # Actor目标：最大化critic1的Q值
            # 注意：只使用critic1来指导策略更新
            predicted_actions = self.actor(states)
            actor_loss = -self.critic1(states, predicted_actions).mean()
            
            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.actor_loss = actor_loss.item()
            
            # 软更新目标网络
            self.soft_update(self.actor, self.actor_target, self.tau)
            self.soft_update(self.critic1, self.critic1_target, self.tau)
            self.soft_update(self.critic2, self.critic2_target, self.tau)
        
        self.train_step += 1
        
        return critic_loss.item(), self.actor_loss if actor_loss is None else actor_loss.item()
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())


# 训练代码与DDPG类似，省略...

if __name__ == "__main__":
    # 简单测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 使用之前的ContinuousGridWorld环境
    from typing import Tuple
    
    class TestEnv:
        def __init__(self):
            self.state_dim = 6
            self.action_dim = 2
        
        def reset(self):
            return np.random.randn(6)
        
        def step(self, action):
            next_state = np.random.randn(6)
            reward = np.random.randn()
            done = np.random.rand() > 0.95
            return next_state, reward, done, {}
    
    env = TestEnv()
    agent = TD3Agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=device
    )
    
    # 简单训练循环
    for episode in range(100):
        state = env.reset()
        episode_reward = 0
        
        for step in range(100):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) > 64:
                agent.learn(64)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        if (episode + 1) % 20 == 0:
            print(f"回合 {episode+1}, 奖励: {episode_reward:.2f}")
    
    print("TD3测试完成!")