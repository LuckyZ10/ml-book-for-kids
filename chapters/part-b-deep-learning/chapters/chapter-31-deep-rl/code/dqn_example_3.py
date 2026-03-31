"""
SAC (Soft Actor-Critic) 完整实现
最大熵强化学习框架，自动温度调节

作者: 机器学习与深度学习：从小学生到大师
参考: Haarnoja et al. (2018)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from torch.distributions import Normal


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32), 
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """
    SAC Actor: 输出高斯分布的参数
    使用重参数化技巧进行采样
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state, deterministic=False):
        """
        从策略中采样动作
        
        Args:
            state: 状态
            deterministic: 是否确定性采样（评估时使用）
        
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            # 评估时使用均值
            action = torch.tanh(mean)
            return action, None
        
        # 重参数化技巧
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 可微采样
        action = torch.tanh(x_t)
        
        # 计算对数概率（包含tanh的雅可比行列式修正）
        log_prob = normal.log_prob(x_t)
        # tanh修正: log(1 - tanh(x)^2)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """Critic网络（Q函数）"""
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


class SACAgent:
    """
    SAC (Soft Actor-Critic) 智能体
    
    特点：
    1. 最大熵框架：鼓励探索，学习鲁棒策略
    2. 双Critic：减少过估计
    3. 重参数化技巧：低方差梯度估计
    4. 自动温度调节：自适应探索-利用权衡
    """
    def __init__(self, state_dim, action_dim,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005,
                 alpha=0.2, automatic_entropy_tuning=True,
                 target_entropy=None,
                 buffer_capacity=100000,
                 hidden_dim=256, device='cpu'):
        """
        初始化SAC智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            actor_lr: Actor学习率
            critic_lr: Critic学习率
            alpha_lr: 温度参数学习率
            gamma: 折扣因子
            tau: 软更新系数
            alpha: 初始温度参数
            automatic_entropy_tuning: 是否自动调节温度
            target_entropy: 目标熵（None时自动设为-action_dim）
            buffer_capacity: 回放缓冲区容量
            hidden_dim: 隐藏层维度
            device: 计算设备
        """
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Actor
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # 双Critic（SAC也使用双Critic减少过估计）
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
        
        # 温度参数 α
        if self.automatic_entropy_tuning:
            # 目标熵：通常设为 -dim(A)
            self.target_entropy = target_entropy if target_entropy is not None else -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = torch.tensor([alpha], device=device)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        self.train_step = 0
    
    def select_action(self, state, evaluate=False):
        """
        选择动作
        
        Args:
            state: 当前状态
            evaluate: 是否评估模式（无噪声）
        
        Returns:
            action: 选择的动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                action, _ = self.actor.sample(state_tensor, deterministic=True)
            else:
                action, _ = self.actor.sample(state_tensor, deterministic=False)
            action = action.cpu().numpy()[0]
        
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
        
        Returns:
            critic_loss: Critic损失
            actor_loss: Actor损失
            alpha_loss: 温度损失（如果使用自动调节）
            alpha: 当前温度值
        """
        if len(self.replay_buffer) < batch_size:
            return None, None, None, self.alpha.item()
        
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
            # 从当前策略采样下一个动作
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # 双Critic目标值
            next_q1 = self.critic1_target(next_states, next_actions)
            next_q2 = self.critic2_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            
            # 软价值：Q - α * log π
            next_q = next_q - self.alpha * next_log_probs
            
            # 目标值
            target_q = rewards + self.gamma * (1 - dones) * next_q
        
        # 当前Q值
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # =============== Actor更新 ===============
        # 重新采样动作（用于计算梯度）
        new_actions, log_probs = self.actor.sample(states)
        
        # 计算新动作的Q值
        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        q = torch.min(q1, q2)
        
        # Actor损失：α * log π - Q
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # =============== 温度更新 ===============
        alpha_loss = None
        if self.automatic_entropy_tuning:
            # 重新计算log_prob（不经过梯度）
            with torch.no_grad():
                _, log_probs_detached = self.actor.sample(states)
            
            # 温度损失
            alpha_loss = -(self.log_alpha * (log_probs_detached + self.target_entropy)).mean()
            
            # 更新温度
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # =============== 软更新目标网络 ===============
        self.soft_update(self.critic1, self.critic1_target, self.tau)
        self.soft_update(self.critic2, self.critic2_target, self.tau)
        
        self.train_step += 1
        
        alpha_loss_val = alpha_loss.item() if alpha_loss is not None else 0
        return critic_loss.item(), actor_loss.item(), alpha_loss_val, self.alpha.item()
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'alpha': self.alpha.item(),
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        if not self.automatic_entropy_tuning:
            self.alpha = torch.tensor([checkpoint['alpha']], device=self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())


if __name__ == "__main__":
    # 简单测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class TestEnv:
        def __init__(self):
            self.state_dim = 6
            self.action_dim = 2
        
        def reset(self):
            return np.random.randn(6).astype(np.float32)
        
        def step(self, action):
            next_state = np.random.randn(6).astype(np.float32)
            reward = np.random.randn()
            done = np.random.rand() > 0.95
            return next_state, reward, done, {}
    
    env = TestEnv()
    agent = SACAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=device,
        automatic_entropy_tuning=True
    )
    
    print("开始SAC训练测试...")
    for episode in range(100):
        state = env.reset()
        episode_reward = 0
        
        for step in range(100):
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) > 64:
                c_loss, a_loss, alpha_loss, alpha = agent.learn(64)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        if (episode + 1) % 20 == 0:
            print(f"回合 {episode+1}, 奖励: {episode_reward:.2f}, alpha: {agent.alpha.item():.4f}")
    
    print("SAC测试完成!")