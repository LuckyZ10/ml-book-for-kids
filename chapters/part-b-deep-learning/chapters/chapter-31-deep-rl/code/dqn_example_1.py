"""
DDPG (Deep Deterministic Policy Gradient) 完整实现
适用于连续动作空间的强化学习任务

作者: 机器学习与深度学习：从小学生到大师
参考: Lillicrap et al. (2016)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


# ==================== 经验回放缓冲区 ====================

class ReplayBuffer:
    """
    经验回放缓冲区：存储和采样转移样本
    """
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储一个转移样本"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样一个批量"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


# ==================== 神经网络定义 ====================

class Actor(nn.Module):
    """
    Actor网络：确定性策略
    输入：状态
    输出：动作（连续值，范围[-1, 1]）
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # 输出范围[-1, 1]
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """
    Critic网络：Q函数
    输入：状态和动作
    输出：Q值（标量）
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # 将状态和动作拼接后输入
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


# ==================== OU噪声 ====================

class OUNoise:
    """
    Ornstein-Uhlenbeck噪声过程
    用于连续动作空间的探索
    
    OU过程具有均值回归特性，产生时间相关的噪声，
    适合物理控制任务（考虑动量）
    """
    def __init__(self, action_dimension, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
    
    def reset(self):
        """重置噪声状态"""
        self.state = np.ones(self.action_dimension) * self.mu
    
    def noise(self):
        """生成噪声"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dimension)
        self.state = x + dx
        return self.state


# ==================== DDPG智能体 ====================

class DDPGAgent:
    """
    DDPG智能体：深度确定性策略梯度
    
    核心组件：
    - Actor：确定性策略网络
    - Critic：Q函数网络
    - 目标网络：用于稳定训练
    - OU噪声：连续动作探索
    - 软更新：平滑更新目标网络
    """
    def __init__(self, state_dim, action_dim, 
                 actor_lr=1e-4, critic_lr=1e-3, 
                 gamma=0.99, tau=0.005, 
                 buffer_capacity=100000, 
                 hidden_dim=256, device='cpu'):
        """
        初始化DDPG智能体
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            actor_lr: Actor学习率
            critic_lr: Critic学习率
            gamma: 折扣因子
            tau: 软更新系数
            buffer_capacity: 回放缓冲区容量
            hidden_dim: 隐藏层维度
            device: 计算设备
        """
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # 创建网络
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        
        # 复制权重到目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # OU噪声
        self.ou_noise = OUNoise(action_dim)
        
        # 训练步数
        self.train_step = 0
    
    def select_action(self, state, add_noise=True, noise_scale=1.0):
        """
        选择动作
        
        Args:
            state: 当前状态
            add_noise: 是否添加探索噪声
            noise_scale: 噪声缩放因子
        
        Returns:
            action: 选择的动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if add_noise:
            noise = self.ou_noise.noise() * noise_scale
            action = action + noise
            # 裁剪到有效范围
            action = np.clip(action, -1, 1)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转移样本"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def soft_update(self, source, target, tau):
        """
        软更新目标网络
        target = tau * source + (1 - tau) * target
        """
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
            actor_loss: Actor损失
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
        # 计算目标Q值
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 当前Q值
        current_q = self.critic(states, actions)
        
        # Critic损失
        critic_loss = F.mse_loss(current_q, target_q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # =============== Actor更新 ===============
        # Actor目标：最大化Q值
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # =============== 软更新目标网络 ===============
        self.soft_update(self.actor, self.actor_target, self.tau)
        self.soft_update(self.critic, self.critic_target, self.tau)
        
        self.train_step += 1
        
        return critic_loss.item(), actor_loss.item()
    
    def reset_noise(self):
        """重置OU噪声"""
        self.ou_noise.reset()
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        # 同步目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())


# ==================== 简单连续控制环境 ====================

class ContinuousGridWorld:
    """
    连续动作网格世界
    一个简单的测试环境，智能体需要到达目标位置
    
    状态：[x, y, vx, vy, gx, gy]
    动作：[ax, ay] (加速度，范围[-1, 1])
    """
    def __init__(self, size=5.0):
        self.size = size
        self.dt = 0.1
        self.max_speed = 2.0
        self.reset()
    
    def reset(self):
        """重置环境"""
        # 随机起始位置
        self.position = np.random.uniform(-self.size, self.size, 2)
        self.velocity = np.zeros(2)
        # 随机目标位置
        self.goal = np.random.uniform(-self.size, self.size, 2)
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态"""
        return np.concatenate([self.position, self.velocity, self.goal])
    
    def step(self, action):
        """执行动作"""
        # 动作是加速度
        action = np.clip(action, -1, 1)
        
        # 更新速度
        self.velocity += action * self.dt
        self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed)
        
        # 更新位置
        self.position += self.velocity * self.dt
        
        # 边界处理（反弹）
        for i in range(2):
            if abs(self.position[i]) > self.size:
                self.position[i] = np.sign(self.position[i]) * self.size
                self.velocity[i] *= -0.5
        
        # 计算奖励
        distance = np.linalg.norm(self.position - self.goal)
        reward = -distance  # 负距离作为奖励
        
        # 到达目标
        done = distance < 0.5
        if done:
            reward += 10.0
        
        return self._get_state(), reward, done, {}
    
    @property
    def state_dim(self):
        return 6
    
    @property
    def action_dim(self):
        return 2


# ==================== 训练脚本 ====================

def train_ddpg():
    """训练DDPG智能体"""
    # 设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建环境
    env = ContinuousGridWorld(size=5.0)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    # 创建智能体
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=100000,
        hidden_dim=256,
        device=device
    )
    
    # 训练参数
    num_episodes = 500
    max_steps = 200
    batch_size = 64
    noise_scale = 1.0
    noise_decay = 0.995
    min_noise = 0.1
    
    # 训练循环
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        agent.reset_noise()
        episode_reward = 0
        
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state, add_noise=True, noise_scale=noise_scale)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 学习
            if len(agent.replay_buffer) > batch_size:
                agent.learn(batch_size)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        noise_scale = max(min_noise, noise_scale * noise_decay)
        
        # 打印进度
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"回合 {episode+1}/{num_episodes}, 平均奖励: {avg_reward:.2f}, 噪声: {noise_scale:.3f}")
    
    print("训练完成!")
    return agent, episode_rewards


if __name__ == "__main__":
    # 运行训练
    agent, rewards = train_ddpg()
    
    # 绘制学习曲线
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DDPG Training Progress')
    plt.grid(True)
    plt.savefig('ddpg_training.png')
    print("学习曲线已保存到 ddpg_training.png")