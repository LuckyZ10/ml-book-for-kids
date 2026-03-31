"""
强化学习基础实现
第40章：强化学习——试错中成长
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QLearningAgent:
    """Q-Learning智能体"""
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha      # 学习率
        self.gamma = gamma      # 折扣因子
        self.epsilon = epsilon  # 探索率
        
        # 初始化Q表
        self.q_table = np.zeros((n_states, n_actions))
    
    def choose_action(self, state):
        """ε-贪心策略"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        """Q-Learning更新"""
        current_q = self.q_table[state, action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        
        # TD更新
        self.q_table[state, action] += self.alpha * (target_q - current_q)
        
        return target_q - current_q  # 返回TD误差


class DQNNetwork(nn.Module):
    """DQN网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    """DQN智能体"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=10000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # 主网络和目标网络
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放
        self.memory = deque(maxlen=buffer_size)
    
    def choose_action(self, state):
        """ε-贪心策略"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self):
        """从经验回放中学习"""
        if len(self.memory) < self.batch_size:
            return None
        
        # 随机采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 当前Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 目标Q值
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 损失
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 衰减epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """同步目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())


class ActorNetwork(nn.Module):
    """Actor网络（策略）"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)


class CriticNetwork(nn.Module):
    """Critic网络（价值）"""
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.fc(x)


class ActorCriticAgent:
    """Actor-Critic智能体"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )
    
    def choose_action(self, state):
        """根据策略选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(state_tensor)
        
        action = np.random.choice(len(probs[0]), p=probs[0].numpy())
        return action
    
    def learn(self, state, action, reward, next_state, done):
        """Actor-Critic更新"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Critic评估
        value = self.critic(state_tensor)
        with torch.no_grad():
            next_value = self.critic(next_state_tensor)
        
        # TD误差（优势）
        target = reward + self.gamma * next_value * (1 - done)
        advantage = target - value
        
        # Critic损失
        critic_loss = advantage.pow(2).mean()
        
        # Actor损失
        probs = self.actor(state_tensor)
        log_prob = torch.log(probs[0, action])
        actor_loss = -(log_prob * advantage.detach()).mean()
        
        # 总损失
        loss = actor_loss + critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


if __name__ == "__main__":
    print("强化学习基础实现")
    print("=" * 50)
    print("\n包含以下算法：")
    print("1. Q-Learning - 基础表格型方法")
    print("2. DQN - 深度Q网络")
    print("3. Actor-Critic - 策略梯度基础")
    print("\n使用方法：")
    print("  from rl_basic import QLearningAgent, DQNAgent, ActorCriticAgent")
    print("  # 参见 train_cartpole.py 完整训练示例")
