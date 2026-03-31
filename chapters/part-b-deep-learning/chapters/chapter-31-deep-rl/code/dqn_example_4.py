"""
PPO (Proximal Policy Optimization) 完整实现
OpenAI的主打算法，稳定、高效

核心特点：
1. 裁剪损失函数（Clipped Surrogate Objective）
2. 广义优势估计（GAE）
3. 多epoch更新
4. 熵奖励鼓励探索

作者: 机器学习与深度学习：从小学生到大师
参考: Schulman et al. (2017)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gym


class ActorCritic(nn.Module):
    """
    PPO的Actor-Critic网络
    共享特征提取层，分别输出策略和价值
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # 共享特征层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor头：输出动作分布（离散动作用softmax）
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic头：输出状态价值
        self.critic = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """前向传播，返回动作分布和价值"""
        features = self.feature(state)
        action_probs = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return action_probs, value
    
    def get_action(self, state, deterministic=False):
        """获取动作"""
        with torch.no_grad():
            action_probs, value = self.forward(state)
            dist = Categorical(action_probs)
            
            if deterministic:
                action = torch.argmax(action_probs)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def evaluate(self, states, actions):
        """
        评估动作
        
        Args:
            states: 状态批量
            actions: 动作批量
        
        Returns:
            log_probs: 动作对数概率
            values: 状态价值
            entropy: 分布熵
        """
        action_probs, values = self.forward(states)
        dist = Categorical(action_probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy


class RolloutBuffer:
    """
    回滚缓冲区：存储一个回合的经验
    PPO是同策略算法，需要收集一批经验后更新
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def push(self, state, action, log_prob, reward, value, done):
        """存储一步经验"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def get(self):
        """获取所有经验"""
        return (np.array(self.states),
                np.array(self.actions),
                np.array(self.log_probs),
                np.array(self.rewards),
                np.array(self.values),
                np.array(self.dones))
    
    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) 智能体
    
    特点：
    1. 裁剪目标防止策略更新过大
    2. GAE估计优势
    3. 多epoch更新提高样本利用
    4. 熵奖励鼓励探索
    """
    def __init__(self, state_dim, action_dim,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5,
                 hidden_dim=64, device='cpu'):
        """
        初始化PPO智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE参数
            clip_epsilon: 裁剪范围
            value_coef: 价值损失系数
            entropy_coef: 熵奖励系数
            max_grad_norm: 梯度裁剪
            hidden_dim: 隐藏层维度
            device: 计算设备
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 网络
        self.network = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # 回滚缓冲区
        self.buffer = RolloutBuffer()
    
    def select_action(self, state, deterministic=False):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, log_prob, value = self.network.get_action(state_tensor, deterministic)
        return action, log_prob, value
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        """存储转移"""
        self.buffer.push(state, action, log_prob, reward, value, done)
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        计算广义优势估计（GAE）
        
        Args:
            rewards: 奖励序列
            values: 价值估计序列
            dones: 终止标志序列
            next_value: 下一状态的价值估计
        
        Returns:
            advantages: 优势估计
            returns: 回报（用于价值函数更新）
        """
        advantages = []
        gae = 0
        
        # 从后向前计算
        values = np.append(values, next_value)
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                # 回合结束，下一状态的引导为0
                next_value = 0
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t+1] - values[t]
            
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def learn(self, next_state, num_epochs=4, batch_size=64):
        """
        学习
        
        Args:
            next_state: 最后一个状态的下一状态
            num_epochs: 更新轮数
            batch_size: 批量大小
        
        Returns:
            policy_loss: 策略损失
            value_loss: 价值损失
            entropy: 平均熵
        """
        if len(self.buffer) == 0:
            return None, None, None
        
        # 获取经验
        states, actions, old_log_probs, rewards, values, dones = self.buffer.get()
        
        # 计算下一状态的价值
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            _, next_value = self.network(next_state_tensor)
            next_value = next_value.item()
        
        # 计算GAE和回报
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 多epoch更新
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        for epoch in range(num_epochs):
            # 随机打乱
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_idx = indices[start:end]
                
                # 获取批量
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # 评估
                log_probs, values, entropy = self.network.evaluate(batch_states, batch_actions)
                
                # =============== 策略损失（PPO核心）===============
                # 概率比
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # 裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # =============== 价值损失 ===============
                value_loss = F.mse_loss(values, batch_returns)
                
                # =============== 总损失 ===============
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # 清空缓冲区
        self.buffer.clear()
        
        return policy_loss.item(), value_loss.item(), entropy.mean().item()
    
    def save(self, filepath):
        """保存模型"""
        torch.save(self.network.state_dict(), filepath)
    
    def load(self, filepath):
        """加载模型"""
        self.network.load_state_dict(torch.load(filepath, map_location=self.device))


def train_ppo_cartpole():
    """在CartPole环境上训练PPO"""
    import gym
    
    # 创建环境
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 创建智能体
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        device=device
    )
    
    # 训练参数
    max_episodes = 1000
    steps_per_update = 2048  # 每收集这么多步更新一次
    
    episode_rewards = []
    step_count = 0
    
    for episode in range(max_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        episode_reward = 0
        
        while True:
            # 选择动作
            action, log_prob, value = agent.select_action(state)
            
            # 执行动作
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result
            
            # 存储经验
            agent.store_transition(state, action, log_prob, reward, value, done)
            
            episode_reward += reward
            step_count += 1
            state = next_state
            
            # 每收集steps_per_update步就更新
            if step_count % steps_per_update == 0 or done:
                # 如果是回合结束，next_state是终止状态
                loss = agent.learn(next_state, num_epochs=10, batch_size=64)
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"回合 {episode+1}/{max_episodes}, 平均奖励: {avg_reward:.1f}")
            
            if avg_reward > 475:
                print("环境已解决！")
                break
    
    env.close()
    return agent, episode_rewards


if __name__ == "__main__":
    print("开始PPO训练...")
    agent, rewards = train_ppo_cartpole()
    print(f"训练完成！最终平均奖励: {np.mean(rewards[-50:]):.1f}")