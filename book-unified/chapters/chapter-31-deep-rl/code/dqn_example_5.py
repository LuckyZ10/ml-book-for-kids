"""
A3C/A2C (Asynchronous/Synchronous Advantage Actor-Critic) 实现
并行训练的优势演员-评论家

作者: 机器学习与深度学习：从小学生到大师
参考: Mnih et al. (2016)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gym


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic网络
    共享特征提取层
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCriticNetwork, self).__init__()
        
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor头
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic头
        self.critic = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """前向传播"""
        features = self.shared(state)
        action_probs = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return action_probs, value
    
    def get_action_and_value(self, state, action=None):
        """
        获取动作和价值
        
        Args:
            state: 状态
            action: 可选，如果提供则计算该动作的对数概率
        
        Returns:
            action: 采样的动作
            log_prob: 动作对数概率
            entropy: 策略熵
            value: 状态价值
        """
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)


class A2CAgent:
    """
    A2C (Advantage Actor-Critic) 智能体
    A3C的同步版本，更简单稳定
    
    特点：
    1. n-step回报估计
    2. 并行环境收集数据
    3. 同策略学习
    4. 熵奖励鼓励探索
    """
    def __init__(self, state_dim, action_dim,
                 lr=7e-4, gamma=0.99, gae_lambda=0.95,
                 value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5,
                 num_steps=5, num_envs=8,
                 hidden_dim=256, device='cpu'):
        """
        初始化A2C智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE参数
            value_coef: 价值损失系数
            entropy_coef: 熵奖励系数
            max_grad_norm: 梯度裁剪
            num_steps: n-step的n
            num_envs: 并行环境数
            hidden_dim: 隐藏层维度
            device: 计算设备
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_steps = num_steps
        self.num_envs = num_envs
        
        # 网络
        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        # 存储
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def select_action(self, state):
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action, log_prob, _, value = self.network.get_action_and_value(state_tensor)
        
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        """存储转移"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_returns_and_advantages(self, next_states, next_dones):
        """
        计算n-step回报和优势
        
        Args:
            next_states: 下一状态
            next_dones: 下一状态是否终止
        
        Returns:
            returns: n-step回报
            advantages: 优势
        """
        with torch.no_grad():
            next_states_tensor = torch.FloatTensor(next_states).to(self.device)
            _, next_values = self.network(next_states_tensor)
            next_values = next_values.squeeze(-1).cpu().numpy()
            next_values = next_values * (1 - next_dones)  # 终止状态价值为0
        
        # 计算n-step回报
        returns = np.zeros((len(self.rewards), self.num_envs))
        advantages = np.zeros((len(self.rewards), self.num_envs))
        
        for env_idx in range(self.num_envs):
            returns_env = []
            advantages_env = []
            gae = 0
            
            # 从后向前计算
            next_value = next_values[env_idx]
            
            for t in reversed(range(len(self.rewards))):
                if t == len(self.rewards) - 1:
                    next_v = next_value
                else:
                    next_v = self.values[t+1][env_idx]
                
                if self.dones[t][env_idx]:
                    next_v = 0
                
                # TD误差
                delta = self.rewards[t][env_idx] + self.gamma * next_v - self.values[t][env_idx]
                
                # GAE
                gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t][env_idx]) * gae
                advantages_env.insert(0, gae)
                
                # n-step回报
                ret = gae + self.values[t][env_idx]
                returns_env.insert(0, ret)
            
            returns[:, env_idx] = returns_env
            advantages[:, env_idx] = advantages_env
        
        return returns, advantages
    
    def learn(self, next_states, next_dones):
        """
        学习
        
        Args:
            next_states: 下一状态
            next_dones: 下一状态终止标志
        
        Returns:
            loss: 总损失
            policy_loss: 策略损失
            value_loss: 价值损失
            entropy: 平均熵
        """
        # 计算回报和优势
        returns, advantages = self.compute_returns_and_advantages(next_states, next_dones)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # 展平
        batch_size = states.shape[0] * states.shape[1]
        states = states.view(batch_size, -1)
        actions = actions.view(batch_size)
        old_log_probs = old_log_probs.view(batch_size)
        returns = returns.view(batch_size)
        advantages = advantages.view(batch_size)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 评估
        _, log_probs, entropy, values = self.network.get_action_and_value(states, actions)
        
        # 策略损失
        policy_loss = -(log_probs * advantages).mean()
        
        # 价值损失
        value_loss = F.mse_loss(values, returns)
        
        # 总损失
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # 清空存储
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        return loss.item(), policy_loss.item(), value_loss.item(), entropy.mean().item()


# ==================== Dummy VecEnv ====================

class DummyVecEnv:
    """
    简单的向量化环境包装器
    并行运行多个环境实例
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
    
    def reset(self):
        """重置所有环境"""
        results = [env.reset() for env in self.envs]
        # 处理gym新版本返回元组的情况
        states = []
        for result in results:
            if isinstance(result, tuple):
                states.append(result[0])
            else:
                states.append(result)
        return np.array(states)
    
    def step(self, actions):
        """执行动作"""
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        
        next_states = []
        rewards = []
        dones = []
        infos = []
        
        for result in results:
            if len(result) == 5:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
                # 处理自动重置
                if done:
                    next_state = self.envs[results.index(result)].reset()
                    if isinstance(next_state, tuple):
                        next_state = next_state[0]
            else:
                next_state, reward, done, info = result
            
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return np.array(next_states), np.array(rewards), np.array(dones), infos
    
    def close(self):
        for env in self.envs:
            env.close()


def make_env(env_id):
    """创建环境的工厂函数"""
    def _init():
        env = gym.make(env_id)
        return env
    return _init


def train_a2c():
    """训练A2C"""
    env_id = 'CartPole-v1'
    num_envs = 8
    num_steps = 5
    total_timesteps = 100000
    
    # 创建并行环境
    env_fns = [make_env(env_id) for _ in range(num_envs)]
    envs = DummyVecEnv(env_fns)
    
    # 获取环境信息
    state_dim = envs.envs[0].observation_space.shape[0]
    action_dim = envs.envs[0].action_space.n
    
    # 创建智能体
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=7e-4,
        gamma=0.99,
        gae_lambda=0.95,
        value_coef=0.5,
        entropy_coef=0.01,
        num_steps=num_steps,
        num_envs=num_envs,
        device=device
    )
    
    # 训练
    states = envs.reset()
    episode_rewards = [0] * num_envs
    all_rewards = []
    
    for step in range(0, total_timesteps, num_envs * num_steps):
        for n in range(num_steps):
            # 选择动作
            actions, log_probs, values = agent.select_action(states)
            
            # 执行动作
            next_states, rewards, dones, _ = envs.step(actions)
            
            # 存储
            agent.store_transition(states, actions, log_probs, rewards, values, dones)
            
            # 统计奖励
            for i in range(num_envs):
                episode_rewards[i] += rewards[i]
                if dones[i]:
                    all_rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0
            
            states = next_states
        
        # 学习
        loss, p_loss, v_loss, entropy = agent.learn(states, dones)
        
        if step > 0 and step % 5000 < num_envs * num_steps:
            avg_reward = np.mean(all_rewards[-50:]) if len(all_rewards) >= 50 else np.mean(all_rewards) if all_rewards else 0
            print(f"步数 {step}/{total_timesteps}, 平均奖励: {avg_reward:.1f}")
            
            if avg_reward > 475:
                print("环境已解决！")
                break
    
    envs.close()
    return agent, all_rewards


if __name__ == "__main__":
    print("开始A2C训练...")
    agent, rewards = train_a2c()
    if len(rewards) >= 50:
        print(f"训练完成！最终平均奖励: {np.mean(rewards[-50:]):.1f}")
    else:
        print(f"训练完成！平均奖励: {np.mean(rewards) if rewards else 0:.1f}")