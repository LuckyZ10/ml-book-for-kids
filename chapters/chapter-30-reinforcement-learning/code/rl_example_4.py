"""
深度Q网络（DQN）实现
使用PyTorch/NumPy实现完整的DQN算法
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, List, Optional
import copy


# ============ 神经网络实现（纯NumPy版本）============

class LinearLayer:
    """全连接层"""
    
    def __init__(self, in_features: int, out_features: int):
        # Xavier初始化
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weights = np.random.uniform(-limit, limit, (in_features, out_features))
        self.bias = np.zeros(out_features)
        
        # 梯度
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        
        # 缓存
        self.input = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.input = x
        return x @ self.weights + self.bias
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """反向传播"""
        self.grad_weights = self.input.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)
        return grad_output @ self.weights.T
    
    def update(self, lr: float):
        """参数更新"""
        self.weights -= lr * self.grad_weights
        self.bias -= lr * self.grad_bias


class ReLU:
    """ReLU激活函数"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.maximum(0, x)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.mask


class QNetwork:
    """
    Q网络：近似Q(s, a)
    简单版本：输入状态，输出每个动作的Q值
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        初始化Q网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            hidden_dim: 隐藏层维度
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 网络层
        self.fc1 = LinearLayer(state_dim, hidden_dim)
        self.relu1 = ReLU()
        self.fc2 = LinearLayer(hidden_dim, hidden_dim)
        self.relu2 = ReLU()
        self.fc3 = LinearLayer(hidden_dim, action_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        x = self.fc3.forward(x)
        return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """反向传播"""
        grad = self.fc3.backward(grad_output)
        grad = self.relu2.backward(grad)
        grad = self.fc2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.fc1.backward(grad)
        return grad
    
    def update(self, lr: float):
        """参数更新"""
        self.fc3.update(lr)
        self.fc2.update(lr)
        self.fc1.update(lr)
    
    def copy_from(self, other: 'QNetwork'):
        """从其他网络复制参数"""
        self.fc1.weights = other.fc1.weights.copy()
        self.fc1.bias = other.fc1.bias.copy()
        self.fc2.weights = other.fc2.weights.copy()
        self.fc2.bias = other.fc2.bias.copy()
        self.fc3.weights = other.fc3.weights.copy()
        self.fc3.bias = other.fc3.bias.copy()


# ============ DQN组件 ============

class ReplayBuffer:
    """
    经验回放缓冲区
    存储和采样经验 (s, a, r, s', done)
    """
    
    def __init__(self, capacity: int = 10000):
        """
        初始化回放缓冲区
        
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        随机采样一批经验
        
        Returns:
            (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN智能体
    
    核心算法：
    1. 使用神经网络 Q(s, a; θ) 近似Q函数
    2. 经验回放存储和采样
    3. 目标网络计算稳定的目标
    4. 最小化 MSE(Q(s,a;θ), r + γ·max_a' Q(s',a'; θ⁻))
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        hidden_dim: int = 64
    ):
        """
        初始化DQN智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            learning_rate: 学习率
            discount_factor: 折扣因子 γ
            epsilon: 探索率
            epsilon_decay: ε衰减
            epsilon_min: 最小ε
            buffer_capacity: 回放缓冲区容量
            batch_size: 训练批量大小
            target_update_freq: 目标网络更新频率
            hidden_dim: 隐藏层维度
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # 网络
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_network.copy_from(self.q_network)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 训练计数
        self.train_step = 0
        
        # 历史记录
        self.episode_rewards = []
        self.losses = []
    
    def _state_to_vector(self, state: Tuple[int, int], env_size: int) -> np.ndarray:
        """将状态转换为向量"""
        # 简单的one-hot编码
        vec = np.zeros(env_size * env_size)
        idx = state[0] * env_size + state[1]
        vec[idx] = 1.0
        return vec
    
    def select_action(self, state: Tuple[int, int], env_size: int, training: bool = True) -> int:
        """
        ε-贪心策略选择动作
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_vec = self._state_to_vector(state, env_size).reshape(1, -1)
        q_values = self.q_network.forward(state_vec)
        return int(np.argmax(q_values))
    
    def learn(self, env_size: int) -> Optional[float]:
        """
        从回放缓冲区学习
        
        Returns:
            损失值（如果缓冲区不够则返回None）
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 采样
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # 转换为向量
        state_vecs = np.array([self._state_to_vector(s, env_size) for s in states])
        next_state_vecs = np.array([self._state_to_vector(s, env_size) for s in next_states])
        
        # 计算当前Q值
        current_q = self.q_network.forward(state_vecs)
        current_q_values = current_q[np.arange(self.batch_size), actions]
        
        # 计算目标Q值
        next_q = self.target_network.forward(next_state_vecs)
        max_next_q = np.max(next_q, axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        # 计算损失和梯度
        td_errors = targets - current_q_values  # [batch_size]
        loss = np.mean(td_errors ** 2)
        self.losses.append(loss)
        
        # 反向传播
        grad = np.zeros_like(current_q)
        grad[np.arange(self.batch_size), actions] = -2 * td_errors / self.batch_size
        
        self.q_network.backward(grad)
        self.q_network.update(self.lr)
        
        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.copy_from(self.q_network)
        
        return loss
    
    def train_episode(self, env) -> Tuple[float, int]:
        """训练一个回合"""
        state = env.reset()
        total_reward = 0.0
        steps = 0
        
        while True:
            # 选择并执行动作
            action = self.select_action(state, env.size, training=True)
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            # 学习
            self.learn(env.size)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.episode_rewards.append(total_reward)
        return total_reward, steps
    
    def train(self, env, num_episodes: int = 1000, verbose: bool = True):
        """训练"""
        for episode in range(num_episodes):
            reward, steps = self.train_episode(env)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                recent_loss = np.mean(self.losses[-100:]) if self.losses else 0
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Loss: {recent_loss:.4f}, "
                      f"ε: {self.epsilon:.3f}")
    
    def evaluate(self, env, num_episodes: int = 10) -> float:
        """评估"""
        rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            total_reward = 0.0
            
            while True:
                action = self.select_action(state, env.size, training=False)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            rewards.append(total_reward)
        
        return np.mean(rewards)


def demonstrate_dqn():
    """演示DQN"""
    from chapter30_env import GridWorld
    
    print("=" * 60)
    print("深度Q网络（DQN）演示")
    print("=" * 60)
    
    env = GridWorld(size=4, seed=42)
    
    agent = DQNAgent(
        state_dim=env.num_states,
        action_dim=env.num_actions,
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        buffer_capacity=5000,
        batch_size=32,
        target_update_freq=50
    )
    
    print("\n开始训练DQN...")
    agent.train(env, num_episodes=2000, verbose=True)
    
    print("\n评估...")
    avg_reward = agent.evaluate(env, num_episodes=100)
    print(f"平均奖励: {avg_reward:.2f}")
    
    return agent, env


if __name__ == "__main__":
    demonstrate_dqn()