"""
策略梯度方法实现
包含REINFORCE和Actor-Critic
"""

import numpy as np
import random
from typing import Tuple, List, Optional


class PolicyNetwork:
    """
    策略网络：输出动作概率分布 π(a|s)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        # 层
        self.fc1 = LinearLayer(state_dim, hidden_dim)
        self.relu1 = ReLU()
        self.fc2 = LinearLayer(hidden_dim, hidden_dim)
        self.relu2 = ReLU()
        self.fc3 = LinearLayer(hidden_dim, action_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播，输出logits"""
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        x = self.fc3.forward(x)
        return x
    
    def get_action_probs(self, x: np.ndarray) -> np.ndarray:
        """获取动作概率（softmax）"""
        logits = self.forward(x)
        # 数值稳定的softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    def sample_action(self, x: np.ndarray) -> Tuple[int, float]:
        """
        根据策略采样动作
        
        Returns:
            (动作, 该动作的对数概率)
        """
        probs = self.get_action_probs(x)
        action = np.random.choice(len(probs[0]), p=probs[0])
        log_prob = np.log(probs[0, action] + 1e-10)
        return action, log_prob
    
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


class REINFORCEAgent:
    """
    REINFORCE算法（蒙特卡洛策略梯度）
    
    核心更新：
    θ ← θ + α · Σ_t ∇_θ log π_θ(a_t|s_t) · G_t
    
    其中 G_t = Σ_{k=0}^{T-t} γ^k · r_{t+k} 是累积回报
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        hidden_dim: int = 64
    ):
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.lr = learning_rate
        self.gamma = discount_factor
        
        self.episode_rewards = []
    
    def _state_to_vector(self, state: Tuple[int, int], env_size: int) -> np.ndarray:
        """状态转向量"""
        vec = np.zeros(env_size * env_size)
        idx = state[0] * env_size + state[1]
        vec[idx] = 1.0
        return vec.reshape(1, -1)
    
    def select_action(self, state: Tuple[int, int], env_size: int) -> Tuple[int, float]:
        """
        根据策略选择动作
        
        Returns:
            (动作, 对数概率)
        """
        state_vec = self._state_to_vector(state, env_size)
        return self.policy.sample_action(state_vec)
    
    def compute_returns(self, rewards: List[float]) -> np.ndarray:
        """
        计算每个时刻的累积回报
        
        G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
        """
        returns = np.zeros(len(rewards))
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns
    
    def train_episode(self, env) -> Tuple[float, int]:
        """
        训练一个回合（REINFORCE）
        """
        state = env.reset()
        
        # 存储轨迹
        states = []
        actions = []
        log_probs = []
        rewards = []
        
        while True:
            state_vec = self._state_to_vector(state, env.size)
            action, log_prob = self.policy.sample_action(state_vec)
            
            next_state, reward, done, _ = env.step(action)
            
            states.append(state_vec)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            
            state = next_state
            
            if done:
                break
        
        # 计算累积回报
        returns = self.compute_returns(rewards)
        
        # 标准化回报（减小方差）
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 策略梯度更新
        for t in range(len(states)):
            # 损失 = -log_prob * return（我们要最大化return，所以最小化负return）
            loss_grad = -log_probs[t] * returns[t]
            
            # 对logits的梯度（简化版，实际应该是softmax的梯度）
            probs = self.policy.get_action_probs(states[t])
            grad_logits = probs.copy()
            grad_logits[0, actions[t]] -= 1
            grad_logits *= -returns[t]  # 乘以return
            
            self.policy.backward(grad_logits)
            self.policy.update(self.lr)
        
        total_reward = sum(rewards)
        self.episode_rewards.append(total_reward)
        
        return total_reward, len(rewards)
    
    def train(self, env, num_episodes: int = 1000, verbose: bool = True):
        """训练"""
        for episode in range(num_episodes):
            reward, steps = self.train_episode(env)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}")
    
    def evaluate(self, env, num_episodes: int = 10) -> float:
        """评估"""
        rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            total_reward = 0.0
            
            while True:
                state_vec = self._state_to_vector(state, env.size)
                probs = self.policy.get_action_probs(state_vec)
                action = np.argmax(probs[0])  # 贪心选择
                
                state, reward, done, _ = env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            rewards.append(total_reward)
        
        return np.mean(rewards)


# 复用之前的层定义
class LinearLayer:
    """全连接层"""
    def __init__(self, in_features: int, out_features: int):
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weights = np.random.uniform(-limit, limit, (in_features, out_features))
        self.bias = np.zeros(out_features)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        self.input = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return x @ self.weights + self.bias
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.grad_weights = self.input.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)
        return grad_output @ self.weights.T
    
    def update(self, lr: float):
        self.weights -= lr * self.grad_weights
        self.bias -= lr * self.grad_bias


class ReLU:
    """ReLU激活"""
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.maximum(0, x)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.mask


def demonstrate_reinforce():
    """演示REINFORCE"""
    from chapter30_env import GridWorld
    
    print("=" * 60)
    print("REINFORCE算法演示")
    print("=" * 60)
    
    env = GridWorld(size=4, seed=42)
    
    agent = REINFORCEAgent(
        state_dim=env.num_states,
        action_dim=env.num_actions,
        learning_rate=0.001,
        discount_factor=0.99
    )
    
    print("\n开始训练REINFORCE...")
    agent.train(env, num_episodes=2000, verbose=True)
    
    print("\n评估...")
    avg_reward = agent.evaluate(env, num_episodes=100)
    print(f"平均奖励: {avg_reward:.2f}")
    
    return agent, env


if __name__ == "__main__":
    demonstrate_reinforce()