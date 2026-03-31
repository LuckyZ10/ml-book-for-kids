"""
Actor-Critic算法实现
包含A2C（同步）版本
"""

import numpy as np
import random
from typing import Tuple, List


class ValueNetwork:
    """价值网络：估计V(s)"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        self.fc1 = LinearLayer(state_dim, hidden_dim)
        self.relu1 = ReLU()
        self.fc2 = LinearLayer(hidden_dim, hidden_dim)
        self.relu2 = ReLU()
        self.fc3 = LinearLayer(hidden_dim, 1)  # 输出单个价值
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        x = self.fc3.forward(x)
        return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = self.fc3.backward(grad_output)
        grad = self.relu2.backward(grad)
        grad = self.fc2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.fc1.backward(grad)
        return grad
    
    def update(self, lr: float):
        self.fc3.update(lr)
        self.fc2.update(lr)
        self.fc1.update(lr)


class ActorCriticAgent:
    """
    Advantage Actor-Critic (A2C) 智能体
    
    Actor: 策略网络 π(a|s)
    Critic: 价值网络 V(s)
    
    更新规则：
    - Actor: θ ← θ + α · ∇_θ log π_θ(a|s) · A(s,a)
    - Critic: φ ← φ - α · ∇_φ (TD_error)²
    
    其中 A(s,a) ≈ r + γ·V(s') - V(s) 是优势函数
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = 0.001,
        critic_lr: float = 0.005,
        discount_factor: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        hidden_dim: int = 64
    ):
        """
        初始化A2C智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            actor_lr: Actor学习率
            critic_lr: Critic学习率
            discount_factor: 折扣因子
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数（鼓励探索）
        """
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.critic = ValueNetwork(state_dim, hidden_dim)
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = discount_factor
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.episode_rewards = []
    
    def _state_to_vector(self, state: Tuple[int, int], env_size: int) -> np.ndarray:
        """状态转向量"""
        vec = np.zeros(env_size * env_size)
        idx = state[0] * env_size + state[1]
        vec[idx] = 1.0
        return vec.reshape(1, -1)
    
    def select_action(self, state: Tuple[int, int], env_size: int) -> Tuple[int, float, float]:
        """
        选择动作
        
        Returns:
            (动作, 对数概率, 状态价值)
        """
        state_vec = self._state_to_vector(state, env_size)
        
        # Actor选择动作
        action, log_prob = self.actor.sample_action(state_vec)
        
        # Critic评估状态价值
        value = self.critic.forward(state_vec)[0, 0]
        
        return action, log_prob, value
    
    def train_episode(self, env) -> Tuple[float, int]:
        """
        训练一个回合（A2C）
        
        使用n步回报或单步TD更新
        """
        state = env.reset()
        
        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        
        # 收集轨迹
        while True:
            state_vec = self._state_to_vector(state, env.size)
            
            action, log_prob, value = self.select_action(state, env.size)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state_vec)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            
            state = next_state
            
            if done:
                # 计算终止状态的bootstrap价值
                next_value = 0.0
                break
        
        # 计算回报和优势
        returns = []
        advantages = []
        G = 0
        
        # 从后向前计算
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns.insert(0, G)
            
            # 优势 = 回报 - 价值估计
            advantage = G - values[t]
            advantages.insert(0, advantage)
        
        returns = np.array(returns)
        advantages = np.array(advantages)
        
        # 标准化优势（减小方差）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 更新Actor和Critic
        actor_loss_total = 0
        critic_loss_total = 0
        
        for t in range(len(states)):
            # Critic损失: (return - value)²
            value_pred = self.critic.forward(states[t])
            td_error = returns[t] - value_pred[0, 0]
            critic_loss = td_error ** 2
            critic_loss_total += critic_loss
            
            # Critic反向传播和更新
            critic_grad = np.array([[-2 * td_error]])
            self.critic.backward(critic_grad)
            self.critic.update(self.critic_lr)
            
            # Actor损失: -log_prob * advantage
            # 策略梯度: ∇_θ log π_θ(a|s) * A(s,a)
            probs = self.actor.get_action_probs(states[t])
            actor_grad = probs.copy()
            actor_grad[0, actions[t]] -= 1
            actor_grad *= -advantages[t]  # 乘以优势
            
            # 添加熵正则化（鼓励探索）
            entropy_grad = -self.entropy_coef * (np.log(probs + 1e-10) + 1)
            actor_grad += entropy_grad
            
            self.actor.backward(actor_grad)
            self.actor.update(self.actor_lr)
            
            actor_loss_total += -log_probs[t] * advantages[t]
        
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
                probs = self.actor.get_action_probs(state_vec)
                action = np.argmax(probs[0])
                
                state, reward, done, _ = env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            rewards.append(total_reward)
        
        return np.mean(rewards)


def demonstrate_actor_critic():
    """演示Actor-Critic"""
    from chapter30_env import GridWorld
    
    print("=" * 60)
    print("Actor-Critic (A2C) 算法演示")
    print("=" * 60)
    
    env = GridWorld(size=4, seed=42)
    
    agent = ActorCriticAgent(
        state_dim=env.num_states,
        action_dim=env.num_actions,
        actor_lr=0.001,
        critic_lr=0.005,
        discount_factor=0.99,
        value_coef=0.5,
        entropy_coef=0.01
    )
    
    print("\n开始训练Actor-Critic...")
    agent.train(env, num_episodes=2000, verbose=True)
    
    print("\n评估...")
    avg_reward = agent.evaluate(env, num_episodes=100)
    print(f"平均奖励: {avg_reward:.2f}")
    
    return agent, env


if __name__ == "__main__":
    demonstrate_actor_critic()