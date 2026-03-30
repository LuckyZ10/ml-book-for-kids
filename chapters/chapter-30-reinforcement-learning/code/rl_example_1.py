"""
Q-Learning算法实现
基于Q表的价值迭代方法
"""

import numpy as np
import random
from collections import defaultdict
from typing import Dict, Tuple, Callable, Optional
import matplotlib.pyplot as plt


class QTable:
    """
    Q表：存储状态-动作对的价值
    """
    
    def __init__(self, num_states: int, num_actions: int, init_value: float = 0.0):
        """
        初始化Q表
        
        Args:
            num_states: 状态数量
            num_actions: 动作数量
            init_value: 初始Q值
        """
        self.num_states = num_states
        self.num_actions = num_actions
        # 使用字典存储，支持稀疏访问
        self.table = defaultdict(lambda: np.ones(num_actions) * init_value)
    
    def get(self, state: Tuple[int, int], action: Optional[int] = None) -> float:
        """获取Q值"""
        if action is None:
            return self.table[state]
        return self.table[state][action]
    
    def set(self, state: Tuple[int, int], action: int, value: float):
        """设置Q值"""
        self.table[state][action] = value
    
    def update(self, state: Tuple[int, int], action: int, delta: float):
        """更新Q值（增加delta）"""
        self.table[state][action] += delta
    
    def get_best_action(self, state: Tuple[int, int]) -> int:
        """获取当前最优动作"""
        return int(np.argmax(self.table[state]))
    
    def get_max_q(self, state: Tuple[int, int]) -> float:
        """获取当前状态下的最大Q值"""
        return float(np.max(self.table[state]))
    
    def to_array(self, env_size: int) -> np.ndarray:
        """将Q表转换为numpy数组用于可视化"""
        arr = np.zeros((env_size, env_size, self.num_actions))
        for (r, c), values in self.table.items():
            if 0 <= r < env_size and 0 <= c < env_size:
                arr[r, c] = values
        return arr


class QLearningAgent:
    """
    Q-Learning智能体
    
    核心算法：
    Q(s,a) ← Q(s,a) + α * [r + γ·max_a' Q(s',a') - Q(s,a)]
    """
    
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        """
        初始化Q-Learning智能体
        
        Args:
            num_states: 状态数量
            num_actions: 动作数量
            learning_rate: 学习率 α
            discount_factor: 折扣因子 γ
            epsilon: 探索率 ε
            epsilon_decay: ε衰减率
            epsilon_min: 最小ε值
        """
        self.q_table = QTable(num_states, num_actions)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 训练历史
        self.episode_rewards = []
        self.episode_lengths = []
        self.q_value_history = []
    
    def select_action(self, state: Tuple[int, int], training: bool = True) -> int:
        """
        使用ε-贪心策略选择动作
        
        Args:
            state: 当前状态
            training: 是否处于训练模式
        
        Returns:
            选择的动作
        """
        if training and random.random() < self.epsilon:
            # 探索：随机选择
            return random.randint(0, self.q_table.num_actions - 1)
        else:
            # 利用：选择Q值最大的动作
            return self.q_table.get_best_action(state)
    
    def learn(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        done: bool
    ) -> float:
        """
        Q-Learning更新
        
        Q(s,a) ← Q(s,a) + α * [r + γ·max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
        
        Returns:
            TD误差
        """
        current_q = self.q_table.get(state, action)
        
        # 计算目标Q值
        if done:
            # 终止状态，没有未来奖励
            target_q = reward
        else:
            # 非终止状态，加上折扣后的最大未来Q值
            max_next_q = self.q_table.get_max_q(next_state)
            target_q = reward + self.gamma * max_next_q
        
        # 计算TD误差
        td_error = target_q - current_q
        
        # 更新Q值
        new_q = current_q + self.lr * td_error
        self.q_table.set(state, action, new_q)
        
        return td_error
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train_episode(self, env) -> Tuple[float, int]:
        """
        训练一个回合
        
        Args:
            env: 环境实例
        
        Returns:
            (总奖励, 步数)
        """
        state = env.reset()
        total_reward = 0.0
        steps = 0
        
        while True:
            # 选择动作
            action = self.select_action(state, training=True)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 学习
            self.learn(state, action, reward, next_state, done)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # 衰减探索率
        self.decay_epsilon()
        
        # 记录历史
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        
        return total_reward, steps
    
    def train(self, env, num_episodes: int = 1000, verbose: bool = True):
        """
        训练智能体
        
        Args:
            env: 环境实例
            num_episodes: 训练回合数
            verbose: 是否打印进度
        """
        for episode in range(num_episodes):
            reward, steps = self.train_episode(env)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
    
    def evaluate(self, env, num_episodes: int = 10) -> float:
        """
        评估智能体性能（不使用探索）
        
        Args:
            env: 环境实例
            num_episodes: 评估回合数
        
        Returns:
            平均奖励
        """
        rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            total_reward = 0.0
            
            while True:
                action = self.select_action(state, training=False)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            rewards.append(total_reward)
        
        return np.mean(rewards)
    
    def get_policy(self, env) -> Dict[Tuple[int, int], int]:
        """
        提取学到的策略
        
        Returns:
            状态到动作的映射
        """
        policy = {}
        for r in range(env.size):
            for c in range(env.size):
                state = (r, c)
                policy[state] = self.q_table.get_best_action(state)
        return policy
    
    def visualize_q_values(self, env):
        """可视化Q值"""
        q_array = self.q_table.to_array(env.size)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        action_names = ['Up', 'Right', 'Down', 'Left']
        
        for i, (ax, name) in enumerate(zip(axes, action_names)):
            im = ax.imshow(q_array[:, :, i], cmap='RdYlGn', 
                          vmin=q_array.min(), vmax=q_array.max())
            ax.set_title(f'Q-Values: {name}')
            
            # 添加数值标注
            for r in range(env.size):
                for c in range(env.size):
                    text = ax.text(c, r, f'{q_array[r, c, i]:.1f}',
                                  ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=axes, orientation='horizontal', pad=0.1)
        plt.tight_layout()
        plt.savefig('q_learning_q_values.png', dpi=150)
        plt.show()
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 奖励曲线
        axes[0].plot(self.episode_rewards, alpha=0.3, color='blue')
        # 平滑曲线
        if len(self.episode_rewards) > 100:
            smoothed = np.convolve(self.episode_rewards, 
                                  np.ones(100)/100, mode='valid')
            axes[0].plot(range(99, len(self.episode_rewards)), 
                        smoothed, color='red', linewidth=2, label='Smoothed')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Training Rewards')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 回合长度
        axes[1].plot(self.episode_lengths, alpha=0.3, color='green')
        if len(self.episode_lengths) > 100:
            smoothed = np.convolve(self.episode_lengths, 
                                  np.ones(100)/100, mode='valid')
            axes[1].plot(range(99, len(self.episode_lengths)), 
                        smoothed, color='darkgreen', linewidth=2)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Steps')
        axes[1].set_title('Episode Lengths')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('q_learning_training.png', dpi=150)
        plt.show()


def demonstrate_q_learning():
    """演示Q-Learning训练过程"""
    from chapter30_env import GridWorld
    
    print("=" * 60)
    print("Q-Learning 算法演示")
    print("=" * 60)
    
    # 创建环境
    env = GridWorld(size=4, seed=42)
    
    # 创建智能体
    agent = QLearningAgent(
        num_states=env.num_states,
        num_actions=env.num_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,        # 初始完全探索
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    print("\n开始训练...")
    print(f"初始探索率 ε = {agent.epsilon}")
    
    # 训练
    agent.train(env, num_episodes=2000, verbose=True)
    
    # 评估
    print("\n" + "=" * 60)
    print("训练完成，评估性能...")
    avg_reward = agent.evaluate(env, num_episodes=100)
    print(f"平均奖励: {avg_reward:.2f}")
    
    # 显示学到的策略
    print("\n学到的策略:")
    policy = agent.get_policy(env)
    for r in range(env.size):
        row = []
        for c in range(env.size):
            action = policy[(r, c)]
            row.append(GridWorld.ACTIONS[action])
        print(' '.join(row))
    
    # 绘制训练历史
    agent.plot_training_history()
    
    return agent, env


if __name__ == "__main__":
    demonstrate_q_learning()