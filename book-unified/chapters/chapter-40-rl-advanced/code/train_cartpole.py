"""
CartPole训练示例
第40章：强化学习——试错中成长
"""

import numpy as np
import matplotlib.pyplot as plt
import gym
from rl_basic import QLearningAgent, DQNAgent


def discretize_state(state, bins=10):
    """将连续状态离散化（用于Q-Learning）"""
    # CartPole状态：位置、速度、角度、角速度
    bounds = [
        (-2.4, 2.4),      # 位置范围
        (-3.0, 3.0),      # 速度范围（估计）
        (-0.209, 0.209),  # 角度范围（约12度）
        (-3.0, 3.0)       # 角速度范围（估计）
    ]
    
    discrete_state = []
    for i, (val, (low, high)) in enumerate(zip(state, bounds)):
        # 裁剪到范围内
        val = np.clip(val, low, high)
        # 离散化
        bucket = int((val - low) / (high - low) * (bins - 1))
        discrete_state.append(bucket)
    
    # 将4维离散状态编码为1维
    return discrete_state[0] * bins**3 + discrete_state[1] * bins**2 + discrete_state[2] * bins + discrete_state[3]


def train_q_learning():
    """使用Q-Learning训练CartPole"""
    print("=" * 60)
    print("Q-Learning on CartPole")
    print("=" * 60)
    
    env = gym.make('CartPole-v1')
    
    # 参数
    n_bins = 10
    n_states = n_bins ** 4  # 10^4 = 10000
    n_actions = env.action_space.n
    
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1
    )
    
    episodes = 500
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        state = discretize_state(state, n_bins)
        score = 0
        
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = discretize_state(next_state, n_bins)
            
            # 修改奖励：更快达到目标
            if not done:
                reward = 1.0
            else:
                reward = -1.0 if score < 500 else 0.0
            
            agent.learn(state, action, reward, next_state, done)
            
            score += 1
            state = next_state
            
            if done:
                break
        
        scores.append(score)
        
        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"Episode {episode}, Avg Score: {avg_score:.2f}")
    
    env.close()
    
    # 可视化
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.plot(np.convolve(scores, np.ones(50)/50, mode='valid'), 'r', label='Moving Avg (50)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Q-Learning on CartPole')
    plt.legend()
    plt.savefig('q_learning_cartpole.png', dpi=150)
    plt.show()
    
    return agent, scores


def train_dqn():
    """使用DQN训练CartPole"""
    print("=" * 60)
    print("DQN on CartPole")
    print("=" * 60)
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=10000,
        batch_size=64
    )
    
    episodes = 500
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        score = 0
        
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 学习
            loss = agent.learn()
            
            score += 1
            state = next_state
            
            if done:
                break
        
        # 每10个episode更新目标网络
        if episode % 10 == 0:
            agent.update_target_network()
        
        scores.append(score)
        
        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:])
            epsilon = agent.epsilon
            print(f"Episode {episode}, Avg Score: {avg_score:.2f}, Epsilon: {epsilon:.3f}")
    
    env.close()
    
    # 可视化
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.plot(np.convolve(scores, np.ones(50)/50, mode='valid'), 'r', label='Moving Avg (50)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DQN on CartPole')
    plt.legend()
    plt.savefig('dqn_cartpole.png', dpi=150)
    plt.show()
    
    return agent, scores


def compare_algorithms():
    """比较Q-Learning和DQN"""
    print("=" * 60)
    print("Comparing Q-Learning vs DQN")
    print("=" * 60)
    
    _, scores_ql = train_q_learning()
    _, scores_dqn = train_dqn()
    
    # 可视化对比
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores_ql, alpha=0.6, label='Q-Learning')
    plt.plot(np.convolve(scores_ql, np.ones(50)/50, mode='valid'), 'b', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Q-Learning')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(scores_dqn, alpha=0.6, label='DQN', color='orange')
    plt.plot(np.convolve(scores_dqn, np.ones(50)/50, mode='valid'), 'r', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DQN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150)
    plt.show()
    
    print("\nComparison complete!")
    print(f"Q-Learning final avg (last 50): {np.mean(scores_ql[-50:]):.2f}")
    print(f"DQN final avg (last 50): {np.mean(scores_dqn[-50:]):.2f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'q':
            train_q_learning()
        elif sys.argv[1] == 'dqn':
            train_dqn()
        elif sys.argv[1] == 'compare':
            compare_algorithms()
    else:
        print("Usage: python train_cartpole.py [q|dqn|compare]")
        print("  q       - Train Q-Learning")
        print("  dqn     - Train DQN")
        print("  compare - Compare both algorithms")
