# 第三十一章 深度强化学习进阶 练习题

## 练习题 1: DQN经验回放实现 (⭐⭐)

**目标**: 理解Experience Replay机制

**题目**: 
实现一个优先经验回放缓冲区：

```python
import numpy as np
from collections import deque

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    def push(self, transition):
        # transition: (state, action, reward, next_state, done)
        # 新样本设为最高优先级
        pass
    
    def sample(self, batch_size, beta=0.4):
        # 根据优先级采样
        # 返回样本和重要性权重
        pass
    
    def update_priorities(self, indices, priorities):
        # 根据TD误差更新优先级
        pass
```

**思考问题**: 
- 为什么经验回放能提高样本效率？
- α=0和α=1分别代表什么？

---

## 练习题 2: 目标网络的作用 (⭐)

**目标**: 理解目标网络的稳定性作用

**题目**: 
比较两种更新方式：

**方式A**（无目标网络）:
```
Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
```

**方式B**（有目标网络）:
```
target = r + γ*max(Q_target(s',a'))
Q(s,a) = Q(s,a) + α[target - Q(s,a)]
# 每隔N步: Q_target = Q
```

**任务**: 
1. 说明方式A的问题（追逐移动目标）
2. 解释方式B如何缓解这个问题
3. 目标网络的更新频率如何影响学习？

---

## 练习题 3: Actor-Critic实现 (⭐⭐⭐)

**目标**: 实现A2C算法

**题目**: 
实现Advantage Actor-Critic：

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor: 输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic: 输出状态价值
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

class A2CAgent:
    def __init__(self, state_dim, action_dim):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def update(self, states, actions, rewards, next_states, dones):
        # 计算优势函数 A = R + γV(s') - V(s)
        # 计算Actor损失（策略梯度）
        # 计算Critic损失（MSE）
        # 反向传播
        pass
```

**要求**: 
- 实现完整的update方法
- 在CartPole上测试
- 绘制学习曲线

---

## 练习题 4: PPO裁剪目标 (⭐⭐⭐)

**目标**: 理解PPO的核心创新

**题目**: 
实现PPO的裁剪目标函数：

```python
def ppo_loss(old_probs, new_probs, advantages, epsilon=0.2):
    """
    计算PPO裁剪损失
    
    参数:
    - old_probs: 旧策略的动作概率
    - new_probs: 新策略的动作概率
    - advantages: 优势函数估计
    - epsilon: 裁剪参数
    
    返回:
    - PPO损失值
    """
    # 计算概率比率 r(θ) = π_new / π_old
    ratio = new_probs / old_probs
    
    # 未裁剪的目标
    surr1 = ratio * advantages
    
    # 裁剪后的目标
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    
    # 取最小值
    loss = -torch.min(surr1, surr2).mean()
    
    return loss
```

**实验**: 
1. 绘制不同epsilon值（0.1, 0.2, 0.3）的裁剪效果
2. 解释为什么取min而不是max
3. 解释为什么这样能防止策略更新过大

---

## 练习题 5: 连续动作空间 (⭐⭐⭐)

**目标**: 处理连续控制问题

**题目**: 
修改Actor网络输出连续动作（高斯分布）：

```python
class ContinuousActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # 输出均值和对数标准差
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
```

**任务**: 
1. 理解为什么连续动作需要概率分布
2. 在Pendulum环境上测试
3. 比较确定性策略（DDPG）和随机策略（SAC）

---

## 练习题 6: Multi-Step Bootstrap (⭐⭐)

**目标**: 理解n步回报

**题目**: 
实现n步Q-Learning：

```python
def compute_nstep_return(rewards, q_values, n=3, gamma=0.99):
    """
    计算n步回报
    
    参数:
    - rewards: [r_1, r_2, ..., r_T]
    - q_values: [Q(s_1), Q(s_2), ..., Q(s_T)]
    - n: bootstrap步数
    
    返回:
    - n步回报序列
    """
    returns = []
    T = len(rewards)
    
    for t in range(T):
        # 计算G_t = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n Q(s_{t+n})
        pass
    
    return returns
```

**实验**: 
- 比较n=1, 3, 10的效果
- 分析n步回报与TD(λ)的关系

---

## 练习题 7: Dueling DQN (⭐⭐⭐)

**目标**: 理解Dueling架构

**题目**: 
实现Dueling DQN网络：

```python
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # 共享特征提取
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values
```

**思考问题**: 
- Dueling DQN在什么情况下优势最明显？
- 为什么优势要减去均值？

---

## 练习题 8: 策略梯度方差 (⭐⭐⭐)

**目标**: 理解基线函数的作用

**题目**: 
比较两种策略梯度估计：

**无基线**:
$$\nabla J = \mathbb{E}[G_t \nabla \log \pi(a_t|s_t)]$$

**有基线**:
$$\nabla J = \mathbb{E}[(G_t - b(s_t)) \nabla \log \pi(a_t|s_t)]$$

**任务**: 
1. 证明基线不会引入偏差
2. 解释为什么能减少方差
3. 实现自适应基线（价值函数）

---

## 练习题 9: Atari游戏挑战 (⭐⭐⭐⭐⭐)

**目标**: 完整DQM实现

**题目**: 
实现完整的DQN（或Double DQN/Dueling DQN）玩Atari游戏：

**组件**: 
1. **预处理**: 帧堆叠、跳帧、灰度化
2. **网络架构**: 卷积+全连接
3. **训练循环**: 采样→计算目标→更新
4. **评估**: 定期测试（无探索）

**环境**: 
```python
import gym
env = gym.make('PongNoFrameskip-v4')
```

**超参数**: 
- Replay buffer: 100k
- Batch size: 32
- Target update: 每1000步
- Epsilon decay: 1.0 → 0.01

**交付物**: 
- 完整代码
- 训练曲线（奖励vs回合）
- 游戏视频（可选）

---

## 参考答案

### 练习1 优先经验回放关键代码

```python
def sample(self, batch_size, beta=0.4):
    # 计算采样概率
    priorities = self.priorities[:len(self.buffer)]
    probs = priorities ** self.alpha
    probs /= probs.sum()
    
    # 采样
    indices = np.random.choice(len(self.buffer), batch_size, p=probs)
    samples = [self.buffer[idx] for idx in indices]
    
    # 计算重要性权重
    weights = (len(self.buffer) * probs[indices]) ** (-beta)
    weights /= weights.max()
    
    return samples, indices, weights
```

### 练习4 PPO vs TRPO

PPO优势:
- 不需要计算Fisher信息矩阵
- 实现更简单
- 训练更稳定
- 通常性能相当或更好

---

**学习建议**: 
- 练习1-3必做，理解DQN和Actor-Critic
- 练习4-6进阶，理解PPO和连续控制
- 练习7-9实战，掌握前沿算法

**推荐资源**: 
- OpenAI Spinning Up
- Stable-Baselines3文档
- "Deep Reinforcement Learning" (Sergey Levine, Berkeley)
