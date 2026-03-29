# 第三十一章：深度强化学习前沿——从智能体到超级智能

*"智能的本质不在于知道答案，而在于知道如何寻找答案。"*

---

## 引言：通往超级智能的阶梯

还记得我们在第三十章学习的Q-Learning和DQN吗？那些算法让AI能够在Atari游戏中达到人类水平，让机器学会了玩《吃豆人》和《太空入侵者》。但是，当我们想让机器人学会走路、让机械臂学会抓取物体、让自动驾驶汽车学会控制方向盘时，这些算法却遇到了一个根本性的问题：**它们只能处理离散的动作**。

想象一下，你在学习骑自行车。如果只能做"向左转45度"或"向右转45度"这样的离散选择，你能顺利骑行吗？显然不行！真正的控制需要连续、平滑的动作——你需要连续地调整车把的角度、连续地控制蹬踏的力度。

这就是**深度强化学习（Deep Reinforcement Learning）**的新篇章所要解决的问题。在本章中，我们将探索一系列革命性的算法，它们让AI能够在连续的动作空间中优雅地导航，像一位经验丰富的骑手那样，做出流畅而精准的决策。

### 为什么需要新算法？

让我们用图31-1来理解离散动作和连续动作的区别：

```
图31-1：离散动作 vs 连续动作

离散动作（如Atari游戏）:                    连续动作（如机器人控制）:
                                             
按键选择:                                    方向盘角度:
┌─────────────────────────────────────┐     ┌─────────────────────────────────────┐
│   ↑   │  ←  ■  →  │   ↓   │   🔥   │     │  -90°     0°      +90°              │
└─────────────────────────────────────┘     └────┼────┼────┼────┼────┼────┼────┘
                                                 │    │    │    │    │    │
动作空间: A = {上, 下, 左, 右, 发射}              ↑    │    ↑    │    ↑
(有限、离散)                                 左转25°  直行  右转15°  右转45°  右转60°
                                             (无限可能、连续)

适用场景:                                    适用场景:
• 游戏AI (围棋、Atari)                       • 机器人运动控制
• 棋类游戏                                   • 自动驾驶
• 网格世界导航                               • 机械臂抓取
• 离散决策问题                               • 无人机飞行
                                             • 连续控制问题
```

Q-Learning和DQN通过为每个离散动作学习一个Q值来工作。但对于连续动作，我们无法为无限多个可能的动作都学习一个Q值——计算上是不可能的！

这就是为什么我们需要全新的算法架构。本章将介绍的五种核心算法，每一种都像是为解决特定挑战而生的英雄：

| 算法 | 比喻 | 核心突破 | 适用场景 |
|------|------|----------|----------|
| **DDPG** | 精准射手 🎯 | 确定性策略 + 演员-评论家 | 连续控制入门 |
| **TD3** | 严谨的科学家 🔬 | 解决过估计问题 | 高精度连续控制 |
| **SAC** | 灵活的探险家 🧭 | 最大熵 + 自动温度调节 | 样本高效学习 |
| **PPO** | 稳健登山者 ⛰️ | 裁剪目标 + 稳定训练 | OpenAI主打算法 |
| **A3C** | 蜂群智者 🐝 | 异步分布式训练 | 大规模并行学习 |

让我们一起踏上这段探索之旅，从DDPG的优雅简洁开始，一步步走向强化学习的巅峰！

---

## 31.1 深度强化学习概述：从DQN到现代算法

### 31.1.1 进化的阶梯：算法的演进历程

让我们用一棵进化树来理解深度强化学习的发展历程（见图31-2）：

```
图31-2：深度强化学习算法进化树

                         ┌─────────────────────────────────────────┐
                         │      深度强化学习进化树 (2013-2020)       │
                         └─────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │                                           │
               【价值方法】                                  【策略方法】
         (Value-Based Methods)                          (Policy-Based Methods)
                    │                                           │
         ┌──────────┴──────────┐                    ┌──────────┴──────────┐
         │                     │                    │                     │
      DQN (2013)          DDPG (2015)           REINFORCE         A3C/A2C (2016)
         │                     │                (Williams 1992)           │
    ┌────┴────┐           ┌────┴────┐                                    │
    │         │           │         │                              TRPO (2015)
 Double    Dueling       TD3      SAC                                    │
 DQN      DQN (2016)  (2018)  (2018)                               ┌────┴────┐
(2015)    /            │         │                                  │         │
          │            │         └──────────────────────────── PPO (2017)
          │            │         (最大熵方法)                   (OpenAI主打算法)
          │            │
          │            └──── Rainbow DQN (2017)
          │                  (DQN改进大全)
          │
          └──────────────────────────────────────────────
                              │
                         【演员-评论家方法】
                    (Actor-Critic Methods - 混合)
                              │
                    ┌─────────┴─────────┐
                    │                   │
               同策略(On-Policy)     异策略(Off-Policy)
                    │                   │
                 A2C/PPO            DDPG/TD3/SAC
                 (样本效率较低)      (样本效率较高)
```

这张图展示了三个主要的发展方向：

1. **价值方法**：从DQN开始，学习Q值函数，然后从中派生出策略
2. **策略方法**：直接学习策略函数，通过梯度上升优化
3. **演员-评论家方法**：结合了两者，用评论家指导演员的学习

### 31.1.2 连续动作空间的挑战

为什么连续动作空间如此困难？让我们深入理解其中的数学本质。

#### 离散动作的Q-Learning

在DQN中，我们对每个离散动作都有一个输出：

$$Q(s, a_1), Q(s, a_2), ..., Q(s, a_n)$$

选择动作非常简单：

$$a^* = \arg\max_a Q(s, a)$$

这是一个在有穷集合上的优化问题，计算复杂度为 $O(|A|)$。

#### 连续动作的困境

但在连续空间中，动作 $a$ 是一个实数向量，例如：

$$a \in \mathbb{R}^n, \quad \text{其中每个维度} a_i \in [-1, 1]$$

可能的动作有无限多个！我们无法为每个动作都学习一个Q值。

解决这个问题的关键洞察来自一个简单而深刻的想法：**如果我们有一个函数，可以直接输出给定状态下的最优动作呢？**

这就是**确定性策略梯度（Deterministic Policy Gradient）**的核心思想。

### 31.1.3 演员-评论家架构的统一视角

让我们用费曼式的生活化比喻来理解演员-评论家架构。

#### 🎭 比喻：戏剧学校

想象一个戏剧学校，里面有两类学生在训练：

**演员（Actor）**：负责在舞台上表演。他学习"在什么情况下该做什么动作"。演员的表演风格就是**策略** $\pi(a|s)$。

**评论家（Critic）**：坐在台下观看表演，负责评价"这个动作在这个情境下有多好"。评论家的评价标准就是**价值函数** $V(s)$ 或 $Q(s,a)$。

训练过程就像这样：
1. 演员上台表演（执行动作）
2. 评论家给出评价（计算优势函数）
3. 演员根据评价改进表演（策略梯度更新）
4. 评论家也从观众的反应中学习（价值函数更新）

这是一个相互促进的过程——好的评论家能指导演员更快进步，而演员的表现越好，评论家也越容易做出准确评价。

#### 数学形式

演员-评论家方法的通用形式包含两个网络：

**演员网络**（策略）：
$$\pi_\theta(a|s) \quad \text{或} \quad \mu_\theta(s) \rightarrow a$$

**评论家网络**（价值）：
$$Q_\phi(s, a) \quad \text{或} \quad V_\phi(s)$$

关键区别：
- **随机策略**（如PPO、A3C）：输出动作的概率分布 $\pi_\theta(a|s)$
- **确定性策略**（如DDPG、TD3）：直接输出动作 $\mu_\theta(s) = a$

### 31.1.4 同策略 vs 异策略：样本效率的权衡

在深入具体算法之前，我们需要理解一个关键的概念区分：

```
图31-3：同策略 vs 异策略学习

同策略 (On-Policy)                    异策略 (Off-Policy)
─────────────────────────────────    ─────────────────────────────────
                                     
数据收集  ←────→  策略更新            经验回放池  ←────  行为策略 μ'
   ↓                ↓                      ↓               │
同一个策略只能学习                      目标策略 μ 可以学习
自己的经验                             任何策略产生的经验
                                     
┌─────────────────────────────┐     ┌─────────────────────────────┐
│  策略 μ ──→ 产生经验        │     │  行为策略 μ' ──→ 产生经验   │
│     ↑              ↓        │     │                    ↓        │
│     └──────── 用这些经验    │     │  经验池 D ←──── 存储经验    │
│              更新策略       │     │     ↓                       │
│                             │     │  采样批量 → 更新目标策略 μ  │
│  样本效率：较低              │     │                             │
│  稳定性：较高                │     │  样本效率：较高              │
│  代表：PPO, A2C, A3C        │     │  稳定性：需要技巧            │
│                             │     │  代表：DQN, DDPG, TD3, SAC  │
└─────────────────────────────┘     └─────────────────────────────┘

关键区别：异策略可以重复利用旧经验，像"翻旧账学习"！
```

**同策略**就像一个学生，只能用自己的错题来学习，做过的题目做完就丢掉了，下次还要重新做一遍才能学到东西。

**异策略**就像一个聪明的学生，有一个错题本，可以把以前做过的所有题目都保存下来，反复学习。这就是DQN中的**经验回放（Experience Replay）**。

这个区别对于实际应用至关重要：
- **异策略算法**（DDPG、TD3、SAC）样本效率更高，适合真实机器人（数据采集昂贵）
- **同策略算法**（PPO、A3C）虽然样本效率较低，但通常更稳定、更容易调参，适合模拟环境

现在，让我们开始探索第一个算法——DDPG，它是理解连续控制的基础！

---

## 31.2 DDPG：深度确定性策略梯度

### 31.2.1 🎯 费曼比喻：精准射手的修炼

想象一位弓箭手（演员）正在练习射箭。他的目标是射中靶心。

- **演员（弓箭手）**：学习如何拉弓、瞄准。他的策略是"看到目标后，手臂应该放在什么位置"。这是一个连续的决策——手臂的角度可以是0°到180°之间的任何值。
- **评论家（教练）**：观察射出的箭，告诉弓箭手"这一箭射得怎么样"。教练不直接说"手臂抬高一点"，而是说"这一箭得分7分，如果手臂再抬高一点可能得9分"。

DDPG的巧妙之处在于：**教练不评价每一个可能的手臂位置（那会太多），而是只评价弓箭手实际选择的那个位置**。同时，教练会告诉弓箭手哪个方向可以让得分更高（梯度方向）。

这就是DDPG的核心：**确定性策略**直接输出动作 + **Q函数**评估动作质量。

### 31.2.2 算法原理

DDPG（Deep Deterministic Policy Gradient）由DeepMind在2015年提出（Lillicrap et al., 2016），是首个成功将深度学习和确定性策略梯度结合，解决连续控制问题的算法。

#### 核心思想

DDPG的关键洞察来自Q函数的梯度：

如果 $Q(s, a)$ 告诉我们"在状态 $s$ 下动作 $a$ 有多好"，那么：

$$\nabla_a Q(s, a)$$

就告诉我们**"如何改变动作 $a$ 可以让Q值变大"**！

如果我们的策略 $\mu_\theta(s)$ 输出一个动作，那么：

$$\nabla_\theta Q(s, \mu_\theta(s)) = \nabla_a Q(s, a)|_{a=\mu_\theta(s)} \cdot \nabla_\theta \mu_\theta(s)$$

这就是**确定性策略梯度定理**！它允许我们通过Q函数的梯度来更新策略。

#### 网络架构

```
图31-4：DDPG网络架构

状态 s (连续向量)                      状态 s (连续向量)
     │                                      │
     ▼                                      ▼
┌─────────┐                           ┌─────────┐
│  Actor  │ ────→ 动作 a (连续) ────→ │ Critic  │
│  μ_θ(s) │                           │ Q_φ(s,a)│
└─────────┘                           └────┬────┘
      │                                    │
      │         ┌──────────────────┐      │
      │         │  目标Q值计算：   │      │
      │         │  y = r + γQ'(s',│      │
      │         │      μ'(s'))    │      │
      │         └────────┬─────────┘      │
      │                  │               │
      │                  ▼               │
      │              ┌──────────┐        │
      └─────────────→│  目标网络 │        │
                     │ Q'_φ', μ'_θ'     │
                     └──────────┘        │
                                          │
                              ┌───────────┴───────────┐
                              │  Critic损失：MSE(Q,y)  │
                              │  Actor损失：-mean(Q)   │
                              └───────────────────────┘

关键组件：
• Actor网络：直接输出确定性动作
• Critic网络：评估状态-动作对的价值
• 目标网络：软更新，提高稳定性
• 经验回放：异策略学习，提高样本效率
```

### 31.2.3 数学推导

#### 贝尔曼方程（异策略版本）

DDPG是异策略算法，使用目标策略 $μ'$ 和行为策略 $μ$（带噪声）。Q函数的目标值：

$$y = r + \gamma Q'(s', \mu'(s'))$$

其中 $Q'$ 和 $μ'$ 是目标网络。

#### Critic损失函数

评论家最小化均方误差：

$$\boxed{L_{critic} = \frac{1}{N} \sum_i (y_i - Q(s_i, a_i))^2}$$

#### Actor损失函数

演员最大化Q值（最小化负Q值）：

$$\boxed{L_{actor} = -\frac{1}{N} \sum_i Q(s_i, \mu(s_i))}$$

#### 策略梯度推导

使用链式法则：

$$\nabla_\theta J \approx \frac{1}{N} \sum_i \nabla_a Q(s, a)|_{s=s_i, a=\mu(s_i)} \cdot \nabla_\theta \mu(s)|_{s=s_i}$$

这就是确定性策略梯度！

#### 软更新（Soft Update）

为了保持目标网络的稳定性，我们使用软更新而非硬复制：

$$\phi' \leftarrow \tau \phi + (1-\tau) \phi'$$
$$\theta' \leftarrow \tau \theta + (1-\tau) \theta'$$

其中 $\tau \ll 1$（通常0.001）。

### 31.2.4 Ornstein-Uhlenbeck噪声

在连续控制中，我们需要探索。DDPG使用Ornstein-Uhlenbeck（OU）过程生成时间相关的噪声：

$$dx_t = \theta(\mu - x_t)dt + \sigma dW_t$$

OU噪声的特点是有"惯性"——如果上一步动作偏左，下一步也倾向于偏左。这适合物理控制任务，因为物理系统有动量。

```python
import numpy as np

class OUNoise:
    """Ornstein-Uhlenbeck过程"""
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
    
    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
```

### 31.2.5 完整DDPG实现

下面是DDPG的完整PyTorch实现：

```python
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
```

### 31.2.6 DDPG的关键特点

1. **异策略学习**：通过经验回放提高样本效率
2. **确定性策略**：直接输出动作，适合连续控制
3. **双网络结构**：在线网络和目标网络分离，提高稳定性
4. **软更新**：平滑更新目标网络参数
5. **OU噪声**：时间相关的探索噪声

---

## 31.3 TD3：双延迟深度确定性策略梯度

### 31.3.1 🔬 费曼比喻：严谨的科学家

想象一下，DDPG中的评论家（教练）是个有点"乐观"的人。当你问他"这个策略能得多少分"时，他总是倾向于高估——"我觉得你应该能得90分！"

这种过度乐观（overestimation）在强化学习中是个大问题。因为演员会根据评论家的评价来调整策略，如果评论家总是高估，演员就会被误导，以为自己比实际表现更好。

TD3（Twin Delayed Deep Deterministic Policy Gradient）就像是请来了一位更加严谨的科学家：
- **双胞胎评论家**：两位评论家独立评估，取较小值（防止乐观偏差）
- **延迟更新**：演员不需要每一步都更新，等评论家更准确了再学习
- **平滑目标**：计算目标值时，给动作加点小噪声，避免过拟合

### 31.3.2 过估计问题的根源

过估计是Q-Learning类算法的固有问题。让我们理解为什么：

$$y = r + \gamma \max_{a'} Q(s', a')$$

假设真正的Q值是 $Q^*$，我们的估计有噪声：

$$Q(s', a') = Q^*(s', a') + \epsilon_{a'}$$

那么：

$$\max_{a'} Q(s', a') = \max_{a'} [Q^*(s', a') + \epsilon_{a'}] \geq Q^*(s', a^*)$$

**最大值的期望大于等于期望的最大值！** 这导致系统性的过估计。

### 31.3.3 TD3的三大改进

```
图31-5：TD3的三大改进

改进1: 双Critic (Clipped Double Q-Learning)
──────────────────────────────────────────────
Critic 1: Q1(s,a) ──┐
                     ├──→ min(Q1, Q2) 用于目标计算
Critic 2: Q2(s,a) ──┘        ↓
                     防止单一Critic的过估计
                     
目标值: y = r + γ * min(Q1'(s',μ'(s')), Q2'(s',μ'(s')))

改进2: 延迟策略更新 (Delayed Policy Updates)
──────────────────────────────────────────────
Critic更新: 每步都更新 (学习更快)
Actor更新: 每2步更新一次 (等Critic稳定了再学)

原因: 策略更新会改变数据分布，等价值估计更准再更新策略

改进3: 目标策略平滑 (Target Policy Smoothing)
──────────────────────────────────────────────
计算目标值时，给目标动作加噪声：

a' = clip(μ'(s') + clip(ε, -c, c), a_low, a_high)
其中 ε ~ N(0, σ)

作用: 类似正则化，让相似的Q值产生相似的目标，防止过拟合
```

### 31.3.4 数学推导

#### Clipped Double Q-Learning

TD3使用两个Critic网络：

$$Q_{\phi_1}(s, a), \quad Q_{\phi_2}(s, a)$$

目标值计算：

$$\boxed{y = r + \gamma \min_{i=1,2} Q_{\phi_i'}(s', \tilde{a}')}$$

其中：
$$\tilde{a}' = \text{clip}(\mu_{\theta'}(s') + \text{clip}(\epsilon, -c, c), a_{low}, a_{high})$$

$$\epsilon \sim \mathcal{N}(0, \sigma)$$

#### 延迟更新

设策略延迟为 $d$（通常 $d=2$）：

- 每一步都更新两个Critic
- 只有第 $d$ 步时才更新Actor

### 31.3.5 完整TD3实现

```python
"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) 完整实现
DDPG的改进版，解决过估计问题

改进：
1. Clipped Double Q-Learning: 两个Critic，取较小值
2. Delayed Policy Updates: 延迟策略更新
3. Target Policy Smoothing: 目标策略平滑

作者: 机器学习与深度学习：从小学生到大师
参考: Fujimoto et al. (2018)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Actor网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
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


class GaussianNoise:
    """高斯噪声（TD3使用高斯噪声而非OU噪声）"""
    def __init__(self, action_dimension, sigma=0.1):
        self.action_dimension = action_dimension
        self.sigma = sigma
    
    def noise(self):
        return self.sigma * np.random.randn(self.action_dimension)


class TD3Agent:
    """
    TD3智能体
    
    相比DDPG的改进：
    1. 双Critic: 解决过估计问题
    2. 延迟策略更新: 每d步更新一次Actor
    3. 目标策略平滑: 给目标动作加噪声
    """
    def __init__(self, state_dim, action_dim, 
                 actor_lr=3e-4, critic_lr=3e-4,
                 gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5,
                 policy_delay=2,
                 buffer_capacity=100000,
                 hidden_dim=256, device='cpu'):
        """
        初始化TD3智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            actor_lr: Actor学习率
            critic_lr: Critic学习率
            gamma: 折扣因子
            tau: 软更新系数
            policy_noise: 目标策略噪声标准差
            noise_clip: 噪声裁剪范围
            policy_delay: 策略更新延迟（每几步更新一次Actor）
            buffer_capacity: 回放缓冲区容量
            hidden_dim: 隐藏层维度
            device: 计算设备
        """
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        
        # Actor网络
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # 双Critic网络（TD3的核心）
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr
        )
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 探索噪声
        self.exploration_noise = GaussianNoise(action_dim, sigma=0.1)
        
        # 训练计数
        self.train_step = 0
        self.actor_loss = 0
    
    def select_action(self, state, add_noise=True, noise_scale=1.0):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if add_noise:
            noise = self.exploration_noise.noise() * noise_scale
            action = action + noise
            action = np.clip(action, -1, 1)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转移"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def soft_update(self, source, target, tau):
        """软更新"""
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
            actor_loss: Actor损失（可能为0如果没有更新Actor）
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
        with torch.no_grad():
            # 目标策略平滑：给目标动作加噪声
            next_actions = self.actor_target(next_states)
            noise = torch.randn_like(next_actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1, 1)
            
            # 双Critic目标值：取较小值（防止过估计）
            next_q1 = self.critic1_target(next_states, next_actions)
            next_q2 = self.critic2_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)  # ★ 关键：取最小值
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 当前Q值
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Critic损失：两个Critic都优化
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # =============== Actor更新（延迟）===============
        actor_loss = None
        
        if self.train_step % self.policy_delay == 0:
            # Actor目标：最大化critic1的Q值
            # 注意：只使用critic1来指导策略更新
            predicted_actions = self.actor(states)
            actor_loss = -self.critic1(states, predicted_actions).mean()
            
            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.actor_loss = actor_loss.item()
            
            # 软更新目标网络
            self.soft_update(self.actor, self.actor_target, self.tau)
            self.soft_update(self.critic1, self.critic1_target, self.tau)
            self.soft_update(self.critic2, self.critic2_target, self.tau)
        
        self.train_step += 1
        
        return critic_loss.item(), self.actor_loss if actor_loss is None else actor_loss.item()
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())


# 训练代码与DDPG类似，省略...

if __name__ == "__main__":
    # 简单测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 使用之前的ContinuousGridWorld环境
    from typing import Tuple
    
    class TestEnv:
        def __init__(self):
            self.state_dim = 6
            self.action_dim = 2
        
        def reset(self):
            return np.random.randn(6)
        
        def step(self, action):
            next_state = np.random.randn(6)
            reward = np.random.randn()
            done = np.random.rand() > 0.95
            return next_state, reward, done, {}
    
    env = TestEnv()
    agent = TD3Agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=device
    )
    
    # 简单训练循环
    for episode in range(100):
        state = env.reset()
        episode_reward = 0
        
        for step in range(100):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) > 64:
                agent.learn(64)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        if (episode + 1) % 20 == 0:
            print(f"回合 {episode+1}, 奖励: {episode_reward:.2f}")
    
    print("TD3测试完成!")
```

### 31.3.6 TD3 vs DDPG 对比

| 特性 | DDPG | TD3 |
|------|------|-----|
| Critic数量 | 1 | 2（Clipped Double Q） |
| 目标计算 | $Q(s', \mu'(s'))$ | $\min(Q_1, Q_2)$ |
| 策略更新频率 | 每步 | 每2步（Delayed） |
| 目标策略 | 无噪声 | 加平滑噪声 |
| 过估计 | 有 | 显著减少 |
| 稳定性 | 一般 | 更好 |

---

## 31.4 SAC：软演员-评论家

### 31.4.1 🧭 费曼比喻：灵活的探险家

想象一个探险家在寻找宝藏。普通的探险家会选择"看起来最好"的路（最大化奖励）。但有时候，这条"最好"的路可能隐藏着未知的危险。

聪明的探险家会采取不同的策略：
- 他仍然会寻找宝藏（最大化奖励）
- 但他也会保持一定的探索（最大化熵）
- 他会主动避免"死路一条"的情况（熵正则化）

SAC（Soft Actor-Critic）就是这样的探险家。它不仅在寻找最优策略，还在保持策略的"多样性"。这就像在投资组合中分散风险——不把所有的鸡蛋放在一个篮子里。

### 31.4.2 最大熵强化学习

SAC的核心思想是**最大熵强化学习（Maximum Entropy RL）**。传统RL的目标是：

$$\max_\pi \sum_t \mathbb{E}[r(s_t, a_t)]$$

最大熵RL的目标多了一个熵项：

$$\boxed{\max_\pi \sum_t \mathbb{E}[r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]}$$

其中熵的定义是：

$$\mathcal{H}(\pi(\cdot|s)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s)]$$

**为什么需要熵？**

1. **探索**：高熵意味着更随机，鼓励探索
2. **鲁棒性**：避免过早收敛到次优策略
3. **多模态**：可以学习多种等效的策略
4. **样本效率**：在异策略学习中尤其重要

### 31.4.3 SAC的关键组件

```
图31-6：SAC架构

SAC = Actor-Critic + 最大熵 + 自动温度调节

状态 s
   │
   ▼
┌─────────────┐
│   Actor     │ ──→ 输出分布参数 (μ, σ)
│  π_θ(a|s)   │     └──→ 采样动作 a
└─────────────┘          ↓
                    重参数化技巧
                    a = tanh(μ + σ * ε), ε ~ N(0,1)
                         │
                         ▼
                    ┌─────────────┐
                    │   Critic    │ ──→ Q(s,a) (两个Critic)
                    │  Q_φ(s,a)   │
                    └─────────────┘

温度参数 α:
┌────────────────────────────────────────┐
│  自动调节目标: E[-log π(a|s)] = H_target │
│  α_loss = -α * (log π(a|s) + H_target)  │
└────────────────────────────────────────┘
```

### 31.4.4 数学推导

#### 软Q函数

在最大熵框架下，软Q函数满足：

$$Q(s, a) = r(s, a) + \gamma \mathbb{E}_{s'}[V(s')]$$

软价值函数：

$$V(s) = \mathbb{E}_{a \sim \pi}[Q(s, a) - \alpha \log \pi(a|s)]$$

#### 策略梯度（重参数化技巧）

由于我们需要从策略中采样，但又需要梯度，SAC使用**重参数化技巧**：

$$a = f_\theta(s, \epsilon) = \tanh(\mu_\theta(s) + \sigma_\theta(s) \odot \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)$$

这样动作 $a$ 关于参数 $\theta$ 可微了！

#### Actor损失

$$\boxed{L_{actor} = \mathbb{E}[\alpha \log \pi_\theta(a|s) - Q_\phi(s, a)]}$$

其中 $a = f_\theta(s, \epsilon)$。

#### Critic损失

$$\boxed{L_{critic} = \mathbb{E}[(Q_\phi(s, a) - y)^2]}$$

目标值：

$$y = r + \gamma (\min_{i=1,2} Q_{\phi_i'}(s', a') - \alpha \log \pi_\theta(a'|s'))$$

#### 自动温度调节

SAC可以自动学习温度参数 $\alpha$：

$$\boxed{L(\alpha) = \mathbb{E}[-\alpha \log \pi(a|s) - \alpha \bar{\mathcal{H}}]}$$

其中 $\bar{\mathcal{H}}$ 是目标熵（通常设为动作空间维度）。

### 31.4.5 完整SAC实现

```python
"""
SAC (Soft Actor-Critic) 完整实现
最大熵强化学习框架，自动温度调节

作者: 机器学习与深度学习：从小学生到大师
参考: Haarnoja et al. (2018)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from torch.distributions import Normal


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32), 
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """
    SAC Actor: 输出高斯分布的参数
    使用重参数化技巧进行采样
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state, deterministic=False):
        """
        从策略中采样动作
        
        Args:
            state: 状态
            deterministic: 是否确定性采样（评估时使用）
        
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            # 评估时使用均值
            action = torch.tanh(mean)
            return action, None
        
        # 重参数化技巧
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 可微采样
        action = torch.tanh(x_t)
        
        # 计算对数概率（包含tanh的雅可比行列式修正）
        log_prob = normal.log_prob(x_t)
        # tanh修正: log(1 - tanh(x)^2)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """Critic网络（Q函数）"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
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


class SACAgent:
    """
    SAC (Soft Actor-Critic) 智能体
    
    特点：
    1. 最大熵框架：鼓励探索，学习鲁棒策略
    2. 双Critic：减少过估计
    3. 重参数化技巧：低方差梯度估计
    4. 自动温度调节：自适应探索-利用权衡
    """
    def __init__(self, state_dim, action_dim,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005,
                 alpha=0.2, automatic_entropy_tuning=True,
                 target_entropy=None,
                 buffer_capacity=100000,
                 hidden_dim=256, device='cpu'):
        """
        初始化SAC智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            actor_lr: Actor学习率
            critic_lr: Critic学习率
            alpha_lr: 温度参数学习率
            gamma: 折扣因子
            tau: 软更新系数
            alpha: 初始温度参数
            automatic_entropy_tuning: 是否自动调节温度
            target_entropy: 目标熵（None时自动设为-action_dim）
            buffer_capacity: 回放缓冲区容量
            hidden_dim: 隐藏层维度
            device: 计算设备
        """
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Actor
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # 双Critic（SAC也使用双Critic减少过估计）
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr
        )
        
        # 温度参数 α
        if self.automatic_entropy_tuning:
            # 目标熵：通常设为 -dim(A)
            self.target_entropy = target_entropy if target_entropy is not None else -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = torch.tensor([alpha], device=device)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        self.train_step = 0
    
    def select_action(self, state, evaluate=False):
        """
        选择动作
        
        Args:
            state: 当前状态
            evaluate: 是否评估模式（无噪声）
        
        Returns:
            action: 选择的动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                action, _ = self.actor.sample(state_tensor, deterministic=True)
            else:
                action, _ = self.actor.sample(state_tensor, deterministic=False)
            action = action.cpu().numpy()[0]
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转移"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def soft_update(self, source, target, tau):
        """软更新"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )
    
    def learn(self, batch_size=64):
        """
        学习一步
        
        Returns:
            critic_loss: Critic损失
            actor_loss: Actor损失
            alpha_loss: 温度损失（如果使用自动调节）
            alpha: 当前温度值
        """
        if len(self.replay_buffer) < batch_size:
            return None, None, None, self.alpha.item()
        
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # =============== Critic更新 ===============
        with torch.no_grad():
            # 从当前策略采样下一个动作
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # 双Critic目标值
            next_q1 = self.critic1_target(next_states, next_actions)
            next_q2 = self.critic2_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            
            # 软价值：Q - α * log π
            next_q = next_q - self.alpha * next_log_probs
            
            # 目标值
            target_q = rewards + self.gamma * (1 - dones) * next_q
        
        # 当前Q值
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # =============== Actor更新 ===============
        # 重新采样动作（用于计算梯度）
        new_actions, log_probs = self.actor.sample(states)
        
        # 计算新动作的Q值
        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        q = torch.min(q1, q2)
        
        # Actor损失：α * log π - Q
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # =============== 温度更新 ===============
        alpha_loss = None
        if self.automatic_entropy_tuning:
            # 重新计算log_prob（不经过梯度）
            with torch.no_grad():
                _, log_probs_detached = self.actor.sample(states)
            
            # 温度损失
            alpha_loss = -(self.log_alpha * (log_probs_detached + self.target_entropy)).mean()
            
            # 更新温度
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # =============== 软更新目标网络 ===============
        self.soft_update(self.critic1, self.critic1_target, self.tau)
        self.soft_update(self.critic2, self.critic2_target, self.tau)
        
        self.train_step += 1
        
        alpha_loss_val = alpha_loss.item() if alpha_loss is not None else 0
        return critic_loss.item(), actor_loss.item(), alpha_loss_val, self.alpha.item()
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'alpha': self.alpha.item(),
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        if not self.automatic_entropy_tuning:
            self.alpha = torch.tensor([checkpoint['alpha']], device=self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())


if __name__ == "__main__":
    # 简单测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class TestEnv:
        def __init__(self):
            self.state_dim = 6
            self.action_dim = 2
        
        def reset(self):
            return np.random.randn(6).astype(np.float32)
        
        def step(self, action):
            next_state = np.random.randn(6).astype(np.float32)
            reward = np.random.randn()
            done = np.random.rand() > 0.95
            return next_state, reward, done, {}
    
    env = TestEnv()
    agent = SACAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=device,
        automatic_entropy_tuning=True
    )
    
    print("开始SAC训练测试...")
    for episode in range(100):
        state = env.reset()
        episode_reward = 0
        
        for step in range(100):
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) > 64:
                c_loss, a_loss, alpha_loss, alpha = agent.learn(64)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        if (episode + 1) % 20 == 0:
            print(f"回合 {episode+1}, 奖励: {episode_reward:.2f}, alpha: {agent.alpha.item():.4f}")
    
    print("SAC测试完成!")
```

### 31.4.6 SAC的优势

1. **样本效率高**：异策略学习 + 最大熵 = 快速学习
2. **稳定鲁棒**：熵正则化防止过早收敛
3. **无需手动调参**：自动温度调节
4. **并行友好**：适合分布式训练

---

## 31.5 PPO：近端策略优化

### 31.5.1 ⛰️ 费曼比喻：稳健登山者

想象你在爬山，目标是到达山顶。你可以：
- **激进的方式**：大步跨出，可能快速上升，但也可能踏空坠落
- **稳健的方式**：小步试探，确保每一步都确实让你更高

PPO（Proximal Policy Optimization）就是那位稳健登山者。

在策略梯度方法中，我们有一个问题：策略更新步长太大，可能导致策略崩溃（performance collapse）。TRPO（Trust Region Policy Optimization）用一种复杂的约束方法来解决这个问题，但计算成本很高。

PPO用一个巧妙的"裁剪"技巧达到了类似的效果，但实现简单得多。

### 31.5.2 从策略梯度到TRPO

#### 策略梯度回顾

REINFORCE的梯度：

$$\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot G_t]$$

#### 重要性采样

异策略版本：

$$\nabla_\theta J = \mathbb{E}_{\pi_{old}}\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} \nabla_\theta \log \pi_\theta(a|s) \cdot A\right]$$

定义**概率比**：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$$

#### TRPO的约束

TRPO的问题是：

$$\max_\theta \mathbb{E}[r_t(\theta) \cdot A_t]$$

约束：

$$D_{KL}(\pi_{old} \| \pi_\theta) \leq \delta$$

这需要用共轭梯度求解，计算复杂。

### 31.5.3 PPO的裁剪目标

PPO用一个简单的裁剪替代复杂的约束：

$$\boxed{L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]}$$

```
图31-7：PPO裁剪损失函数

概率比 r = π_new / π_old
优势 A > 0 (好的动作，应该增加概率)
──────────────────────────────────────────

        L^{CLIP}
          │
    1+ε   ├───────────────      ← 上限（裁剪）
          │           ╱
          │         ╱
          │       ╱
    1     ├─────●────────────   ← r = 1 时（无变化）
          │   ╱
          │ ╱
    1-ε   ├─────────────────    ← 下限（裁剪）
          │
          └──────────────────→ r
            0    1    2

当 A > 0 时：
• 如果 r < 1+ε: L = r * A （正常增加）
• 如果 r > 1+ε: L = (1+ε) * A （被裁剪，停止增加）

这防止了新策略变得与旧策略太不同！
```

### 31.5.4 PPO的核心组件

```
图31-8：PPO架构

状态 s ──→ [特征提取] ──→ Actor ──→ 动作分布 π(a|s)
                              │
                              └──→ log_prob(a|s)
                              
                    ┌──────────────────┐
                    │  优势函数估计    │
                    │  A = G - V(s)    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │    GAE计算       │
                    │  多步优势估计    │
                    └──────────────────┘

训练流程（多epoch更新）：
────────────────────────
1. 收集N个轨迹（同策略）
2. 对每个批量：
   a. 计算概率比 r = π_new / π_old
   b. 计算裁剪目标 L^CLIP
   c. 计算价值损失 L^VF
   d. 更新网络（Adam）
3. 重复K次（通常K=4-10）
```

### 31.5.5 广义优势估计（GAE）

PPO使用GAE来平衡偏差和方差：

$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

其中TD误差：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

GAE参数 $\lambda$：
- $\lambda = 0$：$\hat{A}_t = \delta_t$（高偏差，低方差）
- $\lambda = 1$：$\hat{A}_t = \sum \gamma^l r_{t+l} - V(s_t)$（低偏差，高方差）
- 通常 $\lambda = 0.95$

### 31.5.6 完整PPO实现

```python
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
```

### 31.5.7 PPO的关键特点

1. **裁剪目标**：简单有效地约束策略更新
2. **GAE**：高效的优势估计
3. **多epoch更新**：充分利用收集的数据
4. **熵奖励**：鼓励探索，防止过早收敛
5. **稳定性**：OpenAI的主打算法，非常稳定

---

## 31.6 A3C：异步优势演员-评论家

### 31.6.1 🐝 费曼比喻：蜂群智者

想象一群蜜蜂（workers）在寻找花蜜。每只蜜蜂独立探索，但都受同一个"蜂后"（全局网络）指导：
- 每只蜜蜂探索不同的区域（异步）
- 当一只蜜蜂找到花蜜，它飞回来告诉蜂后（梯度更新）
- 蜂后更新策略，所有蜜蜂都获得新知识
- 蜂后始终在家，不会迷失

这就是A3C的精髓：多个智能体并行探索，异步地更新全局网络。

### 31.6.2 从A3C到A2C

A3C（Asynchronous Advantage Actor-Critic）由DeepMind在2016年提出（Mnih et al., 2016）。它的核心思想是：
- 使用多个worker并行探索
- 每个worker独立计算梯度
- 异步更新全局网络
- 无需经验回放，因为并行本身提供了去相关性

```
图31-9：A3C架构

全局网络 (Global Network)
├── 全局Actor π_global
└── 全局Critic V_global
         │
    ┌────┴────┬────────┬────────┐
    │         │        │        │
  Worker 1 Worker 2 Worker 3 Worker N
    │         │        │        │
  独立探索  独立探索  独立探索  独立探索
    │         │        │        │
  计算梯度  计算梯度  计算梯度  计算梯度
    └─────────┴────────┴────────┘
              │
         异步更新全局网络
         (异步SGD)

优点：
• 无需经验回放（并行本身就是去相关）
• 多核CPU即可训练
• 探索更加多样化
• 训练速度快
```

### 31.6.3 A3C的数学

#### n-step回报

A3C使用n步回报来平衡偏差和方差：

$$R_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n})$$

优势函数：

$$A_t = R_t - V(s_t)$$

#### 损失函数

策略损失：

$$L_{policy} = -\log \pi(a_t|s_t) \cdot A_t$$

价值损失：

$$L_{value} = (R_t - V(s_t))^2$$

熵奖励：

$$L_{entropy} = -\mathcal{H}(\pi(\cdot|s_t))$$

总损失：

$$\boxed{L = L_{policy} + c_v L_{value} + c_e L_{entropy}}$$

### 31.6.4 A3C vs A2C

A3C是异步的，但实现复杂。A2C（Advantage Actor-Critic）是它的同步版本：
- 等待所有worker完成
- 收集所有梯度
- 同步更新

A2C更简单，A3C更快。

### 31.6.5 完整A3C/A2C实现

```python
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
```

### 31.6.6 A3C/A2C的关键特点

1. **并行探索**：多个worker同时探索，数据去相关
2. **n-step回报**：平衡偏差和方差
3. **同策略**：无需经验回放
4. **CPU友好**：可以在多核CPU上高效训练
5. **简洁**：相比A3C，A2C更易于实现和调参

---

## 31.7 深度RL实践技巧与前沿展望

### 31.7.1 算法选择指南

```
图31-10：深度RL算法选择决策树

开始
 │
 ├─ 动作空间是连续的吗？
 │   │
 │   ├─ 否（离散动作）
 │   │   │
 │   │   ├─ 环境复杂、需要高样本效率？
 │   │   │   ├─ 是 → PPO（最稳定）
 │   │   │   └─ 否 → A2C/A3C（最简单）
 │   │   │
 │   └─ 是（连续动作）
 │       │
 │       ├─ 需要最高的样本效率？
 │       │   ├─ 是 → SAC（推荐！）
 │       │   └─ 否 → 继续...
 │       │
 │       ├─ 需要最稳定的训练？
 │       │   ├─ 是 → TD3
 │       │   └─ 否 → DDPG（最简单）
 │       │
 └─ 快速参考表：

 ┌─────────────┬─────────────┬─────────────┬─────────────┐
 │    算法     │  样本效率   │   稳定性    │   实现难度  │
 ├─────────────┼─────────────┼─────────────┼─────────────┤
 │    DDPG     │     ★★☆     │    ★★☆      │    ★☆☆     │
 │    TD3      │     ★★★     │    ★★★      │    ★★☆     │
 │    SAC      │    ★★★★     │   ★★★★      │    ★★★     │
 │    PPO      │     ★★☆     │   ★★★★      │    ★★☆     │
 │  A2C/A3C    │     ★★☆     │    ★★☆      │    ★☆☆     │
 └─────────────┴─────────────┴─────────────┴─────────────┘

 推荐：
 • 真实机器人（样本贵）→ SAC
 • 游戏/模拟环境 → PPO
 • 入门学习 → DDPG/A2C
```

### 31.7.2 超参数调优技巧

#### 学习率

- 通常从 $3 \times 10^{-4}$ 开始
- Actor学习率通常比Critic小（DDPG：Actor 1e-4, Critic 1e-3）

#### 折扣因子 $\gamma$

- 大多数任务：0.99
- 长期任务：0.995或更高
- 短视任务：0.9

#### GAE参数 $\lambda$

- 默认：0.95
- 高偏差任务：减小到0.9
- 高方差任务：增加到0.99

#### 批量大小

- PPO：2048步
- DDPG/TD3/SAC：64-256
- 大的批量更稳定但样本效率低

### 31.7.3 常见问题和解决方案

```
图31-11：深度RL调试指南

问题1: 奖励不增长
─────────────────
可能原因：
• 学习率太高/太低
• 探索不足
• 奖励缩放问题

解决方案：
□ 调整学习率
□ 增加噪声/熵系数
□ 归一化奖励（减去均值，除以标准差）

问题2: 训练不稳定
─────────────────
可能原因：
• 策略更新步长太大
• 价值估计不准

解决方案：
□ 使用PPO代替DDPG
□ 增加target network的软更新系数
□ 减小学习率
□ 增加批量大小

问题3: 过拟合到早期经验
───────────────────────
可能原因：
• 同策略算法（PPO/A2C）的固有局限
• 样本多样性不足

解决方案：
□ 增加并行环境数
□ 增加熵奖励
□ 尝试异策略算法（SAC/TD3）

问题4: 收敛到次优策略
─────────────────────
可能原因：
• 局部最优
• 探索不足

解决方案：
□ 增加探索噪声
□ 参数随机化（Domain Randomization）
□ 多起点训练
```

### 31.7.4 前沿研究方向

#### 模型基础方法（Model-Based RL）

学习环境的动态模型，然后使用模型进行规划：

- **PETS** (Chua et al., 2018): 概率集合
- **MBPO** (Janner et al., 2019): 模型基础的策略优化
- **Dreamer** (Hafner et al., 2019): 在隐空间学习世界模型

#### 离线强化学习（Offline RL）

从固定数据集学习，无需在线交互：

- **CQL** (Kumar et al., 2020): 保守Q学习
- **IQL** (Kostrikov et al., 2021): 隐式Q学习

#### 多智能体强化学习

多个智能体同时学习和交互：

- **MADDPG** (Lowe et al., 2017)
- **MAPPO** (Yu et al., 2021)

#### 分层强化学习

学习多层次策略：

- **Option-Critic** (Bacon et al., 2017)
- **FeUdal Networks** (Vezhnevets et al., 2017)

#### 人类反馈强化学习（RLHF）

结合人类偏好训练：

- ChatGPT、Claude等LLM的核心训练方法
- **PPO+KL** (Ziegler et al., 2019)

### 31.7.5 学习路线图

```
图31-12：深度RL学习路径

入门级
──────
✅ 理解MDP和贝尔曼方程（第30章）
✅ 实现DQN（第30章）
✅ 实现REINFORCE（第30章）

进阶级
──────
✅ DDPG（本章）
  └─ 确定性策略 + Actor-Critic架构
✅ A2C（本章）
  └─ 并行训练 + n-step回报

高级
──────
✅ TD3（本章）
  └─ 解决过估计问题
✅ SAC（本章）
  └─ 最大熵框架 + 自动调参
✅ PPO（本章）
  └─ 裁剪目标 + GAE

专家级
──────
□ 模型基础RL（MBPO, Dreamer）
□ 离线RL（CQL, IQL）
□ 多智能体RL（MADDPG, MAPPO）
□ 分层RL（Option-Critic）
□ RLHF（用于大语言模型）

研究前沿
────────
□ Transformer在RL中的应用
□ 基于扩散模型的规划
□ 世界模型与世界模型智能体
□ 持续学习与终身学习
```

---

## 本章总结

### 核心概念回顾

本章我们探索了深度强化学习的五大核心算法：

| 算法 | 核心思想 | 关键创新 |
|------|----------|----------|
| **DDPG** | 确定性策略 + Actor-Critic | 连续控制的基础 |
| **TD3** | 双Critic + 延迟更新 | 解决过估计问题 |
| **SAC** | 最大熵 + 自动温度调节 | 样本效率最高 |
| **PPO** | 裁剪目标 + GAE | 最稳定、最流行 |
| **A3C** | 异步并行 + n-step | 简单高效 |

### 关键公式总结

**DDPG Actor梯度**：
$$\nabla_\theta J \approx \frac{1}{N} \sum_i \nabla_a Q(s, a) \cdot \nabla_\theta \mu(s)$$

**TD3目标值**：
$$y = r + \gamma \min_{i=1,2} Q_{\phi_i'}(s', \tilde{a}')$$

**SAC软Q函数**：
$$Q(s, a) = r + \gamma \mathbb{E}[Q(s', a') - \alpha \log \pi(a'|s')]$$

**PPO裁剪目标**：
$$L^{CLIP} = \mathbb{E}\left[\min\left(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

**GAE优势估计**：
$$\hat{A}_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

### 实践建议

1. **从PPO开始**：最稳定，调参友好
2. **连续控制用SAC**：样本效率最高
3. **注意归一化**：状态和奖励的归一化至关重要
4. **监控训练**：可视化奖励、Q值、策略熵
5. **多跑几次**：RL训练有随机性，多次运行取平均

### 进一步学习资源

- **书籍**: Sutton & Barto (2018) - Reinforcement Learning: An Introduction
- **课程**: CS285 (Berkeley) - Deep Reinforcement Learning
- **代码库**: Stable-Baselines3, CleanRL
- **论文**: Spinning Up in Deep RL (OpenAI)

---

## 参考文献

Bacon, P. L., Harb, J., & Precup, D. (2017). The option-critic architecture. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 31, No. 1).

Fujimoto, S., Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. In *International Conference on Machine Learning* (pp. 1587-1596). PMLR.

Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In *International Conference on Machine Learning* (pp. 1861-1870). PMLR.

Hafner, D., Lillicrap, T., Fischer, I., Villegas, R., Ha, D., Lee, H., & Davidson, J. (2019). Learning latent dynamics for planning from pixels. In *International Conference on Machine Learning* (pp. 2555-2565). PMLR.

Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., ... & Wierstra, D. (2016). Continuous control with deep reinforcement learning. In *International Conference on Learning Representations*.

Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., ... & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. In *International Conference on Machine Learning* (pp. 1928-1937). PMLR.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., & Wu, Y. (2021). The surprising effectiveness of PPO in cooperative multi-agent games. *arXiv preprint arXiv:2103.01955*.

Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., ... & Irving, G. (2019). Fine-tuning language models from human preferences. *arXiv preprint arXiv:1909.08593*.

---

## 练习题

### 基础练习

**练习 31.1**：DDPG的核心思想
解释DDPG如何利用Q函数的梯度来更新确定性策略。为什么这对连续控制很重要？

**练习 31.2**：TD3的三大改进
TD3相比DDPG有哪些改进？解释每一项改进如何解决DDPG的问题。

**练习 31.3**：最大熵强化学习
解释SAC中的最大熵损失函数。为什么熵正则化能提高学习的鲁棒性？

### 进阶练习

**练习 31.4**：PPO的裁剪目标
推导PPO的裁剪损失函数。解释为什么裁剪能防止策略更新过大。

**练习 31.5**：GAE计算
给定一个5步的轨迹，手动计算GAE优势估计。设 $\gamma=0.99$, $\lambda=0.95$。

**练习 31.6**：策略梯度比较
比较DDPG、PPO和A2C的策略梯度计算方法。各自的优缺点是什么？

### 挑战练习

**练习 31.7**：实现一个自定义环境
实现一个连续的Pendulum环境，使用DDPG或SAC进行训练。

**练习 31.8**：算法融合
设计一个结合了PPO稳定性和SAC样本效率的新算法。描述你的设计思路。

**练习 31.9**：多智能体扩展
思考如何将DDPG扩展到多智能体场景（MADDPG）。需要考虑哪些问题？

---

*本章完。下一章，我们将探索图神经网络——让AI学会理解关系和连接！*
