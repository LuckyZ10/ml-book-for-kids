# 第三十章：强化学习基础——在试错中成长

*"学习不是关于拥有正确答案，而是关于发现更好的答案。"*

---

## 引言：什么是强化学习？

还记得你第一次学骑自行车时的情景吗？没有人在旁边一步步教你"脚应该放在踏板的具体哪个位置"、"手应该握把手的精确角度"。相反，你通过不断的尝试和失败来学习——当你向右倾斜太多时摔倒了（惩罚），当你保持平衡骑出几米时感到兴奋（奖励）。渐渐地，你的身体学会了如何调整重心、如何蹬踏、如何转弯。

**强化学习（Reinforcement Learning, RL）** 正是这种学习方式在机器世界中的映射。与监督学习不同，强化学习没有成对的"输入-正确答案"数据集；与无监督学习也不同，强化学习有明确的反馈信号——**奖励（Reward）**。

让我用一个比喻来理解这三者的区别：

| 学习方式 | 老师角色 | 学习过程 | 例子 |
|---------|---------|---------|------|
| **监督学习** | 严格的指导老师，给出标准答案 | 学习输入到输出的映射 | 学生通过看带答案的习题册学习 |
| **无监督学习** | 没有老师，自己发现规律 | 发现数据中的隐藏结构 | 学生自己整理笔记，发现知识点之间的联系 |
| **强化学习** | 裁判只告诉"好"或"坏" | 通过试错最大化累积奖励 | 学生通过实际做题考试，根据分数调整学习策略 |

强化学习的核心思想来自**行为心理学**。20世纪初，心理学家斯金纳（B.F. Skinner）通过著名的"斯金纳箱"实验发现：动物（包括人类）会通过试错来学习那些能够获得奖励、避免惩罚的行为模式。一只老鼠学会按下杠杆获取食物，不是因为有人教它"如何按"，而是因为按下杠杆后获得了奖励，这种正反馈强化了该行为。

在强化学习中，我们的"学习者"称为**智能体（Agent）**，它所处的世界称为**环境（Environment）**。智能体通过**观察（Observation）**感知环境状态，通过**动作（Action）**影响环境，环境则通过**奖励（Reward）**反馈动作的好坏。智能体的目标是学习一个**策略（Policy）**——一个从状态到动作的映射，使得长期来看累积的奖励最大化。

本章我们将一起踏上这段探索之旅，从马尔可夫决策过程的数学基础，到Q-Learning的经典算法，再到深度Q网络（DQN）的革命性突破，最后探索策略梯度方法和Actor-Critic架构。每一节都配有完整的数学推导和可运行的代码实现，让你不仅理解"是什么"，更理解"为什么"和"怎么做"。

准备好了吗？让我们开始这场通过试错来成长的冒险吧！

---

## 30.1 马尔可夫决策过程（MDP）：强化学习的数学骨架

### 30.1.1 小狗训练的生活化比喻

想象一下你正在训练一只小狗"豆豆"学会在一个迷宫中找到出口：

- **状态（State）**：豆豆在迷宫中的位置，比如"在十字路口A"、"在死胡同B"
- **动作（Action）**：豆豆可以选择的行为，比如"向前走"、"向左转"、"向右转"
- **奖励（Reward）**：环境给出的即时反馈
  - 找到出口：+10分（大骨头！）
  - 撞墙：-1分（疼！）
  - 每一步：-0.1分（鼓励尽快找到出口）
- **策略（Policy）**：豆豆的"行为准则"——在每个位置选择哪个方向

豆豆一开始可能随机乱撞，但渐渐地，它会学会：在某个路口向左走通常会更快找到出口，而在另一个路口向右走会撞到墙。这就是强化学习的精髓！

### 30.1.2 MDP的正式定义

**马尔可夫决策过程（Markov Decision Process, MDP）** 是强化学习的数学框架。一个MDP由五元组 $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ 定义：

1. **状态空间** $\mathcal{S}$：所有可能状态的集合
2. **动作空间** $\mathcal{A}$：所有可能动作的集合  
3. **状态转移概率** $\mathcal{P}(s'|s, a)$：在状态 $s$ 执行动作 $a$ 后转移到状态 $s'$ 的概率
4. **奖励函数** $\mathcal{R}(s, a, s')$：在状态 $s$ 执行动作 $a$ 并转移到 $s'$ 获得的即时奖励
5. **折扣因子** $\gamma \in [0, 1]$：未来奖励的折扣率

#### 马尔可夫性质：未来的独立

MDP的核心假设是**马尔可夫性质**：

$$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ..., s_0, a_0) = P(s_{t+1} | s_t, a_t)$$

这意味着：**未来只依赖于现在，而与过去无关**。用小狗训练的比喻来说：豆豆下一步会去哪里，只取决于它现在的位置和它的选择，而与它是如何到达这里的无关。

这个性质虽然看起来简化了很多，但在实际中非常有用——它让我们可以用相对简单的数学模型来刻画复杂的世界。

### 30.1.3 策略、回报与价值函数

#### 策略（Policy）

策略 $\pi$ 是从状态到动作的映射。它可以是：

- **确定性策略**：$\pi(s) = a$，在状态 $s$ 总是选择动作 $a$
- **随机性策略**：$\pi(a|s) = P(A=a | S=s)$，在状态 $s$ 以概率选择动作 $a$

#### 回报（Return）

智能体的目标是最大化**累积奖励**（也称为**回报**）：

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

折扣因子 $\gamma$ 的重要性：
- 当 $\gamma = 0$：只看即时奖励，目光短浅
- 当 $\gamma = 1$：未来奖励与现在同等重要，可能忽略即时风险
- 通常取 $0.9 \leq \gamma < 1$：既考虑长远，又不过度牺牲当下

#### 状态价值函数 $V^\pi(s)$

在策略 $\pi$ 下，从状态 $s$ 开始的期望回报：

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

可以理解为：如果我按照策略 $\pi$ 行动，从现在状态 $s$ 出发，长期来看我能获得多少奖励的期望。

#### 动作价值函数 $Q^\pi(s, a)$

在状态 $s$ 执行动作 $a$，然后遵循策略 $\pi$ 的期望回报：

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

$Q$ 函数告诉我们：在状态 $s$ 执行动作 $a$ 有多"好"。这是强化学习中最重要的函数之一！

### 30.1.4 贝尔曼方程的推导

#### 贝尔曼期望方程

$V^\pi$ 和 $Q^\pi$ 之间存在深刻的关系。让我们推导这个关系：

从 $V^\pi(s)$ 的定义出发：

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

展开回报的定义：

$$V^\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s]$$

根据期望的线性性质：

$$V^\pi(s) = \mathbb{E}_\pi[R_{t+1} | S_t = s] + \gamma \mathbb{E}_\pi[G_{t+1} | S_t = s]$$

注意到 $\mathbb{E}_\pi[R_{t+1} | S_t = s]$ 可以通过对所有可能的动作和下一状态求期望：

$$\mathbb{E}_\pi[R_{t+1} | S_t = s] = \sum_a \pi(a|s) \sum_{s'} \mathcal{P}(s'|s,a) \mathcal{R}(s,a,s')$$

而 $\mathbb{E}_\pi[G_{t+1} | S_t = s]$ 实际上是 $V^\pi$ 在下一状态的期望：

$$\mathbb{E}_\pi[G_{t+1} | S_t = s] = \sum_a \pi(a|s) \sum_{s'} \mathcal{P}(s'|s,a) V^\pi(s')$$

因此，我们得到**贝尔曼期望方程**：

$$\boxed{V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} \mathcal{P}(s'|s,a) \left[\mathcal{R}(s,a,s') + \gamma V^\pi(s')\right]}$$

类似地，对于 $Q$ 函数：

$$Q^\pi(s, a) = \sum_{s'} \mathcal{P}(s'|s,a) \left[\mathcal{R}(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')\right]$$

#### 贝尔曼最优方程

如果我们想要最优策略，对应的**最优价值函数**满足：

$$\boxed{V^*(s) = \max_a \sum_{s'} \mathcal{P}(s'|s,a) \left[\mathcal{R}(s,a,s') + \gamma V^*(s')\right]}$$

$$\boxed{Q^*(s, a) = \sum_{s'} \mathcal{P}(s'|s,a) \left[\mathcal{R}(s,a,s') + \gamma \max_{a'} Q^*(s', a')\right]}$$

这两个方程是强化学习算法的基石！它们表达了最优价值函数的递归性质——最优的长期价值等于当前的即时奖励加上下一状态的最优价值的折扣值。

### 30.1.5 Grid World环境实现

让我们实现一个简单的网格世界环境来实践这些概念：

```python
"""
Grid World环境：强化学习的"Hello World"
一个简单的网格世界，智能体需要找到从起点到终点的最短路径

网格布局：
S . . .
. X . G
. . . .

S: 起点 (0,0)
G: 终点/目标 (3,1) - 奖励 +10
X: 陷阱 (1,1) - 奖励 -10
.: 普通格子 - 每步奖励 -0.1
"""

import numpy as np
import random
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class Transition:
    """存储状态转移"""
    state: Tuple[int, int]
    action: int
    reward: float
    next_state: Tuple[int, int]
    done: bool


class GridWorld:
    """
    网格世界环境
    
    状态：智能体的位置 (row, col)
    动作：0=上, 1=右, 2=下, 3=左
    """
    
    # 动作映射
    ACTIONS = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    ACTION_DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    
    def __init__(self, size: int = 4, seed: Optional[int] = None):
        """
        初始化Grid World环境
        
        Args:
            size: 网格大小 (size x size)
            seed: 随机种子
        """
        self.size = size
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        
        # 设置陷阱位置
        self.traps = set()
        if size >= 4:
            self.traps.add((size // 2, size // 2))
            self.traps.add((size // 2 - 1, size // 2))
        
        # 奖励设置
        self.goal_reward = 10.0
        self.trap_reward = -10.0
        self.step_reward = -0.1
        
        # 状态
        self.agent_pos = self.start_pos
        self.steps = 0
        self.max_steps = size * size * 4
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def reset(self) -> Tuple[int, int]:
        """重置环境，返回初始状态"""
        self.agent_pos = self.start_pos
        self.steps = 0
        return self.agent_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        """
        执行动作，返回 (下一状态, 奖励, 是否结束, 额外信息)
        
        有10%的概率会随机执行一个动作（模拟环境的不确定性）
        """
        self.steps += 1
        
        # 10%概率随机动作（环境随机性）
        if random.random() < 0.1:
            action = random.randint(0, 3)
        
        # 计算新位置
        dr, dc = self.ACTION_DELTA[action]
        new_row = max(0, min(self.size - 1, self.agent_pos[0] + dr))
        new_col = max(0, min(self.size - 1, self.agent_pos[1] + dc))
        new_pos = (new_row, new_col)
        
        self.agent_pos = new_pos
        
        # 计算奖励
        if new_pos == self.goal_pos:
            reward = self.goal_reward
            done = True
        elif new_pos in self.traps:
            reward = self.trap_reward
            done = True
        else:
            reward = self.step_reward
            done = self.steps >= self.max_steps
        
        info = {'steps': self.steps, 'action_taken': action}
        return new_pos, reward, done, info
    
    def get_valid_actions(self, state: Optional[Tuple[int, int]] = None) -> List[int]:
        """获取在指定状态下的有效动作"""
        if state is None:
            state = self.agent_pos
        return [0, 1, 2, 3]  # 在这个简单环境中所有动作都有效
    
    def render(self) -> str:
        """渲染当前环境状态"""
        lines = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                pos = (r, c)
                if pos == self.agent_pos:
                    row.append(' A ')  # 智能体
                elif pos == self.start_pos:
                    row.append(' S ')  # 起点
                elif pos == self.goal_pos:
                    row.append(' G ')  # 终点
                elif pos in self.traps:
                    row.append(' X ')  # 陷阱
                else:
                    row.append(' . ')  # 普通格子
            lines.append(''.join(row))
        return '\n'.join(lines)
    
    def get_state_index(self, state: Tuple[int, int]) -> int:
        """将2D状态转换为1D索引"""
        return state[0] * self.size + state[1]
    
    def get_state_from_index(self, index: int) -> Tuple[int, int]:
        """将1D索引转换为2D状态"""
        return (index // self.size, index % self.size)
    
    @property
    def num_states(self) -> int:
        """状态数量"""
        return self.size * self.size
    
    @property
    def num_actions(self) -> int:
        """动作数量"""
        return 4


def demo_gridworld():
    """演示Grid World环境"""
    print("=" * 50)
    print("Grid World 环境演示")
    print("=" * 50)
    
    env = GridWorld(size=4, seed=42)
    print(f"\n网格大小: {env.size}x{env.size}")
    print(f"起点: {env.start_pos}")
    print(f"终点: {env.goal_pos}")
    print(f"陷阱: {env.traps}")
    
    print("\n初始状态:")
    print(env.render())
    
    # 随机运行几步
    print("\n随机执行动作:")
    state = env.reset()
    total_reward = 0
    
    for i in range(10):
        action = random.randint(0, 3)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"\n第 {i+1} 步:")
        print(f"  动作: {GridWorld.ACTIONS[action]}")
        print(f"  位置: {state} -> {next_state}")
        print(f"  奖励: {reward:.2f}")
        print(env.render())
        
        if done:
            print(f"\n回合结束！总奖励: {total_reward:.2f}")
            break
        
        state = next_state


if __name__ == "__main__":
    demo_gridworld()
```

运行这段代码，你会看到一个简单的网格世界环境。智能体（A）需要在4×4的网格中从起点（S）移动到终点（G），避开陷阱（X）。每一步都会产生-0.1的奖励，到达终点获得+10，掉入陷阱获得-10。

这个环境虽然简单，但包含了MDP的所有要素：
- **状态**：智能体的位置 (row, col)
- **动作**：上、下、左、右
- **转移概率**：90%执行选定动作，10%随机执行其他动作
- **奖励**：根据位置给予不同奖励
- **折扣因子**：将在后续算法中使用

---

## 30.2 Q-Learning：从探索中学习

### 30.2.1 游戏玩家的成长历程

想象一个新手玩家在第一次玩《超级马里奥》游戏：

**第一阶段：完全随机探索**
- 不知道按哪个键会让马里奥跳跃
- 不知道哪些敌人会伤害他
- 只是随机按键，偶尔误打误撞跳过一个坑

**第二阶段：建立粗略的因果联系**
- 发现按"A"键时马里奥会跳（A键 → 跳跃）
- 发现碰到敌人会死（敌人 → 避免）
- 发现金币是好东西（金币 → 收集）

**第三阶段：学会策略性思考**
- "在这个悬崖前，我应该先跳再冲刺"
- "如果前面有敌人，我应该从上方踩它"
- "为了拿到那个隐藏金币，我需要先跳到那个平台"

Q-Learning正是模拟这种学习过程！它通过不断地"尝试-观察结果-更新知识"来逐渐建立一个**Q表**——记录"在某个状态下采取某个动作有多好"。

### 30.2.2 Q-Learning算法原理

Q-Learning由Watkins在1989年提出，是一种**离策略（Off-Policy）**的时序差分控制算法。

#### Q值更新的直觉

假设你刚学到一个新知识：
- **之前**：我以为"在雨天带伞"的Q值是5分（还不错）
- **经验**：今天我带了伞，真的下雨了，我很高兴没有被淋湿
- **修正**：原来"在雨天带伞"的Q值应该更高，比如8分

Q-Learning就是这样不断更新Q值的：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \underbrace{[r + \gamma \max_{a'} Q(s', a') - Q(s, a)]}_{\text{时序差分误差}}$$

其中：
- $\alpha$：学习率（0 < $\alpha$ ≤ 1），控制新信息覆盖旧知识的速度
- $r + \gamma \max_{a'} Q(s', a')$：目标Q值（当前奖励 + 下一状态的最佳Q值折扣）
- $Q(s, a)$：当前的Q值估计
- 方括号内：**时序差分误差（TD Error）**

#### Q-Learning的完整算法

```
初始化 Q(s, a) 对所有 s, a
对于每个回合：
    观察初始状态 s
    对于每一步：
        根据Q选择动作 a（例如 ε-贪心）
        执行动作 a，观察奖励 r 和下一状态 s'
        更新 Q(s, a) ← Q(s, a) + α[r + γ·max_a' Q(s', a') - Q(s, a)]
        s ← s'
        如果 s 是终止状态：跳出
```

### 30.2.3 Q值更新公式的数学推导

让我们更严谨地推导Q-Learning的更新规则。

#### 从贝尔曼最优方程出发

我们知道最优Q函数满足：

$$Q^*(s, a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') | S_t = s, A_t = a]$$

如果环境是确定性的，可以去掉期望：

$$Q^*(s, a) = r + \gamma \max_{a'} Q^*(s', a')$$

#### 时序差分学习

我们的目标是让Q值逼近最优Q值。定义**时序差分目标**：

$$y = r + \gamma \max_{a'} Q(s', a')$$

这个目标使用了当前的Q估计来构造一个"更好的"估计（因为包含了新的奖励信息）。

然后，我们用梯度下降的思想来更新Q值，使得Q(s, a)向y靠近：

$$Q(s, a) \leftarrow Q(s, a) + \alpha [y - Q(s, a)]$$

这就得到了Q-Learning的更新规则！

#### 为什么叫"时序差分"？

**时序差分（Temporal Difference, TD）** 的核心思想是：**用当前的预测来更新之前的预测**。

- **蒙特卡洛方法**：等一个完整的回合结束，用实际的累积回报来更新
- **时序差分方法**：每走一步就更新，用"当前的奖励 + 下一状态的估计"来更新

TD方法的优势在于**可以在线学习**，不需要等待回合结束。就像一个学生在考试过程中就能根据即时反馈调整策略，而不需要等到期末考试结束才知道自己哪里错了。

### 30.2.4 ε-贪心探索策略

Q-Learning面临一个**探索-利用困境（Exploration-Exploitation Dilemma）**：
- **利用（Exploitation）**：选择当前Q值最高的动作，最大化即时收益
- **探索（Exploration）**：尝试其他动作，可能发现更好的策略

如果只利用不探索，可能陷入局部最优；如果只探索不利用，则无法收敛到最优策略。

**ε-贪心（Epsilon-Greedy）** 策略提供了一个平衡方案：

$$\pi(a|s) = \begin{cases} 
1 - \epsilon + \frac{\epsilon}{|A|} & \text{if } a = \arg\max_{a'} Q(s, a') \\
\frac{\epsilon}{|A|} & \text{otherwise}
\end{cases}$$

实现上很简单：
- 以概率 $1 - \epsilon$ 选择Q值最大的动作（利用）
- 以概率 $\epsilon$ 随机选择动作（探索）

通常随着学习进行，逐渐减小$\epsilon$（如从1.0降到0.01），让智能体从"广泛探索"过渡到"精细利用"。

### 30.2.5 Q-Learning完整实现

```python
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
```

### 30.2.6 训练过程可视化与分析

运行上述代码，你会看到智能体逐渐学会如何在Grid World中找到最优路径。让我们分析一下训练过程中发生了什么：

**训练初期（Episode 1-200）**：
- ε接近1.0，智能体几乎完全随机探索
- 奖励波动很大，有时很快找到终点，有时陷入陷阱
- Q值几乎均匀分布，没有明显的策略

**训练中期（Episode 200-800）**：
- ε逐渐减小，智能体开始更多地利用已有知识
- 奖励逐渐提高，步数逐渐减少
- Q值开始分化，靠近终点的状态-动作对的Q值变高

**训练后期（Episode 800-2000）**：
- ε接近最小值，智能体主要利用学到的策略
- 奖励稳定在较高水平
- Q值收敛，形成清晰的价值梯度指向终点

这种渐进式的学习过程正是强化学习的魅力所在——**从无到有的涌现智能**。

---

## 30.3 深度Q网络（DQN）：结合深度学习

### 30.3.1 从Q表到神经网络的飞跃

让我们回到Q-Learning。在Grid World中，我们只有16个状态和4个动作，Q表只有64个条目。但如果状态空间很大呢？

考虑Atari游戏：
- 屏幕分辨率：84×84像素
- 每个像素：256种可能值
- 可能的状态数：$256^{84×84}$ ≈ $10^{16934}$

这个数字比宇宙中的原子数量还多！显然，我们无法用表格存储这么多Q值。

**深度Q网络（Deep Q-Network, DQN）**的洞见是：用**神经网络**来近似Q函数！

$$Q(s, a; \theta) \approx Q^*(s, a)$$

其中 $\theta$ 是神经网络的参数。神经网络可以**泛化**：即使它没见过某个具体状态，也能基于见过的相似状态做出合理预测。

### 30.3.2 DQN的网络架构

对于Atari游戏，DQN使用卷积神经网络（CNN）处理原始像素：

```
输入: 84×84×4 (最近4帧的灰度图像)
    ↓
Conv1: 32 filters, 8×8, stride 4 + ReLU
    ↓
Conv2: 64 filters, 4×4, stride 2 + ReLU  
    ↓
Conv3: 64 filters, 3×3, stride 1 + ReLU
    ↓
Flatten
    ↓
FC: 512 units + ReLU
    ↓
输出: 每个动作的Q值 (如4个动作就有4个输出)
```

这种架构的美妙之处在于：**网络直接学习从原始像素到动作价值的映射**，不需要人工设计特征。

### 30.3.3 DQN的两个关键创新

Mnih等人在2013/2015年的DQN论文中提出了两个关键创新，使深度强化学习首次成功：

#### 1. 经验回放（Experience Replay）

**问题**：连续的状态转移是高度相关的（时间序列相关性），这违反了神经网络训练需要独立同分布样本的假设。

**解决方案**：存储经验 $(s, a, r, s', \text{done})$ 到一个**回放缓冲区（Replay Buffer）**，然后随机采样小批量来训练。

```python
# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)  # 解压为独立数组
```

经验回放的好处：
- **打破相关性**：随机采样消除时间序列相关性
- **提高数据效率**：每个经验可以被用于多次训练
- **平滑数据分布**：避免连续的极端样本

#### 2. 目标网络（Target Network）

**问题**：Q-Learning的更新目标是 $y = r + \gamma \max_{a'} Q(s', a'; \theta)$。如果用同一套参数 $\theta$ 来计算目标和更新，就像"自己追着自己的尾巴跑"，训练会不稳定。

**解决方案**：使用两套参数：
- **行为网络** $\theta$：用于选择动作和计算当前Q值
- **目标网络** $\theta^-$：用于计算目标Q值，每隔若干步才从行为网络复制一次

```python
# 目标网络更新
if step % target_update_freq == 0:
    target_network.load_state_dict(behavior_network.state_dict())
```

这使得训练更加稳定，因为目标值在短时间内是固定的。

### 30.3.4 DQN损失函数与梯度推导

#### 损失函数定义

DQN的目标是最小化预测Q值与目标Q值之间的**均方误差（MSE）**：

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ \left( y - Q(s, a; \theta) \right)^2 \right]$$

其中目标：

$$y = \begin{cases} r & \text{if done} \\ r + \gamma \max_{a'} Q(s', a'; \theta^-) & \text{otherwise} \end{cases}$$

#### 梯度推导

对损失函数求梯度：

$$\nabla_\theta L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[ 2(y - Q(s, a; \theta)) \cdot \nabla_\theta Q(s, a; \theta) \right]$$

注意到我们只对**行为网络**的参数求导，目标网络 $\theta^-$ 被视为常数。

#### 时序差分误差的视角

定义TD误差：

$$\delta = y - Q(s, a; \theta)$$

则梯度可以写成：

$$\nabla_\theta L(\theta) = -2 \cdot \mathbb{E}[\delta \cdot \nabla_\theta Q(s, a; \theta)]$$

这与Q-Learning中的TD更新有异曲同工之妙——只是现在用梯度下降来实现。

### 30.3.5 DQN完整实现

```python
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
```

### 30.3.6 DQN的局限与改进

虽然DQN取得了突破性进展，但它也有局限性：

**1. 高估问题（Overestimation）**：
- `max`操作会系统性地高估Q值
- **解决方案**：Double DQN，使用两个网络分别选择和评估动作

**2. 均匀采样问题**：
- 回放缓冲区均匀采样，但某些经验更重要
- **解决方案**：Prioritized Experience Replay，根据TD误差优先级采样

**3. 动作空间离散**：
- DQN只能处理离散动作
- **解决方案**：连续动作空间需要其他方法（如Policy Gradient）

---

## 30.4 策略梯度方法

### 30.4.1 直接优化策略的直觉

到目前为止，我们的方法都是**基于价值的**：
1. 学习价值函数（Q值）
2. 从价值函数推导出策略（如ε-贪心）

**策略梯度方法**采取不同的思路：**直接参数化策略**，并用梯度上升来优化策略参数！

这就像一个演员（策略）直接学习"怎么演"，而不是先学会"评价演技好坏"再推导出"怎么演"。

#### 为什么要直接优化策略？

1. **连续动作空间**：自动驾驶的方向盘角度可以是任意实数，Q表无法表示
2. **随机策略**：某些任务需要随机策略（如石头剪刀布，确定性策略会被对手利用）
3. **更好的收敛性**：价值方法的最大化操作可能引入误差累积

### 30.4.2 策略梯度定理的推导

#### 策略参数化

我们将策略表示为参数化函数：

$$\pi_\theta(a | s) = P(A = a | S = s; \theta)$$

对于离散动作，通常使用**softmax**输出：

$$\pi_\theta(a | s) = \frac{\exp(f_\theta(s, a))}{\sum_{a'} \exp(f_\theta(s, a'))}$$

其中 $f_\theta(s, a)$ 是神经网络输出的"动作偏好"分数。

#### 损失函数

策略的目标是最大化**期望累积奖励**：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

其中 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ...)$ 是一条轨迹。

#### 策略梯度定理（Policy Gradient Theorem）

**定理**：对于任何可微分策略 $\pi_\theta$，策略梯度可以写成：

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a | s) \cdot Q^{\pi_\theta}(s, a)\right]
}$$

#### 定理的推导

**第一步：写出损失函数**

对于起始状态分布 $d_0(s)$：

$$J(\theta) = \sum_s d_0(s) V^{\pi_\theta}(s)$$

**第二步：计算梯度**

$$\nabla_\theta J(\theta) = \sum_s d_0(s) \nabla_\theta V^{\pi_\theta}(s)$$

**第三步：展开 $V^\pi(s)$**

$$V^\pi(s) = \sum_a \pi_\theta(a|s) Q^\pi(s, a)$$

$$\nabla_\theta V^\pi(s) = \sum_a \left[\nabla_\theta \pi_\theta(a|s) \cdot Q^\pi(s, a) + \pi_\theta(a|s) \cdot \nabla_\theta Q^\pi(s, a)\right]$$

**第四步：展开 $Q^\pi$**

$$Q^\pi(s, a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

这会导致递归关系。经过一系列推导（涉及状态访问分布），最终得到策略梯度定理。

#### 直观理解

$$\nabla_\theta \log \pi_\theta(a | s) = \frac{\nabla_\theta \pi_\theta(a | s)}{\pi_\theta(a | s)}$$

这被称为**得分函数（Score Function）**或**似然比（Likelihood Ratio）**。

策略梯度更新规则：

$$\theta \leftarrow \theta + \alpha \cdot \underbrace{\nabla_\theta \log \pi_\theta(a | s)}_{\text{增加该动作的概率}} \cdot \underbrace{Q^\pi(s, a)}_{\text{该动作的好坏程度}}$$

- 如果 $Q > 0$（动作好）：增加选择该动作的概率
- 如果 $Q < 0$（动作差）：减少选择该动作的概率

### 30.4.3 REINFORCE算法

REINFORCE（Williams, 1992）是蒙特卡洛策略梯度的实现：

**算法**：
1. 根据当前策略采样一条完整轨迹
2. 计算每个时刻的累积回报 $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$
3. 更新参数：$\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t$

```python
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
```

### 30.4.4 基线（Baseline）减小方差

REINFORCE有一个问题：**方差很大**。同样的策略可能因为环境随机性产生截然不同的回报。

**解决方案**：使用**基线（Baseline）**来减小方差。

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a | s) \cdot (Q^\pi(s, a) - b(s))\right]$$

通常选择 $b(s) = V^\pi(s)$（状态价值）作为基线。这样：
- 如果动作比平均水平好（$Q > V$），增加其概率
- 如果动作比平均水平差（$Q < V$），减少其概率

$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ 称为**优势函数（Advantage Function）**，这正是下一节Actor-Critic的核心！

---

## 30.5 Actor-Critic：价值与策略的结合

### 30.5.1 演员与评论家的协作

**Actor-Critic架构**结合了价值方法和策略方法的优点：

| 角色 | 类比 | 功能 | 更新目标 |
|-----|-----|-----|---------|
| **Actor（演员）** | 演员表演 | 策略网络 $\pi_\theta(a|s)$ | 根据Critic的反馈调整表演方式 |
| **Critic（评论家）** | 影评人 | 价值网络 $V_\phi(s)$ | 学会准确评价Actor的表演 |

**协作过程**：
1. Actor选择动作并执行
2. Critic观察结果，给出一个"评价"
3. Actor根据Critic的评价调整自己的策略
4. Critic根据实际回报更新自己的评价标准

### 30.5.2 A2C与A3C算法

#### Advantage Actor-Critic（A2C）

A2C使用**优势函数**作为策略梯度中的权重：

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a | s) \cdot A(s, a)\right]$$

其中优势函数通过Critic估计：

$$A(s, a) \approx r + \gamma V_\phi(s') - V_\phi(s)$$

这就是**时序差分优势**！

**两个损失函数**：
1. **策略损失**：$L_\pi = -\log \pi_\theta(a|s) \cdot A(s, a)$
2. **价值损失**：$L_v = (r + \gamma V_\phi(s') - V_\phi(s))^2$
3. **总损失**：$L = L_\pi + c_v \cdot L_v$（$c_v$是价值损失的系数）

#### Asynchronous Advantage Actor-Critic（A3C）

A3C（Mnih et al., 2016）的洞见是：**并行化**

- 多个"工作者"智能体在各自的环境中并行探索
- 异步地更新共享的全局网络参数
- 不需要经验回放（因为并行本身就打破了相关性）

A3C的优势：
- **更快的训练**：并行探索，数据收集更快
- **更好的探索**：不同工作者可能探索环境的不同部分
- **无需大内存**：不需要存储大量经验

### 30.5.3 Actor-Critic完整实现

```python
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
```

---

## 30.6 前沿与应用

### 30.6.1 AlphaGo：围棋的巅峰

2016年，DeepMind的**AlphaGo**以4:1击败世界围棋冠军李世石，这是人工智能历史上的里程碑事件。

#### AlphaGo的核心组件

AlphaGo结合了三种技术：

**1. 策略网络（Policy Network）**
- 先通过人类棋谱监督学习（SL策略网络）
- 然后通过自我对弈强化学习（RL策略网络）
- 输入：当前棋盘状态
- 输出：每个可能落子位置的概率

**2. 价值网络（Value Network）**
- 评估当前局面的胜率
- 输入：棋盘状态
- 输出：当前玩家获胜的概率

**3. 蒙特卡洛树搜索（MCTS）**
- 结合策略网络和价值网络进行搜索
- 在有限时间内探索最佳落子

$$U(s, a) = Q(s, a) + c_{puct} \cdot P(a|s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

其中：
- $Q(s, a)$：动作价值（来自价值网络和 rollout）
- $P(a|s)$：先验概率（来自策略网络）
- $N(s)$, $N(s, a)$：访问计数

#### AlphaZero：超越人类的自学

AlphaGo的后续版本**AlphaZero**（2017）更进一步：
- **完全自学**：不需要人类棋谱，完全通过自我对弈学习
- **通用算法**：同样的算法可以学会围棋、国际象棋、将棋
- **超越人类**：成为历史上最强的棋类AI

### 30.6.2 机器人控制

强化学习在机器人领域的应用：

**机器人行走**
- 使用PPO/TRPO训练四足机器人行走
- 奖励函数设计：前进速度 + 姿态稳定性 - 能量消耗

**灵巧操作**
- OpenAI的Dactyl系统学会单手解魔方
- 使用大规模并行强化学习

**自动驾驶**
- 路径规划和决策制定
- 安全性是关键考虑因素

### 30.6.3 游戏AI

**Atari游戏**
- DQN在49款Atari游戏中达到人类水平
- 直接从像素学习，无需游戏规则

**Dota 2**
- OpenAI Five击败职业战队
- 使用大规模分布式PPO训练

**星际争霸**
- AlphaStar达到宗师水平
- 结合深度强化学习和模仿学习

### 30.6.4 大语言模型与RLHF

强化学习最近的突破性应用是**基于人类反馈的强化学习（RLHF）**，用于训练ChatGPT等大语言模型：

**三阶段流程**：
1. **预训练**：在大规模文本上训练基础模型
2. **SFT（监督微调）**：在人类标注的对话数据上微调
3. **RLHF**：用PPO优化模型，奖励模型基于人类偏好

**RLHF的关键**：
- 训练一个**奖励模型**来预测人类偏好
- 使用PPO优化语言模型以最大化奖励
- 加入KL散度约束，防止模型偏离太远

这使得AI助手能够更好地遵循人类指令，产生更有用、更安全的回答。

---

## 练习题

### 基础题（3道）

**练习30.1**：贝尔曼方程验证

考虑一个简单的MDP：
- 状态：A, B（终止状态）
- 从A执行"右"动作：以概率1转移到B，获得奖励+5
- 折扣因子 $\gamma = 0.9$

如果最优策略是从A执行"右"动作到B，请计算 $V^*(A)$ 和 $Q^*(A, \text{右})$。

<details>
<summary>答案提示</summary>

$V^*(B) = 0$（终止状态）

根据贝尔曼最优方程：
$Q^*(A, \text{右}) = R + \gamma V^*(B) = 5 + 0.9 \times 0 = 5$

$V^*(A) = \max_a Q^*(A, a) = 5$

</details>

---

**练习30.2**：Q-Learning更新计算

给定：
- 当前Q值：$Q(s_1, a_1) = 3.0$
- 学习率：$\alpha = 0.1$
- 折扣因子：$\gamma = 0.9$
- 经验：$(s_1, a_1, r=2, s_2, \text{done}=\text{False})$
- $\max_{a'} Q(s_2, a') = 5.0$

请计算新的 $Q(s_1, a_1)$ 值。

<details>
<summary>答案提示</summary>

目标Q值：$y = r + \gamma \max_{a'} Q(s_2, a') = 2 + 0.9 \times 5 = 6.5$

TD误差：$\delta = y - Q(s_1, a_1) = 6.5 - 3.0 = 3.5$

新Q值：$Q(s_1, a_1) \leftarrow 3.0 + 0.1 \times 3.5 = 3.35$

</details>

---

**练习30.3**：ε-贪心策略

一个智能体使用ε-贪心策略，当前$\epsilon = 0.2$，有4个可选动作。

如果Q值为：$Q(s, a_1)=5, Q(s, a_2)=8, Q(s, a_3)=3, Q(s, a_4)=6$

请计算：
1. 选择最优动作（$a_2$）的概率
2. 选择非最优动作（如$a_1$）的概率

<details>
<summary>答案提示</summary>

1. 选择最优动作的概率：$P(a_2) = 1 - \epsilon + \frac{\epsilon}{4} = 0.8 + 0.05 = 0.85$

2. 选择非最优动作的概率：$P(a_1) = P(a_3) = P(a_4) = \frac{\epsilon}{4} = 0.05$

</details>

---

### 进阶题（3道）

**练习30.4**：策略梯度推导

证明策略梯度定理中的对数技巧：

$$\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s)$$

并用此解释为什么策略梯度公式中使用 $\nabla_\theta \log \pi_\theta(a|s)$ 而不是直接计算 $\nabla_\theta \pi_\theta(a|s)$。

<details>
<summary>答案提示</summary>

证明：
$\nabla_\theta \log \pi_\theta(a|s) = \frac{1}{\pi_\theta(a|s)} \cdot \nabla_\theta \pi_\theta(a|s)$

因此：$\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s)$

使用对数梯度的优点：
1. **数值稳定**：softmax输出很小，直接梯度会有数值问题
2. **直观解释**：梯度方向表示"如何改变以增加该动作的概率"
3. **方差减小**：与重要性采样相关

</details>

---

**练习30.5**：经验回放的作用

解释为什么DQN中的经验回放（Experience Replay）能够：
1. 打破样本相关性
2. 提高数据效率
3. 平滑数据分布

如果不用经验回放，直接在线更新会有什么潜在问题？

<details>
<summary>答案提示</summary>

1. **打破相关性**：连续经验高度相关（时间序列），神经网络假设样本独立同分布。随机采样打破时间相关性。

2. **提高数据效率**：每个经验可以被多次采样用于训练，而不是只用一次就丢弃。

3. **平滑数据分布**：随机采样使得每个batch包含不同时间点的经验，避免连续的极端样本导致训练不稳定。

不用经验回放的潜在问题：
- 训练不稳定，神经网络发散
- 样本利用效率低
- 可能陷入局部最优

</details>

---

**练习30.6**：Actor-Critic优势分析

比较REINFORCE、Q-Learning和Actor-Critic三种方法：

| 特性 | REINFORCE | Q-Learning | Actor-Critic |
|-----|-----------|-----------|--------------|
| 在线/离线 | ? | ? | ? |
| 离散/连续动作 | ? | ? | ? |
| 方差高低 | ? | ? | ? |
| 内存需求 | ? | ? | ? |

请填写表格并解释为什么Actor-Critic通常被认为是一个较好的折中选择。

<details>
<summary>答案提示</summary>

| 特性 | REINFORCE | Q-Learning | Actor-Critic |
|-----|-----------|-----------|--------------|
| 在线/离线 | 在线 | 离线 | 在线 |
| 离散/连续动作 | 两者皆可 | 离散 | 两者皆可 |
| 方差高低 | 高 | 中 | 低 |
| 内存需求 | 低 | 高（Q表或网络） | 中 |

Actor-Critic的优势：
- 可以处理连续动作（vs Q-Learning）
- 使用Critic减小方差（vs REINFORCE）
- 在线更新，不需要大量内存

</details>

---

### 挑战题（2道）

**练习30.7**：实现Double DQN

Double DQN是对DQN的改进，解决Q值高估问题。其核心思想是：
- 用**行为网络**选择最优动作：$a^* = \arg\max_a Q(s', a; \theta)$
- 用**目标网络**评估该动作的Q值：$y = r + \gamma Q(s', a^*; \theta^-)$

请修改本章的DQN代码，实现Double DQN，并在Grid World环境中比较原始DQN和Double DQN的性能差异。

<details>
<summary>答案提示</summary>

关键修改在`learn`方法中计算目标Q值的部分：

```python
# Double DQN
if double_dqn:
    # 用行为网络选择动作
    next_q_behavior = self.q_network.forward(next_state_vecs)
    best_actions = np.argmax(next_q_behavior, axis=1)
    
    # 用目标网络评估
    next_q_target = self.target_network.forward(next_state_vecs)
    max_next_q = next_q_target[np.arange(batch_size), best_actions]
else:
    # 原始DQN
    next_q = self.target_network.forward(next_state_vecs)
    max_next_q = np.max(next_q, axis=1)
```

</details>

---

**练习30.8**：设计奖励塑造函数

在Grid World环境中，假设终点在右下角，智能体容易陷入局部最优或探索效率低。

设计一个**奖励塑造（Reward Shaping）**函数：

$$R'(s, a, s') = R(s, a, s') + F(s, s')$$

使得学习更快更稳定，但**不改变最优策略**。

提示：考虑到目标的曼哈顿距离。

<details>
<summary>答案提示</summary>

可以使用势函数（Potential-based Reward Shaping）：

$$F(s, s') = \gamma \Phi(s') - \Phi(s)$$

其中$\Phi(s)$是状态势函数。选择$\Phi(s)$为到目标的负曼哈顿距离：

$$\Phi(s) = -(|s_{row} - goal_{row}| + |s_{col} - goal_{col}|)$$

这鼓励智能体向目标靠近，但不会改变最优策略（定理保证）。

实现：
```python
def potential(state, goal):
    return -(abs(state[0] - goal[0]) + abs(state[1] - goal[1]))

def shaped_reward(r, s, s_next, done, gamma, goal):
    if done:
        return r  # 终止状态不加塑造
    return r + gamma * potential(s_next, goal) - potential(s, goal)
```

</details>

---

## 本章小结

在这一章中，我们一起探索了强化学习这个令人兴奋的领域：

**理论基础**：
- 马尔可夫决策过程（MDP）为强化学习提供了数学框架
- 贝尔曼方程刻画了最优价值的递归结构
- 价值函数（$V$和$Q$）和策略是RL的核心概念

**经典算法**：
- **Q-Learning**：离策略的时序差分方法，通过Q表学习最优动作价值
- **DQN**：将深度学习与Q-Learning结合，使用神经网络近似Q函数
- **策略梯度**：直接优化策略参数，适用于连续动作空间
- **Actor-Critic**：结合价值方法和策略方法，使用优势函数减小方差

**实践要点**：
- ε-贪心平衡探索与利用
- 经验回放和目标网络稳定DQN训练
- 基线和优势函数减小策略梯度方差

**前沿应用**：
- AlphaGo/AlphaZero在围棋和棋类游戏中的突破
- 机器人控制和自动驾驶
- RLHF训练大语言模型（如ChatGPT）

强化学习的核心思想——**通过试错和奖励来学习最优行为**——不仅是人工智能的重要范式，也是理解智能本身的一扇窗口。从训练小狗到训练超级AI，这个原理贯穿始终。

正如Sutton和Barto在《Reinforcement Learning: An Introduction》中所说：*"强化学习是第一个真正理解智能的计算范式。"*

希望这一章能为你打开强化学习的大门，期待看到你在这个领域创造出惊人的成果！

---

## 参考文献

Bellman, R. (1957). A Markovian decision process. *Journal of Mathematics and Mechanics*, 6(5), 679-684.

Konda, V. R., & Tsitsiklis, J. N. (2000). Actor-critic algorithms. *Advances in Neural Information Processing Systems*, 12, 1008-1014.

Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., ... & Wierstra, D. (2015). Continuous control with deep reinforcement learning. *arXiv preprint arXiv:1509.02971*.

Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing Atari with deep reinforcement learning. *arXiv preprint arXiv:1312.5602*.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.

Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., ... & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. *International Conference on Machine Learning*, 1928-1937.

Schulman, J., Levine, S., Moritz, P., Jordan, M. I., & Abbeel, P. (2015). Trust region policy optimization. *International Conference on Machine Learning*, 1889-1897.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., ... & Hassabis, D. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484-489.

Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., ... & Hassabis, D. (2017). Mastering the game of Go without human knowledge. *Nature*, 550(7676), 354-359.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

Sutton, R. S., McAllester, D. A., Singh, S. P., & Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. *Advances in Neural Information Processing Systems*, 13, 1057-1063.

Watkins, C. J. C. H. (1989). *Learning from delayed rewards* (PhD thesis). King's College, Cambridge.

Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.

Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3-4), 229-256.

---

*本章完。在下一章中，我们将探索图神经网络——让AI学会处理图结构数据的强大工具。*
