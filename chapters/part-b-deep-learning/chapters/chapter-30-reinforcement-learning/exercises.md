# 第三十章 强化学习基础 练习题

## 练习题 1: 马尔可夫决策过程建模 (⭐)

**目标**: 理解MDP的基本概念

**题目**: 
将"井字棋"游戏建模为MDP：

**要求**: 
1. 定义状态空间 S（描述一个游戏局面）
2. 定义动作空间 A（玩家可以做什么）
3. 定义转移概率 P（假设对手随机下棋）
4. 定义奖励函数 R（赢/输/平的奖励）
5. 定义折扣因子 γ（为什么通常设为1？）

**思考问题**: 
- 井字棋是有限MDP还是无限MDP？
- 如果对手不是随机的，而是最优策略，转移概率会怎样？

---

## 练习题 2: 策略迭代手动计算 (⭐⭐)

**目标**: 理解策略评估和策略改进

**题目**: 
考虑一个3x3的网格世界：

```
┌───┬───┬───┐
│ S │   │ +1│
├───┼───┼───┤
│   │ X │ -1│
├───┼───┼───┤
│   │   │   │
└───┴───┴───┘
```

- S: 起点
- X: 障碍物（不可进入）
- +1: 目标（奖励+1，结束）
- -1: 陷阱（奖励-1，结束）
- 其他: 每步奖励-0.04

**初始策略**: 随机策略（上下左右各25%）

**任务**: 
1. 计算起点 (0,0) 的状态价值 V(s)（假设γ=0.9，迭代3轮）
2. 基于当前价值函数，进行策略改进
3. 新策略是什么？

---

## 练习题 3: Q-Learning实现 (⭐⭐⭐)

**目标**: 实现基础Q-Learning算法

**题目**: 
实现Q-Learning解决网格世界问题：

```python
import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
    
    def choose_action(self, state):
        # ε-贪心策略
        pass
    
    def update(self, state, action, reward, next_state):
        # Q-Learning更新
        # Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        pass
```

**要求**: 
1. 完成`choose_action`和`update`方法
2. 在一个5x5网格世界上训练1000回合
3. 记录并绘制Q值收敛曲线
4. 测试最终策略（设ε=0）

---

## 练习题 4: 探索 vs 利用 (⭐⭐)

**目标**: 理解ε-贪心和其他探索策略

**题目**: 
实现并比较以下探索策略：

1. **ε-贪心**: 以ε概率随机，1-ε概率选最优
2. **衰减ε**: ε随时间衰减，如 ε = ε₀ / (1 + episode/100)
3. **Boltzmann探索**: 根据Q值概率选择
   $$P(a) = \frac{e^{Q(s,a)/T}}{\sum_{a'} e^{Q(s,a')/T}}$$

**实验**: 
- 在同一环境上运行三种策略各1000回合
- 比较：
  - 累计奖励随回合的变化
  - 最终策略质量
  - 收敛速度

**思考问题**: 
- 哪种策略收敛最快？
- 哪种策略最终性能最好？
- 温度参数T如何影响Boltzmann探索？

---

## 练习题 5: 时序差分对比 (⭐⭐⭐)

**目标**: 理解SARSA和Q-Learning的区别

**题目**: 
实现SARSA和Q-Learning，在"悬崖行走"环境上对比：

```
┌───┬───┬───┬───┬───┬───┐
│ S │   │   │   │   │ G │
│   │   │   │   │   │   │
│ X │ X │ X │ X │ X │ X │
└───┴───┴───┴───┴───┴───┘
```

- S: 起点
- G: 目标（奖励+1）
- X: 悬崖（奖励-100，回到起点）

**任务**: 
1. 实现SARSA（on-policy）
2. 实现Q-Learning（off-policy）
3. 对比两种算法学习到的路径：
   - SARSA通常学什么路径？（安全还是最优？）
   - Q-Learning通常学什么路径？
4. 解释为什么不同（on-policy vs off-policy）

---

## 练习题 6: 多臂老虎机 (⭐⭐)

**目标**: 理解非关联性决策

**题目**: 
实现一个5臂老虎机，每个臂的真实价值服从N(0,1)：

```python
import numpy as np

class MultiArmedBandit:
    def __init__(self, n_arms=5):
        self.true_values = np.random.normal(0, 1, n_arms)
        self.best_arm = np.argmax(self.true_values)
    
    def pull(self, arm):
        # 返回奖励：真实价值 + N(0,1)噪声
        return np.random.normal(self.true_values[arm], 1)
```

**实现以下算法**：
1. **贪婪算法**（ε=0）
2. **ε-贪心**（ε=0.1）
3. **乐观初始值**（初始Q=5）
4. **UCB**（上置信界）

**对比**: 
- 运行1000步，记录累计奖励
- 每种算法多运行100次，取平均
- 绘制"累计奖励-步数"曲线

---

## 练习题 7: 资格迹 (⭐⭐⭐⭐)

**目标**: 理解TD(λ)算法

**题目**: 
实现SARSA(λ)算法：

```python
class SARSALambdaAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.9, lambda_=0.9):
        self.q_table = np.zeros((n_states, n_actions))
        self.e_trace = np.zeros((n_states, n_actions))  # 资格迹
        self.lr = lr
        self.gamma = gamma
        self.lambda_ = lambda_
    
    def update(self, state, action, reward, next_state, next_action):
        # SARSA(λ)更新
        # 1. 计算TD误差
        # 2. 更新资格迹
        # 3. 更新Q表
        pass
```

**实验**: 
- 比较SARSA(λ=0)、SARSA(λ=0.5)、SARSA(λ=0.9)
- 观察不同λ对收敛速度和最终性能的影响

**思考问题**: 
- λ=0时，SARSA(λ)退化成什么？
- λ=1时，类似于什么方法？

---

## 练习题 8: Dyna-Q规划 (⭐⭐⭐)

**目标**: 理解学习与规划的结合

**题目**: 
实现Dyna-Q算法：

```python
class DynaQAgent:
    def __init__(self, n_states, n_actions, n_planning=5):
        self.q_table = np.zeros((n_states, n_actions))
        self.model = {}  # 环境模型 (s,a) -> (r, s')
        self.n_planning = n_planning
    
    def learn(self, state, action, reward, next_state):
        # 1. 直接强化学习（Q-Learning）
        # 2. 更新模型
        # 3. 规划：从模型中随机采样n_planning次进行虚拟更新
        pass
```

**对比实验**: 
- 比较Q-Learning（n_planning=0）和Dyna-Q（n_planning=5, 10, 50）
- 观察规划步数对学习速度的影响

---

## 练习题 9: 强化学习完整项目 (⭐⭐⭐⭐⭐)

**目标**: 综合应用本章知识

**题目**: 
实现一个完整的RL系统，解决"迷宫寻路"问题：

**环境要求**: 
- 10x10迷宫，随机生成（20%墙壁）
- 起点和终点随机
- 动作：上下左右
- 奖励：每步-0.1，撞墙-1，到达终点+10

**实现要求**: 

1. **环境类** `MazeEnv`:
   - reset(): 重置环境
   - step(action): 执行动作，返回(s', r, done, info)
   - render(): 可视化（可选）

2. **智能体类**: 
   - 实现至少两种算法：Q-Learning和SARSA
   - 支持不同探索策略

3. **训练与评估**:
   - 训练5000回合
   - 每100回合评估一次（ε=0）
   - 绘制学习曲线

4. **消融实验**:
   - 不同学习率的影响
   - 不同探索策略的影响
   - 不同折扣因子的影响

5. **可视化**:
   - 绘制最优策略路径
   - 绘制Q值热力图

**交付物**: 
- 完整代码
- 实验报告（包含图表和分析）
- 最终演示（GIF或视频）

---

## 参考答案

### 练习3 Q-Learning参考实现

```python
def choose_action(self, state):
    if np.random.random() < self.epsilon:
        return np.random.randint(self.n_actions)
    return np.argmax(self.q_table[state])

def update(self, state, action, reward, next_state):
    current_q = self.q_table[state, action]
    max_next_q = np.max(self.q_table[next_state])
    new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
    self.q_table[state, action] = new_q
```

### 练习5 SARSA vs Q-Learning

- **SARSA** (on-policy): 学习实际执行的动作序列，通常选择更安全的路径
- **Q-Learning** (off-policy): 学习最优策略，即使有风险也选择最短路径

在悬崖行走中：
- SARSA会学到远离悬崖的路径（保险）
- Q-Learning会学到沿着悬崖的最短路径（冒险）

---

**学习建议**: 
- 练习1-3是基础，必须完成
- 练习4-6是进阶，理解探索策略
- 练习7-9是实战，完成后可独立解决RL问题

**推荐资源**: 
- OpenAI Gym: 标准环境库
- Stable-Baselines3: 高级RL算法实现
- "Reinforcement Learning: An Introduction" (Sutton & Barto)
