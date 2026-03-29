# 第二十章 优化器——更快更好的下降

> *"优化是一门艺术，在无数可能中寻找最优的路径。"
> 
> —— 莱昂·博图 (Léon Bottou)

## 开场故事：下山竞赛

想象你站在珠穆朗玛峰的山顶，浓雾弥漫，能见度为零。你的目标只有一个：**以最快的速度安全到达山脚**。

这不是虚构的场景——这正是深度学习训练过程的写照！山峰代表损失函数的"高地"，山脚代表最优解，而你就是优化器。

现在，有五个人决定比赛，看谁能最先到达山脚：

**小明**是个谨慎的人。他每走一步都要仔细观察地面，朝着最陡的下坡方向迈出一小步。他走得很稳，但速度很慢。他叫**SGD**（随机梯度下降）。

**小红**滑雪下山。她利用惯性，一旦开始下滑就保持速度，即使遇到小坑也不会停下来。她叫**Momentum**（动量法）。

**小蓝**是个细心的人。他会记住每个地方的陡峭程度，在经常很陡的地方迈小步，在平缓的地方迈大步。他叫**AdaGrad**（自适应梯度）。

**小绿**是小蓝的改进版。他不会一直记住所有历史，而是只关注最近的路况。他叫**RMSprop**。

**小黄**结合了小红和小绿的优点——既有惯性，又会根据路况调整步伐。他叫**Adam**（自适应矩估计），也是目前最受欢迎的选手。

这场比赛谁会赢？让我们拭目以待！

---

## 20.1 为什么需要优化器？

### 20.1.1 回顾梯度下降

在第四章，我们学习了梯度下降的基本思想：

```
参数新值 = 参数旧值 - 学习率 × 梯度
```

数学公式为：

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t)$$

其中：
- $\theta_t$ 是第 $t$ 步的参数值
- $\eta$ 是学习率（步长）
- $\nabla_\theta J(\theta_t)$ 是损失函数对参数的梯度

这个方法很直观，但它有一个致命问题：**如何选择学习率？**

```
📊 学习率选择的困境

损失
  │
  │      ⛰️          ⛰️  学习率太大：
  │     /  \        /  \    在谷底两侧来回震荡
  │    /    \      /    \   永远停不下来
  │   /      \    /      \
  │  /   🏔️   \  /   🏔️   \
  │ /  最优解  \/         \
  │/                       \
  └──────────────────────────► 参数值
  
  
损失
  │
  │      ⛰️
  │     /  \
  │    /    \
  │   /      \        🐢 学习率太小：
  │  /   🏔️   \          收敛速度极慢
  │ /  最优解  \         可能需要百万步
  │/            \     
  └──────────────────────────► 参数值
```

**问题来了**：
- 如果学习率太大，算法会在最优解附近震荡，甚至发散
- 如果学习率太小，收敛速度极慢，训练可能需要几天甚至几周
- 更糟糕的是，不同参数可能需要不同的学习率！

### 20.1.2 优化器登场

优化器的核心使命就是解决这个问题。让我们看看不同的优化器是如何应对这个挑战的：

| 优化器 | 核心思想 | 主要优势 | 发布时间 |
|--------|---------|---------|----------|
| SGD | 基础梯度下降 | 简单、稳定 | 1951 |
| Momentum | 引入惯性 | 加速收敛、减少震荡 | 1964 |
| AdaGrad | 自适应学习率 | 适合稀疏数据 | 2011 |
| RMSprop | 指数移动平均 | 解决AdaGrad学习率衰减过快 | 2012 |
| Adam | Momentum + RMSprop | 快速、稳定、通用 | 2014 |

```
📈 优化器发展时间线

1951    1964    2011    2012    2014
 │       │       │       │       │
 ▼       ▼       ▼       ▼       ▼
SGD ──► Momentum  │       │       │
                 ▼       ▼       ▼
              AdaGrad  RMSprop  Adam
                 │       │       │
                 └───────┴───────┘
                    演进融合
```

---

## 20.2 SGD：稳健的基准

### 20.2.1 算法原理

SGD（Stochastic Gradient Descent，随机梯度下降）是最基础的优化器。它的更新规则简单直接：

$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$

其中 $g_t = \nabla_\theta J(\theta_t; x^{(i)}, y^{(i)})$ 是在单个样本（或小批量）上计算的梯度。

```
🎯 SGD的直觉理解

想象你在一个山谷中摸索前进：

1. 站在当前位置 (θₜ)
2. 感受脚下的坡度 (gₜ)
3. 朝着下坡方向走一小步 (η·gₜ)
4. 重复直到到达谷底

    ⛰️
   /  \
  / 🚶 \
 /  θₜ  \
/        \
    ↓ 迈一小步
    
   ⛰️
  /  \
 /    \
/  🚶   \
   θₜ₊₁
```

### 20.2.2 从零实现SGD

```python
"""
SGD优化器从零实现
================
最基础的优化器，但也是最常用的baseline
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional


class SGD:
    """
    随机梯度下降优化器
    
    参数:
        lr: 学习率 (learning rate)
    """
    
    def __init__(self, lr: float = 0.01):
        """
        初始化SGD优化器
        
        参数:
            lr: 学习率，控制每次更新的步长
        """
        self.lr = lr
        self.name = "SGD"
        self.history = []  # 记录优化轨迹
    
    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        执行一次参数更新
        
        参数:
            params: 当前参数值，形状为 (n,)
            grad: 当前梯度，形状为 (n,)
        
        返回:
            new_params: 更新后的参数值
        
        数学公式:
            θ_{t+1} = θ_t - η * ∇J(θ_t)
        """
        # 记录当前位置
        self.history.append(params.copy())
        
        # 核心更新规则
        new_params = params - self.lr * grad
        
        return new_params
    
    def reset(self):
        """重置优化器状态"""
        self.history = []


def test_sgd():
    """测试SGD优化器的基本功能"""
    print("=" * 60)
    print("SGD优化器基本功能测试")
    print("=" * 60)
    
    # 创建一个简单的二次函数: f(x, y) = x² + 2y²
    def loss_fn(params):
        return params[0]**2 + 2 * params[1]**2
    
    def grad_fn(params):
        return np.array([2 * params[0], 4 * params[1]])
    
    # 初始化优化器
    sgd = SGD(lr=0.1)
    
    # 从远处开始
    params = np.array([5.0, 3.0])
    
    print(f"初始参数: {params}, 损失: {loss_fn(params):.4f}")
    
    # 优化50步
    for i in range(50):
        grad = grad_fn(params)
        params = sgd.step(params, grad)
        if (i + 1) % 10 == 0:
            print(f"第{i+1}步: 参数=[{params[0]:.4f}, {params[1]:.4f}], "
                  f"损失={loss_fn(params):.4f}")
    
    print(f"最终参数: {params}, 损失: {loss_fn(params):.6f}")
    print("✓ SGD基本功能测试通过\n")


if __name__ == "__main__":
    test_sgd()
```

### 20.2.3 SGD的优缺点

```
✅ SGD的优点                    ❌ SGD的缺点
─────────────────────────────────────────────────
• 实现极其简单                    • 收敛速度慢
• 内存占用极小                    • 容易陷入局部最优
• 泛化性能通常很好                • 学习率难调
• 适合大规模数据                  • 在峡谷/鞍点处震荡
• 理论基础扎实                    • 对参数缩放敏感
```

**为什么SGD仍然被广泛使用？**

尽管有更先进的优化器，SGD在深度学习领域仍然占据重要地位，原因是：

1. **泛化能力**：研究表明，SGD找到的解往往有更好的泛化性能
2. **简单稳定**：没有额外的超参数需要调节
3. **理论保证**：收敛性分析最成熟

---

## 20.3 Momentum：给优化器装上"引擎"

### 20.3.1 物理直觉

想象一个球从山上滚下来：

- **SGD**就像一个没有质量的小点，每一步只依赖当前坡度
- **Momentum**就像一个有质量的球，会累积速度，即使遇到小坑也能冲过去

```
🎱 Momentum的物理直觉

场景1：SGD vs Momentum 在平缓区域

SGD:              Momentum:
  🚶                ⚽──→
  │
  │                  (球有惯性，保持速度)
  ↓
  (容易停下来，
   需要很多步)


场景2：SGD vs Momentum 在峡谷中

SGD:              Momentum:
    ↗ 🚶            ↗──→⚽
   ↗  ↘           ↗     
  ↗    ↘         ↗      (惯性帮助穿越)
 ↗      ↘       ↗
(来回震荡)      (平滑前进)
```

### 20.3.2 数学原理

Momentum引入了一个速度变量 $v$，记录历史梯度的累积：

$$v_t = \gamma v_{t-1} + \eta \cdot g_t$$

$$\theta_{t+1} = \theta_t - v_t$$

其中：
- $\gamma$ 是动量系数（通常设为 0.9）
- $v_t$ 是第 $t$ 步的速度
- $g_t$ 是当前梯度

**展开速度公式，我们可以看到指数加权平均的本质：**

$$v_t = \eta \cdot g_t + \gamma \eta \cdot g_{t-1} + \gamma^2 \eta \cdot g_{t-2} + ...$$

这意味着**最近的梯度贡献最大，但历史梯度也有影响**。

```
📊 指数加权平均的权重分布 (γ = 0.9)

权重
  │
  │████
  │███████
  │█████████
  │████████████
  │███████████████
  │███████████████████
  │███████████████████████
  │███████████████████████████  ← 当前梯度权重最大
  └──────────────────────────────► 时间
   很早    较早前   最近    现在
```

### 20.3.3 从零实现Momentum

```python
"""
Momentum优化器从零实现
=====================
引入动量的概念，加速收敛并减少震荡
"""

import numpy as np


class Momentum:
    """
    动量优化器（Polyak, 1964）
    
    物理直觉：像滚下山坡的球一样，累积动量帮助穿越平坦区域和峡谷
    
    参数:
        lr: 学习率
        momentum: 动量系数，通常设为0.9
    """
    
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        """
        初始化Momentum优化器
        
        参数:
            lr: 学习率
            momentum: 动量系数 (0到1之间)
                     0.9表示保留90%的历史速度
        """
        self.lr = lr
        self.momentum = momentum
        self.name = "Momentum"
        self.velocity = None  # 速度变量
        self.history = []
    
    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        执行一次参数更新
        
        参数:
            params: 当前参数值
            grad: 当前梯度
        
        返回:
            new_params: 更新后的参数值
        
        数学公式:
            v_t = γ * v_{t-1} + η * g_t
            θ_{t+1} = θ_t - v_t
        """
        self.history.append(params.copy())
        
        # 初始化速度
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        # 核心：更新速度（指数加权移动平均）
        # 保留一部分历史速度，加上当前梯度
        self.velocity = self.momentum * self.velocity + self.lr * grad
        
        # 用速度更新参数
        new_params = params - self.velocity
        
        return new_params
    
    def reset(self):
        """重置优化器状态"""
        self.velocity = None
        self.history = []
    
    def get_velocity(self) -> np.ndarray:
        """获取当前速度（用于可视化）"""
        return self.velocity.copy() if self.velocity is not None else None


def visualize_momentum_vs_sgd():
    """
    可视化对比：Momentum vs SGD在峡谷中的表现
    """
    print("\n" + "=" * 70)
    print("Momentum vs SGD 可视化对比")
    print("=" * 70)
    
    # 创建一个狭长的峡谷函数
    # 等高线是拉长的椭圆
    def valley_loss(params):
        x, y = params
        # x方向梯度大，y方向梯度小
        return 0.5 * x**2 + 0.05 * y**2
    
    def valley_grad(params):
        x, y = params
        return np.array([x, 0.1 * y])
    
    # 对比实验
    sgd = SGD(lr=0.9)
    momentum = Momentum(lr=0.1, momentum=0.9)
    
    # 相同起点
    start = np.array([8.0, 8.0])
    
    sgd_params = start.copy()
    momentum_params = start.copy()
    
    sgd_losses = []
    momentum_losses = []
    
    # 运行优化
    for _ in range(100):
        # SGD
        sgd_params = sgd.step(sgd_params, valley_grad(sgd_params))
        sgd_losses.append(valley_loss(sgd_params))
        
        # Momentum
        momentum_params = momentum.step(momentum_params, valley_grad(momentum_params))
        momentum_losses.append(valley_loss(momentum_params))
    
    print(f"起点: {start}, 损失: {valley_loss(start):.4f}")
    print(f"\nSGD最终: 参数={sgd_params}, 损失={sgd_losses[-1]:.6f}")
    print(f"Momentum最终: 参数={momentum_params}, 损失={momentum_losses[-1]:.6f}")
    
    # 打印轨迹统计
    print(f"\n损失下降对比 (前20步):")
    for i in range(min(20, len(sgd_losses))):
        print(f"  第{i+1:2d}步: SGD={sgd_losses[i]:8.4f}, "
              f"Momentum={momentum_losses[i]:8.4f}")
    
    print("\n✓ Momentum在峡谷中收敛更快，震荡更少\n")


if __name__ == "__main__":
    visualize_momentum_vs_sgd()
```

### 20.3.4 Nesterov加速梯度（NAG）

Momentum的一个改进版本是**Nesterov Accelerated Gradient**（NAG）。它的核心思想是：**在看清楚未来的路之后，再决定如何加速**。

```
🎯 Nesterov vs 标准Momentum

标准Momentum:
1. 计算当前位置的梯度
2. 更新速度
3. 移动到新的位置

Nesterov:
1. 先"预览"一下如果按当前速度会到哪里
2. 在那个"未来位置"计算梯度
3. 用这个"未来梯度"修正速度
4. 移动

直观理解：
• 标准Momentum："基于我现在在哪，决定往哪走"
• Nesterov："基于我将会在哪，决定现在往哪走"

        标准Momentum           Nesterov
        
         θₜ ──→ θₜ₊₁           θₜ ──→ θₜ₊₁
          ↓                     ↓
        计算梯度               先"预览"
                              ↓
                            在θₜ + γ·vₜ处
                            计算梯度
```

Nesterov的更新公式：

$$v_t = \gamma v_{t-1} + \eta \cdot \nabla_\theta J(\theta_t - \gamma v_{t-1})$$

$$\theta_{t+1} = \theta_t - v_t$$

---

## 20.4 AdaGrad：为每个参数定制学习率

### 20.4.1 核心问题

想象你在一个山谷中行走：
- **陡峭的方向**：应该迈小步，避免越过谷底
- **平缓的方向**：应该迈大步，快速前进

但SGD对所有方向使用相同的学习率，这显然不合理！

```
📊 不同方向需要不同步长

        陡峭方向              平缓方向
          ↓                     ↓
    需要小步长            需要大步长
    
    ⛰️                      ⛰️
   /\                      /    \
  /  \                    /      \
 / 🚶 \                  /        🚶
/      \                /           \
         \              /            
          \            /
           🏁          🏁
```

### 20.4.2 算法原理

AdaGrad（Adaptive Gradient，自适应梯度）的核心思想是：

> **对于经常更新的参数（梯度大），减小学习率；
> 对于很少更新的参数（梯度小），增大学习率。**

具体实现：累积历史梯度的平方，用这个累积值来缩放学习率。

$$G_t = G_{t-1} + g_t \odot g_t$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$$

其中：
- $G_t$ 是累积的梯度平方
- $\odot$ 表示逐元素乘法
- $\epsilon$ 是一个很小的常数（如 $10^{-8}$），防止除以零

**直觉理解**：
- 如果某个参数的梯度一直很大，$G_t$ 会很大，有效学习率就小
- 如果某个参数的梯度一直很小，$G_t$ 会很小，有效学习率就大

### 20.4.3 从零实现AdaGrad

```python
"""
AdaGrad优化器从零实现
====================
自适应地为每个参数调整学习率
"""

import numpy as np


class AdaGrad:
    """
    AdaGrad优化器（Duchi et al., 2011）
    
    核心思想：根据历史梯度的累积平方，自适应调整每个参数的学习率
    适合：稀疏数据、特征频率差异大的场景
    
    参数:
        lr: 全局学习率
        eps: 数值稳定性常数
    """
    
    def __init__(self, lr: float = 0.01, eps: float = 1e-8):
        """
        初始化AdaGrad优化器
        
        参数:
            lr: 学习率
            eps: 防止除以零的小常数
        """
        self.lr = lr
        self.eps = eps
        self.name = "AdaGrad"
        self.cache = None  # 累积梯度平方
        self.history = []
    
    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        执行一次参数更新
        
        参数:
            params: 当前参数值
            grad: 当前梯度
        
        返回:
            new_params: 更新后的参数值
        
        数学公式:
            G_t = G_{t-1} + g_t ⊙ g_t
            θ_{t+1} = θ_t - (η / √(G_t + ε)) ⊙ g_t
        """
        self.history.append(params.copy())
        
        # 初始化累积平方
        if self.cache is None:
            self.cache = np.zeros_like(params)
        
        # 核心：累积梯度平方
        self.cache = self.cache + grad ** 2
        
        # 计算自适应学习率
        # 梯度大的方向：分母大，步长小
        # 梯度小的方向：分母小，步长大
        adaptive_lr = self.lr / (np.sqrt(self.cache) + self.eps)
        
        # 更新参数
        new_params = params - adaptive_lr * grad
        
        return new_params
    
    def reset(self):
        """重置优化器状态"""
        self.cache = None
        self.history = []
    
    def get_effective_lr(self) -> np.ndarray:
        """获取每个参数的有效学习率"""
        if self.cache is None:
            return None
        return self.lr / (np.sqrt(self.cache) + self.eps)


def demonstrate_adagrad_adaptivity():
    """
    演示AdaGrad的自适应能力
    场景：两个参数，一个梯度大，一个梯度小
    """
    print("\n" + "=" * 70)
    print("AdaGrad自适应性演示")
    print("=" * 70)
    
    # 创建一个各向异性函数：x方向陡峭，y方向平缓
    def anisotropic_loss(params):
        x, y = params
        # x方向梯度是y方向的10倍
        return 10 * x**2 + 0.1 * y**2
    
    def anisotropic_grad(params):
        x, y = params
        return np.array([20 * x, 0.2 * y])
    
    adagrad = AdaGrad(lr=0.5)
    sgd = SGD(lr=0.05)  # 为了稳定性，SGD需要更小的学习率
    
    # 从同一点开始
    start = np.array([5.0, 5.0])
    adagrad_params = start.copy()
    sgd_params = start.copy()
    
    print(f"起点: {start}")
    print(f"初始梯度: {anisotropic_grad(start)}")
    print(f"注意：x方向梯度(100)是y方向(1)的100倍！\n")
    
    print("前10步的有效学习率对比:")
    print("-" * 60)
    
    for i in range(10):
        # AdaGrad
        grad_ag = anisotropic_grad(adagrad_params)
        effective_lr = adagrad.get_effective_lr()
        adagrad_params = adagrad.step(adagrad_params, grad_ag)
        
        # SGD
        grad_sgd = anisotropic_grad(sgd_params)
        sgd_params = sgd.step(sgd_params, grad_sgd)
        
        if effective_lr is not None:
            print(f"第{i+1:2d}步: AdaGrad有效lr=[{effective_lr[0]:.6f}, {effective_lr[1]:.6f}]")
    
    print("\n观察：AdaGrad自动为x方向（梯度大）分配小学习率，")
    print("       为y方向（梯度小）分配大学习率\n")
    print("✓ AdaGrad自适应测试通过\n")


if __name__ == "__main__":
    demonstrate_adagrad_adaptivity()
```

### 20.4.4 AdaGrad的局限性

AdaGrad有一个致命缺点：**学习率单调递减**。

由于 $G_t$ 只增不减，有效学习率 $\frac{\eta}{\sqrt{G_t + \epsilon}}$ 会越来越小。这导致：
- 初期收敛快
- 后期几乎停止更新

```
📉 AdaGrad的学习率衰减问题

有效学习率
   │
   │\\
   │ \\
   │  \\
   │   \\
   │    \\
   │     \\
   │      \\
   │       \\
   │        \\
   │         \\
   └───────────► 迭代次数
   
问题：学习率持续衰减，后期几乎无法更新
```

这引出了下一个优化器——**RMSprop**。

---

## 20.5 RMSprop：忘记久远的过去

### 20.5.1 核心改进

RMSprop（Root Mean Square Propagation）是Hinton在他的Coursera课程中提出的（2012）。它解决了AdaGrad学习率单调递减的问题。

**关键洞察**：与其记住所有历史，不如只关注最近的过去。

```
📊 记忆策略对比

AdaGrad: "我记得一切"           RMSprop: "我主要关注最近"

累积所有历史梯度                指数移动平均
███████████████              ▓▓▓▓▓▓▓███
(权重相同)                   (近期权重更高)

    ↓                              ↓
  G持续增长                      G相对稳定
  学习率持续衰减                 学习率不会趋于零
```

### 20.5.2 算法原理

RMSprop用**指数移动平均**代替累积和：

$$G_t = \beta G_{t-1} + (1-\beta) g_t \odot g_t$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$$

通常 $\beta = 0.9$，意味着主要关注最近10步的梯度。

```
🧠 RMSprop的直觉

想象你是一个经验丰富的司机：
• AdaGrad会记住你开过的每一条路
• RMSprop主要关注最近的路况

当路况变化时：
• AdaGrad反应很慢（被历史拖累）
• RMSprop能快速适应
```

### 20.5.3 从零实现RMSprop

```python
"""
RMSprop优化器从零实现
=====================
使用指数移动平均解决AdaGrad学习率衰减问题
"""

import numpy as np


class RMSprop:
    """
    RMSprop优化器（Tieleman & Hinton, 2012）
    
    核心思想：用指数移动平均代替累积和，防止学习率过早衰减
    
    参数:
        lr: 学习率
        beta: 衰减率，通常0.9
        eps: 数值稳定性常数
    """
    
    def __init__(self, lr: float = 0.01, beta: float = 0.9, eps: float = 1e-8):
        """
        初始化RMSprop优化器
        
        参数:
            lr: 学习率
            beta: 指数衰减率 (0到1之间)
                   0.9表示保留90%的历史累积
            eps: 防止除以零的小常数
        """
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.name = "RMSprop"
        self.cache = None  # 梯度的指数移动平均平方
        self.history = []
    
    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        执行一次参数更新
        
        参数:
            params: 当前参数值
            grad: 当前梯度
        
        返回:
            new_params: 更新后的参数值
        
        数学公式:
            G_t = β·G_{t-1} + (1-β)·g_t²
            θ_{t+1} = θ_t - (η / √(G_t + ε)) ⊙ g_t
        """
        self.history.append(params.copy())
        
        # 初始化缓存
        if self.cache is None:
            self.cache = np.zeros_like(params)
        
        # 核心：指数移动平均
        # 保留大部分历史信息，但加入新的梯度信息
        self.cache = self.beta * self.cache + (1 - self.beta) * (grad ** 2)
        
        # 计算自适应学习率
        adaptive_lr = self.lr / (np.sqrt(self.cache) + self.eps)
        
        # 更新参数
        new_params = params - adaptive_lr * grad
        
        return new_params
    
    def reset(self):
        """重置优化器状态"""
        self.cache = None
        self.history = []


def compare_adagrad_rmsprop():
    """
    对比AdaGrad和RMSprop在长期训练中的表现
    展示AdaGrad学习率过早衰减的问题
    """
    print("\n" + "=" * 70)
    print("AdaGrad vs RMSprop 长期训练对比")
    print("=" * 70)
    
    # 简单的二次函数
    def simple_loss(params):
        return np.sum(params ** 2)
    
    def simple_grad(params):
        return 2 * params
    
    # 两个优化器
    adagrad = AdaGrad(lr=1.0)
    rmsprop = RMSprop(lr=0.1, beta=0.9)
    
    # 从远处开始
    start = np.array([10.0, 10.0])
    ag_params = start.copy()
    rms_params = start.copy()
    
    ag_losses = []
    rms_losses = []
    
    # 长期训练500步
    for _ in range(500):
        ag_params = adagrad.step(ag_params, simple_grad(ag_params))
        rms_params = rmsprop.step(rms_params, simple_grad(rms_params))
        
        ag_losses.append(simple_loss(ag_params))
        rms_losses.append(simple_loss(rms_params))
    
    print(f"起点: {start}, 损失: {simple_loss(start):.4f}")
    print(f"\n500步后:")
    print(f"  AdaGrad: 参数={ag_params}, 损失={ag_losses[-1]:.8f}")
    print(f"  RMSprop: 参数={rms_params}, 损失={rms_losses[-1]:.8f}")
    
    print(f"\n关键观察:")
    print(f"  AdaGrad前期收敛快，但后期停滞在第{np.argmin(ag_losses)+1}步左右")
    print(f"  RMSprop保持稳定下降，最终收敛到更接近最优的解")
    print("\n✓ RMSprop解决了AdaGrad学习率过早衰减的问题\n")


if __name__ == "__main__":
    compare_adagrad_rmsprop()
```

---

## 20.6 Adam：集大成者

### 20.6.1 算法概述

Adam（Adaptive Moment Estimation，自适应矩估计）是Kingma和Ba在2014年提出的。它结合了Momentum和RMSprop的优点：

- **Momentum**：累积动量（一阶矩），加速收敛
- **RMSprop**：累积梯度平方（二阶矩），自适应学习率

```
🎯 Adam的核心思想

Adam = Momentum + RMSprop

  Momentum: 记录"速度"（一阶矩）
  ════════════════════════════════
    v_t = β₁·v_{t-1} + (1-β₁)·g_t
  
  RMSprop: 记录"不确定性"（二阶矩）  
  ════════════════════════════════
    s_t = β₂·s_{t-1} + (1-β₂)·g_t²
  
  结合两者：
  ════════════════════════════════
    θ_{t+1} = θ_t - η·v_t̂ / (√s_t̂ + ε)
```

### 20.6.2 完整算法

Adam的更新步骤：

**步骤1：计算梯度**
$$g_t = \nabla_\theta J(\theta_t)$$

**步骤2：更新一阶矩（动量）**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

**步骤3：更新二阶矩（梯度平方的指数移动平均）**
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

**步骤4：偏差校正（Bias Correction）**
这是Adam的精妙之处！由于 $m_0$ 和 $v_0$ 初始化为0，初期估计会偏向0。偏差校正解决了这个问题：

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

**步骤5：更新参数**
$$\theta_{t+1} = \theta_t - \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### 20.6.3 偏差校正的直觉

为什么需要偏差校正？

```
📊 偏差校正的作用

初期（t很小）：
  m_t = (1-β₁)·g_t  （因为m₀=0）
  
  如果没有校正：m_t 被低估了 (1-β₁) 倍
  
  校正后：m̂_t = m_t / (1-β₁ᵗ) ≈ m_t / (1-β₁) = g_t
  
         ✓ 正确估计！

随着t增大：
  β₁ᵗ → 0，校正因子 (1-β₁ᵗ) → 1
  
  校正影响逐渐消失（这正是我们想要的）

可视化：
  未校正的估计    校正后的估计
      │                │
    ──┼──            ──┼──
      │   ↗            │↗
      │  ↗             │
      │ ↗              ├──────→ 真实值
      │↗               │
      └────────►       └────────►
       时间              时间
       
      (初期被低估)      (从一开始就是无偏的)
```

### 20.6.4 从零实现Adam

```python
"""
Adam优化器从零实现
==================
结合Momentum和RMSprop的优点
Kingma & Ba, 2014
"""

import numpy as np


class Adam:
    """
    Adam优化器（Kingma & Ba, 2014）
    
    全称：Adaptive Moment Estimation
    核心思想：同时估计梯度的一阶矩（均值）和二阶矩（未中心化的方差）
    
    参数:
        lr: 学习率，通常0.001
        beta1: 一阶矩衰减率，通常0.9
        beta2: 二阶矩衰减率，通常0.999
        eps: 数值稳定性常数，通常1e-8
    """
    
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, eps: float = 1e-8):
        """
        初始化Adam优化器
        
        参数:
            lr: 学习率
            beta1: 一阶矩指数衰减率
            beta2: 二阶矩指数衰减率
            eps: 防止除以零的小常数
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.name = "Adam"
        
        # 状态变量
        self.m = None  # 一阶矩（动量）
        self.v = None  # 二阶矩（梯度平方的移动平均）
        self.t = 0     # 时间步（用于偏差校正）
        self.history = []
    
    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        执行一次参数更新
        
        参数:
            params: 当前参数值，形状 (n,)
            grad: 当前梯度，形状 (n,)
        
        返回:
            new_params: 更新后的参数值
        
        完整算法：
            g_t = ∇J(θ_t)
            m_t = β₁·m_{t-1} + (1-β₁)·g_t
            v_t = β₂·v_{t-1} + (1-β₂)·g_t²
            m̂_t = m_t / (1-β₁ᵗ)    ← 偏差校正
            v̂_t = v_t / (1-β₂ᵗ)    ← 偏差校正
            θ_{t+1} = θ_t - η·m̂_t / (√v̂_t + ε)
        """
        self.history.append(params.copy())
        self.t += 1
        
        # 初始化矩估计
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        # 步骤2：更新一阶矩（动量）
        # 类似于Momentum，记录历史梯度的加权平均
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        
        # 步骤3：更新二阶矩（梯度平方的指数移动平均）
        # 类似于RMSprop，记录梯度幅度的历史
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        # 步骤4：偏差校正
        # 解决初始化时m和v为0导致的初期估计偏差
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # 步骤5：更新参数
        # 结合动量方向和自适应学习率
        new_params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        return new_params
    
    def reset(self):
        """重置优化器状态"""
        self.m = None
        self.v = None
        self.t = 0
        self.history = []
    
    def get_moments(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取当前的一阶矩和二阶矩（用于调试）"""
        if self.m is None:
            return None, None
        return self.m.copy(), self.v.copy()


def test_adam_basic():
    """测试Adam优化器的基本功能"""
    print("=" * 70)
    print("Adam优化器基本功能测试")
    print("=" * 70)
    
    # 简单的二次函数
    def loss_fn(params):
        return np.sum(params ** 2)
    
    def grad_fn(params):
        return 2 * params
    
    # 创建Adam优化器
    adam = Adam(lr=0.1, beta1=0.9, beta2=0.999)
    
    # 从远处开始
    params = np.array([5.0, 5.0])
    
    print(f"初始参数: {params}, 损失: {loss_fn(params):.4f}")
    
    # 优化
    for i in range(100):
        grad = grad_fn(params)
        params = adam.step(params, grad)
        
        if (i + 1) % 20 == 0:
            print(f"第{i+1:3d}步: 参数=[{params[0]:.6f}, {params[1]:.6f}], "
                  f"损失={loss_fn(params):.10f}")
    
    print(f"\n最终参数: {params}")
    print(f"最终损失: {loss_fn(params):.12f}")
    print("✓ Adam基本功能测试通过\n")


if __name__ == "__main__":
    test_adam_basic()
```

### 20.6.5 为什么Adam这么受欢迎？

```
🌟 Adam的优势

1. 快速收敛
   ├── 动量帮助穿越平坦区域
   └── 自适应学习率帮助处理不同尺度的参数

2. 鲁棒性
   ├── 对学习率不太敏感
   ├── 自动处理稀疏梯度
   └── 适合非平稳目标

3. 内存效率高
   └── 只需要存储m和v两个向量

4. 实现简单
   └── 代码清晰，易于理解和修改

5. 理论保证
   └── 在凸优化问题上有收敛性证明

实际表现：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
在大多数深度学习任务中，Adam都是默认选择
特别是：NLP、计算机视觉、强化学习
```

---

## 20.7 五大优化器对比实验

### 20.7.1 可视化优化轨迹

```python
"""
五大优化器综合对比实验
======================
在多种测试函数上对比SGD、Momentum、AdaGrad、RMSprop、Adam
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


class OptimizerBenchmark:
    """
    优化器基准测试套件
    """
    
    @staticmethod
    def beale_function(params):
        """
        Beale函数 - 非凸优化测试函数
        全局最小值在 (3, 0.5)，值为0
        """
        x, y = params
        term1 = (1.5 - x + x * y) ** 2
        term2 = (2.25 - x + x * y ** 2) ** 2
        term3 = (2.625 - x + x * y ** 3) ** 2
        return term1 + term2 + term3
    
    @staticmethod
    def beale_gradient(params):
        """Beale函数的梯度"""
        x, y = params
        # df/dx
        dfdx = 2 * (1.5 - x + x*y) * (-1 + y) + \
               2 * (2.25 - x + x*y**2) * (-1 + y**2) + \
               2 * (2.625 - x + x*y**3) * (-1 + y**3)
        # df/dy
        dfdy = 2 * (1.5 - x + x*y) * x + \
               2 * (2.25 - x + x*y**2) * (2*x*y) + \
               2 * (2.625 - x + x*y**3) * (3*x*y**2)
        return np.array([dfdx, dfdy])
    
    @staticmethod
    def rosenbrock_function(params, a=1, b=100):
        """
        Rosenbrock函数（香蕉函数）
        全局最小值在 (a, a²)，值为0
        特点：窄而弯曲的山谷，是优化算法的经典挑战
        """
        x, y = params
        return (a - x)**2 + b * (y - x**2)**2
    
    @staticmethod
    def rosenbrock_gradient(params, a=1, b=100):
        """Rosenbrock函数的梯度"""
        x, y = params
        dfdx = -2*(a - x) - 4*b*x*(y - x**2)
        dfdy = 2*b*(y - x**2)
        return np.array([dfdx, dfdy])
    
    @staticmethod
    def quadratic_function(params):
        """简单的二次函数，用于基础测试"""
        return np.sum(params ** 2)
    
    @staticmethod
    def quadratic_gradient(params):
        """二次函数的梯度"""
        return 2 * params


def run_comparison():
    """
    运行五大优化器的对比实验
    """
    print("\n" + "=" * 80)
    print("五大优化器综合对比实验")
    print("=" * 80)
    
    # 初始化所有优化器
    sgd = SGD(lr=0.001)
    momentum = Momentum(lr=0.001, momentum=0.9)
    adagrad = AdaGrad(lr=0.5)
    rmsprop = RMSprop(lr=0.01, beta=0.99)
    adam = Adam(lr=0.01, beta1=0.9, beta2=0.999)
    
    optimizers = [sgd, momentum, adagrad, rmsprop, adam]
    
    # 测试1：简单的二次函数
    print("\n" + "-" * 60)
    print("测试1：简单二次函数 f(x,y) = x² + y²")
    print("-" * 60)
    
    start_point = np.array([5.0, 5.0])
    n_iterations = 500
    
    results = {}
    
    for opt in optimizers:
        opt.reset()
        params = start_point.copy()
        losses = []
        
        for _ in range(n_iterations):
            grad = OptimizerBenchmark.quadratic_gradient(params)
            params = opt.step(params, grad)
            losses.append(OptimizerBenchmark.quadratic_function(params))
        
        results[opt.name] = {
            'final_params': params,
            'final_loss': losses[-1],
            'losses': losses
        }
    
    print(f"起点: {start_point}, 初始损失: {OptimizerBenchmark.quadratic_function(start_point):.4f}")
    print(f"\n{n_iterations}步后的结果:")
    print("-" * 60)
    for name, result in results.items():
        print(f"{name:12s}: 最终损失 = {result['final_loss']:.10f}, "
              f"参数 = [{result['final_params'][0]:.6f}, {result['final_params'][1]:.6f}]")
    
    # 测试2：Rosenbrock函数（更有挑战性）
    print("\n" + "-" * 60)
    print("测试2：Rosenbrock函数（香蕉函数）")
    print("      全局最小值在 (1, 1)，这是一个狭长的弯曲山谷")
    print("-" * 60)
    
    # 调整学习率以适应更难的问题
    sgd_rosen = SGD(lr=0.0001)
    momentum_rosen = Momentum(lr=0.0001, momentum=0.9)
    adagrad_rosen = AdaGrad(lr=0.1)
    rmsprop_rosen = RMSprop(lr=0.001, beta=0.99)
    adam_rosen = Adam(lr=0.001, beta1=0.9, beta2=0.999)
    
    optimizers_rosen = [sgd_rosen, momentum_rosen, adagrad_rosen, 
                        rmsprop_rosen, adam_rosen]
    
    start_point_rosen = np.array([-1.0, 2.0])
    n_iterations_rosen = 2000
    
    results_rosen = {}
    
    for opt in optimizers_rosen:
        opt.reset()
        params = start_point_rosen.copy()
        losses = []
        
        for _ in range(n_iterations_rosen):
            grad = OptimizerBenchmark.rosenbrock_gradient(params)
            params = opt.step(params, grad)
            losses.append(OptimizerBenchmark.rosenbrock_function(params))
        
        results_rosen[opt.name] = {
            'final_params': params,
            'final_loss': losses[-1],
            'losses': losses
        }
    
    print(f"起点: {start_point_rosen}, "
          f"初始损失: {OptimizerBenchmark.rosenbrock_function(start_point_rosen):.4f}")
    print(f"\n{n_iterations_rosen}步后的结果:")
    print("-" * 60)
    for name, result in results_rosen.items():
        print(f"{name:12s}: 最终损失 = {result['final_loss']:.6f}")
        print(f"              最终参数 = [{result['final_params'][0]:.6f}, "
              f"{result['final_params'][1]:.6f}]")
    
    print("\n" + "=" * 80)
    print("实验结论:")
    print("  • SGD: 稳定但需要精细调整学习率，收敛较慢")
    print("  • Momentum: 在峡谷中表现好，能加速收敛")
    print("  • AdaGrad: 适合稀疏问题，但长期可能停滞")
    print("  • RMSprop: 在各种问题上表现均衡")
    print("  • Adam: 综合表现最好，收敛快且稳定")
    print("=" * 80 + "\n")
    
    return results, results_rosen


if __name__ == "__main__":
    run_comparison()
```

### 20.7.2 实验结果分析

```
📊 五大优化器性能对比总结

┌─────────────┬────────────┬────────────┬────────────┬────────────┐
│   优化器    │  收敛速度   │  稳定性    │  内存使用   │  超参数敏感度│
├─────────────┼────────────┼────────────┼────────────┼────────────┤
│    SGD      │     ★★☆    │    ★★★     │    ★★★     │    ★☆☆     │
│  Momentum   │     ★★★    │    ★★☆     │    ★★☆     │    ★★☆     │
│  AdaGrad    │     ★★☆    │    ★★★     │    ★★☆     │    ★★★     │
│  RMSprop    │     ★★★    │    ★★★     │    ★★☆     │    ★★☆     │
│    Adam     │     ★★★    │    ★★★     │    ★★☆     │    ★★★     │
└─────────────┴────────────┴────────────┴────────────┴────────────┘

选择建议：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 初学者/快速原型：Adam（lr=0.001，几乎不需要调参）
• 追求最佳泛化性能：SGD + Momentum（需要更多调参）
• 稀疏数据（NLP）：AdaGrad或Adam
• RNN/LSTM：RMSprop或Adam
• 计算机视觉：SGD + Momentum 或 Adam
```

---

## 20.8 数学理论：收敛性分析

### 20.8.1 凸优化收敛性

对于**凸函数**，我们可以给出收敛速度的严格保证。

**定义**：函数 $f$ 是凸函数，如果对于所有 $x, y$ 和 $\alpha \in [0, 1]$：

$$f(\alpha x + (1-\alpha)y) \leq \alpha f(x) + (1-\alpha)f(y)$$

```
📐 凸函数的可视化

    非凸函数                 凸函数
      ╱╲                    
     ╱  ╲                   ╱
    ╱    ╲                 ╱
   ╱      ╲               ╱
  ╱   🕳️   ╲             ╱
 ╱          ╲           ╱
╱            ╲    🎾───╱
                            
  有多个谷底             只有一个谷底
  （局部最优）           （全局最优）
```

### 20.8.2 SGD的收敛性

对于 $L$-光滑的凸函数，SGD的收敛速度为：

$$\mathbb{E}[f(\bar{\theta}_T)] - f(\theta^*) \leq \frac{||\theta_0 - \theta^*||^2}{2\eta T} + \frac{\eta L \sigma^2}{2}$$

其中：
- $\theta^*$ 是最优解
- $\sigma^2$ 是梯度噪声的方差
- $T$ 是迭代次数

**关键洞察**：
- 第一项随 $T$ 减小（收敛到最优）
- 第二项是学习率和噪声的乘积（收敛后的误差）

### 20.8.3 学习率调度

为了让SGD收敛，学习率需要逐渐减小：

$$\eta_t = \frac{\eta_0}{\sqrt{t}} \quad \text{或} \quad \eta_t = \frac{\eta_0}{t}$$

```
📉 学习率衰减策略

学习率
   │
   │\\          \\          
   │ \\          \\         1/√t 衰减
   │  \\          \\        
   │   \\          \\       
   │    \\          \\      
   │     \\          \\     
   │      \\          \\    
   │       \\          \\   
   └──────────────────────► 迭代次数
   
   
学习率
   │
   │\\
   │ \\
   │  \\
   │   \\                  阶梯衰减
   │    \\________
   │              \\
   │               \\______
   │                       \\
   └──────────────────────────► 迭代次数
```

### 20.8.4 Adam的收敛性证明

Adam在凸优化问题上的收敛性由Kingma和Ba在原始论文中证明。核心结果：

$$R(T) = \sum_{t=1}^T [f_t(\theta_t) - f_t(\theta^*)] = O(\sqrt{T})$$

这意味着平均遗憾（average regret）随 $1/\sqrt{T}$ 递减：

$$\frac{R(T)}{T} = O\left(\frac{1}{\sqrt{T}}\right) \to 0 \text{ as } T \to \infty$$

### 20.8.5 非凸优化

深度学习中的损失函数通常是非凸的。在这种情况下，我们关注**收敛到平稳点**（stationary point）：

$$\mathbb{E}[||\nabla f(\theta_T)||^2] \to 0$$

对于非凸问题，Adam和SGD都能在适当条件下保证收敛到局部最优或鞍点。

---

## 20.9 实践指南

### 20.9.1 如何选择优化器？

```
🎯 优化器选择决策树

开始
 │
 ├─ 数据稀疏？（如NLP任务）
 │   ├─ 是 → AdaGrad或Adam
 │   └─ 否 → 继续
 │
 ├─ 需要最快收敛？
 │   ├─ 是 → Adam或RMSprop
 │   └─ 否 → 继续
 │
 ├─ 追求最佳泛化性能？
 │   ├─ 是 → SGD + Momentum（需要调参）
 │   └─ 否 → 继续
 │
 └─ 默认推荐 → Adam (lr=0.001)
```

### 20.9.2 学习率调参技巧

```python
"""
学习率调度器实现
===============
学习率不是固定的，而是随着训练动态调整
"""

class LearningRateScheduler:
    """学习率调度器基类"""
    
    def get_lr(self, epoch: int, initial_lr: float) -> float:
        """获取当前epoch的学习率"""
        raise NotImplementedError


class StepDecay(LearningRateScheduler):
    """
    阶梯衰减：每n个epoch衰减一次
    
    例如：每30个epoch，学习率乘以0.1
    """
    
    def __init__(self, drop_every: int = 30, drop_factor: float = 0.1):
        self.drop_every = drop_every
        self.drop_factor = drop_factor
    
    def get_lr(self, epoch: int, initial_lr: float) -> float:
        """
        计算当前epoch的学习率
        
        公式: lr = initial_lr × (drop_factor)^(epoch // drop_every)
        """
        exp = epoch // self.drop_every
        return initial_lr * (self.drop_factor ** exp)


class ExponentialDecay(LearningRateScheduler):
    """
    指数衰减：学习率按指数函数衰减
    """
    
    def __init__(self, decay_rate: float = 0.96, decay_steps: int = 100):
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
    
    def get_lr(self, epoch: int, initial_lr: float) -> float:
        """
        公式: lr = initial_lr × decay_rate^(epoch / decay_steps)
        """
        return initial_lr * (self.decay_rate ** (epoch / self.decay_steps))


class CosineAnnealing(LearningRateScheduler):
    """
    余弦退火：学习率按余弦函数周期性变化
    
    常用于训练Transformer模型
    """
    
    def __init__(self, T_max: int = 100, eta_min: float = 0):
        self.T_max = T_max  # 周期长度
        self.eta_min = eta_min  # 最小学习率
    
    def get_lr(self, epoch: int, initial_lr: float) -> float:
        """
        公式: lr = eta_min + 0.5×(initial_lr - eta_min)×(1 + cos(π×epoch/T_max))
        """
        import math
        return self.eta_min + 0.5 * (initial_lr - self.eta_min) * \
               (1 + math.cos(math.pi * epoch / self.T_max))


def demonstrate_schedulers():
    """演示不同的学习率调度策略"""
    print("\n" + "=" * 70)
    print("学习率调度策略对比")
    print("=" * 70)
    
    schedulers = {
        '固定': None,
        '阶梯衰减': StepDecay(drop_every=30, drop_factor=0.1),
        '指数衰减': ExponentialDecay(decay_rate=0.95, decay_steps=100),
        '余弦退火': CosineAnnealing(T_max=100, eta_min=0.0001)
    }
    
    initial_lr = 0.1
    epochs = list(range(0, 101, 10))
    
    print(f"\n初始学习率: {initial_lr}")
    print("-" * 60)
    print(f"{'Epoch':<10}", end='')
    for name in schedulers.keys():
        print(f"{name:<15}", end='')
    print()
    print("-" * 60)
    
    for epoch in epochs:
        print(f"{epoch:<10}", end='')
        for name, scheduler in schedulers.items():
            if scheduler is None:
                lr = initial_lr
            else:
                lr = scheduler.get_lr(epoch, initial_lr)
            print(f"{lr:<15.6f}", end='')
        print()
    
    print("\n✓ 余弦退火和阶梯衰减常用于深度学习训练\n")


if __name__ == "__main__":
    demonstrate_schedulers()
```

---

## 20.10 本章总结

### 20.10.1 核心概念回顾

```
📚 五大优化器速查表

┌──────────┬─────────────────┬─────────────────┬────────────────┐
│  优化器   │     核心公式     │    超参数建议    │    适用场景     │
├──────────┼─────────────────┼─────────────────┼────────────────┤
│   SGD    │ θ -= η·g        │ η: 0.01-0.1     │ 大规模数据      │
│          │                 │                 │ 追求泛化性能    │
├──────────┼─────────────────┼─────────────────┼────────────────┤
│ Momentum │ v = γ·v + η·g   │ η: 0.01-0.1     │ 加速SGD        │
│          │ θ -= v          │ γ: 0.9          │ 处理峡谷地形    │
├──────────┼─────────────────┼─────────────────┼────────────────┤
│ AdaGrad  │ G += g²         │ η: 0.01-0.1     │ 稀疏数据        │
│          │ θ -= η·g/√(G+ε) │                 │ 特征频率差异大  │
├──────────┼─────────────────┼─────────────────┼────────────────┤
│ RMSprop  │ G = β·G +       │ η: 0.001-0.01   │ RNN/LSTM       │
│          │     (1-β)·g²    │ β: 0.9-0.99     │ 非平稳目标      │
│          │ θ -= η·g/√(G+ε) │                 │                │
├──────────┼─────────────────┼─────────────────┼────────────────┤
│   Adam   │ m = β₁·m +      │ η: 0.001        │ 默认选择        │
│          │     (1-β₁)·g    │ β₁: 0.9         │ 大多数深度学习  │
│          │ v = β₂·v +      │ β₂: 0.999       │ 任务            │
│          │     (1-β₂)·g²   │                 │                │
│          │ θ -= η·m̂/(√v̂+ε) │                 │                │
└──────────┴─────────────────┴─────────────────┴────────────────┘
```

### 20.10.2 关键洞察

1. **没有最好的优化器，只有最合适的优化器**
   - Adam是默认的好选择
   - 但SGD + Momentum在精心调参后通常有更好的泛化性能

2. **学习率是最重要的超参数**
   - 好的优化器可以减小对学习率的敏感度
   - 但学习率调度仍然能显著提升性能

3. **动量和自适应是两大正交的思想**
   - 动量：加速收敛，减少震荡
   - 自适应：为不同参数定制学习率
   - Adam成功地将两者结合

4. **理论分析指导实践**
   - 收敛性分析告诉我们什么条件下算法会收敛
   - 虽然深度学习是非凸的，但这些理论仍然有价值

### 20.10.3 下一步学习

- **高级优化器**：Lookahead、LAMB、LAMA、Shampoo
- **二阶方法**：L-BFGS、自然梯度
- **分布式优化**：数据并行、模型并行、联邦学习

---

## 练习题

### 基础题（3道）

#### 练习20.1：手动计算一次Adam更新

给定：
- 当前参数：$\theta_t = [2.0, -1.0]$
- 当前梯度：$g_t = [0.5, -0.3]$
- 一阶矩估计：$m_{t-1} = [0.1, -0.05]$
- 二阶矩估计：$v_{t-1} = [0.01, 0.02]$
- 时间步：$t = 10$
- 超参数：$\eta = 0.1$，$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$

请手动计算 $\theta_{t+1}$。

<details>
<summary>点击查看答案</summary>

**步骤1：更新一阶矩**

$$m_t = 0.9 \times [0.1, -0.05] + 0.1 \times [0.5, -0.3]$$
$$= [0.09, -0.045] + [0.05, -0.03]$$
$$= [0.14, -0.075]$$

**步骤2：更新二阶矩**

$$v_t = 0.999 \times [0.01, 0.02] + 0.001 \times [0.25, 0.09]$$
$$= [0.00999, 0.01998] + [0.00025, 0.00009]$$
$$= [0.01024, 0.02007]$$

**步骤3：偏差校正**

$$\hat{m}_t = \frac{[0.14, -0.075]}{1 - 0.9^{10}} = \frac{[0.14, -0.075]}{0.6513} = [0.215, -0.115]$$

$$\hat{v}_t = \frac{[0.01024, 0.02007]}{1 - 0.999^{10}} = \frac{[0.01024, 0.02007]}{0.00996} = [1.028, 2.015]$$

**步骤4：更新参数**

$$\theta_{t+1} = [2.0, -1.0] - 0.1 \times \frac{[0.215, -0.115]}{\sqrt{[1.028, 2.015]} + 10^{-8}}$$

$$= [2.0, -1.0] - 0.1 \times \frac{[0.215, -0.115]}{[1.014, 1.419]}$$

$$= [2.0, -1.0] - 0.1 \times [0.212, -0.081]$$

$$= [2.0, -1.0] - [0.0212, -0.0081]$$

$$= [1.9788, -0.9919]$$

</details>

---

#### 练习20.2：理解动量系数的影响

在Momentum优化器中，动量系数 $\gamma$ 通常设为0.9。如果将其改为0.5或0.99，会发生什么？

请分析：
1. $\gamma = 0.5$ 时的行为
2. $\gamma = 0.99$ 时的行为
3. 为什么0.9是一个常用的选择？

<details>
<summary>点击查看答案</summary>

**1. $\gamma = 0.5$**

- 历史梯度的权重呈指数快速衰减：$0.5^n$
- 10步前的梯度权重：$0.5^{10} \approx 0.001$
- 效果：几乎只看最近几步的梯度
- 行为类似于SGD，加速效果不明显

**2. $\gamma = 0.99$**

- 历史梯度的权重衰减慢：$0.99^n$
- 100步前的梯度仍有约37%的权重
- 效果：累积了很长时间的梯度信息
- 问题：
  - 初期可能朝错误方向冲得过远
  - 改变方向时需要很长时间"刹车"
  - 在最优解附近容易震荡

**3. 为什么0.9是常用选择？**

- 平衡点：考虑约10步的历史（$0.9^{10} \approx 0.35$）
- 既能累积足够的动量来加速
- 又不会因历史负担过重而难以调整方向
- 实践中被验证是有效的

</details>

---

#### 练习20.3：AdaGrad的单调性

证明AdaGrad的有效学习率是单调递减的。

给定：$G_t = G_{t-1} + g_t^2$，有效学习率为 $\eta_t^{eff} = \frac{\eta}{\sqrt{G_t + \epsilon}}$

请证明对于所有 $t > 0$，有 $\eta_t^{eff} \leq \eta_{t-1}^{eff}$。

<details>
<summary>点击查看答案</summary>

**证明：**

1. 由于 $g_t^2 \geq 0$（平方总是非负的）

2. 所以 $G_t = G_{t-1} + g_t^2 \geq G_{t-1}$

3. 因此 $\sqrt{G_t + \epsilon} \geq \sqrt{G_{t-1} + \epsilon}$

4. 倒数后不等式方向改变：
   $$\frac{1}{\sqrt{G_t + \epsilon}} \leq \frac{1}{\sqrt{G_{t-1} + \epsilon}}$$

5. 两边乘以正数 $\eta$：
   $$\frac{\eta}{\sqrt{G_t + \epsilon}} \leq \frac{\eta}{\sqrt{G_{t-1} + \epsilon}}$$

6. 即：$\eta_t^{eff} \leq \eta_{t-1}^{eff}$ **得证** □

</details>

---

### 进阶题（3道）

#### 练习20.4：实现Nesterov加速梯度

Nesterov Accelerated Gradient (NAG) 是Momentum的改进版本。

**任务**：
1. 根据以下算法实现Nesterov优化器
2. 与标准Momentum在Rosenbrock函数上进行对比

**Nesterov算法**：
```
v_t = γ·v_{t-1} + η·∇J(θ_t - γ·v_{t-1})
θ_{t+1} = θ_t - v_t
```

**提示**：先在"前瞻位置"$\theta_t - \gamma v_{t-1}$计算梯度，再用这个梯度更新速度。

<details>
<summary>点击查看参考答案</summary>

```python
class Nesterov:
    """
    Nesterov加速梯度优化器
    """
    
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.name = "Nesterov"
        self.velocity = None
    
    def step(self, params, grad_fn):
        """
        注意：这里需要传入梯度函数，而不是梯度值
        因为Nesterov需要在"前瞻位置"重新计算梯度
        
        参数:
            params: 当前参数
            grad_fn: 梯度函数，接受参数返回梯度
        """
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        # 计算"前瞻位置"
        lookahead_params = params - self.momentum * self.velocity
        
        # 在"前瞻位置"计算梯度
        lookahead_grad = grad_fn(lookahead_params)
        
        # 更新速度（使用"未来"的梯度）
        self.velocity = self.momentum * self.velocity + self.lr * lookahead_grad
        
        # 更新参数
        new_params = params - self.velocity
        
        return new_params
```

Nesterov的理论优势：在某些凸优化问题上可以达到最优的 $O(1/T^2)$ 收敛速度。

</details>

---

#### 练习20.5：自适应学习率的边界分析

对于Adam优化器，分析其有效学习率的范围。

给定：
- 假设梯度 $g_t$ 满足 $|g_t| \leq G$（有界）
- 使用默认参数：$\beta_1 = 0.9$，$\beta_2 = 0.999$

**问题**：
1. 推导有效学习率 $\eta_t^{eff} = \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$ 的理论边界
2. 讨论当 $t$ 很大时，有效学习率的行为

<details>
<summary>点击查看答案</summary>

**分析：**

**对于 $\hat{m}_t$：**

$$m_t = \sum_{i=1}^{t} (1-\beta_1)\beta_1^{t-i} g_i$$

$$|m_t| \leq G \cdot (1-\beta_1) \sum_{i=1}^{t} \beta_1^{t-i} = G \cdot (1-\beta_1^t)$$

$$|\hat{m}_t| = \frac{|m_t|}{1-\beta_1^t} \leq G$$

**对于 $\hat{v}_t$：**

类似地：

$$v_t = \sum_{i=1}^{t} (1-\beta_2)\beta_2^{t-i} g_i^2 \leq G^2 \cdot (1-\beta_2^t)$$

$$\hat{v}_t = \frac{v_t}{1-\beta_2^t} \leq G^2$$

**因此有效学习率：**

$$\eta_t^{eff} = \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

边界：
- 上界：$\frac{\eta \cdot G}{\epsilon}$（当 $\hat{v}_t$ 很小时）
- 下界：趋近于0（当 $\hat{v}_t$ 很大时）

**当 $t \to \infty$：**

- $\hat{m}_t$ 和 $\hat{v}_t$ 变成真正的指数移动平均
- 如果梯度稳定，$\hat{m}_t \approx \mathbb{E}[g]$，$\hat{v}_t \approx \mathbb{E}[g^2]$
- 有效学习率趋于稳定

</details>

---

#### 练习20.6：二阶方法的启示

牛顿法使用Hessian矩阵进行优化：

$$\theta_{t+1} = \theta_t - H^{-1} \nabla J(\theta_t)$$

其中 $H$ 是Hessian矩阵，$H_{ij} = \frac{\partial^2 J}{\partial \theta_i \partial \theta_j}$。

**问题**：
1. 解释为什么牛顿法可以被视为对每个参数"自适应"调整步长
2. AdaGrad、RMSprop、Adam中的二阶矩估计 $v_t$ 与Hessian对角线有什么关系？
3. 讨论一阶自适应方法的优缺点（相对于真正的二阶方法）

<details>
<summary>点击查看答案</summary>

**1. 牛顿法的自适应性质**

牛顿法的更新可以写成：
$$\Delta \theta = -H^{-1} \nabla J$$

对于每个参数 $i$：
$$\Delta \theta_i = -\sum_j (H^{-1})_{ij} \frac{\partial J}{\partial \theta_j}$$

在Hessian对角占优的情况下近似：
$$\Delta \theta_i \approx -\frac{1}{H_{ii}} \frac{\partial J}{\partial \theta_i}$$

这意味着：
- 在曲率大的方向（$H_{ii}$ 大），步长自动减小
- 在曲率小的方向（$H_{ii}$ 小），步长自动增大

这正是自适应学习率的核心思想！

**2. 二阶矩与Hessian的关系**

对于二次损失函数 $J(\theta) = \frac{1}{2}(y - X\theta)^2$：

- Hessian：$H = X^T X$（常数，与当前位置无关）
- 梯度方差：$\mathbb{E}[g^2]$ 与Hessian对角线相关

直观上：
- 梯度平方的期望反映了该方向的不确定性
- 不确定性大的方向，类似于曲率大的方向

因此，$\frac{1}{\sqrt{v_t}}$ 可以看作是对 $\frac{1}{\sqrt{H_{ii}}}$ 的近似。

**3. 一阶自适应 vs 二阶方法**

| 特性 | 一阶自适应 (Adam等) | 二阶方法 (牛顿法) |
|------|-------------------|-----------------|
| 计算复杂度 | $O(d)$ | $O(d^2)$ 或 $O(d^3)$ |
| 内存需求 | $O(d)$ | $O(d^2)$ |
| 自适应精度 | 近似（仅对角） | 精确（全矩阵） |
| 对非凸问题的鲁棒性 | 好 | 差（Hessian可能不正定） |
| 实际应用 | 深度学习的标准选择 | 小规模问题、凸优化 |

结论：一阶自适应方法是计算效率和优化效果之间的优秀折中。

</details>

---

### 挑战题（2道）

#### 练习20.7：实现AdamW（权重衰减正确版本）

原始的Adam使用L2正则化：
```python
grad = grad + lambda * params  # L2正则化
```

但研究发现这与自适应学习率机制不兼容。AdamW将权重衰减与梯度更新分离：

**AdamW算法**：
```
# 参数更新（与Adam相同）
m_t = β₁·m_{t-1} + (1-β₁)·g_t
v_t = β₂·v_{t-1} + (1-β₂)·g_t²
# ...偏差校正...

# 不同的参数更新：
θ_{t+1} = θ_t - η·(m̂_t/(√v̂_t + ε) + λ·θ_t)
```

**任务**：
1. 实现AdamW优化器
2. 解释为什么这种分离式权重衰减优于L2正则化
3. 在简单的线性回归问题上对比Adam和AdamW

<details>
<summary>点击查看参考答案和解释</summary>

```python
class AdamW:
    """
    AdamW优化器（Loshchilov & Hutter, 2017）
    
    将权重衰减与自适应梯度更新分离
    """
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, 
                 eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.name = "AdamW"
        
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, params, grad):
        self.t += 1
        
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        # 一阶和二阶矩估计（不使用L2正则化！）
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        # 偏差校正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # AdamW的核心：权重衰减与自适应更新分离
        # 1. 自适应梯度步
        adaptive_step = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        # 2. 权重衰减步（直接使用params，不按自适应学习率缩放）
        decay_step = self.lr * self.weight_decay * params
        
        # 3. 合并更新
        new_params = params - adaptive_step - decay_step
        
        return new_params
```

**为什么AdamW更好？**

在原始Adam中，L2正则化项也被除以了 $\sqrt{v_t}$：

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon}(g_t + \lambda \theta_t)$$

这意味着：
- 梯度大的参数，L2正则化效果被削弱
- 这与L2正则化的初衷矛盾（所有参数应该被同等惩罚）

AdamW解决了这个问题，使权重衰减真正按预期工作。

</details>

---

#### 练习20.8：优化器的几何解释

考虑二维优化问题，损失函数的等高线是椭圆：

$$f(x, y) = \frac{x^2}{a^2} + \frac{y^2}{b^2}, \quad a > b$$

**问题**：
1. 画出该函数的等高线，标注Hessian矩阵的特征值和特征向量方向
2. 分析SGD、Momentum、AdaGrad在这个问题上的行为差异
3. 推导AdaGrad在这类问题上的最优学习率调度
4. （编程）实现可视化，展示不同优化器在该问题上的轨迹

<details>
<summary>点击查看答案</summary>

**1. 等高线与Hessian**

函数：$f(x,y) = \frac{x^2}{a^2} + \frac{y^2}{b^2}$

梯度：$\nabla f = [\frac{2x}{a^2}, \frac{2y}{b^2}]^T$

Hessian：$H = \begin{bmatrix} \frac{2}{a^2} & 0 \\ 0 & \frac{2}{b^2} \end{bmatrix}$

特征值：$\lambda_1 = \frac{2}{a^2}$（x方向，较平缓），$\lambda_2 = \frac{2}{b^2}$（y方向，较陡峭）

等高线：
```
         │
    ─────┼─────
        /│\
       / │ \        y方向（陡峭）
      │  │  │       等高线密集
    ──┼──┼──┼──
      │  │  │       x方向（平缓）
       \ │ /        等高线稀疏
        \│/
    ─────┼─────
         │
```

**2. 各优化器行为分析**

**SGD**：
- 固定学习率 $\eta$
- 收敛条件：$\eta < \frac{2}{\lambda_{max}} = b^2$
- 问题：在陡峭方向收敛快，在平缓方向收敛慢
- 收敛速度由条件数 $\kappa = \frac{a^2}{b^2}$ 决定

**Momentum**：
- 帮助穿越长而窄的峡谷
- 在平缓方向累积速度，加速收敛
- 最优动量：$\gamma = \frac{(\sqrt{\kappa}-1)^2}{(\sqrt{\kappa}+1)^2}$

**AdaGrad**：
- 自动调整：x方向学习率大（梯度小），y方向学习率小（梯度大）
- 有效预条件：将椭圆变为近似圆形
- 但学习率单调递减是问题

**3. AdaGrad的最优学习率**

对于此问题，$G_t^{(x)} \approx \frac{4t}{a^4}x^2$，$G_t^{(y)} \approx \frac{4t}{b^4}y^2$

有效学习率：
$$\eta_t^{(x),eff} = \frac{\eta a^2}{2\sqrt{t}|x|}, \quad \eta_t^{(y),eff} = \frac{\eta b^2}{2\sqrt{t}|y|}$$

为了保证收敛，需要：
$$\sum_t \eta_t = \infty, \quad \sum_t \eta_t^2 < \infty$$

**4. 可视化代码**

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_optimizer_trajectories():
    a, b = 5, 1  # 椭圆参数
    
    def loss(params):
        x, y = params
        return (x/a)**2 + (y/b)**2
    
    def grad(params):
        x, y = params
        return np.array([2*x/a**2, 2*y/b**2])
    
    # 创建优化器
    optimizers = [
        SGD(lr=0.5),
        Momentum(lr=0.1, momentum=0.9),
        AdaGrad(lr=2.0),
        RMSprop(lr=0.5, beta=0.9),
        Adam(lr=0.5)
    ]
    
    # 绘制等高线
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = (X/a)**2 + (Y/b)**2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, opt in enumerate(optimizers):
        ax = axes[idx]
        ax.contour(X, Y, Z, levels=20, alpha=0.5)
        
        # 运行优化并记录轨迹
        opt.reset()
        params = np.array([5.0, 1.5])
        trajectory = [params.copy()]
        
        for _ in range(100):
            params = opt.step(params, grad(params))
            trajectory.append(params.copy())
        
        trajectory = np.array(trajectory)
        
        # 绘制轨迹
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', markersize=3)
        ax.plot(0, 0, 'g*', markersize=15)  # 最优解
        ax.set_title(opt.name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
    
    # 隐藏多余的子图
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('optimizer_trajectories.png', dpi=150)
    plt.show()

# 运行可视化
plot_optimizer_trajectories()
```

预期结果：
- SGD：在y方向快速震荡，x方向缓慢移动
- Momentum：更平滑的轨迹，更快到达中心
- AdaGrad/RMSprop/Adam：轨迹更接近直线，有效预条件效果

</details>

---

## 参考文献

Bottou, L., Curtis, F. E., & Nocedal, J. (2018). Optimization methods for large-scale machine learning. *SIAM Review*, 60(2), 223-311.

Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. *Journal of Machine Learning Research*, 12, 2121-2159.

Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*.

Nesterov, Y. (1983). A method for solving the convex programming problem with convergence rate $O(1/k^2)$. *Doklady Akademii Nauk SSSR*, 269, 543-547.

Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods. *USSR Computational Mathematics and Mathematical Physics*, 4(5), 1-17.

Reddi, S. J., Kale, S., & Kumar, S. (2018). On the convergence of Adam and beyond. *International Conference on Learning Representations*.

Robbins, H., & Monro, S. (1951). A stochastic approximation method. *The Annals of Mathematical Statistics*, 22(3), 400-407.

Ruder, S. (2016). An overview of gradient descent optimization algorithms. *arXiv preprint arXiv:1609.04747*.

Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). On the importance of initialization and momentum in deep learning. *International Conference on Machine Learning*, 1139-1147.

Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. *COURSERA: Neural Networks for Machine Learning*, 4(2), 26-31.

Ward, R., Wu, X., & Bottou, L. (2020). Adagrad stepsizes: Sharp convergence over nonconvex landscapes. *Journal of Machine Learning Research*, 21(1), 1-36.

---

> *"优化器的演进史，就是人类对'如何更快更好地学习'这一问题不断探索的历史。"
>
> —— 本章完 ——
