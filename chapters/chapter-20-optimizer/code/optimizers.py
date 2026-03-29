# 第二十章：优化器——更快更好的下降

> *"如果你只有一种工具——锤子，你会把所有问题都看成钉子。"* 
> 
> 同样地，如果你只知道一种优化器——梯度下降，你会错过机器学习中最精彩的一部分。*

## 开场故事：寻找迷失的山谷

想象一下，你是一名探险家，站在一座巨大的山脉之巅。你的目标是找到山谷中的最低点——那里据说埋藏着失落的宝藏。但这座山脉有个特点：它被浓雾笼罩，你只能看清脚下几步远的地方。

这就是**优化问题**！山脉是**损失函数**，你所在的海拔高度是**损失值**，而你的目标是找到**全局最小值**（最低点）。

在之前的章节中，我们学习了最基本的下山方法：**梯度下降**（Gradient Descent）。就像闭着眼睛，每次都朝着最陡的下坡方向迈出一步。但这种方法太慢了，而且很容易被困在局部坑洼中。

本章将带你认识一群更聪明的"下山专家"——**优化器**（Optimizers）。它们各有绝招：有的像滑雪高手，借助惯性加速冲刺；有的像经验丰富的向导，会根据地形自动调整步伐大小；还有的像精通地形的测绘师，能记住每一条路线的陡峭程度。

让我们一起踏上这段寻找最优解的奇妙旅程！

---

## 20.1 为什么需要更好的优化器？

### 20.1.1 梯度下降的困境

让我们先回顾一下最基础的**随机梯度下降**（SGD）的更新规则：

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla L(\theta_t)$$

其中：
- $\theta_t$ 是第 $t$ 步的参数
- $\eta$ 是学习率（步长）
- $\nabla L(\theta_t)$ 是当前位置的梯度

**生活中的比喻**：这就像一个固执的登山者，每次只根据脚下最陡的方向走固定长度的一步。

但这个方法有三个致命问题：

**问题一：之字形震荡（Zigzag Problem）**

想象你在一个狭长的山谷中行走，两侧的山坡很陡峭，但山谷底部本身很平缓。SGD会在山谷两侧来回 bouncing，像乒乓球一样，进展缓慢。

**问题二：学习率 dilemma**

- 学习率太大 → 在山谷两侧跳来跳去，甚至 diverge（发散）
- 学习率太小 → 在平缓区域移动慢如蜗牛

**问题三：局部最优陷阱**

SGD容易被困在局部最小值或鞍点，无法继续探索更好的解。

### 20.1.2 优化器进化史

```
1964年 → Polyak 提出 Momentum（动量法）
    ↓
2011年 → Duchi 等人提出 AdaGrad
    ↓
2012年 → Hinton 等人提出 RMSprop
    ↓
2014年 → Kingma & Ba 提出 Adam（集大成者）
    ↓
2018年 → 优化理论全面综述（Bottou 等人）
```

每个新优化器都是为了解决前一个的问题而诞生的。让我们逐一认识它们！

---

## 20.2 动量法（Momentum）：滚下山的雪球

### 20.2.1 物理直觉

想象一个雪球从山顶滚下来：

- 一开始雪球很小，速度很慢
- 随着滚下山坡，雪球越滚越快（积累动量）
- 即使遇到小坑洼，雪球也能凭借惯性冲过去
- 在平坦区域，雪球依靠惯性继续前进

这就是**动量法**的核心思想！SGD就像一个人一步一步地走，而动量法就像一个滚动的雪球。

### 20.2.2 数学原理

动量法引入了一个**速度变量** $v_t$，它记录了过去的梯度信息：

**初始化**：
$$v_0 = 0$$

**速度更新**（积累动量）：
$$v_{t+1} = \gamma \cdot v_t + \eta \cdot \nabla L(\theta_t)$$

**参数更新**（按照速度移动）：
$$\theta_{t+1} = \theta_t - v_{t+1}$$

其中：
- $\gamma$ 是**动量系数**（通常设为 0.9）
- $\gamma \cdot v_t$ 是**惯性项**（保留上一时刻的速度）
- $\eta \cdot \nabla L(\theta_t)$ 是**梯度贡献**

**关键洞察**：动量法实际上是对梯度做了**指数加权移动平均**（Exponentially Weighted Moving Average）！

展开 $v_t$：
$$v_t = \eta \sum_{i=0}^{t} \gamma^{t-i} \nabla L(\theta_i)$$

这意味着：
- 最近的梯度权重最高（$\gamma^0 = 1$）
- 过去的梯度贡献按指数衰减
- 当 $\gamma = 0.9$ 时，大约 10 步前的梯度贡献已经很小了

### 20.2.3 为什么动量法有效？

**场景一：之字形山谷**

假设你在一个南北方向陡峭、东西方向平缓的山谷中：

| 时间 | 南北梯度 | 东西梯度 | SGD 移动 | Momentum 移动 |
|------|----------|----------|----------|---------------|
| t=1  | +5       | +1       | 向北5步，向东1步 | 向北5步，向东1步 |
| t=2  | -4       | +1       | 向南4步，向东1步 | 向北1步，向东2步 |
| t=3  | +4       | +1       | 向北4步，向东1步 | 向北3步，向东3步 |

**观察**：南北方向的梯度来回震荡，相互抵消；东西方向的梯度一致，被累积放大。这就是为什么动量法能减少之字形震荡！

**场景二：逃离局部最优**

想象一个平坦的 plateau，上面有几个小坑洼（局部最小值）。

- SGD：很容易被困在小坑洼里
- Momentum：像有惯性的雪球，能凭借之前的速度冲出小坑洼

**场景三：鞍点逃脱**

在高维空间中，鞍点（saddle points）比局部最小值更常见。在鞍点处，某些方向的梯度为正，某些为负。

- SGD：在鞍点附近的平缓区域移动缓慢
- Momentum：凭借积累的速度，能快速穿过鞍点

### 20.2.4 Nesterov 加速梯度（NAG）

**思想**：与其到了位置再看梯度，不如**预判**一下未来位置！

想象你正在滑雪：
- 普通动量法：先滑到当前位置，再看前方坡度
- Nesterov：根据当前速度预判未来位置，**在那个预判位置看坡度**

**数学形式**：

先计算"预判位置"：
$$\tilde{\theta}_t = \theta_t - \gamma \cdot v_t$$

然后在预判位置计算梯度：
$$v_{t+1} = \gamma \cdot v_t + \eta \cdot \nabla L(\tilde{\theta}_t)$$

$$\theta_{t+1} = \theta_t - v_{t+1}$$

**为什么更好？**

Nesterov 的预判机制让它能**提前知道速度会把自己带向哪里**，并相应调整。这就像开车时看前方的路况，而不是只看眼前的路面。

---

## 20.3 AdaGrad：自适应学习率的智者

### 20.3.1 新的问题

动量法解决了"震荡"和"逃逸"问题，但还有一个问题没解决：

**不同参数需要不同的学习率！**

想象你在一个峡谷中：
- 峡谷很窄很陡（某个参数的梯度很大）→ 应该迈小步，否则会在两侧 bouncing
- 峡谷很长很平缓（某个参数的梯度很小）→ 应该迈大步，快速前进

SGD 和动量法对所有参数使用**相同的学习率**，这显然不合理。

### 20.3.2 核心思想：根据历史梯度调整学习率

**AdaGrad**（Adaptive Gradient Algorithm）的核心洞察：

> 如果一个参数的历史梯度一直很大，说明它变化剧烈，我们应该降低它的学习率；
> 如果一个参数的历史梯度一直很小，说明它变化平缓，我们应该提高它的学习率。

**生活中的比喻**：

想象你是一名园丁，在花园里种花：
- 某些花（参数）生长很快（梯度大），你需要经常浇水但每次都少浇一点（小学习率）
- 某些花（参数）生长很慢（梯度小），你需要少浇水但每次多浇一点（大学习率）

### 20.3.3 数学原理

AdaGrad 维护一个**累积梯度平方**的变量 $G_t$：

**初始化**：
$$G_0 = 0$$

**累积平方梯度**：
$$G_{t+1} = G_t + (\nabla L(\theta_t))^2$$

**自适应更新**：
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_{t+1} + \epsilon}} \cdot \nabla L(\theta_t)$$

其中：
- $\epsilon$ 是一个很小的常数（如 $10^{-8}$），防止除以零
- 这里的平方和开根号都是**逐元素**（element-wise）操作

**逐元素解释**：

对于第 $i$ 个参数 $\theta^{(i)}$：

$$G_{t+1}^{(i)} = \sum_{\tau=0}^{t} (\nabla L(\theta_\tau)^{(i)})^2$$

$$\theta_{t+1}^{(i)} = \theta_t^{(i)} - \frac{\eta}{\sqrt{G_{t+1}^{(i)} + \epsilon}} \cdot \nabla L(\theta_t)^{(i)}$$$$

这意味着每个参数都有自己的**有效学习率**：

$$\eta_t^{(i)} = \frac{\eta}{\sqrt{\sum_{\tau=0}^{t} (\nabla L(\theta_\tau)^{(i)})^2 + \epsilon}}$$

### 20.3.4 为什么 AdaGrad 有效？

**案例一：稀疏梯度**

在 NLP 任务中，词嵌入（word embeddings）的梯度是稀疏的——大多数词的梯度为零，只有出现过的词才有梯度。

- 频繁出现的词（如"the"）累积了很大的 $G$，学习率自动变小
- 罕见词累积了很小的 $G$，学习率自动变大

这正是 AdaGrad 的强项！

**案例二：不同尺度的参数**

假设两个参数 $\theta_1$ 和 $\theta_2$：
- $\theta_1$ 的梯度范围是 $[0.01, 0.1]$（小梯度）
- $\theta_2$ 的梯度范围是 $[1, 10]$（大梯度）

AdaGrad 会自动：
- 给 $\theta_1$ 较大的学习率
- 给 $\theta_2$ 较小的学习率

### 20.3.5 AdaGrad 的致命缺陷

**问题：学习率单调递减，最终趋于零！**

因为 $G_t$ 是单调不减的（不断累加正数），所以：

$$\lim_{t \to \infty} \frac{\eta}{\sqrt{G_t + \epsilon}} = 0$$

这就像一辆汽车，油门越踩越轻，最后完全停了下来——即使还没到目的地！

**什么时候这是个问题？**

- 深度神经网络训练后期
- 需要精细调整参数的微调阶段
- 非凸优化问题（损失地形复杂）

这就引出了我们的下一个优化器...

---

## 20.4 RMSprop：遗忘过去的聪明人

### 20.4.1 解决 AdaGrad 的缺陷

**核心问题**：AdaGrad 永远记住所有历史梯度，导致学习率越来越激进地减小。

**RMSprop** 的解决方案：**只记住最近的梯度，忘记久远的过去！**

**生活中的比喻**：

想象你在驾驶一艘船：
- AdaGrad：船上的航海日志记录了所有历史风浪，船长因此越来越谨慎，最后船几乎不动
- RMSprop：船长只看最近几天的天气，根据最新情况调整航行策略

### 20.4.2 数学原理：指数移动平均

RMSprop 用**指数移动平均**（Exponential Moving Average, EMA）代替累积和：

**初始化**：
$$E[g^2]_0 = 0$$

**EMA 更新**：
$$E[g^2]_{t+1} = \beta \cdot E[g^2]_t + (1 - \beta) \cdot (\nabla L(\theta_t))^2$$

**参数更新**：
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_{t+1} + \epsilon}} \cdot \nabla L(\theta_t)$$

其中：
- $\beta$ 通常设为 0.9（控制记忆长度）
- $(1 - \beta)$ 是新梯度的权重

**对比 AdaGrad**：

| 特性 | AdaGrad | RMSprop |
|------|---------|---------|
| 记忆方式 | 累积所有历史 | 指数衰减记忆 |
| 学习率趋势 | 单调递减 | 可以自适应调整 |
| 适用场景 | 稀疏梯度、凸优化 | 非凸优化、深度学习 |

### 20.4.3 为什么指数移动平均更好？

展开 EMA：

$$E[g^2]_t = (1 - \beta) \sum_{\tau=0}^{t} \beta^{t-\tau} (\nabla L(\theta_\tau))^2$$

当 $\beta = 0.9$ 时：
- 最近 10 步的梯度贡献了约 65% 的权重
- 10 步前的梯度贡献已经很小了

这意味着 RMSprop 会**自动适应最近的梯度变化**，而不会被困在过去的"阴影"中。

**场景演示**：

假设训练过程中损失地形发生了变化（如学习率衰减导致进入新的优化阶段）：
- AdaGrad：还记着很久以前的陡峭梯度，学习率依然很小
- RMSprop：只关注最近的梯度，能快速调整学习率

### 20.4.4 与动量法的对比

有趣的是，RMSprop 和动量法都使用了指数移动平均，但用途不同：

| 方法 | EMA 的对象 | 用途 |
|------|-----------|------|
| Momentum | 梯度本身 | 积累速度，减少震荡 |
| RMSprop | 梯度平方 | 调整学习率，适应不同参数 |

这就像汽车的两个系统：
- 动量法 = 惯性系统（让车保持运动趋势）
- RMSprop = 自动变速器（根据路况调整档位）

---

## 20.5 Adam：集大成者的智慧

### 20.5.1 为什么叫 Adam？

**Adam** = **A**daptive **M**oment **E**stimation（自适应矩估计）

它同时结合了：
1. **Momentum**（一阶矩估计：梯度的 EMA）
2. **RMSprop**（二阶矩估计：梯度平方的 EMA）

就像一辆配备了**惯性系统**和**自动变速器**的超级赛车！

### 20.5.2 数学原理

Adam 维护两个变量：
- $m_t$：一阶矩（梯度的 EMA）→ 类似 Momentum
- $v_t$：二阶矩（梯度平方的 EMA）→ 类似 RMSprop

**初始化**：
$$m_0 = 0, \quad v_0 = 0$$

**一阶矩估计（Momentum 部分）**：
$$m_{t+1} = \beta_1 \cdot m_t + (1 - \beta_1) \cdot \nabla L(\theta_t)$$

**二阶矩估计（RMSprop 部分）**：
$$v_{t+1} = \beta_2 \cdot v_t + (1 - \beta_2) \cdot (\nabla L(\theta_t))^2$$

**偏差修正**（Bias Correction）：

由于初始化为零，前几步的估计会有偏差。Adam 通过除以 $(1 - \beta^t)$ 来修正：

$$\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}$$

$$\hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}$$

**参数更新**：

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \cdot \hat{m}_{t+1}$$

**默认超参数**：
- $\beta_1 = 0.9$（控制一阶矩衰减）
- $\beta_2 = 0.999$（控制二阶矩衰减）
- $\epsilon = 10^{-8}$
- $\eta = 0.001$（学习率）

### 20.5.3 为什么需要偏差修正？

让我们直观理解：

假设 $\beta_1 = 0.9$，第一步：
$$m_1 = 0.9 \cdot 0 + 0.1 \cdot g_1 = 0.1 \cdot g_1$$

真实的梯度是 $g_1$，但我们只估计了 $0.1 \cdot g_1$，**低估了 10 倍**！

修正后：
$$\hat{m}_1 = \frac{0.1 \cdot g_1}{1 - 0.9^1} = \frac{0.1 \cdot g_1}{0.1} = g_1$$

完美！

随着 $t$ 增大，$\beta_1^t \to 0$，修正因子趋近于 1，说明偏差逐渐消失。

### 20.5.4 Adam 的更新规则解读

$$\theta_{t+1} = \theta_t - \underbrace{\frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon}}_{\text{自适应学习率}} \cdot \underbrace{\hat{m}_{t+1}}_{\text{动量方向}}$$

这可以分解为：

1. **方向**：$\hat{m}_{t+1}$ 提供了平滑的梯度方向（Momentum 的作用）
2. **步长**：$\frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon}$ 为每个参数提供了自适应的学习率（RMSprop 的作用）

**为什么 Adam 这么强大？**

因为它同时解决了：
- ✅ 之字形震荡（Momentum 的平滑作用）
- ✅ 不同参数需要不同学习率（自适应学习率）
- ✅ 鞍点问题（Momentum 的惯性帮助逃离）
- ✅ 学习率单调递减（EMA 的遗忘机制）

### 20.5.5 Adam 的变体

**AdamW**：将权重衰减（Weight Decay）从梯度中解耦出来，效果更好：

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}} + \epsilon} + \lambda \theta_t \right)$$

**AMSGrad**：解决 Adam 可能不收敛的问题，使用最大值而非 EMA：

$$v_{t+1}^{\max} = \max(v_t^{\max}, v_{t+1})$$

---

## 20.6 优化器选择指南

### 20.6.1 快速决策树

```
开始
  │
  ├── 数据稀疏吗？（如 NLP 的词嵌入）
  │     ├── 是 → 考虑 AdaGrad
  │     └── 否 → 继续
  │
  ├── 问题凸吗？
  │     ├── 是 → SGD + Momentum 或 AdaGrad
  │     └── 否 → 继续
  │
  ├── 需要快速收敛吗？
  │     ├── 是 → Adam / AdamW（默认选择）
  │     └── 否 → RMSprop 或 SGD + Momentum
  │
  └── 追求最终精度？（如 CV 竞赛）
        ├── 是 → 先用 Adam 预训练，再用 SGD 微调
        └── 否 → Adam 即可
```

### 20.6.2 实际建议

| 场景 | 推荐优化器 | 学习率起点 |
|------|-----------|-----------|
| 通用深度学习 | Adam | 3e-4 |
| 计算机视觉 | SGD + Momentum / AdamW | 1e-3 / 1e-2 |
| 自然语言处理 | AdamW | 5e-5 (BERT) / 1e-4 |
| 强化学习 | RMSprop / Adam | 3e-4 |
| 生成模型 | Adam | 1e-4 |
| 大规模训练 | LAMB / LARS | 自适应 |

---

## 20.7 完整代码实现

现在让我们亲手实现这些优化器！下面的实现不依赖 sklearn 或 PyTorch，完全从零开始。

### 20.7.1 基础设置和测试函数

```python
"""
第二十章：优化器实现
从零实现各种优化器，包括 SGD、Momentum、AdaGrad、RMSprop 和 Adam
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 测试函数（损失地形） ====================

class LossLandscape:
    """各种用于测试优化器的损失函数地形"""
    
    @staticmethod
    def quadratic(x, y, a=1, b=10):
        """
        二次函数: f(x,y) = a*x^2 + b*y^2
        用于测试基本的收敛行为
        当 a ≠ b 时，会形成一个狭长的山谷（之字形问题的理想测试）
        """
        return a * x**2 + b * y**2
    
    @staticmethod
    def quadratic_grad(x, y, a=1, b=10):
        """二次函数的梯度"""
        dx = 2 * a * x
        dy = 2 * b * y
        return np.array([dx, dy])
    
    @staticmethod
    def rosenbrock(x, y, a=1, b=100):
        """
        Rosenbrock函数（香蕉函数）: f(x,y) = (a-x)^2 + b*(y-x^2)^2
        这是一个经典的非凸优化测试函数
        全局最小值在 (a, a^2)，位于一个狭长的抛物线形山谷中
        """
        return (a - x)**2 + b * (y - x**2)**2
    
    @staticmethod
    def rosenbrock_grad(x, y, a=1, b=100):
        """Rosenbrock函数的梯度"""
        dx = -2 * (a - x) - 4 * b * x * (y - x**2)
        dy = 2 * b * (y - x**2)
        return np.array([dx, dy])
    
    @staticmethod
    def beale(x, y):
        """
        Beale函数: 多峰函数，有多个局部最小值
        全局最小值在 (3, 0.5)
        """
        term1 = (1.5 - x + x*y)**2
        term2 = (2.25 - x + x*y**2)**2
        term3 = (2.625 - x + x*y**3)**2
        return term1 + term2 + term3
    
    @staticmethod
    def beale_grad(x, y):
        """Beale函数的梯度（数值近似）"""
        eps = 1e-7
        dx = (LossLandscape.beale(x + eps, y) - 
              LossLandscape.beale(x - eps, y)) / (2 * eps)
        dy = (LossLandscape.beale(x, y + eps) - 
              LossLandscape.beale(x, y - eps)) / (2 * eps)
        return np.array([dx, dy])
    
    @staticmethod
    def rastrigin(x, y, A=10):
        """
        Rastrigin函数: 有很多局部最小值
        用于测试全局优化能力
        """
        return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + \
               (y**2 - A * np.cos(2 * np.pi * y))
    
    @staticmethod
    def rastrigin_grad(x, y, A=10):
        """Rastrigin函数的梯度"""
        dx = 2*x + 2*np.pi*A*np.sin(2*np.pi*x)
        dy = 2*y + 2*np.pi*A*np.sin(2*np.pi*y)
        return np.array([dx, dy])


# ==================== 基础优化器类 ====================

class Optimizer:
    """所有优化器的基类"""
    
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.history = []  # 记录优化轨迹
        self.loss_history = []  # 记录损失值
        
    def step(self, params, grad_fn):
        """执行一步优化，需要子类实现"""
        raise NotImplementedError
        
    def optimize(self, init_params, grad_fn, loss_fn, n_steps=100):
        """
        执行完整优化过程
        
        Parameters:
        -----------
        init_params : np.array
            初始参数值 [x, y]
        grad_fn : callable
            梯度函数，输入 (x, y) 返回 [dx, dy]
        loss_fn : callable
            损失函数，输入 (x, y) 返回标量
        n_steps : int
            优化步数
            
        Returns:
        --------
        params : np.array
            最终参数
        history : list
            优化轨迹
        loss_history : list
            损失值历史
        """
        params = np.array(init_params, dtype=float)
        self.history = [params.copy()]
        self.loss_history = [loss_fn(params[0], params[1])]
        
        for _ in range(n_steps):
            params = self.step(params, grad_fn)
            self.history.append(params.copy())
            self.loss_history.append(loss_fn(params[0], params[1]))
            
        return params, self.history, self.loss_history
    
    def reset(self):
        """重置优化器状态"""
        self.history = []
        self.loss_history = []


# ==================== SGD 优化器 ====================

class SGD(Optimizer):
    """
    随机梯度下降（Stochastic Gradient Descent）
    
    更新规则: θ = θ - lr * gradient
    """
    
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
        self.name = "SGD"
        
    def step(self, params, grad_fn):
        grad = grad_fn(params[0], params[1])
        params = params - self.lr * grad
        return params


class SGDMomentum(Optimizer):
    """
    带动量的 SGD（SGD with Momentum）
    
    物理直觉: 像雪球滚下山，积累速度
    
    更新规则:
        v = gamma * v + lr * gradient
        θ = θ - v
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.gamma = momentum
        self.velocity = None
        self.name = f"SGD+Momentum(γ={momentum})"
        
    def reset(self):
        super().reset()
        self.velocity = None
        
    def step(self, params, grad_fn):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
            
        grad = grad_fn(params[0], params[1])
        
        # 更新速度: v = gamma * v + lr * grad
        self.velocity = self.gamma * self.velocity + self.lr * grad
        
        # 更新参数: θ = θ - v
        params = params - self.velocity
        
        return params


class SGDWithDecay(Optimizer):
    """
    带学习率衰减的 SGD
    
    学习率随时间递减: lr_t = lr_0 / (1 + decay * t)
    或指数衰减: lr_t = lr_0 * gamma^t
    """
    
    def __init__(self, learning_rate=0.01, decay=0.01, decay_type='inverse'):
        super().__init__(learning_rate)
        self.initial_lr = learning_rate
        self.decay = decay
        self.decay_type = decay_type
        self.t = 0
        self.name = f"SGD+Decay({decay_type})"
        
    def reset(self):
        super().reset()
        self.t = 0
        self.lr = self.initial_lr
        
    def get_lr(self):
        """计算当前学习率"""
        if self.decay_type == 'inverse':
            # 逆时间衰减: lr / (1 + decay * t)
            return self.initial_lr / (1 + self.decay * self.t)
        elif self.decay_type == 'exponential':
            # 指数衰减: lr * gamma^t
            return self.initial_lr * (self.decay ** self.t)
        elif self.decay_type == 'step':
            # 阶梯衰减: 每 N 步衰减一次
            drop = np.floor(self.t / 10)
            return self.initial_lr * (self.decay ** drop)
        return self.initial_lr
        
    def step(self, params, grad_fn):
        self.lr = self.get_lr()
        grad = grad_fn(params[0], params[1])
        params = params - self.lr * grad
        self.t += 1
        return params


# ==================== AdaGrad 优化器 ====================

class AdaGrad(Optimizer):
    """
    AdaGrad: Adaptive Gradient Algorithm
    
    核心思想: 根据历史梯度的累积调整学习率
    稀疏特征获得更大学习率，频繁特征获得更小学习率
    
    更新规则:
        G = G + gradient^2
        θ = θ - lr * gradient / sqrt(G + epsilon)
    
    优点: 适合稀疏梯度（如 NLP 中的词嵌入）
    缺点: 学习率单调递减，最终趋于零
    """
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.G = None  # 累积梯度平方
        self.name = "AdaGrad"
        
    def reset(self):
        super().reset()
        self.G = None
        
    def step(self, params, grad_fn):
        if self.G is None:
            self.G = np.zeros_like(params)
            
        grad = grad_fn(params[0], params[1])
        
        # 累积梯度平方
        self.G = self.G + grad ** 2
        
        # 自适应学习率更新
        adapted_lr = self.lr / (np.sqrt(self.G) + self.epsilon)
        params = params - adapted_lr * grad
        
        return params


# ==================== RMSprop 优化器 ====================

class RMSprop(Optimizer):
    """
    RMSprop: Root Mean Square Propagation
    
    解决 AdaGrad 学习率单调递减的问题
    使用指数移动平均（EMA）代替累积和
    
    更新规则:
        E[g^2] = beta * E[g^2] + (1-beta) * gradient^2
        θ = θ - lr * gradient / sqrt(E[g^2] + epsilon)
    
    优点: 适合非凸优化，深度学习中最常用之一
    """
    
    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.E_g2 = None  # 梯度平方的 EMA
        self.name = f"RMSprop(β={beta})"
        
    def reset(self):
        super().reset()
        self.E_g2 = None
        
    def step(self, params, grad_fn):
        if self.E_g2 is None:
            self.E_g2 = np.zeros_like(params)
            
        grad = grad_fn(params[0], params[1])
        
        # EMA 更新: E[g^2] = beta * E[g^2] + (1-beta) * g^2
        self.E_g2 = self.beta * self.E_g2 + (1 - self.beta) * (grad ** 2)
        
        # 自适应学习率更新
        adapted_lr = self.lr / (np.sqrt(self.E_g2) + self.epsilon)
        params = params - adapted_lr * grad
        
        return params


# ==================== Adam 优化器 ====================

class Adam(Optimizer):
    """
    Adam: Adaptive Moment Estimation
    
    集大成者: 结合了 Momentum 和 RMSprop 的优点
    - 一阶矩 m: 梯度的 EMA（类似 Momentum）
    - 二阶矩 v: 梯度平方的 EMA（类似 RMSprop）
    
    更新规则:
        m = beta1 * m + (1-beta1) * gradient
        v = beta2 * v + (1-beta2) * gradient^2
        m_hat = m / (1 - beta1^t)  # 偏差修正
        v_hat = v / (1 - beta2^t)  # 偏差修正
        θ = θ - lr * m_hat / (sqrt(v_hat) + epsilon)
    
    默认超参数:
        beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, lr = 0.001
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # 一阶矩
        self.v = None  # 二阶矩
        self.t = 0     # 时间步
        self.name = f"Adam(β1={beta1}, β2={beta2})"
        
    def reset(self):
        super().reset()
        self.m = None
        self.v = None
        self.t = 0
        
    def step(self, params, grad_fn):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            
        grad = grad_fn(params[0], params[1])
        self.t += 1
        
        # 一阶矩估计（Momentum）
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        
        # 二阶矩估计（RMSprop）
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        
        # 偏差修正
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # 参数更新
        params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params


# ==================== Nesterov 加速梯度 ====================

class Nesterov(Optimizer):
    """
    Nesterov Accelerated Gradient (NAG)
    
    改进的动量法: 在当前位置"预判"未来，在那个预判位置计算梯度
    
    更新规则:
        θ_lookahead = θ - gamma * v
        v = gamma * v + lr * grad(θ_lookahead)
        θ = θ - v
    
    相比普通 Momentum，Nesterov 更加"前瞻"，收敛更快
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.gamma = momentum
        self.velocity = None
        self.name = f"Nesterov(γ={momentum})"
        
    def reset(self):
        super().reset()
        self.velocity = None
        
    def step(self, params, grad_fn):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
            
        # 计算预判位置
        lookahead_params = params - self.gamma * self.velocity
        
        # 在预判位置计算梯度
        grad = grad_fn(lookahead_params[0], lookahead_params[1])
        
        # 更新速度
        self.velocity = self.gamma * self.velocity + self.lr * grad
        
        # 更新参数
        params = params - self.velocity
        
        return params


# ==================== 优化器对比可视化 ====================

class OptimizerVisualizer:
    """优化器可视化工具类"""
    
    def __init__(self, loss_landscape=None):
        self.landscape = loss_landscape or LossLandscape()
        
    def create_contour_plot(self, loss_fn, x_range=(-2, 2), y_range=(-2, 2), 
                           n_points=100, title="Loss Landscape"):
        """创建损失函数的等高线图"""
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)
        
        # 计算损失值
        Z = np.zeros_like(X)
        for i in range(n_points):
            for j in range(n_points):
                Z[i, j] = loss_fn(X[i, j], Y[i, j])
                
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制等高线
        levels = np.logspace(0, np.log10(Z.max() + 1), 20)
        contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)
        
        # 添加颜色填充
        ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        return fig, ax, X, Y, Z
    
    def compare_optimizers(self, optimizers, loss_fn, grad_fn, init_params,
                          n_steps=100, x_range=(-2, 2), y_range=(-2, 2),
                          title="Optimizer Comparison"):
        """
        对比多个优化器的性能
        
        Parameters:
        -----------
        optimizers : list
            优化器实例列表
        loss_fn, grad_fn : callable
            损失函数和梯度函数
        init_params : tuple
            初始位置 (x, y)
        n_steps : int
            优化步数
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图: 优化轨迹
        ax1 = axes[0]
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[loss_fn(xi, yi) for xi in x] for yi in y])
        
        levels = np.logspace(0, np.log10(Z.max() + 1), 15)
        ax1.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.4)
        ax1.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.2)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(optimizers)))
        
        results = {}
        
        for opt, color in zip(optimizers, colors):
            opt.reset()
            final_params, history, loss_history = opt.optimize(
                init_params, grad_fn, loss_fn, n_steps
            )
            
            history = np.array(history)
            
            # 绘制轨迹
            ax1.plot(history[:, 0], history[:, 1], 
                    color=color, linewidth=2, label=opt.name, alpha=0.8)
            ax1.scatter(history[0, 0], history[0, 1], 
                       color=color, s=100, marker='o', zorder=5)
            ax1.scatter(history[-1, 0], history[-1, 1], 
                       color=color, s=100, marker='*', zorder=5)
            
            results[opt.name] = {
                'history': history,
                'loss_history': loss_history,
                'final_loss': loss_history[-1],
                'color': color
            }
        
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        ax1.set_title('Optimization Trajectories', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 右图: 损失曲线
        ax2 = axes[1]
        for name, data in results.items():
            ax2.semilogy(data['loss_history'], 
                        color=data['color'], linewidth=2, label=name)
        
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Loss (log scale)', fontsize=12)
        ax2.set_title('Convergence Speed', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig, results


# ==================== 演示代码 ====================

def demo_basic_optimizers():
    """演示基础优化器的使用"""
    print("=" * 60)
    print("演示: 基础优化器对比")
    print("=" * 60)
    
    # 创建优化器
    optimizers = [
        SGD(learning_rate=0.02),
        SGDMomentum(learning_rate=0.02, momentum=0.9),
        AdaGrad(learning_rate=0.5),
        RMSprop(learning_rate=0.02),
        Adam(learning_rate=0.02),
    ]
    
    # 测试函数: 狭长的二次函数（之字形问题）
    init_params = [1.5, 1.5]
    loss_fn = lambda x, y: LossLandscape.quadratic(x, y, a=1, b=20)
    grad_fn = lambda x, y: LossLandscape.quadratic_grad(x, y, a=1, b=20)
    
    print(f"\n初始位置: {init_params}")
    print(f"目标函数: f(x,y) = x² + 20y² (之字形山谷)")
    print(f"优化步数: 50\n")
    
    visualizer = OptimizerVisualizer()
    fig, results = visualizer.compare_optimizers(
        optimizers, loss_fn, grad_fn, init_params, 
        n_steps=50, x_range=(-2, 2), y_range=(-2, 2),
        title="Optimizer Comparison on Quadratic Function"
    )
    
    # 打印最终结果
    print("\n优化结果:")
    print("-" * 60)
    for name, data in results.items():
        final_pos = data['history'][-1]
        print(f"{name:25s} | 最终位置: ({final_pos[0]:8.6f}, {final_pos[1]:8.6f}) | 最终损失: {data['final_loss']:.8f}")
    
    plt.savefig('/root/.openclaw/workspace/ml-book-for-kids/chapter20_optimizer_comparison.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存到 chapter20_optimizer_comparison.png")
    
    return fig, results


def demo_rosenbrock():
    """在 Rosenbrock 函数上测试优化器"""
    print("\n" + "=" * 60)
    print("演示: Rosenbrock 函数（香蕉函数）")
    print("=" * 60)
    
    optimizers = [
        SGD(learning_rate=0.001),
        SGDMomentum(learning_rate=0.001, momentum=0.9),
        RMSprop(learning_rate=0.01),
        Adam(learning_rate=0.01),
        Nesterov(learning_rate=0.001, momentum=0.9),
    ]
    
    # Rosenbrock 函数
    init_params = [-1.0, 1.0]
    loss_fn = LossLandscape.rosenbrock
    grad_fn = LossLandscape.rosenbrock_grad
    
    print(f"\n初始位置: {init_params}")
    print(f"全局最小值: (1, 1)")
    print(f"优化步数: 2000\n")
    
    visualizer = OptimizerVisualizer()
    fig, results = visualizer.compare_optimizers(
        optimizers, loss_fn, grad_fn, init_params,
        n_steps=2000, x_range=(-2, 2), y_range=(-1, 3),
        title="Optimizer Comparison on Rosenbrock Function"
    )
    
    print("\n优化结果:")
    print("-" * 60)
    for name, data in results.items():
        final_pos = data['history'][-1]
        distance_to_opt = np.sqrt((final_pos[0]-1)**2 + (final_pos[1]-1)**2)
        print(f"{name:25s} | 最终位置: ({final_pos[0]:8.4f}, {final_pos[1]:8.4f}) | 距离最优: {distance_to_opt:.6f}")
    
    plt.savefig('/root/.openclaw/workspace/ml-book-for-kids/chapter20_rosenbrock_comparison.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存到 chapter20_rosenbrock_comparison.png")
    
    return fig, results


def demo_learning_rate_decay():
    """演示学习率衰减的效果"""
    print("\n" + "=" * 60)
    print("演示: 学习率衰减的效果")
    print("=" * 60)
    
    optimizers = [
        SGD(learning_rate=0.1),
        SGDWithDecay(learning_rate=0.1, decay=0.01, decay_type='inverse'),
        SGDWithDecay(learning_rate=0.1, decay=0.95, decay_type='exponential'),
    ]
    
    init_params = [1.5, 1.5]
    loss_fn = lambda x, y: LossLandscape.quadratic(x, y, a=1, b=10)
    grad_fn = lambda x, y: LossLandscape.quadratic_grad(x, y, a=1, b=10)
    
    print(f"\n初始学习率: 0.1")
    print(f"衰减类型: 无衰减, 逆时间衰减, 指数衰减\n")
    
    visualizer = OptimizerVisualizer()
    fig, results = visualizer.compare_optimizers(
        optimizers, loss_fn, grad_fn, init_params,
        n_steps=100, x_range=(-2, 2), y_range=(-2, 2),
        title="Effect of Learning Rate Decay"
    )
    
    plt.savefig('/root/.openclaw/workspace/ml-book-for-kids/chapter20_lr_decay.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存到 chapter20_lr_decay.png")
    
    return fig, results


def print_algorithm_summary():
    """打印算法总结"""
    print("\n" + "=" * 70)
    print("优化器算法总结")
    print("=" * 70)
    
    summary = """
┌─────────────────┬─────────────────────────────────────────────────────┐
│ 优化器           │ 核心特点与适用场景                                    │
├─────────────────┼─────────────────────────────────────────────────────┤
│ SGD             │ • 最简单，所有参数共享学习率                          │
│                 │ • 适合凸优化和大规模数据                              │
│                 │ • 需要仔细调整学习率                                  │
├─────────────────┼─────────────────────────────────────────────────────┤
│ SGD+Momentum    │ • 引入动量，减少震荡，加速收敛                        │
│                 │ • 像滚雪球一样积累速度                                │
│                 │ • 适合大多数深度学习任务                              │
├─────────────────┼─────────────────────────────────────────────────────┤
│ Nesterov        │ • 改进的动量法，预判未来梯度                          │
│                 │ • 收敛更快，更稳定                                    │
│                 │ • 适合需要快速收敛的场景                              │
├─────────────────┼─────────────────────────────────────────────────────┤
│ AdaGrad         │ • 自适应学习率，适合稀疏梯度                          │
│                 │ • 学习率单调递减                                      │
│                 │ • 适合 NLP 的词嵌入等稀疏特征                         │
├─────────────────┼─────────────────────────────────────────────────────┤
│ RMSprop         │ • 使用 EMA 解决 AdaGrad 的递减问题                    │
│                 │ • 适合非凸优化                                        │
│                 │ • 深度学习中广泛使用                                  │
├─────────────────┼─────────────────────────────────────────────────────┤
│ Adam            │ • 结合 Momentum 和 RMSprop 的优点                     │
│                 │ • 自适应学习率 + 动量                                 │
│                 │ • 深度学习默认选择                                    │
│                 │ • 超参数: β1=0.9, β2=0.999, lr=0.001                 │
└─────────────────┴─────────────────────────────────────────────────────┘

选择建议:
1. 快速原型 → 用 Adam (lr=3e-4)
2. 追求最终精度 → 先用 Adam 预训练，再用 SGD+Momentum 微调
3. NLP / 稀疏数据 → AdamW 或 AdaGrad
4. CV / 图像任务 → SGD+Momentum (lr=0.1, momentum=0.9)
5. 强化学习 → RMSprop 或 Adam
"""
    print(summary)


# 主程序
if __name__ == "__main__":
    # 运行演示
    demo_basic_optimizers()
    demo_rosenbrock()
    demo_learning_rate_decay()
    print_algorithm_summary()
    
    plt.show()
