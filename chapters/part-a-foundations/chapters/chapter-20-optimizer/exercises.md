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
