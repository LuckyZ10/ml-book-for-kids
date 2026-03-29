# 第十章：支持向量机——寻找最优分界线

> *"The art of doing mathematics consists in finding that special case which contains all the germs of generality."*  
> *—— David Hilbert*

---

## 开篇故事：怎样划分两个班级学生的座位

新学期开始了，阳光小学五年级的两个班级——**向日葵班**（🌻）和**星空班**（⭐）要共用一间大教室上课。校长给班主任李老师出了个难题：

> "这两个班的学习风格很不一样。向日葵班的同学喜欢明亮、靠近窗户的位置；星空班的同学偏爱安静、靠墙的位置。你能不能想办法，**只用一条走道**就把两个班级分开，让每个班的同学都坐在最适合自己的区域？"

李老师看着教室的平面图，发现两个班的同学分布是这样的：

```
    窗户
    ═══════════════════
    🌻    🌻      ⭐    ⭐
      🌻  🌻🌻   ⭐⭐   ⭐
    🌻  🌻        ⭐  ⭐⭐
    ───────────────────
    墙壁
```

"我可以这样划分！"李老师在中间画了一条走道：

```
    ═══════════════════
    🌻    🌻  ║  ⭐    ⭐
      🌻  🌻🌻║ ⭐⭐   ⭐
    🌻  🌻    ║  ⭐  ⭐⭐
    ───────────────────
              ↑
           走道
```

但李老师很快发现，**能划分两个班级的走道有无数条**：

```
    ═══════════════════
    🌻    🌻  ║  ⭐    ⭐
      🌻  🌻  ║🌻 ⭐⭐  ⭐     ← 太靠近向日葵班了！
    🌻  🌻    ║  ⭐  ⭐⭐
    ───────────────────
    
    ═══════════════════
    🌻    🌻     ⭐  ║ ⭐
      🌻  🌻🌻  ⭐⭐ ║ ⭐     ← 太靠近星空班了！
    🌻  🌻       ⭐ ║⭐⭐
    ───────────────────
```

到底哪条走道才是最好的呢？

李老师想了想，提出了一个聪明的方案：

> "我们应该找**最宽的走道**！这样两个班级的同学都有足够的空间，不会因为走道太窄而互相干扰。"

这就是**支持向量机（Support Vector Machine, SVM）**的核心思想！

---

## 10.1 什么是最优分界线？

### 10.1.1 从走道到"间隔"

在数学上，我们称这条走道为**决策边界**（Decision Boundary），而这条走道的宽度叫做**间隔**（Margin）。

**间隔** = 最近的🌻同学到走道的距离 + 最近的⭐同学到走道的距离

```
    ═══════════════════
    🌻    🌻     ⭐    ⭐
      🌻  ↓    ↓  ⭐  ⭐
    🌻  [🌻]──走道──[⭐] ⭐⭐
          ↑    ↑
       支持向量(最近的点)
    
    ←───── 间隔(Margin) ─────→
```

那些距离走道最近的点（用方框标记的），我们称之为**支持向量**（Support Vectors）。它们是"支撑"着整条走道的关键同学——只要这些同学的位置不变，走道的位置就不会变！

### 10.1.2 最大间隔原理

**SVM的核心思想**：在所有能正确分开两个班级的走道中，选择**最宽的那一条**。

为什么最宽的走道最好？想象一下：

| 走道类型 | 特点 | 问题 |
|---------|------|------|
| 很窄的走道 | 勉强能分开 | 稍微有同学移动就会越界 |
| 中等走道 | 有一定空间 | 可以容忍小的变动 |
| **最宽的走道** | **两边空间最大** | **最稳定，最能容忍新同学** |

这就是**最大间隔原理**：**最宽的间隔 = 最好的泛化能力**

> 💡 **费曼比喻**：想象你在两群吵架的孩子中间拉一条警戒线。如果你把线拉得离某一帮孩子很近，他们一伸手就能碰到对方，很容易再吵起来。但如果你找到"最公平"的位置，让两边都有足够的空间，那么这条线就最稳定！

---

## 10.2 数学推导：从几何到优化

现在，让我们用数学语言来描述这个问题。

### 10.2.1 用向量描述走道

在二维平面上，一条直线（走道）可以用下面的方程描述：

$$\mathbf{w} \cdot \mathbf{x} + b = 0$$

其中：
- $\mathbf{x} = (x_1, x_2)$ 是教室里的任意位置
- $\mathbf{w} = (w_1, w_2)$ 是垂直于走道的方向向量（像是指向"上方"的箭头）
- $b$ 是偏置项（决定走道离原点的距离）

**走道的两条边界**可以表示为：
- 向日葵班一侧：$\mathbf{w} \cdot \mathbf{x} + b = +1$
- 星空班一侧：$\mathbf{w} \cdot \mathbf{x} + b = -1$

### 10.2.2 计算间隔的宽度

两条平行直线之间的距离公式是：

$$\text{间隔} = \frac{2}{\|\mathbf{w}\|}$$

其中 $\|\mathbf{w}\| = \sqrt{w_1^2 + w_2^2}$ 是向量 $\mathbf{w}$ 的长度。

**我们的目标**是：
> **最大化间隔** $\frac{2}{\|\mathbf{w}\|}$，这等价于 **最小化** $\frac{1}{2}\|\mathbf{w}\|^2$

### 10.2.3 约束条件

走道必须正确分开两个班级：

对于向日葵班的同学（标签 $y_i = +1$）：
$$\mathbf{w} \cdot \mathbf{x}_i + b \geq 1$$

对于星空班的同学（标签 $y_i = -1$）：
$$\mathbf{w} \cdot \mathbf{x}_i + b \leq -1$$

这两个条件可以合并写成：
$$y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1, \quad \text{对于所有 } i$$

### 10.2.4 优化问题

现在，我们可以写出SVM的**原始优化问题**：

$$\begin{aligned}
\min_{\mathbf{w}, b} \quad & \frac{1}{2}\|\mathbf{w}\|^2 \\
\text{约束：} \quad & y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1, \quad i = 1, 2, ..., n
\end{aligned}$$

> 🌈 **彩色标记理解**：
> - 🔵 **蓝色**：我们要最小化的目标（让间隔尽可能大）
> - 🟢 **绿色**：约束条件（必须正确分类所有点）

### 10.2.5 拉格朗日乘子法

这是一个带约束的优化问题，我们使用**拉格朗日乘子法**来解决。

构造拉格朗日函数：

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^{n} \alpha_i [y_i(\mathbf{w} \cdot \mathbf{x}_i + b) - 1]$$

其中 $\alpha_i \geq 0$ 是拉格朗日乘子。

### 10.2.6 对偶问题

通过对 $\mathbf{w}$ 和 $b$ 求偏导并令其为零，我们得到：

$$\mathbf{w} = \sum_{i=1}^{n} \alpha_i y_i \mathbf{x}_i$$

$$\sum_{i=1}^{n} \alpha_i y_i = 0$$

将这些代回拉格朗日函数，得到**对偶问题**：

$$\begin{aligned}
\max_{\boldsymbol{\alpha}} \quad & \sum_{i=1}^{n} \alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j (\mathbf{x}_i \cdot \mathbf{x}_j) \\
\text{约束：} \quad & \sum_{i=1}^{n} \alpha_i y_i = 0 \\
& \alpha_i \geq 0, \quad i = 1, 2, ..., n
\end{aligned}$$

> 💡 **为什么对偶问题更好？**
> 1. 只涉及 $\alpha$（标量），而不是整个 $\mathbf{w}$（向量）
> 2. 损失函数只依赖于样本之间的内积 $\mathbf{x}_i \cdot \mathbf{x}_j$
> 3. 为后续的"核技巧"铺平道路！

### 10.2.7 KKT条件

在最优解处，必须满足**Karush-Kuhn-Tucker (KKT) 条件**：

1. **原始可行**：$y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1$
2. **对偶可行**：$\alpha_i \geq 0$
3. **互补松弛**：$\alpha_i [y_i(\mathbf{w} \cdot \mathbf{x}_i + b) - 1] = 0$

**互补松弛条件**告诉我们一个重要的结论：

> 只有当 $y_i(\mathbf{w} \cdot \mathbf{x}_i + b) = 1$ 时，$\alpha_i > 0$
> 
> 换句话说，**只有支持向量对应的 $\alpha_i$ 才不为零！**

这就是"支持向量"名字的由来——它们"支撑"着整个模型！

---

## 10.3 软间隔：允许犯错的智慧

### 10.3.1 现实世界不完美

在实际问题中，数据往往不是**完全线性可分**的。可能有些"调皮"的同学：

```
    🌻    🌻  [⭐]   ⭐    ← 星空班有个同学坐在了向日葵班区域！
      🌻  🌻🌻   ⭐⭐   ⭐
    🌻 [🌻]        ⭐  ⭐⭐
```

如果强行要求100%正确分类，可能导致：
- 模型过于复杂
- 间隔变得极小
- 泛化能力很差（过拟合）

### 10.3.2 松弛变量

Cortes 和 Vapnik 在1995年提出了**软间隔 SVM**（Soft Margin SVM），允许某些点"违规"。

我们引入**松弛变量** $\xi_i \geq 0$，表示第 $i$ 个点"违规的程度"：

```
                 松弛变量 ξ_i
                      ↓
    ───────────────────────────────────
    🌻    🌻  [⭐══════►]   ⭐
              ↑         
         这个点违规了，但它到正确边的距离就是 ξ_i
```

新的约束条件：
$$y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i$$

### 10.3.3 带正则化的损失函数

优化问题变为：

$$\begin{aligned}
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \quad & \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n} \xi_i \\
\text{约束：} \quad & y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i \\
& \xi_i \geq 0, \quad i = 1, 2, ..., n
\end{aligned}$$

其中 $C$ 是**正则化参数**：
- $C$ 很大：严格分类，不允许犯错（硬间隔）
- $C$ 很小：允许较多错误，追求大间隔
- $C$ 适中：平衡两者

> 💡 **费曼比喻**：$C$ 就像老师对学生的"严厉程度"。$C$ 很大 = 严厉的老师，不允许任何违规；$C$ 很小 = 宽容的老师，允许学生偶尔越界，只要整体秩序好就行。

---

## 10.4 核技巧：折叠纸张的魔法

### 10.4.1 线性不可分的问题

有些情况下，两个班级根本无法用一条直线分开：

```
    🌻  🌻  🌻
  🌻         🌻
      ⭐⭐⭐      ← 星空班在中间！
      ⭐⭐⭐
  🌻         🌻
    🌻  🌻  🌻
```

这种情况下，**无论怎么画直线都不行**！

### 10.4.2 高维映射的直觉

想象你有一张纸，上面画着这样的图案：

```
    平面上的分布：        折叠后的立体：
    🌻 🌻 🌻           
  🌻       🌻            🌻  ⭐
      ⭐⭐⭐     ──→     🌻  ⭐   （在3D空间中变得线性可分！）
      ⭐⭐⭐            🌻  ⭐
  🌻       🌻
    🌻 🌻 🌻
```

如果我们把中间的点"向上拉"，边缘的点"向下压"，在三维空间中，就可能找到一个平面把🌻和⭐分开！

数学上，这就是**特征映射** $\phi(\mathbf{x})$，把数据从低维空间映射到高维空间：

$$\phi: \mathbb{R}^2 \rightarrow \mathbb{R}^3$$
$$(x_1, x_2) \mapsto (x_1, x_2, x_1^2 + x_2^2)$$

### 10.4.3 核函数的奇迹

但是，直接计算高维映射 $\phi(\mathbf{x})$ 可能非常复杂，甚至维度是**无限**的！

**核技巧（Kernel Trick）**的魔法在于：

> 我们不需要显式计算 $\phi(\mathbf{x})$，只需要计算**核函数** $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)$

对偶问题中的损失函数只依赖于内积，所以我们可以直接用核函数代替：

$$\sum_{i=1}^{n} \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$

### 10.4.4 常用核函数

#### 1️⃣ 线性核（Linear Kernel）

$$K(\mathbf{x}, \mathbf{x}') = \mathbf{x} \cdot \mathbf{x}'$$

- 就是原始空间的内积
- 适用于线性可分的数据

#### 2️⃣ 多项式核（Polynomial Kernel）

$$K(\mathbf{x}, \mathbf{x}') = (\gamma \mathbf{x} \cdot \mathbf{x}' + r)^d$$

其中 $d$ 是多项式次数，$\gamma$ 和 $r$ 是参数。

**例子**：当 $d=2$，$\gamma=1$，$r=0$ 时，对于二维向量：

$$K(\mathbf{x}, \mathbf{x}') = (x_1 x_1' + x_2 x_2')^2$$

展开后等价于映射到特征 $(x_1^2, x_2^2, \sqrt{2}x_1 x_2)$ 空间！

#### 3️⃣ 高斯径向基核（RBF Kernel）

$$K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right)$$

- 最常用的核函数
- 把数据映射到**无限维**空间
- $\gamma$ 越大，决策边界越复杂（越"弯曲"）

> 💡 **直观理解 RBF 核**：
> - 两个点越近，核函数值越接近 1（越"相似"）
> - 两个点越远，核函数值越接近 0（越"不相似"）
> - 这就像在问："这两个学生坐在多近的位置？"

#### 4️⃣ Sigmoid核

$$K(\mathbf{x}, \mathbf{x}') = \tanh(\gamma \mathbf{x} \cdot \mathbf{x}' + r)$$

- 类似于神经网络中的激活函数
- 较少使用，但在某些特定问题上表现好

### 10.4.5 折叠纸张的比喻

> 📝 **费曼式解释**：想象你有一张纸，上面用红笔和蓝笔画了两个交错的圆圈。如果你只在纸面上找直线，无论如何都分不开红蓝两色。
> 
> 但是！如果你**把纸张折叠**一下，让中间鼓起来，边缘压下去，那么在三维空间中，你就能找到一个平面把红蓝两色分开！
> 
> 核技巧就是这个"折叠"操作——它不改变点在纸上的位置关系，只是给了它们一个新的"高度"，让原本纠缠的数据变得可分！

---

## 10.5 代码实现：从零开始写SVM

现在让我们用 NumPy 实现 SVM！我们会实现：
1. 线性SVM（使用梯度下降）
2. 核SVM（使用SMO算法简化版）

### 10.5.1 线性SVM（梯度下降法）

这是一个简化版，使用次梯度下降来优化软间隔目标：

```python
"""
线性SVM - 简化实现
使用次梯度下降法优化软间隔损失函数
"""
import numpy as np
import matplotlib.pyplot as plt

class LinearSVM:
    """
    线性支持向量机
    
    参数:
        C: 正则化参数（越大越严格）
        learning_rate: 学习率
        n_iterations: 迭代次数
    """
    def __init__(self, C=1.0, learning_rate=0.001, n_iterations=1000):
        self.C = C
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = None  # 权重向量
        self.b = None  # 偏置项
        
    def fit(self, X, y):
        """
        训练SVM
        
        参数:
            X: 训练数据，形状 (n_samples, n_features)
            y: 标签，形状 (n_samples,)，取值为 +1 或 -1
        """
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.w = np.zeros(n_features)
        self.b = 0
        
        # 梯度下降优化
        for iteration in range(self.n_iterations):
            # 计算每个样本的约束违反情况
            margins = y * (np.dot(X, self.w) + self.b)
            
            # 计算次梯度
            # 对于 w: 如果 margin < 1，梯度包含 -C * y_i * x_i
            # 对于 b: 如果 margin < 1，梯度包含 -C * y_i
            
            # 找出违反约束的样本（margin < 1）
            misclassified = margins < 1
            
            # 计算 w 的梯度
            # ∇_w = w - C * Σ(y_i * x_i) for misclassified
            grad_w = self.w - self.C * np.sum((y[misclassified][:, None] * X[misclassified]), axis=0) / n_samples
            
            # 计算 b 的梯度
            # ∇_b = -C * Σ(y_i) for misclassified
            grad_b = -self.C * np.sum(y[misclassified]) / n_samples
            
            # 更新参数
            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b
            
            # 每100次迭代打印一次损失
            if (iteration + 1) % 100 == 0:
                loss = self._compute_loss(X, y)
                print(f"Iteration {iteration + 1}/{self.n_iterations}, Loss: {loss:.4f}")
    
    def _compute_loss(self, X, y):
        """计算 hinge loss + L2 正则化的损失函数值"""
        # Hinge loss: max(0, 1 - y * (w·x + b))
        margins = y * (np.dot(X, self.w) + self.b)
        hinge_loss = np.maximum(0, 1 - margins)
        
        # 总损失 = 0.5 * ||w||^2 + C * Σ hinge_loss
        loss = 0.5 * np.dot(self.w, self.w) + self.C * np.sum(hinge_loss)
        return loss
    
    def predict(self, X):
        """
        预测类别
        
        参数:
            X: 测试数据
            
        返回:
            预测标签 (+1 或 -1)
        """
        scores = np.dot(X, self.w) + self.b
        return np.sign(scores)
    
    def decision_function(self, X):
        """
        计算决策函数值（到超平面的有符号距离）
        
        参数:
            X: 测试数据
            
        返回:
            决策函数值
        """
        return np.dot(X, self.w) + self.b
    
    def get_support_vectors(self, X, y, tolerance=1e-5):
        """
        获取支持向量（距离决策边界最近的点）
        
        参数:
            X: 数据
            y: 标签
            tolerance: 判定为支持向量的阈值
            
        返回:
            支持向量的索引
        """
        margins = np.abs(y * self.decision_function(X) - 1)
        return np.where(margins < tolerance)[0]


def visualize_linear_svm():
    """可视化线性SVM的分类效果"""
    np.random.seed(42)
    
    # 生成线性可分的数据
    # 向日葵班（类别 +1）
    X_sunflower = np.random.randn(50, 2) + np.array([2, 2])
    # 星空班（类别 -1）
    X_starry = np.random.randn(50, 2) + np.array([-2, -2])
    
    X = np.vstack([X_sunflower, X_starry])
    y = np.hstack([np.ones(50), -np.ones(50)])
    
    # 训练SVM
    svm = LinearSVM(C=1.0, learning_rate=0.01, n_iterations=1000)
    svm.fit(X, y)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    
    # 绘制数据点
    plt.scatter(X[:50, 0], X[:50, 1], c='gold', s=100, marker='o', 
                edgecolors='black', label='🌻 向日葵班 (+1)', alpha=0.8)
    plt.scatter(X[50:, 0], X[50:, 1], c='navy', s=100, marker='s', 
                edgecolors='black', label='⭐ 星空班 (-1)', alpha=0.8)
    
    # 获取支持向量
    sv_indices = svm.get_support_vectors(X, y, tolerance=0.1)
    if len(sv_indices) > 0:
        plt.scatter(X[sv_indices, 0], X[sv_indices, 1], s=300, 
                   facecolors='none', edgecolors='red', linewidths=2,
                   label='🔴 支持向量')
    
    # 绘制决策边界和间隔边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx = np.linspace(x_min, x_max, 100)
    
    # 决策边界: w·x + b = 0  =>  y = -(w[0]*x + b) / w[1]
    yy_decision = -(svm.w[0] * xx + svm.b) / svm.w[1]
    # 间隔边界: w·x + b = ±1
    yy_plus = -(svm.w[0] * xx + svm.b - 1) / svm.w[1]
    yy_minus = -(svm.w[0] * xx + svm.b + 1) / svm.w[1]
    
    plt.plot(xx, yy_decision, 'k-', linewidth=2, label='决策边界')
    plt.plot(xx, yy_plus, 'k--', linewidth=1, alpha=0.5, label='间隔边界')
    plt.plot(xx, yy_minus, 'k--', linewidth=1, alpha=0.5)
    
    # 填充间隔区域
    plt.fill_between(xx, yy_minus, yy_plus, alpha=0.1, color='gray', label='间隔区域')
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('位置 x₁', fontsize=12)
    plt.ylabel('位置 x₂', fontsize=12)
    plt.title('🎓 线性SVM：寻找最宽的走道', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('linear_svm.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n训练完成！")
    print(f"权重向量 w = {svm.w}")
    print(f"偏置项 b = {svm.b:.4f}")
    print(f"间隔宽度 = {2 / np.linalg.norm(svm.w):.4f}")
    print(f"支持向量数量 = {len(sv_indices)}")
    
    return svm


# 运行可视化
if __name__ == "__main__":
    print("=" * 60)
    print("🌻 Linear SVM Demo - 寻找最宽的走道 🌻")
    print("=" * 60)
    svm = visualize_linear_svm()
```

### 10.5.2 核函数实现

```python
"""
核函数实现
包含线性核、多项式核、RBF核
"""
import numpy as np

class Kernels:
    """核函数集合"""
    
    @staticmethod
    def linear():
        """
        线性核: K(x, x') = x · x'
        """
        def kernel(X1, X2):
            return np.dot(X1, X2.T)
        return kernel
    
    @staticmethod
    def polynomial(gamma=1.0, coef0=1.0, degree=3):
        """
        多项式核: K(x, x') = (γ · x·x' + r)^d
        
        参数:
            gamma: 缩放参数 γ
            coef0: 常数项 r
            degree: 多项式次数 d
        """
        def kernel(X1, X2):
            return (gamma * np.dot(X1, X2.T) + coef0) ** degree
        return kernel
    
    @staticmethod
    def rbf(gamma=1.0):
        """
        RBF（高斯径向基）核: K(x, x') = exp(-γ ||x - x'||²)
        
        参数:
            gamma: 控制高斯函数的宽度
                   越大 → 核函数越"尖锐" → 模型越复杂
                   越小 → 核函数越"平坦" → 模型越简单
        """
        def kernel(X1, X2):
            # 计算两两之间的欧氏距离平方
            # ||x - x'||² = ||x||² + ||x'||² - 2x·x'
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            
            # 距离平方矩阵
            dist_sq = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
            
            return np.exp(-gamma * dist_sq)
        return kernel
    
    @staticmethod
    def sigmoid(gamma=1.0, coef0=0.0):
        """
        Sigmoid核: K(x, x') = tanh(γ · x·x' + r)
        
        参数:
            gamma: 缩放参数
            coef0: 常数项
        """
        def kernel(X1, X2):
            return np.tanh(gamma * np.dot(X1, X2.T) + coef0)
        return kernel


def demo_kernels():
    """演示不同核函数的效果"""
    # 两个示例向量
    x1 = np.array([[1, 2]])
    x2 = np.array([[3, 4]])
    
    print("=" * 60)
    print("🔍 核函数演示")
    print("=" * 60)
    print(f"向量 x1 = {x1}")
    print(f"向量 x2 = {x2}")
    print(f"x1 和 x2 的距离 = {np.linalg.norm(x1 - x2):.4f}")
    print()
    
    # 线性核
    linear_k = Kernels.linear()
    print(f"📏 线性核 K(x1, x2) = {linear_k(x1, x2)[0, 0]:.4f}")
    print(f"   （就是两个向量的内积）")
    print()
    
    # 多项式核
    poly_k = Kernels.polynomial(gamma=1.0, coef0=1.0, degree=2)
    print(f"📐 多项式核(degree=2) K(x1, x2) = {poly_k(x1, x2)[0, 0]:.4f}")
    print(f"   （等价于映射到高维后的内积）")
    print()
    
    # RBF核
    rbf_k = Kernels.rbf(gamma=0.5)
    print(f"🔵 RBF核(gamma=0.5) K(x1, x2) = {rbf_k(x1, x2)[0, 0]:.4f}")
    print(f"   （距离越远，核函数值越小）")
    print()
    
    # 展示 gamma 对 RBF 的影响
    print("🎚️ gamma 参数对 RBF 核的影响:")
    print("-" * 40)
    for gamma in [0.1, 0.5, 1.0, 5.0, 10.0]:
        rbf = Kernels.rbf(gamma=gamma)
        value = rbf(x1, x2)[0, 0]
        print(f"  gamma={gamma:4.1f}: K(x1, x2) = {value:.6f}")
    print("\n  gamma 越大 → 核函数衰减越快 → 模型越"复杂"（容易过拟合）")


if __name__ == "__main__":
    demo_kernels()
```

### 10.5.3 SMO算法简化实现

```python
"""
SMO (Sequential Minimal Optimization) 算法简化实现
用于高效求解SVM对偶问题

参考: Platt, J. (1998). Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines
"""
import numpy as np
import random

class SimplifiedSMO:
    """
    SMO算法简化实现
    
    核心思想：每次只优化两个拉格朗日乘子 α_i 和 α_j，
    这样可以解析求解，而不需要复杂的QP优化器。
    """
    def __init__(self, X, y, C=1.0, tolerance=0.001, max_passes=100, kernel_type='linear', gamma=1.0):
        """
        初始化SMO
        
        参数:
            X: 训练数据 (n_samples, n_features)
            y: 标签 (n_samples,)，取值为 +1 或 -1
            C: 正则化参数
            tolerance: KKT条件违反的容差
            max_passes: 最大迭代轮数
            kernel_type: 'linear' 或 'rbf'
            gamma: RBF核参数
        """
        self.X = X
        self.y = y
        self.C = C
        self.tol = tolerance
        self.max_passes = max_passes
        self.kernel_type = kernel_type
        self.gamma = gamma
        
        self.m, self.n = X.shape  # 样本数和特征数
        
        # 初始化拉格朗日乘子 α 和偏置 b
        self.alphas = np.zeros(self.m)
        self.b = 0.0
        
        # 预计算核矩阵（简化版，适用于中小数据集）
        self.K = self._compute_kernel_matrix()
        
    def _compute_kernel_matrix(self):
        """计算核矩阵 K[i,j] = K(x_i, x_j)"""
        if self.kernel_type == 'linear':
            # 线性核: K(x, x') = x · x'
            return np.dot(self.X, self.X.T)
        elif self.kernel_type == 'rbf':
            # RBF核: K(x, x') = exp(-γ ||x - x'||²)
            X_norm = np.sum(self.X**2, axis=1).reshape(-1, 1)
            dist_sq = X_norm + X_norm.T - 2 * np.dot(self.X, self.X.T)
            return np.exp(-self.gamma * dist_sq)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def _kernel(self, i, j):
        """获取核矩阵元素 K[i,j]"""
        return self.K[i, j]
    
    def _predict_output(self, i):
        """
        计算样本 i 的预测输出 f(x_i)
        f(x_i) = Σ(α_k · y_k · K(x_k, x_i)) + b
        """
        return np.sum(self.alphas * self.y * self.K[:, i]) + self.b
    
    def _calculate_error(self, i):
        """计算样本 i 的预测误差 E_i = f(x_i) - y_i"""
        return self._predict_output(i) - self.y[i]
    
    def _select_j_randomly(self, i):
        """随机选择 j ≠ i"""
        j = i
        while j == i:
            j = random.randint(0, self.m - 1)
        return j
    
    def _clip_alpha(self, alpha, H, L):
        """将 α 裁剪到 [L, H] 范围内"""
        if alpha > H:
            return H
        if alpha < L:
            return L
        return alpha
    
    def _take_step(self, i, j):
        """
        尝试优化 α_i 和 α_j 这一对乘子
        
        返回 True 如果成功更新，False 如果没有进展
        """
        if i == j:
            return False
        
        alpha_i_old = self.alphas[i].copy()
        alpha_j_old = self.alphas[j].copy()
        yi, yj = self.y[i], self.y[j]
        
        # 计算误差
        Ei = self._calculate_error(i)
        Ej = self._calculate_error(j)
        
        # 计算 α_j 的边界 L 和 H
        if yi != yj:
            # 当 y_i ≠ y_j 时
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)
        else:
            # 当 y_i = y_j 时
            L = max(0, alpha_i_old + alpha_j_old - self.C)
            H = min(self.C, alpha_i_old + alpha_j_old)
        
        if L == H:
            return False
        
        # 计算 η = K_ii + K_jj - 2K_ij
        eta = self._kernel(i, i) + self._kernel(j, j) - 2 * self._kernel(i, j)
        
        if eta <= 0:
            return False
        
        # 计算未裁剪的新 α_j
        alpha_j_new_unc = alpha_j_old + yj * (Ei - Ej) / eta
        
        # 裁剪 α_j 到 [L, H]
        alpha_j_new = self._clip_alpha(alpha_j_new_unc, H, L)
        
        # 检查变化是否显著
        if abs(alpha_j_new - alpha_j_old) < 1e-5:
            return False
        
        # 计算新的 α_i
        # α_i^new = α_i^old + y_i·y_j·(α_j^old - α_j^new)
        alpha_i_new = alpha_i_old + yi * yj * (alpha_j_old - alpha_j_new)
        
        # 更新偏置 b
        b1 = (self.b - Ei - yi * (alpha_i_new - alpha_i_old) * self._kernel(i, i) 
              - yj * (alpha_j_new - alpha_j_old) * self._kernel(i, j))
        b2 = (self.b - Ej - yi * (alpha_i_new - alpha_i_old) * self._kernel(i, j) 
              - yj * (alpha_j_new - alpha_j_old) * self._kernel(j, j))
        
        # 根据 α 是否在边界内来选择 b
        if 0 < alpha_i_new < self.C:
            self.b = b1
        elif 0 < alpha_j_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2.0
        
        # 更新 α
        self.alphas[i] = alpha_i_new
        self.alphas[j] = alpha_j_new
        
        return True
    
    def _examine_example(self, i):
        """
        检查样本 i 的KKT条件，尝试优化它
        
        KKT条件：
        - 如果 α_i = 0，则 y_i·f(x_i) ≥ 1（样本正确分类且在间隔外）
        - 如果 0 < α_i < C，则 y_i·f(x_i) = 1（样本在间隔边界上）
        - 如果 α_i = C，则 y_i·f(x_i) ≤ 1（样本在间隔内或分类错误）
        """
        yi = self.y[i]
        alpha_i = self.alphas[i]
        Ei = self._calculate_error(i)
        
        # 检查KKT条件是否违反
        r = Ei * yi
        
        # 违反条件的情况：
        # 1. r < -tol 且 α_i < C（应该增加 α_i）
        # 2. r > tol 且 α_i > 0（应该减小 α_i）
        violate_kkt = (r < -self.tol and alpha_i < self.C) or (r > self.tol and alpha_i > 0)
        
        if not violate_kkt:
            return False
        
        # 启发式1：优先选择非边界上的 α_j（0 < α < C）
        non_bound_idx = np.where((self.alphas > 0) & (self.alphas < self.C))[0]
        
        if len(non_bound_idx) > 1:
            # 选择使 |Ei - Ej| 最大的 j
            max_delta_E = 0
            best_j = -1
            for k in non_bound_idx:
                if k == i:
                    continue
                Ek = self._calculate_error(k)
                delta_E = abs(Ei - Ek)
                if delta_E > max_delta_E:
                    max_delta_E = delta_E
                    best_j = k
            
            if best_j != -1 and self._take_step(i, best_j):
                return True
        
        # 启发式2：在非边界点中随机尝试
        non_bound_list = list(non_bound_idx)
        random.shuffle(non_bound_list)
        for j in non_bound_list:
            if j != i and self._take_step(i, j):
                return True
        
        # 启发式3：在所有点中随机尝试
        all_idx = list(range(self.m))
        random.shuffle(all_idx)
        for j in all_idx:
            if j != i and self._take_step(i, j):
                return True
        
        return False
    
    def fit(self):
        """训练SVM"""
        print("🚀 开始SMO训练...")
        
        num_changed = 0
        examine_all = True
        passes = 0
        
        while (num_changed > 0 or examine_all) and passes < self.max_passes:
            num_changed = 0
            
            if examine_all:
                # 遍历所有样本
                for i in range(self.m):
                    if self._examine_example(i):
                        num_changed += 1
                print(f"  全遍历轮次: 更新了 {num_changed} 个 α")
            else:
                # 只遍历非边界样本
                non_bound_idx = np.where((self.alphas > 0) & (self.alphas < self.C))[0]
                for i in non_bound_idx:
                    if self._examine_example(i):
                        num_changed += 1
                print(f"  非边界遍历: 更新了 {num_changed} 个 α")
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            passes += 1
            print(f"  完成第 {passes} 轮")
        
        print(f"✅ 训练完成！共 {passes} 轮")
        
        # 提取支持向量
        self.support_vector_idx = np.where(self.alphas > 1e-5)[0]
        self.support_vectors = self.X[self.support_vector_idx]
        self.support_vector_labels = self.y[self.support_vector_idx]
        self.support_vector_alphas = self.alphas[self.support_vector_idx]
        
        print(f"📊 支持向量数量: {len(self.support_vector_idx)} / {self.m}")
        
    def predict(self, X):
        """
        预测新样本的类别
        
        f(x) = Σ(α_sv · y_sv · K(x_sv, x)) + b
        """
        if self.kernel_type == 'linear':
            # 线性核可以直接计算 w·x + b
            w = np.sum((self.alphas * self.y).reshape(-1, 1) * self.X, axis=0)
            scores = np.dot(X, w) + self.b
        else:
            # 非线性核需要计算与所有支持向量的核函数
            scores = np.zeros(X.shape[0])
            for i, x in enumerate(X):
                # 计算 x 与所有支持向量的核函数值
                if self.kernel_type == 'rbf':
                    # RBF核
                    dist_sq = np.sum((self.support_vectors - x)**2, axis=1)
                    k_values = np.exp(-self.gamma * dist_sq)
                scores[i] = np.sum(self.support_vector_alphas * self.support_vector_labels * k_values) + self.b
        
        return np.sign(scores)
    
    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


def demo_smo():
    """演示SMO算法"""
    from sklearn.datasets import make_blobs, make_circles, make_moons
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("🎯 SMO算法演示")
    print("=" * 60)
    
    # 测试1: 线性可分数据
    print("\n📌 测试1: 线性可分数据")
    X1, y1 = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)
    y1 = np.where(y1 == 0, -1, 1)
    
    svm1 = SimplifiedSMO(X1, y1, C=1.0, kernel_type='linear', max_passes=50)
    svm1.fit()
    acc1 = svm1.score(X1, y1)
    print(f"训练准确率: {acc1 * 100:.2f}%")
    
    # 测试2: 非线性数据（月亮形状）
    print("\n📌 测试2: 非线性数据（月亮形状）- 使用RBF核")
    X2, y2 = make_moons(n_samples=100, noise=0.1, random_state=42)
    y2 = np.where(y2 == 0, -1, 1)
    
    svm2 = SimplifiedSMO(X2, y2, C=10.0, kernel_type='rbf', gamma=5.0, max_passes=100)
    svm2.fit()
    acc2 = svm2.score(X2, y2)
    print(f"训练准确率: {acc2 * 100:.2f}%")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 图1: 线性数据
    plot_svm_boundary(axes[0], X1, y1, svm1, "线性SVM")
    
    # 图2: 非线性数据
    plot_svm_boundary(axes[1], X2, y2, svm2, "RBF核SVM")
    
    plt.tight_layout()
    plt.savefig('smo_demo.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_svm_boundary(ax, X, y, svm, title):
    """绘制SVM决策边界"""
    # 绘制数据点
    pos_idx = y == 1
    neg_idx = y == -1
    ax.scatter(X[pos_idx, 0], X[pos_idx, 1], c='gold', s=50, 
              edgecolors='black', label='Class +1', alpha=0.8)
    ax.scatter(X[neg_idx, 0], X[neg_idx, 1], c='navy', s=50, 
              edgecolors='black', label='Class -1', alpha=0.8)
    
    # 绘制支持向量
    if len(svm.support_vector_idx) > 0:
        ax.scatter(X[svm.support_vector_idx, 0], X[svm.support_vector_idx, 1], 
                  s=200, facecolors='none', edgecolors='red', linewidths=2,
                  label='Support Vectors')
    
    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, levels=[-2, 0, 2], colors=['blue', 'red'])
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    demo_smo()
```

### 10.5.4 完整演示脚本

```python
"""
SVM完整演示：比较不同核函数的效果
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons

# 导入我们实现的类
from linear_svm import LinearSVM
from smo_svm import SimplifiedSMO

def compare_kernels():
    """比较不同核函数在各类数据集上的表现"""
    
    # 生成三种不同类型的数据集
    datasets = []
    
    # 1. 线性可分数据
    X1, y1 = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.2)
    y1 = np.where(y1 == 0, -1, 1)
    datasets.append(("🌻 线性可分数据", X1, y1))
    
    # 2. 同心圆数据（必须使用核函数）
    X2, y2 = make_circles(n_samples=100, factor=0.5, noise=0.08, random_state=42)
    y2 = np.where(y2 == 0, -1, 1)
    datasets.append(("⭐ 同心圆数据", X2, y2))
    
    # 3. 月亮数据
    X3, y3 = make_moons(n_samples=100, noise=0.15, random_state=42)
    y3 = np.where(y3 == 0, -1, 1)
    datasets.append(("🌙 月亮形状数据", X3, y3))
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for row, (data_name, X, y) in enumerate(datasets):
        print(f"\n{'='*60}")
        print(f"📊 数据集: {data_name}")
        print(f"{'='*60}")
        
        # 测试三种核函数
        configs = [
            ("线性核", "linear", {}),
            ("RBF核(γ=1)", "rbf", {"gamma": 1.0}),
            ("RBF核(γ=10)", "rbf", {"gamma": 10.0}),
        ]
        
        for col, (kernel_name, kernel_type, kernel_params) in enumerate(configs):
            print(f"\n  🔧 {kernel_name}")
            
            try:
                # 训练SMO SVM
                svm = SimplifiedSMO(X, y, C=1.0, kernel_type=kernel_type, 
                                   max_passes=50, **kernel_params)
                svm.fit()
                accuracy = svm.score(X, y)
                print(f"     准确率: {accuracy*100:.1f}%")
                
                # 绘制结果
                ax = axes[row, col]
                plot_decision_boundary(ax, X, y, svm, f"{data_name}\n{kernel_name}")
                
            except Exception as e:
                print(f"     错误: {e}")
                ax = axes[row, col]
                ax.text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10)
                ax.set_title(f"{data_name}\n{kernel_name}", fontsize=10)
    
    plt.tight_layout()
    plt.savefig('kernel_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n{'='*60}")
    print("🎉 所有测试完成！")
    print("观察结果：")
    print("  - 线性数据：线性核表现最好")
    print("  - 圆形/月亮数据：RBF核能处理非线性边界")
    print("  - gamma越大：决策边界越复杂，可能过拟合")
    print(f"{'='*60}")


def plot_decision_boundary(ax, X, y, svm, title):
    """绘制决策边界"""
    # 确定绘图范围
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # 创建网格
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # 预测网格点
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    # 绘制决策区域
    ax.contourf(xx, yy, Z, alpha=0.3, levels=[-2, 0, 2], 
               colors=['#4488ff', '#ff8844'])
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    
    # 绘制数据点
    pos_idx = y == 1
    neg_idx = y == -1
    ax.scatter(X[pos_idx, 0], X[pos_idx, 1], c='gold', s=50, 
              edgecolors='black', linewidths=1, label='+1', zorder=5)
    ax.scatter(X[neg_idx, 0], X[neg_idx, 1], c='navy', s=50, 
              edgecolors='white', linewidths=1, label='-1', zorder=5)
    
    # 绘制支持向量
    if len(svm.support_vector_idx) > 0:
        ax.scatter(X[svm.support_vector_idx, 0], X[svm.support_vector_idx, 1],
                  s=150, facecolors='none', edgecolors='red', 
                  linewidths=2, label='SV', zorder=6)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])


def demonstrate_margin():
    """演示最大间隔原理"""
    print("\n" + "="*60)
    print("📏 演示：最大间隔原理")
    print("="*60)
    
    # 生成线性可分数据
    np.random.seed(42)
    X_pos = np.random.randn(20, 2) + np.array([2, 2])
    X_neg = np.random.randn(20, 2) + np.array([-2, -2])
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(20), -np.ones(20)])
    
    # 使用不同的C值
    C_values = [0.01, 0.1, 1.0, 100.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, C in enumerate(C_values):
        print(f"\n  训练 C = {C}...")
        
        svm = SimplifiedSMO(X, y, C=C, kernel_type='linear', max_passes=50)
        svm.fit()
        
        ax = axes[i]
        
        # 计算权重向量 w
        w = np.sum((svm.alphas * y).reshape(-1, 1) * X, axis=0)
        margin = 2 / np.linalg.norm(w)
        
        # 绘制
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        xx = np.linspace(x_min, x_max, 100)
        
        # 决策边界和间隔边界
        if abs(w[1]) > 1e-10:
            yy_decision = -(w[0] * xx + svm.b) / w[1]
            yy_plus = -(w[0] * xx + svm.b - 1) / w[1]
            yy_minus = -(w[0] * xx + svm.b + 1) / w[1]
            
            ax.plot(xx, yy_decision, 'k-', linewidth=2, label='决策边界')
            ax.plot(xx, yy_plus, 'k--', linewidth=1, alpha=0.5, label='间隔边界')
            ax.plot(xx, yy_minus, 'k--', linewidth=1, alpha=0.5)
            ax.fill_between(xx, yy_minus, yy_plus, alpha=0.1, color='gray')
        
        # 数据点
        ax.scatter(X[:20, 0], X[:20, 1], c='gold', s=60, edgecolors='black', label='+1')
        ax.scatter(X[20:, 0], X[20:, 1], c='navy', s=60, edgecolors='black', label='-1')
        
        # 支持向量
        if len(svm.support_vector_idx) > 0:
            ax.scatter(X[svm.support_vector_idx, 0], X[svm.support_vector_idx, 1],
                      s=200, facecolors='none', edgecolors='red', linewidths=2)
        
        ax.set_title(f'C = {C}\n间隔宽度 = {margin:.3f}, 支持向量数 = {len(svm.support_vector_idx)}',
                    fontsize=12, fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('📊 C参数对间隔的影响', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('margin_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n结论：")
    print("  C越小 → 间隔越大 → 允许更多分类错误 → 模型更简单 → 可能欠拟合")
    print("  C越大 → 间隔越小 → 严格要求正确分类 → 模型更复杂 → 可能过拟合")


if __name__ == "__main__":
    # 运行所有演示
    compare_kernels()
    demonstrate_margin()
```

---

## 10.6 练习题

### 🌱 基础练习

**练习 10.1** 间隔计算

给定二维空间中的决策边界方程 $2x_1 + 3x_2 + 1 = 0$，计算：

1. 权重向量 $\mathbf{w}$ 的长度 $\|\mathbf{w}\|$
2. 间隔的宽度
3. 点 $(1, 1)$ 到决策边界的距离
4. 判断点 $(1, 1)$ 属于哪一侧

<details>
<summary>💡 提示</summary>

- 权重向量 $\mathbf{w} = (2, 3)$
- 间隔宽度 = $\frac{2}{\|\mathbf{w}\|}$
- 点到直线的距离 = $\frac{|\mathbf{w} \cdot \mathbf{x} + b|}{\|\mathbf{w}\|}$

</details>

---

**练习 10.2** 支持向量识别

给定以下数据点和已训练好的SVM：

| 样本 | 坐标 $(x_1, x_2)$ | 标签 $y$ | $\mathbf{w} \cdot \mathbf{x} + b$ |
|------|-------------------|---------|----------------------------------|
| A | (1, 2) | +1 | 1.5 |
| B | (2, 1) | +1 | 1.0 |
| C | (3, 3) | +1 | 2.5 |
| D | (-1, -1) | -1 | -1.0 |
| E | (-2, -3) | -1 | -2.0 |

1. 哪些样本是支持向量？
2. 间隔边界方程是什么？

---

**练习 10.3** 核函数计算

设 $\mathbf{x} = (1, 2)$，$\mathbf{x}' = (3, 1)$，计算：

1. 线性核 $K(\mathbf{x}, \mathbf{x}')$
2. 多项式核（$d=2$，$\gamma=1$，$r=0$）
3. RBF核（$\gamma=0.5$）

---

### 🌿 进阶练习

**练习 10.4** 软间隔SVM推导

考虑软间隔SVM的优化问题：

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n} \xi_i$$

约束：$y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i$，$\xi_i \geq 0$

1. 写出该问题的拉格朗日函数
2. 推导KKT条件
3. 解释互补松弛条件 $\alpha_i [y_i(\mathbf{w} \cdot \mathbf{x}_i + b) - 1 + \xi_i] = 0$ 的含义

---

**练习 10.5** SMO算法分析

SMO算法每次选择两个拉格朗日乘子 $\alpha_i$ 和 $\alpha_j$ 进行优化。

1. 为什么不能只优化一个 $\alpha$？
2. 在更新 $\alpha_j$ 时，为什么要裁剪到 $[L, H]$ 区间？
3. 解释启发式选择策略：为什么优先选择违反KKT条件的样本？

---

### 🌳 挑战练习

**练习 10.6** 实现多分类SVM

SVM本质上是二分类器。请实现一个**一对多（One-vs-Rest）**策略的多分类SVM：

1. 对于 $K$ 个类别，训练 $K$ 个二分类SVM
2. 每个SVM将一个类别与所有其他类别分开
3. 预测时，选择决策函数值最大的类别

用鸢尾花数据集（Iris）测试你的实现，并比较不同核函数的效果。

<details>
<summary>💡 提示</summary>

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 实现 OvR 策略...
```

</details>

---

## 10.7 本章总结

### 🎯 核心概念回顾

| 概念 | 解释 | 类比 |
|------|------|------|
| **超平面** | 决策边界，分隔两个类别 | 教室里的走道 |
| **间隔** | 最近的正负样本到超平面的距离之和 | 走道的宽度 |
| **支持向量** | 距离超平面最近的样本，决定模型 | 坐在走道边的同学 |
| **最大间隔** | 选择最宽走道的策略 | 让两边都有最大空间 |
| **软间隔** | 允许部分样本分类错误 | 容忍调皮的违规者 |
| **核技巧** | 隐式映射到高维空间 | 折叠纸张创造新维度 |
| **拉格朗日乘子** | 将约束优化转化为无约束优化 | 引入"监督员"检查约束 |
| **对偶问题** | 原问题的等价形式，更易求解 | 换一种角度看问题 |
| **SMO算法** | 每次优化两个变量的高效算法 | 一次只调整两块积木 |

### 📊 SVM的优势与局限

**优势：**
- ✅ 理论基础坚实，有全局最优解
- ✅ 泛化能力强，不容易过拟合（间隔最大化）
- ✅ 核技巧可以处理非线性问题
- ✅ 最终模型只依赖支持向量，存储高效

**局限：**
- ⚠️ 大规模数据集训练较慢（$O(n^2)$ 到 $O(n^3)$）
- ⚠️ 核函数和参数选择需要经验
- ⚠️ 对噪声敏感（特别是硬间隔）
- ⚠️ 不能直接输出概率

### 🔄 与其他算法的比较

| 特性 | SVM | 逻辑回归 | 决策树 | 神经网络 |
|------|-----|---------|--------|---------|
| 决策边界 | 光滑超平面 | 光滑超平面 | 轴对齐的矩形 | 任意复杂形状 |
| 训练速度 | 中等 | 快 | 快 | 慢 |
| 可解释性 | 中等 | 高 | 高 | 低 |
| 处理高维 | 优秀 | 需正则化 | 困难 | 优秀 |
| 非线性 | 核技巧 | 特征工程 | 天然支持 | 天然支持 |
| 概率输出 | 需额外处理 | 天然支持 | 天然支持 | 天然支持 |

### 🚀 延伸学习

1. **支持向量回归（SVR）**：将SVM扩展到回归问题
2. **核方法的其他应用**：核PCA、核K-means
3. **在线SVM**：处理流式数据的SVM变体
4. **深度学习与SVM**：用神经网络提取特征 + SVM分类

---

## 参考文献

Boser, B. E., Guyon, I. M., & Vapnik, V. N. (1992). A training algorithm for optimal margin classifiers. In *Proceedings of the 5th Annual Workshop on Computational Learning Theory* (pp. 144-152). ACM.

Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297. https://doi.org/10.1007/BF00994018

Cristianini, N., & Shawe-Taylor, J. (2000). *An Introduction to Support Vector Machines and Other Kernel-based Learning Methods*. Cambridge University Press.

Platt, J. C. (1998). Sequential minimal optimization: A fast algorithm for training support vector machines. *Microsoft Research Technical Report MSR-TR-98-14*.

Schölkopf, B., & Smola, A. J. (2002). *Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond*. MIT Press.

Vapnik, V. N. (1995). *The Nature of Statistical Learning Theory*. Springer-Verlag.

Vapnik, V. N. (1998). *Statistical Learning Theory*. Wiley.

Vapnik, V. N., & Chervonenkis, A. Y. (1964). A note on one class of perceptrons. *Automation and Remote Control*, 25(1).

---

> 🌻 **本章格言**：就像最宽的走道能让两个班级和谐共处，最大的间隔能让机器学习模型拥有最好的泛化能力。在数学的世界里，"留有余地"不仅是处世智慧，更是最优解的秘诀！

---

*本章完*
