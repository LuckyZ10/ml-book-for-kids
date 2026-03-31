# 第二十一章：正则化——防止网络"死记硬背"

**"优秀的学生不是背答案，而是理解原理。"**  
*——本章格言*

---

## 开篇故事：小明的故事

小明是个刻苦的学生。期末考试前，他把教材上所有的例题都背了下来——每个数字、每个步骤，甚至老师的板书都记得一清二楚。考试那天，他信心满满。

结果成绩出来：**58分**。

为什么？因为考试题目稍微变了一下数字，他就完全不会了。他把例题"背"下来了，但没有真正"理解"解题方法。

神经网络的训练也是如此。当一个网络"记住"了训练数据的所有细节（包括噪声），却在测试数据上表现糟糕时，我们称之为**过拟合（Overfitting）**。这就像小明死记硬背例题，却没有掌握真正的解题思路。

本章要学习的**正则化（Regularization）**，就是教网络"真正理解"而不是"死记硬背"的技术。

---

## 21.1 过拟合：神经网络的"死记硬背"

### 21.1.1 什么是过拟合？

想象你在学习识别猫咪：
- **训练阶段**：你看了一百张猫的照片，记住了每只猫的毛色、花纹、耳朵形状
- **测试阶段**：你看到一只你没见过的猫——它是白色的，而你之前看的猫都是橘色的

如果你"死记硬背"了训练集中的每只猫，你可能会说："这不是猫，因为它是白色的！" 但如果你真正理解了"猫"的概念，你就会认出它。

**过拟合的本质**：模型在训练数据上表现太好，以至于把数据中的噪声也当成了规律来学习。

### 21.1.2 过拟合的数学表现

让我们用多项式拟合来直观理解过拟合：

```
假设真实关系：y = 2x + 1（一条直线）
我们采集了10个带噪声的数据点
```

**欠拟合（Underfitting）**：用直线拟合 → 太简单，无法捕捉规律  
**正常拟合**：用二次曲线拟合 → 刚好捕捉真实规律  
**过拟合（Overfitting）**：用9次多项式拟合 → 穿过所有训练点，但波动剧烈

```
训练误差：过拟合模型 ≈ 0（完美！）
测试误差：过拟合模型 >> 正常模型（糟糕！）
```

### 21.1.3 过拟合的可视化

```
数据点分布：
    ●                    ●
      ●    ●        ●
         ●    ●  ●
    ────────────────→ x
    
欠拟合（直线）：          正常拟合（曲线）：         过拟合（复杂曲线）：
    ●                    ●    ╭─╮               ●  ╭╮    ╭╮  ●
    │●    ●        ●    │●  ╭╯ ╰╮        ●    │●╭╯╰╮  ╭╯╰╮●│
────┼───┼───┼────┼──  ──┼─╭╯   ╰╮──┼──  ────┼╯    ╰╮╭╯    ╰┼
    │   │   │        ●  │●╯     ╰●│●       ●╯      ╰╯      ╰●
    
    太简单，漏掉规律      恰到好处               太复杂，记住噪声
```

**检测过拟合的方法**：
- 训练误差持续下降，但验证误差开始上升 → 过拟合信号！
- 训练准确率很高，但测试准确率很低 → 明显过拟合！

---

## 21.2 正则化的核心思想

### 21.2.1 从"死记硬背"到"素质教育"

**正则化的哲学**：惩罚复杂度，奖励简洁。

就像学校考试：
- **死记硬背**：背下所有例题的答案 → 考试一变通就挂
- **素质教育**：理解解题思路和方法 → 题目再变也不怕

神经网络的正则化，就是防止网络"记住"太多细节，强迫它学习"更通用"的模式。

### 21.2.2 奥卡姆剃刀原则

> "如无必要，勿增实体。" —— 威廉·奥卡姆，14世纪

**奥卡姆剃刀在机器学习中的含义**：
- 如果两个模型效果相近，选更简单的那个
- 简单的模型更可能泛化得好（在未知数据上表现好）

**为什么简单模型更好？**
1. **更少的参数** → 更不容易记住噪声
2. **更平滑的函数** → 对输入的小变化更鲁棒
3. **更可解释** → 我们更容易理解它在做什么

### 21.2.3 正则化的通用框架

所有正则化技术的核心思想都是在**损失函数**中加入一个**惩罚项**：

$$
\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \cdot \mathcal{L}_{regularization}
$$

其中：
- $\mathcal{L}_{data}$：数据损失（比如MSE、交叉熵）
- $\mathcal{L}_{regularization}$：正则化惩罚（衡量模型复杂度）
- $\lambda$：正则化强度（超参数）

**直观理解**：
- 数据损失鼓励模型拟合训练数据
- 正则化惩罚惩罚模型复杂度
- 两者平衡，得到既拟合数据又不过于复杂的模型

---

## 21.3 L2正则化：权重衰减

### 21.3.1 历史背景：从Tikhonov到Ridge

**1943年** —— 安德烈·Tikhonov在苏联科学院发表了关于"不适定问题"的研究，提出了**Tikhonov正则化**（后来被称为L2正则化）。

**1970年** —— Arthur Hoerl和Robert Kennard在《Technometrics》期刊发表了**Ridge回归**论文，将L2正则化应用于线性回归。

> Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67.

### 21.3.2 L2正则化的数学定义

L2正则化惩罚权重的**平方和**：

$$
\mathcal{L}_{L2} = \mathcal{L}_{data} + \lambda \sum_{i} w_i^2
$$

展开写法：
$$
\mathcal{L}_{L2} = \mathcal{L}_{data} + \lambda (w_1^2 + w_2^2 + ... + w_n^2)
$$

**为什么叫"权重衰减（Weight Decay）"？**

看梯度下降的更新规则：

$$
\frac{\partial \mathcal{L}_{L2}}{\partial w_i} = \frac{\partial \mathcal{L}_{data}}{\partial w_i} + 2\lambda w_i
$$

权重更新：
$$
w_i^{new} = w_i^{old} - \eta \frac{\partial \mathcal{L}_{L2}}{\partial w_i} = w_i^{old} - \eta \frac{\partial \mathcal{L}_{data}}{\partial w_i} - 2\eta\lambda w_i^{old}
$$

整理后：
$$
w_i^{new} = (1 - 2\eta\lambda) w_i^{old} - \eta \frac{\partial \mathcal{L}_{data}}{\partial w_i}
$$

**关键洞察**：每次更新，权重先乘以一个小于1的因子 $(1 - 2\eta\lambda)$，这就是"衰减"！

### 21.3.3 几何解释

```
L2正则化的几何意义：

权重空间俯视图：
    w2
    ↑
    │    ╭──────╮
    │   ╱   ●    ╲      ● = 最优解（无正则化）
    │  │  /│\    │     ○ = 带L2正则化的最优解
    │  │ / │ \   │      圆圈 = L2惩罚等高线（圆形）
    │  ╰/  │  \  ╯           = 数据损失等高线（椭圆形）
    │  ○   │
    └──────┼────────→ w1
           │
    
L2正则化将最优解"拉向"原点，使权重变小但不为零
```

**为什么权重变小能防止过拟合？**
- 大权重意味着网络对某个特征极度敏感
- 小权重意味着网络综合考虑多个特征
- 极端大权重往往是记住了训练数据的噪声

### 21.3.4 代码实现：L2正则化

```python
"""
L2正则化（权重衰减）完整实现
包含手动实现和PyTorch自动实现
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ========== NumPy手动实现 ==========

class L2RegularizedLinearRegression:
    """
    带L2正则化的线性回归（Ridge回归）
    
    公式: L = MSE + λ * ||w||²
    """
    
    def __init__(self, learning_rate=0.01, lambda_l2=0.1, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_l2 = lambda_l2  # L2正则化强度
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def fit(self, X, y):
        """
        训练模型
        
        参数:
            X: shape (n_samples, n_features)
            y: shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            # 前向传播
            y_pred = self.predict(X)
            
            # 计算MSE损失
            mse_loss = np.mean((y_pred - y) ** 2)
            
            # 计算L2惩罚（注意：通常不惩罚偏置项）
            l2_penalty = self.lambda_l2 * np.sum(self.weights ** 2)
            
            # 总损失
            total_loss = mse_loss + l2_penalty
            self.loss_history.append(total_loss)
            
            # 反向传播 - 计算梯度
            # MSE的梯度
            dw_mse = (2 / n_samples) * X.T @ (y_pred - y)
            db_mse = (2 / n_samples) * np.sum(y_pred - y)
            
            # L2正则化的梯度: d/dw(λw²) = 2λw
            dw_l2 = 2 * self.lambda_l2 * self.weights
            
            # 总梯度
            dw = dw_mse + dw_l2
            db = db_mse  # 偏置不添加L2惩罚
            
            # 参数更新（权重衰减在这里体现）
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # 每100轮打印损失
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}: MSE={mse_loss:.4f}, L2={l2_penalty:.4f}, Total={total_loss:.4f}")
    
    def predict(self, X):
        """预测"""
        return X @ self.weights + self.bias
    
    def get_weights_norm(self):
        """获取权重的L2范数"""
        return np.sqrt(np.sum(self.weights ** 2))


# ========== 演示：L2正则化的效果 ==========

def demonstrate_l2_effect():
    """演示不同L2强度对模型的影响"""
    
    # 生成数据：y = 2x + 1 + 噪声
    np.random.seed(42)
    n_samples = 50
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y_true = 2 * X.flatten() + 1
    y = y_true + np.random.normal(0, 2, n_samples)  # 添加噪声
    
    # 划分为训练集和测试集
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 测试不同的L2强度
    lambda_values = [0, 0.01, 0.1, 1.0, 10.0]
    
    results = []
    
    print("=" * 60)
    print("L2正则化效果对比实验")
    print("=" * 60)
    
    for lambda_l2 in lambda_values:
        print(f"\n【λ = {lambda_l2}】")
        print("-" * 40)
        
        model = L2RegularizedLinearRegression(
            learning_rate=0.01,
            lambda_l2=lambda_l2,
            n_iterations=1000
        )
        
        model.fit(X_train, y_train)
        
        # 评估
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mse = np.mean((train_pred - y_train) ** 2)
        test_mse = np.mean((test_pred - y_test) ** 2)
        weight_norm = model.get_weights_norm()
        
        print(f"权重: w={model.weights[0]:.4f}, b={model.bias:.4f}")
        print(f"权重L2范数: {weight_norm:.4f}")
        print(f"训练MSE: {train_mse:.4f}")
        print(f"测试MSE: {test_mse:.4f}")
        print(f"泛化差距: {abs(train_mse - test_mse):.4f}")
        
        results.append({
            'lambda': lambda_l2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'weight_norm': weight_norm,
            'weights': model.weights.copy(),
            'bias': model.bias
        })
    
    return results, X, y, X_train, y_train, X_test, y_test


# 运行演示
if __name__ == "__main__":
    results, X, y, X_train, y_train, X_test, y_test = demonstrate_l2_effect()
```

---

（继续下一部分...）

*第21章正在创作中，已包含：*
- ✅ 开篇故事与过拟合概念
- ✅ L2正则化理论基础与代码
- 🔄 接下来：L1正则化、Dropout、BatchNorm、Early Stopping、完整实验
