## 18.7 练习题

### 基础练习（3题）

**练习18.1：链式法则基础**

给定函数 $f(x) = (2x^2 + 3)^3$，使用链式法则计算 $f'(x)$。

<details>
<summary>提示</summary>
令 $u = 2x^2 + 3$，则 $f = u^3$。先计算 $\frac{df}{du}$ 和 $\frac{du}{dx}$。
</details>

**练习18.2：神经网络前向传播**

考虑一个简单的神经网络：
- 输入: $x = 0.5$
- 权重: $w_1 = 0.3, w_2 = 0.7$
- 偏置: $b_1 = 0.1, b_2 = 0.2$
- 隐藏层激活: sigmoid
- 输出激活: 线性

网络结构: $x \rightarrow h = \sigma(w_1 x + b_1) \rightarrow y = w_2 h + b_2$

计算输出 $y$。

**练习18.3：单步反向传播**

使用练习18.2的网络，假设目标输出是 $y_{target} = 1.0$，损失函数 $L = \frac{1}{2}(y - y_{target})^2$。

计算：
1. $\frac{\partial L}{\partial y}$
2. $\frac{\partial L}{\partial w_2}$
3. $\frac{\partial L}{\partial w_1}$

### 进阶练习（3题）

**练习18.4：实现单个神经元**

用Python实现一个单个神经元类，包含：
- `forward()` 方法进行前向传播
- `backward()` 方法计算梯度
- `update()` 方法更新权重

使用sigmoid激活函数和MSE损失。

```python
class Neuron:
    def __init__(self, input_size):
        # 初始化权重和偏置
        pass
    
    def forward(self, X):
        # 实现前向传播
        pass
    
    def backward(self, X, y_true, y_pred):
        # 实现反向传播，返回梯度
        pass
    
    def update(self, gradients, learning_rate):
        # 更新参数
        pass
```

**练习18.5：矩阵求导验证**

给定 $\mathbf{W} \in \mathbb{R}^{3 \times 2}$，$\mathbf{x} \in \mathbb{R}^{2 \times 1}$，$\mathbf{b} \in \mathbb{R}^{3 \times 1}$，令 $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$。

证明：$\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \mathbf{W}^T$（在适当布局约定下）。

**练习18.6：激活函数对比**

比较sigmoid和ReLU激活函数：
1. 绘制它们的函数曲线
2. 绘制它们的导数曲线
3. 解释为什么ReLU有助于缓解梯度消失问题

使用以下代码框架：

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

# Sigmoid
sigmoid = 1 / (1 + np.exp(-x))
sigmoid_deriv = sigmoid * (1 - sigmoid)

# ReLU
relu = np.maximum(0, x)
relu_deriv = (x > 0).astype(float)

# 绘制对比图
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# ... 完成绘图代码
```

### 挑战练习（2题）

**练习18.7：实现批归一化层**

批归一化（Batch Normalization）是缓解梯度消失的重要技术。实现一个带批归一化的全连接层：

```python
class BatchNormLayer:
    def __init__(self, num_features):
        self.gamma = np.ones((num_features, 1))
        self.beta = np.zeros((num_features, 1))
        self.epsilon = 1e-8
    
    def forward(self, X, training=True):
        """
        X: (num_features, batch_size)
        返回归一化后的数据
        """
        if training:
            self.mean = np.mean(X, axis=1, keepdims=True)
            self.var = np.var(X, axis=1, keepdims=True)
            self.X_norm = (X - self.mean) / np.sqrt(self.var + self.epsilon)
            self.X_hat = self.gamma * self.X_norm + self.beta
            return self.X_hat
        else:
            # 使用训练时的统计量
            pass
    
    def backward(self, dY):
        """计算对gamma、beta和输入X的梯度"""
        pass
```

**练习18.8：手写数字识别**

使用本章实现的MLP类，在手写数字数据集（MNIST或简化版）上进行分类：

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 加载数据
digits = load_digits()
X = digits.data.T  # 转置为 (features, samples)
y = digits.target

# 转换为one-hot编码
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1)).T

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X.T, y_onehot.T, test_size=0.2, random_state=42)

