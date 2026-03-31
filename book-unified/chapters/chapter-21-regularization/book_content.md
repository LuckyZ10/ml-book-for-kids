

## 21.4 L1正则化：稀疏性的魔法

### 21.4.1 历史背景：Lasso的诞生

**1996年** —— Robert Tibshirani在《Journal of the Royal Statistical Society》发表了划时代的论文《Regression shrinkage and selection via the lasso》，正式提出了**L1正则化**（Lasso回归）。

> Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B (Methodological)*, 58(1), 267-288.

**Lasso = Least Absolute Shrinkage and Selection Operator**

这个名字揭示了L1正则化的两大特性：
1. **Shrinkage（收缩）**：让权重变小
2. **Selection（选择）**：自动选择重要特征

### 21.4.2 L1正则化的数学定义

L1正则化惩罚权重的**绝对值之和**：

$$
\mathcal{L}_{L1} = \mathcal{L}_{data} + \lambda \sum_{i} |w_i|
$$

**与L2的关键区别**：
- L2: $\sum w_i^2$ → 平方惩罚
- L1: $\sum |w_i|$ → 绝对值惩罚

这个看似微小的差别，带来了完全不同的行为！

### 21.4.3 为什么L1会产生稀疏性？

**关键：L1在零点不可导**

看梯度：
$$
\frac{\partial |w|}{\partial w} = \begin{cases} 
+1 & \text{if } w > 0 \\
-1 & \text{if } w < 0 \\
\text{undefined} & \text{if } w = 0
\end{cases}
$$

**几何解释**：

```
L1 vs L2的几何差异：

L2惩罚等高线（圆形）        L1惩罚等高线（菱形）
    ╭────╮                    ╭╮
   ╱      ╲                  ╱  ╲
  │   ●    │                │ ●  │
  │  /|\   │                │/|\│
   ╲/ │ \ /                  ╲/ │ \/
    ○──┘                      ○──┘
     
   ● = 数据损失最小点          ● = 数据损失最小点
   ○ = 带正则化的最优点        ○ = 带正则化的最优点
   
L2：最优点在圆内，所有权重都小    L1：最优点在顶点上，某些权重为0
```

**稀疏性的好处**：
1. **特征选择**：自动找出最重要的特征
2. **模型简化**：减少需要存储的参数
3. **可解释性**：告诉我们哪些特征真的重要

### 21.4.4 L1 vs L2：何时用哪个？

根据Andrew Ng 2004年的经典分析：

> Ng, A. Y. (2004). Feature selection, L1 vs. L2 regularization, and rotational invariance. *ICML*, 78.

| 特性 | L1 (Lasso) | L2 (Ridge) |
|------|------------|------------|
| 惩罚形式 | $\sum \|w_i\|$ | $\sum w_i^2$ |
| 解的稀疏性 | ✅ 稀疏（很多0） | ❌ 不稀疏（都小） |
| 特征选择 | ✅ 自动选择 | ❌ 不选择 |
| 异常值敏感 | ⚠️ 较敏感 | ✅ 较鲁棒 |
| 优化难度 | ⚠️ 较难（不可导） | ✅ 容易（可导） |
| 计算效率 | ⚠️ 一般 | ✅ 高效 |

**使用建议**：
- **特征很多， suspected只有少部分重要** → 用L1
- **所有特征都可能有用** → 用L2
- **想要可解释的特征选择** → 用L1
- **追求最高预测准确率** → 通常L2更好

### 21.4.5 代码实现：L1正则化

```python
"""
L1正则化（Lasso）完整实现
包含软阈值求解和近似梯度下降
"""
import numpy as np


class L1RegularizedLinearRegression:
    """
    带L1正则化的线性回归（Lasso）
    
    公式: L = MSE + λ * |w|
    
    使用近似次梯度下降（实际应用中会使用坐标下降或 proximal gradient）
    """
    
    def __init__(self, learning_rate=0.01, lambda_l1=0.1, n_iterations=5000):
        self.lr = learning_rate
        self.lambda_l1 = lambda_l1  # L1正则化强度
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def soft_threshold(self, x, threshold):
        """
        软阈值算子（Soft Thresholding）
        用于L1的 proximal operator
        
        soft(x, λ) = sign(x) * max(|x| - λ, 0)
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def fit(self, X, y):
        """
        使用ISTA（Iterative Soft Thresholding Algorithm）训练
        """
        n_samples, n_features = X.shape
        
        # 初始化
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            # 前向传播
            y_pred = self.predict(X)
            
            # 计算损失
            mse_loss = np.mean((y_pred - y) ** 2)
            l1_penalty = self.lambda_l1 * np.sum(np.abs(self.weights))
            total_loss = mse_loss + l1_penalty
            self.loss_history.append(total_loss)
            
            # 计算梯度（MSE部分）
            error = y_pred - y
            dw_mse = (2 / n_samples) * X.T @ error
            db_mse = (2 / n_samples) * np.sum(error)
            
            # 梯度下降步骤
            weights_intermediate = self.weights - self.lr * dw_mse
            
            # 软阈值（L1的 proximal operator）
            self.weights = self.soft_threshold(weights_intermediate, self.lr * self.lambda_l1)
            
            # 偏置更新（L1不惩罚偏置）
            self.bias -= self.lr * db_mse
            
            # 每500轮打印
            if (i + 1) % 500 == 0:
                non_zero = np.sum(np.abs(self.weights) > 1e-5)
                print(f"Iter {i+1}: MSE={mse_loss:.4f}, L1={l1_penalty:.4f}, Non-zero weights: {non_zero}/{len(self.weights)}")
    
    def predict(self, X):
        """预测"""
        return X @ self.weights + self.bias
    
    def get_sparsity(self, threshold=1e-5):
        """获取权重稀疏度（0的比例）"""
        return np.mean(np.abs(self.weights) < threshold)


def demonstrate_l1_vs_l2():
    """对比L1和L2的效果差异"""
    
    # 生成稀疏数据：y = 3*x1 + 0*x2 + 0*x3 + 0*x4 + 1 + 噪声
    # 只有x1是真正重要的
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([3, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 只有第一个特征重要
    y = X @ true_weights + 1 + np.random.normal(0, 0.5, n_samples)
    
    print("=" * 70)
    print("L1 vs L2 对比实验")
    print("=" * 70)
    print(f"真实权重: {true_weights}")
    print(f"生成数据: y = 3*x1 + 0*x2 + ... + 1 + noise\n")
    
    # L1 (Lasso)
    print("\n【L1正则化 (Lasso)】")
    print("-" * 50)
    model_l1 = L1RegularizedLinearRegression(learning_rate=0.01, lambda_l1=0.5, n_iterations=5000)
    model_l1.fit(X, y)
    print(f"\n学习到的权重: {model_l1.weights.round(3)}")
    print(f"稀疏度: {model_l1.get_sparsity()*100:.1f}% (即{model_l1.get_sparsity()*10:.0f}/10个权重≈0)")
    
    # L2 (Ridge)
    print("\n【L2正则化 (Ridge)】")
    print("-" * 50)
    model_l2 = L2RegularizedLinearRegression(learning_rate=0.01, lambda_l2=0.5, n_iterations=5000)
    model_l2.fit(X, y)
    print(f"\n学习到的权重: {model_l2.weights.round(3)}")
    print(f"稀疏度: {model_l2.get_sparsity()*100:.1f}%")
    print(f"权重L2范数: {model_l2.get_weights_norm():.4f}")


# 运行对比
if __name__ == "__main__":
    demonstrate_l1_vs_l2()
```

---

## 21.5 Dropout：随机的力量

### 21.5.1 历史背景：Hinton的"顿悟"

**2012年** —— Geoffrey Hinton在授课时突然想到一个想法："为什么不让神经网络在训练时'忘记'一些神经元呢？"

他回忆起以前在银行工作时的防欺诈机制：银行员工不能总是处理同一种欺诈，以防有人贿赂他们。

这个类比启发了**Dropout**：让每个神经元不能总是依赖其他特定的神经元。

**2014年** —— Nitish Srivastava、Hinton等人正式发表Dropout论文：

> Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1), 1929-1958.

### 21.5.2 Dropout的核心思想

**训练阶段**：每次前向传播时，**随机**将一部分神经元"关掉"（输出设为0）

```
正常网络：                 应用Dropout后的网络（随机的）：
  ┌───┐                    ┌───┐
  │ x │                    │ x │
  └─┬─┘                    └─┬─┘
    │                        │
  ┌─┴─┐                    ┌─┴─┐
  │ ● │ ← 神经元1          │ ● │ ← 保留（概率p）
  └───┘                    └───┘
    │                        │
  ┌─┴─┐                    ┌───┐
  │ ● │ ← 神经元2          │ X │ ← 丢弃（概率1-p）
  └───┘                    └───┘
    │                        │
  ┌─┴─┐                    ┌─┴─┐
  │ ● │ ← 神经元3          │ ● │ ← 保留
  └───┘                    └───┘
    │                        │
  ┌─┴─┐                    ┌─┴─┐
  │ y │                    │ y │
  └───┘                    └───┘
```

**为什么这样有用？**

1. **防止共适应（Co-adaptation）**：神经元不能依赖特定的其他神经元，必须学会独立工作
2. **集成效应**：相当于同时训练很多"子网络"，测试时相当于这些子网络的平均
3. **强迫冗余**：网络必须学会多种表示方式，而不是依赖单一路径

### 21.5.3 Dropout的数学原理

**训练时的前向传播**：

对于第$l$层的输出 $\mathbf{h}^{(l)}$：

$$
\mathbf{r}^{(l)} \sim \text{Bernoulli}(p)
$$

$$
\tilde{\mathbf{h}}^{(l)} = \mathbf{r}^{(l)} \odot \mathbf{h}^{(l)}
$$

$$
\mathbf{h}^{(l+1)} = f(\tilde{\mathbf{h}}^{(l)})
$$

其中：
- $p$：保留概率（通常0.5）
- $\mathbf{r}^{(l)}$：随机掩码向量（每个元素以概率p为1）
- $\odot$：逐元素乘法

**关键问题：测试时怎么办？**

如果测试时只用完整网络，输出期望值会是训练时的$p$倍！

**解决方案（Inverted Dropout）**：

训练时将被保留的神经元输出**除以p**：

$$
\tilde{\mathbf{h}}^{(l)} = \frac{\mathbf{r}^{(l)} \odot \mathbf{h}^{(l)}}{p}
$$

这样测试时可以直接使用完整网络，无需任何修改！

**期望值的验证**：

$$
E[\tilde{h}_i] = E\left[\frac{r_i \cdot h_i}{p}\right] = \frac{h_i}{p} \cdot E[r_i] = \frac{h_i}{p} \cdot p = h_i
$$

### 21.5.4 为什么Dropout是一种正则化？

**类比理解**：

想象一个团队在做一个项目：
- **正常训练**：每个人都知道其他人会做什么，可能过度依赖某个"专家"
- **Dropout训练**：每次开会随机"请假"一些人，其他人必须学会独立完成各种任务
- **结果**：每个人能力更全面，团队更鲁棒

**与L2正则化的区别**：
- **L2**：限制权重大小
- **Dropout**：限制神经元之间的依赖关系

**实际效果**：
- Dropout通常比L2更强，尤其适合深层网络
- 可以将训练/验证误差差距缩小
- 现代深度学习几乎必备

### 21.5.5 代码实现：Dropout层

```python
"""
Dropout层完整实现
包含前向/反向传播、Inverted Dropout、以及可视化
"""
import numpy as np


class Dropout:
    """
    Dropout层实现（Inverted Dropout版本）
    
    参数:
        p: 保留概率（默认0.5）
    """
    
    def __init__(self, p=0.5):
        self.p = p  # 保留概率
        self.mask = None  # 用于反向传播
        self.training = True  # 训练/测试模式
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入，shape (batch_size, features)
        返回:
            输出（应用dropout）
        """
        if self.training:
            # 训练阶段：生成随机掩码并应用
            self.mask = (np.random.rand(*x.shape) < self.p).astype(float)
            # Inverted Dropout：除以p保持期望值
            return x * self.mask / self.p
        else:
            # 测试阶段：直接返回输入
            return x
    
    def backward(self, dout):
        """
        反向传播
        
        参数:
            dout: 上游梯度
        返回:
            对输入的梯度
        """
        if self.training:
            # 只有被保留的神经元才有梯度
            return dout * self.mask / self.p
        else:
            return dout
    
    def train(self, mode=True):
        """设置训练/测试模式"""
        self.training = mode
    
    def eval(self):
        """设置评估模式"""
        self.training = False


class DropoutNetwork:
    """
    带Dropout的简单神经网络演示
    
    结构: Input -> Linear -> ReLU -> Dropout -> Linear -> Output
    """
    
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.5):
        # 初始化权重
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)
        
        # Dropout层
        self.dropout = Dropout(p=dropout_p)
        
        # 缓存
        self.cache = {}
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X, training=True):
        """前向传播"""
        self.dropout.train(training)
        
        # 第一层
        z1 = X @ self.W1 + self.b1
        h1 = self.relu(z1)
        
        # Dropout
        h1_dropped = self.dropout.forward(h1)
        
        # 第二层
        z2 = h1_dropped @ self.W2 + self.b2
        
        # 缓存用于反向传播
        self.cache = {'X': X, 'z1': z1, 'h1': h1, 'h1_dropped': h1_dropped, 'z2': z2}
        
        return z2
    
    def backward(self, dout):
        """反向传播"""
        cache = self.cache
        
        # 第二层梯度
        dW2 = cache['h1_dropped'].T @ dout
        db2 = np.sum(dout, axis=0)
        
        dh1_dropped = dout @ self.W2.T
        
        # Dropout梯度
        dh1 = self.dropout.backward(dh1_dropped)
        
        # ReLU梯度
        dz1 = dh1 * self.relu_derivative(cache['z1'])
        
        # 第一层梯度
        dW1 = cache['X'].T @ dz1
        db1 = np.sum(dz1, axis=0)
        
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}


def visualize_dropout_effect():
    """可视化Dropout的效果"""
    
    print("=" * 60)
    print("Dropout可视化演示")
    print("=" * 60)
    
    # 创建简单的输入
    np.random.seed(42)
    x = np.ones((1, 10))  # 10个神经元，全部为1
    
    print(f"\n原始输入: {x[0]}")
    print(f"神经元数量: {len(x[0])}")
    
    # 应用不同概率的dropout
    for p in [1.0, 0.7, 0.5, 0.3]:
        dropout = Dropout(p=p)
        dropout.train(True)
        
        # 多次采样看统计效果
        samples = []
        for _ in range(100):
            out = dropout.forward(x.copy())
            samples.append(out[0])
        
        samples = np.array(samples)
        mean_output = np.mean(samples, axis=0)
        survival_rate = np.mean(samples > 0, axis=0)
        
        print(f"\n【保留概率 p = {p}】")
        print(f"  平均输出: {mean_output.round(2)}")
        print(f"  期望输出: 1.0 (理论值)")
        print(f"  实际均值: {np.mean(mean_output):.3f}")
        print(f"  存活率: {np.mean(survival_rate)*100:.0f}%")


# 运行可视化
if __name__ == "__main__":
    visualize_dropout_effect()
```

---

*第21章持续创作中，已完成：*
- ✅ 开篇故事与过拟合概念  
- ✅ L2正则化理论与代码
- ✅ L1正则化（Lasso）与对比
- ✅ Dropout完整实现
- 🔄 接下来：Batch Normalization、Early Stopping、完整实验
