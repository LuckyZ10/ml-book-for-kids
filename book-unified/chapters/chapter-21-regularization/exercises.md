## 21.10 练习题

### 基础概念题

**问题1：过拟合与欠拟合**

请解释以下概念，并各举一个实际例子：
- 过拟合（Overfitting）
- 欠拟合（Underfitting）
- 刚好拟合（Just Right Fitting）

**提示**：可以结合多项式拟合的角度思考。

<details>
<summary>参考答案</summary>

**过拟合**：模型过于复杂，记住了训练数据的噪声和细节，泛化能力差。
- 例子：用100次多项式拟合10个数据点，曲线剧烈抖动。

**欠拟合**：模型过于简单，无法捕捉数据的基本规律。
- 例子：用直线拟合明显呈抛物线分布的数据。

**刚好拟合**：模型复杂度适中，能够捕捉数据规律而不受噪声干扰。
- 例子：用2次多项式拟合抛物线分布的数据。

</details>

---

**问题2：L1 vs L2正则化**

比较L1和L2正则化的以下方面：
1. 惩罚项的形式
2. 产生的解的特性（稀疏性）
3. 优化方法的区别
4. 各自适合的应用场景

<details>
<summary>参考答案</summary>

| 方面 | L1正则化 | L2正则化 |
|------|----------|----------|
| 惩罚项 | $\lambda \sum \|\theta_j\|$ | $\lambda \sum \theta_j^2$ |
| 解的特性 | 稀疏，许多权重为零 | 稠密，权重小而均匀 |
| 优化方法 | 次梯度下降 | 标准梯度下降 |
| 适用场景 | 特征选择、高维数据 | 一般情况、稳定性要求高 |

</details>

---

**问题3：Dropout的工作原理**

请回答：
1. Dropout在训练时和测试时的行为有何不同？
2. 为什么Dropout能够防止过拟合？
3. 如果Dropout率为0.5，训练时某个神经元的输出期望是多少？测试时如何处理？

<details>
<summary>参考答案</summary>

1. **训练时**：以概率 $p$ 随机丢弃神经元（输出置零），保留的神经元的输出缩放为原来的 $1/p$ 倍。
   
   **测试时**：使用所有神经元，不丢弃任何神经元，也不进行缩放。

2. **防止过拟合的原因**：
   - 相当于训练了多个子网络的集成
   - 打破神经元之间的共适应
   - 添加随机噪声，增强鲁棒性

3. 训练时期望输出是 $0.5 \times \text{原输出} \times 2 = \text{原输出}$（考虑缩放），测试时直接使用原输出。

</details>

---

### 进阶推导题

**问题4：L2正则化的闭式解**

对于线性回归问题，原始损失函数为：
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$$

加入L2正则化后：
$$J_{L2}(\theta) = J(\theta) + \frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2$$

请推导Ridge回归的闭式解：
$$\theta = (X^T X + \lambda I)^{-1} X^T y$$

<details>
<summary>参考答案</summary>

**步骤1：写出矩阵形式的损失函数**

$$J_{L2} = \frac{1}{2m}(X\theta - y)^T(X\theta - y) + \frac{\lambda}{2m}\theta^T\theta$$

**步骤2：对 $\theta$ 求导**

$$\frac{\partial J_{L2}}{\partial \theta} = \frac{1}{m}X^T(X\theta - y) + \frac{\lambda}{m}\theta$$

**步骤3：令导数为零**

$$X^T(X\theta - y) + \lambda\theta = 0$$

$$X^TX\theta - X^Ty + \lambda\theta = 0$$

$$(X^TX + \lambda I)\theta = X^Ty$$

**步骤4：求解**

$$\theta = (X^TX + \lambda I)^{-1}X^Ty$$

**证毕。**

</details>

---

**问题5：Batch Normalization的梯度推导**

给定Batch Normalization的前向传播：
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

其中 $\mu_B$ 和 $\sigma_B^2$ 是批量均值和方差。

请推导反向传播中 $\frac{\partial L}{\partial x_i}$ 的表达式。

<details>
<summary>参考答案</summary>

设 $L$ 是损失函数，我们需要计算 $\frac{\partial L}{\partial x_i}$。

**步骤1：关于 $\gamma$ 和 $\beta$ 的梯度**

$$\frac{\partial L}{\partial \gamma} = \sum_i \frac{\partial L}{\partial y_i} \hat{x}_i$$

$$\frac{\partial L}{\partial \beta} = \sum_i \frac{\partial L}{\partial y_i}$$

**步骤2：关于 $\hat{x}_i$ 的梯度**

$$\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma$$

**步骤3：关于方差的梯度**

设 $\sigma^2 = \sigma_B^2 + \epsilon$

$$\frac{\partial L}{\partial \sigma^2} = \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot (x_i - \mu_B) \cdot (-\frac{1}{2})(\sigma^2)^{-3/2}$$

**步骤4：关于均值的梯度**

$$\frac{\partial L}{\partial \mu_B} = \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot (-\frac{1}{\sqrt{\sigma^2}}) + \frac{\partial L}{\partial \sigma^2} \cdot \frac{\sum_i -2(x_i - \mu_B)}{m}$$

**步骤5：关于 $x_i$ 的梯度**

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma^2}} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{2(x_i - \mu_B)}{m} + \frac{\partial L}{\partial \mu_B} \cdot \frac{1}{m}$$

**简化形式**：

$$\frac{\partial L}{\partial x_i} = \frac{\gamma}{\sqrt{\sigma_B^2 + \epsilon}} \left( \frac{\partial L}{\partial y_i} - \frac{1}{m}\sum_j \frac{\partial L}{\partial y_j} - \hat{x}_i \cdot \frac{1}{m}\sum_j \frac{\partial L}{\partial y_j} \hat{x}_j \right)$$

</details>

---

**问题6：L1正则化产生稀疏性的数学证明**

考虑一维的Lasso问题：
$$\min_\theta \frac{1}{2}(y - \theta)^2 + \lambda |\theta|$$

请证明：当 $|y| \leq \lambda$ 时，最优解是 $\theta^* = 0$。

<details>
<summary>参考答案</summary>

**步骤1：写出损失函数**

$$f(\theta) = \frac{1}{2}(y - \theta)^2 + \lambda |\theta|$$

**步骤2：考虑 $\theta > 0$ 的情况**

$$f(\theta) = \frac{1}{2}(y - \theta)^2 + \lambda \theta$$

$$f'(\theta) = -(y - \theta) + \lambda = \theta - y + \lambda$$

令 $f'(\theta) = 0$：$\theta = y - \lambda$

若 $y > \lambda$，则 $\theta^* = y - \lambda > 0$

若 $y \leq \lambda$，则 $\theta^* = y - \lambda \leq 0$，与假设矛盾，最小值在边界 $\theta = 0$ 处。

**步骤3：考虑 $\theta < 0$ 的情况**

$$f(\theta) = \frac{1}{2}(y - \theta)^2 - \lambda \theta$$

$$f'(\theta) = -(y - \theta) - \lambda = \theta - y - \lambda$$

令 $f'(\theta) = 0$：$\theta = y + \lambda$

若 $y < -\lambda$，则 $\theta^* = y + \lambda < 0$

若 $y \geq -\lambda$，则 $\theta^* = y + \lambda \geq 0$，与假设矛盾，最小值在边界 $\theta = 0$ 处。

**步骤4：综合**

当 $|y| \leq \lambda$ 时：
- $\theta > 0$ 的最优解不满足条件，边界值为 $f(0) = \frac{1}{2}y^2$
- $\theta < 0$ 的最优解不满足条件，边界值为 $f(0) = \frac{1}{2}y^2$

因此，当 $|y| \leq \lambda$ 时，$\theta^* = 0$。

**证毕。**

这证明了L1正则化的**软阈值**特性：小信号被压缩为零，大信号被收缩 $\lambda$。

</details>

---

### 挑战编程题

**问题7：实现带正则化的神经网络**

请实现一个单隐藏层的神经网络类 `RegularizedNN`，支持以下功能：
- L1/L2/Elastic Net正则化
- Dropout
- Batch Normalization（可选）
- Early Stopping

在MNIST或合成数据集上测试不同正则化组合的效果。

**要求**：
1. 从零实现，不使用PyTorch/TensorFlow的高级API
2. 对比不同正则化策略的验证集准确率
3. 可视化训练曲线和权重分布

<details>
<summary>提示与框架</summary>

```python
class RegularizedNN:
    def __init__(self, input_size, hidden_size, output_size,
                 regularizer=None, dropout_rate=0.0, use_batchnorm=False):
        # 初始化权重和偏置
        # 初始化正则化器
        # 初始化Dropout层
        # 初始化BatchNorm层
        pass
    
    def forward(self, X, training=True):
        # 前向传播
        # 应用Dropout（训练时）
        # 应用BatchNorm
        pass
    
    def backward(self, X, y, learning_rate):
        # 反向传播
        # 计算梯度
        # 添加正则化梯度
        # 更新参数
        pass
    
    def train_with_early_stopping(self, X_train, y_train, X_val, y_val,
                                   max_epochs, patience, ...):
        # 实现早停逻辑
        pass
```

</details>

---

**问题8：自适应正则化强度**

设计一个自适应调整正则化强度的算法：

1. 初始设置较大的 $\lambda$ 值
2. 监控训练误差和验证误差
3. 如果训练误差下降缓慢但验证误差上升，增大 $\lambda$
4. 如果训练误差和验证误差都高，减小 $\lambda$

在多项式拟合任务上测试你的自适应算法，与固定 $\lambda$ 进行比较。

**评价指标**：
- 最终测试误差
- 收敛速度（达到目标误差所需的epoch数）
- 超参数调节的难易程度

---