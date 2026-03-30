## 7.10 练习题

### 📝 基础练习

**7.1** 用最小二乘法公式计算以下数据的回归直线：
```
X = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
```
要求：
- 计算 $\bar{x}$ 和 $\bar{y}$
- 计算斜率 $w$ 和截距 $b$
- 写出回归方程
- 计算 $R^2$

**7.2** 解释为什么最小二乘法使用"平方"而不是"绝对值"。

**7.3** 已知某线性回归模型的 $R^2 = 0.85$，这意味着什么？如果 $R^2 = 0$ 呢？

### 🔬 进阶挑战

**7.4** 正规方程推导  
从损失函数 $J(\boldsymbol{\theta}) = (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$ 出发，
证明 $\boldsymbol{\theta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ 是最优解。

**7.5** 相关系数与斜率的关系  
证明在单变量线性回归中：
$$
w = r \cdot \frac{\sigma_y}{\sigma_x}
$$
其中 $r$ 是相关系数，$\sigma_x$ 和 $\sigma_y$ 分别是 $X$ 和 $Y$ 的标准差。

**7.6** 🏆 挑战题：岭回归（Ridge Regression）  
当特征之间存在多重共线性时，$\mathbf{X}^T \mathbf{X}$ 可能不可逆。岭回归通过添加L2正则化解决这个问题：

$$
\boldsymbol{\theta} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}
$$

实现岭回归类 `RidgeRegression`，并测试它在以下数据上的表现：
```python
X = [[1, 2], [2, 4], [3, 6], [4, 8]]  # 注意：第二列是第一列的2倍（共线性）
y = [3, 6, 9, 12]
```
比较普通最小二乘法和岭回归的结果。

---

## 参考文献
