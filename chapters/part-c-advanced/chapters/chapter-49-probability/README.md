

---

## 49.8 本章小结

### 核心概念回顾

**1. 凸优化基础**
- **凸集**：任意两点连线仍在集合内
- **凸函数**：弦在函数上方，切线在函数下方
- **关键性质**：局部最优 = 全局最优

**2. 梯度下降理论**
- **凸+L-光滑**：收敛率 $O(1/T)$
- **强凸+L-光滑**：线性收敛 $O(e^{-T/\kappa})$
- **学习率选择**：$\eta = 1/L$ 是安全选择

**3. 约束优化**
- **拉格朗日乘子法**：处理等式约束
- **KKT条件**：处理不等式约束的四条黄金法则
- **对偶理论**：弱对偶恒成立，强对偶需凸性

**4. 优化算法工具箱**
| 算法 | 特点 | 适用场景 |
|------|------|----------|
| 梯度下降 | 简单、内存低 | 大规模问题 |
| 动量法 | 加速收敛 | 峡谷型损失 |
| Nesterov | $O(1/k^2)$收敛 | 凸问题 |
| Adam | 自适应学习率 | 深度学习 |
| 牛顿法 | 二次收敛 | 中小规模 |
| BFGS | 超线性收敛 | 中等规模 |

### 费曼法一句话总结

> 优化就像在山上找最低点——凸函数是一座碗，你滑到底就对了；非凸函数是群山，要小心别困在小山丘上。梯度告诉你哪里最陡，牛顿法告诉你山有多弯，对偶理论告诉你宝藏至少值多少钱。

---

## 49.9 练习题

### 基础练习

**练习1：凸性判断**

判断下列函数是否是凸函数，并说明理由：

(1) $f(x) = |x|$ on $\mathbb{R}$

(2) $f(x) = x^3$ on $\mathbb{R}$

(3) $f(\mathbf{x}) = \|\mathbf{x}\|_2^2$ on $\mathbb{R}^n$

(4) $f(\mathbf{x}) = \max\{x_1, x_2, ..., x_n\}$ on $\mathbb{R}^n$

<details>
<summary>点击查看答案</summary>

(1) **是凸函数**。绝对值函数的二阶导数在非零点为0，在0点不存在但满足凸函数定义：$|tx + (1-t)y| \leq t|x| + (1-t)|y|$（三角不等式）。

(2) **不是凸函数**。$f''(x) = 6x$，当 $x < 0$ 时 $f''(x) < 0$，不是凸函数。实际上在 $x < 0$ 区域是凹的。

(3) **是凸函数**。Hessian矩阵 $\nabla^2 f = 2I$ 正定，满足二阶条件。

(4) **是凸函数**。多个仿射函数的逐点最大值保持凸性。

</details>

**练习2：梯度下降收敛**

设 $f(x) = \frac{1}{2}(x-3)^2$，初始点 $x_0 = 0$，学习率 $\eta = 0.1$。

(1) 计算梯度下降的前5步迭代值

(2) 证明收敛到最优解 $x^* = 3$

<details>
<summary>点击查看答案</summary>

(1) 梯度为 $\nabla f(x) = x - 3$

- $x_0 = 0$
- $x_1 = 0 - 0.1(0-3) = 0.3$
- $x_2 = 0.3 - 0.1(0.3-3) = 0.57$
- $x_3 = 0.57 - 0.1(0.57-3) = 0.813$
- $x_4 = 0.813 - 0.1(0.813-3) = 1.032$
- $x_5 = 1.032 - 0.1(1.032-3) = 1.229$

(2) 迭代公式：$x_{k+1} = x_k - \eta(x_k - 3) = (1-\eta)x_k + \eta \cdot 3$

令误差 $e_k = x_k - 3$，则 $e_{k+1} = (1-\eta)e_k = (1-\eta)^{k+1}e_0$

当 $0 < \eta < 2$ 时 $|1-\eta| < 1$，所以 $e_k \to 0$，即 $x_k \to 3$。

</details>

**练习3：KKT条件应用**

求解以下优化问题：
$$\min_{x,y} \quad x^2 + y^2$$
$$\text{s.t.} \quad x + y \geq 1$$

<details>
<summary>点击查看答案</summary>

**步骤1**：识别约束是活跃的（在最优解处取等号）。

**步骤2**：写拉格朗日函数：$\mathcal{L} = x^2 + y^2 - \lambda(x + y - 1)$

**步骤3**：KKT条件：
- 平稳性：$2x - \lambda = 0$，$2y - \lambda = 0$
- 原始可行性：$x + y \geq 1$
- 对偶可行性：$\lambda \geq 0$
- 互补松弛：$\lambda(x + y - 1) = 0$

**步骤4**：求解：
由前两个方程：$x = y = \lambda/2$

由互补松弛（假设约束活跃）：$x + y = 1$

所以 $\lambda/2 + \lambda/2 = 1 \Rightarrow \lambda = 1$

**答案**：$x^* = y^* = 0.5$，$f^* = 0.5$

</details>

### 进阶练习

**练习4：强凸性分析**

证明：若 $f$ 是 $\mu$-强凸且 $L$-光滑的，则对于梯度下降（步长 $\eta = 1/L$）：
$$\|\mathbf{x}_k - \mathbf{x}^*\|^2 \leq \left(1 - \frac{\mu}{L}\right)^k \|\mathbf{x}_0 - \mathbf{x}^*\|^2$$

**提示**：使用强凸性和光滑性的定义。

**练习5：对偶间隙计算**

考虑线性规划问题：
$$\min_x \; c^T x \quad \text{s.t.} \; Ax \leq b$$

(1) 写出拉格朗日对偶问题

(2) 证明对偶函数是凹函数

(3) 解释为什么线性规划总是满足强对偶性

### 编程练习

**练习6：实现投影梯度下降**

实现投影梯度下降算法，并用它求解：
$$\min_{\mathbf{x}} \|\mathbf{x} - \mathbf{a}\|^2 \quad \text{s.t.} \quad \|\mathbf{x}\| \leq 1$$

其中 $\mathbf{a} = [2, 2]^T$。理论最优解是什么？

**练习7：比较优化算法**

使用本章代码，在Rosenbrock函数上比较：
- 标准梯度下降
- 动量梯度下降
- Nesterov加速
- Adam

绘制收敛曲线并分析。

---

## 49.10 参考文献

### 经典教材

Boyd, S., & Vandenberghe, L. (2004). *Convex optimization*. Cambridge University Press.

Nesterov, Y. (2013). *Introductory lectures on convex optimization: A basic course* (Vol. 87). Springer Science & Business Media.

Bertsekas, D. P. (2016). *Nonlinear programming* (3rd ed.). Athena Scientific.

Nocedal, J., & Wright, S. J. (2006). *Numerical optimization* (2nd ed.). Springer.

### 学术论文

Nesterov, Y. (1983). A method for solving the convex programming problem with convergence rate $O(1/k^2)$. *Soviet Mathematics Doklady*, 27(2), 372-376.

Polyak, B. T. (1964). Some methods of speeding up the convergence of iteration methods. *USSR Computational Mathematics and Mathematical Physics*, 4(5), 1-17.

Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. *Journal of Machine Learning Research*, 12, 2121-2159.

Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

Karush, W. (1939). *Minima of functions of several variables with inequalities as side conditions*. Master's thesis, University of Chicago.

Kuhn, H. W., & Tucker, A. W. (1951). Nonlinear programming. *Proceedings of the Second Berkeley Symposium on Mathematical Statistics and Probability*, 481-492.

### 在线资源

- Boyd & Vandenberghe, Convex Optimization: https://web.stanford.edu/~boyd/cvxbook/
- CVXPY Documentation: https://www.cvxpy.org/
- scipy.optimize tutorial: https://docs.scipy.org/doc/scipy/reference/optimize.html

---

## 附录：数学符号速查

| 符号 | 含义 |
|------|------|
| $\mathbb{R}^n$ | n维实向量空间 |
| $\nabla f(\mathbf{x})$ | 函数$f$在$\mathbf{x}$处的梯度 |
| $\nabla^2 f(\mathbf{x})$ | 函数$f$的Hessian矩阵 |
| $\|\mathbf{x}\|$ | 向量$\mathbf{x}$的范数 |
| $\|\mathbf{x}\|_2$ | L2范数：$\sqrt{\sum_i x_i^2}$ |
| $\|\mathbf{x}\|_1$ | L1范数：$\sum_i |x_i|$ |
| $A \succeq 0$ | 矩阵$A$半正定 |
| $A \succ 0$ | 矩阵$A$正定 |
| $\mathcal{L}$ | 拉格朗日函数 |
| $\lambda, \nu$ | 拉格朗日乘子 |
| $O(\cdot)$ | 大O符号（渐近上界） |
| $\kappa$ | 条件数：$\kappa = L/\mu$ |

---

*本章完。继续加油，下一章更精彩！*
