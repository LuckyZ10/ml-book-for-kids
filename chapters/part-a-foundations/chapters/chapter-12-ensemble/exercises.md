# 第十二章 集成学习 练习题

## 练习题

### 基础练习

**练习12.1：Bootstrap抽样**

假设你有一个包含100个样本的数据集。进行Bootstrap抽样：

1. 大约有多少比例的样本会被选中至少一次？
2. 计算当样本数 $n \to \infty$ 时，某个特定样本未被选中的概率。
3. 验证：$\lim_{n \to \infty} (1 - 1/n)^n = 1/e \approx 0.368$

**练习12.2：方差计算**

假设你有5个独立的分类器，每个的预测方差为 $\sigma^2 = 4$。

1. 如果使用简单平均集成，集成的方差是多少？
2. 如果增加模型数量到20个，方差变为多少？
3. 从数学上解释为什么"越多越好"。

**练习12.3：AdaBoost权重**

在AdaBoost中，假设某一轮的加权错误率 $\epsilon_t = 0.2$。

1. 计算该轮学习器的权重 $\alpha_t$。
2. 如果一个样本被正确分类，它的权重会如何变化（增大还是减小）？
3. 如果一个样本被错误分类，它的权重会乘以多少倍？

### 进阶挑战

**练习12.4：实现OOB误差估计**

扩展`BaggingClassifier`类，添加OOB（Out-of-Bag）误差估计功能。

提示：
- 对每个基学习器，记录哪些样本没有被Bootstrap选中
- 用这些OOB样本评估该学习器
- 平均所有学习器的OOB误差

**练习12.5：堆叠集成（Stacking）**

研究并实现**Stacking**集成方法：

1. 用K折交叉验证训练多个基学习器
2. 用基学习器的预测作为特征，训练一个元学习器
3. 比较Stacking与Bagging、Boosting的性能

参考文献：Wolpert, D. H. (1992). Stacked generalization. Neural Networks, 5(2), 241-259.

### 终极挑战

**练习12.6：梯度提升（Gradient Boosting）**

实现**梯度提升**算法，这是AdaBoost的推广，也是XGBoost的核心思想。

要求：
1. 理解梯度提升与AdaBoost的区别
2. 用回归树作为基学习器
3. 实现平方损失函数的梯度提升
4. 在回归数据集上测试

提示：
- 梯度提升每一步拟合的是"残差"（当前预测与真实值的差）
- 学习率（shrinkage）是一个重要的超参数

参考文献：Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 1189-1232.

---

