# 第五十章 概率图模型与推断

## 写作计划 (2026-03-26 14:45)

### 章节定位
- **主题**: 概率图模型与推断 (Probabilistic Graphical Models and Inference)
- **目标读者**: 已有前49章基础，准备掌握概率建模核心方法
- **衔接性**: 第48章贝叶斯深度学习→第50章概率图模型→第51章因果推断

### 内容架构 (~16,000字)

#### 50.1 引言：为什么需要概率图模型
- 从联合分布的诅咒到图表示的优势
- 有向 vs 无向图模型概览
- 费曼比喻：家族族谱 vs 社交网络

#### 50.2 贝叶斯网络：有向图的因果之美
- 条件独立性：D-分离原理
- 因子分解：链式法则的简化
- CPD表示：表格、确定性、上下文特定
- 代码：BayesianNetwork类实现

#### 50.3 马尔可夫随机场：无向图的和谐之美
- 团与势函数
- Gibbs分布与配分函数
- 马尔可夫毯
- 代码：MRF类实现

#### 50.4 精确推断：变量消除与信念传播
- 变量消除算法
- 团树与连接树
- 信念传播（和积算法）
- 代码：VariableElimination、BeliefPropagation

#### 50.5 近似推断：采样方法
- MCMC基础
- Metropolis-Hastings算法
- Gibbs采样
- 代码：MCMC采样器实现

#### 50.6 近似推断：变分方法
- 变分推断基本原理
- 平均场近似
- ELBO推导与优化
- 代码：MeanFieldVI实现

#### 50.7 应用案例
- 医学诊断系统
- 图像去噪
- 主题模型LDA简介

#### 50.8 前沿与展望
- 深度生成模型中的图模型
- 神经变分推断
- 概率编程

#### 50.9 本章小结

#### 50.10 练习题 (9道)
- 3基础 + 3进阶 + 3挑战

### 费曼法比喻设计

1. **贝叶斯网络** → "家族族谱"
   - 节点：家族成员
   - 边：血缘关系（有方向）
   - 条件概率：父母影响孩子的概率

2. **马尔可夫随机场** → "朋友圈"
   - 节点：个人
   - 边：朋友关系（无方向）
   - 势函数：朋友间的默契程度

3. **信念传播** → "谣言传播"
   - 消息传递：告诉邻居你听到的事
   - 聚合：综合所有邻居的信息
   - 收敛：谣言最终稳定

4. **MCMC采样** → "随机漫步探索城市"
   - 马尔可夫链：根据当前位置决定下一步
   - 稳态分布：长时间后在各区域停留的概率
   - 采样：记录经过的位置

5. **变分推断** → "用简单形状近似复杂形状"
   - 真实后验：复杂的云状
   - 变分分布：简单的球状
   - KL散度：两个形状的差异

### 代码架构

```
pgm/
├── __init__.py
├── bayesian_network.py      # 贝叶斯网络
├── mrf.py                   # 马尔可夫随机场
├── factor.py                # 因子操作
├── variable_elimination.py  # 变量消除
├── belief_propagation.py    # 信念传播
├── mcmc.py                  # MCMC采样
├── variational_inference.py # 变分推断
└── utils.py                 # 工具函数
```

### 参考文献 (10+篇APA)

1. Koller, D., & Friedman, N. (2009). Probabilistic graphical models: Principles and techniques. MIT Press.
2. Pearl, J. (1988). Probabilistic reasoning in intelligent systems: Networks of plausible inference. Morgan Kaufmann.
3. Murphy, K. P. (2012). Machine learning: A probabilistic perspective. MIT Press.
4. Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.
5. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. Journal of the American Statistical Association, 112(518), 859-877.
6. Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). An introduction to variational methods for graphical models. Machine Learning, 37(2), 183-233.
7. Wainwright, M. J., & Jordan, M. I. (2008). Graphical models, exponential families, and variational inference. Foundations and Trends in Machine Learning, 1(1-2), 1-305.
8. Andrieu, C., De Freitas, N., Doucet, A., & Jordan, M. I. (2003). An introduction to MCMC for machine learning. Machine Learning, 50(1-2), 5-43.
9. Geman, S., & Geman, D. (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. IEEE Transactions on Pattern Analysis and Machine Intelligence, 6(6), 721-741.
10. Yedidia, J. S., Freeman, W. T., & Weiss, Y. (2005). Constructing free-energy approximations and generalized belief propagation algorithms. IEEE Transactions on Information Theory, 51(7), 2282-2312.

### 预期产出
- 正文字数: ~16,000字
- 代码行数: ~1,800行
- 图表: 8-10个
- 练习题: 9道
- 参考文献: 10+篇

---
*开始写作: 2026-03-26 14:45*
*目标完成: 2026-03-27*
