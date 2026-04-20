# 第十三章 K-Means聚类 练习题

> 本章练习题覆盖K-Means算法的核心概念、数学推导、实际应用和编程实现。
> 建议完成时间：6-8小时

---

## 基础题（10道）

### 练习1：算法步骤理解 ⭐

**问题1**：请用文字描述K-Means算法的两个主要步骤（分配步和更新步），并解释每一步分别解决了什么问题。

<details>
<summary>参考答案</summary>

**分配步（Assignment Step / E步）**：
- 给定当前簇中心，将每个样本分配到距离最近的中心所在的簇
- 解决的问题：在固定中心的情况下，找到最优的样本分配方案
- 为什么最优？因为把点分配到最近中心，直接最小化了该点到其所属簇中心的距离平方和

**更新步（Update Step / M步）**：
- 给定当前的样本分配，重新计算每个簇的中心（即簇内所有样本的均值）
- 解决的问题：在固定分配的情况下，找到最优的簇中心位置
- 为什么最优？因为对损失函数求导后，均值恰好使损失函数最小

**类比**：
- 分配步就像学生选择离自己最近的食堂吃饭
- 更新步就像学校根据每个食堂的就餐人数重新调整食堂的位置

</details>

**问题2**：为什么K-Means被称为"Lloyd算法"？它和"Forgy算法"有什么区别？

<details>
<summary>参考答案</summary>

**历史背景**：
- **Lloyd算法**：1957年由贝尔实验室的Stuart Lloyd提出，最初用于信号量化（PCM），1982年正式发表
- **Forgy算法**：1965年由Edward Forgy独立提出，与Lloyd的算法完全相同
- **MacQueen算法**：1967年由James MacQueen正式命名为"K-Means"并给出理论分析

**区别**：
实际上，Lloyd算法和Forgy算法**是同一个算法**，只是由不同的人在不同领域独立发现：
- Lloyd在信号处理领域
- Forgy在统计学领域
- MacQueen在机器学习领域

因此，严格来说应该叫"Lloyd-Forgy-MacQueen算法"，但"K-Means"这个名字更简洁，被广泛接受。

</details>

---

### 练习2：损失函数 ⭐

**问题**：K-Means要最小化的损失函数（目标函数）是什么？为什么使用距离平方而不是距离本身？

<details>
<summary>参考答案</summary>

**损失函数**：

$$J = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2$$

其中：
- $K$ 是聚类数
- $C_i$ 是第$i$个簇的样本集合
- $\mu_i$ 是第$i$个簇的中心
- $\|x - \mu_i\|^2$ 是样本到中心的欧氏距离平方

**为什么用距离平方而不是距离**：

1. **数学可导性**：
   - 距离平方 $\|x - \mu\|^2$ 对 $\mu$ 可导
   - 距离 $\|x - \mu\|$ 在 $x = \mu$ 处不可导
   - 可导性使得我们可以通过求导找到解析解（均值）

2. **解析解存在**：
   - 对 $\|x - \mu\|^2$ 求导并令导数为0：
     $$\frac{\partial}{\partial \mu} \sum_{x \in C} \|x - \mu\|^2 = -2\sum_{x \in C}(x - \mu) = 0$$
   - 解得：$\mu = \frac{1}{|C|}\sum_{x \in C} x$（均值）
   - 如果用距离本身，没有简单的解析解

3. **凸性**：
   - 距离平方是凸函数，保证优化问题的良好性质
   - 距离本身也是凸函数，但不如平方光滑

4. **统计意义**：
   - 距离平方和对应于方差
   - 最小化距离平方和等价于最小化簇内方差
   - 这与统计学中的最小二乘思想一致

**注意**：使用距离平方会使算法对异常值更敏感，因为一个远离中心的点会产生很大的惩罚。

</details>

---

### 练习3：收敛性分析 ⭐

**问题1**：为什么K-Means算法一定会收敛？它保证收敛到全局最优吗？

<details>
<summary>参考答案</summary>

**一定会收敛的原因**：

1. **损失函数有下界**：
   - $J = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2 \geq 0$
   - 距离平方和不可能为负

2. **每步迭代不增加损失**：
   - 分配步：固定中心，最优分配使损失最小化或不变
   - 更新步：固定分配，最优中心使损失最小化或不变
   - 因此 $J^{(t+1)} \leq J^{(t)}$

3. **有限种分配方式**：
   - $N$ 个样本分配到 $K$ 个簇，最多有 $K^N$ 种分配方式
   - 虽然这个数字很大，但是有限的
   - 算法不可能无限循环，因为损失严格递减（或不变时停止）

**不保证全局最优**：

K-Means**只保证收敛到局部最优**，原因：

1. **目标函数非凸**：
   - K-Means的目标函数是NP难问题
   - 存在多个局部最优解

2. **初始化敏感性**：
   - 不同的初始中心可能导致不同的收敛结果
   - 例如：三个簇排成一条线，初始中心都在左边，可能收敛到错误的结果

3. **示例**：
   ```
   数据分布：三个明显的簇，分别在位置 -10, 0, 10
   错误初始化：两个中心在 -10 附近，一个在 10 附近
   可能结果：-10处的簇被分成两半，0处的簇被忽略
   ```

**改善方法**：
- 多次随机初始化，选择损失最小的结果
- 使用K-Means++初始化
- 使用模拟退火等全局优化方法

</details>

**问题2**：K-Means最多需要多少轮迭代才能收敛？

<details>
<summary>参考答案</summary>

**理论上限**：

在最坏情况下，K-Means可能需要指数级迭代次数才能收敛。

- 每次迭代改变至少一个点的分配（否则已收敛）
- 共有 $K^N$ 种可能的分配方式
- 因此理论上最多 $K^N$ 次迭代

**实际经验**：

- 实际应用中，K-Means通常在 **20-100 次迭代**内收敛
- 数据维度低、簇分离明显时收敛更快
- 数据维度高、簇重叠严重时可能需要更多迭代

**设置上限**：
```python
from sklearn.cluster import KMeans
# 设置最大迭代次数，防止无限循环
kmeans = KMeans(n_clusters=3, max_iter=300, random_state=42)
```

</details>

---

### 练习4：K-Means++初始化 ⭐

**问题1**：K-Means++的核心思想是什么？相比随机初始化有什么优势？

<details>
<summary>参考答案</summary>

**核心思想**：

让初始中心点**彼此尽可能远离**，从而：
1. 覆盖数据的不同区域
2. 减少局部最优的风险
3. 提高收敛速度和聚类质量

**算法步骤**：

1. 随机选择第一个中心点
2. 对每个数据点 $x$，计算它到最近已选中心的距离 $D(x)$
3. 以概率 $\frac{D(x)^2}{\sum_{x'} D(x')^2}$ 选择下一个中心点
4. 重复步骤2-3，直到选够 $K$ 个中心

**为什么用 $D(x)^2$ 而不是 $D(x)$**：
- 平方使得远离的点被选中的概率更大
- 保证期望近似比为 $O(\log K)$

**相比随机初始化的优势**：

| 方面 | 随机初始化 | K-Means++ |
|:---|:---|:---|
| 收敛速度 | 可能很慢 | 通常更快 |
| 最终损失 | 可能很差 | 通常更好 |
| 理论保证 | 无 | $O(\log K)$近似比 |
| 计算成本 | $O(K)$ | $O(N \cdot K)$ |
| 稳定性 | 波动大 | 更稳定 |

**理论保证**（Arthur & Vassilvitskii, 2007）：

K-Means++的期望损失不超过最优解的 $8(\ln K + 2)$ 倍。

</details>

**问题2**：K-Means++是否完全消除了随机性？为什么？

<details>
<summary>参考答案</summary>

**没有完全消除随机性**，原因：

1. **第一个中心是随机选的**：
   - K-Means++的第一步是随机选择一个中心点
   - 不同的第一个中心可能导致不同的最终配置

2. **概率采样**：
   - 后续中心按概率 $D(x)^2$ 采样
   - 概率采样本身具有随机性
   - 距离最大的点不一定被选中，只是被选中的概率最大

3. **示例**：
   ```
   假设有两个远离的数据区域A和B
   如果第一个中心选在A区，第二个中心大概率选在B区
   但如果第一个中心恰好选在A和B之间呢？
   结果可能就不理想了
   ```

**改善方法**：
- 多次运行K-Means++，选择损失最小的结果
- sklearn中设置 `n_init=10` 或更大
- 结合其他启发式方法

```python
# sklearn默认n_init=10（新版本）
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
```

</details>

---

### 练习5：选择K值 ⭐

**问题1**：什么是肘部法则（Elbow Method）？请画出以下数据的肘部曲线并判断最佳K值。

| K | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| WCSS | 1000 | 400 | 200 | 150 | 120 | 100 | 95 | 90 |

<details>
<summary>参考答案</summary>

**肘部法则原理**：

1. 尝试不同的K值（通常1到10）
2. 对每个K运行K-Means，记录WCSS（Within-Cluster Sum of Squares）
3. 画出WCSS随K变化的曲线
4. 寻找曲线的"肘部"——即K增加时WCSS下降速度明显变缓的点

**肘部曲线分析**：

```
WCSS
1000 |*
 900 |
 800 |
 700 |
 600 |
 500 |    *
 400 |        *
 300 |            *
 200 |                *  ← 肘部！
 150 |                    *
 120 |                        *
 100 |                            *
  95 |                                *
  90 |                                    *
     +----+----+----+----+----+----+----+----→ K
     1    2    3    4    5    6    7    8
```

**判断最佳K值**：

- K=1→2：WCSS下降 600（大幅下降）
- K=2→3：WCSS下降 200（大幅下降）
- K=3→4：WCSS下降 50（明显下降减缓）← **肘部**
- K=4→5：WCSS下降 30（下降更缓）
- K=5→8：WCSS下降缓慢

**最佳K值 = 3**

原因：K=3之后，增加K带来的WCSS减少收益明显递减。这意味着3个簇已经能很好地解释数据结构。

**肘部法则的局限性**：
1. 有时肘部不明显（曲线平滑）
2. 受初始化和随机性影响
3. 只能给出参考，不能替代领域知识

</details>

**问题2**：除了肘部法则，还有哪些方法可以选择K值？

<details>
<summary>参考答案</summary>

**常用方法**：

1. **轮廓系数（Silhouette Score）**：
   - 范围 $[-1, 1]$，越接近1越好
   - 选择使平均轮廓系数最大的K
   - 同时考虑簇内紧密度和簇间分离度

2. **Davies-Bouldin指数**：
   - 越小越好
   - 衡量簇内距离与簇间距离的比值

3. **Calinski-Harabasz指数**：
   - 越大越好
   - 基于类间散度矩阵和类内散度矩阵

4. **Gap Statistic**：
   - 比较实际数据的WCSS与随机数据的WCSS
   - 选择Gap值最大的K

5. **信息准则**：
   - AIC（Akaike Information Criterion）
   - BIC（Bayesian Information Criterion）
   - 考虑模型复杂度惩罚

6. **业务知识**：
   - 根据实际应用场景确定K
   - 例如：客户分群可能只需要3-5类

**综合策略**：
```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

scores = {}
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    scores[k] = {
        'silhouette': silhouette_score(X, labels),
        'calinski': calinski_harabasz_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels),
        'wcss': kmeans.inertia_
    }
```

</details>

---

### 练习6：轮廓系数计算 ⭐

**问题**：假设一个点到同簇其他3个点的距离分别为2, 3, 4，到最近其他簇3个点的距离分别为5, 6, 7。计算该点的轮廓系数。

<details>
<summary>参考答案</summary>

**轮廓系数公式**：

$$s = \frac{b - a}{\max(a, b)}$$

其中：
- $a$ = 点到同簇其他点的平均距离（簇内不相似度）
- $b$ = 点到最近其他簇的平均距离（簇间不相似度）

**计算步骤**：

1. 计算 $a$（簇内平均距离）：
   $$a = \frac{2 + 3 + 4}{3} = \frac{9}{3} = 3$$

2. 计算 $b$（最近簇平均距离）：
   $$b = \frac{5 + 6 + 7}{3} = \frac{18}{3} = 6$$

3. 计算轮廓系数：
   $$s = \frac{b - a}{\max(a, b)} = \frac{6 - 3}{\max(3, 6)} = \frac{3}{6} = 0.5$$

**解释**：
- $s = 0.5$ 表示该点的聚类质量中等偏上
- 该点与同簇点的平均距离是3
- 该点与最近其他簇的平均距离是6
- 簇间距离是簇内距离的2倍，说明聚类效果不错

**完整轮廓系数评估**：
- 计算所有点的轮廓系数，取平均
- 平均轮廓系数 > 0.5：聚类效果较好
- 平均轮廓系数 < 0：聚类可能有问题

</details>

---

### 练习7：算法比较 ⭐

**问题**：比较K-Means和K-Means++在相同数据集上的表现，分析随机初始化对结果方差的影响。

<details>
<summary>参考答案</summary>

**实验设计**：

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成测试数据
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# 比较不同初始化
n_runs = 50
random_inertias = []
plusplus_inertias = []

for i in range(n_runs):
    # 随机初始化
    km_random = KMeans(n_clusters=4, init='random', n_init=1, random_state=i)
    km_random.fit(X)
    random_inertias.append(km_random.inertia_)
    
    # K-Means++初始化
    km_pp = KMeans(n_clusters=4, init='k-means++', n_init=1, random_state=i)
    km_pp.fit(X)
    plusplus_inertias.append(km_pp.inertia_)

print(f"随机初始化 - 均值: {np.mean(random_inertias):.2f}, 标准差: {np.std(random_inertias):.2f}")
print(f"K-Means++ - 均值: {np.mean(plusplus_inertias):.2f}, 标准差: {np.std(plusplus_inertias):.2f}")

# 可视化
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(random_inertias, bins=15, alpha=0.7, label='Random')
plt.hist(plusplus_inertias, bins=15, alpha=0.7, label='K-Means++')
plt.xlabel('Inertia (WCSS)')
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution of Final Loss')

plt.subplot(1, 2, 2)
plt.plot(range(n_runs), sorted(random_inertias), 'o-', label='Random', alpha=0.7)
plt.plot(range(n_runs), sorted(plusplus_inertias), 's-', label='K-Means++', alpha=0.7)
plt.xlabel('Run (sorted by loss)')
plt.ylabel('Inertia')
plt.legend()
plt.title('Sorted Final Loss')
plt.tight_layout()
plt.show()
```

**预期结果**：

| 指标 | 随机初始化 | K-Means++ |
|:---|:---|:---|
| 平均损失 | 较高 | 较低 |
| 损失方差 | 较大 | 较小 |
| 最差结果 | 可能很差 | 通常可接受 |
| 收敛速度 | 不稳定 | 更稳定 |

**结论**：
1. K-Means++显著降低了对随机初始化的敏感性
2. K-Means++的损失分布更集中，方差更小
3. 即使只运行一次，K-Means++也大概率给出不错的结果
4. 随机初始化可能需要多次运行（n_init > 1）才能获得稳定结果

</details>

---

### 练习8：K-Means++概率计算 ⭐

**问题**：有4个点在一条直线上，位置为0, 1, 3, 6。第一个中心选在位置0，计算其他三个点被选为第二个中心的概率分别是多少？

<details>
<summary>参考答案</summary>

**已知条件**：
- 点位置：$x_1 = 0$, $x_2 = 1$, $x_3 = 3$, $x_4 = 6$
- 第一个中心：$c_1 = 0$

**计算步骤**：

1. 计算每个点到最近中心的距离：
   - $D(0) = |0 - 0| = 0$
   - $D(1) = |1 - 0| = 1$
   - $D(3) = |3 - 0| = 3$
   - $D(6) = |6 - 0| = 6$

2. 计算距离平方：
   - $D(0)^2 = 0$
   - $D(1)^2 = 1$
   - $D(3)^2 = 9$
   - $D(6)^2 = 36$

3. 计算分母（距离平方和）：
   $$\sum D(x)^2 = 0 + 1 + 9 + 36 = 46$$

4. 计算每个点被选为第二个中心的概率：
   - $P(x_1 = 0) = \frac{0}{46} = 0$ （已经是中心，概率为0）
   - $P(x_2 = 1) = \frac{1}{46} \approx 0.022$ （约2.2%）
   - $P(x_3 = 3) = \frac{9}{46} \approx 0.196$ （约19.6%）
   - $P(x_4 = 6) = \frac{36}{46} \approx 0.783$ （约78.3%）

**结果**：

| 点位置 | 距离 | 距离平方 | 被选概率 |
|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 | 0% |
| 1 | 1 | 1 | 2.2% |
| 3 | 3 | 9 | 19.6% |
| 6 | 6 | 36 | 78.3% |

**解释**：
- 位置6的点被选中的概率最高（78.3%），因为它离现有中心最远
- 这符合K-Means++的设计思想：让初始中心尽可能分散
- 位置0的点已经是中心，所以不可能再被选为第二个中心

</details>

---

### 练习9：初始化影响分析 ⭐

**问题**：假设二维数据点均匀分布在单位正方形 $[0,1] \times [0,1]$ 中，K=2。如果两个初始中心点都在左下角区域（比如(0.1, 0.1)和(0.2, 0.2)），会发生什么？

<details>
<summary>参考答案</summary>

**分析**：

1. **初始状态**：
   - 中心1: (0.1, 0.1)
   - 中心2: (0.2, 0.2)
   - 两个中心都在左下角，距离很近

2. **分配步**：
   - 单位正方形中的点，到(0.1, 0.1)和(0.2, 0.2)的距离差异很小
   - 大致上，对角线 $y = x$ 下方的点可能分配到中心1
   - 对角线 $y = x$ 上方的点可能分配到中心2
   - 但实际上，由于两个中心很近，分配边界会很不稳定

3. **更新步**：
   - 两个新中心都会向数据的几何中心(0.5, 0.5)移动
   - 经过多次迭代后，一个中心可能移动到左上角区域
   - 另一个中心移动到右下角区域

4. **可能的结果**：
   - **情况A**：收敛到正确的划分（左半部分 vs 右半部分，或上半部分 vs 下半部分）
   - **情况B**：收敛到对角线划分（不稳定）
   - **情况C**：一个中心"吞噬"另一个，最终两个中心都在中心附近

**可视化**：

```
初始状态（不好）：              理想状态：
┌─────────┐                   ┌────┬────┐
│         │                   │    │    │
│    ★    │                   │  ★ │ ★  │
│   ★     │                   │    │    │
└─────────┘                   └────┴────┘
★ = 初始中心                   ★ = 理想中心位置
```

**解决方法**：
1. 使用K-Means++初始化
2. 多次运行，选择损失最小的结果
3. 手动设置更好的初始中心

</details>

---

### 练习10：Mini-Batch K-Means ⭐

**问题**：研究Mini-Batch K-Means算法，解释它如何加速大规模数据的聚类，以及和标准K-Means的权衡。

<details>
<summary>参考答案</summary>

**Mini-Batch K-Means原理**：

标准K-Means每次迭代需要计算所有样本到所有中心的距离，时间复杂度 $O(N \cdot K \cdot d)$。

Mini-Batch K-Means每次只使用一小批样本（batch）来更新中心：

```python
# Mini-Batch K-Means伪代码
for iteration in range(max_iter):
    # 随机采样一个batch
    batch = random_sample(X, batch_size)
    
    # 只计算batch样本的分配
    for x in batch:
        # 找到最近中心
        nearest_center = argmin_i ||x - mu_i||
        
        # 更新该中心（使用学习率）
        mu_nearest += learning_rate * (x - mu_nearest)
```

**加速机制**：

1. **减少计算量**：
   - 每次迭代只处理 batch_size 个样本
   - 时间复杂度从 $O(N \cdot K \cdot d)$ 降到 $O(b \cdot K \cdot d)$
   - $b$ 通常远小于 $N$（如 $b = 100$, $N = 1,000,000$）

2. **在线更新**：
   - 中心点逐步移动，不需要等待所有样本
   - 类似随机梯度下降的思想

3. **内存友好**：
   - 不需要一次性加载所有数据
   - 适合数据流场景

**权衡**：

| 方面 | 标准K-Means | Mini-Batch K-Means |
|:---|:---|:---|
| 收敛速度 | 慢（每次处理全部数据） | 快（每次处理少量数据） |
| 最终质量 | 通常更好 | 可能略差 |
| 内存需求 | 高 | 低 |
| 适用数据规模 | 中小规模 | 大规模/流式数据 |
| 确定性 | 确定（给定初始化） | 有随机性 |

**使用场景**：
- **标准K-Means**：数据量 < 10万，追求最佳聚类质量
- **Mini-Batch**：数据量 > 100万，或内存受限，或实时聚类需求

```python
from sklearn.cluster import MiniBatchKMeans

# 大规模数据聚类
mbkmeans = MiniBatchKMeans(
    n_clusters=100,
    batch_size=1000,  # 每批处理1000个样本
    max_iter=100,
    random_state=42
)
mbkmeans.fit(X_large)
```

</details>

---

## 进阶题（8道）

### 练习11：数学证明——均值最优性 ⭐⭐

**问题**：证明在固定样本分配的情况下，簇中心取样本均值时，损失函数最小。

<details>
<summary>参考答案</summary>

**证明**：

给定簇 $C$ 中的样本 $\{x_1, x_2, ..., x_n\}$，我们要最小化：

$$J(\mu) = \sum_{i=1}^{n} \|x_i - \mu\|^2$$

**步骤1：展开平方距离**

$$J(\mu) = \sum_{i=1}^{n} (x_i - \mu)^T(x_i - \mu)$$

**步骤2：对 $\mu$ 求导**

$$\frac{\partial J}{\partial \mu} = \sum_{i=1}^{n} \frac{\partial}{\partial \mu} (x_i - \mu)^T(x_i - \mu)$$

$$= \sum_{i=1}^{n} (-2)(x_i - \mu)$$

$$= -2\sum_{i=1}^{n} x_i + 2n\mu$$

**步骤3：令导数为0**

$$-2\sum_{i=1}^{n} x_i + 2n\mu = 0$$

$$2n\mu = 2\sum_{i=1}^{n} x_i$$

$$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$$

**步骤4：验证二阶导数**

$$\frac{\partial^2 J}{\partial \mu^2} = 2n > 0$$

二阶导数为正，说明这是一个极小值点。

**结论**：

当簇中心取样本均值时，损失函数达到最小值。

这就是为什么K-Means的更新步要计算均值，而不是中位数或其他统计量。

**注意**：
- 如果使用曼哈顿距离（L1距离），最优中心是中位数而不是均值
- 这就是K-Medoids算法使用 medoid（实际样本点）而非均值的原因

</details>

---

### 练习12：空簇问题 ⭐⭐

**问题**：在K-Means运行过程中，可能会出现某个簇没有样本（空簇）的情况。为什么会发生？如何解决？

<details>
<summary>参考答案</summary>

**空簇产生的原因**：

1. **初始化问题**：
   - 某个初始中心远离所有数据点
   - 所有样本都分配到其他更近的中心

2. **数据分布问题**：
   - 数据本身不是均匀分布的
   - 某些区域数据稀疏，某些区域数据密集

3. **K值过大**：
   - K超过了数据中实际存在的簇数
   - 多余的中心无法"抢到"样本

**示例**：

```
数据分布：        初始化（K=4）：
● ●              ●
● ●              
    ○ ○              ○
    ○ ○          ★

★ = 空簇中心（远离所有数据）
```

**解决方法**：

**方法1：重新初始化空簇中心**
```python
def handle_empty_clusters(X, centroids, labels, k):
    empty_clusters = [i for i in range(k) if np.sum(labels == i) == 0]
    
    for empty_idx in empty_clusters:
        # 找到距离当前中心最远的点
        distances = np.linalg.norm(X - centroids[empty_idx], axis=1)
        farthest_idx = np.argmax(distances)
        centroids[empty_idx] = X[farthest_idx]
    
    return centroids
```

**方法2：减少K值**
- 如果经常出现空簇，说明K可能设置得太大
- 使用肘部法则重新评估K值

**方法3：使用K-Medoids**
- K-Medoids选择实际样本点作为中心
- 不容易出现空簇问题

**sklearn的处理方式**：

sklearn的KMeans实现会自动处理空簇：
- 当某个簇为空时，会重新初始化该中心
- 如果没有可重新初始化的样本（极端情况），会发出警告

```python
# sklearn会自动处理，但你可以检查
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X)

# 检查每个簇的样本数
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))
# 如果某个簇的count为0，就是空簇
```

</details>

---

### 练习13：特征缩放的重要性 ⭐⭐

**问题**：假设有一个二维数据集，特征1的范围是[0, 1]，特征2的范围是[0, 1000]。如果不进行特征缩放，K-Means会出现什么问题？

<details>
<summary>参考答案</summary>

**问题分析**：

在不缩放的情况下，欧氏距离计算为：
$$d(x, y) = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2}$$

由于特征2的范围是特征1的1000倍：
- $(x_2 - y_2)^2$ 会主导距离计算
- $(x_1 - y_1)^2$ 的贡献几乎被忽略

**后果**：

1. **特征2主导聚类**：
   - 聚类结果几乎只由特征2决定
   - 特征1的信息被"淹没"

2. **错误的聚类结构**：
   ```
   原始数据（未缩放）：
   特征1: [0, 1]      特征2: [0, 1000]
   
   数据分布：
   ● ●      ○ ○
   ● ●      ○ ○
   
   聚类结果（未缩放）：
   可能按照特征2的值分成上下两组
   忽略了特征1中左右分组的真实结构
   ```

3. **可视化失真**：
   - 在二维平面上，数据点看起来压缩成一条线
   - 真实的二维结构无法展现

**解决方法**：

**标准化（Standardization）**：
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 每个特征均值为0，标准差为1
```

**归一化（Normalization）**：
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# 每个特征缩放到[0, 1]
```

**对比实验**：

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 生成数据：特征1范围[0,1]，特征2范围[0,1000]
np.random.seed(42)
X = np.random.rand(100, 2)
X[:, 1] *= 1000

# 未缩放聚类
kmeans_unscaled = KMeans(n_clusters=2, random_state=42)
labels_unscaled = kmeans_unscaled.fit_predict(X)

# 标准化后聚类
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans_scaled = KMeans(n_clusters=2, random_state=42)
labels_scaled = kmeans_scaled.fit_predict(X_scaled)

print("未缩放中心：", kmeans_unscaled.cluster_centers_)
print("缩放后中心：", scaler.inverse_transform(kmeans_scaled.cluster_centers_))
```

**最佳实践**：
- **总是**在K-Means之前进行特征缩放
- 标准化（z-score）通常比归一化（min-max）更稳健
- 如果特征是同一量纲（如像素值），可以不缩放

</details>

---

### 练习14：K-Means与GMM的关系 ⭐⭐

**问题**：解释K-Means与高斯混合模型（GMM）之间的关系。在什么条件下，K-Means可以看作是GMM的特例？

<details>
<summary>参考答案</summary>

**GMM回顾**：

高斯混合模型假设数据来自 $K$ 个高斯分布的混合：

$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

其中：
- $\pi_k$ 是混合系数（$\sum_k \pi_k = 1$）
- $\mu_k$ 是均值
- $\Sigma_k$ 是协方差矩阵

**GMM的EM算法**：

**E步**：计算后验概率（责任值）
$$\gamma_{nk} = \frac{\pi_k \mathcal{N}(x_n | \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_n | \mu_j, \Sigma_j)}$$

**M步**：更新参数
$$\mu_k = \frac{\sum_n \gamma_{nk} x_n}{\sum_n \gamma_{nk}}$$

**K-Means作为GMM的特例**：

当GMM满足以下三个条件时，EM算法退化为K-Means：

1. **等协方差**：$\Sigma_k = \sigma^2 I$（所有簇的协方差相同且是各向同性）
2. **极限情况**：$\sigma^2 \to 0$（协方差趋近于0）
3. **等混合系数**：$\pi_k = \frac{1}{K}$

**推导**：

当 $\sigma^2 \to 0$ 时：

$$\mathcal{N}(x | \mu_k, \sigma^2 I) \propto \exp\left(-\frac{\|x - \mu_k\|^2}{2\sigma^2}\right)$$

指数项中的 $\|x - \mu_k\|^2$ 决定高斯值的大小：
- 距离最近的中心，指数值最大
- 其他中心的指数值快速趋近于0

因此：
$$\gamma_{nk} \to \begin{cases} 1 & \text{if } k = \arg\min_j \|x_n - \mu_j\|^2 \\ 0 & \text{otherwise} \end{cases}$$

这就是**硬分配**（Hard Assignment），与K-Means的分配步完全一致！

**对比**：

| 特性 | K-Means | GMM |
|:---|:---|:---|
| 分配方式 | 硬分配（0或1） | 软分配（概率） |
| 簇形状 | 球形 | 椭圆形 |
| 簇大小 | 相同 | 可以不同 |
| 目标函数 | 距离平方和 | 对数似然 |
| 优化方法 | EM算法（硬分配） | EM算法（软分配） |

**直观理解**：

- K-Means：每个点**必须**属于且仅属于一个簇
- GMM：每个点**可能**属于多个簇，只是概率不同

当高斯分布变得非常"尖锐"（方差很小）时，GMM中概率最大的那个簇会占据绝对优势，从而退化为硬分配。

</details>

---

### 练习15：核K-Means ⭐⭐

**问题**：什么是核K-Means？它如何解决非凸簇的聚类问题？

<details>
<summary>参考答案</summary>

**动机**：

标准K-Means假设簇是球形的，对非凸形状（如环形、月牙形）效果差。

**核技巧**：

核K-Means通过核函数将数据映射到高维特征空间，在特征空间中进行K-Means聚类。

**算法**：

在特征空间中，点到中心的距离为：

$$\|\phi(x) - \mu_k^\phi\|^2 = K(x, x) - \frac{2}{|C_k|}\sum_{x' \in C_k} K(x, x') + \frac{1}{|C_k|^2}\sum_{x', x'' \in C_k} K(x', x'')$$

其中 $K(x, y) = \phi(x)^T\phi(y)$ 是核函数。

**常用核函数**：

1. **RBF核（高斯核）**：
   $$K(x, y) = \exp\left(-\gamma \|x - y\|^2\right)$$

2. **多项式核**：
   $$K(x, y) = (x^T y + c)^d$$

**示例**：

```python
from sklearn.cluster import SpectralClustering
import numpy as np

# 生成环形数据
theta = np.linspace(0, 2*np.pi, 200)
r1, r2 = 2, 4
X1 = np.c_[r1*np.cos(theta), r1*np.sin(theta)] + np.random.randn(200, 2)*0.1
X2 = np.c_[r2*np.cos(theta), r2*np.sin(theta)] + np.random.randn(200, 2)*0.1
X = np.vstack([X1, X2])

# 标准K-Means（效果差）
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
labels_kmeans = kmeans.fit_predict(X)

# 谱聚类（使用RBF核）
spectral = SpectralClustering(n_clusters=2, affinity='rbf', gamma=1.0, random_state=42)
labels_spectral = spectral.fit_predict(X)

# 谱聚类能正确分离环形数据
```

**与谱聚类的关系**：

核K-Means与谱聚类有密切联系：
- 谱聚类先对相似度矩阵进行特征分解
- 然后在特征向量上进行K-Means
- 可以看作是一种特定的核K-Means

**局限性**：

1. 计算成本高（需要计算核矩阵）
2. 核函数和参数选择困难
3. 大规模数据不适用

</details>

---

### 练习16：时间复杂度分析 ⭐⭐

**问题**：分析K-Means算法的时间复杂度。假设有 $N$ 个样本，$K$ 个簇，$d$ 维特征，平均迭代 $T$ 次。

<details>
<summary>参考答案</summary>

**每次迭代的时间复杂度**：

**分配步（E步）**：
- 计算每个样本到每个中心的距离
- $N$ 个样本 × $K$ 个中心 × $d$ 维距离计算
- 时间复杂度：$O(N \cdot K \cdot d)$

**更新步（M步）**：
- 计算每个簇的均值
- 每个簇需要计算 $d$ 维向量的平均
- 所有簇的样本数之和为 $N$
- 时间复杂度：$O(N \cdot d)$

**每次迭代总复杂度**：
$$O(N \cdot K \cdot d) + O(N \cdot d) = O(N \cdot K \cdot d)$$

**总时间复杂度**（$T$ 次迭代）：
$$O(T \cdot N \cdot K \cdot d)$$

**空间复杂度**：
- 存储数据：$O(N \cdot d)$
- 存储中心：$O(K \cdot d)$
- 存储标签：$O(N)$
- **总空间复杂度**：$O(N \cdot d + K \cdot d)$

**实际经验**：

| 参数 | 典型值 | 影响 |
|:---|:---|:---|
| $N$ | $10^3$ ~ $10^6$ | 主要瓶颈 |
| $K$ | 2 ~ 100 | 影响较小 |
| $d$ | 2 ~ 1000 | 中等影响 |
| $T$ | 20 ~ 100 | 通常不大 |

**优化方向**：

1. **减少距离计算**：
   - 使用三角不等式剪枝
   - Elkan算法：$O(N \cdot K)$ 减少到接近 $O(N)$

2. **使用Mini-Batch**：
   - 每次处理 $b$ 个样本
   - 复杂度降为 $O(T \cdot b \cdot K \cdot d)$

3. **并行化**：
   - 分配步可以并行
   - GPU加速距离计算

</details>

---

### 练习17：图像颜色量化 ⭐⭐

**问题**：使用K-Means实现图像颜色量化。将一张彩色图像压缩为仅使用16种颜色，并比较压缩前后的效果。

<details>
<summary>参考答案</summary>

**原理**：

图像颜色量化是将24位真彩色（约1670万种颜色）压缩为少量代表色的过程。

1. 将每个像素的RGB值看作3维空间中的点
2. 使用K-Means聚类（K=16）
3. 每个像素用所属簇的中心颜色代替

**实现**：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import resample

# 加载图像
from sklearn.datasets import load_sample_image
china = load_sample_image('china.jpg')

# 归一化到[0, 1]
china = np.array(china, dtype=np.float64) / 255

# 转换为样本矩阵 (height*width, 3)
w, h, d = original_shape = tuple(china.shape)
image_array = np.reshape(china, (w * h, d))

# 为加速，随机采样部分像素进行聚类
image_array_sample = resample(image_array, n_samples=1000, random_state=42)

# K-Means聚类（16种颜色）
kmeans = KMeans(n_clusters=16, random_state=42)
kmeans.fit(image_array_sample)

# 为所有像素分配颜色
labels = kmeans.predict(image_array)
quantized_image = kmeans.cluster_centers_[labels]
quantized_image = np.reshape(quantized_image, original_shape)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(china)
axes[0].set_title('Original Image (16777216 colors)')
axes[0].axis('off')

axes[1].imshow(quantized_image)
axes[1].set_title('Quantized Image (16 colors)')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# 计算压缩率
original_size = w * h * d * 8  # 每个通道8位
compressed_size = 16 * d * 8 + w * h * 4  # 16个颜色 + 每个像素4位索引
compression_ratio = original_size / compressed_size
print(f"Compression ratio: {compression_ratio:.2f}x")
```

**结果分析**：

| 指标 | 原始图像 | 量化后 |
|:---|:---|:---|
| 颜色数 | 16777216 | 16 |
| 每像素位数 | 24 | 4（索引） |
| 压缩率 | 1x | ~6x |
| 视觉质量 | 完美 | 可接受 |

**应用**：
- GIF格式（最多256色）
- 早期网页图像优化
- 风格化艺术效果

</details>

---

### 练习18：异常检测应用 ⭐⭐

**问题**：如何利用K-Means进行异常检测？请设计一个完整的方案。

<details>
<summary>参考答案</summary>

**思路**：

异常点通常远离正常数据的聚类中心，或者属于小簇/空簇。

**方案设计**：

**方法1：基于距离阈值**
```python
def anomaly_detection_distance(X, k=5, threshold_percentile=95):
    """
    基于到最近中心距离的异常检测
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # 计算每个样本到其所属簇中心的距离
    distances = np.linalg.norm(X - kmeans.cluster_centers_[labels], axis=1)
    
    # 使用百分位数确定阈值
    threshold = np.percentile(distances, threshold_percentile)
    
    # 标记异常
    anomalies = distances > threshold
    
    return anomalies, distances
```

**方法2：基于簇大小**
```python
def anomaly_detection_cluster_size(X, k=10, min_cluster_size=10):
    """
    小簇中的点可能是异常
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # 统计每个簇的大小
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    
    # 标记小簇中的点为异常
    anomalies = np.array([cluster_sizes[l] < min_cluster_size for l in labels])
    
    return anomalies
```

**方法3：综合方案**
```python
class KMeansAnomalyDetector:
    def __init__(self, k=5, distance_threshold=95, min_cluster_size=5):
        self.k = k
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.kmeans = None
    
    def fit(self, X):
        self.kmeans = KMeans(n_clusters=self.k, random_state=42)
        self.labels_ = self.kmeans.fit_predict(X)
        
        # 计算距离分布
        self.distances_ = np.linalg.norm(
            X - self.kmeans.cluster_centers_[self.labels_], axis=1
        )
        self.distance_threshold_value = np.percentile(
            self.distances_, self.distance_threshold
        )
        
        # 计算簇大小
        unique, counts = np.unique(self.labels_, return_counts=True)
        self.cluster_sizes_ = dict(zip(unique, counts))
        
        return self
    
    def predict(self, X):
        labels = self.kmeans.predict(X)
        distances = np.linalg.norm(
            X - self.kmeans.cluster_centers_[labels], axis=1
        )
        
        # 异常条件：距离远 或 属于小簇
        is_far = distances > self.distance_threshold_value
        is_small_cluster = np.array([
            self.cluster_sizes_.get(l, 0) < self.min_cluster_size 
            for l in labels
        ])
        
        return is_far | is_small_cluster
```

**使用示例**：

```python
# 生成带异常的数据
from sklearn.datasets import make_blobs
X_normal, _ = make_blobs(n_samples=300, centers=3, random_state=42)
X_anomalies = np.random.uniform(low=-10, high=10, size=(20, 2))
X = np.vstack([X_normal, X_anomalies])

# 检测异常
detector = KMeansAnomalyDetector(k=3, distance_threshold=95)
detector.fit(X)
anomalies = detector.predict(X)

print(f"Detected {np.sum(anomalies)} anomalies out of {len(X)} samples")
```

**优缺点**：

| 优点 | 缺点 |
|:---|:---|
| 简单直观 | 对簇形状敏感 |
| 计算高效 | 需要预设参数 |
| 可解释性强 | 高维数据效果差 |

</details>

---

## 挑战题（5道）

### 练习19：实现完整的K-Means算法 ⭐⭐⭐

**问题**：从零开始实现K-Means算法（不调用sklearn），包括：
1. 标准K-Means
2. K-Means++初始化
3. 肘部法则自动选K
4. 轮廓系数评估

<details>
<summary>参考答案</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

class KMeansFromScratch:
    def __init__(self, k=3, max_iter=100, tol=1e-4, init='k-means++'):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.centroids = None
        self.labels = None
        self.inertia_ = None
    
    def _kmeans_plus_plus(self, X):
        """K-Means++初始化"""
        n_samples, n_features = X.shape
        centroids = np.zeros((self.k, n_features))
        
        # 随机选第一个中心
        centroids[0] = X[np.random.choice(n_samples)]
        
        for i in range(1, self.k):
            # 计算每个样本到最近中心的距离
            distances = np.array([
                min([np.linalg.norm(x - c)**2 for c in centroids[:i]])
                for x in X
            ])
            
            # 按概率选择下一个中心
            probabilities = distances / distances.sum()
            centroids[i] = X[np.random.choice(n_samples, p=probabilities)]
        
        return centroids
    
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # 初始化中心
        if self.init == 'k-means++':
            self.centroids = self._kmeans_plus_plus(X)
        else:
            indices = np.random.choice(n_samples, self.k, replace=False)
            self.centroids = X[indices]
        
        # 迭代优化
        for iteration in range(self.max_iter):
            # 分配步
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)
            
            # 更新步
            new_centroids = np.array([
                X[self.labels == i].mean(axis=0) if np.sum(self.labels == i) > 0
                else self.centroids[i]  # 处理空簇
                for i in range(self.k)
            ])
            
            # 检查收敛
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            
            self.centroids = new_centroids
        
        # 计算最终损失
        self.inertia_ = sum([
            np.sum((X[self.labels == i] - self.centroids[i])**2)
            for i in range(self.k)
        ])
        
        return self
    
    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

def elbow_method(X, k_range=range(1, 11)):
    """肘部法则"""
    wcss = []
    for k in k_range:
        kmeans = KMeansFromScratch(k=k, init='k-means++')
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    # 可视化
    plt.plot(k_range, wcss, 'bo-')
    plt.xlabel('K')
    plt.ylabel('WCSS')
    plt.title('Elbow Method')
    plt.show()
    
    return wcss

def silhouette_score_scratch(X, labels):
    """轮廓系数"""
    n_samples = len(X)
    scores = []
    
    for i in range(n_samples):
        # 同簇距离
        same_cluster = X[labels == labels[i]]
        if len(same_cluster) == 1:
            scores.append(0)
            continue
        a = np.mean([np.linalg.norm(X[i] - x) for x in same_cluster if not np.array_equal(x, X[i])])
        
        # 最近其他簇距离
        b = float('inf')
        for label in np.unique(labels):
            if label != labels[i]:
                other_cluster = X[labels == label]
                avg_dist = np.mean([np.linalg.norm(X[i] - x) for x in other_cluster])
                b = min(b, avg_dist)
        
        scores.append((b - a) / max(a, b))
    
    return np.mean(scores)

# 测试
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    
    # 肘部法则
    wcss = elbow_method(X)
    
    # 聚类
    kmeans = KMeansFromScratch(k=4, init='k-means++')
    kmeans.fit(X)
    
    # 评估
    score = silhouette_score_scratch(X, kmeans.labels)
    print(f"Silhouette Score: {score:.3f}")
    
    # 可视化
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels, cmap='viridis')
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
                c='red', marker='x', s=200, linewidths=3)
    plt.title('K-Means Clustering')
    plt.show()
```

</details>

---

### 练习20：K-Means收敛性证明 ⭐⭐⭐

**问题**：严格证明K-Means算法的收敛性，并讨论收敛到全局最优的条件。

<details>
<summary>参考答案</summary>

**定理**：K-Means算法必然在有限步内收敛到局部最优。

**证明**：

**定义**：
- 设 $N$ 个样本，$K$ 个簇
- 损失函数：$J(C, \mu) = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2$
- 其中 $C = \{C_1, ..., C_K\}$ 是分配，$\mu = \{\mu_1, ..., \mu_K\}$ 是中心

**步骤1：证明损失函数有下界**

$$J(C, \mu) = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2 \geq 0$$

因为距离平方非负，所以损失函数有下界0。

**步骤2：证明分配步不增加损失**

固定中心 $\mu$，分配步将每个样本分配到最近的中心：

$$c(x) = \arg\min_i \|x - \mu_i\|^2$$

对于任意样本 $x$：
$$\|x - \mu_{c(x)}\|^2 \leq \|x - \mu_i\|^2, \quad \forall i$$

因此，新的分配 $C'$ 满足：
$$J(C', \mu) = \sum_{x} \|x - \mu_{c(x)}\|^2 \leq \sum_{x} \|x - \mu_{c_{old}(x)}\|^2 = J(C, \mu)$$

**步骤3：证明更新步不增加损失**

固定分配 $C$，更新步将每个中心更新为簇内均值：

$$\mu_i' = \frac{1}{|C_i|}\sum_{x \in C_i} x$$

由练习11的证明，均值使损失最小：
$$\sum_{x \in C_i} \|x - \mu_i'\|^2 \leq \sum_{x \in C_i} \|x - \mu_i\|^2, \quad \forall i$$

因此：
$$J(C, \mu') = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i'\|^2 \leq \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2 = J(C, \mu)$$

**步骤4：证明有限步收敛**

- 每次迭代，损失函数要么严格减小，要么不变
- 如果损失不变，则算法已收敛（分配和中心都不再变化）
- 共有有限种分配方式（最多 $K^N$ 种）
- 因此算法不可能无限循环，必在有限步内收敛

**全局最优条件**：

K-Means收敛到全局最优的**充分条件**：

1. **数据只有一个簇**：$K = 1$，此时全局最优=局部最优
2. **数据完全分离**：簇之间距离远大于簇内距离，且 $K$ 等于真实簇数
3. **初始中心在全局最优附近**：K-Means++有较高概率做到这一点

**全局最优的困难**：

K-Means的目标函数是NP难的，原因：
- 目标函数非凸
- 存在指数级多个局部最优
- 没有已知的多项式时间精确算法

**近似保证**：

K-Means++提供理论保证：
$$\mathbb{E}[J_{final}] \leq 8(\ln K + 2) \cdot J_{opt}$$

即期望损失不超过最优解的 $O(\log K)$ 倍。

</details>

---

### 练习21：设计新的初始化方法 ⭐⭐⭐

**问题**：设计一种新的K-Means初始化方法，使其在某些特定场景下比K-Means++表现更好。

<details>
<summary>参考答案</summary>

**思路**：K-Means++的核心是让初始中心分散。但在某些场景下，我们可以利用数据的额外信息。

**方案1：基于密度的初始化**

动机：K-Means++可能选中异常点作为初始中心（因为异常点距离远）。

```python
def density_based_initialization(X, k, n_neighbors=5):
    """
    基于局部密度的初始化
    选择高密度区域的点作为初始中心
    """
    from sklearn.neighbors import NearestNeighbors
    
    # 计算每个点的局部密度（近邻距离倒数）
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    distances, _ = nbrs.kneighbors(X)
    avg_distances = distances.mean(axis=1)
    densities = 1 / (avg_distances + 1e-10)
    
    # 选择密度最高的k个点，但要保证彼此距离足够远
    centroids = []
    candidate_indices = np.argsort(densities)[::-1]  # 按密度降序
    
    for idx in candidate_indices:
        if len(centroids) == k:
            break
        
        # 检查与已选中心的距离
        if len(centroids) == 0:
            centroids.append(X[idx])
        else:
            min_dist = min([np.linalg.norm(X[idx] - c) for c in centroids])
            if min_dist > np.median(avg_distances):  # 距离阈值
                centroids.append(X[idx])
    
    # 如果选不够k个，用K-Means++补充
    if len(centroids) < k:
        from sklearn.cluster import KMeans
        remaining = k - len(centroids)
        kmeans_temp = KMeans(n_clusters=remaining, init='k-means++', n_init=1)
        kmeans_temp.fit(X)
        centroids.extend(kmeans_temp.cluster_centers_)
    
    return np.array(centroids)
```

**方案2：基于PCA的初始化**

动机：在数据具有明显主方向时，沿主方向分布中心可能更好。

```python
def pca_based_initialization(X, k):
    """
    基于PCA的初始化
    沿主成分方向均匀分布初始中心
    """
    from sklearn.decomposition import PCA
    
    # PCA降维到主成分方向
    pca = PCA(n_components=min(k-1, X.shape[1]))
    X_pca = pca.fit_transform(X)
    
    # 在每个主成分方向上均匀选择分位点
    centroids_pca = []
    for i in range(min(k-1, X.shape[1])):
        component = X_pca[:, i]
        percentiles = np.percentile(component, np.linspace(0, 100, k))
        centroids_pca.append(percentiles)
    
    # 组合并映射回原空间
    # ...（实现略）
    
    return centroids
```

**方案3：最远点采样（Farthest Point Sampling）**

```python
def farthest_point_sampling(X, k):
    """
    最远点采样初始化
    每次选择与已选中心距离最远的点
    """
    n_samples = X.shape[0]
    centroids = [X[np.random.choice(n_samples)]]
    
    for _ in range(1, k):
        distances = np.array([
            min([np.linalg.norm(x - c)**2 for c in centroids])
            for x in X
        ])
        farthest_idx = np.argmax(distances)
        centroids.append(X[farthest_idx])
    
    return np.array(centroids)
```

**对比实验**：

```python
# 比较不同初始化方法
methods = {
    'random': 'random',
    'k-means++': 'k-means++',
    'farthest': farthest_point_sampling,
    'density': density_based_initialization
}

results = {}
for name, init in methods.items():
    if callable(init):
        # 自定义初始化
        centroids = init(X, k=5)
        kmeans = KMeans(n_clusters=5, init=centroids, n_init=1)
    else:
        kmeans = KMeans(n_clusters=5, init=init, n_init=10)
    
    kmeans.fit(X)
    results[name] = kmeans.inertia_

print(results)
```

</details>

---

### 练习22：分布式K-Means ⭐⭐⭐

**问题**：设计一个分布式K-Means算法，使其能够处理单机无法容纳的大规模数据集。

<details>
<summary>参考答案</summary>

**MapReduce风格的分布式K-Means**：

**核心思想**：
- 分配步可以并行（每个样本独立计算）
- 更新步可以并行（每个簇独立计算）

**算法流程**：

```python
"""
分布式K-Means（MapReduce风格）

假设数据分布在M台机器上
"""

class DistributedKMeans:
    def __init__(self, k=5, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
    
    def map_assignment(self, X_shard):
        """
        Map阶段：每个worker计算本地样本的分配
        输入：数据分片 X_shard
        输出：(cluster_id, (point, 1)) 的列表
        """
        distances = np.linalg.norm(X_shard[:, np.newaxis] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # 输出格式：(cluster_id, (point_sum, count))
        results = []
        for i in range(len(X_shard)):
            results.append((labels[i], (X_shard[i], 1)))
        
        return results
    
    def reduce_update(self, mapped_results):
        """
        Reduce阶段：聚合每个簇的统计信息
        输入：所有worker的map结果
        输出：新的中心点
        """
        # 按cluster_id分组并聚合
        cluster_stats = {i: (np.zeros_like(self.centroids[0]), 0) 
                        for i in range(self.k)}
        
        for cluster_id, (point, count) in mapped_results:
            current_sum, current_count = cluster_stats[cluster_id]
            cluster_stats[cluster_id] = (current_sum + point, current_count + count)
        
        # 计算新中心
        new_centroids = np.array([
            cluster_stats[i][0] / cluster_stats[i][1] 
            if cluster_stats[i][1] > 0 else self.centroids[i]
            for i in range(self.k)
        ])
        
        return new_centroids
    
    def fit(self, X_shards):
        """
        训练过程
        X_shards: 分布在多个worker上的数据分片列表
        """
        # 初始化：从第一个分片中随机采样
        self.centroids = X_shards[0][np.random.choice(
            len(X_shards[0]), self.k, replace=False
        )]
        
        for iteration in range(self.max_iter):
            # Map阶段（并行）
            all_mapped = []
            for shard in X_shards:
                mapped = self.map_assignment(shard)
                all_mapped.extend(mapped)
            
            # Reduce阶段
            new_centroids = self.reduce_update(all_mapped)
            
            # 检查收敛
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
        
        return self
```

**使用Spark MLlib的简化版**：

```python
from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DistributedKMeans").getOrCreate()

# 加载大规模数据
df = spark.read.format("parquet").load("hdfs://data/large_dataset.parquet")

# 分布式K-Means
kmeans = KMeans(k=10, initMode='k-means||')  # k-means||是K-Means++的分布式版本
model = kmeans.fit(df)

# 保存模型
model.save("hdfs://models/kmeans_model")
```

**k-means||初始化**：

这是K-Means++的分布式版本：
1. 每轮迭代中，每个worker独立采样候选中心
2. 收集所有候选中心，计算权重
3. 重复多轮，最终得到K个中心

优势：
- 通信开销小（每轮只需传输候选中心，不是全部数据）
- 理论保证与K-Means++类似

</details>

---

### 练习23：K-Means在推荐系统中的应用 ⭐⭐⭐

**问题**：设计一个基于K-Means的推荐系统方案。如何利用用户聚类进行个性化推荐？

<details>
<summary>参考答案</summary>

**方案设计**：

**步骤1：用户特征工程**
```python
def build_user_features(interactions):
    """
    从用户行为数据构建特征
    """
    features = {}
    for user_id, items in interactions.items():
        features[user_id] = {
            'avg_rating': np.mean([r for _, r in items]),
            'rating_std': np.std([r for _, r in items]),
            'num_items': len(items),
            'genre_pref': compute_genre_preference(items),
            'time_pattern': compute_time_pattern(items),
            'price_sensitivity': compute_price_sensitivity(items)
        }
    return features
```

**步骤2：用户聚类**
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 构建用户-特征矩阵
user_features_matrix = np.array([...])  # (n_users, n_features)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(user_features_matrix)

# K-Means聚类
kmeans = KMeans(n_clusters=20, random_state=42)
user_clusters = kmeans.fit_predict(X_scaled)

# 每个簇代表一类用户
# 例如：高价值用户、价格敏感用户、新品尝试者等
```

**步骤3：簇内推荐**
```python
class ClusterBasedRecommendation:
    def __init__(self, kmeans_model, interactions):
        self.kmeans = kmeans_model
        self.interactions = interactions
        self.cluster_preferences = self._compute_cluster_preferences()
    
    def _compute_cluster_preferences(self):
        """计算每个簇的偏好"""
        preferences = {}
        for cluster_id in range(self.kmeans.n_clusters):
            cluster_users = [u for u, c in user_clusters.items() if c == cluster_id]
            
            # 聚合该簇用户的评分
            item_scores = defaultdict(list)
            for user in cluster_users:
                for item, rating in self.interactions[user]:
                    item_scores[item].append(rating)
            
            # 计算平均评分
            preferences[cluster_id] = {
                item: np.mean(scores) 
                for item, scores in item_scores.items()
            }
        
        return preferences
    
    def recommend(self, user_id, user_features, n_recommendations=10):
        """为用户推荐"""
        # 确定用户所属簇
        user_cluster = self.kmeans.predict([user_features])[0]
        
        # 获取该簇的热门物品
        cluster_pref = self.cluster_preferences[user_cluster]
        
        # 排除用户已交互过的物品
        seen_items = set(item for item, _ in self.interactions[user_id])
        candidates = {
            item: score 
            for item, score in cluster_pref.items() 
            if item not in seen_items
        }
        
        # 返回Top-N
        return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
```

**优势与局限**：

| 优势 | 局限 |
|:---|:---|
| 可解释性强 | 冷启动问题 |
| 计算高效 | 簇内个性化不足 |
| 适合大规模用户 | 需要特征工程 |

**改进方向**：
1. 结合协同过滤（混合推荐）
2. 使用软分配（用户属于多个簇的概率）
3. 动态更新聚类（适应用户兴趣变化）

</details>

---

## 编程实践题

### 练习24：完整项目——客户分群分析

**目标**：使用K-Means对电商客户进行分群，并给出营销策略建议。

**数据集**：
```python
import pandas as pd
import numpy as np

# 模拟客户数据
np.random.seed(42)
n_customers = 1000

data = pd.DataFrame({
    'customer_id': range(n_customers),
    'age': np.random.normal(35, 10, n_customers).clip(18, 70),
    'annual_income': np.random.lognormal(10.5, 0.5, n_customers),
    'purchase_frequency': np.random.poisson(12, n_customers),
    'avg_order_value': np.random.lognormal(4, 0.6, n_customers),
    'days_since_last_purchase': np.random.exponential(30, n_customers),
    'total_spent': np.random.lognormal(12, 0.8, n_customers)
})
```

**任务清单**：
1. 数据探索与可视化
2. 特征工程（RFM分析：Recency, Frequency, Monetary）
3. 数据标准化
4. 使用肘部法则和轮廓系数选择K
5. K-Means聚类
6. 分析每个簇的特征
7. 为每个簇制定营销策略
8. 可视化结果

**参考实现**：见本章代码文件 `kmeans_clustering.py`

---

### 练习25：算法对比实验

**目标**：在同一数据集上比较以下聚类算法：
1. K-Means
2. K-Means++
3. Mini-Batch K-Means
4. 层次聚类
5. DBSCAN

**评估指标**：
- 轮廓系数
- Calinski-Harabasz指数
- Davies-Bouldin指数
- 运行时间

**数据集**：
```python
from sklearn.datasets import make_moons, make_blobs, make_circles

# 生成不同类型的数据
datasets = {
    'blobs': make_blobs(n_samples=300, centers=4, random_state=42)[0],
    'moons': make_moons(n_samples=300, noise=0.1, random_state=42)[0],
    'circles': make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)[0]
}
```

**输出**：对比表格和可视化

---

## 思考题

### 练习26：开放问题

1. **K-Means的K值一定需要预先指定吗？** 是否有方法可以自动确定K值？

2. **如果数据是流式的（不断有新数据到来）**，如何设计在线K-Means算法？

3. **K-Means假设簇是球形的**，如果数据具有复杂的流形结构（如瑞士卷），应该如何处理？

4. **在隐私保护场景下**，如何在不解密数据的情况下进行K-Means聚类？（提示：同态加密、安全多方计算）

5. **K-Means可以用于半监督学习吗？** 如何利用少量标注信息指导聚类？

---

## 参考答案汇总

| 题号 | 难度 | 考点 |
|:---:|:---:|:---|
| 1 | ⭐ | 算法步骤、历史 |
| 2 | ⭐ | 损失函数 |
| 3 | ⭐ | 收敛性 |
| 4 | ⭐ | K-Means++ |
| 5 | ⭐ | 选择K值 |
| 6 | ⭐ | 轮廓系数 |
| 7 | ⭐ | 算法比较 |
| 8 | ⭐ | 概率计算 |
| 9 | ⭐ | 初始化影响 |
| 10 | ⭐ | Mini-Batch |
| 11 | ⭐⭐ | 均值最优性证明 |
| 12 | ⭐⭐ | 空簇问题 |
| 13 | ⭐⭐ | 特征缩放 |
| 14 | ⭐⭐ | K-Means与GMM |
| 15 | ⭐⭐ | 核K-Means |
| 16 | ⭐⭐ | 时间复杂度 |
| 17 | ⭐⭐ | 图像量化 |
| 18 | ⭐⭐ | 异常检测 |
| 19 | ⭐⭐⭐ | 完整实现 |
| 20 | ⭐⭐⭐ | 收敛性证明 |
| 21 | ⭐⭐⭐ | 新初始化方法 |
| 22 | ⭐⭐⭐ | 分布式K-Means |
| 23 | ⭐⭐⭐ | 推荐系统 |

---

*练习题完。建议先独立完成，再对照参考答案学习。*
