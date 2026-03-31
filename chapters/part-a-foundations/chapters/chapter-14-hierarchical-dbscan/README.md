# 第十四章：层次聚类与DBSCAN——数据的家谱与邻居

> "物以类聚，人以群分。"——《易经·系辞上》

小朋友们，还记得我们在上一章学习的K-Means聚类吗？它像一位急性子的图书管理员，把书分成几堆就完事了。但是，如果我们要知道书与书之间的"亲戚关系"呢？比如《哈利波特》和《魔戒》都属于奇幻小说，而奇幻小说又属于更广泛的"虚构文学"大家族。这时候，我们就需要一种能像画"家谱"一样的聚类方法——**层次聚类**！

还有，如果数据分布像天上的云朵一样，有的密集、有的稀疏，K-Means就会犯迷糊。这时候，一个聪明的"找朋友"算法**DBSCAN**就派上用场了！

准备好了吗？让我们开始这场探索数据家族关系的奇妙之旅！

---

## 费曼比喻速览 🎯

在深入学习之前，让我们先用三个生活化的比喻来理解本章的核心概念：

### 比喻1：书架整理法（层次聚类）
想象你在整理书架：
- **AGNES**：先把每本书单独放，然后把最相似的逐渐合并（科幻和奇幻合并为"想象类"，再和推理合并为"小说类"...）
- **DIANA**：先把所有书放在一起，然后逐步细分（先按语言分，再按类型分...）

### 比喻2：朋友圈游戏（DBSCAN）
想象你在新学校找朋友：
- **核心点**：身边有至少3个朋友的人（人气王）
- **边界点**：虽然朋友不多，但在人气王的朋友圈范围内
- **噪声点**：孤零零一个人，不在任何人的朋友圈里

### 比喻3：归属感指数（轮廓系数）
想象你参加社团：
- **归属感≈1**：和社团成员很熟，和其他社团的人很疏远（完美归属）
- **归属感≈0**：在两个社团之间摇摆不定（边界成员）
- **归属感≈-1**：可能去错了社团（分错簇了）

---

## 第一部分：层次聚类——数据的家谱

### 14.1 故事引入：生物分类学的智慧

在18世纪，有一位名叫卡尔·林奈（Carl Linnaeus）的瑞典植物学家，他面临一个巨大的难题：地球上那么多生物，该怎么给它们分类呢？

林奈想出了一个绝妙的主意——**层级分类法**！他把生物分成：

```
界（Kingdom）
 └── 门（Phylum）
      └── 纲（Class）
           └── 目（Order）
                └── 科（Family）
                     └── 属（Genus）
                          └── 种（Species）
```

比如我们家的小狗，它的"家谱"是这样的：

```
动物界
 └── 脊索动物门
      └── 哺乳纲
           └── 食肉目
                └── 犬科
                     └── 犬属
                          └── 家犬（Canis lupus familiaris）
```

这种分类方法的神奇之处在于：**它告诉我们生物之间的亲疏关系**！狗和狼都属于犬属，它们是近亲；狗和猫虽然都属于食肉目，但关系就远一些了；狗和鱼的亲缘关系就更远了。

**层次聚类**正是受到了这种生物分类学的启发！它不像K-Means那样一次性把数据分成几堆，而是像画"家谱树"一样，告诉我们哪些数据点是最亲密的"兄弟姐妹"，哪些是较远的"表亲"。

### 14.2 两种构建家谱的方式

想象你有一堆乐高积木，每个积木代表一个数据点。现在你要把它们组织起来，有两种思路：

**方式一：自底向上（AGNES）**
先让每个积木自己站一边，然后把最近的两个积木粘在一起，再把粘好的小块逐步合并成更大的块……就像搭积木一样，从小变大！

**方式二：自顶向下（DIANA）**
先把所有积木看作一个大团体，然后找出"最不合群"的积木，把它分出去，再在新的团体中继续分裂……就像切蛋糕一样，从大变小！

让我们详细看看这两种方法！

### 14.3 AGNES：像搭积木一样聚类

AGNES的全称是**A**gglomerative **N**esting（凝聚嵌套），名字听起来有点复杂，但思路其实非常简单！

#### 算法步骤

```
步骤1：把每个数据点都看作一个独立的簇（N个簇）
步骤2：计算所有簇两两之间的距离
步骤3：找出距离最近的两个簇，把它们合并成一个新簇
步骤4：重新计算新簇与其他簇之间的距离
步骤5：重复步骤2-4，直到所有数据点都合并成一个簇
```

让我们用一个具体的例子来理解。假设我们有5个同学，记录了他们的身高和体重：

```
小明：(160cm, 50kg)
小红：(162cm, 52kg)
小刚：(180cm, 75kg)
小丽：(158cm, 48kg)
小强：(178cm, 72kg)
```

**第一轮**：计算所有同学之间的距离（用欧氏距离），发现小红和小丽最近，先把她们合并成一个簇。

```
簇A：[小红, 小丽]
簇B：[小明]
簇C：[小刚]
簇D：[小强]
```

**第二轮**：重新计算距离，发现小明和簇A最近，把小明也加进去。

```
簇A：[小红, 小丽, 小明]  ← 这是我们的"小个子组"
簇B：[小刚]
簇C：[小强]
```

**第三轮**：发现小刚和小强这两个"大个子"最近，把他们合并。

```
簇A：[小红, 小丽, 小明]
簇B：[小刚, 小强]  ← 这是我们的"大个子组"
```

**第四轮**：把两个大簇合并，所有人都聚在一起了！

#### 簇与簇之间的距离怎么算？

这里有个关键问题：当簇里有多个点时，怎么计算两个簇之间的距离呢？科学家们发明了四种主要的方法：

**1. 单链接（Single Linkage）——"最胆小鬼原则"**

```
簇A和簇B的距离 = min{d(a, b) | a∈A, b∈B}
```

意思是：看两个簇中**最近**的两个点的距离。就像两个害羞的人，只要有一对人是邻居，就认为两簇很接近。

```
      ● A1                    ● B1
         \                   /
          \    距离=5      /
           \____________/
           
    ● A2 ─────── ● B2  距离=3
           ↑
    单链接距离 = min(5, 3) = 3
```

**优点**：能处理形状不规则的簇，像链子一样延伸。
**缺点**：容易产生"链式反应"，把不该在一起的点连在一起。

**2. 全链接（Complete Linkage）——"最挑剔鬼原则"**

```
簇A和簇B的距离 = max{d(a, b) | a∈A, b∈B}
```

意思是：看两个簇中**最远**的两个点的距离。就像两个挑剔的人，必须所有人都靠近才算接近。

```
    ● A1                    ● B1
       \                   /
        \    距离=5      /
         \____________/
         
    ● A2 ─────── ● B2  距离=3
           ↑
    全链接距离 = max(5, 3) = 5
```

**优点**：产生比较紧凑的球形簇。
**缺点**：对异常值敏感，容易被"离群点"影响。

**3. 平均链接（Average Linkage）——"民主投票原则"**

```
簇A和簇B的距离 = avg{d(a, b) | a∈A, b∈B}
```

意思是：计算所有点对之间距离的**平均值**。这是最民主的方法，大家都算一票！

**优点**：介于单链接和全链接之间，比较平衡。
**缺点**：计算量稍大，但现代计算机完全能应付。

**4. Ward法——"最节俭原则"**

Ward法是最常用的方法，它的思想是：每次合并时，选择使**组内方差增加最小**的两个簇合并。

简单来说：Ward法喜欢把"同类"聚在一起，让簇内成员尽量相似。

#### 【数学小角落】Lance-Williams递推公式

当两个簇合并后，如何快速计算新簇与其他簇的距离？Lance和Williams在1967年提出了一个神奇的通用公式：

$$d(C_k, C_i \cup C_j) = \alpha_i \cdot d(C_k, C_i) + \alpha_j \cdot d(C_k, C_j) + \beta \cdot d(C_i, C_j) + \gamma \cdot |d(C_k, C_i) - d(C_k, C_j)|$$

不同方法的参数表：

| 方法 | αi | αj | β | γ |
|------|-----|-----|-----|-----|
| 单链接 | 1/2 | 1/2 | 0 | -1/2 |
| 全链接 | 1/2 | 1/2 | 0 | 1/2 |
| 平均链接 | \|Ci\|/(\|Ci\|+\|Cj\|) | \|Cj\|/(\|Ci\|+\|Cj\|) | 0 | 0 |
| Ward法 | (\|Ck\|+\|Ci\|)/(总和) | (\|Ck\|+\|Cj\|)/(总和) | -\|Ck\|/(总和) | 0 |

这个公式的好处是：**不需要重新计算所有点之间的距离**，大大节省了计算时间！

### 14.4 DIANA：像分蛋糕一样分裂

DIANA的全称是**DI**visive **AN**alysis **A**lgorithm（分裂分析法），它是AGNES的"反义词"！

#### 算法步骤

```
步骤1：把所有数据点看作一个大簇
步骤2：在当前的簇中，找出"最不合适"的点（离其他点最远的）
步骤3：把这些"不合适"的点分出去，形成新簇
步骤4：重复步骤2-3，直到每个点都自成一簇
```

**DIANA vs AGNES**：
- AGNES更简单、更常用
- DIANA更适合已知大致有几个簇的情况
- AGNES是自底向上，DIANA是自顶向下

### 14.5 树状图：家谱的可视化

层次聚类最酷的地方就是能画出一棵**树状图（Dendrogram）**！

```
Height
   │
 5 │                    ┌─────────────┐
   │                    │             │
 4 │            ┌───────┴───────┐     │
   │            │               │     │
 3 │    ┌───────┴───────┐       │     │
   │    │               │       │     │
 2 │  ┌─┴─┐           ┌─┴─┐     │     │
   │  │   │           │   │     │     │
 1 │  A   B           C   D     E     F
   │
   └──────────────────────────────────────
     小红 小丽   小明  小刚  小强
```

**如何确定聚类数？**

在树状图上画一条水平线，穿过几条竖线，就能得到几个簇！

```
Height
   │
 3 │══════════════ 切一刀
   │    ┌───────┴───────┐
 2 │  ┌─┴─┐           ┌─┴─┐
   │  A   B           C   D
   └──────────────────────────
          簇1          簇2    ← 2个簇！
```

### 14.6 层次聚类的优缺点

**优点：**
1. ✅ 不需要预先指定聚类数K
2. ✅ 提供丰富的层次信息，可解释性强
3. ✅ 树状图直观易懂
4. ✅ 可以捕捉簇的嵌套关系

**缺点：**
1. ❌ 计算复杂度高，通常是O(n³)或O(n²logn)
2. ❌ 一旦合并或分裂，就不能撤销
3. ❌ 对噪声和异常值敏感
4. ❌ 大数据集时效率较低

---

## 第二部分：DBSCAN密度聚类——数据的邻居

### 14.7 故事引入：找朋友的游戏

想象一下，你刚转学到一所新学校，老师让你们通过"找朋友"的游戏来互相认识。规则是这样的：

1. **核心同学**：身边有至少3个同学站在距离你1米之内，你就是"核心同学"
2. **边界同学**：虽然你身边不够3个同学，但你站在某个"核心同学"的1米范围内
3. **孤独同学**：你身边没有同学，也不在任何一个"核心同学"的范围内

这个游戏的结果会怎样？
- **核心同学**会形成一个个小团体
- **边界同学**会成为某个小团体的"边缘成员"
- **孤独同学**可能站在角落，不属于任何团体

DBSCAN算法正是基于这个"找朋友"的思想！

### 14.8 DBSCAN的核心概念

#### ε-邻域（Epsilon Neighborhood）

想象你站在操场中央，画一个半径为ε（epsilon）的圆圈，圆圈里的所有人就构成了你的**ε-邻域**！

数学定义：
$$N_\varepsilon(p) = \{q \in D \mid distance(p, q) \leq \varepsilon\}$$

#### 最小点数（MinPts）

光有人还不够，我们需要足够多的"朋友"才算一个"核心人物"。**MinPts**就是这个门槛值。

```
如果 |Nε(p)| ≥ MinPts：
    p是核心点
否则：
    p不是核心点
```

#### 三类点的精确定义

**核心点（Core Point）**：
$$|N_\varepsilon(p)| \geq \text{MinPts}$$

**边界点（Border Point）**：不是核心点，但在某个核心点的ε-邻域内

**噪声点（Noise Point）**：既不是核心点也不是边界点

```
              ● 核心点
            / | \
           /  |  \
          ○   ○   ○  ← 边界点
              
    ○  ○        ○  ← 噪声点（散落在远处）
    
┌───────────────────────┐
│   一个完整的簇！       │
│   包含1个核心点        │
│   和3个边界点          │
└───────────────────────┘
```

### 14.9 DBSCAN算法步骤详解

```
DBSCAN(D, ε, MinPts)
    C = 0                                    // 簇的编号
    标记所有点为"未访问"
    
    for each point p in D:
        if p已经访问过:
            continue
        标记p为"已访问"
        
        Neighbors = RegionQuery(p, ε)        // 找出p的ε-邻域
        
        if |Neighbors| < MinPts:             // p不是核心点
            标记p为"噪声"
        else:                                // p是核心点，发现新簇！
            C = C + 1
            ExpandCluster(p, Neighbors, C, ε, MinPts)
    
    return 簇的集合
```

### 14.10 密度可达与密度相连

**密度直达（Directly Density-Reachable）**：
$$q \text{ 从 } p \text{ 密度直达} \Leftrightarrow p \text{ 是核心点} \land q \in N_\varepsilon(p)$$

**密度可达（Density-Reachable）**：通过一串核心点可以"接力"到达

**密度相连（Density-Connected）**：两个点可以通过某个共同的核心点连接起来

### 14.11 参数选择技巧

**如何选择MinPts？**
- 经验法则：MinPts ≥ 维度数 + 1
- 通常设置MinPts = 2×维度数
- 更大的MinPts对噪声更鲁棒

**如何选择ε？——K-距离图法**

1. 对每个点，计算它到第k个最近邻居的距离（k = MinPts - 1）
2. 把这些距离从小到大排序
3. 画出K-距离曲线
4. 找到曲线的"拐点"（elbow），对应的距离就是ε

```
K-距离
  │        ╱
  │       ╱
  │      ╱ ← 拐点在这里！ε ≈ 这个值
  │     ╱
  │    ╱
  │___╱
  └─────────────────
    排序后的点
```

### 14.12 DBSCAN vs K-Means

| 特性 | K-Means | DBSCAN |
|------|---------|--------|
| 簇的形状 | 只能发现球形簇 | 能发现任意形状的簇 |
| 聚类数 | 需要预先指定K | 自动确定簇的数量 |
| 噪声处理 | 所有点都必须属于某个簇 | 能自动识别噪声点 |
| 数据分布 | 适合分布均匀的数据 | 适合密度不均匀的数据 |
| 计算复杂度 | O(n×K×I) | O(n×log n)（使用空间索引） |
| 参数 | K | ε, MinPts |

---

## 第三部分：聚类评估指标——给聚类打分

### 14.13 为什么需要评估聚类？

聚类是无监督学习，没有"正确答案"。但我们需要知道：
- 聚类质量如何？
- 参数选择是否合理？
- 不同的聚类方法哪个更好？

**好的聚类应该满足**：
1. **紧密度（Compactness）**：同一个簇内的点应该很接近
2. **分离度（Separation）**：不同簇之间的点应该很疏远

### 14.14 Silhouette Score（轮廓系数）

轮廓系数是最常用的聚类评估指标之一。

对于每个数据点：
- **a**：到同簇其他点的平均距离（越小越好）
- **b**：到最近其他簇的平均距离（越大越好）

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

**取值范围**：-1 ≤ s(i) ≤ 1

- **s(i) ≈ 1**：完美聚类！
- **s(i) ≈ 0**：点在两个簇的边界上
- **s(i) ≈ -1**：点可能被分到了错误的簇

### 14.15 其他评估指标

**Dunn Index**：
$$\text{Dunn} = \frac{\min(\text{簇间距离})}{\max(\text{簇内直径})}$$
越大越好！

**Davies-Bouldin Index**：
$$DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left(\frac{s_i + s_j}{d_{ij}}\right)$$
越小越好！

**肘部法则（Elbow Method）**：
通过SSE-K曲线找到"肘部"，确定最佳聚类数。

---

## 第四部分：实战项目

### 14.16 客户分群分析

假设你是一家超市的数据分析师，老板给你了一份客户消费数据，包含：
- 客户的年龄
- 年收入
- 年度消费金额

**分析步骤**：
1. 数据探索与预处理
2. 使用层次聚类观察树状图
3. 使用DBSCAN自动发现簇和噪声
4. 使用轮廓系数评估聚类效果
5. 为每个客户群体制定营销策略

### 14.17 代码实践

本章配套的代码文件：

| 文件 | 内容 |
|------|------|
| `code/hierarchical_numpy.py` | 层次聚类NumPy实现（AGNES + DIANA） |
| `code/hierarchical_torch.py` | 层次聚类PyTorch实现（GPU加速） |
| `code/dbscan_numpy.py` | DBSCAN NumPy实现 |

---

## 14.18 本章小结

在这一章中，我们学习了两种重要的聚类方法：

**层次聚类**：
- AGNES（自底向上）：像搭积木一样逐步合并
- DIANA（自顶向下）：像切蛋糕一样逐步分裂
- 四种链接方法：单链接、全链接、平均链接、Ward法
- Lance-Williams递推公式
- 树状图可视化

**DBSCAN密度聚类**：
- 核心概念：ε-邻域、MinPts、核心点、边界点、噪声点
- 密度可达性：直接密度可达、密度可达、密度相连
- 参数选择：K-距离图法
- 自动识别噪声和任意形状簇

**聚类评估**：
- 轮廓系数：衡量紧密度和分离度
- Dunn Index、Davies-Bouldin Index
- 肘部法则

**算法对比**：
- K-Means：适合球形簇，需要指定K
- 层次聚类：提供层次结构，不需要指定K
- DBSCAN：能发现任意形状，自动识别噪声

---

## 参考文献

1. Johnson, S. C. (1967). Hierarchical clustering schemes. *Psychometrika*, 32(3), 241-254.
2. Lance, G. N., & Williams, W. T. (1967). A general theory of classificatory sorting strategies. *The Computer Journal*, 9(4), 373-380.
3. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *KDD'96* (pp. 226-231).
4. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.
5. Dunn, J. C. (1974). Well-separated clusters and optimal fuzzy partitions. *Journal of Cybernetics*, 4(1), 95-104.
6. Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. *IEEE TPAMI*, (2), 224-227.
7. Calinski, T., & Harabasz, J. (1974). A dendrite method for cluster analysis. *Communications in Statistics*, 3(1), 1-27.
8. Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *JASA*, 58(301), 236-244.
9. Kaufman, L., & Rousseeuw, P. J. (1990). *Finding groups in data*. Wiley.
10. Schubert, E., et al. (2017). DBSCAN revisited, revisited. *ACM TODS*, 42(3), 1-21.

---

## 附录A：数学推导详解

### A.1 Lance-Williams递推公式完整推导

Lance-Williams公式是层次聚类的核心数学工具，它让我们能够递推计算合并后的簇间距离，而不用重新计算所有点对的距离。

**问题设定**：
- 簇 $C_i$ 和 $C_j$ 合并为新簇 $C_k$
- 需要计算 $C_k$ 与任意其他簇 $C_m$ 的距离 $d(C_k, C_m)$

**通用递推公式**：

$$d(C_k, C_m) = \alpha_i d(C_i, C_m) + \alpha_j d(C_j, C_m) + \beta d(C_i, C_j) + \gamma |d(C_i, C_m) - d(C_j, C_m)|$$

**四种链接方法的参数**：

| 链接方法 | $\alpha_i$ | $\alpha_j$ | $\beta$ | $\gamma$ |
|:---:|:---:|:---:|:---:|:---:|
| 单链接 | $\frac{1}{2}$ | $\frac{1}{2}$ | $0$ | $-\frac{1}{2}$ |
| 全链接 | $\frac{1}{2}$ | $\frac{1}{2}$ | $0$ | $+\frac{1}{2}$ |
| 平均链接 | $\frac{n_i}{n_i+n_j}$ | $\frac{n_j}{n_i+n_j}$ | $0$ | $0$ |
| Ward法 | $\frac{n_i+n_m}{n_i+n_j+n_m}$ | $\frac{n_j+n_m}{n_i+n_j+n_m}$ | $\frac{-n_m}{n_i+n_j+n_m}$ | $0$ |

**单链接的推导**：

$$d_{\text{single}}(C_k, C_m) = \min_{x \in C_i \cup C_j, y \in C_m} d(x, y)$$

$$= \min\left\{\min_{x \in C_i, y \in C_m} d(x, y), \min_{x \in C_j, y \in C_m} d(x, y)\right\}$$

$$= \min\{d(C_i, C_m), d(C_j, C_m)\}$$

$$= \frac{1}{2}d(C_i, C_m) + \frac{1}{2}d(C_j, C_m) - \frac{1}{2}|d(C_i, C_m) - d(C_j, C_m)|$$

**Ward法的推导**：

Ward法使用平方欧氏距离，最小化合并导致的SSE增量：

$$\Delta(C_i, C_j) = \frac{n_i n_j}{n_i + n_j} \|\mu_i - \mu_j\|^2$$

其中 $\mu_i$ 和 $\mu_j$ 分别是两个簇的质心。

合并后的新质心：
$$\mu_k = \frac{n_i \mu_i + n_j \mu_j}{n_i + n_j}$$

### A.2 DBSCAN密度可达性证明

**定理**：如果 $p$ 是核心点，$q$ 是从 $p$ 密度可达的，那么 $q$ 属于以 $p$ 为核心的同一个簇。

**证明**：

设存在点链 $p_1, p_2, ..., p_n$，其中 $p_1 = p$, $p_n = q$，且每个 $p_{i+1}$ 都从 $p_i$ 直接密度可达。

由直接密度可达的定义：
$$p_{i+1} \in N_\varepsilon(p_i) \text{ 且 } |N_\varepsilon(p_i)| \geq \text{MinPts}$$

因此 $p_i$ 都是核心点，这条链上的所有点都属于同一个簇。

### A.3 轮廓系数的性质

**性质1**：$-1 \leq s(i) \leq 1$

**证明**：
$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

- 当 $a(i) \to 0$ 时，$s(i) \to 1$（完美聚类）
- 当 $a(i) = b(i)$ 时，$s(i) = 0$（边界情况）
- 当 $b(i) \to 0$ 时，$s(i) \to -1$（错误聚类）

**性质2**：全局轮廓系数是各点轮廓系数的平均

$$\text{Silhouette}_{\text{avg}} = \frac{1}{n} \sum_{i=1}^{n} s(i)$$

### A.4 复杂度分析

**AGNES时间复杂度**：
- 朴素实现：$O(n^3)$ — 每次合并需要重新计算所有簇间距离
- 优化实现（优先队列）：$O(n^2 \log n)$
- 空间复杂度：$O(n^2)$ — 存储距离矩阵

**DBSCAN时间复杂度**：
- 朴素实现：$O(n^2)$ — 对每个点计算ε-邻域
- 空间索引（KD-Tree/R-Tree）：$O(n \log n)$
- 空间复杂度：$O(n)$ — 仅需存储点的标签

---

## 附录B：算法伪代码

### B.1 AGNES算法伪代码

```
算法: AGNES(D, k)
输入: 数据集 D = {x_1, x_2, ..., x_n}, 目标簇数 k
输出: 聚类结果 C = {C_1, C_2, ..., C_k}

1. 初始化: 每个点作为一个簇
   C = {{x_1}, {x_2}, ..., {x_n}}
   
2. 计算初始距离矩阵 M
   对于每对簇 C_i, C_j:
       M[i,j] = d(C_i, C_j)  // 使用选定的链接方法

3. 当 |C| > k 时:
   a. 在 M 中找到距离最近的两个簇 C_i, C_j
   b. 合并: C_new = C_i ∪ C_j
   c. 从 C 中移除 C_i 和 C_j
   d. 将 C_new 加入 C
   e. 更新距离矩阵 M:
      对于 C 中的每个其他簇 C_m:
          M[new, m] = LanceWilliams(C_new, C_m, M)

4. 返回 C
```

### B.2 DBSCAN算法伪代码

```
算法: DBSCAN(D, ε, MinPts)
输入: 数据集 D, 半径 ε, 最小点数 MinPts
输出: 聚类标签 labels, 噪声点集合 Noise

1. 初始化: 所有点标记为未访问
   labels = [-1] * n
   cluster_id = 0

2. 对于 D 中的每个未访问点 p:
   a. 标记 p 为已访问
   b. neighbors = ε-邻域查询(p)
   c. 如果 |neighbors| < MinPts:
      标记 p 为噪声
   d. 否则:
      开始新簇: cluster_id += 1
      labels[p] = cluster_id
      种子集 seeds = neighbors \ {p}
      
      当 seeds 不为空:
         q = seeds.pop()
         如果 q 是未访问:
            标记 q 为已访问
            neighbors_q = ε-邻域查询(q)
            如果 |neighbors_q| >= MinPts:
               seeds = seeds ∪ neighbors_q
         如果 q 是噪声或未分配:
            labels[q] = cluster_id

3. 返回 labels
```

---

## 附录C：Python实现要点

### C.1 距离矩阵计算优化

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

def compute_distance_matrix(X):
    """
    计算距离矩阵的优化方法
    时间复杂度: O(n^2)
    """
    # 方法1: 使用scipy (推荐)
    condensed_dist = pdist(X, metric='euclidean')
    dist_matrix = squareform(condensed_dist)
    return dist_matrix
    
    # 方法2: 向量化计算
    # diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    # dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    # return dist_matrix
```

### C.2 优先队列优化AGNES

```python
import heapq

def agnes_optimized(X, k, linkage='single'):
    """
    使用优先队列优化的AGNES
    时间复杂度: O(n^2 log n)
    """
    n = len(X)
    # 计算初始距离矩阵
    dist_matrix = compute_distance_matrix(X)
    
    # 初始化优先队列
    heap = []
    for i in range(n):
        for j in range(i+1, n):
            heapq.heappush(heap, (dist_matrix[i,j], i, j))
    
    # 并查集跟踪簇归属
    parent = list(range(n))
    
    # ... 合并逻辑
```

### C.3 DBSCAN的空间索引优化

```python
from sklearn.neighbors import KDTree

def dbscan_optimized(X, eps, min_pts):
    """
    使用KD-Tree加速的DBSCAN
    时间复杂度: O(n log n) 平均情况
    """
    tree = KDTree(X)
    n = len(X)
    labels = np.full(n, -1)  # -1表示未分类
    visited = np.zeros(n, dtype=bool)
    
    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
            
        # KD-Tree快速查询ε-邻域
        neighbors = tree.query_radius([X[i]], r=eps)[0]
        
        if len(neighbors) < min_pts:
            labels[i] = -1  # 噪声
            visited[i] = True
        else:
            # 扩展簇...
            pass
    
    return labels
```

---

## 附录D：与其他聚类算法的对比

| 特性 | K-Means | 层次聚类 | DBSCAN | GMM |
|:---:|:---:|:---:|:---:|:---:|
| **时间复杂度** | O(n×k×i) | O(n²)~O(n³) | O(n log n) | O(n×k×i) |
| **空间复杂度** | O(n+k) | O(n²) | O(n) | O(n+k) |
| **需指定K** | ✅ 必须 | ❌ 不需要 | ❌ 不需要 | ✅ 必须 |
| **形状适应** | 球形 | 任意 | 任意 | 椭圆形 |
| **噪声处理** | ❌ 不能 | ❌ 不能 | ✅ 自动识别 | ❌ 不能 |
| **可解释性** | 中 | 高（树状图） | 中 | 低 |
| **大数据集** | ✅ 适合 | ❌ 不适合 | ⚠️ 中等 | ✅ 适合 |
| **确定性** | 随机初始化 | 确定 | 确定 | EM可能收敛到局部最优 |

**选择指南**：
- 数据量小、需要可解释性 → **层次聚类**
- 有噪声、簇形状不规则 → **DBSCAN**
- 大数据集、球形簇 → **K-Means**
- 需要概率解释 → **GMM**
- 不确定时 → 先用DBSCAN探索，再用K-Means验证

---

## 练习与实践

完成本章学习后，请尝试：
1. 完成 `exercises.md` 中的9道练习题
2. 运行 `code/` 目录下的代码示例
3. 使用自己的数据进行聚类分析
4. 尝试改进算法（如实现OPTICS算法）

---

*本章完。在下一章中，我们将探索更高级的聚类技术！*
