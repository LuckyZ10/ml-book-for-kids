# 第十四章：层次聚类与DBSCAN——探索数据的隐秘结构

> **本章目标**：理解层次聚类和密度聚类的核心思想，掌握AGNES、DIANA和DBSCAN算法的原理与实现，能够评估聚类质量。

---

## 开篇故事：宇宙学家的难题 🌌

1960年代的剑桥大学，一位年轻的宇宙学家**斯蒂芬·约翰逊（Stephen C. Johnson）**正面临一个困扰他已久的问题。

他的桌子上堆满了来自射电望远镜的数据——数以千计的类星体位置坐标。科学界想知道：这些类星体是均匀分布在宇宙中的，还是形成了某种**结构**？如果形成结构，它们又是如何组织在一起的？

"也许，"约翰逊自言自语道，"我可以从最近的类星体开始，逐步把它们组合成越来越大的群体。"

他拿起铅笔，在纸上画出了第一个**树状图（Dendrogram）**——一种展示数据层次结构的图形。从最底层的单个类星体开始，逐步合并，直到所有类星体都连接在一起。这幅图看起来像是一棵倒置的树。

1967年，约翰逊在《Psychometrika》上发表了《Hierarchical clustering schemes》这篇论文。他提出了一个优雅的框架，将各种层次聚类方法统一在一个理论之下。这篇论文成为了层次聚类领域的奠基之作。

与此同时，在大西洋彼岸的慕尼黑工业大学，四位科学家**Ester、Kriegel、Sander和Xu**正在研究一个不同但相关的问题：

> **传统的聚类方法假设簇是球形的，但现实世界的数据往往有更复杂的形状。**

1996年，在波特兰举行的KDD会议上，他们提出了**DBSCAN**算法——一种基于密度的聚类方法。这个算法革命性地改变了人们对聚类的理解：簇不再是基于距离的球形区域，而是**密度相连**的点集。

这两种方法——层次聚类和密度聚类——代表了无监督学习中最深刻的洞察：**数据的结构可能比我们先验的假设更加复杂和有趣**。

---

## 费曼四步检验框 📚

让我们用费曼学习法来预览本章的核心概念：

```
┌─────────────────────────────────────────────────────────────────┐
│                    🔍 费曼四步检验法                             │
├─────────────────────────────────────────────────────────────────┤
│ 1️⃣ 选择概念：层次聚类与DBSCAN密度聚类                            │
│                                                                 │
│ 2️⃣ 教给别人：想象你在向一个小学生解释...                          │
│    "层次聚类就像是整理你的书架——你可以把书按大小先分成几摞，      │
│     然后再把相似的书放在一起，直到整个书架井井有条。"              │
│                                                                 │
│    "DBSCAN就像是找人群中的朋友——朋友是离你很近的人，朋友的朋友    │
│     也是你的朋友。离你很远、没有朋友的人就是孤独者。"               │
│                                                                 │
│ 3️⃣ 发现差距：当你说"Linkage"时，需要解释清楚：                    │
│    - Single linkage：最短的桥                                     │
│    - Complete linkage：最长的桥                                   │
│    - Average linkage：平均距离                                    │
│    - Ward's method：最小化方差                                    │
│                                                                 │
│ 4️⃣ 简化语言：用生活化的比喻替代专业术语                            │
│    "层次聚类" → "逐步分组法"                                      │
│    "密度可达" → "朋友圈连接"                                      │
│    "噪声点" → "孤独的人"                                          │
│    "轮廓系数" → "归属感指数"                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 14.1 层次聚类的两种哲学

### 14.1.1 什么是层次聚类？

想象你是一位博物馆馆长，要把一批文物分类展示。你有两种思路：

**思路一（自底向上）**：先把每件文物单独摆放，然后把最相似的逐渐合并，直到所有文物都归到一个大类中。

**思路二（自顶向下）**：先把所有文物放在一起，然后逐步细分，直到每件文物都单独成类。

这就是层次聚类的两种基本策略！

**层次聚类（Hierarchical Clustering）**的核心思想是：构建一个**嵌套的簇层次结构**，形成一棵树状图（Dendrogram）。这棵树展示了数据点是如何一步步合并或分裂的。

### 14.1.2 AGNES：自底向上的凝聚

**AGNES**（**AG**glomerative **NES**ting，嵌套凝聚法）是最常用的层次聚类方法。它的工作原理简单直观：

```
AGNES算法步骤：

初始状态：每个点自成一簇
    ●     ●     ●     ●     ●
    A     B     C     D     E

第1步：找到最近的两个簇，合并它们
    ●─────●     ●     ●
    A  B C      D     E

第2步：继续合并最近的簇
    ●─────●─────●
    A  B C   D  E

第3步：直到所有点都在一个簇中
    ●─────────────────●
    A  B C   D  E
```

**算法流程**：

1. **初始化**：将每个数据点视为一个单独的簇（共$n$个簇）
2. **迭代合并**：
   - 计算所有簇之间的距离
   - 找到距离最近的两个簇
   - 将它们合并成一个新的簇
3. **终止**：直到所有点都合并到一个簇中（或达到预定的簇数）

### 14.1.3 DIANA：自顶向下的分裂

**DIANA**（**DI**visive **ANA**lysis，分裂分析法）采用相反的策略：

```
DIANA算法步骤：

初始状态：所有点在一个大簇中
    ●─────────────────●
    A  B C   D  E

第1步：将最不一致的点分离出去
    ●────────●     ●───●
    A  B C   D     E

第2步：继续分裂
    ●─────●     ●     ●
    A  B C      D     E

第3步：直到每个点自成一簇
    ●     ●     ●     ●     ●
    A     B     C     D     E
```

DIANA的计算复杂度通常比AGNES高，因为每次分裂都需要考虑所有可能的划分方式。因此，在实践中AGNES更为常用。

### 14.1.4 树状图（Dendrogram）

层次聚类的结果最直观的展示方式是**树状图**：

```
树状图示例（5个数据点）：

高度
 5 │                    ╭───────╮
   │            ╭───────╯       │
 4 │    ╭───────╯               │
   │    │       ╭───────────────╯
 3 │    │ ╭─────╯
   │  ╭─╯─╯
 2 │ ╭╯   ╭──────────╮
   │ │  ╭─╯          │
 1 │╭╯ ╭╯           ╭╯
   ├┴──┴┴───────────┴┘
     A B C D E

横轴：数据点
纵轴：合并时的距离（或相似度）
```

通过在某个高度"切断"树状图，我们可以得到不同数量的簇：
- 在高度4切断 → 2个簇
- 在高度3切断 → 3个簇
- 在高度2切断 → 4个簇

---

## 14.2 Linkage Methods：如何度量簇间距离

### 14.2.1 核心问题

当我们要合并两个簇时，必须回答一个问题：**如何度量两个簇之间的距离？**

不同的定义方式会产生完全不同的聚类结果！

```
两个簇的情况：

簇A:    ●      ○           ○
       ○ ○              ●
          ○    ●
          
簇B:              ●  ●
                 ●    ●

● 属于簇A的点    ○ 属于簇B的点

问：簇A和簇B之间的距离是多少？
```

### 14.2.2 Single Linkage（最短距离/单连接）

**定义**：两个簇之间的距离定义为它们**最近的两个点**之间的距离。

$$d_{SL}(A, B) = \min_{x \in A, y \in B} d(x, y)$$

**形象比喻**：想象两个岛屿，最短距离就是它们之间最短桥的长度。

```
Single Linkage可视化：

    ●───────┐
   ● │ ●    │
    ─┼──    ○
     │    ○ ○
   ──┴───────→ 最近距离
```

**特点**：
- 倾向于形成"链状"的簇（Chaining Effect）
- 可以发现非球形的簇
- 对噪声敏感

### 14.2.3 Complete Linkage（最长距离/全连接）

**定义**：两个簇之间的距离定义为它们**最远的两个点**之间的距离。

$$d_{CL}(A, B) = \max_{x \in A, y \in B} d(x, y)$$

**形象比喻**：两个城堡之间的最远距离——从一座城堡最远的角到另一座城堡最远的角。

```
Complete Linkage可视化：

    ●←──────────────→○
   ● │ ●           ○ ○
    ─┼──            ○
     │
   最远距离
```

**特点**：
- 倾向于形成紧凑的、球形的簇
- 对噪声相对不敏感
- 可能将大的簇分裂

### 14.2.4 Average Linkage（平均距离）

**定义**：两个簇之间**所有点对**的平均距离。

$$d_{AL}(A, B) = \frac{1}{|A| \cdot |B|} \sum_{x \in A} \sum_{y \in B} d(x, y)$$

**形象比喻**：两座城市之间，计算所有可能的"从A城任一建筑到B城任一建筑"的距离，然后取平均。

**特点**：
- Single和Complete之间的折中
- 计算复杂度较高（$O(|A| \cdot |B|)$）
- 结果通常比较稳定

### 14.2.5 Ward's Method（Ward法/最小方差法）

**定义**：合并两个簇时，使得**合并后的簇内平方和增量最小**。

$$
\Delta(A, B) = \sum_{x \in A \cup B} \|x - \mu_{AB}\|^2 - \sum_{x \in A} \|x - \mu_A\|^2 - \sum_{x \in B} \|x - \mu_B\|^2$$

其中 $\mu_A, \mu_B, \mu_{AB}$ 分别是簇A、簇B、合并后簇的中心。

**等价形式**：

$$\Delta(A, B) = \frac{|A| \cdot |B|}{|A| + |B|} \|\mu_A - \mu_B\|^2$$

**特点**：
- 倾向于形成大小相近的簇
- 类似于K-Means的损失函数
- 在实践中效果很好

### 14.2.6 Linkage方法比较

```
不同Linkage方法形成的簇形状：

Single Linkage:      Complete Linkage:     Ward's Method:

●───●───●            ●───●                 ●───●
    │                │   │                 │   │
●───●            ●───┴───┴───●         ●───┴───┴───●
                      紧凑球形             大小均匀
长链状簇
```

---

## 14.3 Lance-Williams递推公式：统一的数学框架

### 14.3.1 递推的必要性

在AGNES算法中，每次合并两个簇后，我们需要重新计算新簇与所有其他簇之间的距离。如果每次都从头计算，复杂度会非常高。

**Lance-Williams递推公式**提供了一个优雅的解决方案：**用合并前的距离来计算合并后的距离**。

### 14.3.2 通用公式

假设我们要将簇 $A$ 和 $B$ 合并成新簇 $C = A \cup B$，需要计算 $C$ 与任意其他簇 $X$ 的距离 $d(C, X)$。

Lance-Williams公式的一般形式为：

$$d(C, X) = \alpha_A \cdot d(A, X) + \alpha_B \cdot d(B, X) + \beta \cdot d(A, B) + \gamma \cdot |d(A, X) - d(B, X)|$$

其中 $\alpha_A, \alpha_B, \beta, \gamma$ 是取决于具体Linkage方法的参数。

### 14.3.3 各Linkage方法的参数

| Linkage方法 | $\alpha_A$ | $\alpha_B$ | $\beta$ | $\gamma$ |
|-------------|-----------|-----------|---------|----------|
| Single | $\frac{1}{2}$ | $\frac{1}{2}$ | $0$ | $-\frac{1}{2}$ |
| Complete | $\frac{1}{2}$ | $\frac{1}{2}$ | $0$ | $+\frac{1}{2}$ |
| Average | $\frac{\|A\|}{\|A\|+\|B\|}$ | $\frac{\|B\|}{\|A\|+\|B\|}$ | $0$ | $0$ |
| Centroid | $\frac{\|A\|}{\|A\|+\|B\|}$ | $\frac{\|B\|}{\|A\|+\|B\|}$ | $-\frac{\|A\|\cdot\|B\|}{(\|A\|+\|B\|)^2}$ | $0$ |
| Ward's | $\frac{\|A\|+\|X\|}{\|A\|+\|B\|+\|X\|}$ | $\frac{\|B\|+\|X\|}{\|A\|+\|B\|+\|X\|}$ | $-\frac{\|X\|}{\|A\|+\|B\|+\|X\|}$ | $0$ |

### 14.3.4 推导示例：Single Linkage

**证明**：当 $\alpha_A = \alpha_B = \frac{1}{2}, \beta = 0, \gamma = -\frac{1}{2}$ 时，公式等价于Single Linkage。

**证明过程**：

Single Linkage的定义：
$$d_{SL}(C, X) = \min_{z \in C, x \in X} d(z, x) = \min\left\{\min_{a \in A, x \in X} d(a, x), \min_{b \in B, x \in X} d(b, x)\right\}$$

即：
$$d_{SL}(C, X) = \min\{d_{SL}(A, X), d_{SL}(B, X)\}$$

利用恒等式 $\min(p, q) = \frac{p + q}{2} - \frac{|p - q|}{2}$：

$$d_{SL}(C, X) = \frac{d(A, X) + d(B, X)}{2} - \frac{|d(A, X) - d(B, X)|}{2}$$

这正是Lance-Williams公式在 $\alpha_A = \alpha_B = \frac{1}{2}, \beta = 0, \gamma = -\frac{1}{2}$ 时的形式！

---

## 14.4 DBSCAN：基于密度的聚类

### 14.4.1 密度聚类的动机

K-Means和层次聚类都有一个**隐藏的假设**：簇是**球形的**（基于距离中心）。

但现实世界的数据往往有更复杂的形状：

```
K-Means失效的例子：

      ○ ○ ○
    ○   ●   ○      两个同心圆
  ○   ● ● ●   ○    K-Means无法区分！
    ○   ●   ○
      ○ ○ ○

DBSCAN能处理的形状：

    ●●●
   ●   ●    ●●●●
  ●     ●  ●    ●
   ●   ●   ●    ●
    ●●●      ●●●●
    
    任意形状都可以！
```

### 14.4.2 DBSCAN的核心概念

DBSCAN（**D**ensity-**B**ased **S**patial **C**lustering of **A**pplications with **N**oise）基于以下关键概念：

#### ε-邻域（Epsilon Neighborhood）

对于点 $p$，其 $\varepsilon$-邻域定义为：

$$N_{\varepsilon}(p) = \{q \in D \mid d(p, q) \leq \varepsilon\}$$

即以 $p$ 为中心、半径为 $\varepsilon$ 的球内的所有点。

```
ε-邻域可视化：

         · · ·
       ·   ●   ·     ● = 点p
       ·  /|\  ·     · · · = ε-邻域边界
       · / | \ ·     ○ = p的邻居
         ○  ○  ○
           ○
```

#### 核心点（Core Point）

点 $p$ 是核心点，如果它的 $\varepsilon$-邻域内至少包含 **MinPts** 个点（包括 $p$ 本身）：

$$|N_{\varepsilon}(p)| \geq \text{MinPts}$$

#### 边界点（Border Point）

点 $p$ 是边界点，如果：
- 它不是核心点
- 但它落在某个核心点的 $\varepsilon$-邻域内

#### 噪声点（Noise Point）

既不是核心点，也不是边界点的点。

```
核心点、边界点、噪声点：

    ○ ○ ○
  ○ ● ● ○ ●      ● = 核心点（ε-邻域内有≥MinPts个点）
  ○ ● ● ○        ○ = 边界点（在核心点的ε-邻域内）
    ○ ○ ○  ×     × = 噪声点（孤立点）

假设 ε 如圆圈所示，MinPts = 4
```

### 14.4.3 密度可达性

#### 直接密度可达（Directly Density-Reachable）

点 $q$ 从点 $p$ 是**直接密度可达**的，如果：
1. $p$ 是核心点
2. $q \in N_{\varepsilon}(p)$

#### 密度可达（Density-Reachable）

点 $q$ 从点 $p$ 是**密度可达**的，如果存在一条链 $p_1, p_2, ..., p_n$ 使得：
- $p_1 = p$, $p_n = q$
- $p_{i+1}$ 从 $p_i$ 是直接密度可达的

```
密度可达链：

p = p₁ ──ε──→ p₂ ──ε──→ p₃ ──ε──→ p₄ = q
   [核心]     [核心]     [核心]

即使 d(p, q) >> ε，p和q仍然是密度可达的！
```

#### 密度相连（Density-Connected）

点 $p$ 和 $q$ 是**密度相连**的，如果存在一个点 $o$ 使得 $p$ 和 $q$ 都从 $o$ 是密度可达的。

### 14.4.4 DBSCAN算法

**定义（簇）**：基于密度的簇 $C$ 是 $D$ 的一个非空子集，满足：
1. **最大性**：如果 $p \in C$ 且 $q$ 从 $p$ 密度可达，则 $q \in C$
2. **连通性**：对于任意 $p, q \in C$，$p$ 和 $q$ 是密度相连的

**算法步骤**：

```
DBSCAN算法：

输入：数据集D，参数ε，MinPts
输出：簇标签（-1表示噪声）

1. 初始化所有点为未访问
2. 对每个未访问点p：
   a. 标记p为已访问
   b. 如果p是核心点（|N_ε(p)| ≥ MinPts）：
      i.  创建新簇C
      ii. 将p加入C
      iii. 对N_ε(p)中每个点q：
           - 如果q未访问：标记为已访问，如果q也是核心点，扩展其邻域
           - 如果q不属于任何簇：将q加入C
   c. 否则：标记p为噪声（暂时）
3. 返回所有簇
```

### 14.4.5 参数选择策略

选择合适的 $\varepsilon$ 和 MinPts 是DBSCAN成功的关键。

#### MinPts的选择

经验法则：
- **MinPts ≥ 维度数 + 1**
- 通常取 **MinPts = 2 × 维度数**
- 对于二维数据，MinPts = 4 是常见选择

#### ε的选择：K-距离图法

**K-距离（K-distance）**：点 $p$ 的K-距离是 $p$ 到它的第 $K$ 个最近邻居的距离。

**方法**：
1. 对所有点计算K-距离（$K = \text{MinPts} - 1$）
2. 将K-距离按降序排列
3. 画出K-距离图
4. 找到"肘部"——曲线变化最剧烈的点
5. 该点对应的K-距离作为 $\varepsilon$

```
K-距离图示例：

K-距离 │
       │         ╭──────
       │        ╱
       │       ╱ ← "肘部"
       │      ╱
       │  ╭───╯
       │ ╱
       │╱
       └───────────────→ 点（按K-距离降序）

肘部对应的K-距离 ≈ 最佳ε
```

---

## 14.5 聚类评估指标

### 14.5.1 为什么需要评估？

聚类是无监督学习，没有"正确答案"来比较。但我们需要知道：
- 聚类质量如何？
- 参数选择是否合理？
- 不同的聚类方法哪个更好？

### 14.5.2 Silhouette Coefficient（轮廓系数）

**定义**：对于点 $i$，设 $a(i)$ 是它到同簇其他点的平均距离，$b(i)$ 是它到最近其他簇所有点的平均距离。

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

**解释**：
- $s(i) \approx 1$：点 $i$ 被分到了正确的簇（簇内紧，簇间疏）
- $s(i) \approx 0$：点 $i$ 在两个簇的边界上
- $s(i) \approx -1$：点 $i$ 可能被分错了

**整体评估**：
$$\text{Silhouette Score} = \frac{1}{n} \sum_{i=1}^{n} s(i)$$

### 14.5.3 Calinski-Harabasz Index（CH指数）

**定义**：

$$CH = \frac{\text{Tr}(B_k) / (k - 1)}{\text{Tr}(W_k) / (n - k)}$$

其中：
- $B_k$：簇间散度矩阵（Between-cluster dispersion）
- $W_k$：簇内散度矩阵（Within-cluster dispersion）
- $k$：簇的数量
- $n$：样本数量

**解释**：
- 分子衡量簇间分离度（越大越好）
- 分母衡量簇内紧密度（越小越好）
- **CH越大，聚类效果越好**

### 14.5.4 Davies-Bouldin Index（DB指数）

**定义**：

$$DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left(\frac{s_i + s_j}{d_{ij}}\right)$$

其中：
- $s_i$：簇 $i$ 内点到中心的平均距离（簇内离散度）
- $d_{ij}$：簇 $i$ 和簇 $j$ 中心之间的距离

**解释**：
- 对于每个簇，找到与它"最相似"的其他簇
- 相似度定义为：(簇内离散度之和) / (簇间距离)
- **DB越小，聚类效果越好**

### 14.5.5 指标比较

```
评估指标总结：

┌─────────────────┬──────────┬─────────────┬──────────────────┐
│ 指标            │ 取值范围 │ 最优值      │ 特点             │
├─────────────────┼──────────┼─────────────┼──────────────────┤
│ Silhouette      │ [-1, 1]  │ 接近1       │ 直观，有边界信息 │
│ Calinski-Harabasz│ [0, +∞) │ 越大越好    │ 计算快，适合大数据│
│ Davies-Bouldin  │ [0, +∞)  │ 接近0       │ 考虑簇内簇间平衡 │
└─────────────────┴──────────┴─────────────┴──────────────────┘
```

---

## 14.6 完整数学推导

### 14.6.1 Ward's Method的增量公式推导

**目标**：证明
$$\Delta(A, B) = \frac{|A| \cdot |B|}{|A| + |B|} \|\mu_A - \mu_B\|^2$$

**证明**：

设 $\mu_{AB}$ 是合并后簇的中心：
$$\mu_{AB} = \frac{|A|\mu_A + |B|\mu_B}{|A| + |B|}$$

合并后的簇内平方和：
$$SS_{AB} = \sum_{x \in A \cup B} \|x - \mu_{AB}\|^2$$

利用 $\|x - \mu_{AB}\|^2 = \|x - \mu_A + \mu_A - \mu_{AB}\|^2$ 展开...

经过代数运算（详见代码注释），可得：

$$\Delta(A, B) = SS_{AB} - SS_A - SS_B = \frac{|A| \cdot |B|}{|A| + |B|} \|\mu_A - \mu_B\|^2$$

### 14.6.2 轮廓系数计算

给定 $n$ 个点，已划分为 $k$ 个簇 $C_1, C_2, ..., C_k$。

对每个点 $i \in C_j$：

**步骤1：计算 $a(i)$（簇内不相似度）**
$$a(i) = \frac{1}{|C_j| - 1} \sum_{\ell \in C_j, \ell \neq i} d(i, \ell)$$

**步骤2：计算 $b(i)$（簇间不相似度）**
$$b(i) = \min_{m \neq j} \frac{1}{|C_m|} \sum_{\ell \in C_m} d(i, \ell)$$

**步骤3：计算 $s(i)$**
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

### 14.6.3 DBSCAN复杂度分析

**时间复杂度**：
- 使用朴素方法：$O(n^2)$（对每个点计算所有距离）
- 使用空间索引（如R*-tree）：$O(n \log n)$
- 最坏情况：$O(n^2)$

**空间复杂度**：$O(n)$（存储点的标签和访问标记）

---

## 14.7 从零实现层次聚类与DBSCAN

本章的完整Python实现请参见 `hierarchical_dbscan.py` 文件，包含：

- **AgglomerativeClustering** 类：支持 single、complete、average、ward 四种 linkage 方法
- **DBSCAN** 类：基于密度的空间聚类，识别核心点、边界点和噪声点
- **ClusteringMetrics** 类：实现轮廓系数、CH指数、DB指数的纯Python版本
- **7个完整演示**：从基础用法到实战案例

### 核心代码结构

```python
# 层次聚类核心
class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='single'):
        # 支持 'single', 'complete', 'average', 'ward'
        
    def fit(self, X):
        # 使用 Lance-Williams 递推公式高效计算
        
    def _lance_williams_update(self, d_ax, d_bx, d_ab, size_a, size_b, size_x):
        # 通用递推框架

# DBSCAN核心  
class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        # ε-邻域半径和最小点数
        
    def fit(self, X):
        # 广度优先搜索扩展簇
        
    def _expand_cluster(self, X, labels, core_idx, neighbors, cluster_id):
        # 从核心点开始密度扩展

# 评估指标
class ClusteringMetrics:
    @staticmethod
    def silhouette_score(X, labels):
        # 轮廓系数 [-1, 1]
        
    @staticmethod
    def calinski_harabasz_score(X, labels):
        # CH指数，越大越好
        
    @staticmethod
    def davies_bouldin_score(X, labels):
        # DB指数，越小越好
```

---

## 14.8 实战案例总结

### 案例1：层次聚类基础

使用不同linkage方法对简单二维数据进行聚类，观察single linkage的链式效应和complete linkage的紧凑性。

### 案例2：链状数据比较

构造两串点形成的"链条"数据，验证single linkage倾向于连接链条，而complete linkage形成更紧凑的簇。

### 案例3：DBSCAN识别噪声

构造带噪声的簇数据，DBSCAN成功识别核心点、边界点和噪声点，而K-Means会将噪声点强制分配到某个簇。

### 案例4：K-距离图参数选择

通过K-距离图找到DBSCAN的最佳ε参数，避免参数调优的盲目性。

### 案例5：评估指标比较

使用轮廓系数、CH指数、DB指数比较层次聚类和DBSCAN在同一数据集上的表现。

### 案例6：同心圆数据

构造非球形的同心圆数据，DBSCAN显著优于层次聚类，验证了密度聚类在处理任意形状数据时的优势。

### 案例7：Ward's Method演示

展示Ward方法如何通过最小化方差增量来形成大小均匀的簇。

---

## 练习题

### 基础题

**练习1**：解释为什么single linkage会产生"链式效应"（chaining effect），并给出一个这种效应可能是有益的实际应用场景。

**练习2**：在DBSCAN中，假设MinPts=5，一个点p的ε-邻域内有4个点（包括p自己）。p是核心点吗？为什么？

**练习3**：计算以下3个点的轮廓系数（使用欧氏距离）：
- 点A：(0, 0)，属于簇1
- 点B：(1, 0)，属于簇1
- 点C：(5, 0)，属于簇2

### 进阶题

**练习4**：给定两个簇A={(0,0), (1,0)}和B={(3,0), (4,0)}，计算：
- Single linkage距离
- Complete linkage距离
- Average linkage距离
- Ward's method的合并方差增量

**练习5**：证明Lance-Williams公式在complete linkage参数下等价于$d_{CL}(C, X) = \max(d(A, X), d(B, X))$。

### 挑战题

**练习6**：实现一个改进的DBSCAN算法，能够自动选择ε参数。思路：使用K-距离图的"肘部检测"算法来自动确定最佳ε值。

---

## 本章小结

在本章中，我们深入学习了两种强大的聚类方法：

### 层次聚类
- **AGNES**：自底向上凝聚，从单点逐步合并
- **DIANA**：自顶向下分裂，从整体逐步细分
- **Linkage方法**：single（最短）、complete（最长）、average（平均）、ward（最小方差）
- **Lance-Williams递推公式**：统一框架，高效计算

### DBSCAN密度聚类
- **核心概念**：ε-邻域、核心点、边界点、噪声点
- **密度可达性**：直接密度可达、密度可达、密度相连
- **参数选择**：MinPts ≥ 维度+1，ε通过K-距离图确定
- **优势**：发现任意形状簇，自动识别噪声

### 聚类评估
- **轮廓系数**：[-1, 1]，接近1表示聚类质量好
- **CH指数**：越大越好，基于方差比
- **DB指数**：越小越好，考虑簇内簇间平衡

### 核心洞察
> "层次聚类揭示了数据的嵌套结构，DBSCAN揭示了数据的密度结构。真正的数据探索需要多种视角。"

---

## 参考文献

1. Johnson, S. C. (1967). Hierarchical clustering schemes. *Psychometrika*, 32(3), 241-254.

2. Kaufman, L., & Rousseeuw, P. J. (1990). *Finding groups in data: An introduction to cluster analysis*. Wiley.

3. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD'96)* (pp. 226-231). AAAI.

4. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.

5. Calinski, T., & Harabasz, J. (1974). A dendrite method for cluster analysis. *Communications in Statistics-Theory and Methods*, 3(1), 1-27.

6. Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, (2), 224-227.

7. Lance, G. N., & Williams, W. T. (1967). A general theory of classificatory sorting strategies. *Computer Journal*, 9(4), 373-380.

8. Schubert, E., Sander, J., Ester, M., Kriegel, H. P., & Xu, X. (2017). DBSCAN revisited, revisited: why and how you should (still) use DBSCAN. *ACM Transactions on Database Systems*, 42(3), 1-21.

---

*本章编写完成于 2026-03-24*  
*代码实现：hierarchical_dbscan.py（514行）*
