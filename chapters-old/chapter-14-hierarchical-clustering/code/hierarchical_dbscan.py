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
- 类似于K-Means的目标函数
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

```python
"""
第十四章：层次聚类与DBSCAN - 从零实现
作者：机器学习教材编写组

参考论文：
- Johnson, S. C. (1967). Hierarchical clustering schemes. Psychometrika.
- Ester et al. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. KDD.
- Lance & Williams (1967). A general theory of classificatory sorting strategies. Computer Journal.
"""

import math
from collections import defaultdict, deque


class AgglomerativeClustering:
    """
    凝聚层次聚类算法（AGNES）- 纯Python实现
    
    支持多种Linkage方法：single, complete, average, ward
    """
    
    def __init__(self, n_clusters=2, linkage='single'):
        """
        初始化层次聚类器
        
        参数:
            n_clusters: 目标聚类数量（在指定高度切割树状图）
            linkage: 连接方式 ('single', 'complete', 'average', 'ward')
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.distances_ = None  # 记录每次合并的距离
        self.merge_history_ = []  # 记录合并历史，用于构建树状图
        
        print(f"[AGNES] 初始化: n_clusters={n_clusters}, linkage={linkage}")
    
    def _euclidean_distance(self, x, y):
        """计算欧氏距离"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))
    
    def _compute_distance_matrix(self, X):
        """
        计算所有点对的距离矩阵
        
        返回: 二维列表，distances[i][j]表示点i和点j之间的距离
        """
        n = len(X)
        distances = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._euclidean_distance(X[i], X[j])
                distances[i][j] = dist
                distances[j][i] = dist
        
        return distances
    
    def _lance_williams_update(self, d_ax, d_bx, d_ab, size_a, size_b, size_x):
        """
        Lance-Williams递推公式 - 统一框架
        
        计算合并簇A和B后，新簇C=A∪B与簇X的距离
        
        参数:
            d_ax: 簇A到簇X的距离
            d_bx: 簇B到簇X的距离
            d_ab: 簇A到簇B的距离
            size_a, size_b, size_x: 各簇的大小
        
        返回:
            新簇C到簇X的距离
        """
        if self.linkage == 'single':
            # Single linkage: min(d(A,X), d(B,X))
            alpha_a = 0.5
            alpha_b = 0.5
            beta = 0.0
            gamma = -0.5
            
        elif self.linkage == 'complete':
            # Complete linkage: max(d(A,X), d(B,X))
            alpha_a = 0.5
            alpha_b = 0.5
            beta = 0.0
            gamma = 0.5
            
        elif self.linkage == 'average':
            # Average linkage: weighted average
            total = size_a + size_b
            alpha_a = size_a / total
            alpha_b = size_b / total
            beta = 0.0
            gamma = 0.0
            
        elif self.linkage == 'ward':
            # Ward's method: minimize variance increase
            total = size_a + size_b + size_x
            alpha_a = (size_a + size_x) / total
            alpha_b = (size_b + size_x) / total
            beta = -size_x / total
            gamma = 0.0
            
        else:
            raise ValueError(f"未知的linkage类型: {self.linkage}")
        
        # Lance-Williams递推公式
        d_cx = (alpha_a * d_ax + alpha_b * d_bx + 
                beta * d_ab + gamma * abs(d_ax - d_bx))
        
        return d_cx
    
    def fit(self, X):
        """
        执行层次聚类
        
        参数:
            X: 训练数据，列表的列表 [[x1, x2, ...], ...]
        
        返回:
            self
        """
        n_samples = len(X)
        
        if n_samples < self.n_clusters:
            raise ValueError("样本数不能小于聚类数")
        
        # 初始化：每个点自成一簇
        # clusters[i] 表示第i个簇包含的原始点索引集合
        clusters = [{i} for i in range(n_samples)]
        
        # 簇大小
        cluster_sizes = [1] * n_samples
        
        # 初始化距离矩阵（簇间距离）
        # 初始时，簇间距离就是点间距离
        point_distances = self._compute_distance_matrix(X)
        cluster_distances = [row[:] for row in point_distances]
        
        # 活跃簇索引（尚未被合并的簇）
        active_clusters = list(range(n_samples))
        
        self.distances_ = []
        self.merge_history_ = []
        
        # 迭代合并，直到达到目标簇数
        while len(active_clusters) > self.n_clusters:
            # 找到距离最近的两个簇
            min_dist = float('inf')
            min_i, min_j = -1, -1
            
            for idx_i, i in enumerate(active_clusters):
                for idx_j, j in enumerate(active_clusters[idx_i + 1:], idx_i + 1):
                    if cluster_distances[i][j] < min_dist:
                        min_dist = cluster_distances[i][j]
                        min_i, min_j = i, j
            
            if min_i == -1:
                break
            
            # 记录合并信息
            self.distances_.append(min_dist)
            self.merge_history_.append((min_i, min_j, min_dist))
            
            # 合并簇min_i和min_j，新簇使用min_i的索引
            new_cluster = clusters[min_i] | clusters[min_j]
            clusters[min_i] = new_cluster
            cluster_sizes[min_i] = len(new_cluster)
            
            # 更新距离矩阵：计算新簇与其他簇的距离
            for k in active_clusters:
                if k != min_i and k != min_j:
                    # 使用Lance-Williams递推公式
                    new_dist = self._lance_williams_update(
                        cluster_distances[min_i][k],
                        cluster_distances[min_j][k],
                        cluster_distances[min_i][min_j],
                        cluster_sizes[min_i] - len(clusters[min_j]),  # 合并前的size_a
                        len(clusters[min_j]),  # size_b
                        cluster_sizes[k]  # size_x
                    )
                    cluster_distances[min_i][k] = new_dist
                    cluster_distances[k][min_i] = new_dist
            
            # 移除被合并的簇min_j
            active_clusters.remove(min_j)
        
        # 生成标签
        self.labels_ = [0] * n_samples
        for cluster_id, cluster_idx in enumerate(active_clusters):
            for point_idx in clusters[cluster_idx]:
                self.labels_[point_idx] = cluster_id
        
        return self
    
    def fit_predict(self, X):
        """拟合并返回标签"""
        self.fit(X)
        return self.labels_


class DBSCAN:
    """
    DBSCAN算法 - 纯Python实现
    
    基于密度的空间聚类算法，能够发现任意形状的簇并识别噪声点
    """
    
    # 标签常量
    UNVISITED = -2    # 未访问
    NOISE = -1        # 噪声点
    
    def __init__(self, eps=0.5, min_samples=5):
        """
        初始化DBSCAN
        
        参数:
            eps: ε-邻域半径（epsilon）
            min_samples: 成为核心点所需的最小邻居数（MinPts）
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.n_clusters_ = 0
        self.core_sample_indices_ = []  # 核心点索引
        
        print(f"[DBSCAN] 初始化: eps={eps}, min_samples={min_samples}")
    
    def _euclidean_distance(self, x, y):
        """计算欧氏距离"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))
    
    def _get_neighbors(self, X, point_idx):
        """
        获取指定点的ε-邻域内的所有点
        
        参数:
            X: 数据集
            point_idx: 中心点索引
        
        返回:
            邻居点索引列表
        """
        neighbors = []
        point = X[point_idx]
        
        for i, other_point in enumerate(X):
            if i != point_idx:
                dist = self._euclidean_distance(point, other_point)
                if dist <= self.eps:
                    neighbors.append(i)
        
        return neighbors
    
    def fit(self, X):
        """
        执行DBSCAN聚类
        
        参数:
            X: 训练数据，列表的列表
        
        返回:
            self
        """
        n_samples = len(X)
        
        # 初始化：所有点为未访问状态
        labels = [self.UNVISITED] * n_samples
        self.core_sample_indices_ = []
        
        cluster_id = 0
        
        for i in range(n_samples):
            # 跳过已访问的点
            if labels[i] != self.UNVISITED:
                continue
            
            # 获取点i的ε-邻域
            neighbors = self._get_neighbors(X, i)
            
            # 检查是否为核心点
            if len(neighbors) + 1 < self.min_samples:  # +1包括自己
                # 标记为噪声（暂时）
                labels[i] = self.NOISE
            else:
                # 发现一个新的簇
                self._expand_cluster(X, labels, i, neighbors, cluster_id)
                cluster_id += 1
        
        self.labels_ = labels
        self.n_clusters_ = cluster_id
        
        print(f"[DBSCAN] 聚类完成: 发现 {cluster_id} 个簇，" +
              f"{labels.count(self.NOISE)} 个噪声点")
        
        return self
    
    def _expand_cluster(self, X, labels, core_idx, neighbors, cluster_id):
        """
        扩展簇 - 从核心点开始，找到所有密度可达的点
        
        参数:
            X: 数据集
            labels: 标签列表（会被修改）
            core_idx: 起始核心点索引
            neighbors: 核心点的ε-邻域
            cluster_id: 当前簇的ID
        """
        # 将核心点加入当前簇
        labels[core_idx] = cluster_id
        self.core_sample_indices_.append(core_idx)
        
        # 使用队列进行广度优先搜索
        # 队列中存储待处理的点（密度可达链）
        queue = deque(neighbors)
        
        # 标记这些邻居为当前簇（它们至少被core_idx密度可达）
        for neighbor_idx in neighbors:
            if labels[neighbor_idx] == self.UNVISITED:
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == self.NOISE:
                # 边界点：之前被标记为噪声，现在发现它在一个簇的边界上
                labels[neighbor_idx] = cluster_id
        
        # 处理队列中的每个点
        while queue:
            current_idx = queue.popleft()
            
            # 获取当前点的ε-邻域
            current_neighbors = self._get_neighbors(X, current_idx)
            
            # 如果当前点也是核心点，继续扩展
            if len(current_neighbors) + 1 >= self.min_samples:
                self.core_sample_indices_.append(current_idx)
                
                for neighbor_idx in current_neighbors:
                    if labels[neighbor_idx] == self.UNVISITED:
                        # 新发现的点，加入队列继续扩展
                        labels[neighbor_idx] = cluster_id
                        queue.append(neighbor_idx)
                    elif labels[neighbor_idx] == self.NOISE:
                        # 边界点
                        labels[neighbor_idx] = cluster_id
    
    def fit_predict(self, X):
        """拟合并返回标签"""
        self.fit(X)
        return self.labels_


class ClusteringMetrics:
    """
    聚类评估指标 - 纯Python实现
    
    包含：轮廓系数(Silhouette)、Calinski-Harabasz指数、Davies-Bouldin指数
    """
    
    @staticmethod
    def euclidean_distance(x, y):
        """计算欧氏距离"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))
    
    @staticmethod
    def silhouette_score(X, labels):
        """
        计算轮廓系数 (Silhouette Coefficient)
        
        对于每个点：
        - a(i): 到同簇其他点的平均距离（簇内不相似度）
        - b(i): 到最近其他簇的平均距离（簇间不相似度）
        - s(i) = (b(i) - a(i)) / max(a(i), b(i))
        
        参数:
            X: 数据集
            labels: 聚类标签
        
        返回:
            平均轮廓系数 [-1, 1]，越接近1越好
        """
        n_samples = len(X)
        unique_labels = set(labels)
        
        # 排除噪声点（DBSCAN可能产生-1标签）
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        if len(unique_labels) < 2:
            return 0.0  # 只有一个簇，无法计算轮廓系数
        
        silhouette_scores = []
        
        for i in range(n_samples):
            label_i = labels[i]
            
            # 跳过噪声点
            if label_i == -1:
                continue
            
            # 计算a(i): 到同簇其他点的平均距离
            same_cluster_indices = [j for j in range(n_samples) 
                                    if labels[j] == label_i and j != i]
            
            if len(same_cluster_indices) == 0:
                continue
            
            a_i = sum(ClusteringMetrics.euclidean_distance(X[i], X[j]) 
                      for j in same_cluster_indices) / len(same_cluster_indices)
            
            # 计算b(i): 到最近其他簇的平均距离
            b_i = float('inf')
            
            for other_label in unique_labels:
                if other_label == label_i:
                    continue
                
                other_cluster_indices = [j for j in range(n_samples) 
                                         if labels[j] == other_label]
                
                avg_dist = sum(ClusteringMetrics.euclidean_distance(X[i], X[j]) 
                               for j in other_cluster_indices) / len(other_cluster_indices)
                
                b_i = min(b_i, avg_dist)
            
            # 计算轮廓系数s(i)
            s_i = (b_i - a_i) / max(a_i, b_i)
            silhouette_scores.append(s_i)
        
        return sum(silhouette_scores) / len(silhouette_scores) if silhouette_scores else 0.0
    
    @staticmethod
    def calinski_harabasz_score(X, labels):
        """
        计算Calinski-Harabasz指数
        
        CH = (Tr(B_k) / (k-1)) / (Tr(W_k) / (n-k))
        
        其中B_k是簇间散度矩阵，W_k是簇内散度矩阵
        
        参数:
            X: 数据集
            labels: 聚类标签
        
        返回:
            CH指数，越大越好
        """
        n_samples = len(X)
        unique_labels = set(labels)
        
        # 排除噪声点
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            return 0.0
        
        # 计算整体中心
        n_features = len(X[0])
        overall_mean = [sum(X[i][j] for i in range(n_samples)) / n_samples 
                        for j in range(n_features)]
        
        # 计算簇间散度 B_k
        between_dispersion = 0.0
        
        for label in unique_labels:
            cluster_indices = [i for i in range(n_samples) if labels[i] == label]
            cluster_size = len(cluster_indices)
            
            # 簇中心
            cluster_mean = [sum(X[i][j] for i in cluster_indices) / cluster_size 
                            for j in range(n_features)]
            
            # 簇中心到整体中心的距离平方 × 簇大小
            dist_to_overall = ClusteringMetrics.euclidean_distance(cluster_mean, overall_mean)
            between_dispersion += cluster_size * (dist_to_overall ** 2)
        
        # 计算簇内散度 W_k
        within_dispersion = 0.0
        
        for label in unique_labels:
            cluster_indices = [i for i in range(n_samples) if labels[i] == label]
            cluster_size = len(cluster_indices)
            
            # 簇中心
            cluster_mean = [sum(X[i][j] for i in cluster_indices) / cluster_size 
                            for j in range(n_features)]
            
            # 簇内点到中心的距离平方和
            for i in cluster_indices:
                dist_to_center = ClusteringMetrics.euclidean_distance(X[i], cluster_mean)
                within_dispersion += dist_to_center ** 2
        
        # 计算CH指数
        ch_score = (between_dispersion / (n_clusters - 1)) / (within_dispersion / (n_samples - n_clusters))
        
        return ch_score
    
    @staticmethod
    def davies_bouldin_score(X, labels):
        """
        计算Davies-Bouldin指数
        
        DB = (1/k) * Σ_i max_{j≠i} ((s_i + s_j) / d_ij)
        
        其中s_i是簇i的平均离散度，d_ij是簇i和j中心之间的距离
        
        参数:
            X: 数据集
            labels: 聚类标签
        
        返回:
            DB指数，越小越好
        """
        n_samples = len(X)
        unique_labels = sorted(set(labels))
        
        # 排除噪声点
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            return float('inf')
        
        # 计算每个簇的中心和平均离散度
        cluster_centers = {}
        cluster_dispersions = {}
        
        for label in unique_labels:
            cluster_indices = [i for i in range(n_samples) if labels[i] == label]
            cluster_size = len(cluster_indices)
            
            # 簇中心
            n_features = len(X[0])
            center = [sum(X[i][j] for i in cluster_indices) / cluster_size 
                      for j in range(n_features)]
            cluster_centers[label] = center
            
            # 平均离散度（到中心的平均距离）
            avg_dispersion = sum(ClusteringMetrics.euclidean_distance(X[i], center) 
                                 for i in cluster_indices) / cluster_size
            cluster_dispersions[label] = avg_dispersion
        
        # 计算DB指数
        db_sum = 0.0
        
        for i, label_i in enumerate(unique_labels):
            max_ratio = 0.0
            
            for j, label_j in enumerate(unique_labels):
                if i == j:
                    continue
                
                # 两个簇中心之间的距离
                center_dist = ClusteringMetrics.euclidean_distance(
                    cluster_centers[label_i], cluster_centers[label_j]
                )
                
                if center_dist > 0:
                    # 相似度 = (离散度之和) / 中心距离
                    ratio = (cluster_dispersions[label_i] + cluster_dispersions[label_j]) / center_dist
                    max_ratio = max(max_ratio, ratio)
            
            db_sum += max_ratio
        
        db_score = db_sum / n_clusters
        return db_score
    
    @staticmethod
    def compute_k_distances(X, k):
        """
        计算所有点的K-距离（用于DBSCAN参数选择）
        
        K-距离：点到其第K个最近邻居的距离
        
        参数:
            X: 数据集
            k: K值
        
        返回:
            所有点的K-距离列表（降序排列）
        """
        n_samples = len(X)
        k_distances = []
        
        for i in range(n_samples):
            # 计算点i到所有其他点的距离
            distances = []
            for j in range(n_samples):
                if i != j:
                    dist = ClusteringMetrics.euclidean_distance(X[i], X[j])
                    distances.append(dist)
            
            # 排序并取第K个
            distances.sort()
            if len(distances) >= k:
                k_distances.append(distances[k - 1])
            else:
                k_distances.append(distances[-1] if distances else 0)
        
        # 降序排列
        k_distances.sort(reverse=True)
        return k_distances


def print_dendrogram_ascii(merge_history, labels=None):
    """
    使用ASCII字符打印简单的树状图
    
    参数:
        merge_history: 合并历史列表 [(cluster_i, cluster_j, distance), ...]
        labels: 点的标签
    """
    print("\n树状图（简化表示）:")
    print("=" * 50)
    
    if not merge_history:
        print("（无合并历史）")
        return
    
    print(f"{'步骤':<6} {'合并的簇':<20} {'距离':<10}")
    print("-" * 50)
    
    for step, (i, j, dist) in enumerate(merge_history):
        print(f"{step+1:<6} {i} + {j:<15} {dist:.4f}")
    
    print("=" * 50)


# ============================================================
# 演示与测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("层次聚类与DBSCAN算法演示")
    print("=" * 60)
    
    # ========== 演示1：层次聚类基础 ==========
    print("\n【演示1】层次聚类基础（AGNES）")
    print("-" * 40)
    
    # 简单的二维数据
    X_simple = [
        [1, 1],   # 簇A
        [1.5, 1.5],  # 簇A
        [1.2, 0.8],  # 簇A
        [5, 5],   # 簇B
        [5.5, 5.5],  # 簇B
        [4.8, 5.2],  # 簇B
    ]
    
    print("数据点:")
    for i, point in enumerate(X_simple):
        print(f"  点{i}: {point}")
    
    # 使用不同的linkage方法
    for linkage in ['single', 'complete', 'average', 'ward']:
        print(f"\n--- Linkage: {linkage} ---")
        agg = AgglomerativeClustering(n_clusters=2, linkage=linkage)
        labels = agg.fit_predict(X_simple)
        print(f"聚类结果: {labels}")
        
        # 打印合并历史
        if agg.merge_history_:
            print(f"合并步骤数: {len(agg.merge_history_)}")
            print(f"最后一次合并距离: {agg.distances_[-1]:.4f}")
    
    # ========== 演示2：不同Linkage方法的比较 ==========
    print("\n\n【演示2】不同Linkage方法的比较（链状数据）")
    print("-" * 40)
    
    # 构造链状数据
    X_chain = [[i, 0] for i in range(5)] + [[i + 4, 1] for i in range(5)]
    
    print("链状数据:")
    print("A链: (0,0), (1,0), (2,0), (3,0), (4,0)")
    print("B链: (4,1), (5,1), (6,1), (7,1), (8,1)")
    print("注意：点(4,0)和(4,1)很接近，但属于不同链")
    
    for linkage in ['single', 'complete']:
        agg = AgglomerativeClustering(n_clusters=2, linkage=linkage)
        labels = agg.fit_predict(X_chain)
        
        # 统计每个簇包含的点
        cluster_0 = [i for i, l in enumerate(labels) if l == 0]
        cluster_1 = [i for i, l in enumerate(labels) if l == 1]
        
        print(f"\n{linkage} linkage:")
        print(f"  簇0: {cluster_0}")
        print(f"  簇1: {cluster_1}")
    
    print("\n说明：Single linkage倾向于形成链状簇，")
    print("      Complete linkage倾向于形成紧凑的簇")
    
    # ========== 演示3：DBSCAN基础 ==========
    print("\n\n【演示3】DBSCAN基础演示")
    print("-" * 40)
    
    # 构造带噪声的数据
    X_dbscan = [
        # 簇1：中心(2, 2)
        [2, 2], [2.1, 2.2], [1.9, 2.1], [2.2, 1.9], [1.8, 2.2],
        # 簇2：中心(5, 5)
        [5, 5], [5.1, 5.2], [4.9, 5.1], [5.2, 4.9], [4.8, 5.2],
        # 噪声点
        [10, 10], [0, 8],
    ]
    
    print("数据点:")
    print("  簇1（中心2,2）: 5个点")
    print("  簇2（中心5,5）: 5个点")
    print("  噪声: (10,10), (0,8)")
    
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    labels = dbscan.fit_predict(X_dbscan)
    
    print(f"\nDBSCAN结果 (eps=0.5, min_samples=3):")
    print(f"  标签: {labels}")
    print(f"  发现 {dbscan.n_clusters_} 个簇")
    print(f"  核心点: {dbscan.core_sample_indices_}")
    
    # 按簇分组显示
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(X_dbscan[i])
    
    print("\n聚类详情:")
    for label in sorted(clusters.keys()):
        if label == -1:
            print(f"  噪声点: {clusters[label]}")
        else:
            print(f"  簇{label}: {clusters[label]}")
    
    # ========== 演示4：DBSCAN参数选择（K-距离图） ==========
    print("\n\n【演示4】K-距离图（用于选择DBSCAN的eps参数）")
    print("-" * 40)
    
    # 使用演示3的数据
    k = 3  # MinPts - 1
    k_distances = ClusteringMetrics.compute_k_distances(X_dbscan, k)
    
    print(f"K-距离图 (K={k}):")
    print(f"{'排名':<6} {'K-距离':<10} {'可视化'}")
    print("-" * 40)
    
    max_dist = max(k_distances) if k_distances else 1
    for i, dist in enumerate(k_distances):
        bar = '█' * int(dist / max_dist * 20)
        print(f"{i+1:<6} {dist:.4f}    {bar}")
    
    print("\n提示：寻找K-距离图中的'肘部'，该点对应的距离适合作为eps值")
    
    # ========== 演示5：聚类评估指标 ==========
    print("\n\n【演示5】聚类评估指标")
    print("-" * 40)
    
    # 使用演示1的数据
    X_eval = X_simple
    
    # 真实标签（用于对比）
    true_labels = [0, 0, 0, 1, 1, 1]
    
    # 层次聚类结果
    agg = AgglomerativeClustering(n_clusters=2, linkage='average')
    agg_labels = agg.fit_predict(X_eval)
    
    # DBSCAN结果
    dbscan = DBSCAN(eps=1.5, min_samples=2)
    dbscan_labels = dbscan.fit_predict(X_eval)
    
    print("数据集：6个点，2个明显的簇")
    print(f"层次聚类标签: {agg_labels}")
    print(f"DBSCAN标签: {dbscan_labels}")
    
    # 计算评估指标
    print("\n评估指标比较:")
    print(f"{'指标':<25} {'层次聚类':<15} {'DBSCAN'}")
    print("-" * 60)
    
    # Silhouette Score
    agg_sil = ClusteringMetrics.silhouette_score(X_eval, agg_labels)
    db_sil = ClusteringMetrics.silhouette_score(X_eval, dbscan_labels)
    print(f"{'Silhouette Score':<25} {agg_sil:>10.4f}     {db_sil:>10.4f}")
    
    # Calinski-Harabasz Index
    agg_ch = ClusteringMetrics.calinski_harabasz_score(X_eval, agg_labels)
    db_ch = ClusteringMetrics.calinski_harabasz_score(X_eval, dbscan_labels)
    print(f"{'Calinski-Harabasz':<25} {agg_ch:>10.4f}     {db_ch:>10.4f}")
    
    # Davies-Bouldin Index
    agg_db = ClusteringMetrics.davies_bouldin_score(X_eval, agg_labels)
    db_db = ClusteringMetrics.davies_bouldin_score(X_eval, dbscan_labels)
    print(f"{'Davies-Bouldin':<25} {agg_db:>10.4f}     {db_db:>10.4f}")
    
    print("\n指标解读:")
    print("  - Silhouette: 越接近1越好")
    print("  - Calinski-Harabasz: 越大越好")
    print("  - Davies-Bouldin: 越小越好")
    
    # ========== 演示6：实战案例 - 环形数据 ==========
    print("\n\n【演示6】实战案例：同心圆数据")
    print("-" * 40)
    
    # 构造简单的同心圆数据（近似）
    import math
    
    X_circles = []
    # 内圆
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        rad = math.radians(angle)
        X_circles.append([2 + 1.5 * math.cos(rad), 2 + 1.5 * math.sin(rad)])
    # 外圆
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        rad = math.radians(angle)
        X_circles.append([2 + 3 * math.cos(rad), 2 + 3 * math.sin(rad)])
    
    print("数据：16个点，形成两个同心圆")
    print("内圆: 8个点（半径1.5）")
    print("外圆: 8个点（半径3.0）")
    
    # 层次聚类
    agg_circles = AgglomerativeClustering(n_clusters=2, linkage='single')
    agg_labels_circles = agg_circles.fit_predict(X_circles)
    
    # DBSCAN
    dbscan_circles = DBSCAN(eps=2.0, min_samples=3)
    dbscan_labels_circles = dbscan_circles.fit_predict(X_circles)
    
    print(f"\n层次聚类结果: {agg_labels_circles}")
    print(f"DBSCAN结果: {dbscan_labels_circles}")
    
    # 检查内圆点（索引0-7）的分配
    print("\n内圆点（索引0-7）的分配:")
    print(f"  层次聚类: {agg_labels_circles[:8]}")
    print(f"  DBSCAN: {dbscan_labels_circles[:8]}")
    
    print("\n说明：对于这种非球形数据，DBSCAN通常比层次聚类表现更好")
    
    # 计算轮廓系数
    sil_agg = ClusteringMetrics.silhouette_score(X_circles, agg_labels_circles)
    sil_dbscan = ClusteringMetrics.silhouette_score(X_circles, dbscan_labels_circles)
    
    print(f"\n轮廓系数:")
    print(f"  层次聚类: {sil_agg:.4f}")
    print(f"  DBSCAN: {sil_dbscan:.4f}")
    
    # ========== 演示7：Ward's Method详细演示 ==========
    print("\n\n【演示7】Ward's Method（最小方差法）")
    print("-" * 40)
    
    X_ward = [
        [0, 0],
        [0.5, 0],
        [0, 0.5],
        [5, 5],
        [5.5, 5],
    ]
    
    print("数据点:")
    for i, p in enumerate(X_ward):
        print(f"  点{i}: {p}")
    
    agg_ward = AgglomerativeClustering(n_clusters=2, linkage='ward')
    labels_ward = agg_ward.fit_predict(X_ward)
    
    print(f"\nWard's Method聚类结果: {labels_ward}")
    print("特点：倾向于形成大小相近的簇，合并后簇内方差增量最小")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
