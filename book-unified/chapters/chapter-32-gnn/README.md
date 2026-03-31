# 第三十二章：图神经网络基础

> *"万物皆有联系。社交网络中的人脉、分子中的原子、网页间的链接——图无处不在。让神经网络理解图，就是让人工智能理解世界的另一种方式。"*

## 本章学习目标

- 理解图数据的本质和图论基础概念
- 掌握图卷积网络(GCN)的数学原理和谱域推导
- 深入理解图注意力网络(GAT)的注意力机制
- 学习GraphSAGE的归纳式学习和邻居采样策略
- 实践分子性质预测、节点分类、链接预测等应用
- 了解图神经网络的前沿发展方向

---

## 32.1 什么是图数据

### 32.1.1 图：连接世界的数学语言

想象你参加了一场生日派对。在场有30个人，有些人互相认识，有些人是第一次见。如果要描述这个场景，你会怎么做？

你可以列一张名单，写下每个人的名字、年龄、职业。但这样缺少了最重要的信息——**谁认识谁**。你的朋友可能认识另一个人，而这个信息对于理解派对上的互动至关重要。

**这就是图的力量。**

在数学中，图(Graph)由两部分组成：
- **节点(Node/Vertex)**：表示实体（派对上的人）
- **边(Edge)**：表示实体之间的关系（认识关系）

$$
G = (V, E)
$$

其中 $V = \{v_1, v_2, ..., v_n\}$ 是节点集合，$E \subseteq V \times V$ 是边集合。

**费曼比喻**：图就像一张蜘蛛网🕸️。每个网丝的交叉点是节点，网丝本身是边。整个网络能捕捉到"谁和谁相连"这种结构信息，这是传统表格数据做不到的。

### 32.1.2 图数据的三大经典场景

#### 场景一：社交网络 🌐

想象一下你的微信朋友圈。你是其中一个节点，你发布的每条动态、点赞、评论都是你和朋友们之间的"边"。

| 节点属性 | 边的含义 |
|---------|---------|
| 用户ID、年龄、兴趣爱好 | 关注关系、互动频率、共同好友数 |

社交网络的核心问题是：
- **推荐好友**："你可能认识的人"
- **社区发现**：找出兴趣相同的小圈子
- **影响力传播**：一条消息如何在网络中传播

#### 场景二：分子结构 🧬

让我们从一个水分子说起：$H_2O$。它包含两个氢原子和一个氧原子，通过化学键连接。

**费曼比喻**：想象一个舞蹈队形。每个舞者是一个原子，舞者的手牵手就是化学键。整个队形（分子结构）决定了它的"舞姿"（化学性质）。

分子图的特征：
- **节点**：原子（碳、氢、氧、氮等）
- **边**：化学键（单键、双键、三键）
- **节点特征**：原子类型、电荷、质量
- **边特征**：键类型、键长、键能

**实际应用**：
- 预测新药分子的有效性
- 估算分子的毒性
- 设计更高效的催化剂

#### 场景三：知识图谱 🧠

想象一下你在学习历史。秦始皇统一中国、修建了长城、焚书坑儒——这些事实不是孤立的，它们通过"关系"连接在一起。

知识图谱用(实体-关系-实体)三元组表示知识：
- (秦始皇, 统一, 中国)
- (秦始皇, 修建, 长城)
- (秦始皇, 实施, 焚书坑儒)

**费曼比喻**：知识图谱就像大脑的神经网络。每个想法（实体）通过联想（关系）与其他想法连接，形成复杂的思维网络。

### 32.1.3 图的数学表示

#### 邻接矩阵(Adjacency Matrix)

邻接矩阵 $A$ 是表示图最直观的方式。对于 $n$ 个节点的图，$A$ 是一个 $n \times n$ 的矩阵：

$$
A_{ij} = \begin{cases} 
1 & \text{如果节点 } i \text{ 和节点 } j \text{ 之间有边} \\
0 & \text{否则}
\end{cases}
$$

例如，一个简单的4节点链状图：

$$
A = \begin{bmatrix}
0 & 1 & 0 & 0 \\
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 \\
0 & 0 & 1 & 0
\end{bmatrix}
$$

#### 度矩阵(Degree Matrix)

度矩阵 $D$ 是一个对角矩阵，对角线元素 $D_{ii}$ 表示节点 $i$ 的邻居数量：

$$
D_{ii} = \sum_j A_{ij}
$$

继续上面的例子：

$$
D = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 2 & 0 & 0 \\
0 & 0 & 2 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

#### 拉普拉斯矩阵(Laplacian Matrix)

拉普拉斯矩阵在图论中非常重要：

$$
L = D - A
$$

它是理解谱图论和图卷积的基础。

### 32.1.4 图神经网络要解决什么问题？

传统神经网络（如CNN、RNN）假设数据有规则的结构：
- **图像**：固定大小的网格，每个像素有固定数量的邻居
- **文本**：一维序列，每个词有明确的前后关系

但图数据是**不规则的**：
- 不同节点可能有不同数量的邻居（度不同）
- 没有固定的"空间"概念（不像图像有上下左右）
- 节点顺序不影响图的含义（排列不变性）

**图神经网络(GNN)的核心目标**：学习节点或图的表示(embedding)，使得结构相似的节点/图有相似的表示。

**费曼比喻**：想象你要给派对上的每个人写一份"简介"。这个简介应该包含：
1. 这个人本身的信息（节点特征）
2. TA的朋友圈特征（邻居信息聚合）

GNN就像是一位聪明的观察者，它不仅会看每个人，还会看每个人和周围人的互动，从而给出更全面的"简介"。

---

## 32.2 图卷积网络(GCN)

### 32.2.1 从CNN到GCN：直觉理解

在图像中，卷积操作是在每个像素的局部邻域（如3×3窗口）上进行加权求和。这种"局部聚合"的思想可以推广到图。

**费曼比喻——朋友圈信息传播**：

想象你发了一条朋友圈。会发生什么？

1. **你的朋友看到**（第一层邻居）
2. **朋友的朋友通过朋友的转发看到**（第二层邻居）
3. **信息逐层传播**

GCN就像这个过程的数学建模：
- 每个节点（你）的信息会传播给邻居（朋友）
- 同时，你也会收到来自邻居的信息
- 通过多层传播，远处的信息也能到达

### 32.2.2 谱域卷积的数学基础

GCN的理论基础来自**谱图理论**。让我们一步步推导。

#### 图上的傅里叶变换

在连续空间中，傅里叶变换将函数分解为不同频率的正弦波。

在图上，我们用**拉普拉斯矩阵的特征向量**作为"基函数"。

拉普拉斯矩阵 $L$ 是对称半正定的，可以进行特征分解：

$$
L = U \Lambda U^T
$$

其中：
- $U = [u_1, u_2, ..., u_n]$ 是特征向量矩阵（正交基）
- $\Lambda = \text{diag}(\lambda_1, \lambda_2, ..., \lambda_n)$ 是特征值对角矩阵

**特征值 $\lambda$ 的意义**：可以看作"频率"。
- 小特征值 → 低频（变化缓慢）
- 大特征值 → 高频（变化剧烈）

#### 图卷积的定义

图上信号 $x$ 的傅里叶变换：

$$
\hat{x} = U^T x
$$

逆变换：

$$
x = U \hat{x}
$$

**谱域卷积定理**：卷积在时域等于频域的乘积。

给定滤波器 $g$，图卷积定义为：

$$
g * x = U \cdot \text{diag}(\hat{g}) \cdot U^T x = U g(\Lambda) U^T x
$$

其中 $\hat{g}$ 是滤波器在频域的表示。

### 32.2.3 GCN的近似与简化

直接使用谱域卷积有两个问题：
1. 需要计算特征分解，$O(n^3)$ 复杂度
2. 滤波器是全局的，没有局部性

**Kipf & Welling (2017)** 提出了一个巧妙的近似方法。

#### 切比雪夫多项式近似

首先，用切比雪夫多项式 $T_k(x)$ 近似滤波器：

$$
g_{\theta}(\Lambda) = \sum_{k=0}^{K} \theta_k T_k(\tilde{\Lambda})
$$

其中 $\tilde{\Lambda} = \frac{2\Lambda}{\lambda_{\max}} - I$ 将特征值归一化到 $[-1, 1]$。

这样，图卷积可以表示为：

$$
g_{\theta} * x = \sum_{k=0}^{K} \theta_k T_k(\tilde{L}) x
$$

其中 $\tilde{L} = \frac{2L}{\lambda_{\max}} - I$。

#### 一阶近似（K=1）

取 $K=1$，只保留一阶邻居：

$$
g_{\theta} * x \approx \theta_0 x + \theta_1 (L - I) x = \theta_0 x - \theta_1 D^{-\frac{1}{2}} A D^{-\frac{1}{2}} x
$$

令 $\theta = \theta_0 = -\theta_1$：

$$
g_{\theta} * x \approx \theta (I + D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) x
$$

#### 归一化技巧

为了避免数值不稳定，Kipf引入了一个技巧：

$$
\tilde{A} = A + I \quad \text{(添加自环)}
$$

$$
\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}
$$

**最终的GCN传播规则**：

$$
H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)})
$$

其中：
- $H^{(l)} \in \mathbb{R}^{n \times d_l}$：第 $l$ 层的节点特征矩阵
- $W^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$：可学习的权重矩阵
- $\sigma$：激活函数（如ReLU）

### 32.2.4 GCN传播规则的直观理解

让我们拆解这个公式：

$$
H^{(l+1)} = \sigma(\underbrace{\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}}_{\text{归一化邻接矩阵}} \cdot \underbrace{H^{(l)}}_{\text{节点特征}} \cdot \underbrace{W^{(l)}}_{\text{线性变换}})
$$

**归一化邻接矩阵** $\hat{A} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$ 的作用：

1. **信息聚合**：每个节点接收邻居的信息
2. **度数归一化**：邻居越多，每个邻居的影响力越小（防止"名人效应"过度放大）

**费曼比喻——加权的朋友圈**：

想象你要做一个人生决定（比如换工作）。你会：
1. **咨询朋友**：但朋友很多，每个人给的建议权重不同
2. **考虑关系强度**：死党（高度数连接）的意见可能更重要
3. **避免过度影响**：有1000个朋友的人和有10个朋友的人，每个朋友的建议权重应该不同

GCN的归一化就是做这个"权重调整"，让每个节点的"声音"被合理听到。

### 32.2.5 GCN的完整算法

```
算法: GCN前向传播
输入: 
  - 特征矩阵 X ∈ R^{n×d}
  - 邻接矩阵 A ∈ R^{n×n}
  - 层数 L
  - 权重矩阵 {W^(0), ..., W^(L-1)}

1. 计算归一化邻接矩阵:
   Ã = A + I  (添加自环)
   D̃_ii = Σ_j Ã_ij
   Â = D̃^{-1/2} Ã D̃^{-1/2}

2. 初始化: H^(0) = X

3. 对于 l = 0 到 L-1:
   H^(l+1) = σ(Â H^(l) W^(l))

4. 输出: Z = H^(L)
```

### 32.2.6 GCN的局限与改进

**主要局限**：
1. **直推式学习(Transductive)**：无法处理训练时没见过的节点
2. **层数限制**：深层GCN会出现过度平滑(over-smoothing)
3. **固定权重**：所有邻居同等重要（虽然做了度数归一化，但没有区分不同邻居的重要性）

**费曼比喻**：GCN像一个传统班级，老师给每个同学布置同样的作业。不管你擅长数学还是语文，作业量一样。这在现实中显然不是最优的。

下一节的GAT会解决这个问题——让不同的邻居有不同的"发言权"。

---

## 32.3 图注意力网络(GAT)

### 32.3.1 注意力机制的直觉

回想一下GCN的局限：所有邻居都被同等对待（除了度数归一化）。但在现实中，不同的朋友对你的影响是不同的。

**费曼比喻——社交聚会**：

想象你在一个派对上，周围有10个人。你会：
- 和最亲密的朋友聊得最久（高注意力）
- 对点头之交只是礼貌寒暄（低注意力）
- 完全忽略角落里你不认识的人（零注意力）

**注意力机制**就是让模型学会"该关注谁"。

### 32.3.2 GAT的核心思想

GAT (Graph Attention Network) 由 Veličković 等人在 2017 年提出，核心思想是：

> 为每条边学习一个注意力权重，表示源节点对目标节点的重要性。

#### 注意力系数的计算

对于节点 $i$ 和它的邻居 $j \in \mathcal{N}(i)$，注意力系数 $e_{ij}$ 表示 $j$ 对 $i$ 的重要性：

$$
e_{ij} = \text{LeakyReLU}(a^T [W h_i \| W h_j])
$$

其中：
- $W \in \mathbb{R}^{F' \times F}$：线性变换矩阵
- $a \in \mathbb{R}^{2F'}$：注意力向量（可学习）
- $\|$：拼接操作
- $\text{LeakyReLU}$：激活函数

#### 归一化注意力

使用 softmax 将注意力系数归一化：

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}
$$

### 32.3.3 GAT的传播规则

聚合邻居信息时使用注意力权重：

$$
h_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(l)} W^{(l)} h_j^{(l)}\right)
$$

完整形式（包含自注意力）：

$$
h_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij}^{(l)} W^{(l)} h_j^{(l)}\right)
$$

### 32.3.4 多头注意力(Multi-Head Attention)

为了增强表达能力，GAT使用**多头注意力**机制：

$$
h_i^{(l+1)} = \|_{k=1}^{K} \sigma\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij}^{(k,l)} W^{(k,l)} h_j^{(l)}\right)
$$

在最后一层，可以将多头输出取平均：

$$
h_i^{(L)} = \sigma\left(\frac{1}{K} \sum_{k=1}^{K} \sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij}^{(k,L)} W^{(k,L)} h_j^{(L-1)}\right)
$$

**费曼比喻——多人评估**：

想象你要选一家公司投资。与其只听一个人的意见，不如：
- 问5个不同领域的专家
- 技术专家看技术实力
- 财务专家看盈利能力
- 市场专家看增长潜力
- 最后综合所有人的意见

多头注意力就是这样做"多角度看问题"。

### 32.3.5 GAT的优势

1. **自适应邻居重要性**：不同邻居可以有不同的权重
2. **归纳式学习(Inductive)**：可以处理新节点
3. **计算高效**：注意力机制是局部操作
4. **可解释性**：注意力权重可以解释模型决策

### 32.3.6 GAT与GCN的比较

| 特性 | GCN | GAT |
|------|-----|-----|
| 邻居权重 | 固定（仅度数归一化） | 自适应（注意力学习） |
| 学习类型 | 直推式 | 归纳式 |
| 计算复杂度 | $O(|E|)$ | $O(|E| \times K)$ |
| 可解释性 | 低 | 高（注意力可视化） |
| 表达能力 | 有限 | 更强 |

---

## 32.4 GraphSAGE

### 32.4.1 大规模图的挑战

真实世界的图可能非常庞大：
- Facebook社交网络：数十亿用户
- 淘宝商品图：数十亿商品和关系
- 学术引用网络：数亿论文

**问题**：
1. **内存限制**：无法将整个图加载到内存
2. **计算效率**：GCN/GAT需要处理所有邻居
3. **动态图**：新节点不断加入

### 32.4.2 GraphSAGE的核心思想

GraphSAGE (SAmple and agGreGatE) 由 Hamilton 等人 2017 年提出，核心思想是：

> 采样固定数量的邻居，然后聚合它们的信息。

**费曼比喻——随机采访**：

想象你要了解一个城市的居民意见。你不需要采访所有1000万人，而是：
1. 随机选择1000个人
2. 确保样本覆盖不同区域、年龄、职业
3. 用这1000人的意见代表整个城市

GraphSAGE就是这样做"代表性采样"。

### 32.4.3 邻居采样策略

对于每个节点，不遍历所有邻居，而是**采样固定数量**的邻居：

$$
\mathcal{N}_S(i) = \text{Sample}(\mathcal{N}(i), K)
$$

采样策略：
- **均匀采样**：每个邻居等概率被选中
- **重要性采样**：根据边权重采样
- **随机游走采样**：通过随机游走选择邻居

### 32.4.4 聚合函数(Aggregator)

GraphSAGE支持多种聚合方式：

#### 1. 均值聚合(Mean Aggregator)

$$
h_{\mathcal{N}(i)}^{(l)} = \frac{1}{|\mathcal{N}_S(i)|} \sum_{j \in \mathcal{N}_S(i)} h_j^{(l)}
$$

类似于GCN，但只对采样邻居求平均。

#### 2. LSTM聚合器

用LSTM处理邻居序列：

$$
h_{\mathcal{N}(i)}^{(l)} = \text{LSTM}(\{h_j^{(l)}: j \in \mathcal{N}_S(i)\})
$$

**注意**：LSTM对输入顺序敏感，需要随机打乱邻居顺序来保证排列不变性。

#### 3. 池化聚合(Pooling Aggregator)

先对邻居特征做非线性变换，再取最大值或平均值：

$$
h_{\mathcal{N}(i)}^{(l)} = \max(\{\sigma(W_{pool} h_j^{(l)} + b): j \in \mathcal{N}_S(i)\})
$$

### 32.4.5 GraphSAGE的传播规则

$$
h_i^{(l+1)} = \sigma(W^{(l)} \cdot \text{CONCAT}(h_i^{(l)}, h_{\mathcal{N}(i)}^{(l)}))
$$

或者：

$$
h_i^{(l+1)} = \sigma(W^{(l)} \cdot (h_i^{(l)} + h_{\mathcal{N}(i)}^{(l)}))
$$

### 32.4.6 归纳式学习的优势

GraphSAGE最大的贡献是支持**归纳式学习(Inductive Learning)**：

**直推式(Transductive)**：
- 需要看到所有节点（包括测试节点）
- 新节点加入时需要重新训练

**归纳式(Inductive)**：
- 学习一个函数 $f$，可以将任意节点映射到嵌入空间
- 新节点加入时，直接应用 $f$ 即可

**费曼比喻——学习vs记忆**：

- 直推式学习像是在背答案：题目必须见过才能答
- 归纳式学习像是在学方法：掌握方法后，新题目也能做

---

## 32.5 GNN应用实践

### 32.5.1 任务类型总览

| 任务类型 | 描述 | 示例 |
|---------|------|------|
| **节点分类** | 预测每个节点的标签 | 社交网络用户兴趣预测 |
| **链接预测** | 预测两个节点之间是否存在边 | 推荐系统中的好友推荐 |
| **图分类** | 预测整个图的标签 | 分子毒性预测 |
| **图生成** | 生成新的图结构 | 新药分子设计 |

### 32.5.2 节点分类

**问题定义**：给定部分有标签的节点，预测其余节点的标签。

**经典数据集**：Cora
- 2,708篇机器学习论文
- 7个类别（神经网络、强化学习等）
- 5,429条引用关系

**实现思路**：

```python
# 伪代码
model = GCN(input_dim=1433, hidden_dim=64, output_dim=7, num_layers=2)
optimizer = Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    logits = model(features, adj)  # 前向传播
    loss = CrossEntropyLoss(logits[train_mask], labels[train_mask])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**费曼比喻**：就像在派对上猜职业。你知道一些人的职业（有标签），通过他们的朋友圈（图结构）和谈话内容（特征），推测其他人的职业。

### 32.5.3 链接预测

**问题定义**：预测两个节点之间是否存在边（或边的权重）。

**应用场景**：
- 社交网络：好友推荐
- 电商平台："买了X的人也买了Y"
- 知识图谱：补全缺失的关系

**常用方法**：

1. **解码器(Decoder)**：给定两个节点的嵌入 $h_i$ 和 $h_j$，预测边的存在概率

常见的解码器：

- **点积**：$P(e_{ij}) = \sigma(h_i^T h_j)$
- **MLP**：$P(e_{ij}) = \text{MLP}(h_i \| h_j)$
- **双线性**：$P(e_{ij}) = \sigma(h_i^T W h_j)$

2. **负采样**：训练时需要构造负样本（不存在边的节点对）

**损失函数**：

$$
\mathcal{L} = -\sum_{(i,j) \in E} \log P(e_{ij}) - \sum_{(i,j) \notin E'} \log (1 - P(e_{ij}))
$$

其中 $E'$ 是采样的负边集合。

### 32.5.4 图分类：分子性质预测

**问题定义**：预测整个分子的性质（如溶解度、毒性）。

**经典数据集**：
- **MUTAG**：188个分子，预测致突变性
- **QM9**：13万个分子，预测量子化学性质
- **ZINC**：25万个分子，预测分子溶解度

**GNN for Graph Classification**的步骤：

1. **节点嵌入**：通过GNN学习每个原子的表示
2. **图池化(Readout)**：将所有节点表示聚合成图表示
   - 全局平均池化：$h_G = \frac{1}{n} \sum_{i=1}^n h_i$
   - 全局最大池化：$h_G = \max_{i=1}^n h_i$
   - 注意力池化：$h_G = \sum_{i=1}^n \alpha_i h_i$
3. **分类/回归**：用MLP预测最终标签

**费曼比喻**：

想象你要判断一道菜是否好吃。你会：
1. 品尝每种食材（节点嵌入）
2. 综合整体味道（图池化）
3. 给出评分（分类/回归）

分子性质预测就是这样的过程：先看每个原子，再看整体结构，最后预测性质。

### 32.5.5 图注意力可视化

GAT的一个重要优势是**可解释性**。我们可以可视化注意力权重来理解模型：

```python
# 可视化GAT的注意力权重
def visualize_attention(model, data, node_idx):
    """可视化某个节点的注意力权重"""
    model.eval()
    with torch.no_grad():
        _, attention_weights = model(data.x, data.edge_index, return_attention=True)
    
    # attention_weights[i] 表示节点i对其邻居的注意力
    neighbor_attention = attention_weights[node_idx]
    
    # 可视化...
```

**实际意义**：
- 在分子图中，可以看到哪些原子对性质预测最重要
- 在社交网络中，可以看到哪些关系对分类最关键

---

## 32.6 前沿与展望

### 32.6.1 从CNN到Transformer：Graph Transformers

Transformer在自然语言处理和计算机视觉中取得了巨大成功。如何将注意力机制推广到图？

**挑战**：
- Transformer假设序列有位置信息（positional encoding）
- 图没有自然的"位置"概念

**解决方案**：

#### Graphormer (2021)

使用三种结构编码：
1. **中心性编码(Centrality Encoding)**：基于节点度数的编码
2. **空间编码(Spatial Encoding)**：基于节点间最短路径的编码
3. **边编码(Edge Encoding)**：在注意力中融入边特征

#### Graph GPS (2022)

结合局部消息传递和全局注意力：

$$
h^{(l+1)} = \text{MPNN}(h^{(l)}) + \text{Transformer}(h^{(l)})
$$

这种混合架构既保留了GNN的局部性，又引入了Transformer的全局建模能力。

### 32.6.2 几何GNN：处理3D分子结构

传统的GNN将分子看作2D图（拓扑结构）。但分子的**三维结构**决定了其性质：

- 两个分子可能有相同的化学式但不同的3D结构（同分异构体）
- 蛋白质的功能高度依赖于其折叠结构

**几何GNN的挑战**：
- 需要处理节点的位置信息 $(x, y, z)$
- 需要满足**等变性(Equivariance)**：旋转/平移输入，输出也应该相应旋转/平移

**代表工作**：

#### SchNet (2018)

使用连续滤波器处理连续距离：

$$
h_i^{(l+1)} = \sum_{j \in \mathcal{N}(i)} h_j^{(l)} \cdot f^{(l)}(d_{ij})
$$

其中 $d_{ij} = \|p_i - p_j\|$ 是原子间距离，$f^{(l)}$ 是可学习的径向基函数。

#### DimeNet (2020)

不仅考虑距离，还考虑**角度信息**：

$$
m_{ij} = f^{(l)}(d_{ij}, \alpha_{ijk})
$$

其中 $\alpha_{ijk}$ 是键角。

#### Equivariant GNNs (EGNN, 2021)

设计严格满足等变性的网络：

$$
h_i^{(l+1)}, p_i^{(l+1)} = f(h_i^{(l)}, p_i^{(l)}, \{h_j^{(l)}, p_j^{(l)}\}_{j \in \mathcal{N}(i)})
$$

其中 $p_i^{(l+1)}$ 是更新的位置。

### 32.6.3 动态图网络

真实世界的图往往是**动态变化**的：

- 社交网络：用户不断加入、关系不断建立
- 交通网络：道路拥堵实时变化
- 金融网络：交易关系持续更新

**动态图GNN的挑战**：
1. **时间建模**：如何建模图的演化
2. **增量更新**：新边/新节点到来时如何高效更新
3. **长期依赖**：捕捉长期的结构变化

**代表工作**：

#### TGAT (Temporal Graph Attention)

将时间编码到注意力机制中：

$$
e_{ij}(t) = f(h_i(t), h_j(t), t - t_{ij})
$$

其中 $t_{ij}$ 是边 $(i,j)$ 建立的时间。

#### TGN (Temporal Graph Networks)

使用记忆模块(Memory Module)存储历史信息：

$$
s_i(t) = \text{GRU}(s_i(t^-), m_i(t))
$$

其中 $s_i$ 是节点 $i$ 的记忆状态，$m_i$ 是新消息。

### 32.6.4 大语言模型与图的结合

最新研究趋势是将大语言模型(LLM)与GNN结合：

#### 1. LLM作为特征提取器

用LLM编码文本特征，再用GNN聚合：

$$
h_i^{text} = \text{LLM}(\text{text}_i)
$$

$$
h_i^{final} = \text{GNN}(\{h_j^{text}\}_{j \in \mathcal{N}(i) \cup \{i\}})
$$

应用：知识图谱补全、学术引用网络分析

#### 2. 图指令微调

设计图相关的指令来微调LLM：

```
指令: 预测以下两个节点之间是否存在边？
节点A: [特征描述]
节点B: [特征描述]
共同邻居: [列表]

回答: 是/否
```

#### 3. GraphGPT 系列

专门用于图数据的预训练模型，结合图结构预训练和语言模型能力。

### 32.6.5 开放问题与未来方向

1. **可扩展性**：如何扩展到十亿级节点？
2. **可解释性**：如何让GNN的决策更可解释？
3. **少样本学习**：如何用小样本适应新任务？
4. **因果推理**：如何从关联中发现因果？
5. **与物理的结合**：如何将物理约束融入GNN？

---

## 本章总结

### 核心概念回顾

1. **图数据**：节点+边，表示实体和关系
2. **GCN**：基于谱图论的消息传递，归一化邻接矩阵
3. **GAT**：注意力机制，自适应邻居重要性
4. **GraphSAGE**：采样+聚合，支持归纳式学习
5. **应用**：节点分类、链接预测、图分类
6. **前沿**：Graph Transformers、几何GNN、动态图网络

### 算法选择指南

| 场景 | 推荐算法 | 理由 |
|------|---------|------|
| 小图，节点分类 | GCN | 简单高效 |
| 需要可解释性 | GAT | 注意力可视化 |
| 大图，新节点 | GraphSAGE | 归纳式学习 |
| 3D分子结构 | SchNet/DimeNet | 处理几何信息 |
| 动态图 | TGAT/TGN | 时间建模 |

### 从本章走向实践

1. **从简单开始**：先用GCN理解图学习的基本思想
2. **可视化**：用注意力权重理解模型在看什么
3. **数据集**：从Cora、PubMed、QM9开始
4. **工具库**：PyTorch Geometric、DGL、NetworkX
5. **竞赛**：Kaggle的图相关比赛、OGB基准测试

---

## 参考文献

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In *International Conference on Learning Representations (ICLR)*.

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. In *International Conference on Learning Representations (ICLR)*.

Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. In *Advances in Neural Information Processing Systems (NIPS)*, 30.

Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. In *International Conference on Machine Learning (ICML)*, 1263-1272.

Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? In *International Conference on Learning Representations (ICLR)*.

Wu, F., Souza, A., Zhang, T., Fifty, C., Yu, T., & Weinberger, K. (2019). Simplifying graph convolutional networks. In *International Conference on Machine Learning (ICML)*, 6861-6871.

Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K., & Pande, V. (2018). MoleculeNet: A benchmark for molecular machine learning. *Chemical Science*, 9(2), 513-530.

Ying, R., He, R., Chen, K., Eksombatchai, P., Hamilton, W. L., & Leskovec, J. (2018). Graph convolutional neural networks for web-scale recommender systems. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 974-983.

Veličković, P., Fedus, W., Hamilton, W. L., Liò, P., Bengio, Y., & Hjelm, R. D. (2019). Deep graph infomax. In *International Conference on Learning Representations (ICLR)*.

Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., & Leskovec, J. (2020). Strategies for pre-training graph neural networks. In *International Conference on Learning Representations (ICLR)*.

---

## 练习题

### 基础练习题

**练习 32.1：图的基本操作**

给定以下邻接矩阵表示的无向图：

$$
A = \begin{bmatrix}
0 & 1 & 1 & 0 \\
1 & 0 & 1 & 1 \\
1 & 1 & 0 & 1 \\
0 & 1 & 1 & 0
\end{bmatrix}
$$

请完成：
1. 画出这个图的结构
2. 计算度矩阵 $D$
3. 计算拉普拉斯矩阵 $L = D - A$
4. 计算归一化邻接矩阵 $\hat{A} = D^{-1/2} A D^{-1/2}$

---

**练习 32.2：GCN的前向传播**

假设有一个3节点图，特征矩阵为：

$$
X = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1
\end{bmatrix}
$$

邻接矩阵为：

$$
A = \begin{bmatrix}
0 & 1 & 0 \\
1 & 0 & 1 \\
0 & 1 & 0
\end{bmatrix}
$$

权重矩阵为 $W = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$（单位矩阵），激活函数为恒等函数。

请计算GCN一层传播后的输出 $H^{(1)}$。

---

**练习 32.3：注意力权重的计算**

在GAT中，假设节点 $i$ 和它的两个邻居 $j_1, j_2$ 的特征分别为：
- $h_i = [1, 0]^T$
- $h_{j_1} = [0, 1]^T$
- $h_{j_2} = [1, 1]^T$

权重矩阵 $W = I$（单位矩阵），注意力向量 $a = [1, 1, 1, 1]^T$。

使用点积注意力（无LeakyReLU简化计算），计算归一化注意力权重 $\alpha_{ij_1}$ 和 $\alpha_{ij_2}$。

---

### 进阶练习题

**练习 32.4：GCN的谱域推导**

证明：当使用一阶切比雪夫多项式近似（K=1）时，图卷积可以表示为：

$$
g_{\theta} * x \approx \theta (I + D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) x
$$

提示：
1. 切比雪夫多项式 $T_0(x) = 1$，$T_1(x) = x$
2. 归一化的拉普拉斯 $\tilde{L} = \frac{2L}{\lambda_{max}} - I$
3. 假设 $\lambda_{max} \approx 2$

---

**练习 32.5：GraphSAGE的归纳式学习分析**

解释为什么GraphSAGE是归纳式的，而GCN是直推式的。从数学角度说明：

1. GCN的归一化邻接矩阵 $\hat{A}$ 依赖于整个图的度数
2. GraphSAGE的聚合函数 $f$ 可以独立于具体图结构定义
3. 当新节点加入时，为什么GCN需要重新训练，而GraphSAGE不需要

---

**练习 32.6：多头注意力的表达能力**

假设使用 $K$ 个注意力头，每个头学习不同的注意力权重。证明：

1. 当 $K=1$ 时，GAT退化为类似GCN的形式（固定权重）
2. 增加 $K$ 可以增加模型的表达能力
3. 过多的 $K$ 会导致什么问题？如何缓解？

---

### 挑战练习题

**练习 32.7：链接预测的负采样策略**

在链接预测任务中，需要构造负样本（不存在的边）。设计一个负采样策略，要求：

1. 简单随机采样的问题是什么？（提示：考虑度分布）
2. 提出一种基于节点度的改进采样策略
3. 分析你的策略的优缺点
4. （选做）实现这个策略并用实验验证

---

**练习 32.8：GNN层数与过度平滑**

研究发现，增加GNN层数会导致**过度平滑(over-smoothing)**现象：所有节点的表示变得相似。请：

1. 从数学角度解释为什么会发生过度平滑（提示：考虑 $\hat{A}^k$ 当 $k \to \infty$ 时的行为）
2. 查阅文献，列举至少3种解决过度平滑的方法
3. 设计一个实验验证过度平滑现象

---

**练习 32.9：设计一个用于分子性质预测的GNN**

设计一个GNN来预测分子的水溶性(logP)。要求：

1. 说明节点特征和边特征的设计（原子类型、电荷、键类型等）
2. 选择GCN、GAT、GraphSAGE中的至少一种作为基础架构，并说明理由
3. 设计图的读出(readout)操作来得到分子级别的表示
4. 画出完整的模型架构图
5. （选做）在QM9或ESOL数据集上实现并测试你的模型

---

## 延伸阅读

### 经典论文

1. **GCN**: Kipf & Welling (2017) - 图卷积网络的奠基工作
2. **GAT**: Veličković et al. (2018) - 注意力机制引入图学习
3. **GraphSAGE**: Hamilton et al. (2017) - 大规模图学习
4. **GIN**: Xu et al. (2019) - 图同构网络，理论分析GNN表达能力

### 综述文章

1. Wu et al. (2020) - "A Comprehensive Survey on Graph Neural Networks"
2. Zhou et al. (2020) - "Graph Neural Networks: A Review of Methods and Applications"

### 在线资源

1. **PyTorch Geometric文档**: https://pytorch-geometric.readthedocs.io/
2. **DGL教程**: https://docs.dgl.ai/
3. **Stanford CS224W**: 图机器学习课程
4. **OGB基准**: https://ogb.stanford.edu/

---

*本章完。继续探索图的奥秘，你会发现世界充满了连接的美妙！*
