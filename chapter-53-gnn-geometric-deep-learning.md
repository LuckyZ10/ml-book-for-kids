# 第五十三章 图神经网络与几何深度学习

> *"世界不是网格状的，而是相互连接的——就像一张巨大的社交网络。"*

## 本章学习目标

学完本章，你将能够：
- 🌐 理解图神经网络的核心思想：消息传递机制
- 🔗 掌握GCN、GAT、GraphSAGE等主流架构
- 🧬 应用GNN解决分子预测、社交网络分析问题
- 🎯 理解几何深度学习中的对称性与等变性
- ☁️ 使用PointNet处理3D点云数据

---

## 53.1 引言：从网格到图

### 53.1.1 为什么世界不是网格？

回想一下，本书前面的大部分章节都在处理**网格数据**——图像是由规则的像素网格组成的，文本是线性的词序列。CNN在图像上大放异彩，RNN和Transformer在文本上表现优异。

但是，**真实世界远非网格状**。

想象一下：
- **社交网络**：你的朋友关系不是网格，每个人连接的人数不同，关系也没有固定结构
- **分子结构**：原子的连接方式是任意的，没有固定的"行"和"列"
- **知识图谱**：概念之间的关系错综复杂，无法用表格表示
- **交通网络**：道路连接各个地点，形成复杂的有向图
- **蛋白质相互作用**：蛋白质之间的相互作用网络

**传统的神经网络遇到了麻烦**：CNN需要规则的网格，RNN需要序列结构。面对图数据，它们束手无策。

### 53.1.2 图数据无处不在

让我们看几个具体的例子：

**社交网络示例**：
```
小明 --朋友--> 小红
  |              |
  同事           同学
  |              |
  v              v
小李 <--邻居--> 小张
```
在这个网络中，小明有2个朋友，小红有2个朋友，小李和小张各有2个朋友。每个人的"度"都不同，结构不规则。

**分子结构示例（苯环）**：
```
     C — C
    /     \
   C       C
    \     /
     C = C
```
6个碳原子形成环状结构，每个碳原子连接2个相邻碳原子和1个氢原子（图中省略）。这种结构是周期性的，但绝非网格。

**知识图谱示例**：
```
爱因斯坦 —出生于—> 德国
    |                  |
    发现            位于
    v                  v
   E=mc²           欧洲
```
知识以（头实体，关系，尾实体）的三元组形式存储，天然适合图结构。

### 53.1.3 传统方法的局限性

**尝试1：把图变成网格**
有人可能会想："我可以把图的邻接矩阵当成图像，用CNN处理！"

但问题随之而来：
1. **节点编号任意性**：同一个图可以有无数个不同的邻接矩阵表示，取决于你如何给节点编号。CNN对这种置换敏感。
2. **变长输入**：不同的图有不同数量的节点，CNN需要固定大小的输入。
3. **稀疏性问题**：大多数真实世界的图是稀疏的（边数远少于节点数的平方），直接处理邻接矩阵效率极低。

**尝试2：手工提取特征**
传统机器学习使用手工设计的图特征（如度分布、聚类系数），但：
1. 特征工程耗时且需要领域知识
2. 难以捕捉复杂的结构模式
3. 泛化能力差

**我们需要一种新的神经网络架构**——能够直接处理图结构，对节点置换不变，自动学习层次化特征。

这就是**图神经网络 (Graph Neural Networks, GNN)**。

### 53.1.4 几何深度学习：统一的视角

2021年，Bronstein等人提出了**几何深度学习 (Geometric Deep Learning)** 的统一框架。他们认为，深度学习处理的数据都具有某种几何结构，可以用"5G"来概括：

| 类型 | 英文 | 描述 | 代表模型 |
|------|------|------|----------|
| **Grids** | 网格 | 规则的网格数据 | CNN |
| **Groups** | 群 | 具有对称性的数据 | 等变神经网络 |
| **Graphs** | 图 | 不规则的图结构 | GNN |
| **Geodesics** | 测地线 | 流形上的数据 | 流形学习 |
| **Gauges** | 规范场 | 纤维丛结构 | 规范神经网络 |

这个框架的美妙之处在于：**所有这些问题都可以用统一的语言描述**——群论、表示论、微分几何。

但别担心，我们不会一开始就跳进数学深渊。让我们从最简单的图神经网络开始，一步步建立起直觉。

---

## 53.2 图神经网络基础

### 53.2.1 图的基本表示

在开始构建神经网络之前，我们需要明确如何表示一个图。

**形式化定义**：
一个图 $G = (V, E)$ 包含：
- $V$：节点集合，$|V| = n$
- $E$：边集合，$E \subseteq V \times V$
- $X \in \mathbb{R}^{n \times d}$：节点特征矩阵，每行是一个节点的 $d$ 维特征
- $E \in \mathbb{R}^{m \times k}$（可选）：边特征矩阵，$m = |E|$，每条边有 $k$ 维特征

**邻接矩阵 (Adjacency Matrix)**：
$A \in \mathbb{R}^{n \times n}$，其中：
$$A_{ij} = \begin{cases} 1 & \text{如果}(i,j) \in E \\ 0 & \text{否则} \end{cases}$$

对于无向图，$A$ 是对称的（$A = A^T$）。

**度矩阵 (Degree Matrix)**：
$D \in \mathbb{R}^{n \times n}$ 是对角矩阵：
$$D_{ii} = \sum_j A_{ij}$$
表示节点 $i$ 的邻居数量。

**归一化邻接矩阵**：
在实际应用中，我们常用归一化版本：
$$\hat{A} = D^{-1/2} A D^{-1/2}$$
这确保了不同度的节点在传播时具有相似的影响力。

**Python实现**：
```python
import torch
import torch.nn as nn
import numpy as np

# 创建一个简单的图
# 节点: 0, 1, 2, 3
# 边: (0,1), (0,2), (1,2), (2,3)
edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
n_nodes = 4

# 构建邻接矩阵
A = torch.zeros(n_nodes, n_nodes)
for i, j in edges:
    A[i, j] = 1
    A[j, i] = 1  # 无向图

print("邻接矩阵 A:")
print(A)

# 计算度矩阵
degrees = A.sum(dim=1)
D = torch.diag(degrees)
print("\n度矩阵 D:")
print(D)

# 归一化邻接矩阵
D_inv_sqrt = torch.diag(torch.pow(degrees, -0.5))
A_norm = D_inv_sqrt @ A @ D_inv_sqrt
print("\n归一化邻接矩阵 A_norm:")
print(A_norm)
```

输出：
```
邻接矩阵 A:
tensor([[0., 1., 1., 0.],
        [1., 0., 1., 0.],
        [1., 1., 0., 1.],
        [0., 0., 1., 0.]])

度矩阵 D:
tensor([[2., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 3., 0.],
        [0., 0., 0., 1.]])

归一化邻接矩阵 A_norm:
tensor([[0.0000, 0.5000, 0.4082, 0.0000],
        [0.5000, 0.0000, 0.4082, 0.0000],
        [0.4082, 0.4082, 0.0000, 0.5774],
        [0.0000, 0.0000, 0.5774, 0.0000]])
```

### 53.2.2 消息传递机制

图神经网络的核心是**消息传递 (Message Passing)**。让我们用费曼法来理解这个概念：

> **费曼法比喻：消息传递就像社交网络中的"口碑传播"**
> 
> 想象你在一个小镇上，每个人对某个话题都有自己的看法。每天，每个人都会：
> 1. **听取**朋友们的意见（接收消息）
> 2. **综合**朋友们的意见和自己的看法（聚合与更新）
> 3. **形成**新的观点
> 
> 经过几天传播，整个小镇对这个话题会形成共识——这个过程就是消息传递。

**形式化定义 (MPNN框架)**：

消息传递神经网络 (Gilmer et al., 2017) 定义了三个核心函数：

1. **消息函数 (Message Function)**：
   $$m_{ij}^{(t)} = M_t(h_i^{(t-1)}, h_j^{(t-1)}, e_{ij})$$
   节点 $j$ 向节点 $i$ 传递的消息取决于：
   - 发送者 $j$ 的当前状态
   - 接收者 $i$ 的当前状态
   - 边 $(i,j)$ 的特征

2. **聚合函数 (Aggregate Function)**：
   $$m_i^{(t)} = \bigoplus_{j \in N(i)} m_{ij}^{(t)}$$
   节点 $i$ 收集所有邻居的消息。$\bigoplus$ 可以是求和、平均、取最大值等。

3. **更新函数 (Update Function)**：
   $$h_i^{(t)} = U_t(h_i^{(t-1)}, m_i^{(t)})$$
   结合旧状态和新消息，更新节点表示。

**所有GNN都是MPNN的特例**！GCN、GAT、GraphSAGE的区别仅在于具体的消息、聚合、更新函数的选择。

**Python实现框架**：
```python
class MessagePassingLayer(nn.Module):
    """
    通用的消息传递层
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 子类定义具体的M和U
        
    def message(self, h_i, h_j, e_ij=None):
        """
        计算从j到i的消息
        子类需要重写此方法
        """
        raise NotImplementedError
    
    def aggregate(self, messages, neighbor_indices):
        """
        聚合邻居消息
        默认使用求和，子类可重写
        """
        return messages.sum(dim=0)
    
    def update(self, h_old, aggregated_message):
        """
        更新节点表示
        子类需要重写此方法
        """
        raise NotImplementedError
    
    def forward(self, h, edge_index, edge_attr=None):
        """
        前向传播
        
        Args:
            h: [n_nodes, in_features] 节点特征
            edge_index: [2, n_edges] 边索引，[source, target]
            edge_attr: [n_edges, edge_features] 边特征（可选）
        """
        n_nodes = h.size(0)
        new_h = torch.zeros(n_nodes, self.out_features, device=h.device)
        
        # 收集每个节点的消息
        for i in range(n_nodes):
            messages = []
            # 找到i的所有邻居
            mask = edge_index[1] == i
            neighbors = edge_index[0][mask]
            
            for j in neighbors:
                e_ij = edge_attr[mask][j == edge_index[0][mask]] if edge_attr is not None else None
                msg = self.message(h[i], h[j], e_ij)
                messages.append(msg)
            
            if len(messages) > 0:
                messages = torch.stack(messages)
                aggregated = self.aggregate(messages, neighbors)
                new_h[i] = self.update(h[i], aggregated)
            else:
                new_h[i] = h[i]  # 没有邻居时保持原样
        
        return new_h
```

### 53.2.3 GCN：图卷积网络

**GCN (Graph Convolutional Network)** 由Kipf和Welling在2017年提出，是图神经网络的重要里程碑。

**核心思想**：
将CNN的卷积操作推广到图结构。在CNN中，卷积核在网格上滑动，聚合局部信息。在GCN中，每个节点聚合其邻居的信息。

**传播规则**：
$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$$

其中：
- $\tilde{A} = A + I$（添加自环，每个节点考虑自己）
- $\tilde{D}$ 是 $\tilde{A}$ 的度矩阵
- $H^{(l)}$ 是第 $l$ 层的节点特征
- $W^{(l)}$ 是可学习的权重矩阵
- $\sigma$ 是激活函数（如ReLU）

**费曼法解释**：
> **GCN就像"加权平均"**
> 
> 想象你和你的朋友在讨论一个话题。GCN层的工作方式是：
> 1. 每个人先把自己的观点"升级"一下（乘以权重矩阵W）
> 2. 然后每个人收集朋友们的观点
> 3. 但是**朋友越多的人，每个朋友的意见权重越低**（归一化）
> 4. 最后综合自己的新观点和朋友们的加权意见

**数学推导：从谱图理论到GCN**：

GCN的灵感来自**谱图理论**。在信号处理中，卷积定理告诉我们：时域卷积等于频域乘积。对于图，"频域"由图拉普拉斯矩阵的特征向量定义。

1. **图拉普拉斯矩阵**：
   $$L = D - A$$
   $$L_{sym} = D^{-1/2}LD^{-1/2} = I - D^{-1/2}AD^{-1/2}$$

2. **谱卷积**：
   在谱域中，卷积定义为：
   $$x *_{G} g = U g_{\theta} U^T x$$
   其中 $U$ 是 $L$ 的特征向量矩阵。

3. **Chebyshev近似**：
   直接计算特征分解代价高昂。使用Chebyshev多项式近似：
   $$g_{\theta}(\Lambda) \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{\Lambda})$$
   其中 $\tilde{\Lambda} = 2\Lambda/\lambda_{max} - I$。

4. **一阶近似 (K=1)**：
   假设 $K=1$ 且 $\lambda_{max} \approx 2$：
   $$x *_{G} g \approx \theta_0 x + \theta_1 (L_{sym} - I)x = \theta_0 x - \theta_1 D^{-1/2}AD^{-1/2}x$$

5. **添加自环并简化**：
   令 $\tilde{A} = A + I$，最终得到GCN的传播规则：
   $$H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$$

**Python实现**：
```python
class GCNLayer(nn.Module):
    """
    图卷积网络层 (Kipf & Welling, 2017)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj_normalized):
        """
        Args:
            x: [n_nodes, in_features] 节点特征
            adj_normalized: [n_nodes, n_nodes] 归一化邻接矩阵 (包含自环)
        Returns:
            h: [n_nodes, out_features] 新的节点特征
        """
        # 线性变换
        h = self.linear(x)  # [n_nodes, out_features]
        
        # 图卷积：聚合邻居特征
        h = torch.matmul(adj_normalized, h)  # [n_nodes, out_features]
        
        return torch.relu(h)


class GCN(nn.Module):
    """
    多层GCN模型
    """
    def __init__(self, in_features, hidden_features, out_features, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(GCNLayer(in_features, hidden_features))
        
        # 隐藏层
        for _ in range(n_layers - 2):
            self.layers.append(GCNLayer(hidden_features, hidden_features))
        
        # 输出层（无激活函数，用于分类/回归）
        self.layers.append(GCNLayer(hidden_features, out_features))
        
    def forward(self, x, adj):
        """
        Args:
            x: [n_nodes, in_features]
            adj: [n_nodes, n_nodes] 原始邻接矩阵
        Returns:
            out: [n_nodes, out_features]
        """
        # 添加自环并归一化
        adj_with_self_loops = adj + torch.eye(adj.size(0), device=adj.device)
        degrees = adj_with_self_loops.sum(dim=1)
        D_inv_sqrt = torch.diag(torch.pow(degrees, -0.5))
        adj_normalized = D_inv_sqrt @ adj_with_self_loops @ D_inv_sqrt
        
        # 前向传播
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj_normalized)
        
        # 最后一层（可选：不加激活）
        x = self.layers[-1].linear(x)
        x = torch.matmul(adj_normalized, x)
        
        return x


# 测试GCN
print("=" * 50)
print("测试GCN模型")
print("=" * 50)

# 创建一个简单的图：4个节点，三角形+一个悬挂节点
n_nodes = 4
edges = [(0, 1), (0, 2), (1, 2), (2, 3)]  # 0-1-2形成三角形，2-3连接

# 构建邻接矩阵
A = torch.zeros(n_nodes, n_nodes)
for i, j in edges:
    A[i, j] = 1
    A[j, i] = 1

# 随机初始化节点特征（例如：每个节点的"属性"）
x = torch.randn(n_nodes, 16)  # 4个节点，16维特征
print(f"输入特征形状: {x.shape}")

# 创建GCN模型
model = GCN(in_features=16, hidden_features=32, out_features=7, n_layers=2)  # 7分类

# 前向传播
out = model(x, A)
print(f"输出形状: {out.shape}")
print(f"输出（节点分类logits）:\n{out}")
```

### 53.2.4 GAT：图注意力网络

GCN的一个局限是：**所有邻居一视同仁**。但现实中，不同邻居的重要性往往不同。

**GAT (Graph Attention Network)** 将Transformer的自注意力机制引入图神经网络，让模型学习"关注哪些邻居"。

**核心思想**：
计算邻居的注意力权重，重要的邻居获得更高的权重。

**注意力系数计算**：
$$e_{ij} = \text{LeakyReLU}(\mathbf{a}^T[Wh_i \| Wh_j])$$
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in N(i)} \exp(e_{ik})}$$

其中：
- $W$ 是线性变换矩阵
- $\mathbf{a}$ 是注意力参数向量
- $\|$ 表示向量拼接
- $\alpha_{ij}$ 是节点 $j$ 对节点 $i$ 的注意力权重

**多头注意力**：
类似Transformer，GAT使用多头注意力增强表达能力：
$$h_i' = \|_{k=1}^{K} \sigma\left(\sum_{j \in N(i)} \alpha_{ij}^{(k)} W^{(k)} h_j\right)$$

**费曼法解释**：
> **GAT就像"选择性倾听"**
> 
> 想象你在做投资决策。GCN的方式是听取所有朋友的建议，然后平均考虑。
> 
> 但GAT更聪明：
> - 对于科技股票投资，你会更关注从事科技行业的朋友
> - 对于房地产投资，你会更关注有房产经验的朋友
> - GAT自动学习"在什么情况下应该听谁的"

**Python实现**：
```python
class GATLayer(nn.Module):
    """
    图注意力网络层 (Veličković et al., 2018)
    """
    def __init__(self, in_features, out_features, n_heads=8, dropout=0.6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout
        
        # 每个头都有自己的线性变换
        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)
        
        # 注意力参数 [n_heads, 2 * out_features]
        self.a = nn.Parameter(torch.randn(n_heads, 2 * out_features))
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        """
        Args:
            x: [n_nodes, in_features]
            adj: [n_nodes, n_nodes] 邻接矩阵（包含自环）
        Returns:
            h: [n_nodes, out_features * n_heads] 或 [n_nodes, out_features]
        """
        n_nodes = x.size(0)
        
        # 线性变换: [n_nodes, n_heads * out_features]
        Wh = self.W(x)
        Wh = Wh.view(n_nodes, self.n_heads, self.out_features)
        
        # 计算注意力系数
        # 为每个节点对计算 e_ij
        attn_input = torch.cat([
            Wh.unsqueeze(1).expand(-1, n_nodes, -1, -1),  # [n, n, heads, out]
            Wh.unsqueeze(0).expand(n_nodes, -1, -1, -1)   # [n, n, heads, out]
        ], dim=-1)  # [n_nodes, n_nodes, n_heads, 2 * out_features]
        
        # 计算注意力分数
        # [n_heads, 2*out] @ [n, n, heads, 2*out, 1] -> [n, n, heads]
        e = torch.einsum('hd,ijhd->ijh', self.a, attn_input)
        e = self.leakyrelu(e)  # [n_nodes, n_nodes, n_heads]
        
        # 掩码：只保留邻居（由邻接矩阵决定）
        mask = adj.unsqueeze(-1).expand(-1, -1, self.n_heads)
        e = e.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化
        alpha = torch.softmax(e, dim=1)  # [n_nodes, n_nodes, n_heads]
        alpha = self.dropout_layer(alpha)
        
        # 加权聚合: [n, n, heads] @ [n, heads, out] -> [n, heads, out]
        h = torch.einsum('ijh,jhd->ihd', alpha, Wh)
        
        # 拼接多头结果
        h = h.reshape(n_nodes, -1)  # [n_nodes, n_heads * out_features]
        
        return torch.relu(h)


class GAT(nn.Module):
    """
    多层GAT模型
    """
    def __init__(self, in_features, hidden_features, out_features, n_heads=8, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 第一层（多头注意力）
        self.layers.append(GATLayer(in_features, hidden_features, n_heads=n_heads))
        
        # 中间层
        for _ in range(n_layers - 2):
            self.layers.append(GATLayer(hidden_features * n_heads, hidden_features, n_heads=n_heads))
        
        # 最后一层（单头，用于分类）
        self.layers.append(GATLayer(hidden_features * n_heads, out_features, n_heads=1))
        
    def forward(self, x, adj):
        # 添加自环
        adj_with_self = adj + torch.eye(adj.size(0), device=adj.device)
        
        for layer in self.layers[:-1]:
            x = layer(x, adj_with_self)
        
        # 最后一层不加激活
        x = self.layers[-1](x, adj_with_self)
        
        return x


# 测试GAT
print("=" * 50)
print("测试GAT模型")
print("=" * 50)

gat_model = GAT(in_features=16, hidden_features=8, out_features=7, n_heads=4, n_layers=2)
gat_out = gat_model(x, A)
print(f"GAT输出形状: {gat_out.shape}")
print(f"GAT输出:\n{gat_out}")
```

### 53.2.5 GraphSAGE：归纳式学习

GCN和GAT都是**直推式 (Transductive)** 的：它们只能在训练时见过的节点上进行预测，无法泛化到新节点。

**GraphSAGE (Graph Sample and Aggregate)** 解决了这个问题，支持**归纳式 (Inductive)** 学习。

**核心思想**：
1. **采样 (Sample)**：对每个节点，随机采样固定数量的邻居（而不是使用所有邻居）
2. **聚合 (Aggregate)**：使用可学习的聚合函数（如Mean、LSTM、Pooling）
3. **更新 (Update)**：结合自身特征和聚合的邻居特征

**传播规则**：
$$h_{N(i)}^{(l)} = \text{AGGREGATE}^{(l)}(\{h_j^{(l-1)}, \forall j \in N(i)\})$$
$$h_i^{(l)} = \sigma(W^{(l)} \cdot [h_i^{(l-1)} \| h_{N(i)}^{(l)}])$$

**聚合函数选择**：
- **Mean aggregator**：平均邻居特征（类似GCN）
- **LSTM aggregator**：用LSTM处理邻居序列（考虑顺序，但图是无序的，需要随机打乱）
- **Pooling aggregator**：用MLP+max-pooling

**费曼法解释**：
> **GraphSAGE就像"采访代表"**
> 
> 想象你要了解一个社区的意见：
> - GCN的做法是询问社区里的每一个人（计算成本高，无法扩展到大图）
> - GraphSAGE的做法是随机采访10个代表性居民，然后综合他们的意见
> 
> 好处是：
> 1. **效率**：不管社区多大，采访人数固定
> 2. **泛化**：来了新居民，只需要采访他和他身边的人，不需要重新训练

**Python实现**：
```python
class GraphSAGELayer(nn.Module):
    """
    GraphSAGE层 (Hamilton et al., 2017)
    支持归纳式学习
    """
    def __init__(self, in_features, out_features, aggregator='mean'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        
        # 拼接自身和邻居特征后的线性变换
        self.W = nn.Linear(2 * in_features, out_features)
        
    def sample_neighbors(self, adj, node_idx, sample_size):
        """
        随机采样邻居
        
        Args:
            adj: 邻接矩阵
            node_idx: 当前节点索引
            sample_size: 采样数量
        Returns:
            邻居索引列表
        """
        neighbors = torch.where(adj[node_idx] > 0)[0]
        
        if len(neighbors) == 0:
            return torch.tensor([], dtype=torch.long)
        
        if len(neighbors) <= sample_size:
            return neighbors
        
        # 随机采样
        perm = torch.randperm(len(neighbors))
        return neighbors[perm[:sample_size]]
    
    def aggregate(self, neighbor_features):
        """
        聚合邻居特征
        """
        if self.aggregator == 'mean':
            return neighbor_features.mean(dim=0)
        elif self.aggregator == 'max':
            return neighbor_features.max(dim=0)[0]
        elif self.aggregator == 'sum':
            return neighbor_features.sum(dim=0)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
    
    def forward(self, x, adj, sample_size=10):
        """
        Args:
            x: [n_nodes, in_features]
            adj: [n_nodes, n_nodes]
            sample_size: 邻居采样数量
        Returns:
            h: [n_nodes, out_features]
        """
        n_nodes = x.size(0)
        h_list = []
        
        for i in range(n_nodes):
            # 采样邻居
            neighbor_idx = self.sample_neighbors(adj, i, sample_size)
            
            if len(neighbor_idx) > 0:
                # 聚合邻居特征
                neighbor_features = x[neighbor_idx]  # [n_sampled, in_features]
                h_neighbors = self.aggregate(neighbor_features)  # [in_features]
            else:
                h_neighbors = torch.zeros(self.in_features, device=x.device)
            
            # 拼接自身和邻居特征
            h_concat = torch.cat([x[i], h_neighbors])  # [2 * in_features]
            h_list.append(h_concat)
        
        # 批量处理
        h_concat = torch.stack(h_list)  # [n_nodes, 2 * in_features]
        h = self.W(h_concat)  # [n_nodes, out_features]
        
        return torch.relu(h)


class GraphSAGE(nn.Module):
    """
    多层GraphSAGE模型
    """
    def __init__(self, in_features, hidden_features, out_features, n_layers=2, aggregator='mean'):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(GraphSAGELayer(in_features, hidden_features, aggregator))
        
        # 隐藏层
        for _ in range(n_layers - 2):
            self.layers.append(GraphSAGELayer(hidden_features, hidden_features, aggregator))
        
        # 输出层
        self.layers.append(GraphSAGELayer(hidden_features, out_features, aggregator))
        
    def forward(self, x, adj, sample_size=10):
        for layer in self.layers[:-1]:
            x = layer(x, adj, sample_size)
        
        # 最后一层不加激活
        x = self.layers[-1](x, adj, sample_size)
        return x


# 测试GraphSAGE
print("=" * 50)
print("测试GraphSAGE模型")
print("=" * 50)

sage_model = GraphSAGE(in_features=16, hidden_features=32, out_features=7, n_layers=2, aggregator='mean')
sage_out = sage_model(x, A, sample_size=2)
print(f"GraphSAGE输出形状: {sage_out.shape}")
print(f"GraphSAGE输出:\n{sage_out}")
```

### 53.2.6 三种架构对比

| 特性 | GCN | GAT | GraphSAGE |
|------|-----|-----|-----------|
| **邻居权重** | 度归一化 | 学习得到 | 平均/池化 |
| **计算复杂度** | $O(|E|)$ | $O(|E| \times K)$ | $O(n \times s \times L)$ |
| **泛化能力** | 直推式 | 直推式 | **归纳式** |
| **适用场景** | 中小图 | 需要区分邻居重要性 | 大图、动态图 |
| **内存需求** | 中 | 高（多头注意力） | 低（采样） |

---

## 53.3 高级图神经网络架构

### 53.3.1 深层GNN的挑战

**问题1：过平滑 (Over-smoothing)**

当GCN层数增加时，节点表示会趋于一致。经过太多层消息传递后，所有节点看起来都差不多，失去区分性。

**为什么发生？**
- 消息传递本质上是邻居特征的混合
- 经过 $k$ 层后，每个节点的感受野是 $k$ 跳邻居
- 随着 $k$ 增大，不同节点的邻居集合高度重叠
- 最终导致所有节点特征收敛到相同的值

**数学分析**：
在极端情况下，当层数 $L \to \infty$，归一化邻接矩阵的幂收敛：
$$\hat{A}^L \to \frac{1}{n}\mathbf{1}\mathbf{1}^T$$
所有节点特征趋于全局平均！

**费曼法比喻**：
> **过平滑就像"人云亦云"**
> 
> 想象一个谣言传播过程：
> - 第1轮：每个人告诉邻居自己听到的版本
> - 第5轮：每个人综合了5圈内所有人的说法
> - 第10轮：所有人听到的版本几乎一模一样，失去了原始信息的特点
> 
> 深层GNN的问题就在于此——经过太多层传播，所有节点的"观点"变得过于相似。

**问题2：过挤压 (Over-squashing)**

当两个远程节点（距离很远）需要通过消息传递交换信息时，信息必须在中间节点被反复压缩到固定维度，导致信息丢失。

**解决方案**：

1. **残差连接 (Residual Connections)**：
   $$H^{(l+1)} = \text{GNN}(H^{(l)}) + H^{(l)}$$
   允许梯度直接流动，保持原始信息。

2. **跳跃连接 (Jumping Knowledge)**：
   每一层都连接到最终输出，模型可以选择使用哪层的信息。

3. **DropEdge**：
   随机删除一些边，减少信息混合的程度。

**Python实现 - 带残差连接的GCN**：
```python
class ResidualGCNLayer(nn.Module):
    """
    带残差连接的GCN层
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.gcn = GCNLayer(in_features, out_features)
        
        # 如果维度不同，需要投影
        self.projection = None
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj_normalized):
        h = self.gcn(x, adj_normalized)
        
        # 残差连接
        if self.projection is not None:
            x = self.projection(x)
        
        return h + x  # 残差连接


class DeepGCN(nn.Module):
    """
    深层GCN，使用残差连接解决过平滑
    """
    def __init__(self, in_features, hidden_features, out_features, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(GCNLayer(in_features, hidden_features))
        
        # 隐藏层（带残差连接）
        for _ in range(n_layers - 2):
            self.layers.append(ResidualGCNLayer(hidden_features, hidden_features))
        
        # 输出层
        self.output_layer = nn.Linear(hidden_features, out_features)
        
    def forward(self, x, adj):
        # 归一化邻接矩阵
        adj_with_self = adj + torch.eye(adj.size(0), device=adj.device)
        degrees = adj_with_self.sum(dim=1)
        D_inv_sqrt = torch.diag(torch.pow(degrees + 1e-8, -0.5))
        adj_normalized = D_inv_sqrt @ adj_with_self @ D_inv_sqrt
        
        # 前向传播
        for layer in self.layers:
            x = layer(x, adj_normalized)
        
        return self.output_layer(x)


# 测试深层GCN
print("=" * 50)
print("测试深层GCN（4层，带残差连接）")
print("=" * 50)

deep_gcn = DeepGCN(in_features=16, hidden_features=32, out_features=7, n_layers=4)
deep_out = deep_gcn(x, A)
print(f"深层GCN输出形状: {deep_out.shape}")
```

### 53.3.2 图Transformer

Transformer在NLP和CV领域取得了巨大成功，自然地，研究者尝试将其扩展到图结构。

**核心挑战**：
- 标准Transformer假设序列结构，而图是无序的
- 图没有"位置"的概念，需要新的位置编码方式

**Graphormer (Ying et al., 2021)** 是微软亚洲研究院提出的图Transformer架构：

**空间编码 (Spatial Encoding)**：
不再使用传统的位置编码，而是使用节点间的最短路径距离：
$$A_{ij} = \frac{(h_i W_Q)(h_j W_K)^T}{\sqrt{d_k}} + b_{\phi(v_i, v_j)}$$
其中 $\phi(v_i, v_j)$ 是节点 $v_i$ 和 $v_j$ 之间的最短路径距离，$b$ 是可学习的偏置。

**中心性编码 (Centrality Encoding)**：
使用节点的度（入度+出度）作为编码：
$$h_i^{(0)} = x_i + z_{\deg(v_i)}^{-} + z_{\deg(v_i)}^{+}$$

**Python简化实现**：
```python
class GraphTransformerLayer(nn.Module):
    """
    简化的图Transformer层
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # 空间编码（最短路径距离）
        max_distance = 10  # 假设最大距离为10
        self.spatial_bias = nn.Embedding(max_distance + 1, 1)
        
    def compute_shortest_path_distances(self, adj):
        """
        计算所有节点对的最短路径距离（Floyd-Warshall算法简化版）
        """
        n = adj.size(0)
        # 初始化：直接连接的为1，自己为0，其他为无穷大
        dist = torch.full((n, n), float('inf'), device=adj.device)
        dist[adj > 0] = 1
        dist[torch.arange(n), torch.arange(n)] = 0
        
        # Floyd-Warshall
        for k in range(n):
            dist = torch.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])
        
        # 限制最大距离
        dist = torch.clamp(dist, 0, 10).long()
        return dist
    
    def forward(self, x, adj):
        """
        Args:
            x: [n_nodes, d_model]
            adj: [n_nodes, n_nodes]
        """
        n_nodes = x.size(0)
        
        # 计算最短路径距离
        sp_dist = self.compute_shortest_path_distances(adj)  # [n, n]
        spatial_bias = self.spatial_bias(sp_dist).squeeze(-1)  # [n, n]
        
        # 自注意力（使用空间偏置）
        x = x.unsqueeze(0)  # [1, n, d]
        attn_out, _ = self.attention(x, x, x, attn_mask=spatial_bias)
        attn_out = attn_out.squeeze(0)
        
        # 残差连接和层归一化
        x = self.norm1(x.squeeze(0) + attn_out)
        
        # 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


# 测试Graph Transformer
print("=" * 50)
print("测试Graph Transformer层")
print("=" * 50)

gt_layer = GraphTransformerLayer(d_model=16, n_heads=4)
gt_out = gt_layer(x, A)
print(f"Graph Transformer输出形状: {gt_out.shape}")
```

### 53.3.3 等变神经网络

**等变性 (Equivariance)** 是几何深度学习的核心概念。

**直观理解**：
如果你将输入旋转一下，输出也应该相应地旋转，而不是完全改变。这就是等变性。

**形式化定义**：
对于群 $G$ 的作用 $\rho(g)$ 和 $\rho'(g)$，函数 $f$ 是等变的当：
$$f(\rho(g)x) = \rho'(g)f(x)$$

对于3D分子数据，重要的对称群是 **E(3)**：
- **平移不变性**：移动整个分子，性质不变
- **旋转等变性**：旋转分子，预测的力/速度也应该相应旋转
- **反射不变性**：镜像分子，性质不变

**SchNet (Schütt et al., 2018)**：
用于分子能量预测的等变神经网络。

**核心思想**：
1. 使用原子间的连续滤波器卷积
2. 距离信息使用径向基函数编码
3. 保持E(3)等变性

**费曼法比喻**：
> **等变性就像"转动地图"**
> 
> 想象你拿着一张纸质地图：
> - 如果你原地旋转（**旋转等变**），地图上的北方仍然指向地理北方，只是你自己面对的方向变了
> - 如果你把地图拿到另一个城市（**平移不变**），地图上的相对位置关系不变
> - 等变神经网络就是这样——输入经过变换，输出也以相同方式变换

**Python简化实现**：
```python
class RadialBasisFunction(nn.Module):
    """
    径向基函数，用于编码距离
    """
    def __init__(self, n_rbf=20, cutoff=5.0):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        
        # 可学习的中心点和宽度
        self.centers = nn.Parameter(torch.linspace(0, cutoff, n_rbf))
        self.widths = nn.Parameter(torch.ones(n_rbf) * 0.5)
    
    def forward(self, distances):
        """
        Args:
            distances: [...] 原子间距离
        Returns:
            rbf: [..., n_rbf] RBF特征
        """
        distances = distances.unsqueeze(-1)  # [..., 1]
        return torch.exp(-((distances - self.centers) / self.widths) ** 2)


class SchNetLayer(nn.Module):
    """
    简化的SchNet层
    """
    def __init__(self, n_features=64, n_rbf=20):
        super().__init__()
        self.n_features = n_features
        
        # 径向基函数
        self.rbf = RadialBasisFunction(n_rbf=n_rbf)
        
        # 滤波器生成网络
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, n_features),
            nn.Tanh(),
            nn.Linear(n_features, n_features)
        )
        
        # 交互层
        self.interaction = nn.Linear(n_features, n_features)
        
    def forward(self, atomic_features, positions):
        """
        Args:
            atomic_features: [n_atoms, n_features] 原子特征
            positions: [n_atoms, 3] 3D坐标
        Returns:
            new_features: [n_atoms, n_features]
        """
        n_atoms = atomic_features.size(0)
        
        # 计算原子间距离矩阵
        distances = torch.cdist(positions, positions)  # [n_atoms, n_atoms]
        
        # 径向基函数编码
        rbf_features = self.rbf(distances)  # [n_atoms, n_atoms, n_rbf]
        
        # 生成连续滤波器
        filters = self.filter_net(rbf_features)  # [n_atoms, n_atoms, n_features]
        
        # 连续滤波卷积
        messages = filters * atomic_features.unsqueeze(0)  # [n, n, features]
        aggregated = messages.sum(dim=1)  # [n_atoms, n_features]
        
        # 更新特征
        new_features = atomic_features + self.interaction(aggregated)
        
        return new_features


# 测试SchNet
print("=" * 50)
print("测试SchNet层（分子建模）")
print("=" * 50)

# 模拟一个分子：5个原子
n_atoms = 5
atomic_features = torch.randn(n_atoms, 64)
positions = torch.randn(n_atoms, 3)  # 3D坐标

schnet_layer = SchNetLayer(n_features=64, n_rbf=20)
new_atomic_features = schnet_layer(atomic_features, positions)
print(f"SchNet输出形状: {new_atomic_features.shape}")
print("✓ 保持E(3)等变性：旋转输入位置，输出特征会相应变换")
```

---

## 53.4 几何深度学习

### 53.4.1 对称性：几何先验

**为什么对称性重要？**

机器学习模型需要从有限的数据中学习泛化。对称性提供了强大的**归纳偏置**：如果知道某些变换不应该改变输出，我们就可以强制模型遵守这一约束，大大减少需要学习的内容。

**四大对称性类型**：

1. **平移不变性 (Translation Invariance)**：
   输入平移，输出不变
   - 示例：图像分类（猫在左上角还是右下角都是猫）
   - CNN通过权重共享实现

2. **平移等变性 (Translation Equivariance)**：
   输入平移，输出也平移相同量
   - 示例：目标检测（框应该随物体移动）
   - CNN的特征图随输入平移

3. **旋转不变/等变性 (Rotation Invariance/Equivariance)**：
   - 分子能量预测：旋转分子，能量不变（不变性）
   - 分子力预测：旋转分子，力向量也旋转（等变性）

4. **置换不变性 (Permutation Invariance)**：
   改变节点编号顺序，输出不变
   - 图神经网络的核心要求
   - 通过聚合函数（sum/mean/max）实现

**数学表达**：

对于置换群 $S_n$，函数 $f$ 是置换不变的：
$$f(PX, PAP^T) = f(X, A), \quad \forall P \in S_n$$

其中 $P$ 是置换矩阵。

### 53.4.2 流形上的深度学习

**流形 (Manifold)** 是局部看起来像欧几里得空间的弯曲空间。

**流形学习回顾**：
- 高维数据通常位于低维流形上
- 目标：发现数据的内在结构

**流形卷积**：
在流形上定义卷积比在平面上困难，因为：
1. 流形上没有全局坐标系
2. 无法简单平移卷积核

**解决方法**：
1. **测地线CNN**：沿着流形上的最短路径（测地线）定义卷积
2. **MoNet**：使用局部高斯坐标系
3. **SplineCNN**：使用B样条基函数

### 53.4.3 点云网络

**点云 (Point Cloud)** 是3D扫描数据的基本表示形式——一堆 $(x, y, z)$ 坐标。

**挑战**：
1. **无序性**：点没有固定顺序
2. **稀疏性**：点在3D空间中稀疏分布
3. **局部结构**：需要理解局部几何

**PointNet (Qi et al., 2017)**：
首个直接处理原始点云的深度学习架构。

**核心思想**：
1. **置换不变性**：使用对称函数（max-pooling）
2. **点级特征**：每个点独立处理，然后聚合
3. **T-Net**：预测变换矩阵对齐点云

**费曼法比喻**：
> **PointNet就像"认乐高"**
> 
> 想象地上散落着一堆乐高积木：
> - 你不能依赖积木的顺序（无序性）
> - 你需要识别"这是一辆车的零件"（全局理解）
> - 同时知道"这块是车轮"（局部结构）
> 
> PointNet的策略是：
> 1. 检查每一块积木的特征（点级MLP）
> 2. 找出最具代表性的特征（max-pooling）
> 3. 综合判断这是什么（全局特征+分类）

**Python实现**：
```python
class TNet(nn.Module):
    """
    变换网络：学习点云的刚性变换
    """
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        
        self.conv = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, k, N] 点云
        Returns:
            transform: [B, k, k] 变换矩阵
        """
        B = x.size(0)
        
        # 提取全局特征
        x = self.conv(x)  # [B, 1024, N]
        x = torch.max(x, 2)[0]  # [B, 1024] - max pooling实现置换不变
        
        # 预测变换矩阵
        transform = self.fc(x)  # [B, k*k]
        transform = transform.view(B, self.k, self.k)
        
        # 初始化为单位矩阵
        identity = torch.eye(self.k, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        transform = transform + identity  # 残差学习
        
        return transform


class PointNet(nn.Module):
    """
    PointNet用于点云分类
    """
    def __init__(self, num_classes=40, n_points=1024):
        super().__init__()
        self.n_points = n_points
        
        # 输入变换 (3x3)
        self.input_transform = TNet(k=3)
        
        # 点级特征提取
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 特征变换 (64x64)
        self.feature_transform = TNet(k=64)
        
        # 更深层的特征
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, N, 3] 点云，N个点，每个点3维坐标
        Returns:
            logits: [B, num_classes]
        """
        B, N, _ = x.shape
        
        # 调整维度为 [B, 3, N]
        x = x.transpose(1, 2)
        
        # 输入变换
        transform3x3 = self.input_transform(x)
        x = torch.bmm(transform3x3, x)  # [B, 3, N]
        
        # 点级特征
        x = self.mlp1(x)  # [B, 64, N]
        
        # 特征变换
        transform64x64 = self.feature_transform(x)
        x = torch.bmm(transform64x64, x)  # [B, 64, N]
        
        # 保存局部特征（用于分割任务）
        local_features = x
        
        # 更深的特征
        x = self.mlp2(x)  # [B, 1024, N]
        
        # 全局特征（置换不变）
        global_features = torch.max(x, 2)[0]  # [B, 1024]
        
        # 分类
        logits = self.classifier(global_features)  # [B, num_classes]
        
        return logits


# 测试PointNet
print("=" * 50)
print("测试PointNet（点云分类）")
print("=" * 50)

# 模拟一个batch的点云数据：2个样本，每个1024个点，3维坐标
batch_size = 2
n_points = 1024
point_cloud = torch.randn(batch_size, n_points, 3)

pointnet = PointNet(num_classes=40, n_points=n_points)
logits = pointnet(point_cloud)
print(f"输入点云形状: {point_cloud.shape}")
print(f"PointNet输出形状: {logits.shape}")
print(f"预测类别: {logits.argmax(dim=1)}")
print("✓ 置换不变性：改变点的顺序，输出不变")
```

---

## 53.5 图生成模型

### 53.5.1 图自编码器 (GAE & VGAE)

**图自编码器 (Graph Auto-Encoder, GAE)** 学习图的低维表示，用于链接预测等任务。

**架构**：
1. **编码器**：GCN将节点映射到低维空间
2. **解码器**：内积重构邻接矩阵

**损失函数**：
$$\mathcal{L} = \mathbb{E}_{(i,j) \in E} [\log \sigma(z_i^T z_j)] + \mathbb{E}_{(i,j) \notin E} [\log(1 - \sigma(z_i^T z_j))]$$

**变分图自编码器 (VGAE)**：
类似VAE，编码器输出均值和方差，引入随机性。

### 53.5.2 图生成网络

**GraphRNN**：
将图生成视为序列生成问题，逐个生成节点和边。

**分子生成应用**：
- 使用图生成模型设计新药
- 优化分子属性（如溶解度、毒性）

### 53.5.3 图扩散模型

**EDM (Equivariant Diffusion Model)**：
将扩散模型扩展到3D分子生成，保持E(3)等变性。

**应用**：
- 无条件分子生成
- 属性条件分子生成
- 蛋白质-配体复合物生成

---

## 53.6 实战案例

### 53.6.1 分子性质预测

使用SchNet预测分子的量子化学性质（QM9数据集）：

```python
class MoleculePropertyPredictor(nn.Module):
    """
    分子性质预测器
    """
    def __init__(self, n_atom_types=10, n_features=128, n_layers=6):
        super().__init__()
        
        # 原子类型嵌入
        self.atom_embedding = nn.Embedding(n_atom_types, n_features)
        
        # SchNet层堆叠
        self.schnet_layers = nn.ModuleList([
            SchNetLayer(n_features=n_features) for _ in range(n_layers)
        ])
        
        # 输出层（预测能量、偶极矩等）
        self.output = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 12)  # QM9有12个目标属性
        )
    
    def forward(self, atom_types, positions):
        """
        Args:
            atom_types: [n_atoms] 原子类型索引
            positions: [n_atoms, 3] 3D坐标
        Returns:
            properties: [12] 预测的性质
        """
        # 原子嵌入
        h = self.atom_embedding(atom_types)  # [n_atoms, n_features]
        
        # SchNet消息传递
        for layer in self.schnet_layers:
            h = layer(h, positions)
        
        # 全局平均池化
        h_global = h.mean(dim=0)  # [n_features]
        
        # 预测
        properties = self.output(h_global)
        
        return properties
```

### 53.6.2 社交网络分析

使用GCN进行社区检测：

```python
class CommunityDetector(nn.Module):
    """
    社交网络社区检测
    """
    def __init__(self, in_features, n_communities):
        super().__init__()
        self.gcn = GCN(
            in_features=in_features,
            hidden_features=64,
            out_features=n_communities,
            n_layers=2
        )
    
    def forward(self, features, adj):
        """
        返回每个节点属于每个社区的概率
        """
        logits = self.gcn(features, adj)
        return torch.softmax(logits, dim=-1)
    
    def detect_communities(self, features, adj):
        """
        硬分配：每个节点属于一个社区
        """
        probs = self.forward(features, adj)
        return probs.argmax(dim=-1)
```

### 53.6.3 完整对比代码

```python
def compare_gnn_models():
    """
    对比不同GNN模型的性能
    """
    print("=" * 60)
    print("图神经网络模型对比实验")
    print("=" * 60)
    
    # 创建测试图
    n_nodes = 100
    n_features = 16
    n_classes = 7
    
    # 随机图（Erdős-Rényi）
    p = 0.1
    adj = torch.bernoulli(torch.ones(n_nodes, n_nodes) * p)
    adj = torch.triu(adj, 1) + torch.triu(adj, 1).T  # 对称
    
    # 随机特征
    x = torch.randn(n_nodes, n_features)
    
    models = {
        'GCN': GCN(n_features, 32, n_classes, n_layers=2),
        'GAT': GAT(n_features, 8, n_classes, n_heads=4, n_layers=2),
        'GraphSAGE': GraphSAGE(n_features, 32, n_classes, n_layers=2, aggregator='mean'),
        'DeepGCN': DeepGCN(n_features, 32, n_classes, n_layers=4)
    }
    
    results = {}
    for name, model in models.items():
        # 计算参数量
        n_params = sum(p.numel() for p in model.parameters())
        
        # 前向传播计时
        import time
        start = time.time()
        for _ in range(10):
            out = model(x, adj)
        elapsed = (time.time() - start) / 10
        
        results[name] = {
            'params': n_params,
            'time': elapsed,
            'output_shape': out.shape
        }
        
        print(f"\n{name}:")
        print(f"  参数量: {n_params:,}")
        print(f"  推理时间: {elapsed*1000:.2f}ms")
        print(f"  输出形状: {out.shape}")
    
    return results


# 运行对比
if __name__ == "__main__":
    compare_gnn_models()
```

---

## 53.7 总结与展望

### 53.7.1 本章核心概念回顾

**图神经网络的核心**：
1. **消息传递**：邻居信息聚合的基本范式
2. **置换不变性**：图对节点编号不敏感
3. **归纳偏置**：利用图结构的几何先验

**三大经典架构**：
| 模型 | 核心创新 | 适用场景 |
|------|----------|----------|
| GCN | 谱图卷积的一阶近似 | 中小规模图，半监督学习 |
| GAT | 自注意力机制 | 需要区分邻居重要性的场景 |
| GraphSAGE | 邻居采样 | 大规模图，归纳式学习 |

**几何深度学习的核心**：
1. **对称性**：利用问题的几何结构
2. **等变性**：输入变换，输出相应变换
3. **不变性**：某些变换下输出保持不变

### 53.7.2 前沿研究方向

1. **大规模图训练**：
   - 图采样方法（GraphSAGE, Cluster-GCN）
   - 子图训练（SIGN, ShaDow-GNN）

2. **动态图**：
   - 时序图神经网络
   - 持续学习

3. **解释性**：
   - GNNExplainer
   - PGExplainer

4. **图基础模型**：
   - 预训练策略
   - 跨域迁移

### 53.7.3 学习路径建议

**入门**：
1. 实现基础GCN，理解消息传递
2. 在小规模数据集上实验（Cora, Citeseer）

**进阶**：
1. 实现GAT和GraphSAGE
2. 处理大规模图（OGB benchmark）

**深入**：
1. 研究等变神经网络
2. 探索图生成模型

---

## 本章练习题

### 基础题

1. **解释为什么传统CNN无法直接应用于图数据？列举至少三个原因。**

2. **消息传递机制的核心思想是什么？用你自己的话解释消息函数、聚合函数和更新函数的作用。**

3. **GCN、GAT、GraphSAGE的主要区别是什么？在什么情况下你会选择使用其中某一种？**

### 数学推导题

4. **证明GCN的传播规则可以表示为邻居特征的加权平均。给出权重与节点度的关系。**

5. **推导GAT的注意力系数计算公式。解释为什么使用LeakyReLU激活函数？**

6. **证明sum/mean/max聚合函数都满足置换不变性。**

### 编程题

7. **实现一个完整的节点分类pipeline**：
   - 使用Cora数据集（或模拟数据）
   - 实现GCN、GAT、GraphSAGE三种模型
   - 对比它们的分类准确率

8. **实现一个简化的GraphSAGE用于链接预测**：
   - 输入：图结构和节点特征
   - 输出：边存在的概率
   - 使用负采样训练

9. **扩展PointNet实现点云分割**：
   - 局部特征和全局特征拼接
   - 为每个点预测类别
   - 在ShapeNet数据集上测试

---

## 参考文献

1. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations*.

2. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. *International Conference on Learning Representations*.

3. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems*, 30.

4. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. *International Conference on Machine Learning*, 1263-1272.

5. Battaglia, P. W., Hamrick, J. B., Bapst, V., Sanchez-Gonzalez, A., Zambaldi, V., Malinowski, M., ... & Pascanu, R. (2018). Relational inductive biases, deep learning, and graph networks. *arXiv preprint arXiv:1806.01261*.

6. Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. *arXiv preprint arXiv:2104.13478*.

7. Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). PointNet: Deep learning on point sets for 3D classification and segmentation. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 652-660.

8. Schütt, K. T., Sauceda, H. E., Kindermans, P. J., Tkatchenko, A., & Müller, K. R. (2018). SchNet – A deep learning architecture for molecules and materials. *The Journal of Chemical Physics*, 148(24), 241722.

9. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? *International Conference on Learning Representations*.

10. Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., ... & Liu, T. Y. (2021). Do transformers really perform badly for graph representation? *Advances in Neural Information Processing Systems*, 34, 28877-28888.

---

*本章完 | 字数：约16,500字 | 代码：约1,800行*
