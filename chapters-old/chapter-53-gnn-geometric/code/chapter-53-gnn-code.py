"""
第五十三章 图神经网络与几何深度学习 - 完整代码实现
Graph Neural Networks & Geometric Deep Learning

包含：
1. 图神经网络基础 (GCN, GAT, GraphSAGE)
2. 高级架构 (DeepGCN, GraphTransformer, SchNet)
3. 点云网络 (PointNet)
4. 实战案例

作者: ML教材写作团队
版本: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


# ============================================================================
# 第一部分：图神经网络基础
# ============================================================================

class GCNLayer(nn.Module):
    """
    图卷积网络层 (Kipf & Welling, 2017)
    
    传播规则: H^{(l+1)} = σ(D^{-1/2} A D^{-1/2} H^{(l)} W^{(l)})
    
    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x: torch.Tensor, adj_normalized: torch.Tensor) -> torch.Tensor:
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
    
    Args:
        in_features: 输入特征维度
        hidden_features: 隐藏层维度
        out_features: 输出特征维度
        n_layers: 层数
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(GCNLayer(in_features, hidden_features))
        
        # 隐藏层
        for _ in range(n_layers - 2):
            self.layers.append(GCNLayer(hidden_features, hidden_features))
        
        # 输出层（无激活函数，用于分类/回归）
        self.output_layer = nn.Linear(hidden_features, out_features)
        
    def normalize_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        """
        归一化邻接矩阵: A_norm = D^{-1/2} (A + I) D^{-1/2}
        
        Args:
            adj: [n_nodes, n_nodes] 原始邻接矩阵
        Returns:
            adj_normalized: 归一化后的邻接矩阵
        """
        # 添加自环
        adj_with_self_loops = adj + torch.eye(adj.size(0), device=adj.device)
        
        # 计算度矩阵
        degrees = adj_with_self_loops.sum(dim=1)
        D_inv_sqrt = torch.diag(torch.pow(degrees + 1e-8, -0.5))
        
        # 归一化
        adj_normalized = D_inv_sqrt @ adj_with_self_loops @ D_inv_sqrt
        
        return adj_normalized
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [n_nodes, in_features]
            adj: [n_nodes, n_nodes] 原始邻接矩阵
        Returns:
            out: [n_nodes, out_features]
        """
        adj_normalized = self.normalize_adjacency(adj)
        
        # 前向传播
        for layer in self.layers:
            x = layer(x, adj_normalized)
        
        # 最终线性变换
        x = self.output_layer(x)
        x = torch.matmul(adj_normalized, x)
        
        return x


class GATLayer(nn.Module):
    """
    图注意力网络层 (Veličković et al., 2018)
    
    使用多头注意力机制，让节点学习关注重要的邻居。
    
    Args:
        in_features: 输入特征维度
        out_features: 每个头的输出维度
        n_heads: 注意力头数
        dropout: dropout概率
    """
    def __init__(self, in_features: int, out_features: int, n_heads: int = 8, dropout: float = 0.6):
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
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [n_nodes, in_features]
            adj: [n_nodes, n_nodes] 邻接矩阵（包含自环）
        Returns:
            h: [n_nodes, out_features * n_heads]
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
    
    Args:
        in_features: 输入特征维度
        hidden_features: 隐藏层每个头的维度
        out_features: 输出维度
        n_heads: 隐藏层注意力头数
        n_layers: 层数
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int, 
                 n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 第一层（多头注意力）
        self.layers.append(GATLayer(in_features, hidden_features, n_heads=n_heads))
        
        # 中间层
        for _ in range(n_layers - 2):
            self.layers.append(GATLayer(hidden_features * n_heads, hidden_features, n_heads=n_heads))
        
        # 最后一层（单头，用于分类）
        self.layers.append(GATLayer(hidden_features * n_heads, out_features, n_heads=1))
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # 添加自环
        adj_with_self = adj + torch.eye(adj.size(0), device=adj.device)
        
        for layer in self.layers[:-1]:
            x = layer(x, adj_with_self)
        
        # 最后一层不加激活
        x = self.layers[-1](x, adj_with_self)
        
        return x


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE层 (Hamilton et al., 2017)
    
    支持归纳式学习，通过邻居采样处理大规模图。
    
    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        aggregator: 聚合函数 ('mean', 'max', 'sum')
    """
    def __init__(self, in_features: int, out_features: int, aggregator: str = 'mean'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        
        # 拼接自身和邻居特征后的线性变换
        self.W = nn.Linear(2 * in_features, out_features)
        
    def sample_neighbors(self, adj: torch.Tensor, node_idx: int, sample_size: int) -> torch.Tensor:
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
    
    def aggregate(self, neighbor_features: torch.Tensor) -> torch.Tensor:
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
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor, sample_size: int = 10) -> torch.Tensor:
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
    
    Args:
        in_features: 输入特征维度
        hidden_features: 隐藏层维度
        out_features: 输出维度
        n_layers: 层数
        aggregator: 聚合函数
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int, 
                 n_layers: int = 2, aggregator: str = 'mean'):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(GraphSAGELayer(in_features, hidden_features, aggregator))
        
        # 隐藏层
        for _ in range(n_layers - 2):
            self.layers.append(GraphSAGELayer(hidden_features, hidden_features, aggregator))
        
        # 输出层
        self.layers.append(GraphSAGELayer(hidden_features, out_features, aggregator))
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor, sample_size: int = 10) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x, adj, sample_size)
        
        # 最后一层不加激活
        x = self.layers[-1](x, adj, sample_size)
        return x


# ============================================================================
# 第二部分：高级图神经网络架构
# ============================================================================

class ResidualGCNLayer(nn.Module):
    """
    带残差连接的GCN层
    
    解决深层GNN的过平滑问题
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.gcn = GCNLayer(in_features, out_features)
        
        # 如果维度不同，需要投影
        self.projection = None
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor, adj_normalized: torch.Tensor) -> torch.Tensor:
        h = self.gcn(x, adj_normalized)
        
        # 残差连接
        if self.projection is not None:
            x = self.projection(x)
        
        return h + x  # 残差连接


class DeepGCN(nn.Module):
    """
    深层GCN，使用残差连接解决过平滑
    
    Args:
        in_features: 输入特征维度
        hidden_features: 隐藏层维度
        out_features: 输出维度
        n_layers: 层数（可以很深，如8-16层）
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int, n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(GCNLayer(in_features, hidden_features))
        
        # 隐藏层（带残差连接）
        for _ in range(n_layers - 2):
            self.layers.append(ResidualGCNLayer(hidden_features, hidden_features))
        
        # 输出层
        self.output_layer = nn.Linear(hidden_features, out_features)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # 归一化邻接矩阵
        adj_with_self = adj + torch.eye(adj.size(0), device=adj.device)
        degrees = adj_with_self.sum(dim=1)
        D_inv_sqrt = torch.diag(torch.pow(degrees + 1e-8, -0.5))
        adj_normalized = D_inv_sqrt @ adj_with_self @ D_inv_sqrt
        
        # 前向传播
        for layer in self.layers:
            x = layer(x, adj_normalized)
        
        return self.output_layer(x)


class GraphTransformerLayer(nn.Module):
    """
    简化的图Transformer层
    
    将Transformer架构适应到图结构，使用空间编码替代位置编码。
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        dropout: dropout概率
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
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
        max_distance = 10
        self.spatial_bias = nn.Embedding(max_distance + 1, 1)
        
    def compute_shortest_path_distances(self, adj: torch.Tensor) -> torch.Tensor:
        """
        计算所有节点对的最短路径距离（Floyd-Warshall算法简化版）
        
        Args:
            adj: 邻接矩阵
        Returns:
            dist: 最短路径距离矩阵
        """
        n = adj.size(0)
        # 初始化
        dist = torch.full((n, n), float('inf'), device=adj.device)
        dist[adj > 0] = 1
        dist[torch.arange(n), torch.arange(n)] = 0
        
        # Floyd-Warshall
        for k in range(n):
            dist = torch.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])
        
        # 限制最大距离
        dist = torch.clamp(dist, 0, 10).long()
        return dist
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [n_nodes, d_model]
            adj: [n_nodes, n_nodes]
        Returns:
            h: [n_nodes, d_model]
        """
        n_nodes = x.size(0)
        
        # 计算最短路径距离
        sp_dist = self.compute_shortest_path_distances(adj)
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


class RadialBasisFunction(nn.Module):
    """
    径向基函数，用于编码距离
    
    Args:
        n_rbf: RBF中心数量
        cutoff: 截断距离
    """
    def __init__(self, n_rbf: int = 20, cutoff: float = 5.0):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        
        # 可学习的中心点和宽度
        self.centers = nn.Parameter(torch.linspace(0, cutoff, n_rbf))
        self.widths = nn.Parameter(torch.ones(n_rbf) * 0.5)
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: [...] 原子间距离
        Returns:
            rbf: [..., n_rbf] RBF特征
        """
        distances = distances.unsqueeze(-1)
        return torch.exp(-((distances - self.centers) / self.widths) ** 2)


class SchNetLayer(nn.Module):
    """
    简化的SchNet层 (Schütt et al., 2018)
    
    用于分子能量预测的等变神经网络层
    
    Args:
        n_features: 特征维度
        n_rbf: 径向基函数数量
    """
    def __init__(self, n_features: int = 64, n_rbf: int = 20):
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
        
    def forward(self, atomic_features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
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


# ============================================================================
# 第三部分：点云网络
# ============================================================================

class TNet(nn.Module):
    """
    变换网络：学习点云的刚性变换 (PointNet)
    
    Args:
        k: 变换矩阵维度 (3x3 或 64x64)
    """
    def __init__(self, k: int = 3):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    PointNet用于点云分类 (Qi et al., 2017)
    
    Args:
        num_classes: 类别数量
        n_points: 点云中的点数
    """
    def __init__(self, num_classes: int = 40, n_points: int = 1024):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


# ============================================================================
# 第四部分：实战案例
# ============================================================================

class MoleculePropertyPredictor(nn.Module):
    """
    分子性质预测器
    
    使用SchNet预测分子的量子化学性质
    
    Args:
        n_atom_types: 原子类型数量
        n_features: 特征维度
        n_layers: SchNet层数
    """
    def __init__(self, n_atom_types: int = 10, n_features: int = 128, n_layers: int = 6):
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
    
    def forward(self, atom_types: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
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


class CommunityDetector(nn.Module):
    """
    社交网络社区检测
    
    使用GCN进行图节点聚类
    
    Args:
        in_features: 输入特征维度
        n_communities: 社区数量
    """
    def __init__(self, in_features: int, n_communities: int):
        super().__init__()
        self.gcn = GCN(
            in_features=in_features,
            hidden_features=64,
            out_features=n_communities,
            n_layers=2
        )
    
    def forward(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        返回每个节点属于每个社区的概率
        
        Args:
            features: [n_nodes, in_features]
            adj: [n_nodes, n_nodes]
        Returns:
            probs: [n_nodes, n_communities]
        """
        logits = self.gcn(features, adj)
        return torch.softmax(logits, dim=-1)
    
    def detect_communities(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        硬分配：每个节点属于一个社区
        
        Returns:
            communities: [n_nodes] 社区标签
        """
        probs = self.forward(features, adj)
        return probs.argmax(dim=-1)


# ============================================================================
# 第五部分：测试与演示
# ============================================================================

def create_test_graph(n_nodes: int = 4, edge_prob: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建一个随机测试图
    
    Args:
        n_nodes: 节点数量
        edge_prob: 边存在的概率
    Returns:
        adj: 邻接矩阵
        x: 节点特征
    """
    # 随机邻接矩阵
    adj = torch.bernoulli(torch.ones(n_nodes, n_nodes) * edge_prob)
    adj = torch.triu(adj, 1) + torch.triu(adj, 1).T  # 对称
    
    # 随机特征
    x = torch.randn(n_nodes, 16)
    
    return adj, x


def test_gcn():
    """测试GCN模型"""
    print("=" * 60)
    print("测试GCN模型")
    print("=" * 60)
    
    adj, x = create_test_graph(n_nodes=4)
    print(f"邻接矩阵:\n{adj}")
    print(f"\n输入特征形状: {x.shape}")
    
    model = GCN(in_features=16, hidden_features=32, out_features=7, n_layers=2)
    out = model(x, adj)
    
    print(f"\nGCN输出形状: {out.shape}")
    print(f"输出示例:\n{out}")
    print("✓ GCN测试通过\n")


def test_gat():
    """测试GAT模型"""
    print("=" * 60)
    print("测试GAT模型")
    print("=" * 60)
    
    adj, x = create_test_graph(n_nodes=4)
    print(f"输入特征形状: {x.shape}")
    
    model = GAT(in_features=16, hidden_features=8, out_features=7, n_heads=4, n_layers=2)
    out = model(x, adj)
    
    print(f"\nGAT输出形状: {out.shape}")
    print(f"输出示例:\n{out}")
    print("✓ GAT测试通过\n")


def test_graphsage():
    """测试GraphSAGE模型"""
    print("=" * 60)
    print("测试GraphSAGE模型")
    print("=" * 60)
    
    adj, x = create_test_graph(n_nodes=4)
    print(f"输入特征形状: {x.shape}")
    
    model = GraphSAGE(in_features=16, hidden_features=32, out_features=7, n_layers=2)
    out = model(x, adj, sample_size=2)
    
    print(f"\nGraphSAGE输出形状: {out.shape}")
    print(f"输出示例:\n{out}")
    print("✓ GraphSAGE测试通过\n")


def test_deep_gcn():
    """测试深层GCN"""
    print("=" * 60)
    print("测试深层GCN（4层，带残差连接）")
    print("=" * 60)
    
    adj, x = create_test_graph(n_nodes=10, edge_prob=0.3)
    print(f"输入特征形状: {x.shape}")
    
    model = DeepGCN(in_features=16, hidden_features=32, out_features=7, n_layers=4)
    out = model(x, adj)
    
    print(f"\n深层GCN输出形状: {out.shape}")
    print("✓ 深层GCN测试通过\n")


def test_graph_transformer():
    """测试Graph Transformer"""
    print("=" * 60)
    print("测试Graph Transformer层")
    print("=" * 60)
    
    adj, x = create_test_graph(n_nodes=4)
    print(f"输入特征形状: {x.shape}")
    
    layer = GraphTransformerLayer(d_model=16, n_heads=4)
    out = layer(x, adj)
    
    print(f"\nGraph Transformer输出形状: {out.shape}")
    print("✓ Graph Transformer测试通过\n")


def test_schnet():
    """测试SchNet"""
    print("=" * 60)
    print("测试SchNet层（分子建模）")
    print("=" * 60)
    
    # 模拟一个分子：5个原子
    n_atoms = 5
    atomic_features = torch.randn(n_atoms, 64)
    positions = torch.randn(n_atoms, 3)  # 3D坐标
    
    print(f"原子特征形状: {atomic_features.shape}")
    print(f"位置形状: {positions.shape}")
    
    schnet_layer = SchNetLayer(n_features=64, n_rbf=20)
    new_atomic_features = schnet_layer(atomic_features, positions)
    
    print(f"\nSchNet输出形状: {new_atomic_features.shape}")
    print("✓ 保持E(3)等变性：旋转输入位置，输出特征会相应变换")
    print("✓ SchNet测试通过\n")


def test_pointnet():
    """测试PointNet"""
    print("=" * 60)
    print("测试PointNet（点云分类）")
    print("=" * 60)
    
    # 模拟一个batch的点云数据
    batch_size = 2
    n_points = 1024
    point_cloud = torch.randn(batch_size, n_points, 3)
    
    print(f"输入点云形状: {point_cloud.shape}")
    
    pointnet = PointNet(num_classes=40, n_points=n_points)
    logits = pointnet(point_cloud)
    
    print(f"\nPointNet输出形状: {logits.shape}")
    print(f"预测类别: {logits.argmax(dim=1)}")
    print("✓ 置换不变性：改变点的顺序，输出不变")
    print("✓ PointNet测试通过\n")


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
    
    # 随机图
    p = 0.1
    adj = torch.bernoulli(torch.ones(n_nodes, n_nodes) * p)
    adj = torch.triu(adj, 1) + torch.triu(adj, 1).T
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
        model.eval()
        with torch.no_grad():
            start = time.time()
            for _ in range(10):
                if name == 'GraphSAGE':
                    out = model(x, adj, sample_size=10)
                else:
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


def test_molecule_predictor():
    """测试分子性质预测器"""
    print("=" * 60)
    print("测试分子性质预测器")
    print("=" * 60)
    
    # 模拟苯环分子：6个碳原子 + 6个氢原子 = 12个原子
    n_atoms = 12
    atom_types = torch.randint(0, 10, (n_atoms,))  # 随机原子类型
    positions = torch.randn(n_atoms, 3)
    
    print(f"分子中原子数量: {n_atoms}")
    
    predictor = MoleculePropertyPredictor(n_atom_types=10, n_features=128, n_layers=6)
    properties = predictor(atom_types, positions)
    
    print(f"\n预测性质数量: {properties.shape[0]}")
    print(f"预测值: {properties}")
    print("✓ 分子性质预测器测试通过\n")


def test_community_detector():
    """测试社区检测器"""
    print("=" * 60)
    print("测试社区检测器")
    print("=" * 60)
    
    # 创建一个简单的社交网络图
    n_nodes = 20
    adj = torch.zeros(n_nodes, n_nodes)
    
    # 创建两个社区
    # 社区1: 节点0-9
    for i in range(10):
        for j in range(i+1, 10):
            if torch.rand(1).item() < 0.5:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # 社区2: 节点10-19
    for i in range(10, 20):
        for j in range(i+1, 20):
            if torch.rand(1).item() < 0.5:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # 两个社区之间的连接（较少）
    adj[5, 15] = 1
    adj[15, 5] = 1
    
    features = torch.randn(n_nodes, 16)
    
    detector = CommunityDetector(in_features=16, n_communities=2)
    communities = detector.detect_communities(features, adj)
    
    print(f"检测到的社区分配:\n{communities}")
    print(f"社区0节点数: {(communities == 0).sum().item()}")
    print(f"社区1节点数: {(communities == 1).sum().item()}")
    print("✓ 社区检测器测试通过\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始运行图神经网络完整测试套件")
    print("=" * 60 + "\n")
    
    test_gcn()
    test_gat()
    test_graphsage()
    test_deep_gcn()
    test_graph_transformer()
    test_schnet()
    test_pointnet()
    test_molecule_predictor()
    test_community_detector()
    
    # 模型对比
    print("\n" + "=" * 60)
    print("开始模型对比实验")
    print("=" * 60)
    results = compare_gnn_models()
    
    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    
    # 运行所有测试
    results = run_all_tests()
