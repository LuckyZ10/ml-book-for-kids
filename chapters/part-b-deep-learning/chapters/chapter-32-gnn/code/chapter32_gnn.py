"""
第三十二章代码：图神经网络基础 (Graph Neural Networks)
包含：GCN、GAT、GraphSAGE完整实现及应用示例

作者: 机器学习从小学生到大师
目标: ~800行代码，完整实现主流GNN架构
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# 设置随机种子保证可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# =============================================================================
# 第一部分：基础图操作和工具函数
# =============================================================================

class GraphUtils:
    """图操作的实用工具类"""
    
    @staticmethod
    def normalize_adjacency(adj: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
        """
        计算归一化邻接矩阵: D^{-1/2} (A + I) D^{-1/2}
        
        参数:
            adj: 邻接矩阵 (n, n)
            add_self_loops: 是否添加自环
            
        返回:
            归一化邻接矩阵 (n, n)
        """
        if add_self_loops:
            adj = adj + np.eye(adj.shape[0])
        
        # 计算度矩阵
        degree = np.array(adj.sum(axis=1)).flatten()
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
        
        # D^{-1/2}
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        
        # D^{-1/2} A D^{-1/2}
        normalized_adj = D_inv_sqrt @ adj @ D_inv_sqrt
        
        return normalized_adj
    
    @staticmethod
    def sparse_matrix_to_torch(adj: np.ndarray) -> torch.Tensor:
        """将稠密邻接矩阵转换为PyTorch张量"""
        return torch.FloatTensor(adj)
    
    @staticmethod
    def edge_index_to_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        将edge_index格式转换为邻接矩阵
        
        参数:
            edge_index: 边索引，形状 (2, num_edges)
            num_nodes: 节点数量
            
        返回:
            邻接矩阵 (num_nodes, num_nodes)
        """
        adj = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0], edge_index[1]] = 1
        return adj
    
    @staticmethod
    def adjacency_to_edge_index(adj: np.ndarray) -> torch.Tensor:
        """
        将邻接矩阵转换为edge_index格式
        
        参数:
            adj: 邻接矩阵 (n, n)
            
        返回:
            edge_index: 形状 (2, num_edges)
        """
        rows, cols = np.where(adj > 0)
        edge_index = torch.LongTensor([rows, cols])
        return edge_index
    
    @staticmethod
    def compute_laplacian(adj: np.ndarray) -> np.ndarray:
        """
        计算拉普拉斯矩阵 L = D - A
        
        参数:
            adj: 邻接矩阵
            
        返回:
            拉普拉斯矩阵
        """
        degree = np.diag(adj.sum(axis=1))
        return degree - adj
    
    @staticmethod
    def sample_neighbors(adj: np.ndarray, node: int, num_samples: int) -> List[int]:
        """
        从邻居中采样
        
        参数:
            adj: 邻接矩阵
            node: 目标节点
            num_samples: 采样数量
            
        返回:
            采样的邻居列表
        """
        neighbors = np.where(adj[node] > 0)[0].tolist()
        
        if len(neighbors) == 0:
            # 如果没有邻居，采样自身
            return [node] * num_samples
        
        if len(neighbors) >= num_samples:
            return random.sample(neighbors, num_samples)
        else:
            # 邻居不足，有放回采样
            return random.choices(neighbors, k=num_samples)


# =============================================================================
# 第二部分：图卷积网络 (GCN) 实现
# =============================================================================

class GCNLayer(nn.Module):
    """
    图卷积层 (Graph Convolutional Layer)
    
    公式: H^{(l+1)} = σ(D^{-1/2} Ã D^{-1/2} H^{(l)} W^{(l)})
    
    其中:
        Ã = A + I (添加自环的邻接矩阵)
        D 是 Ã 的度矩阵
        σ 是激活函数
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        参数:
            in_features: 输入特征维度
            out_features: 输出特征维度
            bias: 是否使用偏置
        """
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 可学习的权重矩阵
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入特征 (num_nodes, in_features)
            adj: 归一化邻接矩阵 (num_nodes, num_nodes)
            
        返回:
            输出特征 (num_nodes, out_features)
        """
        # 线性变换: XW
        support = torch.mm(x, self.weight)
        
        # 图卷积: AXW (这里adj已经包含了归一化)
        output = torch.mm(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GCN(nn.Module):
    """
    图卷积网络 (Graph Convolutional Network)
    
    参考: Kipf & Welling (2017) "Semi-Supervised Classification with GCN"
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 activation: str = 'relu'):
        """
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（类别数）
            num_layers: GCN层数
            dropout: Dropout概率
            activation: 激活函数类型
        """
        super(GCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 创建GCN层
        self.layers = nn.ModuleList()
        
        # 第一层
        self.layers.append(GCNLayer(input_dim, hidden_dim))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        # 输出层
        self.layers.append(GCNLayer(hidden_dim, output_dim))
        
        # 激活函数
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = lambda x: x
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入特征 (num_nodes, input_dim)
            adj: 归一化邻接矩阵 (num_nodes, num_nodes)
            
        返回:
            输出 logits (num_nodes, output_dim)
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最后一层（通常不加激活和dropout）
        x = self.layers[-1](x, adj)
        
        return x
    
    def get_embeddings(self, x: torch.Tensor, adj: torch.Tensor, layer: int = -2) -> torch.Tensor:
        """
        获取中间层嵌入（用于可视化或下游任务）
        
        参数:
            x: 输入特征
            adj: 归一化邻接矩阵
            layer: 要获取的层索引，-2表示倒数第二层
            
        返回:
            节点嵌入
        """
        with torch.no_grad():
            for i, gcn_layer in enumerate(self.layers[:layer+1]):
                x = gcn_layer(x, adj)
                if i < len(self.layers) - 1:  # 非最后一层
                    x = self.activation(x)
        return x


# =============================================================================
# 第三部分：图注意力网络 (GAT) 实现
# =============================================================================

class GATLayer(nn.Module):
    """
    图注意力层 (Graph Attention Layer)
    
    参考: Veličković et al. (2018) "Graph Attention Networks"
    
    公式:
        e_{ij} = LeakyReLU(a^T [Wh_i || Wh_j])
        α_{ij} = softmax_j(e_{ij}) = exp(e_{ij}) / Σ_k exp(e_{ik})
        h_i' = σ(Σ_j α_{ij} Wh_j)
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 num_heads: int = 1,
                 dropout: float = 0.6,
                 alpha: float = 0.2,
                 concat: bool = True,
                 bias: bool = True):
        """
        参数:
            in_features: 输入特征维度
            out_features: 每个头的输出维度
            num_heads: 注意力头数
            dropout: Dropout概率
            alpha: LeakyReLU的负斜率
            concat: 是否拼接多头输出（False则取平均）
            bias: 是否使用偏置
        """
        super(GATLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # 每个头有自己的权重矩阵 W
        self.W = nn.Parameter(torch.FloatTensor(num_heads, in_features, out_features))
        
        # 注意力向量 a
        self.a = nn.Parameter(torch.FloatTensor(num_heads, 2 * out_features, 1))
        
        if bias:
            if concat:
                self.bias = nn.Parameter(torch.FloatTensor(num_heads * out_features))
            else:
                self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, 
                x: torch.Tensor, 
                adj: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入特征 (num_nodes, in_features)
            adj: 邻接矩阵 (num_nodes, num_nodes)
            return_attention: 是否返回注意力权重
            
        返回:
            输出特征 (num_nodes, num_heads * out_features) 或 (num_nodes, out_features)
            如果return_attention=True，还返回注意力权重
        """
        num_nodes = x.size(0)
        
        # 线性变换: h = Wx
        # 对每个头分别计算: (num_heads, num_nodes, out_features)
        h = torch.einsum('hij,nj->hni', self.W, x)
        
        # 计算注意力分数
        # 对每个头，计算所有节点对之间的注意力
        attn_input = torch.cat([
            h.unsqueeze(2).expand(-1, -1, num_nodes, -1),  # (h, n, 1, f) -> (h, n, n, f)
            h.unsqueeze(1).expand(-1, num_nodes, -1, -1)   # (h, 1, n, f) -> (h, n, n, f)
        ], dim=-1)  # (num_heads, num_nodes, num_nodes, 2*out_features)
        
        # e = LeakyReLU(a^T [Wh_i || Wh_j])
        e = self.leakyrelu(torch.einsum('hnkf,hfk->hnn', attn_input, self.a).squeeze(-1))
        # e的形状: (num_heads, num_nodes, num_nodes)
        
        # 掩码：只保留有边连接的节点对
        # 将adj扩展为(num_heads, num_nodes, num_nodes)
        mask = adj.unsqueeze(0).expand(self.num_heads, -1, -1)
        e = e.masked_fill(mask == 0, float('-inf'))
        
        # 添加自环
        eye = torch.eye(num_nodes, device=x.device).unsqueeze(0).expand(self.num_heads, -1, -1)
        e = e.masked_fill(eye == 1, 0)
        
        # Softmax归一化: α = softmax(e)
        alpha = F.softmax(e, dim=-1)  # (num_heads, num_nodes, num_nodes)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # 聚合: h' = Σ_j α_{ij} Wh_j
        h_prime = torch.einsum('hmn,hmf->hnf', alpha, h)  # (num_heads, num_nodes, out_features)
        
        if self.concat:
            # 拼接多头输出: (num_nodes, num_heads * out_features)
            out = h_prime.permute(1, 0, 2).contiguous().view(num_nodes, -1)
        else:
            # 取平均: (num_nodes, out_features)
            out = h_prime.mean(dim=0)
        
        if self.bias is not None:
            out = out + self.bias
        
        if return_attention:
            return out, alpha
        return out


class GAT(nn.Module):
    """
    图注意力网络 (Graph Attention Network)
    
    支持多层和多头注意力
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.6,
                 alpha: float = 0.2):
        """
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: 层数
            num_heads: 注意力头数（隐藏层）
            dropout: Dropout概率
            alpha: LeakyReLU斜率
        """
        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.layers = nn.ModuleList()
        
        # 第一层: 使用多头注意力，输出拼接
        self.layers.append(GATLayer(input_dim, hidden_dim, num_heads=num_heads,
                                    dropout=dropout, alpha=alpha, concat=True))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(hidden_dim * num_heads, hidden_dim, 
                                       num_heads=num_heads, dropout=dropout, 
                                       alpha=alpha, concat=True))
        
        # 输出层: 单头，不拼接
        self.layers.append(GATLayer(hidden_dim * num_heads, output_dim, 
                                   num_heads=1, dropout=dropout, 
                                   alpha=alpha, concat=False))
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor, 
                return_attention: bool = False) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入特征
            adj: 邻接矩阵
            return_attention: 是否返回注意力权重
            
        返回:
            输出 logits
        """
        attention_weights = []
        
        for i, layer in enumerate(self.layers[:-1]):
            if return_attention:
                x, attn = layer(x, adj, return_attention=True)
                attention_weights.append(attn)
            else:
                x = layer(x, adj)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 输出层
        if return_attention:
            x, attn = self.layers[-1](x, adj, return_attention=True)
            attention_weights.append(attn)
            return x, attention_weights
        
        x = self.layers[-1](x, adj)
        return x


# =============================================================================
# 第四部分：GraphSAGE 实现
# =============================================================================

class SAGELayer(nn.Module):
    """
    GraphSAGE层
    
    参考: Hamilton et al. (2017) "Inductive Representation Learning on Large Graphs"
    
    支持多种聚合函数:
        - 'mean': 均值聚合
        - 'max': 最大池化
        - 'lstm': LSTM聚合
    """
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 aggregator: str = 'mean',
                 bias: bool = True):
        """
        参数:
            in_features: 输入特征维度
            out_features: 输出特征维度
            aggregator: 聚合函数类型 ('mean', 'max', 'lstm')
            bias: 是否使用偏置
        """
        super(SAGELayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        
        # 线性变换参数
        self.weight = nn.Parameter(torch.FloatTensor(2 * in_features, out_features))
        
        if aggregator == 'max':
            # 最大池化需要先投影
            self.pool_proj = nn.Linear(in_features, in_features)
        elif aggregator == 'lstm':
            # LSTM聚合
            self.lstm = nn.LSTM(in_features, in_features, batch_first=True)
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def aggregate(self, neighbor_features: torch.Tensor) -> torch.Tensor:
        """
        聚合邻居特征
        
        参数:
            neighbor_features: (num_nodes, num_samples, in_features)
            
        返回:
            聚合后的特征: (num_nodes, in_features)
        """
        if self.aggregator == 'mean':
            return neighbor_features.mean(dim=1)
        
        elif self.aggregator == 'max':
            # 先投影，再取最大
            projected = self.pool_proj(neighbor_features)
            return projected.max(dim=1)[0]
        
        elif self.aggregator == 'lstm':
            # 随机打乱顺序以保证排列不变性
            num_nodes, num_samples, in_features = neighbor_features.shape
            # 扩展维度以匹配LSTM输入
            lstm_out, (hidden, cell) = self.lstm(neighbor_features)
            return hidden.squeeze(0)
        
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
    
    def forward(self, 
                x: torch.Tensor,
                neighbor_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 节点自身特征 (num_nodes, in_features)
            neighbor_features: 邻居特征 (num_nodes, num_samples, in_features)
            
        返回:
            输出特征 (num_nodes, out_features)
        """
        # 聚合邻居信息
        h_nei = self.aggregate(neighbor_features)
        
        # 拼接自身特征和邻居聚合特征
        h_concat = torch.cat([x, h_nei], dim=1)
        
        # 线性变换
        output = torch.mm(h_concat, self.weight)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GraphSAGE(nn.Module):
    """
    GraphSAGE完整模型
    
    支持邻居采样和多种聚合函数
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 num_samples: int = 10,
                 aggregator: str = 'mean',
                 dropout: float = 0.5):
        """
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: 层数
            num_samples: 每层采样的邻居数
            aggregator: 聚合函数类型
            dropout: Dropout概率
        """
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.num_samples = num_samples
        self.dropout = dropout
        
        self.layers = nn.ModuleList()
        
        # 第一层
        self.layers.append(SAGELayer(input_dim, hidden_dim, aggregator))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(SAGELayer(hidden_dim, hidden_dim, aggregator))
        
        # 输出层
        self.layers.append(SAGELayer(hidden_dim, output_dim, aggregator))
    
    def forward(self, 
                x: torch.Tensor,
                adj: torch.Tensor,
                sampled_neighbors: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入特征 (num_nodes, input_dim)
            adj: 邻接矩阵 (num_nodes, num_nodes)
            sampled_neighbors: 预采样的邻居索引列表
            
        返回:
            输出特征
        """
        num_nodes = x.size(0)
        
        # 如果没有预采样，进行现场采样
        if sampled_neighbors is None:
            sampled_neighbors = self.sample_neighbors_batch(adj, self.num_samples)
        
        h = x
        
        for i, layer in enumerate(self.layers):
            # 获取当前层的邻居特征
            neighbor_features = self.get_neighbor_features(h, sampled_neighbors[i])
            
            # SAGE层前向
            h = layer(h, neighbor_features)
            
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
    def sample_neighbors_batch(self, adj: torch.Tensor, num_samples: int) -> List[List[torch.Tensor]]:
        """
        为所有层采样邻居
        
        返回:
            每层的邻居采样结果列表
        """
        num_nodes = adj.size(0)
        all_samples = []
        
        for _ in range(self.num_layers):
            layer_samples = []
            for node in range(num_nodes):
                neighbors = adj[node].nonzero(as_tuple=True)[0]
                
                if len(neighbors) == 0:
                    # 没有邻居，采样自身
                    sampled = torch.tensor([node] * num_samples)
                elif len(neighbors) >= num_samples:
                    # 随机采样
                    indices = torch.randperm(len(neighbors))[:num_samples]
                    sampled = neighbors[indices]
                else:
                    # 有放回采样
                    indices = torch.randint(0, len(neighbors), (num_samples,))
                    sampled = neighbors[indices]
                
                layer_samples.append(sampled)
            all_samples.append(layer_samples)
        
        return all_samples
    
    def get_neighbor_features(self, 
                             h: torch.Tensor, 
                             neighbor_indices: List[torch.Tensor]) -> torch.Tensor:
        """
        根据邻居索引获取邻居特征
        
        参数:
            h: 节点特征 (num_nodes, feature_dim)
            neighbor_indices: 邻居索引列表
            
        返回:
            邻居特征 (num_nodes, num_samples, feature_dim)
        """
        num_nodes = h.size(0)
        num_samples = len(neighbor_indices[0])
        feature_dim = h.size(1)
        
        neighbor_features = torch.zeros(num_nodes, num_samples, feature_dim, device=h.device)
        
        for i, neighbors in enumerate(neighbor_indices):
            neighbor_features[i] = h[neighbors]
        
        return neighbor_features


# =============================================================================
# 第五部分：应用示例 - 节点分类
# =============================================================================

class NodeClassificationTrainer:
    """节点分类训练器"""
    
    def __init__(self, model: nn.Module, lr: float = 0.01, weight_decay: float = 5e-4):
        """
        参数:
            model: GNN模型
            lr: 学习率
            weight_decay: L2正则化系数
        """
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, 
                    x: torch.Tensor, 
                    adj: torch.Tensor,
                    labels: torch.Tensor,
                    train_mask: torch.Tensor) -> float:
        """
        训练一个epoch
        
        返回:
            平均损失
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        logits = self.model(x, adj)
        
        # 只计算训练集的损失
        loss = self.criterion(logits[train_mask], labels[train_mask])
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self,
                 x: torch.Tensor,
                 adj: torch.Tensor,
                 labels: torch.Tensor,
                 mask: torch.Tensor) -> Tuple[float, float]:
        """
        评估模型
        
        返回:
            (损失, 准确率)
        """
        self.model.eval()
        
        logits = self.model(x, adj)
        loss = self.criterion(logits[mask], labels[mask])
        
        pred = logits[mask].argmax(dim=1)
        acc = (pred == labels[mask]).float().mean().item()
        
        return loss.item(), acc


# =============================================================================
# 第六部分：应用示例 - 链接预测
# =============================================================================

class LinkPredictionModel(nn.Module):
    """链接预测模型"""
    
    def __init__(self, encoder: nn.Module, decoder_type: str = 'dot'):
        """
        参数:
            encoder: GNN编码器（如GCN、GAT等）
            decoder_type: 解码器类型 ('dot', 'mlp', 'bilinear')
        """
        super(LinkPredictionModel, self).__init__()
        self.encoder = encoder
        self.decoder_type = decoder_type
        
        if decoder_type == 'mlp':
            # MLP解码器
            hidden_dim = encoder.layers[-1].out_features if hasattr(encoder, 'layers') else 64
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        elif decoder_type == 'bilinear':
            # 双线性解码器
            hidden_dim = encoder.layers[-1].out_features if hasattr(encoder, 'layers') else 64
            self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)
    
    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """编码节点特征"""
        return self.encoder(x, adj)
    
    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        解码边的存在概率
        
        参数:
            z: 节点嵌入 (num_nodes, hidden_dim)
            edge_index: 边索引 (2, num_edges)
            
        返回:
            边的存在概率 (num_edges,)
        """
        src, dst = edge_index[0], edge_index[1]
        z_src = z[src]
        z_dst = z[dst]
        
        if self.decoder_type == 'dot':
            # 点积
            prob = (z_src * z_dst).sum(dim=1)
        elif self.decoder_type == 'mlp':
            # MLP
            z_cat = torch.cat([z_src, z_dst], dim=1)
            prob = self.decoder(z_cat).squeeze(-1)
        elif self.decoder_type == 'bilinear':
            # 双线性
            prob = self.bilinear(z_src, z_dst).squeeze(-1)
        
        return torch.sigmoid(prob)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """完整前向传播"""
        z = self.encode(x, adj)
        return self.decode(z, edge_index)


class LinkPredictionTrainer:
    """链接预测训练器"""
    
    def __init__(self, model: LinkPredictionModel, lr: float = 0.01):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
    
    def train_epoch(self,
                    x: torch.Tensor,
                    adj: torch.Tensor,
                    pos_edge_index: torch.Tensor,
                    neg_edge_index: torch.Tensor) -> float:
        """训练一个epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 正样本预测
        pos_pred = self.model(x, adj, pos_edge_index)
        # 负样本预测
        neg_pred = self.model(x, adj, neg_edge_index)
        
        # 计算损失
        pos_loss = -torch.log(pos_pred + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_pred + 1e-15).mean()
        loss = pos_loss + neg_loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self,
                 x: torch.Tensor,
                 adj: torch.Tensor,
                 pos_edge_index: torch.Tensor,
                 neg_edge_index: torch.Tensor) -> dict:
        """评估模型"""
        self.model.eval()
        
        pos_pred = self.model(x, adj, pos_edge_index)
        neg_pred = self.model(x, adj, neg_edge_index)
        
        # 计算AUC
        pos_y = torch.ones(pos_pred.size(0))
        neg_y = torch.zeros(neg_pred.size(0))
        
        y_true = torch.cat([pos_y, neg_y])
        y_pred = torch.cat([pos_pred, neg_pred])
        
        # 简单准确率
        y_pred_binary = (y_pred > 0.5).float()
        acc = (y_pred_binary == y_true).float().mean().item()
        
        return {'accuracy': acc, 'pos_pred_mean': pos_pred.mean().item()}


# =============================================================================
# 第七部分：应用示例 - 分子性质预测
# =============================================================================

class MolecularGNN(nn.Module):
    """
    用于分子性质预测的GNN
    
    特点:
        - 支持边特征（键类型）
        - 全局池化得到分子表示
        - 输出连续值（回归任务）
    """
    
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 1,
                 num_layers: int = 3,
                 pooling: str = 'mean'):
        """
        参数:
            node_dim: 原子特征维度
            edge_dim: 键特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（1表示回归）
            num_layers: GNN层数
            pooling: 池化方式 ('mean', 'max', 'attention')
        """
        super(MolecularGNN, self).__init__()
        
        self.num_layers = num_layers
        self.pooling = pooling
        
        # 节点嵌入层
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        
        # GNN层（使用简单的消息传递）
        self.convs = nn.ModuleList()
        self.edge_nets = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(nn.Linear(hidden_dim, hidden_dim))
            # 边特征变换网络
            self.edge_nets.append(nn.Linear(edge_dim, hidden_dim))
        
        # 注意力池化
        if pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # 输出层
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, 
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 原子特征 (num_nodes, node_dim)
            edge_index: 边索引 (2, num_edges)
            edge_attr: 边特征 (num_edges, edge_dim)
            batch: 每个节点所属的分子 (num_nodes,)
            
        返回:
            分子性质预测 (num_graphs, output_dim)
        """
        # 初始嵌入
        h = self.node_embed(x)
        
        # 消息传递
        for conv, edge_net in zip(self.convs, self.edge_nets):
            # 收集邻居消息
            src, dst = edge_index[0], edge_index[1]
            
            # 边特征变换
            edge_msg = edge_net(edge_attr)
            
            # 消息聚合 (简化版)
            msg = torch.zeros_like(h)
            msg.index_add_(0, dst, h[src] + edge_msg)
            
            # 更新
            h = conv(h + msg)
            h = F.relu(h)
        
        # 全局池化
        num_graphs = batch.max().item() + 1
        
        if self.pooling == 'mean':
            # 均值池化
            h_graph = torch.zeros(num_graphs, h.size(1), device=h.device)
            count = torch.zeros(num_graphs, device=h.device)
            h_graph.index_add_(0, batch, h)
            count.index_add_(0, batch, torch.ones(h.size(0), device=h.device))
            h_graph = h_graph / count.unsqueeze(1).clamp(min=1)
        
        elif self.pooling == 'max':
            # 最大池化
            h_graph = torch.zeros(num_graphs, h.size(1), device=h.device)
            for i in range(num_graphs):
                mask = (batch == i)
                if mask.any():
                    h_graph[i] = h[mask].max(dim=0)[0]
        
        elif self.pooling == 'attention':
            # 注意力池化
            attn_weights = self.attention(h).squeeze(-1)
            attn_weights = torch.exp(attn_weights)
            
            h_graph = torch.zeros(num_graphs, h.size(1), device=h.device)
            attn_sum = torch.zeros(num_graphs, device=h.device)
            
            h_weighted = h * attn_weights.unsqueeze(1)
            h_graph.index_add_(0, batch, h_weighted)
            attn_sum.index_add_(0, batch, attn_weights)
            
            h_graph = h_graph / attn_sum.unsqueeze(1).clamp(min=1)
        
        # 输出
        out = self.readout(h_graph)
        
        return out


# =============================================================================
# 第八部分：数据集和辅助函数
# =============================================================================

def create_synthetic_graph(num_nodes: int = 100, 
                           num_classes: int = 4,
                           feature_dim: int = 16) -> dict:
    """
    创建一个合成图数据集（用于测试）
    
    返回:
        包含x, adj, labels, train_mask, val_mask, test_mask的字典
    """
    # 创建特征
    x = torch.randn(num_nodes, feature_dim)
    
    # 创建标签（基于社区结构）
    nodes_per_class = num_nodes // num_classes
    labels = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(num_classes):
        start = i * nodes_per_class
        end = start + nodes_per_class if i < num_classes - 1 else num_nodes
        labels[start:end] = i
    
    # 创建邻接矩阵（社区内部连接紧密，社区之间连接稀疏）
    adj = torch.zeros(num_nodes, num_nodes)
    
    for i in range(num_nodes):
        # 同社区连接
        same_community = (labels == labels[i]).nonzero(as_tuple=True)[0]
        for j in same_community:
            if i != j and torch.rand(1).item() < 0.3:
                adj[i, j] = 1
        
        # 跨社区连接
        for j in range(num_nodes):
            if labels[i] != labels[j] and torch.rand(1).item() < 0.05:
                adj[i, j] = 1
    
    # 划分训练/验证/测试集
    indices = torch.randperm(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    return {
        'x': x,
        'adj': adj,
        'labels': labels,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask
    }


def visualize_graph(adj: np.ndarray, labels: Optional[np.ndarray] = None, 
                   title: str = "Graph Visualization"):
    """
    可视化图结构（使用matplotlib）
    
    参数:
        adj: 邻接矩阵
        labels: 节点标签（用于着色）
        title: 图标题
    """
    # 使用简单的圆形布局
    n = adj.shape[0]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.column_stack([np.cos(angles), np.sin(angles)])
    
    plt.figure(figsize=(10, 10))
    
    # 绘制边
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] > 0:
                plt.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 
                        'gray', alpha=0.3, linewidth=0.5)
    
    # 绘制节点
    if labels is not None:
        scatter = plt.scatter(pos[:, 0], pos[:, 1], c=labels, cmap='tab10', s=100)
        plt.colorbar(scatter, label='Class')
    else:
        plt.scatter(pos[:, 0], pos[:, 1], s=100)
    
    plt.title(title)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    return plt


def visualize_attention(alpha: torch.Tensor, node_idx: int = 0):
    """
    可视化注意力权重
    
    参数:
        alpha: 注意力权重 (num_heads, num_nodes, num_nodes)
        node_idx: 要可视化的中心节点
    """
    num_heads = alpha.size(0)
    
    fig, axes = plt.subplots(1, min(num_heads, 4), figsize=(16, 4))
    if num_heads == 1:
        axes = [axes]
    
    for h in range(min(num_heads, 4)):
        ax = axes[h] if num_heads > 1 else axes[0]
        attn = alpha[h, node_idx].cpu().numpy()
        
        # 只显示非零的注意力
        nonzero_indices = np.where(attn > 0.01)[0]
        if len(nonzero_indices) > 0:
            ax.bar(range(len(nonzero_indices)), attn[nonzero_indices])
            ax.set_xticks(range(len(nonzero_indices)))
            ax.set_xticklabels(nonzero_indices, rotation=45)
        
        ax.set_title(f'Head {h+1}')
        ax.set_xlabel('Neighbor Index')
        ax.set_ylabel('Attention Weight')
    
    plt.suptitle(f'Attention Weights for Node {node_idx}')
    plt.tight_layout()
    return plt


# =============================================================================
# 第九部分：运行示例
# =============================================================================

def demo_node_classification():
    """节点分类演示"""
    print("="*60)
    print("演示: 节点分类 (Node Classification)")
    print("="*60)
    
    # 创建合成数据集
    data = create_synthetic_graph(num_nodes=200, num_classes=4, feature_dim=16)
    
    x = data['x']
    adj = data['adj']
    labels = data['labels']
    train_mask = data['train_mask']
    val_mask = data['val_mask']
    test_mask = data['test_mask']
    
    # 归一化邻接矩阵
    adj_norm = GraphUtils.normalize_adjacency(adj.numpy())
    adj_norm = torch.FloatTensor(adj_norm)
    
    # 创建GCN模型
    model = GCN(input_dim=16, hidden_dim=32, output_dim=4, num_layers=2)
    trainer = NodeClassificationTrainer(model, lr=0.01)
    
    print(f"\n数据集信息:")
    print(f"  节点数: {x.size(0)}")
    print(f"  特征维度: {x.size(1)}")
    print(f"  类别数: {labels.max().item() + 1}")
    print(f"  训练集: {train_mask.sum().item()} 个节点")
    print(f"  验证集: {val_mask.sum().item()} 个节点")
    print(f"  测试集: {test_mask.sum().item()} 个节点")
    
    print(f"\n模型: GCN (2层, 隐藏维度=32)")
    
    # 训练
    best_val_acc = 0
    for epoch in range(100):
        loss = trainer.train_epoch(x, adj_norm, labels, train_mask)
        
        if (epoch + 1) % 20 == 0:
            val_loss, val_acc = trainer.evaluate(x, adj_norm, labels, val_mask)
            test_loss, test_acc = trainer.evaluate(x, adj_norm, labels, test_mask)
            print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
    
    # 最终测试
    _, final_test_acc = trainer.evaluate(x, adj_norm, labels, test_mask)
    print(f"\n最终测试准确率: {final_test_acc:.4f}")


def demo_gat_attention():
    """GAT注意力可视化演示"""
    print("\n" + "="*60)
    print("演示: GAT注意力机制可视化")
    print("="*60)
    
    # 创建小图
    num_nodes = 10
    x = torch.randn(num_nodes, 8)
    
    # 创建星型图结构
    adj = torch.zeros(num_nodes, num_nodes)
    center = 0
    for i in range(1, num_nodes):
        adj[center, i] = 1
        adj[i, center] = 1
    
    # 添加一些外围边
    adj[1, 2] = adj[2, 1] = 1
    adj[3, 4] = adj[4, 3] = 1
    
    print(f"\n图结构: 星型图 (中心节点0，{num_nodes-1}个外围节点)")
    
    # 创建GAT模型
    gat = GAT(input_dim=8, hidden_dim=8, output_dim=4, num_layers=2, num_heads=2)
    
    # 前向传播并获取注意力权重
    logits, attention_weights = gat(x, adj, return_attention=True)
    
    print(f"\n注意力分析 (中心节点 {center}):")
    print(f"  层数: {len(attention_weights)}")
    print(f"  头数: {attention_weights[0].size(0)}")
    
    # 分析第一层的注意力
    attn_layer1 = attention_weights[0]  # (num_heads, num_nodes, num_nodes)
    
    for head in range(attn_layer1.size(0)):
        attn_center = attn_layer1[head, center]  # 中心节点对所有节点的注意力
        neighbor_attn = attn_center[1:].sum().item()  # 对外围节点的总注意力
        self_attn = attn_center[center].item()  # 自注意力
        
        print(f"\n  Head {head+1}:")
        print(f"    对外围节点的注意力总和: {neighbor_attn:.4f}")
        print(f"    自注意力: {self_attn:.4f}")


def demo_link_prediction():
    """链接预测演示"""
    print("\n" + "="*60)
    print("演示: 链接预测 (Link Prediction)")
    print("="*60)
    
    # 创建图
    num_nodes = 100
    data = create_synthetic_graph(num_nodes=num_nodes, num_classes=2, feature_dim=16)
    
    x = data['x']
    adj = data['adj']
    
    # 划分训练边和测试边
    edges = adj.nonzero(as_tuple=False).t()
    num_edges = edges.size(1)
    
    # 随机划分
    perm = torch.randperm(num_edges)
    train_size = int(0.8 * num_edges)
    
    train_edges = edges[:, perm[:train_size]]
    test_edges = edges[:, perm[train_size:]]
    
    # 创建负样本（不存在的边）
    def sample_negative_edges(adj, num_samples):
        """采样负边"""
        neg_edges = []
        while len(neg_edges) < num_samples:
            i = torch.randint(0, num_nodes, (1,)).item()
            j = torch.randint(0, num_nodes, (1,)).item()
            if i != j and adj[i, j] == 0 and [i, j] not in neg_edges:
                neg_edges.append([i, j])
        return torch.tensor(neg_edges).t()
    
    neg_train = sample_negative_edges(adj, train_size)
    neg_test = sample_negative_edges(adj, num_edges - train_size)
    
    # 训练邻接矩阵（只包含训练边）
    adj_train = torch.zeros_like(adj)
    adj_train[train_edges[0], train_edges[1]] = 1
    adj_train = GraphUtils.normalize_adjacency(adj_train.numpy())
    adj_train = torch.FloatTensor(adj_train)
    
    print(f"\n数据集信息:")
    print(f"  节点数: {num_nodes}")
    print(f"  总边数: {num_edges}")
    print(f"  训练边: {train_size}")
    print(f"  测试边: {num_edges - train_size}")
    
    # 创建模型
    encoder = GCN(input_dim=16, hidden_dim=32, output_dim=16, num_layers=2)
    model = LinkPredictionModel(encoder, decoder_type='dot')
    trainer = LinkPredictionTrainer(model, lr=0.01)
    
    print(f"\n模型: GCN编码器 + 点积解码器")
    
    # 训练
    for epoch in range(50):
        loss = trainer.train_epoch(x, adj_train, train_edges, neg_train)
        
        if (epoch + 1) % 10 == 0:
            metrics = trainer.evaluate(x, adj_train, test_edges, neg_test)
            print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | Test Acc: {metrics['accuracy']:.4f}")


def demo_graphsage():
    """GraphSAGE演示"""
    print("\n" + "="*60)
    print("演示: GraphSAGE (归纳式学习)")
    print("="*60)
    
    # 创建训练图和测试图
    train_data = create_synthetic_graph(num_nodes=150, num_classes=3, feature_dim=16)
    test_data = create_synthetic_graph(num_nodes=50, num_classes=3, feature_dim=16)
    
    # GraphSAGE可以进行归纳式学习：在训练图上训练，在测试图上测试
    x_train = train_data['x']
    adj_train = train_data['adj']
    labels_train = train_data['labels']
    train_mask = train_data['train_mask']
    
    x_test = test_data['x']
    adj_test = test_data['adj']
    labels_test = test_data['labels']
    
    print(f"\n训练图:")
    print(f"  节点数: {x_train.size(0)}")
    print(f"  边数: {adj_train.sum().item()}")
    
    print(f"\n测试图 (全新图结构):")
    print(f"  节点数: {x_test.size(0)}")
    print(f"  边数: {adj_test.sum().item()}")
    
    # 创建GraphSAGE模型
    model = GraphSAGE(input_dim=16, hidden_dim=32, output_dim=3, 
                     num_layers=2, num_samples=5, aggregator='mean')
    
    # 归一化邻接矩阵
    adj_train_norm = torch.FloatTensor(GraphUtils.normalize_adjacency(adj_train.numpy()))
    adj_test_norm = torch.FloatTensor(GraphUtils.normalize_adjacency(adj_test.numpy()))
    
    # 简单训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n训练GraphSAGE...")
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        
        logits = model(x_train, adj_train_norm)
        loss = criterion(logits[train_mask], labels_train[train_mask])
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            # 在训练集上评估
            model.eval()
            with torch.no_grad():
                train_logits = model(x_train, adj_train_norm)
                train_acc = (train_logits[train_mask].argmax(1) == labels_train[train_mask]).float().mean()
                
                # 在全新测试图上评估（归纳式学习！）
                test_logits = model(x_test, adj_test_norm)
                test_acc = (test_logits.argmax(1) == labels_test).float().mean()
            
            print(f"Epoch {epoch+1:3d} | Train Acc: {train_acc:.4f} | Test Acc (新图): {test_acc:.4f}")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  第三十二章代码: 图神经网络基础 (Graph Neural Networks)")
    print("="*70)
    
    # 运行演示
    demo_node_classification()
    demo_gat_attention()
    demo_link_prediction()
    demo_graphsage()
    
    print("\n" + "="*70)
    print("  所有演示完成!")
    print("="*70)
    
    # 统计代码行数
    import inspect
    source = inspect.getsource(inspect.currentframe().f_code)
    lines = len(source.split('\n'))
    print(f"\n本代码文件总行数: {lines} 行")
