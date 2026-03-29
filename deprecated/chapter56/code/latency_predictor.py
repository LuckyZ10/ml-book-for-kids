"""
LatencyPredictor: 神经网络延迟预测器
====================================

在实际部署中，准确预测模型的推理延迟至关重要。
本模块实现两种延迟预测方法：
1. 查找表法 (Lookup Table)
2. 图神经网络法 (GNN-based)

费曼法比喻：
查找表法就像查字典：
- 事先测量好各种操作的延迟
- 需要预测时，把各操作的延迟相加

GNN法就像预测交通时间：
- 不仅看每条路的限速（操作类型）
- 还要看道路之间的连接（数据依赖）
- 以及拥堵情况（硬件资源竞争）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class OpConfig:
    """操作配置"""
    name: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    params: Dict = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class LatencyRecord:
    """延迟记录"""
    op_config: OpConfig
    device: str
    batch_size: int
    latency_ms: float
    
    def to_key(self) -> str:
        """生成查找键"""
        shape_str = 'x'.join(map(str, self.op_config.input_shape))
        return f"{self.op_config.name}_{shape_str}_{self.device}_{self.batch_size}"


class LookupTablePredictor:
    """
    基于查找表的延迟预测器
    
    原理：
    1. 预测量常见操作的延迟
    2. 存储在查找表中
    3. 预测时查表并累加
    
    优点：简单、快速、准确
    缺点：需要预测量、无法处理未见过的配置
    
    比喻：就像餐厅的菜单和价格表
    - 每个菜（操作）都有定价（延迟）
    - 顾客点餐时直接查表计算总价
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.lookup_table: Dict[str, float] = {}
        self.op_statistics: Dict[str, List[float]] = defaultdict(list)
    
    def add_measurement(self, record: LatencyRecord):
        """添加测量记录到查找表"""
        key = record.to_key()
        self.lookup_table[key] = record.latency_ms
        self.op_statistics[record.op_config.name].append(record.latency_ms)
    
    def predict(self, op_config: OpConfig, batch_size: int = 1) -> float:
        """
        预测单个操作的延迟
        
        如果找不到精确匹配，使用插值或默认值
        """
        key = LatencyRecord(op_config, self.device, batch_size, 0).to_key()
        
        # 精确匹配
        if key in self.lookup_table:
            return self.lookup_table[key]
        
        # 查找相似配置进行插值
        similar = self._find_similar(op_config, batch_size)
        if similar:
            return np.mean(similar)
        
        # 使用操作类型的平均值
        if op_config.name in self.op_statistics:
            return np.mean(self.op_statistics[op_config.name])
        
        # 默认估计
        return self._estimate_default(op_config)
    
    def predict_model(self, operations: List[OpConfig], 
                     batch_size: int = 1) -> float:
        """预测整个模型的延迟"""
        total_latency = 0.0
        for op in operations:
            total_latency += self.predict(op, batch_size)
        return total_latency
    
    def _find_similar(self, op_config: OpConfig, 
                     batch_size: int, threshold: float = 0.2) -> List[float]:
        """查找相似配置的延迟"""
        candidates = []
        
        for key, latency in self.lookup_table.items():
            parts = key.split('_')
            if len(parts) < 2:
                continue
            
            op_name = parts[0]
            if op_name != op_config.name:
                continue
            
            # 简单的形状相似度检查
            candidates.append(latency)
        
        return candidates[:5]  # 返回最多5个相似值
    
    def _estimate_default(self, op_config: OpConfig) -> float:
        """默认延迟估计"""
        # 基于FLOPs的简单估计
        input_size = np.prod(op_config.input_shape)
        
        base_latency = {
            'conv1d': 0.1,
            'conv2d': 0.5,
            'conv3d': 2.0,
            'linear': 0.1,
            'relu': 0.01,
            'maxpool': 0.05,
            'avgpool': 0.05,
            'batchnorm': 0.02,
            'dropout': 0.01,
        }
        
        op_type = op_config.name.lower()
        base = base_latency.get(op_type, 0.1)
        
        # 根据输入大小调整
        return base * (input_size / 1000 + 1)
    
    def calibrate(self, measure_fn: Callable[[OpConfig], float],
                 op_configs: List[OpConfig]):
        """
        校准查找表
        
        使用实际测量更新查找表
        """
        print("开始校准查找表...")
        for i, config in enumerate(op_configs):
            latency = measure_fn(config)
            record = LatencyRecord(config, self.device, 1, latency)
            self.add_measurement(record)
            
            if (i + 1) % 10 == 0:
                print(f"  已校准 {i+1}/{len(op_configs)} 个操作")
        
        print(f"校准完成！查找表包含 {len(self.lookup_table)} 条记录")
    
    def save(self, filepath: str):
        """保存查找表到文件"""
        data = {
            'lookup_table': self.lookup_table,
            'device': self.device,
            'statistics': dict(self.op_statistics)
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"查找表已保存到 {filepath}")
    
    def load(self, filepath: str):
        """从文件加载查找表"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.lookup_table = data['lookup_table']
        self.device = data['device']
        self.op_statistics = defaultdict(list, data['statistics'])
        print(f"已从 {filepath} 加载查找表，包含 {len(self.lookup_table)} 条记录")


class GNNEncoder(nn.Module):
    """
    基于GNN的延迟预测编码器
    
    将网络架构视为图，使用图神经网络学习延迟预测。
    
    节点特征：
    - 操作类型（one-hot）
    - 输入/输出形状
    - 参数数量
    
    边特征：
    - 数据依赖关系
    - 张量大小
    """
    
    def __init__(self, 
                 node_feature_dim: int = 32,
                 edge_feature_dim: int = 16,
                 hidden_dim: int = 64,
                 num_layers: int = 3):
        super().__init__()
        
        self.num_layers = num_layers
        
        # 节点嵌入
        self.node_embed = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 边嵌入
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 图卷积层（简化版GAT）
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=4,
                    dim_feedforward=hidden_dim * 2,
                    batch_first=True
                )
            )
        
        # 输出头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, 
               node_features: torch.Tensor,
               edge_index: torch.Tensor,
               edge_features: torch.Tensor,
               batch_mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_features: [num_nodes, node_feature_dim]
            edge_index: [2, num_edges]
            edge_features: [num_edges, edge_feature_dim]
            batch_mask: [num_nodes] 标识每个节点属于哪个图
        
        Returns:
            latency: [num_graphs] 每个图的预测延迟
        """
        # 嵌入
        h_nodes = self.node_embed(node_features)
        h_edges = self.edge_embed(edge_features)
        
        # 使用边信息更新节点特征（简化版消息传递）
        for gnn_layer in self.gnn_layers:
            # 添加位置编码
            h_nodes = gnn_layer(h_nodes.unsqueeze(0)).squeeze(0)
        
        # 全局平均池化
        if batch_mask is not None:
            # 多个图的情况
            num_graphs = batch_mask.max().item() + 1
            graph_embeddings = []
            for i in range(num_graphs):
                mask = batch_mask == i
                graph_emb = h_nodes[mask].mean(dim=0)
                graph_embeddings.append(graph_emb)
            h_graph = torch.stack(graph_embeddings)
        else:
            # 单个图
            h_graph = h_nodes.mean(dim=0, keepdim=True)
        
        # 预测延迟
        latency = self.predictor(h_graph).squeeze(-1)
        return latency


class GNNLatencyPredictor:
    """
    基于GNN的延迟预测器
    
    能够捕捉：
    1. 操作之间的数据依赖
    2. 硬件特定的优化效果
    3. 复杂的并行模式
    
    比喻：就像预测城市交通时间
    - 不仅看每条路的长度（操作FLOPs）
    - 还要看红绿灯（同步点）
    - 拥堵情况（内存带宽瓶颈）
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = GNNEncoder().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        
        # 操作类型编码
        self.op_types = [
            'conv1d', 'conv2d', 'conv3d', 'linear',
            'relu', 'leaky_relu', 'gelu',
            'maxpool', 'avgpool', 'adaptive_avgpool',
            'batchnorm', 'layernorm', 'instancenorm',
            'dropout', 'dropout2d', 'dropout3d',
            'add', 'mul', 'concat',
            'none', 'identity', 'skip'
        ]
    
    def encode_operation(self, op_config: OpConfig) -> torch.Tensor:
        """将操作配置编码为特征向量"""
        # one-hot编码操作类型
        op_type_onehot = torch.zeros(len(self.op_types))
        if op_config.name in self.op_types:
            idx = self.op_types.index(op_config.name)
            op_type_onehot[idx] = 1.0
        
        # 编码形状信息
        shape_features = torch.zeros(8)
        for i, dim in enumerate(op_config.input_shape[:4]):
            shape_features[i] = np.log2(dim + 1) / 10
        for i, dim in enumerate(op_config.output_shape[:4]):
            shape_features[i + 4] = np.log2(dim + 1) / 10
        
        # 参数信息
        params = torch.tensor([
            op_config.params.get('kernel_size', 0) / 10,
            op_config.params.get('stride', 0) / 10,
            op_config.params.get('groups', 1) / 100,
        ])
        
        return torch.cat([op_type_onehot, shape_features, params])
    
    def build_graph(self, operations: List[OpConfig]) -> Tuple[torch.Tensor, ...]:
        """
        从操作列表构建计算图
        
        简化版：假设顺序执行
        """
        num_ops = len(operations)
        
        # 节点特征
        node_features = []
        for op in operations:
            feat = self.encode_operation(op)
            node_features.append(feat)
        node_features = torch.stack(node_features).to(self.device)
        
        # 边：顺序连接
        edges = []
        edge_features = []
        for i in range(num_ops - 1):
            edges.append([i, i + 1])
            # 边特征：数据大小
            data_size = np.prod(operations[i].output_shape) / 1e6  # MB
            edge_features.append([data_size])
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().to(self.device)
            edge_features = torch.tensor(edge_features, dtype=torch.float).to(self.device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long).to(self.device)
            edge_features = torch.zeros((0, 1)).to(self.device)
        
        return node_features, edge_index, edge_features
    
    def predict(self, operations: List[OpConfig]) -> float:
        """预测模型延迟"""
        self.model.eval()
        
        with torch.no_grad():
            node_feat, edge_idx, edge_feat = self.build_graph(operations)
            
            # 添加batch维度
            latency = self.model(
                node_feat.unsqueeze(0).expand(10, -1, -1).reshape(-1, node_feat.size(-1)),
                edge_idx,
                edge_feat
            )
        
        return latency[0].item()
    
    def train_step(self, 
                  operations_batch: List[List[OpConfig]],
                  latencies: torch.Tensor) -> float:
        """训练一步"""
        self.model.train()
        self.optimizer.zero_grad()
        
        predictions = []
        for ops in operations_batch:
            node_feat, edge_idx, edge_feat = self.build_graph(ops)
            pred = self.model(node_feat, edge_idx, edge_feat)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        loss = self.criterion(predictions, latencies.to(self.device))
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class HardwareAwareLatencyPredictor:
    """
    硬件感知延迟预测器
    
    结合多种预测方法，并根据硬件特性调整
    """
    
    def __init__(self, hardware_name: str = 'generic'):
        self.hardware_name = hardware_name
        self.lookup_table = LookupTablePredictor()
        self.gnn_predictor = None
        
        # 硬件特定参数
        self.hardware_params = self._get_hardware_params(hardware_name)
    
    def _get_hardware_params(self, name: str) -> Dict:
        """获取硬件参数"""
        params = {
            'generic': {
                'memory_bw': 100,  # GB/s
                'compute_peak': 10,  # TFLOPS
                'cache_size': 32,  # MB
            },
            'mobile_cpu': {
                'memory_bw': 25,
                'compute_peak': 0.5,
                'cache_size': 4,
            },
            'mobile_gpu': {
                'memory_bw': 50,
                'compute_peak': 2.0,
                'cache_size': 2,
            },
            'edge_tpu': {
                'memory_bw': 10,
                'compute_peak': 4.0,
                'cache_size': 8,
            },
            'server_gpu': {
                'memory_bw': 900,
                'compute_peak': 300,
                'cache_size': 80,
            }
        }
        return params.get(name, params['generic'])
    
    def predict(self, operations: List[OpConfig], 
               method: str = 'lookup') -> float:
        """
        预测延迟
        
        Args:
            operations: 操作列表
            method: 'lookup', 'gnn', 或 'hybrid'
        """
        if method == 'lookup':
            return self.lookup_table.predict_model(operations)
        
        elif method == 'gnn':
            if self.gnn_predictor is None:
                self.gnn_predictor = GNNLatencyPredictor()
            return self.gnn_predictor.predict(operations)
        
        elif method == 'hybrid':
            # 结合两种方法
            lookup_latency = self.lookup_table.predict_model(operations)
            
            # 如果GNN可用，使用它进行微调
            if self.gnn_predictor is not None:
                gnn_latency = self.gnn_predictor.predict(operations)
                # 加权平均
                return 0.7 * lookup_latency + 0.3 * gnn_latency
            
            return lookup_latency
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def roofline_model(self, op_config: OpConfig) -> float:
        """
        Roofline模型预测
        
        理论性能 = min(计算峰值, 内存带宽 × 计算强度)
        """
        # 估计FLOPs和内存访问
        flops = self._estimate_flops(op_config)
        bytes_accessed = self._estimate_memory_access(op_config)
        
        if bytes_accessed == 0:
            return 0.0
        
        # 计算强度
        intensity = flops / bytes_accessed  # FLOPs/Byte
        
        # Roofline
        compute_bound = flops / (self.hardware_params['compute_peak'] * 1e12)
        memory_bound = bytes_accessed / (self.hardware_params['memory_bw'] * 1e9)
        
        return max(compute_bound, memory_bound) * 1000  # ms
    
    def _estimate_flops(self, op_config: OpConfig) -> float:
        """估计FLOPs"""
        name = op_config.name.lower()
        
        if 'conv' in name:
            # 卷积: FLOPs = 2 * Cin * Cout * K * K * H * W
            in_shape = op_config.input_shape
            out_shape = op_config.output_shape
            kernel = op_config.params.get('kernel_size', 3)
            
            if len(in_shape) >= 4:  # 2D conv
                cin, h, w = in_shape[1], in_shape[2], in_shape[3]
                cout = out_shape[1]
                return 2 * cin * cout * kernel * kernel * h * w
        
        elif 'linear' in name:
            # 全连接: FLOPs = 2 * in_features * out_features
            return 2 * np.prod(op_config.input_shape[1:]) * np.prod(op_config.output_shape[1:])
        
        return 0.0
    
    def _estimate_memory_access(self, op_config: OpConfig) -> float:
        """估计内存访问量（字节）"""
        input_bytes = np.prod(op_config.input_shape) * 4  # float32
        output_bytes = np.prod(op_config.output_shape) * 4
        return input_bytes + output_bytes


def demo_latency_predictor():
    """
    延迟预测器演示
    
    小测试：
    1. 查找表法和GNN法各有什么优缺点？
    2. Roofline模型如何帮助理解性能瓶颈？
    3. 如何为新的硬件平台构建延迟预测器？
    """
    print("=" * 70)
    print("延迟预测器演示")
    print("=" * 70)
    
    # 创建测试操作
    operations = [
        OpConfig('conv2d', (1, 3, 32, 32), (1, 64, 32, 32), 
                {'kernel_size': 3, 'stride': 1}),
        OpConfig('relu', (1, 64, 32, 32), (1, 64, 32, 32)),
        OpConfig('conv2d', (1, 64, 32, 32), (1, 128, 16, 16),
                {'kernel_size': 3, 'stride': 2}),
        OpConfig('batchnorm', (1, 128, 16, 16), (1, 128, 16, 16)),
        OpConfig('relu', (1, 128, 16, 16), (1, 128, 16, 16)),
        OpConfig('avgpool', (1, 128, 16, 16), (1, 128, 1, 1),
                {'kernel_size': 16}),
        OpConfig('linear', (1, 128), (1, 10)),
    ]
    
    print(f"\n测试模型结构：")
    print(f"  操作数: {len(operations)}")
    for i, op in enumerate(operations):
        print(f"  {i+1}. {op.name}: {op.input_shape} -> {op.output_shape}")
    
    # 查找表预测器
    print("\n" + "-" * 70)
    print("1. 查找表预测器")
    print("-" * 70)
    
    lookup = LookupTablePredictor(device='cpu')
    
    # 添加一些模拟测量值
    for i, op in enumerate(operations):
        # 模拟测量值
        base_latency = {
            'conv2d': 2.5, 'relu': 0.1, 'batchnorm': 0.3,
            'avgpool': 0.5, 'linear': 0.8
        }.get(op.name, 1.0)
        
        record = LatencyRecord(op, 'cpu', 1, base_latency * (1 + np.random.rand() * 0.2))
        lookup.add_measurement(record)
    
    lookup_latency = lookup.predict_model(operations)
    print(f"预测总延迟: {lookup_latency:.2f} ms")
    
    # 单个操作延迟
    print("\n各操作延迟:")
    for op in operations:
        lat = lookup.predict(op)
        print(f"  {op.name}: {lat:.2f} ms")
    
    # 硬件感知预测器
    print("\n" + "-" * 70)
    print("2. 硬件感知预测器")
    print("-" * 70)
    
    for hw in ['mobile_cpu', 'mobile_gpu', 'server_gpu']:
        hw_predictor = HardwareAwareLatencyPredictor(hardware_name=hw)
        latency = hw_predictor.predict(operations, method='lookup')
        
        print(f"\n{hw}:")
        print(f"  参数: {hw_predictor.hardware_params}")
        print(f"  预测延迟: {latency:.2f} ms")
        
        # Roofline分析
        op = operations[0]  # 以第一个卷积为例
        roofline_lat = hw_predictor.roofline_model(op)
        print(f"  Roofline预测(Conv2D): {roofline_lat:.2f} ms")
    
    # GNN预测器
    print("\n" + "-" * 70)
    print("3. GNN预测器")
    print("-" * 70)
    print("  (需要训练数据才能使用)")
    print("  GNN可以捕捉操作间的依赖关系")
    print("  适合复杂网络和异构硬件")
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    
    return lookup, operations


if __name__ == "__main__":
    demo_latency_predictor()
