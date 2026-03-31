"""
第三十四章：神经架构搜索(NAS) - 代码实现
=====================================
包含以下核心模块：
1. SearchSpace: 定义Cell-based搜索空间
2. NASNetController: RNN控制器生成架构
3. ENASSearcher: 权重共享超网络
4. DARTSSearcher: 可微分架构搜索核心
5. MixedOperation: 连续松弛操作
6. ArchitectureEvaluator: 架构性能评估
7. ZeroCostProxy: NASWOT/SNIP/SynFlow实现
8. NASVisualizer: 搜索过程可视化

作者: 《机器学习与深度学习：从小学生到大师》
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import namedtuple
import matplotlib.pyplot as plt
import networkx as nx
from abc import ABC, abstractmethod


# ============================================================================
# 配置和工具函数
# ============================================================================

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

# 候选操作集合（NASNet/DARTS标准操作）
PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths module."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ============================================================================
# 基础操作定义
# ============================================================================

class ReLUConvBN(nn.Module):
    """ReLU -> Conv -> BN"""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    """Dilated separable convolution."""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, 
                     padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    """Separable convolution."""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, 
                     padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, 
                     padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    """Identity operation."""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    """Zero operation (disconnected)."""
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
    """Reduce feature map size by factorized convolution."""
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


# ============================================================================
# 1. SearchSpace: Cell-based搜索空间定义
# ============================================================================

class SearchSpace:
    """
    NASNet/DARTS风格的Cell-based搜索空间。
    
    每个Cell是一个有向无环图(DAG)，包含：
    - 2个输入节点
    - num_nodes个中间节点
    - 1个输出节点（拼接所有中间节点）
    """
    
    def __init__(self, num_nodes: int = 4, num_ops: int = 8):
        """
        Args:
            num_nodes: 中间节点数量
            num_ops: 候选操作数量
        """
        self.num_nodes = num_nodes
        self.num_ops = num_ops
        self.primitives = PRIMITIVES[:num_ops]
        
    def get_num_edges(self) -> int:
        """计算Cell中的边数量。"""
        # 每个节点i连接前面的所有节点0,1,...,i-1
        # 节点0和1是输入，所以从节点2开始
        num_edges = 0
        for i in range(2, 2 + self.num_nodes):
            num_edges += i  # 节点i有i个输入选择
        return num_edges
    
    def get_architecture_size(self) -> int:
        """获取架构描述的大小。"""
        return self.get_num_edges()
    
    def sample_random_architecture(self) -> List[Tuple[int, int]]:
        """
        随机采样一个架构。
        
        Returns:
            架构描述: [(op_id, prev_node), ...]
        """
        arch = []
        for node_idx in range(2, 2 + self.num_nodes):
            # 每个节点有2条输入边
            for _ in range(2):
                # 随机选择前驱节点
                prev_node = np.random.randint(0, node_idx)
                # 随机选择操作
                op_id = np.random.randint(0, self.num_ops)
                arch.append((op_id, prev_node))
        return arch
    
    def encode_architecture(self, arch: List[Tuple[int, int]]) -> np.ndarray:
        """
        将架构编码为向量。
        
        Args:
            arch: 架构描述
        Returns:
            编码向量
        """
        return np.array([op_id for op_id, _ in arch])
    
    def decode_to_genotype(self, arch: List[Tuple[int, int]]) -> Genotype:
        """
        将架构描述解码为Genotype格式。
        
        Args:
            arch: 架构描述
        Returns:
            Genotype命名元组
        """
        normal = []
        idx = 0
        for node_idx in range(2, 2 + self.num_nodes):
            edges = []
            for _ in range(2):
                op_id, prev_node = arch[idx]
                edges.append((self.primitives[op_id], prev_node))
                idx += 1
            normal.extend(edges)
        
        return Genotype(
            normal=normal,
            normal_concat=list(range(2, 2 + self.num_nodes)),
            reduce=normal,  # 简化：normal和reduce使用相同结构
            reduce_concat=list(range(2, 2 + self.num_nodes))
        )


# ============================================================================
# 2. NASNetController: RNN控制器
# ============================================================================

class NASNetController(nn.Module):
    """
    NASNet风格的RNN控制器，用于生成神经网络架构。
    
    控制器是一个自回归模型，按顺序决策：
    1. 选择前驱节点
    2. 选择操作类型
    """
    
    def __init__(self, 
                 search_space: SearchSpace,
                 lstm_hidden_size: int = 100,
                 lstm_num_layers: int = 1,
                 temperature: float = 5.0,
                 tanh_constant: float = 2.5):
        super(NASNetController, self).__init__()
        
        self.search_space = search_space
        self.lstm_hidden_size = lstm_hidden_size
        self.num_nodes = search_space.num_nodes
        self.temperature = temperature
        self.tanh_constant = tanh_constant
        
        # 嵌入层
        self.encoder = nn.Embedding(search_space.num_ops + 1, lstm_hidden_size)
        
        # LSTM控制器
        self.lstm = nn.LSTM(lstm_hidden_size, lstm_hidden_size, lstm_num_layers, batch_first=True)
        
        # 决策头
        self.decoders = nn.ModuleList()
        for node_idx in range(2, 2 + self.num_nodes):
            # 为每个节点选择前驱节点
            self.decoders.append(nn.Linear(lstm_hidden_size, node_idx))
            # 为每条边选择操作
            self.decoders.append(nn.Linear(lstm_hidden_size, search_space.num_ops))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """重置参数。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.uniform_(param, -0.1, 0.1)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
    
    def forward(self, batch_size: int = 1) -> Tuple[List[Tuple[int, int]], torch.Tensor, torch.Tensor]:
        """
        前向传播，生成架构。
        
        Args:
            batch_size: 批次大小
        Returns:
            arch: 架构描述列表
            log_probs: 对数概率
            entropies: 熵
        """
        arch = []
        log_probs = []
        entropies = []
        
        # 初始化LSTM隐藏状态
        hidden = (
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm_hidden_size, device=next(self.parameters()).device),
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm_hidden_size, device=next(self.parameters()).device)
        )
        
        # 输入嵌入（起始token）
        inputs = torch.zeros(batch_size, 1, dtype=torch.long, device=next(self.parameters()).device)
        
        for node_idx in range(2, 2 + self.num_nodes):
            for _ in range(2):  # 每个节点有2条输入边
                # LSTM前向
                embed = self.encoder(inputs)
                output, hidden = self.lstm(embed, hidden)
                output = output.squeeze(1)
                
                # 选择前驱节点
                logits = self.decoders[(node_idx - 2) * 2](output)
                logits = logits / self.temperature
                logits = self.tanh_constant * torch.tanh(logits)
                
                probs = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                
                # 采样
                action = probs.multinomial(num_samples=1)
                selected_log_prob = log_prob.gather(1, action)
                
                arch.append(('node', action.item()))
                log_probs.append(selected_log_prob)
                entropies.append(-(log_prob * probs).sum(1, keepdim=True))
                
                # 选择操作
                logits = self.decoders[(node_idx - 2) * 2 + 1](output)
                logits = logits / self.temperature
                
                probs = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                
                action = probs.multinomial(num_samples=1)
                selected_log_prob = log_prob.gather(1, action)
                
                arch.append(('op', action.item()))
                log_probs.append(selected_log_prob)
                entropies.append(-(log_prob * probs).sum(1, keepdim=True))
                
                # 更新输入
                inputs = action
        
        # 转换为标准格式
        formatted_arch = self._format_architecture(arch)
        
        return formatted_arch, torch.cat(log_probs), torch.cat(entropies)
    
    def _format_architecture(self, raw_arch: List) -> List[Tuple[int, int]]:
        """将原始架构描述转换为标准格式。"""
        formatted = []
        i = 0
        while i < len(raw_arch):
            prev_node = raw_arch[i][1]
            op_id = raw_arch[i + 1][1]
            formatted.append((op_id, prev_node))
            i += 2
        return formatted


# ============================================================================
# 3. ENASSearcher: 权重共享超网络
# ============================================================================

class ENASCell(nn.Module):
    """ENAS风格的权重共享Cell。"""
    
    def __init__(self, steps: int, multiplier: int, C_prev_prev: int, C_prev: int, 
                 C: int, reduction: bool, reduction_prev: bool):
        super(ENASCell, self).__init__()
        
        self.reduction = reduction
        self.num_nodes = steps
        
        # 预处理层
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        
        # 所有候选操作的共享参数
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        
        for i in range(steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = self._create_op(C, stride)
                self._ops.append(op)
    
    def _create_op(self, C: int, stride: int) -> nn.Module:
        """创建候选操作集合。"""
        ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            ops.append(op)
        return ops
    
    def forward(self, s0, s1, arch: List[Tuple[int, int]]):
        """
        前向传播，根据架构选择激活的操作。
        
        Args:
            s0, s1: 输入特征
            arch: 架构描述 [(op_id, prev_node), ...]
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        offset = 0
        
        for i in range(self.num_nodes):
            # 每个节点聚合两个前驱
            node_inputs = []
            for j in range(2):
                op_id, prev_node = arch[offset + j]
                x = states[prev_node]
                op = self._ops[offset + j][op_id]
                node_inputs.append(op(x))
            
            s = sum(node_inputs)
            states.append(s)
            offset += 2
        
        return torch.cat(states[-self.num_nodes:], dim=1)


class ENASSearcher(nn.Module):
    """ENAS权重共享超网络。"""
    
    def __init__(self, C: int = 16, num_classes: int = 10, layers: int = 8, 
                 search_space: SearchSpace = None):
        super(ENASSearcher, self).__init__()
        
        self.search_space = search_space or SearchSpace()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )
        
        # 构建堆叠的Cell
        self.cells = nn.ModuleList()
        C_prev_prev, C_prev, C_curr = C, C, C
        reduction_prev = False
        
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            
            cell = ENASCell(
                steps=search_space.num_nodes if search_space else 4,
                multiplier=4,
                C_prev_prev=C_prev_prev,
                C_prev=C_prev,
                C=C_curr,
                reduction=reduction,
                reduction_prev=reduction_prev
            )
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, 4 * C_curr
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
    
    def forward(self, x, arch: List[Tuple[int, int]]):
        """
        前向传播。
        
        Args:
            x: 输入图像
            arch: 架构描述
        """
        s0 = s1 = self.stem(x)
        
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, arch)
        
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


# ============================================================================
# 4. DARTSSearcher: 可微分架构搜索
# ============================================================================

class MixedOperation(nn.Module):
    """
    DARTS的混合操作：连续松弛。
    
    将离散的操作选择松弛为所有操作的加权和：
    \bar{o}(x) = sum_i softmax(alpha_i) * o_i(x)
    """
    
    def __init__(self, C: int, stride: int):
        super(MixedOperation, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            self._ops.append(op)
    
    def forward(self, x, weights):
        """
        Args:
            x: 输入特征
            weights: 架构参数 alpha (经过softmax)
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class DARTSCell(nn.Module):
    """DARTS风格的Cell。"""
    
    def __init__(self, steps: int, multiplier: int, C_prev_prev: int, 
                 C_prev: int, C: int, reduction: bool, reduction_prev: bool):
        super(DARTSCell, self).__init__()
        
        self.reduction = reduction
        self.num_nodes = steps
        
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        
        self._steps = steps
        self._multiplier = multiplier
        
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        
        for i in range(steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOperation(C, stride)
                self._ops.append(op)
    
    def forward(self, s0, s1, weights):
        """
        Args:
            s0, s1: 输入
            weights: 架构参数 [num_edges, num_ops]
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        offset = 0
        
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) 
                   for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        
        return torch.cat(states[-self._multiplier:], dim=1)


class DARTSSearcher(nn.Module):
    """DARTS可微分架构搜索网络。"""
    
    def __init__(self, C: int = 16, num_classes: int = 10, layers: int = 8, 
                 steps: int = 4, multiplier: int = 4, 
                 search_space: SearchSpace = None):
        super(DARTSSearcher, self).__init__()
        
        self.search_space = search_space or SearchSpace(steps)
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )
        
        # 构建Cell
        self.cells = nn.ModuleList()
        C_prev_prev, C_prev, C_curr = C, C, C
        reduction_prev = False
        
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            
            cell = DARTSCell(steps, multiplier, C_prev_prev, C_prev, 
                           C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        
        # 初始化架构参数
        self._initialize_alphas()
    
    def _initialize_alphas(self):
        """初始化架构参数。"""
        num_ops = len(PRIMITIVES)
        num_edges = sum(1 for i in range(self._steps) for _ in range(2 + i))
        
        self.alphas_normal = nn.Parameter(torch.randn(num_edges, num_ops))
        self.alphas_reduce = nn.Parameter(torch.randn(num_edges, num_ops))
        
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]
    
    def arch_parameters(self):
        """返回架构参数。"""
        return self._arch_parameters
    
    def forward(self, x):
        s0 = s1 = self.stem(x)
        
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
    
    def genotype(self):
        """导出离散架构。"""
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                
                # 选择top2边
                edges = sorted(range(n), key=lambda x: max(W[x]), reverse=True)[:2]
                
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene
        
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
        
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype


# ============================================================================
# 5. OPS: 操作字典
# ============================================================================

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
}


# ============================================================================
# 6. ArchitectureEvaluator: 架构性能评估
# ============================================================================

class ArchitectureEvaluator:
    """
    架构性能评估器。
    
    支持多种评估策略：
    - 完整训练
    - 短周期评估
    - 权重继承
    """
    
    def __init__(self, 
                 train_loader,
                 val_loader,
                 device: str = 'cuda',
                 epochs: int = 50,
                 lr: float = 0.025,
                 momentum: float = 0.9,
                 weight_decay: float = 3e-4):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
    
    def evaluate(self, model: nn.Module, short_train: bool = False) -> float:
        """
        评估架构性能。
        
        Args:
            model: 待评估模型
            short_train: 是否只训练少量epoch
        Returns:
            验证集准确率
        """
        epochs = 5 if short_train else self.epochs
        
        model = model.to(self.device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            # 训练
            model.train()
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
        
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def evaluate_architecture(self, 
                            arch: List[Tuple[int, int]], 
                            search_space: SearchSpace,
                            base_model: nn.Module = None) -> float:
        """
        根据架构描述评估性能。
        
        Args:
            arch: 架构描述
            search_space: 搜索空间
            base_model: 基础模型（用于权重继承）
        Returns:
            验证集准确率
        """
        # 构建模型
        # 这里简化处理，实际应该根据arch构建具体的网络
        # ...
        pass


# ============================================================================
# 7. ZeroCostProxy: 零成本代理实现
# ============================================================================

class ZeroCostProxy:
    """
    零成本代理评估。
    
    实现NASWOT、SNIP、SynFlow等方法。
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
    
    def naswot(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor = None) -> float:
        """
        NASWOT: Neural Architecture Search Without Training.
        
        基于ReLU激活的二值化模式评估架构。
        
        Args:
            model: 待评估模型
            inputs: 输入数据
            targets: 标签（不需要，但为了接口统一）
        Returns:
            NASWOT分数
        """
        model.eval()
        model.to(self.device)
        inputs = inputs.to(self.device)
        
        # 注册hook收集激活
        activations = []
        
        def hook_fn(module, input, output):
            if isinstance(module, nn.ReLU):
                activations.append(output.detach())
        
        hooks = []
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # 前向传播
        with torch.no_grad():
            _ = model(inputs)
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        # 计算NASWOT分数
        scores = []
        for activation in activations:
            # 二值化
            binary = (activation > 0).float()
            # 扁平化
            binary = binary.view(binary.size(0), -1)
            # 计算核矩阵
            K = torch.mm(binary, binary.t())
            # 计算行列式（使用log det避免数值问题）
            try:
                score = torch.logdet(K + 1e-5 * torch.eye(K.size(0), device=K.device))
                scores.append(score.item())
            except:
                scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def snip(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        SNIP: Single-shot Network Pruning.
        
        基于参数显著性评估架构。
        
        Args:
            model: 待评估模型
            inputs: 输入数据
            targets: 标签
        Returns:
            SNIP分数
        """
        model.train()
        model.to(self.device)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # 前向+反向传播
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        
        # 计算SNIP分数
        score = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                score += torch.sum(torch.abs(param * param.grad)).item()
        
        return score
    
    def synflow(self, model: nn.Module, inputs: torch.Tensor = None, 
                targets: torch.Tensor = None, input_shape: Tuple = None) -> float:
        """
        SynFlow: Synaptic Flow.
        
        无需数据的参数显著性评估。
        
        Args:
            model: 待评估模型
            inputs: 输入数据（不需要）
            targets: 标签（不需要）
            input_shape: 输入形状
        Returns:
            SynFlow分数
        """
        model.eval()
        model.to(self.device)
        
        # 创建全1输入
        if input_shape is None:
            input_shape = (1, 3, 32, 32)
        inputs = torch.ones(input_shape, device=self.device, requires_grad=True)
        
        # 前向传播
        output = model(inputs)
        
        # 计算虚拟损失（输出的和）
        loss = torch.sum(output)
        
        # 反向传播
        loss.backward()
        
        # 计算SynFlow分数
        score = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                score += torch.sum(torch.abs(param * param.grad)).item()
        
        return score
    
    def grasp(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        GraSP: Gradient Signal Preservation.
        
        基于梯度变化的参数评估。
        
        Args:
            model: 待评估模型
            inputs: 输入数据
            targets: 标签
        Returns:
            GraSP分数
        """
        model.train()
        model.to(self.device)
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # 第一次反向传播
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        
        # 计算梯度范数
        grad_norm = 0
        for grad in grads:
            grad_norm += grad.pow(2).sum()
        
        # 第二次反向传播（Hessian近似）
        grad_norm.backward()
        
        # 计算GraSP分数
        score = 0.0
        for param in model.parameters():
            if param.grad is not None:
                score += torch.sum(torch.abs(param * param.grad)).item()
        
        return -score  # 注意GraSP使用负号
    
    def evaluate(self, 
                model: nn.Module, 
                proxy_type: str = 'naswot',
                inputs: torch.Tensor = None,
                targets: torch.Tensor = None,
                input_shape: Tuple = None) -> float:
        """
        统一的评估接口。
        
        Args:
            model: 待评估模型
            proxy_type: 代理类型 ['naswot', 'snip', 'synflow', 'grasp']
            inputs: 输入数据
            targets: 标签
            input_shape: 输入形状
        Returns:
            评估分数
        """
        if proxy_type == 'naswot':
            return self.naswot(model, inputs, targets)
        elif proxy_type == 'snip':
            return self.snip(model, inputs, targets)
        elif proxy_type == 'synflow':
            return self.synflow(model, inputs, targets, input_shape)
        elif proxy_type == 'grasp':
            return self.grasp(model, inputs, targets)
        else:
            raise ValueError(f"Unknown proxy type: {proxy_type}")


# ============================================================================
# 8. NASVisualizer: 搜索过程可视化
# ============================================================================

class NASVisualizer:
    """
    NAS搜索过程可视化工具。
    """
    
    def __init__(self):
        self.history = {
            'architectures': [],
            'rewards': [],
            'best_reward': [],
            'avg_reward': []
        }
    
    def log(self, arch, reward):
        """记录搜索历史。"""
        self.history['architectures'].append(arch)
        self.history['rewards'].append(reward)
        
        if len(self.history['best_reward']) == 0:
            self.history['best_reward'].append(reward)
        else:
            self.history['best_reward'].append(max(self.history['best_reward'][-1], reward))
        
        self.history['avg_reward'].append(np.mean(self.history['rewards'][-10:]))
    
    def plot_reward_curve(self, save_path: str = None):
        """绘制奖励曲线。"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['rewards'], alpha=0.3, label='Reward')
        plt.plot(self.history['best_reward'], label='Best Reward', linewidth=2)
        plt.plot(self.history['avg_reward'], label='Moving Average (10)', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Reward (Accuracy %)')
        plt.title('NAS Search Progress')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_cell(self, genotype: Genotype, filename: str = None):
        """
        可视化Cell结构。
        
        Args:
            genotype: Genotype命名元组
            filename: 保存路径
        """
        G = nx.DiGraph()
        
        # 添加节点
        G.add_node('c_{k-2}', layer=0)
        G.add_node('c_{k-1}', layer=0)
        
        # 解析normal cell
        offset = 0
        for i in range(4):  # 假设4个中间节点
            node_name = f'node_{i}'
            G.add_node(node_name, layer=i+1)
            
            # 添加两条边
            for j in range(2):
                if offset + j < len(genotype.normal):
                    op, prev = genotype.normal[offset + j]
                    prev_name = ['c_{k-2}', 'c_{k-1}'][prev] if prev < 2 else f'node_{prev-2}'
                    G.add_edge(prev_name, node_name, operation=op)
            
            offset += 2
        
        G.add_node('c_k', layer=5)
        
        # 添加输出连接
        for i in range(4):
            G.add_edge(f'node_{i}', 'c_k')
        
        # 绘制
        pos = {}
        for node in G.nodes():
            layer = G.nodes[node]['layer'] if 'layer' in G.nodes[node] else 0
            idx = int(node.split('_')[1]) if '_' in node else 0
            pos[node] = (layer, idx)
        
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000)
        nx.draw_networkx_labels(G, pos)
        
        edge_labels = {(u, v): d['operation'] for u, v, d in G.edges(data=True) if 'operation' in d}
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20)
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        plt.title('Cell Architecture Visualization')
        plt.axis('off')
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_architecture_heatmap(self, search_space: SearchSpace, 
                                 arch_probs: np.ndarray = None):
        """
        绘制架构选择热图。
        
        Args:
            search_space: 搜索空间
            arch_probs: 架构选择概率
        """
        if arch_probs is None:
            # 随机生成示例
            num_edges = search_space.get_num_edges()
            arch_probs = np.random.rand(num_edges, search_space.num_ops)
            arch_probs = arch_probs / arch_probs.sum(axis=1, keepdims=True)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(arch_probs, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Selection Probability')
        plt.xlabel('Operation ID')
        plt.ylabel('Edge ID')
        plt.title('Architecture Selection Heatmap')
        plt.xticks(range(search_space.num_ops), search_space.primitives, rotation=45)
        plt.tight_layout()
        plt.show()


# ============================================================================
# 9. 训练工具函数
# ============================================================================

def train_darts(model: DARTSSearcher,
                train_loader,
                val_loader,
                epochs: int = 50,
                arch_lr: float = 3e-4,
                weight_lr: float = 0.025,
                device: str = 'cuda'):
    """
    训练DARTS模型。
    
    Args:
        model: DARTS搜索网络
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        arch_lr: 架构参数学习率
        weight_lr: 网络权重学习率
        device: 设备
    """
    model = model.to(device)
    
    # 优化器
    optimizer_weight = torch.optim.SGD(
        model.parameters(),
        lr=weight_lr,
        momentum=0.9,
        weight_decay=3e-4
    )
    optimizer_arch = torch.optim.Adam(
        model.arch_parameters(),
        lr=arch_lr,
        betas=(0.5, 0.999),
        weight_decay=1e-3
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_weight, epochs)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        
        for step in range(len(train_loader)):
            # 更新架构参数
            try:
                val_inputs, val_targets = next(val_iter)
            except:
                val_iter = iter(val_loader)
                val_inputs, val_targets = next(val_iter)
            
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            
            optimizer_arch.zero_grad()
            val_outputs = model(val_inputs)
            arch_loss = criterion(val_outputs, val_targets)
            arch_loss.backward()
            optimizer_arch.step()
            
            # 更新网络权重
            try:
                train_inputs, train_targets = next(train_iter)
            except:
                break
            
            train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)
            
            optimizer_weight.zero_grad()
            train_outputs = model(train_inputs)
            weight_loss = criterion(train_outputs, train_targets)
            weight_loss.backward()
            optimizer_weight.step()
        
        scheduler.step()
        
        # 验证
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            acc = 100. * correct / total
            print(f'Epoch [{epoch+1}/{epochs}], Val Acc: {acc:.2f}%')
            print(f'Genotype: {model.genotype()}')


# ============================================================================
# 10. 主函数示例
# ============================================================================

def demo_search_space():
    """演示搜索空间。"""
    print("=" * 60)
    print("Demo: SearchSpace")
    print("=" * 60)
    
    search_space = SearchSpace(num_nodes=4, num_ops=8)
    print(f"Number of nodes: {search_space.num_nodes}")
    print(f"Number of operations: {search_space.num_ops}")
    print(f"Number of edges: {search_space.get_num_edges()}")
    print(f"Primitives: {search_space.primitives}")
    
    # 随机采样架构
    arch = search_space.sample_random_architecture()
    print(f"\nRandom architecture: {arch}")
    
    # 解码为Genotype
    genotype = search_space.decode_to_genotype(arch)
    print(f"\nGenotype: {genotype}")


def demo_zero_cost_proxy():
    """演示零成本代理。"""
    print("\n" + "=" * 60)
    print("Demo: ZeroCostProxy")
    print("=" * 60)
    
    # 创建简单模型
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10)
    )
    
    # 创建代理评估器
    proxy = ZeroCostProxy(device='cpu')
    
    # 创建随机输入
    inputs = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))
    
    # 评估
    print("\nEvaluating with different proxies:")
    
    naswot_score = proxy.naswot(model, inputs)
    print(f"NASWOT score: {naswot_score:.4f}")
    
    snip_score = proxy.snip(model, inputs, targets)
    print(f"SNIP score: {snip_score:.4f}")
    
    synflow_score = proxy.synflow(model, input_shape=(1, 3, 32, 32))
    print(f"SynFlow score: {synflow_score:.4f}")


def demo_visualization():
    """演示可视化。"""
    print("\n" + "=" * 60)
    print("Demo: NASVisualizer")
    print("=" * 60)
    
    visualizer = NASVisualizer()
    
    # 模拟搜索历史
    for i in range(50):
        arch = [(i % 8, i % 4) for _ in range(8)]
        reward = 50 + i * 0.5 + np.random.randn() * 5
        visualizer.log(arch, reward)
    
    print(f"Logged {len(visualizer.history['rewards'])} architectures")
    print(f"Best reward: {max(visualizer.history['rewards']):.2f}")
    
    # 绘制奖励曲线
    # visualizer.plot_reward_curve()


if __name__ == '__main__':
    # 运行演示
    demo_search_space()
    demo_zero_cost_proxy()
    demo_visualization()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
