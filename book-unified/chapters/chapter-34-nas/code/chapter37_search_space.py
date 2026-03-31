"""
NAS搜索空间完整实现
包含：全局搜索、单元搜索、层次化搜索
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from collections import namedtuple

# ==================== 基础操作定义 ====================

class Operation(nn.Module):
    """NAS搜索空间中的基础操作"""
    
    def __init__(self, C, stride=1):
        super().__init__()
        self.C = C
        self.stride = stride
    
    def forward(self, x):
        raise NotImplementedError


class ReLUConvBN(Operation):
    """ReLU -> Conv -> BN"""
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__(C_out, stride)
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, 
                     padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    
    def forward(self, x):
        return self.op(x)


class SepConv(Operation):
    """可分离卷积"""
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__(C_out, stride)
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride, 
                     padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=1, 
                     padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    
    def forward(self, x):
        return self.op(x)


class Identity(Operation):
    """恒等映射"""
    
    def __init__(self):
        super().__init__(0, 1)
    
    def forward(self, x):
        return x


class Zero(Operation):
    """零操作（断开连接）"""
    
    def __init__(self, stride):
        super().__init__(0, stride)
        self.stride = stride
    
    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class AvgPool(Operation):
    """平均池化"""
    
    def __init__(self, kernel_size, stride):
        super().__init__(0, stride)
        self.op = nn.AvgPool2d(kernel_size, stride=stride, 
                               padding=kernel_size//2, count_include_pad=False)
    
    def forward(self, x):
        return self.op(x)


class MaxPool(Operation):
    """最大池化"""
    
    def __init__(self, kernel_size, stride):
        super().__init__(0, stride)
        self.op = nn.MaxPool2d(kernel_size, stride=stride, 
                               padding=kernel_size//2)
    
    def forward(self, x):
        return self.op(x)


# 操作工厂函数
def get_operation(op_name, C, stride, affine=True):
    """根据操作名创建操作实例"""
    
    ops_dict = {
        'none': lambda: Zero(stride),
        'avg_pool_3x3': lambda: AvgPool(3, stride),
        'max_pool_3x3': lambda: MaxPool(3, stride),
        'skip_connect': lambda: Identity() if stride == 1 else 
                                 nn.Sequential(
                                     nn.AvgPool2d(kernel_size=stride, stride=stride),
                                     nn.Conv2d(C, C, 1, bias=False)
                                 ),
        'sep_conv_3x3': lambda: SepConv(C, C, 3, stride, 1, affine),
        'sep_conv_5x5': lambda: SepConv(C, C, 5, stride, 2, affine),
        'sep_conv_7x7': lambda: SepConv(C, C, 7, stride, 3, affine),
        'dil_conv_3x3': lambda: DilConv(C, C, 3, stride, 2, 2, affine),
        'dil_conv_5x5': lambda: DilConv(C, C, 5, stride, 4, 2, affine),
        'conv_7x1_1x7': lambda: nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
            nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(C, affine=affine)
        ),
    }
    
    return ops_dict[op_name]()


class DilConv(Operation):
    """空洞卷积"""
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__(C_out, stride)
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride, 
                     padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    
    def forward(self, x):
        return self.op(x)


# ==================== 单元搜索空间 ====================

class NASCell(nn.Module):
    """
    NASNet/DARTS风格的搜索单元
    
    结构：
    - 输入：两个前驱节点的输出
    - 内部：B个块，每个块2个操作
    - 输出：所有未使用中间节点的拼接
    """
    
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, 
                 reduction, reduction_prev, ops_list=None):
        """
        Args:
            steps: 块的数量（中间节点数）
            multiplier: 输出通道扩展倍数
            C_prev_prev: 前前个单元的通道数
            C_prev: 前一个单元的通道数
            C: 当前单元通道数
            reduction: 是否降采样
            reduction_prev: 前一个单元是否降采样
            ops_list: 预设的操作列表（用于固定架构）
        """
        super().__init__()
        
        self.reduction = reduction
        self.reduction_prev = reduction_prev
        self.steps = steps
        self.multiplier = multiplier
        
        # 预处理输入
        if reduction_prev:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        
        # 构建搜索空间
        self._ops = nn.ModuleList()
        self._bilinear = reduction
        
        for i in range(self.steps):
            for j in range(2 + i):  # 每个新节点可以连接所有前面节点
                stride = 2 if reduction and j < 2 else 1
                
                if ops_list is None:
                    # 搜索模式：所有操作都可选
                    op = MixedOp(C, stride)
                else:
                    # 固定模式：使用给定的操作
                    op = get_operation(ops_list[i][j], C, stride)
                
                self._ops.append(op)
    
    def forward(self, s0, s1, weights=None):
        """
        前向传播
        
        Args:
            s0, s1: 两个输入
            weights: 架构权重（DARTS模式）
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        offset = 0
        
        for i in range(self.steps):
            # 聚合前驱节点的贡献
            if weights is None:
                # 固定架构模式
                s = sum(self._ops[offset + j](h) 
                       for j, h in enumerate(states))
            else:
                # DARTS模式：加权组合
                s = sum(w * op(h) for w, op, h in 
                       zip(weights[offset:offset + len(states)], 
                           self._ops[offset:offset + len(states)], 
                           states))
            
            offset += len(states)
            states.append(s)
        
        # 输出：最后multiplier个节点的拼接
        return torch.cat(states[-self.multiplier:], dim=1)


class MixedOp(nn.Module):
    """
    DARTS中的混合操作
    将离散的操作选择松弛为连续的加权和
    """
    
    def __init__(self, C, stride):
        super().__init__()
        
        # 标准DARTS操作集合
        self._primitives = [
            'none',
            'max_pool_3x3',
            'avg_pool_3x3',
            'skip_connect',
            'sep_conv_3x3',
            'sep_conv_5x5',
            'dil_conv_3x3',
            'dil_conv_5x5',
        ]
        
        # 所有候选操作
        self._ops = nn.ModuleList()
        for primitive in self._primitives:
            op = get_operation(primitive, C, stride, affine=False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)
    
    def forward(self, x, weights):
        """
        Args:
            x: 输入特征
            weights: 每个操作的权重（softmax后的）
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))


# ==================== 全局搜索空间 ====================

class MacroSearchSpace:
    """
    宏观搜索空间
    搜索整个网络的拓扑结构
    """
    
    def __init__(self, num_layers=8, num_classes=10):
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # 每层可选的操作
        self.layer_ops = ['conv3x3', 'conv5x5', 'sep_conv3x3', 
                         'sep_conv5x5', 'max_pool', 'avg_pool', 'identity']
        
        # 每层可选的通道数
        self.channels = [16, 32, 64, 128, 256]
        
        # 连接方式
        self.connections = ['sequential', 'skip', 'dense']
    
    def sample_random_architecture(self):
        """随机采样一个架构"""
        arch = []
        for i in range(self.num_layers):
            layer = {
                'operation': random.choice(self.layer_ops),
                'channels': random.choice(self.channels),
                'connection': random.choice(self.connections),
                'kernel_size': random.choice([3, 5, 7]),
            }
            arch.append(layer)
        return arch
    
    def architecture_to_network(self, arch):
        """将架构编码转换为实际网络"""
        layers = []
        in_channels = 3
        
        for i, layer_spec in enumerate(arch):
            op_type = layer_spec['operation']
            out_channels = layer_spec['channels']
            kernel = layer_spec['kernel_size']
            
            if op_type == 'conv3x3':
                layers.append(nn.Conv2d(in_channels, out_channels, 
                                       kernel_size=3, padding=1))
            elif op_type == 'conv5x5':
                layers.append(nn.Conv2d(in_channels, out_channels, 
                                       kernel_size=5, padding=2))
            elif op_type == 'sep_conv3x3':
                layers.append(SepConv(in_channels, out_channels, 3, 1, 1))
            
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            
            in_channels = out_channels
        
        return nn.Sequential(*layers)


# ==================== 层次化搜索空间 ====================

class HierarchicalSearchSpace:
    """
    层次化搜索空间
    在不同粒度上进行搜索
    """
    
    def __init__(self):
        # 第一层：网络级别（深度、宽度模式）
        self.network_level = {
            'num_cells': [4, 8, 12, 16],
            'width_multipliers': [0.5, 0.75, 1.0, 1.25],
            'input_size': [224, 192, 160, 128],
        }
        
        # 第二层：单元级别（Normal/Reduction单元结构）
        primitives = [
            'none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect',
            'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'
        ]
        self.cell_level = {
            'num_nodes': [3, 4, 5],
            'operations': primitives,
        }
        
        # 第三层：操作级别（卷积核大小、膨胀率等）
        self.op_level = {
            'kernel_sizes': [3, 5, 7],
            'dilation_rates': [1, 2, 3],
            'group_sizes': [1, 2, 4],
        }
    
    def sample_hierarchical_architecture(self):
        """采样层次化架构"""
        arch = {
            'network': {
                'num_cells': random.choice(self.network_level['num_cells']),
                'width_mult': random.choice(self.network_level['width_multipliers']),
                'input_size': random.choice(self.network_level['input_size']),
            },
            'normal_cell': self._sample_cell(),
            'reduction_cell': self._sample_cell(),
        }
        return arch
    
    def _sample_cell(self):
        """采样一个单元结构"""
        num_nodes = random.choice(self.cell_level['num_nodes'])
        
        cell = {'num_nodes': num_nodes, 'edges': []}
        
        # 为每个节点采样连接和操作
        for i in range(num_nodes):
            for j in range(i + 2):  # 可以连接前面所有节点
                edge = {
                    'from': j,
                    'to': i + 2,
                    'op': random.choice(self.cell_level['operations'])
                }
                cell['edges'].append(edge)
        
        return cell


Architecture = namedtuple('Architecture', 'normal_cell reduction_cell')


# ==================== 搜索空间工具函数 ====================

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input_size=(1, 3, 224, 224)):
    """
    估算模型FLOPs（简化版）
    注意：这是近似计算，真实值需要更精确的profile
    """
    # 使用hook统计
    flops = [0]
    
    def conv_hook(module, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * \
                     (input_channels / module.groups)
        output_size = output_height * output_width * output_channels
        flops[0] += kernel_ops * output_size * batch_size
    
    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
    
    # 前向传播一次
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    model(dummy_input)
    
    # 移除hooks
    for h in hooks:
        h.remove()
    
    return flops[0]


if __name__ == '__main__':
    # 测试搜索空间
    print("=" * 60)
    print("NAS搜索空间测试")
    print("=" * 60)
    
    # 1. 测试单元搜索空间
    print("\n1. 单元搜索空间测试")
    cell = NASCell(steps=4, multiplier=4, C_prev_prev=48, C_prev=64, 
                   C=64, reduction=False, reduction_prev=False)
    print(f"Cell parameters: {count_parameters(cell):,}")
    
    x0 = torch.randn(2, 48, 32, 32)
    x1 = torch.randn(2, 64, 32, 32)
    out = cell(x0, x1)
    print(f"Input shapes: {x0.shape}, {x1.shape}")
    print(f"Output shape: {out.shape}")
    
    # 2. 测试宏观搜索空间
    print("\n2. 宏观搜索空间测试")
    macro_space = MacroSearchSpace(num_layers=5)
    arch = macro_space.sample_random_architecture()
    print(f"Sampled architecture with {len(arch)} layers")
    
    # 3. 测试层次化搜索空间
    print("\n3. 层次化搜索空间测试")
    hier_space = HierarchicalSearchSpace()
    hier_arch = hier_space.sample_hierarchical_architecture()
    print(f"Network config: {hier_arch['network']}")
    print(f"Normal cell has {hier_arch['normal_cell']['num_nodes']} nodes")
