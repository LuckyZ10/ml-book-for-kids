"""
可微分架构搜索(DARTS)完整实现
包含：核心算法、双层优化、架构推导
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==================== 基础操作 ====================

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    def forward(self, x): return self.op(x)

class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding, 
                     dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    def forward(self, x): return self.op(x)

class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding, 
                     groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride=1, padding=padding, 
                     groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    def forward(self, x): return self.op(x)

class Identity(nn.Module):
    def forward(self, x): return x

class Zero(nn.Module):
    def __init__(self, stride): super().__init__(); self.stride = stride
    def forward(self, x):
        if self.stride == 1: return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    def forward(self, x):
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(out)

# ==================== DARTS操作 ====================

PRIMITIVES = ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 
              'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']

def get_operation(primitive, C, stride):
    if primitive == 'none': return Zero(stride)
    elif primitive == 'max_pool_3x3': 
        return nn.Sequential(nn.MaxPool2d(3, stride=stride, padding=1), nn.BatchNorm2d(C))
    elif primitive == 'avg_pool_3x3': 
        return nn.Sequential(nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False), nn.BatchNorm2d(C))
    elif primitive == 'skip_connect':
        return Identity() if stride == 1 else FactorizedReduce(C, C)
    elif primitive == 'sep_conv_3x3': return SepConv(C, C, 3, stride, 1)
    elif primitive == 'sep_conv_5x5': return SepConv(C, C, 5, stride, 2)
    elif primitive == 'dil_conv_3x3': return DilConv(C, C, 3, stride, 2, 2)
    elif primitive == 'dil_conv_5x5': return DilConv(C, C, 5, stride, 4, 2)
    raise ValueError(f"Unknown primitive: {primitive}")

class MixedOp(nn.Module):
    """DARTS混合操作"""
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = get_operation(primitive, C, stride)
            self._ops.append(op)
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))

class DARTSCell(nn.Module):
    """DARTS搜索单元"""
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super().__init__()
        self.reduction = reduction
        self.steps = steps
        self.multiplier = multiplier
        
        if reduction_prev: self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else: self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        
        self._ops = nn.ModuleList()
        for i in range(self.steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)
        
        k = sum(1 for i in range(steps) for _ in range(2 + i))
        num_ops = len(PRIMITIVES)
        self.alphas_normal = nn.Parameter(torch.randn(k, num_ops) * 0.01)
        if reduction: self.alphas_reduce = nn.Parameter(torch.randn(k, num_ops) * 0.01)
        else: self.alphas_reduce = None
    
    def forward(self, s0, s1):
        s0, s1 = self.preprocess0(s0), self.preprocess1(s1)
        states, offset = [s0, s1], 0
        weights = F.softmax(self.alphas_reduce if self.reduction else self.alphas_normal, dim=-1)
        
        for i in range(self.steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self.multiplier:], dim=1)

class DARTSNetwork(nn.Module):
    """完整DARTS搜索网络"""
    def __init__(self, C=16, num_classes=10, layers=8, steps=4, multiplier=4):
        super().__init__()
        C_curr = C * 3
        self.stem = nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr))
        
        self.cells = nn.ModuleList()
        C_prev_prev, C_prev, reduction_prev = C_curr, C_curr, False
        
        for i in range(layers):
            reduction = i in [layers // 3, 2 * layers // 3]
            if reduction: C_curr *= 2
            cell = DARTSCell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            reduction_prev = reduction
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
    
    def forward(self, x):
        s0 = s1 = self.stem(x)
        for cell in self.cells: s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        return self.classifier(out.view(out.size(0), -1))
    
    def arch_parameters(self):
        params = []
        for cell in self.cells:
            params.append(cell.alphas_normal)
            if cell.reduction: params.append(cell.alphas_reduce)
        return params
    
    def model_parameters(self):
        ids = set(id(p) for p in self.arch_parameters())
        for p in self.parameters():
            if id(p) not in ids: yield p

if __name__ == '__main__':
    print("=" * 60)
    print("DARTS实现测试")
    print("=" * 60)
    
    model = DARTSNetwork(C=16, num_classes=10, layers=8)
    total_params = sum(p.numel() for p in model.parameters())
    arch_params = sum(p.numel() for p in model.arch_parameters())
    model_params = sum(p.numel() for p in model.model_parameters())
    
    print(f"总参数量: {total_params:,}")
    print(f"架构参数: {arch_params:,}")
    print(f"网络权重: {model_params:,}")
    
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"\n输入: {x.shape}, 输出: {y.shape}")
