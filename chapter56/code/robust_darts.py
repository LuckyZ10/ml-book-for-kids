"""
RobustDARTS: 解决性能崩溃的改进版DARTS
=====================================

本章实现针对DARTS性能崩溃问题的改进算法。

核心改进:
- 早停机制：当架构熵过低时触发早停
- 正则化项：避免架构权重过于极端
- 混合训练策略：交替优化架构和网络权重
- 跳过连接惩罚：避免过度依赖跳过连接

费曼法比喻：
走钢丝时，DARTS像柔软的绳索，但如果不加控制，
绳索会变得像橡皮筋一样被拉向一个方向。
RobustDARTS就像给绳索加上张力限制器，
保持柔软的同时避免过度拉伸。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import copy


class MixedOperation(nn.Module):
    """
    混合操作：学习不同操作的权重
    
    想象成一个工具箱，里面有锤子、螺丝刀、扳手。
    DARTS不选择用哪个，而是学习每个工具的"适用度"，
    最后根据适用度加权组合使用。
    """
    
    PRIMITIVES = [
        'none',         # 空操作
        'skip_connect', # 跳过连接
        'conv_3x3',     # 3x3卷积
        'conv_5x5',     # 5x5卷积
        'dil_conv_3x3', # 3x3空洞卷积
        'sep_conv_3x3', # 3x3可分离卷积
    ]
    
    def __init__(self, C: int, stride: int):
        super().__init__()
        self.C = C
        self.stride = stride
        self.ops = nn.ModuleList()
        
        for primitive in self.PRIMITIVES:
            op = self._create_op(primitive, C, stride)
            self.ops.append(op)
        
        # 架构参数（可学习）
        self.register_parameter(
            'alphas', 
            nn.Parameter(torch.randn(len(self.PRIMITIVES)))
        )
    
    def _create_op(self, primitive: str, C: int, stride: int) -> nn.Module:
        """创建具体操作"""
        if primitive == 'none':
            return Zero(C, stride)
        elif primitive == 'skip_connect':
            return Identity() if stride == 1 else FactorizedReduce(C, C)
        elif primitive == 'conv_3x3':
            return ReLUConvBN(C, C, 3, stride, 1)
        elif primitive == 'conv_5x5':
            return ReLUConvBN(C, C, 5, stride, 2)
        elif primitive == 'dil_conv_3x3':
            return DilConv(C, C, 3, stride, 2, 2)
        elif primitive == 'sep_conv_3x3':
            return SepConv(C, C, 3, stride, 1)
        else:
            raise ValueError(f"Unknown primitive: {primitive}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：加权组合所有操作"""
        # 使用softmax获得概率分布
        weights = F.softmax(self.alphas, dim=0)
        
        # 加权求和
        output = sum(w * op(x) for w, op in zip(weights, self.ops))
        return output
    
    def get_entropy(self) -> torch.Tensor:
        """计算架构分布的熵（衡量多样性）"""
        weights = F.softmax(self.alphas, dim=0)
        # 熵 = -sum(p * log(p))
        entropy = -(weights * torch.log(weights + 1e-8)).sum()
        return entropy
    
    def get_skip_probability(self) -> torch.Tensor:
        """获取跳过连接的概率"""
        weights = F.softmax(self.alphas, dim=0)
        skip_idx = self.PRIMITIVES.index('skip_connect')
        return weights[skip_idx]
    
    def genotype(self) -> str:
        """获取离散化后的架构基因型"""
        weights = F.softmax(self.alphas, dim=0)
        best_idx = weights.argmax().item()
        return self.PRIMITIVES[best_idx]


class Zero(nn.Module):
    """零操作"""
    def __init__(self, C: int, stride: int):
        super().__init__()
        self.stride = stride
        self.C = C
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return torch.zeros_like(x)
        else:
            return torch.zeros(
                x.size(0), self.C, 
                x.size(2) // self.stride, 
                x.size(3) // self.stride,
                device=x.device
            )


class Identity(nn.Module):
    """恒等映射"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class FactorizedReduce(nn.Module):
    """下采样恒等映射"""
    def __init__(self, C_in: int, C_out: int):
        super().__init__()
        assert C_out % 2 == 0
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(out)


class ReLUConvBN(nn.Module):
    """ReLU + Conv + BN"""
    def __init__(self, C_in: int, C_out: int, kernel_size: int, 
                 stride: int, padding: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class DilConv(nn.Module):
    """空洞卷积"""
    def __init__(self, C_in: int, C_out: int, kernel_size: int,
                 stride: int, padding: int, dilation: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, 
                     dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_out),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class SepConv(nn.Module):
    """可分离卷积"""
    def __init__(self, C_in: int, C_out: int, kernel_size: int,
                 stride: int, padding: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, 
                     groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_in),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, 1, padding, 
                     groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, 1, padding=0, bias=False),
            nn.BatchNorm2d(C_out),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class RobustDARTSCell(nn.Module):
    """
    RobustDARTS搜索单元
    
    就像乐高积木盒，每次可以从盒子里选两块积木连接。
    但有规则：
    1. 每个积木块最多接收2个连接
    2. 总共选择4个中间节点
    3. 最后的输出连接所有中间节点
    """
    
    def __init__(self, steps: int, multiplier: int, C: int, 
                 reduction: bool, reduction_prev: bool):
        super().__init__()
        self.steps = steps  # 中间节点数
        self.multiplier = multiplier
        self.reduction = reduction
        
        # 输入预处理
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C, C)
        else:
            self.preprocess0 = ReLUConvBN(C, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C, C, 1, 1, 0)
        
        # 混合操作
        self.edges = nn.ModuleDict()
        for i in range(2):  # 来自前两个节点的输入
            for j in range(self.steps):
                stride = 2 if reduction and i < 2 else 1
                key = f"edge_{i}_{j}"
                self.edges[key] = MixedOperation(C, stride)
        
        for i in range(2, 2 + self.steps - 1):  # 来自中间节点的输入
            for j in range(i - 1, self.steps):
                stride = 2 if reduction else 1
                key = f"edge_{i}_{j}"
                self.edges[key] = MixedOperation(C, stride)
    
    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        
        # 构建中间节点
        for j in range(self.steps):
            # 收集所有指向节点j的边
            edges_input = []
            for i in range(len(states)):
                key = f"edge_{i}_{j}"
                if key in self.edges:
                    edges_input.append(self.edges[key](states[i]))
            
            # 累加（每个节点接收2个输入）
            if len(edges_input) >= 2:
                state = sum(edges_input[:2])
            else:
                state = edges_input[0] if edges_input else states[0]
            states.append(state)
        
        # 输出 = 最后4个中间节点的拼接
        return torch.cat(states[-self.multiplier:], dim=1)
    
    def get_entropy(self) -> torch.Tensor:
        """获取单元内所有边的平均熵"""
        entropies = [edge.get_entropy() for edge in self.edges.values()]
        return torch.stack(entropies).mean()
    
    def get_skip_ratio(self) -> float:
        """获取跳过连接的比例"""
        skip_probs = [edge.get_skip_probability().item() 
                     for edge in self.edges.values()]
        return np.mean(skip_probs)


class RobustDARTSNetwork(nn.Module):
    """
    完整的RobustDARTS网络
    
    就像设计一座智能大桥：
    - 有多个桥墩（网络层）
    - 每个桥墩有不同的结构
    - 目标是让车（数据）能最快通过
    - RobustDARTS确保桥不会太"偷懒"（只用跳过连接）
    """
    
    def __init__(self, C: int = 16, num_classes: int = 10, 
                 layers: int = 8, steps: int = 4, 
                 multiplier: int = 4):
        super().__init__()
        self.C = C
        self.num_classes = num_classes
        self.layers = layers
        self.steps = steps
        
        # 初始卷积
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )
        
        # 构建网络层
        C_prev_prev, C_prev, C_curr = C, C, C
        reduction_prev = False
        self.cells = nn.ModuleList()
        
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            
            cell = RobustDARTSCell(steps, multiplier, C_curr, 
                                  reduction, reduction_prev)
            self.cells.append(cell)
            
            C_prev_prev = C_prev
            C_prev = multiplier * C_curr
            reduction_prev = reduction
        
        # 分类头
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        
        # 记录训练状态
        self.skip_penalty_weight = 0.1
        self.entropy_threshold = 0.5
        self.early_stop_triggered = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        s0 = s1 = self.stem(x)
        
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
    
    def get_skip_penalty(self) -> torch.Tensor:
        """获取跳过连接惩罚项"""
        skip_ratios = [cell.get_skip_ratio() for cell in self.cells]
        avg_skip = np.mean(skip_ratios)
        # 鼓励多样化，惩罚过高的跳过连接比例
        penalty = F.relu(avg_skip - 0.3) ** 2
        return torch.tensor(penalty, requires_grad=True)
    
    def get_architecture_entropy(self) -> torch.Tensor:
        """获取整体架构熵"""
        entropies = [cell.get_entropy() for cell in self.cells]
        return torch.stack(entropies).mean()
    
    def check_early_stop(self) -> bool:
        """检查是否应该早停"""
        entropy = self.get_architecture_entropy().item()
        skip_ratio = np.mean([cell.get_skip_ratio() for cell in self.cells])
        
        # 如果熵太低或跳过连接比例过高，触发早停
        if entropy < self.entropy_threshold or skip_ratio > 0.5:
            self.early_stop_triggered = True
            return True
        return False
    
    def get_arch_parameters(self) -> List[nn.Parameter]:
        """获取架构参数"""
        params = []
        for cell in self.cells:
            for edge in cell.edges.values():
                params.append(edge.alphas)
        return params
    
    def get_model_parameters(self) -> List[nn.Parameter]:
        """获取模型参数（不包括架构参数）"""
        params = []
        for name, param in self.named_parameters():
            if 'alphas' not in name:
                params.append(param)
        return params


class RobustDARTSTrainer:
    """
    RobustDARTS训练器
    
    核心改进：
    1. 交替训练：先训练网络权重，再训练架构
    2. 早停机制：监控架构熵和跳过连接比例
    3. 正则化：避免架构参数过于极端
    4. 混合训练：部分epoch使用完整验证集
    """
    
    def __init__(self, model: RobustDARTSNetwork, 
                 train_loader, val_loader,
                 device: str = 'cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 优化器
        self.w_optimizer = torch.optim.SGD(
            model.get_model_parameters(),
            lr=0.025, momentum=0.9, weight_decay=3e-4
        )
        self.alpha_optimizer = torch.optim.Adam(
            model.get_arch_parameters(),
            lr=3e-4, betas=(0.5, 0.999), weight_decay=1e-3
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.epoch = 0
        
        # 早停相关
        self.entropy_history = []
        self.skip_ratio_history = []
    
    def train_epoch(self, alternate: bool = True) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        if alternate:
            # 交替训练：第一步优化权重
            self._train_weights()
            # 第二步优化架构
            self._train_architecture()
        else:
            # 联合训练
            self._train_joint()
        
        # 记录指标
        entropy = self.model.get_architecture_entropy().item()
        skip_ratio = np.mean([cell.get_skip_ratio() 
                             for cell in self.model.cells])
        
        self.entropy_history.append(entropy)
        self.skip_ratio_history.append(skip_ratio)
        
        self.epoch += 1
        
        return {
            'epoch': self.epoch,
            'entropy': entropy,
            'skip_ratio': skip_ratio,
            'early_stop': self.model.check_early_stop()
        }
    
    def _train_weights(self):
        """训练网络权重"""
        self.model.train()
        for step, (inputs, targets) in enumerate(self.train_loader):
            if step >= 50:  # 限制步数以加快搜索
                break
            
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.w_optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.w_optimizer.step()
    
    def _train_architecture(self):
        """训练架构参数"""
        self.model.train()
        
        # 从验证集采样
        val_iter = iter(self.val_loader)
        
        for step in range(50):
            try:
                inputs, targets = next(val_iter)
            except StopIteration:
                val_iter = iter(self.val_loader)
                inputs, targets = next(val_iter)
            
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.alpha_optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 添加跳过连接惩罚
            skip_penalty = self.model.get_skip_penalty().to(self.device)
            loss += self.model.skip_penalty_weight * skip_penalty
            
            loss.backward()
            self.alpha_optimizer.step()
    
    def _train_joint(self):
        """联合训练权重和架构"""
        self.model.train()
        
        for step, (train_inputs, train_targets) in enumerate(self.train_loader):
            if step >= 50:
                break
            
            train_inputs = train_inputs.to(self.device)
            train_targets = train_targets.to(self.device)
            
            # 优化权重
            self.w_optimizer.zero_grad()
            outputs = self.model(train_inputs)
            loss_w = self.criterion(outputs, train_targets)
            loss_w.backward()
            self.w_optimizer.step()
            
            # 从验证集采样优化架构
            try:
                val_inputs, val_targets = next(iter(self.val_loader))
            except:
                continue
            
            val_inputs = val_inputs.to(self.device)
            val_targets = val_targets.to(self.device)
            
            self.alpha_optimizer.zero_grad()
            outputs = self.model(val_inputs)
            loss_alpha = self.criterion(outputs, val_targets)
            
            # 添加正则化
            skip_penalty = self.model.get_skip_penalty().to(self.device)
            loss_alpha += self.model.skip_penalty_weight * skip_penalty
            
            loss_alpha.backward()
            self.alpha_optimizer.step()
    
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        return {
            'accuracy': accuracy,
            'entropy': self.model.get_architecture_entropy().item(),
            'skip_ratio': np.mean([cell.get_skip_ratio() 
                                  for cell in self.model.cells])
        }
    
    def get_genotype(self) -> Dict:
        """获取最终架构基因型"""
        genotype = {}
        for cell_idx, cell in enumerate(self.model.cells):
            cell_genes = {}
            for key, edge in cell.edges.items():
                cell_genes[key] = edge.genotype()
            genotype[f'cell_{cell_idx}'] = cell_genes
        return genotype


def demo_robust_darts():
    """
    RobustDARTS演示
    
    小测试：运行这个演示，观察：
    1. 架构熵如何变化？
    2. 跳过连接比例是否被有效控制？
    3. 早停机制何时触发？
    """
    print("=" * 60)
    print("RobustDARTS 演示")
    print("=" * 60)
    
    # 创建模型
    model = RobustDARTSNetwork(C=16, num_classes=10, layers=4)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    arch_params = sum(p.numel() for p in model.get_arch_parameters())
    model_params = sum(p.numel() for p in model.get_model_parameters())
    
    print(f"\n模型参数统计:")
    print(f"  架构参数: {arch_params:,}")
    print(f"  网络权重: {model_params:,}")
    print(f"  总计: {total_params:,}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {logits.shape}")
    
    # 初始架构状态
    initial_entropy = model.get_architecture_entropy().item()
    initial_skip_ratio = np.mean([cell.get_skip_ratio() 
                                  for cell in model.cells])
    
    print(f"\n初始架构状态:")
    print(f"  架构熵: {initial_entropy:.4f}")
    print(f"  跳过连接比例: {initial_skip_ratio:.4f}")
    
    # 展示基因型
    print(f"\n初始基因型示例 (Cell 0):")
    if model.cells:
        for key, edge in list(model.cells[0].edges.items())[:3]:
            weights = F.softmax(edge.alphas, dim=0)
            best_op = edge.PRIMITIVES[weights.argmax().item()]
            print(f"  {key}: {best_op} (置信度: {weights.max().item():.3f})")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    demo_robust_darts()
