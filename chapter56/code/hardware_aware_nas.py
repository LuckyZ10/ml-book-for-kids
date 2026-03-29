"""
HardwareAwareNAS: 硬件感知神经架构搜索
=====================================

在实际部署中，模型需要在特定硬件上高效运行。
本模块实现硬件感知的NAS，同时优化准确率和硬件效率。

核心方法：
1. FBNet: 基于延迟预测的可微分搜索
2. OFA: Once-for-All网络，支持多种配置
3. 多目标优化：准确率 vs 延迟的Pareto前沿

费曼法比喻：
普通NAS就像设计一双通用跑鞋，不考虑场地。
硬件感知NAS就像为特定场地定制跑鞋：
- 塑胶跑道 → 轻质跑鞋
- 越野山地 → 防滑高帮鞋
- 雪地冰面 → 带钉冰鞋

同样的脚（架构），不同的鞋（硬件优化）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class HardwareConstraint:
    """硬件约束配置"""
    target_latency: float = 10.0  # ms
    target_memory: float = 100.0   # MB
    target_power: float = 5.0      # W
    
    # 硬件特性
    compute_capability: float = 1.0  # 相对计算能力
    memory_bandwidth: float = 100.0   # GB/s


class SuperNet(nn.Module):
    """
    超级网络（支持多种子网络的超网）
    
    就像一个乐高积木盒，可以组装出各种形态。
    训练时同时学习所有可能的子网络，
    搜索时从中选择最优的子集。
    """
    
    def __init__(self, 
                 num_classes: int = 10,
                 base_channels: int = 16,
                 depth_list: List[int] = [2, 3, 4],
                 width_list: List[int] = [16, 32, 48, 64],
                 kernel_list: List[int] = [3, 5, 7],
                 expand_list: List[int] = [3, 4, 6]):
        super().__init__()
        
        self.num_classes = num_classes
        self.depth_list = depth_list
        self.width_list = width_list
        self.kernel_list = kernel_list
        self.expand_list = expand_list
        
        # 最大配置
        max_depth = max(depth_list)
        max_width = max(width_list)
        
        # 第一个卷积
        self.first_conv = nn.Conv2d(3, base_channels, 3, padding=1, bias=False)
        self.first_bn = nn.BatchNorm2d(base_channels)
        
        # 构建阶段
        self.blocks = nn.ModuleList()
        self.stage_info = []  # 记录每个阶段的信息
        
        channels = base_channels
        for stage_idx in range(4):
            stage_blocks = nn.ModuleList()
            
            for block_idx in range(max_depth):
                # 每个块支持多种配置
                block = FlexibleBlock(
                    in_channels=channels,
                    out_channels=channels * 2 if stage_idx > 0 else channels,
                    width_list=width_list,
                    kernel_list=kernel_list,
                    expand_list=expand_list,
                    stride=2 if block_idx == 0 and stage_idx > 0 else 1
                )
                stage_blocks.append(block)
            
            self.blocks.append(stage_blocks)
            channels *= 2 if stage_idx > 0 else 1
            self.stage_info.append({
                'depth': max_depth,
                'channels': channels
            })
        
        # 分类头
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)
    
    def forward(self, x: torch.Tensor, 
               config: Dict = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            config: 子网络配置，包含depth, width, kernel, expand等
        """
        if config is None:
            # 使用最大配置
            config = self.sample_max_config()
        
        x = F.relu(self.first_bn(self.first_conv(x)))
        
        # 根据配置选择块
        for stage_idx, stage_blocks in enumerate(self.blocks):
            depth = config.get('depth', {}).get(stage_idx, len(stage_blocks))
            
            for block_idx in range(depth):
                block_config = {
                    'width': config.get('width', {}).get((stage_idx, block_idx), 
                                                        max(self.width_list)),
                    'kernel': config.get('kernel', {}).get((stage_idx, block_idx),
                                                          max(self.kernel_list)),
                    'expand': config.get('expand', {}).get((stage_idx, block_idx),
                                                          max(self.expand_list))
                }
                x = stage_blocks[block_idx](x, block_config)
        
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def sample_max_config(self) -> Dict:
        """采样最大配置"""
        return {
            'depth': {i: max(self.depth_list) for i in range(len(self.blocks))},
            'width': {},  # 使用默认值
            'kernel': {},
            'expand': {}
        }
    
    def sample_random_config(self) -> Dict:
        """随机采样一个子网络配置"""
        config = {
            'depth': {},
            'width': {},
            'kernel': {},
            'expand': {}
        }
        
        for stage_idx in range(len(self.blocks)):
            config['depth'][stage_idx] = random.choice(self.depth_list)
            
            for block_idx in range(max(self.depth_list)):
                config['width'][(stage_idx, block_idx)] = random.choice(self.width_list)
                config['kernel'][(stage_idx, block_idx)] = random.choice(self.kernel_list)
                config['expand'][(stage_idx, block_idx)] = random.choice(self.expand_list)
        
        return config


class FlexibleBlock(nn.Module):
    """
    灵活的可搜索块
    
    支持运行时选择：
    - 通道数（宽度）
    - 卷积核大小
    - 扩展比率
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 width_list: List[int],
                 kernel_list: List[int],
                 expand_list: List[int],
                 stride: int = 1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width_list = width_list
        self.kernel_list = kernel_list
        self.expand_list = expand_list
        self.stride = stride
        
        max_width = max(width_list)
        max_expand = max(expand_list)
        max_kernel = max(kernel_list)
        
        # 扩展卷积
        self.expand_conv = nn.Conv2d(
            in_channels, in_channels * max_expand, 1, bias=False
        )
        self.expand_bn = nn.BatchNorm2d(in_channels * max_expand)
        
        # 深度可分离卷积（支持多种核大小）
        self.dw_convs = nn.ModuleDict()
        for k in kernel_list:
            padding = k // 2
            self.dw_convs[str(k)] = nn.Conv2d(
                in_channels * max_expand,
                in_channels * max_expand,
                k, stride, padding,
                groups=in_channels * max_expand,
                bias=False
            )
        self.dw_bn = nn.BatchNorm2d(in_channels * max_expand)
        
        # 投影卷积
        self.project_conv = nn.Conv2d(
            in_channels * max_expand, max_width, 1, bias=False
        )
        self.project_bn = nn.BatchNorm2d(max_width)
        
        # 跳跃连接
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        elif stride == 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(stride),
                nn.Conv2d(in_channels, out_channels, 1, bias=False)
            )
    
    def forward(self, x: torch.Tensor, config: Dict) -> torch.Tensor:
        """前向传播，根据配置动态选择"""
        width = config.get('width', max(self.width_list))
        kernel = config.get('kernel', max(self.kernel_list))
        expand = config.get('expand', max(self.expand_list))
        
        in_ch = x.size(1)
        
        # 扩展
        out = self.expand_conv(x)
        out = self.expand_bn(out)
        out = F.relu(out)
        
        # 只使用需要的通道
        expand_ch = in_ch * expand
        out = out[:, :expand_ch]
        
        # 深度可分离卷积
        out = self.dw_convs[str(kernel)](out)
        out = self.dw_bn(out)
        out = F.relu(out)
        
        # 投影
        out = self.project_conv(out)
        out = self.project_bn(out)
        
        # 只使用需要的通道
        out = out[:, :width]
        
        # 残差连接
        if self.stride == 1 and width == self.out_channels:
            shortcut = self.shortcut(x)
            if shortcut.size(1) != width:
                shortcut = shortcut[:, :width]
            out = out + shortcut
        
        return out


class FBNetSearchSpace(nn.Module):
    """
    FBNet搜索空间
    
    特点：
    1. 使用Gumbel-Softmax实现可微分采样
    2. 直接在目标硬件延迟上优化
    3. 搜索和训练一体化
    
    比喻：就像餐厅后厨，厨师（控制器）根据
    客人需求（硬件约束）调整菜谱（架构）
    """
    
    def __init__(self,
                 num_classes: int = 10,
                 num_blocks: int = 22,
                 num_choices: int = 9):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_choices = num_choices
        
        # 每个块的选择
        self.block_choices = nn.Parameter(
            torch.randn(num_blocks, num_choices)
        )
        
        # 构建候选操作
        self.blocks = nn.ModuleList()
        in_channels = 16
        
        for i in range(num_blocks):
            if i in [5, 12, 18]:  # 下采样点
                stride = 2
                out_channels = in_channels * 2
            else:
                stride = 1
                out_channels = in_channels
            
            block_ops = nn.ModuleList()
            for _ in range(num_choices):
                op = self._create_candidate_op(in_channels, out_channels, stride)
                block_ops.append(op)
            
            self.blocks.append(block_ops)
            in_channels = out_channels
        
        # 分类头
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, num_classes)
        
        # 延迟预测
        self.latency_lookup = self._build_latency_lookup()
        self.target_latency = 25.0  # ms
        self.latency_weight = 0.1
    
    def _create_candidate_op(self, in_ch: int, out_ch: int, stride: int) -> nn.Module:
        """创建候选操作"""
        ops = [
            lambda: nn.Sequential(  # 3x3可分离卷积
                nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            ),
            lambda: nn.Sequential(  # 5x5可分离卷积
                nn.Conv2d(in_ch, in_ch, 5, stride, 2, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            ),
            lambda: nn.Sequential(  # 7x7可分离卷积
                nn.Conv2d(in_ch, in_ch, 7, stride, 3, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            ),
            lambda: nn.Sequential(  # 3x3空洞卷积
                nn.Conv2d(in_ch, in_ch, 3, stride, 2, dilation=2, 
                         groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            ),
            lambda: nn.Identity() if stride == 1 and in_ch == out_ch else \
                   nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
        ]
        
        # 随机选择一个基础操作并包装
        base_op = random.choice(ops[:3])()  # 通常选择可分离卷积
        return base_op
    
    def _build_latency_lookup(self) -> torch.Tensor:
        """构建延迟查找表（模拟数据）"""
        # 每个操作的延迟（毫秒）
        latencies = torch.tensor([
            [2.5, 3.2, 4.1, 3.0, 0.5] + [2.0] * 4  # 实际应该有9个选择
            for _ in range(self.num_blocks)
        ])
        return latencies
    
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            logits: 分类输出
            expected_latency: 期望延迟
        """
        # Gumbel-Softmax采样
        weights = F.gumbel_softmax(
            self.block_choices, 
            tau=temperature, 
            hard=False,
            dim=1
        )
        
        # 前向传播
        for i, block_ops in enumerate(self.blocks):
            outputs = []
            for j, op in enumerate(block_ops):
                outputs.append(weights[i, j] * op(x))
            x = sum(outputs)
            x = F.relu(x)
        
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        
        # 计算期望延迟
        expected_latency = (weights * self.latency_lookup.to(x.device)).sum()
        
        return logits, expected_latency
    
    def get_latency_loss(self, expected_latency: torch.Tensor) -> torch.Tensor:
        """计算延迟损失"""
        return F.relu(expected_latency - self.target_latency) ** 2
    
    def get_architecture(self, hard: bool = True) -> List[int]:
        """获取当前架构"""
        if hard:
            return self.block_choices.argmax(dim=1).tolist()
        else:
            return F.softmax(self.block_choices, dim=1)


class HardwareAwareTrainer:
    """
    硬件感知训练器
    
    同时优化：
    1. 分类准确率
    2. 硬件延迟
    3. 模型大小
    """
    
    def __init__(self, 
                 model: nn.Module,
                 hardware_constraint: HardwareConstraint,
                 train_loader,
                 val_loader,
                 device: str = 'cuda'):
        
        self.model = model.to(device)
        self.hardware = hardware_constraint
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200
        )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        total_acc = 0
        total_latency_loss = 0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            if isinstance(self.model, FBNetSearchSpace):
                logits, expected_latency = self.model(inputs, temperature=max(0.5, 1 - epoch/100))
                latency_loss = self.model.get_latency_loss(expected_latency)
            else:
                logits = self.model(inputs)
                latency_loss = torch.tensor(0.0)
            
            # 分类损失
            ce_loss = self.criterion(logits, targets)
            
            # 总损失
            loss = ce_loss + self.model.latency_weight * latency_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_acc += (logits.argmax(1) == targets).float().mean().item()
            total_latency_loss += latency_loss.item()
        
        self.scheduler.step()
        
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'accuracy': total_acc / n_batches * 100,
            'latency_loss': total_latency_loss / n_batches
        }
    
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if isinstance(self.model, FBNetSearchSpace):
                    logits, _ = self.model(inputs)
                else:
                    logits = self.model(inputs)
                
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return {'accuracy': 100. * correct / total}


import random


def demo_hardware_aware_nas():
    """
    硬件感知NAS演示
    
    思考题：
    1. 为什么要同时训练超网的所有子网络？
    2. FBNet和OFA各有什么优缺点？
    3. 如何在不同的硬件约束之间做权衡？
    """
    print("=" * 70)
    print("硬件感知NAS演示")
    print("=" * 70)
    
    # 1. 超级网络演示
    print("\n" + "-" * 70)
    print("1. 超级网络 (SuperNet)")
    print("-" * 70)
    
    supernet = SuperNet(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    
    # 测试不同配置
    configs = [
        ("最大配置", supernet.sample_max_config()),
        ("随机配置1", supernet.sample_random_config()),
        ("随机配置2", supernet.sample_random_config()),
    ]
    
    print("\n不同配置的输出形状:")
    for name, config in configs:
        y = supernet(x, config)
        print(f"  {name}: {y.shape}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in supernet.parameters())
    print(f"\n超级网络总参数量: {total_params:,}")
    
    # 2. 硬件约束
    print("\n" + "-" * 70)
    print("2. 硬件约束配置")
    print("-" * 70)
    
    hardware_configs = [
        HardwareConstraint(target_latency=10, target_memory=50, target_power=2),
        HardwareConstraint(target_latency=25, target_memory=100, target_power=5),
        HardwareConstraint(target_latency=50, target_memory=200, target_power=10),
    ]
    
    for i, hw in enumerate(hardware_configs):
        print(f"\n配置{i+1}:")
        print(f"  目标延迟: {hw.target_latency}ms")
        print(f"  目标内存: {hw.target_memory}MB")
        print(f"  目标功耗: {hw.target_power}W")
    
    # 3. FBNet搜索空间
    print("\n" + "-" * 70)
    print("3. FBNet搜索空间")
    print("-" * 70)
    
    fbnet = FBNetSearchSpace(num_classes=10, num_blocks=10, num_choices=5)
    
    # 测试前向传播
    x = torch.randn(2, 3, 32, 32)
    logits, latency = fbnet(x)
    
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"期望延迟: {latency.item():.2f}ms")
    print(f"目标延迟: {fbnet.target_latency}ms")
    print(f"延迟损失: {fbnet.get_latency_loss(latency).item():.4f}")
    
    # 获取当前架构
    arch = fbnet.get_architecture(hard=True)
    print(f"\n当前架构选择: {arch}")
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    
    return supernet, fbnet


if __name__ == "__main__":
    demo_hardware_aware_nas()
