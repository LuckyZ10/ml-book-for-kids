"""
硬件感知NAS (Hardware-Aware NAS) 完整实现
包含：延迟预测器、多目标优化、FBNet风格训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Dict

# ==================== 延迟预测器 ====================

class LookupTablePredictor:
    """基于查找表的延迟预测器"""
    
    def __init__(self, device_type='cpu'):
        self.device_type = device_type
        self.lookup_table = {}
        self._build_table()
    
    def _build_table(self):
        """构建查找表（模拟测量数据）"""
        operations = ['conv3x3', 'conv5x5', 'sep_conv3x3', 'sep_conv5x5',
                     'max_pool', 'avg_pool', 'skip']
        base_latencies = {
            'conv3x3': 1.0, 'conv5x5': 1.5, 'sep_conv3x3': 0.6,
            'sep_conv5x5': 0.9, 'max_pool': 0.1, 'avg_pool': 0.1, 'skip': 0.0,
        }
        
        for op in operations:
            for cin in [16, 32, 64, 128, 256]:
                for cout in [16, 32, 64, 128, 256]:
                    for size in [56, 28, 14, 7]:
                        key = (op, cin, cout, size)
                        base = base_latencies[op]
                        channels_factor = (cin * cout) / (64 * 64)
                        size_factor = (size ** 2) / (28 ** 2)
                        noise = random.uniform(0.9, 1.1)
                        latency = base * channels_factor * size_factor * noise
                        self.lookup_table[key] = max(0.01, latency)
    
    def predict(self, op_type, in_channels, out_channels, input_size):
        """预测单个操作的延迟"""
        key = (op_type, in_channels, out_channels, input_size)
        if key in self.lookup_table:
            return self.lookup_table[key]
        closest = min(self.lookup_table.keys(), 
                     key=lambda k: abs(k[1] - in_channels) + abs(k[2] - out_channels) + abs(k[3] - input_size))
        return self.lookup_table[closest]
    
    def predict_network(self, architecture):
        """预测整个网络的延迟"""
        return sum(self.predict(layer['op_type'], layer['in_channels'], 
                               layer['out_channels'], layer['input_size']) 
                  for layer in architecture)


# ==================== 硬件感知NAS模型 ====================

class HardwareAwareNASCell(nn.Module):
    """硬件感知NAS单元"""
    
    def __init__(self, C, num_ops=7):
        super().__init__()
        self.C = C
        self.num_ops = num_ops
        
        self.ops = nn.ModuleList([
            nn.Identity(),
            nn.Conv2d(C, C, 3, padding=1, groups=C),
            nn.Conv2d(C, C, 5, padding=2, groups=C),
            nn.MaxPool2d(3, padding=1),
            nn.AvgPool2d(3, padding=1),
            nn.Conv2d(C, C, 1),
            nn.Sequential(nn.Conv2d(C, C*2, 1), nn.Conv2d(C*2, C*2, 3, padding=1, groups=C*2), nn.Conv2d(C*2, C, 1)),
        ])
        
        self.alphas = nn.Parameter(torch.randn(num_ops) * 0.01)
        # 模拟操作延迟（毫秒）
        self.op_latencies = torch.tensor([0.1, 1.0, 1.5, 0.3, 0.3, 0.8, 2.0])
    
    def forward(self, x):
        weights = F.softmax(self.alphas, dim=-1)
        return sum(w * op(x) for w, op in zip(weights, self.ops))
    
    def expected_latency(self):
        """计算期望延迟"""
        weights = F.softmax(self.alphas, dim=-1)
        return (weights * self.op_latencies.to(weights.device)).sum()


class HardwareAwareNetwork(nn.Module):
    """硬件感知NAS网络"""
    
    def __init__(self, num_classes=10, base_channels=16, num_cells=4,
                 target_latency=50.0, latency_weight=0.1):
        super().__init__()
        self.target_latency = target_latency
        self.latency_weight = latency_weight
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels), nn.ReLU()
        )
        
        self.cells = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_cells):
            if i in [num_cells // 3, 2 * num_cells // 3]: in_ch *= 2
            self.cells.append(HardwareAwareNASCell(in_ch))
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_ch, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        for cell in self.cells: x = cell(x)
        x = self.global_pool(x)
        return self.classifier(x.view(x.size(0), -1))
    
    def expected_latency(self):
        return sum(cell.expected_latency() for cell in self.cells)
    
    def hardware_aware_loss(self, logits, targets):
        """硬件感知损失函数"""
        ce_loss = F.cross_entropy(logits, targets)
        expected_lat = self.expected_latency()
        
        if expected_lat > self.target_latency:
            latency_loss = ((expected_lat - self.target_latency) / self.target_latency) ** 2
        else:
            latency_loss = torch.tensor(0.0, device=logits.device)
        
        total_loss = ce_loss + self.latency_weight * latency_loss
        return total_loss, ce_loss, latency_loss, expected_lat
    
    def get_architecture_params(self): return [cell.alphas for cell in self.cells]
    
    def get_network_params(self):
        arch_ids = set(id(p) for p in self.get_architecture_params())
        for p in self.parameters():
            if id(p) not in arch_ids: yield p


if __name__ == '__main__':
    print("=" * 60)
    print("硬件感知NAS测试")
    print("=" * 60)
    
    # 测试延迟预测器
    lut_predictor = LookupTablePredictor()
    latency = lut_predictor.predict('conv3x3', 64, 64, 28)
    print(f"\n单个操作延迟: {latency:.4f} ms")
    
    architecture = [
        {'op_type': 'conv3x3', 'in_channels': 3, 'out_channels': 32, 'input_size': 224},
        {'op_type': 'sep_conv3x3', 'in_channels': 32, 'out_channels': 64, 'input_size': 112},
    ]
    total_latency = lut_predictor.predict_network(architecture)
    print(f"网络总延迟: {total_latency:.4f} ms")
    
    # 测试硬件感知网络
    model = HardwareAwareNetwork(num_classes=10, target_latency=50.0, latency_weight=0.1)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"\n输入: {x.shape}, 输出: {y.shape}")
    print(f"期望延迟: {model.expected_latency():.4f} ms")
    
    targets = torch.tensor([0, 1])
    loss, ce_loss, lat_loss, exp_lat = model.hardware_aware_loss(y, targets)
    print(f"\n损失分解: 总={loss.item():.4f}, CE={ce_loss.item():.4f}, 延迟={lat_loss.item():.4f}")
