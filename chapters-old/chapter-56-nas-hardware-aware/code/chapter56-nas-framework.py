## 六、大模型的架构优化

### 6.1 大模型时代的NAS挑战

当GPT-4、PaLM、LLaMA等大语言模型拥有**数千亿参数**时，传统的NAS方法遇到了前所未有的挑战：

| 挑战 | 传统NAS | 大模型NAS |
|------|---------|-----------|
| 搜索空间 | 10^5 ~ 10^6 | 10^12+ |
| 单次评估成本 | 几GPU小时 | 几百万美元 |
| 训练稳定性 | 相对容易 | 极易发散 |
| 内存需求 | 几十GB | 几十TB |
| 目标 | 准确率+效率 | 效率+可扩展性+推理速度 |

**直接对大模型做NAS是不可能的！**我们需要新的策略。

---

### 6.2 高效Transformer架构搜索

**1. 搜索空间精简**

不是搜索整个模型，而是搜索**关键模块**：

```python
# 传统：搜索整个网络（不可能）
search_space = {
    'num_layers': range(1, 1000),  # 太广！
    'hidden_dim': range(256, 65536),
    'num_heads': range(1, 128),
    ...
}

# 改进：固定大结构，搜索微观配置
search_space = {
    'attention_pattern': ['full', 'local', 'sparse', 'linear'],  # 注意力模式
    'ffn_structure': ['mlp', 'gated', 'expert_choice'],  # 前馈网络结构
    'normalization': ['layernorm', 'rmsnorm', 'scale_norm'],
    'activation': ['gelu', 'swiglu', 'relu2'],
}
```

**2. 渐进式缩放法则**

先在小模型上搜索，再按比例放大：

```python
"""
缩放法则：大模型的最优配置 ≈ 小模型的最优配置按比例放大

例如：
- 在125M参数模型上搜索 → 找到最优depth=12, heads=12
- 应用到1B模型 → depth=24, heads=24（按比例）
- 应用到10B模型 → depth=48, heads=48
"""
```

**3. 参数高效搜索（PEFT + NAS）**

只搜索**少量新增参数**，冻结预训练权重：

```python
class PEFT_NAS:
    """
    参数高效神经架构搜索
    基于LoRA等技术的思想
    """
    def __init__(self, pretrained_model):
        self.backbone = pretrained_model
        
        # 冻结 backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 只搜索这些新增的adapter模块
        self.searchable_adapters = nn.ModuleList([
            SearchableAdapter() for _ in range(num_layers)
        ])
```

---

### 6.3 混合专家模型（MoE）的架构设计

**MoE**（Mixture of Experts）是大模型的核心技术之一。如何让专家网络的数量、容量、路由策略都是最优的？

**MoE-NAS搜索空间**：

```python
moe_search_space = {
    # 专家配置
    'num_experts': [4, 8, 16, 32, 64, 128],  # 专家数量
    'expert_capacity': [0.5, 1.0, 1.5, 2.0],  # 每个专家的容量
    
    # 路由策略
    'router_type': ['softmax', 'expert_choice', 'hash'],
    'top_k': [1, 2, 4],  # 每个token选几个专家
    
    # 负载均衡
    'load_balance_loss': [0.01, 0.05, 0.1, 0.2],
    'aux_loss_weight': [0.001, 0.01, 0.1],
    
    # 专家结构
    'expert_arch': ['mlp', 'mlp_gated', 'conv1d', 'attention_expert'],
}
```

**路由策略搜索示例**：

```python
class SearchableMoELayer(nn.Module):
    """可搜索的MoE层"""
    def __init__(self, d_model, num_experts_list=[8, 16, 32]):
        super().__init__()
        
        # 创建不同配置的专家池
        self.expert_pools = nn.ModuleDict({
            f'expert_{n}': nn.ModuleList([
                Expert(d_model) for _ in range(n)
            ])
            for n in num_experts_list
        })
        
        # 可学习的架构参数：选择哪个专家池
        self.alphas = nn.Parameter(torch.randn(len(num_experts_list)))
        
        # 路由网络（共享）
        self.router = nn.Linear(d_model, max(num_experts_list))
    
    def forward(self, x):
        # 软选择专家池
        weights = F.softmax(self.alphas, dim=0)
        
        # 对每个候选专家池计算输出
        outputs = []
        for i, (name, experts) in enumerate(self.expert_pools.items()):
            output = self.route_and_compute(x, experts)
            outputs.append(weights[i] * output)
        
        return sum(outputs)
```

---

### 6.4 推理优化：早期退出与动态计算

大模型的推理成本极高，如何让模型在**简单输入上快速退出**？

**早期退出（Early Exit）架构搜索**：

```python
class SearchableEarlyExit(nn.Module):
    """可搜索的早期退出架构"""
    def __init__(self, base_model, num_layers):
        super().__init__()
        self.layers = base_model.layers
        
        # 可搜索的退出点
        self.exit_gates = nn.ModuleList([
            ExitGate(hidden_size) for _ in range(num_layers)
        ])
        
        # 架构参数：在哪些层设置退出点
        self.exit_alphas = nn.Parameter(torch.randn(num_layers))
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # 动态决定是否退出
            if self.training:
                # 训练时：软退出（所有路径都走，加权）
                exit_prob = torch.sigmoid(self.exit_alphas[i])
                exit_output = self.exit_gates[i](x)
            else:
                # 推理时：硬退出（条件判断）
                confidence = self.exit_gates[i].confidence(x)
                if confidence > threshold:
                    return exit_output  # 提前退出！
        
        return x  # 完整前向
```

**动态深度搜索**：

```python
"""
动态深度：根据输入复杂度自适应选择网络深度

简单输入 → 走浅层网络（快）
复杂输入 → 走深层网络（准）

搜索目标：
1. 在哪些层可以截断？
2. 截断后的分类头如何设计？
3. 决策阈值如何设置？
"""
```

---

## 七、实战：综合NAS框架实现

### 7.1 统一搜索空间设计

我们要实现一个支持多种改进的统一NAS框架：

```python
"""
综合NAS框架：整合DARTS改进 + 多目标 + 硬件感知
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple

# ========== 操作定义 ==========

class ConvBNReLU(nn.Module):
    """标准卷积块"""
    def __init__(self, C_in, C_out, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=False)
        )
    
    def forward(self, x):
        return self.op(x)

class DepthwiseConv(nn.Module):
    """深度可分离卷积"""
    def __init__(self, C_in, C_out, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, 
                     groups=C_in, bias=False),  # depthwise
            nn.Conv2d(C_in, C_out, 1, bias=False),  # pointwise
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=False)
        )
    
    def forward(self, x):
        return self.op(x)

class DilatedConv(nn.Module):
    """空洞卷积"""
    def __init__(self, C_in, C_out, kernel_size, stride=1, dilation=2):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, 
                     dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU(inplace=False)
        )
    
    def forward(self, x):
        return self.op(x)

class Identity(nn.Module):
    """跳跃连接"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

class Zero(nn.Module):
    """空操作"""
    def __init__(self, stride=1):
        super().__init__()
        self.stride = stride
    
    def forward(self, x):
        if self.stride == 1:
            return x * 0
        return x[:, :, ::self.stride, ::self.stride] * 0

# ========== 可搜索的混合操作 ==========

class FairDARTSMixedOp(nn.Module):
    """
    FairDARTS混合操作：使用Sigmoid替代Softmax
    """
    def __init__(self, C, stride, use_fairdarts=True):
        super().__init__()
        self.use_fairdarts = use_fairdarts
        
        # 定义候选操作
        self.ops = nn.ModuleList([
            ConvBNReLU(C, C, 3, stride),      # conv 3x3
            ConvBNReLU(C, C, 5, stride),      # conv 5x5
            DepthwiseConv(C, C, 3, stride),   # depthwise 3x3
            DepthwiseConv(C, C, 5, stride),   # depthwise 5x5
            DilatedConv(C, C, 3, stride, 2),  # dilated 3x3
            Identity() if stride == 1 else Zero(stride),  # skip / zero
        ])
        
        self.num_ops = len(self.ops)
        # 架构参数（可学习）
        self.alphas = nn.Parameter(torch.zeros(self.num_ops))
    
    def forward(self, x):
        if self.use_fairdarts:
            # FairDARTS: 独立Sigmoid
            weights = torch.sigmoid(self.alphas)
        else:
            # 标准DARTS: Softmax竞争
            weights = F.softmax(self.alphas, dim=0)
        
        # 加权混合
        output = sum(w * op(x) for w, op in zip(weights, self.ops))
        return output
    
    def get_zero_one_loss(self):
        """FairDARTS的零一正则化"""
        if not self.use_fairdarts:
            return 0
        probs = torch.sigmoid(self.alphas)
        return torch.sum(probs * (1 - probs))

class PCDARTSMixedOp(nn.Module):
    """
    PC-DARTS：部分通道连接，节省内存
    """
    def __init__(self, C, stride, K=4):
        super().__init__()
        self.K = K  # 采样因子
        self.channel_stride = K
        
        # 只在采样通道上进行混合
        self.mixed_op = FairDARTSMixedOp(C // K, stride, use_fairdarts=True)
        
        # 保存跳跃连接用于未采样通道
        self.skip = Identity() if stride == 1 else Zero(stride)
    
    def forward(self, x):
        batch_size, C, H, W = x.shape
        
        # 随机采样通道
        sampled_c = C // self.K
        mask = torch.zeros(C, device=x.device)
        indices = torch.randperm(C)[:sampled_c]
        mask[indices] = 1
        
        # 分割输入
        x_sampled = x[:, mask.bool(), :, :]
        x_unchanged = x[:, ~mask.bool(), :, :]
        
        # 对采样通道进行混合操作
        out_sampled = self.mixed_op(x_sampled)
        
        # 未采样通道直接通过
        if self.skip is not None:
            out_unchanged = self.skip(x_unchanged)
        else:
            out_unchanged = x_unchanged
        
        # 合并
        return torch.cat([out_sampled, out_unchanged], dim=1)

# ========== 搜索单元（Cell） ==========

class SearchCell(nn.Module):
    """
    可搜索的DARTS单元
    """
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, 
                 reduction, reduction_prev, use_pc_darts=False):
        super().__init__()
        self.reduction = reduction
        self.steps = steps  # 中间节点数
        self.multiplier = multiplier
        
        # 预处理层
        if reduction_prev:
            self.preprocess0 = ConvBNReLU(C_prev_prev, C, 1, 2)
        else:
            self.preprocess0 = ConvBNReLU(C_prev_prev, C, 1, 1)
        self.preprocess1 = ConvBNReLU(C_prev, C, 1, 1)
        
        # 构建DAG
        self.ops = nn.ModuleList()
        
        MixedOpClass = PCDARTSMixedOp if use_pc_darts else FairDARTSMixedOp
        
        for i in range(self.steps):
            for j in range(2 + i):  # 连接到前两个输入和所有前面的中间节点
                stride = 2 if reduction and j < 2 else 1
                op = MixedOpClass(C, stride)
                self.ops.append(op)
    
    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        offset = 0
        
        for i in range(self.steps):
            # 当前节点 = 所有输入的混合
            s = sum(self.ops[offset + j](h) 
                   for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        
        # 连接最后multiplier个中间节点作为输出
        return torch.cat(states[-self.multiplier:], dim=1)

# ========== 硬件延迟预测器 ==========

class LatencyPredictor(nn.Module):
    """
    基于MLP的延迟预测器
    """
    def __init__(self, input_dim=10, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x).squeeze(-1)

def extract_arch_features(arch_config: Dict) -> torch.Tensor:
    """
    从架构配置中提取特征向量
    """
    features = [
        arch_config.get('num_layers', 0),
        arch_config.get('num_channels', 0),
        arch_config.get('num_params', 0) / 1e6,  # 以M为单位
        arch_config.get('num_flops', 0) / 1e9,   # 以G为单位
        arch_config.get('num_conv3x3', 0),
        arch_config.get('num_conv1x1', 0),
        arch_config.get('num_dwconv', 0),
        arch_config.get('num_skip', 0),
        arch_config.get('max_feature_size', 0) / 1e6,
        arch_config.get('avg_channel', 0),
    ]
    return torch.tensor(features, dtype=torch.float32)

# ========== 网络定义 ==========

class SearchNetwork(nn.Module):
    """
    可搜索的网络（整合所有改进）
    """
    def __init__(self, C=16, num_classes=10, layers=8, 
                 criterion=nn.CrossEntropyLoss(),
                 use_fairdarts=True,
                 use_pc_darts=False,
                 use_hardware_aware=False,
                 latency_predictor=None,
                 latency_weight=0.1,
                 steps=4, multiplier=4):
        super().__init__()
        self.criterion = criterion
        self.use_hardware_aware = use_hardware_aware
        self.latency_weight = latency_weight
        self.latency_predictor = latency_predictor
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )
        
        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        
        for i in range(layers):
            # 在1/3和2/3处设置reduction cell
            if layers in [8, 14]:
                reduction = (i in [layers // 3, 2 * layers // 3])
            else:
                reduction = (i in [layers // 3, 2 * layers // 3])
            
            if reduction:
                C_curr *= 2
            
            cell = SearchCell(steps, multiplier, C_prev_prev, C_prev, C_curr,
                            reduction, reduction_prev, use_pc_darts)
            
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
    
    def forward(self, x):
        s0 = s1 = self.stem(x)
        
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
    
    def loss(self, x, target):
        """计算损失（可能包含延迟项）"""
        logits = self.forward(x)
        ce_loss = self.criterion(logits, target)
        
        # 硬件感知损失
        if self.use_hardware_aware and self.latency_predictor is not None:
            # 提取当前架构特征
            arch_features = self.get_arch_features()
            predicted_latency = self.latency_predictor(arch_features)
            
            # 延迟正则化
            latency_loss = self.latency_weight * predicted_latency
            
            return ce_loss + latency_loss, ce_loss, latency_loss
        
        return ce_loss, ce_loss, torch.tensor(0.0)
    
    def get_arch_features(self):
        """提取当前架构的特征（用于延迟预测）"""
        # 简化版：统计当前操作分布
        config = {
            'num_layers': len(self.cells),
            'num_channels': 16,
            'num_params': sum(p.numel() for p in self.parameters()),
        }
        return extract_arch_features(config)
    
    def get_fairdarts_loss(self):
        """获取FairDARTS的零一损失"""
        loss = 0
        for cell in self.cells:
            for op in cell.ops:
                if hasattr(op, 'get_zero_one_loss'):
                    loss += op.get_zero_one_loss()
        return loss
    
    def arch_parameters(self):
        """获取架构参数（alphas）"""
        params = []
        for cell in self.cells:
            for op in cell.ops:
                params.append(op.alphas)
        return params
    
    def network_parameters(self):
        """获取网络权重参数"""
        params = []
        for name, param in self.named_parameters():
            if 'alphas' not in name:
                params.append(param)
        return params

# ========== 架构派生 ==========

def derive_architecture(model, k=2):
    """
    从训练好的模型中派生离散架构
    
    k: 每个节点选择top-k个连接
    """
    genotype = []
    
    for cell_idx, cell in enumerate(model.cells):
        cell_genotype = []
        offset = 0
        
        for i in range(cell.steps):
            edges = []
            for j in range(2 + i):
                op = cell.ops[offset + j]
                
                # 获取权重最大的操作
                if hasattr(op, 'alphas'):
                    if op.use_fairdarts:
                        weights = torch.sigmoid(op.alphas)
                    else:
                        weights = F.softmax(op.alphas, dim=0)
                    
                    # 找出权重最大的操作（排除zero）
                    max_idx = weights[:-1].argmax().item()  # 排除最后一个（通常是zero）
                    max_op = ['conv3x3', 'conv5x5', 'dw3x3', 'dw5x5', 'dil3x3', 'skip'][max_idx]
                    max_weight = weights[max_idx].item()
                    
                    edges.append((max_op, j, max_weight))
                
                offset += 1
            
            # 选择top-k
            edges.sort(key=lambda x: x[2], reverse=True)
            selected = edges[:k]
            cell_genotype.append(selected)
        
        genotype.append(cell_genotype)
    
    return genotype

# ========== 训练函数 ==========

def train_search(model, train_loader, val_loader, epochs=50, 
                 lr=0.025, momentum=0.9, weight_decay=3e-4,
                 arch_lr=3e-4, arch_weight_decay=1e-3,
                 fairdarts_weight=0.0):
    """
    训练搜索网络（双层优化）
    """
    # 优化器
    network_optimizer = torch.optim.SGD(
        model.network_parameters(),
        lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    
    arch_optimizer = torch.optim.Adam(
        model.arch_parameters(),
        lr=arch_lr, betas=(0.5, 0.999), weight_decay=arch_weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        network_optimizer, float(epochs), eta_min=lr/100
    )
    
    for epoch in range(epochs):
        model.train()
        
        for step, (x, y) in enumerate(train_loader):
            # 阶段1：更新网络权重（在训练集上）
            network_optimizer.zero_grad()
            logits = model(x)
            loss = model.criterion(logits, y)
            
            # 添加FairDARTS正则化
            if fairdarts_weight > 0:
                loss += fairdarts_weight * model.get_fairdarts_loss()
            
            loss.backward()
            network_optimizer.step()
            
            # 阶段2：更新架构参数（在验证集上）- 每k步做一次
            if step % 5 == 0:
                try:
                    x_val, y_val = next(val_iter)
                except:
                    val_iter = iter(val_loader)
                    x_val, y_val = next(val_iter)
                
                arch_optimizer.zero_grad()
                loss, ce_loss, lat_loss = model.loss(x_val, y_val)
                
                # 添加FairDARTS正则化
                if fairdarts_weight > 0:
                    loss += fairdarts_weight * model.get_fairdarts_loss()
                
                loss.backward()
                arch_optimizer.step()
        
        scheduler.step()
        
        # 打印进度
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
            
            # 监控跳跃连接数量（DARTS+早期停止）
            num_skips = count_skip_connections(model)
            print(f"  Skip connections: {num_skips}")
            
            if num_skips > len(model.cells) * 2:
                print("⚠️ Warning: Too many skip connections! Consider early stopping.")
    
    return model

def count_skip_connections(model):
    """统计模型中跳跃连接的数量（用于DARTS+监控）"""
    count = 0
    for cell in model.cells:
        for op in cell.ops:
            if hasattr(op, 'alphas'):
                weights = torch.sigmoid(op.alphas) if op.use_fairdarts else F.softmax(op.alphas, dim=0)
                # 如果skip（通常是倒数第二个）权重最大
                if weights[-2] == weights.max():
                    count += 1
    return count

# ========== 多目标进化搜索 ==========

def evaluate_architecture(model, arch_config, train_loader, val_loader, 
                          num_epochs=10):
    """
    评估一个具体架构的性能
    """
    # 构建具体架构
    # ...（简化，实际需要根据config构建）
    
    # 训练几轮
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
    
    # 评估
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            _, pred = logits.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    
    accuracy = correct / total
    
    # 估计延迟（使用预测器或查找表）
    latency = estimate_latency(arch_config)
    
    return {'accuracy': accuracy, 'latency': latency}

def estimate_latency(arch_config):
    """估算架构延迟（简化版）"""
    # 简化：基于FLOPs估计
    flops = arch_config.get('num_flops', 1e9)
    # 假设1G FLOPs = 10ms（非常简化的估计）
    return flops / 1e8

def nsga2_nas(search_space, population_size=20, generations=10,
              train_loader=None, val_loader=None):
    """
    NSGA-II多目标架构搜索（简化版）
    """
    # 初始化种群
    population = [random_architecture(search_space) 
                  for _ in range(population_size)]
    
    for gen in range(generations):
        print(f"Generation {gen+1}/{generations}")
        
        # 评估种群
        for arch in population:
            if 'accuracy' not in arch:
                result = evaluate_architecture(
                    None, arch, train_loader, val_loader, num_epochs=5
                )
                arch.update(result)
        
        # 非支配排序
        fronts = non_dominated_sort(population)
        
        # 选择下一代
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= population_size:
                new_population.extend(front)
            else:
                # 计算拥挤距离，选择最分散的
                compute_crowding_distance(front)
                front.sort(key=lambda x: x.get('crowding', 0), reverse=True)
                remaining = population_size - len(new_population)
                new_population.extend(front[:remaining])
                break
        
        population = new_population
        
        # 生成后代（交叉和变异）
        offspring = generate_offspring(population, search_space)
        population.extend(offspring)
    
    # 返回Pareto前沿
    return non_dominated_sort(population)[0]

def random_architecture(search_space):
    """随机采样一个架构"""
    return {
        key: np.random.choice(values)
        for key, values in search_space.items()
    }

def non_dominated_sort(population):
    """非支配排序"""
    fronts = [[]]
    
    for i, p in enumerate(population):
        p['dominated_set'] = []
        p['domination_count'] = 0
        
        for j, q in enumerate(population):
            if i == j:
                continue
            
            # 检查支配关系
            if dominates(p, q):
                p['dominated_set'].append(q)
            elif dominates(q, p):
                p['domination_count'] += 1
        
        if p['domination_count'] == 0:
            p['rank'] = 0
            fronts[0].append(p)
    
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in p['dominated_set']:
                q['domination_count'] -= 1
                if q['domination_count'] == 0:
                    q['rank'] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    
    return fronts[:-1]  # 去掉最后一个空的前沿

def dominates(a, b):
    """检查a是否支配b"""
    # 目标：最大化accuracy，最小化latency
    better_or_equal = (
        a.get('accuracy', 0) >= b.get('accuracy', 0) and
        a.get('latency', float('inf')) <= b.get('latency', float('inf'))
    )
    strictly_better = (
        a.get('accuracy', 0) > b.get('accuracy', 0) or
        a.get('latency', float('inf')) < b.get('latency', float('inf'))
    )
    return better_or_equal and strictly_better

def compute_crowding_distance(front):
    """计算拥挤距离"""
    if len(front) <= 2:
        for p in front:
            p['crowding'] = float('inf')
        return
    
    for p in front:
        p['crowding'] = 0
    
    # 对每个目标
    objectives = [('accuracy', True), ('latency', False)]  # (name, maximize)
    
    for obj_name, maximize in objectives:
        front.sort(key=lambda x: x.get(obj_name, 0), reverse=maximize)
        
        front[0]['crowding'] = front[-1]['crowding'] = float('inf')
        
        f_max = front[0].get(obj_name, 0)
        f_min = front[-1].get(obj_name, 0)
        
        if f_max - f_min > 0:
            for i in range(1, len(front) - 1):
                front[i]['crowding'] += (
                    abs(front[i+1].get(obj_name, 0) - front[i-1].get(obj_name, 0))
                    / (f_max - f_min)
                )

def generate_offspring(parents, search_space, num_offspring=10):
    """生成后代（交叉和变异）"""
    offspring = []
    
    for _ in range(num_offspring):
        # 随机选择两个父代
        p1, p2 = np.random.choice(parents, 2, replace=False)
        
        # 交叉
        child = {}
        for key in search_space:
            child[key] = p1[key] if np.random.random() < 0.5 else p2[key]
        
        # 变异
        for key in search_space:
            if np.random.random() < 0.1:  # 10%变异率
                child[key] = np.random.choice(search_space[key])
        
        offspring.append(child)
    
    return offspring

# ========== 使用示例 ==========

def demo():
    """
    演示如何使用综合NAS框架
    """
    print("=" * 60)
    print("综合NAS框架演示")
    print("=" * 60)
    
    # 1. 创建搜索网络（整合FairDARTS + PC-DARTS）
    print("\n1. 创建可搜索网络...")
    model = SearchNetwork(
        C=16,
        num_classes=10,
        layers=8,
        use_fairdarts=True,
        use_pc_darts=True,
        use_hardware_aware=False,
        fairdarts_weight=0.01
    )
    
    print(f"   总参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"   架构参数数量: {sum(p.numel() for p in model.arch_parameters())}")
    
    # 2. 显示搜索空间
    print("\n2. 候选操作:")
    ops = ['conv3x3', 'conv5x5', 'dw3x3', 'dw5x5', 'dil3x3', 'skip']
    for i, op in enumerate(ops):
        print(f"   [{i}] {op}")
    
    # 3. 架构派生示例
    print("\n3. 当前架构分布:")
    genotype = derive_architecture(model)
    print(f"   已派生 {len(genotype)} 个cell的架构")
    
    print("\n" + "=" * 60)
    print("演示完成！要运行完整搜索，请提供数据集。")
    print("=" * 60)

if __name__ == "__main__":
    demo()
