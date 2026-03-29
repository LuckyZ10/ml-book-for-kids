# 第五十八章 模型压缩与边缘部署——让AI走进千家万户

> *"让复杂的AI模型在手机上实时运行，就像把一座图书馆装进你的口袋——这需要智慧的压缩艺术。"*

## 58.1 引言：为什么需要模型压缩？

### 58.1.1 从云端到边缘的AI革命

想象这样一个场景：你打开手机相机，对准街边的花草，手机立即告诉你这是"绣球花"，甚至还能讲解它的养护知识——整个过程不到0.1秒，而且不需要联网。这不是科幻，而是边缘AI（Edge AI）每天都在发生的事情。

但在这神奇体验的背后，隐藏着一个巨大的技术挑战：**如何让庞大复杂的深度学习模型在资源受限的设备上高效运行？**

让我们先看一些数字：

| 模型 | 参数量 | 存储大小 | 推理计算量 |
|------|--------|----------|------------|
| GPT-4 | ~1.8万亿 | ~3.6TB | ~10^12 FLOPs |
| ResNet-152 | 6000万 | ~230MB | ~11.6 GFLOPs |
| VGG-16 | 1.38亿 | ~528MB | ~15.5 GFLOPs |
| BERT-base | 1.1亿 | ~440MB | ~22 GFLOPs |

表58.1：主流深度学习模型的资源需求

这些模型在服务器上运行毫无压力——服务器有强大的GPU、充足的内存和稳定的供电。但当我们要把它们部署到手机、智能手表、IoT传感器甚至自动驾驶汽车上时，问题就出现了：

**资源约束三重奏：**

1. **计算能力受限**：手机芯片的算力只有服务器GPU的1/100甚至1/1000
2. **内存捉襟见肘**：旗舰手机通常只有8-12GB RAM，而许多模型就需要几GB
3. **能耗严格受限**：电池电量有限，一次推理如果消耗太多电，用户体验就会极差

这就好比要把一头大象装进一辆小汽车——我们需要一种"压缩魔法"。

### 58.1.2 模型压缩的核心思想

**费曼比喻：模型压缩就像整理行李箱**

想象你要去长途旅行，但航空公司只允许带一个登机箱。你的衣柜里有一百件衣服（就像神经网络的百万参数），该怎么办？

聪明的做法是：
- **剪掉不重要的**：那件"万一需要"的燕尾服？留下
delete
- **压缩体积**：把毛衣卷起来而不是折叠，节省空间
- **只带精华**：选择百搭的基础款，少即是多
- **学习打包技巧**：把袜子塞进鞋子里，最大化利用空间

模型压缩正是用类似的思路来处理神经网络：

| 压缩技术 | 行李箱类比 | 核心思想 |
|----------|------------|----------|
| **剪枝 (Pruning)** | 扔掉不常穿的衣服 | 删除不重要的权重/神经元 |
| **量化 (Quantization)** | 把厚毛衣压缩成真空袋 | 用更少的位数表示参数 |
| **知识蒸馏 (Distillation)** | 老旅行者传授打包秘诀 | 大模型教小模型如何预测 |
| **高效架构设计** | 选择多功能旅行装备 | 设计天生紧凑的网络结构 |

表58.2：模型压缩技术类比

### 58.1.3 边缘部署的独特挑战

边缘设备（Edge Devices）泛指那些在网络"边缘"、靠近数据源的设备——你的手机、智能摄像头、无人机、车载电脑都是。它们有一个共同点：**必须在本地完成AI推理，不能依赖云端**。

为什么必须在本地运行？

1. **实时性需求**：自动驾驶汽车必须在毫秒级做出决策，等云端响应可能车都撞了
2. **隐私保护**：人脸识别、健康监测等敏感数据不应该离开设备
3. **网络不稳定**：地铁、飞机、偏远地区没有可靠网络连接
4. **成本考量**：云端API调用需要付费，本地运行只需一次性硬件成本

**费曼比喻：边缘部署就像把图书馆搬进手机**

想象一个没有互联网的年代，你想随时查阅百科知识。有两种选择：
- **云端方案**：每次想查资料都写信给远方的图书馆，等他们寄书过来（延迟高、依赖通信）
- **边缘方案**：把图书馆的精华内容摘录成一本便携的"袖珍百科"（本地、快速、独立）

模型压缩和边缘部署的目标，就是把"大图书馆"（复杂模型）变成"袖珍百科"（压缩模型），让它既便携又实用。

### 58.1.4 本章内容概览

本章将深入探讨模型压缩的核心技术和边缘部署的实战方法：

```
本章知识地图
│
├── 58.2 模型剪枝：精简而不简单
│   ├── 非结构化剪枝 vs 结构化剪枝
│   └── 彩票假说 (Lottery Ticket Hypothesis)
│
├── 58.3 模型量化：用更少位数存储
│   ├── INT8/INT4量化原理
│   └── PTQ vs QAT
│
├── 58.4 知识蒸馏：师承名师
│   ├── 教师-学生框架
│   └── 温度参数的妙用
│
├── 58.5 高效神经网络架构
│   ├── MobileNet深度可分离卷积
│   └── EfficientNet复合缩放
│
└── 58.6 边缘部署实战
    ├── ONNX导出与优化
    ├── TensorRT加速
    └── TFLite移动端部署
```

## 58.2 模型剪枝：精简而不简单

### 58.2.1 剪枝的基本概念

**费曼比喻：剪枝就像修剪盆栽**

想象你有一盆繁茂的榕树，枝叶过于浓密反而影响整体形态和生长。园艺师会告诉你：剪掉那些瘦弱、交叉、向内生长的枝条，让植株更健康、更美观。

神经网络剪枝（Neural Network Pruning）正是类似的"园艺工作"：

神经网络中的许多连接（权重）对最终输出的贡献微乎其微。就像盆栽中的细弱枝条，它们消耗资源却不创造价值。剪枝的目标就是识别并移除这些"冗余连接"，同时尽量保持模型的预测能力。

**数学视角：稀疏性引入**

设神经网络第$l$层的权重矩阵为$\mathbf{W}^{(l)} \in \mathbb{R}^{d_{out} \times d_{in}}$。剪枝的目标是找到一个二值掩码矩阵$\mathbf{M}^{(l)} \in \{0, 1\}^{d_{out} \times d_{in}}$，使得：

$$\hat{\mathbf{W}}^{(l)} = \mathbf{W}^{(l)} \odot \mathbf{M}^{(l)}$$

其中$\odot$表示逐元素乘法，$\mathbf{M}^{(l)}_{ij} = 0$表示剪去该权重，$\mathbf{M}^{(l)}_{ij} = 1$表示保留。

稀疏度（Sparsity）定义为：

$$\text{Sparsity} = \frac{\sum_l \|\mathbf{M}^{(l)}\|_0}{\sum_l d_{out}^{(l)} \cdot d_{in}^{(l)}}$$

其中$\|\cdot\|_0$表示L0范数（非零元素个数）。

### 58.2.2 剪枝的粒度：非结构化 vs 结构化

剪枝可以按照"粒度"分为两大类：

#### 非结构化剪枝 (Unstructured Pruning)

**特点**：独立地剪除单个权重，不考虑它们在矩阵中的位置。

**优点**：
- 灵活性最高，可以达到极高的稀疏度（90%以上）
- 精度损失最小，因为只移除最不重要的连接

**缺点**：
- 需要专门的稀疏矩阵运算库支持
- 硬件加速困难，因为零值分布不规则

**重要性度量方法**：

最常见的是**幅度剪枝 (Magnitude Pruning)**——认为绝对值小的权重不重要：

$$\mathbf{M}_{ij} = \mathbb{1}[|W_{ij}| > \theta]$$

其中$\theta$是剪枝阈值。

**更精细的方法**：

1. **敏感度分析**：测量每个权重对损失的敏感度
   $$S_{ij} = \left|\frac{\partial \mathcal{L}}{\partial W_{ij}} \cdot W_{ij}\right|$$

2. **二阶方法 (Optimal Brain Damage/Surgeon)**：使用Hessian矩阵评估权重重要性
   $$\Delta \mathcal{L} \approx \frac{1}{2} \mathbf{w}^T \mathbf{H} \mathbf{w}$$

#### 结构化剪枝 (Structured Pruning)

**特点**：按结构单元剪除——整个卷积核、通道或层。

**费曼比喻**：
- 非结构化剪枝：随机拔掉树上的几片叶子
- 结构化剪枝：剪掉整根枝条

**常见粒度**：

| 粒度 | 说明 | 硬件友好度 | 精度损失 |
|------|------|------------|----------|
| **权重级** | 单个权重 | 低 | 低 |
| **向量级** | 权重矩阵的行/列 | 中 | 中 |
| **核级** | 卷积核(2D) | 高 | 中 |
| **通道级** | 整个特征通道 | 高 | 中-高 |
| **层间** | 整个层 | 极高 | 高 |

表58.3：不同剪枝粒度的对比

**通道剪枝的核心问题**：

给定第$l$层的输出特征图$\mathbf{X}^{(l)} \in \mathbb{R}^{C_{out} \times H \times W}$，如何评估第$c$个通道的重要性？

常用指标：

1. **L1范数**：$\text{Importance}_c = \sum_{i,j} |W_{c,:,:}^{(l)}|$

2. **BN层缩放因子** (Network Slimming)：利用BatchNorm的$\gamma$系数
   $$y = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
   如果$\gamma \approx 0$，说明该通道可以被移除。

3. **特征图激活**：基于该通道输出的平均激活值评估重要性

### 58.2.3 剪枝策略：一次性 vs 渐进式

#### 一次性剪枝 (One-shot Pruning)

流程：训练 → 剪枝 → 微调

```python
# 伪代码：一次性剪枝
def one_shot_pruning(model, pruning_ratio):
    # 1. 训练完整模型
    train(model)
    
    # 2. 基于幅度计算掩码
    for layer in model.layers:
        threshold = percentile(abs(layer.weights), pruning_ratio * 100)
        layer.mask = abs(layer.weights) > threshold
        layer.weights *= layer.mask
    
    # 3. 微调恢复精度
    fine_tune(model)
    
    return model
```

**缺点**：一次性剪掉太多权重，模型可能"休克"，难以恢复。

#### 渐进式剪枝 (Iterative Pruning)

流程：训练 → 剪枝一点 → 训练 → 再剪一点 → ... → 微调

每次只剪掉目标比例的一部分（如$1 - (1 - p)^{1/n}$），共进行$n$轮。

**优点**：模型有时间逐步适应新的稀疏结构，最终效果更好。

### 58.2.4 彩票假说：寻找"天选之子"

2019年，Jonathan Frankle和Michael Carbin在ICLR上提出了一个震撼业界的发现——**彩票假说 (Lottery Ticket Hypothesis, LTH)**。

**核心观点**：

> 一个随机初始化的密集神经网络中，存在一个稀疏子网络（称为"中奖彩票"），如果单独训练这个子网络（使用原网络的初始化权重），它能在相同或更少的迭代次数内达到与原始网络相当的测试精度。

**费曼比喻：彩票假说就像寻找天选之才**

想象一个庞大的交响乐团（密集网络），里面有很多乐手。但实际上，只要找到一小群特别优秀的乐手（中奖子网络），用他们最初的排练状态（原始初始化）重新训练，就能演奏出同样精彩的音乐——甚至学得更快！这些幸运儿"赢在了起跑线上"。

**算法流程（迭代幅度剪枝）**：

```
算法：寻找中奖彩票 (Iterative Magnitude Pruning)
输入：初始化网络θ₀，训练数据D，目标稀疏度s，剪枝轮数n
输出：中奖彩票（掩码M，初始化权重θ₀）

1. θ ← 复制(θ₀)                    # 保存原始初始化
2. for i = 1 to n:
3.     θ ← Train(θ, D)             # 训练当前网络
4.     p_i ← 1 - (1 - s)^(1/n)     # 本轮剪枝比例
5.     M ← MagnitudePruning(θ, p_i) # 基于幅度剪枝
6.     θ ← θ₀ ⊙ M                  # 重置为原始初始化
7. end for
8. 返回 (M, θ₀)
```

**关键发现**：

1. **重置的重要性**：必须回到原始初始化，而不是随机重新初始化
2. **早停的权重**：在某些工作中，发现使用训练早期（如前10%迭代）的权重重置效果更好
3. **普遍存在**：彩票假说在CNN、ResNet、Transformer等架构中都被验证

**数学解释**：

彩票假说的理论研究表明，一个足够过参数化的随机网络以高概率包含一个"好的"子网络，可以在不训练的情况下就接近损失函数。

形式化地，对于任意有界的目标网络$f$，存在常数$C$使得：当网络宽度$m > C$，随机初始化的网络$g_{\theta_0}$以概率$1 - \delta$包含一个子网络$g_{\theta_0 \odot M}$满足：

$$\|g_{\theta_0 \odot M} - f\| \leq \epsilon$$

**实战代码：彩票假说实现**

```python
"""
彩票假说实现：寻找稀疏可训练子网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import copy


class LotteryTicketHypothesis:
    """
    彩票假说：寻找中奖子网络
    
    费曼比喻：就像在一群人中找到那些"天生就适合"某个任务的人
    """
    
    def __init__(self, model, device='cuda'):
        self.device = device
        self.model = model.to(device)
        # 保存原始初始化（这是彩票假说的关键！）
        self.initial_state = copy.deepcopy(model.state_dict())
        self.masks = {}  # 存储各层掩码
        
    def compute_magnitude_mask(self, sparsity_ratio):
        """
        基于权重大小计算剪枝掩码
        
        参数:
            sparsity_ratio: 剪枝比例 (0-1)
        """
        all_weights = []
        
        # 收集所有可剪枝权重
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:  # 只剪枝卷积和全连接层
                all_weights.extend(param.data.cpu().numpy().flatten())
        
        # 计算阈值
        all_weights = np.array(all_weights)
        threshold = np.percentile(np.abs(all_weights), sparsity_ratio * 100)
        
        # 为每层创建掩码
        masks = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                mask = (torch.abs(param.data) > threshold).float()
                masks[name] = mask
                
        return masks
    
    def apply_mask(self, masks):
        """应用掩码到模型权重"""
        for name, param in self.model.named_parameters():
            if name in masks:
                param.data *= masks[name].to(self.device)
                
    def reset_to_initial(self, masks):
        """
        重置到原始初始化（彩票假说的核心！）
        
        费曼比喻：就像让选手回到起点，但告诉他们哪些赛道是"死路"
        """
        initial_dict = copy.deepcopy(self.initial_state)
        current_dict = self.model.state_dict()
        
        for name, param in current_dict.items():
            if name in masks:
                # 保留初始化的值，但被掩码的位置保持为0
                param.data = initial_dict[name].to(self.device) * masks[name].to(self.device)
            else:
                param.data = initial_dict[name].to(self.device)
                
    def iterative_magnitude_pruning(self, train_loader, val_loader, 
                                    target_sparsity, num_iterations, 
                                    epochs_per_iteration, optimizer_fn, 
                                    criterion):
        """
        迭代幅度剪枝算法
        
        参数:
            train_loader: 训练数据
            val_loader: 验证数据
            target_sparsity: 目标稀疏度
            num_iterations: 剪枝迭代次数
            epochs_per_iteration: 每轮迭代训练轮数
            optimizer_fn: 优化器构造函数
            criterion: 损失函数
        """
        results = []
        current_masks = None
        
        # 计算每轮剪枝比例
        # 使用公式：s_total = 1 - (1 - s_round)^n
        # 解得：s_round = 1 - (1 - s_total)^(1/n)
        per_iteration_sparsity = 1 - (1 - target_sparsity) ** (1 / num_iterations)
        
        print(f"开始迭代剪枝...")
        print(f"目标稀疏度: {target_sparsity:.2%}")
        print(f"迭代次数: {num_iterations}")
        print(f"每轮剪枝比例: {per_iteration_sparsity:.2%}")
        
        for iteration in range(num_iterations):
            print(f"\n{'='*60}")
            print(f"迭代 {iteration + 1}/{num_iterations}")
            print(f"{'='*60}")
            
            # 计算当前累计稀疏度
            cumulative_sparsity = 1 - (1 - per_iteration_sparsity) ** (iteration + 1)
            print(f"当前目标稀疏度: {cumulative_sparsity:.2%}")
            
            # 1. 训练（或微调）当前模型
            self.model.train()
            optimizer = optimizer_fn(self.model.parameters())
            
            for epoch in range(epochs_per_iteration):
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    
                    # 应用掩码梯度（防止被剪枝的权重更新）
                    if current_masks:
                        for name, param in self.model.named_parameters():
                            if name in current_masks and param.grad is not None:
                                param.grad *= current_masks[name].to(self.device)
                    
                    optimizer.step()
                    
                    # 确保权重遵守掩码
                    if current_masks:
                        self.apply_mask(current_masks)
                    
                    total_loss += loss.item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                
                acc = 100. * correct / total
                print(f"  Epoch {epoch+1}/{epochs_per_iteration}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%")
            
            # 2. 基于幅度计算新掩码
            current_masks = self.compute_magnitude_mask(cumulative_sparsity)
            
            # 3. 重置到原始初始化（这是彩票假说的关键步骤！）
            self.reset_to_initial(current_masks)
            
            # 评估当前状态
            val_acc = self.evaluate(val_loader)
            actual_sparsity = self.compute_actual_sparsity(current_masks)
            
            results.append({
                'iteration': iteration + 1,
                'sparsity': actual_sparsity,
                'val_acc': val_acc
            })
            
            print(f"  本轮结果: 稀疏度={actual_sparsity:.2%}, 验证精度={val_acc:.2f}%")
        
        # 最终训练
        print(f"\n{'='*60}")
        print("最终训练阶段")
        print(f"{'='*60}")
        
        optimizer = optimizer_fn(self.model.parameters())
        for epoch in range(epochs_per_iteration * 2):
            self.model.train()
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # 应用掩码梯度
                for name, param in self.model.named_parameters():
                    if name in current_masks and param.grad is not None:
                        param.grad *= current_masks[name].to(self.device)
                
                optimizer.step()
                self.apply_mask(current_masks)
        
        final_acc = self.evaluate(val_loader)
        print(f"最终验证精度: {final_acc:.2f}%")
        
        self.masks = current_masks
        return results
    
    def evaluate(self, val_loader):
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return 100. * correct / total
    
    def compute_actual_sparsity(self, masks):
        """计算实际稀疏度"""
        total_params = 0
        zero_params = 0
        
        for name, param in self.model.named_parameters():
            if name in masks:
                total_params += param.numel()
                zero_params += (masks[name] == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0


# 用于测试的简单CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
```

### 58.2.5 结构化剪枝实战

```python
"""
结构化剪枝实现：通道剪枝
"""
import torch
import torch.nn as nn


class StructuredPruner:
    """
    结构化剪枝：按通道/滤波器剪枝
    
    费曼比喻：不是拔掉几片叶子，而是剪掉整根枝条
    """
    
    def __init__(self, model):
        self.model = model
        
    def compute_channel_importance_l1(self, conv_layer):
        """
        基于L1范数计算通道重要性
        
        参数:
            conv_layer: 卷积层 (out_channels, in_channels, k, k)
        返回:
            importance: 每个输出通道的重要性分数
        """
        # 对每个输出通道，计算所有权重的L1范数
        weights = conv_layer.weight.data  # (out_c, in_c, h, w)
        importance = torch.sum(torch.abs(weights), dim=[1, 2, 3])  # (out_c,)
        return importance
    
    def compute_channel_importance_bn(self, bn_layer):
        """
        基于BatchNorm的gamma系数计算重要性
        
        原理：gamma接近0的通道对输出贡献小，可以剪除
        """
        if bn_layer is None:
            return None
        return torch.abs(bn_layer.weight.data)
    
    def prune_conv_layer(self, conv_layer, bn_layer, next_conv_layer, 
                         prune_ratio, importance_fn='l1'):
        """
        剪枝单个卷积层
        
        参数:
            conv_layer: 当前卷积层
            bn_layer: 对应的BN层
            next_conv_layer: 下一层卷积（用于同步输入通道）
            prune_ratio: 剪枝比例
            importance_fn: 重要性评估方法
        """
        # 计算通道重要性
        if importance_fn == 'l1':
            importance = self.compute_channel_importance_l1(conv_layer)
        elif importance_fn == 'bn' and bn_layer is not None:
            importance = self.compute_channel_importance_bn(bn_layer)
        else:
            importance = self.compute_channel_importance_l1(conv_layer)
        
        # 确定保留哪些通道
        num_channels = conv_layer.out_channels
        num_keep = int(num_channels * (1 - prune_ratio))
        
        # 选择重要性最高的通道
        _, keep_indices = torch.topk(importance, num_keep, largest=True, sorted=True)
        keep_indices = keep_indices.sort()[0]  # 排序以保持顺序
        
        # 创建新的卷积层
        new_conv = nn.Conv2d(
            in_channels=conv_layer.in_channels,
            out_channels=num_keep,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            bias=conv_layer.bias is not None
        )
        
        # 复制保留通道的权重
        new_conv.weight.data = conv_layer.weight.data[keep_indices]
        if conv_layer.bias is not None:
            new_conv.bias.data = conv_layer.bias.data[keep_indices]
        
        # 同步修剪下一层的输入通道
        if next_conv_layer is not None:
            new_next_conv = nn.Conv2d(
                in_channels=num_keep,
                out_channels=next_conv_layer.out_channels,
                kernel_size=next_conv_layer.kernel_size,
                stride=next_conv_layer.stride,
                padding=next_conv_layer.padding,
                bias=next_conv_layer.bias is not None
            )
            new_next_conv.weight.data = next_conv_layer.weight.data[:, keep_indices]
            if next_conv_layer.bias is not None:
                new_next_conv.bias.data = next_conv_layer.bias.data
            return new_conv, new_next_conv, keep_indices
        
        return new_conv, None, keep_indices
    
    def prune_model(self, prune_config):
        """
        剪枝整个模型
        
        参数:
            prune_config: 字典，{layer_name: prune_ratio}
        """
        # 这里需要根据具体模型架构实现
        # 简化示例：假设模型是Sequential结构
        new_layers = []
        prev_keep_indices = None
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                ratio = prune_config.get(name, 0.0)
                # 找到对应的BN层
                bn_name = name.replace('conv', 'bn')  # 简化假设
                bn_module = dict(self.model.named_modules()).get(bn_name)
                
                # 实际剪枝操作...
                pass
        
        return self.model
```

## 58.3 模型量化：用更少位数存储

### 58.3.1 量化的基本原理

**费曼比喻：量化就像压缩图片**

想象你有一张照片，原始格式是RAW格式（每像素48位），文件巨大。你可以：
- 转成JPEG（有损压缩，文件小10倍，肉眼几乎看不出差别）
- 进一步降低分辨率（节省更多空间）
- 转成黑白（只用1位表示每个像素）

模型量化就是类似的"数据压缩"：把模型权重和激活值从32位浮点数（FP32）转成8位整数（INT8）甚至4位（INT4），大幅减少存储和计算量。

**数学原理：线性量化**

给定一个浮点值$x$，量化到$b$位整数的公式为：

$$x_q = \text{round}\left(\frac{x - z}{s}\right)$$

其中：
- $s$是**缩放因子 (scale)**：$s = \frac{r_{max} - r_{min}}{2^b - 1}$
- $z$是**零点 (zero-point)**：$z = \text{round}\left(r_{min} / s\right)$
- $x_q$是量化后的整数值

反量化（还原近似值）：

$$\hat{x} = s \cdot (x_q - z)$$

**量化误差**：

$$\epsilon = x - \hat{x}$$

### 58.3.2 对称量化 vs 非对称量化

#### 对称量化 (Symmetric Quantization)

假设权重分布关于0对称，使用对称映射：

$$x_q = \text{round}\left(\frac{x}{s}\right)$$

其中$s = \frac{\max(|x|)}{2^{b-1} - 1}$（对于INT8）。

**优点**：
- 零点为0，计算简单
- 硬件实现高效

**缺点**：
- 如果分布不关于0对称，会浪费动态范围

#### 非对称量化 (Asymmetric Quantization)

考虑实际的$[r_{min}, r_{max}]$范围：

$$s = \frac{r_{max} - r_{min}}{2^b - 1}, \quad z = \text{round}\left(r_{min} / s\right)$$

**优点**：
- 充分利用整个整数范围
- 适合ReLU输出（总是非负）

### 58.3.3 权重量化 vs 激活量化

| 类型 | 目标 | 特点 | 挑战 |
|------|------|------|------|
| **权重量化** | 模型参数 | 静态，只量化一次 | 容易，精度损失小 |
| **激活量化** | 特征图 | 动态，每个batch不同 | 需要校准数据 |
| **全量化** | 两者都量化 | 最大加速 | 精度损失较大 |

表58.4：不同量化目标的对比

### 58.3.4 后训练量化 (PTQ) vs 量化感知训练 (QAT)

#### 后训练量化 (Post-Training Quantization, PTQ)

**流程**：训练好的FP32模型 → 统计权重/激活分布 → 确定量化参数 → 直接量化

**优点**：
- 无需重新训练，速度快
- 只需少量校准数据

**缺点**：
- 对于极低精度（如INT4），精度损失较大

**关键步骤——校准**：

```python
def calibrate_activation_ranges(model, dataloader, num_batches=100):
    """
    校准激活值的动态范围
    
    费曼比喻：就像试衣时量尺寸，确定衣服要多大
    """
    activation_ranges = {}
    hooks = []
    
    def get_range(name):
        def hook(module, input, output):
            if name not in activation_ranges:
                activation_ranges[name] = {'min': float('inf'), 'max': float('-inf')}
            activation_ranges[name]['min'] = min(
                activation_ranges[name]['min'], 
                output.min().item()
            )
            activation_ranges[name]['max'] = max(
                activation_ranges[name]['max'], 
                output.max().item()
            )
        return hook
    
    # 注册hook
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(get_range(name)))
    
    # 前向传播收集统计信息
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            _ = model(data)
    
    # 移除hooks
    for hook in hooks:
        hook.remove()
    
    return activation_ranges
```

#### 量化感知训练 (Quantization-Aware Training, QAT)

**核心思想**：在训练过程中模拟量化效果，让模型学习适应量化误差。

**模拟量化 (Fake Quantization)**：

```
前向传播:  w_q = fake_quantize(w)  # 模拟量化效果
反向传播:  梯度直通估计器(STE)
          ∂L/∂w ≈ ∂L/∂w_q
```

**Straight-Through Estimator (STE)**：

量化函数$q(x)$不可微，反向传播时梯度为0。STE假设：

$$\frac{\partial q(x)}{\partial x} \approx 1$$

这让梯度能够"穿透"量化层。

**流程**：

```python
# 伪代码：QAT流程
model = load_pretrained_model()
model = prepare_for_qat(model)  # 插入FakeQuantize层

for epoch in range(num_epochs):
    for data, target in dataloader:
        output = model(data)  # 前向：模拟量化
        loss = criterion(output, target)
        loss.backward()       # 反向：STE估计梯度
        optimizer.step()
```

**PTQ vs QAT对比**：

| 特性 | PTQ | QAT |
|------|-----|-----|
| 是否需要重新训练 | 否 | 是 |
| 所需数据 | 少（校准） | 多（训练） |
| 时间成本 | 低 | 高 |
| INT8精度 | 通常足够 | 更好 |
| INT4精度 | 可能不足 | 推荐 |

表58.5：PTQ与QAT的对比

### 58.3.5 实战：INT8量化实现

```python
"""
模型量化完整实现：PTQ和QAT
"""
import torch
import torch.nn as nn
import numpy as np


class QuantizationConfig:
    """量化配置"""
    def __init__(self, num_bits=8, symmetric=True, per_channel=False):
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.qmin = -(2 ** (num_bits - 1)) if symmetric else 0
        self.qmax = (2 ** (num_bits - 1)) - 1 if symmetric else (2 ** num_bits) - 1


class FakeQuantize(nn.Module):
    """
    模拟量化层（用于QAT）
    
    费曼比喻：就像在正式压缩前试穿，看看效果如何
    """
    
    def __init__(self, config: QuantizationConfig):
        super().__init__()
        self.config = config
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.zero_point = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.observer_enabled = True
        self.fake_quant_enabled = True
        
    def forward(self, x):
        if self.observer_enabled:
            # 观察并更新统计信息
            self._update_observer(x)
        
        if self.fake_quant_enabled:
            # 模拟量化效果
            return self._fake_quantize(x)
        return x
    
    def _update_observer(self, x):
        """更新观察到的数值范围"""
        if self.config.symmetric:
            amax = torch.max(torch.abs(x))
            self.scale.data = amax / (2 ** (self.config.num_bits - 1) - 1)
            self.zero_point.data = torch.tensor(0.0)
        else:
            xmin, xmax = x.min(), x.max()
            self.scale.data = (xmax - xmin) / (2 ** self.config.num_bits - 1)
            self.zero_point.data = torch.round(xmin / self.scale)
    
    def _fake_quantize(self, x):
        """模拟量化（STE梯度）"""
        # 量化
        x_int = torch.round(x / self.scale + self.zero_point)
        x_int = torch.clamp(x_int, self.config.qmin, self.config.qmax)
        
        # 反量化
        x_quant = (x_int - self.zero_point) * self.scale
        
        # STE: 返回量化值，但梯度如同没有量化
        return x + (x_quant - x).detach()


class QuantizedLinear(nn.Module):
    """
    量化全连接层
    """
    
    def __init__(self, in_features, out_features, bias=True, config=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 原始浮点权重
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 量化配置
        self.config = config or QuantizationConfig()
        
        # 量化参数（通过校准或训练获得）
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_zero_point', torch.tensor(0.0))
        self.register_buffer('input_scale', torch.tensor(1.0))
        self.register_buffer('input_zero_point', torch.tensor(0.0))
        
    def quantize_weight(self):
        """量化权重"""
        w = self.weight.detach()
        
        if self.config.symmetric:
            amax = torch.max(torch.abs(w))
            self.weight_scale = amax / (2 ** (self.config.num_bits - 1) - 1)
            self.weight_zero_point = torch.tensor(0.0)
        else:
            wmin, wmax = w.min(), w.max()
            self.weight_scale = (wmax - wmin) / (2 ** self.config.num_bits - 1)
            self.weight_zero_point = torch.round(wmin / self.weight_scale)
        
        # 量化并反量化（模拟）
        w_int = torch.round(w / self.weight_scale + self.weight_zero_point)
        w_int = torch.clamp(w_int, self.config.qmin, self.config.qmax)
        w_quant = (w_int - self.weight_zero_point) * self.weight_scale
        
        return w_quant
    
    def forward(self, x):
        # 使用量化权重进行计算
        if self.training:
            # 训练时使用fake quantization
            weight_quant = self.quantize_weight()
            return F.linear(x, weight_quant, self.bias)
        else:
            # 推理时可以进行真正的整数运算
            # 这里简化处理，实际应使用INT8 GEMM
            return F.linear(x, self.quantize_weight(), self.bias)


class PostTrainingQuantizer:
    """
    后训练量化器
    
    费曼比喻：衣服做好了再改尺寸，需要一些调整
    """
    
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or QuantizationConfig()
        self.calibration_data = []
        
    def calibrate(self, dataloader, num_batches=100):
        """
        校准：收集激活值的统计信息
        
        费曼比喻：就像裁缝量体裁衣前要先量身材
        """
        print("开始校准...")
        
        # 注册钩子收集统计信息
        activation_stats = {}
        
        def get_stats_hook(name):
            def hook(module, input, output):
                if name not in activation_stats:
                    activation_stats[name] = {'min_vals': [], 'max_vals': []}
                activation_stats[name]['min_vals'].append(output.min().item())
                activation_stats[name]['max_vals'].append(output.max().item())
            return hook
        
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(get_stats_hook(name)))
        
        # 收集数据
        self.model.eval()
        with torch.no_grad():
            for i, (data, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                _ = self.model(data)
                if (i + 1) % 10 == 0:
                    print(f"  已处理 {i+1}/{num_batches} 批量")
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 计算量化参数
        self.quantization_params = {}
        for name, stats in activation_stats.items():
            rmin, rmax = min(stats['min_vals']), max(stats['max_vals'])
            
            if self.config.symmetric:
                amax = max(abs(rmin), abs(rmax))
                scale = amax / (2 ** (self.config.num_bits - 1) - 1)
                zero_point = 0
            else:
                scale = (rmax - rmin) / (2 ** self.config.num_bits - 1)
                zero_point = round(rmin / scale)
            
            self.quantization_params[name] = {
                'scale': scale,
                'zero_point': zero_point,
                'rmin': rmin,
                'rmax': rmax
            }
        
        print(f"校准完成，收集了 {len(self.quantization_params)} 层的统计信息")
        return self.quantization_params
    
    def quantize_model(self):
        """
        应用量化到模型
        
        注意：实际生产环境应导出到TensorRT/TFLite等推理引擎
        """
        quantized_model = copy.deepcopy(self.model)
        
        for name, module in quantized_model.named_modules():
            if isinstance(module, nn.Linear) and name in self.quantization_params:
                params = self.quantization_params[name]
                
                # 量化权重
                w = module.weight.data
                w_int = torch.round(w / params['scale'] + params['zero_point'])
                w_int = torch.clamp(w_int, self.config.qmin, self.config.qmax)
                
                # 存储量化权重（实际应使用int8类型）
                module.register_buffer('weight_int8', w_int.to(torch.int8))
                module.register_buffer('quant_scale', torch.tensor(params['scale']))
                module.register_buffer('quant_zero_point', torch.tensor(params['zero_point']))
        
        return quantized_model
    
    def evaluate_quantized(self, dataloader):
        """评估量化模型精度"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy


def compare_precision(model, test_loader, bit_configs=[32, 8, 4]):
    """
    比较不同精度下的模型表现
    """
    results = []
    
    # FP32基准
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    fp32_acc = 100. * correct / total
    results.append({'bits': 32, 'accuracy': fp32_acc, 'size_ratio': 1.0})
    
    print(f"FP32 精度: {fp32_acc:.2f}%")
    
    # 不同位数量化
    for bits in bit_configs[1:]:
        config = QuantizationConfig(num_bits=bits)
        quantizer = PostTrainingQuantizer(model, config)
        quantizer.calibrate(test_loader, num_batches=50)
        
        acc = quantizer.evaluate_quantized(test_loader)
        size_ratio = bits / 32
        
        results.append({
            'bits': bits,
            'accuracy': acc,
            'size_ratio': size_ratio,
            'degradation': fp32_acc - acc
        })
        
        print(f"INT{bits} 精度: {acc:.2f}% (下降: {fp32_acc - acc:.2f}%), "
              f"大小比例: {size_ratio:.2%}")
    
    return results
```

## 58.4 知识蒸馏：师承名师

### 58.4.1 知识蒸馏的基本框架

**费曼比喻：知识蒸馏就像老教授带学生**

想象一位博学的老教授（大模型）毕生研究某个领域，积累了深厚的理解。现在他要带一位年轻学生（小模型）。最好的教学方式不是让他从零开始读所有书籍，而是：
- **传授思维方法**：不仅告诉他答案，还解释"为什么其他选项不对"
- **分享概率直觉**："这道题A选项有80%可能是对的，B选项15%，C选项5%"
- **揭示类间关系**："狮子和老虎更像，和桌子完全不像"

这就是知识蒸馏的核心思想：**让小模型学习大模型的"软标签"（概率分布），而不仅仅是正确的类别**。

### 58.4.2 温度参数：软化概率分布

标准softmax：

$$q_i = \frac{\exp(z_i)}{
\sum_j \exp(z_j)}$$

**带温度T的softmax**：

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

**温度的作用**：

| T值 | 效果 | 比喻 |
|-----|------|------|
| T → 0 | 接近one-hot（最确定） | 教授直接说"答案是A" |
| T = 1 | 标准softmax | 教授给出概率估计 |
| T → ∞ | 均匀分布（最"软"） | 教授说"每个选项都有道理" |

表58.6：温度参数对软标签的影响

**为什么需要高温？**

高温能保留类间的相对信息。例如，一个图像分类为"狗"，但教师模型可能给出：
- **硬标签**：[1, 0, 0, 0]（只告诉你是狗）
- **低温软标签**：[0.9, 0.05, 0.03, 0.02]（主要是狗）
- **高温软标签**：[0.5, 0.3, 0.15, 0.05]（揭示狗和狼、狐狸的相似性）

高温软标签包含更多"暗知识"——关于样本如何与其他类别相关的信息。

### 58.4.3 蒸馏损失函数

**KL散度 (Kullback-Leibler Divergence)**：

$$\mathcal{L}_{KD} = T^2 \cdot KL(p^{teacher} \| p^{student}) = T^2 \sum_i p_i^{teacher} \log \frac{p_i^{teacher}}{p_i^{student}}$$

因子$T^2$是为了平衡软损失和硬损失的梯度幅度。

**联合损失**：

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{CE}(y_{hard}, p_{student}) + (1 - \alpha) \cdot \mathcal{L}_{KD}$$

其中：
- $\mathcal{L}_{CE}$是学生输出与真实标签的交叉熵
- $\alpha$是平衡系数，通常设为0.5或0.7

### 58.4.4 特征蒸馏：从中间层学习

除了输出层，还可以让学生学习教师的中间表示：

$$\mathcal{L}_{feature} = \| f_{teacher}(\mathbf{x}) - f_{student}(\mathbf{x})\|^2$$

**适配器 (Adaptation Layer)**：当教师和学生的特征维度不同时，需要引入一个可学习的适配层：

$$f'_{student} = W_{adapt} \cdot f_{student} + b_{adapt}$$

然后最小化：

$$\mathcal{L}_{feature} = \| f_{teacher} - f'_{student}\|^2$$

### 58.4.5 实战：知识蒸馏完整实现

```python
"""
知识蒸馏完整实现：教师-学生训练框架
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class DistillationLoss(nn.Module):
    """
    知识蒸馏损失
    
    包含两部分：
    1. 硬标签损失（与真实标签的交叉熵）
    2. 软标签损失（与教师输出的KL散度）
    
    费曼比喻：既看标准答案，也学习老师的解题思路
    """
    
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, targets):
        """
        参数:
            student_logits: 学生模型输出 (batch_size, num_classes)
            teacher_logits: 教师模型输出 (batch_size, num_classes)
            targets: 真实标签 (batch_size,)
        """
        # 软标签损失（使用高温softmax）
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # KL散度，乘以T^2来平衡梯度
        soft_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # 硬标签损失
        hard_loss = self.ce(student_logits, targets)
        
        # 联合损失
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return total_loss, hard_loss, soft_loss


class FeatureDistillationLoss(nn.Module):
    """
    特征蒸馏损失：让学生学习教师的中间表示
    
    费曼比喻：不仅学习最终答案，还学习老师的思考过程
    """
    
    def __init__(self, mode='mse', margin=0.0):
        super().__init__()
        self.mode = mode
        self.margin = margin
        
    def forward(self, student_features, teacher_features, adaptation_layer=None):
        """
        参数:
            student_features: 学生特征
            teacher_features: 教师特征
            adaptation_layer: 适配层（当维度不同时使用）
        """
        if adaptation_layer is not None:
            student_features = adaptation_layer(student_features)
        
        # 特征对齐（可能需要归一化）
        student_features = F.normalize(student_features, p=2, dim=1)
        teacher_features = F.normalize(teacher_features, p=2, dim=1)
        
        if self.mode == 'mse':
            loss = F.mse_loss(student_features, teacher_features)
        elif self.mode == 'cosine':
            # 余弦相似度损失
            loss = 1 - F.cosine_similarity(student_features, teacher_features, dim=1).mean()
        elif self.mode == 'attention':
            # 注意力转移（FitNets）
            # 计算特征图的重要性映射
            student_attention = torch.sum(torch.abs(student_features), dim=1, keepdim=True)
            teacher_attention = torch.sum(torch.abs(teacher_features), dim=1, keepdim=True)
            loss = F.mse_loss(student_attention, teacher_attention)
        else:
            loss = F.mse_loss(student_features, teacher_features)
        
        return loss


class KnowledgeDistiller:
    """
    知识蒸馏训练器
    
    费曼比喻：老教授（教师）指导学生（学生）的过程
    """
    
    def __init__(self, teacher_model, student_model, device='cuda'):
        self.device = device
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        
        # 教师模型固定，不参与训练
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.distillation_loss_fn = None
        self.feature_loss_fn = None
        self.adaptation_layers = {}
        
    def setup_losses(self, temperature=4.0, alpha=0.5, 
                     use_feature_distill=False, feature_weight=0.1):
        """配置损失函数"""
        self.distillation_loss_fn = DistillationLoss(temperature, alpha)
        self.use_feature_distill = use_feature_distill
        self.feature_weight = feature_weight
        
        if use_feature_distill:
            self.feature_loss_fn = FeatureDistillationLoss(mode='mse')
            # 为特征维度不匹配的情况创建适配层
            self._setup_adaptation_layers()
    
    def _setup_adaptation_layers(self):
        """
        设置特征适配层
        假设教师和学生有对应的特征提取层
        """
        # 这里需要根据具体模型架构实现
        # 简化为假设特征维度已知的情况
        pass
    
    def train_epoch(self, train_loader, optimizer, epoch):
        """训练一个epoch"""
        self.student_model.train()
        total_loss = 0
        total_hard_loss = 0
        total_soft_loss = 0
        total_feature_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # 教师前向（无梯度）
            with torch.no_grad():
                if self.use_feature_distill:
                    teacher_output, teacher_features = self.teacher_model(data, return_features=True)
                else:
                    teacher_output = self.teacher_model(data)
            
            # 学生前向
            if self.use_feature_distill:
                student_output, student_features = self.student_model(data, return_features=True)
            else:
                student_output = self.student_model(data)
            
            # 蒸馏损失
            loss, hard_loss, soft_loss = self.distillation_loss_fn(
                student_output, teacher_output, target
            )
            
            # 特征蒸馏损失
            feature_loss = 0
            if self.use_feature_distill and self.feature_loss_fn is not None:
                # 假设有多个特征层需要对齐
                for sf, tf in zip(student_features, teacher_features):
                    feature_loss += self.feature_loss_fn(sf, tf)
                feature_loss = feature_loss / len(student_features)
                loss = loss + self.feature_weight * feature_loss
            
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_hard_loss += hard_loss.item()
            total_soft_loss += soft_loss.item()
            if self.use_feature_distill:
                total_feature_loss += feature_loss.item()
            
            _, predicted = student_output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f'  Batch {batch_idx+1}/{len(train_loader)}: '
                      f'Loss={loss.item():.4f}, Acc={100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100. * correct / total
        
        print(f'Epoch {epoch} 完成: Avg Loss={avg_loss:.4f}, Avg Acc={avg_acc:.2f}%')
        print(f'  Hard Loss={total_hard_loss/len(train_loader):.4f}, '
              f'Soft Loss={total_soft_loss/len(train_loader):.4f}')
        
        return avg_loss, avg_acc
    
    def evaluate(self, test_loader):
        """评估学生模型"""
        self.student_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.student_model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def compare_with_baseline(self, test_loader, num_epochs, lr, 
                              train_loader=None):
        """
        对比：蒸馏训练 vs 从零训练
        
        费曼比喻：有老师教的学生 vs 自学成才的学生
        """
        print("="*70)
        print("对比实验：知识蒸馏 vs 基线训练")
        print("="*70)
        
        # 1. 评估教师模型
        self.teacher_model.eval()
        teacher_correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.teacher_model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                teacher_correct += predicted.eq(target).sum().item()
        
        teacher_acc = 100. * teacher_correct / total
        print(f"教师模型参数量: {sum(p.numel() for p in self.teacher_model.parameters()):,}")
        print(f"教师模型精度: {teacher_acc:.2f}%")
        
        # 2. 蒸馏训练
        print("\n--- 知识蒸馏训练 ---")
        student_with_distill = copy.deepcopy(self.student_model)
        self.student_model = student_with_distill
        
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            self.train_epoch(train_loader, optimizer, epoch+1)
        
        distill_acc = self.evaluate(test_loader)
        
        # 3. 基线训练（从头训练学生模型）
        print("\n--- 基线训练（无蒸馏）---")
        student_baseline = copy.deepcopy(student_with_distill)
        baseline_optimizer = torch.optim.Adam(student_baseline.parameters(), lr=lr)
        
        student_baseline.train()
        for epoch in range(num_epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                baseline_optimizer.zero_grad()
                output = student_baseline(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                baseline_optimizer.step()
        
        student_baseline.eval()
        baseline_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = student_baseline(data)
                _, predicted = output.max(1)
                baseline_correct += predicted.eq(target).sum().item()
        
        baseline_acc = 100. * baseline_correct / total
        
        student_params = sum(p.numel() for p in student_baseline.parameters())
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        
        print("\n" + "="*70)
        print("对比结果")
        print("="*70)
        print(f"教师模型: {teacher_params:,} 参数, {teacher_acc:.2f}% 精度")
        print(f"学生模型（蒸馏）: {student_params:,} 参数 ({student_params/teacher_params:.1%}), "
              f"{distill_acc:.2f}% 精度")
        print(f"学生模型（基线）: {student_params:,} 参数, {baseline_acc:.2f}% 精度")
        print(f"蒸馏收益: +{distill_acc - baseline_acc:.2f}% 精度")
        print("="*70)
        
        return {
            'teacher_acc': teacher_acc,
            'distill_acc': distill_acc,
            'baseline_acc': baseline_acc,
            'compression_ratio': teacher_params / student_params
        }


# 示例：用于MNIST的教师和学生模型
class TeacherNet(nn.Module):
    """大模型（教师）"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, return_features=False):
        feats = self.features(x)
        x = feats.view(feats.size(0), -1)
        out = self.classifier(x)
        if return_features:
            return out, [feats]
        return out


class StudentNet(nn.Module):
    """小模型（学生）"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x, return_features=False):
        feats = self.features(x)
        x = feats.view(feats.size(0), -1)
        out = self.classifier(x)
        if return_features:
            return out, [feats]
        return out
```

## 58.5 高效神经网络架构

### 58.5.1 设计原则：效率vs精度权衡

传统CNN设计追求准确率，而移动端模型需要在有限计算预算下最大化效率。这引出了几个关键设计原则：

1. **减少乘法操作**：乘法是计算最昂贵的基本运算
2. **利用分组/分离卷积**：将大卷积分解为多个小操作
3. **早期降采样**：快速减小特征图空间维度
4. **平衡深度和宽度**：深层网络学习复杂模式，宽网络捕捉更多特征

### 58.5.2 MobileNet：深度可分离卷积的革命

2017年Google提出的MobileNet引入了**深度可分离卷积 (Depthwise Separable Convolution)**，成为移动端视觉的基石。

**费曼比喻：深度可分离卷积就像分工协作的工厂**

想象一个生产彩色玻璃窗户的工厂：
- **传统卷积**：每个工人要同时负责切割形状和染色（一步完成所有工作）
- **深度可分离卷积**：
  - **Depthwise**：一组工人只负责把玻璃切成各种形状（每个输入通道单独处理空间信息）
  - **Pointwise**：另一组工人专门负责给玻璃上色（1×1卷积混合通道信息）

这种分工让每个工人更专业化，大大提高了效率。

**数学分析**：

标准卷积的计算量：
$$\text{FLOPs}_{\text{std}} = D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$$

深度可分离卷积的计算量：
$$\text{FLOPs}_{\text{sep}} = D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F$$

压缩比：
$$\frac{\text{FLOPs}_{\text{sep}}}{\text{FLOPs}_{\text{std}}} = \frac{1}{N} + \frac{1}{D_K^2}$$

通常$N$（输出通道）远大于1，所以计算量大幅减少（典型情况下减少8-9倍）。

**MobileNet架构**：

```python
"""
MobileNet深度可分离卷积实现
"""
import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积 = Depthwise + Pointwise
    
    费曼比喻：先分别处理每个通道的空间信息，再混合通道信息
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1, bias=False):
        super().__init__()
        
        # Depthwise：每个输入通道单独卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, 
            groups=in_channels,  # groups=in_channels 实现depthwise
            bias=bias
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Pointwise：1x1卷积混合通道
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, padding=0, bias=bias
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x


class MobileNetV1(nn.Module):
    """
    MobileNet V1 完整实现
    
    核心创新：用深度可分离卷积替代所有标准卷积
    """
    
    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        
        def make_divisible(v, divisor=8):
            return int((v + divisor // 2) // divisor * divisor)
        
        # 宽度乘数：调整每层通道数
        def conv_bn(inp, oup, stride):
            oup = make_divisible(oup * width_mult)
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        
        def conv_dw(inp, oup, stride):
            inp = make_divisible(inp * width_mult)
            oup = make_divisible(oup * width_mult)
            return nn.Sequential(
                # Depthwise
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                # Pointwise
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        
        self.model = nn.Sequential(
            conv_bn(3, 32, 2),  # 输入224x224，输出112x112
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),  # 56x56
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),  # 28x28
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),  # 14x14
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),  # 5个重复的512通道层
            conv_dw(512, 1024, 2),  # 7x7
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(make_divisible(1024 * width_mult), num_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def compare_conv_flops():
    """
    对比标准卷积和深度可分离卷积的计算量
    """
    import torch
    
    # 假设输入输出配置
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    h, w = 56, 56
    
    # 标准卷积FLOPs
    std_flops = kernel_size * kernel_size * in_channels * out_channels * h * w
    
    # 深度可分离卷积FLOPs
    dw_flops = kernel_size * kernel_size * in_channels * h * w  # depthwise
    pw_flops = in_channels * out_channels * h * w  # pointwise
    sep_flops = dw_flops + pw_flops
    
    print("="*60)
    print("卷积类型计算量对比")
    print("="*60)
    print(f"输入通道: {in_channels}, 输出通道: {out_channels}")
    print(f"特征图尺寸: {h}x{w}, 卷积核: {kernel_size}x{kernel_size}")
    print("-"*60)
    print(f"标准卷积 FLOPs: {std_flops:,}")
    print(f"深度可分离 FLOPs: {sep_flops:,}")
    print(f"计算量节省: {(1 - sep_flops/std_flops)*100:.1f}%")
    print(f"压缩比: {std_flops/sep_flops:.2f}x")
    print("="*60)
    
    return std_flops, sep_flops
```

### 58.5.3 EfficientNet：复合缩放的艺术

2019年Google提出的EfficientNet回答了这样一个问题：**当有更多计算预算时，应该增加网络的深度、宽度还是输入分辨率？**

**费曼比喻：复合缩放就像调配披萨配方**

想象你要做更大份的披萨喂饱更多人，你有三个选择：
- **增加深度（层数）**：做更多层披萨叠加 → 可以学习更复杂的"味道层次"
- **增加宽度（通道数）**：每层放更多配料 → 可以捕捉更多"风味特征"
- **增加分辨率**：用更大的饼底 → 可以看到更细的"纹理细节"

EfficientNet发现：三者同时适度增加，比单独大幅增加某一个效果更好！

**复合缩放公式**：

给定复合系数$\phi$，缩放三个维度：

$$\begin{aligned}
d &= \alpha^{\phi} \quad \text{(深度)} \\
w &= \beta^{\phi} \quad \text{(宽度)} \\
r &= \gamma^{\phi} \quad \text{(分辨率)}
\end{aligned}$$

约束条件（FLOPs约正比于深度×宽度²×分辨率²）：

$$\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$$

通过网格搜索，EfficientNet-B0的最优系数为：
$$\alpha = 1.2, \quad \beta = 1.1, \quad \gamma = 1.15$$

**不同缩放策略的对比**：

| 模型 | 缩放策略 | Top-1精度 | 参数量 | FLOPs |
|------|----------|-----------|--------|-------|
| Baseline | - | 77.1% | 5.3M | 0.39B |
| + Depth | d=2 | 78.3% | 7.0M | 0.86B |
| + Width | w=2 | 78.4% | 21.5M | 1.12B |
| + Resolution | r=2 | 79.1% | 5.3M | 1.57B |
| **Compound** | d,w,r | **80.0%** | 10.1M | 1.81B |

表58.7：不同缩放策略的效果对比

**EfficientNet块结构**：

```python
"""
EfficientNet核心模块：MBConv (Mobile Inverted Bottleneck)
"""
import torch
import torch.nn as nn
import math


class Swish(nn.Module):
    """Swish激活函数：x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation模块
    
    学习通道间的注意力权重
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck卷积
    
    EfficientNet的核心构建块
    
    结构：
    1. 1x1扩展卷积（增加通道）
    2. Depthwise 3x3卷积（空间特征）
    3. SE注意力模块
    4. 1x1投影卷积（减少通道）
    5. 残差连接（如果stride=1且输入输出通道相同）
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, expand_ratio=6, se_ratio=0.25, drop_rate=0):
        super().__init__()
        
        self.use_residual = stride == 1 and in_channels == out_channels
        self.drop_rate = drop_rate
        
        hidden_dim = int(in_channels * expand_ratio)
        
        layers = []
        
        # 扩展（只在expand_ratio > 1时）
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        ])
        
        # SE模块
        if se_ratio > 0:
            se_channels = max(1, int(hidden_dim * se_ratio))
            layers.append(SEBlock(hidden_dim, reduction=hidden_dim // se_channels))
        
        # 投影
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class EfficientNet(nn.Module):
    """
    EfficientNet简化实现
    
    通过复合系数phi可以生成B0-B7系列模型
    """
    
    # EfficientNet-B0配置
    CONFIG = [
        # (expand_ratio, channels, repeats, stride, kernel_size)
        (1, 16, 1, 1, 3),
        (6, 24, 2, 2, 3),
        (6, 40, 2, 2, 5),
        (6, 80, 3, 2, 3),
        (6, 112, 3, 1, 5),
        (6, 192, 4, 2, 5),
        (6, 320, 1, 1, 3),
    ]
    
    def __init__(self, num_classes=1000, width_mult=1.0, depth_mult=1.0, 
                 resolution=224, dropout_rate=0.2):
        super().__init__()
        
        # 初始卷积
        out_channels = self._round_channels(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            Swish()
        )
        
        # MBConv块
        self.blocks = nn.ModuleList()
        in_channels = out_channels
        
        for expand_ratio, channels, repeats, stride, kernel_size in self.CONFIG:
            out_channels = self._round_channels(channels, width_mult)
            num_repeats = self._round_repeats(repeats, depth_mult)
            
            for i in range(num_repeats):
                s = stride if i == 0 else 1
                self.blocks.append(
                    MBConv(in_channels, out_channels, kernel_size, 
                          s, expand_ratio)
                )
                in_channels = out_channels
        
        # 头部
        head_channels = self._round_channels(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Linear(head_channels, num_classes)
        
        self._initialize_weights()
        
    def _round_channels(self, channels, width_mult, divisor=8):
        """按宽度乘数缩放通道数"""
        channels *= width_mult
        new_channels = max(divisor, int(channels + divisor / 2) // divisor * divisor)
        if new_channels < 0.9 * channels:
            new_channels += divisor
        return new_channels
    
    def _round_repeats(self, repeats, depth_mult):
        """按深度乘数缩放重复次数"""
        return int(math.ceil(repeats * depth_mult))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_efficientnet_version(version='b0'):
    """
    创建不同版本的EfficientNet
    
    phi: 复合系数，控制深度、宽度、分辨率的缩放
    """
    configs = {
        'b0': {'width_mult': 1.0, 'depth_mult': 1.0, 'resolution': 224, 'dropout': 0.2},
        'b1': {'width_mult': 1.0, 'depth_mult': 1.1, 'resolution': 240, 'dropout': 0.2},
        'b2': {'width_mult': 1.1, 'depth_mult': 1.2, 'resolution': 260, 'dropout': 0.3},
        'b3': {'width_mult': 1.2, 'depth_mult': 1.4, 'resolution': 300, 'dropout': 0.3},
        'b4': {'width_mult': 1.4, 'depth_mult': 1.8, 'resolution': 380, 'dropout': 0.4},
        'b5': {'width_mult': 1.6, 'depth_mult': 2.2, 'resolution': 456, 'dropout': 0.4},
        'b6': {'width_mult': 1.8, 'depth_mult': 2.6, 'resolution': 528, 'dropout': 0.5},
        'b7': {'width_mult': 2.0, 'depth_mult': 3.1, 'resolution': 600, 'dropout': 0.5},
    }
    
    config = configs.get(version, configs['b0'])
    return EfficientNet(
        width_mult=config['width_mult'],
        depth_mult=config['depth_mult'],
        resolution=config['resolution'],
        dropout_rate=config['dropout']
    )
```

## 58.6 边缘部署实战

### 58.6.1 ONNX：跨框架的桥梁

**费曼比喻：ONNX就像音乐的五线谱**

想象不同乐器（PyTorch、TensorFlow、MXNet等）说不同的语言。ONNX就像五线谱——一种通用的记谱法，让任何乐器都能演奏同一首曲子。

ONNX (Open Neural Network Exchange) 定义了一种标准的模型表示格式，使得模型可以在不同框架间自由转换。

**导出流程**：

```python
"""
ONNX导出与优化
"""
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np


def export_to_onnx(model, dummy_input, output_path, input_names=None, 
                   output_names=None, dynamic_axes=None):
    """
    将PyTorch模型导出为ONNX格式
    
    费曼比喻：把你的乐谱翻译成世界通用的五线谱
    """
    model.eval()
    
    input_names = input_names or ['input']
    output_names = output_names or ['output']
    
    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,  # 支持动态batch size
        opset_version=11,
        do_constant_folding=True,  # 优化常量表达式
        verbose=False
    )
    
    print(f"模型已导出到: {output_path}")
    
    # 验证模型
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX模型验证通过！")
    
    return output_path


def optimize_onnx_model(input_path, output_path):
    """
    使用ONNX优化工具优化模型
    """
    import onnx.optimizer
    
    # 加载模型
    model = onnx.load(input_path)
    
    # 可用的优化passes
    passes = [
        'eliminate_identity',
        'fuse_consecutive_transposes',
        'fuse_transpose_into_gemm',
        'extract_constant_to_initializer',
        'fuse_add_bias_into_conv',
        'fuse_bn_into_conv',
    ]
    
    # 应用优化
    optimized_model = onnx.optimizer.optimize(model, passes)
    
    # 保存
    onnx.save(optimized_model, output_path)
    print(f"优化后的模型已保存到: {output_path}")
    
    return output_path


def benchmark_onnx(onnx_path, input_shape, num_runs=100):
    """
    基准测试ONNX模型
    """
    # 创建推理会话
    session = ort.InferenceSession(onnx_path)
    
    # 准备输入
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # 热身
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_input})
    
    # 正式测试
    import time
    start = time.perf_counter()
    for _ in range(num_runs):
        outputs = session.run(None, {input_name: dummy_input})
    elapsed = time.perf_counter() - start
    
    avg_latency = (elapsed / num_runs) * 1000  # ms
    throughput = num_runs / elapsed
    
    print(f"ONNX Runtime 性能:")
    print(f"  平均延迟: {avg_latency:.2f} ms")
    print(f"  吞吐量: {throughput:.2f} infer/sec")
    
    return avg_latency, throughput
```

### 58.6.2 TensorRT：NVIDIA GPU的加速神器

TensorRT是NVIDIA专门为深度学习推理优化的运行时库，可以：
- 层融合（Layer Fusion）：将多个层合并为单个核函数
- 精度校准：自动选择FP32/FP16/INT8
- 动态张量内存：减少内存占用
- Kernel自动调优：为特定GPU选择最优实现

```python
"""
TensorRT优化示例

注意：需要NVIDIA GPU和TensorRT库支持
"""
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("TensorRT未安装，跳过相关代码")


def build_tensorrt_engine(onnx_path, engine_path, fp16_mode=True, max_batch_size=1):
    """
    从ONNX构建TensorRT引擎
    
    费曼比喻：TensorRT就像给赛车专业调校，榨干GPU的每一滴性能
    """
    if not TENSORRT_AVAILABLE:
        print("TensorRT不可用")
        return None
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    # 解析ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 配置builder
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB工作空间
    
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
        print("启用FP16模式")
    
    # 构建引擎
    profile = builder.create_optimization_profile()
    # 设置输入形状范围（支持动态batch）
    input_name = network.get_input(0).name
    profile.set_shape(input_name, 
                     min=(1, 3, 224, 224),
                     opt=(max_batch_size, 3, 224, 224),
                     max=(max_batch_size, 3, 224, 224))
    config.add_optimization_profile(profile)
    
    engine = builder.build_engine(network, config)
    
    # 保存引擎
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"TensorRT引擎已保存到: {engine_path}")
    return engine


def infer_with_tensorrt(engine_path, input_data):
    """使用TensorRT引擎进行推理"""
    if not TENSORRT_AVAILABLE:
        return None
    
    logger = trt.Logger(trt.Logger.WARNING)
    
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # 分配GPU内存
    d_input = cuda.mem_alloc(input_data.nbytes)
    output = np.empty(engine.get_binding_shape(1), dtype=np.float32)
    d_output = cuda.mem_alloc(output.nbytes)
    
    # 数据传输和推理
    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_input, input_data, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], 
                            stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()
    
    return output
```

### 58.6.3 TensorFlow Lite：移动端部署

TFLite是Google专门为移动和嵌入式设备设计的轻量级推理框架。

```python
"""
TensorFlow Lite部署示例
"""
import tensorflow as tf
import numpy as np


def convert_to_tflite(saved_model_path, output_path, 
                      quantization_mode='dynamic'):
    """
    转换SavedModel为TFLite格式
    
    参数:
        quantization_mode: 'none', 'dynamic', 'float16', 'int8'
    
    费曼比喻：把你的专业相机照片转换成手机壁纸格式
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    
    if quantization_mode == 'dynamic':
        # 动态范围量化：仅权重量化
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("使用动态范围量化")
        
    elif quantization_mode == 'float16':
        # FP16量化
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("使用FP16量化")
        
    elif quantization_mode == 'int8':
        # 全整数量化（需要代表性数据集）
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # 定义代表性数据集生成器
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(1, 224, 224, 3).astype(np.float32)
                yield [data]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        print("使用INT8全整数量化")
    
    # 转换
    tflite_model = converter.convert()
    
    # 保存
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite模型已保存到: {output_path}")
    
    # 计算模型大小
    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"模型大小: {size_mb:.2f} MB")
    
    return output_path


def benchmark_tflite(tflite_path, input_shape, num_runs=100):
    """
    基准测试TFLite模型
    """
    # 加载模型
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # 获取输入输出详情
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 准备输入
    input_data = np.random.randn(*input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # 热身
    for _ in range(10):
        interpreter.invoke()
    
    # 正式测试
    import time
    start = time.perf_counter()
    for _ in range(num_runs):
        interpreter.invoke()
    elapsed = time.perf_counter() - start
    
    avg_latency = (elapsed / num_runs) * 1000  # ms
    throughput = num_runs / elapsed
    
    print(f"TFLite性能:")
    print(f"  平均延迟: {avg_latency:.2f} ms")
    print(f"  吞吐量: {throughput:.2f} infer/sec")
    
    return avg_latency, throughput


class EdgeDeploymentPipeline:
    """
    边缘部署完整流程
    
    费曼比喻：把你的AI模型打包成可以在任何设备上运行的App
    """
    
    def __init__(self, model, model_name='my_model'):
        self.model = model
        self.model_name = model_name
        
    def full_pipeline(self, dummy_input, target_platform='mobile'):
        """
        完整部署流程
        
        参数:
            target_platform: 'mobile', 'edge', 'server'
        """
        print("="*70)
        print("边缘部署流程")
        print("="*70)
        
        # Step 1: 导出ONNX
        print("\n[1/5] 导出ONNX格式...")
        onnx_path = f'{self.model_name}.onnx'
        export_to_onnx(self.model, dummy_input, onnx_path)
        
        # Step 2: 优化ONNX
        print("\n[2/5] 优化ONNX模型...")
        optimized_onnx = f'{self.model_name}_optimized.onnx'
        optimize_onnx_model(onnx_path, optimized_onnx)
        
        # Step 3: 根据目标平台选择部署方案
        if target_platform == 'server' and TENSORRT_AVAILABLE:
            print("\n[3/5] 构建TensorRT引擎...")
            engine_path = f'{self.model_name}.trt'
            build_tensorrt_engine(optimized_onnx, engine_path)
            
        elif target_platform == 'mobile':
            print("\n[3/5] 转换为TensorFlow Lite...")
            # 先导出为SavedModel
            import torch
            import torchvision
            # 这里需要PyTorch到TensorFlow的转换，或使用ONNX-TF
            print("（需要onnx-tf或其他转换工具）")
            
        # Step 4: 量化
        print("\n[4/5] 模型量化...")
        print("  - 权重量化: INT8")
        print("  - 激活量化: 动态范围")
        
        # Step 5: 基准测试
        print("\n[5/5] 性能基准测试...")
        input_shape = dummy_input.shape
        benchmark_onnx(optimized_onnx, input_shape)
        
        print("\n" + "="*70)
        print("部署完成！")
        print("="*70)


def compare_deployment_options():
    """
    对比不同部署方案
    """
    results = []
    
    print("="*70)
    print("边缘部署方案对比")
    print("="*70)
    
    options = [
        {'name': 'PyTorch (FP32)', 'format': 'PyTorch', 'size': '100%', 'speed': '1x'},
        {'name': 'ONNX Runtime (FP32)', 'format': 'ONNX', 'size': '100%', 'speed': '1.2x'},
        {'name': 'ONNX Runtime (INT8)', 'format': 'ONNX', 'size': '25%', 'speed': '2.5x'},
        {'name': 'TensorRT (FP16)', 'format': 'TensorRT', 'size': '50%', 'speed': '4x'},
        {'name': 'TensorRT (INT8)', 'format': 'TensorRT', 'size': '25%', 'speed': '8x'},
        {'name': 'TFLite (INT8)', 'format': 'TFLite', 'size': '25%', 'speed': '3x'},
    ]
    
    print(f"{'方案':<30} {'格式':<15} {'大小':<10} {'速度':<10}")
    print("-"*70)
    for opt in options:
        print(f"{opt['name']:<30} {opt['format']:<15} {opt['size']:<10} {opt['speed']:<10}")
    
    print("="*70)
    
    return options
```

### 58.6.4 端到端部署实战

```python
"""
端到端模型压缩与部署完整示例
"""
import torch
import torch.nn as nn
import copy


class ModelCompressionPipeline:
    """
    模型压缩与部署流水线
    
    整合：剪枝 + 量化 + 蒸馏 + 导出
    """
    
    def __init__(self, model, device='cuda'):
        self.device = device
        self.model = model.to(device)
        self.compression_history = []
        
    def apply_pruning(self, target_sparsity=0.5):
        """应用剪枝"""
        print(f"\n[剪枝] 目标稀疏度: {target_sparsity:.1%}")
        
        # 简化的幅度剪枝
        total_params = 0
        pruned_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data
                flat_weight = weight.abs().flatten()
                
                # 计算阈值
                k = int(target_sparsity * flat_weight.numel())
                threshold = torch.kthvalue(flat_weight, k).values
                
                # 创建掩码
                mask = (weight.abs() > threshold).float()
                
                # 应用剪枝
                module.weight.data *= mask
                
                total_params += weight.numel()
                pruned_params += (mask == 0).sum().item()
                
                # 注册掩码用于后续梯度masking
                module.register_buffer('prune_mask', mask)
        
        actual_sparsity = pruned_params / total_params
        print(f"  实际稀疏度: {actual_sparsity:.1%}")
        self.compression_history.append({'stage': 'pruning', 'sparsity': actual_sparsity})
        
        return self
    
    def apply_quantization(self, num_bits=8):
        """应用量化"""
        print(f"\n[量化] 位宽: {num_bits} bits")
        
        # 记录量化配置
        self.quantization_config = {
            'num_bits': num_bits,
            'qmin': -(2 ** (num_bits - 1)),
            'qmax': (2 ** (num_bits - 1)) - 1
        }
        
        # 这里简化处理，实际应导出到支持INT8的推理引擎
        print(f"  量化范围: [{self.quantization_config['qmin']}, {self.quantization_config['qmax']}]")
        self.compression_history.append({'stage': 'quantization', 'bits': num_bits})
        
        return self
    
    def export(self, format='onnx', dummy_input=None):
        """导出模型"""
        print(f"\n[导出] 格式: {format.upper()}")
        
        if format == 'onnx':
            if dummy_input is None:
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            
            output_path = f'model_compressed.onnx'
            torch.onnx.export(
                self.model, dummy_input, output_path,
                opset_version=11, do_constant_folding=True
            )
            print(f"  已导出到: {output_path}")
            
        self.compression_history.append({'stage': 'export', 'format': format})
        
        return self
    
    def summary(self):
        """打印压缩总结"""
        print("\n" + "="*60)
        print("模型压缩总结")
        print("="*60)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"模型参数量: {total_params:,}")
        
        for record in self.compression_history:
            print(f"  - {record}")
        
        print("="*60)


def run_complete_example():
    """
    运行完整示例
    """
    print("="*70)
    print("模型压缩与边缘部署完整示例")
    print("="*70)
    
    # 创建示例模型
    model = SimpleCNN(num_classes=10)
    print(f"原始模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建pipeline
    pipeline = ModelCompressionPipeline(model)
    
    # 应用压缩技术
    (pipeline
        .apply_pruning(target_sparsity=0.5)    # 剪枝50%
        .apply_quantization(num_bits=8)         # INT8量化
        .export(format='onnx')                  # 导出ONNX
        .summary()                              # 打印总结
    )
    
    # 展示部署选项
    compare_deployment_options()
    
    print("\n完整示例运行完成！")


# 如果直接运行此脚本
if __name__ == "__main__":
    run_complete_example()
```

## 58.7 本章总结

### 58.7.1 核心概念回顾

**模型压缩三剑客**：

1. **剪枝 (Pruning)**：识别并移除不重要的权重或结构
   - 非结构化剪枝：灵活性高，但硬件支持有限
   - 结构化剪枝：硬件友好，可实际加速推理
   - 彩票假说：随机初始化网络中存在可独立训练的高性能子网络

2. **量化 (Quantization)**：用更少的位数表示参数
   - PTQ：快速，无需重新训练
   - QAT：精度更高，通过训练适应量化误差
   - INT8可将模型大小和计算量减少75%

3. **知识蒸馏 (Distillation)**：让大模型教小模型
   - 软标签包含更多类别关系信息
   - 温度参数控制分布的"软化"程度
   - 可以蒸馏输出和中间特征

**高效架构设计**：

4. **MobileNet**：深度可分离卷积将计算量减少8-9倍
5. **EfficientNet**：复合缩放同时优化深度、宽度和分辨率

**边缘部署**：

6. **ONNX**：跨框架的中间表示
7. **TensorRT**：NVIDIA GPU的最优推理引擎
8. **TFLite**：移动端的标准部署方案

### 58.7.2 技术选择指南

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| 服务器GPU推理 | ONNX → TensorRT (FP16/INT8) | 最大吞吐量 |
| 移动端(iOS) | Core ML | 原生Apple生态支持 |
| 移动端(Android) | TFLite + NNAPI | 硬件加速支持 |
| 浏览器 | ONNX Runtime Web / TensorFlow.js | Web兼容性 |
| 嵌入式设备 | TFLite Micro / CMSIS-NN | 极致轻量 |

表58.8：边缘部署方案选择指南

### 58.7.3 实践建议

**模型压缩的最佳实践**：

1. **从预训练模型开始**：不要从头训练压缩模型
2. **渐进式压缩**：先剪枝、再量化、必要时蒸馏
3. **验证每一步**：确保压缩后的精度满足要求
4. **端到端测试**：在目标设备上验证推理性能
5. **监控延迟分布**：不仅看平均延迟，还要看P99延迟

**常见陷阱**：

- ❌ 在验证集上过度调优导致过拟合
- ❌ 忽略内存带宽瓶颈（计算量减少不一定意味着速度提升）
- ❌ 量化校准数据与真实分布不匹配
- ❌ 剪枝后没有进行充分的微调

### 58.7.4 本章的费曼比喻总结

| 概念 | 比喻 | 核心要点 |
|------|------|----------|
| 模型压缩 | 整理行李箱 | 只带必要的，压缩占空间的 |
| 模型剪枝 | 修剪盆栽 | 去掉细弱枝条，保持整体形态 |
| 彩票假说 | 寻找天选之才 | 某些人天生就适合某个任务 |
| 量化 | 压缩图片 | 用更少位数，损失有限精度 |
| 知识蒸馏 | 老教授带学生 | 传授思维方法，不仅是答案 |
| 温度参数 | 老师讲解方式 | 高温=更详细的解释 |
| 深度可分离卷积 | 分工协作的工厂 | 专业化分工提高效率 |
| 复合缩放 | 调配披萨配方 | 深度、宽度、分辨率同时增加 |
| 边缘部署 | 把图书馆搬进手机 | 便携、本地、随时可用 |
| ONNX | 音乐五线谱 | 通用表示，任何乐器都能演奏 |
| TensorRT | 赛车专业调校 | 针对特定硬件榨干性能 |

### 58.7.5 展望未来

模型压缩和边缘部署技术正在快速发展：

- **神经架构搜索 (NAS)**：自动发现高效的模型结构
- **动态推理**：根据输入复杂度调整计算量
- **硬件-软件协同设计**：专用AI芯片（NPU）的普及
- **大模型压缩**：如何让百亿参数模型在消费级硬件上运行

随着AI模型越来越大、应用越来越广，模型压缩和边缘部署将成为AI工程的核心能力——**让AI真正走进千家万户**。

---

## 参考文献

Frankle, J., & Carbin, M. (2019). The lottery ticket hypothesis: Finding sparse, trainable neural networks. In *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=rJl-b3RcF7

Gong, R., Liu, X., Jiang, S., Li, T., Hu, P., Lin, J., Yu, F., & Yan, J. (2019). Differentiable soft quantization: Bridging full-precision and low-bit neural networks. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 4852-4861. https://doi.org/10.1109/ICCV.2019.00495

Gou, J., Yu, B., Maybank, S. J., & Tao, D. (2021). Knowledge distillation: A survey. *International Journal of Computer Vision*, 129(6), 1789-1819. https://doi.org/10.1007/s11263-021-01453-z

Han, S., Mao, H., & Dally, W. (2016). Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=S1O8Kjlb

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*. https://arxiv.org/abs/1503.02531

Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., & Adam, H. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. *arXiv preprint arXiv:1704.04861*. https://arxiv.org/abs/1704.04861

Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., & Kalenichenko, D. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2704-2713. https://doi.org/10.1109/CVPR.2018.00286

Krishnamoorthi, R. (2018). Quantizing deep convolutional networks for efficient inference: A whitepaper. *arXiv preprint arXiv:1806.08342*. https://arxiv.org/abs/1806.08342

Li, H., Kadav, A., Durdanovic, I., Samet, H., & Graf, H. P. (2017). Pruning filters for efficient ConvNets. In *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=rJqFGTslg

Liu, Z., Sun, M., Zhou, T., Huang, G., & Darrell, T. (2019). Rethinking the value of network pruning. In *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=rJlnB3C5Ym

Malach, E., Yehudai, G., Shalev-Schwartz, S., & Shamir, O. (2020). Proving the lottery ticket hypothesis: Pruning is all you need. In *International Conference on Machine Learning (ICML)*, 6682-6691. PMLR. https://proceedings.mlr.press/v119/malach20a.html

Rastegari, M., Ordonez, V., Redmon, J., & Farhadi, A. (2016). XNOR-Net: ImageNet classification using binary convolutional neural networks. In *European Conference on Computer Vision (ECCV)*, 525-542. Springer. https://doi.org/10.1007/978-3-319-46493-0_32

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4510-4520. https://doi.org/10.1109/CVPR.2018.00474

Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *International Conference on Machine Learning (ICML)*, 6105-6114. PMLR. https://proceedings.mlr.press/v97/tan19a.html

Wu, S., Li, G., Chen, F., & Shi, L. (2016). Training and inference with integers in deep neural networks. In *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=HJGXzmspb

Zhu, X., Li, J., Liu, Y., Ma, C., & Zhang, S. (2024). A survey on model compression for large language models. *Transactions of the Association for Computational Linguistics*, 12, 1556-1577. https://doi.org/10.1162/tacl_a_00704

---

**本章完**

> *"让AI触手可及，是模型压缩与边缘部署的终极使命。"*
