# 第三十六章：联邦学习 - 保护隐私的分布式智能

> **"数据不动模型动"** - 联邦学习的核心理念

## 本章导读

想象这样一个场景：全球数十亿部智能手机，每一部都存储着主人独特的打字习惯、常用的表情符号、个性化的词汇偏好。如果能把所有这些知识汇集起来，训练一个超级智能的键盘预测模型，将会是多么美妙的事情！

但是，这里有一个巨大的障碍：**隐私**。没有人愿意把自己的聊天记录、输入习惯发送到某个公司的服务器上。这就好像你想让一群专家一起写一本百科全书，但没人愿意离开自己的书房，更不愿意让别人看到自己书房里的藏书。

联邦学习（Federated Learning）就是解决这个问题的魔法钥匙。它让数据留在本地，只让"智慧"（模型）在设备之间流动。这就像是一场跨越千山万水的"思维共振"——每个参与者贡献自己的学习成果，却从不暴露自己的秘密。

在本章中，我们将：
- 🎯 理解联邦学习的核心原理和三大挑战
- 📊 深入推导FedAvg算法及其收敛性
- 🔬 探索FedProx、SCAFFOLD等高级优化算法
- 🛡️ 学习差分隐私和安全聚合技术
- 👤 掌握个性化联邦学习方法
- 🌍 了解真实世界的应用场景

准备好了吗？让我们开始这场隐私保护的分布式学习之旅！

---

## 36.1 什么是联邦学习？

### 36.1.1 联邦学习的诞生

2016年，Google的研究团队面临一个难题：如何让手机上的键盘输入法变得更智能？传统的机器学习方案需要把用户的输入数据上传到服务器，但这引发了严重的隐私担忧。

Google的解决方案是革命性的：**让模型去找数据，而不是让数据去找模型**。

2017年，McMahan等人在AISTATS会议上发表了论文《Communication-Efficient Learning of Deep Networks from Decentralized Data》，正式提出了**联邦平均（Federated Averaging, FedAvg）**算法，标志着联邦学习这一领域的诞生。

### 36.1.2 费曼法理解：联合考试

让我们用一个生动的比喻来理解联邦学习：

**集中式学习**就像把所有学生集中到一个大教室考试。老师可以看到每个学生的答卷，知道他们答对了什么、答错了什么。这显然侵犯了学生的隐私。

**分布式学习**就像把试卷寄给学生，学生答完后把答卷寄回。虽然学生不用出门，但答卷上的个人信息还是被老师看到了。

**联邦学习**则像是一场"联合考试"：
1. 老师把题目发给每个学生（下发全局模型）
2. 学生在自己家里答题（本地训练）
3. 学生只提交"答对题目的比例"和"常见错误的统计"（上传模型更新）
4. 老师汇总所有统计，改进教学方法（聚合全局模型）
5. 老师把改进后的方法发给所有学生（新一轮迭代）

**关键洞察**：老师从未看到任何一份完整的答卷，但通过汇总的学习模式，教学质量却能不断提升！

### 36.1.3 三大学习范式对比

| 特性 | 集中式学习 | 分布式学习 | 联邦学习 |
|------|-----------|-----------|---------|
| **数据位置** | 中央服务器 | 分布式节点 | 本地设备 |
| **数据隐私** | 低（全部集中） | 中（节点间共享） | 高（永不离开本地） |
| **通信对象** | 无 | 梯度/参数 | 模型参数 |
| **节点状态** | 同构 | 同构/异构 | 高度异构 |
| **典型场景** | 数据中心 | 服务器集群 | 移动设备、IoT |
| **主要挑战** | 存储/计算 | 通信瓶颈 | 统计异质性、隐私 |

### 36.1.4 联邦学习的核心挑战

联邦学习面临三大核心挑战，学术界称之为"**3H问题**"：

#### 1. 统计异质性（Statistical Heterogeneity）

想象一个学生来自数学世家，另一个来自文学世家，还有一个来自艺术世家。他们的知识背景完全不同！在联邦学习中，这表现为**Non-IID（非独立同分布）数据**：

```python
# 集中式学习：数据是IID的
# 每个批量的数据分布相同
data_pool = [sample_from_global_distribution() for _ in range(N)]

# 联邦学习：数据是Non-IID的
# 每个客户端有自己的数据分布
client_1_data = [sample_from_distribution_A()]  # 数学天才的数据
client_2_data = [sample_from_distribution_B()]  # 文学爱好者的数据
client_3_data = [sample_from_distribution_C()]  # 艺术家的数据
```

**数学表达**：
- 全局最优解：$\min_{w} \frac{1}{N} \sum_{i=1}^N f_i(w)$
- 但每个客户端的损失函数 $f_i$ 来自不同的分布
- 直接应用FedAvg可能导致收敛缓慢或发散

#### 2. 系统异质性（System Heterogeneity）

不同设备的计算能力、网络条件、电量状态千差万别：
- 📱 高端手机：8核CPU，快充，WiFi
- 📵 低端手机：4核CPU，电池老化，3G网络
- 🔋 某些设备：电量不足20%，可能随时掉线

这导致：
- **掉队者问题（Stragglers）**：某些客户端计算太慢
- **间歇性参与**：设备时而在线，时而离线
- **通信瓶颈**：带宽受限，上传下载困难

#### 3. 隐私与安全（Privacy & Security）

虽然原始数据不离开本地，但模型参数仍可能泄露信息：
- **模型反演攻击**：从模型参数推断训练数据
- **成员推断攻击**：判断某个样本是否参与了训练
- **拜占庭攻击**：恶意客户端上传有毒更新

---

## 36.2 FedAvg算法：联邦学习的基石

### 36.2.1 算法原理

FedAvg的核心思想简单优雅：**在本地多训练几步，减少通信次数**。

**标准流程**：

```
第t轮通信：
1. 服务器广播全局模型 w^t 给选中的K个客户端
2. 每个客户端k执行：
   - 初始化：w_k^t = w^t
   - 本地训练：w_k^{t+1} = LocalUpdate(w_k^t, data_k, E epochs)
3. 服务器聚合：w^{t+1} = Σ(n_k/n) * w_k^{t+1}
```

**关键参数**：
- $K$：每轮参与的客户端数量
- $E$：本地训练的epoch数
- $B$：本地训练的batch size
- $\eta$：学习率

### 36.2.2 数学推导

#### 本地更新

每个客户端k在本地执行E个epoch的SGD：

$$w_k^{t+1} = w^t - \eta \sum_{e=1}^E \sum_{b \in \mathcal{B}_e} \nabla f_k(w_k^{t,e,b}; b)$$

其中：
- $\mathcal{B}_e$ 是第e个epoch的mini-batch集合
- $w_k^{t,e,b}$ 是处理batch b时的参数
- $\nabla f_k$ 是客户端k的本地梯度

#### 全局聚合

服务器按数据量加权平均：

$$w^{t+1} = \sum_{k=1}^K \frac{n_k}{n} w_k^{t+1}$$

其中：
- $n_k$ 是客户端k的数据量
- $n = \sum_k n_k$ 是总数据量
- $\frac{n_k}{n}$ 是客户端k的权重

**为什么是加权平均？**

直觉上，数据量大的客户端应该更有话语权。数学上，这等价于最小化全局经验风险：

$$\min_w \sum_{k=1}^K \frac{n_k}{n} f_k(w)$$

### 36.2.3 完整Python实现

```python
"""
联邦学习基础实现 - FedAvg算法
================================
目标：实现完整的FedAvg训练流程
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import copy
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from collections import defaultdict


# ==================== 1. 模型定义 ====================

class SimpleCNN(nn.Module):
    """用于联邦学习的简单CNN模型（MNIST）"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimpleMLP(nn.Module):
    """用于联邦学习的简单MLP模型"""
    
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平
        return self.network(x)


# ==================== 2. 联邦客户端 ====================

class FederatedClient:
    """
    联邦学习客户端
    
    职责：
    1. 接收全局模型
    2. 在本地数据上训练
    3. 返回更新后的模型
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_data: DataLoader,
        test_data: DataLoader = None,
        device: str = 'cpu'
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_data = train_data
        self.test_data = test_data
        self.device = device
        self.data_size = len(train_data.dataset)
        
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """获取模型参数"""
        return {name: param.cpu().clone() 
                for name, param in self.model.named_parameters()}
    
    def set_weights(self, weights: Dict[str, torch.Tensor]):
        """设置模型参数"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in weights:
                    param.copy_(weights[name].to(self.device))
    
    def local_train(
        self,
        epochs: int = 1,
        lr: float = 0.01,
        verbose: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        本地训练
        
        Args:
            epochs: 本地训练的epoch数
            lr: 学习率
            verbose: 是否打印训练信息
            
        Returns:
            (更新后的权重, 训练统计信息)
        """
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_samples = 0
        correct = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(self.train_data):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)
                
                # 计算准确率
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            
            if verbose and epoch % max(1, epochs // 3) == 0:
                print(f"  Client {self.client_id}, Epoch {epoch+1}/{epochs}, "
                      f"Loss: {epoch_loss/epoch_samples:.4f}")
        
        train_stats = {
            'loss': total_loss / total_samples,
            'accuracy': correct / total_samples,
            'samples': self.data_size
        }
        
        return self.get_weights(), train_stats
    
    def local_evaluate(self) -> Dict:
        """本地评估"""
        if self.test_data is None:
            return {}
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_data:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)
        
        return {
            'loss': test_loss / total,
            'accuracy': correct / total,
            'samples': total
        }


# ==================== 3. FedAvg服务器 ====================

class FedAvgServer:
    """
    FedAvg服务器
    
    职责：
    1. 维护全局模型
    2. 协调客户端选择
    3. 聚合客户端更新
    4. 评估全局模型
    """
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[FederatedClient],
        device: str = 'cpu',
        fraction: float = 0.1  # 每轮参与的客户端比例
    ):
        self.global_model = model.to(device)
        self.clients = clients
        self.num_clients = len(clients)
        self.device = device
        self.fraction = fraction
        
        # 历史记录
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'communication_rounds': []
        }
    
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """获取全局模型权重"""
        return {name: param.cpu().clone() 
                for name, param in self.global_model.named_parameters()}
    
    def set_global_weights(self, weights: Dict[str, torch.Tensor]):
        """设置全局模型权重"""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in weights:
                    param.copy_(weights[name].to(self.device))
    
    def select_clients(self, num_clients: int = None) -> List[int]:
        """
        随机选择参与的客户端
        
        策略：均匀随机采样（可扩展为其他策略）
        """
        if num_clients is None:
            num_clients = max(1, int(self.fraction * self.num_clients))
        
        return np.random.choice(
            self.num_clients, 
            size=num_clients, 
            replace=False
        ).tolist()
    
    def aggregate(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_samples: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        FedAvg聚合：按数据量加权平均
        
        w_global = Σ(n_k / n_total) * w_k
        """
        total_samples = sum(client_samples)
        
        # 初始化聚合结果
        aggregated = {}
        for name in client_weights[0].keys():
            aggregated[name] = torch.zeros_like(client_weights[0][name])
        
        # 加权平均
        for weights, n_k in zip(client_weights, client_samples):
            weight = n_k / total_samples
            for name in aggregated.keys():
                aggregated[name] += weight * weights[name]
        
        return aggregated
    
    def train_round(
        self,
        epochs: int = 1,
        lr: float = 0.01,
        verbose: bool = False
    ) -> Dict:
        """
        执行一轮联邦学习
        
        流程：
        1. 选择参与的客户端
        2. 广播全局模型
        3. 本地训练
        4. 聚合更新
        """
        # 1. 选择客户端
        selected_clients = self.select_clients()
        if verbose:
            print(f"Round: 选择 {len(selected_clients)}/{self.num_clients} 个客户端")
        
        # 2. 获取全局权重并广播
        global_weights = self.get_global_weights()
        
        # 3. 本地训练
        client_updates = []
        client_samples = []
        client_stats = []
        
        for client_idx in selected_clients:
            client = self.clients[client_idx]
            
            # 同步全局模型到客户端
            client.set_weights(global_weights)
            
            # 本地训练
            updated_weights, stats = client.local_train(
                epochs=epochs, 
                lr=lr, 
                verbose=verbose
            )
            
            client_updates.append(updated_weights)
            client_samples.append(stats['samples'])
            client_stats.append(stats)
            
            if verbose:
                print(f"  Client {client_idx}: "
                      f"Loss={stats['loss']:.4f}, "
                      f"Acc={stats['accuracy']:.4f}")
        
        # 4. 聚合更新
        aggregated_weights = self.aggregate(client_updates, client_samples)
        self.set_global_weights(aggregated_weights)
        
        # 统计本轮训练结果
        avg_loss = np.mean([s['loss'] for s in client_stats])
        avg_acc = np.mean([s['accuracy'] for s in client_stats])
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'num_clients': len(selected_clients)
        }
    
    def evaluate_global(self, test_loader: DataLoader) -> Dict:
        """评估全局模型"""
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)
        
        return {
            'loss': test_loss / total,
            'accuracy': correct / total
        }
    
    def fit(
        self,
        rounds: int = 10,
        epochs: int = 1,
        lr: float = 0.01,
        test_loader: DataLoader = None,
        verbose: bool = True
    ) -> Dict:
        """
        完整训练流程
        
        Args:
            rounds: 通信轮数
            epochs: 每轮本地训练epoch数
            lr: 学习率
            test_loader: 测试数据（用于评估全局模型）
            verbose: 是否打印信息
        """
        print(f"开始FedAvg训练: {rounds}轮, "
              f"每轮{epochs}个epoch, 学习率{lr}")
        print(f"总客户端数: {self.num_clients}, "
              f"每轮参与比例: {self.fraction}")
        
        for round_idx in range(rounds):
            if verbose:
                print(f"\n{'='*50}")
                print(f"第 {round_idx + 1}/{rounds} 轮联邦学习")
                print('='*50)
            
            # 执行一轮训练
            train_stats = self.train_round(epochs, lr, verbose)
            
            # 评估全局模型
            test_stats = {'accuracy': 0.0}
            if test_loader is not None:
                test_stats = self.evaluate_global(test_loader)
            
            # 记录历史
            self.history['train_loss'].append(train_stats['loss'])
            self.history['train_accuracy'].append(train_stats['accuracy'])
            self.history['test_accuracy'].append(test_stats['accuracy'])
            self.history['communication_rounds'].append(round_idx + 1)
            
            if verbose:
                print(f"\n本轮结果:")
                print(f"  训练Loss: {train_stats['loss']:.4f}")
                print(f"  训练Acc:  {train_stats['accuracy']:.4f}")
                if test_loader:
                    print(f"  测试Acc:  {test_stats['accuracy']:.4f}")
        
        print("\n训练完成!")
        return self.history


# ==================== 4. 数据分发工具 ====================

class DataDistributor:
    """
    将数据分发给多个客户端
    
    支持多种数据划分策略：
    - IID：独立同分布
    - Non-IID Dirichlet：按Dirichlet分布划分
    - Pathological：每个客户端只有部分类别
    """
    
    @staticmethod
    def iid_split(
        dataset,
        num_clients: int
    ) -> List[Subset]:
        """
        IID划分：每个客户端随机获得等量的数据
        """
        num_items = len(dataset) // num_clients
        client_datasets = []
        
        all_indices = np.random.permutation(len(dataset))
        
        for i in range(num_clients):
            start = i * num_items
            end = (i + 1) * num_items if i < num_clients - 1 else len(dataset)
            indices = all_indices[start:end]
            client_datasets.append(Subset(dataset, indices))
        
        return client_datasets
    
    @staticmethod
    def dirichlet_split(
        dataset,
        num_clients: int,
        alpha: float = 0.5,
        seed: int = 42
    ) -> List[Subset]:
        """
        Dirichlet划分：Non-IID场景
        
        alpha越小，数据异构性越高
        alpha -> ∞ 时趋近于IID
        alpha -> 0 时趋近于极端Non-IID
        
        参考: Hsu et al. 2019 "Measuring the Effects of Non-Identical Data..."
        """
        np.random.seed(seed)
        
        # 按标签组织数据
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        num_classes = len(np.unique(labels))
        
        # 每个类别的样本索引
        class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
        
        # 为每个类别按Dirichlet分布分配给客户端
        client_indices = [[] for _ in range(num_clients)]
        
        for c in range(num_classes):
            # 采样Dirichlet分布
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([
                p * (len(class_indices[c]) < num_clients) / p.sum() 
                if p * len(class_indices[c]) < 1 
                else p 
                for p in proportions
            ])
            proportions = proportions / proportions.sum()
            
            # 计算每个客户端分配的数量
            splits = (proportions * len(class_indices[c])).astype(int)
            splits[-1] = len(class_indices[c]) - splits[:-1].sum()
            
            # 分配索引
            start = 0
            for k in range(num_clients):
                end = start + splits[k]
                client_indices[k].extend(class_indices[c][start:end])
                start = end
        
        return [Subset(dataset, indices) for indices in client_indices]
    
    @staticmethod
    def pathological_split(
        dataset,
        num_clients: int,
        classes_per_client: int = 2,
        seed: int = 42
    ) -> List[Subset]:
        """
        病态划分：每个客户端只有指定数量的类别
        
        这是最极端的Non-IID场景
        """
        np.random.seed(seed)
        
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        num_classes = len(np.unique(labels))
        
        # 按标签组织
        class_indices = [np.where(labels == c)[0].tolist() for c in range(num_classes)]
        
        client_indices = [[] for _ in range(num_clients)]
        
        # 为每个客户端分配类别
        class_assignments = []
        for k in range(num_clients):
            # 随机选择classes_per_client个类别
            assigned = np.random.choice(num_classes, classes_per_client, replace=False)
            class_assignments.append(assigned)
        
        # 分配数据
        for c in range(num_classes):
            # 找出拥有这个类别的客户端
            clients_with_class = [k for k in range(num_clients) if c in class_assignments[k]]
            
            if len(clients_with_class) == 0:
                continue
            
            # 平均分配给这些客户端
            indices = class_indices[c]
            np.random.shuffle(indices)
            
            samples_per_client = len(indices) // len(clients_with_class)
            for i, k in enumerate(clients_with_class):
                start = i * samples_per_client
                end = (i + 1) * samples_per_client if i < len(clients_with_class) - 1 else len(indices)
                client_indices[k].extend(indices[start:end])
        
        return [Subset(dataset, indices) for indices in client_indices]


# ==================== 5. 可视化工具 ====================

def plot_training_history(history: Dict, save_path: str = None):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rounds = history['communication_rounds']
    
    # 损失曲线
    axes[0].plot(rounds, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0].set_xlabel('Communication Round', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(rounds, history['train_accuracy'], 'b-', linewidth=2, label='Train Accuracy')
    if any(acc > 0 for acc in history['test_accuracy']):
        axes[1].plot(rounds, history['test_accuracy'], 'r-', linewidth=2, label='Test Accuracy')
    axes[1].set_xlabel('Communication Round', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Model Accuracy', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()


def visualize_data_distribution(client_datasets: List[Subset], num_classes: int = 10):
    """可视化各客户端的数据分布"""
    client_labels = []
    for dataset in client_datasets:
        labels = [dataset[i][1] for i in range(len(dataset))]
        client_labels.append(labels)
    
    # 统计每个客户端的类别分布
    fig, ax = plt.subplots(figsize=(12, 6))
    
    client_ids = range(len(client_datasets))
    bottom = np.zeros(len(client_datasets))
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    for c in range(num_classes):
        counts = [np.sum(np.array(labels) == c) for labels in client_labels]
        ax.bar(client_ids, counts, bottom=bottom, label=f'Class {c}', color=colors[c])
        bottom += counts
    
    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Data Distribution Across Clients', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


# ==================== 6. 运行示例 ====================

def run_fedavg_example():
    """
    FedAvg完整运行示例
    """
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 加载MNIST数据
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 参数设置
    NUM_CLIENTS = 100
    CLIENT_FRACTION = 0.1  # 每轮10%客户端参与
    ROUNDS = 20
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    
    print("="*60)
    print("FedAvg联邦学习实验")
    print("="*60)
    print(f"客户端数量: {NUM_CLIENTS}")
    print(f"每轮参与比例: {CLIENT_FRACTION}")
    print(f"通信轮数: {ROUNDS}")
    print(f"本地训练epoch: {LOCAL_EPOCHS}")
    print(f"批量大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")
    print("="*60)
    
    # 创建Non-IID数据划分
    print("\n[1/4] 划分Non-IID数据...")
    client_datasets = DataDistributor.dirichlet_split(
        train_dataset, 
        NUM_CLIENTS, 
        alpha=0.5  # Non-IID程度
    )
    
    # 可视化数据分布
    visualize_data_distribution(client_datasets)
    
    # 创建客户端
    print("\n[2/4] 创建联邦客户端...")
    clients = []
    for i in range(NUM_CLIENTS):
        train_loader = DataLoader(
            client_datasets[i], 
            batch_size=BATCH_SIZE, 
            shuffle=True
        )
        
        # 每个客户端有自己的模型副本
        client_model = SimpleCNN()
        
        client = FederatedClient(
            client_id=i,
            model=client_model,
            train_data=train_loader,
            device='cpu'
        )
        clients.append(client)
    
    print(f"已创建 {len(clients)} 个客户端")
    
    # 创建服务器
    print("\n[3/4] 初始化FedAvg服务器...")
    global_model = SimpleCNN()
    server = FedAvgServer(
        model=global_model,
        clients=clients,
        device='cpu',
        fraction=CLIENT_FRACTION
    )
    
    # 训练
    print("\n[4/4] 开始联邦学习训练...")
    history = server.fit(
        rounds=ROUNDS,
        epochs=LOCAL_EPOCHS,
        lr=LEARNING_RATE,
        test_loader=test_loader,
        verbose=True
    )
    
    # 最终评估
    print("\n" + "="*60)
    print("最终评估")
    print("="*60)
    final_stats = server.evaluate_global(test_loader)
    print(f"最终测试准确率: {final_stats['accuracy']:.4f}")
    
    # 可视化训练过程
    plot_training_history(history)
    
    return server, history


if __name__ == "__main__":
    # 运行完整示例
    server, history = run_fedavg_example()
```

