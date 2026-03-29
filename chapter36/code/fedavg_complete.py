# 第三十六章 联邦学习

"数据不动模型动，隐私安全两相宜。"

![联邦学习示意图](images/federated_learning_concept.png)
*图36-1：联邦学习让数据留在本地，模型参数在云端聚合*

## 36.1 什么是联邦学习？

### 36.1.1 一个合唱团的比喻 🎵

想象一个大型合唱团要准备一场演出。传统的做法是：
- **集中式训练**：让所有歌手把嗓子"寄"到总部，由一位指挥统一训练
- **问题**：歌手的嗓子（数据）离开了自己，失去了控制

**联邦学习**就像这样：
- 每个歌手在自己的家里练习（本地训练）
- 指挥给所有人发一份乐谱（全局模型）
- 每个歌手在家练习后，把"练习心得"（模型更新）发回给指挥
- 指挥整合所有心得，更新乐谱，再发给大家
- **关键**：歌手的嗓子从未离开家！

> **费曼洞察**：联邦学习的精髓是"把模型带到数据那里，而不是把数据带到模型那里"。就像医生出诊，而不是病人集体去医院。

### 36.1.2 为什么需要联邦学习？

**现实困境**：
1. **隐私法规**：GDPR、CCPA等法规限制了数据共享
2. **商业机密**：银行、医院的数据不能外流
3. **数据孤岛**：各机构数据分散，形成"孤岛"
4. **通信成本**：海量原始数据传输代价高昂

| 场景 | 数据位置 | 隐私风险 | 联邦学习方案 |
|------|----------|----------|--------------|
| 智能手机 | 用户设备 | 个人敏感信息 | 本地训练键盘模型 |
| 医院 | 各医疗机构 | 患者隐私 | 联合训练诊断模型 |
| 银行 | 各金融机构 | 交易机密 | 联合风控模型 |
| 工厂 | 各生产线 | 工艺秘密 | 联合质量预测 |

### 36.1.3 联邦学习的基本架构

联邦学习系统由两类角色组成：

**1. 参数服务器（Parameter Server）**
- 维护全局模型参数
- 协调训练过程
- 聚合客户端更新
- **不接触原始数据**

**2. 客户端（Clients）**
- 持有本地数据
- 执行本地训练
- 上传模型更新
- 接收全局模型

```
┌─────────────────────────────────────────┐
│         参数服务器 (云端)                │
│  ┌─────────┐      ┌─────────┐          │
│  │ 全局模型 │◄────►│ 聚合算法 │          │
│  └────┬────┘      └─────────┘          │
│       │                                 │
│   下发模型 / 上传更新                     │
│       │                                 │
└───────┼─────────────────────────────────┘
        │
   ┌────┴────┬────────┬────────┐
   ▼         ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│客户端1│ │客户端2│ │客户端3│ │...   │
│本地数据│ │本地数据│ │本地数据│ │      │
│本地模型│ │本地模型│ │本地模型│ │      │
└──────┘ └──────┘ └──────┘ └──────┘
```

## 36.2 FedAvg算法

### 36.2.1 算法起源

联邦平均算法（Federated Averaging，**FedAvg**）由McMahan等人于2017年提出，是联邦学习领域的奠基性算法。

> **原始论文**：McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS*.

### 36.2.2 算法核心思想

FedAvg的核心是一个简单的迭代过程：

**第1步：广播（Broadcast）**
服务器将当前全局模型参数 $\theta_t$ 发送给选定的客户端集合 $S_t$

**第2步：本地训练（Local Training）**
每个客户端 $k$ 在自己的数据 $D_k$ 上进行 $E$ 轮本地训练：
$$\theta_{t,E}^k = \text{ClientUpdate}(\theta_t, D_k, E)$$

**第3步：上传（Upload）**
客户端将更新后的模型参数 $\theta_{t,E}^k$ 发送回服务器

**第4步：聚合（Aggregation）**
服务器按数据量加权平均所有客户端的模型：
$$\theta_{t+1} = \sum_{k \in S_t} \frac{n_k}{\sum_{j \in S_t} n_j} \theta_{t,E}^k$$

其中 $n_k$ 是客户端 $k$ 的样本数量。

### 36.2.3 完整代码实现

```python
"""
第三十六章：联邦学习 - FedAvg完整实现
Federated Averaging Algorithm Implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict

# 设置随机种子保证可复现
torch.manual_seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════
# 第一部分：神经网络模型定义
# ═══════════════════════════════════════════════════════════════

class SimpleNN(nn.Module):
    """
    简单的神经网络用于联邦学习演示
    架构：输入层 → 隐藏层(128) → 隐藏层(64) → 输出层
    """
    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


class CNN(nn.Module):
    """
    卷积神经网络用于更复杂的联邦学习实验
    """
    def __init__(self, num_classes: int = 10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
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


# ═══════════════════════════════════════════════════════════════
# 第二部分：客户端类实现
# ═══════════════════════════════════════════════════════════════

class FederatedClient:
    """
    联邦学习客户端
    
    每个客户端持有：
    - 本地数据集
    - 本地模型副本
    - 客户端ID和元信息
    
    主要职责：
    - 接收全局模型
    - 本地训练
    - 返回更新后的模型
    """
    
    def __init__(
        self,
        client_id: int,
        train_data: torch.utils.data.Dataset,
        model: nn.Module,
        device: str = 'cpu',
        lr: float = 0.01,
        batch_size: int = 32
    ):
        self.client_id = client_id
        self.train_data = train_data
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        
        # 记录客户端统计信息
        self.data_size = len(train_data)
        self.local_history = []
        
    def receive_global_model(self, global_state: Dict):
        """接收服务器下发的全局模型参数"""
        self.model.load_state_dict(global_state)
        
    def local_train(self, epochs: int = 5) -> Dict:
        """
        在本地数据上训练模型
        
        Args:
            epochs: 本地训练轮数
            
        Returns:
            更新后的模型状态字典
        """
        self.model.train()
        dataloader = DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        epoch_losses = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            avg_loss = total_loss / len(dataloader)
            accuracy = 100. * correct / total
            epoch_losses.append(avg_loss)
            
            if (epoch + 1) % 2 == 0:
                print(f"  Client {self.client_id} - Epoch {epoch+1}/{epochs}: "
                      f"Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
        
        self.local_history.append({
            'epochs': epochs,
            'final_loss': epoch_losses[-1],
            'loss_history': epoch_losses
        })
        
        # 返回模型参数（而非整个模型）
        return copy.deepcopy(self.model.state_dict())
    
    def evaluate(self, test_data: torch.utils.data.Dataset) -> float:
        """在测试数据上评估模型"""
        self.model.eval()
        dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy


# ═══════════════════════════════════════════════════════════════
# 第三部分：服务器类实现
# ═══════════════════════════════════════════════════════════════

class FederatedServer:
    """
    联邦学习服务器
    
    负责：
    - 维护全局模型
    - 选择参与客户端
    - 聚合客户端更新
    - 协调训练流程
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        fraction_clients: float = 1.0
    ):
        self.global_model = model.to(device)
        self.device = device
        self.fraction_clients = fraction_clients  # 每轮参与的客户端比例
        
        # 训练历史记录
        self.history = {
            'rounds': [],
            'train_loss': [],
            'test_accuracy': [],
            'participating_clients': []
        }
        
    def select_clients(self, all_clients: List[FederatedClient], round_num: int) -> List[FederatedClient]:
        """
        随机选择参与本轮训练的客户端
        
        Args:
            all_clients: 所有可用客户端
            round_num: 当前轮次（用于可复现的随机种子）
            
        Returns:
            选中的客户端列表
        """
        np.random.seed(round_num)  # 保证可复现
        num_clients = max(1, int(self.fraction_clients * len(all_clients)))
        selected = np.random.choice(all_clients, num_clients, replace=False)
        return list(selected)
    
    def aggregate_fedavg(self, client_updates: List[Tuple[Dict, int]]) -> Dict:
        """
        FedAvg聚合算法
        
        Args:
            client_updates: 列表，每个元素是(模型参数, 数据量)的元组
            
        Returns:
            聚合后的全局模型参数
        """
        # 计算总数据量
        total_data = sum(n_samples for _, n_samples in client_updates)
        
        # 初始化聚合结果
        aggregated = {}
        
        # 按数据量加权平均
        for key in client_updates[0][0].keys():
            aggregated[key] = sum(
                client_state[key] * (n_samples / total_data)
                for client_state, n_samples in client_updates
            )
        
        return aggregated
    
    def distribute_model(self, clients: List[FederatedClient]):
        """将全局模型分发给客户端"""
        global_state = copy.deepcopy(self.global_model.state_dict())
        for client in clients:
            client.receive_global_model(global_state)
    
    def run_round(
        self,
        all_clients: List[FederatedClient],
        local_epochs: int = 5
    ) -> Tuple[float, int]:
        """
        执行一轮联邦学习
        
        Returns:
            (平均训练损失, 参与客户端数)
        """
        # 1. 选择客户端
        selected_clients = self.select_clients(all_clients, len(self.history['rounds']))
        print(f"\nRound {len(self.history['rounds'])+1}: "
              f"Selected {len(selected_clients)}/{len(all_clients)} clients")
        
        # 2. 分发全局模型
        self.distribute_model(selected_clients)
        
        # 3. 客户端本地训练
        client_updates = []
        total_loss = 0.0
        
        for client in selected_clients:
            print(f"  Training client {client.client_id}...")
            updated_state = client.local_train(local_epochs)
            client_updates.append((updated_state, client.data_size))
            total_loss += client.local_history[-1]['final_loss']
        
        # 4. 聚合更新
        aggregated_state = self.aggregate_fedavg(client_updates)
        self.global_model.load_state_dict(aggregated_state)
        
        avg_loss = total_loss / len(selected_clients)
        
        # 记录历史
        self.history['rounds'].append(len(self.history['rounds']) + 1)
        self.history['train_loss'].append(avg_loss)
        self.history['participating_clients'].append(len(selected_clients))
        
        return avg_loss, len(selected_clients)
    
    def evaluate_global_model(self, test_data: torch.utils.data.Dataset) -> float:
        """评估全局模型性能"""
        self.global_model.eval()
        dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        self.history['test_accuracy'].append(accuracy)
        return accuracy
    
    def train(
        self,
        all_clients: List[FederatedClient],
        test_data: torch.utils.data.Dataset,
        num_rounds: int = 20,
        local_epochs: int = 5,
        eval_every: int = 5
    ):
        """
        完整的联邦训练流程
        """
        print("=" * 60)
        print("开始联邦学习训练")
        print(f"总轮数: {num_rounds}, 本地训练轮数: {local_epochs}")
        print(f"客户端总数: {len(all_clients)}")
        print("=" * 60)
        
        for round_num in range(num_rounds):
            # 执行一轮训练
            avg_loss, num_participants = self.run_round(all_clients, local_epochs)
            
            # 定期评估
            if (round_num + 1) % eval_every == 0:
                accuracy = self.evaluate_global_model(test_data)
                print(f"\n[评估] Round {round_num+1}: "
                      f"Avg Loss={avg_loss:.4f}, "
                      f"Test Accuracy={accuracy:.2f}%")
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)


# ═══════════════════════════════════════════════════════════════
# 第四部分：数据划分工具（IID vs Non-IID）
# ═══════════════════════════════════════════════════════════════

def partition_data_iid(
    dataset: torch.utils.data.Dataset,
    num_clients: int
) -> List[torch.utils.data.Dataset]:
    """
    将数据均匀随机划分给各个客户端（IID设置）
    
    IID = Independent and Identically Distributed
    每个客户端的数据分布与整体相同
    """
    num_items = len(dataset)
    items_per_client = num_items // num_clients
    client_datasets = []
    
    # 随机打乱所有数据索引
    all_idxs = np.random.permutation(num_items)
    
    for i in range(num_clients):
        start_idx = i * items_per_client
        end_idx = (i + 1) * items_per_client if i < num_clients - 1 else num_items
        client_idxs = all_idxs[start_idx:end_idx]
        
        # 创建子数据集
        client_data = torch.utils.data.Subset(dataset, client_idxs)
        client_datasets.append(client_data)
    
    return client_datasets


def partition_data_dirichlet(
    dataset: torch.utils.data.Dataset,
    num_clients: int,
    alpha: float = 0.5
) -> List[torch.utils.data.Dataset]:
    """
    使用Dirichlet分布划分数据（Non-IID设置）
    
    Args:
        dataset: 原始数据集
        num_clients: 客户端数量
        alpha: Dirichlet分布的浓度参数
               - alpha → ∞: 接近IID
               - alpha → 0: 极端Non-IID（每个客户端只有少数几类）
    
    Returns:
        每个客户端的数据集列表
    """
    # 获取所有标签
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        # 遍历获取标签
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_classes = len(np.unique(labels))
    num_samples = len(dataset)
    
    # 为每个类别，按Dirichlet分布分配给各客户端
    client_indices = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        # 获取类别k的所有样本索引
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        # 从Dirichlet分布采样分配比例
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array([
            p * (1.0 / num_clients) / p.sum() 
            for p in np.split(proportions, num_clients)
        ])
        
        # 按比例分配样本
        splits = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        client_idx_k = np.split(idx_k, splits)
        
        for i in range(num_clients):
            client_indices[i].extend(client_idx_k[i])
    
    # 创建子数据集
    client_datasets = []
    for indices in client_indices:
        np.random.shuffle(indices)
        subset = torch.utils.data.Subset(dataset, indices)
        client_datasets.append(subset)
    
    return client_datasets


def visualize_data_distribution(
    client_datasets: List[torch.utils.data.Dataset],
    num_classes: int = 10,
    title: str = "Data Distribution"
):
    """可视化各客户端的数据分布"""
    import matplotlib.pyplot as plt
    
    num_clients = len(client_datasets)
    distribution = np.zeros((num_clients, num_classes))
    
    for i, dataset in enumerate(client_datasets):
        # 统计每个类别的样本数
        if hasattr(dataset, 'dataset'):
            # 是Subset对象
            indices = dataset.indices
            labels = [dataset.dataset.targets[idx] for idx in indices]
        else:
            labels = [dataset[j][1] for j in range(len(dataset))]
        
        for label in labels:
            distribution[i][label] += 1
    
    # 归一化
    distribution = distribution / distribution.sum(axis=1, keepdims=True)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(distribution, cmap='YlOrRd', aspect='auto')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Client ID', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_clients))
    
    plt.colorbar(im, ax=ax, label='Proportion')
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=150)
    plt.show()
    
    return distribution


# ═══════════════════════════════════════════════════════════════
# 第五部分：完整实验流程
# ═══════════════════════════════════════════════════════════════

def run_federated_learning_experiment(
    dataset_name: str = 'MNIST',
    num_clients: int = 10,
    num_rounds: int = 20,
    local_epochs: int = 5,
    iid: bool = True,
    alpha: float = 0.5,
    device: str = 'cpu'
):
    """
    运行完整的联邦学习实验
    
    Args:
        dataset_name: 'MNIST' 或 'FashionMNIST'
        num_clients: 客户端数量
        num_rounds: 联邦训练轮数
        local_epochs: 每轮本地训练轮数
        iid: 是否使用IID数据划分
        alpha: Non-IID时的Dirichlet参数
        device: 计算设备
    """
    from torchvision import datasets, transforms
    
    print(f"\n{'='*60}")
    print(f"联邦学习实验配置")
    print(f"{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"客户端数: {num_clients}")
    print(f"数据分布: {'IID' if iid else f'Non-IID (alpha={alpha})'}")
    print(f"联邦轮数: {num_rounds}")
    print(f"本地训练轮数: {local_epochs}")
    print(f"{'='*60}\n")
    
    # 1. 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    if dataset_name == 'MNIST':
        train_dataset = datasets.MNIST(
            './data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            './data', train=False, download=True, transform=transform
        )
    else:
        train_dataset = datasets.FashionMNIST(
            './data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            './data', train=False, download=True, transform=transform
        )
    
    # 2. 划分数据
    if iid:
        client_datasets = partition_data_iid(train_dataset, num_clients)
    else:
        client_datasets = partition_data_dirichlet(
            train_dataset, num_clients, alpha
        )
    
    # 可视化数据分布
    visualize_data_distribution(
        client_datasets, 
        num_classes=10,
        title=f"{'IID' if iid else 'Non-IID'} Data Distribution"
    )
    
    # 3. 创建模型
    model = CNN(num_classes=10)
    
    # 4. 创建客户端
    clients = []
    for i in range(num_clients):
        client_model = CNN(num_classes=10)
        client = FederatedClient(
            client_id=i,
            train_data=client_datasets[i],
            model=client_model,
            device=device,
            lr=0.01,
            batch_size=32
        )
        clients.append(client)
        print(f"Client {i}: {len(client_datasets[i])} samples")
    
    # 5. 创建服务器
    server = FederatedServer(
        model=model,
        device=device,
        fraction_clients=1.0  # 每轮所有客户端都参与
    )
    
    # 6. 训练
    server.train(
        all_clients=clients,
        test_data=test_dataset,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        eval_every=5
    )
    
    # 7. 最终评估
    final_accuracy = server.evaluate_global_model(test_dataset)
    print(f"\n最终测试准确率: {final_accuracy:.2f}%")
    
    # 8. 绘制训练曲线
    plot_training_history(server.history, iid, alpha)
    
    return server, clients


def plot_training_history(history: Dict, iid: bool, alpha: float):
    """绘制训练历史曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    title_prefix = 'IID' if iid else f'Non-IID (α={alpha})'
    
    # 训练损失
    ax1.plot(history['rounds'], history['train_loss'], 'b-o', linewidth=2)
    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('Average Training Loss', fontsize=12)
    ax1.set_title(f'{title_prefix}: Training Loss', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 测试准确率
    eval_rounds = history['rounds'][4::5]  # 每5轮评估一次
    ax2.plot(eval_rounds, history['test_accuracy'], 'r-s', linewidth=2)
    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title(f'{title_prefix}: Test Accuracy', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    suffix = 'iid' if iid else f'noniid_alpha{alpha}'
    plt.savefig(f'training_history_{suffix}.png', dpi=150)
    plt.show()


# ═══════════════════════════════════════════════════════════════
# 第六部分：主函数入口
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 实验1：IID数据分布
    print("\n" + "="*60)
    print("实验1: IID数据分布")
    print("="*60)
    server_iid, clients_iid = run_federated_learning_experiment(
        dataset_name='MNIST',
        num_clients=10,
        num_rounds=20,
        local_epochs=5,
        iid=True,
        device=device
    )
    
    # 实验2：Non-IID数据分布
    print("\n" + "="*60)
    print("实验2: Non-IID数据分布")
    print("="*60)
    server_noniid, clients_noniid = run_federated_learning_experiment(
        dataset_name='MNIST',
        num_clients=10,
        num_rounds=20,
        local_epochs=5,
        iid=False,
        alpha=0.5,  # 较小的alpha表示更强的Non-IID
        device=device
    )
    
    # 对比结果
    print("\n" + "="*60)
    print("实验结果对比")
    print("="*60)
    print(f"IID最终准确率: {server_iid.history['test_accuracy'][-1]:.2f}%")
    print(f"Non-IID最终准确率: {server_noniid.history['test_accuracy'][-1]:.2f}%")
    print(f"性能差距: {server_iid.history['test_accuracy'][-1] - server_noniid.history['test_accuracy'][-1]:.2f}%")
