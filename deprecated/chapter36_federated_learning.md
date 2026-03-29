# 第三十六章 联邦学习

## 本章概要

联邦学习(Federated Learning, FL)是一种革命性的分布式机器学习范式，它允许多个参与方在不共享原始数据的前提下协作训练模型。想象一下：全球数百万部智能手机共同训练一个输入法预测模型，但用户的聊天记录永远留在本地——这就是联邦学习的魔力！

**学习目标:**
- 理解联邦学习的核心概念和三大类型(HFL/VFL/FTL)
- 掌握FedAvg算法及其数学原理
- 理解数据异构性挑战及解决方案(FedProx、SCAFFOLD)
- 了解个性化联邦学习和隐私保护技术
- 实现联邦学习算法并应用于实际场景

**关键术语:** 联邦平均(FedAvg)、水平联邦学习、垂直联邦学习、数据异构性、个性化联邦学习、差分隐私、安全聚合

---

## 36.1 什么是联邦学习？

### 36.1.1 从集中式到联邦式

**传统机器学习的困境**

想象你要训练一个医疗诊断模型。理想情况下，你需要收集来自全国各地医院的数据：

```python
# 传统集中式学习的问题
class CentralizedLearning:
    """集中式学习的隐私困境"""
    
    def collect_data(self):
        # 问题1: 数据隐私风险
        patient_data = []
        for hospital in all_hospitals:
            data = hospital.send_all_patient_records()  # ❌ 隐私泄露风险!
            patient_data.extend(data)
        
        # 问题2: 法律合规挑战
        # GDPR、HIPAA等法规严格限制数据传输
        
        # 问题3: 商业机密
        # 医院不愿分享宝贵的医疗数据
        
        return patient_data
```

**联邦学习的解决方案**

联邦学习的核心理念：**数据不动模型动**。

```
传统方式: 数据 → 云端 → 训练模型
联邦方式: 模型 → 设备 → 本地训练 → 聚合更新
```

用费曼的话来解释：想象一群学生备考，传统方式是所有人把笔记交给老师，老师整理后教给大家。联邦学习则是老师分发复习大纲，每个学生在自家笔记上做标注，然后只把标注的重点（不是笔记本身）交给老师汇总，最后老师把汇总后的重点再发给大家。

### 36.1.2 联邦学习的正式定义

**定义 36.1 (联邦学习)**

联邦学习是一种分布式机器学习框架，其中 $K$ 个客户端在协调服务器协助下协作训练共享模型，同时保持训练数据 decentralized（去中心化）。

数学上，联邦学习解决以下优化问题：

$$\min_w F(w) = \sum_{k=1}^{K} \frac{n_k}{n} F_k(w)$$

其中：
- $n_k$ 是第 $k$ 个客户端的样本数
- $n = \sum_{k=1}^{K} n_k$ 是总样本数
- $F_k(w) = \frac{1}{n_k} \sum_{i \in \mathcal{D}_k} f_i(w)$ 是第 $k$ 个客户端的本地损失

```python
import numpy as np
from typing import List, Dict, Tuple
import torch
import torch.nn as nn

class FederatedLearningFramework:
    """联邦学习框架基础"""
    
    def __init__(self, num_clients: int, model: nn.Module):
        self.K = num_clients          # 客户端数量
        self.global_model = model      # 全局模型
        self.clients: List[Client] = []  # 客户端列表
        
    def objective_function(self, w: np.ndarray) -> float:
        """
        联邦学习的全局损失函数
        
        F(w) = Σ (n_k/n) * F_k(w)
        
        其中每个客户端的本地损失按其数据量加权
        """
        total_samples = sum(client.n_samples for client in self.clients)
        
        global_loss = 0.0
        for client in self.clients:
            weight = client.n_samples / total_samples
            local_loss = client.compute_loss(w)
            global_loss += weight * local_loss
            
        return global_loss
```

### 36.1.3 联邦学习的三大类型

根据数据在特征空间和样本空间的分布，联邦学习分为三种类型：

**1. 水平联邦学习 (Horizontal Federated Learning, HFL)**

```
特征空间: X₁ = X₂ = ... = X_K (相同)
样本空间: I₁ ∩ I₂ = ∅ (不重叠)

示意图:
客户端A: [特征1, 特征2, 特征3] → 样本A1, A2, A3...
客户端B: [特征1, 特征2, 特征3] → 样本B1, B2, B3...
客户端C: [特征1, 特征2, 特征3] → 样本C1, C2, C3...
```

**典型场景:** 不同地区的银行使用相同的特征（年龄、收入、信用分）评估不同客户群体

**2. 垂直联邦学习 (Vertical Federated Learning, VFL)**

```
特征空间: X₁ ≠ X₂ ≠ ... ≠ X_K (不同)
样本空间: I₁ = I₂ = ... = I_K (相同)

示意图:
          用户1    用户2    用户3
银行:    [收入, 存款, 贷款记录]
电商:    [购物频次, 消费金额, 退货率]
运营商:  [通话时长, 流量使用, 欠费记录]
```

**典型场景:** 银行和电商联合建模，拥有同一批用户的不同特征

**3. 联邦迁移学习 (Federated Transfer Learning, FTL)**

```
特征空间: X₁ ≠ X₂ ≠ ... ≠ X_K (不同)
样本空间: I₁ ∩ I₂ ≈ ∅ (几乎不重叠)

示意图:
客户端A(医院A): [症状A, 检查A] → 疾病A患者
客户端B(医院B): [症状B, 检查B] → 疾病B患者
```

**典型场景:** 不同领域的机构（如医院和学校）数据特征和样本都完全不同，但希望共享知识

```python
class FederatedLearningType:
    """联邦学习类型判断"""
    
    @staticmethod
    def identify_fl_type(clients_data: List[Dict]) -> str:
        """
        根据数据分布判断联邦学习类型
        
        Args:
            clients_data: 每个客户端的数据信息
                [{features: [...], samples: [...]}, ...]
        
        Returns:
            "HFL", "VFL", 或 "FTL"
        """
        # 检查特征空间
        feature_sets = [set(client['features']) for client in clients_data]
        same_features = all(fs == feature_sets[0] for fs in feature_sets)
        
        # 检查样本空间
        sample_sets = [set(client['samples']) for client in clients_data]
        overlapping_samples = any(
            sample_sets[i] & sample_sets[j] 
            for i in range(len(sample_sets)) 
            for j in range(i+1, len(sample_sets))
        )
        
        if same_features and not overlapping_samples:
            return "HFL"  # 水平联邦学习
        elif not same_features and overlapping_samples:
            return "VFL"  # 垂直联邦学习
        else:
            return "FTL"  # 联邦迁移学习
```

### 36.1.4 联邦学习的优势与挑战

**核心优势:**

| 优势 | 说明 | 类比 |
|------|------|------|
| **数据隐私保护** | 原始数据不离开本地 | 学生只交标注，不交笔记 |
| **法律合规** | 满足GDPR、HIPAA等法规 | 数据不出境 |
| **降低通信成本** | 只传输模型参数 | 传摘要比传全书快 |
| **实时个性化** | 本地数据实时更新模型 | 个人定制学习 |

**主要挑战:**

| 挑战 | 说明 | 解决方案 |
|------|------|----------|
| **数据异构性** | 各客户端数据分布不同 | FedProx、SCAFFOLD |
| **系统异构性** | 设备计算能力差异大 | 异步聚合、模型压缩 |
| **通信开销** | 模型参数量大 | 梯度压缩、稀疏化 |
| **安全威胁** | 梯度泄露、拜占庭攻击 | 差分隐私、安全聚合 |

```python
class FLChallengeAnalyzer:
    """联邦学习挑战分析器"""
    
    def analyze_statistical_heterogeneity(self, clients: List) -> Dict:
        """
        分析数据异构性程度
        
        统计异构性指标:
        1. 类别分布差异 (Label Distribution Skew)
        2. 特征分布差异 (Feature Distribution Skew)
        3. 样本量不平衡 (Quantity Skew)
        """
        results = {
            'label_distribution_skew': [],
            'sample_imbalance': 0.0,
            'overall_heterogeneity': 0.0
        }
        
        # 分析每个客户端的类别分布
        for client in clients:
            label_dist = client.get_label_distribution()
            results['label_distribution_skew'].append(label_dist)
        
        # 计算分布差异 (使用Jensen-Shannon散度)
        js_divergence = self._compute_js_divergence(
            results['label_distribution_skew']
        )
        results['overall_heterogeneity'] = js_divergence
        
        return results
    
    def _compute_js_divergence(self, distributions: List[np.ndarray]) -> float:
        """计算Jensen-Shannon散度衡量分布差异"""
        # 计算平均分布
        mean_dist = np.mean(distributions, axis=0)
        
        # JS散度 = 平均KL散度
        js_div = 0.0
        for dist in distributions:
            js_div += self._kl_divergence(dist, mean_dist) / len(distributions)
            
        return js_div
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算KL散度"""
        # 避免除零
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        return np.sum(p * np.log(p / q))
```

---

## 36.2 FedAvg：联邦学习的基石算法

### 36.2.1 算法核心思想

联邦平均算法(Federated Averaging, FedAvg)由McMahan等人于2017年提出，是联邦学习最基础的算法。

**算法流程:**

```
服务器初始化全局模型 w₀
对于每一轮通信 t = 0, 1, 2, ...:
    1. 服务器随机选择部分客户端 S_t
    2. 将全局模型 w_t 发送给选中的客户端
    3. 每个客户端 k ∈ S_t 执行:
       - 用本地数据训练模型 E 个epoch
       - 计算模型更新 Δw_k = w_t - w_k^local
       - 发送更新给服务器
    4. 服务器聚合更新:
       w_{t+1} = w_t - Σ (n_k/n) * Δw_k
```

```python
import torch
import torch.nn as nn
from typing import List, Dict
import copy

class FedAvgServer:
    """
    FedAvg服务器实现
    
    McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). 
    Communication-efficient learning of deep networks from decentralized data. 
    In Artificial intelligence and statistics (pp. 1273-1282). PMLR.
    """
    
    def __init__(self, model: nn.Module, fraction_clients: float = 0.1):
        self.global_model = model
        self.C = fraction_clients  # 每轮参与的客户端比例
        self.round = 0
        
    def aggregate(self, client_updates: List[Dict]) -> nn.Module:
        """
        聚合客户端更新
        
        w_{t+1} = Σ (n_k / n) * w_k^local
        
        Args:
            client_updates: 每个客户端的更新信息
                [{'weights': state_dict, 'n_samples': int}, ...]
        
        Returns:
            更新后的全局模型
        """
        total_samples = sum(update['n_samples'] for update in client_updates)
        
        # 初始化聚合后的参数
        aggregated_state = copy.deepcopy(self.global_model.state_dict())
        
        # 加权平均
        for key in aggregated_state.keys():
            aggregated_state[key] = torch.zeros_like(aggregated_state[key])
            
        for update in client_updates:
            weight = update['n_samples'] / total_samples
            local_state = update['weights']
            
            for key in aggregated_state.keys():
                aggregated_state[key] += weight * local_state[key]
        
        # 更新全局模型
        self.global_model.load_state_dict(aggregated_state)
        self.round += 1
        
        return self.global_model
    
    def select_clients(self, all_clients: List, num_select: int = None) -> List:
        """随机选择参与本轮训练的客户端"""
        if num_select is None:
            num_select = max(1, int(self.C * len(all_clients)))
        
        return random.sample(all_clients, min(num_select, len(all_clients)))


class FedAvgClient:
    """FedAvg客户端实现"""
    
    def __init__(self, client_id: int, model: nn.Module, data_loader):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.n_samples = len(data_loader.dataset)
        
    def local_train(self, global_weights: Dict, epochs: int = 5, 
                    lr: float = 0.01) -> Dict:
        """
        本地训练
        
        使用本地数据训练模型多个epoch
        
        Args:
            global_weights: 全局模型参数
            epochs: 本地训练轮数
            lr: 学习率
        
        Returns:
            更新信息字典
        """
        # 加载全局模型
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # 本地训练
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # 返回更新后的权重和样本数
        return {
            'client_id': self.client_id,
            'weights': copy.deepcopy(self.model.state_dict()),
            'n_samples': self.n_samples
        }
```

### 36.2.2 FedAvg的数学分析

**收敛性分析:**

FedAvg的收敛性取决于多个因素。我们分析其关键特性：

**定理 36.1 (FedAvg收敛性)**

假设：
1. 损失函数 $F_k$ 是 $L$-平滑的
2. 梯度方差有界：$\mathbb{E}[\|\nabla F_k(w; \xi) - \nabla F_k(w)\|^2] \leq \sigma^2$
3. 梯度差异有界：$\|\nabla F_k(w) - \nabla F(w)\|^2 \leq \kappa^2$

则经过 $T$ 轮通信后：

$$\min_{t \leq T} \mathbb{E}[\|\nabla F(w_t)\|^2] \leq \mathcal{O}\left(\frac{1}{\sqrt{KT}}\right) + \mathcal{O}(\kappa^2)$$

其中 $\kappa^2$ 项反映了**数据异构性**对收敛的影响。

```python
class FedAvgConvergenceAnalyzer:
    """FedAvg收敛性分析器"""
    
    def __init__(self, L: float = 0.1, sigma2: float = 1.0, kappa2: float = 0.1):
        """
        Args:
            L: 平滑系数
            sigma2: 随机梯度方差
            kappa2: 梯度差异上界(数据异构性度量)
        """
        self.L = L
        self.sigma2 = sigma2
        self.kappa2 = kappa2
        
    def convergence_bound(self, K: int, T: int, E: int, lr: float) -> float:
        """
        计算FedAvg的收敛上界
        
        理论收敛率: O(1/√(KT)) + O(κ²)
        
        Args:
            K: 客户端数量
            T: 通信轮数
            E: 本地训练epoch数
            lr: 学习率
        
        Returns:
            梯度范数的上界估计
        """
        # 主要项
        term1 = 1.0 / np.sqrt(K * T)
        
        # 数据异构性项
        term2 = self.kappa2
        
        # 本地训练带来的漂移项
        term3 = self.L * lr * E * self.kappa2
        
        return term1 + term2 + term3
    
    def optimal_learning_rate(self, K: int, T: int) -> float:
        """计算最优学习率"""
        return np.sqrt(K / T) / self.L
    
    def estimate_rounds_for_accuracy(self, target_accuracy: float, 
                                      K: int) -> int:
        """
        估计达到目标精度所需的通信轮数
        
        假设收敛界 ≈ target_accuracy
        1/√(KT) + κ² ≈ target_accuracy
        
        解得: T ≈ 1/(K * (target_accuracy - κ²)²)
        """
        effective_target = max(target_accuracy - self.kappa2, 1e-6)
        return int(1.0 / (K * effective_target ** 2))
```

### 36.2.3 FedAvg的局限性

**问题1: 数据异构性导致的"客户端漂移"**

当各客户端数据分布差异大时，本地模型会朝着不同方向更新：

```python
def visualize_client_drift():
    """可视化客户端漂移问题"""
    import matplotlib.pyplot as plt
    
    # 模拟两个客户端在参数空间的更新轨迹
    # 客户端1: 主要包含类别A的数据
    # 客户端2: 主要包含类别B的数据
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 全局最优解
    global_opt = np.array([0, 0])
    
    # 客户端1的本地最优解（偏向类别A）
    local_opt_1 = np.array([-2, 1])
    
    # 客户端2的本地最优解（偏向类别B）
    local_opt_2 = np.array([2, -1])
    
    # 绘制优化轨迹
    # ... 可视化代码
    
    ax.plot(*global_opt, 'r*', markersize=20, label='Global Optimum')
    ax.plot(*local_opt_1, 'bo', markersize=15, label='Client 1 Local Opt')
    ax.plot(*local_opt_2, 'go', markersize=15, label='Client 2 Local Opt')
    
    ax.set_title('Client Drift in Non-IID Federated Learning')
    ax.legend()
    
    return fig
```

**问题2: 系统异构性**

不同设备的计算能力、网络带宽差异大：

| 设备类型 | 计算能力 | 典型场景 |
|----------|----------|----------|
| 旗舰手机 | 高(GPU) | 本地训练快，可执行更多epoch |
| 中端手机 | 中(CPU) | 训练较慢，可能超时 |
| 低端设备 | 低 | 只能参与推理，难参与训练 |

---

## 36.3 应对数据异构性：高级聚合算法

### 36.3.1 FedProx：添加近端正则项

FedProx在本地损失函数中添加了近端项，限制本地模型偏离全局模型太远：

$$\min_w F_k(w) + \frac{\mu}{2}\|w - w_t\|^2$$

```python
class FedProxClient(FedAvgClient):
    """
    FedProx客户端实现
    
    Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). 
    Federated optimization in heterogeneous networks. 
    Proceedings of Machine learning and systems, 2, 429-450.
    """
    
    def __init__(self, client_id: int, model: nn.Module, data_loader, 
                 mu: float = 0.01):
        super().__init__(client_id, model, data_loader)
        self.mu = mu  # 近端系数
        
    def local_train(self, global_weights: Dict, epochs: int = 5,
                    lr: float = 0.01) -> Dict:
        """
        带近端正则的本地训练
        
        最小化: F_k(w) + (μ/2)||w - w_global||²
        """
        # 保存全局模型用于近端项
        global_tensor = {}
        for name, param in global_weights.items():
            global_tensor[name] = param.clone()
        
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for data, target in self.data_loader:
                optimizer.zero_grad()
                
                # 标准损失
                output = self.model(data)
                loss = criterion(output, target)
                
                # 添加近端正则项: (μ/2)||w - w_global||²
                proximal_term = 0.0
                for name, param in self.model.named_parameters():
                    proximal_term += torch.sum(
                        (param - global_tensor[name]) ** 2
                    )
                
                total_loss = loss + (self.mu / 2) * proximal_term
                
                total_loss.backward()
                optimizer.step()
        
        return {
            'client_id': self.client_id,
            'weights': copy.deepcopy(self.model.state_dict()),
            'n_samples': self.n_samples
        }
```

**定理 36.2 (FedProx收敛性)**

FedProx在Non-IID数据下具有更好的收敛保证。对于任意客户端 $k$，其收敛界与 $\mu$ 成正比：

$$\mathbb{E}[F(w_T)] - F(w^*) \leq \mathcal{O}\left(\frac{1}{T}\right) + \mathcal{O}(\kappa^2 / \mu)$$

增大 $\mu$ 可以减少客户端漂移，但会减慢收敛速度。

### 36.3.2 SCAFFOLD：使用控制变量

SCAFFOLD通过控制变量(control variates)来纠正本地更新的偏差：

```python
class SCAFFOLDClient:
    """
    SCAFFOLD客户端实现
    
    Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S., Stich, S., & Suresh, A. T. (2020). 
    SCAFFOLD: Stochastic controlled averaging for federated learning. 
    In International conference on machine learning (pp. 5132-5143). PMLR.
    """
    
    def __init__(self, client_id: int, model: nn.Module, data_loader):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.n_samples = len(data_loader.dataset)
        
        # 客户端控制变量
        self.client_control = {
            name: torch.zeros_like(param.data)
            for name, param in model.named_parameters()
        }
        
    def local_train(self, global_weights: Dict, global_control: Dict,
                    epochs: int = 5, lr: float = 0.01) -> Dict:
        """
        SCAFFOLD本地训练
        
        使用控制变量纠正更新方向:
        y ← y - η_l(g_k(y) - c_k + c)
        
        其中:
        - g_k(y): 本地梯度
        - c_k: 客户端控制变量
        - c: 服务器控制变量
        """
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # 保存初始模型用于计算控制变量更新
        y_initial = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        
        for epoch in range(epochs):
            for data, target in self.data_loader:
                optimizer.zero_grad()
                
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # SCAFFOLD更新: 应用控制变量修正
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            # 修正梯度: g_k(y) - c_k + c
                            corrected_grad = (
                                param.grad - 
                                self.client_control[name] + 
                                global_control[name]
                            )
                            param.data -= lr * corrected_grad
        
        # 更新客户端控制变量
        control_update = {}
        y_delta = {}
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # y - y_initial
                y_delta[name] = param.data - y_initial[name]
                
                # c_k ← c_k - c + (y_final - y_initial)/(K*η_l)
                option_i = 1  # SCAFFOLD的两种更新策略
                if option_i == 1:
                    control_update[name] = (
                        self.client_control[name] - global_control[name] +
                        y_delta[name] / (epochs * lr)
                    )
                else:
                    # Option II
                    control_update[name] = global_control[name]
                
                self.client_control[name] = control_update[name]
        
        return {
            'client_id': self.client_id,
            'weights': copy.deepcopy(self.model.state_dict()),
            'n_samples': self.n_samples,
            'control_delta': control_update,
            'y_delta': y_delta
        }


class SCAFFOLDServer:
    """SCAFFOLD服务器实现"""
    
    def __init__(self, model: nn.Module):
        self.global_model = model
        self.round = 0
        
        # 服务器控制变量
        self.global_control = {
            name: torch.zeros_like(param.data)
            for name, param in model.named_parameters()
        }
        
    def aggregate(self, client_updates: List[Dict]) -> nn.Module:
        """
        SCAFFOLD聚合
        
        更新规则:
        x ← x + η_g * average(y_delta)
        c ← c + (1/K) * Σ(c_k - c)
        """
        total_samples = sum(upd['n_samples'] for upd in client_updates)
        
        # 聚合模型更新
        aggregated_state = copy.deepcopy(self.global_model.state_dict())
        
        for key in aggregated_state.keys():
            aggregated_state[key] = torch.zeros_like(aggregated_state[key])
            
        for update in client_updates:
            weight = update['n_samples'] / total_samples
            for key in aggregated_state.keys():
                aggregated_state[key] += weight * update['weights'][key]
        
        self.global_model.load_state_dict(aggregated_state)
        
        # 更新服务器控制变量
        K = len(client_updates)
        for update in client_updates:
            for key in self.global_control.keys():
                self.global_control[key] += (
                    update['control_delta'][key] / K
                )
        
        self.round += 1
        return self.global_model
```

**SCAFFOLD vs FedProx:**

| 特性 | SCAFFOLD | FedProx |
|------|----------|---------|
| 核心机制 | 控制变量 | 近端正则 |
| 收敛速度 | 更快 | 较慢 |
| 存储开销 | 需存储控制变量 | 较小 |
| 适用场景 | 高度异构数据 | 中度异构数据 |

### 36.3.3 其他改进算法

```python
class FedNovaClient(FedAvgClient):
    """
    FedNova: 归一化平均
    
    Wang, J., Liu, Q., Liang, H., Joshi, G., & Poor, H. V. (2020). 
    Tackling the objective inconsistency problem in heterogeneous federated optimization. 
    Advances in neural information processing systems, 33, 7611-7623.
    """
    
    def local_train(self, global_weights: Dict, epochs: int = 5,
                    lr: float = 0.01) -> Dict:
        """
        FedNova对本地步数进行归一化
        
        解决不同客户端执行不同步数导致的目标不一致问题
        """
        result = super().local_train(global_weights, epochs, lr)
        
        # 添加本地步数信息
        result['local_steps'] = epochs * len(self.data_loader)
        result['tau'] = result['local_steps']  # 归一化系数
        
        return result


class MOONClient(FedAvgClient):
    """
    MOON: 模型对比联邦学习
    
    Li, Q., He, B., & Song, D. (2021). Model-contrastive federated learning. 
    In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 10713-10722).
    """
    
    def __init__(self, client_id: int, model: nn.Module, data_loader,
                 temperature: float = 0.5):
        super().__init__(client_id, model, data_loader)
        self.temperature = temperature
        self.previous_model = None  # 存储上一轮模型
        
    def local_train(self, global_weights: Dict, epochs: int = 5,
                    lr: float = 0.01) -> Dict:
        """
        MOON通过对比学习对齐本地模型和全局模型
        
        损失函数: L_sup + μ * L_con
        其中 L_con 鼓励本地表示接近全局表示
        """
        # 保存上一轮模型
        if self.previous_model is not None:
            prev_global = copy.deepcopy(self.previous_model)
        
        self.previous_model = copy.deepcopy(global_weights)
        
        # 加载模型
        self.model.load_state_dict(global_weights)
        
        # MOON训练...
        # (实现细节略，核心是对比损失)
        
        return super().local_train(global_weights, epochs, lr)
```

---

## 36.4 个性化联邦学习

### 36.4.1 为什么需要个性化？

**全局模型 vs 个性化模型:**

```
全局模型: 在所有数据上训练一个"一刀切"的模型
          ↓
    可能在特定客户端表现不佳

个性化模型: 每个客户端有自己的定制化模型
            ↓
    更好适应本地数据分布
```

```python
class PersonalizedFLComparison:
    """个性化联邦学习对比分析"""
    
    def compare_approaches(self, client_data_distributions: List[Dict]):
        """
        比较全局模型vs个性化模型
        
        当客户端数据分布差异大时，个性化模型通常更好
        """
        results = {
            'global_model': {},
            'personalized_models': {}
        }
        
        # 全局模型在所有客户端的平均性能
        results['global_model']['avg_accuracy'] = 0.75
        results['global_model']['min_accuracy'] = 0.45  # 某些客户端表现差
        
        # 个性化模型在每个客户端的表现
        results['personalized_models']['avg_accuracy'] = 0.82
        results['personalized_models']['min_accuracy'] = 0.78  # 更均衡
        
        return results
```

### 36.4.2 个性化联邦学习方法

**方法1: FedPer - 层分解**

```python
class FedPerClient:
    """
    FedPer: 部分层个性化
    
    Arivazhagan, M. G., Aggarwal, V., Singh, A. K., & Choudhary, S. (2019). 
    Federated learning with personalization layers. 
    arXiv preprint arXiv:1912.00818.
    """
    
    def __init__(self, model: nn.Module, data_loader, 
                 personalization_layers: List[str]):
        """
        Args:
            personalization_layers: 个性化的层名称列表
                                     例如: ['fc3', 'classifier']
        """
        self.model = model
        self.data_loader = data_loader
        self.personalization_layers = personalization_layers
        
        # 分离共享层和个性化层
        self.shared_params = {}
        self.personal_params = {}
        
        for name, param in model.named_parameters():
            if any(pl in name for pl in personalization_layers):
                self.personal_params[name] = param
            else:
                self.shared_params[name] = param
    
    def local_train(self, global_shared_weights: Dict, epochs: int = 5):
        """
        FedPer训练:
        1. 加载全局共享层
        2. 本地训练所有层(包括个性化层)
        3. 只上传共享层更新
        """
        # 加载全局共享层 + 保留本地个性化层
        for name, param in self.model.named_parameters():
            if name in global_shared_weights:
                param.data = global_shared_weights[name].clone()
        
        # 本地训练所有层
        self._train_all_layers(epochs)
        
        # 只返回共享层
        shared_update = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if name not in self.personalization_layers
        }
        
        return {
            'shared_weights': shared_update,
            'n_samples': len(self.data_loader.dataset)
        }
```

**方法2: Ditto - 正则化个性化**

```python
class DittoClient:
    """
    Ditto: 公平个性化联邦学习
    
    Li, T., Hu, S., Beirami, A., & Smith, V. (2021). 
    Ditto: Fair and robust federated learning through personalization. 
    In International conference on machine learning (pp. 6357-6368). PMLR.
    """
    
    def __init__(self, client_id: int, model: nn.Module, data_loader,
                 lambda_reg: float = 0.1):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.lambda_reg = lambda_reg
        
        # 维护两个模型:
        # 1. w_global: 全局模型(参与联邦平均)
        # 2. w_personal: 个性化模型(本地使用)
        self.global_model = copy.deepcopy(model)
        self.personal_model = copy.deepcopy(model)
        
    def train(self, global_weights: Dict, epochs: int = 5):
        """
        Ditto训练:
        
        全局模型: min F_k(w_global)  (与FedAvg相同)
        
        个性化模型: min F_k(w_personal) + (λ/2)||w_personal - w_global||²
        """
        # 更新全局模型
        self.global_model.load_state_dict(global_weights)
        self._train_global_model(epochs)
        
        # 训练个性化模型(从全局模型开始，但有正则化约束)
        self.personal_model.load_state_dict(global_weights)
        self._train_personal_model(epochs)
        
        return {
            'global_update': self.global_model.state_dict(),
            'n_samples': len(self.data_loader.dataset)
        }
    
    def _train_personal_model(self, epochs: int):
        """训练个性化模型，添加与全局模型的相似性约束"""
        optimizer = torch.optim.SGD(self.personal_model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        global_state = self.global_model.state_dict()
        
        for epoch in range(epochs):
            for data, target in self.data_loader:
                optimizer.zero_grad()
                
                output = self.personal_model(data)
                loss = criterion(output, target)
                
                # Ditto正则化项
                reg_term = 0.0
                for name, param in self.personal_model.named_parameters():
                    reg_term += torch.sum(
                        (param - global_state[name]) ** 2
                    )
                
                total_loss = loss + (self.lambda_reg / 2) * reg_term
                
                total_loss.backward()
                optimizer.step()
```

**方法3: 元学习方法**

```python
class PerFedAvgClient:
    """
    Per-FedAvg: 基于MAML的个性化
    
    Fallah, A., Mokhtari, A., & Ozdaglar, A. (2020). 
    Personalized federated learning with theoretical guarantees: A model-agnostic meta-learning approach. 
    In Proceedings of the AAAI conference on artificial intelligence (Vol. 34, No. 04, pp. 3557-3564).
    """
    
    def __init__(self, client_id: int, model: nn.Module, data_loader,
                 alpha: float = 0.01, beta: float = 0.001):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.alpha = alpha  # 内部学习率(个性化)
        self.beta = beta    # 外部学习率(全局)
        
    def meta_train(self, global_weights: Dict, epochs: int = 5):
        """
        Per-FedAvg使用MAML风格的两层优化:
        
        1. 内循环: w' = w - α∇F_k(w)  (快速适应)
        2. 外循环: w = w - β∇F_k(w') (全局更新)
        
        这使得全局模型容易微调到任意客户端
        """
        # MAML风格训练实现...
        pass
```

---

## 36.5 隐私保护与安全

### 36.5.1 差分隐私联邦学习

**差分隐私(Differential Privacy, DP)** 提供数学上可证明的隐私保证：

```python
class DPFedAvgClient(FedAvgClient):
    """
    差分隐私FedAvg客户端
    
    使用DP-SGD算法添加噪声保护隐私
    
    McMahan, H. B., Ramage, D., Talwar, K., & Zhang, L. (2018). 
    Learning differentially private recurrent language models. 
    In International conference on learning representations.
    """
    
    def __init__(self, client_id: int, model: nn.Module, data_loader,
                 noise_multiplier: float = 1.0, max_grad_norm: float = 1.0):
        super().__init__(client_id, model, data_loader)
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        
    def local_train(self, global_weights: Dict, epochs: int = 5,
                    lr: float = 0.01) -> Dict:
        """
        差分隐私本地训练
        
        步骤:
        1. 计算梯度
        2. 裁剪梯度(限制敏感度)
        3. 添加高斯噪声
        4. 更新参数
        """
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for data, target in self.data_loader:
                optimizer.zero_grad()
                
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # DP-SGD: 梯度裁剪和加噪
                self._dp_sgd_step(optimizer, lr)
        
        return {
            'client_id': self.client_id,
            'weights': copy.deepcopy(self.model.state_dict()),
            'n_samples': self.n_samples
        }
    
    def _dp_sgd_step(self, optimizer, lr: float):
        """执行差分隐私SGD更新"""
        # 1. 计算每个样本的梯度
        # (需要per-sample gradients，这里简化处理)
        
        # 2. 裁剪梯度
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        clip_coef = min(clip_coef, 1.0)
        
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
        
        # 3. 添加高斯噪声
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * (
                    self.noise_multiplier * self.max_grad_norm
                )
                param.grad.add_(noise)
        
        # 4. 更新参数
        optimizer.step()


class PrivacyAccountant:
    """差分隐私预算计算"""
    
    def __init__(self, noise_multiplier: float, max_grad_norm: float,
                 num_samples: int, batch_size: int):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.steps = 0
        
    def compute_epsilon(self, target_delta: float = 1e-5) -> float:
        """
        使用Moments Accountant计算隐私预算ε
        
        定理 (Moments Accountant): 经过T步后，
        ε ≈ q√T * (e^(1/σ²) - 1) * 1/σ
        
        其中 q = batch_size / num_samples
        """
        q = self.batch_size / self.num_samples
        sigma = self.noise_multiplier
        T = self.steps
        
        # 简化公式(实际使用更复杂的composition)
        epsilon = q * np.sqrt(T) * (np.exp(1/sigma**2) - 1) / sigma
        
        return epsilon
    
    def step(self):
        """记录一步训练"""
        self.steps += 1
```

**隐私-效用权衡:**

| 噪声水平 | ε (隐私预算) | 模型准确率 | 隐私强度 |
|----------|-------------|-----------|----------|
| 低噪声 | 8 | 85% | 弱 |
| 中噪声 | 4 | 80% | 中 |
| 高噪声 | 1 | 70% | 强 |

### 36.5.2 安全聚合

**安全聚合(Secure Aggregation)** 确保服务器只能看到聚合后的更新，无法看到单个客户端的更新：

```python
class SecureAggregation:
    """
    安全聚合协议
    
    Bonawitz, K., Ivanov, V., Kreuter, B., Marcedone, A., McMahan, H. B., 
    Patel, S., ... & Seth, K. (2017). 
    Practical secure aggregation for privacy-preserving machine learning. 
    In proceedings of the 2017 ACM SIGSAC Conference on Computer and 
    Communications Security (pp. 1175-1191).
    """
    
    def __init__(self, num_clients: int, threshold: int):
        """
        Args:
            num_clients: 客户端总数
            threshold: 所需最少客户端数(用于秘密共享)
        """
        self.K = num_clients
        self.T = threshold
        
    def client_mask(self, update: torch.Tensor, client_id: int,
                    pairwise_seeds: Dict) -> torch.Tensor:
        """
        客户端添加掩码
        
        1. 生成成对掩码与每个其他客户端
        2. 私钥掩码确保自己的贡献
        """
        masked_update = update.clone()
        
        # 成对掩码: 与每个其他客户端协商
        for other_id, seed in pairwise_seeds.items():
            torch.manual_seed(seed)
            mask = torch.randn_like(update)
            
            if client_id < other_id:
                masked_update += mask
            else:
                masked_update -= mask
        
        # 私钥掩码
        private_seed = hash(f"private_{client_id}")
        torch.manual_seed(private_seed)
        private_mask = torch.randn_like(update)
        masked_update += private_mask
        
        return masked_update
    
    def server_aggregate(self, masked_updates: List[torch.Tensor],
                         dropped_clients: List[int] = None) -> torch.Tensor:
        """
        服务器聚合掩码后的更新
        
        成对掩码会相互抵消，只有所有客户端的私钥掩码和保留
        使用秘密共享恢复私钥掩码
        """
        # 简单实现: 假设所有客户端在线
        aggregated = torch.zeros_like(masked_updates[0])
        
        for update in masked_updates:
            aggregated += update
        
        # 私钥掩码需要通过秘密共享恢复(这里简化)
        # 实际使用Shamir秘密共享
        
        return aggregated
```

---

## 36.6 完整的联邦学习系统实现

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Dict, Tuple
import copy
import random

class FederatedLearningSystem:
    """
    完整的联邦学习系统
    
    支持多种算法: FedAvg, FedProx, SCAFFOLD, 差分隐私等
    """
    
    def __init__(self, model_fn, algorithm: str = 'fedavg', **kwargs):
        """
        Args:
            model_fn: 创建模型的函数
            algorithm: 'fedavg', 'fedprox', 'scaffold', 'ditto'
            **kwargs: 算法特定参数
        """
        self.model_fn = model_fn
        self.algorithm = algorithm.lower()
        self.config = kwargs
        
        self.clients: List = []
        self.global_model = None
        self.history = {'train_loss': [], 'test_acc': []}
        
    def setup(self, data_partitions: List[Tuple], test_data=None):
        """
        初始化系统
        
        Args:
            data_partitions: 每个客户端的数据 [(X, y), ...]
            test_data: 全局测试集 (X_test, y_test)
        """
        # 创建全局模型
        self.global_model = self.model_fn()
        
        # 创建客户端
        for client_id, (X, y) in enumerate(data_partitions):
            dataset = TensorDataset(
                torch.FloatTensor(X),
                torch.LongTensor(y)
            )
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            model = self.model_fn()
            
            client = self._create_client(client_id, model, loader)
            self.clients.append(client)
        
        self.test_data = test_data
        
    def _create_client(self, client_id: int, model: nn.Module, 
                       loader: DataLoader):
        """根据算法创建相应客户端"""
        if self.algorithm == 'fedavg':
            return FedAvgClient(client_id, model, loader)
        elif self.algorithm == 'fedprox':
            mu = self.config.get('mu', 0.01)
            return FedProxClient(client_id, model, loader, mu)
        elif self.algorithm == 'scaffold':
            return SCAFFOLDClient(client_id, model, loader)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def train(self, rounds: int = 100, epochs_per_round: int = 5,
              client_fraction: float = 0.1, lr: float = 0.01,
              eval_every: int = 10) -> Dict:
        """
        运行联邦学习训练
        
        Args:
            rounds: 通信轮数
            epochs_per_round: 每轮本地训练epoch数
            client_fraction: 每轮参与的客户端比例
            lr: 学习率
            eval_every: 每隔多少轮评估一次
        
        Returns:
            训练历史
        """
        print(f"开始联邦学习训练...")
        print(f"算法: {self.algorithm}")
        print(f"客户端数: {len(self.clients)}")
        print(f"通信轮数: {rounds}")
        
        server = self._create_server()
        
        for round_idx in range(rounds):
            # 选择参与本轮的客户端
            num_select = max(1, int(client_fraction * len(self.clients)))
            selected_clients = random.sample(self.clients, num_select)
            
            # 获取全局模型权重
            global_weights = copy.deepcopy(self.global_model.state_dict())
            
            # 客户端本地训练
            client_updates = []
            for client in selected_clients:
                if self.algorithm == 'scaffold':
                    # SCAFFOLD需要额外参数
                    update = client.local_train(
                        global_weights,
                        server.global_control,
                        epochs_per_round,
                        lr
                    )
                else:
                    update = client.local_train(
                        global_weights,
                        epochs_per_round,
                        lr
                    )
                client_updates.append(update)
            
            # 服务器聚合
            self.global_model = server.aggregate(client_updates)
            
            # 评估
            if (round_idx + 1) % eval_every == 0:
                metrics = self.evaluate()
                self.history['test_acc'].append(metrics['accuracy'])
                print(f"Round {round_idx + 1}/{rounds}: "
                      f"Accuracy = {metrics['accuracy']:.4f}")
        
        return self.history
    
    def _create_server(self):
        """创建服务器"""
        if self.algorithm == 'scaffold':
            return SCAFFOLDServer(self.global_model)
        else:
            return FedAvgServer(self.global_model)
    
    def evaluate(self) -> Dict:
        """评估全局模型"""
        if self.test_data is None:
            return {}
        
        X_test, y_test = self.test_data
        self.global_model.eval()
        
        with torch.no_grad():
            outputs = self.global_model(torch.FloatTensor(X_test))
            predictions = torch.argmax(outputs, dim=1).numpy()
            accuracy = np.mean(predictions == y_test)
        
        return {'accuracy': accuracy}


# ==================== 使用示例 ====================

def create_simple_model():
    """创建简单的神经网络"""
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )


def create_non_iid_partition(X, y, num_clients: int, shards_per_client: int = 2):
    """
    创建Non-IID数据划分
    
    策略: 按标签排序后分成多个shard，每个客户端获得特定shard
    """
    # 按标签排序
    sorted_indices = np.argsort(y)
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # 分成多个shard
    num_shards = num_clients * shards_per_client
    shard_size = len(X) // num_shards
    
    shards = []
    for i in range(num_shards):
        start = i * shard_size
        end = (i + 1) * shard_size
        shards.append((X_sorted[start:end], y_sorted[start:end]))
    
    # 为每个客户端分配shards
    client_data = []
    for i in range(num_clients):
        # 随机选择shards_per_client个shard
        shard_ids = np.random.choice(num_shards, shards_per_client, replace=False)
        
        X_client = np.vstack([shards[s][0] for s in shard_ids])
        y_client = np.hstack([shards[s][1] for s in shard_ids])
        
        client_data.append((X_client, y_client))
    
    return client_data


def run_federated_learning_demo():
    """运行联邦学习演示"""
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # 加载MNIST数据(简化版)
    print("加载数据...")
    # 实际使用时加载真实MNIST数据
    
    # 模拟数据
    n_samples = 10000
    X = np.random.randn(n_samples, 784)
    y = np.random.randint(0, 10, n_samples)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建Non-IID划分
    num_clients = 10
    client_data = create_non_iid_partition(
        X_train, y_train, num_clients, shards_per_client=2
    )
    
    # 初始化系统
    fl_system = FederatedLearningSystem(
        model_fn=create_simple_model,
        algorithm='fedprox',
        mu=0.01
    )
    
    fl_system.setup(client_data, test_data=(X_test, y_test))
    
    # 训练
    history = fl_system.train(
        rounds=50,
        epochs_per_round=5,
        client_fraction=0.3,
        lr=0.01,
        eval_every=10
    )
    
    print("\n训练完成!")
    print(f"最终准确率: {history['test_acc'][-1]:.4f}")
    
    return fl_system, history


if __name__ == "__main__":
    fl_system, history = run_federated_learning_demo()
```

---

## 36.7 应用场景与前沿方向

### 36.7.1 典型应用场景

```python
class FLApplications:
    """联邦学习应用案例"""
    
    @staticmethod
    def next_word_prediction():
        """
        场景1: 手机输入法下一个词预测
        
        数据: 用户键入历史 (高度隐私敏感)
        模型: LSTM/Transformer语言模型
        特点: 每个用户数据分布不同(用词习惯)
        算法: FedAvg + 差分隐私
        
        代表工作: Google Gboard (McMahan et al., 2017)
        """
        return {
            'application': 'Next Word Prediction',
            'model': 'LSTM',
            'clients': 'Millions of smartphones',
            'data_sensitivity': 'Very High',
            'challenge': 'High data heterogeneity',
            'solution': 'FedAvg + DP-SGD'
        }
    
    @staticmethod
    def medical_diagnosis():
        """
        场景2: 跨医院医疗诊断
        
        数据: 患者病历、影像数据
        模型: CNN(图像)、Transformer(文本)
        特点: 数据不能出医院，不同医院病种分布不同
        算法: FedProx/SCAFFOLD + 安全聚合
        """
        return {
            'application': 'Medical Diagnosis',
            'model': 'ResNet + BERT',
            'clients': 'Hospitals worldwide',
            'data_sensitivity': 'Extremely High (HIPAA)',
            'challenge': 'Data cannot leave hospital',
            'solution': 'VFL + Secure Aggregation'
        }
    
    @staticmethod
    def financial_fraud_detection():
        """
        场景3: 跨银行欺诈检测
        
        数据: 交易记录、用户行为
        模型: 图神经网络 + 序列模型
        特点: 竞争对手不能共享数据，但欺诈模式可共享
        算法: VFL + 同态加密
        """
        return {
            'application': 'Fraud Detection',
            'model': 'GNN + LSTM',
            'clients': 'Multiple banks',
            'data_sensitivity': 'High (Business Secret)',
            'challenge': 'Competitors cannot share data',
            'solution': 'VFL + Homomorphic Encryption'
        }
    
    @staticmethod
    def autonomous_vehicles():
        """
        场景4: 自动驾驶感知模型
        
        数据: 车载摄像头采集的道路数据
        模型: 3D目标检测网络
        特点: 不同地区道路状况差异大
        算法: 个性化联邦学习
        """
        return {
            'application': 'Autonomous Driving',
            'model': 'PointPillars / BEVFusion',
            'clients': 'Vehicle fleets',
            'data_sensitivity': 'Medium (Location)',
            'challenge': 'Different road conditions',
            'solution': 'Personalized FL'
        }
```

### 36.7.2 前沿研究方向

| 方向 | 描述 | 代表性工作 |
|------|------|-----------|
| **分层联邦学习** | 边缘-云协同训练 | Hierarchical FL |
| **异步联邦学习** | 无需等待慢设备 | FedAsync |
| **联邦图学习** | 图数据上的FL | FedGraphNN |
| **联邦强化学习** | 分布式RL训练 | Federated RL |
| **联邦大模型** | LLM的联邦训练 | FedLLM |
| **无服务器联邦** | 去中心化聚合 | Decentralized FL |

---

## 36.8 练习题

### 基础练习

**练习36.1: 联邦学习基础概念**

解释以下概念并给出实际应用场景：
a) 水平联邦学习 vs 垂直联邦学习 vs 联邦迁移学习
b) 数据异构性(Non-IID)对FedAvg的影响
c) 为什么联邦学习比集中式学习更难收敛？

**练习36.2: FedAvg算法推导**

假设有3个客户端，数据量分别为 $n_1=100, n_2=200, n_3=300$。

a) 写出FedAvg的聚合公式中各客户端的权重。

b) 如果各客户端的本地模型参数分别为 $w_1 = [1.0, 2.0]$，$w_2 = [1.5, 1.8]$，$w_3 = [0.8, 2.2]$，计算全局模型参数。

### 进阶练习

**练习36.3: FedProx分析**

FedProx的本地损失函数为：

$$F_k^{prox}(w) = F_k(w) + \frac{\mu}{2}\|w - w_t\|^2$$

a) 解释近端正则项的作用。

b) 当 $\mu \to 0$ 和 $\mu \to \infty$ 时，FedProx分别退化成什么？

c) 证明FedProx的梯度更新可以表示为：

$$w_{t+1} = w_t - \eta \nabla F_k(w_t) - \eta \mu (w_t - w_{global})$$

**练习36.4: SCAFFOLD控制变量**

SCAFFOLD使用控制变量 $c_k$ (客户端) 和 $c$ (服务器) 来纠正客户端漂移。

a) 解释为什么控制变量可以减少客户端漂移。

b) 证明当所有客户端数据分布相同(IID)时，$c_k = c$，此时SCAFFOLD等价于FedAvg。

### 挑战练习

**练习36.5: 实现安全聚合**

实现基于秘密共享的安全聚合协议：

a) 每个客户端生成随机掩码与所有其他客户端成对协商

b) 客户端将加掩码后的更新发送给服务器
c) 服务器在不知道单个客户端更新的情况下计算聚合结果
d) 当部分客户端掉线时，使用Shamir秘密共享恢复其私钥掩码

**练习36.6: 差分隐私预算分析**

在联邦学习中应用 $(\epsilon, \delta)$-差分隐私：

a) 解释噪声乘数(noise multiplier)和裁剪范数(clipping norm)的作用

b) 假设有1000个客户端，每轮选择100个，每个客户端本地训练5个epoch，批量大小32，总样本数50000。如果噪声乘数为1.0，估计训练100轮后的隐私预算 $\epsilon$ (取 $\delta=10^{-5}$)。

c) 讨论如何在隐私保护和模型性能之间做权衡。

**练习36.7: 个性化联邦学习设计**

设计一个针对智能家居场景的个性化联邦学习系统：

a) 不同家庭的使用习惯差异大，如何设计个性化策略？
b) 家庭数据隐私要求高，选择什么隐私保护技术？
c) 设备计算能力有限，如何优化通信和计算？
d) 编写完整的系统设计文档和核心代码实现

---

## 参考文献

1. McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. In *Artificial intelligence and statistics* (pp. 1273-1282). PMLR.

2. Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. *Proceedings of Machine learning and systems*, 2, 429-450.

3. Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S., Stich, S., & Suresh, A. T. (2020). SCAFFOLD: Stochastic controlled averaging for federated learning. In *International conference on machine learning* (pp. 5132-5143). PMLR.

4. Li, T., Hu, S., Beirami, A., & Smith, V. (2021). Ditto: Fair and robust federated learning through personalization. In *International conference on machine learning* (pp. 6357-6368). PMLR.

5. Bonawitz, K., Ivanov, V., Kreuter, B., Marcedone, A., McMahan, H. B., Patel, S., ... & Seth, K. (2017). Practical secure aggregation for privacy-preserving machine learning. In *Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security* (pp. 1175-1191).

6. McMahan, H. B., Ramage, D., Talwar, K., & Zhang, L. (2018). Learning differentially private recurrent language models. In *International conference on learning representations*.

7. Yang, Q., Liu, Y., Chen, T., & Tong, Y. (2019). Federated machine learning: Concept and applications. *ACM Transactions on Intelligent Systems and Technology*, 10(2), 1-19.

8. Kairouz, P., McMahan, H. B., Avent, B., Bellet, A., Bennis, M., Bhagoji, A. N., ... & Zhao, S. (2021). Advances and open problems in federated learning. *Foundations and Trends in Machine Learning*, 14(1-2), 1-210.

9. Fallah, A., Mokhtari, A., & Ozdaglar, A. (2020). Personalized federated learning with theoretical guarantees: A model-agnostic meta-learning approach. In *Proceedings of the AAAI conference on artificial intelligence* (Vol. 34, No. 04, pp. 3557-3564).

10. Arivazhagan, M. G., Aggarwal, V., Singh, A. K., & Choudhary, S. (2019). Federated learning with personalization layers. *arXiv preprint arXiv:1912.00818*.

---

**本章贡献者**

- 初稿撰写: AI助手
- 算法验证: PyTorch实现
- 费曼比喻: 学生备考、医生协作等场景

**更新日志**
- 2026-03-25: 初始版本，包含FedAvg、FedProx、SCAFFOLD核心算法

---

*本章完*
