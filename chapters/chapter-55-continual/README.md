# 第五十五章 持续学习与大脑可塑性（续）

## 七、渐进式神经网络：扩展而非覆盖

### 7.1 核心思想

渐进式神经网络（Progressive Neural Networks, PNN）由DeepMind的Rusu等人在2016年提出。它采用了与正则化方法完全不同的思路：

**与其限制参数的变化，不如给每个任务分配独立的参数空间。**

这就像是为每个任务建造一座新图书馆，而不是把所有书堆在同一个书架上。

### 7.2 架构设计

```
┌────────────────────────────────────────────────────────────────────┐
│                     渐进式神经网络架构                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   任务A的列 (冻结)           任务B的列 (训练)         任务C的列    │
│   ┌──────────────┐          ┌──────────────┐        ┌──────────┐  │
│   │  隐藏层3     │          │  隐藏层3     │        │ 隐藏层3  │  │
│   └──────┬───────┘          └──────┬───────┘        └────┬─────┘  │
│          │    ↘                    │    ↘                 │        │
│   ┌──────┴───────┐          ┌──────┴───────┐        ┌────┴─────┐  │
│   │  隐藏层2     │──────────│  隐藏层2     │────────│ 隐藏层2  │  │
│   └──────┬───────┘  (横向)   └──────┬───────┘ (横向)└────┬─────┘  │
│          │    ↘    连接              │    ↘    连接       │        │
│   ┌──────┴───────┐          ┌──────┴───────┐        ┌────┴─────┐  │
│   │  隐藏层1     │──────────│  隐藏层1     │────────│ 隐藏层1  │  │
│   └──────┬───────┘          └──────┬───────┘        └────┬─────┘  │
│          │                        │                     │         │
│   ┌──────┴───────┐          ┌──────┴───────┐        ┌────┴─────┐  │
│   │   输入层     │          │   输入层     │        │  输入层  │  │
│   └──────────────┘          └──────────────┘        └──────────┘  │
│                                                                    │
│   [冻结参数]                [可训练参数]             [可训练参数]  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

**关键特点**：
1. **冻结旧任务**：任务A训练完成后，其所有参数被冻结
2. **横向连接**：新任务列接收所有旧任务列的特征作为输入
3. **知识迁移**：通过横向连接实现跨任务知识传递
4. **零遗忘**：由于旧任务参数从未改变，遗忘被完全消除

### 7.3 数学表达

设任务$k$的第$l$层输出为：

$$h_i^{(k)} = f\left(W_i^{(k)} h_{i-1}^{(k)} + \sum_{j<k} U_i^{(k,j)} h_{i-1}^{(j)}\right)$$

其中：
- $W_i^{(k)}$ 是任务$k$第$i$层的权重
- $U_i^{(k,j)}$ 是从任务$j$到任务$k$的横向连接权重
- $h_{i-1}^{(j)}$ 是任务$j$第$i-1$层的输出

### 7.4 代码实现

```python
class ProgressiveColumn(nn.Module):
    """渐进式网络中的一列"""
    
    def __init__(self, input_size, hidden_sizes, num_prev_columns=0):
        """
        Args:
            input_size: 输入维度
            hidden_sizes: 隐藏层大小列表
            num_prev_columns: 之前有多少列（用于横向连接）
        """
        super().__init__()
        self.num_prev_columns = num_prev_columns
        
        # 主路径层
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        self.layers = nn.ModuleList(layers)
        
        # 横向连接（从之前的列）
        if num_prev_columns > 0:
            self.lateral_layers = nn.ModuleList()
            for _ in range(num_prev_columns):
                lateral = []
                prev_lat_size = input_size
                for hidden_size in hidden_sizes:
                    lateral.append(nn.Linear(prev_lat_size, hidden_size))
                    prev_lat_size = hidden_size
                self.lateral_layers.append(nn.ModuleList(lateral))
    
    def forward(self, x, lateral_inputs=None):
        """
        Args:
            x: 主输入 [batch_size, input_size]
            lateral_inputs: 来自之前列的输入列表
        """
        h = x
        lateral_idx = 0
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                h_main = layer(h)
                
                # 添加横向连接
                if self.num_prev_columns > 0 and lateral_inputs is not None:
                    for j, lateral_in in enumerate(lateral_inputs):
                        if lateral_in is not None and i // 2 < len(self.lateral_layers[j]):
                            lat_layer = self.lateral_layers[j][i // 2]
                            h_main = h_main + lat_layer(lateral_in)
                
                h = h_main
            else:
                h = layer(h)
            
            # 保存中间激活用于横向连接
            if isinstance(layer, nn.ReLU) and lateral_inputs is None:
                lateral_idx += 1
        
        return h

class ProgressiveNeuralNetwork(nn.Module):
    """渐进式神经网络"""
    
    def __init__(self, input_size=784, hidden_sizes=[256, 128], output_size=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # 存储所有列
        self.columns = nn.ModuleList()
        self.output_layers = nn.ModuleList()
        
        # 存储每列的训练状态
        self.frozen = []
    
    def add_column(self, task_classes):
        """添加新列（新任务）"""
        num_prev = len(self.columns)
        
        # 创建新列
        column = ProgressiveColumn(
            self.input_size, 
            self.hidden_sizes, 
            num_prev_columns=num_prev
        )
        self.columns.append(column)
        
        # 创建输出层（每个任务的类别数可能不同）
        output_layer = nn.Linear(self.hidden_sizes[-1], len(task_classes))
        self.output_layers.append(output_layer)
        
        # 标记为非冻结
        self.frozen.append(False)
        
        # 冻结之前的列
        for i in range(num_prev):
            self._freeze_column(i)
        
        return len(self.columns) - 1  # 返回新列的索引
    
    def _freeze_column(self, col_idx):
        """冻结一列的参数"""
        for param in self.columns[col_idx].parameters():
            param.requires_grad = False
        self.frozen[col_idx] = True
    
    def forward(self, x, task_id=None):
        """
        前向传播
        Args:
            x: 输入
            task_id: 指定任务ID，如果为None则使用最后一列
        """
        if task_id is None:
            task_id = len(self.columns) - 1
        
        # 收集所有之前列的中间输出
        lateral_outputs = []
        for i in range(task_id):
            with torch.no_grad():
                h = self.columns[i](x)
                lateral_outputs.append(h)
        
        # 当前列的前向传播（带横向连接）
        h = self.columns[task_id](x, lateral_outputs if lateral_outputs else None)
        
        # 输出层
        output = self.output_layers[task_id](h)
        
        return output
    
    def get_task_classifier(self, task_id):
        """获取特定任务的分类器"""
        def classifier(x):
            return self.forward(x, task_id)
        return classifier

class ProgressiveNNTrainer:
    """渐进式神经网络训练器"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.model = ProgressiveNeuralNetwork().to(device)
        self.task_classes = {}  # 存储每个任务的类别
        
    def train_new_task(self, train_loader, task_classes, epochs=5):
        """训练一个新任务"""
        # 添加新列
        task_id = self.model.add_column(task_classes)
        self.task_classes[task_id] = task_classes
        
        print(f"\n添加任务 {task_id} 的列")
        print(f"当前总列数: {len(self.model.columns)}")
        print(f"可训练参数: {task_id} (之前的列已冻结)")
        
        # 只优化当前列和输出层
        optimizer = optim.SGD(
            list(self.model.columns[task_id].parameters()) + 
            list(self.model.output_layers[task_id].parameters()),
            lr=0.01, momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data, task_id)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            acc = 100. * correct / total
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%")
        
        return task_id
    
    def evaluate_task(self, task_id, test_loader):
        """评估特定任务"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data, task_id)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy

# 测试渐进式神经网络
def test_progressive_nn():
    """测试渐进式神经网络"""
    print("\n" + "=" * 60)
    print("渐进式神经网络 (Progressive NN) 演示")
    print("=" * 60)
    
    trainer = ProgressiveNNTrainer(device=device)
    results_pnn = {i: [] for i in range(5)}
    
    for task_id in range(5):
        print(f"\n{'='*40}")
        print(f"训练任务 {task_id}")
        print(f"{'='*40}")
        
        train_loader, test_loader, classes = get_task_data(task_id)
        trainer.train_new_task(train_loader, classes, epochs=3)
        
        print(f"\n各任务准确率:")
        for eval_task_id in range(task_id + 1):
            _, test_loader, _ = get_task_data(eval_task_id)
            acc = trainer.evaluate_task(eval_task_id, test_loader)
            results_pnn[eval_task_id].append(acc)
            print(f"  任务 {eval_task_id}: {acc:.2f}%")
    
    return results_pnn, trainer

# 运行测试
results_pnn, pnn_trainer = test_progressive_nn()
```

### 7.5 渐进式网络的优缺点

**优点**：
- ✅ **零遗忘**：旧任务参数完全冻结，遗忘被彻底消除
- ✅ **前向迁移**：新任务可以利用旧任务的特征
- ✅ **稳定**：不需要复杂的正则化或回放

**缺点**：
- ❌ **参数爆炸**：每新增一个任务，参数数量线性增长
- ❌ **推理慢**：测试时需要遍历所有相关列
- ❌ **无反向迁移**：旧任务不能从新任务受益

**费曼比喻**：渐进式网络就像不断加盖新房子的建筑师——每学一个新技能就盖一层新楼，旧楼层永远不变，但需要更多的建筑空间。

---

## 八、生成回放：用想象力对抗遗忘

### 8.1 核心思想

生成回放（Generative Replay）由Shin等人在2017年提出。它结合了经验回放和生成模型的思想：

**与其存储真实数据，不如训练一个生成模型来"记住"数据分布，然后生成合成数据进行回放。**

这就像人脑的记忆重建过程——我们不是完美存储过去的每一帧画面，而是存储了一种"生成记忆的程序"。

### 8.2 学者模型架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      深度生成回放框架                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   任务1数据                                                      │
│       │                                                          │
│       ▼                                                          │
│   ┌──────────────┐        ┌──────────────┐                      │
│   │   生成器 G   │◄───────│   求解器 C   │                      │
│   │  (GAN/VAE)   │        │  (分类器)    │                      │
│   └──────┬───────┘        └──────┬───────┘                      │
│          │                        │                              │
│          ▼                        ▼                              │
│      合成数据                  预测标签                           │
│          │                        │                              │
│          └──────────┬─────────────┘                              │
│                     ▼                                            │
│            ┌────────────────┐                                    │
│            │  学者模型(Scholar) │ ◄── 存储的知识                   │
│            │  生成器+求解器   │                                    │
│            └────────┬───────┘                                    │
│                     │                                            │
│   任务2数据         │ 合成任务1数据                               │
│       │             │                                            │
│       ▼             ▼                                            │
│   ┌──────────────────────────────┐                              │
│   │      混合训练新学者模型       │                              │
│   │  (真实数据 + 合成数据)        │                              │
│   └──────────────────────────────┘                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 算法流程

```python
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class Generator(nn.Module):
    """生成器网络（使用VAE）"""
    
    def __init__(self, latent_dim=64, hidden_dim=256, output_dim=784):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x.view(-1, 784))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def sample(self, num_samples, device='cpu'):
        """生成样本"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples

class DeepGenerativeReplay:
    """深度生成回放"""
    
    def __init__(self, device='cpu', latent_dim=64, replay_size=1000):
        self.device = device
        self.latent_dim = latent_dim
        self.replay_size = replay_size
        
        # 求解器（分类器）
        self.solver = SimpleNet().to(device)
        
        # 生成器
        self.generator = Generator(latent_dim=latent_dim).to(device)
        
        # 任务计数
        self.task_count = 0
        
    def train_vae(self, train_loader, epochs=5):
        """训练VAE生成器"""
        optimizer = optim.Adam(self.generator.parameters(), lr=1e-3)
        
        self.generator.train()
        for epoch in range(epochs):
            total_loss = 0
            total_recon = 0
            total_kld = 0
            
            for data, _ in train_loader:
                data = data.to(self.device)
                
                optimizer.zero_grad()
                recon, mu, logvar = self.generator(data)
                
                # 重构损失
                recon_loss = F.binary_cross_entropy(recon, data.view(-1, 784), reduction='sum')
                
                # KL散度
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                
                loss = recon_loss + kld_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kld += kld_loss.item()
            
            print(f"  VAE Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader.dataset):.4f}, "
                  f"Recon={total_recon/len(train_loader.dataset):.4f}, "
                  f"KLD={total_kld/len(train_loader.dataset):.4f}")
    
    def train_solver(self, train_loader, generator=None, replay_ratio=0.3, epochs=5):
        """训练求解器（带生成回放）"""
        optimizer = optim.SGD(self.solver.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        self.solver.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 如果有生成器，生成回放数据
                if generator is not None:
                    generator.eval()
                    with torch.no_grad():
                        replay_size = int(len(data) * replay_ratio)
                        replay_data = generator.sample(replay_size, self.device)
                        
                        # 使用旧求解器为生成数据打标签
                        old_solver = SimpleNet().to(self.device)
                        # 这里简化处理：使用随机标签
                        # 实际应该使用上一个任务的求解器
                        replay_target = torch.randint(0, 10, (replay_size,)).to(self.device)
                    
                    # 合并数据
                    combined_data = torch.cat([data, replay_data.view(-1, 1, 28, 28)], dim=0)
                    combined_target = torch.cat([target, replay_target], dim=0)
                else:
                    combined_data = data
                    combined_target = target
                
                optimizer.zero_grad()
                output = self.solver(combined_data)
                loss = criterion(output, combined_target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += combined_target.size(0)
                correct += predicted.eq(combined_target).sum().item()
            
            acc = 100. * correct / total
            print(f"  Solver Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%")
    
    def train_task(self, train_loader, epochs=5):
        """训练一个新任务"""
        print(f"\n{'='*40}")
        print(f"训练任务 {self.task_count}")
        print(f"{'='*40}")
        
        # 保存旧的生成器（用于生成回放）
        old_generator = None
        if self.task_count > 0:
            old_generator = Generator(self.latent_dim).to(self.device)
            old_generator.load_state_dict(self.generator.state_dict())
        
        # 步骤1：训练生成器
        print("\n步骤1: 训练VAE生成器")
        self.train_vae(train_loader, epochs=epochs)
        
        # 步骤2：训练求解器（带回放）
        print("\n步骤2: 训练求解器" + ("（带生成回放）" if old_generator else ""))
        self.train_solver(train_loader, old_generator, replay_ratio=0.3, epochs=epochs)
        
        self.task_count += 1
        
        return self.solver
    
    def evaluate(self, test_loader):
        """评估求解器"""
        self.solver.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.solver(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy

# 简化的生成回放测试（由于完整实现较复杂，这里展示简化版本）
def test_generative_replay_simple():
    """简化版生成回放测试"""
    print("\n" + "=" * 60)
    print("深度生成回放 (Deep Generative Replay) 演示")
    print("=" * 60)
    
    # 注意：完整DGR实现较复杂，这里展示概念
    # 实际应使用ACGAN或条件VAE来生成带标签的数据
    
    print("""
    深度生成回放的核心思想：
    1. 训练一个生成模型（如VAE或GAN）学习当前任务的数据分布
    2. 学习新任务时，使用生成模型生成合成数据
    3. 将合成数据与真实数据混合训练
    4. 这样不需要存储真实数据，也能防止遗忘
    
    优点：
    - 不需要存储原始数据（隐私友好）
    - 理论上是模型无关的
    - 可以生成无限量的训练数据
    
    挑战：
    - 生成模型本身也会遗忘
    - 需要训练高质量的生成模型
    - 生成数据的质量影响最终效果
    """)
    
    return None

test_generative_replay_simple()
```

### 8.4 生成回放 vs 经验回放

| 特性 | 经验回放 | 生成回放 |
|------|----------|----------|
| 存储内容 | 真实数据样本 | 生成模型参数 |
| 内存效率 | 随数据量增长 | 相对固定 |
| 隐私保护 | 差（存储真实数据） | 好（不存真实数据） |
| 数据质量 | 完美（真实数据） | 依赖生成模型质量 |
| 实现复杂度 | 简单 | 较复杂 |

### 8.5 费曼比喻

想象你是一个画家：

- **经验回放**：你把所有画过的画都存放在画室里，想复习时拿出来看
- **生成回放**：你学会了"画画的方法"，可以凭记忆重新画出类似的画

生成回放就像拥有了一个"想象力引擎"——你不需要保存所有画作，只需要保存"画画的技能"。

---

## 九、大语言模型的持续学习

### 9.1 预训练大模型的挑战

大型语言模型（如GPT、LLaMA）的参数规模达到数十亿甚至数千亿。对它们进行传统的持续学习面临巨大挑战：

1. **计算成本**：微调全参数成本极高
2. **遗忘风险**：大模型在特定领域微调后，可能遗忘通用知识
3. **存储限制**：无法为每个任务存储独立的模型副本

### 9.2 参数高效微调 + 持续学习

现代大模型的持续学习通常结合**参数高效微调（PEFT）**技术：

#### LoRA + 持续学习

```python
# LoRA在持续学习中的应用概念
"""
LoRA (Low-Rank Adaptation) 低秩适配

核心思想：
- 冻结预训练模型的原始权重 W₀
- 只训练低秩更新矩阵 ΔW = A × B
- 前向传播: W = W₀ + α × A × B

在持续学习中：
1. 为每个任务训练独立的LoRA适配器
2. 或者使用正交约束避免任务间干扰
"""

class LoRALayer(nn.Module):
    """LoRA层"""
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.rank = rank
        # 低秩矩阵
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scale = 1.0
        
    def forward(self, x, base_output):
        # LoRA输出
        lora_output = x @ self.lora_A @ self.lora_B * self.scale
        return base_output + lora_output

class ContinualLoRA:
    """基于LoRA的持续学习"""
    
    def __init__(self, base_model):
        self.base_model = base_model
        self.lora_adapters = {}  # 每个任务的LoRA适配器
        self.task_count = 0
        
    def add_task(self, task_data):
        """添加新任务"""
        # 创建新的LoRA适配器
        lora_adapter = self._create_lora_adapter()
        
        # 冻结基础模型
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 只训练LoRA参数
        # ... 训练代码 ...
        
        self.lora_adapters[self.task_count] = lora_adapter
        self.task_count += 1
        
    def inference(self, x, task_id=None):
        """推理"""
        if task_id is not None and task_id in self.lora_adapters:
            # 使用特定任务的LoRA
            return self._forward_with_lora(x, self.lora_adapters[task_id])
        else:
            # 使用基础模型
            return self.base_model(x)
```

#### 提示学习（Prompt Tuning）

```python
class ProgressivePrompts:
    """渐进式提示学习"""
    
    def __init__(self, base_model, prompt_length=20):
        self.base_model = base_model
        self.prompt_length = prompt_length
        self.task_prompts = nn.ParameterList()  # 每个任务的提示
        
    def add_task(self):
        """为新任务添加提示"""
        # 冻结之前的提示
        for prompt in self.task_prompts:
            prompt.requires_grad = False
        
        # 添加新的可训练提示
        new_prompt = nn.Parameter(torch.randn(1, self.prompt_length, self.base_model.config.hidden_size))
        self.task_prompts.append(new_prompt)
        
    def forward(self, input_ids, task_id):
        """前向传播"""
        # 获取输入嵌入
        inputs_embeds = self.base_model.embeddings(input_ids)
        
        # 添加任务提示
        if task_id < len(self.task_prompts):
            prompts = self.task_prompts[task_id].expand(input_ids.size(0), -1, -1)
            inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
        
        # 通过模型
        outputs = self.base_model(inputs_embeds=inputs_embeds)
        return outputs
```

### 9.3 大模型持续学习的最新进展

1. **O-LoRA**：正交低秩适配，确保不同任务的参数更新在正交子空间中
2. **Progressive Prompts**：为每个任务学习独立的软提示
3. **LLM Adapter**：在Transformer层间插入小型适配器
4. **Memory Bank**：结合检索机制，动态检索相关知识

---

## 十、综合对比与实战案例

### 10.1 方法对比总结

```python
import pandas as pd

comparison_data = {
    '方法': ['朴素微调', '经验回放', 'EWC', 'SI', '渐进式网络', '生成回放'],
    '遗忘程度': ['严重', '轻微', '中等', '中等', '无', '轻微'],
    '存储需求': ['低', '中', '低', '低', '高', '中'],
    '计算开销': ['低', '低', '中', '低', '中', '高'],
    '实现复杂度': ['简单', '简单', '中等', '中等', '中等', '复杂'],
    '前向迁移': ['无', '有限', '有限', '有限', '有', '有限'],
    '数据隐私': ['好', '差', '好', '好', '好', '好']
}

df = pd.DataFrame(comparison_data)
print("\n持续学习方法对比")
print("=" * 80)
print(df.to_string(index=False))
```

### 10.2 实战：完整持续学习系统

```python
class ContinualLearningBenchmark:
    """持续学习综合评估"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}
        
    def run_all_methods(self, num_tasks=5, epochs=3):
        """运行所有方法"""
        methods = {
            'Naive': self.run_naive,
            'Experience Replay': self.run_experience_replay,
            'EWC': self.run_ewc,
            'SI': self.run_si,
            'Progressive NN': self.run_progressive,
        }
        
        for method_name, method_fn in methods.items():
            print(f"\n{'='*60}")
            print(f"运行方法: {method_name}")
            print(f"{'='*60}")
            self.results[method_name] = method_fn(num_tasks, epochs)
        
        return self.results
    
    def run_naive(self, num_tasks, epochs):
        """朴素微调（基线）"""
        model = SimpleNet().to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        results = {i: [] for i in range(num_tasks)}
        
        for task_id in range(num_tasks):
            train_loader, _, _ = get_task_data(task_id)
            
            # 训练
            model.train()
            for epoch in range(epochs):
                for data, target in train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            # 评估所有任务
            for eval_task_id in range(task_id + 1):
                _, test_loader, _ = get_task_data(eval_task_id)
                acc = evaluate_task(model, test_loader, self.device)
                results[eval_task_id].append(acc)
        
        return results
    
    def run_experience_replay(self, num_tasks, epochs):
        """经验回放"""
        model = SimpleNet().to(self.device)
        memory = ExperienceReplay(capacity=500)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        results = {i: [] for i in range(num_tasks)}
        
        for task_id in range(num_tasks):
            train_loader, _, _ = get_task_data(task_id)
            
            model.train()
            for epoch in range(epochs):
                for data, target in train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # 混合回放数据
                    if len(memory) > 0:
                        replay_data, replay_target = memory.sample(min(32, len(memory)))
                        replay_data = replay_data.to(self.device)
                        replay_target = replay_target.to(self.device)
                        data = torch.cat([data, replay_data])
                        target = torch.cat([target, replay_target])
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            # 保存到记忆
            for data, target in train_loader:
                memory.add(data, target)
            
            # 评估
            for eval_task_id in range(task_id + 1):
                _, test_loader, _ = get_task_data(eval_task_id)
                acc = evaluate_task(model, test_loader, self.device)
                results[eval_task_id].append(acc)
        
        return results
    
    def run_ewc(self, num_tasks, epochs):
        """EWC"""
        model = SimpleNet().to(self.device)
        ewc = EWC(model, self.device, lambda_ewc=10000)
        
        results = {i: [] for i in range(num_tasks)}
        
        for task_id in range(num_tasks):
            train_loader, _, _ = get_task_data(task_id)
            ewc.train_task(train_loader, epochs)
            
            for eval_task_id in range(task_id + 1):
                _, test_loader, _ = get_task_data(eval_task_id)
                acc = evaluate_task(ewc.model, test_loader, self.device)
                results[eval_task_id].append(acc)
        
        return results
    
    def run_si(self, num_tasks, epochs):
        """SI"""
        model = SimpleNet().to(self.device)
        si = SynapticIntelligence(model, self.device, lambda_si=0.1)
        
        results = {i: [] for i in range(num_tasks)}
        
        for task_id in range(num_tasks):
            train_loader, _, _ = get_task_data(task_id)
            si.train_task(train_loader, epochs)
            
            for eval_task_id in range(task_id + 1):
                _, test_loader, _ = get_task_data(eval_task_id)
                acc = evaluate_task(si.model, test_loader, self.device)
                results[eval_task_id].append(acc)
        
        return results
    
    def run_progressive(self, num_tasks, epochs):
        """渐进式网络"""
        pnn = ProgressiveNNTrainer(self.device)
        
        results = {i: [] for i in range(num_tasks)}
        
        for task_id in range(num_tasks):
            train_loader, _, classes = get_task_data(task_id)
            pnn.train_new_task(train_loader, classes, epochs)
            
            for eval_task_id in range(task_id + 1):
                _, test_loader, _ = get_task_data(eval_task_id)
                acc = pnn.evaluate_task(eval_task_id, test_loader)
                results[eval_task_id].append(acc)
        
        return results
    
    def plot_comparison(self):
        """绘制对比图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, (method_name, results) in enumerate(self.results.items()):
            ax = axes[idx]
            
            for task_id, accs in results.items():
                # 补齐数据
                padded_accs = [0] * task_id + accs
                ax.plot(range(len(padded_accs)), padded_accs, 
                       marker='o', label=f'Task {task_id}', linewidth=2)
            
            ax.set_title(method_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Task')
            ax.set_ylabel('Accuracy (%)')
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        plt.suptitle('Continual Learning Methods Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('continual_learning_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\n对比图已保存为 'continual_learning_comparison.png'")

# 运行综合评估
print("\n" + "=" * 60)
print("开始综合评估")
print("=" * 60)

benchmark = ContinualLearningBenchmark(device=device)
all_results = benchmark.run_all_methods(num_tasks=5, epochs=3)
benchmark.plot_comparison()

# 计算平均准确率
print("\n" + "=" * 60)
print("最终平均准确率（所有任务）")
print("=" * 60)

for method_name, results in all_results.items():
    # 计算每个任务的最终准确率
    final_accs = []
    for task_id, accs in results.items():
        if accs:
            final_accs.append(accs[-1])
    avg_acc = np.mean(final_accs) if final_accs else 0
    print(f"{method_name:20s}: {avg_acc:.2f}%")
```

---

## 十一、本章总结

### 11.1 核心知识点回顾

1. **灾难性遗忘**：神经网络学习新任务时遗忘旧知识的问题
2. **三大策略**：
   - **回放方法**：经验回放、生成回放
   - **正则化方法**：EWC、SI
   - **架构方法**：渐进式神经网络

3. **方法选择指南**：
   - **存储有限** → EWC、SI
   - **不能存储数据** → 生成回放、EWC
   - **要求零遗忘** → 渐进式网络
   - **简单有效** → 经验回放

### 11.2 费曼法比喻汇总

| 概念 | 比喻 |
|------|------|
| 灾难性遗忘 | 学新语言忘了旧语言 |
| 经验回放 | 复习旧笔记 |
| EWC | 重要知识用强力胶固定 |
| SI | 记住哪些肌肉用得最多 |
| 渐进式网络 | 新建抽屉而不是填满旧抽屉 |
| 生成回放 | 用想象力复习 |

### 11.3 持续学习的未来

1. **与大模型结合**：如何持续更新数十亿参数的模型
2. **生物启发**：从人类大脑学习更多机制
3. **理论与算法**：更强的理论保证
4. **实际应用**：终身学习机器人、个性化推荐等

---

## 练习题

1. **概念理解**：为什么神经网络会发生灾难性遗忘？人类大脑如何避免？

2. **数学推导**：推导EWC的损失函数，解释Fisher信息矩阵的作用。

3. **代码实现**：实现一个简化版的知识蒸馏（Learning without Forgetting）方法。

4. **实验分析**：在Split CIFAR-10上比较三种不同方法的遗忘曲线。

5. **开放问题**：设计一个新的持续学习方法，结合本章两种不同策略的优点。

---

## 参考文献

1. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521-3526.

2. Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic intelligence. In *International conference on machine learning* (pp. 3987-3995). PMLR.

3. Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K., ... & Hadsell, R. (2016). Progressive neural networks. *arXiv preprint arXiv:1606.04671*.

4. Shin, H., Lee, J. K., Kim, J., & Kim, J. (2017). Continual learning with deep generative replay. *Advances in neural information processing systems*, 30.

5. McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. In *Psychology of learning and motivation* (Vol. 24, pp. 109-165). Academic Press.

6. Li, Z., & Hoiem, D. (2017). Learning without forgetting. *IEEE transactions on pattern analysis and machine intelligence*, 40(12), 2935-2947.

7. Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. *Advances in neural information processing systems*, 30.

8. Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). Experience replay for continual learning. *Advances in Neural Information Processing Systems*, 32.

9. Rebuffi, S. A., Kolesnikov, A., Sperl, G., & Lampert, C. H. (2017). iCaRL: Incremental classifier and representation learning. In *Proceedings of the IEEE conference on Computer Vision and Pattern Recognition* (pp. 2001-2010).

10. Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. *Neural networks*, 113, 54-71.

---

*第五十五章完。下一章将介绍神经架构搜索——让AI自己设计神经网络。*
