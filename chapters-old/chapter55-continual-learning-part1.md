# 第五十五章 持续学习与大脑可塑性——AI如何终身成长

> *"教育就是当一个人把在学校所学全部忘光之后剩下的东西。"*
> *—— 阿尔伯特·爱因斯坦*

## 一、开篇故事：会遗忘的AI

想象一下，你花了三个月时间教一只聪明的小狗学会了握手。然后你开始教它坐下——三天后，你兴奋地伸出手说"握手"，小狗却茫然地看着你，完全忘了这个指令。它只记得"坐下"。

这听起来很荒谬，对吧？但这就是**大多数AI系统的真实写照**。

小明是一个AI研究员。他训练了一个神经网络来识别猫。模型表现得很好——准确率99%！然后他想让同一个模型学会识别狗。他收集了狗狗的图片，开始训练。

几个小时后，模型学会了识别狗。但当他再次测试猫的识别时——灾难发生了。模型的准确率从99%暴跌到了30%。

"怎么会这样？"小明崩溃了，"猫和狗是完全不同的东西啊！"

这就是人工智能领域最著名的难题之一：**灾难性遗忘（Catastrophic Forgetting）**。

### 本章核心问题

1. **为什么AI会像金鱼一样"健忘"？**
2. **人类大脑如何避免这个问题？**
3. **我们能让AI像人类一样终身学习吗？**

准备好了吗？让我们开始这场关于"AI记忆"的探索之旅！

---

## 二、什么是灾难性遗忘？

### 2.1 一个直观的演示

让我们用一个简单的例子来理解这个问题。假设你有一个神经网络，它需要学会两个任务：

**任务A**：识别手写数字0和1  
**任务B**：识别手写数字2和3

首先，你训练网络学会任务A。它表现得很好——能够准确区分0和1。然后，你用任务B的数据继续训练同一个网络。当你测试任务B时，它也表现得很好。

但是，当你再次测试任务A时——**它几乎完全忘记了如何区分0和1**！

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子确保可重复
torch.manual_seed(42)
np.random.seed(42)

# 定义简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_task_data(task_id, batch_size=64):
    """获取特定任务的数据（每个任务包含两个数字）"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # 定义任务：每个任务包含两个数字
    task_classes = {
        0: [0, 1],  # 任务A: 0和1
        1: [2, 3],  # 任务B: 2和3
        2: [4, 5],  # 任务C: 4和5
        3: [6, 7],  # 任务D: 6和7
        4: [8, 9]   # 任务E: 8和9
    }
    
    classes = task_classes[task_id]
    
    # 筛选训练数据
    train_mask = torch.tensor([label in classes for label in train_dataset.targets])
    train_indices = torch.where(train_mask)[0]
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    
    # 筛选测试数据
    test_mask = torch.tensor([label in classes for label in test_dataset.targets])
    test_indices = torch.where(test_mask)[0]
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, classes

def train_task(model, train_loader, epochs=5, device='cpu'):
    """训练模型"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        acc = 100. * correct / total
        print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%")
    
    return model

def evaluate_task(model, test_loader, device='cpu'):
    """评估模型"""
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

# ========================================
# 演示灾难性遗忘
# ========================================

print("=" * 60)
print("灾难性遗忘演示")
print("=" * 60)

# 创建模型
model = SimpleNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}\n")

# 存储每个任务在各阶段的准确率
results = {i: [] for i in range(5)}

# 顺序学习5个任务
for task_id in range(5):
    print(f"\n{'='*40}")
    print(f"开始训练任务 {task_id} (数字: {[[0,1],[2,3],[4,5],[6,7],[8,9]][task_id]})")
    print(f"{'='*40}")
    
    # 获取当前任务数据
    train_loader, test_loader, classes = get_task_data(task_id)
    
    # 训练当前任务
    model = train_task(model, train_loader, epochs=3, device=device)
    
    # 评估所有已学习的任务
    print(f"\n训练任务 {task_id} 后，各任务准确率:")
    print("-" * 40)
    
    for eval_task_id in range(task_id + 1):
        _, test_loader, _ = get_task_data(eval_task_id)
        acc = evaluate_task(model, test_loader, device=device)
        results[eval_task_id].append(acc)
        print(f"  任务 {eval_task_id}: {acc:.2f}%")

print("\n" + "=" * 60)
print("最终结果 - 灾难性遗忘可视化")
print("=" * 60)

# 绘制遗忘曲线
fig, ax = plt.subplots(figsize=(12, 6))

task_names = ['0&1', '2&3', '4&5', '6&7', '8&9']
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

for task_id in range(5):
    # 补齐数据（任务还没学习时的准确率为0）
    y_values = [0] * task_id + results[task_id]
    x_values = list(range(5))
    
    ax.plot(x_values, y_values, marker='o', linewidth=2, 
            label=f'任务 {task_id} ({task_names[task_id]})', 
            color=colors[task_id], markersize=8)

ax.set_xlabel('训练阶段 (任务ID)', fontsize=12)
ax.set_ylabel('准确率 (%)', fontsize=12)
ax.set_title('灾难性遗忘：学习新任务时旧任务性能迅速下降', fontsize=14, fontweight='bold')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('catastrophic_forgetting_demo.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n图表已保存为 'catastrophic_forgetting_demo.png'")
```

运行这段代码，你会看到一条触目惊心的曲线：

- **任务A**在训练后达到98%准确率
- 学完任务B后，任务A的准确率跌到20%以下
- 学完任务C后，任务A和B都几乎完全遗忘
- 最终，模型只记得最后一个任务

这就是**灾难性遗忘**。

### 2.2 为什么会发生灾难性遗忘？

#### 费曼解释：图书馆的困境

想象一个图书馆只有100个书架位置（神经网络的参数）。

**第一天**：图书馆收到一批关于"猫"的书籍。图书管理员把书放在书架1-50号。

**第二天**：图书馆收到一批关于"狗"的书籍。图书管理员发现书架满了！于是他把"猫"的书扔掉，把"狗"的书放在书架1-50号。

**结果**：关于"猫"的知识全部丢失！

这就是神经网络的问题：**所有任务共享同一组参数**。当你用新数据训练时，梯度更新会覆盖旧的权重值，就像图书管理员不断用新书替换旧书。

#### 数学解释

神经网络的学习过程是：**最小化损失函数**。

对于任务A，我们优化：
$$\theta_A^* = \arg\min_\theta \mathcal{L}_A(\theta)$$

对于任务B，我们继续优化：
$$\theta_B^* = \arg\min_\theta \mathcal{L}_B(\theta)$$

问题是：**优化任务B时，我们没有约束$\theta$必须保持在$\theta_A^*$附近**。因此，梯度下降会把参数推向任务B的最优解，而不管这对任务A的影响。

```python
# 可视化参数空间中的灾难性遗忘
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.animation as animation

# 创建参数空间的简化可视化（2D投影）
fig, ax = plt.subplots(figsize=(10, 8))

# 定义两个任务的最优点
task_a_opt = np.array([2, 3])
task_b_opt = np.array([7, 6])

# 绘制损失等高线（椭圆）
def plot_loss_contour(center, color, label, alpha=0.3):
    for i in range(1, 5):
        ellipse = Ellipse(center, width=i*1.5, height=i*1.2, 
                         angle=np.random.uniform(-30, 30),
                         facecolor='none', edgecolor=color, 
                         linewidth=2, alpha=alpha)
        ax.add_patch(ellipse)
    ax.plot(center[0], center[1], 'o', color=color, markersize=15, 
            label=label, zorder=5)

plot_loss_contour(task_a_opt, '#3498db', '任务A最优解')
plot_loss_contour(task_b_opt, '#e74c3c', '任务B最优解')

# 绘制学习轨迹
theta_trajectory = [
    [0, 0],      # 初始点
    [1.5, 2.5],  # 学习任务A中
    [2, 3],      # 任务A最优点
    [4, 4.5],    # 学习任务B中（忘记A）
    [7, 6],      # 任务B最优点
]

trajectory = np.array(theta_trajectory)
ax.plot(trajectory[:, 0], trajectory[:, 1], 'g--', linewidth=2, 
        label='参数轨迹', marker='o', markersize=8)

# 标注关键点
ax.annotate('起点', theta_trajectory[0], xytext=(-20, -20), 
            textcoords='offset points', fontsize=10, 
            arrowprops=dict(arrowstyle='->', color='black'))
ax.annotate('学会任务A', theta_trajectory[2], xytext=(10, 20), 
            textcoords='offset points', fontsize=10,
            arrowprops=dict(arrowstyle='->', color='#3498db'))
ax.annotate('忘记任务A\n学会任务B', theta_trajectory[4], xytext=(20, -20), 
            textcoords='offset points', fontsize=10,
            arrowprops=dict(arrowstyle='->', color='#e74c3c'))

# 绘制"遗忘箭头"
ax.annotate('', xy=theta_trajectory[4], xytext=theta_trajectory[2],
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=3, ls='--'))
ax.text(4.5, 4, '遗忘！', fontsize=14, color='#e74c3c', fontweight='bold')

ax.set_xlabel('参数 θ₁', fontsize=12)
ax.set_ylabel('参数 θ₂', fontsize=12)
ax.set_title('参数空间视角的灾难性遗忘', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 9)
ax.set_ylim(-1, 8)

plt.tight_layout()
plt.savefig('parameter_space_forgetting.png', dpi=150, bbox_inches='tight')
plt.show()

print("参数空间可视化已保存")
```

### 2.3 人类为什么不会灾难性遗忘？

人类大脑有三个关键机制：

1. **双系统记忆**：
   - **海马体**：快速学习新信息（短期记忆）
   - **大脑皮层**：缓慢巩固长期记忆
   - 睡眠时，海马体将记忆"回放"给皮层

2. **分布式表示**：
   - 记忆分散存储在大规模神经网络中
   - 新学习不会完全覆盖旧记忆

3. **结构可塑性**：
   - 大脑可以生长新的神经连接
   - 而不是仅仅修改现有连接的权重

这些机制启发了AI研究者们开发各种**持续学习（Continual Learning）**算法。

---

## 三、持续学习的三大策略

为了克服灾难性遗忘，研究者们提出了三大类方法：

### 策略一：回放方法（Replay Methods）
**核心思想**：像复习一样，重播旧数据
- 经验回放（Experience Replay）
- 生成回放（Generative Replay）

### 策略二：正则化方法（Regularization Methods）
**核心思想**：像保护重要文物一样，保护重要参数
- 弹性权重巩固（EWC）
- 突触智能（SI）

### 策略三：架构方法（Architectural Methods）
**核心思想**：像扩展图书馆一样，增加新空间
- 渐进式神经网络
- 参数隔离方法

让我们逐一深入这些策略！

---

## 四、经验回放：大脑的记忆重播

### 4.1 大脑的启示：互补学习系统

神经科学家发现，人类大脑有两个互补的学习系统：

1. **海马体（Hippocampus）**：
   - 快速学习新信息
   - 容量有限
   - 类似"临时硬盘"

2. **大脑皮层（Cortex）**：
   - 缓慢学习，容量巨大
   - 存储长期记忆
   - 类似"长期档案库"

**关键机制**：当我们睡觉时，海马体会"重播"白天的经历，帮助皮层巩固记忆。

### 4.2 经验回放的原理

AI中的经验回放模拟了这一机制：

```
┌─────────────────────────────────────────────────────────────┐
│                    经验回放的流程                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   任务A数据 → [神经网络] → 存储到记忆缓冲区                   │
│                              ↓                              │
│   任务B数据 → [神经网络] ← 随机抽取旧数据                    │
│                    ↑                                        │
│              新旧数据混合训练                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**核心算法**：

```python
import torch
import random
from collections import deque

class ExperienceReplay:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=2000):
        """
        Args:
            capacity: 缓冲区容量（最多存储多少样本）
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)  # 自动淘汰旧数据
        
    def add(self, data, target):
        """添加样本到缓冲区"""
        # 将数据移到CPU以节省GPU内存
        data = data.cpu()
        target = target.cpu()
        
        # 逐个添加样本（而非整个batch）
        for i in range(len(target)):
            self.buffer.append((data[i], target[i]))
    
    def sample(self, batch_size):
        """随机采样"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        samples = random.sample(self.buffer, batch_size)
        
        # 重新组合成batch
        data = torch.stack([s[0] for s in samples])
        targets = torch.stack([s[1] for s in samples])
        
        return data, targets
    
    def __len__(self):
        return len(self.buffer)

class ReplayTrainer:
    """带经验回放的训练器"""
    
    def __init__(self, model, device='cpu', memory_size=2000, replay_ratio=0.3):
        """
        Args:
            model: 神经网络模型
            device: 计算设备
            memory_size: 记忆缓冲区大小
            replay_ratio: 每个batch中旧数据的比例
        """
        self.model = model.to(device)
        self.device = device
        self.memory = ExperienceReplay(capacity=memory_size)
        self.replay_ratio = replay_ratio
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
    def train_task(self, train_loader, epochs=5):
        """训练一个新任务"""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 如果有记忆数据，混合训练
                if len(self.memory) > 0:
                    # 计算回放的样本数
                    replay_size = int(len(data) * self.replay_ratio)
                    
                    # 从记忆中采样
                    replay_data, replay_target = self.memory.sample(replay_size)
                    replay_data = replay_data.to(self.device)
                    replay_target = replay_target.to(self.device)
                    
                    # 合并新旧数据
                    combined_data = torch.cat([data, replay_data], dim=0)
                    combined_target = torch.cat([target, replay_target], dim=0)
                else:
                    combined_data = data
                    combined_target = target
                
                # 训练
                self.optimizer.zero_grad()
                output = self.model(combined_data)
                loss = self.criterion(output, combined_target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += combined_target.size(0)
                correct += predicted.eq(combined_target).sum().item()
            
            # 保存当前任务的数据到记忆
            for data, target in train_loader:
                self.memory.add(data, target)
            
            acc = 100. * correct / total
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%")
        
        return self.model

# 测试经验回放
def test_experience_replay():
    """测试经验回放效果"""
    print("\n" + "=" * 60)
    print("经验回放演示")
    print("=" * 60)
    
    model = SimpleNet()
    trainer = ReplayTrainer(model, device=device, memory_size=500, replay_ratio=0.3)
    
    # 存储结果
    results_with_replay = {i: [] for i in range(5)}
    
    for task_id in range(5):
        print(f"\n{'='*40}")
        print(f"训练任务 {task_id}")
        print(f"{'='*40}")
        
        train_loader, _, _ = get_task_data(task_id)
        trainer.train_task(train_loader, epochs=3)
        
        # 评估所有任务
        print(f"\n各任务准确率:")
        for eval_task_id in range(task_id + 1):
            _, test_loader, _ = get_task_data(eval_task_id)
            acc = evaluate_task(trainer.model, test_loader, device=device)
            results_with_replay[eval_task_id].append(acc)
            print(f"  任务 {eval_task_id}: {acc:.2f}%")
    
    return results_with_replay

# 运行测试
results_replay = test_experience_replay()
```

### 4.3 不同的采样策略

**Reservoir Sampling（水库采样）**：
当缓冲区已满时，新样本以概率 N/M 替换旧样本（N是缓冲区大小，M是已见样本总数）。这保证了每个样本被选中的概率相等。

```python
class ReservoirSamplingReplay:
    """水库采样经验回放"""
    
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.buffer = []
        self.total_seen = 0  # 总共见过的样本数
        
    def add(self, data, target):
        """使用水库采样添加样本"""
        data = data.cpu()
        target = target.cpu()
        
        for i in range(len(target)):
            self.total_seen += 1
            sample = (data[i], target[i])
            
            if len(self.buffer) < self.capacity:
                # 缓冲区未满，直接添加
                self.buffer.append(sample)
            else:
                # 水库采样：以 capacity/total_seen 的概率替换
                idx = random.randint(0, self.total_seen - 1)
                if idx < self.capacity:
                    self.buffer[idx] = sample
    
    def sample(self, batch_size):
        """随机采样"""
        samples = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        data = torch.stack([s[0] for s in samples])
        targets = torch.stack([s[1] for s in samples])
        return data, targets
    
    def __len__(self):
        return len(self.buffer)
```

### 4.4 经验回放的优缺点

**优点**：
- ✅ 简单易实现
- ✅ 效果通常很好
- ✅ 与模型无关

**缺点**：
- ❌ 需要存储原始数据（隐私问题）
- ❌ 内存需求随任务增加
- ❌ 对于大量任务，缓冲区可能不够

**费曼比喻**：经验回放就像学生在考试前复习笔记——把旧知识拿出来再看一遍，就不会忘记了。

---

## 五、弹性权重巩固（EWC）：保护重要参数

### 5.1 贝叶斯视角的持续学习

EWC（Elastic Weight Consolidation）是由DeepMind的Kirkpatrick等人在2017年提出的，它从**贝叶斯角度**看待持续学习。

**核心思想**：
- 不是所有参数都同等重要
- 有些参数对旧任务至关重要，应该"锁定"
- 其他参数可以自由调整

这就像保护历史建筑——你不能拆掉承重墙，但可以装修其他部分。

### 5.2 数学推导

#### 贝叶斯定理回顾

我们想要在给定所有任务数据的情况下，找到最优参数：

$$p(\theta | D_A, D_B) = \frac{p(D_B | \theta) p(\theta | D_A)}{p(D_B)}$$

其中：
- $p(\theta | D_A)$ 是学任务A后的参数分布（先验）
- $p(D_B | \theta)$ 是任务B的似然
- $p(\theta | D_A, D_B)$ 是联合后验

#### Laplace近似

假设 $p(\theta | D_A)$ 可以用一个高斯分布近似：

$$p(\theta | D_A) \approx \mathcal{N}(\theta_A^*, F^{-1})$$

其中：
- $\theta_A^*$ 是任务A的最优参数
- $F$ 是**Fisher信息矩阵**

#### Fisher信息矩阵

Fisher信息矩阵衡量了参数对模型输出的敏感度：

$$F_{ij} = \mathbb{E}_{x \sim D_A} \left[ \frac{\partial \log p(x|\theta)}{\partial \theta_i} \frac{\partial \log p(x|\theta)}{\partial \theta_j} \right]$$

对角线元素 $F_{ii}$ 表示参数 $i$ 的重要性。

#### EWC损失函数

将上述近似代入，得到EWC的损失函数：

$$\mathcal{L}_{EWC}(\theta) = \mathcal{L}_B(\theta) + \frac{\lambda}{2} \sum_i F_{ii} (\theta_i - \theta_{A,i}^*)^2$$

其中：
- $\mathcal{L}_B(\theta)$ 是任务B的标准损失
- 第二项是**正则化项**，惩罚对重要参数的修改
- $\lambda$ 控制记忆强度

### 5.3 EWC代码实现

```python
class EWC:
    """弹性权重巩固 (Elastic Weight Consolidation)"""
    
    def __init__(self, model, device='cpu', lambda_ewc=10000):
        """
        Args:
            model: 神经网络
            device: 计算设备
            lambda_ewc: EWC正则化强度
        """
        self.model = model.to(device)
        self.device = device
        self.lambda_ewc = lambda_ewc
        
        # 存储每个任务的参数和Fisher信息
        self.params = {}      # {task_id: {param_name: param_value}}
        self.fisher = {}      # {task_id: {param_name: fisher_diag}}
        self.task_count = 0
        
    def compute_fisher(self, train_loader, num_samples=200):
        """
        计算Fisher信息矩阵的对角线（参数重要性）
        
        Args:
            train_loader: 训练数据
            num_samples: 用于估计Fisher的样本数
        """
        self.model.eval()
        fisher_diag = {}
        
        # 初始化Fisher矩阵
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_diag[name] = torch.zeros_like(param)
        
        # 收集样本
        samples_collected = 0
        for data, target in train_loader:
            if samples_collected >= num_samples:
                break
                
            data = data.to(self.device)
            
            self.model.zero_grad()
            output = self.model(data)
            
            # 使用预测类别作为目标（无标签估计）
            pred = output.max(1)[1]
            log_likelihood = F.log_softmax(output, dim=1)[range(len(pred)), pred]
            
            # 计算梯度平方
            for i in range(len(log_likelihood)):
                if samples_collected >= num_samples:
                    break
                    
                self.model.zero_grad()
                log_likelihood[i].backward(retain_graph=True)
                
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher_diag[name] += param.grad.data ** 2
                
                samples_collected += 1
        
        # 平均
        for name in fisher_diag:
            fisher_diag[name] /= samples_collected
            
        return fisher_diag
    
    def update_fisher_and_params(self, train_loader):
        """学习完一个任务后，保存参数和Fisher信息"""
        # 保存当前参数
        params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params[name] = param.data.clone()
        
        # 计算Fisher信息
        fisher = self.compute_fisher(train_loader)
        
        # 存储
        self.params[self.task_count] = params
        self.fisher[self.task_count] = fisher
        self.task_count += 1
        
        print(f"  已保存任务 {self.task_count-1} 的参数和Fisher信息")
    
    def penalty(self, model):
        """计算EWC惩罚项"""
        loss = 0
        
        for task_id in range(self.task_count):
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.params[task_id]:
                    # 计算参数变化
                    param_diff = param - self.params[task_id][name]
                    # 加权平方误差
                    loss += (self.fisher[task_id][name] * param_diff ** 2).sum()
        
        return loss
    
    def train_task(self, train_loader, epochs=5):
        """训练一个新任务"""
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            total_ce_loss = 0
            total_ewc_loss = 0
            correct = 0
            total = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # 分类损失
                output = self.model(data)
                ce_loss = criterion(output, target)
                
                # EWC惩罚
                ewc_loss = self.penalty(self.model) if self.task_count > 0 else 0
                
                # 总损失
                loss = ce_loss + (self.lambda_ewc / 2) * ewc_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                if isinstance(ewc_loss, torch.Tensor):
                    total_ewc_loss += ewc_loss.item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            acc = 100. * correct / total
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, "
                  f"CE={total_ce_loss/len(train_loader):.4f}, "
                  f"EWC={total_ewc_loss/len(train_loader):.6f}, Acc={acc:.2f}%")
        
        # 学习完任务后，保存参数和Fisher信息
        self.update_fisher_and_params(train_loader)
        
        return self.model

# 测试EWC
def test_ewc():
    """测试EWC效果"""
    print("\n" + "=" * 60)
    print("弹性权重巩固 (EWC) 演示")
    print("=" * 60)
    
    model = SimpleNet()
    ewc = EWC(model, device=device, lambda_ewc=10000)
    
    results_ewc = {i: [] for i in range(5)}
    
    for task_id in range(5):
        print(f"\n{'='*40}")
        print(f"训练任务 {task_id}")
        print(f"{'='*40}")
        
        train_loader, _, _ = get_task_data(task_id)
        ewc.train_task(train_loader, epochs=3)
        
        print(f"\n各任务准确率:")
        for eval_task_id in range(task_id + 1):
            _, test_loader, _ = get_task_data(eval_task_id)
            acc = evaluate_task(ewc.model, test_loader, device=device)
            results_ewc[eval_task_id].append(acc)
            print(f"  任务 {eval_task_id}: {acc:.2f}%")
    
    return results_ewc

# 运行EWC测试
results_ewc = test_ewc()
```

### 5.4 EWC可视化

```python
# 可视化EWC的参数保护效果
def visualize_ewc_protection():
    """可视化EWC如何保护重要参数"""
    
    # 训练EWC并收集数据
    model = SimpleNet()
    ewc = EWC(model, device=device, lambda_ewc=10000)
    
    # 训练任务A
    train_loader_a, _, _ = get_task_data(0)
    ewc.train_task(train_loader_a, epochs=5)
    
    # 获取任务A的参数和Fisher信息
    param_a = ewc.params[0]
    fisher_a = ewc.fisher[0]
    
    # 训练任务B（使用EWC）
    train_loader_b, _, _ = get_task_data(1)
    ewc.train_task(train_loader_b, epochs=5)
    
    # 获取训练后的参数
    param_after_b = {}
    for name, p in ewc.model.named_parameters():
        param_after_b[name] = p.data.clone()
    
    # 可视化第一个全连接层的变化
    layer_name = 'fc1.weight'
    param_a_flat = param_a[layer_name].cpu().numpy().flatten()
    param_after_flat = param_after_b[layer_name].cpu().numpy().flatten()
    fisher_flat = fisher_a[layer_name].cpu().numpy().flatten()
    
    # 计算参数变化
    param_change = np.abs(param_after_flat - param_a_flat)
    
    # 绘制散点图：Fisher重要性 vs 参数变化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：Fisher重要性分布
    ax1.hist(fisher_flat, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    ax1.set_xlabel('Fisher信息 (参数重要性)', fontsize=11)
    ax1.set_ylabel('参数数量', fontsize=11)
    ax1.set_title('任务A的参数重要性分布', fontsize=12, fontweight='bold')
    ax1.axvline(np.mean(fisher_flat), color='red', linestyle='--', 
                label=f'平均值: {np.mean(fisher_flat):.4f}')
    ax1.legend()
    
    # 右图：Fisher vs 参数变化
    scatter = ax2.scatter(fisher_flat, param_change, c=fisher_flat, 
                         cmap='YlOrRd', alpha=0.6, s=20)
    ax2.set_xlabel('Fisher信息 (重要性)', fontsize=11)
    ax2.set_ylabel('参数变化幅度', fontsize=11)
    ax2.set_title('EWC保护效果：重要参数变化小', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='重要性')
    
    # 添加趋势线
    z = np.polyfit(fisher_flat, param_change, 1)
    p = np.poly1d(z)
    ax2.plot(sorted(fisher_flat), p(sorted(fisher_flat)), 
             "r--", alpha=0.8, linewidth=2, label='趋势')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('ewc_protection_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nEWC保护可视化已保存")
    print(f"高Fisher参数的平均变化: {np.mean(param_change[fisher_flat > np.percentile(fisher_flat, 90)]):.6f}")
    print(f"低Fisher参数的平均变化: {np.mean(param_change[fisher_flat < np.percentile(fisher_flat, 10)]):.6f}")

visualize_ewc_protection()
```

### 5.5 EWC的优缺点

**优点**：
- ✅ 不需要存储原始数据
- ✅ 有理论支撑（贝叶斯视角）
- ✅ 与模型无关

**缺点**：
- ❌ Fisher矩阵计算开销大
- ❌ 对角近似可能不够精确
- ❌ 多个任务时惩罚项累积

**费曼比喻**：EWC就像给重要的书贴上"请勿移动"的标签——图书管理员知道哪些书不能动，就可以安全地整理其他区域。

---

## 六、突触智能（SI）：在线估计重要性

### 6.1 SI的核心思想

SI（Synaptic Intelligence）由Zenke等人于2017年提出。它与EWC类似，但有一个关键区别：

- **EWC**：训练后计算Fisher信息（离线）
- **SI**：训练过程中在线估计参数重要性

这就像：
- EWC：考试后分析哪些知识点重要
- SI：学习过程中记录每个知识点的使用频率

### 6.2 数学原理

SI通过**轨迹积分**估计参数重要性：

$$\Omega_i = \int_0^{t_f} |g_i(t) \cdot \dot{\theta}_i(t)| dt$$

其中：
- $g_i(t) = \frac{\partial \mathcal{L}}{\partial \theta_i}$ 是参数$\theta_i$的梯度
- $\dot{\theta}_i(t)$ 是参数的变化速度
- 积分遍历整个训练过程

**直观理解**：
- 如果一个参数对降低损失贡献很大（梯度大）
- 并且这个参数变化了很多（移动距离大）
- 那么这个参数对任务很重要

### 6.3 SI代码实现

```python
class SynapticIntelligence:
    """突触智能 (Synaptic Intelligence)"""
    
    def __init__(self, model, device='cpu', lambda_si=0.1, xi=0.1):
        """
        Args:
            model: 神经网络
            device: 计算设备
            lambda_si: SI正则化强度
            xi: 数值稳定性常数
        """
        self.model = model.to(device)
        self.device = device
        self.lambda_si = lambda_si
        self.xi = xi
        
        # 初始化
        self.params = {}      # 每个任务的参数
        self.omega = {}       # 每个任务的参数重要性
        self.task_count = 0
        
        # 在线追踪的变量
        self.prev_params = {}    # 上一步的参数值
        self.contribution = {}   # 参数贡献累积
        
        self._initialize_tracking()
    
    def _initialize_tracking(self):
        """初始化追踪变量"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.prev_params[name] = param.data.clone()
                self.contribution[name] = torch.zeros_like(param)
    
    def update_importance(self):
        """更新参数重要性估计（在每个优化步骤后调用）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # 计算参数变化
                delta_theta = param.data - self.prev_params[name]
                
                # 计算贡献：梯度 × 参数变化
                # 使用绝对值确保贡献为正
                contribution = torch.abs(param.grad.data * delta_theta)
                
                # 累积贡献
                self.contribution[name] += contribution
                
                # 更新前一步参数
                self.prev_params[name] = param.data.clone()
    
    def consolidate(self):
        """任务结束时，保存参数和重要性"""
        # 保存参数
        params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params[name] = param.data.clone()
        
        # 归一化重要性（除以参数变化的平方 + xi）
        omega = {}
        for name in self.contribution:
            delta_theta_sq = (params[name] - self.params.get(self.task_count-1, {}).get(name, params[name])) ** 2
            omega[name] = self.contribution[name] / (delta_theta_sq + self.xi)
        
        self.params[self.task_count] = params
        self.omega[self.task_count] = omega
        self.task_count += 1
        
        # 重置追踪
        self._initialize_tracking()
        
        print(f"  已保存任务 {self.task_count-1} 的参数和SI重要性")
    
    def penalty(self, model):
        """计算SI惩罚项"""
        loss = 0
        
        for task_id in range(self.task_count):
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.params[task_id]:
                    # 参数变化
                    param_diff = param - self.params[task_id][name]
                    # 加权惩罚
                    loss += (self.omega[task_id][name] * param_diff ** 2).sum()
        
        return loss
    
    def train_task(self, train_loader, epochs=5):
        """训练一个新任务"""
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            total_ce_loss = 0
            total_si_loss = 0
            correct = 0
            total = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # 分类损失
                output = self.model(data)
                ce_loss = criterion(output, target)
                
                # SI惩罚
                si_loss = self.penalty(self.model) if self.task_count > 0 else 0
                
                # 总损失
                loss = ce_loss + self.lambda_si * si_loss
                
                loss.backward()
                optimizer.step()
                
                # 更新重要性估计
                self.update_importance()
                
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                if isinstance(si_loss, torch.Tensor):
                    total_si_loss += si_loss.item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            acc = 100. * correct / total
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, "
                  f"CE={total_ce_loss/len(train_loader):.4f}, "
                  f"SI={total_si_loss/len(train_loader):.6f}, Acc={acc:.2f}%")
        
        # 任务结束，巩固记忆
        self.consolidate()
        
        return self.model

# 测试SI
def test_si():
    """测试SI效果"""
    print("\n" + "=" * 60)
    print("突触智能 (SI) 演示")
    print("=" * 60)
    
    model = SimpleNet()
    si = SynapticIntelligence(model, device=device, lambda_si=0.1)
    
    results_si = {i: [] for i in range(5)}
    
    for task_id in range(5):
        print(f"\n{'='*40}")
        print(f"训练任务 {task_id}")
        print(f"{'='*40}")
        
        train_loader, _, _ = get_task_data(task_id)
        si.train_task(train_loader, epochs=3)
        
        print(f"\n各任务准确率:")
        for eval_task_id in range(task_id + 1):
            _, test_loader, _ = get_task_data(eval_task_id)
            acc = evaluate_task(si.model, test_loader, device=device)
            results_si[eval_task_id].append(acc)
            print(f"  任务 {eval_task_id}: {acc:.2f}%")
    
    return results_si

# 运行SI测试
results_si = test_si()
```

### 6.4 SI vs EWC

| 特性 | EWC | SI |
|------|-----|-----|
| 重要性计算 | 训练后离线计算 | 训练中在线估计 |
| 计算开销 | 需要额外前向传播 | 几乎无额外开销 |
| 理论基础 | Fisher信息 | 轨迹积分 |
| 内存需求 | 存储Fisher矩阵 | 存储贡献累积 |

**费曼比喻**：
- EWC像期末考试后复盘，看哪些知识点最重要
- SI像平时做笔记，记录每个知识点的练习次数

---

（因篇幅限制，剩余部分包括渐进式神经网络、生成回放、大模型持续适配和实战案例将在后续内容中完成。本章已完成约10,000字，涵盖核心概念、EWC和SI的完整实现。）

## 本章小结

### 核心概念
1. **灾难性遗忘**：神经网络学习新任务时遗忘旧知识的问题
2. **经验回放**：存储和重播旧数据防止遗忘
3. **EWC**：使用Fisher信息保护重要参数
4. **SI**：在线估计参数重要性

### 关键公式
- EWC损失: $\mathcal{L} = \mathcal{L}_{new} + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$
- SI重要性: $\Omega_i = \int |g_i \cdot \dot{\theta}_i| dt$

### 下一步
接下来将继续介绍渐进式神经网络、生成回放方法，以及大语言模型的持续学习技术。

---

*本章为第五十五章第一部分，涵盖持续学习的核心概念和正则化方法。完整章节预计16,000字。*
