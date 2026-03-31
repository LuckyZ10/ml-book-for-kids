# 第三十七章：神经架构搜索 (Neural Architecture Search, NAS)

> **"让机器自动发现比人类设计更好的神经网络"** - NAS的雄心壮志

## 章节引言

想象你是一位建筑师，面前摆着一张白纸，任务是设计一座既美观又实用的大楼。你会考虑无数种可能：房间如何布局？窗户多大？用什么材料？传统上，这需要数十年的经验积累和反复试错。

现在，想象有一个智能助手，它可以**自动尝试数百万种设计方案**，从中找出最优的那一个——不仅考虑美观，还考虑成本、结构稳定性、环保指标。这听起来像科幻，但在机器学习领域，这正是**神经架构搜索(Neural Architecture Search, NAS)**正在做的事情！

**费曼法理解：自动化建筑师**

想象神经网络是一座大楼：
- **人工设计网络**（如ResNet、VGG）就像请一位著名建筑师手工设计——质量高，但需要专业知识和灵感
- **神经架构搜索**就像让一支机器人建筑团队自动尝试各种设计组合——不停试验、评估、改进，直到找到最佳方案

NAS的目标很简单却野心勃勃：**让算法自动发现比人类设计更好的神经网络架构**。

在本章中，我们将：
- 🏗️ 理解NAS的三大核心组件：搜索空间、搜索策略、性能评估
- 🤖 探索强化学习、进化算法、梯度优化等搜索方法
- 🚀 深入DARTS、ENAS、ProxylessNAS等经典算法
- ⚡ 学习硬件感知NAS：如何让模型在手机上跑得又快又好
- 🎯 掌握权重共享和超网训练技术

准备好了吗？让我们开始这场自动化设计神经网络的奇妙之旅！

---

## 37.1 什么是神经架构搜索？

### 37.1.1 NAS的诞生与动机

2017年，Google Brain的研究团队提出了一个大胆的问题：**既然神经网络可以学习识别图像、翻译语言，为什么不能学习设计自己呢？**

这个问题催生了神经架构搜索领域。Zoph和Le在ICLR 2017发表的论文《Neural Architecture Search with Reinforcement Learning》中，首次展示了用强化学习自动发现神经网络架构的可能性。

**为什么要用NAS？**

1. **人工设计的局限性**：人类专家设计的网络（如ResNet、DenseNet）虽然成功，但可能不是最优的
2. **领域迁移困难**：为ImageNet设计的网络不一定适合医疗影像或卫星图像
3. **效率需求**：移动设备、IoT设备需要极致高效的模型
4. **计算资源爆炸**：手动尝试所有可能的架构组合是不可能的

**NAS的魔力**：
- NASNet在CIFAR-10上发现的架构，迁移到ImageNet后超越了所有人工设计的网络
- 搜索成本从最初的几千GPU天降到了现在的几分钟
- 发现的架构在准确率-效率权衡上全面超越人类设计

### 37.1.2 NAS的三大核心组件

NAS可以形式化为一个优化问题，包含三个关键组件：

$$\alpha^* = \underset{\alpha \in \mathcal{A}}{\arg\max} \, \text{Accuracy}(\mathcal{N}(\alpha, w^*))$$

其中 $w^* = \arg\min_w \mathcal{L}_{train}(\mathcal{N}(\alpha, w))$

#### 1. 搜索空间 ($\mathcal{A}$)

**生活化比喻：乐高积木库**

想象你有一盒乐高积木：
- 不同颜色和形状的积木 = 各种神经网络操作（卷积、池化、跳跃连接）
- 积木的组合规则 = 搜索空间的约束条件
- 能搭出的所有可能模型 = 搜索空间

搜索空间定义了哪些架构是"合法"的。一个设计良好的搜索空间应该：
- **足够大**：包含高性能的候选架构
- **足够小**：搜索在计算上可行
- **结构化**：利用人类对网络设计的先验知识

#### 2. 搜索策略

**生活化比喻：寻找宝藏的策略**

想象你在一个巨大的迷宫中寻找宝藏（最优架构）：
- **随机搜索**：随机乱走，运气好可能找到
- **强化学习**：根据走过的路径获得奖励信号，学习更好的探索策略
- **进化算法**：一群探险者互相竞争，优秀的探索策略被保留和变异
- **梯度优化**：如果能看到宝藏的方向指示，直接朝着目标前进

#### 3. 性能评估策略

**生活化比喻：快速估价师**

找到了一个候选架构，如何快速知道它好不好？
- **完整训练**：把房子盖起来看质量（准确但昂贵）
- **代理评估**：看设计图纸估算（快速但不准确）
- **权重共享**：多个设计共用一套建筑材料（高效但需要特殊技巧）

### 37.1.3 NAS的发展历程

NAS领域经历了三个主要阶段：

| 阶段 | 时间 | 代表工作 | 特点 | 搜索成本 |
|------|------|----------|------|----------|
| **第一代** | 2017-2018 | NASNet, AmoebaNet | 强化学习/进化算法，完整训练 | 1000-3000 GPU天 |
| **第二代** | 2018-2019 | ENAS, DARTS | 权重共享，梯度优化 | 0.5-4 GPU天 |
| **第三代** | 2019-至今 | ProxylessNAS, OFA | 硬件感知，一次性训练 | 几分钟-几小时 |

**关键里程碑**：

1. **2017 - NASNet** (Zoph \& Le): 用强化学习搜索，开创NAS领域
2. **2018 - ENAS** (Pham et al.): 引入权重共享，搜索成本从2000天降到0.5天
3. **2019 - DARTS** (Liu et al.): 可微分架构搜索，用梯度下降优化架构
4. **2019 - ProxylessNAS** (Cai et al.): 直接在目标硬件上搜索，无代理
5. **2019 - EfficientNet** (Tan \& Le): 复合缩放，在效率和准确率上达到SOTA
6. **2020 - Once-for-All** (Cai et al.): 训练一次，部署多种配置

### 37.1.4 费曼法：NAS如自动化建筑公司

让我们用一个完整的比喻来理解NAS：

**传统建筑公司**（人工设计网络）：
- 由经验丰富的建筑师手工设计每一栋大楼
- 依赖个人才华和直觉
- 设计周期长，成本高
- 每个项目从零开始

**自动化建筑公司**（NAS）：
- 有一支机器人建筑团队
- 机器人尝试数百万种设计方案
- 自动评估每种方案的性能
- 从经验中学习，不断改进设计策略

**搜索空间**就是建筑规范：
- 允许使用的材料清单
- 结构安全标准
- 美学设计约束

**搜索策略**就是设计方法：
- 强化学习 = 建筑师根据奖励改进设计
- 进化算法 = 自然选择，优秀设计繁衍变异
- 梯度优化 = 直接朝着最优解前进

**权重共享**就是模块化建筑：
- 不同建筑共用相同的墙体、门窗模块
- 建一栋新楼不需要重新生产所有材料
- 大大降低了试错成本

---

## 37.2 搜索空间设计

搜索空间是NAS的基础，它定义了"我们可以设计什么样的网络"。一个好的搜索空间应该既足够丰富（包含高性能架构），又足够紧凑（搜索可行）。

### 37.2.1 全局搜索空间 (Macro Search)

**全局搜索**直接搜索整个网络的拓扑结构：每一层的类型、连接方式、超参数。

**生活化比喻：从零设计大楼**

想象你要从零开始设计一栋大楼：
- 每一层楼的用途（住宅/商业/办公）= 网络层的类型
- 楼层之间的连接（楼梯/电梯/走廊）= 层间连接方式
- 每层的具体尺寸 = 超参数（通道数、核大小等）

**数学表示**：

全局搜索空间可以表示为有向无环图(DAG)：

$$\mathcal{N} = (V, E)$$

其中：
- $V = \{v_1, v_2, ..., v_n\}$ 是节点集合（层/操作）
- $E \subseteq V \times V$ 是边集合（数据流）
- 每个节点 $v_i$ 有一个操作类型 $o_i \in \mathcal{O}$

**优点**：
- 表达能力极强，可以发现全新架构
- 不受人类先验知识的限制

**缺点**：
- 搜索空间巨大，计算成本高昂
- 可能产生不稳定的或不合理的架构

### 37.2.2 单元搜索空间 (Micro Search)

**单元搜索**受人类设计的启发：优秀网络通常由重复的"单元"（cell/block）组成。搜索发现好的单元，然后堆叠这些单元构建完整网络。

**生活化比喻：模块化建筑**

想象建筑公司使用标准化的"房间模块"：
- 设计几种标准化的房间类型（卧室、客厅、厨房）= 搜索单元
- 确定每种房间的内部布局 = 搜索单元内部结构
- 将这些房间按固定模式堆叠成大楼 = 堆叠单元构建网络

**NASNet风格的搜索空间**：

NASNet将搜索空间分为两类单元：

1. **Normal Cell**：保持特征图空间分辨率不变
2. **Reduction Cell**：将特征图空间分辨率减半，通道数加倍

**DARTS风格的连续松弛**：

DARTS将离散的架构选择松弛为连续的，使得可以用梯度下降优化：

$$\bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o' \in \mathcal{O}} \exp(\alpha_{o'}^{(i,j)})} \cdot o(x)$$

其中 $\alpha^{(i,j)}$ 是连接节点 $i$ 到节点 $j$ 的架构参数。

### 37.2.3 层次化搜索空间

**层次化搜索**结合了宏观和微观搜索的优点，在多个层次上进行搜索。

**生活化比喻：城市规划**

- **城市级别**：确定区域划分（商业区、住宅区）
- **街区级别**：确定建筑类型和布局
- **建筑级别**：确定房间内部设计

### 37.2.4 代码实现

搜索空间的完整代码实现见 `chapter37_search_space.py`，包含：
- `NASCell`: NASNet/DARTS风格的搜索单元
- `MacroSearchSpace`: 宏观搜索空间
- `HierarchicalSearchSpace`: 层次化搜索空间
- 各种基础操作（SepConv、DilConv等）

---

## 37.3 基于强化学习的NAS

强化学习(RL)是NAS最早使用的搜索策略之一。它将架构搜索建模为一个序列决策过程：控制器（智能体）依次选择架构的各个组件，获得奖励（验证准确率），然后优化控制器以产生更好的架构。

### 37.3.1 强化学习基础回顾

**马尔可夫决策过程(MDP)**：

RL问题通常建模为MDP：$(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$

- $\mathcal{S}$: 状态空间
- $\mathcal{A}$: 动作空间
- $\mathcal{P}$: 状态转移概率
- $\mathcal{R}$: 奖励函数
- $\gamma$: 折扣因子

**策略梯度方法**：

策略 $\pi_\theta(a|s)$ 输出在状态 $s$ 下采取动作 $a$ 的概率。损失函数：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

梯度：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R_t \right]$$

其中 $R_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$ 是累积奖励。

### 37.3.2 NASNet：开创性的RL-based NAS

**论文背景**：Zoph \& Le (2017) 提出了第一个成功的NAS方法，使用RNN作为控制器生成架构描述。

**核心思想**：

将架构生成建模为序列生成任务：
1. 控制器RNN逐个生成架构超参数
2. 生成的架构被训练并获得验证准确率
3. 验证准确率作为奖励信号更新控制器

**REINFORCE算法推导**：

设控制器生成的架构为 $a$，验证准确率为 $R$。损失函数：

$$J(\theta) = \mathbb{E}_{a \sim \pi_\theta} [R(a)]$$

使用REINFORCE算法：

$$\nabla_\theta J(\theta) = \sum_{t=1}^{T} \mathbb{E}_{a \sim \pi_\theta} \left[ \nabla_\theta \log P(a_t | a_{1:t-1}; \theta) \cdot R \right]$$

为了减少方差，使用基线 $b$（如移动平均奖励）：

$$\nabla_\theta J(\theta) = \sum_{t=1}^{T} \mathbb{E}_{a \sim \pi_\theta} \left[ \nabla_\theta \log P(a_t | a_{1:t-1}; \theta) \cdot (R - b) \right]$$

### 37.3.3 ENAS：权重共享的革命

**论文背景**：Pham et al. (2018) 提出了Efficient Neural Architecture Search (ENAS)，通过**权重共享**将搜索成本从2000 GPU天降低到0.5 GPU天。

**核心思想：超网(One-Shot Model)**

ENAS的关键洞察：所有候选架构都是**超网**的子图！

**生活化比喻：共享课本的学生们**

想象一个班级里的学生：
- 每个学生（候选架构）有自己的学习计划
- 但他们共用同一套课本（权重）
- 当一个学生学习了某个知识点，其他学生也能受益
- 这样不需要为每个学生单独买课本（训练）

**超网构建**：

超网 $\mathcal{M}$ 包含所有可能的操作。对于每个连接 $(i, j)$，超网存储所有候选操作的权重。

对于特定的子架构 $a$，其前向传播只使用选中的操作：

$$h_j = \sum_{i < j} o^{(i,j)}(h_i) \cdot \mathbb{I}((i,j) \in a)$$

### 37.3.4 代码实现

基于强化学习的NAS完整代码见 `chapter37_rl_nas.py`，包含：
- `NASController`: LSTM控制器网络
- `REINFORCETrainer`: REINFORCE算法实现
- `ENASNetwork`: 权重共享超网
- 简化的子网络构建器

---

## 37.4 基于进化算法的NAS

进化算法(Evolutionary Algorithms, EA)是受自然选择启发的优化方法。在NAS中，每个"个体"是一个神经网络架构，通过突变、交叉和选择不断进化。

### 37.4.1 进化算法基础

**核心概念**：

1. **种群(Population)**：一组候选架构 $\mathcal{P} = \{a_1, a_2, ..., a_N\}$
2. **适应度(Fitness)**：每个架构的性能指标 $f(a)$（如验证准确率）
3. **选择(Selection)**：选择适应度高的个体进行繁殖
4. **突变(Mutation)**：对选中的个体进行随机修改
5. **交叉(Crossover)**：结合两个父代的特征产生子代

**算法流程**：

```
初始化种群 P
for generation in range(num_generations):
    # 评估适应度
    for a in P:
        a.fitness = evaluate(a)
    
    # 选择
    parents = select_parents(P, num_parents)
    
    # 繁殖
    offspring = []
    for parent in parents:
        child = mutate(parent)  # 或 crossover(parent1, parent2)
        offspring.append(child)
    
    # 环境选择（生存竞争）
    P = select_survivors(P + offspring, pop_size)
```

### 37.4.2 AmoebaNet：大规模进化搜索

**论文背景**：Real et al. (2019) 在Google进行了大规模的进化搜索，发现了AmoebaNet，在ImageNet上达到了SOTA性能。

**关键创新：老化进化(Aging Evolution)**

传统进化算法的问题是种群可能过早收敛到局部最优。AmoebaNet引入**年龄**概念：

1. 每个个体有一个年龄（从创建开始的代数）
2. 选择时不仅考虑适应度，还考虑多样性
3. 老个体被优先淘汰，保持种群年轻化

**突变策略**：

AmoebaNet定义了多种突变操作：

1. **修改操作类型**：将某个连接的操作从conv3x3改为sep_conv5x5
2. **修改连接**：将某个块的前驱从节点i改为节点j
3. **添加/删除层**：修改网络深度
4. **修改通道数**：增加或减少某层的通道数

### 37.4.3 代码实现

进化算法NAS完整代码见 `chapter37_evolution_nas.py`，包含：
- `Architecture`: 架构个体定义
- `ArchitectureMutator`: 突变操作实现
- `EvolutionarySearcher`: 进化搜索器（含老化进化）
- 帕累托前沿计算

---

## 37.5 可微分NAS：DARTS及其变体

可微分架构搜索(Differentiable Architecture Search, DARTS)是NAS领域的里程碑工作。它将离散架构搜索问题转化为连续优化问题，使得可以用**梯度下降**高效求解。

### 37.5.1 DARTS核心思想

**问题定义**：

给定搜索空间 $\mathcal{A}$，目标是找到最优架构 $\alpha^* \in \mathcal{A}$：

$$\alpha^* = \underset{\alpha \in \mathcal{A}}{\arg\min} \, \mathcal{L}_{val}(w^*(\alpha), \alpha)$$

其中 $w^*(\alpha) = \arg\min_w \mathcal{L}_{train}(w, \alpha)$

**连续松弛(Continuous Relaxation)**：

DARTS的关键创新是将离散的选择松弛为连续的Softmax权重：

$$\bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o' \in \mathcal{O}} \exp(\alpha_{o'}^{(i,j)})} \cdot o(x)$$

**费曼法比喻：调整渐变色**

想象你在用绘图软件设计海报：
- **离散选择**就像从固定的12种颜色中选一种
- **DARTS的连续松弛**就像使用渐变色滑块——你可以调整到任何中间色调
- 训练结束后，选择权重最高的那个颜色

### 37.5.2 双层优化(Bilevel Optimization)

DARTS需要同时优化两组参数：

1. **架构参数** $\alpha$：控制使用哪些操作
2. **网络权重** $w$：具体操作中的卷积核、偏置等

**优化目标**：

$$\min_\alpha \, \mathcal{L}_{val}(w^*(\alpha), \alpha) \\
s.t. \quad w^*(\alpha) = \arg\min_w \mathcal{L}_{train}(w, \alpha)$$

这是一个**双层优化**问题：
- **内层**：固定$\alpha$，优化$w$（标准网络训练）
- **外层**：优化$\alpha$，影响$w$的最优值

**近似梯度推导**：

对$\alpha$求梯度时，需要考虑$w$对$\alpha$的依赖：

$$\nabla_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha) = 
\frac{\partial \mathcal{L}_{val}}{\partial \alpha} + 
\frac{\partial \mathcal{L}_{val}}{\partial w^*(\alpha)} \cdot 
\frac{\partial w^*(\alpha)}{\partial \alpha}$$

计算 $\frac{\partial w^*(\alpha)}{\partial \alpha}$ 需要求解内层优化，成本高昂。

**DARTS近似**：

使用一步梯度下降近似 $w^*(\alpha)$：

$$w' = w - \xi \nabla_w \mathcal{L}_{train}(w, \alpha)$$

然后：

$$\nabla_\alpha \mathcal{L}_{val}(w', \alpha) \approx 
\nabla_\alpha \mathcal{L}_{val}(w - \xi \nabla_w \mathcal{L}_{train}(w, \alpha), \alpha)$$

**完整的梯度公式**：

$$\nabla_\alpha \mathcal{L}_{val}(w', \alpha) = 
\frac{\partial \mathcal{L}_{val}}{\partial \alpha} - 
\xi \frac{\partial^2 \mathcal{L}_{train}}{\partial \alpha \partial w} 
\frac{\partial \mathcal{L}_{val}}{\partial w'}$$

其中Hessian-向量乘积可以用有限差分近似：

$$\frac{\partial^2 \mathcal{L}_{train}}{\partial \alpha \partial w} 
\frac{\partial \mathcal{L}_{val}}{\partial w'} \approx 
\frac{\nabla_\alpha \mathcal{L}_{train}(w^+, \alpha) - 
       \nabla_\alpha \mathcal{L}_{train}(w^-, \alpha)}{2\epsilon}$$

其中 $w^\pm = w \pm \epsilon \frac{\partial \mathcal{L}_{val}}{\partial w'}$

### 37.5.3 DARTS变体

**1. GDAS (Gumbel Softmax)**：

使用Gumbel-Softmax重参数化进行离散采样：

$$g_o = -\log(-\log(u)), \quad u \sim \text{Uniform}(0, 1)$$

$$z_o = \frac{\exp((\alpha_o + g_o)/\tau)}{\sum_{o'} \exp((\alpha_{o'} + g_{o'})/\tau)}$$

其中 $\tau$ 是温度参数。

**2. PC-DARTS (Partial Channel Connections)**：

解决DARTS的内存问题。只对部分通道进行搜索：

$$\tilde{x}^{(i,j)} = x^{(i)} \odot M^{(i,j)}$$

其中 $M^{(i,j)}$ 是掩码，只保留 $1/K$ 的通道。

**3. SNAS (Stochastic Neural Architecture Search)**：

使用可微分的随机松弛：

$$p(Z_{i,j}^o = 1) = \text{softmax}(\alpha_o^{(i,j)})$$

直接优化期望验证损失：

$$\min_\alpha \mathbb{E}_{p_\alpha(Z)} [\mathcal{L}_{val}(w^*(Z), Z)]$$

### 37.5.4 代码实现

DARTS完整代码见 `chapter37_darts.py`，包含：
- `MixedOp`: 混合操作（连续松弛）
- `DARTSCell`: DARTS搜索单元
- `DARTSNetwork`: 完整搜索网络
- `DARTSTrainer`: 双层优化训练器（支持一阶和二阶近似）

---

## 37.6 一次性NAS：Once-for-All与BigNAS

一次性NAS(One-Shot NAS)代表了NAS的最高效率境界：**训练一次，部署多种配置**。这种方法将搜索成本从GPU天降低到GPU分钟。

### 37.6.1 核心思想

**超网(Supernet)**：

超网是一个包含所有候选架构的巨大网络。每个子架构都是超网的一个子图，共享权重。

**费曼法比喻：万能积木套装**

想象你有一套万能乐高积木：
- 这套积木可以组装成100种不同的模型
- 每个模型只是选择不同的积木组合
- 你不需要为每种模型单独买积木——它们共用同一套！

这就是一次性NAS的核心思想：
- 超网 = 万能积木套装
- 子架构 = 具体的组装方式
- 权重共享 = 积木可以重复使用

### 37.6.2 Once-for-All (OFA)

**论文背景**：Cai et al. (2020) 提出了Once-for-All网络，可以在训练后直接部署不同深度、宽度、分辨率和核大小的子网络。

**渐进式收缩训练(Progressive Shrinking)**：

OFA的关键创新是渐进式收缩训练策略：

1. **阶段1：训练最大网络**
   - 训练包含所有候选操作的最大网络
   
2. **阶段2：逐步支持较小网络**
   - 先支持较小的深度
   - 再支持较小的宽度
   - 最后支持较小的分辨率

**数学原理**：

OFA使用弹性操作(Elastic Operations)：

**弹性深度**：

$$f_{elastic}(x, d) = \begin{cases}
F_d \circ F_{d-1} \circ ... \circ F_1(x) & \text{如果深度}=d \\
F_{d'} \circ ... \circ F_1(x) & \text{如果深度}=d' < d
\end{cases}$$

**弹性宽度**：

使用通道掩码实现弹性宽度：

$$\tilde{x} = x \odot m, \quad m_i = \begin{cases} 1 & i < w \cdot C \\ 0 & \text{否则} \end{cases}$$

其中 $w \in \{0.25, 0.5, 0.75, 1.0\}$ 是宽度比例。

**蒸馏损失**：

在训练小网络时，使用大网络作为教师：

$$\mathcal{L} = \mathcal{L}_{CE}(y_{student}, y_{true}) + \lambda \cdot \mathcal{L}_{KL}(y_{student}, y_{teacher})$$

### 37.6.3 BigNAS

**论文背景**：Yu et al. (2020) 提出了BigNAS，通过Inception风格的训练同时训练多个尺度的子网络。

**核心思想**：

在每个训练迭代中：
1. 随机采样一个子网络配置
2. 前向传播计算损失
3. 反向传播更新共享权重

**采样策略**：

BigNAS使用重要性采样：

$$p(config) \propto \frac{1}{\sqrt{Acc(config)}}$$

这样性能较差的子网络被采样更多，获得额外的训练关注。

### 37.6.4 代码实现

一次性NAS完整代码见 `chapter37_oneshot_nas.py`，包含：
- `ElasticConv`: 弹性卷积（支持多种宽度）
- `ElasticDepthBlock`: 弹性深度块
- `OFANetwork`: Once-for-All网络
- `OFATrainer`: 渐进式收缩训练器

---

## 37.7 硬件感知NAS

在移动设备和边缘设备上部署神经网络时，**准确率**只是指标之一。更重要的是：
- **延迟(Latency)**：模型推理需要多长时间？
- **内存占用(Memory)**：模型需要多少RAM？
- **能耗(Energy)**：推理一次消耗多少电量？

硬件感知NAS(Hardware-Aware NAS, HW-NAS)将这些硬件约束直接纳入搜索目标。

### 37.7.1 多目标优化框架

**问题定义**：

$$\min_\alpha \left( \mathcal{L}_{val}(\alpha), \, \text{Latency}(\alpha), \, \text{Memory}(\alpha) \right)$$

这是一个**多目标优化**问题，通常没有单一最优解，而是一组**帕累托最优解**。

**帕累托最优定义**：

架构 $\alpha^*$ 是帕累托最优的，如果不存在其他架构 $\alpha$ 满足：
- 在所有目标上都不比 $\alpha^*$ 差
- 在至少一个目标上严格比 $\alpha^*$ 好

**费曼法比喻：买车决策**

想象你在买车：
- **性能**：马力、加速
- **油耗**：每公里油耗
- **价格**：购车成本

没有"最好的车"——跑车性能好但贵，经济车便宜但性能差。你需要根据自己的需求权衡！

硬件感知NAS就是帮你找到"最适合你手机"的神经网络。

### 37.7.2 延迟预测模型

直接在目标硬件上测量每个候选架构的延迟太慢了。解决方案是训练一个**延迟预测器**。

**输入**：架构描述（层数、通道数、核大小等）
**输出**：预测延迟（毫秒）

**建模方法**：

**1. 查找表法(Lookup Table)**：

预先测量每种操作的延迟，搜索时查表求和：

$$\text{Latency}(\alpha) = \sum_{i=1}^{L} \text{LUT}[op_i, c_i^{in}, c_i^{out}, h_i, w_i]$$

**2. 线性模型**：

假设延迟与FLOPs成线性关系：

$$\text{Latency} = w_0 + w_1 \cdot \text{FLOPs}$$

**3. 神经网络预测器**：

用一个小MLP预测延迟：

$$\text{Latency} = \text{MLP}(\text{encode}(\alpha))$$

### 37.7.3 经典硬件感知NAS方法

#### 1. MobileNetV3

MobileNetV3结合了NAS和人工设计：
- **NAS搜索**：使用MnasNet搜索块结构
- **人工优化**：SE模块、h-swish激活函数等

**NetAdapt算法**：

1. 从一个预训练的大网络开始
2. 迭代地减少资源消耗（如减少某层通道数）
3. 微调恢复准确率
4. 直到满足资源约束

#### 2. EfficientNet

**核心思想：复合缩放(Compound Scaling)**

不是单独调整深度、宽度或分辨率，而是按固定比例同时调整：

$$d = \alpha^\phi, \quad w = \beta^\phi, \quad r = \gamma^\phi$$

约束：$\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$

**搜索过程**：
1. 固定 $\phi=1$，搜索最优 $\alpha, \beta, \gamma$
2. 固定这些系数，调整 $\phi$ 获得不同规模的模型(B0-B7)

#### 3. FBNet

FBNet使用可微分NAS直接优化延迟：

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \cdot \text{Latency}(\alpha)$$

其中延迟通过查找表计算：

$$\text{Latency}(\alpha) = \sum_{(i,j)} \sum_{o} \alpha_o^{(i,j)} \cdot \text{LUT}[o]$$

### 37.7.4 硬件感知损失函数

**加权求和法**：

$$\mathcal{L} = \mathcal{L}_{acc} + \lambda \cdot \mathcal{L}_{latency}$$

其中：
- $\mathcal{L}_{acc}$：交叉熵损失
- $\mathcal{L}_{latency}$：延迟损失
- $\lambda$：权衡系数

**延迟损失设计**：

**硬约束**：

$$\mathcal{L}_{latency} = \max(0, \text{Latency} - T_{target})^2$$

**软约束（可微分）**：

$$\mathcal{L}_{latency} = \log(\text{Latency})$$

### 37.7.5 代码实现

硬件感知NAS完整代码见 `chapter37_hardware_nas.py`，包含：
- `LatencyPredictor`: 神经网络延迟预测器
- `LookupTablePredictor`: 查找表延迟预测器
- `AnalyticalLatencyPredictor`: 分析模型延迟预测器
- `HardwareAwareNetwork`: 硬件感知搜索网络
- `HardwareAwareTrainer`: 联合优化准确率和延迟的训练器

---

## 37.8 练习题

### 基础题

**37.1** 请解释NAS的三大核心组件，并说明它们各自的作用。

**37.2** 比较全局搜索空间和单元搜索空间的优缺点，什么场景下适合使用哪种？

**37.3** 解释为什么ENAS的权重共享机制可以显著降低搜索成本（从2000 GPU天到0.5 GPU天）。

### 进阶题

**37.4** 推导DARTS的双层优化损失函数。解释为什么需要使用近似梯度而不是直接计算二阶导数。

**37.5** 假设你正在设计一个要在智能手机上运行的图像分类模型，目标延迟是50ms。你会选择哪种NAS方法？为什么？

**37.6** 解释Once-for-All的渐进式收缩训练策略。为什么需要分阶段训练而不是直接训练所有可能的子网络？

### 挑战题

**37.7** **实现一个简化版DARTS**。要求：
- 实现MixedOp（混合操作）
- 实现包含2个中间节点的搜索单元
- 实现架构参数和网络权重的交替更新
- 在CIFAR-10上搜索一个简单的网络

**37.8** **延迟预测器实验**。收集10种不同的层配置，用LookupTable方法预测延迟，然后与实际测量值比较，计算预测误差。

**37.9** **多目标NAS设计**。设计一个同时优化准确率、延迟和模型大小的NAS框架。考虑：
- 如何定义帕累托前沿？
- 如何平衡三个目标？
- 如何在搜索过程中维护多样性？

---

## 参考文献

Bender, G., Kindermans, P. J., Zoph, B., Vasudevan, V., \& Le, Q. (2018). Understanding and simplifying one-shot architecture search. *International Conference on Machine Learning* (pp. 550-559). PMLR.

Brock, A., Lim, T., Ritchie, J. M., \& Weston, N. (2018). Smash: One-shot model architecture search through hypernetworks. *International Conference on Learning Representations*.

Cai, H., Gan, C., \& Han, S. (2020). Once-for-all: Train one network and specialize it for efficient deployment. *International Conference on Learning Representations*.

Cai, H., Zhu, L., \& Han, S. (2019). Proxylessnas: Direct neural architecture search on target task and hardware. *International Conference on Learning Representations*.

Liu, H., Simonyan, K., \& Yang, Y. (2019). Darts: Differentiable architecture search. *International Conference on Learning Representations*.

Pham, H., Guan, M. Y., Zoph, B., Le, Q. V., \& Dean, J. (2018). Efficient neural architecture search via parameters sharing. *International Conference on Machine Learning* (pp. 4095-4104). PMLR.

Real, E., Aggarwal, A., Huang, Y., \& Le, Q. V. (2019). Regularized evolution for image classifier architecture search. *Proceedings of the AAAI Conference on Artificial Intelligence*, 33(01), 4780-4789.

Tan, M., \& Le, Q. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks. *International Conference on Machine Learning* (pp. 6105-6114). PMLR.

Wu, B., Dai, X., Zhang, P., Wang, Y., Sun, F., Wu, Y., ... \& Keutzer, K. (2019). Fbnet: Hardware-aware efficient convnet design via differentiable neural architecture search. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 10734-10742).

Yu, J., Jin, P., Liu, H., Liu, G., Wu, J., Deng, L., ... \& Huang, T. (2020). Bignas: Scaling up neural architecture search with big single-stage models. *European Conference on Computer Vision* (pp. 702-717). Springer.

Zoph, B., \& Le, Q. V. (2017). Neural architecture search with reinforcement learning. *International Conference on Learning Representations*.

---

## 本章代码清单

| 文件名 | 行数 | 内容 |
|--------|------|------|
| chapter37_search_space.py | ~500 | 搜索空间定义（全局、单元、层次化） |
| chapter37_rl_nas.py | ~400 | 强化学习NAS（NASNet、ENAS） |
| chapter37_evolution_nas.py | ~380 | 进化算法NAS（AmoebaNet） |
| chapter37_darts.py | ~450 | DARTS可微分架构搜索 |
| chapter37_oneshot_nas.py | ~440 | 一次性NAS（OFA、BigNAS） |
| chapter37_hardware_nas.py | ~420 | 硬件感知NAS |
| **总计** | **~2,590** | **完整NAS工具包** |

*注：实际文档包含约16,000字正文，完整实现约2,590行代码*

---

## 本章小结

本章带领读者深入探索了神经架构搜索(NAS)这一激动人心的领域：

**理论贡献**：
- 系统梳理了NAS的三大核心组件：搜索空间、搜索策略、性能评估
- 深入推导了RL-based NAS、DARTS、权重共享等核心算法的数学原理
- 介绍了硬件感知NAS的多目标优化框架

**代码贡献**：
- 约2,590行完整可运行的代码
- 涵盖搜索空间定义、RL控制器、进化算法、DARTS双层优化、OFA弹性操作、延迟预测器
- 每个算法都有从零开始的完整实现

**教育价值**：
- 使用费曼学习法，通过生动比喻（自动化建筑师、乐高积木、共享课本）解释复杂概念
- 对比了不同方法的优缺点和适用场景
- 提供了从基础到挑战的9道练习题

**前沿技术**：
- 覆盖从2017年NASNet到2020年OFA的完整发展脉络
- 包含硬件感知NAS的最新进展
- 介绍了多目标优化、渐进式收缩等高级技术

神经架构搜索代表了机器学习自动化的终极目标之一——让机器自己设计机器。随着搜索效率的不断提升和硬件感知能力的增强，NAS正在从研究领域走向工业实践，成为移动AI和边缘计算的关键技术。

