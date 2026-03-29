# 第三十四章：神经架构搜索——让机器自己设计神经网络

## 本章概览

在前面的章节中，我们学习了各种神经网络架构：从卷积神经网络(CNN)到循环神经网络(RNN)，从Transformer到图神经网络。这些架构大多由人类专家精心设计，需要深厚的领域知识和大量试错。但想象一下，如果机器能够像人类专家一样，自动设计出高效的神经网络架构，那将是多么令人兴奋的事情！

神经架构搜索(Neural Architecture Search, NAS)正是实现这一愿景的技术。作为AutoML的核心组成部分，NAS让机器自动探索神经网络的设计空间，发现超越人类专家设计的架构。从2016年Zoph和Le的开创性工作至今，NAS已经发展出多种搜索范式：强化学习、进化算法、可微分搜索，以及最新的Training-free方法。

本章将带你深入了解NAS的奥秘：从搜索空间的巧妙设计，到DARTS的可微分优化，再到无需训练的零成本代理。我们将用生动的比喻解释抽象概念，用完整的数学推导展示算法原理，用手写代码实现核心模块。准备好了吗？让我们一起探索机器自主设计神经网络的美妙世界！

---

## 34.1 什么是神经架构搜索？

### 34.1.1 从手工设计到自动搜索的演进

**生活比喻：建筑师与自动设计软件**

想象你是一位建筑师，要设计一座房子。传统方式是建筑师凭借经验和灵感，一砖一瓦地设计每个细节——这就像人类专家手工设计神经网络，需要深厚的专业知识、大量的试错和无数次的调优。

现在，想象一下有一款"智能建筑设计软件"：你只需要告诉它"我需要一座三居室的房子，预算100万，要采光好"，软件就能自动生成上百个设计方案，自动评估每个方案的成本、采光、空间利用率，最终推荐最优方案。这就是NAS的魅力所在——让机器自动探索、评估和选择最佳的神经网络架构。

**NAS的发展历程**

神经架构搜索的历史可以追溯到2016年，Google Brain的Zoph和Le发表了开创性论文《Neural Architecture Search with Reinforcement Learning》，首次使用强化学习来自动设计神经网络架构。这项工作在CIFAR-10和Penn Treebank上取得了当时最先进的性能，但代价是巨大的计算资源——需要800个GPU训练28天！

这个惊人的计算成本激发了研究者们的创新：能否让NAS更快、更便宜？于是，一系列突破相继诞生：

- **2017年**：Baker等人提出MetaQNN，使用强化学习探索架构空间
- **2018年**：Pham等人提出ENAS，引入权重共享，将搜索成本从数千GPU天降至不到1天
- **2018年**：Zoph等人提出NASNet，设计可迁移的Cell结构，让小数据集上学到的架构可以迁移到大数据集
- **2019年**：Liu等人提出DARTS，将离散的架构搜索转化为连续的优化问题，使用梯度下降高效求解
- **2019年**：Real等人提出AmoebaNet，使用正则化进化算法搜索架构，首次超越人工设计
- **2019年**：Tan和Le提出EfficientNet，通过复合缩放统一优化网络深度、宽度和分辨率
- **2020年**：Cai等人提出Once-for-All，训练一个超网络，可以导出无数个专用子网络
- **2021年**：Mellor等人提出NASWOT，发现无需训练就能预测架构性能的方法

如今，NAS已经从实验室走向工业界：Google的AutoML、华为的AutoML套件、微软的NNI等工具让普通开发者也能享受自动架构设计的便利。

### 34.1.2 NAS的三大组件

神经架构搜索可以被抽象为一个优化问题，包含三个核心组件：

$$
\alpha^* = \arg\max_{\alpha \in \mathcal{A}} \mathcal{P}(\mathcal{N}_\alpha, \mathcal{D}_{val})
$$

其中，$\alpha$表示架构，$\mathcal{A}$是搜索空间，$\mathcal{P}$是性能评估函数，$\mathcal{D}_{val}$是验证集。

**组件一：搜索空间(Search Space)**

搜索空间定义了所有可能的神经网络架构集合。设计一个好的搜索空间就像设计乐高的积木套装——积木的种类和连接方式决定了你能搭建出什么。

搜索空间可以分为两类：

1. **全局搜索空间**：直接搜索整个网络的拓扑结构，包括层数、每层的操作类型、连接方式等。这种空间非常庞大，搜索难度高。

2. **Cell-based搜索空间**：这是更实用的设计。Cell是一个小的计算单元，就像乐高积木块。我们只需要搜索Cell的结构，然后通过重复堆叠Cell来构建完整网络。NASNet、DARTS、ENAS都采用这种设计。

一个典型的Cell包含：
- **输入节点**：接收前两个Cell的输出
- **中间节点**：执行具体操作（如卷积、池化）
- **输出节点**：聚合中间节点的结果

**组件二：搜索策略(Search Strategy)**

搜索策略决定如何探索搜索空间，找到最优架构。主要有三种范式：

1. **强化学习(Reinforcement Learning)**：将架构设计看作序列决策问题。控制器(RNN)生成架构描述，训练该架构得到奖励(准确率)，用策略梯度更新控制器。

2. **进化算法(Evolutionary Algorithms)**：模拟自然选择。维护一个架构种群，通过变异和选择不断进化，优胜劣汰。

3. **可微分搜索(Differentiable Search)**：将离散的架构选择松弛为连续的权重，使用梯度下降同时优化架构参数和网络权重。

**组件三：性能评估(Performance Estimation)**

评估每个候选架构的性能是最耗时的部分。原始方法是完整训练每个架构，这在计算上不可行。研究者们提出了多种加速方法：

1. **低精度估计**：用少量epoch、小数据集或低分辨率图像快速评估
2. **权重共享**：让多个架构共享参数，避免从头训练
3. **性能预测器**：训练一个预测器，根据架构描述直接预测性能
4. **零成本代理**：无需训练，在初始化时就评估架构潜力

---

## 34.2 搜索空间设计

### 34.2.1 全局搜索 vs Cell-based搜索

**生活比喻：从零设计 vs 乐高积木**

全局搜索就像从零开始设计一座房子——你需要决定每一面墙的位置、每一扇窗的大小、每一根梁的粗细。这种自由度很高，但设计空间极其庞大，容易迷失方向。

Cell-based搜索就像用乐高积木搭建——乐高公司已经为你设计好了标准化的积木块（Cell），你只需要决定如何组合这些积木。这种方法大大减小了搜索空间，同时保持了足够的表达能力。

**全局搜索空间的挑战**

全局搜索空间面临的主要问题是"维度灾难"。假设我们要设计一个CNN：
- 网络深度：1-50层
- 每层操作类型：卷积、池化、跳跃连接等10种选择
- 卷积核大小：3×3、5×5、7×7
- 通道数：16、32、64、128、256

仅这些选择就产生了天文数字般的组合。更糟糕的是，不同层之间还存在复杂的依赖关系。

**Cell-based搜索的优势**

Cell-based设计借鉴了人类专家的经验：优秀的人工设计网络（如ResNet、DenseNet）往往采用重复模块的结构。Cell正是这种重复模块的抽象。

一个Cell通常包含B个块(block)，每个块包含：
- 2个输入选择（从前面节点中选择）
- 2个操作选择（从候选操作集合中选择）
- 1个合并操作（通常是拼接或相加）

假设B=5，候选操作有7种，则搜索空间大小约为$(7 \times 5)^4 \approx 10^8$，虽然仍然很大，但比全局搜索可行多了。

### 34.2.2 NASNet搜索空间详解

NASNet搜索空间是NAS领域最具影响力的设计之一，被NASNet、AmoebaNet、DARTS等多种方法采用。

**搜索空间结构**

NASNet搜索两种Cell：
1. **Normal Cell**：保持特征图空间分辨率不变
2. **Reduction Cell**：将特征图空间分辨率减半（通常通道数翻倍）

一个完整的网络由这两种Cell按固定模式堆叠而成：

```
输入 → [Normal Cell × N] → [Reduction Cell] → [Normal Cell × N] → [Reduction Cell] → [Normal Cell × N] → 输出
```

**Cell内部结构**

每个Cell是一个有向无环图(DAG)，包含：
- **2个输入节点**：前两个Cell的输出（对于第一个Cell，输入是图像）
- **B个中间节点**：每个节点通过二元操作组合两个前驱节点
- **1个输出节点**：拼接所有未被使用的中间节点

**候选操作集合**

每条边（从前驱节点到当前节点）的操作从以下候选集合中选择：

1. **3×3可分离卷积(Depthwise Separable Conv)**
2. **5×5可分离卷积**
3. **3×3空洞可分离卷积(Dilated Conv)**
4. **3×3最大池化**
5. **3×3平均池化**
6. **跳跃连接(Identity)**
7. **无连接(None)**

**二元操作**

两个前驱节点的输出通过以下方式合并：
1. **逐元素相加(Element-wise Addition)**
2. **通道拼接(Concatenation)**

**数学表示**

设Cell有B个中间节点，第i个节点的计算为：

$$
x^{(i)} = \text{op}_i(x^{(j)}) \oplus \text{op}'_i(x^{(k)})
$$

其中$j, k < i$是前驱节点索引，$\text{op}_i, \text{op}'_i$是从候选集合中选择的操作，$\oplus$是相加或拼接。

### 34.2.3 操作集合定义

**生活比喻：食谱配料**

候选操作就像食谱中的配料——不同的配料组合产生不同的菜肴。NAS的任务就是找到最佳的"配方"。

让我们详细了解每种操作：

**1. 可分离卷积(Depthwise Separable Convolution)**

这是MobileNet引入的高效卷积方式，将标准卷积分解为两步：

第一步：Depthwise卷积——每个输入通道单独做空间卷积
```
输入: [H, W, C] → 输出: [H, W, C]
```

第二步：Pointwise卷积——1×1卷积混合通道信息
```
输入: [H, W, C] → 输出: [H, W, C']
```

计算复杂度从$O(H \times W \times C \times K^2 \times C')$降至$O(H \times W \times C \times K^2 + H \times W \times C \times C')$，通常减少8-9倍。

**2. 空洞卷积(Dilated Convolution)**

在卷积核中插入"空洞"，扩大感受野而不增加参数：

```
标准3×3卷积核:    空洞3×3卷积核(rate=2):
[1 1 1]           [1 0 1 0 1]
[1 1 1]    →      [0 0 0 0 0]
[1 1 1]           [1 0 1 0 1]
                  [0 0 0 0 0]
                  [1 0 1 0 1]
```

感受野从3×3扩大到5×5，参数数量不变。

**3. 池化操作**

- **最大池化**：保留最显著特征，具有平移不变性
- **平均池化**：平滑特征，保留背景信息

**4. 跳跃连接(Identity)**

直接传递特征，不改变内容。这是ResNet的核心设计，帮助梯度流动，训练更深网络。

**5. 无连接(None)**

表示两个节点之间没有连接，增加拓扑灵活性。

---

## 34.3 强化学习方法：NASNet与ENAS

### 34.3.1 RNN控制器原理

**生活比喻：选秀比赛**

想象一个选秀节目：主持人（RNN控制器）需要决定选手的出场顺序、表演曲目、舞台风格（架构决策）。每个决定后，选手上台表演（训练网络），评委打分（验证准确率）。主持人根据分数学习，逐渐掌握什么样的组合更受欢迎。

**控制器设计**

NASNet使用循环神经网络(RNN)作为控制器来生成架构描述。控制器是一个自回归模型，按顺序决策：

1. 为Cell中的每条边选择隐藏状态（从前驱节点中选）
2. 为每个隐藏状态选择操作类型
3. 为每个块选择合并操作

**Softmax采样过程**

控制器的第t步决策：

$$
a_t \sim \text{Softmax}(W_t \cdot h_t + b_t)
$$

其中$h_t$是RNN的隐藏状态，$W_t, b_t$是可学习参数。

RNN状态更新：

$$
h_{t+1} = \text{LSTM}(e_t, h_t)
$$

其中$e_t$是上一决策$a_t$的嵌入向量。

### 34.3.2 策略梯度训练

**奖励函数**

控制器的目标是最大化期望奖励：

$$
J(\theta_c) = \mathbb{E}_{P(a_{1:T}; \theta_c)}[R]
$$

其中$R$是生成的架构在验证集上的准确率。

**策略梯度推导**

根据策略梯度定理(REINFORCE)：

$$
\nabla_{\theta_c} J(\theta_c) = \sum_{t=1}^{T} \mathbb{E}_{P(a_{1:T}; \theta_c)}\left[ \nabla_{\theta_c} \log P(a_t | a_{1:t-1}; \theta_c) \cdot R \right]
$$

实际实现使用蒙特卡洛采样估计梯度：

$$
\nabla_{\theta_c} J(\theta_c) \approx \frac{1}{m} \sum_{k=1}^{m} \sum_{t=1}^{T} \nabla_{\theta_c} \log P(a_t | a_{1:t-1}; \theta_c) \cdot R_k
$$

**基线减小方差**

原始REINFORCE梯度估计方差很大。引入基线$b$（通常是移动平均奖励）：

$$
\nabla_{\theta_c} J(\theta_c) \approx \frac{1}{m} \sum_{k=1}^{m} \sum_{t=1}^{T} \nabla_{\theta_c} \log P(a_t | a_{1:t-1}; \theta_c) \cdot (R_k - b)
$$

**完整训练流程**

```python
# 伪代码
for epoch in range(num_epochs):
    # 控制器采样m个架构
    architectures = controller.sample(m)
    
    rewards = []
    for arch in architectures:
        # 训练子网络
        train(arch, train_data)
        # 验证准确率作为奖励
        acc = evaluate(arch, val_data)
        rewards.append(acc)
    
    # 更新基线
    baseline = 0.9 * baseline + 0.1 * mean(rewards)
    
    # 更新控制器
    for arch, reward in zip(architectures, rewards):
        advantage = reward - baseline
        loss = -log_prob(arch) * advantage
        loss.backward()
    
    optimizer_controller.step()
```

### 34.3.3 ENAS权重共享机制

**生活比喻：多功能工具箱**

想象你有一个多功能工具箱，里面有螺丝刀、扳手、钳子等工具（候选操作）。不同的维修任务需要不同的工具组合（不同架构）。ENAS的聪明之处在于：所有任务共享同一个工具箱，你只需要选择用什么工具，而不用为每个任务单独购买工具（训练参数）。

**超网络设计**

ENAS的核心创新是构建一个包含所有候选架构的超网络(supernetwork)。超网络中，每个候选操作都对应一组共享参数。

对于搜索空间中的每条边$e$，定义：

$$
o^{(e)}(x) = \sum_{i=1}^{K} \text{mask}_i^{(e)} \cdot \text{op}_i(x; \theta_i)
$$

其中$\text{mask}_i^{(e)} \in \{0, 1\}$表示边$e$是否使用操作$\text{op}_i$，$\theta_i$是共享的操作参数。

**子图采样**

ENAS的控制器不仅决定架构，还决定激活超网络中的哪个子图。每次迭代：

1. 控制器采样一个架构（即一组mask）
2. 只训练该架构对应的子图
3. 更新子图参数（其他架构共享这些参数）
4. 根据验证性能更新控制器

**数学形式**

设完整超网络为$\mathcal{N}_{super}$，采样子网络为$\mathcal{N}_\alpha$，则：

$$
\mathcal{L}_{train}(\theta, \alpha) = \frac{1}{N} \sum_{i=1}^{N} \ell(f(x_i; \theta \odot \alpha), y_i)
$$

其中$\alpha$是架构的one-hot编码，$\theta \odot \alpha$表示只保留选定架构的参数。

**训练效率提升**

原始NAS需要训练每个架构数千个step：
- 训练20000个架构 × 50 epoch = 100万epoch

ENAS只需要训练超网络：
- 训练1个超网络 × 150 epoch = 150 epoch

搜索成本从约2000 GPU天降至约0.5 GPU天，提升了4000倍！

---

## 34.4 可微分搜索：DARTS

### 34.4.1 连续松弛思想

**生活比喻：食谱调配**

想象你在调配一种特殊的酱汁，有7种配料可选（候选操作）。传统方法是选择一种或几种配料，这就像做离散选择——要么用，要么不用。

DARTS的巧妙之处在于：它允许你以连续的方式"混合"配料。比如用30%酱油+40%醋+30%糖，这种"软选择"是可微分的，可以用梯度下降优化。当优化完成后，你选择占比最高的配料作为最终配方。

**连续松弛原理**

DARTS的核心创新是将离散的架构选择转化为连续的权重优化。

对于Cell中的每条边$(i, j)$，传统方法是选择一个操作：

$$
o^{(i,j)} \in \mathcal{O} = \{o_1, o_2, ..., o_K\}
$$

DARTS将其松弛为所有操作的加权和：

$$
\bar{o}^{(i,j)}(x) = \sum_{k=1}^{K} \frac{\exp(\alpha_k^{(i,j)})}{\sum_{k'=1}^{K} \exp(\alpha_{k'}^{(i,j)})} \cdot o_k(x)
$$

其中$\alpha^{(i,j)} \in \mathbb{R}^K$是架构参数，通过softmax归一化得到混合权重。

**Softmax近似的优势**

1. **可微分**：架构参数$\alpha$可以像普通网络参数一样求梯度
2. **连续空间**：优化在连续空间进行，比离散搜索高效得多
3. **梯度共享**：所有操作同时参与训练，共享梯度信息

### 34.4.2 双层优化问题

DARTS将NAS形式化为双层优化(bi-level optimization)问题：

**内层优化（网络权重）**：

$$
\theta^*(\alpha) = \arg\min_\theta \mathcal{L}_{train}(\theta, \alpha)
$$

**外层优化（架构参数）**：

$$
\alpha^* = \arg\min_\alpha \mathcal{L}_{val}(\theta^*(\alpha), \alpha)
$$

这是一个嵌套优化问题：架构参数$\alpha$影响最优网络权重$\theta^*$，而$\theta^*$又影响验证损失。

**近似梯度计算**

直接计算$\nabla_\alpha \mathcal{L}_{val}(\theta^*(\alpha), \alpha)$需要求解完整的内层优化，计算量巨大。DARTS使用一步近似：

$$
\theta' = \theta - \xi \nabla_\theta \mathcal{L}_{train}(\theta, \alpha)
$$

然后近似架构梯度：

$$
\nabla_\alpha \mathcal{L}_{val}(\theta^*(\alpha), \alpha) \approx \nabla_\alpha \mathcal{L}_{val}(\theta', \alpha)
$$

**链式法则展开**

使用链式法则：

$$
\nabla_\alpha \mathcal{L}_{val}(\theta', \alpha) = \frac{\partial \mathcal{L}_{val}}{\partial \alpha} + \frac{\partial \mathcal{L}_{val}}{\partial \theta'} \cdot \frac{\partial \theta'}{\partial \alpha}
$$

其中：

$$
\frac{\partial \theta'}{\partial \alpha} = -\xi \frac{\partial}{\partial \alpha} \nabla_\theta \mathcal{L}_{train}(\theta, \alpha)
$$

这涉及二阶导数（Hessian矩阵），计算复杂度高。DARTS提供了一阶近似（忽略二阶项）：

$$
\nabla_\alpha \mathcal{L}_{val}(\theta', \alpha) \approx \frac{\partial \mathcal{L}_{val}}{\partial \alpha}
$$

即假设$\theta$与$\alpha$无关，大大简化计算。

### 34.4.3 离散化与架构导出

搜索完成后，需要从连续的架构参数导出离散架构。DARTS采用以下策略：

**边选择**：

对于每个中间节点，保留权重最大的两条输入边：

$$
\text{TopK}_{k=1}^{2} \left\{ \max_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o'} \exp(\alpha_{o'}^{(i,j)})} \right\}
$$

**操作选择**：

对于保留的每条边，选择权重最大的操作：

$$
o^{(i,j)*} = \arg\max_{o \in \mathcal{O}} \alpha_o^{(i,j)}
$$

**完整导出流程**：

```python
def derive_architecture(alphas, num_nodes, top_k=2):
    """
    从连续架构参数导出离散架构
    """
    genotype = []
    
    for node_idx in range(2, num_nodes + 2):  # 中间节点
        # 收集所有前驱边的权重
        edge_weights = []
        for prev_idx in range(node_idx):
            weights = F.softmax(alphas[prev_idx][node_idx], dim=-1)
            max_weight = weights.max().item()
            best_op = weights.argmax().item()
            edge_weights.append((max_weight, prev_idx, best_op))
        
        # 选择top_k条边
        edge_weights.sort(reverse=True)
        selected = edge_weights[:top_k]
        
        genotype.append([
            (PRIMITIVES[best_op], prev_idx) 
            for _, prev_idx, best_op in selected
        ])
    
    return genotype
```

**DARTS的局限与改进**

原始DARTS存在一些问题：
1. **跳跃连接崩塌**：倾向于选择跳跃连接，导致性能退化
2. **搜索-评估差距**：搜索时用浅网络，评估时用深网络，存在差异

后续改进包括：
- **DARTS+**：引入早停机制
- **P-DARTS**：渐进式增加搜索深度
- **PC-DARTS**：部分通道连接，减少内存

---

## 34.5 效率优化技术

### 34.5.1 权重共享与超网络

**生活比喻：通用零件系统**

想象汽车制造：如果每款车都要重新设计所有零件，成本将高得惊人。现代汽车工业使用通用零件系统——多个车型共享相同的发动机、底盘、电子系统（权重共享），只需要组合不同的零件就能生产出不同定位的车型（不同架构）。

**超网络的形式化定义**

超网络$\mathcal{N}_{super}$包含所有候选架构的参数。对于搜索空间$\mathcal{A}$中的任意架构$\alpha$，存在参数选择函数：

$$
\theta_\alpha = \text{Select}(\theta_{super}, \alpha)
$$

架构$\alpha$的前向传播：

$$
y = f(x; \theta_\alpha) = f(x; \text{Select}(\theta_{super}, \alpha))
$$

**权重共享的训练**

1. **单路径采样**：每次迭代采样一个架构，只更新其对应参数
   
2. **多路径梯度累积**：采样多个架构，累积梯度后统一更新

3. **前向-反向分离**：前向传播时使用采样架构，反向传播时考虑所有可能路径

**Once-for-All：超网络的极致**

Once-for-All(OFA)将超网络思想推向极致：

1. **弹性维度**：网络宽度（通道数）、深度（层数）、卷积核大小、输入分辨率都可变
2. **渐进式收缩训练**：从最大架构开始，逐步引入子架构训练
3. **特殊化部署**：训练完成后，无需重新训练即可导出任意大小的子网络

OFA的超网络包含约$10^{19}$个子网络，覆盖从边缘设备到云服务器的各种部署场景。

### 34.5.2 性能预测器

**生活比喻：房产估价**

评估每个候选架构像评估一栋房子——完整翻新再看效果显然不现实。房产估价师通过学习大量成交案例，掌握了一套快速估价方法：根据面积、地段、房龄等特征直接预测价格。性能预测器就是架构的"估价师"。

**预测器设计**

性能预测器$\hat{\mathcal{P}}$是一个回归模型：

$$
\hat{\mathcal{P}}: \mathcal{A} \rightarrow \mathbb{R}
$$

输入是架构编码，输出是预测性能。

**架构编码方式**：

1. **序列编码**：将架构描述为字符串或序列
2. **图编码**：将架构视为图，使用图神经网络编码
3. **路径编码**：编码从输入到输出的路径信息

**预测器训练**：

收集少量$(\alpha_i, \mathcal{P}_i)$样本对，训练预测器：

$$
\min_\phi \frac{1}{N} \sum_{i=1}^{N} (\hat{\mathcal{P}}(\alpha_i; \phi) - \mathcal{P}_i)^2
$$

**神经预测器**

使用神经网络作为预测器，常见架构：
- **LSTM**：处理序列化架构描述
- **GNN**：处理图结构
- **MLP**：处理手工设计的特征（如FLOPs、参数量）

### 34.5.3 Early Stopping

**生活比喻：考试中途交卷**

想象一场数学竞赛：你发现某道题的思路有问题，越早意识到，越能及时调整。Early stopping就是在训练过程中及早发现"没希望"的架构，停止训练，节省计算资源。

**早停策略**

1. **基于学习曲线**：如果前k个epoch的验证损失没有明显下降，提前终止

2. **基于预测器**：用轻量级预测器快速评估，只训练有潜力的架构

3. **基于资源分配**：为每个架构分配初始预算，表现好的增加预算(Hyperband)

**Learning Curve Prediction**

训练初期就预测最终性能：

$$
\mathcal{P}_\infty \approx f(\mathcal{P}_1, \mathcal{P}_2, ..., \mathcal{P}_k; \psi)
$$

其中$\mathcal{P}_t$是第t个epoch的性能，$f$是学习曲线预测模型。

---

## 34.6 Training-free NAS

### 34.6.1 零成本代理概述

**生活比喻：考试零分预测**

想象你在监考一场数学考试。学生刚拿到试卷（网络初始化），还没开始答题（训练），你能预测谁能考高分吗？

Training-free NAS的神奇之处就在于：它发现了一些"蛛丝马迹"——在训练开始前，网络的某些特性就能预示其最终性能。这就像观察学生的答题姿势、读题速度，就能大致判断水平高低。

**为什么Training-free可行？**

研究表明，神经网络在初始化时就已经蕴含了其表达能力的重要信息：

1. **梯度流动**：梯度在网络中的流动特性决定训练效率
2. **激活模式**：前向传播的激活分布反映网络容量
3. **参数结构**：权重矩阵的结构特性影响表达能力

**零成本代理分类**

1. **基于梯度**：SNIP、GraSP、SynFlow
2. **基于激活**：NASWOT、Zen-NAS
3. **基于理论**：NTK条件数、线性区域数
4. **混合方法**：TE-NAS、HNAS

### 34.6.2 NASWOT原理

NASWOT(Neural Architecture Search Without Training)由Mellor等人于2021年提出，是第一个证明无需训练即可有效排序架构的工作。

**核心洞察**

NASWOT基于一个观察：好的架构应该能在训练前就区分不同的输入。

具体来说，给定一个mini-batch的输入$X = \{x_1, x_2, ..., x_N\}$，通过随机初始化的网络，得到每层的激活：

$$
Z^{(l)} = f^{(l)}(X) \in \mathbb{R}^{N \times C_l}
$$

**二值化激活**

将ReLU后的激活二值化（正为1，负为0）：

$$
B^{(l)} = \mathbb{1}_{Z^{(l)} > 0} \in \{0, 1\}^{N \times C_l}
$$

**汉明距离度量**

计算不同输入产生的二值编码之间的汉明距离：

$$
H_{ij} = \frac{1}{C} \sum_{c=1}^{C} \mathbb{1}_{B_{ic} \neq B_{jc}}
$$

汉明距离越大，表示网络对这两个输入的区分能力越强。

**NASWOT分数**

定义样本间的核矩阵：

$$
K_{ij} = \exp(-\gamma \cdot H_{ij})
$$

NASWOT分数是该核矩阵的对数行列式：

$$
\text{NASWOT}(\mathcal{A}) = \log\det(K + \epsilon I)
$$

行列式反映了样本间的区分度：值越大，网络区分能力越强，预期性能越好。

**计算复杂度**

- 一次前向传播：$O(N \cdot F)$，N是样本数，F是FLOPs
- 核矩阵计算：$O(N^2 \cdot L)$，L是层数
- 总复杂度：与完整训练相比降低数千倍

### 34.6.3 SynFlow与SNIP

**生活比喻：管道通畅度检查**

想象你要检查一栋大楼的排水系统：
- **SNIP**：倒一盆水看流速（需要数据）
- **SynFlow**：检查管道直径和连接方式（无需数据）

两者都能评估系统的"通畅度"（梯度流动），但SynFlow更"省事"。

**SNIP: Single-shot Network Pruning**

SNIP最初用于初始化时的网络剪枝，后来被用作零成本代理。

**核心思想**：评估每个参数对损失函数的影响。

定义参数的显著性(saliency)：

$$
S(\theta) = \left| \frac{\partial \mathcal{L}}{\partial \theta} \odot \theta \right|
$$

其中$\odot$是逐元素乘积。

SNIP分数是整个网络的显著性之和：

$$
\text{SNIP}(\mathcal{A}) = \sum_{\theta \in \mathcal{A}} S(\theta) = \sum_{\theta \in \mathcal{A}} \left| \frac{\partial \mathcal{L}}{\partial \theta} \cdot \theta \right|
$$

**直观解释**：$\frac{\partial \mathcal{L}}{\partial \theta} \cdot \theta$近似了移除该参数对损失的影响（一阶泰勒展开）。

**SynFlow: Synaptic Flow**

SynFlow改进了SNIP，使其完全不依赖数据。

**核心思想**：使用一个虚拟损失函数，它是所有参数绝对值的乘积：

$$
\mathcal{R}(\theta) = \mathbf{1}^T \left( \prod_{l=1}^{L} |\theta^{(l)}| \right) \mathbf{1}
$$

这个损失函数衡量了从输入到输出的"信号流"强度。

**前向传播**：输入全1向量
$$
x_0 = \mathbf{1}
$$

**反向传播**：计算虚拟损失的梯度
$$
\frac{\partial \mathcal{R}}{\partial \theta^{(l)}}
$$

**显著性计算**：与SNIP类似
$$
\text{SynFlow}(\theta) = \frac{\partial \mathcal{R}}{\partial \theta} \odot \theta
$$

**SynFlow分数**：
$$
\text{SynFlow}(\mathcal{A}) = \sum_{\theta \in \mathcal{A}} \left| \frac{\partial \mathcal{R}}{\partial \theta} \cdot \theta \right|
$$

**数据无关的优势**

1. **无需标注数据**：降低数据准备成本
2. **避免标签错误**：标注错误不会影响评估
3. **计算更快**：不需要数据加载和预处理

**数学对比**

| 代理 | 公式 | 需要数据 | 需要梯度 |
|-----|------|---------|---------|
| SNIP | $\sum |\frac{\partial \mathcal{L}}{\partial \theta} \cdot \theta|$ | ✅ | ✅ |
| GraSP | $\sum |(H \frac{\partial \mathcal{L}}{\partial \theta}) \cdot \theta|$ | ✅ | ✅ |
| SynFlow | $\sum |\frac{\partial \mathcal{R}}{\partial \theta} \cdot \theta|$ | ❌ | ✅ |
| NASWOT | $\log\det(K)$ | ✅ | ❌ |

---

## 34.7 硬件感知NAS与前沿

### 34.7.1 MNasNet与延迟约束

**生活比喻：量身定制的西装**

一套高级定制西装要考虑穿着者的身材、场合、活动需求。同样，神经网络也要"量身定制"——根据目标硬件的计算能力、内存限制、功耗预算来设计。MNasNet就是这方面的开创者。

**多目标优化**

MNasNet将NAS扩展为多目标优化问题：

$$
\max_{\alpha} \quad \text{ACC}(\alpha) \times \left[ \frac{\text{LAT}(\alpha)}{T} \right]^w
$$

其中：
- $\text{ACC}(\alpha)$：架构$\alpha$的准确率
- $\text{LAT}(\alpha)$：在目标设备上的实际推理延迟
- $T$：目标延迟
- $w$：权重因子（通常$w=-0.07$），控制延迟惩罚强度

**延迟建模**

不同于使用FLOPs作为代理，MNasNet直接在实际移动设备上测量延迟：

1. 将候选架构转换为TensorFlow Lite格式
2. 部署到Pixel手机
3. 运行100次取平均延迟

**搜索空间设计**

MNasNet的搜索空间包含：
- 卷积核大小：{3, 5, 7}
- 扩展比例：{3, 6}
- 通道数：根据输入分辨率动态调整
- SE模块：是否使用Squeeze-and-Excitation
- 激活函数：ReLU或swish

**分层搜索**

网络分为多个阶段(stage)，每个阶段独立搜索：

$$
\alpha = (\alpha_1, \alpha_2, ..., \alpha_S)
$$

每个阶段$\alpha_s$有自己的搜索空间，适应不同的特征分辨率。

### 34.7.2 ProxylessNAS

**生活比喻：量体裁衣 vs 买成衣改**

之前的NAS方法就像：在小号人台上设计衣服，然后放大到真人尺寸（在小数据集/浅网络上搜索，迁移到大数据集/深网络）。

ProxylessNAS的颠覆性在于：直接在目标身材上量体裁衣——在大规模数据集和目标硬件上直接搜索。

**路径二值化**

ProxylessNAS的核心创新是路径级二值化，解决超网络的内存爆炸问题。

在传统的超网络中，每条边包含所有候选操作：

$$
m^{(i,j)} = \sum_{k=1}^{K} g_k^{(i,j)} \cdot o_k(x)
$$

这需要在GPU上同时存储所有操作的输出，内存消耗巨大。

ProxylessNAS将门控$g$二值化（每次只选一个操作）：

$$
g_k^{(i,j)} = \text{Binary}(\text{softmax}(\alpha_k^{(i,j)}))
$$

**可微分采样**

使用Gumbel-Softmax技巧实现可微分的离散采样：

$$
g_k = \frac{\exp((\log \pi_k + G_k) / \tau)}{\sum_{k'} \exp((\log \pi_{k'} + G_{k'}) / \tau)}
$$

其中$G_k \sim \text{Gumbel}(0,1)$是Gumbel噪声，$\tau$是温度参数。

**硬件感知损失**

ProxylessNAS将硬件延迟直接纳入损失函数：

$$
\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{latency}
$$

延迟损失使用查找表近似：

$$
\mathcal{L}_{latency} = \sum_{(i,j)} \sum_{k} g_k^{(i,j)} \cdot \text{LAT}_k^{(i,j)}
$$

其中$\text{LAT}_k^{(i,j)}$是操作$o_k$在边$(i,j)$的预测量延迟。

**直接搜索的优势**

1. **无代理差距**：搜索和评估使用相同设置
2. **硬件定制**：为特定硬件优化架构
3. **更大搜索空间**：不受限于可迁移性

实验表明，ProxylessNAS在ImageNet上直接搜索，仅用200 GPU小时就达到75.1% top-1准确率，比MobileNetV2快1.2倍。

### 34.7.3 大模型时代的NAS

随着GPT、PaLM等大语言模型的兴起，NAS也面临新的挑战和机遇。

**大模型架构搜索**

传统NAS专注于CNN，大模型时代的NAS关注：
- **Transformer架构变体**：搜索注意力模式、FFN结构
- **混合架构**：CNN与Transformer的结合
- **稀疏架构**：MoE(Mixture of Experts)、稀疏注意力

**代表性工作**

1. **Evolved Transformer**：使用进化算法搜索Transformer变体，在机器翻译上超越原版

2. **Autoformer**：专门搜索视觉Transformer的架构，发现局部注意力比全局注意力更高效

3. **Swin Transformer Search**：搜索分层视觉Transformer的最佳窗口大小和深度配置

**效率挑战**

大模型NAS面临的主要挑战：

1. **计算成本**：训练一个大模型需要数千GPU天，完整搜索不可行
2. **评估困难**：大模型需要大量数据和长时间训练才能收敛
3. **泛化问题**：在小规模任务上搜索的架构能否泛化到大规模？

**解决方向**

1. **基于Scaling Law的预测**：利用规模法则外推大模型性能
2. **零成本代理**：Training-free方法在大模型上的应用
3. **渐进式搜索**：从small-scale开始，逐步扩大

**未来展望**

NAS的未来发展方向：

1. **多模态NAS**：同时搜索视觉-语言架构
2. **神经符号架构**：结合神经网络和符号推理
3. **自动发现学习算法**：AutoML-Zero的延伸，从零开始发现算法
4. **绿色AI**：考虑碳排放的NAS，追求环境友好

---

## 34.8 练习题

### 基础题

**34.1** NAS的三大组件是什么？请用你自己的话解释每个组件的作用。

**34.2** Cell-based搜索空间相比全局搜索空间有什么优势？

**34.3** 解释为什么ENAS的权重共享机制能大幅降低搜索成本。

### 进阶题

**34.4** DARTS的核心创新是什么？解释连续松弛如何将离散优化转化为连续优化。

**34.5** 推导DARTS的架构参数梯度公式，说明为什么需要近似。

**34.6** 比较SNIP和SynFlow两种零成本代理：
- 它们的计算公式分别是什么？
- 各自的优势和局限是什么？
- 为什么SynFlow不需要数据？

### 挑战题

**34.7** **设计一个硬件感知的NAS搜索空间**

假设你要为边缘设备（如智能摄像头）设计一个目标检测网络的NAS搜索空间：
- 约束：延迟 < 50ms，内存 < 100MB
- 请定义候选操作集合
- 设计延迟预测模型
- 编写搜索算法伪代码

**34.8** **实现一个简化的DARTS**

基于本章的代码框架，在CIFAR-10上实现一个简化版DARTS：
- 搜索空间：3个中间节点的Cell
- 候选操作：3×3 conv, 5×5 conv, 3×3 max pool, skip connection
- 训练10个epoch，导出最佳架构
- 可视化搜索到的Cell结构

**34.9** **理论分析**

证明SynFlow分数在参数重标度下的不变性：
如果将某层的所有参数乘以常数$c$，SynFlow分数如何变化？
这个性质对架构搜索有什么意义？

---

## 参考文献

Baker, B., Gupta, O., Naik, N., & Raskar, R. (2016). Designing neural network architectures using reinforcement learning. *arXiv preprint arXiv:1611.02167*.

Cai, H., Gan, C., & Han, S. (2019). ProxylessNAS: Direct neural architecture search on target task and hardware. *International Conference on Learning Representations (ICLR)*.

Cai, H., Gan, C., Wang, T., Zhang, Z., & Han, S. (2020). Once-for-all: Train one network and specialize it for efficient deployment. *International Conference on Learning Representations (ICLR)*.

Elsken, T., Metzen, J. H., & Hutter, F. (2019). Neural architecture search: A survey. *The Journal of Machine Learning Research*, 20(1), 1997-2017.

Lee, N., Ajanthan, T., & Torr, P. (2019). Snip: Single-shot network pruning based on connection sensitivity. *International Conference on Learning Representations (ICLR)*.

Liu, H., Simonyan, K., & Yang, Y. (2019). DARTS: Differentiable architecture search. *International Conference on Learning Representations (ICLR)*.

Mellor, J., Turner, J., Storkey, A., & Crowley, E. J. (2021). Neural architecture search without training. *International Conference on Machine Learning (ICML)*, 7588-7598.

Pham, H., Guan, M. Y., Zoph, B., Le, Q. V., & Dean, J. (2018). Efficient neural architecture search via parameter sharing. *International Conference on Machine Learning (ICML)*, 4095-4104.

Real, E., Aggarwal, A., Huang, Y., & Le, Q. V. (2019). Regularized evolution for image classifier architecture search. *Proceedings of the AAAI Conference on Artificial Intelligence*, 33(1), 4780-4789.

Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *International Conference on Machine Learning (ICML)*, 6105-6114.

Tan, M., Chen, B., Pang, R., Vasudevan, V., Sandler, M., Howard, A., & Le, Q. V. (2019). Mnasnet: Platform-aware neural architecture search for mobile. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2820-2828.

Tanaka, H., Kunin, D., Yamins, D. L., & Ganguli, S. (2020). Pruning neural networks without any data by iteratively conserving synaptic flow. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 6377-6389.

Zoph, B., & Le, Q. V. (2017). Neural architecture search with reinforcement learning. *International Conference on Learning Representations (ICLR)*.

Zoph, B., Vasudevan, V., Shlens, J., & Le, Q. V. (2018). Learning transferable architectures for scalable image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 8697-8710.

---

## 本章小结

本章带你深入了解了神经架构搜索(NAS)的完整图景：

**核心概念**：
- NAS的三大组件：搜索空间、搜索策略、性能评估
- Cell-based搜索空间的设计原理
- 权重共享、超网络等效率优化技术

**搜索范式**：
- 强化学习：RNN控制器 + 策略梯度
- 进化算法：模拟自然选择的优胜劣汰
- 可微分搜索：DARTS的连续松弛和双层优化

**前沿方向**：
- Training-free NAS：零成本代理快速评估
- 硬件感知NAS：为特定设备量身定制
- 大模型时代的架构搜索

**代码实现**：
- SearchSpace、NASNetController、ENASSearcher
- DARTSSearcher、MixedOperation
- ZeroCostProxy（NASWOT、SNIP、SynFlow）
- NASVisualizer搜索可视化

神经架构搜索代表了机器学习自动化的重要方向——让机器自己设计学习机器。从最初需要数千GPU天的强化学习，到如今数秒即可评估的零成本代理，NAS的效率提升了数万倍。这不仅让普通研究者能够探索创新的架构，也为AI民主化铺平了道路。随着大模型时代的到来，NAS将继续演进，为发现下一代神经网络架构贡献力量。
