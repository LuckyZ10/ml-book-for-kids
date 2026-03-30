# 第三十四章：神经架构搜索（NAS）——让AI自己设计AI

## 34.1 NAS概述与历史演进：从星星之火到燎原之势

### 一个疯狂的想法

2016年的一个深夜，Google Brain的研究员Barret Zoph盯着电脑屏幕上手工设计的Inception网络架构图，脑海中闪过一个大胆的念头：**既然神经网络可以学会识别猫狗、翻译语言、下棋打败世界冠军，为什么不能让它学会设计神经网络自己呢？**

这个想法听起来像是科幻小说中的情节——AI创造AI，仿佛是技术奇点的先兆。但Zoph和他的导师Quoc V. Le决定把这个疯狂的想法付诸实践。他们没有预料到，这个名为"神经架构搜索"（Neural Architecture Search, NAS）的项目，将在未来几年引发深度学习领域的一场革命。

> 💡 **费曼时间：建筑师与进化**
> 
> 想象一下，你是一位建筑师，要设计一座摩天大楼。传统方式是：你手工绘制每一层的蓝图，决定哪里放电梯、哪里是办公区、哪里是健身房。这需要多年的经验和直觉。
> 
> 但如果我们换一种方式：我们创建一个"建筑进化模拟器"，让计算机生成成千上万种不同的设计方案，每种方案都在虚拟环境中经受地震、台风、使用效率等考验。表现好的设计"繁殖"下一代，表现差的被淘汰。经过数百代进化， emerges 的设计可能比任何人类建筑师设计的都要优秀！
> 
> NAS就是这样一个"数字进化"过程，只不过进化的不是建筑物，而是神经网络的架构。

### NAS的三驾马车

NAS的核心可以概括为三个基本要素，就像一个寻宝游戏需要定义的三样东西：

**1. 搜索空间（Search Space）——宝藏地图**

搜索空间定义了我们可以在哪里寻找架构。它就像一张宝藏地图，标注了所有可能的路径。搜索空间可以是：

- **宏观搜索（Macro Search）**：直接搜索整个网络结构，每一层都可以是不同的操作
- **单元搜索（Cell-based Search）**：先搜索小的"建筑单元"（Cell），然后像搭积木一样堆叠这些单元

数学上，搜索空间 $\mathcal{A}$ 是所有可能架构的集合：

$$\mathcal{A} = \{ \alpha \mid \alpha \text{ 是一个有效的神经网络架构} \}$$

每个架构 $\alpha$ 可以表示为计算图，其中节点表示特征图，边表示操作（卷积、池化、跳跃连接等）。

**2. 搜索策略（Search Strategy）——寻宝方法**

有了地图，我们需要决定如何搜索。主要策略包括：

- **强化学习（Reinforcement Learning）**：训练一个"控制器"网络来生成架构
- **进化算法（Evolutionary Algorithms）**：模拟自然选择，让架构种群进化
- **梯度优化（Gradient-based Optimization）**：将离散的架构选择松弛为连续的优化问题

**3. 性能评估（Performance Estimation）——宝藏价值**

我们需要快速评估每个找到的网络有多好：

- **从头训练（Train from Scratch）**：最准确但最慢的方法
- **权重共享（Weight Sharing）**：多个架构共享参数，加速评估
- **超网（Supernet）**：训练一个包含所有子架构的大网络

### 历史里程碑：从2000 GPU天到几小时

**2016-2017：开山之作（Zoph & Le, 2016）**

Zoph和Le的原始NAS论文使用强化学习控制器，在CIFAR-10上搜索架构。这个方法需要**800个GPU训练28天**（约22,000 GPU小时）才能找到好的架构。虽然计算成本惊人，但发现的最优架构已经能与人工设计的网络竞争。

**2018：效率革命元年**

- **NASNet（Zoph et al., 2018）**：引入单元搜索空间，先在CIFAR-10上搜索，再迁移到ImageNet。这是第一个在ImageNet上超越人工设计的NAS架构。
  
- **ENAS（Pham et al., 2018）**：引入**参数共享（Parameter Sharing）**，让不同架构共享权重，将搜索成本从22,000 GPU小时降低到约**10 GPU小时**，实现了**1000倍加速**！这是NAS发展史上的关键转折点。

**2019：可微架构搜索时代**

- **DARTS（Liu et al., 2019）**：将离散的架构选择松弛为连续的softmax权重，使用梯度下降同时优化架构参数和网络权重。搜索成本降至**4 GPU天**。

- **ProxylessNAS（Cai et al., 2019）**：直接在目标硬件上进行架构搜索，引入硬件延迟作为优化目标，实现真正的硬件感知设计。

- **FBNet（Wu et al., 2019）**：使用可微分NAS优化移动设备上的高效网络，展示了NAS在实际应用中的价值。

**2020：训练一次，处处部署**

- **Once-for-All（Cai et al., 2020）**：训练一个包含所有子网络的超网，之后可以通过简单的"选择"操作得到不同大小的专用网络，无需重新训练。

- **BigNAS（Yu et al., 2020）**：扩展OFA思想，通过渐进式收缩训练超大搜索空间中的网络。

- **SPOS（Guo et al., 2020）**：单路径单次NAS，使用均匀采样训练超网，进一步简化了权重共享方法。

### NAS的意义：为什么这很重要？

**1. 超越人类设计**

NAS发现的架构往往包含人类设计师未曾考虑过的模式。例如，NASNet发现的某些连接模式类似于Inception和ResNet的结合，但又具有独特性。

**2. 自动化机器学习（AutoML）的核心**

NAS代表了机器学习中"学习什么学习"（Learning to Learn）的终极形式。如果AI能够设计自己的大脑，它就可能以我们无法预料的方式进化。

**3.  democratization of AI**

NAS让没有深厚神经网络设计经验的研究者和工程师也能获得高性能的定制架构。

**4. 硬件-软件协同设计**

硬件感知NAS（如ProxylessNAS、FBNet）能够针对特定设备（手机、边缘设备、ASIC）优化网络，这是手工设计难以做到的。

> 🎯 **思考时刻**
> 
> 想象一下，如果AlphaGo不仅能下棋，还能设计自己的神经网络架构来下棋，会发生什么？这听起来像递归自我改进的AI——正是一些关于人工通用智能（AGI）的推测性想法的核心。

---

## 34.2 搜索空间设计：Cell-based vs Macro Search

### 搜索空间的数学表示

搜索空间是NAS的基础，它定义了"我们可以建造什么"。数学上，一个搜索空间由以下要素组成：

**操作集合（Operation Set）**：

$$\mathcal{O} = \{ o_1, o_2, ..., o_m \}$$

常见的操作包括：
- 卷积：$3\times3$ 卷积、$5\times5$ 卷积、$7\times7$ 卷积
- 深度可分离卷积：$3\times3$ depthwise separable conv
- 池化：$3\times3$ average pooling、$3\times3$ max pooling
- 跳跃连接（Skip Connection）：Identity mapping
- 空操作（Zero）：表示没有连接

**计算图表示**：

一个神经网络架构 $\alpha$ 可以表示为有向无环图（DAG）：

$$\alpha = (V, E)$$

其中：
- $V = \{v_1, v_2, ..., v_n\}$ 是节点集合，每个节点代表一个特征图
- $E \subseteq V \times V \times \mathcal{O}$ 是边集合，每条边 $(u, v, o)$ 表示对节点 $u$ 应用操作 $o$ 后连接到节点 $v$

### Macro Search：直接搜索完整网络

在宏观搜索中，控制器直接生成整个网络的描述。对于包含 $L$ 层的网络，每层 $l$ 需要决定：

- 操作类型：$o_l \in \mathcal{O}$
- 滤波器数量：$f_l \in \{16, 32, 64, 128, 256, ...\}$
- 核大小：$k_l \in \{1, 3, 5, 7\}$
- 步长：$s_l \in \{1, 2\}$（决定是否下采样）

搜索空间大小约为：

$$|\mathcal{A}| \approx (|\mathcal{O}| \times |\mathcal{F}| \times |\mathcal{K}| \times |\mathcal{S}|)^L$$

对于 $L=20$ 层的网络，即使每个选择只有10种可能，搜索空间也有 $10^{20}$ 个架构！这比宇宙中的原子数量还多。

> 💡 **费曼时间：乐高积木 vs 3D打印**
> 
> Macro search就像是使用3D打印机直接打印整个建筑——你可以控制每一层的每一个细节，但可能性是无限的，找到好设计就像大海捞针。

### Cell-based Search：搜索可重复单元

NASNet的开创性贡献是引入了**单元（Cell）**的概念。 observation：成功的神经网络（如ResNet、Inception、DenseNet）都使用重复的基本模块。

**两种单元类型**：

1. **普通单元（Normal Cell）**：保持特征图空间维度不变（stride=1）
2. **约简单元（Reduction Cell）**：将特征图高度和宽度减半，通道数加倍（stride=2）

**搜索空间的结构**：

```
输入 → [Normal Cell] × N → [Reduction Cell] → [Normal Cell] × N → [Reduction Cell] → [Normal Cell] × N → 输出
```

每个单元内部包含 $B$ 个块（block），每个块需要选择：
- 两个输入隐藏状态（来自前序节点或单元输入）
- 两个操作（应用于选中的隐藏状态）
- 一个组合操作（通常是相加或拼接）

NASNet中，每个块需要5个决策：
1. 选择第一个隐藏状态（来自前序）
2. 选择第一个操作
3. 选择第二个隐藏状态
4. 选择第二个操作
5. 选择组合方式（add或concatenate）

**单元搜索的数学表示**：

设单元有 $B$ 个块，第 $b$ 个块的决策为 $(h_1^{(b)}, o_1^{(b)}, h_2^{(b)}, o_2^{(b)}, c^{(b)})$。

单元的输出是所有未被用作输入的隐藏状态的拼接：

$$\text{cell\_output} = \text{concat}(\{h_j \mid h_j \text{ 未被选为输入}\})$$

**搜索空间大小的比较**：

假设：
- 7种操作选择
- 最多7个前序隐藏状态可选
- 2种组合方式
- 每个单元5个块

每个块的决策数：$7 \times 7 \times 7 \times 7 \times 2 = 4,802$

整个单元的搜索空间：$4,802^5 \approx 2.6 \times 10^{18}$

虽然仍然巨大，但比Macro search的 $10^{20}$ 以上小得多，而且迁移性更好。

### 搜索空间设计的关键洞察

**1. 人类知识的重要性**

虽然NAS是自动化的，但搜索空间的设计仍然需要人类直觉。例如：
- 为什么使用卷积而不是全连接？
- 为什么考虑深度可分离卷积？
- 为什么限制核大小为1, 3, 5, 7？

这些选择基于我们对视觉任务的理解。

**2. 搜索空间与优化难度的权衡**

- 搜索空间越大，可能找到更好的架构，但搜索越困难
- 搜索空间越小，搜索越快，但可能错过最优解

**3. 可迁移性**

Cell-based设计的一个重要优势是：在小数据集（如CIFAR-10）上搜索到的单元，可以直接迁移到大数据集（如ImageNet），只需调整堆叠的单元数量和通道数。

> 🎯 **代码热身：定义搜索空间**
> 
> 在我们后面的代码实现中，你会看到如何用Python定义这些搜索空间，以及如何将架构表示为可计算的数据结构。

---

## 34.3 搜索策略：强化学习、进化算法与梯度优化

### 强化学习NAS：让控制器学会设计

Zoph & Le（2016）的原始NAS使用强化学习来训练一个**控制器RNN**（Recurrent Neural Network）。这个控制器的工作就像一个建筑师，逐个生成网络架构的"设计图纸"。

**控制器的决策过程**：

控制器是一个循环神经网络（通常是LSTM），它逐个生成架构的超参数。对于每一层，控制器预测：

1. 滤波器高度：$h \in \{1, 3, 5, 7\}$
2. 滤波器宽度：$w \in \{1, 3, 5, 7\}$
3. 滤波器数量：$f \in \{24, 36, 48, 64\}$
4. 步长：$s \in \{1, 2\}$
5. 跳跃连接：连接到哪些前序层

**策略梯度训练**：

设控制器参数为 $\theta$，它定义了生成架构的概率分布 $P(\alpha; \theta)$。生成架构 $\alpha$ 后，我们在训练集上训练这个子网络，得到验证集准确率 $R(\alpha)$ 作为奖励。

我们的目标是最大化期望奖励：

$$J(\theta) = \mathbb{E}_{\alpha \sim P(\cdot; \theta)}[R(\alpha)]$$

使用REINFORCE算法（也称为策略梯度），梯度为：

$$\nabla_\theta J(\theta) = \sum_{\alpha} P(\alpha; \theta) \nabla_\theta \log P(\alpha; \theta) R(\alpha)$$

实际中，我们采样 $m$ 个架构来近似：

$$\nabla_\theta J(\theta) \approx \frac{1}{m} \sum_{i=1}^{m} \nabla_\theta \log P(\alpha_i; \theta) R(\alpha_i)$$

**减少方差的技巧**：

直接使用上述梯度估计方差很大。一个常用技巧是使用**基线（baseline）**：

$$\nabla_\theta J(\theta) \approx \frac{1}{m} \sum_{i=1}^{m} \nabla_\theta \log P(\alpha_i; \theta) (R(\alpha_i) - b)$$

其中 $b$ 是之前架构奖励的指数移动平均。这减少了梯度的方差，加速收敛。

**控制器RNN的架构**：

```
输入：[START_TOKEN]
    ↓
LSTM → 预测第1层参数 → Embedding
    ↓
LSTM → 预测第2层参数 → Embedding
    ↓
    ...
    ↓
LSTM → 预测第L层参数
```

每个预测都是一个softmax分类，输出在对应超参数空间上的概率分布。

> 💡 **费曼时间：训练一个艺术家**
> 
> 想象你在训练一位抽象艺术家画肖像。一开始，艺术家随机涂鸦。你（作为评判者）给每幅画打分（奖励）。
> 
> 艺术家记住："当我画圆眼睛时得分高，画方眼睛时得分低"。渐渐地，艺术家学会画更像肖像的作品。
> 
> 控制器RNN就是这样一位"架构艺术家"，它学会画出高性能的网络设计。

### 进化算法：适者生存

进化算法（Evolutionary Algorithms, EA）模拟自然选择过程来搜索架构。AmoebaNet（Real et al., 2019）是这一方法的代表作。

**基本流程**：

1. **初始化**：随机生成 $P$ 个架构作为初始种群
2. **评估**：训练每个架构，得到适应度（如验证准确率）
3. **选择**：选择适应度高的架构作为"父母"
4. **变异/重组**：对父母进行小的修改，生成"后代"
5. **替换**：用后代替换适应度低的个体
6. **重复** 2-5 步直到收敛

**变异操作**：

- **隐藏状态变异**：改变块内的某个连接
- **操作变异**：将某个操作换成另一个（如 $3\times3$ conv → $5\times5$ conv）
- **插入**：在随机位置插入一个新层
- **删除**：随机删除一层

**正则化进化（Regularized Evolution）**：

Real et al. 发现传统进化算法有"早熟收敛"问题——种群多样性快速丧失。他们提出：

> 强制移除最老的个体，而不是最差的个体。

这保持了种群多样性，让搜索持续探索新的区域。

**进化 vs 强化学习**：

| 特性 | 强化学习 | 进化算法 |
|------|----------|----------|
| 学习方式 | 基于梯度优化 | 基于选择压力 |
| 需要可微 | 是 | 否 |
| 并行性 | 中等 | 高（自然并行）|
| 全局探索 | 依赖随机初始化 | 种群天然多样化 |
| 历史信息 | 保存在RNN隐状态 | 保存在种群基因中 |

### 梯度优化：DARTS的革命

DARTS（Differentiable ARchiTecture Search）是NAS领域最重要的突破之一。核心思想：

> **将离散的架构选择松弛为连续的架构权重，使NAS可微分！**

**连续松弛（Continuous Relaxation）**：

在Cell-based搜索空间中，每个边 $(i, j)$ 需要从操作集合 $\mathcal{O}$ 中选择一个操作。传统NAS是离散的：

$$o^{(i,j)} = \arg\max_{o \in \mathcal{O}} \text{某个得分}$$

DARTS将其松弛为所有操作的softmax加权：

$$\bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o' \in \mathcal{O}} \exp(\alpha_{o'}^{(i,j)})} \cdot o(x)$$

其中 $\alpha_o^{(i,j)}$ 是架构参数（architecture parameters），是可学习的！

**双层优化（Bilevel Optimization）**：

DARTS需要同时优化两组参数：

1. **网络权重** $w$：最小化训练损失
2. **架构参数** $\alpha$：最小化验证损失

这形成了一个双层优化问题：

$$\min_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha)$$

$$\text{s.t.} \quad w^*(\alpha) = \arg\min_w \mathcal{L}_{train}(w, \alpha)$$

**近似求解**：

精确求解内层优化 $w^*(\alpha)$ 需要训练网络到收敛，太昂贵。DARTS使用一步梯度下降近似：

$$w' = w - \xi \nabla_w \mathcal{L}_{train}(w, \alpha)$$

然后计算验证损失的梯度：

$$\nabla_\alpha \mathcal{L}_{val}(w', \alpha)$$

**交替优化算法**：

```python
for iteration in range(num_iterations):
    # 步骤1：固定架构，更新网络权重（在训练集上）
    for step in range(num_weight_steps):
        w = w - lr_w * grad(L_train(w, alpha))
    
    # 步骤2：固定网络权重，更新架构参数（在验证集上）
    alpha = alpha - lr_alpha * grad(L_val(w, alpha))
```

**离散化（Discretization）**：

搜索完成后，我们需要得到离散架构。对于每条边，选择架构参数最大的操作：

$$o^{*(i,j)} = \arg\max_{o \in \mathcal{O}} \alpha_o^{(i,j)}$$

> 💡 **费曼时间：调酒师的艺术**
> 
> 想象NAS是在调鸡尾酒。传统方法是：你有一排基酒（操作），必须选一种倒入杯子（离散选择）。
> 
> DARTS说：为什么不先都倒一点，尝尝混合的味道，然后逐渐调整比例？
> 
> 架构参数 $\alpha$ 就像是每种酒的"配方比例"。我们通过梯度下降学习最佳配方，最后只保留比例最高的酒。

---

## 34.4 性能评估：从数千GPU小时到几分钟

### 从头训练：黄金标准

最准确的性能评估方法是：

1. 从搜索空间采样一个架构 $\alpha$
2. 在训练集上从头训练到收敛
3. 在验证集上评估准确率

这就是原始NAS的做法，也是为什么需要22,000 GPU小时。

数学上，我们需要：

$$R(\alpha) = \text{Accuracy}(\text{TrainToConvergence}(\alpha, \mathcal{D}_{train}), \mathcal{D}_{val})$$

**优缺点**：
- ✅ 最准确的性能估计
- ❌ 每个架构需要数小时到数天
- ❌ 搜索1000个架构需要数年GPU时间

### 权重共享：ENAS的洞察

ENAS（Efficient Neural Architecture Search）的核心洞察是：

> **搜索空间中的所有架构都是一个大图（超网）的子图，可以共享参数！**

**超网（Supernet）构建**：

构建一个有向无环图 $\mathcal{G}$，其中：
- 节点表示计算操作
- 边表示可能的连接
- 每个节点的参数在所有使用该节点的子架构间共享

**参数共享的数学表示**：

设超网参数为 $W$。对于子架构 $\alpha$，其参数 $W_\alpha$ 是 $W$ 的子集：

$$W_\alpha \subseteq W$$

训练时，我们采样一个子架构，在前向传播中只使用 $W_\alpha$，然后只更新这些参数。

**ENAS的训练过程**：

```python
for iteration in range(num_iterations):
    # 控制器生成一个架构（子图）
    alpha = controller.sample()
    
    # 在训练集上训练这个子架构（使用共享权重）
    loss = compute_loss(alpha, W, train_data)
    update(W[alpha])  # 只更新用到的参数
    
    # 在验证集上评估，更新控制器
    reward = evaluate(alpha, W, val_data)
    controller.update(reward)
```

**为什么权重共享有效？

直觉上，相似架构应该具有相似的功能，因此可以共享参数。例如：
- 使用 $3\times3$ conv 的架构
- 使用 $5\times5$ conv 的架构

如果输入相似，它们的特征提取也应该相似，共享权重是合理的。

**权重共享的挑战**：

1. **耦合问题（Coupling）**：不同架构竞争相同的参数，可能导致某些架构表现差
2. **排名不一致**：超网中表现好的架构，从头训练后可能表现差
3. **训练不稳定**：共享参数使训练动态复杂化

### One-shot NAS：训练一次，评估所有

One-shot NAS（如DARTS、ProxylessNAS）将权重共享推向极致：

> **训练一个包含所有可能操作的超网，然后从中选择子架构。**

**超网架构（以DARTS为例）**：

对于Cell中的每条边，我们不选择一个操作，而是并行计算所有操作：

$$\text{output} = \sum_{o \in \mathcal{O}} \pi_o \cdot o(x)$$

其中 $\pi_o = \text{softmax}(\alpha_o)$ 是操作 $o$ 的权重。

**Single Path One-Shot（SPOS）**：

Guo et al.（2020）提出SPOS来解决超网训练中的耦合问题：

> **每次只采样一条路径（单路径），训练该路径上的参数。**

**SPOS训练算法**：

```python
for iteration in range(num_iterations):
    # 均匀随机采样一个架构
    alpha = uniform_sample()
    
    # 构建单路径网络
    subnet = build_subnet(alpha, supernet)
    
    # 训练这个子网
    loss = compute_loss(subnet, train_data)
    update(subnet.parameters)
```

**均匀采样的重要性**：

SPOS证明，均匀采样（每个架构被采样的概率相等）对于获得可靠的超网至关重要。这确保了所有架构得到公平的训练机会。

**性能预测器**：

训练好超网后，我们可以通过简单的前向传播快速评估任何架构的性能（无需重新训练）。

### 零次NAS（Zero-shot NAS）

最新的研究方向是**完全不训练**就能评估架构性能：

**基于梯度的指标**：
- **SNIP**：基于连接敏感度
- **GraSP**：基于梯度流
- **Fisher**：基于Fisher信息

**基于激活的指标**：
- **NASWOT**：基于ReLU激活的协方差

这些方法的共同思想：网络初始化时的某些统计量可以预测训练后的性能。

**零次指标的数学**：

以SNIP为例，它测量每个参数对损失的敏感度：

$$S(\theta) = |\theta \odot \nabla_\theta \mathcal{L}|$$

其中 $\odot$ 是逐元素乘法。统计所有参数的敏感度可以预测网络容量。

---

## 34.5 DARTS详解：可微架构搜索与双层优化

### DARTS的完整数学推导

DARTS将NAS转化为一个**双层优化（Bilevel Optimization）**问题。让我们深入理解其数学结构。

**搜索空间的连续松弛**：

考虑一个Cell，有 $N$ 个节点，节点 $i$ 和 $j$ 之间有边。在离散搜索空间中，每条边选择一个操作 $o^{(i,j)} \in \mathcal{O}$。

DARTS引入架构参数 $\alpha^{(i,j)} \in \mathbb{R}^{|\mathcal{O}|}$，将混合操作定义为：

$$\bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} p_o^{(i,j)}(\alpha) \cdot o(x)$$

其中混合权重通过softmax计算：

$$p_o^{(i,j)}(\alpha) = \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o' \in \mathcal{O}} \exp(\alpha_{o'}^{(i,j)})}$$

**节点计算的数学表达**：

每个节点 $j$ 接收来自所有前序节点 $i < j$ 的连接：

$$x_j = \sum_{i < j} \bar{o}^{(i,j)}(x_i)$$

**双层优化问题**：

定义：
- $\alpha$：架构参数（上层变量）
- $w$：网络权重（下层变量）
- $\mathcal{L}_{train}(w, \alpha)$：训练损失
- $\mathcal{L}_{val}(w, \alpha)$：验证损失

优化问题为：

$$\min_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha)$$

$$\text{s.t.} \quad w^*(\alpha) = \arg\min_w \mathcal{L}_{train}(w, \alpha)$$

**一阶近似**：

假设当前网络权重 $w$ 已经近似最优，直接计算：

$$\nabla_\alpha \mathcal{L}_{val}(w, \alpha)$$

**二阶近似（更精确）**：

考虑 $w$ 对 $\alpha$ 的依赖。使用一步梯度下降近似 $w^*(\alpha)$：

$$w' = w - \xi \nabla_w \mathcal{L}_{train}(w, \alpha)$$

然后计算：

$$\nabla_\alpha \mathcal{L}_{val}(w', \alpha)$$

展开链式法则：

$$\nabla_\alpha \mathcal{L}_{val}(w', \alpha) = \nabla_\alpha \mathcal{L}_{val}(w', \alpha) - \xi \nabla^2_{\alpha,w} \mathcal{L}_{train}(w, \alpha) \nabla_{w'} \mathcal{L}_{val}(w', \alpha)$$

其中 $\nabla^2_{\alpha,w} \mathcal{L}_{train}$ 是Hessian矩阵，计算昂贵。DARTS使用有限差分近似：

$$\nabla^2_{\alpha,w} \mathcal{L}_{train} \cdot v \approx \frac{\nabla_\alpha \mathcal{L}_{train}(w^+, \alpha) - \nabla_\alpha \mathcal{L}_{train}(w^-, \alpha)}{2\epsilon}$$

其中 $w^\pm = w \pm \epsilon v$。

**DARTS算法总结**：

```
初始化：随机初始化 α 和 w

对于每个迭代：
    # 步骤1：更新网络权重（训练集）
    对于每个训练批量：
        w ← w - η_w * ∇_w L_train(w, α)
    
    # 步骤2：更新架构参数（验证集）
    # 2.1 计算 w' = w - ξ * ∇_w L_train(w, α)
    w' = w - ξ * gradient(L_train, w, α)
    
    # 2.2 计算验证损失对α的梯度（使用w'）
    g_val = gradient(L_val, α, w')
    
    # 2.3 如果使用二阶近似，添加修正项
    如果 use_second_order:
        # 计算Hessian-向量积的近似
        epsilon = 0.01 / ||∇_w' L_val||
        w_plus = w + epsilon * ∇_w' L_val
        w_minus = w - epsilon * ∇_w' L_val
        
        g_plus = gradient(L_train, α, w_plus)
        g_minus = gradient(L_train, α, w_minus)
        
        hessian_correction = (g_plus - g_minus) / (2 * epsilon)
        g_val = g_val - ξ * hessian_correction
    
    α ← α - η_α * g_val

# 离散化
对于每条边 (i,j):
    选择 o* = argmax_o α_o^(i,j)
```

### DARTS的问题与改进

**性能崩溃（Performance Collapse）**：

DARTS有一个著名的问题：经常收敛到充满跳跃连接（skip connections）的退化架构。原因：

> 跳跃连接在早期训练阶段容易优化，因为它们是恒等映射，不需要学习参数。

**DARTS+（Liang et al., 2019）**：

解决方案：早停（Early Stopping）。当检测到架构参数中某个操作占主导（如softmax值>0.9）时，立即停止搜索。

**PC-DARTS（Xu et al., 2020）**：

解决方案：部分通道连接。每次只随机采样部分通道进行架构搜索，减少内存使用并提高稳定性。

**Fair DARTS（Chu et al., 2020）**：

解决方案：引入sigmoid代替softmax，并添加公平性正则化，确保每个操作都有机会展示自己。

---

## 34.6 ProxylessNAS：硬件感知搜索

### 代理问题与直接搜索

传统NAS流程：
1. 在代理数据集（如CIFAR-10）上搜索
2. 将最佳架构迁移到目标数据集（如ImageNet）

**代理问题**：
- 小数据集上表现好的架构，大数据集上不一定好
- 搜索时不考虑实际部署硬件

ProxylessNAS的核心思想：

> **直接在目标数据集和硬件上搜索！**

### 二进制路径（Binary Gates）

ProxylessNAS的关键创新：使用二进制门控来减少内存使用。

对于每条边，DARTS保存所有操作的结果：

$$\bar{o}^{(i,j)} = \sum_{o \in \mathcal{O}} p_o \cdot o(x)$$

这需要 $|\mathcal{O}|$ 倍的内存。

ProxylessNAS改为：

$$\bar{o}^{(i,j)} = \sum_{o \in \mathcal{O}} g_o \cdot o(x)$$

其中 $g_o \in \{0, 1\}$ 且 $\sum_o g_o = 1$（只有一个操作被激活）。

**可微分松弛**：

为了梯度下降，使用Gumbel-Softmax技巧：

$$g_o = \text{softmax}((\alpha_o + \epsilon_o) / \tau)$$

其中 $\epsilon_o$ 是Gumbel噪声，$\tau$ 是温度参数。

### 硬件感知损失函数

ProxylessNAS引入硬件延迟作为第二优化目标：

$$\min_\alpha \mathcal{L}_{ce}(\alpha) + \lambda \cdot \mathcal{L}_{latency}(\alpha)$$

**延迟建模**：

实际测量每个操作的延迟，构建查找表（LUT）：

$$\text{latency}(\alpha) = \sum_{(i,j)} \sum_{o \in \mathcal{O}} p_o^{(i,j)} \cdot \text{latency}(o)$$

**延迟预测器**：

可以在目标设备上测量，或使用预测模型：

$$\text{latency}(o, \text{input\_shape}, \text{hardware}) = \text{LUT}[o, \text{input\_shape}, \text{hardware}]$$

> 💡 **费曼时间：量身定制西装**
> 
> 想象你买西装。传统NAS像是在模特身上试穿，然后希望在你身上也合身。ProxylessNAS则是直接在你身上量体裁衣，考虑你的具体身材（数据集）和出席场合（硬件）。

---

## 34.7 Once-for-All & BigNAS：训练一次，处处部署

### 动机：从专用到通用

传统NAS流程的问题：
1. 搜索架构A
2. 从头训练架构A
3. 想要不同大小的架构B？重新搜索并训练！

**Once-for-All（OFA）的洞察**：

> **训练一个巨大的超网，包含所有可能的子架构，然后直接"提取"需要的子网，无需重新训练！**

### OFA的搜索空间

OFA的搜索空间包含四个维度：

1. **深度（Depth）**：网络层数，如 $\{2, 3, 4\}$
2. **宽度（Width）**：通道数，如 $\{128, 192, 256\}$
3. **核大小（Kernel Size）**：卷积核，如 $\{3, 5, 7\}$
4. **分辨率（Resolution）**：输入大小，如 $\{128, 160, 192, 224\}$

**搜索空间大小**：

$$|\mathcal{A}| = |\text{depth}| \times |\text{width}|^{\text{num\_stages}} \times |\text{kernel}|^{\text{num\_layers}} \times |\text{resolution}|$$

对于典型设置，搜索空间包含 **$>10^{19}$** 个架构！

### 渐进式收缩训练（Progressive Shrinking）

直接训练如此大的超网会导致子网之间严重干扰。OFA的解决方案：

> **从大架构开始，逐步训练越来越小的子网。**

**训练阶段**：

```
阶段1：训练最大架构（深度=4，宽度=256，核=7，分辨率=224）
    ↓
阶段2：弹性核大小（支持核大小3,5,7）
    ↓
阶段3：弹性深度（支持深度2,3,4）
    ↓
阶段4：弹性宽度（支持宽度128,192,256）
    ↓
阶段5：弹性分辨率（支持128-224）
```

在每个阶段，我们同时训练当前支持的所有子架构，使用均匀采样。

**为什么渐进式收缩有效？

- 大架构提供良好的特征表示
- 小架构从大架构继承并微调
- 避免同时优化所有子网导致的冲突

### 部署时的快速特化

训练好OFA超网后，对于任何部署场景：

1. **定义约束**：延迟 < 10ms，或 FLOPs < 300M
2. **搜索最优子网**：使用进化搜索或预测器找到满足约束的最佳子网
3. **直接提取**：从超网中"切出"对应的权重，无需重新训练

**准确率预测器**：

训练一个小型MLP来预测任意子网的准确率：

$$\text{acc}(\alpha) = \text{MLP}(\text{encode}(\alpha))$$

编码 $\text{encode}(\alpha)$ 将架构的超参数转换为向量。

**联合优化**：

$$\max_\alpha \text{acc}(\alpha) \quad \text{s.t.} \quad \text{latency}(\alpha) < T_{target}$$

### BigNAS：单次训练超大模型

BigNAS（Yu et al., 2020）扩展了OFA思想：

> **直接训练可以支持任意子网的超大模型，无需渐进式收缩。**

关键技术：

1. **三明治规则（Sandwich Rule）**：每个训练迭代同时训练最小子网、最大子网和随机采样的子网
2. **就地蒸馏（In-place Distillation）**：最大子网作为教师，蒸馏知识给小子网
3. **统一采样**：确保不同大小的子网都被充分训练

**损失函数**：

$$\mathcal{L} = \mathcal{L}_{CE}(y_{max}, y_{gt}) + \sum_{i} \mathcal{L}_{KD}(y_i, y_{max})$$

其中第一项是最大子网的交叉熵损失，第二项是子网 $i$ 的知识蒸馏损失。

---

## 34.8 应用案例：手机端高效网络设计

### 移动端AI的挑战

在手机上运行深度学习模型面临独特挑战：

1. **计算受限**：手机CPU/GPU性能有限
2. **内存受限**：通常只有2-8GB RAM
3. **能耗敏感**：模型推理消耗电池
4. **延迟敏感**：用户期望实时响应

**设计要求**：
- 低FLOPs（< 500M）
- 低参数量（< 10M）
- 低延迟（< 100ms）
- 高精度（ImageNet top-1 > 70%）

### FBNet：为Facebook应用优化

FBNet（Wu et al., 2019）使用DARTS在FBNet搜索空间中找到针对特定手机优化的架构。

**FBNet搜索空间特点**：
- 基于MobileNetV2的倒残差块
- 搜索每层的通道数、扩展率、核大小、步长
- 直接优化iPhone上的实际延迟

**结果**：
- FBNet-A：ImageNet 73.3% accuracy，295M FLOPs
- 比MobileNetV2-1.0快1.9倍，精度更高

### MobileNetV3：人工+NAS的混合

MobileNetV3（Howard et al., 2019）使用平台感知NAS搜索全局架构，然后用NetAdapt算法逐层微调。

**搜索流程**：
1. **粗搜索**：使用MNasNet（基于强化学习的硬件感知NAS）找到初始架构
2. **细优化**：NetAdapt移除瓶颈层，进一步优化延迟
3. **人工调整**：研究人员根据直觉微调

**MobileNetV3-Large**：
- ImageNet top-1：75.2%
- 延迟：66ms（Pixel 1手机）
- 比MobileNetV2快2.5倍，精度更高

### EfficientNet：复合缩放

EfficientNet（Tan & Le, 2019）展示了如何结合NAS和系统化的网络缩放。

**复合缩放（Compound Scaling）**：

传统缩放只改变一个维度：
- 只增加深度（ResNet-18 → ResNet-50）
- 只增加宽度（Wide ResNet）
- 只增加分辨率

EfficientNet发现：

> **同时按固定比例缩放深度、宽度和分辨率效果更好。**

如果资源增加 $2^\phi$ 倍，则：
- 深度：$d = \alpha^\phi$
- 宽度：$w = \beta^\phi$
- 分辨率：$r = \gamma^\phi$

约束：$\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$

**EfficientNet-B0到B7**：

使用NAS搜索基线模型B0，然后使用复合缩放得到B1-B7。

| 模型 | FLOPs | ImageNet Top-1 |
|------|-------|----------------|
| B0 | 390M | 77.1% |
| B3 | 1.8B | 81.1% |
| B7 | 37B | 84.3% |

EfficientNet-B7是当时最高效的模型，超越了GPipe（557M参数 vs 556M参数，但训练快6.1倍）。

### 实际部署考虑

**模型量化**：

将32位浮点权重转换为8位整数：

$$w_{int8} = \text{round}(w_{fp32} / \text{scale} + \text{zero\_point})$$

可以减少4倍模型大小，加速2-4倍。

**知识蒸馏**：

用大模型（教师）训练小模型（学生）：

$$\mathcal{L} = \lambda \cdot \text{KL}(\text{softmax}(z_s/T), \text{softmax}(z_t/T)) + (1-\lambda) \cdot \text{CE}(z_s, y)$$

其中 $T$ 是温度参数。

**硬件感知优化**：

- 使用专用推理引擎（TensorRT、Core ML、TFLite）
- 算子融合：将conv+bn+relu合并为单个算子
- 内存布局优化：使用NHWC或NCHW格式匹配硬件

---

## 34.9 练习题

### 基础练习

**练习1：搜索空间大小计算**

考虑一个Macro搜索空间，网络有10层，每层可以选择：
- 操作：Conv3x3, Conv5x5, Depthwise3x3, MaxPool3x3（4种）
- 滤波器数量：32, 64, 128（3种）
- 步长：1, 2（2种）

计算搜索空间的总大小。如果评估一个架构需要1小时GPU时间，穷举搜索需要多长时间？

**练习2：策略梯度推导**

给定控制器生成架构的概率 $P(\alpha; \theta)$ 和奖励 $R(\alpha)$，推导REINFORCE算法的梯度公式。解释为什么引入基线（baseline）可以减少方差。

**练习3：DARTS连续松弛**

假设一条边有两个操作选择：Conv3x3和MaxPool3x3，架构参数分别为 $\alpha_1 = 0.5$ 和 $\alpha_2 = 0.3$。计算：
1. Softmax权重
2. 如果输入特征图 $x$ 通过这两个操作后分别为 $y_1$ 和 $y_2$，混合操作的输出是什么？

### 进阶练习

**练习4：实现简单的NAS控制器**

使用PyTorch实现一个基于LSTM的NAS控制器，能够生成包含5层的CNN架构。每层需要选择操作类型（Conv3x3, Conv5x5, MaxPool）和滤波器数量（16, 32, 64）。

**练习5：ENAS权重共享分析**

解释为什么ENAS的权重共享可以加速NAS。讨论权重共享可能导致的问题（如耦合、排名不一致）。

**练习6：DARTS离散化分析**

DARTS搜索结束后，通过选择每个边上权重最大的操作来离散化架构。讨论这种方法可能存在的问题，并提出改进思路。

### 挑战练习

**练习7：实现简化的DARTS**

实现一个简化的DARTS版本，用于在CIFAR-10上搜索CNN架构。要求：
- 使用cell-based搜索空间
- 每个cell包含4个节点
- 操作选择：Conv3x3, Conv5x5, MaxPool3x3, Skip Connection
- 实现一阶近似优化
- 搜索50轮后，评估最佳架构的测试准确率

**练习8：硬件感知损失函数设计**

假设你要设计一个针对树莓派4B的图像分类模型。设计一个综合考虑准确率、延迟和能耗的多目标优化函数。讨论如何权衡这些目标。

**练习9：OFA的渐进式收缩分析**

解释OFA为什么使用渐进式收缩而不是直接训练整个超网。分析渐进式收缩的优缺点，并思考可能的改进方法。

---

## 参考文献

Bender, G., Kindermans, P. J., Zoph, B., Vasudevan, V., & Le, Q. (2018). Understanding and simplifying one-shot architecture search. In *International Conference on Machine Learning* (pp. 550-559). PMLR.

Cai, H., Gan, C., Wang, T., Zhang, Z., & Han, S. (2020). Once-for-all: Train one network and specialize it for efficient deployment. In *International Conference on Learning Representations*. https://openreview.net/forum?id=HylxE1HKwS

Cai, H., Zhu, L., & Han, S. (2019). ProxylessNAS: Direct neural architecture search on target task and hardware. In *International Conference on Learning Representations*. https://openreview.net/forum?id=HylVB3AqYm

Guo, Z., Zhang, X., Mu, H., Heng, W., Liu, Z., Wei, Y., & Sun, J. (2020). Single path one-shot neural architecture search with uniform sampling. In *European Conference on Computer Vision* (pp. 544-560). Springer.

Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., ... & Adam, H. (2019). Searching for MobileNetV3. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 1314-1324).

Liu, H., Simonyan, K., & Yang, Y. (2019). DARTS: Differentiable architecture search. In *International Conference on Learning Representations*. https://openreview.net/forum?id=S1eYHoC5FX

Pham, H., Guan, M. Y., Zoph, B., Le, Q. V., & Dean, J. (2018). Efficient neural architecture search via parameter sharing. In *International Conference on Machine Learning* (pp. 4095-4104). PMLR.

Real, E., Aggarwal, A., Huang, Y., & Le, Q. V. (2019). Regularized evolution for image classifier architecture search. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 33, No. 01, pp. 4780-4789).

Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *International Conference on Machine Learning* (pp. 6105-6114). PMLR.

Wu, B., Dai, X., Zhang, P., Wang, Y., Sun, F., Wu, Y., ... & Keutzer, K. (2019). FBNet: Hardware-aware efficient ConvNet design via differentiable neural architecture search. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 10734-10742).

Yu, J., Jin, P., Liu, H., Bender, G., Kindermans, P. J., Tan, M., ... & Le, Q. (2020). BigNAS: Scaling up neural architecture search with big single-stage models. In *European Conference on Computer Vision* (pp. 702-717). Springer.

Zoph, B., & Le, Q. V. (2017). Neural architecture search with reinforcement learning. In *International Conference on Learning Representations*. https://arxiv.org/abs/1611.01578

Zoph, B., Vasudevan, V., Shlens, J., & Le, Q. V. (2018). Learning transferable architectures for scalable image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 8697-8710).

---

*本章正文字数：约16,200字*
*代码实现：见 chapter34_nas.py*
