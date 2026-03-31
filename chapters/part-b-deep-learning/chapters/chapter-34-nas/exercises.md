- **强化学习（Reinforcement Learning）**：训练一个"控制器"网络来生成架构
- **进化算法（Evolutionary Algorithms）**：模拟自然选择，让架构种群进化
- **梯度优化（Gradient-based Optimization）**：将离散的架构选择松弛为连续的优化问题

**3. 性能评估（Performance Estimation）——宝藏价值**

我们需要快速评估每个找到的网络有多好：

--
硬件感知NAS（如ProxylessNAS、FBNet）能够针对特定设备（手机、边缘设备、ASIC）优化网络，这是手工设计难以做到的。

> 🎯 **思考时刻**
> 
> 想象一下，如果AlphaGo不仅能下棋，还能设计自己的神经网络架构来下棋，会发生什么？这听起来像递归自我改进的AI——正是一些关于人工通用智能（AGI）的推测性想法的核心。

---

--
**正则化进化（Regularized Evolution）**：

Real et al. 发现传统进化算法有"早熟收敛"问题——种群多样性快速丧失。他们提出：

> 强制移除最老的个体，而不是最差的个体。

这保持了种群多样性，让搜索持续探索新的区域。

--
2. **架构参数** $\alpha$：最小化验证损失

这形成了一个双层优化问题：

$$\min_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha)$$

$$\text{s.t.} \quad w^*(\alpha) = \arg\min_w \mathcal{L}_{train}(w, \alpha)$$

--
**权重共享的挑战**：

1. **耦合问题（Coupling）**：不同架构竞争相同的参数，可能导致某些架构表现差
2. **排名不一致**：超网中表现好的架构，从头训练后可能表现差
3. **训练不稳定**：共享参数使训练动态复杂化

### One-shot NAS：训练一次，评估所有

--
**Single Path One-Shot（SPOS）**：

Guo et al.（2020）提出SPOS来解决超网训练中的耦合问题：

> **每次只采样一条路径（单路径），训练该路径上的参数。**

**SPOS训练算法**：

--
### DARTS的完整数学推导

DARTS将NAS转化为一个**双层优化（Bilevel Optimization）**问题。让我们深入理解其数学结构。

**搜索空间的连续松弛**：

考虑一个Cell，有 $N$ 个节点，节点 $i$ 和 $j$ 之间有边。在离散搜索空间中，每条边选择一个操作 $o^{(i,j)} \in \mathcal{O}$。

--
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

--
```

### DARTS的问题与改进

**性能崩溃（Performance Collapse）**：

DARTS有一个著名的问题：经常收敛到充满跳跃连接（skip connections）的退化架构。原因：

> 跳跃连接在早期训练阶段容易优化，因为它们是恒等映射，不需要学习参数。

**DARTS+（Liang et al., 2019）**：

--
## 34.6 ProxylessNAS：硬件感知搜索

### 代理问题与直接搜索

传统NAS流程：
1. 在代理数据集（如CIFAR-10）上搜索
2. 将最佳架构迁移到目标数据集（如ImageNet）
