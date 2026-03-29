# 第五十六章 神经架构搜索进阶（续）

## 56.4 多目标NAS：准确率vs效率——帕累托最优的追求

### 56.4.1 多目标优化的挑战

在现实世界中，我们很少只追求一个目标。想象一下买车：你想要**速度快**，又想要**省油**，还想要**价格便宜**。但这些目标往往是冲突的——跑车快但费油，电动车省油但可能不够快，便宜的车可能哪方面都不够好。

神经架构搜索面临同样的困境：
- **准确率（Accuracy）** vs **延迟（Latency）**
- **准确率** vs **参数量（Parameters）**
- **准确率** vs **能耗（Energy）**
- **准确率** vs **内存占用（Memory）**

**为什么不能简单加权求和？**

一个常见的想法是：把多个目标变成一个目标：

$$\text{目标} = \text{准确率} - \lambda \cdot \text{延迟}$$

但问题是：$\lambda$ 怎么选？
- 如果 $\lambda$ 太小，你得到的模型可能很慢
- 如果 $\lambda$ 太大，你得到的模型可能准确率太低
- 而且不同应用场景需要不同的 trade-off

### 56.4.2 帕累托最优：一组最优解，而非一个

**帕累托最优**（Pareto Optimality）是多目标优化的核心概念。

**费曼法比喻**：想象你在规划一次旅行。有人说："我想花最少的钱"，有人说："我想花最少的时间"，有人说："我想最舒服"。不可能有一个方案同时满足所有人，但我们可以找到一组"最优"方案——对于这组方案中的任何一个，你不可能在不牺牲其他目标的情况下改进某一个目标。

**数学定义**：

一个解 $\mathbf{x}^*$ 是帕累托最优的，如果不存在另一个解 $\mathbf{x}$ 满足：
1. 对所有目标，$f_i(\mathbf{x}) \leq f_i(\mathbf{x}^*)$（不差于）
2. 对至少一个目标，$f_j(\mathbf{x}) < f_j(\mathbf{x}^*)$（严格更好）

**帕累托前沿**（Pareto Front）：所有帕累托最优解在目标空间中的像形成的曲线/曲面。

![帕累托前沿示意图](pareto_front.png)

*图示：帕累托前沿将可行解空间分成两个区域——前沿上的点是帕累托最优的，下方的点是支配的。*

### 56.4.3 NSGA-II：多目标进化算法

**NSGA-II**（Non-dominated Sorting Genetic Algorithm II）是解决多目标优化问题最流行的算法之一。

**核心思想**：

1. **非支配排序**：将种群分成不同的"前沿层"（Fron tiers）
   - 第1层：不被任何其他解支配的解
   - 第2层：被第1层支配，但不被其他解支配
   - 以此类推...

2. **拥挤度距离**：在同一层内，优先选择"孤独"的解（周围没有其他解）
   - 这保证了帕累托前沿的多样性

3. **遗传操作**：选择、交叉、变异，产生新一代种群

**NSGA-II在NAS中的应用**：

```python
# 伪代码
population = 随机初始化N个架构
for generation in range(max_gen):
    offspring = 遗传操作(population)
    combined = population + offspring
    
    # 非支配排序
    fronts = 非支配排序(combined)
    
    # 选择下一代
    new_population = []
    for front in fronts:
        if len(new_population) + len(front) <= N:
            new_population.extend(front)
        else:
            # 按拥挤度距离选择
            front.sort(key=拥挤度距离, reverse=True)
            new_population.extend(front[:N - len(new_population)])
            break
    
    population = new_population
```

### 56.4.4 超体积指标（Hypervolume）

如何评估一组帕累托解的好坏？**超体积指标**是最常用的指标。

**定义**：

给定一个参考点 $r$（通常是最差的点），超体积是帕累托解集在目标空间中与 $r$ 形成的超立方体的并集体积。

$$HV(\mathcal{P}, r) = \text{Volume}\left(\bigcup_{p \in \mathcal{P}} [p, r]\right)$$

**为什么超体积好？**

1. **同时评估收敛性和多样性**：既要接近真实帕累托前沿，又要分布均匀
2. **Pareto-compliant**：如果一个解集支配另一个，其超体积一定更大
3. **直观**：面积/体积越大越好

![超体积示意图](hypervolume.png)

*图示：超体积是帕累托解集与参考点形成的阴影区域。*

---

## 56.5 硬件感知神经架构搜索

### 56.5.1 为什么要硬件感知？

**一个惊人的发现**：同样的神经网络，在不同的硬件上运行，最优架构完全不同。

**ProxylessNAS**（Cai et al., 2019）的研究表明：

| 模型 | GPU延迟 | CPU延迟 | 移动端延迟 |
|------|---------|---------|-----------|
| Proxyless-GPU | 5.1ms | 204.9ms | 124ms |
| Proxyless-CPU | 7.4ms | **138.7ms** | 116ms |
| Proxyless-Mobile | 7.2ms | 164.1ms | **78ms** |

**关键洞察**：
- GPU优化的模型在CPU上可能比CPU优化的模型慢**47%**
- 移动端优化的模型在GPU上表现平平

**为什么？**

不同硬件有不同的特性：
- **GPU**：并行计算能力强，喜欢"宽"的层（大量并行操作）
- **CPU**：串行执行，喜欢"深"的层（更好的缓存局部性）
- **移动端NPU**：有特定的算子优化，某些操作特别快

### 56.5.2 路径二值化：ProxylessNAS的核心创新

**问题**：传统的DARTS需要存储所有候选路径的权重，内存开销巨大。

**ProxylessNAS的解决方案**：**路径二值化**（Path Binarization）。

**核心思想**：

1. 在训练时，只激活**一条路径**（通过Gumbel-Softmax采样）
2. 其他路径被"屏蔽"，不占用内存
3. 这样，内存开销与正常训练单个模型相同

**数学表达**：

$$m = \text{BinaryGate}(GumbelSoftmax(\alpha))$$
$$
\text{output} = \sum_{i=1}^{n} m_i \cdot \text{path}_i(x)$$

其中，$m$ 是one-hot向量（只有一个1，其余为0）。

**效果**：
- 内存消耗降低**一个数量级**
- 可以直接在ImageNet上搜索（不需要代理任务）
- 搜索成本降低到**200 GPU-hours**（相比MnasNet的40,000 GPU-hours）

### 56.5.3 延迟预测模型

**如何在搜索过程中考虑延迟？**

直接测量每个候选架构的延迟太慢了。ProxylessNAS的做法是：

1. **建立延迟查找表**（Latency Lookup Table）：
   - 预先测量每个基本操作（3×3 conv, 5×5 conv, etc.）在不同输入尺寸下的延迟
   - 存储在LUT中

2. **延迟预测**：
   $$\text{Latency}(\text{arch}) \approx \sum_{i} \text{LUT}[\text{op}_i, \text{shape}_i]$$

3. **可微分延迟**：
   为了让延迟可以端到端优化，ProxylessNAS使用：
   $$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \cdot \log(\text{LAT})^\beta$$

**实际误差**：< 1ms，足够用于指导搜索。

### 56.5.4 不同硬件的架构偏好

**ProxylessNAS发现**：

**GPU偏好**：
- **浅而宽**：更多通道数，更少层数
- **早期池化**：尽早降低特征图分辨率
- **原因**：GPU的并行计算能力适合处理大张量

**CPU偏好**：
- **深而窄**：更多层数，更少通道数
- **晚期池化**：保持高分辨率更久
- **原因**：CPU的缓存机制喜欢小内存访问模式

**移动端**：
- 介于两者之间
- 更注重整体延迟而非某一层

![不同硬件的架构对比](hardware_arch.png)

*图示：GPU、CPU、移动端的最优架构可视化对比。*

---

## 56.6 大模型的架构优化

### 56.6.1 大模型架构搜索的挑战

当模型大到无法完整训练时，传统的NAS方法失效了：

**挑战1：训练成本**
- GPT-3级别的模型训练一次需要数百万美元
- 无法像DARTS那样训练"超网络"包含所有候选

**挑战2：搜索空间爆炸**
- 大模型有更多可调参数：层数、注意力头数、FFN维度、上下文长度...
- 组合爆炸使得穷举不可能

**挑战3：Scaling Law的约束**
- 大模型的性能与计算量、参数量、数据量有幂律关系
- 架构设计必须遵循这些规律

### 56.6.2 Mixture of Experts (MoE) 的自动设计

**MoE核心思想**：

不是所有参数都参与每次前向传播。模型由多个"专家"（小网络）组成，每次只激活其中一部分。

$$y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)$$

其中，$G$ 是门控网络，$E_i$ 是第$i$个专家。

**Switch Transformers**（Fedus et al., 2022）的发现：

1. **专家数越多越好**：在固定计算预算下，更多专家（每个更小）通常更好
2. **负载均衡很重要**：需要确保所有专家都被使用，而不是总是选同样的几个
3. **通信开销是关键**：专家分布在不同设备上时，通信可能成为瓶颈

**MoE的架构搜索维度**：
- 专家数量
- 每个专家的规模
- 门控网络设计
- 专家分配策略（数据并行 vs 模型并行）

### 56.6.3 长上下文架构的搜索

**长上下文是LLM的新战场**：

- GPT-4支持128K token
- Claude支持200K+
- Gemini支持1M+

**挑战**：标准Transformer的注意力复杂度是 $O(n^2)$，对于长上下文不友好。

**架构搜索方向**：

1. **稀疏注意力模式**：
   - 局部注意力（只关注附近的token）
   - 全局注意力（只关注特定的全局token）
   - 滑动窗口注意力
   - 随机注意力

2. **线性注意力变体**：
   - 用核技巧近似softmax注意力
   - 复杂度降低到 $O(n)$

3. **混合架构**：
   - 某些层用标准注意力
   - 某些层用稀疏/线性注意力

**搜索目标**：在给定上下文长度下，最大化性能同时保持可接受的计算成本。

---

## 56.7 实战案例：完整的AutoML流水线

### 56.7.1 端到端AutoML系统架构

一个完整的AutoML系统包括：

```
┌─────────────────────────────────────────────────────────────┐
│                    AutoML 流水线                            │
├─────────────────────────────────────────────────────────────┤
│  1. 数据准备                                                │
│     ├── 数据清洗                                            │
│     ├── 数据增强策略搜索（AutoAugment）                      │
│     └── 特征工程（如果需要）                                 │
│                                                             │
│  2. 架构搜索（NAS）                                          │
│     ├── 定义搜索空间                                         │
│     ├── 选择搜索算法（DARTS+/ProxylessNAS/进化算法）          │
│     └── 运行搜索                                            │
│                                                             │
│  3. 超参数优化（HPO）                                        │
│     ├── 学习率调度                                           │
│     ├── 优化器选择                                           │
│     └── 正则化强度                                           │
│                                                             │
│  4. 模型训练                                                 │
│     └── 使用最优配置完整训练                                  │
│                                                             │
│  5. 模型部署                                                 │
│     ├── 模型压缩（剪枝/量化/蒸馏）                             │
│     └── 部署到目标平台                                        │
└─────────────────────────────────────────────────────────────┘
```

### 56.7.2 案例：用AutoML解决图像分类问题

**任务**：在CIFAR-10上自动搜索最优架构

**步骤1：数据准备**
```python
# 使用AutoAugment自动搜索最优数据增强策略
from autoaugment import AutoAugment

transform = AutoAugment(policy='cifar10')
```

**步骤2：架构搜索（使用DARTS+）**
```python
# 见代码实现部分
searcher = DARTSPlusSearcher(
    search_space='darts',
    early_stop_threshold=0.3,
    max_epochs=50
)
best_arch = searcher.search(train_loader, val_loader)
```

**步骤3：超参数优化**
```python
# 使用贝叶斯优化或Hyperband
from ray import tune

config = {
    'lr': tune.loguniform(1e-4, 1e-1),
    'weight_decay': tune.loguniform(1e-5, 1e-2),
    'dropout': tune.uniform(0, 0.5)
}
```

**步骤4：完整训练**
```python
model = build_model(best_arch)
train(model, train_loader, val_loader, config)
```

**结果对比**：

| 方法 | CIFAR-10错误率 | 搜索成本 |
|------|----------------|----------|
| 手工设计ResNet | 6.4% | - |
| 标准DARTS | 3.0% | 1 GPU-day |
| **DARTS+ (本章)** | **2.32%** | **0.3 GPU-day** |

### 56.7.3 AutoML的未来方向

1. **零成本NAS**：进一步降低搜索成本，实现实时架构调整
2. **终身NAS**：模型在使用中不断进化架构
3. **跨任务迁移**：在一个任务上搜索的架构知识迁移到新任务
4. **AutoML + LLM**：用大模型指导架构搜索，结合符号推理和神经网络
5. **绿色AutoML**：将能耗作为优化目标，推动可持续AI

---

## 56.8 本章总结

### 核心概念回顾

| 概念 | 核心思想 | 应用场景 |
|------|----------|----------|
| DARTS+ | 早停防止性能崩溃 | 稳定可微分NAS |
| LoRA-DARTS | 低秩适应公平竞争 | 解决skip-connection主导 |
| SD-DARTS | 自蒸馏减少离散化差距 | 提高搜索稳定性 |
| Zero-Cost NAS | 零成本评估架构 | 极速搜索 |
| As-ViT | 无需训练搜索ViT | Transformer架构优化 |
| NSGA-II | 多目标进化算法 | 准确率vs效率 trade-off |
| ProxylessNAS | 路径二值化+硬件感知 | 移动端/边缘部署 |
| MoE搜索 | 专家混合自动设计 | 大模型扩展 |

### 费曼法金句

> "DARTS的性能崩溃就像是过度训练运动员——如果不及时喊停，状态反而越来越差。"

> "硬件感知NAS告诉我们：一双鞋不可能适合所有人，为特定硬件定制的架构才能发挥最佳性能。"

> "AutoML的终极目标不是取代人类，而是把人类从重复性的调参工作中解放出来，专注于更高层次的创新。"

### 下一步

恭喜你完成了第56章的学习！现在你可以：
1. 实现本章介绍的改进版DARTS算法
2. 尝试为自己的硬件平台搜索专用架构
3. 探索多目标优化在你的应用场景中的应用
4. 将AutoML思想应用到其他领域（数据增强、超参数优化等）

下一章预告：**第五十七章——自动化机器学习系统：AutoML平台全景**

我们将探索完整的AutoML生态系统，包括Google AutoML、AutoGluon、NNI等开源平台，学习如何构建企业级的自动化机器学习流水线。

---

## 参考文献

Cai, H., Zhu, L., & Han, S. (2019). ProxylessNAS: Direct neural architecture search on target task and hardware. In *ICLR*.

Chen, W., Huang, W., Du, X., Song, X., Wang, Z., & Zhou, D. (2022). As-ViT: Auto-scaling vision transformers without training. In *ICLR*.

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.

Elsken, T., Metzen, J. H., & Hutter, F. (2019). Neural architecture search: A survey. *Journal of Machine Learning Research*, 20(55), 1-21.

Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(1), 5232-5270.

Liang, H., Zhang, S., Sun, J., He, X., Huang, W., Zhuang, K., & Li, Z. (2020). DARTS+: Improved differentiable architecture search with early stopping. *arXiv preprint arXiv:1909.06035*.

Liu, H., Simonyan, K., & Yang, Y. (2019). DARTS: Differentiable architecture search. In *ICLR*.

Tan, M., Chen, B., Pang, R., Vasudevan, V., Sandler, M., Howard, A., & Le, Q. V. (2019). Mnasnet: Platform-aware neural architecture search for mobile. In *CVPR* (pp. 2820-2828).

Xiang, L., Dudziak, L., Abdelfattah, M. S., Chau, T., Lane, N. D., & Wen, H. (2023). Zero-cost operation scoring in differentiable architecture search. In *AAAI* (Vol. 37, No. 9, pp. 10453-10463).

Zela, A., Klein, A., Falkner, S., & Hutter, F. (2020). Understanding and robustifying differentiable architecture search. In *ICLR*.

Zhu, X., Li, J., Liu, Y., & Ma, Z. M. (2023). Improving differentiable architecture search via self-distillation. *Neurocomputing*, 549, 126438.

---

*本章完*
