# 第四十八章 不确定性量化与贝叶斯深度学习

## 章节目标 🎯
- 理解不确定性的本质：认知不确定性与偶然不确定性
- 掌握贝叶斯神经网络的核心原理与变分推断
- 学习MC-Dropout作为贝叶斯近似的方法
- 深入理解深度集成(Deep Ensembles)的数学基础
- 探索证据深度学习(EDL)的前瞻方法
- 实现完整的不确定性量化工具包

---

## 48.1 为什么需要不确定性？

### 48.1.1 确定性预测的陷阱

想象你是一位医生，AI诊断系统告诉你："这位患者患癌症的概率是85%。"你会立即开始治疗吗？

**等一下！** 这个85%背后隐藏着什么问题？

**费曼法解释**：确定性预测就像一个**盲目自信的预言家**。他说"明天会下雨"，但不说"我有70%的把握"。当预言出错时，你无法判断是他能力不行，还是天气本来就难以预测。

**真实案例**：
- 2016年，一辆特斯拉自动驾驶汽车在佛罗里达州发生事故，因为系统**过于自信**地将白色卡车侧面识别为天空
- 2020年，多个COVID-19诊断AI在面对新型变异病毒时给出错误诊断，因为它们无法表达"我不确定"

### 48.1.2 两种不确定性

在机器学习中，不确定性可以分为两种本质不同的类型：

**偶然不确定性 (Aleatoric Uncertainty)** —— **"世界本身就随机"**

> 费曼比喻：想象你在抛一枚硬币。即使你是世界上最聪明的人，也无法准确预测单次抛掷的结果。这种不确定性来自数据本身的随机性，是**不可减少**的。

**例子**：
- 医学影像中的噪声
- 股票价格的随机波动
- 天气预报中的混沌效应

$$
\text{偶然不确定性} = \text{数据本身的固有噪声}
$$

**认知不确定性 (Epistemic Uncertainty)** —— **"我学得还不够"**

> 费曼比喻：想象一个从未见过企鹅的人第一次看到帝企鹅和蓝企鹅。他会困惑："这是两种不同的鸟，还是同一种鸟的不同颜色？"这种不确定性来自**知识的缺乏**，可以通过学习更多数据来减少。

**例子**：
- 模型从未见过的输入（OOD样本）
- 训练数据覆盖不足的区域
- 模型参数的不确定性

$$
\text{认知不确定性} = \text{模型对知识的缺乏}
$$

### 48.1.3 不确定性分解的数学表达

对于回归问题，假设我们预测的目标 $y$ 服从高斯分布：

$$
p(y|\mathbf{x}, \mathbf{w}) = \mathcal{N}(y; f^{\mathbf{w}}(\mathbf{x}), \sigma^2)
$$

其中：
- $f^{\mathbf{w}}(\mathbf{x})$ 是神经网络的预测
- $\sigma^2$ 是偶然不确定性（数据噪声）

但在贝叶斯框架中，我们对权重 $\mathbf{w}$ 也有不确定性：

$$
p(y|\mathbf{x}, \mathcal{D}) = \int p(y|\mathbf{x}, \mathbf{w}) p(\mathbf{w}|\mathcal{D}) d\mathbf{w}
$$

**总不确定性**可以分解为：

$$
\underbrace{\mathbb{V}[y]}_{\text{总不确定性}} = \underbrace{\mathbb{E}_{p(\mathbf{w}|\mathcal{D})}[\sigma^2(\mathbf{x})]}_{\text{偶然不确定性}} + \underbrace{\mathbb{V}_{p(\mathbf{w}|\mathcal{D})}[f^{\mathbf{w}}(\mathbf{x})]}_{\text{认知不确定性}}
$$

**费曼法解释**：
- **偶然不确定性** = 即使我告诉你正确答案，数据本身的噪声仍然存在
- **认知不确定性** = 因为我的模型参数不确定，导致预测结果有波动

### 48.1.4 应用场景

**医疗诊断**：
```
患者X光片 → 模型预测
  ├─ 高置信度："肺炎概率92%" → 直接治疗
  └─ 低置信度："可能是肺炎(45%)，也可能是正常(40%)" → 需要进一步检查
```

**自动驾驶**：
```
前方物体识别 → 不确定性评估
  ├─ 低不确定性："确定是行人" → 紧急刹车
  └─ 高不确定性："可能是行人，也可能是阴影" → 减速并鸣笛警告
```

**主动学习**：
```
大量未标注数据 → 不确定性排序
  ├─ 选择不确定性最高的样本
  └─ 人工标注这些样本，最大化学习效率
```

---

## 48.2 贝叶斯神经网络基础

### 48.2.1 从频率派到贝叶斯派

**频率派观点**：
- 模型参数 $\mathbf{w}$ 是固定但未知的常数
- 通过最大似然估计(MLE)找到最优参数：

$$
\mathbf{w}_{\text{MLE}} = \arg\max_{\mathbf{w}} p(\mathcal{D}|\mathbf{w})
$$

**贝叶斯派观点**：
- 模型参数 $\mathbf{w}$ 是随机变量，有概率分布
- 我们关心的是**后验分布**：

$$
p(\mathbf{w}|\mathcal{D}) = \frac{p(\mathcal{D}|\mathbf{w}) p(\mathbf{w})}{p(\mathcal{D})}
$$

**费曼法比喻**：
- **频率派**像一位**固执的工程师**："这个零件的寿命就是1000小时，我算出来的！"
- **贝叶斯派**像一位**谦逊的科学家**："根据现有数据，这个零件寿命在900-1100小时的概率是95%"

### 48.2.2 贝叶斯推断的核心公式

**贝叶斯定理**：

$$
\underbrace{p(\mathbf{w}|\mathcal{D})}_{\text{后验}} = \frac{\overbrace{p(\mathcal{D}|\mathbf{w})}^{\text{似然}} \overbrace{p(\mathbf{w})}^{\text{先验}}}{\underbrace{p(\mathcal{D})}_{\text{证据}}}
$$

其中：
- **先验** $p(\mathbf{w})$：在看到数据之前，我们对参数的初始信念
- **似然** $p(\mathcal{D}|\mathbf{w})$：给定参数时，观察到数据的概率
- **后验** $p(\mathbf{w}|\mathcal{D})$：在看到数据之后，更新了的参数信念
- **证据** $p(\mathcal{D})$：归一化常数（通常难以计算）

**预测分布**：

对于新输入 $\mathbf{x}^*$，贝叶斯神经网络不给出点估计，而是给出**预测分布**：

$$
p(y^*|\mathbf{x}^*, \mathcal{D}) = \int p(y^*|\mathbf{x}^*, \mathbf{w}) p(\mathbf{w}|\mathcal{D}) d\mathbf{w}
$$

**费曼法解释**：传统神经网络说"房价是100万"，贝叶斯神经网络说"房价在90-110万之间的概率是68%，在80-120万之间的概率是95%"。

### 48.2.3 变分推断 (Variational Inference)

**问题**：后验分布 $p(\mathbf{w}|\mathcal{D})$ 对于神经网络来说**难以计算**！

**解决方案**：变分推断用简单的分布 $q(\mathbf{w}|\theta)$ 来近似复杂的后验。

**优化目标**：最小化KL散度

$$
\theta^* = \arg\min_\theta \text{KL}(q(\mathbf{w}|\theta) || p(\mathbf{w}|\mathcal{D}))
$$

**推导变分下界 (ELBO)**：

$$
\begin{aligned}
\text{KL}(q(\mathbf{w}|\theta) || p(\mathbf{w}|\mathcal{D})) &= \mathbb{E}_q[\log q(\mathbf{w}|\theta) - \log p(\mathbf{w}|\mathcal{D})] \\
&= \mathbb{E}_q[\log q(\mathbf{w}|\theta) - \log p(\mathcal{D}|\mathbf{w}) - \log p(\mathbf{w})] + \log p(\mathcal{D})
\end{aligned}
$$

因此：

$$
\log p(\mathcal{D}) = \text{ELBO}(\theta) + \text{KL}(q(\mathbf{w}|\theta) || p(\mathbf{w}|\mathcal{D}))
$$

其中**证据下界 (ELBO)** 为：

$$
\boxed{\text{ELBO}(\theta) = \mathbb{E}_q[\log p(\mathcal{D}|\mathbf{w})] - \text{KL}(q(\mathbf{w}|\theta) || p(\mathbf{w}))}
$$

**直观理解**：
- **第一项**：模型拟合数据的程度（似然期望）
- **第二项**：近似分布与先验的接近程度（正则化）

**训练目标**：最大化ELBO，等价于最小化负ELBO：

$$
\mathcal{L}(\theta) = -\mathbb{E}_q[\log p(\mathcal{D}|\mathbf{w})] + \text{KL}(q(\mathbf{w}|\theta) || p(\mathbf{w}))
$$

### 48.2.4 Bayes by Backprop

Blundell等人(2015)提出了"Bayes by Backprop"，使用**重参数化技巧**来训练贝叶斯神经网络。

**重参数化技巧**：

假设 $q(\mathbf{w}|\theta) = \mathcal{N}(\mathbf{w}; \mu, \sigma^2)$，我们可以将权重表示为：

$$
\mathbf{w} = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)
$$

这样，梯度可以通过 $\mu$ 和 $\sigma$ 反向传播。

**完整算法**：

```
对于每个训练迭代：
  1. 从标准正态分布采样噪声 ε ~ N(0, I)
  2. 计算权重 w = μ + σ ⊙ ε
  3. 前向传播计算损失 L(w)
  4. 反向传播得到 ∂L/∂w
  5. 更新变分参数：
     μ ← μ - lr * ∂L/∂w
     σ ← σ - lr * ∂L/∂w * ε
```

---

## 48.3 MC-Dropout：贝叶斯的近似

### 48.3.1 核心洞察

Gal和Ghahramani(2016)发现了一个令人惊讶的事实：

> **在测试时保持Dropout开启，就相当于进行贝叶斯推断！**

**费曼法比喻**：
想象你在做一个重要决定，但你不确定哪个专家的意见最可靠。MC-Dropout就像是**询问同一个专家多次，但每次让他随机"忘记"一部分知识**。通过观察他答案的变化，你可以判断他对这个问题的**确定程度**。

### 48.3.2 数学原理

** dropout作为贝叶斯近似**：

标准的Dropout训练目标可以看作是在优化一个**变分下界**。

对于一层网络，Dropout相当于对权重进行随机掩码：

$$
\mathbf{w} = \mathbf{M} \odot \tilde{\mathbf{w}}
$$

其中 $\mathbf{M}_{ij} \sim \text{Bernoulli}(p)$，$p$ 是保留概率。

**关键证明**：

Gal和Ghahramani证明了，使用Dropout训练神经网络等价于在近似一个**深度高斯过程**的后验。

**预测时的不确定性估计**：

对于输入 $\mathbf{x}$，进行 $T$ 次前向传播（每次开启Dropout）：

$$
\hat{y}_t = f(\mathbf{x}; \mathbf{w}_t), \quad t = 1, ..., T
$$

**预测均值**：

$$
\bar{y} = \frac{1}{T} \sum_{t=1}^T \hat{y}_t
$$

**预测方差**（总不确定性）：

$$
\mathbb{V}[y] \approx \frac{1}{T} \sum_{t=1}^T \hat{y}_t^2 - \bar{y}^2
$$

### 48.3.3 偶然不确定性与认知不确定性的分离

对于回归问题，假设网络输出两个值：预测值 $f(\mathbf{x})$ 和噪声方差 $\sigma^2(\mathbf{x})$。

**偶然不确定性**（数据噪声）：

$$
\hat{\sigma}^2 = \frac{1}{T} \sum_{t=1}^T \sigma^2_t(\mathbf{x})
$$

**认知不确定性**（模型不确定性）：

$$
\hat{\sigma}^2_{\text{epistemic}} = \frac{1}{T} \sum_{t=1}^T f_t(\mathbf{x})^2 - \left(\frac{1}{T} \sum_{t=1}^T f_t(\mathbf{x})\right)^2
$$

**总不确定性**：

$$
\hat{\sigma}^2_{\text{total}} = \hat{\sigma}^2_{\text{aleatoric}} + \hat{\sigma}^2_{\text{epistemic}}
$$

### 48.3.4 MC-Dropout算法

```
训练阶段（标准Dropout训练）：
  1. 前向传播时随机丢弃神经元（概率p）
  2. 计算损失并反向传播
  3. 更新权重

测试阶段（MC采样）：
  输入: 测试样本x, 采样次数T
  对于t = 1到T:
    1. 开启Dropout（随机丢弃神经元）
    2. 前向传播得到预测 ŷ_t
    3. 记录预测结果
  
  计算:
    - 预测均值: ȳ = mean(ŷ_1, ..., ŷ_T)
    - 预测方差: σ² = var(ŷ_1, ..., ŷ_T)
  
  返回: ȳ, σ²
```

### 48.3.5 为什么MC-Dropout有效？

**直观解释**：

1. **每次Dropout对应不同的子网络**：$2^H$ 个可能的子网络（$H$是隐藏单元数）
2. **多次采样 ≈ 对多个模型进行贝叶斯平均**
3. **预测方差大** = 不同子网络给出不同答案 = **认知不确定性高**

**理论保证**：

在适当条件下，MC-Dropout的预测分布收敛到真实的贝叶斯后验预测分布。

---

## 48.4 深度集成 (Deep Ensembles)

### 48.4.1 基本原理

Lakshminarayanan等人(2017)提出了一个更简单但强大的方法：

> **训练多个独立模型，用它们的多样性来估计不确定性**

**费曼法比喻**：
想象你要预测明天的股市。与其相信一个"超级AI"，不如**询问10位不同背景的专家**：
- 技术分析师
- 基本面分析师
- 宏观经济专家
- ...

如果他们都说"会涨"，你很有信心。如果有人说涨、有人说跌，你就知道这个预测**不确定**。

### 48.4.2 算法流程

```
训练阶段：
  对于m = 1到M（集成大小）：
    1. 随机初始化网络权重
    2. 随机打乱训练数据顺序
    3. 独立训练网络直到收敛
    4. 保存模型

预测阶段：
  输入: 测试样本x
  对于m = 1到M:
    ŷ_m = f_m(x)  # 第m个模型的预测
  
  集成预测: ȳ = (1/M) Σ ŷ_m
  预测熵（分类）: H = -Σ_k p̄_k log p̄_k
  预测方差（回归）: σ² = (1/M) Σ (ŷ_m - ȳ)²
```

### 48.4.3 数学分析

**对于分类问题**：

每个模型输出概率分布：$\mathbf{p}_m = [p_{m1}, ..., p_{mK}]$

**平均预测**（预测分布）：

$$
\bar{\mathbf{p}} = \frac{1}{M} \sum_{m=1}^M \mathbf{p}_m
$$

**总不确定性**（预测熵）：

$$
H(\bar{\mathbf{p}}) = -\sum_{k=1}^K \bar{p}_k \log \bar{p}_k
$$

**知识不确定性**（平均单个模型的熵）：

$$
\bar{H} = \frac{1}{M} \sum_{m=1}^M H(\mathbf{p}_m)
$$

**数据不确定性**：

$$
H_{\text{data}} = \bar{H}
$$

**模型不确定性**：

$$
H_{\text{model}} = H(\bar{\mathbf{p}}) - \bar{H}
$$

**关键洞察**：
- **模型不确定性大** = 不同模型给出不同的预测（认知不确定性高）
- **数据不确定性大** = 每个模型都觉得"这个样本很难"（偶然不确定性高）

### 48.4.4 深度集成 vs MC-Dropout

| 特性 | MC-Dropout | 深度集成 |
|------|-----------|---------|
| **训练成本** | 低（单次训练） | 高（M次独立训练） |
| **推理成本** | M次前向传播 | M次前向传播 |
| **实现难度** | 简单（几行代码） | 中等（需管理多个模型） |
| **性能** | 好 | 更好（通常SOTA） |
| **理论基础** | 贝叶斯近似 | 贝叶斯模型平均近似 |

**推荐实践**：
- **快速原型** → MC-Dropout
- **生产环境** → Deep Ensembles（通常M=5-10足够）

---

## 48.5 证据深度学习 (Evidential Deep Learning)

### 48.5.1 从预测到证据

传统神经网络预测**概率分布的参数**（如分类的概率）。

证据深度学习更进一步，预测**概率分布的分布的参数**！

**费曼法比喻**：
- **传统神经网络**："我认为明天80%会下雨"
- **证据深度学习**："根据我掌握的证据，明天下雨的概率服从Beta(8, 2)分布"

### 48.5.2 分类问题：Dirichlet分布

对于K类分类问题，Softmax输出概率 $\mathbf{p} = [p_1, ..., p_K]$。

**Dirichlet分布**是类别分布的共轭先验：

$$
\text{Dir}(\mathbf{p}|\boldsymbol{\alpha}) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{k=1}^K p_k^{\alpha_k - 1}
$$

其中：
- $\boldsymbol{\alpha} = [\alpha_1, ..., \alpha_K]$ 是浓度参数
- $\alpha_k > 0$ 可以看作对类别k的"伪计数"
- $B(\boldsymbol{\alpha})$ 是Beta函数

**关键性质**：
- **期望概率**：$\mathbb{E}[p_k] = \frac{\alpha_k}{\alpha_0}$，其中 $\alpha_0 = \sum_{k=1}^K \alpha_k$（总证据强度）
- **方差**：随$\alpha_0$增大而减小
- **总证据强度** $\alpha_0$ 越大，模型越**确定**

**证据深度学习的目标**：

神经网络输出Dirichlet参数 $\boldsymbol{\alpha}$ 而不是直接输出概率。

### 48.5.3 损失函数推导

**最大似然目标**：

给定真实标签 $\mathbf{y}$（one-hot编码），我们希望最大化：

$$
p(\mathbf{y}|\boldsymbol{\alpha}) = \int p(\mathbf{y}|\mathbf{p}) \text{Dir}(\mathbf{p}|\boldsymbol{\alpha}) d\mathbf{p}
$$

由于 $p(\mathbf{y}|\mathbf{p}) = \prod_{k=1}^K p_k^{y_k}$，这个积分有解析解：

$$
p(\mathbf{y}|\boldsymbol{\alpha}) = \frac{\alpha_y}{\alpha_0}
$$

**负面对数似然**：

$$
\mathcal{L}_{\text{NLL}} = -\log \frac{\alpha_y}{\alpha_0} = \log \alpha_0 - \log \alpha_y
$$

**正则化项**（防止过拟合）：

$$
\mathcal{L}_{\text{reg}} = \text{KL}(\text{Dir}(\mathbf{p}|\boldsymbol{\alpha}) || \text{Dir}(\mathbf{p}|\mathbf{1}))
$$

其中 $\mathbf{1} = [1, ..., 1]$ 是均匀Dirichlet先验。

**总损失**（Sensoy等人, 2018）：

$$
\boxed{\mathcal{L} = \sum_{i=1}^N \left(\log \alpha_0^{(i)} - \log \alpha_{y_i}^{(i)}\right) + \lambda \sum_{i=1}^N \text{KL}(\text{Dir}(\mathbf{p}|\boldsymbol{\alpha}^{(i)}) || \text{Dir}(\mathbf{p}|\mathbf{1}))}
$$

### 48.5.4 不确定性度量

给定预测的Dirichlet参数 $\boldsymbol{\alpha}$：

**期望概率**：

$$
\hat{p}_k = \frac{\alpha_k}{\alpha_0}
$$

**预测方差**：

$$
\mathbb{V}[p_k] = \frac{\hat{p}_k(1 - \hat{p}_k)}{\alpha_0 + 1}
$$

**总不确定性**（预测熵）：

$$
H = -\sum_{k=1}^K \hat{p}_k \log \hat{p}_k
$$

**数据不确定性**（期望熵）：

$$
\mathbb{E}_{p \sim \text{Dir}}[H(p)]
$$

**认知不确定性**（互信息）：

$$
I[y, \mathbf{p}|\mathbf{x}, \mathcal{D}] = H - \mathbb{E}[H(p)]
$$

**直观理解**：
- **总证据强度** $\alpha_0$ 小 → 认知不确定性高（证据不足）
- **概率接近均匀** → 数据不确定性高（样本本身模糊）

### 48.5.5 回归问题：Normal-Inverse-Gamma分布

对于回归问题，Amini等人(2020)使用Normal-Inverse-Gamma (NIG) 分布。

**NIG分布**是高斯分布的共轭先验，参数为 $(\gamma, \nu, \alpha, \beta)$：

$$
p(\mu, \sigma^2|\gamma, \nu, \alpha, \beta) = \text{NIG}(\mu, \sigma^2; \gamma, \nu, \alpha, \beta)
$$

**神经网络输出**四个参数：
- $\gamma$：预测均值
- $\nu > 0$：与均值精度相关的参数
- $\alpha > 1$：形状参数
- $\beta > 0$：尺度参数

**预测分布**（Student-t分布）：

$$
p(y|\mathbf{x}) = \mathcal{T}(y; \gamma, \frac{\beta(1+\nu)}{\nu\alpha}, 2\alpha)
$$

**不确定性分解**：

$$
\underbrace{\mathbb{V}[y]}_{\text{总不确定性}} = \underbrace{\frac{\beta}{\alpha - 1}}_{\text{偶然不确定性}} \cdot \underbrace{\left(1 + \frac{1}{\nu}\right)}_{\text{认知不确定性因子}}
$$

**损失函数**（Deep Evidential Regression）：

$$
\mathcal{L}(\theta) = \underbrace{\frac{1}{2} \log \left(\frac{\pi}{\nu}\right) - \alpha \log(2\beta) + \left(\alpha + \frac{1}{2}\right) \log \left((y - \gamma)^2 \nu + 2\beta\right) + \log \frac{\Gamma(\alpha)}{\Gamma(\alpha + 1/2)}}_{\text{负对数似然}} + \lambda |y - \gamma|
$$

---

## 48.6 完整代码实现

### 48.6.1 MC-Dropout实现

```python
"""
MC-Dropout不确定性估计实现
基于Gal & Ghahramani (2016)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class MCDropoutNet(nn.Module):
    """支持MC-Dropout的神经网络"""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list = [128, 64],
        output_dim: int = 1,
        dropout_rate: float = 0.1,
        task_type: str = 'regression'
    ):
        super().__init__()
        self.task_type = task_type
        self.dropout_rate = dropout_rate
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)  # 关键：使用Dropout层
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # 输出层
        if task_type == 'regression':
            # 回归：输出均值和方差
            self.mean_layer = nn.Linear(prev_dim, output_dim)
            self.var_layer = nn.Linear(prev_dim, output_dim)
        else:
            # 分类：输出logits
            self.logit_layer = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x: torch.Tensor, dropout: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            dropout: 是否启用Dropout（测试时设为True用于MC采样）
        
        Returns:
            预测输出
        """
        features = self.feature_layers(x)
        
        if self.task_type == 'regression':
            mean = self.mean_layer(features)
            # 使用softplus确保方差为正
            var = F.softplus(self.var_layer(features)) + 1e-6
            return torch.cat([mean, var], dim=-1)
        else:
            return self.logit_layer(features)
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MC-Dropout预测，返回均值、偶然不确定性和认知不确定性
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            n_samples: MC采样次数
        
        Returns:
            mean: 预测均值 [batch_size, output_dim]
            aleatoric_unc: 偶然不确定性 [batch_size, output_dim]
            epistemic_unc: 认知不确定性 [batch_size, output_dim]
        """
        self.train()  # 关键：保持train模式以启用Dropout
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                if self.task_type == 'regression':
                    output = self.forward(x, dropout=True)
                    pred_mean = output[:, :output.shape[1]//2]
                    predictions.append(pred_mean)
                else:
                    logits = self.forward(x, dropout=True)
                    probs = F.softmax(logits, dim=-1)
                    predictions.append(probs)
        
        predictions = torch.stack(predictions)  # [n_samples, batch_size, output_dim]
        
        # 计算统计量
        pred_mean = predictions.mean(dim=0)
        pred_var = predictions.var(dim=0)
        
        if self.task_type == 'regression':
            # 偶然不确定性：预测方差的平均值
            aleatoric_unc = predictions.var(dim=0, unbiased=False).mean(dim=0, keepdim=True).T
            # 认知不确定性：预测均值的方差
            epistemic_unc = pred_var
        else:
            # 分类：使用预测熵
            aleatoric_unc = None
            epistemic_unc = pred_var
        
        self.eval()
        return pred_mean, aleatoric_unc, epistemic_unc


class MCDropoutTrainer:
    """MC-Dropout模型训练器"""
    
    def __init__(
        self, 
        model: MCDropoutNet,
        lr: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """计算损失函数"""
        if self.model.task_type == 'regression':
            # 分离均值和方差
            pred_mean = predictions[:, :predictions.shape[1]//2]
            pred_var = predictions[:, predictions.shape[1]//2:]
            
            # 负对数似然（考虑异方差噪声）
            nll = 0.5 * torch.log(pred_var) + 0.5 * (targets - pred_mean)**2 / pred_var
            return nll.mean()
        else:
            # 交叉熵损失
            return F.cross_entropy(predictions, targets)
    
    def train_epoch(self, train_loader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            self.optimizer.zero_grad()
            
            predictions = self.model(batch_x, dropout=True)
            loss = self.compute_loss(predictions, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
```

### 48.6.2 深度集成实现

```python
"""
深度集成不确定性估计实现
基于Lakshminarayanan et al. (2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from copy import deepcopy


class EnsembleNet(nn.Module):
    """单个集成成员网络"""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list = [128, 64],
        output_dim: int = 1,
        task_type: str = 'regression'
    ):
        super().__init__()
        self.task_type = task_type
        
        # 构建网络
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        if task_type == 'regression':
            self.mean_layer = nn.Linear(prev_dim, output_dim)
            self.var_layer = nn.Linear(prev_dim, output_dim)
        else:
            self.logit_layer = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(x)
        
        if self.task_type == 'regression':
            mean = self.mean_layer(features)
            var = F.softplus(self.var_layer(features)) + 1e-6
            return torch.cat([mean, var], dim=-1)
        else:
            return self.logit_layer(features)


class DeepEnsemble:
    """深度集成模型"""
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: list = [128, 64],
        output_dim: int = 1,
        n_models: int = 5,
        task_type: str = 'regression'
    ):
        self.n_models = n_models
        self.task_type = task_type
        
        # 创建多个独立模型
        self.models = nn.ModuleList([
            EnsembleNet(input_dim, hidden_dims, output_dim, task_type)
            for _ in range(n_models)
        ])
    
    def fit(
        self, 
        train_loader, 
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        """训练所有集成成员"""
        for i, model in enumerate(self.models):
            print(f"训练集成成员 {i+1}/{self.n_models}...")
            
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
            
            for epoch in range(epochs):
                model.train()
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    
                    predictions = model(batch_x)
                    
                    if self.task_type == 'regression':
                        pred_mean = predictions[:, :predictions.shape[1]//2]
                        pred_var = predictions[:, predictions.shape[1]//2:]
                        loss = 0.5 * torch.log(pred_var) + \
                               0.5 * (batch_y - pred_mean)**2 / pred_var
                        loss = loss.mean()
                    else:
                        loss = F.cross_entropy(predictions, batch_y)
                    
                    loss.backward()
                    optimizer.step()
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        集成预测与不确定性估计
        
        Returns:
            mean: 集成预测均值
            data_uncertainty: 数据不确定性（偶然不确定性）
            model_uncertainty: 模型不确定性（认知不确定性）
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                
                if self.task_type == 'regression':
                    pred_mean = output[:, :output.shape[1]//2]
                    predictions.append(pred_mean)
                else:
                    probs = F.softmax(output, dim=-1)
                    predictions.append(probs)
        
        predictions = torch.stack(predictions)  # [n_models, batch_size, output_dim]
        
        # 集成均值
        ensemble_mean = predictions.mean(dim=0)
        
        # 总不确定性
        total_unc = predictions.var(dim=0)
        
        if self.task_type == 'regression':
            # 对于回归，模型不确定性是预测均值的方差
            model_uncertainty = total_unc
            # 数据不确定性需要额外计算
            data_uncertainty = None
        else:
            # 对于分类，使用熵分解
            mean_pred = ensemble_mean
            total_entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10), dim=-1)
            
            # 平均单个模型的熵
            individual_entropies = -torch.sum(
                predictions * torch.log(predictions + 1e-10), 
                dim=-1
            ).mean(dim=0)
            
            model_uncertainty = total_entropy - individual_entropies
            data_uncertainty = individual_entropies
        
        return ensemble_mean, data_uncertainty, model_uncertainty
```

### 48.6.3 证据深度学习实现

```python
"""
证据深度学习(EDL)实现
基于Sensoy et al. (2018) for Classification
基于Amini et al. (2020) for Regression
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def dirichlet_loss(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算Dirichlet损失函数
    
    Args:
        alpha: Dirichlet参数 [batch_size, num_classes]
        y: 真实标签（one-hot） [batch_size, num_classes]
    
    Returns:
        损失值
    """
    # 总证据强度
    alpha_0 = alpha.sum(dim=1, keepdim=True)
    
    # 负面对数似然
    nll = torch.lgamma(alpha_0) - torch.lgamma(alpha).sum(dim=1, keepdim=True) + \
          ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(alpha_0))).sum(dim=1, keepdim=True)
    
    # KL散度正则化（相对于均匀先验）
    beta = torch.ones_like(alpha)
    kl = torch.lgamma(alpha_0) - torch.lgamma(alpha).sum(dim=1, keepdim=True) + \
         torch.lgamma(beta.sum(dim=1, keepdim=True)) - torch.lgamma(beta).sum(dim=1, keepdim=True) + \
         ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(alpha_0))).sum(dim=1, keepdim=True)
    
    loss = (nll + kl).mean()
    return loss


class EvidentialClassificationNet(nn.Module):
    """证据深度学习分类网络"""
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: list = [128, 64],
        num_classes: int = 10
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        self.alpha_layer = nn.Linear(prev_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，输出Dirichlet参数
        
        Returns:
            alpha: Dirichlet浓度参数 [batch_size, num_classes]
        """
        features = self.feature_layers(x)
        # 使用softplus确保alpha > 1（有证据）
        alpha = F.softplus(self.alpha_layer(features)) + 1.0
        return alpha
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测并计算不确定性
        
        Returns:
            probs: 期望概率 [batch_size, num_classes]
            total_uncertainty: 总不确定性（预测熵）
            vacuity: 认知不确定性（基于证据强度）
        """
        alpha = self.forward(x)
        alpha_0 = alpha.sum(dim=1, keepdim=True)
        
        # 期望概率
        probs = alpha / alpha_0
        
        # 总不确定性（预测熵）
        total_uncertainty = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        
        # 认知不确定性：证据越少，不确定性越高
        # vacuity = K / alpha_0，其中K是类别数
        num_classes = alpha.shape[1]
        vacuity = num_classes / alpha_0.squeeze()
        
        return probs, total_uncertainty, vacuity


class NIGLoss(nn.Module):
    """Normal-Inverse-Gamma损失函数（用于回归）"""
    
    def __init__(self, lambda_reg: float = 0.01):
        super().__init__()
        self.lambda_reg = lambda_reg
    
    def forward(
        self, 
        gamma: torch.Tensor, 
        nu: torch.Tensor, 
        alpha: torch.Tensor, 
        beta: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        计算NIG损失
        
        Args:
            gamma: 预测均值 [batch_size, 1]
            nu: 精度参数 [batch_size, 1]
            alpha: 形状参数 [batch_size, 1]
            beta: 尺度参数 [batch_size, 1]
            y: 真实值 [batch_size, 1]
        """
        # 确保参数有效
        nu = F.softplus(nu) + 1e-6
        alpha = F.softplus(alpha) + 1.01  # alpha > 1
        beta = F.softplus(beta) + 1e-6
        
        # NLL损失
        omega = 2 * beta * (1 + nu)
        nll = 0.5 * torch.log(np.pi / nu) - alpha * torch.log(2 * beta) + \
              (alpha + 0.5) * torch.log((y - gamma)**2 * nu + omega) + \
              torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        
        # 正则化项
        reg = torch.abs(y - gamma)
        
        return (nll + self.lambda_reg * reg).mean()


class EvidentialRegressionNet(nn.Module):
    """证据深度学习回归网络"""
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: list = [128, 64]
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # 输出NIG的四个参数
        self.gamma_layer = nn.Linear(prev_dim, 1)  # 均值
        self.nu_layer = nn.Linear(prev_dim, 1)     # 精度
        self.alpha_layer = nn.Linear(prev_dim, 1)  # 形状
        self.beta_layer = nn.Linear(prev_dim, 1)   # 尺度
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        前向传播，输出NIG参数
        
        Returns:
            gamma, nu, alpha, beta: NIG分布参数
        """
        features = self.feature_layers(x)
        
        gamma = self.gamma_layer(features)
        nu = F.softplus(self.nu_layer(features)) + 1e-6
        alpha = F.softplus(self.alpha_layer(features)) + 1.01
        beta = F.softplus(self.beta_layer(features)) + 1e-6
        
        return gamma, nu, alpha, beta
    
    def predict(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测并计算不确定性
        
        Returns:
            pred_mean: 预测均值
            aleatoric: 偶然不确定性
            epistemic: 认知不确定性
        """
        gamma, nu, alpha, beta = self.forward(x)
        
        # 预测均值
        pred_mean = gamma
        
        # 偶然不确定性：数据噪声
        aleatoric = beta / (alpha - 1)
        
        # 认知不确定性：模型不确定性
        epistemic = beta / (nu * (alpha - 1))
        
        return pred_mean, aleatoric, epistemic


class EvidentialTrainer:
    """证据深度学习训练器"""
    
    def __init__(
        self, 
        model: nn.Module,
        task_type: str = 'classification',
        lr: float = 1e-3
    ):
        self.model = model
        self.task_type = task_type
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        if task_type == 'regression':
            self.criterion = NIGLoss()
    
    def train_epoch(self, train_loader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            self.optimizer.zero_grad()
            
            if self.task_type == 'classification':
                alpha = self.model(batch_x)
                # 将标签转为one-hot
                y_onehot = F.one_hot(batch_y, num_classes=alpha.shape[1]).float()
                loss = dirichlet_loss(alpha, y_onehot)
            else:
                gamma, nu, alpha, beta = self.model(batch_x)
                loss = self.criterion(gamma, nu, alpha, beta, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
```

### 48.6.4 可视化工具

```python
"""
不确定性可视化工具
"""
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_uncertainty_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    model,
    title: str = "Uncertainty Estimation"
):
    """可视化回归任务的不确定性"""
    
    # 预测
    X_test_tensor = torch.FloatTensor(X_test)
    
    if hasattr(model, 'predict_with_uncertainty'):
        mean, aleatoric, epistemic = model.predict_with_uncertainty(X_test_tensor)
        mean = mean.numpy()
        total_unc = np.sqrt(aleatoric.numpy() + epistemic.numpy()) if aleatoric is not None else np.sqrt(epistemic.numpy())
    else:
        mean, aleatoric, epistemic = model.predict(X_test_tensor)
        mean = mean.detach().numpy()
        total_unc = np.sqrt(aleatoric.detach().numpy() + epistemic.detach().numpy())
    
    # 绘制
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 训练数据
    axes[0].scatter(X_train, y_train, c='red', alpha=0.5, label='Training data')
    axes[0].plot(X_test, mean, 'b-', label='Prediction')
    axes[0].fill_between(
        X_test.flatten(), 
        (mean - 2*total_unc).flatten(), 
        (mean + 2*total_unc).flatten(),
        alpha=0.3, label='95% confidence'
    )
    axes[0].set_title('Total Uncertainty')
    axes[0].legend()
    
    # 偶然不确定性
    if aleatoric is not None:
        axes[1].plot(X_test, aleatoric, 'g-', label='Aleatoric')
        axes[1].set_title('Aleatoric Uncertainty')
        axes[1].legend()
    
    # 认知不确定性
    axes[2].plot(X_test, epistemic, 'm-', label='Epistemic')
    axes[2].set_title('Epistemic Uncertainty')
    axes[2].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_ood_detection(
    in_distribution_unc: np.ndarray,
    ood_unc: np.ndarray,
    title: str = "OOD Detection"
):
    """可视化OOD检测结果"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 直方图
    axes[0].hist(in_distribution_unc, bins=50, alpha=0.7, label='In-distribution')
    axes[0].hist(ood_unc, bins=50, alpha=0.7, label='OOD')
    axes[0].set_xlabel('Uncertainty')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Uncertainty Distribution')
    axes[0].legend()
    
    # ROC曲线（简化的）
    all_unc = np.concatenate([in_distribution_unc, ood_unc])
    labels = np.concatenate([
        np.zeros(len(in_distribution_unc)),
        np.ones(len(ood_unc))
    ])
    
    # 计算AUC
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(labels, all_unc)
    roc_auc = auc(fpr, tpr)
    
    axes[1].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig
```

---

## 48.7 应用场景

### 48.7.1 医疗诊断中的不确定性

```python
"""
医疗影像诊断的不确定性量化示例
"""
import torch
import torch.nn as nn
from torchvision import models


class MedicalImageClassifier(nn.Module):
    """带不确定性的医学影像分类器"""
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5):
        super().__init__()
        
        # 使用预训练的ResNet
        self.backbone = models.resnet18(pretrained=True)
        
        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, dropout: bool = False):
        if dropout:
            self.train()  # 启用Dropout
        else:
            self.eval()
        return self.backbone(x)


def medical_diagnosis_with_uncertainty(
    image: torch.Tensor,
    model: MedicalImageClassifier,
    n_samples: int = 50,
    uncertainty_threshold: float = 0.5
) -> dict:
    """
    带不确定性的医疗诊断
    
    Returns:
        包含诊断结果和不确定性的字典
    """
    # MC采样
    predictions = []
    for _ in range(n_samples):
        with torch.no_grad():
            logits = model(image, dropout=True)
            probs = torch.softmax(logits, dim=-1)
            predictions.append(probs)
    
    predictions = torch.stack(predictions)
    
    # 计算统计量
    mean_probs = predictions.mean(dim=0)
    pred_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)
    
    # 互信息（认知不确定性）
    individual_entropies = -torch.sum(
        predictions * torch.log(predictions + 1e-10), dim=-1
    ).mean(dim=0)
    epistemic_unc = pred_entropy - individual_entropies
    
    # 决策
    pred_class = mean_probs.argmax(dim=-1)
    confidence = mean_probs.max(dim=-1).values
    
    # 生成报告
    if epistemic_unc > uncertainty_threshold:
        recommendation = "建议进一步检查或专家会诊"
    elif confidence < 0.8:
        recommendation = "建议复查"
    else:
        recommendation = "诊断可信"
    
    return {
        'diagnosis': '阳性' if pred_class.item() == 1 else '阴性',
        'confidence': confidence.item(),
        'epistemic_uncertainty': epistemic_unc.item(),
        'predictive_entropy': pred_entropy.item(),
        'recommendation': recommendation
    }
```

### 48.7.2 自动驾驶安全决策

```python
"""
自动驾驶中的不确定性感知决策
"""
import torch
import numpy as np


class UncertaintyAwareAutopilot:
    """不确定性感知的自动驾驶决策系统"""
    
    def __init__(
        self,
        perception_model,
        uncertainty_threshold_high: float = 0.7,
        uncertainty_threshold_medium: float = 0.4
    ):
        self.perception_model = perception_model
        self.unc_high = uncertainty_threshold_high
        self.unc_medium = uncertainty_threshold_medium
    
    def make_decision(
        self,
        sensor_data: torch.Tensor,
        current_speed: float
    ) -> dict:
        """
        基于不确定性的驾驶决策
        """
        # 获取感知结果和不确定性
        detections, uncertainties = self.perceive(sensor_data)
        
        max_uncertainty = max(uncertainties.values()) if uncertainties else 0
        
        # 不确定性驱动的决策
        if max_uncertainty > self.unc_high:
            return {
                'action': 'EMERGENCY_STOP',
                'reason': f'高不确定性({max_uncertainty:.2f})，安全优先',
                'speed_target': 0,
                'alert_level': 'CRITICAL'
            }
        elif max_uncertainty > self.unc_medium:
            return {
                'action': 'REDUCE_SPEED',
                'reason': f'中等不确定性({max_uncertainty:.2f})，谨慎驾驶',
                'speed_target': current_speed * 0.5,
                'alert_level': 'WARNING'
            }
        else:
            return {
                'action': 'NORMAL_DRIVE',
                'reason': '低不确定性，正常驾驶',
                'speed_target': current_speed,
                'alert_level': 'NORMAL'
            }
    
    def perceive(self, sensor_data: torch.Tensor):
        """感知环境并估计不确定性"""
        # 使用MC-Dropout或Deep Ensemble
        mean_pred, data_unc, model_unc = self.perception_model.predict_with_uncertainty(
            sensor_data
        )
        
        detections = {
            'vehicles': mean_pred['vehicles'],
            'pedestrians': mean_pred['pedestrians'],
            'traffic_signs': mean_pred['traffic_signs']
        }
        
        uncertainties = {
            'vehicles': model_unc['vehicles'].mean().item(),
            'pedestrians': model_unc['pedestrians'].mean().item(),
            'traffic_signs': model_unc['traffic_signs'].mean().item()
        }
        
        return detections, uncertainties
```

### 48.7.3 主动学习样本选择

```python
"""
基于不确定性的主动学习
"""
import torch
import numpy as np
from typing import List, Tuple


class UncertaintySampler:
    """基于不确定性的主动学习采样器"""
    
    def __init__(self, strategy: str = 'entropy'):
        """
        Args:
            strategy: 采样策略 ('entropy', 'margin', 'random')
        """
        self.strategy = strategy
    
    def select_samples(
        self,
        model,
        unlabeled_data: torch.Tensor,
        n_samples: int
    ) -> List[int]:
        """
        选择最有价值的样本进行标注
        
        Args:
            model: 训练好的模型
            unlabeled_data: 未标注数据 [N, ...]
            n_samples: 需要选择的样本数
        
        Returns:
            选中样本的索引列表
        """
        # 获取不确定性估计
        uncertainties = self.compute_uncertainty(model, unlabeled_data)
        
        if self.strategy == 'entropy':
            # 选择熵最大的样本（最不确定的）
            selected_indices = uncertainties.argsort(descending=True)[:n_samples]
        elif self.strategy == 'margin':
            # 选择置信度最低的样本
            selected_indices = uncertainties.argsort()[:n_samples]
        else:  # random
            selected_indices = torch.randperm(len(unlabeled_data))[:n_samples]
        
        return selected_indices.tolist()
    
    def compute_uncertainty(
        self,
        model,
        data: torch.Tensor
    ) -> torch.Tensor:
        """计算数据的不确定性"""
        
        # 使用MC-Dropout获取预测分布
        model.train()  # 启用Dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(20):  # 20次采样
                output = model(data)
                if len(output.shape) > 1 and output.shape[1] > 1:
                    # 分类任务
                    probs = torch.softmax(output, dim=-1)
                    predictions.append(probs)
                else:
                    # 回归任务
                    predictions.append(output)
        
        predictions = torch.stack(predictions)  # [n_samples, batch_size, num_classes]
        
        if predictions.shape[-1] > 1:
            # 分类：使用预测熵
            mean_pred = predictions.mean(dim=0)
            entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10), dim=-1)
            return entropy
        else:
            # 回归：使用预测方差
            return predictions.var(dim=0).squeeze()


class ActiveLearningLoop:
    """主动学习循环"""
    
    def __init__(
        self,
        model,
        trainer,
        sampler: UncertaintySampler,
        initial_labeled_size: int = 100,
        budget_per_iteration: int = 50,
        max_iterations: int = 10
    ):
        self.model = model
        self.trainer = trainer
        self.sampler = sampler
        self.initial_size = initial_labeled_size
        self.budget = budget_per_iteration
        self.max_iter = max_iterations
    
    def run(
        self,
        full_train_data: torch.Tensor,
        full_train_labels: torch.Tensor,
        test_data: torch.Tensor,
        test_labels: torch.Tensor
    ) -> List[dict]:
        """
        执行主动学习循环
        """
        results = []
        
        # 初始随机采样
        labeled_indices = torch.randperm(len(full_train_data))[:self.initial_size].tolist()
        unlabeled_indices = list(set(range(len(full_train_data))) - set(labeled_indices))
        
        for iteration in range(self.max_iter):
            print(f"\n=== Active Learning Iteration {iteration + 1}/{self.max_iter} ===")
            print(f"Labeled samples: {len(labeled_indices)}")
            
            # 在已标注数据上训练
            labeled_data = full_train_data[labeled_indices]
            labeled_labels = full_train_labels[labeled_indices]
            
            self.trainer.fit(labeled_data, labeled_labels)
            
            # 评估
            test_acc = self.trainer.evaluate(test_data, test_labels)
            results.append({
                'iteration': iteration,
                'n_labeled': len(labeled_indices),
                'test_accuracy': test_acc
            })
            
            print(f"Test accuracy: {test_acc:.4f}")
            
            # 选择新样本
            if len(unlabeled_indices) > 0:
                unlabeled_data = full_train_data[unlabeled_indices]
                selected_relative = self.sampler.select_samples(
                    self.model, unlabeled_data, self.budget
                )
                selected_absolute = [unlabeled_indices[i] for i in selected_relative]
                
                # 更新索引
                labeled_indices.extend(selected_absolute)
                unlabeled_indices = list(set(unlabeled_indices) - set(selected_absolute))
        
        return results
```

---

## 48.8 练习题

### 基础题

**48.1 概念理解**
解释以下概念的区别：
1. 偶然不确定性 vs 认知不确定性
2. 贝叶斯神经网络 vs 频率派神经网络
3. MC-Dropout vs 深度集成
4. 证据深度学习中的Dirichlet分布有什么特殊意义？

**48.2 贝叶斯定理应用**

假设你正在开发一个疾病诊断AI：
- 疾病在人群中的患病率（先验）：$p(D) = 0.01$
- 如果患病，测试阳性的概率：$p(T|D) = 0.95$
- 如果未患病，测试阳性的概率：$p(T|\neg D) = 0.05$

如果一个患者测试结果为阳性，他实际患病的概率是多少？

提示：使用贝叶斯定理计算后验概率 $p(D|T)$。

**48.3 MC-Dropout分析**

一个神经网络在MC-Dropout采样中得到以下预测结果（10次采样）：
```
[0.82, 0.78, 0.85, 0.80, 0.83, 0.79, 0.81, 0.84, 0.77, 0.86]
```

计算：
1. 预测均值
2. 认知不确定性（预测方差）
3. 如果这个模型还输出了偶然不确定性方差为0.02，总不确定性是多少？

### 进阶题

**48.4 不确定性分解推导**

证明对于回归问题，总不确定性可以分解为偶然不确定性和认知不确定性之和：

$$
\mathbb{V}[y] = \mathbb{E}_{p(\mathbf{w}|\mathcal{D})}[\sigma^2(\mathbf{x})] + \mathbb{V}_{p(\mathbf{w}|\mathcal{D})}[f^{\mathbf{w}}(\mathbf{x})]
$$

提示：使用条件方差公式（Law of Total Variance）。

**48.5 EDL损失函数分析**

考虑证据深度学习的损失函数：

$$
\mathcal{L} = \sum_{i=1}^N \left(\log \alpha_0^{(i)} - \log \alpha_{y_i}^{(i)}\right) + \lambda \sum_{i=1}^N \text{KL}(\text{Dir}(\mathbf{p}|\boldsymbol{\alpha}^{(i)}) || \text{Dir}(\mathbf{p}|\mathbf{1}))
$$

分析：
1. 当模型对某个样本非常确定时（某个 $\alpha_k$ 很大），第一项的值是多少？
2. 当模型对所有类别都没有证据时（所有 $\alpha_k = 1$），KL散度项的值是多少？
3. 超参数 $\lambda$ 增大会有什么效果？

**48.6 OOD检测设计**

设计一个基于不确定性的OOD（分布外）检测系统：

要求：
1. 使用MC-Dropout估计不确定性
2. 设定一个不确定性阈值来判断OOD
3. 在MNIST训练集上训练，在Fashion-MNIST上测试OOD检测性能
4. 计算AUROC评估检测性能

### 挑战题

**48.7 多任务不确定性**

在多任务学习中（如同时预测深度、语义分割、边界检测），不同任务有不同的不确定性水平。

Kendall等人(2018)提出了多任务不确定性：

$$
\mathcal{L}(\mathbf{W}, \sigma_1, \sigma_2) = \frac{1}{2\sigma_1^2} \mathcal{L}_1(\mathbf{W}) + \frac{1}{2\sigma_2^2} \mathcal{L}_2(\mathbf{W}) + \log \sigma_1 \sigma_2
$$

其中 $\sigma_1, \sigma_2$ 是可学习的任务不确定性。

问题：
1. 解释这个损失函数的直观意义
2. 为什么 $\sigma$ 越大，对应任务的损失权重越小？
3. 实现一个多任务网络，使用这种方法自动平衡两个回归任务

**48.8 贝叶斯优化集成**

结合贝叶斯优化和不确定性量化：

设计一个系统，使用：
1. 高斯过程作为代理模型
2. 采集函数考虑预测不确定性
3. 深度集成提供不确定性估计
4. 在一个高维优化问题上测试（如超参数调优）

**48.9 理论分析**

考虑一个简单的一维回归问题：$y = \sin(x) + \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, 0.1^2)$。

在区间 $[-\pi, \pi]$ 上均匀采样20个训练点，使用MC-Dropout神经网络拟合。

分析并可视化：
1. 在训练区域内（内插）的不确定性
2. 在训练区域外（外推，如 $[\pi, 2\pi]$）的不确定性
3. 理论上，认知不确定性在数据稀疏区域应该更高，你的实验结果是否符合预期？
4. 如何量化这种"远离训练数据"的检测能力？

---

## 48.9 参考文献

1. Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight uncertainty in neural network. *International Conference on Machine Learning* (pp. 1613-1622). PMLR.

2. Gal, Y., & Ghahramani, Z. (2016). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. *International Conference on Machine Learning* (pp. 1050-1059). PMLR.

3. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems*, 30.

4. Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential deep learning to quantify classification uncertainty. *Advances in Neural Information Processing Systems*, 31.

5. Amini, A., Schwarting, W., Soleimany, A., & Rus, D. (2020). Deep evidential regression. *Advances in Neural Information Processing Systems*, 33, 14927-14937.

6. Kendall, A., & Gal, Y. (2017). What uncertainties do we need in bayesian deep learning for computer vision? *Advances in Neural Information Processing Systems*, 30.

7. Malinin, A., & Gales, M. (2018). Predictive uncertainty estimation via prior networks. *Advances in Neural Information Processing Systems*, 31.

8. Charpentier, B., Zügner, D., & Günnemann, S. (2020). Posterior network: Uncertainty estimation without OOD samples via density-based pseudo-counts. *International Conference on Learning Representations*.

9. Mucsányi, B., Seong, J., Kim, S., Lee, H. J., & Seo, S. H. (2024). Faithful explainability of uncertainty in machine learning. *arXiv preprint arXiv:2401.06521*.

10. Hüllermeier, E., & Waegeman, W. (2021). Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods. *Machine Learning*, 110(3), 457-506.

---

*本章完*

> **本章核心思想**：不确定性不是bug，而是feature。一个优秀的AI系统不仅要给出答案，还要知道"我有多确定这个答案"。贝叶斯深度学习让我们从"盲目自信"走向"知之为知之，不知为不知"的智慧。

**关键公式速查**：
- 不确定性分解：$\mathbb{V}[y] = \mathbb{E}[\sigma^2] + \mathbb{V}[f^{\mathbf{w}}(\mathbf{x})]$
- 贝叶斯定理：$p(\mathbf{w}|\mathcal{D}) \propto p(\mathcal{D}|\mathbf{w})p(\mathbf{w})$
- ELBO：$\mathbb{E}_q[\log p(\mathcal{D}|\mathbf{w})] - \text{KL}(q(\mathbf{w}) || p(\mathbf{w}))$
- Dirichlet期望：$\mathbb{E}[p_k] = \frac{\alpha_k}{\alpha_0}$
- NIG不确定性：$\mathbb{V}[y] = \frac{\beta}{\alpha-1} \cdot (1 + \frac{1}{\nu})$
