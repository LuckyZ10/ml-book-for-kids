# 第二十一章 正则化——防止网络"死记硬背"

> *"简单是终极的复杂。"*
> 
> —— 列奥纳多·达·芬奇 (Leonardo da Vinci)

## 开场故事："背题大王"小明

想象一下，班上有个同学叫小明，他有一个"超能力"——能背诵整本教科书的每一个字。

期末考试来了，小明信心满满：
- **情况一**：如果考题和书上的例题**一字不差**，小明就能得满分！
- **情况二**：如果老师把题目改了一点点——比如把"小明有5个苹果"改成"小明有3个苹果"——小明就傻眼了，因为他只背下了"5"，不理解"加法"的本质。

同学们，你觉得小明真的"学会"了吗？

**在机器学习中，我们把这种"只会背题、不懂原理"的现象叫做过拟合（Overfitting）**。

神经网络就像一个超级聪明的学生，它有数百万个参数（相当于脑细胞），理论上可以"记住"训练数据中的每一个细节——包括噪声和错误！但这并不是我们想要的，我们希望它**真正理解规律**，而不是**死记硬背答案**。

正则化（Regularization）就是防止神经网络变成"小明"的法宝。它就像一位严格的老师，时刻提醒网络："不要背答案，要理解原理！"

---

## 21.1 过拟合：当网络"学得太死"

### 21.1.1 什么是过拟合？

让我们先用一个简单的例子理解过拟合：

```
📊 多项式拟合的故事

假设我们有5个数据点（用●表示），想要找一条曲线拟合它们：

      │
    5 ●         ●
      │
    4 │    ●
      │
    3 │         ●
      │
    2 │              ●
      │
    1 └────────────────────
      1    2    3    4    5
      
方案一：直线拟合（太简单）
      │
    5 ●  ╲      ●
      │    ╲
    4 │    ●   ╲
      │         ╲
    3 │    ●     ╲  ●
      │             ╲
    2 │              ●╲
      │                ╲
    1 └────────────────────
      1    2    3    4    5
      
      很多点都不在直线上 → 欠拟合（Underfitting）

方案二：4次多项式（刚好）
      │
    5 ●─────────●
      │    ╭╮
    4 │────●────╯╮
      │         ╭╯
    3 │    ●────╯  ●
      │             ╲
    2 │              ●
      │
    1 └────────────────────
      1    2    3    4    5
      
      曲线平滑地穿过所有点 → 刚好拟合（Just Right）

方案三：9次多项式（太复杂）
      │
    5 ●         ●
      │╲       ╱╲╱╲
    4 │─●─────●─╲╱╲╱
      │ ╲    ╱    ╲╱╲
    3 │  ●──●      ╲╱●
      │   ╲╱        ╲╱╲
    2 │    ╲         ●╲
      │     ╲          ╲
    1 └────────────────────
      1    2    3    4    5
      
      曲线剧烈抖动，完美穿过每个点 → 过拟合（Overfitting）！
      
      问题：新数据点(3.5, 3.5)会落在哪？
      - 方案二预测：约3.2（合理）
      - 方案三预测：约5.8（荒谬！）
```

**过拟合的本质**：模型对训练数据"学得太死"，把噪声也当成了规律，导致在新数据上表现很差。

### 21.1.2 过拟合的数学解释

从数学角度看，过拟合发生在模型复杂度超过数据所需复杂度时。

**训练误差 vs 泛化误差**：

$$J_{train}(\theta) = \frac{1}{m_{train}} \sum_{i=1}^{m_{train}} L(f(x^{(i)}; \theta), y^{(i)})$$

$$J_{val}(\theta) = \frac{1}{m_{val}} \sum_{i=1}^{m_{val}} L(f(x^{(i)}; \theta), y^{(i)})$$

其中：
- $J_{train}$ 是训练误差（模型在训练集上的表现）
- $J_{val}$ 是验证误差（模型在新数据上的表现，也叫泛化误差）
- $L$ 是损失函数

```
📈 过拟合的典型表现

误差
  │
  │           ⛰️ 过拟合区域
  │          ╱ ╲
  │         ╱   ╲ 训练误差（继续下降）
  │        ╱     ╲
  │       ╱       ╲________
  │      ╱         ╲       ╲
  │     ╱    🏔️     ╲       ╲ 验证误差（开始上升！）
  │    ╱   最佳点    ╲________╲
  │   ╱                ╲
  │  ╱                  ╲
  │ ╱                    ╲
  │╱                      ╲
  └────────────────────────────► 模型复杂度
     ↑
   最佳复杂度
   
关键观察：
- 简单模型：训练误差高，验证误差也高（欠拟合）
- 复杂模型：训练误差低，但验证误差高（过拟合）
- 理想模型：训练误差和验证误差都较低（泛化好）
```

### 21.1.3 过拟合的原因

1. **模型太复杂**：参数数量远超数据量（如用1000次多项式拟合10个点）
2. **数据太少**：训练样本不足以代表真实分布
3. **训练时间太长**：迭代次数过多，模型开始"记忆"噪声
4. **噪声干扰**：数据中的随机误差被当成了信号学习

---

## 21.2 正则化的本质：素质教育 vs 应试教育

### 21.2.1 费曼法解释正则化

让我们用教育领域的类比来理解正则化：

```
🏫 应试教育 vs 素质教育的类比

【应试教育】
┌─────────────────────────────────────────┐
│ 学生：背诵标准答案                       │
│ 特点：                                   │
│  - 看到"A+B"立刻反应"等于C"              │
│  - 不理解为什么等于C                      │
│  - 题目稍有变化就不会                     │
│  - 考试成绩高，实际问题解决不了           │
│                                          │
│ 神经网络版本：                            │
│  - 参数值很大、很极端                     │
│  - 对每个训练样本都"死记硬背"            │
│  - 泛化能力差                             │
└─────────────────────────────────────────┘
                      ↓ 正则化的作用
【素质教育】
┌─────────────────────────────────────────┐
│ 学生：理解原理 + 适度练习                  │
│ 特点：                                   │
│  - 理解"加法"的本质                      │
│  - 能举一反三                             │
│  - 题目变化也能解决                       │
│  - 考试成绩可能略低，但实际应用能力强     │
│                                          │
│ 神经网络版本（正则化后）：                │
│  - 参数值适中、平滑                       │
│  - 学习通用规律而非记忆具体样本           │
│  - 泛化能力强                             │
└─────────────────────────────────────────┘
```

**正则化的核心思想**：在损失函数中加入"惩罚项"，限制模型参数的大小，迫使模型学习更简洁、更通用的规律。

### 21.2.2 正则化的一般形式

正则化的损失函数可以统一表示为：

$$J_{regularized}(\theta) = J_{original}(\theta) + \lambda \cdot \Omega(\theta)$$

其中：
- $J_{original}(\theta)$ 是原始损失（如均方误差、交叉熵）
- $\Omega(\theta)$ 是正则化项（惩罚项）
- $\lambda$ 是正则化强度（超参数）

就像老师给学生布置作业：
- **原始损失**：考试分数（做对了多少题）
- **正则化项**：作业负担（作业太多会受惩罚）
- **λ**：家长对"作业负担"的重视程度

---

## 21.3 L1正则化（Lasso）：稀疏的艺术家

### 21.3.1 什么是L1正则化？

L1正则化在损失函数中加入**权重绝对值之和**：

$$J_{L1}(\theta) = J_{original}(\theta) + \lambda \sum_{j} |\theta_j|$$

为什么叫"L1"？因为它是参数的**L1范数**（曼哈顿距离）。

### 21.3.2 L1正则化的特点：产生稀疏性

L1正则化的神奇之处在于：**它会让很多权重变成精确的零！**

```
🎯 L1正则化的几何直观

假设我们有两个参数 θ₁ 和 θ₂：

      θ₂
      │
      │      ⬤ 最优解（无约束）
      │     ╱
      │    ╱ 等高线（损失函数值相同）
      │   ╱
      │  ╱
      │ ╱
      │╱
      └────────────── θ₁
      
L1约束区域（菱形）：
      θ₂
      │
    1 ┤─────●─────
      │    ╱│╲
      │   ╱ │ ╲
      │  ╱  │  ╲
      │ ╱   │   ╲
    0 ┼─────┼─────
      │╲    │    ╱
      │ ╲   │   ╱
      │  ╲  │  ╱
      │   ╲ │ ╱
   -1 ┤─────●─────
      └──────────────
        -1  0   1    θ₁
      
最优解通常在菱形的"角"上：
      θ₂
      │
    1 ┤    🎯 最优解落在坐标轴上
      │    │
      │    │    ⬤ θ₂=0, θ₁≠0
      │    │       （θ₂被"淘汰"了！）
      │    │
    0 ┼────┼────────
      │
      │
   -1 ┤
      └──────────────
        -1  0   1    θ₁
```

这个几何特性意味着：**L1正则化会自动选择重要的特征，把不重要的特征权重压缩到零**。

### 21.3.3 次梯度下降：优化L1目标

由于 $|\theta|$ 在 $\theta=0$ 处不可导，我们需要使用**次梯度**（Subgradient）：

$$\frac{\partial |\theta_j|}{\partial \theta_j} = \text{sign}(\theta_j) = \begin{cases} 1 & \text{if } \theta_j > 0 \\ -1 & \text{if } \theta_j < 0 \\ [-1, 1] & \text{if } \theta_j = 0 \end{cases}$$

参数更新规则变为：

$$\theta_j := \theta_j - \eta \left( \frac{\partial J}{\partial \theta_j} + \lambda \cdot \text{sign}(\theta_j) \right)$$

---

## 21.4 L2正则化（Ridge）：平滑的绅士

### 21.4.1 什么是L2正则化？

L2正则化在损失函数中加入**权重平方和**：

$$J_{L2}(\theta) = J_{original}(\theta) + \lambda \sum_{j} \theta_j^2$$

也叫**权重衰减**（Weight Decay），因为它的效果相当于每次迭代都让权重"衰减"一点点。

### 21.4.2 L2正则化的特点：权重平滑

与L1不同，L2不会让权重变为零，而是让它们**变小且均匀**：

```
🎯 L2正则化的几何直观

L2约束区域（圆形）：
      θ₂
      │
    1 ┤    ●
      │   ╱│╲
      │  ╱ │ ╲
      │ ●──┼──●
      │  ╲ │ ╱
    0 ┤───●┼●───
      │  ╱ │ ╲
      │ ●──┼──●
      │   ╲│╱
   -1 ┤    ●
      └──────────────
        -1  0   1    θ₁

最优解通常在圆的"边界"但不是坐标轴：
      θ₂
      │
      │       🎯 最优解
      │      ╱│
      │     ╱ │
      │    ⬤  │  θ₁≠0, θ₂≠0
      │       │  （两个参数都被保留）
      │       │
      └──────────────
              
结果：
- L2产生的是"小而均匀"的权重
- 所有特征都会有一定贡献
- 解更平滑、更稳定
```

### 21.4.3 权重衰减的闭式解（线性回归）

对于线性回归，L2正则化有漂亮的闭式解：

**原始最小二乘**：

$$\hat{\theta} = (X^T X)^{-1} X^T y$$

**Ridge回归**：

$$\hat{\theta}_{ridge} = (X^T X + \lambda I)^{-1} X^T y$$

加入的 $\lambda I$ 确保了矩阵可逆，解决了多重共线性问题！

### 21.4.4 L1 vs L2：何时用哪个？

| 特性 | L1正则化 | L2正则化 |
|------|----------|----------|
| 惩罚项 | $\sum \|\theta_j\|$ | $\sum \theta_j^2$ |
| 最优解特性 | 稀疏（很多零） | 稠密（小而均匀） |
| 特征选择 | ✅ 自动选择 | ❌ 不选择 |
| 计算 | 较慢（需次梯度） | 快（可解析解） |
| 多重共线性 | 部分解决 | ✅ 有效解决 |
| 适用场景 | 高维数据、特征选择 | 一般情况、稳定性优先 |

---

## 21.5 Dropout：随机关闭神经元的智慧

### 21.5.1 什么是Dropout？

Dropout是2014年由Srivastava等人提出的正则化技术。它的核心思想非常简单：**训练时随机"关闭"一部分神经元**。

```
🧠 Dropout的工作原理

训练时（Dropout率=0.5）：

输入层        隐藏层1        隐藏层2       输出层
  ○            ○             ○            ○
  │           ╱│╲           ╱│            │
  │          ╱ │ ╲         ╱ │            │
  ○─────────○──●──○───────○──●────────────○
  │          ╲ │ ╱         ╲ │            │
  │           ╲│╱           ╲│            │
  ○            ●             ○            ○
              被"丢弃"
              （输出=0）

每个训练样本都面对一个不同的"瘦小网络"
相当于同时训练了很多个不同的子网络！

测试时（使用全部神经元，权重缩放）：

输入层        隐藏层1        隐藏层2       输出层
  ○            ○             ○            ○
  │           ╱│╲           ╱│╲           │
  │          ╱ │ ╲         ╱ │ ╲          │
  ○─────────○──○──○───────○──○────────────○
  │          ╱ │ ╲         ╱ │ ╲          │
  │           ╲│╱           ╲│╱           │
  ○            ○             ○            ○
  
所有神经元都参与，但权重乘以保留概率p
```

### 21.5.2 Dropout的数学推导

**训练时的前向传播**：

对于第 $l$ 层，设 $r^{(l)}$ 是服从Bernoulli分布的掩码向量（以概率 $p$ 取1，概率 $1-p$ 取0）：

$$\tilde{y}^{(l)} = r^{(l)} \odot y^{(l)}$$

$$z^{(l+1)} = W^{(l+1)} \tilde{y}^{(l)} + b^{(l+1)}$$

其中 $\odot$ 表示逐元素乘法。

**推理时的近似**：

由于测试时不能使用随机性，我们使用**权重缩放**来近似所有子网络的平均：

$$W_{test}^{(l)} = p \cdot W_{train}^{(l)}$$

这保证了输出的期望相同：

$$E[r_i \cdot y_i] = p \cdot y_i$$

### 21.5.3 Dropout为什么有效？

1. **集成学习效果**：相当于训练了 $2^N$ 个不同的子网络（$N$ 是神经元数量），预测时是这些子网络的平均
2. **打破共适应**：神经元不能依赖其他特定神经元，必须学习更鲁棒的特征
3. **添加噪声**：相当于给输入和中间层添加了噪声，增强泛化能力

---

## 21.6 Batch Normalization：训练稳定的守护者

### 21.6.1 什么是Batch Normalization？

Batch Normalization（BN）是2015年由Ioffe和Szegedy提出的技术。它解决了一个关键问题：**内部协变量偏移**（Internal Covariate Shift）。

```
📊 内部协变量偏移问题

没有BN时：
Layer 1     Layer 2     Layer 3
   ↓           ↓           ↓
  训练        训练        训练
   ↓           ↓           ↓
  参数更新 → 分布变化 → 需要重新适应
   ↓           ↓           ↓
  再次训练    再次适应    ...
  
每一层的输入分布都在变，就像瞄准移动的目标！

有BN时：
Layer 1     Layer 2     Layer 3
   ↓           ↓           ↓
  BN(标准化)  BN(标准化)  BN(标准化)
   ↓           ↓           ↓
  稳定分布    稳定分布    稳定分布
   ↓           ↓           ↓
  每层面对稳定的输入分布，训练更快更稳定！
```

### 21.6.2 Batch Normalization的数学

**训练时的前向传播**：

对于一个小批量 $B = \{x_1, ..., x_m\}$：

1. **计算批量均值**：
   $$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i$$

2. **计算批量方差**：
   $$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$$

3. **标准化**：
   $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

4. **缩放和平移**（可学习的参数）：
   $$y_i = \gamma \cdot \hat{x}_i + \beta$$

其中 $\gamma$ 和 $\beta$ 是可学习的参数，让网络可以恢复原始分布如果需要。

### 21.6.3 训练和推理的差异

**训练时**：使用当前批量的统计量（均值、方差）

**推理时**：使用训练期间累积的**移动平均**：

$$\mu_{moving} = \alpha \cdot \mu_{moving} + (1-\alpha) \cdot \mu_B$$

$$\sigma^2_{moving} = \alpha \cdot \sigma^2_{moving} + (1-\alpha) \cdot \sigma^2_B$$

推理公式：

$$y = \gamma \cdot \frac{x - \mu_{moving}}{\sqrt{\sigma^2_{moving} + \epsilon}} + \beta$$

### 21.6.4 Batch Normalization的好处

1. ✅ **加速训练**：可以使用更大的学习率（通常10倍）
2. ✅ **减少对初始化的敏感性**：更稳定的训练过程
3. ✅ **轻微的正则化效果**：因为每个样本的标准化依赖于同批其他样本
4. ✅ **允许更高的学习率**：梯度不会爆炸或消失

---

## 21.7 Early Stopping：及时止损的智慧

### 21.7.1 什么是Early Stopping？

Early Stopping是最简单但非常有效的正则化技术：**监控验证集性能，当验证误差不再下降时停止训练**。

```
📈 Early Stopping示意

验证误差
  │
  │         ⛰️ 最佳停止点
  │        ╱│╲
  │       ╱ │ ╲
  │      ╱  │  ╲
  │     ╱   │   ╲  继续训练
  │    ╱    │    ╲ 验证误差上升（过拟合！）
  │   ╱     │     ╲
  │  ╱      │      ╲
  │ ╱       │       ╲
  │╱        │        ╲
  └────────────────────────────► 迭代次数
           ↑
        在这里停止！
        
训练误差会继续下降，但验证误差开始上升
这就是过拟合的信号，应该及时停止。
```

### 21.7.2 Early Stopping的实现

基本算法：

1. 每 $k$ 个epoch评估一次验证误差
2. 如果验证误差连续 $p$ 次没有改善，停止训练
3. 返回验证误差最低时的模型参数

**耐心参数**（Patience）的选择很重要：
- 太小：可能过早停止，还没收敛
- 太大：可能错过最佳点，已经过拟合

### 21.7.3 Early Stopping与L2正则化的等价性

理论研究表明，Early Stopping实际上等价于L2正则化！

直观理解：
- **L2正则化**：限制参数的大小
- **Early Stopping**：限制参数更新的步数

两者都限制了模型的"复杂度"。

---

## 21.8 Data Augmentation：数据扩充的魔法

### 21.8.1 什么是Data Augmentation？

Data Augmentation通过对训练数据进行**变换**来扩充数据集，让模型见到更多样的样本。

```
🖼️ 图像数据增强示例

原始图像：          变换后：
┌─────────┐        ┌─────────┐ ┌─────────┐ ┌─────────┐
│  🐱     │   →    │  🐱     │ │  🐱     │ │  🐱     │
│    猫   │        │    猫   │ │    猫   │ │    猫   │
│         │        │（翻转） │ │（裁剪） │ │（旋转） │
└─────────┘        └─────────┘ └─────────┘ └─────────┘
                    水平翻转    随机裁剪    随机旋转
                    
┌─────────┐        ┌─────────┐ ┌─────────┐
│  🐱     │   →    │  🐱     │ │░░🐱░░░░│
│    猫   │        │（缩放） │ │  加噪声 │
│         │        └─────────┘ └─────────┘
└─────────┘        颜色抖动    高斯噪声
```

### 21.8.2 常用的数据增强方法

**图像领域**：
- 几何变换：翻转、旋转、裁剪、缩放、平移
- 颜色变换：亮度、对比度、饱和度调整
- 噪声添加：高斯噪声、椒盐噪声
- 高级方法：Mixup、Cutout、AutoAugment

**文本领域**：
- 同义词替换
- 随机插入/删除/交换词语
- 回译（翻译到另一语言再翻译回来）

**音频领域**：
- 时间拉伸
- 音高变换
- 添加背景噪音

### 21.8.3 数据增强的正则化原理

数据增强通过增加数据的**多样性**来防止过拟合：

1. **增加有效数据量**：一个样本变成多个样本
2. **学习不变性**：猫旋转后还是猫，模型学会识别本质特征
3. **平滑决策边界**：相邻样本的标签一致，强制决策边界平滑

---

## 21.9 完整代码实现

现在让我们实现各种正则化技术的代码。我们将从零开始实现，不依赖高级框架的现成模块，以便真正理解其工作原理。

```python
"""
第二十一章代码：正则化技术完整实现
========================================
包含：
- L1/L2正则化手动实现
- Dropout层完整实现（训练/推理模式切换）
- Batch Normalization层（含移动平均）
- EarlyStopping回调类
- 多项式拟合过拟合演示
- 正则化强度对比实验

作者：机器学习教材编写组
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 第一部分：基础正则化实现
# ============================================================================

class L1Regularization:
    """
    L1正则化（Lasso）
    
    惩罚项: λ * Σ|θ|
    特点：产生稀疏解，自动特征选择
    """
    
    def __init__(self, lambda_reg: float = 0.01):
        """
        初始化L1正则化
        
        参数:
            lambda_reg: 正则化强度
        """
        self.lambda_reg = lambda_reg
    
    def compute_penalty(self, weights: np.ndarray) -> float:
        """
        计算L1惩罚项
        
        公式: λ * Σ|w|
        """
        return self.lambda_reg * np.sum(np.abs(weights))
    
    def compute_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        计算L1惩罚项的次梯度
        
        次梯度: λ * sign(w)
        在w=0时，次梯度是[-λ, λ]之间的任意值，这里取0
        """
        grad = self.lambda_reg * np.sign(weights)
        # 处理w=0的情况（可选：使用软阈值）
        return grad
    
    def soft_threshold(self, weights: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        软阈值操作（用于ISTA/FISTA算法）
        
        公式: S_λ(w) = sign(w) * max(|w| - λ, 0)
        """
        return np.sign(weights) * np.maximum(np.abs(weights) - self.lambda_reg * learning_rate, 0)


class L2Regularization:
    """
    L2正则化（Ridge / Weight Decay）
    
    惩罚项: λ * Σθ²
    特点：权重平滑衰减，解稳定
    """
    
    def __init__(self, lambda_reg: float = 0.01):
        """
        初始化L2正则化
        
        参数:
            lambda_reg: 正则化强度
        """
        self.lambda_reg = lambda_reg
    
    def compute_penalty(self, weights: np.ndarray) -> float:
        """
        计算L2惩罚项
        
        公式: λ * Σw²
        """
        return self.lambda_reg * np.sum(weights ** 2)
    
    def compute_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        计算L2惩罚项的梯度
        
        梯度: 2λ * w （通常简化为 λ * w）
        """
        return 2 * self.lambda_reg * weights
    
    def weight_decay(self, weights: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        权重衰减（Weight Decay）
        
        公式: w := w - η * 2λ * w = w * (1 - 2ηλ)
        """
        decay_factor = 1 - 2 * learning_rate * self.lambda_reg
        return weights * decay_factor


class ElasticNet:
    """
    Elastic Net正则化（L1 + L2的组合）
    
    惩罚项: λ₁ * Σ|θ| + λ₂ * Σθ²
    """
    
    def __init__(self, lambda_l1: float = 0.01, lambda_l2: float = 0.01):
        """
        初始化Elastic Net
        
        参数:
            lambda_l1: L1正则化强度
            lambda_l2: L2正则化强度
        """
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.l1 = L1Regularization(lambda_l1)
        self.l2 = L2Regularization(lambda_l2)
    
    def compute_penalty(self, weights: np.ndarray) -> float:
        """计算Elastic Net惩罚项"""
        return self.l1.compute_penalty(weights) + self.l2.compute_penalty(weights)
    
    def compute_gradient(self, weights: np.ndarray) -> np.ndarray:
        """计算Elastic Net梯度"""
        return self.l1.compute_gradient(weights) + self.l2.compute_gradient(weights)


# ============================================================================
# 第二部分：Dropout层实现
# ============================================================================

class Dropout:
    """
    Dropout层完整实现
    
    训练时：随机将部分神经元输出置零
    推理时：使用所有神经元，权重缩放
    """
    
    def __init__(self, dropout_rate: float = 0.5):
        """
        初始化Dropout层
        
        参数:
            dropout_rate: 丢弃概率（0-1之间）
                         0.5表示50%的神经元被丢弃
        """
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate必须在[0, 1)之间")
        
        self.dropout_rate = dropout_rate
        self.keep_rate = 1 - dropout_rate
        self.mask = None
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        参数:
            x: 输入数组，形状 (batch_size, n_features)
        
        返回:
            输出数组
        """
        if self.training:
            # 训练时：生成随机掩码并应用
            self.mask = (np.random.rand(*x.shape) < self.keep_rate).astype(np.float32)
            # 反向Dropout：训练时缩放，推理时不缩放
            return x * self.mask / self.keep_rate
        else:
            # 推理时：使用所有神经元（已在训练时缩放）
            return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        反向传播
        
        参数:
            grad_output: 上游梯度
        
        返回:
            传递给下游的梯度
        """
        if self.training:
            # 只传播保留神经元的梯度
            return grad_output * self.mask / self.keep_rate
        else:
            return grad_output
    
    def train(self):
        """设置为训练模式"""
        self.training = True
    
    def eval(self):
        """设置为评估模式"""
        self.training = False


# ============================================================================
# 第三部分：Batch Normalization层实现
# ============================================================================

class BatchNormalization:
    """
    Batch Normalization层完整实现
    
    包含：
    - 训练时使用批量统计量
    - 推理时使用移动平均
    - 可学习的缩放(γ)和平移(β)参数
    """
    
    def __init__(self, n_features: int, momentum: float = 0.9, eps: float = 1e-5):
        """
        初始化Batch Normalization层
        
        参数:
            n_features: 特征数量
            momentum: 移动平均的动量系数
            eps: 数值稳定性常数
        """
        self.n_features = n_features
        self.momentum = momentum
        self.eps = eps
        
        # 可学习参数
        self.gamma = np.ones((1, n_features))  # 缩放参数
        self.beta = np.zeros((1, n_features))  # 平移参数
        
        # 移动平均统计量（用于推理）
        self.running_mean = np.zeros((1, n_features))
        self.running_var = np.ones((1, n_features))
        
        # 训练缓存
        self.training = True
        self.batch_mean = None
        self.batch_var = None
        self.x_normalized = None
        self.x_centered = None
        self.std_inv = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        参数:
            x: 输入，形状 (batch_size, n_features)
        
        返回:
            归一化后的输出
        """
        if self.training:
            # 训练模式：使用批量统计量
            self.batch_mean = np.mean(x, axis=0, keepdims=True)
            self.batch_var = np.var(x, axis=0, keepdims=True)
            
            # 更新移动平均
            self.running_mean = self.momentum * self.running_mean + \
                               (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + \
                              (1 - self.momentum) * self.batch_var
            
            # 标准化
            self.x_centered = x - self.batch_mean
            self.std_inv = 1.0 / np.sqrt(self.batch_var + self.eps)
            self.x_normalized = self.x_centered * self.std_inv
            
            # 缩放和平移
            out = self.gamma * self.x_normalized + self.beta
            
        else:
            # 推理模式：使用移动平均
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_normalized + self.beta
        
        return out
    
    def backward(self, grad_output: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        反向传播
        
        参数:
            grad_output: 上游梯度，形状 (batch_size, n_features)
            x: 前向传播时的输入
        
        返回:
            (dx, dgamma, dbeta)
        """
        batch_size = x.shape[0]
        
        # 关于gamma和beta的梯度
        dgamma = np.sum(grad_output * self.x_normalized, axis=0, keepdims=True)
        dbeta = np.sum(grad_output, axis=0, keepdims=True)
        
        # 关于x_normalized的梯度
        dx_normalized = grad_output * self.gamma
        
        # 关于方差的梯度
        dvar = np.sum(dx_normalized * self.x_centered * (-0.5) * (self.std_inv ** 3), axis=0, keepdims=True)
        
        # 关于均值的梯度
        dmean = np.sum(dx_normalized * (-self.std_inv), axis=0, keepdims=True) + \
                dvar * np.mean(-2 * self.x_centered, axis=0, keepdims=True)
        
        # 关于x的梯度
        dx = dx_normalized * self.std_inv + dvar * 2 * self.x_centered / batch_size + dmean / batch_size
        
        return dx, dgamma, dbeta
    
    def train(self):
        """设置为训练模式"""
        self.training = True
    
    def eval(self):
        """设置为评估模式"""
        self.training = False
    
    def update_params(self, dgamma: np.ndarray, dbeta: np.ndarray, learning_rate: float):
        """更新参数"""
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta


# ============================================================================
# 第四部分：Early Stopping回调
# ============================================================================

class EarlyStopping:
    """
    早停机制
    
    监控验证集性能，当连续多轮没有改善时停止训练
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        """
        初始化Early Stopping
        
        参数:
            patience: 耐心值，连续多少轮没有改善就停止
            min_delta: 最小改善量，小于此值不算改善
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0
        self.should_stop = False
    
    def __call__(self, val_loss: float, model_weights: Optional[Dict] = None) -> bool:
        """
        检查是否应该停止训练
        
        参数:
            val_loss: 当前验证损失
            model_weights: 当前模型权重（用于恢复最佳）
        
        返回:
            True如果应该停止训练
        """
        if val_loss < self.best_loss - self.min_delta:
            # 有改善
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights and model_weights is not None:
                self.best_weights = {k: v.copy() for k, v in model_weights.items()}
        else:
            # 没有改善
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False
    
    def get_best_weights(self) -> Optional[Dict]:
        """获取最佳权重"""
        return self.best_weights
    
    def reset(self):
        """重置状态"""
        self.best_loss = float('inf')
        self.best_weights = None
        self.counter = 0
        self.should_stop = False


# ============================================================================
# 第五部分：多项式拟合演示（过拟合/正则化可视化）
# ============================================================================

class PolynomialFitter:
    """
    多项式拟合器（用于演示过拟合和正则化）
    """
    
    def __init__(self, degree: int):
        """
        初始化
        
        参数:
            degree: 多项式次数
        """
        self.degree = degree
        self.weights = None
        self.regularizer = None
    
    def set_regularizer(self, regularizer):
        """设置正则化器"""
        self.regularizer = regularizer
    
    def _design_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        构造设计矩阵（Vandermonde矩阵）
        
        X = [1, x, x², ..., xⁿ]
        """
        return np.vander(x, self.degree + 1, increasing=True)
    
    def fit(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, 
            epochs: int = 10000, verbose: bool = False) -> List[float]:
        """
        使用梯度下降拟合
        
        参数:
            x: 输入数据
            y: 目标数据
            learning_rate: 学习率
            epochs: 迭代次数
            verbose: 是否打印进度
        
        返回:
            损失历史
        """
        X = self._design_matrix(x)
        n_samples = len(x)
        
        # 初始化权重
        self.weights = np.random.randn(self.degree + 1) * 0.01
        
        losses = []
        
        for epoch in range(epochs):
            # 前向传播
            predictions = X @ self.weights
            
            # 计算损失（均方误差）
            mse_loss = np.mean((predictions - y) ** 2)
            
            # 添加正则化项
            reg_loss = 0
            if self.regularizer is not None:
                reg_loss = self.regularizer.compute_penalty(self.weights)
            
            total_loss = mse_loss + reg_loss
            losses.append(total_loss)
            
            # 计算梯度
            grad_mse = (2 / n_samples) * X.T @ (predictions - y)
            
            if self.regularizer is not None:
                grad_reg = self.regularizer.compute_gradient(self.weights)
                grad = grad_mse + grad_reg
            else:
                grad = grad_mse
            
            # 更新权重
            self.weights -= learning_rate * grad
            
            if verbose and epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.6f}")
        
        return losses
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测"""
        X = self._design_matrix(x)
        return X @ self.weights
    
    def get_weights(self) -> np.ndarray:
        """获取权重"""
        return self.weights.copy()


def demonstrate_overfitting():
    """
    演示过拟合现象和正则化的效果
    """
    # 设置随机种子
    np.random.seed(42)
    
    # 生成数据：真实的函数是二次函数
    def true_function(x):
        return 0.5 * x**2 - 2 * x + 1
    
    # 生成训练数据（带噪声）
    n_train = 15
    x_train = np.linspace(-3, 3, n_train)
    y_train = true_function(x_train) + np.random.randn(n_train) * 0.5
    
    # 生成测试数据（无噪声）
    x_test = np.linspace(-3, 3, 100)
    y_test = true_function(x_test)
    
    # 测试不同复杂度的模型
    degrees = [2, 5, 15]
    lambdas = [0, 0.001, 0.01, 0.1]
    
    fig, axes = plt.subplots(len(degrees), len(lambdas), figsize=(16, 12))
    fig.suptitle('Polynomial Fitting: Effect of Model Complexity and Regularization', fontsize=14)
    
    for i, degree in enumerate(degrees):
        for j, lambda_reg in enumerate(lambdas):
            ax = axes[i, j]
            
            # 创建拟合器
            fitter = PolynomialFitter(degree)
            
            if lambda_reg > 0:
                fitter.set_regularizer(L2Regularization(lambda_reg))
            
            # 拟合
            fitter.fit(x_train, y_train, learning_rate=0.01, epochs=5000)
            
            # 预测
            x_plot = np.linspace(-3.5, 3.5, 200)
            y_pred = fitter.predict(x_plot)
            y_train_pred = fitter.predict(x_train)
            
            # 计算误差
            train_error = np.mean((y_train_pred - y_train) ** 2)
            test_error = np.mean((fitter.predict(x_test) - y_test) ** 2)
            
            # 绘图
            ax.scatter(x_train, y_train, c='red', s=50, zorder=3, label='Training Data')
            ax.plot(x_test, y_test, 'g-', linewidth=2, label='True Function', alpha=0.7)
            ax.plot(x_plot, y_pred, 'b-', linewidth=2, label='Fitted Curve')
            
            # 设置标题
            reg_text = f"λ={lambda_reg}" if lambda_reg > 0 else "No Regularization"
            ax.set_title(f'Degree={degree}, {reg_text}\n'
                        f'Train Error: {train_error:.3f}, Test Error: {test_error:.3f}',
                        fontsize=10)
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-3, 8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regularization_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("演示图已保存为 'regularization_demo.png'")


# ============================================================================
# 第六部分：权重分布可视化
# ============================================================================

def compare_l1_l2_weights():
    """
    比较L1和L2正则化对权重分布的影响
    """
    np.random.seed(42)
    
    # 生成数据
    n_samples, n_features = 100, 50
    X = np.random.randn(n_samples, n_features)
    
    # 真实权重（稀疏）
    true_weights = np.zeros(n_features)
    true_weights[[0, 5, 10, 15, 20]] = [2, -1.5, 3, -2, 1]
    
    y = X @ true_weights + np.random.randn(n_samples) * 0.5
    
    # 训练/测试分割
    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 训练函数
    def train_with_regularization(X, y, regularizer, epochs=5000, lr=0.01):
        weights = np.random.randn(X.shape[1]) * 0.1
        for _ in range(epochs):
            pred = X @ weights
            grad = (2 / len(y)) * X.T @ (pred - y)
            if regularizer:
                grad += regularizer.compute_gradient(weights)
            weights -= lr * grad
        return weights
    
    # 训练不同正则化的模型
    reg_none = None
    reg_l1_001 = L1Regularization(0.01)
    reg_l1_01 = L1Regularization(0.1)
    reg_l2_001 = L2Regularization(0.01)
    reg_l2_01 = L2Regularization(0.1)
    
    weights_none = train_with_regularization(X_train, y_train, reg_none)
    weights_l1_001 = train_with_regularization(X_train, y_train, reg_l1_001)
    weights_l1_01 = train_with_regularization(X_train, y_train, reg_l1_01)
    weights_l2_001 = train_with_regularization(X_train, y_train, reg_l2_001)
    weights_l2_01 = train_with_regularization(X_train, y_train, reg_l2_01)
    
    # 计算测试误差
    def mse(X, y, w):
        return np.mean((X @ w - y) ** 2)
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Weight Distribution: L1 vs L2 Regularization', fontsize=14)
    
    methods = [
        ('No Regularization', weights_none, 'gray'),
        ('L1 (λ=0.01)', weights_l1_001, 'blue'),
        ('L1 (λ=0.1)', weights_l1_01, 'darkblue'),
        ('L2 (λ=0.01)', weights_l2_001, 'red'),
        ('L2 (λ=0.1)', weights_l2_01, 'darkred'),
    ]
    
    # 权重分布直方图
    ax = axes[0, 0]
    for name, weights, color in methods:
        ax.hist(weights, bins=20, alpha=0.5, label=name, color=color)
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Weight Distribution Histogram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 权重绝对值条形图
    ax = axes[0, 1]
    x_pos = np.arange(n_features)
    width = 0.15
    for idx, (name, weights, color) in enumerate(methods):
        ax.bar(x_pos + idx * width, np.abs(weights), width, 
               label=name, color=color, alpha=0.7)
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('|Weight|')
    ax.set_title('Absolute Weight Values')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 零权重数量对比
    ax = axes[0, 2]
    names = [m[0] for m in methods]
    zero_counts = [np.sum(np.abs(m[1]) < 0.01) for m in methods]
    colors_list = [m[2] for m in methods]
    bars = ax.bar(range(len(names)), zero_counts, color=colors_list, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Number of Near-Zero Weights')
    ax.set_title('Sparsity Comparison (|w| < 0.01)')
    ax.grid(True, alpha=0.3)
    
    # L1权重路径
    ax = axes[1, 0]
    for i in range(min(10, n_features)):
        weights_path = []
        reg = L1Regularization(0.05)
        w = np.random.randn(n_features) * 0.1
        for _ in range(2000):
            pred = X_train @ w
            grad = (2 / len(y_train)) * X_train.T @ (pred - y_train)
            grad += reg.compute_gradient(w)
            w -= 0.01 * grad
            weights_path.append(w[i])
        ax.plot(weights_path, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weight Value')
    ax.set_title('L1: Weight Paths During Training')
    ax.grid(True, alpha=0.3)
    
    # L2权重路径
    ax = axes[1, 1]
    for i in range(min(10, n_features)):
        weights_path = []
        reg = L2Regularization(0.05)
        w = np.random.randn(n_features) * 0.1
        for _ in range(2000):
            pred = X_train @ w
            grad = (2 / len(y_train)) * X_train.T @ (pred - y_train)
            grad += reg.compute_gradient(w)
            w -= 0.01 * grad
            weights_path.append(w[i])
        ax.plot(weights_path, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weight Value')
    ax.set_title('L2: Weight Paths During Training')
    ax.grid(True, alpha=0.3)
    
    # 测试误差对比
    ax = axes[1, 2]
    test_errors = [mse(X_test, y_test, m[1]) for m in methods]
    bars = ax.bar(range(len(names)), test_errors, color=colors_list, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Test MSE')
    ax.set_title('Test Error Comparison')
    ax.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值
    for bar, error in zip(bars, test_errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('l1_l2_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("对比图已保存为 'l1_l2_comparison.png'")


# ============================================================================
# 第七部分：Dropout效果演示
# ============================================================================

def demonstrate_dropout():
    """
    演示Dropout的效果
    """
    np.random.seed(42)
    
    # 生成数据
    n_samples = 200
    X = np.random.randn(n_samples, 2)
    # 创建圆形决策边界
    y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int)
    
    # 添加噪声
    y = y ^ (np.random.rand(n_samples) < 0.1)
    
    # 划分训练/测试集
    train_size = 150
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 简单的两层神经网络
    class SimpleNN:
        def __init__(self, hidden_size=50, dropout_rate=0.0):
            self.W1 = np.random.randn(2, hidden_size) * 0.1
            self.b1 = np.zeros(hidden_size)
            self.W2 = np.random.randn(hidden_size, 1) * 0.1
            self.b2 = np.zeros(1)
            self.dropout = Dropout(dropout_rate) if dropout_rate > 0 else None
            self.losses_train = []
            self.losses_val = []
        
        def forward(self, X, training=True):
            self.z1 = X @ self.W1 + self.b1
            self.a1 = np.maximum(0, self.z1)  # ReLU
            
            if self.dropout and training:
                self.a1 = self.dropout.forward(self.a1)
            
            self.z2 = self.a1 @ self.W2 + self.b2
            self.a2 = 1 / (1 + np.exp(-self.z2))  # Sigmoid
            return self.a2
        
        def backward(self, X, y, learning_rate):
            m = X.shape[0]
            
            # 输出层梯度
            dz2 = self.a2 - y.reshape(-1, 1)
            dW2 = (self.a1.T @ dz2) / m
            db2 = np.sum(dz2, axis=0) / m
            
            # 隐藏层梯度
            da1 = dz2 @ self.W2.T
            if self.dropout:
                da1 = self.dropout.backward(da1)
            
            dz1 = da1 * (self.z1 > 0)  # ReLU导数
            dW1 = (X.T @ dz1) / m
            db1 = np.sum(dz1, axis=0) / m
            
            # 更新参数
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
        
        def train(self, X, y, X_val, y_val, epochs=1000, lr=0.1):
            for epoch in range(epochs):
                # 训练
                if self.dropout:
                    self.dropout.train()
                pred_train = self.forward(X, training=True)
                loss_train = -np.mean(y * np.log(pred_train + 1e-8) + 
                                     (1-y) * np.log(1 - pred_train + 1e-8))
                self.backward(X, y, lr)
                
                # 验证
                if self.dropout:
                    self.dropout.eval()
                pred_val = self.forward(X_val, training=False)
                loss_val = -np.mean(y_val * np.log(pred_val + 1e-8) + 
                                   (1-y_val) * np.log(1 - pred_val + 1e-8))
                
                self.losses_train.append(loss_train)
                self.losses_val.append(loss_val)
        
        def predict(self, X):
            if self.dropout:
                self.dropout.eval()
            return (self.forward(X, training=False) > 0.5).astype(int)
    
    # 训练不同dropout率的模型
    dropout_rates = [0.0, 0.3, 0.5, 0.7]
    models = []
    
    for rate in dropout_rates:
        model = SimpleNN(hidden_size=50, dropout_rate=rate)
        model.train(X_train, y_train, X_test, y_test, epochs=1000, lr=0.1)
        models.append(model)
    
    # 可视化
    fig, axes = plt.subplots(2, len(dropout_rates), figsize=(16, 10))
    fig.suptitle('Dropout Effect on Training and Generalization', fontsize=14)
    
    for idx, (rate, model) in enumerate(zip(dropout_rates, models)):
        # 训练曲线
        ax = axes[0, idx]
        ax.plot(model.losses_train, label='Train Loss', alpha=0.7)
        ax.plot(model.losses_val, label='Val Loss', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Dropout Rate = {rate}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 决策边界
        ax = axes[1, idx]
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', 
                  edgecolors='k', s=30, label='Train')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', 
                  marker='s', edgecolors='k', s=30, label='Test')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_title(f'Decision Boundary (Dropout={rate})')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('dropout_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Dropout演示图已保存为 'dropout_demo.png'")


# ============================================================================
# 第八部分：完整运行演示
# ============================================================================

def run_all_demos():
    """
    运行所有演示
    """
    print("=" * 60)
    print("第二十一章：正则化技术完整演示")
    print("=" * 60)
    
    print("\n1. 演示过拟合和正则化效果...")
    demonstrate_overfitting()
    
    print("\n2. 比较L1和L2正则化...")
    compare_l1_l2_weights()
    
    print("\n3. 演示Dropout效果...")
    demonstrate_dropout()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)


# ============================================================================
# 单元测试
# ============================================================================

def test_regularization():
    """
    测试正则化实现
    """
    print("\n" + "=" * 40)
    print("单元测试")
    print("=" * 40)
    
    # 测试L1正则化
    l1 = L1Regularization(lambda_reg=0.1)
    weights = np.array([1.0, -2.0, 0.0, 3.0])
    penalty = l1.compute_penalty(weights)
    assert abs(penalty - 0.6) < 1e-6, f"L1 penalty计算错误: {penalty}"
    print("✓ L1正则化测试通过")
    
    # 测试L2正则化
    l2 = L2Regularization(lambda_reg=0.1)
    penalty = l2.compute_penalty(weights)
    assert abs(penalty - 1.4) < 1e-6, f"L2 penalty计算错误: {penalty}"
    print("✓ L2正则化测试通过")
    
    # 测试Dropout
    dropout = Dropout(dropout_rate=0.5)
    x = np.ones((1000, 10))
    dropout.train()
    y = dropout.forward(x)
    # 大约50%的神经元被保留
    keep_ratio = np.mean(y > 0)
    assert 0.4 < keep_ratio < 0.6, f"Dropout比例异常: {keep_ratio}"
    print("✓ Dropout测试通过")
    
    # 测试BatchNorm
    bn = BatchNormalization(n_features=5)
    x = np.random.randn(32, 5)
    bn.train()
    y = bn.forward(x)
    # 输出应该近似均值为0，方差为1（考虑gamma=1, beta=0）
    assert np.abs(np.mean(y)) < 0.1, "BatchNorm均值异常"
    print("✓ BatchNorm测试通过")
    
    print("=" * 40)
    print("所有测试通过！")
    print("=" * 40)


if __name__ == "__main__":
    # 运行单元测试
    test_regularization()
    
    # 运行完整演示
    run_all_demos()
```

---

## 21.10 练习题

### 基础概念题

**问题1：过拟合与欠拟合**

请解释以下概念，并各举一个实际例子：
- 过拟合（Overfitting）
- 欠拟合（Underfitting）
- 刚好拟合（Just Right Fitting）

**提示**：可以结合多项式拟合的角度思考。

<details>
<summary>参考答案</summary>

**过拟合**：模型过于复杂，记住了训练数据的噪声和细节，泛化能力差。
- 例子：用100次多项式拟合10个数据点，曲线剧烈抖动。

**欠拟合**：模型过于简单，无法捕捉数据的基本规律。
- 例子：用直线拟合明显呈抛物线分布的数据。

**刚好拟合**：模型复杂度适中，能够捕捉数据规律而不受噪声干扰。
- 例子：用2次多项式拟合抛物线分布的数据。

</details>

---

**问题2：L1 vs L2正则化**

比较L1和L2正则化的以下方面：
1. 惩罚项的形式
2. 产生的解的特性（稀疏性）
3. 优化方法的区别
4. 各自适合的应用场景

<details>
<summary>参考答案</summary>

| 方面 | L1正则化 | L2正则化 |
|------|----------|----------|
| 惩罚项 | $\lambda \sum \|\theta_j\|$ | $\lambda \sum \theta_j^2$ |
| 解的特性 | 稀疏，许多权重为零 | 稠密，权重小而均匀 |
| 优化方法 | 次梯度下降 | 标准梯度下降 |
| 适用场景 | 特征选择、高维数据 | 一般情况、稳定性要求高 |

</details>

---

**问题3：Dropout的工作原理**

请回答：
1. Dropout在训练时和测试时的行为有何不同？
2. 为什么Dropout能够防止过拟合？
3. 如果Dropout率为0.5，训练时某个神经元的输出期望是多少？测试时如何处理？

<details>
<summary>参考答案</summary>

1. **训练时**：以概率 $p$ 随机丢弃神经元（输出置零），保留的神经元的输出缩放为原来的 $1/p$ 倍。
   
   **测试时**：使用所有神经元，不丢弃任何神经元，也不进行缩放。

2. **防止过拟合的原因**：
   - 相当于训练了多个子网络的集成
   - 打破神经元之间的共适应
   - 添加随机噪声，增强鲁棒性

3. 训练时期望输出是 $0.5 \times \text{原输出} \times 2 = \text{原输出}$（考虑缩放），测试时直接使用原输出。

</details>

---

### 进阶推导题

**问题4：L2正则化的闭式解**

对于线性回归问题，原始损失函数为：
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$$

加入L2正则化后：
$$J_{L2}(\theta) = J(\theta) + \frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2$$

请推导Ridge回归的闭式解：
$$\theta = (X^T X + \lambda I)^{-1} X^T y$$

<details>
<summary>参考答案</summary>

**步骤1：写出矩阵形式的损失函数**

$$J_{L2} = \frac{1}{2m}(X\theta - y)^T(X\theta - y) + \frac{\lambda}{2m}\theta^T\theta$$

**步骤2：对 $\theta$ 求导**

$$\frac{\partial J_{L2}}{\partial \theta} = \frac{1}{m}X^T(X\theta - y) + \frac{\lambda}{m}\theta$$

**步骤3：令导数为零**

$$X^T(X\theta - y) + \lambda\theta = 0$$

$$X^TX\theta - X^Ty + \lambda\theta = 0$$

$$(X^TX + \lambda I)\theta = X^Ty$$

**步骤4：求解**

$$\theta = (X^TX + \lambda I)^{-1}X^Ty$$

**证毕。**

</details>

---

**问题5：Batch Normalization的梯度推导**

给定Batch Normalization的前向传播：
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

其中 $\mu_B$ 和 $\sigma_B^2$ 是批量均值和方差。

请推导反向传播中 $\frac{\partial L}{\partial x_i}$ 的表达式。

<details>
<summary>参考答案</summary>

设 $L$ 是损失函数，我们需要计算 $\frac{\partial L}{\partial x_i}$。

**步骤1：关于 $\gamma$ 和 $\beta$ 的梯度**

$$\frac{\partial L}{\partial \gamma} = \sum_i \frac{\partial L}{\partial y_i} \hat{x}_i$$

$$\frac{\partial L}{\partial \beta} = \sum_i \frac{\partial L}{\partial y_i}$$

**步骤2：关于 $\hat{x}_i$ 的梯度**

$$\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma$$

**步骤3：关于方差的梯度**

设 $\sigma^2 = \sigma_B^2 + \epsilon$

$$\frac{\partial L}{\partial \sigma^2} = \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot (x_i - \mu_B) \cdot (-\frac{1}{2})(\sigma^2)^{-3/2}$$

**步骤4：关于均值的梯度**

$$\frac{\partial L}{\partial \mu_B} = \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot (-\frac{1}{\sqrt{\sigma^2}}) + \frac{\partial L}{\partial \sigma^2} \cdot \frac{\sum_i -2(x_i - \mu_B)}{m}$$

**步骤5：关于 $x_i$ 的梯度**

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma^2}} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{2(x_i - \mu_B)}{m} + \frac{\partial L}{\partial \mu_B} \cdot \frac{1}{m}$$

**简化形式**：

$$\frac{\partial L}{\partial x_i} = \frac{\gamma}{\sqrt{\sigma_B^2 + \epsilon}} \left( \frac{\partial L}{\partial y_i} - \frac{1}{m}\sum_j \frac{\partial L}{\partial y_j} - \hat{x}_i \cdot \frac{1}{m}\sum_j \frac{\partial L}{\partial y_j} \hat{x}_j \right)$$

</details>

---

**问题6：L1正则化产生稀疏性的数学证明**

考虑一维的Lasso问题：
$$\min_\theta \frac{1}{2}(y - \theta)^2 + \lambda |\theta|$$

请证明：当 $|y| \leq \lambda$ 时，最优解是 $\theta^* = 0$。

<details>
<summary>参考答案</summary>

**步骤1：写出损失函数**

$$f(\theta) = \frac{1}{2}(y - \theta)^2 + \lambda |\theta|$$

**步骤2：考虑 $\theta > 0$ 的情况**

$$f(\theta) = \frac{1}{2}(y - \theta)^2 + \lambda \theta$$

$$f'(\theta) = -(y - \theta) + \lambda = \theta - y + \lambda$$

令 $f'(\theta) = 0$：$\theta = y - \lambda$

若 $y > \lambda$，则 $\theta^* = y - \lambda > 0$

若 $y \leq \lambda$，则 $\theta^* = y - \lambda \leq 0$，与假设矛盾，最小值在边界 $\theta = 0$ 处。

**步骤3：考虑 $\theta < 0$ 的情况**

$$f(\theta) = \frac{1}{2}(y - \theta)^2 - \lambda \theta$$

$$f'(\theta) = -(y - \theta) - \lambda = \theta - y - \lambda$$

令 $f'(\theta) = 0$：$\theta = y + \lambda$

若 $y < -\lambda$，则 $\theta^* = y + \lambda < 0$

若 $y \geq -\lambda$，则 $\theta^* = y + \lambda \geq 0$，与假设矛盾，最小值在边界 $\theta = 0$ 处。

**步骤4：综合**

当 $|y| \leq \lambda$ 时：
- $\theta > 0$ 的最优解不满足条件，边界值为 $f(0) = \frac{1}{2}y^2$
- $\theta < 0$ 的最优解不满足条件，边界值为 $f(0) = \frac{1}{2}y^2$

因此，当 $|y| \leq \lambda$ 时，$\theta^* = 0$。

**证毕。**

这证明了L1正则化的**软阈值**特性：小信号被压缩为零，大信号被收缩 $\lambda$。

</details>

---

### 挑战编程题

**问题7：实现带正则化的神经网络**

请实现一个单隐藏层的神经网络类 `RegularizedNN`，支持以下功能：
- L1/L2/Elastic Net正则化
- Dropout
- Batch Normalization（可选）
- Early Stopping

在MNIST或合成数据集上测试不同正则化组合的效果。

**要求**：
1. 从零实现，不使用PyTorch/TensorFlow的高级API
2. 对比不同正则化策略的验证集准确率
3. 可视化训练曲线和权重分布

<details>
<summary>提示与框架</summary>

```python
class RegularizedNN:
    def __init__(self, input_size, hidden_size, output_size,
                 regularizer=None, dropout_rate=0.0, use_batchnorm=False):
        # 初始化权重和偏置
        # 初始化正则化器
        # 初始化Dropout层
        # 初始化BatchNorm层
        pass
    
    def forward(self, X, training=True):
        # 前向传播
        # 应用Dropout（训练时）
        # 应用BatchNorm
        pass
    
    def backward(self, X, y, learning_rate):
        # 反向传播
        # 计算梯度
        # 添加正则化梯度
        # 更新参数
        pass
    
    def train_with_early_stopping(self, X_train, y_train, X_val, y_val,
                                   max_epochs, patience, ...):
        # 实现早停逻辑
        pass
```

</details>

---

**问题8：自适应正则化强度**

设计一个自适应调整正则化强度的算法：

1. 初始设置较大的 $\lambda$ 值
2. 监控训练误差和验证误差
3. 如果训练误差下降缓慢但验证误差上升，增大 $\lambda$
4. 如果训练误差和验证误差都高，减小 $\lambda$

在多项式拟合任务上测试你的自适应算法，与固定 $\lambda$ 进行比较。

**评价指标**：
- 最终测试误差
- 收敛速度（达到目标误差所需的epoch数）
- 超参数调节的难易程度

---

## 21.11 参考文献

1. **Tikhonov, A. N.** (1943). On the stability of inverse problems. *Doklady Akademii Nauk SSSR*, 39(5), 195-198.
   - 正则化方法的起源，解决不适定问题的数学基础

2. **Hoerl, A. E., & Kennard, R. W.** (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67.
   - Ridge回归的经典论文，提出L2正则化解决多重共线性问题

3. **Tibshirani, R.** (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B (Methodological)*, 58(1), 267-288.
   - Lasso方法的开创性论文，L1正则化产生稀疏解的理论基础

4. **Ng, A. Y.** (2004). Feature selection, L1 vs. L2 regularization, and rotational invariance. *Proceedings of the Twenty-First International Conference on Machine Learning* (p. 78).
   - 系统比较L1和L2正则化的样本复杂度

5. **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R.** (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1), 1929-1958.
   - Dropout技术的原始论文，现代深度学习的重要正则化方法

6. **Ioffe, S., & Szegedy, C.** (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *Proceedings of the 32nd International Conference on Machine Learning*, 448-456.
   - Batch Normalization的原始论文，稳定训练并允许更大学习率

7. **Prechelt, L.** (1998). Early stopping-but when? In *Neural Networks: Tricks of the Trade* (pp. 55-69). Springer.
   - Early Stopping策略的经典讨论

8. **Krizhevsky, A., Sutskever, I., & Hinton, G. E.** (2012). ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems*, 25, 1097-1105.
   - AlexNet论文，展示了数据增强和Dropout在深度学习中的效果

9. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep learning*. MIT Press.
   - 深度学习教材，第7章详细讨论正则化

10. **Zou, H., & Hastie, T.** (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 67(2), 301-320.
    - Elastic Net正则化，结合L1和L2的优点

---

## 总结

正则化是深度学习中防止过拟合的核心技术。让我们回顾本章的关键点：

```
🎯 核心概念总结

过拟合（死记硬背）
    │
    ├── 症状：训练误差低，验证误差高
    │
    ├── 原因：模型太复杂、数据太少、训练太久
    │
    └── 解决：正则化！
        │
        ├── L1正则化（Lasso）
        │   ├── 惩罚：Σ|θ|
        │   ├── 效果：稀疏解，自动特征选择
        │   └── 适用：高维数据，需要可解释性
        │
        ├── L2正则化（Ridge/Weight Decay）
        │   ├── 惩罚：Σθ²
        │   ├── 效果：小而均匀的权重
        │   └── 适用：一般情况，稳定性优先
        │
        ├── Dropout
        │   ├── 方法：训练时随机丢弃神经元
        │   ├── 效果：集成学习，打破共适应
        │   └── 适用：深层网络，全连接层
        │
        ├── Batch Normalization
        │   ├── 方法：标准化每层输入
        │   ├── 效果：稳定训练，允许大学习率
        │   └── 适用：几乎所有深层网络
        │
        ├── Early Stopping
        │   ├── 方法：监控验证误差，及时停止
        │   ├── 效果：防止训练过度
        │   └── 适用：任何迭代训练
        │
        └── Data Augmentation
            ├── 方法：扩充训练数据
            ├── 效果：增加数据多样性
            └── 适用：图像、文本、音频

选择建议：
- 图像任务：Dropout + Data Augmentation + BatchNorm
- 表格数据：L1/L2正则化
- 大模型：所有技术组合使用
```

记住：**正则化的本质是让模型"真正学会"而不是"死记硬背"**。就像好的教育应该培养学生的理解能力而不是记忆能力一样，好的神经网络应该学习数据的内在规律，而不是记住每一个训练样本。

---

> *"正则化不仅是一种技术，更是一种哲学——追求简单和泛化的智慧。"*
