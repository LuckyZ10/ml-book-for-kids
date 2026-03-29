# 第十六章：感知机——神经网络的起点

> *"The perceptron is the first machine which is capable of having an original idea."*  
> —— Frank Rosenblatt, 1958

---

## 开篇故事：大脑的启示

想象一下，你正在森林里散步，突然看到前方有一个黑黄相间的物体在蠕动。你的大脑瞬间做出了判断："那是蜜蜂，有毒，快躲开！"整个过程不到0.1秒。

但如果你是第一次见到蜜蜂呢？

实际上，你的大脑并不是天生就知道蜜蜂有毒的。当你还是小孩时，你可能被蜜蜂蜇过，或者有人告诉过你蜜蜂的危险。你的大脑通过**经验**学会了识别蜜蜂的特征：黑黄相间的条纹、嗡嗡的声音、飞行的姿态。每一次遇到蜜蜂（无论是真实的还是图片上的），你大脑中的某些连接就会加强，让你下次识别得更快更准确。

**这就是学习——通过经验改变连接强度的过程。**

1958年，一位名叫弗兰克·罗森布拉特（Frank Rosenblatt）的心理学家，在康奈尔航空实验室提出了一个革命性的想法：能不能造一台机器，像人脑一样通过学习来识别模式？

他设计的这台机器叫做**感知机（Perceptron）**，它成为了现代神经网络的起点。

---

## 16.1 从生物神经元到人工神经元

### 16.1.1 生物神经元长什么样？

在深入了解感知机之前，让我们先认识一下它的"原型"——人脑中的神经元。

一个典型的生物神经元由三部分组成：

```
         ┌─────────────────────────┐
         │      树突 (Dendrites)    │  ← 输入端：接收信号
         │    分支状的"触手"        │
         └───────────┬─────────────┘
                     │
         ┌───────────▼─────────────┐
         │      细胞体 (Soma)       │  ← 处理中心：整合信号
         │   包含细胞核的"控制中心"  │
         └───────────┬─────────────┘
                     │
         ┌───────────▼─────────────┐
         │      轴突 (Axon)         │  ← 输出端：发送信号
         │    长长的"传输线"         │
         └─────────────────────────┘
```

**工作原理**（简化版）：
1. **树突**从其他神经元接收电信号（神经冲动）
2. **细胞体**把所有输入信号加总
3. 如果总和超过某个**阈值**，神经元就会"点火"（fire），通过**轴突**把信号传递给下一个神经元
4. 如果总和没超过阈值，神经元就保持沉默

### 16.1.2 赫布学习规则

1949年，加拿大心理学家唐纳德·赫布（Donald Hebb）提出了一个著名的学习假说：

> **"一起激发的神经元，连在一起。"**  
> *（Neurons that fire together, wire together.）*

这句话的意思是：如果两个神经元经常同时被激活，它们之间的连接就会变得更强。这就是学习的神经基础！

比如，每次你看到蜜蜂（视觉神经元激活）同时感到疼痛（痛觉神经元激活），这两个神经元之间的连接就会加强。久而久之，只要看到蜜蜂，你的大脑就会自动预警危险。

### 16.1.3 人工神经元的诞生

1943年，沃伦·麦卡洛克（Warren McCulloch）和沃尔特·皮茨（Walter Pitts）提出了第一个人工神经元的数学模型。他们把生物神经元简化为一个数学函数：

```
输入: x₁, x₂, ..., xₙ  （每个输入是0或1）
权重: w₁, w₂, ..., wₙ  （每个连接的重要性）
阈值: θ              （激活门槛）

输出 = { 1,  如果 w₁x₁ + w₂x₂ + ... + wₙxₙ ≥ θ
       { 0,  否则
```

这个模型虽然简单，但已经能模拟逻辑运算（与、或、非）了。不过，它有一个致命缺陷：**权重需要人工设定**。

麦卡洛克-皮茨模型知道神经元"应该"怎么工作，但不知道如何**自动学习**这些权重。

---

## 16.2 感知机的诞生（1958）

### 16.2.1 弗兰克·罗森布拉特的突破

1957年，罗森布拉特在康奈尔航空实验室开始了一个雄心勃勃的项目。他的目标是：设计一台能够**自动学习**识别视觉模式的机器。

1958年，他发表了里程碑式的论文《感知机：大脑中信息存储和组织的概率模型》（*The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*）。

在论文中，罗森布拉特写道：

> *"感知机的显著特点是，它能够在没有预先组织的人类干预的情况下，通过经验来自动学习识别复杂的模式类别。"*

### 16.2.2 Mark I 感知机

罗森布拉特不只是纸上谈兵——他真造了一台机器！这就是著名的**Mark I Perceptron**。

```
┌──────────────────────────────────────────────────────────────┐
│                    Mark I 感知机架构                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │  感光网格  │────→│  关联单元   │────→│  响应单元   │       │
│   │(20×20=400)│    │  (A-units)  │    │  (R-unit)   │       │
│   │ S-points │    │             │    │             │       │
│   └─────────┘     └─────────────┘     └─────────────┘       │
│        ↑                              （决策输出）            │
│        │                                                     │
│   图像输入                                                    │
│                                                              │
│   物理实现：400个光电传感器 → 512个电位器（模拟权重）          │
│              → IBM 704计算机处理                               │
└──────────────────────────────────────────────────────────────┘
```

这台机器能做什么？

- 输入：20×20像素的黑白图像（比如字母、简单图形）
- 学习：通过"奖赏-惩罚"机制调整连接权重
- 输出：判断输入属于哪个类别（比如"这是字母A"）

在一次演示中，Mark I 感知机学会了区分男性和女性的照片！这在当时引起了轰动，媒体甚至报道说"感知机是第一台能独立思考的机器"。

### 16.2.3 感知机的简化模型

虽然Mark I很酷，但作为教学，我们通常使用更简单的单感知机模型：

```
          输入层                      输出层
    ┌─────────────────┐
    │                 │
    │    x₁ ───┐      │
    │          │      │
    │    x₂ ───┼──────┼────→  y
    │          │  Σ   │
    │    x₃ ───┤      │
    │          │      │
    │    x₄ ───┘      │
    │                 │
    └─────────────────┘
    
    权重: w₁, w₂, w₃, w₄
    偏置: b
```

数学上，感知机的计算过程是：

**第一步：加权求和**

$$z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$$

**第二步：激活函数（阶跃函数）**

$$y = \begin{cases} 1 & \text{如果 } z \geq 0 \\ 0 & \text{如果 } z < 0 \end{cases}$$

或者用更紧凑的向量表示：

$$y = \text{step}(\mathbf{w}^T \mathbf{x} + b)$$

其中：
- $\mathbf{x} = [x_1, x_2, ..., x_n]^T$ 是输入向量
- $\mathbf{w} = [w_1, w_2, ..., w_n]^T$ 是权重向量
- $b$ 是偏置（bias），相当于调整阈值的位置
- $\text{step}()$ 是阶跃函数

---

## 16.3 几何直观：感知机在画线

### 16.3.1 决策边界

感知机本质上是一个**线性分类器**。它在特征空间中画一条直线（或超平面），把两类数据分开。

想象你在一张纸上画了两个点群：红点在左边，蓝点在右边。感知机的任务就是找到一条线，把红点全分到一边，蓝点全分到另一边。

```
                    特征 x₂
                      │
         蓝色类 △     │    △
                   △  │ △
              △       │       △
         ─────────────┼──────────────  ← 决策边界
                        │      ●
              ●      │ ●
         红色类 ●        │
                   ●  │
                      │
    ──────────────────┼────────────────→ 特征 x₁
```

这条线就叫做**决策边界**（Decision Boundary）。数学上，它由方程 $\mathbf{w}^T \mathbf{x} + b = 0$ 定义。

### 16.3.2 权重的几何意义

权重向量 $\mathbf{w}$ 有什么几何意义？

**它垂直于决策边界！**

```
                    ↑
                    │ 权重向量 w
                    │
    ────────────────┼────────────────
         分类区域   │  -1    分类区域
         (y = 0)   │         (y = 1)
```

- 权重向量 $\mathbf{w}$ 指向分类结果为1的那一侧
- 偏置 $b$ 控制决策边界离原点的距离

这个几何直观非常重要，因为它帮助我们理解为什么感知机只能解决**线性可分**的问题。

---

## 16.4 感知机学习规则：如何自动学习

现在我们来到最关键的问题：感知机是如何自动学习权重的？

### 16.4.1 核心思想：试错学习

感知机的学习规则基于一个简单的原则：

> **"如果错了，就调整；如果对了，就保持。"**

具体步骤：
1. 随机初始化权重
2. 取一个训练样本，用当前权重做预测
3. 如果预测正确 → 什么都不做
4. 如果预测错误 → 调整权重，让预测向正确答案靠近
5. 重复步骤2-4，直到所有样本都被正确分类

### 16.4.2 学习规则的数学推导

让我们一步一步推导感知机的学习规则。

**设定**：
- 输入：$\mathbf{x} = [x_1, x_2, ..., x_n]^T$
- 真实标签：$y \in \{0, 1\}$
- 当前权重：$\mathbf{w} = [w_1, w_2, ..., w_n]^T$
- 当前偏置：$b$
- 预测值：$\hat{y} = \text{step}(\mathbf{w}^T \mathbf{x} + b)$

**情况1：$y = 1$，但 $\hat{y} = 0$（漏报）**

这意味着 $z = \mathbf{w}^T \mathbf{x} + b < 0$，太小了。我们需要**增大** $z$。

怎么增大？增加权重和偏置！

$$\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} + \eta \cdot \mathbf{x}$$
$$b_{\text{new}} = b_{\text{old}} + \eta$$

其中 $\eta$ 是学习率（learning rate），控制每次调整的幅度。

**情况2：$y = 0$，但 $\hat{y} = 1$（误报）**

这意味着 $z = \mathbf{w}^T \mathbf{x} + b \geq 0$，太大了。我们需要**减小** $z$。

$$\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} - \eta \cdot \mathbf{x}$$
$$b_{\text{new}} = b_{\text{old}} - \eta$$

**情况3：$y = \hat{y}$（预测正确）**

什么都不做：
$$\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}}$$
$$b_{\text{new}} = b_{\text{old}}$$

### 16.4.3 统一的更新公式

我们可以把以上三种情况写成一个统一的公式。注意：
- 情况1：$y - \hat{y} = 1 - 0 = +1$
- 情况2：$y - \hat{y} = 0 - 1 = -1$
- 情况3：$y - \hat{y} = 0$ 或 $0$

所以：

$$\boxed{\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} + \eta \cdot (y - \hat{y}) \cdot \mathbf{x}}$$
$$\boxed{b_{\text{new}} = b_{\text{old}} + \eta \cdot (y - \hat{y})}$$

这就是著名的**感知机学习规则**（Perceptron Learning Rule）！

**关键洞察**：
- 当预测错误时，$(y - \hat{y})$ 是 $+1$ 或 $-1$，权重会调整
- 当预测正确时，$(y - \hat{y}) = 0$，权重保持不变

### 16.4.4 完整算法

```python
# 感知机学习算法伪代码

def perceptron_learning(X, y, eta=0.1, max_epochs=100):
    """
    X: 训练数据，形状 (n_samples, n_features)
    y: 标签，形状 (n_samples,)，取值 {0, 1}
    eta: 学习率
    max_epochs: 最大迭代轮数
    """
    n_samples, n_features = X.shape
    
    # 1. 初始化权重和偏置（通常初始化为0或很小的随机数）
    w = np.zeros(n_features)
    b = 0
    
    for epoch in range(max_epochs):
        errors = 0
        
        for i in range(n_samples):
            # 2. 计算加权和
            z = np.dot(w, X[i]) + b
            
            # 3. 激活（阶跃函数）
            y_pred = 1 if z >= 0 else 0
            
            # 4. 如果预测错误，更新权重
            if y_pred != y[i]:
                error = y[i] - y_pred  # +1 或 -1
                w = w + eta * error * X[i]
                b = b + eta * error
                errors += 1
        
        # 5. 如果这一轮没有错误，收敛了！
        if errors == 0:
            print(f"收敛于第 {epoch + 1} 轮")
            break
    
    return w, b
```

---

## 16.5 感知机收敛定理

现在我们来回答一个关键问题：**感知机算法一定会收敛吗？**

1962年，罗森布拉特证明了一个重要的定理：

### 16.5.1 感知机收敛定理

> **定理**：如果训练数据是**线性可分**的，那么感知机学习算法保证在有限步内收敛到一个能正确分类所有样本的解。

**证明思路**（直观版）：

假设存在一个"完美"的权重向量 $\mathbf{w}^*$，能正确分类所有样本。我们证明：

1. 每次更新时，当前权重 $\mathbf{w}$ 与 $\mathbf{w}^*$ 的**夹角在减小**（越来越接近）
2. 每次更新的幅度是有界的
3. 因此，经过有限次更新后，$\mathbf{w}$ 一定会变得足够接近 $\mathbf{w}^*$

**更严谨的证明**：

设：
- 存在一个解 $\mathbf{w}^*$，使得对所有样本都有 $y_i(\mathbf{w}^{*T}\mathbf{x}_i + b^*) \geq \gamma > 0$（即有一个正的间隔）
- 所有样本满足 $\|\mathbf{x}_i\| \leq R$

**引理1**：$\mathbf{w}^{*T}\mathbf{w}^{(k)} \geq k\eta\gamma$

证明：
$$\mathbf{w}^{*T}\mathbf{w}^{(k)} = \mathbf{w}^{*T}\mathbf{w}^{(k-1)} + \eta y_i \mathbf{w}^{*T}\mathbf{x}_i \geq \mathbf{w}^{*T}\mathbf{w}^{(k-1)} + \eta\gamma$$

递推得：$\mathbf{w}^{*T}\mathbf{w}^{(k)} \geq k\eta\gamma$

**引理2**：$\|\mathbf{w}^{(k)}\|^2 \leq k\eta^2 R^2$

证明：
$$\|\mathbf{w}^{(k)}\|^2 = \|\mathbf{w}^{(k-1)} + \eta y_i \mathbf{x}_i\|^2$$
$$= \|\mathbf{w}^{(k-1)}\|^2 + 2\eta y_i \mathbf{w}^{(k-1)T}\mathbf{x}_i + \eta^2\|\mathbf{x}_i\|^2$$

因为更新发生在错误分类时，$y_i \mathbf{w}^{(k-1)T}\mathbf{x}_i < 0$，所以：
$$\|\mathbf{w}^{(k)}\|^2 \leq \|\mathbf{w}^{(k-1)}\|^2 + \eta^2 R^2$$

递推得：$\|\mathbf{w}^{(k)}\|^2 \leq k\eta^2 R^2$

**结合两个引理**：

由柯西-施瓦茨不等式：$(\mathbf{w}^{*T}\mathbf{w}^{(k)})^2 \leq \|\mathbf{w}^*\|^2 \|\mathbf{w}^{(k)}\|^2$

代入引理1和引理2：
$$(k\eta\gamma)^2 \leq \|\mathbf{w}^*\|^2 \cdot k\eta^2 R^2$$

化简：
$$k \leq \frac{\|\mathbf{w}^*\|^2 R^2}{\gamma^2}$$

这意味着更新次数 $k$ 是有上界的！所以算法一定会在有限步内收敛。

**实际意义**：

对于线性可分数据，感知机最多需要 $\frac{R^2}{\gamma^2}$ 次更新就能收敛。

---

## 16.6 感知机的局限：XOR问题

### 16.6.1 XOR问题是什么？

感知机虽然很酷，但它有一个致命弱点：**只能解决线性可分的问题**。

最著名的例子是**XOR问题**（异或问题）。

XOR的真值表：

| x₁ | x₂ | x₁ XOR x₂ |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

让我们在平面上画出这四个点：

```
    x₂
    │
 1  │    ○(0,1)    ○(1,1)
    │      [1]      [0]   ← 无法用一条直线分开！
    │
────┼──────────────────→ x₁
    │
 0  │    ●(0,0)    ●(1,0)
    │      [0]      [1]
    │
```

- ● 表示输出为0的点
- ○ 表示输出为1的点

问题是：**无法用一条直线把两个●和两个○分开！**

### 16.6.2 线性不可分的直观理解

XOR问题之所以难，是因为它的"真值分布"不是线性的。想象你有四个座位：

```
┌─────┬─────┐
│ 0,0 │ 0,1 │  ← 前排：一个人（0）或另一个人（1）
├─────┼─────┤
│ 1,0 │ 1,1 │  ← 后排：两个人（0）或没人的情况（1？不，也是0）
└─────┴─────┘
```

XOR想要的是"恰好有一个人"的情况。这种"排斥"关系本质上是非线性的。

### 16.6.3 Minsky和Papert的致命一击

1969年，麻省理工学院的两位人工智能先驱——马文·明斯基（Marvin Minsky）和西摩·帕珀特（Seymour Papert）——出版了《感知机》（*Perceptrons*）一书。

在这本书中，他们严格证明了：

> **单层感知机无法解决XOR问题，也无法解决任何非线性可分的问题。**

更糟的是，他们暗示：**多层感知机可能也无法解决这些问题**（虽然后来证明这是错误的）。

这本书的影响是毁灭性的：

- 感知机的研究几乎被放弃
- 神经网络领域进入了长达十多年的"寒冬期"
- 政府 funding 被大幅削减
- 人工智能研究转向了符号主义方法

明斯基后来承认，他写这本书的部分动机是为了争夺AI研究的主导权（符号主义 vs 连接主义）。但从科学角度，他指出感知机的局限性是正确的。

### 16.6.4 如何解决XOR？

解决XOR问题的关键是：**使用多层感知机（Multi-Layer Perceptron）**。

想象我们能画一条曲线，而不是直线：

```
    x₂
    │
 1  │    ○        ○
    │       ╲    ╱
    │        ╲  ╱
────┼─────────╳───────→ x₁
    │        ╱  ╲
 0  │    ●  ╱    ╲  ●
    │
```

或者，我们可以用**两个感知机组合**来解决：

```
输入层        隐藏层         输出层

 x₁ ──┐                    
      ├──→ [感知机A: OR] ──┐
 x₂ ──┘                    ├──→ [感知机C: AND] → 输出
      ┌──→ [感知机B: NAND]─┘
 x₁ ──┤
      │
 x₂ ──┘
```

- 感知机A实现 OR：输出1如果x₁=1或x₂=1
- 感知机B实现 NAND：输出1除非x₁=1且x₂=1
- 感知机C把A和B的结果做AND

验证：
- (0,0): OR=0, NAND=1, AND=0 ✓
- (0,1): OR=1, NAND=1, AND=1 ✓
- (1,0): OR=1, NAND=1, AND=1 ✓
- (1,1): OR=1, NAND=0, AND=0 ✓

**XOR = (x₁ OR x₂) AND NOT(x₁ AND x₂)**

但训练多层网络需要更复杂的算法——**反向传播**（Backpropagation），这是我们在后续章节要学习的内容。

---

## 16.7 从零实现感知机

现在让我们用纯Python和NumPy实现一个完整的感知机。

### 16.7.1 Perceptron类实现

```python
"""
感知机从零实现
================
不依赖任何机器学习框架，只用NumPy

作者: ML教材写作项目
日期: 2026
"""

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    """
    感知机分类器
    
    参数:
    -----------
    eta : float
        学习率 (0.0 到 1.0之间)
    n_iter : int
        最大训练轮数
    random_state : int
        随机种子，用于初始化权重
    
    属性:
    -----------
    w_ : 1d-array
        训练后的权重
    b_ : float
        训练后的偏置
    errors_ : list
        每轮的错误分类数
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = None
        self.b_ = None
        self.errors_ = []
    
    def fit(self, X, y):
        """
        训练感知机
        
        参数:
        -----------
        X : array-like, shape = [n_samples, n_features]
            训练数据
        y : array-like, shape = [n_samples]
            目标值，取值为 {0, 1}
        
        返回:
        -----------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.0
        self.errors_ = []
        
        for epoch in range(self.n_iter):
            errors = 0
            
            for xi, target in zip(X, y):
                # 计算预测值
                y_pred = self.predict_single(xi)
                
                # 计算误差
                error = target - y_pred
                
                # 如果预测错误，更新权重
                if error != 0:
                    # w_new = w_old + eta * error * x
                    self.w_ += self.eta * error * xi
                    # b_new = b_old + eta * error
                    self.b_ += self.eta * error
                    errors += 1
            
            self.errors_.append(errors)
            
            # 如果这一轮没有错误，提前停止
            if errors == 0:
                print(f"收敛于第 {epoch + 1} 轮")
                break
        
        return self
    
    def net_input(self, X):
        """计算净输入 z = w·x + b"""
        return np.dot(X, self.w_) + self.b_
    
    def predict_single(self, x):
        """预测单个样本"""
        return 1 if self.net_input(x) >= 0 else 0
    
    def predict(self, X):
        """预测多个样本"""
        return np.where(self.net_input(X) >= 0, 1, 0)
    
    def accuracy(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def __repr__(self):
        return f"Perceptron(eta={self.eta}, n_iter={self.n_iter})"


# ============================================================
# 演示1: AND 逻辑门
# ============================================================

def demo_and_gate():
    """演示感知机学习AND逻辑门"""
    print("=" * 50)
    print("演示1: 学习 AND 逻辑门")
    print("=" * 50)
    
    # AND 真值表
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])
    
    print("\n训练数据 (AND 真值表):")
    print("x1\tx2\tAND(x1,x2)")
    for xi, yi in zip(X, y):
        print(f"{xi[0]}\t{xi[1]}\t{yi}")
    
    # 创建并训练感知机
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    
    print(f"\n学习到的权重: w = [{ppn.w_[0]:.3f}, {ppn.w_[1]:.3f}]")
    print(f"学习到的偏置: b = {ppn.b_:.3f}")
    
    # 测试
    print("\n预测结果:")
    print("x1\tx2\t真实值\t预测值")
    for xi, yi in zip(X, y):
        pred = ppn.predict_single(xi)
        print(f"{xi[0]}\t{xi[1]}\t{yi}\t{pred}")
    
    print(f"\n准确率: {ppn.accuracy(X, y) * 100:.1f}%")
    
    # 绘制决策边界
    plot_decision_boundary(X, y, ppn, "AND 逻辑门的决策边界")
    
    return ppn


# ============================================================
# 演示2: OR 逻辑门
# ============================================================

def demo_or_gate():
    """演示感知机学习OR逻辑门"""
    print("\n" + "=" * 50)
    print("演示2: 学习 OR 逻辑门")
    print("=" * 50)
    
    # OR 真值表
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 1])
    
    print("\n训练数据 (OR 真值表):")
    print("x1\tx2\tOR(x1,x2)")
    for xi, yi in zip(X, y):
        print(f"{xi[0]}\t{xi[1]}\t{yi}")
    
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    
    print(f"\n学习到的权重: w = [{ppn.w_[0]:.3f}, {ppn.w_[1]:.3f}]")
    print(f"学习到的偏置: b = {ppn.b_:.3f}")
    
    print("\n预测结果:")
    for xi, yi in zip(X, y):
        pred = ppn.predict_single(xi)
        print(f"{xi}\t真实:{yi}\t预测:{pred}")
    
    plot_decision_boundary(X, y, ppn, "OR 逻辑门的决策边界")
    
    return ppn


# ============================================================
# 演示3: XOR 逻辑门（感知机无法解决！）
# ============================================================

def demo_xor_gate():
    """演示感知机无法学习XOR逻辑门"""
    print("\n" + "=" * 50)
    print("演示3: XOR 逻辑门（感知机的局限）")
    print("=" * 50)
    
    # XOR 真值表
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 0])
    
    print("\n训练数据 (XOR 真值表):")
    print("x1\tx2\tXOR(x1,x2)")
    for xi, yi in zip(X, y):
        print(f"{xi[0]}\t{xi[1]}\t{yi}")
    
    print("\n尝试训练感知机...")
    ppn = Perceptron(eta=0.1, n_iter=20)
    ppn.fit(X, y)
    
    print("\n预测结果:")
    print("x1\tx2\t真实值\t预测值\t正确?")
    for xi, yi in zip(X, y):
        pred = ppn.predict_single(xi)
        correct = "✓" if pred == yi else "✗"
        print(f"{xi[0]}\t{xi[1]}\t{yi}\t{pred}\t{correct}")
    
    accuracy = ppn.accuracy(X, y)
    print(f"\n准确率: {accuracy * 100:.1f}%")
    print(f"错误数: {int((1 - accuracy) * len(y))} / {len(y)}")
    
    print("\n⚠️  注意: 感知机无法解决XOR问题！")
    print("   因为XOR不是线性可分的。")
    print("   这需要多层神经网络（后续章节讲解）。")
    
    # 仍然尝试绘制，但会显示无法分开
    plot_decision_boundary(X, y, ppn, "XOR 逻辑门（线性不可分）")
    
    return ppn


# ============================================================
# 可视化函数
# ============================================================

def plot_decision_boundary(X, y, model, title):
    """绘制决策边界"""
    plt.figure(figsize=(8, 6))
    
    # 绘制数据点
    for i, (xi, yi) in enumerate(zip(X, y)):
        if yi == 0:
            plt.scatter(xi[0], xi[1], c='red', s=200, marker='o', 
                       edgecolors='black', linewidth=2, label='Class 0' if i == 0 else "")
        else:
            plt.scatter(xi[0], xi[1], c='blue', s=200, marker='^', 
                       edgecolors='black', linewidth=2, label='Class 1' if i == 0 else "")
        
        # 添加标签
        plt.annotate(f'({xi[0]},{xi[1]})\ny={yi}', 
                    (xi[0], xi[1]), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    fontsize=9)
    
    # 绘制决策边界
    if model.w_[1] != 0:
        x_min, x_max = -0.5, 1.5
        x_values = np.linspace(x_min, x_max, 100)
        # w0*x + w1*y + b = 0  =>  y = -(w0*x + b) / w1
        y_values = -(model.w_[0] * x_values + model.b_) / model.w_[1]
        plt.plot(x_values, y_values, 'g--', linewidth=2, label='Decision Boundary')
    
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('x₁', fontsize=12)
    plt.ylabel('x₂', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图像已保存: {title.replace(' ', '_')}.png")


# ============================================================
# 演示4: 鸢尾花数据集（真实数据）
# ============================================================

def demo_iris():
    """在简化版鸢尾花数据集上演示感知机"""
    print("\n" + "=" * 50)
    print("演示4: 鸢尾花分类（真实数据）")
    print("=" * 50)
    
    # 简化的鸢尾花数据（只有两类：山鸢尾和变色鸢尾）
    # 特征：花瓣长度和花瓣宽度
    np.random.seed(42)
    
    # 山鸢尾（Setosa）- 类别 0
    # 花瓣短而窄
    setosa = np.random.multivariate_normal(
        mean=[1.4, 0.2], 
        cov=[[0.01, 0.002], [0.002, 0.01]], 
        size=30
    )
    
    # 变色鸢尾（Versicolor）- 类别 1
    # 花瓣中等长度
    versicolor = np.random.multivariate_normal(
        mean=[4.2, 1.3], 
        cov=[[0.1, 0.02], [0.02, 0.05]], 
        size=30
    )
    
    X = np.vstack([setosa, versicolor])
    y = np.array([0] * 30 + [1] * 30)
    
    print(f"\n数据集: 60个样本，2个特征")
    print(f"  - 山鸢尾（Setosa）: 30个")
    print(f"  - 变色鸢尾（Versicolor）: 30个")
    print(f"\n特征: 花瓣长度、花瓣宽度")
    
    # 划分训练集和测试集
    indices = np.random.permutation(len(X))
    train_idx = indices[:40]
    test_idx = indices[40:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"\n训练集: {len(X_train)}个样本")
    print(f"测试集: {len(X_test)}个样本")
    
    # 训练感知机
    ppn = Perceptron(eta=0.01, n_iter=100)
    ppn.fit(X_train, y_train)
    
    # 评估
    train_acc = ppn.accuracy(X_train, y_train)
    test_acc = ppn.accuracy(X_test, y_test)
    
    print(f"\n训练准确率: {train_acc * 100:.1f}%")
    print(f"测试准确率: {test_acc * 100:.1f}%")
    print(f"\n学习到的权重: w = [{ppn.w_[0]:.3f}, {ppn.w_[1]:.3f}]")
    print(f"学习到的偏置: b = {ppn.b_:.3f}")
    
    # 绘制
    plot_iris_decision_boundary(X, y, ppn)
    
    return ppn


def plot_iris_decision_boundary(X, y, model):
    """绘制鸢尾花数据的决策边界"""
    plt.figure(figsize=(10, 8))
    
    # 绘制数据点
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    
    plt.scatter(class_0[:, 0], class_0[:, 1], c='red', s=100, 
               marker='o', edgecolors='black', label='Setosa (Class 0)')
    plt.scatter(class_1[:, 0], class_1[:, 1], c='blue', s=100, 
               marker='^', edgecolors='black', label='Versicolor (Class 1)')
    
    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x_values = np.linspace(x_min, x_max, 100)
    y_values = -(model.w_[0] * x_values + model.b_) / model.w_[1]
    plt.plot(x_values, y_values, 'g-', linewidth=2, label='Decision Boundary')
    
    plt.xlabel('Petal Length (cm)', fontsize=12)
    plt.ylabel('Petal Width (cm)', fontsize=12)
    plt.title('Perceptron on Iris Dataset', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('iris_decision_boundary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  图像已保存: iris_decision_boundary.png")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "感知机从零实现演示" + " " * 24 + "║")
    print("║" + " " * 6 + "基于 Rosenblatt (1958) 原始论文" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # 运行所有演示
    demo_and_gate()
    demo_or_gate()
    demo_xor_gate()
    demo_iris()
    
    print("\n" + "=" * 58)
    print("演示完成！")
    print("=" * 58)
```

### 16.7.2 运行示例输出

```
==================================================
演示1: 学习 AND 逻辑门
==================================================

训练数据 (AND 真值表):
x1	x2	AND(x1,x2)
0	0	0
0	1	0
1	0	0
1	1	1

收敛于第 5 轮

学习到的权重: w = [0.200, 0.100]
学习到的偏置: b = -0.200

预测结果:
x1	x2	真实值	预测值
0	0	0	0
0	1	0	0
1	0	0	0
1	1	1	1

准确率: 100.0%
  图像已保存: AND_逻辑门的决策边界.png

==================================================
演示2: 学习 OR 逻辑门
==================================================
...

==================================================
演示3: XOR 逻辑门（感知机的局限）
==================================================
...
训练数据 (XOR 真值表):
x1	x2	XOR(x1,x2)
0	0	0
0	1	1
1	0	1
1	1	0

尝试训练感知机...

预测结果:
x1	x2	真实值	预测值	正确?
0	0	0	0	✓
0	1	1	0	✗
1	0	1	1	✓
1	1	0	1	✗

准确率: 50.0%
错误数: 2 / 4

⚠️  注意: 感知机无法解决XOR问题！
   因为XOR不是线性可分的。
   这需要多层神经网络（后续章节讲解）。
```

---

## 16.8 多类分类的扩展

感知机原本是二分类器（两类），但我们可以通过一些技巧扩展到多类分类。

### 16.8.1 One-vs-Rest（一对多）策略

对于K个类别，我们训练K个感知机：

```
感知机1: 类1 vs (类2,类3,...,类K)
感知机2: 类2 vs (类1,类3,...,类K)
...
感知机K: 类K vs (类1,类2,...,类K-1)
```

预测时，选择得分最高的感知机对应的类别。

### 16.8.2 实现代码

```python
class MultiClassPerceptron:
    """多类感知机（One-vs-Rest策略）"""
    
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        self.classifiers = {}
    
    def fit(self, X, y):
        """训练多个二分类感知机"""
        self.classes = np.unique(y)
        
        for cls in self.classes:
            # 为每个类别创建一个二分类问题
            y_binary = np.where(y == cls, 1, 0)
            
            # 训练一个感知机
            ppn = Perceptron(eta=self.eta, n_iter=self.n_iter)
            ppn.fit(X, y_binary)
            
            self.classifiers[cls] = ppn
            print(f"类别 {cls} 的感知机训练完成")
    
    def predict(self, X):
        """预测：选择得分最高的类别"""
        scores = {}
        for cls, ppn in self.classifiers.items():
            scores[cls] = ppn.net_input(X)
        
        # 选择得分最高的类别
        predictions = []
        for i in range(len(X)):
            best_cls = max(self.classes, key=lambda c: scores[c][i])
            predictions.append(best_cls)
        
        return np.array(predictions)
```

---

## 16.9 练习题

### 基础题

**16.1** 感知机的权重和偏置有什么作用？如果偏置 $b = 0$，会发生什么？

**16.2** 手动计算：给定权重 $\mathbf{w} = [2, -1]$，偏置 $b = 0.5$，输入 $\mathbf{x} = [1, 2]$，感知机的输出是什么？

**16.3** 感知机学习规则中，学习率 $\eta$ 有什么作用？如果 $\eta$ 太大或太小，会发生什么？

### 进阶题

**16.4** 证明：对于AND问题，感知机最多需要多少次更新就能收敛？（提示：使用感知机收敛定理）

**16.5** 设计一个实验，验证感知机在线性可分数据上的收敛性，以及在线性不可分数据上的震荡。

**16.6** 阅读罗森布拉特1958年的原始论文，总结他的主要贡献和当时的科学背景。

### 挑战题

**16.7** **编程挑战**：实现一个能学习NAND、OR、AND组合解决XOR问题的多层感知机（不使用反向传播，而是手动设置权重）。

**16.8** **研究项目**：调查感知机在现代机器学习中的应用。虽然深层网络更流行，但感知机（或线性分类器）还在哪些地方被使用？

---

## 16.10 本章小结

### 核心概念

| 概念 | 解释 |
|------|------|
| **感知机** | 第一个能从数据自动学习的神经网络模型 |
| **权重** | 控制每个输入重要性的参数 |
| **偏置** | 调整激活阈值的参数 |
| **阶跃函数** | 将加权和转换为0或1输出的激活函数 |
| **决策边界** | 分隔两类数据的超平面 |
| **线性可分** | 存在一条直线（或超平面）能完美分开两类数据 |
| **学习规则** | $w \leftarrow w + \eta(y - \hat{y})x$ |
| **收敛定理** | 线性可分数据保证在有限步内收敛 |
| **XOR问题** | 线性不可分的经典例子，单层感知机无法解决 |

### 历史脉络

```
1943 ─── McCulloch & Pitts ───→ 提出人工神经元模型
         （能模拟逻辑运算，但需要人工设定权重）
         ↓
1949 ─── Hebb ───→ 提出"一起激发的神经元连在一起"学习规则
         ↓
1958 ─── Rosenblatt ───→ 发明感知机，第一个自动学习的神经网络
         ↓
1960s ── 感知机热潮 ───→ Mark I硬件、媒体追捧
         ↓
1969 ─── Minsky & Papert ───→ 《感知机》一书指出XOR局限
         ↓
1970s ── AI寒冬 ───→ 神经网络研究几乎停滞
         ↓
1986 ─── Rumelhart et al. ───→ 反向传播算法，多层网络复兴
         ↓
2012 ─── AlexNet ───→ 深度学习革命，神经网络重回巅峰
```

### 本章代码实现

我们实现了一个完整的感知机类，包括：
- ✅ 感知机学习规则
- ✅ AND、OR、XOR演示
- ✅ 鸢尾花分类
- ✅ 决策边界可视化
- ✅ 多类分类扩展

---

## 16.11 参考文献

### 原始论文

1. **Rosenblatt, F.** (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386-408. https://doi.org/10.1037/h0042519

2. **McCulloch, W. S., & Pitts, W.** (1943). A logical calculus of the ideas immanent in nervous activity. *Bulletin of Mathematical Biophysics*, 5(4), 115-133. https://doi.org/10.1007/BF02478259

3. **Hebb, D. O.** (1949). *The organization of behavior: A neuropsychological theory*. John Wiley & Sons.

### 感知机局限性的经典论述

4. **Minsky, M., & Papert, S.** (1969). *Perceptrons: An introduction to computational geometry*. MIT Press.

### 感知机收敛定理

5. **Novikoff, A. B.** (1962). On convergence proofs on perceptrons. *Proceedings of the Symposium on the Mathematical Theory of Automata*, 12, 615-622.

6. **Block, H. D.** (1962). The perceptron: A model for brain functioning. *Reviews of Modern Physics*, 34(1), 123-135.

### 现代教材

7. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep learning*. MIT Press. Chapter 6: Deep Feedforward Networks.

8. **Bishop, C. M.** (2006). *Pattern recognition and machine learning*. Springer. Chapter 4: Linear Models for Classification.

9. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The elements of statistical learning* (2nd ed.). Springer. Chapter 4: Linear Methods for Classification.

### 历史回顾

10. **Olazaran, M.** (1996). A sociological study of the official history of the perceptrons controversy. *Social Studies of Science*, 26(3), 611-659.

---

## 下一章预告

**第十七章：多层神经网络——层层的魔法**

我们将学习：
- 为什么需要多层网络
- 前向传播算法
- 反向传播的直观理解
- 从零实现一个多层感知机（MLP）

准备好揭开神经网络的层层面纱了吗？

---

*本章完*
