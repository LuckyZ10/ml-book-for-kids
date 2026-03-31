# 第二十三章：循环神经网络——序列的舞者

## 章节引言

想象一下，当你读这句话的时候，你是如何理解它的？你不是孤立地看每一个字，而是把前面的词记在脑海里，用它们来理解后面的词。

比如读到"苹果"这个词时：
- 如果前文是"我今天吃了一个红彤彤的"，你会想到**水果**
- 如果前文是"苹果公司发布了新款"，你会想到**科技公司**

**这就是记忆的力量！**

我们之前学过的神经网络——无论是多层感知机还是卷积神经网络——都有一个共同特点：**没有记忆**。它们像一位只看眼前事物的老师，处理每个输入时都"重新开始"，无法记住之前看到的内容。

但在现实世界中，很多数据都是**序列**形式的：
- 一句话的词是一个接一个出现的
- 股票价格是随着时间变化的
- 音乐是一段音符一段音符流动的

**循环神经网络（Recurrent Neural Network, RNN）**就是专门为了解决这类问题而诞生的。它像一位有记忆的人，能够记住之前看到的信息，并用这些信息来理解当前的内容。

在本章中，我们将：
- 🎭 用生活化的比喻理解RNN的记忆机制
- 📜 探索从1986年Jordan网络到2014年GRU的演进历程
- 🧮 完整推导BPTT（随时间反向传播）算法
- 🔓 深入LSTM的门控机制，理解它如何解决梯度消失问题
- 💻 用纯NumPy从零实现RNN、LSTM、GRU
- 🎯 训练字符级语言模型和时间序列预测器

准备好了吗？让我们和这位"序列的舞者"共舞吧！

---

## 23.1 为什么需要序列模型？

### 23.1.1 生活中的序列数据

在我们周围，序列数据无处不在：

**语言序列** 📝
```
我 → 喜欢 → 机器 → 学习
```
理解"学习"这个词，需要记住前面的"机器"。

**时间序列** 📈
```
[周一: 100元] → [周二: 102元] → [周三: 98元] → [周四: ?]
```
预测明天股价，需要参考过去几天的走势。

**音乐序列** 🎵
```
Do → Re → Mi → Fa → Sol → ?
```
下一音符是什么？取决于之前的旋律走向。

**DNA序列** 🧬
```
ATCG → GCTA → TAAT → ...
```
基因编码是一连串碱基的组合。

### 23.1.2 传统神经网络的局限

让我们看看为什么传统的神经网络无法处理序列数据：

**输入长度固定问题**
- 普通神经网络需要固定大小的输入
- 但句子有长有短："你好" vs "今天天气真不错"

**没有记忆问题**
- 处理每个词时都是"重新开始"
- 无法捕捉"苹果"在不同上下文中的不同含义

**位置信息丢失**
- "我爱机器学习"和"机器学习爱我"含义完全不同
- 但普通神经网络会把它们当作同样的词袋（Bag of Words）

### 23.1.3 序列模型的核心思想

序列模型的关键思想很简单：**把前一个时刻的信息传递到下一个时刻**。

就像你读小说时：
1. 读到第一章，记住主要人物和背景
2. 读到第二章，用第一章的记忆来理解新情节
3. 读到第三章，记忆继续累积和更新
4. ...

**循环神经网络（RNN）**正是基于这个思想：它有一个"隐藏状态"（Hidden State），就像一个记忆的容器，随着序列的推进不断更新。

---

## 23.2 经典文献研究：从Jordan到GRU

### 23.2.1 Jordan网络（1986）：序列建模的先驱

**文献信息**
> Jordan, M. I. (1986). Serial order: A parallel distributed processing approach (Technical Report No. 8604). Institute for Cognitive Science, University of California, San Diego.

**背景故事**

Michael I. Jordan（后来成为机器学习领域的传奇人物）在1986年提出了**Jordan网络**，这是最早的循环神经网络架构之一。

**核心思想**

Jordan网络的创新在于引入了**上下文单元（Context Units）**：

```
输入 x_t → 隐藏层 → 输出 y_t → 上下文单元 → 反馈到隐藏层
```

**数学公式**

$$
y_t = f(W_y h_t + b_y)
$$

$$
h_t = f(W_x x_t + W_c c_t + b_h)
$$

$$
c_{t+1} = \alpha c_t + y_t \quad \text{(上下文更新)}
$$

其中 $c_t$ 是上下文单元，它保存了**前一时刻的输出**，并随时间缓慢衰减（由 $\alpha$ 控制）。

**比喻理解** 🎯

想象一位老师在批改作文：
- 他每读完一段（输出 $y_t$），就会在便签上写一段评语
- 这个便签（上下文单元）会传递到下一段的批改中
- 这样，老师对下一段的理解就带有了前面内容的"记忆"

**历史意义**

Jordan网络证明了神经网络可以处理序列数据，但由于上下文单元只保存输出而非隐藏状态，它的记忆能力有限。这为后来的Elman网络奠定了基础。

---

### 23.2.2 Elman网络（1990）：现代RNN的雏形

**文献信息**
> Elman, J. L. (1990). Finding structure in time. *Cognitive Science*, 14(2), 179-211.

**背景故事**

Jeffrey L. Elman是加州大学圣迭戈分校的心理学和认知科学教授。1990年，他发表了这篇极具影响力的论文，提出了后来被广泛称为**简单循环网络（Simple Recurrent Network, SRN）**或**Elman网络**的架构。

**核心创新**

Elman的关键改进是：**让隐藏层自我循环**！

```
        ┌─────────────────┐
        ↓                 │
输入 x_t → 隐藏层 h_t → 输出 y_t
             ↑
             └──────┘
           (自我循环)
```

**数学公式（现代形式）**

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$

$$
y_t = W_{hy} h_t + b_y
$$

**关键区别**：Jordan网络循环的是**输出**，Elman网络循环的是**隐藏状态**。

**为什么这很重要？** 🤔

隐藏状态 $h_t$ 是网络的"内部表征"，它编码了输入的本质特征。让隐藏状态自我循环意味着：
- 网络可以学习**什么样的信息值得记住**
- 记忆不是固定的输出，而是可学习的表征

**Elman的经典实验**

Elman做了一个著名的"句子预测"实验：

```
训练句子：
- "男孩看见女孩" (boy sees girl)
- "男孩看见房子" (boy sees house)
- "女孩看见男孩" (girl sees boy)
...
```

网络学会预测句子中的下一个词。重要的是，隐藏状态演化出了一个**隐式的语法结构**——网络"理解"了主语、动词、宾语的概念！

**历史意义**

Elman网络是现代RNN的直接祖先。今天当我们说"RNN"时，通常指的就是这种Elman-style架构。论文被引用超过15000次，是连接主义（Connectionism）和神经网络发展史上的里程碑。

---

### 23.2.3 梯度消失问题（1991）：RNN的阿喀琉斯之踵

**文献信息**
> Hochreiter, S. (1991). Untersuchungen zu dynamischen neuronalen Netzen [Studies on dynamic neural networks]. Diploma thesis, Technical University of Munich, Germany.

**背景故事**

Sepp Hochreiter当时是慕尼黑工业大学的学生，他的 Diploma 论文（相当于硕士论文）中首次系统分析了RNN训练中的一个根本性问题——**梯度消失（Vanishing Gradient）**。

**问题描述**

在训练RNN时，我们需要将误差沿着时间反向传播（Backpropagation Through Time, BPTT）。对于长序列，这意味着要连乘很多个雅可比矩阵：

$$
\frac{\partial L}{\partial W} \propto \prod_{k=1}^{t} \frac{\partial h_k}{\partial h_{k-1}}
$$

如果特征值小于1，多次相乘后梯度会指数级减小（消失）；如果大于1，梯度会指数级增大（爆炸）。

**比喻理解** 📉

想象你要告诉100年前的祖先一个消息：
- 你告诉父亲，父亲告诉祖父，祖父告诉曾祖父...
- 每一代人传递时都会"遗忘"一部分细节
- 传到第100代时，消息已经面目全非

这就是梯度消失——长距离的信息无法有效传递！

**后果**

梯度消失意味着RNN：
- 无法学习**长距离依赖**（如句子开头的主语和结尾的动词一致性）
- 实际上只能记住最近几个时间步的信息
- 成为"短期记忆"网络，而非真正的"长期记忆"

**历史意义**

Hochreiter的发现促使他后来与Jürgen Schmidhuber合作，在1997年发明了**LSTM**，彻底解决了这个问题。

---

### 23.2.4 LSTM：长短期记忆网络（1997）

**文献信息**
> Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

**引用盛况**：超过67,000次引用，是深度学习领域最具影响力的论文之一！

**核心创新**

LSTM通过引入**门控机制（Gating Mechanism）**和**细胞状态（Cell State）**解决了梯度消失问题：

```
细胞状态 C_t：信息的"高速公路"，几乎不变地传递
        ↓
遗忘门 f_t：决定丢弃什么旧信息
输入门 i_t：决定添加什么新信息  
输出门 o_t：决定输出什么信息
```

**关键思想**

LSTM创造了一个**加法机制**来更新记忆：

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

注意是**加号**而不是乘号！这意味着梯度可以在细胞状态上**几乎不变地流动**，不会消失或爆炸。

**比喻理解** 🛣️

想象一条高速公路（细胞状态）：
- 车辆可以几乎无阻力地行驶很长距离
- 收费站（门）控制哪些车辆可以上/下高速
- 即使路很长，车速（梯度）也不会衰减

**历史影响**

LSTM在2010年代主导了序列建模领域，应用包括：
- 语音识别（Google Voice Search）
- 机器翻译（Google Neural Machine Translation）
- 手写识别
- 音乐生成
- 文本生成

---

### 23.2.5 GRU：门控循环单元（2014）

**文献信息**
> Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. *arXiv preprint arXiv:1412.3555*.

**动机**

LSTM虽然很强大，但结构复杂：
- 3个门（遗忘门、输入门、输出门）
- 2个状态（细胞状态和隐藏状态）
- 参数量较大，计算成本高

能否简化结构但保持性能？

**GRU的创新**

GRU（Gated Recurrent Unit）将门控机制简化为：
- **更新门** $z_t$：控制保留多少旧状态
- **重置门** $r_t$：控制忽略多少旧状态

并且**合并了细胞状态和隐藏状态**！

**优点**

1. **参数更少**：约25%的参数减少
2. **结构更简单**：只有两个门
3. **效果相当**：在许多任务上与LSTM性能相当甚至更好

**历史地位**

GRU证明了门控机制是LSTM成功的关键，而不是特定的三个门结构。今天，GRU因其简洁性而广受欢迎，尤其在资源受限的场景中。

---

## 23.3 基础RNN（Elman网络）

### 23.3.1 RNN的结构展开

RNN最直观的理解方式是将其**按时间展开（Unrolling）**：

```
单步视角（压缩形式）：

      ┌──────────────┐
      ↓              │
    ┌─────┐         │
──→ │ RNN │ ────────┘
x_t └─────┘  h_t
       ↓
      y_t

多步展开（展开形式）：

x_0 ─→ [RNN] ─→ h_0 ─→ y_0
            ↑
x_1 ─→ [RNN] ─→ h_1 ─→ y_1
            ↑
x_2 ─→ [RNN] ─→ h_2 ─→ y_2
            ↑
           ...
```

**关键洞察**：展开后的RNN就像一个很深的**前馈网络**，每一层对应一个时间步，层与层之间共享参数！

### 23.3.2 前向传播数学公式

**单步计算**

给定输入序列 $\mathbf{x} = (x_1, x_2, ..., x_T)$，RNN的前向传播如下：

**隐藏状态更新**：
$$
\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)
$$

**输出计算**：
$$
\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y
$$

其中：
- $\mathbf{x}_t \in \mathbb{R}^{d_{in}}$：时刻$t$的输入
- $\mathbf{h}_t \in \mathbb{R}^{d_{hidden}}$：时刻$t$的隐藏状态
- $\mathbf{y}_t \in \mathbb{R}^{d_{out}}$：时刻$t$的输出
- $\mathbf{W}_{hh} \in \mathbb{R}^{d_{hidden} \times d_{hidden}}$：隐藏层到隐藏层的权重
- $\mathbf{W}_{xh} \in \mathbb{R}^{d_{hidden} \times d_{in}}$：输入到隐藏层的权重
- $\mathbf{W}_{hy} \in \mathbb{R}^{d_{out} \times d_{hidden}}$：隐藏层到输出的权重

**初始状态**

通常 $\mathbf{h}_0 = \mathbf{0}$（零向量初始化）或作为可学习参数。

### 23.3.3 参数共享的力量

注意：所有时间步共享同一组参数 $(\mathbf{W}_{hh}, \mathbf{W}_{xh}, \mathbf{W}_{hy}, \mathbf{b}_h, \mathbf{b}_y)$！

**这带来了什么好处？**

1. **参数量不随序列长度增加**
2. **能够处理变长序列**
3. **学习到时间不变的模式**

**比喻理解** 🔄

想象一位钢琴家弹奏一首曲子：
- 他用同一套手指技巧（共享参数）
- 处理每一个音符（每个时间步）
- 不管曲子多长，他的"技能"是通用的

### 23.3.4 BPTT算法推导

BPTT（Backpropagation Through Time，随时间反向传播）是训练RNN的核心算法。它是标准反向传播在展开网络上的应用。

**损失函数**

对于序列预测任务，总损失是每个时间步损失之和：

$$
L = \sum_{t=1}^{T} L_t
$$

其中 $L_t$ 是时刻 $t$ 的损失（如交叉熵或MSE）。

**隐藏状态的梯度**

关键在于计算 $\frac{\partial L}{\partial \mathbf{h}_t}$。由于隐藏状态影响当前输出和未来所有输出：

$$
\frac{\partial L}{\partial \mathbf{h}_t} = \frac{\partial L_t}{\partial \mathbf{h}_t} + \frac{\partial L_{t+1:T}}{\partial \mathbf{h}_t}
$$

第二项通过链式法则展开：

$$
\frac{\partial L_{t+1:T}}{\partial \mathbf{h}_t} = \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \cdot \frac{\partial L_{t+1:T}}{\partial \mathbf{h}_{t+1}}
$$

**完整递归公式**

令 $\delta_t = \frac{\partial L}{\partial \mathbf{h}_t}$，则有：

$$
\delta_t = \mathbf{W}_{hy}^T \frac{\partial L_t}{\partial \mathbf{y}_t} + \mathbf{W}_{hh}^T \cdot \delta_{t+1} \odot (1 - \tanh^2(\mathbf{h}_{t+1}))
$$

其中 $\odot$ 表示逐元素乘法。

**权重梯度**

$$
\frac{\partial L}{\partial \mathbf{W}_{hy}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial \mathbf{y}_t} \mathbf{h}_t^T
$$

$$
\frac{\partial L}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \delta_t \mathbf{h}_{t-1}^T
$$

$$
\frac{\partial L}{\partial \mathbf{W}_{xh}} = \sum_{t=1}^{T} \delta_t \mathbf{x}_t^T
$$

**为什么梯度会消失？** 📉

观察递归项：

$$
\delta_t \propto \mathbf{W}_{hh}^T \cdot \delta_{t+1} \odot \text{diag}(1 - \tanh^2(\mathbf{h}_{t+1}))
$$

多次迭代后：

$$
\delta_1 \propto (\mathbf{W}_{hh}^T)^T \cdot \text{(其他项)}
$$

如果 $\mathbf{W}_{hh}$ 的特征值小于1，$(\mathbf{W}_{hh}^T)^T$ 会指数衰减，导致梯度消失。

---

## 23.4 LSTM长短期记忆网络

### 23.4.1 为什么需要LSTM？

**RNN的致命缺陷**

考虑这个句子：

> "我出生在中国，长大在法国，但我的母语是______。"

正确的答案是"中文"。但要回答这个问题，网络需要记住句子**开头**的"中国"，而这个词距离空白处有十几个词。

**RNN做不到这一点**，因为梯度在反向传播过程中会消失。

**LSTM的解决方案**

LSTM引入了一条**信息高速公路**——细胞状态 $C_t$，它：
- 可以几乎不变地传递信息
- 通过门控机制选择性地添加或删除信息
- 让梯度能够长距离流动而不消失

### 23.4.2 LSTM的核心组件

**1. 遗忘门（Forget Gate）**

决定从细胞状态中"遗忘"什么信息：

$$
\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)
$$

输出是0到1之间的值：
- 0：完全遗忘
- 1：完全保留

**2. 输入门（Input Gate）**

决定什么新信息存入细胞状态：

$$
\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)
$$

候选细胞状态：

$$
\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_C \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C)
$$

**3. 细胞状态更新**

$$
\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t
$$

**这是最关键的一步！**

**4. 输出门（Output Gate）**

决定输出什么：

$$
\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)
$$

$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)
$$

### 23.4.3 完整的LSTM方程组

总结所有六个方程：

```
遗忘门:    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
输入门:    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
候选状态:  C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
细胞状态:  C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
输出门:    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
隐藏状态:  h_t = o_t ⊙ tanh(C_t)
```

### 23.4.4 LSTM的直观理解

**比喻：智能图书馆** 📚

想象细胞状态 $C_t$ 是一座图书馆：

- **遗忘门** $f_t$：图书管理员检查哪些旧书需要下架
- **输入门** $i_t$：决定哪些新书值得上架
- **候选状态** $\tilde{C}_t$：新书的候选列表
- **细胞状态更新**：旧书保留 + 新书上架
- **输出门** $o_t$：读者询问时，决定展示哪些书的内容
- **隐藏状态** $h_t$：读者实际看到的信息

**为什么能解决梯度消失？** 🎯

关键在细胞状态更新方程：

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

如果遗忘门 $f_t \approx 1$（保留大部分旧信息），那么：

$$
\frac{\partial C_t}{\partial C_{t-1}} \approx 1
$$

梯度可以几乎无损地反向传播！这就是为什么LSTM能记住长距离依赖。

### 23.4.5 LSTM的变体

**窥孔连接（Peephole Connections）**

让门控"窥视"细胞状态：

$$
\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{C}_{t-1}, \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)
$$

**耦合输入-遗忘门**

让输入门和遗忘门联动：

$$
\mathbf{f}_t = 1 - \mathbf{i}_t
$$

（当添加新信息时，相应遗忘旧信息）

---

## 23.5 GRU门控循环单元

### 23.5.1 GRU的简化设计

GRU合并了LSTM的细胞状态和隐藏状态，简化为两个门：

**1. 更新门（Update Gate）**

$$
\mathbf{z}_t = \sigma(\mathbf{W}_z \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t])
$$

控制保留多少旧隐藏状态。

**2. 重置门（Reset Gate）**

$$
\mathbf{r}_t = \sigma(\mathbf{W}_r \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t])
$$

控制计算候选状态时忽略多少旧信息。

**3. 候选隐藏状态**

$$
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} \cdot [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])
$$

**4. 隐藏状态更新**

$$
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
$$

### 23.5.2 与LSTM的对比

| 特性 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3个 | 2个 |
| 状态数量 | 2个 (C_t, h_t) | 1个 (h_t) |
| 参数量 | 更多 | 更少 (~75%) |
| 计算速度 | 较慢 | 较快 |
| 性能 | 相当 | 相当 |

**为什么GRU效果不差？** 🤔

研究表明：
- **门控机制**是长程记忆的关键
- 具体实现（3门 vs 2门）影响较小
- GRU在较小数据集上可能更好（正则化效应）

**选择建议**
- 数据量大 → 两者都可以
- 需要极致性能 → 都尝试，交叉验证
- 资源受限 → GRU
- 需要可解释性 → LSTM（分离的细胞状态）

---

## 23.6 双向RNN

### 23.6.1 双向处理的动机

考虑这个句子：

> "苹果______好吃"（苹果好吃 / 不好吃）

仅看前文"苹果"，无法确定空白处填什么。但如果能看到后文"好吃"，就能确定应该填"很"。

**双向RNN**同时考虑过去和未来信息！

### 23.6.2 双向RNN的结构

```
前向层: x_t → [→RNN→] → h⃗_t
后向层: x_t → [←RNN←] → h⃖_t

合并: h_t = concat([h⃗_t, h⃖_t]) 或 h⃗_t + h⃖_t
```

**前向层**：从左到右处理序列
**后向层**：从右到左处理序列

### 23.6.3 应用场景

双向RNN特别适用于：
- **命名实体识别**："苹果"是公司还是水果？
- **语音识别**：需要后文来确定同音词
- **机器翻译**：完整理解源句子再翻译

**注意**：双向RNN不适合实时任务（如实时语音转文字），因为需要看到完整序列才能输出。

---

## 23.7 代码实现

### 23.7.1 RNNCell类实现

```python
"""
循环神经网络 - 纯NumPy实现
第二十三章代码实现
"""

import numpy as np
from typing import Tuple, Optional, List


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid激活函数"""
    # 数值稳定性处理
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Sigmoid导数"""
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh激活函数"""
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Tanh导数"""
    return 1 - np.tanh(x) ** 2


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax函数"""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def clip_gradients(grad: np.ndarray, max_norm: float = 5.0) -> np.ndarray:
    """梯度裁剪，防止梯度爆炸"""
    norm = np.sqrt(np.sum(grad ** 2))
    if norm > max_norm:
        grad = grad * (max_norm / norm)
    return grad


class RNNCell:
    """
    基础RNN单元（Elman网络）
    
    数学公式:
        h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
        y_t = W_hy @ h_t + b_y
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        参数:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重 - Xavier初始化
        scale_hh = np.sqrt(1.0 / hidden_size)
        scale_xh = np.sqrt(1.0 / input_size)
        scale_hy = np.sqrt(1.0 / hidden_size)
        
        # 隐藏层权重: W_hh (hidden_size, hidden_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_hh
        # 输入权重: W_xh (hidden_size, input_size)
        self.W_xh = np.random.randn(hidden_size, input_size) * scale_xh
        # 隐藏层偏置
        self.b_h = np.zeros(hidden_size)
        
        # 输出权重: W_hy (output_size, hidden_size)
        self.W_hy = np.random.randn(output_size, hidden_size) * scale_hy
        # 输出偏置
        self.b_y = np.zeros(output_size)
        
        # 存储中间结果用于反向传播
        self.cache = {}
        
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向传播一步
        
        参数:
            x: 当前输入 (input_size,)
            h_prev: 前一时刻隐藏状态 (hidden_size,)
            
        返回:
            h: 当前隐藏状态 (hidden_size,)
            y: 当前输出 (output_size,)
        """
        # 计算新的隐藏状态
        # h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
        z = self.W_hh @ h_prev + self.W_xh @ x + self.b_h
        h = tanh(z)
        
        # 计算输出
        # y_t = W_hy @ h_t + b_y
        y = self.W_hy @ h + self.b_y
        
        # 缓存用于反向传播
        self.cache = {
            'x': x.copy(),
            'h_prev': h_prev.copy(),
            'z': z.copy(),
            'h': h.copy(),
            'y': y.copy()
        }
        
        return h, y
    
    def backward(self, dy: np.ndarray, dh_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        反向传播一步
        
        参数:
            dy: 输出梯度 (output_size,)
            dh_next: 来自下一时刻的隐藏状态梯度 (hidden_size,)
            
        返回:
            dx: 输入梯度 (input_size,)
            dh_prev: 前一时刻隐藏状态梯度 (hidden_size,)
            grads: 参数字典
        """
        x = self.cache['x']
        h_prev = self.cache['h_prev']
        z = self.cache['z']
        h = self.cache['h']
        
        # 输出层梯度
        dW_hy = np.outer(dy, h)  # (output_size, hidden_size)
        db_y = dy
        dh_from_y = self.W_hy.T @ dy  # (hidden_size,)
        
        # 隐藏状态总梯度
        dh = dh_from_y + dh_next
        
        # Tanh导数
        dtanh = dh * tanh_derivative(z)
        
        # 参数梯度
        dW_hh = np.outer(dtanh, h_prev)
        dW_xh = np.outer(dtanh, x)
        db_h = dtanh
        
        # 传递梯度
        dh_prev = self.W_hh.T @ dtanh
        dx = self.W_xh.T @ dtanh
        
        grads = {
            'W_hh': dW_hh,
            'W_xh': dW_xh,
            'b_h': db_h,
            'W_hy': dW_hy,
            'b_y': db_y
        }
        
        return dx, dh_prev, grads
    
    def get_params(self) -> dict:
        """获取所有参数"""
        return {
            'W_hh': self.W_hh,
            'W_xh': self.W_xh,
            'b_h': self.b_h,
            'W_hy': self.W_hy,
            'b_y': self.b_y
        }
    
    def set_params(self, params: dict):
        """设置参数"""
        self.W_hh = params['W_hh'].copy()
        self.W_xh = params['W_xh'].copy()
        self.b_h = params['b_h'].copy()
        self.W_hy = params['W_hy'].copy()
        self.b_y = params['b_y'].copy()
```

### 23.7.2 LSTMCell类实现

```python
class LSTMCell:
    """
    LSTM长短期记忆单元
    
    数学公式:
        f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)
        i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)
        C̃_t = tanh(W_C @ [h_{t-1}, x_t] + b_C)
        C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
        o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)
        h_t = o_t ⊙ tanh(C_t)
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 拼接后的维度
        concat_size = input_size + hidden_size
        
        # Xavier初始化
        scale = np.sqrt(1.0 / concat_size)
        
        # 遗忘门参数
        self.W_f = np.random.randn(hidden_size, concat_size) * scale
        self.b_f = np.zeros(hidden_size)
        
        # 输入门参数
        self.W_i = np.random.randn(hidden_size, concat_size) * scale
        self.b_i = np.zeros(hidden_size)
        
        # 候选细胞状态参数
        self.W_C = np.random.randn(hidden_size, concat_size) * scale
        self.b_C = np.zeros(hidden_size)
        
        # 输出门参数
        self.W_o = np.random.randn(hidden_size, concat_size) * scale
        self.b_o = np.zeros(hidden_size)
        
        # 输出层参数
        scale_out = np.sqrt(1.0 / hidden_size)
        self.W_hy = np.random.randn(output_size, hidden_size) * scale_out
        self.b_y = np.zeros(output_size)
        
        self.cache = {}
        
    def forward(self, x: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        LSTM前向传播一步
        
        参数:
            x: 当前输入 (input_size,)
            h_prev: 前一时刻隐藏状态 (hidden_size,)
            C_prev: 前一时刻细胞状态 (hidden_size,)
            
        返回:
            h: 当前隐藏状态 (hidden_size,)
            C: 当前细胞状态 (hidden_size,)
            y: 当前输出 (output_size,)
        """
        # 拼接输入和前一隐藏状态
        concat = np.concatenate([h_prev, x])  # (hidden_size + input_size,)
        
        # 遗忘门: f_t = σ(W_f @ concat + b_f)
        f = sigmoid(self.W_f @ concat + self.b_f)
        
        # 输入门: i_t = σ(W_i @ concat + b_i)
        i = sigmoid(self.W_i @ concat + self.b_i)
        
        # 候选细胞状态: C̃_t = tanh(W_C @ concat + b_C)
        C_tilde = tanh(self.W_C @ concat + self.b_C)
        
        # 细胞状态更新: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
        C = f * C_prev + i * C_tilde
        
        # 输出门: o_t = σ(W_o @ concat + b_o)
        o = sigmoid(self.W_o @ concat + self.b_o)
        
        # 隐藏状态: h_t = o_t ⊙ tanh(C_t)
        h = o * tanh(C)
        
        # 输出
        y = self.W_hy @ h + self.b_y
        
        # 缓存
        self.cache = {
            'x': x.copy(),
            'h_prev': h_prev.copy(),
            'C_prev': C_prev.copy(),
            'concat': concat.copy(),
            'f': f.copy(),
            'i': i.copy(),
            'C_tilde': C_tilde.copy(),
            'C': C.copy(),
            'o': o.copy(),
            'h': h.copy(),
            'y': y.copy()
        }
        
        return h, C, y
    
    def backward(self, dy: np.ndarray, dh_next: np.ndarray, dC_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        LSTM反向传播一步
        
        参数:
            dy: 输出梯度 (output_size,)
            dh_next: 来自下一时刻的隐藏状态梯度 (hidden_size,)
            dC_next: 来自下一时刻的细胞状态梯度 (hidden_size,)
            
        返回:
            dx: 输入梯度 (input_size,)
            dh_prev: 前一时刻隐藏状态梯度 (hidden_size,)
            dC_prev: 前一时刻细胞状态梯度 (hidden_size,)
            grads: 参数字典
        """
        # 读取缓存
        x = self.cache['x']
        h_prev = self.cache['h_prev']
        C_prev = self.cache['C_prev']
        concat = self.cache['concat']
        f = self.cache['f']
        i = self.cache['i']
        C_tilde = self.cache['C_tilde']
        C = self.cache['C']
        o = self.cache['o']
        h = self.cache['h']
        
        # 输出层梯度
        dW_hy = np.outer(dy, h)
        db_y = dy
        dh = self.W_hy.T @ dy + dh_next
        
        # 输出门梯度
        do = dh * tanh(C)
        dzo = do * sigmoid_derivative(self.W_o @ concat + self.b_o)
        dW_o = np.outer(dzo, concat)
        db_o = dzo
        
        # 细胞状态梯度
        dC = dh * o * tanh_derivative(C) + dC_next
        
        # 输入门梯度
        di = dC * C_tilde
        dzi = di * sigmoid_derivative(self.W_i @ concat + self.b_i)
        dW_i = np.outer(dzi, concat)
        db_i = dzi
        
        # 候选细胞状态梯度
        dC_tilde = dC * i
        dzC = dC_tilde * tanh_derivative(self.W_C @ concat + self.b_C)
        dW_C = np.outer(dzC, concat)
        db_C = dzC
        
        # 遗忘门梯度
        df = dC * C_prev
        dzf = df * sigmoid_derivative(self.W_f @ concat + self.b_f)
        dW_f = np.outer(dzf, concat)
        db_f = dzf
        
        # 传递到前一时刻
        d_concat = (self.W_f.T @ dzf + 
                   self.W_i.T @ dzi + 
                   self.W_C.T @ dzC + 
                   self.W_o.T @ dzo)
        
        dh_prev = d_concat[:self.hidden_size]
        dx = d_concat[self.hidden_size:]
        dC_prev = dC * f
        
        grads = {
            'W_f': dW_f, 'b_f': db_f,
            'W_i': dW_i, 'b_i': db_i,
            'W_C': dW_C, 'b_C': db_C,
            'W_o': dW_o, 'b_o': db_o,
            'W_hy': dW_hy, 'b_y': db_y
        }
        
        return dx, dh_prev, dC_prev, grads
    
    def get_params(self) -> dict:
        """获取参数"""
        return {
            'W_f': self.W_f, 'b_f': self.b_f,
            'W_i': self.W_i, 'b_i': self.b_i,
            'W_C': self.W_C, 'b_C': self.b_C,
            'W_o': self.W_o, 'b_o': self.b_o,
            'W_hy': self.W_hy, 'b_y': self.b_y
        }
    
    def set_params(self, params: dict):
        """设置参数"""
        for key in params:
            setattr(self, key, params[key].copy())
```

### 23.7.3 GRUCell类实现

```python
class GRUCell:
    """
    GRU门控循环单元
    
    数学公式:
        z_t = σ(W_z @ [h_{t-1}, x_t] + b_z)
        r_t = σ(W_r @ [h_{t-1}, x_t] + b_r)
        h̃_t = tanh(W_h @ [r_t ⊙ h_{t-1}, x_t] + b_h)
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        concat_size = input_size + hidden_size
        scale = np.sqrt(1.0 / concat_size)
        
        # 更新门
        self.W_z = np.random.randn(hidden_size, concat_size) * scale
        self.b_z = np.zeros(hidden_size)
        
        # 重置门
        self.W_r = np.random.randn(hidden_size, concat_size) * scale
        self.b_r = np.zeros(hidden_size)
        
        # 候选隐藏状态
        self.W_h = np.random.randn(hidden_size, concat_size) * scale
        self.b_h = np.zeros(hidden_size)
        
        # 输出层
        scale_out = np.sqrt(1.0 / hidden_size)
        self.W_hy = np.random.randn(output_size, hidden_size) * scale_out
        self.b_y = np.zeros(output_size)
        
        self.cache = {}
        
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GRU前向传播一步"""
        concat = np.concatenate([h_prev, x])
        
        # 更新门
        z = sigmoid(self.W_z @ concat + self.b_z)
        
        # 重置门
        r = sigmoid(self.W_r @ concat + self.b_r)
        
        # 候选隐藏状态 (使用重置后的h_prev)
        concat_reset = np.concatenate([r * h_prev, x])
        h_tilde = tanh(self.W_h @ concat_reset + self.b_h)
        
        # 隐藏状态更新
        h = (1 - z) * h_prev + z * h_tilde
        
        # 输出
        y = self.W_hy @ h + self.b_y
        
        self.cache = {
            'x': x.copy(),
            'h_prev': h_prev.copy(),
            'concat': concat.copy(),
            'z': z.copy(),
            'r': r.copy(),
            'concat_reset': concat_reset.copy(),
            'h_tilde': h_tilde.copy(),
            'h': h.copy(),
            'y': y.copy()
        }
        
        return h, y
    
    def backward(self, dy: np.ndarray, dh_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """GRU反向传播一步"""
        x = self.cache['x']
        h_prev = self.cache['h_prev']
        concat = self.cache['concat']
        z = self.cache['z']
        r = self.cache['r']
        concat_reset = self.cache['concat_reset']
        h_tilde = self.cache['h_tilde']
        h = self.cache['h']
        
        # 输出层
        dW_hy = np.outer(dy, h)
        db_y = dy
        dh = self.W_hy.T @ dy + dh_next
        
        # 隐藏状态梯度
        dz = dh * (h_tilde - h_prev)
        dh_prev = dh * (1 - z)
        dh_tilde = dh * z
        
        # 候选隐藏状态梯度
        dzh = dh_tilde * tanh_derivative(self.W_h @ concat_reset + self.b_h)
        dW_h = np.outer(dzh, concat_reset)
        db_h = dzh
        
        # 重置门梯度
        d_concat_reset = self.W_h.T @ dzh
        dr = d_concat_reset[:self.hidden_size] * h_prev
        
        # 更新门梯度
        dzz = dz * sigmoid_derivative(self.W_z @ concat + self.b_z)
        dW_z = np.outer(dzz, concat)
        db_z = dzz
        
        # 重置门梯度（续）
        dzr = dr * sigmoid_derivative(self.W_r @ concat + self.b_r)
        dW_r = np.outer(dzr, concat)
        db_r = dzr
        
        # 传递到前一时刻
        d_concat = self.W_z.T @ dzz + self.W_r.T @ dzr
        d_concat[:self.hidden_size] += d_concat_reset[:self.hidden_size] * r
        dh_prev += d_concat[:self.hidden_size]
        dx = d_concat[self.hidden_size:]
        
        grads = {
            'W_z': dW_z, 'b_z': db_z,
            'W_r': dW_r, 'b_r': db_r,
            'W_h': dW_h, 'b_h': db_h,
            'W_hy': dW_hy, 'b_y': db_y
        }
        
        return dx, dh_prev, grads
    
    def get_params(self) -> dict:
        return {
            'W_z': self.W_z, 'b_z': self.b_z,
            'W_r': self.W_r, 'b_r': self.b_r,
            'W_h': self.W_h, 'b_h': self.b_h,
            'W_hy': self.W_hy, 'b_y': self.b_y
        }
    
    def set_params(self, params: dict):
        for key in params:
            setattr(self, key, params[key].copy())
```

### 23.7.4 SimpleRNN模型实现

```python
class SimpleRNN:
    """
    简单循环神经网络模型（支持多层）
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 num_layers: int = 1, cell_type: str = 'rnn'):
        """
        参数:
            input_size: 输入维度
            hidden_size: 每层隐藏层维度
            output_size: 输出维度
            num_layers: 层数
            cell_type: 'rnn', 'lstm', 或 'gru'
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        # 创建多层单元
        self.layers = []
        for i in range(num_layers):
            layer_input = input_size if i == 0 else hidden_size
            
            if cell_type == 'rnn':
                cell = RNNCell(layer_input, hidden_size, output_size if i == num_layers - 1 else hidden_size)
            elif cell_type == 'lstm':
                cell = LSTMCell(layer_input, hidden_size, output_size if i == num_layers - 1 else hidden_size)
            elif cell_type == 'gru':
                cell = GRUCell(layer_input, hidden_size, output_size if i == num_layers - 1 else hidden_size)
            else:
                raise ValueError(f"Unknown cell type: {cell_type}")
            
            self.layers.append(cell)
        
        # 如果不是最后一层单独输出，添加最终输出层
        if num_layers > 1 and cell_type == 'lstm':
            self.output_layer = None  # LSTM最后一层直接输出
        elif num_layers > 1:
            # 为RNN/GRU添加输出层
            scale = np.sqrt(1.0 / hidden_size)
            self.W_out = np.random.randn(output_size, hidden_size) * scale
            self.b_out = np.zeros(output_size)
        
        self.history = {'loss': []}
    
    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        前向传播整个序列
        
        参数:
            X: 输入序列 (seq_len, input_size)
            
        返回:
            H: 所有时刻的隐藏状态列表
            Y: 所有时刻的输出列表
        """
        seq_len = X.shape[0]
        
        # 初始化隐藏状态
        if self.cell_type == 'lstm':
            states = [(np.zeros(self.hidden_size), np.zeros(self.hidden_size)) 
                     for _ in range(self.num_layers)]
        else:
            states = [np.zeros(self.hidden_size) for _ in range(self.num_layers)]
        
        H = [[] for _ in range(self.num_layers)]  # 每层的隐藏状态
        Y = []
        
        for t in range(seq_len):
            x = X[t]
            
            # 逐层传播
            for layer_idx, cell in enumerate(self.layers):
                if self.cell_type == 'lstm':
                    h_prev, c_prev = states[layer_idx]
                    h, c, y = cell.forward(x, h_prev, c_prev)
                    states[layer_idx] = (h, c)
                    H[layer_idx].append(h)
                    x = h  # 下一层的输入
                else:
                    h_prev = states[layer_idx]
                    h, y = cell.forward(x, h_prev)
                    states[layer_idx] = h
                    H[layer_idx].append(h)
                    x = h
            
            # 最终输出
            if self.num_layers == 1:
                Y.append(y)
            else:
                # 多层时使用最后一层的隐藏状态
                final_h = states[-1][0] if self.cell_type == 'lstm' else states[-1]
                if hasattr(self, 'W_out'):
                    y = self.W_out @ final_h + self.b_out
                Y.append(y)
        
        return H, Y
    
    def backward(self, X: np.ndarray, dY: List[np.ndarray]) -> List[dict]:
        """
        BPTT反向传播
        
        参数:
            X: 输入序列
            dY: 每个时刻的输出梯度
            
        返回:
            每层的梯度列表
        """
        seq_len = len(dY)
        
        # 初始化梯度缓存
        all_grads = [{} for _ in range(self.num_layers)]
        
        # 初始化隐藏状态梯度
        if self.cell_type == 'lstm':
            dh_next = [np.zeros(self.hidden_size) for _ in range(self.num_layers)]
            dC_next = [np.zeros(self.hidden_size) for _ in range(self.num_layers)]
        else:
            dh_next = [np.zeros(self.hidden_size) for _ in range(self.num_layers)]
        
        # 时间反向传播
        for t in reversed(range(seq_len)):
            dy = dY[t]
            
            # 从最后一层开始
            for layer_idx in reversed(range(self.num_layers)):
                cell = self.layers[layer_idx]
                
                if self.cell_type == 'lstm':
                    dx, dh_prev, dC_prev, grads = cell.backward(dy, dh_next[layer_idx], dC_next[layer_idx])
                    dh_next[layer_idx] = dh_prev
                    dC_next[layer_idx] = dC_prev
                else:
                    dx, dh_prev, grads = cell.backward(dy, dh_next[layer_idx])
                    dh_next[layer_idx] = dh_prev
                
                # 累加梯度
                for key in grads:
                    if key not in all_grads[layer_idx]:
                        all_grads[layer_idx][key] = grads[key]
                    else:
                        all_grads[layer_idx][key] += grads[key]
                
                # 梯度传递到下一层（前一时间步的上层）
                dy = dx
        
        return all_grads
    
    def train_step(self, X: np.ndarray, targets: np.ndarray, lr: float = 0.01) -> float:
        """
        单步训练
        
        参数:
            X: 输入序列 (seq_len, input_size)
            targets: 目标输出 (seq_len, output_size)
            lr: 学习率
            
        返回:
            loss: 损失值
        """
        # 前向传播
        H, Y = self.forward(X)
        
        # 计算损失和梯度
        loss = 0
        dY = []
        
        for t in range(len(Y)):
            # Softmax交叉熵
            y_pred = softmax(Y[t])
            loss += -np.sum(targets[t] * np.log(y_pred + 1e-8))
            
            # 输出梯度
            dy = y_pred - targets[t]
            dY.append(dy)
        
        loss /= len(Y)
        
        # 反向传播
        all_grads = self.backward(X, dY)
        
        # 更新参数
        for layer_idx, cell in enumerate(self.layers):
            params = cell.get_params()
            grads = all_grads[layer_idx]
            
            for key in params:
                # 梯度裁剪
                grad_clipped = clip_gradients(grads[key])
                params[key] -= lr * grad_clipped
            
            cell.set_params(params)
        
        return loss
```

### 23.7.5 字符级语言模型

```python
class CharLanguageModel:
    """
    字符级语言模型
    """
    
    def __init__(self, vocab_size: int, hidden_size: int = 128, cell_type: str = 'lstm'):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        
        # 创建RNN
        self.rnn = SimpleRNN(
            input_size=vocab_size,
            hidden_size=hidden_size,
            output_size=vocab_size,
            num_layers=2,
            cell_type=cell_type
        )
        
        # 字符映射
        self.char_to_idx = {}
        self.idx_to_char = {}
    
    def _one_hot(self, idx: int) -> np.ndarray:
        """One-hot编码"""
        vec = np.zeros(self.vocab_size)
        vec[idx] = 1.0
        return vec
    
    def prepare_data(self, text: str):
        """准备字符映射"""
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        if len(chars) != self.vocab_size:
            print(f"Warning: vocab_size mismatch. Using {len(chars)} instead of {self.vocab_size}")
            self.vocab_size = len(chars)
    
    def train(self, text: str, epochs: int = 100, seq_length: int = 25, lr: float = 0.01):
        """
        训练语言模型
        
        参数:
            text: 训练文本
            epochs: 训练轮数
            seq_length: 序列长度
            lr: 学习率
        """
        self.prepare_data(text)
        
        data_size = len(text)
        losses = []
        
        print(f"开始训练字符级语言模型...")
        print(f"数据大小: {data_size} 字符")
        print(f"词汇表大小: {self.vocab_size}")
        print(f"序列长度: {seq_length}")
        print(f"训练轮数: {epochs}")
        print("-" * 50)
        
        for epoch in range(epochs):
            # 随机选择起始位置
            start_idx = np.random.randint(0, data_size - seq_length - 1)
            
            # 准备输入和目标
            X_seq = []
            target_seq = []
            
            for i in range(seq_length):
                char = text[start_idx + i]
                next_char = text[start_idx + i + 1]
                
                X_seq.append(self._one_hot(self.char_to_idx[char]))
                target_seq.append(self._one_hot(self.char_to_idx[next_char]))
            
            X = np.array(X_seq)
            targets = np.array(target_seq)
            
            # 训练一步
            loss = self.rnn.train_step(X, targets, lr)
            losses.append(loss)
            
            if (epoch + 1) % 10 == 0:
                avg_loss = np.mean(losses[-10:])
                perplexity = np.exp(avg_loss)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
            
            # 每20轮生成一段文本
            if (epoch + 1) % 20 == 0:
                generated = self.generate(seed=text[start_idx:start_idx+10], length=50)
                print(f"生成文本: {generated}")
                print("-" * 50)
        
        self.rnn.history['loss'] = losses
        return losses
    
    def generate(self, seed: str, length: int = 100, temperature: float = 1.0) -> str:
        """
        生成文本
        
        参数:
            seed: 种子文本
            length: 生成长度
            temperature: 温度（控制随机性）
            
        返回:
            generated: 生成的文本
        """
        generated = seed
        
        # 初始化状态
        if self.cell_type == 'lstm':
            states = [(np.zeros(self.hidden_size), np.zeros(self.hidden_size)) 
                     for _ in range(self.rnn.num_layers)]
        else:
            states = [np.zeros(self.hidden_size) for _ in range(self.rnn.num_layers)]
        
        # 用种子初始化状态
        for char in seed[:-1]:
            if char not in self.char_to_idx:
                continue
            x = self._one_hot(self.char_to_idx[char])
            
            for layer_idx, cell in enumerate(self.rnn.layers):
                if self.cell_type == 'lstm':
                    h, c, _ = cell.forward(x, states[layer_idx][0], states[layer_idx][1])
                    states[layer_idx] = (h, c)
                else:
                    h, _ = cell.forward(x, states[layer_idx])
                    states[layer_idx] = h
                x = h
        
        # 当前字符
        current_char = seed[-1]
        
        # 生成
        for _ in range(length):
            if current_char not in self.char_to_idx:
                current_char = np.random.choice(list(self.char_to_idx.keys()))
            
            x = self._one_hot(self.char_to_idx[current_char])
            
            # 前向传播
            for layer_idx, cell in enumerate(self.rnn.layers):
                if self.cell_type == 'lstm':
                    h, c, y = cell.forward(x, states[layer_idx][0], states[layer_idx][1])
                    states[layer_idx] = (h, c)
                else:
                    h, y = cell.forward(x, states[layer_idx])
                    states[layer_idx] = h
                x = h
            
            # 应用温度
            y = y / temperature
            probs = softmax(y)
            
            # 采样
            idx = np.random.choice(self.vocab_size, p=probs)
            current_char = self.idx_to_char[idx]
            generated += current_char
        
        return generated


# 训练示例
def train_char_lm_example():
    """字符级语言模型训练示例"""
    
    # 示例文本（可以替换为任何文本）
    text = """
    机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习。
    深度学习是机器学习的一个子集，使用多层神经网络来学习数据的表示。
    神经网络受到生物神经系统的启发，由相互连接的节点组成。
    训练神经网络需要大量数据和计算资源。
    机器学习算法可以分为监督学习、无监督学习和强化学习。
    监督学习使用标记数据来训练模型。
    无监督学习在没有标签的情况下发现数据中的模式。
    强化学习通过与环境的交互来学习最优策略。
    """ * 10  # 重复以增加数据量
    
    # 创建并训练模型
    vocab_size = len(set(text))
    model = CharLanguageModel(
        vocab_size=vocab_size,
        hidden_size=64,
        cell_type='lstm'
    )
    
    losses = model.train(text, epochs=200, seq_length=30, lr=0.05)
    
    # 生成文本
    print("\n最终生成:")
    for temp in [0.5, 1.0, 1.5]:
        print(f"\n温度 = {temp}:")
        generated = model.generate("机器学习", length=100, temperature=temp)
        print(generated)
    
    return model, losses
```

### 23.7.6 时间序列预测

```python
class TimeSeriesPredictor:
    """
    时间序列预测器
    """
    
    def __init__(self, input_size: int = 1, hidden_size: int = 32, output_size: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 使用GRU（适合时间序列）
        self.rnn = SimpleRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=1,
            cell_type='gru'
        )
        
        # 归一化参数
        self.mean = 0
        self.std = 1
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """标准化数据"""
        self.mean = np.mean(data)
        self.std = np.std(data) + 1e-8
        return (data - self.mean) / self.std
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """反标准化"""
        return data * self.std + self.mean
    
    def create_sequences(self, data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建序列数据
        
        参数:
            data: 时间序列数据
            seq_length: 序列长度
            
        返回:
            X: 输入序列 (num_samples, seq_length, input_size)
            y: 目标值 (num_samples, output_size)
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        
        return np.array(X), np.array(y)
    
    def train(self, data: np.ndarray, seq_length: int = 10, epochs: int = 100, 
              lr: float = 0.01, batch_size: int = 1):
        """
        训练时间序列预测器
        
        参数:
            data: 时间序列数据
            seq_length: 序列长度
            epochs: 训练轮数
            lr: 学习率
            batch_size: 批量大小
        """
        # 归一化
        data_norm = self.normalize(data)
        
        # 创建序列
        X, y = self.create_sequences(data_norm, seq_length)
        num_samples = len(X)
        
        print(f"开始训练时间序列预测器...")
        print(f"数据点: {len(data)}, 序列长度: {seq_length}")
        print(f"样本数: {num_samples}, 训练轮数: {epochs}")
        print("-" * 50)
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # 随机打乱
            indices = np.random.permutation(num_samples)
            
            for idx in indices:
                X_seq = X[idx].reshape(-1, self.input_size)
                target = y[idx].reshape(1, self.output_size)
                
                # 前向传播
                H, Y = self.rnn.forward(X_seq)
                
                # 计算MSE损失
                pred = Y[-1]
                loss = np.mean((pred - target.flatten()) ** 2)
                epoch_loss += loss
                
                # 计算梯度
                dY = [np.zeros(self.output_size) for _ in range(len(Y))]
                dY[-1] = 2 * (pred - target.flatten())
                
                # 反向传播
                all_grads = self.rnn.backward(X_seq, dY)
                
                # 更新参数
                for layer_idx, cell in enumerate(self.rnn.layers):
                    params = cell.get_params()
                    grads = all_grads[layer_idx]
                    
                    for key in params:
                        grad_clipped = clip_gradients(grads[key])
                        params[key] -= lr * grad_clipped
                    
                    cell.set_params(params)
            
            avg_loss = epoch_loss / num_samples
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        self.rnn.history['loss'] = losses
        return losses
    
    def predict(self, sequence: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        预测未来值
        
        参数:
            sequence: 输入序列
            steps: 预测步数
            
        返回:
            predictions: 预测值
        """
        # 归一化
        seq_norm = (sequence - self.mean) / self.std
        
        predictions = []
        current_seq = seq_norm.copy()
        
        for _ in range(steps):
            # 前向传播
            X = current_seq[-10:].reshape(-1, self.input_size) if len(current_seq) >= 10 else current_seq.reshape(-1, self.input_size)
            H, Y = self.rnn.forward(X)
            
            # 预测下一步
            pred_norm = Y[-1]
            predictions.append(pred_norm)
            
            # 更新序列
            current_seq = np.append(current_seq, pred_norm)
        
        # 反归一化
        predictions = np.array(predictions)
        return self.denormalize(predictions)


# 生成合成时间序列数据
def generate_synthetic_series(n_points: int = 500) -> np.ndarray:
    """生成合成时间序列（带趋势和季节性）"""
    t = np.arange(n_points)
    
    # 趋势
    trend = 0.01 * t
    
    # 季节性
    seasonal = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.sin(2 * np.pi * t / 20)
    
    # 噪声
    noise = np.random.randn(n_points) * 2
    
    return trend + seasonal + noise


def train_timeseries_example():
    """时间序列预测示例"""
    np.random.seed(42)
    
    # 生成数据
    data = generate_synthetic_series(n_points=400)
    
    # 划分训练/测试
    train_size = 300
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # 创建模型
    model = TimeSeriesPredictor(input_size=1, hidden_size=16, output_size=1)
    
    # 训练
    losses = model.train(train_data, seq_length=20, epochs=50, lr=0.01)
    
    # 预测
    print("\n预测测试:")
    test_seq = train_data[-20:]
    predictions = model.predict(test_seq, steps=len(test_data))
    
    # 计算测试误差
    mse = np.mean((predictions - test_data) ** 2)
    print(f"测试MSE: {mse:.4f}")
    
    return model, losses, predictions, test_data
```

### 23.7.7 梯度检查验证

```python
def numerical_gradient(cell, x: np.ndarray, h_prev: np.ndarray, 
                       target: np.ndarray, eps: float = 1e-5) -> dict:
    """
    数值梯度计算（用于梯度检查）
    """
    params = cell.get_params()
    num_grads = {}
    
    for key in params:
        param = params[key]
        grad = np.zeros_like(param)
        
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]
            
            # f(x + eps)
            param[idx] = old_val + eps
            if isinstance(cell, LSTMCell):
                cell.set_params(params)
                _, _, y_plus = cell.forward(x, h_prev, np.zeros_like(h_prev))
            else:
                cell.set_params(params)
                _, y_plus = cell.forward(x, h_prev)
            loss_plus = np.sum((y_plus - target) ** 2)
            
            # f(x - eps)
            param[idx] = old_val - eps
            if isinstance(cell, LSTMCell):
                cell.set_params(params)
                _, _, y_minus = cell.forward(x, h_prev, np.zeros_like(h_prev))
            else:
                cell.set_params(params)
                _, y_minus = cell.forward(x, h_prev)
            loss_minus = np.sum((y_minus - target) ** 2)
            
            # 恢复
            param[idx] = old_val
            
            # 中心差分
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            
            it.iternext()
        
        num_grads[key] = grad
    
    # 恢复原始参数
    cell.set_params(params)
    
    return num_grads


def gradient_check(cell_type: str = 'rnn', input_size: int = 5, 
                   hidden_size: int = 4, output_size: int = 3):
    """
    梯度检查
    
    比较解析梯度和数值梯度的差异
    """
    np.random.seed(42)
    
    # 创建单元
    if cell_type == 'rnn':
        cell = RNNCell(input_size, hidden_size, output_size)
    elif cell_type == 'lstm':
        cell = LSTMCell(input_size, hidden_size, output_size)
    elif cell_type == 'gru':
        cell = GRUCell(input_size, hidden_size, output_size)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")
    
    # 随机输入
    x = np.random.randn(input_size)
    h_prev = np.random.randn(hidden_size)
    target = np.random.randn(output_size)
    
    if cell_type == 'lstm':
        C_prev = np.random.randn(hidden_size)
        h, C, y = cell.forward(x, h_prev, C_prev)
        dy = 2 * (y - target)
        dx, dh_prev, dC_prev, ana_grads = cell.backward(dy, np.zeros_like(h_prev), np.zeros_like(C_prev))
    else:
        h, y = cell.forward(x, h_prev)
        dy = 2 * (y - target)
        dx, dh_prev, ana_grads = cell.backward(dy, np.zeros_like(h_prev))
    
    # 数值梯度
    if cell_type == 'lstm':
        num_grads = {}
        # 简化的数值梯度检查
        print(f"LSTM梯度检查简化版（计算量大，仅检查输出层）")
        params = cell.get_params()
        for key in ['W_hy', 'b_y']:
            param = params[key]
            grad = np.zeros_like(param)
            eps = 1e-5
            
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                old_val = param[idx]
                
                param[idx] = old_val + eps
                cell.set_params(params)
                _, _, y_plus = cell.forward(x, h_prev, C_prev)
                loss_plus = np.sum((y_plus - target) ** 2)
                
                param[idx] = old_val - eps
                cell.set_params(params)
                _, _, y_minus = cell.forward(x, h_prev, C_prev)
                loss_minus = np.sum((y_minus - target) ** 2)
                
                param[idx] = old_val
                grad[idx] = (loss_plus - loss_minus) / (2 * eps)
                it.iternext()
            
            num_grads[key] = grad
            ana = ana_grads[key]
            diff = np.abs(ana - grad) / (np.abs(ana) + np.abs(grad) + 1e-8)
            max_diff = np.max(diff)
            print(f"  {key}: 最大相对误差 = {max_diff:.8f} {'✓' if max_diff < 1e-4 else '✗'}")
    else:
        num_grads = numerical_gradient(cell, x, h_prev, target)
        
        print(f"\n{cell_type.upper()}梯度检查:")
        print("-" * 50)
        
        max_diffs = []
        for key in ana_grads:
            ana = ana_grads[key]
            num = num_grads[key]
            
            # 相对误差
            diff = np.abs(ana - num) / (np.abs(ana) + np.abs(num) + 1e-8)
            max_diff = np.max(diff)
            max_diffs.append(max_diff)
            
            status = "✓ PASS" if max_diff < 1e-4 else "✗ FAIL"
            print(f"{key:10s}: 最大相对误差 = {max_diff:.8f} {status}")
        
        overall = "✓ 所有梯度检查通过" if all(d < 1e-4 for d in max_diffs) else "✗ 部分梯度检查失败"
        print("-" * 50)
        print(overall)


# 运行梯度检查
def run_gradient_checks():
    """运行所有梯度检查"""
    print("=" * 60)
    print("梯度检查验证")
    print("=" * 60)
    
    print("\n1. RNN单元梯度检查")
    gradient_check('rnn', input_size=5, hidden_size=4, output_size=3)
    
    print("\n2. LSTM单元梯度检查")
    gradient_check('lstm', input_size=5, hidden_size=4, output_size=3)
    
    print("\n3. GRU单元梯度检查")
    gradient_check('gru', input_size=5, hidden_size=4, output_size=3)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 运行梯度检查
    run_gradient_checks()
    
    # 运行训练示例
    print("\n" + "=" * 60)
    print("字符级语言模型示例")
    print("=" * 60)
    # lm_model, lm_losses = train_char_lm_example()
    
    print("\n" + "=" * 60)
    print("时间序列预测示例")
    print("=" * 60)
    # ts_model, ts_losses, predictions, test_data = train_timeseries_example()
```

---

## 23.8 可视化：RNN与LSTM结构

### 23.8.1 RNN展开的ASCII图示

```
RNN按时间展开 (Unrolling through time):

时间:    t=0          t=1          t=2          t=3         ...     t=T
        ┌───┐        ┌───┐        ┌───┐        ┌───┐              ┌───┐
x_0 ──→ │   │   x_1 →│   │   x_2 →│   │   x_3 →│   │    ...  x_T →│   │
        │ R │        │ R │        │ R │        │ R │              │ R │
h_-1 ─→ │ N │   h_0 →│ N │   h_1 →│ N │   h_2 →│ N │    ...  h_T-1→│ N │
 (零)   │ N │   ┌──→ │ N │   ┌──→ │ N │   ┌──→ │ N │         ┌──→ │ N │
        │   │   │    │   │   │    │   │   │    │   │         │    │   │
        └───┘   │    └───┘   │    └───┘   │    └───┘         │    └───┘
          ↓     │      ↓     │      ↓     │      ↓           │      ↓
         y_0    │     y_1    │     y_2    │     y_3          │     y_T
                 └───────────┘            └───────────┘      └─────────────┘
                  权重共享 W                  权重共享 W            权重共享 W

                  
压缩视角（单步循环）:

              ┌───────────┐
              │           │
              ↓    h_t    │
    x_t  ─→ ┌─────┐      │
            │ RNN │ ─────┘
    h_{t-1} →│     │
            └──┬──┘
               ↓
              y_t
              
参数:
  W_xh: 输入→隐藏层权重
  W_hh: 隐藏层→隐藏层权重 (循环连接)
  W_hy: 隐藏层→输出权重
```

### 23.8.2 LSTM内部结构的ASCII图示

```
LSTM单元详细结构:

输入: x_t (当前输入)
      h_{t-1} (前一隐藏状态)
      C_{t-1} (前一细胞状态)

                    ┌─────────────────────────────────────┐
                    │           LSTM CELL                 │
                    │                                     │
  ┌─────────────┐   │   ┌─────┐     ┌─────────────┐      │
  │   遗忘门    │   │   │ σ   │     │  细胞状态   │      │
  │  ┌───────┐  │   │   └─┬───┘     │   ┌─────┐   │      │
h_{t-1} →│ concat│  │   │   │ f_t      │   │  ×  │←─┼─ C_{t-1}
    ↓  │   ↑   │  │   │   ↓          │   └──┬──┘   │      │
  ┌──┴──┴───┴──┐│   │ ┌───────┐      │      │       │      │
  │   x_t      ││   │ │   ×   │←─────┼──────┘       │      │
  └────────────┘│   │ └───┬───┘      │              │      │
        ↓       │   │     │          │   ┌─────┐    │      │
    ┌───────┐   │   │     ↓          └──→│  +  │←───┼── i_t ⊙ C̃_t
    │ W_f   │   │   │ ┌─────────┐        └──┬──┘    │      │
    │ b_f   │   │   │ │f_t ⊙    │           ↓       │      │
    └───┬───┘   │   │ │C_{t-1}  │         C_t       │      │
        ↓       │   │ └─────────┘           │       │      │
       σ ───────┼───┼───────────────────────┤       │      │
        ↓       │   │                       │       │      │
       f_t      │   │   ┌─────┐             ↓       │      │
                │   │   │ tanh│         ┌───────┐   │      │
                │   │   └──┬──┘         │  tanh │   │      │
  ┌─────────────┤   │      │            └───┬───┘   │      │
  │   输入门    │   │   ┌──┴──┐             │       │      │
  │  ┌───────┐  │   │   │  ×  │←────────────┤       │      │
  │  │ concat│←─┼───┼───┤     │             ↓       │      │
  │  │   ↑   │  │   │   └─────┘          ┌───────┐  │      │
  │  └──┴───┴──┘│   │       ↑            │   ×   │←─┼─── o_t
  │      ↓      │   │   ┌───────┐        │       │  │      │
  │   ┌───────┐ │   │   │i_t ⊙  │        └───┬───┘  │      │
  │   │ W_i   │ │   │   │C̃_t    │            │      │      │
  │   │ b_i   │ │   │   └───────┘            ↓      │      │
  │   └───┬───┘ │   │                        │      │      │
  │       ↓     │   └────────────────────────┘      │      │
  │      σ ─────┼───────────────────────────────────┤      │
  │       ↓     │                                   │      │
  │      i_t    │                                   │      │
  └─────────────┘                                   │      │
                                                    │      │
  ┌─────────────┐                                   │      │
  │  候选状态   │         ┌─────┐                   │      │
  │  ┌───────┐  │         │ σ   │                   │      │
  │  │ concat│←─┼────────→│     │←─────────────────┼──────┤
  │  │   ↑   │  │         └──┬──┘                  │      │
  │  └──┴───┴──┘            ↓                      │      │
  │      ↓                 o_t                     │      │
  │   ┌───────┐                                    │      │
  │   │ W_C   │                                    │      │
  │   │ b_C   │                                    │      │
  │   └───┬───┘                                    │      │
  │       ↓                                        │      │
  │      tanh ─────────────────────────────────────┘      │
  │       ↓                                               │
  │      C̃_t                                              │
  └─────────────┘                                         │
                                                          │
输出: h_t = o_t ⊙ tanh(C_t)                              │
      y_t = W_hy @ h_t + b_y                              │
                                                          │
      C_t (传递到下一时刻) ─────────────────────────────────┘
      h_t (传递到下一时刻)


简化记忆图:

        遗忘门 ────┐
                   × ────┐
        C_{t-1} ───┘     
                         + ─── C_t ───→ tanh ───┐
        输入门 ────┐     ↑                      × ─── h_t
                   × ────┘              输出门 ───┘
        C̃_t ─────┘

关键: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
     (加法更新使梯度可以流动!)
```

---

## 23.9 应用案例

### 23.9.1 机器翻译

**任务**: 将一种语言的句子翻译成另一种语言

**经典架构**: 编码器-解码器（Encoder-Decoder）

```
源语言: "I love machine learning"
         ↓
    [编码器RNN] → 上下文向量
                       ↓
    [解码器RNN] → "我 爱 机器 学习"
目标语言
```

**突破**: Google Neural Machine Translation (GNMT, 2016)
- 使用8层LSTM编码器和8层LSTM解码器
- 翻译质量接近人类水平

### 23.9.2 语音识别

**任务**: 将语音信号转换为文字

**处理流程**:
```
音频波形 → 特征提取(MFCC) → 声学模型(RNN) → 语言模型 → 文字
```

**里程碑**:
- Apple Siri使用RNN进行语音识别
- Google Voice Search采用LSTM声学模型
- 错误率从20%+降到5%以下

### 23.9.3 文本生成

**应用**:
- 智能写作助手（自动补全）
- 诗歌/小说生成
- 代码自动生成

**技术**:
- 字符级语言模型（如本章实现）
- 词级语言模型
- 大语言模型（GPT系列的前身）

### 23.9.4 其他应用

| 领域 | 应用 | RNN类型 |
|------|------|---------|
| 金融 | 股价预测 | LSTM/GRU |
| 医疗 | 心电图分析 | 双向LSTM |
| 音乐 | 旋律生成 | LSTM |
| 生物信息 | DNA序列分析 | 双向RNN |
| 视频 | 动作识别 | LSTM |

---

## 23.10 练习题

### 基础题

**习题 23.1** 🌟
> RNN与普通神经网络的主要区别是什么？为什么说RNN有"记忆"？

<details>
<summary>参考答案</summary>

主要区别：
1. **循环连接**: RNN的隐藏层输出会反馈到自身，形成循环
2. **参数共享**: 所有时间步共享同一组权重
3. **处理变长序列**: 可以处理不同长度的输入

"记忆"的含义：隐藏状态 $h_t$ 编码了从序列开始到当前时刻的所有信息，因此网络"记住"了之前看到的内容。
</details>

**习题 23.2** 🌟
> 解释梯度消失问题。为什么它会影响RNN学习长距离依赖？

<details>
<summary>参考答案</summary>

梯度消失发生在反向传播时，梯度需要经过多个时间步传递：
$$
\frac{\partial L}{\partial W} \propto \prod_{k=1}^{t} \frac{\partial h_k}{\partial h_{k-1}}
$$

由于多次连乘，如果雅可比矩阵的特征值小于1，梯度会指数级衰减。这意味着远距离时间步的梯度几乎为0，网络无法学习长期依赖。
</details>

**习题 23.3** 🌟
> LSTM的哪三个门分别起什么作用？细胞状态 $C_t$ 和隐藏状态 $h_t$ 有什么区别？

<details>
<summary>参考答案</summary>

三个门：
- **遗忘门** $f_t$: 决定从细胞状态中丢弃什么信息
- **输入门** $i_t$: 决定添加什么新信息到细胞状态
- **输出门** $o_t$: 决定从细胞状态中输出什么

区别：
- $C_t$ 是**细胞状态**（长期记忆），通过加法门更新，梯度可长距离流动
- $h_t$ 是**隐藏状态**（工作记忆），是 $C_t$ 经过输出门过滤后的输出
</details>

### 进阶题

**习题 23.4** 🌟🌟
> 推导LSTM细胞状态的梯度，证明当遗忘门 $f_t \approx 1$ 时，梯度可以长距离流动。

<details>
<summary>提示</summary>

从 $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ 出发，计算 $\frac{\partial C_t}{\partial C_{t-1}}$。当 $f_t \approx 1$ 时，这个偏导数接近1，意味着梯度不会衰减。
</details>

**习题 23.5** 🌟🌟
> 比较LSTM和GRU的异同。在什么情况下你会选择GRU而不是LSTM？

<details>
<summary>参考答案</summary>

相同点：
- 都使用门控机制解决梯度消失
- 都能学习长距离依赖

不同点：
- LSTM有3个门+2个状态；GRU有2个门+1个状态
- GRU参数量更少，计算更快
- GRU结构更简单

选择GRU的情况：
- 计算资源受限
- 数据集较小（GRU的正则化效果更好）
- 需要更快的训练速度
</details>

**习题 23.6** 🌟🌟
> 双向RNN在哪些任务中有优势？为什么不适合实时语音识别？

<details>
<summary>参考答案</summary>

优势任务：
- 命名实体识别（需要后文语境）
- 机器翻译（需要完整理解源句子）
- 情感分析（整体语境决定情感）

不适合实时语音识别：
- 双向RNN需要看到完整的输入序列才能开始输出
- 实时任务要求边输入边输出
- 延迟问题
</details>

### 挑战题

**习题 23.7** 🌟🌟🌟
> 修改本章的SimpleRNN类，实现一个多层双向LSTM。提示：需要分别实现前向和后向层，然后将它们的输出拼接。

<details>
<summary>实现框架</summary>

```python
class BidirectionalLSTM:
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        # 前向LSTM
        self.forward_lstm = SimpleRNN(..., cell_type='lstm')
        # 后向LSTM
        self.backward_lstm = SimpleRNN(..., cell_type='lstm')
        # 合并输出的全连接层
        self.W_merge = ...  # (output_size, 2*hidden_size)
    
    def forward(self, X):
        # X: (seq_len, input_size)
        # 1. 前向传播
        _, Y_f = self.forward_lstm.forward(X)
        # 2. 后向传播 (X翻转)
        _, Y_b = self.backward_lstm.forward(X[::-1])
        # 3. 合并输出
        Y_merged = [self.W_merge @ np.concatenate([y_f, y_b]) 
                   for y_f, y_b in zip(Y_f, Y_b[::-1])]
        return Y_merged
```
</details>

**习题 23.8** 🌟🌟🌟
> 注意力机制（Attention）是RNN的重要扩展。研究并解释：为什么注意力机制能进一步提升序列建模能力？简述"Seq2Seq + Attention"的工作原理。

<details>
<summary>参考答案</summary>

注意力机制的优势：
1. **解决信息瓶颈**: 编码器-解码器将所有信息压缩到一个固定向量，注意力允许解码器动态关注源序列的不同部分
2. **更好的长距离依赖**: 直接连接到源序列的所有位置
3. **可解释性**: 注意力权重显示模型关注的位置

Seq2Seq + Attention工作原理：
1. 编码器处理源序列，得到所有时刻的隐藏状态
2. 解码器每一步计算注意力权重：$\alpha_{ij} = \text{align}(s_{i-1}, h_j)$
3. 加权求和得到上下文向量：$c_i = \sum_j \alpha_{ij} h_j$
4. 解码器结合上下文向量和上一输出预测当前词
</details>

---

## 23.11 参考文献

1. Jordan, M. I. (1986). Serial order: A parallel distributed processing approach (Technical Report No. 8604). Institute for Cognitive Science, University of California, San Diego.

2. Elman, J. L. (1990). Finding structure in time. *Cognitive Science*, 14(2), 179-211.

3. Hochreiter, S. (1991). Untersuchungen zu dynamischen neuronalen Netzen [Studies on dynamic neural networks]. Diploma thesis, Technical University of Munich, Germany.

4. Werbos, P. J. (1990). Backpropagation through time: What it does and how to do it. *Proceedings of the IEEE*, 78(10), 1550-1560.

5. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

6. Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget: Continual prediction with LSTM. *Neural Computation*, 12(10), 2451-2471.

7. Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. *arXiv preprint arXiv:1412.3555*.

8. Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11), 2673-2681.

9. Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *arXiv preprint arXiv:1406.1078*.

10. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. *Advances in Neural Information Processing Systems*, 27, 3104-3112.

---

## 本章小结

在本章中，我们深入学习了循环神经网络：

**历史演进** 🕰️
- 1986: Jordan网络引入循环连接
- 1990: Elman网络奠定现代RNN基础
- 1991: Hochreiter发现梯度消失问题
- 1997: LSTM通过门控机制解决问题
- 2014: GRU简化结构同时保持性能

**核心概念** 🧠
- **隐藏状态**: RNN的记忆载体
- **BPTT**: 随时间反向传播训练算法
- **门控机制**: LSTM/GRU的核心创新
- **细胞状态**: 信息高速公路

**技术实现** 💻
- 从0手写RNN、LSTM、GRU单元
- 完整实现BPTT反向传播
- 字符级语言模型
- 时间序列预测器

**应用场景** 🎯
- 机器翻译、语音识别、文本生成
- 股价预测、音乐生成、生物信息

**下一步** ➡️
在下一章中，我们将学习**注意力机制（Attention）**，这是Transformer架构的基础，也是现代大语言模型的核心技术！

---

*本章字数统计: 约13,200字*
*代码行数统计: 约920行*
