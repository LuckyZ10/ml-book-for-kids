# 第二十四章：注意力机制与Transformer——全局的观察者

## 章节引言

想象你正在阅读一本精彩的小说。

当你读到"他举起剑，向敌人刺去"这句话时，你需要知道：
- **"他"** 指的是谁？是男主角还是反派？
- **"剑"** 从何而来？前文提到的神器吗？
- **"敌人"** 是谁？为什么成为敌人？

要理解这句话，你需要**回头看**——回顾前文的语境，建立词语之间的关联。

**RNN** 就像一个逐字阅读的人，它从左到右慢慢"读"，把前面的信息压缩成一个小小的"记忆"传递给下一个时刻。但当文本很长时，这个"记忆"会变得模糊，远距离的词语之间很难建立联系。

**Transformer** 则不同。它像一个**拥有全局视野的观察者**，能够一眼看完整篇文章，直接计算出任意两个词之间的关联程度。

```
RNN: 我→喜→欢→机→器→学→习  (逐个处理，传递记忆)
                              ↓
Transformer: 我喜欢机器学习  (同时看到所有词，计算每对词的关联)
```

这就是**注意力机制（Attention）**的魔力——它让模型学会"看哪里"。

在本章中，我们将：
- 🎭 用生活化的比喻理解Attention（读书时的"划重点"能力）
- 📜 探索从2015年Bahdanau Attention到2017年Transformer的革命
- 🧮 完整推导Self-Attention和Multi-Head Attention的数学原理
- 🔓 深入理解位置编码的三角函数设计
- 💻 用纯NumPy和PyTorch从零实现完整的Transformer
- 🎯 训练一个中英翻译模型和文本生成器

准备好了吗？让我们开启这场深度学习史上最重要的旅程！

---

## 24.1 为什么RNN不够用了？

### 24.1.1 RNN的困境

让我们回顾一下RNN的问题：

**困境一：顺序处理，无法并行**

```
时间步:  t1    t2    t3    t4    ...    t100
         ↓     ↓     ↓     ↓            ↓
RNN:   h1 → h2 → h3 → h4 → ... → h100
         ↓     ↓     ↓     ↓            ↓
         y1    y2    y3    y4         y100

问题：必须等h1算完才能算h2，无法并行！
```

假设你有100个词的句子，RNN需要**依次计算100步**。如果用GPU并行计算，本来可以同时算100个，现在只能一个个来，就像一条车道的高速公路 vs 100条车道——效率差距巨大！

**困境二：长距离依赖问题**

```
句子: "虽然小明不喜欢数学，但是经过努力，他在期末考试中取得了____的好成绩。"
        ↑                                            ↑
      位置1                                         位置50

RNN需要把"数学"的信息传递49步才能到达"好成绩"处！
```

信息在RNN的传递过程中会逐渐衰减，就像传话游戏，传的人越多，信息失真越严重。

**困境三：计算复杂度随序列长度线性增长**

对于长度为$n$的序列：
- RNN的计算量是 $O(n)$，但无法并行
- 如果需要捕捉任意两个位置的关系，需要 $O(n^2)$ 的复杂度

### 24.1.2 人类阅读vs机器阅读

**人类是如何阅读的？**

当你读到上面那个句子时：
1. 你会**先看整句话**，了解大概意思
2. 填空时，你会**回头找相关词**
3. 你会注意到"虽然...但是..."的转折关系
4. 你会把"数学"和空格处的词建立**直接联系**

**关键在于：人类不是逐字读的，而是带着问题去寻找答案！**

这就像考试时做阅读理解：
- 先看问题（**Query**）
- 再带着问题去文章中找相关信息（**Key**）
- 找到后提取答案（**Value**）

这就是**注意力机制**的核心思想！

### 24.1.3 Attention的诞生

2015年，Dzmitry Bahdanau等人在论文《Neural Machine Translation by Jointly Learning to Align and Translate》中提出了**注意力机制**。

核心想法很简单：

```
传统的Encoder-Decoder:
Encoder: 源句子 → [压缩向量] 
                          ↓
Decoder: [压缩向量] → 目标句子

问题：所有信息都压缩到一个向量，长句子会丢失信息！

带Attention的Encoder-Decoder:
Encoder: 源句子 → [向量1, 向量2, ..., 向量n] 
                          ↓
Decoder: 生成每个词时，选择性地"看"相关的源词
```

**类比**：传统的翻译就像让你背下整篇文章再翻译，而Attention允许你边看原文边翻译！

---

## 24.2 经典文献研究：从Attention到Transformer

### 24.2.1 Bahdanau Attention（2015）：注意力机制的诞生

**文献信息**
> Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. In *Proceedings of ICLR 2015*.

**背景故事**

这篇论文发表于2015年，当时神经机器翻译（NMT）正面临一个核心问题：**信息瓶颈**。Encoder必须把整个源句子压缩成一个固定长度的向量，无论句子长短都是同样大小的向量——这显然不合理。

**核心创新**

Bahdanau提出了**加性注意力（Additive Attention）**：

```
对于Decoder的每一步t:
  1. 计算当前状态s_t与所有Encoder隐状态h_i的相关性
     e_ti = v_a^T tanh(W_s s_t + W_h h_i)
  
  2. 归一化得到注意力权重
     α_ti = softmax(e_ti) = exp(e_ti) / Σ_j exp(e_tj)
  
  3. 计算上下文向量（加权和）
     c_t = Σ_i α_ti * h_i
  
  4. 用c_t帮助生成下一个词
```

**可视化理解**

```
源句子: 我  喜欢  机器  学习
         ↓   ↓    ↓    ↓
隐状态: h1  h2   h3   h4
         ↘  ↓   ↙ ↘  ↓  ↙
          ↘ ↓  ↙   ↘↓ ↙
            s_t (Decoder当前状态)
              ↓
        e_t1 e_t2 e_t3 e_t4  (相关性分数)
         ↓    ↓    ↓    ↓
        0.1  0.2  0.4  0.3  (注意力权重)
         ↓    ↓    ↓    ↓
        ↘└───┴────┴───┴──→ c_t (上下文向量)
```

**费曼比喻** 🎯

想象你在做一道阅读理解题：
- **Decoder状态s_t** = 当前的问题
- **Encoder隐状态h_i** = 文章中的每个句子
- **注意力权重α_ti** = 你对每个句子的关注程度
- **上下文向量c_t** = 你综合相关句子后得到的答案

这就是Bahdanau Attention：**根据当前需要，有选择地关注源句子中的不同部分**。

---

### 24.2.2 Luong Attention（2015）：更高效的点积注意力

**文献信息**
> Luong, M. T., Pham, H., & Manning, C. D. (2015). Effective approaches to attention-based neural machine translation. In *Proceedings of EMNLP 2015*.

**核心创新**

Thang Luong等人提出了**乘性注意力（Multiplicative/Dot-Product Attention）**：

```
score(s_t, h_i) = s_t^T h_i  (点积)

或者使用可学习的矩阵:
score(s_t, h_i) = s_t^T W h_i
```

相比于Bahdanau的加法注意力：
- **Bahdanau**: additive = v^T tanh(W_q q + W_k k) —— 更灵活
- **Luong**: multiplicative = q^T W k —— 更快更简单

**注意力类型**

Luong提出了两种注意力类型：

1. **全局注意力（Global）**：关注所有源词
   ```
   适合：短句子
   复杂度：O(源长度 × 目标长度)
   ```

2. **局部注意力（Local）**：只关注窗口内的源词
   ```
   适合：长句子
   复杂度：O(窗口大小 × 目标长度)
   ```

**历史意义**

Luong Attention证明了**点积可以高效地计算相关性**，这为后来Transformer中Scaled Dot-Product Attention奠定了基础。

---

### 24.2.3 Self-Attention（2017）：革命性的突破

**核心思想**

之前的Attention都是**Encoder-Decoder Attention**：查询来自Decoder，键和值来自Encoder。

**Self-Attention**提出了一个惊人的想法：**让序列中的每个词都去关注序列中的其他词！**

```
句子:  猫  坐在  垫子  上
        ↓  ↓   ↓   ↓
      [Self-Attention计算每对词的关联]
        ↓  ↓   ↓   ↓
输出:  猫' 坐在' 垫子' 上'
      (每个词都融合了其他词的信息)
```

**为什么这是革命性的？**

1. **完全并行**：所有位置的Self-Attention可以同时计算
2. **长距离依赖**：任意两个词的距离都是$O(1)$
3. **可解释性强**：注意力权重直接显示词与词的关系

```
RNN: 距离为n的两个词需要O(n)步传播
Self-Attention: 任意两个词直接计算关联，O(1)
```

---

### 24.2.4 Transformer（2017）：Attention Is All You Need

**文献信息**
> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems* (pp. 5998-6008).

**背景故事**

2017年，Google Brain团队的这篇论文彻底改变了深度学习的格局。标题本身就是一个宣言："**Attention Is All You Need**"（你只需要注意力）。

他们证明了：**完全不需要RNN或CNN，只用Attention就能达到SOTA效果**。

**架构图**

```
┌─────────────────────────────────────────────────────────┐
│                    Transformer 架构                     │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐   ┌─────────────────────┐      │
│  │     ENCODER (×N)    │   │     DECODER (×N)    │      │
│  │                     │   │                     │      │
│  │  Input Embedding    │   │  Output Embedding   │      │
│  │       +             │   │       +             │      │
│  │  Positional Encoding│   │  Positional Encoding│      │
│  │         ↓           │   │         ↓           │      │
│  │  ┌───────────────┐  │   │  ┌───────────────┐  │      │
│  │  │ Multi-Head    │  │   │  │ Masked Multi  │  │      │
│  │  │ Self-Attention│  │   │  │ -Head Self    │  │      │
│  │  │               │  │   │  │ -Attention    │  │      │
│  │  │ Add & Norm   │  │   │  │ Add & Norm   │  │      │
│  │  └───────────────┘  │   │  └───────────────┘  │      │
│  │         ↓           │   │         ↓           │      │
│  │  ┌───────────────┐  │   │  ┌───────────────┐  │      │
│  │  │ Feed Forward  │  │   │  │ Multi-Head    │  │      │
│  │  │   (FFN)       │  │   │  │ Cross-Attention│ │      │
│  │  │               │  │   │  │ Add & Norm   │  │      │
│  │  │ Add & Norm   │  │   │  └───────────────┘  │      │
│  │  └───────────────┘  │   │         ↓           │      │
│  │         ↓           │   │  ┌───────────────┐  │      │
│  └─────────────────────┘   │  │ Feed Forward  │  │      │
│              ↓             │  │ Add & Norm   │  │      │
│         [输出]             │  └───────────────┘  │      │
│                              └─────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

**核心创新**

1. **Scaled Dot-Product Attention**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

2. **Multi-Head Attention**: 并行的多组注意力

3. **Positional Encoding**: 用正弦/余弦函数编码位置信息

4. **彻底摒弃RNN**: 完全并行计算

**历史影响**

- WMT 2014英德翻译任务：**28.4 BLEU** (比之前SOTA高2+ BLEU)
- WMT 2014英法翻译任务：**41.8 BLEU**
- 训练时间大幅减少（并行化）
- 催生了BERT、GPT、T5等一系列里程碑模型

---

### 24.2.5 BERT（2019）：双向预训练的霸主

**文献信息**
> Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019* (pp. 4171-4186).

**核心创新**

BERT提出了两个关键思想：

**1. 双向上下文理解**

```
GPT (自回归): 我 喜欢 [MASK] → 预测"学习" (只看左边)
BERT (双向): [MASK] 喜欢 机器 [MASK] → 预测两个空 (看两边)
```

**2. 预训练任务**

- **MLM (Masked Language Model)**: 随机mask 15%的词，预测原词
- **NSP (Next Sentence Prediction)**: 判断两个句子是否连续

**架构对比**

```
BERT-base:  12层, 768维, 12头,  110M参数
BERT-large: 24层, 1024维, 16头, 340M参数

相比GPT:
- GPT是"从左到右"的生成模型
- BERT是"双向理解"的编码器模型
```

**影响**

BERT在11个NLP任务上取得SOTA，开启了"预训练+微调"的时代。

---

### 24.2.6 GPT系列（2018-2020）：生成式预训练的崛起

**GPT-1 (2018)**
> Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training.

**GPT-2 (2019)**
> Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI Blog*.

**GPT-3 (2020)** ⭐
> Brown, T. B., Mann, B., Ryder, N., et al. (2020). Language models are few-shot learners. In *Advances in Neural Information Processing Systems* (pp. 1877-1901).

**GPT-3的核心突破**

| 特性 | GPT-3 |
|------|-------|
| 参数规模 | **1750亿** (是当时最大非稀疏模型) |
| 层数 | 96层 |
| 维度 | 12288维 |
| 注意力头 | 96头 |

**Few-Shot Learning**

GPT-3展示了惊人的"上下文学习"能力：

```
Zero-shot: 直接问"Translate to French: Hello"
One-shot: 给一个例子后问
Few-shot: 给几个例子后问
```

不需要梯度更新，仅通过文本交互就能完成各种任务！

**历史意义**

GPT-3证明了：
1. 模型规模的力量——越大越强
2. 生成式模型也能做理解任务
3. "涌现能力"——大模型会产生小模型没有的能力

---

### 24.2.7 T5（2020）：统一的Text-to-Text框架

**文献信息**
> Raffel, C., Shazeer, N., Roberts, A., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140), 1-67.

**核心思想**

T5的核心创新是：**把所有NLP任务都转化为文本到文本的格式**。

```
翻译:       "translate English to German: Hello" → "Hallo"
分类:       "cola sentence: The cat sat." → "acceptable"
问答:       "question: Who wrote Hamlet? context: ..." → "Shakespeare"
摘要:       "summarize: [长文章]" → "[摘要]"
```

**架构特点**

- 使用完整的Encoder-Decoder架构
- 相对位置编码（Relative Positional Encoding）
- 预训练任务：span corruption（span填空）

**影响**

T5统一了NLP任务的范式，后续的T5.1.1、mT5、UL2等都是基于这个框架。

---

### 24.2.8 位置编码的演进

**1. 正弦位置编码（Sinusoidal, Vaswani 2017）**

原始Transformer使用：
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

优点：
- 不需要学习参数
- 可以泛化到训练时未见过的长度

**2. 可学习位置编码（Learned, BERT/GPT）**

把位置编码当作可学习的参数：
```python
position_embeddings = nn.Embedding(max_position, d_model)
```

**3. 旋转位置编码（RoPE, Su 2021）**

将位置信息编码到query和key的旋转中：
$$f(q, m) = qe^{im\theta}$$

被LLaMA、PaLM等模型采用。

**4. ALiBi（Press 2022）**

直接给注意力分数加上基于距离的惩罚：
$$\text{softmax}(q^Tk - m|i-j|)$$

简单有效，支持长度外推。

---

## 24.3 Self-Attention深度解析

### 24.3.1 核心思想：Query, Key, Value

Self-Attention的核心是三个概念，类比**信息检索系统**：

```
┌──────────────────────────────────────────────┐
│              信息检索类比                     │
├──────────────────────────────────────────────┤
│                                              │
│  你想搜索: "深度学习入门书籍"                 │
│         ↓                                    │
│   Query（查询）← 你提出的需求                │
│         ↓                                    │
│   [搜索引擎匹配]                              │
│         ↓                                    │
│   Key（索引）  ← 每本书的标签/关键词         │
│         ↓                                    │
│   匹配分数 = Query · Key（点积）              │
│         ↓                                    │
│   Value（值）  ← 书的内容                    │
│         ↓                                    │
│   返回：按匹配分数加权的书籍列表              │
│                                              │
└──────────────────────────────────────────────┘
```

**在Self-Attention中**：

对于输入句子中的**每个词**，我们都生成：
- **Query (Q)**: 我想查询什么信息？
- **Key (K)**: 我有什么信息？
- **Value (V)**: 我的实际内容是什么？

### 24.3.2 费曼比喻：图书馆找书

想象你是一位研究员，需要写一篇关于"机器学习"的报告。你走进一座巨大的图书馆：

**场景一：传统RNN**

```
你从第一排书架开始，一本一本读
每读完一本，在笔记本上记摘要
读下一本时参考笔记本
...
问题：读了很多书后，笔记本内容太杂乱
      第一本书的内容早就被稀释了
```

**场景二：Self-Attention**

```
你写下你想了解的问题（Query）:
"机器学习的核心概念和应用"

图书馆里每本书都有一个标签（Key）:
- 书A: "深度学习导论"
- 书B: "Python编程"
- 书C: "统计学习方法"
- 书D: "机器学习实践"

你快速比较问题和标签的匹配度:
- 书A: 90% 相关
- 书B: 10% 相关
- 书C: 85% 相关
- 书D: 95% 相关

然后你同时阅读这四本书，但注意力分配不同:
- 花95%精力读书D
- 花90%精力读书A
- 花85%精力读书C
- 花10%精力读书B

这就是加权平均：输出 = 0.95×V_D + 0.90×V_A + 0.85×V_C + 0.10×V_B
```

### 24.3.3 数学推导

**输入**：
- 输入矩阵 $X \in \mathbb{R}^{n \times d_{model}}$，其中$n$是序列长度

**生成Q, K, V**：
$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

其中 $W^Q, W^K \in \mathbb{R}^{d_{model} \times d_k}$，$W^V \in \mathbb{R}^{d_{model} \times d_v}$

**计算注意力分数**：
$$\text{scores} = QK^T \in \mathbb{R}^{n \times n}$$

矩阵$(i,j)$位置表示第$i$个词对第$j$个词的注意力分数。

**缩放（Scaling）**：
$$\text{scaled\_scores} = \frac{QK^T}{\sqrt{d_k}}$$

**为什么要除以 $\sqrt{d_k}$？**

当$d_k$很大时，点积的数值会很大，导致softmax进入梯度很小的区域：

```
softmax([10, 20, 30]) ≈ [0.000, 0.000, 1.000]  (梯度几乎为0)
softmax([1, 2, 3])    ≈ [0.090, 0.245, 0.665]  (梯度正常)
```

除以$\sqrt{d_k}$把数值缩放到合适的范围，避免梯度消失。

**应用Softmax**：
$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

$A$是注意力权重矩阵，每行和为1。

**加权求和**：
$$\text{Output} = AV = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**完整流程可视化**

```
输入 X:          计算 Q, K, V:       Attention分数:     输出:
┌───┐           ┌───┐                ┌─────────┐       ┌───┐
│我 │ ────────→ │Q1 │                │ 0.5 0.3 │       │O1 │
├───┤           ├───┤                │ 0.2 0.4 │       ├───┤
│喜 │ ────────→ │Q2 │                │ ...     │       │O2 │
├───┤           ├───┤         QK^T   └─────────┘       ├───┤
│欢 │ ────────→ │Q3 │        ─────→  softmax ──→  ×V  │O3 │
├───┤           ├───┤                                ├───┤
│学 │ ────────→ │Q4 │                                │O4 │
└───┘           └───┘                                └───┘

           ┌───┬───┬───┬───┐
           │K1 │K2 │K3 │K4 │
           └───┴───┴───┴───┘
           ┌───┬───┬───┬───┐
           │V1 │V2 │V3 │V4 │
           └───┴───┴───┴───┘
```

---

## 24.4 Multi-Head Attention：多头注意力

### 24.4.1 为什么需要多头？

**类比：多个观察角度**

想象你在分析一部电影：
- 一个观众关注**剧情**
- 一个观众关注**演技**
- 一个观众关注**摄影**
- 一个观众关注**配乐**

每个观众从不同的角度理解电影，最后把各自的见解综合起来，就得到了对电影的全面理解。

**在语言中**：

对于"The animal didn't cross the street because it was too tired"：

- **头1**关注**句法**：it 指的是 animal（语法角色相同）
- **头2**关注**语义**：tired 和 animal 相关（语义关联）
- **头3**关注**位置**：cross 和 street 是相邻的（位置接近）

### 24.4.2 数学公式

**单头注意力**：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**多头拼接**：
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

**维度设计**

假设 $d_{model} = 512$，使用$h=8$个头：
- 每个头的维度：$d_k = d_v = 512 / 8 = 64$
- 每个头的计算：$Q_i, K_i, V_i \in \mathbb{R}^{n \times 64}$

**总参数量**：
- $W_i^Q, W_i^K, W_i^V$：$8 \times (512 \times 64 \times 3) = 786,432$
- $W^O$：$512 \times 512 = 262,144$
- 总计：约1M参数

### 24.4.3 并行计算

多头注意力的美妙之处在于：**所有头可以并行计算**。

```python
# 原始实现：h个独立的注意力
heads = [attention(Q @ W_q[i], K @ W_k[i], V @ W_v[i]) 
         for i in range(h)]

# 优化实现：通过reshape一次性计算
Q_reshaped = Q.reshape(batch, n, h, d_k).transpose(0, 2, 1, 3)
# shape: (batch, h, n, d_k)

# 并行计算所有头的注意力分数
scores = Q_reshaped @ K_reshaped.transpose(0, 1, 3, 2) / sqrt(d_k)
# shape: (batch, h, n, n)

# 并行应用softmax，并行加权
output = softmax(scores) @ V_reshaped
```

在GPU上，这可以高效地并行执行。

---

## 24.5 位置编码的奥秘

### 24.5.1 为什么需要位置编码？

Self-Attention有一个致命缺陷：**它是位置无关的（permutation-invariant）**。

```python
"我喜欢机器学习" → Self-Attention → 某个输出
"机器学习喜欢我" → Self-Attention → 同样的输出！(只是顺序不同)
```

因为Self-Attention计算的是所有词对之间的关系，不考虑词的位置。

**解决方案**：给每个词加上位置信息！

### 24.5.2 正弦位置编码的数学

原始Transformer使用的位置编码：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**为什么这样设计？**

**1. 周期性**

正弦/余弦函数是周期性的，可以泛化到任意长度。

**2. 相对位置关系**

对于固定的偏移量$k$，$PE_{pos+k}$可以表示为$PE_{pos}$的线性函数：

$$\sin(\omega \cdot (pos + k)) = \sin(\omega \cdot pos)\cos(\omega \cdot k) + \cos(\omega \cdot pos)\sin(\omega \cdot k)$$

这意味着模型可以轻松学习**相对位置**关系。

**3. 不同频率的波长**

- 维度小（i小）：波长$2\pi \cdot 10000^{2i/d_{model}}$小，变化快 → 捕捉**精细**位置
- 维度大（i大）：波长长，变化慢 → 捕捉**粗粒度**位置

**可视化**

```
位置编码矩阵 (pos × dimension):

      dim0    dim2    dim4    ...   dim510
     (高频)   ↓      ↓             (低频)
       │
pos 0  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░
pos 1  ░░████░░░░░░░░░░░░░░░░░░░░░░░░░░░░
pos 2  ░░░░████░░░░░░░░░░░░░░░░░░░░░░░░░░
pos 3  ░░░░░░████░░░░░░░░░░░░░░░░░░░░░░░░
       │
       ↓
     高频维度随位置快速变化
     低频维度随位置缓慢变化
```

### 24.5.3 其他位置编码方法

**可学习位置编码**
```python
position_embedding = nn.Embedding(max_position, d_model)
```

- 更灵活，但无法泛化到训练时未见过的长度

**旋转位置编码（RoPE）**

将位置信息编码到query和key向量的旋转中：
$$f(q, m) = qe^{im\theta}$$

优点是可以在Attention计算中自然体现相对位置。

---

## 24.6 Encoder-Decoder协同工作

### 24.6.1 架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                    Transformer 完整架构                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT                        OUTPUT                         │
│    ↓                            ↑                            │
│  Input Embedding           Output Embedding                  │
│    ↓                            ↓                            │
│  Positional Encoding       Positional Encoding               │
│    ↓                            ↓                            │
│  ┌─────────────────┐        ┌─────────────────┐              │
│  │   ENCODER ×6    │        │   DECODER ×6    │              │
│  │                 │        │                 │              │
│  │ ┌─────────────┐ │        │ ┌─────────────┐ │              │
│  │ │Multi-Head   │ │        │ │Masked Multi │ │              │
│  │ │Self-Attn   │ │───────→│ │-Head Self   │ │              │
│  │ │Add & Norm  │ │        │ │-Attn        │ │              │
│  │ └─────────────┘ │        │ │Add & Norm  │ │              │
│  │       ↓         │        │ └─────────────┘ │              │
│  │ ┌─────────────┐ │        │       ↓         │              │
│  │ │Feed Forward │ │        │ ┌─────────────┐ │              │
│  │ │Add & Norm  │ │        │ │Multi-Head   │ │              │
│  │ └─────────────┘ │        │ │Cross-Attn   │ │              │
│  │       ↓         │        │ │Add & Norm  │ │              │
│  │ [重复6层]       │        │ └─────────────┘ │              │
│  │                 │        │       ↓         │              │
│  │                 │        │ ┌─────────────┐ │              │
│  │                 │        │ │Feed Forward │ │              │
│  │                 │        │ │Add & Norm  │ │              │
│  │                 │        │ └─────────────┘ │              │
│  │                 │        │ [重复6层]       │              │
│  └─────────────────┘        └─────────────────┘              │
│           ↓                            ↓                     │
│      Encoder Output ───────────────→ 输出到Linear + Softmax   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 24.6.2 三种Attention类型

**1. Encoder Self-Attention**

```
输入: [我, 喜欢, 机器, 学习]
         ↓
每个词都关注所有输入词
         ↓
输出: [我', 喜欢', 机器', 学习']
      (每个词都融合了整句信息)
```

**2. Decoder Masked Self-Attention**

```
已生成: [I, like]
            ↓
    只能关注已生成的词（mask未来）
            ↓
下一个预测: machine
```

为什么要mask？因为解码器是自回归的，不能"偷看"未来的词。

**3. Cross-Attention (Encoder-Decoder Attention)**

```
Decoder当前状态: "预测第3个词"
         ↓
   Query来自Decoder
   Key, Value来自Encoder输出
         ↓
在源句子中找到最相关的词
         ↓
输出: 帮助解码的信息
```

这就是**翻译**的核心机制：解码时不断查询源句子的相关信息。

---

## 24.7 Transformer家族

### 24.7.1 Encoder-only架构：BERT

**特点**：
- 双向注意力
- 适合理解任务（分类、抽取、问答）

**代表模型**：
- BERT, RoBERTa, ALBERT, DeBERTa
- DistilBERT（蒸馏版）

### 24.7.2 Decoder-only架构：GPT

**特点**：
- 因果（自回归）注意力
- 适合生成任务（写作、对话、续写）

**代表模型**：
- GPT系列
- LLaMA, Mistral, Claude

### 24.7.3 Encoder-Decoder架构：T5

**特点**：
- 完整的编解码结构
- 适合翻译、摘要等seq2seq任务

**代表模型**：
- T5, BART, mT5

### 24.7.4 对比总结

| 特性 | BERT (Encoder) | GPT (Decoder) | T5 (Encoder-Decoder) |
|------|----------------|---------------|----------------------|
| 注意力 | 双向 | 因果（单向） | 编码器双向+解码器因果 |
| 预训练 | MLM + NSP | 语言模型 | Span Corruption |
| 主要用途 | 理解 | 生成 | 翻译/摘要 |
| 代表任务 | 分类、NER | 写作、对话 | 机器翻译 |

---

## 24.8 从零实现Transformer

现在让我们用PyTorch从零实现一个完整的Transformer模型。

*代码见 code/transformer_implementation.py*

### 24.8.1 Scaled Dot-Product Attention

```python
import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    计算缩放点积注意力
    
    参数:
        Q: Query矩阵, shape (batch, seq_len, d_k)
        K: Key矩阵, shape (batch, seq_len, d_k)
        V: Value矩阵, shape (batch, seq_len, d_v)
        mask: 可选的mask矩阵
    
    返回:
        output: 注意力输出
        attention_weights: 注意力权重
    """
    d_k = Q.size(-1)
    
    # 1. 计算点积: Q @ K^T
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq_len, seq_len)
    
    # 2. 缩放
    scores = scores / math.sqrt(d_k)
    
    # 3. 应用mask（如果需要）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 4. Softmax得到注意力权重
    attention_weights = torch.softmax(scores, dim=-1)
    
    # 5. 加权求和
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

### 24.8.2 Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """多头注意力模块"""
    
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def split_heads(self, x, batch_size):
        """将最后一个维度分割为(num_heads, d_k)"""
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. 线性投影
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. 分割成多头
        Q = self.split_heads(Q, batch_size)  # (batch, h, seq_len, d_k)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # 3. 计算缩放点积注意力
        attn_output, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask
        )
        
        # 4. 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # 5. 最终线性变换
        output = self.W_o(attn_output)
        
        return output, attn_weights
```

### 24.8.3 Positional Encoding

```python
class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # 计算div_term
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # 正弦和余弦
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_seq_length, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
```

### 24.8.4 Transformer Block

```python
class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 1. 多头自注意力 + 残差连接
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### 24.8.5 完整Transformer模型

```python
class Transformer(nn.Module):
    """完整的Transformer模型"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, max_seq_length=5000, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 嵌入
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 编码器
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 解码器
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src, src_mask=None):
        """编码源序列"""
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        """解码目标序列"""
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.decoder_layers:
            x = layer(x, tgt_mask)
        
        return self.output_layer(x)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        return output
```

---

## 24.9 应用示例：机器翻译

### 24.9.1 简易中英翻译Demo

```python
# 简化的中英翻译示例
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class SimpleTranslationDataset(Dataset):
    """简化的翻译数据集"""
    
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = [self.src_vocab.get(w, 0) for w in src.split()]
        tgt_ids = [self.tgt_vocab.get(w, 0) for w in tgt.split()]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

# 示例数据
translation_pairs = [
    ("我 喜欢 机器 学习", "I like machine learning"),
    ("你 好 世界", "hello world"),
    ("这 是 一个 测试", "this is a test"),
    ("今天 天气 很 好", "the weather is good today"),
]

# 构建词汇表
src_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}

for src, tgt in translation_pairs:
    for word in src.split():
        if word not in src_vocab:
            src_vocab[word] = len(src_vocab)
    for word in tgt.split():
        if word not in tgt_vocab:
            tgt_vocab[word] = len(tgt_vocab)
```

### 24.9.2 训练循环

```python
def train_transformer(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)
            
            # 目标输入是前n-1个词，目标是后n-1个词
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # 前向传播
            optimizer.zero_grad()
            output = model(src, tgt_input)
            
            # 计算损失
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
```

### 24.9.3 推理（翻译）

```python
def translate(model, src_sentence, src_vocab, tgt_vocab, max_len=50):
    """使用Transformer进行翻译"""
    model.eval()
    
    # 编码源句子
    src_ids = [src_vocab.get(w, 0) for w in src_sentence.split()]
    src_tensor = torch.tensor([src_ids]).to(device)
    
    # 从<sos>开始解码
    tgt_ids = [tgt_vocab['<sos>']]
    
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids]).to(device)
            output = model(src_tensor, tgt_tensor)
            
            # 取最后一个位置的预测
            next_token = output[0, -1].argmax().item()
            tgt_ids.append(next_token)
            
            # 如果预测到<eos>，停止
            if next_token == tgt_vocab['<eos>']:
                break
    
    # 转换回单词
    id_to_word = {v: k for k, v in tgt_vocab.items()}
    translation = [id_to_word[i] for i in tgt_ids[1:-1]]  # 去掉<sos>和<eos>
    return ' '.join(translation)
```

---

## 24.10 文本生成器

### 24.10.1 基于Transformer的文本生成

```python
class TextGenerator:
    """基于Transformer的文本生成器"""
    
    def __init__(self, model, vocab, device='cpu'):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.id_to_word = {v: k for k, v in vocab.items()}
    
    def generate(self, prompt, max_length=100, temperature=1.0, top_k=None):
        """
        生成文本
        
        参数:
            prompt: 起始文本
            max_length: 最大生成长度
            temperature: 采样温度（越高越随机）
            top_k: 只从top k个候选词中采样
        """
        self.model.eval()
        
        # 编码prompt
        tokens = [self.vocab.get(w, 0) for w in prompt.split()]
        input_tensor = torch.tensor([tokens]).to(self.device)
        
        generated = tokens[:]
        
        with torch.no_grad():
            for _ in range(max_length):
                # 准备输入
                input_ids = torch.tensor([generated]).to(self.device)
                
                # 获取模型输出
                outputs = self.model(input_ids, input_ids)
                logits = outputs[0, -1, :] / temperature
                
                # Top-k采样
                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('inf')
                
                # Softmax采样
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                generated.append(next_token)
                
                # 如果生成了结束符
                if next_token == self.vocab.get('<eos>', 0):
                    break
        
        # 转换为文本
        text = ' '.join([self.id_to_word.get(i, '<unk>') for i in generated])
        return text

# 使用示例
generator = TextGenerator(model, vocab, device)
result = generator.generate("机器学习 是", max_length=50, temperature=0.8, top_k=40)
print(result)
```

### 24.10.2 不同的采样策略

```python
def greedy_decode(logits):
    """贪心解码：总是选择概率最高的词"""
    return logits.argmax(dim=-1)

def beam_search(model, input_ids, beam_width=5, max_length=50):
    """束搜索：保留top k个候选序列"""
    sequences = [[input_ids, 0.0]]  # [序列, 分数]
    
    for _ in range(max_length):
        all_candidates = []
        
        for seq, score in sequences:
            output = model(seq, seq)
            logits = output[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
            # 获取top k
            top_k_probs, top_k_indices = torch.topk(probs, beam_width)
            
            for prob, idx in zip(top_k_probs, top_k_indices):
                new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                new_score = score + torch.log(prob).item()
                all_candidates.append([new_seq, new_score])
        
        # 选择得分最高的beam_width个
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    
    return sequences[0][0]  # 返回得分最高的序列
```

---

## 24.11 注意力权重可视化

### 24.11.1 可视化注意力热力图

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def visualize_attention(attention_weights, tokens, layer=0, head=0):
    """
    可视化注意力权重
    
    参数:
        attention_weights: 注意力权重矩阵 (num_layers, num_heads, seq_len, seq_len)
        tokens: 词列表
        layer: 要可视化的层
        head: 要可视化的头
    """
    attn = attention_weights[layer, head].cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, 
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='YlOrRd',
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Attention Weight'})
    
    plt.title(f'Attention Weights - Layer {layer}, Head {head}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    plt.show()

# 获取并可视化注意力
with torch.no_grad():
    output, attn_weights = model.encoder_layers[0].self_attn(
        embedded_input, embedded_input, embedded_input
    )

tokens = ["我", "喜欢", "机器", "学习"]
visualize_attention(attn_weights.unsqueeze(0), tokens, layer=0, head=0)
```

### 24.11.2 BertViz风格的注意力可视化

```python
def plot_attention_flow(tokens, attention_weights):
    """绘制注意力流向图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    seq_len = len(tokens)
    x_pos = np.arange(seq_len)
    
    # 绘制词的位置
    ax.scatter(x_pos, np.zeros(seq_len), s=200, c='blue', zorder=3)
    
    for i, token in enumerate(tokens):
        ax.annotate(token, (x_pos[i], 0), 
                   textcoords="offset points", 
                   xytext=(0, -20),
                   ha='center',
                   fontsize=12,
                   fontweight='bold')
    
    # 绘制注意力连接
    for i in range(seq_len):
        for j in range(seq_len):
            weight = attention_weights[i, j]
            if weight > 0.1:  # 只显示显著的连接
                ax.annotate('', 
                           xy=(x_pos[j], 0),
                           xytext=(x_pos[i], 1),
                           arrowprops=dict(arrowstyle='-',
                                         color='red',
                                         alpha=weight,
                                         lw=weight*3))
    
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')
    ax.set_title('Attention Flow Visualization', fontsize=14)
    plt.tight_layout()
    plt.show()
```

---

## 24.12 本章小结

### 24.12.1 核心概念回顾

**1. Attention机制的本质**
- Query-Key-Value的信息检索框架
- 让模型学会"看哪里"，有选择地关注相关信息
- 解决了RNN的长距离依赖问题

**2. Self-Attention的革命性**
- 完全并行计算
- 任意两个位置的距离都是$O(1)$
- 可解释性强（注意力权重直观显示词关系）

**3. Transformer架构**
- Encoder: 双向Self-Attention，理解输入
- Decoder: 因果Self-Attention + Cross-Attention，生成输出
- Multi-Head: 多角度的信息提取
- Positional Encoding: 注入位置信息

**4. 缩放因子的重要性**
- $\frac{1}{\sqrt{d_k}}$ 防止softmax进入饱和区
- 保证梯度正常流动

**5. Transformer家族**
- Encoder-only (BERT): 适合理解任务
- Decoder-only (GPT): 适合生成任务
- Encoder-Decoder (T5): 适合翻译/摘要

### 24.12.2 数学公式汇总

**Scaled Dot-Product Attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Multi-Head Attention:**
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

**Positional Encoding:**
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**Layer Normalization:**
$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

### 24.12.3 关键对比

| 特性 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 并行性 | ❌ 顺序计算 | ✅ 完全并行 |
| 长距离依赖 | ❌ 梯度消失 | ✅ 直接连接 |
| 计算复杂度 | $O(n)$ | $O(n^2)$ |
| 训练速度 | 慢 | 快（GPU并行） |
| 可解释性 | 弱 | 强（注意力权重） |

---

## 24.13 练习题

### 基础题

**题24.1** [理解Self-Attention] 给定Query向量 $q = [1, 0, 1]$，Key矩阵 $K = [[1, 0, 0], [0, 1, 1], [1, 1, 0]]$，Value矩阵 $V = [[2, 3], [1, 4], [5, 2]]$，假设$d_k = 3$，手动计算Attention输出。

**提示**：
1. 计算 $qK^T$ 得到分数
2. 除以$\sqrt{d_k}$
3. Softmax归一化
4. 与$V$加权求和

---

**题24.2** [理解位置编码] 对于$d_{model} = 4$，计算位置0、1、2的位置编码向量。验证：位置1和2之间的相对位置关系可以通过位置编码的线性变换表示。

**提示**：
$$PE_{(pos, 0)} = \sin(pos/10000^0)$$
$$PE_{(pos, 1)} = \cos(pos/10000^0)$$
$$PE_{(pos, 2)} = \sin(pos/10000^{2/4})$$
$$PE_{(pos, 3)} = \cos(pos/10000^{2/4})$$

---

**题24.3** [理解Multi-Head] 假设$d_{model} = 512$，使用8个头。
1. 每个头的$d_k$是多少？
2. 如果序列长度为100，每个头的Attention矩阵形状是什么？
3. 拼接后的输出形状是什么？

---

### 进阶题

**题24.4** [推导Masked Attention] 在Decoder的Self-Attention中，为什么要使用上三角mask？请画出对于序列长度为4的mask矩阵，并解释其作用。

**答案框架**：
```
mask = [[1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]]
```
解释每个位置的mask如何阻止"看到"未来信息。

---

**题24.5** [复杂度分析] 比较RNN和Transformer的计算复杂度：
1. 对于序列长度$n$，RNN每步的计算复杂度是多少？总复杂度是多少？
2. Transformer Self-Attention的复杂度是多少？
3. 在什么情况下Transformer比RNN更快？

---

**题24.6** [实现LayerNorm] 从零实现Layer Normalization，并比较与Batch Normalization的区别。

```python
class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
    
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        # 请实现LayerNorm
        pass
```

---

### 挑战题

**题24.7** [实现RoPE] 旋转位置编码（RoPE）是现代大模型（如LLaMA）使用的位置编码方法。请实现RoPE，并比较其与正弦位置编码的优缺点。

**RoPE公式**：
$$f(q, m) = qe^{im\theta}$$
其中$m$是位置，$\theta$是频率。

**参考实现思路**：
```python
def apply_rope(x, positions, base=10000):
    """
    对输入x应用旋转位置编码
    x: (batch, seq_len, d_model)
    positions: (seq_len,)
    """
    # 将相邻维度组成复数对
    # 应用旋转
    # 返回旋转后的结果
    pass
```

---

**题24.8** [KV Cache优化] 在Transformer的自回归生成中，我们可以使用KV Cache来避免重复计算。请：
1. 解释KV Cache的原理
2. 计算使用KV Cache前后的计算量对比
3. 实现带KV Cache的Decoder推理

**提示**：
- 不缓存时，每次生成都重新计算所有位置的K和V
- 缓存后，只需计算新位置的K和V，与之前的拼接

---

## 24.14 参考文献

### 核心必读

1. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.** (2017). Attention is all you need. In *Advances in Neural Information Processing Systems* (pp. 5998-6008).
   - ⭐ Transformer原始论文，深度学习史上最重要的论文之一

2. **Bahdanau, D., Cho, K., & Bengio, Y.** (2015). Neural machine translation by jointly learning to align and translate. In *Proceedings of ICLR 2015*.
   - 第一个成功的Attention机制，开创了Sequence-to-Sequence+Attention的范式

3. **Luong, M. T., Pham, H., & Manning, C. D.** (2015). Effective approaches to attention-based neural machine translation. In *Proceedings of EMNLP 2015* (pp. 1412-1421).
   - 提出了全局和局部Attention，以及乘法Attention

### 预训练模型

4. **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K.** (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019* (pp. 4171-4186).
   - 双向Transformer的里程碑，开启了预训练时代

5. **Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D.** (2020). Language models are few-shot learners. In *Advances in Neural Information Processing Systems* (pp. 1877-1901).
   - GPT-3，展示了规模带来的"涌现能力"

6. **Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J.** (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140), 1-67.
   - T5，统一的Text-to-Text框架

### 位置编码演进

7. **Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y.** (2024). RoFormer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568, 127063.
   - 旋转位置编码，被LLaMA等模型采用

8. **Press, O., Smith, N. A., & Lewis, M.** (2022). Train short, test long: Attention with linear biases enables input length extrapolation. In *Proceedings of ICLR 2022*.
   - ALiBi位置编码，简单有效的长度外推方法

9. **Shaw, P., Uszkoreit, J., & Vaswani, A.** (2018). Self-attention with relative position representations. In *Proceedings of NAACL-HLT 2018* (pp. 464-468).
   - 相对位置编码的早期工作

### 优化与扩展

10. **Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q. V., & Salakhutdinov, R.** (2019). Transformer-XL: Attentive language models beyond a fixed-length context. In *Proceedings of ACL 2019* (pp. 2978-2988).
    - 解决Transformer的长文本问题

11. **Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V.** (2019). RoBERTa: A robustly optimized BERT pre-training approach. *arXiv preprint arXiv:1907.11692*.
    - BERT的优化版本，展示了训练技巧的重要性

12. **Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I.** (2019). Language models are unsupervised multitask learners. *OpenAI Blog*.
    - GPT-2，展示了大规模语言模型的潜力

---

*本章完*

> 💡 **寄语**：理解Transformer是理解现代AI的关键。从Bahdanau Attention到Transformer，再到BERT和GPT，每一步都是对"如何让机器理解语言"这个问题的深入探索。掌握了这些，你就掌握了深度学习的核心。

*本章总字数：约15,000字 | 代码行数：约1,000行 | 参考文献：12篇*
