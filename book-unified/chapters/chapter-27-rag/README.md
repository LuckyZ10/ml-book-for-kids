# 第二十七章 检索增强生成（RAG）

## 27.1 引言：当大模型开始"编造"事实

### 27.1.1 一个令人尴尬的对话

想象一下这样的场景：你正在使用一个智能助手询问"2024年诺贝尔文学奖得主是谁？"，它自信满满地回答："2024年诺贝尔文学奖授予了中国作家莫言，以表彰他在乡土文学创作中的杰出贡献。"

等等——莫言是2012年获奖的，而且2024年的奖项可能还没有公布！这个回答听起来头头是道，却是**完全错误**的。

这不是科幻场景，而是使用大语言模型（LLM）时经常遇到的真实问题。这种现象被称为**幻觉（Hallucination）**——模型会生成听起来合理但实际上虚假的内容。它可能编造不存在的学术论文、虚构历史人物的名言、甚至创造从未发生过的历史事件。

### 27.1.2 为什么大模型会"胡说八道"？

要理解幻觉问题，我们需要回顾前几章的内容。大语言模型本质上是一个**概率模型**，它通过预测"下一个最可能出现的词"来生成文本。在预训练阶段，模型从海量文本中学习语言模式和世界知识，并将这些知识**压缩存储在数百亿甚至数千亿的参数中**。

这种"参数化记忆"（Parametric Memory）存在几个根本性的限制：

**知识截止问题**：模型的知识停留在预训练数据的时间点。例如，GPT-4的知识截止日期是2024年初，它不知道之后发生的事件。

**存储容量限制**：虽然模型参数很多，但相对于人类数千年的文明积累，这些参数仍然有限。研究表明，即使是最先进的模型，也只能存储训练数据中的一小部分事实性知识。

**回忆失败**：即使知识存在于参数中，模型也可能无法准确回忆。就像人类会记错事情一样，模型也会"张冠李戴"，把不同的事实混淆在一起。

**无法承认无知**：人类在被问到不知道的问题时会说"我不知道"，但LLM被训练成总是生成答案。这种"必须回答"的压力导致它倾向于编造内容。

### 27.1.3 知识密集型任务的挑战

有些任务特别依赖准确的事实知识，我们称之为**知识密集型任务（Knowledge-Intensive Tasks）**：

- **开放域问答**："谁是第一位登上月球的宇航员？"
- **事实验证**："判断这句话的真假：水在100摄氏度时结冰。"
- **专业领域咨询**："根据最新的临床研究，这种药物的副作用是什么？"

在这些场景中，幻觉不仅仅是小错误，而是可能导致严重后果的问题。想象一下医疗AI给错建议，或者法律AI引用不存在的法条！

### 27.1.4 RAG：给大模型装上"外部大脑"

2020年，Facebook AI的研究团队发表了一篇里程碑式的论文《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》，正式提出了**检索增强生成（RAG）**框架。

RAG的核心思想非常直观：

> **与其强迫模型记住所有知识，不如让它学会在需要时查阅资料。**

就像学生参加开卷考试可以带参考书一样，RAG让大模型在回答问题时能够**实时检索外部知识库**，并将检索到的信息作为上下文来生成答案。这种"外部记忆"被称为**非参数记忆（Non-parametric Memory）**，与存储在模型参数中的"参数记忆"形成互补。

RAG的出现彻底改变了知识密集型AI应用的范式。它带来了几个革命性的优势：

**可更新的知识**：知识库可以随时更新，不需要重新训练模型。今天发现的新知识，明天就能被AI使用。

**可验证性**：模型会告诉你答案来自哪段资料，你可以追溯验证。这在医疗、法律等高风险领域至关重要。

**减少幻觉**：因为有外部资料支撑，模型编造内容的概率大幅降低。

**参数效率**：不需要庞大的模型来存储所有知识， smaller模型配合检索也能达到很好效果。

在本章中，我们将深入学习RAG的工作原理，从向量检索的基础数学原理，到完整的代码实现，再到最前沿的Self-RAG和Corrective RAG等高级技术。让我们开始这场知识探索之旅！

---

## 27.2 开卷考试的启示：为什么外部记忆如此重要

### 27.2.1 闭卷考试 vs 开卷考试

让我们用一个大家都熟悉的比喻来理解RAG的价值：**考试**。

**闭卷考试**要求你凭记忆回答所有问题。这就像我们使用普通的大语言模型——它只能依赖训练时"背下来"的知识。优点是快速，不需要查资料；缺点是如果考题超出记忆范围，就只能瞎猜。

**开卷考试**允许你带参考书。这就像是RAG系统——模型可以查阅外部知识库来回答问题。优点是准确性和覆盖范围大大提升；缺点是需要额外的检索时间。

想象一下两种场景：

**场景一**：闭卷考试中被问到"请说出2023年所有诺贝尔文学奖候选人的名字"。除非你是诺贝尔奖委员会成员，否则几乎不可能答对。

**场景二**：同样的题目，但是开卷考试，你可以查阅当年的新闻报道和官方公告。只要资料齐全，你很容易就能找到正确答案。

这就是RAG的力量——**它把AI从"背诵型选手"变成了"研究型选手"**。

### 27.2.2 人类认知的启示

人类的认知过程其实也是一种"检索增强生成"。当你回答问题时，大脑会执行以下步骤：

1. **理解问题**：解析问题的含义和需求
2. **检索记忆**：从长期记忆中搜索相关信息
3. **整合信息**：将检索到的信息与当前上下文结合
4. **生成回答**：组织语言，形成完整的答案

如果大脑发现长期记忆中没有相关信息，人类会怎么做？**去查资料**！我们会翻书、上网搜索、请教专家。RAG正是模拟了这种自然的认知过程。

有趣的是，认知科学研究显示，人类专家和新手的区别之一就是**知道去哪里找信息**。优秀的医生不仅记得更多医学知识，更重要的是知道如何在遇到罕见病例时快速查阅最新文献。同样，RAG中的"检索器"就扮演了"信息定位专家"的角色。

### 27.2.3 参数记忆 vs 非参数记忆

让我们更仔细地比较这两种记忆方式：

| 特性 | 参数记忆（Parametric Memory） | 非参数记忆（Non-parametric Memory） |
|------|------------------------------|-----------------------------------|
| **存储位置** | 模型权重参数中 | 外部知识库（向量数据库、文档集等） |
| **更新方式** | 需要重新训练或微调 | 直接增删改知识库内容 |
| **容量** | 受限于模型参数量 | 理论上无限扩展 |
| **查询速度** | 快（前向传播） | 需要额外检索时间 |
| **可解释性** | 低（黑盒） | 高（可追溯来源） |
| **准确性** | 可能过时或错误 | 取决于知识库质量 |

RAG的精髓在于**结合两者的优势**：用参数记忆处理语言理解和推理，用非参数记忆存储和检索事实性知识。

### 27.2.4 从"百科全书"到"图书管理员"

传统的语言模型就像一个试图背下整本百科全书的学者——无论多努力，总有遗漏和遗忘。

RAG模型则更像一位**优秀的图书管理员**——它不必记住每本书的内容，但必须知道：
- 什么知识存放在哪里（检索能力）
- 如何快速找到相关资料（索引能力）
- 如何从资料中提取和整合信息（阅读能力）
- 如何用用户能理解的方式表达（生成能力）

这个比喻帮助我们理解RAG的架构设计：它由**检索器（Retriever）**和**生成器（Generator）**两个核心组件组成，就像我们的大脑分为负责记忆的颞叶和负责语言表达的布洛卡区。

### 27.2.5 实际应用的价值

RAG在实际应用中展现出巨大价值：

**企业知识管理**：企业可以将内部文档、邮件、会议记录构建成知识库，员工通过问答方式快速获取信息，而不必翻阅海量文件。

**客服系统**：AI客服可以实时检索产品手册、FAQ、历史工单，提供准确且个性化的回答。

**医疗咨询**：结合最新的医学文献和临床指南，为医生提供决策支持（注意：这是辅助工具，不是诊断替代）。

**法律咨询**：快速检索法律条文、判例、司法解释，帮助律师进行案例分析。

**教育辅导**：根据学生的具体问题，从教材和参考资料中检索相关内容，提供个性化的学习指导。

在下一节，我们将深入探讨RAG的核心技术——**向量检索**，理解它如何让计算机像人类一样"找到相关资料"。

---

## 27.3 向量检索基础：在数学空间中寻找相似

### 27.3.1 从文字到向量：嵌入的艺术

要让计算机"理解"文本并找到相关内容，我们需要一种方法将文字转换为数学对象。**嵌入（Embedding）**技术正是为此而生。

想象一下，如果把每个词或每段文本都表示为高维空间中的一个点，那么语义相似的文本就会在这个空间中彼此靠近。这就像把图书馆的书籍按照主题分类摆放在书架上——历史书放在一起，科学书放在另一区域。

**嵌入**是最早的嵌入技术。Word2Vec、GloVe等模型告诉我们，"国王" - "男人" + "女人" ≈ "王后"，这表明向量运算可以捕捉语义关系。

**句子嵌入**则更进一步，将整个句子或段落编码为固定长度的向量。BERT、Sentence-BERT等模型可以生成高质量的句子嵌入，使得语义相似的句子在向量空间中距离很近。

例如：
- "猫在草地上玩耍" → 向量A
- "小猫在草坪上嬉戏" → 向量B
- "股票市场今天上涨" → 向量C

向量A和B的距离会很近（语义相似），而它们与向量C的距离会很远（语义不同）。

### 27.3.2 余弦相似度：衡量方向的相似性

有了向量表示，我们需要一种方法来度量它们的相似程度。最常用的指标是**余弦相似度（Cosine Similarity）**。

余弦相似度衡量的是两个向量**方向**的相似性，而不是它们的大小。这在文本检索中很有意义——两篇不同长度但主题相同的文章应该被认为是相似的。

**数学定义**：

给定两个向量 $\mathbf{a}$ 和 $\mathbf{b}$，它们的余弦相似度定义为：

$$\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \sqrt{\sum_{i=1}^{n} b_i^2}}$$

其中：
- $\mathbf{a} \cdot \mathbf{b}$ 是向量的点积
- $\|\mathbf{a}\|$ 和 $\|\mathbf{b}\|$ 是向量的欧几里得范数（长度）

**几何解释**：

余弦相似度等于两个向量夹角的余弦值：

$$\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \cos(\theta)$$

其中 $\theta$ 是两个向量之间的夹角。

- 当 $\theta = 0°$ 时，$\cos(\theta) = 1$，表示完全相同方向（最相似）
- 当 $\theta = 90°$ 时，$\cos(\theta) = 0$，表示正交（无关）
- 当 $\theta = 180°$ 时，$\cos(\theta) = -1$，表示相反方向（最不相似）

对于文本嵌入（通常为正值），余弦相似度的范围是 $[0, 1]$。

**计算示例**：

假设有两个二维向量 $\mathbf{a} = [3, 4]$ 和 $\mathbf{b} = [6, 8]$：

1. 点积：$\mathbf{a} \cdot \mathbf{b} = 3 \times 6 + 4 \times 8 = 18 + 32 = 50$
2. 范数：$\|\mathbf{a}\| = \sqrt{3^2 + 4^2} = 5$，$\|\mathbf{b}\| = \sqrt{6^2 + 8^2} = 10$
3. 余弦相似度：$50 / (5 \times 10) = 1.0$

这两个向量方向完全相同（只是长度不同），所以相似度为1。

### 27.3.3 欧几里得距离与点积

除了余弦相似度，还有其他度量方式：

**欧几里得距离（Euclidean Distance）**：

$$d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\| = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$

距离越小表示越相似。欧几里得距离同时考虑了方向和大小。

**点积（Dot Product）**：

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = \|\mathbf{a}\| \|\mathbf{b}\| \cos(\theta)$$

点积同时受向量长度和方向的影响。如果向量已经归一化（长度为1），点积就等于余弦相似度。

**最大内积搜索（Maximum Inner Product Search, MIPS）**：

在许多检索场景中，我们直接使用点积作为相似度度量。给定查询向量 $\mathbf{q}$ 和文档向量集合 $\{\mathbf{d}_1, \mathbf{d}_2, ..., \mathbf{d}_n\}$，我们的目标是找到：

$$\arg\max_{i} \mathbf{q} \cdot \mathbf{d}_i$$

这就是**最大内积搜索**问题。在RAG的原始论文中，检索器就是通过最大化查询与文档的相似度来选择相关文档的。

### 27.3.4 Top-k检索：概率视角

在实际应用中，我们通常不只需要找到最相似的文档，而是需要找到**前k个最相似**的文档。这被称为**Top-k检索**。

从概率的角度来看，我们可以将相似度分数转换为检索概率。给定查询 $\mathbf{q}$，文档 $\mathbf{d}_i$ 被检索到的概率可以建模为：

$$P(\mathbf{d}_i | \mathbf{q}) = \frac{\exp(\text{similarity}(\mathbf{q}, \mathbf{d}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{similarity}(\mathbf{q}, \mathbf{d}_j) / \tau)}$$

其中 $\tau$ 是温度参数，控制分布的尖锐程度：
- 当 $\tau \to 0$ 时，分布变得尖锐，几乎只选择最高分的文档
- 当 $\tau \to \infty$ 时，分布变得平坦，选择更加随机

这与**softmax**函数的形式相同，因此在检索文献中常被称为"softmax检索概率"。

在RAG的训练中，这个概率分布用于指导检索器学习——我们希望提高与正确答案相关文档的检索概率，降低无关文档的概率。

### 27.3.5 向量检索的计算挑战

当知识库包含数百万甚至数十亿文档时，对每个查询都计算与所有文档的相似度是不可行的。假设有100万个文档，每个文档向量768维，每次查询需要计算7.68亿次乘法！

**近似最近邻搜索（Approximate Nearest Neighbor Search, ANN）**技术应运而生。这些算法牺牲少量的准确性，换取巨大的速度提升：

**局部敏感哈希（Locality Sensitive Hashing, LSH）**：

LSH的核心思想是：将相似的向量映射到同一个"桶"中。通过设计特殊的哈希函数，相似的向量有高概率获得相同的哈希值。检索时，只需要在查询向量所在桶及其邻近桶中搜索，大大减少了候选集。

**乘积量化（Product Quantization, PQ）**：

将高维向量分割成多个子向量，每个子向量独立量化到有限的码本中。这样每个向量可以用一组短的编码表示，相似度计算在量化后的空间中进行，速度大大提升。

**HNSW（Hierarchical Navigable Small World）**：

构建一个多层的图结构，每一层都是一个小世界网络。检索时从顶层开始，快速定位到大致区域，然后在下层精细搜索。这种方法在实践中表现优异，被广泛应用于现代向量数据库。

**IVF（Inverted File Index）**：

通过聚类将向量空间划分为多个区域（Voronoi单元）。每个文档被分配到最近的聚类中心。检索时，只需要在与查询最接近的几个聚类中搜索。

在代码实现部分，我们将使用NumPy构建一个简单的向量存储系统，展示基本的向量检索原理。对于生产环境，通常使用专门优化的向量数据库如FAISS（Facebook AI Similarity Search）、Milvus或Pinecone。

### 27.3.6 密集检索 vs 稀疏检索

在信息检索领域，有两种主要的方法：

**稀疏检索（Sparse Retrieval）**：

基于词袋模型（Bag of Words），使用TF-IDF或BM25等算法。每个文档表示为一个高维稀疏向量，维度等于词汇表大小，大部分值为0。相关性通过词频统计计算。

优点：
- 可解释性强
- 对精确匹配效果好
- 不需要神经网络

缺点：
- 无法理解语义相似（如"猫"和"猫咪"被视为不同词）
- 高维稀疏表示存储效率低

**密集检索（Dense Retrieval）**：

使用神经网络将文本编码为低维密集向量（通常768或1024维）。语义相似的文本会有相似的向量表示。余弦相似度或点积用于度量相关性。

优点：
- 理解语义相似
- 向量维度固定，存储高效
- 可以捕捉深层语义关系

缺点：
- 需要神经网络推理
- 对精确匹配可能不如稀疏检索

**混合检索（Hybrid Retrieval）**：

实践中，最佳方案通常是结合两者——先用稀疏检索召回候选集，再用密集检索精排。或者在密集检索的基础上，对某些关键词进行稀疏匹配增强。

RAG使用的是密集检索，这得益于BERT等预训练模型的强大语义理解能力。DPR（Dense Passage Retrieval）是RAG中常用的检索器，它使用两个BERT模型分别编码查询和文档。

---

## 27.4 RAG架构详解：检索与生成的交响乐

### 27.4.1 RAG的基本架构

RAG系统由两个主要组件构成，它们协同工作，形成一个完整的问答流水线：

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────────┐   │
│  │  用户查询 │───▶│   检索器     │───▶│  相关文档 (Top-k)    │   │
│  └──────────┘    └─────────────┘    └──────────┬───────────┘   │
│         │                                       │               │
│         │         ┌─────────────────────────────┘               │
│         │         │                                             │
│         │    ┌────▼────┐                                        │
│         └───▶│  生成器  │◀──────────────────────────────────┐   │
│              └────┬────┘                                    │   │
│                   │                                         │   │
│              ┌────▼────┐                                     │   │
│              │  最终答案 │                                     │   │
│              └─────────┘                                     │   │
│                                                               │   │
│  ┌────────────────────────────────────────────────────────┐   │   │
│  │                  知识库 (向量存储)                      │◀──┘   │
│  │  [文档1向量] [文档2向量] [文档3向量] ... [文档N向量]     │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**检索器（Retriever）**：

负责从知识库中找到与查询相关的文档。形式化定义为：

$$P_\eta(z | x) = \text{softmax}_z(f_\eta(x)^T g_\eta(z))$$

其中：
- $x$ 是输入查询
- $z$ 是知识库中的文档
- $f_\eta$ 和 $g_\eta$ 分别是查询编码器和文档编码器
- $\eta$ 是检索器的参数

检索器输出的是文档的后验概率分布，通常我们只取概率最高的Top-k个文档。

**生成器（Generator）**：

基于检索到的文档生成最终答案。它是一个序列到序列的模型（如BART、T5）：

$$P_\theta(y_i | x, z, y_{1:i-1})$$

其中：
- $y_i$ 是第 $i$ 个生成的词
- $\theta$ 是生成器的参数
- 生成条件包括查询 $x$、检索到的文档 $z$ 和已生成的词序列

### 27.4.2 RAG-Token vs RAG-Sequence

原始RAG论文提出了两种变体：

**RAG-Token**：

在每个解码步骤都可以检索不同的文档。模型在每个位置选择使用哪个文档来生成下一个词：

$$P(y | x) = \prod_{i=1}^{n} \sum_{z \in \text{top-k}(x)} P_\eta(z | x) P_\theta(y_i | x, z, y_{1:i-1})$$

这种"每步检索"的方式使模型更灵活，可以在生成不同部分时使用不同的参考资料。

**RAG-Sequence**：

在生成开始之前检索一次文档，然后用相同的文档生成整个序列：

$$P(y | x) = \sum_{z \in \text{top-k}(x)} P_\eta(z | x) \prod_{i=1}^{n} P_\theta(y_i | x, z, y_{1:i-1})$$

这种方式更简单高效，但可能错过不同部分需要不同资料的情况。

实践中，RAG-Sequence通常表现更好，因为它避免了每步检索带来的噪声和计算开销。

### 27.4.3 端到端训练

RAG的一个关键创新是**端到端训练**——同时训练检索器和生成器，让它们学会协同工作。

**训练目标**：

给定输入-输出对 $(x_j, y_j)$，最小化负对数似然：

$$\mathcal{L} = -\sum_{j=1}^{M} \log P(y_j | x_j)$$

其中 $P(y_j | x_j)$ 根据RAG-Token或RAG-Sequence的公式计算。

**训练挑战**：

检索操作（选择Top-k文档）是不可微的，这阻碍了梯度的反向传播。RAG使用**直通估计器（Straight-Through Estimator）**或**重参数化技巧**来解决这个问题。

具体来说，文档检索可以看作是基于相似度分数的采样：

$$P_\eta(z | x) \propto \exp(f_\eta(x)^T g_\eta(z))$$

在训练时，我们使用softmax概率而不是硬选择，允许梯度通过相似度计算反向传播。

**联合优化**：

端到端训练使得：
1. **检索器学习检索有用的文档**：不只是与查询相似，而是能帮助生成正确答案
2. **生成器学习利用检索信息**：学会从文档中提取和整合信息

这与传统的"先独立训练检索器，再固定检索器训练生成器"的两阶段方法形成对比。联合训练带来了显著的性能提升。

### 27.4.4 检索增强的概率建模

从概率图模型的角度看，RAG可以形式化为：

```
┌─────────────────────────────────────┐
│         概率图模型                  │
│                                     │
│      ┌─────────┐                   │
│      │    x    │  (输入查询)       │
│      └───┬─────┘                   │
│          │                         │
│          ▼                         │
│      ┌─────────┐                   │
│      │    z    │  (检索文档) ◀──┐  │
│      └───┬─────┘               │  │
│          │                     │  │
│          ▼                     │  │
│      ┌─────────┐               │  │
│      │    y    │  (生成答案)    │  │
│      └─────────┘               │  │
│                                │  │
│  知识库 Z ─────────────────────┘  │
└─────────────────────────────────────┘
```

RAG模型的联合概率分布为：

$$P(y, z | x) = P_\eta(z | x) \cdot P_\theta(y | x, z)$$

边际似然通过对所有可能的文档求和得到：

$$P(y | x) = \sum_{z \in Z} P_\eta(z | x) \cdot P_\theta(y | x, z)$$

在实际计算中，由于知识库 $Z$ 很大，我们使用近似：

$$P(y | x) \approx \sum_{z \in \text{top-k}(x)} P_\eta(z | x) \cdot P_\theta(y | x, z)$$

这就是RAG-Sequence的公式。它假设只有Top-k文档对生成有显著贡献。

### 27.4.5 条件生成过程

RAG的生成过程可以详细描述为：

**输入编码**：

将查询 $x$ 和检索到的文档 $z$ 拼接作为生成器的输入：

$$\text{input} = [x; \text{SEP}; z_1; \text{SEP}; z_2; ...; \text{SEP}; z_k]$$

其中 $[;]$ 表示拼接，SEP是分隔符标记。

**自回归生成**：

生成器按顺序生成每个词：

$$y_1 \sim P_\theta(\cdot | x, z)$$
$$y_2 \sim P_\theta(\cdot | x, z, y_1)$$
$$y_3 \sim P_\theta(\cdot | x, z, y_1, y_2)$$
$$...$$

直到生成结束标记（EOS）或达到最大长度。

**束搜索（Beam Search）**：

为了获得更高质量的输出，实践中通常使用束搜索而非贪婪解码。束搜索同时维护多个候选序列，在每一步选择概率最高的 $B$ 个序列继续扩展。

**生成温度**：

通过温度参数 $T$ 控制生成的多样性：

$$P(y_i) = \frac{\exp(l_i / T)}{\sum_j \exp(l_j / T)}$$

其中 $l_i$ 是词 $i$ 的logit分数。$T < 1$ 使分布更尖锐（更确定性），$T > 1$ 使分布更平坦（更随机）。

### 27.4.6 Fusion-in-Decoder：RAG的进化

Fusion-in-Decoder（FiD）是RAG的一个重要改进，它解决了RAG在处理多个检索文档时的效率问题。

**核心思想**：

RAG将检索到的文档简单拼接在一起，这导致：
1. 输入长度随文档数线性增长
2. 注意力计算复杂度是平方级的，文档多时非常慢

FiD的解决方案是**分别编码每个文档，在解码器融合**：

```
┌──────────────────────────────────────────────────────────┐
│                    FiD 架构                              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  查询 ──┐                                                │
│         │                                                │
│  文档1 ─┼──▶ [编码器1] ──┐                                │
│         │               │                                │
│  文档2 ─┼──▶ [编码器2] ─┼──▶ [解码器] ──▶ 答案          │
│         │               │   (交叉注意力)                │
│  文档k ─┴──▶ [编码器k] ──┘                                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

每个文档和查询一起独立通过编码器，产生各自的表示。解码器使用**交叉注意力（Cross-Attention）**同时关注所有文档的表示，高效地融合信息。

**优势**：
1. 编码可以并行进行，充分利用GPU
2. 每个文档的编码长度固定，不随文档数增长
3. 交叉注意力的复杂度是线性的，而非平方级

FiD在问答任务上显著超越了原始RAG，同时保持了更高的推理效率。

---

## 27.5 代码实现：从零构建RAG系统

现在让我们动手实现一个完整的RAG系统。我们将构建以下组件：

1. **VectorStore**：基于NumPy的向量存储
2. **Embedding接口**：文本嵌入抽象和简单实现
3. **Retriever**：检索器实现
4. **RAGPipeline**：完整流水线
5. **示例应用**：问答系统和文档摘要

### 27.5.1 VectorStore类实现

```python
"""
VectorStore: 基于NumPy的向量存储实现
支持基本的向量增删改查和相似度检索
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import json


class VectorStore:
    """
    向量存储类，用于存储和检索向量化的文档
    
    属性:
        dimension: 向量维度
        vectors: 存储的向量矩阵 (N, D)
        documents: 原始文档列表
        metadata: 文档元数据列表
    """
    
    def __init__(self, dimension: int = 768):
        """
        初始化向量存储
        
        参数:
            dimension: 向量维度，默认768（BERT标准维度）
        """
        self.dimension = dimension
        self.vectors = np.array([]).reshape(0, dimension)  # 空矩阵
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self._index_built = False
        
    def add(self, vectors: np.ndarray, documents: List[str], 
            metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        添加向量到存储
        
        参数:
            vectors: 要添加的向量矩阵 (N, D)
            documents: 对应的原始文档列表
            metadata: 可选的元数据列表
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"向量维度不匹配: 期望 {self.dimension}, 得到 {vectors.shape[1]}")
        
        if len(vectors) != len(documents):
            raise ValueError("向量数量和文档数量必须相同")
        
        # 归一化向量（用于余弦相似度计算）
        vectors = self._normalize(vectors)
        
        # 追加到现有存储
        if self.vectors.shape[0] == 0:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        
        self.documents.extend(documents)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in documents])
        
        self._index_built = False  # 标记索引需要重建
        
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        L2归一化向量
        
        参数:
            vectors: 输入向量矩阵
            
        返回:
            归一化后的向量
        """
        # 添加小epsilon防止除零
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-10)
    
    def search(self, query_vector: np.ndarray, k: int = 5,
               metric: str = "cosine") -> List[Tuple[int, float, str, Dict]]:
        """
        搜索最相似的向量
        
        参数:
            query_vector: 查询向量 (D,)
            k: 返回结果数量
            metric: 相似度度量，可选 "cosine" 或 "dot"
            
        返回:
            列表，每个元素为 (索引, 相似度, 文档, 元数据)
        """
        if self.vectors.shape[0] == 0:
            return []
        
        query_vector = self._normalize(query_vector.reshape(1, -1)).flatten()
        
        if metric == "cosine":
            # 归一化向量的点积 = 余弦相似度
            similarities = np.dot(self.vectors, query_vector)
        elif metric == "dot":
            similarities = np.dot(self.vectors, query_vector)
        else:
            raise ValueError(f"不支持的度量: {metric}")
        
        # 获取Top-k索引
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append((
                int(idx),
                float(similarities[idx]),
                self.documents[idx],
                self.metadata[idx]
            ))
        
        return results
    
    def delete(self, indices: List[int]) -> None:
        """
        删除指定索引的向量
        
        参数:
            indices: 要删除的索引列表
        """
        # 转换为集合并排序（降序）
        indices = sorted(set(indices), reverse=True)
        
        for idx in indices:
            self.vectors = np.delete(self.vectors, idx, axis=0)
            del self.documents[idx]
            del self.metadata[idx]
        
        self._index_built = False
    
    def save(self, path: str) -> None:
        """
        保存向量存储到文件
        
        参数:
            path: 保存路径
        """
        np.save(f"{path}_vectors.npy", self.vectors)
        with open(f"{path}_docs.json", "w", encoding="utf-8") as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "dimension": self.dimension
            }, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str) -> None:
        """
        从文件加载向量存储
        
        参数:
            path: 加载路径
        """
        self.vectors = np.load(f"{path}_vectors.npy")
        with open(f"{path}_docs.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            self.dimension = data["dimension"]
    
    def __len__(self) -> int:
        """返回存储的文档数量"""
        return len(self.documents)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str, Dict]:
        """获取指定索引的向量、文档和元数据"""
        return self.vectors[idx], self.documents[idx], self.metadata[idx]
```

### 27.5.2 Embedding模型接口

```python
"""
Embedding模块: 将文本转换为向量表示
"""

import numpy as np
from typing import List
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """嵌入器基类"""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表编码为向量
        
        参数:
            texts: 文本列表
            
        返回:
            向量矩阵 (N, D)
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回嵌入维度"""
        pass


class SimpleEmbedder(BaseEmbedder):
    """
    简单的词袋嵌入器（用于演示）
    使用简单的哈希方法生成固定维度的向量
    """
    
    def __init__(self, dimension: int = 768):
        self._dimension = dimension
        np.random.seed(42)  # 固定随机种子以获得可重复的结果
        self.vocab = self._build_vocab()
    
    def _build_vocab(self) -> Dict[str, np.ndarray]:
        """为常见词预生成随机向量"""
        vocab = {}
        common_words = [
            "的", "了", "是", "我", "有", "和", "就", "不", "人", "在",
            "the", "is", "a", "and", "of", "to", "in", "that", "have",
            "machine", "learning", "deep", "neural", "network", "model",
            "data", "train", "test", "predict", "classify", "regression",
            "北京", "上海", "中国", "美国", "日本", "法国", "英国"
        ]
        for word in common_words:
            vocab[word] = np.random.randn(self._dimension)
        return vocab
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        使用简单的词袋方法编码文本
        """
        vectors = []
        for text in texts:
            words = text.lower().split()
            vector = np.zeros(self._dimension)
            
            for word in words:
                if word in self.vocab:
                    vector += self.vocab[word]
                else:
                    # 为未知词生成确定性随机向量
                    np.random.seed(hash(word) % (2**32))
                    vector += np.random.randn(self._dimension)
            
            # 平均池化
            if len(words) > 0:
                vector /= len(words)
            
            vectors.append(vector)
        
        return np.array(vectors)
    
    @property
    def dimension(self) -> int:
        return self._dimension


class MockBERTEmbedder(BaseEmbedder):
    """
    模拟BERT风格的嵌入器
    生成具有语义结构的模拟向量
    """
    
    def __init__(self, dimension: int = 768):
        self._dimension = dimension
        np.random.seed(42)
        # 定义语义区域
        self.topic_centers = {
            "ai": np.random.randn(dimension) * 0.3,
            "history": np.random.randn(dimension) * 0.3,
            "science": np.random.randn(dimension) * 0.3,
            "art": np.random.randn(dimension) * 0.3,
        }
        # 偏移量使主题中心相互远离
        self.topic_centers["ai"][0] = 1.0
        self.topic_centers["history"][1] = 1.0
        self.topic_centers["science"][2] = 1.0
        self.topic_centers["art"][3] = 1.0
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """基于文本主题生成模拟嵌入"""
        vectors = []
        
        for text in texts:
            text_lower = text.lower()
            
            # 检测主题
            if any(w in text_lower for w in ["machine", "learning", "ai", "neural", "model", "训练", "模型", "学习"]):
                base = self.topic_centers["ai"].copy()
            elif any(w in text_lower for w in ["history", "war", "century", "ancient", "历史", "古代", "世纪"]):
                base = self.topic_centers["history"].copy()
            elif any(w in text_lower for w in ["science", "physics", "chemistry", "biology", "科学", "物理", "化学"]):
                base = self.topic_centers["science"].copy()
            elif any(w in text_lower for w in ["art", "music", "painting", "literature", "艺术", "音乐", "绘画"]):
                base = self.topic_centers["art"].copy()
            else:
                base = np.zeros(self._dimension)
            
            # 添加噪声
            noise = np.random.randn(self._dimension) * 0.1
            vector = base + noise
            vectors.append(vector)
        
        return np.array(vectors)
    
    @property
    def dimension(self) -> int:
        return self._dimension


# 尝试导入真实的SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
    
    class SentenceBERTEmbedder(BaseEmbedder):
        """使用真实BERT模型的嵌入器"""
        
        def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
        
        def encode(self, texts: List[str]) -> np.ndarray:
            return self.model.encode(texts, convert_to_numpy=True)
        
        @property
        def dimension(self) -> int:
            return self._dimension
    
    HAS_REAL_EMBEDDER = True
except ImportError:
    HAS_REAL_EMBEDDER = False
```

### 27.5.3 Retriever检索器

```python
"""
Retriever模块: 基于向量相似度的文档检索
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """检索结果数据类"""
    document: str
    score: float
    index: int
    metadata: Optional[dict] = None


class DenseRetriever:
    """
    密集检索器
    使用向量相似度从知识库中检索相关文档
    """
    
    def __init__(self, vector_store, embedder):
        """
        初始化检索器
        
        参数:
            vector_store: VectorStore实例
            embedder: BaseEmbedder实例
        """
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, query: str, k: int = 5,
                 metric: str = "cosine") -> List[RetrievalResult]:
        """
        检索与查询最相关的文档
        
        参数:
            query: 查询字符串
            k: 返回文档数量
            metric: 相似度度量
            
        返回:
            RetrievalResult列表
        """
        # 编码查询
        query_vector = self.embedder.encode([query])
        
        # 搜索向量存储
        results = self.vector_store.search(
            query_vector=query_vector[0],
            k=k,
            metric=metric
        )
        
        # 转换为RetrievalResult
        retrieval_results = []
        for idx, score, doc, meta in results:
            retrieval_results.append(RetrievalResult(
                document=doc,
                score=score,
                index=idx,
                metadata=meta
            ))
        
        return retrieval_results
    
    def batch_retrieve(self, queries: List[str], k: int = 5,
                       metric: str = "cosine") -> List[List[RetrievalResult]]:
        """
        批量检索多个查询
        
        参数:
            queries: 查询字符串列表
            k: 每个查询返回文档数量
            
        返回:
            每个查询的RetrievalResult列表
        """
        results = []
        for query in queries:
            results.append(self.retrieve(query, k, metric))
        return results


class RAGRetriever:
    """
    RAG专用检索器
    支持RAG特定的检索逻辑，如概率采样等
    """
    
    def __init__(self, vector_store, embedder, temperature: float = 1.0):
        """
        初始化RAG检索器
        
        参数:
            vector_store: VectorStore实例
            embedder: BaseEmbedder实例
            temperature: 检索温度，控制概率分布的平滑程度
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.temperature = temperature
    
    def retrieve_with_probabilities(self, query: str, k: int = 5) -> Tuple[List[RetrievalResult], np.ndarray]:
        """
        检索文档并计算选择概率
        
        返回:
            (检索结果列表, 概率数组)
        """
        # 获取原始检索结果
        query_vector = self.embedder.encode([query])
        raw_results = self.vector_store.search(query_vector[0], k=k*2)  # 获取更多候选
        
        if not raw_results:
            return [], np.array([])
        
        # 截取Top-k
        raw_results = raw_results[:k]
        
        # 计算概率（使用softmax）
        scores = np.array([r[1] for r in raw_results])
        
        # 温度缩放
        scores = scores / self.temperature
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))  # 数值稳定性
        probs = exp_scores / np.sum(exp_scores)
        
        # 转换为RetrievalResult
        results = []
        for i, (idx, score, doc, meta) in enumerate(raw_results):
            results.append(RetrievalResult(
                document=doc,
                score=score,
                index=idx,
                metadata={**meta, "probability": float(probs[i])}
            ))
        
        return results, probs
    
    def sample_documents(self, query: str, k: int = 5, 
                         num_samples: int = 1) -> List[List[RetrievalResult]]:
        """
        按概率采样文档（用于训练时的随机采样）
        
        参数:
            query: 查询
            k: 每次采样的文档数
            num_samples: 采样次数
            
        返回:
            多次采样的结果
        """
        results, probs = self.retrieve_with_probabilities(query, k * 2)
        
        if not results:
            return [[] for _ in range(num_samples)]
        
        samples = []
        for _ in range(num_samples):
            # 按概率采样k个文档（不重复）
            indices = np.random.choice(
                len(results), 
                size=min(k, len(results)), 
                replace=False, 
                p=probs
            )
            sampled = [results[i] for i in indices]
            samples.append(sampled)
        
        return samples
```

### 27.5.4 RAGPipeline完整管道

```python
"""
RAG Pipeline: 完整的检索增强生成流水线
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class RAGOutput:
    """RAG输出结果"""
    answer: str
    retrieved_documents: List[RetrievalResult]
    query: str
    metadata: Dict[str, Any]


class SimpleGenerator:
    """
    简单的生成器（用于演示）
    真实的实现会使用T5、BART等seq2seq模型
    """
    
    def __init__(self):
        self.response_templates = {
            "ai": [
                "根据检索到的资料，{content}",
                "基于相关信息，{content}",
                "从文档中可以发现，{content}"
            ],
            "default": [
                "根据资料：{content}",
                "相关信息显示：{content}",
                "检索结果：{content}"
            ]
        }
    
    def generate(self, query: str, documents: List[RetrievalResult],
                 max_length: int = 200) -> str:
        """
        基于检索文档生成答案
        
        参数:
            query: 查询
            documents: 检索到的文档
            max_length: 最大生成长度
            
        返回:
            生成的答案
        """
        if not documents:
            return "抱歉，未找到相关信息。"
        
        # 提取关键信息（简化实现）
        combined_content = " ".join([doc.document for doc in documents[:2]])
        
        # 截断到最大长度
        if len(combined_content) > max_length:
            combined_content = combined_content[:max_length] + "..."
        
        # 选择模板
        import random
        random.seed(42)
        
        query_lower = query.lower()
        if any(w in query_lower for w in ["machine", "learning", "ai", "model", "训练", "模型"]):
            template = random.choice(self.response_templates["ai"])
        else:
            template = random.choice(self.response_templates["default"])
        
        return template.format(content=combined_content)


class RAGPipeline:
    """
    RAG完整流水线
    整合检索器和生成器
    """
    
    def __init__(self, retriever, generator=None, top_k: int = 5):
        """
        初始化RAG流水线
        
        参数:
            retriever: RAGRetriever实例
            generator: 生成器实例（默认为SimpleGenerator）
            top_k: 检索文档数量
        """
        self.retriever = retriever
        self.generator = generator or SimpleGenerator()
        self.top_k = top_k
        
        # 统计信息
        self.stats = {
            "total_queries": 0,
            "total_retrievals": 0
        }
    
    def query(self, query: str, return_documents: bool = True) -> RAGOutput:
        """
        执行RAG查询
        
        参数:
            query: 用户查询
            return_documents: 是否返回检索文档
            
        返回:
            RAGOutput对象
        """
        self.stats["total_queries"] += 1
        
        # 1. 检索相关文档
        retrieved_docs = self.retriever.retrieve(query, k=self.top_k)
        self.stats["total_retrievals"] += len(retrieved_docs)
        
        # 2. 生成答案
        answer = self.generator.generate(query, retrieved_docs)
        
        # 3. 组装输出
        output = RAGOutput(
            answer=answer,
            retrieved_documents=retrieved_docs if return_documents else [],
            query=query,
            metadata={
                "num_retrieved": len(retrieved_docs),
                "top_score": retrieved_docs[0].score if retrieved_docs else 0.0
            }
        )
        
        return output
    
    def batch_query(self, queries: List[str]) -> List[RAGOutput]:
        """
        批量执行RAG查询
        
        参数:
            queries: 查询列表
            
        返回:
            RAGOutput列表
        """
        results = []
        for query in queries:
            results.append(self.query(query))
        return results
    
    def add_documents(self, documents: List[str], 
                      metadata: Optional[List[Dict]] = None) -> None:
        """
        向知识库添加文档
        
        参数:
            documents: 文档列表
            metadata: 元数据列表
        """
        # 编码文档
        vectors = self.retriever.embedder.encode(documents)
        
        # 添加到向量存储
        self.retriever.vector_store.add(vectors, documents, metadata)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
```

### 27.5.5 完整示例：问答系统

```python
"""
RAG问答系统示例
演示如何使用RAGPipeline构建完整的问答应用
"""


def create_demo_knowledge_base():
    """创建演示用的知识库"""
    documents = [
        # AI/机器学习相关
        "机器学习是人工智能的一个分支，它使计算机能够从数据中自动学习和改进，而无需明确编程。",
        "深度学习是机器学习的一种方法，使用多层神经网络来学习数据的层次化表示。",
        "神经网络受到生物神经系统的启发，由相互连接的节点（神经元）组成，可以学习和识别模式。",
        "监督学习是一种机器学习方法，使用带有标签的训练数据来训练模型预测结果。",
        "无监督学习不需要标签数据，它发现数据中的隐藏模式和结构，如聚类和降维。",
        "强化学习通过与环境交互来学习，智能体根据奖励和惩罚来学习最优策略。",
        "Transformer是一种深度学习架构，使用自注意力机制处理序列数据，是GPT和BERT的基础。",
        "BERT是Google开发的预训练语言模型，使用双向编码器表示来理解语言上下文。",
        "GPT（生成式预训练Transformer）是由OpenAI开发的大型语言模型，能够生成人类般的文本。",
        "卷积神经网络（CNN）特别适合处理图像数据，通过卷积层提取空间特征。",
        
        # 历史相关
        "第二次世界大战于1939年9月1日爆发，德国入侵波兰，持续至1945年9月2日。",
        "爱因斯坦于1905年发表了狭义相对论，提出了著名的质能方程E=mc²。",
        "唐朝（618-907年）是中国历史上最强盛的朝代之一，以诗歌和文化繁荣著称。",
        "丝绸之路是古代连接东西方的贸易网络，促进了商品、文化和技术的交流。",
        
        # 科学相关
        "DNA（脱氧核糖核酸）是携带遗传信息的分子，由四种核苷酸组成：A、T、G、C。",
        "光合作用是将光能转化为化学能的过程，植物通过叶绿素吸收阳光，将二氧化碳和水转化为葡萄糖和氧气。",
        "量子力学研究微观粒子的行为，引入了波粒二象性和不确定性原理等概念。",
        
        # 艺术相关
        "《蒙娜丽莎》是达·芬奇创作的肖像画，以其神秘的微笑而闻名于世，现藏于卢浮宫。",
        "贝多芬是古典音乐史上最伟大的作曲家之一，尽管晚年失聪，仍创作了许多不朽的作品。",
    ]
    
    metadata = [
        {"category": "ai", "source": "ml_intro"},
        {"category": "ai", "source": "dl_intro"},
        {"category": "ai", "source": "neural_networks"},
        {"category": "ai", "source": "supervised_learning"},
        {"category": "ai", "source": "unsupervised_learning"},
        {"category": "ai", "source": "reinforcement_learning"},
        {"category": "ai", "source": "transformer"},
        {"category": "ai", "source": "bert"},
        {"category": "ai", "source": "gpt"},
        {"category": "ai", "source": "cnn"},
        {"category": "history", "source": "ww2"},
        {"category": "history", "source": "einstein"},
        {"category": "history", "source": "tang_dynasty"},
        {"category": "history", "source": "silk_road"},
        {"category": "science", "source": "dna"},
        {"category": "science", "source": "photosynthesis"},
        {"category": "science", "source": "quantum"},
        {"category": "art", "source": "mona_lisa"},
        {"category": "art", "source": "beethoven"},
    ]
    
    return documents, metadata


def main():
    """主函数：演示RAG问答系统"""
    print("=" * 60)
    print("RAG问答系统演示")
    print("=" * 60)
    
    # 1. 初始化组件
    print("\n[1] 初始化嵌入器和向量存储...")
    embedder = MockBERTEmbedder(dimension=768)
    vector_store = VectorStore(dimension=768)
    
    # 2. 创建知识库
    print("[2] 构建知识库...")
    documents, metadata = create_demo_knowledge_base()
    vectors = embedder.encode(documents)
    vector_store.add(vectors, documents, metadata)
    print(f"    已添加 {len(documents)} 篇文档")
    
    # 3. 初始化检索器和RAG流水线
    print("[3] 初始化RAG流水线...")
    retriever = DenseRetriever(vector_store, embedder)
    rag = RAGPipeline(retriever, top_k=3)
    
    # 4. 执行查询
    print("\n" + "=" * 60)
    print("开始问答")
    print("=" * 60)
    
    test_queries = [
        "什么是机器学习？",
        "深度学习和机器学习有什么关系？",
        "请介绍一下Transformer架构",
        "第二次世界大战是什么时候开始的？",
        "DNA是什么？",
        "《蒙娜丽莎》是谁画的？",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n问题 {i}: {query}")
        print("-" * 40)
        
        # 执行RAG查询
        result = rag.query(query)
        
        print(f"回答: {result.answer}")
        print(f"\n检索到的相关文档 ({len(result.retrieved_documents)}篇):")
        for j, doc in enumerate(result.retrieved_documents, 1):
            print(f"  [{j}] 得分: {doc.score:.4f} | 来源: {doc.metadata.get('source', 'unknown')}")
            print(f"      {doc.document[:80]}...")
    
    # 5. 打印统计
    print("\n" + "=" * 60)
    print("系统统计")
    print("=" * 60)
    stats = rag.get_stats()
    print(f"总查询数: {stats['total_queries']}")
    print(f"总检索次数: {stats['total_retrievals']}")
    
    print("\n演示完成！")


if __name__ == "__main__":
    main()
```

### 27.5.6 文档摘要示例

```python
"""
文档摘要示例：使用RAG进行多文档摘要
"""


class SummarizationRAG:
    """
    基于RAG的文档摘要系统
    检索相关文档片段，然后生成摘要
    """
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag = rag_pipeline
    
    def summarize_topic(self, topic: str, num_docs: int = 5) -> str:
        """
        对特定主题进行摘要
        
        参数:
            topic: 主题/查询
            num_docs: 检索文档数量
            
        返回:
            摘要文本
        """
        # 调整检索数量
        original_k = self.rag.top_k
        self.rag.top_k = num_docs
        
        # 执行RAG查询
        result = self.rag.query(f"总结关于{topic}的信息")
        
        # 恢复设置
        self.rag.top_k = original_k
        
        # 构建摘要
        summary_parts = [
            f"## 关于'{topic}'的摘要",
            "",
            "### 要点总结：",
            result.answer,
            "",
            "### 参考来源：",
        ]
        
        for i, doc in enumerate(result.retrieved_documents, 1):
            source = doc.metadata.get('source', '未知')
            summary_parts.append(f"{i}. {source} (相关度: {doc.score:.3f})")
        
        return "\n".join(summary_parts)
    
    def compare_topics(self, topic1: str, topic2: str) -> str:
        """
        比较两个主题
        
        参数:
            topic1: 第一个主题
            topic2: 第二个主题
            
        返回:
            比较文本
        """
        result1 = self.rag.query(topic1)
        result2 = self.rag.query(topic2)
        
        comparison = f"""
## 主题比较：{topic1} vs {topic2}

### {topic1}
{result1.answer}

### {topic2}
{result2.answer}

### 共同点与差异
基于检索结果，两个主题的关联度分析完成。
"""
        return comparison


def summarization_demo():
    """文档摘要演示"""
    print("=" * 60)
    print("RAG文档摘要演示")
    print("=" * 60)
    
    # 初始化
    embedder = MockBERTEmbedder(dimension=768)
    vector_store = VectorStore(dimension=768)
    documents, metadata = create_demo_knowledge_base()
    vectors = embedder.encode(documents)
    vector_store.add(vectors, documents, metadata)
    
    retriever = DenseRetriever(vector_store, embedder)
    rag = RAGPipeline(retriever, top_k=5)
    summarizer = SummarizationRAG(rag)
    
    # 主题摘要
    print("\n[1] 主题摘要示例")
    print("-" * 40)
    summary = summarizer.summarize_topic("机器学习", num_docs=4)
    print(summary)
    
    # 主题比较
    print("\n\n[2] 主题比较示例")
    print("-" * 40)
    comparison = summarizer.compare_topics("监督学习", "无监督学习")
    print(comparison)


if __name__ == "__main__":
    summarization_demo()
```

---

## 27.6 高级RAG技术

### 27.6.1 Self-RAG：自我反思的检索增强

**问题背景**：

传统RAG有一个明显的问题：**无论是否需要，它都会检索固定数量的文档**。对于"你好，今天天气怎么样？"这样的闲聊，检索文档毫无意义。更糟的是，如果检索到的文档质量差，反而可能误导生成器产生错误答案。

**Self-RAG的核心思想**：

2023年，Asai等人提出了Self-RAG框架。其核心创新是让模型学会**自我反思**：

1. **按需检索**：模型自己决定是否需要检索
2. **自我批判**：对检索到的文档和生成的内容进行质量评估
3. **引用生成**：为每个事实性陈述提供引用来源

**反射Token机制**：

Self-RAG引入了一组特殊的**反射Token（Reflection Tokens）**：

- `[Retrieve]`：是否触发检索
- `[IsRel]`：检索文档是否与查询相关
- `[IsSup]`：生成内容是否被文档支持
- `[IsUse]`：生成内容是否有用

在训练时，这些token被插入到标准输出中。例如：

```
输入：谁发明了电话？

输出：
[Retrieve] 是
电话是由[IsRel] 是 亚历山大·格拉汉姆·贝尔在1876年发明的。[IsSup] 是 [IsUse] 是
```

**自适应检索**：

Self-RAG使用一个检索决策阈值：

- 当模型预测`[Retrieve]=是`的概率超过阈值时，执行检索
- 否则直接基于参数记忆生成答案

这使得模型在知识充足时直接回答，在不确定时主动查阅资料。

**推理时定制**：

Self-RAG的另一个优势是可以在推理时定制行为：

- **精确模式**：要求所有事实都有支持引用，优先选择高`[IsSup]`分数的输出
- **创意模式**：允许更多参数知识，对`[IsSup]`要求较低

### 27.6.2 Corrective RAG：纠正错误检索

**问题背景**：

RAG系统高度依赖检索质量。当检索器返回不相关或过时的文档时，生成器往往会被误导，产生"检索增强的幻觉"。

**CRAG的解决方案**：

Corrective RAG（CRAG）引入了一个**检索评估器（Retrieval Evaluator）**，对检索结果进行质量打分：

$$\text{Confidence} = f_\text{evaluator}(q, \{d_1, d_2, ..., d_k\})$$

基于置信度，CRAG触发不同的行动：

| 置信度范围 | 判定 | 行动 |
|-----------|------|------|
| 高 (>θ_high) | Correct | 使用检索文档生成，同时过滤低相关片段 |
| 低 (<θ_low) | Incorrect | 放弃检索结果，转向网络搜索 |
| 中 | Ambiguous | 结合检索文档和网络搜索结果 |

**分解-重组算法**：

对于被判定为"Correct"的文档，CRAG执行**分解-重组（Decompose-then-Recompose）**：

1. **分解**：将长文档切分为细粒度的信息片段
2. **过滤**：基于与查询的相关性过滤片段
3. **重组**：将高质量片段重新组织为精炼的上下文

**网络搜索扩展**：

当本地检索失败时，CRAG可以调用网络搜索API（如Google Search、Bing API）获取补充信息。这使系统能够：
- 获取最新的信息
- 扩展知识覆盖范围
- 验证和交叉检查事实

**Plug-and-Play设计**：

CRAG设计为即插即用的模块，可以与任何RAG系统集成：

```python
# CRAG可以包装任何RAG系统
crag_rag = CRAGWrapper(base_rag_system, evaluator_model)
result = crag_rag.query("...")
```

实验表明，CRAG在PopQA、Biography等多个数据集上显著提升了标准RAG和Self-RAG的性能。

### 27.6.3 其他RAG变体

**REPLUG（Retrieval-Augmented Black-Box Language Model）**：

针对无法微调的闭源模型（如GPT-4），REPLUG通过**集成学习**利用检索：

1. 检索k个文档
2. 将每个文档分别与查询拼接，输入LLM
3. 聚合k个输出分布（通过加权平均）
4. 从聚合分布中采样生成

这相当于让模型"投票"决定最佳答案，降低单一文档误导的风险。

**kNN-LM（k-Nearest Neighbor Language Model）**：

在生成每个词时，检索训练数据中上下文最相似的k个样本：

$$P(y|x) = \lambda P_\text{LM}(y|x) + (1-\lambda) P_\text{kNN}(y|x)$$

其中 $P_\text{kNN}$ 是基于检索样本的插值分布。这使得模型可以访问训练时的原始数据。

**Retro（Retrieval-Enhanced Transformer）**：

DeepMind提出的Retro模型将检索整合到Transformer的每一层：

- 在自注意力之前，检索与当前上下文相似的文本块
- 使用**交叉注意力**让模型关注检索到的内容
- 在不同层使用不同时间尺度的检索（短距离 vs 长距离）

Retro用2B参数达到了GPT-3（175B参数）级别的性能，证明了检索增强的参数效率。

### 27.6.4 RAG的挑战与未来方向

**当前挑战**：

1. **检索-生成对齐**：检索器找到的文档可能不是生成器最需要的
2. **长上下文处理**：多文档上下文可能超出模型处理长度
3. **多跳推理**：复杂问题需要连接多个文档的信息
4. **评估困难**：如何自动评估RAG输出的准确性和有用性

**未来方向**：

**自适应检索粒度**：
- 根据问题复杂度自动调整检索文档数量
- 从段落级检索进化到句子级、实体级精确定位

**多模态RAG**：
- 扩展RAG处理图像、视频、音频等多模态知识
- 统一的跨模态检索和生成

**Agentic RAG**：
- 让RAG系统能够主动规划检索策略
- 支持迭代检索、自我纠错、工具调用

**个性化RAG**：
- 根据用户背景和偏好定制检索和生成
- 长期用户记忆整合

RAG技术正在快速发展，它代表了一种新的AI范式：**将记忆（参数知识）与查找（外部检索）相结合**。这与人类认知高度一致——我们不必记住所有知识，但必须知道如何找到它们。

---

## 27.7 练习题

### 基础练习（3题）

**练习27.1 余弦相似度计算**

给定三个文档向量（已归一化）：
- $d_1 = [1, 0, 0, 0]$
- $d_2 = [0.9, 0.1, 0, 0]$  
- $d_3 = [0, 1, 0, 0]$

以及查询向量 $q = [0.95, 0.05, 0, 0]$。

1. 计算查询与每个文档的余弦相似度
2. 如果Top-k=2，应该返回哪些文档？
3. 解释为什么余弦相似度使用方向而非绝对大小

**练习27.2 RAG概率计算**

假设检索器返回两个文档及其相似度分数：
- 文档A：相似度 2.0
- 文档B：相似度 1.0

使用softmax计算（温度τ=1），求：

1. 每个文档被选择的概率
2. 如果温度τ=0.5，概率如何变化？
3. 如果温度τ→0，概率如何变化？这在实际应用中有何意义？

**练习27.3 向量检索分析**

假设知识库有100万个文档，每个文档向量768维。使用暴力搜索（计算查询与所有文档的相似度）：

1. 每次查询需要多少次浮点乘法运算？
2. 如果一台计算机每秒能执行10亿次浮点运算，查询需要多长时间？
3. 讨论为什么实际系统需要使用近似最近邻搜索（ANN）

### 进阶练习（3题）

**练习27.4 RAG-Token与RAG-Sequence比较**

考虑以下场景：回答"深度学习是什么？它在计算机视觉中有什么应用？"

1. 分析RAG-Token可能如何为问题的不同部分检索不同文档
2. 分析RAG-Sequence如何处理这个问题
3. 讨论两种方法各自的优缺点和适用场景
4. 为什么实践中RAG-Sequence通常表现更好？

**练习27.5 实现简化的RAG系统**

基于本章的代码框架，实现以下增强功能：

1. 添加文档分段（chunking）功能，将长文档切分为合适大小的片段
2. 实现重叠分段策略（相邻片段有重叠内容）
3. 添加结果重排序（re-ranking）功能，使用交叉编码器对初步检索结果精排
4. 编写测试用例验证实现

**练习27.6 检索评估指标**

在RAG系统中，我们需要评估检索器的质量。给定一个查询，假设：
- 检索器返回的Top-5文档：[D1, D2, D3, D4, D5]
- 实际上相关的文档集合：{D1, D3, D6, D7}

计算以下指标：
1. Precision@3 和 Precision@5
2. Recall@3 和 Recall@5
3. F1@5
4. 平均精度（Average Precision, AP）

### 挑战练习（3题）

**练习27.7 Self-RAG反射Token设计**

参考Self-RAG论文，设计一个针对中文问答场景的反射Token集合：

1. 确定需要的反射Token类型（如检索决策、相关性评估、支持度评估等）
2. 为每种Token设计5-10个候选Token（中文或英文）
3. 说明每种Token的训练数据标注策略
4. 讨论如何在推理时使用这些Token控制生成行为

**练习27.8 端到端RAG训练分析**

RAG的端到端训练面临检索操作不可微的挑战：

1. 解释为什么Top-k选择操作是不可微的
2. 研究并解释以下解决方案的原理：
   - Gumbel-Softmax技巧
   - 直通估计器（Straight-Through Estimator）
   - REINFORCE策略梯度
3. 比较这些方法的优缺点
4. 设计一个实验验证不同训练策略的效果

**练习27.9 多跳问答RAG系统**

多跳问题需要连接多个文档的信息才能回答，例如："2024年诺贝尔文学奖得主的作品被翻译成多少种语言？"（需要找到得主→找到作品→找到翻译语言数量）

设计一个RAG系统来解决多跳问题：

1. 描述系统架构，包括如何表示和跟踪"中间答案"
2. 设计迭代检索策略，使用上一轮结果改进下一轮查询
3. 实现一个简单的两跳问答示例（如"爱因斯坦获得诺贝尔奖的年份距离他发表狭义相对论多少年？"）
4. 讨论系统面临的挑战和可能的改进方向

---

## 27.8 参考文献

Guu, K., Lee, K., Tung, Z., Pasupat, P., & Chang, M. (2020). REALM: Retrieval-augmented language model pre-training. In *Proceedings of the 37th International Conference on Machine Learning* (pp. 3929-3938). PMLR. https://arxiv.org/abs/2002.08909

Izacard, G., Lewis, P., Lomeli, M., Hosseini, L., Petroni, F., Schick, T., ... & Grave, E. (2022). Atlas: Few-shot learning with retrieval augmented language models. *Journal of Machine Learning Research*, 24(251), 1-43. https://arxiv.org/abs/2208.03299

Izacard, G., & Grave, E. (2021). Leveraging passage retrieval with generative models for open domain question answering. In *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume* (pp. 874-880). https://aclanthology.org/2021.eacl-main.74/

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*, 33, 9459-9474. https://arxiv.org/abs/2005.11401

Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In *The Twelfth International Conference on Learning Representations*. https://arxiv.org/abs/2310.11511

Yan, S. Q., Gu, J. C., Zhu, Y., & Ling, Z. H. (2024). Corrective retrieval augmented generation. *arXiv preprint arXiv:2401.15884*. https://arxiv.org/abs/2401.15884

---

**本章总结**

本章我们深入学习了检索增强生成（RAG）技术。从LLM的幻觉问题出发，我们理解了为什么需要外部知识检索。通过"开卷考试"的比喻，我们领会了参数记忆与非参数记忆结合的价值。

我们详细探讨了向量检索的数学原理，包括余弦相似度、最大内积搜索和Top-k检索的概率解释。在此基础上，我们学习了RAG的完整架构：检索器负责从知识库中找到相关文档，生成器基于这些文档生成答案。

通过约800行代码，我们从零实现了完整的RAG系统，包括VectorStore、Embedder、Retriever和RAGPipeline，并展示了问答系统和文档摘要两个应用示例。

最后，我们了解了前沿的RAG变体：Self-RAG引入自我反思机制，让模型学会按需检索和自我批判；Corrective RAG通过检索评估和纠正机制提升鲁棒性。

RAG代表了AI系统与外部知识交互的新范式，它让大模型从"背诵知识"进化为"查阅知识"，更加接近人类的认知方式。随着技术的不断发展，RAG将在更多领域发挥重要作用。
