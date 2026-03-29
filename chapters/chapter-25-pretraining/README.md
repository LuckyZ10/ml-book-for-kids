# 第二十五章：预训练与微调——站在巨人的肩膀上

> **导读**：想象一下，如果你每学一个新技能都要从认识字母开始，那该多慢啊！幸运的是，我们可以先学习通用的知识（预训练），然后再专注于特定任务（微调）。这一章，我们将探索现代AI最强大的秘密武器——预训练与微调，揭开BERT、GPT等模型背后的魔法！✨

---

## 一、故事的起点：从Word2Vec到BERT

### 1.1 嵌入的困境

还记得我们在第十六章学过的感知机吗？当时我们处理的是一个个独立的词。但是，语言有一个大问题：**同一个词在不同语境下可能有完全不同的意思**。

比如"bank"这个词：
- "I went to the **bank** to deposit money."（银行）
- "The river **bank** was covered with flowers."（河岸）

传统的嵌入（如Word2Vec、GloVe）给每个词分配一个固定的向量，就像给每个人发一张身份证，上面只有一个固定的描述。这显然不够！

### 1.2 预训练的曙光：ELMo（2018）

2018年，Peters等人提出了**ELMo**（Embeddings from Language Models），这是一个革命性的突破！

**核心思想**：词的表示应该依赖于它的上下文。

```
传统方法：bank = [0.3, -0.2, 0.8, ...]  # 固定向量
ELMo方法：bank = f("I went to the bank", 位置5)  # 上下文相关
```

ELMo使用双向LSTM训练语言模型，然后将不同层的隐藏状态组合起来作为词表示。这就像问一个人问题，不仅听他现在的回答，还要参考他之前学过的所有知识。

### 1.3 ULMFiT：预训练+微调的范式（2018）

同年，Howard和Ruder提出了**ULMFiT**（Universal Language Model Fine-tuning），确立了现代NLP的黄金法则：

> **先在大规模无标注数据上预训练，再在小规模标注数据上微调**

这就像：
- 🏫 **预训练** = 在小学、中学、大学学习通用知识（12年+）
- 🎯 **微调** = 参加3个月的职业培训，成为医生/律师/程序员

**ULMFiT的三步策略**：
1. **通用领域预训练**：在Wikipedia等大语料上训练语言模型
2. **目标任务微调**：在特定领域数据上继续训练
3. **分类器微调**：添加分类层，针对具体任务训练

### 1.4 GPT：生成式预训练（2018-2020）

OpenAI的GPT系列采用了不同的路线——**生成式预训练**（Generative Pre-training）：

| 模型 | 年份 | 参数量 | 特点 |
|------|------|--------|------|
| GPT-1 | 2018 | 1.17亿 | 证明Transformer预训练有效 |
| GPT-2 | 2019 | 15亿 | 零样本（Zero-shot）能力 |
| GPT-3 | 2020 | 1750亿 | 少样本（Few-shot）学习 |

GPT的核心是**因果语言模型**（Causal Language Modeling, CLM）：

```
输入："今天天气很"
预测："好"

输入："今天天气很好，我想去"
预测："公园"
```

它只能看到左边的上下文（从左到右），就像写故事时只能回顾已经写过的内容。

### 1.5 BERT：双向编码器的胜利（2018）

Google的BERT（Bidirectional Encoder Representations from Transformers）彻底改变了NLP领域！

**BERT的核心创新**：

#### （1）掩码语言模型（Masked Language Model, MLM）

BERT不是预测下一个词，而是**随机遮住一些词，让模型预测它们**！

```
原句：今天 [MASK] 气很好，我想去公园 [MASK] 步。
目标：     天             散
```

这就像做"完形填空"——你必须理解整句话才能填对空。

**MLM的数学表示**：

给定输入序列 $x = [x_1, x_2, ..., x_n]$，我们随机选择15%的位置进行掩码：

$$\mathcal{L}_{MLM} = -\mathbb{E}_{x \sim \mathcal{D}} \sum_{i \in \mathcal{M}} \log P(x_i | x_{\setminus \mathcal{M}})$$

其中：
- $\mathcal{M}$ 是被掩码的位置集合
- $x_{\setminus \mathcal{M}}$ 是未被掩码的上下文
- $P(x_i | x_{\setminus \mathcal{M}})$ 是模型预测被掩码词的概率

#### （2）下一句预测（Next Sentence Prediction, NSP）

BERT还学习句子间的关系。给定两个句子A和B，判断B是否是A的下一句：

```
正例：
A: 我今天去银行。
B: 存了一些钱。
标签：IsNext

负例：
A: 我今天去银行。
B: 猫在树上睡觉。  ← 随机抽的
标签：NotNext
```

**NSP的数学表示**：

$$\mathcal{L}_{NSP} = -\mathbb{E}_{(A,B) \sim \mathcal{D}} [y \log P(\text{IsNext}) + (1-y) \log P(\text{NotNext})]$$

#### （3）双向编码

BERT的Transformer编码器可以同时看到左右两边的上下文：

```
GPT（单向）：今天 天气 很 → 好  （只能看左边）
BERT（双向）：今天 [MASK] 很 好  （看两边）
               ↑
              预测"天气"
```

---

## 二、预训练的核心概念

### 2.1 什么是预训练？

**预训练**（Pre-training）是指在大规模无标注数据上训练模型，使其学习到通用的语言表示能力。

**费曼法解释**：

> 想象你要教一个小孩子认字。你有两个选择：
> 
> **选择A**：直接给他一本医学教科书，告诉他"把这些术语都背下来，以后当医生用"
> 
> **选择B**：先让他读大量的故事书、报纸、科普文章，学习语言的基本规律。等他掌握了阅读和写作，再让他看医学书。
> 
> 显然，选择B更有效！预训练就是"先读万卷书"。

### 2.2 预训练任务类型

#### （1）语言模型（Language Modeling, LM）

**自回归语言模型（Autoregressive LM）**：

$$P(x) = \prod_{i=1}^{n} P(x_i | x_{<i})$$

每次只预测下一个词，基于前面所有的词。GPT系列使用这种方式。

**掩码语言模型（Masked LM）**：

$$P(x_{\mathcal{M}} | x_{\setminus \mathcal{M}})$$

同时预测多个被掩码的词。BERT使用这种方式。

#### （2）排列语言模型（Permutation LM）

XLNet提出了一种更通用的预训练目标。对于长度为 $n$ 的序列，考虑所有可能的排列：

$$\mathcal{L}_{XLNet} = \mathbb{E}_{z \sim \mathcal{Z}_n} \left[ \sum_{i=1}^{n} \log P(x_{z_i} | x_{z_{<i}}) \right]$$

其中 $z$ 是 $[1, 2, ..., n]$ 的一个排列，$\mathcal{Z}_n$ 是所有排列的集合。

这就像打乱句子的顺序，让模型学会更灵活的依赖关系。

#### （3）对比学习（Contrastive Learning）

SimCSE等模型使用对比学习进行预训练：

$$\mathcal{L}_{contrast} = -\log \frac{\exp(\text{sim}(h_i, h_i^+) / \tau)}{\sum_j \exp(\text{sim}(h_i, h_j) / \tau)}$$

其中：
- $h_i$ 是原始句子的表示
- $h_i^+$ 是同一句子的扰动版本（如dropout）
- $h_j$ 是其他句子的表示
- $\tau$ 是温度参数

**直觉**：让相似的句子靠近，不相似的句子远离。

### 2.3 预训练数据

BERT和GPT-3使用的预训练数据规模：

| 数据集 | BERT | GPT-3 |
|--------|------|-------|
| BooksCorpus | 800M词 | 12B词 |
| Wikipedia (EN) | 2,500M词 | 3B词 |
| WebText | - | 410B词 |
| **总计** | **~3.3B词** | **~500B词** |

**数据预处理**：
1. **Tokenization**：将文本切分成子词（Subword）单元
2. **清理**：去除HTML标签、规范化Unicode
3. **去重**：删除重复文档
4. **过滤**：去除低质量内容

### 2.4 预训练的数学优化

#### 损失函数

BERT的总损失是MLM和NSP的加权和：

$$\mathcal{L}_{BERT} = \mathcal{L}_{MLM} + \mathcal{L}_{NSP}$$

#### 优化器设置

| 超参数 | BERT-Base | BERT-Large |
|--------|-----------|------------|
| 隐藏层维度 | 768 | 1024 |
| 注意力头数 | 12 | 16 |
| Transformer层数 | 12 | 24 |
| 参数量 | 1.1亿 | 3.4亿 |
| 批大小 | 256 | 256 |
| 学习率 | 1e-4 | 1e-4 |
| 训练步数 | 1M | 1M |

#### 学习率预热（Warmup）

BERT使用学习率预热策略：

$$\text{lr}(t) = \text{lr}_{\max} \times \min\left(\frac{t}{t_{warmup}}, \frac{T - t}{T - t_{warmup}}\right)$$

其中 $t_{warmup}$ 通常是总步数的10%（如10,000步）。

这就像运动员热身——先慢慢加速，避免一开始就跑太快受伤。

---

## 三、微调的艺术

### 3.1 什么是微调？

**微调**（Fine-tuning）是指在预训练模型的基础上，针对特定下游任务进行进一步训练。

**费曼法解释**：

> 想象你请了一位名牌大学的毕业生到你公司工作。
> 
> - **预训练**：他已经在大学学了4年的通用知识
> - **微调**：你给他3个月的岗前培训，教他你们公司的具体业务
> 
> 微调就是让"通用人才"变成"专业人才"的过程！

### 3.2 微调的基本方法

#### （1）完整微调（Full Fine-tuning）

所有参数都参与训练：

```python
# 伪代码
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
optimizer = AdamW(model.parameters(), lr=2e-5)

for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

**优点**：效果通常最好
**缺点**：计算量大，容易过拟合

#### （2）冻结微调（Frozen Feature Extractor）

只训练顶部的任务特定层，冻结预训练部分：

```python
# 冻结BERT主体
for param in model.bert.parameters():
    param.requires_grad = False

# 只训练分类层
optimizer = AdamW(model.classifier.parameters(), lr=1e-3)
```

**优点**：训练快，不易过拟合
**缺点**：可能无法充分适应新任务

#### （3）逐层解冻（Gradual Unfreezing）

ULMFiT提出的策略：

1. 先只训练最后一层
2. 收敛后解冻倒数第二层，一起训练
3. 逐步向上解冻，直到所有层都参与训练

这就像教一个人新技能：先让他用已有的能力处理，再逐步开放更多"高级功能"。

#### （4）差分学习率（Discriminative Fine-tuning）

不同层使用不同的学习率：

$$\text{lr}_l = \text{lr}_{base} \times \eta^{L-l}$$

其中 $l$ 是层索引，$L$ 是总层数，$\eta$ 是衰减系数（如0.95）。

**直觉**：底层学习通用特征，学习率小；顶层学习任务特定特征，学习率大。

### 3.3 微调的变体技术

#### （1）提示学习（Prompt Tuning）

不修改模型参数，而是在输入中加入可学习的"提示"（Prompt）：

```
输入：这部电影很好看。 → 情感：[POSITIVE]
      ↓ 添加提示
输入：[PROMPT] [PROMPT] ... [PROMPT] 这部电影很好看。
```

只需要训练少量提示参数（如100-1000个），就能达到接近完整微调的效果。

#### （2）前缀微调（Prefix Tuning）

在每层Transformer的key和value前面添加可学习的前缀：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q[K_{prefix}; K]^T}{\sqrt{d_k}}\right)[V_{prefix}; V]$$

其中 $[;]$ 表示拼接。

#### （3）LoRA：低秩适应（Low-Rank Adaptation）

假设权重更新具有低秩结构：

$$W = W_0 + \Delta W = W_0 + BA$$

其中：
- $W_0$ 是预训练权重（冻结）
- $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times d}$ 是可学习参数
- $r \ll d$ 是低秩（如4、8、16）

**LoRA的优势**：
- 参数量极小（通常<1%）
- 不增加推理延迟（可合并到原权重）
- 效果接近完整微调

#### （4）Adapter层

在Transformer的每个子层后插入小型Adapter模块：

$$h \leftarrow h + f(hW_{down})W_{up}$$

其中 $W_{down} \in \mathbb{R}^{d \times r}$，$W_{up} \in \mathbb{R}^{r \times d}$，$r \ll d$。

**Adapter的优势**：
- 每任务只需存储少量参数
- 可以同时适配多个任务

### 3.4 不同下游任务的微调

#### （1）文本分类

```
输入：[CLS] 这部电影太棒了 [SEP]
输出：池化[CLS] → 全连接层 → Softmax → 类别概率
```

#### （2）句子对分类（如NLI）

```
输入：[CLS] 前提句子 [SEP] 假设句子 [SEP]
输出：池化[CLS] → 全连接层 → 3类（蕴含/矛盾/中立）
```

#### （3）问答（QA）

```
输入：[CLS] 问题 [SEP] 段落 [SEP]
输出：
  - 开始位置概率：P(start=i)
  - 结束位置概率：P(end=j)
答案：argmax P(start=i) × P(end=j)，其中 i ≤ j
```

#### （4）序列标注（如NER）

```
输入：[CLS] 北京 是 中国 的 首都 [SEP]
输出：每个token → 全连接层 → BIO标签
      B-LOC  O  B-LOC O  O
```

### 3.5 微调的实践技巧

#### （1）学习率选择

- **完整微调**：2e-5 到 5e-5（很小！）
- **只训练顶层**：1e-3 到 1e-4
- **LoRA/Adapter**：1e-4 到 1e-3

#### （2）批大小

- 越大越好（如果显存允许）
- BERT通常用16-32
- 梯度累积可以模拟大批量

#### （3）早停（Early Stopping）

在验证集上监控性能，如果连续N个epoch不提升就停止：

```python
best_val_loss = float('inf')
patience = 3
no_improve_count = 0

for epoch in range(max_epochs):
    train(model, train_loader)
    val_loss = evaluate(model, val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)
        no_improve_count = 0
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

#### （4）数据增强

- **回译**（Back-translation）：翻译成其他语言再翻译回来
- **EDA**（Easy Data Augmentation）：同义词替换、随机插入、随机交换、随机删除
- **对抗训练**：添加微小扰动，提高鲁棒性

---

## 四、预训练模型的家族

### 4.1 编码器模型（Encoder-only）

这些模型都是BERT的后代，专注于理解任务：

| 模型 | 特点 | 适用任务 |
|------|------|----------|
| **BERT** | 双向Transformer，MLM+NSP | 分类、NER、问答 |
| **RoBERTa** | 去除NSP，更大batch，更多数据 | 分类、NER、问答 |
| **ALBERT** | 参数共享，矩阵分解，更高效 | 资源受限场景 |
| **DistilBERT** | 知识蒸馏，轻量版BERT | 移动端、边缘设备 |
| **ELECTRA** | 替换token检测，样本效率更高 | 分类、NER |

### 4.2 解码器模型（Decoder-only）

这些模型是GPT的后代，专注于生成任务：

| 模型 | 年份 | 参数量 | 特点 |
|------|------|--------|------|
| **GPT-1** | 2018 | 1.17亿 | 证明Transformer预训练有效 |
| **GPT-2** | 2019 | 15亿 | 零样本能力 |
| **GPT-3** | 2020 | 1750亿 | 少样本学习，上下文学习 |
| **GPT-4** | 2023 | 未公开 | 多模态，推理能力大幅提升 |
| **LLaMA** | 2023 | 7B-65B | Meta开源，高效训练 |
| **Claude** | 2023 | 未公开 | Anthropic，安全对齐 |

### 4.3 编码器-解码器模型（Encoder-Decoder）

| 模型 | 特点 | 适用任务 |
|------|------|----------|
| **T5** | 所有任务统一为text-to-text | 翻译、摘要、问答 |
| **BART** | BERT的编码器+GPT的解码器 | 生成式任务 |
| **mT5** | T5的多语言版本 | 跨语言任务 |

### 4.4 模型选择的决策树

```
你的任务是什么？
├── 理解任务（分类、NER、相似度）
│   └── 选择编码器模型 → BERT/RoBERTa/ALBERT
├── 生成任务（写作、对话、翻译）
│   └── 选择解码器/编解码器模型 → GPT/T5/BART
└── 资源受限？
    ├── 是 → DistilBERT/MobileBERT/TinyBERT
    └── 否 → 选择性能最好的大模型
```

---

## 五、从零实现预训练与微调

现在让我们亲手实现一个简化版的预训练和微调流程！

### 5.1 预训练：掩码语言模型

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class MaskedLMModel(nn.Module):
    """
    简化版BERT风格的掩码语言模型
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, 
                 dim_feedforward=1024, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 嵌入 + 位置编码
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层：预测被掩码的词
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids, attention_mask=None):
        """
        参数:
            input_ids: [batch_size, seq_len]，部分词被[MASK]替换
            attention_mask: [batch_size, seq_len]，1表示有效位置
        返回:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # 位置编码
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        
        # 嵌入 = 嵌入 + 位置编码
        token_embed = self.token_embedding(input_ids)
        pos_embed = self.position_embedding(positions)
        x = self.dropout(token_embed + pos_embed)
        
        # Transformer编码
        if attention_mask is not None:
            # 转换为Transformer需要的格式（True表示被掩码）
            mask = (attention_mask == 0)
        else:
            mask = None
        
        hidden = self.transformer(x, src_key_padding_mask=mask)
        
        # 预测每个位置的词
        logits = self.output_layer(hidden)
        
        return logits
    
    def get_embeddings(self, input_ids, attention_mask=None):
        """获取句子的表示（用于下游任务）"""
        logits = self.forward(input_ids, attention_mask)
        # 取[CLS]位置（第一个token）的表示
        return logits[:, 0, :]


class MaskedLMDataset(Dataset):
    """掩码语言模型数据集"""
    
    def __init__(self, texts, tokenizer, max_length=128, mask_prob=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.vocab_size = len(tokenizer)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 分词
        tokens = self.tokenizer.encode(text, max_length=self.max_length, 
                                       padding='max_length', truncation=True)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # 创建标签（复制原始token）
        labels = input_ids.clone()
        
        # 随机掩码
        masked_input_ids = input_ids.clone()
        rand = torch.rand(input_ids.shape)
        
        # 只掩码非padding位置
        mask_candidates = (input_ids != self.tokenizer.pad_token_id) & (rand < self.mask_prob)
        
        for i in range(len(masked_input_ids)):
            if mask_candidates[i]:
                rand_val = torch.rand(1).item()
                if rand_val < 0.8:
                    # 80%概率替换为[MASK]
                    masked_input_ids[i] = self.tokenizer.mask_token_id
                elif rand_val < 0.9:
                    # 10%概率替换为随机词
                    masked_input_ids[i] = torch.randint(0, self.vocab_size, (1,)).item()
                # 10%概率保持不变
        
        # 只计算被掩码位置的损失
        labels[~mask_candidates] = -100  # PyTorch忽略-100的标签
        
        # 注意力掩码
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            'input_ids': masked_input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def pretrain_mlm(model, dataloader, epochs=3, lr=1e-4, device='cuda'):
    """预训练掩码语言模型"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            logits = model(input_ids, attention_mask)
            
            # 计算损失
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    return model


# ============== 简单分词器 ==============

class SimpleTokenizer:
    """简化版分词器（基于字符级）"""
    
    def __init__(self):
        # 特殊token
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.mask_token = '<MASK>'
        self.cls_token = '<CLS>'
        self.sep_token = '<SEP>'
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.mask_token_id = 2
        self.cls_token_id = 3
        self.sep_token_id = 4
        
        self.special_tokens = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.mask_token: self.mask_token_id,
            self.cls_token: self.cls_token_id,
            self.sep_token: self.sep_token_id
        }
        
        self.token2id = {**self.special_tokens}
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.next_id = len(self.special_tokens)
    
    def build_vocab(self, texts, min_freq=2):
        """从文本构建词汇表"""
        from collections import Counter
        char_counter = Counter()
        
        for text in texts:
            for char in text:
                char_counter[char] += 1
        
        for char, freq in char_counter.items():
            if freq >= min_freq and char not in self.token2id:
                self.token2id[char] = self.next_id
                self.id2token[self.next_id] = char
                self.next_id += 1
        
        print(f"词汇表大小: {len(self.token2id)}")
    
    def encode(self, text, max_length=128, padding='max_length', truncation=True):
        """编码文本"""
        tokens = [self.cls_token_id]
        
        for char in text[:max_length-2] if truncation else text:
            tokens.append(self.token2id.get(char, self.unk_token_id))
        
        tokens.append(self.sep_token_id)
        
        # Padding
        if padding == 'max_length':
            while len(tokens) < max_length:
                tokens.append(self.pad_token_id)
        
        return tokens[:max_length]
    
    def decode(self, token_ids, skip_special_tokens=True):
        """解码token序列"""
        chars = []
        for idx in token_ids:
            if idx in [self.pad_token_id, self.cls_token_id, self.sep_token_id]:
                if not skip_special_tokens:
                    chars.append(self.id2token.get(idx, self.unk_token))
            else:
                chars.append(self.id2token.get(idx, self.unk_token))
        return ''.join(chars)
    
    def __len__(self):
        return len(self.token2id)


# ============== 演示 ==============
if __name__ == "__main__":
    # 示例语料（实际应用中使用更大规模的语料）
    corpus = [
        "机器学习是人工智能的一个重要分支",
        "深度学习使用神经网络进行特征学习",
        "自然语言处理让计算机理解人类语言",
        "计算机视觉使机器能够看懂图像",
        "强化学习通过与环境交互来学习策略",
        "预训练模型在大规模数据上学习通用知识",
        "微调使预训练模型适应特定任务",
        "注意力机制帮助模型关注重要信息",
        "Transformer架构彻底改变了自然语言处理",
        "BERT使用双向编码器进行语言理解"
    ] * 100  # 重复100次增加数据量
    
    print("=" * 60)
    print("步骤1: 构建词汇表")
    print("=" * 60)
    
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(corpus, min_freq=1)
    
    print("\n" + "=" * 60)
    print("步骤2: 创建数据集和数据加载器")
    print("=" * 60)
    
    dataset = MaskedLMDataset(corpus, tokenizer, max_length=32, mask_prob=0.15)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 展示一个样本
    sample = dataset[0]
    print(f"\n样本示例:")
    print(f"原始编码: {tokenizer.decode(dataset.texts[0])}")
    print(f"掩码后:   {tokenizer.decode(sample['input_ids'].tolist())}")
    print(f"标签:     {tokenizer.decode(sample['labels'].tolist())}")
    
    print("\n" + "=" * 60)
    print("步骤3: 创建模型")
    print("=" * 60)
    
    model = MaskedLMModel(
        vocab_size=len(tokenizer),
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        max_seq_len=32
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    print("\n" + "=" * 60)
    print("步骤4: 预训练（演示用，仅训练1个epoch）")
    print("=" * 60)
    
    model = pretrain_mlm(model, dataloader, epochs=1, lr=1e-3, device=device)
    
    print("\n预训练完成！")
```

### 5.2 微调：情感分类

```python
class TextClassifier(nn.Module):
    """
    基于预训练模型的文本分类器
    """
    def __init__(self, pretrained_model, num_classes, freeze_pretrained=False):
        super().__init__()
        self.pretrained = pretrained_model
        
        # 是否冻结预训练部分
        if freeze_pretrained:
            for param in self.pretrained.parameters():
                param.requires_grad = False
        
        d_model = pretrained_model.d_model
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask=None):
        """
        返回:
            logits: [batch_size, num_classes]
        """
        # 获取[CLS]位置的表示
        logits = self.pretrained(input_ids, attention_mask)  # [batch, seq, vocab]
        cls_output = logits[:, 0, :]  # [CLS]位置的向量
        
        # 分类
        logits = self.classifier(cls_output)
        return logits


class ClassificationDataset(Dataset):
    """分类任务数据集"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 编码（不需要掩码）
        tokens = self.tokenizer.encode(text, max_length=self.max_length, 
                                       padding='max_length', truncation=True)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


def fine_tune_classifier(model, dataloader, epochs=5, lr=2e-5, device='cuda'):
    """
    微调分类器
    
    参数:
        lr: 学习率（微调时通常很小，2e-5到5e-5）
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                   lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
    
    return model


def predict(model, text, tokenizer, device='cuda'):
    """预测单条文本的情感"""
    model.eval()
    
    tokens = tokenizer.encode(text, max_length=128, padding='max_length', truncation=True)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)
        prediction = logits.argmax(dim=-1).item()
    
    return prediction, probs[0].cpu().numpy()


# ============== 演示 ==============
if __name__ == "__main__":
    # 假设我们已经有了预训练好的model
    
    print("=" * 60)
    print("步骤1: 准备分类数据")
    print("=" * 60)
    
    # 示例分类数据（情感分析）
    train_texts = [
        "这部电影太精彩了", "演员表演出色", "剧情引人入胜",  # 正面
        "完全看不懂", "浪费时间的烂片", "演技太差了"          # 负面
    ] * 50
    
    train_labels = [1, 1, 1, 0, 0, 0] * 50  # 1=正面, 0=负面
    
    classifier_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, max_length=32)
    classifier_loader = DataLoader(classifier_dataset, batch_size=8, shuffle=True)
    
    print(f"训练样本数: {len(train_texts)}")
    print(f"类别分布: 正面={train_labels.count(1)}, 负面={train_labels.count(0)}")
    
    print("\n" + "=" * 60)
    print("步骤2: 创建分类器（使用预训练权重）")
    print("=" * 60)
    
    # 创建分类器，传入预训练好的模型
    classifier = TextClassifier(model, num_classes=2, freeze_pretrained=False)
    
    print("\n参数统计:")
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"冻结比例: {(1 - trainable_params/total_params)*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("步骤3: 微调分类器")
    print("=" * 60)
    
    classifier = fine_tune_classifier(classifier, classifier_loader, epochs=5, 
                                      lr=2e-4, device=device)
    
    print("\n" + "=" * 60)
    print("步骤4: 测试预测")
    print("=" * 60)
    
    test_texts = [
        "非常好看的电影",
        "令人失望的作品",
        "演员演得很棒"
    ]
    
    for text in test_texts:
        pred, probs = predict(classifier, text, tokenizer, device)
        sentiment = "正面" if pred == 1 else "负面"
        confidence = probs[pred]
        print(f"\n文本: {text}")
        print(f"预测: {sentiment} (置信度: {confidence:.2%})")
        print(f"概率分布: 负面={probs[0]:.2%}, 正面={probs[1]:.2%}")
```

### 5.3 使用Hugging Face Transformers库

在实际应用中，我们通常会使用成熟的库如`transformers`：

```python
# 安装: pip install transformers datasets

from transformers import (
    BertTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ============== 1. 加载预训练模型和分词器 ==============

model_name = "bert-base-chinese"  # 或 "bert-base-uncased" 用于英文
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2  # 二分类
)

print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ============== 2. 准备数据 ==============

# 加载GLUE的SST-2情感分析数据集（英文）
dataset = load_dataset("glue", "sst2")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ============== 3. 数据整理器 ==============

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ============== 4. 评估指标 ==============

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# ============== 5. 训练参数 ==============

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,              # 微调学习率很小
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    logging_dir="./logs",
    logging_steps=100,
    warmup_ratio=0.1,                # 学习率预热
)

# ============== 6. 创建Trainer ==============

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ============== 7. 训练 ==============

trainer.train()

# ============== 8. 评估 ==============

results = trainer.evaluate()
print(f"验证集结果: {results}")

# ============== 9. 保存模型 ==============

trainer.save_model("./my_fine_tuned_model")

# ============== 10. 推理 ==============

from transformers import pipeline

# 创建分类pipeline
classifier = pipeline("sentiment-analysis", model="./my_fine_tuned_model")

# 测试
print(classifier("This movie is absolutely fantastic!"))
print(classifier("I really hated this film."))
```

### 5.4 使用LoRA进行高效微调

```python
# 安装: pip install peft

from peft import LoraConfig, get_peft_model, TaskType

# ============== 配置LoRA ==============

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,     # 序列分类任务
    r=8,                             # 低秩
    lora_alpha=32,                   # 缩放参数
    lora_dropout=0.1,
    bias="none",
    target_modules=["query", "key", "value", "dense"]  # 应用LoRA的层
)

# ============== 创建LoRA模型 ==============

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# 添加LoRA
model = get_peft_model(model, lora_config)

# 查看可训练参数
model.print_trainable_parameters()
# 输出类似: trainable params: 296,450 || all params: 109,489,410 || trainable%: 0.2707

# ============== 训练 ==============

training_args = TrainingArguments(
    output_dir="./lora_results",
    learning_rate=1e-3,              # LoRA可以用更大的学习率
    per_device_train_batch_size=16,
    num_train_epochs=3,
    # ... 其他参数
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    # ...
)

trainer.train()

# ============== 保存和加载 ==============

# 只保存LoRA参数（很小！）
model.save_pretrained("./lora_adapter")

# 加载时
from peft import PeftModel

base_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = PeftModel.from_pretrained(base_model, "./lora_adapter")
```

---

## 六、预训练与微调的理论分析

### 6.1 为什么预训练有效？

**1. 迁移学习视角**

预训练模型的知识可以表示为学到的特征提取函数 $f_{pre}$。微调时，我们学习新的分类器 $g$：

$$y = g(f_{pre}(x))$$

如果 $f_{pre}$ 提取的特征与下游任务相关，那么只需要很少的数据就能训练好 $g$。

**2. 表示学习视角**

预训练学习到的是数据的**层次化表示**：

- **底层**：语法、词法特征（词性、句法结构）
- **中层**：语义特征（词义、指代关系）
- **高层**：任务相关特征（情感、意图）

微调只需调整顶层，底层特征可以直接复用。

**3. 优化视角**

预训练提供了一个好的**初始化点**：

$$\theta_{init} = \theta_{pretrain} \text{ vs. } \theta_{random}$$

好的初始化使得：
- 收敛更快（需要的epoch更少）
- 收敛到更好的局部最优
- 更稳定的训练过程

### 6.2 微调的学习动态

** catastrophic forgetting（灾难性遗忘）**

微调时，模型可能"忘记"预训练学到的通用知识：

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \mathcal{L}_{pretrain}$$

解决方案：
- 使用小的学习率
- 使用正则化（如EWC：Elastic Weight Consolidation）
- 使用Adapter/LoRA等参数高效方法

** layer-wise learning rate decay**

不同层使用不同的学习率：

$$\text{lr}_l = \text{lr}_{base} \times \gamma^{L-l}$$

通常 $\gamma \in [0.9, 0.95]$。

### 6.3 预训练规模与性能

OpenAI的研究揭示了惊人的规律——**规模定律**（Scaling Laws）：

$$L(N) \propto N^{-\alpha}$$

其中：
- $L$ 是损失
- $N$ 是模型参数量
- $\alpha \approx 0.07$ 对于语言模型

这意味着：
- 模型越大，性能越好（但边际收益递减）
- 数据量也需要相应增加
- 计算量与模型大小和数据的乘积成正比

| 模型 | 参数量 | 训练数据量 | 计算量 (PF-days) |
|------|--------|------------|------------------|
| GPT-1 | 1.17亿 | 5GB | 0.02 |
| GPT-2 | 15亿 | 40GB | 0.3 |
| GPT-3 | 1750亿 | 570GB | 3,640 |

---

## 七、前沿发展

### 7.1 指令微调（Instruction Tuning）

不仅仅是针对特定任务微调，而是让模型学会**遵循指令**：

```
输入：
请将以下中文翻译成英文：
机器学习是人工智能的一个分支。

输出：
Machine learning is a branch of artificial intelligence.
```

**代表性工作**：FLAN、InstructGPT、Alpaca

### 7.2 基于人类反馈的强化学习（RLHF）

结合强化学习让模型输出更符合人类偏好：

1. **收集人类偏好数据**：对同一输入的多个输出进行排序
2. **训练奖励模型**：学习人类偏好
3. **使用PPO优化**：最大化奖励

$$\mathcal{L}_{RLHF} = \mathbb{E}_{(x,y) \sim \pi_{\theta}} [R(x,y)] - \beta \mathbb{D}_{KL}(\pi_{\theta} || \pi_{ref})$$

### 7.3 多模态预训练

将预训练扩展到图像、音频、视频：

| 模型 | 模态 | 特点 |
|------|------|------|
| CLIP | 图像+文本 | 对比学习对齐视觉-语言 |
| DALL-E | 文本→图像 | 生成式建模 |
| GPT-4V | 图像+文本 | 多模态理解 |
| Whisper | 音频→文本 | 语音识别 |

---

## 八、练习与挑战

### 基础练习

**练习1**：理解掩码策略

BERT的掩码策略中，为什么80%替换为[MASK]，10%替换为随机词，10%保持不变？如果100%都替换为[MASK]会有什么后果？

**练习2**：计算微调参数量

假设使用BERT-Base（隐藏维度768，12层）进行文本分类：
1. 完整微调需要更新多少参数？
2. 如果只训练最后的分类层（输入768，输出2）需要多少参数？
3. 使用LoRA（r=8）需要多少参数？

**练习3**：学习率调度

解释为什么微调时学习率要比预训练小得多。如果在微调时使用lr=1e-3（预训练的学习率）会发生什么？

### 进阶练习

**练习4**：实现NSP任务

在预训练代码的基础上，添加下一句预测（NSP）任务。需要：
1. 修改数据集，生成50%连续的句子对和50%随机句子对
2. 添加NSP输出头
3. 修改损失函数为MLM + NSP

**练习5**：不同微调策略对比

实现并比较以下微调策略在情感分类任务上的表现：
- 完整微调
- 只微调顶层
- 逐层解冻
- LoRA (r=4, 8, 16)

记录每种方法的：
- 收敛速度（达到90%准确率需要的epoch）
- 最终准确率
- 训练时间和内存占用

**练习6**：灾难性遗忘实验

设计实验验证灾难性遗忘现象：
1. 先在任务A（如情感分析）上预训练/微调
2. 然后在任务B（如主题分类）上微调
3. 测试模型在任务A上的性能是否下降
4. 尝试使用EWC或其他正则化方法缓解遗忘

### 挑战题目

**挑战1**：实现一个简单的GPT模型

基于本章的Transformer代码，实现一个GPT风格的自回归语言模型（只使用解码器，因果掩码）。训练它在给定前文的情况下预测下一个字符。

**挑战2**：多任务学习

扩展分类器，使其能够同时处理多个任务（如情感分析、主题分类、语言识别），通过任务特定的提示或标签来区分不同任务。

**挑战3**：探索Prompt Tuning

实现Prompt Tuning：
1. 冻结整个预训练模型
2. 在输入前添加可学习的虚拟token
3. 只训练这些虚拟token的嵌入
4. 与LoRA的效果进行对比

---

## 九、本章小结

### 核心概念回顾

1. **预训练**（Pre-training）：在大规模无标注数据上学习通用表示
2. **微调**（Fine-tuning）：在特定任务数据上调整模型
3. **MLM**（Masked Language Model）：BERT的预训练目标，完形填空
4. **CLM**（Causal Language Model）：GPT的预训练目标，自回归预测
5. **参数高效微调**：LoRA、Adapter、Prompt Tuning等

### 重要公式

**BERT总损失**：
$$\mathcal{L}_{BERT} = \mathcal{L}_{MLM} + \mathcal{L}_{NSP}$$

**LoRA低秩适应**：
$$W = W_0 + BA$$

**对比学习损失**：
$$\mathcal{L} = -\log \frac{\exp(\text{sim}(h_i, h_i^+) / \tau)}{\sum_j \exp(\text{sim}(h_i, h_j) / \tau)}$$

### 关键时间线

```
2013 - Word2Vec: 静态嵌入
2017 - Transformer: "Attention Is All You Need"
2018 - ELMo: 上下文相关词表示
2018 - ULMFiT: 预训练+微调范式确立
2018 - GPT-1: 生成式预训练
2018 - BERT: 双向编码器，NLP新时代
2019 - GPT-2: 零样本能力
2019 - RoBERTa: BERT优化版
2019 - DistilBERT: 知识蒸馏
2020 - GPT-3: 1750亿参数，少样本学习
2021 - LoRA: 参数高效微调
2022 - InstructGPT: RLHF
2023 - GPT-4: 多模态，推理能力飞跃
```

### 进一步阅读

**经典论文**：
1. Devlin et al. (2018) - BERT
2. Radford et al. (2018) - GPT-1
3. Radford et al. (2019) - GPT-2
4. Brown et al. (2020) - GPT-3
5. Liu et al. (2019) - RoBERTa
6. Hu et al. (2021) - LoRA

**推荐资源**：
- Hugging Face Transformers文档
- "Natural Language Processing with Transformers"（书籍）
- Stanford CS224N课程

---

## 参考文献

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT*, 4171-4186. https://doi.org/10.18653/v1/N19-1423

Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. *Proceedings of ACL*, 328-339. https://doi.org/10.18653/v1/P18-1031

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pre-training approach. *arXiv preprint arXiv:1907.11692*.

Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). Deep contextualized word representations. *Proceedings of NAACL-HLT*, 2227-2237.

Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training. *OpenAI Technical Report*.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI Blog*, 1(8), 9.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877-1901.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*.

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.

Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., & Soricut, R. (2019). ALBERT: A lite BERT for self-supervised learning of language representations. *arXiv preprint arXiv:1909.11942*.

---

*本章完。恭喜你完成了第25章的学习！下一章我们将探索更强大的大语言模型（LLM）及其应用。* 🚀
