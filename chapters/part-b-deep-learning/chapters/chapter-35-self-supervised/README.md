# 第三十五章：自监督学习前沿 (Self-Supervised Learning Frontier)

## 章节引言

想象你正在学习一门新语言。老师给了你一本没有答案的练习册，里面全是填空题。你只能靠上下文猜测每个空格应该填什么词，然后对照词典检查自己的答案。令人惊讶的是，通过这种"自我测试"的方式，你的语言能力竟然突飞猛进！

这就是**自监督学习(Self-Supervised Learning, SSL)**的核心理念——让模型自己给自己出题目，自己给自己打分，从海量无标注数据中学习有用的表示。本章将带领你探索这个机器学习领域最激动人心的前沿，理解BERT、GPT、CLIP等大模型背后的预训练秘密。

---

## 35.1 什么是自监督学习？

### 35.1.1 三种学习范式的对比

在机器学习中，我们有三种主要的学习方式，就像学生有三种不同的学习模式：

**监督学习(Supervised Learning)**：就像有老师手把手教你。老师给你大量的练习题，每道题都有标准答案。你通过对比自己的答案和正确答案来学习。
- 优点：目标明确，学习效率高
- 缺点：需要大量标注数据，标注成本高
- 例子：图像分类、情感分析

**无监督学习(Unsupervised Learning)**：就像没有老师，你自己在图书馆里翻阅资料，试图发现其中的规律。
- 优点：不需要标注数据
- 缺点：学习目标不明确，难以评估
- 例子：聚类分析、降维

**自监督学习(Self-Supervised Learning)**：这是一种特殊的"无监督学习"，但巧妙地设计了"预文本任务(Pretext Task)"。就像让学生做填空练习，虽然没人告诉他正确答案，但正确答案其实就隐藏在上下文里！
- 优点：利用数据本身的结构生成监督信号，不需要人工标注
- 缺点：需要设计巧妙的预文本任务
- 例子：BERT的MLM、GPT的自回归预测

### 35.1.2 自监督学习的核心思想

**生活化比喻：填空练习**

想象你正在做一个完形填空练习：

> "今天天气很___，我决定去公园___。"

虽然没有老师告诉你空格的正确答案，但你可以根据上下文推断：
- 第一个空格可能是"好"、"晴朗"、"糟糕"等
- 第二个空格可能是"散步"、"跑步"、"野餐"等

自监督学习正是这样——我们从数据本身创造"填空题"，让模型学习预测被隐藏的部分。

**数学视角**

自监督学习的核心可以用以下框架描述：

$$\mathcal{L}_{\text{self}} = \mathbb{E}_{x \sim \mathcal{D}} \left[ \ell(x, f(g(x))) \right]$$

其中：
- $g(x)$ 是数据变换函数（如遮罩、裁剪、旋转）
- $f$ 是模型
- $\ell$ 是损失函数，衡量模型预测与原始数据的差异

### 35.1.3 为什么自监督学习如此重要？

1. **数据效率**：互联网上有海量的无标注数据（图片、文本、视频）
2. **表示学习**：预训练得到的表示可以迁移到各种下游任务
3. **大模型的基石**：GPT、BERT、CLIP等革命性模型都依赖自监督预训练

---

## 35.2 NLP中的自监督：从Word2Vec到BERT

### 35.2.1 回顾：Word2Vec的CBOW与Skip-gram

还记得我们在第二十四章学习过的Word2Vec吗？它是最早成功的自监督学习方法之一！

**CBOW（连续词袋模型）**：
> 给定周围的词，预测中间的词
> 
> 例如："我 ___ 北京" → 预测"爱"

**Skip-gram**：
> 给定中间的词，预测周围的词
> 
> 例如：给定"爱"，预测"我"和"北京"

这两种方法都是自监督的——不需要人工标注，直接从文本的结构中学习！

### 35.2.2 BERT：遮蔽语言模型(MLM)的革命

**论文背景**

BERT（Bidirectional Encoder Representations from Transformers）由Google在2018年提出（Devlin et al., 2019），彻底改变了NLP领域。

**生活化比喻：完形填空大师**

BERT就像一个完形填空的大师。它被训练来完成这样的任务：

> "我今天去[MASK]市图书馆[MASK]书。"

BERT需要同时考虑左边和右边的上下文来预测被遮罩的词：
- 第一个[MASK]：可能是"图"、"图"是正确的前一个字
- 第二个[MASK]：可能是"读"、"借"、"看"等

**MLM的数学原理**

1. **输入处理**：对于输入序列$x = (x_1, x_2, ..., x_n)$，随机选择15%的token进行遮罩

2. **遮罩策略**（这是BERT的关键技巧）：
   - 80%的概率替换为[MASK] token
   - 10%的概率替换为随机token
   - 10%的概率保持不变

   这种混合策略防止了预训练和微调之间的不匹配。

3. **损失函数**：

$$\mathcal{L}_{\text{MLM}} = -\mathbb{E}_{x \sim \mathcal{D}} \sum_{i \in \mathcal{M}} \log P(x_i | x_{\backslash \mathcal{M}})$$

其中：
- $\mathcal{M}$是被遮罩的位置集合
- $x_{\backslash \mathcal{M}}$表示未被遮罩的token
- $P(x_i | \cdot)$是模型预测位置$i$的词的概率

**完整推导**

设Transformer编码器为$f_\theta$，输出为隐藏状态$h_i$：

$$h = f_\theta(x_{\backslash \mathcal{M}})$$

对每个被遮罩位置$i$，计算词汇表上的分布：

$$P(x_i = w | x_{\backslash \mathcal{M}}) = \frac{\exp(W_w^T h_i + b_w)}{\sum_{w'} \exp(W_{w'}^T h_i + b_{w'})}$$

其中$W$和$b$是输出层的参数。

**BERT的架构细节**

- **BERT-Base**: 12层，768维，12个注意力头，1.1亿参数
- **BERT-Large**: 24层，1024维，16个注意力头，3.4亿参数

### 35.2.3 GPT：自回归语言建模

**论文背景**

GPT（Generative Pre-training）由OpenAI在2018年提出（Radford et al., 2018），采用自回归方式训练。

**生活化比喻：接龙游戏**

GPT就像一个玩词语接龙的高手。游戏规则是：
> 给定前面的词，预测下一个词是什么

例如：
```
输入："我今天"
预测："去"

输入："我今天去"
预测："图书馆"

输入："我今天去图书馆"
预测："借书"
```

**自回归建模的数学原理**

对于序列$x = (x_1, x_2, ..., x_n)$，自回归模型建模条件概率：

$$P(x) = \prod_{i=1}^{n} P(x_i | x_{<i})$$

其中$x_{<i} = (x_1, ..., x_{i-1})$表示位置$i$之前的所有token。

**损失函数**

$$\mathcal{L}_{\text{AR}} = -\frac{1}{n} \sum_{i=1}^{n} \log P(x_i | x_{<i}; \theta)$$

**GPT vs BERT的关键区别**

| 特性 | GPT | BERT |
|------|-----|------|
| 方向 | 单向（左→右） | 双向 |
| 预训练任务 | 下一个token预测 | 遮罩token预测 |
| 架构 | 仅Decoder | 仅Encoder |
| 适用任务 | 生成任务 | 理解任务 |
| 典型应用 | 文本生成、对话 | 分类、NER、问答 |

### 35.2.4 代码实现：MaskedLM预训练

```python
"""
Masked Language Modeling (MLM) 预训练实现
模拟BERT的核心预训练机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TokenEmbedding(nn.Module):
    """嵌入层"""
    def __init__(self, vocab_size, d_model, max_len=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.position_embed = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        
        tok_emb = self.token_embed(x)
        pos_emb = self.position_embed(positions)
        
        return self.dropout(self.norm(tok_emb + pos_emb))

class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # 线性变换
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(context)

class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_out)
        
        # 前馈网络 + 残差连接
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

class MaskedLM(nn.Module):
    """
    遮蔽语言模型实现
    类似BERT的预训练架构
    """
    def __init__(self, vocab_size, d_model=256, n_layers=6, 
                 n_heads=8, d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 嵌入层
        self.embedding = TokenEmbedding(vocab_size, d_model, max_len)
        
        # Transformer编码器
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出层：预测被遮罩的词
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # 特殊token
        self.mask_token_id = vocab_size - 1  # 假设最后一个id是[MASK]
        
    def forward(self, x, mask_positions=None):
        """
        前向传播
        
        Args:
            x: 输入token ids [batch_size, seq_len]
            mask_positions: 遮罩位置 [batch_size, num_masked]
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        x = self.embedding(x)
        
        # 创建padding mask（假设0是padding）
        padding_mask = (x.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(1)
        
        for layer in self.encoder_layers:
            x = layer(x, padding_mask)
        
        logits = self.output_layer(x)
        return logits
    
    def create_masked_input(self, input_ids, mask_prob=0.15):
        """
        创建MLM的遮罩输入
        遵循BERT的遮罩策略
        
        Returns:
            masked_input: 遮罩后的输入
            labels: 原始标签（用于计算损失）
            mask_positions: 遮罩位置
        """
        batch_size, seq_len = input_ids.shape
        labels = input_ids.clone()
        masked_input = input_ids.clone()
        
        # 随机选择15%的位置
        rand = torch.rand(batch_size, seq_len)
        mask_positions = (rand < mask_prob) & (input_ids != 0)  # 不遮罩padding
        
        for i in range(batch_size):
            positions = mask_positions[i].nonzero(as_tuple=True)[0]
            for pos in positions:
                rand_val = torch.rand(1).item()
                if rand_val < 0.8:
                    # 80%替换为[MASK]
                    masked_input[i, pos] = self.mask_token_id
                elif rand_val < 0.9:
                    # 10%替换为随机词
                    masked_input[i, pos] = torch.randint(1, self.vocab_size - 1, (1,))
                # 10%保持不变
        
        # 只在遮罩位置计算损失
        labels[~mask_positions] = -100  # 忽略非遮罩位置的损失
        
        return masked_input, labels, mask_positions
    
    def compute_loss(self, input_ids):
        """计算MLM损失"""
        masked_input, labels, _ = self.create_masked_input(input_ids)
        
        logits = self.forward(masked_input)
        
        # 展平以计算交叉熵损失
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        
        return loss

class AutoregressiveLM(nn.Module):
    """
    自回归语言模型实现
    类似GPT的预训练架构
    """
    def __init__(self, vocab_size, d_model=256, n_layers=6,
                 n_heads=8, d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = TokenEmbedding(vocab_size, d_model, max_len)
        
        # Transformer解码器层（带因果mask）
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def create_causal_mask(self, seq_len, device):
        """创建因果mask（上三角为0）"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        x = self.embedding(x)
        
        # 因果mask
        causal_mask = self.create_causal_mask(seq_len, x.device)
        
        for layer in self.decoder_layers:
            x = layer(x, causal_mask)
        
        return self.output_layer(x)
    
    def compute_loss(self, input_ids):
        """计算自回归语言建模损失"""
        # 输入：x_1, x_2, ..., x_{n-1}
        # 目标：x_2, x_3, ..., x_n
        input_seq = input_ids[:, :-1]
        target_seq = input_ids[:, 1:]
        
        logits = self.forward(input_seq)
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
        loss = loss_fct(logits.reshape(-1, self.vocab_size), target_seq.reshape(-1))
        
        return loss
    
    @torch.no_grad()
    def generate(self, prompt, max_length=50, temperature=1.0, top_k=None):
        """自回归生成"""
        self.eval()
        generated = prompt.clone()
        
        for _ in range(max_length):
            logits = self.forward(generated)[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            if next_token.item() == 2:  # 假设2是EOS token
                break
        
        return generated

# 训练示例
def train_mlm_example():
    """MLM训练示例"""
    vocab_size = 10000
    batch_size = 32
    seq_len = 128
    
    model = MaskedLM(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 模拟数据
    dummy_data = torch.randint(1, vocab_size - 1, (batch_size, seq_len))
    
    for epoch in range(10):
        optimizer.zero_grad()
        loss = model.compute_loss(dummy_data)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_mlm_example()
```

---

## 35.3 计算机视觉中的对比学习

### 35.3.1 核心思想：找相同与找不同

**生活化比喻：视觉记忆游戏**

想象你在玩一个记忆游戏。老师给你看两张图片：
- 图片A：一只金毛犬在草地上奔跑
- 图片B：同一只金毛犬在沙滩上坐下

虽然背景不同、姿势不同，但你知道这两张图片展示的是"同一只狗"。

对比学习就是让模型学会这样的能力：
- **拉近(Align)**：同一张图片的不同变换版本（正样本）
- **推开(Uniform)**：不同图片（负样本）

### 35.3.2 SimCLR：简单对比学习框架

**论文背景**

SimCLR（Simple Framework for Contrastive Learning of Visual Representations）由Google在2020年提出（Chen et al., 2020），是一个简洁而强大的对比学习框架。

**核心发现**

SimCLR通过系统性研究发现对比学习的几个关键要素：

1. **数据增强的组合至关重要**：随机裁剪+颜色失真是关键组合
2. **非线性投影头很重要**：在表示和对比损失之间加入MLP投影头
3. **大batch size有帮助**：更多的负样本提高学习效果
4. **更长的训练时间**：对比学习比监督学习需要更多epoch

**SimCLR架构**

```
输入图片 x
    ↓
[数据增强] → x_i, x_j (同一张图的两个视角)
    ↓
[编码器 f] → h_i, h_j (表示向量)
    ↓
[投影头 g] → z_i, z_j (投影空间)
    ↓
[对比损失] 拉近z_i和z_j，推开其他样本
```

**InfoNCE损失的数学推导**

对比学习使用**InfoNCE（Noise Contrastive Estimation）**损失：

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{N} \mathbb{1}_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}$$

其中：
- $\text{sim}(u, v) = \frac{u^T v}{||u|| \cdot ||v||}$ 是余弦相似度
- $\tau$ 是温度参数（temperature）
- $N$ 是batch中所有样本（包括正负样本）

**温度系数的作用**

温度系数$\tau$控制分布的"尖锐"程度：

- **$\tau$ 很小（如0.1）**：分布更尖锐，模型对困难负样本更敏感
- **$\tau$ 很大（如1.0）**：分布更平缓，模型对所有样本一视同仁

数学上，当$\tau \to 0$，损失变成hard negative mining；当$\tau \to \infty$，所有样本权重相同。

### 35.3.3 MoCo：动量对比学习

**论文背景**

MoCo（Momentum Contrast）由何恺明团队在2020年提出（He et al., 2020），解决了对比学习中负样本数量受限的问题。

**核心问题**

SimCLR需要非常大的batch size（如4096、8192）来获得足够的负样本。这在计算资源有限的情况下很难实现。

**MoCo的创新：动态字典**

MoCo将对比学习看作**字典查找问题**：
- 编码的查询(query)：当前样本
- 编码的键(keys)：字典中的样本（包括正样本和负样本）

**生活化比喻：记忆卡片盒**

想象你有一个巨大的记忆卡片盒：
- 每张卡片记录一个图片的编码
- 新图片与当前卡片对比（正样本匹配）
- 也与盒子里其他卡片对比（负样本区分）

MoCo的独特之处是使用**队列(queue)**存储之前的编码作为负样本，并使用**动量编码器**保持一致性。

**动量更新公式**

MoCo使用两个编码器：
- $f_q$：查询编码器（正常梯度更新）
- $f_k$：键编码器（动量更新）

键编码器的参数$\theta_k$通过动量方式更新：

$$\theta_k \leftarrow m \theta_k + (1 - m) \theta_q$$

其中$m \in [0, 1)$是动量系数（通常设为0.999）。

**为什么动量更新有效？**

1. **一致性**：缓慢更新的键编码器提供更一致的目标
2. **稳定性**：避免了查询编码器的快速振荡
3. **历史信息**：队列中的键都来自相似的编码器

**MoCo的损失函数**

$$\mathcal{L}_q = -\log \frac{\exp(q \cdot k_+ / \tau)}{\exp(q \cdot k_+ / \tau) + \sum_{i=1}^{K} \exp(q \cdot k_i / \tau)}$$

其中：
- $q$：查询表示
- $k_+$：正样本（同一张图的另一个视角）
- $k_i$：队列中的$K$个负样本

### 35.3.4 数据增强的重要性

对比学习的成功很大程度上依赖于精心设计的数据增强。以下是常用的增强策略：

| 增强类型 | 描述 | 重要性 |
|----------|------|--------|
| 随机裁剪+缩放 | 随机裁剪图片并resize | ⭐⭐⭐ 关键 |
| 颜色抖动 | 调整亮度、对比度、饱和度、色调 | ⭐⭐⭐ 关键 |
| 高斯模糊 | 应用高斯模糊 | ⭐⭐ 重要 |
| 灰度化 | 转换为灰度图 | ⭐⭐ 重要 |
| 水平翻转 | 随机水平翻转 | ⭐⭐ 重要 |

**关键发现**：随机裁剪+颜色失真的组合是SimCLR成功的关键。单独使用任何一种效果都不好。

### 35.3.5 代码实现：SimCLR与MoCo

```python
"""
对比学习实现：SimCLR 和 MoCo
包含数据增强、对比损失和训练循环
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import ImageFilter
import random

# ============ 数据增强 ============

class GaussianBlur:
    """高斯模糊增强"""
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma
    
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))

class ContrastiveTransform:
    """
    对比学习的数据增强
    为每张图片生成两个增强视角
    """
    def __init__(self, image_size=224):
        # SimCLR的数据增强pipeline
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        # 返回同一张图的两个不同增强版本
        return self.transform(x), self.transform(x)

# ============ SimCLR ============

class SimCLR(nn.Module):
    """
    SimCLR: 简单对比学习框架
    """
    def __init__(self, encoder, projection_dim=128, hidden_dim=2048):
        super().__init__()
        self.encoder = encoder  # 例如ResNet
        
        # 获取编码器输出维度
        encoder_dim = self.get_encoder_dim()
        
        # 投影头：编码器输出 -> 隐藏层 -> 投影空间
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        
    def get_encoder_dim(self):
        """获取编码器输出维度"""
        # 假设encoder有fc层
        if hasattr(self.encoder, 'fc'):
            return self.encoder.fc.in_features
        return 2048  # 默认ResNet维度
    
    def forward(self, x):
        """
        前向传播
        
        Returns:
            h: 编码器表示 [batch_size, encoder_dim]
            z: 投影表示 [batch_size, projection_dim]
        """
        h = self.encoder(x)
        z = self.projection_head(h)
        # L2归一化（对比学习的关键）
        z = F.normalize(z, dim=-1)
        return h, z
    
    def contrastive_loss(self, z_i, z_j, temperature=0.5):
        """
        NT-Xent损失（归一化温度标度交叉熵损失）
        
        Args:
            z_i, z_j: 同一张图的两个视角 [batch_size, projection_dim]
            temperature: 温度系数
        
        Returns:
            loss: 对比损失
        """
        batch_size = z_i.shape[0]
        
        # 拼接所有表示
        z = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, projection_dim]
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(z, z.T) / temperature  # [2N, 2N]
        
        # 创建标签：正样本对的位置
        # 对于第i个样本，正样本是i+batch_size（它的配对视角）
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        
        # 正样本索引
        pos_indices = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z.device)
        
        # 使用交叉熵损失
        loss = F.cross_entropy(similarity_matrix, pos_indices)
        
        return loss

# ============ MoCo ============

class MoCo(nn.Module):
    """
    MoCo: 动量对比学习
    使用队列存储负样本，使用动量编码器保持一致性
    """
    def __init__(self, encoder, dim=128, K=65536, m=0.999, T=0.07):
        """
        Args:
            encoder: 基础编码器
            dim: 特征维度
            K: 队列大小（负样本数）
            m: 动量系数
            T: 温度系数
        """
        super().__init__()
        
        self.K = K
        self.m = m
        self.T = T
        
        # 创建查询编码器和键编码器
        self.encoder_q = encoder
        self.encoder_k = self._create_momentum_encoder(encoder)
        
        # 获取编码器输出维度
        encoder_dim = self._get_encoder_dim(encoder)
        
        # 投影头
        self.proj_q = nn.Linear(encoder_dim, dim)
        self.proj_k = nn.Linear(encoder_dim, dim)
        
        # 初始化队列
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    def _create_momentum_encoder(self, encoder):
        """创建动量编码器（初始与查询编码器相同，但不参与梯度更新）"""
        momentum_encoder = type(encoder)(**encoder.init_kwargs) if hasattr(encoder, 'init_kwargs') else encoder
        momentum_encoder.load_state_dict(encoder.state_dict())
        
        # 冻结参数
        for param in momentum_encoder.parameters():
            param.requires_grad = False
        
        return momentum_encoder
    
    def _get_encoder_dim(self, encoder):
        """获取编码器输出维度"""
        if hasattr(encoder, 'fc'):
            return encoder.fc.in_features
        return 2048
    
    @torch.no_grad()
    def _momentum_update(self):
        """动量更新键编码器"""
        for param_q, param_k in zip(
            self.encoder_q.parameters(), 
            self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        
        # 同时更新投影头
        for param_q, param_k in zip(
            self.proj_q.parameters(),
            self.proj_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """更新队列"""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # 替换队列中的旧样本
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # 处理循环
            remaining = self.K - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr
    
    def forward(self, im_q, im_k):
        """
        前向传播
        
        Args:
            im_q: 查询图像
            im_k: 键图像（同一张图的不同增强）
        
        Returns:
            logits, labels: 用于计算对比损失
        """
        # 计算查询表示
        q = self.encoder_q(im_q)
        q = self.proj_q(q)
        q = F.normalize(q, dim=-1)  # [batch_size, dim]
        
        # 计算键表示（无梯度）
        with torch.no_grad():
            self._momentum_update()
            
            k = self.encoder_k(im_k)
            k = self.proj_k(k)
            k = F.normalize(k, dim=-1)  # [batch_size, dim]
        
        # 计算logits
        # 正样本相似度
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # [batch_size, 1]
        
        # 负样本相似度（与队列中的所有样本）
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # [batch_size, K]
        
        # 拼接logits
        logits = torch.cat([l_pos, l_neg], dim=-1)  # [batch_size, 1+K]
        logits /= self.T
        
        # 标签：正样本在位置0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # 更新队列
        self._dequeue_and_enqueue(k)
        
        return logits, labels
    
    def compute_loss(self, logits, labels):
        """计算对比损失"""
        return F.cross_entropy(logits, labels)

# ============ 对比学习训练 ============

class ContrastiveLearner:
    """对比学习训练器"""
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
    def train_epoch(self, dataloader, optimizer, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, _) in enumerate(dataloader):
            if isinstance(self.model, SimCLR):
                # SimCLR训练
                (x_i, x_j), _ = images
                x_i, x_j = x_i.to(self.device), x_j.to(self.device)
                
                _, z_i = self.model(x_i)
                _, z_j = self.model(x_j)
                
                loss = self.model.contrastive_loss(z_i, z_j)
                
            elif isinstance(self.model, MoCo):
                # MoCo训练
                (x_q, x_k), _ = images
                x_q, x_k = x_q.to(self.device), x_k.to(self.device)
                
                logits, labels = self.model(x_q, x_k)
                loss = self.model.compute_loss(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return total_loss / len(dataloader)

# 使用示例
def create_simclr_model():
    """创建SimCLR模型"""
    import torchvision.models as models
    
    # 使用ResNet作为backbone
    encoder = models.resnet50(pretrained=False)
    encoder.fc = nn.Identity()  # 移除最后的fc层
    
    model = SimCLR(encoder, projection_dim=128, hidden_dim=2048)
    return model

def create_moco_model():
    """创建MoCo模型"""
    import torchvision.models as models
    
    encoder = models.resnet50(pretrained=False)
    encoder.fc = nn.Identity()
    
    model = MoCo(encoder, dim=128, K=65536, m=0.999, T=0.07)
    return model

# 线性评估：冻结编码器，只训练线性分类器
def linear_evaluation(encoder, train_loader, test_loader, num_classes, epochs=100):
    """
    线性评估协议：评估学习到的表示质量
    """
    device = next(encoder.parameters()).device
    encoder.eval()
    
    # 获取表示维度
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        features = encoder(dummy_input)
        feature_dim = features.shape[1]
    
    # 线性分类器
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        classifier.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                features = encoder(images)
            
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 评估
        if epoch % 10 == 0:
            acc = evaluate(classifier, encoder, test_loader, device)
            print(f"Epoch {epoch}, Accuracy: {acc:.2f}%")
    
    return classifier

def evaluate(classifier, encoder, test_loader, device):
    """评估准确率"""
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = encoder(images)
            outputs = classifier(features)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total
```

---

## 35.4 非对比式自监督学习

### 35.4.1 一个问题：为什么需要负样本？

在SimCLR和MoCo中，负样本起到了防止**表示崩溃(Representation Collapse)**的关键作用：
- 如果没有负样本，模型可能学到一个平凡解：把所有输入映射到相同的表示
- 负样本强制模型区分不同的输入

但负样本也带来了问题：
- 需要大batch size或复杂的队列机制
- 计算和内存开销大

**能否在没有负样本的情况下学习？**

### 35.4.2 BYOL：自举你的潜在表示

**论文背景**

BYOL（Bootstrap Your Own Latent）由DeepMind在2020年提出（Grill et al., 2020），它惊人地发现：**不需要负样本也能进行自监督学习！**

**生活化比喻：自我提升的画家**

想象一个画家在学习绘画：
- 他画了一幅画（预测）
- 然后看一眼参考图（目标）
- 虽然参考图也是他自己画的（只是稍微不同的版本），但他可以通过不断改进来进步

BYOL使用两个网络：
- **在线网络(Online Network)**：接收一个增强视角，预测另一个视角的表示
- **目标网络(Target Network)**：生成预测目标

**BYOL架构**

```
图片 x
  ├──[增强1]→ x_online
  │           ↓
  │      [在线编码器]→ [在线投影]→ [预测器]→ q_θ (预测)
  │                                    ↑
  │                              最小化 ||q_θ - sg(z_ξ)||²
  │                                    ↓
  └──[增强2]→ x_target              sg(z_ξ) (目标，stop-gradient)
              ↓
         [目标编码器]→ [目标投影]→ z_ξ
              ↑
         动量更新：ξ ← τξ + (1-τ)θ
```

**损失函数**

BYOL使用均方误差损失：

$$\mathcal{L}_{\text{BYOL}} = ||q_\theta(z_\theta) - \text{sg}(z_\xi)||_2^2$$

其中：
- $q_\theta$ 是在线网络的预测器输出
- $z_\xi$ 是目标网络的投影输出
- $\text{sg}$ 表示stop-gradient（不计算梯度）

**为什么BYOL不会崩溃？**

这是一个长期的研究问题。关键理论解释包括：

1. **预测器的存在**：预测器网络引入了额外的非对称性
2. **批归一化(BatchNorm)**：有研究表明BN在防止崩溃中起重要作用
3. **权重衰减**：正则化防止平凡解
4. **动量编码器**：EMA更新提供了"移动的目标"

**BYOL的数学推导**

设在线网络参数为$\theta$，目标网络参数为$\xi$。

在线网络包含：
- 编码器 $f_\theta$
- 投影器 $g_\theta$ 
- 预测器 $q_\theta$

目标网络包含：
- 编码器 $f_\xi$（EMA更新）
- 投影器 $g_\xi$（EMA更新）

对于输入$x$的两个增强版本$v$和$v'$：

$$z_\theta = g_\theta(f_\theta(v)), \quad p_\theta = q_\theta(z_\theta)$$
$$z_\xi = g_\xi(f_\xi(v'))$$

对称损失：
$$\mathcal{L} = ||p_\theta - \text{sg}(z_\xi)||^2 + ||q_\xi(f_\xi(v)) - \text{sg}(g_\theta(f_\theta(v')))||^2$$

### 35.4.3 DINO：自蒸馏与视觉Transformer

**论文背景**

DINO（Self-distillation with no labels）由Facebook AI在2021年提出（Caron et al., 2021），将自蒸馏方法应用到Vision Transformer上，并发现了一些令人惊奇的"涌现性质"。

**DINO的架构**

DINO与BYOL类似，使用学生-教师框架：
- **学生网络**：接收局部裁剪（small crops）
- **教师网络**：接收全局视图（large crops），使用EMA更新

**DINO的关键创新**

1. **多裁剪训练(Multi-crop)**：使用多个小裁剪视图作为学生输入
2. **centering**：对教师输出进行中心化处理，防止某一维度过大
3. **sharpening**：使用较低的温度参数使分布更尖锐

**DINO损失函数**

使用交叉熵损失（而非BYOL的MSE）：

$$\min_\theta \sum_{x \in \{x_1^g, x_2^g\}} \sum_{x' \in V} H(P_t(x), P_s(x'))$$

其中：
- $P_t$ 是教师网络的输出分布
- $P_s$ 是学生网络的输出分布
- $H$ 是交叉熵

**Centering操作**

$$g_t(x) \leftarrow g_t(x) + c$$

其中$c$是EMA更新的均值：
$$c \leftarrow m c + (1 - m) \frac{1}{B} \sum_{i=1}^{B} g_t(x_i)$$

**DINO的涌现性质**

最令人兴奋的是，DINO发现自监督ViT自动学习到了：

1. **语义分割能力**：自注意力图自然地对应物体边界
2. **k-NN分类器**：不需要训练就能达到很好的分类效果
3. **局部到全局的对应**：不同裁剪之间的一致性

### 35.4.4 代码实现：BYOL与DINO

```python
"""
非对比式自监督学习：BYOL 和 DINO 实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        layers = []
        
        # 第一层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        
        # 中间层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class BYOL(nn.Module):
    """
    BYOL: Bootstrap Your Own Latent
    不需要负样本的自监督学习方法
    """
    def __init__(self, encoder, dim=256, proj_hidden_dim=4096, pred_hidden_dim=4096):
        super().__init__()
        
        self.encoder = encoder
        encoder_dim = self._get_encoder_dim()
        
        # 在线网络
        self.online_encoder = encoder
        self.online_projector = MLP(encoder_dim, proj_hidden_dim, dim)
        self.predictor = MLP(dim, pred_hidden_dim, dim)
        
        # 目标网络（初始与在线网络相同）
        self.target_encoder = self._create_momentum_encoder(encoder)
        self.target_projector = MLP(encoder_dim, proj_hidden_dim, dim)
        
        # 同步权重
        self._initialize_target_network()
        
    def _get_encoder_dim(self):
        """获取编码器输出维度"""
        if hasattr(self.encoder, 'fc'):
            return self.encoder.fc.in_features
        return 2048
    
    def _create_momentum_encoder(self, encoder):
        """创建动量编码器"""
        import copy
        momentum_encoder = copy.deepcopy(encoder)
        for param in momentum_encoder.parameters():
            param.requires_grad = False
        return momentum_encoder
    
    def _initialize_target_network(self):
        """初始化目标网络与在线网络相同"""
        for param_o, param_t in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False
    
    @torch.no_grad()
    def update_target_network(self, tau=0.996):
        """动量更新目标网络"""
        for param_o, param_t in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_t.data = tau * param_t.data + (1 - tau) * param_o.data
        
        for param_o, param_t in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters()
        ):
            param_t.data = tau * param_t.data + (1 - tau) * param_o.data
    
    def forward(self, x1, x2):
        """
        前向传播
        
        Args:
            x1, x2: 同一张图的两个不同增强版本
        
        Returns:
            loss: BYOL损失
        """
        # 在线网络：x1 -> 预测
        online_z1 = self.online_projector(self.online_encoder(x1))
        online_q1 = self.predictor(online_z1)
        
        # 在线网络：x2 -> 预测
        online_z2 = self.online_projector(self.online_encoder(x2))
        online_q2 = self.predictor(online_z2)
        
        # 目标网络：x2 -> 目标（无梯度）
        with torch.no_grad():
            target_z2 = self.target_projector(self.target_encoder(x2))
        
        # 目标网络：x1 -> 目标（无梯度）
        with torch.no_grad():
            target_z1 = self.target_projector(self.target_encoder(x1))
        
        # 计算对称损失
        loss1 = self.regression_loss(online_q1, target_z2)
        loss2 = self.regression_loss(online_q2, target_z1)
        
        loss = (loss1 + loss2) / 2
        
        return loss
    
    def regression_loss(self, x, y):
        """
        回归损失：归一化后的MSE
        """
        # L2归一化
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        
        # 均方误差
        return 2 - 2 * (x * y).sum(dim=-1).mean()

class DINOHead(nn.Module):
    """DINO投影头"""
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        
        self.last_layer = nn.Linear(bottleneck_dim, out_dim)
        
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class DINO(nn.Module):
    """
    DINO: 自蒸馏无标签学习
    结合多裁剪训练和centering/sharpening
    """
    def __init__(self, student, teacher, out_dim=65536, 
                 teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        
        self.student = student
        self.teacher = teacher
        
        # 投影头
        self.student_head = DINOHead(self._get_dim(student), out_dim)
        self.teacher_head = DINOHead(self._get_dim(teacher), out_dim)
        
        # 温度参数
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        
        # centering参数
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.center_momentum = center_momentum
        
        # 冻结教师网络
        for param in self.teacher.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False
    
    def _get_dim(self, model):
        """获取模型输出维度"""
        if hasattr(model, 'num_features'):
            return model.num_features
        return 768  # ViT默认
    
    @torch.no_grad()
    def update_teacher(self, m):
        """EMA更新教师网络"""
        for param_s, param_t in zip(
            self.student.parameters(),
            self.teacher.parameters()
        ):
            param_t.data = m * param_t.data + (1 - m) * param_s.data
        
        for param_s, param_t in zip(
            self.student_head.parameters(),
            self.teacher_head.parameters()
        ):
            param_t.data = m * param_t.data + (1 - m) * param_s.data
    
    def forward(self, student_inputs, teacher_inputs):
        """
        前向传播
        
        Args:
            student_inputs: 学生输入（包含多个小裁剪）
            teacher_inputs: 教师输入（2个全局裁剪）
        
        Returns:
            loss: DINO损失
        """
        # 教师前向（无梯度）
        with torch.no_grad():
            teacher_out = []
            for x in teacher_inputs:
                t = self.teacher_head(self.teacher(x))
                # centering + sharpening
                t = F.softmax((t - self.center) / self.teacher_temp, dim=-1)
                teacher_out.append(t)
        
        # 学生前向
        student_out = []
        for x in student_inputs:
            s = self.student_head(self.student(x))
            s = F.log_softmax(s / self.student_temp, dim=-1)
            student_out.append(s)
        
        # 计算损失：每个学生输出与所有教师输出的交叉熵
        loss = 0
        n_loss_terms = 0
        
        for iq, q in enumerate(teacher_out):
            for iv, v in enumerate(student_out):
                # 如果来自同一图像（检查索引）
                if iv // 2 == iq:  # 简化的匹配逻辑
                    loss += torch.sum(-q * v, dim=-1).mean()
                    n_loss_terms += 1
        
        loss /= n_loss_terms
        
        # 更新center
        self.update_center(torch.cat(teacher_out))
        
        return loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """更新centering参数"""
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        
        # EMA更新
        self.center = self.center * self.center_momentum + \
                      batch_center * (1 - self.center_momentum)

# 训练循环
def train_byol(model, dataloader, optimizer, device, epochs=100):
    """训练BYOL"""
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, ((x1, x2), _) in enumerate(dataloader):
            x1, x2 = x1.to(device), x2.to(device)
            
            loss = model(x1, x2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新目标网络
            model.update_target_network()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def train_dino(model, dataloader, optimizer, device, epochs=100):
    """训练DINO"""
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (multi_crop_images, _) in enumerate(dataloader):
            # multi_crop_images: 包含全局和局部裁剪的列表
            teacher_crops = [img.to(device) for img in multi_crop_images[:2]]
            student_crops = [img.to(device) for img in multi_crop_images]
            
            loss = model(student_crops, teacher_crops)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # EMA更新教师
            m = 0.996  # 动量系数
            model.update_teacher(m)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
```

---

## 35.5 掩码图像建模

### 35.5.1 从BERT到视觉：掩码建模的统一框架

**核心思想**

BERT的成功启发了研究者：能否像遮罩文本一样遮罩图像，让模型学习重建被遮罩的部分？

**生活化比喻：拼图游戏**

想象你玩一个拼图游戏：
- 一张图片的大部分被遮住了
- 你只能看到一小部分碎片
- 你的任务是根据可见的部分，想象并重建完整的图片

这就是**掩码图像建模(Masked Image Modeling, MIM)**的核心思想。

### 35.5.2 MAE：掩码自编码器

**论文背景**

MAE（Masked Autoencoders）由何恺明团队在2022年提出（He et al., 2022），是一个简单但极其有效的自监督学习方法。

**MAE的核心设计**

1. **非对称编码器-解码器架构**
   - **编码器**：只在可见的patch上运行（非遮罩部分）
   - **解码器**：轻量级，重建完整图像

2. **高遮罩比例**：遮罩75%的patch！
   - 这创造了一个具有挑战性的预训练任务
   - 迫使模型学习高级语义而非简单复制相邻像素

**MAE架构**

```
图像 (224x224 = 196个16x16 patch)
    ↓
随机遮罩 75% (147个patch被遮)
    ↓
[编码器ViT] → 处理可见的49个patch
    ↓
表示向量
    ↓
[解码器] → 加入遮罩token，重建全部196个patch
    ↓
像素值预测（MSE损失）
```

**MAE的数学原理**

设输入图像$x$被划分为$N$个patch，遮罩集合为$\mathcal{M}$。

编码器$f$只在可见patch上运行：
$$h = f(x_{\backslash \mathcal{M}})$$

解码器$g$重建所有patch：
$$\hat{x} = g(h, \{m_i\}_{i \in \mathcal{M}})$$

其中$m_i$是可学习的遮罩token。

损失函数（只在遮罩patch上计算MSE）：
$$\mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} ||\hat{x}_i - x_i||^2$$

**为什么MAE有效？**

1. **高遮罩比例**：75%的遮罩比例迫使模型学习全局理解
2. **像素重建目标**：比离散token更细粒度
3. **非对称设计**：编码器更高效，解码器专注重建

### 35.5.3 BEiT：BERT风格的图像预训练

**论文背景**

BEiT（BERT Pre-Training of Image Transformers）由微软在2021年提出（Bao et al., 2021/2022），将BERT的MLM直接应用到图像领域。

**BEiT的关键区别**

与MAE不同，BEiT预测的是**离散视觉token**而非原始像素：

1. **图像Tokenizer**：使用预训练的VQ-VAE将图像转换为离散token
2. **MLM目标**：像BERT一样预测被遮罩的离散token

**BEiT架构**

```
图像
  ↓
[离散的视觉token]（由预训练tokenizer生成，作为目标）
  ↓
随机遮罩部分图像patch
  ↓
[ViT编码器]
  ↓
预测遮罩位置的离散token（交叉熵损失）
```

**BEiT vs MAE**

| 特性 | BEiT | MAE |
|------|------|-----|
| 重建目标 | 离散视觉token | 原始像素 |
| 需要预训练tokenizer | 是 | 否 |
| 遮罩比例 | ~40% | ~75% |
| 损失函数 | 交叉熵 | MSE |
| 解码器 | 线性层 | Transformer |

### 35.5.4 代码实现：MAE与BEiT

```python
"""
掩码图像建模：MAE 和 BEiT 实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PatchEmbedding(nn.Module):
    """图像patch嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: [batch_size, channels, img_size, img_size]
        x = self.proj(x)  # [batch_size, embed_dim, n_patches^0.5, n_patches^0.5]
        x = x.flatten(2)  # [batch_size, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [batch_size, n_patches, embed_dim]
        return x

class MAEEncoder(nn.Module):
    """MAE编码器：只在可见patch上运行"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, channels, img_size, img_size]
            mask: 遮罩 [batch_size, n_patches]，True表示保留，False表示遮罩
        """
        # Patch嵌入
        x = self.patch_embed(x)  # [B, n_patches, embed_dim]
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 如果只处理可见patch
        if mask is not None:
            B = x.shape[0]
            # 收集所有可见patch
            visible_patches = []
            for i in range(B):
                visible = x[i][mask[i]]  # [n_visible, embed_dim]
                visible_patches.append(visible)
            # 这里简化处理，实际需要更复杂的batch处理
            x = torch.cat(visible_patches, dim=0)
        
        x = self.transformer(x)
        x = self.norm(x)
        
        return x

class MAEDecoder(nn.Module):
    """MAE解码器：轻量级，重建完整图像"""
    def __init__(self, embed_dim=768, decoder_dim=512, 
                 depth=8, num_heads=16, patch_size=16, 
                 n_patches=196, out_channels=3):
        super().__init__()
        
        # 将编码器输出投影到解码器维度
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        
        # 可学习的遮罩token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        # 解码器位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, decoder_dim))
        
        # Transformer解码器
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=num_heads,
            dim_feedforward=int(decoder_dim * 4),
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        
        # 输出头：预测每个patch的像素值
        self.head = nn.Linear(decoder_dim, patch_size * patch_size * out_channels)
        
        self.patch_size = patch_size
        self.out_channels = out_channels
        
    def forward(self, x_encoded, mask, ids_restore):
        """
        Args:
            x_encoded: 编码器输出 [B, n_visible, embed_dim]
            mask: 遮罩信息
            ids_restore: 恢复原始顺序的索引
        """
        # 投影到解码器维度
        x = self.decoder_embed(x_encoded)
        
        # 添加遮罩token
        B = x.shape[0]
        n_visible = x.shape[1]
        n_masked = ids_restore.shape[1] - n_visible
        
        mask_tokens = self.mask_token.expand(B, n_masked, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        
        # 恢复原始顺序
        x = torch.gather(x, dim=1, 
                        index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 解码
        x = self.transformer(x)
        
        # 预测像素值
        x = self.head(x)  # [B, n_patches, patch_size^2 * 3]
        
        return x

class MAE(nn.Module):
    """
    Masked Autoencoder (MAE)
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, encoder_depth=12, decoder_depth=8,
                 mask_ratio=0.75):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 编码器
        self.encoder = MAEEncoder(
            img_size, patch_size, in_channels,
            embed_dim, encoder_depth
        )
        
        # 解码器
        self.decoder = MAEDecoder(
            embed_dim=embed_dim,
            decoder_dim=512,
            depth=decoder_depth,
            patch_size=patch_size,
            n_patches=self.n_patches,
            out_channels=in_channels
        )
        
    def random_masking(self, x):
        """
        随机遮罩patch
        
        Returns:
            x_visible: 可见patch
            mask: 遮罩 [B, n_patches]
            ids_restore: 恢复索引
        """
        B, N, D = x.shape
        n_keep = int(N * (1 - self.mask_ratio))
        
        # 随机排列
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 保留前n_keep个
        ids_keep = ids_shuffle[:, :n_keep]
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # 生成遮罩mask：0为保留，1为移除
        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_visible, mask, ids_restore
    
    def patchify(self, imgs):
        """将图像转换为patch"""
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p * p * 3)
        return x
    
    def unpatchify(self, patches):
        """将patch还原为图像"""
        p = self.patch_size
        h = w = int(np.sqrt(patches.shape[1]))
        x = patches.reshape(patches.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(patches.shape[0], 3, h * p, w * p)
        return imgs
    
    def forward(self, imgs):
        """
        前向传播
        
        Args:
            imgs: [B, 3, H, W]
        
        Returns:
            loss: 重建损失
            pred: 重建结果
            mask: 遮罩
        """
        # 编码
        x = self.encoder.patch_embed(imgs)
        x = x + self.encoder.pos_embed
        
        # 随机遮罩
        x_visible, mask, ids_restore = self.random_masking(x)
        
        # 编码可见patch
        latent = self.encoder.transformer(x_visible)
        latent = self.encoder.norm(latent)
        
        # 解码
        pred = self.decoder(latent, mask, ids_restore)
        
        # 计算损失（只在遮罩patch上）
        target = self.patchify(imgs)
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, n_patches]，每个patch的MSE
        
        # 只在遮罩patch上计算损失
        loss = (loss * mask).sum() / mask.sum()
        
        return loss, pred, mask
    
    @torch.no_grad()
    def reconstruct(self, imgs):
        """重建图像（可视化用）"""
        self.eval()
        loss, pred, mask = self.forward(imgs)
        
        # 重建图像
        pred_img = self.unpatchify(pred)
        
        # 遮罩后的输入图像
        mask_img = self.unpatchify(self.patchify(imgs) * (1 - mask.unsqueeze(-1)))
        
        return mask_img, pred_img

class BEiT(nn.Module):
    """
    BEiT: BERT风格的图像预训练
    预测离散的视觉token而非像素值
    """
    def __init__(self, img_size=224, patch_size=16, vocab_size=8192,
                 embed_dim=768, depth=12, num_heads=12, mask_ratio=0.4):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.vocab_size = vocab_size
        
        # Patch嵌入
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        
        # 可学习的遮罩token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * 4),
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 预测头：预测离散token
        self.head = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, imgs, visual_tokens):
        """
        Args:
            imgs: [B, 3, H, W]
            visual_tokens: [B, n_patches] 离散的视觉token（由tokenizer生成）
        
        Returns:
            loss: 交叉熵损失
        """
        B = imgs.shape[0]
        
        # Patch嵌入
        x = self.patch_embed(imgs)
        x = x + self.pos_embed
        
        # 随机遮罩
        n_patches = x.shape[1]
        n_mask = int(n_patches * self.mask_ratio)
        
        # 随机选择遮罩位置
        noise = torch.rand(B, n_patches, device=imgs.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask_positions = ids_shuffle[:, :n_mask]
        
        # 替换为mask token
        mask_tokens = self.mask_token.expand(B, n_mask, -1)
        batch_indices = torch.arange(B, device=imgs.device).unsqueeze(1).expand(-1, n_mask)
        x[batch_indices, mask_positions] = mask_tokens
        
        # 编码
        x = self.transformer(x)
        x = self.norm(x)
        
        # 只在遮罩位置预测
        masked_output = x[batch_indices, mask_positions]
        logits = self.head(masked_output)
        
        # 目标
        targets = visual_tokens[batch_indices, mask_positions]
        
        # 交叉熵损失
        loss = F.cross_entropy(logits, targets)
        
        return loss

# 可视化MAE的遮罩和重建
def visualize_mae_reconstruction(model, img_tensor):
    """可视化MAE的重建效果"""
    import matplotlib.pyplot as plt
    
    mask_img, pred_img = model.reconstruct(img_tensor.unsqueeze(0))
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # 原始图像
    axes[0].imshow(img_tensor.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # 遮罩图像
    axes[1].imshow(mask_img[0].permute(1, 2, 0).cpu().numpy())
    axes[1].set_title(f'Masked ({model.mask_ratio*100}% masked)')
    axes[1].axis('off')
    
    # 重建图像
    axes[2].imshow(pred_img[0].permute(1, 2, 0).cpu().numpy())
    axes[2].set_title('Reconstructed')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig
```

---

## 35.6 多模态自监督学习

### 35.6.1 CLIP：连接图像与语言的桥梁

**论文背景**

CLIP（Contrastive Language-Image Pre-training）由OpenAI在2021年提出（Radford et al., 2021），是一个革命性的多模态模型。

**核心思想**

CLIP使用**自然语言监督**来学习视觉表示：
- 从互联网上收集4亿对(图像, 文本)
- 训练模型将匹配的图像-文本对在嵌入空间中拉近
- 不匹配的推远

**生活化比喻：看图说话**

想象你在学习一个新概念。老师给你看一张"金毛犬"的图片，同时告诉你"这是一只金毛犬在草地上奔跑"。

通过大量这样的"图片+描述"配对，你学会了：
- 什么样的图片对应"金毛犬"
- "草地上奔跑"对应什么样的场景

CLIP正是这样做的！

**CLIP架构**

```
图像                         文本
  ↓                           ↓
[图像编码器]               [文本编码器]
 (ResNet/ViT)              (Transformer)
  ↓                           ↓
图像嵌入 [I1, I2, ...]     文本嵌入 [T1, T2, ...]
  ↓                           ↓
      ↘                   ↙
        对比学习：匹配(Ii, Ti)
```

**对比损失（对称形式）**

对于batch中的$N$对(图像, 文本)：

$$\mathcal{L} = \frac{1}{2} \left[ \underbrace{-\sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j)/\tau)}}_{\text{图像到文本}} + \underbrace{-\sum_{i=1}^{N} \log \frac{\exp(\text{sim}(T_i, I_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(T_i, I_j)/\tau)}}_{\text{文本到图像}} \right]$$

其中$\text{sim}(u, v) = \frac{u \cdot v}{||u|| \cdot ||v||}$是余弦相似度。

**CLIP的涌现能力**

1. **零样本分类**：不需要训练就能对新类别进行分类
2. **开放词汇**：可以理解训练时未见过的概念
3. **跨模态检索**：用文字搜图片，用图片搜文字

**零样本分类示例**

```python
# 给定一张图片
image = load_image("dog.jpg")

# 定义候选标签
texts = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

# CLIP计算相似度
image_features = clip_encode_image(image)
text_features = clip_encode_text(texts)
similarities = cosine_similarity(image_features, text_features)

# 预测最相似的标签
predicted_label = texts[argmax(similarities)]
# 输出: "a photo of a dog"
```

### 35.6.2 代码实现：CLIP

```python
"""
CLIP: 对比语言-图像预训练实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIP(nn.Module):
    """
    CLIP模型：对比语言-图像预训练
    """
    def __init__(self, 
                 image_encoder,
                 text_encoder,
                 embed_dim=512,
                 temperature=0.07):
        super().__init__()
        
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        
        # 投影层：将编码器输出映射到共享嵌入空间
        self.image_projection = nn.Linear(
            self._get_image_dim(), embed_dim
        )
        self.text_projection = nn.Linear(
            self._get_text_dim(), embed_dim
        )
        
        # 温度参数（可学习或固定）
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
        
    def _get_image_dim(self):
        """获取图像编码器输出维度"""
        if hasattr(self.image_encoder, 'num_features'):
            return self.image_encoder.num_features
        return 2048
    
    def _get_text_dim(self):
        """获取文本编码器输出维度"""
        if hasattr(self.text_encoder, 'hidden_size'):
            return self.text_encoder.hidden_size
        return 768
    
    def encode_image(self, images):
        """编码图像"""
        features = self.image_encoder(images)
        
        # 全局平均池化（如果是特征图）
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])
        
        # 投影到共享空间并归一化
        embeddings = self.image_projection(features)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings
    
    def encode_text(self, text_tokens):
        """编码文本"""
        # 文本编码器输出 [batch_size, seq_len, hidden_dim]
        text_features = self.text_encoder(text_tokens)
        
        # 取[CLS] token或平均池化
        if text_features.dim() == 3:
            text_features = text_features[:, 0]  # [CLS]
        
        # 投影到共享空间并归一化
        embeddings = self.text_projection(text_features)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings
    
    def forward(self, images, text_tokens):
        """
        前向传播
        
        Args:
            images: [batch_size, 3, H, W]
            text_tokens: [batch_size, seq_len]
        
        Returns:
            logits_per_image: [batch_size, batch_size]
            logits_per_text: [batch_size, batch_size]
        """
        # 获取嵌入
        image_features = self.encode_image(images)      # [B, embed_dim]
        text_features = self.encode_text(text_tokens)   # [B, embed_dim]
        
        # 温度缩放
        logit_scale = self.logit_scale.exp()
        
        # 计算相似度
        logits_per_image = logit_scale * image_features @ text_features.T  # [B, B]
        logits_per_text = logits_per_image.T
        
        return logits_per_image, logits_per_text
    
    def compute_loss(self, logits_per_image, logits_per_text):
        """
        计算对比损失
        """
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        # 图像到文本的损失
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        
        # 文本到图像的损失
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        
        # 对称损失
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss
    
    @torch.no_grad()
    def zero_shot_classify(self, images, class_names, templates=None):
        """
        零样本分类
        
        Args:
            images: [batch_size, 3, H, W]
            class_names: 类别名称列表
            templates: 文本模板列表，如["a photo of a {}"]
        
        Returns:
            predictions: 预测的类别索引
        """
        if templates is None:
            templates = ["a photo of a {}"]
        
        self.eval()
        
        # 生成所有类别的文本描述
        all_texts = []
        for classname in class_names:
            for template in templates:
                all_texts.append(template.format(classname))
        
        # 编码图像
        image_features = self.encode_image(images)
        
        # 编码文本（需要tokenizer，这里简化）
        # text_tokens = tokenize(all_texts)
        # text_features = self.encode_text(text_tokens)
        
        # 计算相似度并预测
        # similarities = image_features @ text_features.T
        # predictions = similarities.argmax(dim=-1)
        
        # 返回占位符
        return torch.zeros(images.shape[0], dtype=torch.long)

def create_simple_clip():
    """创建一个简单的CLIP模型示例"""
    import torchvision.models as models
    
    # 图像编码器（ResNet）
    image_encoder = models.resnet50(pretrained=False)
    image_encoder.fc = nn.Identity()
    
    # 文本编码器（简化的Transformer）
    class SimpleTextEncoder(nn.Module):
        def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            
        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            return x
    
    text_encoder = SimpleTextEncoder()
    
    clip = CLIP(image_encoder, text_encoder, embed_dim=512)
    return clip

# 训练示例
def train_clip_example():
    """CLIP训练示例"""
    clip = create_simple_clip()
    optimizer = torch.optim.AdamW(clip.parameters(), lr=3e-4)
    
    batch_size = 256
    
    for epoch in range(10):
        # 模拟数据
        images = torch.randn(batch_size, 3, 224, 224)
        text_tokens = torch.randint(0, 10000, (batch_size, 77))
        
        logits_per_image, logits_per_text = clip(images, text_tokens)
        loss = clip.compute_loss(logits_per_image, logits_per_text)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

## 35.7 自监督学习的未来

### 35.7.1 统一预训练框架

当前，NLP、CV、多模态领域各自有不同的预训练方法。未来的趋势是：

1. **统一的预训练目标**：一个损失函数适用于所有模态
2. **多模态融合**：同时理解文本、图像、音频、视频
3. **世界模型**：理解物理世界的因果关系

**代表性工作**：
- **Data2Vec**：统一的自监督框架
- **ImageBind**：连接六种模态
- **GPT-4V**：多模态大模型

### 35.7.2 世界模型与预测

自监督学习的终极目标之一是构建**世界模型(World Model)**：
- 预测未来的帧（视频预测）
- 理解因果关系（物理推理）
- 支持规划和决策

**代表性工作**：
- **JEPA（Joint Embedding Predictive Architecture）**：LeCun提出的非生成式预测架构
- **GAIA-1**：自动驾驶世界模型
- **Sora**：视频生成世界模型

### 35.7.3 与强化学习的结合

自监督学习可以为强化学习提供：
- **状态表示**：从高维观测中学习有用的状态表示
- **世界模型**：预测环境动态
- **探索信号**：好奇心驱动的探索

**代表性工作**：
- **CURL**：对比无监督表示用于RL
- **Dreamer**：基于世界模型的RL
- **ICM（Intrinsic Curiosity Module）**：好奇心驱动的探索

### 35.7.4 关键挑战与开放问题

1. **表示崩溃**：如何在不需要负样本的情况下防止崩溃
2. **样本效率**：如何用小数据学习更好的表示
3. **可解释性**：理解自监督模型学到了什么
4. **下游迁移**：如何设计更好的预训练任务以提高下游性能

---

## 35.8 练习题

### 基础题

**35.1** 解释自监督学习与传统监督学习、无监督学习的区别。为什么自监督学习被称为"不需要人工标注的监督学习"？

**35.2** BERT的MLM任务中，为什么采用80%遮罩、10%随机替换、10%不变的混合策略？如果只遮罩会怎样？

**35.3** 在SimCLR中，为什么数据增强的组合如此重要？请解释"随机裁剪+颜色失真"为什么是有效的组合。

### 进阶题

**35.4** **动量更新的数学推导**

给定MoCo中的动量更新公式：$\theta_k \leftarrow m \theta_k + (1 - m) \theta_q$，其中$m=0.999$。

(1) 证明经过$t$次更新后，$\theta_k^{(t)} = m^t \theta_k^{(0)} + (1-m)\sum_{i=0}^{t-1} m^{t-1-i} \theta_q^{(i)}$

(2) 解释当$m \to 1$时，键编码器的更新特性。

**35.5** **温度系数的作用**

给定InfoNCE损失中的温度系数$\tau$，证明：

(1) 当$\tau \to 0$时，损失退化为只关注最困难的负样本

(2) 当$\tau \to \infty$时，所有负样本的权重趋于相同

**35.6** **BYOL的非崩溃分析**

BYOL使用预测器和动量编码器防止表示崩溃。请设计一个实验验证：
- 移除预测器后会发生什么？
- 移除动量更新（目标网络与在线网络同步更新）后会发生什么？
- 解释为什么这两种修改会导致崩溃。

### 挑战题

**35.7** **实现多视图对比学习**

DINO使用多裁剪策略，学生网络接收多个局部裁剪，教师网络接收全局裁剪。请实现一个支持$K$个视图（2个全局+$K-2$个局部）的通用对比学习框架，并分析不同$K$值对性能的影响。

**35.8** **MAE的遮罩策略分析**

MAE使用随机遮罩75%的patch。请设计并比较以下遮罩策略：
- 随机遮罩（原始MAE）
- 块状遮罩（遮罩连续的patch块）
- 网格遮罩（按规则间隔遮罩）

分析不同策略对重建质量和下游任务性能的影响。

**35.9** **设计新的预训练任务**

基于本章所学，设计一个新的自监督预训练任务，可以是：
- 结合NLP和CV的多模态任务
- 适用于特定领域（如医学影像、分子结构）的任务
- 融合对比学习和掩码建模的混合任务

请详细描述：(1) 任务定义；(2) 损失函数设计；(3) 预期优势；(4) 可能的应用场景。

---

## 本章小结

### 核心概念回顾

| 方法 | 类型 | 核心思想 | 关键创新 |
|------|------|----------|----------|
| BERT MLM | NLP | 预测遮罩词 | 双向编码，混合遮罩策略 |
| GPT | NLP | 自回归预测 | 单向生成，可扩展性强 |
| SimCLR | CV-对比 | 拉近同图不同增强 | 大batch，强数据增强 |
| MoCo | CV-对比 | 动态字典+动量编码 | 队列存储负样本 |
| BYOL | CV-非对比 | 在线网络预测目标网络 | 无需负样本 |
| DINO | CV-非对比 | 自蒸馏+多裁剪 | ViT的涌现性质 |
| MAE | CV-掩码 | 重建像素 | 高遮罩比，非对称编解码 |
| BEiT | CV-掩码 | 预测离散token | VQ-VAE tokenizer |
| CLIP | 多模态 | 图文对比对齐 | 自然语言监督 |

### 关键公式总结

**MLM损失**：
$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i | x_{\backslash \mathcal{M}})$$

**InfoNCE损失**：
$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k} \exp(\text{sim}(z_i, z_k) / \tau)}$$

**动量更新**：
$$\theta_k \leftarrow m \theta_k + (1 - m) \theta_q$$

**MAE重建损失**：
$$\mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} ||\hat{x}_i - x_i||^2$$

### 学习路径建议

1. **入门**：理解BERT和SimCLR的基本原理
2. **进阶**：深入MoCo和BYOL的实现细节
3. **精通**：研究MAE和CLIP的架构设计
4. **前沿**：关注世界模型、统一预训练框架的最新进展

---

## 参考文献

Bao, H., Dong, L., Piao, S., & Wei, F. (2021). BEiT: BERT pre-training of image transformers. *arXiv preprint arXiv:2106.08254*.

Bao, H., Dong, L., & Wei, F. (2022). BEiT: BERT pre-training of image transformers. In *International Conference on Learning Representations*.

Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging properties in self-supervised vision transformers. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 9650-9660).

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. In *International Conference on Machine Learning* (pp. 1597-1607). PMLR.

Chen, X., & He, K. (2021). Exploring simple siamese representation learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 15750-15758).

Chen, M., Radford, A., Child, R., Wu, J., Jun, H., Luan, D., & Sutskever, I. (2020). Generative pre-training from pixels. In *International Conference on Machine Learning* (pp. 1691-1703). PMLR.

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)* (pp. 4171-4186).

Grill, J. B., Strub, F., Altché, F., Tallec, C., Richemond, P., Buchatskaya, E., ... & Valko, M. (2020). Bootstrap your own latent: A new approach to self-supervised learning. In *Advances in Neural Information Processing Systems* (Vol. 33, pp. 21271-21284).

He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for unsupervised visual representation learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 9729-9738).

He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 16000-16009).

Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training. *OpenAI Technical Report*.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. In *International Conference on Machine Learning* (pp. 8748-8763). PMLR.

---

*本章完。你已掌握了自监督学习的核心原理和实现，这是理解现代大模型技术的必备知识。*
