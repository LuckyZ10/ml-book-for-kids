# 第二十四章：注意力机制与Transformer——专注的力量

> *Attention is all you need.* —— Vaswani et al., 2017

---

## 开场故事：图书馆里的专注读者

想象你走进一座巨大的图书馆，里面存放着人类所有的知识。你需要找到关于"如何制作巧克力蛋糕"的所有信息。

传统的阅读方式是什么？你拿起第一本书，从头到尾读一遍；然后拿起第二本，再读一遍……等你读完所有书，可能已经几个月过去了。

但人类的阅读方式是这样的：
- 你扫一眼书名，快速筛选出可能相关的书
- 翻开目录，找到"蛋糕"或"巧克力"相关章节
- 跳转到具体页面，只读那几段关键内容
- 如果不够清晰，再去翻阅另一本书的相关部分

**注意力机制**，就是让机器学会这种"聪明阅读"的方法。

---

## 24.1 从瓶颈到突破：注意力机制的诞生

### 24.1.1 Seq2Seq的困境

还记得上一章的RNN吗？Seq2Seq模型用它来做机器翻译：

```
"我爱机器学习" → [Encoder RNN] → 一个固定向量 → [Decoder RNN] → "I love ML"
```

这个"固定向量"是问题的关键——**瓶颈问题**。

**类比**：想象你要把一整本书的内容压缩成一张便利贴上的摘要。对于短文章，这还能勉强做到；但对于《战争与和平》这样的巨著，无论怎么压缩，都会丢失大量信息。

| 句子长度 | 信息保留率 | 翻译质量 |
|---------|-----------|---------|
| 5-10词  | 95%       | 优秀    |
| 20-30词 | 70%       | 一般    |
| 50+词   | 40%       | 很差    |

### 24.1.2 人类的翻译智慧

人类翻译家是如何工作的？

假设你在翻译这句话：
> "尽管天气预报说会下雨，但他还是决定不带伞，结果中午的时候被淋成了落汤鸡。"

当你翻译到"落汤鸡"时，你会回头看哪些词？
- **不会看**："天气预报"（太远且不直接相关）
- **会重点看**："下雨"、"不带伞"、"被淋"

这就是**注意力**——翻译每个词时，有选择地关注源语言中相关的部分。

---

## 24.2 注意力机制的核心思想

### 24.2.1 查询-键-值：图书馆隐喻

想象一个智能图书馆系统：

**查询（Query）**：你提出的问题
> "我想了解巧克力蛋糕的做法"

**键（Key）**：每本书的标题/标签
> Book1: "法式甜点大全"
> Book2: "健康素食指南" 
> Book3: "巧克力艺术"

**值（Value）**：书里的实际内容

系统会计算你的查询与每本书标题的**相似度**：
- "巧克力蛋糕" vs "法式甜点大全" → 相似度：0.6
- "巧克力蛋糕" vs "健康素食指南" → 相似度：0.1
- "巧克力蛋糕" vs "巧克力艺术" → 相似度：0.9

然后根据相似度**加权组合**书中的内容：
```
答案 = 0.6 × 法式甜点内容 + 0.1 × 素食内容 + 0.9 × 巧克力内容
```

### 24.2.2 数学表达

注意力机制的完整公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

分解来看：

**Step 1: 计算相似度（点积）**
$$\text{scores} = QK^T$$

这相当于查询与所有键的"匹配分数"。

**Step 2: 缩放（防止softmax饱和）**
$$\text{scaled\_scores} = \frac{\text{scores}}{\sqrt{d_k}}$$

$d_k$是键向量的维度。除以$\sqrt{d_k}$防止点积过大导致softmax梯度消失。

**Step 3: Softmax归一化**
$$\text{weights} = \text{softmax}(\text{scaled\_scores})$$

将分数转换为概率分布（所有权重之和为1）。

**Step 4: 加权求和**
$$\text{output} = \text{weights} \times V$$

---

## 24.3 经典文献深度研究

### 24.3.1 Bahdanau Attention (2014)

**论文**: *Neural Machine Translation by Jointly Learning to Align and Translate*  
**作者**: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio  
**会议**: ICLR 2015

**核心创新**:
1. **Additive Attention**: 使用一个小型前馈网络计算对齐分数
   $$\text{score}(s_t, h_s) = v_a^T \tanh(W_s s_t + W_h h_s)$$

2. **双向RNN编码器**: 同时考虑前文和后文信息

3. **软对齐**: 不硬选择某个词，而是对所有词加权

**贡献**: 首次将注意力机制引入NMT，BLEU分数提升高达5个点。

### 24.3.2 Luong Attention (2015)

**论文**: *Effective Approaches to Attention-based Neural Machine Translation*  
**作者**: Minh-Thang Luong, Hieu Pham, Christopher D. Manning  
**会议**: EMNLP 2015

**核心创新**:
1. **Global Attention**: 关注所有源位置（与Bahdanau类似但更简单）
2. **Local Attention**: 只关注源序列的一个窗口，计算更高效
3. **多种打分函数**:
   - Dot: $\text{score}(s_t, h_s) = s_t^T h_s$
   - General: $\text{score}(s_t, h_s) = s_t^T W h_s$
   - Concat: $\text{score}(s_t, h_s) = v^T \tanh(W_s s_t + W_h h_s)$

**对比**:
| 特性 | Bahdanau | Luong |
|------|---------|-------|
| 对齐位置 | 解码器状态 | 编码器状态 |
| 计算方式 | 相加后非线性 | 直接点积 |
| 归一化 | 在softmax之前 | 在softmax之前 |

### 24.3.3 The Transformer Revolution (2017)

**论文**: *Attention Is All You Need*  
**作者**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin  
**会议**: NeurIPS 2017

**历史地位**: 这篇论文彻底改变了NLP领域，GPT、BERT等模型都基于此架构。

**核心突破**:
1. **完全抛弃RNN/CNN**: 仅用注意力机制
2. **Self-Attention**: 序列中每个位置都能关注所有其他位置
3. **Multi-Head Attention**: 多组并行注意力，捕捉不同关系
4. **位置编码**: 用正弦/余弦函数注入位置信息
5. **完全并行化**: 训练速度比RNN快数十倍

**性能成果**:
- WMT 2014 英德翻译: BLEU 28.4 (SOTA)
- WMT 2014 英法翻译: BLEU 41.8 (SOTA)
- 训练时间: 3.5天 (8块P100)

---

## 24.4 Transformer架构详解

### 24.4.1 整体结构

```
输入 → [编码器] × 6 → [解码器] × 6 → 输出
```

**编码器**（左半部分）：
- 输入嵌入 + 位置编码
- N个相同的编码器层
- 每层 = 多头自注意力 + 前馈网络

**解码器**（右半部分）：
- 输出嵌入 + 位置编码
- N个相同的解码器层
- 每层 = 掩码自注意力 + 交叉注意力 + 前馈网络

### 24.4.2 缩放点积注意力

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**维度说明**:
- $Q \in \mathbb{R}^{n \times d_k}$: 查询矩阵（n个查询，每个$d_k$维）
- $K \in \mathbb{R}^{m \times d_k}$: 键矩阵（m个键）
- $V \in \mathbb{R}^{m \times d_v}$: 值矩阵（m个值，每个$d_v$维）
- 输出: $\mathbb{R}^{n \times d_v}$

**为什么除以$\sqrt{d_k}$？**

当$d_k$较大时，点积的数值会变得很大，softmax函数进入饱和区（梯度极小）。除以$\sqrt{d_k}$将方差控制在合理范围，保持梯度流动。

### 24.4.3 多头注意力

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**直观理解**:

想象8个不同的读者同时阅读一篇文章：
- 读者1关注语法结构
- 读者2关注情感色彩
- 读者3关注实体名称
- ...

每个读者（注意力头）都能学到不同的关系模式，最终综合所有人的见解。

**维度**:
- 原始论文: $d_{model} = 512$, $h = 8$个头
- 每个头: $d_k = d_v = 512 / 8 = 64$

### 24.4.4 位置编码

由于Transformer没有RNN的时序处理，必须显式注入位置信息。

**正弦位置编码**:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中：
- $pos$: 词在序列中的位置
- $i$: 维度索引
- $d_{model}$: 模型维度

**为什么用正弦/余弦？**

1. **唯一性**: 每个位置有独特的编码
2. **相对位置**: 对于固定的偏移$k$，$PE_{pos+k}$可以表示为$PE_{pos}$的线性函数
3. **外推性**: 可以处理训练时未见过的更长序列
4. **有界性**: 值域在[-1, 1]之间

### 24.4.5 前馈网络

每个编码器/解码器层包含一个全连接前馈网络：

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

这是一个两层的MLP，隐藏层维度$d_{ff} = 2048$（4倍于模型维度），使用ReLU激活。

**为什么需要FFN？**

注意力层执行的是**线性组合**（加权求和），而FFN引入**非线性变换**，增强模型的表达能力。

### 24.4.6 残差连接与层归一化

每个子层（注意力或FFN）都采用：

$$\text{LayerNorm}(x + \text{Sublayer}(x))$$

**残差连接（Residual Connection）**:
- 允许梯度直接回传，缓解梯度消失
- 帮助训练深层网络（原始论文6层，后续模型可达100+层）

**层归一化（Layer Normalization）**:
- 对每个样本的所有特征做归一化
- 稳定训练，加速收敛

---

## 24.5 从零实现Transformer

### 24.5.1 基础组件

```python
"""
Transformer 从零实现
包含完整的注意力机制和Transformer架构
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


class ScaledDotProductAttention:
    """
    缩放点积注意力
    
    数学公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    
    def __init__(self, dropout: float = 0.1):
        self.dropout_rate = dropout
        self.attention_weights = None  # 保存用于可视化
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        前向传播
        
        Args:
            Q: 查询矩阵 [batch_size, n_queries, d_k]
            K: 键矩阵 [batch_size, n_keys, d_k]
            V: 值矩阵 [batch_size, n_keys, d_v]
            mask: 可选的掩码矩阵
            
        Returns:
            输出 [batch_size, n_queries, d_v]
        """
        batch_size, n_queries, d_k = Q.shape
        n_keys = K.shape[1]
        
        # Step 1: 计算点积 Q @ K^T
        # [batch, n_queries, d_k] @ [batch, d_k, n_keys] -> [batch, n_queries, n_keys]
        scores = np.matmul(Q, K.transpose(0, 2, 1))
        
        # Step 2: 缩放
        scores = scores / np.sqrt(d_k)
        
        # Step 3: 应用掩码（用于解码器的自注意力，防止看到未来信息）
        if mask is not None:
            scores = scores + mask  # mask中-INF的位置会被softmax变成0
        
        # Step 4: Softmax获取注意力权重
        # 对最后一个维度（keys）做softmax
        attention_weights = self._softmax(scores, axis=-1)
        self.attention_weights = attention_weights
        
        # Step 5: Dropout（简化版，实际应实现随机丢弃）
        if self.training and self.dropout_rate > 0:
            attention_weights = attention_weights * (1 - self.dropout_rate)
        
        # Step 6: 加权求和权重 @ V
        # [batch, n_queries, n_keys] @ [batch, n_keys, d_v] -> [batch, n_queries, d_v]
        output = np.matmul(attention_weights, V)
        
        return output
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """数值稳定的softmax实现"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    @property
    def training(self) -> bool:
        return getattr(self, '_training', False)
    
    def train(self, mode: bool = True):
        self._training = mode
        return self
    
    def eval(self):
        return self.train(False)


class MultiHeadAttention:
    """
    多头注意力机制
    
    将查询、键、值投影到多个子空间，分别计算注意力，最后拼接
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: dropout概率
        """
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 投影矩阵参数
        self.W_Q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_K = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_V = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_O = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        
        # 偏置
        self.b_Q = np.zeros(d_model)
        self.b_K = np.zeros(d_model)
        self.b_V = np.zeros(d_model)
        self.b_O = np.zeros(d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        多头注意力前向传播
        
        Args:
            Q: [batch_size, seq_len_q, d_model]
            K: [batch_size, seq_len_k, d_model]
            V: [batch_size, seq_len_v, d_model]
            mask: 可选掩码
            
        Returns:
            输出 [batch_size, seq_len_q, d_model]
        """
        batch_size = Q.shape[0]
        
        # 1. 线性投影并reshape为多头形式
        # [batch, seq, d_model] -> [batch, seq, num_heads, d_k] -> [batch, num_heads, seq, d_k]
        Q_proj = self._project_and_split(Q, self.W_Q, self.b_Q)
        K_proj = self._project_and_split(K, self.W_K, self.b_K)
        V_proj = self._project_and_split(V, self.W_V, self.b_V)
        
        # 2. 对每个头计算注意力
        # 需要reshape: [batch*num_heads, seq, d_k]以便批量计算
        Q_reshaped = Q_proj.reshape(-1, Q_proj.shape[2], self.d_k)
        K_reshaped = K_proj.reshape(-1, K_proj.shape[2], self.d_k)
        V_reshaped = V_proj.reshape(-1, V_proj.shape[2], self.d_k)
        
        # 调整mask以匹配多头格式
        if mask is not None:
            # 复制mask到每个头
            mask_expanded = np.repeat(mask, self.num_heads, axis=0)
        else:
            mask_expanded = None
        
        attention_output = self.attention.forward(
            Q_reshaped, K_reshaped, V_reshaped, mask_expanded
        )
        
        # 3. reshape回来并拼接
        # [batch*num_heads, seq, d_k] -> [batch, num_heads, seq, d_k] -> [batch, seq, d_model]
        attention_output = attention_output.reshape(
            batch_size, self.num_heads, -1, self.d_k
        ).transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        
        # 4. 最终线性投影
        output = np.matmul(attention_output, self.W_O) + self.b_O
        
        return output
    
    def _project_and_split(self, x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        线性投影并分割成多头
        
        [batch, seq, d_model] -> [batch, num_heads, seq, d_k]
        """
        # 线性投影
        proj = np.matmul(x, W) + b  # [batch, seq, d_model]
        
        # reshape为多头形式
        batch_size, seq_len, _ = proj.shape
        proj = proj.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        proj = proj.transpose(0, 2, 1, 3)  # [batch, num_heads, seq, d_k]
        
        return proj


class PositionalEncoding:
    """
    正弦位置编码
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int = 512, max_len: int = 5000):
        self.d_model = d_model
        
        # 预计算位置编码
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)  # [max_len, 1]
        
        # 计算分母项 10000^(2i/d_model)
        div_term = np.exp(
            np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )  # [d_model/2]
        
        # 正弦位置（偶数维度）
        pe[:, 0::2] = np.sin(position * div_term)
        # 余弦位置（奇数维度）
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe  # [max_len, d_model]
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        添加位置编码
        
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            添加位置编码后的张量
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]
    
    def visualize(self, max_len: int = 100, d_model: int = 64):
        """可视化位置编码"""
        plt.figure(figsize=(12, 6))
        
        pe_subset = self.pe[:max_len, :d_model]
        plt.imshow(pe_subset, cmap='RdBu_r', aspect='auto')
        plt.colorbar(label='Value')
        plt.xlabel('Dimension')
        plt.ylabel('Position')
        plt.title('Positional Encoding Visualization')
        plt.tight_layout()
        plt.savefig('positional_encoding.png', dpi=150)
        plt.show()
        
        return pe_subset


class FeedForwardNetwork:
    """
    位置前馈网络
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        """
        Args:
            d_model: 输入/输出维度
            d_ff: 隐藏层维度（通常是d_model的4倍）
        """
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Xavier初始化
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / (d_model + d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / (d_ff + d_model))
        self.b2 = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, d_model]
            
        Returns:
            输出 [batch_size, seq_len, d_model]
        """
        # 第一层: 线性变换 + ReLU
        hidden = np.maximum(0, np.matmul(x, self.W1) + self.b1)  # ReLU激活
        
        # 第二层: 线性变换
        output = np.matmul(hidden, self.W2) + self.b2
        
        return output


class LayerNorm:
    """
    层归一化
    
    LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
    """
    
    def __init__(self, features: int, eps: float = 1e-6):
        self.eps = eps
        self.gamma = np.ones(features)
        self.beta = np.zeros(features)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        层归一化
        
        Args:
            x: [..., features]
            
        Returns:
            归一化后的张量
        """
        # 计算均值和方差（对最后一个维度）
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # 归一化
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        
        # 缩放和平移
        return self.gamma * x_normalized + self.beta


class EncoderLayer:
    """
    Transformer编码器层
    
    结构: Multi-Head Attention -> Add&Norm -> FFN -> Add&Norm
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, 
                 d_ff: int = 2048, dropout: float = 0.1):
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout_rate = dropout
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        编码器层前向传播
        
        Args:
            x: [batch_size, seq_len, d_model]
            mask: 源序列掩码（用于处理padding）
            
        Returns:
            输出 [batch_size, seq_len, d_model]
        """
        # 1. 多头自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attn.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_output)
        
        # 2. 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward.forward(x)
        x = self.norm2.forward(x + ff_output)
        
        return x


class DecoderLayer:
    """
    Transformer解码器层
    
    结构: Masked Self-Attention -> Add&Norm 
         -> Cross-Attention -> Add&Norm 
         -> FFN -> Add&Norm
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.1):
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout_rate = dropout
    
    def forward(self, x: np.ndarray, encoder_output: np.ndarray,
                src_mask: Optional[np.ndarray] = None,
                tgt_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        解码器层前向传播
        
        Args:
            x: 目标序列 [batch_size, tgt_len, d_model]
            encoder_output: 编码器输出 [batch_size, src_len, d_model]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码（防止看到未来信息）
            
        Returns:
            输出 [batch_size, tgt_len, d_model]
        """
        # 1. 掩码自注意力 + 残差 + 归一化
        self_attn_output = self.self_attn.forward(x, x, x, tgt_mask)
        x = self.norm1.forward(x + self_attn_output)
        
        # 2. 交叉注意力（关注编码器输出）+ 残差 + 归一化
        cross_attn_output = self.cross_attn.forward(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2.forward(x + cross_attn_output)
        
        # 3. 前馈网络 + 残差 + 归一化
        ff_output = self.feed_forward.forward(x)
        x = self.norm3.forward(x + ff_output)
        
        return x


def create_look_ahead_mask(size: int) -> np.ndarray:
    """
    创建前瞻掩码（用于解码器自注意力）
    
    防止解码器在预测位置i时看到位置i之后的信息
    
    例如size=5时:
    [[0, -inf, -inf, -inf, -inf],
     [0,   0,  -inf, -inf, -inf],
     [0,   0,    0,  -inf, -inf],
     [0,   0,    0,    0,  -inf],
     [0,   0,    0,    0,    0]]
    """
    mask = np.triu(np.ones((size, size)), k=1) * (-1e9)
    return mask


# ========== 完整示例 ==========

def demo_attention():
    """演示注意力机制的核心概念"""
    print("=" * 60)
    print("演示: 缩放点积注意力")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 模拟一个简单的翻译场景
    # 源语言: "猫 坐 在 垫子 上" (5个token)
    # 目标语言正在预测第3个词 "the"
    
    batch_size, seq_len, d_k = 1, 5, 8
    
    # 查询: 当前要预测的词的状态
    Q = np.random.randn(batch_size, 1, d_k)  # 1个查询
    
    # 键和值: 源语言所有词的表示
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)
    
    # 给每个词打上标签方便理解
    tokens = ["猫", "坐", "在", "垫子", "上"]
    
    attention = ScaledDotProductAttention()
    output = attention.forward(Q, K, V)
    
    print(f"\n源语言token: {tokens}")
    print(f"\n注意力权重:")
    weights = attention.attention_weights[0, 0, :]
    for token, weight in zip(tokens, weights):
        bar = "█" * int(weight * 50)
        print(f"  {token:4s}: {weight:.4f} {bar}")
    
    print(f"\n输出维度: {output.shape}")
    print("=" * 60)


def demo_multi_head_attention():
    """演示多头注意力"""
    print("\n" + "=" * 60)
    print("演示: 多头注意力")
    print("=" * 60)
    
    np.random.seed(42)
    
    batch_size, seq_len = 2, 10
    d_model = 64
    num_heads = 8
    
    # 随机输入
    x = np.random.randn(batch_size, seq_len, d_model)
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    output = mha.forward(x, x, x)
    
    print(f"\n输入形状:  {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型维度: {d_model}")
    print(f"注意力头数: {num_heads}")
    print(f"每头维度: {d_model // num_heads}")
    print("=" * 60)


def demo_transformer_layer():
    """演示完整的Transformer编码器层"""
    print("\n" + "=" * 60)
    print("演示: Transformer编码器层")
    print("=" * 60)
    
    np.random.seed(42)
    
    batch_size = 2
    seq_len = 10
    d_model = 64
    
    # 输入嵌入
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # 添加位置编码
    pos_enc = PositionalEncoding(d_model=d_model, max_len=100)
    x_with_pos = pos_enc.forward(x)
    
    print(f"\n原始输入形状: {x.shape}")
    print(f"添加位置编码后: {x_with_pos.shape}")
    
    # 编码器层
    encoder = EncoderLayer(d_model=d_model, num_heads=4, d_ff=256)
    output = encoder.forward(x_with_pos)
    
    print(f"\n编码器输出形状: {output.shape}")
    print(f"参数量估计: ~{((d_model * d_model * 4) + (d_model * 256 * 2)) / 1000:.1f}K")
    print("=" * 60)


def visualize_attention_matrix():
    """可视化注意力矩阵"""
    print("\n" + "=" * 60)
    print("可视化: 注意力权重矩阵")
    print("=" * 60)
    
    # 模拟翻译任务中的注意力
    source = ["The", "cat", "sat", "on", "the", "mat"]
    target = ["猫", "坐", "在", "垫子", "上"]
    
    np.random.seed(42)
    
    # 创建模拟的注意力权重（真实的应该来自模型）
    # 让这个看起来合理一些
    attention_weights = np.random.rand(len(target), len(source)) * 0.3
    
    # 添加一些合理的对齐
    attention_weights[0, 1] = 0.8  # "猫" -> "cat"
    attention_weights[1, 2] = 0.8  # "坐" -> "sat"
    attention_weights[2, 0] = 0.3  # "在" -> "The" (弱化)
    attention_weights[2, 3] = 0.5  # "在" -> "on"
    attention_weights[3, 5] = 0.8  # "垫子" -> "mat"
    attention_weights[4, 4] = 0.6  # "上" -> "the"
    attention_weights[4, 5] = 0.3  # "上" -> "mat"
    
    # 归一化
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights, cmap='Blues', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xticks(range(len(source)), source)
    plt.yticks(range(len(target)), target)
    plt.xlabel('Source (English)')
    plt.ylabel('Target (Chinese)')
    plt.title('Attention Alignment Visualization')
    
    # 添加数值标注
    for i in range(len(target)):
        for j in range(len(source)):
            plt.text(j, i, f'{attention_weights[i, j]:.2f}',
                    ha='center', va='center', color='red', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('attention_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n注意力矩阵已保存到 attention_matrix.png")
    print("=" * 60)


if __name__ == "__main__":
    demo_attention()
    demo_multi_head_attention()
    demo_transformer_layer()
    visualize_attention_matrix()
