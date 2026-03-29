# 第二十四章：注意力机制与Transformer——全局的观察者

> 🎯 **章节定位**：本章是深度学习从"序列"到"全局"的跃迁。我们将告别RNN的顺序处理，迎接Transformer的并行革命——这是GPT、BERT等大模型的基石。

---

## 为什么学习本章？

如果你只学一章深度学习的内容，**应该是这一章**。

为什么？因为：
- 🏆 **Transformer** 是目前几乎所有大语言模型（GPT、Claude、Llama等）的基础架构
- ⚡ **Attention机制** 解决了RNN无法并行、长距离依赖差的问题
- 🌍 **BERT** 和 **GPT** 分别统治了自然语言理解的半壁江山
- 🔮 理解了Transformer，你就理解了现代AI的"引擎"

---

## 本章内容导航

### 1. 理论篇
| 章节 | 核心内容 | 预计阅读时间 |
|------|----------|--------------|
| 24.1 | RNN的困境与Attention的诞生 | 20分钟 |
| 24.2 | 经典文献：从Bahdanau到Transformer | 30分钟 |
| 24.3 | Self-Attention：自注意力机制 | 40分钟 |
| 24.4 | Multi-Head Attention：多头注意力 | 30分钟 |
| 24.5 | Positional Encoding：位置编码 | 25分钟 |
| 24.6 | Encoder-Decoder：编解码器协同 | 30分钟 |
| 24.7 | Transformer家族：BERT、GPT、T5 | 20分钟 |

### 2. 实践篇
| 章节 | 核心内容 | 预计时间 |
|------|----------|----------|
| 24.8 | 从零实现Transformer | 60分钟 |
| 24.9 | 机器翻译Demo | 30分钟 |
| 24.10 | 文本生成器 | 20分钟 |
| 24.11 | 注意力可视化 | 15分钟 |

### 3. 练习与巩固
| 章节 | 内容 |
|------|------|
| 24.12 | 本章小结 |
| 24.13 | 练习题（8道：3基础+3进阶+2挑战） |
| 24.14 | 参考文献（12篇经典论文） |

---

## 前置知识

在学习本章之前，建议掌握：
- ✅ 线性代数（矩阵乘法、点积）
- ✅ 神经网络基础（前向传播、反向传播）
- ✅ RNN/LSTM基础（第二十三章内容）

---

## 核心公式速查

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Multi-Head Attention:**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**Positional Encoding:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

---

## 学习路径图

```
第二十三章：RNN/LSTM ──┬──> 理解序列建模的挑战
                       │
第二十四章：Transformer ─┼──> 自注意力机制
                       │
                       ├──> 位置编码
                       │
                       ├──> 编解码器架构
                       │
                       └──> BERT/GPT/T5
                              │
                              v
                        现代大语言模型
```

---

## 参考资源

### 必读论文
1. Vaswani et al. (2017) - "Attention Is All You Need" ⭐
2. Bahdanau et al. (2015) - Neural Machine Translation with Attention
3. Devlin et al. (2019) - BERT
4. Brown et al. (2020) - GPT-3

### 可视化工具
- [Transformer可视化](http://jalammar.github.io/illustrated-transformer/)
- [Attention可视化](https://github.com/jessevig/bertviz)

---

## 开始阅读

👉 点击阅读：[chapter24_attention_transformer.md](./chapter24_attention_transformer.md)

准备好了吗？让我们进入深度学习最激动人心的架构——**Transformer**！
