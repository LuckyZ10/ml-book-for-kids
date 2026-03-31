# 第二十三章 循环神经网络 练习题

## 练习题 1: RNN前向传播手算 (⭐)

**目标**: 理解RNN的前向计算

**题目**: 
给定一个简单的RNN单元：

```
h_t = tanh(W_ih · x_t + W_hh · h_{t-1} + b)
y_t = W_ho · h_t
```

参数：
- W_ih = [[0.5]] (输入到隐藏)
- W_hh = [[0.8]] (隐藏到隐藏)
- W_ho = [[1.0]] (隐藏到输出)
- b = [0.0]
- h_0 = [0.0]

输入序列：x = [1, 2, 3]

**任务**: 
1. 手动计算h_1, h_2, h_3
2. 计算y_1, y_2, y_3
3. 观察隐藏状态如何随时间变化

**提示**: tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})

---

## 练习题 2: BPTT梯度计算 (⭐⭐)

**目标**: 理解BPTT的梯度回传

**题目**: 
对于练习1的RNN，假设损失函数为：

$$L = \frac{1}{2}(y_3 - t)^2$$

其中 t = 5（目标值）

**任务**: 
1. 计算损失对W_ho的梯度
2. 计算损失对W_hh的梯度（考虑时间依赖）
3. 解释梯度消失问题（如果序列更长会怎样？）

---

## 练习题 3: NumPy实现RNN (⭐⭐⭐)

**目标**: 从零实现RNN

**题目**: 
实现一个简单的RNN用于字符级语言模型：

```python
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        
        # 初始化权重
        self.W_ih = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_ho = np.random.randn(output_size, hidden_size) * 0.01
        
        self.b_h = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((output_size, 1))
    
    def forward(self, inputs):
        """
        前向传播
        inputs: 长度为T的输入序列，每个元素是one-hot向量
        返回: 隐藏状态序列和输出序列
        """
        # 你的实现
        pass
    
    def backward(self, targets, learning_rate=0.1):
        """
        BPTT反向传播
        """
        # 你的实现
        pass
```

**要求**: 
1. 完成forward和backward方法
2. 使用给定的文本训练（如"hello world"）
3. 让模型学会生成类似文本

---

## 练习题 4: LSTM vs GRU对比 (⭐⭐)

**目标**: 理解门控机制

**题目**: 
完成下表比较LSTM和GRU：

| 特性 | LSTM | GRU |
|-----|------|-----|
| 门数量 | ? | ? |
| 遗忘门 | ? | ? |
| 细胞状态 | ? | ? |
| 参数量 | ? | ? |
| 计算速度 | ? | ? |

**实验**: 
在同一任务（如情感分类）上对比LSTM和GRU：
- 训练时间
- 最终准确率
- 模型大小

**思考问题**: 
- 什么时候选LSTM？什么时候选GRU？
- 为什么GRU可以"融合"遗忘门和输入门？

---

## 练习题 5: 序列到序列模型 (⭐⭐⭐)

**目标**: 实现Seq2Seq基础

**题目**: 
实现一个简单的编码器-解码器模型：

```python
class Encoder:
    def __init__(self, input_size, hidden_size):
        self.rnn = LSTM(input_size, hidden_size)
    
    def forward(self, input_seq):
        # 返回最后一个隐藏状态（上下文向量）
        pass

class Decoder:
    def __init__(self, output_size, hidden_size):
        self.rnn = LSTM(output_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, context, target_seq=None):
        # 使用上下文向量开始解码
        # 如果是训练，使用teacher forcing
        # 如果是推理，使用前一步的输出作为下一步输入
        pass
```

**任务**: 
1. 实现简单的英法翻译（数据：hello→bonjour, world→monde）
2. 观察上下文向量的作用
3. 尝试更长的句子

---

## 练习题 6: 梯度消失实验 (⭐⭐⭐)

**目标**: 直观感受梯度消失

**题目**: 
实现梯度流可视化：

```python
def compute_gradient_norms(model, seq_length):
    """
    计算每一层（时间步）的梯度范数
    """
    # 生成长序列输入
    x = np.random.randn(seq_length, input_size)
    
    # 前向传播
    # ...
    
    # 反向传播
    # ...
    
    # 返回每个时间步的梯度范数
    return gradient_norms
```

**实验**: 
1. 比较tanh和ReLU作为激活函数的梯度流
2. 比较有/无LSTM的梯度流
3. 绘制"时间步-梯度范数"图

**预期结果**: 
- 普通RNN：梯度随时间指数衰减
- LSTM：梯度更稳定

---

## 练习题 7: 文本生成器 (⭐⭐⭐)

**目标**: 构建完整的文本生成系统

**题目**: 
使用LSTM训练一个字符级文本生成器：

**步骤**: 
1. 准备数据（下载莎士比亚/鲁迅文本）
2. 构建字符词汇表
3. 训练LSTM语言模型
4. 实现生成函数（温度采样）

**温度采样代码**: 
```python
def generate(model, seed_text, length=500, temperature=1.0):
    result = seed_text
    
    for _ in range(length):
        # 准备输入
        x = char_to_tensor(result[-1])
        
        # 预测
        output = model(x)
        
        # 温度缩放
        output = output / temperature
        
        # 采样
        probs = softmax(output)
        next_char = sample(probs)
        
        result += next_char
    
    return result
```

**实验**: 
- 尝试不同温度（0.5, 1.0, 2.0）
- 观察生成文本的多样性和质量

---

## 练习题 8: 时间序列预测 (⭐⭐⭐)

**目标**: 应用RNN到实际问题

**题目**: 
使用LSTM预测股票价格（或气温数据）：

```python
class StockPredictor:
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=2):
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])  # 只取最后一个时间步
        return predictions
```

**要求**: 
1. 数据预处理（归一化、划分训练/测试集）
2. 创建滑动窗口输入
3. 训练模型并评估
4. 可视化预测结果

**思考问题**: 
- 为什么股票预测这么难？
- RNN能学到什么模式？

---

## 练习题 9: RNN vs Transformer对比 (⭐⭐⭐⭐)

**目标**: 理解RNN的局限和Transformer的优势

**题目**: 
在同一文本分类任务上对比：

1. **RNN模型** (LSTM)
2. **Transformer模型** (使用PyTorch内置)

**对比维度**: 
| 维度 | RNN | Transformer |
|-----|-----|-------------|
| 训练速度 | ? | ? |
| 长序列性能 | ? | ? |
| 参数量 | ? | ? |
| 内存占用 | ? | ? |
| 可解释性 | ? | ? |

**分析**: 
- 为什么Transformer在长序列上更好？
- 什么情况下RNN仍有优势？

---

## 参考答案

### 练习1 RNN前向计算

**解答**: 
- h_1 = tanh(0.5*1 + 0.8*0) = tanh(0.5) ≈ 0.462
- h_2 = tanh(0.5*2 + 0.8*0.462) = tanh(1.37) ≈ 0.879
- h_3 = tanh(0.5*3 + 0.8*0.879) = tanh(2.20) ≈ 0.976

### 练习3 NumPy RNN提示

关键部分:
```python
def forward(self, inputs):
    h = np.zeros((self.hidden_size, 1))
    self.hidden_states = [h]
    self.inputs = inputs
    
    for x in inputs:
        h = np.tanh(np.dot(self.W_ih, x) + 
                    np.dot(self.W_hh, h) + self.b_h)
        self.hidden_states.append(h)
    
    return h
```

---

**学习建议**: 
- 练习1-3必做，理解RNN基础
- 练习4-6进阶，理解门控和梯度
- 练习7-9实战，掌握应用

**推荐资源**: 
- "Understanding LSTM Networks" (Christopher Olah博客)
- PyTorch官方RNN教程
- Kaggle时间序列竞赛
