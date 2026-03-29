# ML教材术语一致性分析报告

## 📊 执行摘要

- **分析文件总数**: 135
- **跟踪术语数量**: 42
- **发现问题文件数**: 319

## 1. 术语使用频率统计表

| 术语 | 出现次数 | 使用变体 | 主要问题 |
|------|----------|----------|----------|
| 特征 | 943 | 特征, feature | 多变体共存 |
| 神经网络 | 577 | 神经网络, neural network, neural networks | 多变体共存 |
| 机器学习 | 487 | 机器学习, ml, machine learning | 多变体共存 |
| 卷积 | 349 | 卷积, convolution | 多变体共存 |
| 感知机 | 342 | 感知机, perceptron | 多变体共存 |
| 推理 | 332 | inference, 推理 | 多变体共存 |
| 学习率 | 233 | 学习率, learning rate | 多变体共存 |
| 损失函数 | 225 | cost function, 损失函数, 损失函数 等5个变体 | 多变体共存 |
| 反向传播 | 222 | 反向传播, bp, backpropagation | 多变体共存 |
| 预训练 | 210 | 预训练, pre-training, pre-training | 多变体共存 |
| 注意力机制 | 198 | 注意力机制, attention | 多变体共存 |
| 深度学习 | 195 | 深度学习, deep learning | 多变体共存 |
| 正则化 | 177 | regularization, 正则化 | 多变体共存 |
| 神经元 | 175 | 神经元 |  |
| epoch | 167 | 轮次, 轮, epoch | 多变体共存 |
| 梯度下降 | 159 | 梯度下降, gradient descent | 多变体共存 |
| 激活函数 | 150 | activation function, 激活函数 | 多变体共存 |
| 微调 | 138 | 微调, fine-tuning | 多变体共存 |
| 过拟合 | 126 | overfitting, 过拟合 | 多变体共存 |
| 超参数 | 120 | 超参数, hyperparameter | 多变体共存 |
| 嵌入 | 114 | 嵌入, 嵌入, embedding | 多变体共存 |
| 归一化 | 113 | normalization, bn, 层归一化 等5个变体 | 多变体共存 |
| 池化 | 111 | 池化, pooling | 多变体共存 |
| 标签 | 106 | 标签, label | 多变体共存 |
| 部署 | 89 | 部署, deployment | 多变体共存 |
| 人工智能 | 71 | 人工智能, ai, artificial intelligence | 多变体共存 |
| 优化器 | 69 | optimizer, 优化器 | 多变体共存 |
| transformer | 58 | 变换器, 转换器, transformer | 多变体共存 |
| 批量 | 57 | batch, 批量, 批量 | 多变体共存 |
| 自注意力 | 56 | self-attention, 自注意力 | 多变体共存 |
| 全连接 | 55 | 全连接, fully connected, dense | 多变体共存 |
| relu | 48 | relu |  |
| softmax | 41 | softmax |  |
| 循环神经网络 | 39 | rnn, recurrent neural network, 循环神经网络 | 多变体共存 |
| 长短期记忆 | 35 | lstm, long short-term memory, 长短期记忆 | 多变体共存 |
| 卷积神经网络 | 33 | convolutional neural network, 卷积神经网络, cnn | 多变体共存 |
| 多层感知机 | 30 | 多层感知机, mlp | 多变体共存 |
| 训练集 | 29 | 训练集 |  |
| dropout | 19 | dropout |  |
| 欠拟合 | 12 | underfitting, 欠拟合 | 多变体共存 |
| 测试集 | 12 | 测试集 |  |
| 验证集 | 11 | 验证集 |  |


## 2. 术语不一致问题列表

### 2.1 需要统一的多变体术语

以下术语在同一文件中使用了多个变体，建议统一：

#### 机器学习
- **影响文件数**: 51
- **发现的变体**: ml, machine learning, 机器学习
- **建议统一为**: 机器学习

#### 神经网络
- **影响文件数**: 32
- **发现的变体**: 神经网络, neural network, neural networks
- **建议统一为**: 神经网络

#### 特征
- **影响文件数**: 19
- **发现的变体**: feature, 特征
- **建议统一为**: 特征

#### 深度学习
- **影响文件数**: 18
- **发现的变体**: 深度学习, deep learning
- **建议统一为**: 深度学习

#### 损失函数
- **影响文件数**: 18
- **发现的变体**: loss function, cost function, 损失函数, 损失函数, 损失函数
- **建议统一为**: 损失函数

#### 注意力机制
- **影响文件数**: 16
- **发现的变体**: 注意力机制, attention
- **建议统一为**: 注意力机制

#### 归一化
- **影响文件数**: 13
- **发现的变体**: 归一化, bn, 层归一化, normalization, batch normalization
- **建议统一为**: 归一化

#### 嵌入
- **影响文件数**: 13
- **发现的变体**: 嵌入, embedding, 嵌入
- **建议统一为**: 嵌入

#### 反向传播
- **影响文件数**: 12
- **发现的变体**: 反向传播, backpropagation, bp
- **建议统一为**: 反向传播

#### 预训练
- **影响文件数**: 11
- **发现的变体**: pre-training, pre-training, 预训练
- **建议统一为**: 预训练

#### 正则化
- **影响文件数**: 8
- **发现的变体**: 正则化, regularization
- **建议统一为**: 正则化

#### 推理
- **影响文件数**: 8
- **发现的变体**: 推理, inference
- **建议统一为**: 推理

#### 批量
- **影响文件数**: 7
- **发现的变体**: batch, 批量, 批量
- **建议统一为**: 批量

#### 人工智能
- **影响文件数**: 7
- **发现的变体**: artificial intelligence, ai, 人工智能
- **建议统一为**: 人工智能

#### 卷积神经网络
- **影响文件数**: 7
- **发现的变体**: cnn, 卷积神经网络, convolutional neural network
- **建议统一为**: 卷积神经网络

#### 学习率
- **影响文件数**: 6
- **发现的变体**: learning rate, 学习率
- **建议统一为**: 学习率

#### 感知机
- **影响文件数**: 6
- **发现的变体**: perceptron, 感知机
- **建议统一为**: 感知机

#### 自注意力
- **影响文件数**: 6
- **发现的变体**: 自注意力, self-attention
- **建议统一为**: 自注意力

#### 池化
- **影响文件数**: 6
- **发现的变体**: 池化, pooling
- **建议统一为**: 池化

#### 部署
- **影响文件数**: 6
- **发现的变体**: deployment, 部署
- **建议统一为**: 部署

#### 过拟合
- **影响文件数**: 5
- **发现的变体**: overfitting, 过拟合
- **建议统一为**: 过拟合

#### 梯度下降
- **影响文件数**: 5
- **发现的变体**: 梯度下降, gradient descent
- **建议统一为**: 梯度下降

#### 微调
- **影响文件数**: 5
- **发现的变体**: 微调, fine-tuning
- **建议统一为**: 微调

#### 循环神经网络
- **影响文件数**: 5
- **发现的变体**: 循环神经网络, rnn, recurrent neural network
- **建议统一为**: 循环神经网络

#### 多层感知机
- **影响文件数**: 4
- **发现的变体**: 多层感知机, mlp
- **建议统一为**: 多层感知机

#### 激活函数
- **影响文件数**: 4
- **发现的变体**: activation function, 激活函数
- **建议统一为**: 激活函数

#### 卷积
- **影响文件数**: 4
- **发现的变体**: convolution, 卷积
- **建议统一为**: 卷积

#### 标签
- **影响文件数**: 3
- **发现的变体**: label, 标签
- **建议统一为**: 标签

#### epoch
- **影响文件数**: 3
- **发现的变体**: 轮次, epoch, 轮
- **建议统一为**: epoch

#### 全连接
- **影响文件数**: 3
- **发现的变体**: dense, fully connected, 全连接
- **建议统一为**: 全连接


## 3. 建议的统一术语标准表

| 标准术语 | 推荐中文 | 推荐英文 | 不推荐使用 | 说明 |
|----------|----------|----------|------------|------|
| 神经网络 | 神经网络 | Neural Network | 神经网路 | 避免错别字 |
| 梯度下降 | 梯度下降 | Gradient Descent | 梯度下將 | 避免错别字 |
| 反向传播 | 反向传播 | Backpropagation | 反向傳播 | 使用简体 |
| 损失函数 | 损失函数 | Loss Function | 损失函数,损失函数 | 统一术语 |
| 激活函数 | 激活函数 | Activation Function | 激励函数 | 统一术语 |
| 优化器 | 优化器 | Optimizer | optimiser | 使用美式拼写 |
| 正则化 | 正则化 | Regularization | regularisation | 使用美式拼写 |
| 卷积神经网络 | 卷积神经网络 | CNN | - | 首次出现用全称 |
| 循环神经网络 | 循环神经网络 | RNN | - | 首次出现用全称 |
| Transformer | Transformer | Transformer | 变换器,转换器 | 保留英文 |
| 注意力机制 | 注意力机制 | Attention | - | 可中英文并用 |
| 嵌入 | 嵌入 | Embedding | 嵌入 | 统一用嵌入 |
| 特征 | 特征 | Feature | - | 可中英文并用 |
| 标签 | 标签 | Label | - | 可中英文并用 |
| 训练集 | 训练集 | Training Set | - | 可中英文并用 |
| 验证集 | 验证集 | Validation Set | - | 可中英文并用 |
| 测试集 | 测试集 | Test Set | - | 可中英文并用 |
| 过拟合 | 过拟合 | Overfitting | 过适配 | 统一术语 |
| 欠拟合 | 欠拟合 | Underfitting | 欠适配 | 统一术语 |
| 超参数 | 超参数 | Hyperparameter | - | 可中英文并用 |
| 批量 | 批量 | Batch | 批量 | 统一用批量 |
| 学习率 | 学习率 | Learning Rate | - | 可中英文并用 |
| epoch | epoch/轮次 | Epoch | 轮 | 建议保留英文或统一用轮次 |
| 推理 | 推理 | Inference | - | 可中英文并用 |
| 部署 | 部署 | Deployment | - | 可中英文并用 |
| 微调 | 微调 | Fine-tuning | fine-tuning | 使用连字符 |
| 预训练 | 预训练 | Pre-training | pre-training | 使用连字符 |
| 感知机 | 感知机 | Perceptron | 感知机 | 统一用感知机 |
| 归一化 | 归一化 | Normalization | - | 可中英文并用 |
| 全连接 | 全连接 | Fully Connected | dense | 中文优先 |
| 池化 | 池化 | Pooling | - | 可中英文并用 |
| 自注意力 | 自注意力 | Self-Attention | - | 可中英文并用 |

## 4. 文件级问题清单

- **book/chapter21_regularization.md**: 神经网络 → '神经网络'(1次), 'neural networks'(1次) 
- **book/chapter21_regularization.md**: 正则化 → '正则化'(10次), 'regularization'(1次) 
- **book/chapter21_regularization.md**: 过拟合 → '过拟合'(1次), 'overfitting'(1次) 
- **book/chapter21_regularization.md**: 特征 → '特征'(5次), 'feature'(1次) 
- **book/chapter21_regularization.md**: 归一化 → 'normalization'(1次), 'batch normalization'(1次) 
- **book/chapter21_regularization_part1.md**: 正则化 → '正则化'(19次), 'regularization'(3次) 
- **book/chapter21_regularization_part1.md**: 过拟合 → '过拟合'(13次), 'overfitting'(2次) 
- **book/chapter21_regularization_part1.md**: 欠拟合 → '欠拟合'(1次), 'underfitting'(1次) 
- **chapter-03-prediction-and-loss.md**: 机器学习 → '机器学习'(4次), 'machine learning'(1次) 
- **chapter-03-prediction-and-loss.md**: 深度学习 → '深度学习'(1次), 'deep learning'(1次) 
- **chapter-04-gradient-descent.md**: 梯度下降 → '梯度下降'(13次), 'gradient descent'(2次) 
- **chapter-04-gradient-descent.md**: 学习率 → '学习率'(23次), 'learning rate'(1次) 
- **chapter-04-gradient-descent.md**: 批量 → '批量'(3次), 'batch'(1次) 
- **chapter-04/chapter-04.md**: 深度学习 → '深度学习'(5次), 'deep learning'(1次) 
- **chapter-04/chapter-04.md**: 梯度下降 → '梯度下降'(31次), 'gradient descent'(1次) 
- **chapter-04/chapter-04.md**: 损失函数 → '损失函数'(4次), 'loss function'(1次) 
- **chapter-04/chapter-04.md**: 学习率 → '学习率'(27次), 'learning rate'(2次) 
- **chapter-05-python-warmup.md**: 机器学习 → '机器学习'(11次), 'ml'(1次) 
- **chapter-06-knn.md**: 机器学习 → '机器学习'(1次), 'machine learning'(1次) 
- **chapter-06-knn.md**: 特征 → '特征'(4次), 'feature'(1次) 
- **chapter-07-linear-regression.md**: 损失函数 → '损失函数'(6次), 'loss function'(1次), 'cost function'(1次), '损失函数'(1次) ⚠️ 包含错误/不推荐使用写法
- **chapter-08.md**: 机器学习 → '机器学习'(1次), 'machine learning'(2次) 
- **chapter-09-decision-tree/README.md**: 机器学习 → '机器学习'(2次), 'machine learning'(3次) 
- **chapter-09.md**: 机器学习 → '机器学习'(2次), 'machine learning'(3次) 
- **chapter-09.md**: 过拟合 → '过拟合'(8次), 'overfitting'(1次) 
- **chapter-12-ensemble/manuscript.md**: 机器学习 → '机器学习'(5次), 'machine learning'(2次) 
- **chapter-12-ensemble/manuscript.md**: 损失函数 → '损失函数'(2次), '损失函数'(1次) ⚠️ 包含错误/不推荐使用写法
- **chapter-15-dimensionality-reduction/chapter-15.md**: 机器学习 → '机器学习'(1次), 'machine learning'(2次) 
- **chapter-15-dimensionality-reduction/chapter-15.md**: 损失函数 → '损失函数'(1次), '损失函数'(2次) ⚠️ 包含错误/不推荐使用写法
- **chapter-16-perceptron.md**: 感知机 → '感知机'(56次), 'perceptron'(24次) 
- **chapter-16-perceptron.md**: 标签 → '标签'(8次), 'label'(4次) 
- **chapter-16-perceptron.md**: epoch → 'epoch'(11次), '轮次'(5次), '轮'(11次) ⚠️ 包含错误/不推荐使用写法
- **chapter-16-perceptron/chapter-16-perceptron.md**: 人工智能 → '人工智能'(1次), 'ai'(1次) 
- **chapter-16-perceptron/chapter-16-perceptron.md**: 感知机 → '感知机'(48次), 'perceptron'(4次) 
- **chapter-16-perceptron/chapter-16-perceptron.md**: 多层感知机 → '多层感知机'(2次), 'mlp'(1次) 
- **chapter-16-perceptron/chapter-16-perceptron.md**: 反向传播 → '反向传播'(3次), 'backpropagation'(1次) 
- **chapter-16-perceptron/chapter-16-perceptron.md**: 学习率 → '学习率'(8次), 'learning rate'(1次) 
- **chapter-17-multilayer-neural-network.md**: 深度学习 → '深度学习'(10次), 'deep learning'(2次) 
- **chapter-17-multilayer-neural-network.md**: 人工智能 → '人工智能'(1次), 'artificial intelligence'(1次) 
- **chapter-17-multilayer-neural-network.md**: 神经网络 → '神经网络'(38次), 'neural networks'(3次) 
- **chapter-17-multilayer-neural-network.md**: 感知机 → '感知机'(28次), 'perceptron'(3次) 
- **chapter-17-multilayer-neural-network.md**: 多层感知机 → '多层感知机'(3次), 'mlp'(1次) 
- **chapter-17-multilayer-neural-network.md**: 反向传播 → '反向传播'(28次), 'backpropagation'(2次) 
- **chapter-17-multilayer-neural-network.md**: 激活函数 → '激活函数'(33次), 'activation function'(1次) 
- **chapter-17-multilayer-neural-network.md**: 批量 → '批量'(2次), 'batch'(1次), '批量'(1次) ⚠️ 包含错误/不推荐使用写法
- **chapter-17-multilayer-neural-network.md**: 全连接 → '全连接'(1次), 'fully connected'(1次), 'dense'(1次) 
- **chapter-19-activation-functions/chapter-19.md**: 激活函数 → '激活函数'(19次), 'activation function'(1次) 
- **chapter-24-attention-transformer/README.md**: 机器学习 → '机器学习'(1次), 'machine learning'(2次) 
- **chapter-24-attention-transformer/README.md**: 注意力机制 → '注意力机制'(2次), 'attention'(17次) 
- **chapter-24-attention-transformer/README.md**: 自注意力 → '自注意力'(1次), 'self-attention'(6次) 


## 5. 修正建议

### 高优先级修正
1. **错别字修正**:
   - 神经网路 → 神经网络
   - 梯度下將 → 梯度下降  
   - 反向傳播 → 反向传播

2. **术语统一**:
   - 损失函数 vs 损失函数 vs 损失函数 → 统一用损失函数
   - 批量 vs 批量 → 统一用批量
   - 感知机 vs 感知机 → 统一用感知机
   - epoch vs 轮次 vs 轮 → 建议保留epoch或统一用轮次

3. **拼写统一**:
   - fine-tuning → fine-tuning
   - pre-training → pre-training

### 中等优先级
- 中英文混用规范化（建议首次出现用中英文对照，后续用中文）
- 大小写统一（如 transformer → Transformer）

---
*报告生成时间: 2025-03-27*
