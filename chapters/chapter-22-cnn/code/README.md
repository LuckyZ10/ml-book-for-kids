# CNN纯NumPy实现

本目录包含使用纯NumPy实现的卷积神经网络(CNN)，用于《机器学习与深度学习：从小学生到大师》第二十二章的教学。

## 文件说明

| 文件 | 说明 |
|------|------|
| `layers.py` | CNN基础层实现（Conv2D、MaxPool、Dense等） |
| `lenet.py` | LeNet模型实现 |
| `train_mnist.py` | MNIST训练脚本 |
| `visualize.py` | 可视化工具 |
| `full_example.py` | 完整使用示例 |
| `utils.py` | 辅助工具函数 |
| `test_layers.py` | 单元测试 |

## 快速开始

### 1. 运行完整示例

```bash
python full_example.py
```

这将：
- 下载MNIST数据集
- 训练LeNet模型
- 生成可视化结果
- 保存到 `./results/` 目录

### 2. 快速测试（3个epoch）

```bash
python full_example.py --quick
```

### 3. 单独训练

```bash
python train_mnist.py
```

### 4. 运行单元测试

```bash
python test_layers.py
```

## 项目结构

```
code/
├── layers.py          # 核心层实现
├── lenet.py           # LeNet模型
├── train_mnist.py     # 训练脚本
├── visualize.py       # 可视化
├── full_example.py    # 完整示例
├── utils.py           # 工具函数
└── test_layers.py     # 测试
```

## 实现的层

### Conv2D
- 前向传播：二维卷积
- 反向传播：计算dW、db、dX
- 支持多通道、步长、填充

### MaxPooling2D
- 前向：最大池化
- 反向：梯度路由到最大值位置
- 支持自定义池化窗口和步长

### Dense
- 全连接层
- 前向：y = xW + b
- 反向：计算各梯度

### 激活函数
- ReLU
- Sigmoid

### 优化器
- SGD
- Adam

## 模型架构（LeNet）

```
输入 (1, 28, 28)
  ↓
Conv2D(1→6, 5×5, padding=2) + ReLU
  ↓
MaxPool(2×2, stride=2)
  ↓
Conv2D(6→16, 5×5) + ReLU
  ↓
MaxPool(2×2, stride=2)
  ↓
Conv2D(16→120, 5×5) + ReLU
  ↓
Flatten
  ↓
Dense(120→84) + ReLU
  ↓
Dense(84→10)
  ↓
Softmax

总参数: ~60,000
```

## 依赖

```
numpy
matplotlib
```

## 示例输出

训练10个epoch后，测试集准确率应达到 **95%+**。

## 学习建议

1. 先阅读 `layers.py` 理解各层实现
2. 运行 `test_layers.py` 验证实现正确性
3. 运行 `full_example.py` 看完整流程
4. 修改超参数观察效果
5. 尝试修改网络结构

## 扩展练习

1. **添加Batch Normalization层**
2. **实现Dropout**
3. **添加学习率衰减**
4. **实现数据增强（旋转、平移）**
5. **在CIFAR-10数据集上测试**

## 许可证

本代码仅供教学使用。
