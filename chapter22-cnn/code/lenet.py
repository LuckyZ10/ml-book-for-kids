"""
lenet.py - LeNet风格的CNN实现

实现经典的LeNet-5架构，用于MNIST手写数字识别。
架构: Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> Conv -> ReLU -> Flatten -> Dense -> ReLU -> Dense

输入: (batch, 1, 28, 28)
输出: (batch, 10) - 10个数字类别的概率
"""

import numpy as np
from layers import (Layer, Conv2D, MaxPooling2D, Flatten, 
                    Dense, ReLU, Sigmoid, SoftmaxCrossEntropy)


class LeNet:
    """
    LeNet风格的卷积神经网络
    
    基于Yann LeCun 1998年的经典架构，适用于MNIST手写数字识别。
    
    网络结构:
    1. C1: Conv2D(1->6, 5x5, padding=2) + ReLU
    2. S2: MaxPool2D(2x2, stride=2)
    3. C3: Conv2D(6->16, 5x5) + ReLU
    4. S4: MaxPool2D(2x2, stride=2)
    5. C5: Conv2D(16->120, 5x5) + ReLU
    6. Flatten
    7. F6: Dense(120->84) + ReLU
    8. Output: Dense(84->num_classes)
    
    参数:
        num_classes: 输出类别数，默认10（MNIST数字0-9）
        use_sigmoid: 是否使用Sigmoid代替ReLU（原始LeNet使用Sigmoid）
    """
    def __init__(self, num_classes=10, use_sigmoid=False):
        self.layers = []
        self.num_classes = num_classes
        
        # 选择激活函数
        Activation = Sigmoid if use_sigmoid else ReLU
        
        # ========== 特征提取部分（卷积层） ==========
        
        # C1: 卷积层, 1通道 -> 6通道, 5x5核, 填充2（保持28x28尺寸）
        # 输入: (batch, 1, 28, 28)
        # 输出: (batch, 6, 28, 28)
        self.layers.append(Conv2D(1, 6, kernel_size=5, stride=1, padding=2))
        self.layers.append(Activation())
        
        # S2: 2x2最大池化, 步长2
        # 输入: (batch, 6, 28, 28)
        # 输出: (batch, 6, 14, 14)
        self.layers.append(MaxPooling2D(pool_size=2, stride=2))
        
        # C3: 卷积层, 6通道 -> 16通道, 5x5核
        # 输入: (batch, 6, 14, 14)
        # 输出: (batch, 16, 10, 10)
        self.layers.append(Conv2D(6, 16, kernel_size=5, stride=1))
        self.layers.append(Activation())
        
        # S4: 2x2最大池化, 步长2
        # 输入: (batch, 16, 10, 10)
        # 输出: (batch, 16, 5, 5)
        self.layers.append(MaxPooling2D(pool_size=2, stride=2))
        
        # C5: 卷积层, 16通道 -> 120通道, 5x5核
        # 输入: (batch, 16, 5, 5)
        # 输出: (batch, 120, 1, 1) - 实际上相当于全连接
        self.layers.append(Conv2D(16, 120, kernel_size=5, stride=1))
        self.layers.append(Activation())
        
        # 展平层: (batch, 120, 1, 1) -> (batch, 120)
        self.layers.append(Flatten())
        
        # ========== 分类部分（全连接层） ==========
        
        # F6: 全连接层, 120 -> 84
        self.layers.append(Dense(120, 84))
        self.layers.append(Activation())
        
        # 输出层: 84 -> num_classes
        self.layers.append(Dense(84, num_classes))
        
        # 损失函数: Softmax + 交叉熵
        self.criterion = SoftmaxCrossEntropy()
        
        # 统计可训练参数
        self._count_parameters()
    
    def _count_parameters(self):
        """统计模型的可训练参数数量"""
        total = 0
        self.param_counts = []
        
        for i, layer in enumerate(self.layers):
            params = sum(p.size for p in layer.params)
            if params > 0:
                self.param_counts.append((i, layer.__class__.__name__, params))
                total += params
        
        self.total_params = total
    
    def summary(self):
        """打印模型摘要"""
        print("=" * 60)
        print("LeNet Model Summary")
        print("=" * 60)
        print(f"{'Layer':<5} {'Type':<20} {'Output Shape':<25} {'Params':<10}")
        print("-" * 60)
        
        # 模拟前向传播来得到每层的输出形状
        x = np.zeros((1, 1, 28, 28))  # 假设输入
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            x = layer.forward(x)
            
            # 查找该层的参数数量
            params = 0
            for idx, name, p in self.param_counts:
                if idx == i:
                    params = p
                    break
            
            shape_str = str(x.shape)
            print(f"{i:<5} {layer_name:<20} {shape_str:<25} {params:<10,}")
        
        print("-" * 60)
        print(f"Total trainable parameters: {self.total_params:,}")
        print("=" * 60)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像 (batch_size, 1, 28, 28)
        返回:
            logits: 网络原始输出 (batch_size, num_classes)
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self):
        """
        反向传播
        
        从损失函数开始，依次调用各层的backward方法。
        """
        # 从SoftmaxCrossEntropy获取初始梯度
        grad = self.criterion.backward()
        
        # 反向遍历各层
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def compute_loss(self, logits, labels):
        """
        计算损失
        
        参数:
            logits: 网络输出 (batch_size, num_classes)
            labels: 真实标签 (batch_size,)
        返回:
            标量损失值
        """
        return self.criterion.forward(logits, labels)
    
    def predict(self, x):
        """
        预测类别
        
        参数:
            x: 输入图像 (batch_size, 1, 28, 28)
        返回:
            预测的类别索引 (batch_size,)
        """
        logits = self.forward(x)
        return np.argmax(logits, axis=1)
    
    def predict_proba(self, x):
        """
        预测类别概率
        
        参数:
            x: 输入图像 (batch_size, 1, 28, 28)
        返回:
            各类别概率 (batch_size, num_classes)
        """
        logits = self.forward(x)
        # 重新计算softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def accuracy(self, x, labels):
        """
        计算准确率
        
        参数:
            x: 输入图像
            labels: 真实标签
        返回:
            准确率 (0.0 - 1.0)
        """
        predictions = self.predict(x)
        return np.mean(predictions == labels)
    
    def get_feature_maps(self, x, layer_idx):
        """
        获取指定层的特征图
        
        用于可视化中间层的激活
        
        参数:
            x: 输入图像
            layer_idx: 层索引
        返回:
            该层的输出（特征图）
        """
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i == layer_idx:
                return x
        return x


class SimpleCNN:
    """
    简化版CNN
    
    比LeNet更简单的架构，适合快速实验。
    
    架构: Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> Flatten -> Dense
    """
    def __init__(self, num_classes=10):
        self.layers = []
        
        # Conv1: 1 -> 32
        self.layers.append(Conv2D(1, 32, kernel_size=3, stride=1, padding=1))
        self.layers.append(ReLU())
        self.layers.append(MaxPooling2D(pool_size=2, stride=2))
        
        # Conv2: 32 -> 64
        self.layers.append(Conv2D(32, 64, kernel_size=3, stride=1, padding=1))
        self.layers.append(ReLU())
        self.layers.append(MaxPooling2D(pool_size=2, stride=2))
        
        # Flatten: 7x7x64 = 3136
        self.layers.append(Flatten())
        
        # Dense
        self.layers.append(Dense(64 * 7 * 7, 128))
        self.layers.append(ReLU())
        self.layers.append(Dense(128, num_classes))
        
        self.criterion = SoftmaxCrossEntropy()
        self._count_parameters()
    
    def _count_parameters(self):
        """统计参数"""
        self.total_params = sum(sum(p.size for p in layer.params) 
                                for layer in self.layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self):
        grad = self.criterion.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def compute_loss(self, logits, labels):
        return self.criterion.forward(logits, labels)
    
    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)
    
    def accuracy(self, x, labels):
        return np.mean(self.predict(x) == labels)
