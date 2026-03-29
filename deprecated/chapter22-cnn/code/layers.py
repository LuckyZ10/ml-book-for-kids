"""
layers.py - CNN基础层的NumPy实现

本模块实现了卷积神经网络的基础组件，完全使用NumPy实现，
用于教学目的，帮助理解CNN的前向传播和反向传播机制。

包含的层：
- Layer: 基类
- Conv2D: 二维卷积层
- MaxPooling2D: 最大池化层
- Flatten: 展平层
- ReLU/Sigmoid: 激活函数
- Dense: 全连接层
- SoftmaxCrossEntropy: Softmax + 交叉熵损失
- SGD/Adam: 优化器
"""

import numpy as np


class Layer:
    """
    所有层的基类
    
    定义了层的通用接口，所有具体层都需要继承此类。
    """
    def __init__(self):
        self.params = []  # 可训练参数列表
        self.grads = []   # 对应的梯度列表
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入数据
        返回:
            输出数据
        """
        raise NotImplementedError
        
    def backward(self, grad_output):
        """
        反向传播
        
        参数:
            grad_output: 从上一层传来的梯度
        返回:
            对输入的梯度
        """
        raise NotImplementedError


class Conv2D(Layer):
    """
    二维卷积层
    
    实现标准的2D卷积操作，支持多通道输入输出、自定义步长和填充。
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小（整数或元组）
        stride: 步长，默认1
        padding: 填充，默认0
        
    输入形状: (batch_size, in_channels, height, width)
    输出形状: (batch_size, out_channels, out_height, out_width)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 处理kernel_size（支持整数或元组）
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        
        self.k_h, self.k_w = self.kernel_size
        
        # 使用He初始化权重（适合ReLU激活函数）
        # He初始化: std = sqrt(2 / fan_in)
        scale = np.sqrt(2.0 / (in_channels * self.k_h * self.k_w))
        self.W = np.random.randn(out_channels, in_channels, self.k_h, self.k_w) * scale
        self.b = np.zeros(out_channels)
        
        self.params = [self.W, self.b]
        self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]
        
    def _pad_input(self, x):
        """
        对输入进行零填充
        
        参数:
            x: 输入张量 (batch, channels, h, w)
        返回:
            填充后的张量
        """
        if self.padding > 0:
            return np.pad(x, 
                         ((0, 0), (0, 0), 
                          (self.padding, self.padding), 
                          (self.padding, self.padding)), 
                         mode='constant')
        return x
    
    def forward(self, x):
        """
        前向传播 - 执行卷积操作
        
        卷积公式: out[i,j] = sum_m sum_n (input[i+m, j+n] * weight[m,n]) + bias
        
        参数:
            x: 输入张量 (batch_size, in_channels, height, width)
        返回:
            卷积结果 (batch_size, out_channels, out_height, out_width)
        """
        self.x = x  # 保存输入用于反向传播
        batch_size, in_c, h, w = x.shape
        
        # 对输入进行填充
        x_padded = self._pad_input(x)
        self.x_padded = x_padded
        _, _, h_p, w_p = x_padded.shape
        
        # 计算输出尺寸: (W - K + 2P) / S + 1
        out_h = (h_p - self.k_h) // self.stride[0] + 1
        out_w = (w_p - self.k_w) // self.stride[1] + 1
        
        # 初始化输出张量
        out = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        # 执行卷积运算
        for i in range(out_h):
            for j in range(out_w):
                # 计算当前输出位置对应的输入区域
                h_start = i * self.stride[0]
                w_start = j * self.stride[1]
                h_end = h_start + self.k_h
                w_end = w_start + self.k_w
                
                # 提取感受野: (batch, in_c, k_h, k_w)
                receptive_field = x_padded[:, :, h_start:h_end, w_start:w_end]
                
                # 对每个输出通道计算卷积
                for c_out in range(self.out_channels):
                    # 逐元素相乘后求和: (batch, in_c, k_h, k_w) * (in_c, k_h, k_w)
                    out[:, c_out, i, j] = np.sum(
                        receptive_field * self.W[c_out], axis=(1, 2, 3)
                    ) + self.b[c_out]
        
        return out
    
    def backward(self, grad_output):
        """
        反向传播 - 计算卷积层的梯度
        
        需要计算:
        1. dW: 对权重的梯度
        2. db: 对偏置的梯度
        3. dX: 对输入的梯度（返回给上一层）
        
        参数:
            grad_output: 从上层传来的梯度 (batch, out_c, out_h, out_w)
        返回:
            对输入的梯度 (batch, in_c, h, w)
        """
        batch_size = grad_output.shape[0]
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]
        
        # 初始化梯度
        self.grads[0][:] = 0  # dW
        self.grads[1][:] = 0  # db
        grad_input_padded = np.zeros_like(self.x_padded)
        
        # 计算梯度
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride[0]
                w_start = j * self.stride[1]
                h_end = h_start + self.k_h
                w_end = w_start + self.k_w
                
                # 提取前向传播时的感受野
                receptive_field = self.x_padded[:, :, h_start:h_end, w_start:w_end]
                
                for c_out in range(self.out_channels):
                    # 当前位置的梯度 (batch,)
                    grad = grad_output[:, c_out, i, j]
                    
                    # dW: 梯度累加
                    # (batch, 1, 1, 1) * (batch, in_c, k_h, k_w) -> (in_c, k_h, k_w)
                    self.grads[0][c_out] += np.sum(
                        grad[:, np.newaxis, np.newaxis, np.newaxis] * receptive_field, 
                        axis=0
                    )
                    
                    # db: 偏置梯度
                    self.grads[1][c_out] += np.sum(grad)
                    
                    # dX: 传播梯度到输入
                    # (batch, 1, 1, 1) * (in_c, k_h, k_w) -> (batch, in_c, k_h, k_w)
                    grad_input_padded[:, :, h_start:h_end, w_start:w_end] += \
                        grad[:, np.newaxis, np.newaxis, np.newaxis] * self.W[c_out]
        
        # 去除填充部分
        if self.padding > 0:
            grad_input = grad_input_padded[:, :, 
                                          self.padding:-self.padding, 
                                          self.padding:-self.padding]
        else:
            grad_input = grad_input_padded
            
        return grad_input


class MaxPooling2D(Layer):
    """
    最大池化层
    
    在池化窗口中选择最大值作为输出，用于降维和提供平移不变性。
    
    参数:
        pool_size: 池化窗口大小，默认2
        stride: 步长，默认等于pool_size
        
    输入形状: (batch_size, channels, height, width)
    输出形状: (batch_size, channels, out_height, out_width)
    """
    def __init__(self, pool_size=2, stride=None):
        super().__init__()
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else self.pool_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
            
    def forward(self, x):
        """
        前向传播 - 最大池化
        
        对每个池化窗口，选择最大值作为输出。
        同时记录最大值的位置（掩码），用于反向传播。
        
        参数:
            x: 输入张量 (batch, channels, h, w)
        返回:
            池化结果 (batch, channels, out_h, out_w)
        """
        self.x = x
        batch_size, channels, h, w = x.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.stride
        
        # 计算输出尺寸
        out_h = (h - pool_h) // stride_h + 1
        out_w = (w - pool_w) // stride_w + 1
        
        # 初始化输出和掩码
        out = np.zeros((batch_size, channels, out_h, out_w))
        self.mask = {}  # 存储最大值位置
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride_h
                w_start = j * stride_w
                h_end = h_start + pool_h
                w_end = w_start + pool_w
                
                # 提取池化区域
                pool_region = x[:, :, h_start:h_end, w_start:w_end]
                
                # 重塑以便求最大值
                pool_flat = pool_region.reshape(batch_size, channels, -1)
                
                # 求最大值
                out[:, :, i, j] = np.max(pool_flat, axis=2)
                
                # 保存最大值索引用于反向传播
                max_indices = np.argmax(pool_flat, axis=2)
                self.mask[(i, j)] = {
                    'start': (h_start, w_start),
                    'indices': max_indices
                }
        
        return out
    
    def backward(self, grad_output):
        """
        反向传播 - 最大池化梯度
        
        梯度只传递给前向传播时取得最大值的位置，其他位置为0。
        
        参数:
            grad_output: 上层梯度 (batch, channels, out_h, out_w)
        返回:
            输入梯度 (batch, channels, h, w)
        """
        batch_size, channels, out_h, out_w = grad_output.shape
        _, _, h, w = self.x.shape
        pool_h, pool_w = self.pool_size
        
        grad_input = np.zeros_like(self.x)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start, w_start = self.mask[(i, j)]['start']
                max_indices = self.mask[(i, j)]['indices']
                
                # 将梯度传递给最大值位置
                for b in range(batch_size):
                    for c in range(channels):
                        idx = max_indices[b, c]
                        h_idx = h_start + idx // pool_w
                        w_idx = w_start + idx % pool_w
                        grad_input[b, c, h_idx, w_idx] += grad_output[b, c, i, j]
        
        return grad_input


class Flatten(Layer):
    """
    展平层
    
    将多维输入展平为二维，用于连接卷积层和全连接层。
    
    输入形状: (batch_size, ...)
    输出形状: (batch_size, product_of_dimensions)
    """
    def __init__(self):
        super().__init__()
        self.input_shape = None
        
    def forward(self, x):
        """
        前向传播 - 展平输入
        
        参数:
            x: 任意形状的输入
        返回:
            展平后的二维数组
        """
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, grad_output):
        """
        反向传播 - 恢复形状
        
        参数:
            grad_output: 展平后的梯度
        返回:
            恢复原始形状的梯度
        """
        return grad_output.reshape(self.input_shape)


class ReLU(Layer):
    """
    ReLU激活函数
    
    f(x) = max(0, x)
    
    最常用的激活函数，缓解梯度消失问题。
    """
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入
        返回:
            max(0, x)
        """
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        """
        反向传播
        
        x > 0时梯度为1，x <= 0时梯度为0
        
        参数:
            grad_output: 上层梯度
        返回:
            传递后的梯度
        """
        return grad_output * (self.x > 0)


class Sigmoid(Layer):
    """
    Sigmoid激活函数
    
    f(x) = 1 / (1 + exp(-x))
    
    将输出压缩到(0, 1)区间，适合二分类问题。
    """
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入
        返回:
            sigmoid(x)
        """
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, grad_output):
        """
        反向传播
        
        sigmoid的导数: f'(x) = f(x) * (1 - f(x))
        
        参数:
            grad_output: 上层梯度
        返回:
            传递后的梯度
        """
        return grad_output * self.out * (1 - self.out)


class Dense(Layer):
    """
    全连接层（线性层）
    
    y = x @ W + b
    
    参数:
        in_features: 输入特征数
        out_features: 输出特征数
        
    输入形状: (batch_size, in_features)
    输出形状: (batch_size, out_features)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # He初始化
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)
        
        self.params = [self.W, self.b]
        self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入 (batch, in_features)
        返回:
            线性变换结果 (batch, out_features)
        """
        self.x = x
        return x @ self.W + self.b
    
    def backward(self, grad_output):
        """
        反向传播
        
        计算:
        - dW = x^T @ grad_output
        - db = sum(grad_output, axis=0)
        - dx = grad_output @ W^T
        
        参数:
            grad_output: 上层梯度 (batch, out_features)
        返回:
            输入梯度 (batch, in_features)
        """
        # dW
        self.grads[0] = self.x.T @ grad_output
        # db
        self.grads[1] = np.sum(grad_output, axis=0)
        # dx
        return grad_output @ self.W.T


class SoftmaxCrossEntropy:
    """
    Softmax + 交叉熵损失
    
    组合Softmax激活和交叉熵损失，数值稳定性更好。
    
    交叉熵损失: L = -sum(y_true * log(y_pred))
    """
    def forward(self, logits, labels):
        """
        前向传播 - 计算损失
        
        参数:
            logits: 网络原始输出 (batch_size, num_classes)
            labels: 类别索引 (batch_size,)
        返回:
            平均交叉熵损失
        """
        self.labels = labels
        batch_size = logits.shape[0]
        
        # Softmax（使用数值稳定的实现）
        # 减去最大值防止exp溢出
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # 交叉熵损失
        # 只取正确类别的概率的对数的负数
        log_probs = np.log(self.probs + 1e-8)  # 加小值防止log(0)
        loss = -np.mean(log_probs[np.arange(batch_size), labels])
        
        return loss
    
    def backward(self):
        """
        反向传播 - 计算对logits的梯度
        
        Softmax + 交叉熵的梯度简化为: probs - one_hot(labels)
        
        返回:
            对logits的梯度 (batch_size, num_classes)
        """
        batch_size = self.labels.shape[0]
        grad = self.probs.copy()
        # 正确类别的梯度减1
        grad[np.arange(batch_size), self.labels] -= 1
        return grad / batch_size  # 平均梯度


class SGD:
    """
    随机梯度下降优化器
    
    参数更新: param = param - lr * grad
    
    参数:
        lr: 学习率
    """
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def step(self, layers):
        """
        执行一步参数更新
        
        参数:
            layers: 所有层的列表
        """
        for layer in layers:
            for i, param in enumerate(layer.params):
                param -= self.lr * layer.grads[i]


class Adam:
    """
    Adam优化器
    
    结合动量和自适应学习率的优化算法。
    
    参数:
        lr: 学习率
        beta1: 一阶矩衰减率
        beta2: 二阶矩衰减率
        epsilon: 数值稳定性常数
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # 一阶矩（动量）
        self.v = {}  # 二阶矩（自适应学习率）
        self.t = 0   # 时间步
        
    def step(self, layers):
        """
        执行一步参数更新
        
        参数:
            layers: 所有层的列表
        """
        self.t += 1
        
        for layer_idx, layer in enumerate(layers):
            for param_idx, param in enumerate(layer.params):
                key = (layer_idx, param_idx)
                grad = layer.grads[param_idx]
                
                # 初始化矩
                if key not in self.m:
                    self.m[key] = np.zeros_like(grad)
                    self.v[key] = np.zeros_like(grad)
                
                # 更新一阶矩（动量）
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                # 更新二阶矩（自适应）
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
                
                # 偏差修正
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                
                # 更新参数
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
