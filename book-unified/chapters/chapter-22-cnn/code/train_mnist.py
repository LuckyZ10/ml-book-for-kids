"""
train_mnist.py - MNIST训练脚本

使用纯NumPy实现的LeNet训练MNIST手写数字识别。
包含数据加载、预处理、训练和评估。
"""

import numpy as np
import pickle
import gzip
import os
from urllib import request

from lenet import LeNet
from layers import SGD, Adam


def load_mnist(data_dir='./data'):
    """
    加载MNIST数据集
    
    如果本地不存在，则从网络下载。
    
    参数:
        data_dir: 数据存储目录
    返回:
        train_set, valid_set, test_set
        每个set是 (data, labels) 元组
    """
    os.makedirs(data_dir, exist_ok=True)
    
    filename = os.path.join(data_dir, 'mnist.pkl.gz')
    url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    
    # 下载数据（如果不存在）
    if not os.path.exists(filename):
        print(f"正在下载MNIST数据集...")
        try:
            request.urlretrieve(url, filename)
            print(f"下载完成: {filename}")
        except Exception as e:
            print(f"下载失败: {e}")
            print("请手动下载并放置在data目录")
            raise
    
    # 加载数据
    with gzip.open(filename, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    
    return train_set, valid_set, test_set


def preprocess_data(x):
    """
    预处理MNIST数据
    
    1. 转换为float32
    2. 归一化到[0, 1]范围
    3. reshape为 (batch, 1, 28, 28)
    
    参数:
        x: 原始数据 (n_samples, 784)
    返回:
        预处理后的数据 (n_samples, 1, 28, 28)
    """
    # 归一化
    x = x.astype(np.float32) / 255.0
    
    # 添加通道维度并reshape
    # MNIST是灰度图，所以通道数为1
    x = x.reshape(-1, 1, 28, 28)
    
    return x


def create_batches(x, y, batch_size=64, shuffle=True):
    """
    创建数据批次
    
    参数:
        x: 输入数据
        y: 标签
        batch_size: 批次大小
        shuffle: 是否打乱数据
    返回:
        生成器，产生 (batch_x, batch_y)
    """
    n = len(x)
    indices = np.arange(n)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_indices = indices[start:end]
        yield x[batch_indices], y[batch_indices]


def train_epoch(model, train_x, train_y, optimizer, batch_size=64):
    """
    训练一个epoch
    
    参数:
        model: 神经网络模型
        train_x: 训练数据
        train_y: 训练标签
        optimizer: 优化器
        batch_size: 批次大小
    返回:
        平均损失
    """
    total_loss = 0
    n_batches = 0
    
    for batch_x, batch_y in create_batches(train_x, train_y, batch_size):
        # 前向传播
        logits = model.forward(batch_x)
        loss = model.compute_loss(logits, batch_y)
        
        # 反向传播
        model.backward()
        
        # 更新参数
        optimizer.step(model.layers)
        
        total_loss += loss
        n_batches += 1
    
    return total_loss / n_batches


def evaluate(model, x, y, batch_size=256):
    """
    评估模型
    
    参数:
        model: 模型
        x: 数据
        y: 标签
        batch_size: 评估时的批次大小（可以较大，因为不需要梯度）
    返回:
        准确率
    """
    predictions = []
    
    for batch_x, _ in create_batches(x, y, batch_size, shuffle=False):
        pred = model.predict(batch_x)
        predictions.extend(pred)
    
    predictions = np.array(predictions)
    return np.mean(predictions == y)


def train(model, train_x, train_y, valid_x, valid_y, 
          epochs=10, batch_size=64, lr=0.001, optimizer_type='adam',
          verbose=True):
    """
    完整的训练流程
    
    参数:
        model: 神经网络模型
        train_x, train_y: 训练数据
        valid_x, valid_y: 验证数据
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        optimizer_type: 优化器类型 ('sgd' 或 'adam')
        verbose: 是否打印进度
    返回:
        训练历史记录
    """
    # 选择优化器
    if optimizer_type == 'sgd':
        optimizer = SGD(lr=lr)
    elif optimizer_type == 'adam':
        optimizer = Adam(lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_acc': []
    }
    
    best_valid_acc = 0
    
    for epoch in range(epochs):
        # 训练
        train_loss = train_epoch(model, train_x, train_y, optimizer, batch_size)
        
        # 评估（使用子集加速）
        train_acc = evaluate(model, train_x[:5000], train_y[:5000])
        valid_acc = evaluate(model, valid_x, valid_y)
        
        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)
        
        # 更新最佳模型
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
        
        if verbose:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Valid Acc: {valid_acc:.4f}")
    
    return history


def main():
    """
    主函数 - 完整的训练和评估流程
    """
    print("=" * 60)
    print("MNIST手写数字识别 - 纯NumPy实现的CNN")
    print("=" * 60)
    
    # 设置随机种子（可重复性）
    np.random.seed(42)
    
    # 1. 加载数据
    print("\n[1] 加载MNIST数据集...")
    train_set, valid_set, test_set = load_mnist()
    
    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x, test_y = test_set
    
    print(f"  训练集: {train_x.shape[0]:,} 样本")
    print(f"  验证集: {valid_x.shape[0]:,} 样本")
    print(f"  测试集: {test_x.shape[0]:,} 样本")
    
    # 2. 预处理
    print("\n[2] 预处理数据...")
    train_x = preprocess_data(train_x)
    valid_x = preprocess_data(valid_x)
    test_x = preprocess_data(test_x)
    
    print(f"  输入形状: {train_x.shape[1:]}")
    
    # 3. 创建模型
    print("\n[3] 创建LeNet模型...")
    model = LeNet(num_classes=10)
    
    # 打印模型摘要
    model.summary()
    
    # 4. 训练
    print("\n[4] 开始训练...")
    print(f"  Epochs: 10, Batch Size: 64, Optimizer: Adam, LR: 0.001\n")
    
    history = train(model, train_x, train_y, valid_x, valid_y,
                    epochs=10, batch_size=64, lr=0.001, optimizer_type='adam')
    
    # 5. 测试集评估
    print("\n[5] 测试集评估...")
    test_acc = evaluate(model, test_x, test_y)
    print(f"  测试集准确率: {test_acc*100:.2f}%")
    
    # 6. 一些预测示例
    print("\n[6] 预测示例...")
    n_samples = 10
    sample_indices = np.random.choice(len(test_x), n_samples, replace=False)
    samples_x = test_x[sample_indices]
    samples_y = test_y[sample_indices]
    
    predictions = model.predict(samples_x)
    
    print("  样本 | 预测 | 真实 | 正确?")
    print("  " + "-" * 30)
    for i in range(n_samples):
        correct = "✓" if predictions[i] == samples_y[i] else "✗"
        print(f"  {i+1:4d} | {predictions[i]:4d} | {samples_y[i]:4d} | {correct}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    return model, history


if __name__ == "__main__":
    model, history = main()
