#!/usr/bin/env python3
"""
正则化工具集 - 补充代码
Chapter 21: 正则化
"""

import numpy as np
from typing import Tuple, Optional, Literal


class L1Regularizer:
    """L1正则化 (Lasso)"""
    
    def __init__(self, lambda_: float = 0.01):
        self.lambda_ = lambda_
    
    def __call__(self, weights: np.ndarray) -> float:
        """计算L1惩罚项"""
        return self.lambda_ * np.sum(np.abs(weights))
    
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """计算次梯度"""
        return self.lambda_ * np.sign(weights)


class L2Regularizer:
    """L2正则化 (Ridge)"""
    
    def __init__(self, lambda_: float = 0.01):
        self.lambda_ = lambda_
    
    def __call__(self, weights: np.ndarray) -> float:
        """计算L2惩罚项"""
        return 0.5 * self.lambda_ * np.sum(weights ** 2)
    
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """计算梯度"""
        return self.lambda_ * weights


class ElasticNetRegularizer:
    """Elastic Net正则化 (L1 + L2)"""
    
    def __init__(self, lambda_: float = 0.01, l1_ratio: float = 0.5):
        self.lambda_ = lambda_
        self.l1_ratio = l1_ratio
        self.l1 = L1Regularizer(lambda_ * l1_ratio)
        self.l2 = L2Regularizer(lambda_ * (1 - l1_ratio))
    
    def __call__(self, weights: np.ndarray) -> float:
        return self.l1(weights) + self.l2(weights)
    
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.l1.gradient(weights) + self.l2.gradient(weights)


class Dropout:
    """Dropout层"""
    
    def __init__(self, drop_rate: float = 0.5):
        self.drop_rate = drop_rate
        self.mask = None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self.mask = (np.random.rand(*x.shape) > self.drop_rate).astype(float)
            return x * self.mask / (1 - self.drop_rate)
        return x
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.mask / (1 - self.drop_rate)


class BatchNorm:
    """批量归一化层"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        
        # 运行时统计
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        # 缓存
        self.x_norm = None
        self.batch_mean = None
        self.batch_var = None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self.batch_mean = np.mean(x, axis=0, keepdims=True)
            self.batch_var = np.var(x, axis=0, keepdims=True)
            
            # 更新运行时统计
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                                self.momentum * self.batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                               self.momentum * self.batch_var
            
            # 标准化
            self.x_norm = (x - self.batch_mean) / np.sqrt(self.batch_var + self.eps)
        else:
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        return self.gamma * self.x_norm + self.beta
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        # 简化版反向传播
        return grad * self.gamma


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
    
    def __call__(self, val_loss: float, model_weights: Optional[dict] = None) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best and model_weights is not None:
                self.best_weights = {k: v.copy() for k, v in model_weights.items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def restore(self, model_weights: dict):
        """恢复最佳权重"""
        if self.best_weights is not None:
            for k in model_weights:
                model_weights[k] = self.best_weights[k].copy()


def train_test_split(X: np.ndarray, y: np.ndarray, 
                     test_ratio: float = 0.2, 
                     random_state: Optional[int] = None) -> Tuple:
    """划分训练集和验证集"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(X)
    indices = np.random.permutation(n)
    test_size = int(n * test_ratio)
    
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def cross_validate(model_class, X: np.ndarray, y: np.ndarray, 
                   n_folds: int = 5, **model_params) -> dict:
    """交叉验证"""
    n = len(X)
    fold_size = n // n_folds
    scores = []
    
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < n_folds - 1 else n
        
        val_idx = slice(val_start, val_end)
        train_idx = np.concatenate([np.arange(0, val_start), 
                                    np.arange(val_end, n)])
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)
    
    return {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores)
    }


class DataAugmentation:
    """简单的数据增强"""
    
    @staticmethod
    def add_noise(X: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, noise_factor, X.shape)
        return X + noise
    
    @staticmethod
    def scale(X: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """随机缩放"""
        scale = np.random.uniform(*scale_range, size=(X.shape[0], 1))
        return X * scale
    
    @staticmethod
    def shift(X: np.ndarray, shift_range: Tuple[float, float] = (-0.1, 0.1)) -> np.ndarray:
        """随机平移"""
        shift = np.random.uniform(*shift_range, size=(X.shape[0], 1))
        return X + shift


# 使用示例
if __name__ == '__main__':
    # 测试正则化器
    weights = np.array([1.0, -2.0, 0.5, -0.1])
    
    l1 = L1Regularizer(lambda_=0.1)
    print(f"L1惩罚: {l1(weights):.4f}")
    print(f"L1梯度: {l1.gradient(weights)}")
    
    l2 = L2Regularizer(lambda_=0.1)
    print(f"L2惩罚: {l2(weights):.4f}")
    
    elastic = ElasticNetRegularizer(lambda_=0.1, l1_ratio=0.5)
    print(f"Elastic Net惩罚: {elastic(weights):.4f}")
    
    # 测试Dropout
    dropout = Dropout(drop_rate=0.3)
    x = np.random.randn(10, 5)
    out = dropout.forward(x, training=True)
    print(f"Dropout输入均值: {np.mean(x):.4f}, 输出均值: {np.mean(out):.4f}")
