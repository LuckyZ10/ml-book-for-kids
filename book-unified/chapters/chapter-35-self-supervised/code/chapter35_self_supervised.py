"""
第三十五章：自监督学习 - 代码实现
包含：对比学习、掩码预测、数据增强
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class ContrastiveLearning:
    """
    对比学习 - SimCLR简化版
    学习样本之间的相似性
    """
    
    def __init__(self, feature_dim: int = 128, temperature: float = 0.5):
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # 编码器（简化版）
        self.W = np.random.randn(784, feature_dim) * 0.01
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """编码样本"""
        h = np.dot(x, self.W)
        return h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-10)
    
    def contrastive_loss(self, z_i: np.ndarray, z_j: np.ndarray) -> float:
        """
        NT-Xent损失（归一化温度缩放交叉熵损失）
        z_i, z_j: 同一样本的两种增强视图
        """
        # 相似度计算
        sim_ij = np.dot(z_i, z_j) / self.temperature
        
        # 简化版损失（仅演示）
        loss = -np.log(np.exp(sim_ij) / (np.exp(sim_ij) + 1))
        return loss
    
    def fit(self, X: np.ndarray, epochs: int = 100, lr: float = 0.001):
        """训练自监督模型"""
        print(f"开始对比学习训练...")
        print(f"  样本数: {len(X)}")
        print(f"  特征维度: {self.feature_dim}")
        
        for epoch in range(epochs):
            # 数据增强（简化：添加噪声）
            X_aug1 = X + np.random.randn(*X.shape) * 0.1
            X_aug2 = X + np.random.randn(*X.shape) * 0.1
            
            # 编码
            z1 = self.encode(X_aug1)
            z2 = self.encode(X_aug2)
            
            # 计算损失并更新（简化）
            loss = 0
            for i in range(len(X)):
                loss += self.contrastive_loss(z1[i], z2[i])
            loss /= len(X)
            
            # 简单梯度更新
            grad = np.dot(X.T, (z1 - z2)) / len(X)
            self.W -= lr * grad
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: Loss = {loss:.4f}")


def self_supervised_demo():
    """自监督学习演示"""
    print("=" * 60)
    print("自监督学习演示")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 模拟图像数据（展平）
    X = np.random.randn(100, 784)
    
    # 创建模型并训练
    model = ContrastiveLearning(feature_dim=64)
    model.fit(X, epochs=50, lr=0.001)
    
    print("\n训练完成！学习到的表示可用于下游任务。")


if __name__ == "__main__":
    self_supervised_demo()
