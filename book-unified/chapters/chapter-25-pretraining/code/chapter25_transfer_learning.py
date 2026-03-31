"""
迁移学习代码实现
包含：特征提取、层冻结、领域适应
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class TransferLearningModel:
    """
    迁移学习模型 - 演示知识迁移
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 预训练层（在大数据集上训练）
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b2 = np.zeros(hidden_dim)
        
        # 任务特定层（在新任务上训练）
        self.W3 = np.random.randn(hidden_dim, num_classes) * 0.01
        self.b3 = np.zeros(num_classes)
        
        # 冻结标记
        self.frozen_layers = []
    
    def freeze_pretrained(self):
        """冻结预训练层"""
        self.frozen_layers = ['W1', 'b1', 'W2', 'b2']
        print("预训练层已冻结")
    
    def unfreeze_all(self):
        """解冻所有层（完全微调）"""
        self.frozen_layers = []
        print("所有层已解冻（完全微调模式）")
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List]:
        """前向传播"""
        cache = []
        
        # 第一层
        z1 = np.dot(X, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        cache.append((X, z1))
        
        # 第二层
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.maximum(0, z2)  # ReLU
        cache.append((a1, z2))
        
        # 输出层
        z3 = np.dot(a2, self.W3) + self.b3
        
        # Softmax
        exp_z = np.exp(z3 - np.max(z3, axis=1, keepdims=True))
        output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        return output, cache
    
    def backward(self, X: np.ndarray, y: np.ndarray, 
                 output: np.ndarray, cache: List, lr: float = 0.01):
        """反向传播"""
        m = X.shape[0]
        
        # 输出层梯度
        dz3 = output.copy()
        dz3[range(m), y] -= 1
        dz3 /= m
        
        a2, _ = cache[1]
        dW3 = np.dot(a2.T, dz3)
        db3 = np.sum(dz3, axis=0)
        
        # 更新输出层（始终更新）
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        
        # 如果预训练层未冻结，继续反向传播
        if 'W2' not in self.frozen_layers:
            da2 = np.dot(dz3, self.W3.T)
            dz2 = da2 * (a2 > 0)  # ReLU导数
            
            a1, _ = cache[0]
            dW2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0)
            
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * (a1 > 0)
            
            X_input, _ = cache[0]
            dW1 = np.dot(X_input.T, dz1)
            db1 = np.sum(dz1, axis=0)
            
            self.W1 -= lr * dW1
            self.b1 -= lr * db1


def transfer_learning_comparison():
    """比较不同的迁移学习策略"""
    print("=" * 60)
    print("迁移学习策略比较")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 模拟数据
    input_dim = 100
    num_classes = 5
    X_train = np.random.randn(200, input_dim)
    y_train = np.random.randint(0, num_classes, 200)
    X_test = np.random.randn(50, input_dim)
    y_test = np.random.randint(0, num_classes, 50)
    
    # 策略1：从头训练
    print("\n【策略1】从头训练（无预训练）")
    model1 = TransferLearningModel(input_dim, num_classes=num_classes)
    # 不加载预训练权重
    train_and_evaluate(model1, X_train, y_train, X_test, y_test, epochs=30)
    
    # 策略2：特征提取（冻结预训练层）
    print("\n【策略2】特征提取（冻结预训练层）")
    model2 = TransferLearningModel(input_dim, num_classes=num_classes)
    model2.load_pretrained_weights()  # 加载预训练权重
    model2.freeze_pretrained()
    train_and_evaluate(model2, X_train, y_train, X_test, y_test, epochs=30)
    
    # 策略3：完全微调
    print("\n【策略3】完全微调（解冻所有层）")
    model3 = TransferLearningModel(input_dim, num_classes=num_classes)
    model3.load_pretrained_weights()
    model3.unfreeze_all()
    train_and_evaluate(model3, X_train, y_train, X_test, y_test, epochs=30)


def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=30):
    """训练和评估"""
    for epoch in range(epochs):
        output, cache = model.forward(X_train)
        model.backward(X_train, y_train, output, cache, lr=0.01)
    
    # 评估
    output_test, _ = model.forward(X_test)
    predictions = np.argmax(output_test, axis=1)
    accuracy = np.mean(predictions == y_test)
    print(f"  测试准确率: {accuracy*100:.2f}%")


if __name__ == "__main__":
    transfer_learning_comparison()
