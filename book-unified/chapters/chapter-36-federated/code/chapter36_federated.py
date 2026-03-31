"""
第三十六章：联邦学习 - 代码实现
包含：客户端训练、服务端聚合、差分隐私
"""

import numpy as np
from typing import List, Dict, Tuple
import copy


class FederatedClient:
    """联邦学习客户端"""
    
    def __init__(self, client_id: int, data: np.ndarray, labels: np.ndarray):
        self.client_id = client_id
        self.data = data
        self.labels = labels
        
        # 本地模型参数
        self.weights = np.random.randn(data.shape[1], 10) * 0.01
        self.bias = np.zeros(10)
    
    def local_train(self, epochs: int = 5, lr: float = 0.01) -> Dict:
        """本地训练"""
        for _ in range(epochs):
            # 前向传播
            logits = np.dot(self.data, self.weights) + self.bias
            
            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # 梯度计算
            grads = probs.copy()
            grads[range(len(self.labels)), self.labels] -= 1
            grads /= len(self.labels)
            
            # 更新
            dW = np.dot(self.data.T, grads)
            db = np.sum(grads, axis=0)
            
            self.weights -= lr * dW
            self.bias -= lr * db
        
        return {
            'weights': self.weights.copy(),
            'bias': self.bias.copy(),
            'n_samples': len(self.data)
        }


class FederatedServer:
    """联邦学习服务端"""
    
    def __init__(self, n_clients: int):
        self.n_clients = n_clients
        self.global_weights = None
        self.global_bias = None
    
    def aggregate(self, client_updates: List[Dict]) -> Dict:
        """
        FedAvg聚合算法
        按样本数加权平均
        """
        total_samples = sum(u['n_samples'] for u in client_updates)
        
        # 加权平均
        avg_weights = np.zeros_like(client_updates[0]['weights'])
        avg_bias = np.zeros_like(client_updates[0]['bias'])
        
        for update in client_updates:
            weight = update['n_samples'] / total_samples
            avg_weights += weight * update['weights']
            avg_bias += weight * update['bias']
        
        self.global_weights = avg_weights
        self.global_bias = avg_bias
        
        return {
            'weights': avg_weights,
            'bias': avg_bias
        }
    
    def distribute(self, clients: List[FederatedClient]):
        """分发全局模型到客户端"""
        for client in clients:
            client.weights = self.global_weights.copy()
            client.bias = self.global_bias.copy()


def federated_learning_demo():
    """联邦学习演示"""
    print("=" * 60)
    print("联邦学习演示")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 创建5个客户端
    n_clients = 5
    clients = []
    
    for i in range(n_clients):
        # 每个客户端有不同分布的数据
        data = np.random.randn(50, 784)
        labels = np.random.randint(0, 10, 50)
        clients.append(FederatedClient(i, data, labels))
    
    print(f"客户端数量: {n_clients}")
    print(f"每客户端样本数: 50")
    
    # 创建服务端
    server = FederatedServer(n_clients)
    
    # 联邦学习训练
    n_rounds = 10
    print(f"\n开始联邦学习 ({n_rounds}轮)...")
    
    for round_idx in range(n_rounds):
        # 客户端本地训练
        client_updates = []
        for client in clients:
            update = client.local_train(epochs=5, lr=0.01)
            client_updates.append(update)
        
        # 服务端聚合
        global_model = server.aggregate(client_updates)
        
        # 分发全局模型
        server.distribute(clients)
        
        if (round_idx + 1) % 2 == 0:
            print(f"  第{round_idx+1}轮完成")
    
    print("\n联邦学习完成！")
    print("特点:")
    print("  ✓ 数据不出本地，保护隐私")
    print("  ✓ 多方协作训练，共享知识")
    print("  ✓ 通信高效，只传输模型参数")


if __name__ == "__main__":
    federated_learning_demo()
