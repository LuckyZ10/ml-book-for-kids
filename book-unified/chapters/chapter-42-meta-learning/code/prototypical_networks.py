"""
Prototypical Networks - 原型网络实现
第42章：元学习与少样本学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def euclidean_distance(x, y):
    """
    计算欧氏距离
    
    Args:
        x: (batch, dim)
        y: (n_classes, dim)
    
    Returns:
        distances: (batch, n_classes)
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    
    # 扩展维度以便广播
    x = x.unsqueeze(1).expand(n, m, d)  # (n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)  # (n, m, d)
    
    # 计算欧氏距离的平方
    return torch.pow(x - y, 2).sum(2)  # (n, m)


class PrototypicalNetworks(nn.Module):
    """
    原型网络
    
    核心思想：每个类别用一个原型（该类样本的平均嵌入）表示
    分类时，将查询样本分配给最近的原型
    """
    def __init__(self, encoder):
        """
        Args:
            encoder: 编码器网络，将输入映射到嵌入空间
        """
        super().__init__()
        self.encoder = encoder
    
    def forward(self, support_x, support_y, query_x, n_classes):
        """
        前向传播
        
        Args:
            support_x: 支持集样本 (n_support, ...)
            support_y: 支持集标签 (n_support,)
            query_x: 查询集样本 (n_query, ...)
            n_classes: 类别数 N
        
        Returns:
            log_p_y: 对数概率 (n_query, n_classes)
        """
        # 1. 将支持集和查询集嵌入到同一空间
        support_embeddings = self.encoder(support_x)  # (n_support, dim)
        query_embeddings = self.encoder(query_x)      # (n_query, dim)
        
        # 2. 计算每个类别的原型（平均嵌入）
        prototypes = []
        for c in range(n_classes):
            # 找到属于类别c的支持集样本
            mask = (support_y == c)
            class_embeddings = support_embeddings[mask]
            # 原型 = 平均
            prototype = class_embeddings.mean(0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # (n_classes, dim)
        
        # 3. 计算查询样本到每个原型的距离
        distances = euclidean_distance(query_embeddings, prototypes)  # (n_query, n_classes)
        
        # 4. 将距离转换为概率
        # p(y=k|x) ∝ exp(-d(f(x), c_k))
        log_p_y = F.log_softmax(-distances, dim=1)
        
        return log_p_y
    
    def loss(self, support_x, support_y, query_x, query_y, n_classes):
        """
        计算损失和准确率
        
        Args:
            support_x, support_y: 支持集
            query_x, query_y: 查询集
            n_classes: 类别数
        
        Returns:
            loss: 负对数似然损失
            acc: 准确率
        """
        log_p_y = self.forward(support_x, support_y, query_x, n_classes)
        loss = F.nll_loss(log_p_y, query_y)
        acc = (log_p_y.argmax(1) == query_y).float().mean()
        return loss, acc
    
    def predict(self, support_x, support_y, query_x, n_classes):
        """
        预测查询样本的类别
        
        Returns:
            predictions: 预测的类别 (n_query,)
        """
        log_p_y = self.forward(support_x, support_y, query_x, n_classes)
        return log_p_y.argmax(1)


class MatchingNetworks(nn.Module):
    """
    匹配网络
    
    核心思想：使用注意力机制，加权支持集样本来预测查询样本
    """
    def __init__(self, encoder_f, encoder_g=None):
        """
        Args:
            encoder_f: 查询样本的编码器
            encoder_g: 支持集样本的编码器（可选，不同时使用）
        """
        super().__init__()
        self.encoder_f = encoder_f
        self.encoder_g = encoder_g if encoder_g is not None else encoder_f
    
    def forward(self, support_x, support_y, query_x):
        """
        前向传播
        
        Args:
            support_x: 支持集样本 (n_support, ...)
            support_y: 支持集标签 (n_support,)
            query_x: 查询集样本 (n_query, ...)
        
        Returns:
            predictions: 预测概率 (n_query, n_classes)
        """
        # 嵌入
        support_embeddings = self.encoder_g(support_x)  # (n_support, dim)
        query_embeddings = self.encoder_f(query_x)      # (n_query, dim)
        
        # 计算注意力权重（余弦相似度）
        # a(x_hat, x_i) = softmax(cos(f(x_hat), g(x_i)))
        support_norm = F.normalize(support_embeddings, dim=1)
        query_norm = F.normalize(query_embeddings, dim=1)
        
        # 相似度矩阵
        similarities = torch.mm(query_norm, support_norm.t())  # (n_query, n_support)
        attention = F.softmax(similarities, dim=1)  # (n_query, n_support)
        
        # 加权预测
        # 将支持集标签转换为one-hot
        n_classes = support_y.max().item() + 1
        support_labels_onehot = F.one_hot(support_y, n_classes).float()  # (n_support, n_classes)
        
        # 预测 = 注意力权重 × 支持集标签
        predictions = torch.mm(attention, support_labels_onehot)  # (n_query, n_classes)
        
        return predictions


if __name__ == "__main__":
    print("Prototypical Networks - 原型网络")
    print("=" * 50)
    print("\n核心思想：")
    print("1. 将样本嵌入到度量空间")
    print("2. 每个类别用原型（平均嵌入）表示")
    print("3. 分类时，查询样本分配给最近的原型")
    print("\n关键公式：")
    print("  原型: c_k = (1/|S_k|) Σ f(x_i)")
    print("  概率: p(y=k|x) ∝ exp(-||f(x) - c_k||²)")
    print("\n使用方法：")
    print("  encoder = SimpleCNN()")
    print("  proto_net = PrototypicalNetworks(encoder)")
    print("  log_p_y = proto_net(support_x, support_y, query_x, n_classes)")
    print("\nMatching Networks - 匹配网络")
    print("核心思想：使用注意力机制加权支持集样本")
