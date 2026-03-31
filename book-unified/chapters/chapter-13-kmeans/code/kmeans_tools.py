#!/usr/bin/env python3
"""
K-Means聚类工具集 - 补充代码
Chapter 13: K-Means聚类
"""

import numpy as np
from typing import Tuple, Optional, List
from collections import defaultdict


class KMeansScratch:
    """从零实现K-Means算法"""
    
    def __init__(self, n_clusters: int = 3, max_iter: int = 100, 
                 tol: float = 1e-4, random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.centroids = None
        self.labels = None
        self.inertia_ = None  # WCSS
        
    def fit(self, X: np.ndarray) -> 'KMeansScratch':
        """训练模型"""
        if self.random_state:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # 随机初始化中心
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx].copy()
        
        for iteration in range(self.max_iter):
            # E步骤：分配样本
            distances = self._compute_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # 保存旧中心
            old_centroids = self.centroids.copy()
            
            # M步骤：更新中心
            for k in range(self.n_clusters):
                mask = (self.labels == k)
                if mask.any():
                    self.centroids[k] = X[mask].mean(axis=0)
            
            # 检查收敛
            if np.all(np.abs(self.centroids - old_centroids) < self.tol):
                print(f"收敛于第 {iteration + 1} 次迭代")
                break
        
        # 计算WCSS
        self.inertia_ = self._compute_wcss(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测新样本的簇标签"""
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """训练并预测"""
        self.fit(X)
        return self.labels
    
    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """计算所有样本到所有中心的距离"""
        # 向量化计算
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return distances
    
    def _compute_wcss(self, X: np.ndarray) -> float:
        """计算Within-Cluster Sum of Squares"""
        wcss = 0.0
        for k in range(self.n_clusters):
            mask = (self.labels == k)
            if mask.any():
                cluster_points = X[mask]
                wcss += np.sum((cluster_points - self.centroids[k]) ** 2)
        return wcss
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """将样本转换到中心距离空间"""
        return self._compute_distances(X)


class KMeansPlusPlus(KMeansScratch):
    """K-Means++初始化"""
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """使用K-Means++初始化中心"""
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        # 随机选择第一个中心
        centroids[0] = X[np.random.randint(n_samples)]
        
        for k in range(1, self.n_clusters):
            # 计算每个样本到最近中心的距离
            distances = np.min([
                np.sum((X - centroids[i]) ** 2, axis=1)
                for i in range(k)
            ], axis=0)
            
            # 按概率选择下一个中心
            probs = distances / distances.sum()
            next_idx = np.random.choice(n_samples, p=probs)
            centroids[k] = X[next_idx]
        
        return centroids
    
    def fit(self, X: np.ndarray) -> 'KMeansPlusPlus':
        """训练模型（使用K-Means++初始化）"""
        if self.random_state:
            np.random.seed(self.random_state)
        
        # 使用K-Means++初始化
        self.centroids = self._initialize_centroids(X)
        
        # 调用父类的fit
        n_samples = X.shape[0]
        
        for iteration in range(self.max_iter):
            distances = self._compute_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            old_centroids = self.centroids.copy()
            
            for k in range(self.n_clusters):
                mask = (self.labels == k)
                if mask.any():
                    self.centroids[k] = X[mask].mean(axis=0)
            
            if np.all(np.abs(self.centroids - old_centroids) < self.tol):
                print(f"收敛于第 {iteration + 1} 次迭代")
                break
        
        self.inertia_ = self._compute_wcss(X)
        return self


def elbow_method(X: np.ndarray, k_range: range = range(1, 11), 
                 plot: bool = False) -> List[Tuple[int, float]]:
    """肘部法则确定最佳聚类数"""
    wcss_values = []
    
    for k in k_range:
        kmeans = KMeansScratch(n_clusters=k, random_state=42)
        kmeans.fit(X)
        wcss_values.append((k, kmeans.inertia_))
        print(f"k={k}, WCSS={kmeans.inertia_:.2f}")
    
    if plot:
        import matplotlib.pyplot as plt
        ks, wcss = zip(*wcss_values)
        plt.figure(figsize=(8, 5))
        plt.plot(ks, wcss, 'bo-')
        plt.xlabel('聚类数 k')
        plt.ylabel('WCSS')
        plt.title('肘部法则')
        plt.grid(True)
        plt.show()
    
    return wcss_values


def silhouette_score_manual(X: np.ndarray, labels: np.ndarray) -> float:
    """手动实现轮廓系数"""
    n_samples = len(X)
    scores = []
    
    for i in range(n_samples):
        # 同一簇的其他点
        same_cluster = X[labels == labels[i]]
        a = np.mean([
            np.linalg.norm(X[i] - x) 
            for x in same_cluster if not np.allclose(x, X[i])
        ]) if len(same_cluster) > 1 else 0
        
        # 最近的其他簇
        other_clusters = [
            X[labels == k] 
            for k in np.unique(labels) 
            if k != labels[i]
        ]
        b = min([
            np.mean([np.linalg.norm(X[i] - x) for x in cluster])
            for cluster in other_clusters
        ]) if other_clusters else 0
        
        score = (b - a) / max(a, b) if max(a, b) > 0 else 0
        scores.append(score)
    
    return np.mean(scores)


class MiniBatchKMeansScratch:
    """Mini-Batch K-Means实现"""
    
    def __init__(self, n_clusters: int = 3, batch_size: int = 100,
                 max_iter: int = 100, random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.centroids = None
        self.counts = None
    
    def fit(self, X: np.ndarray) -> 'MiniBatchKMeansScratch':
        """训练模型"""
        if self.random_state:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # 初始化
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx].copy()
        self.counts = np.zeros(self.n_clusters)
        
        for _ in range(self.max_iter):
            # 随机采样一个batch
            batch_idx = np.random.choice(n_samples, self.batch_size)
            batch = X[batch_idx]
            
            # 分配batch到最近中心
            distances = np.sqrt(((batch[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)
            
            # 增量更新中心
            for k in range(self.n_clusters):
                mask = (labels == k)
                if mask.any():
                    self.counts[k] += mask.sum()
                    # 增量平均
                    lr = mask.sum() / self.counts[k]
                    self.centroids[k] = (1 - lr) * self.centroids[k] + lr * batch[mask].mean(axis=0)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)


def visualize_clusters(X: np.ndarray, labels: np.ndarray, 
                       centroids: Optional[np.ndarray] = None,
                       title: str = "K-Means聚类结果"):
    """可视化聚类结果"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    
    # 绘制数据点
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='簇标签')
    
    # 绘制中心
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], 
                   c='red', marker='x', s=200, linewidths=3,
                   label='中心点')
    
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# 使用示例
if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    
    # 生成测试数据
    X, y_true = make_blobs(n_samples=300, centers=4, 
                          cluster_std=0.60, random_state=42)
    
    print("=== 标准K-Means ===")
    kmeans = KMeansScratch(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X)
    print(f"WCSS: {kmeans.inertia_:.2f}")
    
    print("\\n=== K-Means++ ===")
    kmeans_pp = KMeansPlusPlus(n_clusters=4, random_state=42)
    labels_pp = kmeans_pp.fit_predict(X)
    print(f"WCSS: {kmeans_pp.inertia_:.2f}")
    
    print("\\n=== Mini-Batch K-Means ===")
    mb_kmeans = MiniBatchKMeansScratch(n_clusters=4, batch_size=50, random_state=42)
    mb_kmeans.fit(X)
    labels_mb = mb_kmeans.predict(X)
    print("Mini-Batch完成")
    
    print("\\n=== 肘部法则 ===")
    elbow_method(X, k_range=range(1, 8))
