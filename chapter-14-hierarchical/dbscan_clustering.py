"""
DBSCAN密度聚类算法实现
第十四章：层次聚类与DBSCAN

基于Ester et al. (1996)的经典算法:
"A density-based algorithm for discovering clusters in large spatial databases with noise"
KDD 1996, pp. 226-231

作者: ML教材写作项目
日期: 2026-03-24
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Set, Tuple, Optional, Union
from collections import deque
import heapq


class DBSCAN:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    密度基于空间聚类算法
    
    核心思想：通过密度连接来发现任意形状的簇，并能识别噪声点
    
    参数:
        eps: ε-邻域半径，决定"邻居"的范围
        min_pts: 成为核心点所需的最小邻居数
        metric: 距离度量方式 ('euclidean', 'manhattan')
    
    属性:
        labels_: 每个点的聚类标签 (-1表示噪声)
        core_sample_indices_: 核心点的索引
        n_clusters_: 发现的簇数量
    """
    
    def __init__(self, eps: float = 0.5, min_pts: int = 5, 
                 metric: str = 'euclidean'):
        self.eps = eps
        self.min_pts = min_pts
        self.metric = metric
        
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = 0
        self._data = None
        self._n_samples = 0
        
    def _distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算两点之间的距离"""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x - y) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x - y))
        else:
            raise ValueError(f"不支持的度量方式: {self.metric}")
    
    def _region_query(self, point_idx: int) -> List[int]:
        """
        区域查询：找出点point_idx的ε-邻域内的所有点
        
        参数:
            point_idx: 查询点的索引
            
        返回:
            邻居点的索引列表
        """
        neighbors = []
        point = self._data[point_idx]
        
        for i in range(self._n_samples):
            if self._distance(point, self._data[i]) <= self.eps:
                neighbors.append(i)
        
        return neighbors
    
    def _expand_cluster(self, core_idx: int, neighbors: List[int], 
                        cluster_id: int) -> None:
        """
        扩展簇：从核心点开始，递归地将密度可达的点加入簇
        
        参数:
            core_idx: 核心点的索引
            neighbors: 核心点的邻居列表
            cluster_id: 当前簇的ID
        """
        # 使用队列进行广度优先搜索
        queue = deque(neighbors)
        self.labels_[core_idx] = cluster_id
        
        # 标记已处理的点
        processed = {core_idx}
        
        while queue:
            point_idx = queue.popleft()
            
            if point_idx in processed:
                continue
            processed.add(point_idx)
            
            # 如果该点还未被标记，加入当前簇
            if self.labels_[point_idx] == -1:
                self.labels_[point_idx] = cluster_id
            
            # 如果这个点也是核心点，扩展其邻居
            point_neighbors = self._region_query(point_idx)
            if len(point_neighbors) >= self.min_pts:
                # 是核心点，将其邻居加入队列
                for neighbor_idx in point_neighbors:
                    if neighbor_idx not in processed:
                        queue.append(neighbor_idx)
                        if self.labels_[neighbor_idx] == -1:
                            self.labels_[neighbor_idx] = cluster_id
    
    def fit(self, X: np.ndarray) -> 'DBSCAN':
        """
        执行DBSCAN聚类
        
        参数:
            X: 形状为(n_samples, n_features)的数据矩阵
            
        返回:
            self
        """
        self._data = np.array(X)
        self._n_samples = len(X)
        
        # 初始化所有点为噪声（-1）
        self.labels_ = np.full(self._n_samples, -1)
        self.core_sample_indices_ = []
        
        cluster_id = 0
        
        for i in range(self._n_samples):
            # 如果点已经被标记，跳过
            if self.labels_[i] != -1:
                continue
            
            # 找出当前点的邻居
            neighbors = self._region_query(i)
            
            # 检查是否是核心点
            if len(neighbors) >= self.min_pts:
                # 是核心点，开始一个新簇
                self.core_sample_indices_.append(i)
                self._expand_cluster(i, neighbors, cluster_id)
                cluster_id += 1
        
        self.n_clusters_ = cluster_id
        self.core_sample_indices_ = np.array(self.core_sample_indices_)
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """拟合数据并返回聚类标签"""
        self.fit(X)
        return self.labels_


class PointClassifier:
    """
    点分类器：将数据点分类为核心点、边界点或噪声点
    """
    
    CORE = 'core'
    BORDER = 'border'
    NOISE = 'noise'
    
    def __init__(self, eps: float = 0.5, min_pts: int = 5):
        self.eps = eps
        self.min_pts = min_pts
        
    def classify_points(self, X: np.ndarray, 
                        labels: np.ndarray) -> dict:
        """
        分类所有点
        
        返回:
            dict: {point_idx: 'core'/'border'/'noise'}
        """
        classifications = {}
        
        for i in range(len(X)):
            if labels[i] == -1:
                classifications[i] = self.NOISE
            else:
                # 检查是否是核心点
                neighbors = self._count_neighbors(X, i)
                if neighbors >= self.min_pts:
                    classifications[i] = self.CORE
                else:
                    classifications[i] = self.BORDER
        
        return classifications
    
    def _count_neighbors(self, X: np.ndarray, point_idx: int) -> int:
        """计算点的邻居数量"""
        point = X[point_idx]
        count = 0
        for i in range(len(X)):
            dist = np.sqrt(np.sum((point - X[i]) ** 2))
            if dist <= self.eps:
                count += 1
        return count


class DBSCANVisualizer:
    """DBSCAN结果可视化器"""
    
    @staticmethod
    def plot_clusters(X: np.ndarray, labels: np.ndarray, 
                      title: str = "DBSCAN Clustering",
                      core_indices: Optional[np.ndarray] = None):
        """
        绘制聚类结果
        
        参数:
            X: 数据点
            labels: 聚类标签
            title: 图表标题
            core_indices: 核心点索引（可选）
        """
        plt.figure(figsize=(12, 8))
        
        # 噪声点用黑色表示
        noise_mask = labels == -1
        if np.any(noise_mask):
            plt.scatter(X[noise_mask, 0], X[noise_mask, 1],
                       c='black', marker='x', s=50, label='Noise', alpha=0.6)
        
        # 绘制各个簇
        unique_labels = set(labels)
        unique_labels.discard(-1)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(X[mask, 0], X[mask, 1],
                       c=[color], marker='o', s=50, alpha=0.6,
                       label=f'Cluster {label}')
            
            # 高亮核心点
            if core_indices is not None:
                core_mask = mask.copy()
                core_points = np.intersect1d(np.where(mask)[0], core_indices)
                if len(core_points) > 0:
                    plt.scatter(X[core_points, 0], X[core_points, 1],
                               c=[color], marker='*', s=200,
                               edgecolors='black', linewidths=1.5,
                               label=f'Core Points (C{label})')
        
        plt.title(title, fontsize=14)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    @staticmethod
    def plot_k_distance_graph(X: np.ndarray, k: int = 4):
        """
        绘制k-距离图，帮助选择eps参数
        
        参数:
            X: 数据点
            k: 邻居数量（通常等于min_pts-1）
        """
        n = len(X)
        k_distances = []
        
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                    distances.append(dist)
            distances.sort()
            k_distances.append(distances[k-1] if k <= len(distances) else distances[-1])
        
        k_distances.sort(reverse=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n+1), k_distances, 'b-', linewidth=2)
        plt.xlabel('Points (sorted by distance)')
        plt.ylabel(f'{k}-th Nearest Neighbor Distance')
        plt.title(f'k-Distance Graph (k={k}) - For Choosing eps')
        plt.grid(True, alpha=0.3)
        
        # 尝试找到"肘部"
        if len(k_distances) > 10:
            # 简单的肘部检测
            diffs = np.diff(k_distances)
            elbow_idx = np.argmax(diffs[:len(diffs)//2]) if len(diffs) > 2 else 0
            if elbow_idx > 0:
                plt.axhline(y=k_distances[elbow_idx], color='r', linestyle='--',
                           label=f'Suggested eps ≈ {k_distances[elbow_idx]:.2f}')
                plt.legend()
        
        return plt.gcf()


class ParameterAnalyzer:
    """DBSCAN参数分析器"""
    
    @staticmethod
    def analyze_eps_sensitivity(X: np.ndarray, 
                                 eps_range: np.ndarray,
                                 min_pts: int = 5):
        """
        分析不同eps值对聚类结果的影响
        
        参数:
            X: 数据
            eps_range: 要测试的eps值范围
            min_pts: 最小点数
            
        返回:
            dict: 包含每个eps值对应的聚类数量和噪声点数量
        """
        results = {
            'eps': [],
            'n_clusters': [],
            'n_noise': [],
            'noise_ratio': []
        }
        
        for eps in eps_range:
            dbscan = DBSCAN(eps=eps, min_pts=min_pts)
            labels = dbscan.fit_predict(X)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_ratio = n_noise / len(X)
            
            results['eps'].append(eps)
            results['n_clusters'].append(n_clusters)
            results['n_noise'].append(n_noise)
            results['noise_ratio'].append(noise_ratio)
        
        return results
    
    @staticmethod
    def plot_parameter_analysis(X: np.ndarray,
                                 eps_range: np.ndarray = None,
                                 min_pts_range: range = None):
        """可视化参数分析结果"""
        if eps_range is None:
            eps_range = np.linspace(0.1, 2.0, 20)
        
        results = ParameterAnalyzer.analyze_eps_sensitivity(
            X, eps_range, min_pts=5 if min_pts_range is None else min(min_pts_range)
        )
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 簇数量 vs eps
        axes[0].plot(results['eps'], results['n_clusters'], 'bo-', linewidth=2)
        axes[0].set_xlabel('eps')
        axes[0].set_ylabel('Number of Clusters')
        axes[0].set_title('Clusters vs eps')
        axes[0].grid(True, alpha=0.3)
        
        # 噪声点数量 vs eps
        axes[1].plot(results['eps'], results['n_noise'], 'ro-', linewidth=2)
        axes[1].set_xlabel('eps')
        axes[1].set_ylabel('Number of Noise Points')
        axes[1].set_title('Noise Points vs eps')
        axes[1].grid(True, alpha=0.3)
        
        # 噪声比例 vs eps
        axes[2].plot(results['eps'], results['noise_ratio'], 'go-', linewidth=2)
        axes[2].set_xlabel('eps')
        axes[2].set_ylabel('Noise Ratio')
        axes[2].set_title('Noise Ratio vs eps')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# ==================== 演示函数 ====================

def demo_basic_dbscan():
    """基础DBSCAN演示"""
    from sklearn.datasets import make_moons, make_blobs
    
    # 创建数据：两个月牙形簇
    X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    # 执行DBSCAN
    dbscan = DBSCAN(eps=0.2, min_pts=5)
    labels = dbscan.fit_predict(X_moons)
    
    # 可视化
    DBSCANVisualizer.plot_clusters(
        X_moons, labels, 
        "DBSCAN on Moon-Shaped Data",
        dbscan.core_sample_indices_
    )
    plt.savefig('dbscan_moons.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"发现 {dbscan.n_clusters_} 个簇")
    print(f"噪声点数量: {list(labels).count(-1)}")
    print(f"核心点数量: {len(dbscan.core_sample_indices_)}")


def demo_comparison_with_kmeans():
    """DBSCAN与K-Means对比演示"""
    from sklearn.datasets import make_moons
    
    # 创建非球形数据
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.2, min_pts=5)
    labels_dbscan = dbscan.fit_predict(X)
    
    # 简单K-Means（用于对比）
    from chapter_13_kmeans.kmeans_clustering import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels_kmeans = kmeans.fit_predict(X)
    
    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # DBSCAN结果
    axes[0].scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='viridis', s=50)
    axes[0].set_title('DBSCAN (发现任意形状簇)')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    # K-Means结果
    axes[1].scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis', s=50)
    axes[1].scatter(kmeans.centroids_[:, 0], kmeans.centroids_[:, 1],
                   c='red', marker='X', s=200, label='Centroids')
    axes[1].set_title('K-Means (倾向于球形簇)')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('dbscan_vs_kmeans.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_parameter_selection():
    """参数选择演示"""
    from sklearn.datasets import make_moons
    
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    # 绘制k-距离图
    DBSCANVisualizer.plot_k_distance_graph(X, k=4)
    plt.savefig('k_distance_graph.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 参数敏感性分析
    eps_range = np.linspace(0.1, 0.5, 20)
    ParameterAnalyzer.plot_parameter_analysis(X, eps_range)
    plt.savefig('parameter_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_noise_handling():
    """噪声处理演示"""
    from sklearn.datasets import make_blobs
    
    # 创建带噪声的数据
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.6, 
                     random_state=42)
    
    # 添加随机噪声
    np.random.seed(42)
    noise_points = np.random.uniform(-3, 3, (30, 2))
    X_with_noise = np.vstack([X, noise_points])
    
    # 执行DBSCAN
    dbscan = DBSCAN(eps=0.8, min_pts=5)
    labels = dbscan.fit_predict(X_with_noise)
    
    # 可视化
    DBSCANVisualizer.plot_clusters(
        X_with_noise, labels,
        "DBSCAN: Automatic Noise Detection",
        dbscan.core_sample_indices_
    )
    plt.savefig('dbscan_noise.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"原始数据点: {len(X)}")
    print(f"添加的噪声点: {len(noise_points)}")
    print(f"DBSCAN识别的噪声点: {list(labels).count(-1)}")


if __name__ == "__main__":
    print("=" * 60)
    print("DBSCAN密度聚类算法演示")
    print("=" * 60)
    
    print("\n【演示1】基础DBSCAN - 月牙形数据")
    demo_basic_dbscan()
    
    print("\n【演示2】DBSCAN vs K-Means 对比")
    demo_comparison_with_kmeans()
    
    print("\n【演示3】参数选择技巧")
    demo_parameter_selection()
    
    print("\n【演示4】自动噪声检测")
    demo_noise_handling()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
