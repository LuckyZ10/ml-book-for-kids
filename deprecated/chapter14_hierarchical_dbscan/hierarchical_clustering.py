"""
层次聚类实现 (Hierarchical Clustering)
基于 Lance & Williams (1967) 递推公式

包含：
- AGNES (凝聚式)
- DIANA (分裂式)
- 树状图可视化
- 多种连接方法
"""

import numpy as np
from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import defaultdict


@dataclass
class ClusterNode:
    """树状图节点"""
    id: int
    left: Optional['ClusterNode'] = None
    right: Optional['ClusterNode'] = None
    distance: float = 0.0
    size: int = 1
    points: List[int] = None
    
    def __post_init__(self):
        if self.points is None:
            self.points = [self.id]


class AgglomerativeClustering:
    """
    凝聚式层次聚类 (AGNES)
    
    基于 Lance-Williams 递推公式实现
    
    Parameters
    ----------
    n_clusters : int
        聚类数量
    method : str
        连接方法: 'single', 'complete', 'average', 'ward', 'centroid', 'median'
    metric : str
        距离度量: 'euclidean', 'manhattan', 'cosine'
    
    References
    ----------
    Lance, G.N. & Williams, W.T. (1967). A general theory of classificatory
    sorting strategies. The Computer Journal, 9(4), 373-380.
    
    Kaufman, L. & Rousseeuw, P.J. (1990). Finding Groups in Data.
    """
    
    METHODS = ['single', 'complete', 'average', 'ward', 'centroid', 'median', 'weighted']
    
    def __init__(
        self,
        n_clusters: int = 2,
        method: Literal['single', 'complete', 'average', 'ward', 'centroid', 'median', 'weighted'] = 'single',
        metric: str = 'euclidean'
    ):
        self.n_clusters = n_clusters
        self.method = method
        self.metric = metric
        self.linkage_matrix_ = None
        self.labels_ = None
        self.n_samples_ = None
        
    def _compute_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算两点间距离"""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x - y) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x - y))
        elif self.metric == 'cosine':
            return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-10)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _compute_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """计算距离矩阵"""
        n = len(X)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._compute_distance(X[i], X[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix
    
    def _lance_williams_params(
        self,
        size_i: int,
        size_j: int,
        size_k: int
    ) -> Tuple[float, float, float, float]:
        """
        计算Lance-Williams递推公式参数
        
        d(k, ij) = αᵢ·d(k,i) + αⱼ·d(k,j) + β·d(i,j) + γ·|d(k,i) - d(k,j)|
        """
        n_total = size_i + size_j
        
        if self.method == 'single':
            # 单连接: min(d_ki, d_kj)
            alpha_i, alpha_j = 0.5, 0.5
            beta, gamma = 0.0, -0.5
            
        elif self.method == 'complete':
            # 全连接: max(d_ki, d_kj)
            alpha_i, alpha_j = 0.5, 0.5
            beta, gamma = 0.0, 0.5
            
        elif self.method == 'average':
            # 平均连接 (UPGMA)
            alpha_i = size_i / n_total
            alpha_j = size_j / n_total
            beta, gamma = 0.0, 0.0
            
        elif self.method == 'weighted':
            # 加权平均 (WPGMA)
            alpha_i, alpha_j = 0.5, 0.5
            beta, gamma = 0.0, 0.0
            
        elif self.method == 'centroid':
            # 质心法 (UPGMC)
            alpha_i = size_i / n_total
            alpha_j = size_j / n_total
            beta = -alpha_i * alpha_j
            gamma = 0.0
            
        elif self.method == 'median':
            # 中间距离 (WPGMC)
            alpha_i, alpha_j = 0.5, 0.5
            beta = -0.25
            gamma = 0.0
            
        elif self.method == 'ward':
            # Ward法: 最小化方差
            T = size_i + size_j + size_k
            alpha_i = (size_i + size_k) / T
            alpha_j = (size_j + size_k) / T
            beta = -size_k / T
            gamma = 0.0
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        return alpha_i, alpha_j, beta, gamma
    
    def _update_distance(
        self,
        d_ki: float,
        d_kj: float,
        d_ij: float,
        size_i: int,
        size_j: int,
        size_k: int
    ) -> float:
        """
        使用Lance-Williams公式更新距离
        """
        alpha_i, alpha_j, beta, gamma = self._lance_williams_params(
            size_i, size_j, size_k
        )
        
        return (
            alpha_i * d_ki +
            alpha_j * d_kj +
            beta * d_ij +
            gamma * abs(d_ki - d_kj)
        )
    
    def fit(self, X: np.ndarray) -> 'AgglomerativeClustering':
        """
        训练层次聚类
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            训练数据
            
        Returns
        -------
        self
        """
        X = np.asarray(X)
        self.n_samples_ = n = len(X)
        
        if n < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters")
        
        # 初始化距离矩阵
        dist_matrix = self._compute_distance_matrix(X)
        
        # 初始化：每个点是一个簇
        clusters = {i: [i] for i in range(n)}
        cluster_sizes = {i: 1 for i in range(n)}
        
        # 用于构建linkage矩阵
        # linkage格式: [idx1, idx2, distance, size]
        linkage = []
        next_cluster_id = n
        
        # 活跃簇的列表
        active = set(range(n))
        
        # 层次聚类主循环
        while len(active) > 1:
            # 找到距离最近的两个簇
            min_dist = float('inf')
            to_merge = None
            
            active_list = list(active)
            for i, ci in enumerate(active_list):
                for cj in active_list[i + 1:]:
                    # 计算簇ci和cj之间的距离
                    dist = self._compute_cluster_distance(
                        ci, cj, clusters, dist_matrix
                    )
                    if dist < min_dist:
                        min_dist = dist
                        to_merge = (ci, cj)
            
            if to_merge is None:
                break
                
            ci, cj = to_merge
            
            # 记录合并
            linkage.append([
                ci, cj, min_dist,
                cluster_sizes[ci] + cluster_sizes[cj]
            ])
            
            # 合并簇
            new_cluster = next_cluster_id
            clusters[new_cluster] = clusters[ci] + clusters[cj]
            cluster_sizes[new_cluster] = cluster_sizes[ci] + cluster_sizes[cj]
            
            # 更新活跃簇集合
            active.remove(ci)
            active.remove(cj)
            active.add(new_cluster)
            
            next_cluster_id += 1
        
        self.linkage_matrix_ = np.array(linkage)
        
        # 生成标签
        self._generate_labels()
        
        return self
    
    def _compute_cluster_distance(
        self,
        ci: int,
        cj: int,
        clusters: dict,
        dist_matrix: np.ndarray
    ) -> float:
        """计算两个簇之间的距离"""
        points_i = clusters[ci]
        points_j = clusters[cj]
        
        if self.method == 'single':
            # 最小距离
            return min(dist_matrix[p1, p2] for p1 in points_i for p2 in points_j)
        
        elif self.method == 'complete':
            # 最大距离
            return max(dist_matrix[p1, p2] for p1 in points_i for p2 in points_j)
        
        elif self.method in ['average', 'ward']:
            # 平均距离
            distances = [dist_matrix[p1, p2] for p1 in points_i for p2 in points_j]
            return np.mean(distances)
        
        else:
            # 默认使用平均距离
            distances = [dist_matrix[p1, p2] for p1 in points_i for p2 in points_j]
            return np.mean(distances)
    
    def _generate_labels(self):
        """根据linkage矩阵生成聚类标签"""
        n = self.n_samples_
        
        # 构建父节点映射
        parent = {i: i for i in range(n)}
        
        for idx1, idx2, dist, size in self.linkage_matrix_:
            new_id = int(max(parent.keys()) + 1) if parent else n
            parent[idx1] = new_id
            parent[idx2] = new_id
            parent[new_id] = new_id
        
        # 找到每个原始点的根节点
        def find_root(x):
            while parent.get(x, x) != x:
                x = parent.get(x, x)
            return x
        
        # 获取最终的簇
        clusters = defaultdict(list)
        for i in range(n):
            root = find_root(i)
            clusters[root].append(i)
        
        # 如果簇太多，需要继续合并
        while len(clusters) > self.n_clusters:
            # 简化的合并策略
            keys = list(clusters.keys())
            for i in range(len(keys) - 1, self.n_clusters - 1, -1):
                # 合并到第一个簇
                clusters[keys[0]].extend(clusters[keys[i]])
                del clusters[keys[i]]
        
        # 生成标签
        self.labels_ = np.zeros(n, dtype=int)
        for label, points in enumerate(clusters.values()):
            for p in points:
                self.labels_[p] = label
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """训练并返回标签"""
        self.fit(X)
        return self.labels_
    
    def plot_dendrogram(self, max_d: Optional[float] = None, **kwargs):
        """
        绘制树状图
        
        Parameters
        ----------
        max_d : float, optional
            切割高度
        """
        if self.linkage_matrix_ is None:
            raise ValueError("Model not fitted yet!")
        
        try:
            from scipy.cluster.hierarchy import dendrogram
            
            plt.figure(figsize=(12, 6))
            dendrogram(
                self.linkage_matrix_,
                truncate_mode='lastp',
                p=30,
                leaf_rotation=90,
                leaf_font_size=12,
                show_contracted=True,
                **kwargs
            )
            
            plt.title(f'Hierarchical Clustering Dendrogram ({self.method})')
            plt.xlabel('Sample Index or (Cluster Size)')
            plt.ylabel('Distance')
            
            if max_d:
                plt.axhline(y=max_d, c='r', linestyle='--', label=f'cut at {max_d}')
                plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            # 简化版树状图
            self._plot_simple_dendrogram()
    
    def _plot_simple_dendrogram(self):
        """绘制简化版树状图"""
        n = len(self.linkage_matrix_) + 1
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 节点位置
        positions = {i: (i, 0) for i in range(n)}
        
        for i, (idx1, idx2, dist, size) in enumerate(self.linkage_matrix_):
            x1, y1 = positions[int(idx1)]
            x2, y2 = positions[int(idx2)]
            
            new_x = (x1 + x2) / 2
            new_y = dist
            
            # 绘制连接线
            ax.plot([x1, x1], [y1, dist], 'b-', linewidth=1)
            ax.plot([x2, x2], [y2, dist], 'b-', linewidth=1)
            ax.plot([x1, x2], [dist, dist], 'b-', linewidth=1)
            
            positions[n + i] = (new_x, new_y)
        
        ax.set_title(f'Hierarchical Clustering ({self.method})')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Distance')
        ax.set_xticks(range(n))
        ax.set_xticklabels([f'{i}' for i in range(n)])
        
        plt.tight_layout()
        plt.show()


class DivisiveClustering:
    """
    分裂式层次聚类 (DIANA)
    
    自顶向下的分裂策略
    
    References
    ----------
    Kaufman, L. & Rousseeuw, P.J. (1990). Finding Groups in Data.
    """
    
    def __init__(self, n_clusters: int = 2, metric: str = 'euclidean'):
        self.n_clusters = n_clusters
        self.metric = metric
        self.labels_ = None
    
    def _dissimilarity(self, point: int, cluster: List[int], X: np.ndarray) -> float:
        """计算点到簇的平均不相似度"""
        if len(cluster) <= 1:
            return 0.0
        
        distances = []
        for other in cluster:
            if other != point:
                dist = np.linalg.norm(X[point] - X[other])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _split_cluster(self, cluster: List[int], X: np.ndarray) -> Tuple[List[int], List[int]]:
        """分裂一个簇为两个"""
        if len(cluster) <= 1:
            return cluster, []
        
        # 找到最不和谐的点（不相似度最高）
        dissimilarities = [
            (self._dissimilarity(p, cluster, X), p)
            for p in cluster
        ]
        
        # 初始分裂：最不像的点和其余
        dissimilarities.sort(reverse=True)
        splinter = [dissimilarities[0][1]]
        remain = [p for _, p in dissimilarities[1:]]
        
        # 迭代改进
        improved = True
        while improved and remain:
            improved = False
            
            # 检查remain中的点是否应该移动到splinter
            for p in remain[:]:
                d_remain = self._dissimilarity(p, remain, X)
                d_splinter = self._dissimilarity(p, splinter, X)
                
                if d_splinter < d_remain:
                    splinter.append(p)
                    remain.remove(p)
                    improved = True
            
            # 检查splinter中的点是否应该移回
            for p in splinter[:]:
                d_splinter = self._dissimilarity(p, splinter, X)
                d_remain = self._dissimilarity(p, remain + [p], X)
                
                if d_remain < d_splinter:
                    remain.append(p)
                    splinter.remove(p)
                    improved = True
        
        return splinter, remain
    
    def fit(self, X: np.ndarray) -> 'DivisiveClustering':
        """训练分裂式层次聚类"""
        X = np.asarray(X)
        n = len(X)
        
        # 初始：所有点在一个簇
        clusters = [list(range(n))]
        
        # 逐步分裂
        while len(clusters) < self.n_clusters:
            # 找到直径最大的簇（直径 = 最大不相似度）
            largest_cluster_idx = max(
                range(len(clusters)),
                key=lambda i: max(
                    np.linalg.norm(X[p1] - X[p2])
                    for p1 in clusters[i]
                    for p2 in clusters[i]
                ) if len(clusters[i]) > 1 else 0
            )
            
            to_split = clusters[largest_cluster_idx]
            
            if len(to_split) <= 1:
                break
            
            # 分裂
            c1, c2 = self._split_cluster(to_split, X)
            
            # 替换原簇
            clusters[largest_cluster_idx] = c1
            clusters.append(c2)
        
        # 生成标签
        self.labels_ = np.zeros(n, dtype=int)
        for label, cluster in enumerate(clusters):
            for p in cluster:
                self.labels_[p] = label
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """训练并返回标签"""
        self.fit(X)
        return self.labels_


# ==================== 工具函数 ====================

def compare_linkage_methods(X: np.ndarray, n_clusters: int = 3):
    """
    比较不同连接方法的聚类效果
    
    Parameters
    ----------
    X : np.ndarray
        数据
    n_clusters : int
        聚类数量
    """
    methods = ['single', 'complete', 'average', 'ward']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for ax, method in zip(axes, methods):
        agg = AgglomerativeClustering(n_clusters=n_clusters, method=method)
        labels = agg.fit_predict(X)
        
        # 绘制散点图
        for label in np.unique(labels):
            mask = labels == label
            ax.scatter(X[mask, 0], X[mask, 1], label=f'Cluster {label}', alpha=0.7)
        
        ax.set_title(f'{method.capitalize()} Linkage')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comparison of Linkage Methods', fontsize=14)
    plt.tight_layout()
    plt.show()


# ==================== 演示 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("层次聚类演示 (Hierarchical Clustering Demo)")
    print("=" * 60)
    
    # 示例1：动物分类
    print("\n📊 示例1: 动物分类")
    print("-" * 40)
    
    animals = np.array([
        [30, 80],   # 狗: 体重30kg, 体长80cm
        [35, 85],   # 狼
        [4, 40],    # 猫
        [200, 250], # 老虎
        [190, 240]  # 狮子
    ])
    animal_names = ['狗', '狼', '猫', '老虎', '狮子']
    
    print("动物数据 (体重kg, 体长cm):")
    for name, (w, l) in zip(animal_names, animals):
        print(f"  {name}: ({w}, {l})")
    
    # 使用不同方法聚类
    for method in ['single', 'complete', 'average', 'ward']:
        agg = AgglomerativeClustering(n_clusters=2, method=method)
        labels = agg.fit_predict(animals)
        print(f"\n{method.capitalize()} 方法:")
        for name, label in zip(animal_names, labels):
            print(f"  {name} -> 簇 {label}")
    
    # 示例2：二维数据聚类
    print("\n\n📊 示例2: 二维数据聚类")
    print("-" * 40)
    
    np.random.seed(42)
    
    # 生成三组数据
    cluster1 = np.random.randn(30, 2) + [0, 0]
    cluster2 = np.random.randn(30, 2) + [5, 5]
    cluster3 = np.random.randn(30, 2) + [0, 5]
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    agg = AgglomerativeClustering(n_clusters=3, method='ward')
    labels = agg.fit_predict(X)
    
    print(f"数据点数量: {len(X)}")
    print(f"聚类数量: {len(np.unique(labels))}")
    print(f"每个簇的大小: {[sum(labels == i) for i in range(3)]}")
    
    # 示例3：分裂式聚类
    print("\n\n📊 示例3: 分裂式聚类 (DIANA)")
    print("-" * 40)
    
    diana = DivisiveClustering(n_clusters=3)
    labels = diana.fit_predict(X[:50])  # 使用部分数据
    
    print(f"DIANA聚类结果: {len(np.unique(labels))} 个簇")
    print(f"簇分布: {[sum(labels == i) for i in range(3)]}")
    
    print("\n" + "=" * 60)
    print("演示完成! 层次聚类算法已准备就绪。")
    print("=" * 60)
