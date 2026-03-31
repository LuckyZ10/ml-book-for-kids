"""
层次聚类 NumPy 实现
第十四章：层次聚类与DBSCAN

包含：
- AGNES (凝聚式层次聚类) - 支持 single, complete, average, ward 四种 linkage
- DIANA (分裂式层次聚类)
- Lance-Williams 递推公式
- 树状图可视化
- 距离矩阵优化计算

作者: ML教材写作项目
日期: 2026-03-30
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Literal
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class ClusterNode:
    """树状图节点，用于表示层次结构"""
    id: int
    left: Optional['ClusterNode'] = None
    right: Optional['ClusterNode'] = None
    distance: float = 0.0
    size: int = 1
    point_indices: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.point_indices is None:
            self.point_indices = [self.id]
    
    def get_leaves(self) -> List[int]:
        """获取该节点下所有叶子节点的索引"""
        if self.left is None and self.right is None:
            return [self.id]
        leaves = []
        if self.left:
            leaves.extend(self.left.get_leaves())
        if self.right:
            leaves.extend(self.right.get_leaves())
        return leaves


class AgglomerativeClusteringNumPy:
    """
    凝聚式层次聚类 (AGNES) - NumPy优化实现
    
    基于 Lance-Williams 递推公式的高效实现
    
    Parameters
    ----------
    n_clusters : int
        目标聚类数量
    linkage : {'single', 'complete', 'average', 'ward'}
        链接方法
    metric : str
        距离度量 ('euclidean', 'manhattan', 'cosine')
    
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        聚类标签
    linkage_matrix_ : ndarray of shape (n_samples-1, 4)
        链接矩阵，用于绘制树状图
        格式: [idx1, idx2, distance, sample_count]
    
    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    >>> agg = AgglomerativeClusteringNumPy(n_clusters=3, linkage='ward')
    >>> labels = agg.fit_predict(X)
    """
    
    LINKAGE_METHODS = ['single', 'complete', 'average', 'ward', 'centroid', 'median']
    
    def __init__(
        self,
        n_clusters: int = 2,
        linkage: Literal['single', 'complete', 'average', 'ward', 'centroid', 'median'] = 'single',
        metric: str = 'euclidean'
    ):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.labels_ = None
        self.linkage_matrix_ = None
        self.n_samples_ = None
        self._distance_matrix = None
        
    def _compute_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        高效计算距离矩阵
        
        使用向量化运算避免显式循环
        """
        if self.metric == 'euclidean':
            # 欧氏距离: ||x - y||² = ||x||² + ||y||² - 2x·y
            sq_norms = np.sum(X ** 2, axis=1).reshape(-1, 1)
            dist_matrix = sq_norms + sq_norms.T - 2 * np.dot(X, X.T)
            # 处理数值误差
            dist_matrix = np.maximum(dist_matrix, 0)
            return np.sqrt(dist_matrix)
        
        elif self.metric == 'manhattan':
            # 曼哈顿距离
            n = X.shape[0]
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                dist_matrix[i] = np.sum(np.abs(X - X[i]), axis=1)
            return dist_matrix
        
        elif self.metric == 'cosine':
            # 余弦距离 = 1 - 余弦相似度
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
            return 1 - np.dot(X_norm, X_norm.T)
        
        else:
            raise ValueError(f"不支持的度量方式: {self.metric}")
    
    def _lance_williams_params(
        self,
        size_i: int,
        size_j: int,
        size_k: int
    ) -> Tuple[float, float, float, float]:
        """
        Lance-Williams递推公式参数
        
        计算合并簇i和j后，新簇与簇k的距离公式参数：
        d(k, ij) = αi·d(k,i) + αj·d(k,j) + β·d(i,j) + γ·|d(k,i) - d(k,j)|
        
        Returns
        -------
        alpha_i, alpha_j, beta, gamma : float
        """
        n_total = size_i + size_j
        
        if self.linkage == 'single':
            # 单链接: min(d_ki, d_kj)
            return 0.5, 0.5, 0.0, -0.5
            
        elif self.linkage == 'complete':
            # 全链接: max(d_ki, d_kj)
            return 0.5, 0.5, 0.0, 0.5
            
        elif self.linkage == 'average':
            # 平均链接 (UPGMA)
            return size_i / n_total, size_j / n_total, 0.0, 0.0
            
        elif self.linkage == 'centroid':
            # 质心法
            alpha_i = size_i / n_total
            alpha_j = size_j / n_total
            beta = -alpha_i * alpha_j
            return alpha_i, alpha_j, beta, 0.0
            
        elif self.linkage == 'median':
            # 中间距离
            return 0.5, 0.5, -0.25, 0.0
            
        elif self.linkage == 'ward':
            # Ward法: 最小化方差增量
            T = size_i + size_j + size_k
            return (size_i + size_k) / T, (size_j + size_k) / T, -size_k / T, 0.0
            
        else:
            raise ValueError(f"不支持的链接方法: {self.linkage}")
    
    def _update_distance(
        self,
        d_ki: float,
        d_kj: float,
        d_ij: float,
        size_i: int,
        size_j: int,
        size_k: int
    ) -> float:
        """使用Lance-Williams公式更新距离"""
        alpha_i, alpha_j, beta, gamma = self._lance_williams_params(
            size_i, size_j, size_k
        )
        
        return (
            alpha_i * d_ki +
            alpha_j * d_kj +
            beta * d_ij +
            gamma * abs(d_ki - d_kj)
        )
    
    def fit(self, X: np.ndarray) -> 'AgglomerativeClusteringNumPy':
        """
        训练层次聚类模型
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            训练数据
        
        Returns
        -------
        self
        """
        X = np.asarray(X)
        self.n_samples_ = n = X.shape[0]
        
        if n < self.n_clusters:
            raise ValueError(f"样本数{n}必须大于等于聚类数{self.n_clusters}")
        
        # 计算距离矩阵
        self._distance_matrix = self._compute_distance_matrix(X)
        
        # 初始化：每个点是一个簇
        # 使用列表存储每个簇包含的原始点索引
        clusters = {i: [i] for i in range(n)}
        cluster_sizes = {i: 1 for i in range(n)}
        
        # 初始化簇间距离矩阵（复用点间距离）
        cluster_dist = self._distance_matrix.copy()
        np.fill_diagonal(cluster_dist, np.inf)  # 对角线设为无穷大
        
        # 链接矩阵: [idx1, idx2, distance, n_samples]
        linkage = []
        next_cluster_id = n
        
        # 活跃簇集合
        active = set(range(n))
        
        print(f"[AGNES] 开始聚类: n_samples={n}, linkage='{self.linkage}'")
        
        # 主循环：合并直到达到目标簇数
        while len(active) > self.n_clusters:
            # 找到距离最近的两个簇
            min_dist = np.inf
            to_merge = None
            
            active_list = list(active)
            for idx, i in enumerate(active_list):
                for j in active_list[idx + 1:]:
                    if cluster_dist[i, j] < min_dist:
                        min_dist = cluster_dist[i, j]
                        to_merge = (i, j)
            
            if to_merge is None or min_dist == np.inf:
                break
            
            ci, cj = to_merge
            
            # 记录合并信息
            linkage.append([
                ci, cj, min_dist,
                cluster_sizes[ci] + cluster_sizes[cj]
            ])
            
            # 创建新簇
            new_cluster_id = next_cluster_id
            clusters[new_cluster_id] = clusters[ci] + clusters[cj]
            cluster_sizes[new_cluster_id] = cluster_sizes[ci] + cluster_sizes[cj]
            
            # 更新距离矩阵
            for k in active:
                if k != ci and k != cj:
                    new_dist = self._update_distance(
                        cluster_dist[ci, k],
                        cluster_dist[cj, k],
                        cluster_dist[ci, cj],
                        cluster_sizes[ci],
                        cluster_sizes[cj],
                        cluster_sizes[k]
                    )
                    cluster_dist[new_cluster_id, k] = new_dist
                    cluster_dist[k, new_cluster_id] = new_dist
            
            # 标记旧簇为不活跃
            cluster_dist[ci, :] = np.inf
            cluster_dist[:, ci] = np.inf
            cluster_dist[cj, :] = np.inf
            cluster_dist[:, cj] = np.inf
            
            active.remove(ci)
            active.remove(cj)
            active.add(new_cluster_id)
            
            next_cluster_id += 1
        
        self.linkage_matrix_ = np.array(linkage)
        
        # 生成标签
        self.labels_ = np.zeros(n, dtype=int)
        for label, cluster_id in enumerate(active):
            for point_idx in clusters[cluster_id]:
                self.labels_[point_idx] = label
        
        print(f"[AGNES] 聚类完成: 发现 {len(active)} 个簇")
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """训练并返回聚类标签"""
        self.fit(X)
        return self.labels_
    
    def plot_dendrogram(self, max_d: Optional[float] = None, 
                        figsize: Tuple[int, int] = (12, 6)):
        """
        绘制树状图
        
        Parameters
        ----------
        max_d : float, optional
            在指定高度画切割线
        figsize : tuple
            图像大小
        """
        if self.linkage_matrix_ is None:
            raise ValueError("请先调用fit()")
        
        try:
            from scipy.cluster.hierarchy import dendrogram
            
            plt.figure(figsize=figsize)
            dendrogram(
                self.linkage_matrix_,
                truncate_mode='lastp',
                p=30,
                leaf_rotation=90,
                leaf_font_size=10,
                show_contracted=True
            )
            
            plt.title(f'Hierarchical Clustering Dendrogram ({self.linkage})')
            plt.xlabel('Sample Index or (Cluster Size)')
            plt.ylabel('Distance')
            
            if max_d:
                plt.axhline(y=max_d, c='r', linestyle='--', 
                           label=f'cut at {max_d:.2f}')
                plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("请安装scipy以使用树状图功能: pip install scipy")


class DivisiveClusteringNumPy:
    """
    分裂式层次聚类 (DIANA) - NumPy实现
    
    自顶向下的分裂策略，每次将直径最大的簇分裂
    
    References
    ----------
    Kaufman, L. & Rousseeuw, P.J. (1990). Finding Groups in Data.
    """
    
    def __init__(self, n_clusters: int = 2, metric: str = 'euclidean'):
        self.n_clusters = n_clusters
        self.metric = metric
        self.labels_ = None
    
    def _compute_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """计算距离矩阵"""
        sq_norms = np.sum(X ** 2, axis=1).reshape(-1, 1)
        dist_matrix = sq_norms + sq_norms.T - 2 * np.dot(X, X.T)
        return np.sqrt(np.maximum(dist_matrix, 0))
    
    def _compute_diameter(self, cluster_indices: List[int], 
                          dist_matrix: np.ndarray) -> float:
        """计算簇的直径（最远距离）"""
        if len(cluster_indices) <= 1:
            return 0.0
        sub_dist = dist_matrix[np.ix_(cluster_indices, cluster_indices)]
        return np.max(sub_dist)
    
    def _compute_dissimilarity(self, point_idx: int, 
                               cluster_indices: List[int],
                               dist_matrix: np.ndarray) -> float:
        """计算点到簇的平均不相似度"""
        if len(cluster_indices) <= 1:
            return 0.0
        
        other_indices = [i for i in cluster_indices if i != point_idx]
        if not other_indices:
            return 0.0
        
        return np.mean(dist_matrix[point_idx, other_indices])
    
    def _split_cluster(self, cluster_indices: List[int],
                       dist_matrix: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        分裂一个簇为两个
        
        使用迭代优化找到最佳分裂
        """
        if len(cluster_indices) <= 1:
            return cluster_indices, []
        
        # 找到最不和谐的点（不相似度最高）
        dissimilarities = [
            (self._compute_dissimilarity(p, cluster_indices, dist_matrix), p)
            for p in cluster_indices
        ]
        dissimilarities.sort(reverse=True)
        
        # 初始分裂
        splinter = [dissimilarities[0][1]]
        remain = [p for _, p in dissimilarities[1:]]
        
        # 迭代改进
        improved = True
        max_iterations = 100
        iteration = 0
        
        while improved and remain and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # 检查remain中的点是否应该移动到splinter
            for p in remain[:]:
                d_remain = self._compute_dissimilarity(p, remain, dist_matrix)
                d_splinter = self._compute_dissimilarity(p, splinter, dist_matrix)
                
                if d_splinter < d_remain:
                    splinter.append(p)
                    remain.remove(p)
                    improved = True
            
            # 检查splinter中的点是否应该移回
            for p in splinter[:]:
                d_splinter = self._compute_dissimilarity(p, splinter, dist_matrix)
                d_remain_with_p = self._compute_dissimilarity(
                    p, remain + [p], dist_matrix
                )
                
                if d_remain_with_p < d_splinter:
                    remain.append(p)
                    splinter.remove(p)
                    improved = True
        
        return splinter, remain
    
    def fit(self, X: np.ndarray) -> 'DivisiveClusteringNumPy':
        """训练分裂式层次聚类"""
        X = np.asarray(X)
        n = X.shape[0]
        
        dist_matrix = self._compute_distance_matrix(X)
        
        # 初始：所有点在一个簇
        clusters = [list(range(n))]
        
        # 逐步分裂
        while len(clusters) < self.n_clusters:
            # 找到直径最大的簇
            diameters = [
                self._compute_diameter(c, dist_matrix) for c in clusters
            ]
            largest_idx = np.argmax(diameters)
            
            to_split = clusters[largest_idx]
            if len(to_split) <= 1:
                break
            
            # 分裂
            c1, c2 = self._split_cluster(to_split, dist_matrix)
            
            # 替换原簇
            clusters[largest_idx] = c1
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


def compare_linkage_methods(X: np.ndarray, n_clusters: int = 3):
    """
    比较不同链接方法的聚类效果
    
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
        agg = AgglomerativeClusteringNumPy(n_clusters=n_clusters, 
                                           linkage=method)
        labels = agg.fit_predict(X)
        
        # 绘制散点图
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, 
                           cmap='viridis', alpha=0.7, edgecolors='k', s=50)
        ax.set_title(f'{method.capitalize()} Linkage', fontsize=12)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comparison of Linkage Methods', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def demo():
    """层次聚类演示"""
    from sklearn.datasets import make_blobs, make_moons
    
    print("=" * 60)
    print("层次聚类 NumPy 实现演示")
    print("=" * 60)
    
    # 示例1：简单数据
    print("\n【示例1】简单二维数据聚类")
    print("-" * 40)
    
    np.random.seed(42)
    X1 = np.array([
        [1, 1], [1.5, 1.5], [1.2, 0.8],  # 簇A
        [5, 5], [5.5, 5.5], [4.8, 5.2],  # 簇B
        [9, 1], [9.5, 1.5], [8.8, 1.2],  # 簇C
    ])
    
    for method in ['single', 'complete', 'average', 'ward']:
        agg = AgglomerativeClusteringNumPy(n_clusters=3, linkage=method)
        labels = agg.fit_predict(X1)
        print(f"{method:10s}: 标签 = {labels}")
    
    # 示例2：链状数据（展示single linkage的特性）
    print("\n【示例2】链状数据 - Single Linkage vs Complete Linkage")
    print("-" * 40)
    
    X_chain = np.array([[i, 0] for i in range(6)] + 
                       [[i + 4, 1] for i in range(6)])
    
    for method in ['single', 'complete']:
        agg = AgglomerativeClusteringNumPy(n_clusters=2, linkage=method)
        labels = agg.fit_predict(X_chain)
        cluster_0 = np.where(labels == 0)[0]
        cluster_1 = np.where(labels == 1)[0]
        print(f"{method:10s}: 簇0={cluster_0.tolist()}, 簇1={cluster_1.tolist()}")
    
    print("\n说明: Single linkage倾向于连接链状结构")
    
    # 示例3：大规模数据
    print("\n【示例3】大规模数据聚类 (1000样本)")
    print("-" * 40)
    
    X_large, _ = make_blobs(n_samples=1000, centers=5, 
                            cluster_std=0.6, random_state=42)
    
    import time
    start = time.time()
    agg = AgglomerativeClusteringNumPy(n_clusters=5, linkage='ward')
    labels = agg.fit_predict(X_large)
    elapsed = time.time() - start
    
    print(f"Ward法聚类 1000 个样本耗时: {elapsed:.3f}秒")
    print(f"每个簇的大小: {[np.sum(labels == i) for i in range(5)]}")
    
    # 示例4：可视化比较
    print("\n【示例4】链接方法可视化比较")
    print("-" * 40)
    
    X_viz, _ = make_blobs(n_samples=150, centers=3, 
                          cluster_std=0.6, random_state=42)
    compare_linkage_methods(X_viz, n_clusters=3)
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == '__main__':
    demo()
