"""
DBSCAN 密度聚类 NumPy 实现
第十四章：层次聚类与DBSCAN

包含：
- DBSCAN核心算法
- 核心点、边界点、噪声点分类
- K-距离图（参数选择）
- 参数敏感性分析
- 可视化工具

作者: ML教材写作项目
日期: 2026-03-30
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Set, Tuple, Optional, Dict
from collections import deque
from enum import Enum


class PointType(Enum):
    """点类型枚举"""
    UNVISITED = -2  # 未访问
    NOISE = -1      # 噪声点
    CORE = 1        # 核心点
    BORDER = 2      # 边界点


class DBSCANNumPy:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    密度基于空间聚类算法 - NumPy优化实现
    
    核心思想：通过密度连接来发现任意形状的簇，并能识别噪声点
    
    Parameters
    ----------
    eps : float
        ε-邻域半径，决定"邻居"的范围
    min_pts : int
        成为核心点所需的最小邻居数（包括自己）
    metric : str
        距离度量方式 ('euclidean', 'manhattan')
    
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        聚类标签，-1表示噪声
    core_sample_indices_ : ndarray
        核心点的索引
    n_clusters_ : int
        发现的簇数量
    point_types_ : ndarray
        每个点的类型（核心点、边界点、噪声点）
    
    Examples
    --------
    >>> from sklearn.datasets import make_moons
    >>> X, _ = make_moons(n_samples=300, noise=0.05)
    >>> dbscan = DBSCANNumPy(eps=0.2, min_pts=5)
    >>> labels = dbscan.fit_predict(X)
    >>> print(f"发现 {dbscan.n_clusters_} 个簇")
    """
    
    def __init__(
        self,
        eps: float = 0.5,
        min_pts: int = 5,
        metric: str = 'euclidean'
    ):
        self.eps = eps
        self.min_pts = min_pts
        self.metric = metric
        
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = 0
        self.point_types_ = None
        
        self._data = None
        self._n_samples = 0
        self._distance_matrix = None
        
    def _compute_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """预计算距离矩阵以加速邻居查询"""
        if self.metric == 'euclidean':
            sq_norms = np.sum(X ** 2, axis=1).reshape(-1, 1)
            dist_matrix = sq_norms + sq_norms.T - 2 * np.dot(X, X.T)
            dist_matrix = np.maximum(dist_matrix, 0)
            return np.sqrt(dist_matrix)
        elif self.metric == 'manhattan':
            n = X.shape[0]
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                dist_matrix[i] = np.sum(np.abs(X - X[i]), axis=1)
            return dist_matrix
        else:
            raise ValueError(f"不支持的度量方式: {self.metric}")
    
    def _get_neighbors(self, point_idx: int) -> List[int]:
        """
        获取指定点的ε-邻域内的所有点
        
        使用预计算的距离矩阵，时间复杂度O(1)
        """
        return list(np.where(self._distance_matrix[point_idx] <= self.eps)[0])
    
    def _expand_cluster(
        self,
        core_idx: int,
        neighbors: List[int],
        cluster_id: int
    ) -> None:
        """
        扩展簇 - 从核心点开始，递归地将密度可达的点加入簇
        
        使用BFS（广度优先搜索）进行高效扩展
        """
        # 使用队列进行BFS
        queue = deque(neighbors)
        self.labels_[core_idx] = cluster_id
        
        # 标记核心点
        self.point_types_[core_idx] = PointType.CORE.value
        
        # 标记邻居为边界点（暂时）
        for n_idx in neighbors:
            if self.point_types_[n_idx] == PointType.UNVISITED.value:
                self.point_types_[n_idx] = PointType.BORDER.value
        
        processed = {core_idx}
        
        while queue:
            point_idx = queue.popleft()
            
            if point_idx in processed:
                continue
            processed.add(point_idx)
            
            # 将点加入当前簇
            if self.labels_[point_idx] == PointType.NOISE.value:
                # 噪声点改判为边界点
                self.labels_[point_idx] = cluster_id
                self.point_types_[point_idx] = PointType.BORDER.value
            elif self.labels_[point_idx] == PointType.UNVISITED.value:
                self.labels_[point_idx] = cluster_id
            
            # 获取该点的邻居
            point_neighbors = self._get_neighbors(point_idx)
            
            # 如果该点也是核心点，继续扩展
            if len(point_neighbors) >= self.min_pts:
                self.point_types_[point_idx] = PointType.CORE.value
                
                for neighbor_idx in point_neighbors:
                    if neighbor_idx not in processed:
                        queue.append(neighbor_idx)
                        
                        if self.labels_[neighbor_idx] == PointType.UNVISITED.value:
                            self.labels_[neighbor_idx] = cluster_id
                            self.point_types_[neighbor_idx] = PointType.BORDER.value
                        elif self.labels_[neighbor_idx] == PointType.NOISE.value:
                            self.labels_[neighbor_idx] = cluster_id
                            self.point_types_[neighbor_idx] = PointType.BORDER.value
    
    def fit(self, X: np.ndarray) -> 'DBSCANNumPy':
        """
        执行DBSCAN聚类
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            训练数据
        
        Returns
        -------
        self
        """
        self._data = np.array(X)
        self._n_samples = len(X)
        
        # 预计算距离矩阵
        print(f"[DBSCAN] 计算距离矩阵: {self._n_samples} x {self._n_samples}")
        self._distance_matrix = self._compute_distance_matrix(self._data)
        
        # 初始化
        self.labels_ = np.full(self._n_samples, PointType.UNVISITED.value)
        self.point_types_ = np.full(self._n_samples, PointType.UNVISITED.value)
        self.core_sample_indices_ = []
        
        cluster_id = 0
        
        print(f"[DBSCAN] 开始聚类: eps={self.eps}, min_pts={self.min_pts}")
        
        for i in range(self._n_samples):
            # 跳过已处理的点
            if self.labels_[i] != PointType.UNVISITED.value:
                continue
            
            # 获取邻居
            neighbors = self._get_neighbors(i)
            
            # 检查是否为核心点
            if len(neighbors) >= self.min_pts:
                # 发现新簇
                self.core_sample_indices_.append(i)
                self._expand_cluster(i, neighbors, cluster_id)
                cluster_id += 1
            else:
                # 标记为噪声（暂时）
                self.labels_[i] = PointType.NOISE.value
                self.point_types_[i] = PointType.NOISE.value
        
        self.n_clusters_ = cluster_id
        self.core_sample_indices_ = np.array(self.core_sample_indices_)
        
        n_noise = np.sum(self.labels_ == PointType.NOISE.value)
        print(f"[DBSCAN] 聚类完成: 发现 {cluster_id} 个簇, "
              f"{len(self.core_sample_indices_)} 个核心点, "
              f"{n_noise} 个噪声点")
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """拟合数据并返回聚类标签"""
        self.fit(X)
        return self.labels_
    
    def get_point_classification(self) -> Dict[int, str]:
        """
        获取每个点的分类（核心点/边界点/噪声点）
        
        Returns
        -------
        dict : {point_idx: 'core'/'border'/'noise'}
        """
        classification = {}
        type_names = {
            PointType.CORE.value: 'core',
            PointType.BORDER.value: 'border',
            PointType.NOISE.value: 'noise',
            PointType.UNVISITED.value: 'unvisited'
        }
        
        for i in range(self._n_samples):
            classification[i] = type_names.get(self.point_types_[i], 'unknown')
        
        return classification


class DBSCANVisualizer:
    """DBSCAN结果可视化工具"""
    
    @staticmethod
    def plot_clusters(
        X: np.ndarray,
        labels: np.ndarray,
        title: str = "DBSCAN Clustering",
        core_indices: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        绘制聚类结果
        
        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            二维数据
        labels : array-like
            聚类标签
        title : str
            图表标题
        core_indices : array-like, optional
            核心点索引
        figsize : tuple
            图像大小
        """
        plt.figure(figsize=figsize)
        
        # 噪声点用黑色表示
        noise_mask = labels == -1
        if np.any(noise_mask):
            plt.scatter(
                X[noise_mask, 0], X[noise_mask, 1],
                c='black', marker='x', s=50, 
                label=f'Noise ({np.sum(noise_mask)})', alpha=0.6
            )
        
        # 绘制各个簇
        unique_labels = set(labels)
        unique_labels.discard(-1)
        
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))
        
        for label, color in zip(sorted(unique_labels), colors):
            mask = labels == label
            plt.scatter(
                X[mask, 0], X[mask, 1],
                c=[color], marker='o', s=50, alpha=0.6,
                label=f'Cluster {label} ({np.sum(mask)})'
            )
            
            # 高亮核心点
            if core_indices is not None:
                core_mask = np.isin(np.where(mask)[0], core_indices)
                core_points = np.where(mask)[0][core_mask]
                if len(core_points) > 0:
                    plt.scatter(
                        X[core_points, 0], X[core_points, 1],
                        c=[color], marker='*', s=200,
                        edgecolors='black', linewidths=1.5
                    )
        
        plt.title(title, fontsize=14)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    @staticmethod
    def plot_k_distance_graph(
        X: np.ndarray,
        k: int = 4,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        绘制K-距离图，帮助选择eps参数
        
        Parameters
        ----------
        X : array-like
            数据
        k : int
            K值（通常等于min_pts - 1）
        figsize : tuple
            图像大小
        """
        n = len(X)
        
        # 计算每个点的K-距离
        sq_norms = np.sum(X ** 2, axis=1).reshape(-1, 1)
        dist_matrix = sq_norms + sq_norms.T - 2 * np.dot(X, X.T)
        dist_matrix = np.sqrt(np.maximum(dist_matrix, 0))
        np.fill_diagonal(dist_matrix, np.inf)  # 排除自身
        
        # 获取第k小的距离
        k_distances = np.partition(dist_matrix, k, axis=1)[:, k]
        k_distances = np.sort(k_distances)[::-1]  # 降序排列
        
        # 绘图
        plt.figure(figsize=figsize)
        plt.plot(range(1, n + 1), k_distances, 'b-', linewidth=2)
        plt.xlabel('Points (sorted by distance)', fontsize=12)
        plt.ylabel(f'{k}-th Nearest Neighbor Distance', fontsize=12)
        plt.title(f'K-Distance Graph (k={k}) - For Choosing eps', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 尝试找到肘部
        if len(k_distances) > 10:
            # 简单的肘部检测：变化率最大的点
            diffs = np.diff(k_distances)
            elbow_candidates = np.where(diffs > np.std(diffs))[0]
            if len(elbow_candidates) > 0:
                elbow_idx = elbow_candidates[len(elbow_candidates) // 3]
                suggested_eps = k_distances[elbow_idx]
                plt.axhline(
                    y=suggested_eps, color='r', linestyle='--',
                    label=f'Suggested eps ≈ {suggested_eps:.2f}'
                )
                plt.legend()
        
        plt.tight_layout()
        return plt.gcf(), k_distances


class ParameterAnalyzer:
    """DBSCAN参数分析器"""
    
    @staticmethod
    def analyze_eps_sensitivity(
        X: np.ndarray,
        eps_range: np.ndarray,
        min_pts: int = 5
    ) -> Dict:
        """
        分析不同eps值对聚类结果的影响
        
        Parameters
        ----------
        X : array-like
            数据
        eps_range : array-like
            要测试的eps值范围
        min_pts : int
            最小点数
        
        Returns
        -------
        dict : 包含每个eps值对应的聚类统计信息
        """
        results = {
            'eps': [],
            'n_clusters': [],
            'n_noise': [],
            'noise_ratio': [],
            'n_core': []
        }
        
        for eps in eps_range:
            dbscan = DBSCANNumPy(eps=eps, min_pts=min_pts)
            labels = dbscan.fit_predict(X)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            n_core = len(dbscan.core_sample_indices_)
            
            results['eps'].append(eps)
            results['n_clusters'].append(n_clusters)
            results['n_noise'].append(n_noise)
            results['noise_ratio'].append(n_noise / len(X))
            results['n_core'].append(n_core)
        
        return results
    
    @staticmethod
    def plot_parameter_analysis(
        X: np.ndarray,
        eps_range: Optional[np.ndarray] = None,
        min_pts: int = 5,
        figsize: Tuple[int, int] = (15, 4)
    ):
        """可视化参数分析结果"""
        if eps_range is None:
            eps_range = np.linspace(0.1, 2.0, 20)
        
        results = ParameterAnalyzer.analyze_eps_sensitivity(
            X, eps_range, min_pts
        )
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
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
        
        # 核心点数量 vs eps
        axes[3].plot(results['eps'], results['n_core'], 'mo-', linewidth=2)
        axes[3].set_xlabel('eps')
        axes[3].set_ylabel('Number of Core Points')
        axes[3].set_title('Core Points vs eps')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def demo_basic():
    """基础演示"""
    from sklearn.datasets import make_moons, make_blobs
    
    print("=" * 60)
    print("DBSCAN NumPy 实现演示")
    print("=" * 60)
    
    # 示例1：月牙形数据
    print("\n【示例1】月牙形数据聚类")
    print("-" * 40)
    
    X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    dbscan = DBSCANNumPy(eps=0.2, min_pts=5)
    labels = dbscan.fit_predict(X_moons)
    
    print(f"发现 {dbscan.n_clusters_} 个簇")
    print(f"核心点数量: {len(dbscan.core_sample_indices_)}")
    print(f"噪声点数量: {np.sum(labels == -1)}")
    
    # 可视化
    DBSCANVisualizer.plot_clusters(
        X_moons, labels, 
        "DBSCAN on Moon-Shaped Data",
        dbscan.core_sample_indices_
    )
    plt.savefig('dbscan_moons.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 示例2：带噪声的数据
    print("\n【示例2】带噪声的数据聚类")
    print("-" * 40)
    
    X_blobs, _ = make_blobs(n_samples=300, centers=3, 
                            cluster_std=0.6, random_state=42)
    
    # 添加噪声
    np.random.seed(42)
    noise = np.random.uniform(-3, 3, (30, 2))
    X_with_noise = np.vstack([X_blobs, noise])
    
    dbscan = DBSCANNumPy(eps=0.8, min_pts=5)
    labels = dbscan.fit_predict(X_with_noise)
    
    print(f"添加噪声点: 30个")
    print(f"DBSCAN识别噪声点: {np.sum(labels == -1)}个")
    
    DBSCANVisualizer.plot_clusters(
        X_with_noise, labels,
        "DBSCAN: Automatic Noise Detection",
        dbscan.core_sample_indices_
    )
    plt.savefig('dbscan_noise.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_parameter_selection():
    """参数选择演示"""
    from sklearn.datasets import make_moons
    
    print("\n【示例3】参数选择 - K-距离图")
    print("-" * 40)
    
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    # 绘制K-距离图
    DBSCANVisualizer.plot_k_distance_graph(X, k=4)
    plt.savefig('k_distance_graph.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("观察K-距离图的'肘部'，选择合适的eps值")
    
    # 参数敏感性分析
    print("\n【示例4】参数敏感性分析")
    print("-" * 40)
    
    eps_range = np.linspace(0.1, 0.5, 20)
    ParameterAnalyzer.plot_parameter_analysis(X, eps_range)
    plt.savefig('parameter_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_comparison():
    """算法对比演示"""
    from sklearn.datasets import make_moons
    
    print("\n【示例5】DBSCAN vs K-Means 对比")
    print("-" * 40)
    
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    # DBSCAN
    dbscan = DBSCANNumPy(eps=0.2, min_pts=5)
    labels_dbscan = dbscan.fit_predict(X)
    
    # K-Means（使用sklearn）
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X)
    
    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='viridis', s=50)
    axes[0].set_title('DBSCAN (发现任意形状簇)')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    axes[1].scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis', s=50)
    axes[1].scatter(kmeans.cluster_centers_[:, 0], 
                   kmeans.cluster_centers_[:, 1],
                   c='red', marker='X', s=200, label='Centroids')
    axes[1].set_title('K-Means (倾向于球形簇)')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('dbscan_vs_kmeans.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("DBSCAN能正确识别月牙形状，而K-Means倾向于直线分割")


def demo():
    """运行所有演示"""
    demo_basic()
    demo_parameter_selection()
    demo_comparison()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)


if __name__ == '__main__':
    demo()
