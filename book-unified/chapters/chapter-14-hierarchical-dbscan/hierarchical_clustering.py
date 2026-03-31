"""
hierarchical_clustering.py
第十四章：层次聚类实现
《机器学习与深度学习：从小学生到大师》

本文件包含：
- AGNES算法实现（单链接、全链接、平均链接）
- 树状图生成
- 距离矩阵计算
- 可视化工具

作者：机器学习小助手
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from collections import deque


class AGNES:
    """
    AGNES (Agglomerative Nesting) 层次聚类算法
    
    自底向上的凝聚层次聚类，支持多种链接方法
    """
    
    LINKAGE_METHODS = ['single', 'complete', 'average', 'ward']
    
    def __init__(self, n_clusters=2, linkage='single'):
        """
        初始化AGNES聚类器
        
        参数:
            n_clusters: 最终聚类数
            linkage: 链接方法 ('single', 'complete', 'average', 'ward')
        """
        if linkage not in self.LINKAGE_METHODS:
            raise ValueError(f"链接方法必须是以下之一: {self.LINKAGE_METHODS}")
        
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.linkage_matrix_ = None
        self.n_samples_ = None
        
    def fit(self, X):
        """
        训练模型
        
        参数:
            X: 输入数据，形状 (n_samples, n_features)
        """
        self.n_samples_ = X.shape[0]
        
        # 使用scipy的linkage函数计算链接矩阵
        # 这会返回一个矩阵，描述每一步合并的过程
        self.linkage_matrix_ = linkage(X, method=self.linkage)
        
        # 根据linkage_matrix生成聚类标签
        self.labels_ = fcluster(self.linkage_matrix_, 
                                t=self.n_clusters, 
                                criterion='maxclust')
        
        # 将标签转换为从0开始的索引
        self.labels_ = self.labels_ - 1
        
        return self
    
    def fit_predict(self, X):
        """训练并返回预测标签"""
        self.fit(X)
        return self.labels_
    
    def plot_dendrogram(self, X=None, truncate_mode='level', 
                        p=5, figsize=(12, 6), title=None):
        """
        绘制树状图
        
        参数:
            X: 输入数据（如果fit未调用则需要）
            truncate_mode: 截断模式 ('level' 或 'lastp')
            p: 截断参数
            figsize: 图像大小
            title: 图表标题
        """
        if self.linkage_matrix_ is None:
            if X is None:
                raise ValueError("请先调用fit()或传入数据X")
            self.fit(X)
        
        plt.figure(figsize=figsize)
        
        dendrogram(
            self.linkage_matrix_,
            truncate_mode=truncate_mode,
            p=p,
            show_leaf_counts=True,
            leaf_font_size=10,
            show_contracted=True,  # 显示被截断的簇的大小
        )
        
        plt.title(title or f'AGNES 树状图 ({self.linkage} 链接)')
        plt.xlabel('样本索引或 (簇内样本数)')
        plt.ylabel('距离')
        plt.tight_layout()
        plt.show()
    
    def get_merge_history(self):
        """
        获取合并历史记录
        
        返回:
            列表，每个元素是 (簇1, 簇2, 距离, 新簇样本数)
        """
        if self.linkage_matrix_ is None:
            raise ValueError("请先调用fit()")
        
        history = []
        for i, row in enumerate(self.linkage_matrix_):
            c1, c2, dist, n_samples = row
            history.append({
                'step': i + 1,
                'cluster_1': int(c1),
                'cluster_2': int(c2),
                'distance': dist,
                'n_samples': int(n_samples)
            })
        return history


class ManualAGNES:
    """
    手动实现的AGNES算法（用于教学目的）
    
    展示了算法内部的每一步是如何工作的
    """
    
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.distance_matrix_ = None
        self.merge_history_ = []
        
    def _compute_distance_matrix(self, X):
        """计算距离矩阵"""
        n = X.shape[0]
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
                
        return dist_matrix
    
    def _linkage_distance(self, cluster_i, cluster_j, dist_matrix):
        """
        计算两个簇之间的距离
        
        参数:
            cluster_i: 簇i中的点索引列表
            cluster_j: 簇j中的点索引列表
            dist_matrix: 距离矩阵
        """
        distances = []
        for i in cluster_i:
            for j in cluster_j:
                distances.append(dist_matrix[i, j])
        
        if self.linkage == 'single':
            return min(distances)
        elif self.linkage == 'complete':
            return max(distances)
        elif self.linkage == 'average':
            return sum(distances) / len(distances)
        else:
            raise ValueError(f"不支持的链接方法: {self.linkage}")
    
    def fit(self, X):
        """
        手动实现AGNES算法
        
        算法步骤：
        1. 每个点作为一个簇
        2. 计算所有簇之间的距离
        3. 合并距离最近的两个簇
        4. 重复直到只剩n_clusters个簇
        """
        n_samples = X.shape[0]
        
        # 初始化：每个点是一个簇
        clusters = [{i} for i in range(n_samples)]
        self.distance_matrix_ = self._compute_distance_matrix(X)
        
        step = 0
        while len(clusters) > self.n_clusters:
            step += 1
            min_dist = float('inf')
            to_merge = (0, 1)
            
            # 找出距离最近的两个簇
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._linkage_distance(
                        list(clusters[i]), 
                        list(clusters[j]), 
                        self.distance_matrix_
                    )
                    if dist < min_dist:
                        min_dist = dist
                        to_merge = (i, j)
            
            # 合并这两个簇
            i, j = to_merge
            self.merge_history_.append({
                'step': step,
                'clusters_merged': (i, j),
                'distance': min_dist,
                'cluster_sizes': (len(clusters[i]), len(clusters[j]))
            })
            
            new_cluster = clusters[i] | clusters[j]
            clusters = [c for idx, c in enumerate(clusters) 
                       if idx not in (i, j)]
            clusters.append(new_cluster)
        
        # 生成标签
        self.labels_ = np.zeros(n_samples, dtype=int)
        for label, cluster in enumerate(clusters):
            for idx in cluster:
                self.labels_[idx] = label
                
        return self
    
    def fit_predict(self, X):
        """训练并返回预测标签"""
        self.fit(X)
        return self.labels_


def compute_distance_matrix_fast(X, metric='euclidean'):
    """
    快速计算距离矩阵
    
    参数:
        X: 输入数据 (n_samples, n_features)
        metric: 距离度量 ('euclidean', 'manhattan', 'cosine')
    
    返回:
        距离矩阵 (n_samples, n_samples)
    """
    if metric == 'euclidean':
        # 使用向量化计算欧氏距离
        # dist(i,j) = sqrt(||x_i||^2 + ||x_j||^2 - 2*x_i·x_j)
        sq_norms = np.sum(X ** 2, axis=1).reshape(-1, 1)
        dist_matrix = sq_norms + sq_norms.T - 2 * np.dot(X, X.T)
        # 处理数值误差
        dist_matrix = np.maximum(dist_matrix, 0)
        return np.sqrt(dist_matrix)
    else:
        # 使用scipy的pdist
        return squareform(pdist(X, metric=metric))


def lance_williams_update(d_ik, d_jk, d_ij, n_i, n_j, n_k, linkage='single'):
    """
    Lance-Williams递推公式
    
    用于计算合并后的新簇与其他簇之间的距离
    
    参数:
        d_ik: 簇i到簇k的距离
        d_jk: 簇j到簇k的距离
        d_ij: 簇i到簇j的距离
        n_i, n_j, n_k: 各簇的样本数
        linkage: 链接方法
    
    返回:
        新簇(i∪j)到簇k的距离
    """
    if linkage == 'single':
        alpha_i, alpha_j = 0.5, 0.5
        beta, gamma = 0, -0.5
    elif linkage == 'complete':
        alpha_i, alpha_j = 0.5, 0.5
        beta, gamma = 0, 0.5
    elif linkage == 'average':
        alpha_i = n_i / (n_i + n_j)
        alpha_j = n_j / (n_i + n_j)
        beta, gamma = 0, 0
    elif linkage == 'ward':
        n_total = n_i + n_j + n_k
        alpha_i = (n_k + n_i) / n_total
        alpha_j = (n_k + n_j) / n_total
        beta = -n_k / n_total
        gamma = 0
    else:
        raise ValueError(f"不支持的链接方法: {linkage}")
    
    return (alpha_i * d_ik + alpha_j * d_jk + 
            beta * d_ij + gamma * abs(d_ik - d_jk))


def demo_hierarchical_clustering():
    """
    层次聚类演示
    
    展示不同链接方法的效果对比
    """
    from sklearn.datasets import make_blobs, make_moons
    
    # 生成三组不同类型的数据
    np.random.seed(42)
    
    # 数据1：三个高斯分布的簇
    X1, y1 = make_blobs(n_samples=150, centers=3, cluster_std=0.6, 
                        random_state=42)
    
    # 数据2：两个半月形（测试单链接的优势）
    X2, y2 = make_moons(n_samples=200, noise=0.05, random_state=42)
    
    # 数据3：不同密度的簇
    X3_a, _ = make_blobs(n_samples=100, centers=[[0, 0]], 
                         cluster_std=0.5, random_state=42)
    X3_b, _ = make_blobs(n_samples=50, centers=[[3, 3]], 
                         cluster_std=1.5, random_state=42)
    X3 = np.vstack([X3_a, X3_b])
    
    datasets = [
        ('三个高斯簇', X1, 3),
        ('两个半月形', X2, 2),
        ('不同密度簇', X3, 2)
    ]
    
    linkages = ['single', 'complete', 'average', 'ward']
    
    fig, axes = plt.subplots(len(datasets), len(linkages) + 1, 
                             figsize=(20, 12))
    
    for row, (name, X, true_k) in enumerate(datasets):
        # 第一列：原始数据
        axes[row, 0].scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6)
        axes[row, 0].set_title(f'{name}\n(原始数据)')
        axes[row, 0].set_xlabel('特征 1')
        axes[row, 0].set_ylabel('特征 2')
        
        # 其他列：不同链接方法
        for col, linkage_method in enumerate(linkages, 1):
            model = AGNES(n_clusters=true_k, linkage=linkage_method)
            labels = model.fit_predict(X)
            
            scatter = axes[row, col].scatter(X[:, 0], X[:, 1], 
                                              c=labels, cmap='viridis',
                                              alpha=0.6)
            axes[row, col].set_title(f'{linkage_method}')
            axes[row, col].set_xlabel('特征 1')
            axes[row, col].set_ylabel('特征 2')
    
    plt.tight_layout()
    plt.suptitle('AGNES层次聚类：不同链接方法对比', y=1.02, fontsize=14)
    plt.show()
    
    print("=" * 60)
    print("层次聚类演示完成！")
    print("观察不同链接方法在各种数据分布上的表现差异。")
    print("=" * 60)


def demo_dendrogram():
    """树状图可视化演示"""
    from sklearn.datasets import make_blobs
    
    np.random.seed(42)
    X, y = make_blobs(n_samples=20, centers=3, cluster_std=0.6, 
                      random_state=42)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    linkages = ['single', 'complete', 'average', 'ward']
    
    for idx, linkage_method in enumerate(linkages):
        model = AGNES(linkage=linkage_method)
        model.fit(X)
        
        ax = axes[idx]
        dendrogram(model.linkage_matrix_, ax=ax)
        ax.set_title(f'{linkage_method.capitalize()} 链接')
        ax.set_xlabel('样本索引')
        ax.set_ylabel('距离')
    
    plt.tight_layout()
    plt.suptitle('不同链接方法的树状图对比', y=1.02, fontsize=14)
    plt.show()


def compare_manual_vs_scipy():
    """对比手动实现和scipy实现的结果"""
    from sklearn.datasets import make_blobs
    
    np.random.seed(42)
    X, _ = make_blobs(n_samples=30, centers=3, cluster_std=0.8, 
                      random_state=42)
    
    # Scipy实现
    scipy_model = AGNES(n_clusters=3, linkage='single')
    scipy_labels = scipy_model.fit_predict(X)
    
    # 手动实现
    manual_model = ManualAGNES(n_clusters=3, linkage='single')
    manual_labels = manual_model.fit_predict(X)
    
    print("=" * 60)
    print("Scipy AGNES 标签:", scipy_labels)
    print("手动 AGNES 标签:", manual_labels)
    print("=" * 60)
    print("合并历史记录:")
    for record in manual_model.merge_history_[:5]:
        print(f"  步骤 {record['step']}: 合并簇 {record['clusters_merged']}, "
              f"距离={record['distance']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    print("=" * 60)
    print("层次聚类 (Hierarchical Clustering) 演示")
    print("=" * 60)
    
    # 运行演示
    demo_hierarchical_clustering()
    demo_dendrogram()
    compare_manual_vs_scipy()
    
    print("\n所有演示完成！")
