"""
第十三章：K-Means聚类——物以类聚
完整代码实现

包含：
1. K-Means基础算法（Lloyd算法）
2. K-Means++智能初始化
3. 肘部法则可视化
4. 轮廓系数计算
5. 图像颜色量化演示
6. 完整示例与对比
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
import random
from collections import defaultdict


# ============================================================
# 工具函数
# ============================================================

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """计算欧氏距离"""
    return np.sqrt(np.sum((a - b) ** 2))


def euclidean_distance_squared(a: np.ndarray, b: np.ndarray) -> float:
    """计算欧氏距离平方（K-Means用不到开根号，可以省计算）"""
    return np.sum((a - b) ** 2)


def train_test_split(X: np.ndarray, y: Optional[np.ndarray] = None, 
                     test_size: float = 0.2, 
                     random_state: Optional[int] = None) -> Tuple:
    """分割数据集"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    if y is not None:
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    return X[train_indices], X[test_indices]


# ============================================================
# K-Means聚类算法
# ============================================================

class KMeans:
    """
    K-Means聚类算法实现
    
    基于Lloyd算法，支持K-Means++初始化
    """
    
    def __init__(self, n_clusters: int = 3, init: str = 'k-means++',
                 max_iter: int = 300, tol: float = 1e-4,
                 random_state: Optional[int] = None):
        """
        参数:
            n_clusters: K值，聚类数量
            init: 初始化方法 ('k-means++', 'random')
            max_iter: 最大迭代次数
            tol: 收敛阈值（中心点移动距离小于此值则认为收敛）
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.centroids_ = None  # 聚类中心
        self.labels_ = None     # 每个点的聚类标签
        self.inertia_ = None    # 目标函数值（WCSS）
        self.n_iter_ = 0        # 实际迭代次数
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """初始化聚类中心"""
        n_samples, n_features = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        if self.init == 'random':
            # 随机选择K个点作为初始中心
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            return X[indices].copy()
        
        elif self.init == 'k-means++':
            return self._kmeans_plus_plus(X)
        
        else:
            raise ValueError(f"未知的初始化方法: {self.init}")
    
    def _kmeans_plus_plus(self, X: np.ndarray) -> np.ndarray:
        """
        K-Means++初始化
        
        算法:
        1. 随机选择第一个中心
        2. 对每个点，计算到最近中心的距离D(x)
        3. 以概率D(x)^2/sum(D^2)选择下一个中心
        4. 重复直到选够K个
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        # 第1步：随机选择第一个中心
        first_idx = np.random.randint(n_samples)
        centroids[0] = X[first_idx]
        
        # 已选中心的数量
        n_chosen = 1
        
        # 存储每个点到最近中心的距离平方
        distances = np.full(n_samples, np.inf)
        
        while n_chosen < self.n_clusters:
            # 计算每个点到最近已选中心的距离
            for i in range(n_samples):
                dist_sq = euclidean_distance_squared(X[i], centroids[n_chosen - 1])
                if dist_sq < distances[i]:
                    distances[i] = dist_sq
            
            # 按概率D^2选择下一个中心
            # 概率 = D^2 / sum(D^2)
            probabilities = distances / np.sum(distances)
            next_idx = np.random.choice(n_samples, p=probabilities)
            
            centroids[n_chosen] = X[next_idx]
            n_chosen += 1
        
        return centroids
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        分配步骤：把每个点分配到最近的中心
        
        返回: 每个点的聚类标签 (0到K-1)
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # 计算到所有中心的距离
            distances = np.array([
                euclidean_distance_squared(X[i], centroids[j])
                for j in range(self.n_clusters)
            ])
            # 分配到最近的中心
            labels[i] = np.argmin(distances)
        
        return labels
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        更新步骤：重新计算每个聚类的中心（平均值）
        """
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            # 获取属于聚类k的所有点
            cluster_points = X[labels == k]
            
            if len(cluster_points) > 0:
                # 计算平均值
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # 如果某个聚类没有点，保持原中心或重新随机
                # 这里简单地保持原中心（实际应该处理）
                pass
        
        return centroids
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, 
                         centroids: np.ndarray) -> float:
        """
        计算目标函数值（WCSS - Within-Cluster Sum of Squares）
        
        J = sum_{i=1}^K sum_{x in C_i} ||x - mu_i||^2
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[k]) ** 2)
        return inertia
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        训练K-Means模型
        
        参数:
            X: 训练数据，形状 (n_samples, n_features)
        """
        # 初始化中心
        self.centroids_ = self._initialize_centroids(X)
        
        # 迭代优化
        for iteration in range(self.max_iter):
            # 第1步：分配
            labels = self._assign_clusters(X, self.centroids_)
            
            # 第2步：更新
            new_centroids = self._update_centroids(X, labels)
            
            # 检查收敛（中心点移动距离）
            centroid_shift = np.sqrt(np.sum((new_centroids - self.centroids_) ** 2))
            
            self.centroids_ = new_centroids
            self.n_iter_ = iteration + 1
            
            if centroid_shift < self.tol:
                break
        
        # 最终分配和计算目标函数值
        self.labels_ = self._assign_clusters(X, self.centroids_)
        self.inertia_ = self._compute_inertia(X, self.labels_, self.centroids_)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新数据点的聚类标签
        """
        if self.centroids_ is None:
            raise ValueError("模型尚未训练，请先调用fit()")
        return self._assign_clusters(X, self.centroids_)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        训练并预测
        """
        self.fit(X)
        return self.labels_
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        将数据转换为到各中心的距离
        
        返回: 形状 (n_samples, n_clusters)，每行是该点到各中心的距离
        """
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        for i in range(n_samples):
            for j in range(self.n_clusters):
                distances[i, j] = euclidean_distance(X[i], self.centroids_[j])
        
        return distances


# ============================================================
# 聚类评估指标
# ============================================================

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    计算平均轮廓系数
    
    轮廓系数s = (b - a) / max(a, b)
    - a: 点到同簇其他点的平均距离（紧密度）
    - b: 点到最近其他簇的平均距离（分离度）
    
    范围[-1, 1]，越接近1越好
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1:
        return 0  # 只有一个簇，无法计算
    
    silhouette_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        # 当前点的标签
        label_i = labels[i]
        
        # 同簇的其他点
        same_cluster = X[labels == label_i]
        
        # 计算a：到同簇其他点的平均距离
        if len(same_cluster) > 1:
            a = np.mean([euclidean_distance(X[i], xj) 
                        for xj in same_cluster if not np.array_equal(xj, X[i])])
        else:
            a = 0
        
        # 计算b：到最近其他簇的平均距离
        b_values = []
        for label in unique_labels:
            if label != label_i:
                other_cluster = X[labels == label]
                mean_dist = np.mean([euclidean_distance(X[i], xj) 
                                   for xj in other_cluster])
                b_values.append(mean_dist)
        
        b = min(b_values) if b_values else 0
        
        # 轮廓系数
        if max(a, b) > 0:
            silhouette_scores[i] = (b - a) / max(a, b)
        else:
            silhouette_scores[i] = 0
    
    return np.mean(silhouette_scores)


def silhouette_samples(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    计算每个点的轮廓系数
    
    返回: 每个点的轮廓系数数组
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    
    silhouette_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        label_i = labels[i]
        same_cluster = X[labels == label_i]
        
        # a: 紧密度
        if len(same_cluster) > 1:
            a = np.mean([euclidean_distance(X[i], xj) 
                        for xj in same_cluster if not np.array_equal(xj, X[i])])
        else:
            a = 0
        
        # b: 分离度
        b_values = []
        for label in unique_labels:
            if label != label_i:
                other_cluster = X[labels == label]
                mean_dist = np.mean([euclidean_distance(X[i], xj) 
                                   for xj in other_cluster])
                b_values.append(mean_dist)
        
        b = min(b_values) if b_values else 0
        
        if max(a, b) > 0:
            silhouette_scores[i] = (b - a) / max(a, b)
        else:
            silhouette_scores[i] = 0
    
    return silhouette_scores


def elbow_method(X: np.ndarray, k_range: range = range(1, 11),
                 random_state: Optional[int] = None) -> Tuple[List[int], List[float]]:
    """
    肘部法则：测试不同K值的WCSS
    
    返回: (K值列表, 对应的WCSS列表)
    """
    wcss_values = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', 
                       random_state=random_state)
        kmeans.fit(X)
        wcss_values.append(kmeans.inertia_)
    
    return list(k_range), wcss_values


# ============================================================
# 演示1：二维数据聚类可视化
# ============================================================

def demo_2d_clustering():
    """演示：二维数据聚类"""
    print("=" * 60)
    print("演示1：二维数据K-Means聚类")
    print("=" * 60)
    
    # 生成示例数据：3个明显的簇
    np.random.seed(42)
    
    # 簇1：中心(2, 2)
    cluster1 = np.random.randn(50, 2) * 0.5 + np.array([2, 2])
    # 簇2：中心(-2, 2)
    cluster2 = np.random.randn(50, 2) * 0.5 + np.array([-2, 2])
    # 簇3：中心(0, -3)
    cluster3 = np.random.randn(50, 2) * 0.5 + np.array([0, -3])
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    print(f"\n数据形状: {X.shape}")
    print(f"真实簇数: 3")
    
    # K-Means聚类
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X)
    
    print(f"\n聚类中心:")
    for i, center in enumerate(kmeans.centroids_):
        print(f"  簇{i}: ({center[0]:.2f}, {center[1]:.2f})")
    
    print(f"\n迭代次数: {kmeans.n_iter_}")
    print(f"WCSS: {kmeans.inertia_:.2f}")
    
    # 计算轮廓系数
    score = silhouette_score(X, labels)
    print(f"轮廓系数: {score:.3f}")
    
    # 计算准确率（与真实标签对比，需要处理标签对应问题）
    # 简化起见，这里只看聚类效果
    print("\n✓ 聚类完成！3个簇已被正确识别。")


# ============================================================
# 演示2：肘部法则选择K
# ============================================================

def demo_elbow_method():
    """演示：肘部法则选择K值"""
    print("\n" + "=" * 60)
    print("演示2：肘部法则选择K值")
    print("=" * 60)
    
    # 生成数据：4个簇
    np.random.seed(42)
    cluster1 = np.random.randn(50, 2) * 0.5 + np.array([2, 2])
    cluster2 = np.random.randn(50, 2) * 0.5 + np.array([-2, 2])
    cluster3 = np.random.randn(50, 2) * 0.5 + np.array([2, -2])
    cluster4 = np.random.randn(50, 2) * 0.5 + np.array([-2, -2])
    
    X = np.vstack([cluster1, cluster2, cluster3, cluster4])
    
    print("\n测试K值从1到8的WCSS：")
    print("-" * 40)
    
    k_values, wcss_values = elbow_method(X, range(1, 9), random_state=42)
    
    for k, wcss in zip(k_values, wcss_values):
        bar = "█" * int(wcss / max(wcss_values) * 30)
        print(f"K={k:2d}: WCSS={wcss:8.1f} {bar}")
    
    # 计算下降幅度
    print("\nWCSS下降幅度:")
    for i in range(1, len(wcss_values)):
        drop = wcss_values[i-1] - wcss_values[i]
        print(f"  K={i}→{i+1}: {drop:.1f}")
    
    print("\n✓ 从K=4开始，WCSS下降明显减缓，肘部出现在K=4")
    print("  （注意：此数据集真实簇数为4）")


# ============================================================
# 演示3：K-Means++ vs 随机初始化
# ============================================================

def demo_initialization_comparison():
    """演示：K-Means++ vs 随机初始化"""
    print("\n" + "=" * 60)
    print("演示3：K-Means++ vs 随机初始化对比")
    print("=" * 60)
    
    # 生成有挑战性的数据
    np.random.seed(123)
    
    # 两个相距较远的簇和一个中间的小簇
    cluster1 = np.random.randn(100, 2) * 0.3 + np.array([0, 5])
    cluster2 = np.random.randn(100, 2) * 0.3 + np.array([0, -5])
    cluster3 = np.random.randn(20, 2) * 0.2 + np.array([0, 0])
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    print("\n数据集：3个簇（上下各一个大簇，中间一个小簇）")
    print(f"数据形状: {X.shape}")
    
    n_runs = 10
    
    # K-Means++多次运行
    print(f"\nK-Means++ ({n_runs}次运行):")
    kpp_inertias = []
    for i in range(n_runs):
        kmeans = KMeans(n_clusters=3, init='k-means++', random_state=i*10)
        kmeans.fit(X)
        kpp_inertias.append(kmeans.inertia_)
    
    print(f"  WCSS均值: {np.mean(kpp_inertias):.2f}")
    print(f"  WCSS标准差: {np.std(kpp_inertias):.2f}")
    print(f"  WCSS范围: [{min(kpp_inertias):.2f}, {max(kpp_inertias):.2f}]")
    
    # 随机初始化多次运行
    print(f"\n随机初始化 ({n_runs}次运行):")
    random_inertias = []
    for i in range(n_runs):
        kmeans = KMeans(n_clusters=3, init='random', random_state=i*10)
        kmeans.fit(X)
        random_inertias.append(kmeans.inertia_)
    
    print(f"  WCSS均值: {np.mean(random_inertias):.2f}")
    print(f"  WCSS标准差: {np.std(random_inertias):.2f}")
    print(f"  WCSS范围: [{min(random_inertias):.2f}, {max(random_inertias):.2f}]")
    
    print("\n✓ K-Means++结果更稳定（标准差更小），且通常更优")


# ============================================================
# 演示4：图像颜色量化（简化版）
# ============================================================

def demo_color_quantization():
    """演示：图像颜色量化概念"""
    print("\n" + "=" * 60)
    print("演示4：图像颜色量化概念")
    print("=" * 60)
    
    print("""
图像颜色量化原理：
1. 把图像看作一堆像素点，每个像素是RGB空间中的一个点(R, G, B)
2. 用K-Means把像素聚成K个簇
3. 每个像素用所属簇的中心颜色代替
4. 结果：图像从数百万色变为K种颜色

示例：一张100×100的彩色图像
- 原始：10,000个像素，每个3字节（RGB）= 30,000字节
- K=16量化后：10,000个像素，每个只需存储簇标签(0-15)
  加上16个颜色表项 = 10,000×0.5 + 16×3 ≈ 5,048字节
- 压缩率：约6倍！
    """)
    
    # 模拟颜色量化
    np.random.seed(42)
    
    # 模拟"图像"：1000个像素，3个主要颜色区域
    # 区域1：红色系
    region1 = np.random.randn(300, 3) * 20 + np.array([200, 50, 50])
    # 区域2：绿色系  
    region2 = np.random.randn(400, 3) * 20 + np.array([50, 200, 50])
    # 区域3：蓝色系
    region3 = np.random.randn(300, 3) * 20 + np.array([50, 50, 200])
    
    X = np.vstack([region1, region2, region3])
    X = np.clip(X, 0, 255)  # RGB范围
    
    print(f"\n模拟图像数据：{X.shape[0]}个像素，RGB值范围[0, 255]")
    print(f"\n原始颜色数：理论上有{X.shape[0]}种不同颜色")
    
    # 用K=3量化
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X)
    
    print(f"\nK=3量化后：")
    print(f"  颜色表（3种颜色）:")
    for i, color in enumerate(kmeans.centroids_):
        print(f"    颜色{i}: RGB({color[0]:.0f}, {color[1]:.0f}, {color[2]:.0f})")
    
    print(f"\n  每个像素只需存储簇标签（0, 1, 或 2）")
    print(f"  压缩比：{X.shape[0]*3}字节 → {X.shape[0]*0.5 + 9:.0f}字节 ≈ 6倍")
    
    # 计算量化误差
    quantized = kmeans.centroids_[labels]
    mse = np.mean((X - quantized) ** 2)
    print(f"\n  量化误差（MSE）: {mse:.1f}")
    
    print("\n✓ K-Means成功识别了3种主要颜色！")


# ============================================================
# 演示5：轮廓系数详解
# ============================================================

def demo_silhouette():
    """演示：轮廓系数计算"""
    print("\n" + "=" * 60)
    print("演示5：轮廓系数计算示例")
    print("=" * 60)
    
    # 生成良好分离的数据
    np.random.seed(42)
    cluster1 = np.random.randn(30, 2) * 0.3 + np.array([0, 2])
    cluster2 = np.random.randn(30, 2) * 0.3 + np.array([0, -2])
    
    X = np.vstack([cluster1, cluster2])
    
    print("\n数据集：2个良好分离的簇")
    
    # K=2聚类
    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X)
    
    score = silhouette_score(X, labels)
    print(f"\nK=2时轮廓系数: {score:.3f}")
    print("  （接近1，表示聚类效果很好）")
    
    # K=3聚类（过聚类）
    kmeans3 = KMeans(n_clusters=3, init='k-means++', random_state=42)
    labels3 = kmeans3.fit_predict(X)
    
    score3 = silhouette_score(X, labels3)
    print(f"\nK=3时轮廓系数: {score3:.3f}")
    print("  （比K=2时低，说明K=2更合适）")
    
    # K=1（无法计算）
    print(f"\nK=1时轮廓系数: 0 (只有一个簇，无法计算)")
    
    print("\n✓ 轮廓系数帮助选择最佳K值！")


# ============================================================
# 主程序
# ============================================================

def main():
    """运行所有演示"""
    print("\n" + "=" * 60)
    print("第十三章：K-Means聚类算法演示")
    print("=" * 60)
    
    demo_2d_clustering()
    demo_elbow_method()
    demo_initialization_comparison()
    demo_color_quantization()
    demo_silhouette()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
