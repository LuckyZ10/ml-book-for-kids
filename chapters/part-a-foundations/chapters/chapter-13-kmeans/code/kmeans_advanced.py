#!/usr/bin/env python3
"""
K-Means高级应用 - 补充代码
Chapter 13: K-Means聚类
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans as SklearnKMeans
import cv2
from typing import Tuple, List


def image_compression_kmeans(image_path: str, n_colors: int = 16, 
                             save_path: str = None) -> np.ndarray:
    """
    使用K-Means进行图像压缩（颜色量化）
    
    参数:
    - image_path: 输入图像路径
    - n_colors: 量化后的颜色数
    - save_path: 保存路径
    
    返回:
    - 压缩后的图像
    """
    # 读取图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 重塑为像素列表
    pixels = img.reshape((-1, 3))
    pixels = np.float32(pixels)
    
    # K-Means聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, 
                                     cv2.KMEANS_RANDOM_CENTERS)
    
    # 转换回uint8
    centers = np.uint8(centers)
    
    # 用中心颜色替换每个像素
    compressed = centers[labels.flatten()]
    compressed = compressed.reshape(img.shape)
    
    # 保存
    if save_path:
        save_img = cv2.cvtColor(compressed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_img)
    
    # 计算压缩率
    original_size = img.size
    compressed_size = n_colors * 3 + len(labels) * np.log2(n_colors) / 8
    ratio = original_size / compressed_size
    
    print(f"原始大小: {original_size} bytes")
    print(f"压缩后: {compressed_size:.0f} bytes")
    print(f"压缩率: {ratio:.2f}x")
    
    return compressed


def compare_initializations(X: np.ndarray, n_clusters: int = 3,
                           n_runs: int = 10) -> dict:
    """
    比较不同初始化方法的效果
    
    返回:
    - 各初始化方法的WCSS统计
    """
    from sklearn.cluster import KMeans
    
    results = {
        'random': [],
        'k-means++': []
    }
    
    for init in ['random', 'k-means++']:
        for _ in range(n_runs):
            kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=1,
                          random_state=None)
            kmeans.fit(X)
            results[init].append(kmeans.inertia_)
    
    print(f"=== {n_runs}次运行统计 ===")
    for init, wcss_list in results.items():
        print(f"\\n{init}:")
        print(f"  平均WCSS: {np.mean(wcss_list):.2f}")
        print(f"  标准差: {np.std(wcss_list):.2f}")
        print(f"  最小WCSS: {np.min(wcss_list):.2f}")
        print(f"  最大WCSS: {np.max(wcss_list):.2f}")
    
    return results


def cluster_validation_metrics(X: np.ndarray, labels: np.ndarray,
                               true_labels: np.ndarray = None) -> dict:
    """
    计算多种聚类评估指标
    
    指标:
    - WCSS (内部指标)
    - 轮廓系数 (内部指标)
    - ARI (外部指标，需要真实标签)
    - NMI (外部指标，需要真实标签)
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    metrics = {}
    
    # 内部指标
    kmeans = SklearnKMeans(n_clusters=len(np.unique(labels)), random_state=42)
    kmeans.fit(X)
    metrics['WCSS'] = kmeans.inertia_
    
    if len(np.unique(labels)) > 1:
        metrics['Silhouette'] = silhouette_score(X, labels)
        metrics['Calinski-Harabasz'] = calinski_harabasz_score(X, labels)
        metrics['Davies-Bouldin'] = davies_bouldin_score(X, labels)
    
    # 外部指标
    if true_labels is not None:
        metrics['ARI'] = adjusted_rand_score(true_labels, labels)
        metrics['NMI'] = normalized_mutual_info_score(true_labels, labels)
    
    print("=== 聚类评估指标 ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    return metrics


def customer_segmentation_example():
    """客户分群实战示例"""
    np.random.seed(42)
    
    # 生成模拟客户数据
    n_customers = 1000
    
    # 特征：年龄、年收入、消费频次、平均订单金额
    age = np.random.randint(18, 70, n_customers)
    income = np.random.normal(50000, 20000, n_customers)
    frequency = np.random.poisson(10, n_customers)
    avg_order = np.random.normal(100, 50, n_customers)
    
    X = np.column_stack([age, income, frequency, avg_order])
    
    # 标准化
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    
    # 肘部法则
    wcss = []
    for k in range(2, 10):
        kmeans = SklearnKMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    # 选择k=4
    kmeans = SklearnKMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    # 分析每个簇
    print("=== 客户分群结果 ===")
    for i in range(4):
        mask = (labels == i)
        cluster_size = mask.sum()
        
        print(f"\\n簇 {i} ({cluster_size}人, {cluster_size/n_customers*100:.1f}%):")
        print(f"  平均年龄: {age[mask].mean():.1f}岁")
        print(f"  平均收入: ${income[mask].mean():.0f}")
        print(f"  平均消费频次: {frequency[mask].mean():.1f}次/年")
        print(f"  平均订单金额: ${avg_order[mask].mean():.0f}")
        
        # 给簇起个名字
        avg_income = income[mask].mean()
        avg_freq = frequency[mask].mean()
        
        if avg_income > 60000 and avg_freq > 12:
            name = "高价值客户"
        elif avg_income > 60000 and avg_freq <= 12:
            name = "潜力客户"
        elif avg_income <= 60000 and avg_freq > 12:
            name = "忠实客户"
        else:
            name = "普通客户"
        
        print(f"  标签: {name}")
    
    return labels, kmeans


def kmeans_mnist_demo():
    """K-Means在MNIST上的演示"""
    print("加载MNIST数据...")
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # 降维到2D用于可视化
    print("PCA降维...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # K-Means聚类
    print("K-Means聚类...")
    kmeans = SklearnKMeans(n_clusters=10, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # 计算与真实标签的匹配度
    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)
    
    print(f"\\nARI (调整兰德指数): {ari:.4f}")
    print(f"NMI (归一化互信息): {nmi:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 真实标签
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.5)
    axes[0].set_title('真实标签')
    axes[0].set_xlabel('第一主成分')
    axes[0].set_ylabel('第二主成分')
    plt.colorbar(scatter1, ax=axes[0], label='数字')
    
    # K-Means结果
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.5)
    axes[1].set_title('K-Means聚类结果')
    axes[1].set_xlabel('第一主成分')
    axes[1].set_ylabel('第二主成分')
    plt.colorbar(scatter2, ax=axes[1], label='簇')
    
    plt.tight_layout()
    plt.savefig('mnist_kmeans_comparison.png', dpi=150)
    print("保存可视化: mnist_kmeans_comparison.png")
    
    return labels


def visualize_decision_boundaries(X: np.ndarray, y: np.ndarray,
                                  n_clusters: int = 3):
    """可视化K-Means决策边界"""
    # 训练K-Means
    kmeans = SklearnKMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # 预测网格点
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.scatter(kmeans.cluster_centers_[:, 0], 
               kmeans.cluster_centers_[:, 1],
               marker='x', s=200, c='red', linewidths=3)
    plt.title('K-Means决策边界 (Voronoi图)')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.savefig('kmeans_decision_boundaries.png', dpi=150)
    print("保存可视化: kmeans_decision_boundaries.png")


# 运行示例
if __name__ == '__main__':
    print("=" * 50)
    print("K-Means高级应用示例")
    print("=" * 50)
    
    # 客户分群
    print("\\n1. 客户分群示例")
    print("-" * 30)
    customer_segmentation_example()
    
    # 比较初始化方法
    print("\\n2. 比较初始化方法")
    print("-" * 30)
    X, _ = make_blobs(n_samples=500, centers=3, cluster_std=0.60, random_state=42)
    compare_initializations(X, n_clusters=3, n_runs=10)
    
    print("\\n完成！")
