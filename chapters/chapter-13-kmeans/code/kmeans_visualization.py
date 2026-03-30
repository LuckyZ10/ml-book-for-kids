#!/usr/bin/env python3
"""
K-Means可视化与动画 - 补充代码
Chapter 13: K-Means聚类
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple


class KMeansVisualizer:
    """K-Means训练过程可视化"""
    
    def __init__(self, X: np.ndarray, n_clusters: int = 3):
        self.X = X
        self.n_clusters = n_clusters
        self.centroids_history = []
        self.labels_history = []
        
    def fit_and_record(self, max_iter: int = 20) -> None:
        """训练并记录每一步"""
        n_samples = self.X.shape[0]
        
        # 随机初始化
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        centroids = self.X[idx].copy()
        
        for iteration in range(max_iter):
            # 记录当前状态
            self.centroids_history.append(centroids.copy())
            
            # 分配
            distances = np.sqrt(((self.X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)
            self.labels_history.append(labels.copy())
            
            # 更新
            old_centroids = centroids.copy()
            for k in range(self.n_clusters):
                mask = (labels == k)
                if mask.any():
                    centroids[k] = self.X[mask].mean(axis=0)
            
            # 检查收敛
            if np.allclose(centroids, old_centroids, atol=1e-4):
                print(f"收敛于第 {iteration + 1} 次迭代")
                break
        
        # 记录最终状态
        self.centroids_history.append(centroids.copy())
        distances = np.sqrt(((self.X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        final_labels = np.argmin(distances, axis=1)
        self.labels_history.append(final_labels.copy())
    
    def plot_iteration(self, iteration: int, ax=None):
        """绘制某次迭代的结果"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = self.labels_history[iteration]
        centroids = self.centroids_history[iteration]
        
        # 绘制数据点
        scatter = ax.scatter(self.X[:, 0], self.X[:, 1], 
                           c=labels, cmap='viridis', alpha=0.6, s=50)
        
        # 绘制中心
        ax.scatter(centroids[:, 0], centroids[:, 1],
                  c='red', marker='x', s=200, linewidths=3,
                  label='中心点')
        
        ax.set_title(f'K-Means迭代 {iteration + 1}')
        ax.set_xlabel('特征1')
        ax.set_ylabel('特征2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return scatter
    
    def create_animation(self, save_path: str = 'kmeans_animation.gif'):
        """创建训练过程动画"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def init():
            ax.clear()
            return []
        
        def update(frame):
            ax.clear()
            self.plot_iteration(frame, ax=ax)
            return []
        
        anim = FuncAnimation(fig, update, frames=len(self.labels_history),
                            init_func=init, blit=True, interval=1000)
        anim.save(save_path, writer='pillow', fps=1)
        print(f"动画已保存: {save_path}")
        plt.close()
    
    def plot_convergence(self):
        """绘制收敛过程"""
        # 计算每次迭代的WCSS
        wcss_list = []
        for centroids in self.centroids_history[:-1]:
            distances = np.sqrt(((self.X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)
            wcss = 0
            for k in range(self.n_clusters):
                mask = (labels == k)
                if mask.any():
                    wcss += np.sum((self.X[mask] - centroids[k]) ** 2)
            wcss_list.append(wcss)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(wcss_list) + 1), wcss_list, 'bo-')
        plt.xlabel('迭代次数')
        plt.ylabel('WCSS')
        plt.title('K-Means收敛过程')
        plt.grid(True)
        plt.savefig('kmeans_convergence.png', dpi=150)
        print("保存收敛图: kmeans_convergence.png")


def plot_voronoi_diagram(centroids: np.ndarray, 
                         x_range: Tuple[float, float] = (-5, 5),
                         y_range: Tuple[float, float] = (-5, 5)):
    """绘制Voronoi图"""
    from scipy.spatial import Voronoi, voronoi_plot_2d
    
    # 添加远点使Voronoi图完整
    points = np.vstack([centroids, [[999, 999], [-999, 999], [999, -999], [-999, -999]]])
    
    vor = Voronoi(points)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange',
                    line_width=2, line_alpha=0.6, point_size=10)
    
    # 绘制中心
    ax.scatter(centroids[:, 0], centroids[:, 1], 
              c='red', marker='x', s=300, linewidths=3, label='中心点')
    
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_title('K-Means决策边界 (Voronoi图)')
    ax.legend()
    ax.set_aspect('equal')
    plt.savefig('voronoi_diagram.png', dpi=150)
    print("保存Voronoi图: voronoi_diagram.png")


def plot_cluster_comparison():
    """比较不同聚类数的效果"""
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    
    # 生成数据
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
    
    # 不同k值
    k_values = [2, 3, 4, 5]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, k in enumerate(k_values):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # 绘制
        scatter = axes[idx].scatter(X[:, 0], X[:, 1], c=labels, 
                                   cmap='viridis', alpha=0.6)
        axes[idx].scatter(kmeans.cluster_centers_[:, 0],
                         kmeans.cluster_centers_[:, 1],
                         c='red', marker='x', s=200, linewidths=3)
        axes[idx].set_title(f'k = {k}, WCSS = {kmeans.inertia_:.1f}')
        axes[idx].set_xlabel('特征1')
        axes[idx].set_ylabel('特征2')
    
    plt.tight_layout()
    plt.savefig('cluster_comparison.png', dpi=150)
    print("保存比较图: cluster_comparison.png")


def visualize_silhouette(X: np.ndarray, labels: np.ndarray):
    """可视化轮廓系数"""
    from sklearn.metrics import silhouette_samples, silhouette_score
    
    n_clusters = len(np.unique(labels))
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_lower = 10
    for i in range(n_clusters):
        # 聚合当前簇的轮廓系数
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)
        
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.set_title(f'轮廓系数 (平均={silhouette_avg:.3f})')
    ax.set_xlabel('轮廓系数值')
    ax.set_ylabel('簇标签')
    ax.axvline(x=silhouette_avg, color="red", linestyle="--",
              label=f'平均={silhouette_avg:.3f}')
    ax.legend()
    plt.savefig('silhouette_visualization.png', dpi=150)
    print("保存轮廓图: silhouette_visualization.png")


def interactive_cluster_explorer():
    """交互式聚类探索"""
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    import matplotlib.widgets as widgets
    
    # 生成数据
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)
    
    # 初始聚类
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    centers_plot = ax.scatter(kmeans.cluster_centers_[:, 0],
                             kmeans.cluster_centers_[:, 1],
                             c='red', marker='x', s=200, linewidths=3)
    ax.set_title('K-Means交互式探索 (k=3)')
    
    # 添加滑块
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = widgets.Slider(ax_slider, 'k', 2, 10, valinit=3, valstep=1)
    
    def update(val):
        k = int(slider.val)
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        
        scatter.set_array(labels)
        centers_plot.set_offsets(kmeans.cluster_centers_)
        ax.set_title(f'K-Means交互式探索 (k={k}, WCSS={kmeans.inertia_:.1f})')
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    plt.show()


# 示例运行
if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    
    print("=" * 50)
    print("K-Means可视化示例")
    print("=" * 50)
    
    # 生成数据
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.60, random_state=42)
    
    print("\\n1. 训练过程可视化")
    print("-" * 30)
    visualizer = KMeansVisualizer(X, n_clusters=3)
    visualizer.fit_and_record(max_iter=10)
    visualizer.plot_convergence()
    
    print("\\n2. 不同k值比较")
    print("-" * 30)
    plot_cluster_comparison()
    
    print("\\n3. Voronoi图")
    print("-" * 30)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
    plot_voronoi_diagram(kmeans.cluster_centers_)
    
    print("\\n完成！")
