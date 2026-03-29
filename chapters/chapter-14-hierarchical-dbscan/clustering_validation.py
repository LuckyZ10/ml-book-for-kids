"""
聚类评估指标实现
第十四章：层次聚类与DBSCAN

包含经典评估指标:
- Silhouette Score (Rousseeuw, 1987)
- Dunn Index (Dunn, 1974)
- Davies-Bouldin Index (Davies & Bouldin, 1979)
- Calinski-Harabasz Index (Calinski & Harabasz, 1974)
- Elbow Method (肘部法则)

参考文献:
[1] Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the 
    interpretation and validation of cluster analysis. Journal of 
    computational and applied mathematics, 20, 53-65.
[2] Dunn, J. C. (1974). Well-separated clusters and optimal fuzzy 
    partitions. Journal of cybernetics, 4(1), 95-104.
[3] Davies, D. L., & Bouldin, D. W. (1979). A cluster separation 
    measure. IEEE transactions on pattern analysis and machine 
    intelligence, (2), 224-227.
[4] Caliński, T., & Harabasz, J. (1974). A dendrite method for 
    cluster analysis. Communications in Statistics, 3(1), 1-27.

作者: ML教材写作项目
日期: 2026-03-24
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


class SilhouetteScore:
    """
    轮廓系数 (Silhouette Score)
    
    由Rousseeuw (1987)提出，用于评估聚类质量。
    
    核心思想：
    - 衡量每个点与其所属簇的相似度（凝聚度 a）
    - 衡量该点与最近的其他簇的相似度（分离度 b）
    - 轮廓系数 s = (b - a) / max(a, b)
    
    取值范围：[-1, 1]
    - 接近1：聚类效果好
    - 接近0：簇有重叠
    - 接近-1：可能被分到错误的簇
    """
    
    @staticmethod
    def calculate(X: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        计算轮廓系数
        
        参数:
            X: 数据矩阵，形状 (n_samples, n_features)
            labels: 聚类标签，形状 (n_samples,)
            
        返回:
            (平均轮廓系数, 每个点的轮廓系数数组)
        """
        n_samples = len(X)
        unique_labels = np.unique(labels)
        
        # 如果只有一个簇或所有点都是噪声，返回0
        if len(unique_labels) <= 1 or -1 in labels:
            return 0.0, np.zeros(n_samples)
        
        silhouette_values = np.zeros(n_samples)
        
        for i in range(n_samples):
            label_i = labels[i]
            
            # 计算凝聚度 a(i)
            same_cluster_mask = labels == label_i
            same_cluster_indices = np.where(same_cluster_mask)[0]
            
            if len(same_cluster_indices) == 1:
                # 如果簇中只有这一个点
                silhouette_values[i] = 0
                continue
            
            # a(i) = 点i到同簇其他点的平均距离
            a_i = np.mean([
                np.linalg.norm(X[i] - X[j])
                for j in same_cluster_indices if j != i
            ])
            
            # 计算分离度 b(i)
            b_i = float('inf')
            for label in unique_labels:
                if label != label_i:
                    other_cluster_mask = labels == label
                    other_cluster_indices = np.where(other_cluster_mask)[0]
                    
                    # 到其他簇的平均距离
                    avg_dist = np.mean([
                        np.linalg.norm(X[i] - X[j])
                        for j in other_cluster_indices
                    ])
                    b_i = min(b_i, avg_dist)
            
            # 计算轮廓系数 s(i)
            silhouette_values[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        
        avg_silhouette = np.mean(silhouette_values)
        return avg_silhouette, silhouette_values
    
    @staticmethod
    def calculate_for_cluster(X: np.ndarray, labels: np.ndarray, 
                              cluster_label: int) -> float:
        """计算特定簇的轮廓系数"""
        _, all_scores = SilhouetteScore.calculate(X, labels)
        cluster_mask = labels == cluster_label
        return np.mean(all_scores[cluster_mask])


class DunnIndex:
    """
    Dunn指数 (Dunn Index)
    
    由Dunn (1974)提出，衡量簇间的最小距离与簇内最大直径的比值。
    
    公式: D = min_{i≠j} d(C_i, C_j) / max_k diam(C_k)
    
    其中:
    - d(C_i, C_j): 簇i和簇j之间的最小距离
    - diam(C_k): 簇k的直径（簇内最远两点距离）
    
    Dunn指数越大，聚类效果越好。
    """
    
    @staticmethod
    def calculate(X: np.ndarray, labels: np.ndarray) -> float:
        """
        计算Dunn指数
        
        参数:
            X: 数据矩阵
            labels: 聚类标签
            
        返回:
            Dunn指数值
        """
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # 排除噪声
        
        if len(unique_labels) < 2:
            return 0.0
        
        # 计算每个簇的直径
        max_diameter = 0
        for label in unique_labels:
            cluster_points = X[labels == label]
            if len(cluster_points) > 1:
                # 计算簇内所有点对距离的最大值
                diam = DunnIndex._cluster_diameter(cluster_points)
                max_diameter = max(max_diameter, diam)
        
        if max_diameter == 0:
            return float('inf')
        
        # 计算簇间最小距离
        min_intercluster_dist = float('inf')
        for i, label_i in enumerate(unique_labels):
            for label_j in unique_labels[i+1:]:
                cluster_i = X[labels == label_i]
                cluster_j = X[labels == label_j]
                dist = DunnIndex._intercluster_distance(cluster_i, cluster_j)
                min_intercluster_dist = min(min_intercluster_dist, dist)
        
        return min_intercluster_dist / max_diameter
    
    @staticmethod
    def _cluster_diameter(points: np.ndarray) -> float:
        """计算簇的直径（最远两点距离）"""
        if len(points) <= 1:
            return 0
        
        max_dist = 0
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                max_dist = max(max_dist, dist)
        return max_dist
    
    @staticmethod
    def _intercluster_distance(cluster1: np.ndarray, 
                               cluster2: np.ndarray) -> float:
        """计算两个簇之间的最小距离"""
        min_dist = float('inf')
        for p1 in cluster1:
            for p2 in cluster2:
                dist = np.linalg.norm(p1 - p2)
                min_dist = min(min_dist, dist)
        return min_dist


class DaviesBouldinIndex:
    """
    Davies-Bouldin指数
    
    由Davies & Bouldin (1979)提出，衡量簇的相似度。
    
    核心思想：
    - 好的聚类应该簇内紧凑、簇间分离
    - 计算每个簇与其最相似簇的相似度
    - 取平均作为最终指标
    
    公式: DB = (1/k) * Σ_{i=1}^k max_{j≠i} (s_i + s_j) / d_{ij}
    
    其中:
    - s_i: 簇i的分散度（通常是簇内点到质心的平均距离）
    - d_{ij}: 簇i和簇j质心之间的距离
    
    DB指数越小，聚类效果越好。
    """
    
    @staticmethod
    def calculate(X: np.ndarray, labels: np.ndarray) -> float:
        """
        计算Davies-Bouldin指数
        
        参数:
            X: 数据矩阵
            labels: 聚类标签
            
        返回:
            DB指数值
        """
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        k = len(unique_labels)
        
        if k < 2:
            return float('inf')
        
        # 计算每个簇的质心和分散度
        centroids = {}
        dispersions = {}
        
        for label in unique_labels:
            cluster_points = X[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            centroids[label] = centroid
            
            # 分散度：簇内点到质心的平均距离
            dispersion = np.mean([
                np.linalg.norm(p - centroid) for p in cluster_points
            ])
            dispersions[label] = dispersion
        
        # 计算DB指数
        db_sum = 0
        for i, label_i in enumerate(unique_labels):
            max_similarity = 0
            for j, label_j in enumerate(unique_labels):
                if i != j:
                    # 计算两个簇的相似度
                    s_i = dispersions[label_i]
                    s_j = dispersions[label_j]
                    d_ij = np.linalg.norm(centroids[label_i] - centroids[label_j])
                    
                    if d_ij > 0:
                        similarity = (s_i + s_j) / d_ij
                        max_similarity = max(max_similarity, similarity)
            
            db_sum += max_similarity
        
        return db_sum / k


class CalinskiHarabaszIndex:
    """
    Calinski-Harabasz指数 (CH指数)
    
    由Caliński & Harabasz (1974)提出，也称为方差比准则。
    
    公式: CH = [B/(k-1)] / [W/(n-k)]
    
    其中:
    - B: 簇间平方和
    - W: 簇内平方和
    - k: 簇数量
    - n: 样本数量
    
    CH指数越大，聚类效果越好。
    """
    
    @staticmethod
    def calculate(X: np.ndarray, labels: np.ndarray) -> float:
        """
        计算Calinski-Harabasz指数
        
        参数:
            X: 数据矩阵
            labels: 聚类标签
            
        返回:
            CH指数值
        """
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        k = len(unique_labels)
        n = len(X)
        
        if k < 2 or n <= k:
            return 0.0
        
        # 全局质心
        overall_centroid = np.mean(X, axis=0)
        
        # 计算簇间平方和 B
        B = 0
        for label in unique_labels:
            cluster_points = X[labels == label]
            n_i = len(cluster_points)
            centroid_i = np.mean(cluster_points, axis=0)
            B += n_i * np.sum((centroid_i - overall_centroid) ** 2)
        
        # 计算簇内平方和 W
        W = 0
        for label in unique_labels:
            cluster_points = X[labels == label]
            centroid_i = np.mean(cluster_points, axis=0)
            for point in cluster_points:
                W += np.sum((point - centroid_i) ** 2)
        
        if W == 0:
            return float('inf')
        
        return (B / (k - 1)) / (W / (n - k))


class ElbowMethod:
    """
    肘部法则 (Elbow Method)
    
    用于确定最佳聚类数量k的经典方法。
    
    原理：
    - 计算不同k值下的簇内平方和(SSE)
    - 随着k增加，SSE单调递减
    - 寻找"肘部"点：SSE下降速度明显变缓的位置
    
    数学上对应寻找二阶导数最大的点。
    """
    
    @staticmethod
    def calculate_sse(X: np.ndarray, labels: np.ndarray) -> float:
        """计算簇内平方和 (Sum of Squared Errors)"""
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        
        sse = 0
        for label in unique_labels:
            cluster_points = X[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            for point in cluster_points:
                sse += np.sum((point - centroid) ** 2)
        
        return sse
    
    @staticmethod
    def find_elbow(sse_values: List[float], k_values: List[int]) -> int:
        """
        自动寻找肘部点
        
        使用二阶导数法寻找曲率最大的点
        """
        if len(sse_values) < 3:
            return k_values[0] if k_values else 2
        
        # 计算二阶差分（离散的二阶导数）
        second_derivatives = []
        for i in range(1, len(sse_values) - 1):
            # 中心差分近似二阶导数
            second_deriv = sse_values[i-1] - 2*sse_values[i] + sse_values[i+1]
            second_derivatives.append(abs(second_deriv))
        
        if not second_derivatives:
            return k_values[len(k_values)//2]
        
        # 找到二阶导数最大的点
        elbow_idx = np.argmax(second_derivatives) + 1
        return k_values[elbow_idx]


class ClusteringVisualizer:
    """聚类评估可视化器"""
    
    @staticmethod
    def plot_silhouette_analysis(X: np.ndarray, labels: np.ndarray,
                                  title: str = "Silhouette Analysis"):
        """
        绘制轮廓分析图
        
        显示每个簇的轮廓系数分布
        """
        avg_score, individual_scores = SilhouetteScore.calculate(X, labels)
        unique_labels = sorted(set(labels))
        unique_labels = [l for l in unique_labels if l != -1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：轮廓系数条形图
        y_lower = 10
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            cluster_scores = individual_scores[labels == label]
            cluster_scores.sort()
            
            size_cluster = len(cluster_scores)
            y_upper = y_lower + size_cluster
            
            color = colors[i]
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_scores,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            ax1.text(-0.05, y_lower + 0.5 * size_cluster, str(label))
            y_lower = y_upper + 10
        
        ax1.set_xlabel("Silhouette Coefficient")
        ax1.set_ylabel("Cluster Label")
        ax1.set_title(title)
        ax1.axvline(x=avg_score, color="red", linestyle="--",
                   label=f'Average: {avg_score:.3f}')
        ax1.legend()
        
        # 右图：聚类散点图
        scatter = ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=50)
        ax2.set_title("Cluster Visualization")
        ax2.set_xlabel("Feature 1")
        ax2.set_ylabel("Feature 2")
        plt.colorbar(scatter, ax=ax2, label='Cluster')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_elbow_curve(X: np.ndarray, k_range: range = range(2, 11),
                         clustering_func=None):
        """
        绘制肘部曲线
        
        参数:
            X: 数据
            k_range: 要测试的k值范围
            clustering_func: 聚类函数（接收n_clusters参数）
        """
        if clustering_func is None:
            # 默认使用简单的K-Means
            from chapter_13_kmeans.kmeans_clustering import KMeans
            clustering_func = lambda k: KMeans(n_clusters=k, random_state=42)
        
        sse_values = []
        k_values = list(k_range)
        
        for k in k_values:
            model = clustering_func(k)
            labels = model.fit_predict(X)
            sse = ElbowMethod.calculate_sse(X, labels)
            sse_values.append(sse)
        
        # 寻找肘部点
        elbow_k = ElbowMethod.find_elbow(sse_values, k_values)
        
        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, sse_values, 'bo-', linewidth=2, markersize=8)
        plt.axvline(x=elbow_k, color='r', linestyle='--',
                   label=f'Elbow at k={elbow_k}')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Sum of Squared Errors (SSE)')
        plt.title('Elbow Method for Optimal k')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf(), elbow_k
    
    @staticmethod
    def plot_validation_comparison(X: np.ndarray, labels_list: List[np.ndarray],
                                   method_names: List[str]):
        """
        比较不同聚类方法的评估指标
        """
        metrics = {
            'Silhouette': [],
            'Dunn': [],
            'Davies-Bouldin': [],
            'Calinski-Harabasz': []
        }
        
        for labels in labels_list:
            metrics['Silhouette'].append(
                SilhouetteScore.calculate(X, labels)[0]
            )
            metrics['Dunn'].append(
                DunnIndex.calculate(X, labels)
            )
            metrics['Davies-Bouldin'].append(
                DaviesBouldinIndex.calculate(X, labels)
            )
            metrics['Calinski-Harabasz'].append(
                CalinskiHarabaszIndex.calculate(X, labels)
            )
        
        # 归一化以便比较（DB指数取倒数）
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx]
            bars = ax.bar(method_names, values, color=plt.cm.Set3(np.linspace(0, 1, len(values))))
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Score Comparison')
            ax.tick_params(axis='x', rotation=45)
            
            # 在条形上添加数值
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig, metrics


# ==================== 综合评估报告 ====================

class ClusteringReport:
    """聚类评估综合报告生成器"""
    
    @staticmethod
    def generate_report(X: np.ndarray, labels: np.ndarray, 
                       method_name: str = "Clustering") -> Dict:
        """
        生成聚类评估报告
        
        返回包含所有评估指标的字典
        """
        report = {
            'method': method_name,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'n_noise': list(labels).count(-1),
        }
        
        # 计算各项指标
        silhouette_avg, silhouette_per_sample = SilhouetteScore.calculate(X, labels)
        report['silhouette_score'] = silhouette_avg
        report['silhouette_per_cluster'] = {}
        
        for label in set(labels):
            if label != -1:
                cluster_sil = SilhouetteScore.calculate_for_cluster(X, labels, label)
                report['silhouette_per_cluster'][label] = cluster_sil
        
        report['dunn_index'] = DunnIndex.calculate(X, labels)
        report['davies_bouldin_index'] = DaviesBouldinIndex.calculate(X, labels)
        report['calinski_harabasz_index'] = CalinskiHarabaszIndex.calculate(X, labels)
        
        return report
    
    @staticmethod
    def print_report(report: Dict):
        """打印评估报告"""
        print("=" * 60)
        print(f"聚类评估报告: {report['method']}")
        print("=" * 60)
        print(f"样本数量: {report['n_samples']}")
        print(f"特征数量: {report['n_features']}")
        print(f"簇数量: {report['n_clusters']}")
        print(f"噪声点: {report['n_noise']} ({report['n_noise']/report['n_samples']*100:.1f}%)")
        print("-" * 60)
        print("评估指标:")
        print(f"  Silhouette Score: {report['silhouette_score']:.4f}")
        print(f"  Dunn Index: {report['dunn_index']:.4f}")
        print(f"  Davies-Bouldin Index: {report['davies_bouldin_index']:.4f}")
        print(f"  Calinski-Harabasz Index: {report['calinski_harabasz_index']:.2f}")
        print("-" * 60)
        print("各簇Silhouette Score:")
        for label, score in report['silhouette_per_cluster'].items():
            print(f"  Cluster {label}: {score:.4f}")
        print("=" * 60)


# ==================== 演示函数 ====================

def demo_silhouette_analysis():
    """轮廓系数分析演示"""
    from sklearn.datasets import make_blobs
    
    # 创建数据
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6,
                          random_state=42)
    
    # 使用K-Means聚类
    from chapter_13_kmeans.kmeans_clustering import KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # 计算轮廓系数
    avg_score, scores = SilhouetteScore.calculate(X, labels)
    print(f"平均轮廓系数: {avg_score:.4f}")
    
    # 可视化
    ClusteringVisualizer.plot_silhouette_analysis(X, labels)
    plt.savefig('silhouette_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_elbow_method():
    """肘部法则演示"""
    from sklearn.datasets import make_blobs
    
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6,
                     random_state=42)
    
    fig, elbow_k = ClusteringVisualizer.plot_elbow_curve(X, range(2, 11))
    plt.savefig('elbow_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"建议的最佳k值: {elbow_k}")


def demo_comparison():
    """多方法评估对比演示"""
    from sklearn.datasets import make_moons, make_blobs
    
    # 创建数据
    X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    X_blobs, _ = make_blobs(n_samples=300, centers=3, random_state=42)
    
    # 不同方法聚类
    from chapter_13_kmeans.kmeans_clustering import KMeans
    from dbscan_clustering import DBSCAN
    
    results = []
    names = []
    
    # K-Means on blobs
    km = KMeans(n_clusters=3, random_state=42)
    results.append(km.fit_predict(X_blobs))
    names.append("K-Means (Blobs)")
    
    # DBSCAN on moons
    db = DBSCAN(eps=0.2, min_pts=5)
    results.append(db.fit_predict(X_moons))
    names.append("DBSCAN (Moons)")
    
    # 生成报告
    for X, labels, name in [(X_blobs, results[0], names[0]),
                           (X_moons, results[1], names[1])]:
        report = ClusteringReport.generate_report(X, labels, name)
        ClusteringReport.print_report(report)
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("聚类评估指标演示")
    print("=" * 60)
    
    print("\n【演示1】轮廓系数分析")
    demo_silhouette_analysis()
    
    print("\n【演示2】肘部法则")
    demo_elbow_method()
    
    print("\n【演示3】综合评估报告")
    demo_comparison()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
