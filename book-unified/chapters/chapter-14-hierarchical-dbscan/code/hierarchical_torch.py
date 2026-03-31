"""
层次聚类 PyTorch 实现
第十四章：层次聚类与DBSCAN

包含：
- AGNES (凝聚式层次聚类) - GPU加速版本
- 支持批量距离计算
- 自动梯度兼容（可用于端到端学习）

作者: ML教材写作项目
日期: 2026-03-30
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass


class AgglomerativeClusteringTorch:
    """
    凝聚式层次聚类 (AGNES) - PyTorch GPU加速实现
    
    利用PyTorch的张量运算和GPU加速，适合大规模数据
    
    Parameters
    ----------
    n_clusters : int
        目标聚类数量
    linkage : {'single', 'complete', 'average', 'ward'}
        链接方法
    device : str
        计算设备 ('cpu', 'cuda', 'auto')
    
    Examples
    --------
    >>> import torch
    >>> X = torch.randn(1000, 10)
    >>> agg = AgglomerativeClusteringTorch(n_clusters=5, device='cuda')
    >>> labels = agg.fit_predict(X)
    """
    
    LINKAGE_METHODS = ['single', 'complete', 'average', 'ward']
    
    def __init__(
        self,
        n_clusters: int = 2,
        linkage: Literal['single', 'complete', 'average', 'ward'] = 'ward',
        device: str = 'auto'
    ):
        self.n_clusters = n_clusters
        self.linkage = linkage
        
        # 自动选择设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.labels_ = None
        self.linkage_matrix_ = None
        self.n_samples_ = None
        
    def _compute_distance_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        高效计算距离矩阵 - GPU加速
        
        使用广播机制避免显式循环
        """
        # X: (n, d) -> ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2*x_i·x_j
        X = X.to(self.device)
        
        # 计算范数平方: (n, 1)
        sq_norms = torch.sum(X ** 2, dim=1, keepdim=True)
        
        # 距离矩阵: (n, n)
        # dist[i,j] = ||x_i||² + ||x_j||² - 2 * dot(x_i, x_j)
        dist_matrix = sq_norms + sq_norms.T - 2 * torch.mm(X, X.T)
        
        # 处理数值误差
        dist_matrix = torch.clamp(dist_matrix, min=0.0)
        
        return torch.sqrt(dist_matrix)
    
    def _lance_williams_params_torch(
        self,
        size_i: int,
        size_j: int,
        size_k: int
    ) -> Tuple[float, float, float, float]:
        """Lance-Williams递推公式参数 - PyTorch版本"""
        n_total = size_i + size_j
        
        if self.linkage == 'single':
            return 0.5, 0.5, 0.0, -0.5
        elif self.linkage == 'complete':
            return 0.5, 0.5, 0.0, 0.5
        elif self.linkage == 'average':
            return size_i / n_total, size_j / n_total, 0.0, 0.0
        elif self.linkage == 'ward':
            T = size_i + size_j + size_k
            return (size_i + size_k) / T, (size_j + size_k) / T, -size_k / T, 0.0
        else:
            raise ValueError(f"不支持的链接方法: {self.linkage}")
    
    def fit(self, X) -> 'AgglomerativeClusteringTorch':
        """
        训练层次聚类模型
        
        Parameters
        ----------
        X : array-like or torch.Tensor of shape (n_samples, n_features)
            训练数据
        
        Returns
        -------
        self
        """
        # 转换为torch张量
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        
        self.n_samples_ = n = X.shape[0]
        
        if n < self.n_clusters:
            raise ValueError(f"样本数{n}必须大于等于聚类数{self.n_clusters}")
        
        # 计算距离矩阵 (在GPU上)
        dist_matrix = self._compute_distance_matrix(X)
        
        # 转换为numpy进行后续处理（聚类算法本身不易并行化）
        dist_matrix_np = dist_matrix.cpu().numpy()
        
        # 初始化簇
        clusters = {i: [i] for i in range(n)}
        cluster_sizes = {i: 1 for i in range(n)}
        
        # 设置对角线为无穷大
        np.fill_diagonal(dist_matrix_np, np.inf)
        
        linkage = []
        next_cluster_id = n
        active = set(range(n))
        
        print(f"[AGNES-Torch] 设备: {self.device}")
        print(f"[AGNES-Torch] 开始聚类: n_samples={n}, linkage='{self.linkage}'")
        
        # 主循环
        while len(active) > self.n_clusters:
            # 在活跃簇中找到距离最近的两个
            min_dist = np.inf
            to_merge = None
            
            active_list = list(active)
            for idx, i in enumerate(active_list):
                for j in active_list[idx + 1:]:
                    if dist_matrix_np[i, j] < min_dist:
                        min_dist = dist_matrix_np[i, j]
                        to_merge = (i, j)
            
            if to_merge is None:
                break
            
            ci, cj = to_merge
            
            linkage.append([
                ci, cj, min_dist,
                cluster_sizes[ci] + cluster_sizes[cj]
            ])
            
            # 创建新簇
            new_cluster_id = next_cluster_id
            clusters[new_cluster_id] = clusters[ci] + clusters[cj]
            cluster_sizes[new_cluster_id] = cluster_sizes[ci] + cluster_sizes[cj]
            
            # 更新距离矩阵
            alpha_i, alpha_j, beta, gamma = self._lance_williams_params_torch(
                cluster_sizes[ci], cluster_sizes[cj], 0
            )
            
            for k in active:
                if k != ci and k != cj:
                    # Lance-Williams更新
                    d_ki = dist_matrix_np[ci, k]
                    d_kj = dist_matrix_np[cj, k]
                    d_ij = dist_matrix_np[ci, cj]
                    
                    new_dist = (
                        alpha_i * d_ki +
                        alpha_j * d_kj +
                        beta * d_ij +
                        gamma * abs(d_ki - d_kj)
                    )
                    dist_matrix_np[new_cluster_id, k] = new_dist
                    dist_matrix_np[k, new_cluster_id] = new_dist
            
            # 标记旧簇
            dist_matrix_np[ci, :] = np.inf
            dist_matrix_np[:, ci] = np.inf
            dist_matrix_np[cj, :] = np.inf
            dist_matrix_np[:, cj] = np.inf
            
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
        
        print(f"[AGNES-Torch] 聚类完成: 发现 {len(active)} 个簇")
        
        return self
    
    def fit_predict(self, X) -> np.ndarray:
        """训练并返回聚类标签"""
        self.fit(X)
        return self.labels_


class DifferentiableLinkage(nn.Module):
    """
    可微分的链接方法 - 用于端到端学习
    
    将层次聚类嵌入神经网络，支持梯度回传
    
    Parameters
    ----------
    linkage : str
        链接方法
    temperature : float
        softmax温度参数，控制离散程度
    """
    
    def __init__(self, linkage: str = 'average', temperature: float = 0.1):
        super().__init__()
        self.linkage = linkage
        self.temperature = temperature
    
    def forward(self, cluster_reps: torch.Tensor) -> torch.Tensor:
        """
        计算簇间距离（可微分版本）
        
        Parameters
        ----------
        cluster_reps : torch.Tensor of shape (n_clusters, n_features)
            簇的表示向量
        
        Returns
        -------
        distances : torch.Tensor of shape (n_clusters, n_clusters)
            簇间距离矩阵
        """
        # 使用欧氏距离，但保持可微分
        sq_norms = torch.sum(cluster_reps ** 2, dim=1, keepdim=True)
        distances = sq_norms + sq_norms.T - 2 * torch.mm(cluster_reps, cluster_reps.T)
        distances = torch.sqrt(torch.clamp(distances, min=1e-8))
        
        return distances
    
    def soft_merge(self, distances: torch.Tensor, 
                   cluster_sizes: torch.Tensor) -> torch.Tensor:
        """
        软合并操作 - 使用softmax近似argmin
        
        这使得合并操作可微分，可用于梯度优化
        """
        # 将对角线设为无穷大（排除自身）
        distances = distances.clone()
        distances.fill_diagonal_(float('inf'))
        
        # softmax权重: 距离越小权重越大
        weights = torch.softmax(-distances / self.temperature, dim=1)
        
        return weights


class NeuralHierarchicalClustering(nn.Module):
    """
    神经层次聚类 - 端到端可学习版本
    
    将数据编码到潜在空间，然后在潜在空间进行层次聚类
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        n_clusters: int = 5
    ):
        super().__init__()
        
        self.n_clusters = n_clusters
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # 可微分链接
        self.linkage = DifferentiableLinkage(linkage='average')
        
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """编码数据到潜在空间"""
        return self.encoder(X)
    
    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns
        -------
        latent : torch.Tensor
            潜在空间表示
        assignments : torch.Tensor
            软聚类分配矩阵
        """
        latent = self.encode(X)
        
        # 使用soft K-Means初始化
        # 然后应用层次聚类的思想
        # ... (简化版本)
        
        return latent, torch.softmax(latent, dim=1)


def benchmark_torch_vs_numpy():
    """对比PyTorch和NumPy实现的性能"""
    import time
    from sklearn.datasets import make_blobs
    
    print("=" * 60)
    print("PyTorch vs NumPy 性能对比")
    print("=" * 60)
    
    sample_sizes = [100, 500, 1000, 2000]
    
    for n_samples in sample_sizes:
        print(f"\n样本数: {n_samples}")
        print("-" * 40)
        
        X, _ = make_blobs(n_samples=n_samples, centers=5, 
                         n_features=10, random_state=42)
        
        # NumPy版本
        from hierarchical_numpy import AgglomerativeClusteringNumPy
        
        start = time.time()
        agg_np = AgglomerativeClusteringNumPy(n_clusters=5, linkage='ward')
        labels_np = agg_np.fit_predict(X)
        time_np = time.time() - start
        
        # PyTorch CPU版本
        start = time.time()
        agg_torch_cpu = AgglomerativeClusteringTorch(
            n_clusters=5, linkage='ward', device='cpu'
        )
        labels_torch_cpu = agg_torch_cpu.fit_predict(X)
        time_torch_cpu = time.time() - start
        
        # PyTorch GPU版本（如果有CUDA）
        if torch.cuda.is_available():
            start = time.time()
            agg_torch_gpu = AgglomerativeClusteringTorch(
                n_clusters=5, linkage='ward', device='cuda'
            )
            labels_torch_gpu = agg_torch_gpu.fit_predict(X)
            time_torch_gpu = time.time() - start
            gpu_info = f"| {time_torch_gpu:.3f}s (GPU)"
        else:
            gpu_info = "| N/A (无GPU)"
        
        print(f"NumPy:     {time_np:.3f}s")
        print(f"PyTorch CPU: {time_torch_cpu:.3f}s {gpu_info}")
        
        # 验证结果一致性
        # 注意：由于浮点精度，结果可能略有不同


def demo_gpu_acceleration():
    """演示GPU加速效果"""
    import time
    
    print("=" * 60)
    print("GPU加速演示")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过GPU演示")
        return
    
    print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    
    # 生成大规模数据
    n_samples = 5000
    n_features = 50
    
    print(f"\n数据规模: {n_samples} 样本 x {n_features} 特征")
    print("-" * 40)
    
    X = torch.randn(n_samples, n_features)
    
    # CPU版本
    start = time.time()
    agg_cpu = AgglomerativeClusteringTorch(n_clusters=10, device='cpu')
    labels_cpu = agg_cpu.fit_predict(X)
    time_cpu = time.time() - start
    
    # GPU版本
    start = time.time()
    agg_gpu = AgglomerativeClusteringTorch(n_clusters=10, device='cuda')
    labels_gpu = agg_gpu.fit_predict(X)
    time_gpu = time.time() - start
    
    print(f"CPU时间: {time_cpu:.3f}s")
    print(f"GPU时间: {time_gpu:.3f}s")
    print(f"加速比: {time_cpu / time_gpu:.2f}x")


def demo():
    """主要演示"""
    from sklearn.datasets import make_blobs
    
    print("=" * 60)
    print("层次聚类 PyTorch 实现演示")
    print("=" * 60)
    
    # 示例1：基本使用
    print("\n【示例1】基本聚类")
    print("-" * 40)
    
    X, y_true = make_blobs(n_samples=300, centers=4, 
                          cluster_std=0.6, random_state=42)
    
    agg = AgglomerativeClusteringTorch(n_clusters=4, linkage='ward')
    labels = agg.fit_predict(X)
    
    print(f"真实标签数: {len(np.unique(y_true))}")
    print(f"预测标签数: {len(np.unique(labels))}")
    print(f"各簇大小: {[np.sum(labels == i) for i in range(4)]}")
    
    # 示例2：不同链接方法
    print("\n【示例2】不同链接方法对比")
    print("-" * 40)
    
    for method in ['single', 'complete', 'average', 'ward']:
        agg = AgglomerativeClusteringTorch(n_clusters=4, linkage=method)
        labels = agg.fit_predict(X)
        
        # 计算轮廓系数
        from sklearn.metrics import silhouette_score
        score = silhouette_score(X, labels)
        print(f"{method:10s}: 轮廓系数 = {score:.4f}")
    
    # 示例3：性能对比
    print("\n【示例3】性能对比")
    print("-" * 40)
    
    benchmark_torch_vs_numpy()
    
    # 示例4：GPU加速（如果有）
    if torch.cuda.is_available():
        print("\n【示例4】GPU加速")
        print("-" * 40)
        demo_gpu_acceleration()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == '__main__':
    demo()
