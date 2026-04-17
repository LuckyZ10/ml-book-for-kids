"""
《机器学习与深度学习：从小学生到大师》
第十五章：降维——抓住主要矛盾
PCA和t-SNE的PyTorch实现

本章内容：
1. GPU加速PCA实现
2. 可微分PCA（支持反向传播）
3. PyTorch版t-SNE
4. 深度学习集成示例

作者：机器学习入门教材
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 第一部分：GPU加速PCA
# ============================================================================

class PCATorch:
    """
    基于PyTorch的PCA实现，支持GPU加速
    
    当数据量很大时，GPU可以显著加速PCA的计算。
    本实现也支持自动微分，可以集成到神经网络中。
    
    生活比喻：
    - CPU就像一个人手工整理照片
    - GPU就像100个人同时整理，速度提升明显！
    
    参数:
        n_components: 保留的主成分数量
        whiten: 是否白化
        device: 计算设备 ('cuda' 或 'cpu')
    """
    
    def __init__(self, n_components: Optional[int] = None, whiten: bool = False, 
                 device: Optional[str] = None):
        self.n_components = n_components
        self.whiten = whiten
        
        # 自动选择设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        
    def fit(self, X: Union[np.ndarray, torch.Tensor]) -> 'PCATorch':
        """
        拟合PCA模型
        
        使用PyTorch的张量运算，可以在GPU上执行。
        """
        # 转换为PyTorch张量
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        X = X.to(self.device)
        n_samples, n_features = X.shape
        
        # 中心化
        self.mean_ = torch.mean(X, dim=0)
        X_centered = X - self.mean_
        
        # 计算协方差矩阵
        # 注意：PyTorch中@表示矩阵乘法
        covariance_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
        
        # 特征值分解
        # torch.linalg.eigh返回特征值（升序）和特征向量
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
        
        # 按特征值从大到小排序
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 确定主成分数量
        if self.n_components is None:
            n_components = n_features
        else:
            n_components = min(self.n_components, n_features)
        
        # 保存结果
        self.components_ = eigenvectors[:, :n_components].T  # (n_components, n_features)
        self.explained_variance_ = eigenvalues[:n_components]
        self.explained_variance_ratio_ = eigenvalues[:n_components] / torch.sum(eigenvalues)
        self.n_components_ = n_components
        
        return self
    
    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """将数据投影到主成分上"""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        X = X.to(self.device)
        X_centered = X - self.mean_
        
        # 投影
        X_transformed = X_centered @ self.components_.T
        
        if self.whiten:
            X_transformed = X_transformed / torch.sqrt(self.explained_variance_)
        
        return X_transformed
    
    def fit_transform(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """拟合并转换"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_reduced: torch.Tensor) -> torch.Tensor:
        """将降维后的数据还原"""
        if self.whiten:
            X_reduced = X_reduced * torch.sqrt(self.explained_variance_)
        
        X_reconstructed = X_reduced @ self.components_
        X_reconstructed = X_reconstructed + self.mean_
        
        return X_reconstructed
    
    def to_numpy(self) -> 'PCATorch':
        """将结果转换为NumPy数组（方便与sklearn兼容）"""
        if self.components_ is not None:
            self.components_ = self.components_.cpu().numpy()
            self.explained_variance_ = self.explained_variance_.cpu().numpy()
            self.explained_variance_ratio_ = self.explained_variance_ratio_.cpu().numpy()
            self.mean_ = self.mean_.cpu().numpy()
        return self


# ============================================================================
# 第二部分：可微分PCA（神经网络层）
# ============================================================================

class PCALayer(nn.Module):
    """
    可微分PCA层，可以作为神经网络的一部分
    
    与普通PCA不同，这个层在训练过程中会更新，
    并且支持反向传播（backpropagation）。
    
    应用场景：
    - 作为降维的预处理层
    - 学习最优的低维表示
    - 端到端的降维+分类联合训练
    
    示例:
        >>> model = nn.Sequential(
        ...     PCALayer(n_components=50),
        ...     nn.Linear(50, 10)
        ... )
    """
    
    def __init__(self, n_components: int, n_features: int):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        
        # 初始化一个随机投影矩阵
        # 在实际使用前需要先fit()
        self.register_buffer('components', torch.randn(n_components, n_features))
        self.register_buffer('mean', torch.zeros(n_features))
        self.register_buffer('explained_variance', torch.ones(n_components))
        
        self.fitted = False
    
    @torch.no_grad()
    def fit(self, X: torch.Tensor):
        """
        计算主成分（不可微）
        
        这通常在训练前做一次，之后组件固定。
        """
        n_samples = X.shape[0]
        
        # 中心化
        mean = torch.mean(X, dim=0)
        X_centered = X - mean
        
        # SVD分解（比特征值分解更稳定）
        # X = U * S * V^T
        # 主成分就是V的前几列
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        
        # 保存结果
        self.mean.copy_(mean)
        self.components.copy_(Vt[:self.n_components])
        
        # 解释的方差
        explained_variance = (S ** 2) / (n_samples - 1)
        self.explained_variance.copy_(explained_variance[:self.n_components])
        
        self.fitted = True
        
        return self
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        前向传播（可微）
        
        注意：虽然fit()使用SVD，但forward使用固定的投影矩阵，
        这个操作是可微的！
        """
        if not self.fitted:
            raise RuntimeError("PCALayer需要先调用fit()")
        
        X_centered = X - self.mean
        return X_centered @ self.components.T


# ============================================================================
# 第三部分：PyTorch版t-SNE
# ============================================================================

class TSNETorch:
    """
    基于PyTorch的t-SNE实现
    
    t-SNE是可视化高维数据的强大工具，但计算量很大。
    GPU加速可以显著加快这个过程。
    
    算法步骤：
    1. 在高维空间计算条件概率 P(j|i)
    2. 用对称SNE得到联合概率 P(i,j) = (P(j|i) + P(i|j)) / 2n
    3. 在低维空间随机初始化点 Y
    4. 迭代优化：计算Q分布，计算KL散度梯度，更新Y
    
    参数:
        n_components: 降维后的维度（通常是2或3）
        perplexity: 困惑度，通常在5-50之间
        learning_rate: 学习率
        n_iter: 迭代次数
        early_exaggeration: 早期放大因子
    """
    
    def __init__(self, n_components: int = 2, perplexity: float = 30.0,
                 learning_rate: float = 200.0, n_iter: int = 1000,
                 early_exaggeration: float = 12.0, early_exaggeration_iter: int = 250,
                 device: Optional[str] = None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.embedding_ = None
        
    def _compute_pairwise_distances(self, X: torch.Tensor) -> torch.Tensor:
        """计算样本间的成对距离"""
        # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i · x_j
        sum_X = torch.sum(X ** 2, dim=1, keepdim=True)
        distances = sum_X + sum_X.T - 2 * X @ X.T
        return torch.clamp(distances, min=0)  # 避免数值误差
    
    def _compute_gaussian_perplexity(self, distances: torch.Tensor, 
                                      perplexity: float, tol: float = 1e-5,
                                      max_iter: int = 50) -> torch.Tensor:
        """
        计算高斯分布下的条件概率P(j|i)
        
        使用二分搜索找到合适的sigma值，使得困惑度等于目标值。
        """
        n_samples = distances.shape[0]
        P = torch.zeros((n_samples, n_samples), device=self.device)
        
        # 目标熵
        target_entropy = np.log(perplexity)
        
        for i in range(n_samples):
            # 二分搜索beta = 1 / (2 * sigma^2)
            beta_min = -np.inf
            beta_max = np.inf
            beta = 1.0
            
            # 当前样本到其他样本的距离
            Di = distances[i].clone()
            Di[i] = np.inf  # 排除自己
            
            for _ in range(max_iter):
                # 计算概率分布
                Pi = torch.exp(-Di * beta)
                Pi = Pi / torch.sum(Pi)
                
                # 计算熵
                # H = -sum(p * log(p))
                # 避免log(0)，只计算p>0的部分
                Pi_nonzero = Pi[Pi > 0]
                entropy = -torch.sum(Pi_nonzero * torch.log(Pi_nonzero))
                
                # 计算困惑度
                # 这里使用熵的差值
                entropy_diff = entropy - target_entropy
                
                if abs(entropy_diff) < tol:
                    break
                
                # 二分搜索更新beta
                if entropy_diff > 0:
                    # 熵太大，需要减小sigma（增大beta）
                    beta_min = beta
                    if beta_max == np.inf:
                        beta *= 2
                    else:
                        beta = (beta + beta_max) / 2
                else:
                    # 熵太小，需要增大sigma（减小beta）
                    beta_max = beta
                    if beta_min == -np.inf:
                        beta /= 2
                    else:
                        beta = (beta + beta_min) / 2
            
            # 保存最终结果
            P[i] = torch.exp(-Di * beta)
            P[i, i] = 0  # p(i|i) = 0
            P[i] = P[i] / torch.sum(P[i])
        
        return P
    
    def fit_transform(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        拟合t-SNE并返回降维结果
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        X = X.to(self.device)
        n_samples = X.shape[0]
        
        print(f"t-SNE: 处理 {n_samples} 个样本...")
        
        # 步骤1: 计算高维空间的条件概率
        print("  计算高维空间概率分布...")
        distances = self._compute_pairwise_distances(X)
        P = self._compute_gaussian_perplexity(distances, self.perplexity)
        
        # 对称化: P(i,j) = (P(j|i) + P(i|j)) / 2n
        P = (P + P.T) / (2 * n_samples)
        
        # 早期放大
        P = P * self.early_exaggeration
        
        # 步骤2: 初始化低维表示
        # 使用小的随机值初始化
        Y = torch.randn(n_samples, self.n_components, device=self.device) * 0.0001
        Y.requires_grad = True
        
        # 优化器
        optimizer = torch.optim.Adam([Y], lr=self.learning_rate)
        
        # 步骤3: 梯度下降优化
        print("  优化低维嵌入...")
        for iter in range(self.n_iter):
            optimizer.zero_grad()
            
            # 计算低维空间的Q分布（t分布）
            sum_Y = torch.sum(Y ** 2, dim=1, keepdim=True)
            num = 1 / (1 + sum_Y + sum_Y.T - 2 * Y @ Y.T)  # t分布的分子
            num.fill_diagonal_(0)  # q(i,i) = 0
            Q = num / torch.sum(num)
            
            # 计算KL散度
            # KL(P||Q) = sum(P * log(P/Q))
            # 只考虑P和Q都大于0的位置
            mask = (P > 1e-10) & (Q > 1e-10)
            loss = torch.sum(P[mask] * torch.log(P[mask] / Q[mask]))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 取消早期放大
            if iter == self.early_exaggeration_iter:
                P = P / self.early_exaggeration
            
            # 打印进度
            if (iter + 1) % 100 == 0 or iter == 0:
                print(f"    迭代 {iter + 1}/{self.n_iter}, KL散度: {loss.item():.4f}")
        
        self.embedding_ = Y.detach().cpu().numpy()
        return self.embedding_


# ============================================================================
# 第四部分：Autoencoder降维（深度学习）
# ============================================================================

class Autoencoder(nn.Module):
    """
    自编码器用于非线性降维
    
    自编码器是一种神经网络，它学习将输入压缩到低维空间，
    然后再重构回原始空间。编码器部分就是我们的降维工具。
    
    与PCA的关系：
    - 如果自编码器只有线性层且没有激活函数，它等价于PCA
    - 加入非线性激活函数后，它可以学习更复杂的降维映射
    
    结构：
        输入(784) → 编码器 → 瓶颈(2) → 解码器 → 输出(784)
        
    生活比喻：
    - 编码器就像把一本书的要点总结成一句话
    - 解码器就像从这句话尽可能还原出原书
    - 自编码器学习的是：如何总结才能最好地还原！
    """
    
    def __init__(self, input_dim: int, latent_dim: int, 
                 hidden_dims: Optional[list] = None):
        super().__init__()
        
        if hidden_dims is None:
            # 默认结构：输入 → 中间 → 瓶颈
            hidden_dims = [512, 256, 128]
        
        # 构建编码器
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 构建解码器（对称结构）
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  # 假设输入归一化到[0,1]
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码：降维"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """解码：重构"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播：编码+解码"""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z
    
    def fit(self, X: torch.Tensor, epochs: int = 100, 
            batch_size: int = 256, learning_rate: float = 1e-3,
            verbose: bool = True):
        """
        训练自编码器
        
        参数:
            X: 训练数据
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            verbose: 是否打印进度
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]
                
                optimizer.zero_grad()
                x_reconstructed, _ = self.forward(x)
                loss = criterion(x_reconstructed, x)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.6f}")
        
        self.eval()
        return self
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """降维（使用编码器）"""
        self.eval()
        with torch.no_grad():
            return self.encode(X)


# ============================================================================
# 第五部分：实用工具函数
# ============================================================================

def compare_pca_methods(X: np.ndarray, n_components: int = 2) -> plt.Figure:
    """
    对比NumPy PCA和PyTorch PCA的结果
    """
    from sklearn.decomposition import PCA as SklearnPCA
    
    # NumPy/Sklearn PCA
    sklearn_pca = SklearnPCA(n_components=n_components)
    X_sklearn = sklearn_pca.fit_transform(X)
    
    # PyTorch PCA
    torch_pca = PCATorch(n_components=n_components)
    X_torch = torch_pca.fit_transform(X).cpu().numpy()
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(X_sklearn[:, 0], X_sklearn[:, 1], alpha=0.6)
    axes[0].set_title('Sklearn PCA')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(X_torch[:, 0], X_torch[:, 1], alpha=0.6)
    axes[1].set_title('PyTorch PCA')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def benchmark_pca_speed(X: np.ndarray, n_components: int = 50) -> dict:
    """
    对比CPU和GPU的PCA计算速度
    """
    import time
    
    results = {}
    
    # NumPy/CPU
    from sklearn.decomposition import PCA as SklearnPCA
    t0 = time.time()
    pca_cpu = SklearnPCA(n_components=n_components)
    X_cpu = pca_cpu.fit_transform(X)
    results['sklearn_cpu'] = time.time() - t0
    
    # PyTorch/CPU
    torch_cpu = PCATorch(n_components=n_components, device='cpu')
    t0 = time.time()
    X_torch_cpu = torch_cpu.fit_transform(X)
    results['torch_cpu'] = time.time() - t0
    
    # PyTorch/GPU (如果可用)
    if torch.cuda.is_available():
        torch_gpu = PCATorch(n_components=n_components, device='cuda')
        X_tensor = torch.from_numpy(X).float()
        
        # 预热GPU
        torch.cuda.synchronize()
        t0 = time.time()
        X_torch_gpu = torch_gpu.fit_transform(X_tensor)
        torch.cuda.synchronize()
        results['torch_gpu'] = time.time() - t0
        
        results['speedup'] = results['torch_cpu'] / results['torch_gpu']
    
    return results


# ============================================================================
# 演示代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch版PCA和t-SNE演示")
    print("=" * 60)
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 5000
    n_features = 100
    
    # 创建有结构的数据
    X = np.random.randn(n_samples, n_features)
    for i in range(10):
        X[:, i] *= (10 - i)
    
    print(f"\n数据: {X.shape}")
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # 基准测试
    print("\n" + "-" * 40)
    print("速度对比测试")
    print("-" * 40)
    results = benchmark_pca_speed(X[:1000], n_components=10)  # 用子集测试
    for method, time_taken in results.items():
        if method != 'speedup':
            print(f"{method}: {time_taken:.4f}秒")
    if 'speedup' in results:
        print(f"GPU加速比: {results['speedup']:.2f}x")
    
    # PyTorch PCA演示
    print("\n" + "-" * 40)
    print("PyTorch PCA演示")
    print("-" * 40)
    pca_torch = PCATorch(n_components=2)
    X_pca = pca_torch.fit_transform(X).cpu().numpy()
    print(f"降维后: {X_pca.shape}")
    print(f"解释方差: {pca_torch.explained_variance_ratio_.cpu().numpy()}")
    
    # t-SNE演示
    print("\n" + "-" * 40)
    print("PyTorch t-SNE演示 (使用小数据集)")
    print("-" * 40)
    X_small = X[:200]  # t-SNE计算量大，用小样本
    tsne = TSNETorch(n_components=2, n_iter=300)
    X_tsne = tsne.fit_transform(X_small)
    print(f"t-SNE降维后: {X_tsne.shape}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
