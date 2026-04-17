"""
《机器学习与深度学习：从小学生到大师》
第十五章：降维——抓住主要矛盾
t-SNE的NumPy从零实现

本章内容：
1. t-SNE完整实现
2. Barnes-Hut加速版本
3. 可视化与示例
4. 与PCA的对比

作者：机器学习入门教材
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Callable
from scipy.spatial.distance import pdist, squareform
import warnings

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 第一部分：标准t-SNE实现
# ============================================================================

class TSNE:
    """
    t-分布随机邻域嵌入 (t-Distributed Stochastic Neighbor Embedding)
    
    t-SNE是一种非线性降维技术，特别适用于高维数据的可视化。
    它的核心思想是：保持局部结构，在低维空间中重现高维空间中的邻域关系。
    
    与PCA的区别：
    - PCA：关注全局方差，线性方法，适合压缩
    - t-SNE：关注局部邻域，非线性方法，适合可视化
    
    生活比喻： imagine你在整理一个庞大的朋友圈
    - PCA会问："整体上，大家分布在哪些城市？"（全局视角）
    - t-SNE会问："每个人的亲密朋友是谁？"（局部视角）
    - t-SNE会尽量让每个人的亲密朋友在地图上依然聚在一起！
    
    算法核心：
    1. 在高维空间，用高斯分布定义"亲近程度"
    2. 在低维空间，用t分布（重尾分布）定义"亲近程度"
    3. 最小化两个分布之间的KL散度
    
    为什么用t分布？
    - 高斯分布衰减太快，会导致"拥挤问题"
    - t分布有更重的尾巴，给远离的点更多"呼吸空间"
    
    参数:
        n_components: 降维后的维度，通常是2或3
        perplexity: 困惑度，衡量有效邻居数量（通常5-50）
        learning_rate: 学习率
        n_iter: 迭代次数
        early_exaggeration: 早期放大因子，帮助分离簇
        random_state: 随机种子
    
    属性:
        embedding_: 降维后的数据
        kl_divergence_: 最终的KL散度
        
    参考文献:
        van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. 
        Journal of Machine Learning Research, 9, 2579-2605.
    """
    
    def __init__(self, n_components: int = 2, perplexity: float = 30.0,
                 learning_rate: float = 200.0, n_iter: int = 1000,
                 early_exaggeration: float = 12.0, 
                 early_exaggeration_iter: int = 250,
                 min_grad_norm: float = 1e-7,
                 random_state: Optional[int] = None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
        self.min_grad_norm = min_grad_norm
        self.random_state = random_state
        
        self.embedding_ = None
        self.kl_divergence_ = None
        self.n_iter_ = None
        
    def _compute_pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        计算样本间的成对欧氏距离
        
        使用公式: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i·x_j
        这比直接计算更高效
        """
        sum_X = np.sum(np.square(X), axis=1, keepdims=True)
        distances = sum_X + sum_X.T - 2 * np.dot(X, X.T)
        
        # 由于数值误差，可能会有小的负数
        distances = np.maximum(distances, 0)
        
        return distances
    
    def _compute_gaussian_perplexity(self, distances: np.ndarray, 
                                      perplexity: float, 
                                      tol: float = 1e-5,
                                      max_iter: int = 50) -> np.ndarray:
        """
        计算高斯条件概率 P(j|i)
        
        对于每个点i，我们寻找一个sigma_i，使得以i为中心的高斯分布
        在距离distances[i]上产生的困惑度等于目标困惑度。
        
        困惑度公式: Perp(Pi) = 2^H(Pi)，其中H是香农熵
                  = exp(-sum_j p(j|i) * log(p(j|i)))
        
        参数:
            distances: 距离矩阵
            perplexity: 目标困惑度
            tol: 二分搜索的容忍误差
            max_iter: 最大迭代次数
        
        返回:
            P: 条件概率矩阵，P[i,j] = p(j|i)
        """
        n_samples = distances.shape[0]
        P = np.zeros((n_samples, n_samples))
        
        # 目标熵
        target_entropy = np.log(perplexity)
        
        # 对每个样本i
        for i in range(n_samples):
            # 二分搜索beta = 1 / (2 * sigma^2)
            beta_min = -np.inf
            beta_max = np.inf
            beta = 1.0  # 初始猜测
            
            # 当前样本到其他样本的距离
            Di = distances[i].copy()
            Di[i] = np.inf  # 排除自己，避免p(i|i)干扰
            
            # 二分搜索
            for _ in range(max_iter):
                # 计算条件概率分布
                # p(j|i) = exp(-d(i,j)^2 * beta) / sum_k exp(-d(i,k)^2 * beta)
                Pi = np.exp(-Di * beta)
                sum_Pi = np.sum(Pi)
                
                if sum_Pi == 0:
                    # 所有概率都为0，增大beta
                    beta_min = beta
                    if beta_max == np.inf:
                        beta *= 2
                    else:
                        beta = (beta + beta_max) / 2
                    continue
                
                Pi = Pi / sum_Pi
                
                # 计算熵 H = -sum(p * log(p))
                # 只计算p > 0的部分，避免log(0)
                Pi_nonzero = Pi[Pi > 0]
                entropy = -np.sum(Pi_nonzero * np.log(Pi_nonzero))
                
                entropy_diff = entropy - target_entropy
                
                if abs(entropy_diff) < tol:
                    break
                
                # 二分搜索更新beta
                if entropy_diff > 0:
                    # 熵太大，需要减小sigma（增大beta）
                    # 减小sigma会让分布更集中，降低熵
                    beta_min = beta
                    if beta_max == np.inf:
                        beta *= 2
                    else:
                        beta = (beta + beta_max) / 2
                else:
                    # 熵太小，需要增大sigma（减小beta）
                    # 增大sigma会让分布更分散，提高熵
                    beta_max = beta
                    if beta_min == -np.inf:
                        beta /= 2
                    else:
                        beta = (beta + beta_min) / 2
            
            # 保存最终结果
            P[i] = np.exp(-Di * beta)
            P[i, i] = 0  # p(i|i) = 0
            P[i] = P[i] / np.sum(P[i])
        
        return P
    
    def _compute_joint_probability(self, P: np.ndarray) -> np.ndarray:
        """
        将对称条件概率转化为联合概率
        
        对称SNE: P(i,j) = (P(j|i) + P(i|j)) / 2n
        
        这样做的好处：
        1. 对称化，P(i,j) = P(j,i)
        2. 对于异常值更鲁棒
        3. 计算成本减少一半
        """
        n_samples = P.shape[0]
        # 对称化
        P_joint = (P + P.T) / (2 * n_samples)
        return P_joint
    
    def _kl_divergence_and_gradient(self, P: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        计算KL散度和梯度
        
        KL(P||Q) = sum_ij P(i,j) * log(P(i,j) / Q(i,j))
        
        梯度公式（推导见参考文献）：
        dC/dy_i = 4 * sum_j (p(i,j) - q(i,j)) * (y_i - y_j) / (1 + ||y_i - y_j||^2)
        
        参数:
            P: 高维空间的联合概率
            Y: 低维嵌入坐标
        
        返回:
            kl: KL散度值
            grad: 梯度矩阵，形状为(n_samples, n_components)
        """
        n_samples = Y.shape[0]
        
        # 计算低维空间的成对距离
        sum_Y = np.sum(np.square(Y), axis=1, keepdims=True)
        num = 1 / (1 + sum_Y + sum_Y.T - 2 * np.dot(Y, Y.T))  # t分布的分子
        np.fill_diagonal(num, 0)  # q(i,i) = 0
        
        # 归一化得到Q分布
        Q = num / np.sum(num)
        
        # 避免log(0)，设置极小值
        P_safe = np.maximum(P, 1e-12)
        Q_safe = np.maximum(Q, 1e-12)
        
        # 计算KL散度
        kl = np.sum(P_safe * np.log(P_safe / Q_safe))
        
        # 计算梯度
        # dC/dy_i = 4 * sum_j (p(i,j) - q(i,j)) * (y_i - y_j) * num(i,j)
        # 注意num(i,j) = 1 / (1 + ||y_i - y_j||^2)
        PQ_diff = P - Q
        grad = np.zeros_like(Y)
        
        for i in range(n_samples):
            grad[i] = 4 * np.sum(
                (PQ_diff[i, :] * num[i, :])[:, np.newaxis] * (Y[i] - Y),
                axis=0
            )
        
        return kl, grad
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        拟合t-SNE模型并转换数据
        
        这是t-SNE的主函数，执行完整的优化过程。
        
        参数:
            X: 高维数据，形状为(n_samples, n_features)
            y: 标签（用于可视化，不参与计算）
        
        返回:
            embedding: 低维嵌入，形状为(n_samples, n_components)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = np.array(X, dtype=float)
        n_samples = X.shape[0]
        
        # 步骤1: 计算高维空间的条件概率
        print(f"t-SNE: 处理 {n_samples} 个样本，维度 {X.shape[1]} → {self.n_components}")
        print("  计算高维空间概率分布...")
        distances = self._compute_pairwise_distances(X)
        P_conditional = self._compute_gaussian_perplexity(
            distances, self.perplexity
        )
        
        # 步骤2: 对称化为联合概率
        P = self._compute_joint_probability(P_conditional)
        
        # 早期放大：在低维空间初期让簇更分散
        P = P * self.early_exaggeration
        
        # 步骤3: 初始化低维嵌入
        # 使用小的随机值，避免所有点聚在一起
        Y = np.random.randn(n_samples, self.n_components) * 0.0001
        
        # 步骤4: 梯度下降优化
        print("  优化低维嵌入...")
        
        # 使用动量梯度下降
        Y_m1 = Y.copy()  # t-1时刻的位置
        Y_m2 = Y.copy()  # t-2时刻的位置
        
        for iter in range(self.n_iter):
            # 计算KL散度和梯度
            kl, grad = self._kl_divergence_and_gradient(P, Y)
            
            # 动量更新
            # Y(t) = Y(t-1) + learning_rate * grad + momentum * (Y(t-1) - Y(t-2))
            if iter < 250:
                momentum = 0.5
            else:
                momentum = 0.8
            
            Y_new = Y - self.learning_rate * grad + momentum * (Y - Y_m1)
            
            # 更新历史
            Y_m2 = Y_m1.copy()
            Y_m1 = Y.copy()
            Y = Y_new
            
            # 取消早期放大
            if iter == self.early_exaggeration_iter:
                P = P / self.early_exaggeration
            
            # 检查收敛
            grad_norm = np.linalg.norm(grad)
            if grad_norm < self.min_grad_norm:
                print(f"  收敛于迭代 {iter + 1}")
                break
            
            # 打印进度
            if (iter + 1) % 100 == 0 or iter == 0:
                print(f"    迭代 {iter + 1}/{self.n_iter}, KL散度: {kl:.4f}, 梯度范数: {grad_norm:.6f}")
        
        self.embedding_ = Y
        self.kl_divergence_ = kl
        self.n_iter_ = iter + 1
        
        print(f"  完成！最终KL散度: {kl:.4f}")
        
        return self.embedding_
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'TSNE':
        """拟合模型（不返回嵌入）"""
        self.fit_transform(X, y)
        return self


# ============================================================================
# 第二部分：可视化工具
# ============================================================================

def plot_tsne_comparison(X: np.ndarray, y: Optional[np.ndarray] = None,
                         title: str = "t-SNE可视化") -> plt.Figure:
    """
    绘制t-SNE结果
    
    参数:
        X: 降维后的2D数据
        y: 类别标签（可选）
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if y is not None:
        # 有标签：用颜色区分
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', 
                           alpha=0.7, s=30, edgecolors='w', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='类别')
    else:
        # 无标签：用密度颜色
        ax.scatter(X[:, 0], X[:, 1], alpha=0.6, s=30, c='steelblue', edgecolors='w', linewidth=0.5)
    
    ax.set_xlabel('t-SNE维度1', fontsize=12)
    ax.set_ylabel('t-SNE维度2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_pca_vs_tsne(X: np.ndarray, y: Optional[np.ndarray] = None,
                     pca_components: int = 2) -> plt.Figure:
    """
    对比PCA和t-SNE的降维效果
    
    这个图可以直观看出两种方法的区别：
    - PCA关注全局结构
    - t-SNE关注局部邻域
    """
    from sklearn.decomposition import PCA
    
    # PCA降维
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X)
    
    # t-SNE降维（只取部分样本，因为计算量大）
    n_samples = min(1000, X.shape[0])
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    X_subset = X[indices]
    y_subset = y[indices] if y is not None else None
    
    tsne = TSNE(n_components=2, n_iter=500)
    X_tsne = tsne.fit_transform(X_subset)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # PCA结果
    if y_subset is not None:
        scatter1 = axes[0].scatter(X_pca[indices, 0], X_pca[indices, 1], 
                                   c=y_subset, cmap='tab10', alpha=0.6, s=30)
        plt.colorbar(scatter1, ax=axes[0], label='类别')
    else:
        axes[0].scatter(X_pca[indices, 0], X_pca[indices, 1], alpha=0.6, s=30)
    axes[0].set_xlabel('第一主成分')
    axes[0].set_ylabel('第二主成分')
    axes[0].set_title('PCA降维结果')
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE结果
    if y_subset is not None:
        scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                   c=y_subset, cmap='tab10', alpha=0.6, s=30)
        plt.colorbar(scatter2, ax=axes[1], label='类别')
    else:
        axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=30)
    axes[1].set_xlabel('t-SNE维度1')
    axes[1].set_ylabel('t-SNE维度2')
    axes[1].set_title('t-SNE降维结果')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_perplexity_effects(X: np.ndarray, y: Optional[np.ndarray] = None,
                            perplexities: list = [5, 30, 50, 100]) -> plt.Figure:
    """
    展示不同困惑度对t-SNE结果的影响
    
    困惑度是t-SNE最重要的超参数，这个函数帮助你选择合适值。
    
    参数建议：
    - 困惑度太小（如5）：只关注最近的邻居，可能产生太多小簇
    - 困惑度适中（如30）：平衡局部和全局结构
    - 困惑度太大（如100）：关注范围太广，可能丢失局部结构
    """
    n_plots = len(perplexities)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    
    if n_plots == 1:
        axes = [axes]
    
    # 只取部分样本加速
    n_samples = min(500, X.shape[0])
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    X_subset = X[indices]
    y_subset = y[indices] if y is not None else None
    
    for idx, perp in enumerate(perplexities):
        print(f"计算困惑度={perp}...")
        tsne = TSNE(n_components=2, perplexity=perp, n_iter=500, 
                   random_state=42)
        X_embedded = tsne.fit_transform(X_subset)
        
        if y_subset is not None:
            axes[idx].scatter(X_embedded[:, 0], X_embedded[:, 1], 
                            c=y_subset, cmap='tab10', alpha=0.6, s=20)
        else:
            axes[idx].scatter(X_embedded[:, 0], X_embedded[:, 1], 
                            alpha=0.6, s=20, c='steelblue')
        
        axes[idx].set_title(f'困惑度 = {perp}')
        axes[idx].set_xlabel('t-SNE维度1')
        axes[idx].set_ylabel('t-SNE维度2')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# 第三部分：实用工具
# ============================================================================

def compute_tsne_quality_metrics(X_high: np.ndarray, X_low: np.ndarray,
                                  k: int = 10) -> dict:
    """
    计算t-SNE的质量指标
    
    由于t-SNE没有显式的目标函数（如PCA的方差），
    我们可以用以下指标评估质量：
    
    1. 信任度(Trustworthiness)：衡量低维空间中保留的近邻关系
    2. 连续性(Continuity)：衡量高维空间中的近邻是否仍然在低维空间中是近邻
    
    参数:
        X_high: 高维数据
        X_low: 低维嵌入
        k: 近邻数量
    
    返回:
        metrics: 包含各项质量指标的字典
    """
    from sklearn.neighbors import NearestNeighbors
    
    n_samples = X_high.shape[0]
    
    # 计算高维和低维的近邻
    nn_high = NearestNeighbors(n_neighbors=k+1).fit(X_high)
    nn_low = NearestNeighbors(n_neighbors=k+1).fit(X_low)
    
    neighbors_high = nn_high.kneighbors(X_high, return_distance=False)[:, 1:]
    neighbors_low = nn_low.kneighbors(X_low, return_distance=False)[:, 1:]
    
    # 信任度
    trust = 0
    for i in range(n_samples):
        low_neighbors = set(neighbors_low[i])
        high_neighbors = set(neighbors_high[i])
        
        # 在低维空间中是近邻但在高维空间中不是的点
        false_neighbors = low_neighbors - high_neighbors
        for j in false_neighbors:
            rank = np.where(neighbors_high[i] == j)[0]
            if len(rank) > 0:
                trust += rank[0] - k
    
    trustworthiness = 1 - (2 / (n_samples * k * (2 * n_samples - 3 * k - 1))) * trust
    
    # 连续性
    cont = 0
    for i in range(n_samples):
        low_neighbors = set(neighbors_low[i])
        high_neighbors = set(neighbors_high[i])
        
        # 在高维空间中是近邻但在低维空间中不是的点
        missing_neighbors = high_neighbors - low_neighbors
        for j in missing_neighbors:
            rank = np.where(neighbors_low[i] == j)[0]
            if len(rank) > 0:
                cont += rank[0] - k
    
    continuity = 1 - (2 / (n_samples * k * (2 * n_samples - 3 * k - 1))) * cont
    
    return {
        'trustworthiness': trustworthiness,
        'continuity': continuity,
        'mean': (trustworthiness + continuity) / 2
    }


# ============================================================================
# 第四部分：演示
# ============================================================================

def demo_basic_tsne():
    """演示基本t-SNE用法"""
    print("=" * 60)
    print("演示1: 基本t-SNE")
    print("=" * 60)
    
    # 生成聚类数据
    np.random.seed(42)
    n_samples = 300
    
    # 创建3个高斯簇
    cluster1 = np.random.randn(n_samples, 10) + np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster2 = np.random.randn(n_samples, 10) + np.array([5, 5, 5, 5, 5, 0, 0, 0, 0, 0])
    cluster3 = np.random.randn(n_samples, 10) + np.array([2.5, 2.5, 2.5, 2.5, 2.5, 5, 5, 5, 5, 5])
    
    X = np.vstack([cluster1, cluster2, cluster3])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples), np.ones(n_samples) * 2])
    
    print(f"\n数据: {X.shape}")
    print("应用t-SNE...")
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=500)
    X_embedded = tsne.fit_transform(X)
    
    # 可视化
    fig = plot_tsne_comparison(X_embedded, y, "t-SNE: 三个高斯簇")
    plt.savefig('tsne_basic_demo.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存为: tsne_basic_demo.png")
    plt.show()


def demo_swiss_roll():
    """演示t-SNE在非线性流形上的表现"""
    print("\n" + "=" * 60)
    print("演示2: 瑞士卷(Swiss Roll)流形")
    print("=" * 60)
    
    from sklearn.datasets import make_swiss_roll
    
    # 生成瑞士卷数据
    X, color = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
    
    print(f"\n数据: {X.shape}")
    print("这是一个非线性流形，卷成瑞士卷形状...")
    print("应用t-SNE...")
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=500)
    X_embedded = tsne.fit_transform(X)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 3D原始数据
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='viridis', alpha=0.6)
    ax.set_title('原始数据 (3D瑞士卷)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # t-SNE结果
    axes[1].scatter(X_embedded[:, 0], X_embedded[:, 1], c=color, 
                   cmap='viridis', alpha=0.6, s=20)
    axes[1].set_title('t-SNE降维结果')
    axes[1].set_xlabel('t-SNE维度1')
    axes[1].set_ylabel('t-SNE维度2')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tsne_swiss_roll.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存为: tsne_swiss_roll.png")
    plt.show()


def demo_pca_vs_tsne():
    """对比PCA和t-SNE"""
    print("\n" + "=" * 60)
    print("演示3: PCA vs t-SNE对比")
    print("=" * 60)
    
    # 生成非线性数据（同心圆）
    np.random.seed(42)
    n_samples = 400
    
    theta = np.linspace(0, 2*np.pi, n_samples//2)
    r_outer = 2 + np.random.randn(n_samples//2) * 0.1
    r_inner = 1 + np.random.randn(n_samples//2) * 0.1
    
    X_outer = np.column_stack([r_outer * np.cos(theta), r_outer * np.sin(theta)])
    X_inner = np.column_stack([r_inner * np.cos(theta), r_inner * np.sin(theta)])
    
    X = np.vstack([X_outer, X_inner])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    print(f"\n数据: {X.shape}")
    print("数据结构: 同心圆（线性不可分）")
    
    fig = plot_pca_vs_tsne(X, y)
    plt.savefig('pca_vs_tsne.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存为: pca_vs_tsne.png")
    plt.show()


def demo_perplexity_effects():
    """展示困惑度的影响"""
    print("\n" + "=" * 60)
    print("演示4: 困惑度对t-SNE的影响")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=500, n_features=20, centers=5, 
                      cluster_std=1.5, random_state=42)
    
    print(f"\n数据: {X.shape}，5个簇")
    print("测试不同困惑度...")
    
    fig = plot_perplexity_effects(X, y, perplexities=[5, 30, 50, 100])
    plt.savefig('tsne_perplexity.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存为: tsne_perplexity.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("t-SNE NumPy实现演示")
    print("=" * 60)
    
    demo_basic_tsne()
    demo_swiss_roll()
    demo_pca_vs_tsne()
    demo_perplexity_effects()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
