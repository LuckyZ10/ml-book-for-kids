"""
《机器学习与深度学习：从小学生到大师》
第十五章：降维——抓住主要矛盾
从零实现PCA和t-SNE

本章内容：
1. PCA (主成分分析) 的完整实现
2. t-SNE (t-分布随机邻域嵌入) 的完整实现
3. 可视化函数
4. 6个演示示例

作者：机器学习入门教材
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import warnings

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 第一部分：PCA (主成分分析) 实现
# ============================================================================

class PCA:
    """
    主成分分析 (Principal Component Analysis)
    
    PCA是一种线性降维技术，通过寻找数据中方差最大的方向(主成分)，
    将高维数据投影到低维空间，同时保留尽可能多的信息。
    
    核心思想：抓住主要矛盾，忽略次要因素
    
    数学原理：
    1. 计算协方差矩阵：Σ = (X - μ)^T(X - μ) / (n - 1)
    2. 特征值分解：Σ = W·Λ·W^(-1)
    3. 选择前k个最大特征值对应的特征向量作为主成分
    4. 投影：Z = X·W_k
    
    参数:
        n_components: int或float，降维后的维度数或保留的方差比例
        whiten: bool，是否进行白化（使各成分方差相等）
    
    属性:
        components_: 主成分向量，形状为(n_components, n_features)
        explained_variance_: 每个主成分解释的方差值
        explained_variance_ratio_: 每个主成分解释的方差比例
        mean_: 数据的均值向量
    """
    
    def __init__(self, n_components: Optional[int] = None, whiten: bool = False):
        """
        初始化PCA
        
        参数:
            n_components: 
                - None: 保留所有特征
                - int: 保留的主成分数量
                - float (0,1]: 保留的解释方差比例
            whiten: 是否进行白化处理
        """
        self.n_components = n_components
        self.whiten = whiten
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_samples_ = None
        self.n_features_ = None
        
    def fit(self, X: np.ndarray) -> 'PCA':
        """
        拟合PCA模型，计算主成分
        
        这是PCA的"学习"阶段：
        1. 中心化数据（减去均值）
        2. 计算协方差矩阵
        3. 特征值分解，得到特征值和特征向量
        4. 按特征值从大到小排序
        5. 选择前n_components个主成分
        
        参数:
            X: 训练数据，形状为(n_samples, n_features)
        
        返回:
            self: 拟合后的PCA对象
        """
        # 转换为numpy数组
        X = np.array(X, dtype=float)
        
        # 保存数据维度信息
        self.n_samples_, self.n_features_ = X.shape
        
        # 步骤1: 中心化数据
        # 为什么要中心化？因为PCA要找的是数据变化的方向，不是绝对位置
        # 中心化使均值为0，方便计算协方差
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 步骤2: 计算协方差矩阵
        # 协方差矩阵 Σ[i,j] = Cov(x_i, x_j)
        # 表示第i维和第j维数据之间的协方差
        # 除以(n_samples - 1)是为了得到无偏估计
        cov_matrix = np.dot(X_centered.T, X_centered) / (self.n_samples_ - 1)
        
        # 步骤3: 特征值分解
        # Σ = W·Λ·W^(-1)
        # eigenvalues: 特征值，表示每个主成分的重要性
        # eigenvectors: 特征向量，表示主成分的方向
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 确保是实数（处理数值误差）
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # 步骤4: 按特征值从大到小排序
        # 特征值越大，对应的主成分包含的信息越多
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 步骤5: 确定保留的主成分数量
        if self.n_components is None:
            # 默认保留所有特征
            n_components = self.n_features_
        elif isinstance(self.n_components, float):
            # 根据方差比例确定保留数量
            # 例如 n_components=0.95 表示保留95%的方差
            cumulative_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            n_components = np.searchsorted(cumulative_ratio, self.n_components) + 1
        else:
            n_components = self.n_components
        
        # 确保不超过特征数
        n_components = min(n_components, self.n_features_)
        
        # 步骤6: 提取主成分
        # components_[i]是第i个主成分的方向向量
        self.components_ = eigenvectors[:, :n_components].T
        
        # 计算解释方差
        self.explained_variance_ = eigenvalues[:n_components]
        
        # 计算解释方差比例
        # 表示每个主成分解释了原始数据多少比例的方差
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        将数据投影到主成分空间（降维）
        
        这是PCA的"应用"阶段：
        1. 中心化数据（使用训练时的均值）
        2. 投影到选定的主成分上
        
        参数:
            X: 输入数据，形状为(n_samples, n_features)
        
        返回:
            X_transformed: 降维后的数据，形状为(n_samples, n_components)
        """
        X = np.array(X, dtype=float)
        
        # 中心化
        X_centered = X - self.mean_
        
        # 投影到主成分空间
        # Z = X·W
        # 每个样本的坐标 = 样本向量与主成分向量的点积
        X_transformed = np.dot(X_centered, self.components_.T)
        
        # 白化处理（可选）
        # 使各主成分的方差相等，消除尺度差异
        if self.whiten:
            X_transformed = X_transformed / np.sqrt(self.explained_variance_)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        拟合模型并转换数据（便捷方法）
        
        相当于 fit(X).transform(X)
        
        参数:
            X: 输入数据
        
        返回:
            X_transformed: 降维后的数据
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        将降维后的数据还原回原始空间（近似）
        
        注意：这是有损还原，因为我们丢弃了一些信息
        
        参数:
            X_transformed: 降维后的数据
        
        返回:
            X_approx: 近似还原的原始维度数据
        """
        # 如果是白化后的数据，先恢复
        if self.whiten:
            X_transformed = X_transformed * np.sqrt(self.explained_variance_)
        
        # 反向投影：X_approx = Z·W^T + μ
        X_approx = np.dot(X_transformed, self.components_) + self.mean_
        return X_approx
    
    def get_feature_importance(self) -> np.ndarray:
        """
        获取各原始特征的重要性（基于主成分载荷）
        
        返回:
            importance: 各特征的重要性分数
        """
        if self.components_ is None:
            raise ValueError("请先调用fit()方法拟合模型")
        
        # 特征重要性 = 各主成分上该特征的载荷绝对值之和
        importance = np.sum(np.abs(self.components_), axis=0)
        return importance / np.sum(importance)


# ============================================================================
# 第二部分：t-SNE 实现
# ============================================================================

class TSNE:
    """
    t-分布随机邻域嵌入 (t-distributed Stochastic Neighbor Embedding)
    
    t-SNE是一种非线性降维技术，特别适用于高维数据的可视化。
    它通过保持数据点之间的"邻居关系"来构造低维表示。
    
    核心思想：
    1. 在高维空间中，用高斯分布定义点对的相似度（条件概率）
    2. 在低维空间中，用t分布（自由度为1）定义相似度
    3. 用KL散度衡量两个分布的差异
    4. 用梯度下降最小化KL散度，学习低维嵌入
    
    为什么选择t分布？
    t分布比高斯分布有"更重的尾巴"，可以缓解"拥挤问题"：
    在低维空间中，有更多空间容纳中等距离的点
    
    参数:
        n_components: 降维后的维度（通常2或3，用于可视化）
        perplexity: 困惑度，类比为每个点的"有效邻居数"
        learning_rate: 学习率
        n_iter: 迭代次数
        early_exaggeration: 早期放大因子（加强吸引力）
        random_state: 随机种子
    """
    
    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        early_exaggeration: float = 12.0,
        random_state: Optional[int] = None
    ):
        """
        初始化t-SNE
        
        参数:
            n_components: 降维后的维度（默认2）
            perplexity: 困惑度，通常在5-50之间（默认30）
                        较小的perplexity关注局部结构
                        较大的perplexity关注全局结构
            learning_rate: 学习率（默认200）
            n_iter: 迭代次数（默认1000）
            early_exaggeration: 早期放大因子（默认12）
            random_state: 随机种子
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.random_state = random_state
        
        # 拟合后的属性
        self.embedding_ = None
        self.kl_divergence_ = None
        self.n_samples_ = None
        
    def _compute_pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        计算样本间的高维距离矩阵
        
        D[i,j] = ||x_i - x_j||^2
        
        使用向量化计算提高效率
        """
        # 方法：||a - b||^2 = ||a||^2 + ||b||^2 - 2a·b
        sum_X = np.sum(np.square(X), axis=1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return np.maximum(D, 0)  # 确保非负（处理数值误差）
    
    def _binary_search_perplexity(
        self, 
        distances: np.ndarray, 
        i: int,
        tol: float = 1e-5,
        max_iter: int = 50
    ) -> Tuple[np.ndarray, float]:
        """
        二分搜索确定高斯分布的sigma值，使得困惑度等于目标值
        
        困惑度定义为：Perp(P_i) = 2^H(P_i)
        其中H是香农熵：H(P_i) = -Σ_j p(j|i) log_2 p(j|i)
        
        参数:
            distances: 距离矩阵
            i: 当前样本索引
            tol: 收敛容差
            max_iter: 最大迭代次数
        
        返回:
            P_i: 条件概率分布 p(j|i)
            sigma: 对应的高斯分布标准差
        """
        # 初始化二分搜索范围
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0  # 初始值
        
        Di = distances[i].copy()
        Di[i] = np.inf  # 排除自己
        
        for _ in range(max_iter):
            # 计算条件概率
            # p(j|i) = exp(-beta * ||x_i - x_j||^2) / Σ_k exp(-beta * ||x_i - x_k||^2)
            P = np.exp(-Di * beta)
            sum_P = np.sum(P)
            
            if sum_P == 0:
                # 数值问题，调整beta
                beta = beta / 2
                beta_min = beta
                continue
            
            # 计算熵
            # H = log(sum_P) + beta * sum(Di * P) / sum_P
            H = np.log(sum_P) + beta * np.sum(Di * P) / sum_P
            
            # 计算困惑度差异
            perplexity_diff = H - np.log(self.perplexity)
            
            # 检查收敛
            if np.abs(perplexity_diff) < tol:
                break
            
            # 二分搜索更新
            if perplexity_diff > 0:
                # 困惑度太大，需要增大beta（减小sigma）
                beta_min = beta
                if beta_max == np.inf:
                    beta = beta * 2
                else:
                    beta = (beta + beta_max) / 2
            else:
                # 困惑度太小，需要减小beta（增大sigma）
                beta_max = beta
                if beta_min == -np.inf:
                    beta = beta / 2
                else:
                    beta = (beta + beta_min) / 2
        
        # 计算最终的条件概率
        P = np.exp(-Di * beta)
        P[i] = 0  # p(i|i) = 0
        P = P / np.sum(P)
        
        sigma = np.sqrt(1 / (2 * beta))
        
        return P, sigma
    
    def _compute_high_dim_affinities(self, X: np.ndarray) -> np.ndarray:
        """
        计算高维空间中的联合概率分布P
        
        p(j|i) = exp(-||x_i-x_j||^2 / 2σ_i^2) / Σ_k exp(-||x_i-x_k||^2 / 2σ_i^2)
        p(i|j) = exp(-||x_j-x_i||^2 / 2σ_j^2) / Σ_k exp(-||x_j-x_k||^2 / 2σ_j^2)
        
        P_ij = (p(j|i) + p(i|j)) / (2n)
        
        这样定义P使得p_ij = p_ji，是一个对称的联合概率分布
        """
        n = X.shape[0]
        distances = self._compute_pairwise_distances(X)
        
        # 条件概率矩阵
        P = np.zeros((n, n))
        
        # 对每个点搜索合适的sigma
        for i in range(n):
            P[i], _ = self._binary_search_perplexity(distances, i)
        
        # 对称化：P_ij = (P_j|i + P_i|j) / (2n)
        # 这样P就是一个有效的概率分布
        P = (P + P.T) / (2 * n)
        
        # 确保数值稳定性
        P = np.maximum(P, 1e-12)
        
        return P
    
    def _compute_low_dim_affinities(self, Y: np.ndarray) -> np.ndarray:
        """
        计算低维空间中的概率分布Q
        
        使用t分布（自由度为1，即柯西分布）：
        q_ij = (1 + ||y_i - y_j||^2)^(-1) / Σ_k (1 + ||y_k - y_l||^2)^(-1)
        
        为什么选择t分布？
        1. 长尾特性：允许中等距离的点在低维空间中有更大的自由度
        2. 缓解拥挤问题：在高维空间中"拥挤"的点可以在低维空间"舒展"
        """
        n = Y.shape[0]
        
        # 计算低维距离
        sum_Y = np.sum(np.square(Y), axis=1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        
        # 对角线设为0（q_ii = 0）
        np.fill_diagonal(num, 0)
        
        # 归一化
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        
        return Q, num
    
    def _compute_gradient(self, P: np.ndarray, Q: np.ndarray, Y: np.ndarray, num: np.ndarray) -> np.ndarray:
        """
        计算KL散度对Y的梯度
        
        KL(P||Q) = Σ_ij P_ij log(P_ij/Q_ij)
        
        梯度：
        ∂KL/∂y_i = 4 Σ_j (P_ij - Q_ij) (y_i - y_j) / (1 + ||y_i - y_j||^2)
        
        参数:
            P: 高维联合概率分布
            Q: 低维概率分布
            Y: 低维嵌入
            num: 1 + ||y_i - y_j||^2 (预先计算)
        
        返回:
            dY: 梯度
        """
        n = Y.shape[0]
        
        # PQ[i,j] = (P[i,j] - Q[i,j]) / (1 + ||y_i - y_j||^2)
        PQ = (P - Q) * num
        
        # 计算梯度
        dY = np.zeros_like(Y)
        for i in range(n):
            dY[i] = np.dot(PQ[i], Y[i] - Y)
        
        dY = dY * 4.0
        
        return dY
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        拟合t-SNE模型并转换数据
        
        算法流程：
        1. 计算高维空间的联合概率P
        2. 随机初始化低维嵌入Y
        3. 对于每次迭代：
           a. 计算低维概率Q
           b. 计算KL散度梯度
           c. 用梯度下降更新Y
        4. 返回最终的低维嵌入
        
        参数:
            X: 高维数据，形状为(n_samples, n_features)
        
        返回:
            Y: 低维嵌入，形状为(n_samples, n_components)
        """
        X = np.array(X, dtype=float)
        self.n_samples_ = X.shape[0]
        
        if self.n_samples_ < 2:
            raise ValueError("至少需要2个样本")
        
        # 检查perplexity有效性
        if self.perplexity >= self.n_samples_:
            self.perplexity = max(1, self.n_samples_ - 1)
            warnings.warn(f"perplexity大于样本数，已调整为{self.perplexity}")
        
        # 步骤1: 计算高维联合概率P
        print("计算高维概率分布...")
        P = self._compute_high_dim_affinities(X)
        
        # 早期夸大：乘以early_exaggeration
        # 这使得在低维空间中，相似的点更有吸引力
        # 有助于形成清晰的簇结构
        P = P * self.early_exaggeration
        
        # 步骤2: 随机初始化低维嵌入
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 使用较小的随机值初始化
        Y = np.random.randn(self.n_samples_, self.n_components) * 0.0001
        
        # 初始化动量优化参数
        dY = np.zeros_like(Y)
        iY = np.zeros_like(Y)  # 累积梯度（用于动量）
        gains = np.ones_like(Y)  # 自适应学习率增益
        
        print(f"开始优化，共{self.n_iter}次迭代...")
        
        # 步骤3: 梯度下降优化
        for iter_num in range(self.n_iter):
            # 计算低维概率Q和距离矩阵num
            Q, num = self._compute_low_dim_affinities(Y)
            
            # 计算梯度
            dY = self._compute_gradient(P, Q, Y, num)
            
            # 自适应学习率（根据梯度符号调整增益）
            # 如果梯度方向改变，减小增益；如果方向一致，增大增益
            if iter_num > 0:
                gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
                gains = np.clip(gains, 0.01, 100)
            
            # 动量更新
            # iY = momentum * iY - learning_rate * gains * dY
            if iter_num < 250:
                momentum = 0.5
            else:
                momentum = 0.8
            
            iY = momentum * iY - self.learning_rate * gains * dY
            Y = Y + iY
            
            # 减去均值，使嵌入居中（去除平移自由度）
            Y = Y - np.mean(Y, axis=0)
            
            # 计算KL散度（每隔一定次数）
            if (iter_num + 1) % 100 == 0 or iter_num == 0:
                # 恢复原始P（去除早期夸大）
                P_normalized = P / self.early_exaggeration
                kl_div = np.sum(P_normalized * np.log(np.maximum(P_normalized / Q, 1e-10)))
                print(f"  迭代 {iter_num + 1}/{self.n_iter}, KL散度: {kl_div:.4f}")
            
            # 早期夸大阶段结束后恢复P
            if iter_num == 250:
                P = P / self.early_exaggeration
                print("早期夸大阶段结束")
        
        # 计算最终KL散度
        P_normalized = P / (self.early_exaggeration if self.n_iter <= 250 else 1)
        self.kl_divergence_ = np.sum(P_normalized * np.log(np.maximum(P_normalized / Q, 1e-10)))
        self.embedding_ = Y
        
        print(f"\n优化完成！最终KL散度: {self.kl_divergence_:.4f}")
        
        return self.embedding_
    
    def fit(self, X: np.ndarray) -> 'TSNE':
        """
        拟合t-SNE模型（仅执行fit_transform）
        """
        self.fit_transform(X)
        return self


# ============================================================================
# 第三部分：可视化函数
# ============================================================================

def plot_2d_scatter(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    title: str = "2D Visualization",
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.7,
    s: int = 50,
    cmap: str = 'viridis',
    save_path: Optional[str] = None
):
    """
    绘制2D散点图
    
    参数:
        X: 数据，形状为(n_samples, 2)
        y: 标签（可选），用于着色
        title: 图表标题
        figsize: 图像大小
        alpha: 透明度
        s: 点的大小
        cmap: 颜色映射
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if y is not None:
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, alpha=alpha, s=s, edgecolors='white', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Class')
    else:
        ax.scatter(X[:, 0], X[:, 1], alpha=alpha, s=s, c='steelblue', edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_pca_variance_explained(
    pca: PCA,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
):
    """
    绘制PCA解释方差图
    
    左图：各主成分解释的方差比例（柱状图）
    右图：累计解释方差比例（折线图）
    
    参数:
        pca: 拟合后的PCA对象
        figsize: 图像大小
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    n_components = len(pca.explained_variance_ratio_)
    components = np.arange(1, n_components + 1)
    
    # 左图：各主成分的方差比例
    axes[0].bar(components, pca.explained_variance_ratio_, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Principal Component', fontsize=12)
    axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
    axes[0].set_title('Variance Explained by Each Component', fontsize=12, fontweight='bold')
    axes[0].set_xticks(components)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 右图：累计方差比例
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(components, cumulative, 'o-', linewidth=2, markersize=8, color='coral')
    axes[1].axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
    axes[1].axhline(y=0.90, color='orange', linestyle='--', label='90% threshold')
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
    axes[1].set_title('Cumulative Explained Variance', fontsize=12, fontweight='bold')
    axes[1].set_xticks(components)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.show()


def plot_comparison_pca_tsne(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "PCA vs t-SNE",
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None
):
    """
    对比PCA和t-SNE的降维效果
    
    参数:
        X: 高维数据
        y: 标签
        title: 总标题
        figsize: 图像大小
        save_path: 保存路径
    """
    # 运行PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 运行t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=500, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # PCA结果
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    axes[0].set_xlabel('First Principal Component', fontsize=12)
    axes[0].set_ylabel('Second Principal Component', fontsize=12)
    axes[0].set_title('PCA', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE结果
    scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[1].set_title('t-SNE', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = fig.colorbar(scatter2, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Class', fontsize=12)
    
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.show()
    
    return X_pca, X_tsne


def plot_pca_components_heatmap(
    pca: PCA,
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    绘制PCA主成分载荷热力图
    
    显示每个主成分与原始特征之间的关系
    
    参数:
        pca: 拟合后的PCA对象
        feature_names: 特征名称列表
        figsize: 图像大小
        save_path: 保存路径
    """
    components = pca.components_
    n_components, n_features = components.shape
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(components, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_components))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels([f'PC {i+1}' for i in range(n_components)])
    
    ax.set_xlabel('Original Features', fontsize=12)
    ax.set_ylabel('Principal Components', fontsize=12)
    ax.set_title('PCA Component Loadings', fontsize=14, fontweight='bold')
    
    # 添加数值标注
    for i in range(n_components):
        for j in range(n_features):
            text = ax.text(j, i, f'{components[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Loading')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.show()


# ============================================================================
# 第四部分：演示示例
# ============================================================================

def demo_1_pca_basics():
    """
    示例1: PCA基础演示
    
    展示PCA如何将2D数据降到1D，并保持最大方差
    """
    print("=" * 60)
    print("示例1: PCA基础演示 - 二维数据降维")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 生成沿45度方向延伸的2D数据
    n_samples = 200
    x = np.random.randn(n_samples)
    y = 0.5 * x + 0.3 * np.random.randn(n_samples)
    X = np.column_stack([x, y])
    
    print(f"\n原始数据形状: {X.shape}")
    print(f"数据范围: x∈[{X[:,0].min():.2f}, {X[:,0].max():.2f}], y∈[{X[:,1].min():.2f}, {X[:,1].max():.2f}]")
    
    # 应用PCA
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)
    
    print(f"\nPCA结果:")
    print(f"  第一主成分方差比例: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    print(f"  第二主成分方差比例: {pca.explained_variance_ratio_[1]:.4f} ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    print(f"  主成分方向1: [{pca.components_[0,0]:.4f}, {pca.components_[0,1]:.4f}]")
    print(f"  主成分方向2: [{pca.components_[1,0]:.4f}, {pca.components_[1,1]:.4f}]")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 原始数据
    axes[0].scatter(X[:, 0], X[:, 1], alpha=0.6, c='steelblue', edgecolors='white', linewidth=0.5)
    # 绘制主成分方向
    scale = 3
    axes[0].arrow(0, 0, scale*pca.components_[0,0], scale*pca.components_[0,1],
                  head_width=0.2, head_length=0.2, fc='red', ec='red', linewidth=2, label='PC1')
    axes[0].arrow(0, 0, scale*pca.components_[1,0], scale*pca.components_[1,1],
                  head_width=0.2, head_length=0.2, fc='green', ec='green', linewidth=2, label='PC2')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    axes[0].set_title('Original Data with Principal Components')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # 变换后数据
    axes[1].scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.6, c='coral', edgecolors='white', linewidth=0.5)
    axes[1].set_xlabel('First Principal Component')
    axes[1].set_ylabel('Second Principal Component')
    axes[1].set_title('Data in Principal Component Space')
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    
    plt.tight_layout()
    plt.savefig('/tmp/pca_basic_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ 示例1完成！")
    print("观察：主成分沿着数据方差最大的方向")


def demo_2_pca_dimension_reduction():
    """
    示例2: PCA降维与数据重建
    
    展示如何降维后再近似重建原始数据
    """
    print("\n" + "=" * 60)
    print("示例2: PCA降维与数据重建")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 生成3D数据（主要变化在2D平面）
    n_samples = 300
    x = np.random.randn(n_samples)
    y = 0.6 * x + 0.4 * np.random.randn(n_samples)
    z = 0.1 * x + 0.05 * y + 0.1 * np.random.randn(n_samples)  # z方向变化很小
    X = np.column_stack([x, y, z])
    
    print(f"\n原始数据维度: {X.shape[1]}D")
    
    # 保留不同数量的主成分
    components_to_try = [1, 2, 3]
    fig = plt.figure(figsize=(16, 4))
    
    for idx, n_comp in enumerate(components_to_try):
        pca = PCA(n_components=n_comp)
        X_reduced = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_reduced)
        
        # 计算重建误差
        mse = np.mean((X - X_reconstructed) ** 2)
        variance_retained = np.sum(pca.explained_variance_ratio_)
        
        print(f"\n保留{n_comp}个主成分:")
        print(f"  保留方差比例: {variance_retained*100:.2f}%")
        print(f"  重建MSE: {mse:.6f}")
        
        # 3D可视化
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='lightgray', alpha=0.3, s=10, label='Original')
        ax.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], X_reconstructed[:, 2],
                   c='red', alpha=0.8, s=10, label='Reconstructed')
        ax.set_title(f'{n_comp} Component(s)\nVariance: {variance_retained*100:.1f}%, MSE: {mse:.4f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('/tmp/pca_reconstruction_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ 示例2完成！")
    print("观察：保留的主成分越多，重建越准确")


def demo_3_pca_wine_dataset():
    """
    示例3: PCA应用于模拟葡萄酒数据集
    
    展示PCA在多特征数据集上的应用
    """
    print("\n" + "=" * 60)
    print("示例3: PCA应用于多特征数据集")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 模拟葡萄酒数据集（3个类别，12个特征）
    n_classes = 3
    n_features = 12
    n_samples_per_class = 50
    
    X_list = []
    y_list = []
    
    for c in range(n_classes):
        # 每个类别有不同的均值
        mean = np.random.randn(n_features) * 2
        cov = np.eye(n_features) + np.random.randn(n_features, n_features) * 0.3
        cov = np.dot(cov, cov.T)  # 确保正定
        
        X_class = np.random.multivariate_normal(mean, cov, n_samples_per_class)
        X_list.append(X_class)
        y_list.extend([c] * n_samples_per_class)
    
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    print(f"\n数据集: {X.shape[0]}个样本, {X.shape[1]}个特征, {n_classes}个类别")
    
    # 应用PCA
    pca = PCA(n_components=6)
    X_pca = pca.fit_transform(X)
    
    print(f"\nPCA结果:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {ratio*100:.2f}%")
    print(f"  累计: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    # 绘制方差解释图
    plot_pca_variance_explained(pca, save_path='/tmp/pca_variance_demo.png')
    
    # 绘制2D散点图
    plot_2d_scatter(X_pca[:, :2], y, title='Wine Dataset - First 2 Principal Components',
                    save_path='/tmp/pca_scatter_demo.png')
    
    print("\n✅ 示例3完成！")


def demo_4_tsne_swiss_roll():
    """
    示例4: t-SNE展开瑞士卷（Swiss Roll）
    
    展示t-SNE如何发现非线性流形结构
    """
    print("\n" + "=" * 60)
    print("示例4: t-SNE展开瑞士卷")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 生成瑞士卷数据
    n_samples = 500
    t = np.linspace(0, 4*np.pi, n_samples)
    
    # 瑞士卷参数方程
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = np.random.uniform(-1, 1, n_samples) * 0.5
    
    X = np.column_stack([x, y, z])
    colors = t  # 用参数t作为颜色
    
    print(f"\n生成瑞士卷数据: {X.shape}")
    print("瑞士卷是3D空间中的2D流形（像一卷纸）")
    
    # 对比PCA和t-SNE
    print("\n运行PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print("运行t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=500, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 原始3D数据
    axes[0].remove()
    ax3d = fig.add_subplot(1, 3, 1, projection='3d')
    scatter = ax3d.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, cmap='rainbow', s=20)
    ax3d.set_title('Original Swiss Roll (3D)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax3d, shrink=0.5, label='Position along roll')
    
    # PCA结果
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='rainbow', s=20)
    axes[1].set_title('PCA (Linear)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].grid(True, alpha=0.3)
    
    # t-SNE结果
    scatter3 = axes[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap='rainbow', s=20)
    axes[2].set_title('t-SNE (Non-linear)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('t-SNE 1')
    axes[2].set_ylabel('t-SNE 2')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/tsne_swiss_roll.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ 示例4完成！")
    print('观察：PCA无法展开瑞士卷，t-SNE成功"展平"了它')


def demo_5_tsne_handwritten_digits():
    """
    示例5: t-SNE可视化模拟手写数字
    
    展示t-SNE在分类数据上的聚类效果
    """
    print("\n" + "=" * 60)
    print("示例5: t-SNE可视化手写数字")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 模拟手写数字数据（10个数字，64维特征）
    n_digits = 10
    n_samples_per_digit = 30
    n_features = 64
    
    X_list = []
    y_list = []
    
    for digit in range(n_digits):
        # 每个数字有不同的特征模式
        base_pattern = np.random.randn(n_features) * 0.5
        # 同一数字有变化
        for _ in range(n_samples_per_digit):
            noise = np.random.randn(n_features) * 0.3
            X_list.append(base_pattern + noise)
            y_list.append(digit)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n数据集: {len(y)}个样本, {n_features}个特征, {n_digits}个数字")
    
    # PCA降维（作为对比）
    print("\n运行PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # t-SNE降维
    print("\n运行t-SNE...")
    tsne = TSNE(n_components=2, perplexity=20, n_iter=500, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PCA
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=50, alpha=0.8, edgecolors='white')
    axes[0].set_title('PCA: Handwritten Digits', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].grid(True, alpha=0.3)
    
    # t-SNE
    scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=50, alpha=0.8, edgecolors='white')
    axes[1].set_title('t-SNE: Handwritten Digits', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')
    axes[1].grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = fig.colorbar(scatter2, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Digit', fontsize=12)
    cbar.set_ticks(range(10))
    
    plt.tight_layout()
    plt.savefig('/tmp/tsne_digits.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ 示例5完成！")
    print("观察：t-SNE将相同数字聚类在一起，分离效果更好")


def demo_6_comparison_all():
    """
    示例6: 综合对比 - 不同perplexity的t-SNE效果
    """
    print("\n" + "=" * 60)
    print("示例6: t-SNE不同困惑度(perplexity)效果对比")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 生成聚类数据
    n_clusters = 3
    n_samples = 150
    X_list = []
    y_list = []
    
    for c in range(n_clusters):
        center = np.random.randn(10) * 3
        X_cluster = center + np.random.randn(n_samples, 10) * 0.8
        X_list.append(X_cluster)
        y_list.extend([c] * n_samples)
    
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    print(f"\n生成数据: {X.shape[0]}个样本, {X.shape[1]}维")
    
    # 测试不同perplexity
    perplexities = [5, 30, 50]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, perp in enumerate(perplexities):
        print(f"\n运行t-SNE，perplexity={perp}...")
        tsne = TSNE(n_components=2, perplexity=perp, n_iter=500, random_state=42)
        X_embedded = tsne.fit_transform(X)
        
        scatter = axes[idx].scatter(X_embedded[:, 0], X_embedded[:, 1], 
                                    c=y, cmap='viridis', s=30, alpha=0.7, edgecolors='white')
        axes[idx].set_title(f'Perplexity = {perp}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Dimension 1')
        axes[idx].set_ylabel('Dimension 2')
        axes[idx].grid(True, alpha=0.3)
    
    fig.suptitle('t-SNE with Different Perplexity Values', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/tmp/tsne_perplexity.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ 示例6完成！")
    print("说明：")
    print("  - 较小perplexity(5)：关注局部结构，可能产生小簇")
    print("  - 适中perplexity(30)：平衡局部和全局")
    print("  - 较大perplexity(50)：关注全局结构")


# ============================================================================
# 第五部分：主程序入口
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║     《机器学习与深度学习：从小学生到大师》                      ║
    ║     第十五章：降维——抓住主要矛盾                                 ║
    ║     PCA和t-SNE完整实现演示                                       ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # 运行所有示例
    demo_1_pca_basics()
    demo_2_pca_dimension_reduction()
    demo_3_pca_wine_dataset()
    demo_4_tsne_swiss_roll()
    demo_5_tsne_handwritten_digits()
    demo_6_comparison_all()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
    print("""
    本章学到的核心概念：
    
    1. PCA (主成分分析):
       - 线性降维方法
       - 寻找方差最大的方向
       - 通过特征值分解实现
       - 适合线性相关的高维数据
    
    2. t-SNE (t-分布随机邻域嵌入):
       - 非线性降维方法
       - 保持邻居关系
       - 使用KL散度优化
       - 适合可视化高维数据
    
    3. 关键区别:
       - PCA是全局线性方法
       - t-SNE是局部非线性方法
       - PCA有解析解，t-SNE需要迭代优化
       - PCA可逆，t-SNE不可逆
    """)
