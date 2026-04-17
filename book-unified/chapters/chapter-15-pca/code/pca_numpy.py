"""
《机器学习与深度学习：从小学生到大师》
第十五章：降维——抓住主要矛盾
PCA的NumPy从零实现

本章内容：
1. 标准PCA实现
2. 增量PCA实现
3. 核PCA实现
4. 可视化与示例

作者：机器学习入门教材
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
import warnings

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 第一部分：标准PCA实现
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
    
    生活比喻：想象你在拍摄一群人的合影
    - 如果从正面拍，你能清楚看到每个人的脸（信息完整）
    - 如果从侧面拍，很多人会被挡住（信息丢失）
    - PCA就是找到最佳拍摄角度，让最多的人能被看清！
    
    参数:
        n_components: int或float，降维后的维度数或保留的方差比例
        whiten: bool，是否进行白化（使各成分方差相等）
    
    属性:
        components_: 主成分向量，形状为(n_components, n_features)
        explained_variance_: 每个主成分解释的方差值
        explained_variance_ratio_: 每个主成分解释的方差比例
        mean_: 数据的均值向量
        
    示例:
        >>> pca = PCA(n_components=2)
        >>> pca.fit(X)
        >>> X_reduced = pca.transform(X)
    """
    
    def __init__(self, n_components: Optional[Union[int, float]] = None, whiten: bool = False):
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
        self.n_components_ = None  # 实际使用的组件数
        
    def fit(self, X: np.ndarray) -> 'PCA':
        """
        拟合PCA模型，计算主成分
        
        这是PCA的"学习"阶段，就像摄影师在拍摄前寻找最佳角度：
        1. 中心化数据（减去均值）- 让数据围绕原点
        2. 计算协方差矩阵 - 了解各维度之间的关系
        3. 特征值分解 - 找到最重要的方向
        4. 按特征值排序 - 从重要到次要排列
        5. 选择主成分 - 决定保留多少信息
        
        参数:
            X: 训练数据，形状为(n_samples, n_features)
        
        返回:
            self: 拟合后的PCA对象
            
        数学推导:
            设X是中心化后的数据矩阵，我们想找到投影方向w，使得投影后的方差最大：
            
            Var(z) = w^T Σ w，其中 Σ = X^T X / (n-1)
            
            在约束 w^T w = 1 下，使用拉格朗日乘数法：
            
            L = w^T Σ w - λ(w^T w - 1)
            
            对w求导并令其为0：
            2Σw - 2λw = 0  =>  Σw = λw
            
            这正是特征值方程！所以w就是Σ的特征向量，λ是对应的特征值。
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
        # 协方差矩阵的(i,j)元素表示第i维和第j维的相关程度
        # 如果为正，表示两维同增同减；如果为负，表示反向变化
        covariance_matrix = np.dot(X_centered.T, X_centered) / (self.n_samples_ - 1)
        
        # 步骤3: 特征值分解
        # 特征值表示该方向上的方差大小
        # 特征向量表示方向本身
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # 步骤4: 按特征值从大到小排序
        # 大的特征值对应的方向包含更多信息
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 步骤5: 确定保留的主成分数量
        if self.n_components is None:
            # 保留所有特征
            self.n_components_ = self.n_features_
        elif isinstance(self.n_components, float):
            # 保留足够多的主成分以解释指定比例的方差
            cumulative_variance_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            self.n_components_ = np.argmax(cumulative_variance_ratio >= self.n_components) + 1
        else:
            # 保留指定数量的主成分
            self.n_components_ = min(self.n_components, self.n_features_)
        
        # 步骤6: 保存结果
        self.components_ = eigenvectors[:, :self.n_components_].T  # 转置为(n_components, n_features)
        self.explained_variance_ = eigenvalues[:self.n_components_]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components_] / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        将数据投影到主成分上（降维）
        
        就像把3D照片拍成2D照片，我们把高维数据压缩到低维空间。
        
        参数:
            X: 输入数据，形状为(n_samples, n_features)
        
        返回:
            X_reduced: 降维后的数据，形状为(n_samples, n_components)
        """
        X = np.array(X, dtype=float)
        X_centered = X - self.mean_
        
        # 投影：将数据乘以主成分向量
        # 这就像是计算每个数据点在各个重要方向上的"投影长度"
        X_transformed = np.dot(X_centered, self.components_.T)
        
        # 白化处理（可选）
        # 白化让每个主成分的方差都变成1，消除量纲影响
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        拟合模型并转换数据（一步完成）
        
        参数:
            X: 输入数据，形状为(n_samples, n_features)
        
        返回:
            X_reduced: 降维后的数据
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """
        将降维后的数据还原回原始空间（重构）
        
        这就像从2D照片想象回3D场景。注意：会有信息丢失！
        
        参数:
            X_reduced: 降维后的数据，形状为(n_samples, n_components)
        
        返回:
            X_reconstructed: 重构后的数据，形状为(n_samples, n_features)
        """
        # 反白化
        if self.whiten:
            X_reduced = X_reduced * np.sqrt(self.explained_variance_)
        
        # 从低维空间映射回高维空间
        X_reconstructed = np.dot(X_reduced, self.components_)
        
        # 加回均值
        X_reconstructed += self.mean_
        
        return X_reconstructed
    
    def get_precision(self) -> np.ndarray:
        """
        获取数据的精度矩阵（协方差矩阵的逆）
        
        用于计算重构误差和概率密度估计。
        """
        n_features = self.components_.shape[1]
        
        # 构造完整的协方差矩阵
        components_full = np.zeros((n_features, n_features))
        components_full[:self.n_components_, :] = self.components_
        
        explained_variance_full = np.zeros(n_features)
        explained_variance_full[:self.n_components_] = self.explained_variance_
        
        # 计算精度矩阵
        precision = np.dot(components_full.T / explained_variance_full, components_full)
        
        return precision


# ============================================================================
# 第二部分：增量PCA实现
# ============================================================================

class IncrementalPCA(PCA):
    """
    增量PCA (Incremental PCA)
    
    标准PCA需要一次性加载所有数据到内存，但当数据量巨大时这不可行。
    增量PCA允许我们分批次处理数据，逐步更新主成分。
    
    生活比喻：
    - 标准PCA就像等所有人到齐后才拍照
    - 增量PCA就像先来10个人拍一张，再来10个人调整角度，再来10个人再调整...
    - 最终的角度虽然不是最优的，但已经很接近了！
    
    适用场景：
    - 数据流（streaming data）
    - 内存无法容纳全部数据集
    - 在线学习场景
    
    算法来源：
        Ross, D. A., Lim, J., Lin, R. S., & Yang, M. H. (2008). 
        Incremental learning for robust visual tracking. 
        International Journal of Computer Vision, 77(1-3), 125-141.
    """
    
    def __init__(self, n_components: Optional[int] = None, whiten: bool = False):
        super().__init__(n_components, whiten)
        self.n_samples_seen_ = 0
        self.var_sum_ = None  # 用于增量计算协方差
        
    def partial_fit(self, X: np.ndarray) -> 'IncrementalPCA':
        """
        增量拟合 - 分批处理数据
        
        参数:
            X: 一批数据，形状为(batch_size, n_features)
        
        返回:
            self
            
        算法步骤：
        1. 更新均值：μ_new = (n_old * μ_old + n_new * μ_new) / (n_old + n_new)
        2. 更新协方差：使用Welford在线算法
        3. 对新协方差进行特征值分解
        """
        X = np.array(X, dtype=float)
        n_samples, n_features = X.shape
        
        if self.mean_ is None:
            # 第一批数据
            self.mean_ = np.mean(X, axis=0)
            self.n_samples_seen_ = n_samples
            X_centered = X - self.mean_
            self.var_sum_ = np.dot(X_centered.T, X_centered)
        else:
            # 更新均值
            last_mean = self.mean_.copy()
            last_n_samples = self.n_samples_seen_
            
            self.n_samples_seen_ += n_samples
            self.mean_ = (last_n_samples * last_mean + n_samples * np.mean(X, axis=0)) / self.n_samples_seen_
            
            # 更新协方差（使用两次中心化的技巧）
            # 这是一个数值稳定的增量算法
            X_centered = X - self.mean_
            last_mean_centered = last_mean - self.mean_
            
            self.var_sum_ += np.dot(X_centered.T, X_centered)
            self.var_sum_ += np.outer(last_mean_centered, last_mean_centered) * last_n_samples * n_samples / self.n_samples_seen_
        
        # 计算协方差矩阵
        covariance_matrix = self.var_sum_ / (self.n_samples_seen_ - 1)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # 排序
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 确定主成分数量
        if self.n_components is None:
            self.n_components_ = n_features
        else:
            self.n_components_ = min(self.n_components, n_features)
        
        # 保存结果
        self.components_ = eigenvectors[:, :self.n_components_].T
        self.explained_variance_ = eigenvalues[:self.n_components_]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components_] / np.sum(eigenvalues)
        
        return self
    
    def fit(self, X: np.ndarray) -> 'IncrementalPCA':
        """
        拟合整个数据集（使用增量方式）
        
        将大数据集分成小批次，逐步处理。
        """
        X = np.array(X, dtype=float)
        n_samples = X.shape[0]
        
        # 分批处理，每批100个样本
        batch_size = min(100, n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch = X[i:i+batch_size]
            self.partial_fit(batch)
        
        return self


# ============================================================================
# 第三部分：核PCA实现
# ============================================================================

class KernelPCA:
    """
    核PCA (Kernel PCA)
    
    标准PCA只能找到数据中的线性结构，但许多真实数据是非线性的。
    核PCA使用核技巧，先将数据映射到高维空间，再在该空间做PCA。
    
    核心思想：
    "如果直线不能分开数据，就用曲线；如果平面不能捕捉结构，就用曲面。"
    
    生活比喻：想象你在整理一团乱麻
    - 在2D平面上，这团线缠在一起，无法分开（线性不可分）
    - 但如果把这团线提起在空中（映射到3D），也许就能看清结构了
    - 核PCA就是帮你"提起线团"的工具！
    
    常用核函数：
    1. RBF核（高斯核）：K(x,y) = exp(-γ||x-y||²)
       局部性核，适合捕捉局部结构
    2. 多项式核：K(x,y) = (x·y + c)^d
       全局性核，适合捕捉多项式关系
    3. Sigmoid核：K(x,y) = tanh(αx·y + c)
       类似神经网络中的激活函数
    
    论文来源：
        Schölkopf, B., Smola, A., & Müller, K. R. (1998). 
        Nonlinear component analysis as a kernel eigenvalue problem. 
        Neural Computation, 10(5), 1299-1319.
    """
    
    def __init__(self, n_components: Optional[int] = None, kernel: str = 'rbf', 
                 gamma: Optional[float] = None, degree: int = 3, 
                 coef0: float = 1, kernel_params: Optional[dict] = None,
                 fit_inverse_transform: bool = False):
        """
        初始化核PCA
        
        参数:
            n_components: 保留的主成分数量
            kernel: 核函数类型 ('linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed')
            gamma: RBF核的系数，None表示使用 1/n_features
            degree: 多项式核的次数
            coef0: 多项式核和sigmoid核的独立项
            kernel_params: 自定义核函数的额外参数
            fit_inverse_transform: 是否学习逆变换（用于重构）
        """
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.fit_inverse_transform = fit_inverse_transform
        
        self.X_fit_ = None
        self.alphas_ = None  # 特征向量（在核空间中）
        self.lambdas_ = None  # 特征值
        self.dual_coef_ = None  # 用于逆变换的系数
        
    def _compute_kernel(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算核矩阵
        
        参数:
            X: 数据矩阵，形状为(n_samples_x, n_features)
            Y: 数据矩阵，形状为(n_samples_y, n_features)。如果为None，使用X
        
        返回:
            K: 核矩阵，形状为(n_samples_x, n_samples_y)
        """
        if Y is None:
            Y = X
            
        if self.kernel == 'linear':
            K = np.dot(X, Y.T)
        elif self.kernel == 'poly':
            K = (np.dot(X, Y.T) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            # RBF核：K(x,y) = exp(-γ||x-y||²)
            gamma = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
            # 计算距离矩阵的平方
            X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
            Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
            dist_sq = X_norm + Y_norm - 2 * np.dot(X, Y.T)
            K = np.exp(-gamma * dist_sq)
        elif self.kernel == 'sigmoid':
            K = np.tanh(self.coef0 * np.dot(X, Y.T) + self.coef0)
        elif self.kernel == 'cosine':
            # 余弦相似度
            X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
            Y_normalized = Y / np.linalg.norm(Y, axis=1, keepdims=True)
            K = np.dot(X_normalized, Y_normalized.T)
        else:
            raise ValueError(f"未知的核函数: {self.kernel}")
        
        return K
    
    def fit(self, X: np.ndarray) -> 'KernelPCA':
        """
        拟合核PCA模型
        
        算法步骤：
        1. 计算核矩阵 K
        2. 中心化核矩阵（这是关键！）
        3. 对中心化后的核矩阵进行特征值分解
        4. 归一化特征向量
        
        中心化公式：
        K_centered = K - 1_n·K - K·1_n + 1_n·K·1_n
        
        其中1_n是所有元素为1/n的矩阵。这等价于将数据在特征空间中中心化。
        """
        X = np.array(X, dtype=float)
        self.X_fit_ = X.copy()
        n_samples = X.shape[0]
        
        # 计算核矩阵
        K = self._compute_kernel(X)
        
        # 中心化核矩阵
        # 这一步非常重要！PCA要求数据是中心化的
        one_n = np.ones((n_samples, n_samples)) / n_samples
        K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        
        # 按特征值从大到小排序
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 去除负的特征值（数值误差）
        positive_mask = eigenvalues > 1e-10
        eigenvalues = eigenvalues[positive_mask]
        eigenvectors = eigenvectors[:, positive_mask]
        
        # 确定主成分数量
        if self.n_components is None:
            n_components = len(eigenvalues)
        else:
            n_components = min(self.n_components, len(eigenvalues))
        
        # 归一化特征向量（在特征空间中）
        # α_i = v_i / sqrt(λ_i)
        self.alphas_ = eigenvectors[:, :n_components] / np.sqrt(eigenvalues[:n_components])
        self.lambdas_ = eigenvalues[:n_components]
        
        # 如果需要逆变换，学习重构系数
        if self.fit_inverse_transform:
            # 使用核岭回归学习逆映射
            K_inv = K_centered + 1e-10 * np.eye(n_samples)  # 正则化
            self.dual_coef_ = np.linalg.solve(K_inv, X)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        将数据投影到核主成分上
        
        参数:
            X: 输入数据，形状为(n_samples, n_features)
        
        返回:
            投影后的数据，形状为(n_samples, n_components)
        """
        X = np.array(X, dtype=float)
        
        # 计算新数据与训练数据的核矩阵
        K = self._compute_kernel(X, self.X_fit_)
        
        # 中心化
        n_test = X.shape[0]
        n_train = self.X_fit_.shape[0]
        one_test = np.ones((n_test, n_train)) / n_train
        one_train = np.ones((n_train, n_train)) / n_train
        
        K_centered = K - one_test.dot(self._compute_kernel(self.X_fit_))
        K_centered -= K.dot(one_train)
        K_centered += one_test.dot(self._compute_kernel(self.X_fit_)).dot(one_train)
        
        # 投影
        return np.dot(K_centered, self.alphas_)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """拟合并转换数据"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        将降维后的数据还原到原始空间（近似）
        
        注意：核PCA的逆变换是近似的，因为我们无法直接知道特征空间到原始空间的映射
        """
        if not self.fit_inverse_transform:
            raise ValueError("inverse_transform需要先设置fit_inverse_transform=True")
        
        # 通过核岭回归近似逆映射
        K_approx = np.dot(X_transformed, self.alphas_.T)
        return np.dot(K_approx, self.dual_coef_)


# ============================================================================
# 第四部分：可视化工具
# ============================================================================

def plot_pca_2d(X: np.ndarray, y: Optional[np.ndarray] = None, 
                title: str = "PCA降维结果", figsize: Tuple[int, int] = (10, 8)):
    """
    可视化PCA降维结果
    
    参数:
        X: 降维后的2D数据
        y: 类别标签（可选）
        title: 图表标题
        figsize: 图表大小
    """
    plt.figure(figsize=figsize)
    
    if y is not None:
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6, edgecolors='w', s=50)
        plt.colorbar(scatter, label='类别')
    else:
        plt.scatter(X[:, 0], X[:, 1], alpha=0.6, edgecolors='w', s=50)
    
    plt.xlabel('第一主成分', fontsize=12)
    plt.ylabel('第二主成分', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def plot_explained_variance(pca: PCA, title: str = "方差解释比例"):
    """
    绘制方差解释比例图
    
    这个图帮助我们决定保留多少个主成分。
    """
    plt.figure(figsize=(12, 5))
    
    # 子图1: 单个方差解释比例
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_, alpha=0.7)
    plt.xlabel('主成分', fontsize=12)
    plt.ylabel('解释的方差比例', fontsize=12)
    plt.title('各主成分解释的方差比例', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 子图2: 累积方差解释比例
    plt.subplot(1, 2, 2)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative) + 1), cumulative, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95%阈值')
    plt.xlabel('主成分数量', fontsize=12)
    plt.ylabel('累积解释的方差比例', fontsize=12)
    plt.title('累积方差解释比例', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt


def plot_pca_reconstruction(X: np.ndarray, pca: PCA, n_samples: int = 5):
    """
    可视化PCA重构效果
    
    对比原始数据和重构后的数据，展示信息丢失情况。
    """
    X_reduced = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_reduced)
    
    plt.figure(figsize=(15, 3))
    
    for i in range(n_samples):
        plt.subplot(2, n_samples, i + 1)
        if len(X.shape) == 2 and X.shape[1] > 10:
            # 如果是高维数据（如图像），展示为一维
            plt.plot(X[i])
            plt.title(f'原始 #{i+1}')
            plt.xticks([])
        else:
            plt.imshow(X[i].reshape(int(np.sqrt(X.shape[1])), -1), cmap='gray')
            plt.title(f'原始 #{i+1}')
            plt.axis('off')
        
        plt.subplot(2, n_samples, n_samples + i + 1)
        if len(X.shape) == 2 and X.shape[1] > 10:
            plt.plot(X_reconstructed[i])
            plt.title(f'重构 #{i+1}')
            plt.xticks([])
        else:
            plt.imshow(X_reconstructed[i].reshape(int(np.sqrt(X.shape[1])), -1), cmap='gray')
            plt.title(f'重构 #{i+1}')
            plt.axis('off')
    
    plt.tight_layout()
    return plt


# ============================================================================
# 第五部分：示例和演示
# ============================================================================

def demo_basic_pca():
    """
    演示基本PCA用法
    
    生成一个2D数据集，然后用PCA降到1D，展示信息保留和丢失。
    """
    print("=" * 60)
    print("演示1: 基本PCA用法")
    print("=" * 60)
    
    # 生成相关数据
    np.random.seed(42)
    n_samples = 200
    
    # 生成沿着某个方向的椭圆数据
    angle = np.pi / 4  # 45度
    X = np.random.randn(n_samples, 2)
    X[:, 0] *= 3  # x方向拉伸
    
    # 旋转
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
    X = np.dot(X, rotation_matrix)
    
    # 应用PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"原始数据形状: {X.shape}")
    print(f"降维后形状: {X_pca.shape}")
    print(f"\n主成分方向:")
    print(f"  第一主成分: {pca.components_[0]}")
    print(f"  第二主成分: {pca.components_[1]}")
    print(f"\n解释的方差比例:")
    print(f"  第一主成分: {pca.explained_variance_ratio_[0]:.4f}")
    print(f"  第二主成分: {pca.explained_variance_ratio_[1]:.4f}")
    print(f"  累计: {sum(pca.explained_variance_ratio_):.4f}")
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    # 原始数据
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('原始数据')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # 画出主成分方向
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_ratio_)):
        plt.arrow(0, 0, comp[0] * var * 5, comp[1] * var * 5,
                 head_width=0.2, head_length=0.2, fc=f'C{i+1}', ec=f'C{i+1}')
    
    # PCA变换后的数据
    plt.subplot(1, 3, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    plt.xlabel('第一主成分')
    plt.ylabel('第二主成分')
    plt.title('PCA变换后')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # 重构（只使用第一主成分）
    plt.subplot(1, 3, 3)
    pca_1d = PCA(n_components=1)
    X_1d = pca_1d.fit_transform(X)
    X_reconstructed = pca_1d.inverse_transform(X_1d)
    
    plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label='原始数据')
    plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], 
               alpha=0.8, label='重构数据（1D）')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('1D重构 vs 原始数据')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_basic_demo.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存为: pca_basic_demo.png")
    plt.show()


def demo_incremental_pca():
    """演示增量PCA"""
    print("\n" + "=" * 60)
    print("演示2: 增量PCA")
    print("=" * 60)
    
    # 生成大数据集
    np.random.seed(42)
    n_samples = 10000
    n_features = 100
    X = np.random.randn(n_samples, n_features)
    
    # 添加一些结构（让PCA有意义）
    for i in range(5):
        X[:, i] *= (5 - i)  # 前5维有更多方差
    
    print(f"数据集大小: {X.shape}")
    
    # 标准PCA
    print("\n标准PCA...")
    pca = PCA(n_components=10)
    import time
    t0 = time.time()
    X_pca = pca.fit_transform(X)
    t1 = time.time()
    print(f"  耗时: {t1-t0:.4f}秒")
    print(f"  解释的方差: {sum(pca.explained_variance_ratio_):.4f}")
    
    # 增量PCA
    print("\n增量PCA...")
    ipca = IncrementalPCA(n_components=10)
    t0 = time.time()
    X_ipca = ipca.fit_transform(X)
    t1 = time.time()
    print(f"  耗时: {t1-t0:.4f}秒")
    print(f"  解释的方差: {sum(ipca.explained_variance_ratio_):.4f}")
    
    # 比较结果
    print(f"\n结果差异: {np.mean((X_pca - X_ipca) ** 2):.6f}")
    print("(增量PCA结果是近似的，但非常接近)")


def demo_kernel_pca():
    """演示核PCA处理非线性数据"""
    print("\n" + "=" * 60)
    print("演示3: 核PCA（非线性降维）")
    print("=" * 60)
    
    # 生成同心圆数据（非线性结构）
    np.random.seed(42)
    n_samples = 500
    
    # 外圆
    theta = np.linspace(0, 2*np.pi, n_samples//2)
    r = 2 + np.random.randn(n_samples//2) * 0.1
    X_outer = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    y_outer = np.zeros(n_samples//2)
    
    # 内圆
    theta = np.linspace(0, 2*np.pi, n_samples//2)
    r = 1 + np.random.randn(n_samples//2) * 0.1
    X_inner = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    y_inner = np.ones(n_samples//2)
    
    X = np.vstack([X_outer, X_inner])
    y = np.hstack([y_outer, y_inner])
    
    print("数据: 同心圆（线性不可分）")
    
    # 标准PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 核PCA
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=1.0)
    X_kpca = kpca.fit_transform(X)
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.title('原始数据（同心圆）')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.title('标准PCA（无法分开）')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.title('核PCA（RBF核，完美分开）')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kernel_pca_demo.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存为: kernel_pca_demo.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("PCA NumPy实现演示")
    print("=" * 60)
    
    demo_basic_pca()
    demo_incremental_pca()
    demo_kernel_pca()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
