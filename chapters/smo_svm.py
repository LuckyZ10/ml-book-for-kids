"""
SMO (Sequential Minimal Optimization) 算法简化实现
用于高效求解SVM对偶问题

参考: Platt, J. (1998). Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines

《机器学习与深度学习：从小学生到大师》第十章配套代码
"""
import numpy as np
import random


class SimplifiedSMO:
    """
    SMO算法简化实现
    
    核心思想：每次只优化两个拉格朗日乘子 α_i 和 α_j，
    这样可以解析求解，而不需要复杂的QP优化器。
    """
    def __init__(self, X, y, C=1.0, tolerance=0.001, max_passes=100, kernel_type='linear', gamma=1.0):
        """
        初始化SMO
        
        参数:
            X: 训练数据 (n_samples, n_features)
            y: 标签 (n_samples,)，取值为 +1 或 -1
            C: 正则化参数
            tolerance: KKT条件违反的容差
            max_passes: 最大迭代轮数
            kernel_type: 'linear' 或 'rbf'
            gamma: RBF核参数
        """
        self.X = X
        self.y = y
        self.C = C
        self.tol = tolerance
        self.max_passes = max_passes
        self.kernel_type = kernel_type
        self.gamma = gamma
        
        self.m, self.n = X.shape  # 样本数和特征数
        
        # 初始化拉格朗日乘子 α 和偏置 b
        self.alphas = np.zeros(self.m)
        self.b = 0.0
        
        # 预计算核矩阵（简化版，适用于中小数据集）
        self.K = self._compute_kernel_matrix()
        
    def _compute_kernel_matrix(self):
        """计算核矩阵 K[i,j] = K(x_i, x_j)"""
        if self.kernel_type == 'linear':
            # 线性核: K(x, x') = x · x'
            return np.dot(self.X, self.X.T)
        elif self.kernel_type == 'rbf':
            # RBF核: K(x, x') = exp(-γ ||x - x'||²)
            X_norm = np.sum(self.X**2, axis=1).reshape(-1, 1)
            dist_sq = X_norm + X_norm.T - 2 * np.dot(self.X, self.X.T)
            return np.exp(-self.gamma * dist_sq)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def _kernel(self, i, j):
        """获取核矩阵元素 K[i,j]"""
        return self.K[i, j]
    
    def _predict_output(self, i):
        """
        计算样本 i 的预测输出 f(x_i)
        f(x_i) = Σ(α_k · y_k · K(x_k, x_i)) + b
        """
        return np.sum(self.alphas * self.y * self.K[:, i]) + self.b
    
    def _calculate_error(self, i):
        """计算样本 i 的预测误差 E_i = f(x_i) - y_i"""
        return self._predict_output(i) - self.y[i]
    
    def _select_j_randomly(self, i):
        """随机选择 j ≠ i"""
        j = i
        while j == i:
            j = random.randint(0, self.m - 1)
        return j
    
    def _clip_alpha(self, alpha, H, L):
        """将 α 裁剪到 [L, H] 范围内"""
        if alpha > H:
            return H
        if alpha < L:
            return L
        return alpha
    
    def _take_step(self, i, j):
        """
        尝试优化 α_i 和 α_j 这一对乘子
        
        返回 True 如果成功更新，False 如果没有进展
        """
        if i == j:
            return False
        
        alpha_i_old = self.alphas[i].copy()
        alpha_j_old = self.alphas[j].copy()
        yi, yj = self.y[i], self.y[j]
        
        # 计算误差
        Ei = self._calculate_error(i)
        Ej = self._calculate_error(j)
        
        # 计算 α_j 的边界 L 和 H
        if yi != yj:
            # 当 y_i ≠ y_j 时
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)
        else:
            # 当 y_i = y_j 时
            L = max(0, alpha_i_old + alpha_j_old - self.C)
            H = min(self.C, alpha_i_old + alpha_j_old)
        
        if L == H:
            return False
        
        # 计算 η = K_ii + K_jj - 2K_ij
        eta = self._kernel(i, i) + self._kernel(j, j) - 2 * self._kernel(i, j)
        
        if eta <= 0:
            return False
        
        # 计算未裁剪的新 α_j
        alpha_j_new_unc = alpha_j_old + yj * (Ei - Ej) / eta
        
        # 裁剪 α_j 到 [L, H]
        alpha_j_new = self._clip_alpha(alpha_j_new_unc, H, L)
        
        # 检查变化是否显著
        if abs(alpha_j_new - alpha_j_old) < 1e-5:
            return False
        
        # 计算新的 α_i
        # α_i^new = α_i^old + y_i·y_j·(α_j^old - α_j^new)
        alpha_i_new = alpha_i_old + yi * yj * (alpha_j_old - alpha_j_new)
        
        # 更新偏置 b
        b1 = (self.b - Ei - yi * (alpha_i_new - alpha_i_old) * self._kernel(i, i) 
              - yj * (alpha_j_new - alpha_j_old) * self._kernel(i, j))
        b2 = (self.b - Ej - yi * (alpha_i_new - alpha_i_old) * self._kernel(i, j) 
              - yj * (alpha_j_new - alpha_j_old) * self._kernel(j, j))
        
        # 根据 α 是否在边界内来选择 b
        if 0 < alpha_i_new < self.C:
            self.b = b1
        elif 0 < alpha_j_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2.0
        
        # 更新 α
        self.alphas[i] = alpha_i_new
        self.alphas[j] = alpha_j_new
        
        return True
    
    def _examine_example(self, i):
        """
        检查样本 i 的KKT条件，尝试优化它
        
        KKT条件：
        - 如果 α_i = 0，则 y_i·f(x_i) ≥ 1（样本正确分类且在间隔外）
        - 如果 0 < α_i < C，则 y_i·f(x_i) = 1（样本在间隔边界上）
        - 如果 α_i = C，则 y_i·f(x_i) ≤ 1（样本在间隔内或分类错误）
        """
        yi = self.y[i]
        alpha_i = self.alphas[i]
        Ei = self._calculate_error(i)
        
        # 检查KKT条件是否违反
        r = Ei * yi
        
        # 违反条件的情况：
        # 1. r < -tol 且 α_i < C（应该增加 α_i）
        # 2. r > tol 且 α_i > 0（应该减小 α_i）
        violate_kkt = (r < -self.tol and alpha_i < self.C) or (r > self.tol and alpha_i > 0)
        
        if not violate_kkt:
            return False
        
        # 启发式1：优先选择非边界上的 α_j（0 < α < C）
        non_bound_idx = np.where((self.alphas > 0) & (self.alphas < self.C))[0]
        
        if len(non_bound_idx) > 1:
            # 选择使 |Ei - Ej| 最大的 j
            max_delta_E = 0
            best_j = -1
            for k in non_bound_idx:
                if k == i:
                    continue
                Ek = self._calculate_error(k)
                delta_E = abs(Ei - Ek)
                if delta_E > max_delta_E:
                    max_delta_E = delta_E
                    best_j = k
            
            if best_j != -1 and self._take_step(i, best_j):
                return True
        
        # 启发式2：在非边界点中随机尝试
        non_bound_list = list(non_bound_idx)
        random.shuffle(non_bound_list)
        for j in non_bound_list:
            if j != i and self._take_step(i, j):
                return True
        
        # 启发式3：在所有点中随机尝试
        all_idx = list(range(self.m))
        random.shuffle(all_idx)
        for j in all_idx:
            if j != i and self._take_step(i, j):
                return True
        
        return False
    
    def fit(self):
        """训练SVM"""
        print("🚀 开始SMO训练...")
        
        num_changed = 0
        examine_all = True
        passes = 0
        
        while (num_changed > 0 or examine_all) and passes < self.max_passes:
            num_changed = 0
            
            if examine_all:
                # 遍历所有样本
                for i in range(self.m):
                    if self._examine_example(i):
                        num_changed += 1
                print(f"  全遍历轮次: 更新了 {num_changed} 个 α")
            else:
                # 只遍历非边界样本
                non_bound_idx = np.where((self.alphas > 0) & (self.alphas < self.C))[0]
                for i in non_bound_idx:
                    if self._examine_example(i):
                        num_changed += 1
                print(f"  非边界遍历: 更新了 {num_changed} 个 α")
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            passes += 1
            print(f"  完成第 {passes} 轮")
        
        print(f"✅ 训练完成！共 {passes} 轮")
        
        # 提取支持向量
        self.support_vector_idx = np.where(self.alphas > 1e-5)[0]
        self.support_vectors = self.X[self.support_vector_idx]
        self.support_vector_labels = self.y[self.support_vector_idx]
        self.support_vector_alphas = self.alphas[self.support_vector_idx]
        
        print(f"📊 支持向量数量: {len(self.support_vector_idx)} / {self.m}")
        
    def predict(self, X):
        """
        预测新样本的类别
        
        f(x) = Σ(α_sv · y_sv · K(x_sv, x)) + b
        """
        if self.kernel_type == 'linear':
            # 线性核可以直接计算 w·x + b
            w = np.sum((self.alphas * self.y).reshape(-1, 1) * self.X, axis=0)
            scores = np.dot(X, w) + self.b
        else:
            # 非线性核需要计算与所有支持向量的核函数
            scores = np.zeros(X.shape[0])
            for i, x in enumerate(X):
                # 计算 x 与所有支持向量的核函数值
                if self.kernel_type == 'rbf':
                    # RBF核
                    dist_sq = np.sum((self.support_vectors - x)**2, axis=1)
                    k_values = np.exp(-self.gamma * dist_sq)
                scores[i] = np.sum(self.support_vector_alphas * self.support_vector_labels * k_values) + self.b
        
        return np.sign(scores)
    
    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


def demo_smo():
    """演示SMO算法"""
    from sklearn.datasets import make_blobs, make_circles, make_moons
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("🎯 SMO算法演示")
    print("=" * 60)
    
    # 测试1: 线性可分数据
    print("\n📌 测试1: 线性可分数据")
    X1, y1 = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)
    y1 = np.where(y1 == 0, -1, 1)
    
    svm1 = SimplifiedSMO(X1, y1, C=1.0, kernel_type='linear', max_passes=50)
    svm1.fit()
    acc1 = svm1.score(X1, y1)
    print(f"训练准确率: {acc1 * 100:.2f}%")
    
    # 测试2: 非线性数据（月亮形状）
    print("\n📌 测试2: 非线性数据（月亮形状）- 使用RBF核")
    X2, y2 = make_moons(n_samples=100, noise=0.1, random_state=42)
    y2 = np.where(y2 == 0, -1, 1)
    
    svm2 = SimplifiedSMO(X2, y2, C=10.0, kernel_type='rbf', gamma=5.0, max_passes=100)
    svm2.fit()
    acc2 = svm2.score(X2, y2)
    print(f"训练准确率: {acc2 * 100:.2f}%")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 图1: 线性数据
    plot_svm_boundary(axes[0], X1, y1, svm1, "线性SVM")
    
    # 图2: 非线性数据
    plot_svm_boundary(axes[1], X2, y2, svm2, "RBF核SVM")
    
    plt.tight_layout()
    plt.savefig('smo_demo.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_svm_boundary(ax, X, y, svm, title):
    """绘制SVM决策边界"""
    # 绘制数据点
    pos_idx = y == 1
    neg_idx = y == -1
    ax.scatter(X[pos_idx, 0], X[pos_idx, 1], c='gold', s=50, 
              edgecolors='black', label='Class +1', alpha=0.8)
    ax.scatter(X[neg_idx, 0], X[neg_idx, 1], c='navy', s=50, 
              edgecolors='black', label='Class -1', alpha=0.8)
    
    # 绘制支持向量
    if len(svm.support_vector_idx) > 0:
        ax.scatter(X[svm.support_vector_idx, 0], X[svm.support_vector_idx, 1], 
                  s=200, facecolors='none', edgecolors='red', linewidths=2,
                  label='Support Vectors')
    
    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, levels=[-2, 0, 2], colors=['blue', 'red'])
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    demo_smo()
