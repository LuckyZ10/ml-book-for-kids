"""
57.3.4 从零实现高斯过程回归
包含RBF核函数、GP回归类、以及不确定性可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

# 设置随机种子以保证可复现性
np.random.seed(42)


class RBFKernel:
    """
    径向基函数(RBF)核，也称高斯核
    
    数学公式: k(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 * l^2))
    
    参数:
        length_scale (l): 长度尺度，控制函数的平滑程度
                        值越大，函数变化越缓慢，越平滑
        sigma_f: 输出信号的标准差，控制函数的振幅
    """
    
    def __init__(self, length_scale=1.0, sigma_f=1.0):
        self.length_scale = length_scale
        self.sigma_f = sigma_f
    
    def __call__(self, X1, X2=None):
        """
        计算核矩阵
        
        Args:
            X1: 形状 (n1, d) 的输入矩阵
            X2: 形状 (n2, d) 的输入矩阵，若为None则计算X1与自身的核
            
        Returns:
            K: 形状 (n1, n2) 的核矩阵
        """
        if X2 is None:
            X2 = X1
        
        # 将输入转换为至少二维数组
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        
        # 计算欧几里得距离的平方: ||x - x'||^2
        # 使用广播技巧: (a-b)^2 = a^2 + b^2 - 2ab
        sq_dists = (
            np.sum(X1**2, axis=1).reshape(-1, 1) + 
            np.sum(X2**2, axis=1) - 
            2 * np.dot(X1, X2.T)
        )
        
        # RBF核公式
        K = self.sigma_f**2 * np.exp(-0.5 * sq_dists / self.length_scale**2)
        return K
    
    def set_params(self, length_scale, sigma_f):
        """更新核参数"""
        self.length_scale = length_scale
        self.sigma_f = sigma_f


class GaussianProcessRegressor:
    """
    高斯过程回归器
    
    使用Cholesky分解高效求解线性系统，数值稳定性更好
    """
    
    def __init__(self, kernel=None, noise_level=1e-5, optimize_hyperparams=True):
        """
        初始化GP回归器
        
        Args:
            kernel: 核函数对象，默认使用RBF核
            noise_level: 观测噪声的标准差
            optimize_hyperparams: 是否自动优化超参数
        """
        self.kernel = kernel if kernel is not None else RBFKernel()
        self.noise_level = noise_level
        self.optimize_hyperparams = optimize_hyperparams
        
        # 训练数据存储
        self.X_train = None
        self.y_train = None
        
        # Cholesky分解的缓存
        self.L = None  # 下三角矩阵
        self.alpha = None  # (K + sigma^2 I)^{-1} y
        
    def fit(self, X, y):
        """
        训练GP模型
        
        Args:
            X: 训练输入，形状 (n_samples, n_features)
            y: 训练输出，形状 (n_samples,) 或 (n_samples, 1)
        """
        self.X_train = np.atleast_2d(X)
        self.y_train = np.atleast_1d(y).reshape(-1, 1)
        
        # 如果需要，优化超参数
        if self.optimize_hyperparams:
            self._optimize_hyperparams()
        
        # 计算核矩阵 K
        K = self.kernel(self.X_train, self.X_train)
        
        # 添加噪声项: K_y = K + sigma_n^2 * I
        K_y = K + self.noise_level**2 * np.eye(len(self.X_train))
        
        # Cholesky分解: K_y = L L^T
        # 这使得求解线性系统更稳定和高效
        try:
            self.L = cholesky(K_y, lower=True)
        except np.linalg.LinAlgError:
            # 如果矩阵不正定，添加小的抖动项
            K_y += 1e-6 * np.eye(len(self.X_train))
            self.L = cholesky(K_y, lower=True)
        
        # 求解 alpha = (K_y)^{-1} y
        # 利用Cholesky分解: 先解 L v = y, 再解 L^T alpha = v
        v = solve_triangular(self.L, self.y_train, lower=True)
        self.alpha = solve_triangular(self.L.T, v, lower=False)
        
        return self
    
    def predict(self, X_test, return_std=True, return_cov=False):
        """
        对新输入进行预测
        
        Args:
            X_test: 测试输入，形状 (n_test, n_features)
            return_std: 是否返回预测标准差
            return_cov: 是否返回预测协方差矩阵
            
        Returns:
            y_mean: 预测均值
            y_std/y_cov: 预测标准差或协方差（根据return_std/return_cov）
        """
        X_test = np.atleast_2d(X_test)
        
        # 计算测试点与训练点的核向量: k_*
        k_star = self.kernel(X_test, self.X_train)
        
        # 预测均值: mu_* = k_*^T alpha
        y_mean = np.dot(k_star, self.alpha).ravel()
        
        if not return_std and not return_cov:
            return y_mean
        
        # 预测方差的计算
        # v = solve(L, k_*^T)，用于数值稳定
        v = solve_triangular(self.L, k_star.T, lower=True)
        
        # 测试点自身的核值
        k_star_star = np.diag(self.kernel(X_test, X_test))
        
        if return_cov:
            # 完整协方差矩阵: K_** - v^T v
            y_cov = self.kernel(X_test, X_test) - np.dot(v.T, v)
            return y_mean, y_cov
        
        if return_std:
            # 只返回标准差（对角线元素）
            y_var = k_star_star - np.sum(v**2, axis=0)
            # 数值稳定性处理，确保方差非负
            y_var = np.maximum(y_var, 1e-10)
            y_std = np.sqrt(y_var)
            return y_mean, y_std
    
    def log_marginal_likelihood(self, params=None):
        """
        计算对数边缘似然，用于超参数优化
        
        公式: log p(y|X) = -1/2 y^T K^{-1} y - 1/2 log|K| - n/2 log(2*pi)
        """
        if params is not None:
            # 临时设置参数
            original_params = (self.kernel.length_scale, self.kernel.sigma_f)
            self.kernel.set_params(params[0], params[1])
            K = self.kernel(self.X_train, self.X_train) + \
                self.noise_level**2 * np.eye(len(self.X_train))
            self.kernel.set_params(*original_params)
        else:
            K = self.kernel(self.X_train, self.X_train) + \
                self.noise_level**2 * np.eye(len(self.X_train))
        
        # 使用Cholesky分解计算
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return -np.inf
        
        # 求解 alpha
        v = solve_triangular(L, self.y_train, lower=True)
        alpha = solve_triangular(L.T, v, lower=False)
        
        # 计算对数边缘似然的各项
        # -1/2 y^T alpha
        data_fit = -0.5 * np.dot(self.y_train.T, alpha).ravel()[0]
        # - log|L| = - sum(log(diag(L)))
        complexity = -np.sum(np.log(np.diag(L)))
        # - n/2 log(2*pi)
        constant = -0.5 * len(self.X_train) * np.log(2 * np.pi)
        
        return data_fit + complexity + constant
    
    def _optimize_hyperparams(self):
        """通过最大化边缘似然来优化核超参数"""
        
        def neg_log_likelihood(params):
            """负对数边缘似然（最小化目标）"""
            if params[0] <= 0 or params[1] <= 0:
                return 1e10
            return -self.log_marginal_likelihood(params)
        
        # 初始猜测
        x0 = np.array([self.kernel.length_scale, self.kernel.sigma_f])
        
        # 使用L-BFGS-B优化
        bounds = [(1e-5, None), (1e-5, None)]
        result = minimize(neg_log_likelihood, x0, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            self.kernel.set_params(result.x[0], result.x[1])
            print(f"超参数优化完成: length_scale={result.x[0]:.4f}, sigma_f={result.x[1]:.4f}")


# ========================================
# 可视化GP回归的效果
# ========================================

def plot_gp_regression_1d():
    """
    一维GP回归可视化
    展示GP如何拟合函数以及不确定性如何变化
    """
    
    # 定义真实函数（待拟合的未知函数）
    def true_function(x):
        return np.sin(x) * np.exp(-0.1 * x**2) + 0.1 * x
    
    # 生成训练数据（少量观测点）
    np.random.seed(42)
    X_train = np.array([-4, -2, 0, 1, 3]).reshape(-1, 1)
    y_train = true_function(X_train).ravel() + np.random.normal(0, 0.1, len(X_train))
    
    # 创建测试点
    X_test = np.linspace(-5, 5, 200).reshape(-1, 1)
    y_true = true_function(X_test)
    
    # 创建并训练GP模型
    kernel = RBFKernel(length_scale=1.0, sigma_f=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, noise_level=0.1, optimize_hyperparams=True)
    gpr.fit(X_train, y_train)
    
    # 预测
    y_mean, y_std = gpr.predict(X_test, return_std=True)
    
    # 绘制结果
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 先验采样（训练前）
    ax1 = axes[0, 0]
    K_prior = kernel(X_test, X_test)
    # 从多元高斯分布采样
    n_samples = 5
    samples_prior = np.random.multivariate_normal(np.zeros(len(X_test)), K_prior, n_samples)
    ax1.plot(X_test, samples_prior.T, alpha=0.5)
    ax1.set_title('Prior Samples (Before Training)', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 2. 后验采样（训练后）
    ax2 = axes[0, 1]
    y_mean_plot, y_cov = gpr.predict(X_test, return_cov=True)
    samples_posterior = np.random.multivariate_normal(y_mean_plot, y_cov, n_samples)
    ax2.plot(X_test, samples_posterior.T, alpha=0.5)
    ax2.scatter(X_train, y_train, c='red', s=100, zorder=5, label='Training Data')
    ax2.set_title('Posterior Samples (After Training)', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.legend()
    
    # 3. 预测均值和置信区间
    ax3 = axes[1, 0]
    ax3.fill_between(X_test.ravel(), y_mean - 1.96*y_std, y_mean + 1.96*y_std,
                     alpha=0.3, color='blue', label='95% Confidence Interval')
    ax3.plot(X_test, y_mean, 'b-', linewidth=2, label='GP Prediction')
    ax3.plot(X_test, y_true, 'g--', linewidth=2, label='True Function')
    ax3.scatter(X_train, y_train, c='red', s=100, zorder=5, label='Training Data')
    ax3.set_title('GP Regression with Uncertainty', fontsize=12)
    ax3.set_xlabel('x')
    ax3.set_ylabel('f(x)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 不确定性可视化
    ax4 = axes[1, 1]
    ax4.fill_between(X_test.ravel(), y_mean - 2*y_std, y_mean + 2*y_std,
                     alpha=0.2, color='purple', label='±2σ')
    ax4.fill_between(X_test.ravel(), y_mean - y_std, y_mean + y_std,
                     alpha=0.3, color='blue', label='±1σ')
    ax4.plot(X_test, y_std, 'r-', linewidth=2, label='Standard Deviation')
    ax4.scatter(X_train, np.zeros_like(y_train), c='green', s=100, 
                marker='|', zorder=5, label='Training Points')
    ax4.set_title('Uncertainty (Std Dev) vs Distance from Data', fontsize=12)
    ax4.set_xlabel('x')
    ax4.set_ylabel('Uncertainty')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gp_regression_demo.png', dpi=150, bbox_inches='tight')
    print("\nGP回归可视化已保存到: gp_regression_demo.png")
    plt.show()


if __name__ == "__main__":
    plot_gp_regression_1d()