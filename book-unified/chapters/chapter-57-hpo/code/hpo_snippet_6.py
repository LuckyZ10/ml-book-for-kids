"""
57.3.6 完整贝叶斯优化器实现
整合GP、采集函数，应用于超参数调优
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# 复用之前实现的GP和采集函数
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import differential_evolution


class RBFKernel:
    """RBF核函数（复用）"""
    def __init__(self, length_scale=1.0, sigma_f=1.0):
        self.length_scale = length_scale
        self.sigma_f = sigma_f
    
    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        sq_dists = (
            np.sum(X1**2, axis=1).reshape(-1, 1) + 
            np.sum(X2**2, axis=1) - 
            2 * np.dot(X1, X2.T)
        )
        return self.sigma_f**2 * np.exp(-0.5 * sq_dists / self.length_scale**2)


class GaussianProcess:
    """简化版GP（复用核心功能）"""
    def __init__(self, noise=1e-5):
        self.kernel = RBFKernel()
        self.noise = noise
        self.X = None
        self.y = None
        self.L = None
        self.alpha = None
    
    def fit(self, X, y):
        self.X = np.atleast_2d(X)
        self.y = np.array(y).reshape(-1, 1)
        
        K = self.kernel(self.X, self.X) + self.noise**2 * np.eye(len(self.X))
        self.L = cholesky(K, lower=True)
        v = solve_triangular(self.L, self.y, lower=True)
        self.alpha = solve_triangular(self.L.T, v, lower=False)
        return self
    
    def predict(self, X_test, return_std=True):
        X_test = np.atleast_2d(X_test)
        k_star = self.kernel(X_test, self.X)
        y_mean = np.dot(k_star, self.alpha).ravel()
        
        if not return_std:
            return y_mean
        
        v = solve_triangular(self.L, k_star.T, lower=True)
        k_star_star = np.diag(self.kernel(X_test, X_test))
        y_var = k_star_star - np.sum(v**2, axis=0)
        y_var = np.maximum(y_var, 1e-10)
        return y_mean, np.sqrt(y_var)


class BayesianOptimizer:
    """
    贝叶斯优化器
    
    用于黑盒函数优化的通用框架
    """
    
    def __init__(self, bounds, acquisition='ei', xi=0.01, kappa=2.0, 
                 random_state=None, verbose=True):
        """
        Args:
            bounds: 搜索空间边界，列表形式 [(min1, max1), (min2, max2), ...]
            acquisition: 采集函数类型 ('pi', 'ei', 'ucb')
            xi: PI/EI的探索参数
            kappa: UCB的探索参数
            random_state: 随机种子
            verbose: 是否打印进度
        """
        self.bounds = np.array(bounds)
        self.acquisition_type = acquisition
        self.xi = xi
        self.kappa = kappa
        self.verbose = verbose
        
        if random_state:
            np.random.seed(random_state)
        
        # 内部状态
        self.gp = None
        self.X_observed = []
        self.y_observed = []
        self.best_y = float('-inf')
        self.best_x = None
        
        # 记录优化历史
        self.history = {
            'X': [],
            'y': [],
            'best_y': [],
            'acquisition_max': []
        }
    
    def _acquisition_function(self, X):
        """
        计算采集函数值（用于优化）
        
        注意：这里返回负值，因为我们要用minimize找到最大值
        """
        X = np.atleast_2d(X)
        mu, sigma = self.gp.predict(X, return_std=True)
        
        if self.acquisition_type == 'pi':
            # Probability of Improvement
            sigma = np.maximum(sigma, 1e-9)
            Z = (mu - self.best_y - self.xi) / sigma
            return -norm.cdf(Z)
        
        elif self.acquisition_type == 'ei':
            # Expected Improvement
            sigma = np.maximum(sigma, 1e-9)
            Z = (mu - self.best_y - self.xi) / sigma
            ei = (mu - self.best_y - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            return -ei
        
        elif self.acquisition_type == 'ucb':
            # Upper Confidence Bound
            return -(mu + self.kappa * sigma)
        
        else:
            raise ValueError(f"Unknown acquisition: {self.acquisition_type}")
    
    def _propose_next_point(self):
        """找到下一个采样点（最大化采集函数）"""
        
        # 使用差分进化全局优化寻找采集函数的最大值
        # 这是一个鲁棒的启发式优化方法
        result = differential_evolution(
            self._acquisition_function,
            self.bounds,
            maxiter=100,
            polish=True,
            seed=np.random.randint(10000)
        )
        
        return result.x
    
    def maximize(self, objective, n_iterations=10, n_initial_points=5):
        """
        最大化损失函数
        
        Args:
            objective: 损失函数，接受数组返回标量
            n_iterations: 优化迭代次数
            n_initial_points: 初始随机采样点数
            
        Returns:
            best_x: 最优输入
            best_y: 最优值
        """
        
        print(f"开始贝叶斯优化 ({self.acquisition_type.upper()}采集函数)")
        print(f"搜索空间维度: {len(self.bounds)}")
        print(f"初始随机点: {n_initial_points}, 优化迭代: {n_iterations}")
        print("=" * 60)
        
        # 1. 初始随机采样（拉丁超立方采样会更优，这里用均匀随机）
        if self.verbose:
            print("\n[阶段1] 初始随机采样...")
        
        for i in range(n_initial_points):
            x_random = np.array([
                np.random.uniform(low, high) for low, high in self.bounds
            ])
            y_random = objective(x_random)
            
            self.X_observed.append(x_random)
            self.y_observed.append(y_random)
            
            if y_random > self.best_y:
                self.best_y = y_random
                self.best_x = x_random.copy()
            
            if self.verbose:
                print(f"  随机点 {i+1}: y={y_random:.4f}, 当前最优={self.best_y:.4f}")
        
        # 2. 贝叶斯优化循环
        if self.verbose:
            print(f"\n[阶段2] 贝叶斯优化迭代...")
        
        for i in range(n_iterations):
            # 拟合GP模型
            X_array = np.array(self.X_observed)
            y_array = np.array(self.y_observed)
            
            # 标准化y值（有助于GP拟合）
            self.y_mean = np.mean(y_array)
            self.y_std = np.std(y_array) if np.std(y_array) > 0 else 1
            y_normalized = (y_array - self.y_mean) / self.y_std
            
            self.gp = GaussianProcess()
            self.gp.fit(X_array, y_normalized)
            
            # 找到下一个采样点
            next_x = self._propose_next_point()
            next_y = objective(next_x)
            
            # 更新观测
            self.X_observed.append(next_x)
            self.y_observed.append(next_y)
            
            # 更新最优
            if next_y > self.best_y:
                self.best_y = next_y
                self.best_x = next_x.copy()
                improved = "*** 改进! ***"
            else:
                improved = ""
            
            # 记录历史
            self.history['X'].append(next_x)
            self.history['y'].append(next_y)
            self.history['best_y'].append(self.best_y)
            
            if self.verbose:
                print(f"  迭代 {i+1}/{n_iterations}: y={next_y:.4f}, "
                      f"最优={self.best_y:.4f} {improved}")
        
        print("=" * 60)
        print(f"优化完成!")
        print(f"最优值: {self.best_y:.4f}")
        print(f"最优参数: {self.best_x}")
        
        return self.best_x, self.best_y


# ========================================
# 应用于超参数调优示例
# ========================================

def optimize_svm_hyperparameters():
    """
    使用贝叶斯优化调优SVM超参数
    优化目标：C (正则化参数) 和 gamma (核系数)
    """
    
    print("\n示例1: SVM超参数优化")
    print("优化参数: C (正则化强度) 和 gamma (RBF核系数)")
    
    # 加载数据集
    X, y = load_digits(n_class=10, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    def svm_objective(params):
        """
        损失函数：SVM在交叉验证上的准确率
        params[0]: C (对数尺度，实际值 10^params[0])
        params[1]: gamma (对数尺度，实际值 10^params[1])
        """
        C = 10 ** params[0]  # 从对数空间转换
        gamma = 10 ** params[1]
        
        try:
            model = SVC(C=C, gamma=gamma, random_state=42)
            # 使用3折交叉验证，取平均准确率
            scores = cross_val_score(model, X_train, y_train, cv=3, 
                                     scoring='accuracy', n_jobs=-1)
            return scores.mean()
        except Exception as e:
            return 0.0  # 出错时返回差值
    
    # 定义搜索空间（对数尺度）
    # C的范围: 10^-3 到 10^3
    # gamma的范围: 10^-4 到 10^1
    bounds = [(-3, 3), (-4, 1)]
    
    # 创建优化器并使用EI采集函数
    optimizer = BayesianOptimizer(
        bounds=bounds, 
        acquisition='ei',
        xi=0.01,
        random_state=42
    )
    
    best_params_log, best_score = optimizer.maximize(
        svm_objective,
        n_iterations=15,
        n_initial_points=5
    )
    
    # 转换回实际值
    best_C = 10 ** best_params_log[0]
    best_gamma = 10 ** best_params_log[1]
    
    print(f"\n最优超参数:")
    print(f"  C = {best_C:.4f}")
    print(f"  gamma = {best_gamma:.6f}")
    
    # 在测试集上验证
    final_model = SVC(C=best_C, gamma=best_gamma, random_state=42)
    final_model.fit(X_train, y_train)
    test_accuracy = final_model.score(X_test, y_test)
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    return optimizer, (best_C, best_gamma, test_accuracy)


def optimize_random_forest():
    """
    使用贝叶斯优化调优随机森林超参数
    优化目标：n_estimators 和 max_depth
    """
    
    print("\n示例2: 随机森林超参数优化")
    print("优化参数: n_estimators (树的数量) 和 max_depth (最大深度)")
    
    # 创建合成数据集
    X, y = make_classification(n_samples=1000, n_features=20, 
                               n_informative=10, n_redundant=5,
                               random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    def rf_objective(params):
        """
        损失函数
        params[0]: n_estimators (50-300)
        params[1]: max_depth (2-50)
        """
        n_estimators = int(params[0])
        max_depth = int(params[1])
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        scores = cross_val_score(model, X_train, y_train, cv=3,
                                 scoring='accuracy', n_jobs=-1)
        return scores.mean()
    
    # 搜索空间
    bounds = [(50, 300), (2, 50)]
    
    optimizer = BayesianOptimizer(
        bounds=bounds,
        acquisition='ei',
        random_state=42
    )
    
    best_params, best_score = optimizer.maximize(
        rf_objective,
        n_iterations=12,
        n_initial_points=5
    )
    
    best_n_estimators = int(best_params[0])
    best_max_depth = int(best_params[1])
    
    print(f"\n最优超参数:")
    print(f"  n_estimators = {best_n_estimators}")
    print(f"  max_depth = {best_max_depth}")
    
    # 测试集验证
    final_model = RandomForestClassifier(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        random_state=42
    )
    final_model.fit(X_train, y_train)
    test_accuracy = final_model.score(X_test, y_test)
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    return optimizer


def visualize_optimization_process():
    """可视化贝叶斯优化过程"""
    
    # 定义一个一维测试函数
    def test_func(x):
        x = x[0] if hasattr(x, '__len__') else x
        return np.sin(3*x) * x**2 * np.exp(-x) + np.random.normal(0, 0.01)
    
    bounds = [(0, 5)]
    
    optimizer = BayesianOptimizer(bounds=bounds, acquisition='ei', verbose=False)
    optimizer.maximize(test_func, n_iterations=10, n_initial_points=3)
    
    # 绘制优化过程
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 收敛曲线
    ax1 = axes[0]
    iterations = range(1, len(optimizer.y_observed) + 1)
    ax1.plot(iterations, optimizer.y_observed, 'bo-', alpha=0.6, label='Observed Value')
    
    best_so_far = np.maximum.accumulate(optimizer.y_observed)
    ax1.plot(iterations, best_so_far, 'r-', linewidth=2, label='Best So Far')
    
    ax1.axvline(x=3.5, color='gray', linestyle='--', alpha=0.5, label='End of Random')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Bayesian Optimization Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 采样点分布
    ax2 = axes[1]
    X_obs = np.array(optimizer.X_observed).ravel()
    y_obs = np.array(optimizer.y_observed)
    
    colors = ['blue'] * 3 + ['red'] * 10  # 前3个是随机，后10个是BO
    labels_rand = ['Random Sample'] * 3 + [''] * 10
    labels_bo = [''] * 3 + ['BO Sample'] * 10
    
    for i, (x, y, c) in enumerate(zip(X_obs, y_obs, colors)):
        label_r = labels_rand[i] if labels_rand[i] else None
        label_b = labels_bo[i] if labels_bo[i] and not labels_rand[i] else None
        ax2.scatter(x, y, c=c, s=100, alpha=0.7, 
                   label=label_r or label_b, edgecolors='black')
    
    # 绘制真实函数
    x_fine = np.linspace(0, 5, 200)
    y_true = [test_func([xi]) for xi in x_fine]
    ax2.plot(x_fine, y_true, 'g--', alpha=0.5, label='True Function')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.set_title('Sample Points Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bayesian_optimization_process.png', dpi=150, bbox_inches='tight')
    print("\n贝叶斯优化过程可视化已保存到: bayesian_optimization_process.png")
    plt.show()


if __name__ == "__main__":
    # 运行示例
    visualize_optimization_process()
    optimizer1, result1 = optimize_svm_hyperparameters()
    optimizer2 = optimize_random_forest()