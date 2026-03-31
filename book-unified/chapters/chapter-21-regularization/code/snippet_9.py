"""
第二十一章代码：正则化技术完整实现
========================================
包含：
- L1/L2正则化手动实现
- Dropout层完整实现（训练/推理模式切换）
- Batch Normalization层（含移动平均）
- EarlyStopping回调类
- 多项式拟合过拟合演示
- 正则化强度对比实验

作者：机器学习教材编写组
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Optional, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 第一部分：基础正则化实现
# ============================================================================

class L1Regularization:
    """
    L1正则化（Lasso）
    
    惩罚项: λ * Σ|θ|
    特点：产生稀疏解，自动特征选择
    """
    
    def __init__(self, lambda_reg: float = 0.01):
        """
        初始化L1正则化
        
        参数:
            lambda_reg: 正则化强度
        """
        self.lambda_reg = lambda_reg
    
    def compute_penalty(self, weights: np.ndarray) -> float:
        """
        计算L1惩罚项
        
        公式: λ * Σ|w|
        """
        return self.lambda_reg * np.sum(np.abs(weights))
    
    def compute_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        计算L1惩罚项的次梯度
        
        次梯度: λ * sign(w)
        在w=0时，次梯度是[-λ, λ]之间的任意值，这里取0
        """
        grad = self.lambda_reg * np.sign(weights)
        # 处理w=0的情况（可选：使用软阈值）
        return grad
    
    def soft_threshold(self, weights: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        软阈值操作（用于ISTA/FISTA算法）
        
        公式: S_λ(w) = sign(w) * max(|w| - λ, 0)
        """
        return np.sign(weights) * np.maximum(np.abs(weights) - self.lambda_reg * learning_rate, 0)


class L2Regularization:
    """
    L2正则化（Ridge / Weight Decay）
    
    惩罚项: λ * Σθ²
    特点：权重平滑衰减，解稳定
    """
    
    def __init__(self, lambda_reg: float = 0.01):
        """
        初始化L2正则化
        
        参数:
            lambda_reg: 正则化强度
        """
        self.lambda_reg = lambda_reg
    
    def compute_penalty(self, weights: np.ndarray) -> float:
        """
        计算L2惩罚项
        
        公式: λ * Σw²
        """
        return self.lambda_reg * np.sum(weights ** 2)
    
    def compute_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        计算L2惩罚项的梯度
        
        梯度: 2λ * w （通常简化为 λ * w）
        """
        return 2 * self.lambda_reg * weights
    
    def weight_decay(self, weights: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        权重衰减（Weight Decay）
        
        公式: w := w - η * 2λ * w = w * (1 - 2ηλ)
        """
        decay_factor = 1 - 2 * learning_rate * self.lambda_reg
        return weights * decay_factor


class ElasticNet:
    """
    Elastic Net正则化（L1 + L2的组合）
    
    惩罚项: λ₁ * Σ|θ| + λ₂ * Σθ²
    """
    
    def __init__(self, lambda_l1: float = 0.01, lambda_l2: float = 0.01):
        """
        初始化Elastic Net
        
        参数:
            lambda_l1: L1正则化强度
            lambda_l2: L2正则化强度
        """
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.l1 = L1Regularization(lambda_l1)
        self.l2 = L2Regularization(lambda_l2)
    
    def compute_penalty(self, weights: np.ndarray) -> float:
        """计算Elastic Net惩罚项"""
        return self.l1.compute_penalty(weights) + self.l2.compute_penalty(weights)
    
    def compute_gradient(self, weights: np.ndarray) -> np.ndarray:
        """计算Elastic Net梯度"""
        return self.l1.compute_gradient(weights) + self.l2.compute_gradient(weights)


# ============================================================================
# 第二部分：Dropout层实现
# ============================================================================

class Dropout:
    """
    Dropout层完整实现
    
    训练时：随机将部分神经元输出置零
    推理时：使用所有神经元，权重缩放
    """
    
    def __init__(self, dropout_rate: float = 0.5):
        """
        初始化Dropout层
        
        参数:
            dropout_rate: 丢弃概率（0-1之间）
                         0.5表示50%的神经元被丢弃
        """
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate必须在[0, 1)之间")
        
        self.dropout_rate = dropout_rate
        self.keep_rate = 1 - dropout_rate
        self.mask = None
        self.training = True
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        参数:
            x: 输入数组，形状 (batch_size, n_features)
        
        返回:
            输出数组
        """
        if self.training:
            # 训练时：生成随机掩码并应用
            self.mask = (np.random.rand(*x.shape) < self.keep_rate).astype(np.float32)
            # 反向Dropout：训练时缩放，推理时不缩放
            return x * self.mask / self.keep_rate
        else:
            # 推理时：使用所有神经元（已在训练时缩放）
            return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        反向传播
        
        参数:
            grad_output: 上游梯度
        
        返回:
            传递给下游的梯度
        """
        if self.training:
            # 只传播保留神经元的梯度
            return grad_output * self.mask / self.keep_rate
        else:
            return grad_output
    
    def train(self):
        """设置为训练模式"""
        self.training = True
    
    def eval(self):
        """设置为评估模式"""
        self.training = False


# ============================================================================
# 第三部分：Batch Normalization层实现
# ============================================================================

class BatchNormalization:
    """
    Batch Normalization层完整实现
    
    包含：
    - 训练时使用批量统计量
    - 推理时使用移动平均
    - 可学习的缩放(γ)和平移(β)参数
    """
    
    def __init__(self, n_features: int, momentum: float = 0.9, eps: float = 1e-5):
        """
        初始化Batch Normalization层
        
        参数:
            n_features: 特征数量
            momentum: 移动平均的动量系数
            eps: 数值稳定性常数
        """
        self.n_features = n_features
        self.momentum = momentum
        self.eps = eps
        
        # 可学习参数
        self.gamma = np.ones((1, n_features))  # 缩放参数
        self.beta = np.zeros((1, n_features))  # 平移参数
        
        # 移动平均统计量（用于推理）
        self.running_mean = np.zeros((1, n_features))
        self.running_var = np.ones((1, n_features))
        
        # 训练缓存
        self.training = True
        self.batch_mean = None
        self.batch_var = None
        self.x_normalized = None
        self.x_centered = None
        self.std_inv = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        参数:
            x: 输入，形状 (batch_size, n_features)
        
        返回:
            归一化后的输出
        """
        if self.training:
            # 训练模式：使用批量统计量
            self.batch_mean = np.mean(x, axis=0, keepdims=True)
            self.batch_var = np.var(x, axis=0, keepdims=True)
            
            # 更新移动平均
            self.running_mean = self.momentum * self.running_mean + \
                               (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + \
                              (1 - self.momentum) * self.batch_var
            
            # 标准化
            self.x_centered = x - self.batch_mean
            self.std_inv = 1.0 / np.sqrt(self.batch_var + self.eps)
            self.x_normalized = self.x_centered * self.std_inv
            
            # 缩放和平移
            out = self.gamma * self.x_normalized + self.beta
            
        else:
            # 推理模式：使用移动平均
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_normalized + self.beta
        
        return out
    
    def backward(self, grad_output: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        反向传播
        
        参数:
            grad_output: 上游梯度，形状 (batch_size, n_features)
            x: 前向传播时的输入
        
        返回:
            (dx, dgamma, dbeta)
        """
        batch_size = x.shape[0]
        
        # 关于gamma和beta的梯度
        dgamma = np.sum(grad_output * self.x_normalized, axis=0, keepdims=True)
        dbeta = np.sum(grad_output, axis=0, keepdims=True)
        
        # 关于x_normalized的梯度
        dx_normalized = grad_output * self.gamma
        
        # 关于方差的梯度
        dvar = np.sum(dx_normalized * self.x_centered * (-0.5) * (self.std_inv ** 3), axis=0, keepdims=True)
        
        # 关于均值的梯度
        dmean = np.sum(dx_normalized * (-self.std_inv), axis=0, keepdims=True) + \
                dvar * np.mean(-2 * self.x_centered, axis=0, keepdims=True)
        
        # 关于x的梯度
        dx = dx_normalized * self.std_inv + dvar * 2 * self.x_centered / batch_size + dmean / batch_size
        
        return dx, dgamma, dbeta
    
    def train(self):
        """设置为训练模式"""
        self.training = True
    
    def eval(self):
        """设置为评估模式"""
        self.training = False
    
    def update_params(self, dgamma: np.ndarray, dbeta: np.ndarray, learning_rate: float):
        """更新参数"""
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta


# ============================================================================
# 第四部分：Early Stopping回调
# ============================================================================

class EarlyStopping:
    """
    早停机制
    
    监控验证集性能，当连续多轮没有改善时停止训练
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        """
        初始化Early Stopping
        
        参数:
            patience: 耐心值，连续多少轮没有改善就停止
            min_delta: 最小改善量，小于此值不算改善
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0
        self.should_stop = False
    
    def __call__(self, val_loss: float, model_weights: Optional[Dict] = None) -> bool:
        """
        检查是否应该停止训练
        
        参数:
            val_loss: 当前验证损失
            model_weights: 当前模型权重（用于恢复最佳）
        
        返回:
            True如果应该停止训练
        """
        if val_loss < self.best_loss - self.min_delta:
            # 有改善
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights and model_weights is not None:
                self.best_weights = {k: v.copy() for k, v in model_weights.items()}
        else:
            # 没有改善
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False
    
    def get_best_weights(self) -> Optional[Dict]:
        """获取最佳权重"""
        return self.best_weights
    
    def reset(self):
        """重置状态"""
        self.best_loss = float('inf')
        self.best_weights = None
        self.counter = 0
        self.should_stop = False


# ============================================================================
# 第五部分：多项式拟合演示（过拟合/正则化可视化）
# ============================================================================

class PolynomialFitter:
    """
    多项式拟合器（用于演示过拟合和正则化）
    """
    
    def __init__(self, degree: int):
        """
        初始化
        
        参数:
            degree: 多项式次数
        """
        self.degree = degree
        self.weights = None
        self.regularizer = None
    
    def set_regularizer(self, regularizer):
        """设置正则化器"""
        self.regularizer = regularizer
    
    def _design_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        构造设计矩阵（Vandermonde矩阵）
        
        X = [1, x, x², ..., xⁿ]
        """
        return np.vander(x, self.degree + 1, increasing=True)
    
    def fit(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.01, 
            epochs: int = 10000, verbose: bool = False) -> List[float]:
        """
        使用梯度下降拟合
        
        参数:
            x: 输入数据
            y: 目标数据
            learning_rate: 学习率
            epochs: 迭代次数
            verbose: 是否打印进度
        
        返回:
            损失历史
        """
        X = self._design_matrix(x)
        n_samples = len(x)
        
        # 初始化权重
        self.weights = np.random.randn(self.degree + 1) * 0.01
        
        losses = []
        
        for epoch in range(epochs):
            # 前向传播
            predictions = X @ self.weights
            
            # 计算损失（均方误差）
            mse_loss = np.mean((predictions - y) ** 2)
            
            # 添加正则化项
            reg_loss = 0
            if self.regularizer is not None:
                reg_loss = self.regularizer.compute_penalty(self.weights)
            
            total_loss = mse_loss + reg_loss
            losses.append(total_loss)
            
            # 计算梯度
            grad_mse = (2 / n_samples) * X.T @ (predictions - y)
            
            if self.regularizer is not None:
                grad_reg = self.regularizer.compute_gradient(self.weights)
                grad = grad_mse + grad_reg
            else:
                grad = grad_mse
            
            # 更新权重
            self.weights -= learning_rate * grad
            
            if verbose and epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.6f}")
        
        return losses
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测"""
        X = self._design_matrix(x)
        return X @ self.weights
    
    def get_weights(self) -> np.ndarray:
        """获取权重"""
        return self.weights.copy()


def demonstrate_overfitting():
    """
    演示过拟合现象和正则化的效果
    """
    # 设置随机种子
    np.random.seed(42)
    
    # 生成数据：真实的函数是二次函数
    def true_function(x):
        return 0.5 * x**2 - 2 * x + 1
    
    # 生成训练数据（带噪声）
    n_train = 15
    x_train = np.linspace(-3, 3, n_train)
    y_train = true_function(x_train) + np.random.randn(n_train) * 0.5
    
    # 生成测试数据（无噪声）
    x_test = np.linspace(-3, 3, 100)
    y_test = true_function(x_test)
    
    # 测试不同复杂度的模型
    degrees = [2, 5, 15]
    lambdas = [0, 0.001, 0.01, 0.1]
    
    fig, axes = plt.subplots(len(degrees), len(lambdas), figsize=(16, 12))
    fig.suptitle('Polynomial Fitting: Effect of Model Complexity and Regularization', fontsize=14)
    
    for i, degree in enumerate(degrees):
        for j, lambda_reg in enumerate(lambdas):
            ax = axes[i, j]
            
            # 创建拟合器
            fitter = PolynomialFitter(degree)
            
            if lambda_reg > 0:
                fitter.set_regularizer(L2Regularization(lambda_reg))
            
            # 拟合
            fitter.fit(x_train, y_train, learning_rate=0.01, epochs=5000)
            
            # 预测
            x_plot = np.linspace(-3.5, 3.5, 200)
            y_pred = fitter.predict(x_plot)
            y_train_pred = fitter.predict(x_train)
            
            # 计算误差
            train_error = np.mean((y_train_pred - y_train) ** 2)
            test_error = np.mean((fitter.predict(x_test) - y_test) ** 2)
            
            # 绘图
            ax.scatter(x_train, y_train, c='red', s=50, zorder=3, label='Training Data')
            ax.plot(x_test, y_test, 'g-', linewidth=2, label='True Function', alpha=0.7)
            ax.plot(x_plot, y_pred, 'b-', linewidth=2, label='Fitted Curve')
            
            # 设置标题
            reg_text = f"λ={lambda_reg}" if lambda_reg > 0 else "No Regularization"
            ax.set_title(f'Degree={degree}, {reg_text}\n'
                        f'Train Error: {train_error:.3f}, Test Error: {test_error:.3f}',
                        fontsize=10)
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-3, 8)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regularization_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("演示图已保存为 'regularization_demo.png'")


# ============================================================================
# 第六部分：权重分布可视化
# ============================================================================

def compare_l1_l2_weights():
    """
    比较L1和L2正则化对权重分布的影响
    """
    np.random.seed(42)
    
    # 生成数据
    n_samples, n_features = 100, 50
    X = np.random.randn(n_samples, n_features)
    
    # 真实权重（稀疏）
    true_weights = np.zeros(n_features)
    true_weights[[0, 5, 10, 15, 20]] = [2, -1.5, 3, -2, 1]
    
    y = X @ true_weights + np.random.randn(n_samples) * 0.5
    
    # 训练/测试分割
    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 训练函数
    def train_with_regularization(X, y, regularizer, epochs=5000, lr=0.01):
        weights = np.random.randn(X.shape[1]) * 0.1
        for _ in range(epochs):
            pred = X @ weights
            grad = (2 / len(y)) * X.T @ (pred - y)
            if regularizer:
                grad += regularizer.compute_gradient(weights)
            weights -= lr * grad
        return weights
    
    # 训练不同正则化的模型
    reg_none = None
    reg_l1_001 = L1Regularization(0.01)
    reg_l1_01 = L1Regularization(0.1)
    reg_l2_001 = L2Regularization(0.01)
    reg_l2_01 = L2Regularization(0.1)
    
    weights_none = train_with_regularization(X_train, y_train, reg_none)
    weights_l1_001 = train_with_regularization(X_train, y_train, reg_l1_001)
    weights_l1_01 = train_with_regularization(X_train, y_train, reg_l1_01)
    weights_l2_001 = train_with_regularization(X_train, y_train, reg_l2_001)
    weights_l2_01 = train_with_regularization(X_train, y_train, reg_l2_01)
    
    # 计算测试误差
    def mse(X, y, w):
        return np.mean((X @ w - y) ** 2)
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Weight Distribution: L1 vs L2 Regularization', fontsize=14)
    
    methods = [
        ('No Regularization', weights_none, 'gray'),
        ('L1 (λ=0.01)', weights_l1_001, 'blue'),
        ('L1 (λ=0.1)', weights_l1_01, 'darkblue'),
        ('L2 (λ=0.01)', weights_l2_001, 'red'),
        ('L2 (λ=0.1)', weights_l2_01, 'darkred'),
    ]
    
    # 权重分布直方图
    ax = axes[0, 0]
    for name, weights, color in methods:
        ax.hist(weights, bins=20, alpha=0.5, label=name, color=color)
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Weight Distribution Histogram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 权重绝对值条形图
    ax = axes[0, 1]
    x_pos = np.arange(n_features)
    width = 0.15
    for idx, (name, weights, color) in enumerate(methods):
        ax.bar(x_pos + idx * width, np.abs(weights), width, 
               label=name, color=color, alpha=0.7)
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('|Weight|')
    ax.set_title('Absolute Weight Values')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 零权重数量对比
    ax = axes[0, 2]
    names = [m[0] for m in methods]
    zero_counts = [np.sum(np.abs(m[1]) < 0.01) for m in methods]
    colors_list = [m[2] for m in methods]
    bars = ax.bar(range(len(names)), zero_counts, color=colors_list, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Number of Near-Zero Weights')
    ax.set_title('Sparsity Comparison (|w| < 0.01)')
    ax.grid(True, alpha=0.3)
    
    # L1权重路径
    ax = axes[1, 0]
    for i in range(min(10, n_features)):
        weights_path = []
        reg = L1Regularization(0.05)
        w = np.random.randn(n_features) * 0.1
        for _ in range(2000):
            pred = X_train @ w
            grad = (2 / len(y_train)) * X_train.T @ (pred - y_train)
            grad += reg.compute_gradient(w)
            w -= 0.01 * grad
            weights_path.append(w[i])
        ax.plot(weights_path, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weight Value')
    ax.set_title('L1: Weight Paths During Training')
    ax.grid(True, alpha=0.3)
    
    # L2权重路径
    ax = axes[1, 1]
    for i in range(min(10, n_features)):
        weights_path = []
        reg = L2Regularization(0.05)
        w = np.random.randn(n_features) * 0.1
        for _ in range(2000):
            pred = X_train @ w
            grad = (2 / len(y_train)) * X_train.T @ (pred - y_train)
            grad += reg.compute_gradient(w)
            w -= 0.01 * grad
            weights_path.append(w[i])
        ax.plot(weights_path, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weight Value')
    ax.set_title('L2: Weight Paths During Training')
    ax.grid(True, alpha=0.3)
    
    # 测试误差对比
    ax = axes[1, 2]
    test_errors = [mse(X_test, y_test, m[1]) for m in methods]
    bars = ax.bar(range(len(names)), test_errors, color=colors_list, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Test MSE')
    ax.set_title('Test Error Comparison')
    ax.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值
    for bar, error in zip(bars, test_errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('l1_l2_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("对比图已保存为 'l1_l2_comparison.png'")


# ============================================================================
# 第七部分：Dropout效果演示
# ============================================================================

def demonstrate_dropout():
    """
    演示Dropout的效果
    """
    np.random.seed(42)
    
    # 生成数据
    n_samples = 200
    X = np.random.randn(n_samples, 2)
    # 创建圆形决策边界
    y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int)
    
    # 添加噪声
    y = y ^ (np.random.rand(n_samples) < 0.1)
    
    # 划分训练/测试集
    train_size = 150
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 简单的两层神经网络
    class SimpleNN:
        def __init__(self, hidden_size=50, dropout_rate=0.0):
            self.W1 = np.random.randn(2, hidden_size) * 0.1
            self.b1 = np.zeros(hidden_size)
            self.W2 = np.random.randn(hidden_size, 1) * 0.1
            self.b2 = np.zeros(1)
            self.dropout = Dropout(dropout_rate) if dropout_rate > 0 else None
            self.losses_train = []
            self.losses_val = []
        
        def forward(self, X, training=True):
            self.z1 = X @ self.W1 + self.b1
            self.a1 = np.maximum(0, self.z1)  # ReLU
            
            if self.dropout and training:
                self.a1 = self.dropout.forward(self.a1)
            
            self.z2 = self.a1 @ self.W2 + self.b2
            self.a2 = 1 / (1 + np.exp(-self.z2))  # Sigmoid
            return self.a2
        
        def backward(self, X, y, learning_rate):
            m = X.shape[0]
            
            # 输出层梯度
            dz2 = self.a2 - y.reshape(-1, 1)
            dW2 = (self.a1.T @ dz2) / m
            db2 = np.sum(dz2, axis=0) / m
            
            # 隐藏层梯度
            da1 = dz2 @ self.W2.T
            if self.dropout:
                da1 = self.dropout.backward(da1)
            
            dz1 = da1 * (self.z1 > 0)  # ReLU导数
            dW1 = (X.T @ dz1) / m
            db1 = np.sum(dz1, axis=0) / m
            
            # 更新参数
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
        
        def train(self, X, y, X_val, y_val, epochs=1000, lr=0.1):
            for epoch in range(epochs):
                # 训练
                if self.dropout:
                    self.dropout.train()
                pred_train = self.forward(X, training=True)
                loss_train = -np.mean(y * np.log(pred_train + 1e-8) + 
                                     (1-y) * np.log(1 - pred_train + 1e-8))
                self.backward(X, y, lr)
                
                # 验证
                if self.dropout:
                    self.dropout.eval()
                pred_val = self.forward(X_val, training=False)
                loss_val = -np.mean(y_val * np.log(pred_val + 1e-8) + 
                                   (1-y_val) * np.log(1 - pred_val + 1e-8))
                
                self.losses_train.append(loss_train)
                self.losses_val.append(loss_val)
        
        def predict(self, X):
            if self.dropout:
                self.dropout.eval()
            return (self.forward(X, training=False) > 0.5).astype(int)
    
    # 训练不同dropout率的模型
    dropout_rates = [0.0, 0.3, 0.5, 0.7]
    models = []
    
    for rate in dropout_rates:
        model = SimpleNN(hidden_size=50, dropout_rate=rate)
        model.train(X_train, y_train, X_test, y_test, epochs=1000, lr=0.1)
        models.append(model)
    
    # 可视化
    fig, axes = plt.subplots(2, len(dropout_rates), figsize=(16, 10))
    fig.suptitle('Dropout Effect on Training and Generalization', fontsize=14)
    
    for idx, (rate, model) in enumerate(zip(dropout_rates, models)):
        # 训练曲线
        ax = axes[0, idx]
        ax.plot(model.losses_train, label='Train Loss', alpha=0.7)
        ax.plot(model.losses_val, label='Val Loss', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Dropout Rate = {rate}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 决策边界
        ax = axes[1, idx]
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', 
                  edgecolors='k', s=30, label='Train')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', 
                  marker='s', edgecolors='k', s=30, label='Test')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_title(f'Decision Boundary (Dropout={rate})')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('dropout_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Dropout演示图已保存为 'dropout_demo.png'")


# ============================================================================
# 第八部分：完整运行演示
# ============================================================================

def run_all_demos():
    """
    运行所有演示
    """
    print("=" * 60)
    print("第二十一章：正则化技术完整演示")
    print("=" * 60)
    
    print("\n1. 演示过拟合和正则化效果...")
    demonstrate_overfitting()
    
    print("\n2. 比较L1和L2正则化...")
    compare_l1_l2_weights()
    
    print("\n3. 演示Dropout效果...")
    demonstrate_dropout()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)


# ============================================================================
# 单元测试
# ============================================================================

def test_regularization():
    """
    测试正则化实现
    """
    print("\n" + "=" * 40)
    print("单元测试")
    print("=" * 40)
    
    # 测试L1正则化
    l1 = L1Regularization(lambda_reg=0.1)
    weights = np.array([1.0, -2.0, 0.0, 3.0])
    penalty = l1.compute_penalty(weights)
    assert abs(penalty - 0.6) < 1e-6, f"L1 penalty计算错误: {penalty}"
    print("✓ L1正则化测试通过")
    
    # 测试L2正则化
    l2 = L2Regularization(lambda_reg=0.1)
    penalty = l2.compute_penalty(weights)
    assert abs(penalty - 1.4) < 1e-6, f"L2 penalty计算错误: {penalty}"
    print("✓ L2正则化测试通过")
    
    # 测试Dropout
    dropout = Dropout(dropout_rate=0.5)
    x = np.ones((1000, 10))
    dropout.train()
    y = dropout.forward(x)
    # 大约50%的神经元被保留
    keep_ratio = np.mean(y > 0)
    assert 0.4 < keep_ratio < 0.6, f"Dropout比例异常: {keep_ratio}"
    print("✓ Dropout测试通过")
    
    # 测试BatchNorm
    bn = BatchNormalization(n_features=5)
    x = np.random.randn(32, 5)
    bn.train()
    y = bn.forward(x)
    # 输出应该近似均值为0，方差为1（考虑gamma=1, beta=0）
    assert np.abs(np.mean(y)) < 0.1, "BatchNorm均值异常"
    print("✓ BatchNorm测试通过")
    
    print("=" * 40)
    print("所有测试通过！")
    print("=" * 40)


if __name__ == "__main__":
    # 运行单元测试
    test_regularization()
    
    # 运行完整演示
    run_all_demos()