"""
线性SVM - 简化实现
使用次梯度下降法优化软间隔目标函数

《机器学习与深度学习：从小学生到大师》第十章配套代码
"""
import numpy as np
import matplotlib.pyplot as plt


class LinearSVM:
    """
    线性支持向量机
    
    参数:
        C: 正则化参数（越大越严格）
        learning_rate: 学习率
        n_iterations: 迭代次数
    """
    def __init__(self, C=1.0, learning_rate=0.001, n_iterations=1000):
        self.C = C
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = None  # 权重向量
        self.b = None  # 偏置项
        
    def fit(self, X, y):
        """
        训练SVM
        
        参数:
            X: 训练数据，形状 (n_samples, n_features)
            y: 标签，形状 (n_samples,)，取值为 +1 或 -1
        """
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.w = np.zeros(n_features)
        self.b = 0
        
        # 梯度下降优化
        for iteration in range(self.n_iterations):
            # 计算每个样本的约束违反情况
            margins = y * (np.dot(X, self.w) + self.b)
            
            # 找出违反约束的样本（margin < 1）
            misclassified = margins < 1
            
            # 计算 w 的梯度
            # ∇_w = w - C * Σ(y_i * x_i) for misclassified
            grad_w = self.w - self.C * np.sum((y[misclassified][:, None] * X[misclassified]), axis=0) / n_samples
            
            # 计算 b 的梯度
            grad_b = -self.C * np.sum(y[misclassified]) / n_samples
            
            # 更新参数
            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b
            
            # 每100次迭代打印一次损失
            if (iteration + 1) % 100 == 0:
                loss = self._compute_loss(X, y)
                print(f"Iteration {iteration + 1}/{self.n_iterations}, Loss: {loss:.4f}")
    
    def _compute_loss(self, X, y):
        """计算 hinge loss + L2 正则化的目标函数值"""
        # Hinge loss: max(0, 1 - y * (w·x + b))
        margins = y * (np.dot(X, self.w) + self.b)
        hinge_loss = np.maximum(0, 1 - margins)
        
        # 总损失 = 0.5 * ||w||^2 + C * Σ hinge_loss
        loss = 0.5 * np.dot(self.w, self.w) + self.C * np.sum(hinge_loss)
        return loss
    
    def predict(self, X):
        """
        预测类别
        
        参数:
            X: 测试数据
            
        返回:
            预测标签 (+1 或 -1)
        """
        scores = np.dot(X, self.w) + self.b
        return np.sign(scores)
    
    def decision_function(self, X):
        """
        计算决策函数值（到超平面的有符号距离）
        
        参数:
            X: 测试数据
            
        返回:
            决策函数值
        """
        return np.dot(X, self.w) + self.b
    
    def get_support_vectors(self, X, y, tolerance=1e-5):
        """
        获取支持向量（距离决策边界最近的点）
        
        参数:
            X: 数据
            y: 标签
            tolerance: 判定为支持向量的阈值
            
        返回:
            支持向量的索引
        """
        margins = np.abs(y * self.decision_function(X) - 1)
        return np.where(margins < tolerance)[0]


def visualize_linear_svm():
    """可视化线性SVM的分类效果"""
    np.random.seed(42)
    
    # 生成线性可分的数据
    # 向日葵班（类别 +1）
    X_sunflower = np.random.randn(50, 2) + np.array([2, 2])
    # 星空班（类别 -1）
    X_starry = np.random.randn(50, 2) + np.array([-2, -2])
    
    X = np.vstack([X_sunflower, X_starry])
    y = np.hstack([np.ones(50), -np.ones(50)])
    
    # 训练SVM
    svm = LinearSVM(C=1.0, learning_rate=0.01, n_iterations=1000)
    svm.fit(X, y)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    
    # 绘制数据点
    plt.scatter(X[:50, 0], X[:50, 1], c='gold', s=100, marker='o', 
                edgecolors='black', label='🌻 向日葵班 (+1)', alpha=0.8)
    plt.scatter(X[50:, 0], X[50:, 1], c='navy', s=100, marker='s', 
                edgecolors='black', label='⭐ 星空班 (-1)', alpha=0.8)
    
    # 获取支持向量
    sv_indices = svm.get_support_vectors(X, y, tolerance=0.1)
    if len(sv_indices) > 0:
        plt.scatter(X[sv_indices, 0], X[sv_indices, 1], s=300, 
                   facecolors='none', edgecolors='red', linewidths=2,
                   label='🔴 支持向量')
    
    # 绘制决策边界和间隔边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx = np.linspace(x_min, x_max, 100)
    
    # 决策边界: w·x + b = 0  =>  y = -(w[0]*x + b) / w[1]
    yy_decision = -(svm.w[0] * xx + svm.b) / svm.w[1]
    # 间隔边界: w·x + b = ±1
    yy_plus = -(svm.w[0] * xx + svm.b - 1) / svm.w[1]
    yy_minus = -(svm.w[0] * xx + svm.b + 1) / svm.w[1]
    
    plt.plot(xx, yy_decision, 'k-', linewidth=2, label='决策边界')
    plt.plot(xx, yy_plus, 'k--', linewidth=1, alpha=0.5, label='间隔边界')
    plt.plot(xx, yy_minus, 'k--', linewidth=1, alpha=0.5)
    
    # 填充间隔区域
    plt.fill_between(xx, yy_minus, yy_plus, alpha=0.1, color='gray', label='间隔区域')
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('位置 x₁', fontsize=12)
    plt.ylabel('位置 x₂', fontsize=12)
    plt.title('🎓 线性SVM：寻找最宽的走道', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('linear_svm.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n训练完成！")
    print(f"权重向量 w = {svm.w}")
    print(f"偏置项 b = {svm.b:.4f}")
    print(f"间隔宽度 = {2 / np.linalg.norm(svm.w):.4f}")
    print(f"支持向量数量 = {len(sv_indices)}")
    
    return svm


if __name__ == "__main__":
    print("=" * 60)
    print("🌻 Linear SVM Demo - 寻找最宽的走道 🌻")
    print("=" * 60)
    svm = visualize_linear_svm()
