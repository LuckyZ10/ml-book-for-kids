"""
第七章：线性回归 - 纯Python实现
====================
不使用NumPy，只用基础Python
包含：
1. SimpleLinearRegression - 简单线性回归（最小二乘闭式解）
2. LinearRegressionGD - 梯度下降版线性回归
3. MultipleLinearRegression - 多元线性回归（正规方程）
4. 完整的矩阵运算实现

作者：ML教材示例代码
日期：2026-03-24
"""


class SimpleLinearRegression:
    """
    简单线性回归：y = w * x + b
    
    使用最小二乘法解析解
    """
    
    def __init__(self):
        self.w = 0  # 斜率（权重）
        self.b = 0  # 截距（偏置）
        self.mean_x = 0
        self.mean_y = 0
    
    def fit(self, X, y):
        """
        使用最小二乘法训练模型
        
        参数：
            X: 输入特征列表，如 [50, 80, 100, 120, 150]
            y: 目标值列表，如 [150, 220, 280, 320, 400]
        """
        n = len(X)
        
        # 计算均值
        self.mean_x = sum(X) / n
        self.mean_y = sum(y) / n
        
        # 计算斜率 w = cov(X, y) / var(X)
        numerator = 0  # 分子：协方差
        denominator = 0  # 分母：X的方差
        
        for i in range(n):
            numerator += (X[i] - self.mean_x) * (y[i] - self.mean_y)
            denominator += (X[i] - self.mean_x) ** 2
        
        self.w = numerator / denominator
        
        # 计算截距 b = mean_y - w * mean_x
        self.b = self.mean_y - self.w * self.mean_x
        
        return self
    
    def predict(self, X):
        """
        对输入数据进行预测
        
        参数：
            X: 输入特征列表或单个数值
            
        返回：
            预测值列表或单个数值
        """
        if isinstance(X, (int, float)):
            return self.w * X + self.b
        
        return [self.w * x + self.b for x in X]
    
    def score(self, X, y):
        """
        计算R²决定系数（拟合优度）
        
        R² = 1 - RSS/TSS
        
        其中：
        - RSS (Residual Sum of Squares) = 残差平方和
        - TSS (Total Sum of Squares) = 总平方和
        
        R²越接近1，表示模型拟合越好
        """
        y_pred = self.predict(X)
        n = len(y)
        
        # 计算 RSS
        rss = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
        
        # 计算 TSS
        tss = sum((y[i] - self.mean_y) ** 2 for i in range(n))
        
        if tss == 0:
            return 1.0  # 所有y值相同，完美拟合
        
        return 1 - rss / tss
    
    def correlation(self, X, y):
        """
        计算皮尔逊相关系数 r
        
        r = cov(X, y) / (std(X) * std(y))
        """
        n = len(X)
        mean_x = sum(X) / n
        mean_y = sum(y) / n
        
        # 计算协方差
        cov = sum((X[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        
        # 计算标准差
        var_x = sum((X[i] - mean_x) ** 2 for i in range(n))
        var_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        std_x = var_x ** 0.5
        std_y = var_y ** 0.5
        
        if std_x == 0 or std_y == 0:
            return 0
        
        return cov / (std_x * std_y)
    
    def __repr__(self):
        return f"SimpleLinearRegression(w={self.w:.4f}, b={self.b:.4f})"


class LinearRegressionGD:
    """
    线性回归 - 梯度下降版
    
    使用梯度下降迭代优化参数
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=False):
        """
        参数：
            learning_rate: 学习率α
            n_iterations: 迭代次数
            verbose: 是否打印训练过程
        """
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.verbose = verbose
        self.w = 0
        self.b = 0
        self.loss_history = []
    
    def fit(self, X, y):
        """
        使用梯度下降训练模型
        """
        n = len(X)
        
        # 初始化参数
        self.w = 0.0
        self.b = 0.0
        self.loss_history = []
        
        for i in range(self.n_iter):
            # 计算预测值
            y_pred = [self.w * X[j] + self.b for j in range(n)]
            
            # 计算损失（MSE）
            loss = sum((y[j] - y_pred[j]) ** 2 for j in range(n)) / n
            self.loss_history.append(loss)
            
            # 计算梯度
            # ∂J/∂w = -2/n * Σ(yᵢ - ŷᵢ) * xᵢ
            # ∂J/∂b = -2/n * Σ(yᵢ - ŷᵢ)
            grad_w = -2 / n * sum((y[j] - y_pred[j]) * X[j] for j in range(n))
            grad_b = -2 / n * sum(y[j] - y_pred[j] for j in range(n))
            
            # 更新参数
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            
            if self.verbose and i % 1000 == 0:
                print(f"迭代 {i}: loss = {loss:.4f}, w = {self.w:.4f}, b = {self.b:.4f}")
        
        return self
    
    def predict(self, X):
        """预测"""
        if isinstance(X, (int, float)):
            return self.w * X + self.b
        return [self.w * x + self.b for x in X]
    
    def score(self, X, y):
        """计算R²"""
        y_pred = self.predict(X)
        n = len(y)
        mean_y = sum(y) / n
        
        rss = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
        tss = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        return 1 - rss / tss if tss != 0 else 1.0
    
    def __repr__(self):
        return f"LinearRegressionGD(w={self.w:.4f}, b={self.b:.4f})"


class MultipleLinearRegression:
    """
    多元线性回归：y = w1*x1 + w2*x2 + ... + wd*xd + b
    
    使用正规方程求解：θ = (X^T X)^(-1) X^T y
    """
    
    def __init__(self):
        self.weights = []  # [w1, w2, ..., wd]
        self.bias = 0      # b
    
    def fit(self, X, y):
        """
        使用正规方程训练模型
        
        参数：
            X: 输入特征矩阵，如 [[50, 2], [80, 3], ...]  # [面积, 卧室数]
            y: 目标值列表，如 [150, 220, ...]
        """
        n = len(X)
        d = len(X[0]) if isinstance(X[0], (list, tuple)) else 1
        
        # 构建增广矩阵 X_augmented = [1, x1, x2, ..., xd]
        X_aug = []
        for i in range(n):
            row = [1]  # 偏置项
            if isinstance(X[i], (list, tuple)):
                row.extend(X[i])
            else:
                row.append(X[i])
            X_aug.append(row)
        
        # 计算 X^T X
        XtX = self._matrix_multiply(self._transpose(X_aug), X_aug)
        
        # 计算 X^T y
        Xty = self._matrix_vector_multiply(self._transpose(X_aug), y)
        
        # 计算 (X^T X)^(-1)
        XtX_inv = self._matrix_inverse(XtX)
        
        # 计算 θ = (X^T X)^(-1) X^T y
        theta = self._matrix_vector_multiply(XtX_inv, Xty)
        
        # 提取偏置和权重
        self.bias = theta[0]
        self.weights = theta[1:]
        
        return self
    
    def predict(self, X):
        """
        对输入数据进行预测
        """
        if isinstance(X, (int, float)):
            return self.bias + self.weights[0] * X
        
        if isinstance(X[0], (int, float)) and not isinstance(X[0], bool):
            # 单样本
            result = self.bias
            for i in range(len(X)):
                result += self.weights[i] * X[i]
            return result
        
        # 多样本
        results = []
        for x in X:
            result = self.bias
            for i in range(len(x)):
                result += self.weights[i] * x[i]
            results.append(result)
        return results
    
    def score(self, X, y):
        """计算R²决定系数"""
        y_pred = self.predict(X)
        n = len(y)
        mean_y = sum(y) / n
        
        rss = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
        tss = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        if tss == 0:
            return 1.0
        
        return 1 - rss / tss
    
    # ============ 矩阵运算辅助方法 ============
    
    def _transpose(self, matrix):
        """矩阵转置"""
        rows = len(matrix)
        cols = len(matrix[0])
        return [[matrix[i][j] for i in range(rows)] for j in range(cols)]
    
    def _matrix_multiply(self, A, B):
        """矩阵乘法 A @ B"""
        rows_A = len(A)
        cols_A = len(A[0])
        cols_B = len(B[0])
        
        result = [[0] * cols_B for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        return result
    
    def _matrix_vector_multiply(self, A, v):
        """矩阵与向量乘法 A @ v"""
        rows = len(A)
        result = []
        for i in range(rows):
            total = 0
            for j in range(len(v)):
                total += A[i][j] * v[j]
            result.append(total)
        return result
    
    def _matrix_inverse(self, matrix):
        """
        计算矩阵的逆（使用高斯-约当消元法）
        注意：这里假设矩阵是方阵且可逆
        """
        n = len(matrix)
        
        # 创建增广矩阵 [A | I]
        aug = []
        for i in range(n):
            row = matrix[i].copy()
            row.extend([1 if j == i else 0 for j in range(n)])
            aug.append(row)
        
        # 高斯-约当消元
        for i in range(n):
            # 找主元
            pivot = aug[i][i]
            if abs(pivot) < 1e-10:
                raise ValueError("矩阵不可逆（奇异矩阵）")
            
            # 归一化当前行
            for j in range(2 * n):
                aug[i][j] /= pivot
            
            # 消去其他行
            for k in range(n):
                if k != i:
                    factor = aug[k][i]
                    for j in range(2 * n):
                        aug[k][j] -= factor * aug[i][j]
        
        # 提取逆矩阵
        inv = []
        for i in range(n):
            inv.append([aug[i][j] for j in range(n, 2 * n)])
        
        return inv
    
    def __repr__(self):
        weights_str = ", ".join([f"{w:.4f}" for w in self.weights])
        return f"MultipleLinearRegression(weights=[{weights_str}], bias={self.bias:.4f})"


# ============== 测试代码 ==============

def demo_simple_linear():
    """演示简单线性回归"""
    print("=" * 70)
    print("🚀 简单线性回归 - 房价预测（最小二乘闭式解）")
    print("=" * 70)
    
    # 房价预测数据
    X_train = [50, 80, 100, 120, 150]  # 房屋面积（平方米）
    y_train = [150, 220, 280, 320, 400]  # 房价（万元）
    
    print(f"\n📊 训练数据：")
    print(f"   面积 (m²): {X_train}")
    print(f"   房价 (万元): {y_train}")
    
    # 创建并训练模型
    model = SimpleLinearRegression()
    model.fit(X_train, y_train)
    
    # 计算相关系数
    r = model.correlation(X_train, y_train)
    
    print(f"\n🎯 模型参数：")
    print(f"   斜率 w = {model.w:.4f}（每平米价格）")
    print(f"   截距 b = {model.b:.4f}（基础价格）")
    print(f"   相关系数 r = {r:.4f}")
    
    print(f"\n📝 回归方程：")
    print(f"   房价 = {model.w:.2f} × 面积 + {model.b:.2f}")
    
    # 预测
    X_test = [60, 90, 110, 180]
    y_pred = model.predict(X_test)
    
    print(f"\n🔮 预测结果：")
    for x, y in zip(X_test, y_pred):
        print(f"   {x} m² → {y:.2f} 万元")
    
    # 计算R²
    r2 = model.score(X_train, y_train)
    print(f"\n📈 拟合优度 R² = {r2:.4f}")
    print(f"   （R² = 1 表示完美拟合，R² = 0 表示不比直接用均值预测好）")


def demo_gradient_descent():
    """演示梯度下降训练"""
    print("\n" + "=" * 70)
    print("🚀 梯度下降训练线性回归")
    print("=" * 70)
    
    X_train = [50, 80, 100, 120, 150]
    y_train = [150, 220, 280, 320, 400]
    
    print(f"\n📊 相同数据，使用梯度下降训练：")
    print(f"   学习率 = 0.00001，迭代次数 = 50000")
    
    # 梯度下降训练（使用更小学习率）
    model_gd = LinearRegressionGD(learning_rate=0.00001, n_iterations=50000, verbose=False)
    model_gd.fit(X_train, y_train)
    
    print(f"\n🎯 训练结果：")
    print(f"   斜率 w = {model_gd.w:.4f}")
    print(f"   截距 b = {model_gd.b:.4f}")
    
    # 对比闭式解
    model_closed = SimpleLinearRegression()
    model_closed.fit(X_train, y_train)
    
    print(f"\n📋 对比：")
    print(f"   闭式解: w = {model_closed.w:.4f}, b = {model_closed.b:.4f}")
    print(f"   梯度下降: w = {model_gd.w:.4f}, b = {model_gd.b:.4f}")
    print(f"   差距: Δw = {abs(model_closed.w - model_gd.w):.6f}, Δb = {abs(model_closed.b - model_gd.b):.6f}")
    
    r2 = model_gd.score(X_train, y_train)
    print(f"\n📈 拟合优度 R² = {r2:.4f}")


def demo_multiple_linear():
    """演示多元线性回归"""
    print("\n" + "=" * 70)
    print("🚀 多元线性回归 - 多因素房价预测")
    print("=" * 70)
    
    # 数据：[面积(m²), 卧室数] → 房价(万元)
    X_train = [
        [50, 2],
        [80, 3],
        [100, 3],
        [120, 4],
        [150, 4]
    ]
    y_train = [150, 220, 280, 320, 400]
    
    print(f"\n📊 训练数据（面积, 卧室数 → 房价）：")
    for x, y in zip(X_train, y_train):
        print(f"   {x[0]}m², {x[1]}室 → {y}万元")
    
    # 训练模型
    model = MultipleLinearRegression()
    model.fit(X_train, y_train)
    
    print(f"\n🎯 模型参数：")
    print(f"   偏置 b = {model.bias:.4f}")
    for i, w in enumerate(model.weights):
        feature_name = ["面积", "卧室数"][i]
        print(f"   {feature_name} 权重 w{i+1} = {w:.4f}")
    
    print(f"\n📝 回归方程：")
    print(f"   房价 = {model.weights[0]:.2f} × 面积 + {model.weights[1]:.2f} × 卧室数 + {model.bias:.2f}")
    
    # 预测
    X_test = [[60, 2], [90, 3], [110, 3], [180, 5]]
    y_pred = model.predict(X_test)
    
    print(f"\n🔮 预测结果：")
    for x, y in zip(X_test, y_pred):
        print(f"   {x[0]}m², {x[1]}室 → {y:.2f} 万元")
    
    r2 = model.score(X_train, y_train)
    print(f"\n📈 拟合优度 R² = {r2:.4f}")


def demo_correlation():
    """演示相关系数的计算"""
    print("\n" + "=" * 70)
    print("📊 相关系数 r 的计算与解释")
    print("=" * 70)
    
    # 不同相关程度的数据
    datasets = [
        ("完全正相关", [1, 2, 3, 4, 5], [2, 4, 6, 8, 10]),
        ("强正相关", [1, 2, 3, 4, 5], [1, 2, 3, 5, 6]),
        ("弱正相关", [1, 2, 3, 4, 5], [1, 3, 2, 5, 4]),
        ("无相关", [1, 2, 3, 4, 5], [5, 2, 4, 1, 3]),
        ("负相关", [1, 2, 3, 4, 5], [10, 8, 6, 4, 2]),
    ]
    
    model = SimpleLinearRegression()
    
    for name, X, y in datasets:
        r = model.correlation(X, y)
        print(f"\n{name}:")
        print(f"   X = {X}")
        print(f"   y = {y}")
        print(f"   相关系数 r = {r:.4f}")


if __name__ == "__main__":
    demo_simple_linear()
    demo_gradient_descent()
    demo_multiple_linear()
    demo_correlation()
    
    print("\n" + "=" * 70)
    print("✅ 第七章学习完成！你已经掌握了线性回归的核心原理！")
    print("=" * 70)
