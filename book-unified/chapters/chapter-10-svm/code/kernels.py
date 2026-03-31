"""
核函数实现
包含线性核、多项式核、RBF核

《机器学习与深度学习：从小学生到大师》第十章配套代码
"""
import numpy as np


class Kernels:
    """核函数集合"""
    
    @staticmethod
    def linear():
        """
        线性核: K(x, x') = x · x'
        """
        def kernel(X1, X2):
            return np.dot(X1, X2.T)
        return kernel
    
    @staticmethod
    def polynomial(gamma=1.0, coef0=1.0, degree=3):
        """
        多项式核: K(x, x') = (γ · x·x' + r)^d
        
        参数:
            gamma: 缩放参数 γ
            coef0: 常数项 r
            degree: 多项式次数 d
        """
        def kernel(X1, X2):
            return (gamma * np.dot(X1, X2.T) + coef0) ** degree
        return kernel
    
    @staticmethod
    def rbf(gamma=1.0):
        """
        RBF（高斯径向基）核: K(x, x') = exp(-γ ||x - x'||²)
        
        参数:
            gamma: 控制高斯函数的宽度
                   越大 → 核函数越"尖锐" → 模型越复杂
                   越小 → 核函数越"平坦" → 模型越简单
        """
        def kernel(X1, X2):
            # 计算两两之间的欧氏距离平方
            # ||x - x'||² = ||x||² + ||x'||² - 2x·x'
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            
            # 距离平方矩阵
            dist_sq = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
            
            return np.exp(-gamma * dist_sq)
        return kernel
    
    @staticmethod
    def sigmoid(gamma=1.0, coef0=0.0):
        """
        Sigmoid核: K(x, x') = tanh(γ · x·x' + r)
        
        参数:
            gamma: 缩放参数
            coef0: 常数项
        """
        def kernel(X1, X2):
            return np.tanh(gamma * np.dot(X1, X2.T) + coef0)
        return kernel


def demo_kernels():
    """演示不同核函数的效果"""
    # 两个示例向量
    x1 = np.array([[1, 2]])
    x2 = np.array([[3, 4]])
    
    print("=" * 60)
    print("🔍 核函数演示")
    print("=" * 60)
    print(f"向量 x1 = {x1}")
    print(f"向量 x2 = {x2}")
    print(f"x1 和 x2 的距离 = {np.linalg.norm(x1 - x2):.4f}")
    print()
    
    # 线性核
    linear_k = Kernels.linear()
    print(f"📏 线性核 K(x1, x2) = {linear_k(x1, x2)[0, 0]:.4f}")
    print(f"   （就是两个向量的内积）")
    print()
    
    # 多项式核
    poly_k = Kernels.polynomial(gamma=1.0, coef0=1.0, degree=2)
    print(f"📐 多项式核(degree=2) K(x1, x2) = {poly_k(x1, x2)[0, 0]:.4f}")
    print(f"   （等价于映射到高维后的内积）")
    print()
    
    # RBF核
    rbf_k = Kernels.rbf(gamma=0.5)
    print(f"🔵 RBF核(gamma=0.5) K(x1, x2) = {rbf_k(x1, x2)[0, 0]:.4f}")
    print(f"   （距离越远，核函数值越小）")
    print()
    
    # 展示 gamma 对 RBF 的影响
    print("🎚️ gamma 参数对 RBF 核的影响:")
    print("-" * 40)
    for gamma in [0.1, 0.5, 1.0, 5.0, 10.0]:
        rbf = Kernels.rbf(gamma=gamma)
        value = rbf(x1, x2)[0, 0]
        print(f"  gamma={gamma:4.1f}: K(x1, x2) = {value:.6f}")
    print("\n  gamma 越大 → 核函数衰减越快 → 模型越"复杂"（容易过拟合）")


if __name__ == "__main__":
    demo_kernels()
