"""
SVM完整演示：比较不同核函数的效果

《机器学习与深度学习：从小学生到大师》第十章配套代码
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smo_svm import SimplifiedSMO


def compare_kernels():
    """比较不同核函数在各类数据集上的表现"""
    
    # 生成三种不同类型的数据集
    datasets = []
    
    # 1. 线性可分数据
    X1, y1 = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.2)
    y1 = np.where(y1 == 0, -1, 1)
    datasets.append(("🌻 线性可分数据", X1, y1))
    
    # 2. 同心圆数据（必须使用核函数）
    X2, y2 = make_circles(n_samples=100, factor=0.5, noise=0.08, random_state=42)
    y2 = np.where(y2 == 0, -1, 1)
    datasets.append(("⭐ 同心圆数据", X2, y2))
    
    # 3. 月亮数据
    X3, y3 = make_moons(n_samples=100, noise=0.15, random_state=42)
    y3 = np.where(y3 == 0, -1, 1)
    datasets.append(("🌙 月亮形状数据", X3, y3))
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for row, (data_name, X, y) in enumerate(datasets):
        print(f"\n{'='*60}")
        print(f"📊 数据集: {data_name}")
        print(f"{'='*60}")
        
        # 测试三种核函数
        configs = [
            ("线性核", "linear", {}),
            ("RBF核(γ=1)", "rbf", {"gamma": 1.0}),
            ("RBF核(γ=10)", "rbf", {"gamma": 10.0}),
        ]
        
        for col, (kernel_name, kernel_type, kernel_params) in enumerate(configs):
            print(f"\n  🔧 {kernel_name}")
            
            try:
                # 训练SMO SVM
                svm = SimplifiedSMO(X, y, C=1.0, kernel_type=kernel_type, 
                                   max_passes=50, **kernel_params)
                svm.fit()
                accuracy = svm.score(X, y)
                print(f"     准确率: {accuracy*100:.1f}%")
                
                # 绘制结果
                ax = axes[row, col]
                plot_decision_boundary(ax, X, y, svm, f"{data_name}\n{kernel_name}")
                
            except Exception as e:
                print(f"     错误: {e}")
                ax = axes[row, col]
                ax.text(0.5, 0.5, f"Error:\n{e}", ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10)
                ax.set_title(f"{data_name}\n{kernel_name}", fontsize=10)
    
    plt.tight_layout()
    plt.savefig('kernel_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n{'='*60}")
    print("🎉 所有测试完成！")
    print("观察结果：")
    print("  - 线性数据：线性核表现最好")
    print("  - 圆形/月亮数据：RBF核能处理非线性边界")
    print("  - gamma越大：决策边界越复杂，可能过拟合")
    print(f"{'='*60}")


def plot_decision_boundary(ax, X, y, svm, title):
    """绘制决策边界"""
    # 确定绘图范围
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # 创建网格
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # 预测网格点
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    # 绘制决策区域
    ax.contourf(xx, yy, Z, alpha=0.3, levels=[-2, 0, 2], 
               colors=['#4488ff', '#ff8844'])
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    
    # 绘制数据点
    pos_idx = y == 1
    neg_idx = y == -1
    ax.scatter(X[pos_idx, 0], X[pos_idx, 1], c='gold', s=50, 
              edgecolors='black', linewidths=1, label='+1', zorder=5)
    ax.scatter(X[neg_idx, 0], X[neg_idx, 1], c='navy', s=50, 
              edgecolors='white', linewidths=1, label='-1', zorder=5)
    
    # 绘制支持向量
    if len(svm.support_vector_idx) > 0:
        ax.scatter(X[svm.support_vector_idx, 0], X[svm.support_vector_idx, 1],
                  s=150, facecolors='none', edgecolors='red', 
                  linewidths=2, label='SV', zorder=6)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])


def demonstrate_margin():
    """演示最大间隔原理"""
    print("\n" + "="*60)
    print("📏 演示：最大间隔原理")
    print("="*60)
    
    # 生成线性可分数据
    np.random.seed(42)
    X_pos = np.random.randn(20, 2) + np.array([2, 2])
    X_neg = np.random.randn(20, 2) + np.array([-2, -2])
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(20), -np.ones(20)])
    
    # 使用不同的C值
    C_values = [0.01, 0.1, 1.0, 100.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, C in enumerate(C_values):
        print(f"\n  训练 C = {C}...")
        
        svm = SimplifiedSMO(X, y, C=C, kernel_type='linear', max_passes=50)
        svm.fit()
        
        ax = axes[i]
        
        # 计算权重向量 w
        w = np.sum((svm.alphas * y).reshape(-1, 1) * X, axis=0)
        margin = 2 / np.linalg.norm(w)
        
        # 绘制
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        xx = np.linspace(x_min, x_max, 100)
        
        # 决策边界和间隔边界
        if abs(w[1]) > 1e-10:
            yy_decision = -(w[0] * xx + svm.b) / w[1]
            yy_plus = -(w[0] * xx + svm.b - 1) / w[1]
            yy_minus = -(w[0] * xx + svm.b + 1) / w[1]
            
            ax.plot(xx, yy_decision, 'k-', linewidth=2, label='决策边界')
            ax.plot(xx, yy_plus, 'k--', linewidth=1, alpha=0.5, label='间隔边界')
            ax.plot(xx, yy_minus, 'k--', linewidth=1, alpha=0.5)
            ax.fill_between(xx, yy_minus, yy_plus, alpha=0.1, color='gray')
        
        # 数据点
        ax.scatter(X[:20, 0], X[:20, 1], c='gold', s=60, edgecolors='black', label='+1')
        ax.scatter(X[20:, 0], X[20:, 1], c='navy', s=60, edgecolors='black', label='-1')
        
        # 支持向量
        if len(svm.support_vector_idx) > 0:
            ax.scatter(X[svm.support_vector_idx, 0], X[svm.support_vector_idx, 1],
                      s=200, facecolors='none', edgecolors='red', linewidths=2)
        
        ax.set_title(f'C = {C}\n间隔宽度 = {margin:.3f}, 支持向量数 = {len(svm.support_vector_idx)}',
                    fontsize=12, fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('📊 C参数对间隔的影响', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('margin_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n结论：")
    print("  C越小 → 间隔越大 → 允许更多分类错误 → 模型更简单 → 可能欠拟合")
    print("  C越大 → 间隔越小 → 严格要求正确分类 → 模型更复杂 → 可能过拟合")


if __name__ == "__main__":
    # 运行所有演示
    compare_kernels()
    demonstrate_margin()
