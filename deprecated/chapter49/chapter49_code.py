"""
第49章 优化理论：Python实现
优化算法工具包 - 可视化与实现
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Arrow
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 49.7 可视化凸函数与梯度下降
# =============================================================================

def visualize_convex_function():
    """可视化凸函数与非凸函数的对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.linspace(-3, 3, 200)
    
    # 凸函数: f(x) = x^2
    ax1 = axes[0]
    y_convex = x**2
    ax1.plot(x, y_convex, 'b-', linewidth=2.5, label='$f(x) = x^2$ (Convex)')
    
    # 画割线展示凸性
    x1, x2 = -2, 1.5
    y1, y2 = x1**2, x2**2
    ax1.plot([x1, x2], [y1, y2], 'r--', linewidth=2, label='Chord (above function)')
    ax1.scatter([x1, x2], [y1, y2], c='red', s=100, zorder=5)
    
    # 画切线
    x_tangent = 0.5
    y_tangent = x_tangent**2
    slope = 2 * x_tangent
    tangent_line = y_tangent + slope * (x - x_tangent)
    ax1.plot(x, tangent_line, 'g:', linewidth=2, label=f'Tangent at x={x_tangent}')
    ax1.scatter([x_tangent], [y_tangent], c='green', s=100, zorder=5)
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('Convex Function: Bowl Shape\nAll chords above function', fontsize=13)
    ax1.legend(loc='upper center')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, 10)
    
    # 非凸函数: f(x) = x^4 - 3x^2 + x
    ax2 = axes[1]
    y_nonconvex = x**4 - 3*x**2 + x
    ax2.plot(x, y_nonconvex, 'purple', linewidth=2.5, label='$f(x) = x^4 - 3x^2 + x$ (Non-convex)')
    
    # 标记局部最优和全局最优
    local_mins = [-1.3, 1.1]  # 近似值
    for xm in local_mins:
        ym = xm**4 - 3*xm**2 + xm
        ax2.scatter([xm], [ym], c='red', s=150, zorder=5, marker='v')
    
    ax2.annotate('Local min', xy=(-1.3, -2.5), xytext=(-2.5, -1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    ax2.annotate('Global min', xy=(1.1, -2.1), xytext=(2, -3.5),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('f(x)', fontsize=12)
    ax2.set_title('Non-Convex Function: Multiple Valleys\nGradient descent may get stuck', fontsize=13)
    ax2.legend(loc='upper center')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convex_vs_nonconvex.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ 凸函数与非凸函数对比可视化已保存")


# =============================================================================
# 49.8 梯度下降实现与收敛分析
# =============================================================================

class GradientDescent:
    """梯度下降优化器（含收敛分析）"""
    
    def __init__(self, learning_rate=0.1, max_iters=1000, tol=1e-6):
        self.lr = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.history = []
        
    def optimize(self, f, df, x0):
        """
        优化函数
        
        Parameters:
        -----------
        f : callable
            目标函数
        df : callable
            梯度函数
        x0 : ndarray
            初始点
            
        Returns:
        --------
        x_opt : ndarray
            最优解
        converged : bool
            是否收敛
        """
        x = np.array(x0, dtype=float)
        self.history = [{'x': x.copy(), 'f': f(x), 'grad': df(x)}]
        
        for i in range(self.max_iters):
            grad = df(x)
            
            # 检查收敛
            if np.linalg.norm(grad) < self.tol:
                return x, True
            
            # 梯度下降更新
            x = x - self.lr * grad
            
            self.history.append({
                'x': x.copy(),
                'f': f(x),
                'grad': grad,
                'grad_norm': np.linalg.norm(grad)
            })
        
        return x, False
    
    def plot_convergence(self, title="Gradient Descent Convergence"):
        """绘制收敛曲线"""
        iterations = range(len(self.history))
        f_values = [h['f'] for h in self.history]
        grad_norms = [h.get('grad_norm', np.linalg.norm(h['grad'])) for h in self.history]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 函数值收敛
        ax1 = axes[0]
        ax1.semilogy(iterations, np.abs(np.array(f_values) - min(f_values) + 1e-10), 'b-', linewidth=2)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('|f(x) - f*| (log scale)', fontsize=12)
        ax1.set_title('Function Value Convergence', fontsize=13)
        ax1.grid(True, alpha=0.3)
        
        # 梯度范数收敛
        ax2 = axes[1]
        ax2.semilogy(iterations, grad_norms, 'r-', linewidth=2)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('||∇f(x)|| (log scale)', fontsize=12)
        ax2.set_title('Gradient Norm Convergence', fontsize=13)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('gd_convergence.png', dpi=150, bbox_inches='tight')
        plt.show()


def demo_gradient_descent_2d():
    """二维梯度下降可视化"""
    # 定义二次函数: f(x,y) = x^2 + 2y^2 (条件数 = 2)
    f = lambda x: x[0]**2 + 2*x[1]**2
    df = lambda x: np.array([2*x[0], 4*x[1]])
    
    # 定义更病态的函数: f(x,y) = x^2 + 10y^2 (条件数 = 10)
    f_ill = lambda x: x[0]**2 + 10*x[1]**2
    df_ill = lambda x: np.array([2*x[0], 20*x[1]])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 良好条件问题
    ax1 = axes[0]
    plot_contour_with_path(ax1, f, df, x0=[2, 2], lr=0.1, 
                          title='Well-Conditioned (κ=2)', color='blue')
    
    # 病态问题
    ax2 = axes[1]
    plot_contour_with_path(ax2, f_ill, df_ill, x0=[2, 2], lr=0.05,
                          title='Ill-Conditioned (κ=10)', color='red')
    
    plt.tight_layout()
    plt.savefig('gd_2d_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ 二维梯度下降对比可视化已保存")


def plot_contour_with_path(ax, f, df, x0, lr, title, color):
    """绘制等高线并显示优化路径"""
    # 生成网格
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f([xi, yi]) for xi in x] for yi in y])
    
    # 绘制等高线
    levels = np.logspace(-1, 2, 15)
    ax.contour(X, Y, Z, levels=levels, alpha=0.6, cmap='viridis')
    
    # 运行梯度下降
    gd = GradientDescent(learning_rate=lr, max_iters=50)
    x_opt, converged = gd.optimize(f, df, x0)
    
    # 提取路径
    path = np.array([h['x'] for h in gd.history])
    
    # 绘制路径
    ax.plot(path[:, 0], path[:, 1], 'o-', color=color, markersize=4, linewidth=1.5, alpha=0.8)
    ax.plot(path[0, 0], path[0, 1], 'go', markersize=12, label='Start', zorder=5)
    ax.plot(path[-1, 0], path[-1, 1], 'r*', markersize=15, label='End', zorder=5)
    
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title(f'{title}\nIterations: {len(gd.history)-1}', fontsize=12)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


# =============================================================================
# 49.9 牛顿法与拟牛顿法
# =============================================================================

class NewtonMethod:
    """牛顿法优化器"""
    
    def __init__(self, max_iters=100, tol=1e-6):
        self.max_iters = max_iters
        self.tol = tol
        self.history = []
    
    def optimize(self, f, df, d2f, x0):
        """
        牛顿法优化
        
        Parameters:
        -----------
        f : callable - 目标函数
        df : callable - 梯度
        d2f : callable - Hessian矩阵
        x0 : ndarray - 初始点
        """
        x = np.array(x0, dtype=float)
        self.history = [{'x': x.copy(), 'f': f(x)}]
        
        for i in range(self.max_iters):
            grad = df(x)
            
            if np.linalg.norm(grad) < self.tol:
                return x, True
            
            hessian = d2f(x)
            
            # 求解牛顿方向: H * d = -grad
            try:
                d = np.linalg.solve(hessian, -grad)
            except np.linalg.LinAlgError:
                # Hessian奇异，使用梯度方向
                d = -grad
            
            # 线搜索（简单版本）
            alpha = 1.0
            while f(x + alpha * d) > f(x) + 0.1 * alpha * grad.dot(d):
                alpha *= 0.5
                if alpha < 1e-10:
                    break
            
            x = x + alpha * d
            self.history.append({'x': x.copy(), 'f': f(x), 'step': alpha})
        
        return x, False


class BFGS:
    """BFGS拟牛顿法"""
    
    def __init__(self, max_iters=1000, tol=1e-6):
        self.max_iters = max_iters
        self.tol = tol
        self.history = []
    
    def optimize(self, f, df, x0):
        """BFGS优化"""
        x = np.array(x0, dtype=float)
        n = len(x)
        
        # 初始Hessian逆近似为单位矩阵
        H = np.eye(n)
        
        self.history = [{'x': x.copy(), 'f': f(x)}]
        
        grad = df(x)
        
        for i in range(self.max_iters):
            if np.linalg.norm(grad) < self.tol:
                return x, True
            
            # 搜索方向
            p = -H.dot(grad)
            
            # 线搜索
            alpha = self._line_search(f, df, x, p, grad)
            
            # 更新
            s = alpha * p
            x_new = x + s
            grad_new = df(x_new)
            y = grad_new - grad
            
            # BFGS更新公式
            rho = 1.0 / (y.dot(s) + 1e-10)
            if rho > 0:  # 确保曲率条件
                I = np.eye(n)
                H = (I - rho * np.outer(s, y)).dot(H).dot(I - rho * np.outer(y, s)) + rho * np.outer(s, s)
            
            x = x_new
            grad = grad_new
            self.history.append({'x': x.copy(), 'f': f(x)})
        
        return x, False
    
    def _line_search(self, f, df, x, p, grad, c1=1e-4, c2=0.9):
        """Wolfe条件线搜索"""
        alpha = 1.0
        f_x = f(x)
        grad_p = grad.dot(p)
        
        for _ in range(20):
            x_new = x + alpha * p
            f_new = f(x_new)
            
            # Armijo条件
            if f_new <= f_x + c1 * alpha * grad_p:
                # 曲率条件
                grad_new = df(x_new)
                if grad_new.dot(p) >= c2 * grad_p:
                    return alpha
            
            alpha *= 0.5
        
        return alpha


def compare_optimization_methods():
    """对比不同优化方法"""
    # Rosenbrock函数（经典的优化测试函数）
    f = lambda x: (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    df = lambda x: np.array([
        -2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2),
        200*(x[1]-x[0]**2)
    ])
    d2f = lambda x: np.array([
        [2 + 1200*x[0]**2 - 400*x[1], -400*x[0]],
        [-400*x[0], 200]
    ])
    
    x0 = [-1.0, 2.0]
    
    # 梯度下降
    gd = GradientDescent(learning_rate=0.001, max_iters=10000)
    x_gd, conv_gd = gd.optimize(f, df, x0)
    
    # 牛顿法
    newton = NewtonMethod(max_iters=100)
    x_newton, conv_newton = newton.optimize(f, df, d2f, x0)
    
    # BFGS
    bfgs = BFGS(max_iters=1000)
    x_bfgs, conv_bfgs = bfgs.optimize(f, df, x0)
    
    print("\n" + "="*60)
    print("优化方法对比 (Rosenbrock函数)")
    print("="*60)
    print(f"{'Method':<15} {'Iterations':<12} {'Final f(x)':<15} {'Converged'}")
    print("-"*60)
    print(f"{'Gradient Descent':<15} {len(gd.history)-1:<12} {f(x_gd):<15.6e} {conv_gd}")
    print(f"{'Newton':<15} {len(newton.history)-1:<12} {f(x_newton):<15.6e} {conv_newton}")
    print(f"{'BFGS':<15} {len(bfgs.history)-1:<12} {f(x_bfgs):<15.6e} {conv_bfgs}")
    print("="*60)
    
    # 可视化收敛
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = [
        ('GD', gd.history, 'blue'),
        ('Newton', newton.history, 'red'),
        ('BFGS', bfgs.history, 'green')
    ]
    
    for name, hist, color in methods:
        f_vals = [h['f'] for h in hist]
        iterations = range(len(f_vals))
        ax.semilogy(iterations, np.array(f_vals) + 1e-10, 
                   label=name, color=color, linewidth=2, marker='o', markersize=3)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('f(x) - f* (log scale)', fontsize=12)
    ax.set_title('Optimization Methods Comparison on Rosenbrock Function', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimization_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ 优化方法对比已保存")


# =============================================================================
# 49.10 约束优化：投影梯度下降
# =============================================================================

def projection_onto_box(x, lower, upper):
    """投影到盒子约束 [lower, upper]"""
    return np.clip(x, lower, upper)

def projection_onto_ball(x, center, radius):
    """投影到球内"""
    diff = x - center
    norm = np.linalg.norm(diff)
    if norm <= radius:
        return x
    return center + radius * diff / norm

class ProjectedGradientDescent:
    """投影梯度下降"""
    
    def __init__(self, learning_rate=0.1, max_iters=1000, tol=1e-6):
        self.lr = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.history = []
    
    def optimize(self, f, df, x0, projection_fn):
        """
        投影梯度下降
        
        Parameters:
        -----------
        projection_fn : callable
            投影函数: x_proj = projection_fn(x)
        """
        x = np.array(x0, dtype=float)
        self.history = [{'x': x.copy(), 'f': f(x)}]
        
        for i in range(self.max_iters):
            grad = df(x)
            
            if np.linalg.norm(grad) < self.tol:
                return x, True
            
            # 梯度步 + 投影
            x_temp = x - self.lr * grad
            x_new = projection_fn(x_temp)
            
            x = x_new
            self.history.append({'x': x.copy(), 'f': f(x)})
        
        return x, False


def demo_constrained_optimization():
    """约束优化示例"""
    # 目标函数: f(x,y) = x^2 + y^2
    f = lambda x: x[0]**2 + x[1]**2
    df = lambda x: np.array([2*x[0], 2*x[1]])
    
    # 约束: x + y >= 1 (即 -x - y <= -1)
    # 使用投影梯度下降
    
    # 投影到半空间: x + y >= 1
    def project_halfspace(x):
        # x + y >= 1
        constraint = x[0] + x[1] - 1
        if constraint >= 0:
            return x
        # 投影
        normal = np.array([1, 1]) / np.sqrt(2)
        return x - constraint * normal
    
    # 从不可行点开始
    x0 = [0, 0]
    pgd = ProjectedGradientDescent(learning_rate=0.1, max_iters=100)
    x_opt, _ = pgd.optimize(f, df, x0, project_halfspace)
    
    print("\n" + "="*50)
    print("约束优化示例: min x² + y² s.t. x + y ≥ 1")
    print("="*50)
    print(f"初始点: {x0}")
    print(f"最优解: {x_opt}")
    print(f"约束值 x + y: {x_opt[0] + x_opt[1]:.6f} (应 ≥ 1)")
    print(f"最优值: {f(x_opt):.6f}")
    print("理论最优解: [0.5, 0.5], 最优值: 0.5")
    print("="*50)


# =============================================================================
# 49.11 拉格朗日乘子法实现
# =============================================================================

def solve_equality_constrained_qp(Q, c, A, b):
    """
    求解等式约束二次规划:
        min  0.5 * x'Qx + c'x
        s.t. Ax = b
    
    使用KKT条件求解
    """
    n = Q.shape[0]
    m = A.shape[0]
    
    # KKT系统:
    # [ Q   A' ] [ x ] = [ -c ]
    # [ A   0  ] [ λ ]   [  b ]
    
    KKT = np.block([
        [Q, A.T],
        [A, np.zeros((m, m))]
    ])
    rhs = np.concatenate([-c, b])
    
    try:
        sol = np.linalg.solve(KKT, rhs)
        x_opt = sol[:n]
        lam_opt = sol[n:]
        return x_opt, lam_opt
    except np.linalg.LinAlgError:
        return None, None


def demo_lagrange_multipliers():
    """拉格朗日乘子法示例"""
    # 问题: min 0.5*(x² + y²) s.t. x + y = 1
    # 解析解: x = y = 0.5
    
    Q = np.eye(2)
    c = np.zeros(2)
    A = np.array([[1, 1]])
    b = np.array([1])
    
    x_opt, lam_opt = solve_equality_constrained_qp(Q, c, A, b)
    
    print("\n" + "="*50)
    print("拉格朗日乘子法示例")
    print("="*50)
    print("问题: min 0.5*(x² + y²) s.t. x + y = 1")
    print("-"*50)
    print(f"最优解 x*: {x_opt}")
    print(f"拉格朗日乘子 λ*: {lam_opt[0]:.6f}")
    print(f"最优值: {0.5 * np.dot(x_opt, x_opt):.6f}")
    print(f"约束验证 x + y = {x_opt[0] + x_opt[1]:.6f}")
    print("="*50)
    
    # 验证KKT条件
    grad_f = Q.dot(x_opt) + c  # 目标函数梯度
    grad_h = A[0]  # 约束梯度
    
    print("\nKKT条件验证:")
    print(f"  ∇f(x*) + λ*∇h(x*) = {grad_f + lam_opt[0] * grad_h}")
    print(f"  (应接近 [0, 0])")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("第49章 优化理论 - Python实现")
    print("="*60)
    
    # 1. 可视化凸函数
    print("\n[1] 可视化凸函数与非凸函数...")
    visualize_convex_function()
    
    # 2. 二维梯度下降
    print("\n[2] 二维梯度下降演示...")
    demo_gradient_descent_2d()
    
    # 3. 优化方法对比
    print("\n[3] 优化方法对比...")
    compare_optimization_methods()
    
    # 4. 约束优化
    print("\n[4] 约束优化演示...")
    demo_constrained_optimization()
    
    # 5. 拉格朗日乘子法
    print("\n[5] 拉格朗日乘子法演示...")
    demo_lagrange_multipliers()
    
    print("\n" + "="*60)
    print("所有演示完成!")
    print("="*60)
