#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四章：一步一步变得更好——梯度下降的直觉
配套代码：从零实现各种梯度下降算法

作者：机器学习教材编写组
日期：2026-03-24
"""

import random
import math
from typing import Callable, List, Tuple, Optional


# ============================================================================
# 第一部分：基础工具函数
# ============================================================================

def numerical_derivative(f: Callable[[float], float], x: float, h: float = 1e-7) -> float:
    """
    数值微分：计算函数f在点x处的导数
    使用中心差分公式：f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f: Callable[[List[float]], float], x: List[float], h: float = 1e-7) -> List[float]:
    """
    数值梯度：计算多元函数f在点x处的梯度
    """
    n = len(x)
    grad = [0.0] * n
    for i in range(n):
        # 在x[i]方向上扰动
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad


def visualize_1d_trajectory(trajectory: List[Tuple[int, float, float]], width: int = 60) -> str:
    """
    可视化1D优化轨迹（ASCII艺术）
    """
    if not trajectory:
        return "Empty trajectory"
    
    min_x = min(t[1] for t in trajectory)
    max_x = max(t[1] for t in trajectory)
    min_f = min(t[2] for t in trajectory)
    max_f = max(t[2] for t in trajectory)
    
    # 防止除零
    x_range = max_x - min_x if max_x != min_x else 1.0
    f_range = max_f - min_f if max_f != min_f else 1.0
    
    lines = ["=" * 70]
    lines.append("优化轨迹可视化（1D）")
    lines.append("X轴：参数值    *表示当前位置    o表示最优解附近")
    lines.append("=" * 70)
    
    for step, x, fx in trajectory[:20]:  # 只显示前20步
        # 将x映射到0-width范围内
        pos = int((x - min_x) / x_range * width) if x_range > 0 else width // 2
        pos = max(0, min(pos, width))
        
        line = f"Step {step:3d}: f={fx:10.4f} |"
        line += " " * pos + "*"
        lines.append(line)
    
    if len(trajectory) > 20:
        lines.append(f"... (共{len(trajectory)}步)")
    
    return "\n".join(lines)


def visualize_loss_curve(losses: List[float], width: int = 50, height: int = 15) -> str:
    """
    可视化损失函数下降曲线（ASCII艺术）
    """
    if not losses:
        return "No loss data"
    
    min_loss = min(losses)
    max_loss = max(losses)
    loss_range = max_loss - min_loss if max_loss != min_loss else 1.0
    
    # 构建画布
    canvas = [[' ' for _ in range(width)] for _ in range(height)]
    
    # 绘制损失曲线
    n = len(losses)
    for i, loss in enumerate(losses):
        x = int(i / n * (width - 1))
        y = height - 1 - int((loss - min_loss) / loss_range * (height - 1))
        y = max(0, min(y, height - 1))
        canvas[y][x] = '*'
    
    lines = ["=" * 70]
    lines.append("损失函数下降曲线")
    lines.append(f"初始损失: {losses[0]:.4f}  最终损失: {losses[-1]:.4f}  下降: {(losses[0]-losses[-1])/losses[0]*100:.1f}%")
    lines.append("-" * 70)
    
    for row in canvas:
        lines.append("|" + "".join(row) + "|")
    
    lines.append("-" * 70)
    return "\n".join(lines)


# ============================================================================
# 第二部分：批量梯度下降（Batch Gradient Descent）
# ============================================================================

def batch_gradient_descent_1d(
    f: Callable[[float], float],
    x0: float,
    lr: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = True
) -> Tuple[float, List[Tuple[int, float, float]]]:
    """
    批量梯度下降（1D版本）
    
    参数:
        f: 目标函数
        x0: 初始点
        lr: 学习率
        max_iter: 最大迭代次数
        tol: 收敛阈值（梯度范数小于此值时停止）
        verbose: 是否打印过程
    
    返回:
        (最优解, 轨迹)
    """
    x = x0
    trajectory = [(0, x, f(x))]
    
    for t in range(1, max_iter + 1):
        # 计算梯度
        grad = numerical_derivative(f, x)
        
        # 检查收敛
        if abs(grad) < tol:
            if verbose:
                print(f"收敛于第{t}步，梯度={grad:.8f}")
            break
        
        # 梯度下降更新
        x = x - lr * grad
        trajectory.append((t, x, f(x)))
        
        if verbose and t <= 5:
            print(f"Step {t}: x = {x:.6f}, f(x) = {f(x):.6f}, grad = {grad:.6f}")
    
    return x, trajectory


def batch_gradient_descent_nd(
    f: Callable[[List[float]], float],
    x0: List[float],
    lr: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = True
) -> Tuple[List[float], List[float]]:
    """
    批量梯度下降（ND版本）
    
    返回:
        (最优解, 损失历史)
    """
    x = x0.copy()
    loss_history = [f(x)]
    
    for t in range(1, max_iter + 1):
        grad = numerical_gradient(f, x)
        grad_norm = math.sqrt(sum(g**2 for g in grad))
        
        if grad_norm < tol:
            if verbose:
                print(f"收敛于第{t}步")
            break
        
        # 更新参数
        x = [x[i] - lr * grad[i] for i in range(len(x))]
        loss_history.append(f(x))
    
    return x, loss_history


# ============================================================================
# 第三部分：随机梯度下降（Stochastic Gradient Descent）
# ============================================================================

def stochastic_gradient_descent(
    loss_fn: Callable[[List[float], int], float],  # 单个样本的损失函数
    data_size: int,
    x0: List[float],
    lr: float = 0.01,
    epochs: int = 10,
    verbose: bool = True
) -> Tuple[List[float], List[float]]:
    """
    随机梯度下降
    
    参数:
        loss_fn: 函数签名 loss_fn(params, sample_idx) -> loss
        data_size: 训练集大小
        x0: 初始参数
        lr: 学习率
        epochs: 遍历数据的轮数
        verbose: 是否打印过程
    
    返回:
        (最优参数, 损失历史)
    """
    x = x0.copy()
    loss_history = []
    
    for epoch in range(epochs):
        # 随机打乱数据顺序
        indices = list(range(data_size))
        random.shuffle(indices)
        
        epoch_loss = 0.0
        
        for i, idx in enumerate(indices):
            # 计算单个样本的梯度（数值方法）
            def single_loss(p):
                return loss_fn(p, idx)
            
            grad = numerical_gradient(single_loss, x)
            
            # 更新参数
            x = [x[j] - lr * grad[j] for j in range(len(x))]
            
            # 计算损失（用于显示）
            loss = single_loss(x)
            epoch_loss += loss
        
        avg_loss = epoch_loss / data_size
        loss_history.append(avg_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")
    
    return x, loss_history


# ============================================================================
# 第四部分：动量方法（Momentum）
# ============================================================================

def sgd_with_momentum(
    f: Callable[[List[float]], float],
    x0: List[float],
    lr: float = 0.01,
    momentum: float = 0.9,
    max_iter: int = 100,
    verbose: bool = True
) -> Tuple[List[float], List[float]]:
    """
    带动量的梯度下降（Polyak Momentum / Heavy Ball）
    
    参数:
        f: 目标函数
        x0: 初始参数
        lr: 学习率
        momentum: 动量系数 (通常设为0.9)
        max_iter: 最大迭代次数
        verbose: 是否打印过程
    
    返回:
        (最优参数, 损失历史)
    """
    x = x0.copy()
    v = [0.0] * len(x)  # 速度（动量）
    loss_history = [f(x)]
    
    for t in range(1, max_iter + 1):
        grad = numerical_gradient(f, x)
        
        # 更新速度：v = momentum * v + lr * grad
        v = [momentum * v[i] + lr * grad[i] for i in range(len(x))]
        
        # 更新参数：x = x - v
        x = [x[i] - v[i] for i in range(len(x))]
        
        loss = f(x)
        loss_history.append(loss)
        
        if verbose and t <= 5:
            v_norm = math.sqrt(sum(vi**2 for vi in v))
            print(f"Step {t}: Loss = {loss:.6f}, |v| = {v_norm:.6f}")
    
    return x, loss_history


def nesterov_accelerated_gradient(
    f: Callable[[List[float]], float],
    x0: List[float],
    lr: float = 0.01,
    momentum: float = 0.9,
    max_iter: int = 100,
    verbose: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Nesterov加速梯度（NAG）
    
    在"预计到达的位置"计算梯度，而不是当前位置
    """
    x = x0.copy()
    v = [0.0] * len(x)
    loss_history = [f(x)]
    
    for t in range(1, max_iter + 1):
        # 计算预计位置
        x_lookahead = [x[i] - momentum * v[i] for i in range(len(x))]
        
        # 在预计位置计算梯度
        grad = numerical_gradient(f, x_lookahead)
        
        # 更新速度
        v = [momentum * v[i] + lr * grad[i] for i in range(len(x))]
        
        # 更新参数
        x = [x[i] - v[i] for i in range(len(x))]
        
        loss = f(x)
        loss_history.append(loss)
        
        if verbose and t <= 5:
            print(f"Step {t}: Loss = {loss:.6f}")
    
    return x, loss_history


# ============================================================================
# 第五部分：演示函数
# ============================================================================

def demo_quadratic_1d():
    """
    演示1：1D二次函数优化 f(x) = x^2
    """
    print("\n" + "=" * 70)
    print("演示1：1D二次函数 f(x) = x^2 的优化")
    print("=" * 70)
    
    f = lambda x: x ** 2
    x0 = 10.0
    
    print(f"初始点: x0 = {x0}, f(x0) = {f(x0)}")
    print(f"理论最优解: x* = 0, f(x*) = 0\n")
    
    # 不同学习率的比较
    learning_rates = [0.01, 0.1, 0.5]
    
    for lr in learning_rates:
        print(f"\n--- 学习率 = {lr} ---")
        x_opt, trajectory = batch_gradient_descent_1d(f, x0, lr=lr, max_iter=100, verbose=True)
        print(f"最终解: x = {x_opt:.6f}, f(x) = {f(x_opt):.6f}")
        print(f"迭代次数: {len(trajectory)}")
    
    # 可视化
    print("\n" + visualize_1d_trajectory(trajectory))


def demo_rosenbrock():
    """
    演示2：Rosenbrock函数（香蕉函数）- 经典的优化测试函数
    f(x,y) = (a-x)^2 + b(y-x^2)^2
    全局最小值在 (a, a^2)
    """
    print("\n" + "=" * 70)
    print("演示2：Rosenbrock函数（香蕉函数）")
    print("=" * 70)
    
    a, b = 1.0, 100.0
    
    def rosenbrock(x: List[float]) -> float:
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    
    x0 = [-1.0, 2.0]
    print(f"初始点: x0 = {x0}")
    print(f"理论最优解: x* = [{a}, {a**2}]")
    print(f"初始损失: {rosenbrock(x0):.4f}\n")
    
    # 批量梯度下降
    print("--- 批量梯度下降 ---")
    x_opt, loss_history = batch_gradient_descent_nd(
        rosenbrock, x0, lr=0.001, max_iter=1000, verbose=False
    )
    print(f"最终解: [{x_opt[0]:.4f}, {x_opt[1]:.4f}]")
    print(f"最终损失: {rosenbrock(x_opt):.6f}")
    print(visualize_loss_curve(loss_history))
    
    # 带动量的梯度下降
    print("\n--- 带动量的梯度下降 ---")
    x_opt_mom, loss_history_mom = sgd_with_momentum(
        rosenbrock, x0, lr=0.001, momentum=0.9, max_iter=1000, verbose=False
    )
    print(f"最终解: [{x_opt_mom[0]:.4f}, {x_opt_mom[1]:.4f}]")
    print(f"最终损失: {rosenbrock(x_opt_mom):.6f}")
    print(visualize_loss_curve(loss_history_mom))


def demo_linear_regression():
    """
    演示3：用SGD训练线性回归
    真实模型: y = 2x + 1 + noise
    """
    print("\n" + "=" * 70)
    print("演示3：用SGD训练线性回归")
    print("=" * 70)
    
    # 生成训练数据
    random.seed(42)
    n_samples = 100
    true_w, true_b = 2.0, 1.0
    
    data = []
    for i in range(n_samples):
        x = random.uniform(-5, 5)
        noise = random.gauss(0, 0.5)
        y = true_w * x + true_b + noise
        data.append((x, y))
    
    print(f"生成{n_samples}个样本")
    print(f"真实参数: w = {true_w}, b = {true_b}")
    
    # 损失函数: MSE = 1/N * sum(y_pred - y_true)^2
    def single_sample_loss(params: List[float], idx: int) -> float:
        w, b = params[0], params[1]
        x, y_true = data[idx]
        y_pred = w * x + b
        return (y_pred - y_true) ** 2
    
    # 使用SGD训练
    x0 = [0.0, 0.0]  # 初始参数
    print(f"\n初始参数: w = {x0[0]:.4f}, b = {x0[1]:.4f}")
    
    w_opt, loss_history = stochastic_gradient_descent(
        single_sample_loss, n_samples, x0, lr=0.01, epochs=20, verbose=True
    )
    
    print(f"\n训练完成!")
    print(f"估计参数: w = {w_opt[0]:.4f}, b = {w_opt[1]:.4f}")
    print(f"真实参数: w = {true_w:.4f}, b = {true_b:.4f}")
    print(visualize_loss_curve(loss_history))


def demo_saddle_point():
    """
    演示4：鞍点问题 f(x,y) = x^2 - y^2
    在(0,0)处是一个鞍点
    """
    print("\n" + "=" * 70)
    print("演示4：鞍点 f(x,y) = x^2 - y^2")
    print("=" * 70)
    
    def saddle_func(x: List[float]) -> float:
        return x[0]**2 - x[1]**2
    
    print("在(0,0)处，沿x方向是最小值，沿y方向是最大值")
    print("这是一个典型的鞍点\n")
    
    # 从鞍点附近开始
    x0 = [0.1, 0.1]
    print(f"初始点: {x0}")
    
    # 梯度下降
    print("\n--- 批量梯度下降 ---")
    x_opt, loss_history = batch_gradient_descent_nd(
        saddle_func, x0, lr=0.1, max_iter=50, verbose=False
    )
    print(f"最终解: [{x_opt[0]:.4f}, {x_opt[1]:.4f}]")
    print(f"注意y方向发散到了{x_opt[1]:.4f}")
    print(visualize_loss_curve(loss_history))
    
    # 添加噪声的梯度下降（模拟SGD）
    print("\n--- 带噪声的梯度下降（帮助逃离鞍点）---")
    x = x0.copy()
    loss_history_noisy = [saddle_func(x)]
    lr = 0.1
    
    for t in range(50):
        grad = numerical_gradient(saddle_func, x)
        # 添加随机噪声
        noise = [random.gauss(0, 0.1) for _ in range(len(x))]
        grad_noisy = [grad[i] + noise[i] for i in range(len(x))]
        x = [x[i] - lr * grad_noisy[i] for i in range(len(x))]
        loss_history_noisy.append(saddle_func(x))
    
    print(f"最终解: [{x[0]:.4f}, {x[1]:.4f}]")
    print(visualize_loss_curve(loss_history_noisy))


def demo_learning_rate_comparison():
    """
    演示5：学习率对收敛的影响
    """
    print("\n" + "=" * 70)
    print("演示5：学习率对收敛的影响")
    print("目标函数: f(x) = x^2")
    print("=" * 70)
    
    f = lambda x: x[0]**2
    x0 = [5.0]
    
    learning_rates = [0.05, 0.3, 0.9, 1.1]
    
    for lr in learning_rates:
        x_opt, loss_history = batch_gradient_descent_nd(
            f, x0, lr=lr, max_iter=20, verbose=False
        )
        
        status = "收敛" if loss_history[-1] < loss_history[0] else "发散" if loss_history[-1] > 100 else "震荡"
        print(f"\n学习率 = {lr}: {status}")
        print(f"  初始损失: {loss_history[0]:.4f}, 最终损失: {loss_history[-1]:.4f}")
        print(f"  损失序列: {[round(l, 2) for l in loss_history[:8]]}")


def demo_3d_visualization():
    """
    演示6：2D参数空间的可视化（文本形式）
    展示优化路径在参数空间中的轨迹
    """
    print("\n" + "=" * 70)
    print("演示6：2D参数空间优化轨迹")
    print("目标函数: f(x,y) = x^2 + 2*y^2")
    print("=" * 70)
    
    def func(x: List[float]) -> float:
        return x[0]**2 + 2*x[1]**2
    
    x0 = [3.0, 3.0]
    
    # 普通梯度下降
    print("\n--- 普通梯度下降 ---")
    x = x0.copy()
    path_gd = [x.copy()]
    lr = 0.1
    
    for _ in range(10):
        grad = [2*x[0], 4*x[1]]  # 解析梯度
        x = [x[i] - lr * grad[i] for i in range(2)]
        path_gd.append(x.copy())
    
    print("优化路径:")
    for i, (px, py) in enumerate(path_gd):
        print(f"  Step {i}: ({px:.3f}, {py:.3f}), f = {func([px, py]):.4f}")
    
    # 带动量的梯度下降
    print("\n--- 带动量的梯度下降 ---")
    x = x0.copy()
    v = [0.0, 0.0]
    path_mom = [x.copy()]
    lr = 0.1
    momentum = 0.9
    
    for _ in range(10):
        grad = [2*x[0], 4*x[1]]
        v = [momentum * v[i] + lr * grad[i] for i in range(2)]
        x = [x[i] - v[i] for i in range(2)]
        path_mom.append(x.copy())
    
    print("优化路径:")
    for i, (px, py) in enumerate(path_mom):
        print(f"  Step {i}: ({px:.3f}, {py:.3f}), f = {func([px, py]):.4f}")
    
    print("\n观察：带动量时，y方向（梯度大的方向）有'冲过头'的现象")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    运行所有演示
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║      第四章：一步一步变得更好——梯度下降的直觉                    ║
    ║                                                                  ║
    ║      配套代码演示：从零实现梯度下降算法                          ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # 设置随机种子保证可重复性
    random.seed(42)
    
    # 运行各个演示
    demo_quadratic_1d()
    demo_rosenbrock()
    demo_linear_regression()
    demo_saddle_point()
    demo_learning_rate_comparison()
    demo_3d_visualization()
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)
    
    # 打印总结
    print("""
    总结：
    1. 梯度下降通过沿负梯度方向迭代更新参数来最小化函数
    2. 学习率是关键超参数，太大导致发散，太小收敛慢
    3. SGD使用随机梯度估计，计算高效且有助于逃离鞍点
    4. 动量方法累积历史梯度，加速收敛并减少震荡
    5. 现代优化器（Adam等）自适应调整学习率，更易使用
    """)


if __name__ == "__main__":
    main()
