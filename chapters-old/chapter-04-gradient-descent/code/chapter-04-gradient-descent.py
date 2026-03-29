#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四章：一步一步变得更好 - 手搓梯度下降
Chapter 04: Getting Better Step by Step - Gradient Descent from Scratch

本章我们将：
1. 用纯 Python 实现梯度下降
2. 可视化优化过程
3. 研究学习率对收敛的影响
4. 理解局部最优与全局最优

作者：ML教材
日期：2026-03-24
"""

import math


class GradientDescent:
    """手搓梯度下降优化器"""
    
    def __init__(self, learning_rate=0.1, max_iterations=1000, tolerance=1e-6):
        """
        初始化梯度下降器
        
        参数:
            learning_rate: 学习率 eta，控制每步走多远
            max_iterations: 最大迭代次数
            tolerance: 收敛阈值，当变化小于此值时停止
        """
        self.lr = learning_rate
        self.max_iter = max_iterations
        self.tol = tolerance
        self.history = []  # 记录优化轨迹
    
    def optimize(self, gradient_func, initial_x, verbose=True):
        """
        执行梯度下降优化
        
        参数:
            gradient_func: 计算梯度的函数，输入x，返回梯度值
            initial_x: 初始值
            verbose: 是否打印过程
            
        返回:
            (最优x, 迭代次数, 历史记录)
        """
        x = initial_x
        self.history = [(0, x)]  # (迭代次数, x值)
        
        if verbose:
            print(f"初始值: x = {x:.6f}")
            print("-" * 50)
        
        for i in range(1, self.max_iter + 1):
            # 计算当前梯度
            gradient = gradient_func(x)
            
            # 梯度下降更新: x_new = x_old - lr * gradient
            x_new = x - self.lr * gradient
            
            # 记录历史
            self.history.append((i, x_new))
            
            # 计算变化量
            change = abs(x_new - x)
            
            if verbose and (i <= 10 or i % 50 == 0):
                print(f"迭代 {i:4d}: x = {x_new:12.8f}, "
                      f"梯度 = {gradient:12.8f}, 变化 = {change:.8f}")
            
            # 检查是否收敛
            if change < self.tol:
                if verbose:
                    print(f"\n[OK] 收敛于迭代 {i}")
                    print(f"[OK] 最优解: x = {x_new:.8f}")
                return x_new, i, self.history
            
            x = x_new
        
        if verbose:
            print(f"\n[!] 达到最大迭代次数 {self.max_iter}")
            print(f"[!] 当前解: x = {x:.8f}")
        
        return x, self.max_iter, self.history
    
    def get_trajectory(self):
        """获取优化轨迹"""
        return self.history


def example_1_find_minimum():
    """
    示例1：找到 f(x) = x^2 的最小值
    解析解：f'(x) = 2x = 0 -> x = 0
    """
    print("=" * 60)
    print("示例1：寻找 f(x) = x^2 的最小值")
    print("=" * 60)
    print()
    
    # 定义梯度函数（f'(x) = 2x）
    def gradient(x):
        return 2 * x
    
    # 创建优化器，学习率 = 0.1
    optimizer = GradientDescent(learning_rate=0.1)
    
    # 从 x = 5 开始优化
    best_x, iterations, history = optimizer.optimize(gradient, initial_x=5.0)
    
    print(f"\n理论最优值: x = 0")
    print(f"找到的值:   x = {best_x:.10f}")
    print(f"误差:       {abs(best_x):.2e}")
    
    return best_x, history


def example_2_quadratic():
    """
    示例2：找到 f(x) = (x - 3)^2 + 5 的最小值
    解析解：f'(x) = 2(x - 3) = 0 -> x = 3, f(3) = 5
    """
    print("\n" + "=" * 60)
    print("示例2：寻找 f(x) = (x - 3)^2 + 5 的最小值")
    print("=" * 60)
    print("这个函数的最低点在 x = 3，最小值是 5")
    print()
    
    # 定义梯度函数（f'(x) = 2(x - 3)）
    def gradient(x):
        return 2 * (x - 3)
    
    optimizer = GradientDescent(learning_rate=0.3)
    best_x, iterations, history = optimizer.optimize(gradient, initial_x=0.0)
    
    # 计算最小值
    min_value = (best_x - 3) ** 2 + 5
    
    print(f"\n理论最优值: x = 3, f(x) = 5")
    print(f"找到的值:   x = {best_x:.10f}, f(x) = {min_value:.10f}")
    
    return best_x, history


def example_3_learning_rate_effects():
    """
    示例3：学习率对收敛的影响
    展示学习率太大或太小会发生什么
    """
    print("\n" + "=" * 60)
    print("示例3：学习率 eta 的影响研究")
    print("=" * 60)
    print()
    
    # 目标：找到 f(x) = x^2 的最小值
    def gradient(x):
        return 2 * x
    
    learning_rates = [0.01, 0.1, 0.5, 0.9, 1.0, 1.1]
    
    print("测试不同学习率的收敛情况:")
    print("-" * 50)
    
    for lr in learning_rates:
        optimizer = GradientDescent(
            learning_rate=lr, 
            max_iterations=100,
            tolerance=1e-10
        )
        best_x, iterations, _ = optimizer.optimize(
            gradient, 
            initial_x=5.0, 
            verbose=False
        )
        
        status = "[OK] 收敛" if abs(best_x) < 0.001 else "[FAIL] 发散"
        print(f"eta = {lr:4.2f}: {iterations:3d} 次迭代 -> x = {best_x:10.4f} {status}")
    
    print()
    print("[*] 观察结果:")
    print("    - eta 太小 (0.01): 收敛很慢，需要很多步")
    print("    - eta 适中 (0.1-0.5): 快速稳定收敛")
    print("    - eta 太大 (>1.0): 可能震荡甚至发散！")


def example_4_visualize_trajectory():
    """
    示例4：用 ASCII 艺术可视化优化轨迹
    """
    print("\n" + "=" * 60)
    print("示例4：梯度下降的轨迹可视化")
    print("=" * 60)
    print()
    
    def gradient(x):
        return 2 * x
    
    optimizer = GradientDescent(learning_rate=0.2)
    best_x, iterations, history = optimizer.optimize(
        gradient, 
        initial_x=4.0, 
        verbose=False
    )
    
    # 绘制 f(x) = x^2 和优化轨迹
    print("f(x) = x^2 的优化轨迹:")
    print()
    print("迭代次数 ->")
    print("  f(x)")
    print("   16 |                              *")
    print("   14 |                           *")
    print("   12 |                        *")
    print("   10 |                     *")
    print("    8 |                  *")
    print("    6 |               *")
    print("    4 |            *")
    print("    2 |         *")
    print("    1 |      *")
    print("    0 *-----|-----|-----|-----|-----|--> x")
    print("       -4   -2    0    2    4")
    print()
    print("* = 优化过程中的位置（从右向左移动）")
    print("轨迹显示了从 x=4 逐步接近 x=0 的过程")
    print(f"共 {iterations} 次迭代收敛")


def example_5_multiple_dimensions():
    """
    示例5：二维梯度下降
    最小化 f(x, y) = x^2 + y^2
    """
    print("\n" + "=" * 60)
    print("示例5：二维梯度下降 f(x, y) = x^2 + y^2")
    print("=" * 60)
    print()
    
    # 初始点
    x, y = 3.0, 4.0
    lr = 0.1
    
    print(f"初始点: ({x}, {y})")
    print(f"初始值: f(x,y) = {x**2 + y**2:.2f}")
    print()
    print("梯度下降过程:")
    print("-" * 50)
    
    for i in range(10):
        # 计算梯度
        grad_x = 2 * x  # partial f / partial x = 2x
        grad_y = 2 * y  # partial f / partial y = 2y
        
        # 更新
        x = x - lr * grad_x
        y = y - lr * grad_y
        
        value = x**2 + y**2
        print(f"迭代 {i+1}: ({x:8.4f}, {y:8.4f}), f = {value:.6f}")
    
    print()
    print(f"最终点: ({x:.6f}, {y:.6f})")
    print(f"最终值: f = {x**2 + y**2:.8f}")


def example_6_local_vs_global():
    """
    示例6：局部最小值 vs 全局最小值
    函数: f(x) = x^4 - 3x^3 + 2
    """
    print("\n" + "=" * 60)
    print("示例6：局部最小值 vs 全局最小值")
    print("=" * 60)
    print()
    print("函数: f(x) = x^4 - 3x^3 + 2")
    print("这是一个有多个山谷的函数")
    print()
    
    # f'(x) = 4x^3 - 9x^2 = x^2(4x - 9)
    # 临界点: x = 0 (拐点), x = 9/4 = 2.25 (最小值)
    def gradient(x):
        return 4 * x**3 - 9 * x**2
    
    # 从不同的起点开始
    starting_points = [-1.0, 0.5, 3.0]
    
    for start in starting_points:
        optimizer = GradientDescent(learning_rate=0.02, max_iterations=200)
        best_x, iterations, _ = optimizer.optimize(
            gradient, 
            initial_x=start, 
            verbose=False
        )
        
        # 计算函数值
        value = best_x**4 - 3*best_x**3 + 2
        
        print(f"起点 x = {start:5.2f} -> 收敛到 x = {best_x:.4f}, f(x) = {value:.4f}")
    
    print()
    print("[!] 注意：起点不同，可能收敛到不同的山谷！")
    print("    在机器学习中，我们通常只能找到足够好的解")
    print("    而不一定是最好的解")


def example_7_gradient_descent_with_momentum():
    """
    示例7：带动量的梯度下降（进阶）
    模拟物理中的惯性概念
    """
    print("\n" + "=" * 60)
    print("示例7：带动量的梯度下降（进阶）")
    print("=" * 60)
    print()
    print("想象一下：一个球滚下山坡...")
    print("球有惯性，即使坡度变缓，它也会继续向前滚动")
    print()
    
    class MomentumOptimizer:
        """带动量的梯度下降"""
        
        def __init__(self, learning_rate=0.1, momentum=0.9):
            self.lr = learning_rate
            self.momentum = momentum
            self.velocity = 0  # 速度（累积的梯度）
        
        def step(self, gradient):
            # 更新速度: v = beta*v - eta*gradient
            self.velocity = self.momentum * self.velocity - self.lr * gradient
            # 更新位置: x = x + v
            return self.velocity
    
    # 比较普通 GD 和带动量的 GD
    def gradient(x):
        return 2 * x
    
    # 普通梯度下降
    print("普通梯度下降 (eta=0.1):")
    x_normal = 5.0
    for i in range(5):
        grad = gradient(x_normal)
        x_normal -= 0.1 * grad
        print(f"  迭代 {i+1}: x = {x_normal:.4f}")
    
    print()
    
    # 带动量的梯度下降
    print("带动量的梯度下降 (eta=0.1, momentum=0.9):")
    optimizer = MomentumOptimizer(learning_rate=0.1, momentum=0.9)
    x_momentum = 5.0
    for i in range(5):
        grad = gradient(x_momentum)
        update = optimizer.step(grad)
        x_momentum += update
        print(f"  迭代 {i+1}: x = {x_momentum:.4f}, 速度 = {update:.4f}")
    
    print()
    print("[*] 动量帮助加速通过平坦区域，减少震荡！")


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("第四章：一步一步变得更好 - 手搓梯度下降")
    print("Gradient Descent from Scratch")
    print("=" * 70)
    print()
    
    # 运行所有示例
    example_1_find_minimum()
    example_2_quadratic()
    example_3_learning_rate_effects()
    example_4_visualize_trajectory()
    example_5_multiple_dimensions()
    example_6_local_vs_global()
    example_7_gradient_descent_with_momentum()
    
    print("\n" + "=" * 70)
    print("第四章实践完成！")
    print("关键收获:")
    print("  1. 梯度下降 = 沿梯度反方向逐步优化")
    print("  2. 学习率控制步长，太大太小都不好")
    print("  3. 可能收敛到局部最优而非全局最优")
    print("  4. 动量可以加速收敛")
    print("=" * 70)
