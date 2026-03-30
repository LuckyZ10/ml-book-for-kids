"""
57.3.5 采集函数的实现与可视化
对比PI、EI和UCB三种采集函数的行为差异
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class AcquisitionFunction:
    """采集函数基类"""
    
    def __init__(self, xi=0.01):
        """
        Args:
            xi: 探索参数，防止过早收敛
        """
        self.xi = xi
    
    def __call__(self, mu, sigma, f_best):
        """
        计算采集函数值
        
        Args:
            mu: 预测均值，形状 (n,)
            sigma: 预测标准差，形状 (n,)
            f_best: 当前最佳观测值
            
        Returns:
            acq_values: 采集函数值，形状 (n,)
        """
        raise NotImplementedError


class ProbabilityOfImprovement(AcquisitionFunction):
    """
    改进概率 (PI) 采集函数
    
    PI(x) = P(f(x) > f_best) = Φ((μ(x) - f_best - ξ) / σ(x))
    """
    
    def __call__(self, mu, sigma, f_best):
        # 避免除零
        sigma = np.maximum(sigma, 1e-9)
        
        # 标准化变量
        Z = (mu - f_best - self.xi) / sigma
        
        # 标准正态CDF
        pi = norm.cdf(Z)
        
        return pi


class ExpectedImprovement(AcquisitionFunction):
    """
    期望改进 (EI) 采集函数
    
    EI(x) = (μ(x) - f_best - ξ) * Φ(Z) + σ(x) * φ(Z)
    其中 Z = (μ(x) - f_best - ξ) / σ(x)
    """
    
    def __call__(self, mu, sigma, f_best):
        # 避免除零
        sigma = np.maximum(sigma, 1e-9)
        
        # 标准化变量
        Z = (mu - f_best - self.xi) / sigma
        
        # EI的闭式解
        # 第一项: (μ - f_best - ξ) * Φ(Z)
        improvement = mu - f_best - self.xi
        ei = improvement * norm.cdf(Z)
        
        # 第二项: σ * φ(Z)
        ei += sigma * norm.pdf(Z)
        
        # 当方差为0时，EI应为0（没有不确定性，也没有改进可能）
        ei[sigma < 1e-9] = 0
        
        return ei


class UpperConfidenceBound(AcquisitionFunction):
    """
    上置信界 (UCB) 采集函数
    
    UCB(x) = μ(x) + κ * σ(x)
    
    注意：UCB不需要f_best参数，但为了接口统一，保留该参数
    """
    
    def __init__(self, kappa=2.0):
        """
        Args:
            kappa: 探索参数，越大越倾向于探索
        """
        self.kappa = kappa
    
    def __call__(self, mu, sigma, f_best=None):
        return mu + self.kappa * sigma


def plot_acquisition_functions():
    """
    可视化三种采集函数的行为
    展示它们如何平衡探索与利用
    """
    
    # 假设的损失函数（GP预测结果）
    x = np.linspace(0, 10, 500)
    
    # 模拟GP预测：一个多峰函数
    # 均值函数
    mu = np.sin(x) + 0.5 * np.sin(3*x) + 0.3 * x - 2
    
    # 方差函数：在"已观测"点（x=2, x=5, x=8）附近小，远离时大
    sigma = 0.5 + 0.3 * (
        np.exp(-0.5 * (x - 2)**2) + 
        np.exp(-0.5 * (x - 5)**2) + 
        np.exp(-0.5 * (x - 8)**2)
    )
    sigma = 1.5 - sigma
    sigma = np.maximum(sigma, 0.1)
    
    # 当前最佳值
    f_best = np.max(mu)
    
    # 创建采集函数实例
    pi_acq = ProbabilityOfImprovement(xi=0.1)
    ei_acq = ExpectedImprovement(xi=0.1)
    ucb_acq = UpperConfidenceBound(kappa=2.0)
    
    # 计算采集函数值
    pi_values = pi_acq(mu, sigma, f_best)
    ei_values = ei_acq(mu, sigma, f_best)
    ucb_values = ucb_acq(mu, sigma, f_best)
    
    # 找到每个采集函数的最大值点
    pi_max_idx = np.argmax(pi_values)
    ei_max_idx = np.argmax(ei_values)
    ucb_max_idx = np.argmax(ucb_values)
    
    # 绘制
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. 均值和方差
    ax1 = axes[0]
    ax1.fill_between(x, mu - 2*sigma, mu + 2*sigma, alpha=0.3, color='blue',
                     label='95% Confidence')
    ax1.plot(x, mu, 'b-', linewidth=2, label='GP Mean μ(x)')
    ax1.axhline(y=f_best, color='red', linestyle='--', linewidth=2,
                label=f'Current Best f* = {f_best:.2f}')
    ax1.set_ylabel('f(x)', fontsize=11)
    ax1.set_title('GP Prediction (Mean and Uncertainty)', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. PI和EI对比
    ax2 = axes[1]
    ax2.plot(x, pi_values, 'g-', linewidth=2, label='PI(x)')
    ax2.plot(x, ei_values, 'm-', linewidth=2, label='EI(x)')
    ax2.axvline(x=x[pi_max_idx], color='green', linestyle='--', alpha=0.5,
                label=f'PI max at x={x[pi_max_idx]:.2f}')
    ax2.axvline(x=x[ei_max_idx], color='magenta', linestyle='--', alpha=0.5,
                label=f'EI max at x={x[ei_max_idx]:.2f}')
    ax2.set_ylabel('Acquisition Value', fontsize=11)
    ax2.set_title('Probability of Improvement vs Expected Improvement', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. UCB
    ax3 = axes[2]
    ax3.plot(x, mu, 'b--', linewidth=1.5, alpha=0.7, label='μ(x)')
    ax3.plot(x, ucb_values, 'r-', linewidth=2, label=f'UCB(x) = μ(x) + {ucb_acq.kappa}σ(x)')
    ax3.fill_between(x, mu, ucb_values, alpha=0.2, color='red', label='Exploration bonus')
    ax3.axvline(x=x[ucb_max_idx], color='red', linestyle='--', alpha=0.5,
                label=f'UCB max at x={x[ucb_max_idx]:.2f}')
    ax3.set_xlabel('x', fontsize=11)
    ax3.set_ylabel('Acquisition Value', fontsize=11)
    ax3.set_title('Upper Confidence Bound', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('acquisition_functions.png', dpi=150, bbox_inches='tight')
    print("\n采集函数对比图已保存到: acquisition_functions.png")
    plt.show()
    
    print(f"\n三种采集函数选择的最优点:")
    print(f"  PI  选择: x = {x[pi_max_idx]:.3f}")
    print(f"  EI  选择: x = {x[ei_max_idx]:.3f}")
    print(f"  UCB 选择: x = {x[ucb_max_idx]:.3f}")


def demonstrate_exploration_exploitation():
    """
    演示不同探索参数对采集函数行为的影响
    """
    
    x = np.linspace(0, 10, 500)
    
    # 模拟GP预测
    mu = 2 * np.sin(x) + 0.3 * x
    sigma = 0.5 + 0.8 * np.exp(-0.3 * (x - 5)**2)
    
    f_best = 1.5  # 当前最佳
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 不同xi值的PI
    ax1 = axes[0, 0]
    for xi in [0.0, 0.1, 0.5]:
        pi = ProbabilityOfImprovement(xi=xi)
        values = pi(mu, sigma, f_best)
        ax1.plot(x, values, linewidth=2, label=f'ξ = {xi}')
    ax1.set_title('PI with Different ξ Values', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('PI(x)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 不同xi值的EI
    ax2 = axes[0, 1]
    for xi in [0.0, 0.1, 0.5]:
        ei = ExpectedImprovement(xi=xi)
        values = ei(mu, sigma, f_best)
        ax2.plot(x, values, linewidth=2, label=f'ξ = {xi}')
    ax2.set_title('EI with Different ξ Values', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('EI(x)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 不同kappa值的UCB
    ax3 = axes[1, 0]
    for kappa in [0.5, 1.0, 2.0, 3.0]:
        ucb = UpperConfidenceBound(kappa=kappa)
        values = ucb(mu, sigma)
        ax3.plot(x, values, linewidth=2, label=f'κ = {kappa}')
    ax3.set_title('UCB with Different κ Values', fontsize=12)
    ax3.set_xlabel('x')
    ax3.set_ylabel('UCB(x)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 探索-利用可视化
    ax4 = axes[1, 1]
    
    # 标记当前最佳点
    best_idx = np.argmax(mu - sigma * 0.5)  # 折中位置
    
    # 利用：选择均值最大的点
    exploit_idx = np.argmax(mu)
    # 探索：选择方差最大的点
    explore_idx = np.argmax(sigma)
    # EI：平衡选择
    ei = ExpectedImprovement(xi=0.1)
    ei_values = ei(mu, sigma, f_best)
    ei_idx = np.argmax(ei_values)
    
    ax4.plot(x, mu, 'b-', linewidth=2, label='μ(x)')
    ax4.fill_between(x, mu - sigma, mu + sigma, alpha=0.2, color='gray')
    ax4.axvline(x=x[exploit_idx], color='green', linestyle='--', linewidth=2,
                label=f'Exploitation: x={x[exploit_idx]:.1f}')
    ax4.axvline(x=x[explore_idx], color='orange', linestyle='--', linewidth=2,
                label=f'Exploration: x={x[explore_idx]:.1f}')
    ax4.axvline(x=x[ei_idx], color='red', linestyle='-', linewidth=2,
                label=f'EI Balance: x={x[ei_idx]:.1f}')
    ax4.set_title('Exploration vs Exploitation Trade-off', fontsize=12)
    ax4.set_xlabel('x')
    ax4.set_ylabel('f(x)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('acquisition_tradeoff.png', dpi=150, bbox_inches='tight')
    print("\n探索-利用权衡图已保存到: acquisition_tradeoff.png")
    plt.show()


if __name__ == "__main__":
    plot_acquisition_functions()
    demonstrate_exploration_exploitation()