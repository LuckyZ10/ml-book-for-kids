"""
57.2.3 网格搜索与随机搜索对比实验
演示为什么随机搜索通常比网格搜索更有效
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class HyperparameterSearch:
    """超参数搜索基类"""
    
    def __init__(self, objective_func):
        """
        初始化搜索器
        
        Args:
            objective_func: 损失函数，接受配置字典，返回性能分数
        """
        self.objective = objective_func
        self.history = []  # 记录所有尝试
        self.best_config = None
        self.best_score = float('-inf')
    
    def evaluate(self, config):
        """评估一个配置"""
        score = self.objective(config)
        self.history.append({
            'config': config.copy(),
            'score': score
        })
        
        if score > self.best_score:
            self.best_score = score
            self.best_config = config.copy()
        
        return score


class GridSearch(HyperparameterSearch):
    """网格搜索实现"""
    
    def search(self, param_grid):
        """
        执行网格搜索
        
        Args:
            param_grid: 字典，键是参数名，值是候选值列表
        """
        # 生成所有组合
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        
        total = 1
        for v in values:
            total *= len(v)
        print(f"网格搜索: 总共 {total} 个配置")
        
        for combo in product(*values):
            config = dict(zip(keys, combo))
            self.evaluate(config)
        
        return self.best_config, self.best_score


class RandomSearch(HyperparameterSearch):
    """随机搜索实现"""
    
    def search(self, param_distributions, n_iter=100, random_state=None):
        """
        执行随机搜索
        
        Args:
            param_distributions: 字典，键是参数名，值是采样函数
            n_iter: 迭代次数
            random_state: 随机种子
        """
        if random_state:
            np.random.seed(random_state)
        
        print(f"随机搜索: 总共 {n_iter} 个配置")
        
        for i in range(n_iter):
            config = {}
            for param_name, sample_func in param_distributions.items():
                config[param_name] = sample_func()
            self.evaluate(config)
        
        return self.best_config, self.best_score


# ========================================
# 实验：展示随机搜索的优势
# ========================================

def test_function(config):
    """
    测试损失函数：模拟一个有两个重要维度的场景
    其中第一个维度非常重要，第二个维度不太重要
    """
    x = config['important_param']
    y = config['less_important_param']
    z = config['noise_param']
    
    # 第一个维度有强烈的峰值效应
    # 第二个维度有微弱影响
    # 第三个维度几乎无影响（噪声）
    score = (
        10 * np.exp(-((x - 0.7) ** 2) / 0.01) +  # 重要维度：尖锐峰值在0.7
        2 * np.sin(y * np.pi) +                   # 次要维度：微弱波动
        0.1 * z +                                 # 噪声维度
        np.random.normal(0, 0.1)                  # 观测噪声
    )
    
    return score


def run_comparison():
    """运行对比实验"""
    
    print("=" * 60)
    print("网格搜索 vs 随机搜索对比实验")
    print("=" * 60)
    
    # 定义搜索空间
    # 注意：网格搜索在"important_param"上只有5个采样点
    # 而随机搜索可以有更多机会命中最优区域附近
    param_grid = {
        'important_param': [0.0, 0.25, 0.5, 0.75, 1.0],
        'less_important_param': [0.0, 0.5, 1.0],
        'noise_param': [0, 1, 2]
    }
    
    # 随机搜索的采样函数
    def sample_important():
        """在重要维度上密集采样"""
        return np.random.uniform(0, 1)
    
    def sample_less_important():
        return np.random.choice([0.0, 0.5, 1.0])
    
    def sample_noise():
        return np.random.choice([0, 1, 2])
    
    param_distributions = {
        'important_param': sample_important,
        'less_important_param': sample_less_important,
        'noise_param': sample_noise
    }
    
    # 运行网格搜索
    print("\n[1] 运行网格搜索...")
    gs = GridSearch(test_function)
    gs_start = time.time()
    gs_best_config, gs_best_score = gs.search(param_grid)
    gs_time = time.time() - gs_start
    
    print(f"    最佳配置: {gs_best_config}")
    print(f"    最佳分数: {gs_best_score:.4f}")
    print(f"    评估次数: {len(gs.history)}")
    print(f"    耗时: {gs_time:.2f}秒")
    
    # 运行随机搜索（相同评估次数）
    print("\n[2] 运行随机搜索...")
    rs = RandomSearch(test_function)
    rs_start = time.time()
    rs_best_config, rs_best_score = rs.search(
        param_distributions, 
        n_iter=len(gs.history),
        random_state=42
    )
    rs_time = time.time() - rs_start
    
    print(f"    最佳配置: {rs_best_config}")
    print(f"    最佳分数: {rs_best_score:.4f}")
    print(f"    评估次数: {len(rs.history)}")
    print(f"    耗时: {rs_time:.2f}秒")
    
    # 结果对比
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"网格搜索最佳分数: {gs_best_score:.4f}")
    print(f"随机搜索最佳分数: {rs_best_score:.4f}")
    print(f"提升幅度: {((rs_best_score - gs_best_score) / abs(gs_best_score) * 100):.2f}%")
    
    # 可视化
    visualize_results(gs, rs)
    
    return gs, rs


def visualize_results(gs, rs):
    """可视化搜索结果"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 提取数据
    gs_configs = [h['config'] for h in gs.history]
    gs_scores = [h['score'] for h in gs.history]
    gs_x = [c['important_param'] for c in gs_configs]
    gs_y = [c['less_important_param'] for c in gs_configs]
    
    rs_configs = [h['config'] for h in rs.history]
    rs_scores = [h['score'] for h in rs.history]
    rs_x = [c['important_param'] for c in rs_configs]
    rs_y = [c['less_important_param'] for c in rs_configs]
    
    # 1. 网格搜索采样点分布
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(gs_x, gs_y, c=gs_scores, cmap='viridis', 
                           s=100, alpha=0.7, edgecolors='black')
    ax1.set_xlabel('Important Parameter', fontsize=11)
    ax1.set_ylabel('Less Important Parameter', fontsize=11)
    ax1.set_title('Grid Search: Sampling Points\n(Regular Grid Pattern)', fontsize=12)
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='True Optimum')
    ax1.legend()
    plt.colorbar(scatter1, ax=ax1, label='Score')
    
    # 2. 随机搜索采样点分布
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(rs_x, rs_y, c=rs_scores, cmap='viridis', 
                           s=100, alpha=0.7, edgecolors='black')
    ax2.set_xlabel('Important Parameter', fontsize=11)
    ax2.set_ylabel('Less Important Parameter', fontsize=11)
    ax2.set_title('Random Search: Sampling Points\n(Uniform Coverage)', fontsize=12)
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='True Optimum')
    ax2.legend()
    plt.colorbar(scatter2, ax=ax2, label='Score')
    
    # 3. 收敛曲线对比
    ax3 = axes[1, 0]
    gs_cumulative_max = np.maximum.accumulate(gs_scores)
    rs_cumulative_max = np.maximum.accumulate(rs_scores)
    
    ax3.plot(range(1, len(gs_cumulative_max)+1), gs_cumulative_max, 
             'b-', linewidth=2, label='Grid Search', marker='o', markersize=4)
    ax3.plot(range(1, len(rs_cumulative_max)+1), rs_cumulative_max, 
             'r-', linewidth=2, label='Random Search', marker='s', markersize=4)
    ax3.set_xlabel('Number of Evaluations', fontsize=11)
    ax3.set_ylabel('Best Score So Far', fontsize=11)
    ax3.set_title('Convergence Comparison', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 重要维度上的投影
    ax4 = axes[1, 1]
    ax4.scatter(gs_x, gs_scores, c='blue', alpha=0.6, s=80, 
                label='Grid Search', edgecolors='black')
    ax4.scatter(rs_x, rs_scores, c='red', alpha=0.6, s=80, 
                label='Random Search', edgecolors='black', marker='s')
    ax4.axvline(x=0.7, color='green', linestyle='--', linewidth=2, 
                alpha=0.7, label='True Optimum (x=0.7)')
    ax4.set_xlabel('Important Parameter Value', fontsize=11)
    ax4.set_ylabel('Score', fontsize=11)
    ax4.set_title('Score vs Important Parameter\n(Random Search covers better)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('grid_vs_random_search.png', dpi=150, bbox_inches='tight')
    print("\n可视化结果已保存到: grid_vs_random_search.png")
    plt.show()


if __name__ == "__main__":
    gs, rs = run_comparison()