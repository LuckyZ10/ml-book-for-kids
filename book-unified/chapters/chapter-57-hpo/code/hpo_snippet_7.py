"""
57.4.1 Successive Halving算法实现
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Dict, Any


class SuccessiveHalving:
    """
    逐次折半算法 (Successive Halving)
    
    通过逐步淘汰表现差的配置来高效搜索超参数空间
    """
    
    def __init__(self, eta=3, random_state=None):
        """
        Args:
            eta: 淘汰比例，每轮保留1/eta的配置
            random_state: 随机种子
        """
        self.eta = eta
        if random_state:
            np.random.seed(random_state)
    
    def maximize(self, configs: List[Dict], train_fn: Callable, 
                 max_resource: float, min_resource: float = 1.0):
        """
        执行Successive Halving
        
        Args:
            configs: 初始配置列表
            train_fn: 训练函数，接收(config, resource)返回性能分数
            max_resource: 最大资源预算（如完整训练的epoch数）
            min_resource: 初始资源预算
            
        Returns:
            best_config: 最优配置
            history: 训练历史记录
        """
        
        n_configs = len(configs)
        
        # 计算需要多少轮才能用max_resource训练一个配置
        # min_resource * eta^s = max_resource
        max_sh_iter = int(np.log(max_resource / min_resource) / np.log(self.eta))
        
        # 根据预算约束，调整初始配置数
        # 总共使用的资源 = sum_{s=0}^{S} N_s * r_s
        # 其中 N_s = n_configs / eta^s, r_s = min_resource * eta^s
        # 所以每轮资源消耗相同！
        
        print(f"Successive Halving 配置:")
        print(f"  初始配置数: {n_configs}")
        print(f"  淘汰比例 η: {self.eta}")
        print(f"  资源范围: {min_resource} → {max_resource}")
        print(f"  迭代轮数: {max_sh_iter + 1}")
        print("=" * 60)
        
        survivors = list(range(n_configs))  # 存活配置的索引
        resource = min_resource
        history = []
        
        for iteration in range(max_sh_iter + 1):
            n_survivors = len(survivors)
            
            print(f"\n[第{iteration+1}轮] {n_survivors}个配置，每个使用{resource:.1f}资源")
            
            # 评估所有存活配置
            scores = []
            for idx in survivors:
                config = configs[idx]
                score = train_fn(config, resource)
                scores.append(score)
                
                history.append({
                    'config_id': idx,
                    'config': config,
                    'resource': resource,
                    'score': score,
                    'iteration': iteration
                })
                
                print(f"  配置{idx}: 分数={score:.4f}")
            
            # 如果不是最后一轮，进行淘汰
            if iteration < max_sh_iter:
                # 保留表现最好的 n_survivors // eta 个
                n_keep = max(1, n_survivors // self.eta)
                
                # 按分数排序，保留最好的
                sorted_indices = np.argsort(scores)[::-1]  # 降序
                keep_indices = sorted_indices[:n_keep]
                survivors = [survivors[i] for i in keep_indices]
                
                print(f"  → 保留表现最好的{n_keep}个配置")
                
                # 增加资源
                resource *= self.eta
            else:
                # 最后一轮，找出最佳配置
                best_local_idx = np.argmax(scores)
                best_config_idx = survivors[best_local_idx]
                best_score = scores[best_local_idx]
                
                print(f"\n最优配置: 配置{best_config_idx}, 分数={best_score:.4f}")
        
        best_config = configs[best_config_idx]
        return best_config, history


# ========================================
# 演示：SH在合成问题上的表现
# ========================================

def demo_successive_halving():
    """演示SH算法的执行过程"""
    
    # 定义一些"假"的配置
    # 真实性能随着训练而提升，但不同配置的提升速度不同
    np.random.seed(42)
    n_configs = 27  # 能被3整除多次
    
    # 为每个配置生成一个"真实"的最终性能
    true_performances = np.random.beta(2, 5, n_configs)  # 多数配置表现一般
    true_performances[5] = 0.95  # 有一个宝藏配置！
    true_performances[12] = 0.88  # 还有一个不错的
    
    # 学习曲线模拟：早期可能无法区分好坏
    def simulate_training(config_id, epochs):
        """
        模拟训练过程
        好的配置早期可能表现一般，但随着训练时间增加，优势显现
        """
        true_perf = true_performances[config_id]
        
        # 学习曲线模型: performance = true_perf * (1 - exp(-k * epochs))
        # 不同配置的k不同（有的学得快，有的学得慢）
        k = 0.1 + 0.05 * np.random.rand()  # 随机学习速度
        
        perf = true_perf * (1 - np.exp(-k * epochs))
        noise = np.random.normal(0, 0.02)  # 观测噪声
        
        return perf + noise
    
    # 将配置表示为字典
    configs = [{'id': i, 'hidden_size': np.random.choice([64, 128, 256])}
               for i in range(n_configs)]
    
    # 创建SH优化器
    sh = SuccessiveHalving(eta=3, random_state=42)
    
    # 训练函数
    def train_fn(config, resource):
        return simulate_training(config['id'], resource)
    
    # 执行SH
    best_config, history = sh.maximize(
        configs=configs,
        train_fn=train_fn,
        max_resource=81,   # 最大epoch数
        min_resource=1     # 初始epoch数
    )
    
    print(f"\n找到的最优配置ID: {best_config['id']}")
    print(f"该配置的真实性能排名: {np.argsort(true_performances)[::-1].tolist().index(best_config['id']) + 1}/{n_configs}")
    
    # 可视化
    visualize_sh_history(history, true_performances)
    
    return best_config, history


def visualize_sh_history(history, true_performances):
    """可视化SH的训练历史"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 学习曲线图
    ax1 = axes[0]
    
    # 按配置ID分组
    config_data = {}
    for record in history:
        cid = record['config_id']
        if cid not in config_data:
            config_data[cid] = {'resources': [], 'scores': []}
        config_data[cid]['resources'].append(record['resource'])
        config_data[cid]['scores'].append(record['score'])
    
    # 绘制每个配置的学习曲线
    for cid, data in config_data.items():
        true_perf = true_performances[cid]
        # 根据真实性能着色
        if true_perf > 0.9:
            color = 'green'
            linewidth = 2.5
            alpha = 0.9
        elif true_perf > 0.7:
            color = 'blue'
            linewidth = 1.5
            alpha = 0.6
        else:
            color = 'gray'
            linewidth = 1
            alpha = 0.3
        
        ax1.plot(data['resources'], data['scores'], 
                color=color, linewidth=linewidth, alpha=alpha,
                marker='o', markersize=4)
    
    ax1.set_xlabel('Resource (Epochs)', fontsize=11)
    ax1.set_ylabel('Performance Score', fontsize=11)
    ax1.set_title('Successive Halving: Learning Curves', fontsize=12)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=2.5, label='Top tier (true>0.9)'),
        Line2D([0], [0], color='blue', linewidth=1.5, label='Mid tier (true>0.7)'),
        Line2D([0], [0], color='gray', linewidth=1, label='Low tier')
    ]
    ax1.legend(handles=legend_elements)
    
    # 2. 资源分配热图
    ax2 = axes[1]
    
    iterations = sorted(set(r['iteration'] for r in history))
    n_iters = len(iterations)
    resource_matrix = np.zeros((n_iters, len(true_performances)))
    
    for record in history:
        i = record['iteration']
        cid = record['config_id']
        resource_matrix[i, cid] = record['resource']
    
    im = ax2.imshow(resource_matrix, aspect='auto', cmap='YlOrRd')
    ax2.set_xlabel('Configuration ID', fontsize=11)
    ax2.set_ylabel('Iteration', fontsize=11)
    ax2.set_title('Resource Allocation per Config per Iteration', fontsize=12)
    plt.colorbar(im, ax=ax2, label='Resource')
    
    plt.tight_layout()
    plt.savefig('successive_halving_demo.png', dpi=150, bbox_inches='tight')
    print("\nSH可视化结果已保存到: successive_halving_demo.png")
    plt.show()


if __name__ == "__main__":
    demo_successive_halving()