"""
57.4.2 HyperBand算法实现
"""

import numpy as np
import math


class HyperBand:
    """
    HyperBand算法
    
    通过运行多个不同规模的Successive Halving，
    自动平衡"探索更多配置"和"给每个配置更多资源"之间的权衡
    """
    
    def __init__(self, max_resource, eta=3, random_state=None):
        """
        Args:
            max_resource: 单个配置的最大资源（如最大epoch数）
            eta: 淘汰比例
            random_state: 随机种子
        """
        self.max_resource = max_resource
        self.eta = eta
        
        if random_state:
            np.random.seed(random_state)
        
        # 计算s_max
        self.s_max = int(math.log(max_resource, eta))
        print(f"HyperBand配置:")
        print(f"  最大资源 R = {max_resource}")
        print(f"  淘汰比例 η = {eta}")
        print(f"  最大迭代 s_max = {self.s_max}")
    
    def run(self, get_config_fn: Callable, train_fn: Callable, 
            total_budget: float = None):
        """
        执行HyperBand
        
        Args:
            get_config_fn: 生成新配置的函数
            train_fn: 训练函数，接收(config, resource)返回性能
            total_budget: 总预算（可选，用于控制总计算量）
            
        Returns:
            best_config: 找到的最优配置
            best_score: 最优分数
            all_history: 所有SH运行的历史记录
        """
        
        best_overall_config = None
        best_overall_score = float('-inf')
        all_history = []
        
        # 从最大的s开始（更激进的资源分配）
        for s in range(self.s_max, -1, -1):
            # 计算这一轮的参数
            # n: 初始配置数
            # r: 初始资源
            n = int(math.ceil(
                (self.s_max + 1) / (s + 1) * self.eta ** s
            ))
            r = self.max_resource * self.eta ** (-s)
            
            print(f"\n{'='*60}")
            print(f"[HyperBand bracket s={s}]")
            print(f"  初始配置数 n = {n}")
            print(f"  初始资源 r = {r:.2f}")
            
            # 生成n个随机配置
            configs = [get_config_fn() for _ in range(n)]
            
            # 运行Successive Halving
            sh = SuccessiveHalving(eta=self.eta)
            best_config, history = sh.maximize(
                configs=configs,
                train_fn=train_fn,
                max_resource=self.max_resource,
                min_resource=r
            )
            
            all_history.extend(history)
            
            # 更新全局最优
            bracket_best_score = max(r['score'] for r in history 
                                     if r['resource'] == self.max_resource)
            if bracket_best_score > best_overall_score:
                best_overall_score = bracket_best_score
                best_overall_config = best_config
                print(f"  → 新的全局最优! 分数={bracket_best_score:.4f}")
        
        print(f"\n{'='*60}")
        print(f"HyperBand完成! 最优分数: {best_overall_score:.4f}")
        
        return best_overall_config, best_overall_score, all_history


# ========================================
# 比较：Random Search vs Successive Halving vs HyperBand
# ========================================

def compare_methods():
    """比较三种方法的效率"""
    
    np.random.seed(42)
    
    # 问题设置
    n_configs_total = 100
    max_epochs = 81
    
    # 生成配置和真实性能
    true_perfs = np.random.beta(2, 5, n_configs_total)
    true_perfs[23] = 0.96  # 最佳配置
    
    def get_config():
        return {'id': np.random.randint(n_configs_total)}
    
    def train(config, resource):
        tid = config['id']
        true_perf = true_perfs[tid]
        k = 0.1 + 0.03 * (tid % 5)  # 不同学习速度
        perf = true_perf * (1 - np.exp(-k * resource / 10))
        return perf + np.random.normal(0, 0.01)
    
    print("="*70)
    print("方法对比实验")
    print("="*70)
    
    # 1. 纯随机搜索（所有配置都用满资源）
    print("\n[方法1] 纯随机搜索 (每个配置用满资源)")
    n_random = 10  # 只能负担10个完整训练
    random_scores = []
    for i in range(n_random):
        config = get_config()
        score = train(config, max_epochs)
        random_scores.append((config['id'], score))
    best_random = max(random_scores, key=lambda x: x[1])
    print(f"  评估配置数: {n_random}")
    print(f"  最优配置ID: {best_random[0]}, 分数: {best_random[1]:.4f}")
    
    # 2. 单次SH
    print("\n[方法2] 单次Successive Halving")
    configs = [get_config() for _ in range(27)]
    sh = SuccessiveHalving(eta=3)
    best_sh, _ = sh.maximize(configs, train, max_epochs, min_resource=1)
    print(f"  最优配置ID: {best_sh['id']}, 真实排名: {np.argsort(true_perfs)[::-1].tolist().index(best_sh['id'])+1}")
    
    # 3. HyperBand
    print("\n[方法3] HyperBand")
    hb = HyperBand(max_resource=max_epochs, eta=3)
    best_hb, score_hb, _ = hb.run(get_config, train)
    print(f"  最优配置ID: {best_hb['id']}, 真实排名: {np.argsort(true_perfs)[::-1].tolist().index(best_hb['id'])+1}")
    
    print("\n" + "="*70)
    print("总结：HyperBand通过智能分配资源，在相同预算下找到更好的配置")


if __name__ == "__main__":
    from successive_halving import SuccessiveHalving  # 复用前面的代码
    from typing import Callable
    
    compare_methods()