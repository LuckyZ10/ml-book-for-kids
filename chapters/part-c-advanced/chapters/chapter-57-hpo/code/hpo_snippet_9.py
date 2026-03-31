"""
57.4.3 BOHB核心思想演示
"""

import numpy as np
from scipy.stats import norm
from collections import defaultdict


class TreeParzenEstimator:
    """
    树形Parzen估计器 (TPE)
    
    TPE是BOHB中使用的贝叶斯优化方法，相比GP更适合混合类型参数
    
    核心思想：
    - 不使用 p(y|x)，而是直接建模 p(x|y)
    - 将观测分为"好的"(y < y*)和"坏的"(y >= y*)
    - 使用核密度估计(KDE)建模 l(x) = p(x|好) 和 g(x) = p(x|坏)
    - 选择使 l(x)/g(x) 最大的x
    """
    
    def __init__(self, gamma=0.15):
        """
        Args:
            gamma: 用于区分"好"和"坏"的分位数
        """
        self.gamma = gamma
        self.observations = []  # (config, loss) 列表
    
    def observe(self, config, loss):
        """记录一次观测"""
        self.observations.append((config, loss))
    
    def suggest(self, n_samples=100):
        """
        建议下一个配置
        
        返回使 EI 近似最大的配置
        """
        if len(self.observations) < 10:
            # 数据不足时随机采样
            return None
        
        # 按损失排序
        sorted_obs = sorted(self.observations, key=lambda x: x[1])
        
        # 分割点
        n_good = max(1, int(self.gamma * len(sorted_obs)))
        
        good_configs = [obs[0] for obs in sorted_obs[:n_good]]
        bad_configs = [obs[0] for obs in sorted_obs[n_good:]]
        
        # 对于每个超参数，计算l(x)/g(x)比率
        # 这里简化处理，实际TPE有更复杂的处理方式
        
        # 随机生成候选并评分
        best_ratio = -1
        best_config = None
        
        for _ in range(n_samples):
            candidate = self._random_config()
            ratio = self._compute_ratio(candidate, good_configs, bad_configs)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_config = candidate
        
        return best_config
    
    def _random_config(self):
        """随机生成配置"""
        return {
            'lr': 10 ** np.random.uniform(-5, -1),
            'batch_size': np.random.choice([16, 32, 64, 128]),
            'dropout': np.random.uniform(0.1, 0.5)
        }
    
    def _compute_ratio(self, candidate, good_configs, bad_configs):
        """计算 l(candidate)/g(candidate) 的近似值"""
        # 简化的基于距离的计算
        def min_distance(configs):
            dists = []
            for c in configs:
                d = (np.log10(candidate['lr']) - np.log10(c['lr']))**2
                d += (candidate['dropout'] - c['dropout'])**2 * 10
                dists.append(d)
            return min(dists) if dists else 1.0
        
        l_prob = np.exp(-min_distance(good_configs))
        g_prob = np.exp(-min_distance(bad_configs)) + 1e-10
        
        return l_prob / g_prob


class BOHB:
    """
    BOHB: 贝叶斯优化 + HyperBand
    
    结合了HyperBand的高效资源分配和贝叶斯优化的智能采样
    """
    
    def __init__(self, max_resource, eta=3):
        self.max_resource = max_resource
        self.eta = eta
        self.tpe = TreeParzenEstimator()
        self.s_max = int(np.log(max_resource) / np.log(eta))
    
    def run(self, train_fn, n_iterations=5):
        """
        执行BOHB
        
        每轮使用TPE生成配置，然后用SH评估
        """
        
        best_config = None
        best_score = float('-inf')
        
        for iteration in range(n_iterations):
            print(f"\n[BOHB Iteration {iteration+1}/{n_iterations}]")
            
            # 前几次迭代随机探索，之后使用TPE
            if iteration < 2:
                get_config = lambda: self.tpe._random_config()
                print("  模式: 随机探索")
            else:
                def get_config():
                    cfg = self.tpe.suggest()
                    return cfg if cfg else self.tpe._random_config()
                print("  模式: TPE指导采样")
            
            # 运行一个HyperBand bracket
            for s in [self.s_max]:  # 可以扩展到多个s值
                n = int((self.s_max + 1) / (s + 1) * self.eta ** s)
                r = self.max_resource * self.eta ** (-s)
                
                # 生成配置
                configs = [get_config() for _ in range(n)]
                
                # 使用SH评估并更新TPE
                sh = SuccessiveHalving(eta=self.eta)
                bracket_best, history = sh.maximize(
                    configs, train_fn, self.max_resource, r
                )
                
                # 更新TPE观测
                for record in history:
                    # 使用最终资源的分数
                    if record['resource'] == self.max_resource:
                        # TPE最小化损失，所以取负值
                        loss = -record['score']
                        self.tpe.observe(record['config'], loss)
                        
                        if record['score'] > best_score:
                            best_score = record['score']
                            best_config = record['config']
        
        return best_config, best_score


if __name__ == "__main__":
    # 演示BOHB流程
    print("BOHB核心思想演示")
    print("="*60)
    
    # 简化演示
    tpe = TreeParzenEstimator()
    
    # 模拟一些观测
    np.random.seed(42)
    for i in range(20):
        config = tpe._random_config()
        # 模拟损失（越小越好）
        loss = -np.log(config['lr']) * 0.1 + config['dropout'] * 2 + np.random.normal(0, 0.1)
        tpe.observe(config, loss)
    
    # 获取TPE建议
    suggestion = tpe.suggest(n_samples=50)
    print(f"\nTPE建议的配置:")
    print(f"  学习率: {suggestion['lr']:.6f}")
    print(f"  Dropout: {suggestion['dropout']:.3f}")
    
    print("\nBOHB结合HyperBand的资源分配和TPE的智能采样，")
    print("是目前超参数调优的最先进方法之一。")