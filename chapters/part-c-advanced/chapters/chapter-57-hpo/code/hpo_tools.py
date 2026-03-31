#!/usr/bin/env python3
"""
超参数优化工具集
Chapter 57: 超参数调优进阶
"""

import numpy as np
from typing import List, Dict, Callable, Tuple, Any
from dataclasses import dataclass
import random


@dataclass
class HyperParameter:
    """超参数定义""
    name: str
    param_type: str  # 'continuous', 'discrete', 'categorical'
    bounds: Tuple[float, float]  # (low, high)
    log_scale: bool = False
    
    def sample(self) -> float:
        """从参数空间中随机采样""
        if self.param_type == 'continuous':
            if self.log_scale:
                log_low, log_high = np.log(self.bounds[0]), np.log(self.bounds[1])
                return np.exp(np.random.uniform(log_low, log_high))
            return np.random.uniform(self.bounds[0], self.bounds[1])
        elif self.param_type == 'discrete':
            return np.random.randint(int(self.bounds[0]), int(self.bounds[1]) + 1)
        else:
            raise ValueError(f"Unknown param_type: {self.param_type}")


class RandomSearch:
    """随机搜索算法""
    
    def __init__(self, hyperparams: List[HyperParameter], objective_fn: Callable):
        self.hyperparams = hyperparams
        self.objective_fn = objective_fn
        self.history = []
    
    def optimize(self, n_trials: int = 100) -> Dict[str, Any]:
        """执行随机搜索""
        best_score = -float('inf')
        best_params = None
        
        for trial in range(n_trials):
            # 采样一组参数
            params = {hp.name: hp.sample() for hp in self.hyperparams}
            
            # 评估
            score = self.objective_fn(params)
            
            self.history.append({
                'trial': trial,
                'params': params,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_params = params
                print(f"Trial {trial}: New best score = {score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'history': self.history
        }


class GridSearch:
    """网格搜索算法""
    
    def __init__(self, param_grid: Dict[str, List], objective_fn: Callable):
        self.param_grid = param_grid
        self.objective_fn = objective_fn
        self.history = []
    
    def _generate_combinations(self) -> List[Dict]:
        """生成所有参数组合""
        import itertools
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def optimize(self) -> Dict[str, Any]:
        """执行网格搜索""
        combinations = self._generate_combinations()
        print(f"Total combinations: {len(combinations)}")
        
        best_score = -float('inf')
        best_params = None
        
        for i, params in enumerate(combinations):
            score = self.objective_fn(params)
            self.history.append({
                'trial': i,
                'params': params,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'history': self.history
        }


class EarlyStoppingScheduler:
    """早停调度器""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float('inf')
        self.counter = 0
        self.should_stop = False
    
    def step(self, score: float) -> bool:
        """记录当前分数，返回是否应该停止""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def get_status(self) -> Dict:
        return {
            'best_score': self.best_score,
            'counter': self.counter,
            'should_stop': self.should_stop
        }


# 使用示例
def demo_objective(params):
    """示例目标函数（Rosenbrock函数）""
    x = params.get('x', 0)
    y = params.get('y', 0)
    # 最大值在 (1, 1) 处，值为0
    # 我们取负值转化为最大化问题
    return -(100 * (y - x**2)**2 + (1 - x)**2)


if __name__ == '__main__':
    # 定义超参数
    hyperparams = [
        HyperParameter('x', 'continuous', (-2, 2)),
        HyperParameter('y', 'continuous', (-1, 3))
    ]
    
    # 随机搜索
    print("=== Random Search ===")
    rs = RandomSearch(hyperparams, demo_objective)
    result = rs.optimize(n_trials=50)
    print(f"\nBest params: {result['best_params']}")
    print(f"Best score: {result['best_score']:.4f}")
