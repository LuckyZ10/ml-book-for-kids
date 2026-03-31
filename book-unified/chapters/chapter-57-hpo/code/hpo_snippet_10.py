"""
57.4.4 ASHA异步算法演示
"""

import numpy as np
from collections import deque


class ASHA:
    """
    ASHA: 异步逐次折半算法
    
    与同步SH不同，ASHA中每个配置独立运行：
    - 当一个配置完成某个资源级别的训练，立即评估
    - 如果表现足够好（排名在前1/η），立即升级到更高资源
    - 不需要等待同级别的其他配置完成
    
    优势：
    - 没有同步等待的开销
    - 更好的分布式并行效率
    - 响应更快，新配置可以立即开始
    """
    
    def __init__(self, min_resource, max_resource, reduction_factor=4, 
                 min_early_stopping_rate=0):
        """
        Args:
            min_resource: 最小资源（如初始epoch数）
            max_resource: 最大资源
            reduction_factor: 淘汰比例
            min_early_stopping_rate: 最早可以停止的轮次
        """
        self.min_resource = min_resource
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor
        self.rung_levels = self._get_rung_levels()
        
        # 存储每个rung的观测
        self.rungs = {r: [] for r in self.rung_levels}
        
        print(f"ASHA配置:")
        print(f"  Rung levels: {self.rung_levels}")
    
    def _get_rung_levels(self):
        """计算各个rung的资源级别"""
        rungs = []
        r = self.min_resource
        while r <= self.max_resource:
            rungs.append(r)
            r *= self.reduction_factor
        return rungs
    
    def should_promote(self, config_id, score, resource):
        """
        判断一个配置是否应该晋升到下一个rung
        
        如果该配置在当前rung的排名在前1/η，则晋升
        """
        # 找到当前所在的rung
        current_rung_idx = self.rung_levels.index(resource)
        
        # 记录观测
        self.rungs[resource].append((config_id, score))
        
        # 检查排名
        sorted_rung = sorted(self.rungs[resource], key=lambda x: x[1], reverse=True)
        rank = [i for i, (cid, _) in enumerate(sorted_rung) if cid == config_id][0]
        
        # 前1/η的配置可以晋升
        promotion_threshold = len(self.rungs[resource]) // self.reduction_factor
        
        should_promote = rank < promotion_threshold
        
        if should_promote and current_rung_idx < len(self.rung_levels) - 1:
            next_resource = self.rung_levels[current_rung_idx + 1]
            return True, next_resource
        
        return False, None
    
    def get_num_to_run(self, resource):
        """
        计算在当前rung应该运行多少配置
        
        ASHA的异步特性允许我们动态决定
        """
        # 简化的策略：每个rung最多运行 reduction_factor^2 个配置
        max_configs = self.reduction_factor ** 2
        current_running = len(self.rungs[resource])
        return max(0, max_configs - current_running)


def demo_asha():
    """演示ASHA的异步特性"""
    
    asha = ASHA(min_resource=1, max_resource=27, reduction_factor=3)
    
    print("\n模拟ASHA执行过程:")
    print("-" * 50)
    
    config_id = 0
    active_configs = deque()  # 正在运行的配置
    
    # 初始启动一些配置
    for _ in range(5):
        active_configs.append({
            'id': config_id,
            'resource': 1,
            'score': None
        })
        config_id += 1
    
    completed = []
    
    while active_configs:
        # 模拟一个配置完成
        current = active_configs.popleft()
        
        # 模拟训练结果
        current['score'] = np.random.beta(2, 5) * (1 - np.exp(-0.1 * current['resource']))
        
        promote, next_r = asha.should_promote(
            current['id'], current['score'], current['resource']
        )
        
        status = "✓ 完成" if not promote else f"↑ 晋升到r={next_r}"
        print(f"配置{current['id']}: r={current['resource']}, "
              f"score={current['score']:.3f} {status}")
        
        if promote:
            active_configs.append({
                'id': current['id'],
                'resource': next_r,
                'score': None
            })
        else:
            completed.append(current)
        
        # 异步启动新配置（如果rung1有空位）
        if len([c for c in active_configs if c['resource'] == 1]) < 3:
            active_configs.append({
                'id': config_id,
                'resource': 1,
                'score': None
            })
            config_id += 1
    
    print("-" * 50)
    print(f"总共评估配置数: {config_id}")
    print(f"完成全部流程的配置: {len(completed)}")


if __name__ == "__main__":
    demo_asha()