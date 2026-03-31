"""
多目标神经架构搜索 (Multi-Objective NAS)
实现Pareto最优架构搜索，平衡准确率、延迟、能耗等多个目标
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
from copy import deepcopy
import random


@dataclass
class Architecture:
    """架构表示"""
    genotype: Dict  # 架构基因型
    objectives: Dict[str, float] = None  # 各目标值
    
    def __hash__(self):
        return hash(str(self.genotype))
    
    def dominates(self, other: 'Architecture') -> bool:
        """
        Pareto支配判断：self是否支配other
        
        支配定义：在所有目标上都不差，至少在一个目标上严格更好
        """
        if self.objectives is None or other.objectives is None:
            return False
        
        not_worse = all(
            self.objectives[k] <= other.objectives[k] 
            for k in self.objectives.keys()
        )
        strictly_better = any(
            self.objectives[k] < other.objectives[k] 
            for k in self.objectives.keys()
        )
        return not_worse and strictly_better


class ParetoFrontier:
    """
    Pareto前沿：维护一组互不支配的架构
    
    费曼法比喻：想象你在买车。
    - 车A：便宜但慢
    - 车B：贵但快
    - 车C：比A贵但比A慢（被A支配，应该淘汰）
    
    Pareto前沿就是那些"各有所长"的选择。
    """
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.architectures: List[Architecture] = []
        
    def add(self, arch: Architecture) -> bool:
        """
        添加新架构到Pareto前沿
        
        Args:
            arch: 待添加的架构
            
        Returns:
            是否成功添加
        """
        # 检查是否被现架构支配
        for existing in self.architectures:
            if existing.dominates(arch):
                return False  # 被支配，不添加
        
        # 移除被新架构支配的现有架构
        self.architectures = [
            existing for existing in self.architectures
            if not arch.dominates(existing)
        ]
        
        # 添加新架构
        self.architectures.append(arch)
        
        # 如果超出容量，移除拥挤度最小的
        if len(self.architectures) > self.max_size:
            self._prune()
            
        return True
    
    def _prune(self):
        """通过拥挤度剪枝"""
        if len(self.architectures) <= 2:
            return
        
        # 计算每个架构的拥挤度
        crowding_distances = self._compute_crowding_distance()
        
        # 按拥挤度排序，保留最分散的
        sorted_indices = np.argsort(crowding_distances)[::-1]
        self.architectures = [
            self.architectures[i] for i in sorted_indices[:self.max_size]
        ]
    
    def _compute_crowding_distance(self) -> np.ndarray:
        """
        计算拥挤度距离
        
        拥挤度高的架构位于Pareto前沿的"稀疏"区域，更值得保留
        """
        if len(self.architectures) <= 2:
            return np.ones(len(self.architectures))
        
        distances = np.zeros(len(self.architectures))
        
        # 对每个目标计算拥挤度
        objectives = list(self.architectures[0].objectives.keys())
        
        for obj in objectives:
            values = np.array([a.objectives[obj] for a in self.architectures])
            sorted_indices = np.argsort(values)
            
            # 边界点距离设为无穷大
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            # 中间点的距离
            obj_range = values.max() - values.min()
            if obj_range > 0:
                for i in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[i]
                    prev_val = values[sorted_indices[i - 1]]
                    next_val = values[sorted_indices[i + 1]]
                    distances[idx] += (next_val - prev_val) / obj_range
        
        return distances
    
    def get_best_for_preference(
        self, 
        preference: Dict[str, float]
    ) -> Architecture:
        """
        根据偏好向量选择最佳架构
        
        Args:
            preference: 各目标的权重，如{'accuracy': -1.0, 'latency': 0.5}
            
        Returns:
            加权得分最高的架构
        """
        best_arch = None
        best_score = float('inf')
        
        for arch in self.architectures:
            score = sum(
                preference[obj] * arch.objectives[obj]
                for obj in preference.keys()
            )
            if score < best_score:
                best_score = score
                best_arch = arch
                
        return best_arch


class NSGA2_NAS:
    """
    NSGA-II算法用于多目标神经架构搜索
    
    NSGA-II (Non-dominated Sorting Genetic Algorithm II) 是一种经典的多目标优化算法。
    核心思想：
    1. 非支配排序：将种群分层，每层是一组互不支配的解
    2. 拥挤度计算：在同一层内，选择分布更分散的解
    
    费曼法比喻：想象你在组织一场才艺比赛。
    - 非支配排序：第一轮选出"各有所长"的选手（会唱歌的、会跳舞的、会魔术的）
    - 拥挤度计算：如果太多人都会唱歌，就选其中风格最独特的
    """
    def __init__(
        self,
        population_size: int = 50,
        num_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.9,
        num_objectives: int = 2
    ):
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_objectives = num_objectives
        
        # 历史最优Pareto前沿
        self.pareto_frontier = ParetoFrontier(max_size=100)
        
    def search(
        self,
        evaluate_fn: Callable[[Architecture], Dict[str, float]],
        init_population_fn: Callable[[], List[Architecture]]
    ) -> ParetoFrontier:
        """
        执行多目标搜索
        
        Args:
            evaluate_fn: 评估函数，返回各目标值
            init_population_fn: 初始化种群函数
            
        Returns:
            最终的Pareto前沿
        """
        # 初始化种群
        population = init_population_fn()
        
        # 评估初始种群
        for arch in population:
            arch.objectives = evaluate_fn(arch)
            self.pareto_frontier.add(arch)
        
        for generation in range(self.num_generations):
            # 生成子代
            offspring = self._generate_offspring(population)
            
            # 评估子代
            for arch in offspring:
                arch.objectives = evaluate_fn(arch)
                self.pareto_frontier.add(arch)
            
            # 合并父代和子代
            combined = population + offspring
            
            # 环境选择
            population = self._environmental_selection(combined)
            
            # 打印进度
            if generation % 10 == 0:
                print(f"Generation {generation}: Pareto size = {len(self.pareto_frontier.architectures)}")
        
        return self.pareto_frontier
    
    def _generate_offspring(
        self, 
        population: List[Architecture]
    ) -> List[Architecture]:
        """生成子代种群"""
        offspring = []
        
        while len(offspring) < self.population_size:
            # 二元锦标赛选择
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)
            
            # 交叉
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = deepcopy(parent1), deepcopy(parent2)
            
            # 变异
            if random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.mutation_rate:
                child2 = self._mutate(child2)
            
            offspring.extend([child1, child2])
        
        return offspring[:self.population_size]
    
    def _tournament_select(
        self, 
        population: List[Architecture],
        tournament_size: int = 2
    ) -> Architecture:
        """二元锦标赛选择"""
        candidates = random.sample(population, min(tournament_size, len(population)))
        
        # 非支配排序和拥挤度已在population中
        # 简单实现：随机选择
        return random.choice(candidates)
    
    def _crossover(
        self, 
        p1: Architecture, 
        p2: Architecture
    ) -> Tuple[Architecture, Architecture]:
        """单点交叉"""
        c1_genotype = deepcopy(p1.genotype)
        c2_genotype = deepcopy(p2.genotype)
        
        # 随机选择交叉点
        keys = list(c1_genotype.keys())
        if len(keys) > 0:
            point = random.randint(1, len(keys) - 1)
            for key in keys[point:]:
                c1_genotype[key], c2_genotype[key] = c2_genotype[key], c1_genotype[key]
        
        return Architecture(c1_genotype), Architecture(c2_genotype)
    
    def _mutate(self, arch: Architecture) -> Architecture:
        """变异操作"""
        mutated = deepcopy(arch)
        
        # 随机改变某个基因
        keys = list(mutated.genotype.keys())
        if keys:
            key = random.choice(keys)
            # 根据基因类型进行不同变异
            if isinstance(mutated.genotype[key], int):
                mutated.genotype[key] = max(0, mutated.genotype[key] + random.randint(-1, 1))
            elif isinstance(mutated.genotype[key], str):
                ops = ['conv3x3', 'conv5x5', 'dconv3x3', 'maxpool', 'avgpool', 'skip']
                mutated.genotype[key] = random.choice(ops)
        
        return mutated
    
    def _environmental_selection(
        self, 
        combined: List[Architecture]
    ) -> List[Architecture]:
        """环境选择：基于非支配排序和拥挤度"""
        # 非支配排序
        fronts = self._non_dominated_sort(combined)
        
        # 选择下一代
        next_population = []
        for front in fronts:
            if len(next_population) + len(front) <= self.population_size:
                next_population.extend(front)
            else:
                # 按拥挤度排序，选择最分散的
                remaining = self.population_size - len(next_population)
                
                # 计算拥挤度
                pf = ParetoFrontier()
                pf.architectures = front
                distances = pf._compute_crowding_distance()
                
                sorted_indices = np.argsort(distances)[::-1]
                for i in range(remaining):
                    next_population.append(front[sorted_indices[i]])
                break
        
        return next_population
    
    def _non_dominated_sort(
        self, 
        population: List[Architecture]
    ) -> List[List[Architecture]]:
        """
        非支配排序
        
        返回分层结果，每层是一组互不支配的架构
        """
        fronts = []
        remaining = set(range(len(population)))
        
        while remaining:
            current_front = []
            to_remove = []
            
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i != j and population[j].dominates(population[i]):
                        dominated = True
                        break
                if not dominated:
                    current_front.append(population[i])
                    to_remove.append(i)
            
            if not current_front:
                break
            
            fronts.append(current_front)
            remaining -= set(to_remove)
        
        return fronts


class ScalarizationNAS:
    """
    基于标量化（Scalarization）的多目标NAS
    
    思想：将多个目标组合成一个标量目标，然后优化
    常用方法：
    1. 加权求和：min λ₁f₁ + λ₂f₂ + ...
    2. 加权切比雪夫：min max(λ₁|f₁-z₁|, λ₂|f₂-z₂|, ...)
    """
    def __init__(
        self,
        num_reference_points: int = 10,
        search_algorithm: str = 'darts'
    ):
        self.num_reference_points = num_reference_points
        self.search_algorithm = search_algorithm
        
        # 生成参考点（均匀分布在单纯形上）
        self.reference_points = self._generate_reference_points()
        
    def _generate_reference_points(self) -> List[np.ndarray]:
        """
        生成均匀分布的参考点
        
        使用单纯形格点法生成均匀分布的权重向量
        """
        points = []
        H = self.num_reference_points - 1
        
        for i in range(H + 1):
            w1 = i / H
            w2 = 1 - w1
            points.append(np.array([w1, w2]))
        
        return points
    
    def scalarized_loss(
        self,
        objectives: Dict[str, float],
        weights: np.ndarray,
        method: str = 'weighted_sum'
    ) -> float:
        """
        计算标量化损失
        
        Args:
            objectives: 各目标值，如{'accuracy': 0.95, 'latency': 10}
            weights: 权重向量
            method: 标量化方法
            
        Returns:
            标量损失值
        """
        values = np.array(list(objectives.values()))
        
        if method == 'weighted_sum':
            return np.dot(weights, values)
        elif method == 'weighted_tchebycheff':
            # 切比雪夫方法
            ideal = np.zeros_like(values)  # 理想点
            return np.max(weights * np.abs(values - ideal))
        elif method == 'achievement_scalarizing':
            # 成就标量化函数
            rho = 0.001
            return np.max(weights * values) + rho * np.sum(weights * values)
        else:
            raise ValueError(f"Unknown method: {method}")


# ====== 使用示例 ======
def demo_multi_objective_nas():
    """多目标NAS演示"""
    print("=" * 60)
    print("多目标神经架构搜索演示")
    print("目标：在准确率和延迟之间寻找平衡")
    print("=" * 60)
    
    # 模拟评估函数
    def evaluate_arch(arch: Architecture) -> Dict[str, float]:
        """
        模拟架构评估
        
        这里使用简单的启发式：
        - 更大的网络通常更准确但更慢
        - 更深的网络通常更准确但更慢
        """
        depth = arch.genotype.get('depth', 5)
        width = arch.genotype.get('width', 32)
        
        # 模拟准确率（随深度和宽度增加）
        accuracy = 0.7 + 0.2 * (1 - np.exp(-depth/10)) + 0.1 * (1 - np.exp(-width/100))
        accuracy += np.random.normal(0, 0.02)  # 添加噪声
        
        # 模拟延迟（随深度和宽度增加）
        latency = 5 + depth * 2 + width * 0.1 + np.random.normal(0, 1)
        
        # 返回（注意：对于最小化问题，准确率取负）
        return {
            'error_rate': 1 - accuracy,  # 错误率（越小越好）
            'latency': latency  # 延迟（越小越好）
        }
    
    # 初始化种群
    def init_population() -> List[Architecture]:
        population = []
        for _ in range(30):
            genotype = {
                'depth': random.randint(3, 20),
                'width': random.randint(16, 128),
                'op_type': random.choice(['conv3x3', 'conv5x5', 'mbconv'])
            }
            population.append(Architecture(genotype))
        return population
    
    # 运行NSGA-II搜索
    print("\n运行NSGA-II多目标优化...")
    nsga2 = NSGA2_NAS(
        population_size=30,
        num_generations=30,
        mutation_rate=0.2,
        crossover_rate=0.8
    )
    
    pareto_front = nsga2.search(evaluate_arch, init_population)
    
    print(f"\n搜索完成！Pareto前沿包含 {len(pareto_front.architectures)} 个架构")
    
    # 打印Pareto前沿
    print("\nPareto最优架构：")
    print("-" * 60)
    print(f"{'架构':<20} {'错误率':<10} {'延迟(ms)':<10}")
    print("-" * 60)
    
    for i, arch in enumerate(pareto_front.architectures[:10]):
        print(f"Arch-{i+1:<15} {arch.objectives['error_rate']:.4f}     {arch.objectives['latency']:.2f}")
    
    # 测试偏好选择
    print("\n根据偏好选择架构：")
    
    # 偏好1：优先考虑准确率
    pref_accuracy = {'error_rate': 1.0, 'latency': 0.1}
    best_acc = pareto_front.get_best_for_preference(pref_accuracy)
    print(f"优先准确率：错误率={best_acc.objectives['error_rate']:.4f}, "
          f"延迟={best_acc.objectives['latency']:.2f}ms")
    
    # 偏好2：优先考虑延迟
    pref_latency = {'error_rate': 0.1, 'latency': 1.0}
    best_lat = pareto_front.get_best_for_preference(pref_latency)
    print(f"优先延迟：错误率={best_lat.objectives['error_rate']:.4f}, "
          f"延迟={best_lat.objectives['latency']:.2f}ms")
    
    # 偏好3：平衡
    pref_balanced = {'error_rate': 0.5, 'latency': 0.5}
    best_bal = pareto_front.get_best_for_preference(pref_balanced)
    print(f"平衡选择：错误率={best_bal.objectives['error_rate']:.4f}, "
          f"延迟={best_bal.objectives['latency']:.2f}ms")
    
    print("\n" + "=" * 60)
    print("多目标NAS演示完成")
    print("=" * 60)


if __name__ == "__main__":
    demo_multi_objective_nas()
