"""
MultiObjectiveNAS: 多目标神经架构搜索
===================================

使用NSGA-II算法同时优化多个目标：
- 准确率 (Accuracy)
- 参数量 (Parameters)
- 计算量 (FLOPs)
- 推理延迟 (Latency)

费曼法比喻：
想象你要买一辆自行车，但你有多个要求：
- 要快（准确率高）
- 要轻（参数量小）
- 要便宜（计算量小）
- 要好看（延迟低）

没有一辆车能同时满足所有要求，
NSGA-II帮你找到一系列"无法互相超越"的选择（Pareto前沿），
让你根据自己的偏好做出最终决定。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from dataclasses import dataclass, field
from collections import deque
import random
import copy
from tqdm import tqdm


@dataclass
class Individual:
    """
    进化算法中的个体
    
    就像一个DNA链，编码了一个神经网络的架构信息
    """
    genome: np.ndarray  # 基因型：编码架构的数组
    objectives: np.ndarray = field(default_factory=lambda: np.array([]))
    rank: int = 0       # 非支配排序等级
    crowding_distance: float = 0.0  # 拥挤距离
    dominates: List[int] = field(default_factory=list)
    dominated_count: int = 0
    
    def __hash__(self):
        return hash(self.genome.tobytes())


class ArchitectureEncoder:
    """
    架构编码器
    
    将神经网络架构编码为固定长度的向量。
    就像把建筑的蓝图转换成一串数字代码。
    """
    
    def __init__(self, max_layers: int = 8, num_operations: int = 7):
        self.max_layers = max_layers
        self.num_operations = num_operations
        self.operation_choices = [
            'none', 'skip_connect', 'conv_3x3', 'conv_5x5',
            'dil_conv_3x3', 'sep_conv_3x3', 'avg_pool_3x3'
        ]
    
    def encode(self, architecture: Dict) -> np.ndarray:
        """
        编码架构为向量
        
        编码格式：[每层的操作, 每层的输入连接, ...]
        """
        genome = []
        
        # 编码每层的操作类型
        for i in range(self.max_layers):
            if i < len(architecture.get('layers', [])):
                op = architecture['layers'][i]
                op_idx = self.operation_choices.index(op) \
                         if op in self.operation_choices else 0
            else:
                op_idx = 0
            genome.append(op_idx)
        
        # 编码连接关系
        for i in range(self.max_layers):
            if i < len(architecture.get('connections', [])):
                conn = architecture['connections'][i]
                genome.append(conn if conn < i else i - 1)
            else:
                genome.append(max(0, i - 1))
        
        # 编码通道数
        for i in range(self.max_layers):
            if i < len(architecture.get('channels', [])):
                ch = architecture['channels'][i]
                genome.append(int(np.log2(ch)) if ch > 0 else 4)
            else:
                genome.append(4)
        
        return np.array(genome, dtype=np.int32)
    
    def decode(self, genome: np.ndarray) -> Dict:
        """解码向量为架构"""
        layers = []
        connections = []
        channels = []
        
        # 解码操作
        for i in range(self.max_layers):
            op_idx = int(genome[i]) % self.num_operations
            layers.append(self.operation_choices[op_idx])
        
        # 解码连接
        for i in range(self.max_layers):
            conn_idx = self.max_layers + i
            if conn_idx < len(genome):
                connections.append(int(genome[conn_idx]) % max(1, i))
            else:
                connections.append(max(0, i - 1))
        
        # 解码通道数
        for i in range(self.max_layers):
            ch_idx = 2 * self.max_layers + i
            if ch_idx < len(genome):
                channels.append(2 ** (int(genome[ch_idx]) % 6 + 4))
            else:
                channels.append(64)
        
        return {
            'layers': layers,
            'connections': connections,
            'channels': channels
        }
    
    def get_genome_length(self) -> int:
        """获取基因长度"""
        return self.max_layers * 3


class NSGA2:
    """
    NSGA-II: 非支配排序遗传算法 II
    
    这是多目标优化的经典算法，就像一个人才选拔系统：
    1. 先按"综合能力"排序（非支配排序）
    2. 同等级内按"独特性"排序（拥挤距离）
    3. 选择优秀的"父母"产生"后代"
    4. 保留最优秀的人才
    
    核心理念：不找一个"最好"的，而是找一群"各有特色"的
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 num_generations: int = 100,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.1,
                 num_objectives: int = 2):
        
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.num_objectives = num_objectives
        
        self.encoder = ArchitectureEncoder()
        self.genome_length = self.encoder.get_genome_length()
        
        # 历史记录
        self.history = {
            'pareto_fronts': [],
            'hypervolume': []
        }
    
    def dominates_check(self, ind1: Individual, ind2: Individual) -> bool:
        """
        检查个体1是否支配个体2
        
        支配定义：
        - 在所有目标上都不比个体2差
        - 在至少一个目标上严格优于个体2
        
        比喻：
        学生A支配学生B，意味着：
        - A的数学 >= B的数学
        - A的语文 >= B的语文
        - A的英语 >= B的英语
        - 且至少有一科 A > B
        """
        not_worse = np.all(ind1.objectives <= ind2.objectives)
        strictly_better = np.any(ind1.objectives < ind2.objectives)
        return not_worse and strictly_better
    
    def non_dominated_sort(self, population: List[Individual]) -> List[List[int]]:
        """
        非支配排序
        
        将种群分成多个"前沿"：
        - 第1前沿：不被任何人支配（最优秀）
        - 第2前沿：只被第1前沿支配
        - 第3前沿：被第1、2前沿支配
        - ...以此类推
        
        就像把运动员按金牌、银牌、铜牌分级
        """
        n = len(population)
        
        # 初始化
        for i, ind in enumerate(population):
            ind.dominates = []
            ind.dominated_count = 0
            ind.rank = 0
        
        fronts = [[]]  # 第0层前沿
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.dominates_check(population[i], population[j]):
                    population[i].dominates.append(j)
                    population[j].dominated_count += 1
                elif self.dominates_check(population[j], population[i]):
                    population[j].dominates.append(i)
                    population[i].dominated_count += 1
            
            # 如果没有被任何人支配，加入第0层前沿
            if population[i].dominated_count == 0:
                population[i].rank = 0
                fronts[0].append(i)
        
        # 逐层构建前沿
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in population[p].dominates:
                    population[q].dominated_count -= 1
                    if population[q].dominated_count == 0:
                        population[q].rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        # 去掉最后一个空的前沿
        if len(fronts[-1]) == 0:
            fronts.pop()
        
        return fronts
    
    def calculate_crowding_distance(self, front: List[int], 
                                    population: List[Individual]):
        """
        计算拥挤距离
        
        衡量个体在目标空间中的"孤独程度"。
        距离越大，说明周围越空旷，越应该保留（保持多样性）。
        
        比喻：在停车场找车位，拥挤距离大的位置周围车少
        """
        if len(front) <= 2:
            for idx in front:
                population[idx].crowding_distance = float('inf')
            return
        
        num_obj = len(population[front[0]].objectives)
        
        for idx in front:
            population[idx].crowding_distance = 0
        
        for m in range(num_obj):
            # 按第m个目标排序
            sorted_front = sorted(front, 
                                 key=lambda i: population[i].objectives[m])
            
            # 边界点设为无穷大
            population[sorted_front[0]].crowding_distance = float('inf')
            population[sorted_front[-1]].crowding_distance = float('inf')
            
            # 计算中间点的距离
            f_max = population[sorted_front[-1]].objectives[m]
            f_min = population[sorted_front[0]].objectives[m]
            
            if f_max - f_min > 1e-10:
                for i in range(1, len(sorted_front) - 1):
                    distance = (population[sorted_front[i + 1]].objectives[m] - 
                               population[sorted_front[i - 1]].objectives[m])
                    distance /= (f_max - f_min)
                    population[sorted_front[i]].crowding_distance += distance
    
    def tournament_selection(self, population: List[Individual], 
                            tournament_size: int = 2) -> int:
        """
        锦标赛选择
        
        随机选几个人比一比，赢的当父母。
        先比等级（rank），等级相同比拥挤距离。
        """
        selected = random.sample(range(len(population)), tournament_size)
        winner = selected[0]
        
        for idx in selected[1:]:
            if population[idx].rank < population[winner].rank:
                winner = idx
            elif population[idx].rank == population[winner].rank:
                if population[idx].crowding_distance > population[winner].crowding_distance:
                    winner = idx
        
        return winner
    
    def crossover(self, parent1: np.ndarray, 
                 parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        单点交叉
        
        就像父母各出一半基因给孩子
        """
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        
        return child1, child2
    
    def mutate(self, genome: np.ndarray) -> np.ndarray:
        """
        变异操作
        
        随机改变某些基因，引入新的可能性。
        就像生物突变，有时会产生更好的特性。
        """
        child = genome.copy()
        
        for i in range(len(child)):
            if random.random() < self.mutation_prob:
                # 随机增加或减少
                delta = random.choice([-1, 1])
                child[i] = max(0, child[i] + delta)
        
        return child
    
    def create_offspring(self, population: List[Individual]) -> List[Individual]:
        """创建子代种群"""
        offspring = []
        
        while len(offspring) < self.population_size:
            # 选择父母
            p1_idx = self.tournament_selection(population)
            p2_idx = self.tournament_selection(population)
            
            # 交叉
            child1_genome, child2_genome = self.crossover(
                population[p1_idx].genome,
                population[p2_idx].genome
            )
            
            # 变异
            child1_genome = self.mutate(child1_genome)
            child2_genome = self.mutate(child2_genome)
            
            offspring.append(Individual(genome=child1_genome))
            if len(offspring) < self.population_size:
                offspring.append(Individual(genome=child2_genome))
        
        return offspring
    
    def environmental_selection(self, combined: List[Individual]) -> List[Individual]:
        """
        环境选择
        
        从父母+孩子的混合种群中选择下一代。
        先按非支配排序分层，优先保留低层的；
        同一层内按拥挤距离选择，优先保留分布广泛的。
        """
        fronts = self.non_dominated_sort(combined)
        
        new_population = []
        front_idx = 0
        
        # 完整容纳的前沿
        while front_idx < len(fronts) and \
              len(new_population) + len(fronts[front_idx]) <= self.population_size:
            
            self.calculate_crowding_distance(fronts[front_idx], combined)
            new_population.extend([combined[i] for i in fronts[front_idx]])
            front_idx += 1
        
        # 部分容纳的前沿（按拥挤距离选择）
        if front_idx < len(fronts) and len(new_population) < self.population_size:
            self.calculate_crowding_distance(fronts[front_idx], combined)
            last_front = sorted(fronts[front_idx],
                              key=lambda i: combined[i].crowding_distance,
                              reverse=True)
            
            remaining = self.population_size - len(new_population)
            new_population.extend([combined[i] for i in last_front[:remaining]])
        
        return new_population
    
    def optimize(self, 
                 evaluate_fn: Callable[[np.ndarray], np.ndarray],
                 verbose: bool = True) -> List[Individual]:
        """
        执行NSGA-II优化
        
        Args:
            evaluate_fn: 评估函数，输入基因型，输出目标值数组
        """
        # 初始化种群
        population = []
        for _ in range(self.population_size):
            genome = np.random.randint(0, 7, size=self.genome_length)
            population.append(Individual(genome=genome))
        
        # 评估初始种群
        for ind in population:
            ind.objectives = evaluate_fn(ind.genome)
        
        iterator = tqdm(range(self.num_generations)) if verbose else \
                   range(self.num_generations)
        
        for gen in iterator:
            # 创建子代
            offspring = self.create_offspring(population)
            
            # 评估子代
            for ind in offspring:
                ind.objectives = evaluate_fn(ind.genome)
            
            # 合并并选择
            combined = population + offspring
            population = self.environmental_selection(combined)
            
            # 记录历史
            fronts = self.non_dominated_sort(population)
            if fronts:
                first_front = [population[i] for i in fronts[0]]
                self.history['pareto_fronts'].append(first_front)
                
                # 计算超体积（简化版本）
                hv = self.calculate_hypervolume(first_front)
                self.history['hypervolume'].append(hv)
            
            if verbose and gen % 10 == 0:
                avg_rank = np.mean([ind.rank for ind in population])
                iterator.set_description(
                    f"Gen {gen}, Avg Rank: {avg_rank:.2f}, HV: {hv:.4f}"
                )
        
        return population
    
    def calculate_hypervolume(self, front: List[Individual],
                             reference_point: Optional[np.ndarray] = None) -> float:
        """
        计算超体积指标
        
        衡量Pareto前沿覆盖的"面积"。
        超体积越大，说明找到的非支配解越好、越多样。
        
        比喻：超体积就像Pareto前沿围成的领地大小
        """
        if not front:
            return 0.0
        
        objectives = np.array([ind.objectives for ind in front])
        
        if reference_point is None:
            reference_point = np.max(objectives, axis=0) + 0.1
        
        # 简化计算：矩形法近似
        # 先按第一个目标排序
        sorted_indices = np.argsort(objectives[:, 0])
        sorted_obj = objectives[sorted_indices]
        
        volume = 0.0
        prev_x = reference_point[0]
        
        for obj in sorted_obj:
            dx = prev_x - obj[0]
            dy = reference_point[1] - obj[1] if len(obj) > 1 else 1.0
            volume += dx * dy
            prev_x = obj[0]
        
        return volume
    
    def get_pareto_front(self, population: List[Individual]) -> List[Individual]:
        """获取最终的Pareto前沿"""
        fronts = self.non_dominated_sort(population)
        if fronts:
            return [population[i] for i in fronts[0]]
        return []


class MultiObjectiveEvaluator:
    """
    多目标评估器
    
    用于演示的简化评估函数。
    实际使用时，这里应该进行真实的网络训练和评估。
    """
    
    def __init__(self, 
                 objectives: List[str] = None,
                 noise_level: float = 0.1):
        self.objectives = objectives or ['error', 'params', 'latency']
        self.noise_level = noise_level
        self.encoder = ArchitectureEncoder()
    
    def evaluate(self, genome: np.ndarray) -> np.ndarray:
        """
        评估架构的多个目标
        
        模拟一个真实场景：
        - 深层网络通常更准确但更慢
        - 卷积操作比池化更昂贵
        - 宽网络参数量大
        """
        architecture = self.encoder.decode(genome)
        
        results = []
        
        for obj in self.objectives:
            if obj == 'error':
                # 错误率：复杂架构通常误差更小
                depth = sum(1 for l in architecture['layers'] if l != 'none')
                conv_ratio = sum(1 for l in architecture['layers'] 
                               if 'conv' in l) / max(1, len(architecture['layers']))
                error = 0.5 - 0.3 * (depth / 8) - 0.1 * conv_ratio
                error += self.noise_level * np.random.randn()
                error = max(0.05, min(0.5, error))
                results.append(error)
            
            elif obj == 'params':
                # 参数量：与通道数和层数相关
                total_channels = sum(architecture['channels'])
                params = total_channels * total_channels / 1000000
                params += self.noise_level * np.random.randn() * 0.1
                params = max(0.1, params)
                results.append(params)
            
            elif obj == 'latency':
                # 延迟：与操作复杂度和层数相关
                latency = 0
                for layer in architecture['layers']:
                    if layer == 'none':
                        latency += 0.1
                    elif layer == 'skip_connect':
                        latency += 0.5
                    elif 'conv' in layer:
                        latency += 2.0
                    else:
                        latency += 1.0
                latency += self.noise_level * np.random.randn()
                latency = max(1.0, latency)
                results.append(latency)
            
            elif obj == 'memory':
                # 内存占用
                memory = sum(architecture['channels']) * 4 / 1024  # MB
                memory += self.noise_level * np.random.randn() * 0.1
                results.append(max(0.5, memory))
        
        return np.array(results)


def demo_nsga2():
    """
    NSGA-II演示
    
    思考题：
    1. 如何修改交叉和变异策略以适应架构搜索？
    2. 在高维目标空间（>3个目标）中，NSGA-II会遇到什么挑战？
    3. 超体积指标有什么优缺点？
    """
    print("=" * 70)
    print("NSGA-II 多目标神经架构搜索演示")
    print("=" * 70)
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 创建评估器
    evaluator = MultiObjectiveEvaluator(
        objectives=['error', 'params', 'latency']
    )
    
    # 创建NSGA-II优化器
    nsga2 = NSGA2(
        population_size=30,
        num_generations=50,
        crossover_prob=0.9,
        mutation_prob=0.2,
        num_objectives=3
    )
    
    print("\n开始优化...")
    print("目标：最小化 [错误率, 参数量, 延迟]")
    print()
    
    # 运行优化
    final_population = nsga2.optimize(
        evaluate_fn=evaluator.evaluate,
        verbose=True
    )
    
    # 获取Pareto前沿
    pareto_front = nsga2.get_pareto_front(final_population)
    
    print(f"\n优化完成！")
    print(f"最终Pareto前沿大小: {len(pareto_front)}")
    print()
    
    # 打印Pareto前沿
    print("Pareto前沿上的解:")
    print("-" * 70)
    print(f"{'Rank':<6}{'Error':<10}{'Params(M)':<12}{'Latency(ms)':<12}")
    print("-" * 70)
    
    for i, ind in enumerate(sorted(pareto_front, 
                                   key=lambda x: x.objectives[0])):
        print(f"{i+1:<6}{ind.objectives[0]:<10.4f}"
              f"{ind.objectives[1]:<12.4f}{ind.objectives[2]:<12.4f}")
    
    print("-" * 70)
    
    # 超体积变化
    if nsga2.history['hypervolume']:
        print(f"\n超体积指标:")
        print(f"  初始: {nsga2.history['hypervolume'][0]:.4f}")
        print(f"  最终: {nsga2.history['hypervolume'][-1]:.4f}")
        print(f"  提升: {(nsga2.history['hypervolume'][-1] / 
                      nsga2.history['hypervolume'][0] - 1) * 100:.2f}%")
    
    # 解码并显示一个示例架构
    print("\n示例架构解码:")
    if pareto_front:
        example = pareto_front[len(pareto_front)//2]
        arch = nsga2.encoder.decode(example.genome)
        print(f"  前4层操作: {arch['layers'][:4]}")
        print(f"  通道数: {arch['channels'][:4]}")
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    
    return nsga2, pareto_front


if __name__ == "__main__":
    demo_nsga2()
