"""
基于进化算法的神经架构搜索(Evolutionary NAS)
实现AmoebaNet风格的大规模进化搜索
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import List
from copy import deepcopy
from dataclasses import dataclass, field

@dataclass
class Architecture:
    """架构个体"""
    normal_cell: List = field(default_factory=list)
    reduction_cell: List = field(default_factory=list)
    fitness: float = 0.0
    age: int = 0
    id: int = 0
    
    def __lt__(self, other): return self.fitness < other.fitness


class ArchitectureMutator:
    """架构突变器"""
    
    def __init__(self, num_blocks=5, num_ops=7):
        self.num_blocks = num_blocks
        self.num_ops = num_ops
        self.operations = ['none', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5',
                          'dil_conv_3x3', 'avg_pool_3x3', 'max_pool_3x3']
    
    def mutate(self, arch: Architecture) -> Architecture:
        """对架构进行突变"""
        new_arch = deepcopy(arch)
        new_arch.age = 0
        
        mutation_type = random.choice(['mutate_normal_op', 'mutate_connection', 'add_block'])
        
        if mutation_type == 'mutate_normal_op':
            new_arch.normal_cell = self._mutate_operations(new_arch.normal_cell)
        elif mutation_type == 'mutate_connection':
            new_arch.normal_cell = self._mutate_connections(new_arch.normal_cell)
        elif mutation_type == 'add_block' and len(new_arch.normal_cell) < 10:
            new_arch.normal_cell = self._add_block(new_arch.normal_cell)
        
        return new_arch
    
    def _mutate_operations(self, cell: List) -> List:
        if not cell: return cell
        new_cell = deepcopy(cell)
        block_idx = random.randint(0, len(new_cell) - 1)
        block = list(new_cell[block_idx])
        op_positions = [i for i, item in enumerate(block) if isinstance(item, tuple) and item[0] in ['op1', 'op2']]
        if op_positions:
            op_idx = random.choice(op_positions)
            old_op_name, _ = block[op_idx]
            block[op_idx] = (old_op_name, random.randint(0, self.num_ops - 1))
        new_cell[block_idx] = tuple(block)
        return new_cell
    
    def _mutate_connections(self, cell: List) -> List:
        if not cell: return cell
        new_cell = deepcopy(cell)
        block_idx = random.randint(0, len(new_cell) - 1)
        block = list(new_cell[block_idx])
        pred_positions = [i for i, item in enumerate(block) if isinstance(item, tuple) and item[0] in ['pre1', 'pre2']]
        if pred_positions:
            pred_idx = random.choice(pred_positions)
            old_pred_name, _ = block[pred_idx]
            max_pred = 2 + block_idx
            block[pred_idx] = (old_pred_name, random.randint(0, max_pred - 1))
        new_cell[block_idx] = tuple(block)
        return new_cell
    
    def _add_block(self, cell: List) -> List:
        new_cell = deepcopy(cell)
        new_block_idx = len(new_cell)
        max_pred = 2 + new_block_idx
        block = (
            ('pre1', random.randint(0, max_pred - 1)),
            ('op1', random.randint(0, self.num_ops - 1)),
            ('pre2', random.randint(0, max_pred - 1)),
            ('op2', random.randint(0, self.num_ops - 1)),
        )
        new_cell.append(block)
        return new_cell


class EvolutionarySearcher:
    """进化搜索器（含老化进化）"""
    
    def __init__(self, population_size=20, sample_size=5, num_blocks=3):
        self.population_size = population_size
        self.sample_size = sample_size
        self.mutator = ArchitectureMutator(num_blocks)
        self.population: List[Architecture] = []
        self.generation = 0
        self.arch_id_counter = 0
    
    def initialize_population(self, evaluator):
        """初始化种群"""
        print("初始化种群...")
        while len(self.population) < self.population_size:
            arch = self._random_architecture()
            arch.fitness = evaluator(arch)
            arch.id = self.arch_id_counter
            self.arch_id_counter += 1
            self.population.append(arch)
            print(f"  个体 {len(self.population)}/{self.population_size}: fitness={arch.fitness:.4f}")
    
    def evolve(self, evaluator, num_generations=10):
        """执行进化搜索"""
        print(f"\n开始进化搜索，目标代数: {num_generations}")
        
        for gen in range(num_generations):
            self.generation = gen
            for arch in self.population: arch.age += 1
            
            parents = self._tournament_selection()
            
            for parent in parents:
                child = self.mutator.mutate(parent)
                child.fitness = evaluator(child)
                child.id = self.arch_id_counter
                self.arch_id_counter += 1
                self.population.append(child)
            
            self._aging_selection()
            
            best_arch = max(self.population, key=lambda x: x.fitness)
            avg_fitness = sum(a.fitness for a in self.population) / len(self.population)
            print(f"Gen {gen+1}/{num_generations}: Best={best_arch.fitness:.4f}, Avg={avg_fitness:.4f}")
        
        return max(self.population, key=lambda x: x.fitness)
    
    def _tournament_selection(self):
        """锦标赛选择（含老化）"""
        parents = []
        for _ in range(self.sample_size // 2):
            tournament = random.sample(self.population, min(self.sample_size, len(self.population)))
            winner = max(tournament, key=lambda a: a.fitness - 0.1 * a.age)
            parents.append(winner)
        return parents
    
    def _aging_selection(self):
        """老化环境选择"""
        def survival_score(arch):
            return arch.fitness + max(0, 10 - arch.age) * 0.01
        self.population.sort(key=survival_score, reverse=True)
        self.population = self.population[:self.population_size]
    
    def _random_architecture(self) -> Architecture:
        arch = Architecture()
        for i in range(self.mutator.num_blocks):
            max_pred = 2 + i
            block = (('pre1', random.randint(0, max_pred - 1)),
                    ('op1', random.randint(0, self.mutator.num_ops - 1)),
                    ('pre2', random.randint(0, max_pred - 1)),
                    ('op2', random.randint(0, self.mutator.num_ops - 1)))
            arch.normal_cell.append(block)
        return arch


def evaluator(arch):
    """模拟评估器"""
    base = random.uniform(0.5, 0.9)
    complexity = len(arch.normal_cell)
    return max(0.0, min(1.0, base - complexity * 0.01 + random.gauss(0, 0.05)))


if __name__ == '__main__':
    print("=" * 60)
    print("进化算法NAS测试")
    print("=" * 60)
    
    searcher = EvolutionarySearcher(population_size=10, sample_size=4, num_blocks=3)
    searcher.initialize_population(evaluator)
    best_arch = searcher.evolve(evaluator, num_generations=5)
    
    print("\n" + "=" * 60)
    print(f"最佳架构 ID: {best_arch.id}, 适应度: {best_arch.fitness:.4f}")
    print("=" * 60)
