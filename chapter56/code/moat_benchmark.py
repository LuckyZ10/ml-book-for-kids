"""
MOATBenchmark: 多目标架构评估基准
================================

MOAT = Multi-Objective Architecture Benchmark
提供标准化的多目标NAS评估框架。

费曼法比喻：
就像学校考试，需要：
- 统一的评分标准（指标）
- 固定的考场（数据集）
- 公平的监考（评估流程）

MOAT就是这样一个"考场"，
让不同的NAS算法在同一起跑线上公平竞争。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time
import json


@dataclass
class ArchitectureMetrics:
    """架构评估指标"""
    # 性能指标
    top1_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    
    # 效率指标
    params_count: int = 0
    flops: float = 0.0
    memory_mb: float = 0.0
    
    # 延迟指标
    latency_cpu_ms: float = 0.0
    latency_gpu_ms: float = 0.0
    throughput: float = 0.0  # images/sec
    
    # 训练指标
    train_time_sec: float = 0.0
    convergence_epochs: int = 0
    
    # 其他
    energy_joules: float = 0.0  # 能耗
    carbon_kg: float = 0.0      # 碳排放
    
    def to_dict(self) -> Dict:
        return {
            'top1_accuracy': self.top1_accuracy,
            'top5_accuracy': self.top5_accuracy,
            'params_count': self.params_count,
            'flops': self.flops,
            'memory_mb': self.memory_mb,
            'latency_cpu_ms': self.latency_cpu_ms,
            'latency_gpu_ms': self.latency_gpu_ms,
            'throughput': self.throughput,
            'train_time_sec': self.train_time_sec,
            'convergence_epochs': self.convergence_epochs,
            'energy_joules': self.energy_joules,
            'carbon_kg': self.carbon_kg
        }


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    architecture_id: str
    metrics: ArchitectureMetrics
    objectives: np.ndarray = field(default_factory=lambda: np.array([]))
    raw_scores: Dict = field(default_factory=dict)
    
    def compute_objectives(self, 
                          objective_names: List[str],
                          normalize: bool = False) -> np.ndarray:
        """计算目标值向量"""
        metrics_dict = self.metrics.to_dict()
        objectives = []
        
        for name in objective_names:
            if name in metrics_dict:
                value = metrics_dict[name]
                # 对于需要最小化的指标，取负值
                if name in ['params_count', 'flops', 'memory_mb', 
                           'latency_cpu_ms', 'latency_gpu_ms',
                           'train_time_sec', 'energy_joules', 'carbon_kg']:
                    value = -value
                elif name == 'top1_accuracy':
                    value = - (100 - value)  # 转换为错误率
                objectives.append(value)
        
        return np.array(objectives)


class MetricsCalculator:
    """指标计算器"""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """统计模型参数量"""
        return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def estimate_flops(model: nn.Module, input_size: Tuple[int, ...]) -> float:
        """估计FLOPs"""
        # 简化的FLOPs估计
        total_flops = 0
        
        def hook_fn(module, input, output):
            nonlocal total_flops
            if isinstance(module, nn.Conv2d):
                # Conv2d FLOPs
                batch_size = output.size(0)
                out_h, out_w = output.size(2), output.size(3)
                kernel_ops = module.kernel_size[0] * module.kernel_size[1] * \
                           module.in_channels / module.groups
                output_size = batch_size * out_h * out_w * module.out_channels
                total_flops += kernel_ops * output_size
            
            elif isinstance(module, nn.Linear):
                # Linear FLOPs
                total_flops += module.in_features * module.out_features * \
                              output.size(0)
        
        hooks = []
        for module in model.modules():
            hooks.append(module.register_forward_hook(hook_fn))
        
        # 前向传播一次
        with torch.no_grad():
            dummy_input = torch.randn(input_size)
            model(dummy_input)
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        return total_flops / 1e9  # GFLOPs
    
    @staticmethod
    def measure_latency(model: nn.Module, 
                       input_size: Tuple[int, ...],
                       device: str = 'cpu',
                       num_runs: int = 100,
                       warmup: int = 10) -> float:
        """测量推理延迟"""
        model = model.to(device)
        model.eval()
        
        dummy_input = torch.randn(input_size).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
        
        # 测量
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        return (elapsed / num_runs) * 1000  # ms


class MOATBenchmark:
    """
    多目标架构评估基准
    
    提供标准化评估流程：
    1. 模型训练和评估
    2. 指标计算
    3. 结果汇总和比较
    """
    
    def __init__(self,
                 dataset_name: str = 'cifar10',
                 input_size: Tuple[int, ...] = (1, 3, 32, 32),
                 num_classes: int = 10):
        self.dataset_name = dataset_name
        self.input_size = input_size
        self.num_classes = num_classes
        self.results: List[BenchmarkResult] = []
    
    def evaluate_architecture(self,
                             model: nn.Module,
                             architecture_id: str,
                             train_loader = None,
                             val_loader = None,
                             quick_mode: bool = True) -> BenchmarkResult:
        """
        评估一个架构
        
        Args:
            model: 要评估的模型
            architecture_id: 架构标识符
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            quick_mode: 是否使用快速评估模式
        
        Returns:
            BenchmarkResult: 评估结果
        """
        print(f"\n评估架构: {architecture_id}")
        print("-" * 50)
        
        metrics = ArchitectureMetrics()
        
        # 1. 统计参数量
        metrics.params_count = MetricsCalculator.count_parameters(model)
        print(f"参数量: {metrics.params_count:,}")
        
        # 2. 估计FLOPs
        metrics.flops = MetricsCalculator.estimate_flops(model, self.input_size)
        print(f"FLOPs: {metrics.flops:.2f}G")
        
        # 3. 估计内存
        metrics.memory_mb = metrics.params_count * 4 / (1024 ** 2)  # float32
        print(f"内存占用: {metrics.memory_mb:.2f}MB")
        
        if not quick_mode and val_loader is not None:
            # 4. 评估准确率
            acc_metrics = self._evaluate_accuracy(model, val_loader)
            metrics.top1_accuracy = acc_metrics['top1']
            metrics.top5_accuracy = acc_metrics['top5']
            print(f"Top-1准确率: {metrics.top1_accuracy:.2f}%")
            
            # 5. 测量延迟
            try:
                metrics.latency_cpu_ms = MetricsCalculator.measure_latency(
                    model, self.input_size, device='cpu'
                )
                print(f"CPU延迟: {metrics.latency_cpu_ms:.2f}ms")
            except Exception as e:
                print(f"CPU延迟测量失败: {e}")
            
            if torch.cuda.is_available():
                try:
                    metrics.latency_gpu_ms = MetricsCalculator.measure_latency(
                        model, self.input_size, device='cuda'
                    )
                    print(f"GPU延迟: {metrics.latency_gpu_ms:.2f}ms")
                except Exception as e:
                    print(f"GPU延迟测量失败: {e}")
        else:
            # 快速模式：使用估计值
            metrics.top1_accuracy = 70 + np.random.rand() * 20  # 模拟
            print("(快速模式 - 使用估计值)")
        
        result = BenchmarkResult(
            architecture_id=architecture_id,
            metrics=metrics
        )
        
        self.results.append(result)
        return result
    
    def _evaluate_accuracy(self, model: nn.Module, 
                          val_loader,
                          device: str = 'cuda') -> Dict[str, float]:
        """评估模型准确率"""
        model = model.to(device)
        model.eval()
        
        correct_1 = 0
        correct_5 = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                _, predicted = outputs.topk(5, 1, True, True)
                predicted = predicted.t()
                correct = predicted.eq(targets.view(1, -1).expand_as(predicted))
                
                correct_1 += correct[:1].sum().item()
                correct_5 += correct[:5].sum().item()
                total += targets.size(0)
        
        return {
            'top1': 100. * correct_1 / total,
            'top5': 100. * correct_5 / total
        }
    
    def compute_pareto_front(self,
                            objective_names: List[str]) -> List[BenchmarkResult]:
        """
        计算Pareto前沿
        
        Args:
            objective_names: 目标名称列表
        
        Returns:
            Pareto前沿上的结果列表
        """
        # 计算每个结果的目标值
        for result in self.results:
            result.objectives = result.compute_objectives(objective_names)
        
        # 非支配排序
        pareto_front = []
        for i, result_i in enumerate(self.results):
            dominated = False
            for j, result_j in enumerate(self.results):
                if i != j:
                    # 检查j是否支配i
                    if np.all(result_j.objectives <= result_i.objectives) and \
                       np.any(result_j.objectives < result_i.objectives):
                        dominated = True
                        break
            
            if not dominated:
                pareto_front.append(result_i)
        
        return pareto_front
    
    def generate_report(self, 
                       pareto_objectives: List[str] = None) -> Dict:
        """生成评估报告"""
        report = {
            'benchmark_info': {
                'dataset': self.dataset_name,
                'num_architectures': len(self.results),
                'input_size': list(self.input_size)
            },
            'summary': {
                'total_params_min': min(r.metrics.params_count for r in self.results),
                'total_params_max': max(r.metrics.params_count for r in self.results),
                'accuracy_min': min(r.metrics.top1_accuracy for r in self.results),
                'accuracy_max': max(r.metrics.top1_accuracy for r in self.results),
            },
            'results': [r.metrics.to_dict() for r in self.results]
        }
        
        # 添加Pareto前沿
        if pareto_objectives:
            pareto = self.compute_pareto_front(pareto_objectives)
            report['pareto_front'] = {
                'objectives': pareto_objectives,
                'size': len(pareto),
                'architectures': [r.architecture_id for r in pareto]
            }
        
        return report
    
    def export_report(self, filepath: str, 
                     pareto_objectives: List[str] = None):
        """导出报告到文件"""
        report = self.generate_report(pareto_objectives)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n报告已导出到: {filepath}")
    
    def compare_algorithms(self,
                          algorithm_results: Dict[str, List[BenchmarkResult]],
                          objective_names: List[str]) -> Dict:
        """比较不同算法的结果"""
        comparison = {}
        
        for alg_name, results in algorithm_results.items():
            # 计算每个算法的Pareto前沿
            self.results = results
            pareto = self.compute_pareto_front(objective_names)
            
            # 计算指标
            metrics = {
                'pareto_size': len(pareto),
                'avg_accuracy': np.mean([r.metrics.top1_accuracy for r in results]),
                'avg_params': np.mean([r.metrics.params_count for r in results]),
            }
            
            comparison[alg_name] = metrics
        
        return comparison


class SimpleNASModel(nn.Module):
    """简单的NAS搜索模型（用于演示）"""
    
    def __init__(self, num_classes: int = 10, 
                 width: int = 32, depth: int = 3):
        super().__init__()
        
        layers = []
        in_ch = 3
        
        for i in range(depth):
            layers.extend([
                nn.Conv2d(in_ch, width * (2 ** (i//2)), 3, padding=1),
                nn.BatchNorm2d(width * (2 ** (i//2))),
                nn.ReLU(),
                nn.MaxPool2d(2) if i % 2 == 1 else nn.Identity()
            ])
            in_ch = width * (2 ** (i//2))
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_ch, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # 全局平均池化
        return self.classifier(x)


def demo_moat_benchmark():
    """
    MOAT基准测试演示
    
    小测试：
    1. 为什么需要标准化基准？
    2. Pareto前沿如何帮助比较算法？
    3. 哪些指标对边缘设备部署最重要？
    """
    print("=" * 70)
    print("MOAT多目标架构评估基准演示")
    print("=" * 70)
    
    # 创建基准
    benchmark = MOATBenchmark(
        dataset_name='cifar10',
        input_size=(1, 3, 32, 32),
        num_classes=10
    )
    
    # 评估多个架构配置
    print("\n评估不同架构配置...")
    
    configs = [
        ('small_fast', {'width': 16, 'depth': 2}),
        ('medium_balanced', {'width': 32, 'depth': 3}),
        ('large_accurate', {'width': 64, 'depth': 5}),
        ('wide_shallow', {'width': 128, 'depth': 2}),
        ('narrow_deep', {'width': 16, 'depth': 6}),
    ]
    
    for name, config in configs:
        model = SimpleNASModel(**config)
        result = benchmark.evaluate_architecture(
            model, name, quick_mode=True
        )
    
    # 生成报告
    print("\n" + "=" * 70)
    print("评估报告")
    print("=" * 70)
    
    report = benchmark.generate_report(
        pareto_objectives=['top1_accuracy', 'params_count']
    )
    
    print(f"\n数据集: {report['benchmark_info']['dataset']}")
    print(f"评估架构数: {report['benchmark_info']['num_architectures']}")
    
    print("\n指标范围:")
    summary = report['summary']
    print(f"  参数量: {summary['total_params_min']:,} - {summary['total_params_max']:,}")
    print(f"  准确率: {summary['accuracy_min']:.1f}% - {summary['accuracy_max']:.1f}%")
    
    # Pareto前沿
    if 'pareto_front' in report:
        print(f"\nPareto前沿:")
        print(f"  前沿大小: {report['pareto_front']['size']}")
        print(f"  目标: {report['pareto_front']['objectives']}")
        print(f"  架构: {', '.join(report['pareto_front']['architectures'])}")
    
    # 详细结果
    print("\n" + "-" * 70)
    print("详细评估结果:")
    print("-" * 70)
    print(f"{'架构':<20} {'准确率':<10} {'参数量':<12} {'FLOPs':<10}")
    print("-" * 70)
    
    for result in benchmark.results:
        m = result.metrics
        print(f"{result.architecture_id:<20} "
              f"{m.top1_accuracy:<10.2f} "
              f"{m.params_count:<12,} "
              f"{m.flops:<10.2f}")
    
    # 算法比较演示
    print("\n" + "-" * 70)
    print("算法比较示例:")
    print("-" * 70)
    
    # 模拟两个算法的结果
    alg_a_results = benchmark.results[:3]
    alg_b_results = benchmark.results[2:]  # 有重叠
    
    comparison = benchmark.compare_algorithms(
        {'Algorithm_A': alg_a_results, 'Algorithm_B': alg_b_results},
        ['top1_accuracy', 'params_count']
    )
    
    for alg, metrics in comparison.items():
        print(f"\n{alg}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    
    return benchmark


if __name__ == "__main__":
    demo_moat_benchmark()
