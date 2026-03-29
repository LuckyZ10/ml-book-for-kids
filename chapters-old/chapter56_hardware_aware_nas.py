"""
硬件感知神经架构搜索 (Hardware-Aware NAS)
直接在搜索过程中考虑实际硬件延迟、能耗等指标
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time


@dataclass
class HardwareSpec:
    """硬件规格"""
    name: str
    compute_capability: float  # 计算能力 (GFLOPS)
    memory_bandwidth: float  # 内存带宽 (GB/s)
    power_budget: float  # 功耗预算 (W)
    
    def __repr__(self):
        return f"{self.name}(compute={self.compute_capability}GFLOPS, " \
               f"bandwidth={self.memory_bandwidth}GB/s)"


# 预定义一些常见硬件平台
HARDWARE_PLATFORMS = {
    'edge_cpu': HardwareSpec('Edge CPU', 10, 25, 5),
    'mobile_gpu': HardwareSpec('Mobile GPU', 200, 50, 10),
    'desktop_gpu': HardwareSpec('Desktop GPU', 10000, 900, 250),
    'server_gpu': HardwareSpec('Server GPU', 30000, 2000, 400),
}


class LatencyPredictor(ABC):
    """
    延迟预测器基类
    
    硬件感知NAS的核心：准确预测架构在目标硬件上的延迟
    """
    @abstractmethod
    def predict(self, architecture: Dict, hardware: HardwareSpec) -> float:
        """预测延迟（毫秒）"""
        pass


class AnalyticalLatencyPredictor(LatencyPredictor):
    """
    基于分析模型的延迟预测器
    
    原理：根据操作的FLOPs和硬件规格估算延迟
    
    费曼法比喻：
    想象你要组装一个乐高城堡。分析模型就像计算：
    - 每个积木块需要多少秒来组装
    - 有多少个积木块
    - 组装速度是否受限于你的手速（计算能力）
                       或积木供应速度（内存带宽）
    """
    def __init__(self):
        # 每种操作类型的开销（相对于FLOPs）
        self.op_cost_factors = {
            'conv3x3': 1.0,
            'conv5x5': 2.8,  # 参数量约是3x3的2.78倍
            'dconv3x3': 0.8,  # 深度可分离卷积更高效
            'maxpool': 0.1,
            'avgpool': 0.1,
            'skip': 0.0,
        }
        
    def predict(self, architecture: Dict, hardware: HardwareSpec) -> float:
        """
        预测延迟
        
        公式：
        latency = max(
            total_flops / compute_capability,
            total_memory / memory_bandwidth
        ) * overhead_factor
        """
        total_flops = architecture.get('total_flops', 0)  # MFLOPs
        total_params = architecture.get('total_params', 0)  # MB
        
        # 计算受限时间
        compute_time = total_flops / hardware.compute_capability  # ms
        
        # 内存受限时间
        memory_time = total_params * 4 / hardware.memory_bandwidth  # ms (4 bytes per float32)
        
        # 实际延迟由瓶颈决定，加上20%开销
        latency = max(compute_time, memory_time) * 1.2
        
        return latency


class LookupTablePredictor(LatencyPredictor):
    """
    基于查找表的延迟预测器
    
    原理：预先在目标硬件上测量常见操作的延迟，建立查找表
    
    这种方法更准确，但需要针对每种硬件进行测量
    """
    def __init__(self, hardware_name: str):
        self.hardware_name = hardware_name
        
        # 模拟查找表（实际中应通过测量获得）
        self.latency_table = self._build_lookup_table()
        
    def _build_lookup_table(self) -> Dict[str, Dict]:
        """构建延迟查找表"""
        # 键：操作类型_输入尺寸_输出通道
        table = {
            'conv3x3_32_32': {'latency': 0.5, 'energy': 0.1},
            'conv3x3_32_64': {'latency': 0.8, 'energy': 0.15},
            'conv3x3_64_64': {'latency': 1.0, 'energy': 0.2},
            'conv3x3_64_128': {'latency': 1.8, 'energy': 0.35},
            'conv5x5_32_32': {'latency': 1.2, 'energy': 0.25},
            'conv5x5_32_64': {'latency': 2.0, 'energy': 0.4},
            'dconv3x3_32_32': {'latency': 0.4, 'energy': 0.08},
            'dconv3x3_64_64': {'latency': 0.7, 'energy': 0.15},
            'maxpool_32_32': {'latency': 0.1, 'energy': 0.02},
            'avgpool_32_32': {'latency': 0.1, 'energy': 0.02},
            'skip': {'latency': 0.0, 'energy': 0.0},
        }
        
        # 根据硬件类型调整
        if self.hardware_name == 'edge_cpu':
            for k in table:
                table[k]['latency'] *= 10
                table[k]['energy'] *= 5
        elif self.hardware_name == 'mobile_gpu':
            for k in table:
                table[k]['latency'] *= 2
                table[k]['energy'] *= 2
        elif self.hardware_name == 'server_gpu':
            for k in table:
                table[k]['latency'] *= 0.3
                table[k]['energy'] *= 3  # 服务器GPU功耗更高
                
        return table
    
    def predict(self, architecture: Dict, hardware: HardwareSpec) -> float:
        """预测延迟"""
        total_latency = 0
        
        # 遍历架构中的所有操作
        for op in architecture.get('operations', []):
            op_type = op['type']
            in_channels = op.get('in_channels', 32)
            out_channels = op.get('out_channels', 32)
            
            # 构建查找键
            key = f"{op_type}_{in_channels}_{out_channels}"
            
            # 查找延迟
            if key in self.latency_table:
                total_latency += self.latency_table[key]['latency']
            else:
                # 使用最接近的匹配或默认值
                total_latency += 1.0
        
        return total_latency
    
    def predict_energy(self, architecture: Dict) -> float:
        """预测能耗"""
        total_energy = 0
        
        for op in architecture.get('operations', []):
            op_type = op['type']
            key = f"{op_type}_{op.get('in_channels', 32)}_{op.get('out_channels', 32)}"
            
            if key in self.latency_table:
                total_energy += self.latency_table[key]['energy']
            else:
                total_energy += 0.2
        
        return total_energy


class NeuralLatencyPredictor(LatencyPredictor):
    """
    基于神经网络的延迟预测器
    
    原理：训练一个小型神经网络来预测延迟
    优势：可以捕捉复杂的非线性关系
    """
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def encode_architecture(self, architecture: Dict) -> torch.Tensor:
        """将架构编码为特征向量"""
        features = []
        
        # 架构统计特征
        features.append(architecture.get('num_layers', 0))
        features.append(architecture.get('total_flops', 0))
        features.append(architecture.get('total_params', 0))
        features.append(architecture.get('num_conv3x3', 0))
        features.append(architecture.get('num_conv5x5', 0))
        features.append(architecture.get('num_dconv', 0))
        features.append(architecture.get('num_pool', 0))
        features.append(architecture.get('max_channels', 0))
        features.append(architecture.get('avg_channels', 0))
        features.append(architecture.get('network_depth', 0))
        
        return torch.tensor(features, dtype=torch.float32)
    
    def predict(self, architecture: Dict, hardware: HardwareSpec) -> float:
        """预测延迟"""
        features = self.encode_architecture(architecture)
        
        # 添加硬件特征
        hardware_features = torch.tensor([
            hardware.compute_capability,
            hardware.memory_bandwidth
        ], dtype=torch.float32)
        
        combined = torch.cat([features, hardware_features])
        
        with torch.no_grad():
            latency = self.model(combined.unsqueeze(0)).item()
        
        return max(latency, 0.1)  # 确保非负


class HardwareAwareNAS:
    """
    硬件感知神经架构搜索
    
    同时优化准确率和硬件效率
    """
    def __init__(
        self,
        target_hardware: HardwareSpec,
        latency_predictor: LatencyPredictor,
        latency_constraint: float,
        accuracy_weight: float = 1.0,
        latency_weight: float = 0.5
    ):
        self.target_hardware = target_hardware
        self.latency_predictor = latency_predictor
        self.latency_constraint = latency_constraint
        self.accuracy_weight = accuracy_weight
        self.latency_weight = latency_weight
        
    def evaluate_architecture(
        self,
        architecture: Dict,
        evaluate_accuracy_fn: Callable[[Dict], float]
    ) -> Dict[str, float]:
        """
        评估架构的多目标性能
        
        Args:
            architecture: 架构描述
            evaluate_accuracy_fn: 评估准确率的函数
            
        Returns:
            包含准确率、延迟、综合得分的字典
        """
        # 评估准确率
        accuracy = evaluate_accuracy_fn(architecture)
        
        # 预测延迟
        latency = self.latency_predictor.predict(
            architecture, 
            self.target_hardware
        )
        
        # 计算违反约束的惩罚
        latency_penalty = max(0, latency - self.latency_constraint)
        
        # 综合得分（越高越好）
        # 准确率越高越好，延迟越低越好
        score = (self.accuracy_weight * accuracy 
                 - self.latency_weight * latency / 100  # 归一化
                 - 10 * latency_penalty)  # 强惩罚
        
        return {
            'accuracy': accuracy,
            'latency': latency,
            'score': score,
            'feasible': latency <= self.latency_constraint
        }
    
    def search(
        self,
        search_space,
        evaluate_accuracy_fn: Callable[[Dict], float],
        num_iterations: int = 100
    ) -> List[Dict]:
        """
        执行硬件感知搜索
        
        Args:
            search_space: 搜索空间
            evaluate_accuracy_fn: 准确率评估函数
            num_iterations: 搜索迭代次数
            
        Returns:
            找到的最佳架构列表
        """
        results = []
        
        for i in range(num_iterations):
            # 采样架构（实际中应使用更智能的策略）
            arch = search_space.sample()
            
            # 评估
            metrics = self.evaluate_architecture(arch, evaluate_accuracy_fn)
            
            results.append({
                'architecture': arch,
                **metrics
            })
            
            if i % 20 == 0:
                feasible_count = sum(1 for r in results if r['feasible'])
                print(f"Iter {i}: Feasible={feasible_count}/{len(results)}, "
                      f"Best score={max(r['score'] for r in results):.3f}")
        
        # 返回满足约束的最佳架构
        feasible_results = [r for r in results if r['feasible']]
        if not feasible_results:
            feasible_results = results
        
        feasible_results.sort(key=lambda x: x['score'], reverse=True)
        return feasible_results[:10]


class MultiHardwareNAS:
    """
    多硬件平台NAS
    
    同时搜索适用于多种硬件平台的架构
    
    费曼法比喻：
    想象你是一个服装设计师，要设计一件衣服同时适合：
    - 办公室（正式、舒适）
    - 健身房（透气、弹性）
    - 晚宴（优雅、华丽）
    
    多硬件NAS就是找到这样一个"全能"的架构。
    """
    def __init__(
        self,
        hardware_platforms: List[HardwareSpec],
        latency_predictors: Dict[str, LatencyPredictor]
    ):
        self.hardware_platforms = hardware_platforms
        self.latency_predictors = latency_predictors
        
    def evaluate_cross_platform(
        self,
        architecture: Dict,
        evaluate_accuracy_fn: Callable[[Dict], float]
    ) -> Dict:
        """
        评估架构在多个硬件平台上的表现
        """
        results = {
            'accuracy': evaluate_accuracy_fn(architecture),
            'latencies': {},
            'harmonic_mean_latency': 0
        }
        
        latencies = []
        for hw in self.hardware_platforms:
            predictor = self.latency_predictors.get(hw.name, 
                        AnalyticalLatencyPredictor())
            latency = predictor.predict(architecture, hw)
            results['latencies'][hw.name] = latency
            latencies.append(latency)
        
        # 计算调和平均延迟（更强调最差平台的表现）
        if all(l > 0 for l in latencies):
            results['harmonic_mean_latency'] = len(latencies) / sum(1/l for l in latencies)
        
        return results


# ====== 使用示例 ======
def demo_hardware_aware_nas():
    """硬件感知NAS演示"""
    print("=" * 70)
    print("硬件感知神经架构搜索 (Hardware-Aware NAS)")
    print("=" * 70)
    
    # 定义目标硬件
    target_hw = HARDWARE_PLATFORMS['mobile_gpu']
    print(f"\n目标硬件: {target_hw}")
    print(f"延迟约束: 50ms")
    
    # 创建延迟预测器
    latency_predictor = AnalyticalLatencyPredictor()
    
    # 创建硬件感知NAS
    hw_nas = HardwareAwareNAS(
        target_hardware=target_hw,
        latency_predictor=latency_predictor,
        latency_constraint=50.0,  # 50ms
        accuracy_weight=1.0,
        latency_weight=0.5
    )
    
    # 模拟搜索空间
    class SimpleSearchSpace:
        def sample(self):
            """随机采样架构"""
            depth = np.random.randint(5, 20)
            width = np.random.randint(32, 128)
            
            return {
                'depth': depth,
                'width': width,
                'total_flops': depth * width * width * 0.1,  # 模拟FLOPs
                'total_params': depth * width * 0.01,  # 模拟参数量
                'num_conv3x3': depth,
                'num_conv5x5': depth // 4,
                'num_dconv': depth // 2,
                'max_channels': width,
            }
    
    search_space = SimpleSearchSpace()
    
    # 模拟准确率评估
    def evaluate_accuracy(arch):
        """
        模拟准确率评估
        
        规律：更深的网络通常更准确，但有边际递减
        """
        depth = arch['depth']
        width = arch['width']
        
        # 基础准确率
        base_acc = 0.5
        
        # 深度带来的收益（边际递减）
        depth_gain = 0.3 * (1 - np.exp(-depth / 10))
        
        # 宽度带来的收益（边际递减）
        width_gain = 0.15 * (1 - np.exp(-width / 100))
        
        # 添加噪声
        noise = np.random.normal(0, 0.02)
        
        accuracy = base_acc + depth_gain + width_gain + noise
        return min(accuracy, 0.99)
    
    print("\n开始硬件感知搜索...")
    results = hw_nas.search(
        search_space=search_space,
        evaluate_accuracy_fn=evaluate_accuracy,
        num_iterations=50
    )
    
    # 显示结果
    print("\n" + "=" * 70)
    print("搜索结果 (Top 5)")
    print("=" * 70)
    print(f"{'排名':<6} {'准确率':<10} {'延迟(ms)':<12} {'得分':<10} {'满足约束':<10}")
    print("-" * 70)
    
    for i, result in enumerate(results[:5], 1):
        print(f"{i:<6} {result['accuracy']:.4f}     {result['latency']:.2f}        "
              f"{result['score']:.3f}      {'✓' if result['feasible'] else '✗'}")
    
    # 演示多硬件NAS
    print("\n" + "=" * 70)
    print("多硬件平台NAS演示")
    print("=" * 70)
    
    multi_hw_nas = MultiHardwareNAS(
        hardware_platforms=[
            HARDWARE_PLATFORMS['edge_cpu'],
            HARDWARE_PLATFORMS['mobile_gpu'],
            HARDWARE_PLATFORMS['server_gpu']
        ],
        latency_predictors={
            'Edge CPU': AnalyticalLatencyPredictor(),
            'Mobile GPU': AnalyticalLatencyPredictor(),
            'Server GPU': AnalyticalLatencyPredictor()
        }
    )
    
    # 评估一个架构在多个平台上的延迟
    test_arch = {
        'depth': 10,
        'width': 64,
        'total_flops': 500,  # MFLOPs
        'total_params': 5    # MB
    }
    
    print(f"\n测试架构: 深度={test_arch['depth']}, 宽度={test_arch['width']}")
    cross_platform_results = multi_hw_nas.evaluate_cross_platform(
        test_arch, evaluate_accuracy
    )
    
    print(f"准确率: {cross_platform_results['accuracy']:.4f}")
    print("各平台延迟:")
    for hw_name, latency in cross_platform_results['latencies'].items():
        print(f"  {hw_name}: {latency:.2f}ms")
    print(f"调和平均延迟: {cross_platform_results['harmonic_mean_latency']:.2f}ms")
    
    print("\n" + "=" * 70)
    print("硬件感知NAS演示完成")
    print("=" * 70)


if __name__ == "__main__":
    demo_hardware_aware_nas()
