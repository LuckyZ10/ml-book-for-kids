"""
ParetoVisualizer: Pareto前沿可视化工具
======================================

可视化多目标优化中的Pareto前沿，
帮助理解不同解决方案之间的权衡关系。

费曼法比喻：
想象你在选择餐厅：
- X轴：价格
- Y轴：口味评分
- Pareto前沿就是那些"没有明显缺陷"的餐厅

这些餐厅要么便宜又好吃（理想但稀少），
要么贵但好吃，要么便宜但一般。
但前沿上的餐厅都有各自的优势。
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import json


@dataclass
class ParetoPoint:
    """Pareto前沿上的一个点"""
    objectives: np.ndarray  # 目标值向量
    solution: any           # 对应的解决方案
    metadata: Dict          # 额外信息
    
    def dominates(self, other: 'ParetoPoint') -> bool:
        """检查是否支配另一个点"""
        not_worse = np.all(self.objectives <= other.objectives)
        strictly_better = np.any(self.objectives < other.objectives)
        return not_worse and strictly_better


class ParetoFront:
    """
    Pareto前沿管理器
    
    维护一组非支配解，支持动态更新。
    """
    
    def __init__(self, minimize: bool = True):
        self.minimize = minimize
        self.points: List[ParetoPoint] = []
        self.history: List[List[ParetoPoint]] = []
    
    def add(self, point: ParetoPoint) -> bool:
        """
        添加一个新点，并更新Pareto前沿
        
        Returns:
            是否添加成功（是否非支配）
        """
        # 检查是否被现有前沿支配
        for existing in self.points:
            if existing.dominates(point):
                return False
        
        # 移除被新点支配的点
        self.points = [p for p in self.points if not point.dominates(p)]
        
        # 添加新点
        self.points.append(point)
        
        # 保存历史
        self.history.append([p for p in self.points])
        
        return True
    
    def get_points(self) -> List[ParetoPoint]:
        """获取当前Pareto前沿"""
        return self.points
    
    def get_objectives_matrix(self) -> np.ndarray:
        """获取目标值的矩阵形式"""
        if not self.points:
            return np.array([])
        return np.array([p.objectives for p in self.points])
    
    def calculate_hypervolume(self, reference_point: np.ndarray = None) -> float:
        """
        计算超体积（Hypervolume）指标
        
        衡量Pareto前沿覆盖的空间大小。
        超体积越大，说明解集质量越高。
        
        对于2D情况，使用矩形法计算。
        对于3D+情况，使用蒙特卡洛近似。
        """
        if len(self.points) == 0:
            return 0.0
        
        objectives = self.get_objectives_matrix()
        
        if reference_point is None:
            reference_point = np.max(objectives, axis=0) * 1.1
        
        if objectives.shape[1] == 2:
            # 2D: 使用精确计算
            return self._hypervolume_2d(objectives, reference_point)
        else:
            # 3D+: 使用蒙特卡洛近似
            return self._hypervolume_mc(objectives, reference_point)
    
    def _hypervolume_2d(self, objectives: np.ndarray, 
                       ref: np.ndarray) -> float:
        """计算2D超体积"""
        # 按第一个目标排序
        sorted_indices = np.argsort(objectives[:, 0])
        sorted_obj = objectives[sorted_indices]
        
        volume = 0.0
        prev_x = ref[0]
        
        for obj in sorted_obj:
            dx = prev_x - obj[0]
            dy = ref[1] - obj[1]
            volume += dx * dy
            prev_x = obj[0]
        
        return volume
    
    def _hypervolume_mc(self, objectives: np.ndarray,
                       ref: np.ndarray, n_samples: int = 10000) -> float:
        """蒙特卡洛近似超体积"""
        # 在参考点范围内随机采样
        samples = np.random.rand(n_samples, objectives.shape[1])
        for i in range(objectives.shape[1]):
            samples[:, i] *= ref[i]
        
        # 计算被支配的样本数
        dominated = 0
        for sample in samples:
            for obj in objectives:
                if np.all(sample >= obj):
                    dominated += 1
                    break
        
        # 超体积 = 总空间 × 被支配比例
        total_volume = np.prod(ref)
        return total_volume * dominated / n_samples
    
    def calculate_spread(self) -> float:
        """
        计算延展度（Spacing）指标
        
        衡量Pareto前沿的多样性。
        值越小，说明分布越均匀。
        """
        if len(self.points) < 2:
            return 0.0
        
        objectives = self.get_objectives_matrix()
        n = len(objectives)
        
        # 计算每个点到最近邻的距离
        distances = []
        for i in range(n):
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(objectives[i] - objectives[j])
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)
        
        d_mean = np.mean(distances)
        spacing = np.sqrt(np.sum((np.array(distances) - d_mean) ** 2) / n)
        
        return spacing
    
    def get_knee_point(self) -> Optional[ParetoPoint]:
        """
        找到膝点（Knee Point）
        
        膝点是Pareto前沿上"弯曲度"最大的点，
        通常代表最佳的权衡方案。
        """
        if len(self.points) < 3:
            return self.points[0] if self.points else None
        
        objectives = self.get_objectives_matrix()
        
        # 标准化
        obj_min = objectives.min(axis=0)
        obj_max = objectives.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1
        normalized = (objectives - obj_min) / obj_range
        
        # 计算弯曲度（使用相邻向量的夹角）
        max_bend = -1
        knee_idx = 0
        
        for i in range(1, len(normalized) - 1):
            v1 = normalized[i-1] - normalized[i]
            v2 = normalized[i+1] - normalized[i]
            
            # 计算夹角（使用叉积）
            bend = np.linalg.norm(v1) * np.linalg.norm(v2)
            if bend > 0:
                bend = np.dot(v1, v2) / bend
                bend = np.arccos(np.clip(bend, -1, 1))
                
                if bend > max_bend:
                    max_bend = bend
                    knee_idx = i
        
        return self.points[knee_idx]
    
    def get_ideal_point(self) -> np.ndarray:
        """获取理想点（每个目标的最优值）"""
        if not self.points:
            return np.array([])
        return np.min(self.get_objectives_matrix(), axis=0)
    
    def get_nadir_point(self) -> np.ndarray:
        """获取最差点（每个目标的最差值）"""
        if not self.points:
            return np.array([])
        return np.max(self.get_objectives_matrix(), axis=0)
    
    def export_json(self, filepath: str):
        """导出Pareto前沿到JSON"""
        data = {
            'points': [
                {
                    'objectives': p.objectives.tolist(),
                    'metadata': p.metadata
                }
                for p in self.points
            ],
            'hypervolume': self.calculate_hypervolume(),
            'spread': self.calculate_spread(),
            'ideal': self.get_ideal_point().tolist(),
            'nadir': self.get_nadir_point().tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Pareto前沿已导出到 {filepath}")


class ParetoVisualizer:
    """
    Pareto前沿可视化器
    
    生成文本形式的Pareto前沿可视化
    """
    
    def __init__(self, width: int = 60, height: int = 20):
        self.width = width
        self.height = height
    
    def plot_2d(self, pareto_front: ParetoFront, 
               xlabel: str = "目标1",
               ylabel: str = "目标2") -> str:
        """
        生成2D Pareto前沿的ASCII图
        """
        if len(pareto_front.points) == 0:
            return "空Pareto前沿"
        
        objectives = pareto_front.get_objectives_matrix()
        
        # 归一化到绘图区域
        x_min, x_max = objectives[:, 0].min(), objectives[:, 0].max()
        y_min, y_max = objectives[:, 1].min(), objectives[:, 1].max()
        
        # 添加边距
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        
        x_min -= x_range * 0.1
        x_max += x_range * 0.1
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        
        # 创建画布
        canvas = [[' ' for _ in range(self.width)] 
                 for _ in range(self.height)]
        
        # 绘制坐标轴
        for i in range(self.height):
            canvas[i][5] = '│'
        for j in range(5, self.width):
            canvas[self.height - 3][j] = '─'
        canvas[self.height - 3][5] = '└'
        
        # 绘制Pareto前沿
        points_set = set()
        for obj in objectives:
            x = int((obj[0] - x_min) / (x_max - x_min) * (self.width - 10) + 5)
            y = int((y_max - obj[1]) / (y_max - y_min) * (self.height - 5))
            x = max(6, min(x, self.width - 1))
            y = max(0, min(y, self.height - 4))
            points_set.add((x, y))
        
        # 绘制点
        for x, y in points_set:
            canvas[y][x] = '●'
        
        # 添加标签
        label_x = min(10, self.width - len(xlabel) - 1)
        for i, c in enumerate(xlabel):
            if label_x + i < self.width:
                canvas[self.height - 2][label_x + i] = c
        
        label_y = ylabel
        for i, c in enumerate(label_y):
            if i < 4:
                canvas[i][0] = c
        
        # 转换字符串
        lines = [''.join(row) for row in canvas]
        return '\n'.join(lines)
    
    def plot_history(self, pareto_front: ParetoFront) -> str:
        """绘制超体积随时间的变化"""
        if not pareto_front.history:
            return "无历史记录"
        
        hvs = []
        for front in pareto_front.history:
            temp_pf = ParetoFront()
            temp_pf.points = front
            hvs.append(temp_pf.calculate_hypervolume())
        
        # 绘制简单的线图
        lines = ["超体积变化历史:"]
        lines.append("-" * 40)
        
        if len(hvs) > 0:
            max_hv = max(hvs)
            min_hv = min(hvs)
            hv_range = max_hv - min_hv if max_hv != min_hv else 1
            
            for i, hv in enumerate(hvs[::max(1, len(hvs)//20)]):
                bar_len = int((hv - min_hv) / hv_range * 30)
                bar = '█' * bar_len
                lines.append(f"Gen {i*max(1, len(hvs)//20):3d}: {bar} {hv:.4f}")
        
        lines.append("-" * 40)
        lines.append(f"初始: {hvs[0]:.4f}, 最终: {hvs[-1]:.4f}, "
                    f"提升: {(hvs[-1]/hvs[0]-1)*100:.1f}%")
        
        return '\n'.join(lines)
    
    def plot_comparison(self, 
                       fronts: Dict[str, ParetoFront]) -> str:
        """比较多个Pareto前沿"""
        lines = ["多Pareto前沿比较:"]
        lines.append("=" * 60)
        
        for name, pf in fronts.items():
            lines.append(f"\n{name}:")
            lines.append(f"  解数量: {len(pf.points)}")
            if pf.points:
                lines.append(f"  超体积: {pf.calculate_hypervolume():.4f}")
                lines.append(f"  延展度: {pf.calculate_spread():.4f}")
                
                # 理想点和最差点
                ideal = pf.get_ideal_point()
                nadir = pf.get_nadir_point()
                lines.append(f"  理想点: [{', '.join([f'{v:.3f}' for v in ideal])}]")
                lines.append(f"  最差点: [{', '.join([f'{v:.3f}' for v in nadir])}]")
        
        lines.append("\n" + "=" * 60)
        return '\n'.join(lines)


def demo_pareto_visualization():
    """
    Pareto前沿可视化演示
    
    小测试：
    1. 为什么Pareto前沿通常是"向下凹"的？
    2. 膝点有什么特殊意义？
    3. 超体积指标如何帮助评估优化算法？
    """
    print("=" * 70)
    print("Pareto前沿可视化演示")
    print("=" * 70)
    
    # 创建示例Pareto前沿
    pf = ParetoFront()
    
    # 添加一些模拟点（错误率 vs 延迟）
    np.random.seed(42)
    
    print("\n添加Pareto点...")
    for i in range(20):
        # 模拟NAS搜索结果：准确率和延迟的权衡
        error = 0.05 + 0.3 * np.random.rand()
        latency = 5 + 50 * np.random.rand() * (1 + error)
        
        point = ParetoPoint(
            objectives=np.array([error, latency]),
            solution=f"model_{i}",
            metadata={'params': 1e6 * (1 + error * 10)}
        )
        
        added = pf.add(point)
        if added:
            print(f"  ✓ 添加点: error={error:.3f}, latency={latency:.1f}ms")
    
    print(f"\n最终Pareto前沿大小: {len(pf.points)}")
    
    # 计算指标
    print("\n" + "-" * 70)
    print("Pareto前沿指标")
    print("-" * 70)
    
    hv = pf.calculate_hypervolume()
    spread = pf.calculate_spread()
    
    print(f"超体积 (Hypervolume): {hv:.4f}")
    print(f"延展度 (Spread): {spread:.4f}")
    
    ideal = pf.get_ideal_point()
    nadir = pf.get_nadir_point()
    print(f"理想点: [{ideal[0]:.3f}, {ideal[1]:.1f}]")
    print(f"最差点: [{nadir[0]:.3f}, {nadir[1]:.1f}]")
    
    # 膝点
    knee = pf.get_knee_point()
    if knee:
        print(f"\n膝点: error={knee.objectives[0]:.3f}, "
              f"latency={knee.objectives[1]:.1f}ms")
        print("  (这是最佳权衡方案)")
    
    # 可视化
    print("\n" + "-" * 70)
    print("Pareto前沿可视化")
    print("-" * 70)
    
    visualizer = ParetoVisualizer(width=60, height=15)
    plot = visualizer.plot_2d(pf, xlabel="Error Rate", ylabel="Latency (ms)")
    print(plot)
    
    # 历史变化
    print("\n" + "-" * 70)
    print(visualizer.plot_history(pf))
    
    # 多个前沿比较
    print("\n" + "-" * 70)
    pf2 = ParetoFront()
    for i in range(15):
        error = 0.03 + 0.25 * np.random.rand()
        latency = 3 + 40 * np.random.rand() * (1 + error)
        pf2.add(ParetoPoint(
            objectives=np.array([error, latency]),
            solution=f"model2_{i}",
            metadata={}
        ))
    
    comparison = visualizer.plot_comparison({
        "算法A": pf,
        "算法B": pf2
    })
    print(comparison)
    
    # 导出
    print("\n" + "-" * 70)
    print("导出Pareto前沿到JSON...")
    # 模拟导出
    print("导出成功！")
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    
    return pf, visualizer


if __name__ == "__main__":
    demo_pareto_visualization()
