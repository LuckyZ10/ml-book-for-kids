#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第2章：机器如何"看见"数据 - 配套代码
《机器学习与深度学习：从小学生到大师》

本代码演示：
1. 数据类型转换（不使用numpy，纯Python手搓）
2. 特征提取示例
3. 向量运算（纯Python实现）
4. 数据可视化（ASCII图表）
5. 距离度量（欧氏距离、曼哈顿距离）
6. 数据归一化

作者：AI Assistant
日期：2026-03-24
"""

import math
from typing import List, Dict, Tuple, Union


# ============================================
# 第1部分：向量基础运算（纯Python实现）
# ============================================

class Vector:
    """向量类：纯Python实现的n维向量"""
    
    def __init__(self, data: List[float]):
        """
        初始化向量
        
        参数:
            data: 一个列表，包含向量的各个分量
        
        示例:
            >>> v = Vector([1, 2, 3])
            >>> print(v.data)
            [1, 2, 3]
        """
        self.data = list(data)  # 复制数据，防止外部修改
        self.dimension = len(data)
    
    def __repr__(self):
        return f"Vector({self.data})"
    
    def __len__(self):
        return self.dimension
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __add__(self, other):
        """向量加法"""
        if len(self) != len(other):
            raise ValueError("向量维度不匹配！")
        return Vector([a + b for a, b in zip(self.data, other.data)])
    
    def __sub__(self, other):
        """向量减法"""
        if len(self) != len(other):
            raise ValueError("向量维度不匹配！")
        return Vector([a - b for a, b in zip(self.data, other.data)])
    
    def __mul__(self, scalar):
        """数乘（标量乘法）"""
        return Vector([a * scalar for a in self.data])
    
    def __rmul__(self, scalar):
        """支持 scalar * vector 的写法"""
        return self.__mul__(scalar)
    
    def dot(self, other):
        """
        向量点积（内积）
        
        公式: a·b = a₁b₁ + a₂b₂ + ... + aₙbₙ
        
        示例:
            >>> v1 = Vector([1, 2, 3])
            >>> v2 = Vector([4, 5, 6])
            >>> v1.dot(v2)
            32  # 1*4 + 2*5 + 3*6 = 32
        """
        if len(self) != len(other):
            raise ValueError("向量维度不匹配！")
        return sum(a * b for a, b in zip(self.data, other.data))
    
    def magnitude(self):
        """
        向量的模（长度）
        
        公式: ||v|| = √(v₁² + v₂² + ... + vₙ²)
        """
        return math.sqrt(sum(x ** 2 for x in self.data))
    
    def normalize(self):
        """
        向量归一化（单位化）
        
        返回与原向量方向相同，长度为1的向量
        """
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("零向量无法归一化！")
        return Vector([x / mag for x in self.data])


def euclidean_distance(v1: Vector, v2: Vector) -> float:
    """
    计算两个向量之间的欧氏距离
    
    欧氏距离是两点之间的直线距离（"乌鸦飞"的距离）
    
    公式: d = √((x₁-y₁)² + (x₂-y₂)² + ... + (xₙ-yₙ)²)
    
    参数:
        v1: 第一个向量
        v2: 第二个向量
    
    返回:
        两个向量之间的欧氏距离
    
    示例:
        >>> a = Vector([1, 2])
        >>> b = Vector([4, 6])
        >>> euclidean_distance(a, b)
        5.0  # √((4-1)² + (6-2)²) = √(9+16) = √25 = 5
    """
    if len(v1) != len(v2):
        raise ValueError("向量维度不匹配！")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1.data, v2.data)))


def manhattan_distance(v1: Vector, v2: Vector) -> float:
    """
    计算两个向量之间的曼哈顿距离
    
    曼哈顿距离是沿着坐标轴方向行走的距离（"出租车"距离）
    
    公式: d = |x₁-y₁| + |x₂-y₂| + ... + |xₙ-yₙ|
    
    参数:
        v1: 第一个向量
        v2: 第二个向量
    
    返回:
        两个向量之间的曼哈顿距离
    
    示例:
        >>> a = Vector([1, 2])
        >>> b = Vector([4, 6])
        >>> manhattan_distance(a, b)
        7  # |4-1| + |6-2| = 3 + 4 = 7
    """
    if len(v1) != len(v2):
        raise ValueError("向量维度不匹配！")
    return sum(abs(a - b) for a, b in zip(v1.data, v2.data))


def cosine_similarity(v1: Vector, v2: Vector) -> float:
    """
    计算两个向量之间的余弦相似度
    
    余弦相似度衡量两个向量的方向相似程度，范围在[-1, 1]之间
    1表示完全相同方向，-1表示完全相反方向，0表示正交
    
    公式: cos(θ) = (a·b) / (||a|| × ||b||)
    
    参数:
        v1: 第一个向量
        v2: 第二个向量
    
    返回:
        余弦相似度值
    """
    dot_product = v1.dot(v2)
    mag_product = v1.magnitude() * v2.magnitude()
    if mag_product == 0:
        return 0.0
    return dot_product / mag_product


# ============================================
# 第2部分：数据预处理
# ============================================

def calculate_mean(data: List[float]) -> float:
    """计算平均值"""
    if not data:
        return 0.0
    return sum(data) / len(data)


def calculate_std(data: List[float]) -> float:
    """
    计算标准差
    
    标准差衡量数据的离散程度
    
    公式: σ = √(Σ(xᵢ - μ)² / n)
    """
    if len(data) < 2:
        return 0.0
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return math.sqrt(variance)


def min_max_normalize(data: List[float]) -> List[float]:
    """
    Min-Max归一化
    
    将数据缩放到[0, 1]范围
    
    公式: x' = (x - min) / (max - min)
    
    参数:
        data: 原始数据列表
    
    返回:
        归一化后的数据列表
    """
    if not data:
        return []
    data_min = min(data)
    data_max = max(data)
    if data_max == data_min:
        return [0.5] * len(data)  # 避免除以零
    return [(x - data_min) / (data_max - data_min) for x in data]


def z_score_standardize(data: List[float]) -> List[float]:
    """
    Z-score标准化
    
    将数据转换为均值为0、标准差为1的分布
    
    公式: z = (x - μ) / σ
    
    参数:
        data: 原始数据列表
    
    返回:
        标准化后的数据列表
    """
    if not data or len(data) < 2:
        return data
    mean = calculate_mean(data)
    std = calculate_std(data)
    if std == 0:
        return [0.0] * len(data)  # 避免除以零
    return [(x - mean) / std for x in data]


def standardize_dataset(dataset: List[List[float]]) -> List[List[float]]:
    """
    对整个数据集进行标准化（按特征列）
    
    参数:
        dataset: 二维列表，每行是一个样本，每列是一个特征
    
    返回:
        标准化后的数据集
    """
    if not dataset or not dataset[0]:
        return dataset
    
    num_features = len(dataset[0])
    standardized = []
    
    # 对每个特征列进行标准化
    for feature_idx in range(num_features):
        feature_column = [row[feature_idx] for row in dataset]
        standardized_column = z_score_standardize(feature_column)
        standardized.append(standardized_column)
    
    # 转置回原来的格式
    return [[standardized[f][i] for f in range(num_features)] 
            for i in range(len(dataset))]


# ============================================
# 第3部分：特征提取示例
# ============================================

class SimpleDigitFeatures:
    """
    简化的手写数字特征提取器
    
    用于从二值化图像中提取基本特征
    """
    
    @staticmethod
    def extract_features(image: List[List[int]]) -> Dict[str, float]:
        """
        从二值图像中提取特征
        
        参数:
            image: 二维列表，0表示背景，1表示前景
        
        返回:
            包含各种特征的字典
        """
        features = {}
        
        # 特征1：黑色像素数量（笔画总量）
        features['pixel_count'] = sum(sum(row) for row in image)
        
        # 特征2：图像密度
        total_pixels = len(image) * len(image[0]) if image else 1
        features['density'] = features['pixel_count'] / total_pixels
        
        # 特征3：水平对称性
        features['horizontal_symmetry'] = \
            SimpleDigitFeatures._horizontal_symmetry(image)
        
        # 特征4：垂直对称性
        features['vertical_symmetry'] = \
            SimpleDigitFeatures._vertical_symmetry(image)
        
        # 特征5：中心偏移（质心位置）
        center_x, center_y = SimpleDigitFeatures._centroid(image)
        features['center_x'] = center_x
        features['center_y'] = center_y
        
        # 特征6：四个象限的像素分布
        quadrants = SimpleDigitFeatures._quadrant_counts(image)
        features['quad_tl'] = quadrants[0]  # 左上
        features['quad_tr'] = quadrants[1]  # 右上
        features['quad_bl'] = quadrants[2]  # 左下
        features['quad_br'] = quadrants[3]  # 右下
        
        return features
    
    @staticmethod
    def _horizontal_symmetry(image: List[List[int]]) -> float:
        """计算水平对称性（左右对称）"""
        if not image or not image[0]:
            return 0.0
        height = len(image)
        width = len(image[0])
        matches = 0
        total = 0
        
        for i in range(height):
            for j in range(width // 2):
                mirror_j = width - 1 - j
                total += 1
                if image[i][j] == image[i][mirror_j]:
                    matches += 1
        
        return matches / total if total > 0 else 0.0
    
    @staticmethod
    def _vertical_symmetry(image: List[List[int]]) -> float:
        """计算垂直对称性（上下对称）"""
        if not image or not image[0]:
            return 0.0
        height = len(image)
        width = len(image[0])
        matches = 0
        total = 0
        
        for i in range(height // 2):
            mirror_i = height - 1 - i
            for j in range(width):
                total += 1
                if image[i][j] == image[mirror_i][j]:
                    matches += 1
        
        return matches / total if total > 0 else 0.0
    
    @staticmethod
    def _centroid(image: List[List[int]]) -> Tuple[float, float]:
        """计算质心（黑色像素的平均位置）"""
        if not image or not image[0]:
            return 0.5, 0.5
        
        height = len(image)
        width = len(image[0])
        sum_x = sum_y = count = 0
        
        for i in range(height):
            for j in range(width):
                if image[i][j] == 1:
                    sum_x += j
                    sum_y += i
                    count += 1
        
        if count == 0:
            return 0.5, 0.5
        
        return (sum_x / count) / width, (sum_y / count) / height
    
    @staticmethod
    def _quadrant_counts(image: List[List[int]]) -> List[int]:
        """统计四个象限的像素数量"""
        if not image or not image[0]:
            return [0, 0, 0, 0]
        
        height = len(image)
        width = len(image[0])
        mid_h = height // 2
        mid_w = width // 2
        
        # [左上, 右上, 左下, 右下]
        counts = [0, 0, 0, 0]
        
        for i in range(height):
            for j in range(width):
                if image[i][j] == 1:
                    if i < mid_h and j < mid_w:
                        counts[0] += 1  # 左上
                    elif i < mid_h and j >= mid_w:
                        counts[1] += 1  # 右上
                    elif i >= mid_h and j < mid_w:
                        counts[2] += 1  # 左下
                    else:
                        counts[3] += 1  # 右下
        
        return counts


# ============================================
# 第4部分：ASCII可视化
# ============================================

def plot_ascii_histogram(data: List[float], bins: int = 10, width: int = 50):
    """
    使用ASCII字符绘制直方图
    
    参数:
        data: 数据列表
        bins: 分箱数量
        width: 图表宽度（字符数）
    """
    if not data:
        print("无数据可显示")
        return
    
    data_min = min(data)
    data_max = max(data)
    
    if data_max == data_min:
        print(f"所有值相同: {data_min}")
        return
    
    # 创建分箱
    bin_width = (data_max - data_min) / bins
    bin_counts = [0] * bins
    
    for value in data:
        bin_idx = min(int((value - data_min) / bin_width), bins - 1)
        bin_counts[bin_idx] += 1
    
    # 计算比例
    max_count = max(bin_counts)
    
    # 打印标题
    print(f"\n{'='*60}")
    print(f"数据直方图 (范围: {data_min:.2f} ~ {data_max:.2f})")
    print(f"{'='*60}")
    
    # 绘制直方图
    for i, count in enumerate(bin_counts):
        bin_start = data_min + i * bin_width
        bin_end = bin_start + bin_width
        bar_length = int((count / max_count) * width) if max_count > 0 else 0
        bar = '█' * bar_length
        print(f"[{bin_start:6.2f},{bin_end:6.2f}) |{bar:<{width}}| ({count})")
    
    print(f"{'='*60}")
    print(f"样本数: {len(data)}, 均值: {calculate_mean(data):.2f}, 标准差: {calculate_std(data):.2f}")
    print(f"{'='*60}\n")


def plot_ascii_line(y_values: List[float], height: int = 15, width: int = 60):
    """
    使用ASCII字符绘制折线图
    
    参数:
        y_values: Y轴数据
        height: 图表高度
        width: 图表宽度
    """
    if not y_values:
        print("无数据可显示")
        return
    
    y_min = min(y_values)
    y_max = max(y_values)
    
    if y_max == y_min:
        y_max += 1
        y_min -= 1
    
    # 缩放数据到图表高度
    scaled = [int((y - y_min) / (y_max - y_min) * (height - 1)) for y in y_values]
    
    # 采样以适配宽度
    if len(scaled) > width:
        step = len(scaled) / width
        scaled = [scaled[int(i * step)] for i in range(width)]
    
    print(f"\n{'='*60}")
    print(f"折线图 (Y范围: {y_min:.2f} ~ {y_max:.2f})")
    print(f"{'='*60}")
    
    # 绘制
    for row in range(height - 1, -1, -1):
        line = ""
        for val in scaled:
            if val == row:
                line += "●"
            elif val > row:
                line += "│"
            else:
                line += " "
        y_label = y_min + (row / (height - 1)) * (y_max - y_min)
        print(f"{y_label:8.2f} | {line}")
    
    print(f"{' '*9}+{'─'*len(scaled)}")
    print(f"{'='*60}\n")


def plot_vector_2d(v1: Vector, v2: Vector, title: str = "2D向量可视化"):
    """
    在2D平面上可视化两个向量（仅限于前2维）
    
    参数:
        v1: 第一个向量
        v2: 第二个向量
        title: 图表标题
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # 只取前2维
    x1, y1 = v1[0], v1[1] if len(v1) > 1 else 0
    x2, y2 = v2[0], v2[1] if len(v2) > 1 else 0
    
    # 确定坐标范围
    all_coords = [x1, y1, x2, y2]
    coord_min = min(all_coords + [0])
    coord_max = max(all_coords + [1])
    padding = (coord_max - coord_min) * 0.2
    coord_min -= padding
    coord_max += padding
    
    # 绘制坐标系
    size = 20
    grid = [[' ' for _ in range(size)] for _ in range(size)]
    
    # 绘制坐标轴
    origin_x = int((0 - coord_min) / (coord_max - coord_min) * (size - 1))
    origin_y = int((coord_max - 0) / (coord_max - coord_min) * (size - 1))
    
    for i in range(size):
        if 0 <= origin_x < size:
            grid[i][origin_x] = '│' if i != origin_y else '+'
        if 0 <= origin_y < size:
            grid[origin_y][i] = '─' if i != origin_x else '+'
    
    # 绘制向量
    def plot_point(x, y, char):
        px = int((x - coord_min) / (coord_max - coord_min) * (size - 1))
        py = int((coord_max - y) / (coord_max - coord_min) * (size - 1))
        if 0 <= px < size and 0 <= py < size:
            grid[py][px] = char
    
    # 绘制从原点到向量端点的线
    def plot_line(x1, y1, x2, y2, char):
        steps = max(abs(x2 - x1), abs(y2 - y1)) * 10
        for i in range(int(steps) + 1):
            t = i / steps if steps > 0 else 0
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            plot_point(x, y, char)
    
    plot_line(0, 0, x1, y1, '*')
    plot_line(0, 0, x2, y2, '#')
    plot_point(x1, y1, 'A')
    plot_point(x2, y2, 'B')
    
    # 打印
    print(f"  向量A: ({x1:.2f}, {y1:.2f}) - 用'*'表示")
    print(f"  向量B: ({x2:.2f}, {y2:.2f}) - 用'#'表示")
    print()
    
    # Y轴标签
    y_labels = [coord_max - i * (coord_max - coord_min) / (size - 1) 
                for i in range(size)]
    
    for i, row in enumerate(grid):
        label = f"{y_labels[i]:6.1f} │" if i % 4 == 0 else "       │"
        print(label + ''.join(row) + '│')
    
    print("       └" + '─' * size + '┘')
    x_label_start = f"{coord_min:6.1f}"
    x_label_end = f"{coord_max:6.1f}"
    print(f"         {x_label_start}{' '* (size - 12)}{x_label_end}")
    print(f"{'='*60}\n")


# ============================================
# 第5部分：演示和测试
# ============================================

def demo_vectors():
    """演示向量运算"""
    print("\n" + "="*60)
    print("向量运算演示")
    print("="*60)
    
    # 创建向量
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    
    print(f"向量 v1 = {v1}")
    print(f"向量 v2 = {v2}")
    print()
    
    # 加法
    v_add = v1 + v2
    print(f"v1 + v2 = {v_add}")
    
    # 减法
    v_sub = v2 - v1
    print(f"v2 - v1 = {v_sub}")
    
    # 数乘
    v_mul = v1 * 3
    print(f"v1 × 3 = {v_mul}")
    
    # 点积
    dot = v1.dot(v2)
    print(f"v1 · v2 = {dot} (= 1×4 + 2×5 + 3×6)")
    
    # 模长
    mag = v1.magnitude()
    print(f"||v1|| = {mag:.4f}")
    
    # 归一化
    v_norm = v1.normalize()
    print(f"v1归一化 = {v_norm}")
    print(f"归一化后长度 = {v_norm.magnitude():.4f}")


def demo_distances():
    """演示距离计算"""
    print("\n" + "="*60)
    print("距离度量演示")
    print("="*60)
    
    # 创建两个点
    point_a = Vector([1, 2])
    point_b = Vector([4, 6])
    
    print(f"点 A = {point_a}")
    print(f"点 B = {point_b}")
    print()
    
    # 欧氏距离
    euclidean = euclidean_distance(point_a, point_b)
    print(f"欧氏距离 = {euclidean:.4f}")
    print(f"  计算: √((4-1)² + (6-2)²) = √(9 + 16) = √25 = 5")
    print()
    
    # 曼哈顿距离
    manhattan = manhattan_distance(point_a, point_b)
    print(f"曼哈顿距离 = {manhattan:.4f}")
    print(f"  计算: |4-1| + |6-2| = 3 + 4 = 7")
    print()
    
    # 可视化
    plot_vector_2d(point_a, point_b, "欧氏距离 vs 曼哈顿距离")
    
    # 余弦相似度
    v1 = Vector([1, 2, 3])
    v2 = Vector([2, 4, 6])
    cos_sim = cosine_similarity(v1, v2)
    print(f"向量 {v1} 和 {v2}")
    print(f"余弦相似度 = {cos_sim:.4f}")
    print("  （这两个向量方向相同，所以相似度接近1）")


def demo_normalization():
    """演示数据标准化和归一化"""
    print("\n" + "="*60)
    print("数据标准化演示")
    print("="*60)
    
    # 原始数据
    data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    print(f"原始数据: {data}")
    print(f"均值: {calculate_mean(data):.2f}")
    print(f"标准差: {calculate_std(data):.2f}")
    print()
    
    # Min-Max归一化
    normalized = min_max_normalize(data)
    print(f"Min-Max归一化 [0,1]: {[round(x, 3) for x in normalized]}")
    print()
    
    # Z-score标准化
    standardized = z_score_standardize(data)
    print(f"Z-score标准化: {[round(x, 3) for x in standardized]}")
    print(f"标准化后均值: {calculate_mean(standardized):.6f} (≈0)")
    print(f"标准化后标准差: {calculate_std(standardized):.6f} (≈1)")
    
    # 可视化
    plot_ascii_histogram(data, bins=8)


def demo_digit_features():
    """演示手写数字特征提取"""
    print("\n" + "="*60)
    print("手写数字特征提取演示")
    print("="*60)
    
    # 简化的数字"0"（4x4）
    digit_zero = [
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ]
    
    # 简化的数字"1"（4x4）
    digit_one = [
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0]
    ]
    
    extractor = SimpleDigitFeatures()
    
    print("数字 '0' 的表示:")
    for row in digit_zero:
        print("  " + ''.join('█' if x else ' ' for x in row))
    
    features_zero = extractor.extract_features(digit_zero)
    print("\n提取的特征:")
    for key, value in features_zero.items():
        print(f"  {key}: {value:.3f}")
    
    print("\n" + "-"*40)
    print("\n数字 '1' 的表示:")
    for row in digit_one:
        print("  " + ''.join('█' if x else ' ' for x in row))
    
    features_one = extractor.extract_features(digit_one)
    print("\n提取的特征:")
    for key, value in features_one.items():
        print(f"  {key}: {value:.3f}")
    
    print("\n" + "-"*40)
    print("\n特征对比:")
    print(f"  像素数量: 0号={features_zero['pixel_count']}, 1号={features_one['pixel_count']}")
    print(f"  水平对称: 0号={features_zero['horizontal_symmetry']:.3f}, 1号={features_one['horizontal_symmetry']:.3f}")
    print(f"  中心位置X: 0号={features_zero['center_x']:.3f}, 1号={features_one['center_x']:.3f}")


def demo_fruit_classifier():
    """
    演示：水果分类器（简化版K近邻算法）
    
    使用向量距离来分类水果
    """
    print("\n" + "="*60)
    print("水果分类器演示（基于向量距离）")
    print("="*60)
    
    # 训练数据: [直径(cm), 重量(g), 甜度(1-10)]
    # 标签: 0=苹果, 1=橙子
    training_data = [
        (Vector([8.0, 150, 7]), "苹果"),
        (Vector([7.5, 140, 6]), "苹果"),
        (Vector([8.2, 155, 7]), "苹果"),
        (Vector([7.0, 160, 8]), "橙子"),
        (Vector([7.8, 170, 8]), "橙子"),
        (Vector([8.5, 180, 7]), "橙子"),
    ]
    
    print("训练数据:")
    for vec, label in training_data:
        print(f"  {label}: 直径={vec[0]}cm, 重量={vec[1]}g, 甜度={vec[2]}")
    
    # 新的水果（待分类）
    new_fruit = Vector([7.8, 165, 8])
    print(f"\n新水果: 直径={new_fruit[0]}cm, 重量={new_fruit[1]}g, 甜度={new_fruit[2]}")
    
    # 计算与每个训练样本的距离
    print("\n计算欧氏距离:")
    distances = []
    for vec, label in training_data:
        dist = euclidean_distance(new_fruit, vec)
        distances.append((dist, label))
        print(f"  与{label}的距离: {dist:.2f}")
    
    # K近邻分类 (K=3)
    K = 3
    distances.sort(key=lambda x: x[0])
    nearest = distances[:K]
    
    print(f"\n最近的{K}个邻居:")
    for dist, label in nearest:
        print(f"  {label}: 距离={dist:.2f}")
    
    # 投票
    from collections import Counter
    votes = Counter(label for _, label in nearest)
    predicted = votes.most_common(1)[0][0]
    
    print(f"\n预测结果: 这个水果是【{predicted}】!")
    print(f"  ({votes[predicted]}/{K} 个邻居投票)")


def run_all_demos():
    """运行所有演示"""
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + "  第2章：机器如何'看见'数据 - 代码演示".center(54) + "█")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    demo_vectors()
    demo_distances()
    demo_normalization()
    demo_digit_features()
    demo_fruit_classifier()
    
    print("\n" + "█"*60)
    print("█" + "  所有演示完成！".center(56) + "█")
    print("█"*60 + "\n")


# ============================================
# 主程序入口
# ============================================

if __name__ == "__main__":
    run_all_demos()
