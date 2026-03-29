"""
第六章：K近邻算法 - 从零实现
K-Nearest Neighbors Algorithm - From Scratch

作者：机器学习教材编写组
参考：Cover & Hart (1967) 经典论文

本模块完全使用纯Python实现KNN算法，不依赖任何外部库（如NumPy/SciPy），
用于教学目的，帮助理解算法的核心原理。
"""

import math
from collections import Counter


class KNNClassifier:
    """
    K近邻分类器 - 纯Python实现
    
    理论基础：
    Cover, T. M., & Hart, P. E. (1967). Nearest neighbor pattern classification. 
    IEEE Transactions on Information Theory, 13(1), 21-27.
    
    KNN是一种"懒惰学习"（lazy learning）算法，训练阶段仅存储数据，
    所有计算延迟到预测阶段进行。
    
    Attributes:
        k (int): 邻居数量
        distance_metric (str): 距离度量方式 ('euclidean', 'manhattan', 'minkowski')
        weights (str): 投票权重 ('uniform' 等权重, 'distance' 距离加权)
        X_train (list): 训练特征
        y_train (list): 训练标签
    """
    
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform'):
        """
        初始化KNN分类器
        
        Args:
            k (int): 邻居数量，默认3
            distance_metric (str): 距离度量方式，可选 'euclidean', 'manhattan', 'minkowski'
            weights (str): 投票权重，可选 'uniform' 或 'distance'
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k必须是正整数，但得到 {k}")
        
        if distance_metric not in ['euclidean', 'manhattan', 'minkowski']:
            raise ValueError(f"未知的距离度量: {distance_metric}")
        
        if weights not in ['uniform', 'distance']:
            raise ValueError(f"未知的权重类型: {weights}")
        
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
        
        print(f"[KNN] 初始化完成: k={k}, metric={distance_metric}, weights={weights}")
    
    def fit(self, X, y):
        """
        "训练"KNN模型
        
        注意：KNN是"懒惰学习"（lazy learning），
        所谓的"训练"其实就是把数据存起来！
        
        Args:
            X (list): 训练特征，列表的列表 [[x1, x2, ...], ...]
            y (list): 训练标签，列表 [y1, y2, ...]
        
        Returns:
            self: 返回自身，支持链式调用
        """
        if len(X) != len(y):
            raise ValueError(f"特征和标签数量不匹配: {len(X)} vs {len(y)}")
        
        if len(X) == 0:
            raise ValueError("训练数据不能为空")
        
        self.X_train = [list(x) for x in X]  # 深拷贝
        self.y_train = list(y)
        
        # 验证数据维度一致性
        if len(X) > 0:
            expected_dim = len(X[0])
            for i, x in enumerate(X):
                if len(x) != expected_dim:
                    raise ValueError(f"第{i}个样本维度不一致: {len(x)} vs {expected_dim}")
        
        print(f"[KNN] 存储了 {len(X)} 个训练样本，维度: {len(X[0]) if X else 0}")
        return self
    
    def _euclidean_distance(self, x1, x2):
        """
        计算欧氏距离（L2范数）
        
        公式: d(x, y) = sqrt(sum((xi - yi)^2))
        
        Args:
            x1 (list): 第一个向量
            x2 (list): 第二个向量
        
        Returns:
            float: 欧氏距离
        """
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))
    
    def _manhattan_distance(self, x1, x2):
        """
        计算曼哈顿距离（L1范数）
        
        公式: d(x, y) = sum(|xi - yi|)
        
        Args:
            x1 (list): 第一个向量
            x2 (list): 第二个向量
        
        Returns:
            float: 曼哈顿距离
        """
        return sum(abs(a - b) for a, b in zip(x1, x2))
    
    def _minkowski_distance(self, x1, x2, p=3):
        """
        计算闵可夫斯基距离（Lp范数）
        
        公式: d(x, y) = (sum(|xi - yi|^p))^(1/p)
        
        特殊情况:
        - p=1: 曼哈顿距离
        - p=2: 欧氏距离
        
        Args:
            x1 (list): 第一个向量
            x2 (list): 第二个向量
            p (float): 范数参数，默认3
        
        Returns:
            float: 闵可夫斯基距离
        """
        return sum(abs(a - b) ** p for a, b in zip(x1, x2)) ** (1/p)
    
    def _compute_distance(self, x1, x2):
        """
        根据配置计算距离
        
        Args:
            x1 (list): 第一个向量
            x2 (list): 第二个向量
        
        Returns:
            float: 距离值
        """
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        elif self.distance_metric == 'minkowski':
            return self._minkowski_distance(x1, x2)
        else:
            raise ValueError(f"未知的距离度量: {self.distance_metric}")
    
    def _get_neighbors(self, x):
        """
        找出x的K个最近邻居
        
        算法:
        1. 计算到所有训练样本的距离
        2. 按距离排序
        3. 取前K个
        
        Args:
            x (list): 查询样本
        
        Returns:
            list: [(距离, 标签), ...] 按距离排序的前K个
        """
        # 计算到所有训练样本的距离
        distances = []
        for xi, yi in zip(self.X_train, self.y_train):
            dist = self._compute_distance(x, xi)
            distances.append((dist, yi))
        
        # 按距离排序，取前K个
        distances.sort(key=lambda item: item[0])
        
        # 如果K大于训练样本数，调整K
        actual_k = min(self.k, len(distances))
        return distances[:actual_k]
    
    def _vote(self, neighbors):
        """
        根据邻居投票决定类别
        
        Args:
            neighbors (list): [(距离, 标签), ...]
        
        Returns:
            预测的类别标签
        """
        if not neighbors:
            raise ValueError("邻居列表为空")
        
        if self.weights == 'uniform':
            # 等权重投票 - 简单多数表决
            votes = [label for _, label in neighbors]
            vote_counts = Counter(votes)
            # 返回得票最多的类别
            return vote_counts.most_common(1)[0][0]
        
        elif self.weights == 'distance':
            # 距离加权投票 - 更近的邻居有更大发言权
            weighted_votes = {}
            for dist, label in neighbors:
                # 避免除以零
                weight = 1.0 / (dist + 1e-10)
                weighted_votes[label] = weighted_votes.get(label, 0) + weight
            # 返回加权票数最多的类别
            return max(weighted_votes, key=weighted_votes.get)
        
        else:
            raise ValueError(f"未知的权重类型: {self.weights}")
    
    def predict_single(self, x):
        """
        预测单个样本的类别
        
        Args:
            x (list): 特征向量
        
        Returns:
            预测的类别标签
        """
        neighbors = self._get_neighbors(x)
        return self._vote(neighbors)
    
    def predict(self, X):
        """
        预测多个样本的类别
        
        Args:
            X (list): 特征向量列表
        
        Returns:
            list: 预测类别列表
        """
        return [self.predict_single(x) for x in X]
    
    def predict_proba(self, x):
        """
        预测概率（每个类别的置信度）
        
        返回每个类别的概率（邻居中该类别的比例或加权比例）
        
        Args:
            x (list): 特征向量
        
        Returns:
            dict: {类别: 概率, ...}
        """
        neighbors = self._get_neighbors(x)
        
        if self.weights == 'uniform':
            votes = [label for _, label in neighbors]
            vote_counts = Counter(votes)
            total = len(neighbors)
            return {label: count/total for label, count in vote_counts.items()}
        
        else:  # distance weighted
            weighted_votes = {}
            for dist, label in neighbors:
                weight = 1.0 / (dist + 1e-10)
                weighted_votes[label] = weighted_votes.get(label, 0) + weight
            total = sum(weighted_votes.values())
            return {label: weight/total for label, weight in weighted_votes.items()}
    
    def get_neighbors_info(self, x):
        """
        获取邻居的详细信息（用于调试和分析）
        
        Args:
            x (list): 特征向量
        
        Returns:
            list: [(距离, 标签, 特征), ...]
        """
        # 计算距离并保存索引
        distances_with_idx = []
        for idx, (xi, yi) in enumerate(zip(self.X_train, self.y_train)):
            dist = self._compute_distance(x, xi)
            distances_with_idx.append((dist, yi, xi, idx))
        
        # 排序并取前K个
        distances_with_idx.sort(key=lambda item: item[0])
        actual_k = min(self.k, len(distances_with_idx))
        
        return [(d, y, xi) for d, y, xi, _ in distances_with_idx[:actual_k]]


class KNNRegressor:
    """
    K近邻回归器 - 纯Python实现
    
    用于连续值的预测，通过K个邻居的目标值平均或加权平均得到预测值。
    
    Attributes:
        k (int): 邻居数量
        distance_metric (str): 距离度量方式
        weights (str): 权重类型
        X_train (list): 训练特征
        y_train (list): 训练目标值
    """
    
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform'):
        """
        初始化KNN回归器
        
        Args:
            k (int): 邻居数量，默认3
            distance_metric (str): 距离度量方式
            weights (str): 权重类型，'uniform' 或 'distance'
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k必须是正整数，但得到 {k}")
        
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        存储训练数据
        
        Args:
            X (list): 特征向量列表
            y (list): 目标值列表
        
        Returns:
            self
        """
        if len(X) != len(y):
            raise ValueError(f"特征和目标值数量不匹配: {len(X)} vs {len(y)}")
        
        self.X_train = [list(x) for x in X]
        self.y_train = list(y)
        return self
    
    def _compute_distance(self, x1, x2):
        """计算距离（复用分类器的实现）"""
        if self.distance_metric == 'euclidean':
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))
        elif self.distance_metric == 'manhattan':
            return sum(abs(a - b) for a, b in zip(x1, x2))
        else:
            return sum(abs(a - b) ** 3 for a, b in zip(x1, x2)) ** (1/3)
    
    def predict_single(self, x):
        """
        预测单个样本的回归值
        
        Args:
            x (list): 特征向量
        
        Returns:
            float: 预测值
        """
        # 计算距离
        distances = [(self._compute_distance(x, xi), yi) 
                     for xi, yi in zip(self.X_train, self.y_train)]
        distances.sort(key=lambda item: item[0])
        neighbors = distances[:self.k]
        
        if self.weights == 'uniform':
            # 简单平均
            return sum(y for _, y in neighbors) / len(neighbors)
        else:
            # 距离加权平均
            weighted_sum = 0
            weight_total = 0
            for dist, y in neighbors:
                weight = 1.0 / (dist + 1e-10)
                weighted_sum += weight * y
                weight_total += weight
            return weighted_sum / weight_total
    
    def predict(self, X):
        """
        预测多个样本
        
        Args:
            X (list): 特征向量列表
        
        Returns:
            list: 预测值列表
        """
        return [self.predict_single(x) for x in X]


# ============================================================
# 可视化辅助函数
# ============================================================

def print_ascii_decision_boundary():
    """打印ASCII艺术风格的决策边界示意图"""
    print("""
    KNN决策边界示意图 (K=1 vs K=5)
    
    K=1 (过拟合)              K=5 (较平滑)
    ┌─────────────────┐      ┌─────────────────┐
    │ ·▓·  ░░  ▓· ▓· │      │ ▓▓▓  ░░░  ▓▓▓  │
    │▓ ▓▓· ░░  ▓·▓▓▓ │      │ ▓▓▓  ░░░  ▓▓▓  │
    │·▓▓·· ░░  ▓▓·▓· │      │ ▓▓▓  ░░░  ▓▓▓  │
    │░░░░░░░░░░░░░░░░│      │░░░░░░░░░░░░░░░░│
    │░░░░░░░░░░░░░░░░│      │░░░░░░░░░░░░░░░░│
    │··▓▓▓ ░░ ▓▓▓···│      │ ▓▓▓  ░░░  ▓▓▓  │
    │·▓▓·▓ ░░ ▓·▓▓▓·│      │ ▓▓▓  ░░░  ▓▓▓  │
    └─────────────────┘      └─────────────────┘
    
    ▓ = 类别A    ░ = 类别B    · = 边界噪声
    """)


def print_distance_comparison():
    """打印距离度量对比图"""
    print("""
    欧氏距离 vs 曼哈顿距离
    
    点A(0,0) 到 点B(3,4):
    
         欧氏距离              曼哈顿距离
            ● B                    ● B
           ╱│                    │
          ╱ │                    │ 4
         ╱  │ 5                  │
        ╱   │                    └───●
       ●────┘                   A      3
       A
    
    欧氏距离 = √(3²+4²) = 5
    曼哈顿距离 = |3|+|4| = 7
    """)


# ============================================================
# 演示与测试
# ============================================================

def demo_fruit_classification():
    """演示：水果分类问题"""
    print("=" * 60)
    print("【演示1】水果分类问题")
    print("=" * 60)
    
    # 训练数据：[重量(克), 直径(cm)]
    X_fruit = [
        [150, 7],   # 苹果
        [170, 7.5], # 苹果
        [140, 6.5], # 苹果
        [130, 6],   # 苹果
        [200, 8],   # 橙子
        [220, 8.5], # 橙子
        [180, 7.8], # 橙子
        [210, 8.2], # 橙子
    ]
    y_fruit = ['🍎 苹果', '🍎 苹果', '🍎 苹果', '🍎 苹果', 
               '🍊 橙子', '🍊 橙子', '🍊 橙子', '🍊 橙子']
    
    # 新来的水果
    new_fruit = [160, 7.2]
    
    print("\n📊 训练样本（重量, 直径）:")
    for (w, d), label in zip(X_fruit, y_fruit):
        print(f"  ({w:3d}g, {d:4.1f}cm) → {label}")
    
    print(f"\n🆕 新水果: ({new_fruit[0]}g, {new_fruit[1]}cm)")
    
    # 训练KNN模型
    knn = KNNClassifier(k=3)
    knn.fit(X_fruit, y_fruit)
    
    # 获取邻居信息
    neighbors = knn.get_neighbors_info(new_fruit)
    print(f"\n🔍 最近的{knn.k}个邻居:")
    for i, (dist, label, features) in enumerate(neighbors, 1):
        print(f"  {i}. {label} @ ({features[0]}g, {features[1]}cm), 距离={dist:.2f}")
    
    # 预测
    prediction = knn.predict_single(new_fruit)
    probabilities = knn.predict_proba(new_fruit)
    
    print(f"\n✅ 预测结果: {prediction}")
    print("📈 各类别概率:")
    for label, prob in sorted(probabilities.items()):
        bar = '█' * int(prob * 30)
        print(f"  {label}: {prob:.2%} {bar}")


def demo_distance_metrics():
    """演示：不同距离度量的比较"""
    print("\n" + "=" * 60)
    print("【演示2】不同距离度量的比较")
    print("=" * 60)
    
    point_a = [0, 0]
    point_b = [3, 4]
    
    print(f"\n📍 点A: {point_a}")
    print(f"📍 点B: {point_b}")
    
    print("\n📏 距离计算:")
    metrics = [
        ('euclidean', '欧氏距离'),
        ('manhattan', '曼哈顿距离'),
        ('minkowski', '闵可夫斯基(p=3)')
    ]
    
    for metric, name in metrics:
        knn_temp = KNNClassifier(k=1, distance_metric=metric)
        dist = knn_temp._compute_distance(point_a, point_b)
        print(f"  {name:20s}: {dist:.2f}")
    
    print("\n📝 数学公式:")
    print("  欧氏距离 = √(3² + 4²) = 5.0")
    print("  曼哈顿距离 = |3| + |4| = 7.0")
    print("  闵可夫斯基 = (|3|³ + |4|³)^(1/3) ≈ 4.50")


def demo_k_value_impact():
    """演示：K值选择的影响"""
    print("\n" + "=" * 60)
    print("【演示3】K值选择的影响")
    print("=" * 60)
    
    # 构造一个稍微复杂的数据集
    X_complex = [
        [1, 1], [1, 2], [2, 1],  # 类别A聚集
        [5, 5], [5, 6], [6, 5],  # 类别B聚集
        [3, 3],  # 边界点，靠近A
    ]
    y_complex = ['A', 'A', 'A', 'B', 'B', 'B', 'A']
    
    test_point = [3, 3.5]  # 测试点
    
    print(f"\n📍 测试点: {test_point}")
    print("\n📊 不同K值的预测结果:")
    
    for k in [1, 3, 5, 7]:
        knn_k = KNNClassifier(k=k)
        knn_k.fit(X_complex, y_complex)
        pred = knn_k.predict_single(test_point)
        
        # 获取邻居信息
        neighbors = knn_k.get_neighbors_info(test_point)
        neighbor_labels = [n[1] for n in neighbors]
        
        print(f"\n  K={k}: 预测类别 = {pred}")
        print(f"       邻居: {neighbor_labels}")


def demo_house_price_prediction():
    """演示：房价预测（KNN回归）"""
    print("\n" + "=" * 60)
    print("【演示4】房价预测（KNN回归）")
    print("=" * 60)
    
    # 训练数据：[面积(平米), 卧室数]
    X_house = [
        [50, 1], [60, 1], [80, 2], [90, 2],
        [100, 3], [120, 3], [150, 4], [200, 5]
    ]
    y_house = [100, 120, 160, 180, 250, 300, 400, 550]  # 价格（万元）
    
    # 新房子
    new_house = [110, 3]
    
    print("\n🏠 训练样本（面积, 卧室）→ 价格:")
    for (area, rooms), price in zip(X_house, y_house):
        print(f"  ({area:3d}m², {rooms}室) → {price}万元")
    
    print(f"\n🆕 新房子: ({new_house[0]}m², {new_house[1]}室)")
    
    # 使用KNN回归
    knn_reg = KNNRegressor(k=3, weights='distance')
    knn_reg.fit(X_house, y_house)
    predicted_price = knn_reg.predict_single(new_house)
    
    print(f"\n💰 预测价格: {predicted_price:.1f}万元")
    
    # 分析邻居
    knn_temp = KNNClassifier(k=3)  # 借用分类器来获取邻居
    knn_temp.X_train = X_house
    knn_temp.y_train = [str(p) for p in y_house]
    neighbors = knn_temp.get_neighbors_info(new_house)
    
    print("\n🔍 最近的3个邻居:")
    for dist, price, features in neighbors:
        print(f"  ({features[0]}m², {features[1]}室) → {price}万元, 距离={dist:.2f}")


def demo_weight_comparison():
    """演示：权重策略对比"""
    print("\n" + "=" * 60)
    print("【演示5】等权重 vs 距离加权")
    print("=" * 60)
    
    # 构造有噪声的数据
    X_noisy = [[1], [2], [3], [20]]  # 20是噪声点，远离其他
    y_noisy = ['A', 'A', 'A', 'B']   # 大多数是A
    
    test_x = [2.5]
    
    print(f"\n📊 训练数据: {list(zip(X_noisy, y_noisy))}")
    print(f"📍 测试点: {test_x}")
    
    print(f"\n  注: 样本[20]是远离其他点的噪声点")
    
    for weight_type in ['uniform', 'distance']:
        knn_w = KNNClassifier(k=3, weights=weight_type)
        knn_w.fit(X_noisy, y_noisy)
        pred = knn_w.predict_single(test_x)
        prob = knn_w.predict_proba(test_x)
        
        name = '等权重' if weight_type == 'uniform' else '距离加权'
        print(f"\n⚖️ {name}:")
        print(f"  预测: {pred}")
        print(f"  概率: {prob}")
        
        if weight_type == 'uniform':
            print("  说明: 远距离的噪声点B有同等投票权")
        else:
            print("  说明: 远距离的噪声点B投票权被稀释")


def demo_knn_from_scratch():
    """完整演示：展示KNN的核心工作原理"""
    print("\n" + "=" * 60)
    print("【演示6】KNN工作原理可视化")
    print("=" * 60)
    
    print("""
    KNN算法工作流程:
    
    1️⃣  训练阶段（存储数据）
        ┌─────────────────────────────────────┐
        │  特征空间                            │
        │                                     │
        │    ▲                                │
        │  y │  ▓        ○                   │
        │    │    ▓      ○ ○                 │
        │    │  ▓  ▓      ○                   │
        │    └──────────────────▶            │
        │         x                           │
        │                                     │
        │    ▓ = 类别A    ○ = 类别B            │
        └─────────────────────────────────────┘
        
    2️⃣  预测阶段（新样本 ?）
        ┌─────────────────────────────────────┐
        │  特征空间                            │
        │                                     │
        │    ▲                                │
        │  y │  ▓        ○      ★ ?           │
        │    │    ▓      ○ ○                 │
        │    │  ▓  ▓      ○                   │
        │    └──────────────────▶            │
        │         x                           │
        │                                     │
        │    ★ = 待预测的新样本               │
        └─────────────────────────────────────┘
        
    3️⃣  计算距离
        ┌─────────────────────────────────────┐
        │  计算★到所有已知点的距离...          │
        │                                     │
        │  d(★,▓₁) = 2.3                      │
        │  d(★,▓₂) = 3.1                      │
        │  d(★,○₁) = 1.5  ← 最近！            │
        │  d(★,○₂) = 2.8                      │
        │  ...                                │
        └─────────────────────────────────────┘
        
    4️⃣  找出K个最近邻居 (假设K=3)
        ┌─────────────────────────────────────┐
        │  最近的3个邻居:                      │
        │    1. ○₁ (距离1.5)                  │
        │    2. ▓₁ (距离2.3)                  │
        │    3. ○₂ (距离2.8)                  │
        │                                     │
        │  类别统计: ○: 2票, ▓: 1票           │
        └─────────────────────────────────────┘
        
    5️⃣  投票决策
        ┌─────────────────────────────────────┐
        │  多数表决: ○ 获胜！                  │
        │                                     │
        │  预测结果: ★ 属于类别B (○)          │
        └─────────────────────────────────────┘
    """)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" " * 20 + "K近邻算法 - 从零实现")
    print(" " * 10 + "K-Nearest Neighbors: From Scratch")
    print("=" * 70)
    
    # 运行所有演示
    demo_fruit_classification()
    demo_distance_metrics()
    demo_k_value_impact()
    demo_house_price_prediction()
    demo_weight_comparison()
    demo_knn_from_scratch()
    
    # 打印可视化
    print("\n" + "=" * 60)
    print("【附录】可视化图表")
    print("=" * 60)
    print_distance_comparison()
    print_ascii_decision_boundary()
    
    print("\n" + "=" * 70)
    print("✅ 所有演示完成！")
    print("=" * 70)
    print("""
    总结:
    • KNN是一种直观、有效的分类/回归算法
    • 核心思想：物以类聚，根据邻居的类别推断新样本
    • 关键参数：K值、距离度量、权重策略
    • 历史意义：Cover & Hart (1967) 的理论证明奠定了其基础
    
    下一步:
    阅读 chapter-06-knn.md 获取完整理论讲解！
    """)
