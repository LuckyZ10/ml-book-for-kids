"""
第九章：决策树——像专家一样做决策
配套代码实现
================

本文件实现：
1. ID3算法 (信息增益)
2. CART算法 (基尼指数)
3. 决策树节点类
4. 预剪枝与后剪枝
5. 实战案例：天气预测、水果分类、贷款审批

作者: ML教材写作项目
日期: 2026-03-24
"""

import math
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple, Union


# ============================================================================
# 数学工具函数
# ============================================================================

def entropy(class_counts: Dict[Any, int], total: int) -> float:
    """
    计算香农熵
    
    H(S) = -Σ pᵢ × log₂(pᵢ)
    
    Args:
        class_counts: 每个类别的样本数 {类别: 数量}
        total: 总样本数
    
    Returns:
        熵值 (bits)
    """
    if total == 0:
        return 0.0
    
    ent = 0.0
    for count in class_counts.values():
        if count > 0:
            p = count / total
            ent -= p * math.log2(p)
    return ent


def gini_impurity(class_counts: Dict[Any, int], total: int) -> float:
    """
    计算基尼不纯度
    
    Gini(S) = 1 - Σ pᵢ²
    
    Args:
        class_counts: 每个类别的样本数
        total: 总样本数
    
    Returns:
        基尼指数 [0, 1)，0表示纯节点
    """
    if total == 0:
        return 0.0
    
    gini = 1.0
    for count in class_counts.values():
        p = count / total
        gini -= p * p
    return gini


def information_gain(
    parent_counts: Dict[Any, int],
    child_counts_list: List[Dict[Any, int]],
    child_sizes: List[int]
) -> float:
    """
    计算信息增益
    
    IG(S, A) = H(S) - Σ (|Sᵥ|/|S|) × H(Sᵥ)
    
    Args:
        parent_counts: 父节点各类别数量
        child_counts_list: 各子节点各类别数量列表
        child_sizes: 各子节点样本数列表
    
    Returns:
        信息增益
    """
    total = sum(parent_counts.values())
    parent_entropy = entropy(parent_counts, total)
    
    weighted_child_entropy = 0.0
    for child_counts, child_size in zip(child_counts_list, child_sizes):
        weight = child_size / total
        child_ent = entropy(child_counts, child_size)
        weighted_child_entropy += weight * child_ent
    
    return parent_entropy - weighted_child_entropy


def split_information(child_sizes: List[int], total: int) -> float:
    """
    计算分裂信息 (C4.5的信息增益率用)
    
    SplitInfo = -Σ |Dⱼ|/|D| × log₂(|Dⱼ|/|D|)
    """
    split_info = 0.0
    for size in child_sizes:
        if size > 0:
            p = size / total
            split_info -= p * math.log2(p)
    return split_info


# ============================================================================
# 决策树节点类
# ============================================================================

class DecisionTreeNode:
    """决策树节点"""
    
    def __init__(self):
        self.feature: Optional[str] = None          # 分裂特征
        self.threshold: Optional[Any] = None        # 分裂阈值(连续特征)
        self.value: Optional[Any] = None            # 叶子节点的预测值
        self.children: Dict[Any, 'DecisionTreeNode'] = {}  # 子节点
        self.left: Optional['DecisionTreeNode'] = None     # CART左子树(是/≤)
        self.right: Optional['DecisionTreeNode'] = None    # CART右子树(否/>)
        self.is_leaf: bool = False                  # 是否为叶子
        self.class_counts: Dict[Any, int] = {}      # 类别统计
        self.total_samples: int = 0                 # 样本数
        self.depth: int = 0                         # 节点深度
    
    def predict(self, sample: Dict[str, Any]) -> Any:
        """
        对单个样本进行预测
        
        Args:
            sample: 样本字典 {特征名: 值}
        
        Returns:
            预测类别
        """
        if self.is_leaf:
            return self.value
        
        # CART二叉树
        if self.left is not None and self.right is not None:
            feature_value = sample.get(self.feature)
            if feature_value is None:
                # 处理缺失值：返回多数类
                return max(self.class_counts, key=self.class_counts.get)
            
            if isinstance(self.threshold, (int, float)):
                # 连续特征
                if feature_value <= self.threshold:
                    return self.left.predict(sample)
                else:
                    return self.right.predict(sample)
            else:
                # 类别特征
                if feature_value == self.threshold:
                    return self.left.predict(sample)
                else:
                    return self.right.predict(sample)
        
        # ID3多叉树
        feature_value = sample.get(self.feature)
        if feature_value in self.children:
            return self.children[feature_value].predict(sample)
        else:
            # 未见过的值，返回多数类
            return max(self.class_counts, key=self.class_counts.get)
    
    def __str__(self) -> str:
        """字符串表示"""
        if self.is_leaf:
            return f"Leaf({self.value}, n={self.total_samples})"
        else:
            return f"Node({self.feature}, n={self.total_samples})"


# ============================================================================
# ID3决策树实现
# ============================================================================

class ID3DecisionTree:
    """
    ID3决策树分类器
    
    基于信息增益进行特征选择，支持多叉树
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_gain: float = 0.0
    ):
        """
        初始化ID3决策树
        
        Args:
            max_depth: 最大深度(None表示无限制)
            min_samples_split: 内部节点最小样本数
            min_samples_leaf: 叶子节点最小样本数
            min_gain: 最小信息增益阈值
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.root: Optional[DecisionTreeNode] = None
        self.feature_names: List[str] = []
        self.target_name: str = ""
    
    def fit(
        self,
        X: List[Dict[str, Any]],
        y: List[Any],
        feature_names: Optional[List[str]] = None
    ) -> 'ID3DecisionTree':
        """
        训练决策树
        
        Args:
            X: 特征列表，每个元素是 {特征名: 值} 字典
            y: 标签列表
            feature_names: 特征名列表(可选)
        
        Returns:
            self
        """
        if len(X) == 0:
            raise ValueError("训练数据为空")
        
        # 自动提取特征名
        if feature_names is None and len(X) > 0:
            feature_names = list(X[0].keys())
        self.feature_names = feature_names or []
        
        # 构建树
        self.root = self._build_tree(X, y, self.feature_names, depth=0)
        return self
    
    def _build_tree(
        self,
        X: List[Dict[str, Any]],
        y: List[Any],
        features: List[str],
        depth: int
    ) -> DecisionTreeNode:
        """递归构建决策树"""
        
        node = DecisionTreeNode()
        node.depth = depth
        node.total_samples = len(y)
        node.class_counts = dict(Counter(y))
        
        # 停止条件1：所有样本属于同一类
        if len(set(y)) == 1:
            node.is_leaf = True
            node.value = y[0]
            return node
        
        # 停止条件2：没有可用特征
        if len(features) == 0:
            node.is_leaf = True
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        # 停止条件3：样本数不足
        if len(y) < self.min_samples_split:
            node.is_leaf = True
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        # 停止条件4：达到最大深度
        if self.max_depth is not None and depth >= self.max_depth:
            node.is_leaf = True
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        # 选择最优特征
        best_feature, best_gain = self._best_feature(X, y, features)
        
        # 停止条件5：信息增益太小
        if best_gain < self.min_gain:
            node.is_leaf = True
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        # 使用最优特征分裂
        node.feature = best_feature
        node.is_leaf = False
        
        # 按特征值分组
        feature_values = set(sample[best_feature] for sample in X)
        
        remaining_features = [f for f in features if f != best_feature]
        
        for value in feature_values:
            # 筛选子集
            sub_X = []
            sub_y = []
            for sample, label in zip(X, y):
                if sample[best_feature] == value:
                    sub_X.append(sample)
                    sub_y.append(label)
            
            # 检查最小叶子样本数
            if len(sub_y) < self.min_samples_leaf:
                # 不满足条件，设为叶子
                child = DecisionTreeNode()
                child.is_leaf = True
                child.value = Counter(y).most_common(1)[0][0]
                child.total_samples = len(sub_y)
                child.class_counts = dict(Counter(sub_y))
                child.depth = depth + 1
            else:
                # 递归构建子树
                child = self._build_tree(sub_X, sub_y, remaining_features, depth + 1)
            
            node.children[value] = child
        
        return node
    
    def _best_feature(
        self,
        X: List[Dict[str, Any]],
        y: List[Any],
        features: List[str]
    ) -> Tuple[str, float]:
        """
        选择最优分裂特征
        
        Returns:
            (最优特征名, 信息增益)
        """
        parent_counts = dict(Counter(y))
        
        best_feature = None
        best_gain = -float('inf')
        
        for feature in features:
            # 按特征值分组
            value_groups: Dict[Any, List[Any]] = {}
            for sample, label in zip(X, y):
                value = sample[feature]
                if value not in value_groups:
                    value_groups[value] = []
                value_groups[value].append(label)
            
            # 计算信息增益
            child_counts_list = [dict(Counter(group)) for group in value_groups.values()]
            child_sizes = [len(group) for group in value_groups.values()]
            
            gain = information_gain(parent_counts, child_counts_list, child_sizes)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        
        return best_feature, best_gain
    
    def predict(self, X: List[Dict[str, Any]]) -> List[Any]:
        """预测多个样本"""
        return [self.root.predict(sample) for sample in X]
    
    def score(self, X: List[Dict[str, Any]], y: List[Any]) -> float:
        """计算准确率"""
        predictions = self.predict(X)
        correct = sum(1 for p, t in zip(predictions, y) if p == t)
        return correct / len(y) if len(y) > 0 else 0.0
    
    def print_tree(self, node: Optional[DecisionTreeNode] = None, indent: str = ""):
        """打印决策树"""
        if node is None:
            node = self.root
        
        if node.is_leaf:
            print(f"{indent}└── 预测: {node.value} (样本数: {node.total_samples})")
        else:
            print(f"{indent}[{node.feature}?]")
            for value, child in node.children.items():
                print(f"{indent}  ├── ={value}:")
                self.print_tree(child, indent + "  │   ")


# ============================================================================
# CART决策树实现
# ============================================================================

class CARTDecisionTree:
    """
    CART决策树分类器
    
    基于基尼指数进行特征选择，二叉树结构
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0
    ):
        """
        初始化CART决策树
        
        Args:
            max_depth: 最大深度
            min_samples_split: 内部节点最小样本数
            min_samples_leaf: 叶子节点最小样本数
            min_impurity_decrease: 最小不纯度减少量
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.root: Optional[DecisionTreeNode] = None
        self.feature_names: List[str] = []
    
    def fit(
        self,
        X: List[Dict[str, Any]],
        y: List[Any],
        feature_names: Optional[List[str]] = None
    ) -> 'CARTDecisionTree':
        """训练决策树"""
        if len(X) == 0:
            raise ValueError("训练数据为空")
        
        if feature_names is None and len(X) > 0:
            feature_names = list(X[0].keys())
        self.feature_names = feature_names or []
        
        self.root = self._build_tree(X, y, self.feature_names, depth=0)
        return self
    
    def _build_tree(
        self,
        X: List[Dict[str, Any]],
        y: List[Any],
        features: List[str],
        depth: int
    ) -> DecisionTreeNode:
        """递归构建CART二叉树"""
        
        node = DecisionTreeNode()
        node.depth = depth
        node.total_samples = len(y)
        node.class_counts = dict(Counter(y))
        
        # 停止条件
        if len(set(y)) == 1:
            node.is_leaf = True
            node.value = y[0]
            return node
        
        if len(features) == 0 or len(y) < self.min_samples_split:
            node.is_leaf = True
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        if self.max_depth is not None and depth >= self.max_depth:
            node.is_leaf = True
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        # 寻找最优分裂
        best_feature, best_threshold, best_impurity = self._best_split(X, y, features)
        
        if best_feature is None or best_impurity < self.min_impurity_decrease:
            node.is_leaf = True
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        # 执行分裂
        node.feature = best_feature
        node.threshold = best_threshold
        node.is_leaf = False
        
        # 分裂数据
        left_X, left_y, right_X, right_y = self._split_data(X, y, best_feature, best_threshold)
        
        # 检查叶子样本数
        if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
            node.is_leaf = True
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        # 递归构建子树
        node.left = self._build_tree(left_X, left_y, features, depth + 1)
        node.right = self._build_tree(right_X, right_y, features, depth + 1)
        
        return node
    
    def _best_split(
        self,
        X: List[Dict[str, Any]],
        y: List[Any],
        features: List[str]
    ) -> Tuple[Optional[str], Optional[Any], float]:
        """
        寻找最优分裂点
        
        Returns:
            (最优特征, 最优阈值, 不纯度减少量)
        """
        parent_counts = dict(Counter(y))
        parent_gini = gini_impurity(parent_counts, len(y))
        
        best_feature = None
        best_threshold = None
        best_gain = 0.0
        
        for feature in features:
            feature_values = sorted(set(sample[feature] for sample in X))
            
            # 尝试所有可能的分裂点
            for i in range(len(feature_values) - 1):
                # 阈值取中间值
                if isinstance(feature_values[0], (int, float)):
                    threshold = (feature_values[i] + feature_values[i + 1]) / 2
                else:
                    threshold = feature_values[i]
                
                # 分裂
                left_y = [label for sample, label in zip(X, y) if sample[feature] <= threshold]
                right_y = [label for sample, label in zip(X, y) if sample[feature] > threshold]
                
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                
                # 计算加权基尼
                left_counts = dict(Counter(left_y))
                right_counts = dict(Counter(right_y))
                
                weighted_gini = (
                    len(left_y) / len(y) * gini_impurity(left_counts, len(left_y)) +
                    len(right_y) / len(y) * gini_impurity(right_counts, len(right_y))
                )
                
                gain = parent_gini - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _split_data(
        self,
        X: List[Dict[str, Any]],
        y: List[Any],
        feature: str,
        threshold: Any
    ) -> Tuple[List[Dict], List[Any], List[Dict], List[Any]]:
        """按阈值分裂数据"""
        left_X, left_y = [], []
        right_X, right_y = [], []
        
        for sample, label in zip(X, y):
            if isinstance(threshold, (int, float)):
                if sample[feature] <= threshold:
                    left_X.append(sample)
                    left_y.append(label)
                else:
                    right_X.append(sample)
                    right_y.append(label)
            else:
                if sample[feature] == threshold:
                    left_X.append(sample)
                    left_y.append(label)
                else:
                    right_X.append(sample)
                    right_y.append(label)
        
        return left_X, left_y, right_X, right_y
    
    def predict(self, X: List[Dict[str, Any]]) -> List[Any]:
        """预测"""
        return [self.root.predict(sample) for sample in X]
    
    def score(self, X: List[Dict[str, Any]], y: List[Any]) -> float:
        """准确率"""
        predictions = self.predict(X)
        correct = sum(1 for p, t in zip(predictions, y) if p == t)
        return correct / len(y) if len(y) > 0 else 0.0
    
    def print_tree(self, node: Optional[DecisionTreeNode] = None, indent: str = ""):
        """打印树结构"""
        if node is None:
            node = self.root
        
        if node.is_leaf:
            print(f"{indent}└── 预测: {node.value} (n={node.total_samples})")
        else:
            if isinstance(node.threshold, (int, float)):
                print(f"{indent}[{node.feature} ≤ {node.threshold:.2f}?]")
            else:
                print(f"{indent}[{node.feature} = {node.threshold}?]")
            print(f"{indent}  ├── 是:")
            self.print_tree(node.left, indent + "  │   ")
            print(f"{indent}  └── 否:")
            self.print_tree(node.right, indent + "      ")


# ============================================================================
# 可视化工具
# ============================================================================

def print_ascii_tree(
    node: DecisionTreeNode,
    indent: str = "",
    is_last: bool = True,
    prefix: str = ""
) -> str:
    """
    生成ASCII艺术树
    
    Args:
        node: 当前节点
        indent: 缩进字符串
        is_last: 是否是最后一个子节点
        prefix: 节点标签前缀
    
    Returns:
        ASCII树字符串
    """
    lines = []
    
    # 当前节点行
    connector = "└── " if is_last else "├── "
    if node.is_leaf:
        lines.append(f"{indent}{connector}{prefix}预测: {node.value} (n={node.total_samples})")
    else:
        if node.threshold is not None:
            if isinstance(node.threshold, (int, float)):
                lines.append(f"{indent}{connector}{prefix}[{node.feature} ≤ {node.threshold:.2f}]")
            else:
                lines.append(f"{indent}{connector}{prefix}[{node.feature} = {node.threshold}]")
        else:
            lines.append(f"{indent}{connector}{prefix}[{node.feature}?]")
    
    # 子节点
    if not node.is_leaf:
        children = []
        if node.left and node.right:
            children = [("是", node.left), ("否", node.right)]
        else:
            children = [(str(k), v) for k, v in node.children.items()]
        
        new_indent = indent + ("    " if is_last else "│   ")
        
        for i, (label, child) in enumerate(children):
            is_last_child = (i == len(children) - 1)
            lines.extend(print_ascii_tree(child, new_indent, is_last_child, f"{label}: ").split('\n'))
    
    return '\n'.join(lines)


# ============================================================================
# 实战案例
# ============================================================================

def demo_weather():
    """天气打球预测演示"""
    print("=" * 60)
    print("🌤️  实战案例1：天气与打球预测")
    print("=" * 60)
    
    # 天气数据集
    data = [
        ({"天气": "晴", "温度": "热", "湿度": "高", "windy": "否"}, "否"),
        ({"天气": "晴", "温度": "热", "湿度": "高", "windy": "是"}, "否"),
        ({"天气": "阴", "温度": "热", "湿度": "高", "windy": "否"}, "是"),
        ({"天气": "雨", "温度": "温和", "湿度": "高", "windy": "否"}, "是"),
        ({"天气": "雨", "温度": "冷", "湿度": "正常", "windy": "否"}, "是"),
        ({"天气": "雨", "温度": "冷", "湿度": "正常", "windy": "是"}, "否"),
        ({"天气": "阴", "温度": "冷", "湿度": "正常", "windy": "是"}, "是"),
        ({"天气": "晴", "温度": "温和", "湿度": "高", "windy": "否"}, "否"),
        ({"天气": "晴", "温度": "冷", "湿度": "正常", "windy": "否"}, "是"),
        ({"天气": "雨", "温度": "温和", "湿度": "正常", "windy": "否"}, "是"),
        ({"天气": "晴", "温度": "温和", "湿度": "正常", "windy": "是"}, "是"),
        ({"天气": "阴", "温度": "温和", "湿度": "高", "windy": "是"}, "是"),
        ({"天气": "阴", "温度": "热", "湿度": "正常", "windy": "否"}, "是"),
        ({"天气": "雨", "温度": "温和", "湿度": "高", "windy": "是"}, "否"),
    ]
    
    X = [d[0] for d in data]
    y = [d[1] for d in data]
    
    print(f"\n📊 数据集: {len(X)} 条记录")
    print(f"   特征: 天气, 温度, 湿度, windy")
    print(f"   标签: 是否适合打球")
    
    # ID3树
    print("\n" + "-" * 40)
    print("🌳 ID3决策树 (信息增益):")
    print("-" * 40)
    id3_tree = ID3DecisionTree(max_depth=3)
    id3_tree.fit(X, y)
    id3_tree.print_tree()
    
    # 预测测试
    test_samples = [
        {"天气": "晴", "温度": "温和", "湿度": "正常", "windy": "否"},
        {"天气": "雨", "温度": "冷", "湿度": "高", "windy": "是"},
    ]
    print("\n🎯 预测测试:")
    for sample in test_samples:
        pred = id3_tree.root.predict(sample)
        print(f"   {sample} → {pred}")
    
    # CART树
    print("\n" + "-" * 40)
    print("🌳 CART决策树 (基尼指数):")
    print("-" * 40)
    # 需要将类别特征编码为数值
    weather_map = {"晴": 1, "阴": 2, "雨": 3}
    temp_map = {"热": 3, "温和": 2, "冷": 1}
    humid_map = {"高": 2, "正常": 1}
    windy_map = {"否": 0, "是": 1}
    
    X_numeric = [
        {"天气": weather_map[d[0]["天气"]], "温度": temp_map[d[0]["温度"]], 
         "湿度": humid_map[d[0]["湿度"]], "windy": windy_map[d[0]["windy"]]}
        for d in data
    ]
    
    cart_tree = CARTDecisionTree(max_depth=3)
    cart_tree.fit(X_numeric, y)
    cart_tree.print_tree()


def demo_fruit():
    """水果分类演示"""
    print("\n" + "=" * 60)
    print("🍎 实战案例2：水果分类")
    print("=" * 60)
    
    # 水果数据集
    data = [
        ({"颜色": "红", "直径": 8, "重量": 150}, "苹果"),
        ({"颜色": "红", "直径": 7, "重量": 140}, "苹果"),
        ({"颜色": "绿", "直径": 7, "重量": 130}, "苹果"),
        ({"颜色": "黄", "直径": 7, "重量": 145}, "苹果"),
        ({"颜色": "橙", "直径": 9, "重量": 200}, "橙子"),
        ({"颜色": "橙", "直径": 8, "重量": 180}, "橙子"),
        ({"颜色": "橙", "直径": 10, "重量": 220}, "橙子"),
        ({"颜色": "紫", "直径": 3, "重量": 10}, "葡萄"),
        ({"颜色": "紫", "直径": 2, "重量": 8}, "葡萄"),
        ({"颜色": "绿", "直径": 3, "重量": 12}, "葡萄"),
        ({"颜色": "黄", "直径": 12, "重量": 300}, "西瓜"),
        ({"颜色": "绿", "直径": 15, "重量": 500}, "西瓜"),
    ]
    
    X = [d[0] for d in data]
    y = [d[1] for d in data]
    
    print(f"\n📊 数据集: {len(X)} 条记录")
    print(f"   特征: 颜色, 直径, 重量")
    print(f"   类别: 苹果, 橙子, 葡萄, 西瓜")
    
    # 使用CART（支持连续特征）
    print("\n" + "-" * 40)
    print("🌳 CART决策树:")
    print("-" * 40)
    cart_tree = CARTDecisionTree(max_depth=4, min_samples_leaf=1)
    cart_tree.fit(X, y)
    cart_tree.print_tree()
    
    # 预测
    print("\n🎯 预测新水果:")
    test_fruits = [
        {"颜色": "红", "直径": 8, "重量": 155},   # 应该是苹果
        {"颜色": "橙", "直径": 9, "重量": 210},   # 应该是橙子
        {"颜色": "紫", "直径": 2, "重量": 9},     # 应该是葡萄
    ]
    
    for fruit in test_fruits:
        pred = cart_tree.root.predict(fruit)
        print(f"   颜色={fruit['颜色']}, 直径={fruit['直径']}cm, 重量={fruit['重量']}g → {pred}")


def demo_loan():
    """贷款审批演示"""
    print("\n" + "=" * 60)
    print("💰 实战案例3：贷款审批")
    print("=" * 60)
    
    # 贷款数据集
    data = [
        ({"年龄": 25, "收入": 3, "学历": "高中", "有工作": "否"}, "拒绝"),
        ({"年龄": 35, "收入": 8, "学历": "大学", "有工作": "是"}, "批准"),
        ({"年龄": 28, "收入": 6, "学历": "大学", "有工作": "是"}, "批准"),
        ({"年龄": 45, "收入": 4, "学历": "高中", "有工作": "是"}, "拒绝"),
        ({"年龄": 32, "收入": 7, "学历": "硕士", "有工作": "是"}, "批准"),
        ({"年龄": 22, "收入": 2, "学历": "高中", "有工作": "否"}, "拒绝"),
        ({"年龄": 38, "收入": 9, "学历": "大学", "有工作": "是"}, "批准"),
        ({"年龄": 29, "收入": 5, "学历": "大学", "有工作": "否"}, "拒绝"),
        ({"年龄": 41, "收入": 10, "学历": "硕士", "有工作": "是"}, "批准"),
        ({"年龄": 26, "收入": 4, "学历": "高中", "有工作": "是"}, "拒绝"),
    ]
    
    X = [d[0] for d in data]
    y = [d[1] for d in data]
    
    print(f"\n📊 数据集: {len(X)} 条记录")
    print(f"   特征: 年龄, 收入(万), 学历, 有工作")
    print(f"   标签: 批准/拒绝")
    
    # ID3处理离散特征，CART处理连续特征
    # 这里用ID3处理类别特征，CART处理数值特征
    
    print("\n" + "-" * 40)
    print("🌳 ID3决策树 (基于学历、有工作):")
    print("-" * 40)
    
    X_id3 = [{"学历": d[0]["学历"], "有工作": d[0]["有工作"]} for d in data]
    id3_tree = ID3DecisionTree(max_depth=3)
    id3_tree.fit(X_id3, y)
    id3_tree.print_tree()
    
    print("\n" + "-" * 40)
    print("🌳 CART决策树 (基于年龄、收入):")
    print("-" * 40)
    
    X_cart = [{"年龄": d[0]["年龄"], "收入": d[0]["收入"]} for d in data]
    cart_tree = CARTDecisionTree(max_depth=3)
    cart_tree.fit(X_cart, y)
    cart_tree.print_tree()
    
    # 预测新申请
    print("\n🎯 新贷款申请预测:")
    new_applications = [
        {"年龄": 30, "收入": 7, "学历": "大学", "有工作": "是"},
        {"年龄": 24, "收入": 3, "学历": "高中", "有工作": "否"},
    ]
    
    for app in new_applications:
        pred_id3 = id3_tree.root.predict({"学历": app["学历"], "有工作": app["有工作"]})
        pred_cart = cart_tree.root.predict({"年龄": app["年龄"], "收入": app["收入"]})
        print(f"   年龄{app['年龄']}岁, 收入{app['收入']}万, {app['学历']}, 有工作:{app['有工作']}")
        print(f"      ID3(学历/工作) → {pred_id3}")
        print(f"      CART(年龄/收入) → {pred_cart}")


def demo_math():
    """数学计算演示"""
    print("\n" + "=" * 60)
    print("🧮 数学计算演示")
    print("=" * 60)
    
    # 熵的计算
    print("\n📐 熵的计算示例:")
    
    # 4红4蓝
    counts_4_4 = {"红": 4, "蓝": 4}
    ent_4_4 = entropy(counts_4_4, 8)
    print(f"   场景A (4红4蓝): H = {ent_4_4:.3f} bits")
    
    # 7红1蓝
    counts_7_1 = {"红": 7, "蓝": 1}
    ent_7_1 = entropy(counts_7_1, 8)
    print(f"   场景B (7红1蓝): H = {ent_7_1:.3f} bits")
    
    # 8红
    counts_8_0 = {"红": 8}
    ent_8_0 = entropy(counts_8_0, 8)
    print(f"   场景C (8红): H = {ent_8_0:.3f} bits")
    
    # 基尼指数
    print("\n📐 基尼指数计算示例:")
    gini_4_4 = gini_impurity(counts_4_4, 8)
    print(f"   场景A (4红4蓝): Gini = {gini_4_4:.3f}")
    
    gini_7_1 = gini_impurity(counts_7_1, 8)
    print(f"   场景B (7红1蓝): Gini = {gini_7_1:.3f}")
    
    gini_8_0 = gini_impurity(counts_8_0, 8)
    print(f"   场景C (8红): Gini = {gini_8_0:.3f}")
    
    # 信息增益
    print("\n📐 信息增益计算示例:")
    parent = {"是": 5, "否": 5}
    child1 = {"是": 5, "否": 0}  # 纯节点
    child2 = {"是": 0, "否": 5}  # 纯节点
    
    ig = information_gain(parent, [child1, child2], [5, 5])
    print(f"   父节点(5是5否) → 分裂为两个纯节点")
    print(f"   信息增益 = {ig:.3f} bits (最大!)")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     第九章：决策树——像专家一样做决策                      ║
    ║     Decision Tree Implementation                         ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # 运行所有演示
    demo_math()
    demo_weather()
    demo_fruit()
    demo_loan()
    
    print("\n" + "=" * 60)
    print("✅ 所有演示完成!")
    print("=" * 60)
    print("""
    📚 本代码实现了:
       • ID3算法 (信息增益, 多叉树)
       • CART算法 (基尼指数, 二叉树)
       • 熵、基尼指数、信息增益计算
       • 预剪枝 (max_depth, min_samples)
       • 实战案例: 天气预测、水果分类、贷款审批
    
    🎯 关键公式:
       • 熵: H(S) = -Σ pᵢ log₂(pᵢ)
       • 基尼: Gini(S) = 1 - Σ pᵢ²
       • 信息增益: IG = H(父) - Σ wᵢ H(子ᵢ)
    """)
