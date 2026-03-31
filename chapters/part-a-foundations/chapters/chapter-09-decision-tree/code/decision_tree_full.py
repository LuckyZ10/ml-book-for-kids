"""
决策树算法完整实现
包含：ID3、C4.5、CART三种算法
作者：《机器学习与深度学习：从小学生到大师》
章节：第九章 决策树——像专家一样做决策
"""

import math
from collections import Counter
from itertools import combinations


class Node:
    """
    决策树节点类
    
    属性:
        feature: 分裂特征的名称（叶节点为None）
        threshold: 连续特征的分裂阈值（离散特征为None）
        children: 子节点字典 {特征值: 子节点}
        label: 叶节点的预测类别（内部节点为None）
        info: 节点的额外信息（用于可视化）
    """
    
    def __init__(self, feature=None, threshold=None, label=None):
        self.feature = feature      # 分裂特征
        self.threshold = threshold  # 分裂阈值（连续特征）
        self.children = {}          # 子节点
        self.label = label          # 叶节点标签
        self.info = {}              # 额外信息
    
    def is_leaf(self):
        """判断是否为叶节点"""
        return self.label is not None
    
    def predict(self, sample):
        """
        对单个样本进行预测
        
        参数:
            sample: 字典，如 {'颜色': '红', '大小': '大'}
        返回:
            预测的类别
        """
        # 如果是叶节点，直接返回标签
        if self.is_leaf():
            return self.label
        
        # 获取当前样本在分裂特征上的值
        feature_value = sample.get(self.feature)
        
        # 连续特征：比较阈值
        if self.threshold is not None:
            if feature_value <= self.threshold:
                key = '<='
            else:
                key = '>'
        else:
            # 离散特征：直接匹配
            key = feature_value
        
        # 递归预测
        if key in self.children:
            return self.children[key].predict(sample)
        else:
            # 如果该取值未在训练集中出现，返回多数类
            return self.label


class DecisionTreeClassifier:
    """
    决策树分类器
    
    支持三种算法：
        - 'id3': 使用信息增益
        - 'c4.5': 使用信息增益比
        - 'cart': 使用基尼指数，二叉树
    
    参数:
        algorithm: 算法类型 ('id3', 'c4.5', 'cart')
        max_depth: 最大深度（预剪枝）
        min_samples_split: 最小分裂样本数（预剪枝）
        min_impurity_decrease: 最小不纯度减少量（预剪枝）
    """
    
    def __init__(self, algorithm='id3', max_depth=None, 
                 min_samples_split=2, min_impurity_decrease=0.0):
        self.algorithm = algorithm.lower()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None
        self.classes_ = None
        self.n_features_ = None
    
    # ==================== 核心数学方法 ====================
    
    @staticmethod
    def entropy(labels):
        """
        计算香农熵
        
        公式: H(X) = -Σ p_i * log2(p_i)
        
        参数:
            labels: 类别标签列表，如 ['是', '否', '是', '是']
        返回:
            熵值（0表示完全纯净，越大越混乱）
        """
        if not labels:
            return 0
        
        n = len(labels)
        counts = Counter(labels)
        
        h = 0
        for count in counts.values():
            p = count / n
            if p > 0:
                h -= p * math.log2(p)
        
        return h
    
    @staticmethod
    def gini(labels):
        """
        计算基尼指数
        
        公式: Gini(D) = 1 - Σ p_i^2
        
        参数:
            labels: 类别标签列表
        返回:
            基尼指数（0表示完全纯净，0.5表示最混乱的二分类）
        """
        if not labels:
            return 0
        
        n = len(labels)
        counts = Counter(labels)
        
        gini = 1
        for count in counts.values():
            p = count / n
            gini -= p * p
        
        return gini
    
    def information_gain(self, y_parent, y_children):
        """
        计算信息增益
        
        公式: IG = H(D) - Σ |D_v|/|D| * H(D_v)
        
        参数:
            y_parent: 父节点标签列表
            y_children: 子节点标签列表的列表
        返回:
            信息增益值
        """
        n_parent = len(y_parent)
        if n_parent == 0:
            return 0
        
        # 父节点熵
        h_parent = self.entropy(y_parent)
        
        # 子节点加权熵
        h_children = 0
        for y_child in y_children:
            weight = len(y_child) / n_parent
            h_children += weight * self.entropy(y_child)
        
        return h_parent - h_children
    
    def gain_ratio(self, y_parent, y_children):
        """
        计算信息增益比
        
        公式: GainRatio = IG / SplitInfo
              SplitInfo = -Σ |D_v|/|D| * log2(|D_v|/|D|)
        
        参数:
            y_parent: 父节点标签列表
            y_children: 子节点标签列表的列表
        返回:
            信息增益比值
        """
        ig = self.information_gain(y_parent, y_children)
        
        n_parent = len(y_parent)
        if n_parent == 0:
            return 0
        
        # 计算分裂信息
        split_info = 0
        for y_child in y_children:
            p = len(y_child) / n_parent
            if p > 0:
                split_info -= p * math.log2(p)
        
        # 避免除零
        if split_info == 0:
            return 0
        
        return ig / split_info
    
    def gini_gain(self, y_parent, y_children):
        """
        计算基尼指数增益（CART使用）
        
        公式: Delta Gini = Gini(D) - Σ |D_v|/|D| * Gini(D_v)
        
        参数:
            y_parent: 父节点标签列表
            y_children: 子节点标签列表的列表
        返回:
            基尼指数减少量（越大越好）
        """
        n_parent = len(y_parent)
        if n_parent == 0:
            return 0
        
        # 父节点基尼指数
        gini_parent = self.gini(y_parent)
        
        # 子节点加权基尼指数
        gini_children = 0
        for y_child in y_children:
            weight = len(y_child) / n_parent
            gini_children += weight * self.gini(y_child)
        
        return gini_parent - gini_children
    
    # ==================== 特征分裂方法 ====================
    
    def _split_discrete(self, X, y, feature):
        """
        对离散特征进行分裂
        
        参数:
            X: 特征字典列表
            y: 标签列表
            feature: 特征名称
        返回:
            groups: {特征值: (X子集, y子集)}
        """
        groups = {}
        for xi, yi in zip(X, y):
            value = xi[feature]
            if value not in groups:
                groups[value] = ([], [])
            groups[value][0].append(xi)
            groups[value][1].append(yi)
        return groups
    
    def _split_continuous(self, X, y, feature):
        """
        对连续特征寻找最佳分裂阈值
        
        算法:
            1. 按特征值排序
            2. 在每对相邻的不同值之间取中点作为候选阈值
            3. 选择使分裂质量最优的阈值
        
        参数:
            X: 特征字典列表
            y: 标签列表
            feature: 特征名称
        返回:
            best_threshold: 最佳阈值
            best_groups: 分裂后的数据组 {'<=': (X_left, y_left), '>': (X_right, y_right)}
            best_quality: 分裂质量
        """
        # 按特征值排序
        sorted_pairs = sorted(zip(X, y), key=lambda pair: pair[0][feature])
        
        best_threshold = None
        best_quality = -float('inf')
        best_groups = None
        
        # 遍历可能的阈值（相邻不同值的中点）
        for i in range(len(sorted_pairs) - 1):
            value_i = sorted_pairs[i][0][feature]
            value_j = sorted_pairs[i + 1][0][feature]
            
            # 跳过相同值
            if value_i == value_j:
                continue
            
            # 取中点作为阈值
            threshold = (value_i + value_j) / 2
            
            # 分裂数据
            X_left, y_left = [], []
            X_right, y_right = [], []
            
            for xi, yi in sorted_pairs:
                if xi[feature] <= threshold:
                    X_left.append(xi)
                    y_left.append(yi)
                else:
                    X_right.append(xi)
                    y_right.append(yi)
            
            # 计算分裂质量
            if self.algorithm == 'cart':
                quality = self.gini_gain(y, [y_left, y_right])
            elif self.algorithm == 'c4.5':
                quality = self.gain_ratio(y, [y_left, y_right])
            else:  # id3
                quality = self.information_gain(y, [y_left, y_right])
            
            if quality > best_quality:
                best_quality = quality
                best_threshold = threshold
                best_groups = {
                    '<=': (X_left, y_left),
                    '>': (X_right, y_right)
                }
        
        return best_threshold, best_groups, best_quality
    
    def _find_best_split(self, X, y, features):
        """
        寻找最佳分裂特征和阈值
        
        参数:
            X: 特征字典列表
            y: 标签列表
            features: 可用特征列表
        返回:
            best_feature: 最佳特征
            best_threshold: 最佳阈值（连续特征）
            best_groups: 分裂后的数据组
            best_quality: 分裂质量
        """
        best_feature = None
        best_threshold = None
        best_groups = None
        best_quality = -float('inf')
        
        # 检测特征类型（简单启发式：第一个样本的值是数字就是连续特征）
        continuous_features = set()
        if X:
            for f in features:
                if isinstance(X[0][f], (int, float)):
                    continuous_features.add(f)
        
        for feature in features:
            if feature in continuous_features:
                # 连续特征
                threshold, groups, quality = self._split_continuous(X, y, feature)
            else:
                # 离散特征
                groups = self._split_discrete(X, y, feature)
                threshold = None
                
                # 计算分裂质量
                y_children = [group[1] for group in groups.values()]
                
                if self.algorithm == 'cart':
                    # CART需要将多值离散特征转化为二分
                    quality = self._best_binary_split_quality(y, groups)
                elif self.algorithm == 'c4.5':
                    quality = self.gain_ratio(y, y_children)
                else:  # id3
                    quality = self.information_gain(y, y_children)
            
            if quality > best_quality:
                best_quality = quality
                best_feature = feature
                best_threshold = threshold
                best_groups = groups
        
        return best_feature, best_threshold, best_groups, best_quality
    
    def _best_binary_split_quality(self, y, groups):
        """
        CART算法：寻找离散特征的最佳二分方式
        
        对于k个取值的特征，尝试所有2^(k-1)-1种二分方式
        返回最优的基尼指数增益
        """
        values = list(groups.keys())
        n = len(values)
        
        if n <= 1:
            return 0
        
        best_gain = -float('inf')
        
        # 生成所有可能的二分方式（使用位运算）
        # 对于n个取值，有2^(n-1)-1种非空真子集
        for r in range(1, n):
            for left_values in combinations(values, r):
                left_values = set(left_values)
                
                y_left = []
                y_right = []
                
                for value, (_, y_group) in groups.items():
                    if value in left_values:
                        y_left.extend(y_group)
                    else:
                        y_right.extend(y_group)
                
                gain = self.gini_gain(y, [y_left, y_right])
                best_gain = max(best_gain, gain)
        
        return best_gain
    
    # ==================== 递归建树 ====================
    
    def _build_tree(self, X, y, features, depth=0):
        """
        递归构建决策树（ID3/C4.5/CART核心算法）
        
        参数:
            X: 特征字典列表
            y: 标签列表
            features: 可用特征列表
            depth: 当前深度
        返回:
            构建好的节点
        """
        # 基本情况1：所有样本属于同一类别
        if len(set(y)) == 1:
            return Node(label=y[0])
        
        # 基本情况2：没有可用特征
        if not features:
            # 返回多数类
            majority = Counter(y).most_common(1)[0][0]
            return Node(label=majority)
        
        # 基本情况3：预剪枝 - 最大深度
        if self.max_depth is not None and depth >= self.max_depth:
            majority = Counter(y).most_common(1)[0][0]
            return Node(label=majority)
        
        # 基本情况4：预剪枝 - 最小样本数
        if len(y) < self.min_samples_split:
            majority = Counter(y).most_common(1)[0][0]
            return Node(label=majority)
        
        # 基本情况5：预剪枝 - 最小不纯度减少
        if len(set(y)) == 0:
            return Node(label=y[0] if y else None)
        
        # 寻找最佳分裂
        best_feature, best_threshold, best_groups, best_quality = \
            self._find_best_split(X, y, features)
        
        # 如果没有找到有效的分裂
        if best_feature is None or best_groups is None:
            majority = Counter(y).most_common(1)[0][0]
            return Node(label=majority)
        
        # 预剪枝：检查最小不纯度减少
        if best_quality < self.min_impurity_decrease:
            majority = Counter(y).most_common(1)[0][0]
            return Node(label=majority)
        
        # 创建节点
        node = Node(feature=best_feature, threshold=best_threshold)
        node.info['quality'] = best_quality
        node.info['n_samples'] = len(y)
        node.info['depth'] = depth
        
        # 递归构建子树
        new_features = [f for f in features if f != best_feature]
        
        for key, (X_child, y_child) in best_groups.items():
            if not X_child:  # 空子集
                majority = Counter(y).most_common(1)[0][0]
                node.children[key] = Node(label=majority)
            else:
                # 对于CART，离散特征可以重复使用（不同分裂方式）
                if self.algorithm == 'cart' and best_threshold is None:
                    # CART的离散特征二分，可以递归时使用所有特征
                    child_features = features
                else:
                    child_features = new_features
                
                node.children[key] = self._build_tree(
                    X_child, y_child, child_features, depth + 1
                )
        
        return node
    
    # ==================== 公开API ====================
    
    def fit(self, X, y):
        """
        训练决策树
        
        参数:
            X: 特征字典列表，如 [{'颜色': '红', '大小': 5}, ...]
            y: 标签列表，如 ['苹果', '橙子', ...]
        返回:
            self
        """
        if not X or not y:
            raise ValueError("训练数据不能为空")
        
        if len(X) != len(y):
            raise ValueError("X和y长度必须相同")
        
        # 记录类别
        self.classes_ = list(set(y))
        
        # 获取所有特征
        features = list(X[0].keys())
        self.n_features_ = len(features)
        
        # 构建树
        self.root = self._build_tree(X, y, features)
        
        return self
    
    def predict(self, X):
        """
        预测多个样本
        
        参数:
            X: 特征字典列表
        返回:
            预测标签列表
        """
        if self.root is None:
            raise ValueError("模型尚未训练，请先调用fit()")
        
        return [self.root.predict(xi) for xi in X]
    
    def predict_single(self, x):
        """
        预测单个样本
        
        参数:
            x: 特征字典
        返回:
            预测标签
        """
        if self.root is None:
            raise ValueError("模型尚未训练，请先调用fit()")
        
        return self.root.predict(x)
    
    def score(self, X, y):
        """
        计算准确率
        
        参数:
            X: 特征字典列表
            y: 真实标签列表
        返回:
            准确率（0-1之间）
        """
        predictions = self.predict(X)
        correct = sum(1 for pred, true in zip(predictions, y) if pred == true)
        return correct / len(y)
    
    def get_depth(self):
        """获取树的深度"""
        if self.root is None:
            return 0
        
        def _depth(node):
            if node.is_leaf():
                return 1
            return 1 + max(_depth(child) for child in node.children.values())
        
        return _depth(self.root)
    
    def get_n_leaves(self):
        """获取叶节点数量"""
        if self.root is None:
            return 0
        
        def _count(node):
            if node.is_leaf():
                return 1
            return sum(_count(child) for child in node.children.values())
        
        return _count(self.root)


# ==================== 可视化工具 ====================

def visualize_tree(node, indent="", last=True):
    """
    ASCII艺术可视化决策树
    
    参数:
        node: 节点
        indent: 当前缩进
        last: 是否是最后一个子节点
    """
    marker = "+-- " if last else "+-- "
    
    if node.is_leaf():
        print(f"{indent}{marker}[LEAF] 预测: {node.label}")
    else:
        if node.threshold is not None:
            print(f"{indent}{marker}[NODE] {node.feature} <= {node.threshold:.2f}?")
        else:
            print(f"{indent}{marker}[NODE] {node.feature}?")
        
        indent += "    " if last else "|   "
        
        items = list(node.children.items())
        for i, (key, child) in enumerate(items):
            is_last = (i == len(items) - 1)
            branch_marker = "+-- " if is_last else "+-- "
            print(f"{indent}{branch_marker}[{key}]")
            visualize_tree(child, indent + ("    " if is_last else "|   "), True)


def print_tree_rules(node, path="", rules=None):
    """
    打印决策树的规则形式
    
    参数:
        node: 当前节点
        path: 当前路径条件
        rules: 存储规则的列表
    返回:
        规则列表
    """
    if rules is None:
        rules = []
    
    if node.is_leaf():
        rules.append(f"IF {path} THEN 预测 = {node.label}")
    else:
        for key, child in node.children.items():
            if node.threshold is not None:
                condition = f"{node.feature} {key} {node.threshold:.2f}"
            else:
                condition = f"{node.feature} = {key}"
            
            new_path = f"{path} AND {condition}" if path else condition
            print_tree_rules(child, new_path, rules)
    
    return rules


# ==================== 实战案例 ====================

def fruit_classification_demo():
    """
    水果分类实战演示
    
    特征：
        - 颜色：青、半黄、金黄
        - 大小：小(1-3)、中(4-6)、大(7-10)
        - 重量：轻(<100g)、中(100-200g)、重(>200g)
    类别：苹果、香蕉、橙子
    """
    print("=" * 60)
    print("水果分类实战 - 决策树演示")
    print("=" * 60)
    
    # 训练数据
    X_train = [
        {'颜色': '青', '大小': 2, '重量': 80},
        {'颜色': '青', '大小': 3, '重量': 90},
        {'颜色': '半黄', '大小': 4, '重量': 120},
        {'颜色': '半黄', '大小': 5, '重量': 150},
        {'颜色': '金黄', '大小': 6, '重量': 180},
        {'颜色': '金黄', '大小': 7, '重量': 200},
        {'颜色': '青', '大小': 8, '重量': 220},
        {'颜色': '青', '大小': 9, '重量': 250},
        {'颜色': '金黄', '大小': 3, '重量': 85},
        {'颜色': '金黄', '大小': 4, '重量': 95},
        {'颜色': '半黄', '大小': 7, '重量': 210},
        {'颜色': '半黄', '大小': 8, '重量': 230},
    ]
    
    y_train = [
        '苹果', '苹果',  # 青苹果
        '苹果', '苹果',  # 半熟苹果
        '香蕉', '香蕉',  # 黄香蕉
        '香蕉', '香蕉',  # 大香蕉（青但大）
        '橙子', '橙子',  # 小橙子
        '橙子', '橙子',  # 大橙子
    ]
    
    # 测试数据
    X_test = [
        {'颜色': '青', '大小': 2, '重量': 85},    # 应该是苹果
        {'颜色': '金黄', '大小': 7, '重量': 220}, # 应该是香蕉
        {'颜色': '金黄', '大小': 3, '重量': 90},  # 应该是橙子
    ]
    y_test = ['苹果', '香蕉', '橙子']
    
    # 使用ID3算法
    print("\n使用ID3算法（信息增益）：")
    clf_id3 = DecisionTreeClassifier(algorithm='id3', max_depth=3)
    clf_id3.fit(X_train, y_train)
    
    print(f"树深度: {clf_id3.get_depth()}")
    print(f"叶节点数: {clf_id3.get_n_leaves()}")
    print("\n决策树结构：")
    visualize_tree(clf_id3.root)
    
    print("\n预测结果：")
    predictions = clf_id3.predict(X_test)
    for i, (pred, true) in enumerate(zip(predictions, y_test)):
        status = "正确" if pred == true else "错误"
        print(f"  样本{i+1}: 预测={pred}, 真实={true} [{status}]")
    
    accuracy = clf_id3.score(X_test, y_test)
    print(f"\n准确率: {accuracy:.1%}")
    
    # 使用CART算法
    print("\n" + "=" * 60)
    print("使用CART算法（基尼指数）：")
    clf_cart = DecisionTreeClassifier(algorithm='cart', max_depth=3)
    clf_cart.fit(X_train, y_train)
    
    print(f"树深度: {clf_cart.get_depth()}")
    print(f"叶节点数: {clf_cart.get_n_leaves()}")
    print("\n决策树结构：")
    visualize_tree(clf_cart.root)
    
    accuracy_cart = clf_cart.score(X_test, y_test)
    print(f"\n准确率: {accuracy_cart:.1%}")


def loan_approval_demo():
    """
    贷款审批决策树演示
    """
    print("\n" + "=" * 60)
    print("贷款审批决策树演示")
    print("=" * 60)
    
    X_train = [
        {'收入': '高', '信用': '优', '有房': '是'},
        {'收入': '高', '信用': '优', '有房': '否'},
        {'收入': '中', '信用': '优', '有房': '是'},
        {'收入': '低', '信用': '良', '有房': '是'},
        {'收入': '低', '信用': '良', '有房': '否'},
        {'收入': '中', '信用': '良', '有房': '是'},
        {'收入': '高', '信用': '良', '有房': '是'},
        {'收入': '高', '信用': '良', '有房': '否'},
        {'收入': '低', '信用': '优', '有房': '否'},
        {'收入': '中', '信用': '优', '有房': '否'},
    ]
    
    y_train = ['通过', '通过', '通过', '通过', '拒绝', 
               '通过', '通过', '拒绝', '拒绝', '通过']
    
    clf = DecisionTreeClassifier(algorithm='c4.5')
    clf.fit(X_train, y_train)
    
    print("\n决策树结构：")
    visualize_tree(clf.root)
    
    print("\n决策规则：")
    rules = print_tree_rules(clf.root)
    for i, rule in enumerate(rules, 1):
        print(f"  规则{i}: {rule}")
    
    # 新申请预测
    print("\n新申请预测：")
    new_applications = [
        {'收入': '高', '信用': '优', '有房': '是'},
        {'收入': '低', '信用': '良', '有房': '否'},
        {'收入': '中', '信用': '良', '有房': '否'},
    ]
    
    for i, app in enumerate(new_applications, 1):
        prediction = clf.predict_single(app)
        status = "通过" if prediction == '通过' else "拒绝"
        print(f"  申请{i} ({app}): {status}")


def entropy_demo():
    """
    熵和基尼指数演示
    """
    print("\n" + "=" * 60)
    print("熵与基尼指数计算演示")
    print("=" * 60)
    
    # 测试不同分布的熵和基尼指数
    distributions = [
        ([1, 0], "100% 正类"),
        ([0.9, 0.1], "90% 正类, 10% 负类"),
        ([0.8, 0.2], "80% 正类, 20% 负类"),
        ([0.7, 0.3], "70% 正类, 30% 负类"),
        ([0.6, 0.4], "60% 正类, 40% 负类"),
        ([0.5, 0.5], "50% 正类, 50% 负类"),
    ]
    
    print("\n分布\t\t\t\t熵\t\t基尼指数")
    print("-" * 60)
    
    for probs, desc in distributions:
        # 生成标签列表
        n = 100
        labels = ['正'] * int(probs[0] * n) + ['负'] * int(probs[1] * n)
        
        h = DecisionTreeClassifier.entropy(labels)
        g = DecisionTreeClassifier.gini(labels)
        
        print(f"{desc}\t\t{h:.4f}\t\t{g:.4f}")
    
    print("\n结论：")
    print("- 当两类均匀分布时，熵和基尼指数都达到最大值")
    print("- 当数据完全纯净时，熵和基尼指数都为0")


if __name__ == "__main__":
    # 运行演示
    entropy_demo()
    fruit_classification_demo()
    loan_approval_demo()
    
    print("\n" + "=" * 60)
    print("决策树算法演示完成！")
    print("=" * 60)
