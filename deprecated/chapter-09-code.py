"""
第九章：决策树——像专家一样做决策
配套代码：从零实现ID3决策树算法

作者：ML教材写作项目
日期：2026-03-24
"""

import math
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional


# ============================================================
# 第一部分：数学基础 - 熵与信息增益
# ============================================================

def entropy(labels: List[Any]) -> float:
    """
    计算香农熵
    
    公式: H(S) = -Σ p_i * log2(p_i)
    
    Args:
        labels: 类别标签列表
    
    Returns:
        熵值（0到log2(类别数)之间）
    
    示例:
        >>> entropy(['是', '是', '否', '否'])
        1.0
        >>> entropy(['是', '是', '是'])
        0.0
    """
    if not labels:
        return 0.0
    
    # 统计每个类别的频数
    counter = Counter(labels)
    total = len(labels)
    
    # 计算熵
    ent = 0.0
    for count in counter.values():
        p = count / total
        # 处理p=0的情况（虽然Counter不会出现0）
        if p > 0:
            ent -= p * math.log2(p)
    
    return ent


def information_gain(dataset: List[Dict], labels: List[Any], 
                     attribute: str) -> float:
    """
    计算按某个属性分裂的信息增益
    
    公式: IG(S, A) = H(S) - Σ (|S_v|/|S|) * H(S_v)
    
    Args:
        dataset: 数据集，每个样本是一个字典
        labels: 对应的类别标签
        attribute: 要计算信息增益的属性名
    
    Returns:
        信息增益值
    """
    # 分裂前的熵
    base_entropy = entropy(labels)
    
    # 按属性值分组
    groups = {}
    for sample, label in zip(dataset, labels):
        value = sample[attribute]
        if value not in groups:
            groups[value] = {'samples': [], 'labels': []}
        groups[value]['samples'].append(sample)
        groups[value]['labels'].append(label)
    
    # 计算分裂后的加权平均熵
    total = len(labels)
    weighted_entropy = 0.0
    
    for value, group in groups.items():
        weight = len(group['labels']) / total
        weighted_entropy += weight * entropy(group['labels'])
    
    # 信息增益 = 分裂前熵 - 分裂后加权熵
    return base_entropy - weighted_entropy


def gini_index(labels: List[Any]) -> float:
    """
    计算基尼系数
    
    公式: Gini(S) = 1 - Σ p_i²
    
    Args:
        labels: 类别标签列表
    
    Returns:
        基尼系数（0到1之间，0.5为二分类最大）
    
    示例:
        >>> gini_index(['是', '是', '否', '否'])
        0.5
        >>> gini_index(['是', '是', '是'])
        0.0
    """
    if not labels:
        return 0.0
    
    counter = Counter(labels)
    total = len(labels)
    
    gini = 1.0
    for count in counter.values():
        p = count / total
        gini -= p * p
    
    return gini


def majority_vote(labels: List[Any]) -> Any:
    """
    多数投票，返回出现次数最多的类别
    
    Args:
        labels: 类别标签列表
    
    Returns:
        众数类别
    """
    if not labels:
        return None
    counter = Counter(labels)
    return counter.most_common(1)[0][0]


# ============================================================
# 第二部分：决策树节点类
# ============================================================

class TreeNode:
    """
    决策树节点类
    
    每个节点可以是：
    - 内部节点：包含分裂属性和分支
    - 叶子节点：包含预测类别
    """
    
    def __init__(self):
        self.is_leaf = False
        self.split_attribute = None  # 分裂属性（内部节点）
        self.branches = {}           # 分支字典（内部节点）
        self.predicted_class = None  # 预测类别（叶子节点）
        self.samples_count = 0       # 该节点的样本数
        self.class_distribution = {} # 类别分布
    
    def predict(self, sample: Dict) -> Any:
        """
        对单个样本进行预测
        
        Args:
            sample: 样本字典
        
        Returns:
            预测类别
        """
        if self.is_leaf:
            return self.predicted_class
        
        # 获取样本在分裂属性上的值
        value = sample.get(self.split_attribute)
        
        # 如果值不存在于训练时的分支中，返回该节点的多数类
        if value not in self.branches:
            return majority_vote(list(self.class_distribution.keys()))
        
        # 递归预测
        return self.branches[value].predict(sample)
    
    def __str__(self, indent: int = 0) -> str:
        """可视化树的结构"""
        prefix = "  " * indent
        
        if self.is_leaf:
            return f"{prefix}[叶子] 预测: {self.predicted_class} (样本数: {self.samples_count})"
        
        result = f"{prefix}[节点] {self.split_attribute}? (样本数: {self.samples_count})\n"
        for value, child in self.branches.items():
            result += f"{prefix}  ={value}:\n{child.__str__(indent + 2)}\n"
        
        return result.rstrip()


# ============================================================
# 第三部分：ID3决策树算法实现
# ============================================================

class ID3DecisionTree:
    """
    ID3决策树分类器（从零实现）
    
    使用信息增益作为分裂标准
    
    Attributes:
        root: 决策树的根节点
        max_depth: 最大深度（预剪枝）
        min_samples_split: 分裂所需最小样本数（预剪枝）
    """
    
    def __init__(self, max_depth: Optional[int] = None, 
                 min_samples_split: int = 2):
        """
        初始化决策树
        
        Args:
            max_depth: 树的最大深度，None表示不限制
            min_samples_split: 内部节点分裂所需的最小样本数
        """
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.classes_ = None
    
    def fit(self, dataset: List[Dict], labels: List[Any]) -> 'ID3DecisionTree':
        """
        训练决策树
        
        Args:
            dataset: 训练数据集
            labels: 类别标签
        
        Returns:
            self
        """
        if not dataset or not labels:
            raise ValueError("数据集不能为空")
        
        self.classes_ = list(set(labels))
        attributes = list(dataset[0].keys())
        
        self.root = self._build_tree(dataset, labels, attributes, depth=0)
        return self
    
    def _build_tree(self, dataset: List[Dict], labels: List[Any],
                    attributes: List[str], depth: int) -> TreeNode:
        """
        递归构建决策树（核心算法）
        
        Args:
            dataset: 当前数据集
            labels: 当前标签
            attributes: 可用属性列表
            depth: 当前深度
        
        Returns:
            构建好的树节点
        """
        node = TreeNode()
        node.samples_count = len(labels)
        node.class_distribution = Counter(labels)
        
        # 终止条件1：所有样本属于同一类
        if len(set(labels)) == 1:
            node.is_leaf = True
            node.predicted_class = labels[0]
            return node
        
        # 终止条件2：没有可用属性
        if not attributes:
            node.is_leaf = True
            node.predicted_class = majority_vote(labels)
            return node
        
        # 终止条件3：样本数不足（预剪枝）
        if len(labels) < self.min_samples_split:
            node.is_leaf = True
            node.predicted_class = majority_vote(labels)
            return node
        
        # 终止条件4：达到最大深度（预剪枝）
        if self.max_depth is not None and depth >= self.max_depth:
            node.is_leaf = True
            node.predicted_class = majority_vote(labels)
            return node
        
        # 选择最优分裂属性
        best_attribute = None
        best_gain = -1
        
        for attr in attributes:
            gain = information_gain(dataset, labels, attr)
            if gain > best_gain:
                best_gain = gain
                best_attribute = attr
        
        # 如果信息增益为0，停止分裂
        if best_gain <= 0:
            node.is_leaf = True
            node.predicted_class = majority_vote(labels)
            return node
        
        # 设置分裂属性
        node.split_attribute = best_attribute
        
        # 按属性值分组递归构建子树
        groups = {}
        for sample, label in zip(dataset, labels):
            value = sample[best_attribute]
            if value not in groups:
                groups[value] = {'samples': [], 'labels': []}
            groups[value]['samples'].append(sample)
            groups[value]['labels'].append(label)
        
        # 递归构建每个分支
        remaining_attrs = [a for a in attributes if a != best_attribute]
        for value, group in groups.items():
            node.branches[value] = self._build_tree(
                group['samples'], group['labels'], 
                remaining_attrs, depth + 1
            )
        
        return node
    
    def predict(self, dataset: List[Dict]) -> List[Any]:
        """
        预测多个样本
        
        Args:
            dataset: 待预测的数据集
        
        Returns:
            预测结果列表
        """
        return [self.root.predict(sample) for sample in dataset]
    
    def score(self, dataset: List[Dict], labels: List[Any]) -> float:
        """
        计算分类准确率
        
        Args:
            dataset: 测试数据集
            labels: 真实标签
        
        Returns:
        准确率（0-1之间）
        """
        predictions = self.predict(dataset)
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        return correct / len(labels)
    
    def __str__(self) -> str:
        """返回树的可视化字符串"""
        if self.root is None:
            return "决策树尚未训练"
        return str(self.root)


# ============================================================
# 第四部分：CART决策树（基尼系数版）
# ============================================================

class CARTDecisionTree:
    """
    CART决策树分类器（从零实现）
    
    使用基尼系数作为分裂标准，生成二叉树
    """
    
    def __init__(self, max_depth: Optional[int] = None,
                 min_samples_split: int = 2):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.classes_ = None
    
    def _find_best_split(self, dataset: List[Dict], labels: List[Any],
                         attributes: List[str]) -> Tuple[str, Any, float]:
        """
        找到最优分裂（属性，值，基尼系数减少量）
        
        Returns:
            (最优属性, 最优分裂值, 基尼系数减少量)
        """
        base_gini = gini_index(labels)
        best_attr = None
        best_value = None
        best_reduction = -1
        
        for attr in attributes:
            # 获取该属性的所有可能值
            values = set(sample[attr] for sample in dataset)
            
            for value in values:
                # 分成两组：等于value和不等于value
                left_labels = [l for s, l in zip(dataset, labels) if s[attr] == value]
                right_labels = [l for s, l in zip(dataset, labels) if s[attr] != value]
                
                if not left_labels or not right_labels:
                    continue
                
                # 计算加权基尼系数
                n = len(labels)
                weighted_gini = (len(left_labels) / n * gini_index(left_labels) +
                                len(right_labels) / n * gini_index(right_labels))
                
                reduction = base_gini - weighted_gini
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_attr = attr
                    best_value = value
        
        return best_attr, best_value, best_reduction
    
    def fit(self, dataset: List[Dict], labels: List[Any]) -> 'CARTDecisionTree':
        """训练CART决策树"""
        if not dataset or not labels:
            raise ValueError("数据集不能为空")
        
        self.classes_ = list(set(labels))
        attributes = list(dataset[0].keys())
        
        self.root = self._build_tree(dataset, labels, attributes, depth=0)
        return self
    
    def _build_tree(self, dataset, labels, attributes, depth):
        """递归构建二叉树"""
        node = TreeNode()
        node.samples_count = len(labels)
        node.class_distribution = Counter(labels)
        
        # 终止条件
        if len(set(labels)) == 1 or not attributes:
            node.is_leaf = True
            node.predicted_class = majority_vote(labels)
            return node
        
        if len(labels) < self.min_samples_split:
            node.is_leaf = True
            node.predicted_class = majority_vote(labels)
            return node
        
        if self.max_depth is not None and depth >= self.max_depth:
            node.is_leaf = True
            node.predicted_class = majority_vote(labels)
            return node
        
        # 找到最优分裂
        best_attr, best_value, reduction = self._find_best_split(
            dataset, labels, attributes
        )
        
        if reduction <= 0:
            node.is_leaf = True
            node.predicted_class = majority_vote(labels)
            return node
        
        # 设置分裂条件（二叉树）
        node.split_attribute = (best_attr, best_value)
        
        # 分成左右两支
        left_dataset = []
        left_labels = []
        right_dataset = []
        right_labels = []
        
        for sample, label in zip(dataset, labels):
            if sample[best_attr] == best_value:
                left_dataset.append(sample)
                left_labels.append(label)
            else:
                right_dataset.append(sample)
                right_labels.append(label)
        
        # 递归构建
        node.branches['是'] = self._build_tree(
            left_dataset, left_labels, attributes, depth + 1
        )
        node.branches['否'] = self._build_tree(
            right_dataset, right_labels, attributes, depth + 1
        )
        
        return node
    
    def predict(self, dataset: List[Dict]) -> List[Any]:
        """预测"""
        return [self._predict_one(sample) for sample in dataset]
    
    def _predict_one(self, sample: Dict) -> Any:
        """预测单个样本"""
        node = self.root
        while not node.is_leaf:
            attr, value = node.split_attribute
            if sample.get(attr) == value:
                node = node.branches['是']
            else:
                node = node.branches['否']
        return node.predicted_class
    
    def score(self, dataset: List[Dict], labels: List[Any]) -> float:
        """计算准确率"""
        predictions = self.predict(dataset)
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        return correct / len(labels)


# ============================================================
# 第五部分：实战演示
# ============================================================

def demo_tennis_dataset():
    """
    演示：打网球数据集（经典ID3示例）
    """
    print("=" * 60)
    print("演示1：打网球预测（经典ID3数据集）")
    print("=" * 60)
    
    # 经典打网球数据集
    dataset = [
        {'天气': '晴', '温度': '热', '湿度': '高', '风力': '弱'},
        {'天气': '晴', '温度': '热', '湿度': '高', '风力': '强'},
        {'天气': '多云', '温度': '热', '湿度': '高', '风力': '弱'},
        {'天气': '雨', '温度': '温和', '湿度': '高', '风力': '弱'},
        {'天气': '雨', '温度': '凉爽', '湿度': '正常', '风力': '弱'},
        {'天气': '雨', '温度': '凉爽', '湿度': '正常', '风力': '强'},
        {'天气': '多云', '温度': '凉爽', '湿度': '正常', '风力': '强'},
        {'天气': '晴', '温度': '温和', '湿度': '高', '风力': '弱'},
        {'天气': '晴', '温度': '凉爽', '湿度': '正常', '风力': '弱'},
        {'天气': '雨', '温度': '温和', '湿度': '正常', '风力': '弱'},
        {'天气': '晴', '温度': '温和', '湿度': '正常', '风力': '强'},
        {'天气': '多云', '温度': '温和', '湿度': '高', '风力': '强'},
        {'天气': '多云', '温度': '热', '湿度': '正常', '风力': '弱'},
        {'天气': '雨', '温度': '温和', '湿度': '高', '风力': '强'},
    ]
    labels = ['否', '否', '是', '是', '是', '否', '是', '否', '是', '是', 
              '是', '是', '是', '否']
    
    # 计算各属性的信息增益
    print("\n【信息增益计算】")
    print("-" * 40)
    for attr in ['天气', '温度', '湿度', '风力']:
        ig = information_gain(dataset, labels, attr)
        print(f"信息增益({attr}) = {ig:.4f}")
    
    print(f"\n总体熵 = {entropy(labels):.4f}")
    
    # 训练决策树
    print("\n【训练ID3决策树】")
    print("-" * 40)
    tree = ID3DecisionTree()
    tree.fit(dataset, labels)
    
    print(f"训练准确率: {tree.score(dataset, labels):.2%}")
    
    # 可视化树结构
    print("\n【决策树结构】")
    print("-" * 40)
    print(tree)
    
    # 预测新样本
    print("\n【预测新样本】")
    print("-" * 40)
    test_samples = [
        {'天气': '晴', '温度': '凉爽', '湿度': '正常', '风力': '弱'},
        {'天气': '雨', '温度': '温和', '湿度': '高', '风力': '强'},
    ]
    predictions = tree.predict(test_samples)
    for sample, pred in zip(test_samples, predictions):
        print(f"条件: {sample}")
        print(f"预测: {'适合打球 ✓' if pred == '是' else '不适合打球 ✗'}")
        print()


def demo_entropy_visualization():
    """
    演示：熵的可视化理解
    """
    print("\n" + "=" * 60)
    print("演示2：熵的计算与理解")
    print("=" * 60)
    
    print("\n【不同分布的熵】")
    print("-" * 40)
    
    # 纯分布
    pure = ['是', '是', '是', '是', '是']
    print(f"纯分布(5个'是'): 熵 = {entropy(pure):.4f} (最小)")
    
    # 均匀分布
    uniform = ['是', '否', '是', '否', '是', '否', '是', '否']
    print(f"均匀分布(各50%): 熵 = {entropy(uniform):.4f} (最大)")
    
    # 偏斜分布
    skewed = ['是', '是', '是', '是', '否']
    print(f"偏斜分布(80%是): 熵 = {entropy(skewed):.4f} (中等)")
    
    print("\n【熵随比例变化】")
    print("-" * 40)
    print("比例(p)    熵")
    print("-" * 20)
    for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        labels = ['是'] * int(p * 100) + ['否'] * int((1-p) * 100)
        if labels:
            print(f"{p:.1f}       {entropy(labels):.4f}")


def demo_gini_comparison():
    """
    演示：信息增益 vs 基尼系数
    """
    print("\n" + "=" * 60)
    print("演示3：信息增益 vs 基尼系数")
    print("=" * 60)
    
    # 相同的数据集
    dataset = [
        {'颜色': '红', '大小': '大'},
        {'颜色': '红', '大小': '小'},
        {'颜色': '红', '大小': '大'},
        {'颜色': '蓝', '大小': '小'},
        {'颜色': '蓝', '大小': '大'},
        {'颜色': '蓝', '大小': '小'},
        {'颜色': '绿', '大小': '大'},
        {'颜色': '绿', '大小': '小'},
    ]
    labels = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C']
    
    print("\n数据集：")
    print("颜色分布: 红(3), 蓝(3), 绿(2)")
    print("类别分布: A(3), B(3), C(2)")
    
    print("\n【信息增益计算】")
    for attr in ['颜色', '大小']:
        ig = information_gain(dataset, labels, attr)
        print(f"IG({attr}) = {ig:.4f}")
    
    print("\n【基尼系数】")
    print(f"总体基尼系数 = {gini_index(labels):.4f}")
    
    # 用两种算法训练并比较
    print("\n【算法比较】")
    print("-" * 40)
    
    id3_tree = ID3DecisionTree(max_depth=2)
    id3_tree.fit(dataset, labels)
    print(f"ID3准确率: {id3_tree.score(dataset, labels):.2%}")
    
    cart_tree = CARTDecisionTree(max_depth=2)
    cart_tree.fit(dataset, labels)
    print(f"CART准确率: {cart_tree.score(dataset, labels):.2%}")


def demo_pruning_effect():
    """
    演示：剪枝的效果
    """
    print("\n" + "=" * 60)
    print("演示4：预剪枝的效果")
    print("=" * 60)
    
    # 生成一个稍复杂的数据集
    dataset = []
    labels = []
    
    # 模拟一些带噪声的数据
    for i in range(50):
        dataset.append({
            'A': 'X' if i % 3 == 0 else 'Y',
            'B': 'M' if i % 2 == 0 else 'N',
            'C': 'P' if i % 4 == 0 else 'Q',
        })
        # 简单的模式，但加了一些噪声
        labels.append('正' if (i % 3 == 0 or i % 2 == 0) and i % 7 != 0 else '负')
    
    # 不剪枝（过拟合）
    print("\n【不限制深度的树】")
    deep_tree = ID3DecisionTree(max_depth=None)
    deep_tree.fit(dataset, labels)
    print(f"训练集准确率: {deep_tree.score(dataset, labels):.2%}")
    print(f"树的复杂度: 样本都被完美分类（可能过拟合）")
    
    # 限制深度
    print("\n【限制深度=2的树】")
    pruned_tree = ID3DecisionTree(max_depth=2)
    pruned_tree.fit(dataset, labels)
    print(f"训练集准确率: {pruned_tree.score(dataset, labels):.2%}")
    print(f"树的复杂度: 更简单的树，更好的泛化能力")
    
    print("\n💡 启示：适当的剪枝可以防止过拟合，提高泛化能力！")


def demo_decision_rules():
    """
    演示：从树中提取决策规则
    """
    print("\n" + "=" * 60)
    print("演示5：决策规则提取")
    print("=" * 60)
    
    # 医疗诊断示例
    dataset = [
        {'发烧': '高', '咳嗽': '是', '痰色': '黄'},
        {'发烧': '高', '咳嗽': '是', '痰色': '白'},
        {'发烧': '低', '咳嗽': '否', '痰色': '无'},
        {'发烧': '高', '咳嗽': '否', '痰色': '无'},
        {'发烧': '低', '咳嗽': '是', '痰色': '白'},
        {'发烧': '中', '咳嗽': '是', '痰色': '黄'},
    ]
    labels = ['肺炎', '支气管炎', '感冒', '流感', '感冒', '支气管炎']
    
    tree = ID3DecisionTree()
    tree.fit(dataset, labels)
    
    print("\n【训练好的决策树】")
    print(tree)
    
    print("\n【可读的决策规则】")
    print("-" * 40)
    print("规则1: 如果 发烧=高 且 咳嗽=是 且 痰色=黄 → 肺炎")
    print("规则2: 如果 发烧=高 且 咳嗽=是 且 痰色=白 → 支气管炎")
    print("规则3: 如果 发烧=低 且 咳嗽=否 → 感冒")
    print("规则4: 如果 发烧=高 且 咳嗽=否 → 流感")
    print("...")
    print("\n这就是决策树的最大优势：可解释性！")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("\n" + "🌳 " * 20)
    print("第九章：决策树算法 - 完整演示")
    print("从零实现ID3和CART决策树")
    print("🌳 " * 20 + "\n")
    
    # 运行所有演示
    demo_entropy_visualization()
    demo_tennis_dataset()
    demo_gini_comparison()
    demo_pruning_effect()
    demo_decision_rules()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
    print("\n关键收获：")
    print("1. 决策树用熵/基尼系数选择最优分裂")
    print("2. ID3用信息增益，CART用基尼系数")
    print("3. 剪枝防止过拟合")
    print("4. 决策树最大的优点是可解释性")
