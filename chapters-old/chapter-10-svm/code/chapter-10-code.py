#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第十章配套代码：集成学习 - 三个臭皮匠顶个诸葛亮
Ensemble Learning: Wisdom of the Crowd

包含：
1. Bootstrap采样实现
2. Bagging分类器（从底层实现）
3. 随机森林（纯Python实现）
4. AdaBoost分类器（纯Python实现）
5. 投票分类器
6. 特征重要性计算
7. OOB误差估计
8. 可视化与实战演示

作者: ML教材写作项目
日期: 2026-03-24
"""

import math
import random
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

# ============================================================
# 第一部分：基础工具函数
# ============================================================

def bootstrap_sample(data: List[Dict], target_col: str) -> Tuple[List[Dict], List[int]]:
    """
    Bootstrap采样 - 有放回抽样
    
    从原始数据集中有放回地抽取n个样本，其中n等于数据集大小。
    未被选中的样本称为OOB (Out-of-Bag)样本，可用于验证。
    
    Args:
        data: 原始数据集
        target_col: 目标列名
        
    Returns:
        (采样后的数据, OOB样本索引列表)
    """
    n = len(data)
    
    # 有放回抽样
    indices = [random.randint(0, n-1) for _ in range(n)]
    sample = [data[i] for i in indices]
    
    # 计算OOB样本索引（未被选中的）
    selected_set = set(indices)
    oob_indices = [i for i in range(n) if i not in selected_set]
    
    return sample, oob_indices


def random_subspace(features: List[str], max_features: Optional[int] = None) -> List[str]:
    """
    随机子空间采样 - 随机选择部分特征
    
    Args:
        features: 所有特征列表
        max_features: 最大特征数，None表示sqrt(len(features))
        
    Returns:
        选中的特征列表
    """
    if max_features is None:
        max_features = int(math.sqrt(len(features)))
    
    # 确保不超过特征总数
    max_features = min(max_features, len(features))
    
    # 随机选择
    return random.sample(features, max_features)


# ============================================================
# 第二部分：决策树桩 (Decision Stump) - 用于AdaBoost的弱学习器
# ============================================================

class DecisionStump:
    """
    决策树桩 - 只有一个分裂节点的简单决策树
    
    作为AdaBoost的弱分类器，只需要比随机猜测略好即可。
    """
    
    def __init__(self):
        self.feature = None      # 分裂特征
        self.threshold = None    # 分裂阈值
        self.polarity = 1        # 极性（1或-1）
        self.alpha = 0.0         # 该弱分类器的权重
        
    def fit(self, X: List[List[float]], y: List[int], 
            sample_weights: List[float]) -> float:
        """
        训练决策树桩
        
        Args:
            X: 特征矩阵
            y: 标签（+1或-1）
            sample_weights: 样本权重
            
        Returns:
            该分类器的权重alpha
        """
        n_samples = len(X)
        n_features = len(X[0])
        
        min_error = float('inf')
        
        # 遍历所有特征
        for feature_i in range(n_features):
            # 获取该特征的所有取值
            feature_values = sorted(set(row[feature_i] for row in X))
            
            # 尝试每个可能的分裂点
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i+1]) / 2
                
                # 尝试两种极性
                for polarity in [1, -1]:
                    error = 0.0
                    for j in range(n_samples):
                        prediction = polarity if X[j][feature_i] <= threshold else -polarity
                        if prediction != y[j]:
                            error += sample_weights[j]
                    
                    # 记录最佳分裂
                    if error < min_error:
                        min_error = error
                        self.feature = feature_i
                        self.threshold = threshold
                        self.polarity = polarity
        
        # 计算分类器权重 alpha
        # epsilon = min_error + 1e-10 防止除零
        epsilon = min_error + 1e-10
        self.alpha = 0.5 * math.log((1 - epsilon) / epsilon)
        
        return self.alpha
    
    def predict(self, x: List[float]) -> int:
        """预测单个样本"""
        if self.feature is None:
            return 1
        
        if x[self.feature] <= self.threshold:
            return self.polarity
        else:
            return -self.polarity
    
    def predict_batch(self, X: List[List[float]]) -> List[int]:
        """批量预测"""
        return [self.predict(x) for x in X]


# ============================================================
# 第三部分：简化版决策树 - 用于Bagging和Random Forest
# ============================================================

class SimpleDecisionTree:
    """
    简化版决策树分类器 - 用于集成学习
    
    实现了CART算法的简化版本，支持：
    - 基尼指数分裂
    - 随机特征子集选择
    - 最大深度限制
    - 最小样本分割限制
    """
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,
                 max_features: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None
        self.feature_indices = None  # 本次训练使用的特征索引
        
    class Node:
        """决策树节点"""
        def __init__(self):
            self.feature = None
            self.threshold = None
            self.left = None
            self.right = None
            self.label = None
            
    def _gini(self, y: List[int]) -> float:
        """计算基尼不纯度"""
        if not y:
            return 0.0
        
        counts = Counter(y)
        n = len(y)
        
        gini = 1.0
        for count in counts.values():
            p = count / n
            gini -= p * p
        
        return gini
    
    def _best_split(self, X: List[List[float]], y: List[int],
                   feature_indices: List[int]) -> Tuple[int, float, float]:
        """
        寻找最佳分裂点
        
        Returns:
            (最佳特征索引, 最佳阈值, 最佳基尼减少量)
        """
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        
        parent_gini = self._gini(y)
        n = len(y)
        
        for feature_i in feature_indices:
            # 获取该特征的所有取值和对应标签
            feature_values = [(X[j][feature_i], y[j]) for j in range(len(X))]
            feature_values.sort()
            
            # 尝试每个可能的分裂点
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i][0] + feature_values[i+1][0]) / 2
                
                # 分裂
                left_y = [y[j] for j in range(len(X)) if X[j][feature_i] <= threshold]
                right_y = [y[j] for j in range(len(X)) if X[j][feature_i] > threshold]
                
                # 计算加权基尼
                n_left = len(left_y)
                n_right = len(right_y)
                
                if n_left == 0 or n_right == 0:
                    continue
                
                weighted_gini = (n_left / n) * self._gini(left_y) + \
                               (n_right / n) * self._gini(right_y)
                
                gain = parent_gini - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_i
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X: List[List[float]], y: List[int],
                   feature_indices: List[int], depth: int) -> 'SimpleDecisionTree.Node':
        """递归构建决策树"""
        node = self.Node()
        
        # 检查停止条件
        # 1. 所有样本属于同一类
        if len(set(y)) == 1:
            node.label = y[0]
            return node
        
        # 2. 达到最大深度
        if depth >= self.max_depth:
            node.label = Counter(y).most_common(1)[0][0]
            return node
        
        # 3. 样本数太少
        if len(y) < self.min_samples_split:
            node.label = Counter(y).most_common(1)[0][0]
            return node
        
        # 寻找最佳分裂
        feature, threshold, gain = self._best_split(X, y, feature_indices)
        
        # 如果无法分裂
        if feature is None or gain <= 0:
            node.label = Counter(y).most_common(1)[0][0]
            return node
        
        # 分裂数据
        left_X, left_y = [], []
        right_X, right_y = [], []
        
        for i in range(len(X)):
            if X[i][feature] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        
        node.feature = feature
        node.threshold = threshold
        node.left = self._build_tree(left_X, left_y, feature_indices, depth + 1)
        node.right = self._build_tree(right_X, right_y, feature_indices, depth + 1)
        
        return node
    
    def fit(self, X: List[List[float]], y: List[int]) -> 'SimpleDecisionTree':
        """
        训练决策树
        
        Args:
            X: 特征矩阵
            y: 标签列表
        """
        n_features = len(X[0])
        
        # 选择特征子集
        if self.max_features is None:
            self.max_features = int(math.sqrt(n_features))
        
        self.feature_indices = random.sample(
            range(n_features), 
            min(self.max_features, n_features)
        )
        
        self.tree = self._build_tree(X, y, self.feature_indices, 0)
        return self
    
    def _predict_one(self, x: List[float], node: 'SimpleDecisionTree.Node') -> int:
        """递归预测单个样本"""
        if node.label is not None:
            return node.label
        
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """批量预测"""
        return [self._predict_one(x, self.tree) for x in X]


# ============================================================
# 第四部分：Bagging分类器
# ============================================================

class BaggingClassifier:
    """
    Bagging分类器 (Bootstrap Aggregating)
    
    由Leo Breiman于1996年提出，核心思想：
    1. 通过Bootstrap采样生成多个训练子集
    2. 在每个子集上训练一个基学习器
    3. 预测时投票决定最终类别
    """
    
    def __init__(self, base_estimator, n_estimators: int = 10,
                 max_samples: float = 1.0, bootstrap: bool = True,
                 oob_score: bool = False):
        """
        Args:
            base_estimator: 基学习器类（需要实现fit和predict）
            n_estimators: 基学习器数量
            max_samples: 采样比例
            bootstrap: 是否有放回采样
            oob_score: 是否计算OOB分数
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        
        self.estimators = []
        self.estimators_samples = []
        self.oob_score_ = None
        
    def fit(self, X: List[List[float]], y: List[int]) -> 'BaggingClassifier':
        """训练Bagging分类器"""
        n_samples = len(X)
        sample_size = int(n_samples * self.max_samples)
        
        self.estimators = []
        self.estimators_samples = []
        
        # 创建OOB预测存储（样本 -> 预测列表）
        oob_predictions = {i: [] for i in range(n_samples)}
        
        for i in range(self.n_estimators):
            # Bootstrap采样
            if self.bootstrap:
                indices = [random.randint(0, n_samples-1) for _ in range(sample_size)]
            else:
                indices = random.sample(range(n_samples), sample_size)
            
            self.estimators_samples.append(indices)
            
            # 创建训练子集
            X_subset = [X[j] for j in indices]
            y_subset = [y[j] for j in indices]
            
            # 训练基学习器
            estimator = self.base_estimator()
            estimator.fit(X_subset, y_subset)
            self.estimators.append(estimator)
            
            # 记录OOB预测
            if self.oob_score:
                oob_indices = set(range(n_samples)) - set(indices)
                for idx in oob_indices:
                    pred = estimator.predict([X[idx]])[0]
                    oob_predictions[idx].append(pred)
        
        # 计算OOB分数
        if self.oob_score:
            correct = 0
            total = 0
            for i in range(n_samples):
                if oob_predictions[i]:
                    # 投票
                    pred = Counter(oob_predictions[i]).most_common(1)[0][0]
                    if pred == y[i]:
                        correct += 1
                    total += 1
            
            self.oob_score_ = correct / total if total > 0 else 0.0
        
        return self
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """
        预测 - 使用多数投票
        """
        predictions = []
        
        for x in X:
            votes = []
            for estimator in self.estimators:
                pred = estimator.predict([x])[0]
                votes.append(pred)
            
            # 多数投票
            final_pred = Counter(votes).most_common(1)[0][0]
            predictions.append(final_pred)
        
        return predictions
    
    def predict_proba(self, X: List[List[float]]) -> List[Dict[int, float]]:
        """
        预测概率 - 基于投票比例
        """
        probabilities = []
        
        for x in X:
            votes = []
            for estimator in self.estimators:
                pred = estimator.predict([x])[0]
                votes.append(pred)
            
            # 计算概率
            counter = Counter(votes)
            n = len(votes)
            proba = {label: count / n for label, count in counter.items()}
            probabilities.append(proba)
        
        return probabilities


# ============================================================
# 第五部分：随机森林
# ============================================================

class RandomForestClassifier:
    """
    随机森林分类器
    
    由Leo Breiman于2001年提出，结合了：
    1. Bagging：Bootstrap采样
    2. 随机特征子空间：每次分裂只考虑随机子集的特征
    
    这两个随机性使得森林中的树 diverse，从而减小方差，提高泛化能力。
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 min_samples_split: int = 2, max_features: Optional[int] = None,
                 oob_score: bool = True, random_state: Optional[int] = None):
        """
        Args:
            n_estimators: 树的数量
            max_depth: 树的最大深度
            min_samples_split: 最小分裂样本数
            max_features: 每次分裂考虑的最大特征数，None表示sqrt(n_features)
            oob_score: 是否计算OOB分数
            random_state: 随机种子
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.oob_score = oob_score
        self.random_state = random_state
        
        self.trees = []
        self.classes = None
        self.oob_score_ = None
        self.feature_importances_ = None
        
    def fit(self, X: List[List[float]], y: List[int]) -> 'RandomForestClassifier':
        """训练随机森林"""
        if self.random_state is not None:
            random.seed(self.random_state)
        
        n_samples = len(X)
        n_features = len(X[0])
        self.classes = list(set(y))
        
        self.trees = []
        
        # OOB预测存储
        oob_votes = {i: [] for i in range(n_samples)}
        
        # 特征重要性累加
        feature_importance_sum = [0.0] * n_features
        
        print(f"🌲 正在训练随机森林（{self.n_estimators}棵树）...")
        
        for i in range(self.n_estimators):
            # Bootstrap采样
            indices = [random.randint(0, n_samples-1) for _ in range(n_samples)]
            X_bootstrap = [X[j] for j in indices]
            y_bootstrap = [y[j] for j in indices]
            
            # 训练决策树
            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            
            # 记录OOB预测
            if self.oob_score:
                oob_indices = set(range(n_samples)) - set(indices)
                for idx in oob_indices:
                    pred = tree.predict([X[idx]])[0]
                    oob_votes[idx].append(pred)
            
            # 累加特征重要性（简化版：基于使用次数）
            for feature_idx in tree.feature_indices:
                feature_importance_sum[feature_idx] += 1
            
            if (i + 1) % 20 == 0:
                print(f"  已完成 {i+1}/{self.n_estimators} 棵树")
        
        # 计算OOB分数
        if self.oob_score:
            correct = 0
            total = 0
            for i in range(n_samples):
                if oob_votes[i]:
                    pred = Counter(oob_votes[i]).most_common(1)[0][0]
                    if pred == y[i]:
                        correct += 1
                    total += 1
            
            self.oob_score_ = correct / total if total > 0 else 0.0
            print(f"📊 OOB Score: {self.oob_score_:.4f}")
        
        # 归一化特征重要性
        total = sum(feature_importance_sum)
        if total > 0:
            self.feature_importances_ = [f / total for f in feature_importance_sum]
        
        print(f"✅ 随机森林训练完成！")
        return self
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """
        预测 - 软投票（多数投票）
        """
        predictions = []
        
        for x in X:
            votes = []
            for tree in self.trees:
                pred = tree.predict([x])[0]
                votes.append(pred)
            
            final_pred = Counter(votes).most_common(1)[0][0]
            predictions.append(final_pred)
        
        return predictions
    
    def predict_proba(self, X: List[List[float]]) -> List[Dict[int, float]]:
        """预测概率"""
        probabilities = []
        
        for x in X:
            votes = []
            for tree in self.trees:
                pred = tree.predict([x])[0]
                votes.append(pred)
            
            counter = Counter(votes)
            n = len(votes)
            proba = {label: count / n for label, count in counter.items()}
            probabilities.append(proba)
        
        return probabilities


# ============================================================
# 第六部分：AdaBoost分类器
# ============================================================

class AdaBoostClassifier:
    """
    AdaBoost分类器 (Adaptive Boosting)
    
    由Freund和Schapire于1995-1996年提出。
    
    核心思想：
    1. 初始化时所有样本权重相等
    2. 每轮训练一个弱分类器
    3. 增加被错误分类样本的权重
    4. 最终的强分类器是弱分类器的加权组合
    
    数学上，AdaBoost可以看作是在指数损失函数上的梯度下降。
    """
    
    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0):
        """
        Args:
            n_estimators: 弱分类器数量
            learning_rate: 学习率，用于缩放分类器权重
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        
        self.estimators = []
        self.estimator_weights = []
        self.classes = None
        
    def fit(self, X: List[List[float]], y: List[int]) -> 'AdaBoostClassifier':
        """
        训练AdaBoost分类器
        
        Args:
            X: 特征矩阵
            y: 标签（假设为+1和-1）
        """
        n_samples = len(X)
        
        # 确保标签是+1和-1
        unique_labels = list(set(y))
        if len(unique_labels) != 2:
            raise ValueError("AdaBoost目前只支持二分类问题")
        
        # 转换为+1/-1
        label_map = {unique_labels[0]: -1, unique_labels[1]: 1}
        y_binary = [label_map[yi] for yi in y]
        self.classes = unique_labels
        
        # 初始化样本权重（均匀分布）
        sample_weights = [1.0 / n_samples] * n_samples
        
        self.estimators = []
        self.estimator_weights = []
        
        print(f"🚀 正在训练AdaBoost（{self.n_estimators}个弱分类器）...")
        
        for i in range(self.n_estimators):
            # 训练弱分类器（决策树桩）
            stump = DecisionStump()
            alpha = stump.fit(X, y_binary, sample_weights)
            alpha *= self.learning_rate
            
            # 保存分类器及其权重
            self.estimators.append(stump)
            self.estimator_weights.append(alpha)
            
            # 更新样本权重
            predictions = stump.predict_batch(X)
            
            new_weights = []
            weight_sum = 0.0
            for j in range(n_samples):
                # 指数损失更新
                if predictions[j] == y_binary[j]:
                    # 正确分类：权重减小
                    new_weight = sample_weights[j] * math.exp(-alpha)
                else:
                    # 错误分类：权重增大
                    new_weight = sample_weights[j] * math.exp(alpha)
                
                new_weights.append(new_weight)
                weight_sum += new_weight
            
            # 归一化
            sample_weights = [w / weight_sum for w in new_weights]
            
            # 计算当前集成错误率
            ensemble_preds = self._predict_binary(X)
            errors = sum(1 for j in range(n_samples) if ensemble_preds[j] != y_binary[j])
            error_rate = errors / n_samples
            
            if (i + 1) % 10 == 0:
                print(f"  迭代 {i+1}/{self.n_estimators} - 错误率: {error_rate:.4f}")
            
            # 如果错误率为0，提前停止
            if error_rate == 0:
                print(f"  提前停止：错误率为0")
                break
        
        print(f"✅ AdaBoost训练完成！")
        return self
    
    def _predict_binary(self, X: List[List[float]]) -> List[int]:
        """预测（返回+1/-1）"""
        predictions = []
        
        for x in X:
            # 加权投票
            score = 0.0
            for stump, alpha in zip(self.estimators, self.estimator_weights):
                score += alpha * stump.predict(x)
            
            predictions.append(1 if score > 0 else -1)
        
        return predictions
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """
        预测（返回原始标签）
        """
        binary_preds = self._predict_binary(X)
        
        # 映射回原始标签
        label_map_inv = {-1: self.classes[0], 1: self.classes[1]}
        return [label_map_inv[p] for p in binary_preds]
    
    def predict_proba(self, X: List[List[float]]) -> List[Dict[int, float]]:
        """
        预测概率（使用sigmoid近似）
        """
        probabilities = []
        label_map_inv = {-1: self.classes[0], 1: self.classes[1]}
        
        for x in X:
            # 计算加权分数
            score = 0.0
            for stump, alpha in zip(self.estimators, self.estimator_weights):
                score += alpha * stump.predict(x)
            
            # 使用sigmoid转换为概率
            # P(y=1|x) = 1 / (1 + exp(-2*score))
            proba_pos = 1.0 / (1.0 + math.exp(-2 * score))
            proba_neg = 1.0 - proba_pos
            
            proba = {
                label_map_inv[1]: proba_pos,
                label_map_inv[-1]: proba_neg
            }
            probabilities.append(proba)
        
        return probabilities


# ============================================================
# 第七部分：投票分类器
# ============================================================

class VotingClassifier:
    """
    投票分类器 - 组合多个不同类型的分类器
    
    支持：
    - 硬投票（Hard Voting）：多数投票
    - 软投票（Soft Voting）：平均概率
    """
    
    def __init__(self, estimators: List[Tuple[str, Any]], voting: str = 'hard'):
        """
        Args:
            estimators: 分类器列表，格式为[(name, estimator), ...]
            voting: 'hard'或'soft'
        """
        self.estimators = estimators
        self.voting = voting
        self.fitted_estimators = []
        
    def fit(self, X: List[List[float]], y: List[int]) -> 'VotingClassifier':
        """训练所有分类器"""
        self.fitted_estimators = []
        
        for name, estimator in self.estimators:
            # 复制并训练
            fitted = estimator.__class__(**estimator.__dict__)
            fitted.fit(X, y)
            self.fitted_estimators.append((name, fitted))
        
        return self
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """预测"""
        if self.voting == 'hard':
            # 硬投票：多数投票
            predictions = []
            
            for x in X:
                votes = []
                for name, estimator in self.fitted_estimators:
                    pred = estimator.predict([x])[0]
                    votes.append(pred)
                
                final_pred = Counter(votes).most_common(1)[0][0]
                predictions.append(final_pred)
            
            return predictions
        else:
            # 软投票：选择概率最高的
            proba_predictions = self.predict_proba(X)
            return [max(p, key=p.get) for p in proba_predictions]
    
    def predict_proba(self, X: List[List[float]]) -> List[Dict[int, float]]:
        """预测概率（软投票）"""
        if self.voting != 'soft':
            raise ValueError("软投票需要voting='soft'")
        
        n_estimators = len(self.fitted_estimators)
        
        # 收集所有分类器的概率
        all_probas = []
        for name, estimator in self.fitted_estimators:
            proba = estimator.predict_proba(X)
            all_probas.append(proba)
        
        # 平均概率
        averaged_probas = []
        for i in range(len(X)):
            avg_proba = {}
            for proba in all_probas:
                for label, p in proba[i].items():
                    avg_proba[label] = avg_proba.get(label, 0.0) + p / n_estimators
            averaged_probas.append(avg_proba)
        
        return averaged_probas


# ============================================================
# 第八部分：可视化与评估工具
# ============================================================

def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    """计算准确率"""
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)


def confusion_matrix(y_true: List[int], y_pred: List[int], labels: List[int] = None) -> Dict:
    """
    计算混淆矩阵
    
    Returns:
        {
            'TP': 真正例数,
            'TN': 真负例数,
            'FP': 假正例数,
            'FN': 假负例数,
            'matrix': 二维矩阵
        }
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    
    n_classes = len(labels)
    matrix = [[0] * n_classes for _ in range(n_classes)]
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    for yt, yp in zip(y_true, y_pred):
        i = label_to_idx[yt]
        j = label_to_idx[yp]
        matrix[i][j] += 1
    
    return {
        'labels': labels,
        'matrix': matrix
    }


def print_confusion_matrix(cm: Dict):
    """打印混淆矩阵"""
    labels = cm['labels']
    matrix = cm['matrix']
    
    print("\n混淆矩阵:")
    print("      ", end="")
    for label in labels:
        print(f" 预测{label:>3}", end="")
    print()
    
    for i, label in enumerate(labels):
        print(f"实际{label:>3}", end="")
        for j in range(len(labels)):
            print(f" {matrix[i][j]:>6}", end="")
        print()


def plot_feature_importance(importances: List[float], feature_names: List[str] = None):
    """
    ASCII可视化特征重要性
    """
    if feature_names is None:
        feature_names = [f"特征{i}" for i in range(len(importances))]
    
    # 排序
    indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
    
    print("\n📊 特征重要性排名:")
    print("-" * 40)
    
    max_importance = max(importances)
    
    for i in indices[:10]:  # 只显示前10个
        bar_length = int(30 * importances[i] / max_importance)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"{feature_names[i]:>12} |{bar}| {importances[i]:.4f}")


# ============================================================
# 第九部分：演示数据集
# ============================================================

def load_iris_binary():
    """
    加载鸢尾花数据集的二分类版本（Setosa vs Versicolor）
    简化为二维特征以便可视化
    
    特征：
    - 花萼长度（sepal length）
    - 花萼宽度（sepal width）
    
    返回: (X, y, feature_names)
    """
    # 简化的鸢尾花数据集（前100个样本，2个类别）
    data = [
        # Setosa (0)
        [5.1, 3.5], [4.9, 3.0], [4.7, 3.2], [4.6, 3.1], [5.0, 3.6],
        [5.4, 3.9], [4.6, 3.4], [5.0, 3.4], [4.4, 2.9], [4.9, 3.1],
        [5.4, 3.7], [4.8, 3.4], [4.8, 3.0], [4.3, 3.0], [5.8, 4.0],
        [5.7, 4.4], [5.4, 3.9], [5.1, 3.5], [5.7, 3.8], [5.1, 3.8],
        [5.4, 3.4], [5.1, 3.7], [4.6, 3.6], [5.1, 3.3], [4.8, 3.4],
        [5.0, 3.0], [5.0, 3.4], [5.2, 3.5], [5.2, 3.4], [4.7, 3.2],
        [4.8, 3.1], [5.4, 3.4], [5.2, 4.1], [5.5, 4.2], [4.9, 3.1],
        [5.0, 3.2], [5.5, 3.5], [4.9, 3.6], [4.4, 3.0], [5.1, 3.4],
        [5.0, 3.5], [4.5, 2.3], [4.4, 3.2], [5.0, 3.5], [5.1, 3.8],
        [4.8, 3.0], [5.1, 3.8], [4.6, 3.2], [5.3, 3.7], [5.0, 3.3],
        # Versicolor (1)
        [7.0, 3.2], [6.4, 3.2], [6.9, 3.1], [5.5, 2.3], [6.5, 2.8],
        [5.7, 2.8], [6.3, 3.3], [4.9, 2.4], [6.6, 2.9], [5.2, 2.7],
        [5.0, 2.0], [5.9, 3.0], [6.0, 2.2], [6.1, 2.9], [5.6, 2.9],
        [6.7, 3.1], [5.6, 3.0], [5.8, 2.7], [6.2, 2.2], [5.6, 2.5],
        [5.9, 3.2], [6.1, 2.8], [6.3, 2.5], [6.1, 2.8], [6.4, 2.9],
        [6.6, 3.0], [6.8, 2.8], [6.7, 3.0], [6.0, 2.9], [5.7, 2.6],
        [5.5, 2.4], [5.5, 2.4], [5.8, 2.7], [6.0, 2.7], [5.4, 3.0],
        [6.0, 3.4], [6.7, 3.1], [6.3, 2.3], [5.6, 3.0], [5.5, 2.5],
        [5.5, 2.6], [6.1, 3.0], [5.8, 2.6], [5.0, 2.3], [5.6, 2.7],
        [5.7, 3.0], [5.7, 2.9], [6.2, 2.9], [5.1, 2.5], [5.7, 2.8],
    ]
    
    labels = [0] * 50 + [1] * 50
    feature_names = ['花萼长度', '花萼宽度']
    
    return data, labels, feature_names


def load_moons_dataset(n_samples: int = 100):
    """
    生成月牙形数据集（非线性可分）
    
    用于展示集成学习在非线性问题上的优势
    """
    X = []
    y = []
    
    # 类别0：上半个月牙
    for i in range(n_samples // 2):
        angle = 3.14159 * i / (n_samples // 2)
        x = 0.5 * math.cos(angle) + random.gauss(0, 0.1)
        y_val = 0.5 * math.sin(angle) + random.gauss(0, 0.1)
        X.append([x, y_val])
        y.append(0)
    
    # 类别1：下半个月牙
    for i in range(n_samples // 2):
        angle = 3.14159 * i / (n_samples // 2)
        x = 0.5 * math.cos(angle) + 0.5 + random.gauss(0, 0.1)
        y_val = 0.5 * math.sin(angle) - 0.3 + random.gauss(0, 0.1)
        X.append([x, y_val])
        y.append(1)
    
    return X, y, ['x', 'y']


def train_test_split(X: List[List[float]], y: List[int], 
                     test_ratio: float = 0.2) -> Tuple[List, List, List, List]:
    """
    划分训练集和测试集
    """
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    
    test_size = int(n * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test


# ============================================================
# 第十部分：实战演示
# ============================================================

def demo_bagging():
    """
    演示Bagging的工作原理
    """
    print("=" * 60)
    print("演示1：Bagging分类器")
    print("=" * 60)
    
    # 加载数据
    X, y, feature_names = load_iris_binary()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 使用Bagging
    bagging = BaggingClassifier(
        base_estimator=SimpleDecisionTree,
        n_estimators=20,
        oob_score=True
    )
    
    bagging.fit(X_train, y_train)
    
    # 预测
    y_pred = bagging.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nBagging测试准确率: {accuracy:.4f}")
    print(f"OOB Score: {bagging.oob_score_:.4f}")
    
    # 打印混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print_confusion_matrix(cm)


def demo_random_forest():
    """
    演示随机森林
    """
    print("\n" + "=" * 60)
    print("演示2：随机森林分类器")
    print("=" * 60)
    
    # 加载数据
    X, y, feature_names = load_iris_binary()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)
    
    # 训练随机森林
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        oob_score=True,
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    
    # 预测
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n随机森林测试准确率: {accuracy:.4f}")
    
    # 特征重要性
    if rf.feature_importances_:
        plot_feature_importance(rf.feature_importances_, feature_names)


def demo_adaboost():
    """
    演示AdaBoost
    """
    print("\n" + "=" * 60)
    print("演示3：AdaBoost分类器")
    print("=" * 60)
    
    # 加载月牙数据集（非线性问题）
    X, y, feature_names = load_moons_dataset(n_samples=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.3)
    
    print(f"数据集: 月牙形（非线性可分）")
    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    
    # 训练AdaBoost
    adaboost = AdaBoostClassifier(n_estimators=30, learning_rate=0.5)
    adaboost.fit(X_train, y_train)
    
    # 预测
    y_pred = adaboost.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAdaBoost测试准确率: {accuracy:.4f}")
    
    # 打印混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print_confusion_matrix(cm)


def demo_comparison():
    """
    对比单棵决策树、Bagging、随机森林和AdaBoost
    """
    print("\n" + "=" * 60)
    print("演示4：集成方法对比")
    print("=" * 60)
    
    # 生成月牙数据集
    X, y, _ = load_moons_dataset(n_samples=200)
    
    # 多次实验取平均
    n_runs = 5
    results = {
        '单棵决策树': [],
        'Bagging': [],
        '随机森林': [],
        'AdaBoost': []
    }
    
    print(f"运行{n_runs}次实验，每次不同随机划分...")
    
    for run in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.3)
        
        # 单棵决策树
        tree = SimpleDecisionTree(max_depth=5)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        results['单棵决策树'].append(accuracy_score(y_test, y_pred))
        
        # Bagging
        bagging = BaggingClassifier(SimpleDecisionTree, n_estimators=20)
        bagging.fit(X_train, y_train)
        y_pred = bagging.predict(X_test)
        results['Bagging'].append(accuracy_score(y_test, y_pred))
        
        # 随机森林
        rf = RandomForestClassifier(n_estimators=30, max_depth=5)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        results['随机森林'].append(accuracy_score(y_test, y_pred))
        
        # AdaBoost
        adaboost = AdaBoostClassifier(n_estimators=30)
        adaboost.fit(X_train, y_train)
        y_pred = adaboost.predict(X_test)
        results['AdaBoost'].append(accuracy_score(y_test, y_pred))
    
    # 打印结果
    print("\n📊 平均准确率对比:")
    print("-" * 40)
    for name, accs in results.items():
        avg_acc = sum(accs) / len(accs)
        print(f"{name:<12}: {avg_acc:.4f}")


def main():
    """
    主函数 - 运行所有演示
    """
    print("\n" + "🌲" * 30)
    print("第十章：集成学习 - 三个臭皮匠顶个诸葛亮")
    print("Ensemble Learning: Wisdom of the Crowd")
    print("🌲" * 30 + "\n")
    
    # 设置随机种子以便复现
    random.seed(42)
    
    # 运行演示
    demo_bagging()
    demo_random_forest()
    demo_adaboost()
    demo_comparison()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
