"""
第十二章：集成学习——三个臭皮匠顶个诸葛亮
完整代码实现

包含：
1. Bootstrap采样实现
2. Bagging分类器（基于决策树）
3. 随机森林分类器
4. AdaBoost分类器
5. 梯度提升（简化版）
6. 特征重要性计算
7. 可视化工具
"""

import numpy as np
from collections import Counter
from typing import List, Tuple, Optional, Callable
import random


# ============================================================
# 工具函数
# ============================================================

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                     random_state: Optional[int] = None) -> Tuple:
    """分割数据集为训练集和测试集"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return (X[train_indices], X[test_indices], y[train_indices], y[test_indices])


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算准确率"""
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算均方误差"""
    return np.mean((y_true - y_pred) ** 2)


# ============================================================
# 决策树（基学习器）
# ============================================================

class DecisionTree:
    """
    决策树基类
    支持分类和回归任务
    """
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, criterion: str = 'gini',
                 max_features: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion  # 'gini', 'entropy', 'mse'
        self.max_features = max_features
        self.tree = None
        self.feature_importances_ = None
        self.n_features = None
        
    def _gini(self, y: np.ndarray) -> float:
        """计算基尼不纯度"""
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _entropy(self, y: np.ndarray) -> float:
        """计算信息熵"""
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        proportions = proportions[proportions > 0]
        return -np.sum(proportions * np.log2(proportions))
    
    def _mse(self, y: np.ndarray) -> float:
        """计算均方误差"""
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)
    
    def _impurity(self, y: np.ndarray) -> float:
        """根据标准计算不纯度"""
        if self.criterion == 'gini':
            return self._gini(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:  # mse
            return self._mse(y)
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """找到最佳分裂点和特征"""
        n_samples, n_features = X.shape
        
        # 选择要考虑的特征
        if self.max_features is not None:
            feature_indices = np.random.choice(n_features, 
                                               min(self.max_features, n_features), 
                                               replace=False)
        else:
            feature_indices = range(n_features)
        
        best_impurity_reduction = 0
        best_feature = None
        best_threshold = None
        
        parent_impurity = self._impurity(y)
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                left_impurity = self._impurity(y[left_mask])
                right_impurity = self._impurity(y[right_mask])
                
                # 加权平均不纯度
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / n_samples
                
                impurity_reduction = parent_impurity - weighted_impurity
                
                if impurity_reduction > best_impurity_reduction:
                    best_impurity_reduction = impurity_reduction
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_impurity_reduction
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        """递归构建决策树"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # 停止条件
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            if self.criterion == 'mse':
                return {'leaf': True, 'prediction': np.mean(y)}
            else:
                return {'leaf': True, 'prediction': Counter(y).most_common(1)[0][0]}
        
        feature, threshold, impurity_reduction = self._best_split(X, y)
        
        if feature is None:
            if self.criterion == 'mse':
                return {'leaf': True, 'prediction': np.mean(y)}
            else:
                return {'leaf': True, 'prediction': Counter(y).most_common(1)[0][0]}
        
        # 分裂数据
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'impurity_reduction': impurity_reduction,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """训练决策树"""
        self.n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features)
        self.tree = self._build_tree(X, y, depth=0)
        self._compute_feature_importance(self.tree, X.shape[0])
        # 归一化
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
        return self
    
    def _compute_feature_importance(self, node: dict, total_samples: int):
        """计算特征重要性"""
        if node['leaf']:
            return
        
        # 当前节点的贡献
        self.feature_importances_[node['feature']] += \
            node.get('impurity_reduction', 0) * total_samples
        
        # 递归计算子树
        self._compute_feature_importance(node['left'], total_samples)
        self._compute_feature_importance(node['right'], total_samples)
    
    def _predict_single(self, x: np.ndarray, node: dict) -> float:
        """预测单个样本"""
        if node['leaf']:
            return node['prediction']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测多个样本"""
        return np.array([self._predict_single(x, self.tree) for x in X])


# ============================================================
# Bagging分类器
# ============================================================

class BaggingClassifier:
    """
    Bagging（装袋）分类器
    使用Bootstrap采样训练多个基学习器，最后投票
    """
    
    def __init__(self, n_estimators: int = 10, max_samples: float = 1.0,
                 max_features: float = 1.0, bootstrap: bool = True,
                 bootstrap_features: bool = False, random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.random_state = random_state
        self.estimators = []
        self.estimators_features = []
        
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """生成Bootstrap样本"""
        n_samples = X.shape[0]
        n_bootstrap = int(n_samples * self.max_samples)
        
        if self.bootstrap:
            indices = np.random.choice(n_samples, n_bootstrap, replace=True)
        else:
            indices = np.random.choice(n_samples, n_bootstrap, replace=False)
        
        return X[indices], y[indices], indices
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaggingClassifier':
        """训练Bagging分类器"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        n_features_bootstrap = int(n_features * self.max_features)
        
        self.estimators = []
        self.estimators_features = []
        self.classes_ = np.unique(y)
        
        for i in range(self.n_estimators):
            # Bootstrap采样
            X_boot, y_boot, _ = self._bootstrap_sample(X, y)
            
            # 特征采样
            if self.bootstrap_features:
                features = np.random.choice(n_features, n_features_bootstrap, replace=False)
            else:
                features = np.arange(n_features)
            
            X_boot = X_boot[:, features]
            
            # 训练基学习器
            estimator = DecisionTree(max_depth=10, min_samples_split=2)
            estimator.fit(X_boot, y_boot)
            
            self.estimators.append(estimator)
            self.estimators_features.append(features)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        # 收集所有基学习器的预测
        predictions = np.zeros((X.shape[0], self.n_estimators))
        
        for i, (estimator, features) in enumerate(zip(self.estimators, self.estimators_features)):
            predictions[:, i] = estimator.predict(X[:, features])
        
        # 多数投票
        final_predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            final_predictions[i] = Counter(predictions[i]).most_common(1)[0][0]
        
        return final_predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率（通过投票比例）"""
        predictions = np.zeros((X.shape[0], self.n_estimators))
        
        for i, (estimator, features) in enumerate(zip(self.estimators, self.estimators_features)):
            predictions[:, i] = estimator.predict(X[:, features])
        
        # 计算概率
        n_classes = len(self.classes_)
        proba = np.zeros((X.shape[0], n_classes))
        
        for i in range(X.shape[0]):
            for j, cls in enumerate(self.classes_):
                proba[i, j] = np.sum(predictions[i] == cls) / self.n_estimators
        
        return proba


# ============================================================
# 随机森林分类器
# ============================================================

class RandomForestClassifier:
    """
    随机森林分类器
    在Bagging基础上增加特征随机选择
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: Optional[str] = 'sqrt', bootstrap: bool = True,
                 oob_score: bool = False, random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth if max_depth is not None else 10
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features  # 'sqrt', 'log2', int, or None
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        
        self.estimators = []
        self.classes_ = None
        self.n_features = None
        self.feature_importances_ = None
        self.oob_score_ = None
        self.oob_predictions = None
    
    def _get_max_features(self) -> int:
        """确定每次分裂要考虑的特征数"""
        if self.max_features is None:
            return self.n_features
        elif self.max_features == 'sqrt':
            return int(np.sqrt(self.n_features))
        elif self.max_features == 'log2':
            return int(np.log2(self.n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, self.n_features)
        elif isinstance(self.max_features, float):
            return int(self.max_features * self.n_features)
        else:
            return self.n_features
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        """训练随机森林"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, self.n_features = X.shape
        self.classes_ = np.unique(y)
        self.estimators = []
        
        max_features = self._get_max_features()
        
        # 用于OOB评估
        if self.oob_score:
            oob_predictions = {i: [] for i in range(n_samples)}
        
        for i in range(self.n_estimators):
            # Bootstrap采样
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
            else:
                indices = np.arange(n_samples)
            
            X_boot = X[indices]
            y_boot = y[indices]
            
            # 训练决策树，使用随机特征选择
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                criterion='gini'
            )
            tree.fit(X_boot, y_boot)
            self.estimators.append(tree)
            
            # 记录OOB样本
            if self.oob_score:
                oob_mask = np.bincount(indices, minlength=n_samples) == 0
                oob_indices = np.where(oob_mask)[0]
                if len(oob_indices) > 0:
                    oob_pred = tree.predict(X[oob_indices])
                    for idx, pred in zip(oob_indices, oob_pred):
                        oob_predictions[idx].append(pred)
        
        # 计算OOB分数
        if self.oob_score:
            oob_preds = []
            oob_true = []
            for i in range(n_samples):
                if len(oob_predictions[i]) > 0:
                    oob_preds.append(Counter(oob_predictions[i]).most_common(1)[0][0])
                    oob_true.append(y[i])
            
            if len(oob_preds) > 0:
                self.oob_score_ = accuracy_score(np.array(oob_true), np.array(oob_preds))
        
        # 计算特征重要性
        self._compute_feature_importances()
        
        return self
    
    def _compute_feature_importances(self):
        """计算平均特征重要性"""
        importances = np.zeros(self.n_features)
        for tree in self.estimators:
            importances += tree.feature_importances_
        self.feature_importances_ = importances / self.n_estimators
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        predictions = np.array([tree.predict(X) for tree in self.estimators])
        
        # 多数投票
        final_predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            final_predictions[i] = Counter(predictions[:, i]).most_common(1)[0][0]
        
        return final_predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        predictions = np.array([tree.predict(X) for tree in self.estimators])
        
        n_classes = len(self.classes_)
        proba = np.zeros((X.shape[0], n_classes))
        
        for i in range(X.shape[0]):
            for j, cls in enumerate(self.classes_):
                proba[i, j] = np.sum(predictions[:, i] == cls) / self.n_estimators
        
        return proba


# ============================================================
# AdaBoost分类器
# ============================================================

class AdaBoostClassifier:
    """
    AdaBoost分类器
    自适应提升算法
    """
    
    def __init__(self, n_estimators: int = 50, learning_rate: float = 1.0,
                 random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.estimators = []
        self.estimator_weights = []
        self.classes_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoostClassifier':
        """训练AdaBoost分类器"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        
        # 初始化样本权重（相等）
        sample_weights = np.ones(n_samples) / n_samples
        
        self.estimators = []
        self.estimator_weights = []
        
        for i in range(self.n_estimators):
            # 根据权重采样
            indices = np.random.choice(n_samples, n_samples, replace=True, p=sample_weights)
            X_sampled = X[indices]
            y_sampled = y[indices]
            
            # 训练弱学习器（决策树桩）
            estimator = DecisionTree(max_depth=1, min_samples_split=2)
            estimator.fit(X_sampled, y_sampled)
            
            # 在整个数据集上预测
            predictions = estimator.predict(X)
            
            # 计算加权错误率
            incorrect = (predictions != y).astype(float)
            error = np.sum(sample_weights * incorrect)
            
            # 如果错误率太高，跳过
            if error >= 1 - 1/len(self.classes_):
                continue
            
            # 计算分类器权重
            estimator_weight = self.learning_rate * 0.5 * np.log((1 - error) / max(error, 1e-10))
            
            # 更新样本权重
            sample_weights *= np.exp(-estimator_weight * (2 * (predictions == y).astype(float) - 1))
            sample_weights /= np.sum(sample_weights)  # 归一化
            
            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if len(self.estimators) == 0:
            raise ValueError("模型尚未训练")
        
        # 收集加权预测
        predictions = np.zeros((X.shape[0], len(self.classes_)))
        
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            preds = estimator.predict(X)
            for i, pred in enumerate(preds):
                class_idx = np.where(self.classes_ == pred)[0][0]
                predictions[i, class_idx] += weight
        
        # 选择权重最高的类别
        return self.classes_[np.argmax(predictions, axis=1)]


# ============================================================
# 简化版梯度提升（回归）
# ============================================================

class GradientBoostingRegressor:
    """
    简化版梯度提升回归器
    使用平方损失
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.estimators = []
        self.initial_prediction = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingRegressor':
        """训练梯度提升回归器"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 初始化：预测目标的均值
        self.initial_prediction = np.mean(y)
        current_pred = np.full(len(y), self.initial_prediction)
        
        self.estimators = []
        
        for i in range(self.n_estimators):
            # 计算负梯度（残差）
            residuals = y - current_pred
            
            # 训练回归树拟合残差
            tree = DecisionTree(max_depth=self.max_depth, criterion='mse')
            tree.fit(X, residuals)
            
            # 更新预测
            tree_pred = tree.predict(X)
            current_pred += self.learning_rate * tree_pred
            
            self.estimators.append(tree)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        predictions = np.full(X.shape[0], self.initial_prediction)
        
        for tree in self.estimators:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions


# ============================================================
# 可视化工具
# ============================================================

def print_ascii_tree(tree: DecisionTree, feature_names: Optional[List[str]] = None,
                     max_depth: int = 5) -> str:
    """打印决策树的ASCII表示"""
    
    def _node_to_str(node: dict, depth: int, prefix: str = "") -> str:
        indent = "    " * depth
        
        if node['leaf']:
            return f"{indent}{prefix}└── 预测: {node['prediction']:.3f}\n"
        
        feature_name = f"特征{node['feature']}"
        if feature_names and node['feature'] < len(feature_names):
            feature_name = feature_names[node['feature']]
        
        result = f"{indent}{prefix}├── {feature_name} <= {node['threshold']:.3f}?\n"
        
        if depth < max_depth:
            result += _node_to_str(node['left'], depth + 1, "是: ")
            result += _node_to_str(node['right'], depth + 1, "否: ")
        else:
            result += f"{indent}    └── ... (深度限制)\n"
        
        return result
    
    return _node_to_str(tree.tree, 0)


def print_feature_importance(importances: np.ndarray, 
                             feature_names: Optional[List[str]] = None) -> str:
    """打印特征重要性"""
    
    if feature_names is None:
        feature_names = [f"特征{i}" for i in range(len(importances))]
    
    # 排序
    indices = np.argsort(importances)[::-1]
    
    result = "\n=== 特征重要性 ===\n\n"
    result += f"{'排名':<6}{'特征':<20}{'重要性':<12}{'柱状图'}\n"
    result += "-" * 60 + "\n"
    
    for rank, idx in enumerate(indices[:10], 1):
        name = feature_names[idx][:18]
        imp = importances[idx]
        bar = "█" * int(imp * 30)
        result += f"{rank:<6}{name:<20}{imp:<12.4f}{bar}\n"
    
    return result


def plot_decision_boundary(X: np.ndarray, y: np.ndarray, model, 
                           title: str = "决策边界") -> str:
    """
    使用ASCII字符绘制二维决策边界
    简化版可视化
    """
    if X.shape[1] != 2:
        return "只能可视化二维数据"
    
    # 创建网格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    grid_size = 20
    xx = np.linspace(x_min, x_max, grid_size)
    yy = np.linspace(y_min, y_max, grid_size)
    
    result = f"\n=== {title} ===\n\n"
    result += "    " + "".join([f"{x:.1f}"[-3] for x in xx]) + "\n"
    result += "    " + "-" * grid_size + "\n"
    
    symbols = ['.', 'o', '+', 'x', '*']
    
    for yi in reversed(yy):
        row = f"{yi:.1f}"[-3] + " |"
        for xi in xx:
            pred = model.predict(np.array([[xi, yi]]))[0]
            # 找到最近的样本点
            distances = np.sqrt((X[:, 0] - xi)**2 + (X[:, 1] - yi)**2)
            if np.min(distances) < 0.3:
                nearest_idx = np.argmin(distances)
                row += symbols[int(y[nearest_idx]) % len(symbols)]
            else:
                row += symbols[int(pred) % len(symbols)].upper()
        result += row + "|\n"
    
    result += "    " + "-" * grid_size + "\n"
    return result


# ============================================================
# 演示函数
# ============================================================

def demo_bagging():
    """演示Bagging"""
    print("=" * 60)
    print("Bagging分类器演示")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # 单棵决策树
    print("\n单棵决策树:")
    single_tree = DecisionTree(max_depth=5)
    single_tree.fit(X_train, y_train)
    single_pred = single_tree.predict(X_test)
    single_acc = accuracy_score(y_test, single_pred)
    print(f"  准确率: {single_acc:.4f}")
    
    # Bagging
    print("\nBagging (10棵树):")
    bagging = BaggingClassifier(n_estimators=10, random_state=42)
    bagging.fit(X_train, y_train)
    bagging_pred = bagging.predict(X_test)
    bagging_acc = accuracy_score(y_test, bagging_pred)
    print(f"  准确率: {bagging_acc:.4f}")
    
    print(f"\n提升幅度: {(bagging_acc - single_acc) / single_acc * 100:.2f}%")


def demo_random_forest():
    """演示随机森林"""
    print("\n" + "=" * 60)
    print("随机森林分类器演示")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    X = np.random.randn(200, 5)
    # 第0、2特征最重要
    y = ((X[:, 0] > 0.5) & (X[:, 2] < -0.3)).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    print("\n随机森林 (100棵树):")
    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', 
                                oob_score=True, random_state=42)
    rf.fit(X_train, y_train)
    
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    print(f"  测试准确率: {rf_acc:.4f}")
    if rf.oob_score_ is not None:
        print(f"  OOB分数: {rf.oob_score_:.4f}")
    
    # 特征重要性
    print(print_feature_importance(rf.feature_importances_, 
                                   [f"特征{i}" for i in range(5)]))


def demo_adaboost():
    """演示AdaBoost"""
    print("\n" + "=" * 60)
    print("AdaBoost分类器演示")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    X = np.random.randn(150, 2)
    # 创建一个复杂的决策边界
    y = (((X[:, 0] > 0) & (X[:, 1] > 0)) | 
         ((X[:, 0] < 0) & (X[:, 1] < 0))).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    print("\nAdaBoost (50个决策树桩):")
    adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, 
                                  random_state=42)
    adaboost.fit(X_train, y_train)
    
    adaboost_pred = adaboost.predict(X_test)
    adaboost_acc = accuracy_score(y_test, adaboost_pred)
    
    print(f"  测试准确率: {adaboost_acc:.4f}")
    print(f"  使用的弱学习器数量: {len(adaboost.estimators)}")


def demo_gradient_boosting():
    """演示梯度提升"""
    print("\n" + "=" * 60)
    print("梯度提升回归器演示")
    print("=" * 60)
    
    # 生成回归数据
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 3 * X[:, 0] ** 2 + 2 * X[:, 0] + 1 + np.random.randn(100) * 0.5
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    print("\n梯度提升回归 (100棵树, 深度3):")
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                   max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    
    y_pred = gb.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"  测试集MSE: {mse:.4f}")
    
    # 展示部分预测
    print("\n  前5个样本的预测 vs 真实值:")
    for i in range(min(5, len(y_test))):
        print(f"    真实: {y_test[i]:.3f}, 预测: {y_pred[i]:.3f}, "
              f"误差: {abs(y_test[i] - y_pred[i]):.3f}")


def demo_comparison():
    """比较所有方法"""
    print("\n" + "=" * 60)
    print("所有方法对比")
    print("=" * 60)
    
    # 生成更复杂的数据
    np.random.seed(42)
    n_samples = 300
    X = np.random.randn(n_samples, 4)
    # XOR-like pattern
    y = ((X[:, 0] * X[:, 1] > 0) ^ (X[:, 2] * X[:, 3] > 0)).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    results = []
    
    # 单棵决策树
    dt = DecisionTree(max_depth=10)
    dt.fit(X_train, y_train)
    acc = accuracy_score(y_test, dt.predict(X_test))
    results.append(("决策树", acc))
    
    # Bagging
    bag = BaggingClassifier(n_estimators=20, random_state=42)
    bag.fit(X_train, y_train)
    acc = accuracy_score(y_test, bag.predict(X_test))
    results.append(("Bagging", acc))
    
    # 随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    results.append(("随机森林", acc))
    
    # AdaBoost
    ada = AdaBoostClassifier(n_estimators=50, random_state=42)
    ada.fit(X_train, y_train)
    acc = accuracy_score(y_test, ada.predict(X_test))
    results.append(("AdaBoost", acc))
    
    print("\n准确率对比:")
    print(f"{'方法':<15}{'准确率':<10}{'柱状图'}")
    print("-" * 40)
    
    for name, acc in sorted(results, key=lambda x: x[1], reverse=True):
        bar = "█" * int(acc * 30)
        print(f"{name:<15}{acc:<10.4f}{bar}")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "第十二章：集成学习" + " " * 25 + "║")
    print("║" + " " * 10 + "三个臭皮匠顶个诸葛亮" + " " * 25 + "║")
    print("╚" + "=" * 58 + "╝")
    
    demo_bagging()
    demo_random_forest()
    demo_adaboost()
    demo_gradient_boosting()
    demo_comparison()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
