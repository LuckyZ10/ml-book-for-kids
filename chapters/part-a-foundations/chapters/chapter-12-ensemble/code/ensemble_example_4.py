import numpy as np
from collections import Counter

class DecisionTree:
    """简化版决策树，用于集成学习"""
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self._grow_tree(X, y, depth=0)
        return self
    
    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # 停止条件
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            # 返回叶节点（多数类）
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        # 寻找最佳分裂
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        if best_gain < 1e-7:  # 无法进一步分裂
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        # 分裂数据
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx
        
        left_subtree = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _information_gain(self, X, y, feature, threshold):
        """计算信息增益"""
        parent_entropy = self._entropy(y)
        
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx
        
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return 0
        
        n = len(y)
        n_left = np.sum(left_idx)
        n_right = np.sum(right_idx)
        
        child_entropy = (n_left / n * self._entropy(y[left_idx]) +
                        n_right / n * self._entropy(y[right_idx]))
        
        return parent_entropy - child_entropy
    
    def _entropy(self, y):
        """计算熵"""
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy
    
    def predict_one(self, x, node):
        """预测单个样本"""
        if node['leaf']:
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self.predict_one(x, node['left'])
        else:
            return self.predict_one(x, node['right'])
    
    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])


class BaggingClassifier:
    """
    Bagging分类器（Bootstrap Aggregating）
    
    原理：通过Bootstrap抽样创建多个训练集，在每个上训练基学习器，最后投票
    """
    def __init__(self, n_estimators=10, max_depth=10, 
                 min_samples_split=2, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.models = []
        
    def fit(self, X, y):
        """训练Bagging集成"""
        n_samples = X.shape[0]
        self.models = []
        
        for i in range(self.n_estimators):
            # Bootstrap抽样
            if self.bootstrap:
                # 有放回抽样
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
            else:
                # 无放回抽样（使用全部数据）
                indices = np.arange(n_samples)
            
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # 训练基学习器
            model = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            model.fit(X_bootstrap, y_bootstrap)
            self.models.append(model)
            
        return self
    
    def predict(self, X):
        """预测：所有基学习器投票"""
        # 收集所有模型的预测
        predictions = np.array([model.predict(X) for model in self.models])
        
        # 对每个样本进行多数投票
        result = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            # 统计每个类别的票数
            vote_counts = Counter(votes)
            # 选择票数最多的类别
            result.append(vote_counts.most_common(1)[0][0])
        
        return np.array(result)
    
    def predict_proba(self, X):
        """预测概率（各类别得票比例）"""
        predictions = np.array([model.predict(X) for model in self.models])
        
        probabilities = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            vote_counts = Counter(votes)
            # 转换为概率
            total = sum(vote_counts.values())
            proba = {cls: count/total for cls, count in vote_counts.items()}
            probabilities.append(proba)
        
        return probabilities


# ============ 演示：Bagging效果 ============
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # 创建数据集
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5,
        n_redundant=3, n_classes=2, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print("=" * 60)
    print("Bagging分类器演示")
    print("=" * 60)
    
    # 1. 单个决策树
    print("\n【单个决策树】")
    single_tree = DecisionTree(max_depth=10)
    single_tree.fit(X_train, y_train)
    y_pred_single = single_tree.predict(X_test)
    acc_single = accuracy_score(y_test, y_pred_single)
    print(f"准确率: {acc_single:.4f}")
    
    # 2. Bagging（10棵树）
    print("\n【Bagging - 10棵树】")
    bagging = BaggingClassifier(n_estimators=10, max_depth=10)
    bagging.fit(X_train, y_train)
    y_pred_bagging = bagging.predict(X_test)
    acc_bagging = accuracy_score(y_test, y_pred_bagging)
    print(f"准确率: {acc_bagging:.4f}")
    print(f"提升: +{(acc_bagging - acc_single)*100:.2f}%")
    
    # 3. Bagging（50棵树）
    print("\n【Bagging - 50棵树】")
    bagging50 = BaggingClassifier(n_estimators=50, max_depth=10)
    bagging50.fit(X_train, y_train)
    y_pred_bagging50 = bagging50.predict(X_test)
    acc_bagging50 = accuracy_score(y_test, y_pred_bagging50)
    print(f"准确率: {acc_bagging50:.4f}")
    print(f"相比单棵树提升: +{(acc_bagging50 - acc_single)*100:.2f}%")