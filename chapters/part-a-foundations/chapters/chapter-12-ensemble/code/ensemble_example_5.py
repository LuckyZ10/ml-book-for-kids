import numpy as np
from collections import Counter

class RandomForestClassifier:
    """
    随机森林分类器
    
    原理：Bagging + 特征随机性
    在每个节点分裂时，只考虑随机选择的m_try个特征
    """
    def __init__(self, n_estimators=100, max_depth=10, 
                 min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # 'sqrt', 'log2', 或整数
        self.trees = []
        self.feature_indices = []  # 记录每棵树使用的特征子集
        
    def _get_n_features(self, n_total_features):
        """确定每棵树使用的特征数量"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_total_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_total_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_total_features)
        else:
            return n_total_features
    
    def fit(self, X, y):
        """训练随机森林"""
        n_samples, n_features = X.shape
        n_features_per_tree = self._get_n_features(n_features)
        
        self.trees = []
        self.feature_indices = []
        
        for i in range(self.n_estimators):
            # 1. Bootstrap抽样
            bootstrap_indices = np.random.choice(
                n_samples, size=n_samples, replace=True
            )
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # 2. 随机选择特征子集
            feature_indices = np.random.choice(
                n_features, size=n_features_per_tree, replace=False
            )
            self.feature_indices.append(feature_indices)
            
            X_subset = X_bootstrap[:, feature_indices]
            
            # 3. 训练决策树（使用带特征随机性的版本）
            tree = RandomTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                feature_indices=feature_indices  # 告诉树使用哪些特征
            )
            tree.fit(X_subset, y_bootstrap)
            self.trees.append(tree)
            
        return self
    
    def predict(self, X):
        """预测：所有树投票"""
        predictions = []
        for tree, feature_indices in zip(self.trees, self.feature_indices):
            # 每棵树只看到它训练时用的特征
            X_subset = X[:, feature_indices]
            pred = tree.predict(X_subset)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # 多数投票
        result = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            result.append(Counter(votes).most_common(1)[0][0])
        
        return np.array(result)
    
    def feature_importances(self, X, y):
        """计算特征重要性（置换法）"""
        baseline_accuracy = self._evaluate(X, y)
        n_features = X.shape[1]
        
        importances = []
        for feature in range(n_features):
            # 打乱这一列
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feature])
            
            # 看准确率下降多少
            permuted_accuracy = self._evaluate(X_permuted, y)
            importance = baseline_accuracy - permuted_accuracy
            importances.append(importance)
        
        # 归一化
        importances = np.array(importances)
        importances = importances / np.sum(importances)
        
        return importances
    
    def _evaluate(self, X, y):
        """评估准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


class RandomTree(DecisionTree):
    """带特征随机性的决策树"""
    def __init__(self, max_depth=10, min_samples_split=2, feature_indices=None):
        super().__init__(max_depth, min_samples_split)
        self.feature_indices = feature_indices
    
    def _grow_tree(self, X, y, depth):
        """重写grow_tree，只在指定特征中寻找分裂"""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # 停止条件
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        # 只在给定的特征中寻找最佳分裂
        features_to_try = range(n_features)  # X已经被子集化
        
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in features_to_try:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        if best_gain < 1e-7:
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        # 分裂
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


# ============ 演示：随机森林效果 ============
if __name__ == "__main__":
    from sklearn.datasets import make_classification, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    print("=" * 60)
    print("随机森林分类器演示")
    print("=" * 60)
    
    # 使用Iris数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n数据集: Iris（{X.shape[1]}个特征，{len(np.unique(y))}个类别）")
    print(f"训练集: {len(X_train)}个样本")
    print(f"测试集: {len(X_test)}个样本")
    
    # 随机森林
    print("\n【随机森林 - 100棵树】")
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        max_features='sqrt'  # 每棵树只考虑sqrt(4)=2个特征
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"准确率: {acc_rf:.4f}")
    
    # 特征重要性
    print("\n【特征重要性】")
    importances = rf.feature_importances(X_train, y_train)
    feature_names = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
    for name, imp in zip(feature_names, importances):
        bar = "█" * int(imp * 50)
        print(f"  {name}: {imp:.4f} {bar}")