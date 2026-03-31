import numpy as np
from collections import Counter

class DecisionStump:
    """
    决策树桩（Decision Stump）
    
    只有一层的决策树，用作AdaBoost的弱学习器
    """
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.polarity = 1  # 1 或 -1，表示分类方向
        self.alpha = None  # 这个学习器的权重
        
    def fit(self, X, y, weights):
        """
        找到最佳的树桩
        
        参数：
            X: 特征矩阵
            y: 标签（假设为 -1 或 +1）
            weights: 样本权重
        """
        n_samples, n_features = X.shape
        
        min_error = float('inf')
        
        # 遍历所有特征
        for feature in range(n_features):
            feature_values = X[:, feature]
            thresholds = np.unique(feature_values)
            
            # 遍历所有可能的分裂点
            for threshold in thresholds:
                # 尝试两种分类方向
                for polarity in [1, -1]:
                    # 预测
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[feature_values < threshold] = -1
                    else:
                        predictions[feature_values >= threshold] = -1
                    
                    # 计算加权错误率
                    error = np.sum(weights[y != predictions])
                    
                    # 更新最佳树桩
                    if error < min_error:
                        min_error = error
                        self.feature = feature
                        self.threshold = threshold
                        self.polarity = polarity
        
        return min_error
    
    def predict(self, X):
        """预测"""
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        
        feature_values = X[:, self.feature]
        
        if self.polarity == 1:
            predictions[feature_values < self.threshold] = -1
        else:
            predictions[feature_values >= self.threshold] = -1
        
        return predictions


class AdaBoostClassifier:
    """
    AdaBoost分类器
    
    自适应Boosting算法，顺序训练弱学习器，重点关注前一轮分类错误的样本
    
    参考：Freund, Y., & Schapire, R. E. (1997). A decision-theoretic 
    generalization of on-line learning and an application to boosting. 
    Journal of Computer and System Sciences, 55(1), 119-139.
    """
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []  # 每个学习器的权重
        
    def fit(self, X, y):
        """
        训练AdaBoost
        
        参数：
            X: 特征矩阵
            y: 标签（将被转换为 -1 或 +1）
        """
        n_samples = X.shape[0]
        
        # 转换标签为 -1 和 +1
        self.classes = np.unique(y)
        if len(self.classes) != 2:
            raise ValueError("AdaBoost当前只支持二分类问题")
        
        # 映射到 -1, +1
        y_transformed = np.where(y == self.classes[0], -1, 1)
        
        # 初始化样本权重（均匀分布）
        weights = np.ones(n_samples) / n_samples
        
        self.stumps = []
        self.alphas = []
        
        for t in range(self.n_estimators):
            # 1. 训练弱学习器
            stump = DecisionStump()
            error = stump.fit(X, y_transformed, weights)
            
            # 如果错误率太高，跳过
            if error > 0.5:
                continue
            
            # 2. 计算学习器的权重 alpha
            # alpha = 0.5 * ln((1-error) / error)
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            stump.alpha = alpha
            
            self.stumps.append(stump)
            self.alphas.append(alpha)
            
            # 3. 更新样本权重
            predictions = stump.predict(X)
            
            # w_i = w_i * exp(-alpha * y_i * h(x_i))
            # 如果预测正确：y_i * h(x_i) = 1，权重乘以 exp(-alpha) < 1（减小）
            # 如果预测错误：y_i * h(x_i) = -1，权重乘以 exp(alpha) > 1（增加）
            weights *= np.exp(-alpha * y_transformed * predictions)
            
            # 归一化
            weights /= np.sum(weights)
            
            # 打印进度
            if (t + 1) % 10 == 0:
                train_pred = self._predict_with_current_stumps(X, t + 1)
                accuracy = np.mean(train_pred == y_transformed)
                print(f"  轮数 {t+1}: 错误率={error:.4f}, alpha={alpha:.4f}, 训练准确率={accuracy:.4f}")
        
        return self
    
    def _predict_with_current_stumps(self, X, n_stumps):
        """使用当前已训练的树桩进行预测"""
        n_samples = X.shape[0]
        ensemble_pred = np.zeros(n_samples)
        
        for i in range(n_stumps):
            ensemble_pred += self.alphas[i] * self.stumps[i].predict(X)
        
        return np.sign(ensemble_pred)
    
    def predict(self, X):
        """预测"""
        n_samples = X.shape[0]
        ensemble_pred = np.zeros(n_samples)
        
        # 加权投票
        for stump, alpha in zip(self.stumps, self.alphas):
            ensemble_pred += alpha * stump.predict(X)
        
        # 转换为原始标签
        predictions = np.sign(ensemble_pred)
        return np.where(predictions == -1, self.classes[0], self.classes[1])
    
    def predict_proba(self, X):
        """预测概率（基于加权投票的强度）"""
        n_samples = X.shape[0]
        ensemble_pred = np.zeros(n_samples)
        
        for stump, alpha in zip(self.stumps, self.alphas):
            ensemble_pred += alpha * stump.predict(X)
        
        # 使用sigmoid转换为概率
        proba_class1 = 1 / (1 + np.exp(-ensemble_pred))
        
        return np.column_stack([1 - proba_class1, proba_class1])


# ============ 演示：AdaBoost效果 ============
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    print("=" * 60)
    print("AdaBoost分类器演示")
    print("=" * 60)
    
    # 创建一个稍微复杂的数据集
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    y = y * 2 - 1  # 转换为 -1 和 +1
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n数据集: make_moons（月牙形数据，线性不可分）")
    print(f"训练集: {len(X_train)}个样本")
    print(f"测试集: {len(X_test)}个样本")
    
    # 单个决策树桩
    print("\n【单个决策树桩】")
    single_stump = DecisionStump()
    weights = np.ones(len(X_train)) / len(X_train)
    error = single_stump.fit(X_train, y_train, weights)
    y_pred_stump = single_stump.predict(X_test)
    acc_stump = accuracy_score(y_test, y_pred_stump)
    print(f"错误率: {error:.4f}")
    print(f"测试准确率: {acc_stump:.4f}")
    
    # AdaBoost
    print("\n【AdaBoost - 50个树桩】")
    ada = AdaBoostClassifier(n_estimators=50)
    ada.fit(X_train, y_train)
    y_pred_ada = ada.predict(X_test)
    acc_ada = accuracy_score(y_test, y_pred_ada)
    print(f"\n最终测试准确率: {acc_ada:.4f}")
    print(f"相比单个树桩提升: +{(acc_ada - acc_stump)*100:.2f}%")
    
    print("\n【学到的弱学习器】")
    print(f"共训练了 {len(ada.stumps)} 个决策树桩")
    print("前5个树桩的信息：")
    for i, stump in enumerate(ada.stumps[:5]):
        print(f"  树桩{i+1}: 特征{stump.feature}, 阈值={stump.threshold:.3f}, "
              f"方向={stump.polarity}, 权重alpha={stump.alpha:.4f}")