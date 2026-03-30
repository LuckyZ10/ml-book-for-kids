"""
三种集成方法对比实验
"""
import numpy as np
import time

# 导入我们手写的实现
from bagging_classifier import BaggingClassifier, DecisionTree
from random_forest_classifier import RandomForestClassifier
from adaboost_classifier import AdaBoostClassifier

from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def compare_methods(X, y, dataset_name):
    """对比三种集成方法"""
    print("\n" + "=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}, 类别数: {len(np.unique(y))}")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    results = []
    
    # 1. 单棵决策树（基准）
    print("\n【基准：单棵决策树】")
    start = time.time()
    tree = DecisionTree(max_depth=10)
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    acc = accuracy_score(y_test, pred)
    elapsed = time.time() - start
    print(f"  准确率: {acc:.4f} | 训练时间: {elapsed:.3f}s")
    results.append(('单棵决策树', acc, elapsed))
    
    # 2. Bagging
    print("\n【Bagging (50棵树)】")
    start = time.time()
    bagging = BaggingClassifier(n_estimators=50, max_depth=10)
    bagging.fit(X_train, y_train)
    pred = bagging.predict(X_test)
    acc = accuracy_score(y_test, pred)
    elapsed = time.time() - start
    print(f"  准确率: {acc:.4f} | 训练时间: {elapsed:.3f}s")
    results.append(('Bagging', acc, elapsed))
    
    # 3. 随机森林
    print("\n【随机森林 (100棵树)】")
    start = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, max_features='sqrt')
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    elapsed = time.time() - start
    print(f"  准确率: {acc:.4f} | 训练时间: {elapsed:.3f}s")
    results.append(('随机森林', acc, elapsed))
    
    # 4. AdaBoost（仅用于二分类）
    if len(np.unique(y)) == 2:
        print("\n【AdaBoost (50个树桩)】")
        start = time.time()
        ada = AdaBoostClassifier(n_estimators=50)
        ada.fit(X_train, y_train)
        pred = ada.predict(X_test)
        acc = accuracy_score(y_test, pred)
        elapsed = time.time() - start
        print(f"  准确率: {acc:.4f} | 训练时间: {elapsed:.3f}s")
        results.append(('AdaBoost', acc, elapsed))
    
    # 总结
    print("\n【结果总结】")
    print("-" * 50)
    print(f"{'方法':<15} {'准确率':<10} {'时间(s)':<10}")
    print("-" * 50)
    for name, acc, t in results:
        print(f"{name:<15} {acc:<10.4f} {t:<10.3f}")
    print("-" * 50)
    
    best = max(results, key=lambda x: x[1])
    print(f"🏆 最佳方法: {best[0]} (准确率: {best[1]:.4f})")
    
    return results


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("██" + " " * 66 + "██")
    print("██" + "  集成学习方法对比实验".center(62) + "██")
    print("██" + " " * 66 + "██")
    print("█" * 70)
    
    # 实验1：标准分类数据集
    X1, y1 = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_classes=2, random_state=42
    )
    compare_methods(X1, y1, "合成二分类数据 (1000样本, 20特征)")
    
    # 实验2：多分类数据集
    X2, y2 = make_classification(
        n_samples=500, n_features=10, n_informative=8,
        n_redundant=2, n_classes=3, random_state=42
    )
    compare_methods(X2, y2, "合成三分类数据 (500样本, 10特征)")
    
    # 实验3：非线性数据集（月牙形）
    X3, y3 = make_moons(n_samples=500, noise=0.25, random_state=42)
    compare_methods(X3, y3, "月牙形非线性数据 (500样本, 2特征)")
    
    print("\n" + "█" * 70)
    print("实验完成！")
    print("█" * 70)