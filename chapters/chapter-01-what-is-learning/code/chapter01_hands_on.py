"""
第一章：动手实践 - 从零开始的机器学习
包含多个可运行的示例，帮助理解机器学习的核心概念
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random


# ==================== 实践1：预测房屋价格 ====================

class HousePricePredictor:
    """
    房屋价格预测器 - 简单的线性回归实现
    这是监督学习的经典例子
    """
    
    def __init__(self):
        self.weight = 0.0  # 每平米价格
        self.bias = 0.0    # 基础价格
        self.history = []
    
    def fit(self, sizes: List[float], prices: List[float], 
            epochs: int = 100, lr: float = 0.0001):
        """
        训练模型
        sizes: 房屋面积列表
        prices: 对应价格列表
        """
        n = len(sizes)
        
        print(f"开始训练... 数据量: {n}, 轮数: {epochs}")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for size, price in zip(sizes, prices):
                # 预测
                predicted = self.weight * size + self.bias
                
                # 计算误差
                error = predicted - price
                total_loss += error ** 2
                
                # 梯度下降更新
                self.weight -= lr * 2 * error * size
                self.bias -= lr * 2 * error
            
            mse = total_loss / n
            self.history.append(mse)
            
            if (epoch + 1) % 20 == 0:
                print(f"  轮次 {epoch+1}: MSE = {mse:.2f}, "
                      f"weight = {self.weight:.4f}, bias = {self.bias:.2f}")
    
    def predict(self, size: float) -> float:
        """预测房价"""
        return self.weight * size + self.bias
    
    def evaluate(self, sizes: List[float], prices: List[float]) -> float:
        """评估模型性能（返回R²分数）"""
        predictions = [self.predict(s) for s in sizes]
        
        # 计算R²
        mean_price = sum(prices) / len(prices)
        ss_tot = sum((p - mean_price) ** 2 for p in prices)
        ss_res = sum((p - pred) ** 2 for p, pred in zip(prices, predictions))
        
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return r2


def house_price_demo():
    """房屋价格预测演示"""
    print("\n" + "=" * 60)
    print("实践1：预测房屋价格")
    print("=" * 60)
    
    # 模拟数据：面积(平米) vs 价格(万元)
    # 真实关系：价格 = 0.8 * 面积 + 50 + 噪声
    np.random.seed(42)
    sizes = np.random.randint(30, 150, 100).tolist()
    prices = [0.8 * s + 50 + np.random.randn() * 10 for s in sizes]
    
    print(f"\n数据集：{len(sizes)}个房屋样本")
    print("样本数据:")
    for i in range(3):
        print(f"  面积: {sizes[i]}m² → 价格: {prices[i]:.1f}万元")
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(sizes))
    train_sizes, test_sizes = sizes[:train_size], sizes[train_size:]
    train_prices, test_prices = prices[:train_size], prices[train_size:]
    
    # 训练模型
    model = HousePricePredictor()
    model.fit(train_sizes, train_prices, epochs=100, lr=0.0001)
    
    # 评估
    train_r2 = model.evaluate(train_sizes, train_prices)
    test_r2 = model.evaluate(test_sizes, test_prices)
    
    print(f"\n训练集 R²: {train_r2:.4f}")
    print(f"测试集 R²: {test_r2:.4f}")
    
    # 预测新房屋
    new_houses = [60, 90, 120]
    print(f"\n预测新房屋价格:")
    for size in new_houses:
        price = model.predict(size)
        print(f"  {size}m² → 约 {price:.1f}万元")
    
    return model


# ==================== 实践2：垃圾邮件分类器 ====================

class SimpleSpamClassifier:
    """
    简单垃圾邮件分类器 - 基于关键词的朴素贝叶斯
    """
    
    def __init__(self):
        self.spam_words = set()      # 垃圾邮件常见词
        self.ham_words = set()       # 正常邮件常见词
        self.spam_count = 0
        self.ham_count = 0
    
    def fit(self, emails: List[Tuple[str, int]]):
        """
        训练分类器
        emails: [(邮件文本, 标签), ...]  标签: 1=垃圾邮件, 0=正常邮件
        """
        for text, label in emails:
            words = set(text.lower().split())
            
            if label == 1:  # 垃圾邮件
                self.spam_words.update(words)
                self.spam_count += 1
            else:  # 正常邮件
                self.ham_words.update(words)
                self.ham_count += 1
        
        print(f"训练完成: {self.ham_count}封正常邮件, {self.spam_count}封垃圾邮件")
    
    def predict(self, text: str) -> int:
        """预测邮件类型，返回0（正常）或1（垃圾）"""
        words = set(text.lower().split())
        
        spam_score = len(words & self.spam_words)
        ham_score = len(words & self.ham_words)
        
        # 简单的决策规则
        if spam_score > ham_score:
            return 1
        return 0
    
    def evaluate(self, emails: List[Tuple[str, int]]) -> float:
        """计算准确率"""
        correct = 0
        for text, label in emails:
            pred = self.predict(text)
            if pred == label:
                correct += 1
        return correct / len(emails)


def spam_classifier_demo():
    """垃圾邮件分类演示"""
    print("\n" + "=" * 60)
    print("实践2：垃圾邮件分类")
    print("=" * 60)
    
    # 模拟邮件数据
    training_emails = [
        # 正常邮件 (标签=0)
        ("Meeting scheduled for tomorrow at 3pm", 0),
        ("Please review the attached document", 0),
        ("Happy birthday! Have a great day", 0),
        ("Your order has been shipped", 0),
        ("Project update: completed phase 1", 0),
        ("Dinner this weekend?", 0),
        ("Invoice for your recent purchase", 0),
        ("Reminder: doctor appointment tomorrow", 0),
        # 垃圾邮件 (标签=1)
        ("Congratulations! You won a prize! Click here", 1),
        ("URGENT: Claim your free gift now!!!", 1),
        ("Make money fast! Work from home", 1),
        ("You have been selected for a cash reward", 1),
        ("Buy cheap viagra pills online", 1),
        ("Act now! Limited time offer!!!", 1),
        ("Earn $5000 per week from home", 1),
        ("Free lottery ticket! You are a winner!", 1),
    ]
    
    print(f"\n训练集: {len(training_emails)}封邮件")
    
    # 训练
    classifier = SimpleSpamClassifier()
    classifier.fit(training_emails)
    
    # 测试
    test_emails = [
        ("Team meeting rescheduled to Friday", 0),
        ("WINNER! Claim your prize money NOW!!!", 1),
        ("Project deadline extended", 0),
        ("Get rich quick! Free money!!!", 1),
    ]
    
    print(f"\n测试集: {len(test_emails)}封邮件")
    accuracy = classifier.evaluate(test_emails)
    print(f"准确率: {accuracy*100:.1f}%")
    
    # 展示预测
    print(f"\n预测示例:")
    for text, true_label in test_emails:
        pred = classifier.predict(text)
        status = "✓" if pred == true_label else "✗"
        label_name = "垃圾邮件" if pred == 1 else "正常邮件"
        print(f"  {status} \"{text[:40]}...\" → {label_name}")


# ==================== 实践3：鸢尾花分类 ====================

class KNNClassifier:
    """K近邻分类器 - 简单的机器学习算法"""
    
    def __init__(self, k: int = 3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """存储训练数据"""
        self.X_train = X
        self.y_train = y
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测新样本的类别"""
        predictions = []
        
        for x in X:
            # 计算与所有训练样本的距离
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
            # 找到k个最近邻
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            # 投票决定类别
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            prediction = unique[np.argmax(counts)]
            predictions.append(prediction)
        
        return np.array(predictions)


def iris_classification_demo():
    """鸢尾花分类演示（使用模拟数据）"""
    print("\n" + "=" * 60)
    print("实践3：鸢尾花分类 (K近邻算法)")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 模拟鸢尾花数据：3个类别，每个类别2个特征
    # 类别0: 山鸢尾 (花瓣较短)
    # 类别1: 变色鸢尾 (中等)  
    # 类别2: 维吉尼亚鸢尾 (花瓣较长)
    
    n_samples = 30
    
    # 类别0
    X_0 = np.random.randn(n_samples, 2) * 0.5 + [1.5, 0.2]
    y_0 = np.zeros(n_samples)
    
    # 类别1
    X_1 = np.random.randn(n_samples, 2) * 0.5 + [4.0, 1.3]
    y_1 = np.ones(n_samples)
    
    # 类别2
    X_2 = np.random.randn(n_samples, 2) * 0.5 + [5.5, 2.0]
    y_2 = np.full(n_samples, 2)
    
    # 合并数据
    X = np.vstack([X_0, X_1, X_2])
    y = np.concatenate([y_0, y_1, y_2])
    
    # 划分训练集和测试集
    indices = np.random.permutation(len(X))
    train_idx = indices[:70]
    test_idx = indices[70:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"\n数据集: {len(X)}个样本")
    print(f"训练集: {len(X_train)}个")
    print(f"测试集: {len(X_test)}个")
    print(f"类别数: 3 (山鸢尾、变色鸢尾、维吉尼亚鸢尾)")
    
    # 训练KNN模型
    model = KNNClassifier(k=3)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    print(f"\n测试准确率: {accuracy*100:.1f}%")
    
    # 可视化
    plt.figure(figsize=(10, 4))
    
    # 训练集
    plt.subplot(1, 2, 1)
    colors = ['red', 'green', 'blue']
    labels = ['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾']
    for i in range(3):
        mask = y_train == i
        plt.scatter(X_train[mask, 0], X_train[mask, 1], 
                   c=colors[i], label=labels[i], alpha=0.6)
    plt.xlabel('花瓣长度 (cm)')
    plt.ylabel('花瓣宽度 (cm)')
    plt.title('训练集')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 测试集
    plt.subplot(1, 2, 2)
    for i in range(3):
        mask = y_test == i
        plt.scatter(X_test[mask, 0], X_test[mask, 1], 
                   c=colors[i], label=labels[i], alpha=0.6)
    plt.xlabel('花瓣长度 (cm)')
    plt.ylabel('花瓣宽度 (cm)')
    plt.title(f'测试集 (准确率: {accuracy*100:.1f}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('iris_classification.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存: iris_classification.png")
    plt.show()


def main():
    """运行所有实践"""
    print("=" * 70)
    print("第一章：动手实践 - 从零开始的机器学习")
    print("=" * 70)
    
    # 实践1：房价预测
    house_price_demo()
    
    # 实践2：垃圾邮件分类
    spam_classifier_demo()
    
    # 实践3：鸢尾花分类
    try:
        iris_classification_demo()
    except Exception as e:
        print(f"鸢尾花分类演示跳过: {e}")
    
    print("\n" + "=" * 70)
    print("所有实践完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
