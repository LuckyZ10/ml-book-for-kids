"""
第一章：什么是机器学习？
基础代码实现 - 从数据中学习模式
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
import random

# ==================== 1. 学习的本质 ====================

def learning_vs_programming_example():
    """
    演示传统编程 vs 机器学习的区别
    传统编程：人写规则 → 计算机执行
    机器学习：人给数据+答案 → 计算机学规则
    """
    print("=" * 60)
    print("学习的本质：从数据中发现规律")
    print("=" * 60)
    
    # 传统编程方式：人工编写规则
    def traditional_approach(fruit_features):
        """传统方法：人工定义规则"""
        color, size, texture = fruit_features
        
        # 人工编写的规则
        if color == 'red' and size == 'small':
            return '苹果'
        elif color == 'yellow' and size == 'large':
            return '香蕉'
        elif color == 'purple' and texture == 'smooth':
            return '葡萄'
        else:
            return '未知'
    
    # 测试传统方法
    test_fruits = [
        ('red', 'small', 'smooth'),
        ('yellow', 'large', 'smooth'),
        ('purple', 'small', 'smooth'),
    ]
    
    print("\n【传统编程方式】")
    print("规则由人工编写：")
    for fruit in test_fruits:
        result = traditional_approach(fruit)
        print(f"  特征{color=}, {size=}, {texture=} → 预测: {result}")
    
    # 机器学习方式：从数据学习
    print("\n【机器学习方式】")
    print("规则从数据中学习（模拟）...")
    
    # 模拟训练数据
    training_data = [
        (('red', 'small', 'smooth'), '苹果'),
        (('red', 'small', 'smooth'), '苹果'),
        (('red', 'medium', 'smooth'), '苹果'),
        (('yellow', 'large', 'smooth'), '香蕉'),
        (('yellow', 'large', 'smooth'), '香蕉'),
        (('purple', 'small', 'smooth'), '葡萄'),
        (('purple', 'small', 'smooth'), '葡萄'),
    ]
    
    # 简单统计学习
    from collections import defaultdict
    feature_to_label = defaultdict(lambda: defaultdict(int))
    
    for features, label in training_data:
        feature_key = tuple(sorted(features))
        feature_to_label[feature_key][label] += 1
    
    print(f"  从{len(training_data)}个样本中学到:")
    for features, labels in feature_to_label.items():
        most_common = max(labels.items(), key=lambda x: x[1])
        print(f"    特征组合 → 最可能是: {most_common[0]} (出现{most_common[1]}次)")
    
    return training_data


# ==================== 2. 简单的学习算法 ====================

class SimpleLearner:
    """简单的学习器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.trained = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练模型"""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        raise NotImplementedError
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


class SimpleClassifier(SimpleLearner):
    """
    简单分类器示例
    使用最近邻思想：找到训练集中最相似的样本
    """
    
    def __init__(self):
        super().__init__("简单分类器")
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """记住训练数据"""
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.trained = True
        print(f"  训练完成：记住{len(X)}个样本")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测：找到最近的训练样本"""
        if not self.trained:
            raise ValueError("模型尚未训练！")
        
        predictions = []
        for x in X:
            # 计算与所有训练样本的距离
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # 找到最近的样本
            nearest_idx = np.argmin(distances)
            predictions.append(self.y_train[nearest_idx])
        
        return np.array(predictions)


class SimpleRegressor(SimpleLearner):
    """
    简单回归器示例
    使用线性模型：y = wx + b
    """
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100):
        super().__init__("简单回归器")
        self.w = 0.0  # 权重
        self.b = 0.0  # 偏置
        self.lr = learning_rate
        self.epochs = epochs
        self.loss_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """使用梯度下降训练"""
        n = len(X)
        
        print(f"  开始训练：{self.epochs}轮")
        for epoch in range(self.epochs):
            # 前向传播
            y_pred = self.w * X + self.b
            
            # 计算损失（均方误差）
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            
            # 反向传播（计算梯度）
            dw = (2/n) * np.sum((y_pred - y) * X)
            db = (2/n) * np.sum(y_pred - y)
            
            # 更新参数
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            if (epoch + 1) % 20 == 0:
                print(f"    第{epoch+1}轮：loss={loss:.4f}, w={self.w:.4f}, b={self.b:.4f}")
        
        self.trained = True
        print(f"  训练完成：w={self.w:.4f}, b={self.b:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.trained:
            raise ValueError("模型尚未训练！")
        return self.w * X + self.b


# ==================== 3. 学习的可视化 ====================

def visualize_learning_process():
    """可视化学习过程"""
    print("\n" + "=" * 60)
    print("可视化：机器学习如何学习？")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y_true = 2 * X + 1  # 真实关系
    y_noisy = y_true + np.random.randn(50) * 2  # 添加噪声
    
    # 划分训练集和测试集
    train_idx = random.sample(range(50), 30)
    test_idx = [i for i in range(50) if i not in train_idx]
    
    X_train, y_train = X[train_idx], y_noisy[train_idx]
    X_test, y_test = X[test_idx], y_noisy[test_idx]
    
    # 训练模型
    model = SimpleRegressor(learning_rate=0.01, epochs=200)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"\n测试集均方误差: {mse:.4f}")
    
    # 绘制学习曲线
    plt.figure(figsize=(15, 5))
    
    # 子图1：数据分布
    plt.subplot(1, 3, 1)
    plt.scatter(X_train, y_train, c='blue', label='训练数据', alpha=0.6)
    plt.scatter(X_test, y_test, c='red', label='测试数据', alpha=0.6)
    plt.plot(X, y_true, 'g--', label='真实关系', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('数据分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：损失下降
    plt.subplot(1, 3, 2)
    plt.plot(model.loss_history, linewidth=2)
    plt.xlabel('训练轮数')
    plt.ylabel('损失')
    plt.title('学习过程：损失下降')
    plt.grid(True, alpha=0.3)
    
    # 子图3：预测效果
    plt.subplot(1, 3, 3)
    plt.scatter(X_test, y_test, c='red', label='真实值', alpha=0.6)
    plt.scatter(X_test, y_pred, c='blue', label='预测值', alpha=0.6)
    X_line = np.linspace(0, 10, 100)
    y_line = model.w * X_line + model.b
    plt.plot(X_line, y_line, 'g-', label='学到的模型', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('预测效果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_visualization.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存到: learning_visualization.png")
    plt.show()
    
    return model


# ==================== 4. 学习 vs 记忆 ====================

def learning_vs_memorization():
    """
    演示学习与记忆的区别
    过拟合 = 死记硬背
    好的学习 = 理解规律
    """
    print("\n" + "=" * 60)
    print("学习与记忆的区别")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 生成数据
    X = np.linspace(0, 10, 20)
    y = np.sin(X) + np.random.randn(20) * 0.1
    
    print("\n【场景】学习正弦函数")
    print(f"训练数据：{len(X)}个点")
    
    # 方法1：死记硬背（过拟合）
    print("\n【死记硬背法】")
    print("  只记住每个训练点的值")
    print("  新数据来了？不知道！")
    
    # 方法2：理解规律（好的学习）
    print("\n【理解规律法】")
    print("  发现数据呈现周期性规律")
    print("  建立数学模型拟合")
    
    # 使用多项式拟合不同复杂度
    from numpy.polynomial import polynomial as P
    
    degrees = [1, 3, 19]  # 不同复杂度
    colors = ['blue', 'green', 'red']
    labels = ['简单模型(欠拟合)', '适中模型', '复杂模型(过拟合)']
    
    plt.figure(figsize=(15, 4))
    
    X_test = np.linspace(0, 10, 200)
    y_true = np.sin(X_test)
    
    for i, (deg, color, label) in enumerate(zip(degrees, colors, labels)):
        plt.subplot(1, 3, i+1)
        
        # 多项式拟合
        coeffs = P.polyfit(X, y, deg)
        y_pred = P.polyval(X_test, coeffs)
        
        plt.scatter(X, y, c='black', s=50, zorder=5, label='训练数据')
        plt.plot(X_test, y_true, 'k--', alpha=0.5, label='真实函数')
        plt.plot(X_test, y_pred, color=color, linewidth=2, label=label)
        
        # 计算训练误差和测试误差
        y_train_pred = P.polyval(X, coeffs)
        train_error = np.mean((y_train_pred - y) ** 2)
        
        y_test_pred = P.polyval(X_test, coeffs)
        test_error = np.mean((y_test_pred - y_true) ** 2)
        
        plt.title(f'{label}\n训练误差: {train_error:.4f}\n测试误差: {test_error:.4f}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('overfitting_demo.png', dpi=150, bbox_inches='tight')
    print("\n过拟合演示已保存到: overfitting_demo.png")
    plt.show()
    
    print("\n【结论】")
    print("✓ 好的学习：在训练集和测试集上都表现好")
    print("✗ 过拟合：训练集表现好，测试集表现差（死记硬背）")
    print("✗ 欠拟合：两者都表现差（理解不足）")


# ==================== 5. 不同类型的学习 ====================

def types_of_learning():
    """演示不同类型的机器学习"""
    print("\n" + "=" * 60)
    print("机器学习的三种主要类型")
    print("=" * 60)
    
    # 1. 监督学习
    print("\n【1. 监督学习】")
    print("  特点：有标签的数据 (X, y)")
    print("  例子：根据房屋特征预测房价")
    
    np.random.seed(42)
    house_size = np.random.randn(100) * 50 + 100  # 房屋面积
    house_price = house_size * 5 + np.random.randn(100) * 50 + 200  # 房价
    
    print(f"  数据示例：")
    for i in range(3):
        print(f"    面积: {house_size[i]:.1f}m² → 价格: ${house_price[i]:.1f}k")
    print(f"  任务：学习'面积→价格'的映射关系")
    
    # 2. 无监督学习
    print("\n【2. 无监督学习】")
    print("  特点：无标签的数据 (X)")
    print("  例子：将客户分成不同群体")
    
    # 生成两类客户数据
    cluster_a = np.random.randn(50, 2) + [2, 2]
    cluster_b = np.random.randn(50, 2) + [-2, -2]
    customers = np.vstack([cluster_a, cluster_b])
    
    print(f"  数据：{len(customers)}个客户，每个客户有2个特征")
    print(f"  任务：发现数据中隐藏的群体结构")
    
    # 3. 强化学习
    print("\n【3. 强化学习】")
    print("  特点：通过与环境交互学习")
    print("  例子：玩游戏、机器人控制")
    
    print(f"  概念：")
    print(f"    - 智能体(Agent)：做决策的实体")
    print(f"    - 环境(Environment)：智能体所处的世界")
    print(f"    - 动作(Action)：智能体可以执行的操作")
    print(f"    - 奖励(Reward)：环境对动作的反馈")
    print(f"  目标：学会最大化长期奖励的策略")
    
    # 可视化三种类型
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 监督学习
    axes[0].scatter(house_size, house_price, alpha=0.5)
    axes[0].set_xlabel('房屋面积 (m²)')
    axes[0].set_ylabel('房价 ($k)')
    axes[0].set_title('监督学习：预测房价')
    axes[0].grid(True, alpha=0.3)
    
    # 无监督学习
    axes[1].scatter(cluster_a[:, 0], cluster_a[:, 1], c='blue', label='群体A', alpha=0.6)
    axes[1].scatter(cluster_b[:, 0], cluster_b[:, 1], c='red', label='群体B', alpha=0.6)
    axes[1].set_xlabel('特征1')
    axes[1].set_ylabel('特征2')
    axes[1].set_title('无监督学习：客户分群')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 强化学习（示意图）
    axes[2].text(0.5, 0.8, '智能体', ha='center', fontsize=14, bbox=dict(boxstyle='round', facecolor='lightblue'))
    axes[2].arrow(0.5, 0.75, 0, -0.2, head_width=0.05, head_length=0.05, fc='blue', ec='blue')
    axes[2].text(0.5, 0.5, '动作', ha='center', fontsize=12)
    axes[2].arrow(0.5, 0.45, 0, -0.2, head_width=0.05, head_length=0.05, fc='green', ec='green')
    axes[2].text(0.5, 0.2, '环境', ha='center', fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgreen'))
    axes[2].arrow(0.6, 0.2, 0.2, 0, head_width=0.05, head_length=0.05, fc='red', ec='red')
    axes[2].text(0.85, 0.2, '奖励', ha='center', fontsize=12)
    axes[2].arrow(0.85, 0.25, -0.2, 0.45, head_width=0.05, head_length=0.05, fc='orange', ec='orange')
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_title('强化学习：智能体-环境交互')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('types_of_learning.png', dpi=150, bbox_inches='tight')
    print("\n三种学习类型已保存到: types_of_learning.png")
    plt.show()


# ==================== 6. 完整演示 ====================

def main():
    """主函数：运行所有演示"""
    print("=" * 70)
    print("第一章：什么是机器学习？ - 代码演示")
    print("=" * 70)
    
    # 1. 学习的本质
    learning_vs_programming_example()
    
    # 2. 简单学习算法演示
    print("\n" + "=" * 60)
    print("简单学习算法演示")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 分类演示
    print("\n【分类问题】")
    X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [5, 6]])
    y_train = np.array([0, 0, 0, 1, 1, 1])  # 0=类A, 1=类B
    
    clf = SimpleClassifier()
    clf.fit(X_train, y_train)
    
    X_test = np.array([[2, 2], [6, 6]])
    predictions = clf.predict(X_test)
    print(f"  测试样本预测: {predictions}")
    
    # 回归演示
    print("\n【回归问题】")
    X_train = np.array([1, 2, 3, 4, 5])
    y_train = np.array([2.1, 4.0, 6.2, 7.8, 10.5])
    
    reg = SimpleRegressor(learning_rate=0.05, epochs=100)
    reg.fit(X_train, y_train)
    
    X_test = np.array([6, 7])
    predictions = reg.predict(X_test)
    print(f"  测试样本预测: {predictions}")
    
    # 3. 可视化
    try:
        visualize_learning_process()
    except Exception as e:
        print(f"可视化跳过: {e}")
    
    # 4. 学习与记忆
    try:
        learning_vs_memorization()
    except Exception as e:
        print(f"过拟合演示跳过: {e}")
    
    # 5. 学习类型
    try:
        types_of_learning()
    except Exception as e:
        print(f"学习类型演示跳过: {e}")
    
    print("\n" + "=" * 70)
    print("第一章演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
