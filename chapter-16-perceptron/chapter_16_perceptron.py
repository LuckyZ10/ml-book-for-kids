# 第十六章：感知机——神经网络的起点

> *"感知机代表了计算本身的全新思考方式，机器可以从经验中学习，而非仅仅执行预定指令。"*
> —— Frank Rosenblatt, 1958

---

## 16.1 一个改变世界的机器

1958年的春天，在纽约州布法罗市康奈尔航空实验室的一间实验室里，一位名叫**弗兰克·罗森布拉特（Frank Rosenblatt）**的年轻心理学家正盯着一台奇怪的机器。这台机器连接着400个光电传感器，通过密密麻麻的电线连接到一个"决策层"，最终输出一个判断结果。

这台机器被命名为**Mark I 感知机（Mark I Perceptron）**，它是人类历史上第一台能够从经验中学习的机器。

### 🧠 一个大胆的假设

Rosenblatt 的思考始于一个关于大脑的问题：**大脑是如何存储信息的？**

在1950年代，主流观点认为大脑像计算机一样，将信息存储在特定的"存储单元"中。但Rosenblatt提出了一个革命性的观点：**信息存储在神经元之间的连接强度中**。

这个观点并非凭空想象。它基于三位科学家的前期工作：

**1. McCulloch & Pitts (1943) —— 神经元的数学模型**

Warren McCulloch 和 Walter Pitts 发表了论文《A Logical Calculus of the Ideas Immanent in Nervous Activity》，首次用数学模型描述了神经元的工作原理：

```
神经元输出 = 阈值函数(加权输入之和)
```

他们证明了，只要连接方式正确，人工神经元网络可以计算**任何逻辑函数**。

**2. Hebb (1949) —— 学习的神经机制**

Donald Hebb 在《The Organization of Behavior》中提出了著名的**Hebb学习规则**：

> *"一起激活的神经元，连接在一起。"*
> （"Neurons that fire together, wire together.")

这个规则解释了大脑如何通过调整突触连接强度来学习。

**3. Rosenblatt (1958) —— 感知机**

Rosenblatt 将上述思想整合成了一个完整的系统——**感知机（Perceptron）**，一个能够**自动学习**的人工神经网络。

---

## 16.2 感知机是什么？

想象你正在教一个小朋友识别苹果。你会怎么做？

1. **展示**：给小朋友看一些苹果，告诉他"这是苹果"
2. **对比**：再给他看一些橙子，告诉他"这不是苹果"
3. **观察**：小朋友会注意到苹果是"圆的"、"红色的"、"有苹果香味"
4. **纠正**：如果他把番茄错认成苹果，你会纠正他
5. **学习**：小朋友调整他脑中的"苹果标准"，直到能正确区分

**感知机做的就是同样的事情！**

### 16.2.1 感知机的结构

一个感知机由三部分组成：

```
        输入层 (感觉单元 S-Points)          输出层 (响应单元 R-Unit)
        ┌─────────┐
   x₁ ──┤         │      ┌──────────┐
        │  加权   ├──────┤  激活    ├───→ 输出 (0 或 1)
   x₂ ──┤  求和   │      │  函数    │
        │         │      └──────────┘
   x₃ ──┤         │           ↑
        └─────────┘        阈值 θ
        
        权重: w₁, w₂, w₃
```

**数学表达：**

$$output = \begin{cases} 1 & \text{if } \sum_{i=1}^{n} w_i x_i + b > 0 \\ 0 & \text{otherwise} \end{cases}$$

其中：
- $x_i$：输入信号（0或1，表示某个特征是否存在）
- $w_i$：权重（连接强度，可正可负）
- $b$：偏置（阈值，决定激活的难易程度）

### 16.2.2 感知机学习什么？

感知机学习的是**决策边界（Decision Boundary）**。

想象一个二维平面，上面有一些红色的点和蓝色的点。感知机的目标是找到一条直线，把红点和蓝点分开。

```
    红点 ●                    决策边界
         ●         ───────────────────
            ●                  ↑
              ●       ● 蓝点   │ w·x + b = 0
                  ●            │
                      ●        │
```

这条直线就是**线性分类器**，它的方程是：

$$w_1 x_1 + w_2 x_2 + b = 0$$

感知机的任务就是通过学习，找到最佳的 $w_1$、$w_2$ 和 $b$。

---

## 16.3 感知机学习规则

### 16.3.1 核心思想：犯错就改

感知机的学习规则非常简单：

> **如果预测错误，就调整权重，让下次更可能正确。**

具体来说：

```python
if 预测 != 真实标签:
    # 调整权重
    w_new = w_old + learning_rate × (真实 - 预测) × x
    b_new = b_old + learning_rate × (真实 - 预测)
```

这就是著名的**感知机学习规则（Perceptron Learning Rule）**：

$$w(t+1) = w(t) + \eta \cdot (d - y) \cdot x$$

其中：
- $\eta$：学习率（learning rate），控制调整幅度
- $d$：期望输出（真实标签）
- $y$：实际输出（预测结果）
- $(d - y)$：误差信号

### 16.3.2 为什么这样调整？

让我们分析三种情况：

**情况1：预测正确（y = d）**

$$w(t+1) = w(t) + \eta \cdot 0 \cdot x = w(t)$$

权重不变，因为不需要调整。

**情况2：假阴性（y = 0, d = 1）**

真实是正类，但预测为负类。说明 $w \cdot x$ 太小了。

$$w(t+1) = w(t) + \eta \cdot 1 \cdot x = w(t) + \eta x$$

权重向 $x$ 的方向增加，下次遇到类似输入时，输出会更大。

**情况3：假阳性（y = 1, d = 0）**

真实是负类，但预测为正类。说明 $w \cdot x$ 太大了。

$$w(t+1) = w(t) + \eta \cdot (-1) \cdot x = w(t) - \eta x$$

权重向 $x$ 的反方向减少，下次遇到类似输入时，输出会更小。

### 16.3.3 感知机收敛定理

1962年，Rosenblatt 证明了著名的**感知机收敛定理（Perceptron Convergence Theorem）**：

> **如果数据是线性可分的，感知机学习算法保证在有限步内收敛到一个能正确分类所有数据的解。**

这个定理非常重要，它证明了感知机学习的**数学可靠性**。

---

## 16.4 从零实现感知机

好了，理论讲完了，让我们动手写一个感知机！

### 16.4.1 基础感知机实现

```python
"""
第十六章：感知机实现
从零开始构建第一个人工神经网络！
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class Perceptron:
    """
    感知机 —— 神经网络的起源
    
    基于 Frank Rosenblatt (1958) 的原始论文实现
    "The Perceptron: A Probabilistic Model for Information Storage 
     and Organization in the Brain"
    """
    
    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 100):
        """
        初始化感知机
        
        Parameters:
        -----------
        learning_rate : float
            学习率 η，控制权重调整的幅度
            Rosenblatt 原始论文建议使用 0.1 左右
        n_iterations : int
            最大训练轮数
        """
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
        self.errors_ = []  # 记录每轮的错误数
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Perceptron':
        """
        训练感知机
        
        Parameters:
        -----------
        X : np.ndarray, shape = [n_samples, n_features]
            训练数据
        y : np.ndarray, shape = [n_samples]
            目标标签（0或1）
            
        Returns:
        --------
        self : Perceptron
        """
        n_samples, n_features = X.shape
        
        # 初始化权重和偏置
        # Rosenblatt 建议小随机值初始化
        np.random.seed(42)
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        
        print("🧠 感知机训练开始...")
        print(f"   样本数: {n_samples}, 特征数: {n_features}")
        print(f"   学习率: {self.lr}")
        print(f"   初始权重: {self.weights}")
        print(f"   初始偏置: {self.bias}")
        print("-" * 50)
        
        # 训练循环
        for epoch in range(self.n_iterations):
            errors = 0
            
            for xi, target in zip(X, y):
                # 计算预测值
                output = self._predict_one(xi)
                
                # 计算误差
                error = target - output
                
                # 感知机学习规则: Δw = η * (d - y) * x
                if error != 0:
                    update = self.lr * error
                    self.weights += update * xi
                    self.bias += update
                    errors += 1
            
            self.errors_.append(errors)
            
            if epoch % 10 == 0 or errors == 0:
                print(f"第 {epoch:3d} 轮: 错误数 = {errors}, "
                      f"权重 = [{self.weights[0]:.3f}, {self.weights[1]:.3f}], "
                      f"偏置 = {self.bias:.3f}")
            
            # 如果完全分类正确，提前停止
            if errors == 0:
                print(f"\n✅ 收敛！在第 {epoch} 轮完成训练")
                break
        
        print("-" * 50)
        print(f"最终权重: {self.weights}")
        print(f"最终偏置: {self.bias}")
        print(f"决策边界: {self.weights[0]:.3f}*x + {self.weights[1]:.3f}*y + "
              f"({self.bias:.3f}) = 0")
        
        return self
    
    def _predict_one(self, x: np.ndarray) -> int:
        """
        预测单个样本
        
        使用 Heaviside 阶跃函数作为激活函数
        这也是 Rosenblatt 原始论文使用的函数
        """
        activation = np.dot(x, self.weights) + self.bias
        return 1 if activation > 0 else 0
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测多个样本
        """
        return np.array([self._predict_one(xi) for xi in X])
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        计算决策函数值（激活前的值）
        """
        return np.dot(X, self.weights) + self.bias


# ============================================
# 演示1: 用感知机学习 AND 逻辑门
# ============================================

def demo_and_gate():
    """
    演示1: 感知机学习 AND 逻辑门
    
    AND 逻辑表:
    0 AND 0 = 0
    0 AND 1 = 0
    1 AND 0 = 0
    1 AND 1 = 1
    
    这是一个线性可分问题！
    """
    print("=" * 60)
    print("演示1: 感知机学习 AND 逻辑门")
    print("=" * 60)
    
    # AND 逻辑门数据
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    print("\n训练数据 (AND):")
    print("输入    输出")
    for xi, yi in zip(X, y):
        print(f"{xi}  ->  {yi}")
    
    # 创建并训练感知机
    p = Perceptron(learning_rate=0.1, n_iterations=100)
    p.fit(X, y)
    
    # 测试
    print("\n测试结果:")
    predictions = p.predict(X)
    for xi, pred, true in zip(X, predictions, y):
        status = "✓" if pred == true else "✗"
        print(f"{xi} 预测={pred}, 真实={true} {status}")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    
    # 子图1: 数据和决策边界
    plt.subplot(1, 2, 1)
    for i, (xi, yi) in enumerate(zip(X, y)):
        color = 'red' if yi == 0 else 'blue'
        marker = 'o' if yi == 0 else 's'
        plt.scatter(xi[0], xi[1], c=color, marker=marker, s=200, 
                   edgecolors='black', linewidths=2, 
                   label=f'{xi} -> {yi}' if i < 2 else "")
    
    # 绘制决策边界
    x_min, x_max = -0.5, 1.5
    if p.weights[1] != 0:
        x_boundary = np.linspace(x_min, x_max, 100)
        y_boundary = -(p.weights[0] * x_boundary + p.bias) / p.weights[1]
        plt.plot(x_boundary, y_boundary, 'g--', linewidth=2, 
                label='决策边界')
    
    plt.xlim(x_min, x_max)
    plt.ylim(x_min, x_max)
    plt.xlabel('输入 x₁', fontsize=12)
    plt.ylabel('输入 x₂', fontsize=12)
    plt.title('AND 逻辑门 - 线性可分', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 错误数变化
    plt.subplot(1, 2, 2)
    plt.plot(range(len(p.errors_)), p.errors_, 'b-o', linewidth=2)
    plt.xlabel('训练轮数', fontsize=12)
    plt.ylabel('错误数', fontsize=12)
    plt.title('学习过程', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('perceptron_and.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存: perceptron_and.png")
    plt.show()


# ============================================
# 演示2: 用感知机学习 OR 逻辑门
# ============================================

def demo_or_gate():
    """
    演示2: 感知机学习 OR 逻辑门
    
    OR 逻辑表:
    0 OR 0 = 0
    0 OR 1 = 1
    1 OR 0 = 1
    1 OR 1 = 1
    
    这也是线性可分的！
    """
    print("\n" + "=" * 60)
    print("演示2: 感知机学习 OR 逻辑门")
    print("=" * 60)
    
    # OR 逻辑门数据
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 1, 1, 1])
    
    print("\n训练数据 (OR):")
    print("输入    输出")
    for xi, yi in zip(X, y):
        print(f"{xi}  ->  {yi}")
    
    p = Perceptron(learning_rate=0.1, n_iterations=100)
    p.fit(X, y)
    
    print("\n测试结果:")
    predictions = p.predict(X)
    for xi, pred, true in zip(X, predictions, y):
        status = "✓" if pred == true else "✗"
        print(f"{xi} 预测={pred}, 真实={true} {status}")


# ============================================
# 演示3: XOR 问题 —— 感知机的局限性
# ============================================

def demo_xor_problem():
    """
    演示3: XOR 问题 —— 感知机的致命弱点
    
    XOR 逻辑表:
    0 XOR 0 = 0
    0 XOR 1 = 1
    1 XOR 0 = 1
    1 XOR 1 = 0
    
    这不是线性可分问题！
    这是 Minsky & Papert (1969) 在《Perceptrons》中指出的关键局限。
    """
    print("\n" + "=" * 60)
    print("演示3: XOR 问题 —— 感知机的局限性")
    print("=" * 60)
    print("""
    这是一个历史性的时刻！
    
    1969年，Marvin Minsky 和 Seymour Papert 出版了《Perceptrons》一书，
    证明了单层感知机无法解决 XOR 问题。
    
    XOR 问题:
    ┌─────────┬─────────┬────────┐
    │   x₁    │   x₂    │  XOR   │
    ├─────────┼─────────┼────────┤
    │    0    │    0    │   0    │
    │    0    │    1    │   1    │  ← 这一类
    │    1    │    0    │   1    │  ← 这一类
    │    1    │    1    │   0    │
    └─────────┴─────────┴────────┘
    
    观察输出为1的两个点 (0,1) 和 (1,0)，
    你会发现它们无法被一条直线从输出为0的点中分离出来！
    """)
    
    # XOR 数据
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 1, 1, 0])
    
    print("训练数据 (XOR):")
    print("输入    输出")
    for xi, yi in zip(X, y):
        print(f"{xi}  ->  {yi}")
    
    # 尝试用感知机学习 XOR
    p = Perceptron(learning_rate=0.1, n_iterations=100)
    p.fit(X, y)
    
    print("\n测试结果:")
    predictions = p.predict(X)
    accuracy = np.mean(predictions == y)
    for xi, pred, true in zip(X, predictions, y):
        status = "✓" if pred == true else "✗"
        print(f"{xi} 预测={pred}, 真实={true} {status}")
    
    print(f"\n准确率: {accuracy * 100:.1f}%")
    print("❌ 感知机无法学习 XOR 问题！")
    
    # 可视化 XOR 问题
    plt.figure(figsize=(8, 6))
    
    # 绘制四个点
    for xi, yi in zip(X, y):
        color = 'red' if yi == 0 else 'blue'
        marker = 'o' if yi == 0 else 's'
        plt.scatter(xi[0], xi[1], c=color, marker=marker, s=400,
                   edgecolors='black', linewidths=2)
        plt.annotate(f'({xi[0]},{xi[1]})→{yi}', 
                    xy=(xi[0], xi[1]), 
                    xytext=(xi[0]+0.1, xi[1]+0.1),
                    fontsize=12)
    
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('输入 x₁', fontsize=12)
    plt.ylabel('输入 x₂', fontsize=12)
    plt.title('XOR 问题 —— 非线性可分！\n无法找到一条直线分离两类', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('perceptron_xor.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存: perceptron_xor.png")
    plt.show()
    
    print("""
    💡 关键启示:
    
    Minsky & Papert 的证明震惊了AI界。
    这导致了对神经网络研究的资助大幅减少，
    进入了所谓的"第一次AI寒冬" (AI Winter)。
    
    但这个"局限"也推动了研究：
    - 多层感知机（MLP）的出现
    - 反向传播算法（Backpropagation, 1986）
    - 深度学习革命
    
    XOR 问题的解决方案需要**非线性决策边界**，
    这正是多层神经网络的优势！
    """)


# ============================================
# 演示4: 二分类问题
# ============================================

def demo_classification():
    """
    演示4: 用感知机解决实际分类问题
    """
    print("\n" + "=" * 60)
    print("演示4: 感知机分类实际数据")
    print("=" * 60)
    
    # 生成线性可分的数据
    np.random.seed(42)
    
    # 类别0: 左下角
    X_0 = np.random.randn(50, 2) + np.array([-2, -2])
    y_0 = np.zeros(50)
    
    # 类别1: 右上角
    X_1 = np.random.randn(50, 2) + np.array([2, 2])
    y_1 = np.ones(50)
    
    # 合并数据
    X = np.vstack([X_0, X_1])
    y = np.hstack([y_0, y_1])
    
    print(f"\n生成数据: {len(X)} 个样本")
    print(f"类别0: {len(X_0)} 个")
    print(f"类别1: {len(X_1)} 个")
    
    # 训练感知机
    p = Perceptron(learning_rate=0.01, n_iterations=100)
    p.fit(X, y)
    
    # 评估
    predictions = p.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\n训练准确率: {accuracy * 100:.1f}%")
    
    # 可视化
    plt.figure(figsize=(10, 8))
    
    # 绘制数据点
    plt.scatter(X_0[:, 0], X_0[:, 1], c='red', marker='o', 
               s=100, label='类别 0', alpha=0.6, edgecolors='black')
    plt.scatter(X_1[:, 0], X_1[:, 1], c='blue', marker='s', 
               s=100, label='类别 1', alpha=0.6, edgecolors='black')
    
    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    if p.weights[1] != 0:
        x_boundary = np.linspace(x_min, x_max, 100)
        y_boundary = -(p.weights[0] * x_boundary + p.bias) / p.weights[1]
        plt.plot(x_boundary, y_boundary, 'g-', linewidth=3, 
                label='决策边界')
    
    plt.xlabel('特征 1', fontsize=12)
    plt.ylabel('特征 2', fontsize=12)
    plt.title('感知机分类结果', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('perceptron_classification.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存: perceptron_classification.png")
    plt.show()


# ============================================
# 主程序
# ============================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║           感知机 —— 神经网络的起点                        ║
    ║           Perceptron: The Beginning of Neural Networks   ║
    ║                                                           ║
    ║   "The Perceptron: A Probabilistic Model for..."         ║
    ║    — Frank Rosenblatt, 1958                               ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # 运行所有演示
    demo_and_gate()
    demo_or_gate()
    demo_xor_problem()
    demo_classification()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
    print("""
    📚 本章要点:
    
    1. 感知机是第一个人工神经网络，由 Rosenblatt 于1958年提出
    
    2. 感知机学习规则:
       w_new = w_old + η * (d - y) * x
       
    3. 感知机收敛定理:
       如果数据线性可分，算法保证收敛
       
    4. 感知机只能解决线性可分问题（AND、OR）
    
    5. 感知机无法解决 XOR 问题 —— Minsky & Papert (1969)
       这导致了第一次AI寒冬
       
    6. 解决方案: 多层感知机 + 反向传播（下一章！）
    """)
