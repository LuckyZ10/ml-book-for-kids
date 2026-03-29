# 第四章：一步一步变得更好——梯度下降的直觉

> *"真正的智慧不在于从不犯错，而在于每次犯错后都能更接近真理。"*
> 
> —— 奥古斯丁-路易·柯西 (Augustin-Louis Cauchy, 1789-1857)

---

## 引子：迷路的小明

想象一下：小明和他的朋友们去登山。夕阳西下，浓雾突然降临，他们迷路了。手机没信号，GPS用不了，能见度只有脚下几米。

**怎么办？**

小明想到了一个好办法：
1. **感受脚下** —— 看看哪个方向是下坡
2. **小步走** —— 朝着下坡方向移动一点点
3. **重复** —— 每走几步，停下来再感受方向

这个方法听起来很笨，但它有一个神奇的名字：**梯度下降**（Gradient Descent）。

178年后的今天，这个方法训练了几乎所有你听说过的AI模型——从ChatGPT到自动驾驶汽车。

---

## 4.1 从山上下到谷底

### 4.1.1 直觉理解

想象一座山，山顶是损失最大（犯错最多）的地方，谷底是损失最小（犯错最少）的地方。我们的目标：**从山上走到谷底**。

**关键洞察**：
- 站在任何一个位置，**感受脚下的坡度**就能知道该往哪走
- **坡度最陡的反方向**就是下降最快的方向
- 每一步**不要迈太大**，否则可能越过谷底

```
                    🚶 小明在这里
                      \
                       \
         ⛰️             \
        /  \             \
       /    \             \
      /      \             \
     /        \             \
    /          \             \
   /            \             \
  /      🏔️      \             \
 /     山顶       \             \
/                   \             \
                     \             \
                      \             \
                       \             🏁 谷底
                        \           /(最优解)
                         \         /
                          \_______/
```

### 4.1.2 坡度 = 梯度

在数学上，**坡度**有个专业的名字——**梯度**（Gradient）。

- 一维：斜率（导数）
- 二维/多维：梯度（各方向偏导数组成的向量）

**直观理解**：
- 梯度指向**上升最快的方向**
- 负梯度指向**下降最快的方向**

```
📊 一维情况

损失
  │    ⛰️
  │   /  \
  │  /    \
  │ /      \
  │/        \
  ├───────────► 参数值
  0        最优值

  导数>0  →  往左走（减小参数）
  导数<0  →  往右走（增大参数）
```

---

## 4.2 数学之美：梯度下降的推导

### 4.2.1 从导数到梯度

假设我们有一个简单的损失函数（比如预测身高的误差）：

$$L(w) = (y - w \cdot x)^2$$

其中：
- $w$ 是我们要学习的参数（比如预测系数）
- $x$ 是输入（比如年龄）
- $y$ 是真实值（实际身高）

**问题**：如何找到让 $L(w)$ 最小的 $w$？

#### 步骤1：求导数

$$\frac{dL}{dw} = 2(y - w \cdot x) \cdot (-x) = -2x(y - w \cdot x)$$

#### 步骤2：梯度下降更新

$$w_{\text{新}} = w_{\text{旧}} - \eta \cdot \frac{dL}{dw}$$

其中 $\eta$（eta）是**学习率**（Learning Rate），控制每一步迈多大。

### 4.2.2 多维情况

当参数不止一个时（比如同时学习年龄系数和性别系数）：

$$\mathbf{w} = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_d \end{bmatrix}$$

**梯度**是所有偏导数组成的向量：

$$\nabla L(\mathbf{w}) = \begin{bmatrix} \frac{\partial L}{\partial w_1} \\ \frac{\partial L}{\partial w_2} \\ \vdots \\ \frac{\partial L}{\partial w_d} \end{bmatrix}$$

**更新规则**：

$$\mathbf{w}_{\text{新}} = \mathbf{w}_{\text{旧}} - \eta \cdot \nabla L(\mathbf{w}_{\text{旧}})$$

### 4.2.3 几何解释

```
📐 二维参数空间

      w₂
      │
      │    ╱──────╲
      │   ╱        ╲
      │  ╱          ╲
      │ ╱            ╲
      │╱              ╲
      ├───────○────────► w₁
      │      最优值
      │
      ↓ 梯度方向 = 上升最快

当前位置 → 沿负梯度方向移动 → 更接近最优值
```

---

## 4.3 学习率：步长的艺术

### 4.3.1 为什么学习率很重要？

学习率 $\eta$ 控制每次更新的步长：

| 学习率 | 效果 | 结果 |
|--------|------|------|
| 太小 | 步子迈太小 | 收敛慢，需要很多步 |
| 合适 | 步子适中 | 快速收敛到最优 |
| 太大 | 步子迈太大 | 震荡，甚至发散 |

```
📊 学习率的影响

学习率太小:                    学习率合适:
损失│                          损失│
  │    ╲                        │    ╲
  │     ╲                       │     ╲
  │      ╲                      │      ╲
  │       ╲                     │       ╲
  │        ╲                    │        ╲
  │         ╲_                  │         ╲_
  └──────────►                  └──────────►
    慢速收敛                      快速收敛

学习率太大:
损失│
  │╲   ╱╲   ╱╲   ╱
  │ ╲ ╱  ╲ ╱  ╲ ╱
  │  ╲    ╲    ╱
  │        ╲  ╱
  │
  └──────────►
    震荡，可能发散
```

### 4.3.2 学习率调度

聪明的做法是：**开始时大步走，接近时小步走**。

**常用策略**：
1. **固定学习率**：简单但不一定最优
2. **学习率衰减**：每过几轮，学习率乘以一个系数（如0.9）
3. **自适应学习率**：根据梯度大小自动调整（后面会讲Adam）

```
📉 学习率衰减

学习率
  │╲
  │ ╲
  │  ╲
  │   ╲
  │    ╲
  │     ╲______
  └───────────► 迭代次数

  开始大步探索，后期精细调整
```

---

## 4.4 鞍点与局部最优：陷阱与迷思

### 4.4.1 局部最优陷阱

想象一座山有好几个谷底：

```
📊 多个局部最优

损失│    ⛰️        ⛰️
  │   /  \      /  \
  │  /    \    /    \
  │ /      \__/      \
  │/    🕳️            \
  ├───────────────────►
       局部最优    全局最优
```

**梯度下降的问题**：可能卡在**局部最优**，而不是**全局最优**。

### 4.4.2 鞍点：更狡猾的陷阱

鞍点（Saddle Point）在某些方向是谷底，在另一些方向是山顶：

```
📐 鞍点示意

       w₂
        │
        │      ╱╲
        │     ╱  ╲
        │    ╱    ╲
        │   ╱  ●   ╲   ← 鞍点：w₁方向是山顶，w₂方向是谷底
        │  ╱        ╲
        │ ╱          ╲
        └─────────────► w₁

像马鞍一样：前后是上坡，左右是下坡
```

在高维空间中，鞍点比局部最优更常见！幸运的是，随机梯度下降（SGD）能帮助我们逃离鞍点。

---

## 4.5 随机梯度下降（SGD）：大数据时代的救星

### 4.5.1 从全量到随机

**问题**：如果数据有100万个样本，每次计算梯度都要遍历全部数据，太慢了！

**解决方案**：随机梯度下降（Stochastic Gradient Descent, SGD）

**核心思想**：
- 不计算全部数据的梯度
- 随机选一个（或一小批）样本
- 用它的梯度近似整体梯度

### 4.5.2 SGD的优势

| 方面 | 批梯度下降 | 随机梯度下降 |
|------|------------|--------------|
| 速度 | 慢（每步算全部） | 快（每步算一个） |
| 内存 | 大 | 小 |
| 收敛 | 稳定 | 有噪声，但能逃离鞍点 |
| 在线学习 | 不支持 | 支持 |

### 4.5.3 小批量（Mini-batch）：折中之道

实践中通常使用**小批量梯度下降**：

```
批量大小: 32, 64, 128, 256...

📦 小批量示意

全部数据: [█][█][█][█][█][█][█][█][█][█][█][█]
          └───┘ 第1批
                └───┘ 第2批
                      └───┘ 第3批

每步只用一批数据计算梯度
```

**小批量的好处**：
- 比单样本稳定（噪声小）
- 比全量快（计算量少）
- 可以利用矩阵运算加速（GPU友好）

---

## 4.6 动量法：借惯性之力

### 4.6.1 直觉：滚下山的球

想象一个球滚下山：
- 它不会每一步都停下来重新判断方向
- 它会**保持惯性**，沿之前的方向继续前进
- 遇到小坑，惯性会带着它越过去

这就是**动量法**（Momentum）的直觉。

### 4.6.2 动量法的数学

引入**速度**变量 $v$：

$$v_t = \beta \cdot v_{t-1} + \nabla L(w_t)$$

$$w_{t+1} = w_t - \eta \cdot v_t$$

其中：
- $v_t$：第 $t$ 步的速度（累积的历史梯度）
- $\beta$：动量系数（通常0.9），控制"惯性"大小

### 4.6.3 动量的效果

```
📊 动量法的优势

无动量：                      有动量：
    │                          │
    │  ↓↓↓ 震荡                │  ↓ 平滑下降
    │ ↓↓↓                      │
    │↓↓↓                       │
    └────►                     └────►

动量帮助：
✓ 加速收敛（沿一致方向累积）
✓ 减少震荡（抵消垂直方向的抖动）
✓ 逃离局部最优（惯性冲过去）
```

---

## 4.7 现代优化器简介

### 4.7.1 AdaGrad：自适应学习率

**问题**：不同参数可能需要不同的学习率。

**AdaGrad的解决方案**：
- 记录每个参数的历史梯度平方和
- 梯度大的参数，学习率自动减小
- 梯度小的参数，学习率保持较大

$$w_{t+1} = w_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot g_t$$

其中 $G_t$ 是历史梯度平方的累积。

### 4.7.2 RMSProp：改进的自适应

AdaGrad的问题是：学习率单调递减，最后可能太小。

RMSProp使用**指数移动平均**替代累积：

$$E[g^2]_t = 0.9 \cdot E[g^2]_{t-1} + 0.1 \cdot g_t^2$$

这样学习率不会无限减小。

### 4.7.3 Adam：集大成者

**Adam**（Adaptive Moment Estimation）结合了动量和自适应学习率：

```
🤖 Adam = 动量 + RMSProp

   动量项: m_t = β₁·m_{t-1} + (1-β₁)·g_t
   二阶项: v_t = β₂·v_{t-1} + (1-β₂)·g_t²
   
   更新: w_{t+1} = w_t - η·m̂_t/(√v̂_t + ε)
```

**Adam的优势**：
- 默认参数通常工作良好（β₁=0.9, β₂=0.999, η=0.001）
- 对稀疏梯度效果好
- 是目前最常用的优化器之一

---

## 4.8 完整代码实现

```python
# chapter-04-gradient-descent.py
# 第四章：梯度下降 - 从零实现

import random
import math


class GradientDescent:
    """纯Python实现的各种梯度下降算法"""
    
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.history = []  # 记录优化轨迹
    
    # ============ 基础梯度下降 ============
    
    def gradient_descent_1d(self, gradient_func, initial_x, iterations=100):
        """
        一维梯度下降
        
        参数:
            gradient_func: 梯度函数，输入x返回导数
            initial_x: 初始值
            iterations: 迭代次数
        """
        x = initial_x
        self.history = [(i, x, None) for i in range(iterations + 1)]
        
        print("=" * 50)
        print("📉 一维梯度下降演示")
        print("=" * 50)
        print(f"初始值: x = {x:.4f}")
        print(f"学习率: η = {self.lr}")
        print("-" * 50)
        
        for i in range(iterations):
            grad = gradient_func(x)
            x = x - self.lr * grad
            
            if i < 5 or i % 20 == 0 or i == iterations - 1:
                print(f"迭代 {i+1:3d}: x = {x:8.4f}, grad = {grad:8.4f}")
        
        print("-" * 50)
        print(f"✅ 最终结果: x = {x:.6f}")
        return x
    
    def gradient_descent_2d(self, gradient_func, initial_point, iterations=100):
        """
        二维梯度下降
        
        参数:
            gradient_func: 梯度函数，输入[x, y]返回[∂L/∂x, ∂L/∂y]
            initial_point: 初始点 [x, y]
        """
        point = list(initial_point)
        self.history = [tuple(point)]
        
        print("\n" + "=" * 50)
        print("📉 二维梯度下降演示")
        print("=" * 50)
        print(f"初始点: ({point[0]:.2f}, {point[1]:.2f})")
        
        for i in range(iterations):
            grad = gradient_func(point)
            point[0] -= self.lr * grad[0]
            point[1] -= self.lr * grad[1]
            self.history.append(tuple(point))
            
            if i < 3 or i == iterations - 1:
                print(f"迭代 {i+1:3d}: ({point[0]:8.4f}, {point[1]:8.4f})")
        
        print(f"✅ 最终点: ({point[0]:.6f}, {point[1]:.6f})")
        return point
    
    # ============ 动量法 ============
    
    def momentum_gd(self, gradient_func, initial_x, beta=0.9, iterations=100):
        """
        带动量的梯度下降
        
        参数:
            beta: 动量系数 (0-1之间，通常0.9)
        """
        x = initial_x
        v = 0  # 初始速度为0
        
        print("\n" + "=" * 50)
        print("🏃 动量梯度下降演示")
        print("=" * 50)
        print(f"动量系数: β = {beta}")
        print(f"学习率: η = {self.lr}")
        
        for i in range(iterations):
            grad = gradient_func(x)
            v = beta * v + grad  # 速度更新
            x = x - self.lr * v  # 位置更新
            
            if i < 5 or i % 25 == 0 or i == iterations - 1:
                print(f"迭代 {i+1:3d}: x = {x:8.4f}, v = {v:8.4f}")
        
        print(f"✅ 最终结果: x = {x:.6f}")
        return x
    
    # ============ 线性回归的梯度下降 ============
    
    def linear_regression_sgd(self, X, y, iterations=1000):
        """
        使用SGD训练线性回归
        
        参数:
            X: 输入数据列表 [[x1], [x2], ...]
            y: 目标值列表 [y1, y2, ...]
        """
        n = len(X)
        # 初始化参数 [权重w, 偏置b]
        w, b = random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)
        
        print("\n" + "=" * 50)
        print("📈 线性回归 SGD 训练")
        print("=" * 50)
        print(f"样本数: {n}")
        print(f"初始参数: w={w:.4f}, b={b:.4f}")
        
        for epoch in range(iterations):
            total_loss = 0
            
            # 对每个样本进行SGD
            for i in range(n):
                xi, yi = X[i][0], y[i]
                
                # 前向传播
                pred = w * xi + b
                error = pred - yi
                loss = 0.5 * error ** 2
                total_loss += loss
                
                # 反向传播（计算梯度）
                dw = error * xi  # ∂L/∂w
                db = error        # ∂L/∂b
                
                # 参数更新
                w -= self.lr * dw
                b -= self.lr * db
            
            avg_loss = total_loss / n
            
            if epoch < 5 or epoch % 200 == 0 or epoch == iterations - 1:
                print(f"Epoch {epoch:4d}: loss={avg_loss:.6f}, w={w:.4f}, b={b:.4f}")
        
        print("-" * 50)
        print(f"✅ 训练完成: w={w:.6f}, b={b:.6f}")
        return w, b


# ============ 测试函数 ============

def test_quadratic():
    """测试：最小化 f(x) = x²"""
    print("\n" + "🧪" * 25)
    print("测试1: 二次函数 f(x) = x²")
    print("最优解: x = 0")
    
    def grad(x):
        return 2 * x  # f'(x) = 2x
    
    gd = GradientDescent(learning_rate=0.1)
    result = gd.gradient_descent_1d(grad, initial_x=5.0, iterations=50)
    
    assert abs(result) < 0.01, "应该收敛到接近0"
    print("✅ 测试通过！")


def test_rosenbrock():
    """测试：Rosenbrock函数（优化领域的经典测试函数）"""
    print("\n" + "🧪" * 25)
    print("测试2: Rosenbrock函数")
    print("f(x,y) = (1-x)² + 100(y-x²)²")
    print("最优解: (1, 1)")
    
    def grad(point):
        x, y = point
        # ∂f/∂x = -2(1-x) - 400x(y-x²)
        # ∂f/∂y = 200(y-x²)
        dx = -2 * (1 - x) - 400 * x * (y - x**2)
        dy = 200 * (y - x**2)
        return [dx, dy]
    
    gd = GradientDescent(learning_rate=0.001)
    result = gd.gradient_descent_2d(grad, initial_point=[-1.0, 1.0], iterations=5000)
    
    print(f"\n距离最优解: {math.sqrt((result[0]-1)**2 + (result[1]-1)**2):.6f}")


def test_linear_fit():
    """测试：用SGD拟合直线 y = 2x + 1"""
    print("\n" + "🧪" * 25)
    print("测试3: 线性回归拟合")
    print("真实函数: y = 2x + 1")
    
    # 生成数据
    X = [[i] for i in range(-10, 11)]
    y = [2 * xi[0] + 1 + random.uniform(-0.5, 0.5) for xi in X]
    
    gd = GradientDescent(learning_rate=0.01)
    w, b = gd.linear_regression_sgd(X, y, iterations=1000)
    
    print("\n预测 vs 真实:")
    for xi in [[0], [5], [10]]:
        pred = w * xi[0] + b
        true = 2 * xi[0] + 1
        print(f"  x={xi[0]:2d}: 预测={pred:6.2f}, 真实={true:6.2f}")


def compare_momentum():
    """对比：普通GD vs 动量GD"""
    print("\n" + "🧪" * 25)
    print("测试4: 对比普通GD与动量GD")
    print("函数: f(x) = 0.5 * x²")
    
    def grad(x):
        return x  # f'(x) = x
    
    print("\n--- 普通梯度下降 ---")
    gd1 = GradientDescent(learning_rate=0.1)
    gd1.gradient_descent_1d(grad, initial_x=10.0, iterations=50)
    
    print("\n--- 动量梯度下降 ---")
    gd2 = GradientDescent(learning_rate=0.1)
    gd2.momentum_gd(grad, initial_x=10.0, beta=0.9, iterations=50)


# ============ 可视化辅助 ============

def visualize_ascii_contour():
    """用ASCII可视化损失函数等高线"""
    print("\n" + "=" * 60)
    print("🗺️ 损失函数地形图（ASCII可视化）")
    print("=" * 60)
    
    # 模拟一个简单的二次函数地形
    size = 15
    center = size // 2
    
    print("\n   " + "".join(f"{i:2d}" for i in range(-center, center+1)))
    
    for y in range(-center, center+1):
        row = f"{y:2d} "
        for x in range(-center, center+1):
            # 损失值（简单的碗状函数）
            loss = (x**2 + y**2) / 50
            
            # 根据损失值选择字符
            if x == 0 and y == 0:
                row += "🏁 "  # 最优解
            elif loss < 0.5:
                row += "∙  "
            elif loss < 2:
                row += "○  "
            elif loss < 5:
                row += "◯  "
            else:
                row += "·  "
        print(row)
    
    print("\n图例: 🏁=最优解, ∙=低损失区, ○=中损失区, ◯=高损失区")


# ============ 主程序 ============

if __name__ == "__main__":
    print("🚀 第四章：梯度下降完整演示")
    print("=" * 50)
    
    # 运行所有测试
    test_quadratic()
    test_rosenbrock()
    test_linear_fit()
    compare_momentum()
    visualize_ascii_contour()
    
    print("\n" + "=" * 50)
    print("✅ 所有测试完成！")
    print("=" * 50)
    print("\n📚 本章要点:")
    print("  1. 梯度下降：沿着负梯度方向更新参数")
    print("  2. 学习率：控制步长，需要仔细选择")
    print("  3. SGD：随机梯度下降，适合大数据")
    print("  4. 动量法：利用惯性加速收敛")
    print("  5. Adam：现代优化器，自适应+动量")
