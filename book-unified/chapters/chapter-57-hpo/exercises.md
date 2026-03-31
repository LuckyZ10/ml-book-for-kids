# 第五十七章 超参数调优进阶 练习题

## 练习题 1: 超参数空间设计 (⭐)

**目标**: 理解超参数搜索空间的设计原则

**题目**: 
为一个简单的CNN图像分类器设计超参数搜索空间：

```python
import numpy as np

# 你的任务：为以下超参数设计合理的搜索范围
search_space = {
    'learning_rate': _______,  # 学习率
    'batch_size': _______,     # 批量大小
    'dropout_rate': _______,   # Dropout比率
    'num_layers': _______,     # 网络层数
    'num_filters': _______     # 卷积核数量
}
```

**要求**: 
1. 每个参数至少提供3个候选值
2. 学习率使用**对数尺度**（为什么？）
3. 解释每个参数的范围选择理由

**思考问题**: 
- 为什么学习率通常在对数尺度上搜索？
- 如果搜索空间太大，会有什么问题？

---

## 练习题 2: 贝叶斯优化模拟 (⭐⭐)

**目标**: 理解贝叶斯优化的核心思想

**题目**: 
手动模拟一轮贝叶斯优化过程：

假设你要优化函数 $f(x) = -(x-2)^2 + 5$，搜索范围 $x \in [0, 4]$。

**初始观测**:
| x | f(x) |
|---|------|
| 0.5 | 2.75 |
| 1.0 | 4.00 |
| 3.0 | 4.00 |

**任务**: 
1. 假设使用高斯过程作为代理模型，画出大致的预测曲线（手绘或描述）
2. 计算每个候选点的**期望改进 (Expected Improvement)**：
   - 候选点: x = 1.5, 2.0, 2.5
   - 当前最优值: $f^* = 4.00$
   - 假设预测均值和方差：
     - x=1.5: μ=4.5, σ=0.5
     - x=2.0: μ=5.0, σ=0.8
     - x=2.5: μ=4.5, σ=0.5

3. 应该选择哪个点进行下一次评估？为什么？

**EI公式提示**: 
$$EI(x) = (\mu(x) - f^*) \Phi(Z) + \sigma(x) \phi(Z)$$
其中 $Z = (\mu(x) - f^*) / \sigma(x)$，$\Phi$是标准正态CDF，$\phi$是标准正态PDF

---

## 练习题 3: 多保真度优化策略 (⭐⭐)

**目标**: 理解Successive Halving和Hyperband的原理

**题目**: 
你有100个超参数配置需要评估，总预算为1000个epoch。使用Successive Halving算法：

**参数**:
- 初始配置数: n = 100
- 初始资源分配: r = 1 epoch/配置
- 淘汰因子: η = 3

**任务**: 
1. 计算每轮保留的配置数
2. 计算每轮每个配置分配的资源
3. 完成下表：

| 轮次 | 配置数 | 每配置资源 | 该轮总资源 | 累计资源 |
|-----|--------|-----------|-----------|---------|
| 1   | 100    | 1         | 100       | 100     |
| 2   | ?      | ?         | ?         | ?       |
| 3   | ?      | ?         | ?         | ?       |
| ... | ...    | ...       | ...       | ...     |

4. 与网格搜索（每个配置训练1000 epoch）相比，节省了多少计算资源？

---

## 练习题 4: 早停策略实现 (⭐⭐⭐)

**目标**: 实现一个简单的早停决策器

**题目**: 
实现一个基于学习曲线的早停策略：

```python
class EarlyStoppingScheduler:
    """早停调度器"""
    
    def __init__(self, patience=5, min_delta=0.001):
        """
        参数:
        - patience: 容忍轮数，连续patience轮无改善则停止
        - min_delta: 改善阈值，必须超过此值才算改善
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -float('inf')
        self.counter = 0
        self.should_stop = False
    
    def step(self, score):
        """
        记录当前轮次的性能分数
        返回: 是否应该停止
        """
        # 你的实现
        pass
    
    def get_status(self):
        """返回当前状态信息"""
        return {
            'best_score': self.best_score,
            'counter': self.counter,
            'should_stop': self.should_stop
        }


# 测试
scheduler = EarlyStoppingScheduler(patience=3)
scores = [0.7, 0.75, 0.76, 0.76, 0.761, 0.760, 0.759]

for i, score in enumerate(scores):
    stop = scheduler.step(score)
    status = scheduler.get_status()
    print(f"轮次 {i+1}: 分数={score}, 最佳={status['best_score']:.3f}, 计数={status['counter']}")
    if stop:
        print(f"*** 在第 {i+1} 轮触发早停 ***")
        break
```

**要求**: 
- 实现 `step()` 方法
- 处理性能波动（噪声）
- 考虑如何区分"过拟合"和"正常波动"

---

## 练习题 5: 采集函数对比 (⭐⭐⭐)

**目标**: 理解不同采集函数的特点

**题目**: 
假设你有一个一维优化问题，代理模型的预测如下：

| x | μ(x) | σ(x) |
|---|------|------|
| 1 | 3.0  | 0.5  |
| 2 | 4.0  | 0.8  |
| 3 | 3.5  | 0.3  |
| 4 | 5.0  | 1.0  |
| 5 | 4.5  | 0.6  |

当前最优值 $f^* = 4.2$

**任务**: 
计算每个点的以下采集函数值：

1. **Probability of Improvement (PI)**:
   $$PI(x) = P(f(x) > f^*) = \Phi\left(\frac{\mu(x) - f^*}{\sigma(x)}\right)$$

2. **Expected Improvement (EI)**

3. **Upper Confidence Bound (UCB)** with κ=2:
   $$UCB(x) = \mu(x) + \kappa \cdot \sigma(x)$$

**思考问题**: 
- 如果只考虑Exploitation（利用），应该选哪个点？
- 如果只考虑Exploration（探索），应该选哪个点？
- 哪种采集函数在工程实践中最常用？为什么？

---

## 练习题 6: AutoML流程设计 (⭐⭐⭐)

**目标**: 设计一个端到端的AutoML流程

**题目**: 
设计一个表格数据的AutoML系统架构：

**输入**: 
- 训练数据集（CSV格式）
- 任务类型：分类/回归
- 时间预算：1小时

**输出**: 
- 最优模型
- 性能报告
- 模型可解释性分析

**要求**: 
绘制系统架构图（可用文本流程图），包含以下组件：

```
┌─────────────┐
│  数据预处理  │
└──────┬──────┘
       │
       v
┌─────────────┐
│   ?         │
└──────┬──────┘
       │
       v
     ...
```

**必须包含的组件**: 
1. 数据预处理（缺失值、编码、缩放）
2. 特征工程（自动特征生成/选择）
3. 模型选择（尝试哪些模型？）
4. 超参数优化（用什么方法？）
5. 模型集成（如何组合多个模型？）
6. 结果验证（交叉验证策略）

**思考问题**: 
- 如果时间预算只有10分钟，你会如何调整策略？
- 如何避免AutoML过拟合？

---

## 练习题 7: NAS基础 (⭐⭐⭐⭐)

**目标**: 理解神经架构搜索的基本概念

**题目**: 
设计一个简单的搜索空间用于图像分类：

**搜索空间定义**: 
```python
search_space = {
    # 第1层
    'layer1': {
        'type': ['conv3x3', 'conv5x5', 'maxpool3x3'],
        'filters': [16, 32, 64],
        'activation': ['relu', 'leaky_relu']
    },
    # 第2层
    'layer2': {
        'type': ['conv3x3', 'conv5x5', 'skip_connection'],
        'filters': [32, 64, 128],
        'activation': ['relu', 'leaky_relu']
    },
    # 其他层...
}
```

**任务**: 
1. 计算上述搜索空间的总架构数（假设有3层，每层参数独立）
2. 如果每个架构训练需要1小时，遍历所有架构需要多长时间？
3. 设计一个"权重共享"策略来加速搜索
4. 讨论：NAS是否总是比人工设计好？什么情况下NAS可能没有优势？

---

## 练习题 8: 多目标权衡分析 (⭐⭐⭐)

**目标**: 理解帕累托最优的概念

**题目**: 
假设你在优化一个深度学习模型，有两个冲突目标：
- 目标A：准确率（越高越好）
- 目标B：推理延迟（越低越好）

以下是10个配置的结果：

| 配置 | 准确率(%) | 延迟(ms) |
|-----|----------|---------|
| A   | 85       | 50      |
| B   | 87       | 60      |
| C   | 90       | 80      |
| D   | 88       | 55      |
| E   | 92       | 120     |
| F   | 86       | 45      |
| G   | 89       | 70      |
| H   | 84       | 40      |
| I   | 91       | 100     |
| J   | 88       | 65      |

**任务**: 
1. 找出所有**帕累托最优**的配置
   - 提示：如果存在另一个配置在两个目标上都更好（或相等），则该配置被支配

2. 绘制帕累托前沿（文本描述或用ASCII字符画图）

3. 如果你的应用要求：
   - 延迟 < 70ms，你会选哪个配置？
   - 准确率 > 90%，你会选哪个配置？
   - 没有硬性约束，你会选哪个配置？为什么？

---

## 练习题 9: HPO完整项目 (⭐⭐⭐⭐⭐)

**目标**: 综合运用本章知识完成一个HPO项目

**题目**: 
使用Optuna对一个scikit-learn模型进行超参数优化：

**基础代码框架**: 
```python
import optuna
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def objective(trial):
    """优化目标函数"""
    # 定义搜索空间
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
    min_samples_split = trial.suggest_float('min_samples_split', 0.01, 0.5)
    
    # 创建模型
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    # 评估
    X, y = load_digits(return_X_y=True)
    score = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()
    
    return score

# 创建study并优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"最佳分数: {study.best_value:.4f}")
print(f"最佳参数: {study.best_params}")
```

**任务**: 
1. 运行上述代码，记录优化过程
2. 绘制优化历史图（使用optuna.visualization）
3. 分析哪些超参数对性能影响最大（使用feature importance）
4. **进阶**: 实现多目标优化（同时优化准确率和模型大小）
5. **进阶**: 使用HyperBand或ASHA进行多保真度优化

**交付物**: 
- 完整代码（包含可视化）
- 优化报告（包含参数重要性分析）
- 与随机搜索的对比实验

---

## 参考答案

### 练习2 贝叶斯优化模拟

对于x=2.0: μ=5.0, σ=0.8, f*=4.0
- Z = (5.0-4.0)/0.8 = 1.25
- Φ(1.25) ≈ 0.894
- φ(1.25) ≈ 0.183
- EI = (5.0-4.0)*0.894 + 0.8*0.183 = 0.894 + 0.146 = 1.04

### 练习3 多保真度优化

| 轮次 | 配置数 | 每配置资源 | 该轮总资源 | 累计资源 |
|-----|--------|-----------|-----------|---------|
| 1   | 100    | 1         | 100       | 100     |
| 2   | 33     | 3         | 99        | 199     |
| 3   | 11     | 9         | 99        | 298     |
| 4   | 3      | 27        | 81        | 379     |
| 5   | 1      | 81        | 81        | 460     |

网格搜索总资源: 100 × 1000 = 100,000
节省: (100000 - 460) / 100000 = 99.5%

### 练习4 早停策略

```python
def step(self, score):
    if score > self.best_score + self.min_delta:
        self.best_score = score
        self.counter = 0
    else:
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
    return self.should_stop
```

---

**学习建议**: 
- 练习1-3是基础，必做
- 练习4-6是进阶，理解贝叶斯优化的核心
- 练习7-9是实战，完成后可独立进行HPO项目

**推荐工具**: 
- Optuna: 最友好的HPO库
- Ray Tune: 分布式HPO
- WandB: 实验跟踪
- BoTorch: 研究级贝叶斯优化
