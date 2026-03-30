# 第五十七章 超参数调优进阶——从网格搜索到AutoML

> *"在机器学习的花园里，超参数是浇灌每一朵花的雨露。调得好，繁花似锦；调不好，枯萎凋零。"*

---

## 57.1 引言：为什么超参数调优如此重要？

想象一下，你是一位摄影师，面前有一台高端单反相机。相机上有三个关键参数：**光圈**（控制进光量）、**快门速度**（控制曝光时间）、**ISO感光度**（控制传感器敏感度）。这三个参数的不同组合，会产生截然不同的照片效果——有的曝光完美、细节清晰；有的过曝或欠曝，惨不忍睹。

机器学习中的**超参数调优**（Hyperparameter Optimization, HPO），就像摄影师调整相机参数一样重要。

### 57.1.1 超参数 vs 参数

在深入HPO之前，我们必须先明确一个关键区别：

| 类型 | 定义 | 例子 | 如何确定 |
|------|------|------|----------|
| **参数 (Parameters)** | 模型从数据中学习得到的值 | 神经网络的权重、决策树的划分阈值 | 通过训练自动优化 |
| **超参数 (Hyperparameters)** | 在训练前需要人为设定的配置 | 学习率、网络层数、批量大小 | 需要手动或通过HPO确定 |

**类比理解**：
- 参数就像画家在画布上画出的每一笔——由创作过程（训练）自然产生
- 超参数就像画家选择的画笔类型、颜料品牌和画布尺寸——需要在开始创作前决定

### 57.1.2 超参数调优的挑战

超参数调优之所以困难，主要有以下几个原因：

#### 1. 组合爆炸（Combinatorial Explosion）

假设我们要调优一个神经网络，考虑以下超参数：
- 学习率：10种选择 [1e-5, 1e-4, ..., 0.1]
- 隐藏层数：5种选择 [1, 2, 3, 4, 5]
- 每层神经元数：5种选择 [64, 128, 256, 512, 1024]
- Dropout率：5种选择 [0.1, 0.2, 0.3, 0.4, 0.5]
- 批量大小：4种选择 [16, 32, 64, 128]

**总的组合数 = 10 × 5 × 5 × 5 × 4 = 5,000 种！**

如果每个配置训练需要1小时，遍历所有组合需要**208天**！这还不包括更深层次的架构搜索。

#### 2. 评估成本高昂

每个超参数配置的评估都需要完整的模型训练，这在深度学习时代尤其昂贵：
- GPT-3级别的模型训练成本超过**460万美元**
- 即使是较小的ResNet模型，完整训练也需要数小时到数天

#### 3. 非凸、非光滑、带噪声

超参数与最终性能的关系具有以下特性：
- **非凸**（Non-convex）：可能存在多个局部最优解
- **非光滑**（Non-smooth）：超参数的微小变化可能导致性能的突然变化
- **带噪声**（Noisy）：由于随机初始化、数据打乱等，相同配置多次运行结果会有差异

### 57.1.3 从盲目尝试到智能优化

超参数调优方法经历了三个阶段的演进：

```
第一阶段：人工经验 (2000s)
    ↓ 问题：耗时、依赖专家、难以复现
第二阶段：系统性搜索 (2010-2015)
    ├── 网格搜索 (Grid Search)
    └── 随机搜索 (Random Search)
    ↓ 问题：资源浪费、探索效率低
第三阶段：智能优化 (2015-至今)
    ├── 贝叶斯优化 (Bayesian Optimization)
    ├── 多保真度优化 (Multi-fidelity Optimization)
    └── 自动化机器学习 (AutoML)
```

### 57.1.4 本章路线图

在本章中，我们将一起探索：

1. **贝叶斯优化**——像品酒师一样，每次尝试都积累经验，下次选择更明智
2. **多保真度优化**——用"快速品尝"预测"完整烹饪"的效果
3. **AutoML系统**——让机器自己学会如何配置机器

准备好了吗？让我们开始这段优化之旅！

---

## 57.2 网格搜索与随机搜索：基础但有效

在介绍高级方法之前，让我们先回顾两种基础的搜索策略。它们虽然简单，但在特定场景下仍然有效，而且理解它们有助于我们认识更高级方法的改进之处。

### 57.2.1 网格搜索：穷举的艺术

**网格搜索**（Grid Search）是最直观的超参数调优方法——它就像在网格上的每个交叉点都插一面旗帜。

#### 原理

对于每个超参数，我们定义一组离散的候选值。网格搜索会遍历所有可能的组合：

```python
# 超参数网格示例
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'num_layers': [2, 3, 4]
}

# 总组合数 = 3 × 3 × 3 = 27
```

#### 优点

1. **简单直观**：易于理解和实现
2. **可复现**：给定相同的网格，结果完全一致
3. **并行友好**：每个配置独立，天然支持并行化

#### 缺点

1. **维度灾难**：随着超参数数量增加，组合数指数爆炸
2. **资源浪费**：在重要性低的维度上浪费大量计算
3. **离散限制**：只能探索预定义的离散值，可能错过最优解

### 57.2.2 随机搜索：更聪明的采样

**随机搜索**（Random Search）由Bergstra和Bengio在2012年的经典论文提出。他们发现：在超参数调优中，**随机搜索往往比网格搜索更有效**！

#### 原理

不从网格中选取，而是在定义的搜索空间内**均匀随机采样**：

```python
import numpy as np

# 随机搜索示例
for trial in range(n_trials):
    config = {
        'learning_rate': 10 ** np.random.uniform(-5, -1),  # 对数均匀
        'batch_size': np.random.choice([16, 32, 64, 128]),
        'dropout': np.random.uniform(0.1, 0.5)
    }
    evaluate(config)
```

#### 为什么随机搜索更好？

让我们用一个可视化来解释：

```
网格搜索 (9个配置)              随机搜索 (9个配置)
┌─────────────────────┐        ┌─────────────────────┐
│ · · · · · · · · ·   │        │         ·           │
│ · · · · · · · · ·   │        │   ·         ·       │
│ · · · · · · · · ·   │        │       ·       ·     │
│ · · · ● ● ● · · ·   │        │ ·           ·       │
│ · · · ● ● ● · · ·   │        │     ·   ·     ·     │
│ · · · ● ● ● · · ·   │        │         ·           │
│ · · · · · · · · ·   │        │   ·       ·         │
│ · · · · · · · · ·   │        │         ·     ·     │
└─────────────────────┘        └─────────────────────┘

● = 有价值区域                    更好地覆盖有价值区域
```

**核心洞察**：
1. **超参数的重要性不均等**：通常只有少数几个超参数对性能影响巨大
2. **网格搜索的浪费**：在重要维度上只探索了少量值，在不重要维度上却过度探索
3. **随机搜索的优势**：每个重要维度都有更多机会取到不同值

### 57.2.3 代码实现：网格搜索 vs 随机搜索对比

让我们实现一个完整的对比实验：

```python
"""
57.2.3 网格搜索与随机搜索对比实验
演示为什么随机搜索通常比网格搜索更有效
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class HyperparameterSearch:
    """超参数搜索基类"""
    
    def __init__(self, objective_func):
        """
        初始化搜索器
        
        Args:
            objective_func: 损失函数，接受配置字典，返回性能分数
        """
        self.objective = objective_func
        self.history = []  # 记录所有尝试
        self.best_config = None
        self.best_score = float('-inf')
    
    def evaluate(self, config):
        """评估一个配置"""
        score = self.objective(config)
        self.history.append({
            'config': config.copy(),
            'score': score
        })
        
        if score > self.best_score:
            self.best_score = score
            self.best_config = config.copy()
        
        return score


class GridSearch(HyperparameterSearch):
    """网格搜索实现"""
    
    def search(self, param_grid):
        """
        执行网格搜索
        
        Args:
            param_grid: 字典，键是参数名，值是候选值列表
        """
        # 生成所有组合
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        
        total = 1
        for v in values:
            total *= len(v)
        print(f"网格搜索: 总共 {total} 个配置")
        
        for combo in product(*values):
            config = dict(zip(keys, combo))
            self.evaluate(config)
        
        return self.best_config, self.best_score


class RandomSearch(HyperparameterSearch):
    """随机搜索实现"""
    
    def search(self, param_distributions, n_iter=100, random_state=None):
        """
        执行随机搜索
        
        Args:
            param_distributions: 字典，键是参数名，值是采样函数
            n_iter: 迭代次数
            random_state: 随机种子
        """
        if random_state:
            np.random.seed(random_state)
        
        print(f"随机搜索: 总共 {n_iter} 个配置")
        
        for i in range(n_iter):
            config = {}
            for param_name, sample_func in param_distributions.items():
                config[param_name] = sample_func()
            self.evaluate(config)
        
        return self.best_config, self.best_score


# ========================================
# 实验：展示随机搜索的优势
# ========================================

def test_function(config):
    """
    测试损失函数：模拟一个有两个重要维度的场景
    其中第一个维度非常重要，第二个维度不太重要
    """
    x = config['important_param']
    y = config['less_important_param']
    z = config['noise_param']
    
    # 第一个维度有强烈的峰值效应
    # 第二个维度有微弱影响
    # 第三个维度几乎无影响（噪声）
    score = (
        10 * np.exp(-((x - 0.7) ** 2) / 0.01) +  # 重要维度：尖锐峰值在0.7
        2 * np.sin(y * np.pi) +                   # 次要维度：微弱波动
        0.1 * z +                                 # 噪声维度
        np.random.normal(0, 0.1)                  # 观测噪声
    )
    
    return score


def run_comparison():
    """运行对比实验"""
    
    print("=" * 60)
    print("网格搜索 vs 随机搜索对比实验")
    print("=" * 60)
    
    # 定义搜索空间
    # 注意：网格搜索在"important_param"上只有5个采样点
    # 而随机搜索可以有更多机会命中最优区域附近
    param_grid = {
        'important_param': [0.0, 0.25, 0.5, 0.75, 1.0],
        'less_important_param': [0.0, 0.5, 1.0],
        'noise_param': [0, 1, 2]
    }
    
    # 随机搜索的采样函数
    def sample_important():
        """在重要维度上密集采样"""
        return np.random.uniform(0, 1)
    
    def sample_less_important():
        return np.random.choice([0.0, 0.5, 1.0])
    
    def sample_noise():
        return np.random.choice([0, 1, 2])
    
    param_distributions = {
        'important_param': sample_important,
        'less_important_param': sample_less_important,
        'noise_param': sample_noise
    }
    
    # 运行网格搜索
    print("\n[1] 运行网格搜索...")
    gs = GridSearch(test_function)
    gs_start = time.time()
    gs_best_config, gs_best_score = gs.search(param_grid)
    gs_time = time.time() - gs_start
    
    print(f"    最佳配置: {gs_best_config}")
    print(f"    最佳分数: {gs_best_score:.4f}")
    print(f"    评估次数: {len(gs.history)}")
    print(f"    耗时: {gs_time:.2f}秒")
    
    # 运行随机搜索（相同评估次数）
    print("\n[2] 运行随机搜索...")
    rs = RandomSearch(test_function)
    rs_start = time.time()
    rs_best_config, rs_best_score = rs.search(
        param_distributions, 
        n_iter=len(gs.history),
        random_state=42
    )
    rs_time = time.time() - rs_start
    
    print(f"    最佳配置: {rs_best_config}")
    print(f"    最佳分数: {rs_best_score:.4f}")
    print(f"    评估次数: {len(rs.history)}")
    print(f"    耗时: {rs_time:.2f}秒")
    
    # 结果对比
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"网格搜索最佳分数: {gs_best_score:.4f}")
    print(f"随机搜索最佳分数: {rs_best_score:.4f}")
    print(f"提升幅度: {((rs_best_score - gs_best_score) / abs(gs_best_score) * 100):.2f}%")
    
    # 可视化
    visualize_results(gs, rs)
    
    return gs, rs


def visualize_results(gs, rs):
    """可视化搜索结果"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 提取数据
    gs_configs = [h['config'] for h in gs.history]
    gs_scores = [h['score'] for h in gs.history]
    gs_x = [c['important_param'] for c in gs_configs]
    gs_y = [c['less_important_param'] for c in gs_configs]
    
    rs_configs = [h['config'] for h in rs.history]
    rs_scores = [h['score'] for h in rs.history]
    rs_x = [c['important_param'] for c in rs_configs]
    rs_y = [c['less_important_param'] for c in rs_configs]
    
    # 1. 网格搜索采样点分布
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(gs_x, gs_y, c=gs_scores, cmap='viridis', 
                           s=100, alpha=0.7, edgecolors='black')
    ax1.set_xlabel('Important Parameter', fontsize=11)
    ax1.set_ylabel('Less Important Parameter', fontsize=11)
    ax1.set_title('Grid Search: Sampling Points\n(Regular Grid Pattern)', fontsize=12)
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='True Optimum')
    ax1.legend()
    plt.colorbar(scatter1, ax=ax1, label='Score')
    
    # 2. 随机搜索采样点分布
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(rs_x, rs_y, c=rs_scores, cmap='viridis', 
                           s=100, alpha=0.7, edgecolors='black')
    ax2.set_xlabel('Important Parameter', fontsize=11)
    ax2.set_ylabel('Less Important Parameter', fontsize=11)
    ax2.set_title('Random Search: Sampling Points\n(Uniform Coverage)', fontsize=12)
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='True Optimum')
    ax2.legend()
    plt.colorbar(scatter2, ax=ax2, label='Score')
    
    # 3. 收敛曲线对比
    ax3 = axes[1, 0]
    gs_cumulative_max = np.maximum.accumulate(gs_scores)
    rs_cumulative_max = np.maximum.accumulate(rs_scores)
    
    ax3.plot(range(1, len(gs_cumulative_max)+1), gs_cumulative_max, 
             'b-', linewidth=2, label='Grid Search', marker='o', markersize=4)
    ax3.plot(range(1, len(rs_cumulative_max)+1), rs_cumulative_max, 
             'r-', linewidth=2, label='Random Search', marker='s', markersize=4)
    ax3.set_xlabel('Number of Evaluations', fontsize=11)
    ax3.set_ylabel('Best Score So Far', fontsize=11)
    ax3.set_title('Convergence Comparison', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 重要维度上的投影
    ax4 = axes[1, 1]
    ax4.scatter(gs_x, gs_scores, c='blue', alpha=0.6, s=80, 
                label='Grid Search', edgecolors='black')
    ax4.scatter(rs_x, rs_scores, c='red', alpha=0.6, s=80, 
                label='Random Search', edgecolors='black', marker='s')
    ax4.axvline(x=0.7, color='green', linestyle='--', linewidth=2, 
                alpha=0.7, label='True Optimum (x=0.7)')
    ax4.set_xlabel('Important Parameter Value', fontsize=11)
    ax4.set_ylabel('Score', fontsize=11)
    ax4.set_title('Score vs Important Parameter\n(Random Search covers better)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('grid_vs_random_search.png', dpi=150, bbox_inches='tight')
    print("\n可视化结果已保存到: grid_vs_random_search.png")
    plt.show()


if __name__ == "__main__":
    gs, rs = run_comparison()
```

### 57.2.4 实验结果分析

运行上述代码，你会观察到：

1. **网格搜索**在"important_param=0.7"（真实最优值附近）可能**没有任何采样点**，因为网格是固定的0.0, 0.25, 0.5, 0.75, 1.0，错过了0.7附近的黄金区域。

2. **随机搜索**在重要维度上有更均匀的覆盖，因此**更可能命中或接近最优区域**。

3. **收敛曲线**显示，随机搜索通常能更快找到更好的解。

**关键结论**：
- 当超参数重要性不均等时，随机搜索优于网格搜索
- 当最优解位于网格点之间时，随机搜索更有优势
- 对于高维搜索空间，随机搜索的可扩展性更好

---

## 57.3 贝叶斯优化：智能探索的艺术

随机搜索虽然比网格搜索更高效，但它仍然是"盲目"的——每次尝试都不利用之前的信息。

**贝叶斯优化**（Bayesian Optimization, BO）改变了这一点。它像一个**经验丰富的品酒师**：每次品尝后都会积累经验，下一次选择更有可能好喝的酒。

### 57.3.1 贝叶斯优化的核心思想

贝叶斯优化的核心思想可以概括为：

```
1. 基于已观察的数据，建立一个对损失函数的代理模型
2. 使用采集函数决定下一个最有价值的采样点
3. 在新的点评估损失函数
4. 更新代理模型，重复步骤1-3
```

#### 为什么叫"贝叶斯"？

因为它使用了**贝叶斯定理**的思想：
- **先验**（Prior）：在观察任何数据之前，我们对损失函数的假设
- **似然**（Likelihood）：观察到的数据
- **后验**（Posterior）：结合先验和数据后，对损失函数的更新认识

在贝叶斯优化中，**高斯过程**（Gaussian Process, GP）充当了这个"概率模型"的角色。

### 57.3.2 高斯过程：函数上的概率分布

#### 什么是高斯过程？

**高斯过程**是一种强大的非参数模型，它定义了**函数上的概率分布**。与普通分布定义在数值上不同，GP定义在函数上！

**类比理解**：
- 普通高斯分布：$x \sim \mathcal{N}(\mu, \sigma^2)$，描述一个随机变量的分布
- 高斯过程：$f \sim \mathcal{GP}(m(x), k(x, x'))$，描述**整个函数**的分布

#### 高斯过程的数学定义

一个高斯过程由两个函数完全确定：

1. **均值函数**（Mean Function）：$m(x) = \mathbb{E}[f(x)]$
2. **协方差函数/核函数**（Kernel Function）：$k(x, x') = \mathbb{E}[(f(x) - m(x))(f(x') - m(x'))]$

通常，我们使用**零均值**假设：$m(x) = 0$

#### 核函数：定义函数的形状

核函数决定了高斯过程的性质。两个最常用的核函数：

**1. RBF核（径向基函数/高斯核）**：
$$k(x, x') = \sigma^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$

- $\sigma^2$：输出尺度（函数值的波动幅度）
- $\ell$：长度尺度（函数变化的平滑程度）

**2. Matérn核**（更灵活的平滑度控制）：
$$k_{\nu}(x, x') = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}\|x - x'\|}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}\|x - x'\|}{\ell}\right)$$

当$\nu = 5/2$时，Matérn核产生两次可微的函数，比RBF更灵活。

### 57.3.3 高斯过程回归：预测与不确定性

给定训练数据 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$，其中 $y_i = f(x_i) + \epsilon_i$，$\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$，GP回归的目标是预测新点 $x_*$ 的函数值 $f(x_*)$。

#### 后验分布的推导

定义：
- $X = [x_1, x_2, ..., x_n]^T$：训练输入
- $y = [y_1, y_2, ..., y_n]^T$：训练输出
- $K_{ij} = k(x_i, x_j)$：核矩阵
- $k_* = [k(x_*, x_1), ..., k(x_*, x_n)]^T$：新点与训练点的核向量

**后验均值**（预测值）：
$$\mu_* = k_*^T(K + \sigma_n^2 I)^{-1}y$$

**后验方差**（预测不确定性）：
$$\sigma_*^2 = k(x_*, x_*) - k_*^T(K + \sigma_n^2 I)^{-1}k_*$$

这个公式告诉我们：
1. **预测值**是训练输出的加权平均，权重由核函数决定
2. **不确定性**在远离训练数据的地方增大，在训练数据附近减小

### 57.3.4 从零实现高斯过程回归

现在，让我们亲手实现一个完整的高斯过程回归模型。这不仅有助于理解GP的工作原理，也为后续构建贝叶斯优化器打下基础。

**费曼法比喻**：想象你是一位地质学家，要在一片未知区域寻找金矿。你已经在几个地点钻探取样（训练数据）。高斯过程就像一张"预测地图"，不仅告诉你每个位置可能有多少金子（均值），还告诉你这个预测的可靠程度（方差）。在已钻探的地方，你很确定；在远离钻探点的地方，不确定性就很大。

#### 完整的GP回归实现

```python
"""
57.3.4 从零实现高斯过程回归
包含RBF核函数、GP回归类、以及不确定性可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

# 设置随机种子以保证可复现性
np.random.seed(42)


class RBFKernel:
    """
    径向基函数(RBF)核，也称高斯核
    
    数学公式: k(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 * l^2))
    
    参数:
        length_scale (l): 长度尺度，控制函数的平滑程度
                        值越大，函数变化越缓慢，越平滑
        sigma_f: 输出信号的标准差，控制函数的振幅
    """
    
    def __init__(self, length_scale=1.0, sigma_f=1.0):
        self.length_scale = length_scale
        self.sigma_f = sigma_f
    
    def __call__(self, X1, X2=None):
        """
        计算核矩阵
        
        Args:
            X1: 形状 (n1, d) 的输入矩阵
            X2: 形状 (n2, d) 的输入矩阵，若为None则计算X1与自身的核
            
        Returns:
            K: 形状 (n1, n2) 的核矩阵
        """
        if X2 is None:
            X2 = X1
        
        # 将输入转换为至少二维数组
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        
        # 计算欧几里得距离的平方: ||x - x'||^2
        # 使用广播技巧: (a-b)^2 = a^2 + b^2 - 2ab
        sq_dists = (
            np.sum(X1**2, axis=1).reshape(-1, 1) + 
            np.sum(X2**2, axis=1) - 
            2 * np.dot(X1, X2.T)
        )
        
        # RBF核公式
        K = self.sigma_f**2 * np.exp(-0.5 * sq_dists / self.length_scale**2)
        return K
    
    def set_params(self, length_scale, sigma_f):
        """更新核参数"""
        self.length_scale = length_scale
        self.sigma_f = sigma_f


class GaussianProcessRegressor:
    """
    高斯过程回归器
    
    使用Cholesky分解高效求解线性系统，数值稳定性更好
    """
    
    def __init__(self, kernel=None, noise_level=1e-5, optimize_hyperparams=True):
        """
        初始化GP回归器
        
        Args:
            kernel: 核函数对象，默认使用RBF核
            noise_level: 观测噪声的标准差
            optimize_hyperparams: 是否自动优化超参数
        """
        self.kernel = kernel if kernel is not None else RBFKernel()
        self.noise_level = noise_level
        self.optimize_hyperparams = optimize_hyperparams
        
        # 训练数据存储
        self.X_train = None
        self.y_train = None
        
        # Cholesky分解的缓存
        self.L = None  # 下三角矩阵
        self.alpha = None  # (K + sigma^2 I)^{-1} y
        
    def fit(self, X, y):
        """
        训练GP模型
        
        Args:
            X: 训练输入，形状 (n_samples, n_features)
            y: 训练输出，形状 (n_samples,) 或 (n_samples, 1)
        """
        self.X_train = np.atleast_2d(X)
        self.y_train = np.atleast_1d(y).reshape(-1, 1)
        
        # 如果需要，优化超参数
        if self.optimize_hyperparams:
            self._optimize_hyperparams()
        
        # 计算核矩阵 K
        K = self.kernel(self.X_train, self.X_train)
        
        # 添加噪声项: K_y = K + sigma_n^2 * I
        K_y = K + self.noise_level**2 * np.eye(len(self.X_train))
        
        # Cholesky分解: K_y = L L^T
        # 这使得求解线性系统更稳定和高效
        try:
            self.L = cholesky(K_y, lower=True)
        except np.linalg.LinAlgError:
            # 如果矩阵不正定，添加小的抖动项
            K_y += 1e-6 * np.eye(len(self.X_train))
            self.L = cholesky(K_y, lower=True)
        
        # 求解 alpha = (K_y)^{-1} y
        # 利用Cholesky分解: 先解 L v = y, 再解 L^T alpha = v
        v = solve_triangular(self.L, self.y_train, lower=True)
        self.alpha = solve_triangular(self.L.T, v, lower=False)
        
        return self
    
    def predict(self, X_test, return_std=True, return_cov=False):
        """
        对新输入进行预测
        
        Args:
            X_test: 测试输入，形状 (n_test, n_features)
            return_std: 是否返回预测标准差
            return_cov: 是否返回预测协方差矩阵
            
        Returns:
            y_mean: 预测均值
            y_std/y_cov: 预测标准差或协方差（根据return_std/return_cov）
        """
        X_test = np.atleast_2d(X_test)
        
        # 计算测试点与训练点的核向量: k_*
        k_star = self.kernel(X_test, self.X_train)
        
        # 预测均值: mu_* = k_*^T alpha
        y_mean = np.dot(k_star, self.alpha).ravel()
        
        if not return_std and not return_cov:
            return y_mean
        
        # 预测方差的计算
        # v = solve(L, k_*^T)，用于数值稳定
        v = solve_triangular(self.L, k_star.T, lower=True)
        
        # 测试点自身的核值
        k_star_star = np.diag(self.kernel(X_test, X_test))
        
        if return_cov:
            # 完整协方差矩阵: K_** - v^T v
            y_cov = self.kernel(X_test, X_test) - np.dot(v.T, v)
            return y_mean, y_cov
        
        if return_std:
            # 只返回标准差（对角线元素）
            y_var = k_star_star - np.sum(v**2, axis=0)
            # 数值稳定性处理，确保方差非负
            y_var = np.maximum(y_var, 1e-10)
            y_std = np.sqrt(y_var)
            return y_mean, y_std
    
    def log_marginal_likelihood(self, params=None):
        """
        计算对数边缘似然，用于超参数优化
        
        公式: log p(y|X) = -1/2 y^T K^{-1} y - 1/2 log|K| - n/2 log(2*pi)
        """
        if params is not None:
            # 临时设置参数
            original_params = (self.kernel.length_scale, self.kernel.sigma_f)
            self.kernel.set_params(params[0], params[1])
            K = self.kernel(self.X_train, self.X_train) + \
                self.noise_level**2 * np.eye(len(self.X_train))
            self.kernel.set_params(*original_params)
        else:
            K = self.kernel(self.X_train, self.X_train) + \
                self.noise_level**2 * np.eye(len(self.X_train))
        
        # 使用Cholesky分解计算
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return -np.inf
        
        # 求解 alpha
        v = solve_triangular(L, self.y_train, lower=True)
        alpha = solve_triangular(L.T, v, lower=False)
        
        # 计算对数边缘似然的各项
        # -1/2 y^T alpha
        data_fit = -0.5 * np.dot(self.y_train.T, alpha).ravel()[0]
        # - log|L| = - sum(log(diag(L)))
        complexity = -np.sum(np.log(np.diag(L)))
        # - n/2 log(2*pi)
        constant = -0.5 * len(self.X_train) * np.log(2 * np.pi)
        
        return data_fit + complexity + constant
    
    def _optimize_hyperparams(self):
        """通过最大化边缘似然来优化核超参数"""
        
        def neg_log_likelihood(params):
            """负对数边缘似然（最小化目标）"""
            if params[0] <= 0 or params[1] <= 0:
                return 1e10
            return -self.log_marginal_likelihood(params)
        
        # 初始猜测
        x0 = np.array([self.kernel.length_scale, self.kernel.sigma_f])
        
        # 使用L-BFGS-B优化
        bounds = [(1e-5, None), (1e-5, None)]
        result = minimize(neg_log_likelihood, x0, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            self.kernel.set_params(result.x[0], result.x[1])
            print(f"超参数优化完成: length_scale={result.x[0]:.4f}, sigma_f={result.x[1]:.4f}")


# ========================================
# 可视化GP回归的效果
# ========================================

def plot_gp_regression_1d():
    """
    一维GP回归可视化
    展示GP如何拟合函数以及不确定性如何变化
    """
    
    # 定义真实函数（待拟合的未知函数）
    def true_function(x):
        return np.sin(x) * np.exp(-0.1 * x**2) + 0.1 * x
    
    # 生成训练数据（少量观测点）
    np.random.seed(42)
    X_train = np.array([-4, -2, 0, 1, 3]).reshape(-1, 1)
    y_train = true_function(X_train).ravel() + np.random.normal(0, 0.1, len(X_train))
    
    # 创建测试点
    X_test = np.linspace(-5, 5, 200).reshape(-1, 1)
    y_true = true_function(X_test)
    
    # 创建并训练GP模型
    kernel = RBFKernel(length_scale=1.0, sigma_f=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, noise_level=0.1, optimize_hyperparams=True)
    gpr.fit(X_train, y_train)
    
    # 预测
    y_mean, y_std = gpr.predict(X_test, return_std=True)
    
    # 绘制结果
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 先验采样（训练前）
    ax1 = axes[0, 0]
    K_prior = kernel(X_test, X_test)
    # 从多元高斯分布采样
    n_samples = 5
    samples_prior = np.random.multivariate_normal(np.zeros(len(X_test)), K_prior, n_samples)
    ax1.plot(X_test, samples_prior.T, alpha=0.5)
    ax1.set_title('Prior Samples (Before Training)', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 2. 后验采样（训练后）
    ax2 = axes[0, 1]
    y_mean_plot, y_cov = gpr.predict(X_test, return_cov=True)
    samples_posterior = np.random.multivariate_normal(y_mean_plot, y_cov, n_samples)
    ax2.plot(X_test, samples_posterior.T, alpha=0.5)
    ax2.scatter(X_train, y_train, c='red', s=100, zorder=5, label='Training Data')
    ax2.set_title('Posterior Samples (After Training)', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.legend()
    
    # 3. 预测均值和置信区间
    ax3 = axes[1, 0]
    ax3.fill_between(X_test.ravel(), y_mean - 1.96*y_std, y_mean + 1.96*y_std,
                     alpha=0.3, color='blue', label='95% Confidence Interval')
    ax3.plot(X_test, y_mean, 'b-', linewidth=2, label='GP Prediction')
    ax3.plot(X_test, y_true, 'g--', linewidth=2, label='True Function')
    ax3.scatter(X_train, y_train, c='red', s=100, zorder=5, label='Training Data')
    ax3.set_title('GP Regression with Uncertainty', fontsize=12)
    ax3.set_xlabel('x')
    ax3.set_ylabel('f(x)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 不确定性可视化
    ax4 = axes[1, 1]
    ax4.fill_between(X_test.ravel(), y_mean - 2*y_std, y_mean + 2*y_std,
                     alpha=0.2, color='purple', label='±2σ')
    ax4.fill_between(X_test.ravel(), y_mean - y_std, y_mean + y_std,
                     alpha=0.3, color='blue', label='±1σ')
    ax4.plot(X_test, y_std, 'r-', linewidth=2, label='Standard Deviation')
    ax4.scatter(X_train, np.zeros_like(y_train), c='green', s=100, 
                marker='|', zorder=5, label='Training Points')
    ax4.set_title('Uncertainty (Std Dev) vs Distance from Data', fontsize=12)
    ax4.set_xlabel('x')
    ax4.set_ylabel('Uncertainty')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gp_regression_demo.png', dpi=150, bbox_inches='tight')
    print("\nGP回归可视化已保存到: gp_regression_demo.png")
    plt.show()


if __name__ == "__main__":
    plot_gp_regression_1d()
```

#### 代码解析

**1. RBF核函数的实现**
```python
K = sigma_f**2 * np.exp(-0.5 * sq_dists / length_scale**2)
```
- 当两点距离很近时，核值接近$\sigma_f^2$（完全相关）
- 当距离很远时，核值接近0（不相关）
- `length_scale`控制这种"相关性"随距离的衰减速度

**2. 后验均值和方差的计算**
- 利用Cholesky分解$K = LL^T$将求逆转化为解三角系统
- 时间复杂度从$O(n^3)$降低到$O(n^3)$（分解仍是$O(n^3)$，但常数更小）
- 数值稳定性更好

**3. 不确定性可视化**
- 在训练数据点处，方差最小（我们有观测值）
- 远离训练数据时，方差增大（不确定性增加）
- 这种特性对贝叶斯优化至关重要——它指导我们在哪里采样

运行上述代码，你会看到四幅图：
1. **先验采样**：训练前，GP从先验分布采样函数
2. **后验采样**：训练后，采样函数都经过训练点
3. **预测结果**：蓝色区域表示95%置信区间，真实函数几乎都在区间内
4. **不确定性**：展示方差如何随距离训练点的远近变化

现在我们已经有了GP回归的实现，下一步是实现**采集函数**来决定下一个采样点。

---

### 57.3.5 采集函数：平衡探索与利用

有了高斯过程模型，我们可以对任意点$x$预测$f(x)$的均值$\mu(x)$和方差$\sigma^2(x)$。但问题是：**下一个采样点应该选在哪里？**

这就是**采集函数**（Acquisition Function）的作用。它量化了在每个点采样的"价值"，我们选择采集函数值最大的点进行下一次评估。

**费曼法比喻**：想象你是一家餐厅的品鉴师，要找出最好吃的菜。你面前有两类选择：
- **利用（Exploitation）**：去那家评分很高的餐厅，可能吃到好吃的（但可能错过更好的）
- **探索（Exploration）**：去那家没人去过的新餐厅，有风险但也可能有惊喜

采集函数就是帮你在这两者之间找到平衡的策略。

#### 三种经典采集函数

**1. 改进概率 (Probability of Improvement, PI)**

PI选择最可能改进当前最佳值的点。设$f^+$是当前最佳观测值：

$$\text{PI}(x) = P(f(x) > f^+) = \Phi\left(\frac{\mu(x) - f^+ - \xi}{\sigma(x)}\right)$$

其中：
- $\Phi$是标准正态分布的累积分布函数（CDF）
- $\xi$是探索参数（避免过早收敛到局部最优）

**直观理解**：选择"我的下一道菜比目前最好吃的还好的概率"最大的餐厅。

---

**2. 期望改进 (Expected Improvement, EI)**

PI只关心"有没有改进"，但EI还关心"改进多少"。它是贝叶斯优化中最常用的采集函数：

$$\text{EI}(x) = \mathbb{E}\left[\max(0, f(x) - f^+)\right]$$

对于高斯分布，EI有闭式解：

$$\text{EI}(x) = (\mu(x) - f^+ - \xi)\Phi(Z) + \sigma(x)\phi(Z)$$

其中：
$$Z = \frac{\mu(x) - f^+ - \xi}{\sigma(x)}$$

- $\phi$是标准正态分布的概率密度函数（PDF）
- 当$\sigma(x) = 0$时，$\text{EI}(x) = 0$

**推导过程**：

令$u = f(x) - f^+ - \xi$，则$u \sim \mathcal{N}(\mu(x) - f^+ - \xi, \sigma^2(x))$

$$\text{EI} = \int_0^{\infty} u \cdot \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(u-\mu')^2}{2\sigma^2}\right) du$$

通过变量替换$v = \frac{u - \mu'}{\sigma}$，可以得到上述闭式解。

**直观理解**：EI不仅关心"能更好"，还关心"能好多少"。一道可能稍微好一点但概率很大的菜，和一道可能好很多但概率较小的菜，EI会选择期望收益最大的。

---

**3. 上置信界 (Upper Confidence Bound, UCB)**

UCB是一种更简单的策略，直接平衡均值和不确定性：

$$\text{UCB}(x) = \mu(x) + \kappa \sigma(x)$$

其中$\kappa$是调节参数：
- $\kappa = 0$：纯利用（选择均值最大的点）
- $\kappa \to \infty$：纯探索（选择不确定性最大的点）

**直观理解**："这道菜平均得分8分，但我不是很确定（方差大），所以实际可能是10分也可能是6分。UCB取一个乐观的估计：8 + 2 = 10分。"

对于优化问题，UCB被称为**GP-UCB**，其理论保证通过选择$\kappa = \sqrt{2\log(t^{d/2+2}\pi^2/3\delta)}$可以获得次线性遗憾（sublinear regret）。

#### 采集函数的实现与对比

```python
"""
57.3.5 采集函数的实现与可视化
对比PI、EI和UCB三种采集函数的行为差异
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class AcquisitionFunction:
    """采集函数基类"""
    
    def __init__(self, xi=0.01):
        """
        Args:
            xi: 探索参数，防止过早收敛
        """
        self.xi = xi
    
    def __call__(self, mu, sigma, f_best):
        """
        计算采集函数值
        
        Args:
            mu: 预测均值，形状 (n,)
            sigma: 预测标准差，形状 (n,)
            f_best: 当前最佳观测值
            
        Returns:
            acq_values: 采集函数值，形状 (n,)
        """
        raise NotImplementedError


class ProbabilityOfImprovement(AcquisitionFunction):
    """
    改进概率 (PI) 采集函数
    
    PI(x) = P(f(x) > f_best) = Φ((μ(x) - f_best - ξ) / σ(x))
    """
    
    def __call__(self, mu, sigma, f_best):
        # 避免除零
        sigma = np.maximum(sigma, 1e-9)
        
        # 标准化变量
        Z = (mu - f_best - self.xi) / sigma
        
        # 标准正态CDF
        pi = norm.cdf(Z)
        
        return pi


class ExpectedImprovement(AcquisitionFunction):
    """
    期望改进 (EI) 采集函数
    
    EI(x) = (μ(x) - f_best - ξ) * Φ(Z) + σ(x) * φ(Z)
    其中 Z = (μ(x) - f_best - ξ) / σ(x)
    """
    
    def __call__(self, mu, sigma, f_best):
        # 避免除零
        sigma = np.maximum(sigma, 1e-9)
        
        # 标准化变量
        Z = (mu - f_best - self.xi) / sigma
        
        # EI的闭式解
        # 第一项: (μ - f_best - ξ) * Φ(Z)
        improvement = mu - f_best - self.xi
        ei = improvement * norm.cdf(Z)
        
        # 第二项: σ * φ(Z)
        ei += sigma * norm.pdf(Z)
        
        # 当方差为0时，EI应为0（没有不确定性，也没有改进可能）
        ei[sigma < 1e-9] = 0
        
        return ei


class UpperConfidenceBound(AcquisitionFunction):
    """
    上置信界 (UCB) 采集函数
    
    UCB(x) = μ(x) + κ * σ(x)
    
    注意：UCB不需要f_best参数，但为了接口统一，保留该参数
    """
    
    def __init__(self, kappa=2.0):
        """
        Args:
            kappa: 探索参数，越大越倾向于探索
        """
        self.kappa = kappa
    
    def __call__(self, mu, sigma, f_best=None):
        return mu + self.kappa * sigma


def plot_acquisition_functions():
    """
    可视化三种采集函数的行为
    展示它们如何平衡探索与利用
    """
    
    # 假设的损失函数（GP预测结果）
    x = np.linspace(0, 10, 500)
    
    # 模拟GP预测：一个多峰函数
    # 均值函数
    mu = np.sin(x) + 0.5 * np.sin(3*x) + 0.3 * x - 2
    
    # 方差函数：在"已观测"点（x=2, x=5, x=8）附近小，远离时大
    sigma = 0.5 + 0.3 * (
        np.exp(-0.5 * (x - 2)**2) + 
        np.exp(-0.5 * (x - 5)**2) + 
        np.exp(-0.5 * (x - 8)**2)
    )
    sigma = 1.5 - sigma
    sigma = np.maximum(sigma, 0.1)
    
    # 当前最佳值
    f_best = np.max(mu)
    
    # 创建采集函数实例
    pi_acq = ProbabilityOfImprovement(xi=0.1)
    ei_acq = ExpectedImprovement(xi=0.1)
    ucb_acq = UpperConfidenceBound(kappa=2.0)
    
    # 计算采集函数值
    pi_values = pi_acq(mu, sigma, f_best)
    ei_values = ei_acq(mu, sigma, f_best)
    ucb_values = ucb_acq(mu, sigma, f_best)
    
    # 找到每个采集函数的最大值点
    pi_max_idx = np.argmax(pi_values)
    ei_max_idx = np.argmax(ei_values)
    ucb_max_idx = np.argmax(ucb_values)
    
    # 绘制
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. 均值和方差
    ax1 = axes[0]
    ax1.fill_between(x, mu - 2*sigma, mu + 2*sigma, alpha=0.3, color='blue',
                     label='95% Confidence')
    ax1.plot(x, mu, 'b-', linewidth=2, label='GP Mean μ(x)')
    ax1.axhline(y=f_best, color='red', linestyle='--', linewidth=2,
                label=f'Current Best f* = {f_best:.2f}')
    ax1.set_ylabel('f(x)', fontsize=11)
    ax1.set_title('GP Prediction (Mean and Uncertainty)', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. PI和EI对比
    ax2 = axes[1]
    ax2.plot(x, pi_values, 'g-', linewidth=2, label='PI(x)')
    ax2.plot(x, ei_values, 'm-', linewidth=2, label='EI(x)')
    ax2.axvline(x=x[pi_max_idx], color='green', linestyle='--', alpha=0.5,
                label=f'PI max at x={x[pi_max_idx]:.2f}')
    ax2.axvline(x=x[ei_max_idx], color='magenta', linestyle='--', alpha=0.5,
                label=f'EI max at x={x[ei_max_idx]:.2f}')
    ax2.set_ylabel('Acquisition Value', fontsize=11)
    ax2.set_title('Probability of Improvement vs Expected Improvement', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. UCB
    ax3 = axes[2]
    ax3.plot(x, mu, 'b--', linewidth=1.5, alpha=0.7, label='μ(x)')
    ax3.plot(x, ucb_values, 'r-', linewidth=2, label=f'UCB(x) = μ(x) + {ucb_acq.kappa}σ(x)')
    ax3.fill_between(x, mu, ucb_values, alpha=0.2, color='red', label='Exploration bonus')
    ax3.axvline(x=x[ucb_max_idx], color='red', linestyle='--', alpha=0.5,
                label=f'UCB max at x={x[ucb_max_idx]:.2f}')
    ax3.set_xlabel('x', fontsize=11)
    ax3.set_ylabel('Acquisition Value', fontsize=11)
    ax3.set_title('Upper Confidence Bound', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('acquisition_functions.png', dpi=150, bbox_inches='tight')
    print("\n采集函数对比图已保存到: acquisition_functions.png")
    plt.show()
    
    print(f"\n三种采集函数选择的最优点:")
    print(f"  PI  选择: x = {x[pi_max_idx]:.3f}")
    print(f"  EI  选择: x = {x[ei_max_idx]:.3f}")
    print(f"  UCB 选择: x = {x[ucb_max_idx]:.3f}")


def demonstrate_exploration_exploitation():
    """
    演示不同探索参数对采集函数行为的影响
    """
    
    x = np.linspace(0, 10, 500)
    
    # 模拟GP预测
    mu = 2 * np.sin(x) + 0.3 * x
    sigma = 0.5 + 0.8 * np.exp(-0.3 * (x - 5)**2)
    
    f_best = 1.5  # 当前最佳
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 不同xi值的PI
    ax1 = axes[0, 0]
    for xi in [0.0, 0.1, 0.5]:
        pi = ProbabilityOfImprovement(xi=xi)
        values = pi(mu, sigma, f_best)
        ax1.plot(x, values, linewidth=2, label=f'ξ = {xi}')
    ax1.set_title('PI with Different ξ Values', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('PI(x)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 不同xi值的EI
    ax2 = axes[0, 1]
    for xi in [0.0, 0.1, 0.5]:
        ei = ExpectedImprovement(xi=xi)
        values = ei(mu, sigma, f_best)
        ax2.plot(x, values, linewidth=2, label=f'ξ = {xi}')
    ax2.set_title('EI with Different ξ Values', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('EI(x)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 不同kappa值的UCB
    ax3 = axes[1, 0]
    for kappa in [0.5, 1.0, 2.0, 3.0]:
        ucb = UpperConfidenceBound(kappa=kappa)
        values = ucb(mu, sigma)
        ax3.plot(x, values, linewidth=2, label=f'κ = {kappa}')
    ax3.set_title('UCB with Different κ Values', fontsize=12)
    ax3.set_xlabel('x')
    ax3.set_ylabel('UCB(x)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 探索-利用可视化
    ax4 = axes[1, 1]
    
    # 标记当前最佳点
    best_idx = np.argmax(mu - sigma * 0.5)  # 折中位置
    
    # 利用：选择均值最大的点
    exploit_idx = np.argmax(mu)
    # 探索：选择方差最大的点
    explore_idx = np.argmax(sigma)
    # EI：平衡选择
    ei = ExpectedImprovement(xi=0.1)
    ei_values = ei(mu, sigma, f_best)
    ei_idx = np.argmax(ei_values)
    
    ax4.plot(x, mu, 'b-', linewidth=2, label='μ(x)')
    ax4.fill_between(x, mu - sigma, mu + sigma, alpha=0.2, color='gray')
    ax4.axvline(x=x[exploit_idx], color='green', linestyle='--', linewidth=2,
                label=f'Exploitation: x={x[exploit_idx]:.1f}')
    ax4.axvline(x=x[explore_idx], color='orange', linestyle='--', linewidth=2,
                label=f'Exploration: x={x[explore_idx]:.1f}')
    ax4.axvline(x=x[ei_idx], color='red', linestyle='-', linewidth=2,
                label=f'EI Balance: x={x[ei_idx]:.1f}')
    ax4.set_title('Exploration vs Exploitation Trade-off', fontsize=12)
    ax4.set_xlabel('x')
    ax4.set_ylabel('f(x)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('acquisition_tradeoff.png', dpi=150, bbox_inches='tight')
    print("\n探索-利用权衡图已保存到: acquisition_tradeoff.png")
    plt.show()


if __name__ == "__main__":
    plot_acquisition_functions()
    demonstrate_exploration_exploitation()
```

#### 采集函数选择指南

| 采集函数 | 优点 | 缺点 | 适用场景 |
|---------|------|------|---------|
| **PI** | 简单直观 | 容易陷入局部最优，不考虑改进幅度 | 低维问题，快速原型 |
| **EI** | 考虑改进幅度，平衡性好 | 需要计算Φ和φ | 大多数场景，推荐默认使用 |
| **UCB** | 计算最简单，有理论保证 | 需要调节κ参数 | 需要理论保证的在线学习 |

**实际建议**：
1. **默认使用EI**：在实践中，EI通常表现最好，是最安全的选择
2. **噪声较大时用UCB**：如果评估噪声很大，UCB比EI更鲁棒
3. **预算有限时用PI**：如果评估次数非常有限，PI的简单性可能更有优势

---

### 57.3.6 完整贝叶斯优化器实现

现在我们将高斯过程和采集函数整合起来，实现一个完整的贝叶斯优化器。

**费曼法比喻**：贝叶斯优化就像一位经验丰富的侦探破案：
1. **收集线索**（观测数据）
2. **绘制嫌疑人画像**（GP建模损失函数）
3. **决定下一步去哪里调查**（采集函数选择下一个点）
4. **重复直到破案**（找到最优超参数）

```python
"""
57.3.6 完整贝叶斯优化器实现
整合GP、采集函数，应用于超参数调优
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# 复用之前实现的GP和采集函数
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import differential_evolution


class RBFKernel:
    """RBF核函数（复用）"""
    def __init__(self, length_scale=1.0, sigma_f=1.0):
        self.length_scale = length_scale
        self.sigma_f = sigma_f
    
    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        sq_dists = (
            np.sum(X1**2, axis=1).reshape(-1, 1) + 
            np.sum(X2**2, axis=1) - 
            2 * np.dot(X1, X2.T)
        )
        return self.sigma_f**2 * np.exp(-0.5 * sq_dists / self.length_scale**2)


class GaussianProcess:
    """简化版GP（复用核心功能）"""
    def __init__(self, noise=1e-5):
        self.kernel = RBFKernel()
        self.noise = noise
        self.X = None
        self.y = None
        self.L = None
        self.alpha = None
    
    def fit(self, X, y):
        self.X = np.atleast_2d(X)
        self.y = np.array(y).reshape(-1, 1)
        
        K = self.kernel(self.X, self.X) + self.noise**2 * np.eye(len(self.X))
        self.L = cholesky(K, lower=True)
        v = solve_triangular(self.L, self.y, lower=True)
        self.alpha = solve_triangular(self.L.T, v, lower=False)
        return self
    
    def predict(self, X_test, return_std=True):
        X_test = np.atleast_2d(X_test)
        k_star = self.kernel(X_test, self.X)
        y_mean = np.dot(k_star, self.alpha).ravel()
        
        if not return_std:
            return y_mean
        
        v = solve_triangular(self.L, k_star.T, lower=True)
        k_star_star = np.diag(self.kernel(X_test, X_test))
        y_var = k_star_star - np.sum(v**2, axis=0)
        y_var = np.maximum(y_var, 1e-10)
        return y_mean, np.sqrt(y_var)


class BayesianOptimizer:
    """
    贝叶斯优化器
    
    用于黑盒函数优化的通用框架
    """
    
    def __init__(self, bounds, acquisition='ei', xi=0.01, kappa=2.0, 
                 random_state=None, verbose=True):
        """
        Args:
            bounds: 搜索空间边界，列表形式 [(min1, max1), (min2, max2), ...]
            acquisition: 采集函数类型 ('pi', 'ei', 'ucb')
            xi: PI/EI的探索参数
            kappa: UCB的探索参数
            random_state: 随机种子
            verbose: 是否打印进度
        """
        self.bounds = np.array(bounds)
        self.acquisition_type = acquisition
        self.xi = xi
        self.kappa = kappa
        self.verbose = verbose
        
        if random_state:
            np.random.seed(random_state)
        
        # 内部状态
        self.gp = None
        self.X_observed = []
        self.y_observed = []
        self.best_y = float('-inf')
        self.best_x = None
        
        # 记录优化历史
        self.history = {
            'X': [],
            'y': [],
            'best_y': [],
            'acquisition_max': []
        }
    
    def _acquisition_function(self, X):
        """
        计算采集函数值（用于优化）
        
        注意：这里返回负值，因为我们要用minimize找到最大值
        """
        X = np.atleast_2d(X)
        mu, sigma = self.gp.predict(X, return_std=True)
        
        if self.acquisition_type == 'pi':
            # Probability of Improvement
            sigma = np.maximum(sigma, 1e-9)
            Z = (mu - self.best_y - self.xi) / sigma
            return -norm.cdf(Z)
        
        elif self.acquisition_type == 'ei':
            # Expected Improvement
            sigma = np.maximum(sigma, 1e-9)
            Z = (mu - self.best_y - self.xi) / sigma
            ei = (mu - self.best_y - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            return -ei
        
        elif self.acquisition_type == 'ucb':
            # Upper Confidence Bound
            return -(mu + self.kappa * sigma)
        
        else:
            raise ValueError(f"Unknown acquisition: {self.acquisition_type}")
    
    def _propose_next_point(self):
        """找到下一个采样点（最大化采集函数）"""
        
        # 使用差分进化全局优化寻找采集函数的最大值
        # 这是一个鲁棒的启发式优化方法
        result = differential_evolution(
            self._acquisition_function,
            self.bounds,
            maxiter=100,
            polish=True,
            seed=np.random.randint(10000)
        )
        
        return result.x
    
    def maximize(self, objective, n_iterations=10, n_initial_points=5):
        """
        最大化损失函数
        
        Args:
            objective: 损失函数，接受数组返回标量
            n_iterations: 优化迭代次数
            n_initial_points: 初始随机采样点数
            
        Returns:
            best_x: 最优输入
            best_y: 最优值
        """
        
        print(f"开始贝叶斯优化 ({self.acquisition_type.upper()}采集函数)")
        print(f"搜索空间维度: {len(self.bounds)}")
        print(f"初始随机点: {n_initial_points}, 优化迭代: {n_iterations}")
        print("=" * 60)
        
        # 1. 初始随机采样（拉丁超立方采样会更优，这里用均匀随机）
        if self.verbose:
            print("\n[阶段1] 初始随机采样...")
        
        for i in range(n_initial_points):
            x_random = np.array([
                np.random.uniform(low, high) for low, high in self.bounds
            ])
            y_random = objective(x_random)
            
            self.X_observed.append(x_random)
            self.y_observed.append(y_random)
            
            if y_random > self.best_y:
                self.best_y = y_random
                self.best_x = x_random.copy()
            
            if self.verbose:
                print(f"  随机点 {i+1}: y={y_random:.4f}, 当前最优={self.best_y:.4f}")
        
        # 2. 贝叶斯优化循环
        if self.verbose:
            print(f"\n[阶段2] 贝叶斯优化迭代...")
        
        for i in range(n_iterations):
            # 拟合GP模型
            X_array = np.array(self.X_observed)
            y_array = np.array(self.y_observed)
            
            # 标准化y值（有助于GP拟合）
            self.y_mean = np.mean(y_array)
            self.y_std = np.std(y_array) if np.std(y_array) > 0 else 1
            y_normalized = (y_array - self.y_mean) / self.y_std
            
            self.gp = GaussianProcess()
            self.gp.fit(X_array, y_normalized)
            
            # 找到下一个采样点
            next_x = self._propose_next_point()
            next_y = objective(next_x)
            
            # 更新观测
            self.X_observed.append(next_x)
            self.y_observed.append(next_y)
            
            # 更新最优
            if next_y > self.best_y:
                self.best_y = next_y
                self.best_x = next_x.copy()
                improved = "*** 改进! ***"
            else:
                improved = ""
            
            # 记录历史
            self.history['X'].append(next_x)
            self.history['y'].append(next_y)
            self.history['best_y'].append(self.best_y)
            
            if self.verbose:
                print(f"  迭代 {i+1}/{n_iterations}: y={next_y:.4f}, "
                      f"最优={self.best_y:.4f} {improved}")
        
        print("=" * 60)
        print(f"优化完成!")
        print(f"最优值: {self.best_y:.4f}")
        print(f"最优参数: {self.best_x}")
        
        return self.best_x, self.best_y


# ========================================
# 应用于超参数调优示例
# ========================================

def optimize_svm_hyperparameters():
    """
    使用贝叶斯优化调优SVM超参数
    优化目标：C (正则化参数) 和 gamma (核系数)
    """
    
    print("\n示例1: SVM超参数优化")
    print("优化参数: C (正则化强度) 和 gamma (RBF核系数)")
    
    # 加载数据集
    X, y = load_digits(n_class=10, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    def svm_objective(params):
        """
        损失函数：SVM在交叉验证上的准确率
        params[0]: C (对数尺度，实际值 10^params[0])
        params[1]: gamma (对数尺度，实际值 10^params[1])
        """
        C = 10 ** params[0]  # 从对数空间转换
        gamma = 10 ** params[1]
        
        try:
            model = SVC(C=C, gamma=gamma, random_state=42)
            # 使用3折交叉验证，取平均准确率
            scores = cross_val_score(model, X_train, y_train, cv=3, 
                                     scoring='accuracy', n_jobs=-1)
            return scores.mean()
        except Exception as e:
            return 0.0  # 出错时返回差值
    
    # 定义搜索空间（对数尺度）
    # C的范围: 10^-3 到 10^3
    # gamma的范围: 10^-4 到 10^1
    bounds = [(-3, 3), (-4, 1)]
    
    # 创建优化器并使用EI采集函数
    optimizer = BayesianOptimizer(
        bounds=bounds, 
        acquisition='ei',
        xi=0.01,
        random_state=42
    )
    
    best_params_log, best_score = optimizer.maximize(
        svm_objective,
        n_iterations=15,
        n_initial_points=5
    )
    
    # 转换回实际值
    best_C = 10 ** best_params_log[0]
    best_gamma = 10 ** best_params_log[1]
    
    print(f"\n最优超参数:")
    print(f"  C = {best_C:.4f}")
    print(f"  gamma = {best_gamma:.6f}")
    
    # 在测试集上验证
    final_model = SVC(C=best_C, gamma=best_gamma, random_state=42)
    final_model.fit(X_train, y_train)
    test_accuracy = final_model.score(X_test, y_test)
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    return optimizer, (best_C, best_gamma, test_accuracy)


def optimize_random_forest():
    """
    使用贝叶斯优化调优随机森林超参数
    优化目标：n_estimators 和 max_depth
    """
    
    print("\n示例2: 随机森林超参数优化")
    print("优化参数: n_estimators (树的数量) 和 max_depth (最大深度)")
    
    # 创建合成数据集
    X, y = make_classification(n_samples=1000, n_features=20, 
                               n_informative=10, n_redundant=5,
                               random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    def rf_objective(params):
        """
        损失函数
        params[0]: n_estimators (50-300)
        params[1]: max_depth (2-50)
        """
        n_estimators = int(params[0])
        max_depth = int(params[1])
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        scores = cross_val_score(model, X_train, y_train, cv=3,
                                 scoring='accuracy', n_jobs=-1)
        return scores.mean()
    
    # 搜索空间
    bounds = [(50, 300), (2, 50)]
    
    optimizer = BayesianOptimizer(
        bounds=bounds,
        acquisition='ei',
        random_state=42
    )
    
    best_params, best_score = optimizer.maximize(
        rf_objective,
        n_iterations=12,
        n_initial_points=5
    )
    
    best_n_estimators = int(best_params[0])
    best_max_depth = int(best_params[1])
    
    print(f"\n最优超参数:")
    print(f"  n_estimators = {best_n_estimators}")
    print(f"  max_depth = {best_max_depth}")
    
    # 测试集验证
    final_model = RandomForestClassifier(
        n_estimators=best_n_estimators,
        max_depth=best_max_depth,
        random_state=42
    )
    final_model.fit(X_train, y_train)
    test_accuracy = final_model.score(X_test, y_test)
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    return optimizer


def visualize_optimization_process():
    """可视化贝叶斯优化过程"""
    
    # 定义一个一维测试函数
    def test_func(x):
        x = x[0] if hasattr(x, '__len__') else x
        return np.sin(3*x) * x**2 * np.exp(-x) + np.random.normal(0, 0.01)
    
    bounds = [(0, 5)]
    
    optimizer = BayesianOptimizer(bounds=bounds, acquisition='ei', verbose=False)
    optimizer.maximize(test_func, n_iterations=10, n_initial_points=3)
    
    # 绘制优化过程
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 收敛曲线
    ax1 = axes[0]
    iterations = range(1, len(optimizer.y_observed) + 1)
    ax1.plot(iterations, optimizer.y_observed, 'bo-', alpha=0.6, label='Observed Value')
    
    best_so_far = np.maximum.accumulate(optimizer.y_observed)
    ax1.plot(iterations, best_so_far, 'r-', linewidth=2, label='Best So Far')
    
    ax1.axvline(x=3.5, color='gray', linestyle='--', alpha=0.5, label='End of Random')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Bayesian Optimization Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 采样点分布
    ax2 = axes[1]
    X_obs = np.array(optimizer.X_observed).ravel()
    y_obs = np.array(optimizer.y_observed)
    
    colors = ['blue'] * 3 + ['red'] * 10  # 前3个是随机，后10个是BO
    labels_rand = ['Random Sample'] * 3 + [''] * 10
    labels_bo = [''] * 3 + ['BO Sample'] * 10
    
    for i, (x, y, c) in enumerate(zip(X_obs, y_obs, colors)):
        label_r = labels_rand[i] if labels_rand[i] else None
        label_b = labels_bo[i] if labels_bo[i] and not labels_rand[i] else None
        ax2.scatter(x, y, c=c, s=100, alpha=0.7, 
                   label=label_r or label_b, edgecolors='black')
    
    # 绘制真实函数
    x_fine = np.linspace(0, 5, 200)
    y_true = [test_func([xi]) for xi in x_fine]
    ax2.plot(x_fine, y_true, 'g--', alpha=0.5, label='True Function')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.set_title('Sample Points Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bayesian_optimization_process.png', dpi=150, bbox_inches='tight')
    print("\n贝叶斯优化过程可视化已保存到: bayesian_optimization_process.png")
    plt.show()


if __name__ == "__main__":
    # 运行示例
    visualize_optimization_process()
    optimizer1, result1 = optimize_svm_hyperparameters()
    optimizer2 = optimize_random_forest()
```

#### 贝叶斯优化总结

**核心优势**：
1. **样本效率高**：通常10-50次评估即可找到不错的解
2. **处理噪声**：GP天然建模噪声，适合随机训练过程
3. **自带不确定性估计**：指导探索策略
4. **可并行**：可以同时评估多个候选点

**适用场景**：
- 评估成本高昂（训练深度学习模型）
- 损失函数黑盒（无法求导）
- 带噪声的评估
- 搜索空间连续或混合

**局限性**：
- 计算成本随样本数增加（$O(n^3)$）
- 高维问题表现下降（维度>20时困难）
- 需要仔细设计搜索空间和核函数

---

## 57.4 多保真度优化：用"快速品尝"预测"完整烹饪"

在超参数调优中，一个巨大的问题是：**完整评估一个配置可能非常昂贵**。例如，训练一个大型神经网络可能需要数小时甚至数天。

**多保真度优化**（Multi-fidelity Optimization）的核心思想是：用便宜的"低保真度"近似来指导搜索，只在最有希望的配置上花费昂贵的"高保真度"评估。

**费曼法比喻**：想象你要在100家餐厅中找出最好吃的。与其在每家都点满汉全席（昂贵），不如：
1. 先在每家点一道招牌小菜（便宜）品尝
2. 根据小菜的口味，淘汰明显不行的
3. 只在最有希望的几家点完整大餐

### 57.4.1 Successive Halving：不断缩小的锦标赛

**逐次折半**（Successive Halving, SH）是最基础的多保真度算法。它像一个锦标赛：
1. 所有选手（配置）进行预赛（低保真度训练）
2. 淘汰一半表现差的
3. 剩下的选手进行复赛（更高保真度训练）
4. 重复直到决出冠军

#### 算法描述

输入：
- $N$：初始配置数量
- $\eta$：淘汰比例（通常为3或4）
- $R$：最大资源预算（如完整训练的迭代次数）

算法步骤：
1. 随机采样$N$个配置
2. 为每个配置分配初始资源$r_0$
3. 对于每一轮$s = 0, 1, ..., S$：
   - 在当前资源$r_s$下评估所有存活配置
   - 保留表现最好的$N_s / \eta$个配置
   - 将资源增加到$r_{s+1} = r_s \times \eta$

#### 代码实现

```python
"""
57.4.1 Successive Halving算法实现
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Dict, Any


class SuccessiveHalving:
    """
    逐次折半算法 (Successive Halving)
    
    通过逐步淘汰表现差的配置来高效搜索超参数空间
    """
    
    def __init__(self, eta=3, random_state=None):
        """
        Args:
            eta: 淘汰比例，每轮保留1/eta的配置
            random_state: 随机种子
        """
        self.eta = eta
        if random_state:
            np.random.seed(random_state)
    
    def maximize(self, configs: List[Dict], train_fn: Callable, 
                 max_resource: float, min_resource: float = 1.0):
        """
        执行Successive Halving
        
        Args:
            configs: 初始配置列表
            train_fn: 训练函数，接收(config, resource)返回性能分数
            max_resource: 最大资源预算（如完整训练的epoch数）
            min_resource: 初始资源预算
            
        Returns:
            best_config: 最优配置
            history: 训练历史记录
        """
        
        n_configs = len(configs)
        
        # 计算需要多少轮才能用max_resource训练一个配置
        # min_resource * eta^s = max_resource
        max_sh_iter = int(np.log(max_resource / min_resource) / np.log(self.eta))
        
        # 根据预算约束，调整初始配置数
        # 总共使用的资源 = sum_{s=0}^{S} N_s * r_s
        # 其中 N_s = n_configs / eta^s, r_s = min_resource * eta^s
        # 所以每轮资源消耗相同！
        
        print(f"Successive Halving 配置:")
        print(f"  初始配置数: {n_configs}")
        print(f"  淘汰比例 η: {self.eta}")
        print(f"  资源范围: {min_resource} → {max_resource}")
        print(f"  迭代轮数: {max_sh_iter + 1}")
        print("=" * 60)
        
        survivors = list(range(n_configs))  # 存活配置的索引
        resource = min_resource
        history = []
        
        for iteration in range(max_sh_iter + 1):
            n_survivors = len(survivors)
            
            print(f"\n[第{iteration+1}轮] {n_survivors}个配置，每个使用{resource:.1f}资源")
            
            # 评估所有存活配置
            scores = []
            for idx in survivors:
                config = configs[idx]
                score = train_fn(config, resource)
                scores.append(score)
                
                history.append({
                    'config_id': idx,
                    'config': config,
                    'resource': resource,
                    'score': score,
                    'iteration': iteration
                })
                
                print(f"  配置{idx}: 分数={score:.4f}")
            
            # 如果不是最后一轮，进行淘汰
            if iteration < max_sh_iter:
                # 保留表现最好的 n_survivors // eta 个
                n_keep = max(1, n_survivors // self.eta)
                
                # 按分数排序，保留最好的
                sorted_indices = np.argsort(scores)[::-1]  # 降序
                keep_indices = sorted_indices[:n_keep]
                survivors = [survivors[i] for i in keep_indices]
                
                print(f"  → 保留表现最好的{n_keep}个配置")
                
                # 增加资源
                resource *= self.eta
            else:
                # 最后一轮，找出最佳配置
                best_local_idx = np.argmax(scores)
                best_config_idx = survivors[best_local_idx]
                best_score = scores[best_local_idx]
                
                print(f"\n最优配置: 配置{best_config_idx}, 分数={best_score:.4f}")
        
        best_config = configs[best_config_idx]
        return best_config, history


# ========================================
# 演示：SH在合成问题上的表现
# ========================================

def demo_successive_halving():
    """演示SH算法的执行过程"""
    
    # 定义一些"假"的配置
    # 真实性能随着训练而提升，但不同配置的提升速度不同
    np.random.seed(42)
    n_configs = 27  # 能被3整除多次
    
    # 为每个配置生成一个"真实"的最终性能
    true_performances = np.random.beta(2, 5, n_configs)  # 多数配置表现一般
    true_performances[5] = 0.95  # 有一个宝藏配置！
    true_performances[12] = 0.88  # 还有一个不错的
    
    # 学习曲线模拟：早期可能无法区分好坏
    def simulate_training(config_id, epochs):
        """
        模拟训练过程
        好的配置早期可能表现一般，但随着训练时间增加，优势显现
        """
        true_perf = true_performances[config_id]
        
        # 学习曲线模型: performance = true_perf * (1 - exp(-k * epochs))
        # 不同配置的k不同（有的学得快，有的学得慢）
        k = 0.1 + 0.05 * np.random.rand()  # 随机学习速度
        
        perf = true_perf * (1 - np.exp(-k * epochs))
        noise = np.random.normal(0, 0.02)  # 观测噪声
        
        return perf + noise
    
    # 将配置表示为字典
    configs = [{'id': i, 'hidden_size': np.random.choice([64, 128, 256])}
               for i in range(n_configs)]
    
    # 创建SH优化器
    sh = SuccessiveHalving(eta=3, random_state=42)
    
    # 训练函数
    def train_fn(config, resource):
        return simulate_training(config['id'], resource)
    
    # 执行SH
    best_config, history = sh.maximize(
        configs=configs,
        train_fn=train_fn,
        max_resource=81,   # 最大epoch数
        min_resource=1     # 初始epoch数
    )
    
    print(f"\n找到的最优配置ID: {best_config['id']}")
    print(f"该配置的真实性能排名: {np.argsort(true_performances)[::-1].tolist().index(best_config['id']) + 1}/{n_configs}")
    
    # 可视化
    visualize_sh_history(history, true_performances)
    
    return best_config, history


def visualize_sh_history(history, true_performances):
    """可视化SH的训练历史"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 学习曲线图
    ax1 = axes[0]
    
    # 按配置ID分组
    config_data = {}
    for record in history:
        cid = record['config_id']
        if cid not in config_data:
            config_data[cid] = {'resources': [], 'scores': []}
        config_data[cid]['resources'].append(record['resource'])
        config_data[cid]['scores'].append(record['score'])
    
    # 绘制每个配置的学习曲线
    for cid, data in config_data.items():
        true_perf = true_performances[cid]
        # 根据真实性能着色
        if true_perf > 0.9:
            color = 'green'
            linewidth = 2.5
            alpha = 0.9
        elif true_perf > 0.7:
            color = 'blue'
            linewidth = 1.5
            alpha = 0.6
        else:
            color = 'gray'
            linewidth = 1
            alpha = 0.3
        
        ax1.plot(data['resources'], data['scores'], 
                color=color, linewidth=linewidth, alpha=alpha,
                marker='o', markersize=4)
    
    ax1.set_xlabel('Resource (Epochs)', fontsize=11)
    ax1.set_ylabel('Performance Score', fontsize=11)
    ax1.set_title('Successive Halving: Learning Curves', fontsize=12)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=2.5, label='Top tier (true>0.9)'),
        Line2D([0], [0], color='blue', linewidth=1.5, label='Mid tier (true>0.7)'),
        Line2D([0], [0], color='gray', linewidth=1, label='Low tier')
    ]
    ax1.legend(handles=legend_elements)
    
    # 2. 资源分配热图
    ax2 = axes[1]
    
    iterations = sorted(set(r['iteration'] for r in history))
    n_iters = len(iterations)
    resource_matrix = np.zeros((n_iters, len(true_performances)))
    
    for record in history:
        i = record['iteration']
        cid = record['config_id']
        resource_matrix[i, cid] = record['resource']
    
    im = ax2.imshow(resource_matrix, aspect='auto', cmap='YlOrRd')
    ax2.set_xlabel('Configuration ID', fontsize=11)
    ax2.set_ylabel('Iteration', fontsize=11)
    ax2.set_title('Resource Allocation per Config per Iteration', fontsize=12)
    plt.colorbar(im, ax=ax2, label='Resource')
    
    plt.tight_layout()
    plt.savefig('successive_halving_demo.png', dpi=150, bbox_inches='tight')
    print("\nSH可视化结果已保存到: successive_halving_demo.png")
    plt.show()


if __name__ == "__main__":
    demo_successive_halving()
```

### 57.4.2 HyperBand：最优的预算分配

Successive Halving有一个关键问题：**如何选择初始配置数$N$？**
- $N$太大：每个配置只能获得很少的资源，可能无法区分好坏
- $N$太小：可能错过好的配置

**HyperBand**解决了这个问题：它运行多个不同$N$值的SH，自动找到最优的平衡点。

#### HyperBand的核心思想

HyperBand通过** Successive Halving with Different Config Counts** 来探索$N$和每轮资源的权衡。

它定义两个循环参数：
- $R$：单个配置的最大资源
- $\eta$：淘汰比例

然后运行不同$N$值的SH：
- $s = s_{max}, s_{max}-1, ..., 0$
- 每轮SH的初始配置数：$N = \lceil \frac{R}{r} \cdot \frac{\eta^s}{s+1} \rceil$
- 初始资源：$r = R \cdot \eta^{-s}$

```python
"""
57.4.2 HyperBand算法实现
"""

import numpy as np
import math


class HyperBand:
    """
    HyperBand算法
    
    通过运行多个不同规模的Successive Halving，
    自动平衡"探索更多配置"和"给每个配置更多资源"之间的权衡
    """
    
    def __init__(self, max_resource, eta=3, random_state=None):
        """
        Args:
            max_resource: 单个配置的最大资源（如最大epoch数）
            eta: 淘汰比例
            random_state: 随机种子
        """
        self.max_resource = max_resource
        self.eta = eta
        
        if random_state:
            np.random.seed(random_state)
        
        # 计算s_max
        self.s_max = int(math.log(max_resource, eta))
        print(f"HyperBand配置:")
        print(f"  最大资源 R = {max_resource}")
        print(f"  淘汰比例 η = {eta}")
        print(f"  最大迭代 s_max = {self.s_max}")
    
    def run(self, get_config_fn: Callable, train_fn: Callable, 
            total_budget: float = None):
        """
        执行HyperBand
        
        Args:
            get_config_fn: 生成新配置的函数
            train_fn: 训练函数，接收(config, resource)返回性能
            total_budget: 总预算（可选，用于控制总计算量）
            
        Returns:
            best_config: 找到的最优配置
            best_score: 最优分数
            all_history: 所有SH运行的历史记录
        """
        
        best_overall_config = None
        best_overall_score = float('-inf')
        all_history = []
        
        # 从最大的s开始（更激进的资源分配）
        for s in range(self.s_max, -1, -1):
            # 计算这一轮的参数
            # n: 初始配置数
            # r: 初始资源
            n = int(math.ceil(
                (self.s_max + 1) / (s + 1) * self.eta ** s
            ))
            r = self.max_resource * self.eta ** (-s)
            
            print(f"\n{'='*60}")
            print(f"[HyperBand bracket s={s}]")
            print(f"  初始配置数 n = {n}")
            print(f"  初始资源 r = {r:.2f}")
            
            # 生成n个随机配置
            configs = [get_config_fn() for _ in range(n)]
            
            # 运行Successive Halving
            sh = SuccessiveHalving(eta=self.eta)
            best_config, history = sh.maximize(
                configs=configs,
                train_fn=train_fn,
                max_resource=self.max_resource,
                min_resource=r
            )
            
            all_history.extend(history)
            
            # 更新全局最优
            bracket_best_score = max(r['score'] for r in history 
                                     if r['resource'] == self.max_resource)
            if bracket_best_score > best_overall_score:
                best_overall_score = bracket_best_score
                best_overall_config = best_config
                print(f"  → 新的全局最优! 分数={bracket_best_score:.4f}")
        
        print(f"\n{'='*60}")
        print(f"HyperBand完成! 最优分数: {best_overall_score:.4f}")
        
        return best_overall_config, best_overall_score, all_history


# ========================================
# 比较：Random Search vs Successive Halving vs HyperBand
# ========================================

def compare_methods():
    """比较三种方法的效率"""
    
    np.random.seed(42)
    
    # 问题设置
    n_configs_total = 100
    max_epochs = 81
    
    # 生成配置和真实性能
    true_perfs = np.random.beta(2, 5, n_configs_total)
    true_perfs[23] = 0.96  # 最佳配置
    
    def get_config():
        return {'id': np.random.randint(n_configs_total)}
    
    def train(config, resource):
        tid = config['id']
        true_perf = true_perfs[tid]
        k = 0.1 + 0.03 * (tid % 5)  # 不同学习速度
        perf = true_perf * (1 - np.exp(-k * resource / 10))
        return perf + np.random.normal(0, 0.01)
    
    print("="*70)
    print("方法对比实验")
    print("="*70)
    
    # 1. 纯随机搜索（所有配置都用满资源）
    print("\n[方法1] 纯随机搜索 (每个配置用满资源)")
    n_random = 10  # 只能负担10个完整训练
    random_scores = []
    for i in range(n_random):
        config = get_config()
        score = train(config, max_epochs)
        random_scores.append((config['id'], score))
    best_random = max(random_scores, key=lambda x: x[1])
    print(f"  评估配置数: {n_random}")
    print(f"  最优配置ID: {best_random[0]}, 分数: {best_random[1]:.4f}")
    
    # 2. 单次SH
    print("\n[方法2] 单次Successive Halving")
    configs = [get_config() for _ in range(27)]
    sh = SuccessiveHalving(eta=3)
    best_sh, _ = sh.maximize(configs, train, max_epochs, min_resource=1)
    print(f"  最优配置ID: {best_sh['id']}, 真实排名: {np.argsort(true_perfs)[::-1].tolist().index(best_sh['id'])+1}")
    
    # 3. HyperBand
    print("\n[方法3] HyperBand")
    hb = HyperBand(max_resource=max_epochs, eta=3)
    best_hb, score_hb, _ = hb.run(get_config, train)
    print(f"  最优配置ID: {best_hb['id']}, 真实排名: {np.argsort(true_perfs)[::-1].tolist().index(best_hb['id'])+1}")
    
    print("\n" + "="*70)
    print("总结：HyperBand通过智能分配资源，在相同预算下找到更好的配置")


if __name__ == "__main__":
    from successive_halving import SuccessiveHalving  # 复用前面的代码
    from typing import Callable
    
    compare_methods()
```

### 57.4.3 BOHB：贝叶斯优化 + HyperBand

HyperBand虽然高效，但它的配置采样是**完全随机**的。如果我们能用贝叶斯优化来指导配置采样，效果会更好。

**BOHB**（Bayesian Optimization and HyperBand）正是这样做的：
- 使用**HyperBand**的预算分配策略
- 使用**贝叶斯优化**（TPE算法）替代随机采样

```python
"""
57.4.3 BOHB核心思想演示
"""

import numpy as np
from scipy.stats import norm
from collections import defaultdict


class TreeParzenEstimator:
    """
    树形Parzen估计器 (TPE)
    
    TPE是BOHB中使用的贝叶斯优化方法，相比GP更适合混合类型参数
    
    核心思想：
    - 不使用 p(y|x)，而是直接建模 p(x|y)
    - 将观测分为"好的"(y < y*)和"坏的"(y >= y*)
    - 使用核密度估计(KDE)建模 l(x) = p(x|好) 和 g(x) = p(x|坏)
    - 选择使 l(x)/g(x) 最大的x
    """
    
    def __init__(self, gamma=0.15):
        """
        Args:
            gamma: 用于区分"好"和"坏"的分位数
        """
        self.gamma = gamma
        self.observations = []  # (config, loss) 列表
    
    def observe(self, config, loss):
        """记录一次观测"""
        self.observations.append((config, loss))
    
    def suggest(self, n_samples=100):
        """
        建议下一个配置
        
        返回使 EI 近似最大的配置
        """
        if len(self.observations) < 10:
            # 数据不足时随机采样
            return None
        
        # 按损失排序
        sorted_obs = sorted(self.observations, key=lambda x: x[1])
        
        # 分割点
        n_good = max(1, int(self.gamma * len(sorted_obs)))
        
        good_configs = [obs[0] for obs in sorted_obs[:n_good]]
        bad_configs = [obs[0] for obs in sorted_obs[n_good:]]
        
        # 对于每个超参数，计算l(x)/g(x)比率
        # 这里简化处理，实际TPE有更复杂的处理方式
        
        # 随机生成候选并评分
        best_ratio = -1
        best_config = None
        
        for _ in range(n_samples):
            candidate = self._random_config()
            ratio = self._compute_ratio(candidate, good_configs, bad_configs)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_config = candidate
        
        return best_config
    
    def _random_config(self):
        """随机生成配置"""
        return {
            'lr': 10 ** np.random.uniform(-5, -1),
            'batch_size': np.random.choice([16, 32, 64, 128]),
            'dropout': np.random.uniform(0.1, 0.5)
        }
    
    def _compute_ratio(self, candidate, good_configs, bad_configs):
        """计算 l(candidate)/g(candidate) 的近似值"""
        # 简化的基于距离的计算
        def min_distance(configs):
            dists = []
            for c in configs:
                d = (np.log10(candidate['lr']) - np.log10(c['lr']))**2
                d += (candidate['dropout'] - c['dropout'])**2 * 10
                dists.append(d)
            return min(dists) if dists else 1.0
        
        l_prob = np.exp(-min_distance(good_configs))
        g_prob = np.exp(-min_distance(bad_configs)) + 1e-10
        
        return l_prob / g_prob


class BOHB:
    """
    BOHB: 贝叶斯优化 + HyperBand
    
    结合了HyperBand的高效资源分配和贝叶斯优化的智能采样
    """
    
    def __init__(self, max_resource, eta=3):
        self.max_resource = max_resource
        self.eta = eta
        self.tpe = TreeParzenEstimator()
        self.s_max = int(np.log(max_resource) / np.log(eta))
    
    def run(self, train_fn, n_iterations=5):
        """
        执行BOHB
        
        每轮使用TPE生成配置，然后用SH评估
        """
        
        best_config = None
        best_score = float('-inf')
        
        for iteration in range(n_iterations):
            print(f"\n[BOHB Iteration {iteration+1}/{n_iterations}]")
            
            # 前几次迭代随机探索，之后使用TPE
            if iteration < 2:
                get_config = lambda: self.tpe._random_config()
                print("  模式: 随机探索")
            else:
                def get_config():
                    cfg = self.tpe.suggest()
                    return cfg if cfg else self.tpe._random_config()
                print("  模式: TPE指导采样")
            
            # 运行一个HyperBand bracket
            for s in [self.s_max]:  # 可以扩展到多个s值
                n = int((self.s_max + 1) / (s + 1) * self.eta ** s)
                r = self.max_resource * self.eta ** (-s)
                
                # 生成配置
                configs = [get_config() for _ in range(n)]
                
                # 使用SH评估并更新TPE
                sh = SuccessiveHalving(eta=self.eta)
                bracket_best, history = sh.maximize(
                    configs, train_fn, self.max_resource, r
                )
                
                # 更新TPE观测
                for record in history:
                    # 使用最终资源的分数
                    if record['resource'] == self.max_resource:
                        # TPE最小化损失，所以取负值
                        loss = -record['score']
                        self.tpe.observe(record['config'], loss)
                        
                        if record['score'] > best_score:
                            best_score = record['score']
                            best_config = record['config']
        
        return best_config, best_score


if __name__ == "__main__":
    # 演示BOHB流程
    print("BOHB核心思想演示")
    print("="*60)
    
    # 简化演示
    tpe = TreeParzenEstimator()
    
    # 模拟一些观测
    np.random.seed(42)
    for i in range(20):
        config = tpe._random_config()
        # 模拟损失（越小越好）
        loss = -np.log(config['lr']) * 0.1 + config['dropout'] * 2 + np.random.normal(0, 0.1)
        tpe.observe(config, loss)
    
    # 获取TPE建议
    suggestion = tpe.suggest(n_samples=50)
    print(f"\nTPE建议的配置:")
    print(f"  学习率: {suggestion['lr']:.6f}")
    print(f"  Dropout: {suggestion['dropout']:.3f}")
    
    print("\nBOHB结合HyperBand的资源分配和TPE的智能采样，")
    print("是目前超参数调优的最先进方法之一。")
```

### 57.4.4 ASHA：异步逐次折半

传统的SH和HyperBand都是**同步**的：必须等一个bracket中所有配置都完成当前轮次，才能进入下一轮。这会造成资源浪费。

**ASHA**（Asynchronous Successive Halving Algorithm）是**异步**的：
- 一个配置完成后立即决定是否升级
- 不需要等待同轮的其他配置
- 更适合分布式并行环境

```python
"""
57.4.4 ASHA异步算法演示
"""

import numpy as np
from collections import deque


class ASHA:
    """
    ASHA: 异步逐次折半算法
    
    与同步SH不同，ASHA中每个配置独立运行：
    - 当一个配置完成某个资源级别的训练，立即评估
    - 如果表现足够好（排名在前1/η），立即升级到更高资源
    - 不需要等待同级别的其他配置完成
    
    优势：
    - 没有同步等待的开销
    - 更好的分布式并行效率
    - 响应更快，新配置可以立即开始
    """
    
    def __init__(self, min_resource, max_resource, reduction_factor=4, 
                 min_early_stopping_rate=0):
        """
        Args:
            min_resource: 最小资源（如初始epoch数）
            max_resource: 最大资源
            reduction_factor: 淘汰比例
            min_early_stopping_rate: 最早可以停止的轮次
        """
        self.min_resource = min_resource
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor
        self.rung_levels = self._get_rung_levels()
        
        # 存储每个rung的观测
        self.rungs = {r: [] for r in self.rung_levels}
        
        print(f"ASHA配置:")
        print(f"  Rung levels: {self.rung_levels}")
    
    def _get_rung_levels(self):
        """计算各个rung的资源级别"""
        rungs = []
        r = self.min_resource
        while r <= self.max_resource:
            rungs.append(r)
            r *= self.reduction_factor
        return rungs
    
    def should_promote(self, config_id, score, resource):
        """
        判断一个配置是否应该晋升到下一个rung
        
        如果该配置在当前rung的排名在前1/η，则晋升
        """
        # 找到当前所在的rung
        current_rung_idx = self.rung_levels.index(resource)
        
        # 记录观测
        self.rungs[resource].append((config_id, score))
        
        # 检查排名
        sorted_rung = sorted(self.rungs[resource], key=lambda x: x[1], reverse=True)
        rank = [i for i, (cid, _) in enumerate(sorted_rung) if cid == config_id][0]
        
        # 前1/η的配置可以晋升
        promotion_threshold = len(self.rungs[resource]) // self.reduction_factor
        
        should_promote = rank < promotion_threshold
        
        if should_promote and current_rung_idx < len(self.rung_levels) - 1:
            next_resource = self.rung_levels[current_rung_idx + 1]
            return True, next_resource
        
        return False, None
    
    def get_num_to_run(self, resource):
        """
        计算在当前rung应该运行多少配置
        
        ASHA的异步特性允许我们动态决定
        """
        # 简化的策略：每个rung最多运行 reduction_factor^2 个配置
        max_configs = self.reduction_factor ** 2
        current_running = len(self.rungs[resource])
        return max(0, max_configs - current_running)


def demo_asha():
    """演示ASHA的异步特性"""
    
    asha = ASHA(min_resource=1, max_resource=27, reduction_factor=3)
    
    print("\n模拟ASHA执行过程:")
    print("-" * 50)
    
    config_id = 0
    active_configs = deque()  # 正在运行的配置
    
    # 初始启动一些配置
    for _ in range(5):
        active_configs.append({
            'id': config_id,
            'resource': 1,
            'score': None
        })
        config_id += 1
    
    completed = []
    
    while active_configs:
        # 模拟一个配置完成
        current = active_configs.popleft()
        
        # 模拟训练结果
        current['score'] = np.random.beta(2, 5) * (1 - np.exp(-0.1 * current['resource']))
        
        promote, next_r = asha.should_promote(
            current['id'], current['score'], current['resource']
        )
        
        status = "✓ 完成" if not promote else f"↑ 晋升到r={next_r}"
        print(f"配置{current['id']}: r={current['resource']}, "
              f"score={current['score']:.3f} {status}")
        
        if promote:
            active_configs.append({
                'id': current['id'],
                'resource': next_r,
                'score': None
            })
        else:
            completed.append(current)
        
        # 异步启动新配置（如果rung1有空位）
        if len([c for c in active_configs if c['resource'] == 1]) < 3:
            active_configs.append({
                'id': config_id,
                'resource': 1,
                'score': None
            })
            config_id += 1
    
    print("-" * 50)
    print(f"总共评估配置数: {config_id}")
    print(f"完成全部流程的配置: {len(completed)}")


if __name__ == "__main__":
    demo_asha()
```

### 57.4.5 多保真度优化总结

| 方法 | 核心思想 | 优势 | 适用场景 |
|------|---------|------|---------|
| **Successive Halving** | 逐轮淘汰差的配置 | 简单有效 | 资源有限，配置数量适中 |
| **HyperBand** | 多轮SH，不同初始配置数 | 自动平衡N和资源 | 不知道最优N时 |
| **BOHB** | HyperBand + 贝叶斯采样 | 智能采样 + 高效评估 | 昂贵评估，需要高质量配置 |
| **ASHA** | 异步执行SH | 无同步等待，适合分布式 | 大规模并行环境 |

**实际建议**：
- **入门**：从HyperBand开始，简单且有效
- **生产环境**：使用ASHA，配合分布式计算框架
- **追求极致**：BOHB，特别是使用Tune、Optuna等成熟库的实现

---

## 57.5 AutoML系统：让机器学会学习

贝叶斯优化和多保真度优化让我们更高效地调优超参数。但还能更进一步：**让机器自动完成从数据清洗到模型部署的整个流程**。

这就是**AutoML**（Automated Machine Learning）的愿景。

**费曼法比喻**：传统的机器学习就像手工裁缝——师傅需要根据客人的身材量体裁衣，选择合适的布料、裁剪方式、缝制技巧。AutoML则像一台智能裁缝机：你把布料放进去，它自动测量、裁剪、缝制，产出一件合身衣服。

### 57.5.1 Auto-sklearn：基于贝叶斯优化的AutoML

**Auto-sklearn**是第一个在图像分类等标准任务上达到人类专家水平的AutoML系统。它结合了：
1. **元学习**（Meta-Learning）：根据数据集特征，推荐好的起点
2. **贝叶斯优化**：搜索模型和超参数的组合
3. **集成学习**：组合多个模型的预测

#### 架构解析

```python
"""
57.5.1 Auto-sklearn核心概念演示
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def autosklearn_concept():
    """
    Auto-sklearn的工作原理概念演示
    
    Auto-sklearn的搜索空间包括：
    1. 预处理：标准化、PCA、特征选择等
    2. 模型：SVM、RF、GBDT、KNN等
    3. 每个模型的超参数
    
    总搜索空间维度：100+
    """
    
    # 使用Auto-sklearn（如果已安装）
    try:
        import autosklearn.classification
        
        print("Auto-sklearn使用示例:")
        print("-" * 50)
        
        # 加载数据
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 创建Auto-sklearn分类器
        # 这里仅展示API，实际运行可能需要较长时间
        print("""
from autosklearn import AutoSklearnClassifier

# 创建AutoML分类器
automl = AutoSklearnClassifier(
    time_left_for_this_task=300,  # 5分钟
    per_run_time_limit=30,        # 每个模型最多30秒
    ensemble_size=50,             # 集成50个模型
    initial_configurations_via_metalearning=25  # 元学习推荐25个起点
)

# 自动搜索最优模型和超参数
automl.fit(X_train, y_train)

# 预测
predictions = automl.predict(X_test)

# 查看最终集成
print(automl.show_models())
        """)
        
        print("\nAuto-sklearn特点:")
        print("  - 自动数据预处理")
        print("  - 自动模型选择")
        print("  - 自动超参数优化")
        print("  - 自动集成多个模型")
        
    except ImportError:
        print("Auto-sklearn未安装，展示核心概念:")
        
        # 手动模拟Auto-sklearn的搜索空间
        search_space = {
            'preprocessor': ['None', 'StandardScaler', 'MinMaxScaler', 'PCA', 'FastICA'],
            'classifier': ['RandomForest', 'SVM', 'GradientBoosting', 'KNN', 'MLP'],
            'hyperparameters': {
                'RandomForest': {
                    'n_estimators': (10, 500),
                    'max_depth': (2, 50),
                    'min_samples_split': (2, 20)
                },
                'SVM': {
                    'C': (1e-5, 1e5, 'log'),
                    'gamma': (1e-5, 1e5, 'log'),
                    'kernel': ['rbf', 'poly', 'sigmoid']
                },
                # ... 其他模型
            }
        }
        
        print(f"搜索空间大小估计: >10^15 种组合")
        print("Auto-sklearn使用贝叶斯优化+元学习在这个巨大空间中高效搜索")


if __name__ == "__main__":
    autosklearn_concept()
```

### 57.5.2 TPOT：基于遗传编程的AutoML

**TPOT**（Tree-based Pipeline Optimization Tool）使用**遗传编程**（Genetic Programming）来搜索最优的数据处理管道。

**核心思想**：
- 每个"个体"是一个完整的机器学习管道（预处理 + 模型）
- 使用交叉、变异等遗传操作进化管道
- 选择表现好的个体进行繁殖

```python
"""
57.5.2 TPOT遗传编程AutoML概念
"""

import numpy as np


def tpot_concept():
    """
    TPOT的核心概念演示
    
    TPOT使用遗传编程进化机器学习管道
    """
    
    print("TPOT工作原理:")
    print("=" * 60)
    
    print("""
# TPOT使用示例
from tpot import TPOTClassifier

tpot = TPOTClassifier(
    generations=5,        # 进化5代
    population_size=20,   # 每代20个个体
    offspring_size=20,    # 产生20个后代
    mutation_rate=0.9,    # 变异率
    crossover_rate=0.1,   # 交叉率
    scoring='accuracy',   # 评估指标
    cv=5,                 # 5折交叉验证
    verbosity=2
)

# 自动进化最优管道
tpot.fit(X_train, y_train)

# 导出最优管道代码
tpot.export('best_pipeline.py')
""")
    
    print("\n遗传编程操作:")
    print("  1. 交叉(Crossover): 两个父代管道交换子树")
    print("     父代1: PCA → RandomForest")
    print("     父代2: StandardScaler → SVM")
    print("     子代:  PCA → SVM")
    
    print("\n  2. 变异(Mutation): 随机修改管道")
    print("     原管道: PCA → RandomForest")
    print("     变异后: SelectKBest → RandomForest")
    
    print("\nTPOT优势:")
    print("  - 可以探索非常复杂的管道组合")
    print("  - 最终输出可读的Python代码")
    print("  - 不限于固定结构，可以发现创新组合")
    
    print("\nTPOT局限:")
    print("  - 计算成本高（需要评估大量个体）")
    print("  - 没有贝叶斯优化的样本效率高")
    print("  - 可能过拟合验证集")


# 模拟一个简单的遗传进化过程
def simulate_evolution():
    """模拟TPOT的进化过程"""
    
    print("\n模拟进化过程:")
    print("-" * 50)
    
    np.random.seed(42)
    
    # 初始种群
    population = [
        {'pipeline': 'Scaler → RF', 'fitness': 0.75},
        {'pipeline': 'PCA → SVM', 'fitness': 0.78},
        {'pipeline': 'None → KNN', 'fitness': 0.72},
        {'pipeline': 'Scaler → GB', 'fitness': 0.80},
    ]
    
    for gen in range(3):
        print(f"\n第{gen+1}代:")
        
        # 按适应度排序
        population.sort(key=lambda x: x['fitness'], reverse=True)
        
        for i, ind in enumerate(population):
            print(f"  排名{i+1}: {ind['pipeline']}, 适应度={ind['fitness']:.3f}")
        
        # 选择最优的繁殖
        survivors = population[:2]
        
        # 产生后代（简化模拟）
        offspring = []
        for parent in survivors:
            # 变异
            new_pipeline = parent['pipeline'].replace('RF', 'RF+SVM')
            new_fitness = min(0.95, parent['fitness'] + np.random.uniform(-0.05, 0.08))
            offspring.append({'pipeline': new_pipeline, 'fitness': new_fitness})
        
        # 新一代
        population = survivors + offspring
    
    print(f"\n最终最优: {population[0]['pipeline']}, 适应度={population[0]['fitness']:.3f}")


if __name__ == "__main__":
    tpot_concept()
    simulate_evolution()
```

### 57.5.3 Auto-Keras：神经网络架构搜索

**Auto-Keras**专注于**神经网络架构搜索**（Neural Architecture Search, NAS），自动设计深度学习模型的结构。

**核心概念**：
- **搜索空间**：定义可能的层类型、连接方式
- **搜索策略**：贝叶斯优化、强化学习、进化算法
- **性能估计**：权重共享、代理模型

```python
"""
57.5.3 Auto-Keras神经网络架构搜索
"""

import numpy as np


def autokeras_concept():
    """
    Auto-Keras核心概念
    
    神经网络架构搜索(NAS)的目标是自动发现最优网络结构
    """
    
    print("Auto-Keras (NAS) 核心概念:")
    print("=" * 60)
    
    print("""
# Auto-Keras使用示例
import autokeras as ak

# 图像分类器搜索
clf = ak.ImageClassifier(
    max_trials=10,      # 最多尝试10个架构
    overwrite=True,
    objective='val_accuracy'
)

# 自动搜索最优架构
clf.fit(x_train, y_train, epochs=10)

# 导出最优模型
model = clf.export_model()
model.save('best_model.h5')
""")
    
    print("\n神经网络架构搜索的三大挑战:")
    
    print("\n1. 搜索空间巨大")
    print("   - 层数: 1-100+")
    print("   - 每层类型: Conv, Pool, Dense, Dropout...")
    print("   - 每层的超参数: 通道数、核大小、激活函数...")
    print("   - 总组合数: 远超宇宙原子数!")
    
    print("\n2. 评估成本极高")
    print("   - 每个候选架构需要完整训练")
    print("   - ImageNet上训练一个ResNet需要数天")
    print("   - 无法穷举所有可能")
    
    print("\n3. 解决策略")
    print("   a) 权重共享 (Weight Sharing)")
    print("      - ENAS: 所有子模型共享同一套权重")
    print("      - 训练一次，评估多个架构")
    print("   b) 代理模型 (Surrogate Model)")
    print("      - 用一个小数据集快速评估架构")
    print("      - 用贝叶斯优化指导搜索")
    print("   c) 渐进式搜索")
    print("      - 从简单网络开始，逐步增加复杂度")
    print("      - 剪枝表现差的分支")
    
    print("\n经典NAS方法:")
    print("  - NASNet (Google): 强化学习搜索，需要数千GPU天")
    print("  - ENAS: 权重共享，大幅降低计算成本")
    print("  - DARTS: 连续松弛，可微分搜索")
    print("  - Auto-Keras: 贝叶斯优化+贪心搜索，实用高效")


def visualize_search_space():
    """可视化NAS的搜索空间概念"""
    
    print("\n神经网络架构搜索空间示例:")
    print("-" * 50)
    
    # 简单的搜索空间
    blocks = [
        {'type': 'Conv3x3', 'filters': [32, 64, 128]},
        {'type': 'Conv5x5', 'filters': [32, 64, 128]},
        {'type': 'MaxPool', 'size': [2, 3]},
        {'type': 'Skip'},  # 跳跃连接
    ]
    
    print("基本块类型:")
    for i, block in enumerate(blocks):
        print(f"  块{i+1}: {block}")
    
    print("\n一个候选架构:")
    architecture = [
        'Input',
        'Conv3x3(32)',
        'Conv3x3(64)',
        'MaxPool(2)',
        'Conv5x5(128)',
        'GlobalAvgPool',
        'Dense(10)'
    ]
    print("  → ".join(architecture))
    
    print("\nNAS自动搜索的目标:")
    print("  找到在验证集上准确率最高的架构序列")
    print("  同时考虑模型大小、推理速度等约束")


if __name__ == "__main__":
    autokeras_concept()
    visualize_search_space()
```

### 57.5.4 AutoML系统对比与选择

| 系统 | 核心算法 | 优势 | 适用场景 |
|------|---------|------|---------|
| **Auto-sklearn** | 贝叶斯优化 + 元学习 + 集成 | 搜索空间完善，效果稳定 | 表格数据分类/回归 |
| **TPOT** | 遗传编程 | 输出可读代码，灵活组合 | 需要理解最终管道 |
| **Auto-Keras** | 贝叶斯优化 + NAS | 自动深度学习 | 图像、文本、时序数据 |
| **H2O AutoML** | 集成多种方法 | 工业级，易用 | 企业级应用 |
| **Optuna** | 贝叶斯优化 + CMA-ES | 灵活框架，社区活跃 | 自定义搜索空间 |

**选择建议**：
- **快速原型**：Auto-sklearn（表格数据）、Auto-Keras（深度学习）
- **生产环境**：H2O AutoML、自定义Optuna管道
- **研究与理解**：TPOT（输出可学习的代码）

---

## 57.6 多目标与条件超参数

现实世界的超参数调优往往更加复杂：
1. **多目标优化**：同时优化准确率和推理速度
2. **条件超参数**：某些参数只在特定条件下才有效

### 57.6.1 帕累托最优：多目标优化的核心概念

当你有多个相互冲突的目标时（如准确率高通常意味着模型大、推理慢），不存在单一的"最优"解。取而代之的是一组**帕累托最优**解。

**费曼法比喻**：想象你在选购手机，想要**性能好**又想要**价格低**。这两个目标通常是冲突的——顶级旗舰性能最好但最贵，低端机便宜但性能差。

帕累托最优就是找到那些"无法在不牺牲另一个目标的情况下改进任何一个目标"的选项：
- 如果有人推出一款性能相同但 cheaper 的手机，原来的就不是帕累托最优
- 帕累托前沿上的每款手机，都代表了性能和价格的一种权衡

#### 数学定义

对于最小化问题，一个解$x^*$是帕累托最优的，如果不存在其他解$x$使得：
$$\forall i: f_i(x) \leq f_i(x^*) \quad \text{且} \quad \exists j: f_j(x) < f_j(x^*)$$

所有帕累托最优解构成**帕累托前沿**（Pareto Front）。

```python
"""
57.6.1 帕累托最优概念演示
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_pareto_front():
    """生成帕累托前沿示例"""
    
    np.random.seed(42)
    
    # 模拟一些超参数配置的评估结果
    # 目标1: 准确率 (越大越好)
    # 目标2: 推理时间 (越小越好)
    
    n_samples = 100
    accuracy = np.random.uniform(0.6, 0.98, n_samples)
    inference_time = np.random.exponential(0.5, n_samples) * (1 - accuracy) + 0.1
    
    # 找到帕累托前沿
    # 准确率越高越好，推理时间越短越好
    pareto_indices = []
    for i in range(n_samples):
        is_pareto = True
        for j in range(n_samples):
            if i != j:
                # 如果j在准确率上更好或相等，且在推理时间上更好或相等，且至少一个严格更好
                if accuracy[j] >= accuracy[i] and inference_time[j] <= inference_time[i]:
                    if accuracy[j] > accuracy[i] or inference_time[j] < inference_time[i]:
                        is_pareto = False
                        break
        if is_pareto:
            pareto_indices.append(i)
    
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 非帕累托最优的点
    non_pareto = [i for i in range(n_samples) if i not in pareto_indices]
    ax.scatter(accuracy[non_pareto], inference_time[non_pareto], 
               c='lightgray', s=50, alpha=0.6, label='Dominated')
    
    # 帕累托最优的点
    pareto_acc = accuracy[pareto_indices]
    pareto_time = inference_time[pareto_indices]
    
    # 排序以绘制连线
    sort_idx = np.argsort(pareto_acc)
    ax.scatter(pareto_acc[sort_idx], pareto_time[sort_idx], 
               c='red', s=100, alpha=0.8, label='Pareto Optimal', zorder=5)
    ax.plot(pareto_acc[sort_idx], pareto_time[sort_idx], 
            'r--', alpha=0.5, linewidth=2, label='Pareto Front')
    
    ax.set_xlabel('Accuracy (higher is better) →', fontsize=12)
    ax.set_ylabel('Inference Time (lower is better) →', fontsize=12)
    ax.set_title('Multi-Objective Optimization: Pareto Front', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pareto_front.png', dpi=150, bbox_inches='tight')
    print("帕累托前沿图已保存到: pareto_front.png")
    plt.show()
    
    return pareto_indices, accuracy, inference_time


def multi_objective_ei():
    """
    多目标期望改进 (MO-EI) 概念
    
    在多目标贝叶斯优化中，我们需要扩展采集函数
    """
    
    print("多目标贝叶斯优化挑战:")
    print("=" * 60)
    
    print("""
1. 标量化方法 (Scalarization)
   将多目标转化为单目标：
   f_combined = w1 * f1 + w2 * f2
   
   优点: 简单，可使用标准BO
   缺点: 需要预先知道权重，可能错过帕累托前沿的某些部分

2. 超体积改进 (Hypervolume Improvement, HV)
   衡量一个点能增加多少"被支配"的空间体积
   
   超体积: 帕累托前沿与参考点之间的空间
   HV越大，解集越好
   
   EHVI (Expected Hypervolume Improvement) 是多目标BO的主流采集函数

3. 分解方法 (Decomposition)
   将多目标问题分解为多个单目标子问题
   每个子问题使用不同的权重
""")


if __name__ == "__main__":
    generate_pareto_front()
    multi_objective_ei()
```

### 57.6.2 条件超参数处理

在实际应用中，某些超参数的存在**依赖于其他超参数的值**。例如：
- 只有选择SVM且kernel='rbf'时，gamma参数才有意义
- 只有使用Batch Normalization时，momentum参数才有效

**费曼法比喻**：想象你在配置一台电脑：
- 如果你选择独立显卡，才需要配置显存大小
- 如果你选择集成显卡，显存大小这个选项就不存在了

```python
"""
57.6.2 条件超参数处理
"""

import numpy as np


def conditional_hyperparameters():
    """
    条件超参数的概念和处理方法
    """
    
    print("条件超参数示例:")
    print("=" * 60)
    
    # 定义一个有条件超参数的搜索空间
    search_space = {
        'classifier': {
            'type': 'categorical',
            'choices': ['SVM', 'RandomForest', 'MLP']
        },
        
        # SVM特有的条件参数
        'svm_kernel': {
            'type': 'categorical',
            'choices': ['linear', 'rbf', 'poly'],
            'condition': 'classifier == SVM'
        },
        'svm_C': {
            'type': 'float',
            'range': (1e-5, 1e5),
            'log_scale': True,
            'condition': 'classifier == SVM'
        },
        'svm_gamma': {
            'type': 'float',
            'range': (1e-5, 1e5),
            'log_scale': True,
            'condition': 'classifier == SVM and svm_kernel == rbf'
        },
        
        # RandomForest特有的条件参数
        'rf_n_estimators': {
            'type': 'int',
            'range': (10, 500),
            'condition': 'classifier == RandomForest'
        },
        'rf_max_depth': {
            'type': 'int',
            'range': (2, 50),
            'condition': 'classifier == RandomForest'
        },
        
        # MLP特有的条件参数
        'mlp_hidden_layers': {
            'type': 'int',
            'range': (1, 5),
            'condition': 'classifier == MLP'
        },
        'mlp_units_per_layer': {
            'type': 'int',
            'range': (32, 512),
            'condition': 'classifier == MLP'
        },
        'mlp_dropout': {
            'type': 'float',
            'range': (0.0, 0.5),
            'condition': 'mlp_hidden_layers > 1'  # 多层时才需要dropout
        }
    }
    
    print("搜索空间结构:")
    for param, config in search_space.items():
        condition = config.get('condition', 'always active')
        print(f"  {param}: {condition}")
    
    print("\n处理条件超参数的方法:")
    
    print("""
1. 条件贝叶斯优化 (Conditional BO)
   - 使用不同的GP模型处理不同的条件分支
   - 或者使用一个GP，但对无效参数进行特殊处理

2. 树形Parzen估计器 (TPE)
   - 天然支持条件超参数
   - 为每个条件分支维护独立的KDE
   - 这是Optuna库使用的方法

3. 配置空间 (ConfigSpace)
   - 专门用于定义条件超参数搜索空间的库
   - 支持复杂的条件关系和约束
""")


class ConditionalConfigSpace:
    """简化演示条件配置空间的处理"""
    
    def __init__(self):
        self.active_params = {}
    
    def sample(self):
        """从条件搜索空间中采样"""
        config = {}
        
        # 首先采样顶层参数
        config['classifier'] = np.random.choice(['SVM', 'RandomForest', 'MLP'])
        
        # 根据顶层参数采样条件参数
        if config['classifier'] == 'SVM':
            config['svm_kernel'] = np.random.choice(['linear', 'rbf', 'poly'])
            config['svm_C'] = 10 ** np.random.uniform(-5, 5)
            
            if config['svm_kernel'] == 'rbf':
                config['svm_gamma'] = 10 ** np.random.uniform(-5, 5)
        
        elif config['classifier'] == 'RandomForest':
            config['rf_n_estimators'] = np.random.randint(10, 500)
            config['rf_max_depth'] = np.random.randint(2, 50)
        
        elif config['classifier'] == 'MLP':
            config['mlp_hidden_layers'] = np.random.randint(1, 5)
            config['mlp_units_per_layer'] = np.random.choice([32, 64, 128, 256, 512])
            
            if config['mlp_hidden_layers'] > 1:
                config['mlp_dropout'] = np.random.uniform(0, 0.5)
        
        return config


def demo_conditional_sampling():
    """演示条件采样"""
    
    print("\n条件配置采样示例:")
    print("-" * 50)
    
    cs = ConditionalConfigSpace()
    
    for i in range(5):
        config = cs.sample()
        print(f"\n样本{i+1}:")
        for k, v in config.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    conditional_hyperparameters()
    demo_conditional_sampling()
```

### 57.6.3 实用建议

**处理多目标优化**：
1. **先明确业务优先级**：如果两个目标有明确的优先级，可以转化为约束优化
2. **使用帕累托前沿可视化**：帮助决策者理解权衡关系
3. **推荐工具**：BoTorch（多目标贝叶斯优化）、Optuna（支持多目标）

**处理条件超参数**：
1. **使用专业库**：Optuna、ConfigSpace、SMAC3都支持条件超参数
2. **避免过度复杂**：条件嵌套太深会使搜索困难
3. **合理分组**：将相关的条件参数放在一起

---

## 57.7 本章总结

本章我们深入探讨了超参数调优的进阶方法：

### 核心知识点回顾

| 主题 | 核心思想 | 关键算法/方法 |
|------|---------|--------------|
| **贝叶斯优化** | 用代理模型指导搜索 | GP + EI/PI/UCB |
| **多保真度优化** | 用低成本近似指导搜索 | SH, HyperBand, BOHB, ASHA |
| **AutoML** | 自动化整个ML流程 | Auto-sklearn, TPOT, Auto-Keras |
| **多目标优化** | 处理多个冲突目标 | 帕累托最优, EHVI |
| **条件超参数** | 处理参数间的依赖关系 | 条件TPE, ConfigSpace |

### 实践建议

1. **从简单开始**：随机搜索 → 贝叶斯优化 → 多保真度 → AutoML
2. **了解你的成本**：评估一个配置需要多长时间？这决定了你能用什么方法
3. **设置合理的搜索空间**：好的搜索空间比好的优化算法更重要
4. **记录一切**：记录所有实验，建立可复现的研究流程
5. **不要过度调优**：超参数调优的收益递减，找到"足够好"即可

### 前沿趋势

- **神经架构搜索**：自动设计深度学习模型结构
- **零成本代理**：用初始化时的指标预测最终性能
- **终身学习AutoML**：利用之前任务的元知识加速新任务
- **多保真度贝叶斯优化**：更智能地分配不同保真度的评估

超参数调优是机器学习中科学与艺术交汇的地方。掌握这些方法，你将能够更高效地训练出性能卓越的模型！

---

## 参考文献

1. Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13(1), 281-305.

2. Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. *Advances in Neural Information Processing Systems*, 25.

3. Feurer, M., Klein, A., Eggensperger, K., Springenberg, J., Blum, M., & Hutter, F. (2015). Efficient and robust automated machine learning. *Advances in Neural Information Processing Systems*, 28.

4. Jamieson, K., & Talwalkar, A. (2016). Non-stochastic best arm identification and hyperparameter optimization. *Artificial Intelligence and Statistics*, 240-248.

5. Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2017). Hyperband: A novel bandit-based approach to hyperparameter optimization. *Journal of Machine Learning Research*, 18(1), 6765-6816.

6. Falkner, S., Klein, A., & Hutter, F. (2018). BOHB: Robust and efficient hyperparameter optimization at scale. *International Conference on Machine Learning*, 1437-1446.

7. Li, L., Jamieson, K., Rostamizadeh, A., Gonina, E., Hardt, M., Recht, B., & Talwalkar, A. (2018). Massively parallel hyperparameter tuning. *arXiv preprint arXiv:1810.05934*.

8. Olson, R. S., Moore, J. H., & others. (2016). TPOT: A tree-based pipeline optimization tool for automating machine learning. *Automated Machine Learning*, 151-160.

9. Jin, H., Song, Q., & Hu, X. (2019). Auto-keras: An efficient neural architecture search system. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 1946-1956.

10. Zoph, B., & Le, Q. V. (2017). Neural architecture search with reinforcement learning. *International Conference on Learning Representations*.

11. Pham, H., Guan, M. Y., Zoph, B., Le, Q. V., & Dean, J. (2018). Efficient neural architecture search via parameters sharing. *International Conference on Machine Learning*, 4095-4104.

12. Liu, C., Zoph, B., Neumann, M., Shlens, J., Hua, W., Li, L. J., ... & Fei-Fei, L. (2018). Progressive neural architecture search. *European Conference on Computer Vision*, 19-34.

13. Hernandez-Lobato, J. M., Gelbart, M. A., Adams, R. P., Hoffman, M. W., & Ghahramani, Z. (2016). A general framework for constrained Bayesian optimization using information-based search. *Journal of Machine Learning Research*, 17(1), 5549-5601.

14. Frazier, P. I. (2018). A tutorial on Bayesian optimization. *arXiv preprint arXiv:1807.02811*.

15. Kandasamy, K., Schneider, J., & Póczos, B. (2020). Query efficient posterior estimation in scientific experiments via Bayesian active learning. *Artificial Intelligence*, 286, 103342.

---

*本章完*

> *"调参之路漫漫，愿贝叶斯之光指引你找到最优解。"*

