# 第五十七章 超参数调优进阶——AutoML的智能引擎

## 本章导读

想象一下，你正在调试一台复杂的赛车。油门响应、悬挂硬度、轮胎气压、空气动力学套件角度——每一个参数都会影响赛车的性能。专业的赛车工程师需要数年经验才能找到最佳配置。而在机器学习领域，我们面临的"赛车"有数十个甚至上百个"调节旋钮"：学习率、批量大小、网络层数、正则化系数、优化器选择……这就是**超参数调优（Hyperparameter Optimization, HPO）**的挑战。

在上一章中，我们学习了神经架构搜索（NAS），它让AI能够自动设计神经网络结构。而本章将探索另一个AutoML的核心支柱——超参数调优的进阶技术。我们将从基础的网格搜索和随机搜索出发，深入到贝叶斯优化的数学原理，探索多保真度优化的巧妙策略，最后领略自动化机器学习（AutoML）框架的强大能力。

通过本章，你将掌握：
- 🎯 **贝叶斯优化**的核心数学原理，理解高斯过程如何构建"智能代理"
- ⚡ **多保真度优化**策略，学会用更少的计算资源找到更好的超参数
- 🤖 **AutoML框架**的使用方法，让机器自动完成特征工程、模型选择和超参数调优
- 🔬 **多目标优化**的思维方式，在多个目标间寻找最佳平衡

准备好了吗？让我们开启这场超参数优化的深度之旅！

---

## 57.1 引言：超参数调优的艺术与科学

### 57.1.1 什么是超参数？

在机器学习中，参数可以分为两类：

**模型参数（Model Parameters）**：
- 从数据中学习得到
- 例如：神经网络的权重、决策树的节点分裂条件
- 训练过程自动优化

**超参数（Hyperparameters）**：
- 在训练前设定，控制学习过程
- 例如：学习率、批量大小、网络深度、正则化强度
- 需要人工或自动调优

让我们用披萨店来做一个费曼法比喻：

> **费曼法比喻：披萨师傅的秘方**
> 
> 想象你经营一家披萨店。面团发酵时间、烤箱温度、酱料配方比例——这些都是"超参数"。它们不是从顾客订单中学来的，而是需要你预先设定的"秘方"。
> 
> 如果发酵时间太短，面团太硬；太长，面团发酸。如果烤箱温度太低，披萨不熟；太高，饼边焦黑。找到最佳组合需要不断尝试和调整。
> 
> 超参数调优就像寻找最佳披萨配方——你需要系统地探索不同组合，记录顾客反馈（验证集性能），最终找到那个让销量暴涨的"黄金配方"。

### 57.1.2 超参数调优的挑战

超参数调优面临三大挑战：

**1. 维度灾难**
- 假设有10个超参数，每个有5个候选值
- 总组合数：$5^{10} = 9,765,625$种
- 即使每次训练只需10分钟，穷举需要**185年**！

**2. 评估成本高昂**
- 深度学习模型训练可能需要数小时到数天
- 每个超参数组合都需要完整训练评估
- 资源消耗巨大

**3. 非凸、非光滑、带噪声**
- 超参数与性能的关系复杂且不规则
- 存在多个局部最优
- 随机初始化导致评估结果有噪声

### 57.1.3 超参数调优方法演进

让我们回顾HPO方法的发展历程：

| 方法 | 时间 | 核心思想 | 优缺点 |
|------|------|----------|--------|
| 网格搜索 | 传统 | 穷举所有组合 | 简单但计算量大 |
| 随机搜索 | 2012 | 随机采样 | 比网格搜索高效 |
| 贝叶斯优化 | 2010s | 构建概率代理模型 | 样本高效，需先验 |
| 多保真度 | 2016+ | 利用低保真度评估 | 大幅加速 |
| AutoML框架 | 2018+ | 端到端自动化 | 易用但灵活性受限 |

```python
# 超参数调优问题形式化
import numpy as np
from typing import Callable, Dict, Any

class HPOProblem:
    """
    超参数优化问题定义
    
    目标: 找到最优超参数配置 x* = argmin f(x)
    其中 f(x) 是在验证集上的损失函数
    """
    
    def __init__(self, search_space: Dict[str, Any], 
                 objective_fn: Callable, budget: int = 100):
        """
        参数:
            search_space: 超参数搜索空间
            objective_fn: 损失函数（返回验证集性能）
            budget: 最大评估次数
        """
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.budget = budget
        self.history = []  # 记录所有评估历史
        
    def evaluate(self, config: Dict[str, Any]) -> float:
        """评估一个超参数配置"""
        performance = self.objective_fn(config)
        self.history.append({'config': config, 'performance': performance})
        return performance
```

---

## 57.2 网格搜索与随机搜索：基础方法回顾

### 57.2.1 网格搜索（Grid Search）

网格搜索是最直观的超参数调优方法：对每个超参数的候选值进行穷举组合。

**算法原理**：
给定超参数空间 $\mathcal{X} = X_1 \times X_2 \times \cdots \times X_d$，网格搜索评估所有组合：

$$\mathcal{X}_{grid} = \{(x_1, x_2, \ldots, x_d) : x_i \in X_i, \forall i\}$$

```python
import itertools
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import time

class GridSearch:
    """网格搜索实现"""
    
    def __init__(self, param_grid: dict):
        """
        参数:
            param_grid: 参数网格，如 {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, 10]}
        """
        self.param_grid = param_grid
        self.results = []
        
    def get_all_combinations(self):
        """生成所有参数组合"""
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def search(self, model_class, X_train, y_train, X_val, y_val):
        """
        执行网格搜索
        
        返回:
            best_params: 最佳参数
            best_score: 最佳验证分数
        """
        combinations = self.get_all_combinations()
        print(f"总组合数: {len(combinations)}")
        
        best_score = -np.inf
        best_params = None
        
        for i, params in enumerate(combinations):
            # 创建并训练模型
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            # 验证集评估
            score = accuracy_score(y_val, model.predict(X_val))
            
            self.results.append({
                'params': params,
                'score': score,
                'iteration': i
            })
            
            if score > best_score:
                best_score = score
                best_params = params
                print(f"迭代 {i+1}/{len(combinations)}: 新最佳分数 {score:.4f}, 参数 {params}")
            else:
                print(f"迭代 {i+1}/{len(combinations)}: 分数 {score:.4f}")
        
        return best_params, best_score

# 示例：使用Iris数据集进行网格搜索
def demo_grid_search():
    """网格搜索演示"""
    # 加载数据
    iris = load_iris()
    X_train, X_val, y_train, y_val = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [3, 5, None],
        'min_samples_split': [2, 5]
    }
    
    # 执行网格搜索
    gs = GridSearch(param_grid)
    best_params, best_score = gs.search(
        RandomForestClassifier, X_train, y_train, X_val, y_val
    )
    
    print(f"\n最佳参数: {best_params}")
    print(f"最佳验证分数: {best_score:.4f}")
    
    return gs

# 运行演示
# gs = demo_grid_search()
```

**网格搜索的优缺点**：

| 优点 | 缺点 |
|------|------|
| 简单直观，易于实现 | 计算成本指数级增长 |
| 保证找到网格内的最优 | 对重要/不重要参数一视同仁 |
| 可并行化 | 可能错过网格点之间的更好值 |

### 57.2.2 随机搜索（Random Search）

2012年，Bergstra和Bengio在论文《Random Search for Hyper-Parameter Optimization》中证明：在相同计算预算下，随机搜索通常比网格搜索更有效。

**核心洞察**：
- 并非所有超参数都同等重要
- 网格搜索浪费资源在无关紧要的参数上
- 随机搜索给每个重要参数更多探索机会

```python
import random
from typing import List, Union

class RandomSearch:
    """随机搜索实现"""
    
    def __init__(self, param_distributions: dict, n_iter: int = 10):
        """
        参数:
            param_distributions: 参数分布，如 {'n_estimators': [10, 50, 100], 'lr': (0.001, 0.1)}
            n_iter: 随机采样次数
        """
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.results = []
        
    def sample_params(self) -> dict:
        """从参数分布中随机采样"""
        params = {}
        for key, dist in self.param_distributions.items():
            if isinstance(dist, list):
                # 离散选择
                params[key] = random.choice(dist)
            elif isinstance(dist, tuple) and len(dist) == 2:
                # 连续均匀分布
                low, high = dist
                if isinstance(low, int):
                    params[key] = random.randint(low, high)
                else:
                    params[key] = random.uniform(low, high)
            elif isinstance(dist, dict) and dist.get('type') == 'log':
                # 对数均匀分布（适合学习率等）
                low, high = dist['range']
                params[key] = 10 ** random.uniform(np.log10(low), np.log10(high))
        return params
    
    def search(self, model_class, X_train, y_train, X_val, y_val):
        """执行随机搜索"""
        best_score = -np.inf
        best_params = None
        
        for i in range(self.n_iter):
            params = self.sample_params()
            
            # 创建并训练模型
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            # 验证集评估
            score = accuracy_score(y_val, model.predict(X_val))
            
            self.results.append({
                'params': params,
                'score': score,
                'iteration': i
            })
            
            if score > best_score:
                best_score = score
                best_params = params
                print(f"迭代 {i+1}/{self.n_iter}: 新最佳分数 {score:.4f}, 参数 {params}")
            else:
                print(f"迭代 {i+1}/{self.n_iter}: 分数 {score:.4f}")
        
        return best_params, best_score

# 示例：使用Iris数据集进行随机搜索
def demo_random_search():
    """随机搜索演示"""
    # 加载数据
    iris = load_iris()
    X_train, X_val, y_train, y_val = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # 定义参数分布
    param_distributions = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [3, 5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # 执行随机搜索
    rs = RandomSearch(param_distributions, n_iter=20)
    best_params, best_score = rs.search(
        RandomForestClassifier, X_train, y_train, X_val, y_val
    )
    
    print(f"\n最佳参数: {best_params}")
    print(f"最佳验证分数: {best_score:.4f}")
    
    return rs

# 运行演示
# rs = demo_random_search()
```

**网格搜索 vs 随机搜索对比实验**：

```python
def compare_search_methods():
    """对比网格搜索和随机搜索"""
    from sklearn.datasets import make_classification
    
    # 生成合成数据
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 网格搜索
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5]
    }
    gs = GridSearch(param_grid)
    gs_start = time.time()
    gs_best_params, gs_best_score = gs.search(
        RandomForestClassifier, X_train, y_train, X_val, y_val
    )
    gs_time = time.time() - gs_start
    
    # 随机搜索（相同评估次数）
    param_distributions = {
        'n_estimators': list(range(10, 101)),
        'max_depth': [3, 5, 10] + list(range(15, 51)),
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    rs = RandomSearch(param_distributions, n_iter=len(gs.get_all_combinations()))
    rs_start = time.time()
    rs_best_params, rs_best_score = rs.search(
        RandomForestClassifier, X_train, y_train, X_val, y_val
    )
    rs_time = time.time() - rs_start
    
    print("\n" + "="*50)
    print("对比结果")
    print("="*50)
    print(f"网格搜索: 分数={gs_best_score:.4f}, 时间={gs_time:.2f}s")
    print(f"随机搜索: 分数={rs_best_score:.4f}, 时间={rs_time:.2f}s")
    print(f"改进: {(rs_best_score - gs_best_score) / gs_best_score * 100:.2f}%")

# 运行对比
# compare_search_methods()
```

> **费曼法比喻：寻宝游戏**
> 
> 想象你在一个巨大的迷宫里寻找宝藏。网格搜索就像按照严格的网格线一步步检查每一块地板砖——不管宝藏是在墙边还是在中央，你都一视同仁地搜索。
> 
> 随机搜索则像是随机跳跃到不同的位置检查。如果宝藏其实只藏在几个关键区域（重要的超参数），随机跳跃让你有更多机会探索这些区域的不同深度，而不是浪费时间在无关紧要的区域上。

---

## 57.3 贝叶斯优化：智能代理引导的探索

### 57.3.1 贝叶斯优化概述

贝叶斯优化（Bayesian Optimization, BO）是一种样本高效的序列优化方法，特别适用于评估成本昂贵的黑箱函数。

**核心思想**：
1. 基于已观测的数据，构建损失函数的**概率代理模型**（通常使用高斯过程）
2. 使用**采集函数**（Acquisition Function）平衡探索与利用
3. 选择采集函数最大化的点进行下一次评估
4. 迭代更新代理模型，直到预算耗尽

**贝叶斯优化流程**：

$$x_{n+1} = \arg\max_{x \in \mathcal{X}} \alpha(x; \mathcal{D}_{1:n})$$

其中：
- $\mathcal{D}_{1:n} = \{(x_i, y_i)\}_{i=1}^n$ 是已观测数据
- $\alpha(x; \mathcal{D}_{1:n})$ 是采集函数
- $x_{n+1}$ 是下一个要评估的点

```python
class BayesianOptimization:
    """
    贝叶斯优化框架
    
    算法流程:
    1. 初始化高斯过程代理模型
    2. 对于每次迭代:
       a. 基于当前GP拟合采集函数
       b. 找到采集函数最大值的超参数配置
       c. 评估该配置的真实性能
       d. 更新GP模型
    """
    
    def __init__(self, surrogate_model, acquisition_fn, bounds, n_init=5):
        """
        参数:
            surrogate_model: 代理模型（如高斯过程）
            acquisition_fn: 采集函数
            bounds: 超参数边界 [(min, max), ...]
            n_init: 初始随机采样点数
        """
        self.surrogate = surrogate_model
        self.acquisition = acquisition_fn
        self.bounds = bounds
        self.n_init = n_init
        self.X_observed = []
        self.y_observed = []
        
    def optimize(self, objective_fn, n_iterations=50):
        """
        执行贝叶斯优化
        
        参数:
            objective_fn: 黑箱损失函数
            n_iterations: 优化迭代次数
        """
        # 1. 初始随机采样
        print("=" * 60)
        print("贝叶斯优化开始")
        print("=" * 60)
        print(f"初始随机采样 {self.n_init} 个点...")
        
        for i in range(self.n_init):
            x = self._random_sample()
            y = objective_fn(x)
            self.X_observed.append(x)
            self.y_observed.append(y)
            print(f"  初始化 {i+1}/{self.n_init}: y={y:.4f}")
        
        # 2. 贝叶斯优化迭代
        for t in range(n_iterations):
            # 拟合代理模型
            self.surrogate.fit(self.X_observed, self.y_observed)
            
            # 优化采集函数找到下一个点
            x_next = self._optimize_acquisition()
            
            # 评估真实损失函数
            y_next = objective_fn(x_next)
            
            # 更新观测数据
            self.X_observed.append(x_next)
            self.y_observed.append(y_next)
            
            best_y = min(self.y_observed)
            print(f"迭代 {t+1}/{n_iterations}: y={y_next:.4f}, 当前最佳={best_y:.4f}")
        
        # 返回最佳结果
        best_idx = np.argmin(self.y_observed)
        return self.X_observed[best_idx], self.y_observed[best_idx]
    
    def _random_sample(self):
        """在边界内随机采样"""
        return [np.random.uniform(low, high) for low, high in self.bounds]
    
    def _optimize_acquisition(self):
        """优化采集函数（使用随机采样作为简单实现）"""
        best_x = None
        best_acq = -np.inf
        
        # 随机采样多个点，选择采集函数值最大的
        for _ in range(1000):
            x = self._random_sample()
            acq = self.acquisition.evaluate(x, self.surrogate)
            if acq > best_acq:
                best_acq = acq
                best_x = x
        
        return best_x
```

### 57.3.2 高斯过程：概率代理模型

高斯过程（Gaussian Process, GP）是贝叶斯优化中最常用的代理模型。它为函数值提供了完整的概率分布，包括预测均值和不确定性估计。

**高斯过程定义**：

高斯过程是一个随机变量的集合，其中任意有限个随机变量的联合分布都是多元高斯分布。它完全由**均值函数** $m(x)$ 和**协方差函数（核函数）** $k(x, x')$ 定义：

$$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$$

**后验预测分布**：

给定训练数据 $\mathbf{X} = [x_1, \ldots, x_n]^T$ 和观测值 $\mathbf{y} = [y_1, \ldots, y_n]^T$，对于新输入 $x_*$，预测分布为：

$$p(f(x_*) | \mathbf{X}, \mathbf{y}, x_*) = \mathcal{N}(\mu(x_*), \sigma^2(x_*))$$

其中：

$$\mu(x_*) = k_*^T (K + \sigma_n^2 I)^{-1} \mathbf{y}$$

$$\sigma^2(x_*) = k(x_*, x_*) - k_*^T (K + \sigma_n^2 I)^{-1} k_*$$

这里：
- $K_{ij} = k(x_i, x_j)$ 是训练点的核矩阵
- $k_* = [k(x_*, x_1), \ldots, k(x_*, x_n)]^T$
- $\sigma_n^2$ 是观测噪声方差

```python
class GaussianProcess:
    """
    高斯过程回归实现
    
    核心数学:
    - 均值函数: μ(x*) = k_*^T (K + σ²I)^{-1} y
    - 方差函数: σ²(x*) = k(x*,x*) - k_*^T (K + σ²I)^{-1} k_*
    """
    
    def __init__(self, kernel=None, noise_var=1e-5):
        """
        参数:
            kernel: 核函数，默认使用RBF核
            noise_var: 观测噪声方差
        """
        self.kernel = kernel if kernel else RBFKernel()
        self.noise_var = noise_var
        self.X_train = None
        self.y_train = None
        self.K_inv = None  # 缓存核矩阵的逆
        
    def fit(self, X, y):
        """
        拟合高斯过程模型
        
        计算并缓存 (K + σ²I)^{-1} 用于后续预测
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        # 计算核矩阵
        K = self.kernel(self.X_train, self.X_train)
        
        # 添加噪声项并求逆
        K_noisy = K + self.noise_var * np.eye(len(K))
        self.K_inv = np.linalg.inv(K_noisy)
        
    def predict(self, X_test, return_std=True):
        """
        预测新点的均值和方差
        
        参数:
            X_test: 测试点
            return_std: 是否返回标准差
            
        返回:
            mean: 预测均值
            std: 预测标准差（如果return_std=True）
        """
        X_test = np.array(X_test)
        
        # 计算训练点和测试点之间的核
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        K_star = self.kernel(X_test, self.X_train)  # k_*
        
        # 预测均值: μ(x*) = k_*^T (K + σ²I)^{-1} y
        mean = K_star @ self.K_inv @ self.y_train
        
        if not return_std:
            return mean
        
        # 预测方差: σ²(x*) = k(x*,x*) - k_*^T (K + σ²I)^{-1} k_*
        K_star_star = self.kernel(X_test, X_test)
        var = np.diag(K_star_star - K_star @ self.K_inv @ K_star.T)
        std = np.sqrt(np.maximum(var, 0))
        
        return mean, std
    
    def log_marginal_likelihood(self, theta):
        """
        计算对数边际似然（用于超参数优化）
        
        log p(y|X,θ) = -1/2 y^T K^{-1} y - 1/2 log|K| - n/2 log(2π)
        """
        # 暂时保存原核函数参数
        original_params = self.kernel.get_params()
        
        # 设置新参数
        self.kernel.set_params(theta)
        
        K = self.kernel(self.X_train, self.X_train)
        K_noisy = K + self.noise_var * np.eye(len(K))
        
        try:
            L = np.linalg.cholesky(K_noisy)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
            
            # log p(y|X) = -1/2 y^T α - Σ log L_ii - n/2 log(2π)
            log_likelihood = -0.5 * self.y_train.T @ alpha
            log_likelihood -= np.sum(np.log(np.diag(L)))
            log_likelihood -= len(self.y_train) / 2 * np.log(2 * np.pi)
        except np.linalg.LinAlgError:
            log_likelihood = -np.inf
        finally:
            # 恢复原参数
            self.kernel.set_params(original_params)
        
        return log_likelihood


class RBFKernel:
    """
    径向基函数（RBF）核，也称为高斯核
    
    k(x, x') = σ_f² * exp(-||x - x'||² / (2 * l²))
    
    参数:
        length_scale (l): 控制函数的"平滑度"，越大越平滑
        signal_variance (σ_f²): 控制函数的幅值
    """
    
    def __init__(self, length_scale=1.0, signal_variance=1.0):
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        
    def __call__(self, X1, X2):
        """计算核矩阵"""
        # 计算欧氏距离的平方
        sq_dists = (
            np.sum(X1**2, axis=1).reshape(-1, 1) + 
            np.sum(X2**2, axis=1) - 
            2 * X1 @ X2.T
        )
        
        # RBF核
        return self.signal_variance * np.exp(-0.5 * sq_dists / self.length_scale**2)
    
    def get_params(self):
        return {'length_scale': self.length_scale, 'signal_variance': self.signal_variance}
    
    def set_params(self, params):
        self.length_scale = params.get('length_scale', self.length_scale)
        self.signal_variance = params.get('signal_variance', self.signal_variance)


class MaternKernel:
    """
    Matérn核函数，比RBF更通用
    
    ν=1/2: 指数核（不光滑）
    ν=3/2: 一次可微
    ν=5/2: 二次可微（推荐）
    ν→∞: 收敛到RBF核
    """
    
    def __init__(self, length_scale=1.0, signal_variance=1.0, nu=2.5):
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.nu = nu
        
    def __call__(self, X1, X2):
        """计算Matérn核矩阵"""
        from scipy.spatial.distance import cdist
        from scipy.special import gamma, kv
        
        dists = cdist(X1 / self.length_scale, X2 / self.length_scale, metric='euclidean')
        
        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = (1 + np.sqrt(3) * dists) * np.exp(-np.sqrt(3) * dists)
        elif self.nu == 2.5:
            K = (1 + np.sqrt(5) * dists + 5 * dists**2 / 3) * np.exp(-np.sqrt(5) * dists)
        else:
            # 通用Matérn核
            sqrt_2nu = np.sqrt(2 * self.nu)
            K = (2 ** (1 - self.nu) / gamma(self.nu)) * (sqrt_2nu * dists) ** self.nu
            K *= kv(self.nu, sqrt_2nu * dists)
            K[dists == 0] = 1.0
        
        return self.signal_variance * K
    
    def get_params(self):
        return {'length_scale': self.length_scale, 'signal_variance': self.signal_variance, 'nu': self.nu}
    
    def set_params(self, params):
        self.length_scale = params.get('length_scale', self.length_scale)
        self.signal_variance = params.get('signal_variance', self.signal_variance)
        self.nu = params.get('nu', self.nu)
```

**高斯过程可视化示例**：

```python
def demo_gaussian_process():
    """高斯过程可视化演示"""
    import matplotlib.pyplot as plt
    
    # 真实函数（带噪声）
    def true_fn(x):
        return np.sin(3 * x) * np.cos(2 * x) + 0.1 * np.random.randn()
    
    # 生成训练数据
    np.random.seed(42)
    X_train = np.random.uniform(0, 5, 8).reshape(-1, 1)
    y_train = [true_fn(x[0]) for x in X_train]
    
    # 拟合高斯过程
    gp = GaussianProcess(kernel=RBFKernel(length_scale=0.5, signal_variance=1.0))
    gp.fit(X_train, y_train)
    
    # 预测
    X_test = np.linspace(0, 5, 200).reshape(-1, 1)
    mean, std = gp.predict(X_test, return_std=True)
    
    # 可视化
    plt.figure(figsize=(10, 6))
    
    # 训练数据
    plt.scatter(X_train, y_train, c='red', s=100, zorder=5, label='观测数据')
    
    # 预测均值
    plt.plot(X_test, mean, 'b-', label='GP预测均值')
    
    # 置信区间
    plt.fill_between(X_test.flatten(), mean - 2*std, mean + 2*std, 
                     alpha=0.3, color='blue', label='95%置信区间')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('高斯过程回归示例')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('gp_demo.png', dpi=150)
    plt.show()
    
    print("高斯过程拟合完成！")
    print(f"在 x=2.5 处预测: mean={gp.predict([[2.5]])[0]:.4f}")

# 运行演示
# demo_gaussian_process()
```

### 57.3.3 采集函数：平衡探索与利用

采集函数决定了在何处进行下一次评估。好的采集函数应该平衡：
- **利用（Exploitation）**：在预测性能好的区域进行采样
- **探索（Exploration）**：在不确定性高的区域进行采样

**常用采集函数**：

#### 1. 概率改进（Probability of Improvement, PI）

$$\alpha_{PI}(x) = P(f(x) \geq f(x^+) + \xi) = \Phi\left(\frac{\mu(x) - f(x^+) - \xi}{\sigma(x)}\right)$$

其中 $\Phi$ 是标准正态CDF，$\xi$ 是探索参数。

```python
class ProbabilityOfImprovement:
    """概率改进采集函数"""
    
    def __init__(self, xi=0.01):
        """
        参数:
            xi: 探索参数，值越大越倾向于探索
        """
        self.xi = xi
        
    def evaluate(self, x, gp):
        """
        计算给定点的PI值
        
        PI(x) = Φ((μ(x) - f(x+) - ξ) / σ(x))
        """
        from scipy.stats import norm
        
        mu, sigma = gp.predict([x], return_std=True)
        mu = mu[0]
        sigma = sigma[0]
        
        # 当前最佳值
        f_best = np.min(gp.y_train)
        
        if sigma == 0:
            return 0.0
        
        z = (mu - f_best - self.xi) / sigma
        return norm.cdf(z)
```

#### 2. 期望改进（Expected Improvement, EI）

EI考虑了改进的**幅度**，而不仅仅是概率：

$$\alpha_{EI}(x) = \mathbb{E}[\max(0, f(x) - f(x^+))] = \sigma(x) [Z \Phi(Z) + \phi(Z)]$$

其中：
- $Z = \frac{\mu(x) - f(x^+)}{\sigma(x)}$
- $\Phi$ 是标准正态CDF
- $\phi$ 是标准正态PDF

```python
class ExpectedImprovement:
    """期望改进采集函数（最常用）"""
    
    def __init__(self, xi=0.01):
        self.xi = xi
        
    def evaluate(self, x, gp):
        """
        计算EI值
        
        EI(x) = σ(x) * [Z * Φ(Z) + φ(Z)]
        其中 Z = (μ(x) - f(x+) - ξ) / σ(x)
        """
        from scipy.stats import norm
        
        mu, sigma = gp.predict([x], return_std=True)
        mu = mu[0]
        sigma = sigma[0]
        
        f_best = np.min(gp.y_train)
        
        if sigma == 0:
            return 0.0
        
        z = (mu - f_best - self.xi) / sigma
        
        ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
        return ei
```

#### 3. 上置信界限（Upper Confidence Bound, UCB）

UCB显式地控制探索-利用权衡：

$$\alpha_{UCB}(x) = \mu(x) + \kappa \sigma(x)$$

其中 $\kappa$ 控制探索程度：
- $\kappa = 0$：纯利用
- $\kappa \to \infty$：纯探索

```python
class UpperConfidenceBound:
    """上置信界限采集函数"""
    
    def __init__(self, kappa=2.0):
        """
        参数:
            kappa: 探索参数，通常取2.0（对应95%置信区间）
        """
        self.kappa = kappa
        
    def evaluate(self, x, gp):
        """
        UCB(x) = μ(x) + κ * σ(x)
        
        注意：对于最小化问题，使用 LCB = μ(x) - κ * σ(x)
        """
        mu, sigma = gp.predict([x], return_std=True)
        mu = mu[0]
        sigma = sigma[0]
        
        # 对于最大化问题（如准确率），使用UCB
        return mu + self.kappa * sigma
```

#### 4. 汤普森采样（Thompson Sampling）

汤普森采样从后验分布中**采样**一个函数，然后选择该函数的最大值点：

```python
class ThompsonSampling:
    """汤普森采样采集函数"""
    
    def __init__(self):
        pass
        
    def evaluate(self, x, gp):
        """
        从后验中采样一个函数值
        
        注意：实际实现需要对每个候选点采样，然后选择采样值最大的点
        这里简化处理，直接返回采样值用于比较
        """
        mu, sigma = gp.predict([x], return_std=True)
        mu = mu[0]
        sigma = sigma[0]
        
        # 从高斯分布采样
        sample = np.random.normal(mu, sigma)
        return sample
```

**采集函数对比可视化**：

```python
def compare_acquisition_functions():
    """对比不同采集函数的行为"""
    import matplotlib.pyplot as plt
    
    # 生成训练数据
    np.random.seed(42)
    X_train = np.array([[0.5], [1.5], [2.5], [3.5], [4.5]])
    y_train = np.sin(X_train.flatten()) + 0.1 * np.random.randn(5)
    
    # 拟合GP
    gp = GaussianProcess(RBFKernel(length_scale=0.5))
    gp.fit(X_train, y_train)
    
    # 测试点
    X_test = np.linspace(0, 5, 200).reshape(-1, 1)
    mean, std = gp.predict(X_test, return_std=True)
    
    # 计算各采集函数
    ei = ExpectedImprovement(xi=0.01)
    ucb = UpperConfidenceBound(kappa=2.0)
    pi = ProbabilityOfImprovement(xi=0.01)
    
    ei_values = [ei.evaluate(x, gp) for x in X_test]
    ucb_values = [ucb.evaluate(x, gp) for x in X_test]
    pi_values = [pi.evaluate(x, gp) for x in X_test]
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # GP预测
    ax = axes[0, 0]
    ax.scatter(X_train, y_train, c='red', s=100, zorder=5)
    ax.plot(X_test, mean, 'b-', label='均值')
    ax.fill_between(X_test.flatten(), mean - 2*std, mean + 2*std, alpha=0.3)
    ax.set_title('高斯过程预测')
    ax.legend()
    
    # EI
    ax = axes[0, 1]
    ax.plot(X_test, ei_values, 'g-', linewidth=2)
    ax.set_title('期望改进 (EI)')
    ax.axvline(X_test[np.argmax(ei_values)], color='r', linestyle='--', label='最大值')
    ax.legend()
    
    # UCB
    ax = axes[1, 0]
    ax.plot(X_test, ucb_values, 'm-', linewidth=2)
    ax.set_title('上置信界限 (UCB)')
    ax.axvline(X_test[np.argmax(ucb_values)], color='r', linestyle='--', label='最大值')
    ax.legend()
    
    # PI
    ax = axes[1, 1]
    ax.plot(X_test, pi_values, 'c-', linewidth=2)
    ax.set_title('概率改进 (PI)')
    ax.axvline(X_test[np.argmax(pi_values)], color='r', linestyle='--', label='最大值')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('acquisition_comparison.png', dpi=150)
    plt.show()
    
    print("采集函数对比完成！")

# 运行对比
# compare_acquisition_functions()
```

> **费曼法比喻：寻宝与地图**
> 
> 想象你在一座未知的荒岛上寻找宝藏。高斯过程就像你手中的地图——它根据已探索区域（观测数据）画出整个岛屿的地形，包括你已知的部分（均值）和你不确定的部分（方差）。
> 
> 采集函数则是你的寻宝策略：
> - **利用**派会建议你去地图上标记为"可能有很多黄金"的地方
> - **探索**派会建议你去地图上标记为"未知区域"的地方
> 
> EI（期望改进）像个精明的商人："如果我挖这里，预期能赚多少钱？"它既考虑发现宝藏的概率，也考虑宝藏的价值。
> 
> UCB像个冒险家："我选择最有潜力的地方——要么确定的好，要么不确定但可能极好的地方。"

### 57.3.4 完整贝叶斯优化实现

现在让我们将所有组件组合起来，实现完整的贝叶斯优化算法：

