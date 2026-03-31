# 第五十一章 因果推断基础

> **从相关到因果，AI的终极追问**

---

## 学习目标

完成本章学习后，你将能够：

- 理解**相关不等于因果**的深刻含义，识别混杂因素
- 掌握因果推断的**三大层级**：关联、干预、反事实
- 运用**潜在结果框架**（Rubin模型）估计因果效应
- 使用**结构因果模型**（Pearl框架）进行因果推理
- 应用**do-演算**和**后门准则**识别因果效应
- 实现因果效应估计的代码，包括IPW、双重稳健等方法

---

## 1. 引言：为什么相关不等于因果？

### 1.1 费曼法解释：开关与影子

> **类比**: 想象你站在房间里，看到墙上的影子随着你的移动而变化。影子**相关于**你的位置，但影子**不是**你移动的原因。真正的**因果**是：你移动（因）→ 影子变化（果）。

因果推断的核心问题是：**当我们看到X和Y一起变化时，如何确定是X导致Y，还是Y导致X，抑或两者都被第三个因素Z所导致？**

机器学习中，我们习惯了预测：给定输入X，预测输出Y。但预测只关心$P(Y|X)$，不关心**如果改变X，Y会如何变化**。这就是因果推断要回答的问题。

### 1.2 辛普森悖论：当统计撒谎

**案例**: 某医院对两种肾结石治疗方法进行比较：

| 结石类型 | 开放手术成功率 | 微创手术成功率 |
|---------|--------------|--------------|
| 小结石 | 93% (81/87) | 87% (234/270) |
| 大结石 | 73% (192/263) | 69% (55/80) |
| **总体** | **78% (273/350)** | **83% (289/350)** |

**悖论**: 无论结石大小，开放手术成功率都更高，但总体成功率却是微创手术更高！

**原因**: 混杂因素——医生倾向于对更严重的患者（大结石）使用开放手术。如果我们只看总体数据，就会得出错误结论。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 辛普森悖论示例数据
np.random.seed(42)

# 创建数据
small_open = {'treatment': '开放手术', 'stone_size': '小', 
              'success': 81, 'total': 87, 'rate': 0.93}
small_minimal = {'treatment': '微创手术', 'stone_size': '小', 
                 'success': 234, 'total': 270, 'rate': 0.87}
large_open = {'treatment': '开放手术', 'stone_size': '大', 
              'success': 192, 'total': 263, 'rate': 0.73}
large_minimal = {'treatment': '微创手术', 'stone_size': '大', 
                 'success': 55, 'total': 80, 'rate': 0.69}

data = pd.DataFrame([small_open, small_minimal, large_open, large_minimal])

# 计算总体成功率
total_open = data[data['treatment'] == '开放手术']
total_minimal = data[data['treatment'] == '微创手术']

overall_open = total_open['success'].sum() / total_open['total'].sum()
overall_minimal = total_minimal['success'].sum() / total_minimal['total'].sum()

print("=== 辛普森悖论演示 ===\n")
print("按结石类型分组:")
print(data.pivot(index='stone_size', columns='treatment', values='rate'))
print(f"\n总体成功率:")
print(f"  开放手术: {overall_open:.1%}")
print(f"  微创手术: {overall_minimal:.1%}")
print(f"\n悖论: 每组内开放手术更好，但总体微创手术'看起来'更好!")
```

### 1.3 机器学习的局限与因果推断的必要性

**机器学习的核心假设**: 训练分布 = 测试分布

**现实世界的问题**:
1. **分布偏移**: 模型部署环境可能与训练环境不同
2. **策略评估**: "如果我们改变策略，会发生什么？"
3. **公平性**: "是种族导致拒贷，还是收入？"
4. **可解释性**: "为什么模型做出这个预测？"

这些问题都需要**因果推理**，而非仅仅相关性分析。

---

## 2. 因果推断的三大层级：Pearl的因果阶梯

### 2.1 费曼法解释：三层楼的房子

> **类比**: 想象一栋三层楼的房子：
> - **一楼（关联层）**: 你观察世界，看到事物的共现——冰淇淋销量和溺水事件同时增加
> - **二楼（干预层）**: 你动手改变——禁止卖冰淇淋，看看溺水事件是否减少
> - **三楼（反事实层）**: 你想象平行世界——如果昨天我没吃冰淇淋，今天会感冒吗？

Judea Pearl（2011年图灵奖得主）将因果推理分为三个递增的层级：

| 层级 | 问题类型 | 符号表示 | 能力 |
|-----|---------|---------|-----|
| **1. 关联** | 观察到X时，Y是什么？ | $P(Y|X)$ | 预测、诊断 |
| **2. 干预** | 如果设置X=x，Y会怎样？ | $P(Y|do(X=x))$ | 策略评估、决策 |
| **3. 反事实** | 如果当初X不同，Y会怎样？ | $P(Y_x=x'\|X=x, Y=y)$ | 归因、学习 |

### 2.2 第一层：关联（Seeing）

这是传统机器学习的舒适区。我们看到数据，学习$P(Y|X)$。

**局限**: 
- 无法回答"如果...会怎样"的问题
- 无法处理分布偏移
- 可能受到混杂因素的误导

**数学表达**:
$$P(Y|X) = \frac{P(X,Y)}{P(X)}$$

### 2.3 第二层：干预（Doing）

这是因果推断的核心。我们不只是观察X和Y的关系，而是**强制设置X的值**，看Y如何变化。

**关键区别**:
- $P(Y|X=x)$: 观察到X=x的人群中Y的分布
- $P(Y|do(X=x))$: **所有人都被强制设置**X=x后Y的分布

**数学表达**:
$$P(Y|do(X=x)) = \sum_{z} P(Y|X=x, Z=z) P(Z=z)$$

这就是**后门调整公式**，我们稍后会详细推导。

### 2.4 第三层：反事实（Imagining）

这是最高级的因果推理。我们已经观察到X=x且Y=y，现在问：**如果在平行宇宙中X=x'，Y会是什么？**

**应用场景**:
- "如果我当年选择了另一所大学，现在收入会不同吗？"
- "如果患者接受了另一种治疗，会康复吗？"
- "为什么模型做出了这个预测？"（因果归因）

**数学表达**:
$$P(Y_{X=x'} = y' | X=x, Y=y)$$

这表示：在已知X=x且Y=y的个体中，如果当初设置X=x'，Y=y'的概率是多少？

### 2.5 代码演示：三个层级的区别

```python
"""
因果推断三大层级演示
模拟一个简单场景：学习小时数(X)对考试成绩(Y)的影响，考虑天赋(Z)作为混杂因素
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
n = 10000

# 混杂因素：天赋 (影响学习时间和考试成绩)
Z = np.random.normal(0, 1, n)

# 学习时间 (受天赋影响)
X = 2 + 0.5 * Z + np.random.normal(0, 0.5, n)
X = np.clip(X, 0, 5)  # 限制在0-5小时

# 考试成绩 (受天赋和学习时间共同影响)
# 真实因果效应: 学习1小时提升2分
Y = 50 + 2 * X + 3 * Z + np.random.normal(0, 2, n)

print("=== 因果推断三大层级演示 ===\n")

# 层级1: 关联层 P(Y|X)
print("【层级1: 关联层】观察性关联")
print(f"  学习时间每增加1小时，成绩平均变化: {np.corrcoef(X, Y)[0,1]:.2f} (相关性)")

# 简单线性回归系数
slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
print(f"  线性回归系数 (观测关联): {slope:.2f}")
print("  ⚠️ 这个系数被天赋混杂了！高估了学习时间的真实效应\n")

# 层级2: 干预层 P(Y|do(X))
print("【层级2: 干预层】干预效应")
print("  真实因果效应: 学习1小时 → 成绩提升2分 (由数据生成机制决定)")
print("  如果我们能随机分配学习时间，就能估计这个效应")

# 模拟随机实验 (RCT)
X_random = np.random.uniform(0, 5, n)
Y_random = 50 + 2 * X_random + 3 * Z + np.random.normal(0, 2, n)
slope_random, _, _, _, _ = stats.linregress(X_random, Y_random)
print(f"  随机实验估计的因果效应: {slope_random:.2f} ✓\n")

# 层级3: 反事实层
print("【层级3: 反事实层】")
sample_idx = 0
print(f"  学生{sample_idx}: 实际学习了{X[sample_idx]:.1f}小时，考了{Y[sample_idx]:.1f}分")
print(f"  问题: 如果当初学习了5小时 (而非{X[sample_idx]:.1f}小时)，会考多少分？")

# 反事实预测 (知道真实模型)
Z_sample = Z[sample_idx]
counterfactual_Y = 50 + 2 * 5 + 3 * Z_sample  # 假设学习5小时
print(f"  反事实预测成绩: {counterfactual_Y:.1f}分")
print(f"  差值: {counterfactual_Y - Y[sample_idx]:.1f}分 (这就是个体因果效应)")
```

---

## 3. 潜在结果框架：Rubin因果模型

### 3.1 费曼法解释：平行宇宙的两条路

> **类比**: 想象你在一个岔路口。你选择向左走，到达了一个城市。你永远不会知道如果当初向右走会怎样。**潜在结果**就是这两个平行宇宙中的结果——一个实际发生（事实），一个永远未知（反事实）。

Donald Rubin（哈佛统计学家）在1974年提出了**潜在结果框架**（Potential Outcomes Framework），也称为**Rubin因果模型**（RCM）。

### 3.2 核心概念与符号

**基本设定**:
- **单元**（Unit）$i$: 研究对象（一个人、一个公司、一个地区）
- **处理**（Treatment）$T_i \in \{0, 1\}$: 1表示接受处理，0表示对照
- **潜在结果**（Potential Outcomes）:
  - $Y_i(1)$: 单元$i$接受处理时的结果
  - $Y_i(0)$: 单元$i$未接受处理时的结果

**个体因果效应**（Individual Treatment Effect, ITE）:
$$\tau_i = Y_i(1) - Y_i(0)$$

### 3.3 因果推断的根本性问题

**Fundamental Problem of Causal Inference**:
> 对于任何单元$i$，我们只能观察到$Y_i(1)$或$Y_i(0)$中的一个，永远不可能同时观察到两者。

这被称为**反事实缺失问题**。

**实际观察到的结果**:
$$Y_i = T_i \cdot Y_i(1) + (1 - T_i) \cdot Y_i(0)$$

### 3.4 平均因果效应（ATE）

由于个体因果效应不可识别，我们转而关注**总体平均**:

**平均因果效应**（Average Treatment Effect, ATE）:
$$\tau = \mathbb{E}[Y(1) - Y(0)] = \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)]$$

**条件平均因果效应**（Conditional Average Treatment Effect, CATE）:
$$\tau(x) = \mathbb{E}[Y(1) - Y(0) | X = x]$$

**处理组平均因果效应**（Average Treatment Effect on the Treated, ATT）:
$$\tau_{ATT} = \mathbb{E}[Y(1) - Y(0) | T = 1]$$

### 3.5 三大基本假设

为了从观测数据识别因果效应，潜在结果框架依赖三个关键假设：

#### 假设1: SUTVA（稳定单位处理值假设）

**Stable Unit Treatment Value Assumption**

包含两个子假设：
1. **无干扰假设**（No Interference）: 一个单元的潜在结果不受其他单元处理状态的影响
   $$Y_i(t_1, ..., t_n) = Y_i(t_i)$$

2. **一致性假设**（Consistency）: 处理的定义是明确的，没有不同"版本"的处理

**反例**: 疫苗接种（一个人的接种会影响周围人的感染概率）就违反了这个假设。

#### 假设2: 可忽略性（Ignorability）

也称为**无混杂假设**（Unconfoundedness）或**条件独立性**:

$$(Y(1), Y(0)) \perp T \,|\, X$$

**解释**: 给定协变量$X$后，处理分配与潜在结果独立。

**直观理解**: 具有相同特征$X$的单元，其处理分配是随机的（如同抛硬币）。

**数学意义**: 
$$\mathbb{E}[Y(0) | T=1, X=x] = \mathbb{E}[Y(0) | T=0, X=x]$$
这允许我们用对照组的平均结果来估计处理组的反事实。

#### 假设3: 正值假设（Positivity）

$$0 < P(T=1 | X=x) < 1 \quad \text{对所有}x$$

**解释**: 对于任何特征组合$x$，都有非零概率接受处理和对照。

**必要性**: 如果某些群体永远不接受处理，我们就无法估计他们接受处理的效应。

### 3.6 数学推导：ATE的识别

在可忽略性假设下，我们可以证明ATE是可识别的：

$$\begin{aligned}
\tau &= \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)] \\
&= \mathbb{E}_X[\mathbb{E}[Y(1)|X]] - \mathbb{E}_X[\mathbb{E}[Y(0)|X]] \quad \text{(全期望公式)} \\
&= \mathbb{E}_X[\mathbb{E}[Y(1)|T=1, X]] - \mathbb{E}_X[\mathbb{E}[Y(0)|T=0, X]] \quad \text{(可忽略性)} \\
&= \mathbb{E}_X[\mathbb{E}[Y|T=1, X]] - \mathbb{E}_X[\mathbb{E}[Y|T=0, X]]
\end{aligned}$$

**识别策略**（Identification Strategy）:
1. 分层估计：在每个$X$层内计算处理组和对照组的平均差异
2. 加权平均：按$P(X)$加权各层的差异

### 3.7 倾向得分（Propensity Score）

**定义**: 给定协变量$X$下接受处理的概率
$$e(x) = P(T=1 | X=x)$$

**Rosenbaum & Rubin (1983) 的重要定理**:
> 如果$(Y(0), Y(1)) \perp T \,|\, X$（可忽略性成立），那么
> $$(Y(0), Y(1)) \perp T \,|\, e(X)$$

**意义**: 我们只需要控制倾向得分，而非整个$X$向量！这大大简化了高维协变量的调整。

**基于倾向得分的ATE估计**:

$$\tau = \mathbb{E}\left[\frac{T \cdot Y}{e(X)} - \frac{(1-T) \cdot Y}{1-e(X)}\right]$$

这是**逆概率加权**（Inverse Probability Weighting, IPW）估计量的理论基础。

### 3.8 代码实现：潜在结果框架基础

```python
"""
潜在结果框架（Rubin因果模型）基础实现
"""

import numpy as np
from typing import Tuple, Optional, Callable
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


class PotentialOutcomeModel:
    """
    潜在结果框架基础类
    
    实现了Rubin因果模型的核心概念和基本估计方法
    """
    
    def __init__(self):
        self.ate_ = None
        self.att_ = None
        self.cate_estimates_ = None
        
    def calculate_individual_effect(self, y1: float, y0: float) -> float:
        """
        计算个体因果效应 (ITE)
        
        注意: 实际上我们无法同时观察到y1和y0
        这个方法主要用于理论演示和模拟数据
        
        Args:
            y1: 接受处理时的潜在结果
            y0: 未接受处理时的潜在结果
            
        Returns:
            个体因果效应 τ_i = Y_i(1) - Y_i(0)
        """
        return y1 - y0
    
    def estimate_ate_naive(self, Y: np.ndarray, T: np.ndarray) -> float:
        """
        朴素ATE估计 (简单均值差)
        
        公式: ATE = E[Y|T=1] - E[Y|T=0]
        
        ⚠️ 警告: 这种方法在有混杂因素时是有偏的！
        
        Args:
            Y: 观察到的结果
            T: 处理指示变量 (0或1)
            
        Returns:
            朴素ATE估计
        """
        treated_mean = np.mean(Y[T == 1])
        control_mean = np.mean(Y[T == 0])
        ate = treated_mean - control_mean
        
        print(f"处理组均值: {treated_mean:.3f}")
        print(f"对照组均值: {control_mean:.3f}")
        print(f"朴素ATE估计: {ate:.3f}")
        print("⚠️ 注意: 此估计可能因混杂因素而有偏")
        
        return ate
    
    def estimate_ate_stratified(self, Y: np.ndarray, T: np.ndarray, 
                                 X: np.ndarray, n_strata: int = 5) -> float:
        """
        分层估计ATE
        
        在每个协变量层内计算处理效应，然后加权平均
        
        公式: ATE = Σ_k P(X∈ strata_k) × (E[Y|T=1,X∈strata_k] - E[Y|T=0,X∈strata_k])
        
        Args:
            Y: 观察到的结果
            T: 处理指示变量
            X: 协变量 (可以是多维，这里简化为一维用于分层)
            n_strata: 分层数量
            
        Returns:
            分层ATE估计
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # 使用第一个维度进行分层
        X_1d = X[:, 0] if X.shape[1] > 1 else X.flatten()
        
        # 创建分层
        strata_bounds = np.percentile(X_1d, np.linspace(0, 100, n_strata + 1))
        
        ate_strata = []
        weights = []
        
        print(f"\n=== 分层估计 (共{n_strata}层) ===")
        
        for i in range(n_strata):
            lower = strata_bounds[i]
            upper = strata_bounds[i + 1]
            
            # 该层内的样本
            if i < n_strata - 1:
                in_strata = (X_1d >= lower) & (X_1d < upper)
            else:
                in_strata = (X_1d >= lower) & (X_1d <= upper)
            
            if np.sum(in_strata) == 0:
                continue
                
            Y_strata = Y[in_strata]
            T_strata = T[in_strata]
            
            # 该层内的处理组和对照组
            treated = Y_strata[T_strata == 1]
            control = Y_strata[T_strata == 0]
            
            if len(treated) > 0 and len(control) > 0:
                effect = np.mean(treated) - np.mean(control)
                weight = np.sum(in_strata) / len(Y)
                
                ate_strata.append(effect)
                weights.append(weight)
                
                print(f"  层{i+1} [{lower:.2f}, {upper:.2f}]: "
                      f"效应={effect:.3f}, 权重={weight:.3f}")
        
        ate = np.average(ate_strata, weights=weights)
        print(f"\n分层ATE估计: {ate:.3f}")
        
        return ate


class PropensityScoreMatcher:
    """
    倾向得分匹配实现
    
    基于Rosenbaum & Rubin (1983)的经典方法
    """
    
    def __init__(self, model: Optional[Callable] = None, 
                 caliper: Optional[float] = None):
        """
        Args:
            model: 用于估计倾向得分的模型，默认使用逻辑回归
            caliper: 匹配半径，超过此距离的配对会被丢弃
        """
        self.model = model or LogisticRegression(max_iter=1000)
        self.caliper = caliper
        self.propensity_scores_ = None
        
    def fit(self, X: np.ndarray, T: np.ndarray) -> 'PropensityScoreMatcher':
        """
        估计倾向得分
        
        Args:
            X: 协变量
            T: 处理指示变量
        """
        self.model.fit(X, T)
        self.propensity_scores_ = self.model.predict_proba(X)[:, 1]
        
        print("=== 倾向得分估计 ===")
        print(f"处理组倾向得分均值: {np.mean(self.propensity_scores_[T==1]):.3f}")
        print(f"对照组倾向得分均值: {np.mean(self.propensity_scores_[T==0]):.3f}")
        
        return self
    
    def match(self, T: np.ndarray, method: str = 'nearest') -> np.ndarray:
        """
        执行匹配
        
        Args:
            T: 处理指示变量
            method: 匹配方法 ('nearest'为最近邻匹配)
            
        Returns:
            匹配对索引
        """
        if self.propensity_scores_ is None:
            raise ValueError("请先调用fit()估计倾向得分")
        
        treated_idx = np.where(T == 1)[0]
        control_idx = np.where(T == 0)[0]
        
        treated_ps = self.propensity_scores_[treated_idx].reshape(-1, 1)
        control_ps = self.propensity_scores_[control_idx].reshape(-1, 1)
        
        # 最近邻匹配
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(control_ps)
        distances, matched_control_local = nn.kneighbors(treated_ps)
        
        matched_control_idx = control_idx[matched_control_local.flatten()]
        
        # 应用caliper
        if self.caliper is not None:
            valid_matches = distances.flatten() <= self.caliper
            treated_idx = treated_idx[valid_matches]
            matched_control_idx = matched_control_idx[valid_matches]
            
            print(f"\n匹配后样本数: {len(treated_idx)} (原处理组: {np.sum(T==1)})")
        
        # 返回匹配对
        matched_pairs = np.column_stack([treated_idx, matched_control_idx])
        
        print(f"平均匹配距离: {np.mean(distances):.4f}")
        
        return matched_pairs
    
    def estimate_ate(self, Y: np.ndarray, matched_pairs: np.ndarray) -> float:
        """
        基于匹配对估计ATE
        
        Args:
            Y: 结果变量
            matched_pairs: 匹配对索引 (n_pairs, 2)
            
        Returns:
            ATE估计
        """
        treated_outcomes = Y[matched_pairs[:, 0]]
        control_outcomes = Y[matched_pairs[:, 1]]
        
        ate = np.mean(treated_outcomes - control_outcomes)
        
        print(f"\n=== 匹配后ATE估计 ===")
        print(f"处理组均值: {np.mean(treated_outcomes):.3f}")
        print(f"对照组均值: {np.mean(control_outcomes):.3f}")
        print(f"匹配ATE估计: {ate:.3f}")
        
        return ate


class IPWEstimator:
    """
    逆概率加权(Inverse Probability Weighting)估计器
    
    基于倾向得分的加权方法
    """
    
    def __init__(self, ps_model: Optional[Callable] = None,
                 trim_threshold: Optional[float] = None):
        """
        Args:
            ps_model: 倾向得分估计模型
            trim_threshold: 倾向得分截断阈值，避免极端权重
        """
        self.ps_model = ps_model or LogisticRegression(max_iter=1000)
        self.trim_threshold = trim_threshold
        self.propensity_scores_ = None
        
    def fit(self, X: np.ndarray, T: np.ndarray) -> 'IPWEstimator':
        """估计倾向得分"""
        self.ps_model.fit(X, T)
        self.propensity_scores_ = self.ps_model.predict_proba(X)[:, 1]
        
        # 避免极端值
        self.propensity_scores_ = np.clip(self.propensity_scores_, 0.05, 0.95)
        
        return self
    
    def estimate_ate(self, Y: np.ndarray, T: np.ndarray) -> float:
        """
        使用IPW估计ATE
        
        公式: ATE = E[T*Y/e(X) - (1-T)*Y/(1-e(X))]
        
        Args:
            Y: 结果变量
            T: 处理指示变量
            
        Returns:
            IPW ATE估计
        """
        if self.propensity_scores_ is None:
            raise ValueError("请先调用fit()")
        
        ps = self.propensity_scores_
        
        # 计算权重
        weights_treated = T / ps
        weights_control = (1 - T) / (1 - ps)
        
        # 标准化ATE (Hajek估计量，更稳定)
        ate = (np.sum(weights_treated * Y) / np.sum(weights_treated) - 
               np.sum(weights_control * Y) / np.sum(weights_control))
        
        print("=== IPW ATE估计 ===")
        print(f"处理组权重范围: [{np.min(weights_treated[T==1]):.2f}, "
              f"{np.max(weights_treated[T==1]):.2f}]")
        print(f"对照组权重范围: [{np.min(weights_control[T==0]):.2f}, "
              f"{np.max(weights_control[T==0]):.2f}]")
        print(f"IPW ATE估计: {ate:.3f}")
        
        return ate


# 演示：使用模拟数据测试各种估计器
if __name__ == "__main__":
    np.random.seed(42)
    n = 2000
    
    # 生成模拟数据
    # 混杂因素: 年龄
    age = np.random.normal(35, 10, n)
    
    # 处理分配 (受年龄影响 - 混杂!)
    # 年龄越大越可能接受治疗
    prob_treatment = 1 / (1 + np.exp(-(age - 35) / 10))
    T = (np.random.uniform(0, 1, n) < prob_treatment).astype(int)
    
    # 真实潜在结果
    # 真实因果效应: 处理使结果增加2
    Y0 = 50 + 0.5 * age + np.random.normal(0, 5, n)
    Y1 = Y0 + 2 + np.random.normal(0, 1, n)  # 真实ATE = 2
    
    # 观察到的结果
    Y = T * Y1 + (1 - T) * Y0
    
    # 真实ATE
    true_ate = np.mean(Y1 - Y0)
    
    print("=" * 60)
    print("潜在结果框架演示")
    print("=" * 60)
    print(f"\n真实ATE: {true_ate:.3f}")
    print(f"处理组比例: {np.mean(T):.1%}")
    print(f"样本量: {n}\n")
    
    # 1. 朴素估计
    print("\n" + "=" * 60)
    pom = PotentialOutcomeModel()
    naive_ate = pom.estimate_ate_naive(Y, T)
    
    # 2. 分层估计
    print("\n" + "=" * 60)
    strat_ate = pom.estimate_ate_stratified(Y, T, age.reshape(-1, 1), n_strata=5)
    
    # 3. 倾向得分匹配
    print("\n" + "=" * 60)
    X = age.reshape(-1, 1)
    psm = PropensityScoreMatcher(caliper=0.1)
    psm.fit(X, T)
    matched_pairs = psm.match(T)
    psm_ate = psm.estimate_ate(Y, matched_pairs)
    
    # 4. IPW估计
    print("\n" + "=" * 60)
    ipw = IPWEstimator()
    ipw.fit(X, T)
    ipw_ate = ipw.estimate_ate(Y, T)
    
    # 总结
    print("\n" + "=" * 60)
    print("结果总结")
    print("=" * 60)
    print(f"{'方法':<20} {'ATE估计':>10} {'偏差':>10}")
    print("-" * 60)
    print(f"{'真实ATE':<20} {true_ate:>10.3f} {'-':>10}")
    print(f"{'朴素估计':<20} {naive_ate:>10.3f} {abs(naive_ate-true_ate):>10.3f}")
    print(f"{'分层估计':<20} {strat_ate:>10.3f} {abs(strat_ate-true_ate):>10.3f}")
    print(f"{'PS匹配':<20} {psm_ate:>10.3f} {abs(psm_ate-true_ate):>10.3f}")
    print(f"{'IPW估计':<20} {ipw_ate:>10.3f} {abs(ipw_ate-true_ate):>10.3f}")
```

---



## 4. 结构因果模型：Pearl的因果图框架

### 4.1 费曼法解释：因果如河流网络

> **类比**: 想象一条河流系统。雨水（原因）汇入小溪，小溪汇入河流，最终到达湖泊（结果）。如果我们想知道在A点建大坝会怎样影响下游B点的水位，我们需要知道整个河流网络的**结构**。因果图就是这样一个网络图，告诉我们变量如何相互影响。

Judea Pearl提出了**结构因果模型**（Structural Causal Model, SCM），使用**有向无环图**（DAG）来表示变量间的因果关系。

### 4.2 有向无环图（DAG）

**定义**: DAG是一个图$G = (V, E)$，其中：
- $V$是节点集合（代表变量）
- $E$是有向边集合（代表因果关系）
- 图中不存在有向环（无循环因果）

**符号约定**:
- $X \rightarrow Y$: X直接导致Y
- $X \leftarrow Y$: Y导致X
- $X \leftrightarrow Y$: X和Y有共同原因（混杂）

### 4.3 基本因果结构

所有因果图都由三种基本结构组成：

#### 链（Chain）: $X \rightarrow Z \rightarrow Y$
- X通过Z间接影响Y
- 控制Z会**阻断**X到Y的因果路径
- 这被称为**中介效应**

**示例**: 吸烟 → 焦油积累 → 肺癌

#### 分叉（Fork）: $X \leftarrow Z \rightarrow Y$
- Z是X和Y的**共同原因**
- X和Y在边际上相关，但条件于Z后独立
- 这是**混杂**的典型结构

**示例**: 基因 → 吸烟倾向, 基因 → 肺癌风险

#### 对撞（Collider）: $X \rightarrow Z \leftarrow Y$
- Z是X和Y的**共同结果**
- X和Y在边际上独立，但条件于Z后相关
- 这被称为**选择偏差**的来源

**示例**: 才华 → 名人, 美貌 → 名人
（才华和美貌原本无关，但在名人这个子集中变得负相关—— Berkson悖论）

### 4.4 d-分离：图论中的条件独立

**定义**: 在DAG中，给定集合$Z$，如果路径$p$被"阻断"，则称$Z$**d-分离**了路径$p$上的节点。

**d-分离规则**:
1. **链** $A \rightarrow B \rightarrow C$: 控制B阻断路径
2. **分叉** $A \leftarrow B \rightarrow C$: 控制B阻断路径
3. **对撞** $A \rightarrow B \leftarrow C$: 控制B**打开**路径（产生虚假相关）

**全局马尔可夫性质**:
> 如果$Z$ d-分离了$X$和$Y$，则在所有与图一致的概率分布中，$X \perp Y \,|\, Z$。

### 4.5 结构方程模型（SEM）

DAG只表示定性关系，SEM给出定量关系：

$$X_i = f_i(PA_i, \epsilon_i), \quad i = 1, ..., n$$

其中：
- $PA_i$: $X_i$的父节点（直接原因）
- $f_i$: 结构函数（可以是线性或非线性）
- $\epsilon_i$: 外生噪声（相互独立）

**线性SEM示例**:
$$\begin{aligned}
Z &= \epsilon_Z \\
X &= \alpha Z + \epsilon_X \\
Y &= \beta X + \gamma Z + \epsilon_Y
\end{aligned}$$

### 4.6 干预与do-演算

#### 干预操作 $do(X=x)$

在SEM框架中，干预意味着**修改结构方程**：

**干预前**:
$$X = f_X(PA_X, \epsilon_X)$$

**干预后** $do(X=x)$:
$$X = x \quad \text{(删除所有指向X的边)}$$

这被称为**图手术**（Graph Surgery）。

#### do-演算三大规则

**规则1: 插入/删除观测**
$$P(y | do(x), z, w) = P(y | do(x), w) \quad \text{if} \quad (Y \perp Z \,|\, X, W)_{G_{\overline{X}}}$$

**规则2: 干预替换观测（行动/观察交换）**
$$P(y | do(x), do(z), w) = P(y | do(x), z, w) \quad \text{if} \quad (Y \perp Z \,|\, X, W)_{G_{\overline{X}, \underline{Z}}}$$

**规则3: 删除干预**
$$P(y | do(x), do(z), w) = P(y | do(x), w) \quad \text{if} \quad (Y \perp Z \,|\, X, W)_{G_{\overline{X}, \overline{Z(W)}}}$$

其中$G_{\overline{X}}$表示删除指向$X$的边后的图。

### 4.7 后门准则（Back-Door Criterion）

**问题**: 我们想估计$P(Y|do(X))$，但只有观测数据$P(Y,X,Z,...)$。如何转换？

**后门路径**: 从$X$到$Y$且以指向$X$的箭头开始的路径（$X \leftarrow ... \rightarrow Y$）

**后门准则**:
> 变量集合$Z$满足相对于$(X, Y)$的后门准则，如果：
> 1. $Z$不包含$X$的后代
> 2. $Z$阻断了$X$和$Y$之间的所有后门路径

**后门调整公式**:
$$P(y | do(x)) = \sum_{z} P(y | x, z) P(z)$$

### 4.8 数学推导：后门调整公式的证明

**目标**: 证明$P(Y|do(X)) = \sum_z P(Y|X, z)P(z)$

**证明步骤**:

1. **do-操作的概率解释**: 在图$G$中$do(X=x)$等价于在修改后的图$G_{\overline{X}}$（删除所有指向$X$的边）中观测$X=x$。

2. **修改后的DAG**: 在$G_{\overline{X}}$中，$X$没有父节点，因此$X$与其前序变量独立。

3. **应用d-分离**: 如果$Z$满足后门准则，则在$G_{\overline{X}}$中，$X$和$Y$之间的所有路径要么：
   - 被$Z$阻断（后门路径）
   - 是因果路径$X \rightarrow ... \rightarrow Y$

4. **条件独立**: 在$G_{\overline{X}}$中，给定$X$和$Z$，$Y$的条件分布与原始图中相同。

5. **推导**:
   $$\begin{aligned}
   P(y | do(x)) &= P_{G_{\overline{X}}}(y | x) \\
   &= \sum_z P_{G_{\overline{X}}}(y | x, z) P_{G_{\overline{X}}}(z | x) \\
   &= \sum_z P_{G_{\overline{X}}}(y | x, z) P_{G_{\overline{X}}}(z) \quad \text{(X与Z在}G_{\overline{X}}\text{中独立)} \\
   &= \sum_z P(y | x, z) P(z) \quad \text{(后门路径被阻断)}
   \end{aligned}$$

**证毕**。

### 4.9 前门准则（Front-Door Criterion）

当无法使用后门准则时（如存在不可观测的混杂因素），前门准则提供了另一种识别策略。

**前门准则**:
> 变量集合$Z$满足相对于$(X, Y)$的前门准则，如果：
> 1. $Z$截断了所有从$X$到$Y$的**有向路径**
> 2. $X$到$Z$没有后门路径（即没有混杂）
> 3. $Z$到$Y$的所有后门路径都被$X$阻断

**前门调整公式**:
$$P(y | do(x)) = \sum_z P(z | x) \sum_{x'} P(y | x', z) P(x')$$

**直观理解**:
1. 第一阶段: $X \rightarrow Z$的效应是干净的（无混杂）
2. 第二阶段: $Z \rightarrow Y$可以通过后门调整（控制$X$）
3. 链式法则: 总效应 = 第一阶段效应 × 第二阶段效应

### 4.10 代码实现：结构因果模型

```python
"""
结构因果模型（Pearl框架）实现
包含有向无环图(DAG)、d-分离检测、后门准则、前门准则
"""

import numpy as np
from typing import Set, List, Tuple, Dict, Optional
from collections import defaultdict, deque


class CausalGraph:
    """
    因果图（有向无环图 DAG）实现
    
    支持基本图操作、d-分离检测、路径查找等
    """
    
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[str, Set[str]] = defaultdict(set)  # parent -> children
        self.parents: Dict[str, Set[str]] = defaultdict(set)  # child -> parents
        
    def add_node(self, node: str):
        """添加节点"""
        self.nodes.add(node)
        
    def add_edge(self, parent: str, child: str):
        """添加有向边 parent -> child"""
        self.add_node(parent)
        self.add_node(child)
        self.edges[parent].add(child)
        self.parents[child].add(parent)
        
        # 检查环
        if self.has_cycle():
            self.edges[parent].remove(child)
            self.parents[child].remove(parent)
            raise ValueError(f"添加边 {parent}->{child} 会产生循环!")
    
    def has_cycle(self) -> bool:
        """检测图中是否存在有向环"""
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.edges[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.nodes:
            if node not in visited:
                if dfs(node):
                    return True
        return False
    
    def get_ancestors(self, node: str) -> Set[str]:
        """获取节点的所有祖先"""
        ancestors = set()
        queue = deque([node])
        visited = {node}
        
        while queue:
            current = queue.popleft()
            for parent in self.parents[current]:
                if parent not in visited:
                    visited.add(parent)
                    ancestors.add(parent)
                    queue.append(parent)
        
        return ancestors
    
    def get_descendants(self, node: str) -> Set[str]:
        """获取节点的所有后代"""
        descendants = set()
        queue = deque([node])
        visited = {node}
        
        while queue:
            current = queue.popleft()
            for child in self.edges[current]:
                if child not in visited:
                    visited.add(child)
                    descendants.add(child)
                    queue.append(child)
        
        return descendants
    
    def get_all_paths(self, start: str, end: str) -> List[List[str]]:
        """
        使用DFS查找从start到end的所有无环路径
        """
        paths = []
        
        def dfs(current, path, visited):
            if current == end and len(path) > 1:
                paths.append(path[:])
                return
            
            for neighbor in self.edges[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        visited = {start}
        dfs(start, [start], visited)
        return paths
    
    def is_backdoor_path(self, path: List[str], X: str, Y: str) -> bool:
        """
        判断路径是否为后门路径
        后门路径: 以指向X的边开始的路径 (X <- ...)
        """
        if len(path) < 2 or path[0] != X or path[-1] != Y:
            return False
        # 检查第一条边是否指向X
        return path[1] in self.parents[X]
    
    def find_backdoor_paths(self, X: str, Y: str) -> List[List[str]]:
        """查找所有从X到Y的后门路径"""
        # 需要找以X <- 开始的路径
        # 这需要反向遍历
        all_paths = []
        
        # 从X的父节点开始找
        for parent in self.parents[X]:
            paths_from_parent = []
            
            def dfs(current, path, visited):
                if current == Y:
                    paths_from_parent.append([X] + path[:])
                    return
                
                # 遍历所有邻居（包括父节点和子节点）
                neighbors = self.edges[current] | self.parents[current]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs(neighbor, path, visited)
                        path.pop()
                        visited.remove(neighbor)
            
            visited = {X, parent}
            dfs(parent, [parent], visited)
            all_paths.extend(paths_from_parent)
        
        return all_paths
    
    def is_d_separated(self, X: Set[str], Y: Set[str], Z: Set[str]) -> bool:
        """
        检查X和Y是否被Z d-分离
        
        算法: 道德图法或活跃路径法
        这里实现活跃路径检测法
        """
        # 简化: 检查是否存在从X到Y的活跃路径
        for x in X:
            for y in Y:
                if self._has_active_path(x, y, Z):
                    return False
        return True
    
    def _has_active_path(self, start: str, end: str, Z: Set[str]) -> bool:
        """检查是否存在从start到end的活跃路径（给定Z）"""
        # 使用BFS，需要记录进入节点的方向
        # (node, came_from_parent) 表示从哪个方向到达该节点
        queue = deque()
        visited = set()
        
        # 初始化: 可以从父节点或子节点方向开始
        for parent in self.parents[start]:
            queue.append((parent, True))  # 从父节点来
        for child in self.edges[start]:
            queue.append((child, False))  # 从子节点来
        
        while queue:
            node, from_parent = queue.popleft()
            
            if node == end:
                return True
            
            state = (node, from_parent)
            if state in visited:
                continue
            visited.add(state)
            
            # 链结构: A -> B -> C 或 A <- B <- C
            # 分叉: A <- B -> C
            # 对撞: A -> B <- C
            
            if from_parent:
                # 我们是从父节点方向到达的 (即从A到B，A是B的父节点)
                # 链: A -> B -> C，B不在Z中，可以继续到子节点C
                if node not in Z:
                    for child in self.edges[node]:
                        queue.append((child, False))
                
                # 分叉: A -> B <- C，如果B在Z中，可以走另一个父节点
                if node in Z:
                    for parent in self.parents[node]:
                        queue.append((parent, True))
            else:
                # 我们是从子节点方向到达的 (即从C到B，C是B的子节点)
                # 链: A <- B <- C，B不在Z中，可以继续到父节点A
                if node not in Z:
                    for parent in self.parents[node]:
                        queue.append((parent, True))
                
                # 对撞: A -> B <- C，如果B或其后代在Z中，可以继续
                # 简化: 检查B是否在Z中
                if node in Z:
                    for child in self.edges[node]:
                        queue.append((child, False))
                    for parent in self.parents[node]:
                        queue.append((parent, True))
        
        return False
    
    def satisfies_backdoor_criterion(self, X: str, Y: str, Z: Set[str]) -> bool:
        """
        检查Z是否满足相对于(X, Y)的后门准则
        
        条件:
        1. Z不包含X的后代
        2. Z阻断所有从X到Y的后门路径
        """
        # 条件1: Z不包含X的后代
        descendants = self.get_descendants(X)
        if Z & descendants:
            return False
        
        # 条件2: Z阻断所有后门路径
        backdoor_paths = self.find_backdoor_paths(X, Y)
        
        for path in backdoor_paths:
            # 检查路径是否被Z阻断
            # 路径被阻断如果存在链或分叉中的节点在Z中
            # 或对撞节点及其后代不在Z中
            blocked = False
            
            for i in range(1, len(path) - 1):
                node = path[i]
                prev_node = path[i-1]
                next_node = path[i+1]
                
                # 判断节点类型
                is_collider = (node in self.edges[prev_node]) and (node in self.edges[next_node])
                is_chain_or_fork = not is_collider
                
                if is_chain_or_fork and node in Z:
                    blocked = True
                    break
                
                # 对撞节点: 如果在Z中，路径反而被"打开"
                # 这里简化处理
            
            if not blocked:
                return False
        
        return True
    
    def get_backdoor_adjustment_set(self, X: str, Y: str) -> Optional[Set[str]]:
        """
        找到一个满足后门准则的调整集合
        
        返回一个最小调整集合，或None如果不存在
        """
        # 简单的贪心算法: 尝试所有X的非后代（除Y外）的子集
        candidates = self.nodes - {X, Y} - self.get_descendants(X)
        
        # 先尝试空集
        if self.satisfies_backdoor_criterion(X, Y, set()):
            return set()
        
        # 尝试单个变量
        for node in candidates:
            if self.satisfies_backdoor_criterion(X, Y, {node}):
                return {node}
        
        # 尝试两个变量
        candidates_list = list(candidates)
        for i in range(len(candidates_list)):
            for j in range(i+1, len(candidates_list)):
                Z = {candidates_list[i], candidates_list[j]}
                if self.satisfies_backdoor_criterion(X, Y, Z):
                    return Z
        
        return None
    
    def print_graph(self):
        """打印图的邻接表表示"""
        print("因果图结构:")
        for node in sorted(self.nodes):
            children = sorted(self.edges[node])
            if children:
                print(f"  {node} -> {', '.join(children)}")
            else:
                print(f"  {node} (无出边)")


class StructuralCausalModel:
    """
    结构因果模型实现
    
    基于结构方程模型(SEM)进行因果推理
    """
    
    def __init__(self, graph: CausalGraph):
        self.graph = graph
        self.structural_equations: Dict[str, callable] = {}
        self.noise_distributions: Dict[str, callable] = {}
        
    def add_equation(self, node: str, equation: callable, noise_sampler: callable):
        """
        添加结构方程
        
        Args:
            node: 结果变量
            equation: 结构方程 f(PA, epsilon)
            noise_sampler: 噪声分布采样函数
        """
        self.structural_equations[node] = equation
        self.noise_distributions[node] = noise_sampler
    
    def simulate(self, n_samples: int, intervention: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """
        模拟数据生成
        
        Args:
            n_samples: 样本数量
            intervention: 干预设置 {node: value}
            
        Returns:
            各变量的观测值
        """
        data = {}
        
        # 拓扑排序确定计算顺序
        topo_order = self._topological_sort()
        
        for node in topo_order:
            if intervention and node in intervention:
                # 干预: 固定值
                data[node] = np.full(n_samples, intervention[node])
            elif node in self.structural_equations:
                # 正常生成
                parents = self.graph.parents[node]
                parent_values = [data[p] for p in parents]
                noise = self.noise_distributions[node](n_samples)
                
                data[node] = self.structural_equations[node](parent_values, noise)
            else:
                # 外生变量
                if node not in data:
                    data[node] = self.noise_distributions.get(node, lambda n: np.random.normal(0, 1, n))(n_samples)
        
        return data
    
    def do_calculus_backdoor(self, X: str, Y: str, data: Dict[str, np.ndarray],
                            adjustment_set: Optional[Set[str]] = None) -> Tuple[float, float]:
        """
        使用后门调整计算P(Y|do(X))
        
        公式: P(Y|do(X=x)) = Σ_z P(Y|X=x, Z=z) P(Z=z)
        
        Returns:
            (do(X=0)时的Y均值, do(X=1)时的Y均值) 或连续值的ATE
        """
        if adjustment_set is None:
            adjustment_set = self.graph.get_backdoor_adjustment_set(X, Y) or set()
        
        Y_data = data[Y]
        X_data = data[X]
        
        if not adjustment_set:
            # 无调整: 直接比较
            y_given_x1 = Y_data[X_data == 1].mean() if np.any(X_data == 1) else 0
            y_given_x0 = Y_data[X_data == 0].mean() if np.any(X_data == 0) else 0
            return y_given_x0, y_given_x1
        
        # 有调整集合
        adjustment_vars = list(adjustment_set)
        
        # 构建调整变量的组合
        Z_data = np.column_stack([data[z] for z in adjustment_vars])
        
        # 离散化用于分层
        if Z_data.shape[1] == 1:
            Z_bins = np.digitize(Z_data[:, 0], np.percentile(Z_data[:, 0], [25, 50, 75]))
        else:
            # 使用第一主维度的分位数
            Z_bins = np.digitize(Z_data[:, 0], np.percentile(Z_data[:, 0], [25, 50, 75]))
        
        # 分层估计
        unique_bins = np.unique(Z_bins)
        effect_x1 = []
        effect_x0 = []
        weights = []
        
        for bin_id in unique_bins:
            mask = Z_bins == bin_id
            bin_weight = mask.mean()
            
            X_bin = X_data[mask]
            Y_bin = Y_data[mask]
            
            if np.sum(X_bin == 1) > 0 and np.sum(X_bin == 0) > 0:
                y_x1 = Y_bin[X_bin == 1].mean()
                y_x0 = Y_bin[X_bin == 0].mean()
                
                effect_x1.append(y_x1)
                effect_x0.append(y_x0)
                weights.append(bin_weight)
        
        y_do_x1 = np.average(effect_x1, weights=weights)
        y_do_x0 = np.average(effect_x0, weights=weights)
        
        return y_do_x0, y_do_x1
    
    def _topological_sort(self) -> List[str]:
        """拓扑排序"""
        in_degree = {node: len(self.graph.parents[node]) for node in self.graph.nodes}
        queue = deque([node for node in self.graph.nodes if in_degree[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for child in self.graph.edges[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        if len(result) != len(self.graph.nodes):
            raise ValueError("图中存在环，无法进行拓扑排序")
        
        return result


# 演示代码
if __name__ == "__main__":
    print("=" * 70)
    print("结构因果模型演示")
    print("=" * 70)
    
    # 示例1: 创建因果图
    print("\n【示例1】创建因果图")
    g = CausalGraph()
    
    # 吸烟与肺癌的经典例子
    # Z: 基因型 (混杂因素)
    # X: 吸烟
    # Y: 肺癌
    # 结构: Z -> X, Z -> Y, X -> Y
    
    g.add_edge("基因型(Z)", "吸烟(X)")
    g.add_edge("基因型(Z)", "肺癌(Y)")
    g.add_edge("吸烟(X)", "肺癌(Y)")
    
    g.print_graph()
    
    # 检查后门路径
    print("\n从'吸烟(X)'到'肺癌(Y)'的后门路径:")
    backdoor_paths = g.find_backdoor_paths("吸烟(X)", "肺癌(Y)")
    for path in backdoor_paths:
        print(f"  {' -> '.join(path)}")
    
    # 检查后门准则
    Z = {"基因型(Z)"}
    satisfies = g.satisfies_backdoor_criterion("吸烟(X)", "肺癌(Y)", Z)
    print(f"\n调整集合 {Z} 满足后门准则: {satisfies}")
    
    # 自动寻找调整集合
    adjustment_set = g.get_backdoor_adjustment_set("吸烟(X)", "肺癌(Y)")
    print(f"推荐调整集合: {adjustment_set}")
    
    # 示例2: 结构因果模型模拟
    print("\n" + "=" * 70)
    print("【示例2】结构因果模型模拟")
    print("=" * 70)
    
    scm = StructuralCausalModel(g)
    
    # 定义结构方程
    # Z = epsilon_Z
    # X = 1 if (0.5*Z + epsilon_X) > 0 else 0
    # Y = 2*X + 3*Z + epsilon_Y
    
    scm.add_equation("基因型(Z)", 
                     lambda parents, noise: noise,
                     lambda n: np.random.normal(0, 1, n))
    
    def x_equation(parents, noise):
        z = parents[0]
        return ((0.5 * z + noise) > 0).astype(int)
    
    scm.add_equation("吸烟(X)",
                     x_equation,
                     lambda n: np.random.normal(0, 1, n))
    
    def y_equation(parents, noise):
        x, z = parents
        return 2 * x + 3 * z + noise
    
    scm.add_equation("肺癌(Y)",
                     y_equation,
                     lambda n: np.random.normal(0, 1, n))
    
    # 观测数据模拟
    print("\n1. 模拟观测数据:")
    obs_data = scm.simulate(n_samples=10000)
    
    # 朴素估计 (有偏)
    x = obs_data["吸烟(X)"]
    y = obs_data["肺癌(Y)"]
    naive_effect = y[x == 1].mean() - y[x == 0].mean()
    print(f"   朴素估计 (E[Y|X=1] - E[Y|X=0]): {naive_effect:.3f}")
    print(f"   ⚠️ 这个估计被基因型混杂了！")
    
    # 使用后门调整
    print("\n2. 使用后门调整计算因果效应:")
    y_do_x0, y_do_x1 = scm.do_calculus_backdoor("吸烟(X)", "肺癌(Y)", 
                                                  obs_data, {"基因型(Z)"})
    causal_effect = y_do_x1 - y_do_x0
    print(f"   P(Y|do(X=1)): {y_do_x1:.3f}")
    print(f"   P(Y|do(X=0)): {y_do_x0:.3f}")
    print(f"   因果效应 (ATE): {causal_effect:.3f}")
    print(f"   ✓ 真实因果效应是 2.0 (由模型结构决定)")
    
    # 干预模拟
    print("\n3. 直接干预模拟 (do(X=1) vs do(X=0)):")
    intv_data_1 = scm.simulate(n_samples=10000, intervention={"吸烟(X)": 1})
    intv_data_0 = scm.simulate(n_samples=10000, intervention={"吸烟(X)": 0})
    
    true_effect = intv_data_1["肺癌(Y)"].mean() - intv_data_0["肺癌(Y)"].mean()
    print(f"   干预模拟的因果效应: {true_effect:.3f}")
    print(f"   ✓ 这与后门调整结果一致！")


## 5. 因果效应估计方法

### 5.1 双重差分法（Difference-in-Differences, DID）

**适用场景**: 当存在**面板数据**（同一单元在不同时间点的观测）且有**政策冲击**时。

**核心思想**: 通过比较处理组和对照组在政策前后的变化差异，来估计因果效应。

**基本假设**: 平行趋势假设（Parallel Trends）
> 如果没有处理，处理组和对照组的结果变量会以相同的趋势变化。

**数学公式**:
$$\tau_{DID} = (\bar{Y}_{treatment, post} - \bar{Y}_{treatment, pre}) - (\bar{Y}_{control, post} - \bar{Y}_{control, pre})$$

**回归形式**:
$$Y_{it} = \alpha + \beta T_i + \gamma Post_t + \tau (T_i \times Post_t) + \epsilon_{it}$$

其中：
- $T_i$: 处理指示变量
- $Post_t$: 时间指示变量（政策后=1）
- $\tau$: 我们关心的处理效应（DID估计量）

```python
class DifferenceInDifferences:
    """
    双重差分法(DID)实现
    """
    
    def __init__(self):
        self.tau_ = None
        self.se_ = None
        
    def fit(self, Y: np.ndarray, T: np.ndarray, Post: np.ndarray) -> 'DifferenceInDifferences':
        """
        估计DID
        
        Args:
            Y: 结果变量
            T: 处理组指示 (0=对照组, 1=处理组)
            Post: 政策后指示 (0=政策前, 1=政策后)
        """
        # 2×2 单元格均值
        y_control_pre = Y[(T == 0) & (Post == 0)].mean()
        y_control_post = Y[(T == 0) & (Post == 1)].mean()
        y_treat_pre = Y[(T == 1) & (Post == 0)].mean()
        y_treat_post = Y[(T == 1) & (Post == 1)].mean()
        
        # DID估计量
        self.tau_ = (y_treat_post - y_treat_pre) - (y_control_post - y_control_pre)
        
        print("=== 双重差分估计 ===")
        print(f"对照组: 政策前={y_control_pre:.3f}, 政策后={y_control_post:.3f}, "
              f"变化={y_control_post - y_control_pre:.3f}")
        print(f"处理组: 政策前={y_treat_pre:.3f}, 政策后={y_treat_post:.3f}, "
              f"变化={y_treat_post - y_treat_pre:.3f}")
        print(f"\nDID估计量 (处理效应): {self.tau_:.3f}")
        
        return self
```

### 5.2 双重稳健估计（Doubly Robust Estimation）

**问题**: IPW要求倾向得分模型正确，回归要求结果模型正确。

**双重稳健估计**: 只要其中一个模型正确，估计就是一致的！

**估计量**:
$$\hat{\tau}_{DR} = \frac{1}{n} \sum_{i=1}^n \left[ \frac{T_i Y_i}{\hat{e}(X_i)} - \frac{T_i - \hat{e}(X_i)}{\hat{e}(X_i)} \hat{\mu}_1(X_i) \right] - \frac{1}{n} \sum_{i=1}^n \left[ \frac{(1-T_i)Y_i}{1-\hat{e}(X_i)} + \frac{T_i - \hat{e}(X_i)}{1-\hat{e}(X_i)} \hat{\mu}_0(X_i) \right]$$

其中：
- $\hat{e}(X)$: 估计的倾向得分
- $\hat{\mu}_1(X)$: 处理组结果回归模型
- $\hat{\mu}_0(X)$: 对照组结果回归模型

```python
class DoublyRobustEstimator:
    """
    双重稳健估计器
    
    结合了倾向得分和结果回归，只要一个正确就能保持一致性
    """
    
    def __init__(self, ps_model=None, outcome_model=None):
        self.ps_model = ps_model or LogisticRegression(max_iter=1000)
        self.outcome_model_treated = outcome_model or None
        self.outcome_model_control = outcome_model or None
        self.tau_ = None
        
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """估计因果效应"""
        # 1. 估计倾向得分
        self.ps_model.fit(X, T)
        ps = self.ps_model.predict_proba(X)[:, 1]
        ps = np.clip(ps, 0.05, 0.95)  # 避免极端值
        
        # 2. 估计结果模型
        from sklearn.linear_model import LinearRegression
        
        self.outcome_model_treated = LinearRegression()
        self.outcome_model_control = LinearRegression()
        
        if np.sum(T == 1) > 0:
            self.outcome_model_treated.fit(X[T == 1], Y[T == 1])
        if np.sum(T == 0) > 0:
            self.outcome_model_control.fit(X[T == 0], Y[T == 0])
        
        # 3. 双重稳健估计
        mu1_hat = self.outcome_model_treated.predict(X)
        mu0_hat = self.outcome_model_control.predict(X)
        
        # AIPW (Augmented IPW) 估计量
        term1 = T * Y / ps - (T - ps) / ps * mu1_hat
        term0 = (1 - T) * Y / (1 - ps) + (T - ps) / (1 - ps) * mu0_hat
        
        self.tau_ = np.mean(term1 - term0)
        
        print("=== 双重稳健估计 ===")
        print(f"DR估计的ATE: {self.tau_:.3f}")
        print("✓ 只要倾向得分模型或结果模型之一正确，估计就是一致的")
        
        return self
```

### 5.3 因果森林（Causal Forest）

**问题**: ATE假设处理效应对所有人都一样，但现实中可能存在异质性。

**CATE估计**: 我们希望估计条件平均因果效应 $\tau(x) = E[Y(1) - Y(0) | X=x]$

**因果森林**: 基于随机森林的CATE估计方法（Athey & Imbens, 2016）

**核心思想**: 
1. 使用随机森林找到相似的单元（叶子节点）
2. 在每个叶子节点内，估计局部ATE
3. 通过"诚实估计"避免过拟合

```python
class SimpleCausalForest:
    """
    简化的因果森林实现
    
    基于递归分治的CATE估计
    """
    
    def __init__(self, n_trees=100, max_depth=5, min_samples_leaf=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        
    def _find_best_split(self, X, T, Y):
        """寻找最佳分裂点（最大化处理效应异质性）"""
        best_gain = -np.inf
        best_split = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            values = np.unique(X[:, feature])
            
            for val in values:
                left = X[:, feature] <= val
                right = ~left
                
                if np.sum(left) < self.min_samples_leaf or np.sum(right) < self.min_samples_leaf:
                    continue
                
                # 计算两边的处理效应
                if np.sum(T[left] == 1) > 0 and np.sum(T[left] == 0) > 0:
                    tau_left = Y[left][T[left] == 1].mean() - Y[left][T[left] == 0].mean()
                else:
                    continue
                    
                if np.sum(T[right] == 1) > 0 and np.sum(T[right] == 0) > 0:
                    tau_right = Y[right][T[right] == 1].mean() - Y[right][T[right] == 0].mean()
                else:
                    continue
                
                # 增益 = 方差减少
                gain = abs(tau_left - tau_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, val, tau_left, tau_right)
        
        return best_split
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """拟合因果森林"""
        self.trees = []
        
        for i in range(self.n_trees):
            # Bootstrap采样
            idx = np.random.choice(len(Y), size=len(Y), replace=True)
            X_boot, T_boot, Y_boot = X[idx], T[idx], Y[idx]
            
            # 递归构建树
            tree = self._build_tree(X_boot, T_boot, Y_boot, depth=0)
            self.trees.append(tree)
            
        return self
    
    def _build_tree(self, X, T, Y, depth):
        """递归构建决策树"""
        # 停止条件
        if depth >= self.max_depth or len(Y) < 2 * self.min_samples_leaf:
            # 叶子节点: 返回处理效应
            if np.sum(T == 1) > 0 and np.sum(T == 0) > 0:
                tau = Y[T == 1].mean() - Y[T == 0].mean()
            else:
                tau = 0
            return {'leaf': True, 'tau': tau}
        
        # 寻找最佳分裂
        split = self._find_best_split(X, T, Y)
        
        if split is None:
            if np.sum(T == 1) > 0 and np.sum(T == 0) > 0:
                tau = Y[T == 1].mean() - Y[T == 0].mean()
            else:
                tau = 0
            return {'leaf': True, 'tau': tau}
        
        feature, val, tau_left, tau_right = split
        
        # 分裂数据
        left_mask = X[:, feature] <= val
        
        return {
            'leaf': False,
            'feature': feature,
            'value': val,
            'left': self._build_tree(X[left_mask], T[left_mask], Y[left_mask], depth + 1),
            'right': self._build_tree(X[~left_mask], T[~left_mask], Y[~left_mask], depth + 1)
        }
    
    def _predict_single(self, x, tree):
        """单样本预测"""
        if tree['leaf']:
            return tree['tau']
        
        if x[tree['feature']] <= tree['value']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测CATE"""
        predictions = np.zeros(len(X))
        
        for i, x in enumerate(X):
            # 平均所有树的预测
            tree_preds = [self._predict_single(x, tree) for tree in self.trees]
            predictions[i] = np.mean(tree_preds)
        
        return predictions
```

---

## 6. 实际应用案例：药物疗效评估

### 6.1 问题背景

某医院想要评估一种新药对高血压患者的疗效。由于伦理原因，不能进行随机对照试验（RCT），只能基于历史电子病历数据进行分析。

**挑战**:
- 医生选择性地给更严重的患者开新药（选择偏差）
- 患者的年龄、性别、基础疾病等都会影响治疗决策和结果
- 需要控制混杂因素才能得出因果结论

### 6.2 完整分析Pipeline

```python
"""
药物疗效因果分析完整案例
评估新药对高血压患者收缩压的影响
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DrugEfficacyAnalysis:
    """
    药物疗效因果分析完整案例
    """
    
    def __init__(self):
        self.data = None
        self.results = {}
        
    def generate_synthetic_data(self, n_samples=5000, seed=42):
        """
        生成模拟的临床数据
        
        数据生成机制:
        - 混杂因素: 疾病严重程度、年龄、并发症
        - 处理分配: 严重患者更可能接受新药
        - 结果: 血压 = 基线 + 年龄效应 + 疾病效应 + 治疗效应 + 噪声
        """
        np.random.seed(seed)
        
        # 协变量 (混杂因素)
        age = np.random.normal(60, 10, n_samples)  # 年龄
        disease_severity = np.random.exponential(2, n_samples)  # 疾病严重程度
        has_complications = (np.random.random(n_samples) < 0.3).astype(int)  # 并发症
        
        # 基线血压 (受混杂因素影响)
        baseline_bp = 140 + 0.5 * age + 5 * disease_severity + 10 * has_complications
        baseline_bp += np.random.normal(0, 5, n_samples)
        
        # 处理分配 (受混杂因素影响 - 选择偏差!)
        # 更严重患者更可能接受新药
        prob_new_drug = 1 / (1 + np.exp(-(disease_severity - 2) / 1.5))
        new_drug = (np.random.random(n_samples) < prob_new_drug).astype(int)
        
        # 真实潜在结果
        # 真实因果效应: 新药使血压平均降低10 mmHg，但严重患者效果稍差
        treatment_effect = -10 + 0.5 * disease_severity + np.random.normal(0, 3, n_samples)
        
        Y0 = baseline_bp + np.random.normal(0, 5, n_samples)  # 未接受治疗
        Y1 = baseline_bp + treatment_effect + np.random.normal(0, 5, n_samples)  # 接受治疗
        
        # 观察到的结果
        blood_pressure = np.where(new_drug == 1, Y1, Y0)
        
        # 构建DataFrame
        self.data = pd.DataFrame({
            'patient_id': range(n_samples),
            'age': age,
            'disease_severity': disease_severity,
            'has_complications': has_complications,
            'baseline_bp': baseline_bp,
            'new_drug': new_drug,
            'blood_pressure': blood_pressure,
            'Y0': Y0,  # 反事实 (仅用于验证)
            'Y1': Y1,  # 反事实 (仅用于验证)
            'true_effect': Y1 - Y0  # 真实个体效应 (仅用于验证)
        })
        
        print("=" * 70)
        print("药物疗效因果分析 - 数据生成")
        print("=" * 70)
        print(f"样本量: {n_samples}")
        print(f"新药使用率: {new_drug.mean():.1%}")
        print(f"真实平均因果效应 (ATE): {treatment_effect.mean():.2f} mmHg")
        print(f"  (负值表示血压降低，即药物有效)")
        
        return self
    
    def naive_analysis(self):
        """朴素分析 (有偏)"""
        print("\n" + "=" * 70)
        print("【步骤1】朴素分析 (忽略混杂因素)")
        print("=" * 70)
        
        drug_group = self.data[self.data['new_drug'] == 1]['blood_pressure']
        control_group = self.data[self.data['new_drug'] == 0]['blood_pressure']
        
        naive_effect = drug_group.mean() - control_group.mean()
        
        print(f"新药组平均血压: {drug_group.mean():.2f} mmHg")
        print(f"对照组平均血压: {control_group.mean():.2f} mmHg")
        print(f"朴素估计效应: {naive_effect:.2f} mmHg")
        print(f"\n⚠️ 警告: 此估计受选择偏差影响!")
        print(f"  医生倾向给严重患者开新药，而这些患者血压本来就高")
        
        self.results['naive'] = naive_effect
        return naive_effect
    
    def propensity_score_analysis(self):
        """倾向得分匹配分析"""
        print("\n" + "=" * 70)
        print("【步骤2】倾向得分匹配分析")
        print("=" * 70)
        
        # 准备协变量
        X = self.data[['age', 'disease_severity', 'has_complications', 'baseline_bp']].values
        T = self.data['new_drug'].values
        Y = self.data['blood_pressure'].values
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 倾向得分匹配
        psm = PropensityScoreMatcher(caliper=0.2)
        psm.fit(X_scaled, T)
        matched_pairs = psm.match(T)
        psm_ate = psm.estimate_ate(Y, matched_pairs)
        
        self.results['psm'] = psm_ate
        return psm_ate
    
    def ipw_analysis(self):
        """逆概率加权分析"""
        print("\n" + "=" * 70)
        print("【步骤3】逆概率加权(IPW)分析")
        print("=" * 70)
        
        X = self.data[['age', 'disease_severity', 'has_complications', 'baseline_bp']].values
        T = self.data['new_drug'].values
        Y = self.data['blood_pressure'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        ipw = IPWEstimator()
        ipw.fit(X_scaled, T)
        ipw_ate = ipw.estimate_ate(Y, T)
        
        self.results['ipw'] = ipw_ate
        return ipw_ate
    
    def doubly_robust_analysis(self):
        """双重稳健分析"""
        print("\n" + "=" * 70)
        print("【步骤4】双重稳健估计")
        print("=" * 70)
        
        X = self.data[['age', 'disease_severity', 'has_complications', 'baseline_bp']].values
        T = self.data['new_drug'].values
        Y = self.data['blood_pressure'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        dr = DoublyRobustEstimator()
        dr.fit(X_scaled, T, Y)
        dr_ate = dr.tau_
        
        self.results['doubly_robust'] = dr_ate
        return dr_ate
    
    def causal_forest_analysis(self):
        """因果森林分析 (异质性效应)"""
        print("\n" + "=" * 70)
        print("【步骤5】因果森林分析 (CATE估计)")
        print("=" * 70)
        
        X = self.data[['age', 'disease_severity', 'has_complications']].values
        T = self.data['new_drug'].values
        Y = self.data['blood_pressure'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 拟合因果森林
        cf = SimpleCausalForest(n_trees=50, max_depth=4)
        cf.fit(X_scaled, T, Y)
        
        # 预测CATE
        cate_preds = cf.predict(X_scaled)
        self.data['cate_pred'] = cate_preds
        
        print(f"平均CATE估计: {cate_preds.mean():.2f} mmHg")
        print(f"\nCATE分布:")
        print(f"  25%分位数: {np.percentile(cate_preds, 25):.2f}")
        print(f"  中位数: {np.median(cate_preds):.2f}")
        print(f"  75%分位数: {np.percentile(cate_preds, 75):.2f}")
        
        print(f"\n按疾病严重程度分组的平均效应:")
        severity_groups = pd.cut(self.data['disease_severity'], bins=3, labels=['轻度', '中度', '重度'])
        for group in ['轻度', '中度', '重度']:
            mask = severity_groups == group
            if mask.sum() > 0:
                avg_effect = cate_preds[mask].mean()
                print(f"  {group}患者: {avg_effect:.2f} mmHg")
        
        self.results['causal_forest_mean'] = cate_preds.mean()
        return cate_preds
    
    def summary(self):
        """结果汇总"""
        print("\n" + "=" * 70)
        print("分析结果汇总")
        print("=" * 70)
        
        true_ate = self.data['true_effect'].mean()
        
        print(f"{'方法':<25} {'估计效应':>12} {'偏差':>10}")
        print("-" * 70)
        print(f"{'真实因果效应(基准)':<25} {true_ate:>12.2f} {'-':>10}")
        
        for method, estimate in self.results.items():
            bias = estimate - true_ate
            print(f"{method:<25} {estimate:>12.2f} {bias:>10.2f}")
        
        print("-" * 70)
        print("\n结论:")
        print("  1. 朴素估计存在严重偏差（选择偏差）")
        print("  2. 各种因果推断方法都能较好地纠正偏差")
        print("  3. 双重稳健估计通常最稳定")
        print("  4. 因果森林揭示了处理效应的异质性")


# 运行完整分析
if __name__ == "__main__":
    analysis = DrugEfficacyAnalysis()
    analysis.generate_synthetic_data(n_samples=3000)
    analysis.naive_analysis()
    analysis.propensity_score_analysis()
    analysis.ipw_analysis()
    analysis.doubly_robust_analysis()
    analysis.causal_forest_analysis()
    analysis.summary()
```

---

## 7. 本章总结与知识图谱

### 7.1 核心概念回顾

| 概念 | 解释 | 类比 |
|-----|-----|-----|
| **因果 vs 相关** | 因果是"改变X会改变Y"，相关只是共现 | 开关与影子 |
| **潜在结果** | 每个单元在两种处理状态下的可能结果 | 平行宇宙的两条路 |
| **混杂因素** | 同时影响处理和结果的第三方变量 | 隐藏的幕后推手 |
| **do-演算** | 从观测到干预的数学转换 | 上帝之手干预 |
| **反事实** | 对过去不同选择的想象 | 时光机假设 |

### 7.2 两大框架对比

| 特性 | Rubin潜在结果框架 | Pearl结构因果模型 |
|-----|------------------|------------------|
| **基本单元** | 潜在结果对 | 结构方程 |
| **表示方式** | 符号系统 $(Y(0), Y(1))$ | 有向图 + 方程 |
| **主要假设** | SUTVA, 可忽略性 | 因果图结构正确 |
| **识别策略** | 调整混杂变量 | do-演算, 后门/前门准则 |
| **优势** | 概念简单，适合复杂设计 | 可视化，支持反事实 |

### 7.3 方法选择指南

**根据数据类型选择**:
- **横截面数据**: 倾向得分匹配、IPW、双重稳健
- **面板数据**: 双重差分 (DID)、合成控制法
- **时间序列**: 中断时间序列分析
- **复杂异质性**: 因果森林、元学习器

---

## 8. 练习题

### 基础概念题

**练习8.1** 解释以下概念的区别：
- (a) $P(Y|X)$ 和 $P(Y|do(X))$
- (b) ATE 和 CATE
- (c) 混杂因素和对撞变量

**练习8.2** 考虑以下因果图: $Z \rightarrow X \rightarrow Y \leftarrow W$，且$Z$和$W$无关。
- (a) 列出所有d-连接的路径
- (b) 控制$X$会阻断哪些路径？
- (c) $Z$和$W$是否独立？条件于$Y$后呢？

**练习8.3** 为什么后门准则要求$Z$不包含$X$的后代？请用一个具体例子说明。

### 数学推导题

**练习8.4** 证明在可忽略性假设下，ATE可以表示为：
$$\tau = \sum_x [E[Y|T=1, X=x] - E[Y|T=0, X=x]] P(X=x)$$

**练习8.5** 考虑线性结构方程模型：
$$\begin{aligned}
Z &= \epsilon_Z \\
X &= \alpha Z + \epsilon_X \\
Y &= \beta X + \gamma Z + \epsilon_Y
\end{aligned}$$
其中所有$\epsilon$独立同分布$N(0,1)$。
- (a) 写出$E[Y|do(X=x)]$的表达式
- (b) 证明$E[Y|X=x]$与$E[Y|do(X=x)]$不同，并计算偏差

**练习8.6** 前门准则证明：证明前门调整公式
$$P(y|do(x)) = \sum_z P(z|x) \sum_{x'} P(y|x', z) P(x')$$
在适当条件下等于真实的干预分布。

### 编程实践题

**练习8.7** 实现一个完整的因果推断分析流程：
```python
# 使用本章提供的代码，对以下场景进行因果分析
# 场景: 某电商想评估"是否发送优惠券"对"用户消费金额"的影响
# 混杂因素: 用户历史消费金额、用户活跃度、会员等级
# 要求:
# 1. 生成模拟数据 (包含选择偏差)
# 2. 计算朴素估计
# 3. 使用至少两种方法进行因果效应估计
# 4. 比较结果并讨论
```

**练习8.8** 扩展`CausalGraph`类，实现：
- (a) 前门准则检测
- (b) 工具变量识别
- (c) 图的道德化 (moralization)

**练习8.9** 实现敏感性分析：
```python
# 给定一个因果效应估计，评估其对未观测混杂因素的敏感性
# 输入: 估计的ATE, 倾向得分范围, 假设的混杂强度
# 输出: 在什么程度的未观测混杂下，结论会逆转
```

---

## 参考文献

1. **Pearl, J.** (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. (因果推断圣经)

2. **Pearl, J., & Mackenzie, D.** (2018). *The Book of Why: The New Science of Cause and Effect*. Basic Books. (费曼法解释因果推断)

3. **Rubin, D. B.** (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. *Journal of Educational Psychology*, 66(5), 688-701. (潜在结果框架奠基)

4. **Rosenbaum, P. R., & Rubin, D. B.** (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55. (倾向得分经典)

5. **Imbens, G. W., & Rubin, D. B.** (2015). *Causal Inference in Statistics, Social, and Biomedical Sciences*. Cambridge University Press. (因果推断教材)

6. **Morgan, S. L., & Winship, C.** (2015). *Counterfactuals and Causal Inference: Methods and Principles for Social Research* (2nd ed.). Cambridge University Press. (社会科学因果推断)

7. **Pearl, J.** (1995). Causal diagrams for empirical research. *Biometrika*, 82(4), 669-688. (因果图奠基)

8. **Athey, S., & Imbens, G. W.** (2016). Recursive partitioning for heterogeneous causal effects. *Proceedings of the National Academy of Sciences*, 113(27), 7353-7360. (因果森林)

9. **Hernán, M. A., & Robins, J. M.** (2020). *Causal Inference: What If*. Boca Raton: Chapman & Hall/CRC. (最新因果推断教材，免费在线)

10. **Guo, R., Cheng, L., Li, J., Hahn, P. R., & Liu, H.** (2020). A survey of learning causality with data: Problems and methods. *ACM Computing Surveys*, 53(4), 1-37. (因果推断综述)

---

## 进一步阅读

- **在线课程**: 
  - "Causal Inference" by Brady Neal (bradyneal.com/causal-inference-course)
  - "Causal Diagrams" by Miguel Hernán (edX)

- **Python库**:
  - `DoWhy`: 端到端因果推断 (microsoft.github.io/dowhy)
  - `EconML`: 微软的因果机器学习库
  - `CausalML`: Uber的因果推断库
  - `pgmpy`: 概率图模型库

- **关键论文**:
  - Pearl (2019). The seven tools of causal inference, with reflections on machine learning. *Communications of the ACM*.
  - Schölkopf et al. (2021). Toward causal representation learning. *Proceedings of the IEEE*.

---

## 下章预告

**第五十二章：深度生成模型进阶**

我们将在本章因果推断的基础上，探索：
- 因果生成模型与反事实生成
- 潜在变量模型中的因果解释
- 因果表示学习前沿

因果推断是现代AI从"预测"走向"理解"、从"相关"走向"因果"的必经之路。掌握了因果思维，你不仅能构建更好的预测模型，还能回答那些真正重要的问题：**"如果...会怎样？"**

---

*本章完*
