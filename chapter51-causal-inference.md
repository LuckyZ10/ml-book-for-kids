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

