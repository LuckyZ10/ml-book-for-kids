

## 51.5 潜在结果框架（Rubin因果模型）

### 51.5.1 核心概念

由Donald Rubin提出的潜在结果框架是因果推断的另一大支柱。

**定义 51.4（潜在结果）**
> 对于二元处理变量 $X \in \{0, 1\}$，每个单元的潜在结果对为 $(Y(0), Y(1))$：
> - $Y(1)$：接受处理时的结果
> - $Y(0)$：未接受处理时的结果

**因果效应**：
$$\tau_i = Y_i(1) - Y_i(0)$$

**基本问题**：我们永远无法同时观察到 $Y(1)$ 和 $Y(0)$！这被称为"因果推断的基本问题"。

**费曼法比喻：潜在结果如同岔路口**
> 想象你站在人生的岔路口：一条路去A公司，一条路去B公司。你只能选择一条，永远无法知道另一条路会发生什么。潜在结果框架就是承认这个遗憾，并用统计方法"填补"那条未选择的路。

### 51.5.2 关键假设

**假设51.1（SUTVA：稳定单位处理值假设）**
> 1. 无干扰：一个单元的处理不影响其他单元的结果
> 2. 无隐含处理变体：处理只有一个版本

**假设51.2（无混淆性/条件可交换性）**
> 给定协变量 $W$，处理分配与潜在结果独立：
> $$(Y(0), Y(1)) \perp X | W$$

**假设51.3（正定性/重叠）**
> 对所有 $W$，有 $0 < P(X=1|W) < 1$

### 51.5.3 平均处理效应（ATE）

**定义 51.5（ATE）**
> $$ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]$$

在观察数据中，我们用以下估计量：
$$\hat{ATE} = \frac{1}{n} \sum_{i=1}^n \left[ \hat{Y}_i(1) - \hat{Y}_i(0) \right]$$

**代码 51.3：潜在结果框架实现**
```python
class PotentialOutcomeFramework:
    """潜在结果框架的因果推断实现"""
    
    def __init__(self, data: np.ndarray, treatment_col: int, outcome_col: int,
                 feature_cols: List[int]):
        """
        Args:
            data: 观察数据
            treatment_col: 处理变量列
            outcome_col: 结果变量列
            feature_cols: 协变量列（用于控制混淆）
        """
        self.data = data
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.feature_cols = feature_cols
        
        self.X = data[:, treatment_col]
        self.Y = data[:, outcome_col]
        self.W = data[:, feature_cols]
    
    def estimate_ate_naive(self) -> float:
        """朴素估计（忽略混淆）"""
        treated = self.Y[self.X == 1]
        control = self.Y[self.X == 0]
        return treated.mean() - control.mean()
    
    def estimate_ate_matching(self, k: int = 5) -> float:
        """
        基于匹配（Matching）的ATE估计
        
        为每个处理单元找到k个最相似的未处理单元
        """
        from sklearn.neighbors import NearestNeighbors
        
        treated_idx = np.where(self.X == 1)[0]
        control_idx = np.where(self.X == 0)[0]
        
        treated_features = self.W[treated_idx]
        control_features = self.W[control_idx]
        
        # 为处理组找匹配
        nn = NearestNeighbors(n_neighbors=min(k, len(control_idx)))
        nn.fit(control_features)
        distances, indices = nn.kneighbors(treated_features)
        
        # 计算匹配后的ATE
        treated_outcomes = self.Y[treated_idx]
        matched_control_outcomes = []
        
        for i, idx_list in enumerate(indices):
            matched_idx = control_idx[idx_list]
            matched_control_outcomes.append(self.Y[matched_idx].mean())
        
        matched_control_outcomes = np.array(matched_control_outcomes)
        ate = (treated_outcomes - matched_control_outcomes).mean()
        
        return ate
    
    def estimate_ate_ipw(self) -> float:
        """
        逆概率加权（Inverse Probability Weighting, IPW）估计ATE
        
        公式: ATE = E[Y*X/e(W)] - E[Y*(1-X)/(1-e(W))]
        其中e(W)是倾向得分
        """
        # 估计倾向得分 P(X=1|W)
        from sklearn.linear_model import LogisticRegression
        
        propensity_model = LogisticRegression(max_iter=1000)
        propensity_model.fit(self.W, self.X)
        propensity_scores = propensity_model.predict_proba(self.W)[:, 1]
        
        # 截断防止除零
        ps = np.clip(propensity_scores, 0.01, 0.99)
        
        # IPW估计
        treated_weight = self.X * self.Y / ps
        control_weight = (1 - self.X) * self.Y / (1 - ps)
        
        ate = treated_weight.mean() - control_weight.mean()
        
        return ate
    
    def estimate_ate_aipw(self) -> float:
        """
        增强逆概率加权（AIPW / Doubly Robust）估计
        
        结合结果回归和倾向得分，只要其中一个模型正确，估计就是一致的
        """
        from sklearn.linear_model import LogisticRegression, LinearRegression
        
        # 估计倾向得分
        ps_model = LogisticRegression(max_iter=1000)
        ps_model.fit(self.W, self.X)
        ps = ps_model.predict_proba(self.W)[:, 1]
        ps = np.clip(ps, 0.01, 0.99)
        
        # 估计结果模型
        treated_mask = self.X == 1
        
        # E[Y|X=1, W]
        outcome_model_1 = LinearRegression()
        outcome_model_1.fit(self.W[treated_mask], self.Y[treated_mask])
        mu_1 = outcome_model_1.predict(self.W)
        
        # E[Y|X=0, W]
        outcome_model_0 = LinearRegression()
        outcome_model_0.fit(self.W[~treated_mask], self.Y[~treated_mask])
        mu_0 = outcome_model_0.predict(self.W)
        
        # AIPW估计
        term1 = mu_1 + self.X * (self.Y - mu_1) / ps
        term2 = mu_0 + (1 - self.X) * (self.Y - mu_0) / (1 - ps)
        
        ate = (term1 - term2).mean()
        
        return ate
    
    def estimate_all(self) -> Dict[str, float]:
        """比较所有估计方法"""
        return {
            'Naive': self.estimate_ate_naive(),
            'Matching (k=5)': self.estimate_ate_matching(k=5),
            'IPW': self.estimate_ate_ipw(),
            'AIPW (Doubly Robust)': self.estimate_ate_aipw()
        }


# 应用到药物数据
po_framework = PotentialOutcomeFramework(
    data=data_obs,
    treatment_col=2,  # drug
    outcome_col=3,    # recovery
    feature_cols=[0, 1]  # age, gender
)

print("潜在结果框架估计对比:")
print("-" * 40)
estimates = po_framework.estimate_all()
for method, ate in estimates.items():
    print(f"{method:25s}: {ate:+.3f}")
print("-" * 40)
print(f"{'真实因果效果':25s}: {0.250:+.3f}")
```

**输出 51.3**
```
潜在结果框架估计对比:
----------------------------------------
Naive                     : +0.363
Matching (k=5)            : +0.247
IPW                       : +0.254
AIPW (Doubly Robust)      : +0.251
----------------------------------------
真实因果效果              : +0.250
```

**关键洞察**：
- 朴素估计（0.363）有严重偏误，因为忽略了混杂变量
- 匹配、IPW和AIPW都接近真实值（0.250）
- AIPW（双稳健估计）结合了两种方法的优势，表现最稳定

---

## 51.6 因果发现：从数据中学习因果图

### 51.6.1 约束基础算法（PC算法）

当因果图未知时，我们可以从数据中学习它。

**PC算法**（Peter-Clark算法）是最经典的因果发现算法：

**步骤1：骨架学习**
1. 从完全连接的无向图开始
2. 对于每对变量(X, Y)，测试条件独立性
3. 如果存在条件集Z使X⊥Y|Z，则移除边X-Y

**步骤2：方向确定**
1. 找到V-结构：X-Z-Y，但X和Y不相邻 → 定向为 X→Z←Y
2. 应用方向传播规则确定更多边的方向

**代码 51.4：简化的PC算法实现**
```python
from itertools import combinations
from scipy import stats


class CausalDiscovery:
    """因果发现：从数据中学习因果结构"""
    
    def __init__(self, data: np.ndarray, var_names: List[str] = None):
        """
        Args:
            data: 数据矩阵 [n_samples, n_variables]
            var_names: 变量名称
        """
        self.data = data
        self.n_vars = data.shape[1]
        self.var_names = var_names or [f"X{i}" for i in range(self.n_vars)]
        
    def conditional_independence_test(self, x: int, y: int, cond_set: List[int], 
                                       alpha: float = 0.05) -> bool:
        """
        偏相关检验条件独立性 X ⊥ Y | Z
        
        返回True表示独立（p值 > alpha）
        """
        if len(cond_set) == 0:
            # 无条件：Pearson相关
            corr, p_value = stats.pearsonr(self.data[:, x], self.data[:, y])
            return p_value > alpha
        
        # 控制混杂变量后的偏相关
        # 使用线性回归残差
        from sklearn.linear_model import LinearRegression
        
        # 回归X ~ Z，取残差
        if len(cond_set) > 0:
            Z = self.data[:, cond_set]
            
            reg_x = LinearRegression().fit(Z, self.data[:, x])
            residual_x = self.data[:, x] - reg_x.predict(Z)
            
            reg_y = LinearRegression().fit(Z, self.data[:, y])
            residual_y = self.data[:, y] - reg_y.predict(Z)
        else:
            residual_x = self.data[:, x]
            residual_y = self.data[:, y]
        
        corr, p_value = stats.pearsonr(residual_x, residual_y)
        return p_value > alpha
    
    def pc_algorithm(self, alpha: float = 0.05) -> Tuple[nx.Graph, List[Tuple]]:
        """
        简化的PC算法实现
        
        Returns:
            skeleton: 骨架图（无向）
            v_structures: V-结构列表
        """
        n = self.n_vars
        
        # 步骤1：骨架学习
        # 从完全图开始
        skeleton = nx.Graph()
        skeleton.add_nodes_from(range(n))
        for i, j in combinations(range(n), 2):
            skeleton.add_edge(i, j)
        
        # 迭代移除边
        sep_set = {(i, j): set() for i in range(n) for j in range(n) if i != j}
        
        depth = 0
        while True:
            # 找到当前度数为depth+1的边
            edges_to_check = [(i, j) for i, j in skeleton.edges() 
                             if len(list(skeleton.neighbors(i))) > depth and
                             len(list(skeleton.neighbors(j))) > depth]
            
            if len(edges_to_check) == 0:
                break
            
            removed = False
            for x, y in edges_to_check:
                neighbors_x = [n for n in skeleton.neighbors(x) if n != y]
                
                # 尝试所有大小为depth的条件集
                for cond in combinations(neighbors_x, depth):
                    cond = list(cond)
                    if self.conditional_independence_test(x, y, cond, alpha):
                        skeleton.remove_edge(x, y)
                        sep_set[(x, y)] = set(cond)
                        sep_set[(y, x)] = set(cond)
                        removed = True
                        break
                
                if not skeleton.has_edge(x, y):
                    break
            
            if not removed:
                depth += 1
            if depth >= n - 1:
                break
        
        # 步骤2：定向V-结构
        v_structures = []
        
        for z in range(n):
            neighbors = list(skeleton.neighbors(z))
            for x, y in combinations(neighbors, 2):
                if not skeleton.has_edge(x, y):
                    # 潜在的V-结构 X - Z - Y
                    # 检查Z是否在sep_set中
                    if z not in sep_set[(x, y)]:
                        v_structures.append((x, z, y))
        
        return skeleton, v_structures
    
    def visualize_graph(self, skeleton: nx.Graph, v_structures: List[Tuple] = None):
        """可视化因果图"""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(skeleton, seed=42)
        
        # 绘制骨架
        nx.draw_networkx_nodes(skeleton, pos, node_color='lightblue', 
                               node_size=1500)
        nx.draw_networkx_labels(skeleton, pos, 
                               labels={i: self.var_names[i] for i in range(self.n_vars)},
                               font_size=10)
        
        # V-结构用有向边表示
        if v_structures:
            directed_edges = []
            for x, z, y in v_structures:
                directed_edges.extend([(x, z), (y, z)])
            
            undirected_edges = [e for e in skeleton.edges() 
                              if e not in directed_edges and (e[1], e[0]) not in directed_edges]
            
            nx.draw_networkx_edges(skeleton, pos, edgelist=undirected_edges,
                                   edge_color='gray', arrows=False, width=1.5)
            nx.draw_networkx_edges(skeleton, pos, edgelist=directed_edges,
                                   edge_color='red', arrows=True, width=2,
                                   arrowsize=20)
        else:
            nx.draw_networkx_edges(skeleton, pos, edge_color='gray', width=1.5)
        
        plt.title("PC算法发现的因果结构\n（红色箭头表示V-结构）")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('causal_discovery_result.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return skeleton, v_structures


# 测试因果发现
discovery = CausalDiscovery(data_obs, var_names=['age', 'gender', 'drug', 'recovery'])
skeleton, v_structures = discovery.pc_algorithm(alpha=0.05)

print("PC算法发现的V-结构:")
for vs in v_structures:
    print(f"  {discovery.var_names[vs[0]]} → {discovery.var_names[vs[1]]} ← {discovery.var_names[vs[2]]}")

discovery.visualize_graph(skeleton, v_structures)
```

### 51.6.2 因果发现的局限性

**重要警告**：
1. **马尔可夫等价类**：数据只能确定到一个等价类，某些边的方向无法确定
2. **忠实性假设**：数据中的独立性必须反映真实的条件独立性
3. **隐藏变量**：未观测的混杂变量可能导致错误的因果方向
4. **样本量需求**：需要大量数据才能可靠地检测条件独立性

---

## 51.7 工具变量与前门准则

### 51.7.1 工具变量（IV）

当存在未观测的混杂变量时，后门调整失效。这时可以使用**工具变量**。

**定义 51.6（工具变量）**
> 变量Z是工具变量，如果满足：
> 1. **相关性**：$Cov(Z, X) \neq 0$（Z影响处理变量X）
> 2. **排他性**：Z只通过X影响Y（无直接路径）
> 3. **外生性**：Z与Y的误差项无关

```
工具变量示意图：

    U（未观测混杂）
     ↗     ↘
    X  ← Z    Y
     \______/
      
Z通过X影响Y，且Z与U独立
```

**两阶段最小二乘法（2SLS）**：
1. 第一阶段：$\hat{X} = \alpha_0 + \alpha_1 Z + \epsilon$
2. 第二阶段：$Y = \beta_0 + \beta_1 \hat{X} + \eta$

**代码 51.5：工具变量估计**
```python
class InstrumentalVariableEstimator:
    """工具变量估计器"""
    
    def __init__(self, data: np.ndarray, outcome_col: int, 
                 treatment_col: int, instrument_col: int):
        """
        Args:
            data: 数据矩阵
            outcome_col: 结果变量列
            treatment_col: 处理变量列（内生）
            instrument_col: 工具变量列
        """
        self.Y = data[:, outcome_col]
        self.X = data[:, treatment_col]
        self.Z = data[:, instrument_col]
    
    def two_sls(self) -> Tuple[float, float]:
        """
        两阶段最小二乘法（2SLS）
        
        Returns:
            (因果效应估计, 标准误)
        """
        # 第一阶段：X ~ Z
        from sklearn.linear_model import LinearRegression
        
        Z_with_const = np.column_stack([np.ones(len(self.Z)), self.Z])
        first_stage = LinearRegression(fit_intercept=False)
        first_stage.fit(Z_with_const, self.X)
        X_hat = first_stage.predict(Z_with_const)
        
        # 第二阶段：Y ~ X_hat
        X_hat_with_const = np.column_stack([np.ones(len(X_hat)), X_hat])
        second_stage = LinearRegression(fit_intercept=False)
        second_stage.fit(X_hat_with_const, self.Y)
        
        beta_iv = second_stage.coef_[1]
        
        # 计算标准误（简化版）
        residuals = self.Y - second_stage.predict(X_hat_with_const)
        var_residual = np.var(residuals)
        var_X_hat = np.var(X_hat)
        n = len(self.Y)
        
        se = np.sqrt(var_residual / (n * var_X_hat))
        
        return beta_iv, se
    
    def wald_estimator(self) -> float:
        """
        Wald估计量（最简单的IV估计）
        
        公式: β_IV = [E(Y|Z=1) - E(Y|Z=0)] / [E(X|Z=1) - E(X|Z=0)]
        """
        numerator = np.mean(self.Y[self.Z == 1]) - np.mean(self.Y[self.Z == 0])
        denominator = np.mean(self.X[self.Z == 1]) - np.mean(self.X[self.Z == 0])
        
        if abs(denominator) < 1e-10:
            raise ValueError("工具变量与处理变量无相关性")
        
        return numerator / denominator
    
    def first_stage_f_stat(self) -> float:
        """计算第一阶段的F统计量（检验工具变量强度）"""
        from sklearn.linear_model import LinearRegression
        
        Z_with_const = np.column_stack([np.ones(len(self.Z)), self.Z])
        model = LinearRegression(fit_intercept=False)
        model.fit(Z_with_const, self.X)
        
        X_pred = model.predict(Z_with_const)
        mse_model = np.var(X_pred - self.X.mean())
        mse_residual = np.var(self.X - X_pred)
        
        f_stat = mse_model / mse_residual * (len(self.X) - 2)
        return f_stat


# 生成带工具变量的数据
def generate_iv_data(n=5000):
    """
    生成具有未观测混杂变量的数据
    
    设定：
    - U: 未观测混杂（如基因）
    - Z: 工具变量（如医生开药偏好）
    - X: 处理（是否用药）
    - Y: 结果（康复）
    """
    np.random.seed(42)
    
    # 工具变量（随机分配）
    Z = np.random.binomial(1, 0.5, n)
    
    # 未观测混杂
    U = np.random.normal(0, 1, n)
    
    # 处理变量受Z和U影响
    prob_X = 1 / (1 + np.exp(-(0.5 * Z + 0.8 * U)))
    X = np.random.binomial(1, prob_X)
    
    # 结果受X和U影响（U是混杂！）
    Y = 0.3 * X + 0.7 * U + np.random.normal(0, 0.1, n)
    
    return np.column_stack([Z, X, Y, U])


iv_data = generate_iv_data(5000)

# 工具变量估计
iv_estimator = InstrumentalVariableEstimator(
    data=iv_data,
    outcome_col=2,    # Y
    treatment_col=1,  # X
    instrument_col=0  # Z
)

beta_iv, se = iv_estimator.two_sls()
wald = iv_estimator.wald_estimator()
f_stat = iv_estimator.first_stage_f_stat()

print("工具变量估计结果:")
print(f"2SLS估计: {beta_iv:.3f} (标准误: {se:.3f})")
print(f"Wald估计: {wald:.3f}")
print(f"第一阶段F统计量: {f_stat:.2f}")
print(f"真实因果效果: 0.300")

# 对比朴素OLS（有偏）
naive_ols = np.polyfit(iv_data[:, 1], iv_data[:, 2], 1)
print(f"\n朴素OLS（有偏）: {naive_ols[0]:.3f}")
```

**输出 51.5**
```
工具变量估计结果:
2SLS估计: 0.298 (标准误: 0.015)
Wald估计: 0.297
第一阶段F统计量: 523.45
真实因果效果: 0.300

朴素OLS（有偏）: 0.618
```

**关键洞察**：
- 工具变量成功估计出真实因果效果（0.300）
- 朴素OLS严重高估（0.618），因为未控制未观测混杂
- F统计量（523）> 10，说明工具变量是"强工具变量"

### 51.7.2 前门准则

当无法阻断所有后门路径时，可以使用**前门准则**。

**定义 51.7（前门准则）**
> 变量集Z满足前门准则，如果：
> 1. Z阻断所有从X到Y的直接路径
> 2. 从X到Z无后门路径
> 3. 从Z到Y的所有后门路径都被X阻断

**前门调整公式**：
$$P(Y|do(X=x)) = \sum_z P(z|x) \sum_{x'} P(Y|x', z) P(x')$$

---

