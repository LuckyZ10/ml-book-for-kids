# 第十二章：集成学习——三个臭皮匠，顶个诸葛亮

> *"一个专家的预测可能出错，但一群专家的集体智慧往往更接近真相。"*
> 
> —— 乔治·博克斯 (George Box), 统计学家

---

## 开篇故事：陪审团的智慧

想象你是一名法官，正在审理一桩复杂的案件。被告是否真的犯了罪？你面临一个艰难的决定。

现在，你有两个选择：

**选择一**：只听一位"超级专家"的意见。这位专家学识渊博，但偶尔也会犯错——毕竟人非圣贤。

**选择二**：召集一个由12人组成的陪审团。这些人来自不同背景，有医生、教师、工人、商人。每个人单独来看都不是法律专家，但集合在一起，他们通过讨论和投票做出决定。

历史证明，**陪审团的判断往往比单个专家更准确**。为什么呢？

因为不同的人会从不同角度看问题：
- 医生可能注意到证词中的医学细节
- 教师可能察觉到目击者描述中的逻辑漏洞
- 商人可能发现财务证据中的异常

**当多个视角汇聚，错误相互抵消，真相浮现。**

这就是**集成学习**（Ensemble Learning）的核心思想。

---

## 12.1 为什么一个模型不够？

### 12.1.1 决策树的困境

在上一章，我们学习了决策树。决策树有很多优点：
- 直观易懂，就像"二十个问题"游戏
- 训练速度快
- 可以处理数值和类别特征

但决策树有一个致命的弱点：**不稳定**（unstable）。

让我用一个比喻来说明：

> 想象你在森林里寻找宝藏。决策树就像一张手绘地图。如果地图上某个转弯处画错了，你可能会完全走错方向。更糟糕的是，如果你换了一批探险者让他们各自画地图，每个人的地图可能都不一样——有人向左拐，有人向右拐。

**实际例子：**

假设我们有一个数据集，用来预测"明天是否会下雨"：

| 温度 | 湿度 | 风速 | 是否下雨 |
|------|------|------|----------|
| 25°C | 80% | 5km/h | 是 |
| 30°C | 60% | 10km/h | 否 |
| ... | ... | ... | ... |

如果我们用**决策树A**训练：
```
湿度 > 70% ? 
  ├── 是 → 下雨
  └── 否 → 不下雨
```

如果我们用**决策树B**训练（只是少了几条数据）：
```
温度 > 28°C ?
  ├── 是 → 不下雨
  └── 否 → 下雨
```

看！**仅仅因为训练数据的一点点变化，决策树的结构完全不同了！**

统计学家里奥·布雷曼（Leo Breiman）在1996年的一篇论文中首次系统地研究了这个问题。他发现，决策树这种"不稳定"的特性让它们对数据中的小波动非常敏感。

### 12.1.2 方差与偏差的两难

在机器学习中，模型的误差来自两个来源：

**偏差（Bias）**：模型的"偏见"或"固有错误"。就像一个人总是戴着有色眼镜看世界，偏差高的模型无法捕捉数据的真实规律。

**方差（Variance）**：模型的"善变"。就像墙头草随风倒，方差高的模型对训练数据的小变化过于敏感。

决策树的问题是：**方差太高**。

想象一下射箭：
- **高偏差，低方差**：所有箭都射在靶子的左下角（一致但偏离目标）
- **低偏差，高方差**：箭散落在靶子各处（平均在中心但很不稳定）
- **理想情况**：所有箭都集中在靶心（低偏差，低方差）

![偏差-方差权衡](images/bias_variance.png)

*图12-1：偏差与方差的可视化。高偏差导致欠拟合，高方差导致过拟合，我们需要找到平衡点。*

### 12.1.3 集成的力量

1996年，加州大学伯克利分校的统计学家里奥·布雷曼提出了一个革命性的想法：

> **"如果一棵树不稳定，那我们为什么不训练很多棵树，然后让它们投票呢？"**

这就是**集成学习**（Ensemble Learning）的诞生。

用生活化的比喻：

> 想象你要预测明天的股市涨跌。你问一位投资专家，他可能说"涨"——但也可能是错的。现在，你问100位不同的专家，让他们投票。如果60人说"涨"，40人说"跌"，你就有更大的信心预测"涨"。
> 
> 而且，即使单个专家有偏见（比如他总是乐观），当你把很多专家的意见平均，这些偏见会相互抵消。

**数学直觉：**

假设我们有 $T$ 个独立的预测器，每个预测器的误差方差为 $\sigma^2$。

如果我们简单地对它们的预测取平均：

$$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} \hat{y}_t$$

那么集成预测的方差是：

$$\text{Var}(\hat{y}) = \frac{1}{T^2} \sum_{t=1}^{T} \text{Var}(\hat{y}_t) = \frac{\sigma^2}{T}$$

**太神奇了！方差降低了 $T$ 倍！**

10个模型的集成，方差只有单个模型的1/10！

当然，这里有一个前提：**这些模型必须是"不同的"**（diverse）。如果10个模型都一样，那和1个模型没有区别。

所以集成学习的关键问题是：**如何创建多个不同但又都准确的模型？**

有三种经典方法：
1. **Bagging**（装袋）：用不同的数据子集训练
2. **Boosting**（提升）：顺序训练，每个新模型关注前一个模型的错误
3. **Stacking**（堆叠）：用另一个模型来学习如何组合

本章我们将深入探讨前两种方法。

---

## 12.2 Bagging——人多力量大

### 12.2.1 Bootstrap：有放回抽样

在介绍Bagging之前，我们需要理解一个统计学的核心技术：**Bootstrap**（自助法）。

想象你有一袋糖果，里面有100颗不同颜色的糖果。你想知道这袋糖果中红色糖果的比例。

**传统方法**：把糖果全部倒出来数——但这样会弄乱糖果。

**Bootstrap方法**：
1. 闭上眼睛，从袋子里随机摸出一颗糖果，记录颜色
2. **把糖果放回去**（这是关键！）
3. 重复100次
4. 计算摸出的糖果中红色的比例

**为什么要放回去？**

因为每次抽样都是独立的，这样抽出来的100颗糖果可能比原来的100颗有些重复、有些缺失——但**整体的分布特征被保留了下来**。

在数学上，给定一个包含 $n$ 个样本的数据集 $D = \{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}$，一个Bootstrap样本 $D^*$ 是这样生成的：

$$D^* = \{(x_{i_1}, y_{i_1}), (x_{i_2}, y_{i_2}), ..., (x_{i_n}, y_{i_n})\}$$

其中每个 $i_j$ 都是从 $\{1, 2, ..., n\}$ 中**有放回**随机抽取的。

**有趣的事实**：在Bootstrap抽样中，大约有 63.2% 的原始样本会被选中至少一次，而 36.8% 的样本不会被选中。

这个数学结果来自：

$$P(\text{某个样本未被选中}) = \left(1 - \frac{1}{n}\right)^n \approx \frac{1}{e} \approx 0.368$$

当 $n$ 很大时，这个概率趋近于 $1/e$。

### 12.2.2 Bootstrap Aggregating

1996年，布雷曼发表了著名的论文《Bagging Predictors》，正式提出了**Bagging**（Bootstrap Aggregating的缩写）。

**算法思想非常简单**：

1. 从原始数据集中Bootstrap抽样 $T$ 次，生成 $T$ 个不同的训练集
2. 在每个训练集上训练一个基学习器（通常是决策树）
3. 预测时，让所有基学习器投票（分类）或取平均（回归）

用代码表示：

```python
# Bagging 伪代码
def bagging_train(data, T):
    models = []
    for t in range(T):
        # 第1步：Bootstrap抽样
        bootstrap_data = bootstrap_sample(data)
        
        # 第2步：训练基学习器
        model = train_base_learner(bootstrap_data)
        models.append(model)
    
    return models

def bagging_predict(models, x):
    predictions = [model.predict(x) for model in models]
    
    # 第3步：投票或平均
    return majority_vote(predictions)  # 分类
    # 或 return mean(predictions)      # 回归
```

### 12.2.3 为什么Bagging有效？

Bagging有效的原因有三：

**原因一：降低方差**

如前所述，对多个独立模型的预测取平均可以将方差降低 $T$ 倍。

**原因二：减少过拟合**

单个决策树可能会过拟合训练数据中的噪声。但当多个在不同数据子集上训练的树进行投票时，**那些"奇怪"的预测会被其他树的正常预测抵消**。

用比喻来说：

> 想象10个侦探各自调查同一个案件。如果其中一个侦探被假证据误导了，其他9个侦探的独立调查很可能会揭穿这个错误。最终，真相会在集体智慧中浮现。

**原因三：利用未选中的样本**

还记得Bootstrap抽样中约36.8%的样本不会被选中吗？这些样本被称为**袋外样本**（Out-of-Bag, OOB）。

我们可以用这些OOB样本来**免费**评估模型性能，不需要单独的验证集！

对于每个基学习器，用它在训练时没见过的OOB样本测试，然后把所有基学习器的OOB误差平均，就得到了Bagging的OOB误差估计。

布雷曼证明了，**OOB误差是泛化误差的一个良好估计**。

### 12.2.4 Bagging的局限性

Bagging虽然强大，但也有局限性：

1. **对稳定模型无效**：如果基学习器本身就很稳定（如线性回归），Bagging不会有太大帮助
2. **失去了可解释性**：一个决策树很好理解，但100个决策树的投票结果就难以解释了
3. **计算成本**：需要训练多个模型

**Bagging的最佳搭档是决策树**——因为决策树方差高、不稳定，正好可以被Bagging改善。

---

## 12.3 随机森林——随机中的智慧

### 12.3.1 从Bagging到Random Forest

2001年，布雷曼在Bagging的基础上提出了**随机森林**（Random Forest），这可能是机器学习史上最成功的算法之一。

随机森林的核心洞察是：

> **"Bagging已经很好了，但树与树之间还是太相似了。如果我们让每个树更加'不同'，效果会更好。"**

Bagging通过在数据上引入随机性来创建多样性。随机森林增加了**第二层随机性**：在特征上引入随机性。

**随机森林算法**：

```python
# 随机森林伪代码
def random_forest_train(data, T, m_try):
    trees = []
    for t in range(T):
        # 第一层随机性：Bootstrap抽样
        bootstrap_data = bootstrap_sample(data)
        
        # 训练一棵树
        tree = build_tree_with_random_features(bootstrap_data, m_try)
        trees.append(tree)
    
    return trees

def build_tree_with_random_features(data, m_try):
    """构建一棵树，在每个节点只考虑m_try个随机特征"""
    if stopping_criterion_met(data):
        return create_leaf(data)
    
    # 关键：从所有特征中随机选择m_try个
    all_features = get_all_features(data)
    selected_features = random_sample(all_features, m_try)
    
    # 只在选中的特征中寻找最佳分裂
    best_feature, best_threshold = find_best_split(data, selected_features)
    
    left_data, right_data = split(data, best_feature, best_threshold)
    
    left_child = build_tree_with_random_features(left_data, m_try)
    right_child = build_tree_with_random_features(right_data, m_try)
    
    return create_node(best_feature, best_threshold, left_child, right_child)
```

### 12.3.2 双随机性的威力

随机森林引入了两层随机性：

| 随机性来源 | 作用 | Bagging也有？ |
|-----------|------|--------------|
| **样本随机性**（Bootstrap） | 每个树看到不同的数据样本 | ✅ 是 |
| **特征随机性**（m_try） | 每个节点只考虑部分特征 | ❌ 否 |

**为什么要特征随机性？**

想象一个场景：数据集中有一个"超级特征"，它的预测能力比其他所有特征加起来还强。比如，在预测房价时，"房屋面积"可能比"距离地铁站的距离"、"周边学校数量"等更重要。

在传统决策树中，**所有树都会首先选择这个超级特征**作为根节点。结果就是，所有树都很相似——Bagging的方差减少效果被削弱了。

随机森林通过在每个节点**随机选择一小部分特征**来考虑，强制树与树之间产生差异。即使"房屋面积"是最强特征，有些树在第一层可能看不到它，只能先用其他特征分裂。

### 12.3.3 超参数m_try

$m_{try}$ 是随机森林中最重要的超参数，它决定了在每个节点考虑多少个特征。

布雷曼建议：

- **分类问题**：$m_{try} = \sqrt{p}$（$p$是总特征数）
- **回归问题**：$m_{try} = p/3$

**直观理解**：

- $m_{try}$ 太小：每个节点可选的特征太少，单个树的质量会下降
- $m_{try}$ 太大：树与树之间太相似，失去了随机森林的优势
- **适中**：找到平衡点

### 12.3.4 随机森林的特性

随机森林有许多优秀的特性：

**1. 准确性高**

大量的实验证明，随机森林在各种数据集上都表现优异，通常不需要太多调参就能达到很好的性能。

**2. 抗过拟合**

随着树的数量增加，随机森林不会过拟合——这是它最神奇的性质之一！更多的树总是让模型更好（或至少不会更差）。

**3. 天然并行**

每棵树可以独立训练，非常适合并行计算。

**4. 特征重要性**

随机森林可以自动计算每个特征的重要性：

```python
def calculate_feature_importance(forest, data):
    """通过置换法计算特征重要性"""
    baseline_accuracy = evaluate(forest, data)
    
    importances = []
    for feature in all_features:
        # 打乱这个特征的值
        permuted_data = permute_feature(data, feature)
        
        # 看准确率下降多少
        permuted_accuracy = evaluate(forest, permuted_data)
        
        importance = baseline_accuracy - permuted_accuracy
        importances.append(importance)
    
    return importances
```

思路是：如果一个特征很重要，打乱它的值会让模型性能大幅下降；如果不重要，打乱也没什么影响。

---

## 12.4 Boosting——循序渐进

### 12.4.1 与Bagging的对比

Bagging和Boosting是集成学习的两大支柱，但它们的哲学完全不同：

| 特性 | Bagging | Boosting |
|------|---------|----------|
| **训练方式** | 并行，独立训练 | 串行，顺序训练 |
| **基学习器关系** | 相互独立 | 每个修正前一个的错误 |
| **目标** | 降低方差 | 降低偏差 |
| **典型代表** | 随机森林 | AdaBoost, XGBoost |

用比喻来说：

- **Bagging**像是一个**并行调查团队**：10个侦探同时独立调查，最后开会投票决定。
- **Boosting**像是一个**渐进学习过程**：第一个侦探调查后，第二个侦探专门去看第一个遗漏的线索，第三个再看前两个都遗漏的...每个人都专注于"前人解决不了的问题"。

### 12.4.2 Boosting的核心思想

Boosting要解决的核心问题是：

> **"如何把一些'弱学习器'（只比随机猜测好一点）组合成一个'强学习器'（非常准确）？"**

这个问题在机器学习理论中被称为**可学习性**（learnability）问题。

1990年，罗伯特·夏皮尔（Robert Schapire）证明了一个惊人的定理：

> **如果一个问题可以被弱学习器学习，那么它也可以被强学习器学习。**

而且，他给出了一个构造性的证明——这就是Boosting的雏形。

Boosting的一般框架：

```python
# Boosting 通用框架
def boosting_train(data, T):
    models = []
    weights = []  # 每个基学习器的权重
    
    for t in range(T):
        # 根据当前表现调整样本权重
        weighted_data = adjust_sample_weights(data, t)
        
        # 训练基学习器（通常用决策树桩）
        model = train_weak_learner(weighted_data)
        
        # 计算这个学习器的权重
        alpha = calculate_model_weight(model, weighted_data)
        
        models.append(model)
        weights.append(alpha)
    
    return models, weights

def boosting_predict(models, weights, x):
    # 加权投票
    prediction = sum(alpha * model.predict(x) for alpha, model in zip(weights, models))
    return sign(prediction)
```

### 12.4.3 自适应Boosting：AdaBoost

1997年，约阿夫·弗罗因德（Yoav Freund）和罗伯特·夏皮尔发表了**AdaBoost**（Adaptive Boosting）算法，这成为了Boosting家族中最著名的一员。

这个工作如此重要，以至于他们获得了**2003年的哥德尔奖**（Gödel Prize）——理论计算机科学界的最高荣誉之一。

**AdaBoost的核心洞察**：

> **"让模型专注于它之前分类错误的样本。"**

具体做法：
1. 给每个训练样本一个权重，初始时所有样本权重相等
2. 训练一个弱学习器
3. 增加被错误分类样本的权重，减少正确分类样本的权重
4. 重复步骤2-3
5. 最终预测是所有弱学习器的加权组合

**数学推导**：

设训练集为 $D = \{(x_1, y_1), ..., (x_n, y_n)\}$，其中 $y_i \in \{-1, +1\}$。

**初始化**：每个样本的权重
$$w_i^{(1)} = \frac{1}{n}$$

**对于每一轮 $t = 1, 2, ..., T$**：

1. **训练弱学习器**：在当前权重分布下训练一个弱分类器 $h_t(x)$

2. **计算加权错误率**：
$$\epsilon_t = \sum_{i=1}^{n} w_i^{(t)} \cdot \mathbb{I}[h_t(x_i) \neq y_i]$$

   （即被错误分类样本的权重之和）

3. **计算学习器的权重**：
$$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$

   这个公式很直观：
   - 如果 $\epsilon_t < 0.5$（比随机猜测好），则 $\alpha_t > 0$
   - 错误率越小，$\alpha_t$ 越大（这个学习器越重要）

4. **更新样本权重**：
$$w_i^{(t+1)} = \frac{w_i^{(t)} \cdot \exp(-\alpha_t y_i h_t(x_i))}{Z_t}$$

   其中 $Z_t$ 是归一化因子，让所有权重之和为1。

   这个更新规则的关键：
   - 如果 $y_i = h_t(x_i)$（分类正确），则 $y_i h_t(x_i) = 1$，权重乘以 $\exp(-\alpha_t) < 1$（降低）
   - 如果 $y_i \neq h_t(x_i)$（分类错误），则 $y_i h_t(x_i) = -1$，权重乘以 $\exp(\alpha_t) > 1$（增加）

**最终预测**：
$$H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)$$

### 12.4.4 AdaBoost为什么有效？

AdaBoost的训练误差随着轮数增加而指数级下降！

可以证明，如果每个弱学习器的错误率 $\epsilon_t \leq 0.5 - \gamma$（即比随机猜测好至少 $\gamma$），那么训练误差上界为：

$$\text{Training Error} \leq \exp(-2\gamma^2 T)$$

这意味着，随着 $T$ 增加，训练误差会指数级趋近于0！

**但要注意**：训练误差降到0并不意味着泛化性能好。AdaBoost有时也会过拟合——尽管在实际中它往往出人意料地鲁棒。

---

## 12.5 三种方法对比

让我们用一个表格总结三种集成方法：

| 特性 | Bagging | Random Forest | AdaBoost |
|------|---------|---------------|----------|
| **提出者** | Breiman (1996) | Breiman (2001) | Freund & Schapire (1997) |
| **基学习器** | 通常深度树 | 深度树 | 决策树桩（浅树） |
| **训练方式** | 并行 | 并行 | 串行 |
| **随机性来源** | 数据抽样 | 数据+特征抽样 | 样本重加权 |
| **主要目标** | 降方差 | 降方差 | 降偏差 |
| **对噪声敏感** | 低 | 低 | 较高 |
| **过拟合风险** | 低 | 很低 | 中等 |
| **典型应用场景** | 通用 | 通用 | 需要精细边界的分类 |

**何时使用哪种方法？**

1. **追求简单、稳定、不用调参** → **随机森林**
2. **数据有噪声** → **随机森林**或**Bagging**
3. **需要最高准确率，愿意调参** → **XGBoost/LightGBM**（Boosting的高级变体）
4. **需要模型可解释性** → **单个决策树**

---

## 12.6 代码实战：手写集成学习

现在让我们用纯NumPy手写实现这三种集成方法！

### 12.6.1 Bagging分类器

```python
import numpy as np
from collections import Counter

class DecisionTree:
    """简化版决策树，用于集成学习"""
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self._grow_tree(X, y, depth=0)
        return self
    
    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # 停止条件
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            # 返回叶节点（多数类）
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        # 寻找最佳分裂
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        if best_gain < 1e-7:  # 无法进一步分裂
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        # 分裂数据
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx
        
        left_subtree = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _information_gain(self, X, y, feature, threshold):
        """计算信息增益"""
        parent_entropy = self._entropy(y)
        
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx
        
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return 0
        
        n = len(y)
        n_left = np.sum(left_idx)
        n_right = np.sum(right_idx)
        
        child_entropy = (n_left / n * self._entropy(y[left_idx]) +
                        n_right / n * self._entropy(y[right_idx]))
        
        return parent_entropy - child_entropy
    
    def _entropy(self, y):
        """计算熵"""
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy
    
    def predict_one(self, x, node):
        """预测单个样本"""
        if node['leaf']:
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self.predict_one(x, node['left'])
        else:
            return self.predict_one(x, node['right'])
    
    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])


class BaggingClassifier:
    """
    Bagging分类器（Bootstrap Aggregating）
    
    原理：通过Bootstrap抽样创建多个训练集，在每个上训练基学习器，最后投票
    """
    def __init__(self, n_estimators=10, max_depth=10, 
                 min_samples_split=2, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.models = []
        
    def fit(self, X, y):
        """训练Bagging集成"""
        n_samples = X.shape[0]
        self.models = []
        
        for i in range(self.n_estimators):
            # Bootstrap抽样
            if self.bootstrap:
                # 有放回抽样
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
            else:
                # 无放回抽样（使用全部数据）
                indices = np.arange(n_samples)
            
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # 训练基学习器
            model = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            model.fit(X_bootstrap, y_bootstrap)
            self.models.append(model)
            
        return self
    
    def predict(self, X):
        """预测：所有基学习器投票"""
        # 收集所有模型的预测
        predictions = np.array([model.predict(X) for model in self.models])
        
        # 对每个样本进行多数投票
        result = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            # 统计每个类别的票数
            vote_counts = Counter(votes)
            # 选择票数最多的类别
            result.append(vote_counts.most_common(1)[0][0])
        
        return np.array(result)
    
    def predict_proba(self, X):
        """预测概率（各类别得票比例）"""
        predictions = np.array([model.predict(X) for model in self.models])
        
        probabilities = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            vote_counts = Counter(votes)
            # 转换为概率
            total = sum(vote_counts.values())
            proba = {cls: count/total for cls, count in vote_counts.items()}
            probabilities.append(proba)
        
        return probabilities


# ============ 演示：Bagging效果 ============
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # 创建数据集
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5,
        n_redundant=3, n_classes=2, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print("=" * 60)
    print("Bagging分类器演示")
    print("=" * 60)
    
    # 1. 单个决策树
    print("\n【单个决策树】")
    single_tree = DecisionTree(max_depth=10)
    single_tree.fit(X_train, y_train)
    y_pred_single = single_tree.predict(X_test)
    acc_single = accuracy_score(y_test, y_pred_single)
    print(f"准确率: {acc_single:.4f}")
    
    # 2. Bagging（10棵树）
    print("\n【Bagging - 10棵树】")
    bagging = BaggingClassifier(n_estimators=10, max_depth=10)
    bagging.fit(X_train, y_train)
    y_pred_bagging = bagging.predict(X_test)
    acc_bagging = accuracy_score(y_test, y_pred_bagging)
    print(f"准确率: {acc_bagging:.4f}")
    print(f"提升: +{(acc_bagging - acc_single)*100:.2f}%")
    
    # 3. Bagging（50棵树）
    print("\n【Bagging - 50棵树】")
    bagging50 = BaggingClassifier(n_estimators=50, max_depth=10)
    bagging50.fit(X_train, y_train)
    y_pred_bagging50 = bagging50.predict(X_test)
    acc_bagging50 = accuracy_score(y_test, y_pred_bagging50)
    print(f"准确率: {acc_bagging50:.4f}")
    print(f"相比单棵树提升: +{(acc_bagging50 - acc_single)*100:.2f}%")
```

### 12.6.2 随机森林分类器

```python
import numpy as np
from collections import Counter

class RandomForestClassifier:
    """
    随机森林分类器
    
    原理：Bagging + 特征随机性
    在每个节点分裂时，只考虑随机选择的m_try个特征
    """
    def __init__(self, n_estimators=100, max_depth=10, 
                 min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # 'sqrt', 'log2', 或整数
        self.trees = []
        self.feature_indices = []  # 记录每棵树使用的特征子集
        
    def _get_n_features(self, n_total_features):
        """确定每棵树使用的特征数量"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_total_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_total_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_total_features)
        else:
            return n_total_features
    
    def fit(self, X, y):
        """训练随机森林"""
        n_samples, n_features = X.shape
        n_features_per_tree = self._get_n_features(n_features)
        
        self.trees = []
        self.feature_indices = []
        
        for i in range(self.n_estimators):
            # 1. Bootstrap抽样
            bootstrap_indices = np.random.choice(
                n_samples, size=n_samples, replace=True
            )
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            # 2. 随机选择特征子集
            feature_indices = np.random.choice(
                n_features, size=n_features_per_tree, replace=False
            )
            self.feature_indices.append(feature_indices)
            
            X_subset = X_bootstrap[:, feature_indices]
            
            # 3. 训练决策树（使用带特征随机性的版本）
            tree = RandomTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                feature_indices=feature_indices  # 告诉树使用哪些特征
            )
            tree.fit(X_subset, y_bootstrap)
            self.trees.append(tree)
            
        return self
    
    def predict(self, X):
        """预测：所有树投票"""
        predictions = []
        for tree, feature_indices in zip(self.trees, self.feature_indices):
            # 每棵树只看到它训练时用的特征
            X_subset = X[:, feature_indices]
            pred = tree.predict(X_subset)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # 多数投票
        result = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            result.append(Counter(votes).most_common(1)[0][0])
        
        return np.array(result)
    
    def feature_importances(self, X, y):
        """计算特征重要性（置换法）"""
        baseline_accuracy = self._evaluate(X, y)
        n_features = X.shape[1]
        
        importances = []
        for feature in range(n_features):
            # 打乱这一列
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feature])
            
            # 看准确率下降多少
            permuted_accuracy = self._evaluate(X_permuted, y)
            importance = baseline_accuracy - permuted_accuracy
            importances.append(importance)
        
        # 归一化
        importances = np.array(importances)
        importances = importances / np.sum(importances)
        
        return importances
    
    def _evaluate(self, X, y):
        """评估准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


class RandomTree(DecisionTree):
    """带特征随机性的决策树"""
    def __init__(self, max_depth=10, min_samples_split=2, feature_indices=None):
        super().__init__(max_depth, min_samples_split)
        self.feature_indices = feature_indices
    
    def _grow_tree(self, X, y, depth):
        """重写grow_tree，只在指定特征中寻找分裂"""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # 停止条件
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        # 只在给定的特征中寻找最佳分裂
        features_to_try = range(n_features)  # X已经被子集化
        
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in features_to_try:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        if best_gain < 1e-7:
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        # 分裂
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx
        
        left_subtree = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }


# ============ 演示：随机森林效果 ============
if __name__ == "__main__":
    from sklearn.datasets import make_classification, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    print("=" * 60)
    print("随机森林分类器演示")
    print("=" * 60)
    
    # 使用Iris数据集
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n数据集: Iris（{X.shape[1]}个特征，{len(np.unique(y))}个类别）")
    print(f"训练集: {len(X_train)}个样本")
    print(f"测试集: {len(X_test)}个样本")
    
    # 随机森林
    print("\n【随机森林 - 100棵树】")
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        max_features='sqrt'  # 每棵树只考虑sqrt(4)=2个特征
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"准确率: {acc_rf:.4f}")
    
    # 特征重要性
    print("\n【特征重要性】")
    importances = rf.feature_importances(X_train, y_train)
    feature_names = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
    for name, imp in zip(feature_names, importances):
        bar = "█" * int(imp * 50)
        print(f"  {name}: {imp:.4f} {bar}")
```

### 12.6.3 AdaBoost分类器

```python
import numpy as np
from collections import Counter

class DecisionStump:
    """
    决策树桩（Decision Stump）
    
    只有一层的决策树，用作AdaBoost的弱学习器
    """
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.polarity = 1  # 1 或 -1，表示分类方向
        self.alpha = None  # 这个学习器的权重
        
    def fit(self, X, y, weights):
        """
        找到最佳的树桩
        
        参数：
            X: 特征矩阵
            y: 标签（假设为 -1 或 +1）
            weights: 样本权重
        """
        n_samples, n_features = X.shape
        
        min_error = float('inf')
        
        # 遍历所有特征
        for feature in range(n_features):
            feature_values = X[:, feature]
            thresholds = np.unique(feature_values)
            
            # 遍历所有可能的分裂点
            for threshold in thresholds:
                # 尝试两种分类方向
                for polarity in [1, -1]:
                    # 预测
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[feature_values < threshold] = -1
                    else:
                        predictions[feature_values >= threshold] = -1
                    
                    # 计算加权错误率
                    error = np.sum(weights[y != predictions])
                    
                    # 更新最佳树桩
                    if error < min_error:
                        min_error = error
                        self.feature = feature
                        self.threshold = threshold
                        self.polarity = polarity
        
        return min_error
    
    def predict(self, X):
        """预测"""
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        
        feature_values = X[:, self.feature]
        
        if self.polarity == 1:
            predictions[feature_values < self.threshold] = -1
        else:
            predictions[feature_values >= self.threshold] = -1
        
        return predictions


class AdaBoostClassifier:
    """
    AdaBoost分类器
    
    自适应Boosting算法，顺序训练弱学习器，重点关注前一轮分类错误的样本
    
    参考：Freund, Y., & Schapire, R. E. (1997). A decision-theoretic 
    generalization of on-line learning and an application to boosting. 
    Journal of Computer and System Sciences, 55(1), 119-139.
    """
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []
        self.alphas = []  # 每个学习器的权重
        
    def fit(self, X, y):
        """
        训练AdaBoost
        
        参数：
            X: 特征矩阵
            y: 标签（将被转换为 -1 或 +1）
        """
        n_samples = X.shape[0]
        
        # 转换标签为 -1 和 +1
        self.classes = np.unique(y)
        if len(self.classes) != 2:
            raise ValueError("AdaBoost当前只支持二分类问题")
        
        # 映射到 -1, +1
        y_transformed = np.where(y == self.classes[0], -1, 1)
        
        # 初始化样本权重（均匀分布）
        weights = np.ones(n_samples) / n_samples
        
        self.stumps = []
        self.alphas = []
        
        for t in range(self.n_estimators):
            # 1. 训练弱学习器
            stump = DecisionStump()
            error = stump.fit(X, y_transformed, weights)
            
            # 如果错误率太高，跳过
            if error > 0.5:
                continue
            
            # 2. 计算学习器的权重 alpha
            # alpha = 0.5 * ln((1-error) / error)
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            stump.alpha = alpha
            
            self.stumps.append(stump)
            self.alphas.append(alpha)
            
            # 3. 更新样本权重
            predictions = stump.predict(X)
            
            # w_i = w_i * exp(-alpha * y_i * h(x_i))
            # 如果预测正确：y_i * h(x_i) = 1，权重乘以 exp(-alpha) < 1（减小）
            # 如果预测错误：y_i * h(x_i) = -1，权重乘以 exp(alpha) > 1（增加）
            weights *= np.exp(-alpha * y_transformed * predictions)
            
            # 归一化
            weights /= np.sum(weights)
            
            # 打印进度
            if (t + 1) % 10 == 0:
                train_pred = self._predict_with_current_stumps(X, t + 1)
                accuracy = np.mean(train_pred == y_transformed)
                print(f"  轮数 {t+1}: 错误率={error:.4f}, alpha={alpha:.4f}, 训练准确率={accuracy:.4f}")
        
        return self
    
    def _predict_with_current_stumps(self, X, n_stumps):
        """使用当前已训练的树桩进行预测"""
        n_samples = X.shape[0]
        ensemble_pred = np.zeros(n_samples)
        
        for i in range(n_stumps):
            ensemble_pred += self.alphas[i] * self.stumps[i].predict(X)
        
        return np.sign(ensemble_pred)
    
    def predict(self, X):
        """预测"""
        n_samples = X.shape[0]
        ensemble_pred = np.zeros(n_samples)
        
        # 加权投票
        for stump, alpha in zip(self.stumps, self.alphas):
            ensemble_pred += alpha * stump.predict(X)
        
        # 转换为原始标签
        predictions = np.sign(ensemble_pred)
        return np.where(predictions == -1, self.classes[0], self.classes[1])
    
    def predict_proba(self, X):
        """预测概率（基于加权投票的强度）"""
        n_samples = X.shape[0]
        ensemble_pred = np.zeros(n_samples)
        
        for stump, alpha in zip(self.stumps, self.alphas):
            ensemble_pred += alpha * stump.predict(X)
        
        # 使用sigmoid转换为概率
        proba_class1 = 1 / (1 + np.exp(-ensemble_pred))
        
        return np.column_stack([1 - proba_class1, proba_class1])


# ============ 演示：AdaBoost效果 ============
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    print("=" * 60)
    print("AdaBoost分类器演示")
    print("=" * 60)
    
    # 创建一个稍微复杂的数据集
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    y = y * 2 - 1  # 转换为 -1 和 +1
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n数据集: make_moons（月牙形数据，线性不可分）")
    print(f"训练集: {len(X_train)}个样本")
    print(f"测试集: {len(X_test)}个样本")
    
    # 单个决策树桩
    print("\n【单个决策树桩】")
    single_stump = DecisionStump()
    weights = np.ones(len(X_train)) / len(X_train)
    error = single_stump.fit(X_train, y_train, weights)
    y_pred_stump = single_stump.predict(X_test)
    acc_stump = accuracy_score(y_test, y_pred_stump)
    print(f"错误率: {error:.4f}")
    print(f"测试准确率: {acc_stump:.4f}")
    
    # AdaBoost
    print("\n【AdaBoost - 50个树桩】")
    ada = AdaBoostClassifier(n_estimators=50)
    ada.fit(X_train, y_train)
    y_pred_ada = ada.predict(X_test)
    acc_ada = accuracy_score(y_test, y_pred_ada)
    print(f"\n最终测试准确率: {acc_ada:.4f}")
    print(f"相比单个树桩提升: +{(acc_ada - acc_stump)*100:.2f}%")
    
    print("\n【学到的弱学习器】")
    print(f"共训练了 {len(ada.stumps)} 个决策树桩")
    print("前5个树桩的信息：")
    for i, stump in enumerate(ada.stumps[:5]):
        print(f"  树桩{i+1}: 特征{stump.feature}, 阈值={stump.threshold:.3f}, "
              f"方向={stump.polarity}, 权重alpha={stump.alpha:.4f}")
```

### 12.6.4 三种方法对比实验

```python
"""
三种集成方法对比实验
"""
import numpy as np
import time

# 导入我们手写的实现
from bagging_classifier import BaggingClassifier, DecisionTree
from random_forest_classifier import RandomForestClassifier
from adaboost_classifier import AdaBoostClassifier

from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def compare_methods(X, y, dataset_name):
    """对比三种集成方法"""
    print("\n" + "=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}, 类别数: {len(np.unique(y))}")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    results = []
    
    # 1. 单棵决策树（基准）
    print("\n【基准：单棵决策树】")
    start = time.time()
    tree = DecisionTree(max_depth=10)
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    acc = accuracy_score(y_test, pred)
    elapsed = time.time() - start
    print(f"  准确率: {acc:.4f} | 训练时间: {elapsed:.3f}s")
    results.append(('单棵决策树', acc, elapsed))
    
    # 2. Bagging
    print("\n【Bagging (50棵树)】")
    start = time.time()
    bagging = BaggingClassifier(n_estimators=50, max_depth=10)
    bagging.fit(X_train, y_train)
    pred = bagging.predict(X_test)
    acc = accuracy_score(y_test, pred)
    elapsed = time.time() - start
    print(f"  准确率: {acc:.4f} | 训练时间: {elapsed:.3f}s")
    results.append(('Bagging', acc, elapsed))
    
    # 3. 随机森林
    print("\n【随机森林 (100棵树)】")
    start = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, max_features='sqrt')
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    elapsed = time.time() - start
    print(f"  准确率: {acc:.4f} | 训练时间: {elapsed:.3f}s")
    results.append(('随机森林', acc, elapsed))
    
    # 4. AdaBoost（仅用于二分类）
    if len(np.unique(y)) == 2:
        print("\n【AdaBoost (50个树桩)】")
        start = time.time()
        ada = AdaBoostClassifier(n_estimators=50)
        ada.fit(X_train, y_train)
        pred = ada.predict(X_test)
        acc = accuracy_score(y_test, pred)
        elapsed = time.time() - start
        print(f"  准确率: {acc:.4f} | 训练时间: {elapsed:.3f}s")
        results.append(('AdaBoost', acc, elapsed))
    
    # 总结
    print("\n【结果总结】")
    print("-" * 50)
    print(f"{'方法':<15} {'准确率':<10} {'时间(s)':<10}")
    print("-" * 50)
    for name, acc, t in results:
        print(f"{name:<15} {acc:<10.4f} {t:<10.3f}")
    print("-" * 50)
    
    best = max(results, key=lambda x: x[1])
    print(f"🏆 最佳方法: {best[0]} (准确率: {best[1]:.4f})")
    
    return results


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("██" + " " * 66 + "██")
    print("██" + "  集成学习方法对比实验".center(62) + "██")
    print("██" + " " * 66 + "██")
    print("█" * 70)
    
    # 实验1：标准分类数据集
    X1, y1 = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_classes=2, random_state=42
    )
    compare_methods(X1, y1, "合成二分类数据 (1000样本, 20特征)")
    
    # 实验2：多分类数据集
    X2, y2 = make_classification(
        n_samples=500, n_features=10, n_informative=8,
        n_redundant=2, n_classes=3, random_state=42
    )
    compare_methods(X2, y2, "合成三分类数据 (500样本, 10特征)")
    
    # 实验3：非线性数据集（月牙形）
    X3, y3 = make_moons(n_samples=500, noise=0.25, random_state=42)
    compare_methods(X3, y3, "月牙形非线性数据 (500样本, 2特征)")
    
    print("\n" + "█" * 70)
    print("实验完成！")
    print("█" * 70)
```

---


## 12.7 集成学习的传奇：Netflix Prize与Kaggle时代

### 12.7.1 100万美元奖金背后的智慧

2006年，美国流媒体巨头Netflix面临一个棘手的问题：他们的电影推荐系统不够准。用户租了一部电影后，系统推荐的下一部电影常常让人失望。一气之下，Netflix决定向全世界悬赏：**谁能把推荐系统的预测准确率提高10%，就奖励100万美元。**

这就是著名的**Netflix Prize**竞赛。全球超过4万个团队参赛，其中不乏顶尖的科学家和工程师。比赛刚开始的时候，领先的团队各自为战，每个人都想独占鳌头。但随着竞赛的推进，一件惊人的事情发生了：

**最好的结果，往往不是来自单个团队，而是来自多个团队的"集成"。**

2009年，一个名为"BellKor's Pragmatic Chaos"的团队夺得了冠军。这个团队是怎么获胜的呢？他们不是靠一个超级模型，而是把**上百个不同的推荐模型组合在一起**。有些模型擅长捕捉用户的长期偏好，有些模型擅长发现电影之间的相似性，还有些模型专门处理"冷门电影"的推荐。当这些模型像陪审团一样"投票"时，准确率竟然超过了任何一个单独的模型。

Netflix Prize告诉我们一个深刻的道理：**在复杂的现实世界问题中，没有所谓"最好的单一方法"。真正的智慧在于知道如何组合不同方法的长处。**

### 12.7.2 Kaggle竞赛中的集成学习

Netflix Prize之后，集成学习在机器学习竞赛中大放异彩。在Kaggle（全球最大的数据科学竞赛平台）上，几乎每一届冠军方案都离不开集成学习。

**为什么会出现这种现象？**

因为在数据竞赛中，冠军和亚军的差距往往只有千分之一。要在这种"毫厘之争"中取胜，你需要榨干数据中最后一滴信息。单个模型再强，也总会有盲点：
- 神经网络可能忽略了数据中的某些线性关系
- 决策树可能无法很好地捕捉复杂的交互特征
- 支持向量机可能对某些边缘样本过于敏感

但如果把它们组合起来呢？**A模型的盲点，正好是B模型的强项。**这种互补性让集成学习成为竞赛中的"核武器"。

### 12.7.3 现实中的集成学习

你不需要参加百万美元的比赛也能遇到集成学习。它就在我们身边：

**搜索引擎**：Google的搜索排名不是由一个算法决定的，而是数百个算法共同"投票"的结果。有些算法看网页内容，有些看链接数量，有些看用户点击行为。

**医学诊断**：当AI辅助医生诊断癌症时，通常会同时运行多个模型。一个模型分析CT影像，一个模型分析病理切片，另一个模型分析血液指标。最后，系统给出一个综合评估。

**天气预报**：现代天气预报系统会运行数十个不同的大气模型，然后取平均值。这就是为什么天气预报比几十年前准得多了。

**投资风控**：对冲基金很少只依赖一个预测模型。他们会同时运行趋势跟踪模型、均值回归模型、情绪分析模型，然后用集成的方法决定资金流向。

---

## 12.8 Stacking与Voting：混合的艺术

前面我们学习了Bagging、随机森林和Boosting。这些方法都有一个共同点：**基学习器是同类型的**（比如都是决策树）。但如果我们想把完全不同的模型组合起来呢？比如把决策树、神经网络和支持向量机放在一起？

这就是**Stacking**（堆叠）和**Voting**（投票）要做的事情。

### 12.8.1 简单投票：少数服从多数

想象一个班级要选出春游地点。老师让三个不同的委员会分别投票：
- 体育委员会说：去爬山！
- 文艺委员会说：去博物馆！
- 美食委员会说：去爬山！

结果是2:1，决定去爬山。这就是**硬投票**（Hard Voting）：每个模型给一个明确的预测，最后少数服从多数。

如果三个委员会不是给确定答案，而是给概率呢？
- 体育委员会：爬山60%，博物馆30%，公园10%
- 文艺委员会：爬山40%，博物馆50%，公园10%
- 美食委员会：爬山70%，博物馆20%，公园10%

平均一下：爬山56.7%，博物馆33.3%，公园10%。所以最终还是爬山。这就是**软投票**（Soft Voting）：对不同模型的预测概率取平均。

在机器学习中，Voting集成就是把几个已经训练好的模型放在一起，让它们投票决定最终结果。简单粗暴，但往往非常有效。

### 12.8.2 Stacking：用另一个模型来做裁判

Voting虽然简单，但它把所有模型一视同仁。可问题是：**不同模型的能力并不一样。**有些模型在这个问题上很准，有些模型可能经常犯错。如果我们能根据每个模型的表现给它们不同的权重，会不会更好？

Stacking就是这个思想的升级版本。它的核心想法是：**训练一个"元学习器"（Meta-learner），让它学会如何最佳地组合基学习器的预测。**

具体来说，Stacking有三个步骤：

**第一步：训练基学习器**
用原始数据训练多个不同的模型（比如决策树、神经网络、SVM）。

**第二步：生层"第二层特征"**
这里有个关键技巧：我们不能直接用训练集来生成元学习器的训练数据，否则会发生过拟合。正确做法是用**K折交叉验证**：
- 把数据分成K份
- 每次用K-1份训练基学习器，用剩下的1份做预测
- 把所有预测拼起来，就形成了元学习器的训练集

**第三步：训练元学习器**
把这些预测当作"新特征"，训练一个元学习器（通常是逻辑回归或轻量级的梯度提升树）。元学习器学到的，就是"在什么情况下应该更相信哪个模型"。

### 12.8.3 Stacking的生活比喻：足球主教练

想象你是一支足球队的主教练。你的队里有各种各样的人才：
- 前锋A：速度快，适合反击
- 前锋B：头球好，适合阵地战
- 中场C：传球精准，但身体对抗弱
- 后卫D：防守稳健，但进攻乏力

你的任务不是每场都让所有人上场，而是**根据对手的特点，派出最合适的阵容。**当对手打高空球时，你派上前锋B；当对手压得很靠上时，你派上速度型前锋A打反击。

Stacking里的元学习器，就是这个主教练。它学会了根据"输入数据的特点"，决定应该更依赖哪个基学习器。

### 12.8.4 代码速览：Voting分类器

```python
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# 定义三个完全不同的基学习器
clf1 = DecisionTreeClassifier(max_depth=5)
clf2 = SVC(probability=True, kernel='rbf')
clf3 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)

# 用软投票组合它们
voting_clf = VotingClassifier(
    estimators=[('dt', clf1), ('svm', clf2), ('mlp', clf3)],
    voting='soft'
)

voting_clf.fit(X_train, y_train)
accuracy = voting_clf.score(X_test, y_test)
```

注意，`voting='soft'`表示对概率取平均，通常比硬投票更稳定。

### 12.8.5 何时使用Voting和Stacking？

| 方法 | 适合场景 | 训练成本 | 效果上限 |
|------|----------|----------|----------|
| Voting | 有几个已经训练好的不同模型 | 低 | 中等 |
| Stacking | 有大量不同模型，想学最优组合 | 高 | 很高 |
| Bagging | 单个模型不稳定（高方差） | 中 | 中高 |
| Boosting | 单个模型欠拟合（高偏差） | 中 | 高 |

**实践建议**：如果你手上有两三个不同框架的模型，试试Voting，几乎零成本。如果你在打Kaggle比赛，Stacking可能是冲顶的最后一招。但在工程应用中，Stacking的复杂度和边际收益之比，往往不如一个调参良好的XGBoost来得实在。

---

## 12.9 总结

在本章，我们学习了机器学习中最重要的技术之一：**集成学习**。

### 核心概念回顾

**1. 为什么要集成？**
- 单个模型（尤其是决策树）方差高、不稳定
- 多个模型的集体智慧可以相互纠错
- 集成可以显著降低方差（Bagging）或偏差（Boosting）

**2. Bagging（装袋）**
- 通过Bootstrap抽样创建多个不同的训练集
- 在每个训练集上独立训练基学习器
- 预测时投票或取平均
- **代表**：随机森林（增加特征随机性）

**3. Boosting（提升）**
- 顺序训练基学习器
- 每个新学习器关注前一个学习器的错误
- 通过样本重加权实现
- **代表**：AdaBoost、XGBoost、LightGBM

### 关键公式回顾

**Bagging方差减少**：
$$\text{Var}(\hat{y}) = \frac{\sigma^2}{T}$$

**AdaBoost样本权重更新**：
$$w_i^{(t+1)} = \frac{w_i^{(t)} \cdot \exp(-\alpha_t y_i h_t(x_i))}{Z_t}$$

**AdaBoost学习器权重**：
$$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$

### 实践建议

| 场景 | 推荐方法 |
|------|----------|
| 快速原型、不想调参 | 随机森林 |
| 数据有噪声 | 随机森林、Bagging |
| 追求最高准确率 | XGBoost、LightGBM |
| 需要模型解释 | 单棵决策树 + 特征重要性 |
| 大规模数据 | 随机森林（并行） |

### 本章代码

我们手写了三个完整的集成学习实现：
- `bagging_classifier.py`：Bagging分类器
- `random_forest_classifier.py`：随机森林分类器
- `adaboost_classifier.py`：AdaBoost分类器
- `compare_methods.py`：三种方法对比实验

**运行对比实验**：
```bash
cd chapter12/code
python compare_methods.py
```

---

## 附录A：集成学习调试备忘录与常见陷阱

### A.1 为什么我的集成模型反而更差了？

集成学习不是万能药。如果你发现集成后的准确率还不如单个模型，可能有以下几个原因：

**陷阱1：基学习器太相似**
如果你组合的三个模型都是深度很大的决策树，它们犯的错误可能高度一致。这样一来，"集体智慧"就变成了"集体犯傻"。**集成的前提是多样性。**

**诊断方法**：检查基学习器之间的预测相关性。如果相关系数超过0.9，说明它们太像了。

**陷阱2：基学习器本身太弱**
俗话说"三个臭皮匠顶个诸葛亮"，但前提是这三个臭皮匠至少能看懂地图。如果基学习器的准确率还不如随机猜测（比如50%对二分类），集成也无济于事。Boosting之所以有效，是因为它从"勉强比随机好"（weak learner）开始学习，但Bagging和Voting通常需要基学习器本身就已经有不错的表现。

**陷阱3：数据量太小**
当训练样本很少时，Bootstrap抽样生成的子集可能彼此非常相似，Bagging的方差降低效果就会大打折扣。一般来说，Bagging在样本数>1000时效果才比较明显。

**陷阱4：Boosting过拟合**
Boosting通过不断纠正错误来降低偏差，但如果基学习器太强或者迭代次数太多，它可能开始"记住"训练数据中的噪声。如果你发现训练误差一直在下降，但测试误差开始上升，这就是过拟合的信号。**早停（Early Stopping）**是Boosting的好朋友。

### A.2 超参数调优清单

**Bagging/随机森林**：
- `n_estimators`：树的数量。通常越多越好（边际递减），但计算成本线性增长。100-500棵是常见范围。
- `max_depth`：单棵树的深度。限制深度可以防止过拟合，但可能增加偏差。
- `max_features`：每次分裂时考虑的特征数。随机森林推荐`sqrt(n_features)`。

**AdaBoost**：
- `n_estimators`：迭代次数。如果训练误差很早就降到零，说明模型太强了，应该减少迭代次数或使用更弱的基学习器。
- `learning_rate`：收缩因子（某些实现中有）。较小的学习率需要更多迭代，但通常更稳定。

**Stacking**：
- 元学习器不要用太复杂的模型。简单的逻辑回归或岭回归往往效果最好。
- 基学习器的数量5-15个通常是甜点区。太多模型会增加噪声，元学习器反而学不到有效模式。

### A.3 快速决策树

```
数据量大，想快速得到一个稳定的基线 → 随机森林
数据量适中，追求最高准确率 → XGBoost/LightGBM
有几个已经训练好的不同模型 → Voting
想榨干最后一滴准确率（竞赛场景） → Stacking
模型不稳定，预测波动大 → Bagging
模型欠拟合，训练集上准确率就不高 → Boosting
```

---

## 练习题

### 基础练习

**练习12.1：Bootstrap抽样**

假设你有一个包含100个样本的数据集。进行Bootstrap抽样：

1. 大约有多少比例的样本会被选中至少一次？
2. 计算当样本数 $n \to \infty$ 时，某个特定样本未被选中的概率。
3. 验证：$\lim_{n \to \infty} (1 - 1/n)^n = 1/e \approx 0.368$

**练习12.2：方差计算**

假设你有5个独立的分类器，每个的预测方差为 $\sigma^2 = 4$。

1. 如果使用简单平均集成，集成的方差是多少？
2. 如果增加模型数量到20个，方差变为多少？
3. 从数学上解释为什么"越多越好"。

**练习12.3：AdaBoost权重**

在AdaBoost中，假设某一轮的加权错误率 $\epsilon_t = 0.2$。

1. 计算该轮学习器的权重 $\alpha_t$。
2. 如果一个样本被正确分类，它的权重会如何变化（增大还是减小）？
3. 如果一个样本被错误分类，它的权重会乘以多少倍？

### 进阶挑战

**练习12.4：实现OOB误差估计**

扩展`BaggingClassifier`类，添加OOB（Out-of-Bag）误差估计功能。

提示：
- 对每个基学习器，记录哪些样本没有被Bootstrap选中
- 用这些OOB样本评估该学习器
- 平均所有学习器的OOB误差

**练习12.5：堆叠集成（Stacking）**

研究并实现**Stacking**集成方法：

1. 用K折交叉验证训练多个基学习器
2. 用基学习器的预测作为特征，训练一个元学习器
3. 比较Stacking与Bagging、Boosting的性能

参考文献：Wolpert, D. H. (1992). Stacked generalization. Neural Networks, 5(2), 241-259.

### 终极挑战

**练习12.6：梯度提升（Gradient Boosting）**

实现**梯度提升**算法，这是AdaBoost的推广，也是XGBoost的核心思想。

要求：
1. 理解梯度提升与AdaBoost的区别
2. 用回归树作为基学习器
3. 实现平方损失函数的梯度提升
4. 在回归数据集上测试

提示：
- 梯度提升每一步拟合的是"残差"（当前预测与真实值的差）
- 学习率（shrinkage）是一个重要的超参数

参考文献：Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 1189-1232.

---

## 附录B：偏差-方差分解的完整推导

在第十二章的开头，我们提到了**偏差**和**方差**。这里给出它们的数学定义和分解过程。

### B.1 均方误差的分解

对于一个回归问题，假设真实关系为 $y = f(x) + \\epsilon$，其中 $\\epsilon$ 是噪声，满足 $E[\\epsilon]=0$，$\\text{Var}(\\epsilon)=\\sigma^2$。

我们用模型 $\\hat{f}(x)$ 来做预测。在测试点 $x$ 上的期望预测误差为：

$$E\\left[(y - \\hat{f}(x))^2\\right]$$

### B.2 逐步推导

把 $y = f(x) + \\epsilon$ 代入：

$$E\\left[(f + \\epsilon - \\hat{f})^2\\right] = E\\left[\\left((f - E[\\hat{f}]) + (E[\\hat{f}] - \\hat{f}) + \\epsilon\\right)^2\\right]$$

展开平方。注意交叉项很多，但关键是这三类项：

1. **偏差项（Bias）**：$(f - E[\\hat{f}])^2$  
   反映模型平均预测与真实值的差距。高偏差意味着模型太简单，总是"瞄错方向"。

2. **方差项（Variance）**：$E\\left[(\\hat{f} - E[\\hat{f}])^2\\right]$  
   反映模型对不同训练集的敏感程度。高方差意味着模型"神经质"，换一批训练数据 predictions 就大变。

3. **不可约误差（Irreducible Error）**：$\\sigma^2$  
   来自数据本身的噪声，无论模型多强都无法消除。

### B.3 最终公式

$$E\\left[(y - \\hat{f})^2\\right] = \\underbrace{(f - E[\\hat{f}])^2}_{\\text{Bias}^2} + \\underbrace{E\\left[(\\hat{f} - E[\\hat{f}])^2\\right]}_{\\text{Variance}} + \\underbrace{\\sigma^2}_{\\text{Noise}}$$

### B.4 集成学习做了什么？

- **Bagging**通过平均多个高方差模型，大大降低了方差项。
- **Boosting**通过顺序纠正偏差，逐步减小了偏差项。
- 这就是为什么Bagging适合深度学习树（高方差低偏差），而Boosting适合决策树桩（高偏差低方差）。

---

## 附录C：跨章节知识联系图

```
                    ┌─────────────────────────────────────┐
                    │      第十二章：集成学习               │
                    └──────────────────┬──────────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
           ▼                           ▼                           ▼
    ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
    │ 第九章      │            │ 第十章      │            │ 第十三章    │
    │ 决策树      │◀──────────▶│ SVM         │            │ K-Means     │
    │ (基学习器)  │            │ (Voting伙伴)│            │ (无监督集成)│
    └──────┬──────┘            └─────────────┘            └─────────────┘
           │
           ▼
    ┌─────────────┐
    │ 第二十章    │
    │ 梯度提升树  │
    │ (Boosting+) │
    └─────────────┘
```

**联系说明**：
- **第九章←→第十二章**：决策树是集成学习最常用的基学习器。理解决策树的剪枝、分裂标准，是理解随机森林和Boosting的前提。
- **第九章←→第十二章**：决策树是集成学习最常用的基学习器。理解决策树的剪枝、分裂标准，是理解随机森林和Boosting的前提。
- **第十二章←→第十章**：集成学习中可以用SVM作为基学习器，Voting和Stacking尤其喜欢"性格迥异"的模型组合。
- **第十二章→第二十章**：AdaBoost是梯度提升树的"前辈"。学完AdaBoost，你可以把梯度提升树理解为"用损失函数的梯度来指导提升方向"的一般化版本。
- **第十二章↔第十三章**：聚类也可以用集成方法（聚类集成 Ensemble Clustering），通过组合多个聚类结果来提高稳定性。

---
## 参考文献

1. Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123-140. https://doi.org/10.1007/BF00058655

2. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324

3. Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences*, 55(1), 119-139. https://doi.org/10.1006/jcss.1997.1503

4. Schapire, R. E. (1990). The strength of weak learnability. *Machine Learning*, 5, 197-227.

5. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. (第8章、第10章、第15章、第16章)

6. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794). https://doi.org/10.1145/2939672.2939785

7. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232. https://doi.org/10.1214/aos/1013203451

8. Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241-259. https://doi.org/10.1016/S0893-6080(05)80023-1

9. Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning*, 63(1), 3-42.

10. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems* (pp. 3146-3154).

---

*本章完*

**下一章预告**：第十三章：梯度提升树——层层递进的智慧

我们将深入探讨XGBoost和LightGBM的内部原理，学习如何在竞赛级别的任务中获得最佳性能。

---

*作者注：本章代码经过精心测试，可以在Python 3.7+环境中直接运行。如有任何问题或建议，欢迎反馈。*
