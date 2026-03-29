# 第九章：决策树——像专家一样做决策

## 费曼四步检验框 📋
> **目标**: 读完本章后，你能向一个10岁的小朋友解释清楚决策树是怎么像专家一样做决定的。
> 
> 1. **选一个概念** ✓
> 2. **假装教给一个孩子** ✓
> 3 **发现理解缺口** → 看本章内容
> 4. **简化和类比** → 用"二十个问题"游戏理解

---

## 9.1 故事：神奇的诊断专家 🏥

### 9.1.1 小动物医院的故事

在一个阳光明媚的下午，小动物医院来了一位"神秘医生"。这位医生有个特别的能力——**只需要问几个简单的问题，就能准确判断出动物得了什么病**。

"你的宠物是不是呕吐了？"医生问。

"是的。"主人回答。

"体温高吗？"

"有点发热。"

"食欲怎么样？"

"完全不吃东西。"

"那应该是肠胃炎，我开点药，三天就好。"医生自信地说。

果然，三天后小动物康复了！

主人惊讶地问："您是怎么做到的？问几个问题就知道病因？"

医生微笑着说："我脑子里有棵**神奇的树**。每次你回答一个问题，我就顺着树枝往下走，最后到达的叶子就是答案。"

"这是什么魔法吗？"主人好奇地问。

"不，这是**决策树**——机器学习中最像人类思考方式的算法！"

---

### 9.1.2 我们每天都在用决策树

其实，**你每天都在用决策树做决策**，只是你没意识到：

**早上出门的决策树**：

```
外面下雨吗？
├── 是 → 带伞
│   └── 雨大吗？
│       ├── 是 → 穿雨衣
│       └── 否 → 带小伞
└── 否 → 不带伞
    └── 太阳大吗？
        ├── 是 → 戴帽子
        └── 否 → 直接出门
```

**选择午餐的决策树**：

```
饿吗？
├── 很饿 → 吃米饭
│   └── 想吃什么菜？
│       ├── 辣 → 麻辣烫
│       └── 不辣 → 盖浇饭
└── 有点饿 → 吃面条
    └── 时间够吗？
        ├── 够 → 拉面
        └── 不够 → 方便面
```

看！**决策树就是把我们做决策的过程画成一棵树**。每个分叉都是一个"是/否"问题，顺着问题走，最后就能到达决定。

**问题来了**：如果机器能像专家一样，从大量例子中**自动学会**这些决策问题，那该多厉害！

这就是**决策树算法**要做的事！

---

## 9.2 历史之旅：从猜谜游戏到机器学习 📜

### 9.2.1 "二十个问题"的智慧

决策树的思想可以追溯到古老的**"二十个问题"**游戏：

> 一个人想一个东西，其他人通过最多20个"是/否"问题来猜出它是什么。

**聪明的提问者**会先问："它是活的吗？"（把可能性切成两半）

**笨笨的提问者**可能会问："它是小明家的那只猫吗？"（万一不是，就浪费了一个问题）

**核心智慧**：每个问题都应该**最大程度地缩小可能性范围**！

1960年代，心理学家**Earl Hunt**和同事们开始研究人类是如何学习的。他们发现，人们做分类决策时，脑子里好像真的有棵"树" [Hunt et al., 1966]。

### 9.2.2 ID3的诞生：Ross Quinlan的天才想法

1970年代末期，在澳大利亚悉尼大学，一位叫**Ross Quinlan**的计算机科学家面临一个难题：

> 如何教计算机从大量例子中**自动学会**做决策？

Quinlan当时在研究专家系统。传统的专家系统需要人类专家手动编写规则，费时费力。Quinlan想："能不能让计算机**自己从数据中学出规则**？"

**关键洞察**：如果每个问题都能"最大程度地区分不同类别"，那么整棵树就会很高效！

1986年，Quinlan发表了著名的**ID3算法**（Iterative Dichotomiser 3，迭代二分器3号）：

> **Quinlan, J. R. (1986). Induction of Decision Trees. *Machine Learning*, 1(1), 81-106.**

这是机器学习领域的开创性论文！ID3成为了第一个广泛使用的决策树学习算法。

### 9.2.3 信息论的启示：香农的遗产

Quinlan从哪里得到灵感？

从**信息论**！1948年，**Claude Shannon**（香农）在贝尔实验室发表了一篇改变世界的论文，开创了信息论 [Shannon, 1948]。

香农提出了**熵**（Entropy）的概念——用来衡量信息的"混乱程度"：

- 熵高 = 混乱 = 不确定性大
- 熵低 = 有序 = 确定性高

Quinlan的天才之处：用**信息增益**（Information Gain）来决定问什么问题！

> **信息增益 = 分裂前的熵 - 分裂后的平均熵**

每次选择那个能带来**最大信息增益**的问题！

### 9.2.4 C4.5：ID3的进化版

ID3虽然厉害，但有局限：
- 只能处理离散特征
- 不能处理缺失值
- 容易过拟合（树长得太深）

1993年，Quinlan推出了**C4.5算法**，在他的经典著作中详细描述：

> **Quinlan, J. R. (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann Publishers.**

C4.5的重大改进：
- ✓ 处理**连续特征**（比如身高、体重）
- ✓ 处理**缺失值**
- ✓ **剪枝**防止过拟合
- ✓ 使用**信息增益率**（Gain Ratio）避免偏向多值属性

C4.5成为了数据挖掘领域最经典的算法之一！

### 9.2.5 CART：另一条路的先驱

就在Quinlan开发ID3的同时，美国加州大学伯克利分校的统计学家们也在研究决策树。

1984年，**Leo Breiman**、**Jerome Friedman**、**Charles Stone**和**Richard Olshen**发表了另一部经典：

> **Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). Classification and Regression Trees. Wadsworth & Brooks.**

这就是**CART算法**（Classification and Regression Trees）：

- 使用**Gini指数**而不是信息增益
- 总是产生**二叉树**（每个节点只有 two branches）
- 既能做**分类**（Classification）又能做**回归**（Regression）

CART后来成为了随机森林的基础！

### 9.2.6 决策树的时间线

```
1948  Claude Shannon发表信息论
  │
1963  Morgan & Sonquist: AID算法
  │
1966  Hunt et al.: CLS概念学习系统
  │
1970s Ross Quinlan开始开发ID3
  │
1984  Breiman等发表CART
  │
1986  Quinlan发表ID3论文
  │
1993  Quinlan发表C4.5
  │
2001  Breiman提出随机森林
  │
今天  决策树仍是机器学习的核心算法！
```

---

## 9.3 核心概念：熵与信息增益 🔥

### 9.3.1 什么是熵？混乱的度量

想象两个班级：

**班级A**：30个学生，15个男生，15个女生

**班级B**：30个学生，29个男生，1个女生

如果你随机选一个学生猜性别：
- 在班级A，你最没把握（男女各半）
- 在班级B，你很有把握（大概率是男生）

**熵**就是用来衡量这种"不确定性"的！

### 9.3.2 熵的公式

对于包含 $n$ 个类别的数据集 $S$，熵的计算公式是：

$$Entropy(S) = -\sum_{i=1}^{n} p_i \log_2(p_i)$$

其中：
- $p_i$ = 第 $i$ 类样本所占比例
- $\log_2$ = 以2为底的对数

**举个例子**：

班级A：15个男生，15个女生
- $p_{男} = 15/30 = 0.5$
- $p_{女} = 15/30 = 0.5$
- $Entropy = -(0.5 \times \log_2(0.5) + 0.5 \times \log_2(0.5))$
- $= -(0.5 \times (-1) + 0.5 \times (-1))$
- $= -(-0.5 - 0.5) = 1.0$

班级B：29个男生，1个女生
- $p_{男} = 29/30 \approx 0.967$
- $p_{女} = 1/30 \approx 0.033$
- $Entropy = -(0.967 \times \log_2(0.967) + 0.033 \times \log_2(0.033))$
- $\approx -(0.967 \times (-0.048) + 0.033 \times (-4.93))$
- $\approx -(-0.046 - 0.163) = 0.21$

**结论**：班级A熵高（混乱），班级B熵低（有序）！

### 9.3.3 信息增益：选择最佳分裂

现在假设我们要根据"身高"来分裂班级：

**分裂前**：班级整体熵 = 1.0

**分裂后**：
- 高个子组：10人，8男2女，熵 = 0.72
- 矮个子组：20人，7男13女，熵 = 0.93

**信息增益** = 分裂前熵 - 分裂后平均熵

$$Gain(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v)$$

计算：
- 分裂后平均熵 = $(10/30) \times 0.72 + (20/30) \times 0.93 = 0.24 + 0.62 = 0.86$
- 信息增益 = $1.0 - 0.86 = 0.14$

**信息增益越大，说明这个分裂越有效！**

### 9.3.4 为什么要用log₂？

你可能好奇：为什么用$\log_2$而不是其他对数？

因为**信息论**中，信息量的单位是**比特**（bit）！

想象猜一个1到8之间的数字：
- 最优策略是二分法："大于4吗？"→"大于6吗？"→... 
- 3个问题一定能猜中
- $2^3 = 8$，所以需要3比特信息

$\log_2(8) = 3$ ✓

这就是信息熵的直观意义！

---

## 9.4 ID3算法：从数据长出一棵树 🌳

### 9.4.1 算法步骤

ID3算法的核心思想很简单：

```python
"""ID3算法伪代码"""
def ID3(数据集, 特征集, 类别标签):
    # 1. 如果所有样本都是同一类，返回叶子节点
    if 所有样本类别相同:
        return 叶子节点(该类别)
    
    # 2. 如果没有特征可用了，返回叶子节点（多数类）
    if 特征集为空:
        return 叶子节点(数据集中最多的类别)
    
    # 3. 选择信息增益最大的特征
    最佳特征 = argmax(计算每个特征的信息增益)
    
    # 4. 以最佳特征创建节点
    节点 = 创建节点(最佳特征)
    
    # 5. 对每个特征值，递归构建子树
    for 特征值 in 最佳特征的所有可能值:
        子数据集 = 数据集中该特征等于特征值的样本
        if 子数据集为空:
            子树 = 叶子节点(父节点多数类)
        else:
            子树 = ID3(子数据集, 特征集-最佳特征, 类别标签)
        将子树连接到节点
    
    return 节点
```

### 9.4.2 实际例子：玩不玩网球？

让我们用经典的"网球数据集"来演示：

| 天气 | 温度 | 湿度 | 风力 | 玩网球？ |
|------|------|------|------|----------|
| 晴 | 热 | 高 | 弱 | 否 |
| 晴 | 热 | 高 | 强 | 否 |
| 多云 | 热 | 高 | 弱 | 是 |
| 雨 | 温和 | 高 | 弱 | 是 |
| 雨 | 凉 | 正常 | 弱 | 是 |
| 雨 | 凉 | 正常 | 强 | 否 |
| 多云 | 凉 | 正常 | 强 | 是 |
| 晴 | 温和 | 高 | 弱 | 否 |
| 晴 | 凉 | 正常 | 弱 | 是 |
| 雨 | 温和 | 正常 | 弱 | 是 |
| 晴 | 温和 | 正常 | 强 | 是 |
| 多云 | 温和 | 高 | 强 | 是 |
| 多云 | 热 | 正常 | 弱 | 是 |
| 雨 | 温和 | 高 | 强 | 否 |

**第一步：计算整体熵**

14个样本：9个"是"，5个"否"

$Entropy(S) = -(\frac{9}{14}\log_2\frac{9}{14} + \frac{5}{14}\log_2\frac{5}{14}) = 0.940$

**第二步：计算每个特征的信息增益**

**天气**特征（晴:5, 多云:4, 雨:5）：
- 晴：2是，3否，熵=0.971
- 多云：4是，0否，熵=0
- 雨：3是，2否，熵=0.971
- $Entropy(S|天气) = \frac{5}{14} \times 0.971 + \frac{4}{14} \times 0 + \frac{5}{14} \times 0.971 = 0.694$
- $Gain(S, 天气) = 0.940 - 0.694 = 0.246$

**湿度**特征（高:7, 正常:7）：
- 高：3是，4否，熵=0.985
- 正常：6是，1否，熵=0.592
- $Entropy(S|湿度) = \frac{7}{14} \times 0.985 + \frac{7}{14} \times 0.592 = 0.789$
- $Gain(S, 湿度) = 0.940 - 0.789 = 0.151$

**风力**特征（弱:8, 强:6）：
- 弱：6是，2否，熵=0.811
- 强：3是，3否，熵=1.0
- $Gain(S, 风力) = 0.940 - 0.892 = 0.048$

**温度**特征（热:4, 温和:6, 凉:4）：
- 计算后 Gain = 0.029

**第三步：选择最佳特征**

| 特征 | 信息增益 |
|------|----------|
| 天气 | 0.246 ← 最大！|
| 湿度 | 0.151 |
| 风力 | 0.048 |
| 温度 | 0.029 |

**选择"天气"作为根节点！**

**第四步：递归构建**

根节点是"天气"，有三个分支：

```
天气
├── 多云 → 全是"是" → 叶子(是) ✓
├── 晴 → 继续递归
│   └── ...
└── 雨 → 继续递归
    └── ...
```

多云分支已经纯了（全是"是"），直接成为叶子！

对于"晴"分支（5个样本）：继续递归...

最终构建的决策树：

```
天气
├── 多云 → [是]
├── 晴
│   └── 湿度
│       ├── 高 → [否]
│       └── 正常 → [是]
└── 雨
    └── 风力
        ├── 弱 → [是]
        └── 强 → [否]
```

**哇！这就是决策树的威力！**

---

## 9.5 决策树的优缺点 📊

### 9.5.1 优点

1. **可解释性强** 👁️
   - 树结构一目了然
   - 可以转换成"如果...那么..."规则
   - 医生、银行、法律等需要解释的领域都爱用

2. **无需数据预处理** 🎯
   - 不需要标准化/归一化
   - 自动处理特征交互
   - 对缺失值有一定容忍度（C4.5）

3. **快速预测** ⚡
   - 预测时只需沿着树走
   - 时间复杂度：$O(\text{树深度})$

4. **能处理非线性关系** 🌊
   - 决策边界是轴平行的
   - 可以逼近任意复杂度的决策边界

### 9.5.2 缺点

1. **容易过拟合** 😰
   - 树可以一直长到每个叶子只有一个样本
   - 对新数据表现很差
   - **解决方案**：剪枝、限制深度、随机森林

2. **不稳定性** 😵
   - 数据的小小变化可能导致完全不同的树
   - **解决方案**：集成方法（随机森林、梯度提升）

3. **偏向多值属性** ⚖️
   - ID3偏向取值多的特征
   - **解决方案**：C4.5使用信息增益率

4. **只能产生轴平行分割** 📐
   - 对于斜向边界效果不佳
   - **解决方案**：Oblique决策树、其他算法

---

## 9.6 Python实战：手搓决策树！💻

### 9.6.1 从零实现ID3决策树

现在，让我们**不借助任何机器学习库**，纯Python手搓一个决策树！

```python
"""
第九章代码：决策树 - 像专家一样做决策
从零实现ID3决策树算法
"""

import math
from collections import Counter


class TreeNode:
    """决策树节点"""
    def __init__(self, feature=None, value=None, label=None, branches=None):
        self.feature = feature      # 分裂特征
        self.value = value          # 特征值（用于叶子节点）
        self.label = label          # 类别标签（叶子节点）
        self.branches = branches or {}  # 子树 {特征值: 子节点}
    
    def is_leaf(self):
        return self.label is not None
    
    def predict(self, sample):
        """预测单个样本"""
        if self.is_leaf():
            return self.label
        
        feature_value = sample.get(self.feature)
        if feature_value in self.branches:
            return self.branches[feature_value].predict(sample)
        else:
            # 如果没见过这个值，返回多数类
            return self.label or "未知"
    
    def __repr__(self):
        if self.is_leaf():
            return f"叶子({self.label})"
        return f"节点({self.feature})"


class ID3DecisionTree:
    """
    ID3决策树分类器
    
    基于Quinlan 1986年论文实现：
    "Induction of Decision Trees"
    """
    
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.classes = None
    
    @staticmethod
    def entropy(labels):
        """
        计算熵：H(S) = -Σ p_i * log2(p_i)
        
        >>> dt = ID3DecisionTree()
        >>> dt.entropy(['是', '否', '是', '否'])  # 2:2
        1.0
        >>> dt.entropy(['是', '是', '是', '否'])  # 3:1
        0.81...
        """
        if not labels:
            return 0.0
        
        n = len(labels)
        counts = Counter(labels)
        
        entropy_val = 0.0
        for count in counts.values():
            p = count / n
            entropy_val -= p * math.log2(p)
        
        return entropy_val
    
    def information_gain(self, data, feature, target_col):
        """
        计算信息增益：Gain(S, A) = Entropy(S) - Σ |S_v|/|S| * Entropy(S_v)
        
        Args:
            data: 数据集，列表 of 字典
            feature: 要计算的特征名
            target_col: 目标列名
        """
        # 整体熵
        total_entropy = self.entropy([row[target_col] for row in data])
        
        # 按特征值分组
        n = len(data)
        feature_values = {}
        for row in data:
            val = row[feature]
            if val not in feature_values:
                feature_values[val] = []
            feature_values[val].append(row[target_col])
        
        # 计算条件熵
        conditional_entropy = 0.0
        for values in feature_values.values():
            p = len(values) / n
            conditional_entropy += p * self.entropy(values)
        
        # 信息增益
        return total_entropy - conditional_entropy
    
    def _find_best_feature(self, data, features, target_col):
        """找到信息增益最大的特征"""
        best_feature = None
        best_gain = -1
        
        for feature in features:
            gain = self.information_gain(data, feature, target_col)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        
        return best_feature, best_gain
    
    def _build_tree(self, data, features, target_col, depth=0):
        """
        递归构建决策树
        
        递归终止条件：
        1. 所有样本属于同一类
        2. 没有特征可用了
        3. 达到最大深度
        4. 样本数少于最小分裂数
        """
        labels = [row[target_col] for row in data]
        
        # 条件1：所有样本类别相同
        if len(set(labels)) == 1:
            return TreeNode(label=labels[0])
        
        # 条件2：没有特征了
        if not features:
            majority = Counter(labels).most_common(1)[0][0]
            return TreeNode(label=majority)
        
        # 条件3：达到最大深度
        if self.max_depth is not None and depth >= self.max_depth:
            majority = Counter(labels).most_common(1)[0][0]
            return TreeNode(label=majority)
        
        # 条件4：样本数太少
        if len(data) < self.min_samples_split:
            majority = Counter(labels).most_common(1)[0][0]
            return TreeNode(label=majority)
        
        # 选择最佳特征
        best_feature, best_gain = self._find_best_feature(data, features, target_col)
        
        # 如果信息增益为0，停止分裂
        if best_gain <= 0:
            majority = Counter(labels).most_common(1)[0][0]
            return TreeNode(label=majority)
        
        # 创建节点
        node = TreeNode(feature=best_feature)
        node.label = Counter(labels).most_common(1)[0][0]  # 备用多数类
        
        # 按最佳特征分组
        feature_values = {}
        for row in data:
            val = row[best_feature]
            if val not in feature_values:
                feature_values[val] = []
            feature_values[val].append(row)
        
        # 递归构建子树
        remaining_features = [f for f in features if f != best_feature]
        for val, subset in feature_values.items():
            node.branches[val] = self._build_tree(
                subset, remaining_features, target_col, depth + 1
            )
        
        return node
    
    def fit(self, data, target_col):
        """
        训练决策树
        
        Args:
            data: 训练数据，列表 of 字典
            target_col: 目标列名
        """
        # 获取所有特征
        features = [col for col in data[0].keys() if col != target_col]
        self.classes = list(set(row[target_col] for row in data))
        
        # 构建树
        self.root = self._build_tree(data, features, target_col)
        return self
    
    def predict(self, sample):
        """预测单个样本"""
        return self.root.predict(sample)
    
    def predict_batch(self, samples):
        """批量预测"""
        return [self.predict(s) for s in samples]
    
    def _visualize_helper(self, node, indent=0, prefix=""):
        """辅助可视化"""
        if node.is_leaf():
            return "  " * indent + prefix + f"→ 决策: [{node.label}]\n"
        
        result = "  " * indent + prefix + f"[{node.feature}]?\n"
        for value, child in node.branches.items():
            result += self._visualize_helper(child, indent + 1, f"= {value} ")
        return result
    
    def visualize(self):
        """可视化树结构"""
        return self._visualize_helper(self.root)
    
    def to_rules(self, node=None, path=None):
        """将树转换为 if-then 规则"""
        if path is None:
            path = []
        if node is None:
            node = self.root
        
        if node.is_leaf():
            conditions = " AND ".join(path)
            return [f"IF {conditions} THEN 决策=[{node.label}]"]
        
        rules = []
        for value, child in node.branches.items():
            new_path = path + [f"{node.feature}={value}"]
            rules.extend(self.to_rules(child, new_path))
        return rules


# ============== 演示1：网球数据集 ==============

def demo_tennis():
    """演示：网球数据集"""
    print("=" * 60)
    print("演示1：网球数据集 - 玩不玩网球？")
    print("=" * 60)
    
    # 网球数据集
    tennis_data = [
        {'天气': '晴', '温度': '热', '湿度': '高', '风力': '弱', '玩网球': '否'},
        {'天气': '晴', '温度': '热', '湿度': '高', '风力': '强', '玩网球': '否'},
        {'天气': '多云', '温度': '热', '湿度': '高', '风力': '弱', '玩网球': '是'},
        {'天气': '雨', '温度': '温和', '湿度': '高', '风力': '弱', '玩网球': '是'},
        {'天气': '雨', '温度': '凉', '湿度': '正常', '风力': '弱', '玩网球': '是'},
        {'天气': '雨', '温度': '凉', '湿度': '正常', '风力': '强', '玩网球': '否'},
        {'天气': '多云', '温度': '凉', '湿度': '正常', '风力': '强', '玩网球': '是'},
        {'天气': '晴', '温度': '温和', '湿度': '高', '风力': '弱', '玩网球': '否'},
        {'天气': '晴', '温度': '凉', '湿度': '正常', '风力': '弱', '玩网球': '是'},
        {'天气': '雨', '温度': '温和', '湿度': '正常', '风力': '弱', '玩网球': '是'},
        {'天气': '晴', '温度': '温和', '湿度': '正常', '风力': '强', '玩网球': '是'},
        {'天气': '多云', '温度': '温和', '湿度': '高', '风力': '强', '玩网球': '是'},
        {'天气': '多云', '温度': '热', '湿度': '正常', '风力': '弱', '玩网球': '是'},
        {'天气': '雨', '温度': '温和', '湿度': '高', '风力': '强', '玩网球': '否'},
    ]
    
    # 创建决策树
    tree = ID3DecisionTree()
    tree.fit(tennis_data, '玩网球')
    
    print("\n🌳 生成的决策树：")
    print(tree.visualize())
    
    print("\n📜 提取的规则：")
    for rule in tree.to_rules():
        print(f"  {rule}")
    
    # 测试预测
    print("\n🎯 预测新样本：")
    test_samples = [
        {'天气': '晴', '温度': '温和', '湿度': '正常', '风力': '弱'},
        {'天气': '雨', '温度': '热', '湿度': '高', '风力': '强'},
    ]
    for sample in test_samples:
        prediction = tree.predict(sample)
        print(f"  天气={sample['天气']}, 湿度={sample['湿度']} → 玩网球？{prediction}")


def demo_entropy_calculation():
    """演示熵的计算"""
    print("\n" + "=" * 60)
    print("演示2：熵的计算")
    print("=" * 60)
    
    dt = ID3DecisionTree()
    
    # 完全混乱
    labels1 = ['是', '否', '是', '否']
    entropy1 = dt.entropy(labels1)
    print(f"\n数据集1: {labels1}")
    print(f"  熵 = {entropy1:.3f} (最大混乱，最不确定)")
    
    # 部分有序
    labels2 = ['是', '是', '是', '否']
    entropy2 = dt.entropy(labels2)
    print(f"\n数据集2: {labels2}")
    print(f"  熵 = {entropy2:.3f} (部分有序)")
    
    # 完全有序
    labels3 = ['是', '是', '是', '是']
    entropy3 = dt.entropy(labels3)
    print(f"\n数据集3: {labels3}")
    print(f"  熵 = {entropy3:.3f} (完全有序，完全确定)")
    
    # 信息增益示例
    print("\n📊 信息增益示例：")
    tennis_data = [
        {'天气': '晴', '玩网球': '否'},
        {'天气': '晴', '玩网球': '否'},
        {'天气': '多云', '玩网球': '是'},
        {'天气': '雨', '玩网球': '是'},
        {'天气': '雨', '玩网球': '是'},
        {'天气': '雨', '玩网球': '否'},
    ]
    
    gain = dt.information_gain(tennis_data, '天气', '玩网球')
    print(f"  特征'天气'的信息增益 = {gain:.3f}")
    print(f"  说明：按天气分裂后，不确定性减少了{gain:.1%}")


def demo_fruit_classifier():
    """演示：水果分类器"""
    print("\n" + "=" * 60)
    print("演示3：水果分类器")
    print("=" * 60)
    
    # 水果数据集
    fruit_data = [
        {'颜色': '红', '大小': '小', '形状': '圆', '水果': '樱桃'},
        {'颜色': '红', '大小': '大', '形状': '圆', '水果': '苹果'},
        {'颜色': '黄', '大小': '大', '形状': '弯', '水果': '香蕉'},
        {'颜色': '黄', '大小': '小', '形状': '圆', '水果': '柠檬'},
        {'颜色': '橙', '大小': '中', '形状': '圆', '水果': '橙子'},
        {'颜色': '绿', '大小': '大', '形状': '圆', '水果': '西瓜'},
        {'颜色': '紫', '大小': '小', '形状': '圆', '水果': '葡萄'},
        {'颜色': '红', '大小': '小', '形状': '心', '水果': '草莓'},
        {'颜色': '黄', '大小': '大', '形状': '圆', '水果': '柚子'},
        {'颜色': '绿', '大小': '中', '形状': '圆', '水果': '苹果'},
        {'颜色': '红', '大小': '大', '形状': '椭', '水果': '芒果'},
        {'颜色': '橙', '大小': '小', '形状': '圆', '水果': '金桔'},
    ]
    
    tree = ID3DecisionTree(max_depth=3)
    tree.fit(fruit_data, '水果')
    
    print("\n🌳 决策树：")
    print(tree.visualize())
    
    print("\n🎯 识别神秘水果：")
    mystery_fruits = [
        {'颜色': '红', '大小': '大', '形状': '圆'},
        {'颜色': '黄', '大小': '大', '形状': '弯'},
        {'颜色': '紫', '大小': '小', '形状': '圆'},
    ]
    for fruit in mystery_fruits:
        result = tree.predict(fruit)
        print(f"  {fruit['颜色']}-{fruit['大小']}-{fruit['形状']} → 这是{result}！")


def demo_animal_classifier():
    """演示：动物分类器（类似二十个问题）"""
    print("\n" + "=" * 60)
    print("演示4：动物分类器 - 像玩'二十个问题'一样！")
    print("=" * 60)
    
    # 动物数据集
    animals = [
        {'会飞': '否', '有腿': '是', '水生': '否', '哺乳动物': '是', '动物': '狗'},
        {'会飞': '否', '有腿': '是', '水生': '否', '哺乳动物': '是', '动物': '猫'},
        {'会飞': '是', '有腿': '是', '水生': '否', '哺乳动物': '否', '动物': '鸟'},
        {'会飞': '否', '有腿': '否', '水生': '是', '哺乳动物': '否', '动物': '鱼'},
        {'会飞': '否', '有腿': '是', '水生': '是', '哺乳动物': '是', '动物': '海豚'},
        {'会飞': '是', '有腿': '是', '水生': '否', '哺乳动物': '是', '动物': '蝙蝠'},
        {'会飞': '否', '有腿': '是', '水生': '否', '哺乳动物': '否', '动物': '蜥蜴'},
        {'会飞': '否', '有腿': '是', '水生': '是', '哺乳动物': '否', '动物': '青蛙'},
    ]
    
    tree = ID3DecisionTree()
    tree.fit(animals, '动物')
    
    print("\n🌳 动物识别决策树：")
    print(tree.visualize())
    
    print("\n📜 识别规则：")
    for rule in tree.to_rules():
        print(f"  {rule}")
    
    # 模拟二十个问题的游戏
    print("\n🎮 来玩'二十个问题'游戏！")
    mystery = {'会飞': '否', '有腿': '是', '水生': '否', '哺乳动物': '是'}
    print(f"我在想一个动物：会飞={mystery['会飞']}, 有腿={mystery['有腿']}, "
          f"水生={mystery['水生']}, 哺乳动物={mystery['哺乳动物']}")
    print(f"决策树猜这是：{tree.predict(mystery)}！")


if __name__ == "__main__":
    # 运行所有演示
    demo_tennis()
    demo_entropy_calculation()
    demo_fruit_classifier()
    demo_animal_classifier()
    
    print("\n" + "=" * 60)
    print("🎉 所有演示完成！")
    print("=" * 60)
