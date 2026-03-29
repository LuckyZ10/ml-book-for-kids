# 第九章：决策树——像专家一样做决策

> **"如果你需要向一个五岁的孩子解释什么是决策树，你会怎么说？"**
> 
> 这是一个关于"提问的艺术"的故事。想象一下，你在玩"猜猜我是谁"的游戏——通过问一系列巧妙的问题，你能在最短的步骤内猜出正确答案。决策树就是这样一位"提问大师"。

---

## 9.1 从一个游戏开始

### 9.1.1 "猜猜我是谁"

还记得童年时玩的"猜猜我是谁"吗？

> **小明心想一个动物**，小红要通过提问来猜出它。
>
> 小红：**"它是哺乳动物吗？"**  
> 小明：是 → 排除了鸟类、鱼类、昆虫  
> 小红：**"它会飞吗？"**  
> 小明：不会 → 排除了蝙蝠  
> 小红：**"它生活在水里吗？"**  
> 小明：是 → 排除了狮子、老虎  
> 小红：**"它有鳍吗？"**  
> 小明：是 → 排除了鲸鱼（鲸鱼是哺乳动物但没有鳍）
>
> **答案：海豚！**

这个游戏的核心是什么？

✅ **每次提问都尽可能排除最多的可能性**  
✅ **问题的顺序很重要**——好问题能事半功倍  
✅ **最终得到清晰的分类结果**

这就是决策树的核心思想：**通过一系列最优的问题（分裂），将复杂的数据分类问题分解成简单的判断**。

---

### 9.1.2 决策树长什么样？

让我们看看决策树的"长相"：

```
                    【是否哺乳动物？】
                         /    \
                       是      否
                      /          \
                【会飞吗？】    【有鳍吗？】
                  /    \
                是      否
               /          \
         【蝙蝠】      【生活在水里？】
                         /    \
                       是      否
                      /          \
                【有鳍吗？】    【狮子】
                  /    \
                是      否
               /          \
          【海豚】      【鲸鱼】
```

**决策树的组成部分：**

| 名称 | 符号 | 作用 |
|------|------|------|
| **根节点** | 🔲 | 树的起点，第一个分裂问题 |
| **内部节点** | ⭕ | 中间的问题节点，继续分裂 |
| **叶子节点** | 🍃 | 最终分类结果，不再分裂 |
| **分支** | → | 问题的不同答案路径 |

---

### 9.1.3 费曼检验框 #1

**🎯 费曼提问法：你能向一个五岁的孩子解释决策树吗？**

> *想象你是一位老师，面前坐着一群五岁的孩子。他们说："老师，什么是决策树？"*

**标准答案模板：**
> "决策树就像一个**聪明的提问游戏**。假设我有20个水果，想找出所有的苹果。我不会一个一个看，而是问：'是红色的吗？'——这样一半水果就被排除了！然后再问：'是圆的吗？'——又排除一半！通过几个聪明的问题，我就能快速找到所有苹果。**决策树就是教电脑学会问这些聪明问题的魔法！**"

**✅ 过关标准：** 如果孩子点头说"哦，我懂了！"，你就过关了！

---

## 9.2 历史溯源：从概念学习到CART

### 9.2.1 早期萌芽（1960年代）

决策树的历史可以追溯到**古希腊的分类思想**，但真正的计算实现始于20世纪60年代。

#### **Hunt的概念学习系统（1962）**

> **研究者**：E. B. Hunt, J. Marin, P. T. Stone  
> **论文**：*Experiments in Induction* (1966)

这是决策树的理论源头。Hunt等人研究如何让计算机像人类一样学习概念。他们提出了**概念学习系统（Concept Learning System, CLS）**的基本框架：

```
CLS核心思想：
┌─────────────────────────────────────┐
│  给定：一组例子（正面+负面）         │
│  目标：找到一个规则区分它们          │
│  方法：递归地将问题分解成子问题      │
└─────────────────────────────────────┘
```

这是**分而治之（Divide and Conquer）**策略在机器学习中的首次应用。

---

#### **Morgan和Sonquist的AID（1963）**

> **研究者**：James N. Morgan, John A. Sonquist  
> **机构**：密歇根大学  
> **系统**：Automatic Interaction Detection (AID)

这是第一个真正用于数据分析的决策树程序。AID用于分析BBC（英国广播公司）的观众调查数据：

```
AID的创新：
• 使用机械计算器和打孔卡片处理数据
• 自动发现数据中的交互效应
• 递归分区方法（Recursive Partitioning）
```

**历史意义**：证明了决策树可以处理真实世界的复杂数据，而不仅仅是玩具问题。

---

### 9.2.2 ID3的诞生（1975-1986）

> **研究者**：John Ross Quinlan  
> **国籍**：澳大利亚  
> **算法**：ID3 (Iterative Dichotomiser 3)  
> **论文**：*Induction of Decision Trees* (1986, Machine Learning)

这是决策树历史上最重要的里程碑之一。

#### **Quinlan的故事**

J. Ross Quinlan当时是新南威尔士大学的计算机科学家。他的灵感来自信息论——特别是**Claude Shannon**在1948年发表的《通信的数学理论》。

> **关键洞察**：如果信息论能衡量通信的不确定性，那么它也能衡量数据分裂的"信息量"！

#### **ID3的核心创新**

```python
# ID3的灵魂：信息增益（Information Gain）
def information_gain(parent_entropy, child_entropies, weights):
    """
    信息增益 = 父节点熵 - 加权子节点熵
    
    原理：选择能让"混乱度"下降最多的属性
    """
    return parent_entropy - sum(w * e for w, e in zip(weights, child_entropies))
```

**信息增益的直观理解：**

想象你有一堆混乱的袜子（红、蓝、绿混在一起）。
- **按颜色分类**：瞬间变整齐，信息增益大！
- **按品牌分类**：可能还是混乱，信息增益小。

ID3总是选择**信息增益最大**的属性来分裂数据。

---

### 9.2.3 CART的革命（1984）

> **研究者**：Leo Breiman, Jerome Friedman, Richard Olshen, Charles Stone  
> **机构**：加州大学伯克利分校 + 斯坦福大学  
> **著作**：*Classification and Regression Trees* (1984)  
> **出版**：Chapman and Hall/CRC

CART是决策树领域另一座丰碑。与ID3几乎同时独立发展，但采用了不同的哲学。

#### **CART的独特之处**

| 特性 | ID3/C4.5 | CART |
|------|----------|------|
| **树结构** | 多叉树 | **二叉树**（总是二分） |
| **分裂准则** | 信息增益/增益比 | **基尼不纯度** |
| **剪枝** | 悲观剪枝 | **成本复杂度剪枝** |
| **任务** | 分类 | **分类+回归** |

#### **Leo Breiman的传奇故事**

Leo Breiman是一位与众不同的统计学家。他最初是概率论专家，后来转向咨询工作。在为美国军方做雷达信号分析时，他遇到了一个难题：**如何从雷达回波中识别舰船类型？**

> *"经过大量绞尽脑汁的思考，决策树的想法突然灵光一现。"* —— Leo Breiman

他和团队花了13年时间完善CART算法。1984年的著作至今仍是决策树领域的"圣经"。

---

### 9.2.4 C4.5的完善（1993）

> **研究者**：J. Ross Quinlan  
> **著作**：*C4.5: Programs for Machine Learning* (1993)

C4.5是ID3的进化版，解决了ID3的多个缺陷，成为数据挖掘的**黄金标准**。

#### **C4.5的重大改进**

```
ID3的缺陷 → C4.5的解决方案
─────────────────────────────────
❌ 偏好多值属性  →  ✅ 使用信息增益比（Gain Ratio）
❌ 不能处理连续值  →  ✅ 自动寻找最优分裂点
❌ 不能处理缺失值  →  ✅ 概率权重处理缺失
❌ 容易过拟合     →  ✅ 悲观剪枝（Pessimistic Pruning）
❌ 只能分类       →  ✅ 支持概率预测
```

**历史地位**：2008年，IEEE评选的**"数据挖掘十大算法"**中，C4.5排名第一！

---

### 9.2.5 历史时间线

```
1962 ─ Hunt的概念学习系统
  │
1963 ─ Morgan & Sonquist的AID系统
  │
1966 ─ Hunt的《Experiments in Induction》
  │
1972 ─ THAID项目（分类树）
  │
1974 ─ Breiman开始CART研究
  │
1975 ─ Quinlan开始ID3研究
  │
1977 ─ CART原型诞生
  │
1979 ─ Quinlan发表ID3
  │
1984 ─ CART正式出版 🏆
  │
1986 ─ Quinlan正式发表ID3论文
  │
1993 ─ C4.5出版 🏆
  │
2001 ─ Breiman提出随机森林
  │
2008 ─ C4.5获评数据挖掘十大算法No.1
```

---

### 9.2.6 费曼检验框 #2

**🎯 历史理解检验：为什么需要三种不同的决策树算法？**

**一句话总结：**
> *"ID3开创了信息论方法，CART带来了统计学严谨，C4.5融合了两者并解决了实际工程问题。就像三个工匠各自打造了更精良的工具——他们用不同的哲学解决同一个核心挑战：如何让计算机学会像人类专家一样做决策。"*

**关键洞察：**
- **ID3** = 信息论的优雅
- **CART** = 统计学的严谨（基尼指数、剪枝理论）
- **C4.5** = 工程实践的完善（处理缺失值、连续值、过拟合）

---

## 9.3 数学原理：信息、熵与纯度

### 9.3.1 Claude Shannon与信息熵

#### **信息论的黎明（1948）**

> **研究者**：Claude Elwood Shannon  
> **论文**：*A Mathematical Theory of Communication* (1948)  
> **期刊**：Bell System Technical Journal  
> **意义**：现代信息论的诞生

1948年，32岁的Shannon在贝尔实验室发表了这篇划时代的论文。他提出了一个革命性的观点：**信息可以像质量、能量一样被精确度量**。

#### **熵：混乱度的度量**

Shannon借用了物理学中的"熵"（Entropy）概念。关于这个名字，有一个有趣的轶事：

> *Shannon问John von Neumann："我该用什么词来描述这种不确定性？"*  
> *von Neumann回答："叫'熵'吧。一来没有人知道熵到底是什么，二来在辩论中你永远占优势。"*

**信息熵的公式：**

$$H(X) = -\sum_{i=1}^{n} p_i \log_2(p_i)$$

其中：
- $p_i$ = 第$i$个类别出现的概率
- $\log_2$ = 以2为底的对数（结果单位是"比特"）

#### **直观理解熵**

```
场景1：硬币正面朝上概率100%
     H = -(1 × log₂(1)) = 0 比特
     → 没有不确定性，熵为0

场景2：公平硬币（正反各50%）
     H = -(0.5 × log₂(0.5) + 0.5 × log₂(0.5))
       = -(0.5 × (-1) + 0.5 × (-1))
       = 1 比特
     → 需要1个比特来编码结果

场景3：8个等概率结果
     H = -8 × (1/8 × log₂(1/8))
       = -8 × (1/8 × (-3))
       = 3 比特
     → 需要3个比特（可以编码8种结果）
```

**核心洞察**：熵衡量的是**不确定性的大小**。熵越大，系统越混乱；熵越小，系统越有序。

---

### 9.3.2 信息增益：选择最佳分裂

#### **什么是信息增益？**

信息增益（Information Gain）衡量的是：**通过某个属性分裂数据后，不确定性减少了多少**。

$$IG(D, A) = H(D) - \sum_{v \in Values(A)} \frac{|D_v|}{|D|} H(D_v)$$

其中：
- $H(D)$ = 分裂前数据集的熵
- $D_v$ = 属性A取值为v的子集
- $\frac{|D_v|}{|D|}$ = 子集的权重

#### **计算示例**

假设我们有10个水果的数据集：

```
原始数据：
┌──────────┬──────────┬──────────┐
│  颜色    │  形状    │  水果    │
├──────────┼──────────┼──────────┤
│  红色    │  圆形    │  苹果    │
│  红色    │  圆形    │  苹果    │
│  红色    │  圆形    │  苹果    │
│  红色    │  圆形    │  苹果    │
│  红色    │  圆形    │  苹果    │
│  黄色    │  弯曲    │  香蕉    │
│  黄色    │  弯曲    │  香蕉    │
│  黄色    │  弯曲    │  香蕉    │
│  紫色    │  圆形    │  葡萄    │
│  紫色    │  圆形    │  葡萄    │
└──────────┴──────────┴──────────┘

分布：苹果5个，香蕉3个，葡萄2个
```

**步骤1：计算父节点的熵**

```
H(D) = -(5/10 × log₂(5/10) + 3/10 × log₂(3/10) + 2/10 × log₂(2/10))
     = -(0.5 × (-1) + 0.3 × (-1.737) + 0.2 × (-2.322))
     = 0.5 + 0.521 + 0.464
     ≈ 1.485 比特
```

**步骤2：按"颜色"分裂后的信息增益**

```
红色子集（5个）：全是苹果 → H = 0
黄色子集（3个）：全是香蕉 → H = 0  
紫色子集（2个）：全是葡萄 → H = 0

加权子节点熵 = (5/10 × 0) + (3/10 × 0) + (2/10 × 0) = 0

信息增益 = 1.485 - 0 = 1.485 比特 ✨
```

**步骤3：按"形状"分裂后的信息增益**

```
圆形子集（7个）：5个苹果 + 2个葡萄
弯曲子集（3个）：3个香蕉

H(圆形) = -(5/7 × log₂(5/7) + 2/7 × log₂(2/7))
        ≈ 0.863 比特

H(弯曲) = 0 （全是香蕉）

加权子节点熵 = (7/10 × 0.863) + (3/10 × 0) ≈ 0.604

信息增益 = 1.485 - 0.604 = 0.881 比特
```

**结论**：按"颜色"分裂的信息增益更大（1.485 > 0.881），因此**选择"颜色"作为分裂属性**！

---

### 9.3.3 基尼不纯度：CART的选择

#### **基尼不纯度的定义**

CART算法使用基尼不纯度（Gini Impurity）而不是信息增益。

$$Gini(D) = 1 - \sum_{i=1}^{n} p_i^2$$

其中$p_i$是第$i$个类别的概率。

#### **基尼不纯度的直观解释**

> **想象**：你随机从数据集中抽取一个样本，然后随机猜它的类别。基尼不纯度就是**猜错的概率**。

```
场景1：全是苹果（纯节点）
     Gini = 1 - (1²) = 0
     → 不可能猜错，纯度最高

场景2：一半苹果，一半香蕉
     Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.5
     → 50%概率猜错

场景3：10个类别，各10%
     Gini = 1 - 10 × (0.1²) = 1 - 0.1 = 0.9
     → 90%概率猜错，非常混乱
```

#### **信息增益 vs 基尼不纯度**

| 特性 | 信息增益 | 基尼不纯度 |
|------|----------|------------|
| **理论基础** | 信息论（Shannon） | 经济学（基尼系数） |
| **计算** | 需要计算对数 | 只算平方，更快 |
| **偏好** | 倾向于多值属性 | 更平衡 |
| **实际效果** | 相似 | 相似 |
| **典型应用** | ID3, C4.5 | CART |

**实践建议**：两者在实际应用中效果相当，基尼不纯度计算稍快。

---

### 9.3.4 信息增益比：C4.5的改进

#### **信息增益的问题**

ID3的信息增益有一个**致命缺陷**：它**偏爱取值较多的属性**。

```
极端例子：每个学生有唯一的"学号"属性
• 按学号分裂：每个学生一个分支，每个分支纯度100%
• 信息增益 = 最大值！
• 但是：对新学生完全没有泛化能力！
```

这就是**过拟合**的典型表现。

#### **信息增益比的解决方案**

C4.5引入了信息增益比（Gain Ratio）：

$$GainRatio(D, A) = \frac{IG(D, A)}{SplitInfo(A)}$$

其中SplitInfo是分裂信息，衡量属性的固有信息量：

$$SplitInfo(A) = -\sum_{v \in Values(A)} \frac{|D_v|}{|D|} \log_2\left(\frac{|D_v|}{|D|}\right)$$

**原理**：如果属性取值很多，SplitInfo会很大，从而惩罚信息增益。

---

### 9.3.5 费曼检验框 #3

**🎯 数学概念检验：为什么熵的公式里有log和对数？**

**直觉解释：**
> *"想象你要用二进制编码（0和1）来传达信息。如果你有8个等可能的结果，需要多少个比特来编码？答案是3，因为2³ = 8。log₂(8) = 3就是计算这个！熵的公式本质上是在问：'描述这个系统需要多少比特？'"

**费曼式类比：**
> *"熵就像整理房间。如果所有东西乱堆在一起（高熵），你需要很多信息才能找到特定物品。如果每样东西都有固定位置（低熵），你几乎不需要信息就能找到它。决策树的目标就是把混乱的房间（高熵数据集）通过不断分类，变成井井有条的抽屉（低熵子集）！"*

---

## 9.4 算法实战：从头实现决策树

### 9.4.1 手搓决策树分类器

现在让我们用纯Python实现一个完整的决策树分类器！

```python
"""
第九章代码：决策树算法实现
chapter-09-decision-tree.py

目标：手搓完整的决策树分类器
包含：
- 信息熵计算
- 信息增益
- 基尼不纯度
- 决策树构建
- 剪枝
- 可视化

作者：AI教育研究团队
日期：2026-03-24
"""

import math
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional


class DecisionNode:
    """
    决策树节点类
    
    每个节点可以是：
    - 内部节点：包含分裂属性和子节点
    - 叶子节点：包含预测类别
    """
    
    def __init__(self, 
                 attribute: Optional[str] = None,
                 threshold: Optional[float] = None,
                 branches: Optional[Dict[Any, 'DecisionNode']] = None,
                 label: Optional[Any] = None,
                 depth: int = 0):
        """
        初始化节点
        
        参数:
            attribute: 分裂属性（内部节点）
            threshold: 分裂阈值（连续属性）
            branches: 子节点字典 {值: 子节点}
            label: 预测类别（叶子节点）
            depth: 节点深度
        """
        self.attribute = attribute  # 分裂属性
        self.threshold = threshold  # 分裂阈值（连续属性用）
        self.branches = branches or {}  # 子节点
        self.label = label  # 叶子节点的类别
        self.depth = depth  # 当前深度
        self.samples = 0  # 到达该节点的样本数
        self.impurity = 0.0  # 节点的不纯度
        
    def is_leaf(self) -> bool:
        """判断是否为叶子节点"""
        return self.label is not None
    
    def predict(self, sample: Dict[str, Any]) -> Any:
        """
        对单个样本进行预测
        
        参数:
            sample: 样本字典 {属性: 值}
        
        返回:
            预测类别
        """
        # 如果是叶子节点，直接返回标签
        if self.is_leaf():
            return self.label
        
        # 获取该属性的值
        value = sample.get(self.attribute)
        
        # 处理连续属性
        if self.threshold is not None:
            if value is None:
                # 缺失值：选择样本最多的分支
                value = '<=' if self.samples_left >= self.samples_right else '>'
            else:
                value = '<=' if value <= self.threshold else '>'
        
        # 如果该值不存在于分支中，返回样本最多的标签
        if value not in self.branches:
            return self._get_majority_label()
        
        # 递归到子节点
        return self.branches[value].predict(sample)
    
    def _get_majority_label(self) -> Any:
        """获取多数类标签（用于处理缺失分支）"""
        if self.is_leaf():
            return self.label
        # 统计各子节点的样本数
        max_samples = 0
        majority_label = None
        for branch in self.branches.values():
            if branch.samples > max_samples:
                max_samples = branch.samples
                majority_label = branch.label if branch.is_leaf() else None
        return majority_label
    
    def __repr__(self) -> str:
        """字符串表示"""
        if self.is_leaf():
            return f"🍃 Leaf(label={self.label}, samples={self.samples})"
        else:
            threshold_info = f" <= {self.threshold:.2f}" if self.threshold else ""
            return f"⭕ Node({self.attribute}{threshold_info}, depth={self.depth})"


class DecisionTreeClassifier:
    """
    决策树分类器（纯Python实现）
    
    支持：
    - ID3算法（信息增益）
    - C4.5算法（信息增益比）
    - CART算法（基尼不纯度）
    - 连续属性处理
    - 缺失值处理
    - 剪枝
    """
    
    def __init__(self, 
                 criterion: str = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[int] = None):
        """
        初始化决策树分类器
        
        参数:
            criterion: 分裂准则 ('gini', 'entropy', 'gain_ratio')
            max_depth: 最大深度（防止过拟合）
            min_samples_split: 分裂所需最少样本数
            min_samples_leaf: 叶子节点最少样本数
            max_features: 每次分裂考虑的最大特征数
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        
        self.root: Optional[DecisionNode] = None
        self.feature_names: List[str] = []
        self.n_classes: int = 0
        self.classes: List[Any] = []
        
    def fit(self, X: List[List[Any]], y: List[Any], 
            feature_names: Optional[List[str]] = None) -> 'DecisionTreeClassifier':
        """
        训练决策树
        
        参数:
            X: 特征矩阵 [[特征1, 特征2, ...], ...]
            y: 标签列表 [类别1, 类别2, ...]
            feature_names: 特征名称列表
        
        返回:
            self
        """
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        
        # 设置特征名称
        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"特征{i}" for i in range(n_features)]
        
        # 获取类别信息
        self.classes = list(set(y))
        self.n_classes = len(self.classes)
        
        print(f"📊 开始训练决策树...")
        print(f"   样本数: {n_samples}")
        print(f"   特征数: {n_features}")
        print(f"   类别数: {self.n_classes} ({self.classes})")
        print(f"   分裂准则: {self.criterion}")
        
        # 构建树
        self.root = self._build_tree(X, y, depth=0, used_features=set())
        
        print(f"✅ 决策树构建完成！")
        
        return self
    
    def _build_tree(self, X: List[List[Any]], y: List[Any], 
                    depth: int, used_features: set) -> DecisionNode:
        """
        递归构建决策树（核心算法）
        
        参数:
            X: 当前节点的特征矩阵
            y: 当前节点的标签列表
            depth: 当前深度
            used_features: 已使用的特征集合
        
        返回:
            构建好的节点
        """
        n_samples = len(y)
        
        # 统计类别分布
        class_counts = Counter(y)
        most_common = class_counts.most_common(1)[0]
        majority_class = most_common[0]
        
        # 创建节点
        node = DecisionNode(depth=depth)
        node.samples = n_samples
        
        # ========== 终止条件 ==========
        
        # 1. 所有样本属于同一类别 → 叶子节点
        if len(class_counts) == 1:
            node.label = y[0]
            node.impurity = 0.0
            return node
        
        # 2. 达到最大深度 → 叶子节点
        if self.max_depth is not None and depth >= self.max_depth:
            node.label = majority_class
            node.impurity = self._impurity(y)
            return node
        
        # 3. 样本数少于最小分裂数 → 叶子节点
        if n_samples < self.min_samples_split:
            node.label = majority_class
            node.impurity = self._impurity(y)
            return node
        
        # 4. 没有可用特征 → 叶子节点
        n_features = len(X[0]) if X else 0
        available_features = set(range(n_features)) - used_features
        if not available_features:
            node.label = majority_class
            node.impurity = self._impurity(y)
            return node
        
        # ========== 选择最佳分裂属性 ==========
        
        best_feature, best_threshold, best_gain = self._find_best_split(X, y, available_features)
        
        # 如果无法找到有效分裂 → 叶子节点
        if best_feature is None or best_gain <= 0:
            node.label = majority_class
            node.impurity = self._impurity(y)
            return node
        
        # ========== 分裂数据 ==========
        
        # 判断是否为连续属性
        is_continuous = best_threshold is not None
        
        # 分裂数据
        branches_data = self._split_data(X, y, best_feature, best_threshold)
        
        # 检查最小叶子样本数约束
        valid_split = all(len(branch_y) >= self.min_samples_leaf 
                         for branch_y in branches_data.values())
        
        if not valid_split:
            node.label = majority_class
            node.impurity = self._impurity(y)
            return node
        
        # 创建内部节点
        node.attribute = self.feature_names[best_feature]
        node.threshold = best_threshold
        node.impurity = self._impurity(y)
        
        # 递归构建子树
        node.branches = {}
        new_used_features = used_features | {best_feature} if not is_continuous else used_features
        
        for value, (branch_X, branch_y) in branches_data.items():
            child_node = self._build_tree(branch_X, branch_y, depth + 1, new_used_features)
            node.branches[value] = child_node
            
            # 记录样本数（用于处理缺失值）
            if is_continuous:
                if value == '<=':
                    node.samples_left = len(branch_y)
                else:
                    node.samples_right = len(branch_y)
        
        return node
    
    def _find_best_split(self, X: List[List[Any]], y: List[Any], 
                         features: set) -> Tuple[Optional[int], Optional[float], float]:
        """
        寻找最佳分裂属性
        
        参数:
            X: 特征矩阵
            y: 标签列表
            features: 候选特征集合
        
        返回:
            (最佳特征索引, 最佳阈值, 最佳增益)
        """
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        
        parent_impurity = self._impurity(y)
        n_samples = len(y)
        
        # 随机选择特征子集（随机森林风格）
        feature_list = list(features)
        if self.max_features and len(feature_list) > self.max_features:
            import random
            feature_list = random.sample(feature_list, self.max_features)
        
        for feature_idx in feature_list:
            # 提取该特征的所有值
            feature_values = [sample[feature_idx] for sample in X]
            
            # 检查是否为连续属性
            is_continuous = self._is_continuous(feature_values)
            
            if is_continuous:
                # 连续属性：寻找最佳分裂点
                thresholds = self._get_candidate_thresholds(feature_values)
                
                for threshold in thresholds:
                    gain = self._calculate_gain_continuous(
                        X, y, feature_idx, threshold, parent_impurity
                    )
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = threshold
            else:
                # 离散属性
                gain = self._calculate_gain_discrete(
                    X, y, feature_idx, parent_impurity
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = None
        
        return best_feature, best_threshold, best_gain
    
    def _is_continuous(self, values: List[Any]) -> bool:
        """判断属性是否为连续型"""
        # 如果有超过10个不同值，且都是数字，认为是连续的
        unique_values = set(v for v in values if v is not None)
        if len(unique_values) > 10:
            try:
                [float(v) for v in unique_values]
                return True
            except (ValueError, TypeError):
                return False
        return False
    
    def _get_candidate_thresholds(self, values: List[Any]) -> List[float]:
        """获取候选分裂点"""
        # 排序后的唯一值的中点
        sorted_values = sorted(set(v for v in values if v is not None))
        thresholds = []
        for i in range(len(sorted_values) - 1):
            threshold = (sorted_values[i] + sorted_values[i + 1]) / 2
            thresholds.append(threshold)
        return thresholds
    
    def _split_data(self, X: List[List[Any]], y: List[Any], 
                    feature_idx: int, threshold: Optional[float]) -> Dict[Any, Tuple[List[List[Any]], List[Any]]]:
        """
        根据特征分裂数据
        
        返回:
            {分支值: (分支X, 分支y)}
        """
        branches = {}
        
        for sample, label in zip(X, y):
            value = sample[feature_idx]
            
            # 处理连续属性
            if threshold is not None:
                if value is None:
                    continue  # 缺失值稍后处理
                branch_key = '<=' if value <= threshold else '>'
            else:
                branch_key = value if value is not None else 'missing'
            
            if branch_key not in branches:
                branches[branch_key] = ([], [])
            
            branches[branch_key][0].append(sample)
            branches[branch_key][1].append(label)
        
        return branches
    
    def _impurity(self, y: List[Any]) -> float:
        """计算不纯度"""
        if not y:
            return 0.0
        
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        else:  # entropy 或 gain_ratio
            return self._entropy(y)
    
    def _entropy(self, y: List[Any]) -> float:
        """计算信息熵"""
        if not y:
            return 0.0
        
        n = len(y)
        class_counts = Counter(y)
        
        entropy = 0.0
        for count in class_counts.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _gini_impurity(self, y: List[Any]) -> float:
        """计算基尼不纯度"""
        if not y:
            return 0.0
        
        n = len(y)
        class_counts = Counter(y)
        
        gini = 1.0
        for count in class_counts.values():
            p = count / n
            gini -= p * p
        
        return gini
    
    def _calculate_gain_discrete(self, X: List[List[Any]], y: List[Any], 
                                  feature_idx: int, parent_impurity: float) -> float:
        """计算离散属性的信息增益"""
        n = len(y)
        
        # 按特征值分组
        value_groups = {}
        for sample, label in zip(X, y):
            value = sample[feature_idx]
            if value not in value_groups:
                value_groups[value] = []
            value_groups[value].append(label)
        
        # 计算加权子节点不纯度
        weighted_impurity = 0.0
        for labels in value_groups.values():
            weight = len(labels) / n
            weighted_impurity += weight * self._impurity(labels)
        
        gain = parent_impurity - weighted_impurity
        
        # 如果使用gain_ratio，需要除以分裂信息
        if self.criterion == 'gain_ratio':
            split_info = 0.0
            for labels in value_groups.values():
                p = len(labels) / n
                if p > 0:
                    split_info -= p * math.log2(p)
            if split_info > 0:
                gain = gain / split_info
        
        return gain
    
    def _calculate_gain_continuous(self, X: List[List[Any]], y: List[Any],
                                    feature_idx: int, threshold: float, 
                                    parent_impurity: float) -> float:
        """计算连续属性的信息增益"""
        n = len(y)
        
        left_labels = []
        right_labels = []
        
        for sample, label in zip(X, y):
            value = sample[feature_idx]
            if value is not None:
                if value <= threshold:
                    left_labels.append(label)
                else:
                    right_labels.append(label)
        
        if not left_labels or not right_labels:
            return 0.0
        
        # 计算加权不纯度
        left_weight = len(left_labels) / n
        right_weight = len(right_labels) / n
        
        weighted_impurity = (left_weight * self._impurity(left_labels) + 
                            right_weight * self._impurity(right_labels))
        
        gain = parent_impurity - weighted_impurity
        
        # gain_ratio处理
        if self.criterion == 'gain_ratio':
            split_info = 0.0
            for weight in [left_weight, right_weight]:
                if weight > 0:
                    split_info -= weight * math.log2(weight)
            if split_info > 0:
                gain = gain / split_info
        
        return gain
    
    def predict(self, X: List[List[Any]]) -> List[Any]:
        """
        预测多个样本
        
        参数:
            X: 特征矩阵
        
        返回:
            预测类别列表
        """
        predictions = []
        for sample in X:
            # 转换为字典格式
            sample_dict = {name: value 
                          for name, value in zip(self.feature_names, sample)}
            pred = self.root.predict(sample_dict)
            predictions.append(pred)
        return predictions
    
    def predict_proba(self, X: List[List[Any]]) -> List[Dict[Any, float]]:
        """预测概率（简化版）"""
        # 实际实现需要统计叶子节点的类别分布
        # 这里返回One-hot编码
        predictions = self.predict(X)
        result = []
        for pred in predictions:
            proba = {c: 0.0 for c in self.classes}
            proba[pred] = 1.0
            result.append(proba)
        return result
    
    def score(self, X: List[List[Any]], y: List[Any]) -> float:
        """计算准确率"""
        predictions = self.predict(X)
        correct = sum(1 for pred, true in zip(predictions, y) if pred == true)
        return correct / len(y)
    
    def visualize(self, node: Optional[DecisionNode] = None, 
                  prefix: str = "", is_last: bool = True) -> str:
        """
        ASCII可视化决策树
        
        参数:
            node: 当前节点
            prefix: 前缀字符串
            is_last: 是否为最后一个子节点
        
        返回:
            可视化字符串
        """
        if node is None:
            node = self.root
            if node is None:
                return "树尚未训练！"
        
        lines = []
        
        # 当前节点的连接符
        connector = "└── " if is_last else "├── "
        
        if node.is_leaf():
            # 叶子节点
            lines.append(f"{prefix}{connector}🍃 预测: {node.label} (样本数: {node.samples})")
        else:
            # 内部节点
            threshold_info = f" <= {node.threshold:.2f}?" if node.threshold else "?"
            lines.append(f"{prefix}{connector}⭕ {node.attribute}{threshold_info}")
            
            # 子节点
            new_prefix = prefix + ("    " if is_last else "│   ")
            items = list(node.branches.items())
            
            for i, (value, child) in enumerate(items):
                is_last_child = (i == len(items) - 1)
                branch_connector = "└── " if is_last_child else "├── "
                lines.append(f"{new_prefix}{branch_connector}【{value}】")
                
                child_prefix = new_prefix + ("    " if is_last_child else "│   ")
                child_lines = self.visualize(child, child_prefix, True)
                lines.append(child_lines)
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"DecisionTreeClassifier(criterion='{self.criterion}', max_depth={self.max_depth})"


# ========== 辅助函数和演示 ==========

def demo_decision_tree():
    """决策树完整演示"""
    
    print("=" * 70)
    print("🌳 决策树分类器演示")
    print("=" * 70)
    
    # 经典数据集：天气与是否打球
    # 特征：Outlook, Temperature, Humidity, Wind
    # 标签：Play (Yes/No)
    
    data = [
        # Outlook,   Temp,      Humidity, Wind,    Play
        ['Sunny',    'Hot',     'High',   'Weak',   'No'],
        ['Sunny',    'Hot',     'High',   'Strong', 'No'],
        ['Overcast', 'Hot',     'High',   'Weak',   'Yes'],
        ['Rain',     'Mild',    'High',   'Weak',   'Yes'],
        ['Rain',     'Cool',    'Normal', 'Weak',   'Yes'],
        ['Rain',     'Cool',    'Normal', 'Strong', 'No'],
        ['Overcast', 'Cool',    'Normal', 'Strong', 'Yes'],
        ['Sunny',    'Mild',    'High',   'Weak',   'No'],
        ['Sunny',    'Cool',    'Normal', 'Weak',   'Yes'],
        ['Rain',     'Mild',    'Normal', 'Weak',   'Yes'],
        ['Sunny',    'Mild',    'Normal', 'Strong', 'Yes'],
        ['Overcast', 'Mild',    'High',   'Strong', 'Yes'],
        ['Overcast', 'Hot',     'Normal', 'Weak',   'Yes'],
        ['Rain',     'Mild',    'High',   'Strong', 'No'],
    ]
    
    # 分离特征和标签
    X = [[row[i] for i in range(4)] for row in data]
    y = [row[4] for row in data]
    feature_names = ['天气', '温度', '湿度', '风力']
    
    print("\n📊 数据集：天气与打球决策")
    print("-" * 50)
    print(f"{'天气':<10} {'温度':<10} {'湿度':<10} {'风力':<10} {'打球':<10}")
    print("-" * 50)
    for row in data:
        print(f"{row[0]:<10} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10}")
    
    # 使用不同准则训练决策树
    criteria = [
        ('entropy', '信息增益 (ID3风格)'),
        ('gain_ratio', '信息增益比 (C4.5风格)'),
        ('gini', '基尼不纯度 (CART风格)')
    ]
    
    for criterion, description in criteria:
        print(f"\n{'=' * 70}")
        print(f"🌲 使用 {description}")
        print(f"{'=' * 70}")
        
        # 训练
        clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=3,
            min_samples_split=2
        )
        clf.fit(X, y, feature_names)
        
        # 可视化
        print("\n🖼️ 决策树结构：")
        print(clf.visualize())
        
        # 预测
        print("\n🔮 预测结果：")
        predictions = clf.predict(X)
        accuracy = clf.score(X, y)
        print(f"训练集准确率: {accuracy:.1%}")
        
        # 新样本预测
        test_samples = [
            ['Sunny', 'Cool', 'Normal', 'Weak'],   # 应该预测 Yes
            ['Rain', 'Hot', 'High', 'Strong'],     # 应该预测 No
        ]
        print(f"\n新样本预测:")
        for sample in test_samples:
            pred = clf.predict([sample])[0]
            print(f"  {sample} → {pred}")


def demo_iris_dataset():
    """在Iris数据集上的演示"""
    
    print("\n" + "=" * 70)
    print("🌸 Iris数据集演示（数值特征）")
    print("=" * 70)
    
    # 简化的Iris数据集（部分样本）
    # 特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度
    iris_data = [
        # 山鸢尾 (Setosa) - 花瓣较小
        ([5.1, 3.5, 1.4, 0.2], 'Setosa'),
        ([4.9, 3.0, 1.4, 0.2], 'Setosa'),
        ([4.7, 3.2, 1.3, 0.2], 'Setosa'),
        ([4.6, 3.1, 1.5, 0.2], 'Setosa'),
        ([5.0, 3.6, 1.4, 0.2], 'Setosa'),
        
        # 变色鸢尾 (Versicolor) - 中等
        ([7.0, 3.2, 4.7, 1.4], 'Versicolor'),
        ([6.4, 3.2, 4.5, 1.5], 'Versicolor'),
        ([6.9, 3.1, 4.9, 1.5], 'Versicolor'),
        ([5.5, 2.3, 4.0, 1.3], 'Versicolor'),
        ([6.5, 2.8, 4.6, 1.5], 'Versicolor'),
        
        # 维吉尼亚鸢尾 (Virginica) - 花瓣较大
        ([6.3, 3.3, 6.0, 2.5], 'Virginica'),
        ([5.8, 2.7, 5.1, 1.9], 'Virginica'),
        ([7.1, 3.0, 5.9, 2.1], 'Virginica'),
        ([6.3, 2.9, 5.6, 1.8], 'Virginica'),
        ([6.5, 3.0, 5.8, 2.2], 'Virginica'),
    ]
    
    X = [d[0] for d in iris_data]
    y = [d[1] for d in iris_data]
    feature_names = ['花萼长', '花萼宽', '花瓣长', '花瓣宽']
    
    print(f"\n样本数: {len(X)}")
    print(f"类别: {set(y)}")
    
    # 训练
    clf = DecisionTreeClassifier(
        criterion='gini',
        max_depth=3,
        min_samples_split=2
    )
    clf.fit(X, y, feature_names)
    
    # 可视化
    print("\n🖼️ 决策树：")
    print(clf.visualize())
    
    # 评估
    accuracy = clf.score(X, y)
    print(f"\n✅ 训练集准确率: {accuracy:.1%}")
    
    # 特征重要性分析（简单版）
    print("\n📈 特征使用统计：")
    features_used = {}
    def count_features(node):
        if node.is_leaf():
            return
        features_used[node.attribute] = features_used.get(node.attribute, 0) + 1
        for child in node.branches.values():
            count_features(child)
    
    count_features(clf.root)
    for feat, count in sorted(features_used.items(), key=lambda x: -x[1]):
        print(f"  {feat}: 使用 {count} 次")


def demo_entropy_calculation():
    """熵计算演示"""
    
    print("\n" + "=" * 70)
    print("📊 信息熵计算演示")
    print("=" * 70)
    
    def entropy(labels):
        from collections import Counter
        import math
        
        if not labels:
            return 0.0
        
        n = len(labels)
        counts = Counter(labels)
        
        h = 0.0
        for count in counts.values():
            p = count / n
            h -= p * math.log2(p)
        return h
    
    scenarios = [
        (['A', 'A', 'A', 'A'], "全部相同"),
        (['A', 'B', 'A', 'B'], "均匀分布"),
        (['A', 'A', 'B', 'B', 'C', 'C'], "三类均匀"),
        (['A', 'A', 'A', 'B'], "3:1分布"),
    ]
    
    print(f"\n{'场景':<15} {'数据':<25} {'熵值':<10} {'解释'}")
    print("-" * 70)
    
    for data, desc in scenarios:
        h = entropy(data)
        explanation = ""
        if h == 0:
            explanation = "完全确定，无不确定性"
        elif h < 1:
            explanation = "较低不确定性"
        else:
            explanation = "较高不确定性"
        
        data_str = str(data)[:23]
        print(f"{desc:<15} {data_str:<25} {h:.3f}      {explanation}")


if __name__ == "__main__":
    # 运行所有演示
    demo_entropy_calculation()
    demo_decision_tree()
    demo_iris_dataset()
    
    print("\n" + "=" * 70)
    print("✨ 第九章代码演示完成！")
    print("=" * 70)
