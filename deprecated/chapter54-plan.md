# 第五十四章 特征工程——数据的炼金术

## 规划文档

### 基本信息
- **章节编号**: 第54章
- **章节标题**: 特征工程——数据的炼金术
- **预计字数**: 16,000字
- **预计代码行数**: 1,800行
- **目标完成时间**: 2026-03-26 22:30

### 核心主题

#### 1. 特征工程概述
- 什么是特征工程？
- 特征工程的重要性（Garbage In, Garbage Out）
- 特征工程 vs 特征学习
- 特征工程的常见流程

#### 2. 特征编码
- 标签编码（Label Encoding）
- 独热编码（One-Hot Encoding）
- 目标编码（Target Encoding）
- 二进制编码（Binary Encoding）
- 频率编码（Frequency Encoding）
- 均值编码（Mean Encoding）

#### 3. 特征缩放与归一化
- 最小-最大缩放（Min-Max Scaling）
- Z-Score标准化（Standardization）
- Robust Scaling
- 归一化（Normalization）
- 对数变换（Log Transformation）
- Box-Cox变换

#### 4. 特征变换
- 多项式特征（Polynomial Features）
- 交互特征（Interaction Features）
- 分箱/离散化（Binning）
- 数学变换（平方、开方、倒数等）
- 时间特征工程

#### 5. 缺失值处理
- 缺失值类型（MCAR, MAR, MNAR）
- 删除策略
- 填充策略（均值、中位数、众数）
- KNN填充
- 插值方法

#### 6. 特征选择
- 过滤法（Filter Methods）：卡方检验、互信息、相关系数
- 包裹法（Wrapper Methods）：RFE、前向选择、后向消除
- 嵌入法（Embedded Methods）：L1正则化、树模型重要性
- 混合方法

#### 7. 降维作为特征工程
- PCA特征提取
- 特征选择与降维的区别

#### 8. 自动化特征工程
- AutoFeat
- Featuretools
- 遗传编程方法

### 费曼法比喻设计

1. **特征工程 → 食材处理**
   - 原始数据是生食材
   - 特征工程是切、洗、调味的过程
   - 好的食材处理让菜品更美味

2. **独热编码 → 开关灯**
   - 每个类别是一个独立的开关
   - 一次只能亮一盏灯
   - 明确区分不同类别

3. **标准化 → 统一度量衡**
   - 把米、厘米、毫米都转换成统一单位
   - 让模型公平看待每个特征

4. **多项式特征 → 混合饮料**
   - 单独喝果汁或汽水都不错
   - 混合在一起可能有新味道
   - 捕捉特征间的相互作用

5. **特征选择 → 选队友**
   - 不是人多力量大
   - 选对人比选多人重要
   - 去掉拖后腿的

### 参考文献

1. Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157-1182.

2. Kuhn, M., & Johnson, K. (2019). *Feature Engineering and Selection: A Practical Approach for Predictive Models*. CRC Press.

3. Zheng, A., & Casari, A. (2018). *Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists*. O'Reilly Media.

4. Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179-188.

5. Box, G. E. P., & Cox, D. R. (1964). An analysis of transformations. *Journal of the Royal Statistical Society*, 26(2), 211-252.

6. Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. *ACM SIGKDD Explorations Newsletter*, 3(1), 27-32.

7. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

8. Tang, J., Alelyani, S., & Liu, H. (2014). Feature selection for classification: A review. In *Data Classification: Algorithms and Applications* (pp. 37-64). CRC Press.

9. Kanter, J. M., & Veeramachaneni, K. (2015). Deep feature synthesis: Towards automating data science endeavors. In *2015 IEEE International Conference on Data Science and Advanced Analytics* (pp. 1-10).

10. Khurana, U., Turaga, D., Samulowitz, H., & Parthasarathy, S. (2016). Cognito: Automated feature engineering for supervised learning. In *2016 IEEE International Conference on Data Mining Workshops* (pp. 1304-1307).

### 代码结构

```
chapter-54-code.py
├── 数据加载与预处理
├── 特征编码实现
│   ├── 标签编码
│   ├── 独热编码
│   ├── 目标编码
│   └── 频率编码
├── 特征缩放实现
│   ├── Min-Max缩放
│   ├── Z-Score标准化
│   ├── Robust缩放
│   └── 对数变换
├── 特征变换实现
│   ├── 多项式特征
│   ├── 交互特征
│   └── 分箱操作
├── 缺失值处理
│   ├── 各类填充策略
│   └── KNN填充
├── 特征选择实现
│   ├── 过滤法（卡方、互信息）
│   ├── 包裹法（RFE）
│   └── 嵌入法（Lasso）
└── 综合实战案例
```

### 数学公式

1. 标准化: $z = \frac{x - \mu}{\sigma}$
2. Min-Max缩放: $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$
3. 卡方统计量: $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$
4. 互信息: $I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$
5. 多项式特征: $[x_1, x_2] \rightarrow [1, x_1, x_2, x_1^2, x_1x_2, x_2^2]$

### 实战案例

1. **房价预测特征工程**
   - 处理各种数据类型
   - 时间特征提取
   - 地理位置特征

2. **客户流失预测**
   - 类别特征编码
   - 特征选择
   - 效果对比

### 练习题设计

1. 为什么独热编码会导致维度灾难？如何处理高基数类别？
2. 什么时候应该使用标准化而不是归一化？
3. 解释过滤法、包裹法、嵌入法的优缺点。
4. 实现一个完整的数据预处理流水线。

---

*规划创建时间: 2026-03-26 21:50*
*状态: 研究中*
