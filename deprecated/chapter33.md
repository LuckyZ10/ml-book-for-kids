# 第三十三章：时序预测与Transformer变体

## 开篇故事：天气预报员的烦恼

想象一下，你是一位经验丰富的天气预报员。每天，你都要面对成堆的数据——过去一周的气温、湿度、气压、风速...你的任务是预测明天、后天，甚至一周后的天气。

**传统方法的困境**：你使用多年的经验法则——"朝霞不出门，晚霞行千里"、"燕子低飞要下雨"。这些方法在短期预测还不错，但要预测两周后的天气，误差就会像滚雪球一样越来越大。

**深度学习的挑战**：后来，你尝试用深度学习模型。但问题来了——Transformer模型虽然强大，但当序列长度达到1000甚至10000时，计算量会爆炸式增长！$O(L^2)$的复杂度让你望而却步。

**新希望的出现**：直到你遇到了Informer、Autoformer这些专为长序列预测设计的Transformer变体。它们用巧妙的稀疏注意力机制，将复杂度降到了$O(L \log L)$，让你终于可以"记住"更长的历史，预测更远的未来。

这就是本章的故事——**如何让机器像经验丰富的天气预报员一样，从漫长的历史中寻找规律，预测遥远的未来**。

---

## 33.1 时间序列分析基础

### 33.1.1 什么是时间序列？

**时间序列**（Time Series）是按照时间顺序排列的数据点集合。与普通的表格数据不同，时间序列的特殊之处在于**时间依赖性**——过去的值会影响未来的值。

**生活化比喻**：想象你在看一部精彩的连续剧。每一集的故事都建立在前几集的基础上。如果你从中间开始看，可能会完全看不懂剧情。时间序列就像连续剧——**顺序至关重要**！

时间序列数据无处不在：
- 🌡️ **气象数据**：气温、降雨量、风速
- 📈 **金融市场**：股票价格、汇率、交易量
- 🚗 **交通流量**：道路车流量、地铁客流量
- ⚡ **能源消耗**：电力负荷、燃气用量
- 🏥 **医疗健康**：心率、血压、血糖值

### 33.1.2 时间序列的组成成分

任何时间序列都可以分解为四个基本成分：

$$
Y_t = T_t + S_t + C_t + \varepsilon_t
$$

其中：
- $Y_t$：时间序列在时间$t$的观测值
- $T_t$：**趋势**（Trend）——长期上升或下降的模式
- $S_t$：**季节性**（Seasonality）——固定周期的重复模式
- $C_t$：**循环**（Cycle）——非固定周期的波动
- $\varepsilon_t$：**残差/噪声**（Residual）——随机波动

**生活化比喻**：想象你在观察一条河流的水位：
- **趋势**就像气候变化导致的冰川融化——长期来看，水位在缓慢上升
- **季节性**就像四季更替——夏天水位高，冬天水位低，每年重复
- **循环**就像厄尔尼诺现象——每隔几年出现一次的大波动，但周期不固定
- **残差**就像每天的降雨——随机、不可预测的小波动

### 33.1.3 平稳性（Stationarity）

**平稳性**是时间序列分析的核心概念。一个平稳时间序列的统计特性（均值、方差、自相关）不随时间变化。

**严格平稳**（Strict Stationarity）：
$$F_Y(y_{t_1}, y_{t_2}, ..., y_{t_n}) = F_Y(y_{t_1+\tau}, y_{t_2+\tau}, ..., y_{t_n+\tau})$$

对于任意时间偏移$\tau$，联合分布保持不变。

**弱平稳**（Weak Stationarity）更常用：
1. 均值恒定：$E[Y_t] = \mu$
2. 方差恒定：$Var[Y_t] = \sigma^2$
3. 自协方差仅依赖于时间间隔：$Cov(Y_t, Y_{t+k}) = \gamma_k$

**生活化比喻**：想象你在游乐场坐旋转木马。如果木马以恒定速度旋转（平稳），你每次经过同一个位置时的感觉都是一样的。但如果速度忽快忽慢（非平稳），你的体验就会完全不同。

### 33.1.4 自相关函数（ACF）与偏自相关函数（PACF）

**自相关函数**（Autocorrelation Function, ACF）衡量时间序列与其滞后版本的相关性：

$$
\rho_k = \frac{Cov(Y_t, Y_{t-k})}{Var(Y_t)} = \frac{\gamma_k}{\gamma_0}
$$

**偏自相关函数**（Partial Autocorrelation Function, PACF）衡量在控制中间滞后项影响后，$Y_t$与$Y_{t-k}$的直接相关性。

**ACF和PACF的应用**：
- 识别时间序列的周期性
- 帮助确定ARIMA模型的参数$(p, d, q)$
- 检验残差是否为白噪声

---

## 33.2 传统方法：ARIMA与统计模型

### 33.2.1 ARIMA模型

**ARIMA**（AutoRegressive Integrated Moving Average）是时间序列预测的经典方法，由Box和Jenkins在1970年代提出。

ARIMA模型记为$ARIMA(p, d, q)$：
- $p$：自回归（AR）阶数
- $d$：差分（I）阶数
- $q$：移动平均（MA）阶数

#### 自回归模型 AR(p)

当前值由过去$p$个值的线性组合加上噪声构成：

$$
Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + ... + \phi_p Y_{t-p} + \varepsilon_t
$$

**生活化比喻**：AR模型就像你的学习习惯。今天的学习状态取决于前几天——如果昨天状态好，今天也可能不错；如果前天很累，影响可能延续到今天。

使用滞后算子$L$（$L Y_t = Y_{t-1}$）：

$$
\phi(L)Y_t = c + \varepsilon_t
$$

其中$\phi(L) = 1 - \phi_1 L - \phi_2 L^2 - ... - \phi_p L^p$

#### 移动平均模型 MA(q)

当前值由过去$q$个预测误差的线性组合构成：

$$
Y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + ... + \theta_q \varepsilon_{t-q}
$$

**生活化比喻**：MA模型就像新闻报道。今天的头条不仅反映当前事件，还包含对前几天新闻遗漏或误报部分的"修正"。

#### 差分运算 I(d)

当时间序列非平稳时，我们通过差分使其平稳：

$$
\nabla Y_t = Y_t - Y_{t-1} = (1 - L)Y_t
$$

二阶差分：
$$
\nabla^2 Y_t = \nabla(\nabla Y_t) = Y_t - 2Y_{t-1} + Y_{t-2}
$$

**生活化比喻**：差分就像看速度而不是位置。即使你在移动（非平稳），你的速度可能是稳定的（平稳）。

#### 完整的ARIMA(p,d,q)模型

$$
(1 - \sum_{i=1}^p \phi_i L^i)(1 - L)^d Y_t = c + (1 + \sum_{j=1}^q \theta_j L^j)\varepsilon_t
$$

### 33.2.2 季节性ARIMA（SARIMA）

对于具有季节性模式的数据，使用SARIMA模型$ARIMA(p,d,q)(P,D,Q)_m$：

$$
\Phi_P(L^m)\phi_p(L)\nabla^d\nabla_m^D Y_t = \Theta_Q(L^m)\theta_q(L)\varepsilon_t
$$

其中$m$是季节周期（如月度数据的$m=12$）。

### 33.2.3 ARIMA建模流程（Box-Jenkins方法）

```
┌─────────────────────────────────────────────────────────┐
│           Box-Jenkins ARIMA建模流程                      │
├─────────────────────────────────────────────────────────┤
│  1. 可视化与平稳性检验                                    │
│     └── ADF检验、KPSS检验、可视化检查                    │
│                                                          │
│  2. 差分处理（如需要）                                    │
│     └── 使序列达到平稳状态                               │
│                                                          │
│  3. 模型识别                                              │
│     └── 通过ACF/PACF图确定p和q                          │
│                                                          │
│  4. 参数估计                                              │
│     └── 最大似然估计或最小二乘估计                       │
│                                                          │
│  5. 模型诊断                                              │
│     └── 残差分析、Ljung-Box检验                          │
│                                                          │
│  6. 预测                                                  │
│     └── 滚动预测、置信区间                               │
└─────────────────────────────────────────────────────────┘
```

### 33.2.4 ACF/PACF解读指南

| 模型 | ACF特征 | PACF特征 |
|------|---------|----------|
| AR(p) | 拖尾（逐渐衰减） | p阶后截尾 |
| MA(q) | q阶后截尾 | 拖尾 |
| ARMA(p,q) | 拖尾 | 拖尾 |

**拖尾**（Tail Off）：值缓慢衰减到零  
**截尾**（Cut Off）：值在某阶后突然降到接近零

---

## 33.3 Informer：突破长序列预测瓶颈

### 33.3.1 长序列预测的挑战

传统Transformer在长序列预测时面临**二次复杂度**问题：

$$
\text{Self-Attention复杂度} = O(L^2 \cdot d)
$$

当序列长度$L$从100增加到10000时，计算量增长10000倍！

**生活化比喻**：想象你在读一本1000页的小说。传统Transformer要求你记住每一页与其他所有页的关系——这会让你的大脑爆炸！而人类阅读时，只会关注与当前情节最相关的几页。

### 33.3.2 Informer的核心创新

Informer（AAAI 2021 Best Paper）提出三大创新：

1. **ProbSparse Self-Attention**：稀疏注意力，复杂度$O(L \log L)$
2. **Self-Attention Distilling**：自注意力蒸馏，压缩特征图
3. **Generative Style Decoder**：生成式解码器，一次前向传播输出长序列

### 33.3.3 ProbSparse Self-Attention

**关键洞察**：自注意力分数呈现**长尾分布**——少数"活跃"查询主导注意力，大多数"惰性"查询被忽略。

**数学定义**：

自注意力公式：
$$
A(Q, K, V) = \text{Softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

第$i$个查询$q_i$的注意力概率分布：
$$
pp(k_j|q_i) = \frac{\exp(q_i k_j^T / \sqrt{d})}{\sum_{l} \exp(q_i k_l^T / \sqrt{d})}
$$

**ProbSparse Attention的核心思想**：只选择"活跃"的查询进行计算。

**活跃查询的判别**：使用KL散度衡量查询的重要性：

$$
M(q_i, K) = \max_j \left\{\frac{q_i k_j^T}{\sqrt{d}}\right\} - \frac{1}{L_K} \sum_{j=1}^{L_K} \frac{q_i k_j^T}{\sqrt{d}}
$$

直观理解：如果某个查询与所有键的点积都差不多（均匀分布），那它就是"惰性"的；如果某个查询与特定键的点积特别大，那它就是"活跃"的。

**Top-u查询选择**：

$$
\bar{Q} = \text{Top-\textit{u}}(Q, M(Q, K))
$$

其中$u = c \cdot \log L_Q$，$c$是常数。

**ProbSparse Attention公式**：

$$
A(Q, K, V) = \text{Softmax}(\frac{\bar{Q}K^T}{\sqrt{d_k}})V
$$

复杂度从$O(L^2)$降低到$O(L \log L)$！

**生活化比喻**：想象你在一个 crowded party 中。ProbSparse Attention就像只关注那些"声音最大"的人，而不是试图听清所有人的对话。事实上，大多数人的闲聊并不重要。

### 33.3.4 Self-Attention Distilling

Informer在编码器中使用**蒸馏操作**，逐层压缩序列长度：

$$
X_{j+1}^t = \text{MaxPool}(\text{ELU}(\text{Conv1d}([X_j^t]_{AB})))
$$

其中$[X_j^t]_{AB}$表示注意力块的输出。

**蒸馏效果**：每一层将序列长度减半，大幅减少内存和计算。

```
Layer 1: L ──────> L/2
Layer 2: L/2 ────> L/4
Layer 3: L/4 ────> L/8
...
```

### 33.3.5 Generative Style Decoder

传统Transformer解码器采用自回归方式，存在**误差累积**问题：

$$
\hat{y}_{t+1} = f(y_1, ..., y_t) \rightarrow \hat{y}_{t+2} = f(y_1, ..., y_t, \hat{y}_{t+1})
$$

Informer的生成式解码器一次性输出整个预测序列：

$$
\hat{Y} = \text{Decoder}(X_{\text{enc}}, X_{\text{dec}})
$$

**生活化比喻**：传统解码器像是一个接一个地拼图，每拼错一块都会影响下一块。而生成式解码器像是直接看完整张图片的草图，一次画出全貌。

### 33.3.6 Informer架构总结

```
┌─────────────────────────────────────────────────────────────┐
│                     Informer架构                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Encoder (Stacked)                                         │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  ProbSparse Attention + Self-Attention Distilling   │   │
│   │  L ──> L/2 ──> L/4 ──> ... ──> L/2^N               │   │
│   └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│   Decoder                                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Generative Style Decoder                           │   │
│   │  - Multi-head Attention                             │   │
│   │  - Cross-attention with Encoder                     │   │
│   │  - Direct multi-step output                         │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 33.4 Autoformer：自相关机制革新

### 33.4.1 序列分解架构

Autoformer（NeurIPS 2021）的核心思想是将**序列分解**融入Transformer架构。

**时间序列分解**：

$$
X_t = Y_t + S_t + R_t
$$

其中：
- $Y_t$：趋势项（Trend）
- $S_t$：季节项（Seasonal）
- $R_t$：残差项（Residual）

传统方法将分解作为预处理步骤，Autoformer将其作为**内建模块**。

**分解模块**（Series Decomposition Block）：

$$
X_t = \text{AvgPool}(\text{Padding}(X))_t
$$
$$
S_t = X_t - Y_t
$$

使用移动平均提取趋势，原始序列减去趋势得到季节项。

### 33.4.2 Auto-Correlation机制

Autoformer最大的创新是用**自相关机制**（Auto-Correlation Mechanism）替代标准自注意力。

**核心洞察**：时间序列的依赖往往基于**周期性**，而非任意时间步之间的点相关。

**自相关函数**：

$$
\mathcal{R}_{XX}(\tau) = \lim_{L \to \infty} \frac{1}{L} \sum_{t=1}^{L} X_t X_{t-\tau}
$$

其中$\tau$是时间延迟（lag）。

**基于FFT的高效计算**：

利用Wiener-Khinchin定理，自相关可以通过FFT快速计算：

$$
\mathcal{R}_{XX}(\tau) = \mathcal{F}^{-1}(|\mathcal{F}(X)|^2)
$$

复杂度：$O(L \log L)$

**时间延迟聚合**（Time Delay Aggregation）：

Autoformer选择最重要的$k$个延迟$\tau_1, ..., \tau_k$，然后聚合相应的子序列：

$$
\hat{X}_t = \sum_{i=1}^{k} \text{Roll}(X, \tau_i) \cdot \alpha_i
$$

其中：
- $\text{Roll}(X, \tau)$：将序列$X$循环移动$\tau$位
- $\alpha_i$：延迟$\tau_i$的注意力权重（通过softmax归一化）

**生活化比喻**：想象你在预测明年的销售额。Autoformer不会看"昨天"、"前天"的具体数字，而是看"去年的同一天"、"前年的同一天"——发现周期性规律。就像农民知道"春种秋收"的循环，而不是只看连续几天的天气。

### 33.4.3 Autoformer的编码器-解码器结构

**Encoder**：
- 输入：季节项$S_{\text{en}}$
- 内部：趋势项逐步累加
- 使用Auto-Correlation机制

**Decoder**：
- 输入：
  - 季节部分：$S_{\text{de}} = \text{Concat}(S_{\text{en}}, X_0)$
  - 趋势部分：$T_{\text{de}}$（初始化为0）
- 输出：预测的趋势和季节成分

```
┌──────────────────────────────────────────────────────────────┐
│                   Autoformer架构                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Input X                                                     │
│    ├──> [Series Decomposition] ──> Trend + Seasonal         │
│                                                              │
│  Encoder (季节项)                                             │
│    ├──> Auto-Correlation Mechanism × N layers               │
│    └──> 输出: 编码后的季节特征                                │
│                                                              │
│  Decoder (分解结构)                                           │
│    ├──> 趋势累积分支 (Trend Accumulation)                    │
│    ├──> 季节细化分支 (Auto-Correlation)                      │
│    └──> 输出: 预测趋势 + 预测季节                             │
│                                                              │
│  Final: 趋势预测 + 季节预测 = 最终预测                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 33.4.4 Autoformer vs 标准Transformer

| 特性 | 标准Transformer | Autoformer |
|------|----------------|------------|
| 注意力机制 | 点向自注意力 | 基于周期的自相关 |
| 复杂度 | $O(L^2)$ | $O(L \log L)$ |
| 序列分解 | 预处理 | 内建模块 |
| 依赖建模 | 任意位置 | 周期性位置 |
| 信息聚合 | 点加权 | 子序列聚合 |

---

## 33.5 Reformer：局部敏感哈希注意力

### 33.5.1 Reformer的设计目标

Reformer（ICLR 2020）的目标是处理**超长序列**（长度可达100万），主要创新：

1. **LSH Attention**：局部敏感哈希注意力
2. **Reversible Layers**：可逆残差层
3. **Chunking**：分块处理

### 33.5.2 局部敏感哈希（LSH）

**核心思想**：相似的向量应该被分到同一个"桶"中，只在桶内计算注意力。

**LSH定义**：

哈希函数$h(x)$满足：
$$
P[h(x) = h(y)] \propto \text{sim}(x, y)
$$

即相似度越高的向量，被分到同一桶的概率越大。

**Reformer使用的LSH（随机投影）**：

$$
h(x) = \arg\max([xR; -xR])
$$

其中$R \in \mathbb{R}^{d \times b/2}$是随机矩阵，$[;]$表示拼接。

**为什么Query = Key？**

在标准注意力中，$Q \neq K$。Reformer设置$Q = K$（self-attention时本来就如此），这使得我们可以用同一个哈希函数处理两者。

### 33.5.3 LSH Attention算法

```python
def lsh_attention(Q, K, V, num_hashes, num_buckets):
    """
    LSH Attention的核心步骤
    """
    # 1. 对Q和K进行LSH哈希
    buckets = lsh_hash(Q, num_hashes, num_buckets)
    
    # 2. 按桶排序
    sorted_indices = sort_by_bucket(buckets)
    
    # 3. 分块处理
    chunks = split_into_chunks(sorted_indices, chunk_size)
    
    # 4. 在每个块及其相邻块内计算注意力
    output = []
    for chunk in chunks:
        # 允许当前块关注自己和前一个块
        attend_to = chunk + previous_chunk
        output.append(standard_attention(Q[chunk], K[attend_to], V[attend_to]))
    
    # 5. 恢复原始顺序
    return unsort(output, sorted_indices)
```

**复杂度分析**：
- 哈希：$O(L)$
- 排序：$O(L \log L)$（主要开销）
- 注意力：$O(L \cdot \text{chunk_size})$

总复杂度：$O(L \log L)$

### 33.5.4 可逆残差层（Reversible Layers）

传统Transformer需要存储每一层的激活值用于反向传播，内存开销$O(N \cdot L \cdot d)$。

Reversible Layers允许**从下一层重建当前层**，只需存储最后一层的激活：

$$
Y_1 = X_1 + F(X_2) \\
Y_2 = X_2 + G(Y_1)
$$

反向传播时：

$$
X_2 = Y_2 - G(Y_1) \\
X_1 = Y_1 - F(X_2)
$$

**内存节省**：从$O(N)$降到$O(1)$（层数维度）。

### 33.5.5 分块处理（Chunking）

前馈网络（FFN）的维度通常很大。Reformer将FFN的计算分块：

```
输入: [x1, x2, x3, x4, ..., xn]
        ↓
分块: [[x1, x2], [x3, x4], ...]
        ↓
分别计算FFN
        ↓
合并: [y1, y2, y3, y4, ..., yn]
```

---

## 33.6 其他重要变体

### 33.6.1 N-BEATS：神经基函数展开

N-BEATS（Neural Basis Expansion Analysis for Time Series, ICLR 2019）是纯深度学习模型，不使用序列特定的特征工程。

**核心思想**：将预测分解为多个基函数的线性组合：

$$
\hat{y}_{t+h} = \sum_{i=1}^{\text{stacks}} \sum_{j=1}^{\text{blocks}} g_{i,j}(h) \cdot f_{i,j}(\mathbf{y}_{1:t})
$$

**双残差堆叠**（Doubly Residual Stacking）：

每个Block输出两个值：
- **Backcast**：对输入的拟合（用于残差学习）
- **Forecast**：对未来预测（累加到最终输出）

$$
\mathbf{y}_{\text{backcast}}^{(l)} = \mathbf{y}^{(l-1)} - \hat{\mathbf{y}}_{\text{backcast}}^{(l)}
$$

```
输入 y
   │
   ├─── Block 1 ───┬──> Backcast ───┐
   │               └──> Forecast ────┤──> 累加
   ↓                                 │
残差 y - Backcast_1                  │
   │                                 │
   ├─── Block 2 ───┬──> Backcast ───┘
   │               └──> Forecast ───────> 累加
   ↓
   ...
```

**可解释性变体N-BEATS-I**：

- **趋势栈**：使用多项式基函数
- **季节栈**：使用傅里叶基函数

$$
g_t(h) = \sum_{i=0}^{p} \alpha_i h^i \quad \text{(趋势)}
$$

$$
g_s(h) = \sum_{i=0}^{\lfloor s/2 \rfloor} \gamma_i \cos(2\pi i h) + \delta_i \sin(2\pi i h) \quad \text{(季节)}
$$

### 33.6.2 N-HiTS：神经层次插值

N-HiTS（Neural Hierarchical Interpolation for Time Series）是N-BEATS的扩展，引入**层次插值**和**多速率采样**。

**核心创新**：

1. **多速率输入采样**：不同Block使用不同采样率
2. **层次插值**：不同频率的信号分别处理，然后插值合并

**插值函数**：

$$
g(\tau, \theta) = \theta[t_1] + \left(\frac{\theta[t_2] - \theta[t_1]}{t_2 - t_1}\right)(\tau - t_1)
$$

其中$t_1 = \arg\min_{t \in \mathcal{T}: t \leq \tau} \tau - t$，$t_2 = t_1 + 1/r_\ell$。

**表达能力比**（Expressiveness Ratio）：$r_\ell$控制每个Block的输出点数，不同Block使用不同的$r_\ell$实现多尺度建模。

### 33.6.3 DLinear & NLinear：简单线性模型的力量

论文"Are Transformers Effective for Time Series Forecasting?"（AAAI 2023）提出了一个惊人的发现：

**简单的线性模型可以击败复杂的Transformer模型！**

#### DLinear（Decomposition Linear）

$$
X = X_{\text{trend}} + X_{\text{seasonal}}
$$

$$
\hat{Y} = W_{\text{trend}} X_{\text{trend}} + W_{\text{seasonal}} X_{\text{seasonal}}
$$

**思想**：
1. 将时间序列分解为趋势和季节成分
2. 对每个成分分别应用一维卷积（线性变换）
3. 简单、高效、 surprisingly effective

#### NLinear（Normalization Linear）

$$
X' = X - X_{\text{last}}
$$
$$
\hat{Y} = W X' + X_{\text{last}}
$$

**思想**：
1. 减去序列的最后一个值（去除近期偏差）
2. 应用线性变换
3. 加回最后一个值

**核心洞察**：
Transformer的复杂架构（自注意力、多层、非线性）可能并非时间序列预测所必需。简单线性模型配合合适的预处理就能取得很好效果。

### 33.6.4 PatchTST：基于Patch的时间序列Transformer

PatchTST（ICLR 2023）受到ViT（Vision Transformer）启发，将时间序列分割成**Patches**作为输入token。

**Patching**：

将长度为$L$的序列分割为$N$个patch，每个patch长度为$P$，步长为$S$：

$$
N = \left\lfloor \frac{L - P}{S} \right\rfloor + 1
$$

**Channel Independence（通道独立）**：

不同于大多数模型同时处理所有变量，PatchTST采用**通道独立**策略——每个变量单独处理，最后合并结果。

这一反直觉的设计取得了更好的效果，可能是因为：
1. 避免了变量间的复杂交互
2. 减少了过拟合
3. 简化了学习问题

**架构**：

```
输入序列 (L, C)
   │
   ├──> 每个变量独立处理
   │       │
   │       ├──> Patch分割: (L,) ──> (N, P)
   │       │
   │       ├──> Projection + Positional Encoding
   │       │
   │       ├──> Transformer Encoder × N
   │       │
   │       └──> Flatten + Linear ──> 预测 (H,)
   │
   └──> 合并所有变量的预测
```

---

## 33.7 应用案例

### 33.7.1 股价预测

**场景**：预测未来7天的股票收盘价。

**特征工程**：
- 原始价格（OHLCV：开盘、最高、最低、收盘、成交量）
- 技术指标（MA、RSI、MACD）
- 收益率和对数收益率

**数据预处理**：

```python
# 标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)

# 创建滑动窗口
def create_sequences(data, seq_length, pred_length):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[(i + seq_length):(i + seq_length + pred_length), 0])  # 预测收盘价
    return np.array(X), np.array(y)
```

**模型选择**：
- 短期预测（1-5天）：N-BEATS、DLinear
- 中期预测（1-4周）：Informer、Autoformer
- 考虑周期性：Autoformer（内置分解）

**评价指标**：
- MSE/MAE：预测准确性
- 方向准确率：预测涨跌方向正确的比例
- 夏普比率：考虑风险调整后的收益

**⚠️ 重要提醒**：股价具有**弱有效性**，历史信息很难预测未来。任何模型都不可能持续击败市场。时间序列预测在股价上的应用应该谨慎，主要用于风险管理而非投机。

### 33.7.2 天气预测

**场景**：预测未来72小时的温度、湿度、风速。

**数据特点**：
- 强烈的周期性（日周期、季节周期）
- 多变量相关性（温度-湿度-气压）
- 长期依赖性（厄尔尼诺等气候模式）

**Autoformer的优势**：

Autoformer的序列分解特别适合天气数据：
- 趋势项捕捉全球变暖等长期变化
- 季节项捕捉日/年周期
- 自相关机制利用周期性依赖

**预处理步骤**：

```python
# 季节性分解
from statsmodels.tsa.seasonal import seasonal_decompose

# 对每个变量进行分解
result = seasonal_decompose(temperature, model='additive', period=24)  # 日周期
trend = result.trend
seasonal = result.seasonal
residual = result.resid
```

### 33.7.3 交通流量预测

**场景**：预测未来1小时各路段的车流量。

**数据特点**：
- 强烈的日周期（早晚高峰）
- 周周期（工作日vs周末）
- 空间相关性（相邻路段相互影响）

**Informer的应用**：

```python
# 多变量输入
features = ['flow', 'speed', 'occupancy', 'hour', 'day_of_week', 'is_holiday']

# 长序列输入（过去7天）
seq_length = 7 * 24  # 168小时
pred_length = 1  # 预测未来1小时
```

**Graph Neural Network扩展**：

考虑空间相关性，可以将道路网络建模为图：
- 节点：路段
- 边：道路连接
- 特征：历史流量

结合Transformer和GNN（如STGNN）可以进一步提升预测效果。

---

## 33.8 练习题

### 基础题

**练习 33.1**：ARIMA参数识别

给定以下ACF和PACF模式，识别合适的ARIMA模型参数：

(a) ACF拖尾，PACF在lag 2后截尾  
(b) ACF在lag 3后截尾，PACF拖尾  
(c) ACF和PACF都拖尾

**解答指引**：
- (a) $p=2, d=0, q=0$ → AR(2)
- (b) $p=0, d=0, q=3$ → MA(3)
- (c) 需要差分，可能是ARIMA(1,1,1)或类似模型

---

**练习 33.2**：时间序列分解

给定以下月销售额数据（单位：万元），进行季节性分解：

```
月份: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
销售额: [120, 115, 130, 140, 150, 160, 170, 165, 155, 145, 135, 125]
```

(a) 计算12期移动平均  
(b) 提取季节成分  
(c) 预测下一年1月的销售额

**解答指引**：
```python
import numpy as np
from scipy.ndimage import uniform_filter1d

sales = np.array([120, 115, 130, 140, 150, 160, 170, 165, 155, 145, 135, 125])

# (a) 移动平均
trend = uniform_filter1d(sales, size=12, mode='nearest')

# (b) 季节成分
seasonal = sales - trend

# (c) 简单预测: 趋势 + 去年同月季节成分
predicted_jan = trend[-1] + seasonal[0]
```

---

**练习 33.3**：复杂度计算

比较以下注意力机制处理序列长度$L=10000$时的计算量：

(a) 标准自注意力：$O(L^2)$  
(b) ProbSparse Attention：$O(L \log L)$，假设$c=5$  
(c) LSH Attention：$O(L \log L)$

**解答**：
- (a) $10000^2 = 10^8$ 次操作
- (b) $5 \times 10000 \times \log_2(10000) \approx 5 \times 10000 \times 13.3 \approx 6.6 \times 10^5$
- (c) 与(b)同阶

加速比：$10^8 / 6.6 \times 10^5 \approx 150$倍

---

### 进阶题

**练习 33.4**：自相关机制推导

证明Wiener-Khinchin定理：功率谱密度是自相关函数的傅里叶变换。

$$
S_{XX}(f) = \mathcal{F}\{\mathcal{R}_{XX}(\tau)\}
$$

**证明指引**：
1. 从自相关定义出发
2. 代入傅里叶变换表达式
3. 利用卷积定理
4. 化简得到功率谱

---

**练习 33.5**：Informer的ProbSparse Attention实现

实现ProbSparse Attention的核心逻辑（Top-u查询选择）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def probsparse_attention(query, key, value, factor=5):
    """
    ProbSparse Attention实现
    
    Args:
        query: (B, L_Q, D)
        key: (B, L_K, D)
        value: (B, L_K, D)
        factor: 稀疏因子
    
    Returns:
        output: (B, L_Q, D)
    """
    B, L_Q, D = query.shape
    _, L_K, _ = key.shape
    
    # 计算查询的重要性分数M
    # M(q_i, K) = max_j{q_i @ k_j} - mean_j{q_i @ k_j}
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(D)  # (B, L_Q, L_K)
    
    max_scores = torch.max(scores, dim=-1)[0]  # (B, L_Q)
    mean_scores = torch.mean(scores, dim=-1)   # (B, L_Q)
    M = max_scores - mean_scores  # (B, L_Q)
    
    # 选择Top-u查询
    u = int(factor * math.log(L_Q))
    _, top_indices = torch.topk(M, u, dim=-1)  # (B, u)
    
    # 只使用活跃的查询
    query_top = torch.gather(query, 1, top_indices.unsqueeze(-1).expand(-1, -1, D))  # (B, u, D)
    
    # 计算稀疏注意力
    attn_scores = torch.matmul(query_top, key.transpose(-2, -1)) / math.sqrt(D)  # (B, u, L_K)
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    output_top = torch.matmul(attn_weights, value)  # (B, u, D)
    
    # 将结果映射回原始查询维度
    output = torch.zeros_like(query)
    output.scatter_(1, top_indices.unsqueeze(-1).expand(-1, -1, D), output_top)
    
    return output
```

---

**练习 33.6**：N-BEATS的backcast机制

解释为什么N-BEATS使用backcast（后向预测）机制，并说明双残差连接的作用。

**解答要点**：

1. **Backcast的作用**：
   - 学习输入序列的表示
   - 通过残差连接逐层细化特征
   - 类似梯度提升中的残差学习

2. **双残差连接**：
   - 后向残差：$x_{l+1} = x_l - \hat{x}_l^{backcast}$
   - 前向累加：$\hat{y} = \sum_l \hat{y}_l^{forecast}$

3. **直观理解**：
   每个block专注学习前一层未能拟合的部分，同时贡献自己的预测。

---

### 挑战题

**练习 33.7**：Autoformer的Auto-Correlation实现

使用FFT实现Auto-Correlation机制：

```python
import torch
import torch.fft as fft

def auto_correlation(query, key, value):
    """
    基于FFT的自相关机制
    
    Args:
        query: (B, L, D)
        key: (B, L, D)
        value: (B, L, D)
    
    Returns:
        output: (B, L, D)
    """
    B, L, D = query.shape
    
    # 计算自相关 (使用FFT)
    # R_XX(tau) = IFFT(|FFT(X)|^2)
    
    # 对query和key的每个head计算
    query_fft = fft.rfft(query, dim=1)  # (B, L//2+1, D)
    key_fft = fft.rfft(key, dim=1)
    
    # 功率谱
    query_power = torch.abs(query_fft) ** 2
    key_power = torch.abs(key_fft) ** 2
    
    # 自相关 (时域)
    query_corr = fft.irfft(query_power, n=L, dim=1)  # (B, L, D)
    key_corr = fft.irfft(key_power, n=L, dim=1)
    
    # 选择Top-k延迟
    k = int(math.log(L))
    _, delays = torch.topk(query_corr + key_corr, k, dim=1)  # (B, k, D)
    
    # 时间延迟聚合
    outputs = []
    for i in range(k):
        delay = delays[:, i:i+1, :]  # (B, 1, D)
        # Roll操作: 根据delay平移value
        delayed_value = torch.gather(value, 1, delay.expand(-1, -1, D))
        outputs.append(delayed_value)
    
    # 加权聚合
    weights = F.softmax(torch.ones(B, k, D), dim=1)  # 简化版
    output = sum(w * out for w, out in zip(weights.unbind(1), outputs))
    
    return output
```

---

**练习 33.8**：长序列预测性能对比

设计一个实验比较以下模型在ETT（电力变压器温度）数据集上的性能：

1. ARIMA（基线）
2. Informer
3. Autoformer
4. DLinear
5. PatchTST

**实验设计要点**：

```python
# 实验配置
config = {
    'dataset': 'ETTh1',
    'seq_length': 96,
    'pred_length': [24, 48, 96, 168],  # 不同预测长度
    'metrics': ['MSE', 'MAE'],
    'models': ['ARIMA', 'Informer', 'Autoformer', 'DLinear', 'PatchTST']
}

# 结果分析维度
analysis = {
    '准确性': 'MSE/MAE对比',
    '效率': '训练时间、推理时间、内存占用',
    '可扩展性': '随序列长度的表现',
    '可解释性': '注意力可视化、分解成分'
}
```

**预期结果**：
- 短序列：线性模型（DLinear）可能最优
- 长序列：稀疏Transformer（Informer/Autoformer）表现更好
- 考虑周期性：Autoformer有优势

---

**练习 33.9**：设计一个混合模型

结合本章所学的不同技术，设计一个用于电力负荷预测的混合模型。

**模型设计**：

```
┌─────────────────────────────────────────────────────────────┐
│               电力负荷预测混合模型                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入层                                                      │
│    ├──> 历史负荷数据 ──> Patch分割                          │
│    └──> 外部特征（天气、日期类型）                             │
│                                                             │
│  特征编码器                                                  │
│    ├──> 序列分解模块（趋势+季节）                             │
│    └──> 通道独立编码（每个变量独立处理）                       │
│                                                             │
│  主干网络                                                    │
│    ├──> Auto-Correlation层（捕捉周期性）                     │
│    ├──> ProbSparse Attention（长程依赖）                     │
│    └──> N-BEATS Block（多尺度分解）                          │
│                                                             │
│  输出层                                                      │
│    ├──> 趋势预测分支（Linear）                                │
│    ├──> 季节预测分支（Auto-Correlation）                     │
│    └──> 融合层                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**设计理由**：
1. **Patch分割**：从ViT借鉴，增强局部语义
2. **序列分解**：Autoformer的核心，处理趋势和季节
3. **通道独立**：PatchTST发现，简单但有效
4. **ProbSparse**：Informer贡献，处理长序列
5. **Auto-Correlation**：捕捉电力数据的日/周周期
6. **N-BEATS Block**：可解释的多尺度分解

---

## 参考文献

Box, G. E. P., & Jenkins, G. M. (1976). *Time series analysis: Forecasting and control*. Holden-Day.

Brockwell, P. J., & Davis, R. A. (2016). *Introduction to time series and forecasting* (3rd ed.). Springer.

Challu, C., Olivares, K. G., Oreshkin, B. N., Garza, F., Mergenthaler-Canseco, M., & Dubrawski, A. (2022). N-HiTS: Neural hierarchical interpolation for time series forecasting. *arXiv preprint arXiv:2201.12886*.

Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and practice* (2nd ed.). OTexts.

Kitaev, N., Kaiser, L., & Levskaya, A. (2020). Reformer: The efficient transformer. In *International Conference on Learning Representations (ICLR)*.

Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers. In *International Conference on Learning Representations (ICLR)*.

Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. In *International Conference on Learning Representations (ICLR)*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998-6008.

Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. In *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 22419-22430.

Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are transformers effective for time series forecasting? In *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(9), 11121-11128.

Zhang, Y., & Yan, J. (2023). Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In *International Conference on Learning Representations (ICLR)*.

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(12), 11106-11115. (AAAI 2021 Best Paper)

Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., & Jin, R. (2022). FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. In *International Conference on Machine Learning (ICML)*, 27268-27286.

---

## 本章小结

### 核心概念回顾

1. **时间序列基础**
   - 平稳性、ACF/PACF、分解（趋势+季节+残差）
   - ARIMA模型：经典统计方法

2. **Informer**（AAAI 2021 Best Paper）
   - ProbSparse Attention：$O(L \log L)$复杂度
   - Self-Attention Distilling：逐层压缩
   - Generative Decoder：一次输出多步预测

3. **Autoformer**（NeurIPS 2021）
   - 内建序列分解：趋势+季节
   - Auto-Correlation机制：基于周期的依赖
   - 子序列级聚合（而非点级）

4. **Reformer**（ICLR 2020）
   - LSH Attention：局部敏感哈希
   - Reversible Layers：可逆残差
   - 适用于超长序列（百万级）

5. **其他变体**
   - N-BEATS：神经基函数展开
   - N-HiTS：层次插值
   - DLinear/NLinear：简单但有效
   - PatchTST：基于Patch的Transformer

### 选型指南

| 场景 | 推荐模型 | 理由 |
|------|----------|------|
| 短序列(<100) | DLinear/NLinear | 简单高效 |
| 长序列(>1000) | Informer/Reformer | 稀疏注意力 |
| 强周期性 | Autoformer | 内置分解+自相关 |
| 需要可解释性 | N-BEATS-I | 显式趋势+季节 |
| 超长序列(>1万) | Reformer | LSH + 可逆层 |
| 多变量预测 | PatchTST | 通道独立策略 |

### 实践建议

1. **从简单开始**：先用DLinear建立基线
2. **检查周期性**：ACF/PACF分析，有周期考虑Autoformer
3. **序列长度**：长序列（>1000）必须考虑稀疏注意力
4. **多尺度建模**：重要时考虑N-HiTS或层次结构
5. **通道策略**：多变量时尝试通道独立 vs 通道混合

### 未来方向

- **大模型+时序**：GPT-style预训练 + 时序微调
- **时空建模**：结合GNN的时空Transformer
- **概率预测**：不确定性量化（CRPS、分位数预测）
- **自动机器学习**：AutoML for Time Series

---

*"预测是困难的，尤其是关于未来的预测。" —— 尼尔斯·玻尔*

*但有了强大的Transformer变体，我们至少可以让机器从历史中学到更多，看得更远。*
