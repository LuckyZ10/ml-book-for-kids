# 第三十三章：时序预测与Transformer变体

> *"时间是最公平的资源，每个人每天都有24小时。但如果你能预测时间的流动，就能在变化中把握先机。"
> —— 本章将带你探索如何让AI学会"预见未来"*

## 本章学习目标

- 理解时间序列数据的本质特征（趋势、季节性、周期性）
- 掌握Informer的ProbSparse注意力机制与长序列预测
- 深入理解Autoformer的自相关机制与序列分解
- 学习Reformer的LSH注意力与高效Transformer
- 实践股价预测、电力负荷预测、天气预报等真实场景
- 了解时序预测领域的最新进展

---

## 33.1 时间序列：数据的"心跳"

### 33.1.1 什么是时间序列

想象你在听一首音乐。每一个音符都是一个时间点上的数值，而整首歌曲就是一段**时间序列(Time Series)**。

**定义**：时间序列是按照时间顺序排列的数据点序列，通常表示为：

$$\mathbf{X} = \{x_1, x_2, ..., x_T\}$$

其中 $x_t$ 表示时刻 $t$ 的观测值，$T$ 是序列长度。

**费曼比喻**：时间序列就像心跳💓。每次跳动是一个数据点，跳动之间的间隔是时间。医生通过分析心电图（时间序列）来诊断健康状态，就像数据科学家通过分析时间序列来预测未来趋势。

### 33.1.2 时间序列的三大特征

#### 1. 趋势(Trend)：长期走向

想象你观察一个孩子从1岁到18岁的身高变化。虽然每年增长的速度不同，但总体呈上升趋势。

**数学表达**：
$$T_t = \alpha + \beta t + \epsilon_t$$

其中 $\alpha$ 是截距，$\beta$ 是斜率，$\epsilon_t$ 是随机噪声。

#### 2. 季节性(Seasonality)：周期性波动

想象一家冰淇淋店的日销售额。每年夏天销量飙升，冬天下降——这就是季节性。

**数学表达**：
$$S_t = \sum_{k=1}^{K} \left[ a_k \sin\left(\frac{2\pi k t}{P}\right) + b_k \cos\left(\frac{2\pi k t}{P}\right) \right]$$

其中 $P$ 是周期长度（如24小时、7天、12个月），$K$ 是谐波次数。

#### 3. 周期性(Cyclical)：非固定周期的波动

经济周期就是典型的周期性——经济扩张和收缩没有固定的时间表，但确实存在规律。

**费曼比喻**：
- 趋势 = 河流的流向（向东还是向西）
- 季节性 = 潮汐（每天两次涨落）
- 周期性 = 气候变化（有冷暖交替，但不固定）

### 33.1.3 时间序列分解

经典的时间序列模型将数据分解为三个成分：

$$X_t = T_t + S_t + R_t$$

或乘法形式：

$$X_t = T_t \times S_t \times R_t$$

其中：
- $T_t$：趋势成分
- $S_t$：季节性成分  
- $R_t$：残差/不规则成分

**Python实现：时间序列分解**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class TimeSeriesDecomposer:
    """时间序列分解器：将时间序列分解为趋势、季节性和残差"""
    
    def __init__(self, period=12):
        """
        Args:
            period: 季节性周期长度
        """
        self.period = period
        
    def moving_average(self, x, window):
        """计算移动平均"""
        return np.convolve(x, np.ones(window)/window, mode='same')
    
    def decompose_additive(self, x):
        """加法分解
        
        X_t = T_t + S_t + R_t
        """
        n = len(x)
        
        # 1. 提取趋势：使用移动平均
        trend = self.moving_average(x, self.period)
        
        # 处理边界
        half_window = self.period // 2
        for i in range(half_window):
            trend[i] = np.mean(x[:i+half_window+1])
            trend[n-1-i] = np.mean(x[n-i-half_window-1:])
        
        # 2. 去趋势
        detrended = x - trend
        
        # 3. 提取季节性
        seasonal = np.zeros(n)
        for i in range(self.period):
            seasonal[i::self.period] = np.mean(detrended[i::self.period])
        
        # 4. 残差
        residual = x - trend - seasonal
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'original': x
        }
    
    def decompose_multiplicative(self, x):
        """乘法分解
        
        X_t = T_t × S_t × R_t
        """
        # 对数转换后使用加法分解
        log_x = np.log(x + 1e-8)  # 避免log(0)
        result = self.decompose_additive(log_x)
        
        return {
            'trend': np.exp(result['trend']),
            'seasonal': np.exp(result['seasonal']),
            'residual': np.exp(result['residual']),
            'original': x
        }
    
    def plot_decomposition(self, result, title="Time Series Decomposition"):
        """可视化分解结果"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        components = ['original', 'trend', 'seasonal', 'residual']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for ax, comp, color in zip(axes, components, colors):
            ax.plot(result[comp], color=color, linewidth=1.5)
            ax.set_ylabel(comp.capitalize(), fontsize=11)
            ax.grid(True, alpha=0.3)
            
            if comp == 'original':
                ax.set_title(title, fontsize=14, fontweight='bold')
        
        axes[-1].set_xlabel('Time', fontsize=11)
        plt.tight_layout()
        return fig


# 演示：生成合成时间序列并分解
def generate_synthetic_series(n=500, seed=42):
    """生成包含趋势、季节性和噪声的合成时间序列"""
    np.random.seed(seed)
    t = np.arange(n)
    
    # 趋势：非线性增长
    trend = 0.001 * t**1.5 + 0.1 * t
    
    # 季节性：多种周期叠加
    seasonal = 10 * np.sin(2 * np.pi * t / 50) + \
               5 * np.sin(2 * np.pi * t / 12) + \
               3 * np.cos(2 * np.pi * t / 7)
    
    # 周期性：低频波动
    cyclical = 8 * np.sin(2 * np.pi * t / 200)
    
    # 噪声
    noise = np.random.normal(0, 2, n)
    
    return trend + seasonal + cyclical + noise


# 运行演示
if __name__ == "__main__":
    # 生成数据
    data = generate_synthetic_series(n=500)
    
    # 分解
    decomposer = TimeSeriesDecomposer(period=50)
    result = decomposer.decompose_additive(data)
    
    # 可视化
    fig = decomposer.plot_decomposition(result)
    plt.savefig('time_series_decomposition.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("时间序列分解完成！")
    print(f"原始序列均值: {np.mean(data):.2f}")
    print(f"趋势成分均值: {np.mean(result['trend']):.2f}")
    print(f"季节性成分均值: {np.mean(result['seasonal']):.4f} (应接近0)")
    print(f"残差标准差: {np.std(result['residual']):.2f}")
```

### 33.1.4 时间序列预测任务

时间序列预测的核心任务是：给定历史观测值 $\{x_1, ..., x_T\}$，预测未来值 $\{x_{T+1}, ..., x_{T+H}\}$，其中 $H$ 是预测范围(Horizon)。

**预测类型**：

| 类型 | 预测范围 | 应用场景 |
|------|---------|---------|
| 短期预测 | $H \leq 24$ | 股票分钟线、交通流量 |
| 中期预测 | $24 < H \leq 168$ | 日电力负荷、周销量 |
| 长期预测 | $H > 168$ | 气候预测、年度规划 |

**挑战**：
1. **误差累积**：多步预测时，误差会逐步放大
2. **非平稳性**：统计特性随时间变化
3. **长程依赖**：需要捕捉遥远的过去信息
4. **多变量关联**：多个时间序列相互影响

---

## 33.2 从Transformer到长序列预测

### 33.2.1 标准Transformer的问题

回顾Transformer的自注意力机制：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**复杂度分析**：
- 计算复杂度：$O(L^2 \cdot d)$ —— 序列长度的平方！
- 内存复杂度：$O(L^2)$ —— 需要存储注意力矩阵

当序列长度 $L = 1000$ 时，注意力矩阵有 $10^6$ 个元素；当 $L = 10000$ 时，就有 $10^8$ 个元素！

**费曼比喻**：想象你在一个拥挤的会议室里。标准Transformer要求每个人都和所有人握手打招呼。100人需要约5000次握手，10000人就需要约5000万次握手！这显然不可行。

### 33.2.2 长序列预测的核心需求

时间序列预测往往需要极长的历史上下文：

- **电力负荷预测**：过去一年数据（8760小时）预测未来一天
- **股价预测**：过去500个交易日预测未来10天
- **天气预报**：过去30天数据预测未来7天

**关键洞察**：在长序列中，并非所有时间点都同等重要。就像回忆过去，你能清晰记得的是"关键时刻"而非每一秒。

---

## 33.3 Informer：稀疏注意力革命

### 33.3.1 ProbSparse自注意力机制

Informer（Zhou et al., 2021, AAAI最佳论文）的核心创新是**ProbSparse Self-Attention**，通过识别"活跃查询"来降低复杂度。

**关键洞察**：
在softmax注意力中，每个查询 $q_i$ 对所有键的注意力分布为：

$$A(q_i, K, V) = \sum_j \frac{\exp(q_i^T k_j / \sqrt{d})}{\sum_l \exp(q_i^T k_l / \sqrt{d})} v_j$$

研究发现，大多数查询的注意力分布接近**均匀分布**（即对所有键的关注度差不多），只有少数"活跃查询"有高度集中的注意力分布。

**ProbSparse的选择策略**：

对于每个查询 $q_i$，定义其"稀疏性度量"：

$$M(q_i, K) = \max_j \left( \frac{q_i^T k_j}{\sqrt{d}} \right) - \frac{1}{L_K} \sum_{j=1}^{L_K} \frac{q_i^T k_j}{\sqrt{d}}$$

这个度量表示查询的最大注意力分数与平均注意力分数的差异。差异越大，说明这个查询越"活跃"。

**复杂度**：$O(L \ln L)$ —— 接近线性！

### 33.3.2 自注意力蒸馏

Informer还引入了**Self-attention Distilling**机制，逐层压缩序列长度：

$$X_{j+1} = \text{MaxPool}\left( \text{ELU}\left( \text{Conv1d}\left( [X_j]_{AB} \right) \right) \right)$$

其中 $[X_j]_{AB}$ 表示注意力块输出。

**效果**：每经过一层，序列长度减半，大幅降低内存占用。

### 33.3.3 生成式解码器

传统Transformer使用自回归解码（逐token生成），Informer采用**一次性生成**所有预测值：

$$\hat{X}_{T+1:T+H} = \text{Decoder}\left( X_{1:T}, X_{T+1:T+H}^{\text{placeholder}} \right)$$

**优势**：
- 避免误差累积
- 推理速度大幅提升
- 支持超长预测范围

### 33.3.4 Informer完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ProbAttention(nn.Module):
    """ProbSparse自注意力机制"""
    
    def __init__(self, d_model, n_heads, dropout=0.1, factor=5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor  # 采样因子
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, attn_mask=None):
        """
        Args:
            Q, K, V: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size = Q.size(0)
        
        # 线性投影并分头
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # ProbSparse注意力
        scores, indices = self.prob_sparse_attention(Q, K)
        
        # 应用mask（如果有）
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        # 对选中的查询计算完整注意力
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, V)
        
        # 合并头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_O(output)
        
        return output
    
    def prob_sparse_attention(self, Q, K):
        """ProbSparse注意力计算"""
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape
        
        # 计算查询稀疏性分数
        # 采样部分键来估计
        K_sample = K[:, :, :L_K // self.factor, :]
        
        # 计算Q与采样K的相似度
        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1)) / math.sqrt(D)
        
        # 计算每个查询的稀疏性度量
        max_sim = Q_K_sample.max(dim=-1)[0]  # (B, H, L_Q)
        mean_sim = Q_K_sample.mean(dim=-1)   # (B, H, L_Q)
        sparse_score = max_sim - mean_sim     # 稀疏性分数
        
        # 选择Top-u个活跃查询
        u = min(self.factor * int(math.log(L_K)), L_Q)
        _, top_indices = torch.topk(sparse_score, u, dim=-1)  # (B, H, u)
        
        # 只计算活跃查询的完整注意力
        Q_sparse = torch.gather(Q, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, D))
        scores_sparse = torch.matmul(Q_sparse, K.transpose(-2, -1)) / math.sqrt(D)
        
        # 扩展回原始大小（非活跃查询设为均匀分布）
        scores = torch.ones(B, H, L_Q, L_K, device=Q.device) * (1.0 / L_K)
        scores = scores.scatter(2, top_indices.unsqueeze(-1).expand(-1, -1, -1, L_K), scores_sparse)
        
        return scores, top_indices


class AttentionLayer(nn.Module):
    """注意力层包装"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = ProbAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None):
        attn_out = self.attention(x, x, x, attn_mask)
        x = self.norm(x + self.dropout(attn_out))
        return x


class EncoderLayer(nn.Module):
    """Informer编码器层"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = AttentionLayer(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 自注意力
        x = self.attention(x)
        
        # 前馈网络
        ff_out = self.feed_forward(x)
        x = self.norm(x + self.dropout(ff_out))
        
        return x


class ConvLayer(nn.Module):
    """自注意力蒸馏层：通过卷积和池化压缩序列"""
    
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.activation = nn.ELU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.conv(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = x.transpose(1, 2)  # (batch, seq_len//2, d_model)
        return x


class InformerEncoder(nn.Module):
    """Informer编码器：多层注意力 + 蒸馏"""
    
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.conv_layers = nn.ModuleList([
            ConvLayer(d_model) for _ in range(n_layers - 1)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            压缩后的表示
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.conv_layers):
                x = self.conv_layers[i](x)
        
        return self.norm(x)


class InformerDecoder(nn.Module):
    """Informer生成式解码器"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention1 = AttentionLayer(d_model, n_heads, dropout)
        self.attention2 = AttentionLayer(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output):
        """
        Args:
            x: 解码器输入 (batch, pred_len, d_model)
            enc_output: 编码器输出
        """
        # 自注意力
        attn1 = self.attention1(x)
        x = self.norm1(x + self.dropout(attn1))
        
        # 交叉注意力
        # 这里简化处理，实际应使用标准交叉注意力
        
        # 前馈
        ff = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff))
        
        return x


class Informer(nn.Module):
    """Informer模型：长序列时间序列预测"""
    
    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_encoder_layers=3,
        n_decoder_layers=2,
        dropout=0.1
    ):
        super().__init__()
        
        # 输入嵌入
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 编码器
        self.encoder = InformerEncoder(
            d_model, n_heads, d_ff, n_encoder_layers, dropout
        )
        
        # 解码器
        self.decoder_layers = nn.ModuleList([
            InformerDecoder(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        # 输出投影
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, x_enc, x_dec=None):
        """
        Args:
            x_enc: 编码器输入 (batch, enc_len, input_dim)
            x_dec: 解码器输入（可选，用于生成式预测）
        Returns:
            predictions: (batch, pred_len, output_dim)
        """
        # 编码器
        enc_out = self.input_embedding(x_enc)
        enc_out = self.pos_encoding(enc_out)
        enc_out = self.encoder(enc_out)
        
        # 解码器（简化版本）
        # 实际Informer使用特殊的生成式解码器
        if x_dec is None:
            # 使用编码器输出的最后一点作为解码器起点
            batch_size = x_enc.size(0)
            pred_len = 24  # 假设预测24步
            x_dec = enc_out[:, -1:, :].expand(-1, pred_len, -1)
        
        dec_out = x_dec
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out)
        
        # 输出
        output = self.output_projection(dec_out)
        
        return output


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# 使用示例
if __name__ == "__main__":
    # 模型参数
    batch_size = 32
    enc_len = 96    # 输入序列长度
    pred_len = 24   # 预测长度
    input_dim = 7   # 输入维度（多变量时间序列）
    output_dim = 7  # 输出维度
    
    # 创建模型
    model = Informer(
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_encoder_layers=2,
        n_decoder_layers=1
    )
    
    # 随机输入
    x_enc = torch.randn(batch_size, enc_len, input_dim)
    
    # 前向传播
    predictions = model(x_enc)
    
    print(f"输入形状: {x_enc.shape}")
    print(f"预测形状: {predictions.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 33.4 Autoformer：自相关机制突破

### 33.4.1 序列分解架构

Autoformer（Wu et al., 2021, NeurIPS）的核心思想是将时间序列分解为**趋势-周期**成分，并分别建模。

**深度分解模块**：

$$\mathcal{X} = \mathcal{X}_{\text{trend}} + \mathcal{X}_{\text{seasonal}}$$

Autoformer在每一层都执行分解，让模型逐步细化对趋势和季节性的理解。

### 33.4.2 自相关机制

传统自注意力计算的是**点级**相似度：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

Autoformer的**自相关机制(Auto-Correlation)**计算的是**周期级**相似度：

$$\text{Auto-Correlation}(\mathcal{X}) = \sum_{\tau \in \mathcal{T}} \text{R}_{\mathcal{X}}(\tau) \cdot \mathcal{X}_{\tau}$$

其中 $\mathcal{T}$ 是选出的top-k周期延迟，$\text{R}_{\mathcal{X}}(\tau)$ 是基于周期的相似度：

$$\text{R}_{\mathcal{X}}(\tau) = \frac{1}{L} \sum_{t=1}^{L} \mathcal{X}_t \cdot \mathcal{X}_{t-\tau}$$

**费曼比喻**：想象你在分析一首重复播放的歌曲。标准注意力是在比较"当前音符和过去每个音符的相似度"，而自相关机制是在问"这首歌的重复周期是什么？"，然后基于这些周期来预测。

### 33.4.3 高效的频域计算

自相关可以通过**快速傅里叶变换(FFT)**高效计算：

$$\mathcal{R}(\tau) = \text{IFFT}\left( |\text{FFT}(\mathcal{X})|^2 \right)$$

**复杂度**：从 $O(L^2)$ 降至 $O(L \log L)$

### 33.4.4 Autoformer核心实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SeriesDecomp(nn.Module):
    """序列分解模块：将输入分解为趋势和季节性成分"""
    
    def __init__(self, kernel_size=25):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size//2)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            seasonal: 季节性成分
            trend: 趋势成分
        """
        # 移动平均提取趋势
        x_permute = x.permute(0, 2, 1)  # (batch, d_model, seq_len)
        trend = self.moving_avg(x_permute).permute(0, 2, 1)
        
        # 残差为季节性
        seasonal = x - trend
        
        return seasonal, trend


class AutoCorrelation(nn.Module):
    """自相关机制：基于周期的注意力"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V):
        """
        Args:
            Q, K, V: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
            delays: 选中的周期延迟
        """
        batch_size = Q.size(0)
        
        # 线性投影
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)
        
        # 计算自相关（使用FFT加速）
        delays, delays_scores = self.time_delay_agg(Q, K, V)
        
        # 聚合
        output = self.aggregate(V, delays, delays_scores)
        
        output = self.W_O(output)
        
        return output, delays
    
    def time_delay_agg(self, Q, K, V):
        """计算时间延迟聚合的周期"""
        # 简化实现：使用平均池化近似
        # 实际应使用FFT计算自相关
        
        B, L, D = Q.shape
        
        # 计算查询和键的FFT
        Q_fft = torch.fft.rfft(Q, dim=1)
        K_fft = torch.fft.rfft(K, dim=1)
        
        # 计算功率谱
        res = Q_fft * torch.conj(K_fft)
        
        # IFFT得到自相关
        corr = torch.fft.irfft(res, dim=1)
        
        # 选择top-k周期
        k = min(3, L // 2)
        _, top_delays = torch.topk(corr.mean(dim=-1), k, dim=-1)
        
        return top_delays, corr
    
    def aggregate(self, V, delays, scores):
        """基于延迟聚合值"""
        B, L, D = V.shape
        k = delays.size(-1)
        
        # 初始化输出
        output = torch.zeros_like(V)
        
        # 对每个选中的延迟进行聚合
        for i in range(k):
            delay = delays[:, i].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            
            # 滚动V
            V_shifted = torch.roll(V, shifts=delay[0, 0, 0].item(), dims=1)
            
            # 加权聚合
            weight = scores[:, :, i:i+1].mean(dim=1, keepdim=True)  # (B, 1, 1)
            output += weight * V_shifted / k
        
        return output


class AutoCorrelationLayer(nn.Module):
    """自相关层包装"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.auto_correlation = AutoCorrelation(d_model, n_heads, dropout)
        self.decomp = SeriesDecomp()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 自相关
        attn_out, _ = self.auto_correlation(x, x, x)
        x = x + self.dropout(attn_out)
        
        # 分解
        seasonal, trend = self.decomp(x)
        x = self.norm(seasonal)
        
        return x, trend


class AutoformerEncoderLayer(nn.Module):
    """Autoformer编码器层"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.auto_corr = AutoCorrelationLayer(d_model, n_heads, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.decomp2 = SeriesDecomp()
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # 自相关 + 分解
        seasonal, trend1 = self.auto_corr(x)
        
        # 前馈 + 分解
        ff_out = self.feed_forward(seasonal)
        seasonal, trend2 = self.decomp2(ff_out)
        seasonal = self.norm(seasonal)
        
        # 累积趋势
        trend = trend1 + trend2
        
        return seasonal, trend


class AutoformerDecoderLayer(nn.Module):
    """Autoformer解码器层"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.auto_corr1 = AutoCorrelationLayer(d_model, n_heads, dropout)
        self.auto_corr2 = AutoCorrelationLayer(d_model, n_heads, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.decomp3 = SeriesDecomp()
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_out):
        # 自相关1
        seasonal, trend1 = self.auto_corr1(x)
        
        # 自相关2（交叉）
        seasonal_cross, trend2 = self.auto_corr2(seasonal)
        
        # 前馈
        ff_out = self.feed_forward(seasonal_cross)
        seasonal, trend3 = self.decomp3(ff_out)
        seasonal = self.norm(seasonal)
        
        trend = trend1 + trend2 + trend3
        
        return seasonal, trend


class Autoformer(nn.Module):
    """Autoformer模型：基于自相关的时间序列预测"""
    
    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_encoder_layers=2,
        n_decoder_layers=1,
        dropout=0.1,
        pred_len=24
    ):
        super().__init__()
        
        self.pred_len = pred_len
        
        # 输入嵌入
        self.enc_embedding = nn.Linear(input_dim, d_model)
        self.dec_embedding = nn.Linear(input_dim, d_model)
        
        # 编码器
        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # 解码器
        self.decoder_layers = nn.ModuleList([
            AutoformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        # 投影层
        self.seasonal_projection = nn.Linear(d_model, output_dim)
        self.trend_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, x_enc, x_dec=None):
        """
        Args:
            x_enc: 编码器输入 (batch, enc_len, input_dim)
            x_dec: 解码器输入（占位符）
        Returns:
            predictions: (batch, pred_len, output_dim)
        """
        # 编码器
        enc_out = self.enc_embedding(x_enc)
        
        enc_seasonal = enc_out
        enc_trend = torch.zeros_like(enc_out)
        
        for layer in self.encoder_layers:
            enc_seasonal, trend = layer(enc_seasonal)
            enc_trend = enc_trend + trend
        
        # 解码器
        # 使用趋势的最后一点作为初始化
        trend_init = enc_trend[:, -1:, :]
        
        if x_dec is None:
            # 创建解码器占位符
            batch_size = x_enc.size(0)
            x_dec = torch.zeros(batch_size, self.pred_len, x_enc.size(-1), 
                               device=x_enc.device)
        
        dec_out = self.dec_embedding(x_dec)
        
        # 加入趋势初始化
        dec_out = dec_out + trend_init.expand(-1, self.pred_len, -1)
        
        dec_seasonal = dec_out
        dec_trend = torch.zeros_like(dec_out)
        
        for layer in self.decoder_layers:
            dec_seasonal, trend = layer(dec_seasonal, enc_seasonal)
            dec_trend = dec_trend + trend
        
        # 分别预测季节性和趋势
        seasonal_pred = self.seasonal_projection(dec_seasonal)
        trend_pred = self.trend_projection(dec_trend)
        
        # 相加得到最终预测
        predictions = seasonal_pred + trend_pred
        
        return predictions


# 使用示例
if __name__ == "__main__":
    # 模型参数
    batch_size = 32
    enc_len = 96
    pred_len = 24
    input_dim = 7
    output_dim = 7
    
    # 创建模型
    model = Autoformer(
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        pred_len=pred_len
    )
    
    # 随机输入
    x_enc = torch.randn(batch_size, enc_len, input_dim)
    
    # 前向传播
    predictions = model(x_enc)
    
    print(f"输入形状: {x_enc.shape}")
    print(f"预测形状: {predictions.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 33.5 Reformer：局部敏感哈希注意力

### 33.5.1 LSH注意力原理

Reformer（Kitaev et al., 2020, ICLR）使用**局部敏感哈希(Locality Sensitive Hashing, LSH)**来近似标准注意力，将复杂度从 $O(L^2)$ 降至 $O(L \log L)$。

**核心思想**：
1. 使用哈希函数将相似的向量映射到相同的"桶"中
2. 只在同一个桶内的向量之间计算注意力

**LSH函数**：

$$h(x) = \arg\max([xR; -xR])$$

其中 $R \in \mathbb{R}^{d \times b/2}$ 是随机投影矩阵，$b$ 是桶的数量。

**费曼比喻**：想象你在一个大型图书馆找书。LSH就像是图书分类系统——相似的书（相似的查询/键）被放在同一个书架上。你只需要在每个书架内查找，而不是遍历整个图书馆。

### 33.5.2 可逆残差层

Reformer还引入了**可逆残差层(Reversible Residual Layers)**，允许只存储最后一层的激活值，大幅减少内存使用。

标准残差：$y = x + F(x)$

可逆残差：
- $y_1 = x_1 + F(x_2)$
- $y_2 = x_2 + G(y_1)$

反向时可以从 $y_1, y_2$ 恢复 $x_1, x_2$：
- $x_2 = y_2 - G(y_1)$
- $x_1 = y_1 - F(x_2)$

### 33.5.3 Reformer LSH注意力实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSHAttention(nn.Module):
    """局部敏感哈希注意力"""
    
    def __init__(self, d_model, n_heads, n_hashes=4, bucket_size=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V):
        """
        Args:
            Q, K, V: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = Q.shape
        
        # 线性投影
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)
        
        # LSH注意力
        output = self.lsh_attention(Q, K, V)
        
        output = self.W_O(output)
        
        return output
    
    def lsh_attention(self, Q, K, V):
        """执行LSH注意力"""
        batch_size, seq_len, d_model = Q.shape
        
        # 合并Q和K用于LSH（共享投影）
        # 实际Reformer使用相同的投影
        
        # 简化实现：随机分桶
        # 实际应使用LSH哈希
        n_buckets = max(1, seq_len // self.bucket_size)
        
        # 随机哈希（演示用）
        hashes = torch.randint(0, n_buckets, (batch_size, seq_len), device=Q.device)
        
        output = torch.zeros_like(Q)
        
        # 在每个桶内计算注意力
        for b in range(n_buckets):
            # 找到属于桶b的位置
            mask = (hashes == b)  # (batch, seq_len)
            
            for i in range(batch_size):
                bucket_indices = mask[i].nonzero(as_tuple=True)[0]
                
                if len(bucket_indices) == 0:
                    continue
                
                # 提取桶内的Q, K, V
                Q_bucket = Q[i, bucket_indices].unsqueeze(0)
                K_bucket = K[i, bucket_indices].unsqueeze(0)
                V_bucket = V[i, bucket_indices].unsqueeze(0)
                
                # 计算注意力
                scores = torch.matmul(Q_bucket, K_bucket.transpose(-2, -1)) / np.sqrt(self.d_k)
                attn = F.softmax(scores, dim=-1)
                attn = self.dropout(attn)
                
                bucket_out = torch.matmul(attn, V_bucket)
                
                # 放回输出
                output[i, bucket_indices] = bucket_out.squeeze(0)
        
        return output


class ReformerLayer(nn.Module):
    """Reformer层：LSH注意力 + 可逆残差"""
    
    def __init__(self, d_model, n_heads, d_ff, n_hashes=4, dropout=0.1):
        super().__init__()
        self.lsh_attention = LSHAttention(d_model, n_heads, n_hashes, dropout=dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSH注意力
        attn_out = self.lsh_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class Reformer(nn.Module):
    """Reformer模型：用于长序列的内存高效Transformer"""
    
    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_layers=6,
        n_hashes=4,
        dropout=0.1
    ):
        super().__init__()
        
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        self.layers = nn.ModuleList([
            ReformerLayer(d_model, n_heads, d_ff, n_hashes, dropout)
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            output: (batch, seq_len, output_dim)
        """
        x = self.input_embedding(x)
        
        for layer in self.layers:
            x = layer(x)
        
        output = self.output_projection(x)
        
        return output


# 使用示例
if __name__ == "__main__":
    batch_size = 8
    seq_len = 4096  # 长序列！
    input_dim = 64
    output_dim = 64
    
    model = Reformer(
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_layers=4,
        n_hashes=4
    )
    
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 33.6 现代时序预测模型

### 33.6.1 N-BEATS：神经基扩展

N-BEATS（Oreshkin et al., 2019, ICLR）是纯深度学习模型，首次挑战传统统计方法在时序预测中的主导地位。

**核心思想**：将预测分解为多个基函数的叠加：

$$\hat{y}_{t+h} = \sum_{i=1}^{\text{stacks}} \sum_{j=1}^{\text{blocks}} g_{i,j}(h) \cdot f_{i,j}(\mathbf{x})$$

其中 $g_{i,j}(h)$ 是基函数（如趋势基、季节基），$f_{i,j}$ 是神经网络学习的权重。

### 33.6.2 深度时间序列模型对比

| 模型 | 核心创新 | 复杂度 | 适用场景 |
|------|---------|--------|---------|
| **Informer** | ProbSparse注意力 | $O(L \ln L)$ | 超长序列预测 |
| **Autoformer** | 自相关机制 | $O(L \ln L)$ | 强周期性数据 |
| **Reformer** | LSH注意力 | $O(L \ln L)$ | 内存受限场景 |
| **N-BEATS** | 神经基扩展 | $O(L)$ | 单变量预测 |
| **N-HiTS** | 多速率采样 | $O(L)$ | 多尺度预测 |
| **FEDformer** | 频域注意力 | $O(L)$ | 频域特征丰富 |

---

## 33.7 应用案例：电力负荷预测

### 33.7.1 问题背景

电力公司需要预测未来24小时的电力需求，以优化发电计划和电网调度。

**数据特征**：
- 强季节性（日周期、周周期、年周期）
- 受天气、节假日影响
- 多变量（温度、湿度、历史负荷等）

### 33.7.2 完整实现

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class ElectricityDataset(Dataset):
    """电力负荷数据集"""
    
    def __init__(self, data, seq_len=96, pred_len=24):
        """
        Args:
            data: DataFrame，包含多变量时间序列
            seq_len: 输入序列长度
            pred_len: 预测长度
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 标准化
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(data.values)
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len
    
    def __getitem__(self, idx):
        # 输入序列
        x = self.data[idx:idx+self.seq_len]
        
        # 预测目标
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)


class TimeSeriesPredictor:
    """时间序列预测器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向传播
            predictions = self.model(batch_x)
            
            # 计算损失
            loss = self.criterion(predictions, batch_y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
                
                predictions_list.append(predictions.cpu().numpy())
                targets_list.append(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        # 计算MAE和RMSE
        preds = np.concatenate(predictions_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        
        mae = np.mean(np.abs(preds - targets))
        rmse = np.sqrt(np.mean((preds - targets)**2))
        
        return {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse
        }


# 演示：使用合成数据进行训练
def train_demo():
    """训练演示"""
    # 生成合成数据
    np.random.seed(42)
    n_samples = 10000
    
    # 模拟电力负荷数据
    t = np.arange(n_samples)
    
    # 日周期
    daily = 10 * np.sin(2 * np.pi * t / 24)
    # 周周期
    weekly = 5 * np.sin(2 * np.pi * t / (24*7))
    # 趋势
    trend = 0.001 * t
    # 噪声
    noise = np.random.normal(0, 1, n_samples)
    
    load = 50 + daily + weekly + trend + noise
    
    # 温度（影响负荷）
    temp = 20 + 10 * np.sin(2 * np.pi * t / 24 + np.pi/4) + np.random.normal(0, 2, n_samples)
    
    # 创建DataFrame
    data = pd.DataFrame({
        'load': load,
        'temperature': temp,
        'hour': t % 24,
        'day_of_week': (t // 24) % 7
    })
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # 创建数据集
    train_dataset = ElectricityDataset(train_data, seq_len=96, pred_len=24)
    test_dataset = ElectricityDataset(test_data, seq_len=96, pred_len=24)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建模型（使用Informer）
    model = Informer(
        input_dim=4,
        output_dim=4,
        d_model=128,
        n_heads=4,
        d_ff=512,
        n_encoder_layers=2,
        n_decoder_layers=1
    )
    
    # 训练
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = TimeSeriesPredictor(model, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("开始训练...")
    for epoch in range(10):
        train_loss = predictor.train_epoch(train_loader, optimizer)
        metrics = predictor.evaluate(test_loader)
        
        print(f"Epoch {epoch+1}/10")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test MAE: {metrics['mae']:.4f}")
        print(f"  Test RMSE: {metrics['rmse']:.4f}")
    
    return predictor


if __name__ == "__main__":
    predictor = train_demo()
```

---

## 33.8 练习题

### 基础练习

**33.1 时间序列分解**

给定一个时间序列 $X_t = 2t + 5\sin(\frac{\pi t}{6}) + \epsilon_t$，其中 $t = 1, 2, ..., 100$，$\epsilon_t \sim N(0, 1)$。

- 手动计算移动平均趋势（窗口大小为12）
- 提取季节性成分
- 验证：趋势 + 季节性 + 残差 ≈ 原始序列

**33.2 Informer复杂度分析**

标准Transformer的注意力复杂度为 $O(L^2 d)$，Informer的ProbSparse注意力为 $O(L \ln L \cdot d)$。

计算当 $L = 1000, 10000, 100000$ 时的计算量比例，理解为什么Informer适合长序列。

**33.3 自相关计算**

给定序列 $X = [1, 2, 3, 2, 1, 2, 3, 2, 1]$，手动计算其自相关函数：

$$R(\tau) = \frac{1}{N-\tau} \sum_{t=1}^{N-\tau} X_t \cdot X_{t+\tau}$$

对于 $\tau = 1, 2, 3$，并解释结果的含义。

### 进阶练习

**33.4 实现简化版Autoformer**

基于本章代码，实现一个简化版Autoformer：
- 只包含1层编码器和1层解码器
- 使用移动平均代替FFT计算自相关
- 在合成数据上测试，比较与标准Transformer的效果

**33.5 多步预测策略对比**

时间序列多步预测有三种策略：
1. **直接多输出**：一次性预测所有未来值
2. **递归预测**：逐步预测，用预测值作为下一步输入
3. **Seq2Seq**：使用编码器-解码器架构

实现这三种策略，在相同数据上比较：
- 预测精度（MSE、MAE）
- 推理速度
- 误差累积情况

**33.6 LSH注意力分析**

LSH注意力的核心是哈希函数。设计一个实验：
- 生成两组向量：相似向量和随机向量
- 实现一个简单的LSH函数（随机投影）
- 计算碰撞概率：相似向量 vs 不相似向量
- 分析：为什么LSH能保持注意力的准确性？

### 挑战练习

**33.7 时序预测竞赛**

选择一个公开时间序列数据集（如Kaggle的电力负荷、天气预测）：
- 实现至少3种模型（统计方法、Informer、Autoformer）
- 设计特征工程策略
- 进行超参数调优
- 提交到Kaggle或记录排行榜成绩
- 撰写技术报告，分析哪种方法最有效及原因

**33.8 多变量时序预测的挑战**

多变量时序预测（如预测多个地区的电力负荷）面临独特挑战：
- 变量间的相关性建模
- 不同变量的不同采样频率
- 缺失值处理

设计一个解决方案：
- 使用图神经网络建模变量间关系
- 实现处理缺失值的策略
- 与基线方法比较

**33.9 时序预测的置信区间**

点预测之外，置信区间对决策同样重要。实现一个概率时序预测模型：
- 使用分位数回归输出预测区间
- 或使用深度集成（Deep Ensemble）估计不确定性
- 评估：预测区间是否覆盖真实值？区间宽度是否合理？

---

## 本章小结

### 核心概念回顾

| 概念 | 关键内容 |
|------|---------|
| **时间序列分解** | 趋势 + 季节性 + 残差 = 原始序列 |
| **ProbSparse注意力** | 只关注活跃查询，复杂度$O(L\ln L)$ |
| **自相关机制** | 基于周期延迟的注意力，适合强周期性数据 |
| **LSH注意力** | 局部敏感哈希分桶，大幅降低内存 |
| **生成式解码** | 一次性生成所有预测值，避免误差累积 |

### 模型选择指南

- **序列长度 > 5000**：选择Informer或Reformer
- **强周期性数据（电力、气象）**：选择Autoformer
- **内存受限环境**：选择Reformer
- **单变量预测**：考虑N-BEATS/N-HiTS
- **需要可解释性**：选择基于分解的模型（Autoformer）

### 关键公式总结

**时间序列加法分解**：
$$X_t = T_t + S_t + R_t$$

**Informer ProbSparse注意力**：
$$M(q_i, K) = \max_j \left( \frac{q_i^T k_j}{\sqrt{d}} \right) - \text{mean}_j \left( \frac{q_i^T k_j}{\sqrt{d}} \right)$$

**Autoformer自相关**：
$$\text{Auto-Correlation}(\mathcal{X}) = \sum_{\tau \in \mathcal{T}} R_{\mathcal{X}}(\tau) \cdot \mathcal{X}_{\tau}$$

**LSH哈希函数**：
$$h(x) = \arg\max([xR; -xR])$$

---

## 参考文献

Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. *arXiv preprint arXiv:1905.10437*.

Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020). Reformer: The efficient transformer. *International Conference on Learning Representations*.

Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. *Advances in Neural Information Processing Systems*, 34, 22419-22430.

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(12), 11106-11115.

Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., & Jin, R. (2022). FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. *International Conference on Machine Learning*, 27268-27286. PMLR.

Challu, C., Olivares, K. G., Oreshkin, B. N., Garza, F., Mergenthaler, T., & Winkler, T. (2023). N-HiTS: Neural hierarchical interpolation for time series forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*, 37(6), 6989-6997.

---

*本章完。你已掌握了长序列时间序列预测的核心技术，从Informer的稀疏注意力到Autoformer的自相关机制，这些技术正在改变电力、金融、气象等领域的预测能力。*
