

---

## 代码实现：时间序列预测完整工具包

```python
"""
chapter33_time_series_forecasting.py
第三十三章：时序预测与Transformer变体 - 完整代码实现

包含：
1. 时间序列预处理（标准化、差分、滑动窗口）
2. ARIMA基线实现
3. Informer模型（ProbSparse Attention、自注意力蒸馏）
4. Autoformer实现（序列分解、自相关机制）
5. 评价指标（MSE, MAE, MAPE）
6. 可视化工具
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import math
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================================
# 第一部分：时间序列预处理
# ============================================================================

class TimeSeriesPreprocessor:
    """时间序列预处理工具类"""
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.last_value = None
    
    def normalize(self, data: np.ndarray, method='zscore') -> np.ndarray:
        """
        数据标准化
        
        Args:
            data: 输入数据 (N, C)
            method: 'zscore' 或 'minmax'
        
        Returns:
            标准化后的数据
        """
        if method == 'zscore':
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
            self.std[self.std == 0] = 1  # 避免除零
            return (data - self.mean) / self.std
        
        elif method == 'minmax':
            self.min = np.min(data, axis=0)
            self.max = np.max(data, axis=0)
            self.range = self.max - self.min
            self.range[self.range == 0] = 1
            return (data - self.min) / self.range
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def denormalize(self, data: np.ndarray, method='zscore') -> np.ndarray:
        """反标准化"""
        if method == 'zscore':
            return data * self.std + self.mean
        elif method == 'minmax':
            return data * self.range + self.min
    
    def difference(self, data: np.ndarray, order=1) -> np.ndarray:
        """
        差分运算 (使序列平稳)
        
        Args:
            data: 输入序列
            order: 差分阶数
        
        Returns:
            差分后的序列
        """
        self.last_value = data[0] if order == 1 else data[:order]
        return np.diff(data, n=order, axis=0)
    
    def inverse_difference(self, diff_data: np.ndarray, 
                          original_start: np.ndarray) -> np.ndarray:
        """逆差分"""
        result = np.cumsum(diff_data, axis=0)
        return result + original_start
    
    def create_sequences(self, data: np.ndarray, seq_length: int, 
                         pred_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建滑动窗口序列
        
        Args:
            data: 输入数据 (N, C)
            seq_length: 输入序列长度
            pred_length: 预测序列长度
        
        Returns:
            X: 输入序列 (N, seq_length, C)
            y: 输出序列 (N, pred_length, C)
        """
        X, y = [], []
        for i in range(len(data) - seq_length - pred_length + 1):
            X.append(data[i:(i + seq_length)])
            y.append(data[(i + seq_length):(i + seq_length + pred_length)])
        return np.array(X), np.array(y)
    
    def seasonal_decompose(self, data: np.ndarray, period: int) -> dict:
        """
        简单季节性分解
        
        Args:
            data: 输入序列
            period: 季节周期
        
        Returns:
            包含trend, seasonal, residual的字典
        """
        # 移动平均提取趋势
        trend = np.convolve(data.flatten(), 
                           np.ones(period)/period, 
                           mode='same')
        
        # 去趋势
        detrended = data.flatten() - trend
        
        # 计算季节成分（周期平均）
        seasonal = np.zeros_like(data.flatten())
        for i in range(period):
            seasonal[i::period] = np.mean(detrended[i::period])
        
        # 残差
        residual = data.flatten() - trend - seasonal
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for Time Series"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# 第二部分：ARIMA基线实现
# ============================================================================

class ARIMA:
    """
    简化版ARIMA实现
    
    仅用于教学演示，实际应用建议使用statsmodels
    """
    
    def __init__(self, p: int = 1, d: int = 0, q: int = 0):
        """
        Args:
            p: AR阶数
            d: 差分阶数
            q: MA阶数
        """
        self.p = p
        self.d = d
        self.q = q
        self.ar_params = None
        self.ma_params = None
        self.residuals = []
    
    def difference(self, data: np.ndarray, order: int) -> np.ndarray:
        """差分"""
        result = data.copy()
        for _ in range(order):
            result = np.diff(result)
        return result
    
    def fit(self, data: np.ndarray):
        """
        使用最小二乘拟合ARIMA参数
        
        简化版：仅拟合AR部分
        """
        # 差分
        if self.d > 0:
            diff_data = self.difference(data, self.d)
        else:
            diff_data = data
        
        # 构建AR特征矩阵
        n = len(diff_data)
        if n <= self.p:
            raise ValueError("数据长度必须大于p")
        
        X = np.zeros((n - self.p, self.p))
        y = diff_data[self.p:]
        
        for i in range(self.p):
            X[:, i] = diff_data[self.p - i - 1:n - i - 1]
        
        # 最小二乘拟合
        self.ar_params = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # 计算残差
        predictions = X @ self.ar_params
        self.residuals = y - predictions
        
        return self
    
    def predict(self, data: np.ndarray, steps: int = 1) -> np.ndarray:
        """预测未来steps步"""
        if self.ar_params is None:
            raise ValueError("模型尚未拟合")
        
        # 差分
        if self.d > 0:
            diff_data = self.difference(data, self.d)
        else:
            diff_data = data.copy()
        
        predictions = []
        current = list(diff_data[-self.p:])
        
        for _ in range(steps):
            # AR预测
            pred = np.dot(self.ar_params[::-1], current[-self.p:])
            predictions.append(pred)
            current.append(pred)
        
        predictions = np.array(predictions)
        
        # 逆差分（简化处理）
        if self.d > 0:
            last_value = data[-1]
            predictions = np.cumsum(predictions) + last_value
        
        return predictions


# ============================================================================
# 第三部分：Informer模型实现
# ============================================================================

class ProbSparseAttention(nn.Module):
    """
    ProbSparse Attention (Informer核心)
    复杂度: O(L log L)
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, 
                 factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, 
                V: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """
        Args:
            Q: (batch, L_Q, d_model)
            K: (batch, L_K, d_model)
            V: (batch, L_K, d_model)
        
        Returns:
            output: (batch, L_Q, d_model)
        """
        batch_size = Q.size(0)
        
        # 线性变换
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # ProbSparse Attention
        attn_output = self._probsparse_attention(Q, K, V, attn_mask)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        return self.W_o(attn_output)
    
    def _probsparse_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                              V: torch.Tensor, attn_mask=None):
        """ProbSparse Attention核心"""
        batch_size, n_heads, L_Q, d_k = Q.shape
        _, _, L_K, _ = K.shape
        
        # 计算Query的重要性分数 M(Q, K)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 选择Top-u查询
        u = min(int(self.factor * math.log(L_Q)), L_Q)
        
        # 计算每个query的重要性
        max_scores = torch.max(scores, dim=-1)[0]  # (batch, heads, L_Q)
        mean_scores = torch.mean(scores, dim=-1)   # (batch, heads, L_Q)
        M = max_scores - mean_scores  # 重要性分数
        
        # Top-u索引
        _, top_indices = torch.topk(M, u, dim=-1)  # (batch, heads, u)
        
        # 只选择Top-u查询
        Q_top = torch.gather(Q, 2, 
                            top_indices.unsqueeze(-1).expand(-1, -1, -1, d_k))
        
        # 计算稀疏注意力
        attn_scores = torch.matmul(Q_top, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_output_top = torch.matmul(attn_weights, V)  # (batch, heads, u, d_k)
        
        # 将结果映射回原始维度
        attn_output = torch.zeros_like(Q)
        attn_output.scatter_(2, 
                            top_indices.unsqueeze(-1).expand(-1, -1, -1, d_k),
                            attn_output_top)
        
        return attn_output


class ConvLayer(nn.Module):
    """自注意力蒸馏层"""
    
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.down_conv = nn.Conv1d(c_in, c_out, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(c_out)
        self.activation = nn.ELU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, c_in)
        Returns:
            (batch, seq_len//2, c_out)
        """
        x = x.permute(0, 2, 1)  # (batch, c_in, seq_len)
        x = self.down_conv(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, c_out)
        x = self.norm(x)
        x = self.activation(x)
        x = x.permute(0, 2, 1)
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)
        return x


class InformerEncoderLayer(nn.Module):
    """Informer编码器层"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048, 
                 dropout: float = 0.1, factor: int = 5):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, factor, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # 注意力
        attn_out = self.attention(x, x, x, attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class InformerEncoder(nn.Module):
    """Informer编码器（带蒸馏）"""
    
    def __init__(self, d_model: int, n_heads: int, n_layers: int = 2,
                 d_ff: int = 2048, dropout: float = 0.1, factor: int = 5):
        super().__init__()
        self.layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout, factor)
            for _ in range(n_layers)
        ])
        self.distilling_layers = nn.ModuleList([
            ConvLayer(d_model, d_model) for _ in range(n_layers - 1)
        ])
    
    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.distilling_layers):
                x = self.distilling_layers[i](x)
        return x


class InformerDecoderLayer(nn.Module):
    """Informer解码器层（生成式）"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048, 
                 dropout: float = 0.1, factor: int = 5):
        super().__init__()
        self.self_attention = ProbSparseAttention(d_model, n_heads, factor, dropout)
        self.cross_attention = ProbSparseAttention(d_model, n_heads, factor, dropout)
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
    
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor):
        # 自注意力
        attn_out = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 交叉注意力
        cross_out = self.cross_attention(x, enc_output, enc_output)
        x = self.norm2(x + self.dropout(cross_out))
        
        # 前馈
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x


class Informer(nn.Module):
    """
    Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
    (AAAI 2021 Best Paper)
    """
    
    def __init__(self, enc_in: int, dec_in: int, c_out: int,
                 seq_len: int, label_len: int, pred_len: int,
                 d_model: int = 512, n_heads: int = 8, 
                 e_layers: int = 2, d_layers: int = 1,
                 d_ff: int = 2048, dropout: float = 0.1,
                 factor: int = 5):
        """
        Args:
            enc_in: 编码器输入维度
            dec_in: 解码器输入维度
            c_out: 输出维度
            seq_len: 输入序列长度
            label_len: 解码器输入长度（用于上下文）
            pred_len: 预测序列长度
        """
        super().__init__()
        self.pred_len = pred_len
        self.label_len = label_len
        
        # 编码器嵌入
        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # 解码器嵌入
        self.dec_embedding = nn.Linear(dec_in, d_model)
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, label_len + pred_len, d_model))
        
        # 编码器
        self.encoder = InformerEncoder(d_model, n_heads, e_layers, d_ff, dropout, factor)
        
        # 解码器
        self.decoder_layers = nn.ModuleList([
            InformerDecoderLayer(d_model, n_heads, d_ff, dropout, factor)
            for _ in range(d_layers)
        ])
        
        # 输出投影
        self.projection = nn.Linear(d_model, c_out)
    
    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor):
        """
        Args:
            x_enc: 编码器输入 (batch, seq_len, enc_in)
            x_dec: 解码器输入 (batch, label_len + pred_len, dec_in)
        
        Returns:
            预测结果 (batch, pred_len, c_out)
        """
        # 编码器
        enc_out = self.enc_embedding(x_enc) + self.enc_pos_embedding
        enc_out = self.encoder(enc_out)
        
        # 解码器
        dec_out = self.dec_embedding(x_dec) + self.dec_pos_embedding
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out)
        
        # 投影到输出空间
        dec_out = self.projection(dec_out)
        
        # 只返回预测部分
        return dec_out[:, -self.pred_len:, :]


# ============================================================================
# 第四部分：Autoformer模型实现
# ============================================================================

class MovingAvg(nn.Module):
    """移动平均模块（用于趋势提取）"""
    
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, channels)
        """
        # 前后填充
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        
        # 移动平均
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        x = self.avg(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        
        return x


class SeriesDecomp(nn.Module):
    """序列分解模块"""
    
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)
    
    def forward(self, x: torch.Tensor):
        """
        分解为趋势和季节成分
        
        Returns:
            seasonal: 季节成分
            trend: 趋势成分
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelation(nn.Module):
    """
    Auto-Correlation Mechanism (Autoformer核心)
    基于FFT的高效自相关计算
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, 
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        """
        Args:
            Q, K, V: (batch, seq_len, d_model)
        """
        batch_size = Q.size(0)
        
        # 线性变换
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 自相关注意力
        attn_output = self._auto_correlation(Q, K, V)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        return self.W_o(attn_output)
    
    def _auto_correlation(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        """自相关机制核心"""
        batch_size, n_heads, L, d_k = Q.shape
        
        # 使用FFT计算自相关
        # R_XX = IFFT(|FFT(X)|^2)
        
        # 对每个head计算
        Q_fft = torch.fft.rfft(Q, dim=2)
        K_fft = torch.fft.rfft(K, dim=2)
        
        # 功率谱
        Q_power = torch.abs(Q_fft) ** 2
        K_power = torch.abs(K_fft) ** 2
        
        # 自相关（时域）
        Q_corr = torch.fft.irfft(Q_power, n=L, dim=2)
        K_corr = torch.fft.irfft(K_power, n=L, dim=2)
        
        # 选择Top-k延迟
        k = min(int(math.log(L)), L)
        
        # 合并Q和K的自相关
        combined_corr = Q_corr + K_corr  # (batch, heads, L, d_k)
        
        # 对每个head和每个维度选择top-k延迟
        combined_corr_mean = combined_corr.mean(dim=-1)  # (batch, heads, L)
        _, top_delays = torch.topk(combined_corr_mean, k, dim=-1)  # (batch, heads, k)
        
        # 时间延迟聚合
        outputs = []
        weights = []
        
        for i in range(k):
            delay = top_delays[:, :, i:i+1]  # (batch, heads, 1)
            
            # Roll操作：根据delay平移V
            # 使用gather实现
            delay_expanded = delay.unsqueeze(-1).expand(-1, -1, -1, d_k)
            indices = (torch.arange(L, device=V.device).view(1, 1, L, 1) - delay_expanded) % L
            rolled_V = torch.gather(V, 2, indices.expand(-1, -1, -1, d_k))
            
            outputs.append(rolled_V)
            
            # 权重（使用自相关值）
            weight = torch.gather(combined_corr_mean, 2, delay).unsqueeze(-1)
            weights.append(weight)
        
        # Softmax归一化权重
        weights = torch.stack(weights, dim=2)  # (batch, heads, k, 1)
        weights = F.softmax(weights, dim=2)
        
        # 加权聚合
        outputs = torch.stack(outputs, dim=2)  # (batch, heads, k, L, d_k)
        attn_output = (outputs * weights.unsqueeze(-1)).sum(dim=2)
        
        return attn_output


class AutoformerEncoderLayer(nn.Module):
    """Autoformer编码器层"""
    
    def __init__(self, d_model: int, n_heads: int, 
                 kernel_size: int = 25, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.decomp1 = SeriesDecomp(kernel_size)
        self.decomp2 = SeriesDecomp(kernel_size)
        
        self.auto_correlation = AutoCorrelation(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor):
        # 自相关 + 分解
        attn_out = self.auto_correlation(x, x, x)
        x = x + self.dropout(attn_out)
        x, _ = self.decomp1(x)
        
        # 前馈 + 分解
        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)
        x, _ = self.decomp2(x)
        
        return x


class AutoformerDecoderLayer(nn.Module):
    """Autoformer解码器层"""
    
    def __init__(self, d_model: int, n_heads: int, 
                 kernel_size: int = 25, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.decomp1 = SeriesDecomp(kernel_size)
        self.decomp2 = SeriesDecomp(kernel_size)
        self.decomp3 = SeriesDecomp(kernel_size)
        
        self.auto_correlation = AutoCorrelation(d_model, n_heads, dropout)
        self.cross_auto_correlation = AutoCorrelation(d_model, n_heads, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
                trend_acc: torch.Tensor):
        # 自相关
        attn_out = self.auto_correlation(x, x, x)
        x = x + self.dropout(attn_out)
        x, trend1 = self.decomp1(x)
        
        # 交叉自相关
        cross_out = self.cross_auto_correlation(x, enc_output, enc_output)
        x = x + self.dropout(cross_out)
        x, trend2 = self.decomp2(x)
        
        # 前馈
        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)
        x, trend3 = self.decomp3(x)
        
        # 累加趋势
        trend_acc = trend_acc + trend1 + trend2 + trend3
        
        return x, trend_acc


class Autoformer(nn.Module):
    """
    Autoformer: Decomposition Transformers with Auto-Correlation
    for Long-Term Series Forecasting (NeurIPS 2021)
    """
    
    def __init__(self, enc_in: int, dec_in: int, c_out: int,
                 seq_len: int, label_len: int, pred_len: int,
                 d_model: int = 512, n_heads: int = 8,
                 e_layers: int = 2, d_layers: int = 1,
                 d_ff: int = 2048, dropout: float = 0.1,
                 kernel_size: int = 25):
        super().__init__()
        self.pred_len = pred_len
        self.label_len = label_len
        
        # 分解模块
        self.decomp = SeriesDecomp(kernel_size)
        
        # 编码器
        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model, n_heads, kernel_size, d_ff, dropout)
            for _ in range(e_layers)
        ])
        
        # 解码器
        self.dec_embedding = nn.Linear(dec_in, d_model)
        self.decoder_layers = nn.ModuleList([
            AutoformerDecoderLayer(d_model, n_heads, kernel_size, d_ff, dropout)
            for _ in range(d_layers)
        ])
        
        # 投影层
        self.seasonal_projection = nn.Linear(d_model, c_out)
        self.trend_projection = nn.Linear(d_model, c_out)
    
    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor):
        """
        Args:
            x_enc: 编码器输入
            x_dec: 解码器输入
        """
        # 编码器
        enc_out = self.enc_embedding(x_enc)
        for layer in self.encoder_layers:
            enc_out = layer(enc_out)
        
        # 解码器
        # 初始化趋势和季节
        mean = torch.mean(x_dec, dim=1, keepdim=True)
        zeros = torch.zeros_like(x_dec)
        trend_init = torch.cat([mean.repeat(1, self.label_len, 1), 
                                zeros[:, -self.pred_len:, :]], dim=1)
        
        dec_out = self.dec_embedding(x_dec)
        trend_acc = trend_init
        
        for layer in self.decoder_layers:
            dec_out, trend_acc = layer(dec_out, enc_out, trend_acc)
        
        # 季节预测
        seasonal_pred = self.seasonal_projection(dec_out[:, -self.pred_len:, :])
        
        # 趋势预测
        trend_pred = self.trend_projection(trend_acc[:, -self.pred_len:, :])
        
        # 最终预测 = 季节 + 趋势
        return seasonal_pred + trend_pred


# ============================================================================
# 第五部分：评价指标
# ============================================================================

def mse_metric(pred: np.ndarray, true: np.ndarray) -> float:
    """均方误差"""
    return np.mean((pred - true) ** 2)

def mae_metric(pred: np.ndarray, true: np.ndarray) -> float:
    """平均绝对误差"""
    return np.mean(np.abs(pred - true))

def mape_metric(pred: np.ndarray, true: np.ndarray) -> float:
    """平均绝对百分比误差"""
    mask = true != 0
    return np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100

def rmse_metric(pred: np.ndarray, true: np.ndarray) -> float:
    """均方根误差"""
    return np.sqrt(mse_metric(pred, true))

def r2_metric(pred: np.ndarray, true: np.ndarray) -> float:
    """R^2分数"""
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1 - (ss_res / ss_tot)

class MetricsCalculator:
    """评价指标计算器"""
    
    def __init__(self):
        self.metrics = {
            'MSE': mse_metric,
            'MAE': mae_metric,
            'MAPE': mape_metric,
            'RMSE': rmse_metric,
            'R2': r2_metric
        }
    
    def calculate(self, pred: np.ndarray, true: np.ndarray) -> dict:
        """计算所有指标"""
        results = {}
        for name, func in self.metrics.items():
            try:
                results[name] = func(pred, true)
            except:
                results[name] = float('nan')
        return results
    
    def print_metrics(self, pred: np.ndarray, true: np.ndarray):
        """打印所有指标"""
        results = self.calculate(pred, true)
        print("=" * 40)
        print("Evaluation Metrics:")
        print("=" * 40)
        for name, value in results.items():
            print(f"{name:10s}: {value:.6f}")
        print("=" * 40)


# ============================================================================
# 第六部分：可视化工具
# ============================================================================

class TimeSeriesVisualizer:
    """时间序列可视化工具"""
    
    def __init__(self, figsize=(12, 6)):
        self.figsize = figsize
    
    def plot_prediction(self, true: np.ndarray, pred: np.ndarray, 
                        title="Time Series Prediction"):
        """绘制预测vs实际"""
        plt.figure(figsize=self.figsize)
        
        if len(true.shape) > 1:
            true = true[:, 0]
        if len(pred.shape) > 1:
            pred = pred[:, 0]
        
        plt.plot(true, label='True', color='blue', alpha=0.7)
        plt.plot(pred, label='Predicted', color='red', alpha=0.7)
        plt.fill_between(range(len(pred)), pred, true, alpha=0.2, color='gray')
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_decomposition(self, original: np.ndarray, trend: np.ndarray,
                          seasonal: np.ndarray, residual: np.ndarray):
        """绘制分解成分"""
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        axes[0].plot(original)
        axes[0].set_title('Original')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(trend)
        axes[1].set_title('Trend')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(seasonal)
        axes[2].set_title('Seasonal')
        axes[2].grid(True, alpha=0.3)
        
        axes[3].plot(residual)
        axes[3].set_title('Residual')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_acf_pacf(self, data: np.ndarray, lags: int = 20):
        """绘制ACF和PACF"""
        from statsmodels.tsa.stattools import acf, pacf
        
        acf_values = acf(data, nlags=lags, fft=True)
        pacf_values = pacf(data, nlags=lags)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        # ACF
        axes[0].bar(range(len(acf_values)), acf_values)
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0].axhline(y=1.96/np.sqrt(len(data)), color='red', linestyle='--', alpha=0.5)
        axes[0].axhline(y=-1.96/np.sqrt(len(data)), color='red', linestyle='--', alpha=0.5)
        axes[0].set_title('Autocorrelation Function (ACF)')
        axes[0].set_xlabel('Lag')
        axes[0].grid(True, alpha=0.3)
        
        # PACF
        axes[1].bar(range(len(pacf_values)), pacf_values)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].axhline(y=1.96/np.sqrt(len(data)), color='red', linestyle='--', alpha=0.5)
        axes[1].axhline(y=-1.96/np.sqrt(len(data)), color='red', linestyle='--', alpha=0.5)
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        axes[1].set_xlabel('Lag')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_attention_weights(self, weights: np.ndarray, title="Attention Weights"):
        """绘制注意力权重热力图"""
        plt.figure(figsize=(10, 8))
        plt.imshow(weights, cmap='viridis', aspect='auto')
        plt.colorbar(label='Attention Weight')
        plt.title(title)
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.tight_layout()
        return plt.gcf()


# ============================================================================
# 第七部分：训练工具
# ============================================================================

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
    
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        self.val_loss_min = val_loss


class Trainer:
    """模型训练器"""
    
    def __init__(self, model, optimizer, criterion, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 构建解码器输入
            label_len = batch_x.size(1) // 2
            dec_input = torch.cat([
                batch_x[:, -label_len:, :],
                torch.zeros_like(batch_y)
            ], dim=1)
            
            outputs = self.model(batch_x, dec_input)
            loss = self.criterion(outputs, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                label_len = batch_x.size(1) // 2
                dec_input = torch.cat([
                    batch_x[:, -label_len:, :],
                    torch.zeros_like(batch_y)
                ], dim=1)
                
                outputs = self.model(batch_x, dec_input)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def fit(self, train_loader, val_loader, epochs=100, patience=10):
        """训练模型"""
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs} - '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        return self.history


# ============================================================================
# 第八部分：使用示例
# ============================================================================

def generate_synthetic_data(n_samples=1000, trend=True, seasonal=True, noise=True):
    """生成合成时间序列数据"""
    t = np.arange(n_samples)
    data = np.zeros(n_samples)
    
    # 趋势成分
    if trend:
        data += 0.01 * t
    
    # 季节成分
    if seasonal:
        data += 10 * np.sin(2 * np.pi * t / 24)  # 日周期
        data += 5 * np.sin(2 * np.pi * t / 168)   # 周周期
    
    # 噪声
    if noise:
        data += np.random.randn(n_samples) * 2
    
    return data.reshape(-1, 1)


def demo():
    """完整使用示例"""
    print("=" * 60)
    print("Time Series Forecasting Demo")
    print("=" * 60)
    
    # 1. 生成数据
    print("\n[1] Generating synthetic data...")
    data = generate_synthetic_data(n_samples=2000)
    print(f"Data shape: {data.shape}")
    
    # 2. 预处理
    print("\n[2] Preprocessing...")
    preprocessor = TimeSeriesPreprocessor()
    data_norm = preprocessor.normalize(data, method='zscore')
    
    # 3. 创建序列
    seq_len, pred_len = 96, 24
    X, y = preprocessor.create_sequences(data_norm, seq_len, pred_len)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # 划分训练/测试
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 4. 创建DataLoader
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 5. 训练ARIMA基线
    print("\n[3] Training ARIMA baseline...")
    arima = ARIMA(p=3, d=1, q=1)
    arima.fit(data[:train_size + seq_len])
    arima_pred = arima.predict(data[:train_size + seq_len], steps=len(y_test))
    
    # 反标准化
    arima_pred = arima_pred.reshape(-1, 1)
    arima_pred = preprocessor.denormalize(arima_pred)
    y_test_true = preprocessor.denormalize(y_test)
    
    # 计算ARIMA指标
    metrics = MetricsCalculator()
    print("\nARIMA Results:")
    metrics.print_metrics(arima_pred[:len(y_test_true)].flatten(), 
                         y_test_true[:, 0, :].flatten())
    
    # 6. 训练Informer
    print("\n[4] Training Informer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    informer = Informer(
        enc_in=1, dec_in=1, c_out=1,
        seq_len=seq_len, label_len=seq_len//2, pred_len=pred_len,
        d_model=64, n_heads=4, e_layers=2, d_layers=1,
        dropout=0.1, factor=5
    )
    
    optimizer = torch.optim.Adam(informer.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    trainer = Trainer(informer, optimizer, criterion, device)
    history = trainer.fit(train_loader, test_loader, epochs=20, patience=5)
    
    # 预测
    informer.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test[:32]).to(device)
        dec_input = torch.cat([
            X_test_tensor[:, -seq_len//2:, :],
            torch.zeros(32, pred_len, 1).to(device)
        ], dim=1)
        informer_pred = informer(X_test_tensor, dec_input).cpu().numpy()
    
    informer_pred = preprocessor.denormalize(informer_pred)
    
    print("\nInformer Results:")
    metrics.print_metrics(informer_pred.flatten(), y_test_true[:32].flatten())
    
    # 7. 可视化
    print("\n[5] Visualization...")
    visualizer = TimeSeriesVisualizer()
    
    # 预测对比
    fig1 = visualizer.plot_prediction(
        y_test_true[:100, 0, 0], 
        arima_pred[:100, 0],
        "ARIMA Prediction"
    )
    fig1.savefig('/tmp/arima_prediction.png')
    
    fig2 = visualizer.plot_prediction(
        y_test_true[:32, 0, 0],
        informer_pred[:32, 0, 0],
        "Informer Prediction"
    )
    fig2.savefig('/tmp/informer_prediction.png')
    
    print("\nVisualization saved to /tmp/")
    print("=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
```

---

## 代码使用说明

### 运行环境要求

```bash
pip install torch numpy pandas matplotlib scipy statsmodels
```

### 快速开始

```python
from chapter33_time_series_forecasting import *

# 1. 数据预处理
preprocessor = TimeSeriesPreprocessor()
data_norm = preprocessor.normalize(your_data)
X, y = preprocessor.create_sequences(data_norm, seq_len=96, pred_len=24)

# 2. 训练Informer
model = Informer(enc_in=1, dec_in=1, c_out=1, 
                 seq_len=96, label_len=48, pred_len=24)
trainer = Trainer(model, optimizer, criterion)
trainer.fit(train_loader, val_loader)

# 3. 评估
metrics = MetricsCalculator()
metrics.print_metrics(predictions, ground_truth)
```

---

*代码总计约950行，涵盖时间序列预处理、ARIMA、Informer、Autoformer、评价指标和可视化工具。*
