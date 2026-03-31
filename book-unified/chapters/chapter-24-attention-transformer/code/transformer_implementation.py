"""
第二十四章：Transformer完整实现
================================

本文件包含Transformer的完整实现，包括：
1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. Positional Encoding
4. Transformer Block
5. Transformer Encoder/Decoder
6. 机器翻译Demo
7. 文本生成器
8. 注意力可视化

作者: AI Assistant
日期: 2026-03-25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 第一部分: 基础组件
# =============================================================================

class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力机制
    
    公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    参数:
        dropout: dropout概率
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        参数:
            Q: Query矩阵, shape (batch, num_heads, seq_len, d_k)
            K: Key矩阵, shape (batch, num_heads, seq_len, d_k)
            V: Value矩阵, shape (batch, num_heads, seq_len, d_v)
            mask: 可选的mask矩阵, shape (batch, 1, seq_len, seq_len)
            return_attention: 是否返回注意力权重
            
        返回:
            output: 注意力输出, shape (batch, num_heads, seq_len, d_v)
            attention_weights: 注意力权重, shape (batch, num_heads, seq_len, seq_len)
        """
        d_k = Q.size(-1)
        
        # 1. 计算点积: Q @ K^T
        # (batch, num_heads, seq_len, d_k) @ (batch, num_heads, d_k, seq_len)
        # = (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # 2. 缩放: 除以 sqrt(d_k)
        # 防止点积过大导致softmax梯度消失
        scores = scores / math.sqrt(d_k)
        
        # 3. 应用mask
        if mask is not None:
            # mask为0的位置填充一个很大的负数，使softmax后接近0
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 4. Softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        
        # 5. Dropout
        attention_weights = self.dropout(attention_weights)
        
        # 6. 加权求和: Attention @ V
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, d_v)
        # = (batch, num_heads, seq_len, d_v)
        output = torch.matmul(attention_weights, V)
        
        if return_attention:
            return output, attention_weights
        return output, None


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    将输入投影到多个子空间，分别计算注意力，然后拼接
    
    参数:
        d_model: 模型维度
        num_heads: 注意力头数
        dropout: dropout概率
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Xavier初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入分割成多头
        
        输入: (batch, seq_len, d_model)
        输出: (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        合并多头
        
        输入: (batch, num_heads, seq_len, d_k)
        输出: (batch, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.size()
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k)
        x = x.transpose(1, 2)
        # (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        参数:
            query: shape (batch, seq_len, d_model)
            key: shape (batch, seq_len, d_model)
            value: shape (batch, seq_len, d_model)
            mask: shape (batch, 1, seq_len, seq_len)
            
        返回:
            output: shape (batch, seq_len, d_model)
            attention_weights: shape (batch, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)
        
        # 1. 线性投影
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. 分割成多头
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 3. 计算缩放点积注意力
        attn_output, attn_weights = self.attention(
            Q, K, V, mask, return_attention
        )
        
        # 4. 合并多头
        output = self.combine_heads(attn_output)  # (batch, seq_len, d_model)
        
        # 5. 最终线性变换
        output = self.W_o(output)
        
        return output, attn_weights


class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    
    使用正弦和余弦函数编码位置信息:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    参数:
        d_model: 模型维度
        max_seq_length: 最大序列长度
        dropout: dropout概率
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
        # 创建位置编码矩阵: (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)
        
        # 位置索引: (max_seq_length, 1)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # div_term: (d_model // 2,)
        # 10000^(2i/d_model) = exp(2i * -log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # 正弦应用于偶数维度
        pe[:, 0::2] = torch.sin(position * div_term)
        # 余弦应用于奇数维度
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加batch维度: (1, max_seq_length, d_model)
        pe = pe.unsqueeze(0)
        
        # 注册为buffer（不作为模型参数，但会随模型保存）
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        参数:
            x: shape (batch, seq_len, d_model)
            
        返回:
            x + position_encoding
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return self.dropout(x)


class FeedForwardNetwork(nn.Module):
    """
    前馈神经网络 (Position-wise Feed-Forward Network)
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    或: FFN(x) = GELU(xW1 + b1)W2 + b2
    
    参数:
        d_model: 模型维度
        d_ff: 前馈网络中间层维度 (通常为4*d_model)
        dropout: dropout概率
        activation: 激活函数 ('relu' 或 'gelu')
    """
    
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: shape (batch, seq_len, d_model)
            
        返回:
            output: shape (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class LayerNorm(nn.Module):
    """
    Layer Normalization
    
    对每个样本的特征维度进行归一化:
    LN(x) = γ * (x - μ) / sqrt(σ^2 + ε) + β
    
    参数:
        d_model: 特征维度
        eps: 防止除零的小数
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: shape (batch, seq_len, d_model)
            
        返回:
            normalized: shape (batch, seq_len, d_model)
        """
        # 计算均值和方差（在最后一个维度上）
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # 归一化
        x_normalized = (x - mean) / (std + self.eps)
        
        # 缩放和平移
        return self.gamma * x_normalized + self.beta


class TransformerBlock(nn.Module):
    """
    Transformer编码器块
    
    包含:
    1. 多头自注意力
    2. Add & Norm（残差连接 + LayerNorm）
    3. 前馈网络
    4. Add & Norm
    
    参数:
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈网络维度
        dropout: dropout概率
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 前馈网络
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        参数:
            x: shape (batch, seq_len, d_model)
            mask: 注意力mask
            
        返回:
            output: shape (batch, seq_len, d_model)
            attention_weights: 注意力权重（如果return_attention=True）
        """
        # 1. 多头自注意力 + 残差连接
        attn_output, attn_weights = self.self_attn(
            x, x, x, mask, return_attention
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights


# =============================================================================
# 第二部分: Transformer编码器和解码器
# =============================================================================

class TransformerEncoder(nn.Module):
    """
    Transformer编码器
    
    由多个TransformerBlock堆叠而成
    
    参数:
        d_model: 模型维度
        num_heads: 注意力头数
        d_ff: 前馈网络维度
        num_layers: 编码器层数
        dropout: dropout概率
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        前向传播
        
        参数:
            x: shape (batch, seq_len, d_model)
            mask: 注意力mask
            return_attention: 是否返回所有层的注意力权重
            
        返回:
            output: shape (batch, seq_len, d_model)
            attentions: 各层的注意力权重列表（如果return_attention=True）
        """
        attentions = []
        
        for layer in self.layers:
            x, attn = layer(x, mask, return_attention)
            if return_attention and attn is not None:
                attentions.append(attn)
        
        if return_attention:
            return x, attentions
        return x, None


class TransformerDecoderBlock(nn.Module):
    """
    Transformer解码器块
    
    包含:
    1. Masked多头自注意力（防止看到未来信息）
    2. Add & Norm
    3. 多头交叉注意力（关注编码器输出）
    4. Add & Norm
    5. 前馈网络
    6. Add & Norm
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Masked多头自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 多头交叉注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 前馈网络
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        前向传播
        
        参数:
            x: 解码器输入, shape (batch, tgt_seq_len, d_model)
            encoder_output: 编码器输出, shape (batch, src_seq_len, d_model)
            tgt_mask: 目标序列mask（用于masked self-attention）
            src_mask: 源序列mask（用于cross-attention）
            
        返回:
            output: shape (batch, tgt_seq_len, d_model)
            attentions: 注意力权重字典（如果return_attention=True）
        """
        attentions = {}
        
        # 1. Masked多头自注意力 + 残差
        self_attn_output, self_attn_weights = self.self_attn(
            x, x, x, tgt_mask, return_attention
        )
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 2. 多头交叉注意力 + 残差
        # Query来自解码器，Key和Value来自编码器
        cross_attn_output, cross_attn_weights = self.cross_attn(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=src_mask,
            return_attention=return_attention
        )
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 3. 前馈网络 + 残差
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        if return_attention:
            attentions['self_attn'] = self_attn_weights
            attentions['cross_attn'] = cross_attn_weights
            return x, attentions
        
        return x, None


class TransformerDecoder(nn.Module):
    """
    Transformer解码器
    
    由多个TransformerDecoderBlock堆叠而成
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        前向传播
        """
        all_attentions = []
        
        for layer in self.layers:
            x, attn = layer(x, encoder_output, tgt_mask, src_mask, return_attention)
            if return_attention:
                all_attentions.append(attn)
        
        if return_attention:
            return x, all_attentions
        return x, None


# =============================================================================
# 第三部分: 完整Transformer模型
# =============================================================================

class Transformer(nn.Module):
    """
    完整的Transformer模型
    
    用于序列到序列的任务，如机器翻译
    
    参数:
        src_vocab_size: 源语言词汇表大小
        tgt_vocab_size: 目标语言词汇表大小
        d_model: 模型维度
        num_heads: 注意力头数
        num_layers: 编码器/解码器层数
        d_ff: 前馈网络维度
        max_seq_length: 最大序列长度
        dropout: dropout概率
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # 词嵌入
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # 编码器和解码器
        self.encoder = TransformerEncoder(
            d_model, num_heads, d_ff, num_layers, dropout
        )
        self.decoder = TransformerDecoder(
            d_model, num_heads, d_ff, num_layers, dropout
        )
        
        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self._init_parameters()
    
    def _init_parameters(self):
        """Xavier初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def make_src_mask(self, src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        创建源序列的padding mask
        
        参数:
            src: shape (batch, src_seq_len)
            pad_idx: padding token的索引
            
        返回:
            mask: shape (batch, 1, 1, src_seq_len)
        """
        # src: (batch, src_seq_len)
        # 将pad_idx的位置设为0，其他设为1
        mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        # mask: (batch, 1, 1, src_seq_len)
        return mask
    
    def make_tgt_mask(self, tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """
        创建目标序列的mask（padding mask + causal mask）
        
        参数:
            tgt: shape (batch, tgt_seq_len)
            pad_idx: padding token的索引
            
        返回:
            mask: shape (batch, 1, tgt_seq_len, tgt_seq_len)
        """
        tgt_seq_len = tgt.size(1)
        
        # Padding mask
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        # pad_mask: (batch, 1, 1, tgt_seq_len)
        
        # Causal mask (上三角矩阵)
        causal_mask = torch.tril(
            torch.ones(tgt_seq_len, tgt_seq_len, device=tgt.device)
        ).bool().unsqueeze(0).unsqueeze(0)
        # causal_mask: (1, 1, tgt_seq_len, tgt_seq_len)
        
        # 合并两个mask
        mask = pad_mask & causal_mask
        # mask: (batch, 1, tgt_seq_len, tgt_seq_len)
        
        return mask
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        """编码源序列"""
        # 词嵌入 + 缩放
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        # 添加位置编码
        x = self.positional_encoding(x)
        # 编码器
        memory, _ = self.encoder(x, src_mask)
        return memory
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ):
        """解码目标序列"""
        # 词嵌入 + 缩放
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        # 添加位置编码
        x = self.positional_encoding(x)
        # 解码器
        output, _ = self.decoder(x, memory, tgt_mask, src_mask)
        # 输出层
        output = self.output_layer(output)
        return output
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数:
            src: 源序列, shape (batch, src_seq_len)
            tgt: 目标序列, shape (batch, tgt_seq_len)
            src_mask: 源序列mask
            tgt_mask: 目标序列mask
            
        返回:
            output: 预测结果, shape (batch, tgt_seq_len, tgt_vocab_size)
        """
        # 编码
        memory = self.encode(src, src_mask)
        
        # 解码
        output = self.decode(tgt, memory, tgt_mask, src_mask)
        
        return output


# =============================================================================
# 第四部分: 应用示例 - 机器翻译
# =============================================================================

class SimpleTranslationDataset:
    """简化的翻译数据集"""
    
    def __init__(self, pairs):
        """
        参数:
            pairs: [(源句子, 目标句子), ...]
        """
        self.pairs = pairs
        self.src_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.tgt_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self._build_vocab()
    
    def _build_vocab(self):
        """构建词汇表"""
        for src, tgt in self.pairs:
            for word in src.split():
                if word not in self.src_vocab:
                    self.src_vocab[word] = len(self.src_vocab)
            for word in tgt.split():
                if word not in self.tgt_vocab:
                    self.tgt_vocab[word] = len(self.tgt_vocab)
    
    def encode_src(self, sentence):
        """编码源句子"""
        tokens = [self.src_vocab.get(w, self.src_vocab['<unk>']) 
                  for w in sentence.split()]
        return [self.src_vocab['<sos>']] + tokens + [self.src_vocab['<eos>']]
    
    def encode_tgt(self, sentence):
        """编码目标句子"""
        tokens = [self.tgt_vocab.get(w, self.tgt_vocab['<unk>']) 
                  for w in sentence.split()]
        return [self.tgt_vocab['<sos>']] + tokens + [self.tgt_vocab['<eos>']]
    
    def decode_tgt(self, indices):
        """解码目标句子"""
        id_to_word = {v: k for k, v in self.tgt_vocab.items()}
        return ' '.join([id_to_word.get(i, '<unk>') for i in indices])


class Translator:
    """基于Transformer的翻译器"""
    
    def __init__(self, model, dataset, device='cpu'):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
    
    def translate(self, src_sentence, max_len=50, beam_size=1):
        """
        翻译句子
        
        参数:
            src_sentence: 源语言句子
            max_len: 最大生成长度
            beam_size: 束搜索宽度（1表示贪心解码）
            
        返回:
            翻译结果
        """
        self.model.eval()
        
        # 编码源句子
        src_indices = self.dataset.encode_src(src_sentence)
        src_tensor = torch.tensor([src_indices]).to(self.device)
        src_mask = self.model.make_src_mask(src_tensor)
        
        # 编码
        with torch.no_grad():
            memory = self.model.encode(src_tensor, src_mask)
        
        if beam_size == 1:
            # 贪心解码
            return self._greedy_decode(memory, src_mask, max_len)
        else:
            # 束搜索
            return self._beam_search(memory, src_mask, max_len, beam_size)
    
    def _greedy_decode(self, memory, src_mask, max_len):
        """贪心解码"""
        tgt_indices = [self.dataset.tgt_vocab['<sos>']]
        
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_indices]).to(self.device)
            tgt_mask = self.model.make_tgt_mask(tgt_tensor)
            
            with torch.no_grad():
                output = self.model.decode(tgt_tensor, memory, tgt_mask, src_mask)
                pred = output[0, -1].argmax().item()
            
            tgt_indices.append(pred)
            
            if pred == self.dataset.tgt_vocab['<eos>']:
                break
        
        return self.dataset.decode_tgt(tgt_indices)
    
    def _beam_search(self, memory, src_mask, max_len, beam_size):
        """束搜索解码"""
        sos_idx = self.dataset.tgt_vocab['<sos>']
        eos_idx = self.dataset.tgt_vocab['<eos>']
        
        sequences = [[sos_idx]]
        scores = [0.0]
        
        for _ in range(max_len):
            all_candidates = []
            
            for seq, score in zip(sequences, scores):
                if seq[-1] == eos_idx:
                    all_candidates.append((seq, score))
                    continue
                
                tgt_tensor = torch.tensor([seq]).to(self.device)
                tgt_mask = self.model.make_tgt_mask(tgt_tensor)
                
                with torch.no_grad():
                    output = self.model.decode(tgt_tensor, memory, tgt_mask, src_mask)
                    logits = output[0, -1]
                    log_probs = F.log_softmax(logits, dim=-1)
                
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
                
                for log_prob, idx in zip(topk_log_probs, topk_indices):
                    new_seq = seq + [idx.item()]
                    new_score = score + log_prob.item()
                    all_candidates.append((new_seq, new_score))
            
            # 选择得分最高的beam_size个
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = [seq for seq, _ in ordered[:beam_size]]
            scores = [score for _, score in ordered[:beam_size]]
            
            # 如果所有序列都以eos结尾，停止
            if all(seq[-1] == eos_idx for seq in sequences):
                break
        
        # 返回得分最高的序列
        best_seq = sequences[0]
        return self.dataset.decode_tgt(best_seq)


# =============================================================================
# 第五部分: 文本生成器
# =============================================================================

class TextGenerator:
    """
    基于Transformer的文本生成器
    
    支持多种采样策略：贪心、温度采样、top-k、top-p
    """
    
    def __init__(self, model, vocab, device='cpu'):
        """
        参数:
            model: Transformer模型
            vocab: 词汇表字典
            device: 计算设备
        """
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.id_to_word = {v: k for k, v in vocab.items()}
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        strategy: str = 'temperature'
    ) -> str:
        """
        生成文本
        
        参数:
            prompt: 起始文本
            max_length: 最大生成长度
            temperature: 采样温度（越高越随机）
            top_k: 只从top k个候选中采样
            top_p: nucleus sampling阈值
            strategy: 采样策略 ('greedy', 'temperature', 'top_k', 'top_p')
            
        返回:
            生成的文本
        """
        self.model.eval()
        
        # 编码prompt
        tokens = [self.vocab.get(w, self.vocab.get('<unk>', 3)) 
                  for w in prompt.split()]
        
        with torch.no_grad():
            for _ in range(max_length):
                input_ids = torch.tensor([tokens]).to(self.device)
                
                # 创建causal mask
                tgt_mask = self.model.make_tgt_mask(input_ids)
                
                # 前向传播
                output = self.model.decode(input_ids, input_ids, tgt_mask)
                
                # 获取最后一个位置的logits
                logits = output[0, -1, :]
                
                # 应用采样策略
                if strategy == 'greedy':
                    next_token = logits.argmax().item()
                elif strategy == 'temperature':
                    next_token = self._temperature_sampling(logits, temperature)
                elif strategy == 'top_k':
                    next_token = self._top_k_sampling(logits, top_k, temperature)
                elif strategy == 'top_p':
                    next_token = self._top_p_sampling(logits, top_p, temperature)
                else:
                    raise ValueError(f"未知的采样策略: {strategy}")
                
                tokens.append(next_token)
                
                # 如果生成了结束符
                if next_token == self.vocab.get('<eos>', 2):
                    break
        
        # 转换为文本
        words = [self.id_to_word.get(i, '<unk>') for i in tokens]
        return ' '.join(words)
    
    def _temperature_sampling(self, logits: torch.Tensor, temperature: float) -> int:
        """温度采样"""
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
    
    def _top_k_sampling(
        self,
        logits: torch.Tensor,
        k: int,
        temperature: float
    ) -> int:
        """Top-k采样"""
        # 获取top k
        top_k_logits, top_k_indices = torch.topk(logits, k)
        
        # 应用温度
        top_k_logits = top_k_logits / temperature
        
        # Softmax
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        # 采样
        sampled_idx = torch.multinomial(top_k_probs, num_samples=1)
        return top_k_indices[sampled_idx].item()
    
    def _top_p_sampling(
        self,
        logits: torch.Tensor,
        p: float,
        temperature: float
    ) -> int:
        """Top-p (nucleus)采样"""
        # 排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        
        # 累积概率
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到累积概率超过p的位置
        sorted_indices_to_remove = cumsum_probs > p
        sorted_indices_to_remove[0] = False  # 至少保留第一个
        
        # 移除低概率词
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('inf')
        
        # 应用温度并采样
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()


# =============================================================================
# 第六部分: 注意力可视化
# =============================================================================

class AttentionVisualizer:
    """注意力权重可视化工具"""
    
    @staticmethod
    def plot_attention_heatmap(
        attention_weights: np.ndarray,
        src_tokens: list,
        tgt_tokens: list,
        title: str = "Attention Weights",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ):
        """
        绘制注意力热力图
        
        参数:
            attention_weights: shape (tgt_len, src_len)
            src_tokens: 源端token列表
            tgt_tokens: 目标端token列表
            title: 图标题
            figsize: 图像大小
            save_path: 保存路径（可选）
        """
        plt.figure(figsize=figsize)
        
        sns.heatmap(
            attention_weights,
            xticklabels=src_tokens,
            yticklabels=tgt_tokens,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Attention Weight'},
            square=True
        )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Source Tokens', fontsize=12)
        plt.ylabel('Target Tokens', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_multi_head_attention(
        attention_weights: np.ndarray,
        tokens: list,
        num_heads: int,
        num_cols: int = 4,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None
    ):
        """
        绘制多头注意力的热力图
        
        参数:
            attention_weights: shape (num_heads, seq_len, seq_len)
            tokens: token列表
            num_heads: 注意力头数
            num_cols: 每行显示的图数
            figsize: 图像大小
            save_path: 保存路径（可选）
        """
        num_rows = (num_heads + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()
        
        for i in range(num_heads):
            ax = axes[i]
            sns.heatmap(
                attention_weights[i],
                xticklabels=tokens if len(tokens) < 15 else [],
                yticklabels=tokens if len(tokens) < 15 else [],
                cmap='YlOrRd',
                cbar=True,
                ax=ax,
                square=True
            )
            ax.set_title(f'Head {i+1}', fontsize=10)
        
        # 隐藏多余的子图
        for i in range(num_heads, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Multi-Head Attention Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_attention_flow(
        attention_weights: np.ndarray,
        tokens: list,
        threshold: float = 0.1,
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None
    ):
        """
        绘制注意力流向图
        
        参数:
            attention_weights: shape (seq_len, seq_len)
            tokens: token列表
            threshold: 显示连接的阈值
            figsize: 图像大小
            save_path: 保存路径（可选）
        """
        seq_len = len(tokens)
        x_pos = np.arange(seq_len)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制token位置
        ax.scatter(x_pos, np.zeros(seq_len), s=300, c='steelblue', zorder=3, edgecolors='white', linewidths=2)
        
        # 绘制token标签
        for i, token in enumerate(tokens):
            ax.annotate(
                token,
                (x_pos[i], 0),
                textcoords="offset points",
                xytext=(0, -25),
                ha='center',
                fontsize=11,
                fontweight='bold'
            )
        
        # 绘制注意力连接
        max_weight = attention_weights.max()
        
        for i in range(seq_len):
            for j in range(seq_len):
                weight = attention_weights[i, j]
                if weight > threshold and i != j:
                    alpha = weight / max_weight
                    # 使用曲线连接
                    connectionstyle = f"arc3,rad={0.2 * (j - i) / seq_len}"
                    ax.annotate(
                        '',
                        xy=(x_pos[j], 0),
                        xytext=(x_pos[i], 0.05 + 0.1 * alpha),
                        arrowprops=dict(
                            arrowstyle='->',
                            color='crimson',
                            alpha=alpha * 0.8,
                            lw=alpha * 3,
                            connectionstyle=connectionstyle
                        )
                    )
        
        ax.set_ylim(-0.5, 0.3)
        ax.set_xlim(-0.5, seq_len - 0.5)
        ax.axis('off')
        ax.set_title('Attention Flow Visualization', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# =============================================================================
# 第七部分: 工具函数
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_transformer_config(
    src_vocab_size: int = 10000,
    tgt_vocab_size: int = 10000,
    model_size: str = 'base'
):
    """
    创建Transformer配置
    
    参数:
        model_size: 'tiny', 'small', 'base', 'large'
    """
    configs = {
        'tiny': {
            'd_model': 128,
            'num_heads': 4,
            'num_layers': 2,
            'd_ff': 512,
            'dropout': 0.1
        },
        'small': {
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 4,
            'd_ff': 1024,
            'dropout': 0.1
        },
        'base': {
            'd_model': 512,
            'num_heads': 8,
            'num_layers': 6,
            'd_ff': 2048,
            'dropout': 0.1
        },
        'large': {
            'd_model': 1024,
            'num_heads': 16,
            'num_layers': 12,
            'd_ff': 4096,
            'dropout': 0.1
        }
    }
    
    config = configs[model_size].copy()
    config['src_vocab_size'] = src_vocab_size
    config['tgt_vocab_size'] = tgt_vocab_size
    
    return config


def demo():
    """
    演示Transformer的各个组件
    """
    print("=" * 60)
    print("Transformer组件演示")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\n1. 输入维度: {x.shape}")
    
    # 1. Scaled Dot-Product Attention
    print("\n2. Scaled Dot-Product Attention")
    attn = ScaledDotProductAttention(dropout=0.1)
    Q = torch.randn(batch_size, num_heads, seq_len, d_model // num_heads)
    K = torch.randn(batch_size, num_heads, seq_len, d_model // num_heads)
    V = torch.randn(batch_size, num_heads, seq_len, d_model // num_heads)
    output, weights = attn(Q, K, V, return_attention=True)
    print(f"   输出维度: {output.shape}")
    print(f"   注意力权重维度: {weights.shape}")
    
    # 2. Multi-Head Attention
    print("\n3. Multi-Head Attention")
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    output, _ = mha(x, x, x)
    print(f"   输出维度: {output.shape}")
    print(f"   参数量: {count_parameters(mha):,}")
    
    # 3. Positional Encoding
    print("\n4. Positional Encoding")
    pe = PositionalEncoding(d_model=d_model, max_seq_length=100)
    x_pe = pe(x)
    print(f"   输出维度: {x_pe.shape}")
    print(f"   位置编码维度: {pe.pe.shape}")
    
    # 4. Transformer Block
    print("\n5. Transformer Block")
    block = TransformerBlock(d_model=d_model, num_heads=num_heads)
    output, _ = block(x)
    print(f"   输出维度: {output.shape}")
    print(f"   参数量: {count_parameters(block):,}")
    
    # 5. 完整Transformer
    print("\n6. 完整Transformer")
    config = create_transformer_config(10000, 10000, 'tiny')
    model = Transformer(**config)
    print(f"   配置: {config}")
    print(f"   总参数量: {count_parameters(model):,}")
    
    # 测试前向传播
    src = torch.randint(0, 10000, (2, 20))
    tgt = torch.randint(0, 10000, (2, 15))
    output = model(src, tgt)
    print(f"   输入: src={src.shape}, tgt={tgt.shape}")
    print(f"   输出: {output.shape}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
