"""
第二十九章：生成模型与扩散模型 - 完整代码实现
================================================

本文件包含：
1. 简化版DDPM的完整实现
2. U-Net噪声预测网络
3. 扩散采样算法
4. 噪声调度策略
5. 训练循环和可视化工具

作者: ML教材编写组
版本: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import math
from tqdm import tqdm


# =============================================================================
# 第一部分：时间步嵌入 (Time Embedding)
# =============================================================================

class TimeEmbedding(nn.Module):
    """
    正弦/余弦时间步嵌入。
    将整数时间步转换为高维向量，类似于Transformer中的位置编码。
    
    原理：不同频率的正弦/余弦函数可以让模型学习时间相关的模式
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        输入: t - 时间步张量，形状 (batch_size,)
        输出: 时间嵌入，形状 (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        # 计算不同频率的嵌入
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        # 应用正弦和余弦
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


# =============================================================================
# 第二部分：基础神经网络模块
# =============================================================================

class Swish(nn.Module):
    """Swish激活函数: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    """
    残差块，用于U-Net。
    包含两个卷积层，带有组归一化和时间嵌入注入。
    """
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)
        
        # 时间嵌入投影
        self.time_mlp = nn.Linear(time_channels, out_channels)
        
        # 残差连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
        self.dropout = nn.Dropout(dropout)
        self.act = Swish()
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: 输入特征图，形状 (B, C, H, W)
            t_emb: 时间嵌入，形状 (B, time_channels)
        输出: 输出特征图，形状 (B, out_channels, H, W)
        """
        h = self.conv1(x)
        h = self.gn1(h)
        h = self.act(h)
        
        # 注入时间信息
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        
        h = self.conv2(h)
        h = self.gn2(h)
        h = self.act(h)
        h = self.dropout(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    自注意力块，用于捕捉长距离依赖关系。
    在低分辨率特征图上使用。
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.gn = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.gn(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 将空间维度展平为序列
        q = q.reshape(B, C, H * W).transpose(1, 2)  # (B, HW, C)
        k = k.reshape(B, C, H * W)  # (B, C, HW)
        v = v.reshape(B, C, H * W).transpose(1, 2)  # (B, HW, C)
        
        # 计算注意力
        attn = torch.bmm(q, k) * (C ** -0.5)  # (B, HW, HW)
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力到值
        h = torch.bmm(attn, v)  # (B, HW, C)
        h = h.transpose(1, 2).reshape(B, C, H, W)
        
        return x + self.proj(h)


class DownBlock(nn.Module):
    """下采样块：卷积 + 下采样"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    """上采样块：最近邻上采样 + 卷积"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# =============================================================================
# 第三部分：U-Net噪声预测网络
# =============================================================================

class UNet(nn.Module):
    """
    U-Net架构用于噪声预测。
    这是DDPM的核心组件，负责根据当前时刻的加噪图像预测添加的噪声。
    
    架构特点：
    - 编码器-解码器结构，带有跳跃连接
    - 时间步嵌入注入到每一层
    - 在低分辨率层使用自注意力
    """
    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 3,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (2, 4),
        dropout: float = 0.1,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_classes: Optional[int] = None,  # 用于条件生成
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        
        # 时间嵌入维度
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            Swish(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # 条件嵌入(类别)
        self.num_classes = num_classes
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        
        # 输入投影
        self.input_proj = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # 编码器
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()
        
        channels = [model_channels]
        ch = model_channels
        
        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(ResidualBlock(ch, out_ch, time_embed_dim, dropout))
                ch = out_ch
                channels.append(ch)
                # 在指定分辨率添加注意力
                if i in attention_resolutions:
                    self.encoder_blocks.append(AttentionBlock(ch))
                    channels.append(ch)
            
            # 下采样(最后一层除外)
            if i != len(channel_mult) - 1:
                self.encoder_downs.append(DownBlock(ch, ch))
                channels.append(ch)
        
        # 中间层
        self.middle_block1 = ResidualBlock(ch, ch, time_embed_dim, dropout)
        self.middle_attn = AttentionBlock(ch)
        self.middle_block2 = ResidualBlock(ch, ch, time_embed_dim, dropout)
        
        # 解码器
        self.decoder_blocks = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            for j in range(num_res_blocks + 1):
                # 跳跃连接会拼接特征，所以输入通道数要翻倍
                self.decoder_blocks.append(ResidualBlock(ch + channels.pop(), out_ch, time_embed_dim, dropout))
                ch = out_ch
                # 在指定分辨率添加注意力
                if len(channel_mult) - 1 - i in attention_resolutions:
                    self.decoder_blocks.append(AttentionBlock(ch))
            
            # 上采样(第一层除外)
            if i != len(channel_mult) - 1:
                self.decoder_ups.append(UpBlock(ch, ch))
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, ch),
            Swish(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: torch.Tensor, 
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播。
        
        输入:
            x: 加噪图像，形状 (B, C, H, W)
            timesteps: 时间步，形状 (B,)
            y: 类别标签(可选)，形状 (B,)
        输出:
            预测的噪声，形状 (B, C, H, W)
        """
        # 时间嵌入
        t_emb = self.time_embed(timesteps)
        
        # 条件嵌入
        if self.num_classes is not None and y is not None:
            t_emb = t_emb + self.label_emb(y)
        
        # 输入投影
        h = self.input_proj(x)
        
        # 编码器
        hs = [h]
        for module in self.encoder_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
            hs.append(h)
        
        for module in self.encoder_downs:
            h = module(h)
            hs.append(h)
        
        # 中间层
        h = self.middle_block1(h, t_emb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, t_emb)
        
        # 解码器
        for module in self.decoder_blocks:
            if isinstance(module, ResidualBlock):
                # 跳跃连接
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, t_emb)
            else:
                h = module(h)
        
        for module in self.decoder_ups:
            h = module(h)
        
        return self.output_proj(h)


# =============================================================================
# 第四部分：DDPM核心类
# =============================================================================

class DDPM:
    """
    去噪扩散概率模型(Denoising Diffusion Probabilistic Model)。
    
    参数:
        model: 噪声预测网络
        timesteps: 扩散时间步数
        beta_schedule: beta调度策略 ('linear', 'cosine', 'quadratic')
        beta_start: beta起始值
        beta_end: beta结束值
    """
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = 'linear',
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device
        
        # 设置beta调度
        self.betas = self._get_beta_schedule(beta_schedule, beta_start, beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 预计算扩散过程的相关值
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 后验方差
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # 将参数移到设备
        self._to_device()
        
    def _get_beta_schedule(
        self, 
        schedule: str, 
        beta_start: float, 
        beta_end: float
    ) -> torch.Tensor:
        """获取beta调度"""
        if schedule == 'linear':
            return torch.linspace(beta_start, beta_end, self.timesteps)
        elif schedule == 'quadratic':
            return torch.linspace(beta_start**0.5, beta_end**0.5, self.timesteps) ** 2
        elif schedule == 'cosine':
            return self._cosine_beta_schedule()
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """余弦调度，来自Improved DDPM"""
        s = 0.008
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _to_device(self):
        """将所有缓冲区移到设备"""
        for name, tensor in self.__dict__.items():
            if isinstance(tensor, torch.Tensor):
                setattr(self, name, tensor.to(self.device))
    
    def q_sample(
        self, 
        x0: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向扩散过程：根据时间步t添加噪声。
        
        输入:
            x0: 原始图像，形状 (B, C, H, W)
            t: 时间步，形状 (B,)
            noise: 噪声(可选)，形状 (B, C, H, W)
        输出:
            xt: 加噪图像
            noise: 添加的噪声
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return xt, noise
    
    def predict_start_from_noise(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        从噪声预测x0。
        
        公式: x0 = (x_t - sqrt(1-ᾱ_t) * ε) / sqrt(ᾱ_t)
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
    
    def q_posterior_mean_variance(
        self, 
        x0: torch.Tensor, 
        x_t: torch.Tensor, 
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差。
        """
        posterior_mean = (
            self._extract(self.alphas, t, x0.shape) * x_t +
            self._extract(self.betas, t, x0.shape) * x0
        ) / self._extract(self.alphas_cumprod, t, x0.shape)
        
        posterior_variance = self._extract(self.posterior_variance, t, x0.shape)
        
        return posterior_mean, posterior_variance
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """从张量a中提取对应时间步t的值，并调整形状"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def p_mean_variance(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算反向过程 p(x_{t-1} | x_t) 的均值和方差。
        """
        # 预测噪声
        pred_noise = self.model(x_t, t, y)
        
        # 预测x0
        x0_pred = self.predict_start_from_noise(x_t, t, pred_noise)
        
        if clip_denoised:
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        # 计算后验均值和方差
        model_mean, model_variance = self.q_posterior_mean_variance(x0_pred, x_t, t)
        
        return model_mean, model_variance
    
    def p_sample(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        从 p(x_{t-1} | x_t) 采样。
        """
        model_mean, model_variance = self.p_mean_variance(x_t, t, y)
        
        noise = torch.randn_like(x_t)
        # 当t=0时不添加噪声
        nonzero_mask = (t != 0).float().reshape(-1, 1, 1, 1)
        
        return model_mean + nonzero_mask * torch.sqrt(model_variance) * noise
    
    @torch.no_grad()
    def sample(
        self, 
        batch_size: int, 
        channels: int, 
        height: int, 
        width: int,
        y: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        生成样本。
        
        输入:
            batch_size: 批量大小
            channels: 通道数
            height: 图像高度
            width: 图像宽度
            y: 类别标签(条件生成)
        输出:
            生成的样本，形状 (batch_size, channels, height, width)
        """
        self.model.eval()
        
        # 从纯噪声开始
        x = torch.randn(batch_size, channels, height, width, device=self.device)
        
        # 反向扩散
        timesteps = range(self.timesteps - 1, -1, -1)
        if verbose:
            timesteps = tqdm(timesteps, desc='Sampling')
        
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t_batch, y)
        
        return x
    
    def training_step(
        self, 
        x0: torch.Tensor, 
        optimizer: optim.Optimizer,
        y: Optional[torch.Tensor] = None
    ) -> float:
        """
        单步训练。
        
        输入:
            x0: 原始图像
            optimizer: 优化器
            y: 类别标签(可选)
        返回:
            损失值
        """
        self.model.train()
        optimizer.zero_grad()
        
        batch_size = x0.shape[0]
        device = x0.device
        
        # 随机采样时间步
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        
        # 添加噪声
        noise = torch.randn_like(x0)
        xt, _ = self.q_sample(x0, t, noise)
        
        # 预测噪声
        pred_noise = self.model(xt, t, y)
        
        # 计算MSE损失
        loss = F.mse_loss(pred_noise, noise)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()


# =============================================================================
# 第五部分：DDIM采样器
# =============================================================================

class DDIMSampler:
    """
    DDIM (Denoising Diffusion Implicit Models) 采样器。
    支持确定性采样和子序列采样，显著加速生成过程。
    """
    def __init__(self, ddpm: DDPM, eta: float = 0.0):
        """
        参数:
            ddpm: 训练好的DDPM模型
            eta: 随机性参数(0为确定性，1为标准DDPM)
        """
        self.ddpm = ddpm
        self.eta = eta
        
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        y: Optional[torch.Tensor] = None,
        ddim_timesteps: int = 50,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        DDIM采样。
        
        参数:
            ddim_timesteps: 实际采样的时间步数(子序列)
        """
        self.ddpm.model.eval()
        
        # 选择子序列时间步
        c = self.ddpm.timesteps // ddim_timesteps
        timesteps = np.asarray(list(range(0, self.ddpm.timesteps, c))) + 1
        timesteps = timesteps[:ddim_timesteps]
        timesteps = torch.from_numpy(timesteps).long().to(self.ddpm.device)
        
        # 从纯噪声开始
        x = torch.randn(batch_size, channels, height, width, device=self.ddpm.device)
        
        # 反向采样
        iterator = range(len(timesteps) - 1, -1, -1)
        if verbose:
            iterator = tqdm(list(iterator), desc='DDIM Sampling')
        
        for i in iterator:
            t = torch.full((batch_size,), timesteps[i], device=self.ddpm.device, dtype=torch.long)
            
            # 预测噪声
            pred_noise = self.ddpm.model(x, t, y)
            
            # 预测x0
            alpha_t = self.ddpm.alphas_cumprod[t][:, None, None, None]
            alpha_prev = self.ddpm.alphas_cumprod[timesteps[i - 1]][:, None, None, None] if i > 0 else torch.ones_like(alpha_t)
            
            x0_pred = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
            
            # 计算方向
            sigma_t = self.eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
            
            # DDIM更新规则
            noise = torch.randn_like(x) if i > 0 else 0
            x = (
                torch.sqrt(alpha_prev) * x0_pred +
                torch.sqrt(1 - alpha_prev - sigma_t**2) * pred_noise +
                sigma_t * noise
            )
        
        return x


# =============================================================================
# 第六部分：Classifier-Free Guidance
# =============================================================================

class CFGDDPM(DDPM):
    """
    支持Classifier-Free Guidance的DDPM。
    通过同时训练条件模型和无条件模型，实现更好的条件生成。
    """
    def __init__(self, *args, p_uncond: float = 0.1, **kwargs):
        """
        参数:
            p_uncond: 训练时丢弃条件的概率
        """
        super().__init__(*args, **kwargs)
        self.p_uncond = p_uncond
    
    def training_step(
        self, 
        x0: torch.Tensor, 
        optimizer: optim.Optimizer,
        y: torch.Tensor
    ) -> float:
        """训练步骤，随机丢弃条件"""
        self.model.train()
        optimizer.zero_grad()
        
        batch_size = x0.shape[0]
        device = x0.device
        
        # 随机丢弃条件
        mask = torch.rand(batch_size, device=device) > self.p_uncond
        y_masked = y.clone()
        y_masked[~mask] = -1  # 使用-1表示无条件
        
        # 随机采样时间步
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        
        # 添加噪声
        noise = torch.randn_like(x0)
        xt, _ = self.q_sample(x0, t, noise)
        
        # 预测噪声
        pred_noise = self.model(xt, t, y_masked)
        
        # 计算损失
        loss = F.mse_loss(pred_noise, noise)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample_cfg(
        self,
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        y: torch.Tensor,
        w: float = 1.0,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        使用Classifier-Free Guidance采样。
        
        参数:
            y: 类别标签
            w: 引导强度(w=0为无条件，w>1增强条件)
        """
        self.model.eval()
        
        # 从纯噪声开始
        x = torch.randn(batch_size, channels, height, width, device=self.device)
        
        # 无条件标签
        y_uncond = torch.full_like(y, -1)
        
        timesteps = range(self.timesteps - 1, -1, -1)
        if verbose:
            timesteps = tqdm(timesteps, desc='CFG Sampling')
        
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # 条件预测
            noise_cond = self.model(x, t_batch, y)
            
            # 无条件预测
            noise_uncond = self.model(x, t_batch, y_uncond)
            
            # CFG公式: ε = ε_uncond + w * (ε_cond - ε_uncond)
            pred_noise = noise_uncond + w * (noise_cond - noise_uncond)
            
            # 采样
            x = self.p_sample_with_noise(x, t_batch, pred_noise)
        
        return x
    
    def p_sample_with_noise(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        pred_noise: torch.Tensor
    ) -> torch.Tensor:
        """使用给定的噪声预测进行采样"""
        x0_pred = self.predict_start_from_noise(x_t, t, pred_noise)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        model_mean, model_variance = self.q_posterior_mean_variance(x0_pred, x_t, t)
        
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(-1, 1, 1, 1)
        
        return model_mean + nonzero_mask * torch.sqrt(model_variance) * noise


# =============================================================================
# 第七部分：可视化工具
# =============================================================================

def visualize_diffusion_process(ddpm: DDPM, x0: torch.Tensor, timesteps_to_show: List[int] = None):
    """
    可视化扩散过程。
    
    参数:
        ddpm: DDPM模型
        x0: 原始图像
        timesteps_to_show: 要显示的时间步列表
    """
    if timesteps_to_show is None:
        timesteps_to_show = [0, 50, 100, 200, 300, 500, 700, 999]
    
    x0 = x0[:1]  # 只取第一个样本
    
    fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(20, 3))
    
    for i, t in enumerate(timesteps_to_show):
        t_tensor = torch.tensor([t], device=x0.device)
        xt, _ = ddpm.q_sample(x0, t_tensor)
        
        img = xt[0].cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2  # 从[-1,1]转换到[0,1]
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f't={t}')
        axes[i].axis('off')
    
    plt.suptitle('Forward Diffusion Process (Adding Noise)')
    plt.tight_layout()
    return fig


def visualize_reverse_process(
    ddpm: DDPM, 
    shape: Tuple[int, ...],
    y: Optional[torch.Tensor] = None,
    save_interval: int = 100
):
    """
    可视化反向扩散过程(去噪)。
    """
    batch_size, channels, height, width = shape
    
    x = torch.randn(shape, device=ddpm.device)
    images = [x[0].cpu().clone()]
    
    for t in tqdm(range(ddpm.timesteps - 1, -1, -1), desc='Reverse Process'):
        t_batch = torch.full((batch_size,), t, device=ddpm.device, dtype=torch.long)
        x = ddpm.p_sample(x, t_batch, y)
        
        if t % save_interval == 0 or t == 0:
            images.append(x[0].cpu().clone())
    
    # 绘制
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(3 * n_images, 3))
    if n_images == 1:
        axes = [axes]
    
    for i, img in enumerate(images):
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np + 1) / 2
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].set_title(f'Step {max(0, ddpm.timesteps - i * save_interval)}')
        axes[i].axis('off')
    
    plt.suptitle('Reverse Diffusion Process (Denoising)')
    plt.tight_layout()
    return fig, images[-1]


def compare_schedules(timesteps: int = 1000):
    """比较不同的beta调度策略"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    schedules = ['linear', 'cosine', 'quadratic']
    
    for schedule in schedules:
        if schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, timesteps)
        elif schedule == 'quadratic':
            betas = torch.linspace(1e-4**0.5, 0.02**0.5, timesteps) ** 2
        else:  # cosine
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        axes[0, 0].plot(betas, label=schedule)
        axes[0, 1].plot(alphas, label=schedule)
        axes[1, 0].plot(alphas_cumprod, label=schedule)
        axes[1, 1].plot(torch.sqrt(alphas_cumprod), label=f'sqrt_ᾱ ({schedule})')
        axes[1, 1].plot(torch.sqrt(1 - alphas_cumprod), label=f'sqrt(1-ᾱ) ({schedule})', linestyle='--')
    
    axes[0, 0].set_title('Beta Schedule')
    axes[0, 0].set_ylabel('β')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title('Alpha Schedule')
    axes[0, 1].set_ylabel('α = 1-β')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].set_title('Cumulative Product of Alpha')
    axes[1, 0].set_ylabel('ᾱ')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].set_title('Signal vs Noise Ratio')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig


# =============================================================================
# 第八部分：训练辅助函数
# =============================================================================

def train_ddpm(
    ddpm: DDPM,
    dataloader: DataLoader,
    epochs: int,
    lr: float = 2e-4,
    save_path: Optional[str] = None,
    log_interval: int = 100,
    device: str = 'cuda'
):
    """
    训练DDPM模型。
    
    参数:
        ddpm: DDPM模型
        dataloader: 数据加载器
        epochs: 训练轮数
        lr: 学习率
        save_path: 模型保存路径
        log_interval: 日志打印间隔
    """
    optimizer = optim.Adam(ddpm.model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, (list, tuple)):
                x0 = batch[0].to(device)
                y = batch[1].to(device) if len(batch) > 1 else None
            else:
                x0 = batch.to(device)
                y = None
            
            loss = ddpm.training_step(x0, optimizer, y)
            epoch_losses.append(loss)
            losses.append(loss)
            
            if batch_idx % log_interval == 0:
                pbar.set_postfix({'loss': loss})
        
        avg_loss = np.mean(epoch_losses)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
        
        if save_path:
            torch.save({
                'epoch': epoch,
                'model_state_dict': ddpm.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'{save_path}_epoch{epoch+1}.pt')
    
    return losses


def create_synthetic_dataset(
    n_samples: int = 10000,
    img_size: int = 32,
    n_classes: int = 10
):
    """
    创建合成的彩色形状数据集，用于测试。
    生成简单的几何形状(圆、方、三角)。
    """
    images = []
    labels = []
    
    for i in range(n_samples):
        # 创建空白图像
        img = np.ones((img_size, img_size, 3)) * 255
        label = i % n_classes
        
        color = np.random.randint(0, 256, 3)
        center = (np.random.randint(img_size//4, 3*img_size//4), 
                  np.random.randint(img_size//4, 3*img_size//4))
        size = np.random.randint(img_size//6, img_size//3)
        
        if label < 3:  # 圆形
            y, x = np.ogrid[:img_size, :img_size]
            mask = (x - center[0])**2 + (y - center[1])**2 <= size**2
            img[mask] = color
        elif label < 6:  # 方形
            x1, y1 = max(0, center[0] - size), max(0, center[1] - size)
            x2, y2 = min(img_size, center[0] + size), min(img_size, center[1] + size)
            img[y1:y2, x1:x2] = color
        else:  # 三角形
            for dy in range(-size, size+1):
                for dx in range(-size, size+1):
                    if abs(dx) + abs(dy) <= size:
                        py, px = center[1] + dy, center[0] + dx
                        if 0 <= px < img_size and 0 <= py < img_size:
                            img[py, px] = color
        
        # 归一化到[-1, 1]
        img = (img / 255.0 - 0.5) * 2
        images.append(img.transpose(2, 0, 1))  # HWC -> CHW
        labels.append(label)
    
    images = torch.tensor(np.array(images), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return TensorDataset(images, labels)


# =============================================================================
# 第九部分：示例用法
# =============================================================================

def demo():
    """
    演示DDPM的完整流程。
    由于计算资源限制，这里使用小规模的合成数据集和简化模型。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 超参数
    img_size = 32
    channels = 3
    batch_size = 64
    timesteps = 1000
    epochs = 10
    
    # 创建模型
    model = UNet(
        in_channels=channels,
        model_channels=64,  # 减小模型规模
        out_channels=channels,
        num_res_blocks=1,
        attention_resolutions=(2,),
        channel_mult=(1, 2, 4),
    )
    
    # 创建DDPM
    ddpm = DDPM(
        model=model,
        timesteps=timesteps,
        beta_schedule='cosine',
        device=device
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建合成数据集
    print("创建合成数据集...")
    dataset = create_synthetic_dataset(n_samples=1000, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 可视化扩散过程
    print("可视化扩散过程...")
    sample_img, _ = dataset[0]
    sample_img = sample_img.unsqueeze(0).to(device)
    fig = visualize_diffusion_process(ddpm, sample_img)
    plt.savefig('forward_diffusion.png')
    plt.close()
    print("保存: forward_diffusion.png")
    
    # 训练模型
    print("开始训练...")
    losses = train_ddpm(
        ddpm=ddpm,
        dataloader=dataloader,
        epochs=epochs,
        lr=1e-3,
        device=device
    )
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    print("保存: training_loss.png")
    
    # 生成样本
    print("生成样本...")
    generated = ddpm.sample(
        batch_size=8,
        channels=channels,
        height=img_size,
        width=img_size
    )
    
    # 可视化生成的样本
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    for i in range(8):
        img = generated[i].cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.suptitle('Generated Samples')
    plt.tight_layout()
    plt.savefig('generated_samples.png')
    plt.close()
    print("保存: generated_samples.png")
    
    # 可视化反向扩散过程
    print("可视化反向扩散过程...")
    fig, final_img = visualize_reverse_process(
        ddpm, 
        shape=(1, channels, img_size, img_size),
        save_interval=200
    )
    plt.savefig('reverse_diffusion.png')
    plt.close()
    print("保存: reverse_diffusion.png")
    
    # 比较不同的调度策略
    print("比较不同的beta调度策略...")
    fig = compare_schedules(timesteps)
    plt.savefig('schedule_comparison.png')
    plt.close()
    print("保存: schedule_comparison.png")
    
    print("\n演示完成!")
    return ddpm


if __name__ == '__main__':
    # 运行演示
    ddpm = demo()
