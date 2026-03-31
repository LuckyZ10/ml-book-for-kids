"""
DDPM - 从零实现（PyTorch版本）
第38章：扩散模型与生成式AI - 代码实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """
    正弦位置编码 - 将时间步t转换为向量表示
    
    原理：使用不同频率的正弦/余弦函数，让模型感知时间信息
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """
    基础卷积块：Conv + GroupNorm + SiLU
    """
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, up: bool = False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.act(self.time_mlp(t))
        
        # 第一个卷积
        h = self.norm1(self.conv1(x))
        h = h + t_emb[:, :, None, None]  # 加入时间信息
        h = self.act(h)
        
        # 第二个卷积
        h = self.norm2(self.conv2(h))
        h = self.act(h)
        
        return h


class SimpleUNet(nn.Module):
    """
    简化的UNet架构用于噪声预测
    
    结构：Encoder -> Bottleneck -> Decoder
    特点：跳跃连接，时间嵌入注入
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        time_emb_dim: int = 256
    ):
        super().__init__()
        
        # 时间编码
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Encoder（下采样）
        self.conv0 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down1 = Block(64, 128, time_emb_dim)
        self.down2 = Block(128, 256, time_emb_dim)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1)
        )
        
        # Decoder（上采样）
        self.up2 = Block(256, 128, time_emb_dim, up=True)
        self.up1 = Block(128, 64, time_emb_dim, up=True)
        self.conv_out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, timestep):
        # 时间编码
        t = self.time_mlp(timestep)
        
        # Encoder
        x0 = self.conv0(x)
        x1 = self.down1(x0, t)
        x2 = self.down2(x1, t)
        
        # Bottleneck
        h = self.bottleneck(x2)
        
        # Decoder（带跳跃连接）
        h = self.up2(h, t)
        h = self.up1(h, t)
        
        return self.conv_out(h)


class DDPM:
    """
    DDPM训练和采样器
    """
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.timesteps = timesteps
        self.device = device
        
        # 线性噪声调度
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        
        # 预计算alpha
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 计算扩散过程的参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 计算后验方差
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散过程：q(x_t | x_0)
        根据x_0和t，采样x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t, t, noise):
        """
        从预测的噪声恢复x_0
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def p_sample(self, x_t, t):
        """
        单步反向采样：p(x_{t-1} | x_t)
        """
        # 预测噪声
        noise_pred = self.model(x_t, t)
        
        # 计算x_0的预测
        x_0_pred = self.predict_start_from_noise(x_t, t, noise_pred)
        
        # 计算后验均值
        alphas_t = self.alphas[t][:, None, None, None]
        betas_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        coef1 = torch.sqrt(self.alphas_cumprod_prev[t])[:, None, None, None] * betas_t / (1 - self.alphas_cumprod[t])[:, None, None, None]
        coef2 = torch.sqrt(alphas_t) * (1 - self.alphas_cumprod_prev[t])[:, None, None, None] / (1 - self.alphas_cumprod[t])[:, None, None, None]
        
        model_mean = coef1 * x_0_pred + coef2 * x_t
        
        # 采样
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None, None, None]
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, batch_size=16, channels=3, image_size=32):
        """
        完整采样过程：从噪声生成图像
        """
        self.model.eval()
        
        # 从标准高斯采样
        img = torch.randn(batch_size, channels, image_size, image_size).to(self.device)
        
        # 迭代去噪
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t_batch)
        
        return img

    def p_losses(self, x_start, t, noise=None):
        """
        训练损失：预测噪声的均方误差
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 采样x_t
        x_t = self.q_sample(x_start, t, noise)
        
        # 预测噪声
        noise_pred = self.model(x_t, t)
        
        # MSE损失
        loss = F.mse_loss(noise_pred, noise)
        
        return loss

    def forward(self, x):
        """
        训练前向传播
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 随机采样时间步
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        
        return self.p_losses(x, t)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    线性噪声调度
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    余弦噪声调度（效果更好）
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# 可视化工具
def visualize_diffusion_process(ddpm, x_start, num_steps=10):
    """
    可视化前向扩散过程
    """
    fig, axes = plt.subplots(1, num_steps, figsize=(20, 2))
    
    timesteps_to_show = torch.linspace(0, ddpm.timesteps-1, num_steps).long()
    
    for idx, t_val in enumerate(timesteps_to_show):
        t = torch.tensor([t_val]).to(ddpm.device)
        x_t = ddpm.q_sample(x_start[:1], t)
        
        # 转换为可显示的格式
        img = (x_t[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
        img = np.clip(img, 0, 1)
        
        axes[idx].imshow(img)
        axes[idx].set_title(f't={t_val.item()}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('diffusion_process.png', dpi=150)
    plt.show()


def visualize_samples(samples, nrow=4):
    """
    可视化生成的样本
    """
    fig, axes = plt.subplots(nrow, nrow, figsize=(8, 8))
    
    for i in range(nrow):
        for j in range(nrow):
            idx = i * nrow + j
            if idx < len(samples):
                img = (samples[idx].cpu().permute(1, 2, 0).numpy() + 1) / 2
                img = np.clip(img, 0, 1)
                axes[i, j].imshow(img)
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('generated_samples.png', dpi=150)
    plt.show()


# 示例用法
if __name__ == "__main__":
    print("DDPM Implementation")
    print("=" * 50)
    
    # 创建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = SimpleUNet(in_channels=3, out_channels=3)
    ddpm = DDPM(model, timesteps=1000, device=device)
    
    # 测试前向过程
    x = torch.randn(4, 3, 32, 32).to(device)
    t = torch.randint(0, 1000, (4,)).to(device)
    x_t = ddpm.q_sample(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Noised shape: {x_t.shape}")
    
    # 测试训练损失
    loss = ddpm.forward(x)
    print(f"Training loss: {loss.item():.4f}")
    
    print("\nDDPM implementation ready!")
    print("To train, use: loss = ddpm.forward(batch)")
    print("To sample, use: samples = ddpm.sample(batch_size=16)")
