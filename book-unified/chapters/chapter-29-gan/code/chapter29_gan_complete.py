"""
第二十九章：生成对抗网络(GAN)与扩散模型 - 完整代码实现
================================================================

本文件包含：
1. GAN基础 - MNIST手写数字生成
2. DCGAN - 深度卷积GAN
3. 条件GAN (cGAN) - 控制生成的数字类别
4. StyleGAN概念演示 - 风格混合与插值
5. 扩散模型 (DDPM) - 简化版完整实现
6. U-Net噪声预测网络
7. 扩散采样算法与可视化

作者: ML教材编写组
版本: 2.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import math
import os
from tqdm import tqdm
from dataclasses import dataclass

# 设置随机种子确保可复现性
def set_seed(seed=42):
    """设置随机种子确保实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# =============================================================================
# 第一部分：GAN基础 - 简单全连接GAN生成MNIST
# =============================================================================

class SimpleGenerator(nn.Module):
    """
    简单生成器 - 全连接网络
    
    架构: 噪声向量(100维) -> 隐藏层(256) -> 隐藏层(512) -> 图像(784维)
    
    原理: 通过多层非线性变换，将随机噪声映射到数据分布
    """
    def __init__(self, latent_dim=100, hidden_dims=[256, 512], image_size=784):
        super().__init__()
        self.latent_dim = latent_dim
        
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # 输出层使用tanh将值映射到[-1, 1]
        layers.append(nn.Linear(prev_dim, image_size))
        layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        输入: z - 噪声向量，形状 (batch_size, latent_dim)
        输出: 生成的图像，形状 (batch_size, 784)
        """
        img = self.model(z)
        return img
    
    def sample(self, batch_size: int, device='cpu') -> torch.Tensor:
        """采样生成图像"""
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.forward(z)


class SimpleDiscriminator(nn.Module):
    """
    简单判别器 - 全连接网络
    
    架构: 图像(784维) -> 隐藏层(512) -> 隐藏层(256) -> 真假概率(1维)
    
    原理: 学习区分真实图像和生成图像
    """
    def __init__(self, image_size=784, hidden_dims=[512, 256]):
        super().__init__()
        
        layers = []
        prev_dim = image_size
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # 输出层使用sigmoid输出概率
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        输入: img - 图像，形状 (batch_size, 784)
        输出: 真假概率，形状 (batch_size, 1)
        """
        validity = self.model(img)
        return validity


class SimpleGAN:
    """
    简单GAN训练器
    
    训练原理:
    1. 训练判别器: 最大化 log(D(x)) + log(1-D(G(z)))
    2. 训练生成器: 最大化 log(D(G(z)))
    
    这是一个极小极大博弈过程
    """
    def __init__(self, latent_dim=100, lr=0.0002, device='cpu'):
        self.latent_dim = latent_dim
        self.device = device
        
        # 初始化网络
        self.generator = SimpleGenerator(latent_dim=latent_dim).to(device)
        self.discriminator = SimpleDiscriminator().to(device)
        
        # 使用Adam优化器
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # 损失函数
        self.criterion = nn.BCELoss()
        
        # 训练历史
        self.history = {'g_loss': [], 'd_loss': [], 'd_real': [], 'd_fake': []}
        
    def train_step(self, real_images: torch.Tensor) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            real_images: 真实图像批次，形状 (batch_size, 784)
            
        Returns:
            包含各损失值的字典
        """
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        
        # 创建标签
        real_label = torch.ones(batch_size, 1, device=self.device)
        fake_label = torch.zeros(batch_size, 1, device=self.device)
        
        # ==================== 训练判别器 ====================
        self.optimizer_D.zero_grad()
        
        # 真实图像的损失
        real_pred = self.discriminator(real_images)
        d_real_loss = self.criterion(real_pred, real_label)
        
        # 生成图像的损失
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(z)
        fake_pred = self.discriminator(fake_images.detach())  # 不计算生成器梯度
        d_fake_loss = self.criterion(fake_pred, fake_label)
        
        # 判别器总损失
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()
        
        # ==================== 训练生成器 ====================
        self.optimizer_G.zero_grad()
        
        # 生成器希望判别器将假图像判断为真
        fake_pred = self.discriminator(fake_images)
        g_loss = self.criterion(fake_pred, real_label)
        
        g_loss.backward()
        self.optimizer_G.step()
        
        # 记录训练指标
        metrics = {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'd_real': real_pred.mean().item(),
            'd_fake': fake_pred.mean().item()
        }
        
        return metrics
    
    def train(self, dataloader: DataLoader, epochs: int, save_dir: str = './gan_outputs'):
        """
        完整训练循环
        
        Args:
            dataloader: MNIST数据加载器
            epochs: 训练轮数
            save_dir: 保存生成样本的目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"开始训练简单GAN，共{epochs}轮...")
        
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            for batch_idx, (real_images, _) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')):
                # 展平图像
                real_images = real_images.view(real_images.size(0), -1)
                
                metrics = self.train_step(real_images)
                epoch_g_loss += metrics['g_loss']
                epoch_d_loss += metrics['d_loss']
                
                # 每100批次打印一次
                if batch_idx % 100 == 0:
                    print(f"  [Batch {batch_idx}] D_loss: {metrics['d_loss']:.4f}, "
                          f"G_loss: {metrics['g_loss']:.4f}, "
                          f"D(real): {metrics['d_real']:.4f}, D(fake): {metrics['d_fake']:.4f}")
            
            # 记录平均损失
            n_batches = len(dataloader)
            self.history['g_loss'].append(epoch_g_loss / n_batches)
            self.history['d_loss'].append(epoch_d_loss / n_batches)
            self.history['d_real'].append(metrics['d_real'])
            self.history['d_fake'].append(metrics['d_fake'])
            
            # 生成并保存样本
            self.save_samples(epoch + 1, save_dir)
            
            print(f"Epoch {epoch+1}/{epochs} - D_loss: {self.history['d_loss'][-1]:.4f}, "
                  f"G_loss: {self.history['g_loss'][-1]:.4f}")
        
        print("训练完成!")
        self.plot_history(save_dir)
    
    def save_samples(self, epoch: int, save_dir: str):
        """保存生成的样本图像"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(64, self.latent_dim, device=self.device)
            generated = self.generator(z).cpu().view(64, 1, 28, 28)
            save_image(generated, f"{save_dir}/epoch_{epoch:03d}.png", 
                      nrow=8, normalize=True)
        self.generator.train()
    
    def plot_history(self, save_dir: str):
        """绘制训练历史曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 损失曲线
        ax1 = axes[0]
        ax1.plot(self.history['d_loss'], label='Discriminator Loss', linewidth=2)
        ax1.plot(self.history['g_loss'], label='Generator Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('GAN Training Losses', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 判别器输出
        ax2 = axes[1]
        ax2.plot(self.history['d_real'], label='D(real)', linewidth=2)
        ax2.plot(self.history['d_fake'], label='D(fake)', linewidth=2)
        ax2.axhline(y=0.5, color='r', linestyle='--', label='Nash Equilibrium')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.set_title('Discriminator Predictions', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_history.png', dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# 第二部分：DCGAN - 深度卷积GAN
# =============================================================================

class DCGANGenerator(nn.Module):
    """
    深度卷积生成器 (DCGAN)
    
    架构特点:
    - 使用转置卷积(反卷积)进行上采样
    - 使用BatchNorm稳定训练
    - 使用ReLU激活(除输出层用Tanh)
    
    输入: (batch_size, latent_dim, 1, 1)
    输出: (batch_size, 1, 28, 28)
    """
    def __init__(self, latent_dim=100, feature_maps=64):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            # 输入: 100 x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_maps * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # 状态: (feature_maps*4) x 4 x 4 = 256 x 4 x 4
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # 状态: (feature_maps*2) x 8 x 8 = 128 x 8 x 8
            
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # 状态: feature_maps x 16 x 16 = 64 x 16 x 16
            
            nn.ConvTranspose2d(feature_maps, 1, 4, 2, 3, bias=False),
            nn.Tanh()
            # 输出: 1 x 28 x 28
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """DCGAN推荐的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        输入: z - 噪声向量，形状 (batch_size, latent_dim)
        输出: 生成的图像，形状 (batch_size, 1, 28, 28)
        """
        # 将向量reshape为空间张量
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.main(z)
    
    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """
        在潜在空间进行插值
        
        Args:
            z1, z2: 两个潜在向量
            steps: 插值步数
            
        Returns:
            插值生成的图像序列
        """
        alphas = torch.linspace(0, 1, steps, device=z1.device).view(-1, 1)
        z_interp = alphas * z2 + (1 - alphas) * z1
        with torch.no_grad():
            images = self.forward(z_interp)
        return images


class DCGANDiscriminator(nn.Module):
    """
    深度卷积判别器 (DCGAN)
    
    架构特点:
    - 使用卷积层进行下采样
    - 使用BatchNorm稳定训练
    - 使用LeakyReLU激活
    - 不使用全连接层，使用全局平均池化
    
    输入: (batch_size, 1, 28, 28)
    输出: (batch_size, 1)
    """
    def __init__(self, feature_maps=64):
        super().__init__()
        
        self.main = nn.Sequential(
            # 输入: 1 x 28 x 28
            nn.Conv2d(1, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: feature_maps x 14 x 14
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (feature_maps*2) x 7 x 7
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (feature_maps*4) x 3 x 3
            
            nn.Conv2d(feature_maps * 4, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
            # 输出: 1 x 1 x 1
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """DCGAN推荐的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        输入: img - 图像，形状 (batch_size, 1, 28, 28)
        输出: 真假概率，形状 (batch_size, 1)
        """
        output = self.main(img)
        return output.view(-1, 1)


class DCGAN:
    """
    DCGAN训练器
    
    改进点:
    1. 使用卷积层替代全连接层
    2. 使用BatchNorm稳定训练
    3. 特定的权重初始化
    4. 使用Adam优化器，特定参数
    """
    def __init__(self, latent_dim=100, feature_maps=64, lr=0.0002, device='cpu'):
        self.latent_dim = latent_dim
        self.device = device
        
        # 初始化网络
        self.generator = DCGANGenerator(latent_dim=latent_dim, feature_maps=feature_maps).to(device)
        self.discriminator = DCGANDiscriminator(feature_maps=feature_maps).to(device)
        
        # 优化器 - DCGAN推荐的参数
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        self.criterion = nn.BCELoss()
        self.history = {'g_loss': [], 'd_loss': [], 'd_real': [], 'd_fake': []}
    
    def train_step(self, real_images: torch.Tensor) -> Dict[str, float]:
        """单步训练"""
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        
        real_label = torch.ones(batch_size, 1, device=self.device)
        fake_label = torch.zeros(batch_size, 1, device=self.device)
        
        # 训练判别器
        self.optimizer_D.zero_grad()
        
        real_pred = self.discriminator(real_images)
        d_real_loss = self.criterion(real_pred, real_label)
        
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(z)
        fake_pred = self.discriminator(fake_images.detach())
        d_fake_loss = self.criterion(fake_pred, fake_label)
        
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()
        
        # 训练生成器
        self.optimizer_G.zero_grad()
        
        fake_pred = self.discriminator(fake_images)
        g_loss = self.criterion(fake_pred, real_label)
        
        g_loss.backward()
        self.optimizer_G.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'd_real': real_pred.mean().item(),
            'd_fake': fake_pred.mean().item()
        }
    
    def train(self, dataloader: DataLoader, epochs: int, save_dir: str = './dcgan_outputs'):
        """完整训练"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"开始训练DCGAN，共{epochs}轮...")
        
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            for batch_idx, (real_images, _) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')):
                metrics = self.train_step(real_images)
                epoch_g_loss += metrics['g_loss']
                epoch_d_loss += metrics['d_loss']
                
                if batch_idx % 100 == 0:
                    print(f"  [Batch {batch_idx}] D_loss: {metrics['d_loss']:.4f}, "
                          f"G_loss: {metrics['g_loss']:.4f}, "
                          f"D(real): {metrics['d_real']:.4f}, D(fake): {metrics['d_fake']:.4f}")
            
            n_batches = len(dataloader)
            self.history['g_loss'].append(epoch_g_loss / n_batches)
            self.history['d_loss'].append(epoch_d_loss / n_batches)
            
            # 保存样本和插值
            self.save_samples(epoch + 1, save_dir)
            if epoch % 5 == 0:
                self.visualize_interpolation(save_dir, epoch + 1)
            
            print(f"Epoch {epoch+1}/{epochs} - D_loss: {self.history['d_loss'][-1]:.4f}, "
                  f"G_loss: {self.history['g_loss'][-1]:.4f}")
        
        self.plot_history(save_dir)
        print("训练完成!")
    
    def save_samples(self, epoch: int, save_dir: str):
        """保存生成的样本"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(64, self.latent_dim, device=self.device)
            generated = self.generator(z)
            save_image(generated, f"{save_dir}/dcgan_epoch_{epoch:03d}.png", 
                      nrow=8, normalize=True)
        self.generator.train()
    
    def visualize_interpolation(self, save_dir: str, epoch: int):
        """可视化潜在空间插值"""
        self.generator.eval()
        with torch.no_grad():
            z1 = torch.randn(1, self.latent_dim, device=self.device)
            z2 = torch.randn(1, self.latent_dim, device=self.device)
            
            images = self.generator.interpolate(z1, z2, steps=10)
            save_image(images, f"{save_dir}/interpolation_epoch_{epoch:03d}.png",
                      nrow=10, normalize=True)
        self.generator.train()
    
    def plot_history(self, save_dir: str):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        ax1.plot(self.history['d_loss'], label='Discriminator Loss', linewidth=2)
        ax1.plot(self.history['g_loss'], label='Generator Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('DCGAN Training Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        ax2.plot(self.history['d_real'], label='D(real)', linewidth=2)
        ax2.plot(self.history['d_fake'], label='D(fake)', linewidth=2)
        ax2.axhline(y=0.5, color='r', linestyle='--', label='Equilibrium')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Probability')
        ax2.set_title('Discriminator Predictions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/dcgan_history.png', dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# 第三部分：条件GAN (cGAN) - 控制生成特定数字
# =============================================================================

class ConditionalGenerator(nn.Module):
    """
    条件生成器 - 可以控制生成哪个数字
    
    通过将类别标签嵌入与噪声向量拼接，实现条件控制
    """
    def __init__(self, latent_dim=100, num_classes=10, feature_maps=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # 标签嵌入层
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # 合并latent_dim + num_classes作为输入
        input_dim = latent_dim + num_classes
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_dim, feature_maps * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_maps, 1, 4, 2, 3, bias=False),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        输入:
            z: 噪声向量，形状 (batch_size, latent_dim)
            labels: 类别标签，形状 (batch_size,)
        输出:
            生成的图像，形状 (batch_size, 1, 28, 28)
        """
        # 将标签转换为嵌入向量
        label_embed = self.label_embedding(labels)  # (batch_size, num_classes)
        
        # 拼接噪声和标签嵌入
        gen_input = torch.cat([z, label_embed], dim=1)  # (batch_size, latent_dim + num_classes)
        
        # reshape为空间张量
        gen_input = gen_input.view(gen_input.size(0), gen_input.size(1), 1, 1)
        
        return self.main(gen_input)


class ConditionalDiscriminator(nn.Module):
    """
    条件判别器 - 同时判断真假和类别
    
    通过将类别信息嵌入并与图像特征结合
    """
    def __init__(self, num_classes=10, feature_maps=64):
        super().__init__()
        self.num_classes = num_classes
        
        # 标签嵌入层
        self.label_embedding = nn.Embedding(num_classes, 28 * 28)
        
        # 输入通道 = 1(图像) + 1(标签嵌入reshape)
        self.main = nn.Sequential(
            nn.Conv2d(2, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 4, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        输入:
            img: 图像，形状 (batch_size, 1, 28, 28)
            labels: 类别标签，形状 (batch_size,)
        输出:
            真假概率，形状 (batch_size, 1)
        """
        # 将标签嵌入并reshape为图像大小
        label_embed = self.label_embedding(labels).view(-1, 1, 28, 28)
        
        # 拼接图像和标签嵌入
        d_input = torch.cat([img, label_embed], dim=1)  # (batch_size, 2, 28, 28)
        
        output = self.main(d_input)
        return output.view(-1, 1)


class ConditionalGAN:
    """
    条件GAN训练器
    
    可以控制生成特定类别的图像
    """
    def __init__(self, latent_dim=100, num_classes=10, feature_maps=64, lr=0.0002, device='cpu'):
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device
        
        self.generator = ConditionalGenerator(latent_dim, num_classes, feature_maps).to(device)
        self.discriminator = ConditionalDiscriminator(num_classes, feature_maps).to(device)
        
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        self.criterion = nn.BCELoss()
        self.history = {'g_loss': [], 'd_loss': []}
    
    def train_step(self, real_images: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """单步训练"""
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        labels = labels.to(self.device)
        
        real_label = torch.ones(batch_size, 1, device=self.device)
        fake_label = torch.zeros(batch_size, 1, device=self.device)
        
        # 训练判别器
        self.optimizer_D.zero_grad()
        
        real_pred = self.discriminator(real_images, labels)
        d_real_loss = self.criterion(real_pred, real_label)
        
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(z, labels)
        fake_pred = self.discriminator(fake_images.detach(), labels)
        d_fake_loss = self.criterion(fake_pred, fake_label)
        
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()
        
        # 训练生成器
        self.optimizer_G.zero_grad()
        
        fake_pred = self.discriminator(fake_images, labels)
        g_loss = self.criterion(fake_pred, real_label)
        
        g_loss.backward()
        self.optimizer_G.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item()
        }
    
    def train(self, dataloader: DataLoader, epochs: int, save_dir: str = './cgan_outputs'):
        """完整训练"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"开始训练条件GAN，共{epochs}轮...")
        
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            for batch_idx, (real_images, labels) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')):
                metrics = self.train_step(real_images, labels)
                epoch_g_loss += metrics['g_loss']
                epoch_d_loss += metrics['d_loss']
            
            n_batches = len(dataloader)
            self.history['g_loss'].append(epoch_g_loss / n_batches)
            self.history['d_loss'].append(epoch_d_loss / n_batches)
            
            # 保存每个数字的样本
            self.save_class_samples(epoch + 1, save_dir)
            
            print(f"Epoch {epoch+1}/{epochs} - D_loss: {self.history['d_loss'][-1]:.4f}, "
                  f"G_loss: {self.history['g_loss'][-1]:.4f}")
        
        print("训练完成!")
        self.visualize_all_classes(save_dir)
    
    def save_class_samples(self, epoch: int, save_dir: str):
        """为每个类别生成样本"""
        self.generator.eval()
        with torch.no_grad():
            fig_samples = []
            for class_idx in range(self.num_classes):
                z = torch.randn(8, self.latent_dim, device=self.device)
                labels = torch.full((8,), class_idx, dtype=torch.long, device=self.device)
                generated = self.generator(z, labels)
                fig_samples.append(generated)
            
            all_samples = torch.cat(fig_samples, dim=0)
            save_image(all_samples, f"{save_dir}/cgan_epoch_{epoch:03d}.png",
                      nrow=8, normalize=True)
        self.generator.train()
    
    def visualize_all_classes(self, save_dir: str):
        """可视化所有类别的生成结果"""
        self.generator.eval()
        with torch.no_grad():
            fig, axes = plt.subplots(self.num_classes, 10, figsize=(15, 15))
            
            for class_idx in range(self.num_classes):
                z = torch.randn(10, self.latent_dim, device=self.device)
                labels = torch.full((10,), class_idx, dtype=torch.long, device=self.device)
                generated = self.generator(z, labels).cpu()
                
                for i in range(10):
                    axes[class_idx, i].imshow(generated[i, 0], cmap='gray')
                    axes[class_idx, i].axis('off')
                
                axes[class_idx, 0].set_ylabel(f'Class {class_idx}', fontsize=12)
            
            plt.suptitle('Conditional GAN - Generated Samples by Class', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/all_classes_grid.png', dpi=150, bbox_inches='tight')
            plt.close()
        self.generator.train()
    
    def generate_specific_digit(self, digit: int, num_samples: int = 16) -> torch.Tensor:
        """
        生成特定数字的图像
        
        Args:
            digit: 要生成的数字 (0-9)
            num_samples: 生成样本数
            
        Returns:
            生成的图像张量
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            labels = torch.full((num_samples,), digit, dtype=torch.long, device=self.device)
            generated = self.generator(z, labels)
        self.generator.train()
        return generated


# =============================================================================
# 第四部分：StyleGAN概念演示 - 风格混合与潜在空间插值
# =============================================================================

class MappingNetwork(nn.Module):
    """
    StyleGAN的映射网络 - 将潜在向量z映射到中间潜在空间W
    
    目的: 解耦潜在空间，实现更好的属性控制和风格混合
    """
    def __init__(self, latent_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = latent_dim if i == 0 else w_dim
            layers.extend([
                nn.Linear(in_dim, w_dim),
                nn.LeakyReLU(0.2)
            ])
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        输入: z - 原始潜在向量，形状 (batch_size, latent_dim)
        输出: w - 中间潜在向量，形状 (batch_size, w_dim)
        """
        # 归一化输入
        z = z / torch.sqrt(torch.sum(z ** 2, dim=1, keepdim=True) + 1e-8)
        return self.mapping(z)


class StyleGANGenerator(nn.Module):
    """
    简化版StyleGAN生成器
    
    核心概念:
    1. 映射网络将z映射到w空间
    2. 自适应实例归一化(AdaIN)注入风格
    3. 风格混合 - 在不同层使用不同的w向量
    """
    def __init__(self, latent_dim=512, w_dim=512, image_size=28):
        super().__init__()
        self.w_dim = w_dim
        self.image_size = image_size
        
        # 映射网络
        self.mapping = MappingNetwork(latent_dim, w_dim)
        
        # 可学习的常数输入
        self.const_input = nn.Parameter(torch.randn(1, 512, 4, 4))
        
        # 合成网络 - 简化为全连接层
        self.synthesis = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, image_size * image_size),
            nn.Tanh()
        )
        
        # 风格调制层参数
        self.style_layers = nn.ModuleList([
            nn.Linear(w_dim, 1024),
            nn.Linear(w_dim, 512)
        ])
    
    def forward(self, z: torch.Tensor, w_mix: Optional[torch.Tensor] = None, 
                mix_layer: int = -1) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: 潜在向量
            w_mix: 用于风格混合的第二个w向量
            mix_layer: 从哪一层开始混合
            
        Returns:
            生成的图像
        """
        batch_size = z.size(0)
        
        # 映射到w空间
        w = self.mapping(z)
        
        # 风格混合
        if w_mix is not None and mix_layer >= 0:
            w_mix_mapped = self.mapping(w_mix)
            # 这里简化处理，实际应该控制每层使用哪个w
        
        # 从常数输入开始
        x = self.const_input.repeat(batch_size, 1, 1, 1)
        x = x.view(batch_size, -1)
        
        # 应用合成网络
        img = self.synthesis(x)
        img = img.view(batch_size, 1, self.image_size, self.image_size)
        
        return img
    
    def style_mixing(self, z1: torch.Tensor, z2: torch.Tensor, 
                     layer_idx: int = 4) -> torch.Tensor:
        """
        风格混合 - 前半部分使用z1的风格，后半部分使用z2的风格
        
        Args:
            z1: 第一个潜在向量(控制粗略特征)
            z2: 第二个潜在向量(控制精细特征)
            layer_idx: 混合的层索引
            
        Returns:
            风格混合后的生成图像
        """
        return self.forward(z1, w_mix=z2, mix_layer=layer_idx)


class StyleGANDemo:
    """
    StyleGAN概念演示
    
    展示:
    1. 潜在空间插值
    2. 风格混合
    3. 属性解耦
    """
    def __init__(self, latent_dim=512, device='cpu'):
        self.latent_dim = latent_dim
        self.device = device
        self.generator = StyleGANGenerator(latent_dim).to(device)
    
    def visualize_interpolation(self, num_steps: int = 10, save_path: str = 'stylegan_interp.png'):
        """
        可视化潜在空间插值
        
        展示从一个人脸(或数字)平滑过渡到另一个人脸(或数字)
        """
        z1 = torch.randn(1, self.latent_dim, device=self.device)
        z2 = torch.randn(1, self.latent_dim, device=self.device)
        
        alphas = torch.linspace(0, 1, num_steps, device=self.device)
        
        self.generator.eval()
        with torch.no_grad():
            images = []
            for alpha in alphas:
                z_interp = alpha * z2 + (1 - alpha) * z1
                img = self.generator(z_interp)
                images.append(img)
            
            all_images = torch.cat(images, dim=0)
            save_image(all_images, save_path, nrow=num_steps, normalize=True)
        
        print(f"插值可视化已保存到: {save_path}")
    
    def visualize_style_mixing(self, save_path: str = 'stylegan_mixing.png'):
        """
        可视化风格混合
        
        展示组合不同源的风格特征
        """
        # 生成4个不同的潜在向量
        z_coarse = torch.randn(1, self.latent_dim, device=self.device)
        z_fine_list = [torch.randn(1, self.latent_dim, device=self.device) for _ in range(4)]
        
        self.generator.eval()
        with torch.no_grad():
            images = []
            
            # 第一行: 使用z_coarse生成的原始图像
            img_coarse = self.generator(z_coarse)
            images.append(img_coarse)
            
            # 后续行: z_coarse控制粗略特征，不同的z_fine控制精细特征
            for z_fine in z_fine_list:
                img_mixed = self.generator.style_mixing(z_coarse, z_fine)
                images.append(img_mixed)
            
            all_images = torch.cat(images, dim=0)
            save_image(all_images, save_path, nrow=1, normalize=True)
        
        print(f"风格混合可视化已保存到: {save_path}")
    
    def explore_w_space(self, num_samples: int = 16, save_path: str = 'w_space_explore.png'):
        """
        探索W空间的性质
        
        展示映射到W空间后的属性解耦
        """
        z = torch.randn(num_samples, self.latent_dim, device=self.device)
        
        self.generator.eval()
        with torch.no_grad():
            # 通过映射网络
            w = self.generator.mapping(z)
            
            # 生成图像
            images = self.generator(z)
            save_image(images, save_path, nrow=4, normalize=True)
        
        # 可视化W空间的统计特性
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Z空间的分布
        ax1 = axes[0]
        ax1.hist(z.cpu().numpy().flatten(), bins=50, alpha=0.7, label='Z space')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution in Z Space')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # W空间的分布
        ax2 = axes[1]
        ax2.hist(w.cpu().numpy().flatten(), bins=50, alpha=0.7, color='orange', label='W space')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution in W Space (after mapping)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"W空间探索结果已保存")


# =============================================================================
# 第五部分：扩散模型 (DDPM) - 简化版完整实现
# =============================================================================

@dataclass
class DDPMConfig:
    """DDPM配置参数"""
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    schedule: str = 'linear'  # 'linear' or 'cosine'
    image_size: int = 28
    channels: int = 1
    batch_size: int = 128
    epochs: int = 50
    lr: float = 2e-4


class TimeEmbedding(nn.Module):
    """
    正弦/余弦时间步嵌入
    
    原理: 不同频率的正弦/余弦函数可以让模型学习时间相关的模式
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
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    残差块，用于U-Net
    """
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)
        
        self.time_mlp = nn.Linear(time_channels, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()  # Swish激活函数
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
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
    """自注意力块，用于捕捉长距离依赖"""
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
        
        q = q.reshape(B, C, H * W).transpose(1, 2)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W).transpose(1, 2)
        
        attn = torch.bmm(q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        h = torch.bmm(attn, v)
        h = h.transpose(1, 2).reshape(B, C, H, W)
        
        return x + self.proj(h)


class UNet(nn.Module):
    """
    U-Net噪声预测网络
    
    这是DDPM的核心组件，负责根据当前时刻的加噪图像预测添加的噪声
    """
    def __init__(
        self,
        in_channels: int = 1,
        model_channels: int = 64,
        out_channels: int = 1,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (2,),
        dropout: float = 0.1,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        time_embed_dim = model_channels * 4
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # 输入投影
        self.input_proj = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        ch_list = [ch]
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(ch, out_ch, time_embed_dim, dropout))
                ch = out_ch
                ch_list.append(ch)
            
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1))
                ch_list.append(ch)
        
        # 中间层
        self.mid_block1 = ResidualBlock(ch, ch, time_embed_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResidualBlock(ch, ch, time_embed_dim, dropout)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                self.up_blocks.append(ResidualBlock(ch + ch_list.pop(), out_ch, time_embed_dim, dropout))
                ch = out_ch
            
            if level != 0:
                self.up_blocks.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(ch, ch, kernel_size=3, padding=1)
                ))
        
        # 输出层
        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x: 加噪图像，形状 (batch_size, channels, height, width)
            timesteps: 时间步，形状 (batch_size,)
        输出:
            预测的噪声，形状与x相同
        """
        # 时间嵌入
        t_emb = self.time_embed(timesteps)
        
        # 输入投影
        h = self.input_proj(x)
        
        # 下采样
        hs = [h]
        for module in self.down_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, t_emb)
            else:
                h = module(h)
            hs.append(h)
        
        # 中间层
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # 上采样
        for module in self.up_blocks:
            if isinstance(module, ResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, t_emb)
            else:
                h = module(h)
        
        return self.out(h)


class DDPM:
    """
    去噪扩散概率模型 (Denoising Diffusion Probabilistic Model)
    
    核心思想:
    1. 前向过程: 逐步添加高斯噪声，直到图像变成纯噪声
    2. 反向过程: 学习逐步去噪，从噪声恢复图像
    3. 训练目标: 预测添加的噪声(MSE损失)
    """
    def __init__(self, config: DDPMConfig, device='cpu'):
        self.config = config
        self.device = device
        
        # 设置噪声调度
        self._setup_noise_schedule()
        
        # 初始化U-Net模型
        self.model = UNet(
            in_channels=config.channels,
            model_channels=64,
            out_channels=config.channels,
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        
        # 训练历史
        self.history = {'loss': []}
    
    def _setup_noise_schedule(self):
        """设置噪声调度参数"""
        config = self.config
        
        if config.schedule == 'linear':
            # 线性调度
            self.betas = torch.linspace(config.beta_start, config.beta_end, config.num_timesteps, device=self.device)
        elif config.schedule == 'cosine':
            # 余弦调度
            s = 0.008
            t = torch.linspace(0, 1, config.num_timesteps + 1, device=self.device)
            alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.betas = torch.clip(1 - alphas_cumprod[1:] / alphas_cumprod[:-1], 0.0001, 0.9999)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 预计算有用的量
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 后验方差
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向扩散过程 - 添加噪声
        
        公式: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        
        Args:
            x0: 原始图像
            t: 时间步
            noise: 噪声(如果为None则随机采样)
            
        Returns:
            x_t: 加噪后的图像
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise
    
    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        使用U-Net预测噪声
        
        Args:
            x_t: 当前时刻的加噪图像
            t: 时间步
            
        Returns:
            预测的噪声
        """
        return self.model(x_t, t)
    
    def p_sample(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        反向去噪过程 - 单步采样
        
        公式: x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta(x_t,t)) + sigma_t * z
        
        Args:
            x_t: 当前时刻的图像
            t: 时间步
            
        Returns:
            x_{t-1}: 去噪后的图像
        """
        t_tensor = torch.full((x_t.size(0),), t, device=self.device, dtype=torch.long)
        
        # 预测噪声
        pred_noise = self.predict_noise(x_t, t_tensor)
        
        # 计算均值
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]
        
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - alpha_cumprod_t)
        
        mean = coef1 * (x_t - coef2 * pred_noise)
        
        # 添加噪声(除了最后一步)
        if t > 0:
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(self.posterior_variance[t])
            return mean + variance * noise
        else:
            return mean
    
    def sample(self, batch_size: int = 64) -> torch.Tensor:
        """
        完整的采样过程 - 从噪声生成图像
        
        Args:
            batch_size: 生成的图像数量
            
        Returns:
            生成的图像
        """
        self.model.eval()
        
        # 从纯噪声开始
        x = torch.randn(batch_size, self.config.channels, 
                       self.config.image_size, self.config.image_size, 
                       device=self.device)
        
        # 逐步去噪
        with torch.no_grad():
            for t in tqdm(reversed(range(self.config.num_timesteps)), 
                         desc='Sampling', total=self.config.num_timesteps):
                x = self.p_sample(x, t)
        
        self.model.train()
        return x
    
    def train_step(self, x0: torch.Tensor) -> float:
        """
        单步训练
        
        训练目标: 最小化预测噪声与实际噪声的MSE
        
        Args:
            x0: 原始图像批次
            
        Returns:
            损失值
        """
        x0 = x0.to(self.device)
        batch_size = x0.size(0)
        
        # 随机采样时间步
        t = torch.randint(0, self.config.num_timesteps, (batch_size,), device=self.device)
        
        # 采样噪声
        noise = torch.randn_like(x0)
        
        # 前向过程: 添加噪声
        x_t = self.q_sample(x0, t, noise)
        
        # 预测噪声
        pred_noise = self.predict_noise(x_t, t)
        
        # 计算MSE损失
        loss = F.mse_loss(pred_noise, noise)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, dataloader: DataLoader, epochs: int, save_dir: str = './ddpm_outputs'):
        """
        完整训练循环
        
        Args:
            dataloader: 数据加载器
            epochs: 训练轮数
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"开始训练DDPM，共{epochs}轮...")
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')):
                loss = self.train_step(images)
                epoch_loss += loss
                num_batches += 1
                
                if batch_idx % 100 == 0:
                    print(f"  [Batch {batch_idx}] Loss: {loss:.6f}")
            
            avg_loss = epoch_loss / num_batches
            self.history['loss'].append(avg_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.6f}")
            
            # 每5轮生成并保存样本
            if (epoch + 1) % 5 == 0:
                samples = self.sample(64)
                save_image(samples, f"{save_dir}/ddpm_epoch_{epoch+1:03d}.png", 
                          nrow=8, normalize=True)
                print(f"  已保存生成样本到 {save_dir}/ddpm_epoch_{epoch+1:03d}.png")
        
        # 绘制训练曲线
        self.plot_history(save_dir)
        print("训练完成!")
    
    def visualize_diffusion_process(self, x0: torch.Tensor, num_steps: int = 10, 
                                     save_path: str = 'diffusion_process.png'):
        """
        可视化前向扩散过程
        
        展示图像如何逐步变成噪声
        """
        x0 = x0[:1].to(self.device)  # 只取一张图像
        
        # 选择均匀分布的时间步
        timesteps = torch.linspace(0, self.config.num_timesteps - 1, num_steps, dtype=torch.long)
        
        images = [x0.cpu()]
        for t in timesteps[1:]:
            t_tensor = torch.tensor([t], device=self.device)
            x_t = self.q_sample(x0, t_tensor)
            images.append(x_t.cpu())
        
        all_images = torch.cat(images, dim=0)
        save_image(all_images, save_path, nrow=num_steps, normalize=True)
        print(f"扩散过程可视化已保存到: {save_path}")
    
    def visualize_denoising_process(self, save_path: str = 'denoising_process.png'):
        """
        可视化反向去噪过程
        
        展示如何从噪声逐步恢复图像
        """
        self.model.eval()
        
        # 从噪声开始
        x = torch.randn(1, self.config.channels, 
                       self.config.image_size, self.config.image_size, 
                       device=self.device)
        
        # 保存特定时间步的图像
        save_steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
        images = []
        
        with torch.no_grad():
            for t in reversed(range(self.config.num_timesteps)):
                x = self.p_sample(x, t)
                if t in save_steps:
                    images.append(x.cpu())
        
        # 反转顺序使时间正确
        images = list(reversed(images))
        all_images = torch.cat(images, dim=0)
        save_image(all_images, save_path, nrow=len(images), normalize=True)
        print(f"去噪过程可视化已保存到: {save_path}")
        
        self.model.train()
    
    def plot_history(self, save_dir: str):
        """绘制训练历史"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['loss'], linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('DDPM Training Loss', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/ddpm_history.png', dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# 第六部分：工具函数与演示
# =============================================================================

def get_mnist_dataloader(batch_size=128, image_size=28):
    """
    获取MNIST数据加载器
    
    Args:
        batch_size: 批次大小
        image_size: 图像大小
        
    Returns:
        训练数据加载器
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
    ])
    
    dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def demo_simple_gan():
    """演示简单GAN的训练"""
    print("=" * 60)
    print("演示1: 简单GAN (全连接网络)")
    print("=" * 60)
    
    set_seed(42)
    dataloader = get_mnist_dataloader(batch_size=64)
    
    gan = SimpleGAN(latent_dim=100, lr=0.0002, device=device)
    gan.train(dataloader, epochs=20, save_dir='./outputs/simple_gan')
    
    print("简单GAN训练完成！\n")


def demo_dcgan():
    """演示DCGAN的训练"""
    print("=" * 60)
    print("演示2: DCGAN (深度卷积GAN)")
    print("=" * 60)
    
    set_seed(42)
    dataloader = get_mnist_dataloader(batch_size=128)
    
    dcgan = DCGAN(latent_dim=100, feature_maps=64, lr=0.0002, device=device)
    dcgan.train(dataloader, epochs=30, save_dir='./outputs/dcgan')
    
    print("DCGAN训练完成！\n")


def demo_conditional_gan():
    """演示条件GAN的训练"""
    print("=" * 60)
    print("演示3: 条件GAN (控制生成数字类别)")
    print("=" * 60)
    
    set_seed(42)
    dataloader = get_mnist_dataloader(batch_size=128)
    
    cgan = ConditionalGAN(latent_dim=100, num_classes=10, feature_maps=64, lr=0.0002, device=device)
    cgan.train(dataloader, epochs=30, save_dir='./outputs/cgan')
    
    # 生成特定数字
    print("\n生成特定数字的样本:")
    for digit in range(10):
        samples = cgan.generate_specific_digit(digit, num_samples=8)
        save_image(samples, f'./outputs/cgan/digit_{digit}.png', nrow=8, normalize=True)
    
    print("条件GAN训练完成！\n")


def demo_stylegan():
    """演示StyleGAN概念"""
    print("=" * 60)
    print("演示4: StyleGAN概念 (风格混合与插值)")
    print("=" * 60)
    
    set_seed(42)
    stylegan = StyleGANDemo(latent_dim=512, device=device)
    
    # 可视化插值
    stylegan.visualize_interpolation(num_steps=10, save_path='./outputs/stylegan/interpolation.png')
    
    # 可视化风格混合
    stylegan.visualize_style_mixing(save_path='./outputs/stylegan/style_mixing.png')
    
    # 探索W空间
    stylegan.explore_w_space(num_samples=16, save_path='./outputs/stylegan/w_space.png')
    
    print("StyleGAN概念演示完成！\n")


def demo_ddpm():
    """演示DDPM的训练与采样"""
    print("=" * 60)
    print("演示5: DDPM (去噪扩散概率模型)")
    print("=" * 60)
    
    set_seed(42)
    dataloader = get_mnist_dataloader(batch_size=128)
    
    # 创建配置
    config = DDPMConfig(
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        schedule='linear',
        image_size=28,
        channels=1,
        batch_size=128,
        epochs=50,
        lr=2e-4
    )
    
    # 初始化DDPM
    ddpm = DDPM(config, device=device)
    
    # 训练
    ddpm.train(dataloader, epochs=20, save_dir='./outputs/ddpm')
    
    # 可视化扩散过程
    for images, _ in dataloader:
        ddpm.visualize_diffusion_process(images, num_steps=10, 
                                         save_path='./outputs/ddpm/diffusion_process.png')
        break
    
    # 可视化去噪过程
    ddpm.visualize_denoising_process(save_path='./outputs/ddpm/denoising_process.png')
    
    print("DDPM训练与演示完成！\n")


def compare_models():
    """比较不同生成模型的特点"""
    print("=" * 60)
    print("生成模型特点比较")
    print("=" * 60)
    
    comparison = """
    ┌───────────────┬─────────────────────┬─────────────────────┬─────────────────────┐
    │     模型      │        优点         │        缺点         │      典型应用       │
    ├───────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
    │   简单GAN     │ 结构简单，易于理解  │ 训练不稳定，质量低  │ 教学演示，概念理解  │
    ├───────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
    │    DCGAN      │ 使用CNN，质量更高   │ 仍可能模式坍塌      │ 图像生成，风格迁移  │
    ├───────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
    │   条件GAN     │ 可控生成，类别指定  │ 需要标签数据        │ 数字生成，标签编辑  │
    ├───────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
    │   StyleGAN    │ 风格解耦，高质量    │ 计算复杂，训练难    │ 人脸生成，风格混合  │
    ├───────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
    │     DDPM      │ 训练稳定，模式完整  │ 采样慢，步骤多      │ 图像合成，超分辨率  │
    └───────────────┴─────────────────────┴─────────────────────┴─────────────────────┘
    
    核心概念总结:
    1. GAN: 生成器与判别器的博弈，追求高质量但训练困难
    2. cGAN: 引入条件控制，实现可控生成
    3. StyleGAN: 映射网络解耦潜在空间，实现精细控制
    4. DDPM: 学习去噪过程，训练稳定但采样慢
    """
    print(comparison)


def main():
    """
    主函数 - 运行所有演示
    
    注意: 完整训练需要较长时间，可以根据需要选择运行
    """
    # 创建输出目录
    os.makedirs('./outputs', exist_ok=True)
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     第二十九章: 生成对抗网络(GAN)与扩散模型 - 完整演示         ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    本演示包含以下内容:
    1. 简单GAN - 理解GAN基本原理
    2. DCGAN - 使用卷积网络提升质量
    3. 条件GAN - 实现可控生成
    4. StyleGAN概念 - 风格混合与插值
    5. DDPM - 扩散模型实现
    
    每个演示会保存生成结果到 ./outputs/ 目录
    """)
    
    # 模型比较表
    compare_models()
    
    # 选择要运行的演示(默认全部运行，可以注释掉不需要的)
    try:
        # 演示1: 简单GAN (约5-10分钟)
        demo_simple_gan()
        
        # 演示2: DCGAN (约10-15分钟)
        demo_dcgan()
        
        # 演示3: 条件GAN (约10-15分钟)
        demo_conditional_gan()
        
        # 演示4: StyleGAN概念 (快速，无需训练)
        demo_stylegan()
        
        # 演示5: DDPM (约30-60分钟，取决于设备)
        demo_ddpm()
        
    except KeyboardInterrupt:
        print("\n用户中断演示")
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    所有演示已完成！                            ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    生成结果保存在 ./outputs/ 目录下的各子文件夹中:
    - simple_gan/: 简单GAN的生成结果
    - dcgan/: DCGAN的生成结果
    - cgan/: 条件GAN的生成结果
    - stylegan/: StyleGAN概念演示结果
    - ddpm/: DDPM的生成结果
    
    思考题:
    1. 比较不同GAN的生成质量，分析原因
    2. 尝试修改潜在维度，观察生成结果变化
    3. 在cGAN中，同一个数字的不同样本有何差异？
    4. DDPM的扩散和去噪过程有何对称性？
    """)


if __name__ == "__main__":
    main()
