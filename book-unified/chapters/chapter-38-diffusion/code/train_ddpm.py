"""
DDPM训练与采样完整示例
第38章：扩散模型与生成式AI
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# 导入之前的实现
from ddpm_from_scratch import SimpleUNet, DDPM, visualize_diffusion_process, visualize_samples


class DiffusionTrainer:
    """
    DDPM训练器
    """
    def __init__(
        self,
        model: nn.Module,
        ddpm: DDPM,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda'
    ):
        self.model = model
        self.ddpm = ddpm
        self.optimizer = optimizer
        self.device = device
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            
            # 计算损失
            loss = self.ddpm.forward(images)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        
        for images, _ in tqdm(dataloader, desc='Validation'):
            images = images.to(self.device)
            loss = self.ddpm.forward(images)
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 100,
        save_dir: str = 'checkpoints',
        sample_every: int = 10
    ):
        """完整训练流程"""
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Train Loss: {train_loss:.4f}")
            
            # 保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f'ddpm_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
            
            # 生成样本
            if (epoch + 1) % sample_every == 0:
                print("Generating samples...")
                samples = self.ddpm.sample(batch_size=16, image_size=32)
                visualize_samples(samples, nrow=4)
        
        print("\nTraining completed!")
    
    def plot_losses(self):
        """绘制损失曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_losses.png', dpi=150)
        plt.show()


def demo_training():
    """
    完整训练演示（使用CIFAR-10）
    """
    print("="*60)
    print("DDPM Training Demo on CIFAR-10")
    print("="*60)
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 超参数
    BATCH_SIZE = 128
    EPOCHS = 100
    LR = 1e-4
    TIMESTEPS = 1000
    IMAGE_SIZE = 32
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
    ])
    
    # 加载CIFAR-10
    print("\nLoading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Iterations per epoch: {len(train_loader)}")
    
    # 创建模型
    print("\nInitializing model...")
    model = SimpleUNet(in_channels=3, out_channels=3, time_emb_dim=256)
    ddpm = DDPM(model, timesteps=TIMESTEPS, device=device)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 训练器
    trainer = DiffusionTrainer(model, ddpm, optimizer, device=device)
    
    # 可视化扩散过程
    print("\nVisualizing forward diffusion process...")
    sample_batch, _ = next(iter(train_loader))
    sample_batch = sample_batch.to(device)
    visualize_diffusion_process(ddpm, sample_batch, num_steps=10)
    
    # 训练
    print("\nStarting training...")
    trainer.train(
        train_loader=train_loader,
        epochs=EPOCHS,
        save_dir='checkpoints',
        sample_every=20
    )
    
    # 绘制损失曲线
    trainer.plot_losses()
    
    # 最终生成
    print("\nGenerating final samples...")
    final_samples = ddpm.sample(batch_size=16, image_size=IMAGE_SIZE)
    visualize_samples(final_samples, nrow=4)
    
    print("\nDemo completed!")


def demo_pretrained():
    """
    使用预训练模型生成图像
    """
    print("="*60)
    print("DDPM Sampling Demo")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = SimpleUNet(in_channels=3, out_channels=3, time_emb_dim=256)
    ddpm = DDPM(model, timesteps=1000, device=device)
    
    # 如果有检查点，加载它
    checkpoint_path = 'checkpoints/ddpm_epoch_100.pt'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded!")
    else:
        print("No checkpoint found. Using random initialization.")
        print("(Results will be random noise)")
    
    # 生成样本
    print("\nGenerating samples...")
    samples = ddpm.sample(batch_size=16, image_size=32)
    visualize_samples(samples, nrow=4)
    
    print("\nSampling completed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'sample':
        demo_pretrained()
    else:
        demo_training()
