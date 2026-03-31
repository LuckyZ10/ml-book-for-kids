"""
NeRF训练脚本
第39章：3D视觉与神经辐射场
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

from nerf_from_scratch import NeRF, get_rays, render_rays, positional_encoding


class RayDataset(Dataset):
    """
    射线数据集
    """
    def __init__(self, images, poses, focal, H, W):
        """
        Args:
            images: 图像数组 (N, H, W, 3)
            poses: 相机位姿 (N, 4, 4)
            focal: 焦距
            H, W: 图像高宽
        """
        self.images = images
        self.poses = poses
        self.focal = focal
        self.H = H
        self.W = W
        
        # 预生成所有射线
        self.all_rays_o = []
        self.all_rays_d = []
        self.all_rgb = []
        
        for i in range(len(images)):
            rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(poses[i]))
            self.all_rays_o.append(rays_o.reshape(-1, 3))
            self.all_rays_d.append(rays_d.reshape(-1, 3))
            self.all_rgb.append(torch.Tensor(images[i]).reshape(-1, 3))
        
        self.all_rays_o = torch.cat(self.all_rays_o, 0)
        self.all_rays_d = torch.cat(self.all_rays_d, 0)
        self.all_rgb = torch.cat(self.all_rgb, 0)
    
    def __len__(self):
        return len(self.all_rgb)
    
    def __getitem__(self, idx):
        return {
            'rays_o': self.all_rays_o[idx],
            'rays_d': self.all_rays_d[idx],
            'rgb': self.all_rgb[idx]
        }


class NeRFTrainer:
    """
    NeRF训练器
    """
    def __init__(
        self,
        model,
        device='cuda',
        lr=5e-4,
        near=2.,
        far=6.,
        N_samples=64
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.near = near
        self.far = far
        self.N_samples = N_samples
        
        self.train_losses = []
    
    def train_step(self, batch):
        """单步训练"""
        rays_o = batch['rays_o'].to(self.device)
        rays_d = batch['rays_d'].to(self.device)
        target_rgb = batch['rgb'].to(self.device)
        
        # 渲染
        rgb_pred, depth_pred = render_rays(
            rays_o, rays_d, self.model,
            self.near, self.far, self.N_samples,
            device=self.device
        )
        
        # 计算损失
        loss = torch.mean((rgb_pred - target_rgb) ** 2)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for batch in pbar:
            loss = self.train_step(batch)
            total_loss += loss
            pbar.set_postfix({'loss': loss})
        
        return total_loss / len(dataloader)
    
    def train(self, dataloader, epochs, save_dir='checkpoints'):
        """完整训练流程"""
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            loss = self.train_epoch(dataloader)
            self.train_losses.append(loss)
            
            print(f"Loss: {loss:.4f}")
            
            # 保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f'nerf_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")


def load_blender_data(basedir, half_res=False, testskip=1):
    """
    加载Blender合成数据集
    """
    splits = ['train', 'val', 'test']
    metas = {}
    
    for s in splits:
        with open(os.path.join(basedir, f'transforms_{s}.json'), 'r') as fp:
            metas[s] = json.load(fp)
    
    all_imgs = []
    all_poses = []
    counts = [0]
    
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        
        for frame in meta['frames'][::testskip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(plt.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        
        imgs = (np.array(imgs) * 255).astype(np.uint8)
        poses = np.array(poses).astype(np.float32)
        
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
        imgs = np.array([np.array(Image.fromarray(img).resize((W, H))) for img in imgs])
    
    return imgs, poses, [counts, H, W, focal]


def demo_training():
    """
    训练演示
    """
    print("="*60)
    print("NeRF Training Demo")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 生成合成数据（示例）
    print("\nGenerating synthetic data...")
    N_images = 100
    H, W = 100, 100
    focal = 100.0
    
    # 相机绕圆环运动
    poses = []
    for i in range(N_images):
        angle = 2 * np.pi * i / N_images
        c2w = np.array([
            [np.cos(angle), -np.sin(angle), 0, 4*np.cos(angle)],
            [np.sin(angle), np.cos(angle), 0, 4*np.sin(angle)],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        poses.append(c2w)
    poses = np.array(poses)
    
    # 创建简单场景（彩色球体）
    images = []
    for pose in poses:
        # 这里应该是真实渲染，简化为随机颜色
        img = np.random.rand(H, W, 3).astype(np.float32)
        images.append(img)
    images = np.array(images)
    
    print(f"Dataset: {len(images)} images, {H}x{W}")
    
    # 创建数据集
    dataset = RayDataset(images, poses, focal, H, W)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=0)
    
    # 创建模型
    print("\nInitializing model...")
    model = NeRF()
    
    # 创建训练器
    trainer = NeRFTrainer(model, device=device)
    
    # 训练
    print("\nStarting training...")
    trainer.train(dataloader, epochs=100)
    
    # 保存最终模型
    torch.save(model.state_dict(), 'nerf_final.pt')
    print("\nTraining completed! Model saved to nerf_final.pt")


def demo_rendering():
    """
    渲染演示
    """
    print("="*60)
    print("NeRF Rendering Demo")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = NeRF().to(device)
    
    # 加载检查点
    if os.path.exists('nerf_final.pt'):
        model.load_state_dict(torch.load('nerf_final.pt'))
        print("Loaded trained model")
    else:
        print("No trained model found, using random initialization")
    
    # 渲染测试视角
    H, W = 100, 100
    focal = 100.0
    
    # 相机位姿
    c2w = np.array([
        [1, 0, 0, 4],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    print("\nRendering...")
    rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(c2w).to(device))
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    # 分批渲染
    rgb_list = []
    chunk = 1024
    with torch.no_grad():
        for i in tqdm(range(0, rays_o.shape[0], chunk)):
            rgb, _ = render_rays(
                rays_o[i:i+chunk], rays_d[i:i+chunk],
                model, near=2., far=6., N_samples=64,
                device=device
            )
            rgb_list.append(rgb.cpu())
    
    rgb = torch.cat(rgb_list, 0).reshape(H, W, 3).numpy()
    
    # 可视化
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.title('Rendered Image')
    plt.axis('off')
    plt.savefig('rendered_image.png', dpi=150)
    plt.show()
    
    print("Rendered image saved to rendered_image.png")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'render':
        demo_rendering()
    else:
        demo_training()
