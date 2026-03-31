"""
NeRF - 从零实现（PyTorch版本）
第39章：3D视觉与神经辐射场 - 代码实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def positional_encoding(x, L):
    """
    位置编码：将低维坐标映射到高频空间
    
    Args:
        x: 输入坐标 (..., 3)
        L: 编码层数
    
    Returns:
        编码后的特征 (..., 3 + 3*2*L)
    """
    encodings = [x]
    for l in range(L):
        encodings.append(torch.sin(2**l * np.pi * x))
        encodings.append(torch.cos(2**l * np.pi * x))
    return torch.cat(encodings, dim=-1)


class NeRF(nn.Module):
    """
    神经辐射场网络
    
    架构：
    - 位置编码 → MLP → 密度 + 特征
    - 特征 + 方向编码 → MLP → 颜色
    """
    def __init__(
        self,
        D: int = 8,              # MLP层数
        W: int = 256,            # 隐藏层维度
        input_ch: int = 63,      # 位置编码后维度 (3 + 3*2*10)
        input_ch_views: int = 27, # 方向编码后维度 (3 + 3*2*4)
        output_ch: int = 4,      # RGB + 密度
        skips: list = [4],       # 跳跃连接层
        use_viewdirs: bool = True
    ):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        # 第一部分：位置 → 密度 + 特征
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + 
            [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_ch, W) 
             for i in range(D-1)]
        )
        
        # 输出层：密度
        self.alpha_linear = nn.Linear(W, 1)
        
        # 第二部分：特征 + 方向 → 颜色
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)
    
    def forward(self, x):
        """
        Args:
            x: 输入 [..., input_ch + input_ch_views]
        
        Returns:
            rgb: [..., 3]
            alpha: [..., 1]
        """
        if self.use_viewdirs:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        else:
            input_pts = x
        
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        
        # 密度输出
        alpha = self.alpha_linear(h)
        
        # 颜色输出
        if self.use_viewdirs:
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            rgb = self.rgb_linear(h)
            rgb = torch.sigmoid(rgb)  # 颜色范围[0,1]
        else:
            outputs = self.output_linear(h)
            rgb = torch.sigmoid(outputs[..., :3])
        
        return rgb, alpha


def get_rays(H, W, focal, c2w):
    """
    生成相机射线
    
    Args:
        H, W: 图像高宽
        focal: 焦距
        c2w: 相机到世界的变换矩阵 (4, 4)
    
    Returns:
        rays_o: 射线原点 (H, W, 3)
        rays_d: 射线方向 (H, W, 3)
    """
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W),
        torch.linspace(0, H-1, H),
        indexing='xy'
    )
    
    # 像素坐标转相机坐标
    dirs = torch.stack([
        (i - W*.5) / focal,
        -(j - H*.5) / focal,  # 注意y轴翻转
        -torch.ones_like(i)
    ], -1)
    
    # 射线方向转世界坐标
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    
    # 射线原点
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    
    return rays_o, rays_d


def sample_along_rays(rays_o, rays_d, near, far, N_samples, perturb=True):
    """
    沿射线采样3D点
    
    Args:
        rays_o: 射线原点 (N_rays, 3)
        rays_d: 射线方向 (N_rays, 3)
        near, far: 近平面和远平面
        N_samples: 采样点数
        perturb: 是否随机扰动采样点
    
    Returns:
        pts: 采样点 (N_rays, N_samples, 3)
        z_vals: 深度值 (N_rays, N_samples)
    """
    N_rays = rays_o.shape[0]
    
    # 均匀采样深度
    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals = near * (1. - t_vals) + far * t_vals
    z_vals = z_vals.expand([N_rays, N_samples])
    
    if perturb:
        # 在区间内随机扰动
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
    
    # 计算3D点
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
    
    return pts, z_vals


def raw2outputs(raw, z_vals, rays_d):
    """
    将NeRF原始输出转换为像素颜色
    
    Args:
        raw: NeRF输出 (N_rays, N_samples, 4) [rgb, density]
        z_vals: 深度值 (N_rays, N_samples)
        rays_d: 射线方向 (N_rays, 3)
    
    Returns:
        rgb_map: 渲染颜色 (N_rays, 3)
        depth_map: 渲染深度 (N_rays)
        acc_map: 累积不透明度 (N_rays)
        weights: 权重 (N_rays, N_samples)
    """
    # 分离rgb和密度
    rgb = torch.sigmoid(raw[..., :3])  # (N_rays, N_samples, 3)
    raw_density = raw[..., 3]  # (N_rays, N_samples)
    
    # 计算相邻采样点的距离
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([
        dists,
        torch.Tensor([1e10]).expand(dists[..., :1].shape)
    ], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    
    # 计算不透明度：α = 1 - exp(-σδ)
    alpha = 1. - torch.exp(-F.relu(raw_density) * dists)
    
    # 计算透射率：T = cumprod(1 - α)
    transmittance = torch.cumprod(
        torch.cat([
            torch.ones((alpha.shape[0], 1)),
            1. - alpha + 1e-10
        ], -1),
        -1
    )[:, :-1]
    
    # 权重：w = T * α
    weights = alpha * transmittance
    
    # 渲染颜色：C = Σ w * c
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    
    # 渲染深度：D = Σ w * z
    depth_map = torch.sum(weights * z_vals, -1)
    
    # 累积不透明度
    acc_map = torch.sum(weights, -1)
    
    return rgb_map, depth_map, acc_map, weights


def render_rays(rays_o, rays_d, model, near, far, N_samples=64, device='cuda'):
    """
    渲染一批射线
    
    Args:
        rays_o: 射线原点 (N_rays, 3)
        rays_d: 射线方向 (N_rays, 3)
        model: NeRF模型
        near, far: 近平面和远平面
        N_samples: 采样点数
    
    Returns:
        rgb_map: 渲染颜色 (N_rays, 3)
        depth_map: 渲染深度 (N_rays)
    """
    # 采样3D点
    pts, z_vals = sample_along_rays(rays_o, rays_d, near, far, N_samples)
    
    # 准备输入
    N_rays, N_samples = pts.shape[:2]
    pts_flat = pts.reshape(-1, 3)
    
    # 位置编码
    pts_encoded = positional_encoding(pts_flat, L=10)
    
    # 方向编码（每个点共享射线方向）
    viewdirs = rays_d[:, None, :].expand(pts.shape)
    viewdirs_flat = viewdirs.reshape(-1, 3)
    viewdirs_encoded = positional_encoding(viewdirs_flat, L=4)
    
    # 合并输入
    inputs = torch.cat([pts_encoded, viewdirs_encoded], -1)
    
    # 分批前向传播（避免显存溢出）
    chunk = 1024 * 32
    outputs = []
    for i in range(0, inputs.shape[0], chunk):
        output = model(inputs[i:i+chunk])
        outputs.append(torch.cat([output[0], output[1]], dim=-1))
    outputs = torch.cat(outputs, 0)
    
    outputs = outputs.reshape(N_rays, N_samples, 4)
    
    # 体积渲染
    rgb_map, depth_map, _, _ = raw2outputs(outputs, z_vals, rays_d)
    
    return rgb_map, depth_map


# 示例用法
if __name__ == "__main__":
    print("NeRF Implementation")
    print("=" * 50)
    
    # 创建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = NeRF().to(device)
    
    # 测试位置编码
    x = torch.randn(100, 3).to(device)
    x_encoded = positional_encoding(x, L=10)
    print(f"Input shape: {x.shape}")
    print(f"Encoded shape: {x_encoded.shape}")  # 应该是 (100, 63)
    
    # 测试NeRF前向传播
    inputs = torch.randn(100, 90).to(device)  # 63 + 27
    rgb, alpha = model(inputs)
    print(f"RGB output shape: {rgb.shape}")  # (100, 3)
    print(f"Alpha output shape: {alpha.shape}")  # (100, 1)
    
    # 测试射线生成
    H, W = 100, 100
    focal = 100.0
    c2w = torch.eye(4).to(device)
    rays_o, rays_d = get_rays(H, W, focal, c2w)
    print(f"Rays origin shape: {rays_o.shape}")  # (100, 100, 3)
    print(f"Rays direction shape: {rays_d.shape}")  # (100, 100, 3)
    
    print("\nNeRF implementation ready!")
    print("To train, use train_nerf.py")
