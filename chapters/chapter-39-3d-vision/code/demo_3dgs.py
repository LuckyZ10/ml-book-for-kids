"""
3D Gaussian Splatting - 概念演示
第39章：3D视觉与神经辐射场
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


class SimpleGaussian:
    """
    简化的3D高斯表示
    """
    def __init__(self, mean, cov, color, alpha):
        """
        Args:
            mean: (3,) 中心位置
            cov: (3, 3) 协方差矩阵
            color: (3,) RGB颜色
            alpha: 标量 不透明度
        """
        self.mean = mean
        self.cov = cov
        self.color = color
        self.alpha = alpha
    
    def evaluate(self, x):
        """
        评估高斯在某点的值
        
        Args:
            x: (N, 3) 查询点
        
        Returns:
            values: (N,) 高斯值
        """
        diff = x - self.mean
        # G(x) = exp(-0.5 * (x-μ)^T Σ^-1 (x-μ))
        exponent = -0.5 * torch.sum(diff @ torch.inverse(self.cov) * diff, dim=-1)
        return torch.exp(exponent)


def project_gaussian_to_2d(mean_3d, cov_3d, camera_matrix):
    """
    将3D高斯投影到2D图像平面
    
    这是3DGS的核心操作之一
    """
    # 简化的透视投影
    # 实际实现需要考虑相机内参、旋转等
    
    # 投影中心点
    mean_2d = camera_matrix @ torch.cat([mean_3d, torch.ones(1)])
    mean_2d = mean_2d[:2] / mean_2d[2]
    
    # 投影协方差（使用仿射近似）
    # Σ' = J W Σ W^T J^T
    # 这里简化为只考虑距离衰减
    distance = torch.norm(mean_3d)
    cov_2d = cov_3d[:2, :2] / (distance ** 2)
    
    return mean_2d, cov_2d


def splat_gaussians(gaussians, image_size=(100, 100), camera_pos=torch.tensor([0., 0., 5.])):
    """
    将3D高斯溅射到2D图像
    
    Args:
        gaussians: 高斯列表
        image_size: (H, W)
    
    Returns:
        image: (H, W, 3) 渲染图像
        alpha_map: (H, W) 不透明度图
    """
    H, W = image_size
    image = torch.zeros(H, W, 3)
    alpha_acc = torch.zeros(H, W)
    
    # 按深度排序（从近到远）
    depths = [torch.norm(g.mean - camera_pos) for g in gaussians]
    sorted_indices = sorted(range(len(gaussians)), key=lambda i: depths[i])
    
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )
    pixels = torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)
    
    for idx in sorted_indices:
        gaussian = gaussians[idx]
        
        # 简化的投影（假设高斯中心投影到图像中心附近）
        mean_2d = torch.tensor([0., 0.])  # 简化
        
        # 计算每个像素的高斯值
        # 这里简化处理，只考虑高斯中心附近的区域
        cov_2d = torch.eye(2) * 0.1  # 简化的协方差
        
        diff = pixels - mean_2d
        exponent = -0.5 * torch.sum(diff @ torch.inverse(cov_2d) * diff, dim=-1)
        gaussian_values = torch.exp(exponent).reshape(H, W)
        
        # 混合
        alpha = gaussian.alpha * gaussian_values
        image = image + alpha.unsqueeze(-1) * gaussian.color.reshape(1, 1, 3) * (1 - alpha_acc.unsqueeze(-1))
        alpha_acc = alpha_acc + alpha * (1 - alpha_acc)
    
    return image, alpha_acc


def demo_gaussian_splatting():
    """
    3DGS概念演示
    """
    print("="*60)
    print("3D Gaussian Splatting Demo")
    print("="*60)
    
    # 创建几个简单的3D高斯
    print("\nCreating 3D Gaussians...")
    gaussians = []
    
    # 红色球体（中心）
    gaussians.append(SimpleGaussian(
        mean=torch.tensor([0., 0., 0.]),
        cov=torch.eye(3) * 0.5,
        color=torch.tensor([1., 0., 0.]),
        alpha=0.8
    ))
    
    # 绿色球体（右侧）
    gaussians.append(SimpleGaussian(
        mean=torch.tensor([1., 0., 0.]),
        cov=torch.eye(3) * 0.3,
        color=torch.tensor([0., 1., 0.]),
        alpha=0.8
    ))
    
    # 蓝色球体（上方）
    gaussians.append(SimpleGaussian(
        mean=torch.tensor([0., 1., 0.]),
        cov=torch.eye(3) * 0.3,
        color=torch.tensor([0., 0., 1.]),
        alpha=0.8
    ))
    
    print(f"Created {len(gaussians)} Gaussians")
    
    # 渲染
    print("\nRendering...")
    image, alpha = splat_gaussians(gaussians, image_size=(100, 100))
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(image.numpy())
    axes[0].set_title('Rendered Image (RGB)')
    axes[0].axis('off')
    
    axes[1].imshow(alpha.numpy(), cmap='gray')
    axes[1].set_title('Accumulated Alpha')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('gaussian_splatting_demo.png', dpi=150)
    plt.show()
    
    print("\nRendered image saved to gaussian_splatting_demo.png")
    print("\nNote: This is a simplified demonstration.")
    print("Real 3DGS implementation includes:")
    print("- Proper camera projection")
    print("- Covariance matrix parameterization")
    print("- Tile-based rasterization")
    print("- Spherical harmonics for view-dependent color")
    print("- Adaptive density control")


def demo_spherical_harmonics():
    """
    球谐函数可视化
    """
    print("="*60)
    print("Spherical Harmonics Visualization")
    print("="*60)
    
    # 简单的球谐函数可视化
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    
    # l=0 (常数)
    Y00 = np.ones_like(theta) * 0.5
    
    # l=1 (偶极)
    Y1m1 = np.sin(theta) * np.sin(phi)  # y
    Y10 = np.cos(theta)  # z
    Y11 = np.sin(theta) * np.cos(phi)  # x
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].imshow(Y00, cmap='RdBu_r')
    axes[0, 0].set_title('l=0, m=0 (Constant)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(Y1m1, cmap='RdBu_r')
    axes[0, 1].set_title('l=1, m=-1 (y)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(Y10, cmap='RdBu_r')
    axes[1, 0].set_title('l=1, m=0 (z)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(Y11, cmap='RdBu_r')
    axes[1, 1].set_title('l=1, m=1 (x)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('spherical_harmonics.png', dpi=150)
    plt.show()
    
    print("\nSpherical harmonics visualization saved")
    print("Used for view-dependent color in 3DGS")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'sh':
        demo_spherical_harmonics()
    else:
        demo_gaussian_splatting()
