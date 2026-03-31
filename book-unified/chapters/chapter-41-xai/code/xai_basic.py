"""
可解释AI - 基础实现
第41章：可解释AI——理解黑盒模型
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def compute_saliency_map(model, image, target_class):
    """
    计算显著性图（基于梯度）
    
    Args:
        model: 神经网络模型
        image: 输入图像 (1, C, H, W)
        target_class: 目标类别索引
    
    Returns:
        saliency: 显著性图 (H, W)
    """
    model.eval()
    
    # 需要梯度
    image.requires_grad_()
    
    # 前向传播
    output = model(image)
    
    # 目标类别的分数
    target_score = output[0, target_class]
    
    # 反向传播
    model.zero_grad()
    target_score.backward()
    
    # 梯度的绝对值作为显著性
    saliency = torch.abs(image.grad).squeeze()
    
    # 如果是RGB图像，取最大值
    if saliency.dim() == 3:
        saliency = saliency.max(dim=0)[0]
    
    return saliency.detach().cpu().numpy()


def integrated_gradients(model, image, target_class, baseline=None, steps=50):
    """
    计算积分梯度
    
    Args:
        model: 神经网络模型
        image: 输入图像 (1, C, H, W)
        target_class: 目标类别索引
        baseline: 基线图像（默认全零）
        steps: 插值步数
    
    Returns:
        ig: 积分梯度归因 (C, H, W)
    """
    if baseline is None:
        baseline = torch.zeros_like(image)
    
    model.eval()
    
    # 生成插值路径
    alphas = torch.linspace(0, 1, steps).view(-1, 1, 1, 1)
    path = baseline + alphas * (image - baseline)
    
    # 计算路径上每一点的梯度
    gradients = []
    for i in range(steps):
        x = path[i:i+1]
        x.requires_grad_()
        
        output = model(x)
        score = output[0, target_class]
        
        model.zero_grad()
        score.backward()
        
        gradients.append(x.grad.detach())
    
    # 平均梯度
    avg_gradients = torch.stack(gradients).mean(dim=0)
    
    # 乘以 (x - x')
    ig = (image - baseline) * avg_gradients
    
    return ig.squeeze().cpu().numpy()


def occlusion_sensitivity(model, image, target_class, occluder_size=50, stride=25):
    """
    遮挡敏感度分析
    
    Args:
        model: 神经网络模型
        image: 输入图像 (1, C, H, W)
        target_class: 目标类别索引
        occluder_size: 遮挡块大小
        stride: 滑动步长
    
    Returns:
        sensitivity: 敏感度图 (H, W)
    """
    model.eval()
    
    _, C, H, W = image.shape
    
    # 原始预测
    with torch.no_grad():
        original_pred = model(image)[0, target_class].item()
    
    sensitivity = np.zeros((H, W))
    counts = np.zeros((H, W))
    
    with torch.no_grad():
        for i in range(0, H - occluder_size + 1, stride):
            for j in range(0, W - occluder_size + 1, stride):
                # 遮挡
                occluded = image.clone()
                occluded[:, :, i:i+occluder_size, j:j+occluder_size] = 0
                
                # 预测
                occluded_pred = model(occluded)[0, target_class].item()
                
                # 敏感度 = 原始预测 - 遮挡后预测
                diff = original_pred - occluded_pred
                
                # 更新敏感度图
                sensitivity[i:i+occluder_size, j:j+occluder_size] += diff
                counts[i:i+occluder_size, j:j+occluder_size] += 1
    
    # 平均
    sensitivity = sensitivity / (counts + 1e-8)
    
    return sensitivity


class SimpleOcclusionExplainer:
    """简单的遮挡解释器"""
    def __init__(self, model, occluder_size=50, stride=25):
        self.model = model
        self.occluder_size = occluder_size
        self.stride = stride
    
    def explain(self, image, target_class):
        """解释预测"""
        return occlusion_sensitivity(
            self.model, image, target_class,
            self.occluder_size, self.stride
        )


def visualize_explanation(image, explanation, title="Explanation", alpha=0.5):
    """
    可视化解释结果
    
    Args:
        image: 原始图像 (C, H, W) 或 (H, W, C)
        explanation: 解释图 (H, W)
        title: 标题
        alpha: 热力图透明度
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    if image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
        image = np.transpose(image, (1, 2, 0))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 解释热力图
    im = axes[1].imshow(explanation, cmap='hot')
    axes[1].set_title(title)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # 叠加
    axes[2].imshow(image)
    axes[2].imshow(explanation, cmap='hot', alpha=alpha)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("可解释AI - 基础实现")
    print("=" * 50)
    print("\n包含以下方法：")
    print("1. Saliency Map - 基于梯度的显著性")
    print("2. Integrated Gradients - 积分梯度")
    print("3. Occlusion Sensitivity - 遮挡敏感度")
    print("\n使用方法：")
    print("  from xai_basic import compute_saliency_map, integrated_gradients")
    print("  saliency = compute_saliency_map(model, image, target_class)")
    print("  ig = integrated_gradients(model, image, target_class)")
