"""
激活函数从零实现 - 完整代码
===========================

《机器学习与深度学习：从小学生到大师》
第十九章：激活函数——神经网络的"开关"

本文件包含所有主要激活函数的纯NumPy实现，以及：
1. 激活函数类封装
2. 可视化对比
3. MNIST演示
4. 梯度流动分析

作者：ML教材编写组
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Tuple, Optional
import pickle
import gzip

# 设置随机种子以保证可重复性
np.random.seed(42)

# =============================================================================
# 第一部分：激活函数基类和具体实现
# =============================================================================

class ActivationFunction:
    """
    激活函数基类
    
    所有激活函数都应该继承此类并实现forward和backward方法。
    """
    
    def __init__(self, name: str):
        self.name = name
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播：计算激活函数的输出"""
        raise NotImplementedError
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """反向传播：计算激活函数的导数"""
        raise NotImplementedError
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """使实例可调用"""
        return self.forward(x)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class Sigmoid(ActivationFunction):
    """
    Sigmoid激活函数
    
    数学定义: σ(x) = 1 / (1 + e^(-x))
    导数: σ'(x) = σ(x) * (1 - σ(x))
    
    历史: 1980年代最流行的激活函数，因梯度消失问题逐渐被ReLU取代
    """
    
    def __init__(self):
        super().__init__("Sigmoid")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        数值稳定实现：
        - 对于正数，直接计算
        - 对于负数，使用等价形式避免溢出
        """
        out = np.zeros_like(x, dtype=float)
        
        # 正数区间
        pos_mask = x >= 0
        out[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        
        # 负数区间（数值稳定）
        neg_mask = x < 0
        exp_x = np.exp(x[neg_mask])
        out[neg_mask] = exp_x / (1 + exp_x)
        
        return out
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        导数: σ'(x) = σ(x) * (1 - σ(x))
        
        证明:
        σ(x) = 1/(1+e^(-x))
        σ'(x) = e^(-x)/(1+e^(-x))^2
              = [1/(1+e^(-x))] * [e^(-x)/(1+e^(-x))]
              = σ(x) * [1 - 1/(1+e^(-x))]
              = σ(x) * (1 - σ(x))
        """
        s = self.forward(x)
        return s * (1 - s)


class Tanh(ActivationFunction):
    """
    双曲正切激活函数
    
    数学定义: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    等价形式: tanh(x) = 2σ(2x) - 1
    导数: tanh'(x) = 1 - tanh^2(x)
    
    特点: 零中心化输出，但仍有梯度消失问题
    """
    
    def __init__(self):
        super().__init__("Tanh")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """使用NumPy内置函数"""
        return np.tanh(x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        导数: tanh'(x) = 1 - tanh^2(x)
        
        证明:
        tanh(x) = sinh(x)/cosh(x) = (e^x - e^(-x))/(e^x + e^(-x))
        使用商法则:
        tanh'(x) = [(e^x+e^(-x))^2 - (e^x-e^(-x))^2] / (e^x+e^(-x))^2
                 = 1 - tanh^2(x)
        """
        t = np.tanh(x)
        return 1 - t ** 2


class ReLU(ActivationFunction):
    """
    修正线性单元 (Rectified Linear Unit)
    
    数学定义: f(x) = max(0, x)
    导数: f'(x) = 1 if x > 0 else 0
    
    历史: 2010年由Nair和Hinton提出，开启了深度学习时代
    特点: 计算高效，缓解梯度消失，但存在"死亡ReLU"问题
    """
    
    def __init__(self):
        super().__init__("ReLU")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """f(x) = max(0, x)"""
        return np.maximum(0, x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        f'(x) = 1 if x > 0 else 0
        在x=0处定义为0
        """
        return (x > 0).astype(float)


class LeakyReLU(ActivationFunction):
    """
    带泄漏的修正线性单元
    
    数学定义: f(x) = x if x >= 0 else αx
    导数: f'(x) = 1 if x > 0 else α
    
    提出: 2013年，Maas等人
    改进: 解决"死亡ReLU"问题，负区间保留小梯度
    """
    
    def __init__(self, alpha: float = 0.01):
        super().__init__(f"Leaky ReLU (α={alpha})")
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """f(x) = x if x >= 0 else αx"""
        return np.where(x >= 0, x, self.alpha * x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """f'(x) = 1 if x > 0 else α"""
        return np.where(x > 0, 1.0, self.alpha)


class PReLU(ActivationFunction):
    """
    参数化修正线性单元 (Parametric ReLU)
    
    数学定义: f(x) = x if x >= 0 else αx
    其中α是可学习的参数（不是超参数）
    
    提出: 2015年，Kaiming He等人
    优势: 自动学习最优的负斜率
    """
    
    def __init__(self, alpha: float = 0.25):
        super().__init__(f"PReLU (learnable α)")
        self.alpha = alpha  # 初始值，会在训练中更新
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """f(x) = x if x >= 0 else αx"""
        return np.where(x >= 0, x, self.alpha * x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """关于x的导数"""
        return np.where(x >= 0, 1.0, self.alpha)
    
    def backward_alpha(self, x: np.ndarray, grad_output: np.ndarray) -> float:
        """
        计算关于α的梯度
        
        用于反向传播时更新α参数
        """
        return np.sum(grad_output * np.minimum(x, 0))


class ELU(ActivationFunction):
    """
    指数线性单元 (Exponential Linear Unit)
    
    数学定义: 
        f(x) = x if x > 0
        f(x) = α(e^x - 1) if x <= 0
    
    导数:
        f'(x) = 1 if x > 0
        f'(x) = αe^x if x <= 0
    
    提出: 2015年，Clevert, Unterthiner, Hochreiter
    特点: 平滑负区间，零均值输出，软饱和
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__(f"ELU (α={alpha})")
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """ELU前向传播"""
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """ELU导数"""
        return np.where(x > 0, 1.0, self.alpha * np.exp(x))


class SELU(ActivationFunction):
    """
    自归一化指数线性单元 (Scaled ELU)
    
    数学定义:
        f(x) = λx if x > 0
        f(x) = λα(e^x - 1) if x <= 0
    
    其中 λ ≈ 1.0507, α ≈ 1.6733
    
    提出: 2017年，Klambauer等人
    特点: 自归一化特性，保持零均值和单位方差
    """
    
    # 论文给出的最优参数
    SCALE = 1.0507009873554804934193349852946  # λ
    SCALE_NEG = 1.6732632423543772848170429916717  # α
    
    def __init__(self):
        super().__init__("SELU")
        self.scale = self.SCALE
        self.scale_neg = self.SCALE_NEG
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """SELU前向传播"""
        return self.scale * np.where(
            x > 0, 
            x, 
            self.scale_neg * (np.exp(x) - 1)
        )
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """SELU导数"""
        return self.scale * np.where(
            x > 0, 
            1.0, 
            self.scale_neg * np.exp(x)
        )


class GELU(ActivationFunction):
    """
    高斯误差线性单元 (Gaussian Error Linear Unit)
    
    数学定义: GELU(x) = x * Φ(x)
    其中Φ(x)是标准正态分布的CDF
    
    近似公式:
        GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
    
    提出: 2016年，Hendrycks和Gimpel
    应用: Transformer架构(BERT, GPT系列)的标准选择
    """
    
    def __init__(self, approximate: bool = True):
        super().__init__("GELU")
        self.approximate = approximate
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        GELU前向传播
        
        使用tanh近似，避免误差函数计算
        """
        if self.approximate:
            # 论文给出的tanh近似
            return 0.5 * x * (1 + np.tanh(
                np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
            ))
        else:
            # 精确计算（需要scipy）
            try:
                from scipy.special import erf
                return 0.5 * x * (1 + erf(x / np.sqrt(2)))
            except ImportError:
                print("Warning: scipy not available, using approximation")
                return 0.5 * x * (1 + np.tanh(
                    np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
                ))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        GELU导数
        
        GELU'(x) = Φ(x) + x * φ(x)
        其中φ(x)是标准正态PDF
        """
        # CDF近似
        cdf = 0.5 * (1 + np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
        ))
        
        # PDF
        pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)
        
        return cdf + x * pdf


class Swish(ActivationFunction):
    """
    Swish激活函数（自门控激活）
    
    数学定义: Swish(x) = x * σ(βx) = x / (1 + e^(-βx))
    
    当β=1时: Swish(x) = x * σ(x)
    当β→∞时: Swish趋近于ReLU
    
    提出: 2017年，Google Brain团队（Ramachandran等）
    发现: 通过神经架构搜索自动发现
    特点: 平滑、非单调、自门控
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__(f"Swish (β={beta})")
        self.beta = beta
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Swish前向传播"""
        return x / (1 + np.exp(-self.beta * x))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        Swish导数
        
        Swish'(x) = σ(βx) + βx * σ(βx) * (1 - σ(βx))
                  = σ(βx) * (1 + βx * (1 - σ(βx)))
        """
        sigmoid_beta_x = 1 / (1 + np.exp(-self.beta * x))
        return sigmoid_beta_x * (1 + self.beta * x * (1 - sigmoid_beta_x))


class Softmax(ActivationFunction):
    """
    Softmax激活函数
    
    数学定义: softmax(x_i) = e^(x_i) / Σ_j e^(x_j)
    
    特点:
    - 将K维实数向量转换为概率分布
    - 输出之和为1
    - 常用于多分类任务的输出层
    
    与交叉熵损失配合使用时，梯度计算非常简洁
    """
    
    def __init__(self):
        super().__init__("Softmax")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Softmax前向传播（数值稳定版本）
        
        技巧: 减去最大值防止指数溢出
        """
        # 数值稳定：减去最大值
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        返回Softmax的对角导数（简化版本）
        
        完整的雅可比矩阵: J_ij = p_i(δ_ij - p_j)
        这里返回: ∂p_i/∂x_i = p_i(1 - p_i)
        """
        p = self.forward(x)
        return p * (1 - p)
    
    def backward_with_label(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        当Softmax与交叉熵损失配合时的梯度
        
        结果异常简洁: ∂L/∂x = p - y
        这是Softmax+CrossEntropy成为多分类标配的原因之一
        """
        p = self.forward(x)
        return p - y


class Mish(ActivationFunction):
    """
    Mish激活函数
    
    数学定义: Mish(x) = x * tanh(softplus(x))
    其中 softplus(x) = ln(1 + e^x)
    
    特点: 平滑、非单调、计算成本略高但性能优异
    """
    
    def __init__(self):
        super().__init__("Mish")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Mish前向传播"""
        return x * np.tanh(np.log(1 + np.exp(x)))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        Mish导数（使用数值近似）
        
        精确导数计算较复杂，这里使用自动微分思想
        """
        # softplus
        sp = np.log(1 + np.exp(x))
        tanh_sp = np.tanh(sp)
        
        # sigmoid
        sigmoid_x = 1 / (1 + np.exp(-x))
        
        # 近似导数
        delta = x * sigmoid_x * (1 - tanh_sp ** 2)
        return tanh_sp + delta


# =============================================================================
# 第二部分：可视化工具
# =============================================================================

def visualize_all_activations(save_path: str = None):
    """
    可视化所有激活函数及其导数
    
    Args:
        save_path: 图片保存路径
    """
    # 创建激活函数实例
    activations = [
        Sigmoid(),
        Tanh(),
        ReLU(),
        LeakyReLU(alpha=0.1),
        PReLU(alpha=0.25),
        ELU(alpha=1.0),
        SELU(),
        GELU(),
        Swish(beta=1.0),
    ]
    
    # 生成输入数据
    x = np.linspace(-5, 5, 1000)
    
    # 创建图形
    n_activations = len(activations)
    n_cols = 3
    n_rows = (n_activations + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_activations > 1 else [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_activations))
    
    for idx, act in enumerate(activations):
        ax = axes[idx]
        
        y = act.forward(x)
        dy = act.backward(x)
        
        ax.plot(x, y, linewidth=2.5, label=f'{act.name}(x)', color=colors[idx])
        ax.plot(x, dy, '--', linewidth=2, label=f"{act.name}'(x)", 
                alpha=0.7, color='red')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_title(act.name, fontsize=12, fontweight='bold')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-2, 3)
        ax.set_xlabel('x')
        ax.set_ylabel('y / dy/dx')
    
    # 隐藏多余的子图
    for idx in range(n_activations, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Activation Functions and Their Derivatives', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def compare_gradient_flow(save_path: str = None):
    """
    比较不同激活函数的梯度流动特性
    
    Args:
        save_path: 图片保存路径
    """
    x = np.linspace(-5, 5, 1000)
    
    activations = [
        ('Sigmoid', Sigmoid()),
        ('Tanh', Tanh()),
        ('ReLU', ReLU()),
        ('Leaky ReLU', LeakyReLU(0.1)),
        ('ELU', ELU(1.0)),
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 绘制函数本身
    ax1 = axes[0]
    for name, act in activations:
        y = act.forward(x)
        ax1.plot(x, y, linewidth=2.5, label=name)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-2, 3)
    ax1.set_xlabel('Input (x)', fontsize=12)
    ax1.set_ylabel('Output f(x)', fontsize=12)
    ax1.set_title('Activation Functions', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 绘制导数
    ax2 = axes[1]
    for name, act in activations:
        dy = act.backward(x)
        ax2.plot(x, dy, linewidth=2.5, label=name)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_xlabel('Input (x)', fontsize=12)
    ax2.set_ylabel("Derivative f'(x)", fontsize=12)
    ax2.set_title('Derivatives (Gradient Flow)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()


def visualize_vanishing_gradient(save_path: str = None):
    """
    可视化梯度消失问题
    
    Args:
        save_path: 图片保存路径
    """
    n_layers = 30
    
    # 不同激活函数的最大导数
    max_gradients = {
        'Sigmoid (max=0.25)': 0.25,
        'Tanh (max=1.0)': 1.0,
        'ReLU (max=1.0)': 1.0,
        'Sigmoid (typical=0.1)': 0.1,  # 更现实的典型值
    }
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    for name, max_grad in max_gradients.items():
        gradients = [max_grad ** i for i in range(1, n_layers + 1)]
        ax.semilogy(range(1, n_layers + 1), gradients, 'o-', 
                   linewidth=2, markersize=4, label=name)
    
    ax.axhline(y=1e-7, color='red', linestyle='--', linewidth=2,
              label='Single precision limit (~1e-7)')
    ax.axhline(y=1e-308, color='orange', linestyle='--', linewidth=2,
              label='Double precision limit (~1e-308)')
    
    ax.set_xlabel('Number of Layers (Depth)', fontsize=12)
    ax.set_ylabel('Gradient Magnitude (log scale)', fontsize=12)
    ax.set_title('Vanishing Gradient Problem in Deep Networks', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(1, n_layers)
    
    # 添加注释
    ax.annotate('Sigmoid gradients vanish\nafter ~10 layers', 
                xy=(10, 0.25**10), xytext=(15, 1e-3),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    print("\n梯度消失分析（30层网络）：")
    print(f"  Sigmoid (max=0.25): 梯度 ≈ {0.25**30:.2e}（完全消失）")
    print(f"  Sigmoid (typical=0.1): 梯度 ≈ {0.1**30:.2e}（立即消失）")
    print(f"  Tanh (max=1.0):     梯度 = {1.0**30:.2e}（保持）")
    print(f"  ReLU (max=1.0):     梯度 = {1.0**30:.2e}（保持）")


def softmax_visualization(save_path: str = None):
    """
    可视化Softmax函数
    
    Args:
        save_path: 图片保存路径
    """
    np.random.seed(42)
    logits = np.array([2.0, 1.0, 0.1, -0.5, -1.0])
    
    softmax = Softmax()
    probs = softmax.forward(logits)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    classes = [f'Class {i+1}' for i in range(len(logits))]
    x_pos = np.arange(len(classes))
    
    # 原始logits
    bars1 = ax1.bar(x_pos, logits, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(classes)
    ax1.set_ylabel('Logit Value', fontsize=12)
    ax1.set_title('Raw Logits (Before Softmax)', fontsize=14, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, logits):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', 
                va='bottom' if height > 0 else 'top', fontsize=10)
    
    # Softmax后的概率
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(probs)))
    bars2 = ax2.bar(x_pos, probs, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(classes)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title(f'Softmax Probabilities (Sum = {probs.sum():.4f})', 
                 fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    
    for bar, prob in zip(bars2, probs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.show()
    
    print(f"\nSoftmax演示:")
    print(f"  输入 logits: {logits}")
    print(f"  输出 probs:  {probs.round(4)}")
    print(f"  概率之和:    {probs.sum():.6f}")


# =============================================================================
# 第三部分：简单的全连接神经网络（用于MNIST演示）
# =============================================================================

class FullyConnectedLayer:
    """
    全连接层（仿射层）
    
    前向: y = activation(x @ W + b)
    """
    
    def __init__(self, input_size: int, output_size: int, 
                 activation: ActivationFunction = None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation if activation else ReLU()
        
        # He初始化（适合ReLU族激活函数）
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros(output_size)
        
        # 缓存用于反向传播
        self.x = None
        self.z = None
        self.a = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.x = x
        self.z = x @ self.W + self.b  # 线性变换
        self.a = self.activation.forward(self.z)  # 激活
        return self.a
    
    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """反向传播并更新参数"""
        # 通过激活函数的梯度
        grad_z = grad_output * self.activation.backward(self.z)
        
        # 计算参数梯度
        grad_W = self.x.T @ grad_z
        grad_b = np.sum(grad_z, axis=0)
        
        # 计算输入梯度（传递给前一层）
        grad_x = grad_z @ self.W.T
        
        # 更新参数
        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b
        
        return grad_x


class SimpleNN:
    """
    简单的多层神经网络
    
    用于演示不同激活函数的效果
    """
    
    def __init__(self, layer_sizes: list, activation: ActivationFunction):
        """
        Args:
            layer_sizes: 各层大小，如[784, 256, 128, 10]
            activation: 隐藏层激活函数
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            is_output = (i == len(layer_sizes) - 2)
            if is_output:
                # 输出层使用Softmax
                act = Softmax()
            else:
                act = activation
            
            layer = FullyConnectedLayer(layer_sizes[i], layer_sizes[i+1], act)
            self.layers.append(layer)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, x: np.ndarray, y_true: np.ndarray, 
                 learning_rate: float) -> float:
        """
        反向传播
        
        返回损失值
        """
        # 前向传播
        output = self.forward(x)
        
        # 计算交叉熵损失
        loss = -np.mean(np.sum(y_true * np.log(output + 1e-8), axis=1))
        
        # 输出层梯度（Softmax + CrossEntropy）
        grad = (output - y_true) / x.shape[0]
        
        # 反向传播各层
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
        
        return loss
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测类别"""
        output = self.forward(x)
        return np.argmax(output, axis=1)
    
    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        predictions = self.predict(x)
        return np.mean(predictions == y)


def load_mnist(data_dir: str = './data') -> Tuple[np.ndarray, ...]:
    """
    加载MNIST数据集
    
    如果本地没有，会尝试下载
    """
    try:
        # 尝试从本地加载
        import os
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # 下载MNIST（简化版本）
        def download_mnist():
            """下载MNIST数据集"""
            from urllib import request
            import gzip
            
            base_url = "http://yann.lecun.com/exdb/mnist/"
            files = {
                'train_images': 'train-images-idx3-ubyte.gz',
                'train_labels': 'train-labels-idx1-ubyte.gz',
                'test_images': 't10k-images-idx3-ubyte.gz',
                'test_labels': 't10k-labels-idx1-ubyte.gz'
            }
            
            data = {}
            for key, filename in files.items():
                filepath = os.path.join(data_dir, filename)
                if not os.path.exists(filepath):
                    print(f"Downloading {filename}...")
                    request.urlretrieve(base_url + filename, filepath)
                
                with gzip.open(filepath, 'rb') as f:
                    if 'images' in key:
                        # 跳过16字节头部，读取图像数据
                        data[key] = np.frombuffer(f.read(), np.uint8, offset=16)
                    else:
                        # 跳过8字节头部，读取标签数据
                        data[key] = np.frombuffer(f.read(), np.uint8, offset=8)
            
            return data
        
        data = download_mnist()
        
        X_train = data['train_images'].reshape(-1, 784).astype(np.float32) / 255.0
        y_train = data['train_labels']
        X_test = data['test_images'].reshape(-1, 784).astype(np.float32) / 255.0
        y_test = data['test_labels']
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        print("Generating synthetic data for demonstration...")
        
        # 生成合成数据用于演示
        n_samples = 1000
        n_features = 784
        n_classes = 10
        
        X_train = np.random.randn(n_samples, n_features).astype(np.float32) * 0.1 + 0.5
        X_train = np.clip(X_train, 0, 1)
        y_train = np.random.randint(0, n_classes, n_samples)
        
        X_test = np.random.randn(n_samples // 5, n_features).astype(np.float32) * 0.1 + 0.5
        X_test = np.clip(X_test, 0, 1)
        y_test = np.random.randint(0, n_classes, n_samples // 5)
        
        return X_train, y_train, X_test, y_test


def one_hot_encode(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """One-hot编码"""
    return np.eye(num_classes)[y]


def train_and_compare_activations(epochs: int = 5, batch_size: int = 128):
    """
    训练并比较不同激活函数在MNIST上的性能
    
    Args:
        epochs: 训练轮数
        batch_size: 批次大小
    """
    print("=" * 70)
    print("激活函数对比实验 - MNIST手写数字识别")
    print("=" * 70)
    
    # 加载数据
    print("\n[1/4] 加载MNIST数据集...")
    X_train, y_train, X_test, y_test = load_mnist()
    y_train_onehot = one_hot_encode(y_train)
    
    print(f"  训练集: {X_train.shape}")
    print(f"  测试集: {X_test.shape}")
    
    # 定义要比较的激活函数
    activations = {
        'ReLU': ReLU(),
        'Leaky ReLU': LeakyReLU(alpha=0.01),
        'ELU': ELU(alpha=1.0),
        'Swish': Swish(beta=1.0),
    }
    
    results = {}
    
    # 网络结构
    layer_sizes = [784, 256, 128, 10]
    learning_rate = 0.1
    
    for name, activation in activations.items():
        print(f"\n[2/4] 训练网络使用激活函数: {name}")
        print("-" * 50)
        
        # 创建网络
        np.random.seed(42)  # 保证可重复性
        nn = SimpleNN(layer_sizes, activation)
        
        losses = []
        accuracies = []
        
        n_batches = len(X_train) // batch_size
        
        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train_onehot[indices]
            
            epoch_loss = 0
            
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                loss = nn.backward(X_batch, y_batch, learning_rate)
                epoch_loss += loss
            
            avg_loss = epoch_loss / n_batches
            train_acc = nn.accuracy(X_train, y_train)
            test_acc = nn.accuracy(X_test, y_test)
            
            losses.append(avg_loss)
            accuracies.append(test_acc)
            
            print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
                  f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
        
        results[name] = {
            'losses': losses,
            'accuracies': accuracies,
            'final_test_acc': test_acc
        }
    
    # 可视化结果
    print(f"\n[3/4] 绘制训练曲线...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, res in results.items():
        ax1.plot(res['losses'], linewidth=2, label=name, marker='o', markersize=4)
        ax2.plot(res['accuracies'], linewidth=2, label=name, marker='s', markersize=4)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test Accuracy', fontsize=12)
    ax2.set_title('Test Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mnist_activation_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: mnist_activation_comparison.png")
    plt.show()
    
    # 打印最终对比
    print(f"\n[4/4] 最终结果对比:")
    print("-" * 50)
    for name, res in sorted(results.items(), 
                           key=lambda x: x[1]['final_test_acc'], reverse=True):
        print(f"  {name:15s}: Test Accuracy = {res['final_test_acc']:.4f}")
    
    return results


# =============================================================================
# 第四部分：辅助工具和演示
# =============================================================================

def print_activation_summary():
    """打印激活函数总结表"""
    print("\n" + "=" * 90)
    print("激活函数特性总结")
    print("=" * 90)
    
    summary = """
    ┌─────────────────┬─────────────────────────────────┬──────────┬──────────┬─────────┐
    │ 激活函数        │ 公式                            │ 输出范围 │ 梯度消失 │ 死亡问题 │
    ├─────────────────┼─────────────────────────────────┼──────────┼──────────┼─────────┤
    │ Sigmoid         │ σ(x) = 1/(1+e^(-x))             │ (0, 1)   │ 严重     │ N/A     │
    │ Tanh            │ tanh(x) = (e^x-e^(-x))/(e^x+e^(-x))│ (-1,1)│ 中等     │ N/A     │
    │ ReLU            │ max(0, x)                       │ [0, +∞)  │ 无       │ 有      │
    │ Leaky ReLU      │ max(αx, x)                      │ (-∞, +∞) │ 无       │ 无      │
    │ PReLU           │ max(αx, x), α可学习              │ (-∞, +∞) │ 无       │ 无      │
    │ ELU             │ x if x>0 else α(e^x-1)          │ (-α, +∞) │ 无       │ 无      │
    │ SELU            │ λ*ELU(x), 自归一化               │ 约(-2,∞) │ 无       │ 无      │
    │ GELU            │ x*Φ(x), Φ为标准正态CDF          │ 约(-0.2,∞│ 无       │ 无      │
    │ Swish           │ x*σ(βx)                         │ 约(-0.3,∞│ 无       │ 无      │
    │ Softmax         │ e^(x_i)/Σe^(x_j)                │ (0, 1)   │ N/A      │ N/A     │
    └─────────────────┴─────────────────────────────────┴──────────┴──────────┴─────────┘
    """
    print(summary)


def run_all_demos():
    """运行所有演示"""
    print("=" * 70)
    print("激活函数从零实现 - 完整演示")
    print("《机器学习与深度学习：从小学生到大师》第十九章")
    print("=" * 70)
    
    # 1. 激活函数总结
    print_activation_summary()
    
    # 2. 可视化所有激活函数
    print("\n[演示 1/5] 可视化所有激活函数及其导数...")
    visualize_all_activations('all_activations.png')
    
    # 3. 比较梯度流动
    print("\n[演示 2/5] 比较不同激活函数的梯度流动...")
    compare_gradient_flow('gradient_flow.png')
    
    # 4. 梯度消失演示
    print("\n[演示 3/5] 演示梯度消失问题...")
    visualize_vanishing_gradient('vanishing_gradient.png')
    
    # 5. Softmax演示
    print("\n[演示 4/5] Softmax函数演示...")
    softmax_visualization('softmax_demo.png')
    
    # 6. MNIST对比实验（可选，因为训练较慢）
    print("\n[演示 5/5] MNIST激活函数对比实验...")
    print("  注意: 此演示需要下载MNIST数据集，可能需要几分钟")
    response = input("  是否运行MNIST实验? (y/n): ").lower().strip()
    
    if response == 'y':
        train_and_compare_activations(epochs=5)
    else:
        print("  跳过MNIST实验")
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    # 运行所有演示
    run_all_demos()
