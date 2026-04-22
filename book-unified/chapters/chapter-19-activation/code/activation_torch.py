"""
激活函数PyTorch实现与对比实验
=============================

《机器学习与深度学习：从小学生到大师》
第十九章：激活函数——神经网络的"开关"

本文件包含：
1. PyTorch版本的激活函数实现
2. NumPy vs PyTorch性能对比
3. GPU加速演示
4. 自动求导验证

作者：ML教材编写组
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# =============================================================================
# 第一部分：PyTorch激活函数封装
# =============================================================================

class PyTorchActivation:
    """
    PyTorch激活函数统一接口
    
    演示如何用PyTorch的autograd自动计算梯度，
    并与NumPy手写实现的结果进行对比验证。
    """
    
    def __init__(self, name: str):
        self.name = name
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class SigmoidTorch(PyTorchActivation):
    """
    Sigmoid的PyTorch实现
    
    与NumPy版本对比：
    - PyTorch内置sigmoid经过高度优化
    - 支持GPU加速
    - 自动求导无需手动写backward
    """
    
    def __init__(self):
        super().__init__("Sigmoid-Torch")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)
    
    def forward_manual(self, x: torch.Tensor) -> torch.Tensor:
        """手动实现，用于教学演示"""
        return 1.0 / (1.0 + torch.exp(-x))


class TanhTorch(PyTorchActivation):
    """Tanh的PyTorch实现"""
    
    def __init__(self):
        super().__init__("Tanh-Torch")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class ReLUTorch(PyTorchActivation):
    """
    ReLU的PyTorch实现
    
    PyTorch中的nn.ReLU()是一个模块，
    F.relu()是一个函数式接口。
    """
    
    def __init__(self):
        super().__init__("ReLU-Torch")
        self.module = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 两种调用方式等价
        # return self.module(x)
        return F.relu(x)


class LeakyReLUTorch(PyTorchActivation):
    """LeakyReLU的PyTorch实现"""
    
    def __init__(self, negative_slope: float = 0.01):
        super().__init__(f"LeakyReLU-Torch({negative_slope})")
        self.negative_slope = negative_slope
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=self.negative_slope)


class ELUTorch(PyTorchActivation):
    """ELU的PyTorch实现"""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__(f"ELU-Torch({alpha})")
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x, alpha=self.alpha)


class GELUTorch(PyTorchActivation):
    """
    GELU的PyTorch实现
    
    PyTorch 1.12+ 开始原生支持gelu，
    之前的版本需要手动实现或使用transformers库。
    """
    
    def __init__(self, approximate: str = 'none'):
        super().__init__("GELU-Torch")
        self.approximate = approximate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)
    
    def forward_approx(self, x: torch.Tensor) -> torch.Tensor:
        """
        近似实现（用于教学理解）
        
        GELU ≈ 0.5 * x * (1 + tanh[√(2/π) * (x + 0.044715 * x³)])
        """
        return 0.5 * x * (1 + torch.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))


class SwishTorch(PyTorchActivation):
    """
    Swish的PyTorch实现
    
    Swish(x) = x * sigmoid(βx)
    当β=1时为SiLU (Sigmoid Linear Unit)
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__(f"Swish-Torch({beta})")
        self.beta = beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class MishTorch(PyTorchActivation):
    """Mish的PyTorch实现"""
    
    def __init__(self):
        super().__init__("Mish-Torch")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


# =============================================================================
# 第二部分：自动求导验证
# =============================================================================

def verify_gradients():
    """
    验证PyTorch自动求导与NumPy手动实现的梯度是否一致
    
    这是深度学习框架的核心优势之一：
    你只需要定义forward，backward自动完成！
    """
    
    print("=" * 60)
    print("梯度验证实验")
    print("=" * 60)
    
    # 创建测试输入
    x = torch.linspace(-3, 3, 100, requires_grad=True)
    
    activations = [
        ("Sigmoid", SigmoidTorch(), lambda x: 1/(1+np.exp(-x))),
        ("ReLU", ReLUTorch(), lambda x: np.maximum(0, x)),
        ("Tanh", TanhTorch(), lambda x: np.tanh(x)),
    ]
    
    for name, torch_act, np_func in activations:
        # PyTorch自动求导
        y = torch_act(x)
        y.sum().backward()  # 关键：一行代码完成反向传播！
        torch_grad = x.grad.clone().detach().numpy()
        x.grad.zero_()  # 清零梯度
        
        # NumPy数值微分（有限差分法）
        x_np = x.detach().numpy()
        dx = 1e-5
        np_grad = (np_func(x_np + dx) - np_func(x_np - dx)) / (2 * dx)
        
        # 比较
        max_diff = np.max(np.abs(torch_grad - np_grad))
        status = "✅ 通过" if max_diff < 1e-4 else "❌ 失败"
        print(f"{name:10s} 最大差异: {max_diff:.2e} {status}")
    
    print()


# =============================================================================
# 第三部分：性能对比实验
# =============================================================================

def benchmark_activations(sizes: list = [1000, 10000, 100000],
                         iterations: int = 100):
    """
    NumPy vs PyTorch CPU vs PyTorch GPU 性能对比
    
    这个实验告诉我们：
    1. 小规模数据：NumPy可能更快（ overhead 小）
    2. 大规模数据：PyTorch GPU优势明显
    3. GPU加速需要数据量足够大才能体现价值
    """
    
    print("=" * 60)
    print("性能对比实验")
    print("=" * 60)
    print(f"{'大小':>10} {'NumPy(ms)':>12} {'PyTorch-CPU(ms)':>16} {'PyTorch-GPU(ms)':>16} {'加速比':>8}")
    print("-" * 70)
    
    for size in sizes:
        # NumPy基准
        x_np = np.random.randn(size).astype(np.float32)
        start = time.time()
        for _ in range(iterations):
            y_np = 1 / (1 + np.exp(-x_np))  # Sigmoid
        np_time = (time.time() - start) * 1000 / iterations
        
        # PyTorch CPU
        x_cpu = torch.randn(size, dtype=torch.float32)
        start = time.time()
        for _ in range(iterations):
            y_cpu = torch.sigmoid(x_cpu)
        cpu_time = (time.time() - start) * 1000 / iterations
        
        # PyTorch GPU (如果可用)
        if torch.cuda.is_available():
            x_gpu = torch.randn(size, dtype=torch.float32, device='cuda')
            # 预热
            for _ in range(10):
                y_gpu = torch.sigmoid(x_gpu)
            torch.cuda.synchronize()
            
            start = time.time()
            for _ in range(iterations):
                y_gpu = torch.sigmoid(x_gpu)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start) * 1000 / iterations
        else:
            gpu_time = float('inf')
        
        speedup = np_time / gpu_time if gpu_time != float('inf') else "N/A"
        print(f"{size:>10} {np_time:>12.4f} {cpu_time:>16.4f} "
              f"{gpu_time:>16.4f} {str(speedup):>8}")
    
    print()


# =============================================================================
# 第四部分：神经网络中的实际使用
# =============================================================================

class ActivationComparisonNet(nn.Module):
    """
    对比不同激活函数的神经网络
    
    用相同的网络结构、不同的激活函数训练，
    观察收敛速度和最终性能的差异。
    """
    
    def __init__(self, input_size: int, hidden_size: int, 
                 output_size: int, activation: str = 'relu'):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # 选择激活函数
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, name: str):
        """根据名称获取激活函数"""
        activations = {
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'swish': SwishTorch(),
            'mish': MishTorch(),
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


def train_comparison():
    """
    训练对比实验
    
    在简单的回归任务上比较不同激活函数的表现。
    """
    
    print("=" * 60)
    print("训练对比实验")
    print("=" * 60)
    
    # 生成合成数据：y = sin(x) + 噪声
    n_samples = 1000
    X = torch.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = torch.sin(X) + 0.1 * torch.randn_like(X)
    
    # 划分训练/测试集
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    activations_to_test = ['sigmoid', 'tanh', 'relu', 'gelu', 'swish', 'mish']
    
    results = {}
    
    for act_name in activations_to_test:
        # 创建模型
        model = ActivationComparisonNet(1, 64, 1, act_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # 训练
        losses = []
        for epoch in range(500):
            optimizer.zero_grad()
            pred = model(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test).item()
        
        results[act_name] = {
            'final_train_loss': losses[-1],
            'test_loss': test_loss,
            'losses': losses
        }
        
        print(f"{act_name:12s} - 训练损失: {losses[-1]:.6f}, 测试损失: {test_loss:.6f}")
    
    print()
    
    # 绘制收敛曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for act_name, data in results.items():
        plt.plot(data['losses'], label=act_name, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('收敛速度对比')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    names = list(results.keys())
    test_losses = [results[n]['test_loss'] for n in names]
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    bars = plt.bar(names, test_losses, color=colors, alpha=0.7)
    plt.ylabel('Test Loss')
    plt.title('最终测试性能对比')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上标注数值
    for bar, val in zip(bars, test_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('activation_comparison.png', dpi=150, bbox_inches='tight')
    print("对比图已保存至 activation_comparison.png")
    plt.close()
    
    return results


# =============================================================================
# 第五部分：实用工具函数
# =============================================================================

def get_activation_summary():
    """
    打印所有激活函数的汇总信息
    
    方便查阅和教学使用。
    """
    
    summary = """
    ╔══════════════════════════════════════════════════════════════╗
    ║              激活函数快速参考表 (PyTorch版)                    ║
    ╠══════════════════════════════════════════════════════════════╣
    ║ 名称        │ 类/函数                    │ 主要用途           ║
    ╠═════════════╪════════════════════════════╪════════════════════╣
    ║ Sigmoid     │ nn.Sigmoid() / F.sigmoid   │ 二分类输出层       ║
    ║ Tanh        │ nn.Tanh() / F.tanh         │ RNN隐藏层          ║
    ║ ReLU        │ nn.ReLU() / F.relu         │ 通用隐藏层 (默认)   ║
    ║ LeakyReLU   │ nn.LeakyReLU(0.1)          │ 避免ReLU死亡神经元 ║
    ║ PReLU       │ nn.PReLU()                 │ 可学习负斜率       ║
    ║ ELU         │ nn.ELU()                   │ 平滑负区间         ║
    ║ SELU        │ nn.SELU()                  │ 自归一化网络       ║
    ║ GELU        │ nn.GELU()                  │ Transformer默认    ║
    ║ Swish/SiLU  │ nn.SiLU()                  │ 现代网络优选       ║
    ║ Mish        │ 需自定义 (见本代码)         │ 强性能替代ReLU     ║
    ║ Softmax     │ F.softmax(dim=...)         │ 多分类输出层       ║
    ╚══════════════════════════════════════════════════════════════╝
    
    使用建议：
    - 默认选择：ReLU 或 GELU
    - 避免梯度消失：Mish 或 Swish
    - 需要平滑输出：ELU 或 GELU
    - 二分类任务：Sigmoid (仅输出层)
    """
    
    print(summary)


def demonstrate_dead_relu():
    """
    演示ReLU的"死亡神经元"问题
    
    当学习率过大或参数初始化不当时，
    ReLU神经元可能永远输出0，导致梯度无法传播。
    """
    
    print("=" * 60)
    print("ReLU死亡神经元演示")
    print("=" * 60)
    
    # 模拟一个神经元
    x = torch.randn(1000, 1)
    weight = torch.tensor([[2.0]])
    bias = torch.tensor([-5.0])  # 大负偏置
    
    # 前向传播
    z = x @ weight + bias
    activated = F.relu(z)
    
    # 统计死亡比例
    dead_ratio = (activated == 0).float().mean().item()
    print(f"输入分布: mean={x.mean():.3f}, std={x.std():.3f}")
    print(f"线性输出: mean={z.mean():.3f}, std={z.std():.3f}")
    print(f"ReLU后死亡比例: {dead_ratio:.1%}")
    print("这意味着该神经元不再学习！")
    print("解决方案: 使用LeakyReLU、ELU或减小学习率")
    print()


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("激活函数PyTorch实现演示")
    print("=" * 60)
    print()
    
    # 1. 梯度验证
    verify_gradients()
    
    # 2. 性能对比
    benchmark_activations()
    
    # 3. 死亡神经元演示
    demonstrate_dead_relu()
    
    # 4. 快速参考表
    get_activation_summary()
    
    # 5. 训练对比（可选，较慢）
    print("开始训练对比实验（可能需要30秒）...")
    train_comparison()
    
    print()
    print("=" * 60)
    print("演示完成！")
    print("=" * 60)
