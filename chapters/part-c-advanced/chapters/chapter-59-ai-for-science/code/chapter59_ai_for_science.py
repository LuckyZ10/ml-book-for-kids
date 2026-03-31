"""
第五十九章：人工智能 for 科学 (AI for Science) - 完整代码实现
============================================================================

本文件包含：
1. 分子动力学模拟 - AI加速分子模拟
2. 天气预报 - 神经网络气象预测
3. 物理方程求解 - 物理信息神经网络(PINN)
4. 材料发现 - 图神经网络预测分子性质
5. 蛋白质结构预测 - 简化版结构预测模型
6. 科学数据可视化与分析工具

作者: ML教材编写组
版本: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from typing import Tuple, List, Dict, Optional, Callable
import os
from tqdm import tqdm
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def set_seed(seed=42):
    """设置随机种子确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# =============================================================================
# 第一部分：分子动力学模拟 - AI加速的粒子系统
# =============================================================================

class LennardJonesPotential:
    """
    伦纳德-琼斯势 (Lennard-Jones Potential)
    
    描述两个中性原子或分子之间的相互作用势能:
    V(r) = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]
    
    其中:
    - epsilon: 势阱深度 (相互作用的强度)
    - sigma: 粒子直径 (相互作用的范围)
    - r: 粒子间距离
    
    这是分子动力学模拟中最常用的势能函数之一
    """
    def __init__(self, epsilon=1.0, sigma=1.0):
        self.epsilon = epsilon
        self.sigma = sigma
    
    def potential(self, r: np.ndarray) -> np.ndarray:
        """
        计算势能
        
        Args:
            r: 粒子间距离数组
            
        Returns:
            势能值数组
        """
        # 避免除零
        r = np.maximum(r, 0.1)
        sr6 = (self.sigma / r) ** 6
        sr12 = sr6 ** 2
        return 4 * self.epsilon * (sr12 - sr6)
    
    def force(self, r: np.ndarray) -> np.ndarray:
        """
        计算力 (势能的负梯度)
        
        F(r) = -dV/dr = 24 * epsilon * [2*(sigma/r)^12 - (sigma/r)^6] / r
        
        Args:
            r: 粒子间距离数组
            
        Returns:
            力的大小数组
        """
        r = np.maximum(r, 0.1)
        sr6 = (self.sigma / r) ** 6
        sr12 = sr6 ** 2
        return 24 * self.epsilon * (2 * sr12 - sr6) / r


class MolecularDynamicsSimulator:
    """
    分子动力学模拟器
    
    使用Velocity Verlet算法进行时间积分
    """
    def __init__(self, n_particles: int = 100, box_size: float = 10.0, 
                 dt: float = 0.001, temperature: float = 1.0):
        self.n_particles = n_particles
        self.box_size = box_size
        self.dt = dt
        self.temperature = temperature
        
        # 初始化位置和速度
        self.positions = np.random.rand(n_particles, 2) * box_size
        self.velocities = np.random.randn(n_particles, 2) * np.sqrt(temperature)
        
        # 势能模型
        self.potential = LennardJonesPotential(epsilon=1.0, sigma=1.0)
        
        # 模拟历史
        self.trajectory = [self.positions.copy()]
        self.energy_history = []
    
    def compute_forces(self) -> np.ndarray:
        """
        计算所有粒子间的力
        
        Returns:
            力矩阵，形状 (n_particles, 2)
        """
        forces = np.zeros((self.n_particles, 2))
        
        for i in range(self.n_particles):
            for j in range(i + 1, self.n_particles):
                # 计算相对位置和距离
                r_ij = self.positions[j] - self.positions[i]
                
                # 应用周期性边界条件
                r_ij -= self.box_size * np.round(r_ij / self.box_size)
                
                r = np.linalg.norm(r_ij)
                
                if r < 3.0:  # 截断距离
                    f_mag = self.potential.force(r)
                    f_vec = f_mag * r_ij / r
                    
                    forces[i] -= f_vec
                    forces[j] += f_vec
        
        return forces
    
    def step(self):
        """执行一步Velocity Verlet积分"""
        # 计算当前力
        forces = self.compute_forces()
        
        # 更新位置: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        self.positions += self.velocities * self.dt + 0.5 * forces * self.dt**2
        
        # 应用周期性边界条件
        self.positions %= self.box_size
        
        # 计算新力
        new_forces = self.compute_forces()
        
        # 更新速度: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        self.velocities += 0.5 * (forces + new_forces) * self.dt
        
        # 记录轨迹
        self.trajectory.append(self.positions.copy())
    
    def compute_energy(self) -> Tuple[float, float, float]:
        """
        计算系统能量
        
        Returns:
            (总能量, 动能, 势能)
        """
        # 动能 = 0.5 * sum(v^2)
        kinetic = 0.5 * np.sum(self.velocities**2)
        
        # 势能
        potential = 0.0
        for i in range(self.n_particles):
            for j in range(i + 1, self.n_particles):
                r_ij = self.positions[j] - self.positions[i]
                r_ij -= self.box_size * np.round(r_ij / self.box_size)
                r = np.linalg.norm(r_ij)
                if r < 3.0:
                    potential += self.potential.potential(r)
        
        total = kinetic + potential
        return total, kinetic, potential
    
    def run(self, n_steps: int):
        """
        运行模拟
        
        Args:
            n_steps: 模拟步数
        """
        print(f"运行分子动力学模拟，共{n_steps}步...")
        
        for step in tqdm(range(n_steps)):
            self.step()
            
            # 每100步记录能量
            if step % 100 == 0:
                total, ke, pe = self.compute_energy()
                self.energy_history.append({
                    'step': step,
                    'total': total,
                    'kinetic': ke,
                    'potential': pe
                })
        
        print("模拟完成!")
    
    def visualize_trajectory(self, save_path: str = 'md_trajectory.png'):
        """可视化粒子轨迹"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 显示几个时间点的快照
        time_points = [0, len(self.trajectory)//4, len(self.trajectory)//2, 
                      3*len(self.trajectory)//4, len(self.trajectory)-1]
        
        for idx, t in enumerate(time_points[:5]):
            ax = axes[idx // 3, idx % 3]
            pos = self.trajectory[t]
            ax.scatter(pos[:, 0], pos[:, 1], s=50, alpha=0.6)
            ax.set_xlim(0, self.box_size)
            ax.set_ylim(0, self.box_size)
            ax.set_aspect('equal')
            ax.set_title(f'Step {t}')
            ax.grid(True, alpha=0.3)
        
        # 能量曲线
        ax = axes[1, 2]
        steps = [e['step'] for e in self.energy_history]
        totals = [e['total'] for e in self.energy_history]
        kinetics = [e['kinetic'] for e in self.energy_history]
        potentials = [e['potential'] for e in self.energy_history]
        
        ax.plot(steps, totals, label='Total', linewidth=2)
        ax.plot(steps, kinetics, label='Kinetic', linewidth=2)
        ax.plot(steps, potentials, label='Potential', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"轨迹可视化已保存到: {save_path}")


class NeuralMDSurrogate(nn.Module):
    """
    神经网络替代模型 - 用于加速分子动力学
    
    使用图神经网络学习粒子间的相互作用
    """
    def __init__(self, n_particles: int, hidden_dim: int = 64):
        super().__init__()
        self.n_particles = n_particles
        
        # 编码器: 将位置映射到嵌入
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 消息传递层
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),  # 包含距离信息
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 更新层
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 解码器: 预测力
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, positions: torch.Tensor, box_size: float) -> torch.Tensor:
        """
        预测粒子受力
        
        Args:
            positions: 粒子位置，形状 (batch_size, n_particles, 2)
            box_size: 盒子大小
            
        Returns:
            预测的力，形状 (batch_size, n_particles, 2)
        """
        batch_size = positions.size(0)
        
        # 编码位置
        h = self.encoder(positions)  # (batch_size, n_particles, hidden_dim)
        
        # 计算粒子间距离和消息
        forces = torch.zeros_like(positions)
        
        for i in range(self.n_particles):
            messages = []
            for j in range(self.n_particles):
                if i != j:
                    # 相对位置
                    r_ij = positions[:, j] - positions[:, i]
                    r_ij -= box_size * torch.round(r_ij / box_size)
                    dist = torch.norm(r_ij, dim=-1, keepdim=True)
                    
                    # 消息
                    message_input = torch.cat([h[:, i], h[:, j], dist], dim=-1)
                    message = self.message_mlp(message_input)
                    messages.append(message)
            
            # 聚合消息
            if messages:
                aggregated = torch.stack(messages, dim=1).mean(dim=1)
                h_updated = self.update_mlp(torch.cat([h[:, i], aggregated], dim=-1))
                
                # 预测力
                force = self.decoder(h_updated)
                forces[:, i] = force
        
        return forces


# =============================================================================
# 第二部分：天气预报 - 神经网络气象预测
# =============================================================================

class WeatherDataset(Dataset):
    """
    合成天气数据集
    
    模拟简化的大气变量场:
    - 温度场
    - 气压场
    - 风速场
    """
    def __init__(self, n_samples: int = 1000, grid_size: int = 32, 
                 n_timesteps: int = 10):
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.n_timesteps = n_timesteps
        
        # 生成合成数据
        self.data = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> List[Dict]:
        """生成合成天气数据"""
        data = []
        
        for _ in range(self.n_samples):
            # 初始场
            temperature = self._generate_field(pattern='wave')
            pressure = self._generate_field(pattern='vortex')
            wind_u = self._generate_field(pattern='gradient')
            wind_v = self._generate_field(pattern='gradient')
            
            # 序列数据 (模拟时间演化)
            sequence = []
            for t in range(self.n_timesteps):
                # 简化的物理演化
                temperature = self._evolve_field(temperature, dt=0.1)
                pressure = self._evolve_field(pressure, dt=0.1)
                wind_u = self._evolve_field(wind_u, dt=0.1)
                wind_v = self._evolve_field(wind_v, dt=0.1)
                
                # 合并通道
                state = np.stack([temperature, pressure, wind_u, wind_v], axis=0)
                sequence.append(state)
            
            data.append(np.array(sequence))
        
        return data
    
    def _generate_field(self, pattern: str = 'wave') -> np.ndarray:
        """生成特定模式的场"""
        x = np.linspace(0, 2*np.pi, self.grid_size)
        y = np.linspace(0, 2*np.pi, self.grid_size)
        X, Y = np.meshgrid(x, y)
        
        if pattern == 'wave':
            return np.sin(X) * np.cos(Y) + 0.1 * np.random.randn(self.grid_size, self.grid_size)
        elif pattern == 'vortex':
            R = np.sqrt((X - np.pi)**2 + (Y - np.pi)**2)
            return np.exp(-R**2/2) + 0.1 * np.random.randn(self.grid_size, self.grid_size)
        else:  # gradient
            return X / (2*np.pi) + 0.1 * np.random.randn(self.grid_size, self.grid_size)
    
    def _evolve_field(self, field: np.ndarray, dt: float) -> np.ndarray:
        """简化的场演化"""
        # 扩散 + 平流
        from scipy.ndimage import gaussian_filter
        diffused = gaussian_filter(field, sigma=0.5)
        return diffused + dt * 0.1 * np.random.randn(*field.shape)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回 (输入序列, 目标序列)
        输入: 前n_timesteps-1步
        目标: 后n_timesteps-1步(预测下一步)
        """
        sequence = self.data[idx]
        x = torch.FloatTensor(sequence[:-1])  # (n_timesteps-1, 4, H, W)
        y = torch.FloatTensor(sequence[1:])   # (n_timesteps-1, 4, H, W)
        return x, y


class ConvLSTMCell(nn.Module):
    """
    卷积LSTM单元
    
    用于时空序列预测的循环神经网络
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # 合并输入和隐藏状态的卷积
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,  # i, f, g, o
            kernel_size,
            padding=self.padding
        )
    
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None, 
                c: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入，形状 (batch, input_dim, H, W)
            h: 隐藏状态
            c: 细胞状态
            
        Returns:
            (新的隐藏状态, 新的细胞状态)
        """
        batch_size, _, height, width = x.size()
        
        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device)
        if c is None:
            c = torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device)
        
        # 合并输入和隐藏状态
        combined = torch.cat([x, h], dim=1)
        conv_output = self.conv(combined)
        
        # 分割为四个门
        cc_i, cc_f, cc_g, cc_o = torch.split(conv_output, self.hidden_dim, dim=1)
        
        # 门控
        i = torch.sigmoid(cc_i)  # 输入门
        f = torch.sigmoid(cc_f)  # 遗忘门
        g = torch.tanh(cc_g)     # 候选状态
        o = torch.sigmoid(cc_o)  # 输出门
        
        # 更新细胞状态和隐藏状态
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class WeatherPredictionModel(nn.Module):
    """
    天气预报神经网络
    
    使用ConvLSTM进行时序预测
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, 
                 num_layers: int = 2, kernel_size: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU()
        )
        
        # ConvLSTM层
        self.convlstm_layers = nn.ModuleList([
            ConvLSTMCell(hidden_dim, hidden_dim, kernel_size)
            for _ in range(num_layers)
        ])
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, input_dim, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, future_steps: int = 1) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列，形状 (batch, seq_len, channels, H, W)
            future_steps: 预测未来步数
            
        Returns:
            预测序列，形状 (batch, future_steps, channels, H, W)
        """
        batch_size, seq_len, _, height, width = x.size()
        
        # 初始化隐藏状态
        h_states = [None] * self.num_layers
        c_states = [None] * self.num_layers
        
        # 编码输入序列
        for t in range(seq_len):
            encoded = self.encoder(x[:, t])
            
            for i, lstm in enumerate(self.convlstm_layers):
                h_states[i], c_states[i] = lstm(encoded, h_states[i], c_states[i])
                encoded = h_states[i]
        
        # 预测未来
        predictions = []
        current = x[:, -1]  # 从最后一步开始
        
        for _ in range(future_steps):
            encoded = self.encoder(current)
            
            for i, lstm in enumerate(self.convlstm_layers):
                h_states[i], c_states[i] = lstm(encoded, h_states[i], c_states[i])
                encoded = h_states[i]
            
            prediction = self.decoder(encoded)
            predictions.append(prediction)
            current = prediction
        
        return torch.stack(predictions, dim=1)
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor, optimizer: optim.Optimizer) -> float:
        """单步训练"""
        optimizer.zero_grad()
        
        pred = self.forward(x, future_steps=y.size(1))
        loss = F.mse_loss(pred, y)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()


class WeatherForecaster:
    """天气预报器"""
    def __init__(self, model: WeatherPredictionModel, device='cpu'):
        self.model = model.to(device)
        self.device = device
    
    def train(self, dataloader: DataLoader, epochs: int = 50):
        """训练模型"""
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        print(f"开始训练天气预报模型，共{epochs}轮...")
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            for x, y in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
                x, y = x.to(self.device), y.to(self.device)
                loss = self.model.train_step(x, y, optimizer)
                total_loss += loss
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
        
        print("训练完成!")
    
    def predict(self, initial_sequence: torch.Tensor, future_steps: int = 5) -> np.ndarray:
        """
        进行预测
        
        Args:
            initial_sequence: 初始观测序列
            future_steps: 预测未来步数
            
        Returns:
            预测结果
        """
        self.model.eval()
        
        with torch.no_grad():
            initial_sequence = initial_sequence.to(self.device)
            prediction = self.model(initial_sequence.unsqueeze(0), future_steps)
        
        return prediction.cpu().numpy()
    
    def visualize_prediction(self, initial: np.ndarray, prediction: np.ndarray, 
                            ground_truth: Optional[np.ndarray] = None,
                            save_path: str = 'weather_prediction.png'):
        """
        可视化预测结果
        
        Args:
            initial: 初始序列
            prediction: 预测序列
            ground_truth: 真实值(可选)
            save_path: 保存路径
        """
        n_input = initial.shape[0]
        n_pred = prediction.shape[1]
        
        fig, axes = plt.subplots(4, n_input + n_pred, figsize=(3*(n_input+n_pred), 12))
        
        variables = ['Temperature', 'Pressure', 'Wind U', 'Wind V']
        
        for var_idx in range(4):
            # 绘制初始序列
            for t in range(n_input):
                ax = axes[var_idx, t]
                im = ax.imshow(initial[t, var_idx], cmap='RdBu_r', vmin=-1, vmax=1)
                ax.set_title(f'Input T-{n_input-t}')
                ax.axis('off')
            
            # 绘制预测
            for t in range(n_pred):
                ax = axes[var_idx, n_input + t]
                im = ax.imshow(prediction[0, t, var_idx], cmap='RdBu_r', vmin=-1, vmax=1)
                ax.set_title(f'Pred T+{t+1}')
                ax.axis('off')
        
        # 添加总标题
        for i, var_name in enumerate(variables):
            axes[i, 0].set_ylabel(var_name, fontsize=12, fontweight='bold')
        
        plt.suptitle('Weather Prediction - Multi-variable Forecast', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"预测可视化已保存到: {save_path}")


# =============================================================================
# 第三部分：物理信息神经网络 (PINN)
# =============================================================================

class PINN(nn.Module):
    """
    物理信息神经网络 (Physics-Informed Neural Network)
    
    用于求解偏微分方程，通过将物理方程作为约束融入神经网络训练
    """
    def __init__(self, layers: List[int], activation: str = 'tanh'):
        super().__init__()
        
        self.layers = layers
        self.activation = self._get_activation(activation)
        
        # 构建网络
        self.network = self._build_network()
    
    def _get_activation(self, name: str) -> Callable:
        """获取激活函数"""
        activations = {
            'tanh': torch.tanh,
            'sin': torch.sin,
            'relu': F.relu,
            'swish': lambda x: x * torch.sigmoid(x)
        }
        return activations.get(name, torch.tanh)
    
    def _build_network(self) -> nn.Module:
        """构建神经网络"""
        modules = []
        for i in range(len(self.layers) - 1):
            modules.append(nn.Linear(self.layers[i], self.layers[i+1]))
            if i < len(self.layers) - 2:
                modules.append(nn.Tanh())
        return nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 空间坐标
            t: 时间坐标
            
        Returns:
            预测的解
        """
        inputs = torch.cat([x, t], dim=1)
        return self.network(inputs)


class BurgersEquationSolver:
    """
    Burgers方程求解器
    
    Burgers方程: du/dt + u * du/dx = nu * d^2u/dx^2
    
    这是一个非线性偏微分方程，常用于测试数值方法
    """
    def __init__(self, nu: float = 0.01, device='cpu'):
        self.nu = nu  # 粘性系数
        self.device = device
        
        # 创建PINN模型
        self.model = PINN(layers=[2, 64, 64, 64, 1], activation='tanh').to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def compute_derivatives(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        计算解的导数 (使用自动微分)
        
        Returns:
            (u, u_x, u_t, u_xx) - 解及其各阶导数
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u = self.model(x, t)
        
        # 一阶导数
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                                   create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
        
        # 二阶导数
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                    create_graph=True)[0]
        
        return u, u_x, u_t, u_xx
    
    def pde_residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        计算PDE残差
        
        残差 = u_t + u * u_x - nu * u_xx
        
        理想情况下，残差应该为0
        """
        u, u_x, u_t, u_xx = self.compute_derivatives(x, t)
        
        residual = u_t + u * u_x - self.nu * u_xx
        return residual
    
    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
        初始条件
        
        u(x, 0) = -sin(pi * x)
        """
        return -torch.sin(np.pi * x)
    
    def boundary_condition(self, t: torch.Tensor, x_val: float) -> torch.Tensor:
        """
        边界条件
        
        u(-1, t) = u(1, t) = 0
        """
        return torch.zeros_like(t)
    
    def train_model(self, n_epochs: int = 10000, n_collocation: int = 10000,
                   n_ic: int = 1000, n_bc: int = 1000):
        """
        训练PINN
        
        损失函数包括:
        1. PDE残差损失 (在配点处)
        2. 初始条件损失
        3. 边界条件损失
        """
        print(f"开始训练PINN，共{n_epochs}轮...")
        
        # 采样配点 (在求解域内)
        x_collocation = torch.rand(n_collocation, 1, device=self.device) * 2 - 1
        t_collocation = torch.rand(n_collocation, 1, device=self.device)
        
        # 采样初始条件点
        x_ic = torch.rand(n_ic, 1, device=self.device) * 2 - 1
        t_ic = torch.zeros(n_ic, 1, device=self.device)
        
        # 采样边界条件点
        t_bc = torch.rand(n_bc, 1, device=self.device)
        x_bc_left = -torch.ones(n_bc // 2, 1, device=self.device)
        x_bc_right = torch.ones(n_bc // 2, 1, device=self.device)
        
        loss_history = []
        
        for epoch in tqdm(range(n_epochs)):
            self.optimizer.zero_grad()
            
            # PDE残差损失
            residual = self.pde_residual(x_collocation, t_collocation)
            loss_pde = torch.mean(residual**2)
            
            # 初始条件损失
            u_ic_pred = self.model(x_ic, t_ic)
            u_ic_true = self.initial_condition(x_ic)
            loss_ic = torch.mean((u_ic_pred - u_ic_true)**2)
            
            # 边界条件损失
            u_bc_left = self.model(x_bc_left, t_bc[:len(x_bc_left)])
            u_bc_right = self.model(x_bc_right, t_bc[:len(x_bc_right)])
            loss_bc = torch.mean(u_bc_left**2) + torch.mean(u_bc_right**2)
            
            # 总损失
            loss = loss_pde + loss_ic + loss_bc
            
            loss.backward()
            self.optimizer.step()
            
            loss_history.append(loss.item())
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}, "
                      f"PDE = {loss_pde.item():.6f}, "
                      f"IC = {loss_ic.item():.6f}, "
                      f"BC = {loss_bc.item():.6f}")
        
        print("训练完成!")
        return loss_history
    
    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        预测解
        
        Args:
            x: 空间坐标数组
            t: 时间坐标数组
            
        Returns:
            预测解
        """
        self.model.eval()
        
        # 创建网格
        X, T = np.meshgrid(x, t)
        x_flat = torch.FloatTensor(X.flatten()).view(-1, 1).to(self.device)
        t_flat = torch.FloatTensor(T.flatten()).view(-1, 1).to(self.device)
        
        with torch.no_grad():
            u_pred = self.model(x_flat, t_flat).cpu().numpy()
        
        return u_pred.reshape(X.shape)
    
    def visualize_solution(self, save_path: str = 'burgers_solution.png'):
        """可视化解"""
        x = np.linspace(-1, 1, 256)
        t = np.linspace(0, 1, 100)
        
        u_pred = self.predict(x, t)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 热力图
        ax1 = axes[0]
        im1 = ax1.imshow(u_pred, extent=[-1, 1, 0, 1], aspect='auto', 
                         origin='lower', cmap='RdBu_r')
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.set_title('PINN Solution - Burgers Equation')
        plt.colorbar(im1, ax=ax1, label='u(x,t)')
        
        # 不同时刻的截面
        ax2 = axes[1]
        time_points = [0, 0.25, 0.5, 0.75, 1.0]
        for tp in time_points:
            t_idx = int(tp * (len(t) - 1))
            ax2.plot(x, u_pred[t_idx, :], label=f't = {tp}', linewidth=2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('u')
        ax2.set_title('Solution at Different Times')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3D表面图
        ax3 = axes[2]
        X, T = np.meshgrid(x, t)
        surf = ax3.contourf(X, T, u_pred, levels=20, cmap='RdBu_r')
        ax3.set_xlabel('x')
        ax3.set_ylabel('t')
        ax3.set_title('Contour Plot')
        plt.colorbar(surf, ax=ax3, label='u(x,t)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"解的可视化已保存到: {save_path}")


# =============================================================================
# 第四部分：材料发现 - 图神经网络预测分子性质
# =============================================================================

class MoleculeGraph:
    """
    分子图表示
    
    将分子表示为图:
    - 节点: 原子
    - 边: 化学键
    """
    def __init__(self, atoms: List[str], bonds: List[Tuple[int, int]], 
                 positions: Optional[np.ndarray] = None):
        self.atoms = atoms
        self.bonds = bonds
        self.positions = positions
        self.n_atoms = len(atoms)
        
        # 原子类型编码
        self.atom_types = ['H', 'C', 'N', 'O', 'F']
        self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_types)}
    
    def get_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取节点特征和邻接矩阵
        
        Returns:
            (节点特征, 边索引)
        """
        # 节点特征 (独热编码)
        node_features = torch.zeros(self.n_atoms, len(self.atom_types))
        for i, atom in enumerate(self.atoms):
            if atom in self.atom_encoder:
                node_features[i, self.atom_encoder[atom]] = 1
        
        # 边索引
        edge_index = []
        for i, j in self.bonds:
            edge_index.append([i, j])
            edge_index.append([j, i])  # 无向图
        
        edge_index = torch.LongTensor(edge_index).t()
        
        return node_features, edge_index


class GNNLayer(nn.Module):
    """
    图神经网络层 (Graph Convolutional Layer)
    
    使用消息传递机制更新节点表示
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.message = nn.Linear(in_dim, out_dim)
        self.update = nn.Linear(out_dim * 2, out_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征，形状 (n_nodes, in_dim)
            edge_index: 边索引，形状 (2, n_edges)
            
        Returns:
            更新后的节点特征，形状 (n_nodes, out_dim)
        """
        # 自变换
        h_self = self.linear(x)
        
        # 消息传递
        row, col = edge_index
        messages = self.message(x[col])  # 邻居消息
        
        # 聚合消息 (平均)
        h_agg = torch.zeros_like(h_self)
        for i in range(x.size(0)):
            mask = (row == i)
            if mask.sum() > 0:
                h_agg[i] = messages[mask].mean(dim=0)
        
        # 更新
        h = torch.cat([h_self, h_agg], dim=1)
        h = self.update(h)
        
        return F.relu(h)


class MolecularPropertyPredictor(nn.Module):
    """
    分子性质预测器
    
    使用GNN预测分子属性，如:
    - 溶解度
    - 毒性
    - 药物活性
    """
    def __init__(self, node_dim: int = 5, hidden_dim: int = 64, 
                 output_dim: int = 1, n_layers: int = 3):
        super().__init__()
        
        # 嵌入层
        self.embedding = nn.Linear(node_dim, hidden_dim)
        
        # GNN层
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        # 读出层 (Readout)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_features: 节点特征
            edge_index: 边索引
            batch: 批次索引(用于多个分子的批次处理)
            
        Returns:
            预测的分子性质
        """
        # 节点嵌入
        h = self.embedding(node_features)
        
        # GNN层
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, edge_index)
        
        # 图级别的读出 (全局平均池化)
        if batch is None:
            h_graph = h.mean(dim=0)
        else:
            # 处理批次
            h_graph = torch.zeros(batch.max() + 1, h.size(1), device=h.device)
            for i in range(batch.max() + 1):
                h_graph[i] = h[batch == i].mean(dim=0)
        
        # 预测
        output = self.readout(h_graph)
        
        return output


class MolecularDiscovery:
    """分子发现系统"""
    def __init__(self, device='cpu'):
        self.device = device
        self.model = MolecularPropertyPredictor(node_dim=5, hidden_dim=64, 
                                                output_dim=1).to(device)
    
    def create_sample_molecules(self) -> List[MoleculeGraph]:
        """创建示例分子"""
        molecules = []
        
        # 水分子 H2O
        h2o = MoleculeGraph(
            atoms=['O', 'H', 'H'],
            bonds=[(0, 1), (0, 2)],
            positions=np.array([[0, 0], [1, 0], [-0.5, 0.866]])
        )
        molecules.append(('Water (H2O)', h2o))
        
        # 甲烷 CH4
        methane = MoleculeGraph(
            atoms=['C', 'H', 'H', 'H', 'H'],
            bonds=[(0, 1), (0, 2), (0, 3), (0, 4)],
            positions=np.array([[0, 0, 0], [1, 1, 1], [1, -1, -1], 
                               [-1, 1, -1], [-1, -1, 1]])
        )
        molecules.append(('Methane (CH4)', methane))
        
        # 乙醇 C2H5OH
        ethanol = MoleculeGraph(
            atoms=['C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'],
            bonds=[(0, 1), (1, 2), (0, 3), (0, 4), (0, 5),
                   (1, 6), (1, 7), (2, 8)],
            positions=None
        )
        molecules.append(('Ethanol (C2H5OH)', ethanol))
        
        # 苯 C6H6
        benzene = MoleculeGraph(
            atoms=['C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'],
            bonds=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
                   (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)],
            positions=None
        )
        molecules.append(('Benzene (C6H6)', benzene))
        
        return molecules
    
    def predict_properties(self, molecules: List[Tuple[str, MoleculeGraph]]) -> Dict[str, float]:
        """
        预测分子性质
        
        Args:
            molecules: 分子列表
            
        Returns:
            预测结果
        """
        self.model.eval()
        predictions = {}
        
        with torch.no_grad():
            for name, mol in molecules:
                node_features, edge_index = mol.get_features()
                node_features = node_features.to(self.device)
                edge_index = edge_index.to(self.device)
                
                pred = self.model(node_features, edge_index)
                predictions[name] = pred.item()
        
        return predictions
    
    def visualize_molecules(self, molecules: List[Tuple[str, MoleculeGraph]], 
                           save_path: str = 'molecules.png'):
        """可视化分子结构"""
        n_mols = len(molecules)
        fig, axes = plt.subplots(1, n_mols, figsize=(5*n_mols, 5))
        
        if n_mols == 1:
            axes = [axes]
        
        atom_colors = {'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red', 'F': 'green'}
        
        for idx, (name, mol) in enumerate(molecules):
            ax = axes[idx]
            
            # 使用随机布局(如果没有位置信息)
            if mol.positions is None:
                np.random.seed(42)
                positions = np.random.randn(mol.n_atoms, 2) * 2
            else:
                positions = mol.positions[:, :2]
            
            # 绘制边
            for i, j in mol.bonds:
                ax.plot([positions[i, 0], positions[j, 0]], 
                       [positions[i, 1], positions[j, 1]], 
                       'k-', linewidth=2)
            
            # 绘制节点
            for i, atom in enumerate(mol.atoms):
                color = atom_colors.get(atom, 'purple')
                ax.scatter(positions[i, 0], positions[i, 1], 
                          c=color, s=500, edgecolors='black', zorder=5)
                ax.annotate(atom, (positions[i, 0], positions[i, 1]),
                           ha='center', va='center', fontsize=12, fontweight='bold')
            
            ax.set_title(name, fontsize=12)
            ax.set_aspect('equal')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"分子可视化已保存到: {save_path}")


# =============================================================================
# 第五部分：蛋白质结构预测 - 简化版
# =============================================================================

class SimplifiedProteinStructurePredictor:
    """
    简化版蛋白质结构预测器
    
    预测蛋白质骨架的二维折叠
    """
    def __init__(self, device='cpu'):
        self.device = device
        
        # 氨基酸编码
        self.amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                           'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        self.aa_encoder = {aa: i for i, aa in enumerate(self.amino_acids)}
    
    def predict_secondary_structure(self, sequence: str) -> Tuple[List[str], np.ndarray]:
        """
        预测二级结构 (简化规则)
        
        Args:
            sequence: 氨基酸序列
            
        Returns:
            (二级结构列表, 置信度分数)
        """
        ss_types = []
        confidence = []
        
        for i, aa in enumerate(sequence):
            # 简化的预测规则
            if aa in 'AILMVFWY':  # 疏水氨基酸，倾向于螺旋
                ss = 'H'  # Alpha螺旋
                conf = 0.7
            elif aa in 'DNST':  # 极性氨基酸，倾向于转角
                ss = 'T'  # 转角
                conf = 0.6
            else:  # 其他
                ss = 'C'  # 无规卷曲
                conf = 0.5
            
            ss_types.append(ss)
            confidence.append(conf)
        
        return ss_types, np.array(confidence)
    
    def fold_protein_2d(self, sequence: str, method: str = 'simple') -> np.ndarray:
        """
        2D蛋白质折叠 (简化模型)
        
        Args:
            sequence: 氨基酸序列
            method: 折叠方法
            
        Returns:
            2D坐标数组
        """
        n = len(sequence)
        coords = np.zeros((n, 2))
        
        if method == 'simple':
            # 简单的螺旋折叠
            for i in range(n):
                angle = i * 0.5
                radius = 0.3 * i
                coords[i] = [radius * np.cos(angle), radius * np.sin(angle)]
        
        elif method == 'hydrophobic':
            # 基于疏水性的折叠
            hydrophobic = [aa in 'AILMVFWY' for aa in sequence]
            
            current_pos = np.array([0.0, 0.0])
            direction = np.array([1.0, 0.0])
            
            for i in range(n):
                coords[i] = current_pos.copy()
                
                # 根据疏水性调整方向
                if hydrophobic[i]:
                    # 疏水氨基酸倾向于向内折叠
                    direction = self._rotate_vector(direction, np.pi / 4)
                else:
                    direction = self._rotate_vector(direction, -np.pi / 6)
                
                current_pos += direction * 0.5
        
        return coords
    
    def _rotate_vector(self, v: np.ndarray, angle: float) -> np.ndarray:
        """旋转向量"""
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        return rotation_matrix @ v
    
    def visualize_protein(self, sequence: str, coords: np.ndarray,
                         save_path: str = 'protein_structure.png'):
        """
        可视化蛋白质结构
        
        Args:
            sequence: 氨基酸序列
            coords: 2D坐标
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 结构可视化
        ax1 = axes[0]
        
        # 绘制骨架
        ax1.plot(coords[:, 0], coords[:, 1], 'k-', linewidth=2, alpha=0.5)
        
        # 颜色编码疏水性
        hydrophobic = [aa in 'AILMVFWY' for aa in sequence]
        colors = ['red' if h else 'blue' for h in hydrophobic]
        
        scatter = ax1.scatter(coords[:, 0], coords[:, 1], 
                             c=colors, s=200, edgecolors='black', zorder=5)
        
        # 标注氨基酸
        for i, (coord, aa) in enumerate(zip(coords, sequence)):
            ax1.annotate(aa, coord, ha='center', va='center', 
                        fontsize=8, fontweight='bold', color='white')
        
        ax1.set_title('Predicted 2D Protein Structure', fontsize=14, fontweight='bold')
        ax1.set_aspect('equal')
        ax1.axis('off')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='Hydrophobic'),
                          Patch(facecolor='blue', label='Hydrophilic')]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # 二级结构预测
        ax2 = axes[1]
        ss_types, confidence = self.predict_secondary_structure(sequence)
        
        ss_colors = {'H': 'red', 'T': 'blue', 'C': 'gray'}
        ss_names = {'H': 'Alpha Helix', 'T': 'Turn', 'C': 'Coil'}
        
        for i, (ss, conf) in enumerate(zip(ss_types, confidence)):
            ax2.barh(i, conf, color=ss_colors[ss], alpha=0.7)
            ax2.text(conf + 0.02, i, sequence[i], va='center', fontsize=10)
        
        ax2.set_yticks(range(len(sequence)))
        ax2.set_yticklabels([f'{i+1}' for i in range(len(sequence))])
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_ylabel('Amino Acid Position', fontsize=12)
        ax2.set_title('Secondary Structure Prediction', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1.2)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 添加图例
        legend_elements = [Patch(facecolor='red', label='Alpha Helix'),
                          Patch(facecolor='blue', label='Turn'),
                          Patch(facecolor='gray', label='Coil')]
        ax2.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"蛋白质结构可视化已保存到: {save_path}")


# =============================================================================
# 第六部分：主函数与演示
# =============================================================================

def demo_molecular_dynamics():
    """演示分子动力学模拟"""
    print("=" * 60)
    print("演示1: 分子动力学模拟")
    print("=" * 60)
    
    set_seed(42)
    
    # 创建模拟器
    simulator = MolecularDynamicsSimulator(
        n_particles=50,
        box_size=10.0,
        dt=0.001,
        temperature=1.0
    )
    
    # 运行模拟
    simulator.run(n_steps=10000)
    
    # 可视化
    os.makedirs('./outputs', exist_ok=True)
    simulator.visualize_trajectory('./outputs/md_trajectory.png')
    
    print("\n分子动力学演示完成!\n")


def demo_weather_prediction():
    """演示天气预报"""
    print("=" * 60)
    print("演示2: 天气预报")
    print("=" * 60)
    
    set_seed(42)
    
    # 创建数据集
    dataset = WeatherDataset(n_samples=500, grid_size=32, n_timesteps=10)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 创建模型
    model = WeatherPredictionModel(input_dim=4, hidden_dim=64, num_layers=2)
    forecaster = WeatherForecaster(model, device=device)
    
    # 训练
    forecaster.train(dataloader, epochs=20)
    
    # 测试预测
    test_data, test_target = dataset[0]
    prediction = forecaster.predict(test_data, future_steps=5)
    
    # 可视化
    forecaster.visualize_prediction(
        test_data.numpy(),
        prediction,
        test_target.numpy(),
        save_path='./outputs/weather_prediction.png'
    )
    
    print("\n天气预报演示完成!\n")


def demo_pinn():
    """演示PINN求解偏微分方程"""
    print("=" * 60)
    print("演示3: 物理信息神经网络 (PINN)")
    print("=" * 60)
    
    set_seed(42)
    
    # 创建求解器
    solver = BurgersEquationSolver(nu=0.01/np.pi, device=device)
    
    # 训练
    loss_history = solver.train_model(
        n_epochs=5000,
        n_collocation=10000,
        n_ic=1000,
        n_bc=1000
    )
    
    # 可视化
    solver.visualize_solution('./outputs/burgers_solution.png')
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.semilogy(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('PINN Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('./outputs/pinn_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nPINN演示完成!\n")


def demo_material_discovery():
    """演示材料发现"""
    print("=" * 60)
    print("演示4: 材料发现 - 分子性质预测")
    print("=" * 60)
    
    set_seed(42)
    
    # 创建发现系统
    discovery = MolecularDiscovery(device=device)
    
    # 创建示例分子
    molecules = discovery.create_sample_molecules()
    
    # 可视化
    discovery.visualize_molecules(molecules, './outputs/molecules.png')
    
    # 预测性质
    predictions = discovery.predict_properties(molecules)
    
    print("\n分子性质预测结果:")
    for name, pred in predictions.items():
        print(f"  {name}: {pred:.4f}")
    
    print("\n材料发现演示完成!\n")


def demo_protein_structure():
    """演示蛋白质结构预测"""
    print("=" * 60)
    print("演示5: 蛋白质结构预测")
    print("=" * 60)
    
    set_seed(42)
    
    # 创建预测器
    predictor = SimplifiedProteinStructurePredictor(device=device)
    
    # 示例蛋白质序列
    sequence = "MKTAYIAKQRQISFVKSHFSRQ"
    
    # 预测二级结构
    ss_types, confidence = predictor.predict_secondary_structure(sequence)
    print(f"\n蛋白质序列: {sequence}")
    print(f"二级结构预测: {''.join(ss_types)}")
    
    # 2D折叠
    coords = predictor.fold_protein_2d(sequence, method='hydrophobic')
    
    # 可视化
    predictor.visualize_protein(sequence, coords, './outputs/protein_structure.png')
    
    print("\n蛋白质结构预测演示完成!\n")


def compare_ai_for_science():
    """比较AI for Science的不同应用"""
    print("=" * 60)
    print("AI for Science 应用比较")
    print("=" * 60)
    
    comparison = """
    ┌─────────────────────┬─────────────────────────────────────────────────────┐
    │       应用领域       │                      核心方法                        │
    ├─────────────────────┼─────────────────────────────────────────────────────┤
    │    分子动力学        │ 神经网络替代传统势能计算，加速模拟                   │
    │                     │ 使用图神经网络学习粒子间相互作用                      │
    ├─────────────────────┼─────────────────────────────────────────────────────┤
    │    天气预报          │ ConvLSTM等时空序列模型预测气象场演化                  │
    │                     │ 融合物理约束的深度学习模型                            │
    ├─────────────────────┼─────────────────────────────────────────────────────┤
    │    物理方程求解      │ PINN: 将PDE作为损失函数约束融入神经网络训练           │
    │                     │ 自动微分计算高阶导数，无需离散化网格                  │
    ├─────────────────────┼─────────────────────────────────────────────────────┤
    │    材料发现          │ GNN预测分子/晶体性质，指导新材料设计                  │
    │                     │ 生成模型设计具有目标性质的分子                        │
    ├─────────────────────┼─────────────────────────────────────────────────────┤
    │    蛋白质结构        │ AlphaFold等基于注意力机制的结构预测                  │
    │                     │ 进化特征 + 几何深度学习                               │
    └─────────────────────┴─────────────────────────────────────────────────────┘
    
    AI for Science 的核心优势:
    1. 加速: 训练后的模型推理速度远超传统数值方法
    2. 融合: 将物理知识嵌入神经网络，提高泛化能力
    3. 发现: 在高维空间中搜索新材料、新药物
    4. 端到端: 直接从原始数据学习，减少人工特征工程
    
    挑战与未来方向:
    - 可解释性: 理解AI模型学到的"物理规律"
    - 泛化性: 确保模型在未见过的条件下依然有效
    - 不确定性量化: 评估预测的可信度
    - 大规模模拟: 扩展到更大规模的科学问题
    """
    print(comparison)


def main():
    """
    主函数 - 运行所有演示
    
    注意: 部分演示需要较长时间运行
    """
    # 创建输出目录
    os.makedirs('./outputs', exist_ok=True)
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         第五十九章: 人工智能 for 科学 - 完整演示               ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    本演示包含以下内容:
    1. 分子动力学模拟 - AI加速的粒子系统模拟
    2. 天气预报 - 神经网络气象预测
    3. 物理信息神经网络 (PINN) - 求解偏微分方程
    4. 材料发现 - 图神经网络预测分子性质
    5. 蛋白质结构预测 - 简化版结构预测
    
    每个演示会保存结果到 ./outputs/ 目录
    """)
    
    # 应用比较
    compare_ai_for_science()
    
    # 运行演示
    try:
        # 演示1: 分子动力学 (约1-2分钟)
        demo_molecular_dynamics()
        
        # 演示2: 天气预报 (约5-10分钟)
        demo_weather_prediction()
        
        # 演示3: PINN (约5-10分钟)
        demo_pinn()
        
        # 演示4: 材料发现 (快速)
        demo_material_discovery()
        
        # 演示5: 蛋白质结构 (快速)
        demo_protein_structure()
        
    except KeyboardInterrupt:
        print("\n用户中断演示")
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    所有演示已完成！                            ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    生成结果保存在 ./outputs/ 目录:
    - md_trajectory.png: 分子动力学轨迹
    - weather_prediction.png: 天气预报可视化
    - burgers_solution.png: PINN求解的Burgers方程
    - molecules.png: 分子结构可视化
    - protein_structure.png: 蛋白质结构预测
    
    思考题:
    1. 分子动力学中，神经网络替代模型如何平衡精度和速度?
    2. PINN方法相比传统有限元方法有什么优势和局限?
    3. 图神经网络为什么适合处理分子数据?
    4. AI方法如何改变传统科学研究范式?
    
    延伸阅读:
    - DeepMind AlphaFold: 蛋白质结构预测的突破
    - FourCastNet: 基于Transformer的天气预报
    - Neural ODE: 连续时间神经网络
    - SchNet/DimeNet: 分子性质的图神经网络模型
    """)


if __name__ == "__main__":
    main()
