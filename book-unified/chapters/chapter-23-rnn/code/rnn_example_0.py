"""
循环神经网络 - 纯NumPy实现
第二十三章代码实现
"""

import numpy as np
from typing import Tuple, Optional, List


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid激活函数"""
    # 数值稳定性处理
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Sigmoid导数"""
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh激活函数"""
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Tanh导数"""
    return 1 - np.tanh(x) ** 2


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax函数"""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def clip_gradients(grad: np.ndarray, max_norm: float = 5.0) -> np.ndarray:
    """梯度裁剪，防止梯度爆炸"""
    norm = np.sqrt(np.sum(grad ** 2))
    if norm > max_norm:
        grad = grad * (max_norm / norm)
    return grad


class RNNCell:
    """
    基础RNN单元（Elman网络）
    
    数学公式:
        h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
        y_t = W_hy @ h_t + b_y
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        参数:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重 - Xavier初始化
        scale_hh = np.sqrt(1.0 / hidden_size)
        scale_xh = np.sqrt(1.0 / input_size)
        scale_hy = np.sqrt(1.0 / hidden_size)
        
        # 隐藏层权重: W_hh (hidden_size, hidden_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_hh
        # 输入权重: W_xh (hidden_size, input_size)
        self.W_xh = np.random.randn(hidden_size, input_size) * scale_xh
        # 隐藏层偏置
        self.b_h = np.zeros(hidden_size)
        
        # 输出权重: W_hy (output_size, hidden_size)
        self.W_hy = np.random.randn(output_size, hidden_size) * scale_hy
        # 输出偏置
        self.b_y = np.zeros(output_size)
        
        # 存储中间结果用于反向传播
        self.cache = {}
        
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向传播一步
        
        参数:
            x: 当前输入 (input_size,)
            h_prev: 前一时刻隐藏状态 (hidden_size,)
            
        返回:
            h: 当前隐藏状态 (hidden_size,)
            y: 当前输出 (output_size,)
        """
        # 计算新的隐藏状态
        # h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
        z = self.W_hh @ h_prev + self.W_xh @ x + self.b_h
        h = tanh(z)
        
        # 计算输出
        # y_t = W_hy @ h_t + b_y
        y = self.W_hy @ h + self.b_y
        
        # 缓存用于反向传播
        self.cache = {
            'x': x.copy(),
            'h_prev': h_prev.copy(),
            'z': z.copy(),
            'h': h.copy(),
            'y': y.copy()
        }
        
        return h, y
    
    def backward(self, dy: np.ndarray, dh_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        反向传播一步
        
        参数:
            dy: 输出梯度 (output_size,)
            dh_next: 来自下一时刻的隐藏状态梯度 (hidden_size,)
            
        返回:
            dx: 输入梯度 (input_size,)
            dh_prev: 前一时刻隐藏状态梯度 (hidden_size,)
            grads: 参数字典
        """
        x = self.cache['x']
        h_prev = self.cache['h_prev']
        z = self.cache['z']
        h = self.cache['h']
        
        # 输出层梯度
        dW_hy = np.outer(dy, h)  # (output_size, hidden_size)
        db_y = dy
        dh_from_y = self.W_hy.T @ dy  # (hidden_size,)
        
        # 隐藏状态总梯度
        dh = dh_from_y + dh_next
        
        # Tanh导数
        dtanh = dh * tanh_derivative(z)
        
        # 参数梯度
        dW_hh = np.outer(dtanh, h_prev)
        dW_xh = np.outer(dtanh, x)
        db_h = dtanh
        
        # 传递梯度
        dh_prev = self.W_hh.T @ dtanh
        dx = self.W_xh.T @ dtanh
        
        grads = {
            'W_hh': dW_hh,
            'W_xh': dW_xh,
            'b_h': db_h,
            'W_hy': dW_hy,
            'b_y': db_y
        }
        
        return dx, dh_prev, grads
    
    def get_params(self) -> dict:
        """获取所有参数"""
        return {
            'W_hh': self.W_hh,
            'W_xh': self.W_xh,
            'b_h': self.b_h,
            'W_hy': self.W_hy,
            'b_y': self.b_y
        }
    
    def set_params(self, params: dict):
        """设置参数"""
        self.W_hh = params['W_hh'].copy()
        self.W_xh = params['W_xh'].copy()
        self.b_h = params['b_h'].copy()
        self.W_hy = params['W_hy'].copy()
        self.b_y = params['b_y'].copy()