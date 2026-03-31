class GRUCell:
    """
    GRU门控循环单元
    
    数学公式:
        z_t = σ(W_z @ [h_{t-1}, x_t] + b_z)
        r_t = σ(W_r @ [h_{t-1}, x_t] + b_r)
        h̃_t = tanh(W_h @ [r_t ⊙ h_{t-1}, x_t] + b_h)
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        concat_size = input_size + hidden_size
        scale = np.sqrt(1.0 / concat_size)
        
        # 更新门
        self.W_z = np.random.randn(hidden_size, concat_size) * scale
        self.b_z = np.zeros(hidden_size)
        
        # 重置门
        self.W_r = np.random.randn(hidden_size, concat_size) * scale
        self.b_r = np.zeros(hidden_size)
        
        # 候选隐藏状态
        self.W_h = np.random.randn(hidden_size, concat_size) * scale
        self.b_h = np.zeros(hidden_size)
        
        # 输出层
        scale_out = np.sqrt(1.0 / hidden_size)
        self.W_hy = np.random.randn(output_size, hidden_size) * scale_out
        self.b_y = np.zeros(output_size)
        
        self.cache = {}
        
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GRU前向传播一步"""
        concat = np.concatenate([h_prev, x])
        
        # 更新门
        z = sigmoid(self.W_z @ concat + self.b_z)
        
        # 重置门
        r = sigmoid(self.W_r @ concat + self.b_r)
        
        # 候选隐藏状态 (使用重置后的h_prev)
        concat_reset = np.concatenate([r * h_prev, x])
        h_tilde = tanh(self.W_h @ concat_reset + self.b_h)
        
        # 隐藏状态更新
        h = (1 - z) * h_prev + z * h_tilde
        
        # 输出
        y = self.W_hy @ h + self.b_y
        
        self.cache = {
            'x': x.copy(),
            'h_prev': h_prev.copy(),
            'concat': concat.copy(),
            'z': z.copy(),
            'r': r.copy(),
            'concat_reset': concat_reset.copy(),
            'h_tilde': h_tilde.copy(),
            'h': h.copy(),
            'y': y.copy()
        }
        
        return h, y
    
    def backward(self, dy: np.ndarray, dh_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """GRU反向传播一步"""
        x = self.cache['x']
        h_prev = self.cache['h_prev']
        concat = self.cache['concat']
        z = self.cache['z']
        r = self.cache['r']
        concat_reset = self.cache['concat_reset']
        h_tilde = self.cache['h_tilde']
        h = self.cache['h']
        
        # 输出层
        dW_hy = np.outer(dy, h)
        db_y = dy
        dh = self.W_hy.T @ dy + dh_next
        
        # 隐藏状态梯度
        dz = dh * (h_tilde - h_prev)
        dh_prev = dh * (1 - z)
        dh_tilde = dh * z
        
        # 候选隐藏状态梯度
        dzh = dh_tilde * tanh_derivative(self.W_h @ concat_reset + self.b_h)
        dW_h = np.outer(dzh, concat_reset)
        db_h = dzh
        
        # 重置门梯度
        d_concat_reset = self.W_h.T @ dzh
        dr = d_concat_reset[:self.hidden_size] * h_prev
        
        # 更新门梯度
        dzz = dz * sigmoid_derivative(self.W_z @ concat + self.b_z)
        dW_z = np.outer(dzz, concat)
        db_z = dzz
        
        # 重置门梯度（续）
        dzr = dr * sigmoid_derivative(self.W_r @ concat + self.b_r)
        dW_r = np.outer(dzr, concat)
        db_r = dzr
        
        # 传递到前一时刻
        d_concat = self.W_z.T @ dzz + self.W_r.T @ dzr
        d_concat[:self.hidden_size] += d_concat_reset[:self.hidden_size] * r
        dh_prev += d_concat[:self.hidden_size]
        dx = d_concat[self.hidden_size:]
        
        grads = {
            'W_z': dW_z, 'b_z': db_z,
            'W_r': dW_r, 'b_r': db_r,
            'W_h': dW_h, 'b_h': db_h,
            'W_hy': dW_hy, 'b_y': db_y
        }
        
        return dx, dh_prev, grads
    
    def get_params(self) -> dict:
        return {
            'W_z': self.W_z, 'b_z': self.b_z,
            'W_r': self.W_r, 'b_r': self.b_r,
            'W_h': self.W_h, 'b_h': self.b_h,
            'W_hy': self.W_hy, 'b_y': self.b_y
        }
    
    def set_params(self, params: dict):
        for key in params:
            setattr(self, key, params[key].copy())