class LSTMCell:
    """
    LSTM长短期记忆单元
    
    数学公式:
        f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)
        i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)
        C̃_t = tanh(W_C @ [h_{t-1}, x_t] + b_C)
        C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
        o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)
        h_t = o_t ⊙ tanh(C_t)
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 拼接后的维度
        concat_size = input_size + hidden_size
        
        # Xavier初始化
        scale = np.sqrt(1.0 / concat_size)
        
        # 遗忘门参数
        self.W_f = np.random.randn(hidden_size, concat_size) * scale
        self.b_f = np.zeros(hidden_size)
        
        # 输入门参数
        self.W_i = np.random.randn(hidden_size, concat_size) * scale
        self.b_i = np.zeros(hidden_size)
        
        # 候选细胞状态参数
        self.W_C = np.random.randn(hidden_size, concat_size) * scale
        self.b_C = np.zeros(hidden_size)
        
        # 输出门参数
        self.W_o = np.random.randn(hidden_size, concat_size) * scale
        self.b_o = np.zeros(hidden_size)
        
        # 输出层参数
        scale_out = np.sqrt(1.0 / hidden_size)
        self.W_hy = np.random.randn(output_size, hidden_size) * scale_out
        self.b_y = np.zeros(output_size)
        
        self.cache = {}
        
    def forward(self, x: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        LSTM前向传播一步
        
        参数:
            x: 当前输入 (input_size,)
            h_prev: 前一时刻隐藏状态 (hidden_size,)
            C_prev: 前一时刻细胞状态 (hidden_size,)
            
        返回:
            h: 当前隐藏状态 (hidden_size,)
            C: 当前细胞状态 (hidden_size,)
            y: 当前输出 (output_size,)
        """
        # 拼接输入和前一隐藏状态
        concat = np.concatenate([h_prev, x])  # (hidden_size + input_size,)
        
        # 遗忘门: f_t = σ(W_f @ concat + b_f)
        f = sigmoid(self.W_f @ concat + self.b_f)
        
        # 输入门: i_t = σ(W_i @ concat + b_i)
        i = sigmoid(self.W_i @ concat + self.b_i)
        
        # 候选细胞状态: C̃_t = tanh(W_C @ concat + b_C)
        C_tilde = tanh(self.W_C @ concat + self.b_C)
        
        # 细胞状态更新: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
        C = f * C_prev + i * C_tilde
        
        # 输出门: o_t = σ(W_o @ concat + b_o)
        o = sigmoid(self.W_o @ concat + self.b_o)
        
        # 隐藏状态: h_t = o_t ⊙ tanh(C_t)
        h = o * tanh(C)
        
        # 输出
        y = self.W_hy @ h + self.b_y
        
        # 缓存
        self.cache = {
            'x': x.copy(),
            'h_prev': h_prev.copy(),
            'C_prev': C_prev.copy(),
            'concat': concat.copy(),
            'f': f.copy(),
            'i': i.copy(),
            'C_tilde': C_tilde.copy(),
            'C': C.copy(),
            'o': o.copy(),
            'h': h.copy(),
            'y': y.copy()
        }
        
        return h, C, y
    
    def backward(self, dy: np.ndarray, dh_next: np.ndarray, dC_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        LSTM反向传播一步
        
        参数:
            dy: 输出梯度 (output_size,)
            dh_next: 来自下一时刻的隐藏状态梯度 (hidden_size,)
            dC_next: 来自下一时刻的细胞状态梯度 (hidden_size,)
            
        返回:
            dx: 输入梯度 (input_size,)
            dh_prev: 前一时刻隐藏状态梯度 (hidden_size,)
            dC_prev: 前一时刻细胞状态梯度 (hidden_size,)
            grads: 参数字典
        """
        # 读取缓存
        x = self.cache['x']
        h_prev = self.cache['h_prev']
        C_prev = self.cache['C_prev']
        concat = self.cache['concat']
        f = self.cache['f']
        i = self.cache['i']
        C_tilde = self.cache['C_tilde']
        C = self.cache['C']
        o = self.cache['o']
        h = self.cache['h']
        
        # 输出层梯度
        dW_hy = np.outer(dy, h)
        db_y = dy
        dh = self.W_hy.T @ dy + dh_next
        
        # 输出门梯度
        do = dh * tanh(C)
        dzo = do * sigmoid_derivative(self.W_o @ concat + self.b_o)
        dW_o = np.outer(dzo, concat)
        db_o = dzo
        
        # 细胞状态梯度
        dC = dh * o * tanh_derivative(C) + dC_next
        
        # 输入门梯度
        di = dC * C_tilde
        dzi = di * sigmoid_derivative(self.W_i @ concat + self.b_i)
        dW_i = np.outer(dzi, concat)
        db_i = dzi
        
        # 候选细胞状态梯度
        dC_tilde = dC * i
        dzC = dC_tilde * tanh_derivative(self.W_C @ concat + self.b_C)
        dW_C = np.outer(dzC, concat)
        db_C = dzC
        
        # 遗忘门梯度
        df = dC * C_prev
        dzf = df * sigmoid_derivative(self.W_f @ concat + self.b_f)
        dW_f = np.outer(dzf, concat)
        db_f = dzf
        
        # 传递到前一时刻
        d_concat = (self.W_f.T @ dzf + 
                   self.W_i.T @ dzi + 
                   self.W_C.T @ dzC + 
                   self.W_o.T @ dzo)
        
        dh_prev = d_concat[:self.hidden_size]
        dx = d_concat[self.hidden_size:]
        dC_prev = dC * f
        
        grads = {
            'W_f': dW_f, 'b_f': db_f,
            'W_i': dW_i, 'b_i': db_i,
            'W_C': dW_C, 'b_C': db_C,
            'W_o': dW_o, 'b_o': db_o,
            'W_hy': dW_hy, 'b_y': db_y
        }
        
        return dx, dh_prev, dC_prev, grads
    
    def get_params(self) -> dict:
        """获取参数"""
        return {
            'W_f': self.W_f, 'b_f': self.b_f,
            'W_i': self.W_i, 'b_i': self.b_i,
            'W_C': self.W_C, 'b_C': self.b_C,
            'W_o': self.W_o, 'b_o': self.b_o,
            'W_hy': self.W_hy, 'b_y': self.b_y
        }
    
    def set_params(self, params: dict):
        """设置参数"""
        for key in params:
            setattr(self, key, params[key].copy())