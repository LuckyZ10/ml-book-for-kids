class SimpleRNN:
    """
    简单循环神经网络模型（支持多层）
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 num_layers: int = 1, cell_type: str = 'rnn'):
        """
        参数:
            input_size: 输入维度
            hidden_size: 每层隐藏层维度
            output_size: 输出维度
            num_layers: 层数
            cell_type: 'rnn', 'lstm', 或 'gru'
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        # 创建多层单元
        self.layers = []
        for i in range(num_layers):
            layer_input = input_size if i == 0 else hidden_size
            
            if cell_type == 'rnn':
                cell = RNNCell(layer_input, hidden_size, output_size if i == num_layers - 1 else hidden_size)
            elif cell_type == 'lstm':
                cell = LSTMCell(layer_input, hidden_size, output_size if i == num_layers - 1 else hidden_size)
            elif cell_type == 'gru':
                cell = GRUCell(layer_input, hidden_size, output_size if i == num_layers - 1 else hidden_size)
            else:
                raise ValueError(f"Unknown cell type: {cell_type}")
            
            self.layers.append(cell)
        
        # 如果不是最后一层单独输出，添加最终输出层
        if num_layers > 1 and cell_type == 'lstm':
            self.output_layer = None  # LSTM最后一层直接输出
        elif num_layers > 1:
            # 为RNN/GRU添加输出层
            scale = np.sqrt(1.0 / hidden_size)
            self.W_out = np.random.randn(output_size, hidden_size) * scale
            self.b_out = np.zeros(output_size)
        
        self.history = {'loss': []}
    
    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        前向传播整个序列
        
        参数:
            X: 输入序列 (seq_len, input_size)
            
        返回:
            H: 所有时刻的隐藏状态列表
            Y: 所有时刻的输出列表
        """
        seq_len = X.shape[0]
        
        # 初始化隐藏状态
        if self.cell_type == 'lstm':
            states = [(np.zeros(self.hidden_size), np.zeros(self.hidden_size)) 
                     for _ in range(self.num_layers)]
        else:
            states = [np.zeros(self.hidden_size) for _ in range(self.num_layers)]
        
        H = [[] for _ in range(self.num_layers)]  # 每层的隐藏状态
        Y = []
        
        for t in range(seq_len):
            x = X[t]
            
            # 逐层传播
            for layer_idx, cell in enumerate(self.layers):
                if self.cell_type == 'lstm':
                    h_prev, c_prev = states[layer_idx]
                    h, c, y = cell.forward(x, h_prev, c_prev)
                    states[layer_idx] = (h, c)
                    H[layer_idx].append(h)
                    x = h  # 下一层的输入
                else:
                    h_prev = states[layer_idx]
                    h, y = cell.forward(x, h_prev)
                    states[layer_idx] = h
                    H[layer_idx].append(h)
                    x = h
            
            # 最终输出
            if self.num_layers == 1:
                Y.append(y)
            else:
                # 多层时使用最后一层的隐藏状态
                final_h = states[-1][0] if self.cell_type == 'lstm' else states[-1]
                if hasattr(self, 'W_out'):
                    y = self.W_out @ final_h + self.b_out
                Y.append(y)
        
        return H, Y
    
    def backward(self, X: np.ndarray, dY: List[np.ndarray]) -> List[dict]:
        """
        BPTT反向传播
        
        参数:
            X: 输入序列
            dY: 每个时刻的输出梯度
            
        返回:
            每层的梯度列表
        """
        seq_len = len(dY)
        
        # 初始化梯度缓存
        all_grads = [{} for _ in range(self.num_layers)]
        
        # 初始化隐藏状态梯度
        if self.cell_type == 'lstm':
            dh_next = [np.zeros(self.hidden_size) for _ in range(self.num_layers)]
            dC_next = [np.zeros(self.hidden_size) for _ in range(self.num_layers)]
        else:
            dh_next = [np.zeros(self.hidden_size) for _ in range(self.num_layers)]
        
        # 时间反向传播
        for t in reversed(range(seq_len)):
            dy = dY[t]
            
            # 从最后一层开始
            for layer_idx in reversed(range(self.num_layers)):
                cell = self.layers[layer_idx]
                
                if self.cell_type == 'lstm':
                    dx, dh_prev, dC_prev, grads = cell.backward(dy, dh_next[layer_idx], dC_next[layer_idx])
                    dh_next[layer_idx] = dh_prev
                    dC_next[layer_idx] = dC_prev
                else:
                    dx, dh_prev, grads = cell.backward(dy, dh_next[layer_idx])
                    dh_next[layer_idx] = dh_prev
                
                # 累加梯度
                for key in grads:
                    if key not in all_grads[layer_idx]:
                        all_grads[layer_idx][key] = grads[key]
                    else:
                        all_grads[layer_idx][key] += grads[key]
                
                # 梯度传递到下一层（前一时间步的上层）
                dy = dx
        
        return all_grads
    
    def train_step(self, X: np.ndarray, targets: np.ndarray, lr: float = 0.01) -> float:
        """
        单步训练
        
        参数:
            X: 输入序列 (seq_len, input_size)
            targets: 目标输出 (seq_len, output_size)
            lr: 学习率
            
        返回:
            loss: 损失值
        """
        # 前向传播
        H, Y = self.forward(X)
        
        # 计算损失和梯度
        loss = 0
        dY = []
        
        for t in range(len(Y)):
            # Softmax交叉熵
            y_pred = softmax(Y[t])
            loss += -np.sum(targets[t] * np.log(y_pred + 1e-8))
            
            # 输出梯度
            dy = y_pred - targets[t]
            dY.append(dy)
        
        loss /= len(Y)
        
        # 反向传播
        all_grads = self.backward(X, dY)
        
        # 更新参数
        for layer_idx, cell in enumerate(self.layers):
            params = cell.get_params()
            grads = all_grads[layer_idx]
            
            for key in params:
                # 梯度裁剪
                grad_clipped = clip_gradients(grads[key])
                params[key] -= lr * grad_clipped
            
            cell.set_params(params)
        
        return loss