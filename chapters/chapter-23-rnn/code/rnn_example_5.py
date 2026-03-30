class TimeSeriesPredictor:
    """
    时间序列预测器
    """
    
    def __init__(self, input_size: int = 1, hidden_size: int = 32, output_size: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 使用GRU（适合时间序列）
        self.rnn = SimpleRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=1,
            cell_type='gru'
        )
        
        # 归一化参数
        self.mean = 0
        self.std = 1
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """标准化数据"""
        self.mean = np.mean(data)
        self.std = np.std(data) + 1e-8
        return (data - self.mean) / self.std
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """反标准化"""
        return data * self.std + self.mean
    
    def create_sequences(self, data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建序列数据
        
        参数:
            data: 时间序列数据
            seq_length: 序列长度
            
        返回:
            X: 输入序列 (num_samples, seq_length, input_size)
            y: 目标值 (num_samples, output_size)
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        
        return np.array(X), np.array(y)
    
    def train(self, data: np.ndarray, seq_length: int = 10, epochs: int = 100, 
              lr: float = 0.01, batch_size: int = 1):
        """
        训练时间序列预测器
        
        参数:
            data: 时间序列数据
            seq_length: 序列长度
            epochs: 训练轮数
            lr: 学习率
            batch_size: 批量大小
        """
        # 归一化
        data_norm = self.normalize(data)
        
        # 创建序列
        X, y = self.create_sequences(data_norm, seq_length)
        num_samples = len(X)
        
        print(f"开始训练时间序列预测器...")
        print(f"数据点: {len(data)}, 序列长度: {seq_length}")
        print(f"样本数: {num_samples}, 训练轮数: {epochs}")
        print("-" * 50)
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # 随机打乱
            indices = np.random.permutation(num_samples)
            
            for idx in indices:
                X_seq = X[idx].reshape(-1, self.input_size)
                target = y[idx].reshape(1, self.output_size)
                
                # 前向传播
                H, Y = self.rnn.forward(X_seq)
                
                # 计算MSE损失
                pred = Y[-1]
                loss = np.mean((pred - target.flatten()) ** 2)
                epoch_loss += loss
                
                # 计算梯度
                dY = [np.zeros(self.output_size) for _ in range(len(Y))]
                dY[-1] = 2 * (pred - target.flatten())
                
                # 反向传播
                all_grads = self.rnn.backward(X_seq, dY)
                
                # 更新参数
                for layer_idx, cell in enumerate(self.rnn.layers):
                    params = cell.get_params()
                    grads = all_grads[layer_idx]
                    
                    for key in params:
                        grad_clipped = clip_gradients(grads[key])
                        params[key] -= lr * grad_clipped
                    
                    cell.set_params(params)
            
            avg_loss = epoch_loss / num_samples
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        self.rnn.history['loss'] = losses
        return losses
    
    def predict(self, sequence: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        预测未来值
        
        参数:
            sequence: 输入序列
            steps: 预测步数
            
        返回:
            predictions: 预测值
        """
        # 归一化
        seq_norm = (sequence - self.mean) / self.std
        
        predictions = []
        current_seq = seq_norm.copy()
        
        for _ in range(steps):
            # 前向传播
            X = current_seq[-10:].reshape(-1, self.input_size) if len(current_seq) >= 10 else current_seq.reshape(-1, self.input_size)
            H, Y = self.rnn.forward(X)
            
            # 预测下一步
            pred_norm = Y[-1]
            predictions.append(pred_norm)
            
            # 更新序列
            current_seq = np.append(current_seq, pred_norm)
        
        # 反归一化
        predictions = np.array(predictions)
        return self.denormalize(predictions)


# 生成合成时间序列数据
def generate_synthetic_series(n_points: int = 500) -> np.ndarray:
    """生成合成时间序列（带趋势和季节性）"""
    t = np.arange(n_points)
    
    # 趋势
    trend = 0.01 * t
    
    # 季节性
    seasonal = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.sin(2 * np.pi * t / 20)
    
    # 噪声
    noise = np.random.randn(n_points) * 2
    
    return trend + seasonal + noise


def train_timeseries_example():
    """时间序列预测示例"""
    np.random.seed(42)
    
    # 生成数据
    data = generate_synthetic_series(n_points=400)
    
    # 划分训练/测试
    train_size = 300
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # 创建模型
    model = TimeSeriesPredictor(input_size=1, hidden_size=16, output_size=1)
    
    # 训练
    losses = model.train(train_data, seq_length=20, epochs=50, lr=0.01)
    
    # 预测
    print("\n预测测试:")
    test_seq = train_data[-20:]
    predictions = model.predict(test_seq, steps=len(test_data))
    
    # 计算测试误差
    mse = np.mean((predictions - test_data) ** 2)
    print(f"测试MSE: {mse:.4f}")
    
    return model, losses, predictions, test_data