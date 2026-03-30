class BidirectionalLSTM:
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        # 前向LSTM
        self.forward_lstm = SimpleRNN(..., cell_type='lstm')
        # 后向LSTM
        self.backward_lstm = SimpleRNN(..., cell_type='lstm')
        # 合并输出的全连接层
        self.W_merge = ...  # (output_size, 2*hidden_size)
    
    def forward(self, X):
        # X: (seq_len, input_size)
        # 1. 前向传播
        _, Y_f = self.forward_lstm.forward(X)
        # 2. 后向传播 (X翻转)
        _, Y_b = self.backward_lstm.forward(X[::-1])
        # 3. 合并输出
        Y_merged = [self.W_merge @ np.concatenate([y_f, y_b]) 
                   for y_f, y_b in zip(Y_f, Y_b[::-1])]
        return Y_merged