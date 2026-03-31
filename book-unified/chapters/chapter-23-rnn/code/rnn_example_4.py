class CharLanguageModel:
    """
    字符级语言模型
    """
    
    def __init__(self, vocab_size: int, hidden_size: int = 128, cell_type: str = 'lstm'):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        
        # 创建RNN
        self.rnn = SimpleRNN(
            input_size=vocab_size,
            hidden_size=hidden_size,
            output_size=vocab_size,
            num_layers=2,
            cell_type=cell_type
        )
        
        # 字符映射
        self.char_to_idx = {}
        self.idx_to_char = {}
    
    def _one_hot(self, idx: int) -> np.ndarray:
        """One-hot编码"""
        vec = np.zeros(self.vocab_size)
        vec[idx] = 1.0
        return vec
    
    def prepare_data(self, text: str):
        """准备字符映射"""
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        if len(chars) != self.vocab_size:
            print(f"Warning: vocab_size mismatch. Using {len(chars)} instead of {self.vocab_size}")
            self.vocab_size = len(chars)
    
    def train(self, text: str, epochs: int = 100, seq_length: int = 25, lr: float = 0.01):
        """
        训练语言模型
        
        参数:
            text: 训练文本
            epochs: 训练轮数
            seq_length: 序列长度
            lr: 学习率
        """
        self.prepare_data(text)
        
        data_size = len(text)
        losses = []
        
        print(f"开始训练字符级语言模型...")
        print(f"数据大小: {data_size} 字符")
        print(f"词汇表大小: {self.vocab_size}")
        print(f"序列长度: {seq_length}")
        print(f"训练轮数: {epochs}")
        print("-" * 50)
        
        for epoch in range(epochs):
            # 随机选择起始位置
            start_idx = np.random.randint(0, data_size - seq_length - 1)
            
            # 准备输入和目标
            X_seq = []
            target_seq = []
            
            for i in range(seq_length):
                char = text[start_idx + i]
                next_char = text[start_idx + i + 1]
                
                X_seq.append(self._one_hot(self.char_to_idx[char]))
                target_seq.append(self._one_hot(self.char_to_idx[next_char]))
            
            X = np.array(X_seq)
            targets = np.array(target_seq)
            
            # 训练一步
            loss = self.rnn.train_step(X, targets, lr)
            losses.append(loss)
            
            if (epoch + 1) % 10 == 0:
                avg_loss = np.mean(losses[-10:])
                perplexity = np.exp(avg_loss)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
            
            # 每20轮生成一段文本
            if (epoch + 1) % 20 == 0:
                generated = self.generate(seed=text[start_idx:start_idx+10], length=50)
                print(f"生成文本: {generated}")
                print("-" * 50)
        
        self.rnn.history['loss'] = losses
        return losses
    
    def generate(self, seed: str, length: int = 100, temperature: float = 1.0) -> str:
        """
        生成文本
        
        参数:
            seed: 种子文本
            length: 生成长度
            temperature: 温度（控制随机性）
            
        返回:
            generated: 生成的文本
        """
        generated = seed
        
        # 初始化状态
        if self.cell_type == 'lstm':
            states = [(np.zeros(self.hidden_size), np.zeros(self.hidden_size)) 
                     for _ in range(self.rnn.num_layers)]
        else:
            states = [np.zeros(self.hidden_size) for _ in range(self.rnn.num_layers)]
        
        # 用种子初始化状态
        for char in seed[:-1]:
            if char not in self.char_to_idx:
                continue
            x = self._one_hot(self.char_to_idx[char])
            
            for layer_idx, cell in enumerate(self.rnn.layers):
                if self.cell_type == 'lstm':
                    h, c, _ = cell.forward(x, states[layer_idx][0], states[layer_idx][1])
                    states[layer_idx] = (h, c)
                else:
                    h, _ = cell.forward(x, states[layer_idx])
                    states[layer_idx] = h
                x = h
        
        # 当前字符
        current_char = seed[-1]
        
        # 生成
        for _ in range(length):
            if current_char not in self.char_to_idx:
                current_char = np.random.choice(list(self.char_to_idx.keys()))
            
            x = self._one_hot(self.char_to_idx[current_char])
            
            # 前向传播
            for layer_idx, cell in enumerate(self.rnn.layers):
                if self.cell_type == 'lstm':
                    h, c, y = cell.forward(x, states[layer_idx][0], states[layer_idx][1])
                    states[layer_idx] = (h, c)
                else:
                    h, y = cell.forward(x, states[layer_idx])
                    states[layer_idx] = h
                x = h
            
            # 应用温度
            y = y / temperature
            probs = softmax(y)
            
            # 采样
            idx = np.random.choice(self.vocab_size, p=probs)
            current_char = self.idx_to_char[idx]
            generated += current_char
        
        return generated


# 训练示例
def train_char_lm_example():
    """字符级语言模型训练示例"""
    
    # 示例文本（可以替换为任何文本）
    text = """
    机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习。
    深度学习是机器学习的一个子集，使用多层神经网络来学习数据的表示。
    神经网络受到生物神经系统的启发，由相互连接的节点组成。
    训练神经网络需要大量数据和计算资源。
    机器学习算法可以分为监督学习、无监督学习和强化学习。
    监督学习使用标记数据来训练模型。
    无监督学习在没有标签的情况下发现数据中的模式。
    强化学习通过与环境的交互来学习最优策略。
    """ * 10  # 重复以增加数据量
    
    # 创建并训练模型
    vocab_size = len(set(text))
    model = CharLanguageModel(
        vocab_size=vocab_size,
        hidden_size=64,
        cell_type='lstm'
    )
    
    losses = model.train(text, epochs=200, seq_length=30, lr=0.05)
    
    # 生成文本
    print("\n最终生成:")
    for temp in [0.5, 1.0, 1.5]:
        print(f"\n温度 = {temp}:")
        generated = model.generate("机器学习", length=100, temperature=temp)
        print(generated)
    
    return model, losses