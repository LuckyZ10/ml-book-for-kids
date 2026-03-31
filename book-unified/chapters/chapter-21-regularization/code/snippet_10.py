class RegularizedNN:
    def __init__(self, input_size, hidden_size, output_size,
                 regularizer=None, dropout_rate=0.0, use_batchnorm=False):
        # 初始化权重和偏置
        # 初始化正则化器
        # 初始化Dropout层
        # 初始化BatchNorm层
        pass
    
    def forward(self, X, training=True):
        # 前向传播
        # 应用Dropout（训练时）
        # 应用BatchNorm
        pass
    
    def backward(self, X, y, learning_rate):
        # 反向传播
        # 计算梯度
        # 添加正则化梯度
        # 更新参数
        pass
    
    def train_with_early_stopping(self, X_train, y_train, X_val, y_val,
                                   max_epochs, patience, ...):
        # 实现早停逻辑
        pass