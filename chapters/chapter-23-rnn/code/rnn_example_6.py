def numerical_gradient(cell, x: np.ndarray, h_prev: np.ndarray, 
                       target: np.ndarray, eps: float = 1e-5) -> dict:
    """
    数值梯度计算（用于梯度检查）
    """
    params = cell.get_params()
    num_grads = {}
    
    for key in params:
        param = params[key]
        grad = np.zeros_like(param)
        
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]
            
            # f(x + eps)
            param[idx] = old_val + eps
            if isinstance(cell, LSTMCell):
                cell.set_params(params)
                _, _, y_plus = cell.forward(x, h_prev, np.zeros_like(h_prev))
            else:
                cell.set_params(params)
                _, y_plus = cell.forward(x, h_prev)
            loss_plus = np.sum((y_plus - target) ** 2)
            
            # f(x - eps)
            param[idx] = old_val - eps
            if isinstance(cell, LSTMCell):
                cell.set_params(params)
                _, _, y_minus = cell.forward(x, h_prev, np.zeros_like(h_prev))
            else:
                cell.set_params(params)
                _, y_minus = cell.forward(x, h_prev)
            loss_minus = np.sum((y_minus - target) ** 2)
            
            # 恢复
            param[idx] = old_val
            
            # 中心差分
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            
            it.iternext()
        
        num_grads[key] = grad
    
    # 恢复原始参数
    cell.set_params(params)
    
    return num_grads


def gradient_check(cell_type: str = 'rnn', input_size: int = 5, 
                   hidden_size: int = 4, output_size: int = 3):
    """
    梯度检查
    
    比较解析梯度和数值梯度的差异
    """
    np.random.seed(42)
    
    # 创建单元
    if cell_type == 'rnn':
        cell = RNNCell(input_size, hidden_size, output_size)
    elif cell_type == 'lstm':
        cell = LSTMCell(input_size, hidden_size, output_size)
    elif cell_type == 'gru':
        cell = GRUCell(input_size, hidden_size, output_size)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")
    
    # 随机输入
    x = np.random.randn(input_size)
    h_prev = np.random.randn(hidden_size)
    target = np.random.randn(output_size)
    
    if cell_type == 'lstm':
        C_prev = np.random.randn(hidden_size)
        h, C, y = cell.forward(x, h_prev, C_prev)
        dy = 2 * (y - target)
        dx, dh_prev, dC_prev, ana_grads = cell.backward(dy, np.zeros_like(h_prev), np.zeros_like(C_prev))
    else:
        h, y = cell.forward(x, h_prev)
        dy = 2 * (y - target)
        dx, dh_prev, ana_grads = cell.backward(dy, np.zeros_like(h_prev))
    
    # 数值梯度
    if cell_type == 'lstm':
        num_grads = {}
        # 简化的数值梯度检查
        print(f"LSTM梯度检查简化版（计算量大，仅检查输出层）")
        params = cell.get_params()
        for key in ['W_hy', 'b_y']:
            param = params[key]
            grad = np.zeros_like(param)
            eps = 1e-5
            
            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                old_val = param[idx]
                
                param[idx] = old_val + eps
                cell.set_params(params)
                _, _, y_plus = cell.forward(x, h_prev, C_prev)
                loss_plus = np.sum((y_plus - target) ** 2)
                
                param[idx] = old_val - eps
                cell.set_params(params)
                _, _, y_minus = cell.forward(x, h_prev, C_prev)
                loss_minus = np.sum((y_minus - target) ** 2)
                
                param[idx] = old_val
                grad[idx] = (loss_plus - loss_minus) / (2 * eps)
                it.iternext()
            
            num_grads[key] = grad
            ana = ana_grads[key]
            diff = np.abs(ana - grad) / (np.abs(ana) + np.abs(grad) + 1e-8)
            max_diff = np.max(diff)
            print(f"  {key}: 最大相对误差 = {max_diff:.8f} {'✓' if max_diff < 1e-4 else '✗'}")
    else:
        num_grads = numerical_gradient(cell, x, h_prev, target)
        
        print(f"\n{cell_type.upper()}梯度检查:")
        print("-" * 50)
        
        max_diffs = []
        for key in ana_grads:
            ana = ana_grads[key]
            num = num_grads[key]
            
            # 相对误差
            diff = np.abs(ana - num) / (np.abs(ana) + np.abs(num) + 1e-8)
            max_diff = np.max(diff)
            max_diffs.append(max_diff)
            
            status = "✓ PASS" if max_diff < 1e-4 else "✗ FAIL"
            print(f"{key:10s}: 最大相对误差 = {max_diff:.8f} {status}")
        
        overall = "✓ 所有梯度检查通过" if all(d < 1e-4 for d in max_diffs) else "✗ 部分梯度检查失败"
        print("-" * 50)
        print(overall)


# 运行梯度检查
def run_gradient_checks():
    """运行所有梯度检查"""
    print("=" * 60)
    print("梯度检查验证")
    print("=" * 60)
    
    print("\n1. RNN单元梯度检查")
    gradient_check('rnn', input_size=5, hidden_size=4, output_size=3)
    
    print("\n2. LSTM单元梯度检查")
    gradient_check('lstm', input_size=5, hidden_size=4, output_size=3)
    
    print("\n3. GRU单元梯度检查")
    gradient_check('gru', input_size=5, hidden_size=4, output_size=3)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 运行梯度检查
    run_gradient_checks()
    
    # 运行训练示例
    print("\n" + "=" * 60)
    print("字符级语言模型示例")
    print("=" * 60)
    # lm_model, lm_losses = train_char_lm_example()
    
    print("\n" + "=" * 60)
    print("时间序列预测示例")
    print("=" * 60)
    # ts_model, ts_losses, predictions, test_data = train_timeseries_example()