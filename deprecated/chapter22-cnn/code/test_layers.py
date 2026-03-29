"""
test_layers.py - 单元测试

验证各层的实现正确性。
"""

import numpy as np
from layers import Conv2D, MaxPooling2D, Flatten, ReLU, Dense, SoftmaxCrossEntropy


def test_conv2d():
    """测试Conv2D层"""
    print("\n测试 Conv2D 层...")
    
    # 创建层
    conv = Conv2D(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0)
    
    # 固定权重以便测试
    conv.W = np.ones_like(conv.W)  # 全1权重
    conv.b = np.zeros_like(conv.b)
    
    # 输入
    x = np.ones((1, 1, 5, 5))  # 5x5的全1图像
    
    # 前向传播
    out = conv.forward(x)
    
    # 检查输出形状: (1, 2, 3, 3) - 因为(5-3)/1+1=3
    assert out.shape == (1, 2, 3, 3), f"输出形状错误: {out.shape}"
    
    # 检查输出值: 3x3全1卷积核在5x5全1输入上的结果是9（sum of 9 ones）
    expected = np.ones((1, 2, 3, 3)) * 9
    assert np.allclose(out, expected), f"前向传播错误"
    
    # 反向传播
    grad_output = np.ones_like(out)
    grad_input = conv.backward(grad_output)
    
    # 检查输入梯度形状
    assert grad_input.shape == x.shape, f"输入梯度形状错误: {grad_input.shape}"
    
    print("  ✓ Conv2D 测试通过")


def test_maxpooling():
    """测试MaxPooling2D层"""
    print("\n测试 MaxPooling2D 层...")
    
    pool = MaxPooling2D(pool_size=2, stride=2)
    
    # 测试输入
    x = np.array([[[[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]]])  # 形状: (1, 1, 4, 4)
    
    # 前向传播
    out = pool.forward(x)
    
    # 期望输出: 每个2x2区域的最大值
    expected = np.array([[[[6, 8],
                           [14, 16]]]])
    
    assert out.shape == (1, 1, 2, 2), f"输出形状错误: {out.shape}"
    assert np.allclose(out, expected), f"前向传播错误: {out} != {expected}"
    
    # 反向传播
    grad_output = np.ones_like(out)
    grad_input = pool.backward(grad_output)
    
    # 最大值位置应该有梯度1
    assert grad_input[0, 0, 1, 1] == 1  # 6的位置
    assert grad_input[0, 0, 1, 3] == 1  # 8的位置
    
    print("  ✓ MaxPooling2D 测试通过")


def test_flatten():
    """测试Flatten层"""
    print("\n测试 Flatten 层...")
    
    flatten = Flatten()
    
    # 输入 (2, 3, 4, 5)
    x = np.arange(120).reshape(2, 3, 4, 5)
    
    # 前向
    out = flatten.forward(x)
    assert out.shape == (2, 60), f"输出形状错误: {out.shape}"
    
    # 反向
    grad = np.ones_like(out)
    grad_input = flatten.backward(grad)
    assert grad_input.shape == x.shape, f"输入梯度形状错误: {grad_input.shape}"
    
    print("  ✓ Flatten 测试通过")


def test_relu():
    """测试ReLU层"""
    print("\n测试 ReLU 层...")
    
    relu = ReLU()
    
    x = np.array([[-1, 2, -3], [4, -5, 6]])
    
    # 前向
    out = relu.forward(x)
    expected = np.array([[0, 2, 0], [4, 0, 6]])
    assert np.allclose(out, expected), f"前向传播错误"
    
    # 反向
    grad_output = np.ones_like(out)
    grad_input = relu.backward(grad_output)
    expected_grad = np.array([[0, 1, 0], [1, 0, 1]])
    assert np.allclose(grad_input, expected_grad), f"反向传播错误"
    
    print("  ✓ ReLU 测试通过")


def test_dense():
    """测试Dense层"""
    print("\n测试 Dense 层...")
    
    dense = Dense(in_features=3, out_features=2)
    
    # 固定权重
    dense.W = np.array([[1, 2], [3, 4], [5, 6]])
    dense.b = np.array([0, 0])
    
    # 输入
    x = np.array([[1, 1, 1]])  # (1, 3)
    
    # 前向: [1,1,1] @ [[1,2],[3,4],[5,6]] = [9, 12]
    out = dense.forward(x)
    expected = np.array([[9, 12]])
    assert np.allclose(out, expected), f"前向传播错误: {out} != {expected}"
    
    # 反向
    grad_output = np.array([[1, 1]])
    grad_input = dense.backward(grad_output)
    
    # dW = x^T @ grad = [[1],[1],[1]] @ [[1,1]] = [[1,1],[1,1],[1,1]]
    expected_dW = np.array([[1, 1], [1, 1], [1, 1]])
    assert np.allclose(dense.grads[0], expected_dW), f"dW错误"
    
    print("  ✓ Dense 测试通过")


def test_softmax_crossentropy():
    """测试SoftmaxCrossEntropy"""
    print("\n测试 SoftmaxCrossEntropy...")
    
    criterion = SoftmaxCrossEntropy()
    
    # logits和标签
    logits = np.array([[1.0, 2.0, 3.0]])
    labels = np.array([2])  # 第三个类别是正确类别
    
    # 前向
    loss = criterion.forward(logits, labels)
    assert loss > 0, "损失应该为正"
    
    # 反向
    grad = criterion.backward()
    assert grad.shape == logits.shape, f"梯度形状错误: {grad.shape}"
    
    print("  ✓ SoftmaxCrossEntropy 测试通过")


def test_integration():
    """集成测试：简单网络的前向和反向传播"""
    print("\n测试 集成网络...")
    
    np.random.seed(42)
    
    # 创建简单网络
    layers = [
        Conv2D(1, 2, kernel_size=3, padding=1),
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2),
        Flatten(),
        Dense(2 * 14 * 14, 10)
    ]
    
    # 输入
    x = np.random.randn(2, 1, 28, 28)
    labels = np.array([1, 5])
    
    # 前向传播
    for layer in layers:
        x = layer.forward(x)
    
    assert x.shape == (2, 10), f"最终输出形状错误: {x.shape}"
    
    # 计算损失
    criterion = SoftmaxCrossEntropy()
    loss = criterion.forward(x, labels)
    
    # 反向传播
    grad = criterion.backward()
    for layer in reversed(layers):
        grad = layer.backward(grad)
    
    assert grad.shape == (2, 1, 28, 28), f"输入梯度形状错误: {grad.shape}"
    
    print("  ✓ 集成测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("运行单元测试")
    print("=" * 60)
    
    test_conv2d()
    test_maxpooling()
    test_flatten()
    test_relu()
    test_dense()
    test_softmax_crossentropy()
    test_integration()
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
