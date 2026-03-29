"""
full_example.py - 完整的使用示例

展示如何使用LeNet进行MNIST训练，并包含完整的可视化。
这是本章的完整演示脚本。
"""

import numpy as np
import os
from train_mnist import load_mnist, preprocess_data, train, evaluate
from lenet import LeNet
from visualize import (
    visualize_kernels, visualize_feature_maps, plot_training_history,
    visualize_predictions, visualize_layer_activations
)


def create_batches(x, y, batch_size=64, shuffle=True):
    """创建数据批次"""
    n = len(x)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_indices = indices[start:end]
        yield x[batch_indices], y[batch_indices]


def train_with_history(model, train_x, train_y, valid_x, valid_y,
                       epochs=10, batch_size=64, lr=0.001):
    """
    带详细历史记录的训练
    
    返回包含每轮详细信息的训练历史。
    """
    from layers import Adam
    optimizer = Adam(lr=lr)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_acc': []
    }
    
    n_batches = len(train_x) // batch_size
    
    print(f"\n开始训练 (Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr})")
    print("=" * 60)
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        # 训练阶段
        for batch_x, batch_y in create_batches(train_x, train_y, batch_size):
            logits = model.forward(batch_x)
            loss = model.compute_loss(logits, batch_y)
            model.backward()
            optimizer.step(model.layers)
            epoch_loss += loss
        
        # 评估阶段
        train_acc = evaluate(model, train_x[:5000], train_y[:5000])
        valid_acc = evaluate(model, valid_x, valid_y)
        
        # 记录
        history['train_loss'].append(epoch_loss / n_batches)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)
        
        # 打印进度
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Loss: {history['train_loss'][-1]:.4f} | "
              f"Train: {train_acc:.4f} | "
              f"Valid: {valid_acc:.4f}")
    
    print("=" * 60)
    return history


def run_complete_demo():
    """运行完整的演示"""
    
    print("=" * 70)
    print("          MNIST手写数字识别 - 完整示例")
    print("          纯NumPy实现的卷积神经网络")
    print("=" * 70)
    
    # 设置随机种子
    np.random.seed(42)
    
    # 创建输出目录
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载数据
    print("\n📊 [步骤 1] 加载MNIST数据集...")
    train_set, valid_set, test_set = load_mnist()
    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x, test_y = test_set
    
    print(f"    ✓ 训练集: {train_x.shape[0]:,} 样本")
    print(f"    ✓ 验证集: {valid_x.shape[0]:,} 样本")
    print(f"    ✓ 测试集: {test_x.shape[0]:,} 样本")
    
    # 2. 预处理
    print("\n🔧 [步骤 2] 预处理数据...")
    train_x = preprocess_data(train_x)
    valid_x = preprocess_data(valid_x)
    test_x = preprocess_data(test_x)
    print(f"    ✓ 输入形状: {train_x.shape[1:]} (批次, 通道, 高, 宽)")
    
    # 3. 创建模型
    print("\n🏗️  [步骤 3] 创建LeNet模型...")
    model = LeNet(num_classes=10)
    model.summary()
    
    # 4. 训练
    print("\n🚀 [步骤 4] 训练模型...")
    history = train_with_history(model, train_x, train_y, valid_x, valid_y,
                                  epochs=10, batch_size=64, lr=0.001)
    
    # 5. 可视化训练历史
    print("\n📈 [步骤 5] 生成训练可视化...")
    
    # 训练历史曲线
    history_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(history, save_path=history_path)
    print(f"    ✓ 训练历史: {history_path}")
    
    # 预测结果
    pred_path = os.path.join(output_dir, 'predictions.png')
    visualize_predictions(model, test_x, test_y, num_samples=16, 
                         save_path=pred_path)
    print(f"    ✓ 预测结果: {pred_path}")
    
    # 卷积核可视化
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'W') and len(layer.W.shape) == 4:
            kernel_path = os.path.join(output_dir, f'layer_{i}_kernels.png')
            visualize_kernels(layer, title=f'Layer {i} Kernels',
                            max_display=8, save_path=kernel_path)
            print(f"    ✓ 卷积核可视化 (Layer {i}): {kernel_path}")
            break  # 只可视化第一个卷积层
    
    # 特征图可视化
    sample_x = test_x[0:1]  # 取第一个测试样本
    for i, layer in enumerate(model.layers):
        if 'Conv2D' in layer.__class__.__name__:
            feature_maps = model.get_feature_maps(sample_x, i)
            feature_path = os.path.join(output_dir, f'layer_{i}_features.png')
            visualize_feature_maps(feature_maps, 
                                  title=f'Layer {i} Feature Maps',
                                  max_display=16, save_path=feature_path)
            print(f"    ✓ 特征图可视化 (Layer {i}): {feature_path}")
    
    # 6. 最终评估
    print("\n✅ [步骤 6] 最终评估...")
    test_acc = evaluate(model, test_x, test_y)
    print(f"    测试集准确率: {test_acc*100:.2f}%")
    
    # 各类别准确率
    predictions = model.predict(test_x)
    print("\n    各类别准确率:")
    for digit in range(10):
        mask = test_y == digit
        digit_acc = np.mean(predictions[mask] == test_y[mask])
        print(f"      数字 {digit}: {digit_acc*100:.2f}%")
    
    # 7. 保存模型参数
    print("\n💾 [步骤 7] 保存模型参数...")
    model_path = os.path.join(output_dir, 'model_weights.npz')
    weights_dict = {}
    for i, layer in enumerate(model.layers):
        if len(layer.params) > 0:
            weights_dict[f'layer_{i}_W'] = layer.params[0]
            weights_dict[f'layer_{i}_b'] = layer.params[1]
    np.savez(model_path, **weights_dict)
    print(f"    ✓ 模型权重: {model_path}")
    
    print("\n" + "=" * 70)
    print("🎉 演示完成！所有结果已保存到 './results/' 目录")
    print("=" * 70)
    
    return model, history


def quick_demo():
    """
    快速演示（少epoch，用于测试）
    """
    print("\n" + "=" * 60)
    print("快速演示模式 (3 epochs)")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 加载数据
    train_set, valid_set, test_set = load_mnist()
    train_x = preprocess_data(train_set[0])
    train_y = train_set[1]
    valid_x = preprocess_data(valid_set[0])
    valid_y = valid_set[1]
    test_x = preprocess_data(test_set[0])
    test_y = test_set[1]
    
    # 创建模型
    model = LeNet(num_classes=10)
    
    # 训练
    history = train_with_history(model, train_x[:10000], train_y[:10000], 
                                  valid_x, valid_y,
                                  epochs=3, batch_size=128, lr=0.001)
    
    # 评估
    test_acc = evaluate(model, test_x, test_y)
    print(f"\n测试集准确率: {test_acc*100:.2f}%")
    
    return model, history


if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # 快速模式
        model, history = quick_demo()
    else:
        # 完整演示
        model, history = run_complete_demo()
