"""
visualize.py - CNN可视化工具

提供各种可视化功能，帮助理解CNN的内部工作机制：
- 卷积核可视化
- 特征图可视化
- 训练历史可视化
- 预测结果可视化
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


def visualize_kernels(conv_layer, title="Convolution Kernels", 
                      max_display=None, save_path=None, show=False):
    """
    可视化卷积核
    
    将卷积层的权重可视化为灰度图像。
    
    参数:
        conv_layer: Conv2D层对象
        title: 图表标题
        max_display: 最多显示的核数量（None表示全部）
        save_path: 保存路径
        show: 是否显示（在服务器环境通常设为False）
    """
    W = conv_layer.W  # 形状: (out_c, in_c, k, k)
    out_c, in_c, k, _ = W.shape
    
    # 限制显示数量
    if max_display is not None:
        out_c = min(out_c, max_display)
        W = W[:out_c]
    
    # 归一化到[0, 1]以便显示
    W_min, W_max = W.min(), W.max()
    W_display = (W - W_min) / (W_max - W_min + 1e-8)
    
    # 创建子图
    fig, axes = plt.subplots(out_c, in_c, figsize=(in_c * 1.5, out_c * 1.5))
    
    # 处理单个子图的情况
    if out_c == 1 and in_c == 1:
        axes = [[axes]]
    elif out_c == 1:
        axes = [axes]
    elif in_c == 1:
        axes = [[ax] for ax in axes]
    
    for i in range(out_c):
        for j in range(in_c):
            ax = axes[i][j] if in_c > 1 else axes[i]
            ax.imshow(W_display[i, j], cmap='gray', interpolation='nearest')
            ax.axis('off')
            if i == 0:
                ax.set_title(f'In {j}', fontsize=8)
            if j == 0:
                ax.set_ylabel(f'Out {i}', fontsize=8)
    
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"卷积核可视化已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_feature_maps(feature_maps, title="Feature Maps", 
                           max_display=16, save_path=None, show=False):
    """
    可视化特征图
    
    显示CNN中间层产生的激活图。
    
    参数:
        feature_maps: 特征图 (batch, channels, h, w) 或 (channels, h, w)
        title: 图表标题
        max_display: 最多显示的通道数
        save_path: 保存路径
        show: 是否显示
    """
    # 处理输入形状
    if len(feature_maps.shape) == 4:
        feature_maps = feature_maps[0]  # 取第一个样本
    
    n_channels = min(feature_maps.shape[0], max_display)
    
    # 计算子图布局
    n_rows = int(np.ceil(np.sqrt(n_channels)))
    n_cols = int(np.ceil(n_channels / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    
    if n_channels == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    # 归一化
    fm_min, fm_max = feature_maps[:n_channels].min(), feature_maps[:n_channels].max()
    
    for i in range(n_channels):
        ax = axes[i]
        im = ax.imshow(feature_maps[i], cmap='viridis', 
                      vmin=fm_min, vmax=fm_max, interpolation='nearest')
        ax.axis('off')
        ax.set_title(f'Ch {i}', fontsize=8)
    
    # 隐藏多余的子图
    for i in range(n_channels, len(axes)):
        axes[i].axis('off')
    
    # 添加颜色条
    fig.colorbar(im, ax=axes, shrink=0.5)
    
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"特征图可视化已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_history(history, save_path=None, show=False):
    """
    绘制训练历史
    
    显示损失和准确率随epoch的变化。
    
    参数:
        history: 包含 'train_loss', 'train_acc', 'valid_acc' 的字典
        save_path: 保存路径
        show: 是否显示
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 损失曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    if 'valid_loss' in history:
        axes[0].plot(epochs, history['valid_loss'], 'r-', linewidth=2, label='Valid Loss')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training Loss', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc')
    if 'valid_acc' in history:
        axes[1].plot(epochs, history['valid_acc'], 'r-', linewidth=2, label='Valid Acc')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy', fontsize=11)
    axes[1].set_title('Training Accuracy', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练历史已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_predictions(model, test_x, test_y, num_samples=16, 
                          save_path=None, show=False):
    """
    可视化预测结果
    
    显示测试样本及其预测结果。
    
    参数:
        model: 训练好的模型
        test_x: 测试数据
        test_y: 测试标签
        num_samples: 显示的样本数
        save_path: 保存路径
        show: 是否显示
    """
    # 随机选择样本
    indices = np.random.choice(len(test_x), num_samples, replace=False)
    samples_x = test_x[indices]
    samples_y = test_y[indices]
    
    # 预测
    predictions = model.predict(samples_x)
    probs = model.predict_proba(samples_x)
    
    # 计算布局
    n_rows = int(np.ceil(np.sqrt(num_samples)))
    n_cols = int(np.ceil(num_samples / n_rows))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        # 显示图像
        ax.imshow(samples_x[i, 0], cmap='gray', interpolation='nearest')
        
        # 获取预测概率
        pred_prob = probs[i, predictions[i]]
        
        # 设置标题
        is_correct = predictions[i] == samples_y[i]
        color = 'green' if is_correct else 'red'
        marker = '✓' if is_correct else '✗'
        
        ax.set_title(f'{marker} Pred: {predictions[i]} ({pred_prob:.2%})\n'
                    f'   True: {samples_y[i]}', 
                    color=color, fontsize=9)
        ax.axis('off')
    
    # 隐藏多余的子图
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Predictions (Green=Correct, Red=Wrong)', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"预测可视化已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_layer_activations(model, sample_x, save_dir='./visualizations'):
    """
    可视化网络各层的激活
    
    对单个输入样本，显示经过每一层后的输出。
    
    参数:
        model: CNN模型
        sample_x: 单个样本 (1, 1, 28, 28)
        save_dir: 保存目录
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 确保是单个样本
    if len(sample_x.shape) == 3:
        sample_x = sample_x[np.newaxis]
    
    x = sample_x
    conv_layer_idx = 0
    
    for i, layer in enumerate(model.layers):
        x = layer.forward(x)
        layer_name = layer.__class__.__name__
        
        # 只可视化卷积层的输出
        if 'Conv2D' in layer_name:
            save_path = os.path.join(save_dir, f'layer_{i}_{layer_name}_features.png')
            visualize_feature_maps(x, 
                                  title=f'Layer {i}: {layer_name} Output',
                                  max_display=16,
                                  save_path=save_path)
            conv_layer_idx += 1


def plot_confusion_matrix(y_true, y_pred, class_names=None, 
                          save_path=None, show=False):
    """
    绘制混淆矩阵
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径
        show: 是否显示
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_true)))]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # 添加数值标注
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"混淆矩阵已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# 辅助函数：创建训练过程的动画帧
def save_training_progress_frames(model, test_x, test_y, history, 
                                   output_dir='./training_frames'):
    """
    保存训练过程中的多个检查点可视化
    
    参数:
        model: 训练好的模型
        test_x: 测试数据
        test_y: 测试标签
        history: 训练历史
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练历史
    plot_training_history(history, 
                         save_path=os.path.join(output_dir, 'training_history.png'))
    
    # 保存预测结果
    visualize_predictions(model, test_x, test_y, num_samples=16,
                         save_path=os.path.join(output_dir, 'predictions.png'))
    
    # 找到第一个卷积层并可视化其卷积核
    for layer in model.layers:
        if hasattr(layer, 'W') and len(layer.W.shape) == 4:
            visualize_kernels(layer,
                            title='First Conv Layer Kernels',
                            save_path=os.path.join(output_dir, 'kernels.png'))
            break
    
    print(f"训练进度可视化已保存到: {output_dir}")
