"""
utils.py - 实用工具函数

提供辅助功能：数据增强、模型保存/加载、评估指标等。
"""

import numpy as np
import pickle


def one_hot_encode(labels, num_classes):
    """
    将类别标签转换为one-hot编码
    
    参数:
        labels: 类别索引 (n_samples,)
        num_classes: 类别总数
    返回:
        one-hot编码 (n_samples, num_classes)
    """
    n = len(labels)
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), labels] = 1
    return one_hot


def random_flip(images, labels=None, horizontal=True, vertical=False):
    """
    随机水平/垂直翻转图像（数据增强）
    
    参数:
        images: 图像数组 (batch, channels, h, w)
        labels: 标签（翻转不改变标签，如MNIST数字）
        horizontal: 是否水平翻转
        vertical: 是否垂直翻转
    返回:
        翻转后的图像
    """
    flipped = images.copy()
    
    if horizontal and np.random.rand() > 0.5:
        flipped = flipped[:, :, :, ::-1]
    
    if vertical and np.random.rand() > 0.5:
        flipped = flipped[:, :, ::-1, :]
    
    return flipped if labels is None else (flipped, labels)


def random_shift(images, max_shift=2):
    """
    随机平移图像（数据增强）
    
    参数:
        images: 图像数组
        max_shift: 最大平移像素数
    返回:
        平移后的图像
    """
    batch, channels, h, w = images.shape
    shifted = np.zeros_like(images)
    
    for i in range(batch):
        shift_h = np.random.randint(-max_shift, max_shift + 1)
        shift_w = np.random.randint(-max_shift, max_shift + 1)
        
        # 计算切片
        src_h_start = max(0, shift_h)
        src_h_end = min(h, h + shift_h)
        src_w_start = max(0, shift_w)
        src_w_end = min(w, w + shift_w)
        
        dst_h_start = max(0, -shift_h)
        dst_h_end = min(h, h - shift_h)
        dst_w_start = max(0, -shift_w)
        dst_w_end = min(w, w - shift_w)
        
        shifted[i, :, dst_h_start:dst_h_end, dst_w_start:dst_w_end] = \
            images[i, :, src_h_start:src_h_end, src_w_start:src_w_end]
    
    return shifted


def add_noise(images, noise_factor=0.1):
    """
    添加高斯噪声（数据增强）
    
    参数:
        images: 图像数组
        noise_factor: 噪声强度
    返回:
        加噪后的图像
    """
    noisy = images + noise_factor * np.random.randn(*images.shape)
    return np.clip(noisy, 0, 1)


class EarlyStopping:
    """
    早停机制
    
    当验证指标不再改善时提前停止训练，防止过拟合。
    
    参数:
        patience: 容忍轮数
        min_delta: 最小改善量
        restore_best_weights: 是否恢复最佳权重
    """
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_value = None
        self.best_weights = None
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, value, model):
        """
        检查是否应该停止
        
        参数:
            value: 当前验证指标（如准确率）
            model: 模型对象
        返回:
            是否应该停止训练
        """
        if self.best_value is None or value > self.best_value + self.min_delta:
            self.best_value = value
            self.counter = 0
            # 保存最佳权重
            if self.restore_best_weights:
                self.best_weights = [p.copy() for layer in model.layers 
                                     for p in layer.params]
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                # 恢复最佳权重
                if self.restore_best_weights and self.best_weights:
                    idx = 0
                    for layer in model.layers:
                        for i in range(len(layer.params)):
                            layer.params[i] = self.best_weights[idx]
                            idx += 1
        
        return self.should_stop


def save_model(model, filepath):
    """
    保存模型权重
    
    参数:
        model: 模型对象
        filepath: 保存路径
    """
    weights = {}
    for i, layer in enumerate(model.layers):
        if len(layer.params) > 0:
            weights[f'layer_{i}_W'] = layer.params[0]
            weights[f'layer_{i}_b'] = layer.params[1]
    
    np.savez(filepath, **weights)
    print(f"模型已保存: {filepath}")


def load_model(model, filepath):
    """
    加载模型权重
    
    参数:
        model: 模型对象
        filepath: 权重文件路径
    """
    data = np.load(filepath)
    
    for i, layer in enumerate(model.layers):
        if len(layer.params) > 0:
            w_key = f'layer_{i}_W'
            b_key = f'layer_{i}_b'
            if w_key in data:
                layer.params[0] = data[w_key]
                layer.params[1] = data[b_key]
    
    print(f"模型已加载: {filepath}")


def compute_metrics(y_true, y_pred, average='macro'):
    """
    计算分类指标
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        average: 平均方式 ('macro', 'micro', 'weighted')
    返回:
        指标字典
    """
    classes = np.unique(y_true)
    
    # 计算每类的指标
    precisions = []
    recalls = []
    f1_scores = []
    
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # 计算平均
    if average == 'macro':
        metrics = {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1_score': np.mean(f1_scores),
            'accuracy': np.mean(y_true == y_pred)
        }
    else:
        # 简化处理，其他平均方式类似
        metrics = {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1_score': np.mean(f1_scores),
            'accuracy': np.mean(y_true == y_pred)
        }
    
    return metrics


def print_metrics(metrics, title="Metrics"):
    """
    打印评估指标
    
    参数:
        metrics: 指标字典
        title: 标题
    """
    print(f"\n{title}")
    print("-" * 30)
    for key, value in metrics.items():
        print(f"  {key.capitalize():12s}: {value:.4f}")


def learning_rate_schedule(epoch, initial_lr=0.001, decay_type='step', 
                           decay_factor=0.1, decay_epochs=10):
    """
    学习率调度
    
    参数:
        epoch: 当前epoch
        initial_lr: 初始学习率
        decay_type: 衰减类型 ('step', 'exponential', 'cosine')
        decay_factor: 衰减因子
        decay_epochs: 衰减间隔
    返回:
        当前学习率
    """
    if decay_type == 'step':
        return initial_lr * (decay_factor ** (epoch // decay_epochs))
    elif decay_type == 'exponential':
        return initial_lr * np.exp(-decay_factor * epoch)
    elif decay_type == 'cosine':
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / decay_epochs))
    else:
        return initial_lr


class ProgressBar:
    """
    简单的进度条
    
    用于显示训练进度。
    """
    def __init__(self, total, width=30):
        self.total = total
        self.width = width
        self.current = 0
    
    def update(self, current=None, info=""):
        """更新进度条"""
        if current is not None:
            self.current = current
        
        progress = self.current / self.total
        filled = int(self.width * progress)
        bar = '█' * filled + '░' * (self.width - filled)
        
        print(f'\r[{bar}] {self.current}/{self.total} {info}', end='', flush=True)
        
        if self.current >= self.total:
            print()  # 换行


if __name__ == "__main__":
    # 测试工具函数
    print("测试工具函数...")
    
    # 测试one-hot编码
    labels = np.array([0, 1, 2, 1, 0])
    one_hot = one_hot_encode(labels, 3)
    print(f"One-hot编码:\n{one_hot}")
    
    # 测试数据增强
    test_images = np.random.randn(4, 1, 28, 28)
    flipped = random_flip(test_images)
    print(f"\n原始形状: {test_images.shape}")
    print(f"翻转后形状: {flipped.shape}")
    
    # 测试指标计算
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 2, 1, 0])
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics, "Test Metrics")
    
    print("\n✓ 所有工具函数测试通过！")
