
"""
第十七章：多层神经网络——从零实现MLP
《机器学习与深度学习：从小学生到大师》

本代码包含：
1. 完整的MLP类实现（前向传播 + 反向传播）
2. XOR问题的完整解决示例
3. 手写数字识别（简化版MNIST）
4. 丰富的可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 第一部分：激活函数及其导数
# ============================================================================

class Activations:
    """激活函数集合"""
    
    @staticmethod
    def sigmoid(z):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(z):
        """Sigmoid的导数"""
        a = Activations.sigmoid(z)
        return a * (1 - a)
    
    @staticmethod
    def relu(z):
        """ReLU激活函数"""
        return np.maximum(0, z)
    
    @staticmethod
    def relu_derivative(z):
        """ReLU的导数"""
        return (z > 0).astype(float)
    
    @staticmethod
    def tanh(z):
        """Tanh激活函数"""
        return np.tanh(z)
    
    @staticmethod
    def tanh_derivative(z):
        """Tanh的导数"""
        return 1 - np.tanh(z) ** 2
    
    @staticmethod
    def linear(z):
        """线性激活（无变换）"""
        return z
    
    @staticmethod
    def linear_derivative(z):
        """线性激活的导数"""
        return np.ones_like(z)
    
    @staticmethod
    def softmax(z):
        """Softmax激活函数（用于多分类输出层）"""
        # 数值稳定性处理
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)


# ============================================================================
# 第二部分：损失函数
# ============================================================================

class LossFunctions:
    """损失函数集合"""
    
    @staticmethod
    def mse(y_true, y_pred):
        """均方误差"""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        """MSE对预测的导数"""
        return -2 * (y_true - y_pred) / y_true.shape[1]
    
    @staticmethod
    def cross_entropy(y_true, y_pred):
        """交叉熵损失（带数值稳定性）"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))
    
    @staticmethod
    def cross_entropy_derivative(y_true, y_pred):
        """交叉熵 + Softmax的组合导数"""
        return y_pred - y_true


# ============================================================================
# 第三部分：多层感知机（MLP）类
# ============================================================================

class MLP:
    """
    多层感知机（Multilayer Perceptron）
    
    参数:
        layer_sizes: 列表，如 [2, 4, 1] 表示输入2维，隐藏层4维，输出1维
        activations: 列表，每层的激活函数名称，如 ['relu', 'sigmoid']
        loss_function: 损失函数名称 ('mse' 或 'cross_entropy')
        learning_rate: 学习率
        random_seed: 随机种子（保证可重复）
    """
    
    def __init__(self, layer_sizes, activations, loss_function='mse',
                 learning_rate=0.1, random_seed=42):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = learning_rate
        self.loss_name = loss_function
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 初始化权重和偏置
        self._initialize_parameters()
        
        # 设置激活函数
        self._setup_activations(activations)
        
        # 设置损失函数
        self._setup_loss_function()
        
        # 存储训练历史
        self.loss_history = []
        
    def _initialize_parameters(self):
        """
        初始化网络参数
        使用Xavier/Glorot初始化，有助于梯度稳定流动
        """
        self.parameters = {}
        self.gradients = {}
        
        for l in range(1, self.num_layers):
            # Xavier初始化：权重从均值为0，方差为 1/n_in 的正态分布采样
            n_in = self.layer_sizes[l-1]
            n_out = self.layer_sizes[l]
            self.parameters[f'W{l}'] = np.random.randn(n_out, n_in) * np.sqrt(2.0 / (n_in + n_out))
            self.parameters[f'b{l}'] = np.zeros((n_out, 1))
    
    def _setup_activations(self, activations):
        """设置每层的激活函数"""
        self.activations = []
        self.activation_derivatives = []
        
        act_map = {
            'sigmoid': (Activations.sigmoid, Activations.sigmoid_derivative),
            'relu': (Activations.relu, Activations.relu_derivative),
            'tanh': (Activations.tanh, Activations.tanh_derivative),
            'linear': (Activations.linear, Activations.linear_derivative),
            'softmax': (Activations.softmax, None)  # softmax通常与cross_entropy配合使用
        }
        
        for act_name in activations:
            if act_name not in act_map:
                raise ValueError(f"未知的激活函数: {act_name}")
            self.activations.append(act_map[act_name][0])
            self.activation_derivatives.append(act_map[act_name][1])
    
    def _setup_loss_function(self):
        """设置损失函数"""
        if self.loss_name == 'mse':
            self.loss_fn = LossFunctions.mse
            self.loss_derivative = LossFunctions.mse_derivative
        elif self.loss_name == 'cross_entropy':
            self.loss_fn = LossFunctions.cross_entropy
            self.loss_derivative = LossFunctions.cross_entropy_derivative
        else:
            raise ValueError(f"未知的损失函数: {self.loss_name}")
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: 输入数据，形状为 (n_features, n_samples)
        
        返回:
            网络输出
        """
        # 存储每层的激活值和加权输入（用于反向传播）
        self.cache = {'A0': X}
        
        A = X
        for l in range(1, self.num_layers):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            # 计算加权输入 Z = W·A + b
            Z = np.dot(W, A) + b
            self.cache[f'Z{l}'] = Z
            
            # 应用激活函数
            A = self.activations[l-1](Z)
            self.cache[f'A{l}'] = A
        
        return A
    
    def backward(self, Y):
        """
        反向传播
        
        参数:
            Y: 真实标签，形状为 (n_outputs, n_samples)
        """
        m = Y.shape[1]  # 样本数量
        L = self.num_layers - 1  # 最后一层的索引
        
        # 获取最后一层的输出
        A_L = self.cache[f'A{L}']
        Z_L = self.cache[f'Z{L}']
        
        # 计算输出层的误差（delta）
        if self.loss_name == 'cross_entropy' and self.activations[-1] == Activations.softmax:
            # 对于Softmax + CrossEntropy的组合，导数简化为 A - Y
            dZ = A_L - Y
        else:
            # 一般情况：损失函数导数 * 激活函数导数
            dA = self.loss_derivative(Y, A_L)
            dZ = dA * self.activation_derivatives[-1](Z_L)
        
        # 从最后一层向前传播误差
        for l in range(L, 0, -1):
            A_prev = self.cache[f'A{l-1}']
            
            # 计算该层的梯度
            self.gradients[f'dW{l}'] = np.dot(dZ, A_prev.T) / m
            self.gradients[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m
            
            # 如果不是第一层，计算前一层的误差
            if l > 1:
                W = self.parameters[f'W{l}']
                dA_prev = np.dot(W.T, dZ)
                Z_prev = self.cache[f'Z{l-1}']
                dZ = dA_prev * self.activation_derivatives[l-2](Z_prev)
    
    def update_parameters(self):
        """使用梯度下降更新参数"""
        for l in range(1, self.num_layers):
            self.parameters[f'W{l}'] -= self.learning_rate * self.gradients[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * self.gradients[f'db{l}']
    
    def train(self, X, Y, epochs=1000, batch_size=None, verbose=True, print_every=100):
        """
        训练网络
        
        参数:
            X: 输入数据 (n_features, n_samples)
            Y: 标签 (n_outputs, n_samples)
            epochs: 训练轮数
            batch_size: 批量大小（None表示使用全部数据）
            verbose: 是否打印进度
            print_every: 每隔多少轮打印一次
        """
        m = X.shape[1]  # 总样本数
        
        if batch_size is None:
            batch_size = m
        
        num_batches = (m + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # 随机打乱数据
            indices = np.random.permutation(m)
            X_shuffled = X[:, indices]
            Y_shuffled = Y[:, indices]
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, m)
                
                X_batch = X_shuffled[:, start_idx:end_idx]
                Y_batch = Y_shuffled[:, start_idx:end_idx]
                
                # 前向传播
                Y_pred = self.forward(X_batch)
                
                # 计算损失
                loss = self.loss_fn(Y_batch, Y_pred)
                epoch_loss += loss * (end_idx - start_idx) / m
                
                # 反向传播
                self.backward(Y_batch)
                
                # 更新参数
                self.update_parameters()
            
            self.loss_history.append(epoch_loss)
            
            if verbose and (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
        
        if verbose:
            print(f"\n训练完成！最终损失: {self.loss_history[-1]:.6f}")
    
    def predict(self, X):
        """
        预测
        
        参数:
            X: 输入数据
        
        返回:
            预测结果
        """
        return self.forward(X)
    
    def predict_class(self, X):
        """
        预测类别（用于分类问题）
        
        返回类别索引
        """
        probs = self.predict(X)
        return np.argmax(probs, axis=0)
    
    def score(self, X, Y):
        """
        计算准确率（分类问题）
        
        参数:
            X: 输入数据
            Y: one-hot编码的标签
        """
        predictions = self.predict_class(X)
        true_labels = np.argmax(Y, axis=0)
        return np.mean(predictions == true_labels)


# ============================================================================
# 第四部分：XOR问题完整解决示例
# ============================================================================

def solve_xor_problem():
    """
    使用MLP解决XOR问题
    这是神经网络历史上的经典问题！
    """
    print("=" * 60)
    print("XOR问题：多层神经网络的Hello World")
    print("=" * 60)
    
    # XOR数据集
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    Y = np.array([[0, 1, 1, 0]])  # XOR真值表
    
    print("\n【数据集】")
    print("输入 X:")
    print("  (0,0) → 0")
    print("  (0,1) → 1")
    print("  (1,0) → 1")
    print("  (1,1) → 0")
    print("\n这是一个线性不可分问题，单层感知机无法解决！")
    
    # 创建MLP：2输入 → 4隐藏 → 1输出
    print("\n【网络结构】")
    print("  输入层: 2个神经元")
    print("  隐藏层: 4个神经元 (ReLU激活)")
    print("  输出层: 1个神经元 (Sigmoid激活)")
    print("  损失函数: MSE")
    print("  学习率: 0.5")
    
    mlp = MLP(
        layer_sizes=[2, 4, 1],
        activations=['relu', 'sigmoid'],
        loss_function='mse',
        learning_rate=0.5,
        random_seed=42
    )
    
    # 训练
    print("\n【训练过程】")
    mlp.train(X, Y, epochs=2000, print_every=200)
    
    # 测试
    print("\n【测试结果】")
    predictions = mlp.predict(X)
    
    for i in range(4):
        x1, x2 = X[0, i], X[1, i]
        true_y = Y[0, i]
        pred_y = predictions[0, i]
        print(f"  输入: ({x1}, {x2}) | 预测: {pred_y:.4f} | 真实: {true_y} | 判断: {'✓' if abs(pred_y - true_y) < 0.5 else '✗'}")
    
    # 可视化
    visualize_xor(mlp, X, Y)
    
    return mlp


def visualize_xor(mlp, X, Y):
    """可视化XOR问题的决策边界"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：决策边界
    ax1 = axes[0]
    
    # 创建网格
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # 预测网格点的类别
    grid_points = np.c_[xx.ravel(), yy.ravel()].T
    Z = mlp.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    contour = ax1.contourf(xx, yy, Z, levels=50, cmap='RdYlBu', alpha=0.6)
    ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # 绘制数据点
    for i in range(X.shape[1]):
        color = 'blue' if Y[0, i] == 0 else 'red'
        marker = 'o' if Y[0, i] == 0 else 's'
        ax1.scatter(X[0, i], X[1, i], c=color, marker=marker, s=200, 
                   edgecolors='black', linewidth=2, zorder=5)
    
    ax1.set_xlabel('输入 1', fontsize=12)
    ax1.set_ylabel('输入 2', fontsize=12)
    ax1.set_title('XOR问题的决策边界\n（黑色线表示分类边界）', fontsize=14)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    plt.colorbar(contour, ax=ax1, label='输出概率')
    
    # 右图：损失曲线
    ax2 = axes[1]
    ax2.plot(mlp.loss_history, linewidth=2, color='purple')
    ax2.set_xlabel('训练轮次', fontsize=12)
    ax2.set_ylabel('损失 (MSE)', fontsize=12)
    ax2.set_title('训练过程中的损失下降', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xor_solution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n可视化已保存为 'xor_solution.png'")


# ============================================================================
# 第五部分：手写数字识别（简化版MNIST）
# ============================================================================

def load_digits_dataset():
    """
    加载手写数字数据集（sklearn内置的简化版MNIST）
    """
    print("\n" + "=" * 60)
    print("手写数字识别：MLP实战")
    print("=" * 60)
    
    # 加载数据
    digits = load_digits()
    X = digits.data  # (1797, 64) - 8x8像素的图像展平
    y = digits.target  # (1797,) - 0-9的数字标签
    
    print(f"\n【数据集信息】")
    print(f"  总样本数: {X.shape[0]}")
    print(f"  特征维度: {X.shape[1]} (8×8像素)")
    print(f"  类别数: 10 (数字0-9)")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 转置以匹配我们的MLP接口 (n_features, n_samples)
    X_train = X_train.T
    X_test = X_test.T
    
    # One-hot编码标签
    encoder = OneHotEncoder(sparse_output=False)
    Y_train = encoder.fit_transform(y_train.reshape(-1, 1)).T
    Y_test = encoder.transform(y_test.reshape(-1, 1)).T
    
    return X_train, X_test, Y_train, Y_test, y_train, y_test, digits


def train_digit_classifier():
    """训练手写数字分类器"""
    # 加载数据
    X_train, X_test, Y_train, Y_test, y_train, y_test, digits = load_digits_dataset()
    
    # 创建MLP
    print("\n【网络结构】")
    print("  输入层: 64个神经元 (8×8图像)")
    print("  隐藏层1: 128个神经元 (ReLU)")
    print("  隐藏层2: 64个神经元 (ReLU)")
    print("  输出层: 10个神经元 (Softmax)")
    print("  损失函数: 交叉熵")
    print("  学习率: 0.1")
    print("  批量大小: 32")
    
    mlp = MLP(
        layer_sizes=[64, 128, 64, 10],
        activations=['relu', 'relu', 'softmax'],
        loss_function='cross_entropy',
        learning_rate=0.1,
        random_seed=42
    )
    
    # 训练
    print("\n【训练过程】")
    mlp.train(X_train, Y_train, epochs=100, batch_size=32, print_every=10)
    
    # 评估
    train_acc = mlp.score(X_train, Y_train)
    test_acc = mlp.score(X_test, Y_test)
    
    print(f"\n【评估结果】")
    print(f"  训练集准确率: {train_acc*100:.2f}%")
    print(f"  测试集准确率: {test_acc*100:.2f}%")
    
    # 可视化结果
    visualize_digits_results(mlp, X_test, y_test, digits)
    
    return mlp


def visualize_digits_results(mlp, X_test, y_test, digits):
    """可视化手写数字识别结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 损失曲线
    ax1 = axes[0, 0]
    ax1.plot(mlp.loss_history, linewidth=2, color='blue')
    ax1.set_xlabel('训练轮次', fontsize=12)
    ax1.set_ylabel('交叉熵损失', fontsize=12)
    ax1.set_title('训练损失曲线', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 2. 随机样本预测展示
    ax2 = axes[0, 1]
    
    # 随机选择16个测试样本
    n_samples = 16
    indices = np.random.choice(X_test.shape[1], n_samples, replace=False)
    
    fig2, sample_axes = plt.subplots(4, 4, figsize=(10, 10))
    sample_axes = sample_axes.flatten()
    
    for i, idx in enumerate(indices):
        img = X_test[:, idx].reshape(8, 8)
        pred = mlp.predict_class(X_test[:, idx:idx+1])[0]
        true = y_test[idx]
        
        sample_axes[i].imshow(img, cmap='gray')
        color = 'green' if pred == true else 'red'
        sample_axes[i].set_title(f'预测: {pred}\n真实: {true}', color=color, fontsize=10)
        sample_axes[i].axis('off')
    
    plt.suptitle('随机测试样本预测结果（绿色=正确，红色=错误）', fontsize=14)
    plt.tight_layout()
    plt.savefig('digit_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 重新使用原来的axes
    predictions = mlp.predict(X_test)
    pred_classes = np.argmax(predictions, axis=0)
    
    # 3. 混淆矩阵
    ax3 = axes[1, 0]
    confusion = np.zeros((10, 10), dtype=int)
    for true, pred in zip(y_test, pred_classes):
        confusion[true, pred] += 1
    
    im = ax3.imshow(confusion, cmap='Blues')
    ax3.set_xlabel('预测标签', fontsize=12)
    ax3.set_ylabel('真实标签', fontsize=12)
    ax3.set_title('混淆矩阵', fontsize=14)
    ax3.set_xticks(range(10))
    ax3.set_yticks(range(10))
    
    # 添加数值标注
    for i in range(10):
        for j in range(10):
            text = ax3.text(j, i, confusion[i, j], ha="center", va="center", 
                           color="white" if confusion[i, j] > confusion.max()/2 else "black",
                           fontsize=9)
    
    plt.colorbar(im, ax=ax3)
    
    # 4. 每个数字的准确率
    ax4 = axes[1, 1]
    digit_accuracy = []
    for digit in range(10):
        mask = y_test == digit
        acc = np.mean(pred_classes[mask] == digit)
        digit_accuracy.append(acc)
    
    bars = ax4.bar(range(10), digit_accuracy, color='steelblue', edgecolor='black')
    ax4.set_xlabel('数字', fontsize=12)
    ax4.set_ylabel('准确率', fontsize=12)
    ax4.set_title('每个数字的分类准确率', fontsize=14)
    ax4.set_xticks(range(10))
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值
    for bar, acc in zip(bars, digit_accuracy):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('digits_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n可视化已保存:")
    print("  - digit_predictions.png: 随机样本预测")
    print("  - digits_analysis.png: 综合分析")


# ============================================================================
# 第六部分：隐藏层激活可视化
# ============================================================================

def visualize_hidden_activations():
    """
    可视化隐藏层学到的特征
    展示网络如何将输入数据映射到新的表示空间
    """
    print("\n" + "=" * 60)
    print("隐藏层激活可视化")
    print("=" * 60)
    
    # 创建一个简单的分类问题（同心圆）
    np.random.seed(42)
    n_samples = 400
    
    # 生成两个同心圆
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    r_inner = np.random.normal(2, 0.3, n_samples//2)
    r_outer = np.random.normal(4, 0.3, n_samples//2)
    
    X_inner = np.column_stack([r_inner * np.cos(theta[:n_samples//2]),
                               r_inner * np.sin(theta[:n_samples//2])])
    X_outer = np.column_stack([r_outer * np.cos(theta[n_samples//2:]),
                               r_outer * np.sin(theta[n_samples//2:])])
    
    X = np.vstack([X_inner, X_outer]).T
    Y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)]).reshape(1, -1)
    
    print("\n【任务】分离两个同心圆（非线性可分问题）")
    
    # 创建MLP
    mlp = MLP(
        layer_sizes=[2, 8, 4, 1],
        activations=['tanh', 'tanh', 'sigmoid'],
        loss_function='mse',
        learning_rate=0.5,
        random_seed=42
    )
    
    print("\n【网络结构】2 → 8 → 4 → 1")
    print("【训练】500轮...")
    mlp.train(X, Y, epochs=500, print_every=50)
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 原始数据
    ax1 = axes[0, 0]
    ax1.scatter(X[0, :200], X[1, :200], c='blue', label='Class 0', alpha=0.6)
    ax1.scatter(X[0, 200:], X[1, 200:], c='red', label='Class 1', alpha=0.6)
    ax1.set_title('原始输入空间', fontsize=14)
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.legend()
    ax1.set_aspect('equal')
    
    # 2. 决策边界
    ax2 = axes[0, 1]
    x_min, x_max = X[0].min() - 1, X[0].max() + 1
    y_min, y_max = X[1].min() - 1, X[1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()].T
    Z = mlp.predict(grid).reshape(xx.shape)
    
    ax2.contourf(xx, yy, Z, levels=50, cmap='RdYlBu', alpha=0.6)
    ax2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    ax2.scatter(X[0, :200], X[1, :200], c='blue', alpha=0.6, edgecolors='white')
    ax2.scatter(X[0, 200:], X[1, 200:], c='red', alpha=0.6, edgecolors='white')
    ax2.set_title('学习到的决策边界', fontsize=14)
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_aspect('equal')
    
    # 3. 损失曲线
    ax3 = axes[0, 2]
    ax3.plot(mlp.loss_history, linewidth=2, color='purple')
    ax3.set_title('损失下降曲线', fontsize=14)
    ax3.set_xlabel('轮次')
    ax3.set_ylabel('MSE损失')
    ax3.grid(True, alpha=0.3)
    
    # 4-6. 隐藏层激活可视化
    # 第一层隐藏层激活
    A1 = mlp.cache['A1']
    ax4 = axes[1, 0]
    
    # 使用PCA降到2D进行可视化
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    A1_pca = pca.fit_transform(A1.T)
    
    ax4.scatter(A1_pca[:200, 0], A1_pca[:200, 1], c='blue', alpha=0.6, label='Class 0')
    ax4.scatter(A1_pca[200:, 0], A1_pca[200:, 1], c='red', alpha=0.6, label='Class 1')
    ax4.set_title(f'第一层隐藏层激活\n(PCA投影, 解释方差: {sum(pca.explained_variance_ratio_)*100:.1f}%)', fontsize=14)
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.legend()
    
    # 第二层隐藏层激活
    A2 = mlp.cache['A2']
    ax5 = axes[1, 1]
    
    # 对于4维，我们可以展示所有两两组合
    ax5.scatter(A2[0, :200], A2[1, :200], c='blue', alpha=0.6, label='Class 0')
    ax5.scatter(A2[0, 200:], A2[1, 200:], c='red', alpha=0.6, label='Class 1')
    ax5.set_title('第二层隐藏层激活\n(维度1 vs 维度2)', fontsize=14)
    ax5.set_xlabel('激活值 1')
    ax5.set_ylabel('激活值 2')
    ax5.legend()
    
    # 最后一层输出
    ax6 = axes[1, 2]
    output = mlp.predict(X).flatten()
    ax6.hist(output[:200], bins=30, alpha=0.6, color='blue', label='Class 0')
    ax6.hist(output[200:], bins=30, alpha=0.6, color='red', label='Class 1')
    ax6.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='决策边界')
    ax6.set_title('输出层概率分布', fontsize=14)
    ax6.set_xlabel('预测概率')
    ax6.set_ylabel('样本数')
    ax6.legend()
    
    plt.suptitle('隐藏层如何将非线性可分问题转换为线性可分', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('hidden_activations.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n可视化已保存为 'hidden_activations.png'")
    print("观察隐藏层激活如何将同心圆数据映射到可分的空间！")


# ============================================================================
# 第七部分：网络容量与参数计算
# ============================================================================

def analyze_network_capacity():
    """
    分析不同网络结构的参数数量和容量
    """
    print("\n" + "=" * 60)
    print("神经网络容量分析")
    print("=" * 60)
    
    architectures = [
        ([2, 4, 1], "简单XOR网络"),
        ([64, 128, 64, 10], "手写数字分类器"),
        ([784, 256, 128, 64, 10], "标准MNIST网络"),
        ([100, 200, 200, 200, 100], "深度特征提取器"),
    ]
    
    print("\n【不同网络结构的参数统计】")
    print("-" * 60)
    print(f"{'结构':<25} {'描述':<20} {'参数量':<15}")
    print("-" * 60)
    
    for arch, desc in architectures:
        # 计算参数数量
        total_params = 0
        for i in range(len(arch) - 1):
            # 权重 + 偏置
            layer_params = arch[i] * arch[i+1] + arch[i+1]
            total_params += layer_params
        
        arch_str = " → ".join(map(str, arch))
        print(f"{arch_str:<25} {desc:<20} {total_params:<15,}")
    
    print("-" * 60)
    
    # 参数数量计算公式说明
    print("\n【参数数量计算公式】")
    print("对于从层 l-1 到层 l 的连接:")
    print("  权重数量 = n^(l) × n^(l-1)")
    print("  偏置数量 = n^(l)")
    print("  该层总参数 = n^(l) × n^(l-1) + n^(l) = n^(l) × (n^(l-1) + 1)")
    print("\n其中 n^(l) 表示第 l 层的神经元数量")
    
    # 可视化参数分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 示例网络
    example_arch = [784, 256, 128, 64, 10]
    layer_names = ['输入→隐藏1', '隐藏1→隐藏2', '隐藏2→隐藏3', '隐藏3→输出']
    weight_counts = []
    bias_counts = []
    
    for i in range(len(example_arch) - 1):
        weight_counts.append(example_arch[i] * example_arch[i+1])
        bias_counts.append(example_arch[i+1])
    
    ax1 = axes[0]
    x = np.arange(len(layer_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, weight_counts, width, label='权重', color='steelblue')
    bars2 = ax1.bar(x + width/2, bias_counts, width, label='偏置', color='coral')
    
    ax1.set_ylabel('参数数量', fontsize=12)
    ax1.set_title('标准MNIST网络各层参数分布', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_names, rotation=15, ha='right')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 总参数量对比
    ax2 = axes[1]
    total_params_per_arch = []
    arch_labels = []
    
    for arch, desc in architectures:
        total = sum(arch[i] * arch[i+1] + arch[i+1] for i in range(len(arch)-1))
        total_params_per_arch.append(total)
        arch_labels.append(desc)
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = ax2.barh(arch_labels, total_params_per_arch, color=colors, edgecolor='black')
    ax2.set_xlabel('总参数数量（对数尺度）', fontsize=12)
    ax2.set_title('不同网络结构容量对比', fontsize=14)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标注
    for bar, val in zip(bars, total_params_per_arch):
        ax2.text(val, bar.get_y() + bar.get_height()/2,
                f' {val:,}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('network_capacity.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n可视化已保存为 'network_capacity.png'")


# ============================================================================
# 第八部分：主程序
# ============================================================================

def main():
    """
    主程序：运行所有示例
    """
    print("\n" + "=" * 70)
    print("   第十七章：多层神经网络——从零实现MLP")
    print("   《机器学习与深度学习：从小学生到大师》")
    print("=" * 70)
    
    # 1. XOR问题（神经网络的Hello World）
    solve_xor_problem()
    
    # 2. 手写数字识别
    train_digit_classifier()
    
    # 3. 隐藏层激活可视化
    visualize_hidden_activations()
    
    # 4. 网络容量分析
    analyze_network_capacity()
    
    print("\n" + "=" * 70)
    print("   所有示例运行完成！")
    print("   生成的可视化文件:")
    print("     - xor_solution.png")
    print("     - digit_predictions.png")
    print("     - digits_analysis.png")
    print("     - hidden_activations.png")
    print("     - network_capacity.png")
    print("=" * 70)


if __name__ == "__main__":
    main()


@staticmethod
def swish(z):
    """Swish激活函数"""
    return z * Activations.sigmoid(z)

@staticmethod
def swish_derivative(z):
    """Swish的导数"""
    sig = Activations.sigmoid(z)
    return sig + z * sig * (1 - sig)  # swish'(x) = sigmoid(x) + x * sigmoid'(x)

@staticmethod
def gelu(z):
    """GELU激活函数（近似实现）"""
    return 0.5 * z * (1 + np.tanh(
        np.sqrt(2/np.pi) * (z + 0.044715 * z**3)
    ))

@staticmethod  
def gelu_derivative(z):
    """GELU的导数（数值近似）"""
    # 可以使用数值微分或更复杂的解析表达式
    eps = 1e-5
    return (Activations.gelu(z + eps) - Activations.gelu(z - eps)) / (2 * eps)

