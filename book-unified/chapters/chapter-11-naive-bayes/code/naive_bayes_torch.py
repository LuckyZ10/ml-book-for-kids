"""
第十一章：朴素贝叶斯分类器 - PyTorch实现
《机器学习与深度学习：从小学生到大师》

本模块使用PyTorch实现朴素贝叶斯分类器，展示如何结合深度学习框架
实现传统机器学习算法。

包含：
1. GaussianNBTorch - PyTorch实现的高斯朴素贝叶斯
2. MultinomialNBTorch - PyTorch实现的多项式朴素贝叶斯
3. 使用神经网络风格的训练流程
4. GPU加速支持（如果可用）
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, Counter
import math
from typing import List, Tuple, Optional, Dict


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ============================================================================
# 第一部分：PyTorch实现的高斯朴素贝叶斯
# ============================================================================

class GaussianNBTorch:
    """
    使用PyTorch实现的高斯朴素贝叶斯分类器
    
    优势：
    - 可以利用GPU加速计算
    - 支持批量处理
    - 与PyTorch生态兼容
    """
    
    def __init__(self, priors: Optional[torch.Tensor] = None, var_smoothing: float = 1e-9):
        """
        初始化
        
        参数:
            priors: 类别先验概率
            var_smoothing: 方差平滑参数
        """
        self.priors = priors
        self.var_smoothing = var_smoothing
        self.classes_: Optional[torch.Tensor] = None
        self.class_prior_: Optional[torch.Tensor] = None
        self.theta_: Optional[torch.Tensor] = None  # 均值
        self.var_: Optional[torch.Tensor] = None    # 方差
        self.epsilon_: Optional[float] = None
        self.n_classes_: int = 0
        self.n_features_: int = 0
        
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> 'GaussianNBTorch':
        """
        训练模型
        
        参数:
            X: 训练数据，shape (n_samples, n_features)
            y: 目标标签，shape (n_samples,)
        """
        # 确保数据在正确的设备上
        X = X.to(device)
        y = y.to(device)
        
        # 获取所有类别
        self.classes_ = torch.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        
        # 初始化参数
        self.theta_ = torch.zeros((self.n_classes_, self.n_features_), device=device)
        self.var_ = torch.zeros((self.n_classes_, self.n_features_), device=device)
        self.class_prior_ = torch.zeros(self.n_classes_, device=device)
        
        # 计算每个类别的统计量
        for idx, c in enumerate(self.classes_):
            mask = y == c
            X_c = X[mask]  # 类别c的所有样本
            
            # 计算均值和方差
            self.theta_[idx, :] = torch.mean(X_c, dim=0)
            self.var_[idx, :] = torch.var(X_c, dim=0, unbiased=False)
            
            # 计算先验概率
            self.class_prior_[idx] = len(X_c) / len(X)
        
        # 如果提供了先验概率，则使用提供的
        if self.priors is not None:
            self.class_prior_ = self.priors.to(device)
        
        # 方差平滑（数值稳定性）
        self.epsilon_ = self.var_smoothing * torch.var(X, dim=0).max().item()
        self.var_ += self.epsilon_
        
        return self
    
    def _calculate_log_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        """
        计算对数似然
        """
        X = X.to(device)
        n_samples = X.shape[0]
        
        # 存储每个样本在每个类别下的对数似然
        log_likelihood = torch.zeros((n_samples, self.n_classes_), device=device)
        
        for idx in range(self.n_classes_):
            # 获取当前类别的均值和方差
            mean = self.theta_[idx, :]
            var = self.var_[idx, :]
            
            # 计算高斯对数概率密度
            # log P(x|C) = -0.5 * log(2*pi*var) - 0.5 * (x-mean)^2 / var
            log_prob = -0.5 * torch.sum(torch.log(2 * math.pi * var))
            log_prob -= 0.5 * torch.sum(((X - mean) ** 2) / var, dim=1)
            
            # 加上先验概率的对数
            log_likelihood[:, idx] = log_prob + torch.log(self.class_prior_[idx])
        
        return log_likelihood
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        预测类别标签
        """
        X = X.to(device)
        log_likelihood = self._calculate_log_likelihood(X)
        
        # 选择对数似然最大的类别
        return self.classes_[torch.argmax(log_likelihood, dim=1)]
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        预测类别概率
        """
        X = X.to(device)
        log_likelihood = self._calculate_log_likelihood(X)
        
        # 使用log-sum-exp技巧进行数值稳定性处理
        log_likelihood_max = torch.max(log_likelihood, dim=1, keepdim=True)[0]
        likelihood = torch.exp(log_likelihood - log_likelihood_max)
        
        # 归一化得到概率
        prob = likelihood / torch.sum(likelihood, dim=1, keepdim=True)
        return prob
    
    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        计算准确率
        """
        predictions = self.predict(X)
        return (predictions == y.to(device)).float().mean().item()


# ============================================================================
# 第二部分：PyTorch实现的多项式朴素贝叶斯
# ============================================================================

class MultinomialNBTorch:
    """
    使用PyTorch实现的多项式朴素贝叶斯分类器
    
    适用于文本分类等离散计数特征
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        初始化
        
        参数:
            alpha: 拉普拉斯平滑参数
        """
        self.alpha = alpha
        self.classes_: Optional[torch.Tensor] = None
        self.class_count_: Optional[torch.Tensor] = None
        self.feature_log_prob_: Optional[torch.Tensor] = None
        self.class_log_prior_: Optional[torch.Tensor] = None
        self.n_classes_: int = 0
        self.n_features_: int = 0
        
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> 'MultinomialNBTorch':
        """
        训练模型
        
        参数:
            X: 训练数据，shape (n_samples, n_features)
            y: 目标标签，shape (n_samples,)
        """
        X = X.to(device)
        y = y.to(device)
        
        # 确保特征非负
        if torch.any(X < 0):
            raise ValueError("输入X必须为非负数")
        
        # 获取所有类别
        self.classes_ = torch.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        
        # 初始化
        self.class_count_ = torch.zeros(self.n_classes_, device=device)
        feature_count = torch.zeros((self.n_classes_, self.n_features_), device=device)
        
        # 计算每个类别的统计量
        for idx, c in enumerate(self.classes_):
            mask = y == c
            X_c = X[mask]  # 类别c的所有样本
            feature_count[idx, :] = torch.sum(X_c, dim=0)
            self.class_count_[idx] = torch.sum(feature_count[idx, :])
        
        # 计算平滑后的条件概率的对数
        smoothed_fc = feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(dim=1) + self.alpha * self.n_features_
        
        self.feature_log_prob_ = torch.log(smoothed_fc) - torch.log(smoothed_cc.unsqueeze(1))
        
        # 计算先验概率
        self.class_log_prior_ = torch.log(self.class_count_ / torch.sum(self.class_count_))
        
        return self
    
    def _joint_log_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        """
        计算联合对数似然
        """
        X = X.to(device)
        return torch.mm(X, self.feature_log_prob_.t()) + self.class_log_prior_
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        预测类别标签
        """
        X = X.to(device)
        joint_log_likelihood = self._joint_log_likelihood(X)
        return self.classes_[torch.argmax(joint_log_likelihood, dim=1)]
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        预测类别概率
        """
        X = X.to(device)
        joint_log_likelihood = self._joint_log_likelihood(X)
        
        # 使用log-sum-exp技巧
        log_prob_max = torch.max(joint_log_likelihood, dim=1, keepdim=True)[0]
        prob = torch.exp(joint_log_likelihood - log_prob_max)
        prob = prob / torch.sum(prob, dim=1, keepdim=True)
        
        return prob
    
    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        计算准确率
        """
        predictions = self.predict(X)
        return (predictions == y.to(device)).float().mean().item()


# ============================================================================
# 第三部分：神经网络风格的朴素贝叶斯（使用nn.Module）
# ============================================================================

class NaiveBayesNN(nn.Module):
    """
    使用神经网络模块实现的朴素贝叶斯
    
    这是一个概念性实现，展示如何将朴素贝叶斯与神经网络框架结合。
    实际上，朴素贝叶斯不需要反向传播训练。
    """
    
    def __init__(self, n_features: int, n_classes: int, model_type: str = 'gaussian'):
        """
        初始化
        
        参数:
            n_features: 特征数量
            n_classes: 类别数量
            model_type: 'gaussian' 或 'multinomial'
        """
        super(NaiveBayesNN, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.model_type = model_type
        
        # 对于高斯模型，存储均值和方差
        if model_type == 'gaussian':
            # 均值参数
            self.theta = nn.Parameter(torch.zeros(n_classes, n_features))
            # 方差参数（使用softplus确保正数）
            self.log_var = nn.Parameter(torch.zeros(n_classes, n_features))
        
        # 类别先验
        self.class_log_prior = nn.Parameter(torch.zeros(n_classes))
        
        # 是否已训练
        self.is_fitted = False
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        前向传播（计算对数似然）
        
        参数:
            X: 输入数据，shape (batch_size, n_features)
            
        返回:
            对数似然，shape (batch_size, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练")
        
        if self.model_type == 'gaussian':
            # 计算高斯对数概率
            var = torch.nn.functional.softplus(self.log_var) + 1e-9
            
            # log P(x|C) = -0.5 * log(2*pi*var) - 0.5 * (x-mean)^2 / var
            log_likelihood = torch.zeros(X.shape[0], self.n_classes, device=X.device)
            
            for c in range(self.n_classes):
                mean = self.theta[c, :]
                variance = var[c, :]
                
                log_prob = -0.5 * torch.sum(torch.log(2 * math.pi * variance))
                log_prob -= 0.5 * torch.sum(((X - mean) ** 2) / variance, dim=1)
                log_likelihood[:, c] = log_prob
            
            # 加上先验
            return log_likelihood + self.class_log_prior
        else:
            raise NotImplementedError("仅支持高斯模型")
    
    def fit_gaussian(self, X: torch.Tensor, y: torch.Tensor):
        """
        为高斯模型计算参数（不使用梯度下降）
        """
        X = X.to(device)
        y = y.to(device)
        
        classes = torch.unique(y)
        
        with torch.no_grad():
            for idx, c in enumerate(classes):
                mask = y == c
                X_c = X[mask]
                
                # 计算均值和方差
                self.theta[idx, :] = torch.mean(X_c, dim=0)
                var = torch.var(X_c, dim=0, unbiased=False) + 1e-9
                self.log_var[idx, :] = torch.log(var)
                
                # 计算先验
                self.class_log_prior[idx] = torch.log(torch.tensor(len(X_c) / len(X)))
        
        self.is_fitted = True
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        预测类别
        """
        log_likelihood = self.forward(X)
        return torch.argmax(log_likelihood, dim=1)
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        预测概率
        """
        log_likelihood = self.forward(X)
        log_prob_max = torch.max(log_likelihood, dim=1, keepdim=True)[0]
        prob = torch.exp(log_likelihood - log_prob_max)
        return prob / torch.sum(prob, dim=1, keepdim=True)


# ============================================================================
# 第四部分：数据加载和预处理工具
# ============================================================================

class TextVectorizer:
    """
    文本向量化器，将文本转换为词频向量
    """
    
    def __init__(self, max_features: int = 10000, min_df: int = 1):
        """
        初始化
        
        参数:
            max_features: 最大特征数量
            min_df: 最小文档频率
        """
        self.max_features = max_features
        self.min_df = min_df
        self.vocab: Dict[str, int] = {}
        self.vocab_list: List[str] = []
        
    def fit(self, texts: List[str]):
        """
        构建词汇表
        """
        word_counts = Counter()
        doc_counts = Counter()
        
        for text in texts:
            words = set(text.split())
            for word in words:
                doc_counts[word] += 1
        
        for text in texts:
            words = text.split()
            for word in words:
                if doc_counts[word] >= self.min_df:
                    word_counts[word] += 1
        
        # 选择最常见的词
        most_common = word_counts.most_common(self.max_features)
        self.vocab_list = [word for word, _ in most_common]
        self.vocab = {word: idx for idx, word in enumerate(self.vocab_list)}
        
        return self
    
    def transform(self, texts: List[str]) -> torch.Tensor:
        """
        将文本转换为词频向量
        """
        vectors = torch.zeros((len(texts), len(self.vocab)))
        
        for i, text in enumerate(texts):
            words = text.split()
            for word in words:
                if word in self.vocab:
                    vectors[i, self.vocab[word]] += 1
        
        return vectors
    
    def fit_transform(self, texts: List[str]) -> torch.Tensor:
        """
        拟合并转换
        """
        self.fit(texts)
        return self.transform(texts)


# ============================================================================
# 第五部分：训练流程和评估
# ============================================================================

class NaiveBayesTrainer:
    """
    朴素贝叶斯训练器，提供类似PyTorch的训练流程
    """
    
    def __init__(self, model, model_type: str = 'gaussian'):
        """
        初始化
        
        参数:
            model: 模型实例
            model_type: 模型类型
        """
        self.model = model
        self.model_type = model_type
        self.history = {'train_acc': [], 'val_acc': []}
    
    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor, 
            X_val: Optional[torch.Tensor] = None, 
            y_val: Optional[torch.Tensor] = None):
        """
        训练模型
        
        朴素贝叶斯不需要迭代训练，但为了保持接口一致性，提供此方法
        """
        print("训练朴素贝叶斯模型...")
        
        # 训练模型
        if self.model_type == 'gaussian':
            if isinstance(self.model, NaiveBayesNN):
                self.model.fit_gaussian(X_train, y_train)
            else:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        # 计算训练准确率
        train_acc = self.model.score(X_train, y_train)
        self.history['train_acc'].append(train_acc)
        
        # 计算验证准确率
        if X_val is not None and y_val is not None:
            val_acc = self.model.score(X_val, y_val)
            self.history['val_acc'].append(val_acc)
            print(f"训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}")
        else:
            print(f"训练准确率: {train_acc:.4f}")
        
        return self
    
    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, float]:
        """
        评估模型
        """
        accuracy = self.model.score(X_test, y_test)
        predictions = self.model.predict(X_test)
        
        # 计算每个类别的准确率和召回率
        classes = torch.unique(y_test)
        per_class_metrics = {}
        
        for c in classes:
            mask = y_test == c
            true_positive = ((predictions == c) & mask).sum().item()
            false_positive = ((predictions == c) & ~mask).sum().item()
            false_negative = ((predictions != c) & mask).sum().item()
            
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            
            per_class_metrics[f'class_{c.item()}_precision'] = precision
            per_class_metrics[f'class_{c.item()}_recall'] = recall
        
        metrics = {'accuracy': accuracy, **per_class_metrics}
        return metrics


# ============================================================================
# 第六部分：鸢尾花分类示例（PyTorch版本）
# ============================================================================

def iris_example_torch():
    """
    使用PyTorch实现的朴素贝叶斯进行鸢尾花分类
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    print("=" * 60)
    print("鸢尾花分类示例 - PyTorch版高斯朴素贝叶斯")
    print("=" * 60)
    
    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 转换为PyTorch张量
    X_train_torch = torch.FloatTensor(X_train)
    y_train_torch = torch.LongTensor(y_train)
    X_test_torch = torch.FloatTensor(X_test)
    y_test_torch = torch.LongTensor(y_test)
    
    # 创建并训练模型
    model = GaussianNBTorch()
    trainer = NaiveBayesTrainer(model, model_type='gaussian')
    trainer.fit(X_train_torch, y_train_torch, X_test_torch, y_test_torch)
    
    # 评估
    metrics = trainer.evaluate(X_test_torch, y_test_torch)
    print(f"\n详细指标:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 使用神经网络风格实现
    print("\n使用神经网络风格实现:")
    model_nn = NaiveBayesNN(n_features=4, n_classes=3)
    model_nn = model_nn.to(device)
    trainer_nn = NaiveBayesTrainer(model_nn, model_type='gaussian')
    trainer_nn.fit(X_train_torch, y_train_torch, X_test_torch, y_test_torch)
    
    return metrics['accuracy']


# ============================================================================
# 第七部分：文本分类示例（PyTorch版本）
# ============================================================================

def text_example_torch():
    """
    使用PyTorch实现的朴素贝叶斯进行文本分类
    """
    print("\n" + "=" * 60)
    print("文本分类示例 - PyTorch版多项式朴素贝叶斯")
    print("=" * 60)
    
    # 模拟数据
    train_texts = [
        "machine learning algorithm", "deep learning neural network",
        "computer graphics rendering", "image processing computer vision",
        "natural language processing", "text classification sentiment analysis"
    ] * 10
    
    # 0: ML/DL, 1: Graphics, 2: NLP
    train_labels = [0, 0, 1, 1, 2, 2] * 10
    
    test_texts = [
        "machine learning model",
        "graphics rendering algorithm",
        "natural language text"
    ]
    test_labels = [0, 1, 2]
    
    # 向量化
    vectorizer = TextVectorizer(max_features=50)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    y_train = torch.LongTensor(train_labels)
    y_test = torch.LongTensor(test_labels)
    
    # 创建并训练模型
    model = MultinomialNBTorch(alpha=1.0)
    trainer = NaiveBayesTrainer(model, model_type='multinomial')
    trainer.fit(X_train, y_train)
    
    # 评估
    accuracy = model.score(X_test, y_test)
    print(f"\n测试准确率: {accuracy:.4f}")
    
    # 预测概率
    proba = model.predict_proba(X_test)
    print(f"\n预测概率:")
    for i, (text, prob) in enumerate(zip(test_texts, proba)):
        print(f"  '{text}': {prob.cpu().numpy()}")
    
    return accuracy


# ============================================================================
# 第八部分：性能对比（CPU vs GPU）
# ============================================================================

def performance_comparison():
    """
    对比CPU和GPU的性能
    """
    print("\n" + "=" * 60)
    print("性能对比: CPU vs GPU")
    print("=" * 60)
    
    # 生成大规模数据
    n_samples = 10000
    n_features = 100
    n_classes = 5
    
    # 生成随机数据
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    
    X_torch = torch.FloatTensor(X)
    y_torch = torch.LongTensor(y)
    
    # CPU测试
    print("\nCPU性能测试:")
    import time
    
    model_cpu = GaussianNBTorch()
    start = time.time()
    model_cpu.fit(X_torch, y_torch)
    train_time_cpu = time.time() - start
    
    start = time.time()
    _ = model_cpu.predict(X_torch)
    predict_time_cpu = time.time() - start
    
    print(f"  训练时间: {train_time_cpu:.4f}秒")
    print(f"  预测时间: {predict_time_cpu:.4f}秒")
    
    # GPU测试（如果可用）
    if torch.cuda.is_available():
        print("\nGPU性能测试:")
        X_gpu = X_torch.cuda()
        y_gpu = y_torch.cuda()
        
        model_gpu = GaussianNBTorch()
        start = time.time()
        model_gpu.fit(X_gpu, y_gpu)
        train_time_gpu = time.time() - start
        
        start = time.time()
        _ = model_gpu.predict(X_gpu)
        predict_time_gpu = time.time() - start
        
        print(f"  训练时间: {train_time_gpu:.4f}秒")
        print(f"  预测时间: {predict_time_gpu:.4f}秒")
        print(f"\nGPU加速比 - 训练: {train_time_cpu/train_time_gpu:.2f}x, 预测: {predict_time_cpu/predict_time_gpu:.2f}x")
    else:
        print("\nGPU不可用")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("朴素贝叶斯分类器 - PyTorch实现")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"设备: {device}")
    print("=" * 60)
    
    # 运行示例
    iris_example_torch()
    text_example_torch()
    performance_comparison()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)
