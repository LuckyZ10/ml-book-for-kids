"""
第十一章：朴素贝叶斯分类器 - 纯Python实现
《机器学习与深度学习：从小学生到大师》

本模块包含：
1. GaussianNB - 高斯朴素贝叶斯（连续特征）
2. MultinomialNB - 多项式朴素贝叶斯（文本/离散特征）
3. BernoulliNB - 伯努利朴素贝叶斯（二元特征）
4. 垃圾邮件分类器示例
5. 手写数字识别示例
"""

import numpy as np
import math
from collections import defaultdict, Counter
import random


# ============================================================================
# 第一部分：高斯朴素贝叶斯（Gaussian Naive Bayes）
# ============================================================================

class GaussianNB:
    """
    高斯朴素贝叶斯分类器
    
    适用于连续特征，假设每个特征在每个类别内服从正态分布。
    
    参数:
        priors: 类别的先验概率，如果为None则从数据中学习
        var_smoothing: 方差平滑参数，用于数值稳定性
    """
    
    def __init__(self, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None  # 每个类别每个特征的均值
        self.var_ = None    # 每个类别每个特征的方差
        self.epsilon_ = None
        
    def fit(self, X, y):
        """
        训练高斯朴素贝叶斯分类器
        
        参数:
            X: 训练数据，shape (n_samples, n_features)
            y: 目标标签，shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        
        # 获取所有类别
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # 初始化参数
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)
        
        # 计算每个类别的统计量
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]  # 类别c的所有样本
            
            # 计算均值和方差
            self.theta_[idx, :] = np.mean(X_c, axis=0)
            self.var_[idx, :] = np.var(X_c, axis=0)
            
            # 计算先验概率
            self.class_prior_[idx] = len(X_c) / len(X)
        
        # 如果提供了先验概率，则使用提供的
        if self.priors is not None:
            self.class_prior_ = np.array(self.priors)
        
        # 方差平滑（数值稳定性）
        self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()
        self.var_ += self.epsilon_
        
        return self
    
    def _calculate_log_likelihood(self, X):
        """
        计算对数似然
        
        对于高斯分布：
        log P(x|C) = -0.5 * log(2*pi*var) - 0.5 * (x-mean)^2 / var
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # 存储每个样本在每个类别下的对数似然
        log_likelihood = np.zeros((n_samples, n_classes))
        
        for idx in range(n_classes):
            # 获取当前类别的均值和方差
            mean = self.theta_[idx, :]
            var = self.var_[idx, :]
            
            # 计算高斯对数概率密度
            # log P(x|C) = -0.5 * log(2*pi*var) - 0.5 * (x-mean)^2 / var
            log_prob = -0.5 * np.sum(np.log(2 * np.pi * var))
            log_prob -= 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            
            # 加上先验概率的对数
            log_likelihood[:, idx] = log_prob + np.log(self.class_prior_[idx])
        
        return log_likelihood
    
    def predict(self, X):
        """
        预测类别标签
        """
        X = np.array(X)
        log_likelihood = self._calculate_log_likelihood(X)
        
        # 选择对数似然最大的类别
        return self.classes_[np.argmax(log_likelihood, axis=1)]
    
    def predict_proba(self, X):
        """
        预测类别概率
        """
        X = np.array(X)
        log_likelihood = self._calculate_log_likelihood(X)
        
        # 使用log-sum-exp技巧进行数值稳定性处理
        # 减去最大值防止指数爆炸
        log_likelihood_max = np.max(log_likelihood, axis=1, keepdims=True)
        likelihood = np.exp(log_likelihood - log_likelihood_max)
        
        # 归一化得到概率
        prob = likelihood / np.sum(likelihood, axis=1, keepdims=True)
        return prob
    
    def score(self, X, y):
        """
        计算准确率
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


# ============================================================================
# 第二部分：多项式朴素贝叶斯（Multinomial Naive Bayes）
# ============================================================================

class MultinomialNB:
    """
    多项式朴素贝叶斯分类器
    
    适用于离散计数特征，特别适用于文本分类。
    使用词频作为特征，假设特征服从多项式分布。
    
    参数:
        alpha: 拉普拉斯平滑参数，默认为1.0（加一平滑）
        fit_prior: 是否从数据中学习先验概率
        class_prior: 指定的类别先验概率
    """
    
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha  # 平滑参数
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.classes_ = None
        self.class_count_ = None
        self.feature_count_ = None
        self.feature_log_prob_ = None
        self.class_log_prior_ = None
        
    def fit(self, X, y):
        """
        训练多项式朴素贝叶斯分类器
        
        参数:
            X: 训练数据，shape (n_samples, n_features)，非负整数
            y: 目标标签，shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        
        # 确保X是非负整数
        if np.any(X < 0):
            raise ValueError("输入X必须包含非负整数（词频计数）")
        
        # 获取所有类别
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # 初始化计数器
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)
        
        # 统计每个类别的特征计数
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_count_[idx] = X_c.shape[0]
            self.feature_count_[idx, :] = np.sum(X_c, axis=0)
        
        # 应用拉普拉斯平滑计算特征对数概率
        # P(w|C) = (N_{w,C} + alpha) / (N_C + alpha * n_features)
        smoothed_feature_count = self.feature_count_ + self.alpha
        smoothed_class_count = np.sum(smoothed_feature_count, axis=1)
        
        self.feature_log_prob_ = np.log(smoothed_feature_count) - \
                                 np.log(smoothed_class_count.reshape(-1, 1))
        
        # 计算类别先验概率
        if self.fit_prior:
            self.class_log_prior_ = np.log(self.class_count_) - \
                                    np.log(np.sum(self.class_count_))
        else:
            if self.class_prior is not None:
                self.class_log_prior_ = np.log(self.class_prior)
            else:
                self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))
        
        return self
    
    def predict(self, X):
        """
        预测类别标签
        """
        X = np.array(X)
        # 计算联合对数似然
        joint_log_likelihood = np.dot(X, self.feature_log_prob_.T) + \
                               self.class_log_prior_
        
        # 选择对数似然最大的类别
        return self.classes_[np.argmax(joint_log_likelihood, axis=1)]
    
    def predict_proba(self, X):
        """
        预测类别概率
        """
        X = np.array(X)
        joint_log_likelihood = np.dot(X, self.feature_log_prob_.T) + \
                               self.class_log_prior_
        
        # log-sum-exp技巧
        log_likelihood_max = np.max(joint_log_likelihood, axis=1, keepdims=True)
        likelihood = np.exp(joint_log_likelihood - log_likelihood_max)
        
        prob = likelihood / np.sum(likelihood, axis=1, keepdims=True)
        return prob
    
    def score(self, X, y):
        """
        计算准确率
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


# ============================================================================
# 第三部分：伯努利朴素贝叶斯（Bernoulli Naive Bayes）
# ============================================================================

class BernoulliNB:
    """
    伯努利朴素贝叶斯分类器
    
    适用于二元/布尔特征，即只关心特征是否存在（0/1）。
    特别适用于短文本分类和二值化图像。
    
    参数:
        alpha: 拉普拉斯平滑参数，默认为1.0
        binarize: 二值化阈值，如果为None则假设输入已经是二值的
        fit_prior: 是否从数据中学习先验概率
        class_prior: 指定的类别先验概率
    """
    
    def __init__(self, alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.classes_ = None
        self.feature_count_ = None
        self.class_count_ = None
        self.feature_log_prob_ = None
        self.feature_log_prob_neg_ = None  # 特征不出现的对数概率
        self.class_log_prior_ = None
        
    def _binarize_features(self, X):
        """将特征二值化"""
        if self.binarize is not None:
            return (X > self.binarize).astype(int)
        return X
    
    def fit(self, X, y):
        """
        训练伯努利朴素贝叶斯分类器
        
        参数:
            X: 训练数据，shape (n_samples, n_features)
            y: 目标标签，shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        
        # 二值化特征
        X = self._binarize_features(X)
        
        # 确保是二元特征
        if not np.all(np.logical_or(X == 0, X == 1)):
            raise ValueError("输入X必须是二元特征（0或1）")
        
        # 获取所有类别
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # 初始化计数器
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)
        
        # 统计每个类别中每个特征出现的文档数
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_count_[idx] = X_c.shape[0]
            # 统计特征出现的文档数（不是总次数）
            self.feature_count_[idx, :] = np.sum(X_c, axis=0)
        
        # 应用拉普拉斯平滑
        # P(x_i=1|C) = (特征i在类别C中出现的文档数 + alpha) / (类别C的文档数 + 2*alpha)
        smoothed_feature_count = self.feature_count_ + self.alpha
        smoothed_class_count = self.class_count_ + 2 * self.alpha
        
        # 特征出现的概率
        feature_prob = smoothed_feature_count / smoothed_class_count.reshape(-1, 1)
        self.feature_log_prob_ = np.log(feature_prob)
        self.feature_log_prob_neg_ = np.log(1 - feature_prob)
        
        # 计算类别先验概率
        if self.fit_prior:
            self.class_log_prior_ = np.log(self.class_count_) - \
                                    np.log(np.sum(self.class_count_))
        else:
            if self.class_prior is not None:
                self.class_log_prior_ = np.log(self.class_prior)
            else:
                self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))
        
        return self
    
    def predict(self, X):
        """
        预测类别标签
        """
        X = np.array(X)
        X = self._binarize_features(X)
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # 计算联合对数似然
        # 对于伯努利模型，需要考虑特征出现和不出现两种情况
        joint_log_likelihood = np.zeros((n_samples, n_classes))
        
        for idx in range(n_classes):
            # 特征出现的贡献
            present = X * self.feature_log_prob_[idx, :]
            # 特征不出现的贡献
            absent = (1 - X) * self.feature_log_prob_neg_[idx, :]
            
            joint_log_likelihood[:, idx] = np.sum(present + absent, axis=1) + \
                                           self.class_log_prior_[idx]
        
        # 选择对数似然最大的类别
        return self.classes_[np.argmax(joint_log_likelihood, axis=1)]
    
    def predict_proba(self, X):
        """
        预测类别概率
        """
        X = np.array(X)
        X = self._binarize_features(X)
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        joint_log_likelihood = np.zeros((n_samples, n_classes))
        
        for idx in range(n_classes):
            present = X * self.feature_log_prob_[idx, :]
            absent = (1 - X) * self.feature_log_prob_neg_[idx, :]
            
            joint_log_likelihood[:, idx] = np.sum(present + absent, axis=1) + \
                                           self.class_log_prior_[idx]
        
        # log-sum-exp技巧
        log_likelihood_max = np.max(joint_log_likelihood, axis=1, keepdims=True)
        likelihood = np.exp(joint_log_likelihood - log_likelihood_max)
        
        prob = likelihood / np.sum(likelihood, axis=1, keepdims=True)
        return prob
    
    def score(self, X, y):
        """
        计算准确率
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


# ============================================================================
# 第四部分：文本预处理工具
# ============================================================================

class SimpleCountVectorizer:
    """
    简单的词频向量化器（词袋模型）
    
    将文本转换为词频矩阵
    """
    
    def __init__(self, max_features=None, min_df=1):
        self.max_features = max_features
        self.min_df = min_df  # 最小文档频率
        self.vocabulary_ = {}
        self.feature_names_ = []
        
    def fit(self, raw_documents):
        """
        学习词汇表
        """
        # 统计词频
        word_counts = Counter()
        doc_counts = Counter()
        
        for doc in raw_documents:
            words = self._tokenize(doc)
            word_counts.update(words)
            doc_counts.update(set(words))
        
        # 过滤低频词
        valid_words = [w for w, count in doc_counts.items() 
                       if count >= self.min_df]
        
        # 如果设置了最大特征数，选择最常见的词
        if self.max_features is not None:
            valid_words = [w for w, _ in word_counts.most_common(self.max_features)
                          if w in valid_words]
        
        # 构建词汇表
        self.vocabulary_ = {word: idx for idx, word in enumerate(valid_words)}
        self.feature_names_ = valid_words
        
        return self
    
    def transform(self, raw_documents):
        """
        将文本转换为词频矩阵
        """
        n_samples = len(raw_documents)
        n_features = len(self.vocabulary_)
        X = np.zeros((n_samples, n_features), dtype=np.int32)
        
        for i, doc in enumerate(raw_documents):
            words = self._tokenize(doc)
            for word in words:
                if word in self.vocabulary_:
                    X[i, self.vocabulary_[word]] += 1
        
        return X
    
    def fit_transform(self, raw_documents):
        """
        学习词汇表并转换文本
        """
        self.fit(raw_documents)
        return self.transform(raw_documents)
    
    def _tokenize(self, text):
        """
        简单的分词：转小写，按空格分割，去除标点
        """
        # 转换为小写
        text = text.lower()
        # 简单处理标点
        for char in '.,!?;:()[]{}"\'':
            text = text.replace(char, ' ')
        # 分割成词
        return text.split()


# ============================================================================
# 第五部分：垃圾邮件分类器示例
# ============================================================================

def demo_spam_classifier():
    """
    垃圾邮件分类器完整示例
    
    使用多项式朴素贝叶斯进行垃圾邮件检测
    """
    print("=" * 60)
    print("垃圾邮件分类器示例 - 多项式朴素贝叶斯")
    print("=" * 60)
    
    # 示例数据集
    emails = [
        # 垃圾邮件 (spam)
        ("免费获得百万大奖 点击链接立即领取", "spam"),
        ("恭喜您中奖了 请回复领取奖品", "spam"),
        ("限时优惠 买二送一 速来抢购", "spam"),
        ("免费送货 特价促销 不买后悔", "spam"),
        ("您被选为幸运用户 免费获得iPhone", "spam"),
        ("点击赢取现金大奖 百分百中奖", "spam"),
        ("免费试用 立即注册 赠送礼品", "spam"),
        ("最后机会 限时折扣 错过不再有", "spam"),
        ("您有一笔待领取的奖金 请确认", "spam"),
        ("特价机票 酒店优惠 旅游套餐", "spam"),
        
        # 正常邮件 (ham)
        ("你好 明天的会议定在下午三点", "ham"),
        ("请查收附件中的项目报告", "ham"),
        ("周末有空一起去爬山吗", "ham"),
        ("妈妈问我今晚回家吃饭吗", "ham"),
        ("这份文档需要你审核一下", "ham"),
        ("生日快乐 祝你新的一岁开心", "ham"),
        ("明天降温记得多穿衣服", "ham"),
        ("论文修改意见已经发给你了", "ham"),
        ("周末的聚餐你来吗", "ham"),
        ("新的项目计划请查收", "ham"),
    ]
    
    # 分离文本和标签
    texts = [email[0] for email in emails]
    labels = [email[1] for email in emails]
    
    # 划分训练集和测试集（简化处理：前8个spam和前8个ham作为训练集）
    train_texts = texts[:8] + texts[10:18]
    train_labels = labels[:8] + labels[10:18]
    test_texts = texts[8:10] + texts[18:20]
    test_labels = labels[8:10] + labels[18:20]
    
    print(f"\n训练集大小: {len(train_texts)}")
    print(f"测试集大小: {len(test_texts)}")
    
    # 文本向量化
    vectorizer = SimpleCountVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    print(f"\n词汇表大小: {len(vectorizer.vocabulary_)}")
    print(f"特征（词汇）: {vectorizer.feature_names_[:10]}...")
    
    # 转换标签为数字
    label_map = {"spam": 0, "ham": 1}
    y_train = np.array([label_map[l] for l in train_labels])
    y_test = np.array([label_map[l] for l in test_labels])
    
    # 训练多项式朴素贝叶斯
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train, y_train)
    
    # 预测
    predictions = clf.predict(X_test)
    probabilities = clf.predict_proba(X_test)
    
    # 评估
    print("\n" + "-" * 40)
    print("预测结果:")
    print("-" * 40)
    
    reverse_label_map = {0: "spam", 1: "ham"}
    for i, (text, true_label, pred, prob) in enumerate(zip(
        test_texts, test_labels, predictions, probabilities)):
        pred_label = reverse_label_map[pred]
        print(f"\n邮件: {text[:30]}...")
        print(f"真实标签: {true_label}")
        print(f"预测标签: {pred_label}")
        print(f"概率: spam={prob[0]:.4f}, ham={prob[1]:.4f}")
    
    accuracy = np.mean(predictions == y_test)
    print(f"\n准确率: {accuracy * 100:.1f}%")
    
    # 显示每个类别最具代表性的词
    print("\n" + "-" * 40)
    print("最具代表性的词汇:")
    print("-" * 40)
    
    feature_names = vectorizer.feature_names_
    
    # 获取特征对数概率
    feature_log_prob = clf.feature_log_prob_
    
    # spam类别（索引0）
    spam_top_indices = np.argsort(feature_log_prob[0])[-5:][::-1]
    print("\n垃圾邮件(spam)代表性词汇:")
    for idx in spam_top_indices:
        print(f"  {feature_names[idx]}: {np.exp(feature_log_prob[0][idx]):.4f}")
    
    # ham类别（索引1）
    ham_top_indices = np.argsort(feature_log_prob[1])[-5:][::-1]
    print("\n正常邮件(ham)代表性词汇:")
    for idx in ham_top_indices:
        print(f"  {feature_names[idx]}: {np.exp(feature_log_prob[1][idx]):.4f}")
    
    return clf, vectorizer


# ============================================================================
# 第六部分：手写数字识别示例
# ============================================================================

def generate_mnist_like_data(n_samples_per_class=100, noise=0.1):
    """
    生成类似MNIST的合成数据（简化版）
    
    生成0-9的数字图像，每个图像是8x8的二值图像
    """
    np.random.seed(42)
    
    # 定义每个数字的基本模式（简化的8x8图案）
    patterns = {
        0: [
            "..####..",
            ".##..##.",
            ".##..##.",
            ".##..##.",
            ".##..##.",
            ".##..##.",
            ".##..##.",
            "..####.."
        ],
        1: [
            "...##...",
            "..###...",
            "...##...",
            "...##...",
            "...##...",
            "...##...",
            "...##...",
            ".######."
        ],
        2: [
            "..####..",
            ".##..##.",
            ".....##.",
            "....##..",
            "...##...",
            "..##....",
            ".##.....",
            ".#######"
        ],
        3: [
            "..####..",
            ".##..##.",
            ".....##.",
            "...###..",
            "...###..",
            ".....##.",
            ".##..##.",
            "..####.."
        ],
        4: [
            "....##..",
            "...###..",
            "..####..",
            ".##.##..",
            ".#######",
            "....##..",
            "....##..",
            "....##.."
        ],
        5: [
            ".#######",
            ".##.....",
            ".##.....",
            ".#####..",
            ".....##.",
            ".....##.",
            ".##..##.",
            "..####.."
        ],
        6: [
            "..####..",
            ".##..##.",
            ".##.....",
            ".#####..",
            ".##..##.",
            ".##..##.",
            ".##..##.",
            "..####.."
        ],
        7: [
            ".#######",
            ".....##.",
            "....##..",
            "...##...",
            "..##....",
            ".##.....",
            ".##.....",
            ".##....."
        ],
        8: [
            "..####..",
            ".##..##.",
            ".##..##.",
            "..####..",
            ".##..##.",
            ".##..##.",
            ".##..##.",
            "..####.."
        ],
        9: [
            "..####..",
            ".##..##.",
            ".##..##.",
            ".##..##.",
            "..#####.",
            ".....##.",
            ".##..##.",
            "..####.."
        ]
    }
    
    X = []
    y = []
    
    for digit in range(10):
        pattern = patterns[digit]
        
        for _ in range(n_samples_per_class):
            # 将模式转换为二值图像
            img = []
            for row in pattern:
                for char in row:
                    # 基础像素值
                    if char == '#':
                        pixel = 1
                    else:
                        pixel = 0
                    
                    # 添加噪声
                    if np.random.random() < noise:
                        pixel = 1 - pixel
                    
                    img.append(pixel)
            
            X.append(img)
            y.append(digit)
    
    return np.array(X), np.array(y)


def demo_digit_recognition():
    """
    手写数字识别示例
    
    使用伯努利朴素贝叶斯进行数字分类
    """
    print("\n" + "=" * 60)
    print("手写数字识别示例 - 伯努利朴素贝叶斯")
    print("=" * 60)
    
    # 生成合成数据
    X, y = generate_mnist_like_data(n_samples_per_class=50, noise=0.15)
    
    print(f"\n数据集大小: {len(X)} 个样本")
    print(f"图像尺寸: 8x8 = 64 像素")
    print(f"类别数: 10 (数字 0-9)")
    
    # 划分训练集和测试集
    # 每个类别的前40个作为训练集，后10个作为测试集
    train_indices = []
    test_indices = []
    
    for digit in range(10):
        digit_indices = np.where(y == digit)[0]
        train_indices.extend(digit_indices[:40])
        test_indices.extend(digit_indices[40:])
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 训练伯努利朴素贝叶斯
    clf = BernoulliNB(alpha=1.0, binarize=None)
    clf.fit(X_train, y_train)
    
    # 预测
    predictions = clf.predict(X_test)
    
    # 评估
    accuracy = np.mean(predictions == y_test)
    print(f"\n准确率: {accuracy * 100:.1f}%")
    
    # 显示每个数字的准确率
    print("\n" + "-" * 40)
    print("每个数字的准确率:")
    print("-" * 40)
    
    for digit in range(10):
        digit_mask = y_test == digit
        if np.sum(digit_mask) > 0:
            digit_accuracy = np.mean(predictions[digit_mask] == y_test[digit_mask])
            print(f"数字 {digit}: {digit_accuracy * 100:.1f}%")
    
    # 显示一些预测示例
    print("\n" + "-" * 40)
    print("预测示例:")
    print("-" * 40)
    
    for i in range(min(5, len(X_test))):
        true_label = y_test[i]
        pred_label = predictions[i]
        status = "✓" if true_label == pred_label else "✗"
        print(f"样本 {i+1}: 真实={true_label}, 预测={pred_label} {status}")
    
    # 可视化每个类别的平均图像（学习到的模式）
    print("\n" + "-" * 40)
    print("学习到的数字模式（每个类别的平均图像）:")
    print("-" * 40)
    
    for digit in range(10):
        # 获取该类别的样本
        digit_samples = X_train[y_train == digit]
        # 计算平均图像
        mean_image = np.mean(digit_samples, axis=0)
        
        print(f"\n数字 {digit}:")
        # 将平均图像显示为ASCII艺术
        for row in range(8):
            row_str = ""
            for col in range(8):
                pixel = mean_image[row * 8 + col]
                if pixel > 0.5:
                    row_str += "#"
                elif pixel > 0.2:
                    row_str += "+"
                else:
                    row_str += "."
            print(f"  {row_str}")
    
    return clf


# ============================================================================
# 第七部分：鸢尾花分类示例（高斯朴素贝叶斯）
# ============================================================================

def demo_iris_classification():
    """
    鸢尾花分类示例
    
    使用高斯朴素贝叶斯进行分类
    """
    print("\n" + "=" * 60)
    print("鸢尾花分类示例 - 高斯朴素贝叶斯")
    print("=" * 60)
    
    # 简化的鸢尾花数据（三类：山鸢尾、变色鸢尾、维吉尼亚鸢尾）
    # 特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度
    
    # 生成合成数据（基于真实鸢尾花的统计特征）
    np.random.seed(42)
    
    # 类别0: 山鸢尾 (Setosa)
    # 特征均值：[5.01, 3.43, 1.46, 0.24]
    X_setosa = np.random.multivariate_normal(
        mean=[5.01, 3.43, 1.46, 0.24],
        cov=np.diag([0.12, 0.14, 0.03, 0.01]),
        size=50
    )
    y_setosa = np.zeros(50, dtype=int)
    
    # 类别1: 变色鸢尾 (Versicolor)
    # 特征均值：[5.94, 2.77, 4.26, 1.33]
    X_versicolor = np.random.multivariate_normal(
        mean=[5.94, 2.77, 4.26, 1.33],
        cov=np.diag([0.26, 0.10, 0.22, 0.04]),
        size=50
    )
    y_versicolor = np.ones(50, dtype=int)
    
    # 类别2: 维吉尼亚鸢尾 (Virginica)
    # 特征均值：[6.59, 2.97, 5.55, 2.03]
    X_virginica = np.random.multivariate_normal(
        mean=[6.59, 2.97, 5.55, 2.03],
        cov=np.diag([0.40, 0.10, 0.30, 0.07]),
        size=50
    )
    y_virginica = np.full(50, 2, dtype=int)
    
    # 合并数据
    X = np.vstack([X_setosa, X_versicolor, X_virginica])
    y = np.concatenate([y_setosa, y_versicolor, y_virginica])
    
    print(f"\n数据集大小: {len(X)} 个样本")
    print(f"特征数: 4 (花萼长度、花萼宽度、花瓣长度、花瓣宽度)")
    print(f"类别数: 3 (山鸢尾、变色鸢尾、维吉尼亚鸢尾)")
    
    # 划分训练集和测试集
    # 每个类别的前40个作为训练集，后10个作为测试集
    train_indices = []
    test_indices = []
    
    for cls in range(3):
        cls_indices = np.where(y == cls)[0]
        train_indices.extend(cls_indices[:40])
        test_indices.extend(cls_indices[40:])
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 训练高斯朴素贝叶斯
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    
    # 预测
    predictions = clf.predict(X_test)
    probabilities = clf.predict_proba(X_test)
    
    # 评估
    accuracy = np.mean(predictions == y_test)
    print(f"\n准确率: {accuracy * 100:.1f}%")
    
    # 显示每个类别的统计信息
    print("\n" + "-" * 40)
    print("学习到的类别统计:")
    print("-" * 40)
    
    class_names = ["山鸢尾", "变色鸢尾", "维吉尼亚鸢尾"]
    feature_names = ["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"]
    
    for idx, name in enumerate(class_names):
        print(f"\n{name} (类别 {idx}):")
        print(f"  先验概率: {clf.class_prior_[idx]:.4f}")
        print(f"  均值: {clf.theta_[idx]}")
        print(f"  方差: {clf.var_[idx]}")
    
    # 显示预测示例
    print("\n" + "-" * 40)
    print("预测示例:")
    print("-" * 40)
    
    for i in range(min(5, len(X_test))):
        true_label = y_test[i]
        pred_label = predictions[i]
        prob = probabilities[i]
        status = "✓" if true_label == pred_label else "✗"
        
        print(f"\n样本 {i+1}:")
        print(f"  特征: {[f'{v:.2f}' for v in X_test[i]]}")
        print(f"  真实类别: {class_names[true_label]}")
        print(f"  预测类别: {class_names[pred_label]} {status}")
        print(f"  概率: {[f'{p:.4f}' for p in prob]}")
    
    return clf


# ============================================================================
# 第八部分：主程序入口
# ============================================================================

def main():
    """
    运行所有示例
    """
    print("\n" + "=" * 70)
    print("朴素贝叶斯分类器 - 完整演示")
    print("《机器学习与深度学习：从小学生到大师》第十一章")
    print("=" * 70)
    
    # 示例1: 垃圾邮件分类（多项式NB）
    demo_spam_classifier()
    
    # 示例2: 鸢尾花分类（高斯NB）
    demo_iris_classification()
    
    # 示例3: 手写数字识别（伯努利NB）
    demo_digit_recognition()
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
