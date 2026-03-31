"""
第十一章：朴素贝叶斯分类器 - NumPy纯手写实现
《机器学习与深度学习：从小学生到大师》

本模块包含：
1. GaussianNB - 高斯朴素贝叶斯（连续特征）
2. MultinomialNB - 多项式朴素贝叶斯（文本/离散特征）
3. BernoulliNB - 伯努利朴素贝叶斯（二元特征）
4. 垃圾邮件分类器示例
5. 手写数字识别示例
6. 鸢尾花分类示例
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
    
    适用于离散计数特征（如文本分类中的词频）。
    
    参数:
        alpha: 拉普拉斯平滑参数，默认为1.0
        fit_prior: 是否从数据中学习先验概率
        class_prior: 指定的先验概率
    """
    
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
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
            X: 训练数据，shape (n_samples, n_features)，特征为计数
            y: 目标标签，shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        
        # 确保特征非负
        if np.any(X < 0):
            raise ValueError("输入X必须为非负数")
        
        # 获取所有类别
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # 初始化
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)
        
        # 计算每个类别的统计量
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]  # 类别c的所有样本
            self.feature_count_[idx, :] = np.sum(X_c, axis=0)
            self.class_count_[idx] = np.sum(self.feature_count_[idx, :])
        
        # 计算平滑后的条件概率的对数
        # log P(x_i|C) = log[(N_{i,c} + alpha) / (N_c + alpha*n_features)]
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1) + self.alpha * n_features
        
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1, 1))
        
        # 计算先验概率
        if self.fit_prior:
            self.class_log_prior_ = np.log(self.class_count_ / np.sum(self.class_count_))
        else:
            if self.class_prior is not None:
                self.class_log_prior_ = np.log(self.class_prior)
            else:
                self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))
        
        return self
    
    def _joint_log_likelihood(self, X):
        """
        计算联合对数似然
        """
        X = np.array(X)
        return np.dot(X, self.feature_log_prob_.T) + self.class_log_prior_
    
    def predict(self, X):
        """
        预测类别标签
        """
        X = np.array(X)
        joint_log_likelihood = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(joint_log_likelihood, axis=1)]
    
    def predict_proba(self, X):
        """
        预测类别概率
        """
        X = np.array(X)
        joint_log_likelihood = self._joint_log_likelihood(X)
        
        # 使用log-sum-exp技巧
        log_prob_max = np.max(joint_log_likelihood, axis=1, keepdims=True)
        prob = np.exp(joint_log_likelihood - log_prob_max)
        prob = prob / np.sum(prob, axis=1, keepdims=True)
        
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
    
    适用于二元特征（0/1），适合短文本分类。
    
    参数:
        alpha: 拉普拉斯平滑参数，默认为1.0
        fit_prior: 是否从数据中学习先验概率
        class_prior: 指定的先验概率
        binarize: 二值化阈值，如果为None则假设输入已经是二值的
    """
    
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None, binarize=0.0):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.binarize = binarize
        self.classes_ = None
        self.class_count_ = None
        self.feature_count_ = None
        self.feature_log_prob_ = None
        self.feature_log_prob_neg_ = None
        self.class_log_prior_ = None
        
    def fit(self, X, y):
        """
        训练伯努利朴素贝叶斯分类器
        
        参数:
            X: 训练数据，shape (n_samples, n_features)
            y: 目标标签，shape (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        
        # 二值化
        if self.binarize is not None:
            X = (X > self.binarize).astype(int)
        
        # 获取所有类别
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # 初始化
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)
        
        # 计算每个类别的统计量
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]  # 类别c的所有样本
            self.feature_count_[idx, :] = np.sum(X_c, axis=0)
            self.class_count_[idx] = len(X_c)
        
        # 计算平滑后的条件概率
        # P(x_i=1|C) = (N_{i,c} + alpha) / (N_c + 2*alpha)
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = self.class_count_.reshape(-1, 1) + 2 * self.alpha
        
        feature_prob = smoothed_fc / smoothed_cc
        
        self.feature_log_prob_ = np.log(feature_prob)
        self.feature_log_prob_neg_ = np.log(1 - feature_prob)
        
        # 计算先验概率
        if self.fit_prior:
            self.class_log_prior_ = np.log(self.class_count_ / np.sum(self.class_count_))
        else:
            if self.class_prior is not None:
                self.class_log_prior_ = np.log(self.class_prior)
            else:
                self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))
        
        return self
    
    def _joint_log_likelihood(self, X):
        """
        计算联合对数似然
        
        对于伯努利模型，我们需要考虑特征出现和不出现的情况：
        log P(x|C) = sum_i [x_i * log P(x_i=1|C) + (1-x_i) * log P(x_i=0|C)]
        """
        X = np.array(X)
        
        # 二值化
        if self.binarize is not None:
            X = (X > self.binarize).astype(int)
        
        # 计算每个特征的对数概率
        # 对于每个特征：如果是1，用log_prob；如果是0，用log_prob_neg
        log_prob = np.dot(X, self.feature_log_prob_.T)
        log_prob_neg = np.dot(1 - X, self.feature_log_prob_neg_.T)
        
        return log_prob + log_prob_neg + self.class_log_prior_
    
    def predict(self, X):
        """
        预测类别标签
        """
        X = np.array(X)
        joint_log_likelihood = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(joint_log_likelihood, axis=1)]
    
    def predict_proba(self, X):
        """
        预测类别概率
        """
        X = np.array(X)
        joint_log_likelihood = self._joint_log_likelihood(X)
        
        # 使用log-sum-exp技巧
        log_prob_max = np.max(joint_log_likelihood, axis=1, keepdims=True)
        prob = np.exp(joint_log_likelihood - log_prob_max)
        prob = prob / np.sum(prob, axis=1, keepdims=True)
        
        return prob
    
    def score(self, X, y):
        """
        计算准确率
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


# ============================================================================
# 第四部分：中文垃圾邮件分类器（完整示例）
# ============================================================================

class ChineseSpamClassifier:
    """
    中文垃圾邮件分类器
    
    基于多项式朴素贝叶斯，针对中文文本进行了优化。
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.vocab = {}  # 词汇表：词 -> 索引
        self.vocab_list = []  # 词汇列表，用于索引到词的映射
        self.class_word_counts = defaultdict(Counter)
        self.class_total_counts = defaultdict(int)
        self.class_counts = defaultdict(int)
        self.nb = MultinomialNB(alpha=alpha)
        self.is_fitted = False
        
        # 中文停用词
        self.stopwords = set(['的', '了', '在', '是', '我', '有', '和', '就', 
                             '不', '人', '都', '一', '一个', '上', '也', '很', 
                             '到', '说', '要', '去', '你', '会', '着', '没有', 
                             '看', '好', '自己', '这', '那', '啊', '呢', '吧', 
                             '吗', '哦', '哈', '啦', '哪', '什么', '怎么', '为什么'])
    
    def _tokenize(self, text):
        """
        简单的中文分词：按字符分割，去除停用词和标点
        """
        import re
        # 去除标点符号和数字，只保留中文字符
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
        # 按字符分割，去除停用词
        tokens = [char for char in text if char not in self.stopwords and len(char.strip()) > 0]
        return tokens
    
    def _build_vocab(self, texts):
        """
        构建词汇表
        """
        word_set = set()
        for text in texts:
            tokens = self._tokenize(text)
            word_set.update(tokens)
        
        self.vocab_list = sorted(list(word_set))
        self.vocab = {word: idx for idx, word in enumerate(self.vocab_list)}
    
    def _text_to_vector(self, text):
        """
        将文本转换为词频向量
        """
        tokens = self._tokenize(text)
        vector = np.zeros(len(self.vocab))
        for token in tokens:
            if token in self.vocab:
                vector[self.vocab[token]] += 1
        return vector
    
    def fit(self, texts, labels):
        """
        训练分类器
        
        参数:
            texts: 文本列表
            labels: 标签列表
        """
        # 构建词汇表
        self._build_vocab(texts)
        
        # 将文本转换为向量
        X = np.array([self._text_to_vector(text) for text in texts])
        y = np.array(labels)
        
        # 训练多项式朴素贝叶斯
        self.nb.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, texts):
        """
        预测文本类别
        
        参数:
            texts: 文本或文本列表
            
        返回:
            预测的标签
        """
        if not self.is_fitted:
            raise ValueError("分类器尚未训练，请先调用fit方法")
        
        if isinstance(texts, str):
            texts = [texts]
        
        X = np.array([self._text_to_vector(text) for text in texts])
        return self.nb.predict(X)
    
    def predict_proba(self, texts):
        """
        预测概率
        """
        if not self.is_fitted:
            raise ValueError("分类器尚未训练，请先调用fit方法")
        
        if isinstance(texts, str):
            texts = [texts]
        
        X = np.array([self._text_to_vector(text) for text in texts])
        return self.nb.predict_proba(X)
    
    def score(self, texts, labels):
        """
        计算准确率
        """
        predictions = self.predict(texts)
        return np.mean(predictions == labels)
    
    def get_top_features(self, class_label, n=10):
        """
        获取对某个类别最重要的n个特征（词）
        
        参数:
            class_label: 类别标签
            n: 返回的特征数量
            
        返回:
            最重要的词及其权重
        """
        if not self.is_fitted:
            raise ValueError("分类器尚未训练")
        
        # 找到类别的索引
        class_idx = list(self.nb.classes_).index(class_label)
        
        # 获取特征对数概率
        feature_log_prob = self.nb.feature_log_prob_[class_idx, :]
        
        # 获取最重要的特征索引
        top_indices = np.argsort(feature_log_prob)[-n:][::-1]
        
        # 返回词和对应的概率
        return [(self.vocab_list[idx], np.exp(feature_log_prob[idx])) for idx in top_indices]


# ============================================================================
# 第五部分：鸢尾花分类示例
# ============================================================================

def iris_classification_example():
    """
    使用高斯朴素贝叶斯进行鸢尾花分类的完整示例
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    print("=" * 60)
    print("鸢尾花分类示例 - 高斯朴素贝叶斯")
    print("=" * 60)
    
    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 创建并训练模型
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    # 预测
    y_pred = gnb.predict(X_test)
    
    # 计算准确率
    accuracy = gnb.score(X_test, y_test)
    
    print(f"训练样本数: {len(X_train)}")
    print(f"测试样本数: {len(X_test)}")
    print(f"特征数: {X.shape[1]}")
    print(f"类别数: {len(np.unique(y))}")
    print(f"类别名称: {iris.target_names}")
    print(f"\n测试准确率: {accuracy:.4f}")
    
    # 打印每个类别的均值和方差
    print("\n每个类别的统计信息:")
    for idx, class_name in enumerate(iris.target_names):
        print(f"\n{class_name}:")
        print(f"  先验概率: {gnb.class_prior_[idx]:.4f}")
        print(f"  特征均值: {gnb.theta_[idx, :]}")
        print(f"  特征方差: {gnb.var_[idx, :]}")
    
    # 预测一个新样本
    new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = gnb.predict(new_sample)
    proba = gnb.predict_proba(new_sample)
    
    print(f"\n新样本预测: {iris.target_names[prediction[0]]}")
    print(f"预测概率: {dict(zip(iris.target_names, proba[0]))}")
    
    return accuracy


# ============================================================================
# 第六部分：20 Newsgroups文本分类示例
# ============================================================================

def text_classification_example():
    """
    使用多项式朴素贝叶斯进行文本分类的示例
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.datasets import fetch_20newsgroups
    
    print("\n" + "=" * 60)
    print("20 Newsgroups文本分类示例 - 多项式朴素贝叶斯")
    print("=" * 60)
    
    # 加载数据（只选4个类别以加快演示）
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    
    try:
        newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, 
                                               remove=('headers', 'footers', 'quotes'))
        newsgroups_test = fetch_20newsgroups(subset='test', categories=categories,
                                              remove=('headers', 'footers', 'quotes'))
    except:
        print("无法下载20 Newsgroups数据集，使用模拟数据")
        # 使用模拟数据
        texts_train = [
            "computer graphics algorithm rendering", "medical doctor hospital patient",
            "atheism religion belief god", "christian church bible faith"
        ] * 50
        texts_test = texts_train[:20]
        y_train = [0, 2, 1, 3] * 50
        y_test = y_train[:20]
        
        # 向量化
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(texts_train).toarray()
        X_test = vectorizer.transform(texts_test).toarray()
    else:
        # 向量化
        vectorizer = CountVectorizer(max_features=10000, stop_words='english')
        X_train = vectorizer.fit_transform(newsgroups_train.data).toarray()
        X_test = vectorizer.transform(newsgroups_test.data).toarray()
        y_train = newsgroups_train.target
        y_test = newsgroups_test.target
    
    # 创建并训练模型
    mnb = MultinomialNB(alpha=0.1)
    mnb.fit(X_train, y_train)
    
    # 计算准确率
    train_accuracy = mnb.score(X_train, y_train)
    test_accuracy = mnb.score(X_test, y_test)
    
    print(f"训练样本数: {len(y_train)}")
    print(f"测试样本数: {len(y_test)}")
    print(f"特征数（词汇量）: {X_train.shape[1]}")
    print(f"训练准确率: {train_accuracy:.4f}")
    print(f"测试准确率: {test_accuracy:.4f}")
    
    return test_accuracy


# ============================================================================
# 第七部分：中文垃圾邮件分类示例
# ============================================================================

def chinese_spam_example():
    """
    中文垃圾邮件分类示例
    """
    print("\n" + "=" * 60)
    print("中文垃圾邮件分类示例")
    print("=" * 60)
    
    # 训练数据
    train_texts = [
        # 正常邮件
        "会议安排在明天下午三点",
        "请查收附件中的报告",
        "下周的行程已确认",
        "发票已开具请查收",
        "项目进度汇报请查看",
        "合同已经签署请查收",
        "下周一开会讨论方案",
        "请回复确认收到邮件",
        "年度报告已经提交",
        "客户反馈需要处理",
        # 垃圾邮件
        "恭喜你中奖了点击领取",
        "免费领取iPhone数量有限",
        "你的账户出现异常请立即验证",
        "轻松赚钱日入万元",
        "点击链接领取大礼包",
        "恭喜您获得百万大奖",
        "限时优惠错过再等一年",
        "投资理财产品高收益",
        "免费抽奖100中奖",
        "速来领取你的专属福利"
    ]
    train_labels = ['ham'] * 10 + ['spam'] * 10
    
    # 测试数据
    test_texts = [
        "明天下午开会请准时参加",
        "恭喜你获得大奖快来领取",
        "项目合同已经准备好了",
        "点击链接免费领取手机"
    ]
    test_labels = ['ham', 'spam', 'ham', 'spam']
    
    # 创建并训练分类器
    clf = ChineseSpamClassifier(alpha=1.0)
    clf.fit(train_texts, train_labels)
    
    # 预测
    predictions = clf.predict(test_texts)
    
    print("\n测试结果:")
    for text, pred, true in zip(test_texts, predictions, test_labels):
        status = "✓" if pred == true else "✗"
        print(f"文本: {text}")
        print(f"预测: {pred}, 实际: {true} {status}\n")
    
    # 计算准确率
    accuracy = clf.score(test_texts, test_labels)
    print(f"测试准确率: {accuracy:.4f}")
    
    # 显示最重要的特征
    print("\n对判断正常邮件最重要的词:")
    for word, prob in clf.get_top_features('ham', 5):
        print(f"  {word}: {prob:.6f}")
    
    print("\n对判断垃圾邮件最重要的词:")
    for word, prob in clf.get_top_features('spam', 5):
        print(f"  {word}: {prob:.6f}")
    
    return accuracy


# ============================================================================
# 第八部分：数值稳定性测试
# ============================================================================

def numerical_stability_test():
    """
    测试对数变换对数值稳定性的影响
    """
    print("\n" + "=" * 60)
    print("数值稳定性测试")
    print("=" * 60)
    
    # 模拟很多个小概率相乘的情况
    probs = [0.0001] * 100  # 100个很小的概率
    
    # 直接相乘
    try:
        direct_product = 1.0
        for p in probs:
            direct_product *= p
        print(f"直接相乘结果: {direct_product}")
    except Exception as e:
        print(f"直接相乘出错: {e}")
    
    # 对数变换后相加
    log_sum = sum(math.log(p) for p in probs)
    print(f"对数变换后相加: {log_sum}")
    print(f"指数化还原: {math.exp(log_sum)}")
    
    # 比较
    print(f"\n直接相乘结果和对数变换结果的差异:")
    print(f"直接相乘在计算机中可能会下溢为0")
    print(f"对数变换后的值可以轻松表示: {log_sum}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    # 运行所有示例
    print("朴素贝叶斯分类器 - NumPy纯手写实现")
    print("=" * 60)
    
    # 1. 鸢尾花分类
    iris_accuracy = iris_classification_example()
    
    # 2. 文本分类
    text_accuracy = text_classification_example()
    
    # 3. 中文垃圾邮件分类
    spam_accuracy = chinese_spam_example()
    
    # 4. 数值稳定性测试
    numerical_stability_test()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)
