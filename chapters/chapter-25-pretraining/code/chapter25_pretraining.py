"""
第二十五章：预训练与微调 - 代码实现
包含：词嵌入训练、掩码语言模型、下游任务微调
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import random
import re


# ==================== 1. 词嵌入训练（Word2Vec简化版） ====================

class SimpleWord2Vec:
    """
    简化的Word2Vec实现 - 演示词嵌入学习
    使用Skip-gram模型
    """
    
    def __init__(self, vector_size: int = 50, window: int = 2, 
                 learning_rate: float = 0.01, epochs: int = 100):
        self.vector_size = vector_size
        self.window = window
        self.lr = learning_rate
        self.epochs = epochs
        
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
        # 权重矩阵
        self.W_input = None   # 输入层 → 隐藏层
        self.W_output = None  # 隐藏层 → 输出层
    
    def _build_vocab(self, sentences: List[List[str]]):
        """构建词汇表"""
        words = set()
        for sent in sentences:
            words.update(sent)
        
        self.word_to_idx = {w: i for i, w in enumerate(sorted(words))}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        self.vocab_size = len(words)
        
        print(f"词汇表大小: {self.vocab_size}")
    
    def _init_weights(self):
        """初始化权重"""
        self.W_input = np.random.randn(self.vocab_size, self.vector_size) * 0.01
        self.W_output = np.random.randn(self.vocab_size, self.vector_size) * 0.01
    
    def _generate_training_data(self, sentences: List[List[str]]) -> List[Tuple[int, int]]:
        """生成训练数据 (target, context)"""
        data = []
        
        for sent in sentences:
            for i, target_word in enumerate(sent):
                target_idx = self.word_to_idx[target_word]
                
                # 窗口内的上下文词
                start = max(0, i - self.window)
                end = min(len(sent), i + self.window + 1)
                
                for j in range(start, end):
                    if i != j:
                        context_idx = self.word_to_idx[sent[j]]
                        data.append((target_idx, context_idx))
        
        return data
    
    def fit(self, sentences: List[List[str]]):
        """训练词嵌入"""
        print("=" * 60)
        print("Word2Vec训练")
        print("=" * 60)
        
        # 构建词汇表
        self._build_vocab(sentences)
        self._init_weights()
        
        # 生成训练数据
        training_data = self._generate_training_data(sentences)
        print(f"训练样本数: {len(training_data)}")
        
        # 训练
        for epoch in range(self.epochs):
            total_loss = 0
            random.shuffle(training_data)
            
            for target_idx, context_idx in training_data:
                # 前向传播
                h = self.W_input[target_idx]  # 隐藏层
                u = np.dot(self.W_output, h)  # 输出层前
                
                # Softmax
                exp_u = np.exp(u - np.max(u))
                y_pred = exp_u / np.sum(exp_u)
                
                # 计算损失
                loss = -np.log(y_pred[context_idx] + 1e-10)
                total_loss += loss
                
                # 反向传播
                e = y_pred.copy()
                e[context_idx] -= 1
                
                # 更新权重
                dW_output = np.outer(e, h)
                dW_input = np.dot(self.W_output.T, e)
                
                self.W_output -= self.lr * dW_output
                self.W_input[target_idx] -= self.lr * dW_input
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(training_data)
                print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    def get_vector(self, word: str) -> np.ndarray:
        """获取词的向量表示"""
        if word not in self.word_to_idx:
            return np.zeros(self.vector_size)
        return self.W_input[self.word_to_idx[word]]
    
    def most_similar(self, word: str, topn: int = 5) -> List[Tuple[str, float]]:
        """找到最相似的词"""
        if word not in self.word_to_idx:
            return []
        
        word_vec = self.get_vector(word)
        similarities = []
        
        for w, idx in self.word_to_idx.items():
            if w != word:
                vec = self.W_input[idx]
                similarity = np.dot(word_vec, vec) / (np.linalg.norm(word_vec) * np.linalg.norm(vec) + 1e-10)
                similarities.append((w, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]


def word2vec_demo():
    """Word2Vec演示"""
    # 模拟语料库
    sentences = [
        ["机器学习", "是", "人工智能", "的", "分支"],
        ["深度学习", "是", "机器学习", "的", "子集"],
        ["神经网络", "用于", "深度学习"],
        ["人工智能", "改变", "世界"],
        ["机器学习", "算法", "很", "强大"],
        ["深度", "神经网络", "效果", "好"],
        ["人工智能", "应用", "广泛"],
        ["学习", "算法", "需要", "数据"],
        ["深度", "学习", "需要", "大量", "数据"],
        ["机器", "智能", "不断", "发展"],
    ]
    
    print("示例语料库:")
    for i, sent in enumerate(sentences[:3]):
        print(f"  {' '.join(sent)}")
    print("  ...")
    
    # 训练模型
    model = SimpleWord2Vec(vector_size=20, window=2, 
                          learning_rate=0.01, epochs=100)
    model.fit(sentences)
    
    # 查看词向量
    print("\n词向量示例:")
    for word in ["机器学习", "深度学习", "人工智能"]:
        vec = model.get_vector(word)
        print(f"  {word}: {vec[:5]}... (维度: {len(vec)})")
    
    # 查找相似词
    print("\n相似词查找:")
    for word in ["机器学习", "深度学习"]:
        similar = model.most_similar(word, topn=3)
        print(f"  与'{word}'最相似的词:")
        for w, sim in similar:
            print(f"    - {w}: {sim:.4f}")
    
    return model


# ==================== 2. 掩码语言模型（MLM） ====================

class MaskedLanguageModel:
    """
    掩码语言模型 - BERT的核心预训练任务
    预测被掩盖的词
    """
    
    def __init__(self, vocab: List[str], embedding_dim: int = 64):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        
        # 嵌入层
        self.embeddings = np.random.randn(self.vocab_size, embedding_dim) * 0.01
        
        # 简单的Transformer编码器（简化版）
        self.W_q = np.random.randn(embedding_dim, embedding_dim) * 0.01
        self.W_k = np.random.randn(embedding_dim, embedding_dim) * 0.01
        self.W_v = np.random.randn(embedding_dim, embedding_dim) * 0.01
        
        # 输出层
        self.W_output = np.random.randn(embedding_dim, self.vocab_size) * 0.01
    
    def _self_attention(self, embeddings: np.ndarray) -> np.ndarray:
        """自注意力机制"""
        Q = np.dot(embeddings, self.W_q)
        K = np.dot(embeddings, self.W_k)
        V = np.dot(embeddings, self.W_v)
        
        # 注意力分数
        scores = np.dot(Q, K.T) / np.sqrt(self.embedding_dim)
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # 加权求和
        output = np.dot(attention_weights, V)
        return output
    
    def mask_and_predict(self, sentence: List[str], mask_ratio: float = 0.15):
        """
        掩码并预测
        返回: (被掩盖的词, 预测概率)
        """
        # 随机选择要掩盖的位置
        n_mask = max(1, int(len(sentence) * mask_ratio))
        mask_positions = random.sample(range(len(sentence)), n_mask)
        
        # 获取嵌入
        embeddings = np.array([self.embeddings[self.word_to_idx[w]] for w in sentence])
        
        # 自注意力
        context = self._self_attention(embeddings)
        
        predictions = {}
        for pos in mask_positions:
            # 使用上下文的表示预测被掩盖的词
            logits = np.dot(context[pos], self.W_output)
            
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            # 预测结果
            predicted_idx = np.argmax(probs)
            predicted_word = self.vocab[predicted_idx]
            confidence = probs[predicted_idx]
            
            predictions[pos] = {
                'original': sentence[pos],
                'predicted': predicted_word,
                'confidence': confidence,
                'top3': [(self.vocab[i], probs[i]) for i in np.argsort(probs)[-3:][::-1]]
            }
        
        return predictions


def mlm_demo():
    """掩码语言模型演示"""
    print("\n" + "=" * 60)
    print("掩码语言模型 (MLM) 演示")
    print("=" * 60)
    
    # 小词汇表
    vocab = ["机器学习", "深度", "学习", "人工智能", "神经网络", 
             "数据", "模型", "训练", "预测", "算法"]
    
    model = MaskedLanguageModel(vocab, embedding_dim=32)
    
    sentences = [
        ["深度", "学习", "是", "机器", "学习", "的", "分支"],
        ["神经", "网络", "用于", "图像", "识别"],
    ]
    
    print("\n示例句子:")
    for sent in sentences[:1]:
        print(f"  {' '.join(sent)}")
    
    print("\n掩码预测:")
    for sent in sentences[:1]:
        predictions = model.mask_and_predict(sent, mask_ratio=0.2)
        
        for pos, pred in predictions.items():
            print(f"\n  位置 {pos}:")
            print(f"    原词: {pred['original']}")
            print(f"    预测: {pred['predicted']} (置信度: {pred['confidence']:.4f})")
            print(f"    Top3: {pred['top3']}")


# ==================== 3. 下游任务微调 ====================

class FineTunedClassifier:
    """
    预训练 + 微调的分类器
    使用预训练的词嵌入，在下游任务上微调
    """
    
    def __init__(self, pretrained_embeddings: Dict[str, np.ndarray], 
                 num_classes: int, learning_rate: float = 0.01):
        self.embeddings = pretrained_embeddings
        self.embedding_dim = len(list(pretrained_embeddings.values())[0])
        self.num_classes = num_classes
        self.lr = learning_rate
        
        # 分类层（随机初始化）
        self.W_classifier = np.random.randn(self.embedding_dim, num_classes) * 0.01
        self.b_classifier = np.zeros(num_classes)
    
    def _get_sentence_vector(self, words: List[str]) -> np.ndarray:
        """获取句子向量（词向量的平均）"""
        vectors = [self.embeddings.get(w, np.zeros(self.embedding_dim)) for w in words]
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.embedding_dim)
    
    def predict(self, words: List[str]) -> Tuple[int, np.ndarray]:
        """预测类别"""
        # 获取句子表示
        sentence_vec = self._get_sentence_vector(words)
        
        # 分类
        logits = np.dot(sentence_vec, self.W_classifier) + self.b_classifier
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        predicted_class = np.argmax(probs)
        return predicted_class, probs
    
    def fit(self, texts: List[List[str]], labels: List[int], epochs: int = 50):
        """微调分类器"""
        print(f"\n开始微调...")
        print(f"  样本数: {len(texts)}")
        print(f"  类别数: {self.num_classes}")
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            
            for words, label in zip(texts, labels):
                # 前向传播
                sentence_vec = self._get_sentence_vector(words)
                logits = np.dot(sentence_vec, self.W_classifier) + self.b_classifier
                
                # Softmax
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)
                
                # 计算损失（交叉熵）
                loss = -np.log(probs[label] + 1e-10)
                total_loss += loss
                
                if np.argmax(probs) == label:
                    correct += 1
                
                # 反向传播
                dlogits = probs.copy()
                dlogits[label] -= 1
                
                dW = np.outer(sentence_vec, dlogits)
                db = dlogits
                
                # 更新（只更新分类层，冻结词嵌入）
                self.W_classifier -= self.lr * dW
                self.b_classifier -= self.lr * db
            
            accuracy = correct / len(texts)
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(texts)
                print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy*100:.1f}%")


def finetuning_demo():
    """微调演示"""
    print("\n" + "=" * 60)
    print("下游任务微调演示")
    print("=" * 60)
    
    # 模拟预训练好的词嵌入
    pretrained_embeddings = {
        "电影": np.array([0.5, 0.3, -0.2, 0.1]),
        "好看": np.array([0.4, 0.5, 0.1, 0.2]),
        "精彩": np.array([0.6, 0.4, 0.0, 0.3]),
        "糟糕": np.array([-0.3, -0.5, 0.2, -0.1]),
        "无聊": np.array([-0.4, -0.3, 0.1, -0.2]),
        "推荐": np.array([0.5, 0.4, 0.1, 0.2]),
        "浪费": np.array([-0.5, -0.4, 0.0, -0.3]),
        "时间": np.array([0.0, 0.1, 0.5, 0.0]),
        "演技": np.array([0.3, 0.2, 0.1, 0.4]),
        "烂": np.array([-0.6, -0.5, 0.2, -0.4]),
    }
    
    # 情感分类任务数据
    # 0 = 负面, 1 = 正面
    train_texts = [
        ["电影", "很", "好看"],
        ["非常", "精彩", "推荐"],
        ["演技", "很棒"],
        ["无聊", "浪费", "时间"],
        ["剧情", "糟糕"],
        ["烂", "电影"],
        ["确实", "精彩"],
        ["不", "好看"],
    ]
    train_labels = [1, 1, 1, 0, 0, 0, 1, 0]
    
    # 创建分类器并微调
    classifier = FineTunedClassifier(pretrained_embeddings, num_classes=2)
    classifier.fit(train_texts, train_labels, epochs=50)
    
    # 测试
    test_texts = [
        ["电影", "精彩"],
        ["无聊", "烂"],
    ]
    
    print("\n测试预测:")
    for text in test_texts:
        pred_class, probs = classifier.predict(text)
        sentiment = "正面" if pred_class == 1 else "负面"
        print(f"  {' '.join(text)} → {sentiment} (置信度: {probs[pred_class]:.4f})")


def main():
    """主函数"""
    print("=" * 70)
    print("第二十五章：预训练与微调 - 代码演示")
    print("=" * 70)
    
    # 1. 词嵌入训练
    word2vec_demo()
    
    # 2. 掩码语言模型
    mlm_demo()
    
    # 3. 下游任务微调
    finetuning_demo()
    
    print("\n" + "=" * 70)
    print("第二十五章演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
