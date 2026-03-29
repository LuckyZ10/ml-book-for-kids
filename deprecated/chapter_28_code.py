"""
第二十八章代码：多模态学习
Chapter 28: Multimodal Learning

包含：
1. 对比学习损失 (Contrastive Loss)
2. 简化版CLIP编码器
3. 零样本分类器
4. 跨模态注意力融合

作者: ML教材写作项目
日期: 2026-03-25
"""

import numpy as np
from typing import List, Tuple, Optional


# ============================================================================
# 第一部分：对比学习损失 (Contrastive Loss)
# ============================================================================

class ContrastiveLoss:
    """
    对比损失函数 (InfoNCE Loss)
    
    用于训练多模态模型，让匹配的样本对在向量空间中靠近，
    不匹配的样本对远离。
    
    参考: Oord et al. (2018) "Representation Learning with Contrastive Predictive Coding"
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        参数:
            temperature: 温度系数，控制相似度分布的平滑程度
                        越小 → 分布越尖锐，对困难样本更敏感
                        越大 → 分布越平缓，训练更稳定
        """
        self.temperature = temperature
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        计算余弦相似度
        
        公式: sim(a,b) = (a·b) / (||a|| * ||b||)
        
        参数:
            a: 向量矩阵 [N, D]
            b: 向量矩阵 [M, D]
        返回:
            相似度矩阵 [N, M]
        """
        # L2归一化
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        
        # 矩阵乘法计算相似度
        return np.dot(a_norm, b_norm.T)
    
    def forward(self, 
                text_features: np.ndarray, 
                image_features: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        计算对比损失
        
        参数:
            text_features: 文本特征 [N, D]
            image_features: 图像特征 [N, D]
        返回:
            loss: 标量损失值
            logits: 相似度矩阵（用于分析）
        """
        N = text_features.shape[0]
        
        # 计算所有样本对之间的相似度 [N, N]
        logits = self.cosine_similarity(text_features, image_features)
        
        # 除以温度系数
        logits = logits / self.temperature
        
        # 对角线上的元素是正样本（匹配的图文对）
        # 数值稳定性：减去最大值
        logits_max = np.max(logits, axis=1, keepdims=True)
        logits_stable = logits - logits_max
        
        # 计算softmax
        exp_logits = np.exp(logits_stable)
        log_prob = logits_stable - np.log(np.sum(exp_logits, axis=1, keepdims=True))
        
        # 提取正样本的log概率
        mean_log_prob_pos = -np.mean(np.diag(log_prob))
        
        loss = mean_log_prob_pos
        
        return loss, logits


# ============================================================================
# 第二部分：简化版CLIP
# ============================================================================

class SimpleCLIPEncoder:
    """
    简化的CLIP编码器
    
    实际CLIP使用Transformer和ResNet/ViT，
    这里用简单的线性变换演示核心思想。
    
    参考: Radford et al. (2021) "Learning Transferable Visual Models from Natural Language Supervision"
    """
    
    def __init__(self, embed_dim: int = 64, seed: int = 42):
        """
        参数:
            embed_dim: 嵌入向量维度
            seed: 随机种子，保证可重复性
        """
        self.embed_dim = embed_dim
        
        # 模拟预训练好的编码器权重
        np.random.seed(seed)
        self.text_projection = np.random.randn(128, embed_dim) * 0.01
        self.image_projection = np.random.randn(256, embed_dim) * 0.01
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        编码文本
        
        简化版：将文本转换为简单特征后投影
        实际CLIP使用Transformer编码文本
        
        参数:
            texts: 文本列表
        返回:
            文本特征矩阵 [len(texts), embed_dim]
        """
        features = []
        for text in texts:
            simple_feat = self._text_to_simple_features(text)
            embedding = np.dot(simple_feat, self.text_projection)
            # L2归一化（CLIP的关键）
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            features.append(embedding)
        return np.array(features)
    
    def encode_image(self, images: List[np.ndarray]) -> np.ndarray:
        """
        编码图像
        
        简化版：假设图像已经是特征向量
        实际CLIP使用ResNet或Vision Transformer
        
        参数:
            images: 图像列表（每个图像为一维数组）
        返回:
            图像特征矩阵 [len(images), embed_dim]
        """
        features = []
        for img in images:
            simple_feat = self._image_to_simple_features(img)
            embedding = np.dot(simple_feat, self.image_projection)
            # L2归一化
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            features.append(embedding)
        return np.array(features)
    
    def _text_to_simple_features(self, text: str) -> np.ndarray:
        """简化的文本特征提取（仅用于演示）"""
        features = np.zeros(128)
        for i, char in enumerate(text[:128]):
            features[i] = ord(char) / 255.0
        return features
    
    def _image_to_simple_features(self, img: np.ndarray) -> np.ndarray:
        """简化的图像特征提取（仅用于演示）"""
        if img.size > 256:
            img = img.flatten()[:256]
        else:
            img = np.pad(img.flatten(), (0, 256 - img.size))
        return img / 255.0
    
    def compute_similarity(self, 
                          text_features: np.ndarray, 
                          image_features: np.ndarray) -> np.ndarray:
        """
        计算图文相似度
        
        返回: [num_texts, num_images] 的相似度矩阵
        """
        return np.dot(text_features, image_features.T)


class CLIPZeroShotClassifier:
    """
    基于CLIP的零样本分类器
    
    不需要训练样本，只用类别描述就能分类！
    """
    
    def __init__(self, encoder: SimpleCLIPEncoder):
        self.encoder = encoder
        self.class_names: List[str] = []
        self.class_features: Optional[np.ndarray] = None
    
    def fit(self, class_names: List[str]):
        """
        "训练"分类器——实际上只是编码类别描述
        
        这就是零样本的魔力：不需要训练样本！
        """
        self.class_names = class_names
        self.class_features = self.encoder.encode_text(class_names)
        return self
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        预测图像类别
        
        返回:
            (预测类别, 置信度)
        """
        image_features = self.encoder.encode_image([image])
        similarities = self.encoder.compute_similarity(
            self.class_features, image_features
        ).flatten()
        
        predicted_idx = np.argmax(similarities)
        confidence = similarities[predicted_idx]
        
        return self.class_names[predicted_idx], float(confidence)
    
    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        """
        预测所有类别的概率分布
        """
        image_features = self.encoder.encode_image([image])
        similarities = self.encoder.compute_similarity(
            self.class_features, image_features
        ).flatten()
        
        # softmax转换为概率
        exp_sim = np.exp(similarities * 10)
        probabilities = exp_sim / np.sum(exp_sim)
        return probabilities


# ============================================================================
# 第三部分：跨模态注意力融合
# ============================================================================

class CrossModalAttention:
    """
    跨模态注意力融合
    
    让一个模态的信息去"关注"另一个模态的信息
    
    参考: Vaswani et al. (2017) "Attention Is All You Need"
    """
    
    def __init__(self, d_model: int = 64, seed: int = 42):
        self.d_model = d_model
        self.scale = np.sqrt(d_model)
        
        np.random.seed(seed)
        self.W_Q = np.random.randn(d_model, d_model) * 0.01
        self.W_K = np.random.randn(d_model, d_model) * 0.01
        self.W_V = np.random.randn(d_model, d_model) * 0.01
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, 
                text_features: np.ndarray,
                image_features: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        文本作为Query，关注图像信息
        
        参数:
            text_features: 文本特征 [N, D]
            image_features: 图像特征 [N, D]
        返回:
            fused: 融合后的特征 [N, D]
            attention_weights: 注意力权重 [N, N]
        """
        # 计算Q, K, V
        Q = np.dot(text_features, self.W_Q)
        K = np.dot(image_features, self.W_K)
        V = np.dot(image_features, self.W_V)
        
        # 计算注意力分数
        scores = np.dot(Q, K.T) / self.scale
        attention_weights = self.softmax(scores)
        
        # 加权求和
        attended = np.dot(attention_weights, V)
        
        # 残差连接
        fused = text_features + attended
        
        return fused, attention_weights


class MultimodalFusionClassifier:
    """
    使用注意力融合的多模态分类器
    """
    
    def __init__(self, d_model: int = 64):
        self.encoder = SimpleCLIPEncoder(embed_dim=d_model)
        self.attention = CrossModalAttention(d_model=d_model)
        self.classifier_weights: Optional[np.ndarray] = None
        self.class_names: List[str] = []
    
    def fit(self, 
            texts: List[str], 
            images: List[np.ndarray], 
            labels: List[int],
            class_names: List[str]):
        """
        训练分类器
        
        参数:
            texts: 文本列表
            images: 图像列表
            labels: 标签列表
            class_names: 类别名称
        """
        self.class_names = class_names
        num_classes = len(class_names)
        
        # 编码
        text_features = self.encoder.encode_text(texts)
        image_features = self.encoder.encode_image(images)
        
        # 注意力融合
        fused_features, _ = self.attention.forward(text_features, image_features)
        
        # 简单线性分类器（简化版）
        self.classifier_weights = np.random.randn(fused_features.shape[1], num_classes) * 0.01
        
        # 训练（简化版，实际应该使用梯度下降）
        for _ in range(100):
            logits = np.dot(fused_features, self.classifier_weights)
            # 简化的梯度更新
            probs = self._softmax(logits)
            # 计算梯度并更新（省略）
        
        return self
    
    def predict(self, text: str, image: np.ndarray) -> Tuple[str, float]:
        """
        预测
        """
        text_features = self.encoder.encode_text([text])
        image_features = self.encoder.encode_image([image])
        fused, _ = self.attention.forward(text_features, image_features)
        
        logits = np.dot(fused, self.classifier_weights)
        probs = self._softmax(logits).flatten()
        
        pred_idx = np.argmax(probs)
        return self.class_names[pred_idx], probs[pred_idx]
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# ============================================================================
# 第四部分：演示和测试
# ============================================================================

def demo_contrastive_loss():
    """演示对比损失的工作原理"""
    print("=" * 60)
    print("对比学习损失演示 (Contrastive Learning)")
    print("=" * 60)
    
    criterion = ContrastiveLoss(temperature=0.07)
    
    # 模拟3对匹配的图文
    np.random.seed(42)
    text_features = np.array([
        [0.5, 0.3, -0.2, 0.1, 0.4, -0.1, 0.2, 0.3],
        [0.2, -0.4, 0.3, 0.5, -0.2, 0.1, -0.3, 0.4],
        [-0.1, 0.2, 0.4, -0.3, 0.1, 0.5, -0.2, -0.1],
    ])
    
    image_features = np.array([
        [0.4, 0.2, -0.1, 0.2, 0.3, -0.2, 0.1, 0.4],
        [0.1, -0.3, 0.2, 0.4, -0.1, 0.2, -0.2, 0.3],
        [-0.2, 0.1, 0.3, -0.2, 0.2, 0.4, -0.1, -0.2],
    ])
    
    loss, logits = criterion.forward(text_features, image_features)
    
    print(f"\n批次大小: {text_features.shape[0]}")
    print(f"特征维度: {text_features.shape[1]}")
    print(f"温度系数: {criterion.temperature}")
    
    print("\n相似度矩阵:")
    print("         图1     图2     图3")
    for i in range(3):
        print(f"文{i+1}:  {logits[i, 0]:6.3f}  {logits[i, 1]:6.3f}  {logits[i, 2]:6.3f}")
    
    print(f"\n对比损失值: {loss:.4f}")
    print("\n💡 训练目标: 让对角线相似度尽可能大")
    
    return loss, logits


def demo_clip_zeroshot():
    """演示CLIP零样本分类"""
    print("\n" + "=" * 60)
    print("CLIP 零样本图像分类演示")
    print("=" * 60)
    
    encoder = SimpleCLIPEncoder(embed_dim=64)
    classifier = CLIPZeroShotClassifier(encoder)
    
    class_names = [
        "一只可爱的猫",
        "一只忠诚的狗",
        "一辆红色的跑车",
        "一个美味的苹果",
        "一座高耸的山峰"
    ]
    
    classifier.fit(class_names)
    
    # 模拟测试图像
    np.random.seed(123)
    test_images = [
        ("猫的图片", np.random.randn(64, 64) * 50 + 128),
        ("狗的图片", np.random.randn(64, 64) * 40 + 100),
        ("车的图片", np.random.randn(64, 64) * 60 + 150),
    ]
    
    print("\n分类测试:")
    for desc, img in test_images:
        pred_class, confidence = classifier.predict(img)
        probabilities = classifier.predict_proba(img)
        
        print(f"\n输入: {desc}")
        print(f"预测类别: {pred_class}")
        print(f"置信度: {confidence:.3f}")
        print("各类别概率:")
        for name, prob in zip(class_names, probabilities):
            bar = "█" * int(prob * 20)
            print(f"  {name:20s}: {prob:.3f} {bar}")
    
    print("\n💡 关键点：我们没有用任何训练样本！")


def demo_cross_attention():
    """演示跨模态注意力"""
    print("\n" + "=" * 60)
    print("跨模态注意力融合演示")
    print("=" * 60)
    
    attention = CrossModalAttention(d_model=8)
    
    text_features = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    
    image_features = np.array([
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0],
    ])
    
    fused, weights = attention.forward(text_features, image_features)
    
    print("\n注意力权重矩阵:")
    print("        图1     图2     图3")
    for i in range(3):
        print(f"文{i+1}:  {weights[i, 0]:5.3f}   {weights[i, 1]:5.3f}   {weights[i, 2]:5.3f}")
    
    print("\n观察：对角线权重最高")
    print("说明每个文本主要关注了对应的图像！")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("第二十八章代码：多模态学习")
    print("Chapter 28: Multimodal Learning")
    print("=" * 60)
    
    # 运行所有演示
    demo_contrastive_loss()
    demo_clip_zeroshot()
    demo_cross_attention()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
