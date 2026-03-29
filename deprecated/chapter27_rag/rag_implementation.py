"""
第二十七章：检索增强生成（RAG）完整代码实现
============================================
本模块实现了一个完整的RAG系统，包括：
1. VectorStore - 基于NumPy的向量存储
2. Embedder - 文本嵌入接口
3. Retriever - 密集检索器
4. RAGPipeline - 完整流水线
5. 示例应用 - 问答系统和文档摘要

作者: ML教材编写组
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import random


# ============================================================================
# 第一部分: VectorStore - 向量存储
# ============================================================================

class VectorStore:
    """
    向量存储类，用于存储和检索向量化的文档。
    
    该类实现了基于NumPy的高性能向量存储，支持：
    - 向量的增删改查
    - 余弦相似度和点积相似度检索
    - Top-k最近邻搜索
    - 数据持久化（保存/加载）
    
    属性:
        dimension: 向量维度
        vectors: 存储的向量矩阵，形状为 (N, D)
        documents: 原始文档列表
        metadata: 文档元数据列表
    
    示例:
        >>> store = VectorStore(dimension=768)
        >>> vectors = np.random.randn(10, 768)
        >>> docs = [f"文档{i}" for i in range(10)]
        >>> store.add(vectors, docs)
        >>> results = store.search(vectors[0], k=3)
    """
    
    def __init__(self, dimension: int = 768):
        """
        初始化向量存储。
        
        参数:
            dimension: 向量维度，默认768（BERT标准维度）
        """
        self.dimension = dimension
        # 初始化为空矩阵，形状为 (0, dimension)
        self.vectors = np.array([]).reshape(0, dimension)
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self._index_built = False
        
    def add(self, vectors: np.ndarray, documents: List[str], 
            metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        添加向量到存储。
        
        参数:
            vectors: 要添加的向量矩阵，形状为 (N, D)
            documents: 对应的原始文档列表，长度为N
            metadata: 可选的元数据列表，长度为N
            
        异常:
            ValueError: 当向量维度不匹配或数量不一致时抛出
            
        示例:
            >>> store = VectorStore(768)
            >>> vecs = np.random.randn(5, 768)
            >>> docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
            >>> meta = [{"source": f"file{i}"} for i in range(5)]
            >>> store.add(vecs, docs, meta)
        """
        # 验证输入
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"向量维度不匹配: 期望 {self.dimension}, 得到 {vectors.shape[1]}"
            )
        
        if len(vectors) != len(documents):
            raise ValueError(
                f"向量数量({len(vectors)})和文档数量({len(documents)})必须相同"
            )
        
        # L2归一化向量，用于高效计算余弦相似度
        vectors = self._normalize(vectors)
        
        # 追加到现有存储
        if self.vectors.shape[0] == 0:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        
        self.documents.extend(documents)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in documents])
        
        self._index_built = False  # 标记索引需要重建
        
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        L2归一化向量。
        
        归一化后的向量满足 ||v||_2 = 1，使得点积等于余弦相似度。
        
        参数:
            vectors: 输入向量矩阵，形状为 (N, D)
            
        返回:
            归一化后的向量矩阵
            
        数学:
            v_normalized = v / ||v||_2
            其中 ||v||_2 = sqrt(sum(v_i^2))
        """
        # 添加小epsilon防止除零错误
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-10)
    
    def search(self, query_vector: np.ndarray, k: int = 5,
               metric: str = "cosine") -> List[Tuple[int, float, str, Dict]]:
        """
        搜索与查询向量最相似的文档。
        
        参数:
            query_vector: 查询向量，形状为 (D,)
            k: 返回结果数量，默认5
            metric: 相似度度量，可选 "cosine" 或 "dot"
            
        返回:
            检索结果列表，每个元素为 (索引, 相似度, 文档, 元数据) 的元组
            结果按相似度降序排列
            
        示例:
            >>> store = VectorStore(768)
            >>> # ... 添加文档 ...
            >>> query = np.random.randn(768)
            >>> results = store.search(query, k=3)
            >>> for idx, score, doc, meta in results:
            ...     print(f"[{idx}] {score:.4f}: {doc[:50]}")
        """
        if self.vectors.shape[0] == 0:
            return []
        
        # 归一化查询向量
        query_vector = self._normalize(query_vector.reshape(1, -1)).flatten()
        
        # 计算相似度
        if metric == "cosine":
            # 归一化向量的点积 = 余弦相似度
            # 这是高效的实现方式，避免了重复计算范数
            similarities = np.dot(self.vectors, query_vector)
        elif metric == "dot":
            similarities = np.dot(self.vectors, query_vector)
        else:
            raise ValueError(f"不支持的度量: {metric}，可选 'cosine' 或 'dot'")
        
        # 获取Top-k索引
        # np.argsort返回升序排列的索引，[::-1]反转得到降序
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        # 组装结果
        results = []
        for idx in top_k_indices:
            results.append((
                int(idx),
                float(similarities[idx]),
                self.documents[idx],
                self.metadata[idx]
            ))
        
        return results
    
    def batch_search(self, query_vectors: np.ndarray, k: int = 5,
                     metric: str = "cosine") -> List[List[Tuple[int, float, str, Dict]]]:
        """
        批量搜索多个查询向量。
        
        参数:
            query_vectors: 查询向量矩阵，形状为 (M, D)
            k: 每个查询返回结果数量
            metric: 相似度度量
            
        返回:
            每个查询的检索结果列表
        """
        results = []
        for query_vector in query_vectors:
            results.append(self.search(query_vector, k, metric))
        return results
    
    def delete(self, indices: List[int]) -> None:
        """
        删除指定索引的向量。
        
        参数:
            indices: 要删除的索引列表
        """
        # 转换为集合并排序（降序）
        # 降序删除确保删除一个元素后不影响其他元素的索引
        indices = sorted(set(indices), reverse=True)
        
        for idx in indices:
            if 0 <= idx < len(self.documents):
                self.vectors = np.delete(self.vectors, idx, axis=0)
                del self.documents[idx]
                del self.metadata[idx]
        
        self._index_built = False
    
    def update(self, idx: int, vector: Optional[np.ndarray] = None,
               document: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        更新指定索引的向量、文档或元数据。
        
        参数:
            idx: 要更新的索引
            vector: 新向量（可选）
            document: 新文档（可选）
            metadata: 新元数据（可选）
        """
        if not (0 <= idx < len(self.documents)):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(self.documents)})")
        
        if vector is not None:
            self.vectors[idx] = self._normalize(vector.reshape(1, -1))
        
        if document is not None:
            self.documents[idx] = document
        
        if metadata is not None:
            self.metadata[idx] = metadata
        
        self._index_built = False
    
    def save(self, path: str) -> None:
        """
        保存向量存储到文件。
        
        参数:
            path: 保存路径前缀
            
        示例:
            >>> store.save("/path/to/knowledge_base")
            # 将创建 /path/to/knowledge_base_vectors.npy
            # 和 /path/to/knowledge_base_docs.json
        """
        np.save(f"{path}_vectors.npy", self.vectors)
        with open(f"{path}_docs.json", "w", encoding="utf-8") as f:
            json.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "dimension": self.dimension
            }, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str) -> None:
        """
        从文件加载向量存储。
        
        参数:
            path: 加载路径前缀
        """
        self.vectors = np.load(f"{path}_vectors.npy")
        with open(f"{path}_docs.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            self.dimension = data["dimension"]
    
    def __len__(self) -> int:
        """返回存储的文档数量。"""
        return len(self.documents)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str, Dict]:
        """获取指定索引的向量、文档和元数据。"""
        return self.vectors[idx], self.documents[idx], self.metadata[idx]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息。"""
        return {
            "num_documents": len(self.documents),
            "dimension": self.dimension,
            "memory_mb": self.vectors.nbytes / (1024 * 1024)
        }


# ============================================================================
# 第二部分: Embedder - 文本嵌入
# ============================================================================

class BaseEmbedder(ABC):
    """
    嵌入器抽象基类。
    
    所有嵌入器必须实现 encode 方法和 dimension 属性。
    """
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表编码为向量。
        
        参数:
            texts: 文本列表，长度为N
            
        返回:
            向量矩阵，形状为 (N, D)
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回嵌入维度D。"""
        pass
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        编码单个文本。
        
        参数:
            text: 输入文本
            
        返回:
            向量，形状为 (D,)
        """
        return self.encode([text])[0]


class SimpleEmbedder(BaseEmbedder):
    """
    简单的词袋嵌入器（用于演示和测试）。
    
    使用简单的哈希方法生成固定维度的向量。
    这不是语义嵌入，仅用于测试RAG系统的功能。
    """
    
    def __init__(self, dimension: int = 768):
        self._dimension = dimension
        np.random.seed(42)
        self.vocab = self._build_vocab()
    
    def _build_vocab(self) -> Dict[str, np.ndarray]:
        """为常见词预生成随机向量。"""
        vocab = {}
        common_words = [
            # 中文
            "的", "了", "是", "我", "有", "和", "就", "不", "人", "在",
            "他", "为", "之", "来", "以", "个", "中", "上", "大", "到",
            "说", "国", "出", "也", "会", "对", "而", "及", "与", "年",
            "要", "得", "里", "后", "自", "家", "可", "下", "天", "去",
            "能", "多", "好", "小", "多", "过", "得", "看", "起", "把",
            # 英文
            "the", "is", "a", "and", "of", "to", "in", "that", "have",
            "i", "it", "for", "not", "on", "with", "he", "as", "you",
            "do", "at", "this", "but", "his", "by", "from", "they",
            "we", "say", "her", "she", "or", "an", "will", "my", "one",
            "all", "would", "there", "their", "what", "so", "up", "out",
            # AI相关
            "machine", "learning", "deep", "neural", "network", "model",
            "data", "train", "test", "predict", "classify", "regression",
            "人工智能", "机器学习", "深度学习", "神经网络", "模型", "训练",
            "数据", "预测", "分类", "回归", "算法", "监督", "无监督",
            # 历史
            "history", "war", "century", "ancient", "emperor", "dynasty",
            "历史", "战争", "世纪", "古代", "皇帝", "朝代", "文明",
            # 科学
            "science", "physics", "chemistry", "biology", "research",
            "科学", "物理", "化学", "生物", "研究", "实验", "理论",
            # 艺术
            "art", "music", "painting", "literature", "artist",
            "艺术", "音乐", "绘画", "文学", "艺术家", "作品",
        ]
        for word in common_words:
            vocab[word] = np.random.randn(self._dimension)
        return vocab
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """使用简单的词袋方法编码文本。"""
        vectors = []
        for text in texts:
            # 简单分词：按空格和标点分割
            import re
            words = re.findall(r'\w+', text.lower())
            vector = np.zeros(self._dimension)
            
            for word in words:
                if word in self.vocab:
                    vector += self.vocab[word]
                else:
                    # 为未知词生成确定性随机向量
                    np.random.seed(hash(word) % (2**32))
                    vector += np.random.randn(self._dimension)
            
            # 平均池化
            if len(words) > 0:
                vector /= len(words)
            
            vectors.append(vector)
        
        return np.array(vectors)
    
    @property
    def dimension(self) -> int:
        return self._dimension


class MockBERTEmbedder(BaseEmbedder):
    """
    模拟BERT风格的嵌入器。
    
    生成具有语义结构的模拟向量，不同主题的文本在向量空间中有不同的聚类中心。
    这使得检索能够基于"主题相似度"工作，适合演示RAG的功能。
    """
    
    def __init__(self, dimension: int = 768):
        self._dimension = dimension
        np.random.seed(42)
        
        # 定义主题聚类中心
        self.topic_centers = {
            "ai": np.random.randn(dimension) * 0.3,
            "history": np.random.randn(dimension) * 0.3,
            "science": np.random.randn(dimension) * 0.3,
            "art": np.random.randn(dimension) * 0.3,
            "general": np.random.randn(dimension) * 0.3,
        }
        
        # 设置不同的偏移量使主题中心相互远离
        # 这样在向量空间中不同主题的文档会聚类在不同区域
        self.topic_centers["ai"][0] = 1.0
        self.topic_centers["ai"][1] = 0.5
        self.topic_centers["history"][1] = 1.0
        self.topic_centers["history"][2] = 0.5
        self.topic_centers["science"][2] = 1.0
        self.topic_centers["science"][3] = 0.5
        self.topic_centers["art"][3] = 1.0
        self.topic_centers["art"][0] = 0.5
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """基于文本主题生成模拟嵌入。"""
        vectors = []
        
        for text in texts:
            text_lower = text.lower()
            
            # 检测主题
            if any(w in text_lower for w in [
                "machine", "learning", "ai", "neural", "model", "train",
                "人工智能", "机器学习", "深度学习", "神经网络", "模型", "训练",
                "监督", "无监督", "强化学习", "transformer", "bert", "gpt"
            ]):
                base = self.topic_centers["ai"].copy()
            elif any(w in text_lower for w in [
                "history", "war", "century", "ancient", "emperor", "dynasty",
                "历史", "战争", "世纪", "古代", "皇帝", "朝代", "文明"
            ]):
                base = self.topic_centers["history"].copy()
            elif any(w in text_lower for w in [
                "science", "physics", "chemistry", "biology", "research",
                "科学", "物理", "化学", "生物", "研究", "实验", "理论"
            ]):
                base = self.topic_centers["science"].copy()
            elif any(w in text_lower for w in [
                "art", "music", "painting", "literature", "artist",
                "艺术", "音乐", "绘画", "文学", "艺术家", "作品"
            ]):
                base = self.topic_centers["art"].copy()
            else:
                base = self.topic_centers["general"].copy()
            
            # 添加随机噪声使同类文档有差异
            noise = np.random.randn(self._dimension) * 0.15
            vector = base + noise
            vectors.append(vector)
        
        return np.array(vectors)
    
    @property
    def dimension(self) -> int:
        return self._dimension


# 尝试导入真实的SentenceTransformer
try:
    from sentence_transformers import SentenceTransformer
    
    class SentenceBERTEmbedder(BaseEmbedder):
        """
        使用真实BERT模型的嵌入器。
        
        需要安装 sentence-transformers 库:
        pip install sentence-transformers
        
        示例:
            >>> embedder = SentenceBERTEmbedder("all-MiniLM-L6-v2")
            >>> vectors = embedder.encode(["Hello world", "你好世界"])
        """
        
        def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
            """
            参数:
                model_name: SentenceTransformer模型名称
            """
            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
        
        def encode(self, texts: List[str]) -> np.ndarray:
            """编码文本列表。"""
            return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        @property
        def dimension(self) -> int:
            return self._dimension
    
    HAS_REAL_EMBEDDER = True
except ImportError:
    HAS_REAL_EMBEDDER = False
    SentenceBERTEmbedder = None


# ============================================================================
# 第三部分: Retriever - 检索器
# ============================================================================

@dataclass
class RetrievalResult:
    """
    检索结果数据类。
    
    属性:
        document: 检索到的文档内容
        score: 相似度分数
        index: 在向量存储中的原始索引
        metadata: 文档元数据
    """
    document: str
    score: float
    index: int
    metadata: Optional[Dict[str, Any]] = None


class DenseRetriever:
    """
    密集检索器。
    
    使用向量相似度从知识库中检索相关文档。
    这是RAG系统的核心组件之一。
    
    示例:
        >>> retriever = DenseRetriever(vector_store, embedder)
        >>> results = retriever.retrieve("什么是机器学习？", k=3)
        >>> for r in results:
        ...     print(f"{r.score:.4f}: {r.document[:50]}")
    """
    
    def __init__(self, vector_store: VectorStore, embedder: BaseEmbedder):
        """
        初始化检索器。
        
        参数:
            vector_store: VectorStore实例
            embedder: BaseEmbedder实例
        """
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, query: str, k: int = 5,
                 metric: str = "cosine") -> List[RetrievalResult]:
        """
        检索与查询最相关的文档。
        
        参数:
            query: 查询字符串
            k: 返回文档数量
            metric: 相似度度量，"cosine" 或 "dot"
            
        返回:
            RetrievalResult列表，按相似度降序排列
        """
        # 编码查询
        query_vector = self.embedder.encode([query])
        
        # 搜索向量存储
        results = self.vector_store.search(
            query_vector=query_vector[0],
            k=k,
            metric=metric
        )
        
        # 转换为RetrievalResult对象
        retrieval_results = [
            RetrievalResult(document=doc, score=score, index=idx, metadata=meta)
            for idx, score, doc, meta in results
        ]
        
        return retrieval_results
    
    def batch_retrieve(self, queries: List[str], k: int = 5,
                       metric: str = "cosine") -> List[List[RetrievalResult]]:
        """
        批量检索多个查询。
        
        参数:
            queries: 查询字符串列表
            k: 每个查询返回文档数量
            metric: 相似度度量
            
        返回:
            每个查询的RetrievalResult列表
        """
        return [self.retrieve(q, k, metric) for q in queries]
    
    def retrieve_with_scores(self, query: str, k: int = 5) -> Tuple[List[RetrievalResult], np.ndarray]:
        """
        检索文档并返回原始分数数组。
        
        返回:
            (检索结果列表, 分数数组)
        """
        results = self.retrieve(query, k)
        scores = np.array([r.score for r in results])
        return results, scores


class RAGRetriever:
    """
    RAG专用检索器。
    
    支持RAG特定的检索逻辑，如概率采样和温度控制。
    适用于训练时需要从检索分布中采样的场景。
    """
    
    def __init__(self, vector_store: VectorStore, embedder: BaseEmbedder, 
                 temperature: float = 1.0):
        """
        初始化RAG检索器。
        
        参数:
            vector_store: VectorStore实例
            embedder: BaseEmbedder实例
            temperature: 检索温度，控制概率分布的平滑程度
                        temperature -> 0 时接近贪婪选择
                        temperature 越大分布越均匀
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.temperature = temperature
    
    def retrieve_with_probabilities(self, query: str, k: int = 5) -> Tuple[List[RetrievalResult], np.ndarray]:
        """
        检索文档并计算选择概率。
        
        使用softmax将相似度分数转换为概率分布。
        
        参数:
            query: 查询字符串
            k: 返回文档数量
            
        返回:
            (检索结果列表, 概率数组)
        """
        # 获取更多候选用于概率计算
        query_vector = self.embedder.encode([query])
        raw_results = self.vector_store.search(query_vector[0], k=k*2)
        
        if not raw_results:
            return [], np.array([])
        
        # 截取Top-k
        raw_results = raw_results[:k]
        
        # 计算softmax概率
        scores = np.array([r[1] for r in raw_results])
        
        # 温度缩放
        scores = scores / self.temperature
        
        # Softmax计算（数值稳定性处理）
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)
        
        # 转换为RetrievalResult
        results = []
        for i, (idx, score, doc, meta) in enumerate(raw_results):
            results.append(RetrievalResult(
                document=doc,
                score=score,
                index=idx,
                metadata={**meta, "probability": float(probs[i])}
            ))
        
        return results, probs
    
    def sample_documents(self, query: str, k: int = 5, 
                         num_samples: int = 1) -> List[List[RetrievalResult]]:
        """
        按概率采样文档（用于训练时的随机采样）。
        
        参数:
            query: 查询
            k: 每次采样的文档数
            num_samples: 采样次数
            
        返回:
            多次采样的结果
        """
        results, probs = self.retrieve_with_probabilities(query, k * 2)
        
        if not results:
            return [[] for _ in range(num_samples)]
        
        samples = []
        for _ in range(num_samples):
            # 按概率采样k个文档（不重复）
            indices = np.random.choice(
                len(results), 
                size=min(k, len(results)), 
                replace=False, 
                p=probs
            )
            sampled = [results[i] for i in indices]
            samples.append(sampled)
        
        return samples


# ============================================================================
# 第四部分: RAGPipeline - 完整流水线
# ============================================================================

@dataclass
class RAGOutput:
    """
    RAG输出结果。
    
    属性:
        answer: 生成的答案
        retrieved_documents: 检索到的文档列表
        query: 原始查询
        metadata: 额外元数据
    """
    answer: str
    retrieved_documents: List[RetrievalResult]
    query: str
    metadata: Dict[str, Any]


class SimpleGenerator:
    """
    简单的生成器（用于演示）。
    
    真实的RAG实现会使用T5、BART等seq2seq模型。
    这里使用模板生成来演示流程。
    """
    
    def __init__(self):
        random.seed(42)
        self.response_templates = {
            "ai": [
                "根据检索到的资料，{content}",
                "基于相关信息，{content}",
                "从文档中可以发现，{content}"
            ],
            "history": [
                "历史资料显示，{content}",
                "根据记载，{content}",
                "文献记录表明，{content}"
            ],
            "science": [
                "科学研究显示，{content}",
                "实验数据表明，{content}",
                "根据科学文献，{content}"
            ],
            "art": [
                "艺术评论认为，{content}",
                "相关介绍提到，{content}",
                "艺术史资料显示，{content}"
            ],
            "default": [
                "根据资料：{content}",
                "相关信息显示：{content}",
                "检索结果：{content}"
            ]
        }
    
    def generate(self, query: str, documents: List[RetrievalResult],
                 max_length: int = 200) -> str:
        """
        基于检索文档生成答案。
        
        参数:
            query: 查询
            documents: 检索到的文档
            max_length: 最大生成长度
            
        返回:
            生成的答案
        """
        if not documents:
            return "抱歉，未找到相关信息。"
        
        # 简单策略：拼接前两个文档的主要内容
        combined_content = " ".join([doc.document for doc in documents[:2]])
        
        # 截断到最大长度
        if len(combined_content) > max_length:
            combined_content = combined_content[:max_length] + "..."
        
        # 根据查询选择模板
        query_lower = query.lower()
        if any(w in query_lower for w in ["machine", "learning", "ai", "model", "训练", "模型", "学习"]):
            category = "ai"
        elif any(w in query_lower for w in ["history", "war", "century", "ancient", "历史", "古代", "世纪"]):
            category = "history"
        elif any(w in query_lower for w in ["science", "physics", "chemistry", "科学", "物理", "化学"]):
            category = "science"
        elif any(w in query_lower for w in ["art", "music", "painting", "艺术", "音乐", "绘画"]):
            category = "art"
        else:
            category = "default"
        
        template = random.choice(self.response_templates[category])
        return template.format(content=combined_content)


class RAGPipeline:
    """
    RAG完整流水线。
    
    整合检索器和生成器，提供端到端的问答能力。
    
    示例:
        >>> rag = RAGPipeline(retriever, generator, top_k=3)
        >>> result = rag.query("什么是深度学习？")
        >>> print(result.answer)
        >>> print([d.document for d in result.retrieved_documents])
    """
    
    def __init__(self, retriever: DenseRetriever, 
                 generator: Optional[SimpleGenerator] = None, 
                 top_k: int = 5):
        """
        初始化RAG流水线。
        
        参数:
            retriever: DenseRetriever实例
            generator: 生成器实例（默认为SimpleGenerator）
            top_k: 检索文档数量
        """
        self.retriever = retriever
        self.generator = generator or SimpleGenerator()
        self.top_k = top_k
        
        # 统计信息
        self.stats = {
            "total_queries": 0,
            "total_retrievals": 0
        }
    
    def query(self, query: str, return_documents: bool = True) -> RAGOutput:
        """
        执行RAG查询。
        
        参数:
            query: 用户查询
            return_documents: 是否返回检索文档
            
        返回:
            RAGOutput对象
        """
        self.stats["total_queries"] += 1
        
        # 1. 检索相关文档
        retrieved_docs = self.retriever.retrieve(query, k=self.top_k)
        self.stats["total_retrievals"] += len(retrieved_docs)
        
        # 2. 生成答案
        answer = self.generator.generate(query, retrieved_docs)
        
        # 3. 组装输出
        output = RAGOutput(
            answer=answer,
            retrieved_documents=retrieved_docs if return_documents else [],
            query=query,
            metadata={
                "num_retrieved": len(retrieved_docs),
                "top_score": retrieved_docs[0].score if retrieved_docs else 0.0,
                "avg_score": np.mean([d.score for d in retrieved_docs]) if retrieved_docs else 0.0
            }
        )
        
        return output
    
    def batch_query(self, queries: List[str]) -> List[RAGOutput]:
        """
        批量执行RAG查询。
        
        参数:
            queries: 查询列表
            
        返回:
            RAGOutput列表
        """
        return [self.query(q) for q in queries]
    
    def add_documents(self, documents: List[str], 
                      metadata: Optional[List[Dict]] = None) -> None:
        """
        向知识库添加文档。
        
        参数:
            documents: 文档列表
            metadata: 元数据列表
        """
        # 编码文档
        vectors = self.retriever.embedder.encode(documents)
        
        # 添加到向量存储
        self.retriever.vector_store.add(vectors, documents, metadata)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。"""
        return self.stats.copy()
    
    def save_knowledge_base(self, path: str) -> None:
        """保存知识库到文件。"""
        self.retriever.vector_store.save(path)
    
    def load_knowledge_base(self, path: str) -> None:
        """从文件加载知识库。"""
        self.retriever.vector_store.load(path)


# ============================================================================
# 第五部分: 示例应用
# ============================================================================

def create_demo_knowledge_base() -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    创建演示用的知识库。
    
    返回:
        (文档列表, 元数据列表)
    """
    documents = [
        # AI/机器学习相关
        "机器学习是人工智能的一个分支，它使计算机能够从数据中自动学习和改进，而无需明确编程。",
        "深度学习是机器学习的一种方法，使用多层神经网络来学习数据的层次化表示。",
        "神经网络受到生物神经系统的启发，由相互连接的节点（神经元）组成，可以学习和识别模式。",
        "监督学习是一种机器学习方法，使用带有标签的训练数据来训练模型预测结果。",
        "无监督学习不需要标签数据，它发现数据中的隐藏模式和结构，如聚类和降维。",
        "强化学习通过与环境交互来学习，智能体根据奖励和惩罚来学习最优策略。",
        "Transformer是一种深度学习架构，使用自注意力机制处理序列数据，是GPT和BERT的基础。",
        "BERT是Google开发的预训练语言模型，使用双向编码器表示来理解语言上下文。",
        "GPT（生成式预训练Transformer）是由OpenAI开发的大型语言模型，能够生成人类般的文本。",
        "卷积神经网络（CNN）特别适合处理图像数据，通过卷积层提取空间特征。",
        
        # 历史相关
        "第二次世界大战于1939年9月1日爆发，德国入侵波兰，持续至1945年9月2日。",
        "爱因斯坦于1905年发表了狭义相对论，提出了著名的质能方程E=mc²。",
        "唐朝（618-907年）是中国历史上最强盛的朝代之一，以诗歌和文化繁荣著称。",
        "丝绸之路是古代连接东西方的贸易网络，促进了商品、文化和技术的交流。",
        "秦始皇统一六国，建立了中国第一个中央集权的封建王朝——秦朝。",
        
        # 科学相关
        "DNA（脱氧核糖核酸）是携带遗传信息的分子，由四种核苷酸组成：A、T、G、C。",
        "光合作用是将光能转化为化学能的过程，植物通过叶绿素吸收阳光，将二氧化碳和水转化为葡萄糖和氧气。",
        "量子力学研究微观粒子的行为，引入了波粒二象性和不确定性原理等概念。",
        "黑洞是时空中的一个区域，其引力如此之强，以至于任何粒子甚至光都无法逃逸。",
        
        # 艺术相关
        "《蒙娜丽莎》是达·芬奇创作的肖像画，以其神秘的微笑而闻名于世，现藏于卢浮宫。",
        "贝多芬是古典音乐史上最伟大的作曲家之一，尽管晚年失聪，仍创作了许多不朽的作品。",
        "印象派是19世纪末的艺术运动，强调光线和色彩的瞬间变化，代表画家包括莫奈和雷诺阿。",
    ]
    
    metadata = [
        {"category": "ai", "source": "ml_intro"},
        {"category": "ai", "source": "dl_intro"},
        {"category": "ai", "source": "neural_networks"},
        {"category": "ai", "source": "supervised_learning"},
        {"category": "ai", "source": "unsupervised_learning"},
        {"category": "ai", "source": "reinforcement_learning"},
        {"category": "ai", "source": "transformer"},
        {"category": "ai", "source": "bert"},
        {"category": "ai", "source": "gpt"},
        {"category": "ai", "source": "cnn"},
        {"category": "history", "source": "ww2"},
        {"category": "history", "source": "einstein"},
        {"category": "history", "source": "tang_dynasty"},
        {"category": "history", "source": "silk_road"},
        {"category": "history", "source": "qin_dynasty"},
        {"category": "science", "source": "dna"},
        {"category": "science", "source": "photosynthesis"},
        {"category": "science", "source": "quantum"},
        {"category": "science", "source": "black_hole"},
        {"category": "art", "source": "mona_lisa"},
        {"category": "art", "source": "beethoven"},
        {"category": "art", "source": "impressionism"},
    ]
    
    return documents, metadata


class SummarizationRAG:
    """
    基于RAG的文档摘要系统。
    
    检索相关文档片段，然后生成摘要。
    """
    
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        参数:
            rag_pipeline: RAGPipeline实例
        """
        self.rag = rag_pipeline
    
    def summarize_topic(self, topic: str, num_docs: int = 5) -> str:
        """
        对特定主题进行摘要。
        
        参数:
            topic: 主题/查询
            num_docs: 检索文档数量
            
        返回:
            摘要文本
        """
        # 调整检索数量
        original_k = self.rag.top_k
        self.rag.top_k = num_docs
        
        # 执行RAG查询
        result = self.rag.query(f"总结关于{topic}的信息")
        
        # 恢复设置
        self.rag.top_k = original_k
        
        # 构建摘要
        summary_parts = [
            f"## 关于'{topic}'的摘要",
            "",
            "### 要点总结：",
            result.answer,
            "",
            "### 参考来源：",
        ]
        
        for i, doc in enumerate(result.retrieved_documents, 1):
            source = doc.metadata.get('source', '未知')
            score = doc.score
            summary_parts.append(f"{i}. {source} (相关度: {score:.3f})")
        
        return "\n".join(summary_parts)
    
    def compare_topics(self, topic1: str, topic2: str) -> str:
        """
        比较两个主题。
        
        参数:
            topic1: 第一个主题
            topic2: 第二个主题
            
        返回:
            比较文本
        """
        result1 = self.rag.query(topic1)
        result2 = self.rag.query(topic2)
        
        comparison = f"""
## 主题比较：{topic1} vs {topic2}

### {topic1}
{result1.answer}

### {topic2}
{result2.answer}

### 共同点与差异
基于检索结果，两个主题的关联度分析完成。
- 主题1相关文档数: {len(result1.retrieved_documents)}
- 主题2相关文档数: {len(result2.retrieved_documents)}
"""
        return comparison


def demo_qa_system():
    """问答系统演示。"""
    print("=" * 70)
    print("RAG问答系统演示")
    print("=" * 70)
    
    # 1. 初始化组件
    print("\n[1] 初始化嵌入器和向量存储...")
    embedder = MockBERTEmbedder(dimension=768)
    vector_store = VectorStore(dimension=768)
    
    # 2. 创建知识库
    print("[2] 构建知识库...")
    documents, metadata = create_demo_knowledge_base()
    vectors = embedder.encode(documents)
    vector_store.add(vectors, documents, metadata)
    print(f"    已添加 {len(documents)} 篇文档")
    print(f"    存储统计: {vector_store.get_stats()}")
    
    # 3. 初始化检索器和RAG流水线
    print("[3] 初始化RAG流水线...")
    retriever = DenseRetriever(vector_store, embedder)
    rag = RAGPipeline(retriever, top_k=3)
    
    # 4. 执行查询
    print("\n" + "=" * 70)
    print("开始问答")
    print("=" * 70)
    
    test_queries = [
        "什么是机器学习？",
        "深度学习和机器学习有什么关系？",
        "请介绍一下Transformer架构",
        "第二次世界大战是什么时候开始的？",
        "DNA是什么？",
        "《蒙娜丽莎》是谁画的？",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n问题 {i}: {query}")
        print("-" * 50)
        
        # 执行RAG查询
        result = rag.query(query)
        
        print(f"回答: {result.answer}")
        print(f"\n检索到的相关文档 ({len(result.retrieved_documents)}篇):")
        for j, doc in enumerate(result.retrieved_documents, 1):
            print(f"  [{j}] 得分: {doc.score:.4f} | 来源: {doc.metadata.get('source', 'unknown')}")
            print(f"      {doc.document[:60]}...")
    
    # 5. 打印统计
    print("\n" + "=" * 70)
    print("系统统计")
    print("=" * 70)
    stats = rag.get_stats()
    print(f"总查询数: {stats['total_queries']}")
    print(f"总检索次数: {stats['total_retrievals']}")
    
    return rag


def demo_summarization(rag_pipeline: RAGPipeline = None):
    """文档摘要演示。"""
    print("\n" + "=" * 70)
    print("RAG文档摘要演示")
    print("=" * 70)
    
    if rag_pipeline is None:
        # 初始化
        embedder = MockBERTEmbedder(dimension=768)
        vector_store = VectorStore(dimension=768)
        documents, metadata = create_demo_knowledge_base()
        vectors = embedder.encode(documents)
        vector_store.add(vectors, documents, metadata)
        retriever = DenseRetriever(vector_store, embedder)
        rag_pipeline = RAGPipeline(retriever, top_k=5)
    
    summarizer = SummarizationRAG(rag_pipeline)
    
    # 主题摘要
    print("\n[1] 主题摘要示例")
    print("-" * 50)
    summary = summarizer.summarize_topic("机器学习", num_docs=4)
    print(summary)
    
    # 主题比较
    print("\n\n[2] 主题比较示例")
    print("-" * 50)
    comparison = summarizer.compare_topics("监督学习", "无监督学习")
    print(comparison)


def demo_vector_operations():
    """演示向量存储的基本操作。"""
    print("\n" + "=" * 70)
    print("向量存储操作演示")
    print("=" * 70)
    
    # 创建向量存储
    store = VectorStore(dimension=10)  # 小维度便于理解
    
    # 添加向量
    vectors = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    ])
    docs = ["文档A", "文档B", "文档C", "文档D"]
    store.add(vectors, docs)
    
    print(f"添加 {len(docs)} 个文档")
    print(f"存储大小: {len(store)}")
    
    # 搜索
    query = np.array([0.95, 0.05, 0, 0, 0, 0, 0, 0, 0, 0])
    results = store.search(query, k=3)
    
    print(f"\n查询向量: {query[:3]}...")
    print("检索结果:")
    for idx, score, doc, _ in results:
        print(f"  [{idx}] {doc}: 相似度 = {score:.4f}")
    
    # 保存和加载
    store.save("/tmp/demo_vectors")
    store2 = VectorStore(dimension=10)
    store2.load("/tmp/demo_vectors")
    print(f"\n加载后存储大小: {len(store2)}")


def main():
    """主函数：运行所有演示。"""
    print("\n" + "=" * 70)
    print("第二十七章：检索增强生成（RAG）代码演示")
    print("=" * 70)
    
    # 演示1: 向量操作
    demo_vector_operations()
    
    # 演示2: 问答系统
    rag = demo_qa_system()
    
    # 演示3: 文档摘要
    demo_summarization(rag)
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
