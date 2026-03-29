"""
RAG系统实现
第45章：检索增强生成
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple


class SimpleEmbeddingModel:
    """简化版嵌入模型（实际应使用预训练模型如Sentence-BERT）"""
    def __init__(self, vocab_size=10000, embed_dim=384):
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(embed_dim, nhead=4, batch_first=True),
            num_layers=2
        )
        self.embed_dim = embed_dim
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """将文本列表编码为向量"""
        # 简化实现：随机初始化向量（实际应使用真实模型）
        vectors = []
        for text in texts:
            # 使用文本的hash作为确定性随机种子
            np.random.seed(hash(text) % 2**32)
            vec = np.random.randn(self.embed_dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # 归一化
            vectors.append(vec)
        return np.array(vectors)


class VectorStore:
    """向量数据库（简化版FAISS）"""
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors = []
        self.documents = []
    
    def add(self, vectors: np.ndarray, documents: List[str]):
        """添加向量到数据库"""
        for vec, doc in zip(vectors, documents):
            self.vectors.append(vec)
            self.documents.append(doc)
    
    def search(self, query_vector: np.ndarray, k: int = 3) -> Tuple[List[str], List[float]]:
        """检索最相似的k个文档"""
        if not self.vectors:
            return [], []
        
        # 计算余弦相似度
        vectors = np.array(self.vectors)
        similarities = np.dot(vectors, query_vector)
        
        # 获取Top-K
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_docs = [self.documents[i] for i in top_k_indices]
        top_k_scores = [float(similarities[i]) for i in top_k_indices]
        
        return top_k_docs, top_k_scores


class RAGSystem:
    """
    基础RAG系统实现
    
    组件：
    - 嵌入模型：将文本转为向量
    - 向量数据库：存储和检索文档
    - 生成器：基于上下文生成回答（简化版）
    """
    def __init__(self, embed_dim: int = 384):
        self.embedder = SimpleEmbeddingModel(embed_dim=embed_dim)
        self.vector_store = VectorStore(dimension=embed_dim)
    
    def add_documents(self, documents: List[str]):
        """
        添加文档到知识库
        
        Args:
            documents: 文档字符串列表
        """
        # 将文档编码为向量
        vectors = self.embedder.encode(documents)
        
        # 存入向量数据库
        self.vector_store.add(vectors, documents)
        print(f"已添加 {len(documents)} 个文档到知识库")
    
    def retrieve(self, query: str, k: int = 3) -> Tuple[List[str], List[float]]:
        """
        检索相关文档
        
        Args:
            query: 查询字符串
            k: 返回文档数量
        
        Returns:
            (文档列表, 相似度分数列表)
        """
        # 将查询编码
        query_vector = self.embedder.encode([query])[0]
        
        # 检索
        docs, scores = self.vector_store.search(query_vector, k)
        return docs, scores
    
    def generate(self, query: str, context: str) -> str:
        """
        基于上下文生成回答（简化版）
        
        实际应使用大语言模型如GPT、LLaMA等
        这里用模板方式模拟
        """
        # 简化生成：基于上下文和查询构造回答
        response = f"根据检索到的信息：{context[:200]}...\n\n针对您的问题'{query}'，"
        response += "这是基于知识库的回答。在实际应用中，这里会调用大语言模型生成完整回答。"
        return response
    
    def query(self, question: str, k: int = 3) -> Dict:
        """
        完整的RAG查询流程
        
        Args:
            question: 用户问题
            k: 检索文档数量
        
        Returns:
            包含回答、源文档、分数的字典
        """
        # 1. 检索
        docs, scores = self.retrieve(question, k)
        
        if not docs:
            return {
                'answer': '知识库中没有相关信息。',
                'source_documents': [],
                'retrieval_scores': []
            }
        
        # 2. 构建上下文
        context = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(docs)])
        
        # 3. 生成回答
        answer = self.generate(question, context)
        
        return {
            'answer': answer,
            'source_documents': docs,
            'retrieval_scores': scores
        }


class AdvancedRAG(RAGSystem):
    """
    高级RAG系统（带查询扩展和重排序）
    """
    def __init__(self, embed_dim: int = 384):
        super().__init__(embed_dim)
    
    def expand_query(self, query: str) -> List[str]:
        """
        查询扩展 - 生成多个查询变体
        
        实际应用中可以使用LLM生成
        """
        expansions = [
            query,
            f"什么是{query}",
            f"{query}的定义",
            f"{query}的原理"
        ]
        return expansions
    
    def reciprocal_rank_fusion(self, results_list: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
        """
        倒数排名融合（RRF）
        
        合并多个检索结果列表
        
        Args:
            results_list: 多个查询的检索结果列表
            k: RRF常数（通常60）
        
        Returns:
            融合后的结果列表
        """
        scores = {}
        
        for results in results_list:
            for rank, (doc, _) in enumerate(results):
                if doc not in scores:
                    scores[doc] = 0
                scores[doc] += 1 / (k + rank + 1)
        
        # 按分数排序
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results
    
    def query(self, question: str, k: int = 3) -> Dict:
        """
        高级RAG查询流程（多查询+RRF融合）
        """
        # 1. 查询扩展
        queries = self.expand_query(question)
        
        # 2. 多查询检索
        all_results = []
        for q in queries:
            docs, scores = self.retrieve(q, k * 2)  # 多检索一些用于融合
            results = list(zip(docs, scores))
            all_results.append(results)
        
        # 3. RRF融合
        fused_results = self.reciprocal_rank_fusion(all_results)
        
        # 取Top-K
        top_k_results = fused_results[:k]
        docs = [doc for doc, _ in top_k_results]
        
        # 4. 生成回答
        context = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(docs)])
        answer = self.generate(question, context)
        
        return {
            'answer': answer,
            'source_documents': docs,
            'expanded_queries': queries,
            'fusion_scores': [score for _, score in top_k_results]
        }


def chunk_documents(documents: List[str], chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """
    文档切分
    
    Args:
        documents: 长文档列表
        chunk_size: 每块的最大字符数
        overlap: 相邻块的重叠字符数
    
    Returns:
        切分后的文档块列表
    """
    chunks = []
    
    for doc in documents:
        # 按句子或段落切分（简化实现）
        # 实际应用应使用更智能的切分策略
        start = 0
        while start < len(doc):
            end = min(start + chunk_size, len(doc))
            chunk = doc[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
    
    return chunks


if __name__ == "__main__":
    print("=" * 60)
    print("RAG系统演示")
    print("=" * 60)
    
    # 创建RAG系统
    rag = RAGSystem()
    
    # 准备知识库文档
    documents = [
        "RAG（Retrieval-Augmented Generation）是一种将信息检索与文本生成结合的AI技术。它通过检索外部知识来增强大语言模型的回答能力。",
        "向量数据库用于存储和检索高维向量，支持语义搜索。常见的有FAISS、Pinecone、Weaviate等。",
        "嵌入模型（Embedding Model）将文本转换为数值向量，捕获语义信息。常用的有BERT、Sentence-BERT、OpenAI的Embedding API等。",
        "稠密检索（Dense Retrieval）使用语义向量进行相似度搜索，相比传统的关键词匹配（BM25）能更好地理解查询意图。",
        "DPR（Dense Passage Retrieval）是Meta提出的双编码器架构，使用两个BERT分别编码查询和文档。",
        "文档切分是RAG的关键步骤。合适的块大小（通常200-500 tokens）和重叠（50-100 tokens）能提升检索质量。",
        "Self-RAG是一种高级RAG技术，让模型自我判断是否需要检索、评估检索质量，实现动态检索。",
        "评估RAG系统需要关注三个层面：检索质量（Recall@K）、生成质量（忠实度）、端到端效果。"
    ]
    
    # 切分文档（如果文档较长）
    chunks = chunk_documents(documents, chunk_size=150, overlap=30)
    print(f"\n文档切分: {len(documents)} 个文档 → {len(chunks)} 个块")
    
    # 添加到知识库
    rag.add_documents(chunks)
    
    # 测试查询
    test_queries = [
        "什么是RAG技术？",
        "向量数据库有什么作用？",
        "如何评估RAG系统？"
    ]
    
    print("\n" + "=" * 60)
    print("查询测试")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n问题: {query}")
        result = rag.query(query, k=2)
        print(f"回答: {result['answer'][:150]}...")
        print(f"来源: {len(result['source_documents'])} 个文档")
        print("-" * 40)
    
    # 高级RAG演示
    print("\n" + "=" * 60)
    print("高级RAG演示（查询扩展+RRF融合）")
    print("=" * 60)
    
    advanced_rag = AdvancedRAG()
    advanced_rag.add_documents(chunks)
    
    query = "RAG的工作原理"
    print(f"\n问题: {query}")
    result = advanced_rag.query(query, k=2)
    print(f"扩展查询: {result['expanded_queries']}")
    print(f"回答: {result['answer'][:150]}...")
