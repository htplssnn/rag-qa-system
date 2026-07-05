"""
检索模块

检索是 RAG 中"R"（Retrieval）的核心：
    用户提问 -> 问题向量化 -> 在向量库中找最相似的文本块

这个模块把 Embedding 和 VectorStore 串起来，
对外提供简洁的 retrieve() 接口。
"""

from typing import List

from config import config
from .embedding import EmbeddingModel
from .vector_store import VectorStore, SearchResult


class Retriever:
    """检索器：根据用户问题检索相关文档片段"""

    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = None) -> List[SearchResult]:
        """检索与问题最相关的文本块

        参数:
            query:  用户的问题
            top_k:  返回的文本块数量（默认用配置中的值）

        返回:
            SearchResult 列表，按相关度从高到低排序
        """
        if top_k is None:
            top_k = config.retrieve.top_k

        # 1. 把问题转成向量
        query_vector = self.embedding_model.embed_query(query)

        # 2. 在向量库中检索
        results = self.vector_store.search(query_vector, top_k=top_k)

        return results

    def retrieve_with_context(self, query: str, top_k: int = None):
        """检索并直接返回格式化好的上下文文本和来源信息

        返回:
            context:    拼接好的参考文本（喂给大模型用）
            sources:    引用来源列表
            results:    原始检索结果（含分数）
        """
        results = self.retrieve(query, top_k)

        if not results:
            return "", [], []

        # 拼接上下文：每段加上编号和来源标注
        context_parts = []
        sources = []
        for i, res in enumerate(results, 1):
            source_name = res.metadata.get("source", "未知来源")
            context_parts.append(f"【参考片段{i}】（来源：{source_name}）\n{res.content}")
            if source_name not in sources:
                sources.append(source_name)

        context = "\n\n".join(context_parts)
        return context, sources, results
