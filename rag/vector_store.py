"""
向量存储模块

使用 ChromaDB 作为向量数据库，负责：
    1. 存储：把文本块的向量 + 原文 + 元数据存入数据库
    2. 检索：根据查询向量，找出最相似的 K 个文本块

ChromaDB 是一个轻量级向量数据库：
    - 数据持久化在本地磁盘（不需要单独部署服务）
    - 自带相似度计算（余弦相似度）
    - 适合学习和中小规模应用
"""

from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

from config import config
from .text_splitter import Chunk
from .embedding import EmbeddingModel


class SearchResult:
    """单条检索结果"""
    def __init__(self, content: str, metadata: dict, score: float):
        self.content = content      # 文本块内容
        self.metadata = metadata    # 来源信息（文件名等）
        self.score = score          # 相似度分数（越高越相关）


class VectorStore:
    """基于 ChromaDB 的向量存储"""

    COLLECTION_NAME = "knowledge_base"

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self._client = None
        self._collection = None

    def _ensure_db(self):
        """延迟初始化数据库连接"""
        if self._client is not None:
            return

        config.ensure_dirs()
        self._client = chromadb.PersistentClient(
            path=str(config.vector_db_dir),
            settings=Settings(anonymized_telemetry=False),  # 关闭遥测
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
        )

    def add_chunks(self, chunks: List[Chunk]):
        """将文本块批量存入向量库

        流程：文本 -> 向量化 -> 存入 ChromaDB（连同原文和元数据）
        """
        if not chunks:
            return

        self._ensure_db()

        # 批量向量化
        texts = [c.content for c in chunks]
        vectors = self.embedding_model.embed(texts)

        # 生成唯一 ID
        existing_count = self._collection.count()
        ids = [f"chunk_{existing_count + i}" for i in range(len(chunks))]

        # 存入 ChromaDB
        self._collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=[c.metadata for c in chunks],
        )

        return len(chunks)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[SearchResult]:
        """根据查询向量检索最相似的文本块"""
        self._ensure_db()

        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # 整理结果
        search_results = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            # ChromaDB 返回的是距离（越小越相似），转换为相似度分数
            score = 1 - dist
            search_results.append(SearchResult(
                content=doc,
                metadata=meta or {},
                score=score,
            ))

        return search_results

    def clear(self):
        """清空知识库"""
        self._ensure_db()
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """返回知识库中的文本块数量"""
        self._ensure_db()
        return self._collection.count()

    def get_all_sources(self) -> List[str]:
        """获取所有已入库的文档来源名称"""
        self._ensure_db()
        if self._collection.count() == 0:
            return []
        all_data = self._collection.get(include=["metadatas"])
        sources = set()
        for meta in all_data["metadatas"]:
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)
