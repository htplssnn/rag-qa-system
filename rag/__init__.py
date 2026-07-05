"""
RAG 核心模块包

模块职责：
    document_loader  —— 加载 PDF / TXT / Markdown 文档
    text_splitter    —— 将长文本切分为合适大小的块
    embedding        —— 将文本转化为向量
    vector_store     —— 向量的存储与检索（基于 ChromaDB）
    retriever        —— 根据问题检索相关文本块
    generator        —— 组装 Prompt，调用大模型生成回答
"""

from .document_loader import DocumentLoader
from .text_splitter import TextSplitter
from .embedding import EmbeddingModel
from .vector_store import VectorStore
from .retriever import Retriever
from .generator import Generator

__all__ = [
    "DocumentLoader",
    "TextSplitter",
    "EmbeddingModel",
    "VectorStore",
    "Retriever",
    "Generator",
]
