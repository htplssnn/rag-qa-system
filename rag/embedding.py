"""
向量化模块

Embedding 是 RAG 的核心概念之一：
    把文本转成一组浮点数向量，让"语义相近"的文本在向量空间中距离也近。
    这样用户提问后，我们就能通过向量距离找到最相关的文档片段。

本模块支持两种模式：
    1. local —— sentence-transformers 本地模型（默认，离线可用）
    2. api   —— OpenAI 兼容的 Embedding API（需要联网 + API Key）

两种模式对外接口完全一致，切换时只需改 config。
"""

from typing import List, Optional

from config import config


class EmbeddingModel:
    """文本向量化模型，统一封装本地和 API 两种模式"""

    def __init__(self):
        self._mode = config.embedding.mode
        self._model = None  # 延迟加载，避免启动时卡住

    def _ensure_model(self):
        """延迟初始化模型（首次调用时加载）"""
        if self._model is not None:
            return

        if self._mode == "local":
            from sentence_transformers import SentenceTransformer
            print(f"[Embedding] 加载本地模型: {config.embedding.local_model} ...")
            self._model = SentenceTransformer(config.embedding.local_model)

        elif self._mode == "api":
            from openai import OpenAI
            self._model = OpenAI(
                base_url=config.embedding.api_base_url,
                api_key=config.embedding.api_key,
            )
            print(f"[Embedding] 使用 API 模型: {config.embedding.api_model}")

        else:
            raise ValueError(f"未知的 embedding 模式: {self._mode}，请使用 'local' 或 'api'")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """将多段文本转为向量

        参数: texts —— 文本列表
        返回: 向量列表，每个向量是一个 float 列表
        """
        self._ensure_model()

        if not texts:
            return []

        if self._mode == "local":
            # sentence-transformers 自带批处理
            vectors = self._model.encode(texts, show_progress_bar=False)
            return vectors.tolist()

        else:
            # API 模式：批量调用
            resp = self._model.embeddings.create(
                input=texts,
                model=config.embedding.api_model,
            )
            # 按索引排序，保证顺序一致
            return [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]

    def embed_query(self, text: str) -> List[float]:
        """将单条查询文本转为向量"""
        return self.embed([text])[0]

    @property
    def dimension(self) -> Optional[int]:
        """返回向量维度（部分模型需先调用 embed 才能知道）"""
        if self._mode == "local" and self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return None
