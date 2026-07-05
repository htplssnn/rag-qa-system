"""
零依赖 RAG 核心模块

不需要安装任何第三方包！
使用 Python 标准库直接调用 OpenAI 兼容 API。

需要的环境变量（至少一个）：
    LLM_API_KEY   - 大模型 API Key
    EMBEDDING_API_KEY - Embedding API Key（不设置则共用 LLM_API_KEY）

支持的 API（修改 BASE_URL 即可切换）：
    DeepSeek:  https://api.deepseek.com/v1
    通义千问:  https://dashscope.aliyuncs.com/compatible-mode/v1
    OpenAI:    https://api.openai.com/v1
"""

import json
import math
import os
import pathlib
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================
# 配置（修改这里或设置环境变量）
# ============================================================

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-383e49c2a8f046ba8f74562732bb0185")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", LLM_BASE_URL)
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", LLM_API_KEY)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

VECTOR_DB_PATH = pathlib.Path("./data/vector_db.json")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    content: str
    metadata: dict
    score: float


# ============================================================
# 工具函数：调用 OpenAI 兼容 API（标准库实现）
# ============================================================

def _api_call(base_url: str, api_key: str, endpoint: str, payload: dict) -> dict:
    """用 urllib 调用 OpenAI 兼容 API（无需 openai 包）"""
    url = base_url.rstrip("/") + endpoint
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ============================================================
# 文档加载
# ============================================================

class DocumentLoader:
    SUPPORTED = {".pdf", ".txt", ".md", ".markdown"}

    def load(self, file_path: str) -> Document:
        path = pathlib.Path(file_path)
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED:
            raise ValueError(f"不支持的格式: {ext}")
        if ext == ".pdf":
            content = self._load_pdf(path)
        else:
            content = self._load_text(path)
        return Document(content=content, metadata={"source": path.name})

    def _load_pdf(self, path: pathlib.Path) -> str:
        # PDF 解析需要 pdfplumber，这里用纯 Python 读取 PDF 文本
        # 如果没有 pdfplumber，尝试用 pypdf 或直接提示
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text.strip())
            return "\n\n".join(pages)
        except ImportError:
            raise RuntimeError(
                "PDF 解析需要 pdfplumber 包。请运行: pip install pdfplumber\n"
                "或者将 PDF 转为 TXT/Markdown 后上传。"
            )

    def _load_text(self, path: pathlib.Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="gbk")


# ============================================================
# 文本切分
# ============================================================

class TextSplitter:
    SEPARATORS = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> List[Chunk]:
        chunks_text = self._split_text(document.content)
        return [
            Chunk(content=t, metadata=document.metadata.copy())
            for t in chunks_text if t.strip()
        ]

    def _split_text(self, text: str) -> List[str]:
        splits = self._recursive_split(text, self.SEPARATORS)
        return self._merge_splits(splits)

    def _recursive_split(self, text: str, seps: list) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        for i, sep in enumerate(seps):
            if sep == "":
                return [text[j:j + self.chunk_size] for j in range(0, len(text), self.chunk_size)]
            if sep in text:
                parts = text.split(sep)
                result = []
                for part in parts:
                    if len(part) <= self.chunk_size:
                        result.append(part)
                    else:
                        result.extend(self._recursive_split(part, seps[i + 1:]))
                joiner = sep if sep in ["。", "！", "？"] else ""
                return [p + joiner for p in result if p] if joiner else [p for p in result if p]
        return [text]

    def _merge_splits(self, splits: List[str]) -> List[str]:
        if not splits:
            return []
        chunks = []
        current = []
        current_len = 0
        for s in splits:
            sl = len(s)
            if current_len + sl > self.chunk_size and current:
                chunks.append("".join(current))
                overlap_text = chunks[-1][-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)
            current.append(s)
            current_len += sl
        if current:
            chunks.append("".join(current))
        return chunks


# ============================================================
# Embedding（调用 API）
# ============================================================

class EmbeddingModel:
    def __init__(self):
        self._check_key()

    def _check_key(self):
        if not EMBEDDING_API_KEY:
            raise ValueError(
                "请设置 EMBEDDING_API_KEY 环境变量，或在代码中填写 EMBEDDING_API_KEY。"
            )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """调用 Embedding API 向量化文本"""
        # OpenAI 兼容 API 一次最多 2048 条，这里简单分批
        batch_size = 100
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = _api_call(
                EMBEDDING_BASE_URL, EMBEDDING_API_KEY,
                "/embeddings",
                {"input": batch, "model": EMBEDDING_MODEL},
            )
            # 按 index 排序
            sorted_data = sorted(resp["data"], key=lambda x: x["index"])
            all_vectors.extend([d["embedding"] for d in sorted_data])
        return all_vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]


# ============================================================
# 向量存储（JSON 文件，无需 ChromaDB）
# ============================================================

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """纯 Python 实现的余弦相似度"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """基于 JSON 文件的轻量向量存储"""

    def __init__(self, db_path: Optional[pathlib.Path] = None):
        self.db_path = db_path or VECTOR_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if self.db_path.exists():
            return json.loads(self.db_path.read_text(encoding="utf-8"))
        return {"chunks": []}

    def _save(self):
        self.db_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_chunks(self, chunks: List[Chunk], vectors: List[List[float]]):
        start = len(self._data["chunks"])
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            self._data["chunks"].append({
                "id": f"chunk_{start + i}",
                "content": chunk.content,
                "metadata": chunk.metadata,
                "vector": vec,
            })
        self._save()

    def search(self, query_vector: List[float], top_k: int = 5) -> List[SearchResult]:
        if not self._data["chunks"]:
            return []
        # 计算相似度并排序
        scored = []
        for item in self._data["chunks"]:
            score = _cosine_similarity(query_vector, item["vector"])
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, item in scored[:top_k]:
            results.append(SearchResult(
                content=item["content"],
                metadata=item["metadata"],
                score=score,
            ))
        return results

    def count(self) -> int:
        return len(self._data["chunks"])

    def clear(self):
        self._data = {"chunks": []}
        self._save()

    def get_all_sources(self) -> List[str]:
        sources = set()
        for item in self._data["chunks"]:
            meta = item.get("metadata", {})
            if "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)


# ============================================================
# 检索器
# ============================================================

class Retriever:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if self.vector_store.count() == 0:
            return []
        query_vector = self.embedding_model.embed_query(query)
        return self.vector_store.search(query_vector, top_k=top_k)

    def retrieve_with_context(self, query: str, top_k: int = 5):
        results = self.retrieve(query, top_k)
        if not results:
            return "", [], []
        parts = []
        sources = []
        for i, res in enumerate(results, 1):
            src = res.metadata.get("source", "未知")
            parts.append(f"【参考片段{i}】（来源：{src}）\n{res.content}")
            if src not in sources:
                sources.append(src)
        return "\n\n".join(parts), sources, results


# ============================================================
# 生成器
# ============================================================

SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请根据下面提供的【参考资料】来回答用户的问题。

回答规则：
1. 只根据参考资料中的内容回答，不要编造资料中没有的信息。
2. 如果参考资料中没有相关内容，请诚实地说"根据知识库中的内容，我无法回答这个问题"。
3. 回答时请清晰、准确、有条理。
4. 如果参考资料中有多个相关片段，请综合它们的信息。
5. 回答末尾标注信息来源，格式如：[来源：文件名]"""


def _build_prompt(query: str, context: str) -> str:
    return f"""【参考资料】
{context}

【用户问题】
{query}

请根据以上参考资料回答用户的问题。如果参考资料不足以回答，请说明。"""


class Generator:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        if not LLM_API_KEY:
            raise ValueError("请设置 LLM_API_KEY 环境变量，或在代码中填写 LLM_API_KEY。")

    def answer(self, query: str) -> dict:
        context, sources, results = self.retriever.retrieve_with_context(query)
        if not context:
            return {"answer": "知识库为空，请先上传文档后再提问。", "sources": [], "contexts": []}

        resp = _api_call(
            LLM_BASE_URL, LLM_API_KEY,
            "/chat/completions",
            {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_prompt(query, context)},
                ],
                "temperature": 0.3,
                "max_tokens": 2048,
            },
        )
        answer = resp["choices"][0]["message"]["content"]
        return {"answer": answer, "sources": sources, "contexts": [r.content for r in results]}
