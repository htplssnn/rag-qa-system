"""
文本切分模块

这是 RAG 中非常关键的一步。
大模型的上下文窗口有限，而且把整篇长文塞进去既贵又容易跑偏。
所以我们需要把文档切成一个个小块（chunk），只把最相关的几块喂给模型。

切分策略：递归字符切分（Recursive Character Splitting）
    依次尝试按 段落(\\n\\n) -> 换行(\\n) -> 句号(。) -> 空格 -> 字符 来切分，
    尽量保持语义完整性，同时控制每块不超过 chunk_size。

chunk_overlap 的作用：
    相邻块之间保留一段重叠文字，避免一句话被切断导致信息丢失。
"""

from dataclasses import dataclass
from typing import List

from .document_loader import Document


@dataclass
class Chunk:
    """切分后的文本块"""
    content: str
    metadata: dict  # 继承自原文档的元数据


class TextSplitter:
    """递归字符文本切分器"""

    # 切分分隔符优先级：从大到小
    # 先尝试用段落切，切出来的还太大就降级用换行，再不行用句号……
    SEPARATORS = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> List[Chunk]:
        """将一个 Document 切分为多个 Chunk"""
        chunks_text = self._split_text(document.content)
        return [
            Chunk(content=text, metadata=document.metadata.copy())
            for text in chunks_text
        ]

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """批量切分多个文档"""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.split(doc))
        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        """递归切分的核心逻辑"""
        # 先用分隔符把文本拆成小块
        splits = self._recursive_split(text, self.SEPARATORS)

        # 再把小块合并成目标大小的 chunk（带 overlap）
        chunks = self._merge_splits(splits, self.chunk_size, self.chunk_overlap)

        # 过滤掉空块
        return [c for c in chunks if c.strip()]

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """递归地用分隔符拆分文本"""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # 依次尝试每个分隔符
        for i, sep in enumerate(separators):
            if sep == "":
                # 最后兜底：按字符数硬切
                return [text[j:j + self.chunk_size]
                        for j in range(0, len(text), self.chunk_size)]

            if sep in text:
                parts = text.split(sep)
                result = []
                for part in parts:
                    if len(part) <= self.chunk_size:
                        result.append(part)
                    else:
                        # 这部分还是太长，用下一个更细的分隔符继续切
                        result.extend(self._recursive_split(part, separators[i + 1:]))
                # 保留分隔符本身（把句号、换行加回去）
                return [p + sep for p in result if p] if sep in ["。", "！", "？"] else [p for p in result if p]

        return [text]

    def _merge_splits(self, splits: List[str], chunk_size: int, overlap: int) -> List[str]:
        """把切出来的小片段合并成目标大小的 chunk"""
        if not splits:
            return []

        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_len = len(split)

            # 如果加上这个片段会超出限制，就先保存当前 chunk，开始新的
            if current_length + split_len > chunk_size and current_chunk:
                chunk_text = "".join(current_chunk)
                chunks.append(chunk_text)

                # 保留 overlap：从当前 chunk 末尾取 overlap 个字符作为新 chunk 的开头
                if overlap > 0:
                    overlap_text = chunk_text[-overlap:]
                    current_chunk = [overlap_text]
                    current_length = len(overlap_text)
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(split)
            current_length += split_len

        # 别忘了最后一个 chunk
        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks
