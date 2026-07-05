"""
文档加载模块

负责读取不同格式的文件，统一输出 Document 对象。
支持格式：PDF、TXT、Markdown

这是 RAG 流程的第一步：把原始文件变成程序能处理的文本。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Document:
    """统一的文档数据结构

    content:  文本内容
    metadata: 元数据（文件名、页码等），会跟着文本块一起存入向量库，
              方便回答时标注"这段话来自哪个文件"
    """
    content: str
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    """文档加载器

    根据文件后缀自动选择对应的解析方法。
    """

    # 后缀 -> 解析方法的映射
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

    def load(self, file_path: str | Path) -> Document:
        """加载单个文件，返回 Document 对象"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {ext}，仅支持 {self.SUPPORTED_EXTENSIONS}")

        if ext == ".pdf":
            content = self._load_pdf(path)
        else:
            # TXT 和 Markdown 都是纯文本，直接读取
            content = self._load_text(path)

        return Document(
            content=content,
            metadata={"source": path.name, "file_path": str(path)}
        )

    def load_many(self, file_paths: List[str | Path]) -> List[Document]:
        """批量加载多个文件"""
        return [self.load(fp) for fp in file_paths]

    def _load_pdf(self, path: Path) -> str:
        """使用 pdfplumber 解析 PDF，逐页提取文本"""
        import pdfplumber

        pages_text = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    pages_text.append(text.strip())

        return "\n\n".join(pages_text)

    def _load_text(self, path: Path) -> str:
        """读取纯文本文件（TXT / Markdown）

        使用 utf-8 编码，兼容部分文件用 gbk 编码的情况。
        """
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="gbk")
