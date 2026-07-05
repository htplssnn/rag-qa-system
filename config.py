"""
RAG 知识库问答系统 - 全局配置

所有可调参数集中在这里，方便统一管理。
修改配置后重启程序即可生效。
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    """大语言模型配置（兼容 OpenAI 接口格式）

    支持所有兼容 OpenAI Chat Completions API 的服务商：
    - OpenAI:       https://api.openai.com/v1
    - DeepSeek:     https://api.deepseek.com/v1
    - 通义千问:      https://dashscope.aliyuncs.com/compatible-mode/v1
    - 智谱 GLM:     https://open.bigmodel.cn/api/paas/v4
    - 本地 Ollama:   http://localhost:11434/v1

    使用前请将 api_key 改为你自己的密钥。
    也可以通过环境变量 LLM_API_KEY 覆盖。
    """
    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = "sk-your-api-key-here"
    model: str = "deepseek-chat"
    temperature: float = 0.3       # 问答场景偏低温度，减少胡编
    max_tokens: int = 2048


@dataclass
class EmbeddingConfig:
    """向量化模型配置

    两种模式：
      1. local  —— 使用 sentence-transformers 本地模型，无需 API，离线可用
                   首次运行会自动下载模型（约 100MB）
      2. api    —— 使用 OpenAI 兼容的 Embedding API，速度快但需要联网

    实习生建议先用 local 模式跑通流程，再按需切换。
    """
    mode: str = "local"  # "local" 或 "api"

    # local 模式：sentence-transformers 模型名
    # paraphrase-multilingual-MiniLM-L12-v2 支持中文，体积小，速度快
    local_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    # api 模式
    api_base_url: str = "https://api.deepseek.com/v1"
    api_key: str = "sk-your-api-key-here"
    api_model: str = "text-embedding-3-small"


@dataclass
class SplitConfig:
    """文档切分配置

    chunk_size:    每个文本块的最大字符数
    chunk_overlap: 相邻块之间的重叠字符数（保证上下文连贯）
    """
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class RetrieveConfig:
    """检索配置"""
    top_k: int = 5  # 检索返回最相关的 K 个文本块


@dataclass
class AppConfig:
    """应用总配置"""
    # 数据存储目录
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    vector_db_dir: Path = field(default_factory=lambda: Path("./data/vector_db"))
    upload_dir: Path = field(default_factory=lambda: Path("./data/uploads"))

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)

    def ensure_dirs(self):
        """确保所有数据目录存在"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)


# 全局配置实例 —— 其他模块直接 import 使用
config = AppConfig()
