# RAG 知识库问答系统

> 一个适合 AI 应用开发实习生学习和展示的 RAG（检索增强生成）项目。

## 项目简介

本项目实现了一个完整的 RAG 知识库问答系统：用户上传文档（PDF / TXT / Markdown），系统自动将文档切分、向量化并存入向量数据库；用户提问时，系统先检索最相关的文档片段，再交给大语言模型基于这些片段生成回答，并标注信息来源。

### 为什么选这个项目？

- **覆盖 AI 应用开发全链路**：文档处理 → 文本切分 → 向量化 → 向量检索 → Prompt 工程 → 大模型调用
- **面试高频考点**：RAG 是目前大模型应用落地最主流的方案，面试必问
- **可扩展性强**：可以在此基础上加 Re-ranking、多路召回、Agent 工具调用等进阶能力
- **能实际用起来**：上传自己的学习笔记 / 公司文档，构建个人知识助手

## 系统架构

```
用户上传文档                    用户提问
     │                            │
     ▼                            ▼
┌──────────┐               ┌──────────┐
│ 文档加载  │               │ 问题向量化│
│ (PDF/TXT)│               │ Embedding │
└────┬─────┘               └─────┬────┘
     │                           │
     ▼                           ▼
┌──────────┐               ┌──────────┐
│ 文本切分  │               │ 向量检索  │
│ Splitter │               │  Search  │
└────┬─────┘               └─────┬────┘
     │                           │
     ▼                           ▼
┌──────────┐               ┌──────────┐
│ 向量化    │               │ 组装Prompt│
│ Embedding│               │ + 上下文  │
└────┬─────┘               └─────┬────┘
     │                           │
     ▼                           ▼
┌──────────┐               ┌──────────┐
│ 存入向量库 │               │ 大模型生成│
│ ChromaDB │               │   LLM    │
└──────────┘               └─────┬────┘
                                  │
                                  ▼
                           ┌──────────┐
                           │ 回答+来源 │
                           └──────────┘
```

## 项目结构

```
rag-qa-system/
├── app.py                  # Gradio Web 界面（主入口）
├── config.py               # 全局配置（模型、切分参数等）
├── requirements.txt        # Python 依赖
├── rag/                    # RAG 核心模块
│   ├── __init__.py
│   ├── document_loader.py  # 文档加载（PDF/TXT/MD）
│   ├── text_splitter.py    # 递归字符文本切分
│   ├── embedding.py        # 向量化（本地/API 双模式）
│   ├── vector_store.py     # ChromaDB 向量存储与检索
│   ├── retriever.py        # 检索器
│   └── generator.py        # 大模型回答生成
└── data/                   # 运行时自动创建
    ├── uploads/            # 上传的原始文件
    └── vector_db/          # ChromaDB 持久化数据
```

## 快速开始

### 1. 安装依赖

```bash
cd rag-qa-system
pip install -r requirements.txt
```

### 2. 配置大模型 API

编辑 `config.py`，修改 `LLMConfig` 中的配置：

```python
@dataclass
class LLMConfig:
    base_url: str = "https://api.deepseek.com/v1"   # 改成你的 API 地址
    api_key: str = "sk-your-api-key-here"            # 改成你的 API Key
    model: str = "deepseek-chat"                     # 改成模型名
```

也可以通过环境变量配置（优先级更高）：

```bash
export LLM_API_KEY="sk-your-key"
export LLM_BASE_URL="https://api.deepseek.com/v1"
export LLM_MODEL="deepseek-chat"
```

#### 支持的模型服务商

| 服务商 | base_url | 推荐模型 |
|--------|----------|---------|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| 本地 Ollama | `http://localhost:11434/v1` | `qwen2.5:7b` |

### 3. 启动系统

```bash
python app.py
```

浏览器打开 `http://localhost:7860` 即可使用。

### 4. 使用流程

1. 在左侧点击「上传文档」按钮，选择 PDF / TXT / Markdown 文件
2. 等待处理完成，查看知识库状态
3. 在右侧对话框提问，AI 会基于文档内容回答并标注来源

## Embedding 模式说明

系统支持两种向量化模式，在 `config.py` 的 `EmbeddingConfig` 中切换：

### 本地模式（默认，推荐入门）

```python
mode: str = "local"
local_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
```

- 使用 `sentence-transformers` 本地模型
- 首次运行自动下载模型（约 100MB）
- 离线可用，不消耗 API 额度

### API 模式

```python
mode: str = "api"
api_base_url: str = "https://api.deepseek.com/v1"
api_key: str = "sk-your-key"
api_model: str = "text-embedding-3-small"
```

- 调用 OpenAI 兼容的 Embedding API
- 速度快，但需要联网和 API Key

## RAG 核心概念详解

### 1. 文本切分（Chunking）

为什么要切分？
- 大模型上下文窗口有限，不能把整本书塞进去
- 切成小块后，可以只检索最相关的几块，减少噪声、节省 Token

切分参数：
- `chunk_size=500`：每块最多 500 字符。太小会丢失上下文，太大会引入噪声
- `chunk_overlap=50`：相邻块重叠 50 字符，避免句子被切断

### 2. 向量化（Embedding）

把文本转成向量（一串数字），使得语义相近的文本在向量空间中距离也近。
用户提问后，把问题也转成向量，通过计算向量距离就能找到最相关的文档片段。

### 3. 向量检索（Retrieval）

使用余弦相似度（Cosine Similarity）衡量向量的相似程度。
ChromaDB 内部使用 HNSW 算法加速近似最近邻搜索。

### 4. Prompt 工程

系统的 Prompt 明确要求模型"只根据参考资料回答"，这是减少大模型幻觉（Hallucination）的关键。

## 进阶方向

完成基础项目后，可以从以下方向扩展：

| 方向 | 说明 | 难度 |
|------|------|------|
| 多路召回 | 同时用向量检索 + 关键词检索（BM25），提升召回率 | 中 |
| Re-Ranking | 用 Cross-Encoder 对检索结果重排序，提升精度 | 中 |
| 引用定位 | 回答时标注具体来自哪个片段的哪一段 | 中 |
| 多轮对话 | 结合对话历史理解上下文追问 | 中 |
| Agent 工具 | 让系统能调用搜索引擎、计算器等外部工具 | 高 |
| 评估体系 | 构建测试集，量化评估检索准确率和回答质量 | 高 |
| 流式上传 | 支持目录监控，自动索引新增文档 | 低 |

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| Web 界面 | Gradio | 快速构建 AI 应用界面 |
| 向量数据库 | ChromaDB | 轻量级，本地持久化，无需部署 |
| Embedding | sentence-transformers | 本地模型，支持中文 |
| 大模型调用 | OpenAI Python SDK | 兼容多家服务商 |
| PDF 解析 | pdfplumber | 纯 Python，跨平台 |
