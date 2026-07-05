# RAG 知识库问答系统（零依赖版）

> 无需安装任何第三方包，修改 API Key 即可运行。

## 快速开始

### 1. 设置 API Key

```bash
# Windows
set LLM_API_KEY=sk-your-key-here
set LLM_BASE_URL=https://api.deepseek.com/v1
set LLM_MODEL=deepseek-chat

# 或使用其他兼容 OpenAI 的 API
# 通义千问: set LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# OpenAI:    set LLM_BASE_URL=https://api.openai.com/v1
```

### 2. 启动系统

```bash
python server_simple.py
```

浏览器打开 `http://localhost:8080`

### 3. 使用

1. 在左侧**粘贴文档内容**或上传 TXT/MD 文件
2. 点击「上传粘贴内容」
3. 在右侧**输入问题**，点击发送
4. AI 会基于文档内容回答，并标注来源

> 未设置 API Key 时，系统使用**模拟模式**（关键词检索 + 显示原文），仍可验证完整流程。

## 文件说明

| 文件 | 说明 |
|------|------|
| `server_simple.py` | Web 服务器（零依赖，Python 标准库实现） |
| `rag_simple.py` | RAG 核心逻辑（可作为模块 import） |
| `test_simple.py` | 测试脚本（离线测试无需 API Key） |
| `config.py` | 原版配置（完整版用） |
| `app.py` | 原版 Gradio 界面（需安装依赖） |

## 离线测试（无需 API Key）

```bash
python test_simple.py
```

离线测试会验证：文档加载 → 文本切分 → 向量存储 → 检索逻辑。

## 完整版（需安装依赖）

完整版支持 PDF 解析、Gradio 界面、本地 Embedding 模型，需安装：

```bash
pip install gradio chromadb openai sentence-transformers pdfplumber
```

然后运行：
```bash
python app.py
```

## 支持的 API

| 服务商 | LLM_BASE_URL | 模型示例 |
|--------|--------------|---------|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Ollama 本地 | `http://localhost:11434/v1` | `qwen2.5:7b` |

## 项目结构（核心逻辑）

```
用户上传文档
    → 加载文本（TXT/MD，或 PDF+pdfplumber）
    → 递归切分（chunk_size=400, overlap=50）
    → 调用 Embedding API 向量化
    → 存入 JSON 文件（data/vector_db.json）
    → 用户提问
    → 问题向量化
    → 余弦相似度检索 Top-K 片段
    → 组装 Prompt（参考资料 + 问题）
    → 调用大模型生成回答
    → 返回回答 + 来源标注
```
