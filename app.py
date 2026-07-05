"""
RAG 知识库问答系统 - Web 界面

使用 Gradio 构建，提供：
    1. 文档上传与管理（PDF / TXT / Markdown）
    2. 知识库状态查看（已入库文档、文本块数量）
    3. 智能问答对话（流式输出 + 引用来源展示）

运行方式：
    python app.py
    然后在浏览器打开 http://localhost:7860
"""

import os
import shutil
from pathlib import Path

import gradio as gr

from config import config
from rag import DocumentLoader, TextSplitter, EmbeddingModel, VectorStore, Retriever, Generator

# ============================================================
# 初始化各模块
# ============================================================
config.ensure_dirs()

print("=" * 50)
print("RAG 知识库问答系统启动中...")
print("=" * 50)

# 从环境变量读取 API Key（优先级高于 config.py 中的默认值）
if os.getenv("LLM_API_KEY"):
    config.llm.api_key = os.getenv("LLM_API_KEY")
if os.getenv("LLM_BASE_URL"):
    config.llm.base_url = os.getenv("LLM_BASE_URL")
if os.getenv("LLM_MODEL"):
    config.llm.model = os.getenv("LLM_MODEL")

embedding_model = EmbeddingModel()
vector_store = VectorStore(embedding_model)
retriever = Retriever(embedding_model, vector_store)
generator = Generator(retriever)
loader = DocumentLoader()
splitter = TextSplitter(
    chunk_size=config.split.chunk_size,
    chunk_overlap=config.split.chunk_overlap,
)


# ============================================================
# 业务逻辑函数
# ============================================================

def upload_and_index(files):
    """上传文件并加入知识库"""
    if not files:
        return "请选择文件上传", get_kb_status()

    results = []
    total_chunks = 0

    for file in files:
        try:
            # 复制文件到上传目录
            src_path = Path(file.name) if hasattr(file, "name") else Path(file)
            dest_path = config.upload_dir / src_path.name
            shutil.copy2(src_path, dest_path)

            # 加载文档
            doc = loader.load(dest_path)

            # 切分文本
            chunks = splitter.split(doc)

            # 存入向量库
            vector_store.add_chunks(chunks)

            results.append(f"  {src_path.name}: {len(chunks)} 个文本块")
            total_chunks += len(chunks)

        except Exception as e:
            results.append(f"  {src_path.name}: 失败 - {str(e)}")

    summary = f"上传完成，共处理 {len(files)} 个文件，新增 {total_chunks} 个文本块：\n" + "\n".join(results)
    return summary, get_kb_status()


def get_kb_status():
    """获取知识库状态"""
    count = vector_store.count()
    sources = vector_store.get_all_sources()
    if sources:
        source_list = "\n".join(f"  - {s}" for s in sources)
    else:
        source_list = "  (空)"
    return f"知识库文本块数量: {count}\n已入库文档:\n{source_list}"


def clear_kb():
    """清空知识库"""
    vector_store.clear()
    return "知识库已清空", get_kb_status()


def chat(query, history):
    """对话问答（流式输出）

    Gradio 的 ChatInterface 会把历史对话传进来，
    我们每次只处理最新一条问题即可。
    """
    if not query.strip():
        yield "请输入问题"
        return

    if vector_store.count() == 0:
        yield "知识库为空，请先在左侧上传文档。"
        return

    # 流式输出回答
    answer_text = ""
    try:
        gen = generator.answer_stream(query)
        try:
            while True:
                chunk = next(gen)
                answer_text += chunk
                yield answer_text + " ▌"  # 加光标效果
        except StopIteration as e:
            # 获取生成器返回的完整结果
            result = e.value or {}
            answer_text = result.get("answer", answer_text)
            sources = result.get("sources", [])

        # 追加来源信息
        if sources:
            source_str = "、".join(sources)
            answer_text += f"\n\n---\n**参考来源**: {source_str}"

        yield answer_text

    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "401" in error_msg:
            yield f"LLM API 调用失败，请在 config.py 中配置正确的 api_key。\n\n错误: {error_msg}"
        else:
            yield f"生成回答时出错: {error_msg}"


# ============================================================
# Gradio 界面布局
# ============================================================

with gr.Blocks(title="RAG 知识库问答系统", theme=gr.themes.Soft()) as app:
    gr.Markdown("# RAG 知识库问答系统")
    gr.Markdown("上传文档构建知识库，然后向 AI 提问，AI 会基于文档内容回答。")

    with gr.Row():
        # ---------- 左侧：知识库管理 ----------
        with gr.Column(scale=1):
            gr.Markdown("### 知识库管理")

            file_output = gr.Textbox(
                label="上传结果",
                lines=6,
                interactive=False,
            )

            upload_btn = gr.UploadButton(
                "上传文档 (PDF/TXT/MD)",
                file_types=[".pdf", ".txt", ".md", ".markdown"],
                file_count="multiple",
            )

            kb_status = gr.Textbox(
                label="知识库状态",
                value=get_kb_status(),
                lines=6,
                interactive=False,
            )

            clear_btn = gr.Button("清空知识库", variant="stop")

        # ---------- 右侧：问答对话 ----------
        with gr.Column(scale=2):
            gr.Markdown("### 智能问答")
            chat_interface = gr.ChatInterface(
                fn=chat,
                title="知识库问答",
                description="基于上传的文档回答问题，回答末尾会标注信息来源。",
                retry_btn="重试",
                undo_btn="撤销",
                clear_btn="清空对话",
            )

    # ---------- 事件绑定 ----------
    upload_btn.upload(
        upload_and_index,
        inputs=[upload_btn],
        outputs=[file_output, kb_status],
    )

    clear_btn.click(
        clear_kb,
        outputs=[file_output, kb_status],
    )


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
