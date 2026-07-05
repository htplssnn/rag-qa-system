"""
RAG 系统基础测试脚本

测试文档加载、文本切分、向量化、向量存储与检索的完整流程。
不需要 LLM API Key，只测试 RAG 的"检索"部分。

运行方式：
    python test_rag.py
"""

import sys
import os
import tempfile
from pathlib import Path

# 确保能 import 项目模块
sys.path.insert(0, str(Path(__file__).parent))

from rag import DocumentLoader, TextSplitter, EmbeddingModel, VectorStore, Retriever
from config import config


def separator(title):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


def test_document_loader():
    """测试文档加载"""
    separator("测试 1: 文档加载")

    loader = DocumentLoader()

    # 创建临时测试文件
    test_content = """人工智能（AI）是计算机科学的一个分支，致力于研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统。

机器学习是人工智能的核心技术之一，它使计算机系统能够从数据中学习并改进，而无需明确编程。

深度学习是机器学习的一个子领域，使用多层神经网络来处理数据。深度学习在图像识别、自然语言处理和语音识别等领域取得了突破性进展。

大语言模型（LLM）是基于深度学习的自然语言处理模型，如 GPT、BERT 等。它们通过在海量文本上预训练，学会了理解和生成人类语言。

RAG（检索增强生成）是一种结合检索和生成的技术。它先从知识库中检索相关文档，再将文档作为上下文交给大模型生成回答，从而减少幻觉问题。"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(test_content)
        temp_path = f.name

    try:
        doc = loader.load(temp_path)
        print(f"  文件名: {doc.metadata['source']}")
        print(f"  内容长度: {len(doc.content)} 字符")
        print(f"  内容前 50 字: {doc.content[:50]}...")
        assert len(doc.content) > 0, "文档内容不应为空"
        print("  [PASS] 文档加载测试通过")
        return doc
    finally:
        os.unlink(temp_path)


def test_text_splitter(doc):
    """测试文本切分"""
    separator("测试 2: 文本切分")

    splitter = TextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.split(doc)

    print(f"  原文长度: {len(doc.content)} 字符")
    print(f"  切分参数: chunk_size=200, chunk_overlap=30")
    print(f"  切分结果: {len(chunks)} 个文本块")
    print()
    for i, chunk in enumerate(chunks):
        print(f"  块 {i + 1} ({len(chunk.content)} 字符): {chunk.content[:60]}...")

    assert len(chunks) > 1, "应该切分出多个块"
    assert all(len(c.content) <= 250 for c in chunks), "块大小不应超出限制太多"
    print("\n  [PASS] 文本切分测试通过")
    return chunks


def test_embedding(chunks):
    """测试向量化"""
    separator("测试 3: 向量化")

    print("  初始化 Embedding 模型（首次运行需下载模型，请耐心等待）...")
    embedding_model = EmbeddingModel()

    texts = [c.content for c in chunks[:3]]  # 只测前 3 块
    vectors = embedding_model.embed(texts)

    print(f"  输入: {len(texts)} 段文本")
    print(f"  输出: {len(vectors)} 个向量")
    print(f"  向量维度: {len(vectors[0])}")
    print(f"  向量前 5 维: {[round(x, 4) for x in vectors[0][:5]]}")

    assert len(vectors) == len(texts), "向量数量应与输入一致"
    assert len(vectors[0]) > 0, "向量维度应大于 0"
    print("\n  [PASS] 向量化测试通过")
    return embedding_model


def test_vector_store(embedding_model, chunks):
    """测试向量存储与检索"""
    separator("测试 4: 向量存储与检索")

    # 使用临时目录
    config.vector_db_dir = Path(tempfile.mkdtemp())
    vector_store = VectorStore(embedding_model)

    # 存入向量
    print(f"  存入 {len(chunks)} 个文本块...")
    vector_store.add_chunks(chunks)
    count = vector_store.count()
    print(f"  向量库当前文本块数: {count}")
    assert count == len(chunks), "存储数量应一致"

    # 检索测试
    retriever = Retriever(embedding_model, vector_store)

    queries = [
        "什么是深度学习？",
        "RAG 技术是什么？",
        "大语言模型有哪些？",
    ]

    for query in queries:
        print(f"\n  查询: 「{query}」")
        results = retriever.retrieve(query, top_k=2)
        print(f"  检索到 {len(results)} 条结果:")
        for i, res in enumerate(results):
            print(f"    {i + 1}. [相似度: {res.score:.4f}] {res.content[:50]}...")

    print("\n  [PASS] 向量存储与检索测试通过")


def test_context_assembly(embedding_model, chunks):
    """测试上下文组装"""
    separator("测试 5: 上下文组装")

    config.vector_db_dir = Path(tempfile.mkdtemp())
    vector_store = VectorStore(embedding_model)
    vector_store.add_chunks(chunks)
    retriever = Retriever(embedding_model, vector_store)

    context, sources, results = retriever.retrieve_with_context("RAG 是什么", top_k=2)

    print(f"  组装的上下文 ({len(context)} 字符):")
    print(f"  {context[:200]}...")
    print(f"  来源: {sources}")
    assert len(context) > 0, "上下文不应为空"
    assert len(sources) > 0, "应有来源信息"
    print("\n  [PASS] 上下文组装测试通过")


def main():
    print("\n" + "=" * 50)
    print("  RAG 知识库问答系统 - 基础测试")
    print("=" * 50)

    try:
        # 测试不依赖网络的模块
        doc = test_document_loader()
        chunks = test_text_splitter(doc)

        # 测试需要模型的模块
        embedding_model = test_embedding(chunks)
        test_vector_store(embedding_model, chunks)
        test_context_assembly(embedding_model, chunks)

        # 总结
        separator("测试总结")
        print("  所有测试通过！")
        print("  RAG 系统的检索部分工作正常。")
        print("  要测试完整的问答功能，请配置 LLM API Key 后运行: python app.py")
        print()

    except Exception as e:
        print(f"\n  [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
