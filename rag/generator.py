"""
生成模块

这是 RAG 中"G"（Generation）的部分：
    把检索到的相关文档片段 + 用户问题组装成 Prompt，
    交给大语言模型生成最终回答。

关键设计：
    1. Prompt 模板 —— 明确告诉模型"只根据参考资料回答"，避免胡编（幻觉）
    2. 流式输出   —— 逐字返回，用户体验更好
    3. 引用来源   —— 回答末尾标注信息来自哪些文档
"""

from typing import Generator, List, Optional

from openai import OpenAI

from config import config
from .retriever import Retriever


# 系统提示词：定义大模型的角色和行为规则
SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请根据下面提供的【参考资料】来回答用户的问题。

回答规则：
1. 只根据参考资料中的内容回答，不要编造资料中没有的信息。
2. 如果参考资料中没有相关内容，请诚实地说"根据知识库中的内容，我无法回答这个问题"。
3. 回答时请清晰、准确、有条理。
4. 如果参考资料中有多个相关片段，请综合它们的信息。
5. 回答末尾标注信息来源，格式如：[来源：文件名]"""


class Generator:
    """大模型回答生成器"""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self._client = None

    def _get_client(self) -> OpenAI:
        """延迟创建 OpenAI 客户端"""
        if self._client is None:
            self._client = OpenAI(
                base_url=config.llm.base_url,
                api_key=config.llm.api_key,
            )
        return self._client

    def answer(self, query: str) -> dict:
        """完整的 RAG 问答流程（非流式）

        返回 dict:
            answer:  回答文本
            sources: 引用来源列表
            contexts: 检索到的参考片段
        """
        # 1. 检索相关文档
        context, sources, results = self.retriever.retrieve_with_context(query)

        if not context:
            return {
                "answer": "知识库为空，请先上传文档后再提问。",
                "sources": [],
                "contexts": [],
            }

        # 2. 组装 Prompt
        user_prompt = self._build_prompt(query, context)

        # 3. 调用大模型
        client = self._get_client()
        response = client.chat.completions.create(
            model=config.llm.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": sources,
            "contexts": [r.content for r in results],
        }

    def answer_stream(self, query: str) -> Generator[str, None, dict]:
        """流式问答（逐字输出）

        用法:
            gen = generator.answer_stream("什么是RAG？")
            for chunk_text in gen:
                print(chunk_text, end="")  # 逐块输出
            result = gen.send(None)  # 获取最终结果（含 sources）

        返回:
            生成器，yield 回答文本片段，最后 return 完整结果 dict
        """
        # 1. 检索
        context, sources, results = self.retriever.retrieve_with_context(query)

        if not context:
            yield "知识库为空，请先上传文档后再提问。"
            return {"answer": "", "sources": [], "contexts": []}

        # 2. 组装 Prompt
        user_prompt = self._build_prompt(query, context)

        # 3. 流式调用大模型
        client = self._get_client()
        stream = client.chat.completions.create(
            model=config.llm.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            stream=True,
        )

        full_answer = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                full_answer += text
                yield text

        # 返回完整结果（通过生成器的 return 值）
        return {
            "answer": full_answer,
            "sources": sources,
            "contexts": [r.content for r in results],
        }

    def _build_prompt(self, query: str, context: str) -> str:
        """组装用户提示词：把检索到的上下文和问题拼在一起"""
        return f"""【参考资料】
{context}

【用户问题】
{query}

请根据以上参考资料回答用户的问题。如果参考资料不足以回答，请说明。"""
