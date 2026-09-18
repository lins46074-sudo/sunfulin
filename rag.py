"""检索工具定义 + Agent 组装。"""

import config
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

from prompts import SYSTEM_PROMPT
from store import search


def build_llm() -> ChatAnthropic:
    """构造对话模型。

    注意：base_url / api_key 是字段别名，实际字段名分别是
    anthropic_api_url / anthropic_api_key，两种写法等价。
    """
    if not config.API_KEY:
        raise RuntimeError(
            "未找到可用的 API 凭证。\n"
            "请设置环境变量 ANTHROPIC_API_KEY，或把凭证写进项目根目录的 .env。"
        )
    return ChatAnthropic(
        model=config.MODEL_NAME,
        base_url=config.API_BASE_URL,
        api_key=config.API_KEY,
        max_tokens=config.MAX_TOKENS,
    )


@tool
def search_knowledge_base(query: str) -> str:
    """知识库检索：用来查询我的私有文档资料。当用户问文档里的内容时调用这个工具。

    入参 query 是字符串，把用户的问题改写成适合搜索的关键词。
    返回相关文档片段和文档编号。

    适用场景：查询上传的文档、文档里的知识点、文档中的数据。
    不适用：日常常识、不需要看文档的问题。
    """
    results = search(query, k=config.TOP_K)
    if not results:
        return "【知识库没有找到相关内容】"

    parts = []
    for idx, (doc, _score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "未知来源")
        parts.append(f"[{idx}] 来源：{source}\n{doc.page_content}")
    return "\n\n".join(parts)


def build_agent(checkpointer=None):
    """组装 Agent：对话模型 + 检索工具 + 系统提示词。"""
    return create_agent(
        model=build_llm(),
        tools=[search_knowledge_base],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def extract_text(content) -> str:
    """把模型返回内容统一转成纯文本（供命令行版和网页版共用）。

    部分模型会返回 [{'type': 'text', ...}] 这样的内容块列表，
    这里统一摊平成字符串，避免打印出原始结构。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(t for t in texts if t).strip()
    return str(content)
