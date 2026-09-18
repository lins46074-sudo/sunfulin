"""网页版 RAG 智能体（Streamlit）。

启动方式：
    双击 run_web.bat
或  在项目目录执行： uv run streamlit run web.py
"""

import shutil
import uuid

import streamlit as st
from langchain.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

import config
from ingest import load_documents, split_documents
from rag import build_agent, extract_text
from store import get_embeddings

st.set_page_config(page_title="知识库问答", page_icon="📚", layout="wide")


# ==================== 资源缓存 ====================
@st.cache_resource(show_spinner=False)
def get_agent():
    """整个进程只构建一次 Agent，避免每次对话都重新初始化。"""
    return build_agent(checkpointer=InMemorySaver())


# ==================== 会话状态 ====================
def init_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = []


def reset_chat() -> None:
    """清空对话：换一个新的 thread_id，等于让 Agent 忘掉上下文。"""
    st.session_state.thread_id = uuid.uuid4().hex
    st.session_state.messages = []


# ==================== 知识库操作 ====================
def list_docs() -> list[str]:
    if not config.DOCS_DIR.exists():
        return []
    return sorted(
        p.name
        for p in config.DOCS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}
    )


def rebuild_index() -> tuple[int, int]:
    """重建向量库，返回 (文档数, 片段数)。"""
    from langchain_chroma import Chroma

    docs = load_documents()
    if not docs:
        return 0, 0

    chunks = split_documents(docs)
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
    )
    return len(docs), len(chunks)


def latest_retrieval(messages) -> list[str]:
    """取出本轮的工具检索结果，即最后一条用户提问之后的 ToolMessage。"""
    last_user = 0
    for idx, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            last_user = idx
    return [m.content for m in messages[last_user:] if isinstance(m, ToolMessage)]


def ask(question: str) -> tuple[str, list[str]]:
    """向 Agent 提问，返回 (回答文本, 本轮检索到的片段)。"""
    try:
        agent = get_agent()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": st.session_state.thread_id}},
        )
    except Exception as exc:
        return f"❌ 出错了：{type(exc).__name__}: {exc}", []

    answer = extract_text(result["messages"][-1].content)
    return answer, latest_retrieval(result["messages"])


def render_sources(sources) -> None:
    """把本轮检索到的文档片段折叠展示，方便核对答案依据。"""
    if not sources:
        return
    with st.expander(f"🔍 查看本轮检索到的内容（{len(sources)} 批）"):
        for idx, text in enumerate(sources, start=1):
            if len(sources) > 1:
                st.markdown(f"**第 {idx} 次检索**")
            st.text(text)


# ==================== 主界面 ====================
def main() -> None:
    init_state()

    # ---------- 侧边栏 ----------
    with st.sidebar:
        st.header("📚 知识库")

        docs = list_docs()
        if docs:
            st.success(f"已收录 {len(docs)} 个文档")
            with st.expander("查看文档列表"):
                for name in docs:
                    st.markdown(f"- {name}")
        else:
            st.warning("docs/ 目录下暂无 .txt / .md 文档")

        if st.button("🔄 重建索引", use_container_width=True):
            with st.spinner("正在重建索引…"):
                try:
                    n_docs, n_chunks = rebuild_index()
                    if n_docs == 0:
                        st.warning("没有找到可索引的文档")
                    else:
                        st.success(f"完成：{n_docs} 个文档 → {n_chunks} 个片段")
                except Exception as exc:
                    st.error(f"重建失败：{type(exc).__name__}: {exc}")

        st.divider()
        st.caption(f"对话模型：{config.MODEL_NAME}")
        st.caption(f"检索条数：{config.TOP_K}")
        st.caption(f"API 凭证：{'已配置 ✅' if config.API_KEY else '未配置 ❌'}")

        if st.button("🗑 清空对话", use_container_width=True):
            reset_chat()
            st.rerun()

    # ---------- 主区域 ----------
    st.title("💬 知识库问答")
    st.caption("直接提问即可。需要查文档时，Agent 会自己判断并调用检索工具。")

    if not config.API_KEY:
        st.error(
            "未检测到 API 凭证。请把有效的 key 填进项目根目录的 .env：\n\n"
            "`ANTHROPIC_API_KEY=sk-你的key`"
        )

    # 历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_sources(msg.get("sources"))

    # 输入
    prompt = st.chat_input("输入你的问题…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并思考…"):
            answer, sources = ask(prompt)
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


if __name__ == "__main__":
    main()
