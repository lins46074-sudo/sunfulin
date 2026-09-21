"""单元测试：覆盖切片、文档加载、工具输出格式化与内容解析。

设计原则：**不依赖 API 凭证，也不依赖已建好的向量库**，
因此可以在 CI 与无网络环境下直接运行。
"""

import re

import pytest
from langchain_core.documents import Document

import config
import ingest
import rag

# 60 句结构规整的中文，方便断言「句子是否被截断」
SENTENCE = "这是第{n}句用来测试中文切分逻辑的示例文本内容。"
SENTENCE_TAIL = "示例文本内容"
LONG_TEXT = "".join(SENTENCE.format(n=i) for i in range(1, 61))


# --------------------------- 切片 ---------------------------
def test_split_produces_multiple_chunks():
    chunks = ingest.split_documents([Document(page_content=LONG_TEXT)])
    assert len(chunks) > 1


def test_split_respects_chunk_size():
    chunks = ingest.split_documents([Document(page_content=LONG_TEXT)])
    assert all(len(c.page_content) <= config.CHUNK_SIZE for c in chunks)


def test_split_never_cuts_a_sentence_in_half():
    """中文标点优先切分：每个句子要么完整，要么在句号处断开。

    默认按固定字数切片会把句子拦腰截断，导致召回片段语义不完整，
    这条用例专门守住这个回归。
    """
    chunks = ingest.split_documents([Document(page_content=LONG_TEXT)])
    for chunk in chunks:
        text = chunk.page_content
        for match in re.finditer(r"这是第\d+句[^。]*", text):
            fragment = match.group()
            ends_with_full_sentence = fragment.endswith(SENTENCE_TAIL)
            followed_by_period = text[match.end() : match.end() + 1] == "。"
            assert ends_with_full_sentence or followed_by_period, f"句子被截断：{fragment!r}"


def test_split_keeps_source_metadata():
    """来源元数据必须跟着切片走，否则回答无法标注出处。"""
    chunks = ingest.split_documents(
        [Document(page_content=LONG_TEXT, metadata={"source": "handbook.md"})]
    )
    assert all(c.metadata["source"] == "handbook.md" for c in chunks)


# --------------------------- 文档加载 ---------------------------
def test_load_documents_only_reads_supported_suffix(tmp_path, monkeypatch):
    (tmp_path / "keep.md").write_text("Markdown 内容", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("纯文本内容", encoding="utf-8")
    (tmp_path / "skip.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "skip.docx").write_bytes(b"PK\x03\x04")
    monkeypatch.setattr(config, "DOCS_DIR", tmp_path)

    docs = ingest.load_documents()

    assert sorted(d.metadata["source"] for d in docs) == ["keep.md", "keep.txt"]


def test_load_documents_skips_empty_file(tmp_path, monkeypatch):
    (tmp_path / "blank.md").write_text("   \n\n  ", encoding="utf-8")
    (tmp_path / "real.md").write_text("有内容", encoding="utf-8")
    monkeypatch.setattr(config, "DOCS_DIR", tmp_path)

    docs = ingest.load_documents()

    assert [d.metadata["source"] for d in docs] == ["real.md"]


def test_load_documents_walks_subdirectories(tmp_path, monkeypatch):
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.md").write_text("子目录内容", encoding="utf-8")
    monkeypatch.setattr(config, "DOCS_DIR", tmp_path)

    docs = ingest.load_documents()

    assert [d.metadata["source"] for d in docs] == ["deep.md"]


# --------------------------- 工具输出 ---------------------------
def test_tool_formats_results_with_numbered_sources(monkeypatch):
    """工具返回值必须带 [1] [2] 编号和来源名，模型才能标注引用。"""
    fake = [
        (Document(page_content="团队版每人每月 35 元。", metadata={"source": "price.md"}), 0.12),
        (Document(page_content="教育邮箱享五折。", metadata={"source": "faq.md"}), 0.31),
    ]
    monkeypatch.setattr(rag, "search", lambda query, k: fake)

    output = rag.search_knowledge_base.invoke("团队版怎么收费")

    assert "[1] 来源：price.md" in output
    assert "[2] 来源：faq.md" in output
    assert "团队版每人每月 35 元。" in output


def test_tool_reports_empty_when_nothing_found(monkeypatch):
    monkeypatch.setattr(rag, "search", lambda query, k: [])

    output = rag.search_knowledge_base.invoke("不存在的主题")

    assert output == "【知识库没有找到相关内容】"


def test_tool_falls_back_for_unknown_source(monkeypatch):
    fake = [(Document(page_content="无元数据的片段"), 0.5)]
    monkeypatch.setattr(rag, "search", lambda query, k: fake)

    output = rag.search_knowledge_base.invoke("任意问题")

    assert "[1] 来源：未知来源" in output


def test_tool_passes_top_k_to_search(monkeypatch):
    """检索条数必须走配置，改 .env 就能生效。"""
    captured = {}

    def fake_search(query, k):
        captured["k"] = k
        return []

    monkeypatch.setattr(rag, "search", fake_search)
    rag.search_knowledge_base.invoke("任意问题")

    assert captured["k"] == config.TOP_K


# --------------------------- 内容解析 ---------------------------
def test_extract_text_plain_string():
    assert rag.extract_text("直接就是字符串") == "直接就是字符串"


def test_extract_text_from_content_blocks():
    content = [
        {"type": "text", "text": "第一段"},
        {"type": "tool_use", "id": "x", "name": "search", "input": {}},
        {"type": "text", "text": "第二段"},
    ]
    assert rag.extract_text(content) == "第一段\n第二段"


def test_extract_text_ignores_empty_blocks():
    content = [{"type": "text", "text": ""}, {"type": "text", "text": "有内容"}]
    assert rag.extract_text(content) == "有内容"


def test_extract_text_tolerates_unknown_type():
    assert rag.extract_text(12345) == "12345"


# --------------------------- 配置 ---------------------------
@pytest.mark.parametrize(
    "name,default",
    [
        ("CHUNK_SIZE", 500),
        ("CHUNK_OVERLAP", 50),
        ("TOP_K", 4),
        ("MODEL_NAME", "deepseek-chat"),
        ("EMBED_MODEL", "BAAI/bge-small-zh-v1.5"),
    ],
)
def test_config_defaults(name, default):
    """默认值即文档中承诺的值，改默认值必须同步改 README。"""
    assert getattr(config, name) == default


def test_chunk_overlap_smaller_than_chunk_size():
    """重叠必须小于块长，否则切片会陷入死循环。"""
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
