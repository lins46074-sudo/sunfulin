"""建库脚本：读取 docs/ 下的 .txt / .md 文档，切片后写入 Chroma 向量库。

用法：
    uv run python ingest.py

每次运行都会重建向量库（先删旧的），文档有改动后重跑本脚本即可。
"""

import shutil

import config
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from store import get_embeddings

SUPPORTED_SUFFIXES = {".txt", ".md"}


def load_documents() -> list[Document]:
    """读取 docs/ 下所有支持的文档。"""
    docs: list[Document] = []
    for path in sorted(config.DOCS_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            print(f"[跳过] {path.name}：内容为空")
            continue
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """按中文标点优先切分，尽量避免把句子拦腰截断。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    return splitter.split_documents(docs)


def main() -> None:
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_documents()
    if not docs:
        print(f"[提示] {config.DOCS_DIR} 中没有找到 .txt / .md 文档。")
        print("       请把文档放进去，然后重新运行：uv run python ingest.py")
        return

    chunks = split_documents(docs)
    print(f"[读取] {len(docs)} 个文档 -> 切分为 {len(chunks)} 个片段")

    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
        print("[清理] 已删除旧向量库")

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
    )
    print(f"[完成] 向量库已保存到 {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
