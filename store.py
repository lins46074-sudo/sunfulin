"""向量库访问层：向量模型加载 + Chroma 读写。"""

# 这一行必须排在 langchain_huggingface 之前！
# huggingface_hub 在「被导入时」就会读取 HF_ENDPOINT 并缓存下来，
# 晚于它设置镜像环境变量将不生效，模型下载会卡住。
import config

from functools import lru_cache
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """加载中文向量模型。进程内只加载一次，避免重复占用内存。"""
    print(f"[向量模型] 正在加载 {config.EMBED_MODEL} ...")
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore() -> Chroma:
    """打开已建好的向量库。"""
    if not config.CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"向量库不存在：{config.CHROMA_DIR}\n"
            f"请先把文档放进 docs/ 目录，然后运行：python ingest.py"
        )
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


def search(query: str, k: int) -> List[Tuple[Document, float]]:
    """检索最相关的 k 个片段，返回 (文档, 距离) 列表，距离越小越相关。"""
    return get_vectorstore().similarity_search_with_score(query, k=k)
