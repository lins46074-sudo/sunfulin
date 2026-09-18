"""集中管理配置：模型、凭证、路径、切片与检索参数。

所有可调项都可以通过项目根目录的 .env 文件覆盖，环境变量优先级更高。
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 加载 .env（override=False：已存在的系统环境变量优先，不会被文件覆盖）
load_dotenv(BASE_DIR / ".env", override=False)

# HuggingFace 国内镜像。必须在加载向量模型之前设置，否则下载会超时。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _force_utf8() -> None:
    """Windows 控制台默认 GBK，直接打印非 GBK 字符会抛 UnicodeEncodeError。

    这里把标准输出强制切到 UTF-8，避免打印文档片段时随机崩溃。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


_force_utf8()

# ---- 对话模型 ----
MODEL_NAME = os.getenv("RAG_MODEL", "deepseek-chat")
API_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "2048"))

# 凭证：优先标准变量，回退到 Claude Code 使用的 AUTH_TOKEN
API_KEY = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")

# ---- 路径 ----
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

# ---- 向量库 ----
COLLECTION_NAME = "knowledge_base"
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")

# ---- 切片 ----
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))

# ---- 检索 ----
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
