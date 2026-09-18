# 轻量版 LangChain RAG Agent

一个能自己判断「要不要查知识库」的问答智能体。用 LangChain 1.x + Chroma + 中文向量模型实现，
对话模型走 Anthropic 兼容接口（默认 DeepSeek）。

信息不够时会自己换关键词重查，查不到就直说，不编造。

---

## 一、环境要求

- Python 3.10+（当前用 3.13.9）
- uv（依赖管理，已安装 0.11.7）

首次安装会下载约 450MB：torch CPU 版 118MB + chromadb 等依赖，以及 95MB 的向量模型。

## 二、快速开始

### 第 1 步：安装依赖（已装好可跳过）

```bash
uv sync
```

### 第 2 步：放入文档并建库

把你的 `.txt` / `.md` 文档放进 `docs/` 目录，然后执行：

```bash
uv run python ingest.py
```

文档有改动后重跑这一步即可（会重建向量库）。
网页版也可以在左侧边栏点「🔄 重建索引」，效果一样。

### 第 3 步：启动

**方式一：网页版（推荐）**

双击 `run_web.bat`，浏览器会自动打开 <http://localhost:8501>。
关掉那个黑色命令行窗口即停止服务。

**方式二：命令行版**

```bash
uv run python chat.py
```

输入 `/exit` 退出。

## 三、目录结构

```
langchain/
├── config.py        配置集中管理（模型、路径、切片、检索参数）
├── prompts.py       系统提示词（那套轻量 RAG 提示词）
├── store.py         向量模型加载 + Chroma 读写
├── ingest.py        建库脚本：读 docs/ → 切片 → 向量化 → 存库
├── rag.py           检索工具定义 + Agent 组装（两个界面共用）
├── chat.py          命令行对话入口
├── web.py           网页版界面（Streamlit）
├── run.bat          双击启动命令行版
├── run_web.bat      双击启动网页版
├── docs/            你的文档放这里（.txt / .md）
├── chroma_db/       向量库（自动生成，可随时删除重建）
└── .env             个人配置（从 .env.example 复制，已被 git 忽略）
```

## 四、凭证是怎么取的

程序按下面的顺序找凭证，找到就用：

1. 环境变量 `ANTHROPIC_API_KEY`
2. 环境变量 `ANTHROPIC_AUTH_TOKEN` ← 本机当前走这条
3. `.env` 文件里同名变量

**注意**：`.env` 是 `override=False` 加载的，**不会覆盖已存在的系统环境变量**。
换句话说，如果某项在系统里已经设了，改 `.env` 是不生效的，要改系统环境变量。
这样做是为了避免不小心破坏 Claude Code 自身依赖的环境变量。

程序**不会**把 key 写进任何文件。

## 五、换模型 / 换接口

改 `.env`（或系统环境变量）即可，代码不用动：

```ini
# 换模型
RAG_MODEL=deepseek-chat

# 换成官方 Claude（需要你自己的 Anthropic API key）
ANTHROPIC_BASE_URL=https://api.anthropic.com
RAG_MODEL=claude-opus-5
```

前提是你要有对应接口的 key。官方 Claude 需要去 console.anthropic.com 申请。

## 六、可调参数

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RAG_TOP_K` | 4 | 每次检索返回几个片段。答不准可调到 6 |
| `RAG_CHUNK_SIZE` | 500 | 切片长度（字符） |
| `RAG_CHUNK_OVERLAP` | 50 | 切片重叠长度 |
| `RAG_EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | 向量模型 |
| `RAG_MAX_TOKENS` | 2048 | 单次回答最大长度 |

## 七、常见问题

**Q：提示向量库不存在？**
先跑 `uv run python ingest.py` 建库。

**Q：模型下载卡住 / 超时？**
必须走国内镜像。`config.py` 里已默认设置 `HF_ENDPOINT=https://hf-mirror.com`。
若仍失败，检查系统环境变量是否把这个值覆盖成了官方地址。

**Q：`uv sync` 报 403？**
镜像源反爬。本项目已改用官方 PyPI，若你手动配了清华源可能触发此问题。

**Q：中文打印乱码或报 `UnicodeEncodeError`？**
Windows 控制台默认 GBK，`config.py` 已强制把标准输出切到 UTF-8。
若终端本身不是 UTF-8（老式 cmd），建议改用 Windows Terminal 或 Git Bash。

**Q：想换掉示例文档？**
删掉 `docs/sample.md`，放入你自己的文档，重跑 `ingest.py`。
