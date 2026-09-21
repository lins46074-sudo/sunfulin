# 📚 本地知识库 RAG 问答智能体

[![CI](https://github.com/lins46074-sudo/sunfulin/actions/workflows/ci.yml/badge.svg)](https://github.com/lins46074-sudo/sunfulin/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C)](https://python.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-21%20passed-brightgreen)](tests/)

> 把私有文档（`.txt` / `.md`）丢进 `docs/` 目录，就能用自然语言向它提问。
> 智能体自己判断要不要查文档，查不到就直说，**绝不编造**，每个回答都标注来源。

基于 **LangChain 1.x** 构建的检索增强生成（RAG）问答系统，支持**命令行**与**网页**双端交互。
向量检索完全在本地运行，**文档内容不出本机**（仅问答文本会发送至模型接口）。

---

## ✨ 核心特性

| 特性 | 说明 |
|---|---|
| 🤖 **Agent 自主决策** | 模型自行判断问题是否需要查文档。日常常识问题直接回答，不做无谓检索 |
| 🔄 **查询改写重试** | 首轮检索结果不足以回答时，自动更换关键词再查一次 |
| 🚫 **拒绝编造** | 只依据检索到的文档作答并标注来源编号；库中没有的内容，明确回复「知识库没有找到相关内容」 |
| 💬 **多轮对话** | 基于 Checkpointer 维护会话上下文，支持「那免费版呢？」这类省略式追问 |
| 🔍 **检索可溯源** | 网页端每条回答下方可展开，查看本轮检索到的原始文档片段 |
| 🧩 **双端复用** | 命令行与网页共用同一套检索工具与 Agent 组装逻辑，零重复代码 |
| 🔌 **模型可插拔** | 兼容任意 Anthropic Messages API 端点（DeepSeek / 官方 Claude 等），改配置即可切换 |
| ✅ **测试与 CI** | 21 个单元测试，无需 API 凭证即可运行；GitHub Actions 双 Python 版本验证 |

## 🏗 架构

```
                          用户提问
                             │
                             ▼
        ┌────────────────────────────────────────────┐
        │  Agent（LangChain create_agent）            │
        │  ├─ 对话模型：Anthropic Messages API 兼容    │
        │  └─ 系统提示词：检索规则 / 来源标注 / 拒答    │
        └───────────────────┬────────────────────────┘
                            │ 模型自主判断是否需要查文档
                  ┌─────────┴─────────┐
              需要 │                   │ 不需要
                  ▼                   ▼
        ┌────────────────────┐  ┌──────────────────┐
        │ 检索工具            │  │ 直接回答常识问题  │
        │ (Tool Calling)     │  └──────────────────┘
        └─────────┬──────────┘
                  ▼
        ┌────────────────────────────────┐
        │ 向量检索（本地执行）             │
        │ Chroma 持久化 + bge-small-zh   │
        └─────────┬──────────────────────┘
                  ▼
        ┌────────────────────────────────┐
        │ 返回带编号的文档片段 [1] [2] …  │
        └─────────┬──────────────────────┘
                  ▼
            生成回答 + 标注来源编号
```

设计取舍与实现细节见 **[docs/DESIGN.md](docs/DESIGN.md)**。

## 🖥 界面预览

**网页版（Streamlit）**

```
┌────────────────────┬──────────────────────────────────────┐
│  📚 知识库          │  💬 知识库问答                        │
│                    │                                      │
│  ✅ 已收录 1 个文档  │   你：团队版怎么收费？                │
│                    │                                      │
│  ┌──────────────┐  │   助手：团队版每人每月 35 元，        │
│  │ 🔄 重建索引   │  │        5 人起订，含协作编辑 【1】     │
│  └──────────────┘  │        🔍 查看本轮检索到的内容 ▾      │
│  ┌──────────────┐  │                                      │
│  │ 🗑 清空对话   │  │  ┌────────────────────────────────┐  │
│  └──────────────┘  │  │ 输入你的问题…            [发送] │  │
│                    │  └────────────────────────────────┘  │
│  模型：deepseek-chat│                                      │
└────────────────────┴──────────────────────────────────────┘
```

**命令行版**

```
================ 轻量版 RAG 问答智能体 ================
· 直接提问，会用知识库检索工具查你的私有文档
· 输入 /exit 退出
=======================================================
你 > 团队版怎么收费？
助手 > 团队版每人每月 35 元，5 人起订，含协作编辑与权限管理 【1】
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)（依赖管理）

> 首次安装需下载约 450MB：torch CPU 版 118MB + 依赖包 + 中文向量模型 95MB。

### 第 1 步 · 安装依赖

```bash
uv sync
```

<details>
<summary>不想用 uv？用 pip 的备选方式</summary>

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

</details>

### 第 2 步 · 配置凭证

复制 `.env.example` 为 `.env`，填入你的 API key：

```ini
ANTHROPIC_API_KEY=sk-你的key
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
RAG_MODEL=deepseek-chat
```

本项目兼容任意 **Anthropic Messages API 格式**的服务端点。切换到官方 Claude 只需改两行：

```ini
ANTHROPIC_BASE_URL=https://api.anthropic.com
RAG_MODEL=claude-opus-5
```

### 第 3 步 · 放入文档并建立索引

把 `.txt` / `.md` 文档放进 `docs/` 目录，然后执行：

```bash
uv run python ingest.py
```

文档有改动后重跑此命令即可（会先清空再重建索引）。

### 第 4 步 · 启动

**网页版（推荐）**

```bash
uv run streamlit run web.py
```

Windows 用户也可直接双击 `run_web.bat`，浏览器会自动打开 <http://localhost:8501>。
关掉命令行窗口即停止服务。

**命令行版**

```bash
uv run python chat.py
```

输入 `/exit` 退出。

## 📁 目录结构

```
langchain/
├── config.py               配置集中管理（模型、凭证、路径、切片、检索参数）
├── prompts.py              系统提示词（检索规则 / 来源标注 / 拒答约束）
├── store.py                向量模型加载 + Chroma 读写
├── ingest.py               建库脚本：读 docs/ → 切片 → 向量化 → 存库
├── rag.py                  检索工具定义 + Agent 组装 + 输出解析（双端共用）
├── chat.py                 命令行对话入口
├── web.py                  网页版界面（Streamlit）
├── run.bat                 双击启动命令行版（Windows）
├── run_web.bat             双击启动网页版（Windows）
├── tests/test_rag.py       单元测试（不依赖 API 凭证与向量库）
├── docs/
│   ├── sample.md           示例文档（可删除，换成你自己的）
│   └── DESIGN.md           技术设计说明：设计决策与取舍
├── chroma_db/              向量库（自动生成，可随时删除重建）
└── .env                    个人配置（已被 .gitignore 忽略，不会被提交）
```

## 🧪 测试

```bash
uv run pytest -q
```

21 个用例全部**不依赖 API 凭证，也不依赖已建好的向量库**，因此可以在 CI 与无网络环境下直接运行。

覆盖重点是容易静默出错的逻辑，而非追求覆盖率数字：

- 中文切片是否真的没截断句子（含用默认分隔符做的对照组）
- 来源元数据是否跟着切片走（丢了就无法标注出处）
- 工具输出是否带编号与来源名（模型能否标注引用全靠它）
- 检索条数是否真的走配置（而非硬编码）
- 内容解析能否容忍多种返回结构

外部依赖（向量检索）用 `monkeypatch` 替换，只测格式化与流转逻辑。
CI 配置见 [.github/workflows/ci.yml](.github/workflows/ci.yml)，在 Python 3.10 与 3.13 上双版本验证。

## ⚙️ 可调参数

全部通过 `.env` 或系统环境变量覆盖，代码无需改动：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RAG_TOP_K` | `4` | 每次检索返回的片段数。答案不准时可调至 6 |
| `RAG_CHUNK_SIZE` | `500` | 切片长度（字符） |
| `RAG_CHUNK_OVERLAP` | `50` | 相邻切片的重叠长度，避免语义被截断 |
| `RAG_EMBED_MODEL` | `BAAI/bge-small-zh-v1.5` | 向量模型 |
| `RAG_MAX_TOKENS` | `2048` | 单次回答的最大长度 |

## 🔧 工作原理

**1. 中文友好的切片策略**

普通按固定字数切片会把中文句子拦腰截断，导致召回片段语义不完整。
本项目自定义了中文标点优先的分隔符序列：

```python
separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
```

优先在段落、句号、问号等自然边界切分，把句子被截断的概率降到最低。

**2. 检索与生成分离**

检索工具独立于 Agent 实现，返回值是**带编号的格式化文本**（`[1] 来源：xxx`），
因此模型在生成回答时天然知道每个片段来自哪份文档，
来源标注不需要靠提示词「求」模型做——**能用数据结构保证的事，不交给提示词**。

**3. 幻觉抑制**

系统提示词中明确约束：只依据检索内容作答、必须标注来源编号、
知识库没有的内容必须回复固定话术。配合 `TOP_K` 召回，让模型在「没查到」时有明确退路，
而不是自行编造。（注：提示词约束能显著降低幻觉，但并非 100% 可靠，详见设计文档。）

## ❓ 常见问题

**Q：提示「向量库不存在」？**
先执行 `uv run python ingest.py` 建立索引。

**Q：模型下载卡住 / 超时？**
需要走国内镜像。`config.py` 已默认设置 `HF_ENDPOINT=https://hf-mirror.com`。
若仍失败，检查系统环境变量是否将该值覆盖成了官方地址。

**Q：`uv sync` 报 403？**
部分国内镜像源对特定 wheel 返回 403。本项目默认使用官方 PyPI，请勿手动改回镜像源。

**Q：中文打印乱码 / 报 `UnicodeEncodeError`？**
Windows 控制台默认 GBK 编码，`config.py` 已强制将标准输出切到 UTF-8。
若终端本身不支持 UTF-8（老式 cmd），建议改用 Windows Terminal 或 Git Bash。

**Q：改了 `.env` 但没生效？**
Python 在启动时读取 `.env`，修改后需**重启程序**（关掉窗口重新双击）。

**Q：想换掉示例文档？**
删除 `docs/sample.md`，放入自己的文档，重跑 `ingest.py`。

## 🛠 技术栈

| 组件 | 选型 |
|---|---|
| 语言 / 依赖管理 | Python 3.10+ / uv |
| Agent 框架 | LangChain 1.x（`create_agent` + Tool Calling） |
| 向量数据库 | Chroma（本地持久化） |
| 向量模型 | BAAI/bge-small-zh-v1.5（中文优化，95MB，CPU 推理） |
| 网页界面 | Streamlit |
| 模型接口 | Anthropic Messages API（兼容 DeepSeek / Claude 等） |
| 测试 | pytest |

## 📖 相关文档

| 文档 | 内容 |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | 技术设计说明：关键决策、取舍与已知局限 |
| [RESUME.md](RESUME.md) | 项目笔记：设计问答与后续改进计划 |

## 📄 许可证

[MIT License](LICENSE)
