# sunfulin
技术栈：Python3.13、LangChain1.4、Chroma、BAAI/bge‑small‑zh‑v1.5、Streamlit、uv、Anthropic Messages API  项目简介 面向私有.txt/.md文档的检索增强生成 RAG 智能问答 Agent。用户自然语言提问，Agent 自主决策是否调用检索工具；检索召回不足时自动改写查询关键词重试；同时提供命令行、Web 两套交互界面  主要工作与亮点  Tool‑Calling Agent 编排 基于 LangChain1.x create_agent实现工具调用智能体，自定义知识库检索工具；借助 checkpointer 持久化维护多轮对话上下文，支持多轮追问；实现「模型自主判断检索时机‑检索不足自动改写关键词重试」完整
