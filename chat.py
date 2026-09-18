"""命令行对话入口。

用法：
    uv run python chat.py
"""

from langgraph.checkpoint.memory import InMemorySaver

from prompts import WELCOME
from rag import build_agent, extract_text

# 多轮对话的会话标识：同一个 thread_id 才能记住上下文
THREAD_ID = "cli-session"
EXIT_WORDS = {"/exit", "/quit", "exit", "quit"}


def main() -> None:
    print(WELCOME)
    print("正在启动 Agent ...")

    try:
        agent = build_agent(checkpointer=InMemorySaver())
    except Exception as exc:
        print(f"[启动失败] {type(exc).__name__}: {exc}")
        return

    print("就绪，开始提问吧。\n")

    while True:
        try:
            question = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return

        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            print("再见。")
            return

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"configurable": {"thread_id": THREAD_ID}},
            )
        except Exception as exc:
            print(f"\n[出错] {type(exc).__name__}: {exc}\n")
            continue

        print(f"\n助手 > {extract_text(result['messages'][-1].content)}\n")


if __name__ == "__main__":
    main()
