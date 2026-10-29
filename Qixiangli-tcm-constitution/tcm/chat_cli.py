"""终端对话问诊：python -m tcm.chat_cli  （设置 DASHSCOPE_API_KEY 则用 Qwen，否则逐题问答）"""
from .interview import Interview, make_llm
from . import pipeline
from .report import render_html, llm_narrative


def main():
    llm = make_llm()
    print("[模式]", "Qwen 对话问诊" if llm else "离线逐题问答（未设置 DASHSCOPE_API_KEY）")
    it = Interview(llm)
    print("助手：", it.start())
    while not it.done:
        try:
            msg = input("您：")
        except (EOFError, KeyboardInterrupt):
            break
        print("助手：", it.reply(msg))
    if it.stopped:
        return
    res = pipeline.run(it.to_answers())
    import os
    os.makedirs("reports", exist_ok=True)
    with open("reports/chat_report.html", "w", encoding="utf-8") as f:
        f.write(render_html(res, llm_narrative(res)))
    print("主体质：", res["primary_name"], "→ reports/chat_report.html")


if __name__ == "__main__":
    main()
