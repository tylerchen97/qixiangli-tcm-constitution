"""Gradio 演示：python app.py → http://127.0.0.1:7860
 - 对话问诊（设置 DASHSCOPE_API_KEY 用 Qwen，否则离线逐题问答）
 - 问卷问诊（一次性填写）
 - 舌/面/脉（可选）；若存在 models/tongue.pt（或环境变量 TONGUE_MODEL）则用 CNN 舌诊，否则颜色基线
"""
import gradio as gr
from tcm.questionnaire import QUESTIONS, NAMES, CHOICES
from tcm.pulse import load_pulse_csv
from tcm.interview import Interview, make_llm
from tcm.tongue_cnn import load_tongue_model
from tcm import pipeline
from tcm.report import render_html, llm_narrative

KEYS = list(QUESTIONS)
TONGUE_MODEL = load_tongue_model()


def start_chat():
    it = Interview(make_llm())
    return [{"role": "assistant", "content": it.start()}], it


def chat(msg, history, it):
    if not msg or not msg.strip():
        return "", history, it
    if it is None:
        history, it = start_chat()
    history = history + [{"role": "user", "content": msg}, {"role": "assistant", "content": it.reply(msg)}]
    return "", history, it


def progress_text(it):
    if it is None:
        return ""
    a, n = it.progress()
    return f"问诊进度：{a}/{n}" + ("（已中止：建议就医）" if it.stopped else "（已完成）" if it.done else "")


def generate(tongue_img, face_img, pulse_file, it, *vals):
    if it is not None and it.stopped:
        return "<p style='color:#a33'>问诊因红旗症状/危机表述已中止，请先寻求医疗帮助，不生成体质报告。</p>"
    if it is not None and it.n_answered() > 0:
        answers = it.to_answers()
    else:
        answers, i = {}, 0
        for k in KEYS:
            n = len(QUESTIONS[k])
            chunk = vals[i:i + n]; i += n
            if any(c is None for c in chunk):
                return "<p style='color:#a33'>请先完成对话问诊，或填完“问卷问诊”的全部题目。</p>"
            answers[k] = [int(c[0]) for c in chunk]
    pulse = load_pulse_csv(pulse_file) if pulse_file else None
    res = pipeline.run(answers, tongue_img, face_img, pulse, tongue_model=TONGUE_MODEL)
    return render_html(res, llm_narrative(res))


with gr.Blocks(title="AI 中医体质辨识") as demo:
    gr.Markdown("# AI 中医体质辨识（演示）\n仅供健康管理参考，非医疗诊断。")
    it_state = gr.State(None)
    with gr.Tab("对话问诊"):
        bot = gr.Chatbot(height=420, label="问诊助手")
        box = gr.Textbox(placeholder="输入回答后回车（说“跳过”/“结束”也可以）", show_label=False)
        prog = gr.Markdown()
        box.submit(chat, [box, bot, it_state], [box, bot, it_state]).then(progress_text, it_state, prog)
    with gr.Tab("问卷问诊"):
        radios = []
        for k in KEYS:
            gr.Markdown(f"### {NAMES[k]}相关")
            for q, _ in QUESTIONS[k]:
                radios.append(gr.Radio(CHOICES, label=q))
    with gr.Tab("舌 / 面 / 脉（可选）"):
        tongue = gr.Image(type="pil", label="舌象照片（自然光、伸舌放松）")
        face = gr.Image(type="pil", label="面部照片（自然光、素颜）")
        pulse = gr.File(label="脉象 CSV（列：light,mid,heavy，100Hz）", file_types=[".csv"])
    btn = gr.Button("生成体质报告", variant="primary")
    out = gr.HTML()
    btn.click(generate, [tongue, face, pulse, it_state, *radios], out)
    demo.load(start_chat, None, [bot, it_state])

if __name__ == "__main__":
    demo.launch()
