"""命令行演示：python -m tcm.cli  （用合成数据生成示例报告到 reports/demo.html）"""
import os, random
from PIL import Image
from .questionnaire import QUESTIONS
from .pulse import synth_pulse
from . import pipeline
from .report import render_html, llm_narrative


def main():
    random.seed(1)
    answers = {k: [random.choice([1, 2, 3]) for _ in v] for k, v in QUESTIONS.items()}
    answers["yangxu"] = [4, 5, 4, 4, 3, 4, 4]
    tongue = Image.new("RGB", (200, 200), (225, 190, 190))
    face = Image.new("RGB", (200, 200), (220, 190, 170))
    res = pipeline.run(answers, tongue, face, synth_pulse(bpm=55, amps=(0.5, 0.8, 1.0)))
    os.makedirs("reports", exist_ok=True)
    with open("reports/demo.html", "w", encoding="utf-8") as f:
        f.write(render_html(res, llm_narrative(res)))
    print("主体质:", res["primary_name"], "-> reports/demo.html")


if __name__ == "__main__":
    main()
