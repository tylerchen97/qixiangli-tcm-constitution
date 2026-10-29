"""问卷规则引擎：九种体质量表计分与判定。

注意：题目为本项目自行表述的示例题库，各量表条目数与标准（ZYYXH/T157-2009）一致，
但正式使用前请对照标准原文核对/替换题目与反向计分条目。
答案：1=没有/根本不  2=很少  3=有时  4=经常  5=总是
每个条目为 (题目, 是否反向计分)。
"""
from __future__ import annotations
import math

NAMES = {
    "pinghe": "平和质", "qixu": "气虚质", "yangxu": "阳虚质", "yinxu": "阴虚质",
    "tanshi": "痰湿质", "shire": "湿热质", "xueyu": "血瘀质", "qiyu": "气郁质",
    "tebing": "特禀质",
}
BIASED = [k for k in NAMES if k != "pinghe"]
CHOICES = ["1 没有/根本不", "2 很少", "3 有时", "4 经常", "5 总是"]

QUESTIONS: dict[str, list[tuple[str, bool]]] = {
    "pinghe": [
        ("您精力充沛吗？", False), ("您容易疲乏吗？", True),
        ("您说话声音低弱无力吗？", True), ("您感到闷闷不乐吗？", True),
        ("您能适应外界自然和社会环境的变化吗？", False), ("您容易失眠吗？", True),
        ("您容易忘事（健忘）吗？", True), ("您对冷热变化的耐受力好吗？", False),
    ],
    "qixu": [
        ("您容易疲乏吗？", False), ("您容易气短、喘不上气吗？", False),
        ("您容易心慌吗？", False), ("您容易头晕或站起时晕眩吗？", False),
        ("您比别人容易感冒吗？", False), ("您喜欢安静、懒得说话吗？", False),
        ("您说话声音低弱无力吗？", False), ("您活动后容易出虚汗吗？", False),
    ],
    "yangxu": [
        ("您手脚发凉吗？", False), ("您胃脘、背部或腰膝部怕冷吗？", False),
        ("您比别人更怕冷（冬天穿得更多）吗？", False), ("您受不了冷空调或电扇的风吗？", False),
        ("您比别人容易因受寒而感冒吗？", False), ("您吃凉东西后会不舒服吗？", False),
        ("您受凉后容易腹泻吗？", False),
    ],
    "yinxu": [
        ("您感到手心脚心发热吗？", False), ("您感到身体或脸上发热吗？", False),
        ("您皮肤或口唇干燥吗？", False), ("您口唇颜色偏红吗？", False),
        ("您容易便秘或大便干燥吗？", False), ("您两颧部潮红或偏红吗？", False),
        ("您感到眼睛干涩吗？", False), ("您感到口干咽燥、总想喝水吗？", False),
    ],
    "tanshi": [
        ("您感到胸闷或腹部胀满吗？", False), ("您感到身体沉重不轻松吗？", False),
        ("您腹部肥满松软吗？", False), ("您额部油脂分泌多吗？", False),
        ("您上眼睑比别人肿吗？", False), ("您嘴里有黏黏的感觉吗？", False),
        ("您平时痰多，咽部总感觉有痰堵吗？", False), ("您舌苔厚腻吗（或自觉舌苔厚）？", False),
    ],
    "shire": [
        ("您面部或鼻部有油腻感或油光发亮吗？", False), ("您容易生痤疮或疮疖吗？", False),
        ("您感到口苦或嘴里有异味吗？", False), ("您大便黏滞不爽、有解不尽的感觉吗？", False),
        ("您小便时尿道有发热感、尿色浓吗？", False), ("您带下色黄（女）/阴囊潮湿（男）吗？", False),
    ],
    "xueyu": [
        ("您的皮肤在不知不觉中会出现青紫瘀斑吗？", False), ("您两颧部有细微红丝吗？", False),
        ("您身体上有哪里疼痛吗？", False), ("您面色晦暗或容易出现褐斑吗？", False),
        ("您容易有黑眼圈吗？", False), ("您容易忘事（健忘）吗？", False),
        ("您口唇颜色偏暗吗？", False),
    ],
    "qiyu": [
        ("您感到闷闷不乐吗？", False), ("您容易精神紧张、焦虑不安吗？", False),
        ("您多愁善感、感情脆弱吗？", False), ("您容易感到害怕或受到惊吓吗？", False),
        ("您胁肋部或乳房胀痛吗？", False), ("您无缘无故叹气吗？", False),
        ("您咽喉部有异物感且吐之不出吗？", False),
    ],
    "tebing": [
        ("您没有感冒时也会打喷嚏吗？", False), ("您没有感冒时也会鼻塞、流鼻涕吗？", False),
        ("您因季节、温度变化或异味而咳喘吗？", False), ("您容易过敏（药物、食物、气味、花粉等）吗？", False),
        ("您的皮肤容易起荨麻疹吗？", False), ("您的皮肤因过敏出现过紫红色瘀点、瘀斑吗？", False),
        ("您的皮肤一抓就红，并出现抓痕吗？", False),
    ],
}


def transformed_scores(answers: dict, allow_partial: bool = False,
                       min_coverage: float = 0.6) -> dict[str, float | None]:
    """answers[scale] 为各题 1~5 的列表。allow_partial=True 时允许 None（未回答）；
    某量表有效作答比例 < min_coverage 则该体质得分为 None（信息不足）。"""
    out: dict[str, float | None] = {}
    for key, items in QUESTIONS.items():
        vals = answers.get(key)
        if vals is None or len(vals) != len(items):
            raise ValueError(f"{NAMES[key]} 需要 {len(items)} 个答案")
        got = []
        for v, (_, rev) in zip(vals, items):
            if v is None:
                if not allow_partial:
                    raise ValueError(f"{NAMES[key]} 存在未回答题目")
                continue
            if not 1 <= v <= 5:
                raise ValueError("答案必须在 1~5 之间")
            got.append((6 - v) if rev else v)
        if len(got) < max(1, math.ceil(len(items) * min_coverage)):
            out[key] = None
            continue
        n = len(got)
        out[key] = round((sum(got) - n) / (n * 4) * 100, 1)
    return out


def judge(scores: dict) -> dict:
    """每种体质“是/倾向是/否”（平和质为“是/基本是/否”），含信息不足的处理；给出主/兼夹体质。"""
    complete = all(v is not None for v in scores.values())
    p = scores["pinghe"]
    others = [scores[k] for k in BIASED if scores[k] is not None]
    verdict: dict[str, str] = {}
    if complete and p >= 60 and all(s < 30 for s in others):
        verdict["pinghe"] = "是"
    elif complete and p >= 60 and all(s < 40 for s in others):
        verdict["pinghe"] = "基本是"
    elif p is not None and p < 60:
        verdict["pinghe"] = "否"
    elif complete:
        verdict["pinghe"] = "否"
    else:
        verdict["pinghe"] = "信息不足"
    for k in BIASED:
        s = scores[k]
        verdict[k] = "未评估" if s is None else "是" if s >= 40 else "倾向是" if s >= 30 else "否"

    if verdict["pinghe"] in ("是", "基本是"):
        primary, secondary = "pinghe", []
    else:
        hit = sorted((k for k in BIASED if verdict[k] in ("是", "倾向是")), key=lambda k: -scores[k])
        if hit:
            primary, secondary = hit[0], hit[1:]
        else:
            primary, secondary = ("pinghe" if verdict["pinghe"] != "信息不足" else None), []
    return {"verdict": verdict, "primary": primary, "secondary": secondary}
