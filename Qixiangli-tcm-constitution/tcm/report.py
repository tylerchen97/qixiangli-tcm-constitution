"""报告生成：HTML（含 SVG 雷达图）+ 可选 Qwen 大模型解读（DashScope 兼容接口）。"""
from __future__ import annotations
import html, math, os
from .questionnaire import NAMES

DISCLAIMER = "本报告由算法生成，仅供健康管理与体质自测参考，不构成医疗诊断或治疗建议，请由执业中医师结合四诊进一步解读。"


def radar_svg(scores: dict[str, float], size: int = 360) -> str:
    keys = list(NAMES)
    cx = cy = size / 2
    R = size / 2 - 55
    pt = lambda i, r: (cx + r * math.sin(2 * math.pi * i / 9), cy - r * math.cos(2 * math.pi * i / 9))
    parts = []
    for lv in (25, 50, 75, 100):
        ring = " ".join("%.1f,%.1f" % pt(i, R * lv / 100) for i in range(9))
        parts.append(f'<polygon points="{ring}" fill="none" stroke="#ccc"/>')
    poly = " ".join("%.1f,%.1f" % pt(i, R * (scores[k] or 0) / 100) for i, k in enumerate(keys))
    parts.append(f'<polygon points="{poly}" fill="rgba(46,125,90,.3)" stroke="#2e7d5a" stroke-width="2"/>')
    for i, k in enumerate(keys):
        x, y = pt(i, R + 28)
        parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="12" text-anchor="middle" fill="currentColor">{NAMES[k]} {'—' if scores[k] is None else format(scores[k], '.0f')}</text>')
    return f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'


def llm_narrative(result: dict) -> str | None:
    """设置环境变量 DASHSCOPE_API_KEY 后调用通义千问；失败返回 None（使用模板文案）。"""
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        scores = {NAMES[k]: v for k, v in result["fused_scores"].items() if v is not None}
        prompt = (
            "你是中医体质健康管理助手。根据下列结构化结果，用通俗中文写一段 200 字以内的体质解读与调养提示。"
            "严禁做疾病诊断、严禁给出具体药物剂量，结尾提醒请咨询执业中医师。\n"
            f"主体质：{result['primary_name']}；兼夹：{result['secondary_names']}；各体质得分：{scores}；"
            f"舌象：{result['tongue']}；面象：{result['face']}；脉象：{result['pulse']}；节气：{result['solar_term']['name']}")
        r = client.chat.completions.create(model=os.getenv("QWEN_MODEL", "qwen-plus"),
                                           messages=[{"role": "user", "content": prompt}], timeout=30)
        return r.choices[0].message.content
    except Exception:
        return None


def _kv(d: dict | None) -> str:
    if not d:
        return "未采集"
    if "error" in d:
        return html.escape(d["error"])
    return "；".join(f"{html.escape(str(k))}：{html.escape(str(v))}" for k, v in d.items() if k != "metrics")


def render_html(result: dict, narrative: str | None = None) -> str:
    adv, esc, v = result["advice"], html.escape, result["verdict"]
    rows = "".join(
        f"<tr><td>{NAMES[k]}</td><td>{result['questionnaire_scores'][k] if result['questionnaire_scores'][k] is not None else '—'}</td><td>{result['fused_scores'][k] if result['fused_scores'][k] is not None else '—'}</td><td>{v[k]}</td></tr>"
        for k in NAMES)
    li = lambda xs: "".join(f"<li>{esc(x)}</li>" for x in xs)
    sec = "、".join(result["secondary_names"]) or "无"
    text = esc(narrative) if narrative else esc(adv["desc"])
    return f"""<div style="font-family:system-ui,'PingFang SC','Microsoft YaHei',sans-serif;max-width:860px;line-height:1.7">
<h2>体质分析报告（{result['date']}）</h2>
<p>使用数据：{'、'.join(result['sources_used'])}</p>
<h3>一、体质分析</h3>
<p><b>主要体质：{result['primary_name']}</b>　兼夹体质：{sec}</p><p>{text}</p>
{radar_svg(result['fused_scores'])}
<table border="1" cellpadding="6" style="border-collapse:collapse"><tr><th>体质</th><th>问卷转化分</th><th>融合分</th><th>判定</th></tr>{rows}</table>
<h3>二、易感倾向提示（非诊断）</h3><ul>{li(adv['tendency'])}</ul>
<h3>三、经络关注</h3><p>{esc('、'.join(adv['meridian']))}；常用穴位参考：{esc('、'.join(adv['acupoint']))}</p>
<h3>四、面诊 / 舌诊 / 脉诊</h3>
<p>面诊：{_kv(result['face'])}</p><p>舌诊：{_kv(result['tongue'])}</p><p>脉诊：{_kv(result['pulse'])}</p>
<h3>五、{result['solar_term']['name']}节气调养</h3><p>{esc(result['solar_term']['tip'])}</p>
<h3>六、饮食 / 起居 / 运动</h3>
<p><b>饮食：</b>{esc(adv['diet'])}</p><p><b>起居：</b>{esc(adv['life'])}</p>
<p><b>运动：</b>{esc(adv['sport'])}</p><p><b>宜避免：</b>{esc(adv['avoid'])}</p>
<h3>七、膏方方向（仅供医师参考）</h3><p>{esc(adv['gao'])}</p>
<hr><p style="color:#a33"><b>免责声明：</b>{DISCLAIMER}</p></div>"""
