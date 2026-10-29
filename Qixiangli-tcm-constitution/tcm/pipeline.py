"""多源融合：问卷（主）+ 舌/面/脉（辅助证据，每种体质最多 +8 分）。"""
from __future__ import annotations
import datetime as dt
from .questionnaire import NAMES, transformed_scores, judge
from .vision import analyze_tongue, analyze_face, vision_evidence
from .pulse import analyze_pulse, pulse_evidence
from .knowledge import KB, INSUFFICIENT, current_solar_term

EVIDENCE_CAP = 8.0


def fuse(q_scores: dict[str, float], *evidences: dict[str, float]) -> dict[str, float]:
    out = dict(q_scores)
    for k in out:
        if k == "pinghe" or out[k] is None:
            continue
        bonus = min(sum(e.get(k, 0) for e in evidences), EVIDENCE_CAP)
        out[k] = round(min(out[k] + bonus, 100), 1)
    return out


def run(answers, tongue_img=None, face_img=None, pulse_signals=None, fs=100.0, today=None,
        tongue_model=None) -> dict:
    q = transformed_scores(answers, allow_partial=True)
    tongue = analyze_tongue(tongue_img, model=tongue_model) if tongue_img is not None else None
    face = analyze_face(face_img) if face_img is not None else None
    pulse = analyze_pulse(pulse_signals, fs) if pulse_signals else None
    fused = fuse(q, vision_evidence(tongue, face), pulse_evidence(pulse))
    res = judge(fused)
    primary = res["primary"]
    term, tip = current_solar_term(today)
    return {
        "date": (today or dt.date.today()).isoformat(),
        "questionnaire_scores": q, "fused_scores": fused, **res,
        "primary_name": NAMES[primary] if primary else "信息不足，暂无法判定",
        "secondary_names": [NAMES[k] for k in res["secondary"]],
        "tongue": tongue, "face": face, "pulse": pulse,
        "advice": KB[primary] if primary else INSUFFICIENT, "solar_term": {"name": term, "tip": tip},
        "sources_used": [n for n, x in (("问卷", True), ("舌诊", tongue), ("面诊", face), ("脉诊", pulse)) if x],
    }
