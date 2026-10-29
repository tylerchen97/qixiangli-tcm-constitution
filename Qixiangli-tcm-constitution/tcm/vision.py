"""舌象/面色分析：基于颜色统计的可解释基线（未经临床标定，仅作演示）。

要换成深度学习模型：实现 TongueModel 协议（predict(img)->dict）并传入 analyze_tongue(model=...)。
"""
from __future__ import annotations
from typing import Protocol
import numpy as np
from PIL import Image


class TongueModel(Protocol):
    def predict(self, img: Image.Image) -> dict: ...


def _center(img: Image.Image, frac: float = 0.5) -> Image.Image:
    w, h = img.size
    dw, dh = int(w * (1 - frac) / 2), int(h * (1 - frac) / 2)
    return img.crop((dw, dh, w - dw, h - dh))


def _hsv(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB").convert("HSV"), dtype=float) / 255.0


def analyze_tongue(img: Image.Image, model: TongueModel | None = None) -> dict:
    if model is not None:
        return model.predict(img)
    reg = _center(img, 0.5)
    rgb = np.asarray(reg.convert("RGB"), dtype=float)
    hsv = _hsv(reg)
    r, g, b = rgb[..., 0].mean(), rgb[..., 1].mean(), rgb[..., 2].mean()
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = float(((s < 0.25) & (v > 0.6)).mean())
    yellow = float(((h > 0.09) & (h < 0.19) & (s > 0.3) & (v > 0.5)).mean())
    redness = r / (r + g + b + 1e-9)
    sat = float(s.mean())

    if b > r * 0.9 and r < 150:
        color = "紫暗"
    elif redness > 0.5 and sat > 0.45:
        color = "红绛"
    elif sat < 0.22:
        color = "淡白"
    else:
        color = "淡红"
    coat = "黄" if yellow > 0.25 else "白" if white > 0.25 else "薄白"
    thick = "厚" if max(white, yellow) > 0.45 else "薄"
    return {"舌色": color, "苔色": coat, "苔质": thick,
            "metrics": {"white": round(white, 3), "yellow": round(yellow, 3),
                        "redness": round(float(redness), 3), "sat": round(sat, 3)}}


def analyze_face(img: Image.Image) -> dict:
    reg = _center(img, 0.4)
    hsv = _hsv(reg)
    h, s, v = hsv[..., 0].mean(), hsv[..., 1].mean(), hsv[..., 2].mean()
    if v < 0.35:
        color = "晦暗"
    elif s < 0.18 and v > 0.7:
        color = "淡白"
    elif s > 0.45 and (h < 0.04 or h > 0.95):
        color = "潮红"
    elif 0.08 < h < 0.14 and s > 0.35:
        color = "萎黄"
    else:
        color = "红黄隐隐"
    gloss = "有光泽" if hsv[..., 2].std() > 0.08 else "少光泽"
    return {"面色": color, "光泽": gloss,
            "metrics": {"h": round(float(h), 3), "s": round(float(s), 3), "v": round(float(v), 3)}}


def vision_evidence(tongue: dict | None, face: dict | None) -> dict[str, float]:
    """将舌/面特征映射为各体质的加分证据（规则可由医师审校调整）。"""
    ev: dict[str, float] = {}

    def add(k, x):
        ev[k] = ev.get(k, 0) + x

    if tongue and "error" not in tongue and tongue.get("metrics", {}).get("conf_min", 1.0) >= 0.5:
        c, coat, th = tongue.get("舌色"), tongue.get("苔色"), tongue.get("苔质")
        if c == "淡白":
            add("yangxu", 4); add("qixu", 3)
        if c == "红":
            add("yinxu", 2); add("shire", 1)
        if c == "红绛":
            add("yinxu", 4); add("shire", 2)
        if c == "紫暗":
            add("xueyu", 5)
        if coat == "少苔":
            add("yinxu", 4)
        if th in ("厚", "腻"):
            if coat == "黄":
                add("shire", 5)
            elif coat == "白":
                add("tanshi", 5)
        if tongue.get("齿痕") == "有":
            add("tanshi", 3); add("qixu", 2)
        if tongue.get("裂纹") == "有":
            add("yinxu", 3)
    if face:
        f = face["面色"]
        if f == "淡白":
            add("qixu", 3); add("yangxu", 3)
        if f == "萎黄":
            add("qixu", 3); add("tanshi", 2)
        if f == "潮红":
            add("yinxu", 3); add("shire", 3)
        if f == "晦暗":
            add("xueyu", 3); add("yangxu", 2)
    return ev
