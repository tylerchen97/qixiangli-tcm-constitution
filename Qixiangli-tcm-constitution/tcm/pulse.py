"""脉象信号分析（演示级）：输入浮/中/沉三个压力层的脉搏波，输出浮沉迟数、节律、强弱。

真实设备请整理成 {"light": arr, "mid": arr, "heavy": arr}，采样率 fs（Hz）。
CSV 格式：表头 light,mid,heavy，每行一个采样点。
"""
from __future__ import annotations
import csv
import numpy as np
from scipy.signal import find_peaks


def load_pulse_csv(path: str) -> dict[str, np.ndarray]:
    cols: dict[str, list[float]] = {"light": [], "mid": [], "heavy": []}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k in cols:
                cols[k].append(float(row[k]))
    return {k: np.array(v) for k, v in cols.items()}


def _peaks(x: np.ndarray, fs: float):
    x = x - np.mean(x)
    thr = 0.3 * (x.max() - x.min())
    idx, _ = find_peaks(x, distance=int(0.3 * fs), prominence=thr)
    return idx, x


def analyze_pulse(signals: dict[str, np.ndarray], fs: float = 100.0, ref_amp: float = 1.0) -> dict:
    amps, rates, cvs = {}, {}, {}
    for lv, sig in signals.items():
        idx, x = _peaks(np.asarray(sig, float), fs)
        if len(idx) < 3:
            amps[lv] = 0.0
            continue
        iv = np.diff(idx) / fs
        rates[lv] = 60.0 / iv.mean()
        cvs[lv] = float(iv.std() / iv.mean())
        amps[lv] = float(np.mean(x[idx]))
    if "mid" not in rates:
        return {"error": "信号质量不足，未检测到稳定脉搏波"}
    bpm, cv = rates["mid"], cvs["mid"]
    tags = ["迟" if bpm < 60 else "数" if bpm > 90 else "平（至数正常）"]
    l, m, h = amps.get("light", 0), amps.get("mid", 0), amps.get("heavy", 0)
    if l >= max(m, h) * 1.1:
        tags.append("浮")
    elif h >= max(l, m) * 1.1:
        tags.append("沉")
    if cv > 0.15:
        tags.append("节律不齐（建议医师复核）")
    if m < 0.5 * ref_amp:
        tags.append("弱")
    return {"心率(次/分)": round(bpm, 1), "间期变异系数": round(cv, 3),
            "三层幅值(浮/中/沉)": [round(l, 3), round(m, 3), round(h, 3)], "脉象标签": tags}


def pulse_evidence(p: dict | None) -> dict[str, float]:
    ev: dict[str, float] = {}
    if not p or "error" in p:
        return ev
    tags = " ".join(p["脉象标签"])
    if "迟" in tags or "沉" in tags:
        ev["yangxu"] = ev.get("yangxu", 0) + 3
    if "数" in tags:
        ev["yinxu"] = ev.get("yinxu", 0) + 3
        ev["shire"] = ev.get("shire", 0) + 2
    if "弱" in tags:
        ev["qixu"] = ev.get("qixu", 0) + 4
    return ev


def synth_pulse(bpm=72, amps=(0.6, 1.0, 0.7), fs=100.0, seconds=20, jitter=0.0, seed=0):
    """合成演示用脉搏波（测试与 demo 使用）。"""
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    w = np.arange(n)
    out = {}
    for lv, a in zip(("light", "mid", "heavy"), amps):
        sig = np.zeros(n)
        t, period = 0.0, 60.0 / bpm
        while t < seconds:
            sig += a * np.exp(-((w - int(t * fs)) / (0.06 * fs)) ** 2)
            t += period * (1 + rng.normal(0, jitter))
        out[lv] = sig + rng.normal(0, 0.01, n)
    return out
