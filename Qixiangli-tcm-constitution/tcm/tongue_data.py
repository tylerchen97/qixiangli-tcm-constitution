"""舌象数据：标签体系、labels.csv 读取、划分、指标、合成冒烟数据（均不依赖 torch）。

labels.csv 列：path, 舌色, 苔色, 苔质, 齿痕, 裂纹, [patient], [split]
 - path 相对于 labels.csv 所在目录；某个标签留空表示该图未标注此项（训练时忽略）。
 - 有 split 列（train/val）则按其划分；否则有 patient 列则按患者划分（避免同一人泄漏到验证集）；否则随机。
标签应由执业中医师标注，建议双人标注并报告一致性（Cohen's κ）。
"""
from __future__ import annotations
import csv, random
from pathlib import Path

HEADS: dict[str, list[str]] = {
    "舌色": ["淡白", "淡红", "红", "红绛", "紫暗"],
    "苔色": ["白", "黄", "少苔"],
    "苔质": ["薄", "厚", "腻"],
    "齿痕": ["无", "有"],
    "裂纹": ["无", "有"],
}
IGNORE = -100


def read_labels(csv_path: str | Path) -> list[dict]:
    csv_path = Path(csv_path)
    root = csv_path.parent
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for ln, r in enumerate(csv.DictReader(f), start=2):
            y = {}
            for h, classes in HEADS.items():
                val = (r.get(h) or "").strip()
                if not val:
                    y[h] = IGNORE
                elif val in classes:
                    y[h] = classes.index(val)
                else:
                    raise ValueError(f"{csv_path}:{ln} 列“{h}”的值“{val}”不在 {classes} 中")
            rows.append({"path": root / r["path"], "y": y,
                         "patient": (r.get("patient") or "").strip(),
                         "split": (r.get("split") or "").strip()})
    return rows


def split_rows(rows: list[dict], val_ratio: float = 0.2, seed: int = 0):
    if any(r["split"] in ("train", "val") for r in rows):
        return [r for r in rows if r["split"] != "val"], [r for r in rows if r["split"] == "val"]
    rng = random.Random(seed)
    if all(r["patient"] for r in rows):
        pats = sorted({r["patient"] for r in rows})
        rng.shuffle(pats)
        val_p = set(pats[:max(1, int(len(pats) * val_ratio))])
        return ([r for r in rows if r["patient"] not in val_p],
                [r for r in rows if r["patient"] in val_p])
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    k = max(1, int(len(rows) * val_ratio))
    val = set(idx[:k])
    return [r for i, r in enumerate(rows) if i not in val], [r for i, r in enumerate(rows) if i in val]


def macro_f1(y_true: list[int], y_pred: list[int], n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        if tp + fn == 0:
            continue
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn)
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def make_synthetic_dataset(root: str | Path, n_per_class: int = 12, size: int = 64, seed: int = 0) -> Path:
    """生成“按舌色纯色+噪声”的合成图，仅用于验证训练流程能跑通，没有任何临床意义。"""
    import numpy as np
    from PIL import Image
    root = Path(root); (root / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed); nrng = np.random.default_rng(seed)
    base = {"淡白": (232, 205, 200), "淡红": (225, 150, 150), "红": (215, 90, 95),
            "红绛": (180, 50, 70), "紫暗": (120, 70, 110)}
    lines = ["path,舌色,苔色,苔质,齿痕,裂纹,patient"]
    n = 0
    for color, rgb in base.items():
        for _ in range(n_per_class):
            arr = np.clip(np.array(rgb) + nrng.normal(0, 12, (size, size, 3)), 0, 255).astype("uint8")
            name = f"images/{n:04d}.png"
            Image.fromarray(arr).save(root / name)
            lines.append(f"{name},{color},{rng.choice(HEADS['苔色'])},{rng.choice(HEADS['苔质'])},"
                         f"{rng.choice(HEADS['齿痕'])},{rng.choice(HEADS['裂纹'])},p{n // 2}")
            n += 1
    p = root / "labels.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
