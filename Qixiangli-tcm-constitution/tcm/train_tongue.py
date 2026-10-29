"""训练舌象多任务 CNN。

  python -m tcm.train_tongue --data data/tongue/labels.csv --epochs 30 --pretrained --out models/tongue.pt
  python -m tcm.train_tongue --synthetic            # 用合成数据做流程冒烟测试（无临床意义）

要点：按患者划分验证集；类别加权；仅轻微亮度增强（不做色相抖动，以免破坏舌色标签）；
以验证集各头 macro-F1 的均值选最佳模型。
"""
from __future__ import annotations
import argparse, json, random, tempfile
from pathlib import Path
from .tongue_data import HEADS, IGNORE, read_labels, split_rows, macro_f1, make_synthetic_dataset


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="labels.csv 路径")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--pretrained", action="store_true", help="使用 ImageNet 预训练权重（需联网下载）")
    ap.add_argument("--out", default="models/tongue.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import numpy as np, torch, torch.nn as nn
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms as T
    from .tongue_cnn import build_model, MEAN, STD

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if a.synthetic:
        a.data = str(make_synthetic_dataset(tempfile.mkdtemp()))
    if not a.data:
        ap.error("请提供 --data 或 --synthetic")
    rows = read_labels(a.data)
    train_rows, val_rows = split_rows(rows, seed=a.seed)
    print(f"样本：train={len(train_rows)} val={len(val_rows)}")

    s = a.img_size
    tf_train = T.Compose([T.RandomResizedCrop(s, scale=(0.75, 1.0)), T.RandomHorizontalFlip(),
                          T.ColorJitter(brightness=0.1), T.ToTensor(), T.Normalize(MEAN, STD)])
    tf_val = T.Compose([T.Resize((s, s)), T.ToTensor(), T.Normalize(MEAN, STD)])
    names = list(HEADS)

    class DS(Dataset):
        def __init__(self, rs, tf): self.rs, self.tf = rs, tf
        def __len__(self): return len(self.rs)
        def __getitem__(self, i):
            r = self.rs[i]
            x = self.tf(Image.open(r["path"]).convert("RGB"))
            return x, torch.tensor([r["y"][h] for h in names])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_model(a.pretrained).to(dev)

    losses = {}
    for j, h in enumerate(names):  # 逐头类别加权
        cnt = [0] * len(HEADS[h])
        for r in train_rows:
            if r["y"][h] != IGNORE:
                cnt[r["y"][h]] += 1
        tot = sum(cnt) or 1
        w = torch.tensor([tot / (len(cnt) * c) if c else 0.0 for c in cnt], dtype=torch.float)
        losses[h] = nn.CrossEntropyLoss(weight=w.to(dev), ignore_index=IGNORE)

    tl = DataLoader(DS(train_rows, tf_train), batch_size=a.batch, shuffle=True, num_workers=0)
    vl = DataLoader(DS(val_rows, tf_val), batch_size=a.batch, num_workers=0)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    def evaluate():
        net.eval()
        yt = {h: [] for h in names}; yp = {h: [] for h in names}
        with torch.no_grad():
            for x, y in vl:
                out = net(x.to(dev))
                for j, h in enumerate(names):
                    pred = out[h].argmax(-1).cpu().tolist()
                    for t, p in zip(y[:, j].tolist(), pred):
                        if t != IGNORE:
                            yt[h].append(t); yp[h].append(p)
        m = {}
        for h in names:
            n = len(yt[h])
            m[h] = {"n": n,
                    "acc": round(sum(t == p for t, p in zip(yt[h], yp[h])) / n, 3) if n else None,
                    "macro_f1": round(macro_f1(yt[h], yp[h], len(HEADS[h])), 3) if n else None}
        vals = [v["macro_f1"] for v in m.values() if v["macro_f1"] is not None]
        return m, (sum(vals) / len(vals) if vals else 0.0)

    best, best_m = -1.0, {}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, a.epochs + 1):
        net.train(); tot_loss = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            out = net(x)
            loss = sum(losses[h](out[h], y[:, j]) for j, h in enumerate(names) if (y[:, j] != IGNORE).any())
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += float(loss) * len(x)
        sched.step()
        m, score = evaluate()
        print(f"epoch {ep:3d} loss={tot_loss / max(1, len(train_rows)):.4f} val_meanF1={score:.3f}")
        if score > best:
            best, best_m = score, m
            torch.save({"state_dict": net.state_dict(), "heads": HEADS, "img_size": s}, a.out)
    Path(a.out).with_suffix(".metrics.json").write_text(
        json.dumps({"best_mean_macro_f1": best, "per_head": best_m}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("最佳模型已保存：", a.out)
    print(json.dumps(best_m, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
