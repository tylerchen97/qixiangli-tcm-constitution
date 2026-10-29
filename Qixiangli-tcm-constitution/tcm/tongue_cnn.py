"""舌象多任务 CNN：ResNet18 主干 + 每个标签一个分类头。需要 torch / torchvision。

推理：  model = load_tongue_model("models/tongue.pt")   # 缺 torch 或缺文件时返回 None
        analyze_tongue(img, model=model)
"""
from __future__ import annotations
import os
from .tongue_data import HEADS

IMG_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_model(pretrained: bool = False):
    import torch.nn as nn
    from torchvision import models

    class TongueNet(nn.Module):
        def __init__(self):
            super().__init__()
            w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            b = models.resnet18(weights=w)
            dim = b.fc.in_features
            b.fc = nn.Identity()
            self.backbone = b
            self.heads = nn.ModuleDict({h: nn.Linear(dim, len(c)) for h, c in HEADS.items()})

        def forward(self, x):
            f = self.backbone(x)
            return {h: layer(f) for h, layer in self.heads.items()}

    return TongueNet()


class CNNTongueModel:
    """实现 vision.TongueModel 协议。"""

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        import torch
        from torchvision import transforms as T
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        self.heads = ck["heads"]
        self.net = build_model(False)
        self.net.load_state_dict(ck["state_dict"])
        self.net.to(device).eval()
        self.device = device
        s = ck.get("img_size", IMG_SIZE)
        self.tf = T.Compose([T.Resize((s, s)), T.ToTensor(), T.Normalize(MEAN, STD)])
        self._torch = torch

    def predict(self, img) -> dict:
        t = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            out = self.net(t)
        res, conf = {}, {}
        for h, logits in out.items():
            p = logits.softmax(-1)[0]
            i = int(p.argmax())
            res[h] = self.heads[h][i]
            conf[h] = round(float(p[i]), 3)
        res["metrics"] = {"conf": conf, "conf_min": min(conf.values())}
        return res


def load_tongue_model(path: str | None = None):
    path = path or os.getenv("TONGUE_MODEL", "models/tongue.pt")
    if not os.path.exists(path):
        return None
    try:
        return CNNTongueModel(path)
    except Exception as e:  # 缺 torch / 权重损坏
        print(f"[tongue] 模型加载失败，回退到颜色基线：{e}")
        return None
