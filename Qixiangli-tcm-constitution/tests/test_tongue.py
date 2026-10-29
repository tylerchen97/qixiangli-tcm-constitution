import tempfile, pytest
from tcm.tongue_data import HEADS, IGNORE, read_labels, split_rows, macro_f1, make_synthetic_dataset
from tcm.vision import vision_evidence


def test_labels_and_split_by_patient():
    p = make_synthetic_dataset(tempfile.mkdtemp(), 6)
    rows = read_labels(p)
    tr, va = split_rows(rows, 0.2)
    assert len(rows) == 30 and tr and va
    assert not ({r["patient"] for r in tr} & {r["patient"] for r in va})


def test_bad_label_rejected(tmp_path):
    (tmp_path / "labels.csv").write_text("path,舌色\na.png,蓝色\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_labels(tmp_path / "labels.csv")


def test_blank_label_ignored(tmp_path):
    (tmp_path / "labels.csv").write_text("path,舌色,苔色\na.png,淡白,\n", encoding="utf-8")
    r = read_labels(tmp_path / "labels.csv")[0]
    assert r["y"]["舌色"] == 0 and r["y"]["苔色"] == IGNORE


def test_macro_f1():
    assert macro_f1([0, 1, 2], [0, 1, 2], 3) == 1.0


def test_evidence_with_cnn_labels_and_confidence_gate():
    t = {"舌色": "淡白", "苔色": "白", "苔质": "腻", "齿痕": "有", "裂纹": "无", "metrics": {"conf_min": 0.9}}
    ev = vision_evidence(t, None)
    assert ev["tanshi"] == 8 and ev["yangxu"] == 4
    t["metrics"]["conf_min"] = 0.3
    assert vision_evidence(t, None) == {}


def test_cnn_roundtrip_if_torch_available(tmp_path):
    pytest.importorskip("torch"); pytest.importorskip("torchvision")
    from tcm.train_tongue import main
    from tcm.tongue_cnn import CNNTongueModel
    from PIL import Image
    out = tmp_path / "t.pt"
    main(["--synthetic", "--epochs", "2", "--batch", "8", "--img-size", "64", "--out", str(out)])
    r = CNNTongueModel(str(out)).predict(Image.new("RGB", (80, 80), (225, 150, 150)))
    assert r["舌色"] in HEADS["舌色"] and "conf_min" in r["metrics"]
