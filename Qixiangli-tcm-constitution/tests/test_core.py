import pytest
from PIL import Image
from tcm.questionnaire import QUESTIONS, transformed_scores, judge
from tcm.pulse import synth_pulse, analyze_pulse
from tcm.vision import analyze_tongue
from tcm import pipeline


def all_answers(v, **over):
    a = {k: [v] * len(q) for k, q in QUESTIONS.items()}
    a.update(over)
    return a


def test_score_range():
    assert transformed_scores(all_answers(1))["qixu"] == 0.0
    assert transformed_scores(all_answers(5))["qixu"] == 100.0


def test_pinghe_judgement():
    a = all_answers(1)
    a["pinghe"] = [5 if not rev else 1 for _, rev in QUESTIONS["pinghe"]]
    r = judge(transformed_scores(a))
    assert r["verdict"]["pinghe"] == "是" and r["primary"] == "pinghe"


def test_biased_primary():
    r = judge(transformed_scores(all_answers(1, yangxu=[5] * 7, qixu=[4] * 8)))
    assert r["primary"] == "yangxu" and "qixu" in r["secondary"]


def test_bad_input():
    with pytest.raises(ValueError):
        transformed_scores(all_answers(1, qixu=[1, 2]))


def test_pulse_slow_deep():
    p = analyze_pulse(synth_pulse(bpm=50, amps=(0.5, 0.8, 1.0)))
    assert abs(p["心率(次/分)"] - 50) < 3
    assert "迟" in p["脉象标签"] and "沉" in p["脉象标签"]


def test_pulse_fast():
    assert "数" in analyze_pulse(synth_pulse(bpm=100, jitter=0.05))["脉象标签"]


def test_tongue_runs():
    assert "舌色" in analyze_tongue(Image.new("RGB", (100, 100), (230, 140, 140)))


def test_pipeline_end_to_end():
    r = pipeline.run(all_answers(2), Image.new("RGB", (80, 80), (200, 150, 150)), None, synth_pulse())
    assert r["primary_name"] and "问卷" in r["sources_used"]
