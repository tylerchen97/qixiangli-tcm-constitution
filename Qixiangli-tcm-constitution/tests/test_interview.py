import json
import pytest
from tcm.questionnaire import QUESTIONS, transformed_scores, judge
from tcm.interview import Interview, FACTS, parse_frequency
from tcm import pipeline


class FakeLLM:
    """按脚本返回；记录调用。json_mode 时返回 answers，否则返回提问文本。"""
    def __init__(self, answers_fn): self.fn, self.calls = answers_fn, []
    def chat(self, messages, json_mode=False):
        self.calls.append(messages)
        if json_mode:
            return "```json\n" + json.dumps({"answers": self.fn(messages[-1]["content"])}) + "\n```"
        return "最近怎么样？可以用没有/很少/有时/经常/总是回答。"


def test_facts_dedupe_shared_text():
    assert len(FACTS) < sum(len(v) for v in QUESTIONS.values())
    f = next(f for f in FACTS if f["text"] == "您容易疲乏吗？")
    assert {s for s, _ in f["slots"]} == {"pinghe", "qixu"}


def test_parse_frequency():
    assert parse_frequency("经常") == 4 and parse_frequency("没有") == 1
    assert parse_frequency("很少") == 2 and parse_frequency("总是这样") == 5
    assert parse_frequency("5") == 5 and parse_frequency("嗯嗯") is None


def test_offline_full_interview_to_report():
    it = Interview(None)
    it.start()
    n = 0
    while not it.done and n < 200:
        it.reply("有时" if n % 3 else "经常")
        n += 1
    assert it.done and it.n_answered() == len(FACTS)
    ans = it.to_answers()
    assert all(v is not None for vs in ans.values() for v in vs)
    assert pipeline.run(ans)["primary_name"]


def test_skip_and_partial_scoring():
    it = Interview(None); it.start()
    it.reply("跳过")
    assert it.progress()[0] == 1 and it.n_answered() == 0
    it.reply("结束")
    res = pipeline.run(it.to_answers())      # 几乎无信息
    assert res["primary_name"].startswith("信息不足")


def test_unparseable_retries_then_skips():
    it = Interview(None); it.start()
    first = it.asked[0]
    it.reply("嗯？")
    assert it.asked == [first] and first not in it.skipped
    it.reply("啊这")
    assert first in it.skipped


def test_red_flag_and_crisis_stop():
    it = Interview(None); it.start()
    assert "120" in it.reply("我最近有胸痛")
    assert it.stopped == "red" and it.done
    it2 = Interview(None); it2.start()
    assert "心理" in it2.reply("我不想活了")
    assert it2.stopped == "crisis"


def test_llm_mode_extracts_only_pending_valid_ids():
    def fn(prompt):
        return {"f0": 5, "f1": 4, "zzz": 3, "f2": 9, "f3": "4", "f4": True, "f5": 2.5}
    it = Interview(FakeLLM(fn)); it.start()
    it.reply("我精力不错，也不怎么累")
    assert it.values == {"f0": 5, "f1": 4}      # 非法 id/越界/非整数/布尔/字符串 全部丢弃


def test_llm_failure_falls_back_to_rules():
    class Boom:
        def chat(self, *a, **k): raise RuntimeError("net")
    it = Interview(Boom()); it.start()
    it.reply("经常")
    assert it.n_answered() == 0 and len(it.asked) == 1 and it.group == 1   # 降级逐题，不套用同一分数
    it.reply("经常")
    assert it.n_answered() == 1


def test_partial_questionnaire_coverage():
    a = {k: [None] * len(v) for k, v in QUESTIONS.items()}
    a["yangxu"] = [5, 5, 5, 5, 5, None, None]     # 5/7 ≥ 60%
    s = transformed_scores(a, allow_partial=True)
    assert s["yangxu"] == 100.0 and s["qixu"] is None
    assert judge(s)["primary"] == "yangxu"
    with pytest.raises(ValueError):
        transformed_scores(a)                      # 严格模式拒绝 None
