<div align="center">

# tcm-constitution

**AI 中医体质辨识 · AI-Assisted TCM Constitution Assessment**

[中文](#中文) | [English](#english)

</div>

> ⚠️ **免责声明 / Disclaimer**
> 本项目仅用于健康管理与科研学习，**不是医疗器械，不提供诊断或治疗建议**，结果须由执业中医师解读。
> This project is for health-management and research use only. It is **not a medical device and does not provide diagnosis or treatment**. Results must be interpreted by a licensed TCM practitioner.

---

<a id="中文"></a>

## 中文

### 简介

`tcm-constitution` 是一个开源的 **九种体质（平和质、气虚质、阳虚质、阴虚质、痰湿质、湿热质、血瘀质、气郁质、特禀质）** 辨识原型。它围绕中医“望、闻、问、切”的思路，把 **问诊 / 舌诊 / 面诊 / 脉诊** 的信息数字化，经多源融合后生成体质分析报告和个性化养生建议，供医生进一步解读。

设计原则：**让大模型负责“聊天”，让确定性算法负责“判定”**——LLM 只做提问和信息提取，体质得分与判定完全由可复现的规则引擎完成。

### 功能与现状

| 模块 | 文件 | 状态 |
|---|---|---|
| 问诊 · 九种体质量表 | `tcm/questionnaire.py` | ✅ 转化分计算、“是/倾向是/否”、平和质判定、部分作答处理。题库为示例表述，**使用前请对照 ZYYXH/T157-2009 原文核对** |
| 对话式问诊 | `tcm/interview.py` | ✅ Qwen 自然对话 + 规则引擎打分；无 API Key 时退回逐题问答；含红旗症状/危机拦截 |
| 舌诊 / 面诊 | `tcm/vision.py` | 🟡 颜色统计基线（未经临床标定）；有 CNN 权重时自动切换 |
| 舌象 CNN | `tcm/tongue_cnn.py`<br>`tcm/train_tongue.py`<br>`tcm/tongue_data.py` | ✅ 多任务 ResNet18（舌色/苔色/苔质/齿痕/裂纹）训练与推理代码；**权重需用你的标注数据训练** |
| 脉诊 | `tcm/pulse.py` | 🟡 浮/中/沉三层信号的浮沉迟数、节律、强弱基础分析；18 种单脉象、178 种复合脉象需自有传感器与标注数据 |
| 多源融合 | `tcm/pipeline.py` | ✅ 问卷为主，舌/面/脉为辅助证据（每种体质最多 +8 分，可调） |
| 报告生成 | `tcm/report.py` | ✅ HTML 报告：体质雷达图、易感倾向、经络与穴位、节气调养、饮食起居运动、膏方方向；可选 Qwen 解读 |
| 知识库 | `tcm/knowledge.py` | 🟡 通用科普内容 + 24 节气，**须经中医师审校** |
| 界面 | `app.py` | ✅ Gradio：对话问诊 / 问卷问诊 / 舌面脉上传 |

**暂未实现**（缺少经验证的数据，贸然实现会误导用户）：168 种疾病风险预警、12 经络量化监测、女性/儿童专项分析、膏方个性化推荐。目前仅给出“体质易感倾向”的科普提示。

### 架构

```
 对话/问卷 ─▶ [LLM 提取：自由表述 → 1~5 频度分] ─┐
 舌象照片 ──▶ [CNN 或颜色基线]  ────────────────┼─▶ 多源融合 ─▶ 规则引擎判定 ─▶ 体质报告
 面部照片 ──▶ [面色分析]        ────────────────┤                              (HTML + Qwen 解读)
 脉象信号 ──▶ [浮沉迟数/节律]   ────────────────┘
```

### 快速开始

```bash
git clone https://github.com/<你的用户名>/tcm-constitution.git
cd tcm-constitution
pip install -r requirements.txt

python -m tcm.cli          # 用合成数据生成示例报告 reports/demo.html
python app.py              # 启动 Web 界面 http://127.0.0.1:7860
python -m tcm.chat_cli     # 终端对话问诊
pytest -q                  # 运行测试
```

启用大模型（可选，阿里云百炼 / 通义千问）：

```bash
export DASHSCOPE_API_KEY=sk-xxxx
export QWEN_MODEL=qwen-plus      # 可改
```

### 对话式问诊的安全设计

- LLM 只负责**提问措辞**与**信息提取**；打分与判定不经过 LLM。
- 用户发言按数据处理；LLM 输出只接受“当前待答题目 id + 1~5 整数”，其余丢弃（防幻觉、防提示词注入）。
- 命中胸痛、呼吸困难、便血等**红旗症状**，或自杀/自残等**危机表述**：立即停止问诊，建议就医或求助，不生成报告。关键词表在 `tcm/interview.py`，**请临床人员审校补充**。
- 相同题干只问一次，某体质有效作答不足 60% 时标记“未评估”，不强行下结论。

### 训练舌象 CNN

```bash
pip install -r requirements-train.txt

# 1) 按 data/tongue_example/README.md 准备 labels.csv 与图片（真实数据放 data/real/，不要提交）
python -m tcm.train_tongue --data data/real/tongue/labels.csv --epochs 30 --pretrained --out models/tongue.pt

# 2) 流程冒烟测试（合成数据，无临床意义）
python -m tcm.train_tongue --synthetic --epochs 2 --img-size 64 --out /tmp/t.pt

# 3) 存在 models/tongue.pt（或环境变量 TONGUE_MODEL）时，app.py 自动启用 CNN
python app.py
```

要点：按患者划分验证集；类别加权；仅轻微亮度增强（不做色相抖动，避免破坏舌色标签）；以各头 macro-F1 均值选最佳模型；最低头置信度 < 0.5 时，舌象证据不参与融合。

> 训练代码在开发环境中仅做了语法检查，未装 PyTorch 实跑，请先在本机用 `--synthetic` 验证。

### 接入自己的模型 / 设备

- **舌象模型**：实现 `predict(PIL.Image) -> {"舌色":.., "苔色":.., "苔质":.., ...}`，传入 `analyze_tongue(img, model=...)`。
- **脉象传感器**：整理为 `{"light","mid","heavy"}` 三路数组 + 采样率；或使用 CSV（列 `light,mid,heavy`，默认 100 Hz）。

### 目录结构

```
tcm-constitution/
├── app.py                  # Gradio 界面
├── tcm/
│   ├── questionnaire.py    # 九种体质量表与判定
│   ├── interview.py        # 对话式问诊（LLM + 规则）
│   ├── vision.py           # 舌/面基线与证据映射
│   ├── tongue_data.py      # 标签体系、数据读取、指标
│   ├── tongue_cnn.py       # 舌象 CNN 模型与推理
│   ├── train_tongue.py     # 训练脚本
│   ├── pulse.py            # 脉象信号分析
│   ├── pipeline.py         # 多源融合
│   ├── report.py           # 报告生成
│   ├── knowledge.py        # 调养知识库与 24 节气
│   ├── cli.py / chat_cli.py
├── tests/                  # 单元测试（CI 自动运行）
├── data/tongue_example/    # 舌象数据格式说明
└── .github/workflows/ci.yml
```

### 数据、隐私与合规

- 舌/面照片、问诊记录属于**个人健康信息**：须知情同意、脱敏，**不要提交到仓库**（`.gitignore` 已排除 `data/real/`、`*.csv`）。
- 面向公众提供服务前，请核查所在地对**医疗器械软件、医疗 AI、个人信息保护**的监管要求；产品定位建议为“体质自测与健康管理参考”。
- 上线前须用独立测试集和中医师判定做一致性检验（如 Cohen's κ），并评估不同相机/光照下的域偏移。

### 路线图

1. 与中医师合作校对题库、知识库与红旗关键词表
2. 采集经伦理审查、知情同意的数据，训练并验证舌/面模型
3. 脉象分类（对接真实传感器与医师标注）
4. 融合权重由数据学习，替代当前人工规则
5. 多语言与更完善的报告模板

### 贡献

欢迎通过 Issues / Pull Requests 参与，尤其欢迎**中医专业人士**对题库、知识库与规则的审校。提交代码前请运行 `pytest -q`。

### 许可证

代码采用 [MIT](LICENSE) 许可证；数据与模型权重请另行声明许可。

---

<a id="english"></a>

## English

### Overview

`tcm-constitution` is an open-source prototype for assessing the **nine Traditional Chinese Medicine (TCM) body constitutions** — Balanced (平和), Qi-deficiency (气虚), Yang-deficiency (阳虚), Yin-deficiency (阴虚), Phlegm-dampness (痰湿), Damp-heat (湿热), Blood-stasis (血瘀), Qi-stagnation (气郁) and Special/allergic (特禀). Following the TCM "four diagnostic methods" (inspection, listening/smelling, inquiry, palpation), it digitizes **inquiry, tongue, face and pulse** information, fuses it, and produces a constitution report with personalized wellness suggestions for a physician to review.

Design principle: **the LLM chats, a deterministic engine decides.** The LLM only phrases questions and extracts information; scores and verdicts come from a reproducible rule engine.

### Features & Status

| Module | Files | Status |
|---|---|---|
| Inquiry · nine-constitution scale | `tcm/questionnaire.py` | ✅ Transformed scores, "yes / tendency / no" verdicts, Balanced-type rule, partial-answer handling. Item wording is an **example bank — verify against the standard ZYYXH/T157-2009 before real use** |
| Conversational interview | `tcm/interview.py` | ✅ Qwen-driven dialogue + rule-engine scoring; falls back to question-by-question mode without an API key; red-flag / crisis guard |
| Tongue / face analysis | `tcm/vision.py` | 🟡 Color-statistics baseline (not clinically calibrated); auto-switches to the CNN when weights exist |
| Tongue CNN | `tcm/tongue_cnn.py`<br>`tcm/train_tongue.py`<br>`tcm/tongue_data.py` | ✅ Multi-task ResNet18 (tongue color / coating color / coating thickness / teeth marks / cracks) training & inference code; **weights must be trained on your own annotated data** |
| Pulse analysis | `tcm/pulse.py` | 🟡 Basic analysis of light/mid/heavy-pressure signals (floating/deep, slow/rapid, rhythm, strength). Recognizing 18 single + 178 compound pulse types requires your sensor hardware and annotated data |
| Multi-source fusion | `tcm/pipeline.py` | ✅ Questionnaire is primary; tongue/face/pulse add supporting evidence (capped at +8 per constitution, tunable) |
| Report generation | `tcm/report.py` | ✅ HTML report: radar chart, susceptibility hints, meridians & acupoints, 24-solar-term regimen, diet / lifestyle / exercise, herbal-paste (膏方) direction; optional Qwen narrative |
| Knowledge base | `tcm/knowledge.py` | 🟡 General wellness content + 24 solar terms — **must be reviewed by a TCM practitioner** |
| UI | `app.py` | ✅ Gradio: chat interview / questionnaire / tongue-face-pulse upload |

**Not implemented** (no validated data; implementing them naively would mislead users): 168-disease risk warnings, quantitative 12-meridian monitoring, women's / children's modules, personalized herbal-paste recommendation. The report only gives general "constitution susceptibility" hints.

### Architecture

```
 Chat / form ─▶ [LLM extractor: free text → 1–5 frequency score] ─┐
 Tongue photo ▶ [CNN or color baseline]  ─────────────────────────┼─▶ Fusion ─▶ Rule-engine verdict ─▶ Report
 Face photo ──▶ [complexion analysis]    ─────────────────────────┤                                  (HTML + Qwen narrative)
 Pulse signal ▶ [pressure layers / rate / rhythm] ────────────────┘
```

### Quick Start

```bash
git clone https://github.com/<your-username>/tcm-constitution.git
cd tcm-constitution
pip install -r requirements.txt

python -m tcm.cli          # demo report from synthetic data → reports/demo.html
python app.py              # web UI at http://127.0.0.1:7860
python -m tcm.chat_cli     # terminal chat interview
pytest -q                  # run tests
```

Enable the LLM (optional; Alibaba Cloud Model Studio / Qwen):

```bash
export DASHSCOPE_API_KEY=sk-xxxx
export QWEN_MODEL=qwen-plus      # optional override
```

### Safety Design of the Chat Interview

- The LLM only handles **question phrasing** and **information extraction**; scoring and verdicts never pass through it.
- User text is treated as data. LLM output is accepted only as "currently pending item id + integer 1–5"; everything else is discarded (guards against hallucination and prompt injection).
- **Red-flag symptoms** (chest pain, dyspnea, blood in stool, …) or **crisis statements** (self-harm / suicide) immediately stop the interview, advise seeking medical help or support, and no report is generated. The keyword lists live in `tcm/interview.py` and **should be reviewed and extended by clinicians**.
- Duplicate items are asked once; a constitution with < 60 % valid answers is marked "not assessed" instead of forcing a conclusion.

### Training the Tongue CNN

```bash
pip install -r requirements-train.txt

# 1) Prepare labels.csv and images per data/tongue_example/README.md
#    (keep real data under data/real/ — never commit it)
python -m tcm.train_tongue --data data/real/tongue/labels.csv --epochs 30 --pretrained --out models/tongue.pt

# 2) Pipeline smoke test (synthetic data, no clinical meaning)
python -m tcm.train_tongue --synthetic --epochs 2 --img-size 64 --out /tmp/t.pt

# 3) If models/tongue.pt (or $TONGUE_MODEL) exists, app.py uses the CNN automatically
python app.py
```

Highlights: patient-level train/val split; class-weighted loss; only mild brightness augmentation (no hue jitter, which would corrupt color labels); best checkpoint chosen by mean macro-F1 across heads; if the lowest head confidence is < 0.5, tongue evidence is excluded from fusion.

> The training code was only syntax-checked in the development sandbox (PyTorch was not installed there). Please validate it locally with `--synthetic` first.

### Plugging in Your Own Model / Hardware

- **Tongue model**: implement `predict(PIL.Image) -> {"舌色":.., "苔色":.., "苔质":.., ...}` and pass it via `analyze_tongue(img, model=...)`.
- **Pulse sensor**: provide `{"light","mid","heavy"}` arrays plus the sampling rate, or a CSV with columns `light,mid,heavy` (default 100 Hz).

### Repository Layout

```
tcm-constitution/
├── app.py                  # Gradio UI
├── tcm/
│   ├── questionnaire.py    # nine-constitution scale & verdicts
│   ├── interview.py        # conversational interview (LLM + rules)
│   ├── vision.py           # tongue/face baseline & evidence mapping
│   ├── tongue_data.py      # label schema, data loading, metrics
│   ├── tongue_cnn.py       # tongue CNN model & inference
│   ├── train_tongue.py     # training script
│   ├── pulse.py            # pulse signal analysis
│   ├── pipeline.py         # multi-source fusion
│   ├── report.py           # report generation
│   ├── knowledge.py        # wellness knowledge base & 24 solar terms
│   ├── cli.py / chat_cli.py
├── tests/                  # unit tests (run by CI)
├── data/tongue_example/    # tongue dataset format guide
└── .github/workflows/ci.yml
```

### Data, Privacy & Compliance

- Tongue/face photos and interview records are **personal health information**: obtain informed consent, de-identify, and **never commit them** (`.gitignore` already excludes `data/real/` and `*.csv`).
- Before offering this to the public, check local regulations on **medical-device software, medical AI and personal-data protection**; position the product as "constitution self-assessment and wellness reference".
- Before deployment, validate against an independent test set and practitioner judgments (e.g., Cohen's κ) and assess domain shift across cameras and lighting.

### Roadmap

1. Work with TCM practitioners to review the item bank, knowledge base and red-flag keywords
2. Collect ethics-approved, consented data; train and validate tongue/face models
3. Pulse-type classification (real sensors + practitioner labels)
4. Learn fusion weights from data instead of hand-written rules
5. Multi-language support and richer report templates

### Contributing

Issues and Pull Requests are welcome — especially **reviews from TCM professionals** of the item bank, knowledge base and rules. Please run `pytest -q` before submitting.

### License

Code is released under the [MIT License](LICENSE). Declare licenses for data and model weights separately.
