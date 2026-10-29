# tcm-constitution：AI 中医体质辨识（九种体质）

四诊信息采集 → 体质量表计分 → 多源融合 → 体质分析报告（含节气调养、饮食起居运动、经络与膏方方向）。

> ⚠️ **免责声明**：本项目仅用于健康管理与体质自测研究，**不是医疗器械，不提供诊断或治疗建议**。结果须由执业中医师解读。

## 功能与现状

| 模块 | 文件 | 现状 |
|---|---|---|
| 问诊 · 九种体质量表 | `tcm/questionnaire.py` | ✅ 规则引擎（转化分、是/倾向是/否、平和质判定）。题库为示例表述，**使用前请对照 ZYYXH/T157-2009 原文核对** |
| 舌诊 / 面诊 | `tcm/vision.py` | 🟡 颜色统计基线（未经临床标定）；有训练好的权重时自动切换为 CNN（见下） |
| 舌象 CNN | `tcm/tongue_cnn.py` `tcm/train_tongue.py` `tcm/tongue_data.py` | ✅ 多任务 ResNet18（舌色/苔色/苔质/齿痕/裂纹）训练与推理代码；**权重需用你的标注数据训练** |
| 对话问诊 | `tcm/interview.py` | ✅ Qwen 对话 + 规则引擎打分；无 API Key 时退回逐题问答；含红旗症状/危机拦截 |
| 脉诊 | `tcm/pulse.py` | 🟡 浮沉迟数/节律/强弱的基础信号分析；18 种单脉象、178 种复合脉象需自有传感器数据与标注 |
| 融合 | `tcm/pipeline.py` | ✅ 问卷为主，舌/面/脉为辅助证据（每体质最多 +8 分，可调） |
| 报告 | `tcm/report.py` | ✅ HTML + 雷达图；设置 `DASHSCOPE_API_KEY` 后由通义千问生成解读 |
| 知识库 | `tcm/knowledge.py` | 🟡 通用科普内容 + 24 节气，**须经中医师审校** |
| 界面 | `app.py` | ✅ Gradio |

**未实现（需要数据/验证后才能做）：** 168 种疾病风险预警、12 经络量化监测、女性/儿童专项、膏方推荐。当前只给"体质易感倾向"提示。

## 快速开始

```bash
pip install -r requirements.txt
python -m tcm.cli        # 生成 reports/demo.html
python app.py            # 打开 http://127.0.0.1:7860（含对话问诊 / 问卷 / 舌面脉）
python -m tcm.chat_cli   # 终端对话问诊
pytest -q
```

启用大模型解读（可选）：

```bash
export DASHSCOPE_API_KEY=sk-xxx     # 阿里云百炼
export QWEN_MODEL=qwen-plus         # 可改
```

## 接入你自己的模型 / 设备

- **舌象模型**：实现 `predict(PIL.Image) -> {"舌色":..,"苔色":..,"苔质":..}`，传入 `analyze_tongue(img, model=...)`。
- **脉象传感器**：整理为 `{"light","mid","heavy"}` 三路数组 + 采样率；或用 CSV（列 `light,mid,heavy`）。

## 路线图

1. 与中医师合作校对题库、知识库；2. 采集经伦理审查/知情同意的数据；
3. 训练舌/面模型，用 Cohen's κ 对比医师判定；4. 脉象分类；5. 融合权重从数据中学习。

## 隐私与合规

不要把真实个人健康数据提交到仓库（`.gitignore` 已排除 `data/real/`、`*.csv`）。面向公众提供服务前，请核查医疗器械软件与个人信息保护的相关法规。

## 许可证

MIT（代码）。数据与模型权重请另行声明许可。

## 对话式问诊

```
用户自由表述 ──▶ [LLM 提取器] ──▶ 题目频度分(1~5，白名单校验) ──▶ 规则引擎打分/判定 ──▶ 报告
                     ▲                                              （LLM 不参与判定）
          [LLM 提问器：把待答题合并成自然口语]
```

- 相同题干只问一次（如“容易疲乏”同属平和质与气虚质），答案共享。
- 用户可以说“跳过”“结束”；某体质有效作答 < 60% 时标记为“未评估”，信息不足时报告不强行下结论。
- 用户发言按数据处理，LLM 输出只接受“当前待答题目 id + 1~5 整数”，其余丢弃（防提示词注入/幻觉）。
- 命中胸痛、呼吸困难、便血等红旗症状或自杀/自残表述：立即停止问诊并建议就医/求助，不生成报告。
- 红旗与危机关键词表在 `tcm/interview.py`，**请中医师/临床人员审校补充**。

## 训练舌象 CNN

```bash
pip install -r requirements-train.txt
# 1) 按 data/tongue_example/README.md 准备 labels.csv 与图片（真实数据放 data/real/，不要提交）
python -m tcm.train_tongue --data data/real/tongue/labels.csv --epochs 30 --pretrained --out models/tongue.pt
# 2) 流程冒烟测试（合成数据，无临床意义）
python -m tcm.train_tongue --synthetic --epochs 2 --img-size 64 --out /tmp/t.pt
# 3) 启动应用：存在 models/tongue.pt（或环境变量 TONGUE_MODEL）即自动启用 CNN
python app.py
```

训练细节：按患者划分验证集、类别加权、仅轻微亮度增强（不做色相抖动，避免破坏舌色标签）、
以各头 macro-F1 均值选最佳模型，指标写入 `models/tongue.metrics.json`。
CNN 置信度（最低头）< 0.5 时，其舌象证据不会参与体质融合。
上线前须用独立测试集 + 中医师一致性（Cohen's κ）验证，并处理不同相机/光照的域偏移。

> 说明：训练部分（`train_tongue.py`、`tongue_cnn.py`）在开发环境中仅做了语法检查，未装 PyTorch 实跑；
> 请在你本机用 `--synthetic` 先跑一遍；`tests/test_tongue.py` 在安装 torch 后会自动执行训练-推理往返测试。
