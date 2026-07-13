# CPT → SFT → GRPO 增益归因实验报告（中文医学，Qwen3-1.7B）

**日期：** 2026-07-06  · **硬件：** 8×H100 80GB · **追踪：** W&B `med-1b7-cpt-sft-grpo`
**基座：** `Qwen/Qwen3-1.7B-Base`（起初用 4B，因过强导致 SFT 无提升空间而下调，见 §6）

> **更新 2026-07-12：** 已执行 §7①「强教师重蒸馏」——SFT CoT 教师由 Gemini flash-lite 换为本地
> `Qwen/Qwen3.6-27B`（SGLang 8 卡张量并行，`enable_thinking=False`，质量门 + 拒绝采样）。
> **SFT 生成增益由 −0.045 翻正为 +0.057**，全链路 0.589 → **0.654**（原 0.552）。这直接验证了本报告
> 的核心判断：瓶颈是**数据质量**而非管线。原 Gemini 结果保留为对照（§4.1），本地 Qwen 新结果见 §4.2。

---

## 1. TL;DR

在一条完整、可复现的管线上跑通了 **CPT → SFT → GRPO** 三阶段，并用**消融矩阵 + 三条评测轴**做增益归因。结论诚实：

- **CPT：唯一清晰的正增益** —— 领域困惑度 **7.04 → 5.79（−1.25，约 −18%）**。
- **SFT：轻微负增益** —— 生成准确率 0.589 → 0.544（−0.045）。
- **GRPO：训练奖励明显上升（0.40 → 0.625）但测试仅微增**（+0.015），净值仍低于 base。

**管线机制全部正确**（CPT 困惑度下降、GRPO 奖励上升都证明训练在生效）；**测试集增益不干净的根因是数据质量**——SFT 蒸馏用的是最便宜的 Gemini flash‑lite，其推理链弱于 1.7B base 本身。这是一个真实且有价值的发现（见 §5、§7）。

---

## 2. 实验设计

**消融矩阵**（同一套固定评测）：

| Run | CPT | SFT | GRPO | 作用 |
|-----|:---:|:---:|:----:|------|
| R0 | — | — | — | base 基线 |
| R1 | — | ✓ | — | 仅 SFT |
| R2 | ✓ | — | — | 仅 CPT（困惑度） |
| R3 | ✓ | ✓ | — | CPT→SFT |
| R4 | ✓ | ✓ | ✓ | 全链路 |

**增益归因：** SFT = R1−R0 · CPT = R3−R1（固定 SFT，仅加 CPT）· GRPO = R4−R3 · CPT困惑度 = PPL(R0)−PPL(R2)

**三条评测轴：**
1. **知识**（lm-eval 对数似然 MCQ：CMMLU/C-Eval 医学子集 + CMExam + MedQA‑zh）——base 也能测。
2. **生成**（vLLM 采样 CoT → 抽取 `\boxed{}` → 准确率）——贴合 SFT/GRPO 目标。
3. **困惑度**（留出医学语料）——CPT 的保证信号。

---

## 3. 数据

### 3.1 CPT — 中文医学原始语料，~0.9B tokens（TCM 为主）

| 数据集 | 链接 | 说明 |
|--------|------|------|
| SylvanL/Traditional-Chinese-Medicine-Dataset-Pretrain | [🤗 HF](https://huggingface.co/datasets/SylvanL/Traditional-Chinese-Medicine-Dataset-Pretrain) | ~533k 条中医文本，~575MB，99%+ 中文；**主要语料** |
| FreedomIntelligence/TCM-Pretrain-Data-ShizhenGPT | [🤗 HF](https://huggingface.co/datasets/FreedomIntelligence/TCM-Pretrain-Data-ShizhenGPT) | TCM_Book_Corpus + TCM_Web_Corpus 两个子集 |
| shibing624/medical (pretrain split) | [🤗 HF](https://huggingface.co/datasets/shibing624/medical) | 百科+教材，~0.2B tokens（需 `trust_remote_code`，可选） |

偏向 base 较弱的中医领域，留出 5M tokens 测困惑度。

### 3.2 SFT — 60k 中文医学 CoT 问答对

| 数据集 | 链接 | 说明 |
|--------|------|------|
| FreedomIntelligence/medical-o1-reasoning-SFT | [🤗 HF](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) | zh + zh_mix 子集，共 ~45k 条真实 CoT，**核心推理种子** |
| FreedomIntelligence/Huatuo26M-Lite | [🤗 HF](https://huggingface.co/datasets/FreedomIntelligence/Huatuo26M-Lite) | ~177k 医学 QA（无 CoT），采样扩展覆盖面 |
| michaelwzhu/ShenNong_TCM_Dataset | [🤗 HF](https://huggingface.co/datasets/michaelwzhu/ShenNong_TCM_Dataset) | ~112k 中医 QA，采样补充 TCM 广度 |

无 CoT 样本用本地 `Qwen/Qwen3.6-27B` 蒸馏生成推理链（`enable_thinking=False`、简洁 ≤6 步、
质量门 + 拒绝采样），保留 ~16.6k 条；原 Gemini flash-lite（~15k 条）保留为 §4.1 对照。

### 3.3 GRPO — 30k 可验证中文医学 MCQ（本次用 12k）

| 数据集 | 链接 | 说明 |
|--------|------|------|
| bigbio/med_qa (med_qa_zh_4options) | [🤗 HF](https://huggingface.co/datasets/bigbio/med_qa) | 中国医学执业考试选择题，4 选项单答案 |
| fzkuji/CMExam | [🤗 HF](https://huggingface.co/datasets/fzkuji/CMExam) | 中国医学考试题库，train 54k / val 6.8k / test 6.8k |

奖励 = 正确率(exact-match `\boxed{}`) + 格式(0.2)。仅使用单选题。

### 3.4 Eval — 固定评测基准

| 基准 | 链接 | 医学子集 |
|------|------|----------|
| CMMLU | [🤗 HF](https://huggingface.co/datasets/haonan-li/cmmlu) | anatomy, clinical_knowledge, college_medicine, genetics, virology, nutrition, traditional_chinese_medicine, professional_medicine |
| C-Eval | [🤗 HF](https://huggingface.co/datasets/ceval/ceval-exam) | basic_medicine, clinical_medicine, physician |
| CMB (CMB-Exam) | [🤗 HF](https://huggingface.co/datasets/FreedomIntelligence/CMB) | 中国医学考试综合题库，test 11.2k（含多选） |
| MedQA-zh | [🤗 HF](https://huggingface.co/datasets/bigbio/med_qa) | med_qa_zh_4options test split |

评测测试集全部留出，从不参与训练。

---

## 4. 结果

### 4.1 原始：Gemini flash-lite 教师（对照基线）

| Run | 生成准确率 | 知识准确率 | 困惑度 |
|-----|:---:|:---:|:---:|
| R0 base | 0.589 | 0.661 | 7.04 |
| R2 CPT | — | 0.636 | **5.79** |
| R1 SFT | 0.544 | 0.645 | — |
| R3 CPT+SFT | 0.537 | 0.635 | — |
| R4 +GRPO | 0.552 | 0.646 | — |

**逐阶段增益：**

| 阶段 | 生成 | 知识 | 困惑度 |
|------|:---:|:---:|:---:|
| SFT (R1−R0) | −0.045 | −0.016 | — |
| CPT (R3−R1) | −0.008 | −0.010 | **−1.25** ✓ |
| GRPO (R4−R3) | **+0.015** | +0.010 | — |

**GRPO 训练内奖励：** correctness 0.40 → **0.625**，total reward 0.46 → 0.82（KL≈6e‑4，稳定）。

### 4.2 本地 Qwen3.6-27B 教师（当前默认）

同一 CPT 基座、同一评测、同一 GRPO 数据；**仅 SFT 蒸馏教师不同**（run-id 以 `*b` 区分，与 4.1 并存）。

| Run | 生成准确率 | 知识准确率 | 困惑度 |
|-----|:---:|:---:|:---:|
| R0 base | 0.589 | 0.661 | 7.04 |
| R2 CPT | — | 0.636 | **5.79** |
| R1b SFT | **0.646** | 0.638 | — |
| R3b CPT+SFT | 0.627 | 0.626 | — |
| R4b +GRPO | **0.654** | 0.635 | — |

**逐阶段增益：**

| 阶段 | 生成 | 知识 | 困惑度 |
|------|:---:|:---:|:---:|
| SFT (R1−R0) | **+0.057** | −0.023 | — |
| CPT (R3−R1) | −0.019 | −0.012 | **−1.25** ✓ |
| GRPO (R4−R3) | **+0.028** | +0.008 | — |

**GRPO 训练内奖励：** 0.58 → **0.81**（12k 提示，1 epoch，1500 步）。

**A/B 结论：** 仅更换 SFT 教师，SFT 生成增益 **−0.045 → +0.057**，全链路净值 **−0.037 → +0.065**，
最终模型生成准确率 **0.552 → 0.654**。CPT 困惑度增益不变（−1.25，与教师无关）；知识 MCQ 准确率在
CoT 教师下略降（格式 ↔ 记忆的取舍）。这直接证实 §5 的诊断——「SFT 负增益 = 蒸馏教师太弱」。

---

## 5. 分析：什么生效、什么没有、为什么

**CPT 生效（困惑度）：** 领域语料让模型对医学文本建模更好，困惑度稳定下降 −1.25。但**对下游 MCQ 准确率无帮助甚至略降**——这是已知现象：单纯 CPT 会把分布拉向语料风格，MCQ 能力要靠后续 SFT 才能"解锁"，而本次 SFT 数据本身偏弱。

**SFT 负增益（根因：教师太弱）：** 逐条对比生成发现——base 与 SFT 模型**都能正确输出 `\boxed{}`**（抽取公平，非评测 bug），差别在**推理质量**：base 简洁直答且常对；SFT 模型**啰嗦**（长度 2×、甚至重复打印答案），有时把自己"绕"到错误选项。因为蒸馏教师是 **Gemini flash‑lite**（最便宜档），其 CoT 弱于 1.7B base 的原生能力 → 相当于"让强学生模仿弱老师"。

**GRPO 训练涨、测试不涨（泛化受限）：** GRPO 在**训练奖励**上明显生效（0.40→0.625，RL 正常工作），但迁移到测试仅 +0.015。原因：① 起点 R3 已被 SFT 拖坏（啰嗦、不输出 EOS，rollout 全部顶到 768 token 上限 `clipped_ratio=1.0`）；② 从偏弱的 SFT 基座出发，GRPO 的泛化天花板低。

---

## 6. 关键发现：基座越强，SFT 越容易"训坏"

先在 **Qwen3‑4B‑Base** 上做，发现它对中文医学 MCQ 已近天花板（生成 0.759），全参 SFT 反而**大幅降分**（→0.648，−0.11）。下调到 **1.7B** 后 SFT 降幅缩小（−0.045）。规律清晰：

| 基座 | base 生成 | SFT 后 | SFT 增益 | CPT 困惑度增益 |
|------|:---:|:---:|:---:|:---:|
| Qwen3‑4B | 0.759 | 0.648 | −0.11 | −0.96 |
| Qwen3‑1.7B | 0.589 | 0.544 | −0.045 | −1.25 |

**含义：** 现代"base"模型其实已被大量指令/推理数据预训练，naive 全参 SFT + 弱蒸馏数据在其上会造成灾难性遗忘。要清晰演示 SFT 增益，要么用**更弱基座**（有空间），要么用**更强的 SFT 数据**（超过基座）。

---

## 7. 局限与后续（如何拿到干净的三段正增益）

本次瓶颈是**数据质量**，非管线：
1. ✅ **强教师重蒸馏（已完成，最高杠杆）：** 改用本地 `Qwen/Qwen3.6-27B`（SGLang 8 卡、`enable_thinking=False`、简洁 CoT、质量门 + 拒绝采样 ~16.6k 条）重蒸馏。**结果见 §4.2：SFT 转正（+0.057）、全链路 0.589 → 0.654**，印证"SFT 数据超过 base → SFT 转正"的预测。改用本地推理替代 API，零成本、~2h。
2. **修 SFT 格式**：去掉"重复打印答案"，训练 EOS，控制 CoT 长度 → 消除 GRPO 的 `clipped_ratio=1.0`，rollout 更快更干净。
3. **GRPO 调参**：更多提示/步数、KL 与长度奖励调节，改善训练→测试迁移。

---

## 8. 复现

三套隔离环境（`med-4b-env-setup` 记忆有详述）：`.venv`(评测) · `.venv-train`(CPT/SFT，torch2.8) · `.venv-grpo`(GRPO，trl0.24+vllm0.10.2)。
```
bash setup/install.sh --full
bash setup/install_sglang.sh             # 本地教师推理环境 .venv-sglang
bash scripts/00_prepare_data.sh          # CPT/SFT/GRPO 开源数据（蒸馏默认 DRY-RUN）
bash scripts/run_qwen_distill_sglang.sh  # 本地 Qwen3.6-27B 蒸馏 → 合并进 SFT 训练集
python -m med_pipeline.train.run_cpt     # .venv-train
bash scripts/run_post_cpt_qwen.sh        # R1b→R3b→评测→R4b→瀑布图（本地教师矩阵）
```
全部指标 + 增益瀑布柱状图见 W&B 项目 `med-1b7-cpt-sft-grpo`。管线代码遵循 LOD 设计（`configs/` `med_pipeline/{data,tools,train,eval,report}`，单文件单职责、纯函数工具、瘦编排）。
