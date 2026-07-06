# 阶段 01 · CPT（继续预训练）

> 一句话：让模型"多读领域的书"，把中文医学（尤其中医/TCM）文本读得更顺。

## 1. 喂的是什么数据

**无标注的原始医学文本**（不是问答对）。约 0.9B tokens，以中医为主。

**数据出处：**

| 数据集 | 链接 | 说明 |
|--------|------|------|
| SylvanL/Traditional-Chinese-Medicine-Dataset-Pretrain | [🤗 HF](https://huggingface.co/datasets/SylvanL/Traditional-Chinese-Medicine-Dataset-Pretrain) | ~533k 条中医文本，~575MB，99%+ 中文；**主要语料** |
| FreedomIntelligence/TCM-Pretrain-Data-ShizhenGPT | [🤗 HF](https://huggingface.co/datasets/FreedomIntelligence/TCM-Pretrain-Data-ShizhenGPT) | 含 TCM_Book_Corpus 与 TCM_Web_Corpus 两个子集，补充中医书籍/网页文本 |
| shibing624/medical | [🤗 HF](https://huggingface.co/datasets/shibing624/medical) | 百科+教材，~0.2B tokens（可选，需 `trust_remote_code`） |

下面是训练语料里的一条真实样本：

```
李某,男,28岁,2017年4月14日初诊。自诉持续性鼻塞10年有余,涕少质黏,色白或黄,头昏头重,
嗅觉减退……检查:双侧下鼻甲肥厚,色黯红,呈桑椹样。辨为邪毒侵犯鼻窍,阻塞脉络,气血流通不畅,
治拟补气活血、祛邪化瘀,方选补阳还五汤加减:黄芪40g,当归、川芎、赤芍……每日1剂,水煎服。
服药7剂,患者鼻塞症状稍减轻……
```

注意这里全是**专业表述**：`下鼻甲`、`桑椹样`、`补阳还五汤`、`祛邪化瘀`、具体药材克数。通用 base 模型
在预训练里见得少。

## 2. 训练目标（机制）

标准的**语言建模**：预测下一个 token。没有"问题/答案"，只是让模型在海量领域文本上继续做
next-token prediction，把领域词汇、搭配、行文方式的概率分布学进参数里。

## 3. 进步体现在哪：困惑度（perplexity）

**困惑度**衡量模型面对一段文本时有多"意外"——可以粗略理解为"每一步平均在多少个候选里犹豫"，
**越低越熟悉**。拿上面这段医案（留出、未训练）实测：

| 模型 | 这段医案的困惑度 |
|------|:---:|
| base（Qwen3-1.7B-Base） | **7.85** |
| CPT 后 | **6.84** |

在整个留出集上（更稳的指标）：**7.04 → 5.79，下降 1.25（约 −18%）**。

**这说明**：CPT 确实把"专业文本"读顺了——面对 `补阳还五汤` 这种表述，CPT 后的模型不再那么"意外"。
这是三个阶段里**最干净、最确定的一个正增益**。

## 4. 诚实说明

- CPT **不直接提升选择题准确率**（本项目里 CPT-only 的知识准确率甚至略降 0.661→0.636）。
  原因：CPT 只教"读文本"，不教"按题目格式作答"——那是下一阶段 SFT 的事。CPT 的价值是把领域知识
  "灌"进参数，为后续阶段打底。
- 想让 CPT 的领域知识转化为下游准确率，通常需要 **CPT→SFT** 串联（本项目的 R3 就是验证这一点的配置）。

→ 下一阶段：[02 · SFT](./02_sft.md)
