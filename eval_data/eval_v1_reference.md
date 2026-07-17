# `eval_v1` 参考运行（非金标结论）

运行日期：2026-07-17

命令：`python eval.py --snapshot eval_data/eval_v1.json --split test`

环境：Python 3.13.14、FastEmbed 0.8.0、`paraphrase-multilingual-MiniLM-L12-v2`

这份结果用于展示评测脚本的实际输出，不是完整真值或跨环境冻结基准。标签是 legacy
LLM-as-a-judge 银标签；向量模型文件未随仓库固定，当前 FastEmbed registry 解析为 mean pooling；
最后一列是 top-10 的 judgment coverage，必须和 Precision 一起读。
随机抽样若 top-10 没有任何 judged，P(judged) 未定义，因此随机 Precision 的均值只统计
有效试验；脚本会同时打印有效试验数。

候选池为 135 篇：排除 3 个 anchor 和 72 个 dev 样本，保留 30 个 test judged
（7 rel / 23 irrel）与 105 个 unjudged 干扰项。LLM 排序只看 RRF top-30，和生产漏斗一致。

| 排序方案 | P@10 (judged) | R@10 (judged) | F1 | Coverage |
|---|---:|---:|---:|---:|
| 随机基线（1000 次） | 0.219 ± 0.291 | 0.076 ± 0.100 | — | 0.231 ± 0.123 |
| 纯 BM25 | 0.667 | 0.571 | 0.615 | 0.600（6/10 judged） |
| 纯向量 | 0.667 | 0.286 | 0.400 | 0.300（3/10 judged） |
| 混合 RRF | 0.833 | 0.714 | 0.769 | 0.600（6/10 judged） |
| 混合 + 冻结 legacy LLM 分数 | 1.000 | 0.571 | 0.727 | 0.400（4/10 judged） |

`score >= 7` 的二分类结果只覆盖 RRF top-30 中 14 个 judged 样本（占候选 14/30，
占 test judged pool 14/30）：TP=5、FP=0、FN=2、TN=7，相关类 Precision=1.000、
Recall=0.714、F1=0.833、Balanced Accuracy=0.857、Cohen's kappa=0.714。

这里最重要的结论不是“Precision=1.0”，而是：在低覆盖下 legacy LLM 精排提高了已判定
top-10 的纯度，却把 judged Recall 从 RRF 的 0.714 降到 0.571。该结果支持继续保留
RRF 降级路径、扩大人工标注覆盖，并避免只汇报单个高 Precision 数字。
