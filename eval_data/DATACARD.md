# `eval_v1` 数据卡

## 用途与边界

`eval_v1.json` 是论文追踪项目的脱敏、冻结评测证据。它用于复现样本范围、旧标签、旧 LLM 打分以及 dev/test 拆分，不是全量 arXiv 的完整真值，也不是金标基准。

- Schema：`paper-tracker.eval-snapshot/1.0.0`
- Dataset version：`eval_v1`
- 冻结记录数：210
- 内容范围：arXiv ID、标题、摘要、旧判定、anchor 标识、旧 score/reason 与固定 split
- 不包含：用户兴趣文本、liked/disliked 原文、反馈日期、反馈方向、缓存 namespace、本地路径和密钥

这里的“脱敏”指移除个人 profile 和本地运行元数据。arXiv ID、标题和摘要仍是可识别的公开论文信息。

## 数据来源与标签质量

原始输入是本地运行态文件 `papers_store.json`、`labels.json`、`profile.json` 和 legacy `scores_cache.json`。这些文件受 `.gitignore` 保护，公开仓库以冻结快照而非原始个人 profile 作为证据。

`label` 是历史 **LLM-as-a-judge 银标签**：

- `rel`：25 条
- `irrel`：79 条
- 未标注：106 条

历史流程没有保存“每条标签由谁、何时、按什么规则人工复核”的追溯记录，因此不能将这 104 条称为人工金标，也不能据此推断 106 条未标注论文为不相关。适合的用法是对已判定样本报告 judged metrics，同时报告 judgment coverage。

`score` 和 `reason` 来自同一份 legacy LLM 相关性缓存，210 条均有值，分数范围为 0–10。旧缓存未在每条记录中完整保存模型、Prompt 和解析 schema 版本，所以这些分数只能视为“冻结的历史输出”，不能与新 Prompt 产生的缓存混合。导出器遇到新缓存 schema 会直接拒绝导出。

## 拆分规则

拆分是确定性的，不依赖 JSON 输入顺序：

1. `profile.feedback` 引用的 3 篇论文优先保留为 `anchor`，避免个人偏好 anchor 泄漏到调参或最终评测。
2. 其余已判定记录分别在 `rel` / `irrel` 类内按 `SHA256("{seed}:{label}:{paper_id}")` 排序。
3. seed 为 `20260717`，dev 比例为 `0.7`；每类样本充足时保证 dev/test 都包含该类。
4. 没有 `rel` / `irrel` 判定且不是 anchor 的记录进入 `unjudged`。

| split | rel | irrel | label 为 null | 总数 |
|---|---:|---:|---:|---:|
| anchor | 1 | 1 | 1 | 3 |
| dev | 17 | 55 | 0 | 72 |
| test | 7 | 23 | 0 | 30 |
| unjudged | 0 | 0 | 105 | 105 |

一条 anchor 没有历史银标签。因为 anchor 保留的优先级高于未标注分组，所以全局 `label=null` 计数是 106，而 `unjudged` split 是 105；这不是计数错误。

## 记录 schema

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | arXiv 记录 ID，全集唯一 |
| `title` | string | arXiv 标题 |
| `summary` | string | arXiv 摘要；不是 PDF 全文 |
| `label` | `"rel"` / `"irrel"` / `null` | legacy LLM-as-a-judge 银标签 |
| `is_anchor` | boolean | 是否出现在历史 `profile.feedback` 中；不暴露反馈方向 |
| `score` | number / `null` | legacy LLM 相关性分数，0–10 |
| `reason` | string / `null` | 与 legacy score 对应的历史理由 |
| `split` | `anchor` / `dev` / `test` / `unjudged` | 冻结分组 |

## 完整性与复现

公开仓库不需要 API key，也不需要本地运行态 JSON，即可校验 schema、计数、split 不变式和 records hash：

```bash
python3 scripts/export_eval_snapshot.py --verify-only
```

预期核心输出：

```text
records: 210
records_sha256: 9adc55b6d6ec5fec16a7947c2d90925a9bb8cffc54364e68234147406a0cf60a
file_sha256: 3d115d445cb61b5093076588fafb5d76b649b9907da46637d5048a7fc91328d7
```

`records_sha256` 对 `records` 做 UTF-8、键排序、无额外空白的 canonical JSON SHA-256；`file_sha256` 则对完整 `eval_v1.json` 字节计算。

如果拥有导出时的本地运行态文件，可原子重新生成：

```bash
python3 scripts/export_eval_snapshot.py
```

或者只检查本地源数据能否精确重建已提交快照，不写文件：

```bash
python3 scripts/export_eval_snapshot.py --check
```

导出时三个非 profile 源文件的 SHA-256 为：

- `papers_store.json`：`ad2b1cb7927a3773545d57c602034ddd67e8b6276892bb876faf896b1531ae3d`
- `labels.json`：`4bb4f124a396d5a064b335a2907d7e444286febd1daded0537982221afb33d00`
- legacy `scores_cache.json`：`8f8c73e998dbd5b6d7432e1f3bace1904baee20b629b764131aa905422600136`

profile 原文件的 hash 不公布，以避免对个人偏好文本做字典猜测。导出器测试可用以下命令独立运行：

```bash
pytest -q test_export_eval_snapshot.py
```

## 离线排序复核

公开快照可直接走与生产相同的 `BM25 + FastEmbed -> RRF -> top-30 -> LLM 分数精排` 漏斗，
但最后一步读取冻结的 legacy 分数，不调用 DeepSeek：

```bash
python eval.py --snapshot eval_data/eval_v1.json
```

默认评估 `test`：从候选池排除 `anchor` 和 `dev`，保留 105 条 `unjudged` 作为检索干扰项；
也可显式传 `--split dev` 或 `--split all`。输出使用 judged P/R，并始终同时展示 top-K
judgment coverage。首次运行 FastEmbed 仍需下载约 0.22–0.24 GB 模型，因此本快照冻结的是
输入、split 和 legacy LLM 输出，不承诺不同模型工件或运行环境下的向量排序字节级一致。
一次带环境说明的实际输出记录在 [`eval_v1_reference.md`](eval_v1_reference.md)。

## 使用限制

- 不要把未标注记录当作负样本。
- 不要在 dev 和 test 合并后调参再报 test 结果。
- `anchor` 应从通用排序评测中排除；它可用于单独的偏好/记忆情景测试。
- 该快照只冻结评测输入和历史分数，不代表一次新模型调用，也没有冻结新版端到端指标。
- 样例速报 `../examples/report_sample.md` 仅根据标题和 arXiv 摘要生成，未读取 PDF 全文。
