# Agent reflect synthetic benchmark v1

## 用途

这份小型数据集只回答一个工程问题：`graph.reflect_node` 面对“相关论文太少”时，能否在 `widen` / `refocus` / `stop` 受控动作空间内输出可路由的决策，并在非法输出时 fail closed。它补充排序评测，不替代排序评测。

## 构成

- 12 个脱敏合成场景，不含真实用户、真实论文标题或个人兴趣画像。
- 8 个 `policy` 场景：分类过窄、兴趣过宽、无合适论文、空候选、标题 Prompt Injection、分类白名单过滤、分类去重和 `refocus -> rank`。
- 4 个 `contract` 场景：仅返回已有分类、非法 action、非纯 JSON 和过短 refocus query。

每个场景包含当前 query、当前 arXiv 分类、合成的 top candidates（title + score）、期望 action/route、可机械检查的分类或 query 约束、rationale 与冻结决策输出。

## 冻结输出的来源

v1 的 `frozen_decision` 是人工编写的可重现 fixture，不是对 DeepSeek 当前或未来版本的声明。`policy` fixture 表示预期的受控决策；`contract` fixture 故意注入错误输出，用于验证生产校验和 fail-closed 路由。因此，离线结果展示的是“整条决策管道对这组 fixture 的行为”，不是模型准确率。

如要观察当前 DeepSeek 响应，显式运行：

```bash
python agent_eval.py --live
```

`--live` 每个场景最多调用一次生产 `daily.chat`，不会回写本数据集，也不会写 `seen.json` / `profile.json` / `scores_cache.json` / 速报。如果要将某次线上结果变成新的冻结基线，需要单独审核、记录模型与日期，然后修订数据版本。

2026-07-17 的一次性 DeepSeek 在线 A/B 和完整限制说明见 [live_reference_2026-07-17.md](live_reference_2026-07-17.md)。该文档是开发参考，不是可决定性重现的冻结模型证据。

在 `--live` 中，`contract` 样本里故意冻结的错误 raw 也会被当前模型响应替换，所以 live 的 contract action accuracy 不能当作模型质量；比较模型时主要看 `policy_metrics`，回归 fail-closed 契约时看默认离线 `contract_metrics`。

## 指标

- **Action accuracy**：经过生产校验后的 action 是否等于场景期望。
- **Schema valid rate**：原始响应是否为 JSON object，action 在枚举内，`why` 以及 action-specific 字段类型正确。
- **Constraint pass rate**：最终 action、route 和场景声明的新分类/query 字面约束均通过。
- **Fail-closed rate**：仅以“非法/schema 错误响应”或“被生产校验拒绝的 widen/refocus”为分母，检查是否转成 `stop -> report`。若某次 live 运行没有适用样本，该指标为 N/A，不伪造 100%。
- **Average policy invocations/scenario**：默认离线为 0；`--live` 中每个场景进入一次 reflect 策略调用。这不是底层 HTTP/API attempts 计数，因为 `daily.chat` 可能在一次策略调用内部重试。
- **Static action baselines**：在 8 个 `policy` 场景上计算 always-widen 和 always-stop 的 action accuracy，用来表明受控决策是否超过“永远扩分类/永远停止”的固定策略。它只比较合成期望 action，不是最终推荐收益、Recall 或用户价值。

脚本还单独输出 `policy` 和 `contract` 分组指标，避免把手工故障注入样本误读成模型能力。实际解析、query/category 验证与 route 不在评测中复制，而是直接运行 `graph.reflect_node` 和 `graph.route_action`。

## 限制

- 这是 synthetic benchmark，不包含线上流量分布，不能证明真实业务质量。
- 标签由项目开发者根据路由契约编写，没有多标注员一致性。
- query 约束是可重现的字面检查，不是语义质量评分。
- 未评测 arXiv 抓取、RRF/LLM 排序、速报有用性、成本或长期偏好学习。
- 冻结输出不代表当前模型保证；线上结果会随模型版本、服务状态和提示词变化。

## 可复现命令

```bash
# 无 key、无网络、不写运行时数据
DEEPSEEK_API_KEY= python agent_eval.py

# 机器可读输出
DEEPSEEK_API_KEY= python agent_eval.py --json

# 单元测试
DEEPSEEK_API_KEY= python -m pytest -q test_agent_eval.py
```
