# 论文追踪速读 Agent

> 给定研究兴趣，每天自动抓取 arxiv 新论文 → 混合检索 + LLM 打分筛出最相关的几篇 → 多步精读生成中文速报；支持论文库 RAG 问答与偏好反馈记忆。全流程用 LangGraph 编排，含"相关不足→自主反思→调整重抓"的 agent 决策闭环，并附一套去泄漏的离线评测。

面向场景：AI 方向学生/研究者每天面对数百篇 arxiv 新论文，人工筛选成本高。本项目把"让 LLM 直接筛论文"做成**可量化、防瞎编、越用越准**的推荐流水线。

---

## 架构

```
              研究兴趣画像(关键词 + 反馈历史)
                          │
   ┌──────────────── LangGraph StateGraph ────────────────┐
   │                                                       │
 [fetch] 抓 arxiv 新论文(按分类, entry_id 去重)            │
   │                                                       │
 [rank]  混合检索召回: 向量(embedding) + 关键词(BM25)       │
   │      ──RRF 融合──▶ LLM 逐篇打分 + 一致性校验           │
   │      ──(可选)向量空间偏好重排──▶ top-K                 │
   │                                                       │
 [decide] 相关篇数够? ──否──▶ [reflect] LLM 诊断原因,       │
   │            │              自主选 widen/refocus/stop ──┘
   │            └──是──▶
 [report] top-K 多步精读(贡献/方法/相关点) → 中文速报
   │
   ├─▶ 论文入本地库  ──▶ [RAG 问答] 检索片段→基于片段作答+标出处
   └─▶ 偏好反馈(有用/没用) ──▶ 更新画像(同时作为评测标签)
```

**为什么算 agent 而非脚本**：`reflect` 节点在运行时根据"相关论文不足"这一观察，让 LLM **自主诊断**并在 `widen`(扩大 arxiv 分类) / `refocus`(收窄兴趣 query) / `stop`(今日无好论文) 三种动作间动态路由，而非跑固定 DAG。这是"运行时自主选择下一步动作"的 agent 本质。

---

## 核心技术点

| 模块 | 技术 | 说明 |
|---|---|---|
| 编排 | **LangGraph StateGraph** | 节点 + 共享 State + 条件边，支撑反思-重抓循环 |
| 召回 | **混合检索 (embedding + BM25 + RRF)** | 语义召回 + 关键词召回，RRF 融合，单路短板互补 |
| 精排 | **LLM 打分 + 一致性校验** | 逐篇打分；回填编号检测漏条/错位/越界，检出即重打 |
| 记忆 | **向量空间偏好重排** | 与 liked 相似加分、disliked 减分；不污染原始 query |
| 问答 | **RAG (检索增强生成)** | 检索片段→基于片段作答→标出处→相似度阈值兜底防瞎编 |
| 评测 | **离线评测 (Precision/Recall@K)** | 银标签+人工校准、随机基线、anchor 排除防泄漏 |
| 安全 | **Prompt 注入加固** | 外部文本用分隔符包裹 + 声明"数据非指令" |

---

## 评测结果

数据集：180 篇论文库，104 篇人工校准标注（25 相关 / 79 不相关），测试集 177 篇。

| 方法 | Precision@10 | Recall@10 |
|---|---|---|
| 随机基线 | 0.20 | 0.08 |
| **混合检索** | 1.00 | **0.40** |

- **混合检索 Recall@10 = 0.40，为随机基线 0.08 的 5 倍**（25 篇相关里 top-10 捞回 10 篇，顶到窗口上限）。
- **LLM 打分与人工标签一致率 0.89**（102 篇），验证相关性阈值可靠。
- 随机抽 18 篇独立标注、0 篇相关 → 相关论文基率约 10%，反衬检索"大海捞针"价值。

**方法学诚实声明**：标注为 LLM-as-a-judge 银标签经人工抽检校准；P@10=1.00 因相关标注集中在检索高排名论文存在选择偏差，故以受偏差影响更小的 **Recall@10** 为主指标。偏好记忆在当前 3 条反馈下无定量增益，定位为定性功能（tune 扫描显示偏好权重过高反而有害，已据此从 5.0 调至 3.0）。

---

## 安装与运行

```bash
# 1. 环境
python -m venv .venv
.venv\Scripts\activate          # Windows;  Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置 LLM(DeepSeek, OpenAI 兼容接口)
echo DEEPSEEK_API_KEY=sk-xxxx > .env

# 3. 跑一次:出当天中文速报
python graph.py                 # 产物: 速报_日期.md

# 4. 反馈记忆: 对速报第 n 篇点有用/没用
python feedback.py 1 up
python feedback.py 3 down

# 5. RAG 问答
python qa.py "论文库里有哪些关于强化学习的研究?"

# 6. 评测
python label.py                 # 看标注清单
python label.py rel 1 5 8       # 批量标相关
python eval.py                  # 出 Precision/Recall@10 + 随机基线 + 一致率
python tune.py                  # 扫描偏好权重取最优

# 7. 单元测试
pytest test_retrieval.py
```

---

## 文件说明

| 文件 | 职责 |
|---|---|
| `graph.py` | **主入口**。LangGraph 编排：fetch→rank→(reflect)→report |
| `daily.py` | arxiv 抓取、去重、统一 LLM 调用封装(超时+重试) |
| `retrieval.py` | 混合检索、LLM 打分+一致性校验、偏好重排、多步精读 |
| `memory.py` | 偏好画像读写、反馈记录、评测锚点 |
| `qa.py` | RAG 问答 + 本地论文库 |
| `feedback.py` | CLI：对速报打"有用/没用"标签 |
| `label.py` / `eval.py` / `tune.py` | 评测：标注 / 算指标 / 调参 |
| `review_list.py` / `random_check.py` | 评测辅助：生成待核对清单 / 随机抽样核对 |
| `test_retrieval.py` | 单元测试(rrf / tokenize / 一致性校验 / metric) |

---

## 设计选型与取舍

- **为什么混合检索而非单一召回**：embedding 懂语义但会被字面词误导（如标题含 "memory" 实则讲 RL 内存优化），BM25 懂关键词但不懂同义，二者用 RRF 融合互补。
- **为什么偏好重排放在向量空间**：不把历史论文标题塞回检索 query——因为 embedding 不懂否定（"不喜欢"会被当成"相关"），且长文本注入会带偏 query；改为打分后按相似度加减分，方向明确且不污染召回。
- **为什么用 LangGraph**：reflect 的"诊断→重抓"是带条件分支与回环的控制流，状态图比顺序函数更清晰，也是 agent 工程的高频实践。
- **为什么 LLM 选 DeepSeek**：中文友好、OpenAI 兼容接口可零成本迁移、价格低适合大量打分；打分 `temperature=0` + 结果缓存保证评测可复现。
- **为什么砍掉 FastAPI/Docker 部署**：当前定位是"可讲透的最小可用闭环"，部署是一天可补的工程外壳，优先把 agent 决策与评测做扎实。
- **评测为什么用银标签 + 人工校准**：无人工金标准时，先用 LLM 代标快速起步，再人工抽检校准并随机采样去偏，诚实交代局限优于伪造金标准。
