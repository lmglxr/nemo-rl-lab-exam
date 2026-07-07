# QA-RL 多轮检索 Agent 实验需求文档

## 1. 目标

在 NeMo Lab 平台上使用 GRPO 微调 `Qwen/Qwen3.5-9B-Base`，让模型在技术培训考试题上学会：

1. 根据题目判断是否需要检索资料。
2. 用多轮工具调用检索 `/data/docs` 下的 markdown 文档。
3. 基于检索结果作答。
4. 始终把最终答案写入 `\boxed{...}`，以便平台判分。

主评分指标是 `validation/accuracy`。

## 2. 输入与资源

数据集路径：

- 集群：`/data/datasets/qa_rl`
- 由平台注入：`QA_RL_DATA_DIR`
- 数据格式：jsonl，每行至少包含 `query` 和 `expected_answer`

文档检索路径：

- 集群：`/data/docs`
- 可由环境变量覆盖：`QA_DOCS_DIR`

题型：

- 单选：`[single] B`
- 多选：`[multiple] A,C,D`
- 判断：`[bool] A`
- 填空：`[fill] answer1 ||| answer2`
- 简答：`[short] keyword1 ||| keyword2`

## 3. 交互协议

模型和环境采用文本协议通信。

检索动作：

```text
<search>关键词</search>
```

环境返回：

```text
[search results for: 关键词]
1. 文档路径 (score=...)
命中的片段...
```

最终作答：

```text
\boxed{答案}
```

约束：

- 每题最多检索 `max_searches` 次，默认 3 次。
- 检索时不要同时给最终答案；环境优先把纯检索动作视为工具调用。
- 一旦输出 `\boxed{...}`，环境会判分并结束该题。
- 未输出 `\boxed{...}` 会被既有 QA reward 逻辑判为格式扣分。

## 4. 已实现代码结构

新增环境：

- `common/environments/qa_search_env.py`

职责：

- 解析 `<search>...</search>`
- 在 markdown 文档中做本地关键词检索
- 返回 top-k 片段给模型
- 检测 `\boxed{...}` 后调用原有 QA reward
- 输出训练指标：平均 reward、满分率、格式扣分率

新增实验入口：

- `experiments/grpo_qwen3.5-9b_qa-rl-agent_lmg/run.py`

职责：

- 读取 `train.jsonl` / `val.jsonl`
- 构造多轮 Agent prompt
- 创建 `QASearchEnv`
- 调用 NeMo-RL `setup()` 和 `grpo_train()`

配置文件：

- `experiments/grpo_qwen3.5-9b_qa-rl-agent_lmg/config.yaml`

关键字段：

- `grpo.max_rollout_turns`：多轮上限
- `policy.max_total_sequence_length`：多轮上下文长度
- `env.qa_search.cfg.docs_dir`：检索文档目录
- `env.qa_search.cfg.max_searches`：每题最大检索次数
- `env.qa_search.cfg.use_judge`：简答是否使用 LLM judge

## 5. 后续执行步骤

1. 检查配置：

```bash
uv run lab validate grpo_qwen3.5-9b_qa-rl-agent_lmg
```

2. 提交训练：

```bash
uv run lab submit grpo_qwen3.5-9b_qa-rl-agent_lmg
```

3. 查看日志：

```bash
uv run lab logs <job_id>
```

4. 在控制台观察：

- `validation/accuracy`
- 验证样本中的检索轨迹
- `qa_search_mean_reward`
- `qa_search_perfect_rate`
- 是否有 OOM 或 rollout 截断

## 6. 调参建议

优先保证跑通：

- `max_num_steps: 50`
- `max_val_samples: 32`
- `num_prompts_per_step: 2` 或 4
- `num_generations_per_prompt: 4` 或 8

如果上下文被截断：

- 增大 `policy.max_total_sequence_length`
- 降低 `env.qa_search.cfg.top_k`
- 降低 `env.qa_search.cfg.snippet_chars`
- 降低 `grpo.max_rollout_turns`

如果显存不足：

- 降低 `gpu_memory_utilization`
- 降低 `max_total_sequence_length`
- 降低 `num_generations_per_prompt`
- 降低 `train_global_batch_size`

如果模型很少检索：

- 强化 `data.system_prompt` 中的检索指令。
- 提高 `max_rollout_turns`。
- 在验证样本日志里检查模型是否学会输出严格的 `<search>...</search>`。
