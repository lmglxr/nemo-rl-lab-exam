"""带本地 markdown 检索工具的多轮 QA 环境。

为什么单独写这个环境：
  单轮 QA 环境只能“模型答一次 -> 环境判分 -> 结束”，无法让模型先查资料。
  考试要求模型可以多次检索 /data/docs 后再作答，所以这里把“检索动作”
  和“最终作答”都放进同一个 Environment.step() 里处理。

交互协议如何流转：
  1. run.py 把题目包装成 user prompt，并在 metadata 中放入 query、expected_answer、
     search_count。
  2. 模型本轮如果输出 <search>关键词</search>，环境把它解释为工具调用。
  3. 环境在 docs_dir 中搜索 markdown，把命中的片段作为 role=environment 的
     observation 返回，同时 terminated=False，NeMo-RL 会继续下一轮生成。
  4. 模型后续可以继续 <search>...</search>，也可以输出最终答案 \boxed{...}。
  5. 一旦检测到 \boxed{...}，环境调用既有 QA reward 判分，并 terminated=True
     结束该样本。
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.rewards.qa_reward import extract_boxed
from common.rewards.search_policy import apply_no_search_policy


class QASearchMetadata(TypedDict, total=False):
    # metadata 是环境跨轮保存状态的地方。message_log 里能看到完整对话，
    # 但像“已检索几次”这种计数状态放在 metadata 更直接，也避免解析历史文本。
    expected_answer: str
    query: str
    search_count: int


# 用正则识别工具调用，而不是引入函数调用框架，是因为本考试协议本身就是纯文本。
# re.DOTALL 允许关键词跨行；IGNORECASE 容忍模型大小写不稳定。
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def _last_assistant_text(message_log: LLMMessageLogType) -> str:
    # 每次 step() 收到的是整段 message_log；环境只需要判断“模型最新一轮说了什么”。
    # 从后往前找 assistant，可以兼容多轮 observation 已经插入日志的情况。
    for msg in reversed(message_log):
        if msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()
    return ""


def _tokens(text: str) -> list[str]:
    # 这里做的是非常轻量的关键词检索，不做复杂分词。考试资料多是技术词、
    # 英文缩写、中文短语混合；保留中英文数字 token 能覆盖大多数定位需求。
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t.strip()) >= 2]


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit].rstrip() + ("..." if len(text) > limit else "")


@dataclass
class SearchHit:
    score: int
    path: str
    snippet: str


class MarkdownSearcher:
    def __init__(self, docs_dir: str, max_files: int = 2000):
        self.docs_dir = Path(docs_dir)
        self.docs: list[tuple[str, str]] = []
        if self.docs_dir.exists():
            # 在 actor 初始化时把 markdown 读入内存：/data/docs 是静态资料，
            # 训练过程中会被反复检索；预加载能避免每个 rollout turn 都扫磁盘。
            # max_files 是兜底保护，防止文档目录异常大时 actor 启动过慢。
            for path in list(self.docs_dir.rglob("*.md"))[:max_files]:
                try:
                    rel = str(path.relative_to(self.docs_dir))
                    self.docs.append((rel, path.read_text(encoding="utf-8", errors="ignore")))
                except OSError:
                    continue

    def search(self, query: str, top_k: int, snippet_chars: int) -> list[SearchHit]:
        terms = _tokens(query)
        if not terms:
            return []

        hits: list[SearchHit] = []
        for rel, text in self.docs:
            lowered = text.lower()
            # 简单按关键词出现次数打分。这里故意不用向量库/BM25 依赖：
            # 集群考试环境越少外部依赖越稳，且题库资料通常是“题干关键词 -> 文档片段”
            # 的定位问题，词频排序已经能给模型提供可用证据。
            score = sum(lowered.count(term) for term in terms)
            if score <= 0:
                continue
            first_positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
            # 片段从第一个命中词附近截取，而不是返回整篇文档。
            # 原因是多轮 RL 的上下文长度很宝贵；返回太长会挤掉题目和最终答案。
            start = max(0, min(first_positions) - snippet_chars // 3) if first_positions else 0
            snippet = _clip(text[start : start + snippet_chars], snippet_chars)
            hits.append(SearchHit(score=score, path=rel, snippet=snippet))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]


@ray.remote  # pragma: no cover
class QASearchEnv(EnvironmentInterface[QASearchMetadata]):
    """把“检索工具”和“答案判分”合并到一个 Ray 环境 actor 中。"""

    def __init__(self, cfg: Optional[dict[str, Any]] = None):
        self.cfg = cfg or {}
        # 路径优先级：环境变量 > config > 默认集群路径。
        # 这样本地调试可临时设置 QA_DOCS_DIR，正式提交则走 /data/docs。
        self.docs_dir = os.environ.get("QA_DOCS_DIR") or self.cfg.get("docs_dir") or "/data/docs"
        self.max_searches = int(self.cfg.get("max_searches", 3))
        self.top_k = int(self.cfg.get("top_k", 4))
        self.snippet_chars = int(self.cfg.get("snippet_chars", 700))
        self.use_judge = bool(self.cfg.get("use_judge", True))
        self.require_search_before_answer = bool(
            self.cfg.get("require_search_before_answer", True)
        )
        self.no_search_penalty = float(self.cfg.get("no_search_penalty", -0.2))
        self.mixed_action_penalty = float(self.cfg.get("mixed_action_penalty", -0.2))
        self.searcher = MarkdownSearcher(self.docs_dir)

        # 复用原来 qa_env 的 reward，而不是重写判题逻辑。
        # 这样单选/多选/判断/填空/简答的评分口径和 baseline 一致，
        # 实验差异主要来自“是否能检索”，便于对比 accuracy。
        if self.use_judge:
            from common.rewards.qa_judge_reward import qa_judge_reward_fn

            self._reward_fn = qa_judge_reward_fn
        else:
            from common.rewards.qa_reward import qa_rule_reward_fn

            self._reward_fn = qa_rule_reward_fn

    def _search_observation(self, query: str) -> str:
        hits = self.searcher.search(query, self.top_k, self.snippet_chars)
        if not hits:
            # 没搜到也返回一条明确 observation，而不是静默失败。
            # 这能训练模型在资料不足时换关键词或直接根据已有知识作答。
            return f"[search results]\nNo markdown hits for: {query}"
        lines = [f"[search results for: {query}]"]
        for i, hit in enumerate(hits, start=1):
            lines.append(f"{i}. {hit.path} (score={hit.score})\n{hit.snippet}")
        return "\n\n".join(lines)

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[QASearchMetadata],
    ) -> EnvironmentReturn[QASearchMetadata]:
        observations = []
        rewards = []
        terminateds = []
        answers = []
        next_metadata: list[QASearchMetadata | None] = []

        for log, meta in zip(message_log_batch, metadata, strict=False):
            completion = _last_assistant_text(log)
            query = str(meta.get("query", ""))
            expected = str(meta.get("expected_answer", ""))
            search_count = int(meta.get("search_count", 0))
            answers.append(expected)

            search_match = _SEARCH_RE.search(completion)
            has_boxed_answer = extract_boxed(completion) is not None
            searched = search_count > 0

            if (
                self.require_search_before_answer
                and not searched
                and search_match
                and has_boxed_answer
            ):
                # A single assistant turn must be either a tool action or a final
                # answer.  Penalizing mixed output teaches the policy to emit a
                # clean `<search>...</search>` first, wait for evidence, and only
                # then produce `\boxed{...}` in a later turn.
                observations.append({
                    "role": "environment",
                    "content": (
                        "Invalid mixed action: search and final answer were emitted "
                        "in the same turn. First search, then answer after the "
                        "search results."
                    ),
                })
                rewards.append(self.mixed_action_penalty)
                terminateds.append(True)
                next_metadata.append(None)
                continue

            if search_match and not has_boxed_answer and search_count < self.max_searches:
                # 协议分支 1：纯检索动作。
                # 这里 reward 给 0，而不是奖励检索本身，是为了避免模型学成“为了拿分不断检索”。
                # 真正的正负反馈只来自最终答案；检索只是帮助模型获得证据的中间动作。
                search_query = _clip(search_match.group(1), 160)
                observations.append({"role": "environment", "content": self._search_observation(search_query)})
                rewards.append(0.0)
                # terminated=False 是多轮协议的关键：它告诉 NeMo-RL 这题还没结束，
                # 要把 observation 追加进 message_log，再让模型生成下一轮。
                terminateds.append(False)
                next_metadata.append({**meta, "search_count": search_count + 1})
                continue

            if search_match and not has_boxed_answer and search_count >= self.max_searches:
                # 协议分支 2：超过检索次数。
                # 不直接结束，是因为模型仍有机会基于已有检索结果给出 \boxed{}。
                # 给一个很小的负分，作用是提示“继续查没有收益”，但不压过最终作答奖励。
                observations.append({
                    "role": "environment",
                    "content": "Search limit reached. Provide the final answer in \\boxed{...}.",
                })
                rewards.append(-0.1)
                terminateds.append(False)
                next_metadata.append(meta)
                continue

            # 协议分支 3：最终作答或无效动作。
            # 只要没有走检索分支，就交给 QA reward 判分；如果模型没写 \boxed{}，
            # 原有 reward 会给格式扣分，这比在环境里另写一套规则更一致。
            raw_reward = float(self._reward_fn([query], [completion], [expected])[0])
            reward = apply_no_search_policy(
                raw_reward,
                searched=searched,
                require_search=self.require_search_before_answer,
                no_search_penalty=self.no_search_penalty,
            )
            note = ""
            if reward != raw_reward:
                note = (
                    " (final answer submitted before any search; "
                    "reward capped by require_search_before_answer)"
                )
            observations.append({
                "role": "environment",
                "content": f"score: {reward:.3f}{note}",
            })
            rewards.append(reward)
            # 最终答案判分后必须结束，否则模型可能在已得分后继续生成，
            # 造成 rollout 轨迹和 reward 归因混乱。
            terminateds.append(True)
            next_metadata.append(None)

        return EnvironmentReturn(
            # observations 会被 NeMo-RL 追加到对话中。检索时它是资料片段；
            # 结束时它只是分数提示，主要用于日志可读性。
            observations=observations,
            metadata=next_metadata,
            next_stop_strings=[None] * len(observations),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=answers,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        # 这些指标是为了在控制台快速判断训练方向：
        # mean_reward 看整体趋势，perfect_rate 近似看满分比例，
        # format_penalty_rate 用来发现模型是否忘记输出 \boxed{}。
        rewards = batch.get("total_reward", torch.tensor([0.0] * len(batch["idx"]))).float()
        if len(rewards) == 0:
            return batch, {}
        metrics = {
            "qa_search_mean_reward": rewards.mean().item(),
            "qa_search_perfect_rate": (rewards >= 1.0).float().mean().item(),
            "qa_search_format_penalty_rate": (rewards < 0).float().mean().item(),
        }
        return batch, metrics
