"""从 NeMo-RL message_logs 提取结构化验证样本（与 console log_parse 字段对齐）。"""
from __future__ import annotations

import re


_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)


def _role_text(message_log: list, role: str) -> str:
    parts: list[str] = []
    want = role.upper()
    for msg in message_log:
        if not isinstance(msg, dict):
            continue
        r = str(msg.get("role", "")).upper()
        if r == want:
            content = msg.get("content")
            if content is not None:
                parts.append(str(content))
    return "\n".join(parts).strip()


def _search_queries(message_log: list) -> list[str]:
    queries: list[str] = []
    for msg in message_log:
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role", "")).upper() != "ASSISTANT":
            continue
        content = str(msg.get("content", ""))
        for match in _SEARCH_RE.finditer(content):
            query = re.sub(r"\s+", " ", match.group(1)).strip()
            if query:
                queries.append(query)
    return queries


def extract_message_log_samples(
    message_logs: list,
    rewards: list[float],
    *,
    num_samples: int | None = None,
) -> tuple[list[dict], list[dict], float | None]:
    """返回 (samples, dist, avg_reward)。

    - num_samples=None（默认）：上报**整轮全量**样本，idx 用验证集原始位置（1-based），
      跨验证轮稳定，便于按题目追踪得分变化。
    - num_samples>0 且小于总数：按 reward 高/低两端采样（与 print_message_log_samples 一致），
      但 idx 仍保留样本在本轮中的原始位置，不重排。
    dist / avg_reward 始终基于**全量** rewards 计算。
    """
    if not message_logs or not rewards:
        return [], [], None
    n = len(message_logs)
    indices = list(range(n))
    if num_samples is not None and 0 < num_samples < n:
        sorted_indices = sorted(indices, key=lambda i: rewards[i], reverse=True)
        half = num_samples // 2
        picked = sorted_indices[:half] + sorted_indices[-half:]
        if num_samples % 2 == 1:
            picked.append(sorted_indices[len(sorted_indices) // 2])
        # 去重并回到原始顺序，保证 idx 稳定、可跨轮对齐
        indices = sorted(dict.fromkeys(picked))[:num_samples]

    samples: list[dict] = []
    for idx in indices:
        ml = message_logs[idx]
        reward = float(rewards[idx])
        search_queries = _search_queries(ml)
        samples.append(
            {
                "idx": idx + 1,  # 原始位置，1-based
                "reward": reward,
                "user": _role_text(ml, "user"),
                "assistant": _role_text(ml, "assistant"),
                "env": _role_text(ml, "environment") or _role_text(ml, "system"),
                # Make retrieval behavior visible in validation dashboards without
                # requiring a human to inspect the full transcript manually.
                "searched": bool(search_queries),
                "search_count": len(search_queries),
                "search_queries": search_queries,
            }
        )

    counts: dict[float, int] = {}
    for r in rewards:
        counts[r] = counts.get(r, 0) + 1
    dist = [{"reward": k, "count": v} for k, v in sorted(counts.items())]
    avg_reward = sum(rewards) / len(rewards) if rewards else None
    return samples, dist, avg_reward
