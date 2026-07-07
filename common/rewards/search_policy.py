"""Reward shaping helpers for document-search QA rollouts."""

from __future__ import annotations


def apply_no_search_policy(
    reward: float,
    *,
    searched: bool,
    require_search: bool,
    no_search_penalty: float,
) -> float:
    """Prevent correct guesses from becoming good trajectories.

    The exam setting expects the model to consult `/data/docs` before answering.
    If a rollout jumps straight to `\boxed{...}`, answer-only reward can train the
    model to guess from priors. This helper keeps malformed answers negative as
    they already are, but caps any non-negative final reward at a small penalty
    until at least one search observation has been received.
    """
    if not require_search or searched:
        return reward
    if reward < 0:
        return reward
    return min(reward, no_search_penalty)
