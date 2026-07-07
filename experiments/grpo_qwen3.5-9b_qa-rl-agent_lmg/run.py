#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import pprint
import sys
from typing import Any

from omegaconf import OmegaConf
from torch.utils.data import Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

from common.environments.qa_search_env import QASearchEnv

# task_name 必须和 config.yaml 里的 env.qa_search 对齐。
# NeMo-RL 会根据每条 DatumSpec 的 task_name，把 rollout 发给对应环境。
TASK_NAME = "qa_search"


def parse_args():
    parser = argparse.ArgumentParser(description="QA multi-turn search-agent GRPO")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    args, overrides = parser.parse_known_args()
    return args, overrides


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class QAAgentJsonlDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        input_key: str,
        output_key: str,
        system_prompt: str | None = None,
    ):
        self.rows = _read_jsonl(path)
        self.tokenizer = tokenizer
        self.input_key = input_key
        self.output_key = output_key
        self.system_prompt = system_prompt or (
            # 这个 system prompt 是协议的“语言层说明书”。
            # 环境只会机械识别 <search>...</search> 和 \boxed{}；
            # 如果不在 prompt 里告诉模型这些格式，模型很可能自由发挥，导致环境无法分支。
            "You are a technical exam QA agent. You may search local markdown documents "
            "before answering. To search, output exactly <search>keywords</search>. "
            "After receiving search results, continue reasoning or search again. "
            "When ready, put the final answer in \\boxed{...}. Do not omit \\boxed{}."
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> DatumSpec:
        row = self.rows[idx]
        query = str(row[self.input_key])
        expected = str(row[self.output_key])

        # 这里把 system + user 拼成 chat template 后，作为 NeMo-RL 的初始 message_log。
        # 后续每一轮模型输出 assistant，环境输出 environment observation，
        # NeMo-RL 会自动把它们接在这段日志后面，形成多轮轨迹。
        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, add_special_tokens=False
        ).strip()
        token_ids = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]

        message_log: LLMMessageLogType = [
            {"role": "user", "content": prompt_text, "token_ids": token_ids}
        ]
        return {
            "message_log": message_log,
            "length": len(token_ids),
            # extra_env_info 不参与模型输入，但会传给环境 step()。
            # 为什么不把 expected_answer 放进 prompt：那会泄漏答案；
            # 它只能作为 reward 的金标存在于环境侧。
            "extra_env_info": {
                "expected_answer": expected,
                "query": query,
                # search_count 从 0 开始，由环境每次成功检索后递增。
                # 这样即使 message_log 很长，也不用回放历史来统计检索次数。
                "search_count": 0,
            },
            "loss_multiplier": 1.0,
            "idx": idx,
            "task_name": TASK_NAME,
        }


def main():
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if not args.config:
        args.config = os.path.join(THIS_DIR, "config.yaml")

    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    if overrides:
        print(f"CLI overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    # MasterConfig 会做结构校验。这里尽早把 yaml 转成 NeMo-RL 期望的配置对象，
    # 可以在训练启动前发现字段名写错、层级不对等问题。
    config: MasterConfig = MasterConfig(**config)
    print("Final config:")
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"Log dir: {config.logger['log_dir']}")

    init_ray()
    set_seed(config.grpo["seed"])

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    # 生成配置需要结合 tokenizer 补齐 stop token / eos 等字段。
    # 这一步保持和官方示例一致，避免自定义 run.py 漏掉生成侧默认值。
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )

    data_cfg: dict[str, Any] = config.data
    # 数据路径优先走平台注入的 QA_RL_DATA_DIR；这样正式提交时能使用集群共享题库。
    # config.data_dir 是兜底，便于本地或容器内直接调试。
    data_dir = os.environ.get("QA_RL_DATA_DIR") or data_cfg.get("data_dir")
    if not data_dir:
        raise SystemExit("Set QA_RL_DATA_DIR or data.data_dir to the qa_rl dataset directory.")

    input_key = data_cfg.get("input_key", "query")
    output_key = data_cfg.get("output_key", "expected_answer")
    system_prompt = data_cfg.get("system_prompt") or None

    train_dataset = QAAgentJsonlDataset(
        os.path.join(data_dir, "train.jsonl"),
        tokenizer,
        input_key,
        output_key,
        system_prompt,
    )
    val_dataset = QAAgentJsonlDataset(
        os.path.join(data_dir, "val.jsonl"),
        tokenizer,
        input_key,
        output_key,
        system_prompt,
    )
    print(f"Train samples: {len(train_dataset)}, val samples: {len(val_dataset)}")

    env_cfg = config.env[TASK_NAME]["cfg"]
    # 环境作为 Ray actor 启动，num_gpus=0 是刻意的：
    # 检索和判分主要是 CPU/文本逻辑，不应该占用唯一的 H100。
    env = QASearchEnv.options(num_gpus=0).remote(cfg=dict(env_cfg))
    # train 和 val 共用同一种环境协议；区别只在 dataloader 输入的数据 split。
    task_to_env = {TASK_NAME: env}

    (
        policy,
        policy_generation,
        _nemo_gym,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    # grpo_train 内部会循环：
    # prompt -> policy_generation 生成 assistant -> QASearchEnv.step()
    # -> 若 terminated=False，把检索 observation 放回上下文继续生成
    # -> 若 terminated=True，用最终 reward 更新策略。
    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
