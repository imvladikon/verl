#!/usr/bin/env python3
"""Create deterministic tiny SFT and GRPO datasets for GLM-5.3-Flash."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SFT_EXAMPLES = (
    ("What is 2 + 3?", "5"),
    ("Reply with the word blue.", "blue"),
    ("What is 7 - 4?", "3"),
    ("Reply with the word small.", "small"),
)

RL_PROMPTS = ("alpha", "beta", "gamma", "delta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sft_rows = [
        {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
        }
        for prompt, answer in SFT_EXAMPLES
    ]
    rl_rows = [
        {
            "data_source": "glm53_flash_smoke",
            "prompt": [{"role": "user", "content": f"Continue with a short answer: {value}"}],
            "ability": "lifecycle_smoke",
            "reward_model": {"style": "rule", "ground_truth": "unused"},
            "extra_info": {"index": index},
        }
        for index, value in enumerate(RL_PROMPTS)
    ]

    sft_path = args.output_dir / "sft.parquet"
    rl_path = args.output_dir / "rl.parquet"
    pq.write_table(pa.Table.from_pylist(sft_rows), sft_path)
    pq.write_table(pa.Table.from_pylist(rl_rows), rl_path)
    print(f"wrote {len(sft_rows)} SFT rows to {sft_path}")
    print(f"wrote {len(rl_rows)} RL rows to {rl_path}")


if __name__ == "__main__":
    main()
