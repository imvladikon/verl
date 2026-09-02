from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from audit_quality_tokens import measure, summarize, token_count  # noqa: E402
from build_quality_dataset import validate_rows  # noqa: E402
from generate_targeted_quality_data import generate_rows  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, **_kwargs):
        length = sum(len(message["content"].split()) + 2 for message in messages)
        input_ids = list(range(length + int(add_generation_prompt)))
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(len(text.split())))}


def test_summarize_uses_nearest_rank_percentiles() -> None:
    assert summarize(list(range(1, 101))) == {
        "count": 100,
        "min": 1,
        "mean": 50.5,
        "p50": 50,
        "p90": 90,
        "p95": 95,
        "p99": 99,
        "max": 100,
        "total": 5050,
    }


def test_token_count_handles_flat_batched_and_mapping_outputs() -> None:
    assert token_count([1, 2, 3]) == 3
    assert token_count([[1, 2, 3]]) == 3
    assert token_count({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}) == 3


def test_measure_covers_every_targeted_row_by_split_and_family() -> None:
    result = measure(validate_rows(generate_rows()), FakeTokenizer())
    assert result["overall"]["full_chat"]["count"] == 720
    assert sum(group["full_chat"]["count"] for group in result["by_split"].values()) == 720
    assert sum(group["full_chat"]["count"] for group in result["by_family"].values()) == 720
    assert result["overall"]["full_chat"]["max"] > result["overall"]["prompt_chat"]["min"]
