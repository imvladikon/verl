from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from build_token_bucket_mixture import normalize_buckets, partition_rows  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return list(range(int(messages[-1]["content"])))


def _row(example_id: str, tokens: int) -> dict:
    return {
        "id": example_id,
        "system": "system",
        "prompt": "prompt",
        "response": str(tokens),
    }


def test_bucket_boundaries_are_exact_and_sorted() -> None:
    rows = [_row("a", 256), _row("b", 257), _row("c", 384), _row("d", 706)]
    partition, lengths = partition_rows(rows, FakeTokenizer(), [768, 256, 384, 384])
    assert normalize_buckets([768, 256, 384, 384]) == (256, 384, 768)
    assert [[row["id"] for row in partition[bucket]] for bucket in partition] == [
        ["a"],
        ["b", "c"],
        ["d"],
    ]
    assert lengths == {256: [256], 384: [257, 384], 768: [706]}


def test_example_above_largest_bucket_fails_closed() -> None:
    try:
        partition_rows([_row("too-long", 769)], FakeTokenizer(), [256, 384, 768])
    except ValueError as error:
        assert "769 tokens exceed largest bucket 768" in str(error)
    else:  # pragma: no cover
        raise AssertionError("oversized example was accepted")
