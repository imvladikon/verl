"""Deterministic response-derived reward for the tiny GRPO lifecycle."""

from __future__ import annotations

import hashlib


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict[str, float]:
    """Return bounded response-dependent variation for a nonzero policy update."""
    del ground_truth, extra_info, kwargs
    if data_source != "glm53_flash_smoke":
        raise ValueError(f"Unexpected data source: {data_source!r}")
    digest = hashlib.blake2b(solution_str.encode("utf-8"), digest_size=8).digest()
    score = int.from_bytes(digest, "big") / float((1 << 64) - 1)
    return {"score": score, "response_hash_reward": score}
