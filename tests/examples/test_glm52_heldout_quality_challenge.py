from __future__ import annotations

import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "glm52_lora"
sys.path.insert(0, str(EXAMPLE))

from build_blind_quality_review import read_contracts  # noqa: E402
from build_heldout_quality_challenge import (  # noqa: E402
    EXPECTED_FAMILIES,
    MINIMUM_SLICE_CLUSTERS,
    MINIMUM_SLICE_ROWS,
    generate_rows,
    validate_challenge,
)
from build_quality_dataset import validate_rows  # noqa: E402
from generate_full_quality_outputs_sglang import (  # noqa: E402
    derive_han_evaluation_mode,
    evaluation_cluster_id,
    request_input_han_count,
    request_messages,
)


def test_generated_challenge_is_held_out_and_meets_every_slice_minimum() -> None:
    rows = validate_rows(generate_rows())
    validation = validate_challenge(rows)

    assert len(rows) == 140
    assert validation["split_counts"] == {"test": 70, "validation": 70}
    assert {row["split"] for row in rows} == {"validation", "test"}
    assert not any(row["use_for_constraint_rl_smoke"] for row in rows)
    assert not any("semantic_score" in row for row in rows)
    assert validation["target_prompt_leak_count"] == 0
    for split in ("validation", "test"):
        assert set(validation["coverage"][split]) == set(EXPECTED_FAMILIES)
        for detail in validation["coverage"][split].values():
            assert detail == {
                "row_count": MINIMUM_SLICE_ROWS,
                "cluster_count": 10,
                "status": "PASS",
            }


def test_generation_and_blind_review_accept_each_generated_split(
    tmp_path: Path,
) -> None:
    rows = validate_rows(generate_rows())
    contracts_path = tmp_path / "eval_contracts.jsonl"
    contracts_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    for split in ("validation", "test"):
        contracts = read_contracts([contracts_path], split=split)
        assert len(contracts) == 70
        assert all(row["split"] == split for row in contracts.values())

        modes = Counter()
        clusters_by_family = {family: set() for family in EXPECTED_FAMILIES}
        for row in contracts.values():
            messages = request_messages(row)
            modes[derive_han_evaluation_mode(row, input_han_count=request_input_han_count(messages))] += 1
            for family in EXPECTED_FAMILIES:
                if family in row["tags"]:
                    clusters_by_family[family].add(evaluation_cluster_id(row))
        assert modes == {
            "spontaneous": 50,
            "input_conditioned_cleanup": 10,
            "excluded_han_allowed": 10,
        }
        assert all(len(clusters) >= MINIMUM_SLICE_CLUSTERS for clusters in clusters_by_family.values())


def test_challenge_validation_rejects_train_target_leak_and_small_slice() -> None:
    rows = generate_rows()

    train_tamper = deepcopy(rows)
    train_tamper[0]["split"] = "train"
    with pytest.raises(ValueError, match="validation and test rows only"):
        validate_challenge(train_tamper)

    target_tamper = deepcopy(rows)
    target_tamper[0]["prompt"] = target_tamper[0]["response"]
    with pytest.raises(ValueError, match="reference response appears verbatim"):
        validate_challenge(target_tamper)

    underpowered = [row for row in rows if row["id"] != "heldout-markdown-table-validation-00"]
    with pytest.raises(ValueError, match="validation/markdown-table"):
        validate_challenge(underpowered)
