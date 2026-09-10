from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "glm52_lora"
sys.path.insert(0, str(EXAMPLE))

from build_clean_v4_training_view import (  # noqa: E402
    CANONICAL_TASKS_BUILDER_SHA256,
    GLOBAL_BATCH_SIZE,
    TARGET_TRAIN_ROWS,
    TARGETED_DATASET,
    TRAINING_STEPS,
    WIKIPEDIA_DATASET,
    select_training_rows,
    sha256_file,
    validate_builder_identity,
)


def _row(example_id: str, dataset: str, tags: list[str]) -> dict:
    return {
        "id": example_id,
        "split": "train",
        "provenance": {"dataset": dataset},
        "tags": tags,
    }


def test_clean_v4_selection_is_deterministic_and_never_omits_targeted_rows() -> None:
    targeted = [_row(f"targeted-{index:04d}", TARGETED_DATASET, ["russian", "markdown-list"]) for index in range(204)]
    wikipedia = [
        _row(
            f"wikipedia-{index:04d}",
            WIKIPEDIA_DATASET,
            [
                "russian",
                "teacher-free",
                "han-cleanup" if index % 2 else "russian-case-period-restoration",
            ],
        )
        for index in range(1608)
    ]
    rows = [*targeted, *wikipedia]
    bucket_by_id = {row["id"]: ("seq256", "seq384", "seq768")[index % 3] for index, row in enumerate(rows)}

    first_selected, first_omitted, first_details = select_training_rows(rows, bucket_by_id)
    second_selected, second_omitted, second_details = select_training_rows(rows, bucket_by_id)

    assert first_selected == second_selected
    assert first_omitted == second_omitted
    assert first_details == second_details
    assert len(first_selected) == TARGET_TRAIN_ROWS
    assert len(first_omitted) == 20
    assert all(row["provenance"]["dataset"] == WIKIPEDIA_DATASET for row in first_omitted)
    assert {row["id"] for row in targeted}.issubset({row["id"] for row in first_selected})
    assert first_details["algorithm"]["sampling"] == "none"
    assert first_details["algorithm"]["replacement"] is False
    assert GLOBAL_BATCH_SIZE * TRAINING_STEPS == len(first_selected)


def test_builder_identity_accepts_only_current_or_canonical_tasks_source() -> None:
    filename = "build_clean_v4_training_view.py"
    validate_builder_identity({"file": filename, "sha256": sha256_file(EXAMPLE / filename)})
    validate_builder_identity({"file": filename, "sha256": CANONICAL_TASKS_BUILDER_SHA256})

    with pytest.raises(ValueError, match="builder-code drift"):
        validate_builder_identity({"file": filename, "sha256": "0" * 64})
