from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from build_quality_dataset import validate_rows  # noqa: E402
from generate_targeted_quality_data import (  # noqa: E402
    DATASET_REVISION,
    generate_rows,
    write_targeted_dataset,
)
from quality_reward import _is_han, contract_from_mapping, score_constraints  # noqa: E402


def test_targeted_dataset_has_exact_family_and_split_coverage() -> None:
    rows = validate_rows(generate_rows())
    assert len(rows) == 720
    assert Counter(row["split"] for row in rows) == {
        "train": 540,
        "validation": 90,
        "test": 90,
    }
    assert Counter(tag for row in rows for tag in row["tags"] if tag.startswith(("markdown-", "russian-", "han-"))) == {
        "markdown-list": 128,
        "markdown-table": 128,
        "markdown-code": 64,
        "markdown-mixed": 64,
        "russian-style": 128,
        "han-cleanup": 128,
        "han-in-code": 16,
        "han-in-quote": 16,
        "han-in-link": 16,
        "han-retention": 32,
    }
    assert sum(row["contract"]["require_markdown"] for row in rows) == 416
    assert sum("accidental-han" in row["tags"] for row in rows) == 128
    assert sum("accidental-han-control" in row["tags"] for row in rows) == 48
    assert sum("chinese-retention" in row["tags"] for row in rows) == 32


def test_semantic_groups_never_cross_splits() -> None:
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for row in generate_rows():
        family_and_group = row["id"].rsplit("-", 1)[0]
        splits_by_group[family_and_group].add(row["split"])
    assert all(len(splits) == 1 for splits in splits_by_group.values())


def test_targeted_prompts_and_ids_are_unique_and_generation_is_deterministic() -> None:
    first = generate_rows()
    second = generate_rows()
    assert first == second
    assert len({row["id"] for row in first}) == len(first)
    assert len({row["prompt"] for row in first}) == len(first)


def test_han_cleanup_and_scope_controls_have_the_intended_boundary() -> None:
    rows = validate_rows(generate_rows())
    for row in rows:
        family = (
            next(tag for tag in row["tags"] if tag.startswith("han-"))
            if any(tag.startswith("han-") for tag in row["tags"])
            else None
        )
        raw_response_han = sum(_is_han(character) for character in row["response"])
        result = score_constraints(row["response"], contract_from_mapping(row["contract"]))
        if family == "han-cleanup":
            assert any(_is_han(character) for character in row["prompt"])
            assert raw_response_han == 0
            assert result.han_count == 0
        elif family in {"han-in-code", "han-in-quote", "han-in-link"}:
            assert raw_response_han > 0
            assert result.han_count == 0
        elif family == "han-retention":
            assert raw_response_han > 0
            assert row["contract"]["allow_han"] is True


def test_targeted_cli_artifacts_are_buildable_and_measured(tmp_path: Path) -> None:
    manifest = write_targeted_dataset(tmp_path)
    assert manifest["row_count"] == 720
    assert manifest["dataset_revision"] == DATASET_REVISION
    assert manifest["split_counts"] == {"test": 90, "train": 540, "validation": 90}
    assert len(manifest["dataset_sha256"]) == 64
    assert len(manifest["generator_sha256"]) == 64
    assert len(manifest["artifact_manifest_sha256"]) == 64
    assert pq.read_table(tmp_path / "artifacts" / "sft_train.parquet").num_rows == 540
    assert pq.read_table(tmp_path / "artifacts" / "sft_validation.parquet").num_rows == 90
    assert pq.read_table(tmp_path / "artifacts" / "sft_test.parquet").num_rows == 90
    assert not (tmp_path / "artifacts" / "rl_constraint_smoke.parquet").exists()
