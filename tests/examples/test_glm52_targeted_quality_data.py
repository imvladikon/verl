from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from audit_split_isolation import audit_rows  # noqa: E402
from build_quality_dataset import validate_rows  # noqa: E402
from build_quality_review_queue import _markdown_contract  # noqa: E402
from generate_targeted_quality_data import (  # noqa: E402
    DATASET_REVISION,
    generate_rows,
    write_targeted_dataset,
)
from quality_reward import (  # noqa: E402
    _is_han,
    contract_from_mapping,
    score_constraints,
)


def test_targeted_dataset_has_exact_family_and_split_coverage() -> None:
    rows = validate_rows(generate_rows())
    assert len(rows) == 244
    assert Counter(row["split"] for row in rows) == {
        "train": 204,
        "validation": 20,
        "test": 20,
    }
    assert Counter(tag for row in rows for tag in row["tags"] if tag.startswith(("markdown-", "russian-", "han-"))) == {
        "markdown-list": 28,
        "markdown-table": 28,
        "markdown-code": 28,
        "markdown-mixed": 28,
        "russian-style": 28,
        "han-cleanup": 28,
        "han-in-code": 16,
        "han-in-quote": 16,
        "han-in-link": 16,
        "han-retention": 28,
    }
    assert sum(row["contract"]["require_markdown"] for row in rows) == 144
    assert sum("accidental-han" in row["tags"] for row in rows) == 28
    assert sum("accidental-han-control" in row["tags"] for row in rows) == 48
    assert sum("chinese-retention" in row["tags"] for row in rows) == 28


def test_targeted_v4_is_clean_under_the_production_split_auditor() -> None:
    audit = audit_rows(
        generate_rows(),
        inline_source_datasets=["project-authored/glm52-targeted-quality"],
    )

    assert audit["status"] == "PASS"
    assert not any(audit["counts"].values())


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


def test_targeted_responses_do_not_cross_splits() -> None:
    splits_by_response: dict[str, set[str]] = defaultdict(set)
    for row in generate_rows():
        splits_by_response[row["response"]].add(row["split"])
    assert all(len(splits) == 1 for splits in splits_by_response.values())


def test_every_required_markdown_block_is_explicitly_requested() -> None:
    for row in generate_rows():
        inferred, _ = _markdown_contract(row["prompt"], row["response"])
        required = set(row["contract"]["required_markdown_blocks"])
        assert required <= set(inferred.required_blocks), row["id"]


def test_table_targets_honor_table_only_prompts() -> None:
    table_rows = [row for row in generate_rows() if "markdown-table" in row["tags"]]
    assert len(table_rows) == 28
    assert all(row["contract"]["required_markdown_blocks"] == ["table"] for row in table_rows)
    assert all(row["response"].startswith("| Статус | Действие |") for row in table_rows)
    assert all(not row["response"].startswith("#") for row in table_rows)


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
    assert manifest["row_count"] == 244
    assert manifest["dataset_revision"] == DATASET_REVISION
    assert manifest["split_counts"] == {"test": 20, "train": 204, "validation": 20}
    assert len(manifest["dataset_sha256"]) == 64
    assert len(manifest["generator_sha256"]) == 64
    assert len(manifest["artifact_manifest_sha256"]) == 64
    assert pq.read_table(tmp_path / "artifacts" / "sft_train.parquet").num_rows == 204
    assert pq.read_table(tmp_path / "artifacts" / "sft_validation.parquet").num_rows == 20
    assert pq.read_table(tmp_path / "artifacts" / "sft_test.parquet").num_rows == 20
    assert not (tmp_path / "artifacts" / "rl_constraint_smoke.parquet").exists()
