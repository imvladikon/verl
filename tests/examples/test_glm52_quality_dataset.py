from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from build_quality_dataset import (  # noqa: E402
    read_jsonl,
    validate_rows,
    write_artifacts,
)


def _row(
    example_id: str,
    split: str,
    prompt: str,
    response: str,
    *,
    use_for_rl: bool = False,
) -> dict:
    return {
        "id": example_id,
        "split": split,
        "prompt": prompt,
        "response": response,
        "contract": {"requested_language": "ru", "allow_han": False},
        "use_for_constraint_rl_smoke": use_for_rl,
    }


def test_example_dataset_builds_disjoint_sft_eval_and_constraint_smoke(
    tmp_path: Path,
) -> None:
    source = ROOT / "examples" / "glm52_lora" / "quality_dataset.example.jsonl"
    rows = validate_rows(read_jsonl(source))
    manifest = write_artifacts(rows, tmp_path)

    assert manifest["counts"] == {"test": 2, "train": 4, "validation": 1}
    assert manifest["rl_constraint_smoke_count"] == 3
    assert manifest["eval_count"] == 3
    assert pq.read_table(tmp_path / "sft_train.parquet").num_rows == 4
    assert pq.read_table(tmp_path / "sft_validation.parquet").num_rows == 1
    assert pq.read_table(tmp_path / "sft_test.parquet").num_rows == 2
    assert pq.read_table(tmp_path / "rl_constraint_smoke.parquet").num_rows == 3
    eval_rows = [json.loads(line) for line in (tmp_path / "eval_contracts.jsonl").read_text().splitlines()]
    assert {row["split"] for row in eval_rows} == {"validation", "test"}


def test_prompt_leakage_is_rejected_even_after_whitespace_normalization() -> None:
    rows = [
        _row("train-1", "train", "Проверь   ответ", "Ответ проверен."),
        _row("test-1", "test", " проверь ответ ", "Результат проверен."),
    ]
    with pytest.raises(ValueError, match="prompt leakage/duplicate"):
        validate_rows(rows)


@pytest.mark.parametrize(
    ("response", "error"),
    [
        ("Ответ содержит 中.", "accidental Han"),
        ("Ответ с **незакрытым выделением", "Markdown defects"),
        ("plain latin response", "no Cyrillic letters"),
    ],
)
def test_dirty_curated_responses_are_rejected(response: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        validate_rows([_row("bad", "train", "Исправь ответ", response)])


def test_rl_artifact_is_explicitly_constraint_only(tmp_path: Path) -> None:
    rows = validate_rows(
        [
            _row(
                "train-1",
                "train",
                "Ответь по-русски",
                "Корректный русский ответ.",
                use_for_rl=True,
            )
        ]
    )
    manifest = write_artifacts(rows, tmp_path)
    rl_row = pq.read_table(tmp_path / "rl_constraint_smoke.parquet").to_pylist()[0]
    assert rl_row["data_source"] == "glm52_quality"
    assert rl_row["extra_info"]["constraint_only_smoke"] is True
    assert "not a production RL dataset" in manifest["warning"]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("tags", "quality", "tags must be a list of strings"),
        (
            "use_for_constraint_rl_smoke",
            "false",
            "use_for_constraint_rl_smoke must be a boolean",
        ),
        ("contract", {"allow_han": "false"}, "allow_han must be a boolean"),
    ],
)
def test_ambiguous_schema_values_are_rejected(field: str, value: object, error: str) -> None:
    row = _row("schema", "train", "Ответь", "Корректный ответ.")
    row[field] = value
    with pytest.raises(TypeError, match=error):
        validate_rows([row])
