from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from evaluate_quality_outputs import evaluate_rows  # noqa: E402


def _row(
    example_id: str,
    completion: str,
    *,
    mode: str = "spontaneous",
    input_han_count: int = 0,
    requested_language: str = "ru",
    allow_han: bool = False,
    token_count: object = 10,
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": example_id,
        "completion": completion,
        "contract": {
            "requested_language": requested_language,
            "allow_han": allow_han,
            "require_markdown": False,
            "required_markdown_blocks": [],
        },
        "input_han_count": input_han_count,
        "input_contains_han": input_han_count > 0,
        "han_evaluation_mode": mode,
        "evaluation_cluster_id": f"cluster:{example_id}",
    }
    if token_count is not None:
        row["completion_token_count"] = token_count
    return row


def test_separates_spontaneous_cleanup_scope_control_and_allowed_han() -> None:
    rows = [
        _row("spontaneous-clean", "Чистый ответ."),
        _row("spontaneous-han", "Ответ 中", token_count=20),
        _row(
            "cleanup-clean",
            "Исправленный ответ.",
            mode="input_conditioned_cleanup",
            input_han_count=2,
        ),
        _row(
            "cleanup-residual",
            "Осталось 结果",
            mode="input_conditioned_cleanup",
            input_han_count=2,
        ),
        _row(
            "scope-clean",
            "Код не повторён.",
            mode="input_conditioned_scope_control",
            input_han_count=1,
        ),
        _row(
            "scope-copied",
            "Скопировано 中",
            mode="input_conditioned_scope_control",
            input_han_count=1,
        ),
        _row(
            "allowed-zh",
            "中文回答",
            mode="excluded_han_allowed",
            input_han_count=3,
            requested_language="zh",
            token_count=None,
        ),
        _row(
            "allowed-contract",
            "Ответ 中",
            mode="excluded_han_allowed",
            input_han_count=0,
            allow_han=True,
            token_count=None,
        ),
    ]

    summary, details = evaluate_rows(rows)

    spontaneous = summary["spontaneous_han"]
    assert spontaneous["row_count"] == 2
    assert spontaneous["han_output_row_count"] == 1
    assert spontaneous["han_output_row_rate"] == 0.5
    assert spontaneous["han_character_count"] == 1
    assert math.isclose(
        spontaneous["han_character_micro_rate_per_1000_visible_characters"],
        1000.0 / 18.0,
    )
    assert math.isclose(
        spontaneous["han_character_micro_rate_per_1000_completion_tokens"],
        1000.0 / 30.0,
    )
    assert spontaneous["completion_token_count_coverage"] == 1.0

    cleanup = summary["input_conditioned_han_cleanup"]
    assert cleanup["row_count"] == 2
    assert cleanup["successful_cleanup_row_count"] == 1
    assert cleanup["cleanup_success_rate"] == 0.5
    assert cleanup["residual_han_output_row_count"] == 1
    assert cleanup["han_character_count"] == 2

    scope = summary["input_conditioned_han_scope_control"]
    assert scope["row_count"] == 2
    assert scope["han_output_row_count"] == 1
    assert scope["han_output_row_rate"] == 0.5
    assert summary["excluded_han_allowed_count"] == 2
    assert {detail["id"]: detail["han_evaluation_mode"] for detail in details} == {
        row["id"]: row["han_evaluation_mode"] for row in rows
    }


def test_cleanup_success_is_not_reported_for_scope_control_rows() -> None:
    summary, _ = evaluate_rows(
        [
            _row(
                "scope",
                "Результат без копирования.",
                mode="input_conditioned_scope_control",
                input_han_count=1,
            )
        ]
    )

    assert summary["input_conditioned_han_cleanup"]["cleanup_success_rate"] is None
    assert "cleanup_success_rate" not in summary["input_conditioned_han_scope_control"]


def test_token_micro_rate_requires_complete_native_positive_integer_coverage() -> None:
    summary, _ = evaluate_rows(
        [
            _row("missing", "Ответ " + "中" * 100, token_count=None),
            _row("covered", "Чистый ответ.", token_count=10),
        ]
    )

    spontaneous = summary["spontaneous_han"]
    assert spontaneous["completion_token_count_coverage"] == 0.5
    assert spontaneous["completion_token_count"] is None
    assert spontaneous["han_character_micro_rate_per_1000_completion_tokens"] is None
    assert (
        "sum(output Han characters)"
        in summary["metric_definitions"]["han_character_micro_rate_per_1000_completion_tokens"]
    )


@pytest.mark.parametrize("bad_token_count", [True, False, 1.0, 3.5, "12", 0, -1])
def test_completion_token_count_rejects_non_native_positive_integers(
    bad_token_count: object,
) -> None:
    error = TypeError if isinstance(bad_token_count, (bool, float, str)) else ValueError
    with pytest.raises(error, match="native positive integer"):
        evaluate_rows([_row("invalid-token", "Ответ.", token_count=bad_token_count)])


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        ({"han_evaluation_mode": "unknown"}, ValueError, "han_evaluation_mode"),
        ({"input_han_count": True}, TypeError, "native nonnegative integer"),
        ({"input_han_count": -1}, ValueError, "native nonnegative integer"),
        ({"input_contains_han": 1}, TypeError, "must be a boolean"),
        (
            {"input_han_count": 0, "input_contains_han": True},
            ValueError,
            "disagrees",
        ),
        (
            {
                "han_evaluation_mode": "spontaneous",
                "input_han_count": 1,
                "input_contains_han": True,
            },
            ValueError,
            "Han-free input",
        ),
        (
            {"han_evaluation_mode": "input_conditioned_cleanup"},
            ValueError,
            "Han-bearing input",
        ),
        (
            {"han_evaluation_mode": "excluded_han_allowed"},
            ValueError,
            "conflicts with the output contract",
        ),
        ({"evaluation_cluster_id": ""}, ValueError, "evaluation_cluster_id"),
    ],
)
def test_mode_count_and_contract_inconsistencies_fail_closed(
    mutation: dict[str, object], error: type[Exception], message: str
) -> None:
    row = _row("invalid", "Ответ.")
    row.update(mutation)
    with pytest.raises(error, match=message):
        evaluate_rows([row])


def test_han_allowed_contract_cannot_enter_spontaneous_cohort() -> None:
    with pytest.raises(ValueError, match="Han-allowed"):
        evaluate_rows([_row("allowed", "Ответ 中", allow_han=True)])
