# ruff: noqa: E402

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from build_blind_quality_review import jsonl_sha256
from compare_quality_outputs import compare_rows, main
from generate_full_quality_outputs_sglang import (
    OFFICIAL_MODEL_ARTIFACTS,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _row(
    example_id: str,
    completion: str,
    semantic_score: float | None,
    *,
    require_markdown: bool = False,
    required_markdown_blocks: tuple[str, ...] | None = None,
    requested_language: str = "ru",
    allow_han: bool = False,
    han_evaluation_mode: str = "spontaneous",
    input_han_count: int = 0,
    evaluation_cluster_id: str | None = None,
) -> dict[str, object]:
    def digest(label: str) -> str:
        return hashlib.sha256(f"{label}:{example_id}".encode()).hexdigest()

    markdown_blocks = (
        list(required_markdown_blocks)
        if required_markdown_blocks is not None
        else (["heading", "list"] if require_markdown else [])
    )
    contract = {
        "requested_language": requested_language,
        "allow_han": allow_han,
        "require_markdown": require_markdown or bool(markdown_blocks),
        "required_markdown_blocks": markdown_blocks,
    }
    decoding = {
        "schema_version": 3,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_completion_tokens": 512,
        "seed": 52,
        "n": 1,
        "stream": False,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    trainer_base = {
        "model_id": "zai-org/GLM-5.2",
        "revision": "cf457fa734ab149ffef225f80893eb38c6ff5cdc",
        "revision_verified": True,
        **OFFICIAL_MODEL_ARTIFACTS[
            (
                "zai-org/GLM-5.2",
                "cf457fa734ab149ffef225f80893eb38c6ff5cdc",
            )
        ],
    }
    inference_base = {
        "model_id": "zai-org/GLM-5.2-FP8",
        "revision": "f33c6dc501ee5a2c7e35155653b1b1abbc320951",
        "revision_verified": True,
        **OFFICIAL_MODEL_ARTIFACTS[
            (
                "zai-org/GLM-5.2-FP8",
                "f33c6dc501ee5a2c7e35155653b1b1abbc320951",
            )
        ],
    }
    pair_runtime = {
        "schema_version": 3,
        "artifact_contract": {
            "trainer_base": trainer_base,
            "inference_base": inference_base,
        },
        "weight_shard_manifest_sha256": {
            "trainer": "1" * 64,
            "inference": "2" * 64,
        },
        "sglang": {
            "repository": "https://github.com/imvladikon/sglang",
            "revision": "0dbdb73509fbf6b3381359df87cde267d453c8d3",
            "tree": "5678fc2ab88fd65411b833c065f510b6d4f5d59c",
        },
        "runtime_script_sha256": {
            "build_quality_sglang_runtime.py": "7" * 64,
            "generate_full_quality_outputs_sglang.py": "8" * 64,
            "launch_quality_sglang_server.py": "9" * 64,
            "build_blind_quality_review.py": "a" * 64,
        },
        "environment_semantics": {
            "python_version": "3.12.11",
            "python_executable_sha256": "b" * 64,
            "installed_distributions_sha256": "c" * 64,
        },
        "server_semantics": {
            "served_base_model": "glm52-base",
            "tp_size": 8,
            "max_model_len": 2048,
        },
    }
    pair_contract = {
        "schema_version": 3,
        "runtime": pair_runtime,
        "decoding": decoding,
        "held_out": {
            "id": example_id,
            "split": "validation",
            "contract": contract,
            "prompt_sha256": digest("prompt"),
            "source_row_sha256": digest("source-row"),
            "reference_response_sha256": digest("reference-response"),
            "request_messages_sha256": digest("messages"),
            "input_han_count": input_han_count,
            "input_contains_han": input_han_count > 0,
            "han_evaluation_mode": han_evaluation_mode,
            "evaluation_cluster_id": evaluation_cluster_id or f"cluster:{example_id}",
        },
    }
    row: dict[str, object] = {
        "id": example_id,
        "split": "validation",
        "prompt_sha256": digest("prompt"),
        "source_row_sha256": digest("source-row"),
        "reference_response_sha256": digest("reference-response"),
        "request_messages_sha256": digest("messages"),
        "input_han_count": input_han_count,
        "input_contains_han": input_han_count > 0,
        "han_evaluation_mode": han_evaluation_mode,
        "evaluation_cluster_id": evaluation_cluster_id or f"cluster:{example_id}",
        "decoding_contract_sha256": _canonical_sha256(decoding),
        "pair_contract": pair_contract,
        "pair_contract_sha256": _canonical_sha256(pair_contract),
        "generation": {
            "variant": "base",
            "runtime_mode": "base",
            "runtime_manifest_sha256": "a" * 64,
            "pair_runtime_contract_sha256": _canonical_sha256(pair_runtime),
            "quality_claim_allowed": True,
            "api_secret_sha256": "d" * 64,
            "trainer_base": trainer_base,
            "inference_base": inference_base,
            "adapter": None,
            "sglang": {"checkout": "/src/sglang", **pair_runtime["sglang"]},
            "server_instance_id": "base-instance",
            "response_id": f"response-base-{example_id}",
            "response_model": "glm52-base",
            "finish_reason": "stop",
            "prompt_tokens": 8,
            "completion_tokens": 12,
            "total_tokens": 20,
        },
        "completion": completion,
        "completion_token_count": 12,
        "contract": contract,
    }
    if semantic_score is not None:
        row["semantic_score"] = semantic_score
        row["semantic_score_provenance"] = {
            "method": "blinded-human-rubric-v1",
            "reviewers": ["reviewer-1", "reviewer-2"],
            "review_artifacts": [
                {"ordinal": 0, "filename": "review-1.jsonl", "sha256": "5" * 64},
                {"ordinal": 1, "filename": "review-2.jsonl", "sha256": "6" * 64},
            ],
            "review_scores": [semantic_score, semantic_score],
            "mean_ratings": {
                "meaning_preservation": 1.0 + 4.0 * semantic_score,
                "russian_naturalness": 1.0 + 4.0 * semantic_score,
                "factual_accuracy": 1.0 + 4.0 * semantic_score,
                "instruction_fulfillment": 1.0 + 4.0 * semantic_score,
            },
            "completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
            "review_item_sha256": "7" * 64,
            "pair_contract_sha256": row["pair_contract_sha256"],
            "adjudication_contract_sha256": "8" * 64,
        }
    return row


def _adapter(row: dict[str, object]) -> dict[str, object]:
    row["generation"] = {
        "variant": "adapter",
        "runtime_mode": "adapter",
        "runtime_manifest_sha256": "b" * 64,
        "pair_runtime_contract_sha256": _canonical_sha256(row["pair_contract"]["runtime"]),
        "quality_claim_allowed": True,
        "api_secret_sha256": "e" * 64,
        "trainer_base": row["pair_contract"]["runtime"]["artifact_contract"]["trainer_base"],
        "inference_base": row["pair_contract"]["runtime"]["artifact_contract"]["inference_base"],
        "adapter": {
            "name": "quality",
            "artifact_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "verification_sha256": "3" * 64,
            "trainer_base_revision": "cf457fa734ab149ffef225f80893eb38c6ff5cdc",
            "profile": "mla-only",
            "rank": 16,
            "alpha": 32,
            "parameter_count": 106_149_888,
            "target_modules": [
                "kv_a_proj_with_mqa",
                "kv_b_proj",
                "o_proj",
                "q_a_proj",
                "q_b_proj",
            ],
        },
        "sglang": {
            "checkout": "/src/sglang",
            **row["pair_contract"]["runtime"]["sglang"],
        },
        "server_instance_id": "adapter-instance",
        "response_id": f"response-adapter-{row['id']}",
        "response_model": "quality",
        "finish_reason": "stop",
        "prompt_tokens": 8,
        "completion_tokens": 12,
        "total_tokens": 20,
    }
    return row


def test_comparator_passes_only_paired_three_target_improvement() -> None:
    base = [
        _row("markdown-a", "сломанный markdown", 0.2, require_markdown=True),
        _row("markdown-b", "ещё один сломанный markdown", 0.3, require_markdown=True),
        _row("han-a", "Русский текст 中", 0.4),
        _row("han-b", "Ещё один русский 结果", 0.5),
        _row("general-russian", "Обычный русский ответ.", 0.8),
    ]
    adapter = [
        _adapter(_row("markdown-a", "## Ответ\n\n1. Первый пункт.", 0.7, require_markdown=True)),
        _adapter(_row("markdown-b", "## Итог\n\n1. Второй пункт.", 0.8, require_markdown=True)),
        _adapter(_row("han-a", "Русский текст.", 0.9)),
        _adapter(_row("han-b", "Ещё один русский текст.", 1.0)),
        _adapter(_row("general-russian", "Обычный русский ответ.", 0.8)),
    ]

    result, details = compare_rows(
        base,
        adapter,
        bootstrap_samples=500,
        minimum_evaluation_rows=1,
        minimum_evaluation_clusters=1,
        minimum_slice_rows=1,
        minimum_slice_clusters=1,
    )

    assert result["status"] == "PASS"
    assert result["target_status"] == {
        "russian_semantic_quality": "PASS",
        "required_markdown_validity": "PASS",
        "accidental_han": "PASS",
        "non_russian_semantic_retention": "NOT_APPLICABLE",
    }
    assert result["base_required_markdown_defects"] == 2
    assert result["adapter_required_markdown_defects"] == 0
    assert result["base_spontaneous_han_output_rows"] == 2
    assert result["adapter_spontaneous_han_output_rows"] == 0
    assert len(details) == 5


def test_comparator_stays_pending_without_semantics_or_reproduced_defects() -> None:
    base = [_row("clean", "Корректный русский текст.", None)]
    adapter = [_adapter(_row("clean", "Другой корректный русский текст.", None))]

    result, _ = compare_rows(base, adapter, bootstrap_samples=100)

    assert result["status"] == "PENDING"
    assert result["target_status"] == {
        "russian_semantic_quality": "PENDING",
        "required_markdown_validity": "NOT_REPRODUCED",
        "accidental_han": "NOT_REPRODUCED",
        "non_russian_semantic_retention": "NOT_APPLICABLE",
    }
    assert result["paired_russian_semantic_coverage"] == 0.0


def test_comparator_reports_semantic_regression_as_failure() -> None:
    base = [_row("semantic", "Хороший русский ответ 中", 0.9)]
    adapter = [_adapter(_row("semantic", "Плохой русский ответ.", 0.1))]

    result, _ = compare_rows(
        base,
        adapter,
        bootstrap_samples=100,
        minimum_evaluation_rows=1,
        minimum_evaluation_clusters=1,
    )

    assert result["status"] == "FAIL"
    assert result["target_status"]["russian_semantic_quality"] == "FAIL"


def test_comparator_requires_legitimate_chinese_semantic_retention() -> None:
    base = [
        _row("markdown", "сломанный markdown", 0.3, require_markdown=True),
        _row("han", "Русский текст 中", 0.5),
        _row("general-russian", "Обычный русский ответ.", 0.8),
        _row(
            "zh-retention",
            "这是一个正确的中文回答。",
            0.9,
            requested_language="zh",
            allow_han=True,
            han_evaluation_mode="excluded_han_allowed",
            input_han_count=3,
        ),
    ]
    adapter = [
        _adapter(
            _row(
                "markdown",
                "## Ответ\n\n- Исправлено.",
                0.6,
                require_markdown=True,
            )
        ),
        _adapter(_row("han", "Русский текст.", 0.8)),
        _adapter(_row("general-russian", "Обычный русский ответ.", 0.8)),
        _adapter(
            _row(
                "zh-retention",
                "Нерелевантный русский ответ.",
                0.1,
                requested_language="zh",
                allow_han=True,
                han_evaluation_mode="excluded_han_allowed",
                input_han_count=3,
            )
        ),
    ]

    result, _ = compare_rows(base, adapter, bootstrap_samples=100)

    assert result["status"] == "FAIL"
    assert result["target_status"]["non_russian_semantic_retention"] == "FAIL"
    assert result["paired_retention_semantic_coverage"] == 1.0


def test_comparator_accepts_noninferior_chinese_retention() -> None:
    base = [
        _row("markdown", "сломанный markdown", 0.3, require_markdown=True),
        _row("han", "Русский текст 中", 0.5),
        _row("general-russian", "Обычный русский ответ.", 0.8),
        _row(
            "zh-retention",
            "这是一个正确的中文回答。",
            0.8,
            requested_language="zh",
            allow_han=True,
            han_evaluation_mode="excluded_han_allowed",
            input_han_count=3,
        ),
    ]
    adapter = [
        _adapter(
            _row(
                "markdown",
                "## Ответ\n\n- Исправлено.",
                0.6,
                require_markdown=True,
            )
        ),
        _adapter(_row("han", "Русский текст.", 0.8)),
        _adapter(_row("general-russian", "Обычный русский ответ.", 0.8)),
        _adapter(
            _row(
                "zh-retention",
                "这是一个同样正确的中文回答。",
                0.8,
                requested_language="zh",
                allow_han=True,
                han_evaluation_mode="excluded_han_allowed",
                input_han_count=3,
            )
        ),
    ]

    result, _ = compare_rows(
        base,
        adapter,
        bootstrap_samples=100,
        minimum_evaluation_rows=1,
        minimum_evaluation_clusters=1,
        minimum_slice_rows=1,
        minimum_slice_clusters=1,
    )

    assert result["status"] == "PASS"
    assert result["target_status"]["non_russian_semantic_retention"] == "PASS"


def test_comparator_stays_pending_without_chinese_retention_scores() -> None:
    base = [
        _row("markdown", "сломанный markdown", 0.3, require_markdown=True),
        _row("han", "Русский текст 中", 0.5),
        _row(
            "zh-retention",
            "这是一个正确的中文回答。",
            None,
            requested_language="zh",
            allow_han=True,
            han_evaluation_mode="excluded_han_allowed",
            input_han_count=3,
        ),
    ]
    adapter = [
        _adapter(
            _row(
                "markdown",
                "## Ответ\n\n- Исправлено.",
                0.6,
                require_markdown=True,
            )
        ),
        _adapter(_row("han", "Русский текст.", 0.8)),
        _adapter(
            _row(
                "zh-retention",
                "这是一个同样正确的中文回答。",
                None,
                requested_language="zh",
                allow_han=True,
                han_evaluation_mode="excluded_han_allowed",
                input_han_count=3,
            )
        ),
    ]

    result, _ = compare_rows(
        base,
        adapter,
        bootstrap_samples=100,
        minimum_slice_rows=1,
        minimum_slice_clusters=1,
    )

    assert result["status"] == "PENDING"
    assert result["target_status"]["non_russian_semantic_retention"] == "PENDING"
    assert result["paired_retention_semantic_coverage"] == 0.0


@pytest.mark.parametrize(
    ("field", "new_value", "message"),
    [
        ("prompt_sha256", "different", "prompt_sha256"),
        ("request_messages_sha256", "different", "request_messages_sha256"),
        ("decoding_contract_sha256", "different", "decoding_contract_sha256"),
        ("contract", {"requested_language": "zh"}, "contracts differ"),
    ],
)
def test_comparator_rejects_unpaired_contracts(field: str, new_value: object, message: str) -> None:
    base = [_row("same", "Ответ 中", 0.4)]
    adapter_row = _adapter(_row("same", "Ответ.", 0.6))
    adapter_row[field] = new_value

    with pytest.raises(ValueError, match=message):
        compare_rows(base, [adapter_row], bootstrap_samples=100)


@pytest.mark.parametrize(
    "field",
    [
        "prompt_sha256",
        "request_messages_sha256",
        "decoding_contract_sha256",
        "pair_contract_sha256",
    ],
)
def test_comparator_rejects_missing_or_malformed_provenance_digest(field: str) -> None:
    for broken in (None, "not-a-digest", "A" * 64):
        base = [_row("same", "Ответ 中", 0.4)]
        adapter = [_adapter(_row("same", "Ответ.", 0.6))]
        if broken is None:
            del base[0][field]
            del adapter[0][field]
        else:
            base[0][field] = broken
            adapter[0][field] = broken
        with pytest.raises(ValueError, match=field):
            compare_rows(base, adapter, bootstrap_samples=100)


@pytest.mark.parametrize("field", ["input_han_count", "input_contains_han"])
def test_official_comparator_rejects_missing_input_han_provenance(field: str) -> None:
    base = [_row("same", "Ответ 中", 0.4)]
    adapter = [_adapter(_row("same", "Ответ.", 0.6))]
    del base[0][field]
    del adapter[0][field]

    with pytest.raises(ValueError, match=field):
        compare_rows(base, adapter, bootstrap_samples=100)


def test_comparator_rejects_unpaired_semantic_coverage_and_ids() -> None:
    base = [_row("base", "Ответ 中", 0.4)]
    adapter = [_adapter(_row("base", "Ответ.", None))]
    with pytest.raises(ValueError, match="semantic score coverage"):
        compare_rows(base, adapter, bootstrap_samples=100)

    with pytest.raises(ValueError, match="ids differ"):
        compare_rows(
            base,
            [_adapter(_row("adapter", "Ответ.", 0.6))],
            bootstrap_samples=100,
        )


def test_cleanup_majority_cannot_mask_spontaneous_han_regression() -> None:
    base: list[dict[str, object]] = []
    adapter: list[dict[str, object]] = []
    for index in range(60):
        example_id = f"cleanup-{index:02d}"
        base.append(
            _row(
                example_id,
                "Удалите 中",
                0.5,
                han_evaluation_mode="input_conditioned_cleanup",
                input_han_count=1,
                evaluation_cluster_id=f"cleanup-source-{index // 3}",
            )
        )
        adapter.append(
            _adapter(
                _row(
                    example_id,
                    "Исправлено.",
                    0.5,
                    han_evaluation_mode="input_conditioned_cleanup",
                    input_han_count=1,
                    evaluation_cluster_id=f"cleanup-source-{index // 3}",
                )
            )
        )
    base.append(_row("spontaneous-regression", "Чисто.", 0.5))
    adapter.append(_adapter(_row("spontaneous-regression", "Новая ошибка 中", 0.5)))

    result, _ = compare_rows(base, adapter, bootstrap_samples=200)

    assert result["status"] == "FAIL"
    assert result["target_status"]["accidental_han"] == "FAIL"
    assert result["base_spontaneous_han_output_rows"] == 0
    assert result["adapter_spontaneous_han_output_rows"] == 1
    assert result["han_cohort_row_counts"]["input_conditioned_cleanup"] == 60
    assert result["han_cohort_row_counts"]["spontaneous"] == 1
    assert (
        result["metrics"]["input_conditioned_cleanup_success_macro_per_row_adapter_minus_base"]["macro_per_row_mean"]
        == 1.0
    )
    assert (
        result["metrics"]["spontaneous_han_output_row_macro_per_row_base_minus_adapter"]["macro_per_row_mean"] == -1.0
    )


def test_cleanup_regression_blocks_spontaneous_improvement() -> None:
    base = [
        _row("spontaneous", "Ошибка 中", 0.5),
        _row(
            "cleanup",
            "Исправлено.",
            0.5,
            han_evaluation_mode="input_conditioned_cleanup",
            input_han_count=1,
        ),
    ]
    adapter = [
        _adapter(_row("spontaneous", "Исправлено.", 0.5)),
        _adapter(
            _row(
                "cleanup",
                "Новая ошибка 中",
                0.5,
                han_evaluation_mode="input_conditioned_cleanup",
                input_han_count=1,
            )
        ),
    ]

    result, _ = compare_rows(
        base,
        adapter,
        bootstrap_samples=100,
        minimum_slice_rows=1,
        minimum_slice_clusters=1,
    )

    assert result["status"] == "FAIL"
    assert result["target_status"]["accidental_han"] == "FAIL"
    assert result["han_component_status"] == {
        "spontaneous": "PASS",
        "input_conditioned_cleanup": "FAIL",
        "input_conditioned_scope_control": "NOT_APPLICABLE",
    }


def test_han_character_severity_regression_blocks_row_count_improvement() -> None:
    base = [
        _row("first", "Ошибка 中", 0.5),
        _row("second", "Ошибка 文", 0.5),
    ]
    adapter = [
        _adapter(_row("first", "Ошибка " + "中" * 1000, 0.5)),
        _adapter(_row("second", "Исправлено.", 0.5)),
    ]

    result, _ = compare_rows(base, adapter, bootstrap_samples=100)

    assert result["status"] == "FAIL"
    assert result["target_status"]["accidental_han"] == "FAIL"
    assert result["base_spontaneous_han_output_rows"] == 2
    assert result["adapter_spontaneous_han_output_rows"] == 1
    assert result["base_han_character_counts_by_cohort"]["spontaneous"] == 2
    assert result["adapter_han_character_counts_by_cohort"]["spontaneous"] == 1000


def test_production_sample_size_gate_rejects_one_row_one_cluster() -> None:
    base = [_row("single", "Ошибка 中", 0.4)]
    adapter = [_adapter(_row("single", "Исправлено.", 0.8))]

    result, _ = compare_rows(base, adapter, bootstrap_samples=100)

    assert result["status"] == "PENDING"
    assert result["sample_size_gate"] == "PENDING"
    assert result["count"] == 1
    assert result["evaluation_cluster_count"] == 1


def test_semantic_score_is_derived_from_bound_reviewer_scores() -> None:
    base = [_row("semantic", "Ответ 中", 0.4)]
    adapter = [_adapter(_row("semantic", "Ответ.", 0.6))]
    adapter[0]["semantic_score"] = 1.0

    with pytest.raises(ValueError, match="differs from reviewer scores"):
        compare_rows(base, adapter, bootstrap_samples=100)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("semantic_noninferiority_margin", float("inf")),
        ("semantic_noninferiority_margin", float("nan")),
        ("retention_noninferiority_margin", float("inf")),
        ("retention_noninferiority_margin", float("nan")),
    ],
)
def test_comparator_rejects_nonfinite_margins(argument: str, value: float) -> None:
    base = [_row("same", "Ответ 中", 0.4)]
    adapter = [_adapter(_row("same", "Ответ.", 0.6))]

    with pytest.raises(ValueError, match="margin must be nonnegative"):
        compare_rows(base, adapter, bootstrap_samples=100, **{argument: value})


def test_regional_language_tags_use_language_root() -> None:
    base = [
        _row("ru", "Русский ответ 中", 0.4, requested_language="ru-RU"),
        _row(
            "zh",
            "中文回答。",
            0.8,
            requested_language="zh-CN",
            allow_han=True,
            han_evaluation_mode="excluded_han_allowed",
            input_han_count=4,
        ),
    ]
    adapter = [
        _adapter(_row("ru", "Русский ответ.", 0.7, requested_language="ru-RU")),
        _adapter(
            _row(
                "zh",
                "另一个中文回答。",
                0.8,
                requested_language="zh-CN",
                allow_han=True,
                han_evaluation_mode="excluded_han_allowed",
                input_han_count=4,
            )
        ),
    ]

    result, _ = compare_rows(
        base,
        adapter,
        bootstrap_samples=100,
        minimum_slice_rows=1,
        minimum_slice_clusters=1,
    )

    assert result["russian_example_count"] == 1
    assert result["retention_example_count"] == 1
    assert result["target_status"]["non_russian_semantic_retention"] == "PASS"


def test_underpowered_legitimate_chinese_slice_cannot_pass_global_gate() -> None:
    base = [_row(f"general-{index}", "Обычный русский ответ.", 0.8) for index in range(18)] + [
        _row(
            f"targeted-han-retention-{index:02d}",
            "这是一个正确的中文回答。",
            0.8,
            requested_language="zh-CN",
            allow_han=True,
            han_evaluation_mode="excluded_han_allowed",
            input_han_count=3,
        )
        for index in range(2)
    ]
    adapter = [_adapter(_row(f"general-{index}", "Обычный русский ответ.", 0.8)) for index in range(18)] + [
        _adapter(
            _row(
                f"targeted-han-retention-{index:02d}",
                "这是一个同样正确的中文回答。",
                0.8,
                requested_language="zh-CN",
                allow_han=True,
                han_evaluation_mode="excluded_han_allowed",
                input_han_count=3,
            )
        )
        for index in range(2)
    ]

    result, _ = compare_rows(
        base,
        adapter,
        bootstrap_samples=100,
        minimum_evaluation_rows=20,
        minimum_evaluation_clusters=10,
        minimum_slice_rows=3,
        minimum_slice_clusters=3,
    )

    assert result["sample_size_gate"] == "PASS"
    assert result["retention_slice_status"]["all_non_russian"]["status"] == ("PENDING")
    chinese = result["retention_slice_status"]["legitimate_chinese"]
    assert chinese["row_count"] == 2
    assert chinese["cluster_count"] == 2
    assert chinese["sample_size_status"] == "PENDING"
    assert chinese["status"] == "PENDING"
    assert result["target_status"]["non_russian_semantic_retention"] == "PENDING"
    assert result["status"] == "PENDING"


def test_markdown_family_regression_cannot_hide_in_aggregate_improvement() -> None:
    base = [
        _row(
            f"targeted-markdown-list-{index:02d}",
            "сломанный список",
            0.4,
            required_markdown_blocks=("list",),
        )
        for index in range(18)
    ] + [
        _row(
            f"targeted-markdown-table-{index:02d}",
            "| Поле | Значение |\n| --- | --- |\n| Язык | русский |",
            0.4,
            required_markdown_blocks=("table",),
        )
        for index in range(2)
    ]
    adapter = [
        _adapter(
            _row(
                f"targeted-markdown-list-{index:02d}",
                "- Исправленный пункт.",
                0.7,
                required_markdown_blocks=("list",),
            )
        )
        for index in range(18)
    ] + [
        _adapter(
            _row(
                f"targeted-markdown-table-{index:02d}",
                "сломанная таблица",
                0.7,
                required_markdown_blocks=("table",),
            )
        )
        for index in range(2)
    ]

    result, _ = compare_rows(
        base,
        adapter,
        bootstrap_samples=500,
        minimum_evaluation_rows=1,
        minimum_evaluation_clusters=1,
        minimum_slice_rows=2,
        minimum_slice_clusters=2,
    )

    aggregate = result["metrics"]["required_markdown_valid_macro_per_row_adapter_minus_base"]
    assert aggregate["macro_per_row_mean"] > 0.0
    assert result["base_required_markdown_defects"] == 18
    assert result["adapter_required_markdown_defects"] == 2
    assert result["required_markdown_family_status"]["list"]["status"] == "PASS"
    assert result["required_markdown_family_status"]["table"]["status"] == "FAIL"
    assert result["target_status"]["required_markdown_validity"] == "FAIL"
    assert result["status"] == "FAIL"


def test_known_russian_improvement_does_not_hide_general_regression() -> None:
    base = [
        _row(
            f"targeted-russian-style-{index:02d}",
            "Русский ответ.",
            0.2,
        )
        for index in range(8)
    ] + [_row(f"general-russian-{index:02d}", "Русский ответ.", 0.8) for index in range(4)]
    adapter = [
        _adapter(
            _row(
                f"targeted-russian-style-{index:02d}",
                "Исправленный русский ответ.",
                0.8,
            )
        )
        for index in range(8)
    ] + [_adapter(_row(f"general-russian-{index:02d}", "Хуже.", 0.7)) for index in range(4)]

    result, _ = compare_rows(
        base,
        adapter,
        bootstrap_samples=500,
        minimum_evaluation_rows=1,
        minimum_evaluation_clusters=1,
        minimum_slice_rows=4,
        minimum_slice_clusters=4,
    )

    aggregate = result["metrics"]["russian_semantic_score_macro_per_row_adapter_minus_base"]
    assert aggregate["macro_per_row_mean"] > 0.0
    assert result["russian_semantic_slice_status"]["known_defect_correction"]["status"] == "PASS"
    assert result["russian_semantic_slice_status"]["general_noninferiority"]["status"] == "FAIL"
    assert result["target_status"]["russian_semantic_quality"] == "FAIL"
    assert result["status"] == "FAIL"


def test_bootstrap_resamples_clusters_and_is_deterministic() -> None:
    base = [
        _row("a-1", "Ответ 中", 0.4, evaluation_cluster_id="source-a"),
        _row("a-2", "Ответ 中", 0.4, evaluation_cluster_id="source-a"),
        _row("b-1", "Ответ 中", 0.4, evaluation_cluster_id="source-b"),
    ]
    adapter = [
        _adapter(_row("a-1", "Ответ.", 0.6, evaluation_cluster_id="source-a")),
        _adapter(_row("a-2", "Ответ.", 0.6, evaluation_cluster_id="source-a")),
        _adapter(_row("b-1", "Ответ 中", 0.6, evaluation_cluster_id="source-b")),
    ]

    first, _ = compare_rows(deepcopy(base), deepcopy(adapter), bootstrap_samples=200, bootstrap_seed=7)
    second, _ = compare_rows(deepcopy(base), deepcopy(adapter), bootstrap_samples=200, bootstrap_seed=7)

    metric_name = "spontaneous_han_output_row_macro_per_row_base_minus_adapter"
    assert first["metrics"][metric_name] == second["metrics"][metric_name]
    assert first["metrics"][metric_name]["method"] == ("paired-cluster-bootstrap-row-macro-mean")
    assert first["metrics"][metric_name]["row_count"] == 3
    assert first["metrics"][metric_name]["cluster_count"] == 2
    assert first["metrics"][metric_name]["macro_per_row_mean"] == pytest.approx(2 / 3)
    assert first["bootstrap_method"] == "paired-cluster-bootstrap-row-macro-mean"
    assert "evaluator_token_micro_rates" in first["metric_definitions"]


def test_comparator_token_normalized_macro_metric_requires_full_pair_coverage() -> None:
    base = [
        _row("covered", "Ответ 中", 0.4),
        _row("missing", "Ответ 中", 0.4),
    ]
    adapter = [
        _adapter(_row("covered", "Ответ.", 0.6)),
        _adapter(_row("missing", "Ответ.", 0.6)),
    ]
    del base[1]["completion_token_count"]

    with pytest.raises(ValueError, match="completion_token_count"):
        compare_rows(base, adapter, bootstrap_samples=100)


def test_semantic_score_requires_valid_paired_provenance() -> None:
    base = [_row("semantic", "Ответ 中", 0.4)]
    adapter = [_adapter(_row("semantic", "Ответ.", 0.6))]
    del base[0]["semantic_score_provenance"]
    with pytest.raises(ValueError, match="semantic_score_provenance"):
        compare_rows(base, adapter, bootstrap_samples=100)

    base = [_row("semantic", "Ответ 中", 0.4)]
    adapter = [_adapter(_row("semantic", "Ответ.", 0.6))]
    adapter[0]["semantic_score_provenance"]["review_artifacts"][1]["sha256"] = "7" * 64
    with pytest.raises(ValueError, match="provenance differs across the pair"):
        compare_rows(base, adapter, bootstrap_samples=100)


def test_comparator_accepts_no_semantic_score_only_without_provenance() -> None:
    base = [_row("unscored", "Ответ 中", None)]
    adapter = [_adapter(_row("unscored", "Ответ.", None))]
    base[0]["semantic_score_provenance"] = {"method": "blinded-human-rubric-v1"}
    with pytest.raises(ValueError, match="must appear together"):
        compare_rows(base, adapter, bootstrap_samples=100)


def test_comparator_rejects_old_pair_schema_and_unbound_cluster() -> None:
    base = [_row("same", "Ответ 中", 0.4)]
    adapter = [_adapter(_row("same", "Ответ.", 0.6))]
    for row in (base[0], adapter[0]):
        row["pair_contract"]["schema_version"] = 2
        row["pair_contract_sha256"] = _canonical_sha256(row["pair_contract"])
    with pytest.raises(ValueError, match="unsupported pair contract schema"):
        compare_rows(base, adapter, bootstrap_samples=100)

    base = [_row("same", "Ответ 中", 0.4)]
    adapter = [_adapter(_row("same", "Ответ.", 0.6))]
    adapter[0]["evaluation_cluster_id"] = "different-cluster"
    with pytest.raises(ValueError, match="evaluation_cluster_id differ"):
        compare_rows(base, adapter, bootstrap_samples=100)


def test_comparator_rejects_surgery_or_unproven_runtime() -> None:
    base = [_row("test-only", "Русский ответ 中", 0.4)]
    adapter = [_adapter(_row("test-only", "Русский ответ.", 0.8))]
    base[0]["generation"]["quality_claim_allowed"] = False
    adapter[0]["generation"]["quality_claim_allowed"] = False

    with pytest.raises(ValueError, match="not a full-model quality oracle"):
        compare_rows(base, adapter, bootstrap_samples=100)

    del base[0]["pair_contract_sha256"]
    with pytest.raises(ValueError, match="pair_contract_sha256"):
        compare_rows(base, adapter, bootstrap_samples=100)


def test_comparator_rejects_surgery_even_if_quality_flag_is_forged() -> None:
    base = [_row("test-only", "Русский ответ 中", 0.4)]
    adapter = [_adapter(_row("test-only", "Русский ответ.", 0.8))]
    pair = deepcopy(base[0]["pair_contract"])
    pair["runtime"]["artifact_contract"]["trainer_base"]["model_id"] = "imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy"
    pair_hash = _canonical_sha256(pair)
    pair_runtime_hash = _canonical_sha256(pair["runtime"])
    for row in (base[0], adapter[0]):
        row["pair_contract"] = deepcopy(pair)
        row["pair_contract_sha256"] = pair_hash
        row["generation"]["pair_runtime_contract_sha256"] = pair_runtime_hash
        row["generation"]["trainer_base"] = pair["runtime"]["artifact_contract"]["trainer_base"]
        row["generation"]["quality_claim_allowed"] = True

    with pytest.raises(ValueError, match="official revision"):
        compare_rows(base, adapter, bootstrap_samples=100)


def test_comparator_cli_writes_hashed_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base_path = tmp_path / "base.jsonl"
    adapter_path = tmp_path / "adapter.jsonl"
    details_path = tmp_path / "details.jsonl"
    base_rows = [_row("paired", "Ответ 中", 0.4)]
    adapter_rows = [_adapter(_row("paired", "Ответ.", 0.7))]
    raw_base = [
        {
            key: value
            for key, value in base_rows[0].items()
            if key not in {"semantic_score", "semantic_score_provenance"}
        }
    ]
    raw_adapter = [
        {
            key: value
            for key, value in adapter_rows[0].items()
            if key not in {"semantic_score", "semantic_score_provenance"}
        }
    ]
    prepared_path = tmp_path / "prepared.json"
    contract_artifacts = [{"ordinal": 0, "filename": "eval_contracts.jsonl", "sha256": "4" * 64}]
    base_predictions_sha256 = jsonl_sha256(raw_base)
    adapter_predictions_sha256 = jsonl_sha256(raw_adapter)
    prepared = {
        "schema_version": 3,
        "status": "BLINDED-REVIEW-PENDING",
        "method": "blinded-human-rubric-v1",
        "count": 1,
        "blinding_key_sha256": "4" * 64,
        "contract_artifacts": contract_artifacts,
        "base_predictions_sha256": base_predictions_sha256,
        "adapter_predictions_sha256": adapter_predictions_sha256,
        "base_generation_bundle": {
            "predictions_sha256": base_predictions_sha256,
            "output_manifest_sha256": "1" * 64,
            "runtime_manifest_sha256": "2" * 64,
        },
        "adapter_generation_bundle": {
            "predictions_sha256": adapter_predictions_sha256,
            "output_manifest_sha256": "3" * 64,
            "runtime_manifest_sha256": "4" * 64,
        },
        "packet_sha256": "5" * 64,
        "rubric": {
            "fields": [
                "meaning_preservation",
                "russian_naturalness",
                "factual_accuracy",
                "instruction_fulfillment",
            ],
            "range": [1, 5],
            "score": "mean((rating - 1) / 4); capped at 0.25 for severe_error",
        },
    }
    prepared_path.write_text(json.dumps(prepared, sort_keys=True) + "\n")
    prepared_sha = hashlib.sha256(prepared_path.read_bytes()).hexdigest()
    review_artifacts = [
        {"ordinal": 0, "filename": "review-1.jsonl", "sha256": "5" * 64},
        {"ordinal": 1, "filename": "review-2.jsonl", "sha256": "6" * 64},
    ]
    adjudication_contract = {
        "schema_version": 3,
        "method": "blinded-human-rubric-v1",
        "count": 1,
        "reviewers": ["reviewer-1", "reviewer-2"],
        "review_artifacts": review_artifacts,
        "blinding_key_sha256": "4" * 64,
        "packet_items_sha256": "3" * 64,
        "prepared_manifest_sha256": prepared_sha,
    }
    adjudication_contract_sha = _canonical_sha256(adjudication_contract)
    for row in (*base_rows, *adapter_rows):
        row["semantic_score_provenance"]["adjudication_contract_sha256"] = adjudication_contract_sha
    base_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in base_rows))
    adapter_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in adapter_rows))
    adjudication_path = tmp_path / "adjudication.json"
    adjudication = {
        "schema_version": 3,
        "status": "BLINDED-REVIEW-COMPLETE",
        "method": "blinded-human-rubric-v1",
        "count": 1,
        "reviewers": ["reviewer-1", "reviewer-2"],
        "review_artifacts": review_artifacts,
        "blinding_key_sha256": "4" * 64,
        "packet_items_sha256": "3" * 64,
        "adjudication_contract": adjudication_contract,
        "adjudication_contract_sha256": adjudication_contract_sha,
        "base_semantic_score_mean": 0.4,
        "adapter_semantic_score_mean": 0.7,
        "contract_artifacts": contract_artifacts,
        "base_input_sha256": base_predictions_sha256,
        "adapter_input_sha256": adapter_predictions_sha256,
        "base_output_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "adapter_output_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
    }
    adjudication_path.write_text(json.dumps(adjudication, sort_keys=True) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_quality_outputs.py",
            str(base_path),
            str(adapter_path),
            "--adjudication-manifest",
            str(adjudication_path),
            "--prepared-review-manifest",
            str(prepared_path),
            "--details",
            str(details_path),
            "--bootstrap-samples",
            "100",
        ],
    )

    main()

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PENDING"
    assert len(result["base_predictions_sha256"]) == 64
    assert len(result["adapter_predictions_sha256"]) == 64
    assert len(result["details_sha256"]) == 64
    assert json.loads(details_path.read_text())["id"] == "paired"

    prepared["base_generation_bundle"]["predictions_sha256"] = "0" * 64
    prepared_path.write_text(json.dumps(prepared, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="base generation bundle does not bind its predictions"):
        main()
