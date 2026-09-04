# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
QUALITY = ROOT / "examples" / "glm52_lora"
sys.path.insert(0, str(QUALITY))

from audit_split_isolation import (
    audit_rows,
    file_sha256,
    load_source_samples,
    shingles,
    source_index_from_records,
)


def row(example_id: str, split: str, prompt: str, response: str) -> dict:
    return {
        "id": example_id,
        "split": split,
        "prompt": prompt,
        "response": response,
        "provenance": {},
    }


def test_clean_splits_pass() -> None:
    result = audit_rows(
        [
            row(
                "a",
                "train",
                "Опиши северное сияние",
                "Свечение возникает в верхней атмосфере.",
            ),
            row(
                "b",
                "validation",
                "Составь план выпуска",
                "Проверь тесты, документацию и версию.",
            ),
            row(
                "c",
                "test",
                "Объясни резервное копирование",
                "Копия позволяет восстановить данные.",
            ),
        ]
    )

    assert result["status"] == "PASS"
    assert not any(result["counts"].values())
    assert result["algorithm"] == {
        "matching": ("exact-and-exhaustive-cross-split-shingle-jaccard-containment-or-token-containment"),
        "minimum_sequence_tokens": 5,
        "near_duplicate_threshold": 0.7,
        "shingle_width": 5,
        "version": "exhaustive-cross-split-source-visible-v4",
    }
    assert len(result["canonical_rows_sha256"]) == 64
    assert len(result["auditor_code_sha256"]) == 64


def test_exact_target_reuse_fails_even_with_distinct_ids_and_prompts() -> None:
    result = audit_rows(
        [
            row("a", "train", "Первый запрос", "Один и тот же развёрнутый ответ."),
            row("b", "test", "Совсем другой запрос", "Один  и тот же развёрнутый ответ."),
        ]
    )

    assert result["status"] == "FAIL-SPLIT-ISOLATION"
    assert result["counts"]["exact_response_groups"] == 1
    assert result["violations"]["exact_response_groups"][0]["pair_counts"] == {"train-test": 1}
    assert result["violations"]["exact_response_groups"][0]["id_count"] == 2
    assert result["violations"]["exact_response_groups"][0]["sample_ids"] == [
        "a",
        "b",
    ]


def test_reused_source_text_and_near_duplicate_prompt_fail() -> None:
    first = row(
        "a",
        "train",
        "Исправь аккуратно этот длинный русский технический текст про резервную копию сегодня",
        "Первый ответ не совпадает с другим.",
    )
    second = row(
        "b",
        "validation",
        "Исправь аккуратно этот длинный русский технический текст про резервную копию завтра",
        "Второй ответ тоже отличается.",
    )
    first["provenance"] = {"source_text_sha256": "a" * 64}
    second["provenance"] = {"source_text_sha256": "a" * 64}

    result = audit_rows([first, second], near_threshold=0.7)

    assert result["status"] == "FAIL-SPLIT-ISOLATION"
    assert result["counts"]["source_text_groups"] == 1
    assert result["counts"]["near_prompt_pairs"] == 1


def test_exhaustive_near_match_does_not_drop_large_postings() -> None:
    common = " ".join(f"слово{index}" for index in range(100))
    rows = [
        row(
            f"train-{index}",
            "train",
            f"{common} вариант{index}",
            f"Уникальный ответ номер {index}.",
        )
        for index in range(65)
    ]
    rows.append(
        row(
            "validation",
            "validation",
            f"{common} контроль",
            "Отдельный проверочный ответ.",
        )
    )

    result = audit_rows(rows)

    assert result["status"] == "FAIL-SPLIT-ISOLATION"
    assert result["counts"]["near_prompt_pairs"] == 65
    assert len(result["violations"]["near_prompt_pairs"]) == 8
    assert result["stored_violation_limit_per_category"] == 8


def test_identical_shingle_sets_are_retained_at_similarity_one() -> None:
    assert shingles("альфа бета") == shingles("бета альфа")
    result = audit_rows(
        [
            row("train", "train", "альфа бета", "Первый отдельный ответ"),
            row("test", "test", "бета альфа", "Второй отдельный ответ"),
        ]
    )

    assert result["counts"]["exact_prompt_groups"] == 0
    assert result["counts"]["near_prompt_pairs"] == 1
    assert result["violations"]["near_prompt_pairs"][0]["similarity"] == 1.0


def test_invisible_format_controls_cannot_hide_cross_split_reuse() -> None:
    prompt = "Одинаковый русский запрос про проверку данных"
    disguised = "\u200b".join(prompt)

    result = audit_rows(
        [
            row("train", "train", prompt, "Первый отдельный ответ"),
            row("test", "test", disguised, "Второй отдельный ответ"),
        ]
    )

    assert result["status"] == "FAIL-SPLIT-ISOLATION"
    assert result["counts"]["exact_prompt_groups"] == 1


def test_default_production_threshold_is_point_seven() -> None:
    left = "один два три четыре пять шесть семь восемь девять десять"
    right = "один два три четыре пять шесть семь восемь девять другой"
    result = audit_rows(
        [
            row("train", "train", left, "Уникальный первый ответ"),
            row("validation", "validation", right, "Уникальный второй ответ"),
        ]
    )

    assert result["near_duplicate_threshold"] == 0.7
    assert result["counts"]["near_prompt_pairs"] == 1


def test_short_sentence_with_one_replaced_word_is_near_duplicate() -> None:
    result = audit_rows(
        [
            row(
                "train",
                "train",
                "Проверка завершена успешно без дополнительных ошибок.",
                "Первый отдельный ответ",
            ),
            row(
                "validation",
                "validation",
                "Проверка завершена успешно без критических ошибок.",
                "Второй отдельный ответ",
            ),
        ]
    )

    assert result["status"] == "FAIL-SPLIT-ISOLATION"
    assert result["counts"]["near_prompt_pairs"] == 1


def test_exact_and_near_prompt_response_cross_field_reuse_fail() -> None:
    common = " ".join(f"лексема{index}" for index in range(100))
    result = audit_rows(
        [
            row(
                "train",
                "train",
                "Точный текст, который нельзя переносить в целевой ответ.",
                f"{common} обучающий",
            ),
            row(
                "validation",
                "validation",
                f"{common} проверочный",
                "Точный текст, который нельзя переносить в целевой ответ.",
            ),
        ]
    )

    assert result["status"] == "FAIL-SPLIT-ISOLATION"
    assert result["counts"]["exact_prompt_response_groups"] == 1
    assert result["counts"]["near_prompt_response_pairs"] == 2
    assert result["violations"]["near_prompt_response_pairs"][0]["similarity"] == 1.0


def sourced_row(
    example_id: str,
    split: str,
    source_record_id: str,
    source_text_sha256: str,
) -> dict:
    value = row(
        example_id,
        split,
        f"Запрос для {example_id}",
        f"Ответ для {example_id}",
    )
    value["provenance"] = {
        "dataset": "example/source",
        "revision": "revision-1",
        "source_record_id": source_record_id,
        "source_split": "ru",
        "source_text_sha256": source_text_sha256,
    }
    return value


def source_record(source_record_id: str, source_text_sha256: str, content: str) -> dict:
    return {
        "config": "ru",
        "dataset": "example/source",
        "id": source_record_id,
        "revision": "revision-1",
        "sentences": [content],
        "text_sha256": source_text_sha256,
        "title": f"Источник {source_record_id}",
    }


def test_external_source_near_reuse_fails() -> None:
    common = " ".join(f"источник{index}" for index in range(100))
    source_index = source_index_from_records(
        [
            source_record("one", "a" * 64, f"{common} первый"),
            source_record("two", "b" * 64, f"{common} второй"),
        ]
    )

    result = audit_rows(
        [
            sourced_row("train", "train", "one:variant", "a" * 64),
            sourced_row("validation", "validation", "two:variant", "b" * 64),
        ],
        source_index=source_index,
        required_source_datasets={"example/source"},
    )

    assert result["status"] == "FAIL-SPLIT-ISOLATION"
    assert result["counts"]["near_source_pairs"] == 1
    assert result["source_coverage"]["resolved_required_row_count"] == 2


def test_source_record_identity_reuse_ignores_revision_drift() -> None:
    first = sourced_row("train", "train", "same-record", "a" * 64)
    second = sourced_row("validation", "validation", "same-record", "b" * 64)
    second["provenance"]["revision"] = "revision-2"

    result = audit_rows([first, second])

    assert result["status"] == "FAIL-SPLIT-ISOLATION"
    assert result["counts"]["source_record_groups"] == 1


def test_external_source_hash_mismatch_fails() -> None:
    source_index = source_index_from_records([source_record("one", "a" * 64, "Полный проверяемый исходный текст.")])

    result = audit_rows(
        [sourced_row("train", "train", "one:variant", "b" * 64)],
        source_index=source_index,
        required_source_datasets={"example/source"},
    )

    assert result["status"] == "FAIL-SPLIT-ISOLATION"
    assert result["counts"]["source_hash_mismatches"] == 1


def test_required_external_source_coverage_is_fail_closed() -> None:
    result = audit_rows(
        [sourced_row("train", "train", "missing:variant", "a" * 64)],
        source_index=source_index_from_records([]),
        required_source_datasets={"example/source"},
    )

    assert result["status"] == "FAIL-SPLIT-ISOLATION"
    assert result["counts"]["source_configuration_gaps"] == 1
    assert result["counts"]["source_coverage_gaps"] == 1
    assert result["source_coverage"]["required_row_count"] == 1
    assert result["source_coverage"]["resolved_required_row_count"] == 0


def test_external_source_file_hash_is_bound(tmp_path: Path) -> None:
    sample = tmp_path / "source.jsonl"
    sample.write_text(
        json.dumps(
            source_record("one", "a" * 64, "Полный проверяемый исходный текст."),
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    digest = file_sha256(sample)

    source_index = load_source_samples([("example/source", sample, digest)])
    result = audit_rows(
        [sourced_row("train", "train", "one:variant", "a" * 64)],
        source_index=source_index,
        required_source_datasets={"example/source"},
    )

    assert result["status"] == "PASS"
    assert result["source_coverage"]["files"] == [
        {
            "dataset": "example/source",
            "file": "source.jsonl",
            "record_count": 1,
            "sha256": digest,
        }
    ]

    try:
        load_source_samples([("example/source", sample, "0" * 64)])
    except ValueError as error:
        assert "source sample SHA-256 mismatch" in str(error)
    else:
        raise AssertionError("source sample with the wrong hash was accepted")

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty source sample"):
        load_source_samples([("example/source", empty, file_sha256(empty))])


@pytest.mark.parametrize(
    ("missing_field", "message"),
    [
        ("revision", "missing revision"),
        ("config", "missing config"),
        ("text_sha256", "invalid text SHA-256"),
        ("content", "missing content"),
    ],
)
def test_external_source_metadata_is_mandatory(missing_field: str, message: str) -> None:
    record = source_record("one", "a" * 64, "Проверяемое содержимое.")
    if missing_field == "content":
        record.pop("sentences")
    else:
        record.pop(missing_field)

    with pytest.raises(ValueError, match=message):
        source_index_from_records([record])


def test_required_source_name_typo_and_zero_rows_fail_closed() -> None:
    source_index = source_index_from_records([source_record("one", "a" * 64, "Проверяемое содержимое.")])
    result = audit_rows(
        [sourced_row("train", "train", "one", "a" * 64)],
        source_index=source_index,
        required_source_datasets={"example/sorce"},
    )

    assert result["status"] == "FAIL-SPLIT-ISOLATION"
    assert result["counts"]["source_configuration_gaps"] == 2
    assert result["source_coverage"]["input_dataset_row_counts"] == {"example/sorce": 0}
    assert result["source_coverage"]["source_dataset_record_counts"] == {"example/sorce": 0}


def test_required_source_with_zero_input_rows_fails_closed() -> None:
    source_index = source_index_from_records([source_record("one", "a" * 64, "Проверяемое содержимое.")])
    result = audit_rows(
        [row("train", "train", "Запрос без внешнего источника", "Отдельный ответ")],
        source_index=source_index,
        required_source_datasets={"example/source"},
    )

    assert result["status"] == "FAIL-SPLIT-ISOLATION"
    assert result["counts"]["source_configuration_gaps"] == 1
    assert result["violations"]["source_configuration_gaps"] == [
        {
            "dataset": "example/source",
            "reason": "required dataset has zero input rows",
        }
    ]


def test_required_row_provenance_metadata_is_mandatory() -> None:
    value = sourced_row("train", "train", "one", "a" * 64)
    value["provenance"].pop("revision")
    value["provenance"].pop("source_split")
    value["provenance"].pop("source_text_sha256")
    result = audit_rows(
        [value],
        source_index=source_index_from_records([source_record("one", "a" * 64, "Проверяемое содержимое.")]),
        required_source_datasets={"example/source"},
    )

    assert result["status"] == "FAIL-SPLIT-ISOLATION"
    assert result["violations"]["source_metadata_gaps"] == [
        {
            "fields": ["revision", "source_split", "source_text_sha256"],
            "id": "train",
        }
    ]


def test_external_source_content_is_compared_to_other_split_visible_fields() -> None:
    content = "один два три четыре пять шесть семь восемь девять десять"
    source_index = source_index_from_records(
        [
            {
                "config": "ru",
                "dataset": "example/source",
                "id": "one",
                "revision": "revision-1",
                "text": content,
                "text_sha256": "a" * 64,
            }
        ]
    )
    train = sourced_row("train", "train", "one", "a" * 64)
    validation = row(
        "validation",
        "validation",
        content,
        "Уникальный проверочный ответ",
    )
    test = row("test", "test", "Уникальный тестовый запрос", content)

    result = audit_rows(
        [train, validation, test],
        source_index=source_index,
        required_source_datasets={"example/source"},
    )

    assert result["counts"]["near_source_prompt_pairs"] == 1
    assert result["counts"]["near_source_response_pairs"] == 1
    assert result["violations"]["near_source_prompt_pairs"][0]["similarity"] == 1.0
    assert result["violations"]["near_source_response_pairs"][0]["similarity"] == 1.0


def test_one_source_sentence_cannot_hide_inside_a_longer_article() -> None:
    copied = "один два три четыре пять шесть семь восемь девять десять"
    source_index = source_index_from_records(
        [
            {
                "config": "ru",
                "dataset": "example/source",
                "id": "one",
                "revision": "revision-1",
                "sentences": [
                    copied,
                    "Совершенно другое длинное предложение для увеличения статьи.",
                    "Ещё один независимый фрагмент исходного материала.",
                ],
                "text_sha256": "a" * 64,
            }
        ]
    )
    train = sourced_row("train", "train", "one", "a" * 64)
    validation = row(
        "validation",
        "validation",
        copied,
        "Отдельный проверочный ответ",
    )

    result = audit_rows(
        [train, validation],
        source_index=source_index,
        required_source_datasets={"example/source"},
    )

    assert result["counts"]["near_source_prompt_pairs"] == 1
    assert result["violations"]["near_source_prompt_pairs"][0]["similarity"] == 1.0


def test_source_hash_with_misspelled_dataset_cannot_bypass_coverage() -> None:
    source_index = source_index_from_records([source_record("one", "a" * 64, "Проверяемое содержимое источника.")])
    correct = sourced_row("train", "train", "one", "a" * 64)
    typo = sourced_row("validation", "validation", "missing", "b" * 64)
    typo["provenance"]["dataset"] = "example/sorce"

    result = audit_rows(
        [correct, typo],
        source_index=source_index,
        required_source_datasets={"example/source"},
    )

    assert result["status"] == "FAIL-SPLIT-ISOLATION"
    assert result["counts"]["source_coverage_gaps"] == 1
    assert result["source_coverage"]["required_row_count"] == 2


def test_misspelled_external_dataset_cannot_bypass_by_omitting_hash() -> None:
    source_index = source_index_from_records([source_record("one", "a" * 64, "Проверяемое содержимое источника.")])
    correct = sourced_row("train", "train", "one", "a" * 64)
    typo = sourced_row("validation", "validation", "missing", "b" * 64)
    typo["provenance"]["dataset"] = "example/sorce"
    typo["provenance"].pop("source_text_sha256")

    result = audit_rows(
        [correct, typo],
        source_index=source_index,
        required_source_datasets={"example/source"},
    )

    assert result["status"] == "FAIL-SPLIT-ISOLATION"
    assert result["counts"]["source_coverage_gaps"] == 1


def test_source_hash_requires_external_index_without_opt_in_flag() -> None:
    result = audit_rows(
        [sourced_row("train", "train", "one", "a" * 64)],
    )

    assert result["status"] == "FAIL-SPLIT-ISOLATION"
    assert result["counts"]["source_coverage_gaps"] == 1
