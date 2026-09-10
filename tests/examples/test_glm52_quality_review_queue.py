from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from build_quality_dataset import read_jsonl, validate_rows  # noqa: E402
from build_quality_review_queue import (  # noqa: E402
    SOURCE_LOCKS,
    candidate_from_aya,
    candidates_from_oasst,
    deduplicate_and_split,
    write_review_queue,
)


def _labels(quality: float) -> dict[str, list]:
    return {
        "name": ["quality", "fails_task", "lang_mismatch", "toxicity"],
        "value": [quality, 0.0, 0.0, 0.0],
        "count": [3, 3, 3, 3],
    }


def _oasst_root() -> dict:
    return {
        "message_id": "root",
        "parent_id": None,
        "role": "prompter",
        "lang": "ru",
        "text": "Объясни, как проверить резервную копию после создания.",
        "review_result": True,
        "deleted": False,
        "synthetic": False,
    }


def _oasst_answer(message_id: str, *, parent_id: str = "root", quality: float = 0.9) -> dict:
    return {
        "message_id": message_id,
        "parent_id": parent_id,
        "role": "assistant",
        "lang": "ru",
        "text": (
            "Сначала восстановите копию в изолированном окружении, затем "
            "сверьте контрольные суммы и выполните проверку основных файлов."
        ),
        "review_result": True,
        "deleted": False,
        "synthetic": False,
        "rank": 0,
        "review_count": 3,
        "labels": _labels(quality),
    }


def test_aya_candidate_is_filtered_and_keeps_pinned_provenance() -> None:
    candidate, reason = candidate_from_aya(
        {
            "inputs": "Объясни, почему после изменения нужна повторная проверка.",
            "targets": (
                "Повторная проверка подтверждает исправление и помогает обнаружить "
                "новые ошибки до публикации результата."
            ),
            "language_code": "rus",
            "annotation_type": "original-annotations",
        },
        17,
    )
    assert reason is None
    assert candidate is not None
    assert candidate["review"]["status"] == "pending"
    assert candidate["review"]["method"] == "human"
    assert candidate["provenance"]["revision"] == SOURCE_LOCKS["aya"].revision
    assert candidate["provenance"]["source_record_id"] == "row:17"

    rejected, reason = candidate_from_aya(
        {
            "inputs": "Объясни результат проверки.",
            "targets": "Проверка 完成 и результат содержит случайный китайский символ.",
            "language_code": "rus",
        },
        18,
    )
    assert rejected is None
    assert reason == "response_han"


def test_requested_markdown_must_exist_before_a_source_row_enters_review() -> None:
    candidate, reason = candidate_from_aya(
        {
            "inputs": "Составь таблицу со статусами проверки и действиями.",
            "targets": (
                "Статус готовности означает сохранение результата, а ошибка "
                "означает необходимость повторной проверки журнала."
            ),
            "language_code": "rus",
            "annotation_type": "original-annotations",
        },
        2,
    )
    assert candidate is None
    assert reason == "requested_markdown_missing:table"


def test_oasst_keeps_only_human_rated_best_root_replies() -> None:
    root = _oasst_root()
    good = _oasst_answer("good")
    bad_quality = _oasst_answer("bad-quality", quality=0.5)
    followup = _oasst_answer("followup", parent_id="non-root")
    rows = [root, good, bad_quality, followup]
    candidates, rejected = candidates_from_oasst(rows, source_split="train")

    assert [row["provenance"]["source_record_id"] for row in candidates] == ["good"]
    assert rejected["quality_below_0.75"] == 1
    assert rejected["not_root_reply"] == 1


def test_holdout_source_wins_prompt_deduplication() -> None:
    root = _oasst_root()
    train, _ = candidates_from_oasst([root, _oasst_answer("train")], source_split="train")
    holdout, _ = candidates_from_oasst([root, _oasst_answer("validation")], source_split="validation")
    rows, rejected = deduplicate_and_split(train + holdout)

    assert len(rows) == 1
    assert rows[0]["split"] == "test"
    assert rows[0]["provenance"]["source_split"] == "validation"
    assert rejected == Counter({"duplicate_prompt": 1})


def test_written_queue_is_measured_but_cannot_be_used_as_training_data(tmp_path: Path) -> None:
    root = _oasst_root()
    candidates, _ = candidates_from_oasst([root, _oasst_answer("good")], source_split="train")
    rows, _ = deduplicate_and_split(candidates)
    manifest = write_review_queue(
        rows,
        tmp_path,
        source_names=("oasst1",),
        processed=Counter({"oasst1:train": 2}),
        accepted_before_dedupe=Counter({"oasst1:train": 1}),
        rejected=Counter(),
        max_source_rows=0,
        started_at=time.monotonic(),
    )

    assert manifest["queue_count"] == 1
    assert manifest["max_source_rows"] is None
    assert manifest["markdown_required_count"] == 0
    assert manifest["required_block_counts"] == {}
    assert len(manifest["queue_sha256"]) == 64
    queued = read_jsonl(tmp_path / "quality_review_queue.jsonl")
    with pytest.raises(ValueError, match="review status must be accepted"):
        validate_rows(queued)
    on_disk_manifest = json.loads((tmp_path / "review_queue_manifest.json").read_text())
    assert on_disk_manifest["source_locks"]["oasst1"]["license"] == "apache-2.0"
