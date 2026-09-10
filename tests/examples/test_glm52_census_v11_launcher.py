from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "glm52_lora"
sys.path.insert(0, str(EXAMPLE))

from verify_census_v11_token_audit import EXPECTED_FAMILIES  # noqa: E402
from verify_census_v11_token_audit import validate as validate_tokens  # noqa: E402


def token_report() -> dict:
    overall = {
        "full_chat": {
            "count": 576,
            "min": 54,
            "mean": 174.69097222222223,
            "p50": 141,
            "p90": 358,
            "p95": 418,
            "p99": 504,
            "max": 548,
            "total": 100622,
        },
        "prompt_chat": {
            "count": 576,
            "min": 53,
            "mean": 116.41666666666667,
            "p50": 99,
            "p90": 218,
            "p95": 248,
            "p99": 291,
            "max": 313,
            "total": 67056,
        },
        "target_text": {
            "count": 576,
            "min": 1,
            "mean": 58.27430555555556,
            "p50": 43,
            "p90": 140,
            "p95": 170,
            "p99": 213,
            "max": 235,
            "total": 33566,
        },
    }
    return {
        "input_sha256": "fa6c257c78d2b2c10f8d2a5d0cf9456a88f6e1e1d4e90189a0d7ec4237938658",
        "tokenizer_json_sha256": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
        "tokenizer_config_sha256": "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
        "overall": overall,
        "by_split": {"train": deepcopy(overall)},
        "by_family": {name: {} for name in EXPECTED_FAMILIES},
    }


def test_census_v11_token_audit_is_exact_and_train_only() -> None:
    result = validate_tokens(token_report())
    assert result["status"] == "TOKEN-AUDIT-PASS"
    assert result["full_chat_max_tokens"] == 548
    assert result["configured_max_length"] == 576
    assert result["heldout_rows_used"] == 0

    report = token_report()
    report["by_split"]["validation"] = deepcopy(report["overall"])
    with pytest.raises(SystemExit, match="heldout split"):
        validate_tokens(report)


def test_census_v11_launcher_is_locked_and_keeps_test_out_of_training() -> None:
    launcher = (EXAMPLE / "run_full_sft_census_v11_megatron.sh").read_text()
    core = (EXAMPLE / "run_full_sft_megatron.sh").read_text()
    ablation = (EXAMPLE / "run_full_sft_census_v11_mla_lm_head_megatron.sh").read_text()

    assert "mixture_targeted_wikipedia_v11_train_576" in launcher
    assert "QUALIFICATION_PROFILE=census-v11-quality-576" in launcher
    assert "data.train_max_samples=576" in launcher
    assert "data.val_max_samples=160" in launcher
    assert "MAX_LENGTH=576" in launcher
    assert "REQUIRED_MAX_TOKENS=548" in launcher
    assert "trainer.total_training_steps=9" in launcher
    assert 'TRAIN_FILE="${view_root}/sft_test.parquet"' not in launcher
    assert 'VAL_FILE="${view_root}/sft_test.parquet"' not in launcher
    assert launcher.index('  "$@" \\\n') < launcher.index('  "data.train_files=${view_root}/sft_train.parquet"')
    assert "census-v11-quality-576 requires STEPS=9" in core
    assert "export LORA_PROFILE=mla-lm-head" in ablation
