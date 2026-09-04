# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FULL_VERL = ROOT / "examples" / "glm52_lora"
sys.path.insert(0, str(FULL_VERL))

from verify_full_sft_config import validate_clean_v4_padding_contract


def test_clean_v4_launcher_disables_cross_microbatch_bshd_padding() -> None:
    launcher = (FULL_VERL / "run_full_sft_clean_v4_megatron.sh").read_text(encoding="utf-8")

    assert "data.micro_batch_size_per_gpu=1" in launcher
    assert "data.pad_mode=no_padding" not in launcher  # inherited from the core launcher
    assert "engine.pad_bshd_to_minibatch_max=false" in launcher
    assert launcher.index('  "$@" \\\n') < launcher.index("  engine.pad_bshd_to_minibatch_max=false \\\n")


def test_clean_v4_validator_locks_the_padding_contract() -> None:
    config = {
        "data": {
            "pad_mode": "no_padding",
            "micro_batch_size_per_gpu": 1,
            "use_dynamic_bsz": False,
        },
        "engine": {"pad_bshd_to_minibatch_max": False},
    }

    assert validate_clean_v4_padding_contract(config) == {
        "dataset_padding": "none",
        "bshd_padding_scope": "one-example-microbatch",
        "max_length_role": "fail-closed-upper-bound",
    }


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("data", "pad_mode", "right", "requires no_padding"),
        ("data", "micro_batch_size_per_gpu", 2, "one example per micro-batch"),
        ("data", "use_dynamic_bsz", True, "forbids multi-example"),
        ("engine", "pad_bshd_to_minibatch_max", True, "at their own length"),
    ],
)
def test_clean_v4_validator_rejects_padding_contract_drift(
    section: str, field: str, value: object, message: str
) -> None:
    config = {
        "data": {
            "pad_mode": "no_padding",
            "micro_batch_size_per_gpu": 1,
            "use_dynamic_bsz": False,
        },
        "engine": {"pad_bshd_to_minibatch_max": False},
    }
    config[section][field] = value

    with pytest.raises(SystemExit, match=message):
        validate_clean_v4_padding_contract(config)
