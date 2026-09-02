import json
from enum import Enum

from verl.workers.rollout.sglang_rollout.utils import normalize_peft_config_for_sglang


class _PeftType(Enum):
    LORA = "LORA"


def test_peft_config_is_json_serializable_and_deterministic():
    config = {
        "peft_type": _PeftType.LORA,
        "task_type": "CAUSAL_LM",
        "target_modules": {"q_b_proj", "q_a_proj"},
        "target_parameters": set(),
        "nested": {"values": {3, 1, 2}},
    }

    normalized = normalize_peft_config_for_sglang(config)

    assert normalized["peft_type"] == "LORA"
    assert normalized["target_modules"] == ["q_a_proj", "q_b_proj"]
    assert normalized["target_parameters"] == []
    assert normalized["nested"]["values"] == [1, 2, 3]
    assert json.loads(json.dumps(normalized)) == normalized


def test_all_linear_target_stays_a_string():
    normalized = normalize_peft_config_for_sglang(
        {"peft_type": "LORA", "target_modules": "all-linear", "target_parameters": set()}
    )

    assert normalized["target_modules"] == "all-linear"
