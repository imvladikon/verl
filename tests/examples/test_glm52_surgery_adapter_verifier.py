import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from verify_surgery_adapter import TARGET_SHAPES, verify  # noqa: E402


def _shape(template: tuple[int | str, ...], rank: int) -> tuple[int, ...]:
    return tuple(rank if value == "rank" else int(value) for value in template)


def test_verifier_accepts_exact_one_layer_mla_adapter(tmp_path: Path) -> None:
    rank = 2
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    tensors = {}
    for target, (a_shape, b_shape) in TARGET_SHAPES.items():
        prefix = f"base_model.model.model.layers.0.self_attn.{target}"
        tensors[f"{prefix}.lora_A.default.weight"] = torch.ones(
            _shape(a_shape, rank), dtype=torch.bfloat16
        )
        tensors[f"{prefix}.lora_B.default.weight"] = torch.ones(
            _shape(b_shape, rank), dtype=torch.bfloat16
        )
    save_file(tensors, adapter_dir / "adapter_model.safetensors")
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": rank,
                "lora_alpha": 4,
                "target_modules": sorted(TARGET_SHAPES),
            }
        )
    )

    result = verify(adapter_dir, layers=1, rank=rank, alpha=4)
    assert result["tensor_count"] == 10
    assert result["parameter_count"] == 170_112
    assert result["all_lora_b_nonzero"] is True


def test_verifier_rejects_an_unchanged_lora_b_tensor(tmp_path: Path) -> None:
    rank = 1
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    tensors = {}
    for target, (a_shape, b_shape) in TARGET_SHAPES.items():
        prefix = f"base_model.model.model.layers.0.self_attn.{target}"
        tensors[f"{prefix}.lora_A.default.weight"] = torch.ones(
            _shape(a_shape, rank), dtype=torch.bfloat16
        )
        tensors[f"{prefix}.lora_B.default.weight"] = torch.ones(
            _shape(b_shape, rank), dtype=torch.bfloat16
        )
    tensors[
        "base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.default.weight"
    ].zero_()
    save_file(tensors, adapter_dir / "adapter_model.safetensors")
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": rank,
                "lora_alpha": 2,
                "target_modules": sorted(TARGET_SHAPES),
            }
        )
    )

    try:
        verify(adapter_dir, layers=1, rank=rank, alpha=2)
    except AssertionError as error:
        assert "LoRA-B tensors changed" in str(error)
    else:  # pragma: no cover
        raise AssertionError("zero LoRA-B tensor was not rejected")
