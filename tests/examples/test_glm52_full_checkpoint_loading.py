import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

import audit_full_checkpoint_loading as checkpoint_audit  # noqa: E402

audit_checkpoint = checkpoint_audit.audit_checkpoint


def write_checkpoint(root: Path) -> None:
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": 2,
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    entries = {
        "model.embed_tokens.weight": {
            "dtype": "BF16",
            "shape": [4, 2],
            "data_offsets": [0, 16],
        },
        "model.layers.1.mlp.experts.0.down_proj.weight": {
            "dtype": "BF16",
            "shape": [2, 2],
            "data_offsets": [16, 24],
        },
        "model.layers.1.mlp.experts.0.gate_proj.weight": {
            "dtype": "BF16",
            "shape": [2, 2],
            "data_offsets": [24, 32],
        },
        "model.layers.1.mlp.experts.0.up_proj.weight": {
            "dtype": "BF16",
            "shape": [2, 2],
            "data_offsets": [32, 40],
        },
        "model.layers.1.mlp.gate.e_score_correction_bias": {
            "dtype": "F32",
            "shape": [2],
            "data_offsets": [40, 48],
        },
    }
    raw_header = json.dumps(entries).encode("utf-8")
    shard = root / "model-00001-of-00001.safetensors"
    with shard.open("wb") as output:
        output.write(struct.pack("<Q", len(raw_header)))
        output.write(raw_header)
        output.truncate(output.tell() + 48)

    index = {
        "metadata": {"total_size": 48},
        "weight_map": {key: shard.name for key in entries},
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def write_fp8_checkpoint(root: Path, *, include_scale: bool = True) -> None:
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": 1,
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["model.embed_tokens"],
        },
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    entries = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": {
            "dtype": "F8_E4M3",
            "shape": [2, 2],
            "data_offsets": [0, 4],
        },
        "model.embed_tokens.weight": {
            "dtype": "BF16",
            "shape": [4, 2],
            "data_offsets": [8 if include_scale else 4, 24 if include_scale else 20],
        },
    }
    if include_scale:
        entries["model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv"] = {
            "dtype": "F32",
            "shape": [1, 1],
            "data_offsets": [4, 8],
        }
    raw_header = json.dumps(entries).encode("utf-8")
    shard = root / "model-00001-of-00001.safetensors"
    payload_bytes = 24 if include_scale else 20
    with shard.open("wb") as output:
        output.write(struct.pack("<Q", len(raw_header)))
        output.write(raw_header)
        output.truncate(output.tell() + payload_bytes)
    index = {
        "metadata": {"total_size": payload_bytes},
        "weight_map": {key: shard.name for key in entries},
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def test_audit_measures_expert_and_dense_read_amplification(tmp_path: Path) -> None:
    write_checkpoint(tmp_path)
    result = audit_checkpoint(
        tmp_path,
        world_size=4,
        tp=2,
        ep=2,
        etp=1,
        pp=1,
        cp=1,
        official_glm52=False,
        bridge_revision="test",
    )

    assert result["status"] == "CHECKPOINT-LOAD-AUDIT-PASS"
    assert result["checkpoint"]["expert_tensor_count"] == 3
    assert result["checkpoint"]["expert_bytes"] == 24
    assert result["checkpoint"]["nonexpert_bytes"] == 24
    assert result["checkpoint"]["dtype_counts"] == {"BF16": 4, "F32": 1}
    assert result["checkpoint"]["dtype_bytes"] == {"BF16": 40, "F32": 8}
    assert result["checkpoint"]["auxiliary_tensor_count"] == 0
    assert result["policy_import"]["tensor_count"] == 5
    assert result["policy_import"]["payload_bytes"] == 48
    assert result["source_working_set"]["largest_tensor_bytes"] == 16
    assert result["source_working_set"]["largest_gate_up_bundle_bytes"] == 16
    assert result["bridge_import"]["nonexpert_read_factor"] == 4
    assert result["bridge_import"]["expert_read_factor"] == 2
    assert result["bridge_import"]["logical_total_read_bytes"] == 144


def test_audit_rejects_header_index_disagreement(tmp_path: Path) -> None:
    write_checkpoint(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"]["unexpected.weight"] = "model-00001-of-00001.safetensors"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(SystemExit, match="header/index key mismatch"):
        audit_checkpoint(
            tmp_path,
            world_size=4,
            tp=2,
            ep=2,
            etp=1,
            pp=1,
            cp=1,
            official_glm52=False,
            bridge_revision="test",
        )


def test_audit_excludes_disabled_mtp_layer_from_policy_reads(tmp_path: Path) -> None:
    write_checkpoint(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["num_hidden_layers"] = 1
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = audit_checkpoint(
        tmp_path,
        world_size=4,
        tp=2,
        ep=2,
        etp=1,
        pp=1,
        cp=1,
        official_glm52=False,
        bridge_revision="test",
    )

    assert result["checkpoint"]["auxiliary_tensor_count"] == 4
    assert result["checkpoint"]["auxiliary_bytes"] == 32
    assert result["policy_import"]["tensor_count"] == 1
    assert result["policy_import"]["payload_bytes"] == 16
    assert result["policy_import"]["expert_tensor_count"] == 0
    assert result["bridge_import"]["logical_total_read_bytes"] == 64
    assert result["bridge_import"]["whole_checkpoint_upper_bound_bytes"] == 144


def test_synthetic_config_hash_is_stable(tmp_path: Path) -> None:
    write_checkpoint(tmp_path)
    expected = hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest()
    result = audit_checkpoint(
        tmp_path,
        world_size=4,
        tp=2,
        ep=2,
        etp=1,
        pp=1,
        cp=1,
        official_glm52=False,
        bridge_revision="test",
    )
    assert result["checkpoint"]["config_sha256"] == expected


def test_audit_validates_fp8_scale_contract_and_bf16_destination(tmp_path: Path) -> None:
    write_fp8_checkpoint(tmp_path)
    result = audit_checkpoint(
        tmp_path,
        world_size=4,
        tp=2,
        ep=2,
        etp=1,
        pp=1,
        cp=1,
        official_glm52=False,
        bridge_revision="test",
    )

    assert result["checkpoint"]["fp8_weight_count"] == 1
    assert result["checkpoint"]["scale_count"] == 1
    assert result["checkpoint"]["excluded_module_count"] == 1
    assert result["policy_import"]["scale_count"] == 1
    assert result["policy_import"]["destination_bf16_bytes"] == 24
    assert result["bridge_import"]["logical_total_read_bytes"] == 80


def test_audit_rejects_fp8_weight_without_scale(tmp_path: Path) -> None:
    write_fp8_checkpoint(tmp_path, include_scale=False)
    with pytest.raises(SystemExit, match="FP8 weights without scales"):
        audit_checkpoint(
            tmp_path,
            world_size=4,
            tp=2,
            ep=2,
            etp=1,
            pp=1,
            cp=1,
            official_glm52=False,
            bridge_revision="test",
        )


def test_official_fp8_profile_locks_checkpoint_and_bridge_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_fp8_checkpoint(tmp_path)
    config_sha = hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest()
    index_sha = hashlib.sha256(
        (tmp_path / "model.safetensors.index.json").read_bytes()
    ).hexdigest()
    expected = {
        "EXPECTED_FP8_CONFIG_SHA256": config_sha,
        "EXPECTED_FP8_INDEX_SHA256": index_sha,
        "EXPECTED_FP8_TOTAL_BYTES": 24,
        "EXPECTED_FP8_TENSOR_COUNT": 3,
        "EXPECTED_FP8_SHARD_COUNT": 1,
        "EXPECTED_FP8_DTYPE_COUNTS": {"BF16": 1, "F32": 1, "F8_E4M3": 1},
        "EXPECTED_FP8_DTYPE_BYTES": {"BF16": 16, "F32": 4, "F8_E4M3": 4},
        "EXPECTED_FP8_WEIGHT_COUNT": 1,
        "EXPECTED_FP8_SCALE_COUNT": 1,
        "EXPECTED_FP8_EXCLUDED_MODULE_COUNT": 1,
        "EXPECTED_FP8_EXPERT_TENSOR_COUNT": 2,
        "EXPECTED_FP8_EXPERT_BYTES": 8,
        "EXPECTED_FP8_AUXILIARY_TENSOR_COUNT": 0,
        "EXPECTED_FP8_AUXILIARY_BYTES": 0,
        "EXPECTED_FP8_POLICY_TENSOR_COUNT": 3,
        "EXPECTED_FP8_POLICY_BYTES": 24,
        "EXPECTED_FP8_POLICY_SCALE_COUNT": 1,
        "EXPECTED_FP8_POLICY_EXPERT_TENSOR_COUNT": 2,
        "EXPECTED_FP8_POLICY_EXPERT_BYTES": 8,
        "EXPECTED_FP8_POLICY_DESTINATION_BF16_BYTES": 24,
        "EXPECTED_FP8_LARGEST_TENSOR_BYTES": 16,
        "EXPECTED_FP8_BRIDGE_BASE_REVISION": "base",
        "EXPECTED_FP8_BRIDGE_PATCH_SHA256": "patch",
    }
    for name, value in expected.items():
        monkeypatch.setattr(checkpoint_audit, name, value)

    result = audit_checkpoint(
        tmp_path,
        world_size=4,
        tp=2,
        ep=2,
        etp=1,
        pp=1,
        cp=1,
        official_glm52=False,
        official_profile="fp8-dequant",
        bridge_revision="overlay-head",
        bridge_base_revision="base",
        bridge_patch_sha256="patch",
    )
    assert result["official_profile"] == "fp8-dequant"
    assert result["bridge_import"]["base_revision"] == "base"
    assert result["bridge_import"]["dequantization_patch_sha256"] == "patch"

    with pytest.raises(SystemExit, match="dequantization patch drift"):
        audit_checkpoint(
            tmp_path,
            world_size=4,
            tp=2,
            ep=2,
            etp=1,
            pp=1,
            cp=1,
            official_glm52=False,
            official_profile="fp8-dequant",
            bridge_revision="overlay-head",
            bridge_base_revision="base",
            bridge_patch_sha256="wrong",
        )
