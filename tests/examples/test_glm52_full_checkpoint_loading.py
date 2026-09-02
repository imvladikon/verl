import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from audit_full_checkpoint_loading import audit_checkpoint  # noqa: E402


def write_checkpoint(root: Path) -> None:
    config = {"architectures": ["GlmMoeDsaForCausalLM"]}
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
