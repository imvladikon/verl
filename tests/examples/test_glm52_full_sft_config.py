import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from verify_full_sft_config import (  # noqa: E402
    EXPECTED_EP_GATE_SHA256,
    EXPECTED_TP_GATE_SHA256,
    compute_parallel_topology,
    file_count,
)


def valid_topology_config() -> dict:
    return {
        "engine": {
            "tensor_model_parallel_size": 8,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "sequence_parallel": True,
        },
        "trainer": {"nnodes": 8, "n_gpus_per_node": 8},
    }


def test_full_topology_uses_independent_dense_and_expert_grids() -> None:
    result = compute_parallel_topology(valid_topology_config())
    assert result == {
        "nodes": 8,
        "gpus_per_node": 8,
        "world_size": 64,
        "tp": 8,
        "ep": 32,
        "etp": 1,
        "pp": 1,
        "cp": 1,
        "dense_dp": 8,
        "expert_dp": 2,
        "experts_per_ep_rank": 8,
    }


def test_full_topology_rejects_invalid_dense_grid() -> None:
    config = deepcopy(valid_topology_config())
    config["engine"]["tensor_model_parallel_size"] = 3
    with pytest.raises(SystemExit, match=r"dense TP\*PP\*CP grid 3"):
        compute_parallel_topology(config)


def test_full_topology_rejects_invalid_expert_grid() -> None:
    config = deepcopy(valid_topology_config())
    config["engine"]["expert_tensor_parallel_size"] = 3
    with pytest.raises(SystemExit, match=r"expert ETP\*EP\*PP grid 96"):
        compute_parallel_topology(config)


def test_full_topology_rejects_nonintegral_expert_ownership() -> None:
    config = deepcopy(valid_topology_config())
    config["engine"]["expert_model_parallel_size"] = 10
    config["trainer"] = {"nnodes": 10, "n_gpus_per_node": 8}
    with pytest.raises(
        SystemExit, match="256 routed experts are not divisible by EP=10"
    ):
        compute_parallel_topology(config)


def test_full_topology_requires_sequence_parallel_for_tp_and_ep() -> None:
    config = deepcopy(valid_topology_config())
    config["engine"]["sequence_parallel"] = False
    with pytest.raises(SystemExit, match=r"TP\+EP requires sequence parallel"):
        compute_parallel_topology(config)


def test_full_launch_requires_exact_validated_gate_roots() -> None:
    launcher = (
        ROOT / "examples" / "glm52_lora" / "run_full_sft_megatron.sh"
    ).read_text(encoding="utf-8")
    assert EXPECTED_TP_GATE_SHA256 in launcher
    assert EXPECTED_EP_GATE_SHA256 in launcher
    assert (
        '"${TP_ADAPTER_GATE_SHA:-}" != "${expected_tp_adapter_gate_sha256}"' in launcher
    )
    assert (
        '"${EP_ROUTING_GATE_SHA:-}" != "${expected_ep_routing_gate_sha256}"' in launcher
    )


def test_file_count_normalizes_single_hydra_path() -> None:
    assert file_count("/data/train.parquet", label="train") == 1
    assert file_count(["/data/a.parquet", "/data/b.parquet"], label="train") == 2


@pytest.mark.parametrize("value", [None, "", ["/data/train.parquet", ""]])
def test_file_count_rejects_invalid_values(value) -> None:
    with pytest.raises(SystemExit, match="CONFIG-FAIL"):
        file_count(value, label="train")
