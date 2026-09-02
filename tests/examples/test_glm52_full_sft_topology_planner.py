import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from plan_full_sft_topologies import (  # noqa: E402
    analyze_topology,
    parse_candidate,
    require_candidate_results,
)
from test_glm52_full_sft_memory import full_config  # noqa: E402


def test_parse_candidate_defaults_pp_and_cp() -> None:
    assert parse_candidate("8:8:8:1") == (8, 8, 8, 1, 1, 1)
    assert parse_candidate("32:8:32:1:1:1") == (32, 8, 32, 1, 1, 1)


@pytest.mark.parametrize("candidate", ["8:8:8", "8:8:x:1", "8:8:0:1"])
def test_parse_candidate_fails_closed(candidate: str) -> None:
    with pytest.raises(ValueError):
        parse_candidate(candidate)


def test_eight_high_capacity_gpus_are_an_analytic_seq768_candidate() -> None:
    result = analyze_topology(
        full_config(),
        world_size=8,
        tp=8,
        ep=8,
        etp=1,
        sequence_length=768,
        device_capacity_gib=270,
    )
    assert result["disposition"] == "CANDIDATE"
    assert result["topology"] == {
        "world_size": 8,
        "tp": 8,
        "ep": 8,
        "etp": 1,
        "pp": 1,
        "cp": 1,
        "dense_dp": 1,
        "expert_dp": 1,
        "experts_per_ep_rank": 32,
        "minimum_factor_world_size": 8,
    }
    assert result["memory"]["planning_envelope_gib"] == pytest.approx(228.573231)
    assert result["memory"]["capacity_headroom_gib"] == pytest.approx(41.426769)
    assert result["checkpoint_loading"]["active_policy_logical_read_tib"] == pytest.approx(
        1.589043
    )


def test_sixteen_141_gib_gpus_reject_seq768_envelope() -> None:
    result = analyze_topology(
        full_config(),
        world_size=16,
        tp=8,
        ep=16,
        etp=1,
        sequence_length=768,
        device_capacity_gib=141,
    )
    assert result["disposition"] == "REJECT-ENVELOPE"
    assert result["memory"]["planning_envelope_gib"] == pytest.approx(144.198231)


def test_sixteen_141_gib_gpus_are_seq384_candidate() -> None:
    result = analyze_topology(
        full_config(),
        world_size=16,
        tp=8,
        ep=16,
        etp=1,
        sequence_length=384,
        device_capacity_gib=141,
    )
    assert result["disposition"] == "CANDIDATE"
    assert result["topology"]["dense_dp"] == 2
    assert result["topology"]["expert_dp"] == 1
    assert result["memory"]["planning_envelope_gib"] == pytest.approx(121.954483)


def test_thirty_two_141_gib_gpus_are_seq768_candidate() -> None:
    result = analyze_topology(
        full_config(),
        world_size=32,
        tp=8,
        ep=32,
        etp=1,
        sequence_length=768,
        device_capacity_gib=141,
    )
    assert result["disposition"] == "CANDIDATE"
    assert result["topology"]["dense_dp"] == 4
    assert result["topology"]["expert_dp"] == 1
    assert result["topology"]["experts_per_ep_rank"] == 8
    assert result["memory"]["planning_envelope_gib"] == pytest.approx(102.010731)


def test_output_head_profile_is_accounted_for() -> None:
    mla_only = analyze_topology(
        full_config(),
        world_size=32,
        tp=8,
        ep=32,
        etp=1,
        sequence_length=768,
        device_capacity_gib=141,
    )
    with_head = analyze_topology(
        full_config(),
        world_size=32,
        tp=8,
        ep=32,
        etp=1,
        include_output_layer=True,
        sequence_length=768,
        device_capacity_gib=141,
    )
    assert mla_only["lora_profile"] == "mla-only"
    assert with_head["lora_profile"] == "mla-lm-head"
    assert (
        with_head["memory"]["adapter_local_16_byte_upper_gib"]
        > mla_only["memory"]["adapter_local_16_byte_upper_gib"]
    )
    assert (
        with_head["memory"]["planning_envelope_gib"]
        > mla_only["memory"]["planning_envelope_gib"]
    )


def test_marginal_capacity_is_not_promoted_to_candidate() -> None:
    result = analyze_topology(
        full_config(),
        world_size=32,
        tp=8,
        ep=32,
        etp=1,
        sequence_length=384,
        device_capacity_gib=80,
        minimum_additional_headroom_gib=8,
    )
    assert result["disposition"] == "MARGINAL"
    assert result["memory"]["capacity_headroom_gib"] == pytest.approx(0.233017)


def test_invalid_process_grid_fails_closed() -> None:
    with pytest.raises(ValueError, match="expert grid"):
        analyze_topology(
            full_config(),
            world_size=8,
            tp=8,
            ep=16,
            etp=1,
            device_capacity_gib=270,
        )


def test_runtime_gate_rejects_marginal_candidate() -> None:
    result = analyze_topology(
        full_config(),
        world_size=32,
        tp=8,
        ep=32,
        etp=1,
        sequence_length=384,
        device_capacity_gib=80,
    )
    with pytest.raises(ValueError, match="MARGINAL"):
        require_candidate_results([result])
