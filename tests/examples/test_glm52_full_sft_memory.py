import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from estimate_full_sft_memory import (  # noqa: E402
    EXPECTED_FULL_POLICY_PARAMETERS,
    EXPECTED_SURGERY_POLICY_PARAMETERS,
    estimate,
    lora_breakdown,
    parameter_breakdown,
)


def full_config() -> dict:
    return {
        "hidden_size": 6144,
        "num_hidden_layers": 78,
        "num_attention_heads": 64,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "intermediate_size": 12288,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "vocab_size": 154880,
        "tie_word_embeddings": False,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
        "indexer_types": [
            "full" if layer < 3 or (layer - 2) % 4 == 0 else "shared"
            for layer in range(78)
        ],
    }


def surgery_config() -> dict:
    config = deepcopy(full_config())
    config.update(
        {
            "num_hidden_layers": 10,
            "n_routed_experts": 16,
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 7,
            "indexer_types": [
                "full",
                "full",
                "full",
                "shared",
                "shared",
                "shared",
                "full",
                "shared",
                "shared",
                "shared",
            ],
        }
    )
    return config


def test_full_policy_parameter_count_matches_official_model() -> None:
    result = parameter_breakdown(full_config())
    assert result.policy_parameters == EXPECTED_FULL_POLICY_PARAMETERS
    assert result.routed_expert_parameters == 724_775_731_200
    assert result.tp_replicated_parameters == 1_573_443_840
    assert result.tp_sharded_parameters == 17_027_825_664
    assert result.full_indexers == 21


def test_surgery_breakdown_matches_checkpoint_and_tp2_runtime() -> None:
    result = parameter_breakdown(surgery_config())
    assert result.policy_parameters == EXPECTED_SURGERY_POLICY_PARAMETERS
    assert result.routed_expert_parameters == 4_227_858_432
    assert result.tp_replicated_parameters == 199_548_928
    assert result.tp_sharded_parameters == 4_335_861_760
    assert (
        result.routed_expert_parameters
        + result.tp_replicated_parameters
        + result.tp_sharded_parameters
        == result.policy_parameters
    )

    # Exact base-model numel printed by each rank in the qualified TP2 gate.
    observed_tp2_base_parameters_per_rank = 6_595_338_240
    analytic_tp2 = (
        result.routed_expert_parameters
        + result.tp_replicated_parameters
        + result.tp_sharded_parameters // 2
    )
    assert analytic_tp2 == observed_tp2_base_parameters_per_rank


def test_surgery_breakdown_matches_ep2_runtime() -> None:
    result = parameter_breakdown(surgery_config())
    # Exact base-model numel printed by each rank in the qualified EP2 gate.
    observed_ep2_base_parameters_per_rank = 6_649_339_904
    analytic_ep2 = (
        result.routed_expert_parameters // 2
        + result.tp_replicated_parameters
        + result.tp_sharded_parameters
    )
    assert analytic_ep2 == observed_ep2_base_parameters_per_rank


def test_five_target_lora_counts_and_tp_replication() -> None:
    full = lora_breakdown(full_config(), rank=16, tp=8)
    assert full.global_parameters == 106_149_888
    assert full.tp_replicated_parameters == 18_610_176
    assert full.tp_sharded_parameters == 87_539_712
    assert full.local_parameters == 29_552_640
    assert full.output_layer_parameters == 0

    surgery = lora_breakdown(surgery_config(), rank=16, tp=2)
    assert surgery.global_parameters == 13_608_960
    # Exact trainable numel printed by each rank in the qualified TP2 gate.
    assert surgery.local_parameters == 7_997_440


def test_mla_lm_head_lora_counts_and_tp_replication() -> None:
    full = lora_breakdown(
        full_config(), rank=16, tp=8, include_output_layer=True
    )
    assert full.global_parameters == 108_726_272
    assert full.tp_replicated_parameters == 18_610_176
    assert full.tp_sharded_parameters == 90_116_096
    assert full.local_parameters == 29_874_688
    assert full.output_layer_parameters == 2_576_384

    surgery = lora_breakdown(
        surgery_config(), rank=16, tp=2, include_output_layer=True
    )
    # Exact global trainable numel from the passed 9B export/reload ablation.
    assert surgery.global_parameters == 16_185_344
    assert surgery.local_parameters == 9_285_632


def test_output_layer_lora_requires_tp_divisible_rank() -> None:
    with pytest.raises(ValueError, match="output-layer LoRA rank"):
        lora_breakdown(
            full_config(), rank=10, tp=8, include_output_layer=True
        )


def test_full_tp8_ep32_static_estimate() -> None:
    result = estimate(
        full_config(), tp=8, ep=32, etp=1, lora_rank=16, sequence_length=768
    )
    assert result["base_local_parameters"] == 26_351_163_648
    assert result["base_local_bf16_gib"] == pytest.approx(49.082867, abs=1e-6)
    assert result["lora"]["local_parameters"] == 29_552_640
    assert result["lora_local_16_byte_upper_gib"] == pytest.approx(0.440369, abs=1e-6)
    assert result["static_upper_gib"] == pytest.approx(49.523236, abs=1e-6)
    projection = result["empirical_activation_projection"]
    assert projection["depth_token_scale"] == pytest.approx(7.8)
    assert projection["projected_torch_allocated_gib"] == pytest.approx(
        86.126332, abs=1e-6
    )
    assert projection["planning_envelope_gib"] == pytest.approx(112.427881, abs=1e-6)
    assert projection["runtime_proof"] is False


def test_full_tp8_ep32_mla_lm_head_static_estimate() -> None:
    result = estimate(
        full_config(),
        tp=8,
        ep=32,
        etp=1,
        lora_rank=16,
        include_output_layer=True,
        sequence_length=768,
    )
    assert result["lora"]["global_parameters"] == 108_726_272
    assert result["lora"]["local_parameters"] == 29_874_688
    assert result["lora"]["output_layer_parameters"] == 2_576_384
    assert result["lora_local_16_byte_upper_gib"] == pytest.approx(
        0.445168, abs=1e-6
    )
    assert result["static_upper_gib"] == pytest.approx(49.528035, abs=1e-6)
    projection = result["empirical_activation_projection"]
    assert projection["projected_torch_allocated_gib"] == pytest.approx(
        86.131131, abs=1e-6
    )
    assert projection["planning_envelope_gib"] == pytest.approx(
        112.432679, abs=1e-6
    )


def test_invalid_expert_sharding_fails_closed() -> None:
    with pytest.raises(ValueError, match=r"EP\*ETP"):
        estimate(surgery_config(), tp=1, ep=5, etp=1, lora_rank=16)
