#!/usr/bin/env python3
"""Analytic GLM-5.2 parameter sharding and LoRA static-memory estimate.

The calculation follows the Megatron-Core layout used by the GLM bridge.  It
does not instantiate the model and therefore is safe to run against a
``config.json`` copied from the 1.5-TB checkpoint.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

GIB = 2**30
MIB = 2**20
EXPECTED_FULL_POLICY_PARAMETERS = 743_377_000_704
EXPECTED_SURGERY_POLICY_PARAMETERS = 8_763_269_120
SURGERY_MLA_R16_PARAMETERS = 13_608_960
SURGERY_ANCHOR_LAYERS = 10
SURGERY_ANCHOR_SEQUENCE_LENGTH = 768
SURGERY_ANCHOR_PEAK_ALLOCATED_GIB = 20.328


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def config_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{key} must be a positive integer, got {value!r}",
    )
    return value


@dataclass(frozen=True)
class ParameterBreakdown:
    policy_parameters: int
    routed_expert_parameters: int
    tp_replicated_parameters: int
    tp_sharded_parameters: int
    attention_parameters: int
    dsa_indexer_parameters: int
    dense_mlp_parameters: int
    shared_expert_parameters: int
    router_parameters: int
    embedding_and_head_parameters: int
    final_norm_parameters: int
    policy_layers: int
    dense_layers: int
    moe_layers: int
    full_indexers: int


@dataclass(frozen=True)
class LoraBreakdown:
    global_parameters: int
    tp_replicated_parameters: int
    tp_sharded_parameters: int
    local_parameters: int
    rank: int


def parameter_breakdown(config: dict[str, Any]) -> ParameterBreakdown:
    """Count the policy model and split it by Megatron parameter domain."""
    hidden = config_int(config, "hidden_size")
    layers = config_int(config, "num_hidden_layers")
    heads = config_int(config, "num_attention_heads")
    q_lora = config_int(config, "q_lora_rank")
    kv_lora = config_int(config, "kv_lora_rank")
    qk_nope = config_int(config, "qk_nope_head_dim")
    qk_rope = config_int(config, "qk_rope_head_dim")
    value_head = config_int(config, "v_head_dim")
    dense_intermediate = config_int(config, "intermediate_size")
    moe_intermediate = config_int(config, "moe_intermediate_size")
    routed_experts = config_int(config, "n_routed_experts")
    shared_experts = config_int(config, "n_shared_experts")
    vocab = config_int(config, "vocab_size")

    mlp_types = config.get("mlp_layer_types")
    require(
        isinstance(mlp_types, list) and len(mlp_types) == layers,
        "invalid mlp_layer_types",
    )
    require(set(mlp_types) <= {"dense", "sparse"}, "unknown MLP layer type")
    dense_layers = mlp_types.count("dense")
    moe_layers = mlp_types.count("sparse")

    indexer_types = config.get("indexer_types")
    require(
        isinstance(indexer_types, list) and len(indexer_types) == layers,
        "invalid indexer_types",
    )
    require(set(indexer_types) <= {"full", "shared"}, "unknown DSA indexer type")
    full_indexers = indexer_types.count("full")
    index_heads = config_int(config, "index_n_heads")
    index_head_dim = config_int(config, "index_head_dim")

    # MLA projections plus both transformer-layer norms.  q_a/kv_a and their
    # norms are duplicated by Megatron's TELinear path; q_b/kv_b/o are TP
    # sharded.  This split is independently checked against the observed TP2
    # surgery-model parameter count in the test suite.
    duplicated_mla_per_layer = (
        hidden * q_lora + q_lora + hidden * (kv_lora + qk_rope) + kv_lora + 2 * hidden
    )
    sharded_mla_per_layer = (
        q_lora * heads * (qk_nope + qk_rope)
        + kv_lora * heads * (qk_nope + value_head)
        + heads * value_head * hidden
    )
    attention_parameters = layers * (duplicated_mla_per_layer + sharded_mla_per_layer)

    # A full DSA indexer contains wq_b, wk, a LayerNorm with weight+bias, and
    # weights_proj.  All four are explicit parallel_mode="duplicated" modules.
    dsa_per_full_indexer = (
        q_lora * index_heads * index_head_dim
        + hidden * index_head_dim
        + 2 * index_head_dim
        + hidden * index_heads
    )
    dsa_indexer_parameters = full_indexers * dsa_per_full_indexer

    dense_mlp_parameters = dense_layers * 3 * hidden * dense_intermediate
    routed_expert_parameters = (
        moe_layers * routed_experts * 3 * hidden * moe_intermediate
    )
    shared_expert_parameters = (
        moe_layers * shared_experts * 3 * hidden * moe_intermediate
    )
    router_parameters = moe_layers * hidden * routed_experts
    embedding_copies = 1 if config.get("tie_word_embeddings") is True else 2
    embedding_and_head_parameters = embedding_copies * vocab * hidden
    final_norm_parameters = hidden

    tp_replicated_parameters = (
        layers * duplicated_mla_per_layer
        + dsa_indexer_parameters
        + router_parameters
        + final_norm_parameters
    )
    tp_sharded_parameters = (
        layers * sharded_mla_per_layer
        + dense_mlp_parameters
        + shared_expert_parameters
        + embedding_and_head_parameters
    )
    policy_parameters = (
        routed_expert_parameters + tp_replicated_parameters + tp_sharded_parameters
    )
    require(
        policy_parameters
        == attention_parameters
        + dsa_indexer_parameters
        + dense_mlp_parameters
        + routed_expert_parameters
        + shared_expert_parameters
        + router_parameters
        + embedding_and_head_parameters
        + final_norm_parameters,
        "internal parameter-accounting mismatch",
    )
    return ParameterBreakdown(
        policy_parameters=policy_parameters,
        routed_expert_parameters=routed_expert_parameters,
        tp_replicated_parameters=tp_replicated_parameters,
        tp_sharded_parameters=tp_sharded_parameters,
        attention_parameters=attention_parameters,
        dsa_indexer_parameters=dsa_indexer_parameters,
        dense_mlp_parameters=dense_mlp_parameters,
        shared_expert_parameters=shared_expert_parameters,
        router_parameters=router_parameters,
        embedding_and_head_parameters=embedding_and_head_parameters,
        final_norm_parameters=final_norm_parameters,
        policy_layers=layers,
        dense_layers=dense_layers,
        moe_layers=moe_layers,
        full_indexers=full_indexers,
    )


def lora_breakdown(config: dict[str, Any], *, rank: int, tp: int) -> LoraBreakdown:
    """Count the five-target MLA LoRA and its local TP representation."""
    require(rank > 0 and tp > 0, "LoRA rank and TP must be positive")
    hidden = config_int(config, "hidden_size")
    layers = config_int(config, "num_hidden_layers")
    heads = config_int(config, "num_attention_heads")
    q_lora = config_int(config, "q_lora_rank")
    kv_lora = config_int(config, "kv_lora_rank")
    qk_nope = config_int(config, "qk_nope_head_dim")
    qk_rope = config_int(config, "qk_rope_head_dim")
    value_head = config_int(config, "v_head_dim")

    replicated_dimensions = hidden + q_lora + hidden + kv_lora + qk_rope
    sharded_dimensions = (
        q_lora
        + heads * (qk_nope + qk_rope)
        + kv_lora
        + heads * (qk_nope + value_head)
        + heads * value_head
        + hidden
    )
    replicated = layers * rank * replicated_dimensions
    sharded = layers * rank * sharded_dimensions
    require(sharded % tp == 0, f"TP={tp} does not evenly shard MLA LoRA parameters")
    return LoraBreakdown(
        global_parameters=replicated + sharded,
        tp_replicated_parameters=replicated,
        tp_sharded_parameters=sharded,
        local_parameters=replicated + sharded // tp,
        rank=rank,
    )


def estimate(
    config: dict[str, Any],
    *,
    tp: int,
    ep: int,
    etp: int,
    lora_rank: int,
    sequence_length: int | None = None,
    base_bytes_per_parameter: int = 2,
    conservative_adapter_bytes_per_parameter: int = 16,
) -> dict[str, Any]:
    require(tp > 0 and ep > 0 and etp > 0, "TP, EP, and ETP must be positive")
    breakdown = parameter_breakdown(config)
    require(
        breakdown.routed_expert_parameters % (ep * etp) == 0,
        "routed experts do not divide evenly over EP*ETP",
    )
    require(
        breakdown.tp_sharded_parameters % tp == 0,
        "non-expert parameters do not divide evenly over TP",
    )
    base_local = (
        breakdown.routed_expert_parameters // (ep * etp)
        + breakdown.tp_replicated_parameters
        + breakdown.tp_sharded_parameters // tp
    )
    adapter = lora_breakdown(config, rank=lora_rank, tp=tp)
    base_bytes = base_local * base_bytes_per_parameter
    adapter_upper_bytes = (
        adapter.local_parameters * conservative_adapter_bytes_per_parameter
    )
    static_upper_bytes = base_bytes + adapter_upper_bytes
    result: dict[str, Any] = {
        "status": "ANALYTIC-STATIC-ONLY",
        "topology": {"tp": tp, "ep": ep, "etp": etp},
        "parameter_breakdown": asdict(breakdown),
        "base_local_parameters": base_local,
        "base_local_bf16_gib": round(base_bytes / GIB, 6),
        "lora": asdict(adapter),
        "lora_global_bf16_mib": round(adapter.global_parameters * 2 / MIB, 6),
        "lora_local_16_byte_upper_gib": round(adapter_upper_bytes / GIB, 6),
        "static_upper_gib": round(static_upper_bytes / GIB, 6),
        "not_included": [
            "activations",
            "CUDA context and libraries",
            "Transformer Engine and communication workspaces",
            "allocator fragmentation",
            "checkpoint conversion staging",
        ],
    }
    if sequence_length is not None:
        require(sequence_length > 0, "sequence length must be positive")
        surgery_static_gib = (
            EXPECTED_SURGERY_POLICY_PARAMETERS * base_bytes_per_parameter
            + SURGERY_MLA_R16_PARAMETERS * conservative_adapter_bytes_per_parameter
        ) / GIB
        anchor_nonstatic_gib = SURGERY_ANCHOR_PEAK_ALLOCATED_GIB - surgery_static_gib
        require(anchor_nonstatic_gib > 0, "invalid surgery memory anchor")
        scale = (
            breakdown.policy_layers
            * sequence_length
            / (SURGERY_ANCHOR_LAYERS * SURGERY_ANCHOR_SEQUENCE_LENGTH)
        )
        projected_nonstatic_gib = anchor_nonstatic_gib * scale
        result["empirical_activation_projection"] = {
            "sequence_length": sequence_length,
            "anchor": {
                "model": "GLM-5.2-9B-LoRA-Surgery-Dummy",
                "layers": SURGERY_ANCHOR_LAYERS,
                "sequence_length": SURGERY_ANCHOR_SEQUENCE_LENGTH,
                "torch_peak_allocated_gib": SURGERY_ANCHOR_PEAK_ALLOCATED_GIB,
                "analytic_static_gib": round(surgery_static_gib, 6),
                "residual_gib": round(anchor_nonstatic_gib, 6),
            },
            "depth_token_scale": round(scale, 6),
            "projected_nonstatic_gib": round(projected_nonstatic_gib, 6),
            "projected_torch_allocated_gib": round(
                static_upper_bytes / GIB + projected_nonstatic_gib, 6
            ),
            "planning_envelope_gib": round(
                static_upper_bytes / GIB + 1.5 * projected_nonstatic_gib + 8.0, 6
            ),
            "planning_envelope_assumptions": [
                "1.5x the linear depth-token projection",
                "8 GiB reserved for non-PyTorch CUDA and communication workspaces",
            ],
            "runtime_proof": False,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--ep", type=int, default=32)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--expect-policy-parameters", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = estimate(
        config,
        tp=args.tp,
        ep=args.ep,
        etp=args.etp,
        lora_rank=args.lora_rank,
        sequence_length=args.sequence_length,
    )
    if args.expect_policy_parameters is not None:
        actual = result["parameter_breakdown"]["policy_parameters"]
        require(
            actual == args.expect_policy_parameters,
            f"policy parameter count drift: {actual} != {args.expect_policy_parameters}",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
