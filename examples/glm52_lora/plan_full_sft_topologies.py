#!/usr/bin/env python3
"""Compare full GLM-5.2 LoRA topologies without loading checkpoint payloads."""

from __future__ import annotations

import argparse
import json
from math import gcd
from pathlib import Path
from typing import Any

from estimate_full_sft_memory import estimate, parameter_breakdown, require


def lcm(left: int, right: int) -> int:
    return left * right // gcd(left, right)


def parse_candidate(value: str) -> tuple[int, int, int, int, int, int]:
    """Parse WORLD:TP:EP:ETP[:PP[:CP]]."""
    fields = value.split(":")
    require(len(fields) in {4, 5, 6}, f"invalid topology candidate: {value!r}")
    try:
        numbers = tuple(int(field) for field in fields)
    except ValueError as error:
        raise ValueError(f"invalid topology candidate: {value!r}") from error
    require(all(number > 0 for number in numbers), "topology values must be positive")
    return (*numbers, *(1 for _ in range(6 - len(numbers))))


def analyze_topology(
    config: dict[str, Any],
    *,
    world_size: int,
    tp: int,
    ep: int,
    etp: int,
    pp: int = 1,
    cp: int = 1,
    lora_rank: int = 16,
    include_output_layer: bool = False,
    sequence_length: int = 768,
    device_capacity_gib: float,
    minimum_additional_headroom_gib: float = 8.0,
) -> dict[str, Any]:
    for name, value in {
        "world size": world_size,
        "TP": tp,
        "EP": ep,
        "ETP": etp,
        "PP": pp,
        "CP": cp,
    }.items():
        require(value > 0, f"{name} must be positive")
    require(device_capacity_gib > 0, "device capacity must be positive")
    require(minimum_additional_headroom_gib >= 0, "extra headroom cannot be negative")

    dense_grid = tp * pp * cp
    expert_grid = etp * ep * pp
    require(
        world_size % dense_grid == 0,
        f"world size {world_size} is not divisible by dense grid {dense_grid}",
    )
    require(
        world_size % expert_grid == 0,
        f"world size {world_size} is not divisible by expert grid {expert_grid}",
    )
    num_experts = config.get("n_routed_experts")
    require(isinstance(num_experts, int) and num_experts > 0, "invalid expert count")
    require(num_experts % ep == 0, f"{num_experts} experts are not divisible by EP={ep}")

    memory = estimate(
        config,
        tp=tp,
        ep=ep,
        etp=etp,
        lora_rank=lora_rank,
        include_output_layer=include_output_layer,
        sequence_length=sequence_length,
    )
    breakdown = parameter_breakdown(config)
    dense_dp = world_size // dense_grid
    expert_dp = world_size // expert_grid

    # Megatron Bridge currently materializes a requested HF tensor on every
    # owning rank before scattering it. Non-expert tensors are therefore read
    # once per world rank; expert tensors are read once per expert-DP replica.
    nonexpert_parameters = (
        breakdown.tp_replicated_parameters + breakdown.tp_sharded_parameters
    )
    logical_read_bytes = 2 * (
        nonexpert_parameters * world_size
        + breakdown.routed_expert_parameters * expert_dp
    )

    projection = memory["empirical_activation_projection"]
    static_gib = float(memory["static_upper_gib"])
    projected_gib = float(projection["projected_torch_allocated_gib"])
    envelope_gib = float(projection["planning_envelope_gib"])
    headroom_gib = device_capacity_gib - envelope_gib
    if static_gib > device_capacity_gib:
        disposition = "REJECT-STATIC"
    elif projected_gib > device_capacity_gib:
        disposition = "REJECT-PROJECTION"
    elif envelope_gib > device_capacity_gib:
        disposition = "REJECT-ENVELOPE"
    elif headroom_gib < minimum_additional_headroom_gib:
        disposition = "MARGINAL"
    else:
        disposition = "CANDIDATE"

    return {
        "status": "ANALYTIC-NOT-RUNTIME-PROOF",
        "disposition": disposition,
        "topology": {
            "world_size": world_size,
            "tp": tp,
            "ep": ep,
            "etp": etp,
            "pp": pp,
            "cp": cp,
            "dense_dp": dense_dp,
            "expert_dp": expert_dp,
            "experts_per_ep_rank": num_experts // ep,
            "minimum_factor_world_size": lcm(dense_grid, expert_grid),
        },
        "sequence_length": sequence_length,
        "lora_profile": "mla-lm-head" if include_output_layer else "mla-only",
        "device_capacity_gib": device_capacity_gib,
        "minimum_additional_headroom_gib": minimum_additional_headroom_gib,
        "memory": {
            "base_local_bf16_gib": memory["base_local_bf16_gib"],
            "adapter_local_conservative_upper_gib": memory[
                "lora_local_conservative_upper_gib"
            ],
            "static_upper_gib": memory["static_upper_gib"],
            "projected_torch_allocated_gib": projected_gib,
            "planning_envelope_gib": envelope_gib,
            "capacity_headroom_gib": round(headroom_gib, 6),
        },
        "checkpoint_loading": {
            "active_policy_logical_read_bytes": logical_read_bytes,
            "active_policy_logical_read_tib": round(logical_read_bytes / 2**40, 6),
            "nonexpert_read_replication": world_size,
            "expert_read_replication": expert_dp,
            "page_cache_savings_assumed": False,
        },
        "caveats": [
            "capacity must be measured with nvidia-smi on the allocated GPU, not inferred from a pool name",
            "the planning envelope is calibrated from the surgery model and is not full-model runtime proof",
            "checkpoint conversion staging and transient grouped-GEMM workspaces still require a guarded first run",
        ],
    }


def require_candidate_results(candidates: list[dict[str, Any]]) -> None:
    rejected = [
        f"W{candidate['topology']['world_size']}/TP{candidate['topology']['tp']}/"
        f"EP{candidate['topology']['ep']}={candidate['disposition']}"
        for candidate in candidates
        if candidate["disposition"] != "CANDIDATE"
    ]
    require(not rejected, "topology is not a guarded candidate: " + ", ".join(rejected))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="WORLD:TP:EP:ETP[:PP[:CP]]",
    )
    parser.add_argument("--device-capacity-gib", type=float, required=True)
    parser.add_argument("--sequence-length", type=int, default=768)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--include-output-layer", action="store_true")
    parser.add_argument("--minimum-additional-headroom-gib", type=float, default=8.0)
    parser.add_argument("--require-candidate", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    candidates = [
        analyze_topology(
            config,
            world_size=world_size,
            tp=tp,
            ep=ep,
            etp=etp,
            pp=pp,
            cp=cp,
            lora_rank=args.lora_rank,
            include_output_layer=args.include_output_layer,
            sequence_length=args.sequence_length,
            device_capacity_gib=args.device_capacity_gib,
            minimum_additional_headroom_gib=args.minimum_additional_headroom_gib,
        )
        for world_size, tp, ep, etp, pp, cp in map(parse_candidate, args.candidate)
    ]
    if args.require_candidate:
        require_candidate_results(candidates)
    print(json.dumps({"candidates": candidates}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
