#!/usr/bin/env python3
"""Verify real optimizer updates in tiny GLM-5.3-Flash SFT and GRPO checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

VISION_CONTROL = "model.visual.patch_embed.proj.weight"
LANGUAGE_PREFIX = "model.language_model."
REQUIRED_TRAINING_PATHS = {
    "kda": ("layers.0.self_attn.g_a_proj.weight",),
    "dsa": ("layers.3.self_attn.q_a_proj.weight",),
    "mhc": ("layers.0.hc_attn_base", "layers.0.attn_hc.base"),
    "moe": ("layers.3.mlp.experts.",),
    "router": ("layers.3.mlp.gate.weight",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    sft = subparsers.add_parser("sft")
    sft.add_argument("--base-model", type=Path, required=True)
    sft.add_argument("--checkpoint", type=Path, required=True)
    sft.add_argument("--output", type=Path, default=None)

    rl = subparsers.add_parser("rl")
    rl.add_argument("--base-model", type=Path, required=True)
    rl.add_argument("--checkpoint-one", type=Path, required=True)
    rl.add_argument("--checkpoint-two", type=Path, required=True)
    rl.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    if path.is_file():
        return load_file(path)
    files = sorted(path.glob("*.safetensors"))
    if not files:
        files = sorted(path.glob("**/*.safetensors"))
    if not files:
        raise RuntimeError(f"No safetensors files found under {path}")
    state: dict[str, torch.Tensor] = {}
    for file in files:
        shard = load_file(file)
        overlap = state.keys() & shard.keys()
        if overlap:
            raise RuntimeError(f"Duplicate safetensors keys in {file}: {sorted(overlap)[:10]}")
        state.update(shard)
    return state


def _base_state(model: Path) -> dict[str, torch.Tensor]:
    direct = model / "model.safetensors"
    return _load_safetensors(direct if direct.is_file() else model)


def _transformers_base_state(model: Path) -> dict[str, torch.Tensor]:
    """Load the tiny checkpoint through the same canonicalizer used by FSDP."""
    from transformers import AutoModelForImageTextToText

    loaded = AutoModelForImageTextToText.from_pretrained(
        model,
        dtype="auto",
        low_cpu_mem_usage=True,
    )
    return {name: tensor.detach().cpu() for name, tensor in loaded.state_dict().items()}


def _compare(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], label: str) -> dict[str, object]:
    if set(before) != set(after):
        raise RuntimeError(
            f"{label} state-dict mismatch: missing={sorted(set(before) - set(after))[:20]}, "
            f"unexpected={sorted(set(after) - set(before))[:20]}"
        )
    deltas: list[tuple[float, float, str]] = []
    for name, before_tensor in before.items():
        right = after[name].detach().float()
        if not torch.isfinite(right).all():
            raise RuntimeError(f"{label} produced a non-finite tensor: {name}")
        delta = (right - before_tensor.detach().float()).abs()
        maximum = delta.max().item() if delta.numel() else 0.0
        if maximum:
            deltas.append((maximum, delta.norm().item(), name))
    deltas.sort(reverse=True)
    changed_names = {name for _, _, name in deltas}
    maximum = deltas[0][0] if deltas else 0.0
    if not math.isfinite(maximum):
        raise RuntimeError(f"{label} maximum delta is not finite")
    return {
        "tensor_count": len(after),
        "changed_tensor_count": len(deltas),
        "changed_language_tensor_count": sum(name.startswith(LANGUAGE_PREFIX) for name in changed_names),
        "max_abs_delta": maximum,
        "changed_names": changed_names,
        "top_changed_tensors": [
            {"name": name, "max_abs_delta": max_delta, "delta_norm": norm} for max_delta, norm, name in deltas[:10]
        ],
    }


def _assert_training_paths(report: dict[str, object], label: str) -> dict[str, bool]:
    changed_names = report.pop("changed_names")
    coverage = {
        component: any(any(pattern in name for pattern in patterns) for name in changed_names)
        for component, patterns in REQUIRED_TRAINING_PATHS.items()
    }
    missing = [component for component, changed in coverage.items() if not changed]
    if missing:
        raise RuntimeError(f"{label} did not update required components: {missing}")
    if report["changed_language_tensor_count"] == 0:
        raise RuntimeError(f"{label} did not update language-model tensors")
    return coverage


def _vision_delta(base: dict[str, torch.Tensor], state: dict[str, torch.Tensor]) -> float:
    return (state[VISION_CONTROL].float() - base[VISION_CONTROL].float()).abs().max().item()


def main() -> None:
    args = parse_args()
    if args.mode == "sft":
        base = _base_state(args.base_model)
        checkpoint = _load_safetensors(args.checkpoint)
        update = _compare(base, checkpoint, "sft")
        coverage = _assert_training_paths(update, "sft")
        vision_delta = _vision_delta(base, checkpoint)
        if vision_delta:
            raise RuntimeError(f"SFT changed the frozen vision control by {vision_delta}")
        result = {
            "status": "pass",
            "mode": "sft",
            "update": update,
            "component_coverage": coverage,
            "vision_control": VISION_CONTROL,
            "vision_control_max_abs_delta": vision_delta,
        }
    else:
        # Transformers 5.16 canonicalizes the released GLM checkpoint names
        # while loading (for example hc_attn_* -> attn_hc.*). Compare the FSDP
        # checkpoints against that in-memory contract, not raw shard labels.
        base = _transformers_base_state(args.base_model)
        step_one = torch.load(args.checkpoint_one, map_location="cpu", weights_only=True)
        step_two = torch.load(args.checkpoint_two, map_location="cpu", weights_only=True)
        base_to_one = _compare(base, step_one, "base-to-step-one")
        one_to_two = _compare(step_one, step_two, "step-one-to-step-two")
        coverage = _assert_training_paths(base_to_one, "GRPO step one")
        _assert_training_paths(one_to_two, "GRPO step two")
        vision_one = _vision_delta(base, step_one)
        vision_two = _vision_delta(base, step_two)
        if vision_one or vision_two:
            raise RuntimeError(f"GRPO changed the frozen vision control: step1={vision_one}, step2={vision_two}")
        result = {
            "status": "pass",
            "mode": "rl",
            "base_to_step_one": base_to_one,
            "step_one_to_step_two": one_to_two,
            "component_coverage": coverage,
            "vision_control": VISION_CONTROL,
            "vision_control_max_abs_delta": {"step_one": vision_one, "step_two": vision_two},
        }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
