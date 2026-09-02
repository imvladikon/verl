#!/usr/bin/env python3
"""Verify the exported GLM-5.2 surgery-dummy MLA LoRA adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

TARGET_SHAPES = {
    "q_a_proj": (("rank", 6144), (2048, "rank")),
    "q_b_proj": (("rank", 2048), (16384, "rank")),
    "kv_a_proj_with_mqa": (("rank", 6144), (576, "rank")),
    "kv_b_proj": (("rank", 512), (28672, "rank")),
    "o_proj": (("rank", 16384), (6144, "rank")),
}
ADAPTER_KEY_RE = re.compile(
    r"(?:^|\.)model\.layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<target>q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)\."
    r"lora_(?P<side>[AB])(?:\.default)?\.weight$"
)


def _shape(template: tuple[int | str, ...], rank: int) -> tuple[int, ...]:
    return tuple(rank if value == "rank" else int(value) for value in template)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_paths(path: Path) -> tuple[Path, Path | None, str]:
    if (path / "adapter_config.json").is_file():
        return path, None, "adapter"
    adapter = path / "model" / "huggingface" / "adapter"
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"HF adapter not found below checkpoint: {path}")
    if (path / "lora_train_meta.json").is_file():
        checkpoint_kind = "sft"
    elif (path / "ckpt_contents.json").is_file() and (
        path / "transformer_config.json"
    ).is_file():
        contents = json.loads((path / "ckpt_contents.json").read_text())
        if contents.get("role") != "actor" or not contents.get("backend", {}).get(
            "peft"
        ):
            raise AssertionError("checkpoint is not a PEFT PPO actor")
        checkpoint_kind = "ppo_actor"
    else:
        raise FileNotFoundError(f"unrecognized adapter checkpoint layout: {path}")
    return adapter, path, checkpoint_kind


def verify(path: Path, *, layers: int, rank: int, alpha: int) -> dict:
    adapter_dir, checkpoint_root, checkpoint_kind = resolve_paths(path)
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    if int(config.get("r", -1)) != rank:
        raise AssertionError(f"adapter rank mismatch: {config.get('r')} != {rank}")
    if int(config.get("lora_alpha", -1)) != alpha:
        raise AssertionError(
            f"adapter alpha mismatch: {config.get('lora_alpha')} != {alpha}"
        )
    targets = set(config.get("target_modules") or ())
    if targets != set(TARGET_SHAPES):
        raise AssertionError(f"unexpected target modules: {sorted(targets)}")

    weights_path = adapter_dir / "adapter_model.safetensors"
    observed: dict[tuple[int, str, str], tuple[int, ...]] = {}
    total_elements = 0
    nonzero_b = 0
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for key in keys:
            match = ADAPTER_KEY_RE.search(key)
            if match is None:
                raise AssertionError(f"non-MLA adapter tensor: {key}")
            identity = (
                int(match.group("layer")),
                match.group("target"),
                match.group("side"),
            )
            if identity in observed:
                raise AssertionError(f"duplicate adapter tensor: {identity}")
            tensor = handle.get_tensor(key)
            if tensor.dtype != torch.bfloat16:
                raise AssertionError(f"{key}: expected BF16, got {tensor.dtype}")
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError(f"non-finite adapter tensor: {key}")
            if identity[2] == "B" and bool(torch.count_nonzero(tensor)):
                nonzero_b += 1
            observed[identity] = tuple(tensor.shape)
            total_elements += tensor.numel()

    expected: dict[tuple[int, str, str], tuple[int, ...]] = {}
    for layer in range(layers):
        for target, (a_shape, b_shape) in TARGET_SHAPES.items():
            expected[(layer, target, "A")] = _shape(a_shape, rank)
            expected[(layer, target, "B")] = _shape(b_shape, rank)
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        wrong = sorted(
            (key, observed[key], expected[key])
            for key in set(observed).intersection(expected)
            if observed[key] != expected[key]
        )
        raise AssertionError(
            f"adapter topology mismatch: missing={missing[:8]}, extra={extra[:8]}, "
            f"wrong_shapes={wrong[:8]}"
        )
    expected_b_tensors = layers * len(TARGET_SHAPES)
    if nonzero_b != expected_b_tensors:
        raise AssertionError(
            f"only {nonzero_b}/{expected_b_tensors} LoRA-B tensors changed from zero"
        )

    if checkpoint_root is not None:
        meta_path = checkpoint_root / "lora_train_meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            if int(meta.get("r", -1)) != rank or int(
                meta.get("lora_alpha", -1)
            ) != alpha:
                raise AssertionError(f"checkpoint LoRA metadata mismatch: {meta}")
        dist_dir = checkpoint_root / "model" / "dist_ckpt"
        if not dist_dir.is_dir() or not any(dist_dir.iterdir()):
            raise AssertionError("Megatron adapter dist checkpoint is missing")

    return {
        "adapter_dir": str(adapter_dir.resolve()),
        "checkpoint_kind": checkpoint_kind,
        "layers": layers,
        "targets_per_layer": len(TARGET_SHAPES),
        "tensor_count": len(observed),
        "parameter_count": total_elements,
        "serialized_sha256": sha256(weights_path),
        "all_lora_b_nonzero": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.checkpoint, layers=args.layers, rank=args.rank, alpha=args.alpha),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
