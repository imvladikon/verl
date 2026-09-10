#!/usr/bin/env python3
"""Plan a bounded-memory GLM-5.2 BF16/FP8 surgery pair.

The target keeps the released width, MLA/DSA geometry, top-k and dense-to-MoE
transition.  It keeps layers 0..4 and reduces each routed MoE from 256 to 32
experts.  Expert medoids are selected from the BF16 router geometry and the
same selection is applied to the official BF16 and FP8 checkpoints.

Only safetensors headers and the two small router tensors per MoE layer are
read.  No complete donor shard or model is loaded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from glm52_surgery_io import (
    DTYPE_BYTES,
    HubRangeReader,
    select_router_medoids,
    sha256_file,
)

BF16_REPOSITORY = "zai-org/GLM-5.2"
BF16_REVISION = "cf457fa734ab149ffef225f80893eb38c6ff5cdc"
FP8_REPOSITORY = "zai-org/GLM-5.2-FP8"
FP8_REVISION = "f33c6dc501ee5a2c7e35155653b1b1abbc320951"
DEFAULT_LAYER_MAP = (0, 1, 2, 3, 4)
TARGET_EXPERTS = 32
TARGET_TOP_K = 8
BLOCK_SIZE = (128, 128)

_LAYER_RE = re.compile(r"^model[.]layers[.](\d+)[.]")
_EXPERT_RE = re.compile(r"[.]mlp[.]experts[.](\d+)[.]")
_CONFIG_LAYER_RE = re.compile(r"model[.]layers[.](\d+)(?=[.]|$)")


def config_diff(left: Any, right: Any, path: str = "config") -> list[str]:
    """Return every structural/value difference with a stable JSON path."""
    if isinstance(left, dict) and isinstance(right, dict):
        result: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left:
                result.append(f"{child}: missing from BF16")
            elif key not in right:
                result.append(f"{child}: missing from FP8")
            else:
                result.extend(config_diff(left[key], right[key], child))
        return result
    if isinstance(left, list) and isinstance(right, list):
        result = []
        if len(left) != len(right):
            result.append(f"{path}: lengths differ ({len(left)} != {len(right)})")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
            result.extend(config_diff(left_item, right_item, f"{path}[{index}]"))
        return result
    return [] if left == right else [f"{path}: {left!r} != {right!r}"]


def parse_layer_map(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("layer map must be comma-separated integers") from error
    return result


def core_contract(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "architectures",
        "model_type",
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "num_attention_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "n_routed_experts",
        "num_experts_per_tok",
        "first_k_dense_replace",
        "moe_layer_freq",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "vocab_size",
    )
    return {key: config.get(key) for key in keys}


def validate_sources(bf16_config: dict[str, Any], fp8_config: dict[str, Any], layer_map: tuple[int, ...]) -> None:
    bf16_unquantized = copy.deepcopy(bf16_config)
    fp8_unquantized = copy.deepcopy(fp8_config)
    bf16_quantization = bf16_unquantized.pop("quantization_config", None)
    fp8_quantization = fp8_unquantized.pop("quantization_config", None)
    if bf16_quantization is not None:
        raise ValueError("official BF16 config unexpectedly declares quantization")
    differences = config_diff(bf16_unquantized, fp8_unquantized)
    if differences:
        raise ValueError(
            "official BF16/FP8 configs differ outside quantization_config:\n" + "\n".join(differences[:20])
        )
    contract = core_contract(bf16_config)
    if contract["model_type"] != "glm_moe_dsa":
        raise ValueError(f"expected glm_moe_dsa, got {contract['model_type']!r}")
    if int(bf16_config["num_hidden_layers"]) != 78:
        raise ValueError("unexpected source depth")
    if int(contract["hidden_size"]) != 6144:
        raise ValueError("unexpected source width")
    if int(contract["n_routed_experts"]) != 256:
        raise ValueError("unexpected source expert count")
    if int(contract["num_experts_per_tok"]) != TARGET_TOP_K:
        raise ValueError("unexpected source top-k")
    if int(contract["first_k_dense_replace"]) != 3:
        raise ValueError("unexpected dense-to-MoE transition")
    if layer_map != DEFAULT_LAYER_MAP:
        raise ValueError("the first qualified surgery pair is exactly layers 0,1,2,3,4")
    quantization = fp8_quantization or {}
    if quantization.get("quant_method") != "fp8":
        raise ValueError("official FP8 config has no FP8 quantization contract")
    if tuple(quantization.get("weight_block_size", ())) != BLOCK_SIZE:
        raise ValueError("unexpected FP8 block size")


def remap_quantization_exclusions(exclusions: list[str], layer_map: tuple[int, ...]) -> list[str]:
    source_to_target = {source: target for target, source in enumerate(layer_map)}
    result: set[str] = set()
    for original in exclusions:
        name = str(original)
        match = _CONFIG_LAYER_RE.search(name)
        if match:
            source_layer = int(match.group(1))
            if source_layer not in source_to_target:
                continue
            target_layer = source_to_target[source_layer]
            name = _CONFIG_LAYER_RE.sub(f"model.layers.{target_layer}", name, count=1)
        result.add(name)
    return sorted(result)


def target_config(source: dict[str, Any], layer_map: tuple[int, ...], *, precision: str) -> dict[str, Any]:
    result = copy.deepcopy(source)
    result.pop("_name_or_path", None)
    result["num_hidden_layers"] = len(layer_map)
    result["n_routed_experts"] = TARGET_EXPERTS
    result["num_experts_per_tok"] = TARGET_TOP_K
    result["num_nextn_predict_layers"] = 0
    result["torch_dtype"] = "bfloat16"
    result["surgery_dummy"] = True
    result["surgery_profile"] = "glm52-5l-h6144-e32-top8"
    result["surgery_precision_role"] = precision
    result["surgery_source_layers"] = list(layer_map)
    result["surgery_source_experts"] = 256
    result["surgery_target_experts"] = TARGET_EXPERTS
    for key in ("indexer_types", "mlp_layer_types", "layer_types"):
        values = source.get(key)
        if values is not None:
            if len(values) != int(source["num_hidden_layers"]):
                raise ValueError(f"source {key} length {len(values)} does not match source depth")
            result[key] = [values[source_layer] for source_layer in layer_map]
    quantization = result.get("quantization_config")
    if precision == "bf16":
        result.pop("quantization_config", None)
    elif quantization:
        quantization["modules_to_not_convert"] = remap_quantization_exclusions(
            list(quantization.get("modules_to_not_convert", [])), layer_map
        )
    else:
        raise ValueError("FP8 target has no quantization config")
    return result


def target_name(source_name: str, target_layer: int, source_layer: int) -> str:
    return source_name.replace(f"model.layers.{source_layer}.", f"model.layers.{target_layer}.", 1)


def select_experts(reader: HubRangeReader, layer_map: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for target_layer, source_layer in enumerate(layer_map):
        if target_layer < 3:
            continue
        prefix = f"model.layers.{source_layer}.mlp.gate"
        selection = select_router_medoids(
            reader.tensor(f"{prefix}.weight"),
            reader.tensor(f"{prefix}.e_score_correction_bias"),
            TARGET_EXPERTS,
        )
        selection["source_layer"] = source_layer
        result[target_layer] = selection
    return result


def model_parameter_name(name: str) -> bool:
    return not name.endswith(("_scale_inv", ".mlp.gate.e_score_correction_bias"))


def build_precision_plan(
    *,
    precision: str,
    repository: str,
    revision: str,
    source_config: dict[str, Any],
    source_index: dict[str, Any],
    reader: HubRangeReader,
    layer_map: tuple[int, ...],
    selections: dict[int, dict[str, Any]],
    pair_id: str,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []

    def direct_entry(source_name: str, rewritten_name: str) -> None:
        location = reader.location(source_name)
        entries.append(
            {
                "kind": "direct",
                "target_name": rewritten_name,
                "dtype": location.dtype,
                "shape": list(location.shape),
                "nbytes": location.nbytes,
                "source": asdict(location),
            }
        )

    source_names = sorted(source_index["weight_map"])
    global_names = [name for name in source_names if not _LAYER_RE.match(name)]
    expected_globals = {
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
    }
    if set(global_names) != expected_globals:
        raise ValueError(f"unexpected global tensor set: {sorted(set(global_names) ^ expected_globals)}")
    for name in global_names:
        direct_entry(name, name)

    for target_layer, source_layer in enumerate(layer_map):
        prefix = f"model.layers.{source_layer}."
        selected = selections.get(target_layer, {}).get("selected_source_experts", [])
        expert_to_target = {int(source_expert): target_expert for target_expert, source_expert in enumerate(selected)}
        for source_name in source_names:
            if not source_name.startswith(prefix):
                continue
            rewritten = target_name(source_name, target_layer, source_layer)
            expert_match = _EXPERT_RE.search(source_name)
            if expert_match:
                source_expert = int(expert_match.group(1))
                if source_expert not in expert_to_target:
                    continue
                rewritten = _EXPERT_RE.sub(
                    f".mlp.experts.{expert_to_target[source_expert]}.",
                    rewritten,
                    count=1,
                )
                direct_entry(source_name, rewritten)
                continue
            if source_name.endswith(".mlp.gate.weight"):
                location = reader.location(source_name)
                entries.append(
                    {
                        "kind": "router_cluster_centroid",
                        "target_name": rewritten,
                        "dtype": location.dtype,
                        "shape": [TARGET_EXPERTS, location.shape[1]],
                        "nbytes": TARGET_EXPERTS * location.shape[1] * DTYPE_BYTES[location.dtype],
                        "source": asdict(location),
                        "target_layer": target_layer,
                    }
                )
                continue
            if source_name.endswith(".mlp.gate.e_score_correction_bias"):
                location = reader.location(source_name)
                entries.append(
                    {
                        "kind": "router_cluster_bias",
                        "target_name": rewritten,
                        "dtype": location.dtype,
                        "shape": [TARGET_EXPERTS],
                        "nbytes": TARGET_EXPERTS * DTYPE_BYTES[location.dtype],
                        "source": asdict(location),
                        "target_layer": target_layer,
                    }
                )
                continue
            direct_entry(source_name, rewritten)

    names = [entry["target_name"] for entry in entries]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate target tensors")
    model_parameters = sum(math.prod(entry["shape"]) for entry in entries if model_parameter_name(entry["target_name"]))
    serialized_elements = sum(math.prod(entry["shape"]) for entry in entries)
    serialized_bytes = sum(int(entry["nbytes"]) for entry in entries)
    repository_suffix = "" if precision == "bf16" else "-FP8"
    config = target_config(source_config, layer_map, precision=precision)
    config["surgery_pair_id"] = pair_id
    target = {
        "repository": (f"imvladikon/GLM-5.2-5L-32E-Surgery-Dummy{repository_suffix}"),
        "profile": "5l-h6144-e32-top8",
        "precision_role": precision,
        "model_parameter_count": model_parameters,
        "serialized_elements_including_fp8_scales_and_router_buffers": serialized_elements,
        "serialized_bytes": serialized_bytes,
    }
    if precision == "fp8-rollout":
        target["fp8_weight_count"] = sum(entry["dtype"] == "F8_E4M3" for entry in entries)
    return {
        "schema_version": 1,
        "status": "planned test-only surgery; weights not built",
        "pair_id": pair_id,
        "source": {
            "repository": repository,
            "revision": revision,
        },
        "target": target,
        "layer_selection": {
            "method": "first-three-dense-plus-first-two-moe",
            "source_layers": list(layer_map),
        },
        "expert_selection": {str(layer): value for layer, value in selections.items()},
        "contract": {
            "purpose": "SFT/RL/LoRA sharding, gradient, checkpoint and hot-sync validation",
            "not_for": ["chat", "benchmark", "quality evaluation"],
            "hidden_size": int(source_config["hidden_size"]),
            "source_layers": int(source_config["num_hidden_layers"]),
            "target_layers": len(layer_map),
            "source_experts": int(source_config["n_routed_experts"]),
            "target_experts": TARGET_EXPERTS,
            "top_k": TARGET_TOP_K,
            "fp8_weight_block_size": list(BLOCK_SIZE),
        },
        "config": config,
        "tensors": sorted(entries, key=lambda item: item["target_name"]),
    }


def logical_tensor_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["target_name"]: entry for entry in plan["tensors"] if not entry["target_name"].endswith("_scale_inv")}


def validate_pair(bf16_plan: dict[str, Any], fp8_plan: dict[str, Any]) -> None:
    bf16 = logical_tensor_map(bf16_plan)
    fp8 = logical_tensor_map(fp8_plan)
    if set(bf16) != set(fp8):
        raise ValueError(
            f"BF16/FP8 logical tensor sets differ: "
            f"bf16_only={len(set(bf16) - set(fp8))}, "
            f"fp8_only={len(set(fp8) - set(bf16))}"
        )
    for name in bf16:
        if bf16[name]["shape"] != fp8[name]["shape"]:
            raise ValueError(f"BF16/FP8 shape mismatch for {name}")

    fp8_all = {entry["target_name"]: entry for entry in fp8_plan["tensors"]}
    scale_count = 0
    for name, entry in fp8.items():
        scale_name = f"{name}_scale_inv"
        if entry["dtype"] == "F8_E4M3":
            if scale_name not in fp8_all:
                raise ValueError(f"missing FP8 scale for {name}")
            shape = entry["shape"]
            if len(shape) != 2:
                raise ValueError(f"block-FP8 tensor is not a matrix: {name}")
            expected = [math.ceil(shape[0] / 128), math.ceil(shape[1] / 128)]
            scale = fp8_all[scale_name]
            if scale["dtype"] != "F32" or scale["shape"] != expected:
                raise ValueError(f"bad FP8 scale grid for {name}: {scale['shape']} != {expected}")
            scale_count += 1
        elif scale_name in fp8_all:
            raise ValueError(f"non-FP8 tensor unexpectedly has a scale: {name}")
    if scale_count == 0:
        raise ValueError("FP8 plan contains no quantized matrices")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def surgery_pair_id(
    *,
    bf16_repository: str,
    bf16_revision: str,
    bf16_config_sha256: str,
    bf16_index_sha256: str,
    fp8_repository: str,
    fp8_revision: str,
    fp8_config_sha256: str,
    fp8_index_sha256: str,
    layer_map: tuple[int, ...],
    selections: dict[int, dict[str, Any]],
) -> str:
    """Bind the pair identity to immutable inputs and the exact surgery result."""
    identity_selection = {
        str(layer): {
            "method": selection["method"],
            "selected_source_experts": selection["selected_source_experts"],
            "source_clusters": selection["source_clusters"],
        }
        for layer, selection in sorted(selections.items())
    }
    seed = {
        "schema_version": 1,
        "bf16": {
            "repository": bf16_repository,
            "revision": bf16_revision,
            "config_sha256": bf16_config_sha256,
            "index_sha256": bf16_index_sha256,
        },
        "fp8": {
            "repository": fp8_repository,
            "revision": fp8_revision,
            "config_sha256": fp8_config_sha256,
            "index_sha256": fp8_index_sha256,
        },
        "layers": list(layer_map),
        "experts": TARGET_EXPERTS,
        "top_k": TARGET_TOP_K,
        "expert_selection": identity_selection,
    }
    encoded = json.dumps(seed, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "glm52-5l32e-" + hashlib.sha256(encoded).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-config", type=Path, required=True)
    parser.add_argument("--bf16-index", type=Path, required=True)
    parser.add_argument("--fp8-config", type=Path, required=True)
    parser.add_argument("--fp8-index", type=Path, required=True)
    parser.add_argument("--bf16-repository", default=BF16_REPOSITORY)
    parser.add_argument("--bf16-revision", default=BF16_REVISION)
    parser.add_argument("--fp8-repository", default=FP8_REPOSITORY)
    parser.add_argument("--fp8-revision", default=FP8_REVISION)
    parser.add_argument("--layer-map", type=parse_layer_map, default=DEFAULT_LAYER_MAP)
    parser.add_argument(
        "--bf16-output",
        type=Path,
        default=Path("plans/glm52_5l32e_surgery_bf16.json"),
    )
    parser.add_argument(
        "--fp8-output",
        type=Path,
        default=Path("plans/glm52_5l32e_surgery_fp8.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bf16_config = json.loads(args.bf16_config.read_text(encoding="utf-8"))
    bf16_index = json.loads(args.bf16_index.read_text(encoding="utf-8"))
    fp8_config = json.loads(args.fp8_config.read_text(encoding="utf-8"))
    fp8_index = json.loads(args.fp8_index.read_text(encoding="utf-8"))
    validate_sources(bf16_config, fp8_config, args.layer_map)

    bf16_reader = HubRangeReader(args.bf16_repository, args.bf16_revision, bf16_index)
    fp8_reader = HubRangeReader(args.fp8_repository, args.fp8_revision, fp8_index)
    selections = select_experts(bf16_reader, args.layer_map)
    source_hashes = {
        "bf16_config": sha256_file(args.bf16_config),
        "bf16_index": sha256_file(args.bf16_index),
        "fp8_config": sha256_file(args.fp8_config),
        "fp8_index": sha256_file(args.fp8_index),
    }
    pair_id = surgery_pair_id(
        bf16_repository=args.bf16_repository,
        bf16_revision=args.bf16_revision,
        bf16_config_sha256=source_hashes["bf16_config"],
        bf16_index_sha256=source_hashes["bf16_index"],
        fp8_repository=args.fp8_repository,
        fp8_revision=args.fp8_revision,
        fp8_config_sha256=source_hashes["fp8_config"],
        fp8_index_sha256=source_hashes["fp8_index"],
        layer_map=args.layer_map,
        selections=selections,
    )

    bf16_plan = build_precision_plan(
        precision="bf16",
        repository=args.bf16_repository,
        revision=args.bf16_revision,
        source_config=bf16_config,
        source_index=bf16_index,
        reader=bf16_reader,
        layer_map=args.layer_map,
        selections=selections,
        pair_id=pair_id,
    )
    fp8_plan = build_precision_plan(
        precision="fp8-rollout",
        repository=args.fp8_repository,
        revision=args.fp8_revision,
        source_config=fp8_config,
        source_index=fp8_index,
        reader=fp8_reader,
        layer_map=args.layer_map,
        selections=selections,
        pair_id=pair_id,
    )
    for plan, config_digest, index_digest in (
        (bf16_plan, source_hashes["bf16_config"], source_hashes["bf16_index"]),
        (fp8_plan, source_hashes["fp8_config"], source_hashes["fp8_index"]),
    ):
        plan["source"]["config_sha256"] = config_digest
        plan["source"]["index_sha256"] = index_digest

    validate_pair(bf16_plan, fp8_plan)
    atomic_json(args.bf16_output, bf16_plan)
    atomic_json(args.fp8_output, fp8_plan)
    print(
        json.dumps(
            {
                "pair_id": pair_id,
                "bf16_output": str(args.bf16_output.resolve()),
                "fp8_output": str(args.fp8_output.resolve()),
                "bf16_target": bf16_plan["target"],
                "fp8_target": fp8_plan["target"],
                "source_layers": list(args.layer_map),
                "selected_experts": {
                    str(layer): selection["selected_source_experts"] for layer, selection in selections.items()
                },
                "headers_read": {
                    "bf16": len(bf16_reader._headers),
                    "fp8": len(fp8_reader._headers),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
