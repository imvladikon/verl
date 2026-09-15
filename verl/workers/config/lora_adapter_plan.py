# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model-aware trainer/rollout LoRA contracts."""

import hashlib
import json
import re
from typing import Any, Optional


GLM5_NEXT_DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "b_proj",
    "f_a_proj",
    "f_b_proj",
    "g_a_proj",
    "g_b_proj",
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
GLM5_NEXT_DEFAULT_ROLLOUT_TARGETS = (
    "qkv_proj",
    "o_proj",
    "b_proj",
    "f_a_proj",
    "f_b_proj",
    "g_a_proj",
    "g_b_proj",
    "fused_qkv_a_proj_with_mqa",
    "q_b_proj",
    "kv_b_proj",
    "gate_up_proj",
    "down_proj",
)
GLM5_NEXT_EXCLUDED_LORA_TARGETS = (
    "model.visual",
    "indexer.wq_b",
    "indexer.wk",
    "indexer.weights_proj",
    "mlp.gate",
    "embed_tokens",
    "lm_head",
)
GLM5_NEXT_VISUAL_EXCLUDE_PATTERN = r"(?:.*\.)?visual(?:\..*)?"


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _trainer_exclude_pattern(exclude_modules: Any) -> str:
    """Add the GLM visual subtree while preserving PEFT exclusion semantics."""
    patterns = [GLM5_NEXT_VISUAL_EXCLUDE_PATTERN]
    if exclude_modules is None:
        pass
    elif isinstance(exclude_modules, str):
        if exclude_modules:
            patterns.insert(0, exclude_modules)
    elif isinstance(exclude_modules, (list, tuple)):
        if not all(isinstance(item, str) and item for item in exclude_modules):
            raise TypeError("exclude_modules entries must be non-empty strings")
        if exclude_modules:
            suffixes = "|".join(re.escape(item) for item in exclude_modules)
            patterns.insert(0, rf"(?:.*\.)?(?:{suffixes})")
    else:
        raise TypeError(
            "exclude_modules must be a regex string or a list of module suffixes, "
            f"but got {type(exclude_modules).__name__}"
        )
    return "|".join(f"(?:{pattern})" for pattern in patterns)


def _lora_dimension(
    name: str,
    *,
    hidden_size: int,
    kda_heads: int,
    kda_head_dim: int,
    attention_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_head_dim: int,
    qk_rope_head_dim: int,
    value_head_dim: int,
    intermediate_size: int,
) -> list[int]:
    kda_projection = kda_heads * kda_head_dim
    dimensions = {
        "q_proj": (hidden_size, kda_projection),
        "k_proj": (hidden_size, kda_projection),
        "v_proj": (hidden_size, kda_projection),
        "b_proj": (hidden_size, kda_heads),
        "f_a_proj": (hidden_size, kda_head_dim),
        "f_b_proj": (kda_head_dim, kda_projection),
        "g_a_proj": (hidden_size, kda_head_dim),
        "g_b_proj": (kda_head_dim, kda_projection),
        "q_a_proj": (hidden_size, q_lora_rank),
        "q_b_proj": (q_lora_rank, attention_heads * qk_head_dim),
        "kv_a_proj_with_mqa": (
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
        ),
        "kv_b_proj": (
            kv_lora_rank,
            attention_heads * (qk_head_dim + value_head_dim),
        ),
        "o_proj_kda": (kda_projection, hidden_size),
        "o_proj_dsa": (attention_heads * value_head_dim, hidden_size),
        "gate_proj": (hidden_size, intermediate_size),
        "up_proj": (hidden_size, intermediate_size),
        "down_proj": (intermediate_size, hidden_size),
    }
    return list(dimensions[name])


def build_glm5_next_lora_adapter_plan(
    hf_config: Any,
    target_modules: Any,
    *,
    rank: int = 0,
    alpha: int = 0,
    exclude_modules: Any = None,
) -> Optional[dict[str, Any]]:
    """Resolve the GLM-5.3-Flash ``all-linear`` shorthand without model I/O.

    PEFT discovers logical HF linears while SGLang allocates packed serving
    modules. This binds both views to one layer-aware geometry and keeps the
    discrete DSA indexer plus embeddings/output head opt-in.
    """
    text_config = _config_value(hf_config, "text_config", hf_config)
    model_types = {
        _config_value(hf_config, "model_type"),
        _config_value(text_config, "model_type"),
    }
    if not ({"glm5_next", "glm5_next_text"} & model_types):
        return None
    if target_modules != "all-linear":
        return None

    trainer_exclude_modules = _trainer_exclude_pattern(exclude_modules)

    hidden_size = int(_config_value(text_config, "hidden_size"))
    num_layers = int(_config_value(text_config, "num_hidden_layers"))
    layer_types = list(_config_value(text_config, "layer_types", ()))
    if len(layer_types) != num_layers:
        raise ValueError(
            "GLM-5.3-Flash LoRA planning requires one layer_types entry per "
            f"decoder layer, got {len(layer_types)} for {num_layers} layers"
        )

    linear_config = _config_value(text_config, "linear_attn_config", {})
    kda_heads = int(_config_value(linear_config, "num_heads"))
    kda_head_dim = int(_config_value(linear_config, "head_dim"))
    attention_heads = int(_config_value(text_config, "num_attention_heads"))
    q_lora_rank = int(_config_value(text_config, "q_lora_rank"))
    kv_lora_rank = int(_config_value(text_config, "kv_lora_rank"))
    qk_rope_head_dim = int(_config_value(text_config, "qk_rope_head_dim", 0))
    qk_head_dim = int(_config_value(text_config, "qk_nope_head_dim")) + qk_rope_head_dim
    value_head_dim = int(_config_value(text_config, "v_head_dim"))
    first_dense = int(_config_value(text_config, "first_k_dense_replace", num_layers))
    moe_frequency = int(_config_value(text_config, "moe_layer_freq", 1) or 1)
    dense_intermediate = int(_config_value(text_config, "intermediate_size"))
    moe_intermediate = int(_config_value(text_config, "moe_intermediate_size"))
    routed_experts = int(_config_value(text_config, "n_routed_experts", 0) or 0)
    shared_experts = int(_config_value(text_config, "n_shared_experts", 0) or 0)
    quantization_config = _config_value(hf_config, "quantization_config", {}) or {}
    if hasattr(quantization_config, "to_dict"):
        quantization_config = quantization_config.to_dict()
    quantization_contract = {
        key: _config_value(quantization_config, key)
        for key in (
            "activation_scheme",
            "block_size",
            "modules_to_not_convert",
            "quant_method",
            "weight_block_size",
        )
        if _config_value(quantization_config, key) is not None
    }

    layers = []
    for layer_index, layer_type in enumerate(layer_types):
        is_kda = layer_type == "linear_attention"
        if not is_kda and layer_type != "deepseek_sparse_attention":
            raise ValueError(
                f"unsupported GLM-5.3-Flash layer type at {layer_index}: {layer_type!r}"
            )
        attention_targets = (
            (
                "q_proj",
                "k_proj",
                "v_proj",
                "b_proj",
                "f_a_proj",
                "f_b_proj",
                "g_a_proj",
                "g_b_proj",
                "o_proj_kda",
            )
            if is_kda
            else (
                "q_a_proj",
                "q_b_proj",
                "kv_a_proj_with_mqa",
                "kv_b_proj",
                "o_proj_dsa",
            )
        )
        sparse = (
            routed_experts > 0
            and layer_index >= first_dense
            and layer_index % moe_frequency == 0
        )
        intermediate = moe_intermediate if sparse else dense_intermediate
        dimensions = {
            name.removesuffix("_kda").removesuffix("_dsa"): _lora_dimension(
                name,
                hidden_size=hidden_size,
                kda_heads=kda_heads,
                kda_head_dim=kda_head_dim,
                attention_heads=attention_heads,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                qk_head_dim=qk_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                value_head_dim=value_head_dim,
                intermediate_size=intermediate,
            )
            for name in (*attention_targets, "gate_proj", "up_proj", "down_proj")
        }
        if is_kda:
            sharding = {
                "q_proj": "attention_tp_column",
                "k_proj": "attention_tp_column",
                "v_proj": "attention_tp_column",
                "b_proj": "attention_tp_column",
                "f_a_proj": "replicated",
                "f_b_proj": "attention_tp_column",
                "g_a_proj": "replicated",
                "g_b_proj": "attention_tp_column",
                "o_proj": "attention_tp_row",
            }
        else:
            sharding = {
                "q_a_proj": "replicated",
                "q_b_proj": "attention_tp_column",
                "kv_a_proj_with_mqa": "replicated",
                "kv_b_proj": "attention_tp_column",
                "o_proj": "attention_tp_row",
            }
        expert_prefix = "expert_" if sparse else ""
        sharding.update(
            {
                "gate_proj": f"{expert_prefix}tp_column",
                "up_proj": f"{expert_prefix}tp_column",
                "down_proj": f"{expert_prefix}tp_row",
            }
        )
        layers.append(
            {
                "attention": "kda" if is_kda else "dsa",
                "dimensions": dimensions,
                "index": layer_index,
                "mlp": "moe" if sparse else "dense",
                "sharding": sharding,
            }
        )

    contract = {
        "dense_prefix_layers": first_dense,
        "excluded_by_default": list(GLM5_NEXT_EXCLUDED_LORA_TARGETS),
        "hidden_size": hidden_size,
        "layers": layers,
        "model_type": "glm5_next",
        "num_layers": num_layers,
        "lora_alpha": int(alpha),
        "lora_rank": int(rank),
        "quantization": quantization_contract,
        "rollout_target_modules": list(GLM5_NEXT_DEFAULT_ROLLOUT_TARGETS),
        "routed_experts": routed_experts,
        "shared_experts": shared_experts,
        "target_modules": list(GLM5_NEXT_DEFAULT_LORA_TARGETS),
        "trainer_exclude_modules": trainer_exclude_modules,
        "version": 2,
    }
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract["fingerprint"] = hashlib.sha256(payload.encode()).hexdigest()
    return contract
