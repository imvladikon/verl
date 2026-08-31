# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import os
import re
from collections.abc import Iterable
from typing import Any

from verl.utils.fp8_utils import FP8QuantizerHelper


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    get_value = getattr(config, "get", None)
    if callable(get_value):
        return get_value(key, default)
    return getattr(config, key, default)


def _normalize_ignored_layers(ignored_layers: Any) -> list[str]:
    if ignored_layers is None:
        return []
    if isinstance(ignored_layers, str):
        ignored_layers = ignored_layers.split(",")
    elif not isinstance(ignored_layers, Iterable):
        ignored_layers = [ignored_layers]

    normalized = []
    for layer in ignored_layers:
        layer_name = str(layer).strip()
        if layer_name:
            normalized.append(layer_name)
    return normalized


def _dedupe_layers(ignored_layers: Iterable[str]) -> list[str]:
    seen = set()
    deduped = []
    for layer in ignored_layers:
        layer_lower = layer.lower()
        if layer_lower in seen:
            continue
        seen.add(layer_lower)
        deduped.append(layer)
    return deduped


def _get_ignored_layers_from_env() -> list[str]:
    return _normalize_ignored_layers(os.getenv("SGLANG_FP8_IGNORED_LAYERS"))


def get_sglang_fp8_ignored_layers(quant_config: Any = None) -> list[str]:
    ignored_layers = []
    ignored_layers.extend(_normalize_ignored_layers(_get_config_value(quant_config, "ignored_layers")))
    ignored_layers.extend(_normalize_ignored_layers(_get_config_value(quant_config, "modules_to_not_convert")))
    ignored_layers.extend(_get_ignored_layers_from_env())
    return _dedupe_layers(ignored_layers)


def is_sglang_fp8_quant_config(quant_config: Any) -> bool:
    """Return whether a checkpoint already declares an FP8 rollout layout."""
    quant_method = _get_config_value(quant_config, "quant_method")
    return isinstance(quant_method, str) and quant_method.lower() == "fp8"


def _path_aliases(path: str) -> set[str]:
    """Return equivalent HF/SGLang module paths for an ignore rule."""
    path = path.lower().strip(".")
    aliases = {path}
    if path.startswith("model.language_model."):
        aliases.add("model." + path.removeprefix("model.language_model."))
    elif path.startswith("language_model."):
        aliases.add("model." + path.removeprefix("language_model."))
    return aliases


def _module_path_candidates(param_name: str) -> set[str]:
    name = param_name.strip(".")
    module_name = name[: -len(".weight")] if name.lower().endswith(".weight") else name
    return _path_aliases(name) | _path_aliases(module_name)


def _dot_path_substrings(path: str):
    """Yield every contiguous, dot-boundary-aligned subpath."""
    parts = path.split(".")
    for start in range(len(parts)):
        candidate = parts[start]
        yield candidate
        for end in range(start + 1, len(parts)):
            candidate = f"{candidate}.{parts[end]}"
            yield candidate


def _matches_ignored_layer(param_name: str, ignored_layer: str) -> bool:
    ignored_layer = ignored_layer.strip()
    if not ignored_layer:
        return False

    candidates = _module_path_candidates(param_name)
    if ignored_layer.startswith("re:"):
        pattern = ignored_layer[3:]
        return any(re.match(pattern, candidate) for candidate in candidates)

    ignored_layer = ignored_layer.lower().strip(".")
    ignored_aliases = _path_aliases(ignored_layer)
    for candidate in candidates:
        for ignored_candidate in ignored_aliases:
            if candidate == ignored_candidate:
                return True
            if candidate.startswith(f"{ignored_candidate}."):
                return True
            if candidate.endswith(f".{ignored_candidate}"):
                return True
            if f".{ignored_candidate}." in f".{candidate}.":
                return True
    return False


def build_sglang_fp8_quant_config(hf_config: Any = None, ignored_layers: Any = None) -> dict[str, Any]:
    """Build SGLang block-wise FP8 config shared by server init and weight sync."""
    fp8_quant_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }

    hf_quant_config = _get_config_value(hf_config, "quantization_config")
    merged_ignored_layers = get_sglang_fp8_ignored_layers(hf_quant_config)
    merged_ignored_layers.extend(_normalize_ignored_layers(ignored_layers))
    merged_ignored_layers = _dedupe_layers(merged_ignored_layers)
    if merged_ignored_layers:
        fp8_quant_config["ignored_layers"] = merged_ignored_layers

    return fp8_quant_config


class SGLangFP8QuantizerHelper(FP8QuantizerHelper):
    _GLM_MLA_FP8_PROJECTIONS = (
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
    )

    def __init__(self, quant_config):
        super().__init__(quant_config)
        # Sending the original BF16 tensor after a failed conversion only moves
        # the failure to SGLang's dtype guard and can partially update a model.
        self.raise_on_quantization_error = True
        self.ignored_layers = get_sglang_fp8_ignored_layers(quant_config)
        self._ignored_exact = set()
        self._ignored_regex = []
        for ignored_layer in self.ignored_layers:
            if ignored_layer.startswith("re:"):
                self._ignored_regex.append(re.compile(ignored_layer[3:]))
            else:
                self._ignored_exact.update(_path_aliases(ignored_layer))

    def _is_ignored_param(self, param_name: str) -> bool:
        candidates = _module_path_candidates(param_name)
        if any(pattern.match(candidate) for pattern in self._ignored_regex for candidate in candidates):
            return True
        return any(
            subpath in self._ignored_exact for candidate in candidates for subpath in _dot_path_substrings(candidate)
        )

    def should_quantize_param(self, param_name):
        if self._is_ignored_param(param_name):
            return False
        param_lower = param_name.lower()
        if param_lower.endswith(".weight") and any(
            projection in param_lower for projection in self._GLM_MLA_FP8_PROJECTIONS
        ):
            return True
        return super().should_quantize_param(param_name)
