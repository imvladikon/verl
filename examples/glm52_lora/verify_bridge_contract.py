#!/usr/bin/env python3
"""Verify the config-only GLM-5.2 surgery checkpoint -> Megatron contract."""

from __future__ import annotations

import argparse
import json

from megatron.bridge import AutoBridge
from megatron.bridge.models.glm_moe_dsa.glm5_bridge import GLM5Bridge


EXPECTED_PROVIDER_FIELDS = {
    "num_layers": 10,
    "hidden_size": 6144,
    "num_moe_experts": 16,
    "moe_router_topk": 8,
    "q_lora_rank": 2048,
    "kv_lora_rank": 512,
    "multi_latent_attention": True,
    "experimental_attention_variant": "dsa",
    "dsa_indexer_n_heads": 32,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="imvladikon/GLM-5.2-9B-LoRA-Surgery-Dummy",
    )
    args = parser.parse_args()

    auto_bridge = AutoBridge.from_hf_pretrained(args.model, trust_remote_code=False)
    model_bridge = auto_bridge._model_bridge
    if not isinstance(model_bridge, GLM5Bridge):
        raise AssertionError(
            f"expected GLM5Bridge for {args.model}, got {type(model_bridge).__name__}"
        )
    provider = auto_bridge.to_megatron_provider()
    observed = {
        name: getattr(provider, name) for name in EXPECTED_PROVIDER_FIELDS
    }
    if observed != EXPECTED_PROVIDER_FIELDS:
        raise AssertionError(
            "unexpected surgery provider contract:\n"
            + json.dumps(
                {"expected": EXPECTED_PROVIDER_FIELDS, "observed": observed},
                indent=2,
                sort_keys=True,
            )
        )

    result = {
        "model": args.model,
        "bridge": type(model_bridge).__name__,
        "provider": type(provider).__name__,
        "provider_fields": observed,
        "moe_token_dispatcher_type": provider.moe_token_dispatcher_type,
        "moe_flex_dispatcher_backend": provider.moe_flex_dispatcher_backend,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
