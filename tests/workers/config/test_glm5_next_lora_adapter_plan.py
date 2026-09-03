from pathlib import Path
from types import SimpleNamespace

import pytest
from peft import LoraConfig
from peft.tuners.tuners_utils import check_target_module_exists

from verl.workers.config.model import build_glm5_next_lora_adapter_plan
from verl.workers.rollout.sglang_rollout.utils import sglang_lora_target_modules


REPO_ROOT = Path(__file__).resolve().parents[3]


def glm_config(layer_types):
    text = SimpleNamespace(
        model_type="glm5_next_text",
        hidden_size=4096,
        num_hidden_layers=len(layer_types),
        layer_types=layer_types,
        first_k_dense_replace=3,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        n_routed_experts=32,
        n_shared_experts=1,
        num_attention_heads=64,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        v_head_dim=256,
        linear_attn_config={"num_heads": 64, "head_dim": 128},
    )
    return SimpleNamespace(model_type="glm5_next", text_config=text)


def test_plan_binds_hf_and_serving_geometry_without_indexer_or_embeddings():
    plan = build_glm5_next_lora_adapter_plan(
        glm_config(
            [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "deepseek_sparse_attention",
            ]
        ),
        "all-linear",
        rank=16,
        alpha=32,
    )

    assert plan is not None
    assert len(plan["fingerprint"]) == 64
    assert plan["lora_rank"] == 16
    assert plan["lora_alpha"] == 32
    assert "indexer.wk" not in plan["target_modules"]
    assert "mlp.gate" in plan["excluded_by_default"]
    assert "model.visual" in plan["excluded_by_default"]
    assert "embed_tokens" not in plan["target_modules"]
    assert "lm_head" not in plan["target_modules"]
    assert plan["layers"][0]["attention"] == "kda"
    assert plan["layers"][0]["dimensions"]["b_proj"] == [4096, 64]
    assert plan["layers"][0]["dimensions"]["o_proj"] == [8192, 4096]
    assert plan["layers"][0]["sharding"]["f_a_proj"] == "replicated"
    assert plan["layers"][0]["sharding"]["b_proj"] == "attention_tp_column"
    assert plan["layers"][3]["attention"] == "dsa"
    assert plan["layers"][3]["mlp"] == "moe"
    assert plan["layers"][3]["dimensions"]["q_b_proj"] == [1536, 16384]
    assert plan["layers"][3]["dimensions"]["kv_b_proj"] == [512, 32768]
    assert plan["layers"][3]["dimensions"]["gate_proj"] == [4096, 2048]
    assert plan["layers"][3]["sharding"]["gate_proj"] == "expert_tp_column"

    assert sglang_lora_target_modules(plan["target_modules"], plan) == plan[
        "rollout_target_modules"
    ]

    peft_config = LoraConfig(
        target_modules=plan["target_modules"],
        exclude_modules=plan["trainer_exclude_modules"],
    )
    assert not check_target_module_exists(
        peft_config, "model.visual.merger.gate_proj"
    )
    assert check_target_module_exists(
        peft_config, "model.language_model.layers.0.mlp.gate_proj"
    )


def test_plan_preserves_user_exclusions_and_fingerprints_them():
    config = glm_config(["linear_attention"])
    base = build_glm5_next_lora_adapter_plan(config, "all-linear")
    custom = build_glm5_next_lora_adapter_plan(
        config, "all-linear", exclude_modules=["layers.0.self_attn.q_proj"]
    )
    peft_config = LoraConfig(
        target_modules=custom["target_modules"],
        exclude_modules=custom["trainer_exclude_modules"],
    )

    assert base["fingerprint"] != custom["fingerprint"]
    assert not check_target_module_exists(
        peft_config, "model.language_model.layers.0.self_attn.q_proj"
    )
    assert not check_target_module_exists(
        peft_config, "base_model.model.model.visual.merger.down_proj"
    )
    assert check_target_module_exists(
        peft_config, "model.language_model.layers.0.self_attn.k_proj"
    )


def test_plan_rejects_invalid_user_exclusions():
    with pytest.raises(TypeError, match="exclude_modules entries"):
        build_glm5_next_lora_adapter_plan(
            glm_config(["linear_attention"]),
            "all-linear",
            exclude_modules=["q_proj", 3],
        )


def test_plan_fingerprint_covers_depth_and_layer_schedule():
    short = build_glm5_next_lora_adapter_plan(
        glm_config(["linear_attention", "deepseek_sparse_attention"]),
        "all-linear",
    )
    deep = build_glm5_next_lora_adapter_plan(
        glm_config(["linear_attention", "linear_attention", "deepseek_sparse_attention"]),
        "all-linear",
    )
    assert short["fingerprint"] != deep["fingerprint"]

    another_rank = build_glm5_next_lora_adapter_plan(
        glm_config(["linear_attention", "deepseek_sparse_attention"]),
        "all-linear",
        rank=8,
    )
    assert short["fingerprint"] != another_rank["fingerprint"]


def test_plan_rejects_incomplete_or_unknown_layer_schedule():
    config = glm_config(["linear_attention"])
    config.text_config.num_hidden_layers = 2
    with pytest.raises(ValueError, match="one layer_types entry"):
        build_glm5_next_lora_adapter_plan(config, "all-linear")

    config = glm_config(["mystery_attention"])
    with pytest.raises(ValueError, match="unsupported GLM"):
        build_glm5_next_lora_adapter_plan(config, "all-linear")


def test_rollout_detects_post_plan_target_drift():
    plan = build_glm5_next_lora_adapter_plan(
        glm_config(["linear_attention"]), "all-linear"
    )
    with pytest.raises(ValueError, match="no longer match"):
        sglang_lora_target_modules(["q_proj"], plan)


def test_non_glm_and_explicit_targets_keep_existing_semantics():
    assert (
        build_glm5_next_lora_adapter_plan(
            SimpleNamespace(model_type="qwen2"), "all-linear"
        )
        is None
    )
    assert build_glm5_next_lora_adapter_plan(
        glm_config(["linear_attention"]), ["q_proj"]
    ) is None
    assert sglang_lora_target_modules("all-linear") == ["all"]


def test_9b_sft_recipe_is_portable_and_runs_two_steps_by_default():
    source = (
        REPO_ROOT
        / "examples/glm53_flash/run_glm53_flash_9b_lora_sft_smoke.sh"
    ).read_text()
    assert "/home/" not in source
    assert "MODEL_PATH" in source
    assert "TRAIN_FILE" in source
    assert "GLM53_SFT_STEPS:-2" in source
    assert "+checkpoint.save_lora_only=true" in source
    assert "torchrun --standalone --nnodes=1 --nproc_per_node=1" in source
