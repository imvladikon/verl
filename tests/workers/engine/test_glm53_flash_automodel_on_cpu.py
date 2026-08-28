from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from transformers import AutoConfig

from verl.workers.engine.automodel.transformer_impl import AutomodelEngine, AutomodelEngineWithLMHead
from verl.workers.engine.automodel.utils import build_automodel_model, is_glm53_flash_config
from verl.workers.engine_workers import _attach_actor_model_config


def _engine_config(**overrides):
    values = {
        "tp_size": 1,
        "pp_size": 1,
        "cp_size": 1,
        "ep_size": 1,
        "backend_config": {
            "attn": "sdpa",
            "linear": "torch",
            "rms_norm": "torch_fp32",
            "experts": "torch_mm",
            "dispatcher": "torch",
            "enable_hf_state_dict_adapter": True,
        },
        "enable_fp8": False,
        "enable_compile": False,
        "attn_implementation": "sdpa",
        "model_dtype": "bf16",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_flash_identity_does_not_alias_older_glm_family():
    flash = SimpleNamespace(model_type="glm5_next", architectures=["Glm5NextForConditionalGeneration"])
    older_glm = SimpleNamespace(model_type="glm_moe_dsa", architectures=["GlmMoeDsaForCausalLM"])

    assert is_glm53_flash_config(flash)
    assert not is_glm53_flash_config(older_glm)
    with pytest.raises(ValueError, match="Inconsistent GLM-5.3-Flash config"):
        is_glm53_flash_config(SimpleNamespace(model_type="glm5_next", architectures=["GlmMoeDsaForCausalLM"]))


@pytest.mark.parametrize("freeze_vision_tower", [False, True])
def test_actor_vision_freeze_switch_reaches_model_contract(freeze_vision_tower):
    actor_config = SimpleNamespace(model_config=None, freeze_vision_tower=freeze_vision_tower)
    model_config = SimpleNamespace(freeze_vision_tower=not freeze_vision_tower)

    attached = _attach_actor_model_config(actor_config, model_config)

    assert attached is model_config
    assert actor_config.model_config.freeze_vision_tower is freeze_vision_tower


def test_flash_build_uses_current_vlm_distributed_setup_api(monkeypatch):
    from nemo_automodel._transformers import NeMoAutoModelForCausalLM, NeMoAutoModelForImageTextToText
    from nemo_automodel.components.models.glm5_next.config import Glm5NextConfig

    hf_config = Glm5NextConfig()
    hf_config.architectures = ["Glm5NextForConditionalGeneration"]
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *_args, **_kwargs: hf_config)
    model_config = SimpleNamespace(
        hf_config=hf_config,
        local_path="unused-local-checkpoint",
        path="unused-local-checkpoint",
        use_remove_padding=True,
        trust_remote_code=False,
        freeze_vision_tower=False,
    )
    calls = []
    monkeypatch.setattr(
        NeMoAutoModelForImageTextToText,
        "from_pretrained",
        lambda **kwargs: calls.append(("vlm", kwargs)) or kwargs,
    )
    monkeypatch.setattr(
        NeMoAutoModelForCausalLM,
        "from_pretrained",
        lambda **kwargs: calls.append(("causal", kwargs)) or kwargs,
    )
    setup = object()

    kwargs = build_automodel_model(model_config, _engine_config(), setup)

    assert [kind for kind, _ in calls] == ["vlm"]
    assert kwargs["distributed_setup"] is setup
    assert kwargs["has_packed_sequence"] is True
    assert kwargs["text_config"] == {
        "num_nextn_predict_layers": 0,
        "output_hidden_states": True,
    }
    assert kwargs["freeze_config"]["freeze_vision_tower"] is False
    assert "freeze_embeddings" not in kwargs["freeze_config"]
    assert kwargs["use_liger_kernel"] is False
    assert kwargs["use_sdpa_patching"] is False
    assert not {"device_mesh", "moe_mesh", "distributed_config", "activation_checkpointing"} & kwargs.keys()


def test_flash_build_honors_model_vision_freeze_switch(monkeypatch):
    from nemo_automodel._transformers import NeMoAutoModelForImageTextToText
    from nemo_automodel.components.models.glm5_next.config import Glm5NextConfig

    hf_config = Glm5NextConfig()
    hf_config.architectures = ["Glm5NextForConditionalGeneration"]
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *_args, **_kwargs: hf_config)
    model_config = SimpleNamespace(
        hf_config=hf_config,
        local_path="unused-local-checkpoint",
        path="unused-local-checkpoint",
        use_remove_padding=True,
        trust_remote_code=False,
        freeze_vision_tower=True,
    )
    monkeypatch.setattr(
        NeMoAutoModelForImageTextToText,
        "from_pretrained",
        lambda **kwargs: kwargs,
    )

    kwargs = build_automodel_model(model_config, _engine_config(), object())

    assert kwargs["freeze_config"]["freeze_vision_tower"] is True


def test_flash_packed_inputs_always_include_flat_document_boundaries():
    input_ids = torch.nested.as_nested_tensor(
        [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
        layout=torch.jagged,
    )
    position_ids = torch.nested.as_nested_tensor(
        [torch.arange(3), torch.arange(2)],
        layout=torch.jagged,
    )
    micro_batch = TensorDict({"input_ids": input_ids, "position_ids": position_ids}, batch_size=[2])
    micro_batch.set_non_tensor("temperature", 1.0)
    micro_batch.set_non_tensor("use_remove_padding", True)
    engine = object.__new__(AutomodelEngineWithLMHead)
    engine.engine_config = SimpleNamespace(attn_implementation="sdpa")
    engine._is_glm53_flash = True

    model_inputs, _ = engine.prepare_model_inputs(micro_batch)

    torch.testing.assert_close(model_inputs["cu_seqlens"], torch.tensor([0, 3, 5], dtype=torch.int32))
    assert model_inputs["cu_seqlens"].ndim == 1
    assert model_inputs["qkv_format"] == "thd"
    assert model_inputs["max_seqlen"] == 3


def test_current_automodel_optimizer_builder_is_callable():
    engine = object.__new__(AutomodelEngine)
    engine.device_mesh = None
    engine.optimizer_config = SimpleNamespace(
        optimizer_impl="torch.optim",
        optimizer="AdamW",
        lr=1e-4,
        weight_decay=0.1,
        eps=1e-8,
        betas=(0.9, 0.95),
        master_weights=False,
        store_param_remainders=False,
        exp_avg_dtype=None,
        exp_avg_sq_dtype=None,
        master_weight_dtype=None,
        override_optimizer_config={},
    )

    optimizer = engine._build_optimizer(torch.nn.Linear(4, 4))

    assert isinstance(optimizer, torch.optim.AdamW)


def test_flash_rollout_export_uses_model_state_dict_adapter(monkeypatch):
    class Adapter:
        def to_hf(self, state, **kwargs):
            assert kwargs == {
                "exclude_key_regex": r".*_extra_state.*",
                "quantization": False,
            }
            return {"released.weight": state["native.weight"]}

    class Module:
        state_dict_adapter = Adapter()

        @staticmethod
        def state_dict():
            return {"native.weight": torch.arange(4)}

    monkeypatch.setattr(
        "verl.workers.engine.automodel.transformer_impl.load_automodel_model_to_gpu",
        lambda _module: None,
    )
    engine = object.__new__(AutomodelEngine)
    engine.module = Module()
    engine.engine_config = SimpleNamespace(ep_size=1)
    engine._is_glm53_flash = True
    engine._is_offload_param = False

    params, peft = engine.get_per_tensor_param()

    assert peft is None
    assert [(name, value.tolist()) for name, value in params] == [("released.weight", [0, 1, 2, 3])]


def test_flash_rollout_export_rejects_rank_local_ep_stream():
    engine = object.__new__(AutomodelEngine)
    engine.engine_config = SimpleNamespace(ep_size=2)
    engine._is_glm53_flash = True

    with pytest.raises(NotImplementedError, match="rank-local experts"):
        engine.get_per_tensor_param()
