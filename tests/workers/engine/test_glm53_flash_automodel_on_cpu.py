import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from transformers import AutoConfig

from verl.trainer.config import CheckpointConfig
from verl.workers.config import AutomodelCheckpointConfig
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


@pytest.mark.parametrize(
    ("checkpoint_config", "expected_consolidated"),
    [
        (CheckpointConfig(), True),
        (AutomodelCheckpointConfig(save_consolidated=False), False),
    ],
)
def test_automodel_hf_consolidation_follows_checkpoint_config(monkeypatch, checkpoint_config, expected_consolidated):
    captured = {}
    engine = object.__new__(AutomodelEngine)
    engine.checkpoint_config = checkpoint_config
    engine.model_config = SimpleNamespace(path="unused-model")
    engine.device_mesh = None
    engine.moe_mesh = None

    monkeypatch.setattr(
        "verl.workers.engine.automodel.transformer_impl.CheckpointingConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        "verl.workers.engine.automodel.transformer_impl.Checkpointer",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )
    monkeypatch.setattr("verl.workers.engine.automodel.transformer_impl.get_dp_rank", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr("verl.workers.engine.automodel.transformer_impl.get_tp_rank", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr("verl.workers.engine.automodel.transformer_impl.get_pp_rank", lambda *_args, **_kwargs: 0)

    engine._build_checkpointer()

    assert captured["config"].save_consolidated is expected_consolidated


def test_automodel_rng_state_round_trip_is_rank_local(monkeypatch, tmp_path):
    state = {
        "cpu": torch.tensor([1, 2, 3], dtype=torch.uint8),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    loaded = []
    engine = object.__new__(AutomodelEngine)
    engine.rank = 3
    engine.world_size = 4
    engine.checkpoint_config = AutomodelCheckpointConfig()
    monkeypatch.setattr(
        "verl.workers.engine.automodel.transformer_impl.BaseCheckpointManager.get_rng_state",
        lambda: state,
    )
    monkeypatch.setattr(
        "verl.workers.engine.automodel.transformer_impl.BaseCheckpointManager.load_rng_state",
        loaded.append,
    )

    engine._save_rng_state(str(tmp_path))
    engine._load_rng_state(str(tmp_path))

    assert (tmp_path / "extra_state_world_size_4_rank_3.pt").is_file()
    assert len(loaded) == 1
    torch.testing.assert_close(loaded[0]["cpu"], state["cpu"])


def test_automodel_rng_state_round_trip_restores_generators(tmp_path):
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    engine = object.__new__(AutomodelEngine)
    engine.rank = 0
    engine.world_size = 1
    engine.checkpoint_config = AutomodelCheckpointConfig()

    engine._save_rng_state(str(tmp_path))
    expected = (random.random(), np.random.random(), torch.rand(1))
    random.seed(404)
    np.random.seed(505)
    torch.manual_seed(606)
    engine._load_rng_state(str(tmp_path))
    actual = (random.random(), np.random.random(), torch.rand(1))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_automodel_extra_load_fails_closed_without_rank_rng(tmp_path):
    engine = object.__new__(AutomodelEngine)
    engine.rank = 2
    engine.world_size = 4
    engine.checkpoint_config = AutomodelCheckpointConfig(strict_rng_state=True)

    with pytest.raises(RuntimeError, match="per-rank RNG state is incomplete"):
        engine._load_rng_state(str(tmp_path))


def test_automodel_missing_rank_rng_is_backward_compatible_by_default(caplog, tmp_path):
    engine = object.__new__(AutomodelEngine)
    engine.rank = 1
    engine.world_size = 2
    engine.checkpoint_config = CheckpointConfig()

    engine._load_rng_state(str(tmp_path))

    assert "all ranks continue without RNG restoration" in caplog.text


@pytest.mark.parametrize(
    ("save_contents", "expected_calls"),
    [
        (["model"], ["model"]),
        (["hf_model"], ["model"]),
        (["optimizer"], ["optimizer"]),
        (["extra"], ["extra"]),
    ],
)
def test_automodel_save_honors_checkpoint_contents(monkeypatch, tmp_path, save_contents, expected_calls):
    calls = []
    engine = object.__new__(AutomodelEngine)
    engine.module = torch.nn.Linear(2, 2)
    engine.optimizer = object()
    engine.lr_scheduler = object()
    engine._is_offload_param = False
    engine.checkpoint_config = AutomodelCheckpointConfig(save_contents=save_contents)
    engine.checkpointer = SimpleNamespace(
        save_model=lambda *_args, **_kwargs: calls.append("model"),
        save_optimizer=lambda *_args, **_kwargs: calls.append("optimizer"),
    )
    monkeypatch.setattr(
        "verl.workers.engine.automodel.transformer_impl.load_automodel_model_to_gpu",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(engine, "_save_rng_state", lambda *_args, **_kwargs: calls.append("extra"))
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    engine.save_checkpoint(str(tmp_path))

    assert calls == expected_calls


def test_automodel_save_materializes_adam_state_for_unused_parameters(monkeypatch, tmp_path):
    class PartiallyUsedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Parameter(torch.ones(2))
            self.unused = torch.nn.Parameter(torch.ones(2))

    model = PartiallyUsedModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.used.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    assert optimizer.state[model.unused] == {}

    captured = {}
    engine = object.__new__(AutomodelEngine)
    engine.module = model
    engine.optimizer = optimizer
    engine.lr_scheduler = None
    engine._is_offload_param = False
    engine.checkpoint_config = AutomodelCheckpointConfig(save_contents=["optimizer"])
    engine.checkpointer = SimpleNamespace(
        save_optimizer=lambda optimizer, *_args, **_kwargs: captured.update(optimizer.state[model.unused])
    )
    monkeypatch.setattr(
        "verl.workers.engine.automodel.transformer_impl.load_automodel_model_to_gpu",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    engine.save_checkpoint(str(tmp_path))

    assert set(captured) == {"step", "exp_avg", "exp_avg_sq"}
    assert captured["step"].item() == 0
    torch.testing.assert_close(captured["exp_avg"], torch.zeros_like(model.unused))
    torch.testing.assert_close(captured["exp_avg_sq"], torch.zeros_like(model.unused))


@pytest.mark.parametrize(
    ("load_contents", "expected_calls"),
    [
        (["model"], ["model"]),
        (["hf_model"], ["model"]),
        (["optimizer"], ["optimizer"]),
        (["extra"], ["extra"]),
    ],
)
def test_automodel_load_honors_checkpoint_contents(monkeypatch, tmp_path, load_contents, expected_calls):
    calls = []
    engine = object.__new__(AutomodelEngine)
    engine.module = torch.nn.Linear(2, 2)
    engine.optimizer = object()
    engine.lr_scheduler = object()
    engine._is_offload_param = False
    engine._is_offload_optimizer = False
    engine.checkpoint_config = AutomodelCheckpointConfig(load_contents=load_contents)
    engine.checkpointer = SimpleNamespace(
        load_model=lambda *_args, **_kwargs: calls.append("model"),
        load_optimizer=lambda *_args, **_kwargs: calls.append("optimizer"),
    )
    monkeypatch.setattr(engine, "_load_rng_state", lambda *_args, **_kwargs: calls.append("extra"))
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    if "hf_model" in load_contents:
        (tmp_path / "model" / "consolidated").mkdir(parents=True)

    engine.load_checkpoint(str(tmp_path))

    assert calls == expected_calls


@pytest.mark.parametrize(
    ("attribute", "contents", "message"),
    [
        ("save_contents", [], "must not be empty"),
        ("load_contents", [], "must not be empty"),
        ("save_contents", ["unknown"], "Unknown AutoModel checkpoint"),
        ("load_contents", ["unknown"], "Unknown AutoModel checkpoint"),
    ],
)
def test_automodel_checkpoint_config_rejects_invalid_contents(attribute, contents, message):
    with pytest.raises(ValueError, match=message):
        AutomodelCheckpointConfig(**{attribute: contents})


def test_automodel_checkpoint_config_rejects_hf_export_without_consolidation():
    with pytest.raises(ValueError, match="hf_model.*save_consolidated is false"):
        AutomodelCheckpointConfig(save_contents=["hf_model"], save_consolidated=False)


def test_legacy_automodel_engine_rejects_empty_checkpoint_contents():
    engine = object.__new__(AutomodelEngine)
    engine.checkpoint_config = CheckpointConfig(save_contents=[])

    with pytest.raises(ValueError, match="must not be empty"):
        engine._checkpoint_contents("save")


def test_automodel_corrupt_rng_payload_fails_closed(tmp_path):
    engine = object.__new__(AutomodelEngine)
    engine.rank = 0
    engine.world_size = 1
    engine.checkpoint_config = AutomodelCheckpointConfig()
    torch.save({"not_rng": True}, engine._rng_state_path(str(tmp_path)))

    with pytest.raises(RuntimeError, match="RNG deserialize failed"):
        engine._load_rng_state(str(tmp_path))


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
