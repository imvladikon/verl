"""Exercise native LoRA construction with real PEFT and a tiny CPU model."""

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from verl.workers.engine.fsdp.transformer_impl import FSDPEngine, _cast_model_dtype_preserving_fp32_buffers


@pytest.mark.parametrize("strategy,adapter_dtype", [("fsdp", torch.bfloat16), ("fsdp2", torch.float32)])
def test_lora_storage_precision_and_effective_updates(strategy, adapter_dtype):
    torch.manual_seed(5309)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    base = LlamaForCausalLM(config).to(dtype=torch.bfloat16)
    frozen = [(p, p.detach().clone()) for p in base.parameters()]
    engine = object.__new__(FSDPEngine)
    engine.engine_config = SimpleNamespace(strategy=strategy)
    engine.model_config = SimpleNamespace(
        lora_adapter_path=None,
        lora_rank=2,
        lora_alpha=4,
        target_modules=["q_proj", "v_proj"],
        target_parameters=None,
        exclude_modules=None,
    )
    model = engine._build_lora_module(base)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name and p.dtype == adapter_dtype for name, p in trainable)
    initial = {name: p.detach().clone() for name, p in trainable}
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-3, weight_decay=0)
    tokens = torch.tensor([[1, 2, 3, 4]])
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=tokens, labels=tokens).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and p.grad.dtype == adapter_dtype for _, p in trainable)
        assert all(torch.isfinite(p.grad).all() for _, p in trainable)
        assert any(torch.count_nonzero(p.grad) for _, p in trainable)
        optimizer.step()
    assert any(not torch.equal(initial[name], p) for name, p in trainable)
    assert all(not p.requires_grad and p.grad is None and torch.equal(p, value) for p, value in frozen)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_model_cast_preserves_exact_fp32_buffers_and_persistence(dtype):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(config)
    bias = torch.tensor([0.123456789, 1e-8], dtype=torch.float32)
    model.register_buffer("routing_bias", bias)
    model.register_buffer("routing_bias_alias", bias, persistent=False)
    model.register_buffer("counter", torch.tensor([2**40], dtype=torch.int64))
    buffers = {name: value.clone() for name, value in model.named_buffers(remove_duplicate=False)}
    state_keys = set(model.state_dict())
    assert any(not torch.equal(value, value.to(dtype).float()) for value in buffers.values() if value.is_floating_point())

    result = _cast_model_dtype_preserving_fp32_buffers(model, dtype)

    assert result is model and all(p.dtype == dtype for p in model.parameters())
    assert model.routing_bias is model.routing_bias_alias
    assert set(model.state_dict()) == state_keys
    assert "routing_bias" in state_keys and "routing_bias_alias" not in state_keys
    for name, value in model.named_buffers(remove_duplicate=False):
        assert value.dtype == buffers[name].dtype and torch.equal(value, buffers[name])
