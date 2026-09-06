"""Real HF layer parity and packed fallback isolation, without model downloads."""

import copy
import os
import unittest
from inspect import unwrap
from types import MethodType, SimpleNamespace

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from verl.models.transformers.qwen3_5 import (
    _call_accepts_kwarg,
    _delta_net_kernel,
    _packed_chunk_gated_delta_rule,
    qwen3_5_decoder_layer_forward,
    qwen3_5_gated_delta_net_forward,
)


def make_layer(kind, device, dtype):
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=4,
        layer_types=[kind],
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(713)
    layer = hf.Qwen3_5DecoderLayer(config, 0).to(device=device, dtype=dtype).eval()
    if hasattr(layer, "linear_attn"):
        with torch.no_grad():
            layer.linear_attn.A_log.fill_(-2)
    return layer


def check_real_decoder_forward_and_backward(kind, legacy_name):
    device = os.environ.get("QWEN_COMPAT_TEST_DEVICE", "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    reference = make_layer(kind, device, dtype)
    actual = copy.deepcopy(reference)
    actual.forward = MethodType(qwen3_5_decoder_layer_forward, actual)
    if hasattr(actual, "linear_attn"):
        actual.linear_attn.forward = MethodType(qwen3_5_gated_delta_net_forward, actual.linear_attn)
    if legacy_name:
        actual.layer_type = kind
        if hasattr(actual, "block_type"):
            del actual.block_type
    torch.manual_seed(217)
    x_ref = (torch.randn(1, 7, 128, device=device, dtype=dtype) * 0.1).requires_grad_()
    x_actual = x_ref.detach().clone().requires_grad_()
    positions = (torch.ones(1, 7, 64, device=device, dtype=dtype), torch.zeros(1, 7, 64, device=device, dtype=dtype))
    out_ref = reference(x_ref, position_embeddings=positions)
    out_actual = actual(x_actual, position_embeddings=positions)
    weight = torch.randn_like(out_ref)
    (out_ref.float() * weight.float()).sum().backward()
    (out_actual.float() * weight.float()).sum().backward()
    tolerance = 0.02 if device == "cuda" else 2e-5
    torch.testing.assert_close(out_actual, out_ref, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(x_actual.grad, x_ref.grad, atol=tolerance, rtol=tolerance)
    errors = []
    for (name, p), (ref_name, r) in zip(actual.named_parameters(), reference.named_parameters(), strict=True):
        assert name == ref_name
        assert (p.grad is None) == (r.grad is None), name
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, r.grad, atol=tolerance, rtol=tolerance, msg=name)
            errors.append(float((p.grad.float() - r.grad.float()).abs().max()))
    print(
        dict(
            kind=kind,
            legacy_name=legacy_name,
            device=device,
            output_max_abs=float((out_actual.detach().float() - out_ref.detach().float()).abs().max()),
            input_grad_max_abs=float((x_actual.grad.float() - x_ref.grad.float()).abs().max()),
            param_grad_max_abs=max(errors),
        )
    )


def check_optional_conv_none_is_preserved():
    assert _delta_net_kernel(SimpleNamespace(causal_conv1d_fn=None), "causal_conv1d_fn") is None
    assert _delta_net_kernel(SimpleNamespace(), "causal_conv1d_fn", use_fast=False) is None


def check_plain_torch_rule_does_not_advertise_packing():
    rule = _delta_net_kernel(SimpleNamespace(), "chunk_gated_delta_rule", use_fast=False)
    assert rule is unwrap(hf.torch_chunk_gated_delta_rule)
    assert not _call_accepts_kwarg(rule, "cu_seqlens")
    assert not _call_accepts_kwarg(rule, "cp_context")


def check_torch_packed_rule_preserves_document_boundaries_and_gradients():
    torch.manual_seed(91)
    tensors = [torch.randn(1, 7, 2, 8).requires_grad_() for _ in range(3)]
    g, beta = -torch.rand(1, 7, 2), torch.rand(1, 7, 2)
    cu = torch.tensor([0, 3, 7])
    q, k, v = tensors
    out, _ = _packed_chunk_gated_delta_rule(SimpleNamespace(), q, k, v, g, beta, cu, cu)
    rule = unwrap(hf.torch_chunk_gated_delta_rule)
    expected, _ = rule(
        q[:, 3:],
        k[:, 3:],
        v[:, 3:],
        g=g[:, 3:],
        beta=beta[:, 3:],
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(out[:, 3:], expected, rtol=0, atol=0)
    out[:, 3:].sum().backward()
    for tensor in tensors:
        assert torch.count_nonzero(tensor.grad[:, :3]) == 0
        assert torch.isfinite(tensor.grad).all()
    altered = [x.detach().clone() for x in tensors]
    for x in altered:
        x[:, :3] += 10
    changed, _ = _packed_chunk_gated_delta_rule(SimpleNamespace(), *altered, g, beta, cu, cu)
    torch.testing.assert_close(changed[:, 3:], out[:, 3:].detach(), rtol=0, atol=0)


def check_unknown_decoder_type_fails_closed():
    layer = SimpleNamespace(block_type="unknown", input_layernorm=lambda x: x)
    with unittest.TestCase().assertRaisesRegex(ValueError, "skip attention"):
        qwen3_5_decoder_layer_forward(layer, torch.zeros(1, 1, 4), position_embeddings=None)


class QwenCompatibilityTests(unittest.TestCase):
    def test_real_decoder_forward_and_backward(self):
        for kind in ["linear_attention", "full_attention"]:
            for legacy in [False, True]:
                with self.subTest(kind=kind, legacy=legacy):
                    check_real_decoder_forward_and_backward(kind, legacy)

    def test_optional_conv_none(self):
        check_optional_conv_none_is_preserved()

    def test_torch_rule_capabilities(self):
        check_plain_torch_rule_does_not_advertise_packing()

    def test_packed_isolation(self):
        check_torch_packed_rule_preserves_document_boundaries_and_gradients()

    def test_unknown_type(self):
        check_unknown_decoder_type_fails_closed()


if __name__ == "__main__":
    unittest.main()
