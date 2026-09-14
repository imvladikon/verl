"""verl's Megatron CheckpointFunction.backward patch frees checkpoint inputs only when nothing else aliases them."""

import pytest
import torch

megatron_random = pytest.importorskip("megatron.core.tensor_parallel.random")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the patch is applied on CUDA only")


@pytest.fixture(scope="module", autouse=True)
def patched_backward():
    from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward

    original = megatron_random.CheckpointFunction.backward
    torch.cuda.set_device(0)
    tracker = megatron_random.get_cuda_rng_tracker()
    if "model-parallel-rng" not in tracker.get_states():
        tracker.add("model-parallel-rng", 1234)
    apply_patch_megatron_recomputation_backward()
    yield
    megatron_random.CheckpointFunction.backward = original


def _attention_like(x, q, weight_view):
    return (x.unsqueeze(1) * q.unsqueeze(1) * weight_view.sum(1).unsqueeze(0)).sum(1)


def _run(case, use_checkpoint):
    torch.manual_seed(0)
    down = torch.nn.Linear(16, 16, device="cuda")
    up = torch.nn.Linear(16, 16, device="cuda")
    weight = torch.nn.Parameter(torch.randn(4, 8, 16, device="cuda"))
    hidden = torch.randn(3, 16, device="cuda", requires_grad=True)
    layer_input = down(hidden)
    # "aliased": the checkpoint input is also saved by a node whose backward runs after the checkpoint's
    q = up(layer_input) if case == "aliased" else down(hidden * 2)
    args = (layer_input, q, weight[:, :4, :])
    out = megatron_random.checkpoint(_attention_like, False, *args) if use_checkpoint else _attention_like(*args)
    out.square().sum().backward()
    grads = [p.grad.clone() for p in (hidden, weight, *down.parameters(), *up.parameters()) if p.grad is not None]
    return layer_input, weight, grads


@pytest.mark.parametrize("case", ["exclusive", "aliased"])
def test_gradients_match_and_only_exclusive_inputs_are_freed(case):
    _, weight_ref, grads_ref = _run(case, use_checkpoint=False)
    layer_input, weight, grads = _run(case, use_checkpoint=True)

    assert len(grads) == len(grads_ref)
    for grad, grad_ref in zip(grads, grads_ref, strict=True):
        torch.testing.assert_close(grad, grad_ref)
    # The parameter was passed as a slice; its storage must survive the checkpoint backward.
    torch.testing.assert_close(weight.detach(), weight_ref.detach())
    if case == "exclusive":
        assert layer_input.untyped_storage().nbytes() == 0
    else:
        assert layer_input.untyped_storage().nbytes() == layer_input.numel() * layer_input.element_size()
