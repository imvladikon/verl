import builtins
import sys
import types

import pytest

pytest.importorskip("megatron.core.distributed")

from verl.utils import megatron_utils  # noqa: E402


def test_bridge_wrapper_is_installed_and_fallback_hooks_are_removed(monkeypatch):
    enabled = []
    bridge_utils = types.ModuleType("megatron.bridge.peft.utils")
    bridge_utils.enable_expert_parallel_grad_sync_in_finalize = enabled.append
    bridge_utils.finalize_model_grads_with_expert_adapter_sync = object()
    monkeypatch.setitem(sys.modules, "megatron.bridge.peft.utils", bridge_utils)

    model = [object()]
    func = megatron_utils._finalize_model_grads_func(model)

    assert func is bridge_utils.finalize_model_grads_with_expert_adapter_sync
    assert enabled == [model]


def test_megatron_finalize_without_bridge(monkeypatch):
    from megatron.core.distributed import finalize_model_grads

    real_import = builtins.__import__

    def no_bridge(name, *args, **kwargs):
        if name.startswith("megatron.bridge"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "megatron.bridge.peft.utils", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_bridge)
    assert megatron_utils._finalize_model_grads_func([object()]) is finalize_model_grads
