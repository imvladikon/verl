from types import SimpleNamespace

import torch

from verl.utils.kernel import fp8_kernel


def _cuda_tensor_stub(index=0):
    return SimpleNamespace(device=torch.device("cuda", index))


def test_ampere_uses_torch_fp8_cast(monkeypatch):
    monkeypatch.setattr(fp8_kernel, "_TRITON_AVAILABLE", True)
    monkeypatch.setattr(fp8_kernel, "_DISABLE_TRITON_FP8", False)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))

    assert not fp8_kernel._can_use_triton_e4m3fn(_cuda_tensor_stub())


def test_ada_and_hopper_can_use_triton_fp8_cast(monkeypatch):
    monkeypatch.setattr(fp8_kernel, "_TRITON_AVAILABLE", True)
    monkeypatch.setattr(fp8_kernel, "_DISABLE_TRITON_FP8", False)
    monkeypatch.setattr(torch.version, "hip", None)

    for capability in ((8, 9), (9, 0)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device, cap=capability: cap)
        assert fp8_kernel._can_use_triton_e4m3fn(_cuda_tensor_stub())


def test_cpu_never_uses_triton_fp8_cast(monkeypatch):
    monkeypatch.setattr(fp8_kernel, "_TRITON_AVAILABLE", True)
    monkeypatch.setattr(fp8_kernel, "_DISABLE_TRITON_FP8", False)

    assert not fp8_kernel._can_use_triton_e4m3fn(SimpleNamespace(device=torch.device("cpu")))
