"""Model-free checks for preserving the CPU budget during NUMA placement.

Extract the function to keep these tests runnable without importing Torch/Ray or
probing GPUs. NVML and OS affinity operations are substituted at the boundary.
"""

import ast
import ctypes
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def run_affinity(monkeypatch, allowed, preferred, *, fail_query=False):
    path = Path(__file__).resolve().parents[2] / "verl/utils/distributed.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "set_numa_affinity")
    events = []
    word_bits = ctypes.sizeof(ctypes.c_ulong) * 8

    def get_mask(handle, size):
        assert handle == "selected-gpu"
        events.append(("query", size))
        if fail_query:
            raise RuntimeError("NVML query failed")
        return [sum(1 << (cpu % word_bits) for cpu in preferred if cpu // word_bits == i) for i in range(size)]

    def forbidden_set(handle):
        pytest.fail("NVML setter would overwrite the inherited CPU budget")

    nvml = SimpleNamespace(
        nvmlInit=lambda: events.append("init"),
        nvmlShutdown=lambda: events.append("shutdown"),
        nvmlDeviceGetHandleByIndex=lambda index: "selected-gpu",
        nvmlDeviceGetCpuAffinity=get_mask,
        nvmlDeviceSetCpuAffinity=forbidden_set,
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    namespace = dict(
        is_npu_available=False,
        ctypes=SimpleNamespace(
            CDLL=lambda _: SimpleNamespace(numa_available=lambda: 0), sizeof=ctypes.sizeof, c_ulong=ctypes.c_ulong
        ),
        os=SimpleNamespace(
            environ={"LOCAL_RANK": "0"},
            sched_getaffinity=lambda _: set(allowed),
            sched_setaffinity=lambda pid, cpus: events.append(("set", pid, set(cpus))),
        ),
        ray=SimpleNamespace(is_initialized=lambda: False),
        get_resource_name=lambda: "GPU",
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    namespace["set_numa_affinity"]()
    return events


def test_numa_preserves_two_cpu_budget(monkeypatch):
    events = run_affinity(monkeypatch, {14, 15}, set(range(48)))
    assert ("set", 0, {14, 15}) in events
    assert events[-1] == "shutdown"


def test_numa_intersects_sparse_cpu_ids(monkeypatch):
    events = run_affinity(monkeypatch, {63, 65, 130}, {65, 130, 131})
    assert ("query", (130 + ctypes.sizeof(ctypes.c_ulong) * 8) // (ctypes.sizeof(ctypes.c_ulong) * 8)) in events
    assert ("set", 0, {65, 130}) in events


def test_disjoint_numa_node_keeps_original_budget(monkeypatch):
    events = run_affinity(monkeypatch, {100, 101}, set(range(64)))
    assert not any(isinstance(event, tuple) and event[0] == "set" for event in events)
    assert events[-1] == "shutdown"


def test_nvml_failure_leaves_budget_and_shuts_down(monkeypatch):
    events = run_affinity(monkeypatch, {14, 15}, {14}, fail_query=True)
    assert not any(isinstance(event, tuple) and event[0] == "set" for event in events)
    assert events[-1] == "shutdown"
