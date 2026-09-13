"""Model-free regression checks for Ray device IDs under inherited GPU masks."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def run_setup(monkeypatch, visible, assigned, *, noset=True):
    source = Path(__file__).resolve().parents[2] / "verl/single_controller/base/worker.py"
    tree = ast.parse(source.read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Worker")
    method = next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == "_setup_env_cuda_visible_devices"
    )
    env = {"LOCAL_RANK": "7"}
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
    selected = []
    monkeypatch.setitem(sys.modules, "verl.utils.ray_utils", SimpleNamespace(ray_noset_visible_devices=lambda: noset))
    namespace = dict(
        os=SimpleNamespace(environ=env),
        ray=SimpleNamespace(
            get_runtime_context=lambda: SimpleNamespace(get_accelerator_ids=lambda: {"GPU": [assigned]})
        ),
        get_resource_name=lambda: "GPU",
        get_visible_devices_keyword=lambda: "CUDA_VISIBLE_DEVICES",
        get_torch_device=lambda: SimpleNamespace(set_device=selected.append),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return lambda: namespace["_setup_env_cuda_visible_devices"](None), env, selected


@pytest.mark.parametrize(
    "visible,assigned,expected",
    [
        (None, "3", 3),
        ("4", "4", 0),
        ("6,4", "4", 1),
        ("6,4", "6", 0),
        ("GPU-a", "GPU-a", 0),
        ("GPU-b,GPU-a", "GPU-a", 1),
        ("MIG-GPU-a/1/2", "MIG-GPU-a/1/2", 0),
    ],
)
def test_assigned_id_maps_to_visible_ordinal(monkeypatch, visible, assigned, expected):
    run, env, selected = run_setup(monkeypatch, visible, assigned)
    run()
    assert selected == [expected]
    assert env["LOCAL_RANK"] == str(expected)
    assert env.get("CUDA_VISIBLE_DEVICES") == visible


@pytest.mark.parametrize("visible,assigned", [("1,3", "4"), ("GPU-a", "GPU-b"), ("", "0"), ("-1", "0")])
def test_unassigned_device_fails_before_selection(monkeypatch, visible, assigned):
    run, env, selected = run_setup(monkeypatch, visible, assigned)
    with pytest.raises(ValueError, match="outside the visible devices"):
        run()
    assert selected == []
    assert env["LOCAL_RANK"] == "7"


def test_ray_managed_visibility_does_not_remap(monkeypatch):
    run, env, selected = run_setup(monkeypatch, "GPU-a", "GPU-a", noset=False)
    run()
    assert selected == []
    assert env["LOCAL_RANK"] == "7"
