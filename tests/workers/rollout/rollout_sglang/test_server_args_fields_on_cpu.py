"""Check the launch capability probe without importing Ray, Torch or SGLang."""

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass
class LegacyArgs:
    enable_weights_cpu_backup: bool = False


class StructArgs:
    __slots__ = ("enable_weights_cpu_backup",)


class UnsupportedArgs:
    pass


@pytest.mark.parametrize("args_type,expected", [(LegacyArgs, True), (StructArgs, True), (UnsupportedArgs, False)])
def test_backup_capability_accepts_both_record_layouts(args_type, expected):
    source = Path(__file__).resolve().parents[4] / "verl/workers/rollout/sglang_rollout/async_sglang_server.py"
    tree = ast.parse(source.read_text())
    probes = [
        node.test
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "ServerArgs" in ast.unparse(node.test)
        and "enable_weights_cpu_backup" in ast.unparse(node.test)
    ]
    assert len(probes) == 1
    probe = compile(ast.Expression(probes[0]), str(source), "eval")
    assert eval(probe, {"ServerArgs": args_type}) is expected
