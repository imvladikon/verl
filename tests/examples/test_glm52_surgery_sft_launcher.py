from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "examples/glm52_lora/run_surgery_sft_megatron.sh"


def test_glm52_surgery_sft_launcher_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", LAUNCHER], check=True)


def test_glm52_surgery_sft_launcher_disables_uniform_layer_recompute() -> None:
    source = LAUNCHER.read_text()

    prefix = "+engine.override_transformer_config."
    assert f"{prefix}recompute_granularity=null" in source
    assert f"{prefix}recompute_method=null" in source
    assert f"{prefix}recompute_num_layers=null" in source
    assert f"{prefix}recompute_granularity=full" not in source


def test_glm52_surgery_sft_launcher_has_no_machine_home_default() -> None:
    assert "/home/" not in LAUNCHER.read_text()
