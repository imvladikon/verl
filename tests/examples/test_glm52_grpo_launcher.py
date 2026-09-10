from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "examples/glm52_lora/run_surgery_grpo_megatron_sglang.sh"


def test_glm52_grpo_launcher_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", LAUNCHER], check=True)


def test_glm52_grpo_launcher_bounds_private_ray_and_cold_jit() -> None:
    source = LAUNCHER.read_text()

    assert "RAY_TMPDIR must be absolute and at most 32 bytes" in source
    assert "+ray_kwargs.ray_init._temp_dir=${ray_tmpdir}" in source
    assert "+ray_kwargs.ray_init.include_dashboard=false" in source
    assert "+ray_kwargs.ray_init.num_cpus=${ray_num_cpus}" in source
    assert "+ray_kwargs.ray_init.object_store_memory=${ray_object_store_memory_bytes}" in source
    assert "+actor_rollout_ref.rollout.engine_kwargs.sglang.watchdog_timeout=${sglang_watchdog_seconds}" in source
    assert "ray_num_cpus=${RAY_NUM_CPUS:-8}" in source
    assert "ray_object_store_memory_bytes=${RAY_OBJECT_STORE_MEMORY_BYTES:-4294967296}" in source
    assert "sglang_watchdog_seconds=${SGLANG_WATCHDOG_SECONDS:-1800}" in source


def test_glm52_grpo_launcher_disables_uniform_layer_recompute() -> None:
    source = LAUNCHER.read_text()

    prefix = "+actor_rollout_ref.actor.megatron.override_transformer_config."
    assert f"{prefix}recompute_granularity=null" in source
    assert f"{prefix}recompute_method=null" in source
    assert f"{prefix}recompute_num_layers=null" in source
    assert f"{prefix}recompute_granularity=full" not in source


def test_glm52_grpo_launcher_has_no_machine_home_or_snapshot_default() -> None:
    source = LAUNCHER.read_text()

    assert "/home/" not in source
    assert "/snapshots/" not in source
