import json
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from verl.workers.config import AutomodelCheckpointConfig
from verl.workers.engine.automodel.transformer_impl import AutomodelEngine


def _distributed_rng_worker(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
    result_dir: str,
    mode: str,
) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        strict = mode == "missing_strict"
        engine = object.__new__(AutomodelEngine)
        engine.rank = rank
        engine.world_size = world_size
        engine.checkpoint_config = AutomodelCheckpointConfig(strict_rng_state=strict)

        random.seed(1000 + rank)
        np.random.seed(2000 + rank)
        torch.manual_seed(3000 + rank)
        engine._save_rng_state(checkpoint_dir)

        if mode == "round_trip":
            expected = torch.rand(1).item()
            torch.manual_seed(9000 + rank)
            engine._load_rng_state(checkpoint_dir)
            actual = torch.rand(1).item()
            status = {"rank": rank, "result": "pass", "expected": expected, "actual": actual}
        else:
            torch.manual_seed(9000 + rank)
            state_before_load = torch.get_rng_state().clone()
            torch.distributed.barrier()
            if rank == 0:
                path = Path(engine._rng_state_path(checkpoint_dir))
                if mode.startswith("missing"):
                    path.unlink()
                elif mode == "corrupt":
                    torch.save({"not_rng": True}, path)
                else:
                    raise AssertionError(f"unknown distributed RNG test mode: {mode}")
            torch.distributed.barrier()

            error = None
            try:
                engine._load_rng_state(checkpoint_dir)
            except RuntimeError as caught:
                error = str(caught)
            torch.distributed.barrier()
            unchanged = torch.equal(torch.get_rng_state(), state_before_load)
            status = {
                "rank": rank,
                "result": "error" if error else "skip",
                "error": error,
                "rng_unchanged": unchanged,
            }

        Path(result_dir, f"rank-{rank}.json").write_text(
            json.dumps(status, sort_keys=True),
            encoding="utf-8",
        )
    finally:
        torch.distributed.destroy_process_group()


def _run_distributed_rng_case(tmp_path: Path, world_size: int, mode: str) -> list[dict]:
    checkpoint_dir = tmp_path / "checkpoint"
    result_dir = tmp_path / "results"
    checkpoint_dir.mkdir()
    result_dir.mkdir()
    torch.multiprocessing.spawn(
        _distributed_rng_worker,
        args=(
            world_size,
            str(tmp_path / "process-group-init"),
            str(checkpoint_dir),
            str(result_dir),
            mode,
        ),
        nprocs=world_size,
        join=True,
    )
    return [json.loads(Path(result_dir, f"rank-{rank}.json").read_text()) for rank in range(world_size)]


def test_four_rank_rng_round_trip_is_rank_distinct(tmp_path):
    results = _run_distributed_rng_case(tmp_path, world_size=4, mode="round_trip")

    assert all(result["result"] == "pass" for result in results)
    assert all(result["actual"] == result["expected"] for result in results)
    assert len({result["actual"] for result in results}) == 4


@pytest.mark.parametrize(
    ("mode", "expected_result"),
    [
        ("missing_strict", "error"),
        ("missing_permissive", "skip"),
        ("corrupt", "error"),
    ],
)
def test_two_rank_rng_fault_policy_finishes_collectively(tmp_path, mode, expected_result):
    results = _run_distributed_rng_case(tmp_path, world_size=2, mode=mode)

    assert [result["result"] for result in results] == [expected_result, expected_result]
    assert all(result["rng_unchanged"] for result in results)
