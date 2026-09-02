import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "glm52_lora"))

from verify_tp2_ep2_resume_run import verify_run  # noqa: E402
from verify_tp2_ep2_sft_config import EXPECTED_TARGETS, verify_config  # noqa: E402


def valid_config(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    model = tmp_path / "model"
    train = tmp_path / "train.parquet"
    run = tmp_path / "run"
    model.mkdir()
    train.touch()
    return (
        {
            "model": {
                "path": str(model),
                "use_remove_padding": False,
                "mtp": {"enable": False},
                "lora": {
                    "target_modules": EXPECTED_TARGETS.copy(),
                    "rank": 16,
                    "alpha": 32,
                    "merge": False,
                    "dtype": "bfloat16",
                },
            },
            "engine": {
                "use_mbridge": True,
                "vanilla_mbridge": False,
                "tensor_model_parallel_size": 2,
                "expert_model_parallel_size": 2,
                "expert_tensor_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
                "sequence_parallel": True,
                "use_distributed_optimizer": True,
                "param_offload": False,
                "optimizer_offload": False,
                "override_transformer_config": {
                    "dsa_kernel_backend": "none",
                    "moe_router_dtype": "fp32",
                    "recompute_granularity": None,
                    "recompute_method": None,
                    "recompute_num_layers": None,
                },
            },
            "data": {
                "train_files": [str(train)],
                "val_files": None,
                "train_batch_size": 2,
                "micro_batch_size_per_gpu": 1,
                "max_length": 256,
                "max_token_len_per_gpu": 256,
                "truncation": "error",
                "tokenize_full_conversation": True,
            },
            "trainer": {
                "nnodes": 1,
                "n_gpus_per_node": 4,
                "total_training_steps": 2,
                "save_freq": 1,
                "test_freq": -1,
                "default_local_dir": str(run),
                "resume_mode": "disable",
                "resume_from_path": None,
            },
            "checkpoint": {
                "save_lora_only": True,
                "save_contents": ["model", "optimizer", "extra"],
            },
        },
        model,
        train,
        run,
    )


def test_tp2_ep2_verifier_accepts_exact_contract(tmp_path: Path) -> None:
    config, model, train, run = valid_config(tmp_path)
    result = verify_config(
        config,
        expected_model_path=model,
        expected_train_file=train,
        expected_run_dir=run,
    )
    assert result["status"] == "CONFIG-PASS/RUNTIME-PENDING"
    assert result["trainable_parameters"] == 13_608_960
    assert result["topology"] == {
        "nodes": 1,
        "gpus": 4,
        "tp": 2,
        "ep": 2,
        "etp": 1,
        "dense_dp": 2,
        "expert_dp": 2,
    }


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("engine", "tensor_model_parallel_size"), 1, "TP drift"),
        (("engine", "expert_model_parallel_size"), 1, "EP drift"),
        (("engine", "sequence_parallel"), False, r"TP\+EP requires sequence parallel"),
        (("trainer", "n_gpus_per_node"), 2, "GPU-count drift"),
        (("data", "train_batch_size"), 4, "global batch drift"),
    ],
)
def test_tp2_ep2_verifier_rejects_topology_drift(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    config, model, train, run = valid_config(tmp_path)
    broken = deepcopy(config)
    target = broken
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(AssertionError, match=message):
        verify_config(
            broken,
            expected_model_path=model,
            expected_train_file=train,
            expected_run_dir=run,
        )


def write_runtime_fixture(tmp_path: Path, *, resumed_grad_norm: str = "9.5") -> Path:
    run = tmp_path / "run"
    step2 = run / "global_step_2"
    step3 = run / "resumed" / "global_step_3"
    for checkpoint in (step2, step3):
        checkpoint.mkdir(parents=True)
        for dp_rank in (0, 1):
            (checkpoint / f"data_{dp_rank}.pt").touch()
        (checkpoint / "ckpt_contents.json").write_text("{}", encoding="utf-8")
    (run / "run.log").write_text(
        "step:1 - train/loss:14.7 - train/grad_norm:243.8 - train/global_tokens:186\n"
        "step:2 - train/loss:13.0 - train/grad_norm:19.7 - train/global_tokens:288\n",
        encoding="utf-8",
    )
    lines = []
    for rank in range(4):
        lines.extend(
            [
                f"[Rank {rank}] Loaded PEFT adapter checkpoint from {step2}/model/dist_ckpt",
                f"[Rank {rank}] Loaded optimizer checkpoint from {step2}/optimizer/dist_ckpt",
                f"[Rank {rank}] Loaded RNG states from {step2}/extra/dist_ckpt",
            ]
        )
    lines.append(
        f"step:3 - train/loss:12.6 - train/grad_norm:{resumed_grad_norm} - train/global_tokens:214"
    )
    (run / "resume.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run


def test_tp2_ep2_runtime_verifier_accepts_complete_resume(tmp_path: Path) -> None:
    result = verify_run(write_runtime_fixture(tmp_path))
    assert result["status"] == "RUNTIME-PASS"
    assert result["dataloader_token_sequence"] == [186, 288, 214]


def test_tp2_ep2_runtime_verifier_rejects_nan_gradient(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="gradient norm is not finite"):
        verify_run(write_runtime_fixture(tmp_path, resumed_grad_norm="nan"))
