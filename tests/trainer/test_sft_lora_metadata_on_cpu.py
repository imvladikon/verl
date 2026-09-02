from omegaconf import OmegaConf

from verl.trainer.sft_trainer import SFTTrainer


def _metadata(model_config: dict) -> dict | None:
    trainer = SFTTrainer.__new__(SFTTrainer)
    trainer.config = OmegaConf.create({"model": model_config})
    return trainer._get_lora_train_meta()


def test_bridge_lora_metadata_uses_nested_rank_and_alpha() -> None:
    assert _metadata(
        {
            "lora": {
                "rank": 16,
                "alpha": 32,
                "adapter_path": None,
            }
        }
    ) == {
        "r": 16,
        "lora_alpha": 32,
        "task_type": "CAUSAL_LM",
    }


def test_bridge_adapter_path_marks_resumed_lora_even_with_zero_rank() -> None:
    assert _metadata(
        {
            "lora": {
                "rank": 0,
                "alpha": 16,
                "adapter_path": "/tmp/adapter",
            }
        }
    ) == {
        "r": 0,
        "lora_alpha": 16,
        "task_type": "CAUSAL_LM",
    }


def test_flat_peft_metadata_remains_backward_compatible() -> None:
    assert _metadata(
        {
            "lora_rank": 8,
            "lora_alpha": 24,
            "lora_adapter_path": None,
        }
    ) == {
        "r": 8,
        "lora_alpha": 24,
        "task_type": "CAUSAL_LM",
    }
