# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An adapter resumed under a different alpha must stop the run, not be rescaled into it."""

import json
import types

import pytest

from verl.workers.config.megatron_peft import adapter_claim, check_adapter_matches_config


def _adapter(tmp_path, **fields):
    (tmp_path / "adapter_config.json").write_text(json.dumps(fields))
    return tmp_path


def _config(adapter_path, nested=True, **lora):
    lora = {"rank": 8, "alpha": 8, **lora}
    if nested:
        lora["adapter_path"] = str(adapter_path)
        return types.SimpleNamespace(lora=lora, lora_adapter_path=None)
    # fsdp-style: the path lives on the flat field instead.
    return types.SimpleNamespace(lora={}, lora_adapter_path=str(adapter_path))


def test_a_matching_adapter_passes(tmp_path):
    path = _adapter(tmp_path, r=8, lora_alpha=8)
    check_adapter_matches_config(_config(path), rank=8, alpha=8)


def test_the_alpha_that_would_be_silently_doubled_is_refused(tmp_path):
    """The case this exists for: SFT wrote alpha 8, the RL config carries the default."""
    path = _adapter(tmp_path, r=8, lora_alpha=8)
    with pytest.raises(ValueError, match="alpha: adapter has 8, this run applies 16"):
        check_adapter_matches_config(_config(path), rank=8, alpha=16)


def test_a_rank_mismatch_is_refused(tmp_path):
    path = _adapter(tmp_path, r=8, lora_alpha=8)
    with pytest.raises(ValueError, match="rank: adapter has 8, this run applies 32"):
        check_adapter_matches_config(_config(path), rank=32, alpha=8)


def test_both_mismatches_are_named_together(tmp_path):
    path = _adapter(tmp_path, r=16, lora_alpha=32)
    with pytest.raises(ValueError) as caught:
        check_adapter_matches_config(_config(path), rank=8, alpha=8)
    assert "rank: adapter has 16" in str(caught.value)
    assert "alpha: adapter has 32" in str(caught.value)


def test_a_field_the_adapter_does_not_claim_is_not_invented(tmp_path):
    """Half a claim is still worth checking; the missing half must not fail the run."""
    path = _adapter(tmp_path, r=8)
    check_adapter_matches_config(_config(path), rank=8, alpha=999)


def test_the_flat_adapter_path_is_checked_too(tmp_path):
    path = _adapter(tmp_path, r=8, lora_alpha=8)
    with pytest.raises(ValueError, match="alpha"):
        check_adapter_matches_config(_config(path, nested=False), rank=8, alpha=16)


def test_no_adapter_and_a_bare_weights_directory_are_both_fine(tmp_path):
    check_adapter_matches_config(types.SimpleNamespace(lora={}, lora_adapter_path=None), rank=8, alpha=8)
    # A directory with weights but no adapter_config.json makes no claim to contradict.
    check_adapter_matches_config(_config(tmp_path / "missing"), rank=8, alpha=8)
    (tmp_path / "bare").mkdir()
    check_adapter_matches_config(_config(tmp_path / "bare"), rank=8, alpha=8)


def _checkpoint(tmp_path, *, hf_adapter=None, train_meta=None):
    """The layout a Megatron run actually resumes from.

    lora.adapter_path points at the torch_dist shards, which is where Bridge reads weights;
    what the adapter was trained with is recorded beside them, not in them.
    """
    step = tmp_path / "global_step_20"
    dist_ckpt = step / "model" / "dist_ckpt"
    dist_ckpt.mkdir(parents=True)
    (dist_ckpt / "common.pt").write_text("")  # shards, carrying no claim of their own
    if hf_adapter is not None:
        adapter_dir = step / "model" / "huggingface" / "adapter"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapter_config.json").write_text(json.dumps(hf_adapter))
    if train_meta is not None:
        (step / "lora_train_meta.json").write_text(json.dumps(train_meta))
    return step, dist_ckpt


def test_the_claim_beside_the_dist_shards_is_found(tmp_path):
    """The case that made the check a no-op on the megatron path."""
    _step, dist_ckpt = _checkpoint(tmp_path, hf_adapter={"r": 8, "lora_alpha": 8})
    found = adapter_claim(str(dist_ckpt))
    assert found is not None and found[1] == {"r": 8, "lora_alpha": 8}

    with pytest.raises(ValueError, match="alpha: adapter has 8, this run applies 16"):
        check_adapter_matches_config(_config(dist_ckpt), rank=8, alpha=16)


def test_lora_train_meta_is_used_when_no_hf_adapter_was_exported(tmp_path):
    _step, dist_ckpt = _checkpoint(tmp_path, train_meta={"r": 8, "lora_alpha": 8, "task_type": "CAUSAL_LM"})
    with pytest.raises(ValueError, match="alpha: adapter has 8, this run applies 32"):
        check_adapter_matches_config(_config(dist_ckpt), rank=8, alpha=32)


def test_the_checkpoint_root_works_as_the_adapter_path_too(tmp_path):
    step, _dist_ckpt = _checkpoint(tmp_path, hf_adapter={"r": 8, "lora_alpha": 8})
    with pytest.raises(ValueError, match="alpha"):
        check_adapter_matches_config(_config(step), rank=8, alpha=16)


def test_the_exported_adapter_wins_over_the_training_record(tmp_path):
    """Both exist in a real checkpoint; they agree, and the adapter is the thing being loaded."""
    _step, dist_ckpt = _checkpoint(
        tmp_path, hf_adapter={"r": 8, "lora_alpha": 8}, train_meta={"r": 99, "lora_alpha": 99}
    )
    assert adapter_claim(str(dist_ckpt))[1]["r"] == 8


def test_a_matching_checkpoint_passes(tmp_path):
    _step, dist_ckpt = _checkpoint(tmp_path, hf_adapter={"r": 8, "lora_alpha": 8})
    check_adapter_matches_config(_config(dist_ckpt), rank=8, alpha=8)


def test_unreadable_json_does_not_stop_a_run_on_its_own(tmp_path, caplog):
    step, dist_ckpt = _checkpoint(tmp_path, hf_adapter={"r": 8, "lora_alpha": 8})
    (step / "model" / "huggingface" / "adapter" / "adapter_config.json").write_text("{ truncated")
    check_adapter_matches_config(_config(dist_ckpt), rank=8, alpha=16)
    assert any("Cannot read" in record.message for record in caplog.records)
