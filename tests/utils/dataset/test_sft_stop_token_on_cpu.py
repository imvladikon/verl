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
"""A target that ends in ordinary text never teaches the model to stop generating."""

from types import SimpleNamespace

import pytest
import torch

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


def _dataset(append_stop_token, stop_token_id=None, eos_token_id=7):
    dataset = MultiTurnSFTDataset.__new__(MultiTurnSFTDataset)
    dataset.append_stop_token = append_stop_token
    dataset.stop_token_id = stop_token_id if stop_token_id is not None else eos_token_id
    dataset.tokenizer = SimpleNamespace(eos_token_id=eos_token_id)
    return dataset


def _fake_processor():
    """Only the attributes the dataset reads during construction."""
    return SimpleNamespace(image_processor=SimpleNamespace(patch_size=14))


def _sample():
    return (
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([1, 1, 1, 1]),
    )


def test_the_stop_token_is_appended_and_supervised():
    input_ids, loss_mask, attention_mask = _dataset(True)._append_stop_token(*_sample())
    assert input_ids.tolist() == [1, 2, 3, 4, 7]
    # The stop token must be part of the loss, otherwise it is never learned.
    assert loss_mask.tolist() == [0, 0, 1, 1, 1]
    assert attention_mask.tolist() == [1, 1, 1, 1, 1]


def test_nothing_changes_when_disabled():
    input_ids, loss_mask, _ = _dataset(False)._append_stop_token(*_sample())
    assert input_ids.tolist() == [1, 2, 3, 4]
    assert loss_mask.tolist() == [0, 0, 1, 1]


def test_a_template_that_already_stops_is_left_alone():
    already = (torch.tensor([1, 2, 7]), torch.tensor([0, 1, 1]), torch.tensor([1, 1, 1]))
    input_ids, loss_mask, _ = _dataset(True)._append_stop_token(*already)
    assert input_ids.tolist() == [1, 2, 7]
    assert loss_mask.tolist() == [0, 1, 1]


def test_an_explicit_id_wins_over_the_tokenizer_default():
    input_ids, _, _ = _dataset(True, stop_token_id=99)._append_stop_token(*_sample())
    assert input_ids.tolist() == [1, 2, 3, 4, 99]


def test_a_tokenizer_without_eos_is_an_error_rather_than_a_silent_skip():
    dataset = _dataset(True, eos_token_id=None)
    dataset.stop_token_id = None
    with pytest.raises(ValueError, match="no eos_token_id"):
        dataset._append_stop_token(*_sample())


def test_a_text_only_set_keeps_full_conversation_on_a_multimodal_checkpoint():
    """GLM-5.3 ships a processor; refusing every such checkpoint would rule out text-only SFT."""
    from verl.utils.dataset.multiturn_sft_dataset import processor_for_full_conversation

    kept = processor_for_full_conversation(_fake_processor(), ["messages", "n_tokens"], ("images", "videos"))
    assert kept is None, "a text-only set falls back to tokenizer-only rendering"


def test_media_columns_are_still_refused():
    from verl.utils.dataset.multiturn_sft_dataset import processor_for_full_conversation

    with pytest.raises(ValueError, match="cannot align media tensors"):
        processor_for_full_conversation(_fake_processor(), ["messages", "images"], ("images", "videos"))


def test_a_text_only_tokenizer_needs_no_decision():
    from verl.utils.dataset.multiturn_sft_dataset import processor_for_full_conversation

    assert processor_for_full_conversation(None, ["messages", "images"], ("images", "videos")) is None
