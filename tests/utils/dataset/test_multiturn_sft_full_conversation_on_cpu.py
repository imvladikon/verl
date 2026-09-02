from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


def contextual_tokenizer() -> PreTrainedTokenizerFast:
    tokens = [
        "[UNK]",
        "<bos>",
        "<system>",
        "<user>",
        "<assistant>",
        "<think>",
        "rules",
        "question",
        "first",
        "answer",
        "followup",
        "second",
    ]
    backend = Tokenizer(WordLevel({token: index for index, token in enumerate(tokens)}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = (
        "<bos> {% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "<assistant> <think> {{ message['content'] }} "
        "{% else %}{{ '<' ~ message['role'] ~ '>' }} {{ message['content'] }} "
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<assistant> <think> {% endif %}"
    )
    return tokenizer


def test_full_conversation_keeps_canonical_tokens_and_masks_only_answers(tmp_path):
    tokenizer = contextual_tokenizer()
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "followup"},
        {"role": "assistant", "content": "second answer"},
    ]
    parquet = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pylist([{"messages": messages}]), parquet)
    config = OmegaConf.create(
        {
            "pad_mode": "no_padding",
            "max_length": 64,
            "truncation": "error",
            "messages_key": "messages",
            "tokenize_full_conversation": True,
        }
    )

    dataset = MultiTurnSFTDataset(str(parquet), tokenizer, config)
    item = dataset[0]
    canonical = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"][0]

    torch.testing.assert_close(item["input_ids"], canonical)
    supervised = tokenizer.decode(item["input_ids"][item["loss_mask"].bool()])
    assert supervised == "first answer second answer"
    assert int(item["loss_mask"].sum()) == 4
