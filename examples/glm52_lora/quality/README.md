# GLM-5.2 quality artifacts

The training and held-out rows are kept in the internal `tasks` repository,
not duplicated in this framework fork. Stage the following immutable
directories under one `GLM52_QUALITY_ROOT` before resolving a full-model run:

- `mixture_targeted_wikipedia_v11_train_576` — exact reviewed 9×64 train view
  plus validation and locked test artifacts.

Canonical source:
[`tasks/glm52/lora/quality`](http://vladigur.vla.yp-c.yandex.net:3020/root/tasks/-/tree/main/glm52/lora/quality).
The complete execution gate and hashes are in
[`QUALITY_EXECUTION_GATE.md`](../QUALITY_EXECUTION_GATE.md).

`run_full_sft_census_v11_megatron.sh` verifies the manifest, selection record,
all JSONL and Parquet hashes, and exact official-tokenizer statistics before
starting VERL. The test split is verified but never passed to the trainer.

Example:

```bash
GLM52_QUALITY_ROOT=/path/to/staged/glm52/lora/quality \
MODEL_PATH=/path/to/immutable/GLM-5.2 \
examples/glm52_lora/run_full_sft_census_v11_megatron.sh
```

This corpus is a formatting and accidental-script repair candidate. It is not
evidence of broad Russian-language quality without the blinded full-model
comparison.
