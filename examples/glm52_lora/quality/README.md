# GLM-5.2 quality artifacts

The training and held-out rows are kept in the internal `tasks` repository,
not duplicated in this framework fork. Stage the following immutable
directories under one `GLM52_QUALITY_ROOT` before resolving a full-model run:

- `mixture_targeted_wikipedia_v4_2240` — locked source mixture;
- `mixture_targeted_wikipedia_v4_train_1792` — exact 28×64 training view;
- `heldout_quality_challenge_v1` — validation/test-only supplement.

Canonical source:
[`tasks/glm52/lora/quality`](http://vladigur.vla.yp-c.yandex.net:3020/root/tasks/-/tree/main/glm52/lora/quality).
The complete execution gate and hashes are in
[`QUALITY_EXECUTION_GATE.md`](../QUALITY_EXECUTION_GATE.md).

`run_full_sft_clean_v4_megatron.sh` verifies the view manifest, omitted-ID
receipt, all JSONL files, and all Parquet files before starting VERL. The
builder additionally binds the source mixture, bucket row streams, split
audit, and `pyarrow==22.0.0`. The test split is verified but is never passed to
the trainer.

Example:

```bash
GLM52_QUALITY_ROOT=/path/to/staged/glm52/lora/quality \
MODEL_PATH=/path/to/immutable/GLM-5.2 \
examples/glm52_lora/run_full_sft_clean_v4_megatron.sh
```

This corpus is a formatting and accidental-script repair candidate. It is not
evidence of broad Russian-language quality without the blinded full-model
comparison.
