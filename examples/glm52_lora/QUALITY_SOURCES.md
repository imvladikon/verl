# GLM-5.2 Russian quality source lock

The training builder accepts only explicitly reviewed rows. Public datasets
feed a review queue; they never feed SFT directly.

## Included candidate sources

- [CohereLabs/aya_dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset),
  revision `f9ea04583f02a8f86404ff6c58bf75fe637df8a2`, Apache-2.0. Human
  annotations and human re-annotations; the Dataset Viewer returns 423 Russian
  train rows. Every pair still requires factual and language review.
- [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1),
  revision `fdf72ae0827c1cda404aff25b6603abec9e3399b`, Apache-2.0.
  Human-generated and human-rated conversation trees. Selection is restricted
  to Russian, non-synthetic, accepted, rank-0 root replies with at least three
  reviews and quality at least 0.75.

`build_quality_review_queue.py` loads these exact revisions in streaming mode,
records source row IDs and licenses, removes prompt duplicates before assigning
hash-stable splits, and marks every output row `pending`. Source validation
rows are forced into the local test split before deduplication.

The final 2026-09-02 full audit processed 202,362 Aya rows and 88,838 OASST1
rows in 38.7 seconds with 1.12 GiB peak RSS. It produced 414 pending candidates after
filters and prompt deduplication: 284 Aya and 130 OASST1, split 362 train, 20
validation and 32 test. Only four candidates explicitly require Markdown
(three lists and one code block), so these sources can anchor general Russian
but cannot by themselves solve the Markdown defect. A separate reviewed,
deterministic targeted set is required before the full-model experiment.

## Project-authored targeted set

`generate_targeted_quality_data.py` creates `targeted-template-v1` without a
teacher model. Each target is an exact rendering or correction of data already
present in its prompt. It emits 720 rows split by semantic group, including
416 Markdown-required rows, 128 accidental-Han removal rows, 48 controls that
preserve Han inside code, blockquotes, or URLs, and 32 controls that preserve
intentional Chinese text. All rows carry project-authored Apache-2.0
provenance and the explicit review method `deterministic-template-audit`.

The validated artifact has dataset SHA-256
`e60cc63ac674b45a5bdc45c3d068e76058024c237a29331c6d56b02bebaf20c4`.
Using the exact surgery-checkpoint tokenizer, its full chat sequences have
p95 166, p99 180, and maximum 187 tokens. The targeted set can be trained and
evaluated independently as a diagnostic, but a quality adapter must combine
it with accepted human-reviewed general-Russian rows and preserve separate
held-out results for each failure family.

## Excluded from the first mixture

- [IlyaGusev/ru_turbo_alpaca](https://huggingface.co/datasets/IlyaGusev/ru_turbo_alpaca):
  the card reports only 68% jointly correct instruction-output pairs in its
  larger crowd evaluation and includes a legal disclaimer tied to the
  generation provider. It is not an ASAP quality anchor.
- [Den4ikAI/russian_instructions_2](https://huggingface.co/datasets/Den4ikAI/russian_instructions_2):
  the card describes translated and aggregated examples but does not provide
  row-level provenance sufficient for this review gate.
- [attn-signs/russian-easy-instructions](https://huggingface.co/datasets/attn-signs/russian-easy-instructions):
  permissive license, but its current card does not document generation,
  review methodology, or row-level provenance.

Exclusion is not a claim that every row is bad. It means the source is not
trusted automatically for the first targeted adapter.

## Review policy

An accepted row must have:

1. a named reviewer and `review.status=accepted`;
2. a factually and stylistically correct target in the requested language;
3. no accidental Han in visible Russian prose;
4. valid CommonMark/table structure when required;
5. immutable dataset, revision, split, source-record and license provenance.

The seven-row `quality_dataset.example.jsonl` remains only an executable schema
fixture. It is not part of the proposed production training count.
