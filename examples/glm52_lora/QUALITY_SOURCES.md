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

## Teacher-free authentic Russian corruption set

`sample_wikipedia_ru.py` streams
[`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) at
revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, configuration
`20231101.ru`. The Hub card reports CC-BY-SA-3.0 and GFDL. The sampler records
the exact article ID, title, URL, source-text SHA-256, dataset revision and
license; it never stores a full article.

`build_teacher_free_russian_corruptions.py` creates four deterministic targets
per accepted article: accidental-Han removal, Cyrillic/Latin confusable repair,
case-and-period repair, and exact Markdown heading/list rendering. It rejects
Han, Markdown metacharacters, URLs, unmatched quotes or parentheses, malformed
empty punctuation such as `"(, род. )"`, low-Cyrillic sentences, and duplicate
prompts. No model generates or rewrites the target.

The bounded 2026-09-02 engineering audit selected 64 articles after reading 96
streamed rows and retained 61 groups after prompt deduplication: 244 examples,
61 per family. The source sample SHA-256 is
`d7027cb0a69dc6ca6786f8de56fa7f37b73476532f692f174974bb967677de48`;
the final attributed rows SHA-256 is
`871ddc86906d51edc6362fd83ce70338773c7543b6c74f5cb93fc942e576ac74`.
Two independent builds from the same materialized sample were byte-identical,
including Parquet output. Exact GLM-5.2 surgery-tokenizer full-chat lengths were
p50 170, p95 472, p99 546, maximum 556. Markdown rows therefore belong in a
640-token no-truncation bucket, not the 256-token smoke bucket.

The generated `ATTRIBUTION.jsonl` and `NOTICE.md` must travel with the derived
artifact. This is a data-engineering pass, not blanket approval of every
Wikipedia sentence: license/distribution and sampled-content review remain
explicit production gates.

The default 512-article audit read 817 streamed rows, accepted 502 unique
article groups after removing ten duplicate-prompt groups, and produced 2,008
examples split 1,588/244/176. Source-sample and final-row SHA-256 are
`39ee6e5e2181c1bce36fa022a037ec54613ecd5a75258fc22a7d03485bba96cd`
and `5841e7a00dd6109269d9a04d92ccfad26b207ab82b6570b17371b6d04f9a0078`.
The deterministic correction bucket contains 1,506 rows with exact-tokenizer
maximum 357; the 502-row Markdown bucket has maximum 706. Their recommended
no-truncation sequence lengths are 384 and 768 respectively. Both bucket
builds were reproduced byte-for-byte from the same source sample.

The locked ASAP mixture combines this 2,008-row artifact with the 720-row
`targeted-template-v1` artifact; it does not include the pending Aya/OASST1
review queue. Exact full-chat tokenization assigns 2,184 rows to `seq256`, 259
to `seq384`, and 285 to `seq768`, with no truncation. Mixture SHA-256 is
`094a0385dcc27d647b92d2d4d40ad4ec7ae1bbeab8de878915efaed88bc824e7`.
Both input JSONL hashes and both tokenizer-file hashes are mandatory arguments,
and the builder fails closed on any unaccepted row or oversized sequence.

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
