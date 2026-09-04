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

`generate_targeted_quality_data.py` creates `targeted-template-v4` without a
teacher model. Each target is an exact rendering or correction of data already
present in its prompt. The current artifact has 244 rows: 204 train, 20
validation, and 20 test. It includes 144 Markdown-required rows, 28
accidental-Han cleanup rows, 48 controls that preserve Han inside code,
blockquotes, or links, and 28 intentional-Chinese retention rows. All rows
carry project-authored Apache-2.0 provenance and the explicit review method
`deterministic-template-audit`.

The v4 dataset SHA-256 is
`38805cecef0615fbf6c25432cd3e384aeea6273f332b4f1812397cea9127c9bd`.
Its training split keeps two deterministic renderings per semantic group,
while validation and test use dedicated held-out renderings. The targeted set
can be trained and evaluated independently as a diagnostic, but a quality
adapter must combine it with authentic Russian material and preserve separate
held-out results for each failure family.

Versions 2 and 3 are historical invalid split-leak artifacts, not quality
selection data. V2 reused targets across splits; the exhaustive v4 audit later
showed that v3 still contained reused targeted templates and near-duplicate
Wikipedia fragments across splits.

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

The split-safe `wikipedia-corruption-v3` rebuild accepts 499 article groups and
produces 1,996 examples split 1,608/224/164. Near-duplicate source fragments
and their rendered prompts and responses are joined into connected components
before each component is assigned atomically to one split. The source sample
remains SHA-256
`39ee6e5e2181c1bce36fa022a037ec54613ecd5a75258fc22a7d03485bba96cd`;
the generated rows SHA-256 is
`4a48a3b01cd6f130ad83a62b7a08fbfb64e9be454e7f3b877c399bdf94400afc`.

The clean engineering candidate is
`mixture_targeted_wikipedia_v4_2240`, combining this 1,996-row component with
the 244-row `targeted-template-v4` artifact; it does not include the pending
Aya/OASST1 review queue. Its split counts are 1,812 train, 244 validation, and
184 untouched test. Exact full-chat tokenization assigns 1,699 rows to
`seq256`, 259 to `seq384`, and 282 to `seq768`, with no truncation. Mixture
SHA-256 is
`34f0d92ad9b46f0289f26c7aec8cee1b4bdae76310bceda3a8bb36a71d211442`.
Both input JSONL hashes, both tokenizer-file hashes, and the external source
sample identity are mandatory production inputs. The builder and exhaustive
split audit fail closed on unaccepted rows, oversized sequences, source
identity/content reuse, prompt/response cross-field reuse, and cross-split
near-duplicates at threshold `0.7`.

The retired v2 and v3 mixtures may retain useful loss, memory, gradient,
adapter-export, and reload evidence, but both are invalid for quality selection
or evaluation. The original v2 incident is machine-readable in
[`split_isolation_incident_2026-09-04.json`](http://vladigur.vla.yp-c.yandex.net:3020/root/tasks/-/blob/main/glm52/lora/quality/split_isolation_incident_2026-09-04.json);
the subsequent v3 failure is recorded in
[`mixture_targeted_wikipedia_v3_2716/split_isolation_audit.json`](http://vladigur.vla.yp-c.yandex.net:3020/root/tasks/-/blob/main/glm52/lora/quality/mixture_targeted_wikipedia_v3_2716/split_isolation_audit.json).
V4's clean data status is necessary but not sufficient: production evaluation
remains `PENDING` until exact trusted trainer and inference shard manifests,
local full-read receipts, and bound base/adapter runtime and generation proof
are supplied.

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
