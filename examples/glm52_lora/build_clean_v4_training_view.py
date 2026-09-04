#!/usr/bin/env python3
"""Build and verify the exact 28x64 clean-v4 SFT training view.

This is a formatting/script-repair dataset view. It is not evidence of broad
Russian language quality. The selection never changes validation or test and
never samples, duplicates, or reorders retained training examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SOURCE_REVISION = "mixture_targeted_wikipedia_v4_2240"
VIEW_REVISION = "mixture_targeted_wikipedia_v4_train_1792"
SOURCE_MANIFEST_SHA256 = "2b4f2f682ce8b7cba8b4e1e22c3867aa05c61fe3d43ae407ec8cefe3a8099281"
SOURCE_MIXTURE_SHA256 = "34f0d92ad9b46f0289f26c7aec8cee1b4bdae76310bceda3a8bb36a71d211442"
SOURCE_SPLIT_AUDIT_SHA256 = "b28439bf6b259fde01a8121c4898a871fe2a036ae07973156fe3ae024babc6d7"
BUCKET_ROWS_SHA256 = {
    "seq256": "4ead2f85d789a8b9374a84f04382bd3a90d0af6f3a649295020d6d2f30af9dbb",
    "seq384": "993f43f3a4338abcb2d38e9d2ffe8cfa3acce4ea4d1176ef3468151338108a87",
    "seq768": "8a2850e3d970efcbf647dc93d3aa3a7b225959843c5d99ab273b3ea493df4bda",
}
BUCKETS = tuple(BUCKET_ROWS_SHA256)
SPLITS = ("train", "validation", "test")
TARGETED_DATASET = "project-authored/glm52-targeted-quality"
WIKIPEDIA_DATASET = "wikimedia/wikipedia"
COMMON_WIKIPEDIA_TAGS = frozenset({"russian", "teacher-free"})
SOURCE_SPLIT_COUNTS = {"train": 1812, "validation": 244, "test": 184}
TARGET_TRAIN_ROWS = 1792
GLOBAL_BATCH_SIZE = 64
TRAINING_STEPS = 28
OMIT_COUNT = SOURCE_SPLIT_COUNTS["train"] - TARGET_TRAIN_ROWS
HASH_NAMESPACE = "glm52-clean-v4-format-script-view-v1"
SCOPE = "formatting-and-script-repair-only"
STATUS = "LOCAL-DATA-PASS/FULL-MODEL-RUNTIME-AND-BROAD-RUSSIAN-QUALITY-PENDING"
PYARROW_WRITER_VERSION = "22.0.0"
PARQUET_WRITER_CONTRACT = {
    "library": "pyarrow",
    "version": PYARROW_WRITER_VERSION,
    "parquet_version": "2.6",
    "compression": "snappy",
    "use_dictionary": True,
    "write_statistics": True,
    "data_page_version": "1.0",
    "use_compliant_nested_type": True,
    "write_page_index": False,
    "write_page_checksum": False,
    "store_schema": True,
    "row_group_size": "exact_split_row_count",
}
CANONICAL_TASKS_BUILDER_SHA256 = "cf7f8fff4d0dce38ca49ff7c68d2a7a1b058ed32e077afddb3158e03282ba975"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_line(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_pyarrow_writer_version() -> None:
    require(
        pa.__version__ == PYARROW_WRITER_VERSION,
        f"byte-reproducible parquet build requires pyarrow=={PYARROW_WRITER_VERSION}, found {pa.__version__}",
    )


def validate_builder_identity(builder: Any) -> None:
    current_sha256 = sha256_file(Path(__file__).resolve())
    require(
        builder
        in (
            {"file": Path(__file__).name, "sha256": current_sha256},
            {
                "file": Path(__file__).name,
                "sha256": CANONICAL_TASKS_BUILDER_SHA256,
            },
        ),
        "view builder-code drift",
    )


def load_locked_source(
    source_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    require(source_root.name == SOURCE_REVISION, f"expected source {SOURCE_REVISION}")
    manifest_path = source_root / "manifest.json"
    mixture_path = source_root / "mixture_rows.jsonl"
    audit_path = source_root / "split_isolation_audit.json"
    require(
        sha256_file(manifest_path) == SOURCE_MANIFEST_SHA256,
        "clean-v4 source manifest SHA-256 drift",
    )
    require(
        sha256_file(mixture_path) == SOURCE_MIXTURE_SHA256,
        "clean-v4 mixture row SHA-256 drift",
    )
    require(
        sha256_file(audit_path) == SOURCE_SPLIT_AUDIT_SHA256,
        "clean-v4 split-isolation audit SHA-256 drift",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(manifest["mixture_rows_sha256"] == SOURCE_MIXTURE_SHA256, "manifest/rows drift")
    require(
        manifest["split_isolation_audit_sha256"] == SOURCE_SPLIT_AUDIT_SHA256,
        "manifest/audit drift",
    )
    require(audit["status"] == "PASS", "clean-v4 split audit is not PASS")
    require(not any(audit["counts"].values()), "clean-v4 split audit has violations")
    require(audit["split_counts"] == SOURCE_SPLIT_COUNTS, "source split counts drift")

    bucket_by_id: dict[str, str] = {}
    for bucket, expected_digest in BUCKET_ROWS_SHA256.items():
        rows_path = source_root / bucket / "rows.jsonl"
        require(
            sha256_file(rows_path) == expected_digest,
            f"{bucket} row-stream SHA-256 drift",
        )
        for row in read_jsonl(rows_path):
            example_id = row.get("id")
            require(isinstance(example_id, str) and example_id, f"{rows_path}: bad id")
            require(example_id not in bucket_by_id, f"duplicate bucket id: {example_id}")
            bucket_by_id[example_id] = bucket

    rows = read_jsonl(mixture_path)
    ids = [row.get("id") for row in rows]
    require(all(isinstance(value, str) and value for value in ids), "bad source id")
    require(
        len(ids) == len(set(ids)) == sum(SOURCE_SPLIT_COUNTS.values()),
        "source IDs drift",
    )
    require(set(ids) == set(bucket_by_id), "bucket/source ID coverage drift")
    require(
        Counter(row.get("split") for row in rows) == SOURCE_SPLIT_COUNTS,
        "source splits drift",
    )
    return rows, bucket_by_id


def wikipedia_task_tag(row: dict[str, Any]) -> str:
    tags = row.get("tags")
    require(isinstance(tags, list), f"{row.get('id')}: tags must be a list")
    task_tags = sorted(set(tags) - COMMON_WIKIPEDIA_TAGS)
    require(len(task_tags) == 1, f"{row.get('id')}: expected one Wikipedia task tag")
    return task_tags[0]


def stable_score(example_id: str, bucket: str, task_tag: str) -> str:
    payload = f"{HASH_NAMESPACE}\0{bucket}\0{task_tag}\0{example_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def omission_quotas(stratum_sizes: dict[str, int], omit_count: int) -> dict[str, int]:
    """Allocate exact omissions by Hamilton/largest-remainder apportionment."""
    total = sum(stratum_sizes.values())
    require(total > omit_count >= 0, "invalid omission budget")
    exact = {key: Fraction(size * omit_count, total) for key, size in stratum_sizes.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = omit_count - sum(quotas.values())
    order = sorted(
        stratum_sizes,
        key=lambda key: (-(exact[key] - quotas[key]), key),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    require(sum(quotas.values()) == omit_count, "apportionment did not close")
    require(
        all(0 <= quotas[key] < stratum_sizes[key] for key in quotas),
        "omission quota would empty a Wikipedia stratum",
    )
    return quotas


def select_training_rows(
    rows: list[dict[str, Any]], bucket_by_id: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train = [row for row in rows if row["split"] == "train"]
    targeted = [row for row in train if row["provenance"]["dataset"] == TARGETED_DATASET]
    wikipedia = [row for row in train if row["provenance"]["dataset"] == WIKIPEDIA_DATASET]
    require(len(train) == SOURCE_SPLIT_COUNTS["train"], "train row count drift")
    require(len(targeted) == 204, "targeted train row count drift")
    require(len(wikipedia) == 1608, "Wikipedia train row count drift")
    require(len(train) == len(targeted) + len(wikipedia), "unexpected train source")

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in wikipedia:
        bucket = bucket_by_id[row["id"]]
        tag = wikipedia_task_tag(row)
        strata[f"{bucket}|{tag}"].append(row)
    sizes = {key: len(value) for key, value in strata.items()}
    quotas = omission_quotas(sizes, OMIT_COUNT)

    omitted_ids: set[str] = set()
    per_stratum: dict[str, dict[str, int]] = {}
    for key in sorted(strata):
        bucket, tag = key.split("|", 1)
        ordered = sorted(
            strata[key],
            key=lambda row: (stable_score(row["id"], bucket, tag), row["id"]),
        )
        omitted = ordered[: quotas[key]]
        omitted_ids.update(row["id"] for row in omitted)
        per_stratum[key] = {
            "source": len(ordered),
            "selected": len(ordered) - len(omitted),
            "omitted": len(omitted),
        }

    selected = [row for row in train if row["id"] not in omitted_ids]
    omitted = [row for row in train if row["id"] in omitted_ids]
    require(len(selected) == TARGET_TRAIN_ROWS, "selected train row count drift")
    require(len(omitted) == OMIT_COUNT, "omitted train row count drift")
    require(
        all(row["provenance"]["dataset"] == WIKIPEDIA_DATASET for row in omitted),
        "selection omitted a non-Wikipedia row",
    )
    require(
        {row["id"] for row in targeted}.issubset({row["id"] for row in selected}),
        "selection omitted a targeted row",
    )

    def accounting(key_fn: Any) -> dict[str, dict[str, int]]:
        before = Counter(key_fn(row) for row in train)
        after = Counter(key_fn(row) for row in selected)
        removed = Counter(key_fn(row) for row in omitted)
        return {
            key: {
                "source": before[key],
                "selected": after[key],
                "omitted": removed[key],
            }
            for key in sorted(before)
        }

    tag_before = Counter(tag for row in train for tag in row["tags"])
    tag_after = Counter(tag for row in selected for tag in row["tags"])
    tag_omitted = Counter(tag for row in omitted for tag in row["tags"])
    per_tag = {
        tag: {
            "source": tag_before[tag],
            "selected": tag_after[tag],
            "omitted": tag_omitted[tag],
        }
        for tag in sorted(tag_before)
    }
    details = {
        "algorithm": {
            "name": "stratified-stable-hash-largest-remainder",
            "version": 1,
            "hash": "sha256(namespace\\0bucket\\0task_tag\\0example_id)",
            "namespace": HASH_NAMESPACE,
            "stratum": "token_bucket|wikipedia_task_tag",
            "quota": "Hamilton/largest-remainder over Wikipedia train strata",
            "within_stratum": "omit lowest hash scores; example_id breaks ties",
            "retained_order": "source mixture_rows.jsonl order",
            "sampling": "none",
            "replacement": False,
        },
        "source_train_rows": len(train),
        "selected_train_rows": len(selected),
        "omitted_train_rows": len(omitted),
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "optimizer_steps": TRAINING_STEPS,
        "consumed_train_rows": GLOBAL_BATCH_SIZE * TRAINING_STEPS,
        "per_source": accounting(lambda row: row["provenance"]["dataset"]),
        "per_bucket": accounting(lambda row: bucket_by_id[row["id"]]),
        "per_stratum": per_stratum,
        "per_tag": per_tag,
        "omitted_ids": [row["id"] for row in omitted],
    }
    require(details["consumed_train_rows"] == len(selected), "batch budget drift")
    return selected, omitted, details


def sft_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ],
        "enable_thinking": False,
        "example_id": row["id"],
        "tags": row["tags"],
        "provenance": row["provenance"],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("wb") as output:
        for row in rows:
            output.write(canonical_json_line(row))


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    require_pyarrow_writer_version()
    table = pa.Table.from_pylist([sft_row(row) for row in rows])
    require(table.num_rows > 0, "cannot write an empty parquet split")
    pq.write_table(
        table,
        path,
        version="2.6",
        compression="snappy",
        use_dictionary=True,
        write_statistics=True,
        data_page_version="1.0",
        use_compliant_nested_type=True,
        write_page_index=False,
        write_page_checksum=False,
        store_schema=True,
        row_group_size=table.num_rows,
    )


def build_view(source_root: Path, output_root: Path) -> dict[str, Any]:
    require_pyarrow_writer_version()
    rows, bucket_by_id = load_locked_source(source_root)
    selected, _omitted, selection = select_training_rows(rows, bucket_by_id)
    by_split = {
        "train": selected,
        "validation": [row for row in rows if row["split"] == "validation"],
        "test": [row for row in rows if row["split"] == "test"],
    }
    require(
        {split: len(split_rows) for split, split_rows in by_split.items()}
        == {"train": TARGET_TRAIN_ROWS, "validation": 244, "test": 184},
        "view split count drift",
    )

    output_root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        jsonl_path = output_root / f"{split}_rows.jsonl"
        parquet_path = output_root / f"sft_{split}.parquet"
        write_jsonl(jsonl_path, by_split[split])
        write_parquet(parquet_path, by_split[split])
        artifacts[split] = {
            "rows": len(by_split[split]),
            "rows_jsonl": jsonl_path.name,
            "rows_jsonl_sha256": sha256_file(jsonl_path),
            "sft_parquet": parquet_path.name,
            "sft_parquet_sha256": sha256_file(parquet_path),
        }

    omitted_path = output_root / "omitted_train_ids.json"
    omitted_path.write_text(
        json.dumps(selection["omitted_ids"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "status": STATUS,
        "dataset_revision": VIEW_REVISION,
        "builder": {
            "file": Path(__file__).name,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "parquet_writer": PARQUET_WRITER_CONTRACT,
        "scope": SCOPE,
        "broad_russian_quality_proof": False,
        "warning": (
            "This view targets formatting and accidental-script repair only; "
            "it is not evidence of broad Russian language quality."
        ),
        "source": {
            "dataset_revision": SOURCE_REVISION,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "mixture_rows_sha256": SOURCE_MIXTURE_SHA256,
            "split_isolation_audit_sha256": SOURCE_SPLIT_AUDIT_SHA256,
            "split_isolation_status": "PASS",
            "bucket_rows_sha256": BUCKET_ROWS_SHA256,
        },
        "selection": selection,
        "omitted_train_ids_file": omitted_path.name,
        "omitted_train_ids_sha256": sha256_file(omitted_path),
        "artifacts": artifacts,
        "evaluation_policy": {
            "validation": "unchanged clean-v4 validation split; never optimized on",
            "test": "unchanged clean-v4 test split; never passed to the trainer",
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_view(source_root: Path, view_root: Path) -> dict[str, Any]:
    rows, bucket_by_id = load_locked_source(source_root)
    expected_selected, expected_omitted, expected_selection = select_training_rows(rows, bucket_by_id)
    manifest = json.loads((view_root / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["schema_version"] == 1, "view schema drift")
    require(manifest["status"] == STATUS, "view status drift")
    require(manifest["dataset_revision"] == VIEW_REVISION, "view revision drift")
    validate_builder_identity(manifest["builder"])
    require(
        manifest.get("parquet_writer") == PARQUET_WRITER_CONTRACT,
        "parquet writer contract drift",
    )
    require(manifest["scope"] == SCOPE, "view scope drift")
    require(manifest["broad_russian_quality_proof"] is False, "invalid quality claim")
    require(manifest["selection"] == expected_selection, "selection manifest drift")

    expected_by_split = {
        "train": expected_selected,
        "validation": [row for row in rows if row["split"] == "validation"],
        "test": [row for row in rows if row["split"] == "test"],
    }
    for split, expected_rows in expected_by_split.items():
        detail = manifest["artifacts"][split]
        rows_path = view_root / detail["rows_jsonl"]
        parquet_path = view_root / detail["sft_parquet"]
        require(detail["rows"] == len(expected_rows), f"{split} manifest row count drift")
        require(
            sha256_file(rows_path) == detail["rows_jsonl_sha256"],
            f"{split} JSONL SHA-256 drift",
        )
        require(read_jsonl(rows_path) == expected_rows, f"{split} row/order drift")
        require(
            sha256_file(parquet_path) == detail["sft_parquet_sha256"],
            f"{split} parquet SHA-256 drift",
        )
        parquet_table = pq.read_table(parquet_path)
        expected_table = pa.Table.from_pylist([sft_row(row) for row in expected_rows])
        parquet_rows = parquet_table.to_pylist()
        require(
            parquet_table.schema.equals(expected_table.schema),
            f"{split} parquet schema drift",
        )
        require(
            parquet_rows == expected_table.to_pylist(),
            f"{split} parquet content/order drift",
        )

    omitted_path = view_root / manifest["omitted_train_ids_file"]
    require(
        sha256_file(omitted_path) == manifest["omitted_train_ids_sha256"],
        "omitted ID file SHA-256 drift",
    )
    omitted_ids = json.loads(omitted_path.read_text(encoding="utf-8"))
    require(
        omitted_ids == [row["id"] for row in expected_omitted],
        "omitted ID list drift",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("source_root", type=Path)
    build_parser.add_argument("output_root", type=Path)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("source_root", type=Path)
    check_parser.add_argument("view_root", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        result = build_view(args.source_root, args.output_root)
    else:
        result = validate_view(args.source_root, args.view_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
