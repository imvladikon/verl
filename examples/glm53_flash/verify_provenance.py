#!/usr/bin/env python3
"""Fail closed unless the active GLM dependencies match the committed pins."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path

EXPECTED = {
    "sglang": {
        "distribution": "sglang",
        "module": "sglang",
        "url": "https://github.com/imvladikon/sglang.git",
        "commit": "35b66043c7ca95d03c1a25d0c49eca7caba8187e",
    },
    "megatron": {
        "distribution": "megatron-core",
        "module": "megatron.core.package_info",
        "url": "https://github.com/imvladikon/Megatron-LM.git",
        "commit": "f3926fee5d6764e2faa1d695da0ce56e1a6cf9c5",
    },
    "automodel": {
        "distribution": "nemo-automodel",
        "module": "nemo_automodel",
        "url": "https://github.com/NVIDIA-NeMo/Automodel.git",
        "commit": "9228f33cf73d66a9b2e84256d298aac9a70283f0",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("flash", "automodel"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _run_git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _checkout_provenance(module: str) -> dict[str, str] | None:
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Cannot locate required module {module!r}")
    origin = Path(spec.origin).resolve()
    for candidate in (origin.parent, *origin.parents):
        try:
            root = Path(_run_git(candidate, "rev-parse", "--show-toplevel"))
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        try:
            relative_origin = origin.relative_to(root)
            _run_git(root, "ls-files", "--error-unmatch", str(relative_origin))
        except (ValueError, subprocess.CalledProcessError):
            # A VCS-installed wheel can live in a .venv below an unrelated
            # project checkout. Its provenance comes from direct_url.json,
            # not from that parent checkout's HEAD.
            continue
        remotes = _run_git(root, "remote", "-v")
        return {
            "kind": "checkout",
            "module_path": str(origin),
            "root": str(root),
            "commit": _run_git(root, "rev-parse", "HEAD"),
            "remotes": remotes,
        }
    return None


def _installed_provenance(distribution: str, module: str) -> dict[str, str]:
    checkout = _checkout_provenance(module)
    if checkout is not None:
        return checkout

    dist = importlib.metadata.distribution(distribution)
    direct_url_text = dist.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError(
            f"{distribution} is not a Git checkout and has no direct_url.json; "
            "the exact source revision cannot be proven"
        )
    direct_url = json.loads(direct_url_text)
    vcs_info = direct_url.get("vcs_info") or {}
    return {
        "kind": "installed-vcs",
        "module_path": str(importlib.util.find_spec(module).origin),
        "url": direct_url.get("url", ""),
        "commit": vcs_info.get("commit_id", ""),
    }


def _normalize_url(value: str) -> str:
    return value.removesuffix("/").removesuffix(".git")


def _verify(name: str) -> dict[str, str]:
    expected = EXPECTED[name]
    actual = _installed_provenance(expected["distribution"], expected["module"])
    if actual["commit"] != expected["commit"]:
        raise RuntimeError(
            f"{name} commit mismatch: expected {expected['commit']}, got {actual['commit']} "
            f"from {actual['module_path']}"
        )
    if actual["kind"] == "checkout":
        if _normalize_url(expected["url"]) not in _normalize_url(actual["remotes"]):
            raise RuntimeError(f"{name} checkout does not expose expected remote {expected['url']}")
    elif _normalize_url(actual["url"]) != _normalize_url(expected["url"]):
        raise RuntimeError(f"{name} source mismatch: expected {expected['url']}, got {actual['url']}")
    return actual


def _verify_lock(repo_root: Path, names: tuple[str, ...]) -> None:
    lock = (repo_root / "uv.lock").read_text()
    for name in names:
        expected = EXPECTED[name]
        if expected["url"] not in lock or expected["commit"] not in lock:
            raise RuntimeError(f"uv.lock does not contain the exact {name} source pin")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    names = ("sglang", "megatron") if args.profile == "flash" else ("automodel", "megatron")
    _verify_lock(repo_root, names)
    result = {
        "status": "pass",
        "profile": args.profile,
        "dependencies": {name: _verify(name) for name in names},
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
