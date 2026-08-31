#!/usr/bin/env python3
"""Fail closed unless active GLM dependencies match their locked revisions."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import tomllib

EXPECTED = {
    "sglang": {
        "distribution": "sglang",
        "module": "sglang",
        "url": "https://github.com/imvladikon/sglang.git",
        "ref": "glm-5.3-flash",
    },
    "megatron": {
        "distribution": "megatron-core",
        "module": "megatron.core.package_info",
        "url": "https://github.com/imvladikon/Megatron-LM.git",
        "ref": "glm-5.3-flash",
    },
    "automodel": {
        "distribution": "nemo-automodel",
        "module": "nemo_automodel",
        "url": "https://github.com/NVIDIA-NeMo/Automodel.git",
        "ref": "9228f33cf73d66a9b2e84256d298aac9a70283f0",
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


def _verify(name: str, locked_commit: str) -> dict[str, str]:
    expected = EXPECTED[name]
    actual = _installed_provenance(expected["distribution"], expected["module"])
    if actual["commit"] != locked_commit:
        raise RuntimeError(
            f"{name} commit mismatch: expected locked {locked_commit}, got {actual['commit']} "
            f"from {actual['module_path']}"
        )
    if actual["kind"] == "checkout":
        if _normalize_url(expected["url"]) not in _normalize_url(actual["remotes"]):
            raise RuntimeError(f"{name} checkout does not expose expected remote {expected['url']}")
    elif _normalize_url(actual["url"]) != _normalize_url(expected["url"]):
        raise RuntimeError(f"{name} source mismatch: expected {expected['url']}, got {actual['url']}")
    return actual


def _locked_commits(repo_root: Path, names: tuple[str, ...]) -> dict[str, str]:
    lock = tomllib.loads((repo_root / "uv.lock").read_text())
    commits = {}
    for name in names:
        expected = EXPECTED[name]
        matches = []
        for package in lock["package"]:
            if package["name"] != expected["distribution"]:
                continue
            git_source = package.get("source", {}).get("git")
            if not git_source:
                continue
            parsed = urlsplit(git_source)
            if _normalize_url(f"{parsed.scheme}://{parsed.netloc}{parsed.path}") != _normalize_url(expected["url"]):
                continue
            if parse_qs(parsed.query).get("rev") != [expected["ref"]]:
                continue
            if not parsed.fragment:
                raise RuntimeError(f"uv.lock has no resolved commit for {name}")
            matches.append(parsed.fragment)
        if len(matches) != 1:
            raise RuntimeError(f"uv.lock has {len(matches)} matching sources for {name}, expected one")
        commits[name] = matches[0]
    return commits


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    names = ("sglang", "megatron") if args.profile == "flash" else ("automodel", "megatron")
    locked_commits = _locked_commits(repo_root, names)
    result = {
        "status": "pass",
        "profile": args.profile,
        "dependencies": {name: _verify(name, locked_commits[name]) for name in names},
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
