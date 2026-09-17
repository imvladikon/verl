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
"""Check that a file taken from a fork head still fits the image it is overlaid onto.

An overlay drops one file from the fork head into a runtime whose sibling modules are older.
The file imports fine in the fork and fails in the job, which costs whatever the job queued for:
a tuning script imported a helper its own repo had gained and the image had not, and a trainer
resolved its Hydra config directory relative to ``__file__`` and stopped finding it once the file
moved. Both were visible before launch, from git alone, without the image.

    python scripts/glm53_overlay_check.py --file ../sglang/benchmark/x.py \\
        --repo ../sglang --base 3b8bf7ed6 --root python

Exit code is 1 when an import cannot resolve in the base revision, so it can gate a launcher.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys

# Deriving a path from the file's own location is what breaks when an overlay moves it. Naming a
# logger after __file__ does not, so require __file__ to meet something that walks the filesystem.
_PATH_FROM_FILE = ("dirname", "abspath", "realpath", "Path(", "parent", "join")


def read_at(repo: str, revision: str, path: str) -> str | None:
    """A file's contents at a revision, or None when the revision does not carry it."""
    result = subprocess.run(
        ["git", "-C", repo, "show", f"{revision}:{path}"], capture_output=True, text=True, timeout=60
    )
    return result.stdout if result.returncode == 0 else None


def module_paths(module: str, root: str) -> list[str]:
    """Where a dotted module could live in the repository, most specific first."""
    relative = module.replace(".", "/")
    prefix = f"{root.rstrip('/')}/" if root else ""
    return [f"{prefix}{relative}.py", f"{prefix}{relative}/__init__.py"]


def resolve_relative(module: str | None, level: int, package: str) -> str:
    """`from ..profiler import *` inside verl.utils.debug means verl.utils.profiler."""
    if not level:
        return module or ""
    base = package.split(".")
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join([*base, module] if module else base)


def star_sources(source: str, package: str) -> list[str]:
    """Modules this one re-exports wholesale, whose names it therefore also binds.

    verl's config and debug packages are built almost entirely out of `from .x import *`, and
    those are relative: resolving them as absolute names loses every re-exported symbol.
    """
    return [
        resolve_relative(node.module, node.level, package)
        for node in ast.parse(source).body
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names)
    ]


def _bound_names(body: list[ast.stmt]) -> set[str]:
    """Names this block binds, descending into the conditionals modules import under.

    Device- and version-dependent imports sit inside a top-level `if` or `try`, so a collector
    that only reads `tree.body` reports every one of them as missing.
    """
    names: set[str] = set()
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.If):
            names |= _bound_names(node.body) | _bound_names(node.orelse)
        elif isinstance(node, ast.Try):
            names |= _bound_names(node.body) | _bound_names(node.orelse) | _bound_names(node.finalbody)
            for handler in node.handlers:
                names |= _bound_names(handler.body)
    return names


def top_level_names(source: str) -> set[str]:
    """Names a module binds at import time, which is what `from module import name` needs."""
    return _bound_names(ast.parse(source).body)


def directory_exists(repo: str, revision: str, path: str) -> bool:
    """Whether a path is a directory at a revision, for packages that carry no __init__."""
    result = subprocess.run(
        ["git", "-C", repo, "ls-tree", "-d", "--name-only", revision, path.rstrip("/")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def names_exported_by(module: str, repo: str, base: str, root: str, depth: int = 3) -> set[str] | None:
    """Every name the module binds at the base revision, following `import *` re-exports.

    `sglang.srt.utils` is a package whose __init__ is one `from ... import *`; without following
    that, every name it re-exports looks missing.
    """
    # An empty __init__.py is an empty string, not a missing file: compare against None.
    candidates = ((path, read_at(repo, base, path)) for path in module_paths(module, root))
    matched = next(((path, text) for path, text in candidates if text is not None), None)
    if matched is None:
        return None
    path, source = matched
    # A package's own name is the base for its relative imports; a plain module's is its parent.
    package = module if path.endswith("/__init__.py") else module.rsplit(".", 1)[0]
    names = top_level_names(source)
    if depth > 0:
        for reexported in star_sources(source, package):
            names |= names_exported_by(reexported, repo, base, root, depth - 1) or set()
    return names


def unresolved_imports(source: str, repo: str, base: str, root: str, package: str) -> list[str]:
    """Imports the overlay makes that the base revision cannot satisfy."""
    problems = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
            continue
        if not node.module.startswith(package):
            continue  # third-party and stdlib come from the image's environment, not this tree
        available = names_exported_by(node.module, repo, base, root)
        if available is None:
            prefix = f"{root.rstrip('/')}/" if root else ""
            if directory_exists(repo, base, prefix + node.module.replace(".", "/")):
                continue  # a package with no __init__: its contents cannot be enumerated from here
            problems.append(f"{node.module} does not exist at {base}")
            continue
        for alias in node.names:
            if alias.name == "*" or alias.name in available:
                continue
            # `from package import submodule` resolves through the filesystem, not through names
            # bound in the package's __init__.
            submodule = module_paths(f"{node.module}.{alias.name}", root)
            if any(read_at(repo, base, path) is not None for path in submodule):
                continue
            problems.append(f"{node.module}.{alias.name} is missing at {base}")
    return problems


def relocation_risks(source: str) -> list[str]:
    """Lines whose meaning depends on where the file sits, which an overlay changes."""
    risks = []
    for number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # `other_module.__file__` belongs to that module and does not move with this overlay.
        own_file = re.search(r"(?<![\w.])__file__", stripped) is not None
        resolves_path = own_file and any(m in stripped for m in _PATH_FROM_FILE)
        # Only a config_path actually set to a literal relocates; a field annotated `= None` does not.
        if resolves_path or re.search(r"""config_path\s*=\s*['"]""", stripped):
            risks.append(f"line {number}: {stripped[:90]}")
    return risks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", required=True, help="the overlay file, as it will be shipped")
    parser.add_argument("--repo", required=True, help="fork checkout the image revision belongs to")
    parser.add_argument("--base", required=True, help="revision the image was built from")
    parser.add_argument("--root", default="", help="source root inside the repo (e.g. python for sglang)")
    parser.add_argument("--package", default="", help="only check imports from this package (default: repo name)")
    args = parser.parse_args()

    source = open(args.file).read()  # noqa: SIM115
    package = args.package or args.repo.rstrip("/").rsplit("/", 1)[-1].replace("-", ".").lower()

    problems = unresolved_imports(source, args.repo, args.base, args.root, package)
    risks = relocation_risks(source)

    for problem in problems:
        print(f"[FAIL] {problem}")
    for risk in risks:
        print(f"[warn] resolves a path from its own location, which the overlay moves: {risk}")
    if not problems and not risks:
        print(f"[ ok ] {args.file} fits {args.base}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
