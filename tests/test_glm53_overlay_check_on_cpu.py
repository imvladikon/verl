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
"""An overlay file has to fit the revision it lands on, and saying so must not cry wolf."""

import importlib.util
import pathlib
import sys

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "glm53_overlay_check.py"


def _load():
    spec = importlib.util.spec_from_file_location("glm53_overlay_check", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["glm53_overlay_check"] = module
    spec.loader.exec_module(module)
    return module


check = _load()


def _repo(monkeypatch, files):
    """Stand in for `git show base:path` over a fixed set of files."""
    monkeypatch.setattr(check, "read_at", lambda repo, revision, path: files.get(path))


def test_a_name_the_base_revision_lacks_is_reported(monkeypatch):
    _repo(monkeypatch, {"pkg/helpers.py": "def old_helper():\n    pass\n"})
    problems = check.unresolved_imports("from pkg.helpers import new_helper\n", "r", "base", "", "pkg")
    assert problems == ["pkg.helpers.new_helper is missing at base"]


def test_a_conditional_import_still_counts_as_bound(monkeypatch):
    # Device- and version-dependent imports live inside a top-level if/try.
    source = "import torch\nif torch.cuda.is_available():\n    from .a import timer\nelse:\n    from .b import timer\n"
    _repo(monkeypatch, {"pkg/profiler/__init__.py": source})
    assert check.unresolved_imports("from pkg.profiler import timer\n", "r", "base", "", "pkg") == []


def test_a_star_reexport_is_followed(monkeypatch):
    _repo(
        monkeypatch,
        {"pkg/utils/__init__.py": "from pkg.utils.common import *\n", "pkg/utils/common.py": "def is_hip():\n    pass\n"},
    )
    assert check.unresolved_imports("from pkg.utils import is_hip\n", "r", "base", "", "pkg") == []


def test_importing_a_submodule_is_not_a_missing_name(monkeypatch):
    # `from pkg import mod` resolves through the filesystem, not through names in __init__.
    _repo(monkeypatch, {"pkg/__init__.py": "", "pkg/mod.py": "x = 1\n"})
    assert check.unresolved_imports("from pkg import mod\n", "r", "base", "", "pkg") == []


def test_a_missing_module_is_reported(monkeypatch):
    _repo(monkeypatch, {})
    assert check.unresolved_imports("from pkg.gone import thing\n", "r", "base", "", "pkg") == [
        "pkg.gone does not exist at base"
    ]


def test_third_party_imports_are_left_alone(monkeypatch):
    _repo(monkeypatch, {})
    assert check.unresolved_imports("import torch\nfrom hydra import main\n", "r", "base", "", "pkg") == []


def test_only_paths_derived_from_the_files_location_are_flagged():
    flagged = check.relocation_risks('@hydra.main(config_path="config")\n')
    assert len(flagged) == 1 and "config_path" in flagged[0]

    assert check.relocation_risks("base = Path(__file__).parent\n"), "a path built from __file__ moves with the file"
    # Naming a logger after __file__ reads no directory, so it must not be reported.
    assert check.relocation_risks("logger = logging.getLogger(__file__)\n") == []
