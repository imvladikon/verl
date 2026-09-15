# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import runpy
from pathlib import Path

import setuptools
import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]


def _requirements_by_name(raw_requirements):
    requirements = (Requirement(raw) for raw in raw_requirements)
    return {canonicalize_name(requirement.name): requirement for requirement in requirements}


def _fallback_requirements(monkeypatch):
    metadata = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: metadata.update(kwargs))
    monkeypatch.chdir(REPO_ROOT)
    runpy.run_path(str(REPO_ROOT / "setup.py"), run_name="__main__")
    return _requirements_by_name(metadata["install_requires"])


def _core_requirements(extras):
    core = _requirements_by_name(extras["verl-core"])
    self_requirement = core.pop("verl")
    assert self_requirement.extras == {"transferqueue"}
    for extra in self_requirement.extras:
        core.update(_requirements_by_name(extras[extra]))
    return core


def test_fallback_metadata_matches_glm53_core_runtime(monkeypatch):
    with open(REPO_ROOT / "pyproject.toml", "rb") as pyproject_file:
        extras = tomllib.load(pyproject_file)["project"]["optional-dependencies"]

    fallback = _fallback_requirements(monkeypatch)
    core = _core_requirements(extras)
    assert fallback.keys() == core.keys()

    for name in core.keys() - {"transformers"}:
        assert fallback[name] == core[name]

    glm_transformers = _requirements_by_name(extras["glm53-flash"])["transformers"]
    glm_version = Version(next(iter(glm_transformers.specifier)).version)
    assert glm_version in fallback["transformers"].specifier
    assert Version("5.6.0") not in fallback["transformers"].specifier
    assert Version("5.16.2") not in fallback["transformers"].specifier
