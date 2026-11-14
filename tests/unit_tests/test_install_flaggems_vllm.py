# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).parents[2] / ".github/scripts/install_flaggems_vllm.py"
_SPEC = importlib.util.spec_from_file_location("install_flaggems_vllm", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
installer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(installer)


@pytest.mark.parametrize(
    ("args", "source"),
    [
        ([], f"git+{installer.REPOSITORY}@{installer.COMMIT}"),
        (
            ["--source-archive"],
            f"{installer.SOURCE_ARCHIVE}#sha256={installer.SOURCE_ARCHIVE_SHA256}",
        ),
        (
            ["--source-archive", "kunlunxin-grammar"],
            "https://codeload.github.com/flagos-ai/FlagGems-vllm/tar.gz/"
            "8631d9682afa57b616322821992d477aa538ebab"
            "#sha256=24c5a0609514761b6bb730e37839220e182822a20a3feef1faa209661f515cf9",
        ),
    ],
)
def test_installer_uses_the_selected_immutable_source(monkeypatch, args, source):
    calls = []
    monkeypatch.setattr(sys, "argv", ["installer", *args])
    monkeypatch.setattr(installer, "package_version", lambda _: "0.1.0")
    monkeypatch.setattr(installer, "release_is_installed", lambda *_: bool(calls))
    monkeypatch.setattr(installer, "pip_install", lambda *args: calls.append(args))
    installer.main()
    assert calls == [
        (
            "--no-deps",
            "--no-build-isolation",
            "--force-reinstall",
            f"flaggems-vllm @ {source}",
        )
    ]


def test_archive_requires_the_selected_hash_and_commit(monkeypatch):
    commit, sha256 = installer.ARCHIVE_PINS["kunlunxin-grammar"]
    url = installer.SOURCE_ARCHIVE.replace(installer.COMMIT, commit)
    source = {"url": url, "archive_info": {"hashes": {"sha256": sha256}}}
    monkeypatch.setattr(
        installer,
        "distribution",
        lambda _: SimpleNamespace(read_text=lambda _: json.dumps(source)),
    )
    assert installer.release_is_installed(commit, url, sha256)
    assert not installer.release_is_installed()
    source["archive_info"]["hashes"]["sha256"] = installer.SOURCE_ARCHIVE_SHA256
    assert not installer.release_is_installed(commit, url, sha256)
