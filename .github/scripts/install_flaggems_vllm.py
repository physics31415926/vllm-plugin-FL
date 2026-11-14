# Copyright (c) 2026 BAAI. All rights reserved.
"""Install the pinned FlagGems-vllm source in the CI interpreter."""

import argparse
import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, version

RELEASE = "v0.2.0"
COMMIT = "b5c2c2249319277ccda4d41aa0f65164bd9b2a19"
REPOSITORY = "https://github.com/flagos-ai/FlagGems-vllm.git"
SOURCE_ARCHIVE = f"https://codeload.github.com/flagos-ai/FlagGems-vllm/tar.gz/{COMMIT}"
SOURCE_ARCHIVE_SHA256 = (
    "1b47e1eb7982acc606e2fc920561d169e2a17fac7c92ff65c5f11cafa6b8b16e"
)
ARCHIVE_PINS = {
    "release": (COMMIT, SOURCE_ARCHIVE_SHA256),
    # v0.2.0 plus the Kunlunxin V2 structured-output grammar operator.
    "kunlunxin-grammar": (
        "8631d9682afa57b616322821992d477aa538ebab",
        "24c5a0609514761b6bb730e37839220e182822a20a3feef1faa209661f515cf9",
    ),
}
RUNTIME_PACKAGES = ("torch", "flagtree", "triton", "flag-gems", "vllm", "numpy")
PACKAGE_DEPENDENCIES = ("packaging", "pybind11", "PyYAML", "sqlalchemy")


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def release_is_installed(
    commit=COMMIT, archive=SOURCE_ARCHIVE, archive_sha256=SOURCE_ARCHIVE_SHA256
):
    try:
        text = distribution("flaggems-vllm").read_text("direct_url.json")
        source = json.loads(text or "{}")
    except (PackageNotFoundError, ValueError):
        return False
    if source.get("url") == archive:
        return source.get("archive_info", {}).get("hashes", {}).get("sha256") == (
            archive_sha256
        )
    return (
        source.get("url", "").removesuffix(".git") == REPOSITORY.removesuffix(".git")
        and source.get("vcs_info", {}).get("commit_id") == commit
    )


def pip_install(*args):
    env = dict(os.environ, PIP_BREAK_SYSTEM_PACKAGES="1")
    subprocess.run([sys.executable, "-m", "pip", "install", *args], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-archive",
        nargs="?",
        const="release",
        choices=ARCHIVE_PINS,
        help="install a named source archive (default: the pinned release)",
    )
    args = parser.parse_args()
    commit, archive_sha256 = ARCHIVE_PINS[args.source_archive or "release"]
    archive = f"https://codeload.github.com/flagos-ai/FlagGems-vllm/tar.gz/{commit}"
    provider = f"git+{REPOSITORY}@{commit}"
    if args.source_archive:
        provider = f"{archive}#sha256={archive_sha256}"

    before = {name: package_version(name) for name in RUNTIME_PACKAGES}
    missing = [name for name in PACKAGE_DEPENDENCIES if package_version(name) is None]
    if missing:
        pip_install(*missing)
    if not release_is_installed(commit, archive, archive_sha256):
        # Metadata still says 0.1.0; force replacement of another 0.1.0 source.
        # Reuse the build tools already used by this platform's plugin setup.
        pip_install(
            "--no-deps",
            "--no-build-isolation",
            "--force-reinstall",
            f"flaggems-vllm @ {provider}",
        )
    if not release_is_installed(commit, archive, archive_sha256):
        raise RuntimeError(f"FlagGems-vllm must come from {RELEASE} commit {commit}")
    after = {name: package_version(name) for name in RUNTIME_PACKAGES}
    if after != before:
        raise RuntimeError(f"CI runtime dependencies changed: {before} -> {after}")
    print(
        f"FlagGems-vllm: {RELEASE}, commit {commit}, "
        f"package metadata {package_version('flaggems-vllm')}"
    )


if __name__ == "__main__":
    main()
