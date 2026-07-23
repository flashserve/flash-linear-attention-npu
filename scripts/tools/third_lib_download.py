#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare and verify the third-party cache used by offline package builds."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request


MANIFEST_NAME = ".fla_npu_offline_manifest.json"
REVISION_NAME = ".fla_npu_revision"
SCHEMA_VERSION = 1

ARCHIVES = (
    {
        "name": "json",
        "filename": "include.zip",
        "url": "https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip",
        "sha256": "a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d",
    },
    {
        "name": "makeself",
        "filename": "makeself-release-2.5.0-patch1.tar.gz",
        "url": (
            "https://gitcode.com/cann-src-third-party/makeself/releases/download/"
            "release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz"
        ),
        "sha256": "bfa730a5763cdb267904a130e02b2e48e464986909c0733ff1c96495f620369a",
    },
    {
        "name": "eigen",
        "filename": "eigen-5.0.0.tar.gz",
        "url": (
            "https://gitcode.com/cann-src-third-party/eigen/releases/download/"
            "5.0.0-h0.trunk/eigen-5.0.0.tar.gz"
        ),
        "sha256": "93f7f0462988b934e632a9fba58af55192ffceae38e8f46233f4f62cb1e79370",
    },
    {
        "name": "protobuf",
        "filename": "protobuf-25.1.tar.gz",
        "url": (
            "https://gitcode.com/cann-src-third-party/protobuf/releases/download/"
            "v25.1/protobuf-25.1.tar.gz"
        ),
        "sha256": "9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42",
    },
    {
        "name": "abseil-cpp",
        "filename": "abseil-cpp-20230802.1.tar.gz",
        "url": (
            "https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/"
            "20230802.1/abseil-cpp-20230802.1.tar.gz"
        ),
        "sha256": "987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed",
    },
)

REPOSITORIES = (
    {
        "name": "opbase",
        "url": "https://gitcode.com/cann/opbase.git",
        "revision": "c8d83f3e57a63a7375e89a2d6937452c0ae2e522",
        "required_files": ("CMakeLists.txt",),
    },
    {
        "name": "catlass",
        "url": "https://gitcode.com/cann/catlass.git",
        "revision": "41bf90da655bba3c66d0acd7e00abe33960ecfd6",
        "required_files": ("include/catlass/catlass.hpp",),
    },
)


def _default_output_dir():
    return Path(__file__).resolve().parents[2] / "third_party"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unlink_if_exists(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _download(url, destination, expected_sha256, retries, timeout):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    _unlink_if_exists(temporary)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "flash-linear-attention-npu-offline-deps/1"},
    )

    for attempt in range(1, retries + 1):
        try:
            print("[DOWNLOAD] {}".format(url), flush=True)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            actual_sha256 = _sha256(temporary)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    "checksum mismatch for {}: expected {}, got {}".format(
                        destination.name, expected_sha256, actual_sha256
                    )
                )
            os.replace(str(temporary), str(destination))
            return
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            _unlink_if_exists(temporary)
            if attempt == retries:
                raise RuntimeError("failed to download {}: {}".format(url, error))
            print(
                "[RETRY] {}/{} failed: {}".format(attempt, retries, error),
                file=sys.stderr,
                flush=True,
            )
            time.sleep(min(attempt * 2, 5))


def _run_git(arguments, cwd=None, timeout=None):
    command = ["git"] + list(arguments)
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=environment,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise RuntimeError("git is required to prepare offline dependencies")
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip()
        raise RuntimeError("{} failed: {}".format(" ".join(command), detail))
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "{} timed out after {} seconds".format(" ".join(command), timeout)
        )
    return result.stdout.strip()


def _replace_directory(source, destination):
    if destination.exists():
        shutil.rmtree(str(destination))
    os.replace(str(source), str(destination))


def _verify_repository(path, dependency):
    name = str(dependency["name"])
    revision = str(dependency["revision"])
    if not path.is_dir():
        return "missing repository directory: {}/".format(name)
    marker = path / REVISION_NAME
    if not marker.is_file():
        return "missing revision marker: {}/{}".format(name, REVISION_NAME)
    actual_revision = marker.read_text(encoding="utf-8").strip()
    if actual_revision != revision:
        return "{} revision mismatch: expected {}, got {}".format(
            name, revision, actual_revision or "<empty>"
        )
    for required_file in dependency["required_files"]:
        if not (path / str(required_file)).is_file():
            return "missing required file: {}/{}".format(name, required_file)
    return None


def _prepare_repository(cache_dir, dependency, retries, git_timeout):
    name = str(dependency["name"])
    url = str(dependency["url"])
    revision = str(dependency["revision"])
    destination = cache_dir / name

    if _verify_repository(destination, dependency) is None:
        print("[FOUND] {} at revision {}".format(name, revision), flush=True)
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".{}-".format(name), dir=str(cache_dir)))
    temporary_repo = temporary_root / name
    try:
        print("[CLONE] {} at {}".format(name, revision), flush=True)
        _run_git(["init", str(temporary_repo)], timeout=git_timeout)
        _run_git(
            ["remote", "add", "origin", url],
            cwd=temporary_repo,
            timeout=git_timeout,
        )
        for attempt in range(1, retries + 1):
            try:
                _run_git(
                    ["fetch", "--depth", "1", "origin", revision],
                    cwd=temporary_repo,
                    timeout=git_timeout,
                )
                break
            except RuntimeError as error:
                if attempt == retries:
                    raise
                print(
                    "[RETRY] {}/{} failed: {}".format(attempt, retries, error),
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(min(attempt * 2, 5))
        _run_git(
            ["checkout", "--detach", "FETCH_HEAD"],
            cwd=temporary_repo,
            timeout=git_timeout,
        )
        actual_revision = _run_git(
            ["rev-parse", "HEAD"],
            cwd=temporary_repo,
            timeout=git_timeout,
        )
        if actual_revision != revision:
            raise RuntimeError(
                "{} revision mismatch: expected {}, got {}".format(
                    name, revision, actual_revision
                )
            )
        shutil.rmtree(str(temporary_repo / ".git"))
        (temporary_repo / REVISION_NAME).write_text(revision + "\n", encoding="utf-8")
        repository_error = _verify_repository(temporary_repo, dependency)
        if repository_error is not None:
            raise RuntimeError(repository_error)
        _replace_directory(temporary_repo, destination)
    finally:
        shutil.rmtree(str(temporary_root), ignore_errors=True)


def _manifest():
    repositories = []
    for dependency in REPOSITORIES:
        repositories.append(
            {
                "name": dependency["name"],
                "url": dependency["url"],
                "revision": dependency["revision"],
                "required_files": list(dependency["required_files"]),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "archives": list(ARCHIVES),
        "repositories": repositories,
    }


def _write_manifest(cache_dir):
    manifest_path = cache_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_cache(cache_dir, retries, timeout, git_timeout):
    package_dir = cache_dir / "pkg"
    package_dir.mkdir(parents=True, exist_ok=True)
    for dependency in ARCHIVES:
        destination = package_dir / dependency["filename"]
        if destination.is_file() and _sha256(destination) == dependency["sha256"]:
            print("[FOUND] {}: {}".format(dependency["name"], destination), flush=True)
            continue
        if destination.exists():
            destination.unlink()
        _download(
            dependency["url"],
            destination,
            dependency["sha256"],
            retries,
            timeout,
        )
    for dependency in REPOSITORIES:
        _prepare_repository(cache_dir, dependency, retries, git_timeout)
    _write_manifest(cache_dir)


def verify_cache(cache_dir):
    errors = []
    for dependency in ARCHIVES:
        archive = cache_dir / "pkg" / dependency["filename"]
        if not archive.is_file():
            errors.append("missing archive: pkg/{}".format(dependency["filename"]))
            continue
        actual_sha256 = _sha256(archive)
        if actual_sha256 != dependency["sha256"]:
            errors.append(
                "checksum mismatch for pkg/{}: expected {}, got {}".format(
                    dependency["filename"], dependency["sha256"], actual_sha256
                )
            )
    for dependency in REPOSITORIES:
        repository_error = _verify_repository(cache_dir / dependency["name"], dependency)
        if repository_error is not None:
            errors.append(repository_error)
    return errors


def create_bundle(cache_dir, bundle_path):
    errors = verify_cache(cache_dir)
    if errors:
        raise RuntimeError(
            "offline dependency cache is incomplete:\n  - " + "\n  - ".join(errors)
        )
    _write_manifest(cache_dir)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle_path.with_name(bundle_path.name + ".part")
    _unlink_if_exists(temporary)
    with tarfile.open(str(temporary), "w:gz", format=tarfile.PAX_FORMAT) as archive:
        archive.add(str(cache_dir / MANIFEST_NAME), arcname=MANIFEST_NAME)
        for dependency in ARCHIVES:
            relative_path = Path("pkg") / dependency["filename"]
            archive.add(str(cache_dir / relative_path), arcname=str(relative_path))
        for dependency in REPOSITORIES:
            archive.add(
                str(cache_dir / dependency["name"]),
                arcname=str(dependency["name"]),
            )
    os.replace(str(temporary), str(bundle_path))
    print("[BUNDLE] {}".format(bundle_path), flush=True)


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Download, verify, and bundle dependencies required by an offline "
            "Way A / --pkg build."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Third-party cache directory (default: <repo>/third_party).",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        help="After preparing the cache, create a transferable .tar.gz bundle.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not access the network; only verify that the cache is complete.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Download retry count (default: 3).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--git-timeout",
        type=int,
        default=300,
        help="Timeout for each git command in seconds (default: 300).",
    )
    arguments = parser.parse_args()
    if arguments.retries < 1:
        parser.error("--retries must be at least 1")
    if arguments.timeout < 1:
        parser.error("--timeout must be at least 1")
    if arguments.git_timeout < 1:
        parser.error("--git-timeout must be at least 1")
    if arguments.verify_only and arguments.bundle is not None:
        parser.error("--verify-only cannot be combined with --bundle")
    return arguments


def main():
    arguments = _parse_arguments()
    cache_dir = arguments.output_dir.expanduser().resolve()
    try:
        if not arguments.verify_only:
            prepare_cache(
                cache_dir,
                arguments.retries,
                arguments.timeout,
                arguments.git_timeout,
            )
        errors = verify_cache(cache_dir)
        if errors:
            print(
                "[ERROR] offline dependency cache is incomplete:",
                file=sys.stderr,
            )
            for error in errors:
                print("  - {}".format(error), file=sys.stderr)
            return 1
        print("[OK] offline dependency cache verified: {}".format(cache_dir))
        if arguments.bundle is not None:
            create_bundle(cache_dir, arguments.bundle.expanduser().resolve())
    except (OSError, RuntimeError, tarfile.TarError) as error:
        print("[ERROR] {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
